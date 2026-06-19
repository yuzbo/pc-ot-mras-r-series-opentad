from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional

import torch
import torch.nn.functional as F


_DEFAULT_WEIGHTS = {
    "body": 0.02,
    "start": 0.05,
    "end": 0.05,
    "boundary": 0.05,
    "uncertainty": 0.01,
    "redundancy": 0.005,
    "process": 0.01,
    "pair": 0.05,
    "allocation": 0.02,
    "regularizer": 0.01,
}


def _require_tensor(mapping: Mapping[str, object], key: str) -> torch.Tensor:
    value = mapping.get(key)
    if not torch.is_tensor(value):
        raise ValueError(f"reader_outputs['{key}'] must be a tensor")
    if torch.is_complex(value):
        raise ValueError(f"reader_outputs['{key}'] must be real-valued")
    if torch.is_floating_point(value) and not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"reader_outputs['{key}'] must be finite")
    return value


def _valid_mask(reader_outputs: Mapping[str, object]) -> torch.Tensor:
    mask = _require_tensor(reader_outputs, "valid_mask")
    if mask.ndim != 2:
        raise ValueError(f"valid_mask must be [B,T], got {tuple(mask.shape)}")
    if mask.dtype != torch.bool and not bool(((mask == 0) | (mask == 1)).all().item()):
        raise ValueError("valid_mask must contain binary 0/1 values")
    return mask.bool()


def _zero_like_loss(reader_outputs: Mapping[str, object]) -> torch.Tensor:
    for key in ("start_logits", "end_logits", "boundary_logits", "pair_logits"):
        value = reader_outputs.get(key)
        if torch.is_tensor(value):
            return value.sum() * 0.0
    raise ValueError("reader_outputs must contain at least one trainable logit tensor")


def _as_segments(value: object, *, device: torch.device) -> torch.Tensor:
    if value is None:
        return torch.empty((0, 2), device=device, dtype=torch.float32)
    if not torch.is_tensor(value):
        value = torch.as_tensor(value, device=device, dtype=torch.float32)
    else:
        value = value.detach().to(device=device, dtype=torch.float32)
    if value.numel() == 0:
        return value.reshape(0, 2)
    if value.ndim != 2 or value.shape[-1] != 2:
        raise ValueError(f"gt_segments entries must be [N,2], got {tuple(value.shape)}")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError("gt_segments must be finite")
    return value


def _segment_targets(
    gt_segments: Iterable[object],
    valid: torch.Tensor,
    *,
    boundary_sigma: float,
    short_action_len: float,
    adjacent_gap: float,
) -> Dict[str, torch.Tensor]:
    if float(boundary_sigma) <= 0.0:
        raise ValueError("boundary_sigma must be positive")
    batch, time = valid.shape
    device = valid.device
    dtype = torch.float32
    grid = torch.arange(time, device=device, dtype=dtype)
    action = torch.zeros((batch, time), device=device, dtype=dtype)
    start = torch.zeros_like(action)
    end = torch.zeros_like(action)
    short_or_adjacent = torch.zeros_like(action)
    process = torch.zeros((batch, time), device=device, dtype=torch.long)
    pair = torch.zeros((batch, time, time), device=device, dtype=dtype)
    valid_lengths = valid.long().sum(dim=1)

    segments_list = list(gt_segments)
    if len(segments_list) != batch:
        raise ValueError(f"gt_segments length {len(segments_list)} must match batch size {batch}")

    for batch_idx, raw_segments in enumerate(segments_list):
        valid_len = int(valid_lengths[batch_idx].item())
        if valid_len <= 0:
            continue
        segments = _as_segments(raw_segments, device=device)
        if segments.numel() == 0:
            continue
        seg_start = segments[:, 0].clamp(0.0, float(valid_len))
        seg_end = segments[:, 1].clamp(0.0, float(valid_len))
        keep = seg_end > seg_start
        if not bool(keep.any().item()):
            continue
        seg_start = seg_start[keep]
        seg_end = seg_end[keep]
        duration = (seg_end - seg_start).clamp_min(1.0e-6)
        valid_grid = grid[:valid_len]

        inside = (valid_grid[:, None] >= seg_start[None, :]) & (valid_grid[:, None] <= seg_end[None, :])
        action[batch_idx, :valid_len] = inside.any(dim=1).to(dtype)

        start_dist = (valid_grid[:, None] - seg_start[None, :]).abs()
        end_dist = (valid_grid[:, None] - seg_end[None, :]).abs()
        start_peak = torch.exp(-0.5 * (start_dist / float(boundary_sigma)).square())
        end_peak = torch.exp(-0.5 * (end_dist / float(boundary_sigma)).square())
        start[batch_idx, :valid_len] = start_peak.max(dim=1).values
        end[batch_idx, :valid_len] = end_peak.max(dim=1).values

        short_mask = duration <= float(short_action_len)
        if bool(short_mask.any().item()):
            short_inside = inside[:, short_mask].any(dim=1)
            short_or_adjacent[batch_idx, :valid_len] = torch.maximum(
                short_or_adjacent[batch_idx, :valid_len],
                short_inside.to(dtype),
            )
        if seg_start.numel() > 1 and float(adjacent_gap) > 0.0:
            order = torch.argsort(seg_start)
            sorted_start = seg_start[order]
            sorted_end = seg_end[order]
            gaps = sorted_start[1:] - sorted_end[:-1]
            close = gaps <= float(adjacent_gap)
            for close_idx in torch.nonzero(close, as_tuple=False).flatten().tolist():
                left_end = sorted_end[close_idx]
                right_start = sorted_start[close_idx + 1]
                gap_region = (valid_grid >= left_end) & (valid_grid <= right_start)
                short_or_adjacent[batch_idx, :valid_len] = torch.maximum(
                    short_or_adjacent[batch_idx, :valid_len],
                    gap_region.to(dtype),
                )

        start_soft = torch.exp(-0.5 * (start_dist / float(boundary_sigma)).square())
        end_soft = torch.exp(-0.5 * (end_dist / float(boundary_sigma)).square())
        pair_weights = start_soft[:, None, :] * end_soft[None, :, :]
        duration_mask = valid_grid[None, :] > valid_grid[:, None]
        pair[batch_idx, :valid_len, :valid_len] = pair_weights.max(dim=2).values * duration_mask.to(dtype)

    boundary = torch.maximum(start, end)
    uncertainty = torch.maximum(boundary, short_or_adjacent)
    redundancy = (1.0 - torch.maximum(action, uncertainty)).clamp(min=0.0, max=1.0)
    budget = (boundary + 0.5 * uncertainty + 0.25 * action).clamp(max=1.0)

    process[action > 0.5] = 3
    process[start > 0.30] = 2
    process[end > 0.30] = 4
    process[uncertainty > 0.75] = 6
    process = process.masked_fill(~valid, 0)

    return {
        "action": action.masked_fill(~valid, 0.0),
        "start": start.masked_fill(~valid, 0.0),
        "end": end.masked_fill(~valid, 0.0),
        "boundary": boundary.masked_fill(~valid, 0.0),
        "uncertainty": uncertainty.masked_fill(~valid, 0.0),
        "redundancy": redundancy.masked_fill(~valid, 0.0),
        "budget": budget.masked_fill(~valid, 0.0),
        "process": process,
        "pair": pair,
    }


def _masked_soft_bce(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if logits.shape != target.shape or logits.shape != valid.shape:
        raise ValueError("logits, target, and valid mask must share shape [B,T]")
    if torch.is_complex(logits) or not bool(torch.isfinite(logits).all().item()):
        raise ValueError("logits must be finite and real-valued")
    valid_f = valid.to(dtype=logits.dtype)
    loss = F.binary_cross_entropy_with_logits(logits, target.to(dtype=logits.dtype), reduction="none")
    return (loss * valid_f).sum() / valid_f.sum().clamp_min(1.0)


def _pair_cross_entropy(
    pair_logits: torch.Tensor,
    pair_target: torch.Tensor,
    pair_valid: torch.Tensor,
) -> torch.Tensor:
    if pair_logits.shape != pair_target.shape or pair_logits.shape != pair_valid.shape:
        raise ValueError("pair logits, target, and valid mask must share shape [B,T,T]")
    if torch.is_complex(pair_logits) or not bool(torch.isfinite(pair_logits).all().item()):
        raise ValueError("pair_logits must be finite and real-valued")
    flat_target = pair_target.masked_fill(~pair_valid, 0.0).flatten(1)
    target_mass = flat_target.sum(dim=1)
    has_target = target_mass > 0.0
    if not bool(has_target.any().item()):
        return pair_logits.sum() * 0.0
    flat_logits = pair_logits.masked_fill(~pair_valid, torch.finfo(pair_logits.dtype).min / 4.0).flatten(1)
    log_prob = F.log_softmax(flat_logits[has_target], dim=-1)
    target = flat_target[has_target].to(dtype=pair_logits.dtype)
    target = target / target.sum(dim=1, keepdim=True).clamp_min(torch.finfo(pair_logits.dtype).eps)
    return -(target * log_prob).sum(dim=1).mean()


def pc_ot_mras_auxiliary_losses(
    reader_outputs: Mapping[str, object],
    gt_segments: Iterable[object],
    *,
    weights: Optional[Mapping[str, float]] = None,
    boundary_sigma: float = 2.0,
    short_action_len: float = 12.0,
    adjacent_gap: float = 8.0,
) -> Dict[str, torch.Tensor]:
    """Train-only supervision for PC-OT-MRAS dense fields and pair distribution."""

    valid = _valid_mask(reader_outputs)
    merged_weights = dict(_DEFAULT_WEIGHTS)
    merged_weights.update(dict(weights or {}))
    targets = _segment_targets(
        gt_segments,
        valid,
        boundary_sigma=float(boundary_sigma),
        short_action_len=float(short_action_len),
        adjacent_gap=float(adjacent_gap),
    )
    losses: Dict[str, torch.Tensor] = {}

    dense_pairs = (
        ("body", "body_logits", "action"),
        ("start", "start_logits", "start"),
        ("end", "end_logits", "end"),
        ("boundary", "boundary_logits", "boundary"),
        ("uncertainty", "uncertainty_logits", "uncertainty"),
        ("redundancy", "redundancy_logits", "redundancy"),
    )
    for name, logit_key, target_key in dense_pairs:
        weight = float(merged_weights.get(name, 0.0))
        if weight == 0.0:
            continue
        logits = _require_tensor(reader_outputs, logit_key)
        losses[f"pc_ot_mras_aux_{name}_loss"] = _masked_soft_bce(logits, targets[target_key], valid) * weight

    process_weight = float(merged_weights.get("process", 0.0))
    if process_weight != 0.0:
        process_logits = _require_tensor(reader_outputs, "process_logits")
        if process_logits.ndim != 3 or process_logits.shape[:2] != valid.shape:
            raise ValueError("process_logits must be [B,T,S] and match valid_mask")
        process_target = targets["process"].clamp(max=int(process_logits.shape[-1]) - 1)
        losses["pc_ot_mras_aux_process_loss"] = (
            F.cross_entropy(process_logits[valid], process_target[valid], reduction="mean") * process_weight
        )

    pair_weight = float(merged_weights.get("pair", 0.0))
    if pair_weight != 0.0:
        pair_logits = _require_tensor(reader_outputs, "pair_logits")
        pair_valid = _require_tensor(reader_outputs, "pair_valid_mask").bool()
        losses["pc_ot_mras_aux_pair_loss"] = _pair_cross_entropy(pair_logits, targets["pair"], pair_valid) * pair_weight

    allocation_weight = float(merged_weights.get("allocation", 0.0))
    if allocation_weight != 0.0:
        acquisition = _require_tensor(reader_outputs, "acquisition_matrix")
        if acquisition.ndim != 3 or acquisition.shape[0] != valid.shape[0] or acquisition.shape[2] != valid.shape[1]:
            raise ValueError("acquisition_matrix must be [B,K,T] and match valid_mask")
        column_importance = 1.0 - torch.exp(-acquisition.sum(dim=1).to(dtype=torch.float32))
        losses["pc_ot_mras_aux_allocation_loss"] = (
            ((column_importance - targets["budget"]).square() * valid.to(dtype=torch.float32)).sum()
            / valid.to(dtype=torch.float32).sum().clamp_min(1.0)
            * allocation_weight
        )

    regularizer_weight = float(merged_weights.get("regularizer", 0.0))
    if regularizer_weight != 0.0:
        regularizers = reader_outputs.get("regularizers")
        if not isinstance(regularizers, Mapping) or not torch.is_tensor(regularizers.get("total_regularizer")):
            raise ValueError("reader_outputs['regularizers']['total_regularizer'] is required")
        losses["pc_ot_mras_aux_regularizer_loss"] = regularizers["total_regularizer"] * regularizer_weight

    if not losses:
        losses["pc_ot_mras_aux_zero_loss"] = _zero_like_loss(reader_outputs)
    return losses


__all__ = ["pc_ot_mras_auxiliary_losses"]
