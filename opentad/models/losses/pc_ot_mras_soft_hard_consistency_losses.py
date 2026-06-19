from __future__ import annotations

from collections.abc import Mapping
from typing import Dict, Optional

import torch
import torch.nn.functional as F


_DEFAULT_WEIGHTS = {
    "slot_allocation": 0.02,
    "global_acquisition": 0.02,
    "selected_time": 0.01,
    "gate_confidence": 0.005,
    "duplicate_mass": 0.005,
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
    valid = _require_tensor(reader_outputs, "valid_mask")
    if valid.ndim != 2:
        raise ValueError(f"valid_mask must be [B,T], got {tuple(valid.shape)}")
    if valid.dtype != torch.bool and not bool(((valid == 0) | (valid == 1)).all().item()):
        raise ValueError("valid_mask must contain binary 0/1 values")
    valid = valid.bool()
    counts = valid.long().sum(dim=1)
    if bool((counts <= 0).any().item()):
        raise ValueError("valid_mask must contain at least one valid position per sample")
    prefix = torch.arange(valid.shape[1], device=valid.device)[None, :] < counts[:, None]
    if not torch.equal(valid, prefix):
        raise ValueError("valid_mask must be a contiguous valid prefix")
    return valid


def _default_time_coords(valid: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    batch, time = valid.shape
    valid_len = valid.long().sum(dim=1).clamp(min=1).to(dtype=dtype)
    pos = torch.arange(time, device=valid.device, dtype=dtype)[None, :].expand(batch, -1)
    denom = (valid_len - 1.0).clamp(min=1.0)
    return (pos / denom[:, None]).masked_fill(~valid, 0.0)


def _time_coords(reader_outputs: Mapping[str, object], valid: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    raw = reader_outputs.get("time_coords")
    if raw is None:
        return _default_time_coords(valid, dtype)
    if not torch.is_tensor(raw):
        raise ValueError("reader_outputs['time_coords'] must be a tensor when provided")
    if raw.shape != valid.shape:
        raise ValueError("time_coords must be [B,T] and match valid_mask")
    if raw.device != valid.device:
        raise ValueError("time_coords must share device with valid_mask")
    if torch.is_complex(raw) or not bool(torch.isfinite(raw).all().item()):
        raise ValueError("time_coords must be finite and real-valued")
    coords = raw.to(dtype=dtype).masked_fill(~valid, 0.0)
    adjacent_valid = valid[:, 1:] & valid[:, :-1]
    if bool(adjacent_valid.any().item()):
        delta = coords[:, 1:] - coords[:, :-1]
        if not bool((delta[adjacent_valid] > 0).all().item()):
            raise ValueError("time_coords valid prefix must be strictly increasing")
    return coords


def _zero_like_loss(reader_outputs: Mapping[str, object]) -> torch.Tensor:
    for key in ("allocation_logits", "allocation", "acquisition_matrix", "gate_logits"):
        value = reader_outputs.get(key)
        if torch.is_tensor(value):
            return value.sum() * 0.0
    raise ValueError("reader_outputs must contain at least one trainable PC-OT-MRAS tensor")


def _reader_shapes(
    reader_outputs: Mapping[str, object],
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    allocation = _require_tensor(reader_outputs, "allocation")
    acquisition = _require_tensor(reader_outputs, "acquisition_matrix")
    if allocation.ndim != 3 or allocation.shape[0] != valid.shape[0] or allocation.shape[2] != valid.shape[1]:
        raise ValueError("allocation must be [B,K,T] and match valid_mask")
    if acquisition.shape != allocation.shape:
        raise ValueError("acquisition_matrix must share shape with allocation")
    if allocation.device != valid.device or acquisition.device != valid.device:
        raise ValueError("reader allocation tensors must share device with valid_mask")
    if bool((allocation < 0).any().item()) or bool((acquisition < 0).any().item()):
        raise ValueError("allocation and acquisition_matrix must be non-negative")
    invalid_mass = allocation.masked_select(~valid[:, None, :].expand_as(allocation))
    if invalid_mass.numel() and bool((invalid_mass.abs() > 1.0e-6).any().item()):
        raise ValueError("allocation must be zero on invalid positions")
    return allocation, acquisition


def _hard_slot_targets(allocation: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch, slots, time = allocation.shape
    masked = allocation.detach().masked_fill(~valid[:, None, :], -1.0)
    hard_index = masked.argmax(dim=-1)
    hard_onehot = F.one_hot(hard_index, num_classes=time).to(dtype=allocation.dtype, device=allocation.device)
    hard_onehot = hard_onehot.masked_fill(~valid[:, None, :].expand(batch, slots, time), 0.0)
    return hard_index, hard_onehot


def pc_ot_mras_soft_hard_consistency_losses(
    reader_outputs: Mapping[str, object],
    *,
    weights: Optional[Mapping[str, float]] = None,
    eps: float = 1.0e-6,
) -> Dict[str, torch.Tensor]:
    """Train-only R19 consistency between soft PC-OT transport and hard export anchors.

    Hard anchors are derived from detached reader allocations. They are used as
    consistency targets only; this loss never imports deploy-time hard-export
    JSON, raw predictions, GT, teacher outputs, checkpoints, or cached results.
    """

    if not isinstance(reader_outputs, Mapping):
        raise ValueError("reader_outputs must be a mapping")
    if float(eps) <= 0.0:
        raise ValueError("eps must be positive")

    valid = _valid_mask(reader_outputs)
    allocation, acquisition = _reader_shapes(reader_outputs, valid)
    coords = _time_coords(reader_outputs, valid, allocation.dtype)
    hard_index, hard_onehot = _hard_slot_targets(allocation, valid)

    merged_weights = dict(_DEFAULT_WEIGHTS)
    merged_weights.update(dict(weights or {}))
    losses: Dict[str, torch.Tensor] = {}
    eps_value = max(float(eps), float(torch.finfo(allocation.dtype).eps))

    if float(merged_weights.get("slot_allocation", 0.0)) != 0.0:
        slot_loss = -(hard_onehot * allocation.clamp_min(eps_value).log()).sum(dim=-1).mean()
        losses["pc_ot_mras_soft_hard_slot_allocation_loss"] = (
            slot_loss * float(merged_weights["slot_allocation"])
        )

    if float(merged_weights.get("global_acquisition", 0.0)) != 0.0:
        hard_global = hard_onehot.max(dim=1).values.masked_fill(~valid, 0.0)
        hard_dist = hard_global / hard_global.sum(dim=-1, keepdim=True).clamp_min(1.0)
        col_mass = acquisition.sum(dim=1).masked_fill(~valid, 0.0)
        col_dist = col_mass / col_mass.sum(dim=-1, keepdim=True).clamp_min(eps_value)
        global_loss = -(hard_dist.to(dtype=col_dist.dtype) * col_dist.clamp_min(eps_value).log()).sum(dim=-1).mean()
        losses["pc_ot_mras_soft_hard_global_acquisition_loss"] = (
            global_loss * float(merged_weights["global_acquisition"])
        )

    if float(merged_weights.get("selected_time", 0.0)) != 0.0:
        selected_times = reader_outputs.get("selected_times")
        if selected_times is None:
            selected_times = torch.einsum("bkt,bt->bk", allocation, coords)
        elif not torch.is_tensor(selected_times):
            raise ValueError("reader_outputs['selected_times'] must be a tensor when provided")
        if selected_times.shape != allocation.shape[:2]:
            raise ValueError("selected_times must be [B,K] and match allocation slots")
        if selected_times.device != allocation.device:
            raise ValueError("selected_times must share device with allocation")
        if torch.is_complex(selected_times) or not bool(torch.isfinite(selected_times).all().item()):
            raise ValueError("selected_times must be finite and real-valued")
        hard_times = coords.detach().gather(1, hard_index)
        time_loss = F.smooth_l1_loss(selected_times, hard_times.to(dtype=selected_times.dtype), reduction="mean")
        losses["pc_ot_mras_soft_hard_selected_time_loss"] = (
            time_loss * float(merged_weights["selected_time"])
        )

    if float(merged_weights.get("gate_confidence", 0.0)) != 0.0:
        gate_logits = _require_tensor(reader_outputs, "gate_logits")
        if gate_logits.shape != allocation.shape[:2]:
            raise ValueError("gate_logits must be [B,K] and match allocation slots")
        hard_conf = allocation.detach().amax(dim=-1).clamp(0.0, 1.0)
        gate_loss = F.binary_cross_entropy_with_logits(
            gate_logits,
            hard_conf.to(device=gate_logits.device, dtype=gate_logits.dtype),
            reduction="mean",
        )
        losses["pc_ot_mras_soft_hard_gate_confidence_loss"] = (
            gate_loss * float(merged_weights["gate_confidence"])
        )

    if float(merged_weights.get("duplicate_mass", 0.0)) != 0.0:
        hard_count = hard_onehot.sum(dim=1)
        duplicate_target = (hard_count > 1.0).to(dtype=allocation.dtype)
        duplicate_mass = (allocation * hard_onehot).sum(dim=1) - hard_onehot.max(dim=1).values
        duplicate_mass = duplicate_mass.clamp_min(0.0).masked_fill(~valid, 0.0)
        denom = duplicate_target.sum().clamp_min(1.0)
        duplicate_loss = (duplicate_mass.square() * duplicate_target).sum() / denom
        losses["pc_ot_mras_soft_hard_duplicate_mass_loss"] = (
            duplicate_loss * float(merged_weights["duplicate_mass"])
        )

    if not losses:
        losses["pc_ot_mras_soft_hard_zero_loss"] = _zero_like_loss(reader_outputs)
    return losses


__all__ = ["pc_ot_mras_soft_hard_consistency_losses"]
