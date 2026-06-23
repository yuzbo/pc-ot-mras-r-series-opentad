from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from tools.bata.dump_pc_ot_mras_reader_snapshots import (  # noqa: E402
    _device_from_arg,
    _load_checkpoint_state,
    _metas_to_list,
    _move_batch_to_device,
    sample_ids_from_metas,
)
from tools.bata.export_pc_ot_mras_hard_positions import strict_json_value, write_json  # noqa: E402


SCHEMA_VERSION = "pc_ot_mras_training_assignment_geometry_replay_v0"
READY = "PC_OT_MRAS_TRAINING_ASSIGNMENT_GEOMETRY_REPLAY_READY"
NO_GO = "PC_OT_MRAS_TRAINING_ASSIGNMENT_GEOMETRY_REPLAY_NO_GO"


def _require_torch():
    module = sys.modules.get("torch")
    if module is not None:
        return module
    import torch as torch_module

    return torch_module


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _stats(values: Sequence[Any]) -> dict[str, Any]:
    finite = [float(item) for item in values if _finite_float(item) is not None]
    if not finite:
        return {"count": 0, "min": None, "mean": None, "max": None}
    return {
        "count": len(finite),
        "min": min(finite),
        "mean": sum(finite) / float(len(finite)),
        "max": max(finite),
    }


def _tensor_values(tensor: Any) -> list[float]:
    torch = _require_torch()
    if not torch.is_tensor(tensor) or tensor.numel() == 0:
        return []
    return [float(item) for item in tensor.detach().float().cpu().flatten().tolist() if math.isfinite(float(item))]


def _expand_bool_mask(mask: Any, tensor: Any) -> Any:
    torch = _require_torch()
    if mask is None:
        return None
    if not torch.is_tensor(mask) or not torch.is_tensor(tensor):
        raise TypeError("mask and tensor must both be torch tensors")
    out = mask.detach().to(device=tensor.device).bool()
    while out.ndim < tensor.ndim:
        out = out.unsqueeze(-1)
    if out.ndim != tensor.ndim:
        raise ValueError(f"mask rank {out.ndim} cannot be applied to tensor rank {tensor.ndim}")
    try:
        return out.expand_as(tensor)
    except RuntimeError as exc:
        raise ValueError(f"mask shape {tuple(mask.shape)} cannot broadcast to tensor shape {tuple(tensor.shape)}") from exc


def tensor_stats(tensor: Any, mask: Any | None = None) -> dict[str, Any]:
    torch = _require_torch()
    if not torch.is_tensor(tensor):
        return {"count": 0, "min": None, "mean": None, "max": None}
    values = tensor.detach()
    if mask is not None:
        values = values[_expand_bool_mask(mask, values)]
    return _stats(_tensor_values(values))


def _positive_count(tensor: Any, threshold: float, mask: Any | None = None) -> int:
    torch = _require_torch()
    if not torch.is_tensor(tensor):
        return 0
    valid = tensor > float(threshold)
    if mask is not None:
        valid = valid & _expand_bool_mask(mask, tensor)
    return int(valid.long().sum().item())


def _tensor_sum(tensor: Any, mask: Any | None = None) -> float:
    torch = _require_torch()
    if not torch.is_tensor(tensor) or tensor.numel() == 0:
        return 0.0
    values = tensor.detach()
    if mask is not None:
        values = values[_expand_bool_mask(mask, values)]
    if values.numel() == 0:
        return 0.0
    return float(values.float().sum().item())


def _meta_positions(meta: Mapping[str, Any]) -> list[float]:
    torch = sys.modules.get("torch")
    raw = meta.get("irregular_selected_positions")
    if raw is None and isinstance(meta.get("pc_ot_mras_bridge"), Mapping):
        raw = meta["pc_ot_mras_bridge"].get("selected_dense_positions")
    if raw is None:
        return []
    if torch is not None and torch.is_tensor(raw):
        raw = raw.detach().float().cpu().flatten().tolist()
    if not isinstance(raw, (list, tuple)):
        return []
    out = []
    for item in raw:
        value = _finite_float(item)
        if value is not None:
            out.append(value)
    return out


def summarize_metas(metas: Sequence[Mapping[str, Any]] | None, sample_ids: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, sample_id in enumerate(sample_ids):
        meta = metas[idx] if metas is not None and idx < len(metas) and isinstance(metas[idx], Mapping) else {}
        positions = _meta_positions(meta)
        deltas = [right - left for left, right in zip(positions, positions[1:])]
        dense_valid_len = _finite_float(meta.get("irregular_dense_valid_len", meta.get("irregular_selected_valid_len")))
        selected_count = meta.get("irregular_selected_count")
        if selected_count is None:
            selected_count = len(positions)
        rows.append(
            {
                "sample_id": str(sample_id),
                "has_irregular_native_axis": bool(meta.get("irregular_native_axis", False)),
                "selected_position_count": len(positions),
                "irregular_selected_count": int(selected_count) if _finite_float(selected_count) is not None else None,
                "dense_valid_len": dense_valid_len,
                "positions_strictly_increasing": all(delta > 0.0 for delta in deltas),
                "position_min": min(positions) if positions else None,
                "position_max": max(positions) if positions else None,
                "position_gap": _stats(deltas),
                "position_preview": positions[:8],
            }
        )
    return rows


def summarize_area_grid(level_idx: int, area_grid: Mapping[str, Any]) -> dict[str, Any]:
    torch = _require_torch()
    obs_valid = area_grid["obs_valid_mask"].bool()
    gap_valid = area_grid["gap_valid_mask"].bool()
    center = area_grid["obs_center"].detach().float()
    monotonic_violations = 0
    center_gaps = []
    for batch_idx in range(center.shape[0]):
        valid_count = int(obs_valid[batch_idx].long().sum().item())
        positions = center[batch_idx, :valid_count]
        if positions.numel() > 1:
            gaps = positions[1:] - positions[:-1]
            monotonic_violations += int((gaps <= 0).long().sum().item())
            center_gaps.extend(_tensor_values(gaps))
    return {
        "level": int(level_idx),
        "obs_shape": list(area_grid["obs_center"].shape),
        "gap_shape": list(area_grid["gap_center"].shape),
        "obs_valid_count": int(obs_valid.long().sum().item()),
        "gap_valid_count": int(gap_valid.long().sum().item()),
        "dense_valid_len": tensor_stats(area_grid["dense_valid_len"]),
        "obs_width": tensor_stats(area_grid["obs_width"], obs_valid),
        "gap_width": tensor_stats(area_grid["gap_width"], gap_valid),
        "center_gap": _stats(center_gaps),
        "center_monotonic_violation_count": int(monotonic_violations),
        "nonpositive_obs_width_count": int(((area_grid["obs_end"] - area_grid["obs_start"]) <= 0).long().sum().item()),
        "nonpositive_gap_width_count": int(((area_grid["gap_end"] - area_grid["gap_start"])[gap_valid] <= 0).long().sum().item())
        if bool(gap_valid.any().item())
        else 0,
    }


def summarize_level_targets(level_idx: int, targets: Mapping[str, Any], area_grid: Mapping[str, Any]) -> dict[str, Any]:
    obs_valid = area_grid["obs_valid_mask"].bool().unsqueeze(-1)
    gap_valid = area_grid["gap_valid_mask"].bool().unsqueeze(-1)
    area = targets["area"]
    start = targets["start_gap"]
    end = targets["end_gap"]
    start_weight = targets["start_offset_weight"]
    end_weight = targets["end_offset_weight"]
    return {
        "level": int(level_idx),
        "area_positive_mass": _tensor_sum(area, obs_valid),
        "start_positive_mass": _tensor_sum(start, gap_valid),
        "end_positive_mass": _tensor_sum(end, gap_valid),
        "area_gt_0_count": _positive_count(area, 0.0, obs_valid),
        "area_ge_0p5_count": _positive_count(area, 0.5, obs_valid),
        "start_gt_0_count": _positive_count(start, 0.0, gap_valid),
        "start_ge_0p5_count": _positive_count(start, 0.5, gap_valid),
        "end_gt_0_count": _positive_count(end, 0.0, gap_valid),
        "end_ge_0p5_count": _positive_count(end, 0.5, gap_valid),
        "start_offset_weight_gt_0_count": _positive_count(start_weight, 0.0, gap_valid),
        "end_offset_weight_gt_0_count": _positive_count(end_weight, 0.0, gap_valid),
        "area_target": tensor_stats(area, obs_valid),
        "start_target": tensor_stats(start, gap_valid),
        "end_target": tensor_stats(end, gap_valid),
        "start_offset_weight": tensor_stats(start_weight, gap_valid),
        "end_offset_weight": tensor_stats(end_weight, gap_valid),
    }


def _duration_range_for_level(head: Any, level_idx: int) -> tuple[float, float] | None:
    value = head._duration_range_for_level(level_idx) if hasattr(head, "_duration_range_for_level") else None
    if value is None:
        return None
    return float(value[0]), float(value[1])


def _gt_coverage_for_level(area_grid: Mapping[str, Any], gt_segment: Any, duration_range: tuple[float, float] | None, batch_idx: int) -> dict[str, Any]:
    torch = _require_torch()
    if gt_segment.numel() == 0:
        return {
            "gt_considered": 0,
            "max_area_coverage": [],
            "max_start_boundary_score": [],
            "max_end_boundary_score": [],
        }
    gt_segment = gt_segment.to(device=area_grid["obs_start"].device, dtype=area_grid["obs_start"].dtype)
    gt_start = gt_segment[:, 0]
    gt_end = gt_segment[:, 1]
    if duration_range is not None:
        duration = gt_end - gt_start
        keep = (duration >= duration_range[0]) & (duration <= duration_range[1])
        gt_start = gt_start[keep]
        gt_end = gt_end[keep]
    if gt_start.numel() == 0:
        return {
            "gt_considered": 0,
            "max_area_coverage": [],
            "max_start_boundary_score": [],
            "max_end_boundary_score": [],
        }

    obs_valid = area_grid["obs_valid_mask"][batch_idx].bool()
    gap_valid = area_grid["gap_valid_mask"][batch_idx].bool()
    obs_start = area_grid["obs_start"][batch_idx]
    obs_end = area_grid["obs_end"][batch_idx]
    obs_width = area_grid["obs_width"][batch_idx].clamp_min(1e-4)
    overlap_start = torch.maximum(obs_start[:, None], gt_start[None, :])
    overlap_end = torch.minimum(obs_end[:, None], gt_end[None, :])
    coverage = (overlap_end - overlap_start).clamp_min(0.0) / obs_width[:, None]
    coverage = coverage * obs_valid[:, None].to(coverage.dtype)
    gap_width = area_grid["gap_width"][batch_idx].clamp_min(1e-4)
    gap_center = area_grid["gap_center"][batch_idx]
    scale = gap_width[:, None]
    start_score = torch.exp(-torch.abs(gap_center[:, None] - gt_start[None, :]) / scale) * gap_valid[:, None].to(
        gap_center.dtype
    )
    end_score = torch.exp(-torch.abs(gap_center[:, None] - gt_end[None, :]) / scale) * gap_valid[:, None].to(
        gap_center.dtype
    )
    return {
        "gt_considered": int(gt_start.numel()),
        "max_area_coverage": _tensor_values(coverage.max(dim=0).values),
        "max_start_boundary_score": _tensor_values(start_score.max(dim=0).values),
        "max_end_boundary_score": _tensor_values(end_score.max(dim=0).values),
    }


def summarize_gt_coverage(area_grids: Sequence[Mapping[str, Any]], gt_segments: Sequence[Any], sample_ids: Sequence[str], head: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch_idx, sample_id in enumerate(sample_ids):
        area_values = []
        start_values = []
        end_values = []
        considered_total = 0
        gt_count = int(gt_segments[batch_idx].shape[0]) if batch_idx < len(gt_segments) else 0
        for level_idx, grid in enumerate(area_grids):
            level = _gt_coverage_for_level(grid, gt_segments[batch_idx], _duration_range_for_level(head, level_idx), batch_idx)
            considered_total += int(level["gt_considered"])
            area_values.extend(level["max_area_coverage"])
            start_values.extend(level["max_start_boundary_score"])
            end_values.extend(level["max_end_boundary_score"])
        rows.append(
            {
                "sample_id": str(sample_id),
                "gt_count": gt_count,
                "level_gt_considered_total": int(considered_total),
                "max_area_coverage": _stats(area_values),
                "max_start_boundary_score": _stats(start_values),
                "max_end_boundary_score": _stats(end_values),
                "gt_with_area_ge_0p5": sum(1 for value in area_values if value >= 0.5),
                "gt_with_start_ge_0p5": sum(1 for value in start_values if value >= 0.5),
                "gt_with_end_ge_0p5": sum(1 for value in end_values if value >= 0.5),
            }
        )
    return rows


def summarize_loss_dict(losses: Mapping[str, Any]) -> dict[str, Any]:
    torch = _require_torch()
    rows = {}
    for key, value in losses.items():
        if torch.is_tensor(value):
            detached = value.detach().float()
            rows[str(key)] = {
                "shape": list(detached.shape),
                "scalar": float(detached.item()) if detached.numel() == 1 else None,
                "stats": tensor_stats(detached),
                "finite": bool(torch.isfinite(detached).all().item()) if detached.numel() else True,
            }
        else:
            rows[str(key)] = {"value": strict_json_value(value)}
    return rows


def _rank_sampling_summary(quality_logit: Any, quality_target: Any, head: Any) -> dict[str, Any]:
    torch = _require_torch()
    pos = quality_target >= float(head.quality_rank_positive_iou)
    neg = quality_target <= float(head.quality_rank_negative_iou)
    pos_logit = quality_logit[pos]
    neg_logit = quality_logit[neg]
    raw_pos_count = int(pos_logit.numel())
    raw_neg_count = int(neg_logit.numel())
    if pos_logit.numel() > int(head.quality_rank_sample_size):
        _, pos_order = torch.topk(-pos_logit.detach(), k=int(head.quality_rank_sample_size))
        pos_logit = pos_logit[pos_order]
    if neg_logit.numel() > int(head.quality_rank_sample_size):
        _, neg_order = torch.topk(neg_logit.detach(), k=int(head.quality_rank_sample_size))
        neg_logit = neg_logit[neg_order]
    out = {
        "raw_positive_count": raw_pos_count,
        "raw_negative_count": raw_neg_count,
        "sampled_positive_count": int(pos_logit.numel()),
        "sampled_negative_count": int(neg_logit.numel()),
        "sampled_positive_logit": tensor_stats(pos_logit),
        "sampled_negative_logit": tensor_stats(neg_logit),
        "rank_pair_count": 0,
        "rank_margin_violation_fraction": None,
        "rank_loss_unweighted": None,
    }
    if pos_logit.numel() == 0 or neg_logit.numel() == 0:
        return out
    diff = pos_logit[:, None] - neg_logit[None, :]
    violation = diff < float(head.quality_rank_margin)
    out["rank_pair_count"] = int(diff.numel())
    out["rank_margin_violation_fraction"] = float(violation.float().mean().item())
    out["rank_loss_unweighted"] = float(head._quality_rank_loss(quality_logit, quality_target).detach().float().item())
    return out


def summarize_quality_calibration_targets(
    head: Any,
    preds: Mapping[str, Any],
    area_grids: Sequence[Mapping[str, Any]],
    gt_segments: Sequence[Any],
    gt_labels: Sequence[Any],
) -> dict[str, Any]:
    torch = _require_torch()
    enabled = bool(getattr(head, "enable_quality_calibration", False))
    rows: list[dict[str, Any]] = []
    if not enabled:
        return {
            "enabled": False,
            "row_count": 0,
            "reason": "quality_calibration_disabled",
            "rank_loss_weight": float(getattr(head, "quality_rank_loss_weight", 0.0)),
            "diagnostic_only": True,
        }
    for level_idx, grid in enumerate(area_grids):
        batch_count = int(preds["area_logits"][level_idx].shape[0])
        for batch_idx in range(batch_count):
            candidates = head._collect_area_level_pairs(preds, grid, level_idx, batch_idx)
            if not candidates or candidates["pair_start"].numel() == 0:
                rows.append(
                    {
                        "level": int(level_idx),
                        "batch_index": int(batch_idx),
                        "candidate_count_raw": 0,
                        "candidate_count_sampled": 0,
                        "status": "no_candidates",
                    }
                )
                continue
            gt_segment = gt_segments[batch_idx].to(
                device=candidates["pair_start"].device,
                dtype=candidates["pair_start"].dtype,
            )
            gt_label = gt_labels[batch_idx].to(device=candidates["pair_start"].device).long()
            if gt_segment.numel() == 0:
                quality_target = candidates["pair_start"].new_zeros(candidates["pair_start"].shape)
                boundary_target = quality_target
            else:
                quality_target = head._pair_iou_quality_target(candidates, gt_segment, gt_label)
                boundary_target = head._pair_boundary_quality_target(
                    candidates,
                    gt_segment,
                    gt_label,
                    tau=head.quality_boundary_tau,
                )
            quality_logit, boundary_logit = head._quality_calibration_logits(candidates)
            sampled_quality_logit, sampled_boundary_logit, sampled_quality_target, sampled_boundary_target = (
                head._sample_quality_training_rows(
                    quality_logit,
                    boundary_logit,
                    quality_target,
                    boundary_target,
                    candidates,
                )
            )
            sampled_count = int(sampled_quality_logit.numel())
            row = {
                "level": int(level_idx),
                "batch_index": int(batch_idx),
                "candidate_count_raw": int(quality_logit.numel()),
                "candidate_count_sampled": sampled_count,
                "max_pairs_per_class": int(getattr(head, "max_pairs_per_class", 0)),
                "num_classes": int(getattr(head, "num_classes", 0)),
                "quality_target": tensor_stats(sampled_quality_target),
                "boundary_target": tensor_stats(sampled_boundary_target),
                "quality_target_ge_rank_positive_count": _positive_count(
                    sampled_quality_target,
                    float(head.quality_rank_positive_iou) - 1.0e-8,
                ),
                "quality_target_le_rank_negative_count": int(
                    (sampled_quality_target <= float(head.quality_rank_negative_iou)).long().sum().item()
                ),
                "quality_logit": tensor_stats(sampled_quality_logit),
                "boundary_logit": tensor_stats(sampled_boundary_logit),
                "quality_probability": tensor_stats(torch.sigmoid(sampled_quality_logit)),
                "boundary_probability": tensor_stats(torch.sigmoid(sampled_boundary_logit)),
                "hand_score": tensor_stats(candidates.get("hand_score")),
                "rank": _rank_sampling_summary(sampled_quality_logit, sampled_quality_target, head)
                if float(getattr(head, "quality_rank_loss_weight", 0.0)) > 0
                else {
                    "raw_positive_count": 0,
                    "raw_negative_count": 0,
                    "sampled_positive_count": 0,
                    "sampled_negative_count": 0,
                    "rank_pair_count": 0,
                    "rank_margin_violation_fraction": None,
                    "rank_loss_unweighted": None,
                    "reason": "quality_rank_loss_weight_zero",
                },
            }
            rows.append(row)
    aggregate = {
        "candidate_count_raw": _stats([row.get("candidate_count_raw") for row in rows]),
        "candidate_count_sampled": _stats([row.get("candidate_count_sampled") for row in rows]),
        "quality_target_mean": _stats(
            [
                row.get("quality_target", {}).get("mean")
                for row in rows
                if isinstance(row.get("quality_target"), Mapping)
            ]
        ),
        "boundary_target_mean": _stats(
            [
                row.get("boundary_target", {}).get("mean")
                for row in rows
                if isinstance(row.get("boundary_target"), Mapping)
            ]
        ),
        "rank_pair_count": _stats(
            [row.get("rank", {}).get("rank_pair_count") for row in rows if isinstance(row.get("rank"), Mapping)]
        ),
        "rank_margin_violation_fraction": _stats(
            [
                row.get("rank", {}).get("rank_margin_violation_fraction")
                for row in rows
                if isinstance(row.get("rank"), Mapping)
            ]
        ),
        "rank_loss_unweighted": _stats(
            [row.get("rank", {}).get("rank_loss_unweighted") for row in rows if isinstance(row.get("rank"), Mapping)]
        ),
    }
    return {
        "enabled": True,
        "row_count": len(rows),
        "rank_positive_iou": float(head.quality_rank_positive_iou),
        "rank_negative_iou": float(head.quality_rank_negative_iou),
        "rank_margin": float(head.quality_rank_margin),
        "rank_sample_size": int(head.quality_rank_sample_size),
        "quality_loss_weight": float(head.quality_calibration_loss_weight),
        "boundary_loss_weight": float(head.quality_boundary_loss_weight),
        "rank_loss_weight": float(head.quality_rank_loss_weight),
        "rows": rows,
        "aggregate": aggregate,
        "diagnostic_only": True,
    }


def aggregate_batches(batch_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    level_rows = [row for batch in batch_rows for row in batch.get("level_target_summary", [])]
    grid_rows = [row for batch in batch_rows for row in batch.get("level_grid_summary", [])]
    meta_rows = [row for batch in batch_rows for row in batch.get("meta_summary", [])]
    gt_rows = [row for batch in batch_rows for row in batch.get("gt_coverage_summary", [])]
    quality_rows = [
        row
        for batch in batch_rows
        for row in batch.get("quality_calibration_summary", {}).get("rows", [])
        if isinstance(row, Mapping)
    ]
    loss_rows = [batch.get("actual_loss_summary", {}) for batch in batch_rows]
    loss_keys = sorted({str(key) for row in loss_rows if isinstance(row, Mapping) for key in row.keys()})
    return {
        "batch_count": len(batch_rows),
        "sample_count": sum(len(batch.get("sample_ids", [])) for batch in batch_rows),
        "area_positive_mass": _stats([row.get("area_positive_mass") for row in level_rows]),
        "start_positive_mass": _stats([row.get("start_positive_mass") for row in level_rows]),
        "end_positive_mass": _stats([row.get("end_positive_mass") for row in level_rows]),
        "area_ge_0p5_count": _stats([row.get("area_ge_0p5_count") for row in level_rows]),
        "start_ge_0p5_count": _stats([row.get("start_ge_0p5_count") for row in level_rows]),
        "end_ge_0p5_count": _stats([row.get("end_ge_0p5_count") for row in level_rows]),
        "center_monotonic_violation_count": int(sum(int(row.get("center_monotonic_violation_count", 0)) for row in grid_rows)),
        "samples_missing_irregular_native_axis": int(
            sum(1 for row in meta_rows if not bool(row.get("has_irregular_native_axis")))
        ),
        "samples_with_nonmonotonic_positions": int(
            sum(1 for row in meta_rows if row.get("positions_strictly_increasing") is False)
        ),
        "gt_with_area_ge_0p5": _stats([row.get("gt_with_area_ge_0p5") for row in gt_rows]),
        "gt_with_start_ge_0p5": _stats([row.get("gt_with_start_ge_0p5") for row in gt_rows]),
        "gt_with_end_ge_0p5": _stats([row.get("gt_with_end_ge_0p5") for row in gt_rows]),
        "quality_candidate_count_sampled": _stats([row.get("candidate_count_sampled") for row in quality_rows]),
        "quality_target_mean": _stats(
            [
                row.get("quality_target", {}).get("mean")
                for row in quality_rows
                if isinstance(row.get("quality_target"), Mapping)
            ]
        ),
        "quality_rank_pair_count": _stats(
            [
                row.get("rank", {}).get("rank_pair_count")
                for row in quality_rows
                if isinstance(row.get("rank"), Mapping)
            ]
        ),
        "quality_rank_margin_violation_fraction": _stats(
            [
                row.get("rank", {}).get("rank_margin_violation_fraction")
                for row in quality_rows
                if isinstance(row.get("rank"), Mapping)
            ]
        ),
        "actual_loss_scalar": {
            key: _stats(
                [
                    row.get(key, {}).get("scalar")
                    for row in loss_rows
                    if isinstance(row, Mapping) and isinstance(row.get(key), Mapping)
                ]
            )
            for key in loss_keys
        },
    }


def _unwrap_model(model: Any) -> Any:
    return model.module if hasattr(model, "module") else model


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(strict_json_value(row), sort_keys=True) + "\n")


def run_training_assignment_replay(
    *,
    config: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    split: str = "train",
    limit_batches: int = 2,
    device: str = "auto",
    batch_size: int | None = None,
    num_workers: int | None = 0,
    use_ema: bool | None = None,
    use_amp: bool = False,
) -> dict[str, Any]:
    torch = _require_torch()
    from mmengine.config import Config
    from opentad.datasets import build_dataloader, build_dataset
    from opentad.models import build_detector
    from opentad.models.utils.pc_ot_mras_raw_prediction_guard import assert_no_raw_prediction_shortcut_for_pc_ot_mras
    from opentad.models.utils.temporal_grid import prepare_area_targets

    if int(limit_batches) <= 0:
        raise ValueError("limit_batches must be positive")
    cfg = Config.fromfile(str(config))
    assert_no_raw_prediction_shortcut_for_pc_ot_mras(cfg)
    if not hasattr(cfg, "dataset") or split not in cfg.dataset:
        raise ValueError(f"config missing dataset.{split}")
    if not hasattr(cfg, "solver") or split not in cfg.solver:
        raise ValueError(f"config missing solver.{split}")

    loader_kwargs = dict(cfg.solver[split])
    if batch_size is not None:
        loader_kwargs["batch_size"] = int(batch_size)
    if num_workers is not None:
        loader_kwargs["num_workers"] = int(num_workers)
    dataset = build_dataset(cfg.dataset[split], default_args=dict(logger=None))
    dataloader = build_dataloader(dataset, rank=0, world_size=1, shuffle=False, drop_last=False, **loader_kwargs)
    torch_device = _device_from_arg(str(device))
    model = build_detector(cfg.model)
    epoch = _load_checkpoint_state(model, checkpoint, use_ema=use_ema)
    model.to(torch_device)
    model.eval()
    head = getattr(_unwrap_model(model), "rpn_head", None)
    if head is None:
        raise ValueError("model has no rpn_head")
    if not hasattr(head, "_prepare_flat_targets") or not hasattr(head, "_duration_range_for_level"):
        raise ValueError("rpn_head does not expose P2 training target helpers")

    original_forward_train = head.forward_train
    original_losses = head.losses
    capture_state: dict[str, Any] = {"metas": None, "batch_rows": [], "active": None}

    def wrapped_forward_train(feat_list, mask_list, gt_segments, gt_labels, metas=None, **kwargs):
        capture_state["metas"] = metas
        return original_forward_train(feat_list, mask_list, gt_segments=gt_segments, gt_labels=gt_labels, metas=metas, **kwargs)

    def wrapped_losses(preds, area_grids, gt_segments, gt_labels):
        active = capture_state.get("active") or {}
        metas = capture_state.get("metas")
        sample_ids = list(active.get("sample_ids", []))
        if not sample_ids and metas is not None:
            sample_ids = sample_ids_from_metas(metas, seen_count=int(active.get("seen_count", 0)))
        level_grid_summary = [summarize_area_grid(level_idx, grid) for level_idx, grid in enumerate(area_grids)]
        level_target_summary = []
        for level_idx, grid in enumerate(area_grids):
            targets = prepare_area_targets(
                grid,
                gt_segments,
                gt_labels,
                num_classes=head.num_classes,
                boundary_tau=head.boundary_tau,
                duration_range=head._duration_range_for_level(level_idx),
            )
            level_target_summary.append(summarize_level_targets(level_idx, targets, grid))
        quality_summary = summarize_quality_calibration_targets(head, preds, area_grids, gt_segments, gt_labels)
        losses = original_losses(preds, area_grids, gt_segments, gt_labels)
        capture_state["batch_rows"].append(
            {
                "schema_version": SCHEMA_VERSION,
                "batch_index": int(active.get("batch_index", len(capture_state["batch_rows"]))),
                "sample_ids": sample_ids,
                "meta_summary": summarize_metas(metas, sample_ids),
                "gt_coverage_summary": summarize_gt_coverage(area_grids, gt_segments, sample_ids, head),
                "level_grid_summary": level_grid_summary,
                "level_target_summary": level_target_summary,
                "quality_calibration_summary": quality_summary,
                "actual_loss_summary": summarize_loss_dict(losses),
                "diagnostic_only": True,
                "uses_train_gt": True,
                "uses_train_gt_for_offline_assignment_replay_only": True,
                "no_optimizer": True,
                "no_backward": True,
                "metric_claim_allowed": False,
                "paper_claim_allowed": False,
            }
        )
        return losses

    head.forward_train = wrapped_forward_train
    head.losses = wrapped_losses
    seen_count = 0
    try:
        with torch.no_grad():
            for batch_idx, data_dict in enumerate(dataloader):
                if batch_idx >= int(limit_batches):
                    break
                batch = _move_batch_to_device(data_dict, torch_device)
                metas = _metas_to_list(batch.get("metas"), batch_size=int(batch["inputs"].shape[0]))
                batch["metas"] = metas
                sample_ids = sample_ids_from_metas(metas, seen_count=seen_count)
                capture_state["active"] = {
                    "batch_index": batch_idx,
                    "sample_ids": sample_ids,
                    "seen_count": seen_count,
                }
                with torch.cuda.amp.autocast(dtype=torch.float16, enabled=bool(use_amp) and torch_device.type == "cuda"):
                    model.forward_train(
                        inputs=batch["inputs"],
                        masks=batch["masks"],
                        metas=metas,
                        gt_segments=batch["gt_segments"],
                        gt_labels=batch["gt_labels"],
                    )
                seen_count += len(sample_ids)
    finally:
        head.forward_train = original_forward_train
        head.losses = original_losses

    batch_rows = list(capture_state["batch_rows"])
    if not batch_rows:
        raise RuntimeError("no training assignment rows were captured")
    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    batch_jsonl = out_dir / "training_assignment_batches.jsonl"
    _write_jsonl(batch_jsonl, batch_rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "decision": READY,
        "diagnostic": "D8_real_training_assignment_geometry_replay",
        "config": str(config),
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": epoch,
        "output_dir": str(out_dir),
        "batch_jsonl": str(batch_jsonl),
        "split": str(split),
        "limit_batches": int(limit_batches),
        "batch_count": len(batch_rows),
        "aggregate": aggregate_batches(batch_rows),
        "exact_training_assignment_replay": True,
        "uses_train_gt": True,
        "uses_train_gt_for_offline_assignment_replay_only": True,
        "uses_validation_gt": False,
        "uses_test_gt": False,
        "uses_teacher": False,
        "uses_oracle_for_training_or_test_protocol": False,
        "uses_raw_prediction": False,
        "no_optimizer": True,
        "no_backward": True,
        "tools_train_allowed": False,
        "tools_test_allowed": False,
        "detector_map_allowed": False,
        "metric_claim_allowed": False,
        "paper_claim_allowed": False,
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def _parse_use_ema(value: str) -> bool | None:
    lowered = str(value).lower()
    if lowered in {"auto", "none"}:
        return None
    if lowered in {"1", "true", "yes", "ema"}:
        return True
    if lowered in {"0", "false", "no", "raw"}:
        return False
    raise argparse.ArgumentTypeError("--use-ema must be auto, true, or false")


def error_payload(exc: BaseException) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": NO_GO,
        "error_type": exc.__class__.__name__,
        "error": str(exc),
        "diagnostic_only": True,
        "metric_claim_allowed": False,
        "paper_claim_allowed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay PC-OT-MRAS P2 training-side assignment geometry without optimizer updates.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit-batches", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--use-ema", type=_parse_use_ema, default=None)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args(argv)

    try:
        summary = run_training_assignment_replay(
            config=args.config,
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            split=args.split,
            limit_batches=int(args.limit_batches),
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_ema=args.use_ema,
            use_amp=bool(args.amp),
        )
    except Exception as exc:  # pragma: no cover - CLI guard
        print(json.dumps(strict_json_value(error_payload(exc)), sort_keys=True))
        return 1
    print(json.dumps(strict_json_value(summary), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
