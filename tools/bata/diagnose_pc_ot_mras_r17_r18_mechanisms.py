from __future__ import annotations

import argparse
import functools
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SCHEMA_VERSION = "pc_ot_mras_r17_r18_mechanism_diagnostic_v0"
READY = "PC_OT_MRAS_R17_R18_MECHANISM_DIAGNOSTIC_READY"
NO_GO = "PC_OT_MRAS_R17_R18_MECHANISM_DIAGNOSTIC_NO_GO"
MATRIX_PRIORITY = ("acquisition_matrix", "allocation", "transport_prob")


def strict_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json_value(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    return str(value)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(strict_json_value(dict(payload)), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(strict_json_value(dict(row)), sort_keys=True) + "\n")


def _to_plain(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "tolist"):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist") and not isinstance(value, (list, tuple, Mapping, str, bytes)):
        try:
            return value.tolist()
        except TypeError:
            pass
    if isinstance(value, Mapping):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    return value


def _depth(value: Any) -> int:
    data = _to_plain(value)
    depth = 0
    while isinstance(data, list):
        depth += 1
        data = data[0] if data else None
    return depth


def _sample(value: Any, batch_idx: int, batch_size: int) -> Any:
    data = _to_plain(value)
    if isinstance(data, list) and batch_size > 1:
        return data[batch_idx]
    if isinstance(data, list) and batch_size == 1 and _depth(data) > 1:
        return data[batch_idx]
    return data


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def stats(values: Sequence[Any]) -> dict[str, Any]:
    finite = [_as_float(value) for value in values]
    finite = [value for value in finite if value is not None]
    if not finite:
        return {"count": 0, "min": None, "mean": None, "max": None}
    return {
        "count": len(finite),
        "min": min(finite),
        "mean": sum(finite) / float(len(finite)),
        "max": max(finite),
    }


def histogram(values: Sequence[Any], bins: int = 10, value_range: tuple[float, float] = (0.0, 1.0)) -> dict[str, Any]:
    lo, hi = value_range
    counts = [0 for _ in range(int(bins))]
    finite = [_as_float(value) for value in values]
    finite = [value for value in finite if value is not None]
    if not finite:
        return {"bins": int(bins), "range": [lo, hi], "counts": counts}
    width = (hi - lo) / float(bins)
    for value in finite:
        idx = int((value - lo) / width) if width > 0 else 0
        idx = max(0, min(int(bins) - 1, idx))
        counts[idx] += 1
    return {"bins": int(bins), "range": [lo, hi], "counts": counts}


def temporal_iou(a: Sequence[float], b: Sequence[float]) -> float:
    start, end = float(a[0]), float(a[1])
    gt_start, gt_end = float(b[0]), float(b[1])
    inter = max(0.0, min(end, gt_end) - max(start, gt_start))
    union = max(0.0, end - start) + max(0.0, gt_end - gt_start) - inter
    return float(inter / union) if union > 0.0 else 0.0


def pearson(xs: Sequence[Any], ys: Sequence[Any]) -> float | None:
    pairs: list[tuple[float, float]] = []
    for x, y in zip(xs, ys):
        fx = _as_float(x)
        fy = _as_float(y)
        if fx is not None and fy is not None:
            pairs.append((fx, fy))
    if len(pairs) < 2:
        return None
    x_mean = sum(x for x, _ in pairs) / float(len(pairs))
    y_mean = sum(y for _, y in pairs) / float(len(pairs))
    x_var = sum((x - x_mean) ** 2 for x, _ in pairs)
    y_var = sum((y - y_mean) ** 2 for _, y in pairs)
    if x_var <= 0.0 or y_var <= 0.0:
        return None
    cov = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    return float(cov / math.sqrt(x_var * y_var))


def _binary_mask_positions(mask: Any) -> list[int]:
    row = _to_plain(mask)
    if not isinstance(row, list):
        raise ValueError("valid_mask sample must be a list")
    positions = []
    for idx, value in enumerate(row):
        if value not in (0, 1, False, True):
            raise ValueError(f"valid_mask[{idx}] must be binary")
        if bool(value):
            positions.append(int(idx))
    if not positions:
        raise ValueError("valid_mask must contain at least one valid position")
    if positions != list(range(len(positions))):
        raise ValueError("valid_mask must be prefix-contiguous")
    return positions


def _matrix_rows_for_sample(reader_out: Mapping[str, Any], batch_idx: int, batch_size: int) -> tuple[str | None, list[list[float]]]:
    for key in MATRIX_PRIORITY:
        if key not in reader_out:
            continue
        rows = _sample(reader_out[key], batch_idx, batch_size)
        if not isinstance(rows, list) or not rows or not all(isinstance(row, list) for row in rows):
            raise ValueError(f"{key} sample must be [K,T]")
        out = []
        width = None
        for row_idx, row in enumerate(rows):
            if width is None:
                width = len(row)
            elif len(row) != width:
                raise ValueError(f"{key} sample must be rectangular")
            checked = []
            for col_idx, value in enumerate(row):
                score = _as_float(value)
                if score is None:
                    raise ValueError(f"{key}[{row_idx}][{col_idx}] must be finite numeric")
                checked.append(score)
            out.append(checked)
        return key, out
    return None, []


def reader_selection_utility(
    reader_out: Mapping[str, Any],
    *,
    batch_idx: int,
    batch_size: int,
    budget: int,
    gt_segments: Sequence[Sequence[float]] | None = None,
    boundary_radius: float = 2.0,
) -> dict[str, Any]:
    valid_mask = _sample(reader_out["valid_mask"], batch_idx, batch_size)
    valid_positions = _binary_mask_positions(valid_mask)
    valid_set = set(valid_positions)
    matrix_key, matrix_rows = _matrix_rows_for_sample(reader_out, batch_idx, batch_size)
    dense_len = len(_to_plain(valid_mask))
    target_budget = min(int(budget), len(valid_positions))
    scores = [0.0 for _ in range(dense_len)]
    selected: dict[int, float] = {}
    duplicate_slot_count = 0
    invalid_slot_count = 0
    if matrix_rows:
        for slot in matrix_rows:
            best_pos = max(valid_positions, key=lambda pos: (slot[pos], -pos))
            if best_pos in selected:
                duplicate_slot_count += 1
            selected[best_pos] = max(float(selected.get(best_pos, float("-inf"))), float(slot[best_pos]))
            for pos in valid_positions:
                scores[pos] += float(slot[pos])
    if len(selected) < target_budget:
        ranked_fill = sorted(valid_positions, key=lambda pos: (-scores[pos], pos))
        for pos in ranked_fill:
            if pos not in selected:
                selected[pos] = scores[pos]
            if len(selected) >= target_budget:
                break
    ranked_selected = sorted(selected.items(), key=lambda item: (-float(item[1]), int(item[0])))[:target_budget]
    selected_positions = sorted(int(pos) for pos, _score in ranked_selected)
    gaps = [b - a for a, b in zip(selected_positions, selected_positions[1:])]
    invalid_slot_count += sum(1 for pos in selected_positions if pos not in valid_set)

    gt_intervals = [[float(seg[0]), float(seg[1])] for seg in (gt_segments or []) if len(seg) >= 2 and float(seg[1]) > float(seg[0])]
    gt_coverage: dict[str, Any]
    if not gt_intervals:
        gt_coverage = {"status": "no_gt_segments_available"}
    else:
        selected_float = [float(pos) for pos in selected_positions]
        touched = 0
        boundary_touched = 0
        for start, end in gt_intervals:
            if any(start <= pos <= end for pos in selected_float):
                touched += 1
            if any(abs(pos - start) <= boundary_radius or abs(pos - end) <= boundary_radius for pos in selected_float):
                boundary_touched += 1
        inside_count = sum(1 for pos in selected_float if any(start <= pos <= end for start, end in gt_intervals))
        gt_coverage = {
            "status": "computed_on_batch_gt_axis_assumed_compatible",
            "gt_count": len(gt_intervals),
            "gt_touched_fraction": touched / float(len(gt_intervals)),
            "boundary_touched_fraction": boundary_touched / float(len(gt_intervals)),
            "selected_inside_gt_fraction": inside_count / float(max(1, len(selected_positions))),
            "boundary_radius": float(boundary_radius),
        }

    return {
        "matrix_key": matrix_key,
        "dense_len": int(dense_len),
        "valid_len": int(len(valid_positions)),
        "target_budget": int(target_budget),
        "selected_count": int(len(selected_positions)),
        "selected_positions": selected_positions,
        "selected_gap": stats(gaps),
        "duplicate_slot_count": int(duplicate_slot_count),
        "invalid_slot_count": int(invalid_slot_count),
        "selected_density": len(selected_positions) / float(max(1, len(valid_positions))),
        "gt_coverage": gt_coverage,
    }


def geometry_contract_audit(meta: Mapping[str, Any], d1: Mapping[str, Any], proposal_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    flags: list[str] = []
    positions = _to_plain(meta.get("irregular_selected_positions"))
    native_axis = meta.get("irregular_native_axis")
    selected_valid_len = _as_float(meta.get("irregular_selected_valid_len"))
    selected_count_meta = _as_int(meta.get("irregular_selected_count"))

    if native_axis is None:
        flags.append("missing_irregular_native_axis")
    if not isinstance(positions, list) or not positions:
        flags.append("missing_irregular_selected_positions")
        positions = []
    numeric_positions = [_as_float(pos) for pos in positions]
    if any(pos is None for pos in numeric_positions):
        flags.append("non_numeric_irregular_selected_positions")
        numeric_positions = [pos for pos in numeric_positions if pos is not None]
    if any(b < a for a, b in zip(numeric_positions, numeric_positions[1:])):
        flags.append("unsorted_irregular_selected_positions")
    if selected_count_meta is not None and selected_count_meta != len(numeric_positions):
        flags.append("selected_count_metadata_mismatch")
    if selected_valid_len is not None and numeric_positions and max(numeric_positions) >= selected_valid_len:
        flags.append("selected_position_outside_valid_len")

    d1_positions = d1.get("selected_positions", [])
    if isinstance(d1_positions, list) and numeric_positions and len(d1_positions) == len(numeric_positions):
        mean_abs_diff = sum(abs(float(a) - float(b)) for a, b in zip(d1_positions, numeric_positions)) / float(len(d1_positions))
    else:
        mean_abs_diff = None

    proposal_starts = [_as_float(row.get("start_coord")) for row in proposal_rows]
    proposal_ends = [_as_float(row.get("end_coord")) for row in proposal_rows]
    bad_duration = 0
    outside_valid = 0
    for start, end in zip(proposal_starts, proposal_ends):
        if start is None or end is None:
            flags.append("proposal_non_numeric_coord")
            continue
        if end <= start:
            bad_duration += 1
        if selected_valid_len is not None and (start < 0.0 or end > selected_valid_len):
            outside_valid += 1
    if bad_duration:
        flags.append("proposal_non_positive_duration")
    if outside_valid:
        flags.append("proposal_outside_selected_valid_len")
    if not proposal_rows:
        flags.append("no_proposals")

    return {
        "irregular_native_axis": native_axis,
        "irregular_selected_count": len(numeric_positions),
        "irregular_selected_valid_len": selected_valid_len,
        "mean_abs_reader_meta_selected_position_diff": mean_abs_diff,
        "proposal_count": len(proposal_rows),
        "proposal_start_coord": stats(proposal_starts),
        "proposal_end_coord": stats(proposal_ends),
        "proposal_non_positive_duration_count": int(bad_duration),
        "proposal_outside_valid_len_count": int(outside_valid),
        "failure_flags": sorted(set(flags)),
    }


def proposal_survival_audit(
    proposal_rows: Sequence[Mapping[str, Any]],
    *,
    topk: Sequence[int] = (100, 300, 1000),
    iou_thresholds: Sequence[float] = (0.3, 0.5, 0.7),
    score_bins: int = 10,
) -> dict[str, Any]:
    rows = list(proposal_rows)
    scores = [_as_float(row.get("final_score")) for row in rows]
    ious = [_as_float(row.get("diagnostic_max_gt_iou")) for row in rows]
    sorted_rows = sorted(rows, key=lambda row: _as_float(row.get("final_score")) or float("-inf"), reverse=True)
    topk_recall = {}
    has_gt = any(iou is not None for iou in ious)
    for k in topk:
        slice_rows = sorted_rows[: int(k)]
        topk_recall[str(int(k))] = {}
        for thr in iou_thresholds:
            if not has_gt:
                topk_recall[str(int(k))][str(float(thr))] = None
                continue
            topk_recall[str(int(k))][str(float(thr))] = any(
                (_as_float(row.get("diagnostic_max_gt_iou")) or 0.0) >= float(thr) for row in slice_rows
            )
    level_counts: dict[str, int] = {}
    for row in rows:
        level = str(row.get("level", "unknown"))
        level_counts[level] = level_counts.get(level, 0) + 1
    return {
        "candidate_count": len(rows),
        "level_counts": level_counts,
        "final_score": stats(scores),
        "score_histogram": histogram(scores, bins=int(score_bins), value_range=(0.0, 1.0)),
        "diagnostic_max_gt_iou": stats(ious),
        "score_iou_pearson": pearson(scores, ious) if has_gt else None,
        "topk_any_iou_at_threshold": topk_recall,
        "post_nms_available": False,
        "post_nms_note": "This D3 light audit inspects pre-NMS proposal survival and score/IoU ordering only.",
    }


def classify_summary_verdict(
    *,
    all_d1: Sequence[Mapping[str, Any]],
    all_d2: Sequence[Mapping[str, Any]],
    all_d3: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    has_gt = any(d1.get("gt_coverage", {}).get("status") == "computed_on_batch_gt_axis_assumed_compatible" for d1 in all_d1)
    d2_flags = [flag for d2 in all_d2 for flag in d2.get("failure_flags", []) if flag != "no_proposals"]
    proposal_counts = [int(d3.get("candidate_count", 0) or 0) for d3 in all_d3]
    no_proposal_fraction = (
        sum(1 for count in proposal_counts if count <= 0) / float(len(proposal_counts)) if proposal_counts else 1.0
    )
    boundary_values = [
        _as_float(d1.get("gt_coverage", {}).get("boundary_touched_fraction"))
        for d1 in all_d1
    ]
    boundary_values = [value for value in boundary_values if value is not None]
    boundary_mean = sum(boundary_values) / float(len(boundary_values)) if boundary_values else None

    if d2_flags:
        return {
            "verdict": "fail",
            "primary_failure_mode": "D2_bad",
            "notes": "Temporal metadata/proposal-coordinate contract emitted failure flags.",
            "has_gt": has_gt,
            "has_meta": False,
        }
    if no_proposal_fraction >= 0.5:
        return {
            "verdict": "fail",
            "primary_failure_mode": "D3_bad",
            "notes": "At least half of inspected samples produced no retained pre-NMS proposal rows.",
            "has_gt": has_gt,
            "has_meta": True,
        }
    if has_gt and boundary_mean is not None and boundary_mean < 0.5:
        return {
            "verdict": "warning",
            "primary_failure_mode": "D1_suspect",
            "notes": "Boundary coverage is low in this diagnostic slice; compare with uniform/random controls before claiming.",
            "has_gt": has_gt,
            "has_meta": True,
        }
    return {
        "verdict": "warning" if not has_gt else "pass",
        "primary_failure_mode": "unknown" if not has_gt else "none_detected_in_slice",
        "notes": "Diagnostic slice completed; this is not an mAP-producing run.",
        "has_gt": has_gt,
        "has_meta": True,
    }


class _RPNInputCapture:
    def __init__(self) -> None:
        self.latest_inputs: tuple[Any, ...] | None = None
        self.latest_kwargs: Mapping[str, Any] = {}

    def __call__(self, _module: Any, inputs: Sequence[Any], kwargs: Mapping[str, Any] | None = None) -> None:
        self.latest_inputs = tuple(inputs)
        self.latest_kwargs = dict(kwargs or {})

    def pop(self) -> tuple[tuple[Any, ...], Mapping[str, Any]]:
        if self.latest_inputs is None:
            raise RuntimeError("RPN head input hook did not capture proposal features")
        inputs = self.latest_inputs
        kwargs = self.latest_kwargs
        self.latest_inputs = None
        self.latest_kwargs = {}
        return inputs, kwargs


def _register_rpn_pre_hook(rpn_head: Any, capture: _RPNInputCapture):
    original = rpn_head.forward_test

    @functools.wraps(original)
    def wrapped_forward_test(*args, **kwargs):
        capture(None, args, kwargs)
        return original(*args, **kwargs)

    rpn_head.forward_test = wrapped_forward_test

    class _Handle:
        def remove(self) -> None:
            rpn_head.forward_test = original

    return _Handle()


def _module(model: Any) -> Any:
    return model.module if hasattr(model, "module") else model


def _metas_to_list(metas: Any, *, batch_size: int) -> list[Mapping[str, Any]]:
    from tools.bata.dump_pc_ot_mras_reader_snapshots import _metas_to_list as base_metas_to_list

    return base_metas_to_list(metas, batch_size=batch_size)


def diagnostic_metas_from_head_kwargs(
    captured_kwargs: Mapping[str, Any],
    original_metas: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
) -> list[Mapping[str, Any]]:
    """Prefer head-received metas because neck metadata is written after the original dataloader metas."""
    head_metas = captured_kwargs.get("metas")
    if head_metas is None:
        return list(original_metas)
    return _metas_to_list(head_metas, batch_size=batch_size)


def _sample_gt_segments(data_dict: Mapping[str, Any], sample_idx: int, batch_size: int) -> list[list[float]]:
    if "gt_segments" not in data_dict:
        return []
    value = _to_plain(data_dict["gt_segments"])
    if not isinstance(value, list):
        return []
    if batch_size > 1 and len(value) > sample_idx and isinstance(value[sample_idx], list):
        value = value[sample_idx]
    elif batch_size == 1 and len(value) == 1 and isinstance(value[0], list):
        value = value[sample_idx]
    out: list[list[float]] = []
    if value and all(isinstance(item, (int, float)) for item in value) and len(value) == 2:
        value = [value]
    if not isinstance(value, list):
        return []
    for item in value:
        if not isinstance(item, list) or len(item) < 2:
            continue
        start = _as_float(item[0])
        end = _as_float(item[1])
        if start is not None and end is not None and end > start:
            out.append([start, end])
    return out


def _sample_gt_segments_from_meta_or_batch(
    meta: Mapping[str, Any],
    data_dict: Mapping[str, Any],
    sample_idx: int,
    batch_size: int,
) -> list[list[float]]:
    from tools.bata.dump_native_irregular_p2_proposal_factors import _segments_from_meta

    gt_segments = _segments_from_meta(meta)
    return gt_segments if gt_segments else _sample_gt_segments(data_dict, sample_idx, batch_size)


def _rows_by_sample(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id", row.get("video_id", "unknown")))
        grouped.setdefault(sample_id, []).append(row)
    return grouped


def _short_sample_id(meta: Mapping[str, Any], fallback: str) -> str:
    video = str(meta.get("video_name") or meta.get("video_id") or fallback)
    window = _as_int(meta.get("window_start_frame"))
    return f"{video}|{window}" if window is not None else video


def run_case(
    *,
    case_name: str,
    config: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    split: str = "val",
    limit_batches: int = 4,
    device: str = "auto",
    use_ema: bool | None = None,
    use_amp: bool = False,
    budget: int = 384,
    topk_per_sample: int = 1000,
    score_bins: int = 10,
) -> dict[str, Any]:
    import torch
    from mmengine.config import Config
    from opentad.datasets import build_dataloader, build_dataset
    from opentad.models import build_detector
    from opentad.models.utils.pc_ot_mras_raw_prediction_guard import assert_no_raw_prediction_shortcut_for_pc_ot_mras
    from tools.bata.dump_native_irregular_p2_proposal_factors import _label_names_from_cfg, _max_iou
    from tools.bata.dump_pc_ot_mras_reader_snapshots import (
        ReaderOutputHook,
        _device_from_arg,
        _load_checkpoint_state,
        _model_reader_module,
        _move_batch_to_device,
    )

    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    proposal_jsonl = out_dir / f"{case_name}_proposal_factors.jsonl"
    sample_jsonl = out_dir / f"{case_name}_sample_diagnostics.jsonl"
    summary_json = out_dir / f"{case_name}_mechanism_summary.json"

    cfg = Config.fromfile(str(config))
    assert_no_raw_prediction_shortcut_for_pc_ot_mras(cfg)
    if not hasattr(cfg, "dataset") or split not in cfg.dataset:
        raise ValueError(f"config missing dataset.{split}")
    if not hasattr(cfg, "solver") or split not in cfg.solver:
        raise ValueError(f"config missing solver.{split}")

    torch_device = _device_from_arg(str(device))
    dataset = build_dataset(cfg.dataset[split], default_args=dict(logger=None))
    loader = build_dataloader(dataset, rank=0, world_size=1, shuffle=False, drop_last=False, **cfg.solver[split])
    model = build_detector(cfg.model)
    epoch = _load_checkpoint_state(model, checkpoint, use_ema=use_ema)
    model.to(torch_device)
    model.eval()
    module = _module(model)
    reader = _model_reader_module(model)
    rpn_head = getattr(module, "rpn_head", None)
    if rpn_head is None or not hasattr(rpn_head, "dump_proposal_factors"):
        raise ValueError("model.rpn_head has no dump_proposal_factors method")
    label_names = _label_names_from_cfg(cfg, split)

    reader_hook = ReaderOutputHook()
    reader_handle = reader.register_forward_hook(reader_hook)
    rpn_capture = _RPNInputCapture()
    rpn_handle = _register_rpn_pre_hook(rpn_head, rpn_capture)
    proposal_rows_out: list[dict[str, Any]] = []
    sample_rows_out: list[dict[str, Any]] = []
    total_samples = 0
    try:
        for batch_idx, data_dict in enumerate(loader):
            if int(limit_batches) > 0 and batch_idx >= int(limit_batches):
                break
            batch = _move_batch_to_device(data_dict, torch_device)
            batch_size = int(batch["inputs"].shape[0])
            metas = _metas_to_list(data_dict.get("metas"), batch_size=batch_size)
            batch["metas"] = metas
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.float16, enabled=bool(use_amp) and torch_device.type == "cuda"):
                    module.forward_test(batch["inputs"], batch["masks"], metas=metas, infer_cfg=cfg.inference)
            reader_out = reader_hook.pop()
            captured_inputs, captured_kwargs = rpn_capture.pop()
            if len(captured_inputs) < 2:
                raise RuntimeError("captured RPN input does not contain feature and mask lists")
            feat_list, mask_list = captured_inputs[0], captured_inputs[1]
            diagnostic_metas = diagnostic_metas_from_head_kwargs(
                captured_kwargs,
                metas,
                batch_size=batch_size,
            )
            proposal_rows = rpn_head.dump_proposal_factors(feat_list, mask_list, metas=diagnostic_metas, label_names=label_names)
            grouped_proposals = _rows_by_sample(proposal_rows)
            for sample_idx, meta in enumerate(diagnostic_metas):
                sample_id = _short_sample_id(meta, f"sample_{total_samples + sample_idx}")
                gt_segments = _sample_gt_segments_from_meta_or_batch(meta, data_dict, sample_idx, batch_size)
                d1 = reader_selection_utility(
                    reader_out,
                    batch_idx=sample_idx,
                    batch_size=batch_size,
                    budget=int(budget),
                    gt_segments=gt_segments,
                )
                matching = grouped_proposals.get(sample_id)
                if matching is None:
                    # dump_proposal_factors includes a richer sample id with batch/window fields.
                    video = str(meta.get("video_name") or meta.get("video_id") or sample_id.split("|")[0])
                    matching = [
                        row
                        for row in proposal_rows
                        if str(row.get("video_id")) == video
                        and int(row.get("batch_sample_idx", -1)) == int(sample_idx)
                    ]
                matching = sorted(
                    matching,
                    key=lambda row: _as_float(row.get("final_score")) or float("-inf"),
                    reverse=True,
                )[: int(topk_per_sample)]
                matching_out_rows: list[dict[str, Any]] = []
                for row in matching:
                    out_row = {
                        "schema_version": SCHEMA_VERSION,
                        "case_name": case_name,
                        "diagnostic": "D3_proposal_factor_row",
                        "diagnostic_only": True,
                        "uses_validation_gt_for_diagnostic_iou_only": bool(gt_segments),
                        "metric_claim_allowed": False,
                        "paper_claim_allowed": False,
                        **dict(row),
                    }
                    if gt_segments:
                        iou = _max_iou(row.get("segment", []), gt_segments)
                        out_row["uses_gt_for_diagnostic_iou"] = iou is not None
                        out_row["diagnostic_max_gt_iou"] = iou
                    else:
                        out_row["uses_gt_for_diagnostic_iou"] = False
                        out_row["diagnostic_max_gt_iou"] = None
                    matching_out_rows.append(out_row)
                proposal_rows_out.extend(matching_out_rows)
                d2 = geometry_contract_audit(meta, d1, matching_out_rows)
                d3 = proposal_survival_audit(
                    matching_out_rows,
                    score_bins=int(score_bins),
                )
                sample_rows_out.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "case_name": case_name,
                        "sample_id": sample_id,
                        "video_id": str(meta.get("video_name") or meta.get("video_id") or sample_id),
                        "batch_idx": int(batch_idx),
                        "sample_idx": int(sample_idx),
                        "epoch": epoch,
                        "diagnostic_only": True,
                        "uses_validation_gt_for_diagnostic_iou_only": bool(gt_segments),
                        "metric_claim_allowed": False,
                        "paper_claim_allowed": False,
                        "D1_reader_acquisition_utility": d1,
                        "D2_temporal_geometry_contract": d2,
                        "D3_proposal_survival_light": d3,
                    }
                )
            total_samples += batch_size
    finally:
        rpn_handle.remove()
        reader_handle.remove()

    write_jsonl(proposal_jsonl, proposal_rows_out)
    write_jsonl(sample_jsonl, sample_rows_out)
    all_d1 = [row["D1_reader_acquisition_utility"] for row in sample_rows_out]
    all_d2 = [row["D2_temporal_geometry_contract"] for row in sample_rows_out]
    all_d3 = [row["D3_proposal_survival_light"] for row in sample_rows_out]
    failure_flags: dict[str, int] = {}
    for d2 in all_d2:
        for flag in d2.get("failure_flags", []):
            failure_flags[flag] = failure_flags.get(flag, 0) + 1
    verdict = classify_summary_verdict(all_d1=all_d1, all_d2=all_d2, all_d3=all_d3)
    video_ids = sorted({str(row.get("video_id", "unknown")) for row in sample_rows_out})
    summary = {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_id": f"{case_name}_D1_D2_D3",
        "decision": READY,
        "case_name": case_name,
        "config": str(config),
        "checkpoint": str(checkpoint),
        "split": split,
        "limit_batches": int(limit_batches),
        "epoch": epoch,
        "status": "complete",
        "has_gt": bool(verdict["has_gt"]),
        "has_meta": bool(verdict["has_meta"]),
        "uses_validation_gt_for_diagnostic_iou_only": bool(verdict["has_gt"]),
        "metric_summary": {},
        "num_videos": len(video_ids),
        "num_valid_videos": len(video_ids),
        "num_failed_videos": 0,
        "sample_count": len(sample_rows_out),
        "proposal_row_count": len(proposal_rows_out),
        "sample_diagnostics_jsonl": str(sample_jsonl),
        "proposal_factors_jsonl": str(proposal_jsonl),
        "baseline_refs": {
            "random_fixed_adapter": 63.77,
            "strict_ema": 63.85,
            "stratified_sampling_best": 64.64,
            "uniform_stride2_50pct": 65.09,
            "exact_uniform_family": [65.57, 65.73],
            "residual64_fallback": 65.46,
            "oracle_residual64": 66.61,
            "oracle_boundary_adapter": 77.62,
        },
        "verdict": verdict["verdict"],
        "primary_failure_mode": verdict["primary_failure_mode"],
        "notes": verdict["notes"],
        "D1_reader_acquisition_utility": {
            "selected_count": stats([d1.get("selected_count") for d1 in all_d1]),
            "valid_len": stats([d1.get("valid_len") for d1 in all_d1]),
            "selected_density": stats([d1.get("selected_density") for d1 in all_d1]),
            "duplicate_slot_count": stats([d1.get("duplicate_slot_count") for d1 in all_d1]),
            "gt_touched_fraction": stats([
                d1.get("gt_coverage", {}).get("gt_touched_fraction")
                for d1 in all_d1
            ]),
            "boundary_touched_fraction": stats([
                d1.get("gt_coverage", {}).get("boundary_touched_fraction")
                for d1 in all_d1
            ]),
        },
        "D2_temporal_geometry_contract": {
            "failure_flag_counts": failure_flags,
            "proposal_outside_valid_len_count": sum(int(d2.get("proposal_outside_valid_len_count", 0)) for d2 in all_d2),
            "proposal_non_positive_duration_count": sum(int(d2.get("proposal_non_positive_duration_count", 0)) for d2 in all_d2),
            "mean_abs_reader_meta_selected_position_diff": stats([
                d2.get("mean_abs_reader_meta_selected_position_diff")
                for d2 in all_d2
            ]),
        },
        "D3_proposal_survival_light": {
            "candidate_count": stats([d3.get("candidate_count") for d3 in all_d3]),
            "score_iou_pearson": stats([d3.get("score_iou_pearson") for d3 in all_d3]),
            "diagnostic_max_gt_iou": stats([
                item.get("diagnostic_max_gt_iou")
                for item in proposal_rows_out
            ]),
            "final_score": stats([item.get("final_score") for item in proposal_rows_out]),
        },
        "claim_boundaries": {
            "diagnostic_only": True,
            "tools_train": False,
            "tools_test": False,
            "detector_map": False,
            "raw_prediction_cache": False,
            "metric_claim_allowed": False,
            "paper_claim_allowed": False,
            "runtime_flops_claim_allowed": False,
            "deploy_claim_allowed": False,
        },
    }
    write_json(summary_json, summary)
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
    parser = argparse.ArgumentParser(description="Run D1-D3 PC-OT-MRAS R17/R18 mechanism diagnostics.")
    parser.add_argument("--case-name", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--limit-batches", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--use-ema", type=_parse_use_ema, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--budget", type=int, default=384)
    parser.add_argument("--topk-per-sample", type=int, default=1000)
    parser.add_argument("--score-bins", type=int, default=10)
    args = parser.parse_args(argv)

    try:
        summary = run_case(
            case_name=args.case_name,
            config=args.config,
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            split=args.split,
            limit_batches=int(args.limit_batches),
            device=args.device,
            use_ema=args.use_ema,
            use_amp=bool(args.amp),
            budget=int(args.budget),
            topk_per_sample=int(args.topk_per_sample),
            score_bins=int(args.score_bins),
        )
    except Exception as exc:  # pragma: no cover - CLI guard
        print(json.dumps(strict_json_value(error_payload(exc)), sort_keys=True))
        return 1

    print(json.dumps(strict_json_value(summary), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
