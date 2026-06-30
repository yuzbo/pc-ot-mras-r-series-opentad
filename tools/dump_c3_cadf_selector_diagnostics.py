import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmengine.config import Config, DictAction


ROUTE_LABELS = ["C3_MAINLINE_OPTIMIZATION", "C3_ORIGINAL_OPTIMIZATION_ROUTE"]
FORBIDDEN_CONFIG_TOKENS = [
    "raw_prediction_cache",
    "teacher",
    "test_gt_decision",
    "validation_gt_decision",
    "p2",
    "pqr",
    "pqr_ranking",
    "bh_sdc",
    "bhsdc",
    "divergent_innovation",
    "divergent",
]


def _to_float(value, default=0.0):
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def _to_int(value, default=0):
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_plain_list(value):
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def _segments_to_list(gt_segments):
    if gt_segments is None:
        return []
    rows = _as_plain_list(gt_segments)
    clean = []
    for row in rows:
        if row is None or len(row) < 2:
            continue
        start = _to_float(row[0])
        end = _to_float(row[1])
        if end > start:
            clean.append([start, end])
    return clean


def _selected_coverage_intervals(selected, valid_len):
    selected = sorted(set(int(x) for x in selected if int(x) >= 0 and int(x) < max(valid_len, 0)))
    if not selected or valid_len <= 0:
        return []
    intervals = []
    for idx, pos in enumerate(selected):
        if idx == 0:
            left = 0.0 if len(selected) == 1 else max(0.0, pos - (selected[idx + 1] - pos) * 0.5)
        else:
            left = (selected[idx - 1] + pos) * 0.5
        if idx == len(selected) - 1:
            right = float(valid_len) if len(selected) == 1 else min(float(valid_len), pos + (pos - selected[idx - 1]) * 0.5)
        else:
            right = (pos + selected[idx + 1]) * 0.5
        if right > left:
            intervals.append((left, right))
    return intervals


def _covered_length(segment, intervals):
    start, end = segment
    total = 0.0
    for left, right in intervals:
        overlap = min(float(end), right) - max(float(start), left)
        if overlap > 0:
            total += overlap
    return total


def compute_gt_diagnostics(meta, selected_dense_indices, gt_segments=None, boundary_radius=2):
    """Offline-only GT statistics; never feeds back into selector decisions."""
    selected = [int(x) for x in _as_plain_list(selected_dense_indices)]
    valid_len = _to_int(meta.get("c3_indirect_valid_len", meta.get("window_size", 0)), default=0)
    valid_len = max(valid_len, 0)
    segments = _segments_to_list(gt_segments)
    intervals = _selected_coverage_intervals(selected, valid_len)

    kept_count = 0
    original_length = 0.0
    kept_length = 0.0
    for segment in segments:
        start = max(0.0, min(float(valid_len), float(segment[0])))
        end = max(0.0, min(float(valid_len), float(segment[1])))
        if end <= start:
            continue
        length = end - start
        covered = _covered_length((start, end), intervals)
        original_length += length
        kept_length += covered
        if covered > 0.0:
            kept_count += 1

    boundary_hits = 0
    selected_boundary_near = 0
    boundary_count = 0
    if selected and segments:
        selected_set = sorted(set(x for x in selected if 0 <= x < valid_len))
        boundaries = []
        for start, end in segments:
            for boundary in (start, end):
                if 0.0 <= boundary <= float(valid_len):
                    boundaries.append(float(boundary))
        boundary_count = len(boundaries)
        selected_boundary_near = sum(
            1
            for pos in selected_set
            if any(abs(float(pos) - boundary) <= float(boundary_radius) for boundary in boundaries)
        )
        covered_boundaries = []
        for boundary in boundaries:
            hit = any(abs(float(pos) - boundary) <= float(boundary_radius) for pos in selected_set)
            covered_boundaries.append(bool(hit))
        boundary_hits = sum(1 for hit in covered_boundaries if hit)

    selected_count = len(selected)
    return {
        "gt_diagnostic_only": True,
        "gt_remap_kept_count": kept_count,
        "gt_remap_total_count": len(segments),
        "gt_remap_kept_fraction": kept_count / float(len(segments)) if segments else 0.0,
        "gt_remap_original_length_sum": original_length,
        "gt_remap_kept_length_sum": kept_length,
        "gt_remap_length_ratio": kept_length / original_length if original_length > 0.0 else 0.0,
        "boundary_count": boundary_count,
        "boundary_recall_count": boundary_hits,
        "boundary_recall_rate": boundary_hits / float(boundary_count) if boundary_count > 0 else 0.0,
        "boundary_near_count": selected_boundary_near,
        "boundary_near_rate": selected_boundary_near / float(selected_count) if selected_count > 0 else 0.0,
        "boundary_radius": int(boundary_radius),
    }


def build_record_from_meta(meta, gt_segments=None, include_selected_indices=False, boundary_radius=2):
    selected = [int(x) for x in _as_plain_list(meta.get("c3_indirect_selected_dense_indices", []))]
    selected_mask = meta.get("c3_indirect_selected_mask", None)
    if selected_mask is not None:
        mask = [bool(x) for x in _as_plain_list(selected_mask)]
        valid_selected = [idx for idx, keep in zip(selected, mask) if keep]
    else:
        valid_len = _to_int(meta.get("c3_indirect_valid_len", meta.get("window_size", 0)), default=0)
        valid_selected = [idx for idx in selected if idx < valid_len] if valid_len > 0 else list(selected)

    record = {
        "diagnostic_only": True,
        "official_map_claim": False,
        "video_name": meta.get("video_name", ""),
        "valid_len": _to_int(meta.get("c3_indirect_valid_len", meta.get("window_size", 0))),
        "selected_count": _to_int(meta.get("c3_indirect_selected_valid_len", len(valid_selected)), len(valid_selected)),
        "selected_unique_count": len(set(valid_selected)),
        "selected_min": min(valid_selected) if valid_selected else None,
        "selected_max": max(valid_selected) if valid_selected else None,
        "selected_first_8": valid_selected[:8],
        "selected_last_8": valid_selected[-8:],
        "max_gap": _to_int(meta.get("c3_density_mesh_max_gap", 0)),
        "mean_gap": _to_float(meta.get("c3_density_mesh_mean_gap", 0.0)),
        "duplicate_count": _to_int(meta.get("c3_density_mesh_duplicate_count", 0)),
        "repair_count": _to_int(meta.get("c3_density_mesh_repair_count", 0)),
        "repair_fraction": _to_float(meta.get("c3_density_mesh_repair_fraction", 0.0)),
        "row_repair_fraction": _to_float(meta.get("c3_density_mesh_row_repair_fraction", 0.0)),
        "dedupe_repair_count": _to_int(meta.get("c3_density_mesh_dedupe_repair_count", 0)),
        "gap_guard_add_count": _to_int(meta.get("c3_density_mesh_gap_guard_add_count", 0)),
        "gap_guard_prune_count": _to_int(meta.get("c3_density_mesh_gap_guard_prune_count", 0)),
        "pad_repair_count": _to_int(meta.get("c3_density_mesh_pad_repair_count", 0)),
        "density_entropy": _to_float(meta.get("c3_density_mesh_entropy", 0.0)),
        "selected_density_sum": _to_float(meta.get("c3_density_mesh_selected_density_sum", 0.0)),
        "density_top_positions": [int(x) for x in _as_plain_list(meta.get("c3_density_mesh_density_top_positions", []))],
        "density_histogram_8": [
            _to_float(x) for x in _as_plain_list(meta.get("c3_density_mesh_density_histogram_8", []))
        ],
        "uncertainty_selected_fraction": _to_float(
            meta.get("c3_density_mesh_uncertainty_selected_fraction", 0.0)
        ),
        "change_selected_fraction": _to_float(meta.get("c3_density_mesh_change_selected_fraction", 0.0)),
        "distribution_target_selected_fraction": _to_float(
            meta.get("c3_density_mesh_distribution_target_selected_fraction", 0.0)
        ),
        "distribution_target_top_positions": [
            int(x) for x in _as_plain_list(meta.get("c3_density_mesh_distribution_target_top_positions", []))
        ],
    }
    if include_selected_indices:
        record["selected_dense_indices"] = selected
    else:
        record["selected_dense_indices_summary"] = {
            "first_16": valid_selected[:16],
            "last_16": valid_selected[-16:],
        }
    record.update(compute_gt_diagnostics(meta, valid_selected, gt_segments, boundary_radius=boundary_radius))
    return record


def _mean(records, key):
    values = [_to_float(row.get(key), None) for row in records if row.get(key) is not None]
    values = [value for value in values if value is not None and math.isfinite(value)]
    return sum(values) / float(len(values)) if values else 0.0


def _percentile(values, q):
    values = sorted(_to_float(value) for value in values if value is not None and math.isfinite(_to_float(value)))
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * float(q)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return values[low]
    frac = position - low
    return values[low] * (1.0 - frac) + values[high] * frac


def aggregate_summary(records, config_path, checkpoint_path, split, warnings=None):
    warnings = list(warnings or [])
    max_gaps = [row.get("max_gap", 0) for row in records]
    status = "ok" if records else "empty"
    return {
        "status": status,
        "diagnostic_only": True,
        "official_map_claim": False,
        "route_labels": list(ROUTE_LABELS),
        "tool": "dump_c3_cadf_selector_diagnostics",
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "split": split,
        "record_count": len(records),
        "max_gap": {
            "mean": _mean(records, "max_gap"),
            "p50": _percentile(max_gaps, 0.50),
            "p90": _percentile(max_gaps, 0.90),
            "p95": _percentile(max_gaps, 0.95),
        },
        "mean_gap_mean": _mean(records, "mean_gap"),
        "duplicate_count_mean": _mean(records, "duplicate_count"),
        "repair_count_mean": _mean(records, "repair_count"),
        "repair_fraction_mean": _mean(records, "repair_fraction"),
        "dedupe_repair_count_mean": _mean(records, "dedupe_repair_count"),
        "gap_guard_add_count_mean": _mean(records, "gap_guard_add_count"),
        "gap_guard_prune_count_mean": _mean(records, "gap_guard_prune_count"),
        "pad_repair_count_mean": _mean(records, "pad_repair_count"),
        "density_entropy_mean": _mean(records, "density_entropy"),
        "selected_density_sum_mean": _mean(records, "selected_density_sum"),
        "uncertainty_selected_fraction_mean": _mean(records, "uncertainty_selected_fraction"),
        "change_selected_fraction_mean": _mean(records, "change_selected_fraction"),
        "distribution_target_selected_fraction_mean": _mean(records, "distribution_target_selected_fraction"),
        "gt_remap_kept_fraction_mean": _mean(records, "gt_remap_kept_fraction"),
        "gt_remap_length_ratio_mean": _mean(records, "gt_remap_length_ratio"),
        "boundary_near_rate_mean": _mean(records, "boundary_near_rate"),
        "warnings": warnings,
    }


def enforce_c3_cadf_selector_dump_config(cfg):
    cfg_text = cfg.pretty_text.lower()
    found = [token for token in FORBIDDEN_CONFIG_TOKENS if token in cfg_text]
    if found:
        raise AssertionError(f"Forbidden route/cache/teacher/test-GT tokens in selector dump config: {found}")
    if bool(cfg.inference.get("load_from_raw_predictions", False)) or bool(cfg.inference.get("save_raw_prediction", False)):
        raise AssertionError("selector dump refuses raw prediction cache load/save flags")
    if cfg.model.get("type", None) != "ActionFormer":
        raise AssertionError("selector dump requires ActionFormer")
    selector = cfg.model.get("frame_selector", None)
    if selector is None:
        raise AssertionError("selector dump requires model.frame_selector")
    if selector.get("type", None) != "PCOTMRASIndirectPreBackboneFrameSelector":
        raise AssertionError("selector dump requires the C3 indirect selector")
    if selector.get("strategy", None) != "cadf_density_mesh_st":
        raise AssertionError("selector dump requires CADF/loss-select V2 strategy")
    scout = selector.get("scout", {})
    if scout.get("type", None) != "PCOTMRASCADFDensityFrameScout":
        raise AssertionError("selector dump requires CADF/loss-select V2 scout")
    if cfg.get("c3_loss_select_v2", None) is not True:
        raise AssertionError("selector dump requires CADF/loss-select V2 config marker")
    if cfg.get("c3_route_label", None) != "C3_MAINLINE_OPTIMIZATION":
        raise AssertionError("selector dump refuses non-C3_MAINLINE_OPTIMIZATION route label")
    if "C3_ORIGINAL_OPTIMIZATION_ROUTE" not in cfg.get("c3_route_labels", []):
        raise AssertionError("selector dump requires C3_ORIGINAL_OPTIMIZATION_ROUTE label")
    if cfg.get("c3_loss_select_v2_test_aux_source_leakage", None) != "forbidden":
        raise AssertionError("selector dump requires test-time auxiliary-source leakage to be forbidden")
    if selector.get("fast_cpu_selection", None) is not True:
        raise AssertionError("selector dump requires fast_cpu_selection=True")

    selector.emit_selection_diagnostics = True
    selector.selection_diagnostics_interval = 1
    cfg.diagnostic_only = True
    cfg.official_map_claim = False
    return cfg


def _record_fieldnames(records):
    keys = set()
    for row in records:
        keys.update(row.keys())
    preferred = [
        "video_name",
        "valid_len",
        "selected_count",
        "selected_unique_count",
        "max_gap",
        "mean_gap",
        "duplicate_count",
        "repair_count",
        "repair_fraction",
        "dedupe_repair_count",
        "gap_guard_add_count",
        "gap_guard_prune_count",
        "pad_repair_count",
        "density_entropy",
        "selected_density_sum",
        "uncertainty_selected_fraction",
        "change_selected_fraction",
        "distribution_target_selected_fraction",
        "gt_remap_kept_count",
        "gt_remap_total_count",
        "gt_remap_kept_fraction",
        "gt_remap_length_ratio",
        "boundary_near_rate",
        "diagnostic_only",
        "official_map_claim",
    ]
    return preferred + sorted(keys.difference(preferred))


def write_outputs(summary, records, summary_path, jsonl_path=None, csv_path=None):
    summary_path = Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary["diagnostic_only"] = True
    summary["official_map_claim"] = False
    for row in records:
        row.setdefault("diagnostic_only", True)
        row.setdefault("official_map_claim", False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if jsonl_path is not None:
        jsonl_path = Path(jsonl_path)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("w", encoding="utf-8") as f:
            for row in records:
                f.write(json.dumps(row, sort_keys=True) + "\n")

    if csv_path is not None:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = _record_fieldnames(records)
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in records:
                clean = {}
                for key in fieldnames:
                    value = row.get(key)
                    clean[key] = json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
                writer.writerow(clean)


def _get_split_cfg(cfg, split):
    if split == "train":
        return cfg.dataset.train
    if split == "val":
        return cfg.dataset.val
    if split == "test":
        return cfg.dataset.test
    raise ValueError(f"Unsupported split: {split}")


def _load_checkpoint_state_dict(checkpoint_path, device, use_ema=False):
    import torch

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if use_ema and isinstance(checkpoint, dict) and "state_dict_ema" in checkpoint:
        return checkpoint["state_dict_ema"], checkpoint
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"], checkpoint
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"], checkpoint
    return checkpoint, checkpoint


def _load_state_dict_flex(model, state_dict):
    if not isinstance(state_dict, dict):
        raise AssertionError("checkpoint does not contain a state_dict-like mapping")
    try:
        return model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        stripped = {}
        for key, value in state_dict.items():
            stripped[key[7:] if key.startswith("module.") else key] = value
        return model.load_state_dict(stripped, strict=True)


def _move_batch_to_device(data_dict, device):
    moved = {}
    for key, value in data_dict.items():
        if hasattr(value, "to") and key in {"inputs", "masks"}:
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def _run_selector_dump(args):
    import torch

    from opentad.datasets import build_dataloader, build_dataset
    from opentad.models import build_detector
    from opentad.utils import set_seed

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    enforce_c3_cadf_selector_dump_config(cfg)
    set_seed(args.seed)

    device = torch.device(args.device)
    dataset = build_dataset(_get_split_cfg(cfg, args.split), default_args=dict(logger=None))
    loader = build_dataloader(
        dataset,
        rank=0,
        world_size=1,
        shuffle=False,
        drop_last=False,
        **cfg.solver.get(args.split if args.split in {"train", "val"} else "test"),
    )

    model = build_detector(cfg.model).to(device)
    state_dict, checkpoint = _load_checkpoint_state_dict(args.checkpoint, device, use_ema=args.use_ema)
    _load_state_dict_flex(model, state_dict)
    model.eval()
    if not getattr(model, "with_frame_selector", False):
        raise AssertionError("built model does not expose frame_selector")
    selector = model.frame_selector
    selector.emit_selection_diagnostics = True
    selector.selection_diagnostics_interval = 1

    records = []
    warnings = ["diagnostic_only=true; official_map_claim=false; no tools/test.py or official evaluator invoked"]
    if args.split in {"val", "test", "train"}:
        warnings.append("GT annotation, when present, is used only for offline diagnostic statistics")
    checkpoint_epoch = checkpoint.get("epoch", None) if isinstance(checkpoint, dict) else None
    if checkpoint_epoch is not None:
        warnings.append(f"checkpoint_epoch={checkpoint_epoch}")

    with torch.no_grad():
        for batch_idx, data_dict in enumerate(loader):
            if args.limit_batches is not None and batch_idx >= args.limit_batches:
                break
            data_dict = _move_batch_to_device(data_dict, device)
            outputs = selector.forward_test(data_dict["inputs"], data_dict["masks"], data_dict.get("metas", []))
            metas = outputs["metas"]
            gt_rows = data_dict.get("gt_segments", [None] * len(metas))
            for meta, gt_segments in zip(metas, gt_rows):
                records.append(
                    build_record_from_meta(
                        meta,
                        gt_segments=gt_segments,
                        include_selected_indices=args.include_selected_indices,
                        boundary_radius=args.boundary_radius,
                    )
                )
                if args.limit_records is not None and len(records) >= args.limit_records:
                    break
            if args.limit_records is not None and len(records) >= args.limit_records:
                break

    summary = aggregate_summary(records, args.config, args.checkpoint, args.split, warnings=warnings)
    write_outputs(summary, records, args.summary_json, args.records_jsonl, args.records_csv)
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnostic-only CADF loss-select V2 selector dump. Does not run official mAP/evaluator."
    )
    parser.add_argument("config", help="CADF loss-select V2 config")
    parser.add_argument("checkpoint", help="checkpoint to inspect")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--records-jsonl", default=None)
    parser.add_argument("--records-csv", default=None)
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("--limit-records", type=int, default=None)
    parser.add_argument("--boundary-radius", type=int, default=2)
    parser.add_argument("--include-selected-indices", action="store_true")
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, help="override settings")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))
    _run_selector_dump(args)


if __name__ == "__main__":
    main()
