import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.acquisition.rba_rbr.adapter_bridge import build_detector_feature_centers_from_raw  # noqa: E402
from opentad.acquisition.rba_rbr.types import ROUTE_LABEL  # noqa: E402
from opentad.acquisition.rba_rbr.validators import validate_route_identity  # noqa: E402


CLAIM_STATUS = "rba_rbr_non_gpu_coordinate_budget_audit_only_no_metric_runtime_deploy_or_paper_claim"
PRO_GATE_LOCK = "valid_gpt_5_5_pro_severe_result_diagnosis_required_before_long_followup_or_route_conclusion"


def _read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {lineno}: {exc}") from exc
    if not rows:
        raise ValueError(f"no audit rows found in {path}")
    return rows


def synthetic_grid_rows(mode="low_density"):
    if mode == "healthy_50pct":
        counts = [(192, 96, 8, 8), (192, 96, 8, 8), (188, 94, 9, 9)]
    elif mode == "low_density":
        counts = [(84, 42, 16, 24), (78, 39, 16, 24), (91, 46, 15, 23)]
    else:
        raise ValueError(f"unsupported synthetic mode: {mode}")
    rows = []
    for idx, (raw_k, detector_k, raw_gap, detector_gap) in enumerate(counts):
        rows.append(
            {
                "audit_type": "rba_rbr_detector_temporal_grid",
                "route_label": ROUTE_LABEL,
                "video_name": f"synthetic_{mode}_{idx}",
                "native_axis": True,
                "status": "PASS_RBA_RBR_NATIVE_AXIS_POSITIONS_ENTERED_MODEL",
                "raw_valid_k": raw_k,
                "mask_true_count": detector_k,
                "detector_feature_valid_k": detector_k,
                "detector_mask_true_count": detector_k,
                "selected_max_gap_after_guard": raw_gap,
                "max_detector_gap_after_guard": detector_gap,
                "dense_T": 384,
                "feature_stride": 2,
            }
        )
    return rows


def _first_number(row, keys):
    for key in keys:
        if key in row and row[key] is not None:
            value = row[key]
            if isinstance(value, bool):
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _stats(values):
    values = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    if not values:
        return {"count": 0, "min": None, "p05": None, "p50": None, "p95": None, "max": None, "mean": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def _require_rba_route(rows):
    for idx, row in enumerate(rows):
        try:
            validate_route_identity(row)
        except ValueError as exc:
            raise ValueError(f"row {idx} failed RBA-RBR route isolation: {exc}") from exc
        if row.get("route_label") != ROUTE_LABEL:
            raise ValueError(f"row {idx} missing exact route_label {ROUTE_LABEL}")
    return True


def summarize_coordinate_budget_rows(
    rows,
    reference_dense_t=384,
    reference_raw_k=192,
    feature_stride=2,
    far_below_ratio=0.60,
):
    rows = list(rows)
    if not rows:
        raise ValueError("RBA-RBR coordinate/budget audit requires at least one row")
    _require_rba_route(rows)

    raw_valid = []
    detector_valid = []
    raw_gap = []
    detector_gap = []
    statuses = {}
    native_axis_false = 0
    for row in rows:
        raw_valid.append(_first_number(row, ("raw_valid_k", "valid_k")))
        detector_valid.append(
            _first_number(
                row,
                (
                    "detector_mask_true_count",
                    "mask_true_count",
                    "detector_feature_valid_k",
                    "meta_detector_feature_position_count",
                ),
            )
        )
        raw_gap.append(
            _first_number(
                row,
                ("selected_max_gap_after_guard", "max_raw_gap_after_guard", "selected_max_gap", "selected_max_gap_before_guard"),
            )
        )
        detector_gap.append(
            _first_number(
                row,
                ("max_detector_gap_after_guard", "detector_max_gap_after_guard", "max_detector_gap_before_guard"),
            )
        )
        status = str(row.get("status", "UNKNOWN"))
        statuses[status] = int(statuses.get(status, 0)) + 1
        if row.get("native_axis") is False:
            native_axis_false += 1

    raw_stats = _stats(raw_valid)
    detector_stats = _stats(detector_valid)
    raw_gap_stats = _stats(raw_gap)
    detector_gap_stats = _stats(detector_gap)
    reference_detector_k = int(math.ceil(float(reference_raw_k) / float(max(int(feature_stride), 1))))
    detector_mean = detector_stats["mean"]
    raw_mean = raw_stats["mean"]
    detector_ratio = None if detector_mean is None else float(detector_mean) / float(reference_detector_k)
    raw_ratio = None if raw_mean is None else float(raw_mean) / float(reference_raw_k)
    far_below = bool(detector_ratio is not None and detector_ratio < float(far_below_ratio))

    return {
        "route_label": ROUTE_LABEL,
        "audit_scope": "rba_rbr_non_gpu_coordinate_budget_audit",
        "row_count": int(len(rows)),
        "status_counts": statuses,
        "native_axis_false_count": int(native_axis_false),
        "reference": {
            "dense_T": int(reference_dense_t),
            "raw_50pct_k": int(reference_raw_k),
            "feature_stride": int(feature_stride),
            "detector_50pct_k": int(reference_detector_k),
            "far_below_ratio_threshold": float(far_below_ratio),
        },
        "raw_valid_k": raw_stats,
        "detector_valid_count": detector_stats,
        "raw_gap_after_guard": raw_gap_stats,
        "detector_gap_after_guard": detector_gap_stats,
        "raw_density_vs_50pct_reference": raw_ratio,
        "effective_detector_density_vs_50pct_reference": detector_ratio,
        "effective_detector_density_far_below_50pct_reference": far_below,
        "diagnosis_hint": (
            "detector-side effective density is far below the fixed 50pct reference; "
            "use this as non-GPU evidence for postprocess-guard shortdiag, matched low-budget uniform, "
            "K=192 forced coverage, and coordinate-closure follow-up planning"
            if far_below
            else "detector-side effective density is not far below the fixed 50pct reference in this audit"
        ),
    }


def coordinate_closure_synthetic_check(feature_stride=2):
    raw_positions = np.asarray([0, 1, 15, 16, 31, 32, 47, 48, 63, 64, 95, 96, 111, 112, 127], dtype=np.int64)
    dense_t = 128
    synthetic_segments = [(16.0, 32.0), (96.0, 112.0)]
    detector_positions = build_detector_feature_centers_from_raw(raw_positions, feature_stride=feature_stride)
    expected = np.asarray(
        [float(np.mean(raw_positions[start : start + feature_stride])) for start in range(0, len(raw_positions), feature_stride)],
        dtype=np.float32,
    )
    centers_match_raw_groups = bool(np.allclose(detector_positions, expected))
    boundary_distances = []
    segment_hits = []
    for left, right in synthetic_segments:
        left_dist = float(np.min(np.abs(raw_positions.astype(np.float32) - float(left))))
        right_dist = float(np.min(np.abs(raw_positions.astype(np.float32) - float(right))))
        boundary_distances.extend([left_dist, right_dist])
        raw_inside = raw_positions[(raw_positions >= left) & (raw_positions <= right)]
        detector_inside = detector_positions[(detector_positions >= left) & (detector_positions <= right)]
        segment_hits.append(
            {
                "segment": [float(left), float(right)],
                "raw_hit_count": int(raw_inside.size),
                "detector_center_hit_count": int(detector_inside.size),
                "nearest_left_raw_distance": left_dist,
                "nearest_right_raw_distance": right_dist,
            }
        )
    max_boundary_distance = float(max(boundary_distances))
    native_axis_not_selected_index = bool(float(np.max(detector_positions)) > float(len(detector_positions) - 1))
    passed = bool(
        centers_match_raw_groups
        and max_boundary_distance <= 1.0
        and native_axis_not_selected_index
        and all(row["raw_hit_count"] > 0 and row["detector_center_hit_count"] > 0 for row in segment_hits)
    )
    return {
        "route_label": ROUTE_LABEL,
        "check_name": "rba_rbr_coordinate_closure_synthetic_native_axis",
        "passed": passed,
        "uses_validation_or_test_gt": False,
        "segment_source": "synthetic_native_axis_only_not_dataset_gt",
        "dense_T": int(dense_t),
        "feature_stride": int(feature_stride),
        "raw_selected_positions": [int(pos) for pos in raw_positions.tolist()],
        "detector_feature_positions": [float(pos) for pos in detector_positions.tolist()],
        "centers_match_raw_groups": centers_match_raw_groups,
        "native_axis_not_selected_index": native_axis_not_selected_index,
        "max_synthetic_boundary_raw_distance": max_boundary_distance,
        "segment_hits": segment_hits,
    }


def build_claim_lock():
    return {
        "route_label": ROUTE_LABEL,
        "claim_status": CLAIM_STATUS,
        "required_gate": PRO_GATE_LOCK,
        "no_gpu": True,
        "no_training": True,
        "no_metric_claim": True,
        "no_runtime_claim": True,
        "no_deploy_claim": True,
        "no_paper_claim": True,
        "full_train_unlocked": False,
        "pro_gate_waived": False,
    }


def run_audit(rows, reference_dense_t=384, reference_raw_k=192, feature_stride=2, far_below_ratio=0.60):
    summary = summarize_coordinate_budget_rows(
        rows,
        reference_dense_t=reference_dense_t,
        reference_raw_k=reference_raw_k,
        feature_stride=feature_stride,
        far_below_ratio=far_below_ratio,
    )
    closure = coordinate_closure_synthetic_check(feature_stride=feature_stride)
    return {
        "route_label": ROUTE_LABEL,
        "summary": summary,
        "coordinate_closure": closure,
        "claim_lock": build_claim_lock(),
    }


def main():
    parser = argparse.ArgumentParser(description="CPU-only RBA-RBR coordinate and budget audit.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--grid-audit-jsonl", help="Path to RBA-RBR detector grid audit JSONL.")
    source.add_argument(
        "--synthetic",
        choices=("low_density", "healthy_50pct"),
        help="Use built-in synthetic audit rows instead of reading JSONL.",
    )
    parser.add_argument("--reference-dense-t", type=int, default=384)
    parser.add_argument("--reference-raw-k", type=int, default=192)
    parser.add_argument("--feature-stride", type=int, default=2)
    parser.add_argument("--far-below-ratio", type=float, default=0.60)
    parser.add_argument("--out", help="Optional JSON output path.")
    args = parser.parse_args()

    rows = _read_jsonl(args.grid_audit_jsonl) if args.grid_audit_jsonl else synthetic_grid_rows(args.synthetic)
    result = run_audit(
        rows,
        reference_dense_t=args.reference_dense_t,
        reference_raw_k=args.reference_raw_k,
        feature_stride=args.feature_stride,
        far_below_ratio=args.far_below_ratio,
    )
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
