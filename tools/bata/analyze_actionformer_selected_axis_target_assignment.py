from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.analyze_actionformer_selected_axis_point_geometry import (  # noqa: E402
    iter_snapshot_samples,
    strict_json_value,
    write_json,
)


SCHEMA_VERSION = "actionformer_selected_axis_target_assignment_audit_v0"
READY = "ACTIONFORMER_SELECTED_AXIS_TARGET_ASSIGNMENT_AUDIT_READY"
NO_GO = "ACTIONFORMER_SELECTED_AXIS_TARGET_ASSIGNMENT_AUDIT_NO_GO"
DEFAULT_STRIDES = (1, 2, 4, 8, 16, 32)
DEFAULT_REGRESSION_RANGES = ((0.0, 4.0), (4.0, 8.0), (8.0, 16.0), (16.0, 32.0), (32.0, 64.0), (64.0, 10000.0))


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _stats(values: Sequence[float]) -> dict[str, Any]:
    cleaned = [float(item) for item in values if math.isfinite(float(item))]
    if not cleaned:
        return {"count": 0}
    ordered = sorted(cleaned)
    mean = sum(ordered) / len(ordered)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": ordered[len(ordered) // 2] if len(ordered) % 2 == 1 else 0.5 * (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]),
        "max": ordered[-1],
        "mean": mean,
    }


def _group_mean_positions(positions: Sequence[float], stride: int) -> list[float]:
    grouped: list[float] = []
    for start in range(0, len(positions), int(stride)):
        chunk = [float(item) for item in positions[start : start + int(stride)]]
        if chunk:
            grouped.append(sum(chunk) / len(chunk))
    return grouped


def parse_regression_ranges(value: str) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for item in str(value).split(","):
        stripped = item.strip()
        if not stripped:
            continue
        if ":" not in stripped:
            raise ValueError(f"regression range must use min:max format: {stripped}")
        left, right = stripped.split(":", 1)
        left_value = _as_float(left)
        right_value = _as_float(right)
        if left_value is None or right_value is None or right_value < left_value:
            raise ValueError(f"invalid regression range: {stripped}")
        out.append((float(left_value), float(right_value)))
    if not out:
        raise ValueError("at least one regression range is required")
    return out


def parse_int_list(value: str) -> list[int]:
    out: list[int] = []
    for item in str(value).split(","):
        stripped = item.strip()
        if stripped:
            parsed = int(stripped)
            if parsed <= 0:
                raise ValueError("strides must be positive")
            out.append(parsed)
    if not out:
        raise ValueError("at least one stride is required")
    return out


def parse_path_list(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        for item in str(value).split(","):
            stripped = item.strip()
            if stripped:
                out.append(stripped)
    if not out:
        raise ValueError("at least one snapshot path is required")
    return out


def load_annotation(path: str | Path, *, split: str = "validation") -> dict[str, dict[str, Any]]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    database = payload.get("database", payload) if isinstance(payload, Mapping) else None
    if not isinstance(database, Mapping):
        raise ValueError("annotation file must contain a database object")
    out: dict[str, dict[str, Any]] = {}
    for video_id, item in database.items():
        if not isinstance(item, Mapping):
            continue
        subset = str(item.get("subset", ""))
        if split and subset and subset.lower() != split.lower():
            continue
        duration = _as_float(item.get("duration"))
        annotations = item.get("annotations", [])
        if duration is None or duration <= 0 or not _is_sequence(annotations):
            continue
        segments: list[dict[str, Any]] = []
        for idx, anno in enumerate(annotations):
            if not isinstance(anno, Mapping):
                continue
            segment = anno.get("segment")
            if not _is_sequence(segment) or len(segment) < 2:
                continue
            start = _as_float(segment[0])
            end = _as_float(segment[1])
            if start is None or end is None or end <= start:
                continue
            segments.append({"index": idx, "segment_seconds": [float(start), float(end)], "label": anno.get("label")})
        out[str(video_id)] = {
            "duration": float(duration),
            "frame": item.get("frame"),
            "annotations": segments,
        }
    return out


def gt_segments_to_dense_positions(video_info: Mapping[str, Any], *, dense_length: int) -> list[dict[str, Any]]:
    duration = _as_float(video_info.get("duration"))
    if duration is None or duration <= 0 or dense_length <= 1:
        return []
    scale = float(dense_length - 1) / duration
    out: list[dict[str, Any]] = []
    for anno in video_info.get("annotations", []):
        if not isinstance(anno, Mapping):
            continue
        segment = anno.get("segment_seconds")
        if not _is_sequence(segment) or len(segment) < 2:
            continue
        start = _as_float(segment[0])
        end = _as_float(segment[1])
        if start is None or end is None or end <= start:
            continue
        dense_start = max(0.0, min(float(dense_length - 1), start * scale))
        dense_end = max(0.0, min(float(dense_length - 1), end * scale))
        if dense_end > dense_start:
            out.append(
                {
                    "index": int(anno.get("index", len(out))),
                    "segment_dense": [dense_start, dense_end],
                    "segment_seconds": [float(start), float(end)],
                    "label": anno.get("label"),
                }
            )
    return out


def build_axis_points(positions: Sequence[float], *, stride: int, mode: str) -> list[float]:
    physical = _group_mean_positions(positions, int(stride))
    if mode == "physical":
        return physical
    if mode == "fake":
        return [float(idx * int(stride)) for idx in range(len(physical))]
    raise ValueError(f"unknown point mode: {mode}")


def assign_points(
    points: Sequence[float],
    *,
    gt_segments: Sequence[Mapping[str, Any]],
    stride: int,
    regression_range: tuple[float, float],
    center_sample_radius: float,
) -> list[int | None]:
    assignments: list[int | None] = []
    range_min, range_max = regression_range
    for point in points:
        candidates: list[tuple[float, int]] = []
        for gt_idx, gt in enumerate(gt_segments):
            segment = gt.get("segment_dense")
            if not _is_sequence(segment) or len(segment) < 2:
                continue
            start = _as_float(segment[0])
            end = _as_float(segment[1])
            if start is None or end is None or end <= start:
                continue
            left = float(point) - start
            right = end - float(point)
            max_distance = max(left, right)
            center = 0.5 * (start + end)
            radius = float(stride) * float(center_sample_radius)
            center_min = max(center - radius, start)
            center_max = min(center + radius, end)
            inside_center = (float(point) - center_min) > 0.0 and (center_max - float(point)) > 0.0
            inside_range = max_distance >= range_min and max_distance <= range_max
            if inside_center and inside_range:
                candidates.append((end - start, gt_idx))
        if not candidates:
            assignments.append(None)
        else:
            assignments.append(min(candidates, key=lambda item: item[0])[1])
    return assignments


def summarize_level_assignment(
    *,
    sample_id: str,
    snapshot_path: str,
    snapshot_line: int,
    snapshot_id: str | None,
    positions: Sequence[float],
    dense_length: int,
    gt_segments: Sequence[Mapping[str, Any]],
    stride: int,
    regression_range: tuple[float, float],
    center_sample_radius: float,
) -> dict[str, Any]:
    fake_points = build_axis_points(positions, stride=int(stride), mode="fake")
    physical_points = build_axis_points(positions, stride=int(stride), mode="physical")
    fake_assign = assign_points(
        fake_points,
        gt_segments=gt_segments,
        stride=int(stride),
        regression_range=regression_range,
        center_sample_radius=float(center_sample_radius),
    )
    physical_assign = assign_points(
        physical_points,
        gt_segments=gt_segments,
        stride=int(stride),
        regression_range=regression_range,
        center_sample_radius=float(center_sample_radius),
    )
    fake_positive = sum(1 for item in fake_assign if item is not None)
    physical_positive = sum(1 for item in physical_assign if item is not None)
    fake_only = sum(1 for fake, physical in zip(fake_assign, physical_assign) if fake is not None and physical is None)
    physical_only = sum(1 for fake, physical in zip(fake_assign, physical_assign) if fake is None and physical is not None)
    both_positive = sum(1 for fake, physical in zip(fake_assign, physical_assign) if fake is not None and physical is not None)
    same_gt = sum(1 for fake, physical in zip(fake_assign, physical_assign) if fake is not None and fake == physical)
    fake_gt_covered = len({item for item in fake_assign if item is not None})
    physical_gt_covered = len({item for item in physical_assign if item is not None})
    return {
        "sample_id": sample_id,
        "snapshot_path": snapshot_path,
        "snapshot_line": int(snapshot_line),
        "snapshot_id": snapshot_id,
        "dense_length": int(dense_length),
        "gt_count": len(gt_segments),
        "nominal_stride": int(stride),
        "regression_range_min": float(regression_range[0]),
        "regression_range_max": float(regression_range[1]),
        "level_length": len(fake_points),
        "fake_positive": fake_positive,
        "physical_positive": physical_positive,
        "positive_delta_physical_minus_fake": physical_positive - fake_positive,
        "fake_only": fake_only,
        "physical_only": physical_only,
        "both_positive": both_positive,
        "same_gt_assignment": same_gt,
        "same_gt_ratio_among_both": None if both_positive == 0 else same_gt / float(both_positive),
        "fake_gt_covered": fake_gt_covered,
        "physical_gt_covered": physical_gt_covered,
        "gt_coverage_delta_physical_minus_fake": physical_gt_covered - fake_gt_covered,
    }


def run_selected_axis_target_assignment_audit(
    *,
    snapshot_jsonl: Sequence[str | Path],
    annotation: str | Path,
    output_dir: str | Path,
    split: str = "validation",
    strides: Sequence[int] = DEFAULT_STRIDES,
    regression_ranges: Sequence[tuple[float, float]] = DEFAULT_REGRESSION_RANGES,
    center_sample_radius: float = 1.5,
) -> dict[str, Any]:
    if len(strides) != len(regression_ranges):
        raise ValueError("strides and regression_ranges must have the same length")
    annotations = load_annotation(annotation, split=split)
    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    skipped_no_gt = 0
    skipped_no_dense_length = 0
    sample_count = 0

    for snapshot_path in snapshot_jsonl:
        for sample in iter_snapshot_samples(snapshot_path):
            sample_count += 1
            dense_length = sample.get("dense_length")
            if not isinstance(dense_length, int) or dense_length <= 1:
                skipped_no_dense_length += 1
                continue
            video_info = annotations.get(str(sample["sample_id"]))
            if not video_info:
                skipped_no_gt += 1
                continue
            gt_segments = gt_segments_to_dense_positions(video_info, dense_length=dense_length)
            if not gt_segments:
                skipped_no_gt += 1
                continue
            for stride, regression_range in zip(strides, regression_ranges):
                rows.append(
                    summarize_level_assignment(
                        sample_id=str(sample["sample_id"]),
                        snapshot_path=str(sample["snapshot_path"]),
                        snapshot_line=int(sample["snapshot_line"]),
                        snapshot_id=None if sample["snapshot_id"] is None else str(sample["snapshot_id"]),
                        positions=sample["positions"],
                        dense_length=dense_length,
                        gt_segments=gt_segments,
                        stride=int(stride),
                        regression_range=regression_range,
                        center_sample_radius=float(center_sample_radius),
                    )
                )

    per_level_path = out_dir / "per_sample_level_assignment.csv"
    if rows:
        with per_level_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(strict_json_value(row) for row in rows)

    fake_positive = [float(row["fake_positive"]) for row in rows]
    physical_positive = [float(row["physical_positive"]) for row in rows]
    positive_delta = [float(row["positive_delta_physical_minus_fake"]) for row in rows]
    fake_only = [float(row["fake_only"]) for row in rows]
    physical_only = [float(row["physical_only"]) for row in rows]
    same_gt_ratio = [
        float(row["same_gt_ratio_among_both"])
        for row in rows
        if _as_float(row.get("same_gt_ratio_among_both")) is not None
    ]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "decision": READY,
        "diagnostic": "actionformer_selected_axis_target_assignment_audit",
        "snapshot_jsonl": [str(Path(item).expanduser()) for item in snapshot_jsonl],
        "annotation": str(Path(annotation).expanduser()),
        "output_dir": str(out_dir),
        "per_sample_level_assignment_csv": str(per_level_path),
        "split": split,
        "sample_count": sample_count,
        "sample_level_rows": len(rows),
        "skipped_no_gt": skipped_no_gt,
        "skipped_no_dense_length": skipped_no_dense_length,
        "strides": [int(item) for item in strides],
        "regression_ranges": [[float(left), float(right)] for left, right in regression_ranges],
        "center_sample_radius": float(center_sample_radius),
        "fake_positive_stats": _stats(fake_positive),
        "physical_positive_stats": _stats(physical_positive),
        "positive_delta_physical_minus_fake_stats": _stats(positive_delta),
        "fake_only_stats": _stats(fake_only),
        "physical_only_stats": _stats(physical_only),
        "same_gt_ratio_among_both_stats": _stats(same_gt_ratio),
        "coordinate_assumption": (
            "GT seconds are mapped to whole-video dense-index coordinates using duration and snapshot dense_length; "
            "this is a diagnostic approximation, not an exact dataloader window-level training-target replay."
        ),
        "no_model_forward": True,
        "no_training": True,
        "uses_existing_reader_snapshot": True,
        "uses_validation_gt": True,
        "uses_validation_gt_for_offline_diagnostic_only": True,
        "exact_training_assignment_replay": False,
        "uses_teacher": False,
        "uses_oracle": False,
        "uses_cache": False,
        "uses_raw_prediction": False,
        "metric_claim_allowed": False,
        "paper_claim_allowed": False,
        "runtime_or_deployment_claim_allowed": False,
        "formal_eval_allowed": False,
    }
    write_json(out_dir / "summary.json", summary)
    return summary


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
    parser = argparse.ArgumentParser(
        description="Compare ActionFormer target assignment under fake selected-axis points versus physical selected-time points."
    )
    parser.add_argument("--snapshot-jsonl", nargs="+", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--strides", default="1,2,4,8,16,32")
    parser.add_argument("--regression-ranges", default="0:4,4:8,8:16,16:32,32:64,64:10000")
    parser.add_argument("--center-sample-radius", type=float, default=1.5)
    args = parser.parse_args(argv)

    try:
        summary = run_selected_axis_target_assignment_audit(
            snapshot_jsonl=parse_path_list(args.snapshot_jsonl),
            annotation=args.annotation,
            output_dir=args.output_dir,
            split=args.split,
            strides=parse_int_list(args.strides),
            regression_ranges=parse_regression_ranges(args.regression_ranges),
            center_sample_radius=args.center_sample_radius,
        )
    except Exception as exc:  # pragma: no cover - CLI guard
        print(json.dumps(strict_json_value(error_payload(exc)), sort_keys=True))
        return 1
    print(json.dumps(strict_json_value(summary), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
