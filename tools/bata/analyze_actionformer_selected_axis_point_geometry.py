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


SCHEMA_VERSION = "actionformer_selected_axis_point_geometry_audit_v0"
READY = "ACTIONFORMER_SELECTED_AXIS_POINT_GEOMETRY_AUDIT_READY"
NO_GO = "ACTIONFORMER_SELECTED_AXIS_POINT_GEOMETRY_AUDIT_NO_GO"
FORBIDDEN_TRUE_FLAGS = (
    "uses_gt",
    "uses_teacher",
    "uses_oracle",
    "uses_cache",
    "uses_raw_prediction",
    "detector_map_allowed",
    "metric_claim_allowed",
    "paper_claim_allowed",
    "runtime_flops_claim_allowed",
    "deploy_claim_allowed",
    "tools_test_allowed",
)


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
        if not math.isfinite(value):
            return None
        return float(value)
    return str(value)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    Path(path).write_text(json.dumps(strict_json_value(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _truthy_number(value: Any) -> bool:
    out = _as_float(value)
    return bool(out is not None and out > 0.5)


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _sample_value(value: Any, index: int) -> Any:
    if not _is_sequence(value):
        return value
    if index >= len(value):
        raise ValueError(f"batch index {index} is out of range for value of length {len(value)}")
    return value[index]


def _float_list(value: Any, *, context: str) -> list[float]:
    if not _is_sequence(value):
        raise ValueError(f"{context} must be a sequence")
    out: list[float] = []
    for idx, item in enumerate(value):
        parsed = _as_float(item)
        if parsed is None:
            raise ValueError(f"{context}[{idx}] must be finite")
        out.append(float(parsed))
    return out


def _row_matrix(value: Any, *, context: str) -> list[list[float]]:
    if not _is_sequence(value):
        raise ValueError(f"{context} must be a sequence of rows")
    return [_float_list(row, context=f"{context}[{idx}]") for idx, row in enumerate(value)]


def _quantile(sorted_values: Sequence[float], q: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = max(0.0, min(1.0, float(q))) * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    weight = pos - lo
    return float(sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight)


def _stats(values: Sequence[float]) -> dict[str, Any]:
    cleaned = [float(item) for item in values if math.isfinite(float(item))]
    if not cleaned:
        return {"count": 0}
    sorted_values = sorted(cleaned)
    mean = sum(sorted_values) / len(sorted_values)
    variance = sum((item - mean) ** 2 for item in sorted_values) / len(sorted_values)
    return {
        "count": len(sorted_values),
        "min": sorted_values[0],
        "p50": _quantile(sorted_values, 0.50),
        "p90": _quantile(sorted_values, 0.90),
        "p95": _quantile(sorted_values, 0.95),
        "max": sorted_values[-1],
        "mean": mean,
        "std": math.sqrt(variance),
    }


def _validate_snapshot_row(row: Mapping[str, Any], *, source: str, line_number: int) -> None:
    if row.get("diagnostic_only") is not True:
        raise ValueError(f"{source}:{line_number} snapshot row must be diagnostic_only=true")
    for flag in FORBIDDEN_TRUE_FLAGS:
        if row.get(flag) is True:
            raise ValueError(f"{source}:{line_number} forbidden provenance flag is true: {flag}")
    reader_out = row.get("reader_out")
    if not isinstance(reader_out, Mapping):
        raise ValueError(f"{source}:{line_number} snapshot row missing reader_out object")
    sample_ids = row.get("sample_ids")
    if not _is_sequence(sample_ids):
        raise ValueError(f"{source}:{line_number} snapshot row missing sample_ids list")


def _valid_dense_length(reader_out: Mapping[str, Any], *, batch_index: int) -> int | None:
    if "valid_lengths" in reader_out:
        value = _sample_value(reader_out["valid_lengths"], batch_index)
        parsed = _as_int(value)
        if parsed is not None and parsed > 0:
            return int(parsed)
    if "valid_mask" in reader_out:
        mask = _sample_value(reader_out["valid_mask"], batch_index)
        if _is_sequence(mask):
            return int(sum(1 for item in mask if _truthy_number(item)))
    return None


def _selected_slot_indices(reader_out: Mapping[str, Any], *, batch_index: int, fallback_length: int) -> list[int]:
    if "selected_mask" not in reader_out:
        return list(range(int(fallback_length)))
    mask = _sample_value(reader_out["selected_mask"], batch_index)
    if not _is_sequence(mask):
        raise ValueError("selected_mask sample must be a sequence")
    return [idx for idx, item in enumerate(mask) if _truthy_number(item)]


def _positions_from_selected_times(
    reader_out: Mapping[str, Any],
    *,
    batch_index: int,
    selected_indices: Sequence[int],
    dense_length: int | None,
) -> list[float] | None:
    if "selected_times" not in reader_out:
        return None
    times = _float_list(_sample_value(reader_out["selected_times"], batch_index), context="selected_times")
    selected = [times[idx] for idx in selected_indices if idx < len(times)]
    if not selected:
        return None
    max_time = max(selected)
    min_time = min(selected)
    if dense_length is not None and dense_length > 1 and min_time >= -1e-6 and max_time <= 1.000001:
        scale = float(dense_length - 1)
        return [float(item * scale) for item in selected]
    return selected


def _positions_from_selected_positions(
    reader_out: Mapping[str, Any],
    *,
    batch_index: int,
    selected_indices: Sequence[int],
) -> list[float] | None:
    for key in ("selected_dense_positions", "selected_positions"):
        if key not in reader_out:
            continue
        positions = _float_list(_sample_value(reader_out[key], batch_index), context=key)
        selected = [positions[idx] for idx in selected_indices if idx < len(positions)]
        if selected:
            return selected
    return None


def _positions_from_acquisition_matrix(
    reader_out: Mapping[str, Any],
    *,
    batch_index: int,
    selected_indices: Sequence[int],
) -> list[float] | None:
    key = "acquisition_matrix"
    if key not in reader_out:
        return None
    matrix = _row_matrix(_sample_value(reader_out[key], batch_index), context=key)
    positions: list[float] = []
    for selected_index in selected_indices:
        if selected_index >= len(matrix):
            continue
        row = matrix[selected_index]
        mass = sum(value for value in row if value > 0.0)
        if mass <= 0.0:
            continue
        weighted = sum(float(idx) * max(0.0, value) for idx, value in enumerate(row)) / mass
        positions.append(float(weighted))
    return positions or None


def _positions_from_centers(
    reader_out: Mapping[str, Any],
    *,
    batch_index: int,
    selected_indices: Sequence[int],
    dense_length: int | None,
) -> list[float] | None:
    if "centers" not in reader_out:
        return None
    centers = _float_list(_sample_value(reader_out["centers"], batch_index), context="centers")
    selected = [centers[idx] for idx in selected_indices if idx < len(centers)]
    if not selected:
        return None
    max_center = max(selected)
    min_center = min(selected)
    if dense_length is not None and dense_length > 1 and min_center >= -1e-6 and max_center <= 1.000001:
        scale = float(dense_length - 1)
        return [float(item * scale) for item in selected]
    return selected


def reconstruct_selected_positions(
    reader_out: Mapping[str, Any],
    *,
    batch_index: int,
) -> tuple[list[float], str, int | None]:
    dense_length = _valid_dense_length(reader_out, batch_index=batch_index)
    fallback_length = 0
    for key in ("selected_times", "selected_positions", "selected_dense_positions", "centers", "selected_mask"):
        if key in reader_out and _is_sequence(_sample_value(reader_out[key], batch_index)):
            fallback_length = len(_sample_value(reader_out[key], batch_index))
            break
    selected_indices = _selected_slot_indices(reader_out, batch_index=batch_index, fallback_length=fallback_length)
    if not selected_indices:
        raise ValueError("snapshot sample has no valid selected tokens")

    candidates = (
        ("selected_positions", _positions_from_selected_positions(reader_out, batch_index=batch_index, selected_indices=selected_indices)),
        (
            "selected_times",
            _positions_from_selected_times(
                reader_out,
                batch_index=batch_index,
                selected_indices=selected_indices,
                dense_length=dense_length,
            ),
        ),
        (
            "acquisition_matrix",
            _positions_from_acquisition_matrix(
                reader_out,
                batch_index=batch_index,
                selected_indices=selected_indices,
            ),
        ),
        (
            "centers",
            _positions_from_centers(
                reader_out,
                batch_index=batch_index,
                selected_indices=selected_indices,
                dense_length=dense_length,
            ),
        ),
    )
    for source, positions in candidates:
        if positions:
            cleaned = [float(item) for item in positions if math.isfinite(float(item))]
            if cleaned:
                return cleaned, source, dense_length
    raise ValueError("snapshot sample has no reconstructable selected positions")


def _group_mean_positions(positions: Sequence[float], stride: int) -> list[float]:
    if stride <= 0:
        raise ValueError("stride must be positive")
    grouped: list[float] = []
    for start in range(0, len(positions), int(stride)):
        chunk = [float(item) for item in positions[start : start + int(stride)]]
        if chunk:
            grouped.append(sum(chunk) / len(chunk))
    return grouped


def _monotonic_violations(values: Sequence[float]) -> int:
    return sum(1 for prev, curr in zip(values, values[1:]) if curr < prev)


def summarize_sample_level(
    *,
    sample_id: str,
    snapshot_path: str,
    snapshot_line: int,
    snapshot_id: str | None,
    positions: Sequence[float],
    position_source: str,
    dense_length: int | None,
    stride: int,
) -> dict[str, Any]:
    physical = _group_mean_positions(positions, int(stride))
    fake = [float(idx * int(stride)) for idx in range(len(physical))]
    deltas = [physical_value - fake_value for physical_value, fake_value in zip(physical, fake)]
    abs_deltas = [abs(item) for item in deltas]
    gaps = [curr - prev for prev, curr in zip(physical, physical[1:])]
    fake_last = fake[-1] if fake else None
    physical_last = physical[-1] if physical else None
    fake_span = fake[-1] - fake[0] if len(fake) >= 2 else 0.0
    physical_span = physical[-1] - physical[0] if len(physical) >= 2 else 0.0
    selected_count = len(positions)
    density_ratio = None if dense_length in (None, 0) else selected_count / float(dense_length)
    return {
        "sample_id": sample_id,
        "snapshot_path": snapshot_path,
        "snapshot_line": int(snapshot_line),
        "snapshot_id": snapshot_id,
        "position_source": position_source,
        "dense_length": dense_length,
        "selected_count": selected_count,
        "selection_density_ratio": density_ratio,
        "nominal_stride": int(stride),
        "level_length": len(physical),
        "actionformer_fake_start": fake[0] if fake else None,
        "actionformer_fake_end": fake_last,
        "physical_start": physical[0] if physical else None,
        "physical_end": physical_last,
        "fake_span": fake_span,
        "physical_span": physical_span,
        "fake_span_to_physical_span_ratio": None if physical_span <= 0 else fake_span / physical_span,
        "fake_end_to_physical_end_ratio": None
        if physical_last is None or physical_last <= 0
        else float(fake_last or 0.0) / physical_last,
        "monotonic_violations": _monotonic_violations(physical),
        "signed_delta_mean": _stats(deltas).get("mean"),
        "signed_delta_min": _stats(deltas).get("min"),
        "signed_delta_max": _stats(deltas).get("max"),
        "abs_delta_mean": _stats(abs_deltas).get("mean"),
        "abs_delta_p90": _stats(abs_deltas).get("p90"),
        "abs_delta_max": _stats(abs_deltas).get("max"),
        "physical_gap_mean": _stats(gaps).get("mean"),
        "physical_gap_p90": _stats(gaps).get("p90"),
        "physical_gap_max": _stats(gaps).get("max"),
    }


def _append_aggregate(
    aggregates: dict[int, dict[str, list[float] | int]],
    *,
    stride: int,
    row: Mapping[str, Any],
) -> None:
    bucket = aggregates.setdefault(
        int(stride),
        {
            "sample_count": 0,
            "point_count": 0,
            "abs_delta_mean_values": [],
            "abs_delta_max_values": [],
            "fake_span_ratios": [],
            "fake_end_ratios": [],
            "selection_density_ratios": [],
            "physical_gap_means": [],
        },
    )
    bucket["sample_count"] = int(bucket["sample_count"]) + 1
    bucket["point_count"] = int(bucket["point_count"]) + int(row.get("level_length") or 0)
    for source_key, target_key in (
        ("abs_delta_mean", "abs_delta_mean_values"),
        ("abs_delta_max", "abs_delta_max_values"),
        ("fake_span_to_physical_span_ratio", "fake_span_ratios"),
        ("fake_end_to_physical_end_ratio", "fake_end_ratios"),
        ("selection_density_ratio", "selection_density_ratios"),
        ("physical_gap_mean", "physical_gap_means"),
    ):
        value = _as_float(row.get(source_key))
        if value is not None:
            bucket[target_key].append(value)  # type: ignore[index, union-attr]


def _finalize_aggregates(aggregates: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for stride, bucket in sorted(aggregates.items()):
        out[str(stride)] = {
            "sample_count": int(bucket["sample_count"]),
            "point_count": int(bucket["point_count"]),
            "abs_delta_mean": _stats(bucket["abs_delta_mean_values"]).get("mean"),
            "abs_delta_max_p90": _stats(bucket["abs_delta_max_values"]).get("p90"),
            "fake_span_to_physical_span_ratio_mean": _stats(bucket["fake_span_ratios"]).get("mean"),
            "fake_end_to_physical_end_ratio_mean": _stats(bucket["fake_end_ratios"]).get("mean"),
            "selection_density_ratio_mean": _stats(bucket["selection_density_ratios"]).get("mean"),
            "physical_gap_mean": _stats(bucket["physical_gap_means"]).get("mean"),
        }
    return out


def iter_snapshot_samples(snapshot_jsonl: str | Path) -> tuple[dict[str, Any], ...]:
    samples: list[dict[str, Any]] = []
    path = Path(snapshot_jsonl).expanduser()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping):
                raise ValueError(f"{path}:{line_number} snapshot line must be an object")
            _validate_snapshot_row(row, source=str(path), line_number=line_number)
            sample_ids = [str(item) for item in row["sample_ids"]]
            reader_out = row["reader_out"]
            for batch_index, sample_id in enumerate(sample_ids):
                positions, source, dense_length = reconstruct_selected_positions(reader_out, batch_index=batch_index)
                samples.append(
                    {
                        "sample_id": sample_id,
                        "snapshot_path": str(path),
                        "snapshot_line": int(line_number),
                        "snapshot_id": row.get("snapshot_id"),
                        "positions": positions,
                        "position_source": source,
                        "dense_length": dense_length,
                        "source_uses_checkpoint": bool(row.get("uses_checkpoint")),
                    }
                )
    return tuple(samples)


def run_selected_axis_point_geometry_audit(
    *,
    snapshot_jsonl: Sequence[str | Path],
    output_dir: str | Path,
    strides: Sequence[int] = (1, 2, 4, 8, 16, 32),
) -> dict[str, Any]:
    if not snapshot_jsonl:
        raise ValueError("at least one snapshot JSONL path is required")
    normalized_strides = [int(item) for item in strides]
    if not normalized_strides or any(item <= 0 for item in normalized_strides):
        raise ValueError("strides must be positive")

    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    per_sample_level_rows: list[dict[str, Any]] = []
    aggregates: dict[int, dict[str, list[float] | int]] = {}
    source_counts: dict[str, int] = {}
    sample_count = 0
    source_uses_checkpoint = False

    for snapshot_path in snapshot_jsonl:
        for sample in iter_snapshot_samples(snapshot_path):
            sample_count += 1
            source_uses_checkpoint = source_uses_checkpoint or bool(sample["source_uses_checkpoint"])
            source_counts[str(sample["position_source"])] = source_counts.get(str(sample["position_source"]), 0) + 1
            for stride in normalized_strides:
                row = summarize_sample_level(
                    sample_id=str(sample["sample_id"]),
                    snapshot_path=str(sample["snapshot_path"]),
                    snapshot_line=int(sample["snapshot_line"]),
                    snapshot_id=None if sample["snapshot_id"] is None else str(sample["snapshot_id"]),
                    positions=sample["positions"],
                    position_source=str(sample["position_source"]),
                    dense_length=sample["dense_length"],
                    stride=int(stride),
                )
                per_sample_level_rows.append(row)
                _append_aggregate(aggregates, stride=int(stride), row=row)

    per_sample_level_path = out_dir / "per_sample_level_geometry.csv"
    if per_sample_level_rows:
        with per_sample_level_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_sample_level_rows[0].keys()))
            writer.writeheader()
            writer.writerows(strict_json_value(row) for row in per_sample_level_rows)

    stride_one_rows = [row for row in per_sample_level_rows if int(row["nominal_stride"]) == 1]
    span_ratios = [
        float(row["fake_span_to_physical_span_ratio"])
        for row in stride_one_rows
        if _as_float(row.get("fake_span_to_physical_span_ratio")) is not None
    ]
    abs_delta_means = [
        float(row["abs_delta_mean"])
        for row in stride_one_rows
        if _as_float(row.get("abs_delta_mean")) is not None
    ]
    density_ratios = [
        float(row["selection_density_ratio"])
        for row in stride_one_rows
        if _as_float(row.get("selection_density_ratio")) is not None
    ]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "decision": READY,
        "diagnostic": "actionformer_selected_axis_point_geometry_audit",
        "snapshot_jsonl": [str(Path(item).expanduser()) for item in snapshot_jsonl],
        "output_dir": str(out_dir),
        "per_sample_level_geometry_csv": str(per_sample_level_path),
        "strides": normalized_strides,
        "sample_count": int(sample_count),
        "sample_level_rows": len(per_sample_level_rows),
        "position_source_counts": source_counts,
        "aggregate_by_nominal_stride": _finalize_aggregates(aggregates),
        "stride1_fake_span_to_physical_span_ratio": _stats(span_ratios),
        "stride1_abs_delta_mean": _stats(abs_delta_means),
        "selection_density_ratio": _stats(density_ratios),
        "interpretation": (
            "Read-only coordinate audit: original ActionFormer points are generated on selected-token "
            "indices, while PC-OT-MRAS selected tokens may occupy irregular physical dense positions."
        ),
        "no_model_forward": True,
        "no_training": True,
        "uses_existing_reader_snapshot": True,
        "uses_checkpoint_through_existing_snapshot": bool(source_uses_checkpoint),
        "uses_gt": False,
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


def parse_path_list(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        for item in str(value).split(","):
            stripped = item.strip()
            if stripped:
                out.append(stripped)
    return out


def parse_int_list(value: str) -> list[int]:
    out: list[int] = []
    for item in str(value).split(","):
        stripped = item.strip()
        if stripped:
            out.append(int(stripped))
    return out


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
        description="Audit original ActionFormer uniform points against PC-OT-MRAS selected-token physical positions."
    )
    parser.add_argument("--snapshot-jsonl", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--strides", default="1,2,4,8,16,32")
    args = parser.parse_args(argv)

    try:
        summary = run_selected_axis_point_geometry_audit(
            snapshot_jsonl=parse_path_list(args.snapshot_jsonl),
            output_dir=args.output_dir,
            strides=parse_int_list(args.strides),
        )
    except Exception as exc:  # pragma: no cover - CLI guard
        print(json.dumps(strict_json_value(error_payload(exc)), sort_keys=True))
        return 1
    print(json.dumps(strict_json_value(summary), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
