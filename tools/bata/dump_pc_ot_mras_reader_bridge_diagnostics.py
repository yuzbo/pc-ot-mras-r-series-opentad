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


SCHEMA_VERSION = "pc_ot_mras_reader_bridge_diagnostic_dump_v0"
SUMMARY_SCHEMA_VERSION = "pc_ot_mras_reader_bridge_diagnostic_summary_v0"
READY = "PC_OT_MRAS_READER_BRIDGE_DIAGNOSTIC_READY"
NO_GO = "PC_OT_MRAS_READER_BRIDGE_DIAGNOSTIC_NO_GO"
AGGREGATE_VALUES_KEY = "_aggregate_values"
MATRIX_PRIORITY = ("acquisition_matrix", "allocation", "transport_prob")
SELECTED_TOKEN_KEYS = ("selected_tokens", "selected_token_features", "selected_features")
BRIDGE_OUTPUT_KEYS = ("bridge_out", "bridge_output", "bridge_outputs", "bridge_features", "pc_ot_mras_bridge_output")


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
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(strict_json_value(dict(row)), sort_keys=True) + "\n")


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


def _sample_rank(value: Any, *, batch_idx: int, expected_rank: int) -> Any:
    data = _to_plain(value)
    depth = _depth(data)
    if depth == expected_rank + 1:
        if not isinstance(data, list) or batch_idx >= len(data):
            raise ValueError("batch index outside batched value")
        return data[batch_idx]
    if depth == expected_rank:
        return data
    raise ValueError(f"expected rank {expected_rank} or batched rank {expected_rank + 1}, got rank {depth}")


def _infer_batch_size(reader_out: Mapping[str, Any]) -> int:
    for key in MATRIX_PRIORITY:
        if key in reader_out and _depth(reader_out[key]) >= 3:
            return len(_to_plain(reader_out[key]))
    for key in ("selected_times", "centers", "gates", "selected_mask", "valid_mask"):
        if key in reader_out and _depth(reader_out[key]) >= 2:
            return len(_to_plain(reader_out[key]))
    for key in SELECTED_TOKEN_KEYS:
        if key in reader_out and _depth(reader_out[key]) >= 3:
            return len(_to_plain(reader_out[key]))
    return 1


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be numeric") from None
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _float_vector(value: Any, *, name: str) -> list[float]:
    data = _to_plain(value)
    if not isinstance(data, list):
        raise ValueError(f"{name} must be a list")
    return [_finite_float(item, name=f"{name}[{idx}]") for idx, item in enumerate(data)]


def _float_matrix(value: Any, *, name: str) -> list[list[float]]:
    data = _to_plain(value)
    if not isinstance(data, list) or not data or not all(isinstance(row, list) for row in data):
        raise ValueError(f"{name} must be a non-empty [K,T] matrix")
    width = len(data[0])
    if width <= 0:
        raise ValueError(f"{name} must have non-empty rows")
    out: list[list[float]] = []
    for row_idx, row in enumerate(data):
        if len(row) != width:
            raise ValueError(f"{name} must be rectangular")
        out.append([_finite_float(item, name=f"{name}[{row_idx}][{col_idx}]") for col_idx, item in enumerate(row)])
    return out


def _binary_vector(value: Any, *, name: str) -> list[bool]:
    data = _to_plain(value)
    if not isinstance(data, list):
        raise ValueError(f"{name} must be a list")
    out = []
    for idx, item in enumerate(data):
        if item not in (0, 1, False, True):
            raise ValueError(f"{name}[{idx}] must be binary")
        out.append(bool(item))
    return out


def _flatten_numeric(value: Any, *, name: str) -> list[float]:
    data = _to_plain(value)
    out: list[float] = []

    def visit(item: Any, path: str) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                visit(child, f"{path}.{key}")
        elif isinstance(item, list):
            for idx, child in enumerate(item):
                visit(child, f"{path}[{idx}]")
        elif item is not None and not isinstance(item, str):
            out.append(_finite_float(item, name=path))

    visit(data, name)
    return out


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else float(sum(float(item) for item in values) / float(len(values)))


def _std(values: Sequence[float]) -> float | None:
    if len(values) <= 1:
        return None
    mean = _mean(values) or 0.0
    return float(math.sqrt(sum((float(item) - mean) ** 2 for item in values) / float(len(values))))


def _quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(item) for item in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = min(max(float(q), 0.0), 1.0) * (len(ordered) - 1)
    left = int(math.floor(pos))
    right = int(math.ceil(pos))
    if left == right:
        return ordered[left]
    frac = pos - left
    return float(ordered[left] * (1.0 - frac) + ordered[right] * frac)


def summarize_values(values: Sequence[float], *, keep_aggregate_values: bool = False) -> dict[str, Any]:
    finite = [float(item) for item in values if math.isfinite(float(item))]
    if not finite:
        summary: dict[str, Any] = {
            "count": 0,
            "min": None,
            "p05": None,
            "mean": None,
            "std": None,
            "p50": None,
            "p95": None,
            "max": None,
        }
        if keep_aggregate_values:
            summary[AGGREGATE_VALUES_KEY] = []
        return summary
    summary = {
        "count": len(finite),
        "min": min(finite),
        "p05": _quantile(finite, 0.05),
        "mean": _mean(finite),
        "std": _std(finite),
        "p50": _quantile(finite, 0.50),
        "p95": _quantile(finite, 0.95),
        "max": max(finite),
    }
    if keep_aggregate_values:
        summary[AGGREGATE_VALUES_KEY] = finite
    return summary


def _summary_values(summary: Mapping[str, Any]) -> list[float]:
    if not summary:
        return []
    values = summary.get(AGGREGATE_VALUES_KEY)
    if values is None:
        raise ValueError(
            "aggregate requires raw per-slot values; rerun diagnostics with "
            f"{AGGREGATE_VALUES_KEY} support instead of aggregating mean-only summaries"
        )
    if not isinstance(values, list):
        raise ValueError(f"{AGGREGATE_VALUES_KEY} must be a list")
    return [float(item) for item in values if math.isfinite(float(item))]


def _strip_aggregate_values(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _strip_aggregate_values(item)
            for key, item in value.items()
            if str(key) != AGGREGATE_VALUES_KEY
        }
    if isinstance(value, list):
        return [_strip_aggregate_values(item) for item in value]
    return value


def _entropy(row: Sequence[float]) -> tuple[float | None, float | None]:
    positive = [max(0.0, float(item)) for item in row]
    total = sum(positive)
    if total <= 0.0:
        return None, None
    probs = [item / total for item in positive if item > 0.0]
    entropy = -sum(prob * math.log(prob) for prob in probs)
    normalizer = math.log(len(positive)) if len(positive) > 1 else 0.0
    return float(entropy), None if normalizer <= 0.0 else float(entropy / normalizer)


def _argmax(values: Sequence[float]) -> int:
    best_idx = 0
    best = float(values[0])
    for idx, value in enumerate(values[1:], start=1):
        if float(value) > best:
            best_idx = idx
            best = float(value)
    return int(best_idx)


def _valid_len_from(reader_out: Mapping[str, Any], *, batch_idx: int, dense_len: int) -> int:
    if "valid_mask" in reader_out:
        mask = _binary_vector(_sample_rank(reader_out["valid_mask"], batch_idx=batch_idx, expected_rank=1), name="valid_mask")
        if len(mask) != dense_len:
            raise ValueError("valid_mask length must match matrix dense axis")
        return int(sum(1 for item in mask if item))
    if "valid_lengths" in reader_out:
        lengths = _to_plain(reader_out["valid_lengths"])
        if isinstance(lengths, list):
            value = lengths[batch_idx] if len(lengths) > 1 else lengths[0]
            return int(_finite_float(value, name="valid_lengths"))
    return int(dense_len)


def _slot_mask(reader_out: Mapping[str, Any], *, batch_idx: int, slot_count: int) -> list[bool]:
    if "selected_mask" not in reader_out:
        return [True] * int(slot_count)
    mask = _binary_vector(_sample_rank(reader_out["selected_mask"], batch_idx=batch_idx, expected_rank=1), name="selected_mask")
    if len(mask) != int(slot_count):
        raise ValueError("selected_mask length must match slot count")
    return mask


def _center_to_dense_index(center: float, *, valid_len: int) -> tuple[float, str]:
    if -1.0e-6 <= float(center) <= 1.0 + 1.0e-6:
        return float(center) * float(max(int(valid_len) - 1, 1)), "normalized_0_1"
    return float(center), "dense_index"


def _slot_norms(value: Any, *, name: str) -> list[float]:
    data = _to_plain(value)
    if not isinstance(data, list):
        raise ValueError(f"{name} must be a list")
    if data and all(isinstance(row, list) for row in data):
        return [
            math.sqrt(sum(item * item for item in _flatten_numeric(row, name=f"{name}[{idx}]")))
            for idx, row in enumerate(data)
        ]
    values = _flatten_numeric(data, name=name)
    return [math.sqrt(sum(item * item for item in values))]


def _extract_matrix(reader_out: Mapping[str, Any], *, batch_idx: int) -> tuple[str | None, list[list[float]] | None]:
    for key in MATRIX_PRIORITY:
        if key in reader_out:
            return key, _float_matrix(_sample_rank(reader_out[key], batch_idx=batch_idx, expected_rank=2), name=key)
    return None, None


def _optional_vector(reader_out: Mapping[str, Any], key: str, *, batch_idx: int, expected_len: int | None = None) -> list[float] | None:
    if key not in reader_out:
        return None
    values = _float_vector(_sample_rank(reader_out[key], batch_idx=batch_idx, expected_rank=1), name=key)
    if expected_len is not None and len(values) != int(expected_len):
        raise ValueError(f"{key} length must be {int(expected_len)}")
    return values


def _optional_token_norms(reader_out: Mapping[str, Any], *, batch_idx: int) -> tuple[str | None, list[float]]:
    for key in SELECTED_TOKEN_KEYS:
        if key in reader_out:
            sampled = _sample_rank(reader_out[key], batch_idx=batch_idx, expected_rank=2)
            return key, _slot_norms(sampled, name=key)
    return None, []


def _bridge_norm_values(row: Mapping[str, Any]) -> tuple[str | None, list[float]]:
    if "bridge_output_norm_hook" in row:
        values = _flatten_numeric(row["bridge_output_norm_hook"], name="bridge_output_norm_hook")
        if values:
            return "bridge_output_norm_hook", values
    for key in BRIDGE_OUTPUT_KEYS:
        if key in row:
            values = _flatten_numeric(row[key], name=key)
            if values:
                return key, [math.sqrt(sum(item * item for item in values))]
    bridge_meta = row.get("pc_ot_mras_bridge")
    if isinstance(bridge_meta, Mapping):
        for key in ("output", "features", "selected_tokens"):
            if key in bridge_meta:
                values = _flatten_numeric(bridge_meta[key], name=f"pc_ot_mras_bridge.{key}")
                if values:
                    return f"pc_ot_mras_bridge.{key}", [math.sqrt(sum(item * item for item in values))]
    return None, []


def _gate_histogram(values: Sequence[float]) -> dict[str, int]:
    buckets = {"[0,0.25)": 0, "[0.25,0.5)": 0, "[0.5,0.75)": 0, "[0.75,1]": 0, "outside_0_1": 0}
    for value in values:
        item = float(value)
        if item < 0.0 or item > 1.0:
            buckets["outside_0_1"] += 1
        elif item < 0.25:
            buckets["[0,0.25)"] += 1
        elif item < 0.5:
            buckets["[0.25,0.5)"] += 1
        elif item < 0.75:
            buckets["[0.5,0.75)"] += 1
        else:
            buckets["[0.75,1]"] += 1
    return buckets


def diagnose_reader_out_sample(
    reader_out: Mapping[str, Any],
    *,
    sample_id: str,
    snapshot_id: str,
    batch_idx: int = 0,
    row_bridge_norm: tuple[str | None, list[float]] | None = None,
) -> dict[str, Any]:
    if not isinstance(reader_out, Mapping):
        raise ValueError("reader_out must be a mapping")
    matrix_key, matrix = _extract_matrix(reader_out, batch_idx=batch_idx)
    if matrix is None:
        raise ValueError("reader_out needs acquisition_matrix, allocation, or transport_prob")
    slot_count = len(matrix)
    dense_len = len(matrix[0])
    valid_len = _valid_len_from(reader_out, batch_idx=batch_idx, dense_len=dense_len)
    valid_slots = [idx for idx, item in enumerate(_slot_mask(reader_out, batch_idx=batch_idx, slot_count=slot_count)) if item]

    centers = _optional_vector(reader_out, "centers", batch_idx=batch_idx, expected_len=slot_count)
    selected_times = _optional_vector(reader_out, "selected_times", batch_idx=batch_idx, expected_len=slot_count)
    gates = _optional_vector(reader_out, "gates", batch_idx=batch_idx, expected_len=slot_count)

    signed_offsets = []
    abs_offsets = []
    if centers is not None and selected_times is not None:
        for slot_idx in valid_slots:
            signed = float(selected_times[slot_idx]) - float(centers[slot_idx])
            signed_offsets.append(signed)
            abs_offsets.append(abs(signed))

    deltas = []
    if selected_times is not None:
        valid_times = [selected_times[idx] for idx in valid_slots]
        deltas = [float(right) - float(left) for left, right in zip(valid_times, valid_times[1:])]
    monotonic_violations = sum(1 for item in deltas if item < -1.0e-8)
    strict_violations = sum(1 for item in deltas if item <= 1.0e-8)

    entropies = []
    norm_entropies = []
    top1_positions = []
    top1_center_dense = []
    top1_center_norm = []
    center_unit = None
    for slot_idx in valid_slots:
        row = matrix[slot_idx][:valid_len]
        entropy, norm_entropy = _entropy(row)
        if entropy is not None:
            entropies.append(entropy)
        if norm_entropy is not None:
            norm_entropies.append(norm_entropy)
        top1 = _argmax(row)
        top1_positions.append(top1)
        if centers is not None:
            center_idx, unit = _center_to_dense_index(centers[slot_idx], valid_len=valid_len)
            center_unit = center_unit or unit
            distance = abs(float(top1) - center_idx)
            top1_center_dense.append(distance)
            top1_center_norm.append(distance / float(max(valid_len - 1, 1)))

    token_source, token_norms = _optional_token_norms(reader_out, batch_idx=batch_idx)
    bridge_source, bridge_norms = row_bridge_norm or (None, [])
    gate_values = [gates[idx] for idx in valid_slots] if gates is not None else []

    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": str(sample_id),
        "snapshot_id": str(snapshot_id),
        "batch_index": int(batch_idx),
        "matrix_key": matrix_key,
        "dense_len": int(dense_len),
        "valid_len": int(valid_len),
        "slot_count": int(slot_count),
        "valid_slot_count": int(len(valid_slots)),
        "centers_selected_times": {
            "available": bool(centers is not None and selected_times is not None),
            "signed_offset": summarize_values(signed_offsets, keep_aggregate_values=True),
            "abs_offset": summarize_values(abs_offsets, keep_aggregate_values=True),
            "preview": signed_offsets[:8],
        },
        "selected_times_monotonicity": {
            "available": selected_times is not None,
            "nondecreasing": None if selected_times is None else monotonic_violations == 0,
            "strictly_increasing": None if selected_times is None else strict_violations == 0,
            "violation_count": int(monotonic_violations),
            "strict_violation_count": int(strict_violations),
            "delta": summarize_values(deltas, keep_aggregate_values=True),
        },
        "gates": {
            "available": gates is not None,
            "stats": summarize_values(gate_values, keep_aggregate_values=True),
            "histogram_0_0p25_0p5_0p75_1": _gate_histogram(gate_values),
        },
        "acquisition": {
            "entropy": summarize_values(entropies, keep_aggregate_values=True),
            "normalized_entropy": summarize_values(norm_entropies, keep_aggregate_values=True),
            "top1_positions_preview": top1_positions[:16],
            "top1_center_distance_dense": summarize_values(top1_center_dense, keep_aggregate_values=True),
            "top1_center_distance_normalized": summarize_values(top1_center_norm, keep_aggregate_values=True),
            "center_unit": center_unit,
        },
        "selected_token_norm": {
            "available": bool(token_norms),
            "source_key": token_source,
            "stats": summarize_values(token_norms, keep_aggregate_values=True),
        },
        "bridge_output_norm": {
            "available": bool(bridge_norms),
            "source_key": bridge_source,
            "stats": summarize_values(bridge_norms, keep_aggregate_values=True),
        },
        "diagnostic_only": True,
        "uses_gt": False,
        "uses_teacher": False,
        "uses_oracle": False,
        "uses_cache": False,
        "uses_raw_prediction": False,
        "metric_claim_allowed": False,
        "paper_claim_allowed": False,
        "runtime_flops_claim_allowed": False,
        "deploy_claim_allowed": False,
    }


def _sample_ids(row: Mapping[str, Any], *, batch_size: int, row_idx: int) -> list[str]:
    ids = row.get("sample_ids")
    if isinstance(ids, list):
        if len(ids) != int(batch_size):
            raise ValueError("sample_ids length must match inferred batch size")
        return [str(item) for item in ids]
    sample_id = str(row.get("sample_id", f"row_{row_idx}"))
    if int(batch_size) == 1:
        return [sample_id]
    return [f"{sample_id}|batch{idx}" for idx in range(int(batch_size))]


def _diagnose_row(row: Mapping[str, Any], *, row_idx: int, snapshot_label: str) -> list[dict[str, Any]]:
    if not isinstance(row, Mapping):
        raise ValueError(f"row {row_idx}: JSONL row must be an object")
    reader_out = row.get("reader_out", row)
    if not isinstance(reader_out, Mapping):
        raise ValueError(f"row {row_idx}: reader_out must be an object")
    batch_size = _infer_batch_size(reader_out)
    ids = _sample_ids(row, batch_size=batch_size, row_idx=row_idx)
    bridge_norm = _bridge_norm_values(row)
    snapshot_id = str(row.get("snapshot_id", snapshot_label))
    return [
        diagnose_reader_out_sample(
            reader_out,
            sample_id=ids[batch_idx],
            snapshot_id=snapshot_id,
            batch_idx=batch_idx,
            row_bridge_norm=bridge_norm,
        )
        for batch_idx in range(batch_size)
    ]


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).expanduser().open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_no}: JSONL row must be an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"JSONL has no rows: {path}")
    return rows


def _aggregate(per_sample: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    center_abs: list[float] = []
    selected_time_deltas: list[float] = []
    gate_values: list[float] = []
    entropy_values: list[float] = []
    norm_entropy_values: list[float] = []
    top1_center_values: list[float] = []
    token_norms: list[float] = []
    bridge_norms: list[float] = []
    monotonic_failures = 0
    for sample in per_sample:
        center_abs.extend(_summary_values(sample["centers_selected_times"]["abs_offset"]))
        selected_time_deltas.extend(_summary_values(sample["selected_times_monotonicity"]["delta"]))
        gate_values.extend(_summary_values(sample["gates"]["stats"]))
        entropy_values.extend(_summary_values(sample["acquisition"]["entropy"]))
        norm_entropy_values.extend(_summary_values(sample["acquisition"]["normalized_entropy"]))
        top1_center_values.extend(_summary_values(sample["acquisition"]["top1_center_distance_dense"]))
        token_norms.extend(_summary_values(sample["selected_token_norm"]["stats"]))
        bridge_norms.extend(_summary_values(sample["bridge_output_norm"]["stats"]))
        if sample["selected_times_monotonicity"]["nondecreasing"] is False:
            monotonic_failures += 1
    return {
        "centers_selected_times_abs_offset": summarize_values(center_abs),
        "selected_time_delta": summarize_values(selected_time_deltas),
        "selected_times_nondecreasing_failure_count": int(monotonic_failures),
        "gate": summarize_values(gate_values),
        "acquisition_entropy": summarize_values(entropy_values),
        "acquisition_normalized_entropy": summarize_values(norm_entropy_values),
        "acquisition_top1_center_distance_dense": summarize_values(top1_center_values),
        "selected_token_norm": summarize_values(token_norms),
        "bridge_output_norm": summarize_values(bridge_norms),
    }


def build_summary(
    per_sample: Sequence[Mapping[str, Any]],
    *,
    source: str,
    input_jsonl: str | Path | None = None,
    output_json: str | Path | None = None,
    snapshot_jsonl: str | Path | None = None,
) -> dict[str, Any]:
    if not per_sample:
        raise ValueError("no samples available for reader/bridge diagnostics")
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "decision": READY,
        "source": str(source),
        "input_jsonl": None if input_jsonl is None else str(input_jsonl),
        "output_json": None if output_json is None else str(output_json),
        "snapshot_jsonl": None if snapshot_jsonl is None else str(snapshot_jsonl),
        "sample_count": len(per_sample),
        "matrix_keys": sorted({str(item["matrix_key"]) for item in per_sample if item.get("matrix_key")}),
        "per_sample": _strip_aggregate_values(list(per_sample)),
        "aggregate": _aggregate(per_sample),
        "diagnostic_only": True,
        "uses_checkpoint": source == "checkpoint",
        "uses_gt": False,
        "uses_teacher": False,
        "uses_oracle": False,
        "uses_cache": False,
        "uses_raw_prediction": False,
        "metric_claim_allowed": False,
        "paper_claim_allowed": False,
        "runtime_flops_claim_allowed": False,
        "deploy_claim_allowed": False,
        "tools_train_allowed": False,
        "detector_map_allowed": False,
        "slurm_gpu_allowed": False,
    }
    if output_json is not None:
        write_json(output_json, summary)
    return summary


def synthetic_reader_out(*, batch_size: int = 1, slots: int = 4, time: int = 8, channels: int = 3) -> dict[str, Any]:
    if min(int(batch_size), int(slots), int(time), int(channels)) <= 0:
        raise ValueError("synthetic dimensions must be positive")
    matrices = []
    allocations = []
    selected_times = []
    centers = []
    gates = []
    tokens = []
    for batch_idx in range(int(batch_size)):
        matrix_rows = []
        allocation_rows = []
        time_values = []
        center_values = []
        gate_values = []
        token_rows = []
        for slot_idx in range(int(slots)):
            center = (slot_idx + 0.5) / float(slots)
            pos = min(int(time) - 1, max(0, int(round(center * (int(time) - 1)))))
            row = [0.0 for _ in range(int(time))]
            row[pos] = 0.8 - 0.05 * (slot_idx % 3)
            if pos + 1 < int(time):
                row[pos + 1] = 0.2
            mass = sum(row) or 1.0
            allocation = [item / mass for item in row]
            gate = 0.35 + 0.1 * ((slot_idx + batch_idx) % 4)
            matrix_rows.append([item * gate for item in allocation])
            allocation_rows.append(allocation)
            time_values.append(sum(allocation[idx] * (idx / float(max(int(time) - 1, 1))) for idx in range(int(time))))
            center_values.append(center)
            gate_values.append(gate)
            token_rows.append([gate * float(slot_idx + 1) / float(dim + 1) for dim in range(int(channels))])
        matrices.append(matrix_rows)
        allocations.append(allocation_rows)
        selected_times.append(time_values)
        centers.append(center_values)
        gates.append(gate_values)
        tokens.append(token_rows)
    return {
        "acquisition_matrix": matrices,
        "allocation": allocations,
        "valid_mask": [[1 for _ in range(int(time))] for _ in range(int(batch_size))],
        "valid_lengths": [int(time) for _ in range(int(batch_size))],
        "selected_mask": [[1 for _ in range(int(slots))] for _ in range(int(batch_size))],
        "selected_times": selected_times,
        "centers": centers,
        "gates": gates,
        "selected_tokens": tokens,
    }


def run_synthetic_diagnostic(
    *,
    output_json: str | Path | None = None,
    batch_size: int = 1,
    slots: int = 4,
    time: int = 8,
    channels: int = 3,
) -> dict[str, Any]:
    row = {
        "sample_ids": [f"synthetic_{idx}" for idx in range(int(batch_size))],
        "snapshot_id": "synthetic",
        "reader_out": synthetic_reader_out(batch_size=batch_size, slots=slots, time=time, channels=channels),
        "bridge_output": [[[0.1 * (idx + 1) for idx in range(int(slots))]]],
    }
    return build_summary(_diagnose_row(row, row_idx=0, snapshot_label="synthetic"), source="synthetic", output_json=output_json)


def run_jsonl_diagnostic(
    input_jsonl: str | Path,
    *,
    output_json: str | Path | None = None,
    snapshot_label: str = "snapshot",
    limit: int | None = None,
) -> dict[str, Any]:
    rows = read_jsonl(input_jsonl)
    if limit is not None:
        if int(limit) <= 0:
            raise ValueError("limit must be positive when provided")
        rows = rows[: int(limit)]
    per_sample: list[dict[str, Any]] = []
    for row_idx, row in enumerate(rows):
        per_sample.extend(_diagnose_row(row, row_idx=row_idx, snapshot_label=snapshot_label))
    return build_summary(per_sample, source="jsonl", input_jsonl=input_jsonl, output_json=output_json)


class _CaptureHook:
    def __init__(self, *, keep_output: bool = False) -> None:
        self.latest = None
        self.keep_output = bool(keep_output)

    def __call__(self, _module, _inputs, output) -> None:
        self.latest = output if self.keep_output else _tensor_tree_norm_summary(output)

    def pop(self) -> Any:
        if self.latest is None:
            raise RuntimeError("diagnostic hook did not capture output")
        out = self.latest
        self.latest = None
        return out


def _tensor_tree_norm_summary(value: Any) -> dict[str, Any]:
    torch = sys.modules.get("torch")
    norms: list[float] = []

    def visit(item: Any) -> None:
        if torch is not None and getattr(torch, "is_tensor")(item):
            tensor = item.detach().float().cpu()
            if tensor.numel() > 0:
                norms.append(float(torch.linalg.vector_norm(tensor).item()))
        elif isinstance(item, Mapping):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return {"source": "forward_hook_tensor_tree", "tensor_count": len(norms), "norms": norms, "stats": summarize_values(norms)}


def _model_reader_module(model: Any) -> Any:
    module = model.module if hasattr(model, "module") else model
    reader = getattr(module, "pc_ot_mras_reader", None)
    if reader is None:
        raise ValueError("model has no pc_ot_mras_reader")
    return reader


def _model_bridge_module(model: Any) -> Any | None:
    module = model.module if hasattr(model, "module") else model
    for candidate in (getattr(module, "neck", None), getattr(module, "pc_ot_mras_bridge", None)):
        if candidate is not None and "PCOTMRASDetectorBridge" in candidate.__class__.__name__:
            return candidate
    return None


def run_checkpoint_diagnostic(
    *,
    config: str | Path,
    checkpoint: str | Path,
    output_json: str | Path | None = None,
    snapshot_jsonl: str | Path | None = None,
    split: str = "val",
    limit_batches: int = 1,
    device: str = "auto",
    use_ema: bool | None = None,
    use_amp: bool = False,
) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(f"torch is required for checkpoint diagnostics: {exc}") from None

    try:
        from mmengine.config import Config
        from opentad.datasets import build_dataloader, build_dataset
        from opentad.models import build_detector
        from opentad.models.utils.pc_ot_mras_raw_prediction_guard import assert_no_raw_prediction_shortcut_for_pc_ot_mras
        from tools.bata.dump_pc_ot_mras_reader_snapshots import (
            _device_from_arg,
            _load_checkpoint_state,
            _move_batch_to_device,
            sample_ids_from_metas,
        )
    except Exception as exc:
        raise RuntimeError(f"OpenTAD checkpoint diagnostic imports failed: {exc}") from None

    if int(limit_batches) <= 0:
        raise ValueError("limit_batches must be positive")
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

    reader_hook = _CaptureHook(keep_output=True)
    bridge_hook = _CaptureHook(keep_output=False)
    reader_handle = _model_reader_module(model).register_forward_hook(reader_hook)
    bridge_module = _model_bridge_module(model)
    bridge_handle = bridge_module.register_forward_hook(bridge_hook) if bridge_module is not None else None
    per_sample: list[dict[str, Any]] = []
    snapshot_rows: list[dict[str, Any]] = []
    samples_seen = 0
    try:
        for batch_idx, data_dict in enumerate(loader):
            if batch_idx >= int(limit_batches):
                break
            batch = _move_batch_to_device(data_dict, torch_device)
            metas = data_dict.get("metas")
            if not isinstance(metas, (list, tuple)):
                raise ValueError("batch metas must be a list/tuple")
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.float16, enabled=bool(use_amp) and torch_device.type == "cuda"):
                    model.forward_test(batch["inputs"], batch["masks"], metas=metas, infer_cfg=cfg.inference)
            reader_outputs = reader_hook.pop()
            bridge_norm = (None, [])
            if bridge_module is not None and bridge_hook.latest is not None:
                bridge_summary = bridge_hook.pop()
                bridge_norm = ("forward_hook_tensor_tree", bridge_summary.get("norms", []))
            sample_ids = sample_ids_from_metas(metas, seen_count=samples_seen)
            plain_reader = _to_plain(reader_outputs)
            row = {
                "schema_version": SCHEMA_VERSION,
                "sample_ids": sample_ids,
                "snapshot_id": f"epoch_{epoch}" if epoch is not None else Path(checkpoint).stem,
                "epoch": epoch,
                "reader_out": plain_reader,
                "bridge_output_norm_hook": bridge_norm[1],
                "diagnostic_only": True,
                "uses_gt": False,
                "uses_teacher": False,
                "uses_oracle": False,
                "uses_cache": False,
                "uses_raw_prediction": False,
            }
            snapshot_rows.append(row)
            for sample_batch_idx, sample_id in enumerate(sample_ids):
                per_sample.append(
                    diagnose_reader_out_sample(
                        plain_reader,
                        sample_id=sample_id,
                        snapshot_id=str(row["snapshot_id"]),
                        batch_idx=sample_batch_idx,
                        row_bridge_norm=bridge_norm,
                    )
                )
            samples_seen += len(sample_ids)
    finally:
        reader_handle.remove()
        if bridge_handle is not None:
            bridge_handle.remove()

    if snapshot_jsonl is not None:
        write_jsonl(snapshot_jsonl, snapshot_rows)
    return build_summary(per_sample, source="checkpoint", output_json=output_json, snapshot_jsonl=snapshot_jsonl)


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
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "decision": NO_GO,
        "error_type": exc.__class__.__name__,
        "error": str(exc),
        "diagnostic_only": True,
        "metric_claim_allowed": False,
        "paper_claim_allowed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dump read-only PC-OT-MRAS reader/bridge diagnostic statistics.")
    parser.add_argument("--mode", choices=("synthetic", "jsonl", "checkpoint"), default="synthetic")
    parser.add_argument("--input-jsonl")
    parser.add_argument("--output-json")
    parser.add_argument("--snapshot-jsonl")
    parser.add_argument("--snapshot-label", default="snapshot")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--synthetic-batch-size", type=int, default=1)
    parser.add_argument("--synthetic-slots", type=int, default=4)
    parser.add_argument("--synthetic-time", type=int, default=8)
    parser.add_argument("--synthetic-channels", type=int, default=3)
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--limit-batches", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--use-ema", type=_parse_use_ema, default=None)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args(argv)

    try:
        if args.mode == "synthetic":
            summary = run_synthetic_diagnostic(
                output_json=args.output_json,
                batch_size=int(args.synthetic_batch_size),
                slots=int(args.synthetic_slots),
                time=int(args.synthetic_time),
                channels=int(args.synthetic_channels),
            )
        elif args.mode == "jsonl":
            if not args.input_jsonl:
                raise ValueError("--input-jsonl is required in jsonl mode")
            summary = run_jsonl_diagnostic(
                args.input_jsonl,
                output_json=args.output_json,
                snapshot_label=args.snapshot_label,
                limit=args.limit,
            )
        else:
            if not args.config or not args.checkpoint:
                raise ValueError("--config and --checkpoint are required in checkpoint mode")
            summary = run_checkpoint_diagnostic(
                config=args.config,
                checkpoint=args.checkpoint,
                output_json=args.output_json,
                snapshot_jsonl=args.snapshot_jsonl,
                split=args.split,
                limit_batches=int(args.limit_batches),
                device=args.device,
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
