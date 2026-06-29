from __future__ import annotations

import argparse
import html
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.export_pc_ot_mras_hard_positions import (  # noqa: E402
    resolve_pc_ot_mras_hard_positions,
    strict_json_value,
    write_json,
)


SCHEMA_VERSION = "pc_ot_mras_selection_visualization_v0"
SUMMARY_SCHEMA_VERSION = "pc_ot_mras_selection_visualization_summary_v0"
READY = "PC_OT_MRAS_SELECTION_VISUALIZATION_READY"
NO_GO = "PC_OT_MRAS_SELECTION_VISUALIZATION_NO_GO"
MATRIX_PRIORITY = ("acquisition_matrix", "allocation", "transport_prob", "allocation_logits")
DENSE_TRACK_KEYS = (
    "start_logits",
    "end_logits",
    "boundary_logits",
    "body_logits",
    "uncertainty_logits",
    "redundancy_logits",
    "value_logits",
    "risk_logits",
    "p_action",
    "entropy",
    "p_change",
    "margin",
    "boundary_score",
    "action_score",
    "background_score",
    "role_overlap",
)
FORBIDDEN_KEY_TOKENS = (
    "gt",
    "groundtruth",
    "teacher",
    "oracle",
    "cache",
    "featurecache",
    "prediction",
    "predictions",
    "predictioncache",
    "rawprediction",
    "rawpredictions",
    "detectionresult",
    "detectionsresult",
    "resultdetection",
    "resultjson",
    "resultartifact",
    "label",
    "labels",
    "segment",
    "segments",
    "annotation",
    "annotations",
)
ALLOWED_FALSE_GUARD_KEYS = {
    "usesgt",
    "usesteacher",
    "usesoracle",
    "usescache",
    "usesrawprediction",
    "metricclaimallowed",
    "paperclaimallowed",
    "runtimeflopsclaimallowed",
    "deployclaimallowed",
}


def _to_plain(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "tolist"):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    return value


def _normalized_key(key: Any) -> str:
    return "".join(ch for ch in str(key).lower() if ch.isalnum())


def _is_forbidden_key(normalized: str) -> bool:
    for token in FORBIDDEN_KEY_TOKENS:
        if token == "gt":
            if normalized == token or normalized.startswith(token) or normalized.endswith(token):
                return True
            continue
        if token in normalized:
            return True
    return False


def _is_false_guard_value(value: Any) -> bool:
    data = _to_plain(value)
    return data is False or data == 0 or data is None


def _validate_no_forbidden_keys(value: Any, *, path: str = "row") -> None:
    data = _to_plain(value)
    if isinstance(data, Mapping):
        for key, item in data.items():
            normalized = _normalized_key(key)
            if _is_forbidden_key(normalized):
                if normalized not in ALLOWED_FALSE_GUARD_KEYS or not _is_false_guard_value(item):
                    raise ValueError(f"{path}.{key}: forbidden diagnostic input key")
            _validate_no_forbidden_keys(item, path=f"{path}.{key}")
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            _validate_no_forbidden_keys(item, path=f"{path}[{idx}]")


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


def _batch_size_from(reader_out: Mapping[str, Any]) -> int:
    for key in MATRIX_PRIORITY:
        value = reader_out.get(key)
        if value is not None and _depth(value) >= 3:
            return len(_to_plain(value))
    for key in (*DENSE_TRACK_KEYS, "selection_logits", "selection_prob", "soft_selection", "valid_mask"):
        value = reader_out.get(key)
        if value is not None and _depth(value) >= 2:
            return len(_to_plain(value))
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


def _float_row(value: Any, *, name: str) -> list[float]:
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
        raise ValueError(f"{name} must be a non-empty [K,T] matrix")
    out: list[list[float]] = []
    for row_idx, row in enumerate(data):
        if len(row) != width:
            raise ValueError(f"{name} must be rectangular [K,T]")
        out.append([_finite_float(item, name=f"{name}[{row_idx}][{col_idx}]") for col_idx, item in enumerate(row)])
    return out


def _valid_positions(valid_mask: Any, *, dense_len: int | None, valid_len: int | None) -> list[int]:
    mask = _to_plain(valid_mask)
    if isinstance(mask, list):
        positions = []
        for idx, item in enumerate(mask):
            if item not in (0, 1, False, True):
                raise ValueError(f"valid_mask[{idx}] must be binary")
            if bool(item):
                positions.append(int(idx))
        if not positions:
            raise ValueError("valid_mask must contain at least one valid position")
        return positions
    if valid_len is not None:
        if int(valid_len) <= 0:
            raise ValueError("valid_len must be positive")
        return list(range(int(valid_len)))
    if dense_len is not None:
        if int(dense_len) <= 0:
            raise ValueError("dense_len must be positive")
        return list(range(int(dense_len)))
    raise ValueError("one of valid_mask, valid_len, or dense_len is required")


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(float(item) for item in values) / float(len(values)))


def _std(values: Sequence[float]) -> float | None:
    if len(values) <= 1:
        return None
    mean = _mean(values) or 0.0
    return float(math.sqrt(sum((float(item) - mean) ** 2 for item in values) / float(len(values))))


def _entropy(values: Sequence[float]) -> float | None:
    positive = [max(0.0, float(item)) for item in values]
    total = sum(positive)
    if total <= 0.0:
        return None
    probs = [item / total for item in positive if item > 0.0]
    return float(-sum(prob * math.log(prob) for prob in probs))


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("._")
    return slug[:120] or "sample"


def _bin_edges(length: int, bins: int) -> list[tuple[int, int]]:
    bins = max(1, min(int(bins), int(length)))
    edges = []
    for idx in range(bins):
        start = int(math.floor(idx * length / bins))
        end = int(math.floor((idx + 1) * length / bins))
        if end <= start:
            end = start + 1
        edges.append((start, min(end, length)))
    return edges


def _downsample_matrix(matrix: Sequence[Sequence[float]], max_rows: int, max_cols: int) -> list[list[float]]:
    rows = len(matrix)
    cols = len(matrix[0]) if rows else 0
    if rows <= 0 or cols <= 0:
        return []
    row_edges = _bin_edges(rows, max_rows)
    col_edges = _bin_edges(cols, max_cols)
    out: list[list[float]] = []
    for row_start, row_end in row_edges:
        out_row: list[float] = []
        for col_start, col_end in col_edges:
            values = [
                float(matrix[row_idx][col_idx])
                for row_idx in range(row_start, row_end)
                for col_idx in range(col_start, col_end)
            ]
            out_row.append(_mean(values) or 0.0)
        out.append(out_row)
    return out


def _downsample_vector(values: Sequence[float], max_cols: int) -> list[float]:
    if not values:
        return []
    return [
        _mean([float(values[idx]) for idx in range(start, end)]) or 0.0
        for start, end in _bin_edges(len(values), max_cols)
    ]


def _value_range(values: Sequence[float]) -> tuple[float, float]:
    finite = [float(item) for item in values if math.isfinite(float(item))]
    if not finite:
        return 0.0, 1.0
    lo = min(finite)
    hi = max(finite)
    if hi <= lo:
        return lo, lo + 1.0
    return lo, hi


def _blend(left: tuple[int, int, int], right: tuple[int, int, int], frac: float) -> str:
    f = min(max(float(frac), 0.0), 1.0)
    rgb = tuple(int(round(left[idx] + (right[idx] - left[idx]) * f)) for idx in range(3))
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def _heat_color(value: float, lo: float, hi: float) -> str:
    frac = 0.0 if hi <= lo else (float(value) - lo) / (hi - lo)
    frac = min(max(frac, 0.0), 1.0)
    stops = (
        (0.00, (246, 248, 250)),
        (0.35, (116, 169, 207)),
        (0.70, (67, 147, 195)),
        (1.00, (215, 84, 63)),
    )
    for idx in range(1, len(stops)):
        left_pos, left_rgb = stops[idx - 1]
        right_pos, right_rgb = stops[idx]
        if frac <= right_pos:
            local = (frac - left_pos) / max(right_pos - left_pos, 1.0e-9)
            return _blend(left_rgb, right_rgb, local)
    return _blend(stops[-2][1], stops[-1][1], 1.0)


def _svg_rect(x: float, y: float, width: float, height: float, fill: str, *, opacity: float = 1.0) -> str:
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" '
        f'fill="{fill}" opacity="{opacity:.3f}"/>'
    )


def _svg_text(x: float, y: float, text: str, *, size: int = 12, weight: str = "400") -> str:
    escaped = html.escape(str(text))
    return f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" font-weight="{weight}" fill="#202124">{escaped}</text>'


def _column_scores(matrix: Sequence[Sequence[float]] | None, dense_len: int) -> list[float]:
    if matrix:
        width = len(matrix[0])
        return [sum(max(0.0, float(row[col])) for row in matrix) for col in range(width)]
    return [0.0 for _ in range(dense_len)]


def _extract_matrix(reader_out: Mapping[str, Any], batch_idx: int, batch_size: int) -> tuple[str | None, list[list[float]] | None]:
    for key in MATRIX_PRIORITY:
        if key not in reader_out:
            continue
        sampled = _sample(reader_out[key], batch_idx, batch_size)
        return key, _float_matrix(sampled, name=key)
    return None, None


def _extract_track(reader_out: Mapping[str, Any], key: str, batch_idx: int, batch_size: int) -> list[float] | None:
    if key not in reader_out:
        return None
    sampled = _sample(reader_out[key], batch_idx, batch_size)
    return _float_row(sampled, name=key)


def _resolve_hard_positions_for_reader(
    reader_out: Mapping[str, Any],
    *,
    batch_idx: int,
    batch_size: int,
    budget: int,
    sample_id: str,
    dense_len: int | None,
    valid_len: int | None,
) -> list[int]:
    sample_reader_out = {
        key: [_sample(value, batch_idx, batch_size)]
        for key, value in reader_out.items()
    }
    rows = resolve_pc_ot_mras_hard_positions(
        sample_reader_out,
        budget=int(budget),
        sample_ids=[sample_id],
        dense_len=dense_len,
        valid_len=valid_len,
    )
    return [int(pos) for pos in rows[0]["selected_positions"]]


def _explicit_positions(value: Any, *, name: str) -> list[int] | None:
    data = _to_plain(value)
    if data is None:
        return None
    if not isinstance(data, list):
        raise ValueError(f"{name} must be a list")
    out: list[int] = []
    for idx, item in enumerate(data):
        if isinstance(item, bool):
            raise ValueError(f"{name}[{idx}] must be an integer")
        try:
            pos = int(item)
        except (TypeError, ValueError):
            raise ValueError(f"{name}[{idx}] must be an integer") from None
        if pos >= 0:
            out.append(pos)
    return sorted(set(out))


def _sample_ids(row: Mapping[str, Any], batch_size: int, row_idx: int) -> list[str]:
    raw_ids = row.get("sample_ids")
    if isinstance(raw_ids, list):
        if len(raw_ids) != batch_size:
            raise ValueError("sample_ids length must match inferred batch size")
        return [str(item) for item in raw_ids]
    raw_id = row.get("sample_id", f"row_{row_idx}")
    if batch_size == 1:
        return [str(raw_id)]
    return [f"{raw_id}|batch{idx}" for idx in range(batch_size)]


def _expand_input_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    default_budget: int | None,
    snapshot_label: str,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for row_idx, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"row {row_idx}: JSONL row must be an object")
        _validate_no_forbidden_keys(row, path=f"row[{row_idx}]")

        if row.get("schema_version") == "pc_ot_mras_hard_positions_v0":
            selected = _explicit_positions(row.get("selected_positions"), name="selected_positions") or []
            dense_len = int(row.get("dense_len", len(row.get("selected_mask", [])) or (max(selected) + 1 if selected else 0)))
            valid_len = int(row.get("valid_len", dense_len))
            samples.append(
                {
                    "sample_id": str(row.get("sample_id", f"row_{row_idx}")),
                    "snapshot_id": str(row.get("snapshot_id", snapshot_label)),
                    "epoch": row.get("epoch"),
                    "matrix_key": None,
                    "matrix": None,
                    "dense_len": dense_len,
                    "valid_len": valid_len,
                    "valid_positions": list(range(valid_len)),
                    "selected_positions": selected,
                    "selected_source": "hard_row_selected_positions",
                    "tracks": {},
                    "gates": None,
                    "centers": None,
                }
            )
            continue

        reader_out = row.get("reader_out", row)
        if not isinstance(reader_out, Mapping):
            raise ValueError(f"row {row_idx}: reader_out must be an object")
        batch_size = _batch_size_from(reader_out)
        ids = _sample_ids(row, batch_size, row_idx)
        for batch_idx in range(batch_size):
            sample_id = ids[batch_idx]
            matrix_key, matrix = _extract_matrix(reader_out, batch_idx, batch_size)
            valid_mask = _sample(reader_out.get("valid_mask"), batch_idx, batch_size) if "valid_mask" in reader_out else None
            dense_len = row.get("dense_len")
            valid_len = row.get("valid_len")
            if matrix is not None:
                dense_len = len(matrix[0]) if dense_len is None else int(dense_len)
            elif valid_mask is not None:
                dense_len = len(_to_plain(valid_mask)) if dense_len is None else int(dense_len)
            else:
                dense_len = int(dense_len) if dense_len is not None else None
            valid_positions = _valid_positions(
                valid_mask,
                dense_len=int(dense_len) if dense_len is not None else None,
                valid_len=int(valid_len) if valid_len is not None else None,
            )
            valid_len_i = int(valid_len) if valid_len is not None else len(valid_positions)
            dense_len_i = int(dense_len) if dense_len is not None else (max(valid_positions) + 1)

            selected = None
            selected_source = "resolved_from_reader_out"
            for key in ("hard_selected_positions", "selected_positions", "selected_dense_positions"):
                if key in reader_out:
                    selected = _explicit_positions(_sample(reader_out[key], batch_idx, batch_size), name=key)
                    selected_source = key
                    break
            budget = row.get("budget", default_budget)
            if selected is None:
                if budget is None:
                    slot_count = len(matrix) if matrix is not None else valid_len_i
                    budget = min(int(slot_count), int(valid_len_i))
                selected = _resolve_hard_positions_for_reader(
                    reader_out,
                    batch_idx=batch_idx,
                    batch_size=batch_size,
                    budget=int(budget),
                    sample_id=sample_id,
                    dense_len=dense_len_i,
                    valid_len=valid_len_i,
                )

            tracks = {
                key: values
                for key in DENSE_TRACK_KEYS
                for values in [_extract_track(reader_out, key, batch_idx, batch_size)]
                if values is not None
            }
            gates = _extract_track(reader_out, "gates", batch_idx, batch_size) if "gates" in reader_out else None
            centers = _extract_track(reader_out, "centers", batch_idx, batch_size) if "centers" in reader_out else None
            samples.append(
                {
                    "sample_id": sample_id,
                    "snapshot_id": str(row.get("snapshot_id", snapshot_label)),
                    "epoch": row.get("epoch"),
                    "matrix_key": matrix_key,
                    "matrix": matrix,
                    "dense_len": dense_len_i,
                    "valid_len": valid_len_i,
                    "valid_positions": valid_positions,
                    "selected_positions": selected,
                    "selected_source": selected_source,
                    "tracks": tracks,
                    "gates": gates,
                    "centers": centers,
                }
            )
    return samples


def _selected_gap_stats(selected: Sequence[int]) -> dict[str, Any]:
    if len(selected) <= 1:
        return {"gap_min": None, "gap_mean": None, "gap_std": None, "gap_max": None}
    gaps = [float(right - left) for left, right in zip(selected, selected[1:])]
    return {
        "gap_min": min(gaps),
        "gap_mean": _mean(gaps),
        "gap_std": _std(gaps),
        "gap_max": max(gaps),
    }


def _render_vector_track(
    parts: list[str],
    *,
    label: str,
    values: Sequence[float],
    y: float,
    left: float,
    cell_w: float,
    max_time_bins: int,
    height: float = 10.0,
) -> float:
    track = _downsample_vector(values, max_time_bins)
    lo, hi = _value_range(track)
    parts.append(_svg_text(12, y + height - 1, label, size=10))
    for idx, value in enumerate(track):
        parts.append(_svg_rect(left + idx * cell_w, y, cell_w, height, _heat_color(value, lo, hi)))
    return y + height + 4.0


def _render_selection_svg(
    sample: Mapping[str, Any],
    svg_path: Path,
    *,
    max_time_bins: int,
    max_slot_bins: int,
) -> dict[str, Any]:
    sample_id = str(sample["sample_id"])
    snapshot_id = str(sample["snapshot_id"])
    matrix = sample.get("matrix")
    matrix_key = sample.get("matrix_key")
    dense_len = int(sample["dense_len"])
    valid_len = int(sample["valid_len"])
    selected = [int(pos) for pos in sample["selected_positions"]]
    tracks: Mapping[str, Sequence[float]] = sample.get("tracks", {})  # type: ignore[assignment]
    gates = sample.get("gates")
    centers = sample.get("centers")

    time_bins = min(max_time_bins, max(1, dense_len))
    cell_w = 5.0
    cell_h = 4.0
    left = 132.0
    top = 34.0
    parts: list[str] = []

    matrix_rows = 0
    if matrix:
        reduced = _downsample_matrix(matrix, max_slot_bins, max_time_bins)
        flat = [value for row in reduced for value in row]
        lo, hi = _value_range(flat)
        matrix_rows = len(reduced)
        parts.append(_svg_text(12, top + 12, f"{matrix_key} slots", size=10))
        for row_idx, row in enumerate(reduced):
            for col_idx, value in enumerate(row):
                parts.append(_svg_rect(left + col_idx * cell_w, top + row_idx * cell_h, cell_w, cell_h, _heat_color(value, lo, hi)))
        heatmap_bottom = top + matrix_rows * cell_h
    else:
        heatmap_bottom = top

    selected_set = set(selected)
    selected_bins = [0.0 for _ in range(time_bins)]
    for pos in selected_set:
        if 0 <= pos < dense_len:
            bin_idx = min(time_bins - 1, int(pos * time_bins / max(dense_len, 1)))
            selected_bins[bin_idx] = 1.0

    y = heatmap_bottom + 16.0
    if matrix:
        scores = _column_scores(matrix, dense_len)
        y = _render_vector_track(
            parts,
            label="marginal",
            values=scores,
            y=y,
            left=left,
            cell_w=cell_w,
            max_time_bins=max_time_bins,
            height=12.0,
        )
    y = _render_vector_track(
        parts,
        label="selected",
        values=selected_bins,
        y=y,
        left=left,
        cell_w=cell_w,
        max_time_bins=max_time_bins,
        height=12.0,
    )
    for key, values in tracks.items():
        y = _render_vector_track(
            parts,
            label=key.replace("_logits", ""),
            values=values[:dense_len],
            y=y,
            left=left,
            cell_w=cell_w,
            max_time_bins=max_time_bins,
            height=10.0,
        )

    axis_y = y + 8.0
    parts.append(f'<line x1="{left:.2f}" y1="{axis_y:.2f}" x2="{left + time_bins * cell_w:.2f}" y2="{axis_y:.2f}" stroke="#5f6368" stroke-width="1"/>')
    for frac, text in ((0.0, "0"), (0.5, str(dense_len // 2)), (1.0, str(max(dense_len - 1, 0)))):
        x = left + frac * time_bins * cell_w
        parts.append(f'<line x1="{x:.2f}" y1="{axis_y:.2f}" x2="{x:.2f}" y2="{axis_y + 5:.2f}" stroke="#5f6368" stroke-width="1"/>')
        parts.append(_svg_text(x - 8, axis_y + 18, text, size=9))
    invalid_x = left + (valid_len / max(dense_len, 1)) * time_bins * cell_w
    if valid_len < dense_len:
        parts.append(_svg_rect(invalid_x, top, left + time_bins * cell_w - invalid_x, max(axis_y - top, 1.0), "#f1f3f4", opacity=0.55))

    title = f"{snapshot_id} | {sample_id}"
    subtitle = (
        f"matrix={matrix_key or 'none'} selected={len(selected)} valid={valid_len}/{dense_len} "
        f"source={sample.get('selected_source')}"
    )
    width = int(left + time_bins * cell_w + 24)
    height = int(axis_y + 30)
    header = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>',
        _svg_text(12, 18, title, size=14, weight="700"),
        _svg_text(12, 31, subtitle, size=10),
    ]
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text("\n".join([*header, *parts, "</svg>", ""]), encoding="utf-8")

    scores = _column_scores(matrix, dense_len) if matrix else [1.0 if pos in selected_set else 0.0 for pos in range(dense_len)]
    selected_scores = [scores[pos] for pos in selected if 0 <= pos < len(scores)]
    center_monotonic = None
    if centers is not None and len(centers) > 1:
        center_monotonic = all(float(right) > float(left_v) for left_v, right in zip(centers, centers[1:]))

    return {
        "sample_id": sample_id,
        "snapshot_id": snapshot_id,
        "epoch": sample.get("epoch"),
        "svg_path": str(svg_path),
        "matrix_key": matrix_key,
        "slot_count": len(matrix) if matrix else 0,
        "dense_len": dense_len,
        "valid_len": valid_len,
        "selected_count": len(selected),
        "selected_source": sample.get("selected_source"),
        "selected_positions_preview": selected[:16],
        "selected_coverage_ratio": None if valid_len <= 0 else float(len(selected) / float(valid_len)),
        "selected_score_mean": _mean(selected_scores),
        "marginal_entropy": _entropy(scores),
        "gate_mean": _mean(gates) if gates is not None else None,
        "gate_std": _std(gates) if gates is not None else None,
        "center_monotonic": center_monotonic,
        **_selected_gap_stats(selected),
    }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).expanduser().open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError(f"line {line_no}: JSONL row must be an object")
            rows.append(payload)
    if not rows:
        raise ValueError(f"JSONL has no rows: {path}")
    return rows


def run_selection_visualization(
    input_jsonl: str | Path,
    *,
    output_dir: str | Path,
    summary_json: str | Path | None = None,
    budget: int | None = None,
    snapshot_label: str = "snapshot",
    limit: int | None = None,
    max_time_bins: int = 192,
    max_slot_bins: int = 96,
) -> dict[str, Any]:
    if budget is not None and int(budget) <= 0:
        raise ValueError("budget must be positive when provided")
    if int(max_time_bins) <= 0 or int(max_slot_bins) <= 0:
        raise ValueError("max_time_bins and max_slot_bins must be positive")

    rows = read_jsonl(input_jsonl)
    samples = _expand_input_rows(rows, default_budget=budget, snapshot_label=snapshot_label)
    if limit is not None:
        if int(limit) <= 0:
            raise ValueError("limit must be positive when provided")
        samples = samples[: int(limit)]
    if not samples:
        raise ValueError("no samples available for visualization")

    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    per_sample: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        filename = f"{idx:04d}_{_safe_slug(str(sample['snapshot_id']))}_{_safe_slug(str(sample['sample_id']))}.svg"
        per_sample.append(
            _render_selection_svg(
                sample,
                out_dir / filename,
                max_time_bins=int(max_time_bins),
                max_slot_bins=int(max_slot_bins),
            )
        )

    selected_counts = [int(item["selected_count"]) for item in per_sample]
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "decision": READY,
        "input_jsonl": str(input_jsonl),
        "output_dir": str(out_dir),
        "snapshot_label": str(snapshot_label),
        "sample_count": len(per_sample),
        "generated_svg_count": len(per_sample),
        "selected_count_mean": _mean([float(item) for item in selected_counts]),
        "selected_count_min": min(selected_counts),
        "selected_count_max": max(selected_counts),
        "matrix_keys": sorted({str(item["matrix_key"]) for item in per_sample if item["matrix_key"] is not None}),
        "per_sample": per_sample,
        "visualization_schema_version": SCHEMA_VERSION,
        "diagnostic_only": True,
        "uses_gt": False,
        "uses_teacher": False,
        "uses_oracle": False,
        "uses_raw_prediction": False,
        "uses_cache": False,
        "metric_claim_allowed": False,
        "paper_claim_allowed": False,
        "runtime_flops_claim_allowed": False,
        "deploy_claim_allowed": False,
        "remote_sync_allowed": False,
        "remote_precheck_allowed": False,
        "slurm_gpu_allowed": False,
        "tools_train_allowed": False,
        "tools_test_allowed": False,
        "detector_map_allowed": False,
    }
    if summary_json is not None:
        write_json(summary_json, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render PC-OT-MRAS reader attention/allocation heatmaps and selected-position timelines."
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--summary-json")
    parser.add_argument("--budget", type=int)
    parser.add_argument("--snapshot-label", default="snapshot")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-time-bins", type=int, default=192)
    parser.add_argument("--max-slot-bins", type=int, default=96)
    args = parser.parse_args(argv)

    try:
        summary = run_selection_visualization(
            args.input_jsonl,
            output_dir=args.output_dir,
            summary_json=args.summary_json,
            budget=args.budget,
            snapshot_label=args.snapshot_label,
            limit=args.limit,
            max_time_bins=args.max_time_bins,
            max_slot_bins=args.max_slot_bins,
        )
    except Exception as exc:  # pragma: no cover - CLI guard
        print(json.dumps({"schema_version": SUMMARY_SCHEMA_VERSION, "decision": NO_GO, "error": str(exc)}))
        return 1

    print(json.dumps(strict_json_value(summary), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
