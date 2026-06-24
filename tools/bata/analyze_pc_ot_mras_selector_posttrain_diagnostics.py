from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.export_pc_ot_mras_hard_positions import strict_json_value, write_json  # noqa: E402


SCHEMA_VERSION = "pc_ot_mras_selector_posttrain_diagnostic_v0"
READY = "PC_OT_MRAS_SELECTOR_POSTTRAIN_DIAGNOSTIC_READY"
READY_WITH_WARNINGS = "PC_OT_MRAS_SELECTOR_POSTTRAIN_DIAGNOSTIC_READY_WITH_WARNINGS"
NO_GO = "PC_OT_MRAS_SELECTOR_POSTTRAIN_DIAGNOSTIC_NO_GO"

POSITION_KEYS = (
    "selected_dense_indices",
    "selected_dense_positions",
    "selected_positions",
    "hard_selected_positions",
    "irregular_selected_positions",
    "bata_selected_dense_indices",
)
VALID_LEN_KEYS = (
    "valid_len",
    "dense_valid_len",
    "irregular_dense_valid_len",
    "irregular_selected_valid_len",
    "valid_lengths",
    "dense_len",
)
DENSE_SCORE_KEYS = (
    "selector_scores",
    "selection_scores",
    "dense_scores",
    "scores_dense",
    "selection_logits",
    "selection_prob",
    "soft_selection",
    "value_logits",
    "risk_logits",
    "boundary_logits",
    "acq_logits",
)
MATRIX_SCORE_KEYS = ("acquisition_matrix", "allocation", "transport_prob", "allocation_logits")
ROLE_ID_TO_NAME = {
    0: "fallback_scaffold",
    1: "start_detail",
    2: "end_detail",
    3: "short_action_body",
    4: "neighbor_separator",
    5: "uncertainty_probe",
}


def _to_plain(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "tolist"):
        return _to_plain(value.detach().cpu().tolist())
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return _to_plain(value.tolist())
        except TypeError:
            pass
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return _to_plain(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, Mapping):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    return value


def _mean(values: Sequence[float]) -> float | None:
    finite = [float(item) for item in values if math.isfinite(float(item))]
    if not finite:
        return None
    return float(sum(finite) / float(len(finite)))


def _percentile(values: Sequence[float], q: float) -> float | None:
    finite = sorted(float(item) for item in values if math.isfinite(float(item)))
    if not finite:
        return None
    if len(finite) == 1:
        return finite[0]
    rank = (len(finite) - 1) * float(q)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return finite[int(rank)]
    return finite[lower] + (finite[upper] - finite[lower]) * (rank - lower)


def _round_float(value: Any, digits: int = 6) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return round(out, int(digits))


def _distribution(values: Sequence[int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = str(int(value))
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda item: int(item[0])))


def _as_finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be numeric") from None
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _as_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and math.isfinite(value) and float(value).is_integer():
        return int(value)
    raise ValueError(f"{name} must be an integer")


def _as_positions(value: Any, *, name: str) -> list[int]:
    data = _to_plain(value)
    if data is None:
        return []
    if not isinstance(data, list):
        raise ValueError(f"{name} must be a list")
    out: list[int] = []
    for idx, item in enumerate(data):
        if isinstance(item, bool):
            raise ValueError(f"{name}[{idx}] must be numeric")
        if isinstance(item, (int, float)):
            score = _as_finite_float(item, name=f"{name}[{idx}]")
            out.append(int(math.floor(score + 0.5)))
            continue
        raise ValueError(f"{name}[{idx}] must be numeric")
    return out


def _as_float_list(value: Any, *, name: str) -> list[float] | None:
    data = _to_plain(value)
    if data is None:
        return None
    if not isinstance(data, list):
        return None
    out: list[float] = []
    for idx, item in enumerate(data):
        if isinstance(item, bool):
            return None
        try:
            score = float(item)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(score):
            return None
        out.append(score)
    return out


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
    if isinstance(data, list) and batch_size == 1 and data and isinstance(data[0], list):
        return data[0]
    return data


def _batch_size_from(reader_out: Mapping[str, Any]) -> int:
    for key in MATRIX_SCORE_KEYS:
        value = reader_out.get(key)
        if value is not None and _depth(value) >= 3:
            return len(_to_plain(value))
    for key in (*POSITION_KEYS, *DENSE_SCORE_KEYS, "valid_mask", "role_ids", "packet_roles"):
        value = reader_out.get(key)
        if value is not None and _depth(value) >= 2:
            return len(_to_plain(value))
    value = reader_out.get("valid_lengths")
    if value is not None and _depth(value) == 1:
        return len(_to_plain(value))
    return 1


def _get_first(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _nested_get(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    found = _get_first(mapping, keys)
    if found is not None:
        return found
    for nested_key in ("meta", "metadata", "metas", "pc_ot_mras_bridge", "dynamic_budget_plan"):
        nested = mapping.get(nested_key)
        if isinstance(nested, Mapping):
            found = _get_first(nested, keys)
            if found is not None:
                return found
    return None


def _bool_or_none(value: Any) -> bool | None:
    data = _to_plain(value)
    if isinstance(data, bool):
        return bool(data)
    if isinstance(data, (int, float)) and data in (0, 1):
        return bool(data)
    if isinstance(data, str):
        lowered = data.lower().strip()
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no"):
            return False
    return None


def _find_non_finite(value: Any, *, path: str = "$", limit: int = 32) -> tuple[int, list[dict[str, str]]]:
    data = _to_plain(value)
    if isinstance(data, Mapping):
        total = 0
        examples: list[dict[str, str]] = []
        for key, item in data.items():
            count, item_examples = _find_non_finite(item, path=f"{path}.{key}", limit=max(0, limit - len(examples)))
            total += count
            examples.extend(item_examples[: max(0, limit - len(examples))])
        return total, examples
    if isinstance(data, list):
        total = 0
        examples = []
        for idx, item in enumerate(data):
            count, item_examples = _find_non_finite(item, path=f"{path}[{idx}]", limit=max(0, limit - len(examples)))
            total += count
            examples.extend(item_examples[: max(0, limit - len(examples))])
        return total, examples
    if isinstance(data, (int, float)) and not isinstance(data, bool):
        score = float(data)
        if not math.isfinite(score):
            return 1, [{"path": path, "value": "nan" if math.isnan(score) else ("inf" if score > 0 else "-inf")}]
    return 0, []


def _coerce_segments(row: Mapping[str, Any]) -> list[tuple[float, float]]:
    raw = _nested_get(row, ("gt_segments", "segments", "annotations"))
    if raw is None:
        return []
    raw = _to_plain(raw)
    if not isinstance(raw, list):
        return []
    out: list[tuple[float, float]] = []
    for item in raw:
        if isinstance(item, Mapping):
            if "segment" in item:
                item = item["segment"]
            elif "start" in item and "end" in item:
                item = [item["start"], item["end"]]
        if not isinstance(item, list) or len(item) != 2:
            continue
        try:
            start = _as_finite_float(item[0], name="segment_start")
            end = _as_finite_float(item[1], name="segment_end")
        except ValueError:
            continue
        if end < start:
            start, end = end, start
        out.append((start, end))
    return out


def _is_inside_any_segment(position: float, segments: Sequence[tuple[float, float]]) -> bool:
    return any(float(start) <= float(position) <= float(end) for start, end in segments)


def _is_boundary_near(position: float, boundaries: Sequence[float], radius: float) -> bool:
    return any(abs(float(position) - float(boundary)) <= float(radius) for boundary in boundaries)


def _boundary_diagnostics(selected: Sequence[int], row: Mapping[str, Any], *, default_radius: float) -> dict[str, Any]:
    segments = _coerce_segments(row)
    if not segments:
        return {"provided": False, "boundary_radius": float(default_radius)}
    radius_value = _nested_get(row, ("boundary_radius",))
    radius = float(default_radius if radius_value is None else _as_finite_float(radius_value, name="boundary_radius"))
    boundaries = [point for segment in segments for point in segment]
    near_selected = [pos for pos in selected if _is_boundary_near(float(pos), boundaries, radius)]
    supported_boundaries = [
        boundary for boundary in boundaries if any(abs(float(pos) - float(boundary)) <= radius for pos in selected)
    ]
    action_selected = [pos for pos in selected if _is_inside_any_segment(float(pos), segments)]
    interior_selected = [pos for pos in action_selected if not _is_boundary_near(float(pos), boundaries, radius)]
    return {
        "provided": True,
        "boundary_radius": int(radius) if float(radius).is_integer() else radius,
        "gt_segment_count": len(segments),
        "boundary_count": len(boundaries),
        "boundary_support_count": len(supported_boundaries),
        "boundary_support_rate": _round_float(len(supported_boundaries) / len(boundaries) if boundaries else None),
        "boundary_near_selected_count": len(near_selected),
        "boundary_near_selected_rate": _round_float(len(near_selected) / len(selected) if selected else None),
        "action_selected_count": len(action_selected),
        "action_selected_rate": _round_float(len(action_selected) / len(selected) if selected else None),
        "interior_selected_count": len(interior_selected),
        "interior_selected_rate": _round_float(len(interior_selected) / len(selected) if selected else None),
    }


def _role_name_from_id(value: Any) -> str:
    try:
        role_id = int(value)
    except (TypeError, ValueError):
        return str(value)
    return ROLE_ID_TO_NAME.get(role_id, f"role_{role_id}")


def _extract_roles(row: Mapping[str, Any], selected: Sequence[int]) -> tuple[list[str], str]:
    for key in ("packet_roles", "roles", "role_names", "selected_roles"):
        value = _nested_get(row, (key,))
        data = _to_plain(value)
        if isinstance(data, list) and len(data) == len(selected):
            return [str(item) for item in data], key

    role_ids = _nested_get(row, ("role_ids",))
    role_data = _to_plain(role_ids)
    if isinstance(role_data, list) and len(role_data) == len(selected):
        return [_role_name_from_id(item) for item in role_data], "role_ids"

    role_meta = _nested_get(row, ("role_round_metadata",))
    meta_data = _to_plain(role_meta)
    if isinstance(meta_data, list) and len(meta_data) == len(selected):
        names: list[str] = []
        for item in meta_data:
            if isinstance(item, Mapping):
                if "role" in item:
                    names.append(str(item["role"]))
                elif "role_name" in item:
                    names.append(str(item["role_name"]))
                elif "role_id" in item:
                    names.append(_role_name_from_id(item["role_id"]))
                else:
                    names.append("unknown")
            else:
                names.append(str(item))
        return names, "role_round_metadata"

    return [], "unavailable"


def _classify_role_name(name: str) -> str:
    normalized = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(name)).strip("_")
    if any(token in normalized for token in ("start", "end", "boundary")):
        return "boundary"
    if any(token in normalized for token in ("interior", "body", "action", "value")):
        return "interior"
    if any(token in normalized for token in ("background", "separator")):
        return "background"
    if any(token in normalized for token in ("coverage", "fallback", "scaffold")):
        return "coverage"
    if any(token in normalized for token in ("uncertain", "uncertainty", "probe", "risk")):
        return "other"
    return "other"


def _infer_roles_from_segments(
    selected: Sequence[int],
    row: Mapping[str, Any],
    *,
    default_radius: float,
) -> tuple[list[str], str]:
    segments = _coerce_segments(row)
    if not segments:
        return [], "unavailable"
    radius_value = _nested_get(row, ("boundary_radius",))
    radius = float(default_radius if radius_value is None else _as_finite_float(radius_value, name="boundary_radius"))
    boundaries = [point for segment in segments for point in segment]
    roles: list[str] = []
    for pos in selected:
        if _is_boundary_near(float(pos), boundaries, radius):
            roles.append("boundary")
        elif _is_inside_any_segment(float(pos), segments):
            roles.append("interior")
        else:
            roles.append("background")
    return roles, "gt_segments_inferred"


def _packet_role_diagnostics(
    selected: Sequence[int],
    row: Mapping[str, Any],
    *,
    default_radius: float,
) -> dict[str, Any]:
    roles, source = _extract_roles(row, selected)
    if not roles:
        roles, source = _infer_roles_from_segments(selected, row, default_radius=default_radius)
    counts = {"boundary": 0, "interior": 0, "background": 0, "coverage": 0, "other": 0}
    for role in roles:
        category = _classify_role_name(role)
        counts[category] = counts.get(category, 0) + 1
    total = len(roles)
    return {
        "available": bool(roles),
        "source": source,
        "count": total,
        "raw_roles_preview": roles[:16],
        "counts": counts,
        "boundary_ratio": _round_float(counts["boundary"] / total if total else None),
        "interior_ratio": _round_float(counts["interior"] / total if total else None),
        "action_ratio": _round_float(counts["interior"] / total if total else None),
        "background_ratio": _round_float(counts["background"] / total if total else None),
        "coverage_ratio": _round_float(counts["coverage"] / total if total else None),
        "other_ratio": _round_float(counts["other"] / total if total else None),
    }


def _dense_scores_from_matrix(row: Mapping[str, Any]) -> list[float] | None:
    for key in MATRIX_SCORE_KEYS:
        matrix = _to_plain(_nested_get(row, (key,)))
        if not isinstance(matrix, list) or not matrix or not all(isinstance(item, list) for item in matrix):
            continue
        width = len(matrix[0])
        if width <= 0:
            continue
        scores = [0.0 for _ in range(width)]
        try:
            for matrix_row in matrix:
                if len(matrix_row) != width:
                    return None
                for idx, value in enumerate(matrix_row):
                    scores[idx] += _as_finite_float(value, name=f"{key}[{idx}]")
        except ValueError:
            return None
        return scores
    return None


def _extract_scores(row: Mapping[str, Any], selected: Sequence[int], valid_len: int) -> tuple[list[float] | None, list[float] | None, str]:
    for key in DENSE_SCORE_KEYS:
        scores = _as_float_list(_nested_get(row, (key,)), name=key)
        if not scores:
            continue
        if len(scores) >= max(valid_len, max(selected) + 1 if selected else 0):
            return scores, None, key
        if len(scores) == len(selected):
            return None, scores, key
    matrix_scores = _dense_scores_from_matrix(row)
    if matrix_scores is not None:
        return matrix_scores, None, "matrix_column_sum"
    return None, None, "unavailable"


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = _mean(xs)
    y_mean = _mean(ys)
    if x_mean is None or y_mean is None:
        return None
    x_centered = [float(x) - x_mean for x in xs]
    y_centered = [float(y) - y_mean for y in ys]
    numerator = sum(x * y for x, y in zip(x_centered, y_centered))
    x_norm = math.sqrt(sum(x * x for x in x_centered))
    y_norm = math.sqrt(sum(y * y for y in y_centered))
    if x_norm == 0.0 or y_norm == 0.0:
        return None
    return numerator / (x_norm * y_norm)


def _proposal_rank_diagnostics(row: Mapping[str, Any]) -> dict[str, Any]:
    proposals = _nested_get(row, ("proposals", "result_proposals"))
    data = _to_plain(proposals)
    if not isinstance(data, list) or not data:
        return {"available": False}
    pairs: list[tuple[float, float]] = []
    for item in data:
        if not isinstance(item, Mapping):
            continue
        if "score" not in item or "iou" not in item:
            continue
        try:
            pairs.append(
                (
                    _as_finite_float(item["score"], name="proposal.score"),
                    _as_finite_float(item["iou"], name="proposal.iou"),
                )
            )
        except ValueError:
            continue
    if not pairs:
        return {"available": False}
    top_k_value = _nested_get(row, ("top_k",))
    top_k = min(len(pairs), max(1, int(top_k_value) if isinstance(top_k_value, int) else 10))
    ranked = sorted(pairs, key=lambda pair: pair[0], reverse=True)
    return {
        "available": True,
        "proposal_count": len(pairs),
        "score_iou_correlation": _round_float(_pearson([item[0] for item in pairs], [item[1] for item in pairs])),
        "top_k": top_k,
        "top_k_mean_iou": _round_float(_mean([pair[1] for pair in ranked[:top_k]])),
    }


def _optional_int(value: Any, *, name: str) -> int | None:
    data = _to_plain(value)
    if data is None:
        return None
    try:
        return _as_int(data, name=name)
    except ValueError:
        return None


def _optional_float(value: Any, *, name: str) -> float | None:
    data = _to_plain(value)
    if data is None:
        return None
    try:
        return _as_finite_float(data, name=name)
    except ValueError:
        return None


def _slot_transport_diagnostics(row: Mapping[str, Any]) -> dict[str, Any]:
    raw_value = _nested_get(
        row,
        (
            "pc_ot_mras_prebackbone_raw_slot_dense_indices",
            "raw_slot_dense_indices",
            "raw_slot_indices",
        ),
    )
    raw_positions: list[int] = []
    if raw_value is not None:
        raw_positions = _as_positions(raw_value, name="raw_slot_dense_indices")

    raw_duplicate_rate = _optional_float(
        _nested_get(
            row,
            (
                "pc_ot_mras_prebackbone_raw_slot_duplicate_rate",
                "raw_slot_duplicate_rate",
            ),
        ),
        name="raw_slot_duplicate_rate",
    )
    if raw_duplicate_rate is None and raw_positions:
        raw_duplicate_rate = 1.0 - float(len(set(raw_positions))) / float(max(1, len(raw_positions)))

    reader_fill_count = _optional_int(
        _nested_get(
            row,
            (
                "pc_ot_mras_prebackbone_reader_fill_count",
                "reader_fill_count",
            ),
        ),
        name="reader_fill_count",
    )
    st_active_row_count = _optional_int(
        _nested_get(
            row,
            (
                "pc_ot_mras_prebackbone_st_active_row_count",
                "st_active_row_count",
            ),
        ),
        name="st_active_row_count",
    )

    return {
        "available": bool(
            raw_positions
            or raw_duplicate_rate is not None
            or reader_fill_count is not None
            or st_active_row_count is not None
        ),
        "raw_slot_count": len(raw_positions),
        "raw_slot_unique_count": len(set(raw_positions)) if raw_positions else None,
        "raw_slot_duplicate_rate": _round_float(raw_duplicate_rate),
        "reader_fill_count": reader_fill_count,
        "st_active_row_count": st_active_row_count,
    }


def _score_rank_diagnostics(selected: Sequence[int], row: Mapping[str, Any], valid_len: int) -> dict[str, Any]:
    dense_scores, per_selected_scores, source = _extract_scores(row, selected, valid_len)
    proposal = _proposal_rank_diagnostics(row)
    if dense_scores is None and per_selected_scores is None:
        return {
            "available": bool(proposal["available"]),
            "source": source,
            "selected_rank_available": False,
            "proposal_rank": proposal,
            "entry_note": "pass selector_scores/dense_scores or proposals with score+iou for deeper score/rank diagnostics",
        }

    selected_scores: list[float] = []
    selected_ranks: list[int] = []
    top_k_hit_rate = None
    if dense_scores is not None:
        valid_scores = dense_scores[: max(0, min(valid_len, len(dense_scores)))]
        finite_ranked = sorted(
            [(idx, score) for idx, score in enumerate(valid_scores) if math.isfinite(float(score))],
            key=lambda item: item[1],
            reverse=True,
        )
        rank_by_pos = {pos: rank for rank, (pos, _score) in enumerate(finite_ranked, start=1)}
        top_k = max(1, len(selected))
        top_k_positions = {pos for pos, _score in finite_ranked[:top_k]}
        for pos in selected:
            if 0 <= pos < len(dense_scores) and pos in rank_by_pos:
                selected_scores.append(float(dense_scores[pos]))
                selected_ranks.append(int(rank_by_pos[pos]))
        top_k_hit_rate = len([pos for pos in selected if pos in top_k_positions]) / float(len(selected)) if selected else None
    elif per_selected_scores is not None:
        selected_scores = list(per_selected_scores)

    return {
        "available": True,
        "source": source,
        "selected_rank_available": bool(selected_ranks),
        "selected_score_mean": _round_float(_mean(selected_scores)),
        "selected_score_min": _round_float(min(selected_scores) if selected_scores else None),
        "selected_score_max": _round_float(max(selected_scores) if selected_scores else None),
        "selected_rank_min": min(selected_ranks) if selected_ranks else None,
        "selected_rank_mean": _round_float(_mean([float(item) for item in selected_ranks])),
        "selected_rank_p95": _round_float(_percentile([float(item) for item in selected_ranks], 0.95)),
        "selected_topk_hit_rate": _round_float(top_k_hit_rate),
        "proposal_rank": proposal,
    }


def _metadata_consistency(row: Mapping[str, Any], selected: Sequence[int], valid_len: int) -> dict[str, Any]:
    native_axis = _bool_or_none(_nested_get(row, ("irregular_native_axis",)))
    physical_grid = _bool_or_none(_nested_get(row, ("physical_grid_actionformer", "physical_grid")))
    dense_valid = _nested_get(row, ("irregular_dense_valid_len", "dense_valid_len"))
    selected_valid = _nested_get(row, ("irregular_selected_valid_len",))
    irregular_positions = _nested_get(row, ("irregular_selected_positions",))

    issues: list[str] = []
    if physical_grid is True and native_axis is not True:
        issues.append("physical_grid_actionformer requires irregular_native_axis=True")
    if native_axis is True and not selected and irregular_positions is None:
        issues.append("irregular_native_axis=True but selected positions are absent")
    if dense_valid is not None and selected_valid is not None:
        try:
            dense_value = _as_finite_float(dense_valid, name="irregular_dense_valid_len")
            selected_value = _as_finite_float(selected_valid, name="irregular_selected_valid_len")
            if abs(dense_value - selected_value) > 1.0e-6:
                issues.append("irregular_dense_valid_len and irregular_selected_valid_len mismatch")
            if valid_len and abs(dense_value - float(valid_len)) > 1.0e-6:
                issues.append("valid_len and irregular_dense_valid_len mismatch")
        except ValueError as exc:
            issues.append(str(exc))
    if selected and valid_len > 0 and max(selected) >= valid_len:
        issues.append("selected_dense_indices exceed valid_len")

    return {
        "consistent": len(issues) == 0,
        "issues": issues,
        "has_irregular_native_axis": native_axis is not None,
        "irregular_native_axis": native_axis,
        "has_physical_grid_actionformer": physical_grid is not None,
        "physical_grid_actionformer": physical_grid,
        "has_irregular_selected_positions": irregular_positions is not None or bool(selected),
        "has_irregular_dense_valid_len": dense_valid is not None,
        "has_irregular_selected_valid_len": selected_valid is not None,
    }


def _gap_diagnostics(selected: Sequence[int]) -> dict[str, Any]:
    gaps = [abs(int(right) - int(left)) for left, right in zip(selected, selected[1:])]
    return {
        "count": len(gaps),
        "mean": _round_float(_mean([float(item) for item in gaps])),
        "max": max(gaps) if gaps else None,
        "p95": _round_float(_percentile([float(item) for item in gaps], 0.95)),
    }


def _valid_len_from_row(row: Mapping[str, Any], selected: Sequence[int]) -> int:
    value = _nested_get(row, VALID_LEN_KEYS)
    data = _to_plain(value)
    if isinstance(data, list):
        if len(data) == 1:
            data = data[0]
        else:
            data = len([item for item in data if bool(item)])
    if data is not None:
        try:
            parsed = int(float(data))
            if parsed >= 0:
                return parsed
        except (TypeError, ValueError):
            pass
    return max(selected) + 1 if selected else 0


def _budget_from_row(row: Mapping[str, Any], selected_count: int) -> int:
    value = _nested_get(row, ("budget", "target_budget", "selected_count", "irregular_selected_count"))
    if value is None:
        return int(selected_count)
    try:
        return int(float(_to_plain(value)))
    except (TypeError, ValueError):
        return int(selected_count)


def _extract_direct_sample(row: Mapping[str, Any], *, row_idx: int) -> dict[str, Any]:
    selected_value = _nested_get(row, POSITION_KEYS)
    selected = _as_positions(selected_value, name="selected_dense_indices")
    valid_len = _valid_len_from_row(row, selected)
    sample_id = str(_nested_get(row, ("sample_id", "video_id", "video_name", "name")) or f"sample_{row_idx}")
    return {
        "sample_id": sample_id,
        "source_row": row,
        "selected": selected,
        "valid_len": valid_len,
        "budget": _budget_from_row(row, len(selected)),
    }


def _expand_dynamic_plan(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    budgets = _to_plain(payload.get("budgets"))
    dense_lens = _to_plain(payload.get("dense_valid_len"))
    positions = _to_plain(payload.get("selected_dense_positions"))
    masks = _to_plain(payload.get("selected_mask"))
    if not isinstance(budgets, list) or not isinstance(dense_lens, list) or not isinstance(positions, list):
        return []
    batch_size = len(budgets)
    sample_ids = _to_plain(payload.get("sample_ids"))
    if not isinstance(sample_ids, list) or len(sample_ids) != batch_size:
        sample_ids = [f"dynamic_plan_{idx}" for idx in range(batch_size)]
    rows: list[dict[str, Any]] = []
    for idx in range(batch_size):
        budget = _as_int(budgets[idx], name=f"budgets[{idx}]")
        row_positions = positions[idx]
        selected = _as_positions(row_positions, name=f"selected_dense_positions[{idx}]")
        if isinstance(masks, list) and len(masks) == batch_size and isinstance(masks[idx], list):
            selected = [pos for pos, keep in zip(selected, masks[idx]) if bool(keep)]
        selected = selected[:budget]
        rows.append(
            {
                "sample_id": str(sample_ids[idx]),
                "selected_dense_indices": selected,
                "valid_len": int(dense_lens[idx]),
                "budget": budget,
                "irregular_native_axis": payload.get("irregular_native_axis"),
                "physical_grid_actionformer": payload.get("physical_grid_actionformer"),
                "selector_scores": _sample(payload.get("budget_scores"), idx, batch_size) if "budget_scores" in payload else None,
            }
        )
    return [_extract_direct_sample(row, row_idx=idx) for idx, row in enumerate(rows)]


def _matrix_column_scores(matrix: Any) -> list[float] | None:
    data = _to_plain(matrix)
    if not isinstance(data, list) or not data or not all(isinstance(row, list) for row in data):
        return None
    width = len(data[0])
    scores = [0.0 for _ in range(width)]
    try:
        for row in data:
            if len(row) != width:
                return None
            for idx, item in enumerate(row):
                scores[idx] += _as_finite_float(item, name=f"matrix[{idx}]")
    except ValueError:
        return None
    return scores


def _expand_reader_snapshot_row(row: Mapping[str, Any], *, row_idx: int) -> list[dict[str, Any]]:
    reader_out = row.get("reader_out")
    if not isinstance(reader_out, Mapping):
        return []
    batch_size = _batch_size_from(reader_out)
    sample_ids = _to_plain(row.get("sample_ids"))
    if not isinstance(sample_ids, list) or len(sample_ids) != batch_size:
        sample_ids = [str(row.get("sample_id", f"row_{row_idx}|batch{idx}")) for idx in range(batch_size)]

    samples: list[dict[str, Any]] = []
    for batch_idx in range(batch_size):
        sample_row: dict[str, Any] = {
            "sample_id": sample_ids[batch_idx],
            "budget": row.get("budget"),
            "snapshot_id": row.get("snapshot_id"),
            "epoch": row.get("epoch"),
        }
        for key in POSITION_KEYS:
            if key in reader_out:
                sample_row["selected_dense_indices"] = _sample(reader_out[key], batch_idx, batch_size)
                break
        if "valid_lengths" in reader_out:
            sample_row["valid_len"] = _sample(reader_out["valid_lengths"], batch_idx, batch_size)
        elif "valid_len" in row:
            sample_row["valid_len"] = row["valid_len"]
        elif "dense_len" in row:
            sample_row["valid_len"] = row["dense_len"]
        for key in ("role_ids", "packet_roles", "roles"):
            if key in reader_out:
                sample_row[key] = _sample(reader_out[key], batch_idx, batch_size)
                break
        for key in DENSE_SCORE_KEYS:
            if key in reader_out:
                sample_row["selector_scores"] = _sample(reader_out[key], batch_idx, batch_size)
                break
        if "selector_scores" not in sample_row:
            for key in MATRIX_SCORE_KEYS:
                if key in reader_out:
                    sample_row["selector_scores"] = _matrix_column_scores(_sample(reader_out[key], batch_idx, batch_size))
                    break
        for meta_key in (
            "irregular_native_axis",
            "physical_grid_actionformer",
            "irregular_selected_positions",
            "irregular_dense_valid_len",
            "irregular_selected_valid_len",
        ):
            if meta_key in row:
                sample_row[meta_key] = row[meta_key]
        samples.append(_extract_direct_sample(sample_row, row_idx=row_idx + batch_idx))
    return samples


def _rows_from_payload(payload: Any) -> list[Mapping[str, Any]]:
    data = _to_plain(payload)
    if isinstance(data, list):
        if not all(isinstance(item, Mapping) for item in data):
            raise ValueError("payload list must contain objects")
        return list(data)
    if isinstance(data, Mapping):
        if isinstance(data.get("samples"), list):
            return [item for item in data["samples"] if isinstance(item, Mapping)]
        if isinstance(data.get("rows"), list):
            return [item for item in data["rows"] if isinstance(item, Mapping)]
        return [data]
    raise ValueError("payload must be an object, list, or {'samples': [...]}")


def _expand_samples(payload: Any) -> list[dict[str, Any]]:
    rows = _rows_from_payload(payload)
    expanded: list[dict[str, Any]] = []
    for row_idx, row in enumerate(rows):
        if {"budgets", "dense_valid_len", "selected_dense_positions"}.issubset(row.keys()):
            expanded.extend(_expand_dynamic_plan(row))
            continue
        reader_samples = _expand_reader_snapshot_row(row, row_idx=row_idx)
        if reader_samples:
            expanded.extend(reader_samples)
            continue
        expanded.append(_extract_direct_sample(row, row_idx=row_idx))
    if not expanded:
        raise ValueError("no selector samples found")
    return expanded


def _diagnose_sample(sample: Mapping[str, Any], *, default_boundary_radius: float) -> dict[str, Any]:
    row = sample["source_row"]
    selected = [int(item) for item in sample["selected"]]
    valid_len = int(sample["valid_len"])
    selected_count = len(selected)
    unique_count = len(set(selected))
    in_range_count = sum(1 for pos in selected if 0 <= int(pos) < valid_len)
    duplicate_rate = 0.0 if selected_count == 0 else (selected_count - unique_count) / float(selected_count)
    boundary = _boundary_diagnostics(selected, row, default_radius=default_boundary_radius)
    packet_roles = _packet_role_diagnostics(selected, row, default_radius=default_boundary_radius)
    metadata = _metadata_consistency(row, selected, valid_len)
    slot_transport = _slot_transport_diagnostics(row)
    return {
        "sample_id": sample["sample_id"],
        "selected_dense_indices": selected,
        "valid_len": valid_len,
        "budget": int(sample["budget"]),
        "selected_count": selected_count,
        "unique_selected_count": unique_count,
        "selected_fraction": _round_float(selected_count / valid_len if valid_len else None),
        "monotonic": all(selected[idx] >= selected[idx - 1] for idx in range(1, selected_count)),
        "duplicate_rate": _round_float(duplicate_rate),
        "repeat_rate": _round_float(duplicate_rate),
        "in_range_count": in_range_count,
        "out_of_range_count": selected_count - in_range_count,
        "gap": _gap_diagnostics(selected),
        "boundary": boundary,
        "packet_roles": packet_roles,
        "score_rank": _score_rank_diagnostics(selected, row, valid_len),
        "slot_transport": slot_transport,
        "metadata_consistency": metadata,
    }


def _aggregate(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected_counts = [int(item["selected_count"]) for item in samples]
    budgets = [int(item["budget"]) for item in samples]
    duplicate_rates = [float(item["duplicate_rate"]) for item in samples if item.get("duplicate_rate") is not None]
    gap_max_values = [float(item["gap"]["max"]) for item in samples if item.get("gap", {}).get("max") is not None]

    boundary_count = 0
    boundary_support = 0
    boundary_near_selected = 0
    boundary_selected_total = 0
    for item in samples:
        boundary = item["boundary"]
        if boundary.get("provided"):
            boundary_count += int(boundary.get("boundary_count", 0))
            boundary_support += int(boundary.get("boundary_support_count", 0))
            boundary_near_selected += int(boundary.get("boundary_near_selected_count", 0))
            boundary_selected_total += int(item["selected_count"])

    role_counts = {"boundary": 0, "interior": 0, "background": 0, "coverage": 0, "other": 0}
    for item in samples:
        counts = item.get("packet_roles", {}).get("counts", {})
        for key in role_counts:
            role_counts[key] += int(counts.get(key, 0))
    role_total = sum(role_counts.values())

    metadata_inconsistent = [
        item for item in samples if not bool(item.get("metadata_consistency", {}).get("consistent", False))
    ]
    slot_items = [item.get("slot_transport", {}) for item in samples]
    raw_slot_duplicate_rates = [
        float(item["raw_slot_duplicate_rate"])
        for item in slot_items
        if item.get("raw_slot_duplicate_rate") is not None
    ]
    reader_fill_counts = [
        float(item["reader_fill_count"])
        for item in slot_items
        if item.get("reader_fill_count") is not None
    ]
    st_active_counts = [
        float(item["st_active_row_count"])
        for item in slot_items
        if item.get("st_active_row_count") is not None
    ]

    return {
        "selected_count_distribution": _distribution(selected_counts),
        "selected_count_min": min(selected_counts),
        "selected_count_max": max(selected_counts),
        "selected_count_mean": _round_float(_mean([float(item) for item in selected_counts])),
        "duplicate_rate_mean": _round_float(_mean(duplicate_rates)),
        "gap_max_mean": _round_float(_mean(gap_max_values)),
        "gap_max_p95": _round_float(_percentile(gap_max_values, 0.95)),
        "boundary": {
            "samples_with_gt": sum(1 for item in samples if item.get("boundary", {}).get("provided")),
            "boundary_support_rate": _round_float(boundary_support / boundary_count if boundary_count else None),
            "near_selected_rate": _round_float(
                boundary_near_selected / boundary_selected_total if boundary_selected_total else None
            ),
            "boundary_near_selected_count": boundary_near_selected,
            "boundary_selected_total": boundary_selected_total,
        },
        "packet_roles": {
            "total": role_total,
            "counts": role_counts,
            "boundary_ratio": _round_float(role_counts["boundary"] / role_total if role_total else None),
            "interior_ratio": _round_float(role_counts["interior"] / role_total if role_total else None),
            "action_ratio": _round_float(role_counts["interior"] / role_total if role_total else None),
            "background_ratio": _round_float(role_counts["background"] / role_total if role_total else None),
            "coverage_ratio": _round_float(role_counts["coverage"] / role_total if role_total else None),
            "other_ratio": _round_float(role_counts["other"] / role_total if role_total else None),
        },
        "dynamic_budget": {
            "budget_distribution": _distribution(budgets),
            "budget_min": min(budgets),
            "budget_max": max(budgets),
            "budget_mean": _round_float(_mean([float(item) for item in budgets])),
            "budget_p50": _round_float(_percentile([float(item) for item in budgets], 0.50)),
            "budget_p95": _round_float(_percentile([float(item) for item in budgets], 0.95)),
        },
        "metadata_consistency": {
            "consistent_sample_count": len(samples) - len(metadata_inconsistent),
            "inconsistent_sample_count": len(metadata_inconsistent),
            "inconsistent_sample_ids": [str(item["sample_id"]) for item in metadata_inconsistent[:32]],
        },
        "score_rank": {
            "samples_with_score_rank": sum(1 for item in samples if item.get("score_rank", {}).get("available")),
            "samples_with_selected_rank": sum(
                1 for item in samples if item.get("score_rank", {}).get("selected_rank_available")
            ),
        },
        "slot_transport": {
            "samples_with_slot_transport": sum(1 for item in slot_items if item.get("available")),
            "samples_with_raw_slot_duplicate": len(raw_slot_duplicate_rates),
            "samples_with_reader_fill_count": len(reader_fill_counts),
            "samples_with_st_active_row_count": len(st_active_counts),
            "raw_slot_duplicate_rate_mean": _round_float(_mean(raw_slot_duplicate_rates)),
            "raw_slot_duplicate_rate_p95": _round_float(_percentile(raw_slot_duplicate_rates, 0.95)),
            "reader_fill_count_mean": _round_float(_mean(reader_fill_counts)),
            "reader_fill_count_p95": _round_float(_percentile(reader_fill_counts, 0.95)),
            "st_active_row_count_mean": _round_float(_mean(st_active_counts)),
            "st_active_row_count_p05": _round_float(_percentile(st_active_counts, 0.05)),
        },
    }


def synthetic_payload() -> dict[str, Any]:
    return {
        "samples": [
            {
                "sample_id": "synthetic_easy",
                "selected_dense_indices": [0, 4, 8, 12],
                "valid_len": 16,
                "gt_segments": [[3, 11]],
                "packet_roles": ["coverage", "boundary", "interior", "boundary"],
                "selector_scores": [0.1, 0.1, 0.2, 0.8, 0.9, 0.4, 0.3, 0.2, 0.7, 0.2, 0.2, 0.75, 0.8, 0.1, 0.1, 0.0],
                "irregular_native_axis": True,
                "physical_grid_actionformer": True,
                "irregular_selected_positions": [0, 4, 8, 12],
                "irregular_dense_valid_len": 16,
                "irregular_selected_valid_len": 16,
            },
            {
                "sample_id": "synthetic_hard",
                "selected_dense_indices": [1, 3, 5, 7, 10, 14],
                "valid_len": 16,
                "gt_segments": [[2, 14]],
                "packet_roles": ["coverage", "boundary", "interior", "interior", "interior", "boundary"],
                "selector_scores": [0.2, 0.4, 0.8, 0.9, 0.5, 0.7, 0.6, 0.75, 0.3, 0.25, 0.65, 0.3, 0.2, 0.4, 0.85, 0.1],
                "irregular_native_axis": True,
                "physical_grid_actionformer": True,
                "irregular_selected_positions": [1, 3, 5, 7, 10, 14],
                "irregular_dense_valid_len": 16,
                "irregular_selected_valid_len": 16,
            },
        ]
    }


def analyze_selector_payload(
    payload: Any,
    *,
    source_format: str = "memory",
    source_path: str | None = None,
    boundary_radius: float = 2.0,
    synthetic_smoke: bool = False,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    non_finite_count, non_finite_examples = _find_non_finite(payload)
    expanded = _expand_samples(payload)
    sample_summaries = [
        _diagnose_sample(sample, default_boundary_radius=float(boundary_radius))
        for sample in expanded
    ]
    aggregate = _aggregate(sample_summaries)
    warning_count = int(non_finite_count) + int(aggregate["metadata_consistency"]["inconsistent_sample_count"])
    decision = READY_WITH_WARNINGS if warning_count else READY
    summary = {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "input": {"format": str(source_format), "path": source_path},
        "synthetic_smoke": bool(synthetic_smoke),
        "sample_count": len(sample_summaries),
        "non_finite": {"count": int(non_finite_count), "examples": non_finite_examples},
        "aggregate": aggregate,
        "samples": sample_summaries,
        "protocol": {
            "diagnostic_only": True,
            "selector_runtime_uses_gt": False,
            "gt_segments_diagnostic_only": True,
            "uses_teacher": False,
            "uses_oracle": False,
            "uses_raw_prediction_shortcut": False,
            "runs_training": False,
            "runs_tools_test": False,
            "tools_train_allowed": False,
            "tools_test_allowed": False,
            "remote_sync_allowed": False,
            "remote_deploy_allowed": False,
            "slurm_gpu_allowed": False,
            "metric_claim_allowed": False,
            "paper_claim_allowed": False,
            "runtime_flops_claim_allowed": False,
            "deploy_claim_allowed": False,
        },
    }
    if provenance is not None:
        summary["provenance"] = strict_json_value(dict(provenance))
    return summary


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_no}: JSONL rows must be objects")
            rows.append(row)
    if not rows:
        raise ValueError(f"JSONL has no rows: {path}")
    return rows


def load_payload(path: str | Path) -> tuple[Any, str]:
    input_path = Path(path).expanduser()
    suffix = input_path.suffix.lower()
    if suffix == ".jsonl":
        return _load_jsonl(input_path), "jsonl"
    if suffix == ".json":
        return json.loads(input_path.read_text(encoding="utf-8-sig")), "json"
    if suffix in (".pkl", ".pickle"):
        with input_path.open("rb") as f:
            return pickle.load(f), "pickle"
    if suffix == ".npz":
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError("numpy is required to read npz input") from exc
        with np.load(input_path, allow_pickle=True) as data:
            return {key: _to_plain(data[key]) for key in data.files}, "npz"
    raise ValueError("input must be .json, .jsonl, .pkl, .pickle, or .npz")


def run_selector_posttrain_diagnostics(
    *,
    input_path: str | Path | None = None,
    output_json: str | Path | None = None,
    boundary_radius: float = 2.0,
    use_synthetic: bool = False,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if use_synthetic:
        payload = synthetic_payload()
        source_format = "synthetic"
        source_text = None
    else:
        if input_path is None:
            raise ValueError("--input is required unless --synthetic-smoke is set")
        payload, source_format = load_payload(input_path)
        source_text = str(input_path)
    summary = analyze_selector_payload(
        payload,
        source_format=source_format,
        source_path=source_text,
        boundary_radius=float(boundary_radius),
        synthetic_smoke=bool(use_synthetic),
        provenance=provenance,
    )
    if output_json is not None:
        write_json(output_json, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze post-train PC-OT-MRAS selector metadata without training, testing, or remote deployment."
    )
    parser.add_argument("--input", help="Input .json/.jsonl/.pkl/.pickle/.npz selector metadata/result dump.")
    parser.add_argument("--output", help="Optional summary JSON output path.")
    parser.add_argument("--boundary-radius", type=float, default=2.0)
    parser.add_argument("--synthetic-smoke", action="store_true", help="Run a built-in synthetic smoke payload.")
    parser.add_argument("--provenance-run-root")
    parser.add_argument("--provenance-work-dir")
    parser.add_argument("--provenance-train-stdout")
    parser.add_argument("--provenance-result-detection-json")
    args = parser.parse_args(argv)
    provenance = {
        key: value
        for key, value in {
            "run_root": args.provenance_run_root,
            "work_dir": args.provenance_work_dir,
            "train_stdout": args.provenance_train_stdout,
            "result_detection_json": args.provenance_result_detection_json,
        }.items()
        if value is not None
    }

    try:
        summary = run_selector_posttrain_diagnostics(
            input_path=args.input,
            output_json=args.output,
            boundary_radius=float(args.boundary_radius),
            use_synthetic=bool(args.synthetic_smoke),
            provenance=provenance or None,
        )
    except Exception as exc:  # pragma: no cover - CLI guard
        print(json.dumps({"schema_version": SCHEMA_VERSION, "decision": NO_GO, "error": str(exc)}))
        return 1

    print(json.dumps(strict_json_value(summary), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
