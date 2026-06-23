from __future__ import annotations

import argparse
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "pc_ot_mras_hard_positions_v0"
SUMMARY_SCHEMA_VERSION = "pc_ot_mras_hard_positions_summary_v0"
GENERATION_SOURCE = "pc_ot_mras_hard_export_resolver_v0"
DYNAMIC_BUDGET_GENERATION_SOURCE = "pc_ot_mras_dynamic_budget_hard_export_resolver_v0"
TEMPORAL_METADATA_SCHEMA_VERSION = "pc_ot_mras_temporal_metadata_v0"
TEMPORAL_METADATA_GENERATION_SOURCE = "pc_ot_mras_hard_rows_to_temporal_metadata_v0"
READY = "PC_OT_MRAS_HARD_EXPORT_READY"
NO_GO = "PC_OT_MRAS_HARD_EXPORT_NO_GO"
MATRIX_PRIORITY = ("acquisition_matrix", "allocation", "transport_prob")
FORBIDDEN_JSONL_KEY_TOKENS = (
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
    "checkpoint",
    "ckpt",
)
HARD_EXPORT_ALLOWED_METADATA_KEYS = frozenset(
    {
        "validlength",
        "validlengths",
    }
)
FALSE_ONLY_DYNAMIC_PLAN_FLAGS = frozenset(
    {
        "uses_gt",
        "uses_teacher",
        "uses_oracle",
        "uses_cache",
        "uses_raw_prediction",
        "uses_prediction_cache",
        "uses_checkpoint",
        "dynamic_budget_validation",
        "metric_claim_allowed",
        "paper_claim_allowed",
    }
)
FALSE_ONLY_HARD_EXPORT_FLAGS = frozenset(
    {
        *FALSE_ONLY_DYNAMIC_PLAN_FLAGS,
        "deploy_claim_allowed",
        "runtime_flops_claim_allowed",
        "scanner_quality_claim_allowed",
        "dynamic_budget_claim_allowed",
    }
)
DYNAMIC_PLAN_ALLOWED_KEYS = frozenset(
    {
        *FALSE_ONLY_DYNAMIC_PLAN_FLAGS,
        "schema_version",
        "controller_family",
        "budget_values",
        "budget_scores",
        "budget_score",
        "budgets",
        "dense_valid_len",
        "utility_scores",
        "selected_dense_positions",
        "selected_mask",
        "coverage_counts",
        "coverage_count",
        "value_counts",
        "value_count",
        "coverage_share",
        "max_coverage_share",
    }
)
DYNAMIC_PLAN_FORBIDDEN_KEY_TOKENS = (
    *FORBIDDEN_JSONL_KEY_TOKENS,
    "label",
    "labels",
    "segment",
    "segments",
    "annotation",
    "annotations",
    "valuetarget",
    "valuetargets",
    "trainvaluetarget",
    "trainvaluetargets",
    "targetlogit",
    "targetlogits",
)
FALSE_ONLY_TEMPORAL_METADATA_FLAGS = frozenset(
    {
        *FALSE_ONLY_DYNAMIC_PLAN_FLAGS,
        "uses_prediction_cache",
        "deploy_claim_allowed",
        "runtime_flops_claim_allowed",
        "scanner_quality_claim_allowed",
        "dynamic_budget_claim_allowed",
        "allow_detector_training",
        "allow_tools_train",
        "allow_tools_test",
        "allow_detector_map",
        "allow_remote_sync",
        "allow_precheck_only",
        "allow_slurm",
        "allow_gpu",
        "allow_real_dataset",
        "allow_checkpoint",
        "allow_raw_prediction_cache",
    }
)


def strict_json_value(value: Any) -> Any:
    if isinstance(value, dict):
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


def _maybe_no_grad():
    module = sys.modules.get("torch")
    no_grad = getattr(module, "no_grad", None) if module is not None else None
    return no_grad() if callable(no_grad) else nullcontext()


def _to_plain(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "tolist"):
        return value.detach().cpu().tolist()
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


def _batch_size_from(reader_out: Mapping[str, Any]) -> int:
    for key in (*MATRIX_PRIORITY, "transport_logits"):
        value = reader_out.get(key)
        if value is not None and _depth(value) >= 3:
            return len(_to_plain(value))
    for key in ("selection_logits", "selection_prob", "soft_selection", "valid_mask"):
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


def _as_int_list(value: Any, *, name: str) -> list[int]:
    data = _to_plain(value)
    if data is None:
        return []
    if not isinstance(data, list):
        raise ValueError(f"{name} must be a list")
    out: list[int] = []
    for idx, item in enumerate(data):
        try:
            out.append(_strict_int_scalar(item, name=f"{name}[{idx}]"))
        except ValueError:
            raise ValueError(f"{name}[{idx}] must be an integer position") from None
    return out


def _as_float_list(value: Any, *, name: str) -> list[float]:
    data = _to_plain(value)
    if data is None:
        return []
    if not isinstance(data, list):
        raise ValueError(f"{name} must be a list")
    return [_finite_float(item, name=f"{name}[{idx}]") for idx, item in enumerate(data)]


def _normalized_key(key: Any) -> str:
    return "".join(ch for ch in str(key).lower() if ch.isalnum())


def _is_json_false(value: Any) -> bool:
    data = _to_plain(value)
    return data is False


def _is_false_only_hard_export_key(key: Any) -> bool:
    key_text = str(key or "")
    normalized = _normalized_key(key_text)
    return key_text in FALSE_ONLY_HARD_EXPORT_FLAGS or normalized in {
        _normalized_key(flag) for flag in FALSE_ONLY_HARD_EXPORT_FLAGS
    }


def _is_allowed_hard_export_metadata_key(key: Any) -> bool:
    return _normalized_key(key) in HARD_EXPORT_ALLOWED_METADATA_KEYS


def _validate_no_forbidden_jsonl_keys(value: Any, *, path: str = "row") -> None:
    data = _to_plain(value)
    if isinstance(data, Mapping):
        for key, item in data.items():
            normalized = _normalized_key(key)
            if _is_false_only_hard_export_key(key):
                if not _is_json_false(item):
                    raise ValueError(f"{path}.{key} must be JSON false for deploy-visible hard export")
                continue
            if not _is_allowed_hard_export_metadata_key(key) and any(
                token in normalized for token in FORBIDDEN_JSONL_KEY_TOKENS
            ):
                raise ValueError(f"{path}.{key}: forbidden deploy-invisible key in hard export input")
            _validate_no_forbidden_jsonl_keys(item, path=f"{path}.{key}")
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            _validate_no_forbidden_jsonl_keys(item, path=f"{path}[{idx}]")
    elif isinstance(data, str) and _contains_forbidden_export_fragment(data):
        raise ValueError(f"{path}: forbidden deploy-invisible value in hard export input")


def _contains_forbidden_export_fragment(value: Any) -> bool:
    normalized = _normalized_key(value)
    return any(token in normalized for token in FORBIDDEN_JSONL_KEY_TOKENS)


def _contains_forbidden_dynamic_plan_fragment(value: Any) -> bool:
    normalized = _normalized_key(value)
    if normalized.startswith("allow"):
        return True
    if normalized.endswith("claimallowed"):
        return True
    return any(token in normalized for token in DYNAMIC_PLAN_FORBIDDEN_KEY_TOKENS)


def _validate_dynamic_budget_plan_payload(value: Any, *, path: str = "dynamic_budget_plan") -> None:
    data = _to_plain(value)
    if isinstance(data, Mapping):
        for key, item in data.items():
            key_text = str(key or "")
            if key_text in FALSE_ONLY_DYNAMIC_PLAN_FLAGS:
                if bool(_to_plain(item)):
                    raise ValueError(f"{path}.{key_text} must be false for deploy-visible hard export")
                continue
            if key_text not in DYNAMIC_PLAN_ALLOWED_KEYS:
                if _contains_forbidden_dynamic_plan_fragment(key_text):
                    raise ValueError(f"{path}.{key_text}: forbidden deploy-invisible key in dynamic budget plan")
                raise ValueError(f"{path}.{key_text}: unsupported key in deploy-visible dynamic budget plan")
            if _contains_forbidden_dynamic_plan_fragment(key_text):
                raise ValueError(f"{path}.{key_text}: forbidden deploy-invisible key in dynamic budget plan")
            _validate_dynamic_budget_plan_payload(item, path=f"{path}.{key_text}")
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            _validate_dynamic_budget_plan_payload(item, path=f"{path}[{idx}]")
    elif isinstance(data, str) and _contains_forbidden_dynamic_plan_fragment(data):
        raise ValueError(f"{path}: forbidden deploy-invisible value in dynamic budget plan")


def _validate_false_only_temporal_flags(value: Any, *, path: str = "row") -> None:
    data = _to_plain(value)
    if isinstance(data, Mapping):
        for key, item in data.items():
            key_text = str(key or "")
            if key_text in FALSE_ONLY_TEMPORAL_METADATA_FLAGS and bool(_to_plain(item)):
                raise ValueError(f"{path}.{key_text} must be false for deploy-visible temporal metadata")
            _validate_false_only_temporal_flags(item, path=f"{path}.{key_text}")
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            _validate_false_only_temporal_flags(item, path=f"{path}[{idx}]")


def _row_budget(row: Mapping[str, Any], *, cli_budget: int, row_idx: int) -> int:
    declared: list[tuple[str, int]] = []
    for key in ("budget", "target_budget"):
        if key not in row:
            continue
        value = row.get(key)
        if isinstance(value, bool):
            raise ValueError(f"row {row_idx}: {key} must be an integer budget")
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"row {row_idx}: {key} must be an integer budget") from None
        declared.append((key, parsed))
    if not declared:
        return int(cli_budget)
    mismatched = [(key, value) for key, value in declared if int(value) != int(cli_budget)]
    if mismatched:
        detail = ", ".join(f"{key}={value}" for key, value in mismatched)
        raise ValueError(f"row {row_idx}: row budget conflicts with CLI budget {int(cli_budget)} ({detail})")
    return int(cli_budget)


def _row_sample_ids(row: Mapping[str, Any], *, row_idx: int) -> list[str]:
    if "sample_ids" in row:
        values = _to_plain(row.get("sample_ids"))
        if not isinstance(values, list) or not values:
            raise ValueError(f"row {row_idx}: sample_ids must be a non-empty list")
        return [str(item) for item in values]
    if "sample_id" in row:
        return [str(row.get("sample_id"))]
    return [f"row_{row_idx}"]


def _argmax(values: Sequence[Any], *, name: str) -> int:
    if not values:
        return -1
    best_idx = 0
    best_value = _finite_float(values[0], name=f"{name}[0]")
    for idx, item in enumerate(values[1:], start=1):
        score = _finite_float(item, name=f"{name}[{idx}]")
        if score > best_value:
            best_idx = idx
            best_value = score
    return int(best_idx)


def _valid_positions(valid_mask: Any, *, dense_len: int | None, valid_len: int | None) -> list[int]:
    if valid_mask is not None:
        mask = _to_plain(valid_mask)
        if not isinstance(mask, list):
            raise ValueError("valid_mask must be a list or tensor-like value")
        positions: list[int] = []
        for idx, item in enumerate(mask):
            if item not in (0, 1, False, True):
                raise ValueError(f"valid_mask[{idx}] must be binary")
            if bool(item):
                positions.append(int(idx))
        if not positions:
            raise ValueError("valid_mask must contain at least one valid position")
        if positions != list(range(len(positions))):
            raise ValueError("valid_mask must be prefix-contiguous")
        if dense_len is not None and len(mask) != int(dense_len):
            raise ValueError("valid_mask length must equal dense_len")
        if valid_len is not None and len(positions) != int(valid_len):
            raise ValueError("valid_mask true count must equal valid_len")
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


def _expected_matrix_width(valid_mask: Any, *, dense_len: int | None, valid_len: int | None, valid: Sequence[int]) -> int:
    if valid_mask is not None:
        mask = _to_plain(valid_mask)
        if not isinstance(mask, list):
            raise ValueError("valid_mask must be a list or tensor-like value")
        return int(len(mask))
    if dense_len is not None:
        return int(dense_len)
    if valid_len is not None:
        return int(valid_len)
    return int(max(valid) + 1)


def _matrix_rows(
    reader_out: Mapping[str, Any],
    matrix_key: str,
    *,
    batch_idx: int,
    batch_size: int,
    expected_width: int,
) -> list[list[float]]:
    rows = _sample(reader_out[matrix_key], batch_idx, batch_size)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{matrix_key} sample must be [K,T]")
    if not all(isinstance(row, list) for row in rows):
        raise ValueError(f"{matrix_key} sample must be [K,T]")

    width: int | None = None
    out: list[list[float]] = []
    for slot_idx, row in enumerate(rows):
        if not row:
            raise ValueError(f"{matrix_key} sample must be [K,T]")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError(f"{matrix_key} sample must be rectangular [K,T]")
        checked_row: list[float] = []
        for pos, item in enumerate(row):
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ValueError(f"{matrix_key}[{slot_idx}][{pos}] must be numeric")
            checked_row.append(_finite_float(item, name=f"{matrix_key}[{slot_idx}][{pos}]"))
        out.append(checked_row)

    if width != int(expected_width):
        raise ValueError(f"{matrix_key} width must equal dense axis T={int(expected_width)}")
    return out


def _score_vector(
    reader_out: Mapping[str, Any],
    batch_idx: int,
    batch_size: int,
    valid: Sequence[int],
    *,
    expected_width: int,
) -> list[float]:
    for matrix_key in MATRIX_PRIORITY:
        if matrix_key not in reader_out:
            continue
        rows = _matrix_rows(reader_out, matrix_key, batch_idx=batch_idx, batch_size=batch_size, expected_width=expected_width)
        width = len(rows[0])
        scores = [0.0] * width
        for slot_idx, row in enumerate(rows):
            for pos, item in enumerate(row):
                scores[pos] += float(item)
        return scores
    for key in ("selection_logits", "selection_prob", "soft_selection"):
        if key in reader_out:
            return _as_float_list(_sample(reader_out[key], batch_idx, batch_size), name=key)
    if not valid:
        return []
    return [1.0 / max(1, abs(pos - valid[len(valid) // 2]) + 1) for pos in range(max(valid) + 1)]


def _candidate_positions(
    reader_out: Mapping[str, Any],
    *,
    batch_idx: int,
    batch_size: int,
    valid: Sequence[int],
    expected_width: int,
) -> list[dict[str, Any]]:
    valid_set = set(int(pos) for pos in valid)
    for matrix_key in MATRIX_PRIORITY:
        if matrix_key not in reader_out:
            continue
        rows = _matrix_rows(reader_out, matrix_key, batch_idx=batch_idx, batch_size=batch_size, expected_width=expected_width)
        candidates: list[dict[str, Any]] = []
        for slot_idx, row in enumerate(rows):
            best_pos = -1
            best_score = -math.inf
            for pos, item in enumerate(row):
                if int(pos) not in valid_set:
                    continue
                score = float(item)
                if score > best_score:
                    best_pos = int(pos)
                    best_score = score
            if best_pos >= 0:
                candidates.append({"pos": best_pos, "slot": slot_idx, "score": best_score})
        return sorted(
            candidates,
            key=lambda item: (-float(item["score"]), int(item["pos"]), int(item["slot"])),
        )

    if "hard_selected_positions" in reader_out:
        positions = _as_int_list(
            _sample(reader_out["hard_selected_positions"], batch_idx, batch_size),
            name="hard_selected_positions",
        )
        return [{"pos": pos, "slot": idx, "score": None} for idx, pos in enumerate(positions) if pos >= 0]

    if "selected_positions" in reader_out:
        positions = _as_int_list(_sample(reader_out["selected_positions"], batch_idx, batch_size), name="selected_positions")
        return [{"pos": pos, "slot": idx, "score": None} for idx, pos in enumerate(positions) if pos >= 0]

    scores = _score_vector(reader_out, batch_idx, batch_size, valid, expected_width=expected_width)
    ranked = sorted(
        ({"pos": int(pos), "slot": idx, "score": scores[pos] if pos < len(scores) else 0.0} for idx, pos in enumerate(valid)),
        key=lambda item: (-float(item["score"]), int(item["pos"])),
    )
    return ranked


def _slot_metadata(reader_out: Mapping[str, Any], key: str, logits_key: str, batch_idx: int, batch_size: int, slot: int) -> int:
    if key in reader_out:
        values = _sample(reader_out[key], batch_idx, batch_size)
        if isinstance(values, list) and 0 <= int(slot) < len(values):
            return int(values[int(slot)])
    if logits_key in reader_out:
        rows = _sample(reader_out[logits_key], batch_idx, batch_size)
        if isinstance(rows, list) and 0 <= int(slot) < len(rows) and isinstance(rows[int(slot)], list):
            return _argmax(rows[int(slot)], name=f"{logits_key}[{slot}]")
    return -1


def _time_metadata(reader_out: Mapping[str, Any], key: str, logits_key: str, batch_idx: int, batch_size: int, pos: int) -> int:
    if key in reader_out:
        values = _sample(reader_out[key], batch_idx, batch_size)
        if isinstance(values, list) and 0 <= int(pos) < len(values):
            return int(values[int(pos)])
    if logits_key in reader_out:
        rows = _sample(reader_out[logits_key], batch_idx, batch_size)
        if isinstance(rows, list) and 0 <= int(pos) < len(rows) and isinstance(rows[int(pos)], list):
            return _argmax(rows[int(pos)], name=f"{logits_key}[{pos}]")
    return -1


def _soft_hard_time_error(scores: Sequence[float], selected: Sequence[int], valid: Sequence[int]) -> float:
    if not selected or not valid:
        return 0.0
    denom = max(float(max(valid) - min(valid)), 1.0)
    hard_mean = sum((float(pos) - float(min(valid))) / denom for pos in selected) / float(len(selected))
    valid_scores = [(pos, max(0.0, float(scores[pos]) if pos < len(scores) else 0.0)) for pos in valid]
    mass = sum(score for _pos, score in valid_scores)
    if mass <= 0.0:
        soft_mean = sum((float(pos) - float(min(valid))) / denom for pos in valid) / float(len(valid))
    else:
        soft_mean = sum(((float(pos) - float(min(valid))) / denom) * score for pos, score in valid_scores) / mass
    return float(abs(hard_mean - soft_mean))


def _resolve_sample(
    reader_out: Mapping[str, Any],
    *,
    batch_idx: int,
    batch_size: int,
    budget: int,
    sample_id: str,
    dense_len: int | None,
    valid_len: int | None,
) -> dict[str, Any]:
    if int(budget) <= 0:
        raise ValueError("budget must be positive")

    sample_valid_mask = None
    if "valid_mask" in reader_out:
        sample_valid_mask = _sample(reader_out["valid_mask"], batch_idx, batch_size)
    valid = _valid_positions(sample_valid_mask, dense_len=dense_len, valid_len=valid_len)
    if int(budget) > len(valid):
        raise ValueError(f"{sample_id}: budget exceeds valid positions")
    expected_width = _expected_matrix_width(sample_valid_mask, dense_len=dense_len, valid_len=valid_len, valid=valid)

    valid_set = set(valid)
    scores = _score_vector(reader_out, batch_idx, batch_size, valid, expected_width=expected_width)
    candidates = _candidate_positions(
        reader_out,
        batch_idx=batch_idx,
        batch_size=batch_size,
        valid=valid,
        expected_width=expected_width,
    )

    selected_by_pos: dict[int, dict[str, Any]] = {}
    duplicate_repair_count = 0
    invalid_repair_count = 0
    for item in candidates:
        pos = int(item["pos"])
        if pos not in valid_set:
            invalid_repair_count += 1
            continue
        if pos in selected_by_pos:
            duplicate_repair_count += 1
            continue
        selected_by_pos[pos] = item
        if len(selected_by_pos) == int(budget):
            break

    repair_fill_count = 0
    if len(selected_by_pos) < int(budget):
        ranked_fill = sorted(
            (pos for pos in valid if pos not in selected_by_pos),
            key=lambda pos: (-(scores[pos] if pos < len(scores) else 0.0), int(pos)),
        )
        need = int(budget) - len(selected_by_pos)
        for pos in ranked_fill[:need]:
            selected_by_pos[int(pos)] = {"pos": int(pos), "slot": None, "score": scores[pos] if pos < len(scores) else None}
            repair_fill_count += 1

    selected = sorted(selected_by_pos)
    if len(selected) != int(budget):
        raise ValueError(f"{sample_id}: failed to resolve exact budget")
    if len(selected) != len(set(selected)):
        raise ValueError(f"{sample_id}: resolved positions are not unique")

    max_position = dense_len if dense_len is not None else (max(valid) + 1)
    selected_mask = [1 if idx in selected_by_pos else 0 for idx in range(int(max_position))]

    role_ids: list[int] = []
    round_ids: list[int] = []
    for pos in selected:
        item = selected_by_pos[pos]
        slot = item.get("slot")
        if slot is None:
            role_ids.append(_time_metadata(reader_out, "role_ids_by_time", "role_logits_by_time", batch_idx, batch_size, pos))
            round_ids.append(_time_metadata(reader_out, "round_ids_by_time", "round_logits_by_time", batch_idx, batch_size, pos))
        else:
            role_ids.append(_slot_metadata(reader_out, "role_ids", "role_logits", batch_idx, batch_size, int(slot)))
            round_ids.append(_slot_metadata(reader_out, "round_ids", "round_logits", batch_idx, batch_size, int(slot)))

    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample_id,
        "batch_index": int(batch_idx),
        "budget": int(budget),
        "dense_len": int(max_position),
        "valid_len": int(len(valid)),
        "selected_positions": selected,
        "selected_mask": selected_mask,
        "duplicate_repair_count": int(duplicate_repair_count),
        "invalid_repair_count": int(invalid_repair_count),
        "repair_fill_count": int(repair_fill_count),
        "soft_hard_time_error": _soft_hard_time_error(scores, selected, valid),
        "role_ids": role_ids,
        "round_ids": round_ids,
        "role_round_metadata": [{"position": pos, "role_id": role, "round_id": rnd} for pos, role, rnd in zip(selected, role_ids, round_ids)],
        "resolver_generation": {
            "source": GENERATION_SOURCE,
            "diagnostic_or_deploy_only": True,
            "training_backprop_allowed": False,
            "detached_reader_tensors": True,
        },
    }


def _require_plan_key(dynamic_plan: Mapping[str, Any], key: str) -> Any:
    if key not in dynamic_plan:
        raise ValueError(f"dynamic_budget_plan missing '{key}'")
    return _to_plain(dynamic_plan[key])


def _strict_int_scalar(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value) or not float(value).is_integer():
            raise ValueError(f"{name} must be an integer")
        return int(value)
    raise ValueError(f"{name} must be an integer")


def _strict_float_scalar(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be numeric") from None
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _plan_batch_vector(value: Any, *, name: str, batch_size: int) -> list[Any]:
    data = _to_plain(value)
    if not isinstance(data, list) or len(data) != int(batch_size):
        raise ValueError(f"{name} must be a batch vector of length {int(batch_size)}")
    return data


def _optional_plan_batch_value(dynamic_plan: Mapping[str, Any], key: str, batch_idx: int, batch_size: int) -> Any:
    if key not in dynamic_plan:
        return None
    data = _to_plain(dynamic_plan[key])
    if isinstance(data, list) and len(data) == int(batch_size):
        return data[batch_idx]
    return data


def _resolve_dynamic_budget_plan_sample(
    dynamic_plan: Mapping[str, Any],
    *,
    batch_idx: int,
    batch_size: int,
    sample_id: str,
    budgets: Sequence[Any],
    dense_valid_lens: Sequence[Any],
    selected_positions_rows: Sequence[Any],
    selected_mask_rows: Sequence[Any],
) -> dict[str, Any]:
    budget = _strict_int_scalar(budgets[batch_idx], name=f"budgets[{batch_idx}]")
    if budget <= 0:
        raise ValueError(f"{sample_id}: dynamic budget must be positive")
    dense_valid_len = _strict_int_scalar(dense_valid_lens[batch_idx], name=f"dense_valid_len[{batch_idx}]")
    if dense_valid_len <= 0:
        raise ValueError(f"{sample_id}: dense_valid_len must be positive")
    if budget > dense_valid_len:
        raise ValueError(f"{sample_id}: dynamic budget exceeds dense_valid_len")

    row_positions = selected_positions_rows[batch_idx]
    row_mask = selected_mask_rows[batch_idx]
    if not isinstance(row_positions, list) or not isinstance(row_mask, list):
        raise ValueError(f"{sample_id}: selected positions and mask must be batch rows")
    if len(row_positions) < budget or len(row_mask) < budget:
        raise ValueError(f"{sample_id}: selected positions/mask shorter than budget")

    mask_values: list[bool] = []
    for idx, item in enumerate(row_mask):
        if item not in (0, 1, False, True):
            raise ValueError(f"{sample_id}: selected_mask[{idx}] must be binary")
        mask_values.append(bool(item))
    mask_count = int(sum(mask_values))
    if mask_count != budget:
        raise ValueError(f"{sample_id}: selected_mask true count must equal dynamic budget")
    if mask_values[:budget] != [True] * budget or any(mask_values[budget:]):
        raise ValueError(f"{sample_id}: selected_mask must be a contiguous prefix of length budget")

    selected: list[int] = []
    for idx, item in enumerate(row_positions[:budget]):
        pos = _strict_int_scalar(item, name=f"{sample_id}.selected_dense_positions[{idx}]")
        if pos < 0 or pos >= dense_valid_len:
            raise ValueError(f"{sample_id}: selected position outside dense_valid_len")
        selected.append(pos)
    if len(selected) != len(set(selected)):
        raise ValueError(f"{sample_id}: dynamic selected positions must be unique")
    if selected != sorted(selected):
        raise ValueError(f"{sample_id}: dynamic selected positions must be sorted")

    selected_set = set(selected)
    dense_selected_mask = [1 if idx in selected_set else 0 for idx in range(dense_valid_len)]
    budget_score = _optional_plan_batch_value(dynamic_plan, "budget_scores", batch_idx, batch_size)
    coverage_count = _optional_plan_batch_value(dynamic_plan, "coverage_counts", batch_idx, batch_size)
    value_count = _optional_plan_batch_value(dynamic_plan, "value_counts", batch_idx, batch_size)
    coverage_share = _optional_plan_batch_value(dynamic_plan, "coverage_share", batch_idx, batch_size)

    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample_id,
        "batch_index": int(batch_idx),
        "budget": int(budget),
        "dense_len": int(dense_valid_len),
        "valid_len": int(dense_valid_len),
        "selected_positions": selected,
        "selected_mask": dense_selected_mask,
        "duplicate_repair_count": 0,
        "invalid_repair_count": 0,
        "repair_fill_count": 0,
        "soft_hard_time_error": 0.0,
        "role_ids": [-1 for _pos in selected],
        "round_ids": [-1 for _pos in selected],
        "role_round_metadata": [
            {"position": pos, "role_id": -1, "round_id": -1}
            for pos in selected
        ],
        "dynamic_budget_plan": {
            "schema_version": str(dynamic_plan.get("schema_version", "")),
            "controller_family": str(dynamic_plan.get("controller_family", "")),
            "budget_score": None
            if budget_score is None
            else _strict_float_scalar(budget_score, name=f"{sample_id}.budget_score"),
            "coverage_count": None
            if coverage_count is None
            else _strict_int_scalar(coverage_count, name=f"{sample_id}.coverage_count"),
            "value_count": None
            if value_count is None
            else _strict_int_scalar(value_count, name=f"{sample_id}.value_count"),
            "coverage_share": None
            if coverage_share is None
            else _strict_float_scalar(coverage_share, name=f"{sample_id}.coverage_share"),
            "dynamic_budget_validation": False,
            "metric_claim_allowed": False,
            "paper_claim_allowed": False,
        },
        "resolver_generation": {
            "source": DYNAMIC_BUDGET_GENERATION_SOURCE,
            "diagnostic_or_deploy_only": True,
            "training_backprop_allowed": False,
            "detached_reader_tensors": True,
            "dynamic_budget_validation": False,
            "metric_claim_allowed": False,
            "paper_claim_allowed": False,
        },
    }


def resolve_pc_ot_mras_dynamic_budget_plan(
    dynamic_plan: Mapping[str, Any],
    *,
    sample_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Resolve an R22 dynamic-budget plan into hard-position rows.

    The input must already contain deploy-visible dynamic-budget tensors such as
    ``budgets``, ``dense_valid_len``, ``selected_dense_positions``, and
    ``selected_mask``. This resolver only validates and serializes that plan; it
    does not run detector evaluation or validate dynamic-budget quality.
    """

    if not isinstance(dynamic_plan, Mapping):
        raise ValueError("dynamic_budget_plan must be a mapping")
    _validate_dynamic_budget_plan_payload(dynamic_plan)

    budgets = _require_plan_key(dynamic_plan, "budgets")
    if not isinstance(budgets, list) or not budgets:
        raise ValueError("budgets must be a non-empty batch vector")
    batch_size = len(budgets)
    dense_valid_lens = _plan_batch_vector(
        _require_plan_key(dynamic_plan, "dense_valid_len"),
        name="dense_valid_len",
        batch_size=batch_size,
    )
    selected_positions_rows = _plan_batch_vector(
        _require_plan_key(dynamic_plan, "selected_dense_positions"),
        name="selected_dense_positions",
        batch_size=batch_size,
    )
    selected_mask_rows = _plan_batch_vector(
        _require_plan_key(dynamic_plan, "selected_mask"),
        name="selected_mask",
        batch_size=batch_size,
    )
    ids = list(sample_ids or [f"sample_{idx}" for idx in range(batch_size)])
    if len(ids) != batch_size:
        raise ValueError("sample_ids length must equal dynamic plan batch size")

    return [
        _resolve_dynamic_budget_plan_sample(
            dynamic_plan,
            batch_idx=batch_idx,
            batch_size=batch_size,
            sample_id=str(ids[batch_idx]),
            budgets=budgets,
            dense_valid_lens=dense_valid_lens,
            selected_positions_rows=selected_positions_rows,
            selected_mask_rows=selected_mask_rows,
        )
        for batch_idx in range(batch_size)
    ]


def _validate_hard_row_for_temporal_meta(row: Mapping[str, Any], *, row_idx: int) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise ValueError(f"row {row_idx}: hard-position row must be a mapping")
    _validate_no_forbidden_jsonl_keys(row, path=f"row[{row_idx}]")
    _validate_false_only_temporal_flags(row, path=f"row[{row_idx}]")
    if "dynamic_budget_plan" in row:
        _validate_dynamic_budget_plan_payload(row["dynamic_budget_plan"], path=f"row[{row_idx}].dynamic_budget_plan")

    schema = str(row.get("schema_version", ""))
    if schema != SCHEMA_VERSION:
        raise ValueError(f"row {row_idx}: schema_version must be {SCHEMA_VERSION}")
    sample_id = str(row.get("sample_id", f"sample_{row_idx}"))
    budget = _strict_int_scalar(row.get("budget"), name=f"row[{row_idx}].budget")
    dense_axis_len = _strict_int_scalar(row.get("dense_len"), name=f"row[{row_idx}].dense_len")
    valid_dense_len = _strict_int_scalar(row.get("valid_len", dense_axis_len), name=f"row[{row_idx}].valid_len")
    if budget <= 0:
        raise ValueError(f"row {row_idx}: budget must be positive")
    if dense_axis_len <= 0 or valid_dense_len <= 0:
        raise ValueError(f"row {row_idx}: dense_len and valid_len must be positive")
    if valid_dense_len > dense_axis_len:
        raise ValueError(f"row {row_idx}: valid_len must not exceed dense_len")

    selected = _as_int_list(row.get("selected_positions"), name=f"row[{row_idx}].selected_positions")
    if len(selected) != budget:
        raise ValueError(f"row {row_idx}: selected_positions length must equal budget")
    if len(selected) != len(set(selected)):
        raise ValueError(f"row {row_idx}: selected_positions must be unique")
    if selected != sorted(selected):
        raise ValueError(f"row {row_idx}: selected_positions must be sorted")
    if selected[0] < 0 or selected[-1] >= valid_dense_len:
        raise ValueError(f"row {row_idx}: selected_positions must stay inside valid_len")

    raw_mask = _to_plain(row.get("selected_mask"))
    if not isinstance(raw_mask, list):
        raise ValueError(f"row {row_idx}: selected_mask must be a dense-axis list")
    if len(raw_mask) != dense_axis_len:
        raise ValueError(f"row {row_idx}: selected_mask length must equal dense_len")
    dense_mask: list[bool] = []
    for pos, item in enumerate(raw_mask):
        if item not in (0, 1, False, True):
            raise ValueError(f"row {row_idx}: selected_mask[{pos}] must be binary")
        dense_mask.append(bool(item))
    if sum(dense_mask) != budget:
        raise ValueError(f"row {row_idx}: selected_mask true count must equal budget")
    mask_positions = [idx for idx, item in enumerate(dense_mask) if item]
    if mask_positions != selected:
        raise ValueError(f"row {row_idx}: selected_mask must match selected_positions")
    if any(dense_mask[valid_dense_len:]):
        raise ValueError(f"row {row_idx}: selected_mask must not select padded dense tail")

    return {
        "sample_id": sample_id,
        "budget": int(budget),
        "dense_axis_len": int(dense_axis_len),
        "valid_dense_len": int(valid_dense_len),
        "selected_positions": selected,
    }


def pc_ot_mras_hard_rows_to_temporal_metas(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Convert hard-position rows into deploy-visible detector metadata.

    The returned metadata is suitable for ``validate_sampling_contract`` and
    ``temporal_grid_from_metas``. It keeps GT/proposal axes in native dense time
    and does not validate dynamic-budget quality or detector accuracy.
    """

    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("rows must be a sequence of hard-position mappings")
    metas: list[dict[str, Any]] = []
    for row_idx, row in enumerate(rows):
        checked = _validate_hard_row_for_temporal_meta(row, row_idx=row_idx)
        selected = [float(pos) for pos in checked["selected_positions"]]
        valid_dense_len = int(checked["valid_dense_len"])
        meta = {
            "sample_id": checked["sample_id"],
            "irregular_selected_positions": selected,
            "irregular_dense_valid_len": valid_dense_len,
            "irregular_selected_valid_len": valid_dense_len,
            "irregular_selected_valid_len_semantics": "carried_forward_dense_valid_len_alias",
            "irregular_selected_count": int(checked["budget"]),
            "irregular_native_axis": True,
            "gt_axis": "dense",
            "segments_axis": "dense",
            "target_axis": "dense",
            "proposal_axis": "dense",
            "temporal_axis": "native_dense",
            "decode_axis": "dense",
            "pc_ot_mras_dynamic_budget_export": {
                "schema_version": TEMPORAL_METADATA_SCHEMA_VERSION,
                "source_schema_version": SCHEMA_VERSION,
                "generation_source": TEMPORAL_METADATA_GENERATION_SOURCE,
                "source_sample_id": checked["sample_id"],
                "source_dense_axis_len": int(checked["dense_axis_len"]),
                "source_valid_dense_len": valid_dense_len,
                "source_budget": int(checked["budget"]),
                "dynamic_budget_validation": False,
                "metric_claim_allowed": False,
                "paper_claim_allowed": False,
                "runtime_flops_claim_allowed": False,
                "deploy_claim_allowed": False,
                "scanner_quality_claim_allowed": False,
                "training_backprop_allowed": False,
            },
        }
        metas.append(meta)
    return metas


def resolve_pc_ot_mras_hard_positions(
    reader_out: Mapping[str, Any],
    *,
    budget: int,
    sample_ids: Sequence[str] | None = None,
    dense_len: int | None = None,
    valid_len: int | None = None,
) -> list[dict[str, Any]]:
    """Resolve soft PC-OT-MRAS reader output into fixed hard positions.

    The returned rows contain only plain JSON values. Tensor-like inputs are
    detached before ranking so this function cannot provide a differentiable
    training path.
    """

    if not isinstance(reader_out, Mapping):
        raise ValueError("reader_out must be a mapping")

    def _run() -> list[dict[str, Any]]:
        batch_size = _batch_size_from(reader_out)
        ids = list(sample_ids or [f"sample_{idx}" for idx in range(batch_size)])
        if len(ids) != batch_size:
            raise ValueError("sample_ids length must equal inferred batch size")
        return [
            _resolve_sample(
                reader_out,
                batch_idx=batch_idx,
                batch_size=batch_size,
                budget=int(budget),
                sample_id=str(ids[batch_idx]),
                dense_len=dense_len,
                valid_len=valid_len,
            )
            for batch_idx in range(batch_size)
        ]

    with _maybe_no_grad():
        return _run()


def export_pc_ot_mras_hard_positions(
    reader_out: Mapping[str, Any],
    valid_mask: Any | None = None,
    *,
    budget: int,
    sample_ids: Sequence[str] | None = None,
    dense_len: int | None = None,
    valid_len: int | None = None,
) -> list[dict[str, Any]]:
    payload = dict(reader_out)
    if valid_mask is not None:
        payload["valid_mask"] = valid_mask
    return resolve_pc_ot_mras_hard_positions(
        payload,
        budget=budget,
        sample_ids=sample_ids,
        dense_len=dense_len,
        valid_len=valid_len,
    )


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


def write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(strict_json_value(dict(row)), sort_keys=True) + "\n")


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(strict_json_value(dict(payload)), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_jsonl_export(input_jsonl: str | Path, output_jsonl: str | Path, *, budget: int, summary_json: str | Path | None = None) -> dict[str, Any]:
    source_rows = read_jsonl(input_jsonl)
    out_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(source_rows):
        _validate_no_forbidden_jsonl_keys(row, path=f"row[{idx}]")
        reader_out = row.get("reader_out", row)
        sample_ids = _row_sample_ids(row, row_idx=idx)
        dense_len = row.get("dense_len")
        valid_len = row.get("valid_len")
        resolved = resolve_pc_ot_mras_hard_positions(
            reader_out,
            budget=_row_budget(row, cli_budget=int(budget), row_idx=idx),
            sample_ids=sample_ids,
            dense_len=int(dense_len) if dense_len is not None else None,
            valid_len=int(valid_len) if valid_len is not None else None,
        )
        out_rows.extend(resolved)

    write_jsonl(output_jsonl, out_rows)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "decision": READY,
        "row_count": len(out_rows),
        "output_jsonl": str(output_jsonl),
        "total_duplicate_repair_count": sum(int(row["duplicate_repair_count"]) for row in out_rows),
        "total_repair_fill_count": sum(int(row["repair_fill_count"]) for row in out_rows),
    }
    if summary_json is not None:
        write_json(summary_json, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resolve PC-OT-MRAS soft reader output to hard positions.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json")
    parser.add_argument("--budget", type=int, required=True)
    args = parser.parse_args(argv)

    try:
        summary = run_jsonl_export(
            args.input_jsonl,
            args.output_jsonl,
            budget=int(args.budget),
            summary_json=args.summary_json,
        )
    except Exception as exc:  # pragma: no cover - CLI guard
        print(json.dumps({"schema_version": SUMMARY_SCHEMA_VERSION, "decision": NO_GO, "error": str(exc)}))
        return 1

    print(json.dumps(strict_json_value(summary), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
