from __future__ import annotations

from typing import Mapping, Sequence

from .types import MDL_KNOT_ROUTE_LABEL, KnotLedger


def _ledger_dict(ledger) -> dict:
    if isinstance(ledger, KnotLedger):
        return ledger.to_dict()
    if isinstance(ledger, Mapping):
        return dict(ledger)
    raise TypeError(f"unsupported ledger type: {type(ledger)!r}")


def validate_no_forbidden_sources(provenance: Mapping[str, object]) -> None:
    checks = {
        "uses_gt": False,
        "uses_teacher": False,
        "uses_prediction_cache": False,
        "dense_raw_backbone_handoff": False,
        "selected_inputs_is_gathered": True,
    }
    for key, expected in checks.items():
        value = bool(provenance.get(key, False if expected is False else True))
        if value is not expected:
            raise ValueError(f"forbidden provenance field {key}={value}, expected {expected}")
    unit = provenance.get("position_unit", "original_dense_time_index")
    if unit != "original_dense_time_index":
        raise ValueError(f"position_unit must be original_dense_time_index, got {unit}")


def validate_knot_ledger(ledger) -> None:
    data = _ledger_dict(ledger)
    if data.get("route_label") != MDL_KNOT_ROUTE_LABEL:
        raise ValueError(f"unexpected route_label: {data.get('route_label')}")
    dense_t = int(data.get("dense_T", data.get("dense_t", -1)))
    positions = [int(v) for v in data.get("selected_positions", [])]
    roles = list(data.get("selected_roles", []))
    valid_k = int(data.get("valid_k", data.get("actual_k", len(positions))))
    if dense_t <= 0:
        raise ValueError("dense_T must be positive")
    if len(positions) == 0:
        raise ValueError("selected_positions must not be empty")
    if positions != sorted(set(positions)):
        raise ValueError("selected_positions must be sorted and unique")
    if positions[0] < 0 or positions[-1] >= dense_t:
        raise ValueError("selected_positions must be in original dense range")
    if valid_k != len(positions):
        raise ValueError(f"valid_k {valid_k} does not match selected positions {len(positions)}")
    if len(roles) != len(positions):
        raise ValueError("selected_roles length must match selected_positions")
    if data.get("position_unit", "original_dense_time_index") != "original_dense_time_index":
        raise ValueError("ledger position_unit must be original_dense_time_index")
    provenance = data.get("provenance", {})
    validate_no_forbidden_sources(provenance)


def _sequence_len(value) -> int:
    if hasattr(value, "shape"):
        shape = getattr(value, "shape")
        if len(shape) == 0:
            return 0
        return int(shape[0])
    return len(value)


def validate_real_sparse_handoff(batch: Mapping[str, object], ledger) -> None:
    data = _ledger_dict(ledger)
    validate_knot_ledger(data)
    selected_inputs = batch.get("selected_inputs")
    dense_inputs = batch.get("dense_inputs")
    meta = batch.get("meta", {})
    if selected_inputs is None:
        raise ValueError("batch must contain selected_inputs")
    if dense_inputs is None:
        raise ValueError("batch must contain dense_inputs for audit comparison")
    selected_len = _sequence_len(selected_inputs)
    dense_len = _sequence_len(dense_inputs)
    valid_k = int(data.get("valid_k", data.get("actual_k")))
    dense_t = int(data.get("dense_T"))
    if valid_k >= dense_t:
        raise ValueError(f"valid_k {valid_k} must be shorter than dense_T {dense_t} for sparse selected-only audit")
    if selected_len != valid_k:
        raise ValueError(f"selected_inputs length {selected_len} must equal valid_k {valid_k}")
    if dense_len != dense_t:
        raise ValueError(f"dense_inputs audit length {dense_len} must equal dense_T {dense_t}")
    if selected_len >= dense_len:
        raise ValueError("selected_inputs must be a real sparse gather, not dense passthrough")
    if selected_inputs is dense_inputs:
        raise ValueError("selected_inputs and dense_inputs must not be the same object")
    meta_positions = [int(v) for v in meta.get("selected_positions", [])]
    if meta_positions != [int(v) for v in data["selected_positions"]]:
        raise ValueError("meta selected_positions must match ledger selected_positions")
    if int(meta.get("valid_k", valid_k)) != valid_k:
        raise ValueError("meta valid_k must match ledger valid_k")
    visibility = meta.get("visibility_mask", [True] * valid_k)
    if len(visibility) < valid_k:
        raise ValueError("visibility_mask shorter than valid_k")
    if sum(bool(v) for v in visibility[:valid_k]) != valid_k:
        raise ValueError("all valid sparse entries must be visible")
    if meta.get("position_unit", "original_dense_time_index") != "original_dense_time_index":
        raise ValueError("meta position_unit must be original_dense_time_index")

