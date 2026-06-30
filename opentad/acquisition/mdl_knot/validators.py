from __future__ import annotations

from typing import Mapping, Sequence

from .types import MDL_KNOT_ROUTE_LABEL, KnotLedger


def _ledger_dict(ledger) -> dict:
    if isinstance(ledger, KnotLedger):
        return ledger.to_dict()
    if isinstance(ledger, Mapping):
        return dict(ledger)
    raise TypeError(f"unsupported ledger type: {type(ledger)!r}")


HANDOFF_AUDIT_MODE_ALIASES = {
    "structural": "structural",
    "metadata": "structural",
    "metadata_only": "structural",
    "sampled_raw": "sampled_raw",
    "selected_only": "sampled_raw",
    "selected_raw": "sampled_raw",
    "full_raw": "full_raw",
    "full_dense": "full_raw",
}


def normalize_handoff_audit_mode(mode: object) -> str:
    normalized = str(mode or "structural").strip().lower()
    if normalized not in HANDOFF_AUDIT_MODE_ALIASES:
        choices = ", ".join(sorted(HANDOFF_AUDIT_MODE_ALIASES))
        raise ValueError(f"unsupported MDL-Knot handoff audit mode {mode!r}; expected one of: {choices}")
    return HANDOFF_AUDIT_MODE_ALIASES[normalized]


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


def _to_numpy_for_compare(value):
    if hasattr(value, "asnumpy"):
        return value.asnumpy()
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    try:
        import numpy as np

        return np.asarray(value)
    except Exception:
        return value


def _first_axis_item(value, index: int):
    return value[int(index)]


def _first_axis_gather(value, positions: Sequence[int]):
    if hasattr(value, "shape") and not isinstance(value, (list, tuple)):
        return value[[int(pos) for pos in positions]]
    return [_first_axis_item(value, int(pos)) for pos in positions]


def _is_raw_sample_like(value) -> bool:
    arr = _to_numpy_for_compare(value)
    shape = getattr(arr, "shape", None)
    if shape is not None:
        return len(shape) >= 2
    return False


def _contains_raw_samples(selected_inputs) -> bool:
    if hasattr(selected_inputs, "shape") and not isinstance(selected_inputs, (list, tuple)):
        shape = getattr(selected_inputs, "shape")
        return len(shape) >= 3
    if len(selected_inputs) == 0:
        return False
    return _is_raw_sample_like(selected_inputs[0])


def _same_raw_values(left, right) -> bool:
    import numpy as np

    left_is_array = hasattr(left, "shape") and not isinstance(left, (list, tuple))
    right_is_array = hasattr(right, "shape") and not isinstance(right, (list, tuple))
    if left_is_array or right_is_array:
        return np.array_equal(_to_numpy_for_compare(left), _to_numpy_for_compare(right))
    if len(left) != len(right):
        return False
    for left_item, right_item in zip(left, right):
        if not np.array_equal(_to_numpy_for_compare(left_item), _to_numpy_for_compare(right_item)):
            return False
    return True


def _handoff_evidence_flags(selected_len: int, dense_t: int, audit_mode: str) -> dict:
    full_observation = int(selected_len) == int(dense_t)
    return {
        "audit_mode": str(audit_mode),
        "selected_len": int(selected_len),
        "dense_T": int(dense_t),
        "full_observation_no_compression": bool(full_observation),
        "no_compression_edge_case": bool(full_observation),
        "sampled_raw_sparse_compute_evidence": bool((not full_observation) and audit_mode == "sampled_raw"),
        "real_sparse_handoff_evidence": bool(not full_observation),
        "raw_sparse_compute_evidence": bool(not full_observation),
        "structural_only_handoff_evidence": bool(full_observation),
        "sparse_compute_claim": False,
        "no_sparse_compute_claim": True,
    }


def validate_real_sparse_handoff(batch: Mapping[str, object], ledger) -> dict:
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
    if selected_inputs is dense_inputs:
        raise ValueError("selected_inputs and dense_inputs must not be the same object")
    if selected_len == dense_len and selected_len != valid_k:
        raise ValueError("selected_inputs must be a real sparse gather, not dense passthrough")
    if selected_len != valid_k:
        raise ValueError(f"selected_inputs length {selected_len} must equal valid_k {valid_k}")
    if dense_len != dense_t:
        raise ValueError(f"dense_inputs audit length {dense_len} must equal dense_T {dense_t}")
    if selected_len > dense_t or selected_len > dense_len:
        raise ValueError("selected_inputs length cannot exceed dense audit length")
    if not _contains_raw_samples(selected_inputs):
        raise ValueError("selected_inputs must contain gathered raw frame/tensor samples, not frame indices")
    positions = [int(v) for v in data["selected_positions"]]
    gathered_inputs = _first_axis_gather(dense_inputs, positions)
    if not _same_raw_values(selected_inputs, gathered_inputs):
        raise ValueError("selected_inputs must equal dense_inputs gathered at ledger selected_positions")
    meta_positions = [int(v) for v in meta.get("selected_positions", [])]
    if meta_positions != positions:
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
    return _handoff_evidence_flags(selected_len, dense_t, "full_raw")


def validate_structural_sparse_handoff(batch: Mapping[str, object], ledger) -> None:
    """Validate sparse indices, metadata, and masks without reading raw frames."""
    data = _ledger_dict(ledger)
    validate_knot_ledger(data)
    meta = batch.get("meta", {})
    valid_k = int(data.get("valid_k", data.get("actual_k")))
    positions = [int(v) for v in data["selected_positions"]]
    meta_positions = [int(v) for v in meta.get("selected_positions", [])]
    if meta_positions != positions:
        raise ValueError("meta selected_positions must match ledger selected_positions")
    if int(meta.get("valid_k", valid_k)) != valid_k:
        raise ValueError("meta valid_k must match ledger valid_k")
    selected_frame_inds = [int(v) for v in batch.get("selected_frame_inds", [])]
    expected = batch.get("expected_selected_frame_inds")
    if selected_frame_inds:
        if expected is not None and selected_frame_inds != [int(v) for v in expected]:
            raise ValueError("selected_frame_inds must match dense_window gathered at selected_positions")
        if len(selected_frame_inds) != valid_k:
            raise ValueError("selected_frame_inds length must equal valid_k")
    detector_frame_inds = batch.get("detector_frame_inds")
    if detector_frame_inds is not None:
        detector_frame_inds = [int(v) for v in detector_frame_inds]
        if detector_frame_inds[:valid_k] != selected_frame_inds:
            raise ValueError("detector frame_inds prefix must match selected sparse frame indices")
    visibility = meta.get("visibility_mask", batch.get("masks", [True] * valid_k))
    if len(visibility) < valid_k:
        raise ValueError("visibility_mask shorter than valid_k")
    if sum(bool(v) for v in visibility[:valid_k]) != valid_k:
        raise ValueError("all valid sparse entries must be visible")
    if meta.get("position_unit", "original_dense_time_index") != "original_dense_time_index":
        raise ValueError("meta position_unit must be original_dense_time_index")


def validate_sampled_sparse_handoff(batch: Mapping[str, object], ledger) -> dict:
    """Validate true sparse raw-frame handoff without materializing dense inputs.

    This is used by short diagnostics where reading every dense-window frame only
    for audit can dominate runtime. It still requires real raw/tensor samples for
    every selected sparse position and rejects index-only inputs. If a short
    dense window selects every position, the audit is valid as a full-observation
    no-compression edge case, but it is not sparse selected-only evidence.
    """
    data = _ledger_dict(ledger)
    validate_knot_ledger(data)
    selected_inputs = batch.get("selected_inputs")
    meta = batch.get("meta", {})
    if selected_inputs is None:
        raise ValueError("batch must contain selected_inputs")
    selected_len = _sequence_len(selected_inputs)
    valid_k = int(data.get("valid_k", data.get("actual_k")))
    dense_t = int(data.get("dense_T"))
    if selected_len != valid_k:
        raise ValueError(f"selected_inputs length {selected_len} must equal valid_k {valid_k}")
    if selected_len > dense_t:
        raise ValueError("selected_inputs length cannot exceed dense_T")
    if not _contains_raw_samples(selected_inputs):
        raise ValueError("selected_inputs must contain gathered raw frame/tensor samples, not frame indices")
    positions = [int(v) for v in data["selected_positions"]]
    meta_positions = [int(v) for v in meta.get("selected_positions", [])]
    if meta_positions != positions:
        raise ValueError("meta selected_positions must match ledger selected_positions")
    if int(meta.get("valid_k", valid_k)) != valid_k:
        raise ValueError("meta valid_k must match ledger valid_k")
    selected_frame_inds = [int(v) for v in batch.get("selected_frame_inds", [])]
    if selected_frame_inds:
        expected = batch.get("expected_selected_frame_inds")
        if expected is not None and selected_frame_inds != [int(v) for v in expected]:
            raise ValueError("selected_frame_inds must match dense_window gathered at selected_positions")
        if len(selected_frame_inds) != valid_k:
            raise ValueError("selected_frame_inds length must equal valid_k")
    visibility = meta.get("visibility_mask", [True] * valid_k)
    if len(visibility) < valid_k:
        raise ValueError("visibility_mask shorter than valid_k")
    if sum(bool(v) for v in visibility[:valid_k]) != valid_k:
        raise ValueError("all valid sparse entries must be visible")
    if meta.get("position_unit", "original_dense_time_index") != "original_dense_time_index":
        raise ValueError("meta position_unit must be original_dense_time_index")
    return _handoff_evidence_flags(selected_len, dense_t, "sampled_raw")


def validate_selected_sparse_handoff(batch: Mapping[str, object], ledger) -> None:
    validate_sampled_sparse_handoff(batch, ledger)

