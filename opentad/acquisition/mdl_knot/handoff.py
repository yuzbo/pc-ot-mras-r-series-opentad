from __future__ import annotations

import time
from typing import Callable, MutableMapping, Sequence

from .selector import greedy_mdl_knot_select
from .types import MDLKnotConfig, ScoutCurve
from .validators import (
    normalize_handoff_audit_mode,
    validate_real_sparse_handoff,
    validate_structural_sparse_handoff,
)


ProfileCallback = Callable[[str, float, dict | None], None]


def _gather_selected_inputs(dense_inputs: Sequence[object], selected_positions: Sequence[int]) -> list[object]:
    return [dense_inputs[int(pos)] for pos in selected_positions]


def _profile_elapsed(callback: ProfileCallback | None, stage: str, start: float, extra: dict | None = None) -> None:
    if callback is not None:
        callback(stage, time.perf_counter() - start, extra)


def apply_mdl_knot_to_dense_window(
    results: MutableMapping[str, object],
    dense_window: Sequence[int],
    scout_curve: ScoutCurve,
    config: MDLKnotConfig,
    adapter_target_len: int | None = None,
    dense_inputs: Sequence[object] | None = None,
    defer_handoff_validation: bool = False,
    handoff_audit_mode: str = "full_raw",
    profile_callback: ProfileCallback | None = None,
) -> MutableMapping[str, object]:
    audit_mode = normalize_handoff_audit_mode(handoff_audit_mode)
    if defer_handoff_validation and audit_mode == "full_raw" and dense_inputs is None:
        audit_mode = "sampled_raw"
    if len(dense_window) != scout_curve.dense_t:
        raise ValueError(f"dense_window length {len(dense_window)} must match scout dense_t {scout_curve.dense_t}")
    if dense_inputs is not None and len(dense_inputs) != len(dense_window):
        raise ValueError(f"dense_inputs audit length {len(dense_inputs)} must match dense_window {len(dense_window)}")
    ledger = greedy_mdl_knot_select(
        scout_curve,
        config,
        video_id=str(results.get("video_name", "unknown")),
        window_id=int(results.get("window_id", 0)),
        profile_callback=profile_callback,
    )
    selected_positions = list(ledger.selected_positions)
    frame_inds = [int(dense_window[pos]) for pos in selected_positions]
    valid_k = int(ledger.valid_k)
    if adapter_target_len is not None:
        adapter_target_len = int(adapter_target_len)
        if adapter_target_len < valid_k:
            raise ValueError(f"adapter_target_len {adapter_target_len} is smaller than MDL valid_k {valid_k}")
        if adapter_target_len > valid_k:
            frame_inds = frame_inds + [frame_inds[-1]] * (adapter_target_len - valid_k)
            masks = [True] * valid_k + [False] * (adapter_target_len - valid_k)
        else:
            masks = [True] * valid_k
    else:
        masks = [True] * valid_k
    start = time.perf_counter()
    sparse_meta = ledger.to_sparse_meta()
    sparse_meta_dict = sparse_meta.to_dict()
    _profile_elapsed(profile_callback, "handoff_metadata_build", start, {"valid_k": int(valid_k)})

    results["frame_inds"] = frame_inds
    results["num_clips"] = 1
    results["clip_len"] = len(frame_inds)
    results["masks"] = masks
    results["mdl_knot_selected_positions"] = selected_positions
    results["mdl_knot_selected_roles"] = list(ledger.selected_roles)
    results["mdl_knot_valid_k"] = valid_k
    results["mdl_knot_adapter_target_len"] = int(adapter_target_len or valid_k)
    results["mdl_knot_bridge"] = "fixed_pad" if adapter_target_len is not None else "variable_sparse"
    results["mdl_knot_padding_counts_as_valid"] = False
    results["mdl_knot_ledger"] = ledger.to_dict()
    results["mdl_knot_sparse_meta"] = sparse_meta_dict
    results["irregular_selected_positions"] = [float(v) for v in selected_positions]
    results["irregular_selected_valid_len"] = float(ledger.dense_t)
    results["irregular_native_axis"] = False

    expected_selected_frame_inds = [int(dense_window[pos]) for pos in selected_positions]
    if audit_mode == "full_raw" and dense_inputs is None:
        raise ValueError("MDL-Knot true sparse handoff requires dense raw inputs for gather validation")
    selected_inputs = _gather_selected_inputs(dense_inputs, selected_positions) if dense_inputs is not None else []
    if audit_mode == "full_raw":
        start = time.perf_counter()
        evidence_flags = validate_real_sparse_handoff(
            batch={
                "selected_inputs": selected_inputs,
                "dense_inputs": dense_inputs,
                "meta": sparse_meta_dict,
            },
            ledger=ledger,
        )
        _profile_elapsed(profile_callback, "handoff_validator", start, {"audit_mode": audit_mode})
    else:
        start = time.perf_counter()
        validate_structural_sparse_handoff(
            batch={
                "selected_frame_inds": expected_selected_frame_inds,
                "expected_selected_frame_inds": expected_selected_frame_inds,
                "detector_frame_inds": frame_inds,
                "masks": masks,
                "meta": sparse_meta_dict,
            },
            ledger=ledger,
        )
        _profile_elapsed(profile_callback, "handoff_validator", start, {"audit_mode": audit_mode})
        evidence_flags = {
            "full_observation_no_compression": False,
            "no_compression_edge_case": False,
            "sampled_raw_sparse_compute_evidence": False,
            "real_sparse_handoff_evidence": False,
            "raw_sparse_compute_evidence": False,
            "structural_only_handoff_evidence": True,
            "sparse_compute_claim": False,
            "no_sparse_compute_claim": True,
        }
    first_shape = getattr(selected_inputs[0], "shape", None) if selected_inputs else None
    full_observation = bool(evidence_flags.get("full_observation_no_compression", False))
    raw_values_compared = audit_mode == "full_raw" and not full_observation
    handoff_audit = {
        "audit_mode": "full_raw_full_observation" if audit_mode == "full_raw" and full_observation else audit_mode,
        "validation_mode": audit_mode,
        "validation_mode_alias": "full_dense" if audit_mode == "full_raw" else audit_mode,
        "selected_inputs_is_gathered": bool(raw_values_compared),
        "structural_sparse_handoff_validated": True,
        "raw_frame_values_compared": bool(raw_values_compared),
        "full_raw_dense_comparison": bool(raw_values_compared),
        "bounded_raw_sample_comparison": False,
        "formal_raw_handoff_evidence": bool(raw_values_compared),
        "full_observation_no_compression": full_observation,
        "no_compression_edge_case": full_observation,
        "sampled_raw_sparse_compute_evidence": False,
        "raw_sparse_compute_evidence": bool(raw_values_compared),
        "structural_only_handoff_evidence": bool(evidence_flags.get("structural_only_handoff_evidence", False)),
        "sparse_compute_claim": False,
        "no_sparse_compute_claim": True,
        "selected_len": valid_k,
        "dense_len": len(dense_inputs) if dense_inputs is not None else len(dense_window),
        "dense_raw_inputs_read": len(dense_inputs) if dense_inputs is not None else 0,
        "raw_audit_frame_count": len(dense_inputs) if dense_inputs is not None else 0,
        "dense_window_materialized_for_audit": bool(dense_inputs is not None),
        "raw_sample_shape": None if first_shape is None else [int(v) for v in first_shape],
        "raw_inputs_retained": False,
        "selected_frame_inds_prefix": [int(v) for v in frame_inds[:valid_k]],
        "selected_positions_prefix": [int(v) for v in selected_positions],
        "detector_frame_inds_len": len(frame_inds),
        "detector_frame_inds_prefix_is_sparse": len(set(frame_inds[:valid_k])) == valid_k,
        "detector_padding_repeats_last_selected": (
            frame_inds[valid_k:] == [frame_inds[valid_k - 1]] * (len(frame_inds) - valid_k)
            if len(frame_inds) > valid_k
            else True
        ),
    }
    results["mdl_knot_sparse_meta"]["selected_frame_inds_prefix"] = [int(v) for v in frame_inds[:valid_k]]
    results["mdl_knot_sparse_meta"]["detector_frame_inds_len"] = len(frame_inds)
    results["mdl_knot_sparse_meta"]["handoff_audit"] = handoff_audit
    results["mdl_knot_real_sparse_handoff_validated"] = bool(raw_values_compared)
    results["mdl_knot_handoff_audit_mode"] = audit_mode
    results["mdl_knot_handoff_audit"] = handoff_audit
    return results
