from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Sequence

from .types import ABR_ROUTE_LABEL, DEFAULT_PROVENANCE


class ABRValidationError(RuntimeError):
    pass


FORBIDDEN_ROUTE_TOKENS = (
    "C3-Pro",
    "C3_MAINLINE",
    "C3_ORIGINAL",
    "GlobalRank",
    "Interval",
)


def assert_no_forbidden_route_tokens(payload: Any) -> None:
    for path, value in _walk_string_values(payload):
        if path == "route_label" and value == ABR_ROUTE_LABEL:
            continue
        for token in FORBIDDEN_ROUTE_TOKENS:
            if token.lower() in value.lower():
                raise ABRValidationError(f"forbidden route token {token!r} found at {path}")


def assert_provenance_clean(provenance: Mapping[str, bool]) -> None:
    expected = dict(DEFAULT_PROVENANCE)
    for key, expected_value in expected.items():
        if bool(provenance.get(key)) != expected_value:
            raise ABRValidationError(f"ABR provenance flag {key} expected {expected_value}, got {provenance.get(key)}")


def assert_no_forbidden_selection_inputs(results: Mapping[str, Any], allow_gt_after_selection: bool = False) -> None:
    forbidden = {
        "teacher_logits": "teacher",
        "teacher_features": "teacher",
        "prediction_cache": "prediction cache",
        "prediction_cache_path": "prediction cache",
        "detector_feedback": "detector feedback",
        "raw_predictions": "detector feedback",
        "dense_backbone_features": "dense raw backbone handoff",
    }
    for key, reason in forbidden.items():
        if key in results:
            raise ABRValidationError(f"ABR selection forbids {reason}: {key}")
    if not allow_gt_after_selection:
        for key in ("gt_segments", "gt_labels"):
            if key in results:
                raise ABRValidationError(f"ABR selection forbids GT in this pre-decode selection path: {key}")


def assert_real_sparse_handoff(handoff: Mapping[str, Any]) -> None:
    provenance = handoff.get("provenance", {})
    assert_provenance_clean(provenance)
    dense_t = int(handoff.get("dense_T", 0))
    local_positions = [int(pos) for pos in handoff.get("selected_positions_window_local", [])]
    global_positions = [int(pos) for pos in handoff.get("selected_positions_original_dense", [])]
    frame_inds_raw = [int(pos) for pos in handoff.get("frame_inds_raw", [])]
    dense_window = [int(pos) for pos in handoff.get("dense_window", [])]
    selected_mask = [bool(v) for v in handoff.get("selected_mask", [True] * len(local_positions))]
    valid_k = int(handoff.get("valid_k", len(local_positions)))

    if dense_t <= 0:
        raise ABRValidationError("dense_T must be positive")
    if valid_k <= 0:
        raise ABRValidationError("valid_k must be positive")
    if not local_positions or not global_positions or not frame_inds_raw:
        raise ABRValidationError("handoff must include local positions, original/global positions, and raw frame_inds")
    if len(local_positions) != len(global_positions) or len(local_positions) != len(frame_inds_raw):
        raise ABRValidationError("local/global/frame handoff vectors must have identical lengths")
    if len(dense_window) != dense_t:
        raise ABRValidationError("dense_window length must equal dense_T for local-to-global validation")
    if valid_k > len(local_positions) or valid_k > len(frame_inds_raw) or valid_k > len(selected_mask):
        raise ABRValidationError("valid_k exceeds handoff vector lengths")
    if sum(1 for item in selected_mask if item) != valid_k:
        raise ABRValidationError("selected_mask true count must equal valid_k; padded duplicates are invalid")

    valid_local = local_positions[:valid_k]
    valid_global = global_positions[:valid_k]
    valid_frames = frame_inds_raw[:valid_k]
    if valid_local != sorted(set(valid_local)):
        raise ABRValidationError("valid local selected positions must be sorted unique")
    if min(valid_local) < 0 or max(valid_local) >= dense_t:
        raise ABRValidationError("local selected positions out of dense-window range")
    expected_global = [dense_window[pos] for pos in valid_local]
    if valid_global != expected_global:
        raise ABRValidationError("global selected positions must equal dense_window[local_positions]")
    if valid_frames != valid_global:
        raise ABRValidationError("raw frame_inds must equal original/global selected positions")
    if valid_k >= dense_t:
        raise ABRValidationError("handoff is not sparse: valid_k must be smaller than dense_T")
    if any(selected_mask[valid_k:]):
        raise ABRValidationError("padding entries after valid_k must be invalid")


def validate_launch_gate_payload(payload: Mapping[str, Any]) -> Dict[str, str]:
    assert_no_forbidden_route_tokens(payload)
    if payload.get("route_label") != ABR_ROUTE_LABEL:
        raise ABRValidationError("launch gate requires the ABR route label")
    if payload.get("method") != "abr_active_bracket_refinement":
        raise ABRValidationError("launch gate requires method=abr_active_bracket_refinement")
    if payload.get("status") != "PASS_PRECHECK_ONLY":
        raise ABRValidationError("LOCKED: precheck status is not PASS_PRECHECK_ONLY")
    if payload.get("precheck_validated") is not True:
        raise ABRValidationError("LOCKED: precheck_validated must be true")
    summary = payload.get("summary", {})
    required_true = (
        "real_sparse_handoff_ok",
        "forbidden_inputs_ok",
        "nonzero_window_ok",
        "val_test_gt_rejection_ok",
        "dynamic_k_nonconstant",
    )
    for key in required_true:
        if summary.get(key) is not True:
            raise ABRValidationError(f"LOCKED: summary.{key} must be true")
    if int(summary.get("detector_forward_count", -1)) != 1:
        raise ABRValidationError("LOCKED: detector_forward_count must be 1")
    return {"allowed_next_action": "REMOTE_PRECHECK_ONLY_REQUEST", "still_locked": "TRAIN_EVAL_SYNC_STAGE_COMMIT_PUSH"}


def _walk_string_values(payload: Any, prefix: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(payload, str):
        yield prefix, payload
        return
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            child = str(key) if prefix == "" else f"{prefix}.{key}"
            yield from _walk_string_values(value, child)
        return
    if isinstance(payload, Sequence) and not isinstance(payload, (bytes, bytearray)):
        for idx, value in enumerate(payload):
            yield from _walk_string_values(value, f"{prefix}[{idx}]")
