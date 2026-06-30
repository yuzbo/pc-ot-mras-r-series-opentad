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
    "BOUNDARY" + "_MICROSCOPE",
    "Boundary" + " Microscope",
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
        "raw_detector_outputs": "detector feedback",
        "detector_outputs": "detector feedback",
        "detector_predictions": "detector feedback",
        "dense_backbone_features": "dense raw backbone handoff",
        "dense_backbone_handoff": "dense raw backbone handoff",
        "dense_raw_backbone_handoff": "dense raw backbone handoff",
        "gt_scout_curve": "GT-derived scout",
        "gt_frame_signal": "GT-derived scout",
        "teacher_scout_curve": "teacher-derived scout",
        "prediction_scout_curve": "prediction-cache-derived scout",
        "detector_scout_curve": "detector-feedback-derived scout",
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
        "deploy_visible_scout_or_explicit_fallback_ok",
    )
    for key in required_true:
        if summary.get(key) is not True:
            raise ABRValidationError(f"LOCKED: summary.{key} must be true")
    if int(summary.get("detector_forward_count", -1)) != 1:
        raise ABRValidationError("LOCKED: detector_forward_count must be 1")
    return {"allowed_next_action": "REMOTE_PRECHECK_ONLY_REQUEST", "still_locked": "TRAIN_EVAL_SYNC_STAGE_COMMIT_PUSH"}


def validate_first_round_bracket_diagnostics(
    diagnostics: Mapping[str, Any],
    min_recall: float = 0.95,
    min_transition_coverage: float = 0.95,
    max_temporal_coverage_fraction: float = 0.70,
    max_bracket_width_fraction: float = 0.30,
    require_deploy_visible_scout: bool = False,
) -> Dict[str, Any]:
    if not isinstance(diagnostics, Mapping):
        raise ABRValidationError("LOCKED: first-round bracket diagnostics must be a mapping")
    if require_deploy_visible_scout:
        source = str(diagnostics.get("scout_source", ""))
        if diagnostics.get("diagnostic_fallback_used") is True or source.startswith("diagnostic_fallback:"):
            raise ABRValidationError("LOCKED: formal ABR requires deploy-visible scout; diagnostic fallback is rejected")
    transition_count = _require_nonnegative_int(diagnostics, "transition_count")
    bracketed_transition_count = _require_nonnegative_int(diagnostics, "bracketed_transition_count")
    missed = _require_nonnegative_int(diagnostics, "missed_transition_count")
    if transition_count <= 0:
        raise ABRValidationError("LOCKED: formal first-round evidence requires transition_count > 0")
    if bracketed_transition_count > transition_count:
        raise ABRValidationError("LOCKED: bracketed_transition_count exceeds transition_count")
    if missed > transition_count:
        raise ABRValidationError("LOCKED: missed_transition_count exceeds transition_count")
    if bracketed_transition_count + missed != transition_count:
        raise ABRValidationError(
            "LOCKED: first-round transition counts are inconsistent; "
            "bracketed_transition_count + missed_transition_count must equal transition_count"
        )
    recall = _require_fraction(diagnostics, "first_round_bracket_recall")
    coverage = _require_fraction(diagnostics, "first_round_transition_coverage")
    expected_recall = bracketed_transition_count / float(transition_count)
    if abs(recall - expected_recall) > 1e-6:
        raise ABRValidationError(
            "LOCKED: first_round_bracket_recall is inconsistent with "
            "bracketed_transition_count / transition_count"
        )
    if recall < float(min_recall):
        raise ABRValidationError(
            f"LOCKED: first_round_bracket_recall {recall:.4f} below required {float(min_recall):.4f}"
        )
    if coverage < float(min_transition_coverage):
        raise ABRValidationError(
            "LOCKED: first_round_transition_coverage "
            f"{coverage:.4f} below required {float(min_transition_coverage):.4f}"
        )
    if missed > 0:
        raise ABRValidationError(f"LOCKED: first-round bracket diagnostics report {missed} missed transitions")
    if bracketed_transition_count != transition_count:
        raise ABRValidationError(
            "LOCKED: formal first-round evidence requires bracketed_transition_count == transition_count"
        )
    if abs(coverage - 1.0) > 1e-6:
        raise ABRValidationError(
            "LOCKED: first_round_transition_coverage is inconsistent with fully bracketed transitions"
        )
    temporal_coverage = None
    if "first_round_temporal_coverage_fraction" in diagnostics:
        temporal_coverage = _require_fraction(diagnostics, "first_round_temporal_coverage_fraction")
        if temporal_coverage > float(max_temporal_coverage_fraction):
            raise ABRValidationError(
                "LOCKED: first_round_temporal_coverage_fraction "
                f"{temporal_coverage:.4f} exceeds allowed {float(max_temporal_coverage_fraction):.4f}; "
                "temporal_coverage guard rejects overwide bracket evidence"
            )
    width_fraction = None
    if "max_bracket_width_fraction" in diagnostics:
        width_fraction = _require_fraction(diagnostics, "max_bracket_width_fraction")
        if width_fraction > float(max_bracket_width_fraction):
            raise ABRValidationError(
                "LOCKED: max_bracket_width_fraction "
                f"{width_fraction:.4f} exceeds allowed {float(max_bracket_width_fraction):.4f}; "
                "single-bracket width guard rejects overwide shortcut evidence"
            )
    return {
        "transition_count": int(transition_count),
        "bracketed_transition_count": int(bracketed_transition_count),
        "missed_transition_count": int(missed),
        "first_round_bracket_recall": float(recall),
        "first_round_transition_coverage": float(coverage),
        "first_round_temporal_coverage_fraction": None
        if temporal_coverage is None
        else float(temporal_coverage),
        "max_bracket_width_fraction": None if width_fraction is None else float(width_fraction),
    }


def validate_formal_readiness_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    assert_no_forbidden_route_tokens(payload)
    if payload.get("route_label") != ABR_ROUTE_LABEL:
        raise ABRValidationError("formal gate requires the ABR route label")
    if payload.get("method") != "abr_active_bracket_refinement":
        raise ABRValidationError("formal gate requires method=abr_active_bracket_refinement")
    if payload.get("status") != "PASS_FORMAL_READINESS_EVIDENCE":
        raise ABRValidationError("LOCKED: formal gate requires PASS_FORMAL_READINESS_EVIDENCE, not PRECHECK_ONLY")
    summary = payload.get("summary", {})
    if not isinstance(summary, Mapping):
        raise ABRValidationError("LOCKED: formal gate requires summary mapping")
    if summary.get("diagnostic_fallback_used") is True:
        raise ABRValidationError("LOCKED: formal gate rejects diagnostic fallback scout")
    scout_source = str(summary.get("scout_source", ""))
    if not scout_source or scout_source.startswith("diagnostic_fallback:"):
        raise ABRValidationError("LOCKED: formal gate requires a deploy-visible scout source")
    if str(summary.get("fallback_stage", "")).upper() == "PRECHECK_ONLY":
        raise ABRValidationError("LOCKED: formal gate rejects PRECHECK_ONLY fallback stage")
    if int(summary.get("detector_forward_count", -1)) != 1:
        raise ABRValidationError("LOCKED: detector_forward_count must be 1")
    diagnostics = summary.get("first_round_bracket_diagnostics", {})
    thresholds = summary.get("formal_thresholds", {})
    if not isinstance(thresholds, Mapping):
        thresholds = {}
    metrics = validate_first_round_bracket_diagnostics(
        diagnostics,
        min_recall=float(thresholds.get("min_first_round_bracket_recall", 0.95)),
        min_transition_coverage=float(thresholds.get("min_first_round_transition_coverage", 0.95)),
        max_temporal_coverage_fraction=float(thresholds.get("max_first_round_temporal_coverage_fraction", 0.70)),
        max_bracket_width_fraction=float(thresholds.get("max_first_round_bracket_width_fraction", 0.30)),
        require_deploy_visible_scout=True,
    )
    return {
        "allowed_next_action": "FORMAL_REVIEW_PACKET_ONLY",
        "formal_readiness_evidence_ok": True,
        "full_train_unlocked": False,
        "route_label": ABR_ROUTE_LABEL,
        "metrics": metrics,
        "still_locked": [
            "FORMAL_FULL_TRAIN_PENDING_REVIEW_AND_COORDINATOR_DECISION",
            "MAPPAPER_CLAIM",
            "RUNTIME_OR_SPARSE_COMPUTE_CLAIM",
            "DEPLOY_CLAIM",
        ],
    }


def _require_fraction(payload: Mapping[str, Any], key: str) -> float:
    if key not in payload:
        raise ABRValidationError(f"LOCKED: missing first-round diagnostic field {key}")
    value = float(payload[key])
    if value < 0.0 or value > 1.0:
        raise ABRValidationError(f"LOCKED: first-round diagnostic field {key} must be in [0, 1]")
    return value


def _require_nonnegative_int(payload: Mapping[str, Any], key: str) -> int:
    if key not in payload:
        raise ABRValidationError(f"LOCKED: missing first-round diagnostic field {key}")
    value = payload[key]
    if isinstance(value, bool):
        raise ABRValidationError(f"LOCKED: first-round diagnostic field {key} must be a non-negative integer")
    try:
        int_value = int(value)
        float_value = float(value)
    except (TypeError, ValueError):
        raise ABRValidationError(f"LOCKED: first-round diagnostic field {key} must be a non-negative integer")
    if int_value != float_value or int_value < 0:
        raise ABRValidationError(f"LOCKED: first-round diagnostic field {key} must be a non-negative integer")
    return int_value


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
