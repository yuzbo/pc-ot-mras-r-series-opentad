from collections import Counter

import numpy as np

from .adapter_bridge import ADAPTER_FIXED_LENGTH_PADDED_BRIDGE, build_detector_feature_centers_from_raw
from .scaffold import gap_statistics
from .types import FORBIDDEN_DEPLOY_KEYS, FORBIDDEN_ROUTE_TOKENS, ROUTE_LABEL, STOP_REASONS, sorted_unique_positions

LOCAL_GATHER_CLAIM_STATUS = "local_gather_smoke_only_no_sparse_compute_or_metric_claim"
SPARSE_FORWARD_SHAPE_ONLY_CLAIM_STATUS = "sparse_forward_precheck_shape_only_no_metric_claim"
SPARSE_FORWARD_REAL_MODULE_CLAIM_STATUS = "sparse_forward_precheck_real_module_no_metric_claim"

SPARSE_FORWARD_PASS_SHAPE_ONLY = "PASS_LOCAL_SHAPE_ONLY_NO_SPARSE_COMPUTE_CLAIM"
SPARSE_FORWARD_PASS_REAL_MODULE = "PASS_REAL_MODULE_FORWARD_NO_METRIC_CLAIM"

SPARSE_FORWARD_FAIL_DENSE_RAW_HANDOFF = "FAIL_DENSE_RAW_HANDOFF"
SPARSE_FORWARD_FAIL_BACKBONE_DENSE_CHUNK_COUNT = "FAIL_BACKBONE_DENSE_CHUNK_COUNT"
SPARSE_FORWARD_FAIL_DETECTOR_PAD_CONFUSED_AS_VALID = "FAIL_DETECTOR_PAD_CONFUSED_AS_VALID"
SPARSE_FORWARD_FAIL_ORIGINAL_TIME_DECODE_MISSING = "FAIL_ORIGINAL_TIME_DECODE_MISSING"
SPARSE_FORWARD_FAIL_LEAKAGE_FIELD_PRESENT = "FAIL_LEAKAGE_FIELD_PRESENT"
SPARSE_FORWARD_FAIL_ROUTE_MIXING = "FAIL_ROUTE_MIXING"
SPARSE_FORWARD_FAIL_SPARSE_CLAIM_UNLOCKED = "FAIL_SPARSE_COMPUTE_CLAIM_UNLOCKED"

VALUE_COMPONENT_KEYS = {
    "expected_entropy_reduction",
    "expected_width_reduction",
    "expected_gap_risk_reduction",
    "short_action_value",
    "two_sided_witness_value",
    "predicted_regret",
    "value_per_cost",
    "actionness_component",
}

FORBIDDEN_DEPLOY_KEY_ALIASES = (
    "ground_truth",
    "label_cache",
)


def _walk_dict(obj, prefix=""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield path, key, value
            yield from _walk_dict(value, path)
    elif isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            yield from _walk_dict(value, f"{prefix}[{idx}]")


def validate_route_identity(metadata, allow_tokens=None):
    allow_tokens = set() if allow_tokens is None else set(allow_tokens)
    text = repr(metadata)
    route_label = None
    if isinstance(metadata, dict):
        route_label = metadata.get("route_label")
    if route_label is not None and route_label != ROUTE_LABEL:
        raise ValueError(f"route_label must be {ROUTE_LABEL}, got {route_label}")
    for token in FORBIDDEN_ROUTE_TOKENS:
        if token in allow_tokens:
            continue
        if token == "C3" and ROUTE_LABEL in text:
            text_without_label = text.replace(ROUTE_LABEL, "")
        else:
            text_without_label = text
        if token.lower() in text_without_label.lower():
            raise ValueError(f"BVR-TWB route metadata contains forbidden route token: {token}")
    return True


def validate_no_leakage(metadata):
    for path, key, value in _walk_dict(metadata):
        key_l = str(key).lower()
        for forbidden in tuple(FORBIDDEN_DEPLOY_KEYS) + FORBIDDEN_DEPLOY_KEY_ALIASES:
            if forbidden.lower() == key_l and value not in (None, False, [], {}):
                raise ValueError(f"deploy/scout metadata contains forbidden leakage field at {path}")
        if key_l.startswith("selection_uses_") and bool(value):
            raise ValueError(f"selection provenance flag must be false at {path}")
    return True


def validate_selected_positions(selected_positions, dense_T, valid_k=None):
    positions = list(selected_positions)
    sorted_unique = sorted_unique_positions(positions, dense_T)
    if positions != sorted_unique:
        raise ValueError("selected_positions must be sorted unique original dense indices")
    if valid_k is not None and int(valid_k) != len(sorted_unique):
        raise ValueError(f"valid_k {valid_k} != len(selected_positions) {len(sorted_unique)}")
    return True


def build_original_time_metadata(dense_T, selected_positions, fps, window_start_sec=0.0, window_end_sec=None):
    dense_T = int(dense_T)
    positions = sorted_unique_positions(selected_positions, dense_T)
    fps = float(fps)
    if fps <= 0:
        raise ValueError("fps must be positive")
    if window_end_sec is None:
        window_end_sec = float(window_start_sec) + dense_T / fps
    selected_times = [float(window_start_sec) + float(pos) / fps for pos in positions]
    cell_boundaries = []
    for pos in positions:
        start = float(window_start_sec) + (float(pos) - 0.5) / fps
        end = float(window_start_sec) + (float(pos) + 0.5) / fps
        cell_boundaries.append([max(float(window_start_sec), start), min(float(window_end_sec), end)])
    gaps_left = []
    gaps_right = []
    for idx, pos in enumerate(positions):
        gaps_left.append(int(pos - positions[idx - 1]) if idx > 0 else int(pos + 1))
        gaps_right.append(int(positions[idx + 1] - pos) if idx + 1 < len(positions) else int(dense_T - pos))
    return {
        "dense_T": dense_T,
        "fps": fps,
        "window_start_sec": float(window_start_sec),
        "window_end_sec": float(window_end_sec),
        "selected_positions": positions,
        "selected_positions_unit": "original_dense_index",
        "selected_times_sec": selected_times,
        "irregular_cell_boundaries_sec": cell_boundaries,
        "gap_left": gaps_left,
        "gap_right": gaps_right,
        "visibility": [1.0 for _ in positions],
        "time_axis_mode": "irregular_original_time",
        "selected_index_is_time": False,
    }


def validate_original_time_metadata(metadata):
    required = {"dense_T", "selected_positions", "selected_positions_unit", "time_axis_mode", "selected_index_is_time"}
    missing = sorted(required.difference(metadata.keys()))
    if missing:
        raise ValueError(f"original-time metadata missing fields: {missing}")
    if metadata.get("selected_positions_unit") != "original_dense_index":
        raise ValueError("selected_positions_unit must be original_dense_index")
    if metadata.get("selected_index_is_time") is True:
        raise ValueError("selected_index_is_time must be false")
    if metadata.get("time_axis_mode") != "irregular_original_time":
        raise ValueError("time_axis_mode must be irregular_original_time")
    validate_selected_positions(metadata["selected_positions"], metadata["dense_T"])
    if "selected_times_sec" in metadata and len(metadata["selected_times_sec"]) != len(metadata["selected_positions"]):
        raise ValueError("selected_times_sec length must match selected_positions")
    if "irregular_cell_boundaries_sec" in metadata and len(metadata["irregular_cell_boundaries_sec"]) != len(metadata["selected_positions"]):
        raise ValueError("cell boundary length must match selected_positions")
    return True


def validate_sparse_gather_evidence(evidence, dense_T, valid_k, require_detector_forward=False):
    if int(evidence.get("dense_temporal_len", -1)) != int(dense_T):
        raise ValueError("sparse gather dense_temporal_len mismatch")
    if int(evidence.get("selected_temporal_len", -1)) != int(valid_k):
        raise ValueError("sparse gather selected temporal length must equal valid_k")
    if int(valid_k) >= int(dense_T):
        raise ValueError("sparse gather selected temporal length must be smaller than dense_T")
    if bool(evidence.get("dense_raw_backbone_handoff", True)):
        raise ValueError("dense_raw_backbone_handoff must be false")
    if not bool(evidence.get("selected_inputs_is_gathered", False)):
        raise ValueError("selected_inputs_is_gathered must be true")
    if not bool(evidence.get("fingerprint_checked", False)):
        raise ValueError("fingerprint audit must pass")
    if not bool(evidence.get("temporal_decode_uses_original_time", False)):
        raise ValueError("temporal decode must use original time")
    if require_detector_forward:
        if int(evidence.get("detector_forward_temporal_len", -1)) != int(valid_k):
            raise ValueError("detector forward temporal length must equal valid_k")
        if evidence.get("status") != "detector_forward_sparse_audited":
            raise ValueError("detector forward sparse audit status missing")
    else:
        if evidence.get("status") != "local_gather_smoke_only":
            raise ValueError("local smoke without detector forward must stay local_gather_smoke_only")
        if bool(evidence.get("sparse_compute_claim", False)):
            raise ValueError("local gather smoke must not claim sparse compute")
    return True


def _fail_sparse_forward(code, message):
    raise ValueError(f"{code}: {message}")


def _require_sparse_forward_fields(ledger, required):
    missing = sorted(key for key in required if key not in ledger)
    if missing:
        _fail_sparse_forward("FAIL_SPARSE_FORWARD_LEDGER_MISSING_FIELD", f"missing fields: {missing}")


def _int_field(ledger, key):
    try:
        return int(ledger[key])
    except (TypeError, ValueError, KeyError) as exc:
        _fail_sparse_forward("FAIL_SPARSE_FORWARD_LEDGER_BAD_FIELD", f"{key} must be an integer")
        raise exc


def validate_sparse_forward_ledger(ledger, allow_sparse_compute_claim=False):
    required = {
        "route_label",
        "claim_status",
        "audit_mode",
        "selector_method",
        "selection_unit",
        "dense_T",
        "selected_positions",
        "valid_k",
        "raw_frame_inds_in",
        "decoded_frame_count",
        "decoded_unique_count",
        "padded_duplicate_count",
        "dense_raw_backbone_handoff",
        "selected_inputs_is_gathered",
        "backbone_input_shape_before_preprocess",
        "backbone_input_shape_after_preprocess",
        "dense_backbone_chunk_count",
        "backbone_forward_chunk_count",
        "time_embed_valid_count",
        "time_embed_total_count",
        "post_backbone_feature_len",
        "detector_prepad_feature_len",
        "detector_pad_len",
        "detector_mask_true_count",
        "rpn_valid_temporal_len",
        "temporal_decode_uses_original_time",
        "original_time_metadata",
        "sparse_compute_claim",
        "forbidden_deploy_fields_absent",
        "module_forward_evidence",
    }
    _require_sparse_forward_fields(ledger, required)

    try:
        validate_route_identity(ledger)
    except ValueError as exc:
        _fail_sparse_forward(SPARSE_FORWARD_FAIL_ROUTE_MIXING, str(exc))
    try:
        validate_no_leakage(ledger)
    except ValueError as exc:
        _fail_sparse_forward(SPARSE_FORWARD_FAIL_LEAKAGE_FIELD_PRESENT, str(exc))

    dense_T = _int_field(ledger, "dense_T")
    valid_k = _int_field(ledger, "valid_k")
    if not (0 < valid_k < dense_T):
        _fail_sparse_forward("FAIL_SPARSE_FORWARD_VALID_K", "valid_k must satisfy 0 < valid_k < dense_T")
    if ledger.get("selection_unit") not in {"frame", "tubelet", "feature"}:
        _fail_sparse_forward("FAIL_SPARSE_FORWARD_SELECTION_UNIT", "selection_unit must be frame, tubelet, or feature")
    try:
        validate_selected_positions(ledger["selected_positions"], dense_T, valid_k=valid_k)
    except ValueError as exc:
        _fail_sparse_forward("FAIL_SPARSE_FORWARD_SELECTED_POSITIONS", str(exc))

    sparse_compute_claim = bool(ledger.get("sparse_compute_claim", False))
    if sparse_compute_claim and not allow_sparse_compute_claim:
        _fail_sparse_forward(
            SPARSE_FORWARD_FAIL_SPARSE_CLAIM_UNLOCKED,
            "local sparse-forward precheck cannot unlock sparse_compute_claim",
        )
    if ledger.get("claim_status") == SPARSE_FORWARD_SHAPE_ONLY_CLAIM_STATUS and sparse_compute_claim:
        _fail_sparse_forward(
            SPARSE_FORWARD_FAIL_SPARSE_CLAIM_UNLOCKED,
            "shape-only claim_status cannot claim sparse compute",
        )
    if ledger.get("claim_status") not in {
        SPARSE_FORWARD_SHAPE_ONLY_CLAIM_STATUS,
        SPARSE_FORWARD_REAL_MODULE_CLAIM_STATUS,
    }:
        _fail_sparse_forward("FAIL_SPARSE_FORWARD_CLAIM_STATUS", f"unsupported claim_status: {ledger.get('claim_status')}")

    raw_frame_inds = [int(pos) for pos in ledger.get("raw_frame_inds_in", [])]
    decoded_frame_count = _int_field(ledger, "decoded_frame_count")
    decoded_unique_count = _int_field(ledger, "decoded_unique_count")
    padded_duplicate_count = _int_field(ledger, "padded_duplicate_count")
    dense_positions = list(range(dense_T))
    if bool(ledger.get("dense_raw_backbone_handoff", True)):
        _fail_sparse_forward(SPARSE_FORWARD_FAIL_DENSE_RAW_HANDOFF, "dense_raw_backbone_handoff must be false")
    if not bool(ledger.get("selected_inputs_is_gathered", False)):
        _fail_sparse_forward(SPARSE_FORWARD_FAIL_DENSE_RAW_HANDOFF, "selected_inputs_is_gathered must be true")
    if raw_frame_inds == dense_positions or decoded_unique_count >= dense_T:
        _fail_sparse_forward(SPARSE_FORWARD_FAIL_DENSE_RAW_HANDOFF, "raw decode saw the dense window")
    if decoded_frame_count != len(raw_frame_inds):
        _fail_sparse_forward("FAIL_SPARSE_FORWARD_RAW_DECODE_COUNT", "decoded_frame_count must equal raw_frame_inds_in length")
    if decoded_unique_count != len(set(raw_frame_inds)):
        _fail_sparse_forward("FAIL_SPARSE_FORWARD_RAW_DECODE_COUNT", "decoded_unique_count must equal unique raw indices")
    if decoded_frame_count - decoded_unique_count != padded_duplicate_count:
        _fail_sparse_forward("FAIL_SPARSE_FORWARD_RAW_DECODE_COUNT", "padded_duplicate_count must match duplicate raw indices")
    if any(pos < 0 or pos >= dense_T for pos in raw_frame_inds):
        _fail_sparse_forward("FAIL_SPARSE_FORWARD_RAW_DECODE_COUNT", "raw_frame_inds_in contains out-of-range positions")

    dense_backbone_chunk_count = _int_field(ledger, "dense_backbone_chunk_count")
    backbone_forward_chunk_count = _int_field(ledger, "backbone_forward_chunk_count")
    if dense_backbone_chunk_count <= 0:
        _fail_sparse_forward("FAIL_SPARSE_FORWARD_BACKBONE_CHUNK_COUNT", "dense_backbone_chunk_count must be positive")
    if backbone_forward_chunk_count == dense_backbone_chunk_count and valid_k < dense_backbone_chunk_count:
        _fail_sparse_forward(
            SPARSE_FORWARD_FAIL_BACKBONE_DENSE_CHUNK_COUNT,
            "backbone forward chunk count equals dense chunk count while valid_k is sparse",
        )
    if backbone_forward_chunk_count != valid_k:
        _fail_sparse_forward(
            SPARSE_FORWARD_FAIL_BACKBONE_DENSE_CHUNK_COUNT,
            "backbone forward chunk count must equal sparse valid_k for this precheck",
        )
    if _int_field(ledger, "time_embed_valid_count") != valid_k:
        _fail_sparse_forward("FAIL_SPARSE_FORWARD_TIME_EMBED_COUNT", "time_embed_valid_count must equal valid_k")
    if _int_field(ledger, "time_embed_total_count") < valid_k:
        _fail_sparse_forward("FAIL_SPARSE_FORWARD_TIME_EMBED_COUNT", "time_embed_total_count cannot be smaller than valid_k")
    if _int_field(ledger, "post_backbone_feature_len") != valid_k:
        _fail_sparse_forward(
            SPARSE_FORWARD_FAIL_BACKBONE_DENSE_CHUNK_COUNT,
            "post_backbone_feature_len must equal sparse valid_k",
        )
    if bool(ledger.get("post_backbone_interpolated_to_dense", False)):
        _fail_sparse_forward(
            SPARSE_FORWARD_FAIL_BACKBONE_DENSE_CHUNK_COUNT,
            "post-backbone interpolation to dense is not valid sparse-forward evidence",
        )

    detector_prepad_feature_len = _int_field(ledger, "detector_prepad_feature_len")
    detector_pad_len = _int_field(ledger, "detector_pad_len")
    detector_mask_true_count = _int_field(ledger, "detector_mask_true_count")
    rpn_valid_temporal_len = _int_field(ledger, "rpn_valid_temporal_len")
    if detector_prepad_feature_len != valid_k:
        _fail_sparse_forward(
            SPARSE_FORWARD_FAIL_DETECTOR_PAD_CONFUSED_AS_VALID,
            "detector_prepad_feature_len must equal valid_k",
        )
    if detector_pad_len < detector_prepad_feature_len:
        _fail_sparse_forward(
            SPARSE_FORWARD_FAIL_DETECTOR_PAD_CONFUSED_AS_VALID,
            "detector_pad_len cannot be smaller than detector_prepad_feature_len",
        )
    if detector_mask_true_count != valid_k or rpn_valid_temporal_len != valid_k:
        _fail_sparse_forward(
            SPARSE_FORWARD_FAIL_DETECTOR_PAD_CONFUSED_AS_VALID,
            "detector/RPN valid temporal counts must equal valid_k",
        )
    if detector_pad_len > detector_prepad_feature_len and detector_mask_true_count == detector_pad_len:
        _fail_sparse_forward(
            SPARSE_FORWARD_FAIL_DETECTOR_PAD_CONFUSED_AS_VALID,
            "detector pad length was counted as valid mask length",
        )

    if not bool(ledger.get("temporal_decode_uses_original_time", False)):
        _fail_sparse_forward(SPARSE_FORWARD_FAIL_ORIGINAL_TIME_DECODE_MISSING, "temporal decode must use original time")
    try:
        validate_original_time_metadata(ledger["original_time_metadata"])
    except ValueError as exc:
        _fail_sparse_forward(SPARSE_FORWARD_FAIL_ORIGINAL_TIME_DECODE_MISSING, str(exc))

    if bool(ledger.get("forbidden_deploy_fields_absent")) is not True:
        _fail_sparse_forward(SPARSE_FORWARD_FAIL_LEAKAGE_FIELD_PRESENT, "forbidden_deploy_fields_absent must be true")

    if bool(ledger.get("module_forward_evidence", False)):
        verdict = SPARSE_FORWARD_PASS_REAL_MODULE
    else:
        verdict = SPARSE_FORWARD_PASS_SHAPE_ONLY
    return {
        "verdict": verdict,
        "claim_status": ledger["claim_status"],
        "sparse_compute_claim": sparse_compute_claim,
        "valid_k": valid_k,
        "dense_T": dense_T,
    }


def validate_bvr_twb_pipeline_ledger(ledger):
    required = {
        "route_label",
        "method",
        "split",
        "dense_T",
        "selected_positions",
        "selected_frame_inds",
        "raw_selected_positions",
        "valid_k",
        "raw_frame_handoff_stage",
        "selected_raw_frames_before_decode",
        "decode_input_frame_inds",
        "fixed_padded_bridge_sparse_compute_claim",
        "budget_stop_reason",
        "selection_gap_diagnostics",
        "original_time_metadata",
        "temporal_decode_uses_original_time",
        "selected_index_is_time",
        "dense_raw_backbone_handoff",
        "selected_inputs_is_gathered",
        "padding_duplicate_count",
        "sparse_compute_claim",
        "claim_status",
        "selector_provenance",
        "adapter_bridge_mode",
        "detector_mask_len",
        "detector_mask_true_count",
        "detector_feature_valid_k",
        "detector_feature_positions",
        "preview_source",
        "scout_source",
        "scout_is_deploy_visible",
        "deterministic_preview_fallback_used",
        "diagnostic_preview_fallback_allowed",
        "value_mode",
        "value_model_used",
        "value_labels_used_at_test",
    }
    missing = sorted(required.difference(ledger.keys()))
    if missing:
        raise ValueError(f"BVR-TWB pipeline ledger missing fields: {missing}")
    validate_route_identity(ledger)
    validate_no_leakage(ledger.get("selector_provenance", {}))
    dense_T = int(ledger["dense_T"])
    valid_k = int(ledger["valid_k"])
    validate_selected_positions(ledger["selected_positions"], dense_T, valid_k=valid_k)
    if len(ledger["selected_frame_inds"]) != valid_k:
        raise ValueError("BVR-TWB selected_frame_inds length must equal valid_k")
    if ledger.get("raw_frame_handoff_stage") != "pre_decode_selected_raw_frames":
        raise ValueError("BVR-TWB raw frame handoff must be explicitly before decode/backbone")
    if bool(ledger.get("selected_raw_frames_before_decode")) is not True:
        raise ValueError("BVR-TWB ledger must prove selected raw frames are handed off before decode")
    if bool(ledger.get("fixed_padded_bridge_sparse_compute_claim", True)):
        raise ValueError("BVR-TWB fixed padded bridge cannot be used as a sparse-compute claim")
    decode_input_frame_inds = [int(pos) for pos in ledger.get("decode_input_frame_inds", [])]
    selected_frame_inds = [int(pos) for pos in ledger.get("selected_frame_inds", [])]
    if len(decode_input_frame_inds) < valid_k:
        raise ValueError("BVR-TWB decode input frame list cannot be shorter than valid_k")
    if decode_input_frame_inds[:valid_k] != selected_frame_inds:
        raise ValueError("BVR-TWB decode inputs must begin with the selected raw frame indices")
    if valid_k >= dense_T:
        raise ValueError("BVR-TWB dynamic pipeline must keep valid_k < dense_T")
    if bool(ledger["dense_raw_backbone_handoff"]):
        raise ValueError("BVR-TWB pipeline ledger has dense raw handoff")
    if not bool(ledger["selected_inputs_is_gathered"]):
        raise ValueError("BVR-TWB pipeline ledger must mark selected inputs gathered")
    adapter_mode = ledger.get("adapter_bridge_mode")
    if adapter_mode == ADAPTER_FIXED_LENGTH_PADDED_BRIDGE:
        _validate_adapter_fixed_length_bridge_ledger(ledger, valid_k)
    elif int(ledger["padding_duplicate_count"]) != 0:
        raise ValueError("BVR-TWB pipeline precheck forbids padding duplicates without adapter bridge metadata")
    if bool(ledger["sparse_compute_claim"]):
        raise ValueError("BVR-TWB local pipeline ledger cannot claim sparse compute")
    if bool(ledger.get("deterministic_preview_fallback_used", False)):
        raise ValueError("BVR-TWB formal pipeline ledger used diagnostic deterministic preview fallback")
    if bool(ledger.get("diagnostic_preview_fallback_allowed", False)):
        raise ValueError("BVR-TWB formal pipeline ledger must not allow diagnostic preview fallback")
    if bool(ledger.get("scout_is_deploy_visible", False)) is not True:
        raise ValueError("BVR-TWB formal pipeline requires deploy-visible scout evidence")
    if ledger.get("preview_source") not in {"deploy_visible_metadata_actionness", "raw_rgb_lowres_scout"}:
        raise ValueError(f"BVR-TWB unsupported formal preview_source: {ledger.get('preview_source')}")
    if ledger.get("scout_source") not in {
        "deploy_visible_raw_or_metadata_scout",
        "deploy_visible_metadata_scout",
        "raw_rgb_lowres_scout",
    }:
        raise ValueError(f"BVR-TWB unsupported formal scout_source: {ledger.get('scout_source')}")
    if ledger.get("value_mode") != "deploy_heuristic_voi":
        raise ValueError("BVR-TWB formal local/precheck path requires deploy_heuristic_voi value_mode")
    if bool(ledger.get("value_model_used", False)):
        raise ValueError("BVR-TWB formal local/precheck path must not silently use a learned value model")
    if bool(ledger.get("value_labels_used_at_test", True)):
        raise ValueError("BVR-TWB train-only value labels must not be used at test/deploy selection time")
    if not bool(ledger["temporal_decode_uses_original_time"]) or bool(ledger["selected_index_is_time"]):
        raise ValueError("BVR-TWB pipeline must preserve original-time decode")
    validate_original_time_metadata(ledger["original_time_metadata"])
    validate_selection_gap_diagnostics(ledger)
    provenance = ledger["selector_provenance"]
    if any(bool(provenance.get(key, False)) for key in provenance):
        raise ValueError("BVR-TWB selector provenance must not use GT/teacher/cache/oracle shortcuts")
    if "bracket_summary" in ledger:
        validate_belief_update_trace_schema(ledger["bracket_summary"].get("belief_update_trace", []))
    validate_detector_feature_positions(ledger)
    return True


def validate_detector_feature_positions(ledger):
    positions = np.asarray(ledger.get("detector_feature_positions", []), dtype=np.float32).reshape(-1)
    detector_feature_valid_k = int(ledger.get("detector_feature_valid_k", ledger.get("detector_mask_true_count", 0)))
    detector_mask_true_count = int(ledger.get("detector_mask_true_count", detector_feature_valid_k))
    if positions.shape[0] != detector_feature_valid_k:
        raise ValueError("detector_feature_positions length must equal detector_feature_valid_k")
    if detector_feature_valid_k != detector_mask_true_count:
        raise ValueError("detector_feature_valid_k must equal detector_mask_true_count")
    feature_stride = int(max(int(ledger.get("bvr_twb_feature_stride", 1)), 1))
    raw_positions = ledger.get("raw_selected_positions", ledger.get("selected_positions", []))
    expected = build_detector_feature_centers_from_raw(raw_positions, feature_stride=feature_stride)
    if expected.shape[0] != positions.shape[0]:
        raise ValueError("detector feature center count does not match raw selected positions and feature_stride")
    if not np.allclose(positions, expected, atol=1e-5):
        raise ValueError("detector_feature_positions must be grouped centers from raw selected positions")
    if ledger.get("adapter_bridge_mode") == ADAPTER_FIXED_LENGTH_PADDED_BRIDGE and feature_stride > 1:
        if positions.shape[0] >= int(ledger.get("valid_k", 0)) and int(ledger.get("valid_k", 0)) > 1:
            raise ValueError("adapter bridge detector positions must be feature/tubelet centers, not raw positions")
    return True


def _validate_adapter_fixed_length_bridge_ledger(ledger, valid_k):
    required = {
        "adapter_target_frame_num",
        "adapter_input_frame_count",
        "adapter_padded_frame_inds",
        "adapter_padded_positions",
        "adapter_valid_raw_mask",
        "adapter_padding_duplicate_count",
        "adapter_padding_counts_as_valid",
        "adapter_fixed_length_padded_bridge",
        "adapter_padding_role",
        "adapter_padding_invalid_for_detector",
    }
    missing = sorted(required.difference(ledger.keys()))
    if missing:
        raise ValueError(f"adapter_fixed_length_padded_bridge ledger missing fields: {missing}")
    target = int(ledger["adapter_target_frame_num"])
    input_count = int(ledger["adapter_input_frame_count"])
    if target <= 0 or input_count != target:
        raise ValueError("adapter fixed-length bridge requires adapter_input_frame_count == adapter_target_frame_num > 0")
    if len(ledger["adapter_padded_frame_inds"]) != input_count:
        raise ValueError("adapter_padded_frame_inds length must equal adapter_input_frame_count")
    if len(ledger["adapter_padded_positions"]) != input_count:
        raise ValueError("adapter_padded_positions length must equal adapter_input_frame_count")
    raw_mask = [bool(value) for value in ledger["adapter_valid_raw_mask"]]
    if len(raw_mask) != input_count:
        raise ValueError("adapter_valid_raw_mask length must equal adapter_input_frame_count")
    if sum(raw_mask) != int(valid_k):
        raise ValueError("adapter_valid_raw_mask true count must equal sparse valid_k")
    pad_count = int(ledger["adapter_padding_duplicate_count"])
    if pad_count != input_count - int(valid_k):
        raise ValueError("adapter_padding_duplicate_count must equal adapter_input_frame_count - valid_k")
    if int(ledger["padding_duplicate_count"]) != pad_count:
        raise ValueError("padding_duplicate_count must mirror adapter padding duplicate count")
    if bool(ledger["adapter_padding_counts_as_valid"]):
        raise ValueError("adapter padding duplicates must not count as valid")
    if ledger.get("adapter_padding_role") != "fixed_length_decode_backbone_compatibility_invalid_observation":
        raise ValueError("adapter padding role must state compatibility-only invalid observation")
    if bool(ledger.get("adapter_padding_invalid_for_detector")) is not True:
        raise ValueError("adapter padding must be invalid for detector/head evidence")
    if input_count > int(valid_k):
        selected = [int(pos) for pos in ledger.get("selected_frame_inds", [])]
        padded = [int(pos) for pos in ledger.get("adapter_padded_frame_inds", [])]
        if padded[: int(valid_k)] != selected:
            raise ValueError("adapter padded frame inputs must preserve selected raw frames as prefix")
        if len(set(padded[int(valid_k) :])) > 1 or (padded[int(valid_k) :] and padded[-1] != selected[-1]):
            raise ValueError("adapter padding must be hold-last duplicate compatibility input")
    if bool(ledger.get("adapter_fixed_length_padded_bridge")) is not True:
        raise ValueError("adapter_fixed_length_padded_bridge must be true for adapter bridge mode")
    detector_mask_len = int(ledger["detector_mask_len"])
    detector_mask_true_count = int(ledger["detector_mask_true_count"])
    detector_feature_valid_k = int(ledger.get("detector_feature_valid_k", detector_mask_true_count))
    if detector_mask_len <= 0 or detector_mask_true_count != detector_feature_valid_k:
        raise ValueError("detector mask true count must equal detector_feature_valid_k")
    if len(ledger.get("detector_feature_positions", [])) != detector_feature_valid_k:
        raise ValueError("detector_feature_positions length must equal detector_feature_valid_k")
    if not (0 < detector_mask_true_count <= detector_mask_len):
        raise ValueError("detector mask true count must be within detector mask length")
    if bool(ledger.get("sparse_compute_claim", False)):
        raise ValueError("adapter fixed-length padded bridge cannot claim sparse compute")
    return True


def validate_summary_claim_status(summary, claim_mode="local_gather_smoke"):
    status = summary.get("claim_status")
    if claim_mode == "local_gather_smoke":
        if status != LOCAL_GATHER_CLAIM_STATUS:
            raise ValueError(f"claim_status must stay locked as {LOCAL_GATHER_CLAIM_STATUS}, got {status}")
    elif claim_mode == "sparse_forward_audit":
        if status == LOCAL_GATHER_CLAIM_STATUS:
            raise ValueError("sparse_forward_audit summary cannot reuse local gather claim_status")
    else:
        raise ValueError(f"unsupported claim_mode: {claim_mode}")
    return True


def validate_value_components(components):
    if not isinstance(components, dict):
        raise ValueError("value_components must be a dict")
    missing = sorted(VALUE_COMPONENT_KEYS.difference(components.keys()))
    if missing:
        raise ValueError(f"value_components missing fields: {missing}")
    for key in VALUE_COMPONENT_KEYS:
        value = float(components[key])
        if not np.isfinite(value):
            raise ValueError(f"value_components contains non-finite field: {key}")
    return True


def validate_voi_component_balance(components, max_actionness_fraction=0.45):
    validate_value_components(components)
    total_abs = max(sum(abs(float(value)) for value in components.values()), 1e-9)
    actionness_fraction = abs(float(components.get("actionness_component", 0.0))) / total_abs
    if actionness_fraction > float(max_actionness_fraction):
        raise ValueError(
            "VOI-BBC value components are actionness-only dominated: "
            f"actionness_fraction={actionness_fraction:.3f}"
        )
    non_action_mass = sum(
        abs(float(components.get(key, 0.0)))
        for key in (
            "expected_entropy_reduction",
            "expected_width_reduction",
            "expected_gap_risk_reduction",
            "short_action_value",
            "two_sided_witness_value",
            "predicted_regret",
        )
    )
    if non_action_mass <= 0.0:
        raise ValueError("VOI-BBC value components require non-actionness contribution")
    return {"actionness_fraction": float(actionness_fraction), "non_action_mass": float(non_action_mass)}


def validate_candidate_packet_ledger(row):
    required = {
        "route_label",
        "method",
        "packet_id",
        "video_id",
        "window_id",
        "split",
        "packet_source",
        "packet_role",
        "packet_positions",
        "packet_cost_frames",
        "rank",
        "reason",
        "value_components",
    }
    missing = sorted(required.difference(row.keys()))
    if missing:
        raise ValueError(f"candidate packet ledger missing fields: {missing}")
    validate_route_identity(row)
    validate_selected_positions(row["packet_positions"], row.get("dense_T", max(row["packet_positions"]) + 1))
    validate_value_components(row["value_components"])
    validate_voi_component_balance(row["value_components"])
    if float(row["packet_cost_frames"]) <= 0:
        raise ValueError("candidate packet cost must be positive")
    return True


def validate_selection_row_schema(row):
    required = {
        "selected_decision",
        "selected_decision_subreason",
        "constraint_state",
        "value_components",
    }
    missing = sorted(required.difference(row.keys()))
    if missing:
        raise ValueError(f"selection row missing fields: {missing}")
    validate_value_components(row["value_components"])
    state = row["constraint_state"]
    if not isinstance(state, dict):
        raise ValueError("constraint_state must be a dict")
    for key in ("max_gap_ok", "budget_ok", "duplicate"):
        if key not in state:
            raise ValueError(f"constraint_state missing field: {key}")
    if "belief_risk_before" in row and "belief_risk_after" in row:
        if float(row["belief_risk_after"]) > float(row["belief_risk_before"]) + 1e-6:
            raise ValueError("selection row belief risk increased after selected witness")
    validate_voi_component_balance(row["value_components"])
    return True


def build_selection_gap_diagnostics(selected_positions, dense_T, max_allowed_gap):
    stats = gap_statistics(selected_positions, dense_T)
    max_allowed_gap = int(max_allowed_gap)
    max_gap = int(stats["max_gap"])
    return {
        "max_gap": max_gap,
        "mean_gap": float(stats["mean_gap"]),
        "gaps": [int(gap) for gap in stats["gaps"]],
        "max_allowed_gap": max_allowed_gap,
        "coverage_violation": bool(max_gap > max_allowed_gap),
    }


def validate_selection_gap_diagnostics(ledger):
    if "selection_gap_diagnostics" not in ledger:
        raise ValueError("deploy ledger missing selection_gap_diagnostics")
    diagnostics = ledger["selection_gap_diagnostics"]
    required = {"max_gap", "max_allowed_gap", "coverage_violation"}
    missing = sorted(required.difference(diagnostics.keys()))
    if missing:
        raise ValueError(f"selection_gap_diagnostics missing fields: {missing}")
    recomputed = build_selection_gap_diagnostics(
        ledger["selected_positions"],
        ledger["dense_T"],
        diagnostics["max_allowed_gap"],
    )
    if int(diagnostics["max_gap"]) != int(recomputed["max_gap"]):
        raise ValueError("selection_gap_diagnostics max_gap does not match selected_positions")
    if int(diagnostics["max_allowed_gap"]) < 1:
        raise ValueError("selection_gap_diagnostics max_allowed_gap must be positive")
    if bool(diagnostics.get("coverage_violation", False)) or int(diagnostics["max_gap"]) > int(diagnostics["max_allowed_gap"]):
        raise ValueError(
            "hard max-gap contract violated: "
            f"max_gap={diagnostics['max_gap']} max_allowed_gap={diagnostics['max_allowed_gap']}"
        )
    return True


def validate_deploy_ledger(ledger):
    validate_route_identity(ledger)
    validate_no_leakage(ledger)
    dense_T = int(ledger["dense_T"])
    valid_k = int(ledger["valid_k"])
    claim_mode = ledger.get("claim_mode")
    if claim_mode not in {"local_gather_smoke", "sparse_forward_audit"}:
        raise ValueError(f"deploy ledger requires explicit claim_mode, got {claim_mode}")
    validate_selected_positions(ledger["selected_positions"], dense_T, valid_k=valid_k)
    if ledger.get("budget_stop_reason") not in STOP_REASONS:
        raise ValueError(f"invalid budget_stop_reason: {ledger.get('budget_stop_reason')}")
    if ledger.get("budget_stop_reason") == "belief_width_safe":
        validate_belief_width_safe_stop_contract(ledger)
    validate_voi_bbc_stop_contract(ledger)
    validate_selection_gap_diagnostics(ledger)
    validate_original_time_metadata(ledger.get("original_time_metadata", ledger))
    validate_sparse_gather_evidence(
        ledger["real_sparse_evidence"],
        dense_T=dense_T,
        valid_k=valid_k,
        require_detector_forward=claim_mode == "sparse_forward_audit",
    )
    if claim_mode == "local_gather_smoke" and bool(ledger["real_sparse_evidence"].get("sparse_compute_claim", False)):
        raise ValueError("local_gather_smoke claim_mode cannot claim sparse compute")
    absent = ledger.get("forbidden_fields_absent", {})
    if not all(bool(value) for value in absent.values()):
        raise ValueError("forbidden_fields_absent flags must all be true")
    return True


def validate_belief_width_safe_stop_contract(ledger):
    summary = ledger.get("bracket_summary", {})
    active_trace = summary.get("active_belief_update_trace")
    if not active_trace:
        raise ValueError("belief_width_safe requires non-empty active_belief_update_trace")
    if summary.get("all_active_beliefs_updated_and_safe") is not True:
        raise ValueError("belief_width_safe requires all_active_beliefs_updated_and_safe is True")
    for idx, row in enumerate(active_trace):
        if row.get("updated_from_selected_witness") is not True:
            raise ValueError(f"belief_width_safe active trace row {idx} is not updated from selected witness")
        if row.get("belief_width_safe") is not True:
            raise ValueError(f"belief_width_safe active trace row {idx} is not individually safe")
    return True


def validate_belief_update_trace_schema(trace):
    if not isinstance(trace, list):
        raise ValueError("belief_update_trace must be a list")
    required = {
        "bracket_id",
        "initial_entropy",
        "posterior_entropy",
        "initial_width_p80_frames",
        "posterior_width_p80_frames",
        "posterior_credible_width_frames",
        "initial_risk_mass",
        "posterior_risk_mass",
        "two_sided_witness_coverage",
        "updated_by_selected_witness",
        "belief_width_safe",
    }
    for idx, row in enumerate(trace):
        missing = sorted(required.difference(row.keys()))
        if missing:
            raise ValueError(f"belief_update_trace row {idx} missing fields: {missing}")
        if bool(row.get("belief_width_safe")) and not bool(row.get("updated_by_selected_witness")):
            raise ValueError("belief_width_safe cannot be forged without selected witness update")
        if float(row["posterior_entropy"]) > float(row["initial_entropy"]) + 1e-6:
            raise ValueError("posterior entropy cannot exceed initial entropy")
        if float(row["posterior_width_p80_frames"]) > float(row["initial_width_p80_frames"]) + 1e-6:
            raise ValueError("posterior credible width cannot exceed initial width")
        if float(row["posterior_risk_mass"]) > float(row["initial_risk_mass"]) + 1e-6:
            raise ValueError("posterior risk mass cannot exceed initial risk mass")
    return True


def validate_voi_bbc_stop_contract(ledger):
    summary = ledger.get("bracket_summary", {})
    trace = summary.get("belief_update_trace", [])
    validate_belief_update_trace_schema(trace)
    stop = ledger.get("budget_stop_reason")
    diagnostics = ledger.get("selection_gap_diagnostics", {})
    if stop in {"belief_width_safe", "risk_constraints_satisfied"}:
        if bool(diagnostics.get("coverage_violation", False)):
            raise ValueError(f"{stop} cannot have a max-gap coverage violation")
        if stop == "risk_constraints_satisfied":
            if float(summary.get("max_posterior_risk_mass", 1.0)) > 0.45:
                raise ValueError("risk_constraints_satisfied requires low posterior risk mass")
    if stop == "regret_saturation":
        trace_summary = ledger.get("controller_trace_summary", {})
        if not bool(trace_summary.get("regret_saturation_seen", False)):
            raise ValueError("regret_saturation stop requires controller trace evidence")
    if stop == "gap_guard" and not bool(diagnostics.get("coverage_violation", False)):
        # gap_guard can also appear when repair consumed the remaining budget.
        if ledger.get("valid_k", 0) < ledger.get("max_k", 0):
            raise ValueError("gap_guard stop without violation requires budget exhaustion evidence")
    return True


def exact_uniform_positions(dense_T, k):
    dense_T = int(dense_T)
    k = int(k)
    if k <= 0:
        return []
    if k >= dense_T:
        return list(range(dense_T))
    return sorted({int(round(x)) for x in np.linspace(0, dense_T - 1, k)})


def overlap_ratio(left, right):
    left_set = set(int(x) for x in left)
    right_set = set(int(x) for x in right)
    if not left_set and not right_set:
        return 1.0
    return len(left_set.intersection(right_set)) / float(max(len(left_set), len(right_set), 1))


def validate_dynamicity_and_uniform_mimicry(ledgers, dynamic_enabled=True, max_uniform_overlap=0.86):
    ledgers = list(ledgers)
    if not ledgers:
        raise ValueError("dynamicity check requires at least one ledger")
    ks = [int(row["valid_k"]) for row in ledgers]
    stops = Counter(row.get("budget_stop_reason") for row in ledgers)
    scaffold_ratios = [float(row.get("scaffold_k", 0)) / max(int(row["valid_k"]), 1) for row in ledgers]
    overlaps = [
        overlap_ratio(row["selected_positions"], exact_uniform_positions(row["dense_T"], row["valid_k"]))
        for row in ledgers
    ]
    if max(scaffold_ratios) > 0.75:
        raise ValueError("scaffold is too thick for BVR-TWB local implementation")
    if dynamic_enabled and len(set(ks)) <= 1 and len(ks) >= 3:
        raise ValueError("dynamic controller collapsed to constant-K across synthetic cases")
    if dynamic_enabled and stops.get("budget_cap", 0) == len(ledgers):
        raise ValueError("dynamic controller is budget-cap-only")
    if max(overlaps) >= float(max_uniform_overlap):
        raise ValueError("BVR-TWB selection has a per-case exact-uniform overlap violation")
    if float(np.mean(overlaps)) >= float(max_uniform_overlap):
        raise ValueError("BVR-TWB selection overlaps exact-uniform too strongly")
    return {
        "k_values": ks,
        "stop_reason_counts": dict(stops),
        "mean_uniform_overlap": float(np.mean(overlaps)),
        "per_case_uniform_overlap": [float(value) for value in overlaps],
        "max_scaffold_ratio": float(max(scaffold_ratios)),
    }
