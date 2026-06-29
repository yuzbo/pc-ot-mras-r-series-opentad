from collections import Counter

import numpy as np

from .scaffold import gap_statistics
from .types import FORBIDDEN_DEPLOY_KEYS, FORBIDDEN_ROUTE_TOKENS, ROUTE_LABEL, STOP_REASONS, sorted_unique_positions

LOCAL_GATHER_CLAIM_STATUS = "local_gather_smoke_only_no_sparse_compute_or_metric_claim"

VALUE_COMPONENT_KEYS = {
    "belief_width_gain",
    "role_gain",
    "gap_gain",
    "short_action_gain",
    "redundancy_repulsion_penalty",
    "low_actionness_component",
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
