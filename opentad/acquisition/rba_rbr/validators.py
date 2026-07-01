from collections import Counter

import numpy as np

from .types import FORBIDDEN_DEPLOY_KEYS, FORBIDDEN_ROUTE_TOKENS, ROUTE_LABEL, STOP_REASONS, sorted_unique_positions


LOCAL_PRECHECK_CLAIM_STATUS = "rba_rbr_local_precheck_only_no_metric_runtime_deploy_or_paper_claim"


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
    if isinstance(metadata, dict):
        route_label = metadata.get("route_label")
        if route_label is not None and route_label != ROUTE_LABEL:
            raise ValueError(f"route_label must be {ROUTE_LABEL}, got {route_label}")
    text_without_label = text.replace(ROUTE_LABEL, "")
    for token in FORBIDDEN_ROUTE_TOKENS:
        if token in allow_tokens:
            continue
        if token.lower() in text_without_label.lower():
            raise ValueError(f"RBA-RBR route metadata contains forbidden route token: {token}")
    return True


def validate_no_leakage(metadata):
    for path, key, value in _walk_dict(metadata):
        key_l = str(key).lower()
        for forbidden in FORBIDDEN_DEPLOY_KEYS:
            if forbidden.lower() == key_l:
                if value is None or value is False:
                    continue
                if isinstance(value, (list, tuple, dict, set)) and len(value) == 0:
                    continue
                raise ValueError(f"forbidden leakage field at {path}")
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
        window_end_sec = float(window_start_sec) + float(dense_T) / fps
    return {
        "dense_T": dense_T,
        "fps": fps,
        "window_start_sec": float(window_start_sec),
        "window_end_sec": float(window_end_sec),
        "selected_positions": positions,
        "selected_positions_unit": "original_dense_index",
        "selected_times_sec": [float(window_start_sec) + float(pos) / fps for pos in positions],
        "time_axis_mode": "irregular_original_time",
        "selected_index_is_time": False,
    }


def validate_original_time_metadata(metadata):
    required = {"dense_T", "selected_positions", "selected_positions_unit", "time_axis_mode", "selected_index_is_time"}
    missing = sorted(required.difference(metadata.keys()))
    if missing:
        raise ValueError(f"original-time metadata missing fields: {missing}")
    if metadata["selected_positions_unit"] != "original_dense_index":
        raise ValueError("selected_positions_unit must be original_dense_index")
    if bool(metadata["selected_index_is_time"]):
        raise ValueError("selected_index_is_time must be false")
    if metadata["time_axis_mode"] != "irregular_original_time":
        raise ValueError("time_axis_mode must be irregular_original_time")
    validate_selected_positions(metadata["selected_positions"], metadata["dense_T"])
    return True


def build_selection_gap_diagnostics(selected_positions, dense_T):
    positions = sorted_unique_positions(selected_positions, dense_T)
    if not positions:
        return {"max_gap": int(dense_T), "mean_gap": float(dense_T), "gaps": [int(dense_T)]}
    gaps = [positions[0] + 1]
    gaps.extend(int(right - left) for left, right in zip(positions, positions[1:]))
    gaps.append(int(dense_T - positions[-1]))
    return {
        "max_gap": int(max(gaps)),
        "mean_gap": float(np.mean(gaps)),
        "gaps": [int(gap) for gap in gaps],
    }


def validate_rba_rbr_ledger(ledger):
    required = {
        "route_label",
        "method",
        "split",
        "dense_T",
        "selected_positions",
        "valid_k",
        "budget_stop_reason",
        "rescue_outside_hard_bracket_count",
        "selector_provenance",
        "claim_status",
        "no_metric_claim",
        "no_runtime_claim",
        "no_deploy_claim",
        "no_paper_claim",
        "full_train_unlocked",
        "original_time_metadata",
        "value_labels_used_at_test",
    }
    missing = sorted(required.difference(ledger.keys()))
    if missing:
        raise ValueError(f"RBA-RBR ledger missing fields: {missing}")
    validate_route_identity(ledger)
    validate_no_leakage(ledger.get("selector_provenance", {}))
    validate_selected_positions(ledger["selected_positions"], ledger["dense_T"], valid_k=ledger["valid_k"])
    if ledger["budget_stop_reason"] not in STOP_REASONS:
        raise ValueError(f"invalid budget_stop_reason: {ledger['budget_stop_reason']}")
    if bool(ledger["value_labels_used_at_test"]):
        raise ValueError("RBA-RBR train-only regret/value labels must not be used at val/test/deploy")
    if bool(ledger["full_train_unlocked"]):
        raise ValueError("RBA-RBR first local implementation must keep full_train_unlocked=false")
    if not bool(ledger["no_metric_claim"]) or not bool(ledger["no_runtime_claim"]):
        raise ValueError("RBA-RBR ledger must not claim metric or runtime evidence")
    if not bool(ledger["no_deploy_claim"]) or not bool(ledger["no_paper_claim"]):
        raise ValueError("RBA-RBR ledger must not claim deploy or paper readiness")
    if ledger["claim_status"] != LOCAL_PRECHECK_CLAIM_STATUS:
        raise ValueError(f"claim_status must stay locked as {LOCAL_PRECHECK_CLAIM_STATUS}")
    validate_original_time_metadata(ledger["original_time_metadata"])
    return True


def exact_uniform_positions(dense_T, k):
    dense_T = int(dense_T)
    k = int(k)
    if k <= 0:
        return []
    if k >= dense_T:
        return list(range(dense_T))
    picked = []
    used = set()
    for value in np.linspace(0, dense_T - 1, k):
        center = int(round(float(value)))
        if center not in used:
            pos = center
        else:
            pos = None
            for offset in range(1, dense_T):
                for candidate in (center - offset, center + offset):
                    if 0 <= candidate < dense_T and candidate not in used:
                        pos = candidate
                        break
                if pos is not None:
                    break
        if pos is None:
            raise ValueError(f"unable to build exact uniform positions for dense_T={dense_T}, k={k}")
        used.add(int(pos))
        picked.append(int(pos))
    return sorted(picked)


def validate_rba_rbr_control_bridge_metadata(ledger, bridge_meta):
    validate_rba_rbr_ledger(ledger)
    if ledger.get("selector_method") != "forced_uniform_control_diagnostic":
        raise ValueError("RBA-RBR control audit requires forced_uniform_control_diagnostic selector_method")
    if ledger.get("rba_rbr_control_mode") != "uniform_raw":
        raise ValueError("RBA-RBR control audit requires rba_rbr_control_mode=uniform_raw")
    if bool(ledger.get("full_train_unlocked", True)):
        raise ValueError("RBA-RBR control diagnostics must keep full_train_unlocked=false")
    for key in ("no_metric_claim", "no_runtime_claim", "no_deploy_claim", "no_paper_claim", "no_sparse_compute_claim"):
        if not bool(ledger.get(key, False)):
            raise ValueError(f"RBA-RBR control diagnostics require {key}=true")

    dense_T = int(ledger["dense_T"])
    selected = [int(pos) for pos in ledger["selected_positions"]]
    validate_selected_positions(selected, dense_T, valid_k=ledger["valid_k"])
    required_meta = {
        "irregular_native_axis",
        "rba_rbr_raw_selected_positions",
        "rba_rbr_raw_selected_valid_len",
        "rba_rbr_detector_feature_positions",
        "rba_rbr_detector_feature_valid_len",
        "detector_valid_mask",
        "adapter_valid_raw_mask",
    }
    missing = sorted(required_meta.difference(bridge_meta.keys()))
    if missing:
        raise ValueError(f"RBA-RBR control bridge metadata missing fields: {missing}")
    if bridge_meta["irregular_native_axis"] is not True:
        raise ValueError("RBA-RBR control bridge requires irregular_native_axis=True")

    raw_positions = np.asarray(bridge_meta["rba_rbr_raw_selected_positions"], dtype=np.float64).reshape(-1)
    detector_positions = np.asarray(bridge_meta["rba_rbr_detector_feature_positions"], dtype=np.float64).reshape(-1)
    detector_mask = np.asarray(bridge_meta["detector_valid_mask"], dtype=np.bool_).reshape(-1)
    raw_mask = np.asarray(bridge_meta["adapter_valid_raw_mask"], dtype=np.bool_).reshape(-1)
    raw_valid_k = int(ledger["valid_k"])
    detector_valid_k = int(ledger["detector_feature_valid_k"])
    if raw_positions.shape[0] != raw_valid_k:
        raise ValueError("RBA-RBR control raw position count does not match valid_k")
    if raw_positions.tolist() != [float(pos) for pos in selected]:
        raise ValueError("RBA-RBR control raw positions drifted from selected_positions")
    if detector_positions.shape[0] != detector_valid_k:
        raise ValueError("RBA-RBR control detector position count does not match detector_feature_valid_k")
    if int(detector_mask.sum()) != detector_valid_k:
        raise ValueError("RBA-RBR control detector mask true count does not match detector_feature_valid_k")
    if int(raw_mask.sum()) != raw_valid_k:
        raise ValueError("RBA-RBR control raw mask true count does not match valid_k")
    if np.any(np.diff(raw_positions) <= 0.0):
        raise ValueError("RBA-RBR control raw positions must be strictly increasing")
    if detector_positions.size > 1 and np.any(np.diff(detector_positions) <= 0.0):
        raise ValueError("RBA-RBR control detector positions must be strictly increasing")

    raw_density = float(raw_valid_k) / float(max(dense_T, 1))
    if abs(float(ledger["control_raw_density"]) - raw_density) > 1e-8:
        raise ValueError("RBA-RBR control raw density summary is incoherent")
    detector_target_len = int(ledger["detector_feature_target_len"])
    detector_density = float(detector_valid_k) / float(max(detector_target_len, 1))
    if abs(float(ledger["control_detector_density"]) - detector_density) > 1e-8:
        raise ValueError("RBA-RBR control detector density summary is incoherent")
    raw_gap = build_selection_gap_diagnostics(selected, dense_T)
    if int(ledger["control_raw_gap_summary"]["max_gap"]) != int(raw_gap["max_gap"]):
        raise ValueError("RBA-RBR control raw gap summary is incoherent")
    return True


def validate_dynamic_precheck_ledgers(ledgers):
    ledgers = list(ledgers)
    if not ledgers:
        raise ValueError("RBA-RBR dynamic precheck requires at least one ledger")
    for ledger in ledgers:
        validate_rba_rbr_ledger(ledger)
    ks = [int(row["valid_k"]) for row in ledgers]
    stops = Counter(row["budget_stop_reason"] for row in ledgers)
    if len(ledgers) >= 3 and len(set(ks)) <= 1:
        raise ValueError("RBA-RBR dynamic controller collapsed to constant-K")
    if stops.get("budget_cap", 0) == len(ledgers):
        raise ValueError("RBA-RBR dynamic controller is budget-cap-only")
    return {"k_values": ks, "stop_reason_counts": dict(stops)}
