import math

import numpy as np

from .adapter_bridge import build_detector_feature_centers_from_raw
from .sparse_gather import sparse_gather
from .types import METHOD_KEY, ROUTE_LABEL
from .validators import (
    LOCAL_PRECHECK_CLAIM_STATUS,
    build_original_time_metadata,
    build_selection_gap_diagnostics,
    exact_uniform_positions,
    validate_no_leakage,
    validate_rba_rbr_ledger,
    validate_route_identity,
)


CONTROL_MODE_UNIFORM_RAW = "uniform_raw"
CONTROL_DIAGNOSTIC_CLAIM_STATUS = LOCAL_PRECHECK_CLAIM_STATUS


def _detector_gap_summary(detector_positions, dense_T):
    centers = np.asarray(detector_positions, dtype=np.float64).reshape(-1)
    dense_T = int(dense_T)
    if centers.size == 0:
        return {"max_gap": float(dense_T), "mean_gap": float(dense_T), "gaps": [float(dense_T)]}
    gaps = [float(centers[0] + 1.0)]
    gaps.extend(float(right - left) for left, right in zip(centers, centers[1:]))
    gaps.append(float(dense_T - centers[-1]))
    return {
        "max_gap": float(max(gaps)),
        "mean_gap": float(np.mean(gaps)),
        "gaps": [float(gap) for gap in gaps],
    }


def build_rba_rbr_uniform_control_selection(
    results,
    dense_window,
    target_frame_num,
    split,
    control_keep=None,
    max_keep=None,
    fps=30.0,
    window_id=0,
    feature_stride=1,
    selector_meta=None,
):
    """Build a selector-free uniform control while preserving the RBA bridge contract."""

    validate_route_identity({"route_label": ROUTE_LABEL, "method": METHOD_KEY, "control_mode": CONTROL_MODE_UNIFORM_RAW})
    validate_no_leakage({} if selector_meta is None else selector_meta)

    dense_window = np.asarray(dense_window, dtype=np.int64).reshape(-1)
    valid_len = int(dense_window.shape[0])
    if valid_len <= 0:
        raise RuntimeError("RBA-RBR uniform control received an empty dense window")

    target_frame_num = int(target_frame_num)
    feature_stride = int(max(feature_stride, 1))
    if max_keep is None:
        max_keep = target_frame_num
    max_keep = max(1, min(int(max_keep), valid_len, target_frame_num))
    if control_keep is None:
        keep_k = max_keep
    else:
        keep_k = max(1, min(int(control_keep), max_keep, valid_len, target_frame_num))

    keep_positions = np.asarray(exact_uniform_positions(valid_len, keep_k), dtype=np.int64)
    if keep_positions.size != keep_k:
        raise ValueError(f"RBA-RBR uniform control expected {keep_k} positions, got {keep_positions.size}")
    selected_frame_inds = dense_window[keep_positions]
    detector_positions = build_detector_feature_centers_from_raw(keep_positions, feature_stride=feature_stride)
    detector_valid_k = int(math.ceil(float(keep_k) / float(feature_stride)))
    detector_target_len = int(math.ceil(float(target_frame_num) / float(feature_stride)))
    dense_inputs = np.stack([np.arange(valid_len, dtype=np.float64), np.ones(valid_len, dtype=np.float64)], axis=1)
    _, gather_evidence = sparse_gather(dense_inputs, keep_positions, temporal_dim=0)
    raw_gap = build_selection_gap_diagnostics(keep_positions.tolist(), valid_len)
    detector_gap = _detector_gap_summary(detector_positions, valid_len)
    video_id = str(results.get("video_name", "unknown"))

    ledger = {
        "route_label": ROUTE_LABEL,
        "method": METHOD_KEY,
        "selector_method": "forced_uniform_control_diagnostic",
        "control_diagnostic_only": True,
        "control_name": "forced_uniform_through_rba_bridge",
        "rba_rbr_control_mode": CONTROL_MODE_UNIFORM_RAW,
        "rba_rbr_control_keep": int(keep_k),
        "split": str(split),
        "video_id": video_id,
        "window_id": int(window_id),
        "dense_T": int(valid_len),
        "selected_positions": [int(pos) for pos in keep_positions.tolist()],
        "valid_k": int(keep_k),
        "dynamic_target_k": int(keep_k),
        "min_k": int(keep_k),
        "max_k": int(max_keep),
        "scaffold_k": 0,
        "target_frame_num": int(target_frame_num),
        "budget_stop_reason": "control_forced_uniform",
        "selection_gap_diagnostics": raw_gap,
        "selected_probe_ids": [],
        "selected_probe_stages": [],
        "rescue_outside_hard_bracket_count": 0,
        "soft_bracket_count": 0,
        "hard_bracket_count": 0,
        "soft_bracket_is_prior_not_mask": False,
        "original_time_metadata": build_original_time_metadata(valid_len, keep_positions.tolist(), fps=fps),
        "real_sparse_evidence": gather_evidence,
        "claim_status": CONTROL_DIAGNOSTIC_CLAIM_STATUS,
        "no_metric_claim": True,
        "no_runtime_claim": True,
        "no_deploy_claim": True,
        "no_paper_claim": True,
        "no_sparse_compute_claim": True,
        "full_train_unlocked": False,
        "value_labels_used_at_test": False,
        "selector_provenance": {
            "selection_uses_gt": False,
            "selection_uses_teacher": False,
            "selection_uses_prediction_cache": False,
            "selection_uses_raw_detector_prediction": False,
            "selection_uses_oracle_boundary": False,
            "selection_uses_oracle_residual": False,
        },
        "selected_frame_inds": [int(pos) for pos in selected_frame_inds.tolist()],
        "preview_source": "control_uniform_no_preview_selector",
        "preview_meta": {"control_diagnostic_only": True, "preview_not_used": True},
        "scout_is_deploy_visible": True,
        "diagnostic_preview_fallback_used": False,
        "diagnostic_preview_fallback_allowed": False,
        "num_candidate_probes": 0,
        "num_regret_labels": 0,
        "train_value_labels_present": False,
        "train_value_labels_allowed": False,
        "detector_feature_valid_k": int(detector_valid_k),
        "detector_feature_target_len": int(detector_target_len),
        "detector_feature_positions": [float(pos) for pos in detector_positions.tolist()],
        "rba_rbr_feature_stride": int(feature_stride),
        "control_raw_gap_summary": raw_gap,
        "control_detector_gap_summary": detector_gap,
        "control_raw_density": float(keep_k) / float(valid_len),
        "control_detector_density": float(detector_valid_k) / float(max(detector_target_len, 1)),
    }
    validate_rba_rbr_ledger(ledger)
    return {
        "keep_positions": keep_positions,
        "selected_frame_inds": selected_frame_inds,
        "selection_result": None,
        "ledger": ledger,
        "candidate_probes": [],
        "regret_labels": [],
    }
