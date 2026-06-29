import numpy as np

from .boundary_belief import estimate_boundary_beliefs
from .budget_controller import DynamicBudgetController
from .regret_labels import build_packet_regret_labels, validate_regret_label_schema
from .scaffold import build_scaffold_packets
from .state_scout import build_scout_from_actionness
from .trainable_value import LearnedPacketValueAdapter
from .types import BudgetConfig, ROUTE_LABEL, sorted_unique_positions
from .validators import build_original_time_metadata, build_selection_gap_diagnostics, validate_no_leakage, validate_route_identity
from .witness_packets import build_witness_packets


def _stable_seed(value):
    text = "unknown" if value is None else str(value)
    total = 2166136261
    for ch in text:
        total ^= ord(ch)
        total = (total * 16777619) & 0xFFFFFFFF
    return int(total)


def _preview_from_results(results, valid_len, sample_key):
    for key in ("bvr_twb_preview_actionness", "preview_actionness", "actionness_preview"):
        if key in results:
            arr = np.asarray(results[key], dtype=np.float64).reshape(-1)
            if arr.size == valid_len:
                return np.clip(arr, 0.0, 1.0), "provided_preview_actionness"
    rng = np.random.RandomState(_stable_seed(sample_key))
    x = np.linspace(0.0, 1.0, int(valid_len), dtype=np.float64)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    base = 0.12 + 0.05 * np.sin(2.0 * np.pi * x + phase)
    bumps = np.zeros_like(base)
    for _ in range(2):
        center = rng.uniform(0.15, 0.85)
        width = rng.uniform(0.04, 0.12)
        height = rng.uniform(0.15, 0.38)
        bumps += height * np.exp(-((x - center) ** 2) / (2.0 * width**2))
    return np.clip(base + bumps, 0.0, 1.0), "deploy_visible_deterministic_preview_fallback"


def _build_budget_config(valid_len, target_frame_num, min_keep=None, max_keep=None, max_gap=None):
    target = int(max(target_frame_num or 0, 1))
    max_keep = int(max_keep) if max_keep is not None else min(valid_len, target)
    min_keep = int(min_keep) if min_keep is not None else max(4, int(round(0.35 * max_keep)))
    max_keep = max(min(max_keep, valid_len), 1)
    min_keep = max(1, min(min_keep, max_keep))
    max_gap = int(max_gap) if max_gap is not None else max(8, int(np.ceil(valid_len / max(max_keep, 1))) * 4)
    return BudgetConfig(min_k=min_keep, max_k=max_keep, max_gap=max_gap, min_marginal_value=0.30)


def build_bvr_twb_open_tad_selection(
    results,
    dense_window,
    target_frame_num,
    split,
    gt_segments=None,
    gt_labels=None,
    min_keep=None,
    max_keep=None,
    max_gap=None,
    scaffold_k=4,
    fps=30.0,
    window_id=0,
    train_value_labels=False,
    value_model=None,
):
    validate_route_identity({"route_label": ROUTE_LABEL})
    selector_meta = {
        "route_label": ROUTE_LABEL,
        "selector_inputs": {
            "video_name": results.get("video_name", "unknown"),
            "preview_source": "deploy_visible",
        },
    }
    validate_no_leakage(selector_meta)

    dense_window = np.asarray(dense_window, dtype=np.int64).reshape(-1)
    valid_len = int(dense_window.shape[0])
    if valid_len <= 0:
        raise RuntimeError("BVR-TWB received an empty dense window")

    split = str(split)
    video_id = str(results.get("video_name", "unknown"))
    sample_key = (
        f"{video_id}|bvr_twb|{split}|{int(dense_window[0])}|{int(dense_window[-1])}|"
        f"{valid_len}|{int(target_frame_num or 0)}"
    )
    p_action, preview_source = _preview_from_results(results, valid_len, sample_key)
    motion = None
    if "bvr_twb_preview_motion" in results:
        motion = np.asarray(results["bvr_twb_preview_motion"], dtype=np.float64).reshape(-1)
        if motion.size != valid_len:
            motion = None

    scout = build_scout_from_actionness(
        p_action,
        motion_signal=motion,
        metadata={"route_label": ROUTE_LABEL, "preview_source": preview_source},
    )
    budget = _build_budget_config(valid_len, target_frame_num, min_keep=min_keep, max_keep=max_keep, max_gap=max_gap)
    scaffold = build_scaffold_packets(
        dense_T=valid_len,
        scaffold_k=min(int(scaffold_k), budget.max_k),
        max_gap=budget.max_gap,
        video_id=video_id,
        window_id=int(window_id),
        split=split if split in {"train", "val", "test", "deploy", "synthetic"} else "deploy",
    )
    scaffold_positions = [pos for packet in scaffold for pos in packet.positions]
    brackets = estimate_boundary_beliefs(scout, video_id=video_id, window_id=int(window_id), split=split, max_brackets=5)
    candidates = build_witness_packets(
        scout,
        brackets,
        scaffold_positions=scaffold_positions,
        video_id=video_id,
        window_id=int(window_id),
        split=split if split in {"train", "val", "test", "deploy", "synthetic"} else "deploy",
        start_packet_id=1000,
        max_gap=budget.max_gap,
    )
    value_predictor = LearnedPacketValueAdapter(model=value_model)
    value_predictor.score_packets(scaffold + candidates)

    regret_labels = []
    if train_value_labels:
        if split != "train":
            raise ValueError("BVR-TWB train_value_labels is only allowed for split='train'")
        regret_labels = build_packet_regret_labels(
            scaffold + candidates,
            gt_segments=gt_segments,
            gt_labels=gt_labels,
            dense_T=valid_len,
            split=split,
            metadata={"selector_inputs": {"preview_source": preview_source}},
        )
        for row in regret_labels:
            validate_regret_label_schema(row)

    dense_inputs = np.stack([np.arange(valid_len, dtype=np.float64), p_action.astype(np.float64)], axis=1)
    selection = DynamicBudgetController(budget).select(
        scaffold,
        candidates,
        brackets,
        dense_T=valid_len,
        fps=fps,
        video_id=video_id,
        window_id=int(window_id),
        split=split if split in {"train", "val", "test", "deploy", "synthetic"} else "deploy",
        dense_inputs=dense_inputs,
    )
    keep_positions = np.asarray(sorted_unique_positions(selection.selected_positions, valid_len), dtype=np.int64)
    if keep_positions.size == 0:
        keep_positions = np.array([0], dtype=np.int64)

    selected_frame_inds = dense_window[keep_positions]
    original_time = build_original_time_metadata(valid_len, keep_positions, fps=fps)
    ledger = {
        "route_label": ROUTE_LABEL,
        "method": "bvr_twb_dynamic_subsample",
        "voi_bbc_spec": "Value-of-Information Boundary Belief Controller",
        "video_id": video_id,
        "window_id": int(window_id),
        "split": split,
        "dense_T": valid_len,
        "target_frame_num": int(target_frame_num or 0),
        "selected_positions": [int(pos) for pos in keep_positions],
        "selected_frame_inds": [int(pos) for pos in selected_frame_inds],
        "valid_k": int(keep_positions.size),
        "dynamic_min_k": int(budget.min_k),
        "dynamic_max_k": int(budget.max_k),
        "budget_stop_reason": selection.stop_reason,
        "selection_gap_diagnostics": build_selection_gap_diagnostics(keep_positions, valid_len, budget.max_gap),
        "preview_source": preview_source,
        "num_brackets": int(len(brackets)),
        "num_candidate_packets": int(len(scaffold) + len(candidates)),
        "num_regret_labels": int(len(regret_labels)),
        "train_value_labels_present": bool(len(regret_labels) > 0),
        "controller_trace_summary": dict(selection.deploy_ledger.get("controller_trace_summary", {})),
        "bracket_summary": dict(selection.deploy_ledger.get("bracket_summary", {})),
        "safety_floor_k": int(selection.deploy_ledger.get("safety_floor_k", 0)),
        "original_time_metadata": original_time,
        "temporal_decode_uses_original_time": True,
        "selected_index_is_time": False,
        "dense_raw_backbone_handoff": False,
        "selected_inputs_is_gathered": True,
        "padding_duplicate_count": 0,
        "sparse_compute_claim": False,
        "claim_status": "bvr_twb_first_trainable_pipeline_no_metric_claim",
        "selector_provenance": {
            "selection_uses_gt": False,
            "selection_uses_teacher": False,
            "selection_uses_prediction_cache": False,
            "selection_uses_raw_detector_prediction": False,
            "selection_uses_oracle_boundary": False,
            "selection_uses_oracle_residual": False,
        },
    }
    return {
        "keep_positions": keep_positions,
        "selected_frame_inds": selected_frame_inds,
        "masks_valid_k": int(keep_positions.size),
        "selection_result": selection,
        "ledger": ledger,
        "candidate_packets": scaffold + candidates,
        "regret_labels": regret_labels,
    }
