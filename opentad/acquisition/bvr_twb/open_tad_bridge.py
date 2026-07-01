import numpy as np

from .boundary_belief import estimate_boundary_beliefs
from .budget_controller import DynamicBudgetController
from .regret_labels import build_packet_regret_labels, validate_regret_label_schema
from .scaffold import build_scaffold_packets
from .state_scout import build_scout_from_actionness
from .trainable_value import LearnedPacketValueAdapter
from .types import BudgetConfig, ROUTE_LABEL, sorted_unique_positions
from .value_predictor import PacketValuePredictor
from .validators import build_original_time_metadata, build_selection_gap_diagnostics, validate_no_leakage, validate_route_identity
from .witness_packets import build_witness_packets


def _stable_seed(value):
    text = "unknown" if value is None else str(value)
    total = 2166136261
    for ch in text:
        total ^= ord(ch)
        total = (total * 16777619) & 0xFFFFFFFF
    return int(total)


FORMAL_SCOUT_SOURCES = {
    "deploy_visible_raw_or_metadata_scout",
    "deploy_visible_metadata_scout",
    "raw_rgb_lowres_scout",
}

DIAGNOSTIC_SCOUT_SOURCE = "diagnostic_deterministic_preview"

DEPLOY_VALUE_MODES = {
    "deploy_heuristic_voi",
    "learned_packet_value",
}


def _preview_from_metadata(results, valid_len):
    for key in ("bvr_twb_preview_actionness", "preview_actionness", "actionness_preview"):
        if key in results:
            arr = np.asarray(results[key], dtype=np.float64).reshape(-1)
            if arr.size == valid_len:
                return np.clip(arr, 0.0, 1.0), "deploy_visible_metadata_actionness", key
            raise ValueError(f"{key} length {arr.size} does not match dense window length {valid_len}")
    return None, None, None


def _frame_to_gray(frame, resize_long_side=48):
    arr = np.asarray(frame)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 2:
        gray = arr.astype(np.float64)
    elif arr.ndim == 3:
        gray = arr[..., :3].astype(np.float64).mean(axis=2)
    else:
        raise ValueError("raw RGB scout frame must be HxW or HxWxC")
    h, w = gray.shape[:2]
    long_side = max(int(h), int(w), 1)
    step = max(1, int(np.floor(long_side / float(max(resize_long_side, 1)))))
    return gray[::step, ::step] / 255.0


def _read_raw_frame(video_reader, frame_index):
    frame_index = int(frame_index)
    if hasattr(video_reader, "get_batch"):
        batch = video_reader.get_batch([frame_index])
        if hasattr(batch, "asnumpy"):
            batch = batch.asnumpy()
        return np.asarray(batch)[0]
    frame = video_reader[frame_index]
    if hasattr(frame, "asnumpy"):
        frame = frame.asnumpy()
    return np.asarray(frame)


def _raw_rgb_lowres_preview(results, dense_window, valid_len, scout_sample_count=32):
    video_reader = results.get("video_reader")
    if video_reader is None:
        return None, None, None
    dense_window = np.asarray(dense_window, dtype=np.int64).reshape(-1)
    sample_count = int(max(2, min(valid_len, scout_sample_count)))
    sample_positions = np.unique(np.linspace(0, valid_len - 1, sample_count).round().astype(np.int64))
    if sample_positions.size == 0:
        return None, None, None

    grays = []
    for pos in sample_positions:
        grays.append(_frame_to_gray(_read_raw_frame(video_reader, dense_window[int(pos)])))
    means = np.asarray([float(gray.mean()) for gray in grays], dtype=np.float64)
    contrasts = np.asarray([float(gray.std()) for gray in grays], dtype=np.float64)
    motion = np.zeros(sample_positions.shape[0], dtype=np.float64)
    for idx in range(1, len(grays)):
        left = grays[idx - 1]
        right = grays[idx]
        h = min(left.shape[0], right.shape[0])
        w = min(left.shape[1], right.shape[1])
        if h > 0 and w > 0:
            motion[idx] = float(np.mean(np.abs(left[:h, :w] - right[:h, :w])))
    if motion.size > 1:
        motion[0] = motion[1]

    def norm(values):
        values = np.asarray(values, dtype=np.float64)
        lo = float(values.min())
        hi = float(values.max())
        if hi <= lo + 1e-12:
            return np.zeros_like(values)
        return (values - lo) / (hi - lo)

    motion_n = norm(motion)
    contrast_n = norm(contrasts)
    mean_change_n = norm(np.abs(np.gradient(means))) if means.size > 1 else np.zeros_like(means)
    sampled_preview = np.clip(0.55 * motion_n + 0.30 * contrast_n + 0.15 * mean_change_n, 0.0, 1.0)
    x = np.arange(valid_len, dtype=np.float64)
    preview = np.interp(x, sample_positions.astype(np.float64), sampled_preview)
    motion_preview = np.interp(x, sample_positions.astype(np.float64), motion_n)
    meta = {
        "scout_sample_count": int(sample_positions.size),
        "scout_positions": [int(pos) for pos in sample_positions.tolist()],
        "scout_frame_inds": [int(dense_window[int(pos)]) for pos in sample_positions.tolist()],
    }
    return np.clip(preview, 0.0, 1.0), "raw_rgb_lowres_scout", (np.clip(motion_preview, 0.0, 1.0), meta)


def _diagnostic_preview(valid_len, sample_key):
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
    return np.clip(base + bumps, 0.0, 1.0), "diagnostic_deterministic_preview_fallback"


def _preview_from_results(
    results,
    dense_window,
    valid_len,
    sample_key,
    scout_source,
    require_deploy_visible_scout,
    allow_diagnostic_preview_fallback,
    scout_sample_count,
):
    scout_source = str(scout_source or "deploy_visible_raw_or_metadata_scout")
    if scout_source not in FORMAL_SCOUT_SOURCES and scout_source != DIAGNOSTIC_SCOUT_SOURCE:
        raise ValueError(f"unsupported BVR-TWB scout source: {scout_source}")

    if scout_source in {"deploy_visible_raw_or_metadata_scout", "deploy_visible_metadata_scout"}:
        preview, source, key = _preview_from_metadata(results, valid_len)
        if preview is not None:
            return preview, source, None, {"metadata_key": key, "formal_scout_source": scout_source}

    if scout_source in {"deploy_visible_raw_or_metadata_scout", "raw_rgb_lowres_scout"}:
        preview, source, raw_extra = _raw_rgb_lowres_preview(
            results,
            dense_window=dense_window,
            valid_len=valid_len,
            scout_sample_count=scout_sample_count,
        )
        if preview is not None:
            motion, raw_meta = raw_extra
            raw_meta["formal_scout_source"] = scout_source
            return preview, source, motion, raw_meta

    if allow_diagnostic_preview_fallback or scout_source == DIAGNOSTIC_SCOUT_SOURCE:
        preview, source = _diagnostic_preview(valid_len, sample_key)
        return preview, source, None, {
            "formal_scout_source": scout_source,
            "diagnostic_only": True,
            "deterministic_fallback_allowed": True,
        }

    if require_deploy_visible_scout:
        raise ValueError(
            "BVR-TWB formal path requires a deploy-visible scout. Provide "
            "bvr_twb_preview_actionness/preview_actionness/actionness_preview metadata, "
            "or run after DecordInit with video_reader for raw_rgb_lowres_scout. "
            "Deterministic preview fallback is diagnostic/precheck-only and disabled here."
        )
    raise ValueError("BVR-TWB scout source unavailable and diagnostic fallback is disabled")


def _build_budget_config(
    valid_len,
    target_frame_num,
    min_keep=None,
    max_keep=None,
    max_gap=None,
    feature_stride=1,
    min_detector_keep=None,
):
    target = int(max(target_frame_num or 0, 1))
    feature_stride = int(max(int(feature_stride), 1))
    max_keep = int(max_keep) if max_keep is not None else min(valid_len, target)
    min_keep = int(min_keep) if min_keep is not None else max(4, int(round(0.35 * max_keep)))
    max_keep = max(min(max_keep, valid_len), 1)
    min_keep = max(1, min(min_keep, max_keep))
    if min_detector_keep is not None:
        min_detector_keep = int(min_detector_keep)
        if min_detector_keep < 1:
            raise ValueError("BVR-TWB min_detector_keep must be positive when set")
        required_raw_keep = int(min_detector_keep * feature_stride)
        if required_raw_keep > max_keep:
            raise ValueError(
                "BVR-TWB effective detector-token floor is infeasible under max_keep: "
                f"min_detector_keep={min_detector_keep} feature_stride={feature_stride} "
                f"required_raw_keep={required_raw_keep} max_keep={max_keep}"
            )
        min_keep = max(min_keep, required_raw_keep)
    max_gap = int(max_gap) if max_gap is not None else max(8, int(np.ceil(valid_len / max(max_keep, 1))) * 4)
    return BudgetConfig(
        min_k=min_keep,
        max_k=max_keep,
        max_gap=max_gap,
        min_detector_k=min_detector_keep,
        feature_stride=feature_stride,
        min_marginal_value=0.30,
    )


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
    scout_source="deploy_visible_raw_or_metadata_scout",
    require_deploy_visible_scout=True,
    allow_diagnostic_preview_fallback=False,
    scout_sample_count=32,
    value_mode="deploy_heuristic_voi",
    feature_stride=1,
    min_detector_keep=None,
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
    p_action, preview_source, raw_motion, scout_meta = _preview_from_results(
        results,
        dense_window=dense_window,
        valid_len=valid_len,
        sample_key=sample_key,
        scout_source=scout_source,
        require_deploy_visible_scout=require_deploy_visible_scout,
        allow_diagnostic_preview_fallback=allow_diagnostic_preview_fallback,
        scout_sample_count=scout_sample_count,
    )
    motion = None
    if "bvr_twb_preview_motion" in results:
        motion = np.asarray(results["bvr_twb_preview_motion"], dtype=np.float64).reshape(-1)
        if motion.size != valid_len:
            raise ValueError("bvr_twb_preview_motion length must match dense window length")
    elif raw_motion is not None:
        motion = raw_motion

    scout = build_scout_from_actionness(
        p_action,
        motion_signal=motion,
        metadata={"route_label": ROUTE_LABEL, "preview_source": preview_source, "scout_meta": scout_meta},
    )
    budget = _build_budget_config(
        valid_len,
        target_frame_num,
        min_keep=min_keep,
        max_keep=max_keep,
        max_gap=max_gap,
        feature_stride=feature_stride,
        min_detector_keep=min_detector_keep,
    )
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
    value_mode = str(value_mode or "deploy_heuristic_voi")
    if value_mode not in DEPLOY_VALUE_MODES:
        raise ValueError(f"unsupported BVR-TWB value_mode: {value_mode}")
    if value_mode == "learned_packet_value":
        if value_model is None:
            raise ValueError("BVR-TWB learned_packet_value mode requires an explicit loaded value_model")
        value_predictor = LearnedPacketValueAdapter(model=value_model)
        value_model_used = True
    else:
        if value_model is not None:
            raise ValueError("BVR-TWB deploy_heuristic_voi mode must not receive value_model")
        value_predictor = PacketValuePredictor(mode="deploy_voi_heuristic")
        value_model_used = False
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
        "min_detector_feature_k": None if budget.min_detector_k is None else int(budget.min_detector_k),
        "dynamic_min_detector_feature_k": None if budget.min_detector_k is None else int(budget.min_detector_k),
        "max_adapter_padding_duplicate_ratio": 0.5,
        "budget_stop_reason": selection.stop_reason,
        "selection_gap_diagnostics": build_selection_gap_diagnostics(keep_positions, valid_len, budget.max_gap),
        "preview_source": preview_source,
        "scout_source": str(scout_source),
        "scout_is_deploy_visible": preview_source in {
            "deploy_visible_metadata_actionness",
            "raw_rgb_lowres_scout",
        },
        "deterministic_preview_fallback_used": preview_source == "diagnostic_deterministic_preview_fallback",
        "diagnostic_preview_fallback_allowed": bool(allow_diagnostic_preview_fallback),
        "scout_meta": scout_meta,
        "value_mode": value_mode,
        "value_model_used": bool(value_model_used),
        "value_labels_used_at_test": False,
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
