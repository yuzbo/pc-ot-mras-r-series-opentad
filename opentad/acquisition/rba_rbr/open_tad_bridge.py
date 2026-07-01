import numpy as np

from .budget_controller import DynamicBudgetController
from .probe_builder import build_probe_candidates, build_scaffold_positions
from .regret_scorer import build_regret_labels
from .risk_map import build_risk_map
from .soft_bracket import build_soft_brackets
from .types import FORBIDDEN_DEPLOY_KEYS, METHOD_KEY, RbaRbrBudgetConfig, ROUTE_LABEL
from .validators import validate_no_leakage, validate_route_identity


FORMAL_PREVIEW_KEYS = {
    "rba_rbr_preview_actionness": "actionness",
    "preview_actionness": "actionness",
    "actionness_preview": "actionness",
}

DOWNSTREAM_GT_PAYLOAD_KEYS = {"gt_segments", "gt_labels"}
SELECTOR_FORBIDDEN_PAYLOAD_KEYS = set(FORBIDDEN_DEPLOY_KEYS).difference(DOWNSTREAM_GT_PAYLOAD_KEYS)


def _metadata_curve(results, keys, valid_len, required=True, default=None):
    for key in keys:
        if key in results:
            arr = np.asarray(results[key], dtype=np.float64).reshape(-1)
            if arr.size != int(valid_len):
                raise ValueError(f"{key} length {arr.size} does not match dense window length {int(valid_len)}")
            return np.clip(arr, 0.0, 1.0), key
    if required:
        raise ValueError(f"RBA-RBR requires deploy-visible preview metadata: one of {keys}")
    return default, None


def _diagnostic_preview(valid_len, sample_key):
    seed = 2166136261
    for ch in str(sample_key):
        seed ^= ord(ch)
        seed = (seed * 16777619) & 0xFFFFFFFF
    rng = np.random.RandomState(seed)
    x = np.arange(int(valid_len), dtype=np.float64)
    actionness = np.zeros(int(valid_len), dtype=np.float64) + 0.05
    center = int(rng.randint(max(2, valid_len // 5), max(3, valid_len * 4 // 5)))
    width = max(3, int(valid_len // 12))
    actionness[max(0, center - width) : min(valid_len, center + width)] = 0.70
    uncertainty = np.clip(0.10 + np.exp(-((x - (center + width + 8)) ** 2) / (2.0 * 3.0**2)), 0.0, 1.0)
    transition = np.clip(np.abs(np.gradient(actionness)) + 0.75 * uncertainty, 0.0, 1.0)
    return actionness, uncertainty, transition


def _normalize(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    lo = float(values.min())
    hi = float(values.max())
    if hi <= lo + 1e-12:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


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
        return None, None, None, None

    dense_window = np.asarray(dense_window, dtype=np.int64).reshape(-1)
    sample_count = int(max(2, min(int(valid_len), int(max(scout_sample_count, 2)))))
    sample_positions = np.unique(np.linspace(0, int(valid_len) - 1, sample_count).round().astype(np.int64))
    if sample_positions.size == 0:
        return None, None, None, None

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

    motion_n = _normalize(motion)
    contrast_n = _normalize(contrasts)
    mean_change_n = _normalize(np.abs(np.gradient(means))) if means.size > 1 else np.zeros_like(means)
    sampled_actionness = np.clip(0.55 * motion_n + 0.30 * contrast_n + 0.15 * mean_change_n, 0.0, 1.0)
    sampled_uncertainty = np.clip(
        0.50 * (1.0 - np.abs(sampled_actionness - 0.5) * 2.0) + 0.35 * contrast_n + 0.15 * mean_change_n,
        0.0,
        1.0,
    )
    sampled_transition = np.clip(0.75 * motion_n + 0.25 * mean_change_n, 0.0, 1.0)

    x = np.arange(int(valid_len), dtype=np.float64)
    xs = sample_positions.astype(np.float64)
    actionness = np.interp(x, xs, sampled_actionness)
    uncertainty = np.interp(x, xs, sampled_uncertainty)
    transition = np.interp(x, xs, sampled_transition)
    meta = {
        "formal_scout_source": "raw_rgb_lowres_scout",
        "scout_sample_count": int(sample_positions.size),
        "scout_positions": [int(pos) for pos in sample_positions.tolist()],
        "scout_frame_inds": [int(dense_window[int(pos)]) for pos in sample_positions.tolist()],
    }
    return (
        np.clip(actionness, 0.0, 1.0),
        np.clip(uncertainty, 0.0, 1.0),
        np.clip(transition, 0.0, 1.0),
        meta,
    )


def _preview_from_results(results, dense_window, allow_diagnostic_preview_fallback, scout_sample_count=32):
    valid_len = int(len(dense_window))
    actionness, action_key = _metadata_curve(results, tuple(FORMAL_PREVIEW_KEYS.keys()), valid_len, required=False)
    uncertainty, uncertainty_key = _metadata_curve(
        results,
        ("rba_rbr_preview_uncertainty", "preview_uncertainty", "uncertainty_preview"),
        valid_len,
        required=False,
    )
    transition, transition_key = _metadata_curve(
        results,
        ("rba_rbr_preview_transition", "preview_transition", "transition_preview", "motion_preview"),
        valid_len,
        required=False,
    )
    if actionness is not None:
        source = "deploy_visible_metadata_preview"
        return actionness, uncertainty, transition, source, {
            "actionness_key": action_key,
            "uncertainty_key": uncertainty_key,
            "transition_key": transition_key,
            "formal_scout_source": "deploy_visible_metadata_preview",
        }
    actionness, uncertainty, transition, raw_meta = _raw_rgb_lowres_preview(
        results,
        dense_window=dense_window,
        valid_len=valid_len,
        scout_sample_count=scout_sample_count,
    )
    if actionness is not None:
        return actionness, uncertainty, transition, "raw_rgb_lowres_scout", raw_meta
    if not allow_diagnostic_preview_fallback:
        raise ValueError(
            "RBA-RBR formal path requires deploy-visible preview metadata or a video_reader raw RGB scout; "
            "diagnostic fallback is disabled"
        )
    actionness, uncertainty, transition = _diagnostic_preview(valid_len, results.get("video_name", "unknown"))
    return actionness, uncertainty, transition, "diagnostic_deterministic_preview_fallback", {
        "diagnostic_only": True,
        "deterministic_fallback_allowed": True,
    }


def _build_budget(
    valid_len,
    target_frame_num,
    min_keep=None,
    max_keep=None,
    scaffold_k=4,
    min_detector_feature_keep=None,
    feature_stride=1,
    max_raw_gap=None,
    max_detector_gap=None,
):
    max_keep = int(max_keep) if max_keep is not None else int(target_frame_num)
    max_keep = max(1, min(int(max_keep), int(valid_len), int(target_frame_num)))
    min_keep = int(min_keep) if min_keep is not None else max(4, int(round(0.35 * max_keep)))
    min_keep = max(1, min(min_keep, max_keep))
    if min_detector_feature_keep is not None:
        min_detector_feature_keep = max(1, int(min_detector_feature_keep))
    if max_raw_gap is not None:
        max_raw_gap = max(1, int(max_raw_gap))
    if max_detector_gap is not None:
        max_detector_gap = max(1, int(max_detector_gap))
    return RbaRbrBudgetConfig(
        min_k=min_keep,
        max_k=max_keep,
        scaffold_k=int(scaffold_k),
        min_detector_feature_k=min_detector_feature_keep,
        feature_stride=int(max(feature_stride, 1)),
        max_raw_gap=max_raw_gap,
        max_detector_gap=max_detector_gap,
    )


def _selector_facing_metadata(results, preview_meta=None):
    """Build the metadata subset the selector is allowed to inspect.

    OpenTAD val/test payloads may carry GT for evaluator or target plumbing.
    RBA-RBR selection must ignore those fields, while still failing closed on
    teacher/cache/oracle/raw-prediction shortcuts or explicit provenance flags.
    """

    selector_meta = {
        "route_label": ROUTE_LABEL,
        "selector_inputs": {
            "preview_meta": {} if preview_meta is None else dict(preview_meta),
        },
        "selector_provenance": {},
    }
    for key in ("selector_provenance", "rba_rbr_selector_provenance"):
        if key in results:
            selector_meta["selector_provenance"].update(dict(results[key]))
    for key, value in results.items():
        key_l = str(key).lower()
        if key_l in SELECTOR_FORBIDDEN_PAYLOAD_KEYS or key_l.startswith("selection_uses_"):
            selector_meta[key] = value
    return selector_meta


def build_rba_rbr_open_tad_selection(
    results,
    dense_window,
    target_frame_num,
    split,
    gt_segments=None,
    gt_labels=None,
    min_keep=None,
    max_keep=None,
    scaffold_k=4,
    fps=30.0,
    window_id=0,
    train_value_labels=False,
    allow_diagnostic_preview_fallback=False,
    scout_sample_count=32,
    min_detector_feature_keep=None,
    feature_stride=1,
    max_raw_gap=None,
    max_detector_gap=None,
):
    validate_route_identity({"route_label": ROUTE_LABEL, "method": METHOD_KEY})
    split = str(split)
    if bool(train_value_labels) and split != "train":
        raise ValueError("RBA-RBR train_value_labels is only allowed for split='train'")
    dense_window = np.asarray(dense_window, dtype=np.int64).reshape(-1)
    valid_len = int(dense_window.shape[0])
    if valid_len <= 0:
        raise RuntimeError("RBA-RBR received an empty dense window")
    actionness, uncertainty, transition, preview_source, preview_meta = _preview_from_results(
        results,
        dense_window,
        allow_diagnostic_preview_fallback=allow_diagnostic_preview_fallback,
        scout_sample_count=scout_sample_count,
    )
    if split in {"val", "test", "deploy"}:
        validate_no_leakage(_selector_facing_metadata(results, preview_meta=preview_meta))
    video_id = str(results.get("video_name", "unknown"))
    budget = _build_budget(
        valid_len,
        target_frame_num,
        min_keep=min_keep,
        max_keep=max_keep,
        scaffold_k=scaffold_k,
        min_detector_feature_keep=min_detector_feature_keep,
        feature_stride=feature_stride,
        max_raw_gap=max_raw_gap,
        max_detector_gap=max_detector_gap,
    )
    scaffold_positions = build_scaffold_positions(valid_len, budget.scaffold_k)
    risk_map = build_risk_map(
        actionness=actionness,
        uncertainty=uncertainty,
        transition=transition,
        observed_positions=scaffold_positions,
        source=preview_source,
    )
    brackets = build_soft_brackets(risk_map)
    probes = build_probe_candidates(
        risk_map,
        brackets,
        scaffold_positions=scaffold_positions,
        video_id=video_id,
        split=split if split in {"train", "val", "test", "deploy", "synthetic"} else "deploy",
    )
    regret_labels = []
    if train_value_labels:
        regret_labels = build_regret_labels(probes, gt_segments=gt_segments, dense_T=valid_len, split=split)
    dense_inputs = np.stack([np.arange(valid_len, dtype=np.float64), actionness.astype(np.float64)], axis=1)
    result = DynamicBudgetController(budget).select(
        risk_map,
        brackets,
        probes,
        scaffold_positions=scaffold_positions,
        video_id=video_id,
        window_id=int(window_id),
        split=split,
        fps=fps,
        dense_inputs=dense_inputs,
    )
    keep_positions = np.asarray(result.selected_positions, dtype=np.int64)
    selected_frame_inds = dense_window[keep_positions]
    result.deploy_ledger.update(
        {
            "method": METHOD_KEY,
            "target_frame_num": int(target_frame_num),
            "selected_frame_inds": [int(pos) for pos in selected_frame_inds.tolist()],
            "preview_source": preview_source,
            "preview_meta": preview_meta,
            "scout_is_deploy_visible": preview_source in {"deploy_visible_metadata_preview", "raw_rgb_lowres_scout"},
            "diagnostic_preview_fallback_used": preview_source == "diagnostic_deterministic_preview_fallback",
            "diagnostic_preview_fallback_allowed": bool(allow_diagnostic_preview_fallback),
            "num_candidate_probes": int(len(probes)),
            "num_regret_labels": int(len(regret_labels)),
            "train_value_labels_present": bool(regret_labels),
            "value_labels_used_at_test": False,
        }
    )
    return {
        "keep_positions": keep_positions,
        "selected_frame_inds": selected_frame_inds,
        "selection_result": result,
        "ledger": result.deploy_ledger,
        "candidate_probes": probes,
        "regret_labels": regret_labels,
    }
