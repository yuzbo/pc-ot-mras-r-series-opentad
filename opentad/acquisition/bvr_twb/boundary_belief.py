import numpy as np

from .types import BracketState, ScoutCurves


def _top_peaks(score, max_peaks, min_distance, threshold):
    order = np.argsort(-score, kind="mergesort")
    peaks = []
    for idx in order.tolist():
        value = float(score[int(idx)])
        if value < threshold:
            break
        if any(abs(int(idx) - prev) < min_distance for prev in peaks):
            continue
        peaks.append(int(idx))
        if len(peaks) >= max_peaks:
            break
    return sorted(peaks)


def _safe_mean(arr, lo, hi):
    lo = max(0, int(lo))
    hi = min(int(arr.shape[0]), int(hi))
    if hi <= lo:
        return 0.0
    return float(np.mean(arr[lo:hi]))


def estimate_boundary_beliefs(
    scout: ScoutCurves,
    video_id="synthetic",
    window_id=0,
    split="synthetic",
    max_brackets=8,
    radius=None,
):
    dense_T = int(scout.dense_T)
    radius = int(radius if radius is not None else max(2, dense_T // 24))
    score = np.clip(
        0.44 * scout.transition_score
        + 0.24 * scout.uncertainty
        + 0.20 * scout.short_action_risk
        + 0.12 * (1.0 - scout.persistence),
        0.0,
        1.0,
    )
    threshold = max(0.18, float(np.percentile(score, 72)))
    peaks = _top_peaks(score, max_peaks=max_brackets, min_distance=max(2, radius), threshold=threshold)

    if not peaks:
        center = int(np.argmax(score))
        peaks = [center]

    brackets = []
    for bid, center in enumerate(peaks):
        left = max(0, int(center) - radius)
        right = min(dense_T - 1, int(center) + radius)
        left_action = _safe_mean(scout.p_action, left, center + 1)
        right_action = _safe_mean(scout.p_action, center, right + 1)
        left_bg = _safe_mean(scout.p_background, left, center + 1)
        right_bg = _safe_mean(scout.p_background, center, right + 1)
        contrast = abs(right_action - left_action) + 0.5 * abs(right_bg - left_bg)
        if scout.short_action_risk[center] > 0.60:
            kind = "action_island"
        elif contrast < 0.18:
            kind = "unknown_transition"
        elif right_action > left_action:
            kind = "start_like"
        else:
            kind = "end_like"
        width = right - left + 1
        entropy = float(np.clip(0.48 * scout.uncertainty[center] + 0.32 * score[center] + 0.20 * (1.0 - contrast), 0.0, 1.0))
        confidence = float(np.clip(1.0 - entropy + 0.25 * scout.transition_score[center], 0.0, 1.0))
        gap_risk = float(np.clip(width / float(max(dense_T, 1)) + 0.60 * scout.uncertainty[center], 0.0, 1.0))
        state = "active"
        if entropy < 0.34 and scout.transition_score[center] < 0.22 and scout.short_action_risk[center] < 0.35:
            state = "belief_safe"
        brackets.append(
            BracketState(
                bracket_id=bid,
                video_id=video_id,
                window_id=window_id,
                split=split,
                kind=kind,
                left_pos=left,
                right_pos=right,
                center_pos=int(center),
                dense_T=dense_T,
                peak_transition=float(scout.transition_score[center]),
                mean_uncertainty=_safe_mean(scout.uncertainty, left, right + 1),
                action_left_mean=left_action,
                action_right_mean=right_action,
                persistence=_safe_mean(scout.persistence, left, right + 1),
                short_action_risk=float(scout.short_action_risk[center]),
                gap_risk=gap_risk,
                entropy=entropy,
                width_p50_frames=float(max(1, width * (0.35 + 0.20 * entropy))),
                width_p80_frames=float(max(1, width * (0.60 + 0.25 * entropy))),
                confidence=confidence,
                two_sided_state_contrast=float(np.clip(contrast, 0.0, 1.0)),
                state=state,
            )
        )
    return brackets
