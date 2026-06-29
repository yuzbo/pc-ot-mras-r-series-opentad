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


def required_witness_roles_for_bracket(bracket):
    if bracket.kind == "start_like":
        return {"transition_before", "transition_center", "action_core"}
    if bracket.kind == "end_like":
        return {"transition_center", "transition_after", "action_core"}
    if bracket.kind == "action_island":
        return {"transition_before", "transition_after", "short_action_guard"}
    if bracket.kind == "unknown_transition":
        return {"transition_before", "transition_center", "transition_after", "ambiguity_probe"}
    return {"transition_center"}


def update_boundary_belief_trace(brackets, selected_packets, safe_width=None):
    selected_by_bracket = {}
    for packet in selected_packets:
        if packet.bracket_id is None:
            continue
        selected_by_bracket.setdefault(packet.bracket_id, []).append(packet)

    trace = []
    for bracket in brackets:
        required = required_witness_roles_for_bracket(bracket)
        selected = selected_by_bracket.get(bracket.bracket_id, [])
        roles = {packet.role for packet in selected}
        positions = sorted({pos for packet in selected for pos in packet.positions})
        role_coverage = len(required.intersection(roles)) / float(max(len(required), 1))
        center_hit = any(abs(int(pos) - int(bracket.center_pos)) <= 1 for pos in positions)
        two_sided = bool({"transition_before", "transition_after"}.issubset(roles))
        short_guard = bool("short_action_guard" in roles)
        reduction = 0.45 * role_coverage + 0.18 * float(center_hit) + 0.12 * float(two_sided) + 0.10 * float(short_guard)
        if not selected:
            reduction = 0.0
        posterior_width = float(max(1.0, float(bracket.width_p80_frames) * (1.0 - min(reduction, 0.72))))
        posterior_entropy = float(max(0.0, float(bracket.entropy) * (1.0 - min(reduction, 0.60))))
        trace.append(
            {
                "bracket_id": int(bracket.bracket_id),
                "kind": bracket.kind,
                "initial_width_p80_frames": float(bracket.width_p80_frames),
                "posterior_width_p80_frames": posterior_width,
                "initial_entropy": float(bracket.entropy),
                "posterior_entropy": posterior_entropy,
                "required_roles": sorted(required),
                "selected_roles": sorted(roles),
                "role_coverage": float(role_coverage),
                "center_hit": bool(center_hit),
                "two_sided_witness": bool(two_sided),
                "updated_from_selected_witness": bool(selected),
                "belief_width_safe": bool(safe_width is not None and posterior_width <= float(safe_width)),
            }
        )
    return trace
