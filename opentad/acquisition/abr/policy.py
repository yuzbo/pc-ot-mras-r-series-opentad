from __future__ import annotations

import hashlib
import math
from typing import Dict, Iterable, List, Sequence, Tuple

from .types import ABR_BRACKET_POLICY_NAME, ABRConfig, BracketState


def stable_seed(value: object) -> int:
    digest = hashlib.sha1(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def clamp_position(position: int, dense_t: int) -> int:
    return int(min(max(int(position), 0), max(int(dense_t) - 1, 0)))


def sorted_unique_in_range(positions: Iterable[int], dense_t: int) -> List[int]:
    return sorted({clamp_position(pos, dense_t) for pos in positions if dense_t > 0})


def state_at_position(curve: Sequence[float], position: int, config: ABRConfig | None = None) -> str:
    cfg = config or ABRConfig()
    if len(curve) == 0:
        return "ambiguous"
    value = float(curve[clamp_position(position, len(curve))])
    if value >= cfg.action_threshold:
        return "action"
    if value <= cfg.background_threshold:
        return "background"
    return "ambiguous"


def deterministic_fallback_scout(dense_t: int, video_id: str = "unknown") -> List[float]:
    if dense_t <= 0:
        return []
    seed = stable_seed(video_id)
    first = 0.20 + ((seed % 17) / 100.0)
    second = 0.62 + (((seed >> 8) % 19) / 100.0)
    width_a = max(3, dense_t // 18)
    width_b = max(4, dense_t // 12)
    curve = []
    for idx in range(dense_t):
        x = idx / max(dense_t - 1, 1)
        bump_a = math.exp(-((x - first) ** 2) / max(2 * (width_a / dense_t) ** 2, 1e-6))
        bump_b = math.exp(-((x - second) ** 2) / max(2 * (width_b / dense_t) ** 2, 1e-6))
        wave = 0.04 * math.sin(2.0 * math.pi * (x * 3.0 + (seed % 11) / 11.0))
        curve.append(max(0.0, min(1.0, 0.08 + 0.70 * bump_a + 0.55 * bump_b + wave)))
    return curve


def build_round0_scaffold(dense_t: int, config: ABRConfig) -> List[int]:
    if dense_t <= 0:
        return []
    k0 = min(max(int(config.k0), 1), dense_t)
    positions = [int(round(x)) for x in _linspace(0, dense_t - 1, k0)]
    if config.max_gap > 0:
        positions.extend(_max_gap_anchors(dense_t, config.max_gap))
    return sorted_unique_in_range(positions, dense_t)


def _linspace(start: int, end: int, count: int) -> List[float]:
    if count <= 1:
        return [float(start)]
    step = (float(end) - float(start)) / float(count - 1)
    return [float(start) + idx * step for idx in range(count)]


def _max_gap_anchors(dense_t: int, max_gap: int) -> List[int]:
    if dense_t <= 0 or max_gap <= 0:
        return []
    return list(range(0, dense_t, max_gap)) + [dense_t - 1]


def build_initial_brackets(curve: Sequence[float], scaffold: Sequence[int], config: ABRConfig) -> List[BracketState]:
    if not curve or not scaffold:
        return []
    states = {pos: state_at_position(curve, pos, config) for pos in scaffold}
    brackets: List[BracketState] = []
    next_id = 1
    sorted_scaffold = sorted_unique_in_range(scaffold, len(curve))

    for left_pos, right_pos in zip(sorted_scaffold[:-1], sorted_scaffold[1:]):
        left_state = states[left_pos]
        right_state = states[right_pos]
        observed_values = [float(curve[left_pos]), float(curve[right_pos])]
        local_uncertainty = max((1.0 - abs(value - 0.5) * 2.0 for value in observed_values), default=0.0)
        local_derivative = abs(observed_values[1] - observed_values[0])
        state_flip = left_state != right_state and "ambiguous" not in {left_state, right_state}
        uncertain = local_uncertainty >= (1.0 - config.uncertainty_band * 2.0)
        derivative_spike = local_derivative >= config.derivative_threshold
        if not (state_flip or uncertain or derivative_spike):
            continue
        kind = "unknown"
        if left_state == "background" and right_state == "action":
            kind = "start"
        elif left_state == "action" and right_state == "background":
            kind = "end"
        bracket = BracketState(
            bracket_id=next_id,
            kind=kind,
            left=int(left_pos),
            right=int(right_pos),
            confidence=min(1.0, 0.35 + local_derivative + 0.20 * int(state_flip)),
            uncertainty=max(0.0, min(1.0, local_uncertainty)),
            state_left=left_state,
            state_right=right_state,
            has_pre_background_witness=left_state == "background",
            has_action_core_witness=left_state == "action" or right_state == "action",
            has_post_background_witness=right_state == "background",
            evidence_source="scaffold_pair",
            score_components={
                "local_derivative": float(local_derivative),
                "local_uncertainty": float(local_uncertainty),
                "state_flip": float(int(state_flip)),
            },
        )
        bracket.priority = score_bracket_priority(bracket, config)
        brackets.append(bracket)
        next_id += 1

    brackets.extend(_multiscale_curve_brackets(curve, config, next_id))
    return _enforce_round0_coverage_guard(_dedupe_overlapping_brackets(brackets), len(curve), config)


def _multiscale_curve_brackets(curve: Sequence[float], config: ABRConfig, start_id: int) -> List[BracketState]:
    if str(config.bracket_policy) != ABR_BRACKET_POLICY_NAME:
        return []
    values = [max(0.0, min(1.0, float(value))) for value in curve]
    if len(values) < 2:
        return []

    brackets: List[BracketState] = []
    next_id = int(start_id)
    composite = _multiscale_composite(values)
    gradients = _gradient_magnitude(composite)

    for left, right, kind in _state_transition_pairs(values, config):
        window = _expanded_window(left, right, len(values), config, source="transition")
        bracket = _make_curve_bracket(
            next_id,
            kind,
            window[0],
            window[1],
            values,
            composite,
            gradients,
            config,
            "multiscale_transition",
            transition_left=left,
            transition_right=right,
        )
        brackets.append(bracket)
        next_id += 1

    peak_floor = max(float(config.action_threshold) * 0.78, 0.42)
    for left, right in _segments_above(composite, peak_floor, max_gap=2):
        window = _expanded_window(left, right, len(values), config, source="peak")
        bracket = _make_curve_bracket(
            next_id,
            "unknown",
            window[0],
            window[1],
            values,
            composite,
            gradients,
            config,
            "multiscale_peak",
        )
        brackets.append(bracket)
        next_id += 1

    gradient_floor = max(float(config.derivative_threshold) * 0.55, 0.10)
    for left, right in _segments_above(gradients, gradient_floor, max_gap=1):
        if right - left > max(2 * max(int(config.max_gap), 1), 12):
            continue
        window = _expanded_window(left, right, len(values), config, source="gradient")
        bracket = _make_curve_bracket(
            next_id,
            "unknown",
            window[0],
            window[1],
            values,
            composite,
            gradients,
            config,
            "multiscale_gradient",
        )
        brackets.append(bracket)
        next_id += 1

    for left, right, score in _adaptive_low_amplitude_activity_segments(values, composite, gradients, config):
        window = _expanded_window(left, right, len(values), config, source="adaptive_low_amplitude")
        bracket = _make_curve_bracket(
            next_id,
            "unknown",
            window[0],
            window[1],
            values,
            composite,
            gradients,
            config,
            "adaptive_low_amplitude_activity",
        )
        bracket.score_components["policy_v2_adaptive_activity"] = float(score)
        bracket.confidence = max(bracket.confidence, min(1.0, 0.25 + 0.65 * float(score)))
        bracket.priority = score_bracket_priority(bracket, config)
        brackets.append(bracket)
        next_id += 1

    for left, right, source, score in _robust_change_extrema_windows(values, composite, gradients, config):
        bracket = _make_curve_bracket(
            next_id,
            "robust_change",
            left,
            right,
            values,
            composite,
            gradients,
            config,
            source,
        )
        bracket.score_components["robust_change_extrema_score"] = float(score)
        bracket.confidence = max(bracket.confidence, min(1.0, 0.28 + 0.62 * float(score)))
        bracket.priority = score_bracket_priority(bracket, config) + 0.35 + 0.30 * float(score)
        brackets.append(bracket)
        next_id += 1

    for left, right, score, anchor_count in _event_train_risk_windows(values, composite, gradients, config):
        bracket = _make_curve_bracket(
            next_id,
            "event_train",
            left,
            right,
            values,
            composite,
            gradients,
            config,
            "event_train_risk_envelope",
        )
        bracket.score_components["event_train_score"] = float(score)
        bracket.score_components["event_train_anchor_count"] = float(anchor_count)
        bracket.confidence = max(bracket.confidence, min(1.0, 0.30 + 0.55 * float(score)))
        bracket.priority = score_bracket_priority(bracket, config) + 0.60 + min(0.60, 0.08 * float(anchor_count))
        brackets.append(bracket)
        next_id += 1

    for left, right, source, score in _risk_gap_probe_windows(brackets, len(values), config):
        bracket = _make_curve_bracket(
            next_id,
            "risk_gap",
            left,
            right,
            values,
            composite,
            gradients,
            config,
            source,
        )
        bracket.score_components["risk_gap_probe_score"] = float(score)
        bracket.confidence = max(bracket.confidence, min(1.0, 0.45 + 0.40 * float(score)))
        bracket.priority = score_bracket_priority(bracket, config) + 0.45 + 0.35 * float(score)
        brackets.append(bracket)
        next_id += 1

    return brackets


def _make_curve_bracket(
    bracket_id: int,
    kind: str,
    left: int,
    right: int,
    values: Sequence[float],
    composite: Sequence[float],
    gradients: Sequence[float],
    config: ABRConfig,
    evidence_source: str,
    transition_left: int | None = None,
    transition_right: int | None = None,
) -> BracketState:
    left = clamp_position(left, len(values))
    right = clamp_position(max(int(right), int(left)), len(values))
    left, right = _clamp_window_width(left, right, len(values), config)
    local_values = values[left : right + 1]
    local_composite = composite[left : right + 1]
    local_gradients = gradients[left : right + 1]
    peak = max(local_composite, default=0.0)
    grad = max(local_gradients, default=0.0)
    uncertainty = max((1.0 - abs(value - 0.5) * 2.0 for value in local_values), default=0.0)
    state_left = state_at_position(values, left)
    state_right = state_at_position(values, right)
    has_action = any(value >= 0.60 for value in local_values) or peak >= 0.60
    bracket = BracketState(
        bracket_id=int(bracket_id),
        kind=kind,
        left=int(left),
        right=int(right),
        confidence=max(0.05, min(1.0, 0.30 + 0.45 * peak + 0.35 * grad)),
        uncertainty=max(0.0, min(1.0, uncertainty)),
        state_left=state_left,
        state_right=state_right,
        has_pre_background_witness=any(value <= 0.35 for value in values[max(0, left - 3) : left + 1]),
        has_action_core_witness=bool(has_action),
        has_post_background_witness=any(value <= 0.35 for value in values[right : min(len(values), right + 4)]),
        evidence_source=evidence_source,
        score_components={
            "policy_v2_peak": float(peak),
            "policy_v2_gradient": float(grad),
            "policy_v2_uncertainty": float(uncertainty),
            "transition_left": float(-1 if transition_left is None else transition_left),
            "transition_right": float(-1 if transition_right is None else transition_right),
        },
    )
    bracket.priority = score_bracket_priority(bracket, config)
    return bracket


def _multiscale_composite(values: Sequence[float]) -> List[float]:
    smooth3 = _moving_average(values, 3)
    smooth7 = _moving_average(values, 7)
    smooth15 = _moving_average(values, 15)
    gradients = _gradient_magnitude(values)
    out = []
    for idx, value in enumerate(values):
        multiscale_peak = max(float(value), smooth3[idx], smooth7[idx], smooth15[idx])
        out.append(max(0.0, min(1.0, 0.72 * multiscale_peak + 0.28 * gradients[idx])))
    return out


def _moving_average(values: Sequence[float], window: int) -> List[float]:
    window = max(int(window), 1)
    radius = window // 2
    out = []
    for idx in range(len(values)):
        left = max(0, idx - radius)
        right = min(len(values), idx + radius + 1)
        out.append(sum(float(v) for v in values[left:right]) / max(right - left, 1))
    return out


def _gradient_magnitude(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    gradients = [0.0] * len(values)
    for idx in range(len(values)):
        prev_value = float(values[max(0, idx - 1)])
        next_value = float(values[min(len(values) - 1, idx + 1)])
        gradients[idx] = abs(next_value - prev_value)
    return gradients


def _adaptive_low_amplitude_activity_segments(
    values: Sequence[float],
    composite: Sequence[float],
    gradients: Sequence[float],
    config: ABRConfig,
) -> List[Tuple[int, int, float]]:
    if len(values) < 3:
        return []

    activity = [max(float(values[idx]), float(composite[idx])) for idx in range(len(values))]
    sorted_activity = sorted(activity)
    low = _percentile(sorted_activity, 0.20)
    median = _percentile(sorted_activity, 0.50)
    high = _percentile(sorted_activity, 0.90)
    top = _percentile(sorted_activity, 0.97)
    dynamic_range = max(top - low, high - low, 0.0)
    if dynamic_range < 0.06:
        return []

    adaptive_floor = max(median + 0.18 * dynamic_range, low + 0.35 * dynamic_range, 0.12)
    adaptive_floor = min(adaptive_floor, max(top - 0.08 * dynamic_range, low + 0.65 * dynamic_range))
    candidate_segments = _segments_above(activity, adaptive_floor, max_gap=1)
    if not candidate_segments:
        return []

    max_short_width = max(8, min(len(values) // 4, 2 * max(int(config.max_gap), 1) + 8))
    min_prominence = max(0.045, 0.18 * dynamic_range)
    scored: List[Tuple[int, int, float]] = []
    for left, right in candidate_segments:
        if right - left + 1 > max_short_width:
            continue
        local_peak = max(activity[left : right + 1], default=0.0)
        shoulder_left = max(0, int(left) - max(3, max(int(config.max_gap), 1) // 4))
        shoulder_right = min(len(activity), int(right) + max(4, max(int(config.max_gap), 1) // 4 + 1))
        shoulder_values = activity[shoulder_left:left] + activity[right + 1 : shoulder_right]
        shoulder_floor = min(shoulder_values, default=low)
        prominence = float(local_peak) - float(shoulder_floor)
        if prominence < min_prominence:
            continue
        local_gradient = max(gradients[max(0, left - 1) : min(len(gradients), right + 2)], default=0.0)
        score = min(1.0, 0.60 * (prominence / max(dynamic_range, 1e-6)) + 0.40 * local_gradient)
        scored.append((int(left), int(right), float(score)))

    scored.sort(key=lambda item: (-item[2], item[0], item[1]))
    max_segments = max(4, min(64, len(values) // max(max(int(config.max_gap), 1) // 2 + 4, 4)))
    return sorted(scored[:max_segments], key=lambda item: (item[0], item[1]))


def _robust_change_extrema_windows(
    values: Sequence[float],
    composite: Sequence[float],
    gradients: Sequence[float],
    config: ABRConfig,
) -> List[Tuple[int, int, str, float]]:
    if len(values) < 5:
        return []

    dense_t = len(values)
    smooth = _moving_average(values, 3)
    sorted_values = sorted(float(value) for value in smooth)
    value_low = _percentile(sorted_values, 0.10)
    value_median = _percentile(sorted_values, 0.50)
    value_high = _percentile(sorted_values, 0.90)
    value_range = max(value_high - value_low, max(sorted_values) - min(sorted_values), 0.0)
    if value_range < 0.018:
        return []

    salience: List[float] = []
    radii = sorted({2, 4, max(2, min(8, dense_t // 32))})
    for idx in range(dense_t):
        local_change = 0.0
        for radius in radii:
            left_values = smooth[max(0, idx - radius) : idx]
            right_values = smooth[idx + 1 : min(dense_t, idx + radius + 1)]
            shoulder_values = left_values + right_values
            if left_values and right_values:
                left_mean = sum(left_values) / float(len(left_values))
                right_mean = sum(right_values) / float(len(right_values))
                local_change = max(local_change, abs(right_mean - left_mean))
            if shoulder_values:
                shoulder_low = min(shoulder_values)
                shoulder_high = max(shoulder_values)
                local_change = max(
                    local_change,
                    abs(float(smooth[idx]) - shoulder_low),
                    abs(shoulder_high - float(smooth[idx])),
                )
        curvature = 0.0
        if 0 < idx < dense_t - 1:
            curvature = abs(float(smooth[idx - 1]) - 2.0 * float(smooth[idx]) + float(smooth[idx + 1]))
        robust_offset = abs(float(smooth[idx]) - value_median)
        salience.append(max(local_change, float(gradients[idx]), curvature, 0.35 * robust_offset))

    sorted_salience = sorted(salience)
    salience_peak = max(sorted_salience, default=0.0)
    if salience_peak < 0.025:
        return []
    salience_median = _percentile(sorted_salience, 0.50)
    salience_mad = _median_abs_deviation(salience, salience_median)
    salience_p90 = _percentile(sorted_salience, 0.90)
    salience_p97 = _percentile(sorted_salience, 0.97)
    floor = max(
        salience_median + max(2.5 * salience_mad, 0.014),
        min(salience_p90 + 0.25 * max(salience_p97 - salience_p90, 0.0), 0.70 * salience_peak),
        min(0.030, 0.35 * salience_peak),
    )
    floor = min(floor, max(0.025, 0.72 * salience_peak))

    bridge_gap = max(2, min(6, dense_t // 48 + max(int(config.max_gap), 0) // 8))
    context = max(2, min(8, int(round(dense_t * 0.025)), max(int(config.max_gap), 1) // 4))
    windows: List[Tuple[int, int, str, float]] = []
    for left, right in _segments_above(salience, floor, max_gap=bridge_gap):
        local_peak = max(salience[left : right + 1], default=0.0)
        if local_peak < floor:
            continue
        score = min(1.0, local_peak / max(salience_peak, floor, 1e-6))
        segment_context = max(0, context - 2)
        windows.append(
            (
                clamp_position(left - segment_context, dense_t),
                clamp_position(right + segment_context, dense_t),
                "robust_local_change_extrema",
                float(score),
            )
        )

    prominence_floor = max(0.025, min(0.10, 0.22 * value_range))
    extrema_radius = max(3, min(8, dense_t // 40 + 2))
    for idx in range(1, dense_t - 1):
        is_peak = smooth[idx] >= smooth[idx - 1] and smooth[idx] >= smooth[idx + 1]
        is_valley = smooth[idx] <= smooth[idx - 1] and smooth[idx] <= smooth[idx + 1]
        if not (is_peak or is_valley):
            continue
        left = max(0, idx - extrema_radius)
        right = min(dense_t, idx + extrema_radius + 1)
        local = smooth[left:right]
        if not local:
            continue
        if is_peak:
            prominence = float(smooth[idx]) - min(local)
        else:
            prominence = max(local) - float(smooth[idx])
        if prominence < prominence_floor or salience[idx] < 0.75 * floor:
            continue
        score = min(1.0, 0.55 * prominence / max(value_range, 1e-6) + 0.45 * salience[idx] / max(salience_peak, 1e-6))
        windows.append(
            (
                clamp_position(idx - context, dense_t),
                clamp_position(idx + context, dense_t),
                "robust_local_extrema_prominence",
                float(score),
            )
        )

    merged = _merge_scored_windows(windows, dense_t, config)
    max_windows = max(4, min(32, dense_t // max(max(int(config.max_gap), 1), 16) + 4))
    merged.sort(key=lambda item: (-item[3], item[0], item[1]))
    return sorted(merged[:max_windows], key=lambda item: (item[0], item[1], item[2]))


def _median_abs_deviation(values: Sequence[float], center: float) -> float:
    deviations = sorted(abs(float(value) - float(center)) for value in values)
    return _percentile(deviations, 0.50)


def _merge_scored_windows(
    windows: Sequence[Tuple[int, int, str, float]],
    dense_t: int,
    config: ABRConfig,
) -> List[Tuple[int, int, str, float]]:
    if not windows:
        return []
    ordered = sorted(
        (
            clamp_position(left, dense_t),
            clamp_position(max(int(right), int(left)), dense_t),
            str(source),
            float(score),
        )
        for left, right, source, score in windows
    )
    merged: List[Tuple[int, int, str, float]] = []
    merge_gap = max(1, min(4, dense_t // 96 + max(int(config.max_gap), 0) // 12))
    max_merge_width = max(4, min(int(math.floor(dense_t * 0.12)), max(int(config.max_gap), 1) + 10))
    for left, right, source, score in ordered:
        if not merged:
            merged.append((left, right, source, score))
            continue
        prev_left, prev_right, prev_source, prev_score = merged[-1]
        merged_width = max(prev_right, right) - min(prev_left, left)
        if left <= prev_right + merge_gap and merged_width <= max_merge_width:
            merged_source = prev_source if prev_score >= score else source
            merged[-1] = (prev_left, max(prev_right, right), merged_source, max(prev_score, score))
            continue
        merged.append((left, right, source, score))

    bounded: List[Tuple[int, int, str, float]] = []
    for left, right, source, score in merged:
        bounded_left, bounded_right = _clamp_window_width(left, right, dense_t, config)
        bounded.append((bounded_left, bounded_right, source, score))
    return bounded


def _event_train_risk_windows(
    values: Sequence[float],
    composite: Sequence[float],
    gradients: Sequence[float],
    config: ABRConfig,
) -> List[Tuple[int, int, float, int]]:
    if len(values) < 8:
        return []

    salience = [
        max(float(values[idx]), float(composite[idx]), float(gradients[idx]) * 1.35)
        for idx in range(len(values))
    ]
    sorted_salience = sorted(salience)
    sorted_gradients = sorted(float(value) for value in gradients)
    low = _percentile(sorted_salience, 0.20)
    high = _percentile(sorted_salience, 0.95)
    dynamic_range = max(high - low, 0.0)
    if dynamic_range < 0.04:
        return []

    peak_floor = max(
        _percentile(sorted_salience, float(config.event_train_peak_quantile)),
        low + 0.40 * dynamic_range,
        0.18,
    )
    gradient_floor = max(
        _percentile(sorted_gradients, float(config.event_train_gradient_quantile)),
        0.08,
    )
    anchors = []
    for idx in range(1, len(values) - 1):
        is_local_peak = salience[idx] >= salience[idx - 1] and salience[idx] >= salience[idx + 1]
        high_salience = salience[idx] >= peak_floor
        high_gradient = float(gradients[idx]) >= gradient_floor and salience[idx] >= low + 0.18 * dynamic_range
        if is_local_peak and (high_salience or high_gradient):
            anchors.append(idx)
    anchors = sorted_unique_in_range(anchors, len(values))
    if len(anchors) < max(int(config.event_train_min_anchors), 2):
        return []

    max_gap = max(int(round(len(values) * float(config.event_train_max_gap_fraction))), 2 * max(int(config.max_gap), 1), 8)
    groups: List[List[int]] = []
    current = [anchors[0]]
    for anchor in anchors[1:]:
        if int(anchor) - int(current[-1]) <= max_gap:
            current.append(anchor)
            continue
        groups.append(current)
        current = [anchor]
    groups.append(current)

    windows: List[Tuple[int, int, float, int]] = []
    for group in groups:
        if len(group) < max(int(config.event_train_min_anchors), 2):
            continue
        windows.extend(_bounded_event_train_group_windows(group, salience, dynamic_range, config, len(values)))

    windows.sort(key=lambda item: (-item[2], item[0], item[1], -item[3]))
    max_windows = max(3, min(24, len(values) // max(max(int(config.max_gap), 1), 12) + 3))
    selected = sorted(windows[:max_windows], key=lambda item: (item[0], item[1]))
    return selected


def _bounded_event_train_group_windows(
    anchors: Sequence[int],
    salience: Sequence[float],
    dynamic_range: float,
    config: ABRConfig,
    dense_t: int,
) -> List[Tuple[int, int, float, int]]:
    max_width = max(1, int(math.floor(dense_t * float(config.first_round_max_bracket_width_fraction))))
    max_width = min(max_width, dense_t)
    context = max(
        int(round(dense_t * float(config.event_train_context_fraction))),
        max(int(config.max_gap), 1),
        3,
    )
    context = min(context, max(max_width // 3, 1))

    chunks: List[List[int]] = []
    current: List[int] = []
    for anchor in anchors:
        proposed = current + [int(anchor)]
        if current and proposed[-1] - proposed[0] + 2 * context + 1 > max_width:
            chunks.append(current)
            current = [int(anchor)]
        else:
            current = proposed
    if current:
        chunks.append(current)

    windows: List[Tuple[int, int, float, int]] = []
    for chunk in chunks:
        if len(chunk) < max(int(config.event_train_min_anchors), 2) and len(anchors) < 3:
            continue
        left = clamp_position(min(chunk) - context, dense_t)
        right = clamp_position(max(chunk) + context, dense_t)
        if right - left + 1 > max_width:
            center = int(round((min(chunk) + max(chunk)) / 2.0))
            half = max_width // 2
            left = clamp_position(center - half, dense_t)
            right = clamp_position(left + max_width - 1, dense_t)
            left = clamp_position(right - max_width + 1, dense_t)
        local_peak = max((float(salience[pos]) for pos in chunk), default=0.0)
        score = min(1.0, 0.45 + 0.10 * len(chunk) + 0.45 * (local_peak / max(dynamic_range, 1e-6)))
        windows.append((int(left), int(right), float(score), int(len(chunk))))
    return windows


def _clamp_window_width(left: int, right: int, dense_t: int, config: ABRConfig) -> Tuple[int, int]:
    if dense_t <= 0:
        return 0, 0
    max_width = max(1, int(math.floor(dense_t * float(config.first_round_max_bracket_width_fraction))))
    if int(right) - int(left) <= max_width:
        return int(left), int(right)
    center = int(round((int(left) + int(right)) / 2.0))
    half = max_width // 2
    new_left = clamp_position(center - half, dense_t)
    new_right = clamp_position(new_left + max_width, dense_t)
    new_left = clamp_position(new_right - max_width, dense_t)
    return int(new_left), int(new_right)


def _risk_gap_probe_windows(
    brackets: Sequence[BracketState],
    dense_t: int,
    config: ABRConfig,
) -> List[Tuple[int, int, str, float]]:
    if dense_t <= 0 or not brackets or int(config.max_gap) <= 0:
        return []
    intervals = _merged_intervals(
        (bracket.left, bracket.right)
        for bracket in brackets
        if bracket.evidence_source
        in {
            "adaptive_low_amplitude_activity",
            "event_train_risk_envelope",
            "multiscale_gradient",
            "multiscale_peak",
            "multiscale_transition",
        }
    )
    if not intervals:
        return []

    windows: List[Tuple[int, int, str, float]] = []
    bridge_limit = max(int(config.max_gap), 4)
    sentinel_stride = max(2 * max(int(config.max_gap), 1), 8)
    sentinel_radius = max(max(int(config.max_gap), 1) // 6, 2)
    edge_radius = max(max(int(config.max_gap), 1) // 2, 4)

    first_left = int(intervals[0][0])
    if first_left > 0:
        right = min(first_left - 1, edge_radius)
        if right >= 0:
            windows.append((0, right, "temporal_edge_risk_guard", 0.75))

    for (prev_left, prev_right), (next_left, next_right) in zip(intervals[:-1], intervals[1:]):
        gap_left = int(prev_right) + 1
        gap_right = int(next_left) - 1
        if gap_right < gap_left:
            continue
        gap_width = gap_right - gap_left + 1
        if gap_width <= bridge_limit:
            windows.append((gap_left, gap_right, "risk_gap_micro_bridge", 0.90))
            continue

        boundary_center = clamp_position(gap_left + max(int(config.max_gap), 1) // 4, dense_t)
        windows.append(
            (
                boundary_center - sentinel_radius,
                boundary_center + sentinel_radius,
                "silent_gap_sentinel",
                0.65,
            )
        )
        center = gap_left + sentinel_stride
        while center <= gap_right:
            windows.append((center - sentinel_radius, center + sentinel_radius, "silent_gap_sentinel", 0.60))
            center += sentinel_stride

    bounded = []
    for left, right, source, score in windows:
        clamped_left = clamp_position(left, dense_t)
        clamped_right = clamp_position(max(int(right), int(left)), dense_t)
        clamped_left, clamped_right = _clamp_window_width(clamped_left, clamped_right, dense_t, config)
        bounded.append((clamped_left, clamped_right, source, float(score)))
    return bounded


def _merged_intervals(intervals: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    ordered = sorted((int(left), int(right)) for left, right in intervals if int(right) >= int(left))
    if not ordered:
        return []
    merged: List[Tuple[int, int]] = [ordered[0]]
    for left, right in ordered[1:]:
        prev_left, prev_right = merged[-1]
        if left <= prev_right + 1:
            merged[-1] = (prev_left, max(prev_right, right))
            continue
        merged.append((left, right))
    return merged


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    idx = min(max(int(round((len(values) - 1) * float(q))), 0), len(values) - 1)
    return float(values[idx])


def _state_transition_pairs(curve: Sequence[float], config: ABRConfig) -> List[Tuple[int, int, str]]:
    transitions: List[Tuple[int, int, str]] = []
    prev_state = state_at_position(curve, 0, config)
    prev_pos = 0
    if prev_state == "ambiguous":
        prev_state = None
        prev_pos = -1
    for pos in range(1, len(curve)):
        state = state_at_position(curve, pos, config)
        if state == "ambiguous":
            continue
        if prev_state is not None and state != prev_state:
            kind = "start" if prev_state == "background" and state == "action" else "end"
            transitions.append((prev_pos, pos, kind))
        prev_state = state
        prev_pos = pos
    return transitions


def _segments_above(values: Sequence[float], threshold: float, max_gap: int) -> List[Tuple[int, int]]:
    segments: List[Tuple[int, int]] = []
    start: int | None = None
    last: int | None = None
    gap = 0
    for idx, value in enumerate(values):
        if float(value) >= float(threshold):
            if start is None:
                start = idx
            last = idx
            gap = 0
            continue
        if start is not None:
            gap += 1
            if gap > max(int(max_gap), 0):
                segments.append((int(start), int(last if last is not None else idx - gap)))
                start = None
                last = None
                gap = 0
    if start is not None:
        segments.append((int(start), int(last if last is not None else len(values) - 1)))
    return [(left, right) for left, right in segments if right >= left]


def _expanded_window(left: int, right: int, dense_t: int, config: ABRConfig, source: str) -> Tuple[int, int]:
    width = max(int(right) - int(left), 1)
    base = max(2, int(round(max(int(config.max_gap), 1) * 0.25)))
    if source == "peak" and width <= max(4, dense_t // 48):
        base = max(base, 3)
    if source == "adaptive_low_amplitude" and width <= max(6, dense_t // 40):
        base = max(base, 2)
    if source == "gradient":
        base = max(1, base - 1)
    return clamp_position(int(left) - base, dense_t), clamp_position(int(right) + base, dense_t)


def _enforce_round0_coverage_guard(
    brackets: Sequence[BracketState], dense_t: int, config: ABRConfig
) -> List[BracketState]:
    if dense_t <= 0:
        return []
    if dense_t <= 8:
        kept = sorted(list(brackets), key=lambda b: (b.left, b.right, -b.priority))
        for idx, bracket in enumerate(kept, start=1):
            bracket.bracket_id = idx
        return kept
    max_fraction = max(0.05, min(float(config.first_round_max_temporal_coverage_fraction), 1.0))
    kept: List[BracketState] = []
    covered: set[int] = set()
    max_positions = max(1, int(math.ceil(dense_t * max_fraction)))
    for bracket in sorted(brackets, key=_round0_guard_rank):
        left = clamp_position(bracket.left, dense_t)
        right = clamp_position(bracket.right, dense_t)
        if right < left:
            continue
        proposed = set(range(left, right + 1))
        marginal = proposed - covered
        if not marginal:
            continue
        if len(covered) + len(marginal) > max_positions:
            continue
        kept.append(bracket)
        covered.update(marginal)
    kept = sorted(kept, key=lambda b: (b.left, b.right, -b.priority))
    for idx, bracket in enumerate(kept, start=1):
        bracket.bracket_id = idx
    return kept


def _covered_fraction(brackets: Sequence[BracketState], dense_t: int) -> float:
    covered = set()
    for bracket in brackets:
        left = clamp_position(bracket.left, dense_t)
        right = clamp_position(bracket.right, dense_t)
        if right >= left:
            covered.update(range(left, right + 1))
    return len(covered) / float(dense_t)


def _round0_guard_rank(bracket: BracketState) -> Tuple[float, float, int, int]:
    width = max(int(bracket.width) + 1, 1)
    risk_density = float(bracket.priority) / math.sqrt(float(width))
    source_bonus = 0.0
    if bracket.evidence_source in {
        "adaptive_low_amplitude_activity",
        "multiscale_gradient",
        "multiscale_transition",
        "robust_local_change_extrema",
        "robust_local_extrema_prominence",
    }:
        source_bonus += 0.15
    if bracket.evidence_source in {"risk_gap_micro_bridge", "silent_gap_sentinel", "temporal_edge_risk_guard"}:
        source_bonus += 0.25
    if bracket.evidence_source == "event_train_risk_envelope":
        source_bonus += 0.45
    if bracket.evidence_source == "scaffold_pair":
        source_bonus -= 0.35
    return (-(risk_density + source_bonus), -float(bracket.priority), int(bracket.left), int(bracket.right))


def _dedupe_overlapping_brackets(brackets: Sequence[BracketState]) -> List[BracketState]:
    if not brackets:
        return []
    ordered = sorted(brackets, key=lambda b: (b.left, b.right, -b.priority))
    kept: List[BracketState] = []
    for bracket in ordered:
        if kept and bracket.left <= kept[-1].right and bracket.kind == kept[-1].kind:
            if bracket.priority > kept[-1].priority:
                kept[-1] = bracket
            continue
        kept.append(bracket)
    for idx, bracket in enumerate(kept, start=1):
        bracket.bracket_id = idx
    return kept


def score_bracket_priority(bracket: BracketState, config: ABRConfig) -> float:
    witness_bonus = 0.35 if two_sided_witness_missing(bracket) else 0.0
    short_action_bonus = 0.25 if bracket.width <= max(config.max_gap, 1) else 0.0
    width_score = min(1.0, bracket.width / max(float(config.max_gap), 1.0))
    return float(width_score + bracket.uncertainty + 0.5 * bracket.confidence + witness_bonus + short_action_bonus)


def two_sided_witness_missing(bracket: BracketState) -> bool:
    if bracket.kind == "start":
        return not (bracket.has_pre_background_witness and bracket.has_action_core_witness)
    if bracket.kind == "end":
        return not (bracket.has_action_core_witness and bracket.has_post_background_witness)
    return not (
        bracket.has_pre_background_witness
        and bracket.has_action_core_witness
        and bracket.has_post_background_witness
    )


def propose_probe_positions(
    bracket: BracketState,
    observed: Iterable[int],
    dense_t: int,
    config: ABRConfig,
    round_id: int,
) -> List[Tuple[int, str]]:
    observed_set = set(int(pos) for pos in observed)
    ratios = (1.0 / 3.0, 2.0 / 3.0) if round_id == 1 else (0.5,)
    proposed: List[Tuple[int, str]] = []
    width = max(bracket.right - bracket.left, 1)
    for ratio in ratios:
        proposed.append((clamp_position(round(bracket.left + ratio * width), dense_t), "bracket_mid"))
    offset = max(int(config.outside_witness_offset), 1)
    if not bracket.has_pre_background_witness:
        proposed.append((clamp_position(bracket.left - offset, dense_t), "pre_background_witness"))
    if not bracket.has_action_core_witness:
        proposed.append((clamp_position((bracket.left + bracket.right) // 2, dense_t), "action_core_witness"))
    if not bracket.has_post_background_witness:
        proposed.append((clamp_position(bracket.right + offset, dense_t), "post_background_witness"))
    deduped = []
    seen = set()
    for pos, role in proposed:
        if pos in observed_set or pos in seen:
            continue
        seen.add(pos)
        deduped.append((pos, role))
    return deduped


def refine_or_split_bracket(
    bracket: BracketState,
    observations: Dict[int, str],
    config: ABRConfig,
    round_id: int,
) -> List[BracketState]:
    local = sorted((int(pos), state) for pos, state in observations.items() if bracket.left <= int(pos) <= bracket.right)
    if len(local) < 2:
        return [_mark_stale(bracket, config, round_id)]

    transitions = _state_transitions(local)
    _update_witness_flags(bracket, observations)
    if len(transitions) == 0:
        stale = _mark_stale(bracket, config, round_id)
        return [stale] if stale.confidence >= config.keep_stale_confidence else []

    if len(transitions) == 1:
        left, right = transitions[0]
        bracket.left = max(bracket.left, left)
        bracket.right = min(bracket.right, right)
        bracket.last_updated_round = round_id
        bracket.status = "resolved" if bracket.width <= config.resolve_width else "narrowed"
        bracket.state_left = observations.get(left, bracket.state_left)
        bracket.state_right = observations.get(right, bracket.state_right)
        bracket.priority = score_bracket_priority(bracket, config)
        return [bracket]

    children: List[BracketState] = []
    for child_idx, (left, right) in enumerate(transitions[: config.max_children_per_bracket], start=1):
        child = BracketState(
            bracket_id=bracket.bracket_id * 100 + child_idx,
            kind="split_child",
            left=left,
            right=right,
            parent_id=bracket.bracket_id,
            round_created=round_id,
            last_updated_round=round_id,
            confidence=max(0.1, bracket.confidence * 0.85),
            uncertainty=bracket.uncertainty,
            state_left=observations.get(left),
            state_right=observations.get(right),
            has_pre_background_witness=bracket.has_pre_background_witness,
            has_action_core_witness=bracket.has_action_core_witness,
            has_post_background_witness=bracket.has_post_background_witness,
            status="active",
        )
        child.priority = score_bracket_priority(child, config)
        children.append(child)
    bracket.status = "split"
    return children


def _state_transitions(local: Sequence[Tuple[int, str]]) -> List[Tuple[int, int]]:
    transitions = []
    prev_pos, prev_state = local[0]
    for pos, state in local[1:]:
        if state == "ambiguous":
            prev_pos, prev_state = pos, state
            continue
        if prev_state != "ambiguous" and state != prev_state:
            transitions.append((prev_pos, pos))
        prev_pos, prev_state = pos, state
    return transitions


def _mark_stale(bracket: BracketState, config: ABRConfig, round_id: int) -> BracketState:
    bracket.status = "stale"
    bracket.last_updated_round = round_id
    bracket.confidence *= config.stale_decay
    bracket.priority = score_bracket_priority(bracket, config)
    return bracket


def _update_witness_flags(bracket: BracketState, observations: Dict[int, str]) -> None:
    for pos, state in observations.items():
        pos = int(pos)
        if state == "background" and pos <= bracket.left:
            bracket.has_pre_background_witness = True
        if state == "action" and bracket.left <= pos <= bracket.right:
            bracket.has_action_core_witness = True
        if state == "background" and pos >= bracket.right:
            bracket.has_post_background_witness = True
