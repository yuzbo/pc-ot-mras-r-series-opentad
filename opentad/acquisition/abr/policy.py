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
    kept = list(sorted(brackets, key=lambda b: (-b.priority, b.left, b.right)))
    while kept and _covered_fraction(kept, dense_t) > max_fraction:
        kept.pop()
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
