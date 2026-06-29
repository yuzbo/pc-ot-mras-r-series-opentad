from __future__ import annotations

import hashlib
import math
from typing import Dict, Iterable, List, Sequence, Tuple

from .types import ABRConfig, BracketState


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
        )
        bracket.priority = score_bracket_priority(bracket, config)
        brackets.append(bracket)
        next_id += 1

    return _dedupe_overlapping_brackets(brackets)


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
