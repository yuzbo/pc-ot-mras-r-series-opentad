from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from .policy import (
    build_initial_brackets,
    build_round0_scaffold,
    deterministic_fallback_scout,
    propose_probe_positions,
    refine_or_split_bracket,
    score_bracket_priority,
    sorted_unique_in_range,
    state_at_position,
)
from .types import (
    ABR_BRACKET_POLICY_NAME,
    ABRConfig,
    ABRCostSummary,
    ABRRoundLedger,
    ABRSelectionResult,
    ABR_ROUTE_LABEL,
    DEFAULT_PROVENANCE,
    BracketState,
)
from .validators import assert_no_forbidden_route_tokens, assert_provenance_clean
from .validators import ABRValidationError


def select_active_bracket_refinement(
    dense_t: int,
    fps: float = 25.0,
    video_id: str = "unknown",
    window_id: str = "window0",
    scout_curve: Optional[Sequence[float]] = None,
    scout_source: str = "unspecified",
    config: Optional[ABRConfig] = None,
) -> ABRSelectionResult:
    cfg = config or ABRConfig()
    dense_t = int(dense_t)
    if dense_t <= 0:
        raise ValueError("dense_t must be positive")
    if cfg.route_label != ABR_ROUTE_LABEL:
        raise ValueError(f"ABR route label mismatch: {cfg.route_label}")
    assert_no_forbidden_route_tokens({"route_label": cfg.route_label, "method": "abr_active_bracket_refinement"})
    provenance = dict(DEFAULT_PROVENANCE)
    assert_provenance_clean(provenance)

    curve, scout_source_detail, fallback_used = _normalize_curve(scout_curve, dense_t, video_id, cfg, scout_source)
    selected_meta: Dict[int, Tuple[int, str, int]] = {}
    round_ledgers: List[ABRRoundLedger] = []
    scout_ms = 0.0
    acquisition_ms = 0.0
    stop_reason = "saturated"

    r0_positions = build_round0_scaffold(dense_t, cfg)
    observed = set(r0_positions)
    for pos in r0_positions:
        selected_meta[pos] = (0, "scaffold", -1)
    scout_ms += len(r0_positions) * cfg.scout_cost_ms_per_position
    brackets = build_initial_brackets(curve, r0_positions, cfg)
    first_round_diagnostics = _first_round_bracket_diagnostics(curve, brackets, cfg, scout_source_detail, fallback_used)
    stop_reason = "no_brackets" if not brackets else "round0_complete"
    round_ledgers.append(
        _make_ledger(
            video_id,
            window_id,
            dense_t,
            fps,
            0,
            r0_positions,
            ["scaffold"] * len(r0_positions),
            [-1] * len(r0_positions),
            "round0_scaffold",
            scout_source_detail,
            len(observed),
            scout_ms,
            cfg,
            stop_reason,
            first_round_diagnostics,
        )
    )

    for round_id, per_round_cap in ((1, cfg.k1_cap), (2, cfg.k2_cap)):
        if round_id == 2 and not cfg.round2_enabled:
            stop_reason = "round2_disabled"
            break
        active = _active_brackets_for_round(brackets, cfg, round_id)
        if not active:
            stop_reason = "no_brackets"
            break
        if len(observed) >= cfg.max_total_k:
            stop_reason = "budget_cap"
            break
        if scout_ms + acquisition_ms >= cfg.deadline_ms:
            stop_reason = "deadline"
            break

        probes, roles, bracket_ids = _select_round_probes(active, observed, dense_t, cfg, round_id, per_round_cap)
        remaining = max(cfg.max_total_k - len(observed), 0)
        probes = probes[:remaining]
        roles = roles[:remaining]
        bracket_ids = bracket_ids[:remaining]
        if not probes:
            stop_reason = "saturated"
            break

        for pos, role, bracket_id in zip(probes, roles, bracket_ids):
            observed.add(pos)
            selected_meta[pos] = (round_id, role, bracket_id)
        scout_ms += len(probes) * cfg.scout_cost_ms_per_position
        acquisition_ms += len(probes) * cfg.acquisition_cost_ms_per_position
        observations = {pos: state_at_position(curve, pos, cfg) for pos in observed}
        brackets = _refine_brackets(brackets, observations, cfg, round_id)
        stop_reason = _round_stop_reason(brackets, cfg, round_id)
        round_ledgers.append(
            _make_ledger(
                video_id,
                window_id,
                dense_t,
                fps,
                round_id,
                probes,
                roles,
                bracket_ids,
                f"round{round_id}_probe",
                scout_source_detail,
                len(observed),
                scout_ms,
                cfg,
                stop_reason,
                {},
            )
        )
        if scout_ms + acquisition_ms >= cfg.deadline_ms:
            stop_reason = "deadline"
            break

    selected_positions = sorted_unique_in_range(observed, dense_t)
    selected_rounds = [selected_meta[pos][0] for pos in selected_positions]
    selected_roles = [selected_meta[pos][1] for pos in selected_positions]
    selected_bracket_ids = [selected_meta[pos][2] for pos in selected_positions]
    cost = ABRCostSummary(
        rounds_used=len(round_ledgers),
        selected_k=len(selected_positions),
        mean_selected_fraction=len(selected_positions) / float(dense_t),
        scout_ms=scout_ms,
        acquisition_wait_ms=acquisition_ms,
        detector_forward_count=1,
        deadline_ms=float(cfg.deadline_ms),
        total_latency_proxy_ms=scout_ms + acquisition_ms,
        stop_reason=stop_reason,
    )
    return ABRSelectionResult(
        route_label=ABR_ROUTE_LABEL,
        selected_positions=selected_positions,
        selected_rounds=selected_rounds,
        selected_roles=selected_roles,
        selected_bracket_ids=selected_bracket_ids,
        valid_k=len(selected_positions),
        dense_T=dense_t,
        fps=float(fps),
        round_ledgers=round_ledgers,
        brackets=brackets,
        cost=cost,
        provenance=provenance,
        scout_source=scout_source_detail,
        diagnostic_fallback_used=fallback_used,
        config=cfg,
    )


def _normalize_curve(
    scout_curve: Optional[Sequence[float]],
    dense_t: int,
    video_id: str,
    config: ABRConfig,
    scout_source: str,
) -> Tuple[List[float], str, bool]:
    if scout_curve is None:
        if not _diagnostic_fallback_allowed(config):
            raise ABRValidationError(
                "LOCKED: ABR requires deploy-visible scout/probe observations; "
                "diagnostic fallback is disabled for this config"
            )
        return deterministic_fallback_scout(dense_t, video_id), f"diagnostic_fallback:{config.fallback_stage}", True
    curve = [float(v) for v in scout_curve]
    if len(curve) == dense_t:
        return [max(0.0, min(1.0, value)) for value in curve], str(scout_source), False
    if len(curve) == 0:
        if not _diagnostic_fallback_allowed(config):
            raise ABRValidationError("LOCKED: empty deploy-visible scout curve is not allowed")
        return deterministic_fallback_scout(dense_t, video_id), f"diagnostic_fallback:{config.fallback_stage}", True
    resized = []
    for idx in range(dense_t):
        src = round(idx * (len(curve) - 1) / max(dense_t - 1, 1))
        resized.append(max(0.0, min(1.0, curve[int(src)])))
    return resized, f"{scout_source}:resized_{len(curve)}_to_{dense_t}", False


def _diagnostic_fallback_allowed(config: ABRConfig) -> bool:
    if not bool(config.allow_diagnostic_fallback_scout):
        return False
    return str(config.fallback_stage).upper() in {"PRECHECK_ONLY", "MOCK_PRECHECK_ONLY", "DIAGNOSTIC_ONLY"}


def _active_brackets_for_round(brackets: Sequence[BracketState], config: ABRConfig, round_id: int) -> List[BracketState]:
    active = [b for b in brackets if b.status in {"active", "narrowed", "stale"}]
    if round_id == 2:
        active = [b for b in active if b.width >= config.round2_min_width and b.priority >= 0.5]
    for bracket in active:
        bracket.priority = score_bracket_priority(bracket, config)
    return sorted(active, key=lambda b: (-b.priority, b.left, b.right))


def _select_round_probes(
    brackets: Sequence[BracketState],
    observed: set[int],
    dense_t: int,
    config: ABRConfig,
    round_id: int,
    per_round_cap: int,
) -> Tuple[List[int], List[str], List[int]]:
    probes: List[int] = []
    roles: List[str] = []
    bracket_ids: List[int] = []
    planned = set(observed)
    for bracket in brackets:
        for pos, role in propose_probe_positions(bracket, planned, dense_t, config, round_id):
            if len(probes) >= max(int(per_round_cap), 0):
                return probes, roles, bracket_ids
            if pos in planned:
                continue
            planned.add(pos)
            probes.append(pos)
            roles.append(role)
            bracket_ids.append(bracket.bracket_id)
    return probes, roles, bracket_ids


def _refine_brackets(
    brackets: Sequence[BracketState],
    observations: Dict[int, str],
    config: ABRConfig,
    round_id: int,
) -> List[BracketState]:
    updated: List[BracketState] = []
    for bracket in brackets:
        updated.extend(refine_or_split_bracket(bracket, observations, config, round_id))
    return sorted(updated, key=lambda b: (b.left, b.right, b.bracket_id))


def _round_stop_reason(brackets: Sequence[BracketState], config: ABRConfig, round_id: int) -> str:
    if not brackets:
        return "no_brackets"
    unresolved = [b for b in brackets if b.status not in {"resolved", "split"} and b.width > config.resolve_width]
    if not unresolved:
        return "saturated"
    if round_id >= 2:
        return "round2_bound"
    return "active"


def _first_round_bracket_diagnostics(
    curve: Sequence[float],
    brackets: Sequence[BracketState],
    config: ABRConfig,
    scout_source: str,
    diagnostic_fallback_used: bool,
) -> Dict[str, object]:
    dense_t = len(curve)
    transitions = _dense_scout_transitions(curve, config)
    covered_pairs = []
    missed_pairs = []
    covered_endpoints = 0
    for left, right, kind in transitions:
        left_covered = any(bracket.left <= left <= bracket.right for bracket in brackets)
        right_covered = any(bracket.left <= right <= bracket.right for bracket in brackets)
        covered_endpoints += int(left_covered) + int(right_covered)
        item = {"left": int(left), "right": int(right), "kind": kind}
        if left_covered and right_covered:
            covered_pairs.append(item)
        else:
            missed_pairs.append(item)

    transition_count = len(transitions)
    bracketed_count = len(covered_pairs)
    endpoint_total = max(2 * transition_count, 1)
    union_coverage = _bracket_union_coverage(brackets, dense_t)
    source_counts: Dict[str, int] = {}
    for bracket in brackets:
        source = str(getattr(bracket, "evidence_source", "unspecified"))
        source_counts[source] = source_counts.get(source, 0) + 1
    return {
        "scope": "first_round_bracket_recall_from_deploy_visible_scout",
        "scout_source": str(scout_source),
        "diagnostic_fallback_used": bool(diagnostic_fallback_used),
        "bracket_policy": str(config.bracket_policy or ABR_BRACKET_POLICY_NAME),
        "bracket_policy_inputs": {
            "deploy_visible_scout_curve": not bool(diagnostic_fallback_used),
            "uses_gt": False,
            "uses_teacher": False,
            "uses_prediction_cache": False,
            "uses_detector_feedback": False,
        },
        "bracket_policy_mechanisms": [
            "scaffold_pair_baseline",
            "multiscale_peak_brackets",
            "full_curve_transition_brackets",
            "gradient_spike_brackets",
            "uncertainty_widening",
            "short_action_boundary_protection",
            "max_gap_span_expansion",
            "first_round_temporal_coverage_guard",
        ],
        "bracket_source_counts": dict(sorted(source_counts.items())),
        "first_round_max_temporal_coverage_fraction": float(config.first_round_max_temporal_coverage_fraction),
        "dense_T": int(dense_t),
        "round_id": 0,
        "transition_count": int(transition_count),
        "bracket_count": int(len(brackets)),
        "bracketed_transition_count": int(bracketed_count),
        "missed_transition_count": int(len(missed_pairs)),
        "first_round_bracket_recall": float(1.0 if transition_count == 0 else bracketed_count / transition_count),
        "first_round_transition_coverage": float(1.0 if transition_count == 0 else covered_endpoints / endpoint_total),
        "first_round_temporal_coverage_fraction": float(union_coverage),
        "missed_transitions": missed_pairs,
        "covered_transitions": covered_pairs[:16],
    }


def _dense_scout_transitions(curve: Sequence[float], config: ABRConfig) -> List[Tuple[int, int, str]]:
    transitions: List[Tuple[int, int, str]] = []
    if len(curve) < 2:
        return transitions
    prev_state: Optional[str] = state_at_position(curve, 0, config)
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


def _bracket_union_coverage(brackets: Sequence[BracketState], dense_t: int) -> float:
    if dense_t <= 0 or not brackets:
        return 0.0
    covered = set()
    for bracket in brackets:
        left = max(0, int(bracket.left))
        right = min(int(bracket.right), dense_t - 1)
        if right >= left:
            covered.update(range(left, right + 1))
    return len(covered) / float(dense_t)


def _make_ledger(
    video_id: str,
    window_id: str,
    dense_t: int,
    fps: float,
    round_id: int,
    positions: Sequence[int],
    roles: Sequence[str],
    bracket_ids: Sequence[int],
    source: str,
    source_detail: str,
    cumulative_k: int,
    cumulative_scout_ms: float,
    config: ABRConfig,
    stop_reason: str,
    diagnostics: Dict[str, object] | None = None,
) -> ABRRoundLedger:
    return ABRRoundLedger(
        video_id=video_id,
        window_id=window_id,
        dense_T=int(dense_t),
        fps=float(fps),
        round_id=int(round_id),
        selected_positions=[int(pos) for pos in positions],
        selected_roles=[str(role) for role in roles],
        selected_bracket_ids=[int(bracket_id) if bracket_id is not None else None for bracket_id in bracket_ids],
        selected_source=[source for _ in positions],
        selected_source_detail=[source_detail for _ in positions],
        selected_cost_ms=[float(config.acquisition_cost_ms_per_position) for _ in positions],
        cumulative_k=int(cumulative_k),
        cumulative_scout_ms=float(cumulative_scout_ms),
        deadline_ms=float(config.deadline_ms),
        stop_reason=stop_reason,
        diagnostics=dict(diagnostics or {}),
        provenance=dict(DEFAULT_PROVENANCE),
    )
