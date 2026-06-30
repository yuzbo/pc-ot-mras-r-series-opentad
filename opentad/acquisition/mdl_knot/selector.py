from __future__ import annotations

import time
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .objective import estimate_islands, estimate_transition_bands, mdl_objective, piecewise_linear_reconstruct
from .types import MDL_KNOT_ROUTE_LABEL, MDLKnotConfig, KnotLedger, ScoutCurve
from .validators import validate_knot_ledger, validate_no_forbidden_sources


ProfileCallback = Callable[[str, float, dict | None], None]


def _profile_elapsed(callback: ProfileCallback | None, stage: str, start: float, extra: dict | None = None) -> None:
    if callback is not None:
        callback(stage, time.perf_counter() - start, extra)


def _unique_sorted(values: Iterable[int], dense_t: int) -> List[int]:
    return sorted({int(v) for v in values if 0 <= int(v) < dense_t})


def _uniform_anchor_positions(dense_t: int, k: int) -> List[int]:
    k = int(max(k, 2))
    if k >= dense_t:
        return list(range(dense_t))
    return _unique_sorted(np.rint(np.linspace(0, dense_t - 1, num=k)).astype(np.int64).tolist(), dense_t)


def _max_gap_midpoints(selected: Sequence[int]) -> List[int]:
    out = []
    for left, right in zip(selected[:-1], selected[1:]):
        if right - left > 1:
            out.append(int(round((left + right) / 2.0)))
    out.sort(key=lambda p: min(abs(p - s) for s in selected), reverse=True)
    return out


def _role_bonus(role: str, cfg: MDLKnotConfig) -> float:
    if role == "transition_guard":
        return cfg.bonus_transition_guard
    if role == "gap_guard":
        return cfg.bonus_gap_guard
    if role == "short_risk_guard":
        return cfg.bonus_short_risk_guard
    return 0.0


def _candidate_pool(curve: ScoutCurve, selected: Sequence[int], cfg: MDLKnotConfig) -> List[Tuple[int, str]]:
    dense_t = curve.dense_t
    selected_set = set(selected)
    pairs: List[Tuple[int, str]] = []

    def add(pos: int, role: str) -> None:
        pos = int(pos)
        if 0 <= pos < dense_t and pos not in selected_set:
            pairs.append((pos, role))

    for pos in _uniform_anchor_positions(dense_t, max(cfg.min_anchor_k, cfg.min_k)):
        add(pos, "scaffold_anchor")

    matrix = curve.as_matrix()
    recon = piecewise_linear_reconstruct(matrix, selected)
    residual = ((matrix - recon) ** 2).mean(axis=1)
    for pos in np.argsort(residual)[-cfg.residual_candidate_count :]:
        add(int(pos), "mdl_knot")

    signal = np.maximum.reduce(
        [
            np.asarray(curve.temporal_change, dtype=np.float64),
            np.asarray(curve.uncertainty, dtype=np.float64),
            np.abs(np.gradient(np.asarray(curve.p_action, dtype=np.float64))),
        ]
    )
    for pos in np.argsort(signal)[-cfg.signal_candidate_count :]:
        role = "transition_guard" if curve.temporal_change[int(pos)] >= cfg.transition_threshold else "stable_span_knot"
        add(int(pos), role)

    for pos in _max_gap_midpoints(selected):
        add(pos, "gap_guard")

    for band in estimate_transition_bands(curve, cfg):
        add(int(band["center"]), "transition_guard")
        add(int(band["start"]), "transition_guard")
        add(int(band["end"]), "transition_guard")

    for island in estimate_islands(curve, cfg):
        if island["short_risk"]:
            add(int(island["center"]), "short_risk_guard")
            add(int(island["start"]), "short_risk_guard")
            add(int(island["end"]), "short_risk_guard")
            add(int(island["peak"]), "short_risk_guard")

    best_role: Dict[int, str] = {}
    role_priority = {
        "short_risk_guard": 5,
        "transition_guard": 4,
        "gap_guard": 3,
        "mdl_knot": 2,
        "stable_span_knot": 1,
        "scaffold_anchor": 0,
    }
    for pos, role in pairs:
        if pos not in best_role or role_priority[role] > role_priority[best_role[pos]]:
            best_role[pos] = role

    scored = []
    for pos, role in best_role.items():
        local_score = float(residual[pos] + signal[pos] + _role_bonus(role, cfg))
        scored.append((local_score, pos, role))
    scored.sort(reverse=True)
    return [(pos, role) for _, pos, role in scored[: cfg.candidate_limit]]


def _risk_uncovered(selected: Sequence[int], curve: ScoutCurve, cfg: MDLKnotConfig) -> bool:
    selected_arr = np.asarray(selected, dtype=np.int64)
    for band in estimate_transition_bands(curve, cfg):
        if selected_arr.size == 0 or np.abs(selected_arr - int(band["center"])).min() > cfg.transition_guard_radius:
            return True
    for island in estimate_islands(curve, cfg):
        if not island["short_risk"]:
            continue
        count = sum(1 for pos in selected if island["start"] <= pos <= island["end"])
        if count < cfg.short_island_min_knots:
            return True
    return False


def _budget_bin(valid_k: int, dense_t: int) -> str:
    ratio = float(valid_k) / max(float(dense_t), 1.0)
    if ratio < 0.30:
        return "low_complexity"
    if ratio < 0.55:
        return "mid_complexity"
    return "high_complexity"


def _build_ledger(
    curve: ScoutCurve,
    selected: Sequence[int],
    roles: Dict[int, str],
    history: List[Dict[str, object]],
    stop_reason: str,
    cfg: MDLKnotConfig,
    video_id: str,
    window_id: int,
    control_name: str = "mdl_plus_transition_gap_duration",
) -> KnotLedger:
    selected = _unique_sorted(selected, curve.dense_t)
    terms = mdl_objective(curve, selected, cfg)
    selected_roles = [roles.get(pos, "mdl_knot") for pos in selected]
    selected_scores = [float(curve.p_action[pos]) for pos in selected]
    provenance = dict(curve.provenance)
    validate_no_forbidden_sources(provenance)
    ledger = KnotLedger(
        route_label=MDL_KNOT_ROUTE_LABEL,
        video_id=video_id,
        window_id=int(window_id),
        dense_t=curve.dense_t,
        selected_positions=selected,
        selected_roles=selected_roles,
        selected_scores=selected_scores,
        target_k=int(cfg.max_k),
        valid_k=len(selected),
        stop_reason=stop_reason,
        mean_reconstruction_error=terms.mean_reconstruction_error,
        weighted_reconstruction_error=terms.weighted_reconstruction_error,
        max_gap=terms.max_gap,
        gap_p95=terms.gap_p95,
        estimated_islands=estimate_islands(curve, cfg),
        transition_bands=estimate_transition_bands(curve, cfg),
        budget_bin=_budget_bin(len(selected), curve.dense_t),
        objective_terms=terms.to_dict(),
        provenance=provenance,
        position_unit="original_dense_time_index",
        control_name=control_name,
        selection_history=history,
    )
    validate_knot_ledger(ledger)
    return ledger


def greedy_mdl_knot_select(
    curve: ScoutCurve,
    cfg: MDLKnotConfig,
    video_id: str = "synthetic",
    window_id: int = 0,
    control_name: str = "mdl_plus_transition_gap_duration",
    profile_callback: ProfileCallback | None = None,
) -> KnotLedger:
    if curve.dense_t < 2:
        raise ValueError("MDL-Knot requires at least two dense cells")
    start = time.perf_counter()
    initial_k = min(max(cfg.min_k, cfg.min_anchor_k, 2), curve.dense_t, cfg.max_k)
    selected = set(_uniform_anchor_positions(curve.dense_t, initial_k))
    selected.add(0)
    selected.add(curve.dense_t - 1)
    roles: Dict[int, str] = {0: "endpoint_anchor", curve.dense_t - 1: "endpoint_anchor"}
    for pos in selected:
        roles.setdefault(pos, "scaffold_anchor")
    _profile_elapsed(profile_callback, "selector_initialization", start, {"dense_T": int(curve.dense_t)})

    history: List[Dict[str, object]] = []
    stop_reason = "no_positive_gain"

    while True:
        current = sorted(selected)
        start = time.perf_counter()
        terms = mdl_objective(curve, current, cfg)
        _profile_elapsed(profile_callback, "selector_objective_loop", start, {"selected_len": int(len(current))})
        start = time.perf_counter()
        safe_by_residual = terms.weighted_reconstruction_error <= cfg.target_weighted_error
        safe_by_gap = terms.max_gap <= cfg.max_gap
        risk_uncovered = _risk_uncovered(current, curve, cfg)
        _profile_elapsed(
            profile_callback,
            "gap_guard",
            start,
            {"max_gap": int(terms.max_gap), "risk_uncovered": bool(risk_uncovered)},
        )
        if len(selected) >= cfg.min_k and safe_by_residual and safe_by_gap and not risk_uncovered:
            stop_reason = "residual_and_gap_safe"
            break
        if len(selected) >= min(cfg.max_k, curve.dense_t):
            stop_reason = "cap_reached"
            break

        start = time.perf_counter()
        candidates = _candidate_pool(curve, current, cfg)
        _profile_elapsed(profile_callback, "selector_candidate_pool", start, {"candidate_count": int(len(candidates))})
        best_gain = -np.inf
        best_pos = None
        best_role = None
        for pos, role in candidates:
            if pos in selected:
                continue
            trial = sorted(selected | {pos})
            start = time.perf_counter()
            trial_terms = mdl_objective(curve, trial, cfg)
            _profile_elapsed(profile_callback, "selector_objective_loop", start, {"selected_len": int(len(trial))})
            gain = terms.total_cost - trial_terms.total_cost + _role_bonus(role, cfg)
            if gain > best_gain:
                best_gain = float(gain)
                best_pos = int(pos)
                best_role = role

        if best_pos is None:
            stop_reason = "no_positive_gain"
            break
        start = time.perf_counter()
        risk_uncovered = _risk_uncovered(current, curve, cfg)
        _profile_elapsed(profile_callback, "gap_guard", start, {"risk_uncovered": bool(risk_uncovered)})
        if best_gain < cfg.min_marginal_gain and safe_by_gap and not risk_uncovered:
            stop_reason = "marginal_gain_low"
            break

        selected.add(best_pos)
        roles[best_pos] = best_role or "mdl_knot"
        history.append({"t": best_pos, "gain": float(best_gain), "role": roles[best_pos]})

    start = time.perf_counter()
    ledger = _build_ledger(curve, sorted(selected), roles, history, stop_reason, cfg, video_id, window_id, control_name)
    _profile_elapsed(profile_callback, "metadata_build", start, {"valid_k": int(ledger.valid_k)})
    return ledger


def build_ledger_from_positions(
    curve: ScoutCurve,
    positions: Sequence[int],
    cfg: MDLKnotConfig,
    roles: Dict[int, str] | None = None,
    control_name: str = "control",
    video_id: str = "synthetic",
    window_id: int = 0,
) -> KnotLedger:
    positions = _unique_sorted(positions, curve.dense_t)
    role_map = {pos: "mdl_knot" for pos in positions}
    role_map[positions[0]] = "endpoint_anchor"
    role_map[positions[-1]] = "endpoint_anchor"
    if roles:
        role_map.update({int(k): v for k, v in roles.items()})
    return _build_ledger(
        curve,
        positions,
        role_map,
        history=[],
        stop_reason="control_fixed_k",
        cfg=cfg,
        video_id=video_id,
        window_id=window_id,
        control_name=control_name,
    )
