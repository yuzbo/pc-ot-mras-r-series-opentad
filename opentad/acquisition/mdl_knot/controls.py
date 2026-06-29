from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .objective import mdl_objective
from .selector import build_ledger_from_positions, greedy_mdl_knot_select
from .types import MDLKnotConfig, KnotLedger, ScoutCurve


def same_k_uniform_positions(dense_t: int, k: int) -> List[int]:
    if dense_t <= 0:
        raise ValueError("dense_t must be positive")
    k = int(max(k, 1))
    if k >= dense_t:
        return list(range(dense_t))
    raw = np.rint(np.linspace(0, dense_t - 1, num=k)).astype(np.int64).tolist()
    selected = sorted(set(int(v) for v in raw))
    if len(selected) < k:
        for pos in range(dense_t):
            if pos not in selected:
                selected.append(pos)
                if len(selected) == k:
                    break
    return sorted(selected[:k])


def _random_same_k_positions(dense_t: int, k: int, seed: int) -> List[int]:
    if k >= dense_t:
        return list(range(dense_t))
    rng = np.random.default_rng(seed)
    middle_count = max(k - 2, 0)
    middle = rng.choice(np.arange(1, dense_t - 1), size=middle_count, replace=False).tolist() if middle_count else []
    return sorted(set([0, dense_t - 1] + [int(v) for v in middle]))


def _resize_positions_to_k(curve: ScoutCurve, positions: Sequence[int], k: int, cfg: MDLKnotConfig) -> List[int]:
    selected = sorted(set(int(v) for v in positions if 0 <= int(v) < curve.dense_t))
    selected = sorted(set([0, curve.dense_t - 1] + selected))
    if len(selected) > k:
        keep = set([0, curve.dense_t - 1])
        costs = []
        for pos in selected:
            if pos in keep:
                continue
            trial = [p for p in selected if p != pos]
            costs.append((mdl_objective(curve, trial, cfg).total_cost, pos))
        for _, pos in sorted(costs):
            if len(keep) >= k:
                break
            keep.add(pos)
        selected = sorted(keep)
    while len(selected) < min(k, curve.dense_t):
        gaps = [(right - left, left, right) for left, right in zip(selected[:-1], selected[1:]) if right - left > 1]
        if not gaps:
            break
        _, left, right = max(gaps)
        selected.append(int(round((left + right) / 2.0)))
        selected = sorted(set(selected))
    return selected


def generate_matched_controls(curve: ScoutCurve, ledger: KnotLedger, seed: int = 0) -> Dict[str, KnotLedger]:
    k = int(ledger.valid_k)
    cfg = MDLKnotConfig(route_label=ledger.route_label, max_k=max(k, 2), min_k=min(max(2, k), max(k, 2)))
    controls: Dict[str, KnotLedger] = {}

    uniform = same_k_uniform_positions(curve.dense_t, k)
    controls["per_video_same_k_uniform"] = build_ledger_from_positions(
        curve, uniform, cfg, control_name="per_video_same_k_uniform"
    )
    controls["mean_k_exact_uniform"] = build_ledger_from_positions(
        curve, uniform, cfg, control_name="mean_k_exact_uniform"
    )
    controls["random_same_k"] = build_ledger_from_positions(
        curve, _random_same_k_positions(curve.dense_t, k, seed), cfg, control_name="random_same_k"
    )
    scaffold = same_k_uniform_positions(curve.dense_t, min(max(4, cfg.min_anchor_k), k))
    controls["scaffold_only"] = build_ledger_from_positions(curve, scaffold, cfg, control_name="scaffold_only")

    mdl_only_cfg = MDLKnotConfig(
        route_label=ledger.route_label,
        min_k=cfg.min_k,
        max_k=k,
        min_anchor_k=cfg.min_anchor_k,
        lambda_gap=0.0,
        lambda_duration=0.0,
        lambda_transition=0.0,
        bonus_transition_guard=0.0,
        bonus_gap_guard=0.0,
        bonus_short_risk_guard=0.0,
    )
    mdl_only = greedy_mdl_knot_select(curve, mdl_only_cfg, control_name="mdl_only")
    mdl_only_positions = _resize_positions_to_k(curve, mdl_only.selected_positions, k, cfg)
    controls["mdl_only"] = build_ledger_from_positions(curve, mdl_only_positions, cfg, control_name="mdl_only")
    controls["mdl_plus_transition_gap_duration"] = build_ledger_from_positions(
        curve,
        ledger.selected_positions,
        cfg,
        roles={pos: role for pos, role in zip(ledger.selected_positions, ledger.selected_roles)},
        control_name="mdl_plus_transition_gap_duration",
    )
    return controls

