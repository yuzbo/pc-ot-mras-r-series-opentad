from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np

from .types import MDLKnotConfig, MDLObjectiveTerms, ScoutCurve


def _clean_positions(selected_positions: Iterable[int], dense_t: int) -> List[int]:
    pos = sorted({int(v) for v in selected_positions})
    if not pos:
        raise ValueError("selected_positions must not be empty")
    if pos[0] < 0 or pos[-1] >= dense_t:
        raise ValueError("selected_positions are out of range")
    return pos


def piecewise_linear_reconstruct(curve_matrix: np.ndarray, selected_positions: Sequence[int]) -> np.ndarray:
    matrix = np.asarray(curve_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("curve_matrix must be [T,D]")
    selected = _clean_positions(selected_positions, matrix.shape[0])
    x_full = np.arange(matrix.shape[0], dtype=np.float64)
    x_sel = np.asarray(selected, dtype=np.float64)
    output = np.zeros_like(matrix, dtype=np.float64)
    for dim in range(matrix.shape[1]):
        output[:, dim] = np.interp(x_full, x_sel, matrix[selected, dim])
    return output


def estimate_islands(curve: ScoutCurve, cfg: MDLKnotConfig) -> List[dict]:
    p = np.asarray(curve.p_action, dtype=np.float64)
    active = p >= float(cfg.island_threshold)
    islands = []
    start = None
    for idx, flag in enumerate(active.tolist() + [False]):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            end = idx - 1
            width = end - start + 1
            peak_idx = int(start + np.argmax(p[start : end + 1]))
            islands.append(
                {
                    "start": int(start),
                    "end": int(end),
                    "center": int(round((start + end) / 2.0)),
                    "peak": peak_idx,
                    "width": int(width),
                    "short_risk": bool(width <= cfg.short_island_max_width),
                }
            )
            start = None
    return islands


def estimate_transition_bands(curve: ScoutCurve, cfg: MDLKnotConfig) -> List[dict]:
    p = np.asarray(curve.p_action, dtype=np.float64)
    change = np.asarray(curve.temporal_change, dtype=np.float64)
    uncertainty = np.asarray(curve.uncertainty, dtype=np.float64)
    state_flip = np.abs(np.diff((p >= cfg.island_threshold).astype(np.float64), prepend=float(p[0] >= cfg.island_threshold)))
    score = np.maximum.reduce([change, state_flip, 0.5 * uncertainty])
    candidate = np.where(score >= cfg.transition_threshold)[0]
    if candidate.size == 0:
        candidate = np.argsort(score)[-min(2, len(score)) :]
    bands = []
    used = np.zeros(len(score), dtype=bool)
    radius = max(int(cfg.transition_guard_radius), 1)
    for idx in candidate[np.argsort(score[candidate])[::-1]]:
        if used[idx]:
            continue
        left = max(0, int(idx) - radius)
        right = min(len(score) - 1, int(idx) + radius)
        used[left : right + 1] = True
        bands.append({"start": left, "end": right, "center": int(idx), "score": float(score[idx])})
    bands.sort(key=lambda item: item["center"])
    return bands


def deploy_weights(curve: ScoutCurve, cfg: MDLKnotConfig) -> np.ndarray:
    p = np.asarray(curve.p_action, dtype=np.float64)
    grad = np.abs(np.gradient(p))
    if grad.max() > 1.0e-12:
        grad = grad / grad.max()
    uncertainty = np.asarray(curve.uncertainty, dtype=np.float64)
    change = np.asarray(curve.temporal_change, dtype=np.float64)
    short_risk = np.zeros(curve.dense_t, dtype=np.float64)
    for island in estimate_islands(curve, cfg):
        if island["short_risk"]:
            short_risk[island["start"] : island["end"] + 1] = 1.0
    return (
        1.0
        + cfg.weight_gradient * grad
        + cfg.weight_uncertainty * uncertainty
        + cfg.weight_change * change
        + cfg.weight_short * short_risk
    )


def _gap_stats(selected: Sequence[int]) -> tuple[int, float, np.ndarray]:
    if len(selected) <= 1:
        return 0, 0.0, np.zeros((0,), dtype=np.float64)
    gaps = np.diff(np.asarray(selected, dtype=np.int64)).astype(np.float64)
    return int(gaps.max()), float(np.percentile(gaps, 95)), gaps


def _duration_risk(selected: Sequence[int], islands: List[dict], cfg: MDLKnotConfig) -> float:
    selected_set = set(int(v) for v in selected)
    risk = 0.0
    for island in islands:
        in_island = [p for p in selected_set if island["start"] <= p <= island["end"]]
        required = cfg.short_island_min_knots if island["short_risk"] else 1
        if len(in_island) < required:
            risk += float(required - len(in_island)) * (2.0 if island["short_risk"] else 1.0)
    return cfg.lambda_duration * risk


def _transition_risk(selected: Sequence[int], bands: List[dict], cfg: MDLKnotConfig) -> float:
    selected_arr = np.asarray(selected, dtype=np.int64)
    if selected_arr.size == 0:
        return cfg.lambda_transition * len(bands)
    risk = 0.0
    radius = max(int(cfg.transition_guard_radius), 1)
    for band in bands:
        center = int(band["center"])
        dist = np.abs(selected_arr - center).min()
        if dist > radius:
            risk += float(dist - radius) / max(radius, 1) * float(band.get("score", 1.0))
    return cfg.lambda_transition * risk


def mdl_objective(curve: ScoutCurve, selected_positions: Sequence[int], cfg: MDLKnotConfig) -> MDLObjectiveTerms:
    selected = _clean_positions(selected_positions, curve.dense_t)
    matrix = curve.as_matrix()
    recon = piecewise_linear_reconstruct(matrix, selected)
    residual = ((matrix - recon) ** 2).mean(axis=1)
    weights = deploy_weights(curve, cfg)
    weighted_error = float((weights * residual).sum() / max(weights.sum(), 1.0e-12))
    mean_error = float(residual.mean())
    max_gap, gap_p95, gaps = _gap_stats(selected)
    over_gap = np.clip(gaps - float(cfg.max_gap), 0.0, None)
    long_gap_risk = float(cfg.lambda_gap * (over_gap**2).sum() / max(curve.dense_t, 1))
    complexity_penalty = float(cfg.lambda_k * len(selected) + cfg.lambda_seg * max(len(selected) - 1, 0))
    islands = estimate_islands(curve, cfg)
    bands = estimate_transition_bands(curve, cfg)
    duration = float(_duration_risk(selected, islands, cfg))
    transition = float(_transition_risk(selected, bands, cfg))
    total = weighted_error + complexity_penalty + long_gap_risk + duration + transition
    return MDLObjectiveTerms(
        weighted_reconstruction_error=weighted_error,
        mean_reconstruction_error=mean_error,
        complexity_penalty=complexity_penalty,
        long_gap_risk=long_gap_risk,
        duration_risk=duration,
        transition_risk=transition,
        total_cost=float(total),
        max_gap=max_gap,
        gap_p95=gap_p95,
    )

