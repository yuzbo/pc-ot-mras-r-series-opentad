from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from .types import ScoutCurve


def _clip01(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("scout curve inputs must be non-empty 1D sequences")
    if not np.isfinite(arr).all():
        raise ValueError("scout curve inputs must be finite")
    return np.clip(arr, 0.0, 1.0)


def _normalized_gradient(values: np.ndarray) -> np.ndarray:
    grad = np.abs(np.gradient(values.astype(np.float64)))
    max_value = float(grad.max()) if grad.size else 0.0
    if max_value <= 1.0e-12:
        return np.zeros_like(values, dtype=np.float64)
    return grad / max_value


def build_deploy_scout_curve(
    p_action: Iterable[float],
    uncertainty: Optional[Iterable[float]] = None,
    temporal_change: Optional[Iterable[float]] = None,
    persistence: Optional[Iterable[float]] = None,
    motion: Optional[Iterable[float]] = None,
    source: str = "deploy_scout",
    provenance: Optional[dict] = None,
) -> ScoutCurve:
    p = _clip01(p_action)
    if uncertainty is None:
        # Bernoulli entropy proxy, deploy-visible from action posterior alone.
        u = np.clip(4.0 * p * (1.0 - p), 0.0, 1.0)
    else:
        u = _clip01(uncertainty)
    if temporal_change is None:
        c = _normalized_gradient(p)
    else:
        c = _clip01(temporal_change)
    if persistence is None:
        pers = np.clip(1.0 - c, 0.0, 1.0)
    else:
        pers = _clip01(persistence)
    m = None if motion is None else _clip01(motion).tolist()
    if not (len(p) == len(u) == len(c) == len(pers)):
        raise ValueError("p_action, uncertainty, temporal_change, and persistence must have the same length")
    prov = {
        "uses_gt": False,
        "uses_teacher": False,
        "uses_prediction_cache": False,
        "dense_raw_backbone_handoff": False,
        "selected_inputs_is_gathered": True,
        "position_unit": "original_dense_time_index",
    }
    prov.update(provenance or {})
    return ScoutCurve(
        p_action=p.tolist(),
        uncertainty=u.tolist(),
        temporal_change=c.tolist(),
        persistence=pers.tolist(),
        motion=m,
        time_index=list(range(len(p))),
        source=source,
        provenance=prov,
    )


def _gaussian_grid(dense_t: int, center: float, width: float) -> np.ndarray:
    x = np.arange(dense_t, dtype=np.float64)
    return np.exp(-0.5 * ((x - center) / max(width, 1.0)) ** 2)


def build_synthetic_scout_curve(pattern: str, dense_t: int = 128) -> ScoutCurve:
    if dense_t < 4:
        raise ValueError("dense_t must be at least 4")
    x = np.linspace(0.0, 1.0, dense_t)
    pattern = pattern.lower()
    if pattern == "stable_background":
        p = 0.06 + 0.015 * np.sin(2.0 * np.pi * x)
    elif pattern == "sharp_transition":
        p = np.where((x > 0.35) & (x < 0.70), 0.82, 0.08).astype(np.float64)
        p += 0.03 * np.sin(8.0 * np.pi * x)
    elif pattern == "two_islands":
        p = 0.05 + 0.72 * _gaussian_grid(dense_t, dense_t * 0.28, dense_t * 0.07)
        p += 0.62 * _gaussian_grid(dense_t, dense_t * 0.70, dense_t * 0.10)
    elif pattern == "short_islands":
        p = 0.04 + 0.80 * _gaussian_grid(dense_t, dense_t * 0.18, dense_t * 0.018)
        p += 0.76 * _gaussian_grid(dense_t, dense_t * 0.48, dense_t * 0.025)
        p += 0.78 * _gaussian_grid(dense_t, dense_t * 0.76, dense_t * 0.020)
        p += 0.10 * np.sin(20.0 * np.pi * x) ** 2
    else:
        raise ValueError(f"unknown synthetic scout pattern: {pattern}")
    p = np.clip(p, 0.0, 1.0)
    change = _normalized_gradient(p)
    uncertainty = np.clip(4.0 * p * (1.0 - p) + 0.30 * change, 0.0, 1.0)
    persistence = np.clip(1.0 - change, 0.0, 1.0)
    motion = np.clip(0.25 * change + 0.10 * np.sin(6.0 * np.pi * x) ** 2, 0.0, 1.0)
    return build_deploy_scout_curve(
        p_action=p,
        uncertainty=uncertainty,
        temporal_change=change,
        persistence=persistence,
        motion=motion,
    )

