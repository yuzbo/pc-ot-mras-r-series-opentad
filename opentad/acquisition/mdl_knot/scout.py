from __future__ import annotations

from typing import Iterable, Optional, Sequence

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


def _normalize01(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo <= 1.0e-12:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - lo) / (hi - lo)


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


def build_frame_metadata_scout_curve(
    dense_t: int,
    time_index: Optional[Sequence[int]] = None,
    total_frames: Optional[int] = None,
    duration: Optional[float] = None,
    fps: Optional[float] = None,
    source: str = "frame_metadata_scout",
    provenance: Optional[dict] = None,
) -> ScoutCurve:
    """Build a deploy-visible scout from timing metadata only.

    This is a real no-GT/no-teacher source, but intentionally records its
    metadata-only nature so formal gates can distinguish it from raw-frame
    motion scouts.
    """
    dense_t = int(dense_t)
    if dense_t < 2:
        raise ValueError("dense_t must be at least 2 for metadata scout")
    if time_index is None:
        idx = np.arange(dense_t, dtype=np.float64)
    else:
        idx = np.asarray([int(v) for v in time_index], dtype=np.float64)
        if idx.size != dense_t:
            raise ValueError("time_index length must match dense_t")
    span = max(float(idx[-1] - idx[0]), 1.0)
    phase = (idx - idx[0]) / span
    fps_value = float(fps) if fps is not None and float(fps) > 0 else 30.0
    duration_value = float(duration) if duration is not None and float(duration) > 0 else dense_t / fps_value
    coverage = float(dense_t) / max(float(total_frames or dense_t), 1.0)

    edge = np.maximum(0.0, 1.0 - np.minimum(phase, 1.0 - phase) / 0.18)
    temporal_density = np.clip(dense_t / max(duration_value * fps_value, 1.0), 0.0, 1.0)
    p_action = np.clip(0.10 + 0.10 * edge + 0.05 * temporal_density + 0.05 * coverage, 0.0, 1.0)
    change = _normalized_gradient(p_action)
    uncertainty = np.clip(4.0 * p_action * (1.0 - p_action) + 0.15 * change, 0.0, 1.0)
    persistence = np.clip(1.0 - change, 0.0, 1.0)
    prov = {
        "scout_source": source,
        "uses_raw_frame_probe": False,
        "uses_frame_metadata": True,
        "metadata_only": True,
        "total_frames": None if total_frames is None else int(total_frames),
        "duration": None if duration is None else float(duration),
        "fps": fps_value,
    }
    prov.update(provenance or {})
    return build_deploy_scout_curve(
        p_action=p_action,
        uncertainty=uncertainty,
        temporal_change=change,
        persistence=persistence,
        source=source,
        provenance=prov,
    )


def build_raw_frame_motion_scout_curve(
    probe_frames: Sequence[object],
    probe_positions: Sequence[int],
    dense_t: int,
    source: str = "raw_frame_motion_scout",
    provenance: Optional[dict] = None,
) -> ScoutCurve:
    """Build a deploy-visible scout from lightweight raw-frame probes."""
    dense_t = int(dense_t)
    if dense_t < 2:
        raise ValueError("dense_t must be at least 2 for raw-frame scout")
    positions = np.asarray([int(v) for v in probe_positions], dtype=np.int64)
    if positions.ndim != 1 or positions.size < 2:
        raise ValueError("raw-frame scout requires at least two probe positions")
    if positions[0] < 0 or positions[-1] >= dense_t or np.any(np.diff(positions) <= 0):
        raise ValueError("probe_positions must be sorted unique positions in dense range")
    features = []
    for frame in probe_frames:
        arr = np.asarray(frame)
        if arr.ndim < 2:
            raise ValueError("probe frames must have spatial dimensions")
        if arr.ndim == 2:
            arr = arr[..., None]
        arr = arr.astype(np.float32)
        step_h = max(arr.shape[0] // 24, 1)
        step_w = max(arr.shape[1] // 24, 1)
        arr = arr[::step_h, ::step_w]
        flat = arr.reshape(-1, arr.shape[-1]) / 255.0
        if flat.shape[1] == 1:
            gray = flat[:, 0]
            color = np.zeros(2, dtype=np.float64)
        else:
            gray = flat[:, :3].mean(axis=1)
            means = flat[:, :3].mean(axis=0)
            color = np.asarray([means[0] - means[1], means[2] - means[1]], dtype=np.float64)
        features.append(
            np.concatenate(
                [
                    np.asarray([gray.mean(), gray.std()], dtype=np.float64),
                    color,
                ]
            )
        )
    feats = np.stack(features, axis=0)
    diffs = np.zeros((feats.shape[0],), dtype=np.float64)
    if feats.shape[0] > 1:
        diffs[1:] = np.linalg.norm(np.diff(feats, axis=0), axis=1)
    motion_probe = _normalize01(diffs)
    texture_probe = _normalize01(feats[:, 1])
    color_probe = _normalize01(np.linalg.norm(feats[:, 2:], axis=1))
    action_probe = np.clip(0.08 + 0.52 * motion_probe + 0.22 * texture_probe + 0.18 * color_probe, 0.0, 1.0)

    x_full = np.arange(dense_t, dtype=np.float64)
    p_action = np.interp(x_full, positions.astype(np.float64), action_probe)
    motion = np.interp(x_full, positions.astype(np.float64), motion_probe)
    change = np.maximum(_normalized_gradient(p_action), motion)
    uncertainty = np.clip(4.0 * p_action * (1.0 - p_action) * 0.45 + 0.55 * change, 0.0, 1.0)
    persistence = np.clip(1.0 - change, 0.0, 1.0)
    prov = {
        "scout_source": source,
        "uses_raw_frame_probe": True,
        "uses_frame_metadata": True,
        "metadata_only": False,
        "raw_probe_frame_count": int(len(probe_frames)),
        "probe_positions": positions.astype(int).tolist(),
    }
    prov.update(provenance or {})
    return build_deploy_scout_curve(
        p_action=p_action,
        uncertainty=uncertainty,
        temporal_change=change,
        persistence=persistence,
        motion=motion,
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
        source="synthetic_precheck_diagnostic",
        provenance={"synthetic_precheck_only": True},
    )
