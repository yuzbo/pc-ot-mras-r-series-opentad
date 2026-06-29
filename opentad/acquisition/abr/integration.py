from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .selector import select_active_bracket_refinement
from .types import ABRConfig
from .validators import assert_no_forbidden_selection_inputs, assert_real_sparse_handoff


def apply_abr_to_results(
    results: Dict[str, Any],
    config: Optional[ABRConfig] = None,
    scout_curve: Optional[Sequence[float]] = None,
    allow_gt_after_selection: bool = False,
    dense_window: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    assert_no_forbidden_selection_inputs(results, allow_gt_after_selection=allow_gt_after_selection)
    cfg = config or ABRConfig()
    dense_window = np.asarray(dense_window, dtype=np.int64) if dense_window is not None else _dense_window_from_results(results)
    dense_t = int(len(dense_window))
    fps = float(results.get("avg_fps", results.get("fps", 25.0)))
    target_frame_num = int(cfg.target_frame_num or min(max(cfg.max_total_k, cfg.k0), dense_t))
    if scout_curve is None:
        scout_curve, scout_source = _build_deploy_visible_scout_curve(results, dense_window, dense_t)
    else:
        scout_source = "call_arg:deploy_visible_scout_curve"

    selection = select_active_bracket_refinement(
        dense_t=dense_t,
        fps=fps,
        video_id=str(results.get("video_name", "unknown")),
        window_id=str(results.get("window_id", "window0")),
        scout_curve=scout_curve,
        scout_source=scout_source,
        config=cfg,
    )
    keep_positions = np.asarray(selection.selected_positions, dtype=np.int64)
    valid_k = int(keep_positions.size)
    if valid_k <= 0:
        raise RuntimeError("ABR produced no selected positions")

    original_positions = dense_window[keep_positions]
    frame_inds = original_positions.copy()
    selected_mask = np.ones(valid_k, dtype=bool)
    if valid_k < target_frame_num:
        pad_count = target_frame_num - valid_k
        frame_inds = np.pad(frame_inds, (0, pad_count), mode="edge")
        padded_positions = np.pad(keep_positions, (0, pad_count), mode="edge")
        padded_original_positions = np.pad(original_positions, (0, pad_count), mode="edge")
        selected_mask = np.concatenate([selected_mask, np.zeros(pad_count, dtype=bool)])
    else:
        frame_inds = frame_inds[:target_frame_num]
        padded_positions = keep_positions[:target_frame_num]
        padded_original_positions = original_positions[:target_frame_num]
        selected_mask = selected_mask[:target_frame_num]
        valid_k = min(valid_k, target_frame_num)

    handoff = {
        "selected_positions_window_local": padded_positions.astype(int).tolist(),
        "selected_positions_original_dense": padded_original_positions.astype(int).tolist(),
        "selected_mask": selected_mask.tolist(),
        "dense_T": dense_t,
        "dense_window": dense_window.astype(int).tolist(),
        "dense_window_start": int(dense_window[0]),
        "dense_window_end": int(dense_window[-1]),
        "valid_k": valid_k,
        "frame_inds_raw": frame_inds.astype(int).tolist(),
        "provenance": selection.provenance,
        "scout_source": selection.scout_source,
        "diagnostic_fallback_used": bool(selection.diagnostic_fallback_used),
    }
    assert_real_sparse_handoff(handoff)

    results["frame_inds"] = frame_inds.astype(int)
    results["num_clips"] = 1
    results["clip_len"] = int(frame_inds.shape[0])
    results["masks"] = selected_mask
    results["abr_selected_positions"] = keep_positions[:valid_k].astype(np.int64)
    results["abr_selected_positions_window_local"] = keep_positions[:valid_k].astype(np.int64)
    results["abr_selected_positions_original_dense"] = original_positions[:valid_k].astype(np.int64)
    results["abr_frame_inds_raw"] = frame_inds.astype(np.int64)
    results["abr_selected_valid_k"] = int(valid_k)
    results["abr_selected_rounds"] = np.asarray(selection.selected_rounds[:valid_k], dtype=np.int64)
    results["abr_selected_bracket_ids"] = np.asarray(selection.selected_bracket_ids[:valid_k], dtype=np.int64)
    results["abr_selected_roles"] = selection.selected_roles[:valid_k]
    results["abr_selection_ledger"] = selection.to_dict()
    results["abr_selection_ledger"]["handoff"] = handoff
    results["abr_dense_T"] = dense_t
    results["abr_route_label"] = selection.route_label
    results["abr_scout_source"] = selection.scout_source
    results["abr_diagnostic_fallback_used"] = bool(selection.diagnostic_fallback_used)
    results["irregular_selected_positions"] = original_positions[:valid_k].astype(np.float32)
    results["irregular_selected_valid_len"] = float(max(int(results.get("total_frames", 0)), int(dense_window[-1]) + 1))
    results["irregular_native_axis"] = "abr_original_dense_time"
    return results


def _build_deploy_visible_scout_curve(
    results: Dict[str, Any],
    dense_window: np.ndarray,
    dense_t: int,
) -> Tuple[Optional[np.ndarray], str]:
    for key in ("abr_scout_curve", "deploy_visible_scout_curve", "low_cost_scout_curve"):
        if key in results:
            signal = _signal_for_dense_window(results[key], dense_window, dense_t)
            return _normalize_signal(signal), f"{key}:deploy_visible"

    for key in ("abr_frame_signal", "raw_frame_signal", "frame_signal", "low_cost_frame_signal"):
        if key in results:
            signal = _signal_for_dense_window(results[key], dense_window, dense_t)
            return _normalize_signal(signal), f"{key}:deploy_visible_metadata"

    for key in ("abr_frame_features", "low_cost_frame_features", "frame_features"):
        if key in results:
            features = _features_for_dense_window(results[key], dense_window, dense_t)
            return _feature_change_curve(features), f"{key}:deploy_visible_feature_change"

    return None, "missing_deploy_visible_scout"


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _signal_for_dense_window(value: Any, dense_window: np.ndarray, dense_t: int) -> np.ndarray:
    signal = _to_numpy(value).astype(np.float32).reshape(-1)
    if signal.size == dense_t:
        return signal
    max_dense_index = int(dense_window.max()) if dense_window.size else -1
    if signal.size > max_dense_index:
        return signal[dense_window.astype(np.int64)]
    if signal.size == 0:
        return signal
    source_idx = np.rint(np.linspace(0, signal.size - 1, num=dense_t)).astype(np.int64)
    return signal[source_idx]


def _features_for_dense_window(value: Any, dense_window: np.ndarray, dense_t: int) -> np.ndarray:
    features = _to_numpy(value).astype(np.float32)
    if features.ndim == 1:
        return _signal_for_dense_window(features, dense_window, dense_t)[:, None]
    if features.shape[0] == dense_t:
        return features
    max_dense_index = int(dense_window.max()) if dense_window.size else -1
    if features.shape[0] > max_dense_index:
        return features[dense_window.astype(np.int64)]
    source_idx = np.rint(np.linspace(0, features.shape[0] - 1, num=dense_t)).astype(np.int64)
    return features[source_idx]


def _normalize_signal(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    if signal.size == 0:
        return signal
    finite = np.isfinite(signal)
    if not finite.all():
        signal = signal.copy()
        signal[~finite] = 0.0
    lo = float(signal.min())
    hi = float(signal.max())
    if hi - lo <= 1e-6:
        return np.full(signal.shape, 0.5, dtype=np.float32)
    return ((signal - lo) / (hi - lo)).astype(np.float32)


def _feature_change_curve(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    energy = np.linalg.norm(features, axis=1)
    change = np.zeros((features.shape[0],), dtype=np.float32)
    if features.shape[0] > 1:
        change[1:] = np.linalg.norm(np.diff(features, axis=0), axis=1)
    return 0.65 * _normalize_signal(change) + 0.35 * _normalize_signal(energy)


def _dense_window_from_results(results: Dict[str, Any]) -> np.ndarray:
    total_frames = int(results.get("total_frames", 0))
    if total_frames <= 0:
        raise ValueError("results must contain positive total_frames")
    snippet_stride = int(results.get("snippet_stride", 1))
    scale_factor = int(results.get("scale_factor", 1))
    frame_stride = max(snippet_stride // max(scale_factor, 1), 1)
    dense = np.arange(0, total_frames, frame_stride, dtype=np.int64)
    if "window_size" in results and "feature_start_idx" in results and "feature_end_idx" in results:
        start = min(max(int(results["feature_start_idx"]) * max(scale_factor, 1), 0), len(dense))
        end = min(max((int(results["feature_end_idx"]) + 1) * max(scale_factor, 1), start), len(dense))
        dense = dense[start:end]
    if dense.size == 0:
        raise ValueError("ABR received an empty dense window")
    return dense
