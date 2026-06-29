from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

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

    selection = select_active_bracket_refinement(
        dense_t=dense_t,
        fps=fps,
        video_id=str(results.get("video_name", "unknown")),
        window_id=str(results.get("window_id", "window0")),
        scout_curve=scout_curve,
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
    results["irregular_selected_positions"] = original_positions[:valid_k].astype(np.float32)
    results["irregular_selected_valid_len"] = float(max(int(results.get("total_frames", 0)), int(dense_window[-1]) + 1))
    results["irregular_native_axis"] = "abr_original_dense_time"
    return results


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
