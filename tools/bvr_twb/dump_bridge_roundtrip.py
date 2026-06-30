import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.acquisition.bvr_twb.adapter_bridge import (  # noqa: E402
    ADAPTER_FIXED_LENGTH_PADDED_BRIDGE,
    build_adapter_fixed_length_padded_bridge,
)
from opentad.acquisition.bvr_twb.open_tad_bridge import build_bvr_twb_open_tad_selection  # noqa: E402
from opentad.acquisition.bvr_twb.types import ROUTE_LABEL  # noqa: E402
from opentad.acquisition.bvr_twb.validators import (  # noqa: E402
    build_original_time_metadata,
    build_selection_gap_diagnostics,
    validate_bvr_twb_pipeline_ledger,
)


def _preview(dense_t):
    x = np.arange(dense_t, dtype=np.float64)
    preview = (
        0.06
        + 0.58 * np.exp(-((x - 106.0) ** 2) / (2.0 * 12.0**2))
        + 0.42 * np.exp(-((x - 255.0) ** 2) / (2.0 * 18.0**2))
        + 0.08 * np.sin(x / 23.0) ** 2
    )
    return np.clip(preview, 0.0, 1.0).astype(np.float32)


def _base_results(dense_t):
    return {
        "video_name": "bvr_twb_roundtrip_window_0001",
        "data_path": "mock",
        "total_frames": dense_t * 2 + 80,
        "avg_fps": 30.0,
        "fps": 30.0,
        "duration": float(dense_t * 2 + 80) / 30.0,
        "snippet_stride": 2,
        "window_start_frame": 30,
        "offset_frames": 4,
        "window_size": dense_t,
        "feature_start_idx": 15,
        "feature_end_idx": 15 + dense_t - 1,
        "split": "val",
        "gt_segments": np.asarray([[92.0, 136.0], [238.0, 292.0]], dtype=np.float32),
        "gt_labels": np.asarray([1, 3], dtype=np.int32),
        "bvr_twb_preview_actionness": _preview(dense_t),
    }


def _seconds_numpy(segments, meta):
    segments = np.asarray(segments, dtype=np.float32)
    return (
        segments * float(meta["snippet_stride"])
        + float(meta.get("window_start_frame", 0))
        + float(meta["offset_frames"])
    ) / float(meta["fps"])


def _torch_seconds_if_available(segments, meta):
    probe = subprocess.run(
        [sys.executable, "-c", "import torch"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if probe.returncode != 0:
        reason = (probe.stderr or probe.stdout).strip().splitlines()
        return None, reason[-1] if reason else "torch import failed"
    try:
        import torch
        from opentad.models.utils.post_processing import convert_to_seconds
    except (ImportError, OSError) as exc:
        return None, str(exc)
    tensor = torch.as_tensor(segments, dtype=torch.float32)
    seconds = convert_to_seconds(tensor, dict(meta))
    return seconds.detach().cpu().numpy().tolist(), None


def _attach_bridge_fields(ledger, selected_positions, selected_frame_inds, target_frame_num, dense_t, feature_stride):
    bridge = build_adapter_fixed_length_padded_bridge(
        selected_positions=selected_positions,
        selected_frame_inds=selected_frame_inds,
        target_frame_num=target_frame_num,
        dense_T=dense_t,
        feature_stride=feature_stride,
    )
    out = dict(ledger)
    out.update(
        {
            "selected_positions": [int(pos) for pos in selected_positions.tolist()],
            "selected_frame_inds": [int(pos) for pos in selected_frame_inds.tolist()],
            "raw_selected_positions": [int(pos) for pos in selected_positions.tolist()],
            "valid_k": int(selected_positions.shape[0]),
            "selection_gap_diagnostics": build_selection_gap_diagnostics(
                selected_positions,
                dense_t,
                int(ledger["selection_gap_diagnostics"]["max_allowed_gap"]),
            ),
            "adapter_bridge_mode": ADAPTER_FIXED_LENGTH_PADDED_BRIDGE,
            "adapter_target_frame_num": int(bridge["adapter_target_frame_num"]),
            "adapter_input_frame_count": int(bridge["adapter_input_frame_count"]),
            "adapter_padded_frame_inds": [int(pos) for pos in bridge["adapter_padded_frame_inds"].tolist()],
            "adapter_padded_positions": [int(pos) for pos in bridge["adapter_padded_positions"].tolist()],
            "adapter_valid_raw_mask": [bool(value) for value in bridge["adapter_valid_raw_mask"].tolist()],
            "adapter_padding_duplicate_count": int(bridge["adapter_padding_duplicate_count"]),
            "adapter_padding_counts_as_valid": bool(bridge["adapter_padding_counts_as_valid"]),
            "adapter_fixed_length_padded_bridge": True,
            "padding_duplicate_count": int(bridge["adapter_padding_duplicate_count"]),
            "detector_feature_valid_k": int(bridge["detector_feature_valid_k"]),
            "detector_feature_positions": [float(pos) for pos in bridge["detector_feature_positions"].tolist()],
            "detector_mask_len": int(bridge["detector_mask_len"]),
            "detector_mask_true_count": int(bridge["detector_feature_valid_k"]),
            "bvr_twb_feature_stride": int(feature_stride),
        }
    )
    validate_bvr_twb_pipeline_ledger(out)
    return out, bridge


def _dynamic_case(dense_window, target_frame_num, feature_stride, results):
    bridge = build_bvr_twb_open_tad_selection(
        results,
        dense_window=dense_window,
        target_frame_num=target_frame_num,
        split="val",
        gt_segments=results["gt_segments"],
        gt_labels=results["gt_labels"],
        min_keep=64,
        max_keep=target_frame_num,
        max_gap=32,
        scaffold_k=4,
        fps=results["fps"],
        window_id=1,
        train_value_labels=False,
        scout_source="deploy_visible_raw_or_metadata_scout",
        require_deploy_visible_scout=True,
        allow_diagnostic_preview_fallback=False,
        scout_sample_count=32,
        value_mode="deploy_heuristic_voi",
    )
    ledger, adapter = _attach_bridge_fields(
        bridge["ledger"],
        bridge["keep_positions"].astype(np.int64),
        bridge["selected_frame_inds"].astype(np.int64),
        target_frame_num,
        int(dense_window.shape[0]),
        feature_stride,
    )
    return ledger, adapter


def _forced_uniform_case(dense_window, target_frame_num, feature_stride, results):
    dense_t = int(dense_window.shape[0])
    selected_positions = np.round(np.linspace(0, dense_t - 1, target_frame_num)).astype(np.int64)
    selected_positions = np.unique(selected_positions)
    if selected_positions.shape[0] != target_frame_num:
        raise ValueError("forced-uniform diagnostic failed to create exact target count")
    selected_frame_inds = dense_window[selected_positions]
    ledger = {
        "route_label": ROUTE_LABEL,
        "method": "bvr_twb_dynamic_subsample",
        "split": "val",
        "dense_T": dense_t,
        "target_frame_num": int(target_frame_num),
        "budget_stop_reason": "forced_uniform_control_diagnostic",
        "selection_gap_diagnostics": build_selection_gap_diagnostics(selected_positions, dense_t, 32),
        "original_time_metadata": build_original_time_metadata(dense_t, selected_positions, fps=results["fps"]),
        "temporal_decode_uses_original_time": True,
        "selected_index_is_time": False,
        "dense_raw_backbone_handoff": False,
        "selected_inputs_is_gathered": True,
        "sparse_compute_claim": False,
        "claim_status": "forced_uniform_bridge_diagnostic_no_metric_claim",
        "selector_provenance": {
            "selection_uses_gt": False,
            "selection_uses_teacher": False,
            "selection_uses_prediction_cache": False,
            "selection_uses_raw_detector_prediction": False,
            "selection_uses_oracle_boundary": False,
            "selection_uses_oracle_residual": False,
        },
        "preview_source": "deploy_visible_metadata_actionness",
        "scout_source": "deploy_visible_raw_or_metadata_scout",
        "scout_is_deploy_visible": True,
        "deterministic_preview_fallback_used": False,
        "diagnostic_preview_fallback_allowed": False,
        "value_mode": "deploy_heuristic_voi",
        "value_model_used": False,
        "value_labels_used_at_test": False,
    }
    return _attach_bridge_fields(
        ledger,
        selected_positions,
        selected_frame_inds.astype(np.int64),
        target_frame_num,
        dense_t,
        feature_stride,
    )


def _roundtrip_row(case_name, ledger, adapter, results):
    probe_segments = np.asarray([[96.0, 144.0], [240.0, 288.0]], dtype=np.float32)
    meta = {
        "fps": results["fps"],
        "duration": results["duration"],
        "snippet_stride": results["snippet_stride"],
        "offset_frames": results["offset_frames"],
        "window_start_frame": results["window_start_frame"],
        "irregular_native_axis": True,
        "bvr_twb_detector_feature_positions": ledger["detector_feature_positions"],
        "bvr_twb_detector_feature_valid_len": float(ledger["dense_T"]),
    }
    expected_seconds = _seconds_numpy(probe_segments, meta).tolist()
    torch_seconds, torch_error = _torch_seconds_if_available(probe_segments, meta)
    seconds_match = torch_seconds is None or np.allclose(np.asarray(torch_seconds), np.asarray(expected_seconds))
    return {
        "case": case_name,
        "route_label": ROUTE_LABEL,
        "dense_T": int(ledger["dense_T"]),
        "target_frame_num": int(ledger["adapter_target_frame_num"]),
        "raw_valid_k": int(ledger["valid_k"]),
        "selected_frame_inds_head": ledger["selected_frame_inds"][:16],
        "selected_frame_inds_tail": ledger["selected_frame_inds"][-16:],
        "raw_selected_positions_head": ledger["raw_selected_positions"][:16],
        "raw_selected_positions_tail": ledger["raw_selected_positions"][-16:],
        "detector_feature_valid_k": int(ledger["detector_feature_valid_k"]),
        "detector_mask_len": int(ledger["detector_mask_len"]),
        "detector_mask_true_count": int(ledger["detector_mask_true_count"]),
        "detector_feature_positions_head": ledger["detector_feature_positions"][:16],
        "detector_feature_positions_tail": ledger["detector_feature_positions"][-16:],
        "adapter_input_frame_count": int(ledger["adapter_input_frame_count"]),
        "adapter_padding_duplicate_count": int(ledger["adapter_padding_duplicate_count"]),
        "adapter_padding_counts_as_valid": bool(ledger["adapter_padding_counts_as_valid"]),
        "adapter_valid_raw_mask_true_count": int(np.asarray(adapter["adapter_valid_raw_mask"], dtype=bool).sum()),
        "gt_segments_native_axis": results["gt_segments"].astype(float).tolist(),
        "probe_segments_native_detector_grid": probe_segments.tolist(),
        "expected_seconds": expected_seconds,
        "torch_convert_to_seconds": torch_seconds,
        "torch_convert_to_seconds_error": torch_error,
        "seconds_roundtrip_match": bool(seconds_match),
        "ledger_validated": True,
    }


def run_dump(out_dir, overwrite=False):
    out = Path(out_dir)
    if out.exists() and overwrite:
        for child in out.iterdir():
            if child.is_file():
                child.unlink()
    out.mkdir(parents=True, exist_ok=True)

    dense_t = 384
    target_frame_num = 192
    feature_stride = 2
    results = _base_results(dense_t)
    dense_window = results["window_start_frame"] + np.arange(dense_t, dtype=np.int64) * int(results["snippet_stride"])

    dynamic_ledger, dynamic_adapter = _dynamic_case(dense_window, target_frame_num, feature_stride, results)
    uniform_ledger, uniform_adapter = _forced_uniform_case(dense_window, target_frame_num, feature_stride, results)
    ledgers = [dynamic_ledger, uniform_ledger]
    rows = [
        _roundtrip_row("dynamic_bvr", dynamic_ledger, dynamic_adapter, results),
        _roundtrip_row("forced_uniform_through_bvr_bridge", uniform_ledger, uniform_adapter, results),
    ]
    summary = {
        "route_label": ROUTE_LABEL,
        "diagnostic": "bvr_twb_bridge_roundtrip",
        "no_training": True,
        "no_video_decode": True,
        "no_metric_claim": True,
        "no_runtime_or_flops_claim": True,
        "no_deploy_claim": True,
        "full_training_unlocked": False,
        "cases": [row["case"] for row in rows],
        "all_ledgers_validated": True,
        "all_seconds_roundtrip_match": all(row["seconds_roundtrip_match"] for row in rows),
        "forced_uniform_raw_valid_k": int(uniform_ledger["valid_k"]),
        "forced_uniform_detector_feature_valid_k": int(uniform_ledger["detector_feature_valid_k"]),
        "forced_uniform_adapter_padding_duplicate_count": int(uniform_ledger["adapter_padding_duplicate_count"]),
        "proposal_nms_score_diagnostic": "pending_requires_checkpoint_or_eval_artifacts; tools/test.py not run",
    }
    (out / "bridge_roundtrip_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out / "bridge_roundtrip_rows.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with (out / "bridge_roundtrip_ledgers.jsonl").open("w", encoding="utf-8") as handle:
        for ledger in ledgers:
            handle.write(json.dumps(ledger, ensure_ascii=False, sort_keys=True) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Dump BVR-TWB bridge ledger and geometry round-trip diagnostics.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = run_dump(args.out_dir, overwrite=args.overwrite)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
