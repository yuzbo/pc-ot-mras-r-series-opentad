import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.acquisition.bvr_twb.types import ROUTE_LABEL
from tools.bvr_twb.audit_runtime_provenance import _base_results
from tools.bvr_twb.audit_sparse_forward_precheck import safe_prepare_output_dir


META_KEYS = [
    "video_name",
    "data_path",
    "fps",
    "duration",
    "snippet_stride",
    "window_start_frame",
    "resize_length",
    "window_size",
    "offset_frames",
    "irregular_selected_positions",
    "irregular_selected_valid_len",
    "irregular_native_axis",
    "bvr_twb_ledger",
    "bvr_twb_raw_selected_positions",
    "bvr_twb_raw_selected_valid_len",
    "bvr_twb_detector_feature_positions",
    "bvr_twb_detector_feature_valid_len",
    "bvr_twb_selected_positions",
    "bvr_twb_selected_valid_len",
    "bvr_twb_dense_valid_len",
    "bvr_twb_train_value_labels",
    "bvr_twb_candidate_count",
]


def _fake_decord_decode(frame_inds, height=4, width=4):
    frame_inds = np.asarray(frame_inds, dtype=np.int64).reshape(-1)
    imgs = np.zeros((frame_inds.shape[0], height, width, 3), dtype=np.uint8)
    imgs[:, :, :, 0] = (frame_inds % 251).astype(np.uint8)[:, None, None]
    imgs[:, :, :, 1] = ((frame_inds // 2) % 251).astype(np.uint8)[:, None, None]
    imgs[:, :, :, 2] = ((frame_inds // 3) % 251).astype(np.uint8)[:, None, None]
    return imgs


def _format_ncthw(imgs, num_clips):
    imgs = np.asarray(imgs)
    clip_len = imgs.shape[0] // int(num_clips)
    return imgs.reshape((int(num_clips), clip_len) + imgs.shape[1:]).transpose(0, 4, 1, 2, 3)


def _collect_meta(results):
    return {key: results[key] for key in META_KEYS if key in results}


def run_one_sample_trace(out_dir=None, overwrite=False, require_runtime=False):
    summary = {
        "route_label": ROUTE_LABEL,
        "audit": "bvr_twb_one_sample_pipeline_trace",
        "no_training": True,
        "no_real_video": True,
        "no_metric_claim": True,
        "no_runtime_or_flops_claim": True,
        "full_training_unlocked": False,
        "blocked": False,
    }
    try:
        proc = subprocess.run(
            [sys.executable, "-c", "import torch"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if proc.returncode != 0:
            message = (proc.stderr or proc.stdout).strip() or f"torch import probe failed with exit code {proc.returncode}"
            raise RuntimeError(f"torch import probe failed: {message.splitlines()[-1]}")
        from opentad.acquisition.bvr_twb.adapter_bridge import ADAPTER_FIXED_LENGTH_PADDED_BRIDGE
        from opentad.datasets.transforms.end_to_end import LoadFrames

        original_gt = np.asarray([[24.0, 34.0]], dtype=np.float32)
        loader = LoadFrames(
            num_clips=1,
            method="bvr_twb_dynamic_subsample",
            method_base="sliding_window",
            keep_ratio=0.5,
            target_len=32,
            scale_factor=1,
            remap_gt_to_selected_axis=False,
            bvr_twb_split="train",
            bvr_twb_min_keep=12,
            bvr_twb_max_keep=32,
            bvr_twb_max_gap=16,
            bvr_twb_scaffold_k=4,
            bvr_twb_train_value_labels=True,
            bvr_twb_feature_stride=2,
            bvr_twb_adapter_bridge_mode=ADAPTER_FIXED_LENGTH_PADDED_BRIDGE,
            bvr_twb_scout_source="deploy_visible_raw_or_metadata_scout",
            bvr_twb_require_deploy_visible_scout=True,
            bvr_twb_allow_diagnostic_preview_fallback=False,
            bvr_twb_scout_sample_count=16,
            bvr_twb_value_mode="deploy_heuristic_voi",
        )
        transformed = loader(_base_results("train", with_gt=True))
        ledger = transformed["bvr_twb_ledger"]
        frame_inds = np.asarray(transformed["frame_inds"], dtype=np.int64)
        decoded = _fake_decord_decode(frame_inds)
        ncthw = _format_ncthw(decoded, transformed["num_clips"])
        meta = _collect_meta(transformed)
        selected_frame_inds = np.asarray(ledger["selected_frame_inds"], dtype=np.int64)
        decoded_unique = set(int(value) for value in np.unique(frame_inds).tolist())

        summary.update(
            {
                "runtime_trace": "passed",
                "dispatch_hit": ledger.get("method") == "bvr_twb_dynamic_subsample",
                "ledger_method": ledger.get("method"),
                "ledger_route_label": ledger.get("route_label"),
                "frame_idxs_shape": list(frame_inds.shape),
                "frame_idxs_count": int(frame_inds.shape[0]),
                "frame_idxs_unique_count": int(np.unique(frame_inds).shape[0]),
                "selected_frame_inds_count": int(selected_frame_inds.shape[0]),
                "selected_frame_inds_subset_of_decoded_unique": all(int(value) in decoded_unique for value in selected_frame_inds),
                "adapter_padding_duplicate_count": int(ledger.get("adapter_padding_duplicate_count", 0)),
                "mask_shape": list(transformed["masks"].shape),
                "mask_true_count": int(transformed["masks"].sum().item()),
                "detector_feature_valid_k": int(ledger["detector_feature_valid_k"]),
                "fake_decord_decode_input_count": int(frame_inds.shape[0]),
                "fake_decord_decode_unique_count": int(np.unique(frame_inds).shape[0]),
                "fake_decord_imgs_shape": list(decoded.shape),
                "fake_backbone_input_ncthw_shape": list(ncthw.shape),
                "collect_meta_keys": sorted(meta.keys()),
                "collect_meta_has_bvr_ledger": "bvr_twb_ledger" in meta,
                "collect_meta_has_detector_positions": "bvr_twb_detector_feature_positions" in meta,
                "gt_native_axis_preserved": bool(np.allclose(transformed["gt_segments"], original_gt)),
                "irregular_native_axis": bool(transformed.get("irregular_native_axis", False)),
            }
        )
        summary["all_required_trace_keys_present"] = all(
            [
                summary["dispatch_hit"],
                summary["selected_frame_inds_subset_of_decoded_unique"],
                summary["collect_meta_has_bvr_ledger"],
                summary["collect_meta_has_detector_positions"],
                summary["gt_native_axis_preserved"],
                summary["irregular_native_axis"],
            ]
        )
    except Exception as exc:
        if require_runtime:
            raise
        summary["runtime_trace"] = "skipped"
        summary["runtime_skip_reason"] = f"{type(exc).__name__}: {exc}"
        summary["all_required_trace_keys_present"] = False
    if require_runtime and not summary["all_required_trace_keys_present"]:
        summary["blocked"] = True
    if out_dir is not None:
        out = safe_prepare_output_dir(out_dir, overwrite=overwrite)
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Trace one synthetic BVR-TWB sample through LoadFrames and fake Decord handoff.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--require-runtime", action="store_true")
    args = parser.parse_args()
    summary = run_one_sample_trace(args.out_dir, overwrite=args.overwrite, require_runtime=args.require_runtime)
    print(
        "BVR-TWB one-sample trace: "
        f"runtime={summary.get('runtime_trace')} "
        f"dispatch_hit={summary.get('dispatch_hit', False)} "
        f"trace_keys={summary.get('all_required_trace_keys_present', False)}"
    )


if __name__ == "__main__":
    main()
