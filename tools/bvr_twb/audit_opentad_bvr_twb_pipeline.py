import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.acquisition.bvr_twb.types import ROUTE_LABEL
from opentad.acquisition.bvr_twb.validators import validate_bvr_twb_pipeline_ledger
from tools.bvr_twb.audit_sparse_forward_precheck import safe_prepare_output_dir, write_jsonl


BVR_TWB_PIPELINE_AUDIT_FILES = (
    "summary.json",
    "bvr_twb_opentad_pipeline_ledgers.jsonl",
)


def _prepare_pipeline_output_dir(out_dir, overwrite=False):
    try:
        return safe_prepare_output_dir(out_dir, overwrite=overwrite)
    except ValueError as exc:
        if not overwrite or "without bvr_twb marker" not in str(exc):
            raise

    out = safe_prepare_output_dir(out_dir, overwrite=False)
    resolved_out = out.resolve()
    for name in BVR_TWB_PIPELINE_AUDIT_FILES:
        path = (out / name).resolve()
        path.relative_to(resolved_out)
        if path.exists():
            path.unlink()
    return out


def _attach_adapter_precheck_fields(bridge, feature_stride=2, target_frame_num=48):
    from opentad.acquisition.bvr_twb.adapter_bridge import (
        ADAPTER_FIXED_LENGTH_PADDED_BRIDGE,
        build_adapter_fixed_length_padded_bridge,
    )
    from opentad.acquisition.bvr_twb.validators import build_selection_gap_diagnostics

    keep_positions = bridge["keep_positions"].astype(np.int64)
    selected_frame_inds = bridge["selected_frame_inds"].astype(np.int64)
    ledger = dict(bridge["ledger"])
    adapter = build_adapter_fixed_length_padded_bridge(
        selected_positions=keep_positions,
        selected_frame_inds=selected_frame_inds,
        target_frame_num=target_frame_num,
        dense_T=int(ledger["dense_T"]),
        feature_stride=int(feature_stride),
    )
    ledger["selected_positions"] = [int(pos) for pos in keep_positions.tolist()]
    ledger["selected_frame_inds"] = [int(pos) for pos in selected_frame_inds.tolist()]
    ledger["valid_k"] = int(len(keep_positions))
    ledger["raw_selected_positions"] = [int(pos) for pos in keep_positions.tolist()]
    ledger["selection_gap_diagnostics"] = build_selection_gap_diagnostics(
        keep_positions,
        int(ledger["dense_T"]),
        ledger["selection_gap_diagnostics"]["max_allowed_gap"],
    )
    ledger["adapter_bridge_mode"] = ADAPTER_FIXED_LENGTH_PADDED_BRIDGE
    ledger["adapter_target_frame_num"] = int(adapter["adapter_target_frame_num"])
    ledger["adapter_input_frame_count"] = int(adapter["adapter_input_frame_count"])
    ledger["adapter_padded_frame_inds"] = [int(pos) for pos in adapter["adapter_padded_frame_inds"].tolist()]
    ledger["adapter_padded_positions"] = [int(pos) for pos in adapter["adapter_padded_positions"].tolist()]
    ledger["adapter_valid_raw_mask"] = [bool(value) for value in adapter["adapter_valid_raw_mask"].tolist()]
    ledger["adapter_padding_duplicate_count"] = int(adapter["adapter_padding_duplicate_count"])
    ledger["adapter_padding_counts_as_valid"] = bool(adapter["adapter_padding_counts_as_valid"])
    ledger["adapter_fixed_length_padded_bridge"] = True
    ledger["padding_duplicate_count"] = int(adapter["adapter_padding_duplicate_count"])
    ledger["detector_feature_valid_k"] = int(adapter["detector_feature_valid_k"])
    ledger["detector_feature_positions"] = [float(pos) for pos in adapter["detector_feature_positions"].tolist()]
    ledger["detector_mask_len"] = int(adapter["detector_mask_len"])
    ledger["detector_mask_true_count"] = int(adapter["detector_feature_valid_k"])
    ledger["bvr_twb_feature_stride"] = int(feature_stride)
    return ledger


def _run_numpy_bridge_audit(out, import_error):
    from opentad.acquisition.bvr_twb.open_tad_bridge import build_bvr_twb_open_tad_selection

    ledgers = []
    rows = []
    dense_len = 96
    dense_window = np.arange(dense_len, dtype=np.int64)
    for split, train_labels in (("train", True), ("val", False), ("test", False)):
        results = _base_results(split)
        gt_segments = results.get("gt_segments") if train_labels else None
        gt_labels = results.get("gt_labels") if train_labels else None
        bridge = build_bvr_twb_open_tad_selection(
            results,
            dense_window=dense_window,
            target_frame_num=48,
            split=split,
            gt_segments=gt_segments,
            gt_labels=gt_labels,
            min_keep=18,
            max_keep=48,
            max_gap=18,
            scaffold_k=4,
            fps=30.0,
            window_id=0,
            train_value_labels=train_labels,
            scout_source="deploy_visible_raw_or_metadata_scout",
            require_deploy_visible_scout=True,
            allow_diagnostic_preview_fallback=False,
            scout_sample_count=16,
            value_mode="deploy_heuristic_voi",
            feature_stride=2,
            min_detector_keep=16,
        )
        ledger = _attach_adapter_precheck_fields(bridge, feature_stride=2, target_frame_num=48)
        validate_bvr_twb_pipeline_ledger(ledger)
        ledgers.append(ledger)
        rows.append(
            {
                "split": split,
                "frame_inds_len": int(ledger["adapter_input_frame_count"]),
                "mask_true_count": int(ledger["detector_mask_true_count"]),
                "raw_valid_k": int(ledger["valid_k"]),
                "dynamic_min_k": int(ledger.get("dynamic_min_k", 0)),
                "detector_feature_valid_k": int(ledger["detector_feature_valid_k"]),
                "min_detector_feature_k": int(ledger.get("min_detector_feature_k") or 0),
                "adapter_bridge_mode": ledger.get("adapter_bridge_mode"),
                "adapter_input_frame_count": int(ledger.get("adapter_input_frame_count", 0)),
                "adapter_padding_duplicate_count": int(ledger.get("adapter_padding_duplicate_count", 0)),
                "adapter_padding_duplicate_ratio": float(ledger.get("adapter_padding_duplicate_count", 0))
                / float(max(int(ledger.get("adapter_input_frame_count", 1)), 1)),
                "max_adapter_padding_duplicate_ratio": float(ledger.get("max_adapter_padding_duplicate_ratio", 0.5)),
                "adapter_padding_counts_as_valid": bool(ledger.get("adapter_padding_counts_as_valid", True)),
                "preview_source": ledger.get("preview_source"),
                "scout_source": ledger.get("scout_source"),
                "deterministic_preview_fallback_used": bool(ledger.get("deterministic_preview_fallback_used", True)),
                "value_mode": ledger.get("value_mode"),
                "value_labels_used_at_test": bool(ledger.get("value_labels_used_at_test", True)),
                "train_value_labels": int(ledger.get("num_regret_labels", 0)),
            }
        )
    write_jsonl(out / "bvr_twb_opentad_pipeline_ledgers.jsonl", ledgers)
    return _summary_from_rows(
        out,
        ledgers,
        rows,
        extra={
            "torch_loadframes_unavailable": True,
            "fallback_audit_mode": "numpy_bridge_adapter_precheck_no_torch",
            "torch_import_error": str(import_error),
        },
    )


def _summary_from_rows(out, ledgers, rows, extra=None):
    summary = {
        "route_label": ROUTE_LABEL,
        "num_ledgers": len(ledgers),
        "rows": rows,
        "all_validated": True,
        "no_video_decode": True,
        "no_training": True,
        "no_metric_claim": True,
        "no_runtime_or_flops_claim": True,
        "no_deploy_claim": True,
        "sparse_compute_claim": False,
        "adapter_bridge_mode": "adapter_fixed_length_padded_bridge",
        "adapter_bridge_modes": sorted({row["adapter_bridge_mode"] for row in rows}),
        "adapter_padding_counts_as_valid": any(row["adapter_padding_counts_as_valid"] for row in rows),
        "min_raw_valid_k": min(row["raw_valid_k"] for row in rows),
        "configured_min_raw_keep": max(row["dynamic_min_k"] for row in rows),
        "min_detector_feature_valid_k": min(row["detector_feature_valid_k"] for row in rows),
        "configured_min_detector_feature_keep": max(row["min_detector_feature_k"] for row in rows),
        "max_adapter_padding_duplicate_ratio": max(row["adapter_padding_duplicate_ratio"] for row in rows),
        "max_allowed_adapter_padding_duplicate_ratio": min(row["max_adapter_padding_duplicate_ratio"] for row in rows),
        "preview_sources": sorted({row["preview_source"] for row in rows}),
        "scout_sources": sorted({row["scout_source"] for row in rows}),
        "deterministic_preview_fallback_used": any(row["deterministic_preview_fallback_used"] for row in rows),
        "value_modes": sorted({row["value_mode"] for row in rows}),
        "value_labels_used_at_test": any(row["value_labels_used_at_test"] for row in rows),
        "ledger_path": str((out / "bvr_twb_opentad_pipeline_ledgers.jsonl").resolve()),
    }
    if extra:
        summary.update(extra)
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _base_results(split):
    total_frames = 240
    dense_len = 96
    result = {
        "video_name": f"bvr_twb_{split}_mock",
        "data_path": "mock",
        "total_frames": total_frames,
        "avg_fps": 30.0,
        "fps": 30.0,
        "duration": float(total_frames) / 30.0,
        "snippet_stride": 1,
        "window_size": dense_len,
        "feature_start_idx": 0,
        "feature_end_idx": dense_len - 1,
        "split": split,
        "bvr_twb_preview_actionness": np.clip(
            0.08
            + 0.55 * np.exp(-((np.arange(dense_len) - 34.0) ** 2) / (2.0 * 6.0**2))
            + 0.38 * np.exp(-((np.arange(dense_len) - 70.0) ** 2) / (2.0 * 9.0**2)),
            0.0,
            1.0,
        ).astype(np.float32),
    }
    if split in {"train", "val"}:
        result["gt_segments"] = np.asarray([[28.0, 42.0], [66.0, 82.0]], dtype=np.float32)
        result["gt_labels"] = np.asarray([1, 3], dtype=np.int32)
    return result


def _loader(split, train_value_labels):
    from opentad.datasets.transforms.end_to_end import LoadFrames

    return LoadFrames(
        num_clips=1,
        method="bvr_twb_dynamic_subsample",
        method_base="sliding_window",
        keep_ratio=0.5,
        target_len=48,
        scale_factor=1,
        remap_gt_to_selected_axis=False,
        bvr_twb_split=split,
                bvr_twb_min_keep=18,
                bvr_twb_max_keep=48,
                bvr_twb_min_detector_keep=16,
                bvr_twb_max_gap=18,
        bvr_twb_scaffold_k=4,
                bvr_twb_train_value_labels=train_value_labels,
                bvr_twb_feature_stride=2,
                bvr_twb_adapter_bridge_mode="adapter_fixed_length_padded_bridge",
                bvr_twb_scout_source="deploy_visible_raw_or_metadata_scout",
                bvr_twb_require_deploy_visible_scout=True,
                bvr_twb_allow_diagnostic_preview_fallback=False,
                bvr_twb_scout_sample_count=16,
                bvr_twb_value_mode="deploy_heuristic_voi",
            )


def run_pipeline_audit(out_dir, overwrite=False, force_numpy_fallback=False):
    out = _prepare_pipeline_output_dir(out_dir, overwrite=overwrite)
    if force_numpy_fallback:
        return _run_numpy_bridge_audit(out, RuntimeError("forced numpy fallback"))
    try:
        import torch  # noqa: F401
        from opentad.datasets.transforms.end_to_end import LoadFrames  # noqa: F401
    except (ImportError, OSError) as exc:
        return _run_numpy_bridge_audit(out, exc)

    ledgers = []
    rows = []
    for split, train_labels in (("train", True), ("val", False), ("test", False)):
        transformed = _loader(split, train_labels)(_base_results(split))
        ledger = transformed["bvr_twb_ledger"]
        validate_bvr_twb_pipeline_ledger(ledger)
        ledgers.append(ledger)
        rows.append(
            {
                "split": split,
                "frame_inds_len": int(len(transformed["frame_inds"])),
                "mask_true_count": int(transformed["masks"].sum().item()),
                "raw_valid_k": int(ledger["valid_k"]),
                "dynamic_min_k": int(ledger.get("dynamic_min_k", 0)),
                "detector_feature_valid_k": int(ledger["detector_feature_valid_k"]),
                "min_detector_feature_k": int(ledger.get("min_detector_feature_k") or 0),
                "adapter_bridge_mode": ledger.get("adapter_bridge_mode"),
                "adapter_input_frame_count": int(ledger.get("adapter_input_frame_count", 0)),
                "adapter_padding_duplicate_count": int(ledger.get("adapter_padding_duplicate_count", 0)),
                "adapter_padding_duplicate_ratio": float(ledger.get("adapter_padding_duplicate_count", 0))
                / float(max(int(ledger.get("adapter_input_frame_count", 1)), 1)),
                "max_adapter_padding_duplicate_ratio": float(ledger.get("max_adapter_padding_duplicate_ratio", 0.5)),
                "adapter_padding_counts_as_valid": bool(ledger.get("adapter_padding_counts_as_valid", True)),
                "preview_source": ledger.get("preview_source"),
                "scout_source": ledger.get("scout_source"),
                "deterministic_preview_fallback_used": bool(ledger.get("deterministic_preview_fallback_used", True)),
                "value_mode": ledger.get("value_mode"),
                "value_labels_used_at_test": bool(ledger.get("value_labels_used_at_test", True)),
                "train_value_labels": int(len(transformed.get("bvr_twb_train_value_labels", []))),
            }
        )
    write_jsonl(out / "bvr_twb_opentad_pipeline_ledgers.jsonl", ledgers)
    return _summary_from_rows(out, ledgers, rows)


def main():
    parser = argparse.ArgumentParser(description="Audit BVR-TWB LoadFrames pipeline without video decode/training.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = run_pipeline_audit(args.out_dir, overwrite=args.overwrite)
    print(
        "BVR-TWB OpenTAD pipeline audit: "
        f"ledgers={summary['num_ledgers']} "
        f"all_validated={summary['all_validated']} "
        f"sparse_compute_claim={summary['sparse_compute_claim']} "
        f"blocked={summary.get('blocked', False)}"
    )


if __name__ == "__main__":
    main()
