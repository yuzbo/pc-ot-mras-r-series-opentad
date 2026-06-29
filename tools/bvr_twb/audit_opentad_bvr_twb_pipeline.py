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
        bvr_twb_max_gap=18,
        bvr_twb_scaffold_k=4,
        bvr_twb_train_value_labels=train_value_labels,
        bvr_twb_feature_stride=2,
        bvr_twb_adapter_bridge_mode="adapter_fixed_length_padded_bridge",
    )


def run_pipeline_audit(out_dir, overwrite=False):
    out = safe_prepare_output_dir(out_dir, overwrite=overwrite)
    try:
        import torch  # noqa: F401
        from opentad.datasets.transforms.end_to_end import LoadFrames  # noqa: F401
    except (ImportError, OSError) as exc:
        summary = {
            "route_label": ROUTE_LABEL,
            "num_ledgers": 0,
            "rows": [],
            "all_validated": False,
            "blocked": True,
            "blocked_stage": "import_torch_or_loadframes",
            "blocked_reason": str(exc),
            "no_video_decode": True,
            "no_training": True,
            "no_metric_claim": True,
            "no_runtime_or_flops_claim": True,
            "no_deploy_claim": True,
            "sparse_compute_claim": False,
        }
        (out / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return summary

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
                "detector_feature_valid_k": int(ledger["detector_feature_valid_k"]),
                "adapter_bridge_mode": ledger.get("adapter_bridge_mode"),
                "adapter_input_frame_count": int(ledger.get("adapter_input_frame_count", 0)),
                "adapter_padding_duplicate_count": int(ledger.get("adapter_padding_duplicate_count", 0)),
                "adapter_padding_counts_as_valid": bool(ledger.get("adapter_padding_counts_as_valid", True)),
                "train_value_labels": int(len(transformed.get("bvr_twb_train_value_labels", []))),
            }
        )
    write_jsonl(out / "bvr_twb_opentad_pipeline_ledgers.jsonl", ledgers)
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
        "ledger_path": str((out / "bvr_twb_opentad_pipeline_ledgers.jsonl").resolve()),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


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
