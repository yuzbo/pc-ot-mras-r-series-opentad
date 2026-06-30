import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.acquisition.bvr_twb.types import ROUTE_LABEL
from tools.bvr_twb.audit_sparse_forward_precheck import safe_prepare_output_dir


def run_postprocess_audit(out_dir=None, overwrite=False, require_runtime=False):
    summary = {
        "route_label": ROUTE_LABEL,
        "audit": "bvr_twb_postprocess_proposal_count",
        "no_training": True,
        "no_real_video": True,
        "no_checkpoint": True,
        "no_metric_claim": True,
        "no_runtime_or_flops_claim": True,
        "full_training_unlocked": False,
        "blocked": False,
    }
    out = None
    audit_path = None
    old_enabled = os.environ.get("BVR_TWB_POSTPROCESS_AUDIT")
    old_path = os.environ.get("BVR_TWB_POSTPROCESS_AUDIT_PATH")
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
        if out_dir is not None:
            out = safe_prepare_output_dir(out_dir, overwrite=overwrite)
            audit_path = out / "postprocess_audit.jsonl"
            os.environ["BVR_TWB_POSTPROCESS_AUDIT_PATH"] = str(audit_path)
        os.environ["BVR_TWB_POSTPROCESS_AUDIT"] = "1"

        import torch

        from opentad.models.detectors.irregular_actionformer import IrregularActionFormer

        proposal_count = 19260
        num_classes = 19
        starts = torch.linspace(0.0, 382.0, proposal_count)
        proposals = torch.stack([starts, starts + 1.0], dim=1)
        scores = torch.full((proposal_count, num_classes), 0.01, dtype=torch.float32)
        metas = [
            {
                "video_name": "bvr_postprocess_365940_synthetic",
                "fps": 30.0,
                "duration": 30.0,
                "snippet_stride": 1,
                "offset_frames": 0,
                "window_start_frame": 0,
                "irregular_native_axis": True,
                "bvr_twb_ledger": {"method": "bvr_twb_dynamic_subsample"},
                "bvr_twb_detector_feature_positions": np.asarray([0.0, 32.0, 96.0], dtype=np.float32),
                "bvr_twb_detector_feature_valid_len": 384.0,
            }
        ]
        ext_cls = [f"class_{idx}" for idx in range(num_classes)]

        legacy_detector = object.__new__(IrregularActionFormer)
        legacy_post_cfg = SimpleNamespace(pre_nms_thresh=0.001, pre_nms_topk=2000, sliding_window=True, nms=None)
        legacy_results = legacy_detector.post_processing(([proposals], [scores]), metas, legacy_post_cfg, ext_cls)
        audit_rows = getattr(legacy_detector, "_last_bvr_twb_postprocess_audit", [])
        if not audit_rows:
            raise ValueError("BVR-TWB postprocess audit env hook did not record a row")
        row = audit_rows[-1]
        expected_flat = proposal_count * num_classes
        if int(row["flattened_candidate_count"]) != expected_flat:
            raise ValueError("postprocess flattened candidate count mismatch")

        guarded_detector = object.__new__(IrregularActionFormer)
        guarded_post_cfg = SimpleNamespace(
            pre_nms_thresh=0.001,
            pre_nms_topk=512,
            sliding_window=True,
            nms=None,
            bvr_twb_postprocess_guard=dict(
                enabled=True,
                require_bvr_meta=True,
                raw_proposal_cap=1024,
                per_class_topk=32,
                total_candidate_cap=512,
                min_score=0.001,
            ),
        )
        guarded_results = guarded_detector.post_processing(([proposals], [scores]), metas, guarded_post_cfg, ext_cls)
        guarded_rows = getattr(guarded_detector, "_last_bvr_twb_postprocess_audit", [])
        if not guarded_rows:
            raise ValueError("BVR-TWB guarded postprocess audit env hook did not record a row")
        guarded_row = guarded_rows[-1]
        if not bool(guarded_row.get("guard_active", False)):
            raise ValueError("BVR-TWB postprocess guard did not activate")
        if int(guarded_row["pre_nms_selected_count"]) > 512:
            raise ValueError("BVR-TWB guarded pre-NMS candidate count exceeded cap")
        if int(guarded_row["above_threshold_count"]) > 32 * num_classes:
            raise ValueError("BVR-TWB guarded per-class candidate count exceeded cap")
        summary.update(
            {
                "runtime_contract": "passed",
                "raw_proposal_count": int(row["raw_proposal_count"]),
                "num_classes": int(row["num_classes"]),
                "flattened_candidate_count": int(row["flattened_candidate_count"]),
                "expected_explosion_formula": "raw_proposal_count * num_classes",
                "explains_365940_predictions": int(row["flattened_candidate_count"]) == 365940,
                "above_threshold_count": int(row["above_threshold_count"]),
                "pre_nms_selected_count": int(row["pre_nms_selected_count"]),
                "post_nms_count": int(row["post_nms_count"]),
                "final_result_count": int(row["final_result_count"]),
                "result_video_count": len(legacy_results),
                "guard_runtime_contract": "passed",
                "guard_active": bool(guarded_row["guard_active"]),
                "guard_candidate_generation_mode": guarded_row["candidate_generation_mode"],
                "guard_raw_proposal_cap": int(guarded_row["guard_raw_proposal_cap"]),
                "guard_raw_input_count": int(guarded_row["guard_raw_input_count"]),
                "guard_raw_selected_count": int(guarded_row["guard_raw_selected_count"]),
                "guard_per_class_topk": int(guarded_row["guard_per_class_topk"]),
                "guard_total_candidate_cap": int(guarded_row["guard_total_candidate_cap"]),
                "guard_above_threshold_before_per_class_cap": int(
                    guarded_row["guard_above_threshold_before_per_class_cap"]
                ),
                "guard_class_candidate_count_before_global_topk": int(
                    guarded_row["guard_class_candidate_count_before_global_topk"]
                ),
                "guard_pre_nms_selected_count": int(guarded_row["pre_nms_selected_count"]),
                "guard_post_nms_count": int(guarded_row["post_nms_count"]),
                "guard_final_result_count": int(guarded_row["final_result_count"]),
                "guard_result_video_count": len(guarded_results),
                "guard_no_metric_claim": True,
                "guard_full_training_unlocked": False,
                "audit_path": str(audit_path.resolve()) if audit_path is not None else None,
            }
        )
    except Exception as exc:
        if require_runtime:
            raise
        summary["runtime_contract"] = "skipped"
        summary["runtime_skip_reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        if old_enabled is None:
            os.environ.pop("BVR_TWB_POSTPROCESS_AUDIT", None)
        else:
            os.environ["BVR_TWB_POSTPROCESS_AUDIT"] = old_enabled
        if old_path is None:
            os.environ.pop("BVR_TWB_POSTPROCESS_AUDIT_PATH", None)
        else:
            os.environ["BVR_TWB_POSTPROCESS_AUDIT_PATH"] = old_path

    summary["all_required_postprocess_audit_passed"] = bool(
        summary.get("runtime_contract") == "passed"
        and summary.get("explains_365940_predictions")
        and summary.get("pre_nms_selected_count") == 2000
        and summary.get("guard_runtime_contract") == "passed"
        and summary.get("guard_active") is True
        and summary.get("guard_pre_nms_selected_count", 10**9) <= 512
        and summary.get("guard_class_candidate_count_before_global_topk", 10**9) <= 32 * 19
    )
    if require_runtime and not summary["all_required_postprocess_audit_passed"]:
        summary["blocked"] = True
    if out is not None:
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Audit BVR-TWB post-processing proposal counts without checkpoint/evaluation.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--require-runtime", action="store_true")
    args = parser.parse_args()
    summary = run_postprocess_audit(args.out_dir, overwrite=args.overwrite, require_runtime=args.require_runtime)
    print(
        "BVR-TWB postprocess proposal audit: "
        f"runtime={summary.get('runtime_contract')} "
        f"flattened={summary.get('flattened_candidate_count')} "
        f"explains_365940={summary.get('explains_365940_predictions', False)} "
        f"guarded_pre_nms={summary.get('guard_pre_nms_selected_count')}"
    )


if __name__ == "__main__":
    main()
