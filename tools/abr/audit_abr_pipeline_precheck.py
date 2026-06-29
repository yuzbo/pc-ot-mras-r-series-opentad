from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from opentad.acquisition.abr import ABRConfig, ABR_ROUTE_LABEL, select_active_bracket_refinement
from opentad.acquisition.abr.integration import apply_abr_to_results
from opentad.acquisition.abr.validators import (
    ABRValidationError,
    assert_no_forbidden_selection_inputs,
    assert_real_sparse_handoff,
)


def build_precheck_summary(require_torch: bool = True) -> dict:
    if require_torch:
        _require_torch()
    easy = [0.05] * 96
    rich = [0.03] * 12 + [0.86] * 6 + [0.11] * 9 + [0.88] * 8 + [0.07] * 20 + [0.82] * 7 + [0.04] * 34
    cfg = ABRConfig(k0=8, k1_cap=14, k2_cap=4, max_total_k=32, max_gap=16, target_frame_num=40)
    easy_result = select_active_bracket_refinement(96, scout_curve=easy, config=cfg)
    rich_result = select_active_bracket_refinement(96, scout_curve=rich, config=cfg)

    mock = {
        "video_name": "abr_precheck_mock",
        "total_frames": 96,
        "avg_fps": 25.0,
        "fps": 25.0,
        "duration": 3.84,
        "snippet_stride": 1,
        "window_size": 96,
        "feature_start_idx": 0,
        "feature_end_idx": 95,
    }
    out = apply_abr_to_results(mock, config=cfg, scout_curve=rich)
    handoff = out["abr_selection_ledger"]["handoff"]
    assert_real_sparse_handoff(handoff)
    assert_no_forbidden_selection_inputs(mock)

    nonzero_dense_window = list(range(120, 216))
    nonzero_mock = {
        "video_name": "abr_precheck_nonzero_window",
        "total_frames": 240,
        "avg_fps": 25.0,
        "fps": 25.0,
        "duration": 9.6,
        "snippet_stride": 1,
    }
    nonzero_out = apply_abr_to_results(
        nonzero_mock,
        config=cfg,
        scout_curve=rich,
        dense_window=nonzero_dense_window,
    )
    nonzero_handoff = nonzero_out["abr_selection_ledger"]["handoff"]
    assert_real_sparse_handoff(nonzero_handoff)
    nonzero_valid_k = int(nonzero_out["abr_selected_valid_k"])
    nonzero_local = nonzero_out["abr_selected_positions_window_local"].tolist()
    nonzero_global = nonzero_out["abr_selected_positions_original_dense"].tolist()
    if nonzero_global != [nonzero_dense_window[pos] for pos in nonzero_local]:
        raise ABRValidationError("nonzero dense-window local/global relationship failed")

    rejection_checks = []
    for split_name in ("val", "test"):
        try:
            apply_abr_to_results(
                {
                    "video_name": f"abr_precheck_{split_name}_gt_reject",
                    "total_frames": 64,
                    "avg_fps": 25.0,
                    "gt_segments": [[1.0, 2.0]],
                },
                config=cfg,
            )
        except ABRValidationError:
            rejection_checks.append(True)
        else:
            rejection_checks.append(False)

    dynamic_k_nonconstant = easy_result.valid_k != rich_result.valid_k
    summary = {
        "real_sparse_handoff_ok": True,
        "forbidden_inputs_ok": True,
        "nonzero_window_ok": True,
        "val_test_gt_rejection_ok": all(rejection_checks),
        "dynamic_k_nonconstant": bool(dynamic_k_nonconstant),
        "detector_forward_count": int(rich_result.cost.detector_forward_count),
        "easy_valid_k": int(easy_result.valid_k),
        "rich_valid_k": int(rich_result.valid_k),
        "rounds_used": int(rich_result.cost.rounds_used),
        "stop_reason": rich_result.cost.stop_reason,
        "pipeline_valid_k": int(out["abr_selected_valid_k"]),
        "pipeline_frame_count": int(out["frame_inds"].shape[0]),
        "nonzero_window_start": int(nonzero_handoff["dense_window_start"]),
        "nonzero_window_end": int(nonzero_handoff["dense_window_end"]),
        "nonzero_window_valid_k": nonzero_valid_k,
        "nonzero_window_first_local": int(nonzero_local[0]),
        "nonzero_window_first_global": int(nonzero_global[0]),
    }
    if not dynamic_k_nonconstant:
        raise ABRValidationError("dynamic K check failed: easy and rich cases selected identical K")
    if not all(rejection_checks):
        raise ABRValidationError("val/test GT rejection check failed")
    return {
        "route_label": ABR_ROUTE_LABEL,
        "method": "abr_active_bracket_refinement",
        "status": "PASS_PRECHECK_ONLY",
        "precheck_validated": True,
        "summary": summary,
    }


def _require_torch() -> None:
    proc = subprocess.run(
        [sys.executable, "-c", "import torch"],
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "torch import subprocess failed").strip()
        raise ABRValidationError(f"LOCKED: torch unavailable for ABR pipeline precheck: {detail}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--mock-only", action="store_true", help="Skip torch import and run selector/handoff checks only.")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    if out_dir.exists() and not args.overwrite:
        print(f"LOCKED: output directory exists: {out_dir}", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "abr_precheck_summary.json"

    try:
        payload = build_precheck_summary(require_torch=not args.mock_only)
        if args.mock_only:
            payload["status"] = "PASS_MOCK_PRECHECK_ONLY"
            payload["precheck_validated"] = False
            payload["mock_only"] = True
    except Exception as exc:
        payload = {
            "route_label": ABR_ROUTE_LABEL,
            "method": "abr_active_bracket_refinement",
            "status": "LOCKED",
            "precheck_validated": False,
            "error": str(exc),
            "summary": {
                "real_sparse_handoff_ok": False,
                "forbidden_inputs_ok": False,
                "nonzero_window_ok": False,
                "val_test_gt_rejection_ok": False,
                "dynamic_k_nonconstant": False,
                "detector_forward_count": 0,
            },
        }
        out_file.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    out_file.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
