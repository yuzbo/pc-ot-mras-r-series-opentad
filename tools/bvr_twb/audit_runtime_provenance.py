import argparse
import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.acquisition.bvr_twb.types import ROUTE_LABEL
from tools.bvr_twb.audit_sparse_forward_precheck import safe_prepare_output_dir


def _sha1(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _source_contract():
    path = ROOT / "opentad/datasets/transforms/end_to_end.py"
    text = path.read_text(encoding="utf-8")
    required_tokens = {
        "method_literal": 'elif self.method == "bvr_twb_dynamic_subsample":',
        "selection_call": "build_bvr_twb_open_tad_selection(",
        "bridge_mode": "build_adapter_fixed_length_padded_bridge(",
        "fail_closed_unknown_method": "Unsupported LoadFrames method",
    }
    missing = [name for name, token in required_tokens.items() if token not in text]
    if missing:
        raise ValueError(f"BVR-TWB LoadFrames source provenance missing tokens: {missing}")
    return {
        "source_file": str(path.resolve()),
        "source_sha1": _sha1(text),
        "required_tokens_present": sorted(required_tokens),
    }


def _probe_torch_runtime():
    proc = subprocess.run(
        [sys.executable, "-c", "import torch"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode == 0:
        return None
    message = (proc.stderr or proc.stdout).strip()
    if not message:
        message = f"torch import probe failed with exit code {proc.returncode}"
    return message.splitlines()[-1]


def _base_results(split="train", with_gt=True):
    dense_len = 64
    x = np.arange(dense_len, dtype=np.float32)
    result = {
        "video_name": f"bvr_runtime_provenance_{split}",
        "data_path": "mock",
        "total_frames": 128,
        "avg_fps": 30.0,
        "fps": 30.0,
        "duration": 128.0 / 30.0,
        "snippet_stride": 1,
        "window_size": dense_len,
        "feature_start_idx": 0,
        "feature_end_idx": dense_len - 1,
        "split": split,
        "bvr_twb_preview_actionness": np.clip(
            0.08 + 0.6 * np.exp(-((x - 28.0) ** 2) / (2.0 * 5.0**2)),
            0.0,
            1.0,
        ).astype(np.float32),
    }
    if with_gt:
        result["gt_segments"] = np.asarray([[24.0, 34.0]], dtype=np.float32)
        result["gt_labels"] = np.asarray([2], dtype=np.int32)
    return result


def _runtime_contract():
    from opentad.acquisition.bvr_twb.adapter_bridge import ADAPTER_FIXED_LENGTH_PADDED_BRIDGE
    from opentad.datasets.transforms.end_to_end import LoadFrames

    from opentad.acquisition.bvr_twb.open_tad_bridge import build_bvr_twb_open_tad_selection

    call_source = inspect.getsource(LoadFrames.__call__)
    selection_source = inspect.getsource(build_bvr_twb_open_tad_selection)
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
    if ledger.get("method") != "bvr_twb_dynamic_subsample":
        raise ValueError(f"BVR-TWB runtime dispatch missed, ledger method={ledger.get('method')}")
    if "build_bvr_twb_open_tad_selection(" not in call_source:
        raise ValueError("Loaded LoadFrames.__call__ does not call build_bvr_twb_open_tad_selection")
    if "selector_provenance" not in selection_source:
        raise ValueError("Loaded build_bvr_twb_open_tad_selection lacks selector provenance ledger")

    return {
        "runtime_loadframes_file": str(Path(inspect.getsourcefile(LoadFrames)).resolve()),
        "runtime_loadframes_call_sha1": _sha1(call_source),
        "runtime_selection_file": str(Path(inspect.getsourcefile(build_bvr_twb_open_tad_selection)).resolve()),
        "runtime_selection_sha1": _sha1(selection_source),
        "loadframes_call_contains_bvr_method": "bvr_twb_dynamic_subsample" in call_source,
        "loadframes_call_contains_selection_call": "build_bvr_twb_open_tad_selection(" in call_source,
        "dispatch_hit": True,
        "ledger_method": ledger.get("method"),
        "ledger_route_label": ledger.get("route_label"),
        "frame_inds_len": int(len(transformed["frame_inds"])),
        "mask_true_count": int(transformed["masks"].sum().item()),
        "raw_valid_k": int(ledger["valid_k"]),
        "detector_feature_valid_k": int(ledger["detector_feature_valid_k"]),
        "gt_native_axis_preserved": bool(np.allclose(transformed["gt_segments"], np.asarray([[24.0, 34.0]], dtype=np.float32))),
    }


def run_runtime_provenance(out_dir=None, overwrite=False, require_runtime=False):
    summary = {
        "route_label": ROUTE_LABEL,
        "audit": "bvr_twb_runtime_provenance",
        "no_training": True,
        "no_video_decode": True,
        "no_metric_claim": True,
        "no_runtime_or_flops_claim": True,
        "full_training_unlocked": False,
        "blocked": False,
    }
    summary.update(_source_contract())
    try:
        torch_error = _probe_torch_runtime()
        if torch_error is not None:
            raise RuntimeError(f"torch import probe failed: {torch_error}")
        summary.update(_runtime_contract())
        summary["runtime_contract"] = "passed"
        summary["runtime_skipped"] = False
    except Exception as exc:
        if require_runtime:
            raise
        summary["runtime_contract"] = "skipped"
        summary["runtime_skipped"] = True
        summary["runtime_skip_reason"] = f"{type(exc).__name__}: {exc}"
    summary["all_required_runtime_provenance_passed"] = bool(
        summary.get("runtime_contract") == "passed"
        and summary.get("loadframes_call_contains_bvr_method")
        and summary.get("loadframes_call_contains_selection_call")
        and summary.get("dispatch_hit")
    )
    if require_runtime and not summary["all_required_runtime_provenance_passed"]:
        summary["blocked"] = True
    if out_dir is not None:
        out = safe_prepare_output_dir(out_dir, overwrite=overwrite)
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Audit runtime provenance for BVR-TWB LoadFrames dispatch.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--require-runtime", action="store_true")
    args = parser.parse_args()
    summary = run_runtime_provenance(args.out_dir, overwrite=args.overwrite, require_runtime=args.require_runtime)
    print(
        "BVR-TWB runtime provenance: "
        f"runtime={summary['runtime_contract']} "
        f"dispatch_hit={summary.get('dispatch_hit', False)} "
        f"blocked={summary.get('blocked', False)}"
    )


if __name__ == "__main__":
    main()
