from __future__ import annotations

import argparse
import json
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.acquisition.mdl_knot import (  # noqa: E402
    MDL_KNOT_ROUTE_LABEL,
    MDLKnotConfig,
    build_synthetic_scout_curve,
    generate_matched_controls,
    greedy_mdl_knot_select,
    validate_knot_ledger,
    validate_real_sparse_handoff,
)


CONFIG_PATH = ROOT / "configs" / "adatad" / "thumos" / "input_mdl_knot_dynamic_adapter_irregular_headv3.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MDL-Knot local synthetic/offline precheck.")
    parser.add_argument("--out-dir", required=True, help="Directory for JSON ledgers and summary.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory.")
    parser.add_argument("--max-k", type=int, default=56)
    return parser.parse_args()


def _check_optional_imports() -> dict:
    status = {}
    torch_probe = subprocess.run([sys.executable, "-c", "import torch"], text=True, capture_output=True)
    if torch_probe.returncode == 0:
        status["torch"] = "available"
    else:
        tail = (torch_probe.stderr or torch_probe.stdout or "").strip()[-500:]
        status["torch"] = f"unavailable: {tail}"
    try:
        import opentad  # noqa: F401

        status["opentad_namespace"] = "available"
    except Exception as exc:
        status["opentad_namespace"] = f"unavailable: {exc}"
    status["real_opentad_training_path"] = "locked_not_run"
    return status


def _collect_config_evidence() -> dict:
    cfg = runpy.run_path(str(CONFIG_PATH))
    dataset = cfg.get("dataset", {})
    load_methods = {}
    bridges = {}
    safety = {}
    for split in ("train", "val", "test"):
        pipeline = dataset.get(split, {}).get("pipeline", [])
        load_steps = [step for step in pipeline if isinstance(step, dict) and step.get("type") == "LoadFrames"]
        load = load_steps[0] if load_steps else {}
        load_methods[split] = load.get("method")
        bridges[split] = load.get("mdl_knot_bridge")
        safety[split] = {
            "no_gt_selector": load.get("mdl_knot_no_gt_selector") is True,
            "no_teacher": load.get("mdl_knot_no_teacher") is True,
            "no_prediction_cache": load.get("mdl_knot_no_prediction_cache") is True,
            "no_dense_raw_backbone_handoff": load.get("mdl_knot_no_dense_raw_backbone_handoff") is True,
            "synthetic_fallback_disabled": load.get("mdl_knot_allow_synthetic_fallback") is False,
            "load_before_decode": (
                len(load_steps) == 1
                and any(step.get("type") == "mmaction.DecordDecode" for step in pipeline if isinstance(step, dict))
                and pipeline.index(load) < [step.get("type") for step in pipeline].index("mmaction.DecordDecode")
            ),
        }

    acq = cfg.get("mdl_knot_acquisition", {})
    forbidden_hits = []
    forbidden_tokens = ("C3_MAINLINE", "C3_ORIGINAL", "GLOBALRANK", "INTERVAL", "BVR", "ABR", "ORACLE", "COMBO")
    text = CONFIG_PATH.read_text(encoding="utf-8")
    text = text.replace(MDL_KNOT_ROUTE_LABEL, "")
    for token in forbidden_tokens:
        if token in text.upper():
            forbidden_hits.append(token)
    active_pipeline_text = json.dumps(dataset, sort_keys=True).upper()
    drift_tokens = []
    if "RANDOM_FIXED_SUBSAMPLE" in active_pipeline_text:
        drift_tokens.append("RANDOM_FIXED_SUBSAMPLE")
    if "C3_" in active_pipeline_text or "C3-" in active_pipeline_text:
        drift_tokens.append("C3_ACTIVE_PIPELINE")

    return {
        "config_path": str(CONFIG_PATH),
        "route_label": cfg.get("route_label"),
        "dataset_pipelines_use_mdl": all(method == "mdl_knot_dynamic_subsample" for method in load_methods.values()),
        "load_methods": load_methods,
        "bridges": bridges,
        "safety": safety,
        "forbidden_route_token_hits": forbidden_hits,
        "drift_tokens": drift_tokens,
        "deploy_scout_source": acq.get("deploy_scout_source", "unknown"),
        "real_scout_unavailable": bool(acq.get("real_scout_unavailable", False)),
        "synthetic_fallback_allowed": bool(acq.get("synthetic_fallback_allowed", False)),
        "no_metric_runtime_deploy_claims": (
            acq.get("no_metric_claims") is True
            and acq.get("no_runtime_claims") is True
            and acq.get("no_deploy_claims") is True
            and acq.get("no_paper_claims") is True
        ),
    }


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if out_dir.exists():
        if not args.overwrite:
            print(f"LOCKED: output directory exists: {out_dir}")
            return 2
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = MDLKnotConfig(route_label=MDL_KNOT_ROUTE_LABEL, max_k=args.max_k, target_weighted_error=0.012)
    patterns = ["stable_background", "sharp_transition", "two_islands", "short_islands"]
    cases = {}
    ledgers = {}
    for pattern in patterns:
        curve = build_synthetic_scout_curve(pattern, dense_t=128)
        ledger = greedy_mdl_knot_select(curve, cfg, video_id=pattern)
        validate_knot_ledger(ledger)
        dense = list(range(curve.dense_t))
        selected = [dense[pos] for pos in ledger.selected_positions]
        validate_real_sparse_handoff(
            batch={"selected_inputs": selected, "dense_inputs": dense, "meta": ledger.to_sparse_meta().to_dict()},
            ledger=ledger,
        )
        controls = generate_matched_controls(curve, ledger, seed=13)
        for control in controls.values():
            validate_knot_ledger(control)
        ledgers[pattern] = ledger.to_dict()
        cases[pattern] = {
            "valid_k": ledger.valid_k,
            "max_gap": ledger.max_gap,
            "gap_p95": ledger.gap_p95,
            "weighted_reconstruction_error": ledger.weighted_reconstruction_error,
            "stop_reason": ledger.stop_reason,
            "roles": sorted(set(ledger.selected_roles)),
            "control_valid_k": {name: item.valid_k for name, item in controls.items()},
        }

    (out_dir / "mdl_knot_ledgers.json").write_text(json.dumps(ledgers, indent=2), encoding="utf-8")
    summary = {
        "route_label": MDL_KNOT_ROUTE_LABEL,
        "validated": True,
        "validation_scope": "synthetic_offline_real_sparse_handoff_only",
        "no_claims": {
            "mAP": True,
            "runtime": True,
            "FLOPs": True,
            "deploy": True,
            "paper": True,
        },
        "locked_actions": {
            "remote_sync": True,
            "slurm": True,
            "training": True,
            "evaluation": True,
            "tools_test_py": True,
        },
        "optional_imports": _check_optional_imports(),
        "config_evidence": _collect_config_evidence(),
        "cases": cases,
    }
    summary_path = out_dir / "mdl_knot_precheck_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"VALIDATED_PRECHECK_SUMMARY={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
