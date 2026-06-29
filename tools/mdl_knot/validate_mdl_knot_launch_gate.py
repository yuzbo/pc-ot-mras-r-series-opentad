from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.acquisition.mdl_knot import MDL_KNOT_ROUTE_LABEL  # noqa: E402


FORBIDDEN_ROUTE_DRIFT = (
    "GLOBALRANK",
    "INTERVAL",
    "TEACHER",
    "PREDICTION_CACHE",
    "VAL_GT",
    "TEST_GT",
    "ORACLE",
    "BVR",
    "ABR",
    "COMBO",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail-closed MDL-Knot PRECHECK_ONLY launch gate.")
    parser.add_argument("--route-label", required=True)
    parser.add_argument("--precheck-summary", required=True)
    return parser.parse_args()


def _locked(message: str, code: int = 2) -> int:
    print(f"LOCKED: {message}")
    print("Still locked: remote_sync, Slurm, training, evaluation, tools/test.py, stage, commit, push, mAP/runtime/FLOPs/deploy/paper claims")
    return code


def main() -> int:
    args = parse_args()
    if args.route_label != MDL_KNOT_ROUTE_LABEL:
        return _locked(f"route label mismatch: {args.route_label}")
    upper_label = args.route_label.upper()
    for token in FORBIDDEN_ROUTE_DRIFT:
        if token in upper_label:
            return _locked(f"forbidden route drift token in label: {token}")

    summary_path = Path(args.precheck_summary)
    if not summary_path.exists():
        return _locked(f"missing precheck summary: {summary_path}")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return _locked(f"cannot read precheck summary: {exc}")
    if summary.get("route_label") != MDL_KNOT_ROUTE_LABEL:
        return _locked("precheck summary route_label mismatch")
    if summary.get("validated") is not True:
        return _locked("precheck summary is not validated")
    cases = summary.get("cases", {})
    if not cases:
        return _locked("precheck summary has no validated cases")
    ks = {int(case.get("valid_k", -1)) for case in cases.values()}
    if len(ks) <= 1:
        return _locked("dynamic K distribution is constant in precheck")
    locked_actions = summary.get("locked_actions", {})
    for key in ("remote_sync", "slurm", "training", "evaluation", "tools_test_py", "stage_commit_push"):
        if locked_actions.get(key) is not True:
            return _locked(f"summary does not keep {key} locked")
    no_claims = summary.get("no_claims", {})
    for key in ("mAP", "runtime", "FLOPs", "deploy", "paper"):
        if no_claims.get(key) is not True:
            return _locked(f"summary does not explicitly lock {key} claims")
    config = summary.get("config_evidence")
    if not isinstance(config, dict):
        return _locked("missing config evidence in precheck summary")
    if config.get("route_label") != MDL_KNOT_ROUTE_LABEL:
        return _locked("config route_label mismatch")
    hits = config.get("forbidden_route_token_hits", [])
    if hits:
        return _locked(f"forbidden route tokens in config evidence: {hits}")
    drift_tokens = [str(token).upper() for token in config.get("drift_tokens", [])]
    if any("RANDOM_FIXED" in token for token in drift_tokens):
        return _locked(f"random-fixed drift token in config evidence: {drift_tokens}")
    if any("C3" in token for token in drift_tokens):
        return _locked(f"C3 drift token in config evidence: {drift_tokens}")
    if any("COMBO" in token for token in drift_tokens):
        return _locked(f"COMBO drift token in config evidence: {drift_tokens}")
    if config.get("dataset_pipelines_use_mdl") is not True:
        return _locked("dataset pipelines do not all use MDL method")
    methods = config.get("load_methods", {})
    for split in ("train", "val", "test"):
        if methods.get(split) != "mdl_knot_dynamic_subsample":
            return _locked(f"{split} LoadFrames method is not mdl_knot_dynamic_subsample")
    bridges = config.get("bridges", {})
    for split in ("train", "val", "test"):
        if bridges.get(split) != "fixed_pad":
            return _locked(f"{split} MDL bridge is not fixed_pad")
    safety = config.get("safety", {})
    for split in ("train", "val", "test"):
        split_safety = safety.get(split, {})
        for key in (
            "no_gt_selector",
            "no_teacher",
            "no_prediction_cache",
            "no_dense_raw_backbone_handoff",
            "load_before_decode",
        ):
            if split_safety.get(key) is not True:
                return _locked(f"{split} config safety check failed: {key}")
    if config.get("no_metric_runtime_deploy_claims") is not True:
        return _locked("config does not explicitly lock metric/runtime/deploy/paper claims")
    scout_source = config.get("deploy_scout_source")
    if scout_source not in ("fallback_synthetic_precheck_only", "real_deploy_visible_scout"):
        return _locked(f"unknown deploy scout source: {scout_source}")
    if scout_source == "fallback_synthetic_precheck_only" and config.get("real_scout_unavailable") is not True:
        return _locked("fallback scout summary must mark real_scout_unavailable")
    print("PRECHECK_ONLY_REQUEST_ALLOWED")
    print("Still locked: remote_sync, Slurm, training, evaluation, tools/test.py, stage, commit, push, mAP/runtime/FLOPs/deploy/paper claims")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
