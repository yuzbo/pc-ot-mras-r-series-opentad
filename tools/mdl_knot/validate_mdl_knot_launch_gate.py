from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.acquisition.mdl_knot import MDL_KNOT_ROUTE_LABEL  # noqa: E402


FORBIDDEN_ROUTE_DRIFT = (
    "EVENT_SURPRISE",
    "GLOBALRANK",
    "INTERVAL",
    "ORACLE",
    "BVR",
    "ABR",
    "COMBO",
)

ALLOWED_DEPLOY_SCOUT_SOURCES = {
    "raw_frame_motion_scout",
    "raw_frame_motion_scout_with_metadata_fallback",
    "frame_metadata_scout",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail-closed MDL-Knot local/precheck launch gate.")
    parser.add_argument("--config", required=True, help="MDL-Knot route config to validate.")
    parser.add_argument("--route-label", default=MDL_KNOT_ROUTE_LABEL)
    parser.add_argument("--precheck-summary", default=None)
    return parser.parse_args()


def _locked(message: str, code: int = 2) -> int:
    print(f"LOCKED: {message}")
    print("Still locked: remote_sync, Slurm, training, evaluation, tools/test.py, mAP/runtime/FLOPs/deploy/paper claims")
    return code


def _load_config(config_arg: str) -> tuple[Path, dict]:
    path = Path(config_arg)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(path)
    return path, runpy.run_path(str(path))


def _load_steps(dataset: dict, split: str) -> tuple[list, dict]:
    pipeline = dataset.get(split, {}).get("pipeline", [])
    load_steps = [step for step in pipeline if isinstance(step, dict) and step.get("type") == "LoadFrames"]
    if len(load_steps) != 1:
        raise ValueError(f"{split} must contain exactly one LoadFrames step, got {len(load_steps)}")
    return pipeline, load_steps[0]


def _validate_config(config_path: Path, cfg: dict) -> tuple[int, dict | None]:
    if cfg.get("route_label") != MDL_KNOT_ROUTE_LABEL:
        return _locked(f"config route_label mismatch: {cfg.get('route_label')}"), None
    route_status = str(cfg.get("route_status", ""))
    if "LOCAL_FINAL_CODE_CANDIDATE" not in route_status:
        return _locked(f"route_status is not a local final-code candidate: {route_status}"), None
    upper_text = config_path.read_text(encoding="utf-8").replace(MDL_KNOT_ROUTE_LABEL, "").upper()
    hits = [token for token in FORBIDDEN_ROUTE_DRIFT if token in upper_text]
    if hits:
        return _locked(f"forbidden route drift tokens in config text: {hits}"), None

    acq = cfg.get("mdl_knot_acquisition", {})
    if acq.get("route_label") != MDL_KNOT_ROUTE_LABEL:
        return _locked("mdl_knot_acquisition route_label mismatch"), None
    if acq.get("deploy_scout_source") not in ALLOWED_DEPLOY_SCOUT_SOURCES:
        return _locked(f"deploy_scout_source is not formal deploy-visible: {acq.get('deploy_scout_source')}"), None
    if acq.get("deploy_scout_source") == "fallback_synthetic_precheck_only":
        return _locked("fallback synthetic scout is not allowed for final-code config"), None
    if acq.get("synthetic_fallback_allowed") is not False:
        return _locked("synthetic fallback must be disabled for the final-code config"), None
    if acq.get("real_scout_unavailable") is True:
        return _locked("real_scout_unavailable must not be true for final-code config"), None
    for key in ("no_val_test_gt_selector", "no_teacher", "no_prediction_cache", "no_dense_raw_backbone_handoff"):
        if acq.get(key) is not True:
            return _locked(f"acquisition safety flag is not true: {key}"), None
    if not (
        acq.get("no_metric_claims") is True
        and acq.get("no_runtime_claims") is True
        and acq.get("no_deploy_claims") is True
        and acq.get("no_paper_claims") is True
    ):
        return _locked("metric/runtime/deploy/paper claims must remain locked"), None

    dataset = cfg.get("dataset", {})
    load_methods = {}
    bridges = {}
    scout_sources = {}
    safety = {}
    for split in ("train", "val", "test"):
        try:
            pipeline, load = _load_steps(dataset, split)
        except ValueError as exc:
            return _locked(str(exc)), None
        step_types = [step.get("type") for step in pipeline if isinstance(step, dict)]
        if "mmaction.DecordDecode" not in step_types:
            return _locked(f"{split} pipeline has no DecordDecode step"), None
        if pipeline.index(load) >= step_types.index("mmaction.DecordDecode"):
            return _locked(f"{split} LoadFrames must run before DecordDecode"), None
        load_methods[split] = load.get("method")
        bridges[split] = load.get("mdl_knot_bridge")
        scout_sources[split] = load.get("mdl_knot_deploy_scout_source")
        if load.get("method") != "mdl_knot_dynamic_subsample":
            return _locked(f"{split} LoadFrames method is not mdl_knot_dynamic_subsample"), None
        if load.get("mdl_knot_bridge") != "fixed_pad":
            return _locked(f"{split} MDL bridge is not fixed_pad"), None
        if load.get("mdl_knot_deploy_scout_source") != acq.get("deploy_scout_source"):
            return _locked(f"{split} deploy scout source does not match acquisition config"), None
        if load.get("mdl_knot_allow_synthetic_fallback") is not False:
            return _locked(f"{split} synthetic fallback is not fail-closed"), None
        safety[split] = {}
        for key in (
            "mdl_knot_no_gt_selector",
            "mdl_knot_no_teacher",
            "mdl_knot_no_prediction_cache",
            "mdl_knot_no_dense_raw_backbone_handoff",
        ):
            safety[split][key] = load.get(key) is True
            if safety[split][key] is not True:
                return _locked(f"{split} config safety check failed: {key}"), None

    evidence = {
        "config_path": str(config_path),
        "route_label": cfg.get("route_label"),
        "route_status": route_status,
        "deploy_scout_source": acq.get("deploy_scout_source"),
        "synthetic_fallback_allowed": acq.get("synthetic_fallback_allowed"),
        "dataset_pipelines_use_mdl": all(method == "mdl_knot_dynamic_subsample" for method in load_methods.values()),
        "load_methods": load_methods,
        "bridges": bridges,
        "scout_sources": scout_sources,
        "safety": safety,
        "changed_surface": acq.get("changed_surface", {}),
        "no_metric_runtime_deploy_claims": True,
    }
    return 0, evidence


def _validate_precheck_summary(summary_arg: str, config_evidence: dict) -> int:
    summary_path = Path(summary_arg)
    if not summary_path.is_absolute():
        summary_path = ROOT / summary_path
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
    summary_config = summary.get("config_evidence", {})
    if summary_config.get("route_label") != MDL_KNOT_ROUTE_LABEL:
        return _locked("summary config route_label mismatch")
    if summary_config.get("dataset_pipelines_use_mdl") is not True:
        return _locked("dataset pipelines do not all use MDL method")
    if summary_config.get("forbidden_route_token_hits"):
        return _locked(f"forbidden route tokens in summary config: {summary_config.get('forbidden_route_token_hits')}")
    drift_tokens = [str(token).upper() for token in summary_config.get("drift_tokens", [])]
    if any("RANDOM_FIXED" in token for token in drift_tokens):
        return _locked(f"random-fixed drift token in summary config: {drift_tokens}")
    if any("C3" in token for token in drift_tokens):
        return _locked(f"C3 drift token in summary config: {drift_tokens}")
    if any("COMBO" in token for token in drift_tokens):
        return _locked(f"COMBO drift token in summary config: {drift_tokens}")
    if summary_config.get("deploy_scout_source") != config_evidence.get("deploy_scout_source"):
        return _locked("summary config deploy scout source does not match --config")
    if summary_config.get("synthetic_fallback_allowed") is not False:
        return _locked("summary config does not keep synthetic fallback disabled")
    locked_actions = summary.get("locked_actions", {})
    for key in ("remote_sync", "slurm", "training", "evaluation", "tools_test_py"):
        if locked_actions.get(key) is not True:
            return _locked(f"summary does not keep {key} locked")
    no_claims = summary.get("no_claims", {})
    for key in ("mAP", "runtime", "FLOPs", "deploy", "paper"):
        if no_claims.get(key) is not True:
            return _locked(f"summary does not explicitly lock {key} claims")
    return 0


def main() -> int:
    args = parse_args()
    if args.route_label != MDL_KNOT_ROUTE_LABEL:
        return _locked(f"route label mismatch: {args.route_label}")
    upper_label = args.route_label.upper()
    for token in FORBIDDEN_ROUTE_DRIFT:
        if token in upper_label:
            return _locked(f"forbidden route drift token in label: {token}")

    try:
        config_path, cfg = _load_config(args.config)
    except Exception as exc:
        return _locked(f"cannot load config: {exc}")
    status, evidence = _validate_config(config_path, cfg)
    if status != 0:
        return status
    if args.precheck_summary:
        status = _validate_precheck_summary(args.precheck_summary, evidence or {})
        if status != 0:
            return status
    print("PRECHECK_ONLY_REQUEST_ALLOWED")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    print("Still locked: remote_sync, Slurm, training, evaluation, tools/test.py, mAP/runtime/FLOPs/deploy/paper claims")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
