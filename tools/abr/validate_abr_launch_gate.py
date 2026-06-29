from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from opentad.acquisition.abr import ABR_ROUTE_LABEL
from opentad.acquisition.abr.validators import (
    ABRValidationError,
    assert_no_forbidden_route_tokens,
    validate_launch_gate_payload,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precheck-json")
    parser.add_argument("--config")
    args = parser.parse_args(argv)
    try:
        if args.config:
            decision = validate_config(Path(args.config))
        elif args.precheck_json:
            payload = json.loads(Path(args.precheck_json).read_text(encoding="utf-8"))
            decision = validate_launch_gate_payload(payload)
        else:
            raise ABRValidationError("requires --config or --precheck-json")
    except ABRValidationError as exc:
        print(f"LOCKED: {exc}")
        return 1
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


def validate_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise ABRValidationError(f"config not found: {config_path}")
    source = config_path.read_text(encoding="utf-8")
    assert_no_forbidden_route_tokens(source)
    namespace = runpy.run_path(str(config_path))

    route = namespace.get("abr_route", {})
    loader = namespace.get("abr_loader", {})
    loader_cfg = dict(loader.get("abr_config", {}))
    if route.get("route_label") != ABR_ROUTE_LABEL:
        raise ABRValidationError("config abr_route has wrong ABR route label")
    if route.get("method") != "abr_active_bracket_refinement":
        raise ABRValidationError("config abr_route must use method=abr_active_bracket_refinement")
    if loader.get("method") != "abr_active_bracket_refinement":
        raise ABRValidationError("config abr_loader must use method=abr_active_bracket_refinement")
    if loader_cfg.get("route_label") != ABR_ROUTE_LABEL:
        raise ABRValidationError("config abr_loader.abr_config has wrong ABR route label")
    if loader_cfg.get("allow_diagnostic_fallback_scout") is not True:
        raise ABRValidationError("config is locked unless PRECHECK_ONLY diagnostic fallback is explicit")
    if str(loader_cfg.get("fallback_stage", "")).upper() != "PRECHECK_ONLY":
        raise ABRValidationError("config diagnostic fallback must be tagged PRECHECK_ONLY")
    if "precheck_only" not in str(route.get("claim_status", "")).lower():
        raise ABRValidationError("config claim_status must state precheck_only")

    dataset = namespace.get("dataset", {})
    for split_name in ("train", "val", "test"):
        pipeline = dataset.get(split_name, {}).get("pipeline", [])
        load_steps = [step for step in pipeline if step.get("type") == "LoadFrames"]
        if len(load_steps) != 1:
            raise ABRValidationError(f"{split_name} pipeline must contain exactly one LoadFrames step")
        step = load_steps[0]
        if step.get("method") != "abr_active_bracket_refinement":
            raise ABRValidationError(f"{split_name} LoadFrames does not use ABR method")
        if split_name in {"val", "test"} and step.get("abr_allow_gt_after_selection") is not False:
            raise ABRValidationError(f"{split_name} must keep abr_allow_gt_after_selection=False")

    return {
        "allowed_next_action": "LOCAL_PRECHECK_ONLY_VALIDATION",
        "route_label": ABR_ROUTE_LABEL,
        "still_locked": "REMOTE_SYNC_FULL_TRAIN_MAPPAPER_CLAIM",
    }


if __name__ == "__main__":
    raise SystemExit(main())
