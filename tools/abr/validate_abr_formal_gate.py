from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from opentad.acquisition.abr import ABR_ROUTE_LABEL
from opentad.acquisition.abr.validators import (
    ABRValidationError,
    assert_no_forbidden_route_tokens,
    validate_formal_readiness_payload,
)

FORMAL_CONFIG_NAME = "input_abr_active_bracket_refinement_adapter_irregular_headv3_formal.py"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate ABR formal-readiness gate without unlocking training.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--formal-json")
    args = parser.parse_args(argv)

    try:
        decision = validate_formal_config(Path(args.config))
        if args.formal_json:
            payload = json.loads(Path(args.formal_json).read_text(encoding="utf-8"))
            decision["formal_payload"] = validate_formal_readiness_payload(payload)
    except ABRValidationError as exc:
        print(f"LOCKED: {exc}")
        return 1

    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


def validate_formal_config(config_path: Path) -> dict[str, Any]:
    config_path = _resolve_inside_repo(config_path)
    if config_path.name != FORMAL_CONFIG_NAME:
        raise ABRValidationError(f"formal config must be named {FORMAL_CONFIG_NAME}")
    source = config_path.read_text(encoding="utf-8")
    assert_no_forbidden_route_tokens(_scrub_route_label(source))
    namespace = runpy.run_path(str(config_path))
    assert_no_forbidden_route_tokens(_scrub_route_label(namespace))

    route = _require_mapping(namespace.get("abr_route"), "abr_route")
    gate = _require_mapping(namespace.get("formal_gate"), "formal_gate")
    loader = _require_mapping(namespace.get("abr_loader"), "abr_loader")
    loader_cfg = _require_mapping(loader.get("abr_config"), "abr_loader.abr_config")
    dataset = _require_mapping(namespace.get("dataset"), "dataset")

    for name, payload in (("abr_route", route), ("formal_gate", gate)):
        if payload.get("route_label") != ABR_ROUTE_LABEL:
            raise ABRValidationError(f"{name}.route_label must be ABR route label")
        if payload.get("method") != "abr_active_bracket_refinement":
            raise ABRValidationError(f"{name}.method must be abr_active_bracket_refinement")
        if payload.get("full_train_unlocked") is not False:
            raise ABRValidationError(f"{name}.full_train_unlocked must remain False")
        for claim_key in ("metric_claim", "sparse_compute_claim", "runtime_claim", "deploy_claim", "paper_claim"):
            if payload.get(claim_key) is not False:
                raise ABRValidationError(f"{name}.{claim_key} must remain False")

    if gate.get("allowed_next_action") != "FORMAL_REVIEW_PACKET_ONLY":
        raise ABRValidationError("formal_gate.allowed_next_action must remain FORMAL_REVIEW_PACKET_ONLY")
    if gate.get("require_deploy_visible_scout") is not True:
        raise ABRValidationError("formal gate must require deploy-visible scout")
    if gate.get("reject_precheck_fallback") is not True:
        raise ABRValidationError("formal gate must reject PRECHECK_ONLY fallback")
    if float(gate.get("min_first_round_bracket_recall", 0.0)) < 0.95:
        raise ABRValidationError("formal gate min_first_round_bracket_recall must be >= 0.95")
    if float(gate.get("min_first_round_transition_coverage", 0.0)) < 0.95:
        raise ABRValidationError("formal gate min_first_round_transition_coverage must be >= 0.95")
    if float(gate.get("max_first_round_temporal_coverage_fraction", 1.0)) > 0.70:
        raise ABRValidationError("formal gate max_first_round_temporal_coverage_fraction must be <= 0.70")

    if loader.get("method") != "abr_active_bracket_refinement":
        raise ABRValidationError("abr_loader.method must be abr_active_bracket_refinement")
    if loader_cfg.get("route_label") != ABR_ROUTE_LABEL:
        raise ABRValidationError("abr_loader.abr_config.route_label must be ABR route label")
    if loader_cfg.get("allow_diagnostic_fallback_scout") is not False:
        raise ABRValidationError("formal abr_config must disable diagnostic fallback scout")
    if str(loader_cfg.get("fallback_stage", "")).upper() == "PRECHECK_ONLY":
        raise ABRValidationError("formal abr_config must not use PRECHECK_ONLY fallback_stage")
    if loader_cfg.get("bracket_policy") != "deploy_visible_multiscale_graydiff_bracket_v2":
        raise ABRValidationError("formal abr_config must use deploy_visible_multiscale_graydiff_bracket_v2")
    if float(loader_cfg.get("first_round_max_temporal_coverage_fraction", 1.0)) > 0.70:
        raise ABRValidationError("formal abr_config first_round_max_temporal_coverage_fraction must be <= 0.70")

    for split_name in ("train", "val", "test"):
        pipeline = dataset.get(split_name, {}).get("pipeline", [])
        load_steps = [step for step in pipeline if isinstance(step, Mapping) and step.get("type") == "LoadFrames"]
        if len(load_steps) != 1:
            raise ABRValidationError(f"{split_name} pipeline must contain exactly one LoadFrames step")
        step = load_steps[0]
        if step.get("method") != "abr_active_bracket_refinement":
            raise ABRValidationError(f"{split_name} LoadFrames must use ABR method")
        step_cfg = step.get("abr_config", {})
        if not isinstance(step_cfg, Mapping) or step_cfg.get("allow_diagnostic_fallback_scout") is not False:
            raise ABRValidationError(f"{split_name} LoadFrames must disable diagnostic fallback")
        if split_name in {"val", "test"} and step.get("abr_allow_gt_after_selection") is not False:
            raise ABRValidationError(f"{split_name} must keep abr_allow_gt_after_selection=False")

    return {
        "allowed_next_action": "FORMAL_REVIEW_PACKET_ONLY",
        "config": str(config_path.relative_to(REPO_ROOT)),
        "formal_config_ok": True,
        "full_train_unlocked": False,
        "route_label": ABR_ROUTE_LABEL,
        "still_locked": [
            "FORMAL_FULL_TRAIN_PENDING_REAL_SCOUT_RECALL_EVIDENCE",
            "TOOLS_TEST_PY",
            "EVALUATION",
            "MAPPAPER_CLAIM",
            "RUNTIME_OR_SPARSE_COMPUTE_CLAIM",
            "DEPLOY_CLAIM",
        ],
    }


def _resolve_inside_repo(path: Path) -> Path:
    resolved = (REPO_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ABRValidationError(f"path must stay inside owned worktree: {resolved}") from exc
    return resolved


def _require_mapping(payload: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ABRValidationError(f"{name} must be a dict")
    return payload


def _scrub_route_label(payload: Any) -> Any:
    if isinstance(payload, str):
        return payload.replace(ABR_ROUTE_LABEL, "ABR_ROUTE_LABEL")
    if isinstance(payload, Mapping):
        return {key: _scrub_route_label(value) for key, value in payload.items() if not str(key).startswith("__")}
    if isinstance(payload, list):
        return [_scrub_route_label(value) for value in payload]
    if isinstance(payload, tuple):
        return tuple(_scrub_route_label(value) for value in payload)
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
