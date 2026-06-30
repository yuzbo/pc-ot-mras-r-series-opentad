from __future__ import annotations

import argparse
import json
import math
import re
import runpy
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from opentad.acquisition.abr import ABR_ROUTE_LABEL
from opentad.acquisition.abr.validators import ABRValidationError, assert_no_forbidden_route_tokens

EXPECTED_BASE = "./input_abr_active_bracket_refinement_adapter_irregular_headv3.py"
SHORTDIAG_CONFIG_NAME = "input_abr_active_bracket_refinement_adapter_irregular_headv3_shortdiag.py"

FAILURE_MARKERS = (
    (re.compile(r"\bTraceback\b", re.IGNORECASE), "Traceback marker"),
    (re.compile(r"\bRuntimeError\b", re.IGNORECASE), "RuntimeError marker"),
    (re.compile(r"CUDA\s+out\s+of\s+memory|out\s+of\s+memory", re.IGNORECASE), "CUDA/OOM marker"),
    (re.compile(r"\bKilled\b", re.IGNORECASE), "Killed marker"),
    (re.compile(r"no\s+space\s+left\s+on\s+device", re.IGNORECASE), "no-space marker"),
    (re.compile(r"no\s+(cuda\s+)?gpu|gpu\s+not\s+available", re.IGNORECASE), "no-GPU marker"),
    (re.compile(r"non[-\s]?finite", re.IGNORECASE), "non-finite marker"),
    (re.compile(r"\bcost\s*[:=]\s*(?:nan|inf|-inf)\b", re.IGNORECASE), "non-finite cost marker"),
    (re.compile(r"(?<![A-Za-z])(?:nan|inf|-inf)(?![A-Za-z])", re.IGNORECASE), "NaN/Inf marker"),
)

EVAL_MARKERS = (
    (re.compile(r"tools[/\\]test\.py", re.IGNORECASE), "tools/test.py marker"),
    (re.compile(r"\bresult_detection(?:\.json)?\b", re.IGNORECASE), "result_detection marker"),
    (re.compile(r"Average-mAP|mAP\s+at\s+tIoU|mAP@", re.IGNORECASE), "mAP marker"),
    (re.compile(r"\beval_one_epoch\b|\bbuild_evaluator\b|\[Eval\]", re.IGNORECASE), "evaluation marker"),
)

CLAIM_MARKERS = (
    (re.compile(r"\bfull_train_unlocked\s*[:=]\s*True\b", re.IGNORECASE), "full_train_unlocked=True"),
    (re.compile(r"\bmetric_claim\s*[:=]\s*True\b", re.IGNORECASE), "metric_claim=True"),
    (re.compile(r"\bsparse_compute_claim\s*[:=]\s*True\b", re.IGNORECASE), "sparse_compute_claim=True"),
    (re.compile(r"\bruntime_claim\s*[:=]\s*True\b", re.IGNORECASE), "runtime_claim=True"),
    (re.compile(r"\bdeploy_claim\s*[:=]\s*True\b", re.IGNORECASE), "deploy_claim=True"),
    (re.compile(r"\bpaper_claim\s*[:=]\s*True\b", re.IGNORECASE), "paper_claim=True"),
    (re.compile(r"\bFULL_TRAIN_(?:UNLOCKED|ALLOWED|APPROVED)\b"), "full-train approval marker"),
    (re.compile(r"\bDEPLOY_(?:UNLOCKED|ALLOWED|APPROVED)\b"), "deploy approval marker"),
    (re.compile(r"\bPAPER_CLAIM_(?:UNLOCKED|ALLOWED|APPROVED)\b"), "paper-claim approval marker"),
)

ROUTE_DRIFT_MARKERS = (
    (re.compile(r"\bC3(?:[-_\s]?Pro)?\b", re.IGNORECASE), "C3 route drift marker"),
    (re.compile(r"\bC3_(?:MAINLINE|ORIGINAL)\b", re.IGNORECASE), "C3 route drift marker"),
    (re.compile(r"\bGlobalRank\b", re.IGNORECASE), "C3/GlobalRank route drift marker"),
    (re.compile(r"\bBVR\b", re.IGNORECASE), "BVR route drift marker"),
    (re.compile(r"\bMDL\b", re.IGNORECASE), "MDL route drift marker"),
    (re.compile(r"\bcombo\s+route\b|\bCOMBO_ROUTE_APPROVED\b", re.IGNORECASE), "combo route drift marker"),
    (re.compile(r"\bBOUNDARY_MICROSCOPE\b|Boundary\s+Microscope", re.IGNORECASE), "Boundary Microscope drift marker"),
)

LOSS_PATTERN = re.compile(
    r"\bLoss\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate ABR SHORT_DIAGNOSTIC_ONLY gate state.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-log", default=None)
    args = parser.parse_args(argv)

    try:
        decision = validate_shortdiag(Path(args.config), Path(args.train_log) if args.train_log else None)
    except ABRValidationError as exc:
        print(f"LOCKED: {exc}")
        return 1

    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


def validate_shortdiag(config_path: Path, train_log_path: Path | None = None) -> dict[str, Any]:
    config_path = _resolve_inside_repo(config_path)
    namespace = validate_config(config_path)
    log_summary = None
    if train_log_path is not None:
        log_summary = validate_train_log(_resolve_log_path(train_log_path))

    decision: dict[str, Any] = {
        "allowed_next_action": "ONE_EPOCH_TRAIN_LOSS_DIAGNOSTIC_ONLY",
        "config": str(config_path.relative_to(REPO_ROOT)),
        "diagnostic_only": True,
        "full_train_unlocked": False,
        "metric_claim": False,
        "sparse_compute_claim": False,
        "route_label": ABR_ROUTE_LABEL,
        "still_locked": [
            "FORMAL_FULL_TRAIN",
            "TOOLS_TEST_PY",
            "EVALUATION",
            "CHECKPOINT_CLAIM",
            "MAPPAPER_CLAIM",
            "RUNTIME_OR_SPARSE_COMPUTE_CLAIM",
            "DEPLOY_OR_REMOTE_SYNC",
        ],
    }
    if log_summary is not None:
        decision["train_log"] = log_summary
    _assert_config_namespace_claim_locked(namespace)
    return decision


def validate_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise ABRValidationError(f"config not found: {config_path}")
    if config_path.name != SHORTDIAG_CONFIG_NAME:
        raise ABRValidationError(f"shortdiag config must be named {SHORTDIAG_CONFIG_NAME}")

    source = config_path.read_text(encoding="utf-8")
    _assert_clean_text(source, context="config", allow_abr_route_label=True)
    namespace = runpy.run_path(str(config_path))
    assert_no_forbidden_route_tokens(_scrub_route_label(namespace))

    bases = namespace.get("_base_")
    if bases != [EXPECTED_BASE]:
        raise ABRValidationError(f"shortdiag config must extend exactly {EXPECTED_BASE}")

    base_path = (config_path.parent / EXPECTED_BASE).resolve()
    if not base_path.exists():
        raise ABRValidationError(f"base ABR config not found: {base_path}")
    base_source = base_path.read_text(encoding="utf-8")
    _assert_clean_text(base_source, context="base config", allow_abr_route_label=True)

    route = namespace.get("abr_route", {})
    gate = namespace.get("shortdiag_gate", {})
    workflow = namespace.get("workflow", {})
    inference = namespace.get("inference", {})
    post_processing = namespace.get("post_processing", {})

    _assert_mapping(route, "abr_route")
    _assert_mapping(gate, "shortdiag_gate")
    _assert_mapping(workflow, "workflow")
    _assert_mapping(inference, "inference")
    _assert_mapping(post_processing, "post_processing")

    for name, payload in (("abr_route", route), ("shortdiag_gate", gate)):
        if payload.get("route_label") != ABR_ROUTE_LABEL:
            raise ABRValidationError(f"{name}.route_label must be ABR route label")
        if payload.get("method") != "abr_active_bracket_refinement":
            raise ABRValidationError(f"{name}.method must be abr_active_bracket_refinement")
        if payload.get("diagnostic_only") is not True:
            raise ABRValidationError(f"{name}.diagnostic_only must be True")
        _assert_claim_locks(payload, name)

    if gate.get("allowed_next_action") != "ONE_EPOCH_TRAIN_LOSS_DIAGNOSTIC_ONLY":
        raise ABRValidationError("shortdiag_gate.allowed_next_action must stay one-epoch diagnostic only")
    if gate.get("tools_test_py_allowed") is not False:
        raise ABRValidationError("shortdiag_gate.tools_test_py_allowed must be False")
    if gate.get("evaluation_allowed") is not False:
        raise ABRValidationError("shortdiag_gate.evaluation_allowed must be False")
    if gate.get("checkpoint_allowed") is not False:
        raise ABRValidationError("shortdiag_gate.checkpoint_allowed must be False")
    if gate.get("result_detection_allowed") is not False:
        raise ABRValidationError("shortdiag_gate.result_detection_allowed must be False")
    if int(gate.get("max_epochs", -1)) != 1:
        raise ABRValidationError("shortdiag_gate.max_epochs must be 1")
    if gate.get("n16r4_child_gpu_context_only") is not True:
        raise ABRValidationError("shortdiag N16R4 gate must require an existing child GPU context")

    if int(workflow.get("end_epoch", -1)) != 1:
        raise ABRValidationError("workflow.end_epoch must be 1")
    if workflow.get("disable_checkpoint") is not True:
        raise ABRValidationError("workflow.disable_checkpoint must be True")
    if int(workflow.get("checkpoint_interval", 0)) <= 1:
        raise ABRValidationError("workflow.checkpoint_interval must not permit epoch-1 checkpoint")
    if int(workflow.get("val_start_epoch", 0)) <= int(workflow.get("end_epoch", 1)):
        raise ABRValidationError("workflow.val_start_epoch must keep validation/eval unreachable")
    if int(workflow.get("val_loss_interval", 0)) > 0:
        raise ABRValidationError("workflow.val_loss_interval must disable val loss")
    if int(workflow.get("val_eval_interval", 0)) > 0:
        raise ABRValidationError("workflow.val_eval_interval must disable eval")
    if inference.get("save_raw_prediction") is not False:
        raise ABRValidationError("inference.save_raw_prediction must be False")
    if inference.get("load_from_raw_predictions") is not False:
        raise ABRValidationError("inference.load_from_raw_predictions must be False")
    if post_processing.get("save_dict") is not False:
        raise ABRValidationError("post_processing.save_dict must be False")

    return namespace


def validate_train_log(train_log_path: Path) -> dict[str, Any]:
    if not train_log_path.exists():
        raise ABRValidationError(f"train log not found: {train_log_path}")
    text = train_log_path.read_text(encoding="utf-8", errors="replace")
    _assert_clean_text(text, context="train log", allow_abr_route_label=True)

    losses = [float(match.group(1)) for match in LOSS_PATTERN.finditer(text)]
    finite_losses = [loss for loss in losses if math.isfinite(loss)]
    if not finite_losses:
        raise ABRValidationError("train log must contain at least one finite Loss=... line")

    return {
        "path": _display_path(train_log_path),
        "finite_loss_count": len(finite_losses),
        "last_loss": finite_losses[-1],
        "diagnostic_only": True,
        "full_train_unlocked": False,
        "metric_claim": False,
        "sparse_compute_claim": False,
    }


def _resolve_inside_repo(path: Path) -> Path:
    resolved = (REPO_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ABRValidationError(f"path must stay inside owned worktree: {resolved}") from exc
    return resolved


def _resolve_log_path(path: Path) -> Path:
    return (REPO_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


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


def _assert_mapping(payload: Any, name: str) -> None:
    if not isinstance(payload, Mapping):
        raise ABRValidationError(f"{name} must be a dict")


def _assert_config_namespace_claim_locked(namespace: Mapping[str, Any]) -> None:
    _assert_claim_locks(namespace.get("abr_route", {}), "abr_route")
    _assert_claim_locks(namespace.get("shortdiag_gate", {}), "shortdiag_gate")


def _assert_claim_locks(payload: Mapping[str, Any], name: str) -> None:
    required_false = (
        "full_train_unlocked",
        "metric_claim",
        "sparse_compute_claim",
        "runtime_claim",
        "deploy_claim",
        "paper_claim",
    )
    for key in required_false:
        if payload.get(key) is not False:
            raise ABRValidationError(f"{name}.{key} must be False")


def _assert_clean_text(text: str, context: str, allow_abr_route_label: bool) -> None:
    scrubbed = text
    if allow_abr_route_label:
        scrubbed = scrubbed.replace(ABR_ROUTE_LABEL, "ABR_ROUTE_LABEL")
        scrubbed = scrubbed.replace("DO_NOT_MERGE_WITH_C3", "DO_NOT_MERGE_WITH_ROUTE")

    for pattern, reason in FAILURE_MARKERS + EVAL_MARKERS + CLAIM_MARKERS + ROUTE_DRIFT_MARKERS:
        if pattern.search(scrubbed):
            raise ABRValidationError(f"{context} contains forbidden {reason}")


if __name__ == "__main__":
    raise SystemExit(main())
