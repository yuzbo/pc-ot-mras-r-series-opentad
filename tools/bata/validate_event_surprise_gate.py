from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


ROUTE_LABEL = "DIVERGENT_INNOVATION_EVENT_SURPRISE_DO_NOT_MERGE_WITH_C3"
GATE_NAME = "event_surprise_acquisition_gate"
META_KEY = "event_surprise_acquisition_plan"
ROUTE = "event_surprise_temporal_acquisition"

FORBIDDEN_CONFIG_TOKENS = (
    "pc_ot_mras_prebackbone_c3",
    "PCOTMRASPreBackboneFrameSelector",
    "PCOTMRASMotionTCNFrameScout",
    "PCOTMRASHybridFrameScout",
    "PCOTMRASTinyTransformerFrameScout",
    "C3-Pro",
    "C3_RS",
    "BH_SDC",
    "boundary_microscope",
    "BoundaryMicroscope",
    "frame_token_hybrid",
    "FrameTokenHybrid",
)

FORBIDDEN_TRUE_KEYS = (
    "remote_sync",
    "allow_remote_sync",
    "slurm",
    "allow_slurm",
    "gpu",
    "allow_gpu",
    "tools_train",
    "allow_tools_train",
    "tools_test",
    "allow_tools_test",
    "full_train",
    "allow_full_train",
    "detector_training",
    "allow_detector_training",
    "detector_map",
    "allow_detector_map",
    "formal_eval",
    "allow_formal_eval",
    "checkpoint_load",
    "allow_checkpoint_load",
    "resume",
    "allow_resume",
    "load_from",
    "allow_load_from",
    "raw_prediction",
    "allow_raw_prediction",
    "raw_predictions",
    "allow_raw_predictions",
    "raw_prediction_cache",
    "allow_raw_prediction_cache",
    "prediction_cache",
    "allow_prediction_cache",
    "load_from_raw_predictions",
    "allow_load_from_raw_predictions",
    "save_raw_prediction",
    "allow_save_raw_prediction",
    "save_raw_predictions",
    "allow_save_raw_predictions",
    "uses_teacher",
    "uses_oracle",
    "uses_test_gt",
    "uses_raw_prediction",
    "metric_claim",
    "allow_metric_claim",
    "metric_claim_allowed",
    "paper_claim",
    "allow_paper_claim",
    "paper_claim_allowed",
    "runtime_claim",
    "allow_runtime_claim",
    "flops_claim",
    "allow_flops_claim",
    "runtime_flops_claim",
    "allow_runtime_flops_claim",
    "runtime_flops_claim_allowed",
    "deploy_claim",
    "allow_deploy_claim",
    "deploy_claim_allowed",
)

REQUIRED_FALSE_KEYS = (
    "default_off",
    "launch_gate_passed",
    "allow_remote_sync",
    "allow_slurm",
    "allow_gpu",
    "allow_tools_train",
    "allow_tools_test",
    "allow_detector_training",
    "allow_detector_map",
    "allow_train_validation_map",
    "allow_long_training",
    "allow_full_train",
    "allow_raw_prediction",
    "allow_raw_predictions",
    "allow_metric_claim",
    "metric_claim_allowed",
    "allow_paper_claim",
    "paper_claim_allowed",
    "allow_runtime_claim",
    "runtime_flops_claim_allowed",
    "allow_deploy_claim",
    "deploy_claim_allowed",
)

REQUIRED_SCOPE_FALSE_KEYS = (
    "changes_dynamic_budget_policy",
    "changes_token_compression",
    "changes_adapter_backbone_neck",
    "changes_detector_head",
    "changes_loss_assignment",
    "changes_test_time_post_processing",
    "uses_pc_ot_mras_detector_bridge",
    "uses_test_gt",
    "uses_oracle",
    "uses_teacher",
    "uses_raw_prediction",
    "metric_claim_allowed",
    "deploy_claim_allowed",
    "runtime_flops_claim_allowed",
    "paper_claim_allowed",
)


class EventSurpriseGateError(RuntimeError):
    pass


def _as_plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _as_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_plain(item) for item in value]
    return value


def _get(mapping: Mapping[str, Any], key: str, default: Any = None) -> Any:
    if hasattr(mapping, "get"):
        return mapping.get(key, default)
    return default


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EventSurpriseGateError(message)


def _is_truthy(value: Any) -> bool:
    return bool(value) if value is not None else False


def _walk_forbidden_true(value: Any, *, location: str = "gate") -> list[str]:
    offenders: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            next_location = f"{location}.{key_text}"
            if key_text in FORBIDDEN_TRUE_KEYS and _is_truthy(item):
                offenders.append(next_location)
            offenders.extend(_walk_forbidden_true(item, location=next_location))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            offenders.extend(_walk_forbidden_true(item, location=f"{location}[{index}]"))
    return offenders


def validate_gate_payload(payload: Mapping[str, Any]) -> bool:
    payload = _as_plain(payload)
    _require(isinstance(payload, Mapping), "gate payload must be a mapping")
    _require(_get(payload, "route_label") == ROUTE_LABEL, "route_label mismatch")
    _require(_get(payload, "default_off") is True, "default_off must be true")
    _require(_get(payload, "launch_gate_passed") is False, "launch_gate_passed must be false")
    _require(tuple(_get(payload, "allowed_entrypoints", ())) == (), "allowed_entrypoints must be empty")

    for key in REQUIRED_FALSE_KEYS:
        if key == "default_off":
            continue
        _require(_get(payload, key) is False, f"{key} must be false")

    context = _get(payload, "entrypoint_gate_context")
    _require(isinstance(context, Mapping), "entrypoint_gate_context must be present")
    configured_forbidden = set(_get(context, "forbidden_true_keys", ()))
    missing = [key for key in FORBIDDEN_TRUE_KEYS if key not in configured_forbidden]
    _require(not missing, f"forbidden_true_keys missing: {missing}")
    _require(tuple(_get(context, "allowed_decisions", ())) == (), "allowed_decisions must be empty")

    offenders = _walk_forbidden_true(payload)
    _require(not offenders, f"forbidden true key(s): {', '.join(offenders)}")
    return True


def validate_config(config_path: str | Path) -> dict[str, Any]:
    try:
        from mmengine.config import Config
    except Exception as exc:  # pragma: no cover - environment guard
        raise EventSurpriseGateError(f"mmengine Config import failed: {exc}") from exc

    cfg = Config.fromfile(str(config_path))
    if not hasattr(cfg, GATE_NAME):
        raise EventSurpriseGateError(f"config missing {GATE_NAME}")
    gate = _as_plain(getattr(cfg, GATE_NAME))
    validate_gate_payload(gate)
    scope = _as_plain(getattr(cfg, "experiment_scope", {}))
    _require(isinstance(scope, Mapping), "experiment_scope must be present")
    _require(_get(scope, "route") == ROUTE, "experiment_scope.route mismatch")
    _require(_get(scope, "route_label") == ROUTE_LABEL, "experiment_scope.route_label mismatch")
    _require(_get(scope, "meta_key") == META_KEY, "experiment_scope.meta_key mismatch")
    _require(_get(scope, "protocol_family") == "event_surprise_sparse_acquisition_contract", "protocol family mismatch")
    _require(_get(scope, "changes_input_sampling") is True, "Event-Surprise must change input sampling")
    for key in REQUIRED_SCOPE_FALSE_KEYS:
        _require(_get(scope, key) is False, f"experiment_scope.{key} must be false")

    model = _as_plain(getattr(cfg, "model", {}))
    _require(isinstance(model, Mapping), "model must be present")
    selector = _get(model, "frame_selector")
    _require(isinstance(selector, Mapping), "model.frame_selector must be present")
    _require(_get(selector, "type") == "EventSurpriseTemporalAcquisitionSelector", "wrong frame_selector type")
    _require(_get(selector, "route_label") == ROUTE_LABEL, "frame_selector.route_label mismatch")
    _require(_get(selector, "meta_key") == META_KEY, "frame_selector.meta_key mismatch")
    _require(_get(selector, "input_layout") == "bct", "frame_selector.input_layout must be bct")
    _require(_get(selector, "remap_gt_to_selected_axis") is True, "frame_selector must remap GT to selected axis")
    _require(int(_get(selector, "target_len", 0)) > 0, "frame_selector.target_len must be positive")
    _require(int(_get(selector, "max_gap", 0)) > 0, "frame_selector.max_gap must be positive")
    neck = _get(model, "neck")
    if isinstance(neck, Mapping):
        _require(_get(neck, "type") != "PCOTMRASDetectorBridge", "Event-Surprise route must not require PCOTMRASDetectorBridge")

    inference = _as_plain(getattr(cfg, "inference", {}))
    _require(_get(inference, "load_from_raw_predictions") is False, "load_from_raw_predictions must be false")
    _require(_get(inference, "save_raw_prediction") is False, "save_raw_prediction must be false")

    text = Path(config_path).read_text(encoding="utf-8")
    resolved_text = cfg.pretty_text
    for token in FORBIDDEN_CONFIG_TOKENS:
        _require(token not in text, f"config contains forbidden route token: {token}")
        _require(token not in resolved_text, f"resolved config contains forbidden route token: {token}")
    return {
        "pass": True,
        "config": str(config_path),
        "route_label": gate["route_label"],
        "stage": gate.get("stage"),
        "selector": selector["type"],
        "input_layout": selector["input_layout"],
        "allowed_entrypoints": list(gate.get("allowed_entrypoints", ())),
        "forbidden_true_key_count": len(tuple(gate["entrypoint_gate_context"]["forbidden_true_keys"])),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Event-Surprise fail-closed gate config.")
    parser.add_argument("--config", required=True, help="Path to Event-Surprise config file.")
    args = parser.parse_args(argv)
    payload = validate_config(args.config)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
