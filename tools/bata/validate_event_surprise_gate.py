from __future__ import annotations

import argparse
import hashlib
import json
import sys
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

FULL_TRAIN_GATE_ALLOWED_KEYS = frozenset(
    {
        "gate_type",
        "route",
        "route_label",
        "action",
        "run_tag",
        "user",
        "coordinator_override_statement",
        "allow_full_train",
        "launch_gate_passed",
        "allowed_entrypoints",
        "decision",
        "config",
        "config_stage",
        "review_status",
        "reviewed_impl_commit",
        "selector_type",
        "no_c3_mixing",
        "no_gt_teacher_cache_leakage",
        "timestamp",
        "active_sha256_manifest_sha256",
        "expected_active_sha256_manifest_sha256",
        "resolved_config_sha256",
        "expected_resolved_config_sha256",
        "allow_slurm",
        "allow_gpu",
        "allow_tools_train",
        "allow_detector_training",
        "allow_train_validation_map",
        "allow_long_training",
        "tools_test",
        "allow_tools_test",
        "detector_map",
        "allow_detector_map",
        "formal_eval",
        "allow_formal_eval",
        "checkpoint_load",
        "allow_checkpoint_load",
        "resume",
        "allow_resume",
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
        "runtime_flops_claim",
        "runtime_flops_claim_allowed",
        "deploy_claim",
        "deploy_claim_allowed",
    }
)

FULL_TRAIN_DECISION = "ALLOW_EVENT_SURPRISE_FULL_TRAIN_CANDIDATE"
FULL_TRAIN_COORDINATOR_OVERRIDE_STATEMENT = "USER_REQUESTED_NORMAL_SPEED_EVENT_SURPRISE_FOLLOWUP_PRO_REQUIRED"
FULL_TRAIN_CONFIG = "configs/adatad/thumos/event_surprise_temporal_acquisition_full_train_candidate_n16r4.py"
FULL_TRAIN_REQUIRED_TRUE_KEYS = ()
FULL_TRAIN_REQUIRED_FALSE_KEYS = (
    "allow_slurm",
    "allow_gpu",
    "allow_tools_train",
    "allow_detector_training",
    "allow_train_validation_map",
    "allow_long_training",
    "allow_full_train",
    "launch_gate_passed",
    "tools_test",
    "allow_tools_test",
    "detector_map",
    "allow_detector_map",
    "formal_eval",
    "allow_formal_eval",
    "checkpoint_load",
    "allow_checkpoint_load",
    "resume",
    "allow_resume",
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
    "runtime_flops_claim",
    "runtime_flops_claim_allowed",
    "deploy_claim",
    "deploy_claim_allowed",
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


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_truthy(value: Any) -> bool:
    return bool(value) if value is not None else False


def _walk_forbidden_true(
    value: Any,
    *,
    location: str = "gate",
    forbidden_keys: tuple[str, ...] = FORBIDDEN_TRUE_KEYS,
) -> list[str]:
    offenders: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            next_location = f"{location}.{key_text}"
            if key_text in forbidden_keys and _is_truthy(item):
                offenders.append(next_location)
            offenders.extend(_walk_forbidden_true(item, location=next_location, forbidden_keys=forbidden_keys))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            offenders.extend(
                _walk_forbidden_true(item, location=f"{location}[{index}]", forbidden_keys=forbidden_keys)
            )
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


def validate_full_train_config_gate_payload(payload: Mapping[str, Any]) -> bool:
    payload = _as_plain(payload)
    _require(isinstance(payload, Mapping), "gate payload must be a mapping")
    _require(_get(payload, "route") == ROUTE, "route mismatch")
    _require(_get(payload, "route_label") == ROUTE_LABEL, "route_label mismatch")
    _require(_get(payload, "stage") == "full_train_candidate_locked", "stage mismatch")
    _require(_get(payload, "default_off") is True, "default_off must be true")
    _require(_get(payload, "formal_train_candidate") is True, "formal_train_candidate must be true")
    _require(_get(payload, "full_train_candidate") is True, "full_train_candidate must be true")
    _require(_get(payload, "requires_launch_gate") is True, "requires_launch_gate must be true")
    _require(_get(payload, "launch_gate_passed") is False, "launch_gate_passed must be false")
    _require(tuple(_get(payload, "allowed_entrypoints", ())) == (), "allowed_entrypoints must be empty")
    _require(
        _get(payload, "review_status") == "FIXED_FOR_LOCAL_SMOKE_PENDING_FOLLOWUP_PRO",
        "review_status mismatch",
    )
    _require(_get(payload, "reviewed_impl_commit") == "FOLLOWUP_PRO_REQUIRED", "reviewed_impl_commit mismatch")
    _require(
        _get(payload, "selector_type") == "EventSurpriseTemporalAcquisitionSelector",
        "selector_type mismatch",
    )
    _require(_get(payload, "no_c3_mixing") is True, "no_c3_mixing must be true")
    _require(_get(payload, "no_gt_teacher_cache_leakage") is True, "no_gt_teacher_cache_leakage must be true")
    _require(_get(payload, "requires_followup_pro_review") is True, "requires_followup_pro_review must be true")

    for key in FULL_TRAIN_REQUIRED_FALSE_KEYS:
        _require(_get(payload, key) is False, f"{key} must be false")

    context = _get(payload, "entrypoint_gate_context")
    _require(isinstance(context, Mapping), "entrypoint_gate_context must be present")
    _require(_get(context, "required") is True, "entrypoint_gate_context.required must be true")
    _require(_get(context, "gate_json_env") == "EVENT_SURPRISE_ENTRYPOINT_GATE_JSON", "gate_json_env mismatch")
    _require(_get(context, "gate_sha256_env") == "EVENT_SURPRISE_ENTRYPOINT_GATE_SHA256", "gate_sha256_env mismatch")
    _require(
        _get(context, "active_manifest_sha256_env") == "EVENT_SURPRISE_ACTIVE_MANIFEST_SHA256",
        "active_manifest_sha256_env mismatch",
    )
    _require(
        _get(context, "resolved_config_sha256_env") == "EVENT_SURPRISE_RESOLVED_CONFIG_SHA256",
        "resolved_config_sha256_env mismatch",
    )
    _require(_get(context, "require_resolved_config_sha256") is True, "resolved config sha256 must be required")
    _require(tuple(_get(context, "allowed_decisions", ())) == (), "allowed_decisions must be empty")
    _require(_get(context, "reviewed_impl_commit") == "FOLLOWUP_PRO_REQUIRED", "context reviewed_impl_commit mismatch")
    _require(_get(context, "route_label") == ROUTE_LABEL, "context route_label mismatch")
    _require(
        _get(context, "selector_type") == "EventSurpriseTemporalAcquisitionSelector",
        "context selector_type mismatch",
    )
    _require(_get(context, "no_c3_mixing") is True, "context no_c3_mixing must be true")
    _require(
        _get(context, "no_gt_teacher_cache_leakage") is True,
        "context no_gt_teacher_cache_leakage must be true",
    )
    _require(_get(context, "requires_followup_pro_review") is True, "context follow-up Pro lock missing")
    _require(_get(context, "strict_payload_validation") is True, "strict payload validation must be true")
    _require(
        _get(context, "unknown_key_policy") == "reject_unknown_except_explicit_harmless_metadata",
        "unknown key policy mismatch",
    )
    _require(
        tuple(_get(context, "required_true_keys", ())) == FULL_TRAIN_REQUIRED_TRUE_KEYS,
        "required_true_keys mismatch",
    )
    _require(
        tuple(_get(context, "required_false_keys", ())) == FULL_TRAIN_REQUIRED_FALSE_KEYS,
        "required_false_keys mismatch",
    )

    exact = _get(context, "required_exact_values")
    _require(isinstance(exact, Mapping), "required_exact_values must be present")
    _require(_get(exact, "gate_type") == "event_surprise_launch_gate", "gate_type exact value mismatch")
    _require(_get(exact, "route") == ROUTE, "route exact value mismatch")
    _require(_get(exact, "route_label") == ROUTE_LABEL, "route_label exact value mismatch")
    _require(_get(exact, "action") == "full_train", "action exact value mismatch")
    _require(_get(exact, "config") == FULL_TRAIN_CONFIG, "config exact value mismatch")
    _require(_get(exact, "config_stage") == "full_train_candidate_locked", "config_stage exact value mismatch")
    _require(
        _get(exact, "coordinator_override_statement") == FULL_TRAIN_COORDINATOR_OVERRIDE_STATEMENT,
        "coordinator override exact value mismatch",
    )

    forbidden = tuple(_get(context, "forbidden_true_keys", ()))
    missing = [key for key in FULL_TRAIN_REQUIRED_FALSE_KEYS if key not in forbidden]
    _require(not missing, f"full-train forbidden_true_keys missing: {missing}")
    offenders = _walk_forbidden_true(payload, forbidden_keys=FULL_TRAIN_REQUIRED_FALSE_KEYS)
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
    if _get(gate, "full_train_candidate") is True:
        validate_full_train_config_gate_payload(gate)
    else:
        validate_gate_payload(gate)
    scope = _as_plain(getattr(cfg, "experiment_scope", {}))
    _require(isinstance(scope, Mapping), "experiment_scope must be present")
    _require(_get(scope, "route") == ROUTE, "experiment_scope.route mismatch")
    _require(_get(scope, "route_label") == ROUTE_LABEL, "experiment_scope.route_label mismatch")
    _require(_get(scope, "meta_key") == META_KEY, "experiment_scope.meta_key mismatch")
    _require(_get(scope, "protocol_family") == "event_surprise_sparse_acquisition_contract", "protocol family mismatch")
    _require(_get(scope, "route_isolation") == "no_c3_mixing", "route isolation mismatch")
    _require(
        _get(scope, "review_status") == "FIXED_FOR_LOCAL_SMOKE_PENDING_FOLLOWUP_PRO",
        "experiment_scope.review_status mismatch",
    )
    _require(
        _get(scope, "selected_axis_inference_mapping") == "selector_selected_axis_to_dense_window_axis",
        "selected-axis inference mapping mismatch",
    )
    _require(
        _get(scope, "preview_feature_status") == "loaded_dense_detector_input_prototype",
        "preview feature status mismatch",
    )
    _require(_get(scope, "changes_input_sampling") is True, "Event-Surprise must change input sampling")
    _require(
        _get(scope, "changes_test_time_post_processing") is True,
        "Event-Surprise must map selected-axis inference proposals back to dense window axis",
    )
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
        "allow_precheck_only": bool(gate.get("allow_precheck_only", False)),
        "formal_train_candidate": bool(gate.get("formal_train_candidate", False)),
        "full_train_candidate": bool(gate.get("full_train_candidate", False)),
        "allowed_entrypoints": list(gate.get("allowed_entrypoints", ())),
        "review_status": scope["review_status"],
        "full_train_locked_pending_followup_pro": bool(gate.get("requires_followup_pro_review", False)),
        "forbidden_true_key_count": len(tuple(gate["entrypoint_gate_context"]["forbidden_true_keys"])),
    }


def _normalize_action(action: str | None) -> str | None:
    if action is None:
        return None
    return str(action).strip().lower().replace("-", "_")


def _read_json_mapping(path: str | Path) -> Mapping[str, Any]:
    gate_path = Path(path)
    try:
        payload = json.loads(gate_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise EventSurpriseGateError(f"failed to read gate json {gate_path}: {exc}") from exc
    _require(isinstance(payload, Mapping), "gate json must be a mapping")
    return _as_plain(payload)


def validate_full_train_gate_payload(
    payload: Mapping[str, Any],
    *,
    active_manifest_sha256: str | None = None,
    resolved_config_sha256: str | None = None,
    run_tag: str | None = None,
) -> bool:
    payload = _as_plain(payload)
    unexpected = sorted(set(payload) - FULL_TRAIN_GATE_ALLOWED_KEYS)
    _require(not unexpected, f"launch gate has unexpected key(s): {unexpected}")
    _require(_get(payload, "gate_type") == "event_surprise_launch_gate", "launch gate type mismatch")
    _require(_get(payload, "route") == ROUTE, "launch gate route mismatch")
    _require(_get(payload, "route_label") == ROUTE_LABEL, "launch gate route_label mismatch")
    _require(_normalize_action(_get(payload, "action")) == "full_train", "launch gate action mismatch")
    _require(_get(payload, "decision") == FULL_TRAIN_DECISION, "launch gate decision mismatch")
    _require(_get(payload, "config") == FULL_TRAIN_CONFIG, "launch gate config mismatch")
    _require(_get(payload, "config_stage") == "full_train_candidate_locked", "launch gate config_stage mismatch")
    _require(isinstance(_get(payload, "user"), str) and bool(_get(payload, "user").strip()), "launch gate user missing")
    _require(
        _get(payload, "coordinator_override_statement") == FULL_TRAIN_COORDINATOR_OVERRIDE_STATEMENT,
        "launch gate coordinator override statement mismatch",
    )
    _require(isinstance(_get(payload, "run_tag"), str) and bool(_get(payload, "run_tag").strip()), "run_tag missing")
    _require("/" not in _get(payload, "run_tag") and "\\" not in _get(payload, "run_tag"), "run_tag must be path-safe")
    _require(_get(payload, "launch_gate_passed") is True, "launch gate must be passed")
    _require(_get(payload, "allow_full_train") is True, "launch gate must allow full_train")
    entrypoints = tuple(_get(payload, "allowed_entrypoints", ()))
    _require("full_train" in entrypoints, "full_train must be allowed entrypoint")
    _require("tools/train.py" in entrypoints, "tools/train.py must be allowed entrypoint")

    for key in FULL_TRAIN_REQUIRED_TRUE_KEYS:
        _require(_get(payload, key) is True, f"{key} must be true")
    for key in FULL_TRAIN_REQUIRED_FALSE_KEYS:
        _require(_get(payload, key) is False, f"{key} must be false")

    expected_manifest = _get(payload, "active_sha256_manifest_sha256") or _get(
        payload, "expected_active_sha256_manifest_sha256"
    )
    expected_resolved = _get(payload, "resolved_config_sha256") or _get(payload, "expected_resolved_config_sha256")
    _require(expected_manifest is not None, "launch gate missing active_sha256_manifest_sha256")
    _require(expected_resolved is not None, "launch gate missing resolved_config_sha256")
    if active_manifest_sha256 is not None:
        _require(
            expected_manifest == active_manifest_sha256,
            f"active_sha256_manifest_sha256 mismatch: expected {expected_manifest} got {active_manifest_sha256}",
        )
    if resolved_config_sha256 is not None:
        _require(
            expected_resolved == resolved_config_sha256,
            f"resolved_config_sha256 mismatch: expected {expected_resolved} got {resolved_config_sha256}",
        )
    if run_tag is not None:
        _require(_get(payload, "run_tag") == run_tag, f"run_tag mismatch: expected {_get(payload, 'run_tag')} got {run_tag}")
    return True


def validate_launch_action(
    config_path: str | Path,
    *,
    action: str,
    gate_json: str | Path | None = None,
    gate_sha256: str | None = None,
    active_manifest_sha256: str | None = None,
    resolved_config_sha256: str | None = None,
    run_tag: str | None = None,
) -> dict[str, Any]:
    action_name = _normalize_action(action)
    summary = validate_config(config_path)
    result = dict(summary)
    result["launch_action"] = action_name
    result["launch_allowed"] = False
    result["train_command_allowed"] = False

    if action_name in {"precheck", "precheck_only"}:
        _require(summary["stage"] == "local_precheck_only", "precheck_only launch requires local_precheck_only stage")
        _require(summary["allow_precheck_only"] is True, "precheck_only launch is not allowed by config gate")
        result["launch_action"] = "precheck_only"
        result["launch_allowed"] = True
        return result

    if action_name == "full_train":
        _require(summary["full_train_candidate"] is True, "full_train launch requires full_train_candidate config")
        _require(
            not summary.get("full_train_locked_pending_followup_pro", False),
            "full_train remains locked pending follow-up Pro review",
        )
        _require(gate_json is not None, "full_train launch requires --gate-json")
        if gate_sha256 is not None:
            actual_gate_sha256 = _sha256_file(gate_json)
            _require(
                actual_gate_sha256 == gate_sha256,
                f"gate json sha256 mismatch: expected {gate_sha256} got {actual_gate_sha256}",
            )
        payload = _read_json_mapping(gate_json)
        validate_full_train_gate_payload(
            payload,
            active_manifest_sha256=active_manifest_sha256,
            resolved_config_sha256=resolved_config_sha256,
            run_tag=run_tag,
        )
        result["launch_allowed"] = True
        result["train_command_allowed"] = True
        result["gate_json"] = str(gate_json)
        if gate_sha256 is not None:
            result["gate_sha256"] = gate_sha256
        result["active_sha256_manifest_sha256"] = _get(payload, "active_sha256_manifest_sha256")
        result["resolved_config_sha256"] = _get(payload, "resolved_config_sha256")
        result["run_tag"] = _get(payload, "run_tag")
        return result

    raise EventSurpriseGateError(f"unsupported launch action: {action}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Event-Surprise fail-closed gate config.")
    parser.add_argument("--config", required=True, help="Path to Event-Surprise config file.")
    parser.add_argument(
        "--action",
        choices=("precheck-only", "precheck", "full-train"),
        help="Optional fail-closed launcher action to validate before any command runs.",
    )
    parser.add_argument("--gate-json", help="Explicit external launch gate JSON for locked full-train actions.")
    parser.add_argument("--gate-sha256", help="Expected SHA256 of --gate-json for full-train actions.")
    parser.add_argument(
        "--active-manifest-sha256",
        help="Expected active SHA256 manifest digest that must match the full-train gate JSON.",
    )
    parser.add_argument(
        "--resolved-config-sha256",
        help="Expected resolved config digest that must match the full-train gate JSON.",
    )
    parser.add_argument("--run-tag", help="Expected fixed RUN_TAG that must match the full-train gate JSON.")
    args = parser.parse_args(argv)
    try:
        if args.action:
            payload = validate_launch_action(
                args.config,
                action=args.action,
                gate_json=args.gate_json,
                gate_sha256=args.gate_sha256,
                active_manifest_sha256=args.active_manifest_sha256,
                resolved_config_sha256=args.resolved_config_sha256,
                run_tag=args.run_tag,
            )
        else:
            payload = validate_config(args.config)
    except EventSurpriseGateError as exc:
        print(f"EventSurpriseGateError: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
