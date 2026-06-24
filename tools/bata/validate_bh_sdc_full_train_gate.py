from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from mmengine.config import Config


ROUTE = "bh_sdc_boundary_hazard_sparse_dense"
ROUTE_LABEL = "DIVERGENT_INNOVATION_BH_SDC_DO_NOT_MERGE_WITH_C3"
FULL_STAGE = "bh_sdc_boundary_hazard_sparse_dense_full_train_candidate_n16r4"
LOCAL_STAGE = "bh_sdc_boundary_hazard_sparse_dense_local_precheck"
LAUNCH_DECISION = "ALLOW_BH_SDC_N16R4_SYNC_AND_FULL_TRAIN_CANDIDATE_V1"
FOLLOWUP_PRO_STATUS = "FIXED_FOR_LOCAL_SMOKE_PENDING_FOLLOWUP_PRO"
REVIEWED_IMPL_COMMIT = "FOLLOWUP_PRO_REQUIRED_AFTER_BH_SDC_PRO_FIX"
SELECTOR_TYPE = "PCOTMRASBoundaryHazardSparseDenseFrameSelector"
COMPLETION_TYPE = "PCOTMRASBoundaryHazardSparseToDenseBridge"
REMOTE_WORKSPACE = "~/run/yuzibo/OpenTAD_Back_check"
SYNC_COMMAND = f"REMOTE_SYNC_TO_N16R4:{REMOTE_WORKSPACE}"
SLURM_SCRIPT = "scripts/run_bh_sdc_full_train_n16r4.sbatch"
SLURM_COMMAND = f"sbatch {SLURM_SCRIPT}"
TRAIN_COMMAND = (
    "python tools/train.py "
    "configs/adatad/thumos/bh_sdc_boundary_hazard_sparse_dense_full_train_candidate_n16r4.py --id 0"
)
COMMAND_WHITELIST = [SYNC_COMMAND, SLURM_COMMAND, TRAIN_COMMAND]
ACTION_COMMANDS = {
    "remote_sync_to_n16r4_workspace": SYNC_COMMAND,
    "slurm_submit_bh_sdc_n16r4": SLURM_COMMAND,
    "slurm_full_train_candidate": TRAIN_COMMAND,
}

FORBIDDEN_ROUTE_TOKENS = (
    "c3",
    "c3_pro",
    "c3-pro",
    "cnn_lite",
    "cnn-lite",
    "motion_tcn",
    "motion-tcn",
    "hybrid",
    "interval_packet",
    "global_rank",
    "global-rank",
    "physical_grid",
    "physical-grid",
    "dynamic_budget_guard",
    "dynamic-budget-guard",
)

LOCKED_FALSE_OR_ABSENT_GATE_KEYS = (
    "allow_remote_sync",
    "allow_slurm",
    "allow_gpu",
    "allow_tools_train",
    "allow_tools_test",
    "allow_detector_map",
    "allow_train_validation_map",
    "allow_long_training",
    "allow_full_train",
    "allow_dataset_access",
    "allow_pretrained_initialization",
    "allow_checkpoint_write",
    "allow_checkpoint_load",
    "allow_resume",
    "allow_raw_prediction_cache",
    "metric_claim_allowed",
    "paper_claim_allowed",
    "runtime_flops_claim_allowed",
    "deploy_claim_allowed",
    "allow_metric_claim",
    "allow_paper_claim",
)

FORBIDDEN_SCOPE_TRUE_KEYS = (
    "uses_p2",
    "uses_offline_ledger",
    "uses_teacher",
    "uses_test_gt",
    "uses_oracle",
    "uses_raw_prediction_cache",
    "metric_claim_allowed",
    "paper_claim_allowed",
    "runtime_flops_claim_allowed",
    "deploy_claim_allowed",
    "pro_code_or_launch_approval",
)

PAYLOAD_KEYS = {
    "schema_version",
    "decision",
    "explicit_user_pro_launch_decision",
    "pro_launch_gate_verdict",
    "pro_launch_gate_session",
    "route",
    "route_label",
    "stage",
    "reviewed_impl_commit",
    "launch_gate_commit",
    "pro_implementation_verdict",
    "pro_implementation_session",
    "final_read_only_review_verdict",
    "final_read_only_review_id",
    "resolved_config_sha256",
    "active_sha256_manifest_sha256",
    "command_whitelist",
    "remote_workspace",
    "remote_workspace_policy",
    "slurm_script",
    "slurm_partition",
    "max_gpus",
    "max_nodes",
    "max_time_hours",
    "max_epochs",
    "train_command",
    "allow_remote_sync",
    "allow_slurm",
    "allow_full_train",
    "allow_tools_train",
    "allow_tools_test",
    "allow_detector_map",
    "allow_metric_claim",
    "allow_paper_claim",
    "allow_runtime_flops_claim",
    "allow_deploy_claim",
    "no_gt_test_leakage_assertion",
    "no_teacher_or_oracle_assertion",
    "no_raw_prediction_cache_assertion",
    "uses_test_gt",
    "uses_val_test_teacher",
    "uses_oracle",
    "uses_raw_prediction_cache",
    "load_from_raw_predictions",
    "save_raw_prediction",
    "allow_checkpoint_load",
    "allow_pretrained_initialization",
    "allow_resume",
    "allow_checkpoint_write",
    "checkpoint_write_policy",
    "checkpoint_load_policy",
    "pretrained_initialization_policy",
    "resume_policy",
    "dataset_scope",
}

PAYLOAD_TRUE_KEYS = (
    "allow_remote_sync",
    "allow_slurm",
    "allow_full_train",
    "allow_tools_train",
    "allow_checkpoint_write",
    "no_gt_test_leakage_assertion",
    "no_teacher_or_oracle_assertion",
    "no_raw_prediction_cache_assertion",
)

PAYLOAD_FALSE_KEYS = (
    "allow_tools_test",
    "allow_detector_map",
    "allow_metric_claim",
    "allow_paper_claim",
    "allow_runtime_flops_claim",
    "allow_deploy_claim",
    "uses_test_gt",
    "uses_val_test_teacher",
    "uses_oracle",
    "uses_raw_prediction_cache",
    "load_from_raw_predictions",
    "save_raw_prediction",
    "allow_checkpoint_load",
    "allow_pretrained_initialization",
    "allow_resume",
)

HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolved_config_sha256(path: Path) -> str:
    cfg = Config.fromfile(path)
    return sha256_text(cfg.pretty_text + "\n")


def load_config_namespace(path: Path) -> dict[str, Any]:
    cfg = Config.fromfile(path)
    return cfg._cfg_dict.to_dict()


def _base_entries_from_source(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "_base_" for target in node.targets):
            continue
        value = ast.literal_eval(node.value)
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
    return []


def config_base_chain(path: Path) -> list[Path]:
    seen: set[Path] = set()
    chain: list[Path] = []

    def visit(item: Path) -> None:
        resolved = item.resolve()
        if resolved in seen or not resolved.exists():
            return
        seen.add(resolved)
        chain.append(resolved)
        for entry in _base_entries_from_source(resolved):
            base_path = Path(entry)
            if not base_path.is_absolute():
                base_path = resolved.parent / base_path
            visit(base_path)

    visit(path)
    return chain


def _normalized_token_text(path: Path) -> str:
    return str(path).replace("\\", "/").lower()


def forbidden_base_chain_hits(path: Path) -> list[str]:
    hits: list[str] = []
    for item in config_base_chain(path):
        if item.resolve() == path.resolve():
            continue
        text = _normalized_token_text(item)
        for token in FORBIDDEN_ROUTE_TOKENS:
            if token in text:
                hits.append(f"{item.name}:{token}")
    return hits


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _get_nested(mapping: dict[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _require_no_runtime_shortcuts(cfg: dict[str, Any], model: dict[str, Any]) -> None:
    backbone_custom = _get_nested(model, "backbone", "custom")
    if isinstance(backbone_custom, dict):
        _require(
            backbone_custom.get("pretrain") in (None, "", False),
            "BH-SDC launch candidate must not inherit a pretrained initialization path",
        )
    for key in ("load_from", "resume", "resume_from", "checkpoint", "checkpoint_path"):
        _require(cfg.get(key) in (None, "", False), f"resolved config must not set {key}")
    inference = cfg.get("inference", {})
    _require(inference.get("load_from_raw_predictions") is False, "raw prediction loading must be disabled")
    _require(inference.get("save_raw_prediction") is False, "raw prediction saving must be disabled")


def _validate_common_route_config(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    base_forbidden_hits = forbidden_base_chain_hits(path)
    _require(
        not base_forbidden_hits,
        f"resolved _base_ chain contains forbidden route/base token(s): {base_forbidden_hits}",
    )
    cfg = load_config_namespace(path)
    scope = cfg.get("experiment_scope")
    gate = cfg.get("bh_sdc_gate")
    model = cfg.get("model")
    _require(isinstance(scope, dict), "experiment_scope must be a dict")
    _require(isinstance(gate, dict), "bh_sdc_gate must be a dict")
    _require(isinstance(model, dict), "model must be a dict")
    _require(scope.get("route") == ROUTE, f"experiment_scope.route must be {ROUTE}")
    _require(scope.get("route_label") == ROUTE_LABEL, f"experiment_scope.route_label must be {ROUTE_LABEL}")
    _require(scope.get("route_family") == "BH_SDC_DIVERGENT_INNOVATION_ROUTE", "BH-SDC route family mismatch")
    _require(scope.get("combo_status") == "NO_COMBO_ROUTE_APPROVED", "BH-SDC must not be a combo route")
    _require(gate.get("route") == ROUTE, f"bh_sdc_gate.route must be {ROUTE}")
    _require(gate.get("route_label") == ROUTE_LABEL, f"bh_sdc_gate.route_label must be {ROUTE_LABEL}")
    _require(gate.get("requires_launch_gate") is True, "BH-SDC gate must require a launch gate")
    for key in FORBIDDEN_SCOPE_TRUE_KEYS:
        _require(scope.get(key) is not True, f"experiment_scope must keep {key}=false/absent before real results")

    frame_selector = model.get("frame_selector")
    token_compressor = model.get("token_compressor")
    _require(isinstance(frame_selector, dict), "model.frame_selector must be configured")
    _require(isinstance(token_compressor, dict), "model.token_compressor must be configured")
    _require(frame_selector.get("type") == SELECTOR_TYPE, f"frame_selector.type must be {SELECTOR_TYPE}")
    _require(token_compressor.get("type") == COMPLETION_TYPE, f"token_compressor.type must be {COMPLETION_TYPE}")
    dense_window_size = int(frame_selector.get("dense_window_size"))
    max_budget = int(frame_selector.get("max_budget"))
    min_budget = int(frame_selector.get("min_budget"))
    target_budget = int(frame_selector.get("target_budget"))
    _require(
        0 < min_budget < target_budget < max_budget <= dense_window_size,
        "must satisfy min_budget < target_budget < max_budget <= dense_window_size",
    )
    _require(int(frame_selector.get("probe_stride", 0)) > 0, "frame_selector.probe_stride must be positive")
    _require(
        float(frame_selector.get("probe_interpolation_temperature", 0.0)) > 0.0,
        "frame_selector.probe_interpolation_temperature must be positive",
    )
    _require(int(token_compressor.get("dense_window_size")) == dense_window_size, "bridge dense_window_size mismatch")
    _require(int(token_compressor.get("target_len")) == dense_window_size, "bridge target_len mismatch")
    _require(
        int(_get_nested(model, "backbone", "backbone", "total_frames")) == max_budget,
        "backbone total_frames must equal max_budget",
    )
    _require(
        int(_get_nested(model, "projection", "max_seq_len")) == dense_window_size,
        "projection max_seq_len must be dense_window_size",
    )
    _require_no_runtime_shortcuts(cfg, model)
    pretty_text = Config.fromfile(path).pretty_text
    forbidden_text = (
        "PCOTMRASPreBackboneFrameSelector",
        "PCOTMRASHybridFrameScout",
        "PCOTMRASRSeriesHybridFrameScout",
        "PCOTMRASTinyTransformerFrameScout",
    )
    for token in forbidden_text:
        _require(token not in pretty_text, f"resolved config still contains old C3 selector/reader token {token}")

    budgets = {
        "dense_window_size": dense_window_size,
        "min_budget": min_budget,
        "target_budget": target_budget,
        "max_budget": max_budget,
        "probe_stride": int(frame_selector.get("probe_stride")),
    }
    return cfg, scope, gate, budgets


def _validate_local_locked_gate(gate: dict[str, Any]) -> None:
    _require(gate.get("launch_gate_passed") is False, "local BH-SDC precheck must remain locked")
    _require(gate.get("allow_precheck_only") is True, "local BH-SDC gate should allow precheck-only validation")
    _require(_as_list(gate.get("allowed_entrypoints", ())) == [], "local allowed_entrypoints must be empty")
    for key in LOCKED_FALSE_OR_ABSENT_GATE_KEYS:
        _require(gate.get(key) is not True, f"local BH-SDC gate must keep {key}=false/absent")


def _validate_full_candidate_gate(gate: dict[str, Any]) -> None:
    _require(gate.get("launch_gate_passed") is False, "full candidate must remain locked pending follow-up Pro review")
    _require(gate.get("static_authorization") is False, "full candidate must not be statically authorized")
    _require(gate.get("external_payload_required") is False, "full candidate must not accept launch payloads yet")
    _require(gate.get("launch_gate_commit_review_required") is True, "launch gate commit must require Pro review")
    _require(gate.get("launch_decision") == FOLLOWUP_PRO_STATUS, f"launch_decision must be {FOLLOWUP_PRO_STATUS}")
    _require(gate.get("reviewed_impl_commit") == REVIEWED_IMPL_COMMIT, "reviewed_impl_commit mismatch")
    _require(gate.get("allow_tools_train") is False, "full candidate must not allow tools/train.py before follow-up Pro")
    _require(gate.get("allow_tools_test") is False, "full candidate must reject tools/test.py")
    _require(gate.get("allow_detector_map") is False, "full candidate must reject direct detector mAP claims")
    _require(gate.get("allow_train_validation_map") is False, "full candidate must not allow train-time validation mAP")
    _require(gate.get("allow_long_training") is False, "full candidate must not allow long training")
    _require(gate.get("allow_remote_sync") is False, "full candidate must not allow remote sync")
    _require(gate.get("allow_slurm") is False, "full candidate must not allow Slurm")
    _require(gate.get("allow_gpu") is False, "full candidate must not allow GPU use")
    _require(gate.get("allow_full_train") is False, "full candidate must not allow full train")
    _require(gate.get("allow_precheck_only") is True, "full candidate should allow local/static/smoke precheck only")
    _require(gate.get("allow_dataset_access") is False, "full candidate must not allow dataset access")
    _require(gate.get("allow_checkpoint_write") is False, "full candidate must not allow checkpoint writes")
    _require(gate.get("allow_checkpoint_load") is False, "checkpoint load must remain disabled")
    _require(gate.get("allow_pretrained_initialization") is False, "pretrained initialization must remain disabled")
    _require(gate.get("allow_resume") is False, "resume must remain disabled")
    _require(gate.get("allow_raw_prediction_cache") is False, "raw prediction cache must remain disabled")
    _require(_as_list(gate.get("allowed_entrypoints")) == [], "full candidate allowed_entrypoints must be empty")
    _require(_as_list(gate.get("command_whitelist")) == [], "full candidate command_whitelist must be empty")
    _require(gate.get("remote_workspace") == REMOTE_WORKSPACE, "remote workspace must be N16R4 ~/run/yuzibo path")
    _require(gate.get("slurm_script") == SLURM_SCRIPT, "slurm script mismatch")
    _require(gate.get("train_command") == TRAIN_COMMAND, "train command mismatch")

    context = gate.get("entrypoint_gate_context")
    _require(isinstance(context, dict), "entrypoint_gate_context must be a dict")
    _require(context.get("required") is True, "entrypoint gate context must be required")
    _require(_as_list(context.get("allowed_decisions")) == [], "full candidate must not allow launch decisions yet")
    _require(context.get("gate_json_env") == "OPENTAD_BH_SDC_GATE_JSON", "gate JSON env mismatch")
    _require(context.get("gate_sha256_env") == "OPENTAD_BH_SDC_GATE_SHA256", "gate SHA env mismatch")
    _require(context.get("active_manifest_sha256_env") == "OPENTAD_BH_SDC_ACTIVE_MANIFEST_SHA256", "manifest env mismatch")
    _require(context.get("resolved_config_sha256_env") == "OPENTAD_BH_SDC_RESOLVED_CONFIG_SHA256", "resolved env mismatch")
    _require(context.get("strict_payload_validation") is True, "entrypoint payload validation must be strict")
    _require(
        context.get("unknown_key_policy") == "reject_unknown_except_explicit_harmless_metadata",
        "entrypoint payload must reject unknown keys",
    )


def validate_static_config(path: Path) -> dict[str, Any]:
    cfg, scope, gate, budgets = _validate_common_route_config(path)
    stage = scope.get("stage") or gate.get("stage")
    if stage == FULL_STAGE:
        _validate_full_candidate_gate(gate)
    else:
        _validate_local_locked_gate(gate)

    allowed_entrypoints = _as_list(gate.get("allowed_entrypoints", ()))
    base_forbidden_hits = forbidden_base_chain_hits(path)
    return {
        "ok": True,
        "config": str(path),
        "config_sha256": sha256_file(path),
        "resolved_config_sha256": resolved_config_sha256(path),
        "route": ROUTE,
        "route_label": ROUTE_LABEL,
        "stage": stage,
        "selector": SELECTOR_TYPE,
        "completion_bridge": COMPLETION_TYPE,
        **budgets,
        "allowed_entrypoints": allowed_entrypoints,
        "base_chain": [str(item) for item in config_base_chain(path)],
        "base_chain_forbidden_tokens": base_forbidden_hits,
        "launch_gate_passed": gate.get("launch_gate_passed"),
        "allow_tools_train": gate.get("allow_tools_train", False),
        "allow_tools_test": gate.get("allow_tools_test", False),
        "allow_long_training": gate.get("allow_long_training", False),
        "launch_decision": gate.get("launch_decision"),
        "reviewed_impl_commit": gate.get("reviewed_impl_commit"),
        "static_authorization": gate.get("static_authorization", False),
        "command_whitelist": _as_list(gate.get("command_whitelist", ())),
        "remote_workspace": gate.get("remote_workspace"),
    }


validate_locked_config = validate_static_config


def _require_key(payload: dict[str, Any], key: str) -> Any:
    _require(key in payload, f"launch gate payload missing {key}")
    return payload[key]


def _require_hex(value: Any, length: int, key: str) -> str:
    text = str(value)
    pattern = HEX40_RE if length == 40 else HEX64_RE
    _require(pattern.match(text) is not None, f"{key} must be a {length}-hex hash")
    return text


def _require_exact(payload: dict[str, Any], key: str, expected: Any) -> None:
    _require(_require_key(payload, key) == expected, f"{key} must be {expected!r}")


def _validate_payload_schema(payload: dict[str, Any]) -> None:
    _require(isinstance(payload, dict), "launch gate payload must be a JSON object")
    if "decision" in payload and payload["decision"] != LAUNCH_DECISION:
        raise ValueError(f"decision must be {LAUNCH_DECISION!r}")
    missing = sorted(PAYLOAD_KEYS - set(payload))
    if missing:
        raise ValueError(f"launch gate payload missing {missing[0]}")
    unknown = sorted(set(payload) - PAYLOAD_KEYS)
    if unknown:
        raise ValueError(f"launch gate payload contains unknown key: {unknown[0]}")


def validate_launch_gate_payload(
    config_path: Path,
    payload: dict[str, Any],
    *,
    requested_action: str | None = None,
    requested_command: str | None = None,
    active_manifest_sha256: str | None = None,
    resolved_config_sha256: str | None = None,
) -> dict[str, Any]:
    static = validate_static_config(config_path)
    _require(static["stage"] == FULL_STAGE, "launch payload is only valid for the BH-SDC full-train candidate config")
    raise ValueError(
        "BH-SDC full train remains locked pending follow-up Pro review; "
        "no payload authorizes remote sync, Slurm, tools/train.py, checkpoint writes, or mAP claims."
    )
    _validate_payload_schema(payload)

    _require_exact(payload, "schema_version", 1)
    _require_exact(payload, "decision", LAUNCH_DECISION)
    _require_exact(payload, "explicit_user_pro_launch_decision", LAUNCH_DECISION)
    _require_exact(payload, "pro_launch_gate_verdict", LAUNCH_DECISION)
    _require(str(payload["pro_launch_gate_session"]).strip(), "pro_launch_gate_session must be non-empty")
    _require_exact(payload, "route", ROUTE)
    _require_exact(payload, "route_label", ROUTE_LABEL)
    _require_exact(payload, "stage", FULL_STAGE)
    _require_exact(payload, "reviewed_impl_commit", REVIEWED_IMPL_COMMIT)
    _require_hex(payload["reviewed_impl_commit"], 40, "reviewed_impl_commit")
    _require_hex(payload["launch_gate_commit"], 40, "launch_gate_commit")
    _require(str(payload["pro_implementation_verdict"]).startswith("PASS_"), "pro_implementation_verdict must be PASS_*")
    _require(str(payload["pro_implementation_session"]).strip(), "pro_implementation_session must be non-empty")
    _require_exact(payload, "final_read_only_review_verdict", "PASS_SUBAGENT_FINAL_REVIEW_ONLY")
    _require(str(payload["final_read_only_review_id"]).strip(), "final_read_only_review_id must be non-empty")

    payload_resolved = _require_hex(payload["resolved_config_sha256"], 64, "resolved_config_sha256")
    expected_resolved = resolved_config_sha256 or static["resolved_config_sha256"]
    _require(payload_resolved == expected_resolved, "resolved_config_sha256 mismatch")
    payload_manifest = _require_hex(payload["active_sha256_manifest_sha256"], 64, "active_sha256_manifest_sha256")
    if active_manifest_sha256 is not None:
        _require(payload_manifest == active_manifest_sha256, "active_sha256_manifest_sha256 mismatch")

    _require(payload["command_whitelist"] == COMMAND_WHITELIST, "command_whitelist must exactly match BH-SDC allowlist")
    _require_exact(payload, "remote_workspace", REMOTE_WORKSPACE)
    _require_exact(payload, "remote_workspace_policy", "N16R4_YUZIBO_ONLY")
    _require_exact(payload, "slurm_script", SLURM_SCRIPT)
    _require_exact(payload, "slurm_partition", "gpu")
    _require_exact(payload, "max_gpus", 1)
    _require_exact(payload, "max_nodes", 1)
    _require(int(payload["max_time_hours"]) <= 48, "max_time_hours must be <=48")
    _require(int(payload["max_epochs"]) <= 60, "max_epochs must be <=60")
    _require_exact(payload, "train_command", TRAIN_COMMAND)

    for key in PAYLOAD_TRUE_KEYS:
        _require(payload[key] is True, f"{key} must be true")
    for key in PAYLOAD_FALSE_KEYS:
        _require(payload[key] is False, f"{key} must be false")

    _require_exact(payload, "checkpoint_write_policy", "route_work_dir_only")
    _require_exact(payload, "checkpoint_load_policy", "none")
    _require_exact(payload, "pretrained_initialization_policy", "none")
    _require_exact(payload, "resume_policy", "none")
    _require_exact(payload, "dataset_scope", "THUMOS14_TAD_ONLY_TRAIN200_VALTEST211")

    if requested_action is not None:
        _require(requested_action in ACTION_COMMANDS, f"requested_action is not allowed: {requested_action}")
        expected_command = ACTION_COMMANDS[requested_action]
        if requested_command is not None:
            _require(requested_command == expected_command, "requested_command does not match requested_action")
    if requested_command is not None:
        _require(requested_command in COMMAND_WHITELIST, "requested_command is not in command_whitelist")
        _require("tools/test.py" not in requested_command, "requested_command must not call tools/test.py")

    return {
        "ok": True,
        "authorized": True,
        "decision": LAUNCH_DECISION,
        "route": ROUTE,
        "route_label": ROUTE_LABEL,
        "stage": FULL_STAGE,
        "requested_action": requested_action,
        "requested_command": requested_command,
        "command_whitelist": list(COMMAND_WHITELIST),
        "remote_workspace": REMOTE_WORKSPACE,
        "resolved_config_sha256": payload_resolved,
        "active_sha256_manifest_sha256": payload_manifest,
        "reviewed_impl_commit": REVIEWED_IMPL_COMMIT,
        "launch_gate_commit": payload["launch_gate_commit"],
        "claims_allowed": False,
    }


def _load_gate_payload(path: Path, expected_sha256: str | None) -> dict[str, Any]:
    _require(path.is_file(), f"gate payload JSON does not exist: {path}")
    if expected_sha256 is not None:
        _require_hex(expected_sha256, 64, "gate payload sha256")
        actual_sha256 = sha256_file(path)
        _require(actual_sha256 == expected_sha256, "gate payload sha256 mismatch")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"gate payload JSON is invalid: {exc}") from exc
    _require(isinstance(payload, dict), "gate payload JSON must contain an object")
    return payload


def _failure_result(config: Path, reason: str, static: dict[str, Any] | None = None) -> dict[str, Any]:
    result = {
        "ok": False,
        "authorized": False,
        "config": str(config),
        "reason": reason,
        "route": ROUTE,
        "route_label": ROUTE_LABEL,
        "launch_decision": static.get("launch_decision") if static else FOLLOWUP_PRO_STATUS,
    }
    if static:
        result.update(
            {
                "stage": static.get("stage"),
                "resolved_config_sha256": static.get("resolved_config_sha256"),
                "launch_gate_passed": static.get("launch_gate_passed"),
                "allowed_entrypoints": static.get("allowed_entrypoints"),
                "command_whitelist": static.get("command_whitelist"),
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate BH-SDC N16R4 launch-gate payload.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--gate-json", type=Path, default=None, help="Launch gate payload JSON.")
    parser.add_argument("--gate-sha256", default=None, help="Expected SHA256 for --gate-json.")
    parser.add_argument("--active-manifest-sha256", default=None, help="Active manifest SHA256.")
    parser.add_argument("--resolved-config-sha256", default=None, help="Resolved config SHA256.")
    parser.add_argument("--requested-action", default=None, help="Action being authorized.")
    parser.add_argument("--requested-command", default=None, help="Exact command being authorized.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    static: dict[str, Any] | None = None
    try:
        static = validate_static_config(args.config)
        if static.get("stage") == FULL_STAGE and static.get("launch_decision") == FOLLOWUP_PRO_STATUS:
            raise ValueError(
                "BH-SDC full train remains locked pending follow-up Pro review; "
                "no payload authorizes remote sync, Slurm, tools/train.py, checkpoint writes, or mAP claims."
            )
        gate_json = args.gate_json or os.environ.get("OPENTAD_BH_SDC_GATE_JSON")
        gate_sha256 = args.gate_sha256 or os.environ.get("OPENTAD_BH_SDC_GATE_SHA256")
        active_manifest = args.active_manifest_sha256 or os.environ.get("OPENTAD_BH_SDC_ACTIVE_MANIFEST_SHA256")
        resolved_hash = args.resolved_config_sha256 or os.environ.get("OPENTAD_BH_SDC_RESOLVED_CONFIG_SHA256")
        if not gate_json:
            raise ValueError("missing launch gate payload: pass --gate-json or set OPENTAD_BH_SDC_GATE_JSON")
        if not gate_sha256:
            raise ValueError("missing launch gate payload sha256: pass --gate-sha256 or set OPENTAD_BH_SDC_GATE_SHA256")
        payload = _load_gate_payload(Path(gate_json), gate_sha256)
        result = validate_launch_gate_payload(
            args.config,
            payload,
            requested_action=args.requested_action,
            requested_command=args.requested_command,
            active_manifest_sha256=active_manifest,
            resolved_config_sha256=resolved_hash,
        )
    except Exception as exc:
        result = _failure_result(args.config, str(exc), static)
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(f"BH-SDC launch gate FAIL: {result['reason']}")
        return 3

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            "BH-SDC launch gate PASS: "
            f"{result['requested_action'] or 'unspecified_action'} is authorized by {result['decision']}."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
