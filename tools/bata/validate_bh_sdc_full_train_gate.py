from __future__ import annotations

import hashlib
import argparse
import json
from pathlib import Path
from typing import Any

from mmengine.config import Config


ROUTE = "bh_sdc_boundary_hazard_sparse_dense"
ROUTE_LABEL = "DIVERGENT_INNOVATION_BH_SDC_DO_NOT_MERGE_WITH_C3"
SELECTOR_TYPE = "PCOTMRASBoundaryHazardSparseDenseFrameSelector"
COMPLETION_TYPE = "PCOTMRASBoundaryHazardSparseToDenseBridge"

FORBIDDEN_TRUE_KEYS = (
    "allow_remote_sync",
    "allow_slurm",
    "allow_gpu",
    "allow_tools_train",
    "allow_tools_test",
    "allow_detector_map",
    "allow_train_validation_map",
    "allow_long_training",
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config_namespace(path: Path) -> dict[str, Any]:
    cfg = Config.fromfile(path)
    return cfg._cfg_dict.to_dict()


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


def validate_locked_config(path: Path) -> dict[str, Any]:
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
    _require(gate.get("launch_gate_passed") is False, "BH-SDC gate must remain locked by default")
    _require(gate.get("allow_precheck_only") is True, "BH-SDC gate should allow precheck-only validation")

    for key in FORBIDDEN_TRUE_KEYS:
        _require(gate.get(key) is not True, f"BH-SDC gate must keep {key}=false/absent before approval")
    for key in FORBIDDEN_SCOPE_TRUE_KEYS:
        _require(scope.get(key) is not True, f"experiment_scope must keep {key}=false/absent before approval")

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
    _require(0 < min_budget <= target_budget <= max_budget < dense_window_size, "dynamic budget bounds are invalid")
    _require(int(token_compressor.get("dense_window_size")) == dense_window_size, "bridge dense_window_size mismatch")
    _require(int(token_compressor.get("target_len")) == dense_window_size, "bridge target_len mismatch")
    _require(int(_get_nested(model, "backbone", "backbone", "total_frames")) == max_budget, "backbone total_frames must equal max_budget")
    _require(int(_get_nested(model, "projection", "max_seq_len")) == dense_window_size, "projection max_seq_len must be dense_window_size")
    backbone_custom = _get_nested(model, "backbone", "custom")
    if isinstance(backbone_custom, dict):
        _require(
            backbone_custom.get("pretrain") in (None, "", False),
            "BH-SDC locked candidate must not inherit a pretrained initialization path",
        )
    for key in ("load_from", "resume", "resume_from", "checkpoint", "checkpoint_path"):
        _require(cfg.get(key) in (None, "", False), f"resolved config must not set {key}")
    inference = cfg.get("inference", {})
    _require(inference.get("load_from_raw_predictions") is False, "raw prediction loading must be disabled")
    _require(inference.get("save_raw_prediction") is False, "raw prediction saving must be disabled")
    pretty_text = Config.fromfile(path).pretty_text
    forbidden_text = (
        "PCOTMRASPreBackboneFrameSelector",
        "PCOTMRASHybridFrameScout",
        "PCOTMRASRSeriesHybridFrameScout",
        "PCOTMRASTinyTransformerFrameScout",
    )
    for token in forbidden_text:
        _require(token not in pretty_text, f"resolved config still contains old C3 selector/reader token {token}")
    return {
        "ok": True,
        "config": str(path),
        "config_sha256": sha256_file(path),
        "route": ROUTE,
        "route_label": ROUTE_LABEL,
        "stage": scope.get("stage"),
        "selector": SELECTOR_TYPE,
        "completion_bridge": COMPLETION_TYPE,
        "dense_window_size": dense_window_size,
        "min_budget": min_budget,
        "target_budget": target_budget,
        "max_budget": max_budget,
        "launch_gate_passed": gate.get("launch_gate_passed"),
        "allow_long_training": gate.get("allow_long_training"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate BH-SDC fail-closed full-train candidate config.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()
    result = validate_locked_config(args.config)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            "BH-SDC gate PASS: "
            f"{result['stage']} remains locked; budgets "
            f"{result['min_budget']}/{result['target_budget']}/{result['max_budget']} over {result['dense_window_size']}."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
