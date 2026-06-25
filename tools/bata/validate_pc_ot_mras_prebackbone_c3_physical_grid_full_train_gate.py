from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


VARIANT_ID = "C3-PhysicalGridActionFormer-PreBackbone-OriginalAdaTAD"
ROUTE_ID = "pc_ot_mras_prebackbone_c3_physical_grid_actionformer"
STAGE_ID = "c3_physical_grid_actionformer_full_train_n16r4"
ALLOW_DECISION = "ALLOW_C3_PHYSICAL_GRID_FULL_TRAIN"
PASS_MESSAGE = "C3_PHYSICAL_GRID_FULL_TRAIN_GATE_VALIDATION_PASS"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _get(node: Any, key: str, default: Any = None) -> Any:
    if isinstance(node, Mapping):
        return node.get(key, default)
    getter = getattr(node, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            try:
                return getter(key)
            except Exception:
                pass
    try:
        return node[key]
    except Exception:
        return getattr(node, key, default)


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _repr_lower(value: Any) -> str:
    return repr(value).lower()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _forbid_tokens(text: str, *, context: str) -> None:
    for token in (
        "divergent_innovation",
        "bh_sdc",
        "event-surprise",
        "boundary microscope",
        "frame/token hybrid",
        "raw_prediction_cache=True",
        "load_from_raw_predictions=True",
        "save_raw_prediction=True",
    ):
        _require(token not in text, f"{context} contains forbidden token {token}")


def validate_config(cfg_path: str | Path) -> bool:
    from mmengine.config import Config

    cfg = Config.fromfile(str(cfg_path))
    scope = cfg.experiment_scope
    gate = cfg.pc_ot_mras_prebackbone_e2e_acquisition_gate
    selector = cfg.model.frame_selector
    head = cfg.model.rpn_head

    _require(cfg.variant_id == VARIANT_ID, f"variant_id must be {VARIANT_ID}")
    _require(cfg.route_id == ROUTE_ID, f"route_id must be {ROUTE_ID}")
    _require(cfg.stage_id == STAGE_ID, f"stage_id must be {STAGE_ID}")
    _require(cfg.route_label == "C3_ORIGINAL_OPTIMIZATION_ROUTE", "route_label must be C3 original")
    _require(cfg.route_family == "C3_MAINLINE_OPTIMIZATION", "route_family must be C3 mainline")
    _require(scope.variant_id == VARIANT_ID, "experiment_scope.variant_id mismatch")
    _require(scope.route == ROUTE_ID, "experiment_scope.route mismatch")
    _require(scope.stage == STAGE_ID, "experiment_scope.stage mismatch")
    _require(scope.selection_surface == "pre_backbone_raw_frame", "selection surface must be pre-backbone")
    _require(scope.backend == "OriginalAdaTAD_ActionFormerPhysicalGrid", "backend must be OriginalAdaTAD physical-grid")
    _require(scope.detector_stack == "physical_grid_actionformer_adapter", "detector stack mismatch")
    _require(scope.temporal_grid_mode == "physical", "temporal grid mode must be physical")

    for key in ("uses_p2", "uses_offline_ledger", "uses_teacher", "uses_test_gt", "uses_raw_prediction_cache"):
        _require(getattr(scope, key) is False, f"experiment_scope.{key} must be false")
    _require(scope.changes_input_sampling is True, "input sampling must be the changed surface")
    _require(scope.changes_detector_head is True, "detector head geometry must be the changed surface")
    _require(scope.changes_loss_assignment is True, "loss/assignment must be marked changed")
    _require(scope.changes_post_processing is False, "post-processing must not be a method change")

    _require(selector.type == "PCOTMRASPreBackboneFrameSelector", "frame_selector must be pre-backbone C3 selector")
    _require(int(selector.target_len) == 384, "frame_selector.target_len must be 384")
    _require(int(selector.dense_window_size) == 768, "frame_selector.dense_window_size must be 768")
    _require(int(selector.descriptor_dim) == 3 * 32 * 32, "frame_selector.descriptor_dim must be 3072")
    _require(selector.remap_gt_to_selected_axis is False, "frame_selector.remap_gt_to_selected_axis must be false")
    _require(selector.reader.type == "PCOTMRASBoundaryDifficultyTemporalFrameScout", "selector reader mismatch")
    _require(int(selector.reader.in_dim) == int(selector.descriptor_dim), "selector reader.in_dim must match descriptor_dim")
    _require(
        len(tuple(selector.reader.dilations)) == int(selector.reader.temporal_layers),
        "selector reader dilations length must match temporal_layers",
    )
    _require(int(cfg.model.projection.max_seq_len) == int(selector.target_len), "projection.max_seq_len must be selected 384")
    _require(
        int(cfg.model.backbone.backbone.total_frames) == int(selector.target_len),
        "backbone.total_frames must be selected 384",
    )
    _require(
        int(cfg.model.backbone.custom.pre_processing_pipeline[0].t1) == 24,
        "backbone pre-processing t1 must be 24 for selected 384",
    )
    _require(
        int(cfg.model.backbone.custom.post_processing_pipeline[1].t1) == 24,
        "backbone post-processing rearrange t1 must be 24 for selected 384",
    )
    _require(
        int(cfg.model.backbone.custom.post_processing_pipeline[2].size) == int(selector.target_len),
        "backbone post-processing interpolate size must be selected 384",
    )

    _require(head.type == "ActionFormerHead", "rpn_head must stay ActionFormerHead")
    physical = head.physical_grid_actionformer
    _require(physical.enabled is True, "physical_grid_actionformer.enabled must be true")
    _require(physical.required is True, "physical_grid_actionformer.required must be true")
    _require(physical.strict is True, "physical_grid_actionformer.strict must be true")

    _require(gate.route == ROUTE_ID, "gate.route mismatch")
    _require(gate.stage == STAGE_ID, "gate.stage mismatch")
    _require(gate.formal_train_candidate is True, "formal_train_candidate must be true")
    _require(gate.allow_detector_training is True, "allow_detector_training must be true")
    _require(gate.requires_launch_gate is True, "requires_launch_gate must be true")
    _require(gate.launch_gate_passed is True, "launch_gate_passed must be true")
    _require(gate.allow_precheck_only is True, "allow_precheck_only must be true")
    _require(gate.allow_slurm is True and gate.allow_gpu is True, "single-GPU Slurm must be explicitly allowed")
    _require(gate.allow_tools_train is True, "allow_tools_train must be true")
    _require(gate.allow_tools_test is False, "allow_tools_test must be false")
    _require(gate.allow_detector_map is False, "allow_detector_map must be false")
    _require(gate.allow_checkpoint_load is False, "checkpoint load must be false")
    _require(gate.allow_resume is False, "resume must be false")
    _require(gate.allow_raw_prediction_cache is False, "raw prediction cache must be false")
    _require(gate.offline_ledger is False, "offline ledger must be false")
    _require(_as_tuple(gate.allowed_entrypoints) == ("tools/train.py",), "allowed_entrypoints must only be tools/train.py")
    _require(gate.entrypoint_gate_context.gate_json_env == "OPENTAD_C3_PHYSICAL_GRID_GATE_JSON", "gate env mismatch")
    _require(gate.entrypoint_gate_context.allowed_decisions == (ALLOW_DECISION,), "allowed decision mismatch")

    _require(cfg.post_processing.save_dict is True, "post_processing.save_dict must be true")
    _require(cfg.inference.load_from_raw_predictions is False, "load_from_raw_predictions must be false")
    _require(cfg.inference.save_raw_prediction is False, "save_raw_prediction must be false")
    _require(_get(cfg, "load_from") in (None, "", "none"), "load_from must stay disabled")
    _require(_get(cfg, "resume") in (None, False, "", "none"), "resume must stay disabled")

    for split in ("train", "val", "test"):
        load_steps = [step for step in cfg.dataset[split].pipeline if _get(step, "type") == "LoadFrames"]
        _require(len(load_steps) == 1, f"{split} pipeline must contain one LoadFrames")
        _require(int(_get(cfg.dataset[split], "window_size", 0)) == 768, f"{split}.window_size must be dense 768")
        if split == "train":
            _require(_get(load_steps[0], "method") == "random_trunc", "train.LoadFrames.method must be random_trunc")
            _require(int(_get(load_steps[0], "trunc_len", 0)) == 768, "train.LoadFrames.trunc_len must be dense 768")
        else:
            _require(_get(load_steps[0], "method") == "sliding_window", f"{split}.LoadFrames.method must be sliding_window")
        _require(
            _get(load_steps[0], "remap_gt_to_selected_axis") is False,
            f"{split}.LoadFrames.remap_gt_to_selected_axis must be false",
        )
        pipeline_text = _repr_lower(cfg.dataset[split].pipeline)
        for token in ("bata_value_transport_ledger_subsample", "hard_positions", "teacher", "oracle", "raw_prediction"):
            _require(token not in pipeline_text, f"{split} pipeline contains forbidden token {token}")

    _forbid_tokens(_repr_lower(cfg), context="config")
    return True


def validate_gate_payload(
    payload: Mapping[str, Any],
    *,
    active_manifest_sha256: str,
    resolved_config_sha256: str,
    pretrained_sha256: str,
) -> Mapping[str, Any]:
    _require(payload.get("decision") == ALLOW_DECISION, f"decision must be {ALLOW_DECISION}")
    _require(payload.get("route") == ROUTE_ID, f"route must be {ROUTE_ID}")
    _require(payload.get("variant_id") == VARIANT_ID, f"variant_id must be {VARIANT_ID}")
    _require(payload.get("stage") == STAGE_ID, f"stage must be {STAGE_ID}")
    _require(payload.get("execution_mode") == "train", "execution_mode must be train")
    _require(payload.get("budget") == 384, "budget must be 384")
    _require(payload.get("dense_window_size") == 768, "dense_window_size must be 768")
    _require(payload.get("active_sha256_manifest_sha256") == active_manifest_sha256, "active manifest sha mismatch")
    _require(payload.get("resolved_config_sha256") == resolved_config_sha256, "resolved config sha mismatch")
    _require(payload.get("pretrained_sha256") == pretrained_sha256, "pretrained sha mismatch")

    for key in (
        "allow_tools_train",
        "allow_slurm",
        "allow_gpu",
        "single_gpu",
        "allow_prebackbone_frame_selector",
        "allow_detector_training",
        "allow_train_validation_map",
        "allow_long_training",
        "allow_pretrained_initialization",
        "allow_checkpoint_write",
    ):
        _require(payload.get(key) is True, f"{key}=true required")
    for key in (
        "uses_p2",
        "uses_offline_ledger",
        "uses_teacher",
        "uses_test_gt",
        "uses_raw_prediction_cache",
        "allow_tools_test",
        "direct_tools_test",
        "allow_checkpoint_load",
        "allow_resume",
        "offline_ledger",
        "raw_prediction_cache",
        "load_from_raw_predictions",
        "save_raw_prediction",
        "metric_claim",
        "metric_claim_allowed",
        "paper_claim",
        "paper_claim_allowed",
        "runtime_flops_claim",
        "runtime_flops_claim_allowed",
        "deploy_claim",
        "deploy_claim_allowed",
    ):
        _require(payload.get(key) is False, f"{key}=false required")
    return payload


def validate_gate_file(
    gate_json: str | Path,
    *,
    gate_sha256: str,
    active_manifest_sha256: str,
    resolved_config_sha256: str,
    pretrained_sha256: str,
) -> Mapping[str, Any]:
    gate_path = Path(gate_json)
    _require(gate_path.is_file(), f"gate JSON does not exist: {gate_path}")
    actual_sha = _sha256_file(gate_path)
    _require(actual_sha == gate_sha256, f"gate sha mismatch: expected={gate_sha256} actual={actual_sha}")
    payload = json.loads(gate_path.read_text(encoding="utf-8"))
    return validate_gate_payload(
        payload,
        active_manifest_sha256=active_manifest_sha256,
        resolved_config_sha256=resolved_config_sha256,
        pretrained_sha256=pretrained_sha256,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate C3 PhysicalGrid full-train fail-closed gates.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--gate-json")
    parser.add_argument("--gate-sha256")
    parser.add_argument("--active-manifest-sha256")
    parser.add_argument("--resolved-config-sha256")
    parser.add_argument("--pretrained-sha256")
    args = parser.parse_args(argv)
    try:
        validate_config(args.config)
        if args.gate_json:
            validate_gate_file(
                args.gate_json,
                gate_sha256=args.gate_sha256 or "",
                active_manifest_sha256=args.active_manifest_sha256 or "",
                resolved_config_sha256=args.resolved_config_sha256 or "",
                pretrained_sha256=args.pretrained_sha256 or "",
            )
    except Exception as exc:
        print(f"C3_PHYSICAL_GRID_FULL_TRAIN_GATE_VALIDATION_FAIL: {exc}", file=sys.stderr)
        return 1
    print(PASS_MESSAGE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
