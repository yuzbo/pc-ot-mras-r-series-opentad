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

ALLOWED_C3_PHYSICAL_GRID_IDENTITIES = {
    "pc_ot_mras_prebackbone_c3_physical_grid_actionformer": {
        "variant_id": "C3-PhysicalGridActionFormer-PreBackbone-OriginalAdaTAD",
        "stage_id": "c3_physical_grid_actionformer_full_train_n16r4",
        "selection_strategy": "frame_score_topk",
        "scope_selection_strategy": "frame_score_topk",
        "selector_reader": "PCOTMRASBoundaryDifficultyTemporalFrameScout",
        "uses_learned_boundary_head": False,
        "policy_kind": "frame_score_topk",
    },
    "pc_ot_mras_coarse_actionness_uncertainty_c3_physical_grid_actionformer": {
        "variant_id": "C3-CoarseActionnessUncertainty-PreBackbone-OriginalAdaTAD",
        "stage_id": "c3_coarse_actionness_uncertainty_fixed384_n16r4",
        "selection_strategy": "coarse_actionness_uncertainty",
        "scope_selection_strategy": "coarse_actionness_uncertainty",
        "selector_reader": "PCOTMRASCoarseActionnessFrameScout",
        "uses_learned_boundary_head": False,
        "policy_kind": "coarse_actionness",
    },
    "pc_ot_mras_exact_uniform_c3_physical_grid_actionformer": {
        "variant_id": "C3-ExactUniformPhysicalGrid-PreBackbone-OriginalAdaTAD",
        "stage_id": "c3_exact_uniform_physical_grid_fixed384_n16r4",
        "selection_strategy": "coarse_actionness_uncertainty",
        "scope_selection_strategy": "exact_uniform_physical_grid_control",
        "selector_reader": "PCOTMRASCoarseActionnessFrameScout",
        "uses_learned_boundary_head": False,
        "policy_kind": "exact_uniform_control",
    },
    "pc_ot_mras_uniform_biased_coarse_actionness_c3_physical_grid_actionformer": {
        "variant_id": "C3-UniformBiasedCoarseActionness-PreBackbone-OriginalAdaTAD",
        "stage_id": "c3_uniform_biased_coarse_actionness_fixed384_n16r4",
        "selection_strategy": "coarse_actionness_uncertainty",
        "scope_selection_strategy": "uniform_scaffold_small_actionness_bias_maxgap3",
        "selector_reader": "PCOTMRASCoarseActionnessFrameScout",
        "uses_learned_boundary_head": False,
        "policy_kind": "uniform_biased_coarse_actionness",
    },
    "pc_ot_mras_a_uniform_scaffold_small_actionness_strict_maxgap_c3_physical_grid_actionformer": {
        "variant_id": "C3-A-UniformScaffoldSmallActionnessStrictMaxGap-PreBackbone-OriginalAdaTAD",
        "stage_id": "c3_a_uniform_scaffold_small_actionness_strict_maxgap_fixed384_n16r4",
        "selection_strategy": "coarse_actionness_uncertainty",
        "scope_selection_strategy": "uniform_scaffold_small_actionness_strict_maxgap",
        "selector_reader": "PCOTMRASCoarseActionnessFrameScout",
        "uses_learned_boundary_head": False,
        "policy_kind": "uniform_scaffold_small_actionness_strict_maxgap",
    },
}

PADDING_DATASET_CONSTRUCTOR_KWARGS = frozenset(
    {
        "ann_file",
        "subset_name",
        "data_path",
        "pipeline",
        "class_map",
        "filter_gt",
        "class_agnostic",
        "block_list",
        "test_mode",
        "feature_stride",
        "sample_stride",
        "offset_frames",
        "fps",
        "logger",
    }
)
SLIDING_DATASET_CONSTRUCTOR_KWARGS = PADDING_DATASET_CONSTRUCTOR_KWARGS | frozenset(
    {
        "window_size",
        "window_overlap_ratio",
        "ioa_thresh",
    }
)
DATASET_CONSTRUCTOR_KWARGS_BY_TYPE = {
    "ThumosPaddingDataset": PADDING_DATASET_CONSTRUCTOR_KWARGS,
    "ThumosSlidingDataset": SLIDING_DATASET_CONSTRUCTOR_KWARGS,
}


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


def resolve_dataset_constructor_kwargs(cfg: Any, split: str) -> dict[str, Any]:
    dataset_cfg = cfg.dataset[split]
    return {str(key): value for key, value in dict(dataset_cfg).items() if key != "type"}


def _validate_dataset_constructor_kwargs(cfg: Any, split: str) -> None:
    dataset_cfg = cfg.dataset[split]
    dataset_type = _get(dataset_cfg, "type")
    allowed_kwargs = DATASET_CONSTRUCTOR_KWARGS_BY_TYPE.get(dataset_type)
    _require(allowed_kwargs is not None, f"{split}.type has unsupported dataset type {dataset_type}")

    dataset_kwargs = resolve_dataset_constructor_kwargs(cfg, split)
    unexpected = sorted(set(dataset_kwargs) - allowed_kwargs)
    _require(
        not unexpected,
        f"{split} {dataset_type} top-level kwargs include unsupported constructor fields: {unexpected}",
    )


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
    identity = ALLOWED_C3_PHYSICAL_GRID_IDENTITIES.get(cfg.route_id)

    _require(identity is not None, f"route_id must be one of {sorted(ALLOWED_C3_PHYSICAL_GRID_IDENTITIES)}")
    _require(cfg.variant_id == identity["variant_id"], f"variant_id must be {identity['variant_id']}")
    _require(cfg.stage_id == identity["stage_id"], f"stage_id must be {identity['stage_id']}")
    _require(cfg.route_label == "C3_ORIGINAL_OPTIMIZATION_ROUTE", "route_label must be C3 original")
    _require(cfg.route_family == "C3_MAINLINE_OPTIMIZATION", "route_family must be C3 mainline")
    _require(scope.variant_id == cfg.variant_id, "experiment_scope.variant_id mismatch")
    _require(scope.route == cfg.route_id, "experiment_scope.route mismatch")
    _require(scope.stage == cfg.stage_id, "experiment_scope.stage mismatch")
    _require(scope.selection_surface == "pre_backbone_raw_frame", "selection surface must be pre-backbone")
    _require(scope.backend == "OriginalAdaTAD_ActionFormerPhysicalGrid", "backend must be OriginalAdaTAD physical-grid")
    _require(scope.detector_stack == "physical_grid_actionformer_adapter", "detector stack mismatch")
    _require(scope.temporal_grid_mode == "physical", "temporal grid mode must be physical")

    for key in (
        "uses_p2",
        "uses_offline_ledger",
        "uses_teacher",
        "uses_test_gt",
        "uses_raw_prediction_cache",
        "uses_learned_boundary_head",
    ):
        _require(_get(scope, key, False) is False, f"experiment_scope.{key} must be false")
    _require(scope.changes_input_sampling is True, "input sampling must be the changed surface")
    _require(scope.changes_detector_head is True, "detector head geometry must be the changed surface")
    _require(scope.changes_loss_assignment is True, "loss/assignment must be marked changed")
    _require(scope.changes_post_processing is False, "post-processing must not be a method change")

    _require(selector.type == "PCOTMRASPreBackboneFrameSelector", "frame_selector must be pre-backbone C3 selector")
    _require(int(selector.target_len) == 384, "frame_selector.target_len must be 384")
    _require(int(selector.dense_window_size) == 768, "frame_selector.dense_window_size must be 768")
    _require(int(selector.descriptor_dim) == 3 * 32 * 32, "frame_selector.descriptor_dim must be 3072")
    _require(selector.remap_gt_to_selected_axis is False, "frame_selector.remap_gt_to_selected_axis must be false")
    _require(scope.selection_strategy == identity["scope_selection_strategy"], "scope selection strategy mismatch")
    _require(selector.selection_strategy == identity["selection_strategy"], "selector strategy mismatch")
    _require(selector.reader.type == identity["selector_reader"], "selector reader mismatch")
    if selector.selection_strategy == "coarse_actionness_uncertainty":
        _require(scope.selector_reader == "PCOTMRASCoarseActionnessFrameScout", "scope selector_reader mismatch")
        _require(selector.aux_frame_score_boundary_loss_weight == 0.0, "coarse selector must not use boundary aux loss")
        _require(selector.aux_uncertainty_loss_weight == 0.0, "coarse selector must not use learned uncertainty aux loss")
        policy_kind = identity["policy_kind"]
        if policy_kind == "coarse_actionness":
            _require(
                scope.budget_protocol == "fixed384_over_dense768_binary_actionness_uncertainty_change",
                "coarse budget protocol mismatch",
            )
            _require(int(selector.coarse_uniform_count) > 0, "coarse selector requires uniform scaffold")
            _require(int(selector.coarse_action_count) > 0, "coarse selector requires action quota")
            _require(int(selector.coarse_uncertainty_count) > 0, "coarse selector requires uncertainty quota")
            _require(int(selector.coarse_change_count) > 0, "coarse selector requires change quota")
            _require(int(selector.coarse_background_count) > 0, "coarse selector requires background quota")
        elif policy_kind == "exact_uniform_control":
            _require(
                scope.budget_protocol == "fixed384_over_dense768_exact_uniform_physical_grid_control",
                "exact-uniform budget protocol mismatch",
            )
            _require(int(selector.coarse_uniform_count) == 384, "exact-uniform control requires 384 uniform anchors")
            _require(int(selector.coarse_action_count) == 0, "exact-uniform control must disable action quota")
            _require(int(selector.coarse_uncertainty_count) == 0, "exact-uniform control must disable uncertainty quota")
            _require(int(selector.coarse_change_count) == 0, "exact-uniform control must disable change quota")
            _require(int(selector.coarse_background_count) == 0, "exact-uniform control must disable background quota")
            _require(float(selector.aux_gt_acquisition_loss_weight) == 0.0, "exact-uniform control must disable selector GT loss")
            _require(float(selector.aux_duplicate_cap_loss_weight) == 0.0, "exact-uniform control must disable duplicate cap loss")
            _require(selector.straight_through_detector_loss is False, "exact-uniform control must disable selector ST gradient")
            _require(int(selector.max_dense_gap) == 0, "exact-uniform control must not add a max-gap guard")
        elif policy_kind == "uniform_biased_coarse_actionness":
            _require(
                scope.budget_protocol == "fixed384_over_dense768_uniform288_action72_uncertainty24_guard12_maxgap3",
                "uniform-biased budget protocol mismatch",
            )
            _require(int(selector.coarse_uniform_count) == 288, "uniform-biased policy requires 288 uniform anchors")
            _require(int(selector.coarse_action_count) == 72, "uniform-biased policy requires 72 action slots")
            _require(int(selector.coarse_uncertainty_count) == 24, "uniform-biased policy requires 24 uncertainty slots")
            _require(int(selector.coarse_change_count) == 0, "uniform-biased policy must disable change quota")
            _require(int(selector.coarse_background_count) == 0, "uniform-biased policy must disable background quota")
            _require(int(selector.max_dense_gap) <= 3, "uniform-biased policy requires max_dense_gap <= 3")
            _require(int(selector.max_gap_guard_count) > 0, "uniform-biased policy requires max-gap guard enabled")
            _require(
                int(selector.max_gap_guard_count) <= 24,
                "uniform-biased policy guard must leave uncertainty quota reachable",
            )
        elif policy_kind == "uniform_scaffold_small_actionness_strict_maxgap":
            _require(
                scope.budget_protocol == "fixed384_over_dense768_uniform_scaffold_small_actionness_strict_maxgap_guard12",
                "A-line budget protocol mismatch",
            )
            _require(int(selector.coarse_uniform_count) == 288, "A-line requires 288 uniform anchors")
            _require(int(selector.coarse_action_count) == 72, "A-line requires 72 action slots")
            _require(int(selector.coarse_uncertainty_count) == 24, "A-line requires 24 uncertainty slots")
            _require(int(selector.coarse_change_count) == 0, "A-line must disable change quota")
            _require(int(selector.coarse_background_count) == 0, "A-line must disable background quota")
            _require(int(selector.max_dense_gap) == 3, "A-line requires max_dense_gap=3")
            _require(int(selector.max_gap_guard_count) == 12, "A-line requires max_gap_guard_count=12")
        else:
            raise ValueError(f"unsupported C3 policy kind {policy_kind}")
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

    _require(gate.route == cfg.route_id, "gate.route mismatch")
    _require(gate.stage == cfg.stage_id, "gate.stage mismatch")
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
    exact = gate.entrypoint_gate_context.required_exact_values
    _require(exact.route == cfg.route_id, "gate exact route mismatch")
    _require(exact.variant_id == cfg.variant_id, "gate exact variant mismatch")
    _require(exact.stage == cfg.stage_id, "gate exact stage mismatch")

    _require(cfg.post_processing.save_dict is True, "post_processing.save_dict must be true")
    _require(cfg.inference.load_from_raw_predictions is False, "load_from_raw_predictions must be false")
    _require(cfg.inference.save_raw_prediction is False, "save_raw_prediction must be false")
    _require(_get(cfg, "load_from") in (None, "", "none"), "load_from must stay disabled")
    _require(_get(cfg, "resume") in (None, False, "", "none"), "resume must stay disabled")

    for split in ("train", "val", "test"):
        _validate_dataset_constructor_kwargs(cfg, split)
        load_steps = [step for step in cfg.dataset[split].pipeline if _get(step, "type") == "LoadFrames"]
        _require(len(load_steps) == 1, f"{split} pipeline must contain one LoadFrames")
        if split == "train":
            _require(_get(cfg.dataset[split], "window_size") is None, "train.window_size must stay absent")
            _require(_get(load_steps[0], "method") == "random_trunc", "train.LoadFrames.method must be random_trunc")
            _require(int(_get(load_steps[0], "trunc_len", 0)) == 768, "train.LoadFrames.trunc_len must be dense 768")
        else:
            _require(int(_get(cfg.dataset[split], "window_size", 0)) == 768, f"{split}.window_size must be dense 768")
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
    identity = ALLOWED_C3_PHYSICAL_GRID_IDENTITIES.get(str(payload.get("route")))
    _require(identity is not None, f"payload route must be one of {sorted(ALLOWED_C3_PHYSICAL_GRID_IDENTITIES)}")
    _require(payload.get("decision") == ALLOW_DECISION, f"decision must be {ALLOW_DECISION}")
    _require(payload.get("variant_id") == identity["variant_id"], f"variant_id must be {identity['variant_id']}")
    _require(payload.get("stage") == identity["stage_id"], f"stage must be {identity['stage_id']}")
    _require(payload.get("execution_mode") == "train", "execution_mode must be train")
    _require(payload.get("budget") == 384, "budget must be 384")
    _require(payload.get("dense_window_size") == 768, "dense_window_size must be 768")
    _require(payload.get("active_sha256_manifest_sha256") == active_manifest_sha256, "active manifest sha mismatch")
    _require(payload.get("resolved_config_sha256") == resolved_config_sha256, "resolved config sha mismatch")
    _require(payload.get("pretrained_sha256") == pretrained_sha256, "pretrained sha mismatch")
    pretrained_resolved_path = str(payload.get("pretrained_resolved_path") or "")
    _require(pretrained_resolved_path.startswith("/"), "pretrained_resolved_path must be a resolved absolute remote path")

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
