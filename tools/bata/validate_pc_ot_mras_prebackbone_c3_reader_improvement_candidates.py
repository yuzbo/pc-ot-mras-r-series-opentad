import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs" / "adatad" / "thumos"
SELECTOR_SOURCE = ROOT / "opentad" / "models" / "selectors" / "pc_ot_mras_prebackbone_frame_selector.py"

EXPECTED = {
    "cnn_lite": {
        "config": "pc_ot_mras_prebackbone_c3_cnn_lite_reader_candidate_n16r4.py",
        "variant_id": "C3-CNN-Lite-OriginalAdaTAD",
        "route": "pc_ot_mras_prebackbone_c3_cnn_lite_reader",
        "stage": "c3_cnn_lite_reader_candidate_n16r4",
        "reader_type": "PCOTMRASCNNFrameScout",
        "selector_support_status": "supported_by_prebackbone_frame_selector",
        "required_reader_keys": {
            "in_dim": 3 * 32 * 32,
            "hidden_dim": 128,
            "num_slots": 384,
            "num_layers": 2,
            "kernel_size": 5,
            "dropout": 0.0,
        },
    },
    "motion_tcn": {
        "config": "pc_ot_mras_prebackbone_c3_motion_tcn_reader_candidate_n16r4.py",
        "variant_id": "C3-Motion-TCN-OriginalAdaTAD",
        "route": "pc_ot_mras_prebackbone_c3_motion_tcn_reader",
        "stage": "c3_motion_tcn_reader_candidate_n16r4",
        "reader_type": "PCOTMRASMotionTCNFrameScout",
        "selector_support_status": "supported_by_prebackbone_frame_selector",
        "required_reader_keys": {
            "in_dim": 3 * 32 * 32,
            "hidden_dim": 128,
            "num_slots": 384,
            "num_layers": 4,
            "kernel_size": 5,
            "dilations": (1, 2, 4, 8),
            "dropout": 0.05,
            "motion_feature_mode": "frame_delta_abs",
            "motion_delta_stride": 1,
            "motion_rgb_fusion": "descriptor_plus_delta",
        },
    },
    "hybrid": {
        "config": "pc_ot_mras_prebackbone_c3_hybrid_reader_candidate_n16r4.py",
        "variant_id": "C3-Hybrid-OriginalAdaTAD",
        "route": "pc_ot_mras_prebackbone_c3_hybrid_reader",
        "stage": "c3_hybrid_reader_candidate_n16r4",
        "reader_type": "PCOTMRASHybridFrameScout",
        "selector_support_status": "supported_by_prebackbone_frame_selector",
        "required_reader_keys": {
            "in_dim": 3 * 32 * 32,
            "hidden_dim": 128,
            "num_slots": 384,
            "temporal_layers": 2,
            "temporal_kernel_size": 5,
            "slot_mlp_layers": 2,
            "slot_hidden_dim": 128,
            "slot_dropout": 0.0,
            "slot_temperature_init": 1.0,
            "local_global_fusion": "temporal_cnn_plus_slot_attention",
            "dropout": 0.05,
        },
    },
}

FORBIDDEN_TRUE_KEYS = (
    "allow_tools_train",
    "allow_tools_test",
    "allow_detector_map",
    "allow_train_validation_map",
    "allow_long_training",
    "allow_remote_sync",
    "allow_slurm",
    "allow_gpu",
    "allow_checkpoint_write",
    "allow_checkpoint_load",
    "allow_resume",
    "metric_claim_allowed",
    "paper_claim_allowed",
    "runtime_flops_claim_allowed",
    "deploy_claim_allowed",
)


def _as_plain(value):
    if isinstance(value, tuple):
        return tuple(value)
    if isinstance(value, list):
        return tuple(value)
    return value


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _load_config(path):
    from mmengine.config import Config

    return Config.fromfile(str(path))


def _reader_is_declared(reader_type):
    if not SELECTOR_SOURCE.is_file():
        return False
    text = SELECTOR_SOURCE.read_text(encoding="utf-8")
    return f"class {reader_type}" in text and f'"{reader_type}"' in text


def validate_one(name, spec):
    config_path = CONFIG_DIR / spec["config"]
    _require(config_path.is_file(), f"missing config: {config_path}")
    cfg = _load_config(config_path)

    _require(cfg.variant_id == spec["variant_id"], f"{name}: variant_id mismatch")
    _require(cfg.route_id == spec["route"], f"{name}: route_id mismatch")
    _require(cfg.stage_id == spec["stage"], f"{name}: stage_id mismatch")
    _require(cfg.window_size == 384, f"{name}: window_size must remain 384")
    _require(cfg.dense_window_size == 768, f"{name}: dense_window_size must remain 768")
    _require(cfg.selection_unit == 1, f"{name}: selection_unit must remain 1")
    _require(cfg.scout_spatial_size == 32, f"{name}: scout_spatial_size must remain 32")
    _require(cfg.scout_descriptor_dim == 3 * 32 * 32, f"{name}: descriptor dim mismatch")

    scope = cfg.experiment_scope
    _require(scope.variant_id == spec["variant_id"], f"{name}: scope variant mismatch")
    _require(scope.route == spec["route"], f"{name}: scope route mismatch")
    _require(scope.stage == spec["stage"], f"{name}: scope stage mismatch")
    _require(scope.detector_stack == "original_adatad_actionformer_adapter", f"{name}: detector stack drift")
    _require(scope.backend == "OriginalAdaTAD", f"{name}: backend drift")
    _require(scope.selection_surface == "pre_backbone_raw_frame", f"{name}: surface drift")
    _require(scope.selection_timing == "online_before_backbone", f"{name}: timing drift")
    _require(scope.acquisition_unit == "frame", f"{name}: acquisition unit drift")
    _require(scope.selector_reader == spec["reader_type"], f"{name}: reader scope mismatch")
    _require(scope.selector_support_status == spec["selector_support_status"], f"{name}: support status mismatch")
    _require(scope.free_frame_level_selector is True, f"{name}: free frame selector must stay true")
    _require(scope.protected_scaffold is False, f"{name}: protected scaffold must stay false")
    _require(scope.c2_320_uniform_plus_64_residual is False, f"{name}: C2 scaffold must stay false")
    _require(scope.s80r16_cell96x4_enabled is False, f"{name}: S80R16 must stay false")
    for key in ("uses_p2", "uses_offline_ledger", "uses_teacher", "uses_test_gt", "uses_raw_prediction_cache"):
        _require(scope[key] is False, f"{name}: {key} must stay false")
    for key in ("changes_detector_head", "changes_neck", "changes_loss_assignment", "changes_post_processing"):
        _require(scope[key] is False, f"{name}: {key} must stay false")
    _require(scope.changes_input_sampling is True, f"{name}: input sampling surface must be explicit")
    for key in ("metric_claim_allowed", "deploy_claim_allowed", "runtime_flops_claim_allowed", "paper_claim_allowed"):
        _require(scope[key] is False, f"{name}: {key} must stay false")

    gate = cfg.pc_ot_mras_prebackbone_e2e_acquisition_gate
    _require(gate.route == spec["route"], f"{name}: gate route mismatch")
    _require(gate.stage == spec["stage"], f"{name}: gate stage mismatch")
    _require(gate.default_off is True, f"{name}: gate must default off")
    _require(gate.formal_train_candidate is False, f"{name}: gate must not be formal train")
    _require(gate.launch_gate_passed is False, f"{name}: launch gate must not be passed")
    _require(gate.allowed_entrypoints == (), f"{name}: allowed_entrypoints must be empty")
    for key in FORBIDDEN_TRUE_KEYS:
        _require(gate[key] is False, f"{name}: gate {key} must be false")
    forbidden = set(gate.entrypoint_gate_context.forbidden_true_keys)
    for key in (
        "allow_tools_test",
        "direct_tools_test",
        "raw_prediction_cache",
        "load_from_raw_predictions",
        "save_raw_prediction",
        "uses_test_gt",
        "paper_claim",
        "deploy_claim",
        "runtime_flops_claim",
    ):
        _require(key in forbidden, f"{name}: gate must forbid {key}")

    selector = cfg.model.frame_selector
    _require(selector.target_len == 384, f"{name}: target_len drift")
    _require(selector.dense_window_size == 768, f"{name}: dense_window_size drift")
    _require(selector.selection_unit == 1, f"{name}: selection_unit drift")
    _require(selector.descriptor_dim == 3 * 32 * 32, f"{name}: descriptor drift")
    _require(selector.scout_feature_source == "compressed_pixels", f"{name}: scout source drift")
    _require(selector.scout_spatial_size == 32, f"{name}: scout size drift")
    _require(selector.protected_uniform_count == 0, f"{name}: protected_uniform_count must remain 0")
    _require(selector.coverage_guard_count == 0, f"{name}: coverage_guard_count must remain 0")
    _require(selector.transport_topk == 1, f"{name}: transport_topk drift")
    _require(selector.eval_transport_topk == 1, f"{name}: eval_transport_topk drift")
    _require(selector.straight_through_downstream is True, f"{name}: straight-through drift")
    _require(selector.remap_gt_to_selected_axis is True, f"{name}: GT remap drift")
    _require(selector.reader.type == spec["reader_type"], f"{name}: reader type mismatch")
    for key, expected in spec["required_reader_keys"].items():
        actual = _as_plain(selector.reader[key])
        _require(actual == expected, f"{name}: reader.{key} expected {expected!r}, got {actual!r}")

    _require(cfg.model.rpn_head.type == "ActionFormerHead", f"{name}: detector head drift")
    _require(cfg.model.neck.type != "PCOTMRASDetectorBridge", f"{name}: bridge neck must stay disabled")
    _require(cfg.model.backbone.backbone.total_frames == 384, f"{name}: backbone frames drift")
    _require(cfg.model.projection.max_seq_len == 384, f"{name}: projection max seq drift")
    _require(cfg.workflow.end_epoch == 60, f"{name}: workflow epoch drift")
    _require(cfg.workflow.checkpoint_interval == 60, f"{name}: checkpoint interval drift")
    _require(cfg.workflow.val_start_epoch == 40, f"{name}: val_start_epoch drift")
    _require(cfg.workflow.val_eval_interval == 2, f"{name}: val_eval_interval drift")
    _require(cfg.inference.load_from_raw_predictions is False, f"{name}: raw prediction load must stay false")
    _require(cfg.inference.save_raw_prediction is False, f"{name}: raw prediction save must stay false")

    train_loadframes = next((step for step in cfg.dataset.train.pipeline if step.get("type") == "LoadFrames"), None)
    _require(train_loadframes is not None, f"{name}: missing train LoadFrames")
    _require(train_loadframes.trunc_len == 768, f"{name}: train trunc_len must stay dense 768")
    for split in ("val", "test"):
        _require(cfg.dataset[split].window_size == 768, f"{name}: {split} window_size must stay dense 768")
    for split in ("train", "val", "test"):
        text = repr(cfg.dataset[split].pipeline).lower()
        for token in ("hard_positions", "teacher", "oracle", "raw_prediction", "value_transport_ledger"):
            _require(token not in text, f"{name}: {split} pipeline contains forbidden token {token}")

    registered = _reader_is_declared(spec["reader_type"])
    return {
        "name": name,
        "config": str(config_path.relative_to(ROOT)),
        "reader_type": spec["reader_type"],
        "reader_declared_in_selector_source": registered,
        "selector_support_status": spec["selector_support_status"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate local C3 reader-improvement candidate configs without launching training."
    )
    parser.add_argument(
        "--candidate",
        choices=tuple(EXPECTED.keys()) + ("all",),
        default="all",
        help="Candidate to validate.",
    )
    parser.add_argument(
        "--require-registered-readers",
        action="store_true",
        help="Fail if any candidate reader class is not declared in selector source.",
    )
    parser.add_argument("--json", action="store_true", help="Print a JSON summary.")
    args = parser.parse_args()

    names = tuple(EXPECTED) if args.candidate == "all" else (args.candidate,)
    results = [validate_one(name, EXPECTED[name]) for name in names]
    missing = [item for item in results if not item["reader_declared_in_selector_source"]]
    if args.require_registered_readers and missing:
        missing_names = ", ".join(f"{item['name']}:{item['reader_type']}" for item in missing)
        raise SystemExit(f"reader classes are not implemented yet: {missing_names}")

    payload = {
        "status": "PASS",
        "note": (
            "Config/gate fields are valid. Missing reader classes are acceptable for candidate-only "
            "preparation unless --require-registered-readers is set."
        ),
        "results": results,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for item in results:
            state = "DECLARED" if item["reader_declared_in_selector_source"] else "PENDING_READER_SOURCE"
            print(f"{item['name']}: {item['reader_type']} {state}")
        print("PC_OT_MRAS_PREBACKBONE_C3_READER_IMPROVEMENT_CANDIDATES_VALIDATION_PASS")


if __name__ == "__main__":
    main()
