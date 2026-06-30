import argparse
import ast
from pathlib import Path

from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_ROUTE_LABEL = "C3_MAINLINE_OPTIMIZATION"
ALLOWED_ROUTE_FAMILY = "C3_ORIGINAL_OPTIMIZATION_ROUTE"
ALLOWED_ROUTE_VARIANTS = {
    "C3_PQR_RankCalV1_MaxIoU",
    "C3_PQR_RankCalV1_MaxIoU_Stride2UniformBackendControl",
}
FORBIDDEN_ROUTE_TOKENS = ("BH", "BH-SDC", "BH_SDC", "DIVERGENT", "CADF")
EXPECTED_PRECHECK_SCOPE = "config_validator_plus_quality_head_unit"
EXPECTED_BUILD_ONLY_STATUS = "pseudo_boundary_dependency_restored_pending_remote_runtime_smoke"
EXPECTED_FULLTRAIN_SCOPE = "user_unlocked_pqr_formal_training_after_diagnostic_controls"
SUPPORTED_QUALITY_HEAD_KEYS = {
    "enabled",
    "kernel_size",
    "target_mode",
    "weight_init",
    "bias_init",
    "loss_weight",
    "score_alpha",
    "positive_weight",
    "negative_weight",
    "loss_normalizer",
    "keep_loss_graph_when_weight_zero",
}


def _assert_false(cfg, dotted_name):
    value = cfg
    for part in dotted_name.split("."):
        value = value[part]
    assert value is False, f"{dotted_name} must be False"


def _load_actionformer_init_args():
    source_path = ROOT / "opentad/models/detectors/actionformer.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "ActionFormer":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    return {arg.arg for arg in item.args.args if arg.arg != "self"}
    raise AssertionError("Could not find ActionFormer.__init__ signature")


def _load_top_level_function_args(source_path, function_name):
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return {arg.arg for arg in node.args.args}
    raise AssertionError(f"Could not find {function_name} in {source_path.as_posix()}")


def validate_detector_consumes_model_keys(cfg, config_path=None):
    assert cfg.model.type == "ActionFormer"
    assert "frame_selector" not in cfg.model, "model.frame_selector is not consumed by this snapshot's ActionFormer"

    accepted = _load_actionformer_init_args()
    supplied = {key for key in cfg.model.keys() if key != "type"}
    unknown = sorted(supplied - accepted)
    label = f" in {Path(config_path).as_posix()}" if config_path is not None else ""
    assert not unknown, f"unconsumed ActionFormer model key(s){label}: {unknown}"


def validate_runtime_max_train_iters_gate_consumed():
    train_path = ROOT / "tools/train.py"
    train_source = train_path.read_text(encoding="utf-8")
    assert 'max_train_iters = cfg.workflow.get("max_train_iters", None)' in train_source
    assert "remaining_train_iters = max_train_iters - completed_train_iters" in train_source
    assert "max_train_iters=remaining_train_iters" in train_source
    assert "skipping checkpoint/val/eval" in train_source

    train_engine_path = ROOT / "opentad/cores/train_engine.py"
    train_engine_source = train_engine_path.read_text(encoding="utf-8")
    train_one_epoch_args = _load_top_level_function_args(train_engine_path, "train_one_epoch")
    assert "max_train_iters" in train_one_epoch_args
    assert "max_train_iters = _normalize_max_train_iters(max_train_iters)" in train_engine_source
    assert "completed_iters >= max_train_iters" in train_engine_source
    assert "return completed_iters" in train_engine_source


def validate_clean_clone_transform_dependencies_present():
    pseudo_boundary_path = ROOT / "opentad/datasets/transforms/pseudo_boundary.py"
    assert pseudo_boundary_path.is_file(), (
        "missing clean-clone transform dependency: "
        "opentad/datasets/transforms/pseudo_boundary.py"
    )

    pseudo_boundary_source = pseudo_boundary_path.read_text(encoding="utf-8")
    for required_symbol in (
        "def load_boundary_scores",
        "def slice_global_scores_for_window",
        "def select_pseudo_boundary_hybrid_positions",
        "def select_pseudo_boundary_snap_positions",
    ):
        assert required_symbol in pseudo_boundary_source, (
            "pseudo_boundary.py is present but missing required API: "
            f"{required_symbol}"
        )

    end_to_end_source = (ROOT / "opentad/datasets/transforms/end_to_end.py").read_text(encoding="utf-8")
    assert "from .pseudo_boundary import" in end_to_end_source


def _load_frame_step(cfg, split):
    return next(step for step in cfg.dataset[split].pipeline if step["type"] == "LoadFrames")


def _validate_route(cfg):
    assert cfg.route_label == ALLOWED_ROUTE_LABEL
    assert cfg.route_family == ALLOWED_ROUTE_FAMILY
    assert cfg.route_variant in ALLOWED_ROUTE_VARIANTS
    joined = " ".join([cfg.route_label, cfg.route_family, cfg.route_variant]).upper()
    found = [token for token in FORBIDDEN_ROUTE_TOKENS if token in joined]
    assert not found, f"forbidden route token(s): {found}"


def _validate_quality_head(cfg):
    quality = cfg.model.rpn_head.quality_head_cfg
    unknown_quality_keys = sorted(set(quality.keys()) - SUPPORTED_QUALITY_HEAD_KEYS)
    assert not unknown_quality_keys, f"unsupported quality_head_cfg key(s): {unknown_quality_keys}"
    assert quality.enabled is True
    assert quality.target_mode == "max_iou"
    assert quality.weight_init == 0.0
    assert float(quality.bias_init) >= 4.0
    assert 0.02 <= float(quality.loss_weight) <= 0.05
    assert 0.05 <= float(quality.score_alpha) <= 0.15
    assert quality.loss_normalizer in {"valid", "weighted"}


def _validate_route_contract(cfg):
    _assert_false(cfg, "inference.load_from_raw_predictions")
    _assert_false(cfg, "inference.save_raw_prediction")
    _assert_false(cfg, "pqr_rankcal_v1.use_teacher")
    _assert_false(cfg, "pqr_rankcal_v1.use_test_gt")
    _assert_false(cfg, "pqr_rankcal_v1.use_raw_prediction_cache")
    _assert_false(cfg, "pqr_rankcal_v1.physical_time_postprocess_claim")
    assert cfg.pqr_rankcal_v1.claim_map_improvement is False
    assert cfg.pqr_rankcal_v1.official_map_claim is False
    assert cfg.pqr_rankcal_v1.experiment_boundary == "adapter_actionformer_backend_ranking_calibration_only"
    assert cfg.pqr_rankcal_v1.c3_selector_input_experiment is False
    assert cfg.pqr_rankcal_v1.requires_c3_selector_tree_for_input_experiment is True


def _validate_diagnostic_gate(cfg):
    assert cfg.pqr_rankcal_v1.diagnostic_only is True
    assert cfg.pqr_rankcal_v1.get("formal_fulltrain", False) is False
    assert cfg.pqr_rankcal_v1.get("user_override_fulltrain", False) is False
    assert cfg.pqr_rankcal_v1.remote_launch_locked is True
    assert cfg.pqr_rankcal_v1.precheck_scope == EXPECTED_PRECHECK_SCOPE
    assert cfg.pqr_rankcal_v1.build_only_status == EXPECTED_BUILD_ONLY_STATUS
    assert "pseudo_boundary" in cfg.pqr_rankcal_v1.build_only_blockers
    assert "restoration" in cfg.pqr_rankcal_v1.build_only_blockers
    assert "remote PRECHECK" in cfg.pqr_rankcal_v1.build_only_blockers


def _validate_formal_fulltrain_gate(cfg):
    assert cfg.pqr_rankcal_v1.diagnostic_only is False
    assert cfg.pqr_rankcal_v1.formal_fulltrain is True
    assert cfg.pqr_rankcal_v1.user_override_fulltrain is True
    assert cfg.pqr_rankcal_v1.remote_launch_locked is False
    assert cfg.pqr_rankcal_v1.fulltrain_scope == EXPECTED_FULLTRAIN_SCOPE
    assert cfg.workflow.end_epoch == 60
    assert cfg.workflow.max_train_iters is None
    assert cfg.workflow.val_eval_interval == 2
    assert cfg.workflow.val_start_epoch == 40
    assert cfg.workflow.disable_checkpoint is False
    assert cfg.scheduler.max_epoch == 60
    assert cfg.scheduler.warmup_epoch == 5
    assert cfg.post_processing.save_dict is True


def _validate_random_fixed_backend(cfg):
    assert cfg.pqr_rankcal_v1.backend_control == "random_fixed_adapter_50pct"
    assert cfg.dense_window_size == 768
    assert _load_frame_step(cfg, "train").method == "random_fixed_subsample"
    assert _load_frame_step(cfg, "train").method_base == "random_trunc"
    assert _load_frame_step(cfg, "train").target_len == 384
    assert _load_frame_step(cfg, "train").source_len == 768
    assert _load_frame_step(cfg, "val").method == "random_fixed_subsample"
    assert _load_frame_step(cfg, "val").method_base == "sliding_window"
    assert _load_frame_step(cfg, "test").method == "random_fixed_subsample"
    assert _load_frame_step(cfg, "test").method_base == "sliding_window"
    assert cfg.dataset.val.window_size == 768
    assert cfg.dataset.test.window_size == 768


def _validate_stride2_uniform_backend(cfg):
    assert cfg.pqr_rankcal_v1.backend_control == "adapter_stride2_uniform_50pct"
    assert cfg.dataset.train.sample_stride == 2
    assert cfg.dataset.val.sample_stride == 2
    assert cfg.dataset.test.sample_stride == 2
    assert _load_frame_step(cfg, "train").method == "random_trunc"
    assert _load_frame_step(cfg, "train").trunc_len == 384
    assert _load_frame_step(cfg, "val").method == "sliding_window"
    assert _load_frame_step(cfg, "test").method == "sliding_window"
    assert cfg.dataset.val.window_size == 384
    assert cfg.dataset.test.window_size == 384


def validate_config(config_path):
    cfg = Config.fromfile(config_path)
    _validate_route(cfg)
    validate_detector_consumes_model_keys(cfg, config_path)
    validate_runtime_max_train_iters_gate_consumed()
    validate_clean_clone_transform_dependencies_present()
    _validate_quality_head(cfg)

    assert cfg.model.rpn_head.type == "ActionFormerHead"
    assert cfg.model.backbone.backbone.type == "VisionTransformerAdapter"
    assert cfg.model.backbone.backbone.total_frames == 384
    assert cfg.model.projection.max_seq_len == 384
    assert cfg.window_size == 384
    assert cfg.chunk_num == 24

    _validate_route_contract(cfg)
    if cfg.pqr_rankcal_v1.get("formal_fulltrain", False):
        _validate_formal_fulltrain_gate(cfg)
    else:
        _validate_diagnostic_gate(cfg)

    if cfg.route_variant == "C3_PQR_RankCalV1_MaxIoU_Stride2UniformBackendControl":
        _validate_stride2_uniform_backend(cfg)
    else:
        _validate_random_fixed_backend(cfg)

    assert cfg.post_processing.nms.max_seg_num == 2000
    assert "pvr_qc_diagnostics" not in cfg.post_processing

    print(f"PASS_C3_PQR_RANKCAL_V1_CONFIG {Path(config_path).as_posix()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="C3 PQR RankCal V1 config path")
    args = parser.parse_args()
    validate_config(args.config)


if __name__ == "__main__":
    main()
