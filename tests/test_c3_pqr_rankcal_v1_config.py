from pathlib import Path

import pytest
from mmengine.config import Config

from tools.validate_c3_pqr_rankcal_v1_config import validate_config, validate_detector_consumes_model_keys


ROOT = Path(__file__).resolve().parents[1]
PRECHECK_CONFIG = ROOT / "configs/adatad/thumos/c3_indirect_original_adatad_32px_a_pqr_rankcal_v1_precheck.py"
SHORTDIAG_CONFIG = ROOT / "configs/adatad/thumos/c3_indirect_original_adatad_32px_a_pqr_rankcal_v1_shortdiag.py"
EXACT_UNIFORM_CONFIG = (
    ROOT
    / "configs/adatad/thumos/c3_indirect_original_adatad_32px_a_exact_uniform_backend_control_pqr_rankcal_v1_shortdiag.py"
)


def _load(config_path):
    assert config_path.exists(), f"missing config: {config_path}"
    return Config.fromfile(config_path)


def _load_frame_step(cfg, split):
    return next(step for step in cfg.dataset[split].pipeline if step["type"] == "LoadFrames")


@pytest.mark.parametrize("config_path", [PRECHECK_CONFIG, SHORTDIAG_CONFIG, EXACT_UNIFORM_CONFIG])
def test_pqr_rankcal_configs_pass_fail_closed_validator(config_path):
    validate_config(config_path)


@pytest.mark.parametrize("config_path", [PRECHECK_CONFIG, SHORTDIAG_CONFIG, EXACT_UNIFORM_CONFIG])
def test_pqr_rankcal_configs_do_not_define_unconsumed_frame_selector(config_path):
    cfg = _load(config_path)

    assert "frame_selector" not in cfg.model
    validate_detector_consumes_model_keys(cfg, config_path)


def test_pqr_rankcal_shortdiag_is_adapter_backend_ranking_calibration_control():
    cfg = _load(SHORTDIAG_CONFIG)
    quality = cfg.model.rpn_head.quality_head_cfg

    assert cfg.route_label == "C3_MAINLINE_OPTIMIZATION"
    assert cfg.route_family == "C3_ORIGINAL_OPTIMIZATION_ROUTE"
    assert cfg.route_variant == "C3_PQR_RankCalV1_MaxIoU"
    assert cfg.pqr_rankcal_v1.experiment_boundary == "adapter_actionformer_backend_ranking_calibration_only"
    assert cfg.pqr_rankcal_v1.requires_c3_selector_tree_for_input_experiment is True
    assert cfg.model.type == "ActionFormer"
    assert cfg.model.backbone.backbone.type == "VisionTransformerAdapter"
    assert cfg.model.rpn_head.type == "ActionFormerHead"
    assert cfg.inference.load_from_raw_predictions is False
    assert cfg.inference.save_raw_prediction is False
    assert cfg.post_processing.nms.max_seg_num == 2000
    assert _load_frame_step(cfg, "train").method == "random_fixed_subsample"
    assert _load_frame_step(cfg, "val").method == "random_fixed_subsample"
    assert _load_frame_step(cfg, "test").method == "random_fixed_subsample"

    assert quality.enabled is True
    assert quality.target_mode == "max_iou"
    assert quality.weight_init == 0.0
    assert quality.bias_init == pytest.approx(4.59511985013459)
    assert 0.02 <= quality.loss_weight <= 0.05
    assert 0.05 <= quality.score_alpha <= 0.15
    assert "pvr_qc_diagnostics" not in cfg.post_processing


def test_pqr_rankcal_precheck_is_short_fail_closed_gate():
    cfg = _load(PRECHECK_CONFIG)

    assert cfg.workflow.end_epoch == 1
    assert cfg.workflow.max_train_iters == 2
    assert cfg.workflow.disable_checkpoint is True
    assert cfg.workflow.val_eval_interval == -1
    assert cfg.solver.train.batch_size == 1
    assert cfg.solver.static_graph is True
    assert cfg.pqr_rankcal_v1["remote_launch_locked"] is True
    assert cfg.pqr_rankcal_v1["diagnostic_only"] is True
    assert cfg.pqr_rankcal_v1.precheck_scope == "config_validator_plus_quality_head_unit"
    assert cfg.pqr_rankcal_v1.build_only_status == "locked_by_baseline_import_dependencies"
    assert "Rearrange" in cfg.pqr_rankcal_v1.build_only_blockers
    assert "pseudo_boundary" in cfg.pqr_rankcal_v1.build_only_blockers


def test_exact_uniform_control_uses_real_stride2_adapter_backend_control():
    control = _load(EXACT_UNIFORM_CONFIG)

    assert control.route_variant == "C3_PQR_RankCalV1_MaxIoU_Stride2UniformBackendControl"
    assert control.pqr_rankcal_v1.backend_control == "adapter_stride2_uniform_50pct"
    assert control.dataset.train.sample_stride == 2
    assert control.dataset.val.sample_stride == 2
    assert control.dataset.test.sample_stride == 2
    assert _load_frame_step(control, "train").method == "random_trunc"
    assert _load_frame_step(control, "train").trunc_len == 384
    assert _load_frame_step(control, "val").method == "sliding_window"
    assert _load_frame_step(control, "test").method == "sliding_window"
    assert control.dataset.val.window_size == 384
    assert control.dataset.test.window_size == 384
    assert control.model.backbone.backbone.total_frames == 384
    assert control.model.projection.max_seq_len == 384


def test_quality_head_config_fields_are_supported_by_anchor_free_head():
    cfg = _load(SHORTDIAG_CONFIG)
    quality_keys = set(cfg.model.rpn_head.quality_head_cfg.keys())
    source = (ROOT / "opentad/models/dense_heads/anchor_free_head.py").read_text(encoding="utf-8")

    for key in quality_keys:
        assert f'"{key}"' in source, f"quality_head_cfg field is not consumed by AnchorFreeHead: {key}"
    assert '"max_iou"' in source
    assert 'valid_quality_target_modes = {"assigned_iou", "max_iou", "positive_max_iou"}' in source


def test_validator_rejects_forbidden_routes_and_switches(tmp_path):
    bad_config = tmp_path / "bad_pqr.py"
    bad_config.write_text(
        "\n".join(
            [
                f'_base_ = [r"{SHORTDIAG_CONFIG.as_posix()}"]',
                'route_label = "DIVERGENT_INNOVATION_TEST"',
                "inference = dict(load_from_raw_predictions=True, save_raw_prediction=False)",
                "pqr_rankcal_v1 = dict(use_teacher=True, use_test_gt=True, physical_time_postprocess_claim=True)",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(AssertionError):
        validate_config(bad_config)


def test_validator_rejects_unconsumed_frame_selector(tmp_path):
    bad_config = tmp_path / "bad_frame_selector.py"
    bad_config.write_text(
        "\n".join(
            [
                f'_base_ = [r"{SHORTDIAG_CONFIG.as_posix()}"]',
                'model = dict(frame_selector=dict(type="PCOTMRASIndirectPreBackboneFrameSelector"))',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(AssertionError, match="frame_selector"):
        validate_config(bad_config)
