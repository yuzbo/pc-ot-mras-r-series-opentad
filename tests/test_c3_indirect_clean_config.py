from pathlib import Path

from mmengine.config import Config

from tools.validate_c3_indirect_clean_config import validate_config


ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = ROOT / "configs/adatad/thumos/c3_indirect_original_adatad_32px_a_short_smoke.py"
FULL_CONFIG = ROOT / "configs/adatad/thumos/c3_indirect_original_adatad_32px_a_full_train.py"


def test_clean_c3_indirect_config_has_no_old_route_surfaces():
    cfg = Config.fromfile(SMOKE_CONFIG)
    cfg_text = cfg.pretty_text.lower()

    assert cfg.model.type == "ActionFormer"
    assert cfg.model.rpn_head.type == "ActionFormerHead"
    assert cfg.model.frame_selector.type == "PCOTMRASIndirectPreBackboneFrameSelector"
    assert cfg.model.frame_selector.scout.type == "PCOTMRASCoarseActionnessFrameScout"
    assert cfg.model.frame_selector.strategy == "coarse_actionness_uncertainty"
    assert cfg.model.frame_selector.scout_spatial_size == 32

    for forbidden in [
        "physicalgrid",
        "physical_grid",
        "rf50",
        "bh_sdc",
        "boundary_head",
        "start_head",
        "end_head",
        "teacher",
        "raw_prediction_cache",
        "p2head",
    ]:
        assert forbidden not in cfg_text


def test_clean_c3_indirect_config_keeps_dense_input_but_original_adatad_backend():
    cfg = Config.fromfile(SMOKE_CONFIG)

    assert cfg.c3_dense_window_size == 768
    assert cfg.window_size == 384
    assert cfg.chunk_num == 24
    assert cfg.model.frame_selector.dense_window_size == 768
    assert cfg.model.frame_selector.target_len == 384

    train_load = next(step for step in cfg.dataset.train.pipeline if step["type"] == "LoadFrames")
    assert train_load["trunc_len"] == 768
    assert cfg.dataset.val.window_size == 768
    assert cfg.dataset.test.window_size == 768

    assert cfg.model.backbone.custom.pre_processing_pipeline[0]["t1"] == 24
    assert cfg.model.backbone.custom.post_processing_pipeline[1]["t1"] == 24
    assert cfg.model.backbone.custom.post_processing_pipeline[2]["size"] == 384
    assert cfg.model.backbone.backbone.total_frames == 384
    assert cfg.model.projection.max_seq_len == 384


def test_train_smoke_controls_are_fail_closed():
    cfg = Config.fromfile(SMOKE_CONFIG)

    assert cfg.workflow.end_epoch == 1
    assert cfg.workflow.max_train_iters == 2
    assert cfg.workflow.disable_checkpoint is True
    assert cfg.workflow.val_eval_interval == -1
    assert cfg.solver.train.batch_size == 1
    assert cfg.solver.static_graph is True


def test_full_train_config_keeps_protocol_but_unlocks_training_schedule():
    cfg = Config.fromfile(FULL_CONFIG)
    cfg_text = cfg.pretty_text.lower()

    assert cfg.model.frame_selector.type == "PCOTMRASIndirectPreBackboneFrameSelector"
    assert cfg.model.frame_selector.strategy == "coarse_actionness_uncertainty"
    assert cfg.model.frame_selector.scout_spatial_size == 32
    assert cfg.model.frame_selector.target_len == 384
    assert cfg.model.frame_selector.dense_window_size == 768
    assert cfg.model.rpn_head.type == "ActionFormerHead"
    assert cfg.inference.load_from_raw_predictions is False
    assert cfg.inference.save_raw_prediction is False

    assert cfg.workflow.end_epoch == 60
    assert cfg.workflow.max_train_iters is None
    assert cfg.workflow.disable_checkpoint is False
    assert cfg.workflow.val_eval_interval == 2
    assert cfg.workflow.val_start_epoch == 40
    assert cfg.solver.train.batch_size == 2
    assert cfg.solver.ema is True
    assert cfg.solver.static_graph is True

    for forbidden in [
        "physicalgrid",
        "physical_grid",
        "rf50",
        "bh_sdc",
        "boundary_head",
        "start_head",
        "end_head",
        "teacher",
        "raw_prediction_cache",
        "p2head",
    ]:
        assert forbidden not in cfg_text


def test_clean_c3_indirect_configs_pass_shared_precheck_validator():
    validate_config(SMOKE_CONFIG)
    validate_config(FULL_CONFIG)
