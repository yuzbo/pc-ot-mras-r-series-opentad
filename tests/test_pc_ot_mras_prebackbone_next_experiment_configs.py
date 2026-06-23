from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "adatad" / "thumos"
C2_CONFIG = CONFIG_DIR / "pc_ot_mras_prebackbone_c2_learned_residual64_tinytransformer_st_original_adatad.py"


def _load_cfg():
    mmengine_config = pytest.importorskip("mmengine.config")
    return mmengine_config.Config.fromfile(str(C2_CONFIG))


def test_c2_learned_residual_config_loads_and_declares_local_selector_gap():
    assert C2_CONFIG.exists()
    cfg = _load_cfg()

    assert cfg.variant_id == "C2-LearnedResidual64-TinyTransformer-ST-OriginalAdaTAD"
    assert cfg.work_dir.endswith("pc_ot_mras_prebackbone_c2_learned_residual64_tinytransformer_st_original_adatad")
    assert cfg.experiment_scope.variant_id == cfg.variant_id
    assert cfg.experiment_scope.stage == "c2_learned_residual64_tinytransformer_local_implementation_candidate"
    assert cfg.experiment_scope.detector_stack == "original_adatad_actionformer_adapter"
    assert cfg.experiment_scope.backend == "OriginalAdaTAD"
    assert cfg.experiment_scope.selection_surface == "pre_backbone_raw_frame"
    assert cfg.experiment_scope.selection_timing == "online_before_backbone"
    assert cfg.experiment_scope.selector_support_status == "supported_by_prebackbone_frame_selector"
    assert cfg.experiment_scope.residual_slot_policy == "learned_residual_tinytransformer"
    assert cfg.experiment_scope.free_frame_level_selector is False
    assert cfg.experiment_scope.protected_scaffold is True
    assert cfg.experiment_scope.c2_320_uniform_plus_64_residual is True
    assert cfg.experiment_scope.s80r16_cell96x4_enabled is False
    assert "C2" in cfg.experiment_scope.boundary_lock
    assert "not C3" in cfg.experiment_scope.boundary_lock
    assert cfg.experiment_scope.local_config_candidate is False
    assert cfg.experiment_scope.local_implementation_candidate is True
    assert cfg.experiment_scope.changes_input_sampling is True
    assert cfg.experiment_scope.changes_detector_head is False
    assert cfg.experiment_scope.changes_neck is False
    assert cfg.experiment_scope.changes_loss_assignment is False
    assert cfg.experiment_scope.changes_post_processing is False
    assert cfg.experiment_scope.deploy_claim_allowed is False
    assert cfg.experiment_scope.runtime_flops_claim_allowed is False
    assert cfg.experiment_scope.paper_claim_allowed is False

    frame_selector = cfg.model.frame_selector
    assert frame_selector.type == "PCOTMRASPreBackboneFrameSelector"
    assert cfg.window_size == frame_selector.target_len == 384
    assert cfg.dense_window_size == frame_selector.dense_window_size == 768
    assert frame_selector.protected_uniform_count == 320
    assert frame_selector.residual_count == 64
    assert frame_selector.residual_slot_role == "learned_residual"
    assert frame_selector.selector_support_status == "supported_by_prebackbone_frame_selector"
    assert frame_selector.scout_feature_source == "compressed_pixels"
    assert frame_selector.scout_spatial_size == 32
    assert frame_selector.descriptor_dim == 3 * 32 * 32
    assert frame_selector.reader.type == "PCOTMRASTinyTransformerFrameScout"
    assert frame_selector.reader.in_dim == 3 * 32 * 32
    assert frame_selector.reader.num_slots == 64
    assert frame_selector.reader.dropout == 0.0
    assert "residual_slot_count" not in frame_selector.reader
    assert "protected_uniform_count" not in frame_selector.reader
    assert frame_selector.straight_through_downstream is True
    assert frame_selector.transport_topk == 1
    assert frame_selector.eval_transport_topk == 1


def test_c2_config_forbids_test_time_shortcuts_training_launch_and_metric_claims():
    cfg = _load_cfg()
    gate = cfg.pc_ot_mras_prebackbone_e2e_acquisition_gate
    forbidden = set(gate.entrypoint_gate_context.forbidden_true_keys)

    assert cfg.experiment_scope.uses_p2 is False
    assert cfg.experiment_scope.uses_offline_ledger is False
    assert cfg.experiment_scope.uses_teacher is False
    assert cfg.experiment_scope.uses_test_gt is False
    assert cfg.experiment_scope.uses_raw_prediction_cache is False
    assert cfg.inference.load_from_raw_predictions is False
    assert cfg.inference.save_raw_prediction is False
    assert gate.stage == "c2_learned_residual64_tinytransformer_local_implementation_candidate_locked"
    assert gate.formal_train_candidate is False
    assert gate.allow_detector_training is False
    assert gate.requires_launch_gate is True
    assert gate.launch_gate_passed is False
    assert gate.allow_remote_sync is False
    assert gate.allow_slurm is False
    assert gate.allow_gpu is False
    assert gate.allow_tools_train is False
    assert gate.allow_tools_test is False
    assert gate.allow_detector_map is False
    assert gate.allow_train_validation_map is False
    assert gate.allow_long_training is False
    assert gate.metric_claim_allowed is False
    assert gate.paper_claim_allowed is False
    assert gate.runtime_flops_claim_allowed is False
    assert gate.deploy_claim_allowed is False
    assert tuple(gate.allowed_entrypoints) == ()

    for key in (
        "tools_test",
        "allow_tools_test",
        "detector_map",
        "allow_detector_map",
        "offline_ledger",
        "allow_offline_ledger",
        "raw_prediction_cache",
        "allow_raw_prediction_cache",
        "load_from_raw_predictions",
        "save_raw_prediction",
        "uses_teacher",
        "uses_oracle",
        "uses_test_gt",
        "uses_raw_prediction",
        "metric_claim",
        "paper_claim",
        "runtime_flops_claim",
        "deploy_claim",
    ):
        assert key in forbidden

    for split in ("train", "val", "test"):
        pipeline_text = repr(cfg.dataset[split].pipeline).lower()
        assert "bata_value_transport_ledger_subsample" not in pipeline_text
        assert "hard_positions" not in pipeline_text
        assert "raw_prediction" not in pipeline_text
        assert "teacher" not in pipeline_text
        assert "oracle" not in pipeline_text
