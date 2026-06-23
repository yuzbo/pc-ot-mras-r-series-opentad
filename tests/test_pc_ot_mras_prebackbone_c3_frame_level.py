from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
C3_CONFIG = (
    ROOT
    / "configs"
    / "adatad"
    / "thumos"
    / "pc_ot_mras_prebackbone_c3_f1_lr_tinytransformer_st_original_adatad.py"
)


def test_c3_config_file_freezes_frame_level_tinytransformer_variant():
    assert C3_CONFIG.exists()
    text = C3_CONFIG.read_text(encoding="utf-8")

    assert 'variant_id = "C3-F1-LR-TinyTransformer-ST-OriginalAdaTAD"' in text
    assert "window_size = 384" in text
    assert "dense_window_size = 768" in text
    assert "selection_unit = 1" in text
    assert "selection_unit2_supported = True" in text
    assert "scout_spatial_size = 32" in text
    assert "scout_feature_source=\"compressed_pixels\"" in text
    assert "scout_descriptor_dim = 3 * scout_spatial_size * scout_spatial_size" in text
    assert 'type="PCOTMRASTinyTransformerFrameScout"' in text
    assert "straight_through_downstream=True" in text
    assert "low_resolution_scout=True" in text
    assert "OriginalAdaTAD" in text
    assert "post_projection_bridge=False" in text
    assert "PCOTMRASDetectorBridge" not in text


def test_selector_source_exposes_optional_cnn_frame_scout_reader():
    selector_path = ROOT / "opentad" / "models" / "selectors" / "pc_ot_mras_prebackbone_frame_selector.py"
    text = selector_path.read_text(encoding="utf-8")

    assert "class PCOTMRASCNNFrameScout" in text
    assert '"PCOTMRASCNNFrameScout"' in text
    assert "Conv1d" in text
    assert "slot_logits" in text
    assert "acquisition_matrix" in text


def test_c3_config_loads_original_adatad_backend_and_forbids_shortcuts():
    mmengine_config = pytest.importorskip("mmengine.config")
    cfg = mmengine_config.Config.fromfile(str(C3_CONFIG))

    assert cfg.variant_id == "C3-F1-LR-TinyTransformer-ST-OriginalAdaTAD"
    assert cfg.window_size == cfg.model.frame_selector.target_len == 384
    assert cfg.dense_window_size == cfg.model.frame_selector.dense_window_size == 768
    assert cfg.model.frame_selector.selection_unit == 1
    assert cfg.model.frame_selector.selection_unit2_supported is True
    assert cfg.model.frame_selector.reader.type == "PCOTMRASTinyTransformerFrameScout"
    assert cfg.model.frame_selector.scout_feature_source == "compressed_pixels"
    assert cfg.model.frame_selector.scout_spatial_size == 32
    assert cfg.model.frame_selector.descriptor_dim == 3 * 32 * 32
    assert cfg.model.frame_selector.reader.in_dim == 3 * 32 * 32
    assert cfg.model.frame_selector.reader.dropout == 0.0
    assert cfg.model.frame_selector.straight_through_downstream is True
    assert cfg.model.frame_selector.transport_topk == 1
    assert cfg.model.frame_selector.eval_transport_topk == 1
    assert cfg.experiment_scope.variant_id == cfg.variant_id
    assert cfg.experiment_scope.free_frame_level_selector is True
    assert cfg.experiment_scope.protected_scaffold is False
    assert cfg.experiment_scope.c2_320_uniform_plus_64_residual is False
    assert cfg.experiment_scope.s80r16_cell96x4_enabled is False
    assert "not protected scaffold" in cfg.experiment_scope.boundary_lock
    assert cfg.experiment_scope.low_resolution_scout is True
    assert cfg.experiment_scope.scout_feature_source == "compressed_pixels"
    assert cfg.experiment_scope.scout_input == "compressed_low_resolution_frame_pixels_before_videomae"
    assert cfg.experiment_scope.detector_stack == "original_adatad_actionformer_adapter"
    assert cfg.experiment_scope.uses_offline_ledger is False
    assert cfg.experiment_scope.uses_p2 is False
    assert cfg.experiment_scope.uses_teacher is False
    assert cfg.experiment_scope.uses_test_gt is False
    assert cfg.inference.load_from_raw_predictions is False
    assert cfg.inference.save_raw_prediction is False
    assert cfg.pc_ot_mras_prebackbone_e2e_acquisition_gate.formal_train_candidate is False
    assert cfg.pc_ot_mras_prebackbone_e2e_acquisition_gate.launch_gate_passed is False
    assert cfg.pc_ot_mras_prebackbone_e2e_acquisition_gate.allow_tools_train is False
    assert cfg.pc_ot_mras_prebackbone_e2e_acquisition_gate.allow_tools_test is False
    assert cfg.pc_ot_mras_prebackbone_e2e_acquisition_gate.allow_detector_map is False
    assert cfg.pc_ot_mras_prebackbone_e2e_acquisition_gate.allow_long_training is False
    assert cfg.pc_ot_mras_prebackbone_e2e_acquisition_gate.allowed_entrypoints == ()
    assert cfg.dataset.train.window_size == 768
    assert cfg.dataset.val.window_size == 768
    assert cfg.dataset.test.window_size == 768
