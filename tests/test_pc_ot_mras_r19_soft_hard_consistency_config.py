import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "adatad" / "thumos" / "ctf_bdi_pc_ot_mras_r19_soft_hard_consistency_candidate.py"
GUARD_PATH = ROOT / "opentad" / "utils" / "training_guard.py"


def _load_guard():
    spec = importlib.util.spec_from_file_location("training_guard_for_r19_test", GUARD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_r19_soft_hard_consistency_config_is_parseable_and_launch_blocked():
    mmengine_config = pytest.importorskip("mmengine.config")
    training_guard = _load_guard()

    cfg = mmengine_config.Config.fromfile(str(CONFIG))

    assert cfg.r18_pc_ot_mras_aux_diag_gate is None
    gate = cfg.r19_pc_ot_mras_soft_hard_consistency_gate
    assert gate.route == "CTF-BDI/PC-OT-MRAS"
    assert gate.stage == "R19_soft_hard_consistency_candidate"
    assert gate.train_only_soft_hard_consistency is True
    assert gate.local_synthetic_gate_only is True
    assert gate.allow_detector_training is False
    assert gate.requires_launch_gate is True
    assert gate.launch_gate_passed is False
    assert gate.allow_tools_train is False
    assert gate.allow_tools_test is False
    assert gate.allow_detector_map is False
    assert gate.metric_claim_allowed is False
    assert gate.paper_claim_allowed is False

    assert cfg.model.pc_ot_mras_reader.type == "PCOTMRASReader"
    assert cfg.model.pc_ot_mras_reader_aux_loss.enabled is True
    assert cfg.model.pc_ot_mras_reader_soft_hard_loss.enabled is True
    assert cfg.model.pc_ot_mras_reader_soft_hard_loss.weights.slot_allocation == 0.02
    assert cfg.model.neck.type == "PCOTMRASDetectorBridge"
    assert cfg.inference.load_from_raw_predictions is False
    assert cfg.inference.save_raw_prediction is False

    with pytest.raises(RuntimeError, match="allow_detector_training=False"):
        training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py")
    with pytest.raises(RuntimeError, match="allow_detector_training=False"):
        training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/test.py")


def test_r19_config_has_no_test_time_shortcut_tokens():
    mmengine_config = pytest.importorskip("mmengine.config")
    cfg = mmengine_config.Config.fromfile(str(CONFIG))

    for split in ("train", "val", "test"):
        pipeline_text = repr(cfg.dataset[split].pipeline).lower()
        assert "raw_prediction" not in pipeline_text
        assert "prediction_cache" not in pipeline_text
        assert "teacher" not in pipeline_text
        assert "oracle" not in pipeline_text
