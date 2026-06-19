import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "adatad" / "thumos" / "ctf_bdi_pc_ot_mras_r17_formal_train_candidate.py"
GUARD_PATH = ROOT / "opentad" / "utils" / "training_guard.py"


def _load_guard():
    spec = importlib.util.spec_from_file_location("training_guard_for_r17_formal_test", GUARD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_r17_formal_train_config_is_parseable_and_trainable():
    mmengine_config = pytest.importorskip("mmengine.config")
    training_guard = _load_guard()

    cfg = mmengine_config.Config.fromfile(str(CONFIG))

    assert cfg.r14_pc_ot_mras_train_candidate_gate is None
    gate = cfg.r17_pc_ot_mras_formal_train_gate
    assert gate.route == "CTF-BDI/PC-OT-MRAS"
    assert gate.stage == "R17_formal_train_candidate"
    assert gate.allow_detector_training is True
    assert gate.requires_launch_gate is True
    assert gate.launch_gate_passed is True
    assert gate.allow_tools_train is True
    assert gate.allow_tools_test is False
    assert gate.allow_long_training is True
    assert gate.metric_claim_allowed is False
    assert gate.paper_claim_allowed is False
    assert tuple(gate.allowed_entrypoints) == ("tools/train.py",)

    assert cfg.workflow.end_epoch == 60
    assert cfg.workflow.val_start_epoch == 40
    assert cfg.workflow.val_eval_interval == 2
    assert cfg.workflow.checkpoint_interval == 2
    assert cfg.solver.train.batch_size == 2
    assert cfg.inference.load_from_raw_predictions is False
    assert cfg.inference.save_raw_prediction is False
    assert cfg.model.pc_ot_mras_reader.type == "PCOTMRASReader"
    assert "pc_ot_mras_reader_aux_loss" not in cfg.model
    assert cfg.model.neck.type == "PCOTMRASDetectorBridge"
    assert cfg.model.rpn_head.type == "NativeIrregularAreaHeadP2"

    assert training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py") is None


def test_r17_formal_train_config_has_no_test_time_shortcut_tokens():
    mmengine_config = pytest.importorskip("mmengine.config")
    cfg = mmengine_config.Config.fromfile(str(CONFIG))

    for split in ("train", "val", "test"):
        pipeline_text = repr(cfg.dataset[split].pipeline).lower()
        assert "raw_prediction" not in pipeline_text
        assert "prediction_cache" not in pipeline_text
        assert "teacher" not in pipeline_text
        assert "oracle" not in pipeline_text
