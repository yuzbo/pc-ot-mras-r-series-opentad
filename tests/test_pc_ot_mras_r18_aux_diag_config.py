import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "adatad" / "thumos" / "ctf_bdi_pc_ot_mras_r18_aux_diag_candidate.py"
GUARD_PATH = ROOT / "opentad" / "utils" / "training_guard.py"


def _load_guard():
    spec = importlib.util.spec_from_file_location("training_guard_for_r18_aux_diag_test", GUARD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_r18_aux_diag_config_is_parseable_opt_in_and_launch_blocked():
    mmengine_config = pytest.importorskip("mmengine.config")
    training_guard = _load_guard()

    cfg = mmengine_config.Config.fromfile(str(CONFIG))

    assert cfg.r17_pc_ot_mras_formal_train_gate is None
    gate = cfg.r18_pc_ot_mras_aux_diag_gate
    assert gate.route == "CTF-BDI/PC-OT-MRAS"
    assert gate.stage == "R18_train_only_reader_aux_diagnostic_candidate"
    assert gate.default_off is True
    assert gate.explicit_config_opt_in is True
    assert gate.train_only_auxiliary_diagnostic is True
    assert gate.allow_detector_training is True
    assert gate.requires_launch_gate is True
    assert gate.launch_gate_passed is False
    assert gate.allow_remote_sync is False
    assert gate.allow_precheck_only is False
    assert gate.allow_slurm is False
    assert gate.allow_gpu is False
    assert gate.metric_claim_allowed is False
    assert gate.paper_claim_allowed is False

    aux_cfg = cfg.model.pc_ot_mras_reader_aux_loss
    assert aux_cfg.enabled is True
    assert aux_cfg.boundary_sigma == 2.0
    assert aux_cfg.short_action_len == 12.0
    assert aux_cfg.adjacent_gap == 8.0
    assert aux_cfg.weights.start == 0.05
    assert aux_cfg.weights.end == 0.05
    assert aux_cfg.weights.pair == 0.05
    assert aux_cfg.weights.allocation == 0.02
    assert cfg.model.pc_ot_mras_reader.type == "PCOTMRASReader"
    assert cfg.model.neck.type == "PCOTMRASDetectorBridge"
    assert cfg.model.rpn_head.type == "NativeIrregularAreaHeadP2"

    with pytest.raises(RuntimeError, match="requires_launch_gate=True"):
        training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py")


def test_r18_aux_diag_config_has_no_test_time_shortcuts_or_external_targets():
    mmengine_config = pytest.importorskip("mmengine.config")
    cfg = mmengine_config.Config.fromfile(str(CONFIG))

    assert cfg.inference.load_from_raw_predictions is False
    assert cfg.inference.save_raw_prediction is False
    assert "gt_segments" in repr(cfg.dataset.train.pipeline)
    assert "gt_segments" in repr(cfg.dataset.val.pipeline)
    assert "gt_segments" not in repr(cfg.dataset.test.pipeline)
    for split in ("train", "val", "test"):
        pipeline_text = repr(cfg.dataset[split].pipeline).lower()
        assert "raw_prediction" not in pipeline_text
        assert "prediction_cache" not in pipeline_text
        assert "teacher" not in pipeline_text
        assert "oracle" not in pipeline_text
    assert "gt_segments" not in repr(cfg.model.pc_ot_mras_reader_aux_loss).lower()
