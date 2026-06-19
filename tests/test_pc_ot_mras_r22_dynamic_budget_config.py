import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "adatad" / "thumos" / "ctf_bdi_pc_ot_mras_r22_dynamic_budget_control.py"
GUARD_PATH = ROOT / "opentad" / "utils" / "training_guard.py"


def _load_guard():
    spec = importlib.util.spec_from_file_location("training_guard_for_r22_test", GUARD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_r22_dynamic_budget_config_is_parseable_and_launch_blocked():
    mmengine_config = pytest.importorskip("mmengine.config")
    training_guard = _load_guard()

    cfg = mmengine_config.Config.fromfile(str(CONFIG))

    assert cfg.r20_pc_ot_mras_value_only_control_gate is None
    gate = cfg.r22_pc_ot_mras_dynamic_budget_control_gate
    assert gate.route == "CTF-BDI/PC-OT-MRAS"
    assert gate.stage == "R22_value_to_dynamic_budget_control_candidate"
    assert gate.deploy_visible_reader_outputs_only is True
    assert gate.dynamic_budget_protocol_candidate is True
    assert gate.dynamic_budget_validation is False
    assert gate.local_synthetic_gate_only is True
    assert gate.allow_detector_training is False
    assert gate.allow_tools_train is False
    assert gate.allow_tools_test is False
    assert gate.allow_detector_map is False
    assert gate.allow_remote_sync is False
    assert gate.allow_slurm is False
    assert gate.allow_gpu is False
    assert gate.dynamic_budget_claim_allowed is False
    assert gate.metric_claim_allowed is False
    assert gate.paper_claim_allowed is False

    controller = cfg.pc_ot_mras_dynamic_budget_controller
    assert controller.type == "PCOTMRASDynamicBudgetController"
    assert tuple(controller.budget_values) == (288, 320, 352, 384, 416)
    assert tuple(controller.budget_thresholds) == (0.20, 0.35, 0.50, 0.65)
    assert controller.require_value_logits is True
    assert controller.coverage_share <= controller.max_coverage_share <= 0.65

    assert cfg.model.pc_ot_mras_reader.type == "PCOTMRASReader"
    assert cfg.model.pc_ot_mras_reader.enable_value_heads is True
    assert cfg.model.pc_ot_mras_reader_value_loss.enabled is True
    assert cfg.inference.load_from_raw_predictions is False
    assert cfg.inference.save_raw_prediction is False

    with pytest.raises(RuntimeError, match="allow_detector_training=False"):
        training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py")
    with pytest.raises(RuntimeError, match="allow_detector_training=False"):
        training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/test.py")


def test_r22_config_does_not_enable_test_time_shortcut_sources():
    mmengine_config = pytest.importorskip("mmengine.config")
    cfg = mmengine_config.Config.fromfile(str(CONFIG))

    for split in ("train", "val", "test"):
        pipeline_text = repr(cfg.dataset[split].pipeline).lower()
        assert "raw_prediction" not in pipeline_text
        assert "prediction_cache" not in pipeline_text
        assert "teacher" not in pipeline_text
        assert "oracle" not in pipeline_text
