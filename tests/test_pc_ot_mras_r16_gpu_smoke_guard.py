import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "adatad" / "thumos" / "ctf_bdi_pc_ot_mras_r16_gpu_smoke_candidate.py"
GUARD_PATH = ROOT / "opentad" / "utils" / "training_guard.py"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_mmengine_config():
    mmengine_config = pytest.importorskip("mmengine.config")
    assert CONFIG.exists()
    return mmengine_config.Config.fromfile(str(CONFIG))


def test_r16_gpu_smoke_config_is_parseable_and_scope_locked():
    cfg = _load_mmengine_config()
    guard = _load_module("training_guard_for_r16_gpu_smoke_test", GUARD_PATH)

    assert cfg.r14_pc_ot_mras_train_candidate_gate is None
    gate = cfg.r16_pc_ot_mras_gpu_smoke_gate
    assert gate.stage == "R16_gpu_slurm_smoke_candidate"
    assert gate.smoke_only is True
    assert gate.launch_gate_passed is True
    assert gate.allow_long_training is False
    assert gate.allow_tools_train is True
    assert gate.allow_tools_test is False
    assert gate.allow_detector_map is False
    assert gate.max_epochs == 1
    assert gate.max_train_iters == 2

    assert cfg.workflow.end_epoch == 1
    assert cfg.workflow.max_train_iters == 2
    assert cfg.workflow.checkpoint_interval == 1
    assert cfg.workflow.val_eval_interval == -1
    assert cfg.workflow.val_loss_interval == -1
    assert cfg.workflow.val_start_epoch == 999
    assert cfg.inference.load_from_raw_predictions is False
    assert cfg.inference.save_raw_prediction is False

    assert guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py") is None


def test_r16_gpu_smoke_guard_rejects_long_training_override():
    cfg = _load_mmengine_config()
    guard = _load_module("training_guard_for_r16_long_override_test", GUARD_PATH)
    cfg.workflow.end_epoch = 60

    with pytest.raises(RuntimeError, match="exceeds smoke max_epochs"):
        guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py")


def test_r16_gpu_smoke_guard_rejects_iter_expansion_override():
    cfg = _load_mmengine_config()
    guard = _load_module("training_guard_for_r16_iter_override_test", GUARD_PATH)
    cfg.workflow.max_train_iters = 3

    with pytest.raises(RuntimeError, match="exceeds smoke max_train_iters"):
        guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py")


def test_r16_gpu_smoke_guard_rejects_smoke_gate_tampering():
    cfg = _load_mmengine_config()
    guard = _load_module("training_guard_for_r16_tamper_test", GUARD_PATH)
    cfg.r16_pc_ot_mras_gpu_smoke_gate.smoke_only = False
    cfg.workflow.end_epoch = 60

    with pytest.raises(RuntimeError, match="smoke gate requires smoke_only=True"):
        guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py")

    cfg = _load_mmengine_config()
    cfg.r16_pc_ot_mras_gpu_smoke_gate.allow_detector_map = True
    with pytest.raises(RuntimeError, match="allow_detector_map"):
        guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py")


def test_r16_gpu_smoke_guard_rejects_eval_and_raw_prediction_shortcuts():
    guard = _load_module("training_guard_for_r16_eval_raw_test", GUARD_PATH)
    cfg = _load_mmengine_config()
    cfg.workflow.val_eval_interval = 1
    with pytest.raises(RuntimeError, match="val_eval_interval"):
        guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py")

    cfg = _load_mmengine_config()
    cfg.workflow.val_loss_interval = 1
    with pytest.raises(RuntimeError, match="val_loss_interval"):
        guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py")

    cfg = _load_mmengine_config()
    cfg.inference.load_from_raw_predictions = True
    with pytest.raises(RuntimeError, match="load_from_raw_predictions"):
        guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py")

    cfg = _load_mmengine_config()
    cfg.inference.save_raw_prediction = True
    with pytest.raises(RuntimeError, match="save_raw_prediction"):
        guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py")


def test_r16_gpu_smoke_guard_rejects_non_train_entrypoint():
    cfg = _load_mmengine_config()
    guard = _load_module("training_guard_for_r16_entrypoint_test", GUARD_PATH)

    with pytest.raises(RuntimeError, match="allow_tools_test"):
        guard.assert_detector_training_allowed(cfg, entrypoint="tools/test.py")


def test_r16_gpu_smoke_guard_rejects_tools_test_even_if_entrypoint_is_added():
    cfg = _load_mmengine_config()
    guard = _load_module("training_guard_for_r16_test_entrypoint_tamper_test", GUARD_PATH)
    cfg.r16_pc_ot_mras_gpu_smoke_gate.allowed_entrypoints = ("tools/train.py", "tools/test.py")

    with pytest.raises(RuntimeError, match="allow_tools_test"):
        guard.assert_detector_training_allowed(cfg, entrypoint="tools/test.py")
