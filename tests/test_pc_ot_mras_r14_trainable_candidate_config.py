import copy
import importlib.util
from collections.abc import Mapping
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "adatad" / "thumos" / "ctf_bdi_pc_ot_mras_r14_trainable_candidate.py"
GUARD_PATH = ROOT / "opentad" / "utils" / "training_guard.py"
OUTPUT_STRIDES = [1, 2, 4, 8, 16, 32]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_mmengine_config():
    mmengine_config = pytest.importorskip("mmengine.config")
    assert CONFIG.exists()
    return mmengine_config.Config.fromfile(str(CONFIG))


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items() if key != "_delete_"}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_plain(item) for item in value)
    return value


def test_r14_trainable_candidate_config_contract_is_parseable_and_still_launch_blocked():
    cfg = _load_mmengine_config()
    guard = _load_module("training_guard_for_r14_config_test", GUARD_PATH)

    assert cfg.r12_pc_ot_mras_gate is None
    gate = cfg.r14_pc_ot_mras_train_candidate_gate
    assert gate.route == "CTF-BDI/PC-OT-MRAS"
    assert gate.stage == "R14_trainable_config_candidate"
    assert gate.reviewed_predecessor == "R13b_Pro_Gemini"
    assert gate.trainable_config_candidate is True
    assert gate.allow_detector_training is True
    assert gate.requires_launch_gate is True
    assert gate.launch_gate_passed is False
    assert gate.allow_remote_sync is False
    assert gate.allow_precheck_only is False
    assert gate.allow_slurm is False
    assert gate.allow_gpu is False
    assert "detector_training_without_launch_gate" in gate.forbidden_checks

    with pytest.raises(RuntimeError, match="requires_launch_gate=True"):
        guard.assert_detector_training_allowed(
            {"r14_pc_ot_mras_train_candidate_gate": gate},
            entrypoint="tools/train.py",
        )


def test_r14_trainable_candidate_preserves_r13_multiscale_geometry_and_raw_prediction_guards():
    cfg = _load_mmengine_config()
    model = cfg.model

    assert model.type == "ActionFormer"
    assert model.pc_ot_mras_reader_feature_level == 0
    assert model.pc_ot_mras_reader.type == "PCOTMRASReader"
    assert model.pc_ot_mras_reader.in_dim == model.projection.out_channels == 512
    assert model.pc_ot_mras_reader.num_slots == 384
    assert model.neck.type == "PCOTMRASDetectorBridge"
    assert model.neck.source_feature_level == 0
    assert list(model.neck.output_strides) == OUTPUT_STRIDES
    assert model.rpn_head.type == "NativeIrregularAreaHeadP2"
    assert list(model.rpn_head.prior_generator.strides) == OUTPUT_STRIDES
    assert model.rpn_head.temporal_grid.required is True
    assert model.rpn_head.temporal_grid.decode_axis == "dense"
    assert model.rpn_head.temporal_grid.positions_key == "irregular_selected_positions"
    assert model.rpn_head.temporal_grid.valid_len_key == "irregular_selected_valid_len"
    assert cfg.inference.load_from_raw_predictions is False
    assert cfg.inference.save_raw_prediction is False


def test_r14_launch_gate_flip_is_the_only_guard_difference_needed_for_future_launch():
    cfg = _load_mmengine_config()
    guard = _load_module("training_guard_for_r14_launch_flip_test", GUARD_PATH)
    gate = _plain(copy.deepcopy(cfg.r14_pc_ot_mras_train_candidate_gate))
    gate["launch_gate_passed"] = True

    assert guard.assert_detector_training_allowed(
        {"r14_pc_ot_mras_train_candidate_gate": gate},
        entrypoint="tools/train.py",
    ) is None
