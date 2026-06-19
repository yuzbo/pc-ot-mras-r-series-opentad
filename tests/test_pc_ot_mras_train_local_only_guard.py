import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
GUARD_PATH = ROOT / "opentad" / "utils" / "training_guard.py"
R12_CONFIG_PATH = ROOT / "configs" / "adatad" / "thumos" / "ctf_bdi_pc_ot_mras_r12_p2_optin_adapter_local.py"
R14_CONFIG_PATH = ROOT / "configs" / "adatad" / "thumos" / "ctf_bdi_pc_ot_mras_r14_trainable_candidate.py"
R17_CONFIG_PATH = ROOT / "configs" / "adatad" / "thumos" / "ctf_bdi_pc_ot_mras_r17_formal_train_candidate.py"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


guard = _load_module("training_guard_under_test", GUARD_PATH)


class ConfigLike:
    def __init__(self, data):
        self._cfg_dict = data


def test_r12_pc_ot_mras_local_config_blocks_tools_train_before_runtime_setup():
    config = _load_module("r12_pc_ot_mras_local_config_under_test", R12_CONFIG_PATH)
    cfg = {"r12_pc_ot_mras_gate": config.r12_pc_ot_mras_gate}

    with pytest.raises(RuntimeError) as exc_info:
        guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py")

    message = str(exc_info.value)
    assert "tools/train.py is blocked" in message
    assert "r12_pc_ot_mras_gate" in message
    assert "allow_detector_training=False" in message
    assert "CTF-BDI/PC-OT-MRAS" in message
    assert "before DDP, dataset, model, or runner construction" in message
    assert "synthetic_cpu_forward_loss_smoke" in message


def test_generic_top_level_local_only_gate_blocks_detector_training():
    cfg = {
        "local_only_gate": {
            "route": "local utility smoke",
            "local_synthetic_gate_only": True,
            "allowed_checks": ("static_config_contract",),
        }
    }

    with pytest.raises(RuntimeError) as exc_info:
        guard.assert_detector_training_allowed(cfg)

    message = str(exc_info.value)
    assert "local_only_gate" in message
    assert "local_synthetic_gate_only=True" in message
    assert "static_config_contract" in message


def test_mmengine_like_config_object_with_gate_blocks_detector_training():
    cfg = ConfigLike(
        {
            "experiment_gate": {
                "route": "config object smoke",
                "allow_detector_training": "false",
            }
        }
    )

    with pytest.raises(RuntimeError) as exc_info:
        guard.assert_detector_training_allowed(cfg)

    assert "experiment_gate" in str(exc_info.value)
    assert "allow_detector_training=False" in str(exc_info.value)


def test_mmengine_like_config_object_with_top_level_training_block_is_blocked():
    cfg = ConfigLike({"allow_detector_training": False, "route": "top level block"})

    with pytest.raises(RuntimeError) as exc_info:
        guard.assert_detector_training_allowed(cfg)

    message = str(exc_info.value)
    assert "<top-level>" in message
    assert "allow_detector_training=False" in message
    assert "top level block" in message


def test_legacy_configs_without_explicit_training_block_are_allowed():
    cfg = {
        "model": {
            "type": "ActionFormer",
            "neck": {"type": "FPNIdentity", "allow_detector_training": False},
        },
        "workflow": {"end_epoch": 12},
    }

    assert guard.assert_detector_training_allowed(cfg) is None


def test_training_gate_with_explicit_allow_and_no_local_only_marker_is_allowed():
    cfg = {
        "experiment_gate": {
            "route": "reviewed detector training",
            "allow_detector_training": True,
        }
    }

    assert guard.assert_detector_training_allowed(cfg) is None


def test_trainable_candidate_requires_launch_gate_before_tools_train():
    config = _load_module("r14_pc_ot_mras_trainable_candidate_under_test", R14_CONFIG_PATH)
    cfg = {"r14_pc_ot_mras_train_candidate_gate": config.r14_pc_ot_mras_train_candidate_gate}

    with pytest.raises(RuntimeError) as exc_info:
        guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py")

    message = str(exc_info.value)
    assert "r14_pc_ot_mras_train_candidate_gate" in message
    assert "requires_launch_gate=True and launch_gate_passed!=True" in message
    assert "R14_trainable_config_candidate" in message


def test_launch_gate_passed_training_candidate_is_allowed_by_guard_contract():
    cfg = {
        "r14_pc_ot_mras_train_candidate_gate": {
            "route": "CTF-BDI/PC-OT-MRAS",
            "stage": "reviewed_launch_gate",
            "allow_detector_training": True,
            "requires_launch_gate": True,
            "launch_gate_passed": True,
        }
    }

    assert guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py") is None


def test_r17_formal_candidate_forbids_direct_tools_test_entrypoint():
    config = _load_module("r17_pc_ot_mras_formal_train_candidate_under_test", R17_CONFIG_PATH)
    cfg = {"r17_pc_ot_mras_formal_train_gate": config.r17_pc_ot_mras_formal_train_gate}

    with pytest.raises(RuntimeError) as exc_info:
        guard.assert_detector_training_allowed(cfg, entrypoint="tools/test.py")

    message = str(exc_info.value)
    assert "r17_pc_ot_mras_formal_train_gate" in message
    assert "allow_tools_test=False forbids tools/test.py" in message
    assert "R17_formal_train_candidate" in message


def test_launch_gate_passed_gate_forbids_tools_train_when_flag_is_false():
    cfg = {
        "reviewed_gate": {
            "route": "CTF-BDI/PC-OT-MRAS",
            "stage": "reviewed_no_train_entrypoint",
            "allow_detector_training": True,
            "requires_launch_gate": True,
            "launch_gate_passed": True,
            "allow_tools_train": False,
            "allowed_entrypoints": ("tools/train.py",),
        }
    }

    with pytest.raises(RuntimeError) as exc_info:
        guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py")

    message = str(exc_info.value)
    assert "reviewed_gate" in message
    assert "allow_tools_train=False forbids tools/train.py" in message
