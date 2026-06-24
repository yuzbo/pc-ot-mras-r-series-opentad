import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "adatad"
    / "thumos"
    / "pc_ot_mras_frontend_joint_soft_transport_actionformer_adapter_candidate_n16r4.py"
)
LAUNCHER = ROOT / "scripts" / "run_pc_ot_mras_frontend_joint_soft_transport_actionformer_adapter_n16r4.sbatch"
VALIDATOR_PATH = ROOT / "tools" / "bata" / "validate_pc_ot_mras_frontend_joint_e2e_gate.py"
GUARD_PATH = ROOT / "opentad" / "utils" / "training_guard.py"


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha_text(path, text):
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _good_gate_payload(manifest="manifest-sha", resolved="resolved-sha", pretrained_sha="pretrained-sha"):
    return {
        "decision": "ALLOW_PC_OT_MRAS_FRONTEND_JOINT_E2E_SOFT_TRANSPORT_TRAIN",
        "route": "pc_ot_mras_frontend_original_adatad",
        "active_sha256_manifest_sha256": manifest,
        "resolved_config_sha256": resolved,
        "pretrained_sha256": pretrained_sha,
        "budget": 384,
        "dense_window_size": 768,
        "max_epochs": 60,
        "val_start_epoch": 40,
        "val_eval_interval": 2,
        "allow_slurm": True,
        "allow_gpu": True,
        "single_gpu": True,
        "allow_soft_transport": True,
        "allow_joint_reader_detector_training": True,
        "allow_tools_train": True,
        "allow_dataset_access": True,
        "allow_pretrained_initialization": True,
        "allow_checkpoint_write": True,
        "allow_train_validation_map": True,
        "allow_long_training": True,
        "tools_test": False,
        "allow_tools_test": False,
        "direct_tools_test": False,
        "detector_map": False,
        "allow_detector_map": False,
        "checkpoint_load": False,
        "allow_checkpoint_load": False,
        "resume": False,
        "allow_resume": False,
        "offline_ledger": False,
        "frozen_reader_ledger": False,
        "raw_prediction_cache": False,
        "allow_raw_prediction_cache": False,
        "load_from_raw_predictions": False,
        "save_raw_prediction": False,
        "uses_gt": False,
        "uses_teacher": False,
        "uses_oracle": False,
        "uses_raw_prediction": False,
        "metric_claim": False,
        "metric_claim_allowed": False,
        "paper_claim": False,
        "paper_claim_allowed": False,
        "runtime_flops_claim": False,
        "runtime_flops_claim_allowed": False,
        "deploy_claim": False,
        "deploy_claim_allowed": False,
    }


def test_frontend_joint_e2e_config_is_trainable_soft_transport_and_gate_bound(tmp_path, monkeypatch):
    mmengine_config = pytest.importorskip("mmengine.config")
    training_guard = _load_module(GUARD_PATH, "training_guard_for_frontend_joint_e2e_test")

    pretrained = tmp_path / "videomae_pretrained.pth"
    pretrained_sha = _sha_text(pretrained, "pretrained bytes")
    monkeypatch.setenv("PC_OT_MRAS_FRONTEND_JOINT_E2E_PRETRAINED_PATH", str(pretrained))
    cfg = mmengine_config.Config.fromfile(str(CONFIG))

    assert cfg.r17_pc_ot_mras_formal_train_gate is None
    assert cfg.r35_pc_ot_mras_actionformer_head_gate is None
    assert cfg.window_size == 384
    assert cfg.dense_window_size == 768
    assert cfg.experiment_scope.stage == "joint_soft_transport_actionformer_adapter_train_fixed50"
    assert cfg.experiment_scope.reader_trainable is True
    assert cfg.experiment_scope.selector_gradient is True
    assert cfg.experiment_scope.uses_offline_ledger is False
    assert cfg.experiment_scope.frozen_reader_selector_only is False
    assert cfg.experiment_scope.changes_input_sampling is True
    assert cfg.experiment_scope.changes_detector_head is False
    assert cfg.experiment_scope.changes_loss_assignment is False
    assert cfg.experiment_scope.changes_neck is True
    assert cfg.experiment_scope.paper_claim_allowed is False

    gate = cfg.pc_ot_mras_frontend_joint_e2e_gate
    assert gate.allow_detector_training is True
    assert gate.requires_launch_gate is True
    assert gate.launch_gate_passed is True
    assert gate.allow_tools_train is True
    assert gate.allow_tools_test is False
    assert gate.allow_detector_map is False
    assert gate.allow_joint_reader_detector_training is True
    assert gate.allow_soft_transport is True
    assert gate.entrypoint_gate_context.required is True
    assert gate.entrypoint_gate_context.allowed_decisions == (
        "ALLOW_PC_OT_MRAS_FRONTEND_JOINT_E2E_SOFT_TRANSPORT_TRAIN",
    )
    assert gate.entrypoint_gate_context.sha256_file_bindings[0]["gate_key"] == "pretrained_sha256"
    assert tuple(gate.allowed_entrypoints) == ("tools/train.py",)

    assert cfg.workflow.end_epoch == 60
    assert cfg.workflow.val_start_epoch == 40
    assert cfg.workflow.val_eval_interval == 2
    assert cfg.inference.load_from_raw_predictions is False
    assert cfg.inference.save_raw_prediction is False
    assert cfg.model.pc_ot_mras_reader.type == "PCOTMRASReader"
    assert cfg.model.pc_ot_mras_reader.num_slots == 384
    assert cfg.model.neck.type == "PCOTMRASDetectorBridge"
    assert cfg.model.rpn_head.type == "ActionFormerHead"
    assert "pc_ot_mras_reader_aux_loss" not in cfg.model
    assert "pc_ot_mras_reader_eval_override" not in cfg.model

    for split in ("train", "val", "test"):
        pipeline_text = repr(cfg.dataset[split].pipeline).lower()
        assert "bata_value_transport_ledger_subsample" not in pipeline_text
        assert "hard_positions" not in pipeline_text
        assert "raw_prediction" not in pipeline_text
        assert "teacher" not in pipeline_text
        assert "oracle" not in pipeline_text

    with pytest.raises(RuntimeError, match="missing required entrypoint gate env"):
        training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py")

    gate_json = tmp_path / "joint_e2e_gate.json"
    gate_json.write_text(json.dumps(_good_gate_payload(pretrained_sha=pretrained_sha)), encoding="utf-8")
    gate_sha = hashlib.sha256(gate_json.read_bytes()).hexdigest()
    monkeypatch.setenv("OPENTAD_PCOTMRAS_FRONTEND_JOINT_E2E_GATE_JSON", str(gate_json))
    monkeypatch.setenv("OPENTAD_PCOTMRAS_FRONTEND_JOINT_E2E_GATE_SHA256", gate_sha)
    monkeypatch.setenv("OPENTAD_PCOTMRAS_FRONTEND_JOINT_E2E_ACTIVE_MANIFEST_SHA256", "manifest-sha")
    monkeypatch.setenv("OPENTAD_PCOTMRAS_FRONTEND_JOINT_E2E_RESOLVED_CONFIG_SHA256", "resolved-sha")

    assert training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py") is None
    with pytest.raises(RuntimeError, match="allow_tools_test=False"):
        training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/test.py")


def test_frontend_joint_e2e_train_guard_rejects_pretrained_sha_mismatch(tmp_path, monkeypatch):
    mmengine_config = pytest.importorskip("mmengine.config")
    training_guard = _load_module(GUARD_PATH, "training_guard_for_frontend_joint_e2e_sha_test")

    pretrained = tmp_path / "videomae_pretrained.pth"
    _sha_text(pretrained, "actual pretrained bytes")
    monkeypatch.setenv("PC_OT_MRAS_FRONTEND_JOINT_E2E_PRETRAINED_PATH", str(pretrained))
    cfg = mmengine_config.Config.fromfile(str(CONFIG))

    gate_json = tmp_path / "bad_joint_e2e_gate.json"
    gate_json.write_text(json.dumps(_good_gate_payload(pretrained_sha="wrong-pretrained-sha")), encoding="utf-8")
    gate_sha = hashlib.sha256(gate_json.read_bytes()).hexdigest()
    monkeypatch.setenv("OPENTAD_PCOTMRAS_FRONTEND_JOINT_E2E_GATE_JSON", str(gate_json))
    monkeypatch.setenv("OPENTAD_PCOTMRAS_FRONTEND_JOINT_E2E_GATE_SHA256", gate_sha)
    monkeypatch.setenv("OPENTAD_PCOTMRAS_FRONTEND_JOINT_E2E_ACTIVE_MANIFEST_SHA256", "manifest-sha")
    monkeypatch.setenv("OPENTAD_PCOTMRAS_FRONTEND_JOINT_E2E_RESOLVED_CONFIG_SHA256", "resolved-sha")

    with pytest.raises(RuntimeError, match="pretrained sha256 mismatch"):
        training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py")


def test_frontend_joint_e2e_gate_validator_accepts_bound_payload(tmp_path):
    validator = _load_module(VALIDATOR_PATH, "validate_frontend_joint_e2e_gate_test")
    gate_json = tmp_path / "gate.json"
    gate_json.write_text(json.dumps(_good_gate_payload()), encoding="utf-8")
    gate_sha = hashlib.sha256(gate_json.read_bytes()).hexdigest()

    payload = validator.validate_gate_file(
        gate_json=gate_json,
        gate_sha256=gate_sha,
        active_manifest_sha256="manifest-sha",
        resolved_config_sha256="resolved-sha",
        pretrained_sha256="pretrained-sha",
        budget=384,
        dense_window_size=768,
    )
    assert payload["decision"] == "ALLOW_PC_OT_MRAS_FRONTEND_JOINT_E2E_SOFT_TRANSPORT_TRAIN"


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("decision", "ALLOW_R35_ACTIONFORMER_HEAD_FORMAL_TRAIN", "decision is not allowed"),
        ("route", "CTF-BDI/PC-OT-MRAS", "route mismatch"),
        ("allow_tools_train", False, "allow_tools_train=true"),
        ("allow_soft_transport", False, "allow_soft_transport=true"),
        ("allow_joint_reader_detector_training", False, "allow_joint_reader_detector_training=true"),
        ("offline_ledger", True, "offline_ledger=false/absent"),
        ("frozen_reader_ledger", True, "frozen_reader_ledger=false/absent"),
        ("tools_test", True, "tools_test=false/absent"),
        ("detector_map", True, "detector_map=false/absent"),
        ("raw_prediction_cache", True, "raw_prediction_cache=false/absent"),
        ("checkpoint_load", True, "checkpoint_load=false/absent"),
        ("uses_gt", True, "uses_gt=false/absent"),
        ("uses_teacher", True, "uses_teacher=false/absent"),
        ("metric_claim", True, "metric_claim=false/absent"),
        ("max_epochs", 2, "max_epochs=60"),
        ("budget", 192, "budget=384"),
        ("dense_window_size", 384, "dense_window_size=768"),
    ],
)
def test_frontend_joint_e2e_gate_validator_rejects_unsafe_payloads(key, value, match):
    validator = _load_module(VALIDATOR_PATH, "validate_frontend_joint_e2e_gate_negative_test")
    payload = _good_gate_payload()
    payload[key] = value
    with pytest.raises(ValueError, match=match):
        validator.validate_gate_payload(
            payload,
            active_manifest_sha256="manifest-sha",
            resolved_config_sha256="resolved-sha",
            pretrained_sha256="pretrained-sha",
            budget=384,
            dense_window_size=768,
        )


def test_frontend_joint_e2e_gate_validator_rejects_unknown_keys():
    validator = _load_module(VALIDATOR_PATH, "validate_frontend_joint_e2e_gate_unknown_test")
    payload = _good_gate_payload()
    payload["allow_secret_eval_mode"] = False
    with pytest.raises(ValueError, match="unknown or unallowlisted key"):
        validator.validate_gate_payload(
            payload,
            active_manifest_sha256="manifest-sha",
            resolved_config_sha256="resolved-sha",
            pretrained_sha256="pretrained-sha",
            budget=384,
            dense_window_size=768,
        )


def test_frontend_joint_e2e_launcher_is_clean_repo_gate_bound_and_no_ledger_dump():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "#SBATCH --gpus=1" in text
    assert "#SBATCH -J pcot_front_e2e" in text
    assert "pc_ot_mras_frontend_joint_soft_transport_actionformer_adapter_candidate_n16r4.py" in text
    assert "PRECHECK_ONLY=0 requires ALLOW_FRONTEND_JOINT_E2E=1" in text
    assert "FRONTEND_JOINT_E2E_GATE_JSON" in text
    assert "validate_pc_ot_mras_frontend_joint_e2e_gate.py" in text
    assert "OPENTAD_PCOTMRAS_FRONTEND_JOINT_E2E_GATE_JSON" in text
    assert "OPENTAD_PCOTMRAS_FRONTEND_JOINT_E2E_RESOLVED_CONFIG_SHA256" in text
    assert "PC_OT_MRAS_FRONTEND_JOINT_E2E_PRETRAINED_PATH" in text
    assert "tools/train.py \"$CONFIG\"" in text
    assert "tools/test.py \"$CONFIG\"" not in text
    assert "dump_pc_ot_mras_reader_snapshots.py" not in text
    assert "convert_pc_ot_mras_hard_positions_to_value_transport_ledger.py" not in text
    assert "ALLOW_PC_OT_MRAS_FRONTEND_JOINT_E2E_SOFT_TRANSPORT_TRAIN" in text
    assert "FRONTEND_JOINT_E2E_PRECHECK_ONLY_PASS_NO_TRAIN" in text
    assert "FRONTEND_JOINT_E2E_TRAIN_PASS_NO_DIRECT_TEST_NO_CLAIMS" in text
