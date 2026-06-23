import importlib.util
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "adatad"
    / "thumos"
    / "ctf_bdi_pc_ot_mras_p2_quality_rank_calibrator_v0_formal_train_candidate.py"
)
LAUNCHER = (
    ROOT
    / "scripts"
    / "run_ctf_bdi_pc_ot_mras_p2_quality_rank_calibrator_v0_formal_train_n16r4.sbatch"
)
VALIDATOR_PATH = ROOT / "tools" / "bata" / "validate_p2qr_formal_train_gate.py"
GUARD_PATH = ROOT / "opentad" / "utils" / "training_guard.py"


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _good_gate_payload(manifest="manifest-sha", resolved="resolved-sha"):
    return {
        "decision": "ALLOW_P2QR_FORMAL_TRAIN",
        "route": "CTF-BDI/PC-OT-MRAS",
        "active_sha256_manifest_sha256": manifest,
        "resolved_config_sha256": resolved,
        "max_epochs": 60,
        "val_start_epoch": 40,
        "val_eval_interval": 2,
        "allow_slurm": True,
        "allow_gpu": True,
        "allow_tools_train": True,
        "single_gpu": True,
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
        "formal_eval": False,
        "allow_formal_eval": False,
        "checkpoint_load": False,
        "allow_checkpoint_load": False,
        "resume": False,
        "allow_resume": False,
        "raw_prediction_cache": False,
        "allow_raw_prediction_cache": False,
        "load_from_raw_predictions": False,
        "save_raw_prediction": False,
        "metric_claim": False,
        "paper_claim": False,
        "runtime_flops_claim": False,
        "deploy_claim": False,
    }


def test_p2qr_formal_config_is_parseable_and_gate_bound(tmp_path, monkeypatch):
    mmengine_config = pytest.importorskip("mmengine.config")
    training_guard = _load_module(GUARD_PATH, "training_guard_for_p2qr_formal_test")

    cfg = mmengine_config.Config.fromfile(str(CONFIG))

    assert cfg.p2_quality_rank_calibrator_v0_gate is None
    assert cfg.r18_pc_ot_mras_aux_diag_gate is None
    gate = cfg.p2_quality_rank_calibrator_v0_formal_train_gate
    assert gate.route == "CTF-BDI/PC-OT-MRAS"
    assert gate.stage == "P2_NIIQ_QualityRank_Calibrator_v0_formal_train_candidate"
    assert gate.allow_detector_training is True
    assert gate.requires_launch_gate is True
    assert gate.launch_gate_passed is True
    assert gate.allow_tools_train is True
    assert gate.allow_tools_test is False
    assert gate.allow_detector_map is False
    assert gate.allow_train_validation_map is True
    assert gate.allow_long_training is True
    assert gate.allow_checkpoint_write is True
    assert gate.allow_checkpoint_load is False
    assert gate.entrypoint_gate_context.required is True
    assert gate.entrypoint_gate_context.allowed_decisions == ("ALLOW_P2QR_FORMAL_TRAIN",)
    assert gate.entrypoint_gate_context.strict_payload_validation is True
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
    assert cfg.model.pc_ot_mras_reader.emit_pair_distribution is False
    assert cfg.model.get("pc_ot_mras_reader_aux_loss", None) is None
    assert cfg.model.rpn_head.area_head.quality_calibration.enable is True
    assert cfg.model.rpn_head.type == "NativeIrregularAreaHeadP2"

    with pytest.raises(RuntimeError, match="missing required entrypoint gate env"):
        training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py")

    gate_json = tmp_path / "p2qr_formal_gate.json"
    gate_json.write_text(json.dumps(_good_gate_payload()), encoding="utf-8")
    gate_sha = hashlib.sha256(gate_json.read_bytes()).hexdigest()
    monkeypatch.setenv("OPENTAD_PCOTMRAS_ENTRYPOINT_GATE_JSON", str(gate_json))
    monkeypatch.setenv("OPENTAD_PCOTMRAS_ENTRYPOINT_GATE_SHA256", gate_sha)
    monkeypatch.setenv("OPENTAD_PCOTMRAS_ACTIVE_MANIFEST_SHA256", "manifest-sha")
    monkeypatch.setenv("OPENTAD_PCOTMRAS_RESOLVED_CONFIG_SHA256", "resolved-sha")

    assert training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py") is None
    with pytest.raises(RuntimeError, match="allow_tools_test=False"):
        training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/test.py")


def test_p2qr_formal_config_has_no_test_time_shortcut_tokens():
    mmengine_config = pytest.importorskip("mmengine.config")
    cfg = mmengine_config.Config.fromfile(str(CONFIG))

    for split in ("train", "val", "test"):
        pipeline_text = repr(cfg.dataset[split].pipeline).lower()
        assert "raw_prediction" not in pipeline_text
        assert "prediction_cache" not in pipeline_text
        assert "teacher" not in pipeline_text
        assert "oracle" not in pipeline_text


def test_p2qr_formal_gate_validator_accepts_bound_payload(tmp_path):
    validator = _load_module(VALIDATOR_PATH, "validate_p2qr_formal_train_gate_test")
    gate_json = tmp_path / "gate.json"
    gate_json.write_text(json.dumps(_good_gate_payload()), encoding="utf-8")
    gate_sha = hashlib.sha256(gate_json.read_bytes()).hexdigest()

    payload = validator.validate_gate_file(
        gate_json=gate_json,
        gate_sha256=gate_sha,
        active_manifest_sha256="manifest-sha",
        resolved_config_sha256="resolved-sha",
    )
    assert payload["decision"] == "ALLOW_P2QR_FORMAL_TRAIN"


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("decision", "ALLOW_P2QR_SHORT_SMOKE", "decision is not allowed"),
        ("allow_tools_train", False, "allow_tools_train=true"),
        ("tools_test", True, "tools_test=false/absent"),
        ("detector_map", True, "detector_map=false/absent"),
        ("raw_prediction_cache", True, "raw_prediction_cache=false/absent"),
        ("checkpoint_load", True, "checkpoint_load=false/absent"),
        ("resume", True, "resume=false/absent"),
        ("metric_claim", True, "metric_claim=false/absent"),
        ("max_epochs", 2, "max_epochs=60"),
        ("val_eval_interval", -1, "val_eval_interval=2"),
    ],
)
def test_p2qr_formal_gate_validator_rejects_unsafe_payloads(key, value, match):
    validator = _load_module(VALIDATOR_PATH, "validate_p2qr_formal_train_gate_negative_test")
    payload = _good_gate_payload()
    payload[key] = value
    with pytest.raises(ValueError, match=match):
        validator.validate_gate_payload(payload, "manifest-sha", "resolved-sha")


def test_p2qr_formal_gate_validator_rejects_unknown_keys():
    validator = _load_module(VALIDATOR_PATH, "validate_p2qr_formal_train_gate_unknown_test")
    payload = _good_gate_payload()
    payload["allow_secret_eval_mode"] = False
    with pytest.raises(ValueError, match="unknown or unallowlisted key"):
        validator.validate_gate_payload(payload, "manifest-sha", "resolved-sha")


def test_p2qr_formal_launcher_is_clean_repo_gate_bound_and_fail_closed():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "#SBATCH --gpus=1" in text
    assert "#SBATCH -J p2qr_formal" in text
    assert "OpenTAD_PCOTMRAS_R16_R18_R20_Clean_20260619_1730" in text
    assert "refusing dirty historical OpenTAD_BATA_Clean path" in text
    assert "expected branch $EXPECTED_GIT_BRANCH" in text
    assert "tracked clean repo files are modified" in text
    assert "ctf_bdi_pc_ot_mras_p2_quality_rank_calibrator_v0_formal_train_candidate.py" in text
    assert 'PRECHECK_ONLY="${PRECHECK_ONLY:-1}"' in text
    assert 'ALLOW_P2QR_FORMAL_TRAIN="${ALLOW_P2QR_FORMAL_TRAIN:-0}"' in text
    assert "PRECHECK_ONLY=0 requires ALLOW_P2QR_FORMAL_TRAIN=1" in text
    assert "PRECHECK_ONLY=1 must not set ALLOW_P2QR_FORMAL_TRAIN=1" in text
    assert "P2QR_FORMAL_TRAIN_GATE_JSON" in text
    assert "P2QR formal gate SHA256 mismatch" in text
    assert "--active-manifest-sha256 \"$ACTIVE_MANIFEST_SHA256\"" in text
    assert "--resolved-config-sha256 \"$RESOLVED_CONFIG_SHA256\"" in text
    assert "OPENTAD_PCOTMRAS_ENTRYPOINT_GATE_JSON" in text
    assert "OPENTAD_PCOTMRAS_RESOLVED_CONFIG_SHA256" in text
    assert "validate_p2qr_formal_train_gate.py" in text
    assert "resolved_config_dependency_count" in text
    assert "RESOLVED_CONFIG_SHA256" in text
    assert "opentad/models/dense_heads/native_irregular_area_head_p2.py" in text
    assert "tools/test.py" in text
    assert 'tools/test.py "$CONFIG"' not in text
    assert 'tools/train.py "$CONFIG"' in text
    assert "RAW_CACHE_STDOUT_PATTERN" in text
    assert "load_from_raw_predictions|RAW_PREDICTION_CACHE|PREDICTION_CACHE" not in text
    assert "P2QR_FORMAL_PRECHECK_ONLY_PASS_NO_TRAIN" in text
    assert "P2QR_FORMAL_TRAIN_PASS_NO_DIRECT_TEST_NO_CLAIMS" in text
