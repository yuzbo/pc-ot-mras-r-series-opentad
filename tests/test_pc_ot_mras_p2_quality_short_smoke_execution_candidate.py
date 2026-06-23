import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "adatad"
    / "thumos"
    / "ctf_bdi_pc_ot_mras_p2_quality_rank_calibrator_v0_short_smoke_exec_candidate.py"
)
LAUNCHER = (
    ROOT
    / "scripts"
    / "run_ctf_bdi_pc_ot_mras_p2_quality_rank_calibrator_v0_short_smoke_execution_candidate_n16r4.sbatch"
)
GUARD_PATH = ROOT / "opentad" / "utils" / "training_guard.py"
TRAIN_PATH = ROOT / "tools" / "train.py"
VALIDATOR_PATH = ROOT / "tools" / "bata" / "validate_p2qr_short_smoke_execution_gate.py"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_config():
    mmengine_config = pytest.importorskip("mmengine.config")
    assert CONFIG.exists()
    return mmengine_config.Config.fromfile(str(CONFIG))


def _valid_execution_payload(active="a" * 64, resolved="b" * 64):
    return {
        "decision": "PASS_ALLOW_BOUNDED_P2QR_SLURM_SHORT_SMOKE_ONLY",
        "route": "CTF-BDI/PC-OT-MRAS",
        "active_sha256_manifest_sha256": active,
        "resolved_config_sha256": resolved,
        "max_epochs": 1,
        "max_train_iters": 2,
        "allow_slurm": True,
        "allow_gpu": True,
        "allow_tools_train": True,
        "single_gpu": True,
        "allow_dataset_access": True,
        "allow_pretrained_initialization": True,
        "disable_checkpoint": True,
        "allow_tools_test": False,
        "allow_detector_map": False,
        "allow_train_validation_map": False,
        "allow_long_training": False,
        "allow_checkpoint_access": False,
        "allow_checkpoint_write": False,
        "raw_prediction_cache": False,
        "allow_metric_claim": False,
        "allow_paper_claim": False,
        "allow_runtime_flops_claim": False,
        "allow_deploy_claim": False,
    }


def test_p2qr_short_smoke_execution_config_is_gate_bound(tmp_path, monkeypatch):
    cfg = _load_config()
    training_guard = _load_module("training_guard_for_p2qr_exec_test", GUARD_PATH)

    assert cfg.p2_quality_rank_calibrator_v0_gate is None
    gate = cfg.p2_quality_rank_calibrator_v0_short_smoke_gate
    assert gate.stage == "P2_NIIQ_QualityRank_Calibrator_v0_short_smoke_execution_candidate"
    assert gate.execution_candidate is True
    assert gate.launch_scaffold_only is False
    assert gate.smoke_only is True
    assert gate.allow_detector_training is True
    assert gate.requires_launch_gate is True
    assert gate.launch_gate_passed is True
    assert gate.allow_slurm is True
    assert gate.allow_gpu is True
    assert gate.allow_tools_train is True
    assert gate.allow_tools_test is False
    assert gate.allow_detector_map is False
    assert gate.allow_train_validation_map is False
    assert gate.allow_long_training is False
    assert gate.allow_dataset_access is True
    assert gate.allow_pretrained_initialization is True
    assert gate.allow_checkpoint_access is False
    assert gate.allow_checkpoint_write is False
    assert gate.disable_checkpoint is True
    assert gate.entrypoint_gate_context.required is True
    assert tuple(gate.entrypoint_gate_context.allowed_decisions) == (
        "PASS_ALLOW_BOUNDED_P2QR_SLURM_SHORT_SMOKE_ONLY",
    )
    assert gate.entrypoint_gate_context.strict_payload_validation is True
    assert gate.entrypoint_gate_context.required_exact_values.max_epochs == 1
    assert gate.entrypoint_gate_context.required_exact_values.max_train_iters == 2
    assert "disable_checkpoint" in tuple(gate.entrypoint_gate_context.required_true_keys)
    assert gate.entrypoint_gate_context.unknown_key_policy == "reject_unknown_except_explicit_harmless_metadata"
    assert tuple(gate.allowed_entrypoints) == ("tools/train.py",)
    assert cfg.workflow.end_epoch == 1
    assert cfg.workflow.max_train_iters == 2
    assert cfg.workflow.val_eval_interval == -1
    assert cfg.workflow.val_loss_interval == -1
    assert cfg.workflow.disable_checkpoint is True
    assert cfg.inference.load_from_raw_predictions is False
    assert cfg.inference.save_raw_prediction is False
    assert cfg.model.pc_ot_mras_reader_aux_loss is None
    assert cfg.model.rpn_head.area_head.quality_calibration.enable is True

    with pytest.raises(RuntimeError, match="missing required entrypoint gate env"):
        training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py")

    gate_json = tmp_path / "p2qr_exec_gate.json"
    gate_json.write_text(json.dumps(_valid_execution_payload()), encoding="utf-8")
    gate_sha = hashlib.sha256(gate_json.read_bytes()).hexdigest()
    monkeypatch.setenv("OPENTAD_PCOTMRAS_ENTRYPOINT_GATE_JSON", str(gate_json))
    monkeypatch.setenv("OPENTAD_PCOTMRAS_ENTRYPOINT_GATE_SHA256", gate_sha)
    monkeypatch.setenv("OPENTAD_PCOTMRAS_ACTIVE_MANIFEST_SHA256", "a" * 64)
    monkeypatch.setenv("OPENTAD_PCOTMRAS_RESOLVED_CONFIG_SHA256", "b" * 64)

    assert training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py") is None
    with pytest.raises(RuntimeError, match="allow_tools_test=False"):
        training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/test.py")


@pytest.mark.parametrize("resume_value", ["checkpoint/epoch_59.pth", ""])
def test_p2qr_short_smoke_execution_rejects_direct_resume_arg_before_training(resume_value):
    cfg = _load_config()
    training_guard = _load_module("training_guard_for_p2qr_exec_resume_arg_test", GUARD_PATH)

    with pytest.raises(RuntimeError, match="--resume"):
        training_guard.assert_safe_entrypoint_args_for_gated_config(
            cfg,
            SimpleNamespace(resume=resume_value),
            entrypoint="tools/train.py",
        )


def test_p2qr_short_smoke_execution_guard_rejects_bad_gate_payload(tmp_path, monkeypatch):
    cfg = _load_config()
    training_guard = _load_module("training_guard_for_p2qr_exec_bad_gate_test", GUARD_PATH)
    gate_json = tmp_path / "p2qr_exec_gate_bad.json"
    payload = _valid_execution_payload()
    payload["decision"] = "PASS_ALLOW_REMOTE_PRECHECK_ONLY_ENV_CHECK"
    gate_json.write_text(json.dumps(payload), encoding="utf-8")
    gate_sha = hashlib.sha256(gate_json.read_bytes()).hexdigest()
    monkeypatch.setenv("OPENTAD_PCOTMRAS_ENTRYPOINT_GATE_JSON", str(gate_json))
    monkeypatch.setenv("OPENTAD_PCOTMRAS_ENTRYPOINT_GATE_SHA256", gate_sha)
    monkeypatch.setenv("OPENTAD_PCOTMRAS_ACTIVE_MANIFEST_SHA256", "a" * 64)
    monkeypatch.setenv("OPENTAD_PCOTMRAS_RESOLVED_CONFIG_SHA256", "b" * 64)

    with pytest.raises(RuntimeError, match="entrypoint gate decision is not allowed"):
        training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py")


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda p: p.update(max_train_iters=3), "max_train_iters=2"),
        (lambda p: p.update(max_epochs=2), "max_epochs=1"),
        (lambda p: p.update(allow_tools_test="true"), "allow_tools_test=false/absent"),
        (lambda p: p.update(allow_checkpoint_load=1), "allow_checkpoint_load=false/absent"),
        (lambda p: p.pop("disable_checkpoint"), "disable_checkpoint=true"),
        (lambda p: p.update(hidden_permission=False), "unknown or unallowlisted key"),
    ],
)
def test_p2qr_short_smoke_execution_entrypoint_gate_is_strict(tmp_path, monkeypatch, mutator, message):
    cfg = _load_config()
    training_guard = _load_module("training_guard_for_p2qr_exec_strict_gate_test", GUARD_PATH)
    gate_json = tmp_path / "p2qr_exec_gate_strict.json"
    payload = _valid_execution_payload()
    mutator(payload)
    gate_json.write_text(json.dumps(payload), encoding="utf-8")
    gate_sha = hashlib.sha256(gate_json.read_bytes()).hexdigest()
    monkeypatch.setenv("OPENTAD_PCOTMRAS_ENTRYPOINT_GATE_JSON", str(gate_json))
    monkeypatch.setenv("OPENTAD_PCOTMRAS_ENTRYPOINT_GATE_SHA256", gate_sha)
    monkeypatch.setenv("OPENTAD_PCOTMRAS_ACTIVE_MANIFEST_SHA256", "a" * 64)
    monkeypatch.setenv("OPENTAD_PCOTMRAS_RESOLVED_CONFIG_SHA256", "b" * 64)

    with pytest.raises(RuntimeError, match=message):
        training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py")


def test_p2qr_short_smoke_execution_gate_validator_accepts_bound_payload():
    validator = _load_module("p2qr_execution_gate_validator_accept_test", VALIDATOR_PATH)
    payload = _valid_execution_payload()

    assert validator.validate_gate_payload(payload, "a" * 64, "b" * 64) is True


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda p: p.update(decision="PASS_ALLOW_REMOTE_PRECHECK_ONLY_ENV_CHECK"), "decision"),
        (lambda p: p.pop("resolved_config_sha256"), "resolved_config_sha256"),
        (lambda p: p.update(resolved_config_sha256="c" * 64), "resolved config sha256 mismatch"),
        (lambda p: p.update(max_epochs=2), "max_epochs=1"),
        (lambda p: p.update(max_train_iters=3), "max_train_iters=2"),
        (lambda p: p.update(allow_dataset_access=False), "allow_dataset_access=true"),
        (lambda p: p.update(allow_pretrained_initialization=False), "allow_pretrained_initialization=true"),
        (lambda p: p.update(disable_checkpoint=False), "disable_checkpoint=true"),
        (lambda p: p.update(allow_checkpoint_write=True), "allow_checkpoint_write=false/absent"),
        (lambda p: p.update(allow_tools_test=True), "allow_tools_test=false/absent"),
        (lambda p: p.update(raw_prediction_cache=True), "raw_prediction_cache=false/absent"),
        (lambda p: p.update(safe_note="not allowlisted"), "unknown or unallowlisted key"),
    ],
)
def test_p2qr_short_smoke_execution_gate_validator_rejects_unsafe_payloads(mutator, message):
    validator = _load_module("p2qr_execution_gate_validator_reject_test", VALIDATOR_PATH)
    payload = _valid_execution_payload()
    mutator(payload)

    with pytest.raises(ValueError, match=message):
        validator.validate_gate_payload(payload, "a" * 64, "b" * 64)


def test_p2qr_short_smoke_execution_gate_validator_checks_file_hash(tmp_path):
    validator = _load_module("p2qr_execution_gate_validator_hash_test", VALIDATOR_PATH)
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps(_valid_execution_payload()), encoding="utf-8")
    good_sha = validator.sha256_file(gate_path)

    assert validator.validate_gate_file(gate_path, good_sha, "a" * 64, "b" * 64)["max_train_iters"] == 2

    with pytest.raises(ValueError, match="sha256 mismatch"):
        validator.validate_gate_file(gate_path, "0" * 64, "a" * 64, "b" * 64)


def test_p2qr_short_smoke_execution_gate_validator_cli(tmp_path):
    validator = _load_module("p2qr_execution_gate_validator_cli_test", VALIDATOR_PATH)
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps(_valid_execution_payload()), encoding="utf-8")
    good_sha = validator.sha256_file(gate_path)

    ok = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR_PATH),
            "--gate-json",
            str(gate_path),
            "--gate-sha256",
            good_sha,
            "--active-manifest-sha256",
            "a" * 64,
            "--resolved-config-sha256",
            "b" * 64,
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert ok.returncode == 0, ok.stderr
    assert "P2QR_SHORT_SMOKE_EXECUTION_GATE_VALIDATION_PASS" in ok.stdout


def test_p2qr_short_smoke_execution_launcher_is_fail_closed_textually():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "#SBATCH --gpus=1" in text
    assert "#SBATCH -J p2qr_exec" in text
    assert "OpenTAD_PCOTMRAS_R16_R18_R20_Clean_20260619_1730" in text
    assert "refusing dirty historical OpenTAD_BATA_Clean path" in text
    assert "ctf_bdi_pc_ot_mras_p2_quality_rank_calibrator_v0_short_smoke_exec_candidate.py" in text
    assert 'PRECHECK_ONLY="${PRECHECK_ONLY:-1}"' in text
    assert 'ALLOW_P2QR_SHORT_SMOKE="${ALLOW_P2QR_SHORT_SMOKE:-0}"' in text
    assert "PRECHECK_ONLY=0 requires ALLOW_P2QR_SHORT_SMOKE=1" in text
    assert "PRECHECK_ONLY=1 must not set ALLOW_P2QR_SHORT_SMOKE=1" in text
    assert "P2QR_SHORT_SMOKE_GATE_JSON" in text
    assert "validate_p2qr_short_smoke_execution_gate.py" in text
    assert "before data/pretrained access" in text
    assert text.index('python tools/bata/validate_p2qr_short_smoke_execution_gate.py \\\n  --gate-json "$P2QR_SHORT_SMOKE_GATE_JSON"') < text.index('test -f "$THUMOS14_ANNOTATION_PATH"')
    assert text.index('python tools/bata/validate_p2qr_short_smoke_execution_gate.py \\\n  --gate-json "$P2QR_SHORT_SMOKE_GATE_JSON"') < text.index('log "python/torch/cuda probe"')
    assert "OPENTAD_PCOTMRAS_ENTRYPOINT_GATE_JSON" in text
    assert "OPENTAD_PCOTMRAS_RESOLVED_CONFIG_SHA256" in text
    assert "resolved_config_dependency_count" in text
    assert "checkpoint writing is disabled" in text
    assert "unexpected checkpoint artifact" in text
    assert "tools/test.py is not approved" in text
    assert 'tools/test.py "$CONFIG"' not in text
    assert 'tools/train.py "$CONFIG" --id "$TRAIN_ID" --seed "$SEED"' in text
    assert "model.projection.pretrained" not in text
    assert "result_detection.json" in text
    assert "P2QR_SHORT_SMOKE_EXECUTION_PRECHECK_PASS_NO_TRAIN" in text
    assert "P2QR_SHORT_SMOKE_EXECUTION_PASS_NO_EVAL_NO_CLAIMS" in text


def test_p2qr_short_smoke_train_entrypoint_can_disable_checkpoints_textually():
    train_text = TRAIN_PATH.read_text(encoding="utf-8")
    guard_text = GUARD_PATH.read_text(encoding="utf-8")

    assert 'disable_checkpoint = cfg.workflow.get("disable_checkpoint", False)' in train_text
    assert "if not disable_checkpoint and (" in train_text
    assert "if not disable_checkpoint and args.rank == 0" in train_text
    assert "assert_safe_entrypoint_args_for_gated_config" in train_text
    assert "workflow.disable_checkpoint=True" in guard_text
