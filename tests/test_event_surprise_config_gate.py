from __future__ import annotations

import hashlib
import json
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

mmengine_config = pytest.importorskip("mmengine.config")


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "adatad" / "thumos"
LOCAL_CONFIG = CONFIG_DIR / "event_surprise_temporal_acquisition_local_precheck.py"
FULL_CONFIG = CONFIG_DIR / "event_surprise_temporal_acquisition_full_train_candidate_n16r4.py"
VALIDATOR = ROOT / "tools" / "bata" / "validate_event_surprise_gate.py"
LAUNCHER = ROOT / "scripts" / "run_event_surprise_temporal_acquisition_precheck_n16r4.sbatch"
FULL_LAUNCHER = ROOT / "scripts" / "run_event_surprise_temporal_acquisition_full_train_n16r4.sbatch"
TRAINING_GUARD = ROOT / "opentad" / "utils" / "training_guard.py"


FORBIDDEN_C3_TOKENS = (
    "pc_ot_mras_prebackbone_c3",
    "PCOTMRASPreBackboneFrameSelector",
    "PCOTMRASMotionTCNFrameScout",
    "PCOTMRASHybridFrameScout",
    "PCOTMRASTinyTransformerFrameScout",
    "PCOTMRASDetectorBridge",
    "BH_SDC",
    "boundary_microscope",
    "BoundaryMicroscope",
    "frame_token_hybrid",
    "FrameTokenHybrid",
)


def _load_validator():
    spec = importlib.util.spec_from_file_location("validate_event_surprise_gate_test", VALIDATOR)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_training_guard():
    spec = importlib.util.spec_from_file_location("training_guard_for_event_surprise_test", TRAINING_GUARD)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _full_train_gate_payload(
    *,
    run_tag="event_surprise_full_train_candidate_gate_test",
    active_manifest_sha256="manifest-sha",
    resolved_config_sha256="resolved-sha",
):
    return {
        "gate_type": "event_surprise_launch_gate",
        "route": "event_surprise_temporal_acquisition",
        "route_label": "DIVERGENT_INNOVATION_EVENT_SURPRISE_DO_NOT_MERGE_WITH_C3",
        "action": "full_train",
        "run_tag": run_tag,
        "user": "skywalker",
        "coordinator_override_statement": "USER_REQUESTED_FAST_EVENT_SURPRISE_FULL_TRAIN_CANDIDATE_NO_PRO_BLOCKER",
        "decision": "ALLOW_EVENT_SURPRISE_FULL_TRAIN_CANDIDATE",
        "config": "configs/adatad/thumos/event_surprise_temporal_acquisition_full_train_candidate_n16r4.py",
        "config_stage": "full_train_candidate_locked",
        "active_sha256_manifest_sha256": active_manifest_sha256,
        "resolved_config_sha256": resolved_config_sha256,
        "allow_slurm": True,
        "allow_gpu": True,
        "allow_tools_train": True,
        "allow_detector_training": True,
        "allow_train_validation_map": True,
        "allow_long_training": True,
        "allow_full_train": True,
        "launch_gate_passed": True,
        "allowed_entrypoints": ["tools/train.py", "full_train"],
        "tools_test": False,
        "allow_tools_test": False,
        "detector_map": False,
        "allow_detector_map": False,
        "formal_eval": False,
        "allow_formal_eval": False,
        "checkpoint_load": False,
        "allow_checkpoint_load": False,
        "resume": False,
        "allow_resume": False,
        "raw_prediction": False,
        "allow_raw_prediction": False,
        "raw_predictions": False,
        "allow_raw_predictions": False,
        "raw_prediction_cache": False,
        "allow_raw_prediction_cache": False,
        "prediction_cache": False,
        "allow_prediction_cache": False,
        "load_from_raw_predictions": False,
        "allow_load_from_raw_predictions": False,
        "save_raw_prediction": False,
        "allow_save_raw_prediction": False,
        "save_raw_predictions": False,
        "allow_save_raw_predictions": False,
        "uses_teacher": False,
        "uses_oracle": False,
        "uses_test_gt": False,
        "uses_raw_prediction": False,
        "metric_claim": False,
        "allow_metric_claim": False,
        "metric_claim_allowed": False,
        "paper_claim": False,
        "allow_paper_claim": False,
        "paper_claim_allowed": False,
        "runtime_flops_claim": False,
        "runtime_flops_claim_allowed": False,
        "deploy_claim": False,
        "deploy_claim_allowed": False,
    }


@pytest.mark.parametrize("config_path", [LOCAL_CONFIG, FULL_CONFIG])
def test_event_surprise_configs_are_fail_closed_and_do_not_contain_old_c3_tokens(config_path):
    cfg = mmengine_config.Config.fromfile(str(config_path))
    gate = cfg.event_surprise_acquisition_gate
    context = gate.entrypoint_gate_context

    assert cfg.experiment_scope.route_label == "DIVERGENT_INNOVATION_EVENT_SURPRISE_DO_NOT_MERGE_WITH_C3"
    assert cfg.experiment_scope.meta_key == "event_surprise_acquisition_plan"
    assert cfg.experiment_scope.protocol_family == "event_surprise_sparse_acquisition_contract"
    assert cfg.experiment_scope.uses_pc_ot_mras_detector_bridge is False
    assert cfg.model.frame_selector.type == "EventSurpriseTemporalAcquisitionSelector"
    assert cfg.model.frame_selector.meta_key == "event_surprise_acquisition_plan"
    assert cfg.model.frame_selector.input_layout == "bct"
    assert cfg.model.frame_selector.remap_gt_to_selected_axis is True
    assert "neck" not in cfg.model

    assert gate.default_off is True
    assert gate.allow_remote_sync is False
    assert gate.allow_tools_test is False
    assert gate.allow_raw_prediction is False
    assert gate.metric_claim_allowed is False
    assert gate.paper_claim_allowed is False

    if config_path == LOCAL_CONFIG:
        assert gate.launch_gate_passed is False
        assert gate.allow_slurm is False
        assert gate.allow_tools_train is False
        assert gate.allow_full_train is False
        assert tuple(gate.allowed_entrypoints) == ()
    else:
        assert gate.full_train_candidate is True
        assert gate.requires_launch_gate is True
        assert gate.launch_gate_passed is True
        assert gate.allow_slurm is True
        assert gate.allow_gpu is True
        assert gate.allow_tools_train is True
        assert gate.allow_detector_training is True
        assert gate.allow_train_validation_map is True
        assert gate.allow_long_training is True
        assert gate.allow_full_train is True
        assert tuple(gate.allowed_entrypoints) == ("tools/train.py",)
        assert context.gate_json_env == "EVENT_SURPRISE_ENTRYPOINT_GATE_JSON"
        assert context.gate_sha256_env == "EVENT_SURPRISE_ENTRYPOINT_GATE_SHA256"
        assert context.active_manifest_sha256_env == "EVENT_SURPRISE_ACTIVE_MANIFEST_SHA256"
        assert context.resolved_config_sha256_env == "EVENT_SURPRISE_RESOLVED_CONFIG_SHA256"

    forbidden = set(context.forbidden_true_keys)
    expected_forbidden = [
        "tools_test",
        "allow_tools_test",
        "raw_prediction",
        "allow_raw_prediction",
        "metric_claim",
        "metric_claim_allowed",
        "paper_claim",
        "paper_claim_allowed",
    ]
    if config_path == LOCAL_CONFIG:
        expected_forbidden.extend(
            [
                "remote_sync",
                "allow_remote_sync",
                "slurm",
                "allow_slurm",
                "tools_train",
                "allow_tools_train",
                "full_train",
                "allow_full_train",
            ]
        )
    for key in expected_forbidden:
        assert key in forbidden

    text = config_path.read_text(encoding="utf-8")
    resolved_text = cfg.pretty_text
    for token in FORBIDDEN_C3_TOKENS:
        assert token not in text
        assert token not in resolved_text


def test_event_surprise_gate_validator_accepts_locked_configs():
    validator = _load_validator()

    for config_path in (LOCAL_CONFIG, FULL_CONFIG):
        payload = validator.validate_config(config_path)
        assert payload["pass"] is True
        expected_entrypoints = [] if config_path == LOCAL_CONFIG else ["tools/train.py"]
        assert payload["allowed_entrypoints"] == expected_entrypoints
        assert payload["route_label"] == "DIVERGENT_INNOVATION_EVENT_SURPRISE_DO_NOT_MERGE_WITH_C3"
        assert payload["selector"] == "EventSurpriseTemporalAcquisitionSelector"
        assert payload["input_layout"] == "bct"


def test_event_surprise_gate_validator_cli_accepts_locked_configs():
    for config_path in (LOCAL_CONFIG, FULL_CONFIG):
        result = subprocess.run(
            [sys.executable, str(VALIDATOR), "--config", str(config_path)],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert '"pass": true' in result.stdout
        assert "EventSurpriseTemporalAcquisitionSelector" in result.stdout


def test_event_surprise_launch_action_precheck_allows_local_precheck_only():
    result = subprocess.run(
        [sys.executable, str(VALIDATOR), "--config", str(LOCAL_CONFIG), "--action", "precheck-only"],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["launch_action"] == "precheck_only"
    assert payload["launch_allowed"] is True
    assert payload["train_command_allowed"] is False


def test_event_surprise_launch_action_full_train_fails_without_gate_json():
    result = subprocess.run(
        [sys.executable, str(VALIDATOR), "--config", str(FULL_CONFIG), "--action", "full-train"],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode != 0
    assert "full_train launch requires --gate-json" in result.stderr


def test_event_surprise_launch_action_full_train_accepts_explicit_gate_json(tmp_path):
    gate_json = tmp_path / "event_surprise_full_train_gate.json"
    gate_json.write_text(json.dumps(_full_train_gate_payload()), encoding="utf-8")
    gate_sha = hashlib.sha256(gate_json.read_bytes()).hexdigest()

    result = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--config",
            str(FULL_CONFIG),
            "--action",
            "full-train",
            "--gate-json",
            str(gate_json),
            "--gate-sha256",
            gate_sha,
            "--active-manifest-sha256",
            "manifest-sha",
            "--resolved-config-sha256",
            "resolved-sha",
            "--run-tag",
            "event_surprise_full_train_candidate_gate_test",
        ],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["launch_action"] == "full_train"
    assert payload["launch_allowed"] is True
    assert payload["train_command_allowed"] is True
    assert payload["gate_json"] == str(gate_json)
    assert payload["active_sha256_manifest_sha256"] == "manifest-sha"
    assert payload["resolved_config_sha256"] == "resolved-sha"
    assert payload["run_tag"] == "event_surprise_full_train_candidate_gate_test"


def test_event_surprise_launch_action_full_train_rejects_gate_missing_hash_bindings(tmp_path):
    payload = _full_train_gate_payload()
    payload.pop("active_sha256_manifest_sha256")
    payload.pop("resolved_config_sha256")
    gate_json = tmp_path / "missing_hash_gate.json"
    gate_json.write_text(json.dumps(payload), encoding="utf-8")
    gate_sha = hashlib.sha256(gate_json.read_bytes()).hexdigest()

    result = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--config",
            str(FULL_CONFIG),
            "--action",
            "full-train",
            "--gate-json",
            str(gate_json),
            "--gate-sha256",
            gate_sha,
            "--active-manifest-sha256",
            "manifest-sha",
            "--resolved-config-sha256",
            "resolved-sha",
            "--run-tag",
            "event_surprise_full_train_candidate_gate_test",
        ],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode != 0
    assert "active_sha256_manifest_sha256" in result.stderr


def test_event_surprise_launch_action_full_train_rejects_wrong_route_gate_json(tmp_path):
    gate_json = tmp_path / "wrong_route_gate.json"
    payload = _full_train_gate_payload()
    payload["route_label"] = "C3_RS_NOT_EVENT_SURPRISE"
    gate_json.write_text(json.dumps(payload), encoding="utf-8")
    gate_sha = hashlib.sha256(gate_json.read_bytes()).hexdigest()

    result = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--config",
            str(FULL_CONFIG),
            "--action",
            "full-train",
            "--gate-json",
            str(gate_json),
            "--gate-sha256",
            gate_sha,
            "--active-manifest-sha256",
            "manifest-sha",
            "--resolved-config-sha256",
            "resolved-sha",
            "--run-tag",
            "event_surprise_full_train_candidate_gate_test",
        ],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode != 0
    assert "launch gate route_label mismatch" in result.stderr


def test_event_surprise_full_train_config_is_guarded_by_external_entrypoint_gate(tmp_path, monkeypatch):
    training_guard = _load_training_guard()
    cfg = mmengine_config.Config.fromfile(str(FULL_CONFIG))

    gate = cfg.event_surprise_acquisition_gate
    assert gate.full_train_candidate is True
    assert gate.allow_detector_training is True
    assert gate.allow_tools_train is True
    assert gate.allow_tools_test is False
    assert gate.requires_launch_gate is True
    assert gate.entrypoint_gate_context.required is True
    assert gate.entrypoint_gate_context.allowed_decisions == ("ALLOW_EVENT_SURPRISE_FULL_TRAIN_CANDIDATE",)
    assert tuple(gate.allowed_entrypoints) == ("tools/train.py",)

    with pytest.raises(RuntimeError, match="missing required entrypoint gate env EVENT_SURPRISE_ENTRYPOINT_GATE_JSON"):
        training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py")

    gate_json = tmp_path / "event_surprise_full_train_gate.json"
    gate_json.write_text(json.dumps(_full_train_gate_payload()), encoding="utf-8")
    gate_sha = hashlib.sha256(gate_json.read_bytes()).hexdigest()
    monkeypatch.setenv("EVENT_SURPRISE_ENTRYPOINT_GATE_JSON", str(gate_json))
    monkeypatch.setenv("EVENT_SURPRISE_ENTRYPOINT_GATE_SHA256", gate_sha)
    monkeypatch.setenv("EVENT_SURPRISE_ACTIVE_MANIFEST_SHA256", "manifest-sha")
    monkeypatch.setenv("EVENT_SURPRISE_RESOLVED_CONFIG_SHA256", "resolved-sha")

    assert training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py") is None
    with pytest.raises(RuntimeError, match="allow_tools_test=False"):
        training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/test.py")


def test_event_surprise_gate_validator_rejects_unlocked_payload():
    validator = _load_validator()
    payload = {
        "route_label": "DIVERGENT_INNOVATION_EVENT_SURPRISE_DO_NOT_MERGE_WITH_C3",
        "default_off": True,
        "launch_gate_passed": False,
        "allow_remote_sync": True,
        "allowed_entrypoints": [],
        "entrypoint_gate_context": {"forbidden_true_keys": validator.FORBIDDEN_TRUE_KEYS},
    }

    with pytest.raises(validator.EventSurpriseGateError, match="allow_remote_sync"):
        validator.validate_gate_payload(payload)


def test_event_surprise_n16r4_launcher_is_precheck_default_and_full_train_fail_closed():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "PRECHECK_ONLY=\"${PRECHECK_ONLY:-1}\"" in text
    assert "OpenTAD_EventSurprise_PrecheckDeploy_20260624_126e04e" in text
    assert "--action precheck-only" in text
    assert "--action full-train --gate-json" in text
    assert "EVENT_SURPRISE_FULL_TRAIN_GATE_JSON" in text
    assert "EVENT_SURPRISE_FULL_TRAIN_GATE_SHA256" in text
    assert "ALLOW_EVENT_SURPRISE_FULL_TRAIN_CANDIDATE" in text
    assert "EVENT_SURPRISE_ENABLE_TOOLS_TRAIN_AFTER_GATE=1" in text
    assert "python tools/train.py" not in text
    assert "tools/test.py" not in text
    assert "python -m pytest \\\n  tests/test_event_surprise_config_gate.py \\" in text
    assert "tests/test_event_surprise_acquisition_route.py \\\n  -q" not in text
    for token in FORBIDDEN_C3_TOKENS:
        assert token not in text


def test_event_surprise_n16r4_full_train_launcher_is_fail_closed_and_runs_train_only_after_gate():
    text = FULL_LAUNCHER.read_text(encoding="utf-8")

    assert "PRECHECK_ONLY=\"${PRECHECK_ONLY:-1}\"" in text
    assert "ALLOW_EVENT_SURPRISE_FULL_TRAIN" in text
    assert "EVENT_SURPRISE_FULL_TRAIN_GATE_JSON" in text
    assert "EVENT_SURPRISE_FULL_TRAIN_GATE_SHA256" in text
    assert "EVENT_SURPRISE_ENTRYPOINT_GATE_JSON" in text
    assert "EVENT_SURPRISE_ACTIVE_MANIFEST_SHA256" in text
    assert "EVENT_SURPRISE_RESOLVED_CONFIG_SHA256" in text
    assert "--active-manifest-sha256 \"$ACTIVE_MANIFEST_SHA256\"" in text
    assert "--resolved-config-sha256 \"$RESOLVED_CONFIG_SHA256\"" in text
    assert "--run-tag \"$RUN_TAG\"" in text
    assert "python -m torch.distributed.run" in text
    assert "tools/train.py \"$FULL_CONFIG\"" in text
    assert "tools/test.py" not in text
    for token in FORBIDDEN_C3_TOKENS:
        assert token not in text
