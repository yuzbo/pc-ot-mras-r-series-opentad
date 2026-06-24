from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
LOCAL_CONFIG = ROOT / "configs" / "adatad" / "thumos" / "frame_token_hybrid_acquisition_local_precheck.py"
FULL_CONFIG = ROOT / "configs" / "adatad" / "thumos" / "frame_token_hybrid_acquisition_full_train_candidate_n16r4.py"
VALIDATOR = ROOT / "tools" / "bata" / "validate_frame_token_hybrid_gate.py"
SELECTOR_INIT = ROOT / "opentad" / "models" / "selectors" / "__init__.py"
N16R4_PRECHECK_LAUNCHER = ROOT / "scripts" / "run_frame_token_hybrid_acquisition_precheck_n16r4.sbatch"


def _load_validator():
    spec = importlib.util.spec_from_file_location("frame_token_hybrid_gate_for_test", VALIDATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_config_or_skip(path: Path):
    mmengine_config = pytest.importorskip("mmengine.config")
    return mmengine_config.Config.fromfile(str(path))


def test_frame_token_hybrid_local_config_is_parseable_and_fail_closed():
    cfg = _load_config_or_skip(LOCAL_CONFIG)

    assert cfg.route_label == "DIVERGENT_INNOVATION_FRAME_TOKEN_HYBRID_DO_NOT_MERGE_WITH_C3"
    assert cfg.route_id == "frame_token_hybrid_acquisition"
    assert cfg.experiment_scope.actual_decode_saving_in_current_pipeline is False
    assert cfg.experiment_scope.raw_decode_saving_claim_allowed is False
    assert cfg.experiment_scope.pre_decode_loader_hook_reviewed is False
    assert cfg.experiment_scope.requires_deploy_preview_probe_signal is True
    assert cfg.model.frame_selector.type == "FrameTokenHybridAcquisitionRoute"
    assert cfg.model.frame_selector.meta_key == "frame_token_hybrid_acquisition_plan"
    assert cfg.model.frame_selector.target_dense_len == 768
    assert cfg.model.frame_selector.require_preview_signal is True
    assert cfg.model.frame_selector.preview_signal_meta_key == "frame_token_hybrid_preview_signal"
    assert cfg.frame_token_hybrid_gate.allow_precheck_only is True
    assert cfg.frame_token_hybrid_gate.allow_tools_train is False
    assert cfg.frame_token_hybrid_gate.allow_tools_test is False
    assert cfg.frame_token_hybrid_gate.allow_remote_sync is False
    assert cfg.frame_token_hybrid_gate.allow_slurm is False
    assert cfg.frame_token_hybrid_gate.allow_gpu is False
    assert cfg.frame_token_hybrid_gate.allow_full_train is False
    assert cfg.frame_token_hybrid_gate.metric_claim_allowed is False
    assert cfg.frame_token_hybrid_gate.paper_claim_allowed is False
    assert cfg.inference.load_from_raw_predictions is False
    assert cfg.inference.save_raw_prediction is False


def test_frame_token_hybrid_selector_is_exported_from_selector_package():
    selector_init = SELECTOR_INIT.read_text(encoding="utf-8")

    assert "frame_token_hybrid_acquisition_route" in selector_init
    assert "FrameTokenHybridAcquisitionRoute" in selector_init
    assert "FRAME_TOKEN_HYBRID_ROUTE_LABEL" in selector_init
    assert "FRAME_TOKEN_HYBRID_META_KEY" in selector_init


def test_frame_token_hybrid_full_train_candidate_config_is_still_fail_closed_until_gate():
    cfg = _load_config_or_skip(FULL_CONFIG)

    gate = cfg.frame_token_hybrid_gate
    assert gate.stage == "full_train_candidate_n16r4"
    assert gate.requires_gate_json is True
    assert gate.allowed_decision == "ALLOW_FRAME_TOKEN_HYBRID_PRECHECK_ONLY"
    assert gate.allow_precheck_only is True
    assert gate.allow_tools_train is False
    assert gate.allow_tools_test is False
    assert gate.allow_remote_sync is False
    assert gate.allow_slurm is False
    assert gate.allow_gpu is False
    assert gate.allow_full_train is False
    assert gate.metric_claim_allowed is False
    assert gate.paper_claim_allowed is False
    assert tuple(gate.allowed_entrypoints) == ()
    assert cfg.workflow.end_epoch == 60
    assert cfg.workflow.val_start_epoch == 40
    assert cfg.workflow.val_eval_interval == 2
    assert cfg.model.frame_selector.target_dense_len == 768
    assert cfg.model.frame_selector.require_preview_signal is True
    assert cfg.model.frame_selector.stable_gap_min_len >= 6


def test_frame_token_hybrid_gate_validator_rejects_open_train_remote_or_claims(tmp_path):
    validator = _load_validator()
    payload = {
        "decision": "ALLOW_FRAME_TOKEN_HYBRID_PRECHECK_ONLY",
        "route": "frame_token_hybrid_acquisition",
        "route_label": "DIVERGENT_INNOVATION_FRAME_TOKEN_HYBRID_DO_NOT_MERGE_WITH_C3",
        "active_sha256_manifest_sha256": "manifest-sha",
        "resolved_config_sha256": "resolved-sha",
        "budget": 384,
        "dense_window_size": 768,
        "target_dense_len": 768,
        "allow_precheck_only": True,
        "allow_tools_train": False,
        "allow_tools_test": False,
        "allow_remote_sync": False,
        "allow_slurm": False,
        "allow_gpu": False,
        "allow_full_train": False,
        "load_from_raw_predictions": False,
        "save_raw_prediction": False,
        "uses_gt_at_test": False,
        "uses_teacher": False,
        "uses_oracle": False,
        "uses_raw_prediction_cache": False,
        "metric_claim_allowed": False,
        "paper_claim_allowed": False,
    }

    assert validator.validate_gate_payload(
        payload,
        active_manifest_sha256="manifest-sha",
        resolved_config_sha256="resolved-sha",
        budget=384,
        dense_window_size=768,
        target_dense_len=768,
    )

    for key in (
        "allow_tools_train",
        "allow_remote_sync",
        "allow_slurm",
        "allow_gpu",
        "allow_full_train",
        "metric_claim_allowed",
    ):
        bad = dict(payload)
        bad[key] = True
        with pytest.raises(ValueError, match=key):
            validator.validate_gate_payload(
                bad,
                active_manifest_sha256="manifest-sha",
                resolved_config_sha256="resolved-sha",
                budget=384,
                dense_window_size=768,
                target_dense_len=768,
            )

    gate_json = tmp_path / "frame_token_hybrid_gate.json"
    gate_json.write_text(json.dumps(payload), encoding="utf-8")
    gate_sha = hashlib.sha256(gate_json.read_bytes()).hexdigest()
    loaded = validator.validate_gate_file(
        gate_json=gate_json,
        gate_sha256=gate_sha,
        active_manifest_sha256="manifest-sha",
        resolved_config_sha256="resolved-sha",
        budget=384,
        dense_window_size=768,
        target_dense_len=768,
    )
    assert loaded["decision"] == "ALLOW_FRAME_TOKEN_HYBRID_PRECHECK_ONLY"


def test_frame_token_hybrid_validator_cli_passes_for_precheck_payload(tmp_path):
    payload = {
        "decision": "ALLOW_FRAME_TOKEN_HYBRID_PRECHECK_ONLY",
        "route": "frame_token_hybrid_acquisition",
        "route_label": "DIVERGENT_INNOVATION_FRAME_TOKEN_HYBRID_DO_NOT_MERGE_WITH_C3",
        "active_sha256_manifest_sha256": "manifest-sha",
        "resolved_config_sha256": "resolved-sha",
        "budget": 384,
        "dense_window_size": 768,
        "target_dense_len": 768,
        "allow_precheck_only": True,
        "allow_tools_train": False,
        "allow_tools_test": False,
        "allow_remote_sync": False,
        "allow_slurm": False,
        "allow_gpu": False,
        "allow_full_train": False,
        "load_from_raw_predictions": False,
        "save_raw_prediction": False,
        "uses_gt_at_test": False,
        "uses_teacher": False,
        "uses_oracle": False,
        "uses_raw_prediction_cache": False,
        "metric_claim_allowed": False,
        "paper_claim_allowed": False,
    }
    gate_json = tmp_path / "frame_token_hybrid_gate.json"
    gate_json.write_text(json.dumps(payload), encoding="utf-8")
    gate_sha = hashlib.sha256(gate_json.read_bytes()).hexdigest()

    result = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--gate-json",
            str(gate_json),
            "--gate-sha256",
            gate_sha,
            "--active-manifest-sha256",
            "manifest-sha",
            "--resolved-config-sha256",
            "resolved-sha",
            "--budget",
            "384",
            "--dense-window-size",
            "768",
            "--target-dense-len",
            "768",
        ],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "FRAME_TOKEN_HYBRID_GATE_VALIDATION_PASS" in result.stdout


def test_frame_token_hybrid_validator_cli_json_config_precheck_is_fail_closed():
    result = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            str(FULL_CONFIG),
            "--json",
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
    assert payload["decision"] == "ALLOW_FRAME_TOKEN_HYBRID_PRECHECK_ONLY"
    assert payload["route_label"] == "DIVERGENT_INNOVATION_FRAME_TOKEN_HYBRID_DO_NOT_MERGE_WITH_C3"
    assert payload["allowed_entrypoints"] == []
    assert payload["forbidden_entrypoints"] == ["tools/train.py", "tools/test.py", "sbatch", "scp", "rsync"]
    assert payload["allow_tools_train"] is False
    assert payload["allow_tools_test"] is False
    assert payload["allow_remote_sync"] is False
    assert payload["allow_slurm"] is False
    assert payload["allow_gpu"] is False
    assert payload["allow_full_train"] is False
    assert payload["actual_decode_saving_in_current_pipeline"] is False
    assert payload["raw_decode_saving_claim_allowed"] is False
    assert payload["pre_decode_loader_hook_reviewed"] is False
    assert payload["requires_deploy_preview_probe_signal"] is True


def test_frame_token_hybrid_validator_accepts_windows_utf8_bom_gate_json(tmp_path):
    validator = _load_validator()
    payload = {
        "decision": "ALLOW_FRAME_TOKEN_HYBRID_PRECHECK_ONLY",
        "route": "frame_token_hybrid_acquisition",
        "route_label": "DIVERGENT_INNOVATION_FRAME_TOKEN_HYBRID_DO_NOT_MERGE_WITH_C3",
        "active_sha256_manifest_sha256": "manifest-sha",
        "resolved_config_sha256": "resolved-sha",
        "budget": 384,
        "dense_window_size": 768,
        "target_dense_len": 768,
        "allow_precheck_only": True,
        "allow_tools_train": False,
        "allow_tools_test": False,
        "allow_remote_sync": False,
        "allow_slurm": False,
        "allow_gpu": False,
        "allow_full_train": False,
        "load_from_raw_predictions": False,
        "save_raw_prediction": False,
        "uses_gt_at_test": False,
        "uses_teacher": False,
        "uses_oracle": False,
        "uses_raw_prediction_cache": False,
        "metric_claim_allowed": False,
        "paper_claim_allowed": False,
    }
    gate_json = tmp_path / "frame_token_hybrid_gate_bom.json"
    gate_json.write_text(json.dumps(payload), encoding="utf-8-sig")
    gate_sha = hashlib.sha256(gate_json.read_bytes()).hexdigest()

    loaded = validator.validate_gate_file(
        gate_json=gate_json,
        gate_sha256=gate_sha,
        active_manifest_sha256="manifest-sha",
        resolved_config_sha256="resolved-sha",
        budget=384,
        dense_window_size=768,
        target_dense_len=768,
    )

    assert loaded["route"] == "frame_token_hybrid_acquisition"


def test_frame_token_hybrid_configs_do_not_reference_old_c3_selector_tokens():
    forbidden = (
        "pc_ot_mras_prebackbone_frame_selector",
        "PCOTMRASPreBackboneFrameSelector",
        "pc_ot_mras_prebackbone_c3",
        "C3-Pro",
        "C3_RS",
    )
    for path in (LOCAL_CONFIG, FULL_CONFIG):
        text = path.read_text(encoding="utf-8")
        resolved_text = _load_config_or_skip(path).pretty_text
        for token in forbidden:
            assert token not in text
            assert token not in resolved_text


def test_frame_token_hybrid_n16r4_precheck_launcher_is_fail_closed():
    text = N16R4_PRECHECK_LAUNCHER.read_text(encoding="utf-8")

    assert "DIVERGENT_INNOVATION_FRAME_TOKEN_HYBRID_DO_NOT_MERGE_WITH_C3" in text
    assert 'PRECHECK_ONLY="${PRECHECK_ONLY:-1}"' in text
    assert "ALLOW_FRAME_TOKEN_HYBRID_FULL_TRAIN" in text
    assert "validate_frame_token_hybrid_gate.py" in text
    assert "FRAME_TOKEN_HYBRID_PRECHECK_ONLY_PASS_NO_TRAIN" in text
    assert "full train gate is still locked" in text
    assert "torchrun" not in text
    assert "python tools/train.py" not in text
    assert "tools/test.py" not in text
