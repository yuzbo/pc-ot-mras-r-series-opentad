from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
LOCAL_CONFIG = ROOT / "configs" / "adatad" / "thumos" / "boundary_microscope_acquisition_local_precheck.py"
FULL_CONFIG = (
    ROOT
    / "configs"
    / "adatad"
    / "thumos"
    / "boundary_microscope_acquisition_full_train_candidate_n16r4.py"
)
VALIDATOR = ROOT / "tools" / "bata" / "validate_boundary_microscope_gate.py"


def _load_validator():
    spec = importlib.util.spec_from_file_location("boundary_microscope_gate_for_test", VALIDATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_config_or_skip(path: Path):
    mmengine_config = pytest.importorskip("mmengine.config")
    return mmengine_config.Config.fromfile(str(path))


def test_boundary_microscope_local_config_is_parseable_and_fail_closed():
    cfg = _load_config_or_skip(LOCAL_CONFIG)

    assert cfg.route_label == "DIVERGENT_INNOVATION_BOUNDARY_MICROSCOPE_DO_NOT_MERGE_WITH_C3"
    assert cfg.route_id == "boundary_microscope_acquisition"
    assert cfg.model.frame_selector.type == "BoundaryMicroscopeAcquisitionRoute"
    assert cfg.model.frame_selector.meta_key == "boundary_microscope_acquisition_plan"
    assert cfg.boundary_microscope_gate.allow_precheck_only is True
    assert cfg.boundary_microscope_gate.allow_tools_train is False
    assert cfg.boundary_microscope_gate.allow_tools_test is False
    assert cfg.boundary_microscope_gate.allow_remote_sync is False
    assert cfg.boundary_microscope_gate.allow_slurm is False
    assert cfg.boundary_microscope_gate.allow_gpu is False
    assert cfg.boundary_microscope_gate.allow_full_train is False
    assert cfg.boundary_microscope_gate.allow_raw_prediction is False
    assert cfg.boundary_microscope_gate.metric_claim_allowed is False
    assert cfg.boundary_microscope_gate.paper_claim_allowed is False
    assert cfg.inference.load_from_raw_predictions is False
    assert cfg.inference.save_raw_prediction is False


def test_boundary_microscope_full_train_candidate_config_is_still_fail_closed_until_gate():
    cfg = _load_config_or_skip(FULL_CONFIG)

    gate = cfg.boundary_microscope_gate
    assert gate.stage == "full_train_candidate_n16r4"
    assert gate.requires_gate_json is True
    assert gate.allow_precheck_only is True
    assert gate.allow_tools_train is False
    assert gate.allow_tools_test is False
    assert gate.allow_remote_sync is False
    assert gate.allow_slurm is False
    assert gate.allow_gpu is False
    assert gate.allow_full_train is False
    assert gate.allow_raw_prediction is False
    assert gate.metric_claim_allowed is False
    assert gate.paper_claim_allowed is False
    assert tuple(gate.allowed_entrypoints) == ()
    assert cfg.workflow.end_epoch == 60
    assert cfg.workflow.val_start_epoch == 40
    assert cfg.workflow.val_eval_interval == 2
    assert cfg.model.frame_selector.max_dense_gap <= 8
    assert cfg.model.frame_selector.target_len == 384


def test_boundary_microscope_gate_validator_rejects_open_train_or_claims(tmp_path):
    validator = _load_validator()
    payload = {
        "decision": "ALLOW_BOUNDARY_MICROSCOPE_PRECHECK_ONLY",
        "route": "boundary_microscope_acquisition",
        "route_label": "DIVERGENT_INNOVATION_BOUNDARY_MICROSCOPE_DO_NOT_MERGE_WITH_C3",
        "active_sha256_manifest_sha256": "manifest-sha",
        "resolved_config_sha256": "resolved-sha",
        "budget": 384,
        "dense_window_size": 768,
        "allow_precheck_only": True,
        "allow_tools_train": False,
        "allow_tools_test": False,
        "allow_remote_sync": False,
        "allow_slurm": False,
        "allow_gpu": False,
        "allow_full_train": False,
        "allow_raw_prediction": False,
        "load_from_raw_predictions": False,
        "save_raw_prediction": False,
        "test_time_gt_allowed": False,
        "teacher_allowed": False,
        "raw_prediction_cache_allowed": False,
        "uses_gt_at_test": False,
        "uses_teacher": False,
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
    )

    for key in ("allow_tools_train", "allow_slurm", "allow_full_train", "metric_claim_allowed"):
        bad = dict(payload)
        bad[key] = True
        with pytest.raises(ValueError, match=key):
            validator.validate_gate_payload(
                bad,
                active_manifest_sha256="manifest-sha",
                resolved_config_sha256="resolved-sha",
                budget=384,
                dense_window_size=768,
            )

    gate_json = tmp_path / "boundary_gate.json"
    gate_json.write_text(json.dumps(payload), encoding="utf-8-sig")
    gate_sha = hashlib.sha256(gate_json.read_bytes()).hexdigest()
    loaded = validator.validate_gate_file(
        gate_json=gate_json,
        gate_sha256=gate_sha,
        active_manifest_sha256="manifest-sha",
        resolved_config_sha256="resolved-sha",
        budget=384,
        dense_window_size=768,
    )
    assert loaded["decision"] == "ALLOW_BOUNDARY_MICROSCOPE_PRECHECK_ONLY"


def test_boundary_microscope_validator_cli_passes_for_precheck_payload(tmp_path):
    payload = {
        "decision": "ALLOW_BOUNDARY_MICROSCOPE_PRECHECK_ONLY",
        "route": "boundary_microscope_acquisition",
        "route_label": "DIVERGENT_INNOVATION_BOUNDARY_MICROSCOPE_DO_NOT_MERGE_WITH_C3",
        "active_sha256_manifest_sha256": "manifest-sha",
        "resolved_config_sha256": "resolved-sha",
        "budget": 384,
        "dense_window_size": 768,
        "allow_precheck_only": True,
        "allow_tools_train": False,
        "allow_tools_test": False,
        "allow_remote_sync": False,
        "allow_slurm": False,
        "allow_gpu": False,
        "allow_full_train": False,
        "allow_raw_prediction": False,
        "load_from_raw_predictions": False,
        "save_raw_prediction": False,
        "test_time_gt_allowed": False,
        "teacher_allowed": False,
        "raw_prediction_cache_allowed": False,
        "uses_gt_at_test": False,
        "uses_teacher": False,
        "uses_raw_prediction_cache": False,
        "metric_claim_allowed": False,
        "paper_claim_allowed": False,
    }
    gate_json = tmp_path / "boundary_gate.json"
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
        ],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "BOUNDARY_MICROSCOPE_GATE_VALIDATION_PASS" in result.stdout


def test_boundary_microscope_configs_do_not_reference_old_c3_selector_tokens():
    forbidden = (
        "pc_ot_mras_prebackbone_frame_selector",
        "PCOTMRASPreBackboneFrameSelector",
        "pc_ot_mras_prebackbone_c3",
        "C3-Pro",
        "C3_RS",
    )
    for path in (LOCAL_CONFIG, FULL_CONFIG):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text


def test_boundary_microscope_selector_is_exported_for_registry_discovery():
    selectors_init = ROOT / "opentad" / "models" / "selectors" / "__init__.py"
    text = selectors_init.read_text(encoding="utf-8")

    assert "boundary_microscope_acquisition_route" in text
    assert "BoundaryMicroscopeAcquisitionRoute" in text
    assert "BOUNDARY_MICROSCOPE_ROUTE_LABEL" in text
