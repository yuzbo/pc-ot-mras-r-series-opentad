from __future__ import annotations

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
    assert gate.launch_gate_passed is False
    assert gate.allow_remote_sync is False
    assert gate.allow_slurm is False
    assert gate.allow_tools_train is False
    assert gate.allow_tools_test is False
    assert gate.allow_full_train is False
    assert gate.allow_raw_prediction is False
    assert gate.metric_claim_allowed is False
    assert gate.paper_claim_allowed is False
    assert tuple(gate.allowed_entrypoints) == ()

    forbidden = set(context.forbidden_true_keys)
    for key in (
        "remote_sync",
        "allow_remote_sync",
        "slurm",
        "allow_slurm",
        "tools_train",
        "allow_tools_train",
        "tools_test",
        "allow_tools_test",
        "full_train",
        "allow_full_train",
        "raw_prediction",
        "allow_raw_prediction",
        "metric_claim",
        "metric_claim_allowed",
        "paper_claim",
        "paper_claim_allowed",
    ):
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
        assert payload["allowed_entrypoints"] == []
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
