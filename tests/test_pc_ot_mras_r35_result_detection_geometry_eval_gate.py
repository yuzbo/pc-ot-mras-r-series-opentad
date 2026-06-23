import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
GUARD_PATH = ROOT / "opentad" / "utils" / "training_guard.py"
CONFIG = ROOT / "configs" / "adatad" / "thumos" / "ctf_bdi_pc_ot_mras_r35_result_detection_geometry_eval_candidate.py"
LAUNCHER = ROOT / "scripts" / "run_ctf_bdi_pc_ot_mras_r35_result_detection_geometry_eval_n16r4.sbatch"


def _load_guard():
    spec = importlib.util.spec_from_file_location("training_guard_for_r35_geometry_eval_test", GUARD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_gate(tmp_path, **overrides):
    gate_json = tmp_path / "allow_r35_geometry_eval.json"
    payload = {
        "decision": "ALLOW_R35_RESULT_DETECTION_GEOMETRY_EVAL",
        "active_sha256_manifest_sha256": "manifest-sha",
        "resolved_config_sha256": "resolved-sha",
        "checkpoint_sha256": "checkpoint-sha",
        "completed_training_evidence": True,
        "tools_train": False,
        "direct_tools_train": False,
        "raw_prediction_cache": False,
        "prediction_cache": False,
        "load_from_raw_predictions": False,
        "save_raw_prediction": False,
        "metric_claim": False,
        "paper_claim": False,
        "runtime_flops_claim": False,
        "deploy_claim": False,
        "dynamic_budget_claim": False,
        "recovery_claim": False,
    }
    payload.update(overrides)
    gate_json.write_text(
        json.dumps(payload, sort_keys=True),
        encoding="utf-8",
    )
    return gate_json, hashlib.sha256(gate_json.read_bytes()).hexdigest()


def test_r35_result_detection_geometry_eval_config_is_tools_test_only_and_save_dict_bound(tmp_path, monkeypatch):
    mmengine_config = pytest.importorskip("mmengine.config")
    training_guard = _load_guard()
    cfg = mmengine_config.Config.fromfile(str(CONFIG))

    assert cfg.r35_pc_ot_mras_actionformer_head_gate is None
    gate = cfg.r35_pc_ot_mras_result_detection_geometry_eval_gate
    assert gate.route == "CTF-BDI/PC-OT-MRAS"
    assert gate.stage == "R35_result_detection_geometry_eval_candidate"
    assert gate.post_train_eval_candidate is True
    assert gate.result_detection_geometry_eval_candidate is True
    assert gate.requires_completed_training_or_recorded_stop is True
    assert gate.allow_tools_train is False
    assert gate.allow_tools_test is True
    assert gate.allow_detector_map is True
    assert gate.allow_result_detection_save is True
    assert gate.allow_r35_geometry_audit is True
    assert gate.metric_claim_allowed is False
    assert gate.paper_claim_allowed is False
    assert gate.recovery_claim_allowed is False
    assert tuple(gate.allowed_entrypoints) == ("tools/test.py",)
    assert tuple(gate.entrypoint_gate_context.allowed_decisions) == ("ALLOW_R35_RESULT_DETECTION_GEOMETRY_EVAL",)
    for forbidden_key in (
        "raw_prediction_cache",
        "prediction_cache",
        "load_from_raw_predictions",
        "save_raw_prediction",
        "metric_claim",
        "paper_claim",
        "runtime_flops_claim",
        "deploy_claim",
        "dynamic_budget_claim",
        "recovery_claim",
    ):
        assert forbidden_key in gate.entrypoint_gate_context.forbidden_true_keys

    assert cfg.model.rpn_head.type == "ActionFormerHead"
    assert cfg.inference.load_from_raw_predictions is False
    assert cfg.inference.save_raw_prediction is False
    assert cfg.post_processing.save_dict is True
    assert cfg.workflow.end_epoch == 0
    assert cfg.workflow.val_eval_interval == -1

    with pytest.raises(RuntimeError, match="allow_tools_train=False"):
        training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py")
    with pytest.raises(RuntimeError, match="missing required entrypoint gate env"):
        training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/test.py")

    gate_json, gate_sha = _write_gate(tmp_path)
    monkeypatch.setenv("OPENTAD_PCOTMRAS_ENTRYPOINT_GATE_JSON", str(gate_json))
    monkeypatch.setenv("OPENTAD_PCOTMRAS_ENTRYPOINT_GATE_SHA256", gate_sha)
    monkeypatch.setenv("OPENTAD_PCOTMRAS_ACTIVE_MANIFEST_SHA256", "manifest-sha")
    monkeypatch.setenv("OPENTAD_PCOTMRAS_RESOLVED_CONFIG_SHA256", "resolved-sha")
    assert training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/test.py") is None


def test_r35_result_detection_geometry_eval_gate_rejects_metric_claim(tmp_path, monkeypatch):
    mmengine_config = pytest.importorskip("mmengine.config")
    training_guard = _load_guard()
    cfg = mmengine_config.Config.fromfile(str(CONFIG))

    gate_json, gate_sha = _write_gate(tmp_path, metric_claim=True)
    monkeypatch.setenv("OPENTAD_PCOTMRAS_ENTRYPOINT_GATE_JSON", str(gate_json))
    monkeypatch.setenv("OPENTAD_PCOTMRAS_ENTRYPOINT_GATE_SHA256", gate_sha)
    monkeypatch.setenv("OPENTAD_PCOTMRAS_ACTIVE_MANIFEST_SHA256", "manifest-sha")
    monkeypatch.setenv("OPENTAD_PCOTMRAS_RESOLVED_CONFIG_SHA256", "resolved-sha")

    with pytest.raises(RuntimeError, match="metric_claim=true"):
        training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/test.py")


def test_r35_result_detection_geometry_eval_launcher_is_fail_closed_and_runs_geometry_audit():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "#SBATCH --gpus=1" in text
    assert "ctf_bdi_pc_ot_mras_r35_result_detection_geometry_eval_candidate.py" in text
    assert "ALLOW_R35_RESULT_DETECTION_GEOMETRY_EVAL" in text
    assert "R35_EVAL_CHECKPOINT" in text
    assert "R35_EVAL_CHECKPOINT_SHA256" in text
    assert "R35_RESULT_DETECTION_GEOMETRY_GATE_JSON" in text
    assert "R35_RESULT_DETECTION_GEOMETRY_GATE_SHA256" in text
    assert 'RESULT_DETECTION_JSON="$FINAL_WORK_DIR/result_detection.json"' in text
    assert 'test -f "$RESULT_DETECTION_JSON"' in text
    assert "analyze_actionformer_result_detection_geometry.py" in text
    assert "ACTIONFORMER_RESULT_DETECTION_GEOMETRY_AUDIT_READY" in text
    assert "post_processing.save_dict=True" in text
    assert "result_detection_saved_for_diagnostic" in text
    assert "recovery_claim_allowed" in text
    assert "metric_claim_allowed" in text
    assert "git status --porcelain --untracked-files=no" in text
    assert "arbitrary cfg-options are forbidden" in text
    assert "OPENTAD_PCOTMRAS_ENTRYPOINT_GATE_JSON" in text
    assert "OPENTAD_PCOTMRAS_RESOLVED_CONFIG_SHA256" in text
    assert "tools/bata/analyze_actionformer_result_detection_geometry.py" in text
    assert "printf '%s  resolved_config.py\\n' \"$RESOLVED_CONFIG_SHA256\"" in text

    assert re.search(r"(^|\s)(torchrun|srun|python(?:\s+-m)?)\b[^\n]*tools/train\.py", text) is None
    assert "--cfg-options" in text
    cfg_options_block = text.split("--cfg-options", 1)[1].split("2>&1", 1)[0]
    assert "post_processing.save_dict" not in cfg_options_block
    manifest_block = text.split('printf \'%s  resolved_config.py\\n\' "$RESOLVED_CONFIG_SHA256"', 1)[1].split(
        'ACTIVE_MANIFEST_SHA256=', 1
    )[0]
    assert '"$RESOLVED_CONFIG_DUMP"' not in manifest_block

    gate_json_loop = text.split('if not (gate.get("completed_training_evidence")', 1)[1].split(
        'print("ctf_bdi_pc_ot_mras_r35_result_detection_geometry_eval_gate=PASS")', 1
    )[0]
    for forbidden_key in (
        "direct_tools_train",
        "raw_prediction_cache",
        "prediction_cache",
        "load_from_raw_predictions",
        "save_raw_prediction",
        "metric_claim",
        "paper_claim",
        "runtime_flops_claim",
        "deploy_claim",
        "dynamic_budget_claim",
        "recovery_claim",
    ):
        assert f'"{forbidden_key}"' in gate_json_loop
