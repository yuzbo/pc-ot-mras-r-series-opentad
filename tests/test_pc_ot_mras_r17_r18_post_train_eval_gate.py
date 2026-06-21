import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
GUARD_PATH = ROOT / "opentad" / "utils" / "training_guard.py"
R17_CONFIG = ROOT / "configs" / "adatad" / "thumos" / "ctf_bdi_pc_ot_mras_r17_post_train_eval_candidate.py"
R18_CONFIG = ROOT / "configs" / "adatad" / "thumos" / "ctf_bdi_pc_ot_mras_r18_post_train_eval_candidate.py"
R17_LAUNCHER = ROOT / "scripts" / "run_ctf_bdi_pc_ot_mras_r17_post_train_eval_n16r4.sbatch"
R18_LAUNCHER = ROOT / "scripts" / "run_ctf_bdi_pc_ot_mras_r18_post_train_eval_n16r4.sbatch"


def _load_guard():
    spec = importlib.util.spec_from_file_location("training_guard_for_post_train_eval_test", GUARD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_gate(tmp_path, *, decision):
    gate_json = tmp_path / f"{decision}.json"
    gate_json.write_text(
        json.dumps(
            {
                "decision": decision,
                "active_sha256_manifest_sha256": "manifest-sha",
                "resolved_config_sha256": "resolved-sha",
                "checkpoint_sha256": "checkpoint-sha",
                "completed_training_evidence": True,
                "tools_train": False,
                "raw_prediction_cache": False,
                "paper_claim": False,
                "runtime_flops_claim": False,
                "deploy_claim": False,
                "dynamic_budget_claim": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return gate_json, hashlib.sha256(gate_json.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("config_path", "gate_name", "decision", "old_gate_name", "stage"),
    (
        (
            R17_CONFIG,
            "r17_pc_ot_mras_post_train_eval_gate",
            "ALLOW_R17_POST_TRAIN_EVAL",
            "r17_pc_ot_mras_formal_train_gate",
            "R17_post_train_eval_candidate",
        ),
        (
            R18_CONFIG,
            "r18_pc_ot_mras_post_train_eval_gate",
            "ALLOW_R18_POST_TRAIN_EVAL",
            "r18_pc_ot_mras_aux_diag_gate",
            "R18_post_train_eval_candidate",
        ),
    ),
)
def test_post_train_eval_configs_are_tools_test_only_and_gate_bound(
    tmp_path, monkeypatch, config_path, gate_name, decision, old_gate_name, stage
):
    mmengine_config = pytest.importorskip("mmengine.config")
    training_guard = _load_guard()
    cfg = mmengine_config.Config.fromfile(str(config_path))

    assert cfg.get(old_gate_name) is None
    gate = cfg.get(gate_name)
    assert gate.route == "CTF-BDI/PC-OT-MRAS"
    assert gate.stage == stage
    assert gate.post_train_eval_candidate is True
    assert gate.requires_completed_training_or_recorded_stop is True
    assert gate.allow_detector_training is True
    assert gate.requires_launch_gate is True
    assert gate.launch_gate_passed is True
    assert gate.allow_tools_train is False
    assert gate.allow_tools_test is True
    assert gate.allow_detector_map is True
    assert gate.allow_post_train_checkpoint_eval is True
    assert gate.detector_map_reporting_allowed is True
    assert gate.metric_claim_allowed is False
    assert gate.paper_claim_allowed is False
    assert tuple(gate.allowed_entrypoints) == ("tools/test.py",)
    assert gate.entrypoint_gate_context.allowed_decisions == (decision,)

    assert cfg.workflow.end_epoch == 0
    assert cfg.workflow.val_eval_interval == -1
    assert cfg.inference.load_from_raw_predictions is False
    assert cfg.inference.save_raw_prediction is False

    with pytest.raises(RuntimeError, match="allow_tools_train=False"):
        training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py")
    with pytest.raises(RuntimeError, match="missing required entrypoint gate env"):
        training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/test.py")

    gate_json, gate_sha = _write_gate(tmp_path, decision=decision)
    monkeypatch.setenv("OPENTAD_PCOTMRAS_ENTRYPOINT_GATE_JSON", str(gate_json))
    monkeypatch.setenv("OPENTAD_PCOTMRAS_ENTRYPOINT_GATE_SHA256", gate_sha)
    monkeypatch.setenv("OPENTAD_PCOTMRAS_ACTIVE_MANIFEST_SHA256", "manifest-sha")
    monkeypatch.setenv("OPENTAD_PCOTMRAS_RESOLVED_CONFIG_SHA256", "resolved-sha")
    assert training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/test.py") is None


def test_r18_post_train_eval_keeps_auxiliary_model_surface():
    mmengine_config = pytest.importorskip("mmengine.config")
    cfg = mmengine_config.Config.fromfile(str(R18_CONFIG))

    assert cfg.model.pc_ot_mras_reader_aux_loss.enabled is True
    assert cfg.model.pc_ot_mras_reader.type == "PCOTMRASReader"
    assert cfg.model.neck.type == "PCOTMRASDetectorBridge"
    assert cfg.model.rpn_head.type == "NativeIrregularAreaHeadP2"


@pytest.mark.parametrize(
    ("launcher", "prefix", "config_name", "decision", "checkpoint_var", "summary_pass"),
    (
        (
            R17_LAUNCHER,
            "R17",
            "ctf_bdi_pc_ot_mras_r17_post_train_eval_candidate.py",
            "ALLOW_R17_POST_TRAIN_EVAL",
            "R17_EVAL_CHECKPOINT",
            "R17_POST_TRAIN_EVAL_PASS_MAP_REPORTING_ONLY_NO_CLAIMS",
        ),
        (
            R18_LAUNCHER,
            "R18",
            "ctf_bdi_pc_ot_mras_r18_post_train_eval_candidate.py",
            "ALLOW_R18_POST_TRAIN_EVAL",
            "R18_EVAL_CHECKPOINT",
            "R18_POST_TRAIN_EVAL_PASS_MAP_REPORTING_ONLY_NO_CLAIMS",
        ),
    ),
)
def test_post_train_eval_launchers_are_fail_closed_and_checkpoint_bound(
    launcher, prefix, config_name, decision, checkpoint_var, summary_pass
):
    text = launcher.read_text(encoding="utf-8")

    assert "#SBATCH --gpus=1" in text
    assert "OpenTAD_PCOTMRAS_R16_R18_R20_Clean_20260619_1730" in text
    assert "refusing dirty historical OpenTAD_BATA_Clean path" in text
    assert config_name in text
    assert 'PRECHECK_ONLY="${PRECHECK_ONLY:-1}"' in text
    assert f'ALLOW_{prefix}_POST_TRAIN_EVAL="${{ALLOW_{prefix}_POST_TRAIN_EVAL:-0}}"' in text
    assert f"{prefix}_POST_TRAIN_EVAL_GATE_JSON" in text
    assert f"{prefix}_POST_TRAIN_EVAL_GATE_SHA256" in text
    assert checkpoint_var in text
    assert f"{checkpoint_var}_SHA256" in text
    assert "checkpoint SHA256 mismatch" in text
    assert "completed_training_evidence" in text
    assert "recorded_stop_continue_decision" in text
    assert decision in text
    assert "raw prediction/cache" in text
    assert "runtime/FLOPs/deploy claims are not approved" in text
    assert "tools/train.py is not part of this" in text
    assert 'tools/train.py "$CONFIG"' not in text
    assert 'tools/test.py "$CONFIG" --checkpoint' in text
    assert "result_detection.json" in text
    assert summary_pass in text
