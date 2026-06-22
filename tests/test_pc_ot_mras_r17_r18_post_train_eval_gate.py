import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
GUARD_PATH = ROOT / "opentad" / "utils" / "training_guard.py"
R17_CONFIG = ROOT / "configs" / "adatad" / "thumos" / "ctf_bdi_pc_ot_mras_r17_post_train_eval_candidate.py"
R18_CONFIG = ROOT / "configs" / "adatad" / "thumos" / "ctf_bdi_pc_ot_mras_r18_post_train_eval_candidate.py"
R17_READER_DISABLED_CONFIG = (
    ROOT / "configs" / "adatad" / "thumos" / "ctf_bdi_pc_ot_mras_r17_reader_disabled_eval_candidate.py"
)
R18_READER_DISABLED_CONFIG = (
    ROOT / "configs" / "adatad" / "thumos" / "ctf_bdi_pc_ot_mras_r18_reader_disabled_eval_candidate.py"
)
R17_LAUNCHER = ROOT / "scripts" / "run_ctf_bdi_pc_ot_mras_r17_post_train_eval_n16r4.sbatch"
R18_LAUNCHER = ROOT / "scripts" / "run_ctf_bdi_pc_ot_mras_r18_post_train_eval_n16r4.sbatch"
READER_DISABLED_LAUNCHER = (
    ROOT / "scripts" / "run_ctf_bdi_pc_ot_mras_reader_disabled_eval_n16r4.sbatch"
)
REQUIRED_EVAL_MANIFEST_PATHS = (
    "tools/test.py",
    "opentad/cores/test_engine.py",
    "opentad/utils/training_guard.py",
    "opentad/models/detectors/actionformer.py",
    "opentad/models/selectors/pc_ot_mras_reader.py",
    "opentad/models/necks/pc_ot_mras_detector_bridge.py",
    "opentad/models/dense_heads/native_irregular_area_head_p2.py",
    "opentad/models/utils/pc_ot_mras_raw_prediction_guard.py",
    "tests/test_pc_ot_mras_r17_r18_post_train_eval_gate.py",
)


def _assert_uses_base_work_dir_and_optional_result_json(text):
    assert 'WORK_DIR_BASE="$RUN_ROOT/eval_workdir"' in text
    assert 'WORK_DIR="$WORK_DIR_BASE"' in text
    assert 'FINAL_WORK_DIR="$WORK_DIR/gpu1_id${EVAL_ID}"' in text
    assert 'final_work_dir=$FINAL_WORK_DIR' in text
    assert 'work_dir="$WORK_DIR"' in text
    assert 'work_dir="$FINAL_WORK_DIR"' not in text
    assert 'WORK_DIR="$WORK_DIR_BASE/gpu1_id${EVAL_ID}"' not in text
    assert 'RESULT_DETECTION_JSON="$FINAL_WORK_DIR/result_detection.json"' in text
    assert 'if [ -f "$RESULT_DETECTION_JSON" ]; then' in text
    assert 'test -f "$RESULT_DETECTION_JSON"' not in text
    assert 'grep -q "Testing Over" "$EVAL_STDOUT"' in text
    assert 'grep -Eiq "Average-mAP|mAP at tIoU|average_mAP|mAP@" "$EVAL_STDOUT"' in text
    assert "result_detection_json=ABSENT_EXPECTED_WHEN_post_processing.save_dict_FALSE" in text


def _assert_manifest_covers_execution_surface(text, launcher):
    assert f"scripts/{launcher.name}" in text
    for path in REQUIRED_EVAL_MANIFEST_PATHS:
        assert path in text


def _assert_no_tools_train_launcher_invocation(text):
    assert re.search(r"(^|\s)(torchrun|srun|python(?:\s+-m)?)\b[^\n]*tools/train\.py", text) is None


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


def test_entrypoint_gate_requires_resolved_config_sha_when_context_requires_it(tmp_path, monkeypatch):
    mmengine_config = pytest.importorskip("mmengine.config")
    training_guard = _load_guard()
    cfg = mmengine_config.Config.fromfile(str(R17_CONFIG))
    gate_json = tmp_path / "missing_resolved.json"
    gate_json.write_text(
        json.dumps(
            {
                "decision": "ALLOW_R17_POST_TRAIN_EVAL",
                "active_sha256_manifest_sha256": "manifest-sha",
                "checkpoint_sha256": "checkpoint-sha",
                "completed_training_evidence": True,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENTAD_PCOTMRAS_ENTRYPOINT_GATE_JSON", str(gate_json))
    monkeypatch.setenv("OPENTAD_PCOTMRAS_ENTRYPOINT_GATE_SHA256", hashlib.sha256(gate_json.read_bytes()).hexdigest())
    monkeypatch.setenv("OPENTAD_PCOTMRAS_ACTIVE_MANIFEST_SHA256", "manifest-sha")
    monkeypatch.setenv("OPENTAD_PCOTMRAS_RESOLVED_CONFIG_SHA256", "resolved-sha")

    with pytest.raises(RuntimeError, match="missing resolved_config_sha256"):
        training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/test.py")


def test_r18_post_train_eval_keeps_auxiliary_model_surface():
    mmengine_config = pytest.importorskip("mmengine.config")
    cfg = mmengine_config.Config.fromfile(str(R18_CONFIG))

    assert cfg.model.pc_ot_mras_reader_aux_loss.enabled is True
    assert cfg.model.pc_ot_mras_reader.type == "PCOTMRASReader"
    assert cfg.model.neck.type == "PCOTMRASDetectorBridge"
    assert cfg.model.rpn_head.type == "NativeIrregularAreaHeadP2"


@pytest.mark.parametrize(
    ("config_path", "gate_name", "decision", "old_gate_name", "stage"),
    (
        (
            R17_READER_DISABLED_CONFIG,
            "r17_pc_ot_mras_reader_disabled_eval_gate",
            "ALLOW_R17_READER_DISABLED_EVAL",
            "r17_pc_ot_mras_post_train_eval_gate",
            "R17_reader_disabled_eval_candidate",
        ),
        (
            R18_READER_DISABLED_CONFIG,
            "r18_pc_ot_mras_reader_disabled_eval_gate",
            "ALLOW_R18_READER_DISABLED_EVAL",
            "r18_pc_ot_mras_post_train_eval_gate",
            "R18_reader_disabled_eval_candidate",
        ),
    ),
)
def test_reader_disabled_eval_configs_are_tools_test_only_and_override_bound(
    tmp_path, monkeypatch, config_path, gate_name, decision, old_gate_name, stage
):
    mmengine_config = pytest.importorskip("mmengine.config")
    training_guard = _load_guard()
    cfg = mmengine_config.Config.fromfile(str(config_path))

    assert cfg.get(old_gate_name) is None
    gate = cfg.get(gate_name)
    assert gate.route == "CTF-BDI/PC-OT-MRAS"
    assert gate.stage == stage
    assert gate.reader_disabled_eval_candidate is True
    assert gate.allow_tools_train is False
    assert gate.allow_tools_test is True
    assert gate.allow_detector_map is True
    assert gate.metric_claim_allowed is False
    assert gate.paper_claim_allowed is False
    assert tuple(gate.entrypoint_gate_context.allowed_decisions) == (decision,)
    assert "reader_disabled_exact_uniform_eval_override" in gate.allowed_checks

    override = cfg.model.pc_ot_mras_reader_eval_override
    assert override.enabled is True
    assert override.mode == "exact_uniform"
    assert override.num_slots == 384
    assert cfg.model.pc_ot_mras_reader.type == "PCOTMRASReader"
    assert cfg.model.neck.type == "PCOTMRASDetectorBridge"

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
    _assert_no_tools_train_launcher_invocation(text)
    _assert_uses_base_work_dir_and_optional_result_json(text)
    _assert_manifest_covers_execution_surface(text, launcher)
    assert summary_pass in text


def test_reader_disabled_eval_launcher_is_fail_closed_checkpoint_bound_and_targeted():
    text = READER_DISABLED_LAUNCHER.read_text(encoding="utf-8")

    assert "#SBATCH --gpus=1" in text
    assert "OpenTAD_PCOTMRAS_R16_R18_R20_Clean_20260619_1730" in text
    assert "refusing dirty historical OpenTAD_BATA_Clean path" in text
    assert 'TARGET_RAW="${PCOTMRAS_READER_DISABLED_TARGET:-${TARGET:-R17}}"' in text
    assert "PCOTMRAS_READER_DISABLED_TARGET must be R17 or R18" in text

    for prefix, config_name, decision in (
        (
            "R17",
            "ctf_bdi_pc_ot_mras_r17_reader_disabled_eval_candidate.py",
            "ALLOW_R17_READER_DISABLED_EVAL",
        ),
        (
            "R18",
            "ctf_bdi_pc_ot_mras_r18_reader_disabled_eval_candidate.py",
            "ALLOW_R18_READER_DISABLED_EVAL",
        ),
    ):
        assert config_name in text
        assert decision in text
        assert f'ALLOW_READER_DISABLED_EVAL="${{ALLOW_{prefix}_READER_DISABLED_EVAL:-0}}"' in text
        assert f"{prefix}_READER_DISABLED_EVAL_GATE_JSON" in text
        assert f"{prefix}_READER_DISABLED_EVAL_GATE_SHA256" in text
        assert f"{prefix}_EVAL_CHECKPOINT" in text
        assert f"{prefix}_EVAL_CHECKPOINT_SHA256" in text

    assert 'PRECHECK_ONLY="${PRECHECK_ONLY:-1}"' in text
    assert "${TARGET_UPPER}_READER_DISABLED_EVAL_PASS_MAP_REPORTING_ONLY_NO_CLAIMS" in text
    assert "PRECHECK_ONLY=0 requires ALLOW_${TARGET_UPPER}_READER_DISABLED_EVAL=1 after review" in text
    assert "reader-disabled eval gate SHA256 mismatch" in text
    assert "checkpoint SHA256 mismatch" in text
    assert "completed_training_evidence" in text
    assert "recorded_stop_continue_decision" in text
    assert "reader_disabled_eval=true" in text
    assert "same_head_control=true" in text
    assert "exact_uniform_reader_override=true" in text
    assert "reader_disabled_exact_uniform_eval_override" in text
    assert "raw prediction/cache" in text
    assert "runtime/FLOPs/deploy claims are not approved" in text
    assert "tools/train.py is not part of this reader-disabled eval launcher" in text
    assert 'tools/train.py "$CONFIG"' not in text
    assert 'tools/test.py "$CONFIG" --checkpoint' in text
    _assert_no_tools_train_launcher_invocation(text)
    _assert_uses_base_work_dir_and_optional_result_json(text)
    _assert_manifest_covers_execution_surface(text, READER_DISABLED_LAUNCHER)
    assert "OPENTAD_PCOTMRAS_ENTRYPOINT_GATE_JSON" in text
    assert "OPENTAD_PCOTMRAS_ACTIVE_MANIFEST_SHA256" in text
