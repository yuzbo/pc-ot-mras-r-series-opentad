import importlib.util
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "adatad" / "thumos" / "ctf_bdi_pc_ot_mras_r18_aux_formal_train_candidate.py"
LAUNCHER = ROOT / "scripts" / "run_ctf_bdi_pc_ot_mras_r18_aux_formal_train_n16r4.sbatch"
GUARD_PATH = ROOT / "opentad" / "utils" / "training_guard.py"


def _load_guard():
    spec = importlib.util.spec_from_file_location("training_guard_for_r18_formal_test", GUARD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_r18_aux_formal_config_is_parseable_trainable_and_aux_on(tmp_path, monkeypatch):
    mmengine_config = pytest.importorskip("mmengine.config")
    training_guard = _load_guard()

    cfg = mmengine_config.Config.fromfile(str(CONFIG))

    assert cfg.r17_pc_ot_mras_formal_train_gate is None
    gate = cfg.r18_pc_ot_mras_aux_diag_gate
    assert gate.route == "CTF-BDI/PC-OT-MRAS"
    assert gate.stage == "R18_aux_on_formal_train_candidate"
    assert gate.train_only_auxiliary_diagnostic is True
    assert gate.allow_detector_training is True
    assert gate.requires_launch_gate is True
    assert gate.launch_gate_passed is True
    assert gate.allow_tools_train is True
    assert gate.allow_tools_test is False
    assert gate.allow_detector_map is False
    assert gate.allow_train_validation_map is False
    assert gate.entrypoint_gate_context.required is True
    assert gate.metric_claim_allowed is False
    assert gate.paper_claim_allowed is False
    assert tuple(gate.allowed_entrypoints) == ("tools/train.py",)

    assert cfg.workflow.end_epoch == 60
    assert cfg.workflow.val_start_epoch == 999
    assert cfg.workflow.val_eval_interval == -1
    assert cfg.model.pc_ot_mras_reader_aux_loss.enabled is True
    assert cfg.model.pc_ot_mras_reader.type == "PCOTMRASReader"
    assert cfg.model.neck.type == "PCOTMRASDetectorBridge"
    assert cfg.model.rpn_head.type == "NativeIrregularAreaHeadP2"

    with pytest.raises(RuntimeError, match="missing required entrypoint gate env"):
        training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py")

    gate_json = tmp_path / "r18_gate.json"
    gate_json.write_text(
        json.dumps(
            {
                "decision": "ALLOW_R18_AUX_FORMAL_TRAIN",
                "active_sha256_manifest_sha256": "manifest-sha",
                "resolved_config_sha256": "resolved-sha",
                "tools_test": False,
                "detector_map": False,
                "metric_claim": False,
                "paper_claim": False,
                "runtime_flops_claim": False,
                "deploy_claim": False,
            }
        ),
        encoding="utf-8",
    )
    gate_sha = hashlib.sha256(gate_json.read_bytes()).hexdigest()
    monkeypatch.setenv("OPENTAD_PCOTMRAS_ENTRYPOINT_GATE_JSON", str(gate_json))
    monkeypatch.setenv("OPENTAD_PCOTMRAS_ENTRYPOINT_GATE_SHA256", gate_sha)
    monkeypatch.setenv("OPENTAD_PCOTMRAS_ACTIVE_MANIFEST_SHA256", "manifest-sha")
    monkeypatch.setenv("OPENTAD_PCOTMRAS_RESOLVED_CONFIG_SHA256", "resolved-sha")

    assert training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py") is None
    with pytest.raises(RuntimeError, match="allow_tools_test=False"):
        training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/test.py")


def test_r18_aux_formal_config_has_no_test_time_shortcuts():
    mmengine_config = pytest.importorskip("mmengine.config")
    cfg = mmengine_config.Config.fromfile(str(CONFIG))

    assert cfg.inference.load_from_raw_predictions is False
    assert cfg.inference.save_raw_prediction is False
    for split in ("train", "val", "test"):
        pipeline_text = repr(cfg.dataset[split].pipeline).lower()
        assert "raw_prediction" not in pipeline_text
        assert "prediction_cache" not in pipeline_text
        assert "teacher" not in pipeline_text
        assert "oracle" not in pipeline_text


def test_r18_aux_formal_launcher_is_fail_closed_by_default():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "#SBATCH --gpus=1" in text
    assert "#SBATCH -J pcot_r18aux" in text
    assert "OpenTAD_PCOTMRAS_R16_R18_R20_Clean_20260619_1730" in text
    assert "refusing dirty historical OpenTAD_BATA_Clean path" in text
    assert "expected branch $EXPECTED_GIT_BRANCH" in text
    assert "tracked clean repo files are modified" in text
    assert "ctf_bdi_pc_ot_mras_r18_aux_formal_train_candidate.py" in text
    assert 'PRECHECK_ONLY="${PRECHECK_ONLY:-1}"' in text
    assert 'ALLOW_R18_AUX_FORMAL_TRAIN="${ALLOW_R18_AUX_FORMAL_TRAIN:-0}"' in text
    assert "PRECHECK_ONLY=0 requires ALLOW_R18_AUX_FORMAL_TRAIN=1" in text
    assert "PRECHECK_ONLY=1 must not set ALLOW_R18_AUX_FORMAL_TRAIN=1" in text
    assert "R18_AUX_FORMAL_GATE_JSON" in text
    assert "R18 execution gate active manifest sha256 mismatch" in text
    assert "R18 execution gate resolved config sha256 mismatch" in text
    assert "OPENTAD_PCOTMRAS_ENTRYPOINT_GATE_JSON" in text
    assert "OPENTAD_PCOTMRAS_RESOLVED_CONFIG_SHA256" in text
    assert "resolved_config_dependency_count" in text
    assert "RESOLVED_CONFIG_SHA256" in text
    assert "opentad/models/dense_heads/__init__.py" in text
    assert "opentad/models/dense_heads/native_irregular_area_head_p2.py" in text
    assert "tools/test.py" in text
    assert 'tools/test.py "$CONFIG"' not in text
    assert 'tools/train.py "$CONFIG"' in text
    assert "result_detection.json" in text
    assert "R18_AUX_FORMAL_PRECHECK_ONLY_PASS_NO_TRAIN" in text
