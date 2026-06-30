import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from tools.mdl_knot.validate_mdl_knot_shortdiag import _load_config_with_base, _validate_shortdiag_config


ROOT = Path(__file__).resolve().parents[1]
ROUTE_LABEL = "DIVERGENT_INNOVATION_MDL_KNOT_DO_NOT_MERGE_WITH_C3"
CONFIG_PATH = ROOT / "configs" / "adatad" / "thumos" / "input_mdl_knot_dynamic_adapter_irregular_headv3_shortdiag.py"
VALIDATOR_PATH = ROOT / "tools" / "mdl_knot" / "validate_mdl_knot_shortdiag.py"


def _run_validator(*extra_args):
    return subprocess.run(
        [sys.executable, str(VALIDATOR_PATH), "--config", str(CONFIG_PATH), *extra_args],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )


def test_shortdiag_config_extends_mdl_route_and_keeps_all_locks():
    cfg = runpy.run_path(str(CONFIG_PATH))

    assert cfg["_base_"] == ["./input_mdl_knot_dynamic_adapter_irregular_headv3.py"]
    assert cfg["route_label"] == ROUTE_LABEL
    assert cfg["shortdiag_gate"]["diagnostic_only"] is True
    assert cfg["shortdiag_gate"]["full_train_unlocked"] is False
    assert cfg["shortdiag_gate"]["metric_claim"] is False
    assert cfg["shortdiag_gate"]["sparse_compute_claim"] is False
    assert cfg["shortdiag_gate"]["max_epochs"] == 1
    assert cfg["shortdiag_gate"]["evaluation_locked"] is True
    assert cfg["shortdiag_gate"]["checkpoint_locked"] is True
    assert cfg["shortdiag_gate"]["tools_test_py_locked"] is True
    assert cfg["shortdiag_gate"]["map_claim_locked"] is True
    assert cfg["shortdiag_gate"]["claim_locked"] is True

    acq = cfg["mdl_knot_acquisition"]
    assert acq["deploy_scout_source"] == "raw_frame_motion_scout_with_metadata_fallback"
    assert acq["synthetic_fallback_allowed"] is False
    assert acq["scout_stride"] == 8
    assert acq["scout_max_frames"] == 96
    assert acq["dense_window_size"] == 768
    assert acq["window_size"] == 384
    assert acq["max_k"] == 384
    assert acq["handoff_audit_mode"] == "sampled_raw"
    assert acq["diagnostic_only"] is True
    assert acq["full_train_unlocked"] is False
    assert acq["metric_claim"] is False
    assert acq["sparse_compute_claim"] is False
    assert acq["route_label"] == ROUTE_LABEL

    assert cfg["shortdiag_gate"]["workflow"] == [("train", 1)]
    assert cfg["workflow"]["end_epoch"] == 1
    assert cfg["workflow"]["disable_checkpoint"] is True
    assert cfg["workflow"]["val_loss_interval"] == -1
    assert cfg["workflow"]["val_eval_interval"] == -1
    assert cfg["workflow"]["val_start_epoch"] > 1
    load = cfg["dataset"]["train"]["pipeline"][2]
    assert load["source_len"] == 768
    assert load["target_len"] == 384
    assert load["mdl_knot_max_k"] == 384
    assert load["mdl_knot_scout_stride"] == 8
    assert load["mdl_knot_scout_max_frames"] == 96
    assert load["mdl_knot_handoff_audit_mode"] == "sampled_raw"
    assert cfg["work_dir"].endswith("input_mdl_knot_dynamic_adapter_irregular_headv3_shortdiag")


def test_shortdiag_validator_config_only_is_static_check_not_execution_evidence():
    proc = _run_validator()

    assert proc.returncode == 0, proc.stderr
    assert "SHORT_DIAGNOSTIC_CONFIG_STATIC_CHECK_ALLOWED" in proc.stdout
    assert "Still locked" in proc.stdout
    evidence = json.loads(proc.stdout.split("SHORTDIAG_EVIDENCE=", 1)[1].splitlines()[0])
    assert evidence["diagnostic_only"] is True
    assert evidence["validated"] is False
    assert evidence["full_train_unlocked"] is False
    assert evidence["metric_claim"] is False
    assert evidence["sparse_compute_claim"] is False
    assert evidence["no_sparse_compute_claim"] is True
    assert evidence["log_evidence"] is None
    assert evidence["execution_evidence_required_for_formal_readiness"] is True
    assert evidence["evidence_scope"] == "static_config_only"
    assert evidence["scout_stride"] == 8
    assert evidence["scout_max_frames"] == 96
    assert evidence["dense_window_size"] == 768
    assert evidence["max_k"] == 384
    assert evidence["handoff_audit_mode"] == "sampled_raw"


def test_shortdiag_validator_accepts_finite_loss_log_but_keeps_claim_locked(tmp_path):
    log_path = tmp_path / "finite_loss.log"
    log_path.write_text(
        "\n".join(
            [
                "Epoch [1][1/2] lr: 1.0e-04 Loss 2.345 loss_cls: 0.55",
                "Epoch [1][2/2] loss_bbox=0.123 finite diagnostic train step complete",
                "Training short diagnostic stopped after one epoch without evaluation",
            ]
        ),
        encoding="utf-8",
    )

    proc = _run_validator("--train-log", str(log_path))

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "SHORT_DIAGNOSTIC_ONLY_REQUEST_ALLOWED" in proc.stdout
    assert "mAP" in proc.stdout
    assert "claims remain locked" in proc.stdout
    evidence = json.loads(proc.stdout.split("SHORTDIAG_EVIDENCE=", 1)[1].splitlines()[0])
    assert evidence["validated"] is True
    assert evidence["log_evidence"]["finite_loss_count"] >= 2
    assert evidence["execution_evidence_required_for_formal_readiness"] is False
    assert evidence["evidence_scope"] == "one_epoch_train_log"


def test_shortdiag_validator_rejects_bad_logs(tmp_path):
    cases = {
        "missing_loss.log": "Epoch [1][1/1] completed with lr 1e-4",
        "nan.log": "Epoch [1][1/1] Loss nan cost nan",
        "traceback.log": "Traceback (most recent call last): RuntimeError: CUDA out of memory",
        "eval.log": "tools/test.py result_detection.json mAP@0.7 0.55",
        "claim.log": "full_train_unlocked=True paper claim sparse_compute_claim=True",
    }

    for name, text in cases.items():
        log_path = tmp_path / name
        log_path.write_text(text, encoding="utf-8")
        proc = _run_validator("--train-log", str(log_path))
        assert proc.returncode != 0, name
        assert "LOCKED" in proc.stdout, name


def test_shortdiag_validator_rejects_route_drift_in_log(tmp_path):
    log_path = tmp_path / "route_drift.log"
    log_path.write_text("Epoch [1] Loss 1.25 C3 BVR ABR Event-Surprise combo", encoding="utf-8")

    proc = _run_validator("--train-log", str(log_path))

    assert proc.returncode != 0
    assert "route drift" in proc.stdout


@pytest.mark.parametrize("token", ["GlobalRank", "GLOBAL_RANK", "GLOBAL-RANK", "GLOBAL RANK"])
def test_shortdiag_validator_rejects_globalrank_drift_in_log(tmp_path, token):
    log_path = tmp_path / f"{token.replace(' ', '_')}.log"
    log_path.write_text(f"Epoch [1] Loss 1.25 {token} diagnostic drift marker", encoding="utf-8")

    proc = _run_validator("--train-log", str(log_path))

    assert proc.returncode != 0
    assert "LOCKED" in proc.stdout
    assert "route drift" in proc.stdout
    assert "SHORTDIAG_EVIDENCE=" not in proc.stdout
    assert "validated" not in proc.stdout


@pytest.mark.parametrize("token", ["GlobalRank", "GLOBAL_RANK", "GLOBAL-RANK", "GLOBAL RANK"])
def test_shortdiag_config_text_rejects_globalrank_drift(tmp_path, token):
    cfg, raw_cfg = _load_config_with_base(CONFIG_PATH)
    drift_config = tmp_path / "input_mdl_knot_dynamic_adapter_irregular_headv3_shortdiag.py"
    drift_config.write_text(
        CONFIG_PATH.read_text(encoding="utf-8") + f"\n# forbidden drift marker: {token}\n",
        encoding="utf-8",
    )

    status, evidence = _validate_shortdiag_config(drift_config, cfg, raw_cfg)

    assert status != 0
    assert evidence is None
