import json
import subprocess
import sys
from pathlib import Path

import pytest

from opentad.acquisition.abr import ABR_ROUTE_LABEL
from opentad.acquisition.abr.validators import ABRValidationError
from tools.abr.validate_abr_shortdiag import validate_config, validate_shortdiag, validate_train_log

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs" / "adatad" / "thumos" / (
    "input_abr_active_bracket_refinement_adapter_irregular_headv3_shortdiag.py"
)
WRAPPER = REPO_ROOT / "logs" / "run_abr_shortdiag_n16r4.sh"


def test_shortdiag_config_extends_abr_and_keeps_all_claim_locks():
    decision = validate_shortdiag(CONFIG)

    assert decision["route_label"] == ABR_ROUTE_LABEL
    assert decision["allowed_next_action"] == "ONE_EPOCH_TRAIN_LOSS_DIAGNOSTIC_ONLY"
    assert decision["diagnostic_only"] is True
    assert decision["full_train_unlocked"] is False
    assert decision["metric_claim"] is False
    assert decision["sparse_compute_claim"] is False
    assert "FORMAL_FULL_TRAIN" in decision["still_locked"]
    assert "TOOLS_TEST_PY" in decision["still_locked"]
    assert "MAPPAPER_CLAIM" in decision["still_locked"]

    namespace = validate_config(CONFIG)
    assert namespace["_base_"] == ["./input_abr_active_bracket_refinement_adapter_irregular_headv3.py"]
    assert namespace["abr_route"]["diagnostic_only"] is True
    assert namespace["abr_route"]["full_train_unlocked"] is False
    assert namespace["abr_route"]["metric_claim"] is False
    assert namespace["abr_route"]["sparse_compute_claim"] is False
    assert namespace["shortdiag_gate"]["checkpoint_allowed"] is False
    assert namespace["shortdiag_gate"]["tools_test_py_allowed"] is False
    assert namespace["shortdiag_gate"]["evaluation_allowed"] is False
    assert namespace["workflow"]["end_epoch"] == 1
    assert namespace["workflow"]["disable_checkpoint"] is True
    assert namespace["workflow"]["val_eval_interval"] == -1
    assert namespace["workflow"]["val_loss_interval"] == -1
    assert namespace["workflow"]["val_start_epoch"] > namespace["workflow"]["end_epoch"]
    assert namespace["inference"]["save_raw_prediction"] is False
    assert namespace["post_processing"]["save_dict"] is False


def test_shortdiag_validator_accepts_pseudo_finite_loss_log_but_keeps_claims_locked(tmp_path):
    log_path = tmp_path / "abr_shortdiag_good.log"
    log_path.write_text(
        "\n".join(
            [
                "route_label=DIVERGENT_INNOVATION_ABR_DO_NOT_MERGE_WITH_C3",
                "[Train]: Epoch 0 started",
                "[Train]: [000][00001/00002] Loss=1.2345 cls_loss=0.1000 reg_loss=0.2000 cost=1.2345",
                "SHORT_DIAGNOSTIC_ONLY full_train_unlocked=False metric_claim=False sparse_compute_claim=False",
                "Training Over",
            ]
        ),
        encoding="utf-8",
    )

    summary = validate_train_log(log_path)
    assert summary["finite_loss_count"] == 1
    assert summary["last_loss"] == pytest.approx(1.2345)
    assert summary["full_train_unlocked"] is False
    assert summary["metric_claim"] is False
    assert summary["sparse_compute_claim"] is False

    decision = validate_shortdiag(CONFIG, log_path)
    assert decision["train_log"]["finite_loss_count"] == 1
    assert decision["metric_claim"] is False
    assert "FORMAL_FULL_TRAIN" in decision["still_locked"]


@pytest.mark.parametrize(
    "log_text",
    [
        "[Train]: Epoch 0 started\nTraining Over\n",
        "[Train]: [000][00001/00002] Loss=nan cost=nan\n",
        "[Train]: [000][00001/00002] Loss=1.0\nRuntimeError: failed\n",
        "[Train]: [000][00001/00002] Loss=1.0\nCUDA out of memory\n",
        "[Train]: [000][00001/00002] Loss=1.0\nKilled\n",
        "[Train]: [000][00001/00002] Loss=1.0\nNo space left on device\n",
        "[Train]: [000][00001/00002] Loss=1.0\nno GPU available\n",
        "[Train]: [000][00001/00002] Loss=1.0\nAverage-mAP: 65.00\n",
        "[Train]: [000][00001/00002] Loss=1.0\ntools/test.py checkpoint.pth\n",
        "[Train]: [000][00001/00002] Loss=1.0\nresult_detection.json written\n",
        "[Train]: [000][00001/00002] Loss=1.0\nborrow C3-Pro GlobalRank\n",
        "[Train]: [000][00001/00002] Loss=1.0\nBVR route improvement\n",
        "[Train]: [000][00001/00002] Loss=1.0\nMDL selector attached\n",
        "[Train]: [000][00001/00002] Loss=1.0\ncombo route approved\n",
        "[Train]: [000][00001/00002] Loss=1.0\nfull_train_unlocked=True\n",
        "[Train]: [000][00001/00002] Loss=1.0\npaper_claim=True\n",
    ],
)
def test_shortdiag_validator_rejects_bad_logs_and_route_drift(tmp_path, log_text):
    log_path = tmp_path / "bad.log"
    log_path.write_text(log_text, encoding="utf-8")

    with pytest.raises(ABRValidationError):
        validate_shortdiag(CONFIG, log_path)


def test_shortdiag_cli_returns_locked_for_bad_log_and_json_for_good_log(tmp_path):
    good = tmp_path / "good.log"
    good.write_text("[Train]: [000][00001/00002] Loss=2.5000 cost=2.5000\n", encoding="utf-8")
    bad = tmp_path / "bad.log"
    bad.write_text("[Train]: [000][00001/00002] Loss=1.0\nmAP@0.7=99.0\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "abr" / "validate_abr_shortdiag.py"),
            "--config",
            str(CONFIG),
            "--train-log",
            str(good),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    payload = json.loads(proc.stdout)
    assert payload["train_log"]["finite_loss_count"] == 1
    assert payload["metric_claim"] is False

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "abr" / "validate_abr_shortdiag.py"),
            "--config",
            str(CONFIG),
            "--train-log",
            str(bad),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert proc.returncode != 0
    assert "LOCKED" in proc.stdout


def test_n16r4_wrapper_is_child_context_only_and_has_no_slurm_control_commands():
    source = WRAPPER.read_text(encoding="utf-8")
    executable_lines = [
        line.strip()
        for line in source.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    executable = "\n".join(executable_lines)

    assert "tools/test.py" not in executable
    assert "torchrun" in source
    assert "--not_eval" in source
    assert "ABR_CHILD_GPU_CONTEXT" in source
    assert "ABR_SHORTDIAG_ACK" in source
    assert "SLURM_JOB_ID" in source
    assert "sbatch" not in executable
    assert "scancel" not in executable
    assert "scontrol release" not in executable
    assert "scontrol hold" not in executable
