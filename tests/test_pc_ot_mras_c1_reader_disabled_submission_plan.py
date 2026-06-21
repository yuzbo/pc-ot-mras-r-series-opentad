import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "bata" / "plan_pc_ot_mras_c1_reader_disabled_submission.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("pcot_c1_submission_plan", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gate_draft():
    return {
        "schema_version": "pc_ot_mras_reader_disabled_c1_gate_binding_draft_v0",
        "status": "DRAFT_ONLY_NOT_EXECUTABLE",
        "execution_unlock": {
            "remote_sync_allowed": False,
            "slurm_allowed": False,
            "tools_test_allowed": False,
            "metric_claim_allowed": False,
            "paper_claim_allowed": False,
        },
        "targets": {
            "R17": {
                "decision_if_formalized": "ALLOW_R17_READER_DISABLED_EVAL",
                "config": "configs/adatad/thumos/ctf_bdi_pc_ot_mras_r17_reader_disabled_eval_candidate.py",
                "checkpoint": "/remote/r17/epoch_59.pth",
                "checkpoint_sha256": "1" * 64,
            },
            "R18": {
                "decision_if_formalized": "ALLOW_R18_READER_DISABLED_EVAL",
                "config": "configs/adatad/thumos/ctf_bdi_pc_ot_mras_r18_reader_disabled_eval_candidate.py",
                "checkpoint": "/remote/r18/epoch_59.pth",
                "checkpoint_sha256": "2" * 64,
            },
        },
    }


def _sync_plan():
    return {
        "schema_version": "pc_ot_mras_c1_remote_sync_plan_v0",
        "status": "PLAN_READY_NOT_EXECUTED",
        "remote_root": "/data/run01/sczc063/yuzibo/OpenTAD_PCOTMRAS_CPU_PRERUN_20260620_1418",
        "execution_unlock": {
            "remote_sync_allowed": False,
            "slurm_allowed": False,
            "tools_test_allowed": False,
            "detector_map_reporting_allowed": False,
        },
    }


def test_submission_plan_is_fail_closed_and_targets_r17_r18():
    tool = _load_tool()

    payload = tool.build_submission_plan(
        _gate_draft(),
        _sync_plan(),
        generated_at="2026-06-22T05:45:00+08:00",
    )

    assert payload["schema_version"] == tool.SCHEMA_VERSION
    assert payload["status"] == "PLAN_READY_NOT_EXECUTED"
    assert payload["execution_unlock"]["formal_gate_generation_allowed"] is False
    assert payload["execution_unlock"]["slurm_allowed"] is False
    assert payload["execution_unlock"]["tools_test_allowed"] is False
    assert payload["execution_unlock"]["detector_map_reporting_allowed"] is False
    assert len(payload["formal_gate_commands"]) == 2
    assert len(payload["submission_commands"]) == 2

    by_target = {item["target"]: item for item in payload["submission_commands"]}
    assert set(by_target) == {"R17", "R18"}
    assert by_target["R17"]["env"]["PRECHECK_ONLY"] == "0"
    assert by_target["R17"]["env"]["ALLOW_R17_READER_DISABLED_EVAL"] == "1"
    assert by_target["R17"]["env"]["R17_EVAL_CHECKPOINT_SHA256"] == "1" * 64
    assert by_target["R17"]["env"]["ALLOW_PCOTMRAS_READER_DISABLED_TOOLS_TRAIN"] == "0"
    assert by_target["R17"]["env"]["ALLOW_PCOTMRAS_READER_DISABLED_RAW_PREDICTION_CACHE"] == "0"
    assert by_target["R18"]["env"]["ALLOW_R18_READER_DISABLED_EVAL"] == "1"
    assert by_target["R18"]["env"]["R18_EVAL_CHECKPOINT_SHA256"] == "2" * 64
    assert all(command["requires_unlock"] is True for command in payload["submission_commands"])
    assert all(command["args"] == ["scripts/run_ctf_bdi_pc_ot_mras_reader_disabled_eval_n16r4.sbatch"] for command in payload["submission_commands"])


def test_submission_plan_rejects_unlocked_gate_draft():
    tool = _load_tool()
    draft = _gate_draft()
    draft["execution_unlock"]["slurm_allowed"] = True

    with pytest.raises(tool.SubmissionPlanError, match="slurm_allowed"):
        tool.build_submission_plan(draft, _sync_plan())


def test_submission_plan_rejects_executed_or_unlocked_sync_plan():
    tool = _load_tool()
    sync_plan = _sync_plan()
    sync_plan["status"] = "SYNC_EXECUTED"

    with pytest.raises(tool.SubmissionPlanError, match="must not be executed"):
        tool.build_submission_plan(_gate_draft(), sync_plan)

    sync_plan = _sync_plan()
    sync_plan["execution_unlock"]["tools_test_allowed"] = True
    with pytest.raises(tool.SubmissionPlanError, match="tools_test_allowed"):
        tool.build_submission_plan(_gate_draft(), sync_plan)


def test_submission_plan_requires_both_targets():
    tool = _load_tool()
    draft = _gate_draft()
    del draft["targets"]["R18"]

    with pytest.raises(tool.SubmissionPlanError, match="R17 and R18"):
        tool.build_submission_plan(draft, _sync_plan())
