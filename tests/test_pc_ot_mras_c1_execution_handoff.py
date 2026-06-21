import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "bata" / "build_pc_ot_mras_c1_execution_handoff.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("pcot_c1_execution_handoff", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _unlock_candidate():
    return {
        "schema_version": "pc_ot_mras_c1_unlock_candidate_v0",
        "status": "UNLOCK_CANDIDATE_READY_NOT_EXECUTABLE",
        "pass": True,
        "missing_before_execution": [
            "explicit_user_approved_replacement_or_unlock_for_cancelled_pro_major_gate"
        ],
        "execution_unlock": {
            "remote_sync_allowed": False,
            "formal_gate_generation_allowed": False,
            "slurm_allowed": False,
            "tools_test_allowed": False,
            "detector_map_reporting_allowed": False,
            "metric_claim_allowed": False,
            "paper_claim_allowed": False,
            "runtime_flops_claim_allowed": False,
            "deploy_claim_allowed": False,
        },
        "diagnostic_scope": {
            "reader_disabled_detector_control_ready_to_unlock": True,
        },
    }


def _remote_sync_plan():
    return {
        "schema_version": "pc_ot_mras_c1_remote_sync_plan_v0",
        "status": "PLAN_READY_NOT_EXECUTED",
        "pass": True,
        "file_count": 2,
        "remote_root": "/data/run01/sczc063/yuzibo/OpenTAD_PCOTMRAS_CPU_PRERUN_20260620_1418",
        "required_before_execution": ["post_sync_remote_sha_verification"],
        "execution_unlock": {
            "remote_sync_allowed": False,
            "slurm_allowed": False,
            "tools_test_allowed": False,
            "detector_map_reporting_allowed": False,
        },
        "commands": {
            "mkdir": [{"program": "ssh"}, {"program": "ssh"}],
            "copy": [{"program": "scp"}, {"program": "scp"}],
            "verify_after_sync": {"program": "ssh"},
        },
    }


def _formal_gate_command(target):
    return {
        "target": target,
        "program": "python",
        "requires_unlock": True,
        "writes_local_gate": True,
        "args": ["tools/bata/finalize_pc_ot_mras_reader_disabled_gate.py"],
    }


def _submission_command(target):
    return {
        "target": target,
        "program": "sbatch",
        "requires_unlock": True,
        "requires_slurm": True,
        "writes_remote": True,
        "args": ["scripts/run_ctf_bdi_pc_ot_mras_reader_disabled_eval_n16r4.sbatch"],
        "expected_artifacts": [
            f"logs/ctf_bdi_pc_ot_mras_{target.lower()}_reader_disabled_eval_*/eval_workdir/gpu1_id0/result_detection.json"
        ],
        "env": {
            "PRECHECK_ONLY": "0",
            "PCOTMRAS_READER_DISABLED_TARGET": target,
            f"ALLOW_{target}_READER_DISABLED_EVAL": "1",
            "ALLOW_PCOTMRAS_READER_DISABLED_TOOLS_TRAIN": "0",
            "ALLOW_PCOTMRAS_READER_DISABLED_RAW_PREDICTION_CACHE": "0",
            "ALLOW_PCOTMRAS_READER_DISABLED_RUNTIME_OR_DEPLOY_CLAIM": "0",
        },
    }


def _submission_plan():
    return {
        "schema_version": "pc_ot_mras_c1_reader_disabled_submission_plan_v0",
        "status": "PLAN_READY_NOT_EXECUTED",
        "pass": True,
        "remote_root": "/data/run01/sczc063/yuzibo/OpenTAD_PCOTMRAS_CPU_PRERUN_20260620_1418",
        "required_before_execution": ["operator_deliberately_submits_sbatch_commands"],
        "execution_unlock": {
            "formal_gate_generation_allowed": False,
            "remote_sync_allowed": False,
            "slurm_allowed": False,
            "tools_test_allowed": False,
            "detector_map_reporting_allowed": False,
            "metric_claim_allowed": False,
            "paper_claim_allowed": False,
        },
        "formal_gate_commands": [_formal_gate_command("R17"), _formal_gate_command("R18")],
        "submission_commands": [_submission_command("R17"), _submission_command("R18")],
    }


def test_execution_handoff_ready_but_non_executable():
    tool = _load_tool()
    payload = tool.build_execution_handoff(
        unlock_candidate=_unlock_candidate(),
        remote_sync_plan=_remote_sync_plan(),
        submission_plan=_submission_plan(),
        generated_at="2026-06-22T06:20:00+08:00",
    )

    assert payload["schema_version"] == tool.SCHEMA_VERSION
    assert payload["status"] == tool.READY
    assert payload["pass"] is True
    assert payload["execution_allowed_now"] is False
    assert payload["execution_unlock"]["remote_sync_allowed"] is False
    assert payload["execution_unlock"]["formal_gate_generation_allowed"] is False
    assert payload["execution_unlock"]["slurm_allowed"] is False
    assert payload["execution_unlock"]["tools_test_allowed"] is False
    assert payload["execution_unlock"]["detector_map_reporting_allowed"] is False
    assert [stage["name"] for stage in payload["handoff_stages"]] == [
        "review_replacement_or_user_unlock",
        "remote_sync_after_unlock",
        "remote_precheck_after_sync",
        "formal_gate_generation_after_precheck",
        "slurm_reader_disabled_eval_submission",
    ]
    assert all(stage["executable_now"] is False for stage in payload["handoff_stages"])
    assert "formal_gate_sha256_r17" in payload["required_before_execution"]
    assert payload["diagnostic_scope"]["detector_map_available"] is False
    assert payload["diagnostic_scope"]["tools_test_executed"] is False


def test_execution_handoff_rejects_unlocked_candidate():
    tool = _load_tool()
    candidate = _unlock_candidate()
    candidate["execution_unlock"]["remote_sync_allowed"] = True

    payload = tool.build_execution_handoff(
        unlock_candidate=candidate,
        remote_sync_plan=_remote_sync_plan(),
        submission_plan=_submission_plan(),
    )

    assert payload["status"] == tool.FAILED
    assert any(item["name"] == "unlock_candidate_ready_not_executable" for item in payload["failed_checks"])
    assert payload["execution_unlock"]["remote_sync_allowed"] is False


def test_execution_handoff_rejects_remote_sync_without_verify_command():
    tool = _load_tool()
    sync_plan = _remote_sync_plan()
    sync_plan["commands"].pop("verify_after_sync")

    payload = tool.build_execution_handoff(
        unlock_candidate=_unlock_candidate(),
        remote_sync_plan=sync_plan,
        submission_plan=_submission_plan(),
    )

    assert payload["status"] == tool.FAILED
    assert any(item["name"] == "remote_sync_plan_ready_with_commands" for item in payload["failed_checks"])


def test_execution_handoff_rejects_submission_env_that_can_run_without_unlock():
    tool = _load_tool()
    submission_plan = _submission_plan()
    submission_plan["submission_commands"][0]["requires_unlock"] = False

    payload = tool.build_execution_handoff(
        unlock_candidate=_unlock_candidate(),
        remote_sync_plan=_remote_sync_plan(),
        submission_plan=submission_plan,
    )

    assert payload["status"] == tool.FAILED
    failed = next(item for item in payload["failed_checks"] if item["name"] == "submission_plan_ready_with_r17_r18_commands")
    assert failed["detail"]["submission_env_offenders"] == ["R17"]


def test_execution_handoff_rejects_remote_root_mismatch():
    tool = _load_tool()
    submission_plan = _submission_plan()
    submission_plan["remote_root"] = "/tmp/wrong"

    payload = tool.build_execution_handoff(
        unlock_candidate=_unlock_candidate(),
        remote_sync_plan=_remote_sync_plan(),
        submission_plan=submission_plan,
    )

    assert payload["status"] == tool.FAILED
    assert any(item["name"] == "remote_roots_consistent" for item in payload["failed_checks"])
