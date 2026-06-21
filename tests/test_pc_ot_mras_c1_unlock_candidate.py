import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "bata" / "build_pc_ot_mras_c1_unlock_candidate.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("pcot_c1_unlock_candidate", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _preflight():
    return {
        "schema_version": "pc_ot_mras_c1_acquisition_preflight_v0",
        "status": "LOCAL_PREFLIGHT_PASS_EXECUTION_LOCKED",
        "pass": True,
        "local_preflight_pass": True,
        "execution_allowed": False,
        "remote_sync_allowed": False,
        "slurm_allowed": False,
        "tools_test_allowed": False,
        "detector_map_reporting_allowed": False,
        "missing_before_execution": ["formal_gate_json_r17"],
    }


def _remote_sync_plan():
    return {
        "schema_version": "pc_ot_mras_c1_remote_sync_plan_v0",
        "status": "PLAN_READY_NOT_EXECUTED",
        "pass": True,
        "file_count": 8,
        "required_before_execution": ["remote_sync_plan_executed_and_verified"],
        "execution_unlock": {
            "remote_sync_allowed": False,
            "slurm_allowed": False,
            "tools_test_allowed": False,
            "detector_map_reporting_allowed": False,
        },
    }


def _submission_plan():
    return {
        "schema_version": "pc_ot_mras_c1_reader_disabled_submission_plan_v0",
        "status": "PLAN_READY_NOT_EXECUTED",
        "pass": True,
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
        "submission_commands": [
            {"target": "R17"},
            {"target": "R18"},
        ],
    }


def _execution_matrix():
    return {
        "schema_version": "pc_ot_mras_execution_matrix_verifier_v0",
        "pass": True,
        "current_execution_conclusion": "no_new_full_training_unlocked",
        "full_training_worthy_now": ["R17", "R18"],
        "c1_unlock_flags": {
            "remote_sync_allowed": False,
            "slurm_allowed": False,
            "tools_test_allowed": False,
            "detector_map_reporting_allowed": False,
        },
    }


def test_unlock_candidate_ready_but_non_executable(tmp_path):
    tool = _load_tool()
    report = tmp_path / "gemini_report.md"
    report.write_text("PASS_LOCAL_REVIEW_ONLY\n", encoding="utf-8")

    payload = tool.build_unlock_candidate(
        preflight=_preflight(),
        gemini_stdout="Verdict: PASS_LOCAL_REVIEW_ONLY",
        gemini_exitcode=0,
        gemini_report_path=report,
        remote_sync_plan=_remote_sync_plan(),
        submission_plan=_submission_plan(),
        execution_matrix=_execution_matrix(),
        generated_at="2026-06-22T06:00:00+08:00",
    )

    assert payload["schema_version"] == tool.SCHEMA_VERSION
    assert payload["status"] == tool.READY
    assert payload["pass"] is True
    assert payload["targets"] == ["R17", "R18"]
    assert payload["failed_checks"] == []
    assert payload["execution_unlock"]["remote_sync_allowed"] is False
    assert payload["execution_unlock"]["formal_gate_generation_allowed"] is False
    assert payload["execution_unlock"]["slurm_allowed"] is False
    assert payload["execution_unlock"]["tools_test_allowed"] is False
    assert payload["execution_unlock"]["detector_map_reporting_allowed"] is False
    assert payload["diagnostic_scope"]["reader_disabled_detector_control_ready_to_unlock"] is True
    assert payload["diagnostic_scope"]["detector_map_available"] is False
    assert payload["diagnostic_scope"]["same_head_control_available"] is False
    assert "explicit_user_approved_replacement_or_unlock_for_cancelled_pro_major_gate" in payload["missing_before_execution"]
    assert "deliberate_sbatch_submission" in payload["missing_before_execution"]


def test_unlock_candidate_rejects_bad_gemini_evidence(tmp_path):
    tool = _load_tool()
    report = tmp_path / "gemini_report.md"
    report.write_text("review text\n", encoding="utf-8")

    payload = tool.build_unlock_candidate(
        preflight=_preflight(),
        gemini_stdout="Verdict: FAIL",
        gemini_exitcode=1,
        gemini_report_path=report,
        remote_sync_plan=_remote_sync_plan(),
        submission_plan=_submission_plan(),
        execution_matrix=_execution_matrix(),
    )

    assert payload["status"] == tool.FAILED
    assert payload["pass"] is False
    assert any(item["name"] == "gemini_cli_review_pass_local_only" for item in payload["failed_checks"])
    assert payload["execution_unlock"]["slurm_allowed"] is False


def test_unlock_candidate_reads_utf16_gemini_stdout(tmp_path):
    tool = _load_tool()
    stdout = tmp_path / "gemini_stdout.txt"
    stdout.write_text("Verdict: PASS_LOCAL_REVIEW_ONLY\n", encoding="utf-16")

    assert "PASS_LOCAL_REVIEW_ONLY" in tool._read_text(stdout)


def test_unlock_candidate_rejects_unlocked_remote_sync_plan(tmp_path):
    tool = _load_tool()
    report = tmp_path / "gemini_report.md"
    report.write_text("PASS_LOCAL_REVIEW_ONLY\n", encoding="utf-8")
    sync_plan = _remote_sync_plan()
    sync_plan["execution_unlock"]["remote_sync_allowed"] = True

    payload = tool.build_unlock_candidate(
        preflight=_preflight(),
        gemini_stdout="PASS_LOCAL_REVIEW_ONLY",
        gemini_exitcode=0,
        gemini_report_path=report,
        remote_sync_plan=sync_plan,
        submission_plan=_submission_plan(),
        execution_matrix=_execution_matrix(),
    )

    assert payload["status"] == tool.FAILED
    assert any(item["name"] == "remote_sync_plan_ready_not_executed" for item in payload["failed_checks"])
    assert payload["execution_unlock"]["remote_sync_allowed"] is False


def test_unlock_candidate_rejects_unlocked_submission_plan(tmp_path):
    tool = _load_tool()
    report = tmp_path / "gemini_report.md"
    report.write_text("PASS_LOCAL_REVIEW_ONLY\n", encoding="utf-8")
    submission_plan = _submission_plan()
    submission_plan["execution_unlock"]["tools_test_allowed"] = True

    payload = tool.build_unlock_candidate(
        preflight=_preflight(),
        gemini_stdout="PASS_LOCAL_REVIEW_ONLY",
        gemini_exitcode=0,
        gemini_report_path=report,
        remote_sync_plan=_remote_sync_plan(),
        submission_plan=submission_plan,
        execution_matrix=_execution_matrix(),
    )

    assert payload["status"] == tool.FAILED
    assert any(item["name"] == "submission_plan_ready_not_executed" for item in payload["failed_checks"])
    assert payload["execution_unlock"]["tools_test_allowed"] is False


def test_unlock_candidate_rejects_execution_matrix_c1_unlock(tmp_path):
    tool = _load_tool()
    report = tmp_path / "gemini_report.md"
    report.write_text("PASS_LOCAL_REVIEW_ONLY\n", encoding="utf-8")
    execution_matrix = _execution_matrix()
    execution_matrix["c1_unlock_flags"]["detector_map_reporting_allowed"] = True

    payload = tool.build_unlock_candidate(
        preflight=_preflight(),
        gemini_stdout="PASS_LOCAL_REVIEW_ONLY",
        gemini_exitcode=0,
        gemini_report_path=report,
        remote_sync_plan=_remote_sync_plan(),
        submission_plan=_submission_plan(),
        execution_matrix=execution_matrix,
    )

    assert payload["status"] == tool.FAILED
    assert any(item["name"] == "execution_matrix_pass_no_new_training_unlocked" for item in payload["failed_checks"])
    assert payload["execution_unlock"]["detector_map_reporting_allowed"] is False
