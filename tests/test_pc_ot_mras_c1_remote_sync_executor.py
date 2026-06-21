import importlib.util
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "bata" / "execute_pc_ot_mras_c1_remote_sync.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("pcot_c1_remote_sync_executor", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _remote_sync_plan():
    return {
        "schema_version": "pc_ot_mras_c1_remote_sync_plan_v0",
        "status": "PLAN_READY_NOT_EXECUTED",
        "pass": True,
        "file_count": 1,
        "execution_unlock": {
            "remote_sync_allowed": False,
            "slurm_allowed": False,
            "tools_test_allowed": False,
            "detector_map_reporting_allowed": False,
            "metric_claim_allowed": False,
            "paper_claim_allowed": False,
        },
        "commands": {
            "mkdir": [
                {
                    "program": "ssh.exe",
                    "args": ["mkdir"],
                    "writes_remote": True,
                    "purpose": "mkdir",
                }
            ],
            "copy": [
                {
                    "program": "scp.exe",
                    "args": ["copy"],
                    "writes_remote": True,
                    "purpose": "copy",
                }
            ],
            "verify_after_sync": {
                "program": "ssh.exe",
                "args": ["bash", "-s"],
                "stdin": "sha256sum file\n",
                "writes_remote": False,
            },
        },
    }


def _unlock_checklist(*, completed=False):
    status = "completed" if completed else "missing"
    return {
        "schema_version": "pc_ot_mras_c1_unlock_checklist_v0",
        "status": "UNLOCK_CHECKLIST_READY_NO_EXECUTION",
        "pass": True,
        "execution_allowed_now": False,
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
        "checklist": [
            {
                "key": "explicit_user_approved_replacement_or_unlock_for_cancelled_pro_major_gate",
                "status": status,
            },
            {
                "key": "tracker_manifest_unlocked_for_c1_remote_sync",
                "status": status,
            },
        ],
    }


def test_remote_sync_executor_default_dryrun_does_not_call_runner():
    tool = _load_tool()
    calls = []

    def runner(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    payload = tool.build_remote_sync_execution(
        remote_sync_plan=_remote_sync_plan(),
        unlock_checklist=_unlock_checklist(),
        allow_remote_write=False,
        runner=runner,
    )

    assert payload["status"] == tool.DRYRUN_STATUS
    assert payload["pass"] is True
    assert payload["remote_write_executed"] is False
    assert calls == []
    assert payload["execution_unlock"]["remote_sync_allowed"] is False


def test_remote_sync_executor_blocks_allow_without_completed_checklist(tmp_path):
    tool = _load_tool()
    evidence = tmp_path / "unlock.md"
    evidence.write_text("manual C1 unlock\n", encoding="utf-8")

    payload = tool.build_remote_sync_execution(
        remote_sync_plan=_remote_sync_plan(),
        unlock_checklist=_unlock_checklist(completed=False),
        allow_remote_write=True,
        user_unlock_evidence=evidence,
    )

    assert payload["status"] == tool.BLOCKED_STATUS
    assert payload["pass"] is False
    assert "explicit_user_approved_replacement_or_unlock_for_cancelled_pro_major_gate" in payload["missing_remote_write_prereqs"]
    assert payload["remote_write_executed"] is False


def test_remote_sync_executor_blocks_unlocked_plan_even_in_dryrun():
    tool = _load_tool()
    plan = _remote_sync_plan()
    plan["execution_unlock"]["remote_sync_allowed"] = True

    payload = tool.build_remote_sync_execution(
        remote_sync_plan=plan,
        unlock_checklist=_unlock_checklist(),
        allow_remote_write=False,
    )

    assert payload["status"] == tool.BLOCKED_STATUS
    assert payload["pass"] is False
    assert any(item["name"] == "remote_sync_plan_ready_not_executed" for item in payload["failed_checks"])
    assert payload["execution_unlock"]["remote_sync_allowed"] is False


def test_remote_sync_executor_executes_with_completed_checklist_and_mock_runner(tmp_path):
    tool = _load_tool()
    evidence = tmp_path / "unlock.md"
    evidence.write_text("manual C1 unlock\n", encoding="utf-8")
    calls = []

    def runner(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    payload = tool.build_remote_sync_execution(
        remote_sync_plan=_remote_sync_plan(),
        unlock_checklist=_unlock_checklist(completed=True),
        allow_remote_write=True,
        user_unlock_evidence=evidence,
        runner=runner,
    )

    assert payload["status"] == tool.EXECUTED_STATUS
    assert payload["pass"] is True
    assert payload["remote_write_executed"] is True
    assert len(calls) == 3
    assert len(payload["execution_records"]) == 3
    assert payload["execution_unlock"]["slurm_allowed"] is False


def test_remote_sync_executor_stops_on_first_command_failure(tmp_path):
    tool = _load_tool()
    evidence = tmp_path / "unlock.md"
    evidence.write_text("manual C1 unlock\n", encoding="utf-8")
    calls = []

    def runner(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=1, stdout="", stderr="boom")

    payload = tool.build_remote_sync_execution(
        remote_sync_plan=_remote_sync_plan(),
        unlock_checklist=_unlock_checklist(completed=True),
        allow_remote_write=True,
        user_unlock_evidence=evidence,
        runner=runner,
    )

    assert payload["status"] == tool.BLOCKED_STATUS
    assert payload["pass"] is False
    assert payload["remote_write_executed"] is False
    assert len(calls) == 1
    assert payload["execution_records"][0]["returncode"] == 1
