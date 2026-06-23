import subprocess
import types

from tools.bata.precheck_p2_quality_rank_calibrator_full_env import (
    HOLD_STATUS,
    PASS_STATUS,
    check_imports,
    run_precheck,
    run_runtime_smoke,
)


def test_check_imports_reports_missing_module():
    def importer(name):
        if name == "missing.module":
            raise ModuleNotFoundError("missing.module")
        return types.SimpleNamespace(__version__="1.0")

    result = check_imports(("ok.module", "missing.module"), importer=importer)

    assert result["pass"] is False
    assert result["checks"]["ok.module"]["pass"] is True
    assert result["checks"]["missing.module"]["pass"] is False
    assert result["checks"]["missing.module"]["error_type"] == "ModuleNotFoundError"


def test_runtime_smoke_not_requested_is_hold_input():
    result = run_runtime_smoke("tests/example.py", run=False)

    assert result["pass"] is False
    assert result["ran"] is False
    assert result["decision"] == "RUNTIME_SMOKE_NOT_REQUESTED"


def test_runtime_smoke_runner_success_and_failure():
    def ok_runner(command, text, stdout, stderr, check):
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    def fail_runner(command, text, stdout, stderr, check):
        return subprocess.CompletedProcess(command, 7, stdout="bad", stderr="err")

    ok = run_runtime_smoke("tests/example.py", run=True, command_prefix=["python", "-m", "pytest"], runner=ok_runner)
    fail = run_runtime_smoke("tests/example.py", run=True, command_prefix=["python", "-m", "pytest"], runner=fail_runner)

    assert ok["pass"] is True
    assert ok["command"] == ["python", "-m", "pytest", "tests/example.py", "-q"]
    assert fail["pass"] is False
    assert fail["returncode"] == 7


def test_run_precheck_holds_when_runtime_not_requested(monkeypatch):
    monkeypatch.setattr(
        "tools.bata.precheck_p2_quality_rank_calibrator_full_env.check_imports",
        lambda: {"pass": True, "checks": {}},
    )
    monkeypatch.setattr(
        "tools.bata.precheck_p2_quality_rank_calibrator_full_env.parse_config",
        lambda config_path: {"pass": True},
    )

    result = run_precheck(run_smoke=False)

    assert result["status"] == HOLD_STATUS
    assert result["pass"] is False
    assert result["blockers"] == ["RUNTIME_SMOKE_FAILED_OR_NOT_RUN"]
    assert result["permissions"]["remote_sync_allowed"] is False


def test_run_precheck_passes_only_when_all_checks_pass(monkeypatch):
    monkeypatch.setattr(
        "tools.bata.precheck_p2_quality_rank_calibrator_full_env.check_imports",
        lambda: {"pass": True, "checks": {}},
    )
    monkeypatch.setattr(
        "tools.bata.precheck_p2_quality_rank_calibrator_full_env.parse_config",
        lambda config_path: {"pass": True},
    )
    monkeypatch.setattr(
        "tools.bata.precheck_p2_quality_rank_calibrator_full_env.run_runtime_smoke",
        lambda runtime_smoke_path, run, command_prefix: {"pass": True, "ran": True},
    )

    result = run_precheck(run_smoke=True)

    assert result["status"] == PASS_STATUS
    assert result["pass"] is True
    assert result["blockers"] == []
    assert result["permissions"]["slurm_allowed"] is False
