import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "bata" / "plan_pc_ot_mras_c1_remote_sync.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("pcot_c1_remote_sync_plan", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _repo_with_files(tmp_path, files):
    repo = tmp_path / "repo"
    for rel_path in files:
        path = repo / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {rel_path}\n", encoding="utf-8")
    return repo


def _sync_package(repo, files):
    return {
        "schema_version": "pc_ot_mras_c1_sync_package_dryrun_v0",
        "status": "DRYRUN_ONLY_NOT_SYNCABLE",
        "pass": True,
        "repo": str(repo),
        "files": [
            {
                "path": rel_path,
                "sha256": "a" * 64,
                "size_bytes": 17,
            }
            for rel_path in files
        ],
        "execution_unlock": {
            "remote_sync_allowed": False,
            "slurm_allowed": False,
            "tools_test_allowed": False,
            "detector_map_reporting_allowed": False,
        },
    }


def _remote_gap():
    return {
        "schema_version": "pc_ot_mras_c1_remote_sync_gap_audit_v0",
        "status": "REMOTE_SYNC_GAP_CONFIRMED_EXECUTION_LOCKED",
        "remote_check": {
            "remote_root": "/data/run01/sczc063/yuzibo/OpenTAD_PCOTMRAS_CPU_PRERUN_20260620_1418",
            "remote_c1_source_ready": False,
        },
        "execution_unlock": {
            "remote_sync_allowed": False,
            "slurm_allowed": False,
            "tools_test_allowed": False,
            "detector_map_reporting_allowed": False,
        },
    }


def test_remote_sync_plan_is_native_openssh_dryrun_only(tmp_path):
    tool = _load_tool()
    files = (
        "opentad/models/detectors/actionformer.py",
        "configs/adatad/thumos/ctf_bdi_pc_ot_mras_r17_reader_disabled_eval_candidate.py",
    )
    repo = _repo_with_files(tmp_path, files)

    payload = tool.build_remote_sync_plan(
        _sync_package(repo, files),
        _remote_gap(),
        generated_at="2026-06-22T05:40:00+08:00",
    )

    assert payload["schema_version"] == tool.SCHEMA_VERSION
    assert payload["status"] == "PLAN_READY_NOT_EXECUTED"
    assert payload["file_count"] == len(files)
    assert payload["execution_unlock"]["remote_sync_allowed"] is False
    assert payload["execution_unlock"]["slurm_allowed"] is False
    assert payload["execution_unlock"]["tools_test_allowed"] is False
    assert all(command["program"].endswith("OpenSSH\\scp.exe") for command in payload["commands"]["copy"])
    assert all("-P" in command["args"] for command in payload["commands"]["copy"])
    assert all("-l" not in command["args"] for command in payload["commands"]["copy"])
    assert all(any(str(arg).startswith("User=sczc063@BSCC-N16R4") for arg in command["args"]) for command in payload["commands"]["copy"])
    assert "sha256sum" in payload["commands"]["verify_after_sync"]["stdin"]


def test_remote_sync_plan_rejects_remote_root_outside_yuzibo(tmp_path):
    tool = _load_tool()
    files = ("opentad/models/detectors/actionformer.py",)
    repo = _repo_with_files(tmp_path, files)
    gap = _remote_gap()
    gap["remote_check"]["remote_root"] = "/tmp/not-yuzibo"

    with pytest.raises(tool.RemoteSyncPlanError, match="outside allowed yuzibo workspace"):
        tool.build_remote_sync_plan(_sync_package(repo, files), gap)


def test_remote_sync_plan_rejects_unlocking_sync_package(tmp_path):
    tool = _load_tool()
    files = ("opentad/models/detectors/actionformer.py",)
    repo = _repo_with_files(tmp_path, files)
    package = _sync_package(repo, files)
    package["execution_unlock"]["remote_sync_allowed"] = True

    with pytest.raises(tool.RemoteSyncPlanError, match="remote_sync_allowed"):
        tool.build_remote_sync_plan(package, _remote_gap())


def test_remote_sync_plan_rejects_forbidden_artifact_path(tmp_path):
    tool = _load_tool()
    files = ("checkpoint/epoch_59.pth",)
    repo = _repo_with_files(tmp_path, files)

    with pytest.raises(tool.RemoteSyncPlanError, match="forbidden artifact"):
        tool.build_remote_sync_plan(_sync_package(repo, files), _remote_gap())
