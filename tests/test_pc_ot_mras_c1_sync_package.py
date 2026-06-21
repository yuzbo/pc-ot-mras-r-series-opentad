import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "bata" / "build_pc_ot_mras_c1_sync_package.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("pcot_c1_sync_package", TOOL)
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


def test_build_sync_package_accepts_expected_c1_source_set(tmp_path):
    tool = _load_tool()
    repo = _repo_with_files(tmp_path, tool.C1_SYNC_FILES)

    payload = tool.build_sync_package(repo, generated_at="2026-06-22T05:10:00+08:00")

    assert payload["schema_version"] == tool.SCHEMA_VERSION
    assert payload["status"] == "DRYRUN_ONLY_NOT_SYNCABLE"
    assert payload["file_count"] == len(tool.C1_SYNC_FILES)
    assert payload["execution_unlock"]["remote_sync_allowed"] is False
    assert payload["execution_unlock"]["slurm_allowed"] is False
    assert payload["execution_unlock"]["tools_test_allowed"] is False
    assert all(len(item["sha256"]) == 64 for item in payload["files"])


def test_build_sync_package_rejects_missing_file(tmp_path):
    tool = _load_tool()
    repo = _repo_with_files(tmp_path, tool.C1_SYNC_FILES[:-1])

    with pytest.raises(tool.SyncPackageError, match="missing sync file"):
        tool.build_sync_package(repo)


def test_build_sync_package_rejects_forbidden_artifacts(tmp_path):
    tool = _load_tool()
    repo = _repo_with_files(tmp_path, ["opentad/models/detectors/actionformer.py", "checkpoint/epoch_59.pth"])

    with pytest.raises(tool.SyncPackageError, match="forbidden artifact"):
        tool.build_sync_package(repo, files=("opentad/models/detectors/actionformer.py", "checkpoint/epoch_59.pth"))


def test_build_sync_package_rejects_path_escape(tmp_path):
    tool = _load_tool()
    repo = _repo_with_files(tmp_path, ["opentad/models/detectors/actionformer.py"])

    with pytest.raises(tool.SyncPackageError, match="escape repo"):
        tool.build_sync_package(repo, files=("../outside.py",))
