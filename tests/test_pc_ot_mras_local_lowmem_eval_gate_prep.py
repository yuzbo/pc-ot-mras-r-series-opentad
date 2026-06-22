import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "bata" / "prepare_pc_ot_mras_local_lowmem_eval_gate.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("pcot_local_lowmem_eval_gate_prep", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeConfig:
    def __init__(self, path):
        self.path = Path(path)
        self.pretty_text = f"resolved_config_from = {self.path.name!r}\n"

    @classmethod
    def fromfile(cls, path):
        return cls(path)

    def merge_from_dict(self, options):
        for key, value in sorted(options.items()):
            self.pretty_text += f"{key} = {value!r}\n"


def _write_fake_repo(tmp_path):
    tool = _load_tool()
    repo = tmp_path / "repo"
    config_dir = repo / "configs" / "adatad" / "thumos"
    config_dir.mkdir(parents=True)
    base = config_dir / "base.py"
    config = config_dir / "ctf_bdi_pc_ot_mras_r17_reader_disabled_eval_local_lowmem_candidate.py"
    base.write_text("model = dict(type='base')\n", encoding="utf-8")
    config.write_text("_base_ = ['base.py']\nwork_dir = 'unused'\n", encoding="utf-8")

    for rel_path in tool.DEFAULT_MANIFEST_PATHS:
        path = repo / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {rel_path}\n", encoding="utf-8")

    checkpoint = tmp_path / "epoch_59.pth"
    checkpoint.write_bytes(b"checkpoint bytes")
    return repo, config, base, checkpoint


def test_sha256_file_matches_hashlib(tmp_path):
    tool = _load_tool()
    path = tmp_path / "sample.bin"
    data = b"pc-ot-mras-local-gate"
    path.write_bytes(data)

    assert tool.sha256_file(path) == hashlib.sha256(data).hexdigest()


def test_prepare_gate_payload_env_command_and_active_manifest(tmp_path):
    tool = _load_tool()
    repo, config, base, checkpoint = _write_fake_repo(tmp_path)
    run_dir = tmp_path / "runs" / "r17_c1_local"

    result = tool.prepare_local_lowmem_eval_gate(
        repo=repo,
        config=config,
        checkpoint=checkpoint,
        run_dir=run_dir,
        decision="ALLOW_R17_READER_DISABLED_EVAL",
        cfg_options=("dataset.test.data_path=/home/skywalker/thumos14/test", "solver.test.batch_size=1"),
        wsl_log_root="/home/skywalker/tad_local_runs",
        generated_at="2026-06-22T12:00:00+08:00",
        config_cls=FakeConfig,
    )

    assert result["status"] == "PREPARED_NOT_LAUNCHED"
    assert result["launch_executed"] is False
    assert result["default_prepare_only"] is True

    outputs = result["outputs"]
    gate = json.loads(Path(outputs["gate_json"]).read_text(encoding="utf-8"))
    assert gate["schema_version"] == "pc_ot_mras_local_lowmem_eval_gate_v0"
    assert gate["decision"] == "ALLOW_R17_READER_DISABLED_EVAL"
    assert gate["local_lowmem_eval"] is True
    assert gate["tools_test"] is True
    assert gate["tools_train"] is False
    assert gate["direct_tools_train"] is False
    assert gate["raw_prediction_cache"] is False
    assert gate["paper_claim"] is False
    assert gate["runtime_flops_claim"] is False
    assert gate["deploy_claim"] is False
    assert gate["dynamic_budget_claim"] is False
    assert gate["checkpoint_sha256"] == tool.sha256_file(checkpoint)
    assert gate["checkpoint_size_bytes"] == checkpoint.stat().st_size
    assert gate["resolved_config_sha256"] == tool.sha256_file(outputs["resolved_config"])
    assert gate["active_sha256_manifest_sha256"] == tool.sha256_file(outputs["active_sha256_manifest"])
    assert gate["cfg_options"]["solver.test.batch_size"] == 1

    env_text = Path(outputs["env_file"]).read_text(encoding="utf-8")
    assert "OPENTAD_PCOTMRAS_ENTRYPOINT_GATE_JSON" in env_text
    assert "OPENTAD_PCOTMRAS_ENTRYPOINT_GATE_SHA256" in env_text
    assert "OPENTAD_PCOTMRAS_ACTIVE_MANIFEST_SHA256" in env_text
    assert "OPENTAD_PCOTMRAS_RESOLVED_CONFIG_SHA256" in env_text

    run_command = Path(outputs["suggested_wsl_run_command"]).read_text(encoding="utf-8")
    assert "torchrun --nproc_per_node=1" in run_command
    assert "tools/test.py" in run_command
    assert tool._to_wsl_path(checkpoint) in run_command
    assert "/home/skywalker/tad_local_runs/r17_c1_local/eval_stdout.log" in run_command
    assert "| tee /home/skywalker/tad_local_runs/r17_c1_local/eval_stdout.log" in run_command

    manifest_text = Path(outputs["active_sha256_manifest"]).read_text(encoding="utf-8")
    required_paths = {
        "tools/test.py",
        "opentad/utils/training_guard.py",
        "opentad/models/detectors/actionformer.py",
        "opentad/models/selectors/pc_ot_mras_reader.py",
        "opentad/models/necks/pc_ot_mras_detector_bridge.py",
        "opentad/models/dense_heads/native_irregular_area_head_p2.py",
        "opentad/models/utils/temporal_grid.py",
        "configs/adatad/thumos/ctf_bdi_pc_ot_mras_r17_reader_disabled_eval_local_lowmem_candidate.py",
        "configs/adatad/thumos/base.py",
    }
    for rel_path in required_paths:
        assert rel_path in manifest_text
    assert str(checkpoint).replace("\\", "/") in manifest_text

    manifest_paths = {item["path"] for item in result["manifest_entries"]}
    assert "tools/test.py" in manifest_paths
    assert "opentad/models/utils/temporal_grid.py" in manifest_paths
    assert str(checkpoint).replace("\\", "/") in manifest_paths
    assert base.relative_to(repo).as_posix() in manifest_paths


def test_prepare_fails_with_structured_json_for_missing_file(tmp_path):
    tool = _load_tool()
    repo, config, _base, checkpoint = _write_fake_repo(tmp_path)
    checkpoint.unlink()

    with pytest.raises(tool.LocalLowmemEvalGateError) as exc_info:
        tool.prepare_local_lowmem_eval_gate(
            repo=repo,
            config=config,
            checkpoint=checkpoint,
            run_dir=tmp_path / "run",
            decision="ALLOW_R17_READER_DISABLED_EVAL",
            config_cls=FakeConfig,
        )

    payload = exc_info.value.to_payload()
    assert payload["status"] == "FAILED"
    assert payload["error"]["code"] == "MISSING_FILE"
    assert "checkpoint" in payload["error"]["details"]["label"]


def test_missing_mmengine_error_is_structured_json(monkeypatch, tmp_path):
    tool = _load_tool()
    config = tmp_path / "config.py"
    config.write_text("model = dict(type='x')\n", encoding="utf-8")

    def raise_missing():
        raise tool.LocalLowmemEvalGateError(
            "MISSING_MMENGINE",
            "mmengine is required",
            details={"missing_package": "mmengine"},
        )

    monkeypatch.setattr(tool, "_load_mmengine_config_cls", raise_missing)

    with pytest.raises(tool.LocalLowmemEvalGateError) as exc_info:
        tool._resolved_config_text(config, ())

    payload = exc_info.value.to_payload()
    assert payload["status"] == "FAILED"
    assert payload["error"]["code"] == "MISSING_MMENGINE"
    assert payload["error"]["details"]["missing_package"] == "mmengine"


def test_tool_does_not_import_torch():
    text = TOOL.read_text(encoding="utf-8")
    assert "import torch" not in text
    assert "from torch" not in text
