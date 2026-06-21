import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "bata" / "verify_pc_ot_mras_trainability_manifest.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("pcot_trainability_manifest_verifier", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _manifest(tmp_path, *, missing_stage=None, unfinished=False):
    stages = ["R16A", "R16B"] + [f"R{i}" for i in range(17, 36)]
    if missing_stage:
        stages.remove(missing_stage)
    models = []
    for stage in stages:
        full = stage in {"R17", "R18"}
        status = "completed_job" if full and not unfinished else "not_started"
        models.append(
            {
                "stage": stage,
                "experiment_model": stage,
                "changed_surface": [],
                "innovation_purpose": "test",
                "implementation_summary": "test",
                "full_training_worthy_now": full,
                "remote_sync_status": "test",
                "training_status": status,
                "next_gate": "test",
            }
        )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"models": models}), encoding="utf-8")
    return path


def _repo(tmp_path, *, extra_formal_launcher=False):
    repo = tmp_path / "repo"
    config_dir = repo / "configs" / "adatad" / "thumos"
    script_dir = repo / "scripts"
    tool_dir = repo / "tools" / "bata"
    test_dir = repo / "tests"
    for directory in (config_dir, script_dir, tool_dir, test_dir):
        directory.mkdir(parents=True, exist_ok=True)

    for name in (
        "ctf_bdi_pc_ot_mras_r12_p2_optin_adapter_local.py",
        "ctf_bdi_pc_ot_mras_r16_gpu_smoke_candidate.py",
        "ctf_bdi_pc_ot_mras_r17_formal_train_candidate.py",
        "ctf_bdi_pc_ot_mras_r18_aux_formal_train_candidate.py",
        "ctf_bdi_pc_ot_mras_r35_future_placeholder.py",
        "ctf_bdi_pc_ot_mras_synthetic_local.py",
    ):
        (config_dir / name).write_text("# test\n", encoding="utf-8")

    for name in (
        "run_ctf_bdi_pc_ot_mras_r16a_gpu_smoke_n16r4.sbatch",
        "run_ctf_bdi_pc_ot_mras_r17_formal_train_n16r4.sbatch",
        "run_ctf_bdi_pc_ot_mras_r18_aux_formal_train_n16r4.sbatch",
        "run_ctf_bdi_pc_ot_mras_r20_value_precheck_n16r4.sbatch",
        "run_ctf_bdi_pc_ot_mras_reader_disabled_eval_n16r4.sbatch",
    ):
        (script_dir / name).write_text("#!/bin/bash\n", encoding="utf-8")
    if extra_formal_launcher:
        (script_dir / "run_ctf_bdi_pc_ot_mras_r40_formal_train_n16r4.sbatch").write_text(
            "#!/bin/bash\n", encoding="utf-8"
        )
    return repo


def test_verify_trainability_manifest_accepts_current_completed_formal_rows(tmp_path):
    tool = _load_tool()
    result = tool.verify_trainability_manifest(_manifest(tmp_path), _repo(tmp_path))

    assert result["pass"] is True
    assert result["missing_required_stages"] == []
    assert result["full_training_worthy_now"] == ["R17", "R18"]
    assert result["unfinished_full_training_worthy"] == []
    assert result["formal_full_training_launcher_stages"] == ["R17", "R18"]
    assert "R12" in result["historical_or_local_config_stages_outside_manifest"]
    assert "SYNTHETIC_LOCAL" in result["historical_or_local_config_stages_outside_manifest"]


def test_verify_trainability_manifest_rejects_missing_current_stage(tmp_path):
    tool = _load_tool()
    result = tool.verify_trainability_manifest(_manifest(tmp_path, missing_stage="R33"), _repo(tmp_path))

    assert result["pass"] is False
    assert result["missing_required_stages"] == ["R33"]


def test_verify_trainability_manifest_rejects_unfinished_full_training_worthy(tmp_path):
    tool = _load_tool()
    result = tool.verify_trainability_manifest(_manifest(tmp_path, unfinished=True), _repo(tmp_path))

    assert result["pass"] is False
    assert result["unfinished_full_training_worthy"] == ["R17", "R18"]


def test_verify_trainability_manifest_rejects_unregistered_formal_launcher(tmp_path):
    tool = _load_tool()
    result = tool.verify_trainability_manifest(_manifest(tmp_path), _repo(tmp_path, extra_formal_launcher=True))

    assert result["pass"] is False
    assert result["unregistered_formal_launcher_stages"] == ["R40"]
