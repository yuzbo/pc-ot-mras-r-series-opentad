import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "bata" / "check_pc_ot_mras_c1_acquisition_preflight.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("pcot_c1_acquisition_preflight", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _draft():
    targets = {}
    for target, checkpoint_sha, config_sha, manifest_sha in (
        ("R17", "1" * 64, "2" * 64, "3" * 64),
        ("R18", "4" * 64, "5" * 64, "6" * 64),
    ):
        targets[target] = {
            "decision_if_formalized": f"ALLOW_{target}_READER_DISABLED_EVAL",
            "checkpoint": f"/remote/{target}/epoch_59.pth",
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_size_bytes": 123,
            "training_over_evidence": "Train INFO: Training Over...",
            "local_resolved_config_sha256_draft": config_sha,
            "local_active_manifest_sha256_draft": manifest_sha,
            "required_formal_gate_fields": {
                "decision": f"ALLOW_{target}_READER_DISABLED_EVAL",
                "completed_training_evidence": True,
                "reader_disabled_eval": True,
                "same_head_control": True,
                "exact_uniform_reader_override": True,
                "tools_train": False,
                "raw_prediction_cache": False,
                "paper_claim": False,
                "runtime_flops_claim": False,
                "deploy_claim": False,
                "dynamic_budget_claim": False,
            },
        }
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
        "targets": targets,
    }


def _fixture_workspace(tmp_path, *, stale_sync=False, unlock_tools_test=False, missing_snapshot=False):
    workspace = tmp_path / "w"
    repo = tmp_path / "r"
    final_figure_dir = "figs"
    file_text = "# source\n"
    source_paths = [
        "opentad/models/detectors/actionformer.py",
        "configs/adatad/thumos/ctf_bdi_pc_ot_mras_r17_reader_disabled_eval_candidate.py",
    ]
    for rel_path in source_paths:
        _write_text(repo / rel_path, file_text)

    expected_sha = "0" * 64 if stale_sync else _sha256_file(repo / source_paths[0])
    _write_json(
        workspace / "research-wiki/experiments/CTF_BDI_PC_OT_MRAS_C1_SYNC_PACKAGE_DRYRUN_20260622.json",
        {
            "status": "DRYRUN_ONLY_NOT_SYNCABLE",
            "pass": True,
            "file_count": len(source_paths),
            "execution_unlock": {
                "remote_sync_allowed": False,
                "slurm_allowed": False,
                "tools_test_allowed": False,
                "detector_map_reporting_allowed": False,
                "metric_claim_allowed": False,
                "paper_claim_allowed": False,
            },
            "files": [{"path": rel_path, "sha256": expected_sha} for rel_path in source_paths],
            "missing_before_remote_sync": ["required_review_or_explicit_replacement"],
            "non_unlocks": ["remote_sync", "tools_test"],
        },
    )
    _write_json(
        workspace / "research-wiki/experiments/CTF_BDI_PC_OT_MRAS_TRAINABILITY_MANIFEST_VERIFY_20260622.json",
        {
            "pass": True,
            "full_training_worthy_now": ["R17", "R18"],
            "unfinished_full_training_worthy": [],
            "current_execution_conclusion": "no_new_full_training_unlocked",
        },
    )
    _write_json(
        workspace / "research-wiki/experiments/CTF_BDI_PC_OT_MRAS_C1_EVIDENCE_ACQUISITION_UNLOCK_PACKET_20260622.json",
        {
            "status": "UNLOCK_PACKET_READY_NON_EXECUTABLE",
            "already_available": {key: True for key in _load_tool().REQUIRED_AVAILABLE_FLAGS},
            "execution_unlock": {
                "remote_sync_allowed": False,
                "slurm_allowed": False,
                "tools_test_allowed": unlock_tools_test,
                "detector_map_reporting_allowed": False,
                "metric_claim_allowed": False,
                "paper_claim_allowed": False,
            },
            "missing_before_execution": ["remote_sync_after_review"],
            "current_allowed_commands": ["dry_run_validate_r17_binding_draft"],
            "non_unlocks": ["slurm_submission", "metric_claim"],
        },
    )
    _write_json(
        workspace / "research-wiki/experiments/CTF_BDI_PC_OT_MRAS_READER_DISABLED_C1_GATE_BINDING_DRAFT_20260622.json",
        _draft(),
    )
    figure_dir = workspace / final_figure_dir
    _write_json(
        figure_dir
        / "pcotmras_selection_learning_final_epoch59_matched_compare_20260622"
        / "final_epoch59_exact_uniform_matched_comparison_summary.json",
        {"R17": {}, "R18": {}},
    )
    _write_text(
        figure_dir
        / "pcotmras_selection_learning_final_epoch59_matched_compare_20260622"
        / "final_epoch59_exact_uniform_matched_comparison_per_sample.csv",
        "target,sample\nR17,0\n",
    )
    for target in ("R17", "R18"):
        lower = target.lower()
        snapshot_dir = figure_dir / f"{target}_final_{lower}_epoch59_final_reader_snapshot"
        if not missing_snapshot:
            _write_text(snapshot_dir / "snapshot.jsonl", "{}\n")
        _write_text(snapshot_dir / "viz" / "sample0.svg", "<svg />\n")
    return workspace, repo, final_figure_dir


def test_c1_acquisition_preflight_accepts_local_ready_but_locked_state(tmp_path):
    tool = _load_tool()
    workspace, repo, final_figure_dir = _fixture_workspace(tmp_path)

    result = tool.build_c1_acquisition_preflight(
        workspace,
        repo,
        final_figure_dir=final_figure_dir,
        generated_at="2026-06-22T05:30:00+08:00",
    )

    assert result["schema_version"] == tool.SCHEMA_VERSION
    assert result["status"] == "LOCAL_PREFLIGHT_PASS_EXECUTION_LOCKED"
    assert result["local_preflight_pass"] is True
    assert result["execution_allowed"] is False
    assert result["remote_sync_allowed"] is False
    assert result["slurm_allowed"] is False
    assert result["tools_test_allowed"] is False
    assert "required_review_or_explicit_replacement" in result["missing_before_execution"]


def test_c1_acquisition_preflight_rejects_stale_sync_package(tmp_path):
    tool = _load_tool()
    workspace, repo, final_figure_dir = _fixture_workspace(tmp_path, stale_sync=True)

    result = tool.build_c1_acquisition_preflight(workspace, repo, final_figure_dir=final_figure_dir)

    assert result["status"] == "LOCAL_PREFLIGHT_FAILED"
    sync_check = next(item for item in result["checks"] if item["name"] == "c1_sync_package_dryrun_current")
    assert sync_check["pass"] is False
    assert sync_check["detail"]["mismatch_count"] == 2


def test_c1_acquisition_preflight_rejects_unlocked_tools_test_flag(tmp_path):
    tool = _load_tool()
    workspace, repo, final_figure_dir = _fixture_workspace(tmp_path, unlock_tools_test=True)

    result = tool.build_c1_acquisition_preflight(workspace, repo, final_figure_dir=final_figure_dir)

    assert result["status"] == "LOCAL_PREFLIGHT_FAILED"
    unlock_check = next(item for item in result["checks"] if item["name"] == "all_execution_unlocks_remain_false")
    assert unlock_check["pass"] is False
    assert result["tools_test_allowed"] is False


def test_c1_acquisition_preflight_rejects_missing_selector_snapshot(tmp_path):
    tool = _load_tool()
    workspace, repo, final_figure_dir = _fixture_workspace(tmp_path, missing_snapshot=True)

    result = tool.build_c1_acquisition_preflight(workspace, repo, final_figure_dir=final_figure_dir)

    assert result["status"] == "LOCAL_PREFLIGHT_FAILED"
    selector_check = next(item for item in result["checks"] if item["name"] == "selector_final_artifacts_available")
    assert selector_check["pass"] is False
    assert "R17_reader_snapshot_jsonl" in selector_check["detail"]["missing"]
