import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "bata" / "finalize_pc_ot_mras_reader_disabled_gate.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("pcot_reader_disabled_gate_finalizer", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _draft():
    return {
        "schema_version": "pc_ot_mras_reader_disabled_c1_gate_binding_draft_v0",
        "timestamp": "2026-06-22T04:42:21+08:00",
        "status": "DRAFT_ONLY_NOT_EXECUTABLE",
        "route": "CTF-BDI/PC-OT-MRAS",
        "execution_unlock": {
            "remote_sync_allowed": False,
            "slurm_allowed": False,
            "tools_test_allowed": False,
            "detector_map_claim_allowed": False,
            "metric_claim_allowed": False,
            "paper_claim_allowed": False,
        },
        "remote_read_only_evidence": {
            "result_detection_json_found": False,
        },
        "targets": {
            "R17": {
                "decision_if_formalized": "ALLOW_R17_READER_DISABLED_EVAL",
                "config": "configs/adatad/thumos/ctf_bdi_pc_ot_mras_r17_reader_disabled_eval_candidate.py",
                "checkpoint": "/remote/r17/epoch_59.pth",
                "checkpoint_sha256": "1" * 64,
                "checkpoint_size_bytes": 123,
                "training_over_evidence": "2026-06-22 03:48:32 Train INFO: Training Over...",
                "local_resolved_config_sha256_draft": "2" * 64,
                "local_active_manifest_sha256_draft": "3" * 64,
                "required_formal_gate_fields": {
                    "decision": "ALLOW_R17_READER_DISABLED_EVAL",
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
            },
            "R18": {
                "decision_if_formalized": "ALLOW_R18_READER_DISABLED_EVAL",
                "config": "configs/adatad/thumos/ctf_bdi_pc_ot_mras_r18_reader_disabled_eval_candidate.py",
                "checkpoint": "/remote/r18/epoch_59.pth",
                "checkpoint_sha256": "4" * 64,
                "checkpoint_size_bytes": 456,
                "training_over_evidence": "2026-06-22 00:40:53 Train INFO: Training Over...",
                "local_resolved_config_sha256_draft": "5" * 64,
                "local_active_manifest_sha256_draft": "6" * 64,
                "required_formal_gate_fields": {
                    "decision": "ALLOW_R18_READER_DISABLED_EVAL",
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
            },
        },
    }


def test_validate_gate_draft_accepts_non_executable_reader_disabled_draft():
    tool = _load_tool()
    assert tool.validate_gate_draft(_draft())["status"] == "DRAFT_ONLY_NOT_EXECUTABLE"


def test_validate_gate_draft_rejects_executable_or_claim_enabled_draft():
    tool = _load_tool()
    draft = _draft()
    draft["execution_unlock"]["tools_test_allowed"] = True
    with pytest.raises(tool.GateDraftError, match="tools_test_allowed"):
        tool.validate_gate_draft(draft)

    draft = _draft()
    draft["targets"]["R17"]["required_formal_gate_fields"]["paper_claim"] = True
    with pytest.raises(tool.GateDraftError, match="paper_claim"):
        tool.validate_gate_draft(draft)


def test_build_formal_gate_requires_review_and_remote_sync_evidence():
    tool = _load_tool()
    with pytest.raises(tool.GateDraftError, match="review evidence"):
        tool.build_formal_gate(
            _draft(),
            "R17",
            remote_active_manifest_sha256="7" * 64,
            remote_resolved_config_sha256="8" * 64,
            review_evidence=None,
            remote_sync_evidence="sync",
            generated_at="2026-06-22T05:00:00+08:00",
        )


def test_build_formal_gate_payload_is_tools_test_only_and_claim_locked():
    tool = _load_tool()
    payload = tool.build_formal_gate(
        _draft(),
        "R18",
        remote_active_manifest_sha256="7" * 64,
        remote_resolved_config_sha256="8" * 64,
        review_evidence="review-report.md",
        remote_sync_evidence="sync-report.md",
        generated_at="2026-06-22T05:00:00+08:00",
    )

    assert payload["schema_version"] == "pc_ot_mras_reader_disabled_c1_formal_gate_v0"
    assert payload["decision"] == "ALLOW_R18_READER_DISABLED_EVAL"
    assert payload["completed_training_evidence"] is True
    assert payload["reader_disabled_eval"] is True
    assert payload["same_head_control"] is True
    assert payload["exact_uniform_reader_override"] is True
    assert payload["tools_train"] is False
    assert payload["raw_prediction_cache"] is False
    assert payload["paper_claim"] is False
    assert payload["runtime_flops_claim"] is False
    assert payload["deploy_claim"] is False
    assert payload["dynamic_budget_claim"] is False


def test_cli_default_validates_draft_without_writing_formal_gate(tmp_path):
    draft_path = tmp_path / "draft.json"
    output_path = tmp_path / "formal_gate.json"
    draft_path.write_text(json.dumps(_draft()), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(TOOL), "--draft", str(draft_path), "--target", "R17"],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)
    assert payload["status"] == "DRAFT_VALID_NOT_EXECUTABLE"
    assert payload["formal_gate_written"] is False
    assert not output_path.exists()


def test_cli_refuses_to_write_gate_without_explicit_allow(tmp_path):
    draft_path = tmp_path / "draft.json"
    output_path = tmp_path / "formal_gate.json"
    draft_path.write_text(json.dumps(_draft()), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--draft",
            str(draft_path),
            "--target",
            "R18",
            "--write-gate-json",
            str(output_path),
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "--write-gate-json requires --allow-executable-gate" in (result.stderr + result.stdout)
    assert not output_path.exists()
