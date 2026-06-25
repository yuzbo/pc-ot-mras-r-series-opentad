import json
import math
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

from tools.bata.analyze_pc_ot_mras_selector_posttrain_diagnostics import (
    NOT_ATTRIBUTION_READY,
    analyze_selector_payload,
)


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "bata" / "analyze_pc_ot_mras_selector_posttrain_diagnostics.py"


def test_selector_posttrain_diagnostics_aggregates_selection_roles_metadata_and_nonfinite():
    payload = {
        "samples": [
            {
                "sample_id": "video_a",
                "selected_dense_indices": [0, 2, 4, 4, 7],
                "valid_len": 10,
                "gt_segments": [[2, 8]],
                "boundary_radius": 1,
                "role_ids": [0, 1, 2, 3, 4],
                "selector_scores": [0.05, 0.10, 0.75, 0.25, 0.40, 0.20, 0.15, 0.95, 0.80, 0.01],
                "pc_ot_mras_prebackbone_raw_slot_dense_indices": [0, 2, 2, 4, 4],
                "pc_ot_mras_prebackbone_reader_fill_count": 2,
                "pc_ot_mras_prebackbone_st_active_row_count": 3,
                "irregular_native_axis": True,
                "physical_grid_actionformer": True,
                "irregular_selected_positions": [0, 2, 4, 4, 7],
                "irregular_dense_valid_len": 10,
                "irregular_selected_valid_len": 10,
                "debug_nan": float("nan"),
            },
            {
                "sample_id": "video_b",
                "selected_dense_indices": [1, 5, 9],
                "valid_len": 12,
                "gt_segments": [[4, 10]],
                "boundary_radius": 1,
                "packet_roles": ["boundary", "interior", "background"],
                "selector_scores": [0.1, 0.8, 0.2, 0.3, 0.4, 0.9, 0.2, 0.1, 0.3, 0.7, 0.5, 0.05],
                "pc_ot_mras_prebackbone_raw_slot_dense_indices": [1, 5, 9],
                "pc_ot_mras_prebackbone_reader_fill_count": 0,
                "pc_ot_mras_prebackbone_st_active_row_count": 3,
                "irregular_native_axis": False,
                "physical_grid_actionformer": True,
                "irregular_selected_positions": [1, 5, 9],
                "irregular_dense_valid_len": 12,
                "irregular_selected_valid_len": 12,
                "debug_inf": float("inf"),
            },
        ]
    }

    summary = analyze_selector_payload(payload)

    assert summary["decision"] == "PC_OT_MRAS_SELECTOR_POSTTRAIN_DIAGNOSTIC_READY_WITH_WARNINGS"
    assert summary["sample_count"] == 2
    assert summary["non_finite"]["count"] == 2
    assert summary["aggregate"]["selected_count_distribution"] == {"3": 1, "5": 1}
    assert summary["aggregate"]["dynamic_budget"]["budget_distribution"] == {"3": 1, "5": 1}
    assert summary["aggregate"]["dynamic_budget"]["budget_mean"] == pytest.approx(4.0)
    assert summary["aggregate"]["duplicate_rate_mean"] == pytest.approx(0.1)
    assert summary["aggregate"]["boundary"]["near_selected_rate"] == pytest.approx(4 / 8)
    assert summary["aggregate"]["packet_roles"]["boundary_ratio"] == pytest.approx(3 / 8)
    assert summary["aggregate"]["packet_roles"]["interior_ratio"] == pytest.approx(2 / 8)
    assert summary["aggregate"]["slot_transport"]["samples_with_raw_slot_duplicate"] == 2
    assert summary["aggregate"]["slot_transport"]["raw_slot_duplicate_rate_mean"] == pytest.approx(0.2)
    assert summary["aggregate"]["slot_transport"]["reader_fill_count_mean"] == pytest.approx(1.0)
    assert summary["aggregate"]["slot_transport"]["st_active_row_count_mean"] == pytest.approx(3.0)
    assert summary["aggregate"]["metadata_consistency"]["inconsistent_sample_count"] == 1
    assert summary["samples"][0]["gap"]["max"] == 3
    assert summary["samples"][0]["duplicate_rate"] == pytest.approx(0.2)
    assert summary["samples"][0]["slot_transport"]["raw_slot_duplicate_rate"] == pytest.approx(0.4)
    assert summary["samples"][0]["slot_transport"]["reader_fill_count"] == 2
    assert summary["samples"][0]["slot_transport"]["st_active_row_count"] == 3
    assert summary["samples"][0]["score_rank"]["available"] is True
    assert summary["samples"][1]["metadata_consistency"]["consistent"] is False


def test_selector_posttrain_diagnostics_jsonl_cli_and_synthetic_smoke(tmp_path):
    input_jsonl = tmp_path / "selector_rows.jsonl"
    output_json = tmp_path / "summary.json"
    rows = [
        {
            "sample_id": "jsonl_a",
            "selected_dense_indices": [0, 3, 6],
            "valid_len": 9,
            "gt_segments": [[2, 7]],
            "packet_roles": ["coverage", "boundary", "interior"],
            "irregular_native_axis": True,
            "physical_grid_actionformer": False,
            "irregular_selected_positions": [0, 3, 6],
            "irregular_dense_valid_len": 9,
            "irregular_selected_valid_len": 9,
        },
        {
            "sample_id": "jsonl_b",
            "selected_dense_indices": [1, 2, 4, 8],
            "valid_len": 10,
            "packet_roles": ["boundary", "boundary", "interior", "background"],
            "irregular_native_axis": True,
            "physical_grid_actionformer": True,
            "irregular_selected_positions": [1, 2, 4, 8],
            "irregular_dense_valid_len": 10,
            "irregular_selected_valid_len": 10,
        },
    ]
    input_jsonl.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--input",
            str(input_jsonl),
            "--output",
            str(output_json),
            "--boundary-radius",
            "1",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    stdout_payload = json.loads(proc.stdout)
    file_payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert stdout_payload == file_payload
    assert stdout_payload["input"]["format"] == "jsonl"
    assert stdout_payload["sample_count"] == 2
    assert stdout_payload["protocol"]["tools_train_allowed"] is False
    assert stdout_payload["protocol"]["remote_sync_allowed"] is False

    smoke = subprocess.run(
        [sys.executable, str(TOOL), "--synthetic-smoke"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        check=False,
    )

    assert smoke.returncode == 0, smoke.stderr + smoke.stdout
    smoke_payload = json.loads(smoke.stdout)
    assert smoke_payload["input"]["format"] == "synthetic"
    assert smoke_payload["synthetic_smoke"] is True
    assert smoke_payload["sample_count"] >= 2
    assert smoke_payload["protocol"]["diagnostic_only"] is True


def test_selector_posttrain_diagnostics_pickle_loader_cli(tmp_path):
    input_pickle = tmp_path / "selector_rows.pkl"
    output_json = tmp_path / "pickle_summary.json"
    with input_pickle.open("wb") as f:
        pickle.dump(
            {
                "samples": [
                    {
                        "sample_id": "pickle_a",
                        "selected_dense_indices": [0, 2, 5],
                        "valid_len": 8,
                        "packet_roles": ["coverage", "boundary", "interior"],
                        "irregular_native_axis": True,
                        "physical_grid_actionformer": True,
                        "irregular_selected_positions": [0, 2, 5],
                        "irregular_dense_valid_len": 8,
                        "irregular_selected_valid_len": 8,
                    }
                ]
            },
            f,
        )

    proc = subprocess.run(
        [sys.executable, str(TOOL), "--input", str(input_pickle), "--output", str(output_json)],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["input"]["format"] == "pickle"
    assert payload["sample_count"] == 1
    assert payload["samples"][0]["sample_id"] == "pickle_a"


def test_selector_posttrain_diagnostics_npz_dynamic_plan_loader_cli(tmp_path):
    np = pytest.importorskip("numpy")
    input_npz = tmp_path / "dynamic_plan.npz"
    np.savez(
        input_npz,
        budgets=np.asarray([2]),
        dense_valid_len=np.asarray([4]),
        selected_dense_positions=np.asarray([[0, 2]]),
        selected_mask=np.asarray([[True, True]]),
    )

    proc = subprocess.run(
        [sys.executable, str(TOOL), "--input", str(input_npz)],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr + proc.stdout
    payload = json.loads(proc.stdout)
    assert payload["input"]["format"] == "npz"
    assert payload["sample_count"] == 1
    assert payload["aggregate"]["dynamic_budget"]["budget_distribution"] == {"2": 1}


def test_selector_posttrain_diagnostics_reports_cap_phase_and_epoch_coverage():
    payload = {
        "samples": [
            {
                "sample_id": "train_early_a",
                "phase": "train",
                "epoch": 0,
                "iter": 3,
                "selected_dense_indices": [0, 2, 4],
                "valid_len": 8,
                "gt_segments": [[1, 5]],
                "selector_scores": [0.1, 0.8, 0.2, 0.6, 0.4, 0.3, 0.2, 0.1],
                "pc_ot_mras_prebackbone_raw_slot_dense_indices": [0, 2, 2],
                "pc_ot_mras_prebackbone_reader_fill_count": 1,
                "pc_ot_mras_prebackbone_st_active_row_count": 2,
                "irregular_selected_positions": [0, 2, 4],
                "irregular_dense_valid_len": 8,
                "irregular_selected_valid_len": 8,
            },
            {
                "sample_id": "train_early_b",
                "phase": "train",
                "epoch": 0,
                "iter": 4,
                "selected_dense_indices": [1, 3, 5],
                "valid_len": 8,
                "gt_segments": [[2, 6]],
                "selector_scores": [0.1, 0.8, 0.2, 0.6, 0.4, 0.3, 0.2, 0.1],
                "pc_ot_mras_prebackbone_raw_slot_dense_indices": [1, 3, 5],
                "pc_ot_mras_prebackbone_reader_fill_count": 0,
                "pc_ot_mras_prebackbone_st_active_row_count": 3,
                "irregular_selected_positions": [1, 3, 5],
                "irregular_dense_valid_len": 8,
                "irregular_selected_valid_len": 8,
            },
        ]
    }

    summary = analyze_selector_payload(
        payload,
        row_cap=2,
        require_train_phase=True,
        require_validation_phase=True,
        min_late_epoch=40,
    )

    coverage = summary["metadata_coverage"]
    assert coverage["row_cap"]["configured"] == 2
    assert coverage["row_cap"]["hit_or_exceeded"] is True
    assert coverage["phase"]["counts"] == {"train": 2}
    assert coverage["phase"]["has_train"] is True
    assert coverage["phase"]["has_validation"] is False
    assert coverage["epoch"]["known_count"] == 2
    assert coverage["epoch"]["max"] == 0
    assert coverage["iter"]["known_count"] == 2
    assert summary["attribution_readiness"]["status"] == NOT_ATTRIBUTION_READY
    assert any("validation selector rows" in item for item in summary["attribution_readiness"]["missing"])
    assert any("row cap appears to have truncated" in item for item in summary["attribution_readiness"]["missing"])
