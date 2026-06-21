import json
from pathlib import Path

import pytest

from tools.bata.visualize_pc_ot_mras_selection import (
    READY,
    run_selection_visualization,
)


def test_selection_visualization_renders_reader_heatmap_and_summary(tmp_path):
    input_jsonl = tmp_path / "reader_rows.jsonl"
    output_dir = tmp_path / "figures"
    summary_json = tmp_path / "summary.json"
    row = {
        "sample_id": "video_test_000001",
        "snapshot_id": "epoch_002",
        "epoch": 2,
        "budget": 2,
        "uses_cache": False,
        "uses_gt": False,
        "reader_out": {
            "acquisition_matrix": [
                [
                    [0.00, 0.10, 0.00, 0.00, 0.00, 0.00],
                    [0.00, 0.00, 0.00, 0.90, 0.00, 0.00],
                    [0.00, 0.80, 0.00, 0.00, 0.00, 0.00],
                ]
            ],
            "allocation": [
                [
                    [0.00, 0.20, 0.00, 0.00, 0.00, 0.00],
                    [0.00, 0.00, 0.00, 0.70, 0.00, 0.00],
                    [0.00, 0.60, 0.00, 0.00, 0.00, 0.00],
                ]
            ],
            "valid_mask": [[1, 1, 1, 1, 1, 1]],
            "valid_lengths": [6],
            "start_logits": [[0.1, 0.4, 0.2, 0.8, 0.0, -0.1]],
            "boundary_logits": [[0.0, 0.9, 0.1, 0.7, 0.2, 0.1]],
            "redundancy_logits": [[0.8, 0.1, 0.7, 0.2, 0.9, 0.4]],
            "gates": [[0.2, 0.9, 0.7]],
            "centers": [[0.15, 0.55, 0.80]],
        },
    }
    input_jsonl.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")

    summary = run_selection_visualization(
        input_jsonl,
        output_dir=output_dir,
        summary_json=summary_json,
        budget=2,
        snapshot_label="epoch_002",
        max_time_bins=12,
        max_slot_bins=6,
    )

    assert summary["decision"] == READY
    assert summary["sample_count"] == 1
    assert summary["generated_svg_count"] == 1
    assert summary["uses_gt"] is False
    assert summary["metric_claim_allowed"] is False
    assert summary["per_sample"][0]["matrix_key"] == "acquisition_matrix"
    assert summary["per_sample"][0]["selected_count"] == 2
    assert summary["per_sample"][0]["selected_positions_preview"] == [1, 3]
    svg_path = Path(summary["per_sample"][0]["svg_path"])
    assert svg_path.is_file()
    svg_text = svg_path.read_text(encoding="utf-8")
    assert "acquisition_matrix" in svg_text
    assert "selected=2" in svg_text
    assert Path(summary_json).is_file()


def test_selection_visualization_resolves_batched_short_windows_independently(tmp_path):
    input_jsonl = tmp_path / "batched_reader_rows.jsonl"
    row = {
        "sample_ids": ["short_window", "long_window"],
        "snapshot_id": "epoch_043",
        "reader_out": {
            "acquisition_matrix": [
                [
                    [0.9, 0.0, 0.0, 0.0],
                    [0.0, 0.8, 0.0, 0.0],
                    [0.0, 0.0, 0.7, 0.0],
                ],
                [
                    [0.0, 0.8, 0.0, 0.0],
                    [0.0, 0.0, 0.7, 0.0],
                    [0.0, 0.0, 0.0, 0.6],
                ],
            ],
            "valid_mask": [
                [1, 1, 0, 0],
                [1, 1, 1, 1],
            ],
            "valid_lengths": [2, 4],
        },
    }
    input_jsonl.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")

    summary = run_selection_visualization(
        input_jsonl,
        output_dir=tmp_path / "figures",
        snapshot_label="epoch_043",
    )

    assert summary["decision"] == READY
    assert summary["sample_count"] == 2
    assert summary["per_sample"][0]["sample_id"] == "short_window"
    assert summary["per_sample"][0]["selected_count"] == 2
    assert summary["per_sample"][1]["sample_id"] == "long_window"
    assert summary["per_sample"][1]["selected_count"] == 3


def test_selection_visualization_renders_hard_position_rows_without_matrix(tmp_path):
    input_jsonl = tmp_path / "hard_rows.jsonl"
    row = {
        "schema_version": "pc_ot_mras_hard_positions_v0",
        "sample_id": "hard_sample",
        "snapshot_id": "epoch_010",
        "budget": 3,
        "dense_len": 8,
        "valid_len": 8,
        "selected_positions": [0, 3, 7],
        "selected_mask": [1, 0, 0, 1, 0, 0, 0, 1],
    }
    input_jsonl.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")

    summary = run_selection_visualization(
        input_jsonl,
        output_dir=tmp_path / "figures",
        snapshot_label="epoch_010",
    )

    assert summary["decision"] == READY
    assert summary["per_sample"][0]["matrix_key"] is None
    assert summary["per_sample"][0]["selected_source"] == "hard_row_selected_positions"
    assert summary["per_sample"][0]["selected_positions_preview"] == [0, 3, 7]
    assert Path(summary["per_sample"][0]["svg_path"]).is_file()


def test_selection_visualization_rejects_gt_or_teacher_payloads(tmp_path):
    input_jsonl = tmp_path / "bad_rows.jsonl"
    row = {
        "sample_id": "bad",
        "reader_out": {
            "acquisition_matrix": [[[0.0, 0.9], [0.8, 0.0]]],
            "valid_mask": [[1, 1]],
        },
        "metadata": {"gt_segments": [[0.1, 0.2]]},
    }
    input_jsonl.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="forbidden diagnostic input key"):
        run_selection_visualization(
            input_jsonl,
            output_dir=tmp_path / "figures",
            budget=1,
        )


def test_selection_visualization_rejects_true_forbidden_guard_flags(tmp_path):
    input_jsonl = tmp_path / "bad_guard_rows.jsonl"
    row = {
        "sample_id": "bad_guard",
        "uses_cache": True,
        "reader_out": {
            "acquisition_matrix": [[[0.0, 0.9], [0.8, 0.0]]],
            "valid_mask": [[1, 1]],
        },
    }
    input_jsonl.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="forbidden diagnostic input key"):
        run_selection_visualization(
            input_jsonl,
            output_dir=tmp_path / "figures",
            budget=1,
        )
