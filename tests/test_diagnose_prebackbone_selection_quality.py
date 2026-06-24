import json
import subprocess
import sys
from pathlib import Path

from tools.bata.diagnose_prebackbone_selection_quality import (
    diagnose_selection_quality,
)


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "bata" / "diagnose_prebackbone_selection_quality.py"


def test_uniform_indices_report_clean_gap_and_no_duplicates():
    summary = diagnose_selection_quality(
        {
            "selected_dense_indices": [0, 2, 4, 6],
            "valid_len": 8,
        }
    )

    assert summary["selected_count"] == 4
    assert summary["valid_len"] == 8
    assert summary["selected_fraction"] == 0.5
    assert summary["monotonic"] is True
    assert summary["duplicate_rate"] == 0.0
    assert summary["gap"]["mean"] == 2.0
    assert summary["gap"]["max"] == 2
    assert summary["gap"]["p95"] == 2
    assert summary["gt_diagnostics"]["provided"] is False
    assert summary["proposal_diagnostics"]["provided"] is False
    assert summary["protocol"]["gt_runtime_allowed"] is False


def test_duplicate_indices_report_duplicate_rate_and_non_monotonic_order():
    summary = diagnose_selection_quality(
        {
            "selected_dense_indices": [0, 3, 3, 2],
            "valid_len": 8,
        }
    )

    assert summary["selected_count"] == 4
    assert summary["unique_selected_count"] == 3
    assert summary["monotonic"] is False
    assert summary["duplicate_rate"] == 0.25
    assert summary["gap"]["max"] == 3


def test_gt_segments_report_boundary_support_at_two():
    summary = diagnose_selection_quality(
        {
            "selected_dense_indices": [8, 10, 12, 30],
            "valid_len": 40,
            "gt_segments": [[9, 20]],
            "boundary_radius": 2,
        }
    )

    assert summary["gt_diagnostics"]["provided"] is True
    assert summary["gt_diagnostics"]["boundary_radius"] == 2
    assert summary["gt_diagnostics"]["boundary_count"] == 2
    assert summary["gt_diagnostics"]["boundary_support_at_radius"] == 1
    assert summary["gt_diagnostics"]["boundary_support_rate"] == 0.5
    assert summary["gt_diagnostics"]["boundary_near_rate"] == 0.5
    assert summary["protocol"]["gt_runtime_allowed"] is False
    assert "diagnostic input only" in summary["protocol"]["gt_usage_note"]


def test_proposals_report_score_iou_correlation_topk_iou_and_positive_ranks():
    summary = diagnose_selection_quality(
        {
            "selected_dense_indices": [0, 1, 2],
            "valid_len": 6,
            "proposals": [
                {"score": 0.9, "iou": 0.1},
                {"score": 0.8, "iou": 0.7},
                {"score": 0.7, "iou": 0.6},
                {"score": 0.1, "iou": 0.9},
            ],
            "top_k": 2,
            "positive_iou_threshold": 0.5,
        }
    )

    proposal = summary["proposal_diagnostics"]
    assert proposal["provided"] is True
    assert round(proposal["score_iou_correlation"], 6) == -0.756223
    assert proposal["top_k"] == 2
    assert proposal["top_k_mean_iou"] == 0.4
    assert proposal["positive_count"] == 3
    assert proposal["positive_rank_min"] == 2
    assert proposal["positive_rank_mean"] == 3.0


def test_json_cli_smoke_reads_input_and_writes_output(tmp_path):
    input_json = tmp_path / "selection.json"
    output_json = tmp_path / "diagnosis.json"
    input_json.write_text(
        json.dumps(
            {
                "sample_id": "video_test_000001",
                "selected_dense_indices": [0, 5, 10],
                "valid_len": 12,
            }
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--input",
            str(input_json),
            "--output",
            str(output_json),
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
    assert stdout_payload["sample_id"] == "video_test_000001"
    assert stdout_payload["selected_count"] == 3
    assert stdout_payload["valid_len"] == 12
