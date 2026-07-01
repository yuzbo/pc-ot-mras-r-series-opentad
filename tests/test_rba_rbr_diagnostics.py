import json
import subprocess
import sys

import pytest

from opentad.acquisition.rba_rbr.types import ROUTE_LABEL
from tools.rba_rbr.audit_coordinate_budget import (
    CLAIM_STATUS,
    PRO_GATE_LOCK,
    coordinate_closure_synthetic_check,
    run_audit,
    summarize_coordinate_budget_rows,
    synthetic_grid_rows,
)


def test_coordinate_budget_audit_flags_low_detector_density_on_synthetic_rows():
    result = run_audit(synthetic_grid_rows("low_density"))
    summary = result["summary"]

    assert summary["route_label"] == ROUTE_LABEL
    assert summary["row_count"] == 3
    assert summary["raw_valid_k"]["p50"] == 84.0
    assert summary["detector_valid_count"]["p50"] == 42.0
    assert summary["reference"]["raw_50pct_k"] == 192
    assert summary["reference"]["detector_50pct_k"] == 96
    assert summary["effective_detector_density_far_below_50pct_reference"] is True
    assert summary["effective_detector_density_vs_50pct_reference"] < 0.60
    assert "postprocess-guard shortdiag" in summary["diagnosis_hint"]
    assert result["coordinate_closure"]["passed"] is True


def test_coordinate_budget_audit_does_not_flag_healthy_50pct_synthetic_rows():
    result = run_audit(synthetic_grid_rows("healthy_50pct"))

    assert result["summary"]["effective_detector_density_far_below_50pct_reference"] is False
    assert result["summary"]["effective_detector_density_vs_50pct_reference"] >= 0.60
    assert result["claim_lock"]["claim_status"] == CLAIM_STATUS
    assert result["claim_lock"]["required_gate"] == PRO_GATE_LOCK
    assert result["claim_lock"]["pro_gate_waived"] is False


def test_coordinate_closure_synthetic_check_uses_native_axis_without_dataset_gt():
    closure = coordinate_closure_synthetic_check(feature_stride=2)

    assert closure["route_label"] == ROUTE_LABEL
    assert closure["passed"] is True
    assert closure["uses_validation_or_test_gt"] is False
    assert closure["segment_source"] == "synthetic_native_axis_only_not_dataset_gt"
    assert closure["centers_match_raw_groups"] is True
    assert closure["native_axis_not_selected_index"] is True
    assert closure["max_synthetic_boundary_raw_distance"] <= 1.0
    assert all(row["raw_hit_count"] > 0 for row in closure["segment_hits"])
    assert all(row["detector_center_hit_count"] > 0 for row in closure["segment_hits"])


def test_coordinate_budget_audit_rejects_wrong_route_label_and_c3_mixing():
    bad_rows = synthetic_grid_rows("low_density")
    bad_rows[0]["route_label"] = "DIVERGENT_INNOVATION_BVR_TWB_DO_NOT_MERGE_WITH_C3"
    with pytest.raises(ValueError, match="route_label"):
        summarize_coordinate_budget_rows(bad_rows)

    c3_rows = synthetic_grid_rows("low_density")
    c3_rows[0]["note"] = "do not borrow C3 shortcut evidence"
    with pytest.raises(ValueError, match="forbidden route token"):
        summarize_coordinate_budget_rows(c3_rows)


def test_coordinate_budget_audit_cli_reads_jsonl_and_preserves_claim_lock(tmp_path):
    audit_path = tmp_path / "rba_rbr_grid_audit.jsonl"
    with audit_path.open("w", encoding="utf-8") as handle:
        for row in synthetic_grid_rows("low_density"):
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    completed = subprocess.run(
        [
            sys.executable,
            "tools/rba_rbr/audit_coordinate_budget.py",
            "--grid-audit-jsonl",
            str(audit_path),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert completed.returncode == 0, completed.stderr
    output = json.loads(completed.stdout)
    assert output["route_label"] == ROUTE_LABEL
    assert output["summary"]["effective_detector_density_far_below_50pct_reference"] is True
    assert output["coordinate_closure"]["passed"] is True
    assert output["claim_lock"] == {
        "route_label": ROUTE_LABEL,
        "claim_status": CLAIM_STATUS,
        "required_gate": PRO_GATE_LOCK,
        "no_gpu": True,
        "no_training": True,
        "no_metric_claim": True,
        "no_runtime_claim": True,
        "no_deploy_claim": True,
        "no_paper_claim": True,
        "full_train_unlocked": False,
        "pro_gate_waived": False,
    }
