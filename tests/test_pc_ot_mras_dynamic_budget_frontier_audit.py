import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


torch_probe = subprocess.run(
    [sys.executable, "-c", "import torch"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    timeout=30,
    check=False,
)
if torch_probe.returncode != 0:
    pytest.skip("torch unavailable", allow_module_level=True)

import torch

from pc_ot_mras_test_utils import load_pc_ot_mras_classes
from tools.bata.audit_pc_ot_mras_dynamic_budget_frontier import (
    NO_GO,
    READY,
    audit_pc_ot_mras_dynamic_budget_frontier,
    run_json_frontier_audit,
)


ROOT = Path(__file__).resolve().parents[1]
load_pc_ot_mras_classes()


def _load_controller_class():
    path = ROOT / "opentad" / "models" / "selectors" / "pc_ot_mras_dynamic_budget_controller.py"
    spec = importlib.util.spec_from_file_location("opentad.models.selectors.pc_ot_mras_dynamic_budget_controller", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.PCOTMRASDynamicBudgetController


PCOTMRASDynamicBudgetController = _load_controller_class()


def _reader_outputs():
    valid = torch.zeros(3, 12, dtype=torch.bool)
    valid[0, :12] = True
    valid[1, :10] = True
    valid[2, :8] = True

    value_logits = torch.full((3, 12), -6.0)
    value_logits[0, :12] = 6.0
    value_logits[1, :10] = 0.0
    value_logits = value_logits.masked_fill(~valid, -20.0)

    risk_logits = torch.full((3, 12), -6.0).masked_fill(~valid, -20.0)
    redundancy_logits = torch.full((3, 12), -6.0).masked_fill(~valid, -20.0)
    acquisition = torch.zeros(3, 2, 12)
    for batch_idx, valid_len in enumerate(valid.long().sum(dim=1).tolist()):
        acquisition[batch_idx, :, :valid_len] = 1.0 / float(valid_len)
    return dict(
        valid_mask=valid,
        value_logits=value_logits,
        risk_logits=risk_logits,
        redundancy_logits=redundancy_logits,
        acquisition_matrix=acquisition,
    )


def _dynamic_plan():
    controller = PCOTMRASDynamicBudgetController(
        budget_values=(4, 6, 8),
        budget_thresholds=(0.25, 0.75),
        transport_weight=0.0,
        coverage_share=0.25,
        max_coverage_share=0.50,
    )
    return controller(_reader_outputs())


def _jsonable_plan(plan):
    out = {}
    for key, value in plan.items():
        if torch.is_tensor(value):
            out[key] = value.detach().cpu().tolist()
        else:
            out[key] = value
    return out


def test_dynamic_budget_frontier_audit_reports_budget_sensitive_frontier():
    summary = audit_pc_ot_mras_dynamic_budget_frontier(
        _dynamic_plan(),
        sample_ids=["hard|0", "medium|1", "easy|2"],
        difficulty_labels=["hard", "medium", "easy"],
        difficulty_scores=[0.9, 0.5, 0.1],
        fixed_budget_reference=8,
        reference_budgets=(4, 6, 8),
    )

    assert summary["decision"] == READY
    assert summary["budget_distribution"] == {"4": 1, "6": 1, "8": 1}
    assert summary["budget_mean"] == pytest.approx(6.0)
    assert summary["savings_vs_fixed_reference"] == pytest.approx(0.25)
    assert summary["coverage_cap_violations"] == 0
    assert summary["short_valid_len_clipped_count"] == 0
    assert summary["budget_score_monotonicity"]["passed"] is True
    assert summary["difficulty_score_monotonicity"]["passed"] is True
    assert summary["difficulty_budget_order"]["passed"] is True
    assert summary["difficulty_budget_order"]["group_means"]["easy"] == pytest.approx(4.0)
    assert summary["difficulty_budget_order"]["group_means"]["hard"] == pytest.approx(8.0)
    assert summary["budget_sensitivity_passed"] is True
    assert summary["hard_protocol_passed"] is True
    assert summary["pipeline_summary"]["exact_budget_violations"] == 0
    assert summary["pipeline_summary"]["center_matches_rows"] is True
    assert summary["dynamic_budget_quality_validation"] is False
    assert summary["dynamic_budget_validation"] is False
    assert summary["scanner_quality_validation"] is False
    assert summary["metric_claim_allowed"] is False
    assert summary["paper_claim_allowed"] is False
    assert summary["remote_sync_allowed"] is False
    assert summary["slurm_gpu_allowed"] is False
    assert summary["detector_map_allowed"] is False


def test_dynamic_budget_frontier_audit_json_roundtrip(tmp_path):
    input_json = tmp_path / "plan.json"
    summary_json = tmp_path / "summary.json"
    input_json.write_text(json.dumps(_jsonable_plan(_dynamic_plan())), encoding="utf-8")

    summary = run_json_frontier_audit(
        input_json,
        summary_json=summary_json,
        sample_ids=["hard|0", "medium|1", "easy|2"],
        difficulty_labels=["hard", "medium", "easy"],
        fixed_budget_reference=8,
        reference_budgets=(4, 6, 8),
    )

    loaded = json.loads(summary_json.read_text(encoding="utf-8"))
    assert loaded["decision"] == READY
    assert loaded["budget_distribution"] == summary["budget_distribution"] == {"4": 1, "6": 1, "8": 1}
    assert loaded["pipeline_summary"]["exact_budget_violations"] == 0


def test_dynamic_budget_frontier_audit_rejects_forbidden_or_unknown_plan_payload():
    plan = _jsonable_plan(_dynamic_plan())
    plan["allow_gpu"] = True

    with pytest.raises(ValueError, match="dynamic budget plan"):
        audit_pc_ot_mras_dynamic_budget_frontier(plan, fixed_budget_reference=8)


def test_dynamic_budget_frontier_audit_marks_uniform_budget_as_no_go():
    plan = _jsonable_plan(_dynamic_plan())
    plan["budgets"] = [4, 4, 4]
    plan["selected_mask"] = [[True, True, True, True, False, False, False, False] for _ in range(3)]

    summary = audit_pc_ot_mras_dynamic_budget_frontier(
        plan,
        difficulty_labels=["hard", "medium", "easy"],
        fixed_budget_reference=8,
        reference_budgets=(4, 6, 8),
    )

    assert summary["decision"] == NO_GO
    assert summary["budget_min"] == summary["budget_max"] == 4
    assert summary["budget_sensitivity_passed"] is False
    assert summary["hard_protocol_passed"] is True
    assert summary["pipeline_summary"]["exact_budget_violations"] == 0


def test_dynamic_budget_frontier_audit_accounts_for_short_valid_len_clipping():
    plan = _jsonable_plan(_dynamic_plan())
    plan["budgets"][2] = 3
    plan["dense_valid_len"][2] = 3
    plan["selected_dense_positions"][2] = [0, 1, 2, 0, 0, 0, 0, 0]
    plan["selected_mask"][2] = [True, True, True, False, False, False, False, False]

    summary = audit_pc_ot_mras_dynamic_budget_frontier(
        plan,
        difficulty_labels=["hard", "medium", "easy"],
        fixed_budget_reference=8,
        reference_budgets=(4, 6, 8),
    )

    assert summary["decision"] == READY
    assert summary["budgets"] == [8, 6, 3]
    assert summary["short_valid_len_clipped_count"] == 1
    assert summary["reference_budget_frontier"][0]["clipped_sample_count"] == 1
    assert summary["pipeline_summary"]["exact_budget_violations"] == 0
