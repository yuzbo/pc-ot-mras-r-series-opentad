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
from tools.bata.audit_pc_ot_mras_synthetic_task_utility import (
    NO_GO,
    READY,
    TASK_SCHEMA_VERSION,
    audit_pc_ot_mras_synthetic_task_utility,
    run_json_synthetic_task_utility_audit,
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

    value_logits = torch.full((3, 12), -8.0)
    value_logits[0, [1, 2, 3, 7, 8, 9, 10, 11]] = 8.0
    value_logits[1, [1, 4, 5, 8, 9]] = 2.5
    value_logits[2, [0, 3]] = 0.5
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
        budget_thresholds=(0.25, 0.60),
        transport_weight=0.0,
        coverage_share=0.25,
        max_coverage_share=0.50,
    )
    return controller(_reader_outputs())


def _task_spec():
    return {
        "schema_version": TASK_SCHEMA_VERSION,
        "synthetic_task_utility_only": True,
        "uses_gt": False,
        "uses_teacher": False,
        "uses_raw_prediction": False,
        "uses_checkpoint": False,
        "windows": [
            {
                "sample_id": "hard|0",
                "start": 1,
                "end": 7,
                "interior_peaks": [3, 8],
                "background_positions": [0, 11],
                "difficulty": "hard",
            },
            {
                "sample_id": "medium|1",
                "start": 1,
                "end": 8,
                "interior_peaks": [4, 5],
                "background_positions": [0, 9],
                "difficulty": "medium",
            },
            {
                "sample_id": "easy|2",
                "start": 1,
                "end": 3,
                "interior_peaks": [2],
                "background_positions": [0],
                "difficulty": "easy",
            },
        ],
    }


def _jsonable_plan(plan):
    out = {}
    for key, value in plan.items():
        if torch.is_tensor(value):
            out[key] = value.detach().cpu().tolist()
        else:
            out[key] = value
    return out


def test_synthetic_task_utility_audit_passes_boundary_and_uniform_controls():
    summary = audit_pc_ot_mras_synthetic_task_utility(
        _dynamic_plan(),
        _task_spec(),
        sample_ids=["hard|0", "medium|1", "easy|2"],
        fixed_budget_reference=8,
        reference_budgets=(4, 6, 8),
    )

    assert summary["decision"] == READY
    assert summary["budgets"] == [8, 6, 4]
    assert summary["boundary_support_mean"] == pytest.approx(1.0)
    assert summary["zero_boundary_support_rate"] == pytest.approx(0.0)
    assert summary["interior_peak_recall_mean"] == pytest.approx(1.0)
    assert summary["background_selected_share_mean"] <= 0.35
    assert summary["selected_vs_oracle_topk_ratio_mean"] >= 0.80
    assert summary["utility_gain_vs_uniform_mean"] >= 1.0
    assert summary["difficulty_budget_order"]["passed"] is True
    assert summary["frontier_audit_summary"]["decision"].endswith("READY")
    assert summary["synthetic_task_utility_audit"] is True
    assert summary["task_utility_passed"] is True
    assert summary["dynamic_budget_quality_validation"] is False
    assert summary["scanner_quality_validation"] is False
    assert summary["metric_claim_allowed"] is False
    assert summary["paper_claim_allowed"] is False
    assert summary["remote_sync_allowed"] is False
    assert summary["slurm_gpu_allowed"] is False
    assert summary["detector_map_allowed"] is False


def test_synthetic_task_utility_audit_json_roundtrip(tmp_path):
    input_json = tmp_path / "plan.json"
    task_json = tmp_path / "task.json"
    summary_json = tmp_path / "summary.json"
    input_json.write_text(json.dumps(_jsonable_plan(_dynamic_plan())), encoding="utf-8")
    task_json.write_text(json.dumps(_task_spec()), encoding="utf-8")

    summary = run_json_synthetic_task_utility_audit(
        input_json,
        task_json,
        summary_json=summary_json,
        sample_ids=["hard|0", "medium|1", "easy|2"],
        fixed_budget_reference=8,
        reference_budgets=(4, 6, 8),
    )

    loaded = json.loads(summary_json.read_text(encoding="utf-8"))
    assert loaded["decision"] == summary["decision"] == READY
    assert loaded["frontier_audit_summary"]["pipeline_summary"]["exact_budget_violations"] == 0
    assert loaded["task_rows"][0]["boundary_support"] == pytest.approx(1.0)


def test_synthetic_task_utility_audit_marks_uniform_budget_as_no_go():
    plan = _jsonable_plan(_dynamic_plan())
    plan["budgets"] = [4, 4, 4]
    plan["selected_mask"] = [[True, True, True, True, False, False, False, False] for _ in range(3)]

    summary = audit_pc_ot_mras_synthetic_task_utility(
        plan,
        _task_spec(),
        sample_ids=["hard|0", "medium|1", "easy|2"],
        fixed_budget_reference=8,
        reference_budgets=(4, 6, 8),
    )

    assert summary["decision"] == NO_GO
    assert summary["frontier_audit_summary"]["budget_sensitivity_passed"] is False


def test_synthetic_task_utility_audit_marks_boundary_misalignment_as_no_go():
    task = _task_spec()
    task["windows"][0]["start"] = 5
    task["windows"][0]["end"] = 6

    summary = audit_pc_ot_mras_synthetic_task_utility(
        _dynamic_plan(),
        task,
        sample_ids=["hard|0", "medium|1", "easy|2"],
        fixed_budget_reference=8,
        reference_budgets=(4, 6, 8),
    )

    assert summary["decision"] == NO_GO
    assert summary["boundary_support_mean"] < 0.95


def test_synthetic_task_utility_audit_rejects_leakage_keys_in_task_spec():
    task = _task_spec()
    task["annotation_path"] = "thumos_gt.json"

    with pytest.raises(ValueError, match="forbidden real-data"):
        audit_pc_ot_mras_synthetic_task_utility(
            _dynamic_plan(),
            task,
            sample_ids=["hard|0", "medium|1", "easy|2"],
            fixed_budget_reference=8,
            reference_budgets=(4, 6, 8),
        )


def test_synthetic_task_utility_audit_rejects_non_synthetic_split():
    with pytest.raises(ValueError, match="split='synthetic'"):
        audit_pc_ot_mras_synthetic_task_utility(
            _dynamic_plan(),
            _task_spec(),
            sample_ids=["hard|0", "medium|1", "easy|2"],
            fixed_budget_reference=8,
            reference_budgets=(4, 6, 8),
            split="val",
        )
