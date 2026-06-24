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
from tools.bata import validate_pc_ot_mras_dynamic_budget_pipeline as pipeline_module
from tools.bata.validate_pc_ot_mras_dynamic_budget_pipeline import (
    NO_GO,
    READY,
    run_json_validation,
    validate_pc_ot_mras_dynamic_budget_pipeline,
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


def test_dynamic_budget_pipeline_validation_closes_r22_r23_r24_protocol_chain():
    summary = validate_pc_ot_mras_dynamic_budget_pipeline(
        _dynamic_plan(),
        sample_ids=["high|0", "mid|1", "low|2"],
    )

    assert summary["decision"] == READY
    assert summary["row_count"] == 3
    assert summary["sample_ids"] == ["high|0", "mid|1", "low|2"]
    assert summary["budgets"] == [8, 6, 4]
    assert summary["selected_counts"] == [8, 6, 4]
    assert summary["grid_valid_counts"] == [8, 6, 4]
    assert summary["dense_valid_lens"] == [12, 10, 8]
    assert summary["exact_budget_violations"] == 0
    assert summary["center_matches_rows"] is True
    assert summary["sampling_contract_passed"] is True
    assert summary["temporal_grid_passed"] is True
    assert summary["dynamic_budget_validation"] is False
    assert summary["scanner_quality_validation"] is False
    assert summary["metric_claim_allowed"] is False
    assert summary["paper_claim_allowed"] is False
    assert summary["remote_sync_allowed"] is False
    assert summary["slurm_gpu_allowed"] is False
    assert summary["detector_map_allowed"] is False


def test_dynamic_budget_pipeline_validation_json_roundtrip(tmp_path):
    input_json = tmp_path / "plan.json"
    summary_json = tmp_path / "summary.json"
    input_json.write_text(json.dumps(_jsonable_plan(_dynamic_plan())), encoding="utf-8")

    summary = run_json_validation(
        input_json,
        summary_json=summary_json,
        sample_ids=["high|0", "mid|1", "low|2"],
    )

    loaded = json.loads(summary_json.read_text(encoding="utf-8"))
    assert loaded["decision"] == READY
    assert loaded["budgets"] == summary["budgets"] == [8, 6, 4]
    assert loaded["exact_budget_violations"] == 0
    assert loaded["center_matches_rows"] is True


def test_dynamic_budget_pipeline_validation_marks_center_mismatch_no_go(monkeypatch):
    real_loader = pipeline_module._load_repo_function

    def fake_loader(rel_path, function_name):
        if function_name == "temporal_grid_from_metas":
            def fake_temporal_grid_from_metas(_metas, selected_mask, *, required=True, strict=True):
                return {
                    "center": torch.zeros_like(selected_mask, dtype=torch.float32),
                    "valid_mask": selected_mask.clone(),
                }

            return fake_temporal_grid_from_metas
        return real_loader(rel_path, function_name)

    monkeypatch.setattr(pipeline_module, "_load_repo_function", fake_loader)

    summary = pipeline_module.validate_pc_ot_mras_dynamic_budget_pipeline(
        _dynamic_plan(),
        sample_ids=["high|0", "mid|1", "low|2"],
    )

    assert summary["decision"] == NO_GO
    assert summary["exact_budget_violations"] == 0
    assert summary["center_matches_rows"] is False


def test_dynamic_budget_pipeline_validation_marks_budget_count_violation_no_go(monkeypatch):
    real_loader = pipeline_module._load_repo_function

    def fake_loader(rel_path, function_name):
        if function_name == "temporal_grid_from_metas":
            def fake_temporal_grid_from_metas(metas, selected_mask, *, required=True, strict=True):
                center = torch.zeros_like(selected_mask, dtype=torch.float32)
                for batch_idx, meta in enumerate(metas):
                    positions = torch.tensor(meta["irregular_selected_positions"], dtype=torch.float32)
                    center[batch_idx, : positions.numel()] = positions
                valid_mask = selected_mask.clone()
                valid_mask[0, 0] = False
                return {"center": center, "valid_mask": valid_mask}

            return fake_temporal_grid_from_metas
        return real_loader(rel_path, function_name)

    monkeypatch.setattr(pipeline_module, "_load_repo_function", fake_loader)

    summary = pipeline_module.validate_pc_ot_mras_dynamic_budget_pipeline(
        _dynamic_plan(),
        sample_ids=["high|0", "mid|1", "low|2"],
    )

    assert summary["decision"] == NO_GO
    assert summary["exact_budget_violations"] == 1
    assert summary["center_matches_rows"] is True


def test_dynamic_budget_pipeline_validation_rejects_forbidden_plan_values():
    plan = _jsonable_plan(_dynamic_plan())
    plan["controller_family"] = "teacher_prediction_side_input"

    with pytest.raises(ValueError, match="forbidden deploy-invisible value"):
        validate_pc_ot_mras_dynamic_budget_pipeline(plan)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("pc_ot_mras_value_targets", [0.1, 0.2]),
        ("value_targets", [0.1, 0.2]),
        ("train_value_targets_at_test", [0.1, 0.2]),
        ("target_logits", [[1.0, 0.0]]),
        ("labels", [1, 2]),
        ("segments", [[0.0, 1.0]]),
        ("annotation_path", "ann.json"),
        ("allow_remote_sync", True),
        ("allow_slurm", True),
        ("allow_gpu", True),
        ("allow_detector_training", True),
        ("allow_tools_train", True),
        ("allow_tools_test", True),
        ("allow_detector_map", True),
        ("deploy_claim_allowed", True),
        ("runtime_flops_claim_allowed", True),
        ("scanner_quality_claim_allowed", True),
        ("dynamic_budget_claim_allowed", True),
        ("scanner_quality_validation", True),
        ("debug_side_channel", {"safe": 1}),
    ],
)
def test_dynamic_budget_pipeline_validation_rejects_forbidden_or_unknown_plan_keys(key, value):
    plan = _jsonable_plan(_dynamic_plan())
    plan[key] = value

    with pytest.raises(ValueError, match="dynamic budget plan"):
        validate_pc_ot_mras_dynamic_budget_pipeline(plan)
