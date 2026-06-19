import importlib.util
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
from tools.bata.export_pc_ot_mras_hard_positions import (
    DYNAMIC_BUDGET_GENERATION_SOURCE,
    resolve_pc_ot_mras_dynamic_budget_plan,
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


def test_dynamic_budget_plan_resolves_to_variable_hard_position_rows():
    rows = resolve_pc_ot_mras_dynamic_budget_plan(
        _dynamic_plan(),
        sample_ids=["high|0", "mid|1", "low|2"],
    )

    assert [row["budget"] for row in rows] == [8, 6, 4]
    assert [row["dense_len"] for row in rows] == [12, 10, 8]
    assert [sum(row["selected_mask"]) for row in rows] == [8, 6, 4]
    for row in rows:
        assert row["schema_version"] == "pc_ot_mras_hard_positions_v0"
        assert row["selected_positions"] == sorted(row["selected_positions"])
        assert len(row["selected_positions"]) == len(set(row["selected_positions"]))
        assert max(row["selected_positions"]) < row["dense_len"]
        assert row["duplicate_repair_count"] == 0
        assert row["repair_fill_count"] == 0
        assert row["dynamic_budget_plan"]["dynamic_budget_validation"] is False
        assert row["dynamic_budget_plan"]["metric_claim_allowed"] is False
        assert row["dynamic_budget_plan"]["paper_claim_allowed"] is False
        assert row["resolver_generation"]["source"] == DYNAMIC_BUDGET_GENERATION_SOURCE
        assert row["resolver_generation"]["training_backprop_allowed"] is False
        assert row["resolver_generation"]["dynamic_budget_validation"] is False


def test_dynamic_budget_plan_rejects_forbidden_payloads_and_true_claim_flags():
    plan = dict(_dynamic_plan())
    plan["uses_raw_prediction"] = True
    with pytest.raises(ValueError, match="uses_raw_prediction"):
        resolve_pc_ot_mras_dynamic_budget_plan(plan)

    plan = dict(_dynamic_plan())
    plan["raw_prediction_cache"] = "cache.json"
    with pytest.raises(ValueError, match="forbidden deploy-invisible key"):
        resolve_pc_ot_mras_dynamic_budget_plan(plan)


def test_dynamic_budget_plan_rejects_mask_budget_mismatch():
    plan = dict(_dynamic_plan())
    selected_mask = plan["selected_mask"].clone()
    selected_mask[0, 0] = False
    plan["selected_mask"] = selected_mask

    with pytest.raises(ValueError, match="true count must equal dynamic budget"):
        resolve_pc_ot_mras_dynamic_budget_plan(plan)


def test_dynamic_budget_plan_rejects_duplicate_or_unsorted_positions():
    plan = dict(_dynamic_plan())
    positions = plan["selected_dense_positions"].clone()
    positions[0, 1] = positions[0, 0]
    plan["selected_dense_positions"] = positions
    with pytest.raises(ValueError, match="unique"):
        resolve_pc_ot_mras_dynamic_budget_plan(plan)

    plan = dict(_dynamic_plan())
    positions = plan["selected_dense_positions"].clone()
    positions[0, 0], positions[0, 1] = positions[0, 1].clone(), positions[0, 0].clone()
    plan["selected_dense_positions"] = positions
    with pytest.raises(ValueError, match="sorted"):
        resolve_pc_ot_mras_dynamic_budget_plan(plan)
