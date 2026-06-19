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


def test_dynamic_budget_controller_maps_value_signal_to_exact_variable_budget_plan():
    controller = PCOTMRASDynamicBudgetController(
        budget_values=(4, 6, 8),
        budget_thresholds=(0.25, 0.75),
        transport_weight=0.0,
        coverage_share=0.25,
        max_coverage_share=0.50,
    )

    plan = controller(_reader_outputs())

    assert plan["schema_version"] == "pc_ot_mras_dynamic_budget_plan_v0"
    assert plan["controller_family"] == "r22_value_to_budget_control"
    assert plan["uses_gt"] is False
    assert plan["uses_teacher"] is False
    assert plan["uses_raw_prediction"] is False
    assert plan["dynamic_budget_validation"] is False
    assert plan["metric_claim_allowed"] is False
    assert plan["paper_claim_allowed"] is False

    assert plan["budgets"].tolist() == [8, 6, 4]
    assert plan["dense_valid_len"].tolist() == [12, 10, 8]
    assert plan["selected_mask"].long().sum(dim=1).tolist() == [8, 6, 4]
    assert torch.all(plan["coverage_share"] <= 0.50 + 1.0e-6)

    for batch_idx, budget in enumerate(plan["budgets"].tolist()):
        positions = plan["selected_dense_positions"][batch_idx, :budget]
        assert positions.tolist() == sorted(positions.tolist())
        assert len(set(positions.tolist())) == int(budget)
        assert int(positions.min().item()) >= 0
        assert int(positions.max().item()) < int(plan["dense_valid_len"][batch_idx].item())


def test_dynamic_budget_controller_rejects_train_only_or_shortcut_payloads():
    controller = PCOTMRASDynamicBudgetController(budget_values=(4, 6), budget_thresholds=(0.5,))
    reader_outputs = _reader_outputs()
    reader_outputs["pc_ot_mras_value_targets"] = {"split": "train"}

    with pytest.raises(ValueError, match="forbidden deploy-time payload"):
        controller(reader_outputs)


def test_dynamic_budget_controller_rejects_missing_value_signal_by_default():
    controller = PCOTMRASDynamicBudgetController(budget_values=(4, 6), budget_thresholds=(0.5,))
    reader_outputs = _reader_outputs()
    reader_outputs.pop("value_logits")

    with pytest.raises(ValueError, match="value_logits"):
        controller(reader_outputs)


def test_dynamic_budget_controller_rejects_non_prefix_valid_mask():
    controller = PCOTMRASDynamicBudgetController(budget_values=(4, 6), budget_thresholds=(0.5,))
    reader_outputs = _reader_outputs()
    reader_outputs["valid_mask"] = torch.tensor([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.float32)
    reader_outputs["value_logits"] = torch.zeros(1, 12)
    reader_outputs["risk_logits"] = torch.zeros(1, 12)
    reader_outputs["redundancy_logits"] = torch.zeros(1, 12)
    reader_outputs["acquisition_matrix"] = torch.zeros(1, 2, 12)

    with pytest.raises(ValueError, match="contiguous valid prefix"):
        controller(reader_outputs)
