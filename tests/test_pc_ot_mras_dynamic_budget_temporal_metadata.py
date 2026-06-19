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
    TEMPORAL_METADATA_GENERATION_SOURCE,
    TEMPORAL_METADATA_SCHEMA_VERSION,
    pc_ot_mras_hard_rows_to_temporal_metas,
    resolve_pc_ot_mras_dynamic_budget_plan,
)


ROOT = Path(__file__).resolve().parents[1]
load_pc_ot_mras_classes()


def _load_function(module_name, file_name, function_name):
    spec = importlib.util.spec_from_file_location(
        module_name,
        ROOT / "opentad" / "models" / "utils" / file_name,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function_name)


def _load_controller_class():
    path = ROOT / "opentad" / "models" / "selectors" / "pc_ot_mras_dynamic_budget_controller.py"
    spec = importlib.util.spec_from_file_location("opentad.models.selectors.pc_ot_mras_dynamic_budget_controller", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.PCOTMRASDynamicBudgetController


temporal_grid_from_metas = _load_function(
    "pc_ot_mras_temporal_grid_r24_test",
    "temporal_grid.py",
    "temporal_grid_from_metas",
)
validate_sampling_contract = _load_function(
    "pc_ot_mras_sampling_contract_r24_test",
    "sampling_contract.py",
    "validate_sampling_contract",
)
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


def _rows():
    return resolve_pc_ot_mras_dynamic_budget_plan(
        _dynamic_plan(),
        sample_ids=["high|0", "mid|1", "low|2"],
    )


def _prefix_mask(budgets):
    max_budget = max(budgets)
    mask = torch.zeros(len(budgets), max_budget, dtype=torch.bool)
    for batch_idx, budget in enumerate(budgets):
        mask[batch_idx, :budget] = True
    return mask


def test_dynamic_budget_hard_rows_convert_to_temporal_grid_metadata():
    rows = _rows()
    metas = pc_ot_mras_hard_rows_to_temporal_metas(rows)
    mask = _prefix_mask([row["budget"] for row in rows])

    assert [meta["irregular_selected_count"] for meta in metas] == [8, 6, 4]
    assert [meta["irregular_dense_valid_len"] for meta in metas] == [12, 10, 8]
    assert [meta["irregular_selected_valid_len"] for meta in metas] == [12, 10, 8]
    assert all(meta["irregular_native_axis"] is True for meta in metas)
    assert all(meta["gt_axis"] == "dense" for meta in metas)
    assert all(meta["proposal_axis"] == "dense" for meta in metas)

    for row, meta in zip(rows, metas):
        export = meta["pc_ot_mras_dynamic_budget_export"]
        assert meta["irregular_selected_positions"] == [float(pos) for pos in row["selected_positions"]]
        assert export["schema_version"] == TEMPORAL_METADATA_SCHEMA_VERSION
        assert export["generation_source"] == TEMPORAL_METADATA_GENERATION_SOURCE
        assert export["dynamic_budget_validation"] is False
        assert export["metric_claim_allowed"] is False
        assert export["paper_claim_allowed"] is False

    assert validate_sampling_contract(metas, mask, split="val") is True
    grid = temporal_grid_from_metas(metas, mask, required=True, strict=True)

    assert torch.equal(grid["valid_mask"], mask)
    assert torch.allclose(grid["dense_valid_len"], torch.tensor([12.0, 10.0, 8.0]))
    for batch_idx, row in enumerate(rows):
        budget = row["budget"]
        expected = torch.tensor(row["selected_positions"], dtype=torch.float32)
        assert torch.allclose(grid["center"][batch_idx, :budget], expected)
        if budget < mask.shape[1]:
            assert torch.all(grid["center"][batch_idx, budget:] == expected[-1])


def test_dynamic_budget_temporal_metadata_rejects_selected_mask_mismatch():
    rows = _rows()
    bad = [dict(row) for row in rows]
    bad[0] = dict(bad[0])
    bad[0]["selected_mask"] = list(bad[0]["selected_mask"])
    bad[0]["selected_mask"][bad[0]["selected_positions"][0]] = 0

    with pytest.raises(ValueError, match="true count must equal budget"):
        pc_ot_mras_hard_rows_to_temporal_metas(bad)

    rows = _rows()
    bad = [dict(row) for row in rows]
    bad[0] = dict(bad[0])
    swapped_mask = list(bad[0]["selected_mask"])
    swapped_mask[bad[0]["selected_positions"][0]] = 0
    replacement = next(
        pos
        for pos in range(bad[0]["valid_len"])
        if pos not in set(bad[0]["selected_positions"])
    )
    swapped_mask[replacement] = 1
    bad[0]["selected_mask"] = swapped_mask

    with pytest.raises(ValueError, match="selected_mask must match selected_positions"):
        pc_ot_mras_hard_rows_to_temporal_metas(bad)


def test_dynamic_budget_temporal_metadata_rejects_invalid_positions_and_payloads():
    rows = _rows()
    bad = [dict(row) for row in rows]
    bad[0] = dict(bad[0])
    bad[0]["selected_positions"] = list(bad[0]["selected_positions"])
    bad[0]["selected_positions"][-1] = bad[0]["valid_len"]
    bad[0]["selected_mask"] = list(bad[0]["selected_mask"])
    bad[0]["selected_mask"][bad[0]["selected_positions"][-2]] = 0
    with pytest.raises(ValueError, match="selected_positions must stay inside valid_len"):
        pc_ot_mras_hard_rows_to_temporal_metas(bad)

    rows = _rows()
    bad = [dict(row) for row in rows]
    bad[0] = dict(bad[0])
    bad[0]["selected_positions"] = list(bad[0]["selected_positions"])
    bad[0]["selected_positions"][0] = float(bad[0]["selected_positions"][0]) + 0.5
    with pytest.raises(ValueError, match="must be an integer position"):
        pc_ot_mras_hard_rows_to_temporal_metas(bad)

    rows = _rows()
    bad = [dict(row) for row in rows]
    bad[0] = dict(bad[0])
    bad[0]["teacher_logits"] = [0.1, 0.2]
    with pytest.raises(ValueError, match="forbidden deploy-invisible key"):
        pc_ot_mras_hard_rows_to_temporal_metas(bad)

    rows = _rows()
    bad = [dict(row) for row in rows]
    bad[0] = dict(bad[0])
    bad[0]["source_note"] = "teacher_prediction_side_input"
    with pytest.raises(ValueError, match="forbidden deploy-invisible value"):
        pc_ot_mras_hard_rows_to_temporal_metas(bad)

    rows = _rows()
    bad = [dict(row) for row in rows]
    bad[0] = dict(bad[0])
    bad[0]["dynamic_budget_plan"] = dict(bad[0]["dynamic_budget_plan"])
    bad[0]["dynamic_budget_plan"]["metric_claim_allowed"] = True
    with pytest.raises(ValueError, match="metric_claim_allowed must be false"):
        pc_ot_mras_hard_rows_to_temporal_metas(bad)
