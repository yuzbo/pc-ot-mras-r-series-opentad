from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
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
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
BH_SDC_PATH = ROOT / "opentad" / "models" / "selectors" / "bh_sdc_frame_selector.py"
BH_SDC_ROUTE_LABEL = "DIVERGENT_INNOVATION_BH_SDC_DO_NOT_MERGE_WITH_C3"


class _Registry:
    def __init__(self):
        self._items = {}

    def register_module(self):
        def _decorator(cls):
            self._items[cls.__name__] = cls
            return cls

        return _decorator

    def build(self, cfg):
        cfg = dict(cfg)
        type_name = cfg.pop("type")
        return self._items[type_name](**cfg)


def _ensure_package(name: str, path: Path):
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    return module


def _load_bh_sdc_module():
    for name in (
        "opentad.models.selectors.bh_sdc_frame_selector",
        "opentad.models.builder",
    ):
        sys.modules.pop(name, None)

    _ensure_package("opentad", ROOT / "opentad")
    _ensure_package("opentad.models", ROOT / "opentad" / "models")
    _ensure_package("opentad.models.selectors", ROOT / "opentad" / "models" / "selectors")

    builder = types.ModuleType("opentad.models.builder")
    builder.MODELS = _Registry()
    builder.SELECTORS = builder.MODELS
    builder.TOKEN_COMPRESSORS = builder.MODELS
    builder.build_selector = lambda cfg: builder.SELECTORS.build(cfg)
    builder.build_token_compressor = lambda cfg: builder.TOKEN_COMPRESSORS.build(cfg)
    sys.modules["opentad.models.builder"] = builder

    spec = importlib.util.spec_from_file_location(
        "opentad.models.selectors.bh_sdc_frame_selector",
        BH_SDC_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, builder


def _time_index_features(batch: int = 2, channels: int = 3, dense_len: int = 16):
    base = torch.arange(dense_len, dtype=torch.float32).view(1, 1, dense_len)
    inputs = base.expand(batch, channels, dense_len).contiguous()
    masks = torch.ones(batch, dense_len, dtype=torch.bool)
    metas = [{"sample_id": f"bh-sdc-{idx}"} for idx in range(batch)]
    gt_segments = [
        torch.tensor([[2.0, 5.0], [10.0, 14.0]], dtype=torch.float32),
        torch.tensor([[4.0, 8.0]], dtype=torch.float32),
    ]
    gt_labels = [torch.tensor([1, 2], dtype=torch.long), torch.tensor([3], dtype=torch.long)]
    return inputs, masks, metas, gt_segments, gt_labels


def _assert_prefix_mask(mask: torch.Tensor):
    for row in mask:
        valid = int(row.long().sum().item())
        assert row[:valid].all()
        assert not row[valid:].any()


def test_scout_outputs_boundary_hazard_contract_and_masks_invalid_tail():
    module, _builder = _load_bh_sdc_module()
    scout = module.BoundaryHazardTemporalScout(in_channels=4, hidden_dim=8, num_layers=2)
    inputs = torch.randn(2, 4, 10)
    mask = torch.ones(2, 10, dtype=torch.bool)
    mask[1, 7:] = False

    outputs = scout(inputs, mask)

    required = {
        "actionness_logits",
        "start_hazard_logits",
        "end_hazard_logits",
        "boundary_logits",
        "difficulty_logits",
        "uncertainty_logits",
        "redundancy_logits",
        "frame_selection_logits",
        "valid_mask",
        "protocol",
    }
    assert required.issubset(outputs)
    for key in required - {"valid_mask", "protocol"}:
        assert outputs[key].shape == (2, 10)
        assert torch.isfinite(outputs[key]).all()
    assert outputs["valid_mask"].dtype == torch.bool
    assert torch.equal(outputs["valid_mask"], mask)
    assert outputs["actionness_logits"][1, 7:].max().item() < -1000.0
    assert outputs["protocol"]["uses_gt"] is False
    assert outputs["protocol"]["uses_teacher"] is False
    assert outputs["protocol"]["uses_raw_prediction_cache"] is False
    assert outputs["protocol"]["route_label"] == BH_SDC_ROUTE_LABEL


def test_dynamic_budget_controller_allocates_more_to_boundary_hazard_samples():
    module, _builder = _load_bh_sdc_module()
    controller = module.BoundaryHazardDynamicBudgetController(
        min_budget=4,
        target_budget=6,
        max_budget=8,
        budget_step=2,
        hazard_weight=2.0,
        uncertainty_weight=1.0,
        redundancy_weight=1.0,
    )
    valid = torch.ones(3, 12, dtype=torch.bool)
    low = torch.full((3, 12), -4.0)
    high = torch.full((3, 12), 4.0)
    scout_out = {
        "actionness_logits": torch.stack([low[0], torch.zeros(12), high[2]]),
        "start_hazard_logits": torch.stack([low[0], torch.zeros(12), high[2]]),
        "end_hazard_logits": torch.stack([low[0], torch.zeros(12), high[2]]),
        "boundary_logits": torch.stack([low[0], torch.zeros(12), high[2]]),
        "difficulty_logits": torch.stack([low[0], torch.zeros(12), high[2]]),
        "uncertainty_logits": torch.stack([low[0], torch.zeros(12), high[2]]),
        "redundancy_logits": torch.stack([high[0], torch.zeros(12), low[2]]),
        "valid_mask": valid,
    }

    budget, meta = controller(scout_out, valid)

    assert budget.tolist() == [4, 6, 8]
    assert meta["protocol"] == "bh_sdc_dynamic_budget_v1"
    assert meta["uses_gt"] is False
    assert meta["uses_teacher"] is False
    assert meta["uses_raw_prediction_cache"] is False


def test_dynamic_budget_controller_maps_neutral_risk_to_declared_target_budget():
    module, _builder = _load_bh_sdc_module()
    controller = module.BoundaryHazardDynamicBudgetController(
        min_budget=256,
        target_budget=384,
        max_budget=448,
        budget_step=32,
    )
    valid = torch.ones(2, 768, dtype=torch.bool)
    zeros = torch.zeros(2, 768)
    scout_out = {
        "actionness_logits": zeros.clone(),
        "start_hazard_logits": zeros.clone(),
        "end_hazard_logits": zeros.clone(),
        "boundary_logits": zeros.clone(),
        "difficulty_logits": zeros.clone(),
        "uncertainty_logits": zeros.clone(),
        "redundancy_logits": zeros.clone(),
        "valid_mask": valid,
    }

    budget, _meta = controller(scout_out, valid)

    assert budget.tolist() == [384, 384]


def test_acquisition_policy_keeps_boundary_peaks_coverage_and_sorted_unique_indices():
    module, _builder = _load_bh_sdc_module()
    policy = module.BoundaryHazardAcquisitionPolicy(
        dense_window_size=16,
        min_budget=4,
        max_budget=8,
        coverage_ratio=0.25,
        boundary_ratio=0.50,
        max_dense_gap=5,
    )
    valid = torch.ones(1, 16, dtype=torch.bool)
    zeros = torch.zeros(1, 16)
    scout_out = {
        "actionness_logits": zeros.clone(),
        "start_hazard_logits": zeros.clone(),
        "end_hazard_logits": zeros.clone(),
        "boundary_logits": zeros.clone(),
        "difficulty_logits": zeros.clone(),
        "uncertainty_logits": zeros.clone(),
        "redundancy_logits": zeros.clone(),
        "valid_mask": valid,
    }
    scout_out["start_hazard_logits"][0, 3] = 9.0
    scout_out["end_hazard_logits"][0, 12] = 9.0
    scout_out["difficulty_logits"][0, 14] = 8.0
    scout_out["redundancy_logits"][0, 7] = 8.0

    plan = policy(
        dense_axis=torch.arange(16)[None, :],
        scout_out=scout_out,
        budget_per_sample=torch.tensor([8]),
        valid_mask=valid,
        metas=[{"sample_id": "policy"}],
    )

    selected = plan.selected_dense_indices[0, plan.selected_mask[0]].tolist()
    assert selected == sorted(set(selected))
    assert 3 in selected
    assert 12 in selected
    assert 0 in selected or 15 in selected
    assert max(b - a for a, b in zip(selected, selected[1:])) <= 5
    assert not any(idx == 7 for idx in selected[:4])
    assert plan.roles["boundary_hazard"].shape == plan.selected_mask.shape
    _assert_prefix_mask(plan.selected_mask)


def test_forward_test_rejects_gt_teacher_or_raw_prediction_meta_payloads():
    module, _builder = _load_bh_sdc_module()
    selector = module.PCOTMRASBoundaryHazardSparseDenseFrameSelector(
        input_channels=3,
        dense_window_size=16,
        min_budget=4,
        target_budget=6,
        max_budget=8,
        budget_step=2,
        scout_hidden_dim=12,
        scout_num_layers=2,
    )
    inputs, masks, metas, _gt_segments, _gt_labels = _time_index_features()

    bad_metas = [
        {"sample_id": "bad-gt", "gt_segments": [[1.0, 2.0]]},
        {"sample_id": "bad-teacher", "teacher_logits": [0.1]},
    ]
    with pytest.raises(ValueError, match="forbidden deploy/test-time payload"):
        selector.forward_test(inputs, masks, bad_metas)

    bridge = module.PCOTMRASBoundaryHazardSparseToDenseBridge(dense_window_size=16, target_len=16)
    good_outputs = selector.forward_test(inputs, masks, metas)
    good_outputs["metas"][0]["raw_prediction_cache"] = "forbidden"
    with pytest.raises(ValueError, match="forbidden deploy/test-time payload"):
        bridge.forward_test(good_outputs["inputs"], good_outputs["masks"], good_outputs["metas"])


def test_selector_and_sparse_dense_bridge_preserve_dense_axis_and_gradient_flow():
    module, builder = _load_bh_sdc_module()
    selector = module.PCOTMRASBoundaryHazardSparseDenseFrameSelector(
        input_channels=3,
        dense_window_size=16,
        min_budget=4,
        target_budget=6,
        max_budget=8,
        budget_step=2,
        scout_hidden_dim=12,
        scout_num_layers=2,
        coverage_ratio=0.25,
        boundary_ratio=0.50,
        max_dense_gap=5,
        aux_hazard_loss_weight=0.10,
    )
    bridge = module.PCOTMRASBoundaryHazardSparseToDenseBridge(
        dense_window_size=16,
        target_len=16,
        interpolation_temperature=2.0,
        refine_channels=3,
        refine_layers=1,
    )
    assert "PCOTMRASBoundaryHazardSparseDenseFrameSelector" in builder.MODELS._items
    assert "PCOTMRASBoundaryHazardSparseToDenseBridge" in builder.MODELS._items

    inputs, masks, metas, gt_segments, gt_labels = _time_index_features()
    inputs.requires_grad_(True)
    selector_outputs = selector.forward_train(inputs, masks, metas, gt_segments, gt_labels)

    assert selector_outputs["inputs"].shape == (2, 3, 8)
    assert selector_outputs["masks"].shape == (2, 8)
    _assert_prefix_mask(selector_outputs["masks"])
    assert selector_outputs["gt_segments"] is gt_segments
    assert selector_outputs["gt_labels"] is gt_labels
    assert "selector_bh_sdc_hazard_loss" in selector_outputs["losses"]

    for meta in selector_outputs["metas"]:
        plan = meta["bh_sdc_acquisition_plan"]
        assert plan["protocol"] == "bh_sdc_acquisition_plan_v1"
        assert plan["route_label"] == BH_SDC_ROUTE_LABEL
        assert plan["uses_gt"] is False
        assert plan["uses_teacher"] is False
        assert plan["uses_raw_prediction_cache"] is False
        selected = plan["selected_dense_indices"][: plan["selected_count"]]
        assert selected == sorted(set(selected))

    bridge_outputs = bridge.forward_train(
        features=selector_outputs["inputs"],
        masks=selector_outputs["masks"],
        metas=selector_outputs["metas"],
        gt_segments=selector_outputs["gt_segments"],
        gt_labels=selector_outputs["gt_labels"],
    )

    assert bridge_outputs["features"].shape == (2, 3, 16)
    assert bridge_outputs["masks"].shape == (2, 16)
    assert torch.equal(bridge_outputs["masks"], masks)
    assert bridge_outputs["gt_segments"] is gt_segments
    assert bridge_outputs["gt_labels"] is gt_labels
    assert "token_bh_sdc_completion_smoothness_loss" in bridge_outputs["losses"]
    assert bridge_outputs["metas"][0]["bh_sdc_completion"]["route_label"] == BH_SDC_ROUTE_LABEL
    for batch_idx, meta in enumerate(selector_outputs["metas"]):
        selected = meta["bh_sdc_acquisition_plan"]["selected_dense_indices"][
            : meta["bh_sdc_acquisition_plan"]["selected_count"]
        ]
        for sparse_col, dense_idx in enumerate(selected):
            assert torch.allclose(
                bridge_outputs["features"][batch_idx, :, dense_idx],
                selector_outputs["inputs"][batch_idx, :, sparse_col],
                atol=1e-6,
            )

    loss = bridge_outputs["features"].square().mean() + sum(selector_outputs["losses"].values())
    loss.backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
