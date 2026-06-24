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
    sys.modules["opentad.models.builder"] = builder

    spec = importlib.util.spec_from_file_location(
        "opentad.models.selectors.bh_sdc_frame_selector",
        BH_SDC_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_bh_sdc_metadata_exposes_probe_physical_and_completion_contract():
    module = _load_bh_sdc_module()
    selector = module.PCOTMRASBoundaryHazardSparseDenseFrameSelector(
        input_channels=2,
        dense_window_size=10,
        min_budget=3,
        target_budget=4,
        max_budget=6,
        budget_step=1,
        scout_hidden_dim=4,
        scout_num_layers=1,
        probe_stride=3,
        coverage_ratio=0.25,
        boundary_ratio=0.50,
        max_dense_gap=4,
        aux_hazard_loss_weight=0.0,
        aux_budget_entropy_loss_weight=0.0,
    )
    bridge = module.PCOTMRASBoundaryHazardSparseToDenseBridge(
        dense_window_size=10,
        target_len=10,
        interpolation_temperature=1.5,
        refine_layers=0,
    )

    class FixedScout(nn.Module):
        def forward(self, features, valid_mask, metas=None):
            logits = features[:, 0, :].clone().masked_fill(~valid_mask, -10000.0)
            return {
                "actionness_logits": logits,
                "start_hazard_logits": logits,
                "end_hazard_logits": logits,
                "boundary_logits": logits,
                "difficulty_logits": logits,
                "uncertainty_logits": logits,
                "redundancy_logits": -logits,
                "frame_selection_logits": logits,
                "valid_mask": valid_mask,
                "protocol": module._deploy_protocol_flags(),
            }

    class FixedBudget(nn.Module):
        def forward(self, scout_out, valid_mask):
            return torch.tensor([5], dtype=torch.long), {
                "min_budget": 3,
                "max_budget": 6,
                "protocol": "unit_fixed_budget",
                "uses_gt": False,
                "uses_teacher": False,
                "uses_raw_prediction_cache": False,
            }

    selector.scout = FixedScout()
    selector.budget_controller = FixedBudget()

    dense_values = torch.arange(10, dtype=torch.float32)
    inputs = torch.stack([dense_values, dense_values + 100.0]).view(1, 2, 10)
    masks = torch.ones(1, 10, dtype=torch.bool)
    metas = [
        {
            "sample_id": "metadata-contract",
            "fps": 25.0,
            "snippet_stride": 4,
            "window_start_frame": 100,
        }
    ]

    selected = selector.forward_test(inputs, masks, metas=metas)
    plan = selected["metas"][0]["bh_sdc_acquisition_plan"]
    selected_count = plan["selected_count"]
    selected_indices = plan["selected_dense_indices"][:selected_count]

    assert plan["route_label"] == BH_SDC_ROUTE_LABEL
    assert plan["probe_only_scout"] is True
    assert plan["probe_dense_indices"] == [0, 3, 6, 9]
    assert plan["probe_mask"] == [True, False, False, True, False, False, True, False, False, True]
    assert len(plan["physical_time_axis"]) == 10
    assert plan["physical_times"] == [plan["physical_time_axis"][idx] for idx in selected_indices]
    assert plan["physical_time_axis"][0] == pytest.approx(100.0 / 25.0)
    assert plan["physical_time_axis"][1] == pytest.approx(104.0 / 25.0)

    completed = bridge.forward_test(selected["inputs"], selected["masks"], selected["metas"])
    features = completed["features"]
    meta = completed["metas"][0]
    completion = meta["bh_sdc_completion"]

    assert completion["route_label"] == BH_SDC_ROUTE_LABEL
    assert len(completion["observed_mask"]) == 10
    assert len(completion["synthetic_mask"]) == 10
    assert len(completion["completion_confidence"]) == 10
    assert len(completion["gap_distance"]) == 10
    assert len(completion["physical_time_axis"]) == 10
    assert completion["observed_physical_times"] == [completion["physical_time_axis"][idx] for idx in selected_indices]
    assert completion["observed_mask"] == [idx in selected_indices for idx in range(10)]
    assert completion["synthetic_mask"] == [idx not in selected_indices for idx in range(10)]
    assert completion["completion_confidence"][selected_indices[0]] == pytest.approx(1.0)
    assert completion["gap_distance"][selected_indices[0]] == pytest.approx(0.0)
    assert meta["irregular_selected_positions"] == selected_indices
    assert meta["irregular_selected_count"] == selected_count
    assert meta["irregular_dense_valid_len"] == 10
    assert meta["irregular_native_axis"] is True
    assert meta["bh_sdc_detector_metadata"]["observed_mask"] == completion["observed_mask"]
    assert meta["bh_sdc_detector_metadata"]["synthetic_mask"] == completion["synthetic_mask"]

    for sparse_col, dense_idx in enumerate(selected_indices):
        assert torch.allclose(features[0, :, dense_idx], selected["inputs"][0, :, sparse_col], atol=1e-6)
