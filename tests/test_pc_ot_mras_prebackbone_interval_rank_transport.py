from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SELECTOR_PATH = ROOT / "opentad" / "models" / "selectors" / "pc_ot_mras_prebackbone_frame_selector.py"


class _Registry:
    def register_module(self):
        def _decorator(cls):
            return cls

        return _decorator


def _ensure_package(name: str, path: Path):
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    return module


def _import_torch_or_skip():
    probe = subprocess.run(
        [sys.executable, "-c", "import torch"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
        check=False,
    )
    if probe.returncode != 0:
        detail = probe.stderr.strip().splitlines()[-1] if probe.stderr.strip() else f"exit {probe.returncode}"
        pytest.skip(f"torch unavailable in this process: {detail}")
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on local DLL state.
        pytest.skip(f"torch unavailable in this process: {exc}")
    return torch


def _load_prebackbone_selector_module(fake_reader):
    for name in (
        "opentad.models.selectors.pc_ot_mras_prebackbone_frame_selector",
        "opentad.models.builder",
    ):
        sys.modules.pop(name, None)

    _ensure_package("opentad", ROOT / "opentad")
    _ensure_package("opentad.models", ROOT / "opentad" / "models")
    _ensure_package("opentad.models.selectors", ROOT / "opentad" / "models" / "selectors")

    builder = types.ModuleType("opentad.models.builder")
    builder.SELECTORS = _Registry()
    builder.build_selector = lambda _cfg: fake_reader
    sys.modules["opentad.models.builder"] = builder

    spec = importlib.util.spec_from_file_location(
        "opentad.models.selectors.pc_ot_mras_prebackbone_frame_selector",
        SELECTOR_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_time_index_inputs(torch, *, batch: int = 1, dense_len: int = 8):
    values = torch.arange(dense_len, dtype=torch.float32).view(1, 1, dense_len, 1, 1)
    inputs = values.expand(batch, 3, dense_len, 2, 2).contiguous()
    masks = torch.ones((batch, dense_len), dtype=torch.bool)
    metas = [{"sample_id": f"interval-rank-{idx}"} for idx in range(batch)]
    gt_segments = [torch.tensor([[4.0, 8.0]], dtype=torch.float32) for _ in range(batch)]
    gt_labels = [torch.tensor([1], dtype=torch.long) for _ in range(batch)]
    return inputs, masks, metas, gt_segments, gt_labels


class _IntervalPacketReader:
    def __call__(self, lowcost_features, valid_mask, time_coords=None):
        batch, time, _dim = lowcost_features.shape
        device = lowcost_features.device
        frame_selection = lowcost_features.new_full((batch, time), -8.0)
        frame_selection[:, :4] = lowcost_features.new_tensor([12.0, 11.0, 10.0, 9.0])
        action = lowcost_features.new_full((batch, time), -8.0)
        action[:, 5:7] = 8.0
        start = lowcost_features.new_full((batch, time), -8.0)
        end = lowcost_features.new_full((batch, time), -8.0)
        boundary = lowcost_features.new_full((batch, time), -8.0)
        start[:, 4] = 9.0
        end[:, 7] = 9.0
        boundary[:, 4] = 8.5
        boundary[:, 7] = 8.5
        uncertainty = lowcost_features.new_zeros((batch, time))
        uncertainty[:, [4, 7]] = 1.0
        redundancy = lowcost_features.new_zeros((batch, time))
        redundancy[:, :4] = 4.0
        slot_logits = lowcost_features.new_zeros((batch, 4, time))
        return {
            "slot_logits": slot_logits,
            "acquisition_matrix": slot_logits.softmax(dim=-1),
            "frame_selection_logits": frame_selection,
            "actionness_logits": action,
            "action_logits": action,
            "value_logits": action,
            "start_logits": start,
            "end_logits": end,
            "boundary_logits": boundary,
            "uncertainty_logits": uncertainty,
            "redundancy_logits": redundancy,
            "regularizers": {"total_regularizer": slot_logits.sum() * 0.0},
        }


class _TrainableGlobalCompetitorReader:
    def __init__(self, torch):
        self.frame_logits = torch.nn.Parameter(
            torch.tensor([10.0, 9.0, -6.0, -7.0, -8.0, -9.0, -10.0, 8.5], dtype=torch.float32)
        )

    def __call__(self, lowcost_features, valid_mask, time_coords=None):
        batch, time, _dim = lowcost_features.shape
        slots = 2
        slot_logits = lowcost_features.new_zeros((batch, slots, time))
        frame_scores = self.frame_logits[:time].to(device=lowcost_features.device).unsqueeze(0).expand(batch, -1)
        return {
            "slot_logits": slot_logits,
            "acquisition_matrix": slot_logits.softmax(dim=-1),
            "frame_selection_logits": frame_scores,
            "actionness_logits": frame_scores,
            "start_logits": lowcost_features.new_zeros((batch, time)),
            "end_logits": lowcost_features.new_zeros((batch, time)),
            "boundary_logits": lowcost_features.new_zeros((batch, time)),
            "uncertainty_logits": lowcost_features.new_zeros((batch, time)),
            "redundancy_logits": lowcost_features.new_zeros((batch, time)),
            "regularizers": {"total_regularizer": frame_scores.sum() * 0.0},
        }


def test_interval_boundary_packet_selection_is_not_bare_frame_score_argsort():
    torch = _import_torch_or_skip()
    module = _load_prebackbone_selector_module(_IntervalPacketReader())
    selector = module.PCOTMRASPreBackboneFrameSelector(
        reader={"type": "IntervalPacketReader"},
        target_len=4,
        dense_window_size=8,
        descriptor_dim=12,
        selection_strategy="interval_boundary_packet",
        protected_uniform_count=0,
        coverage_guard_count=0,
        interval_boundary_budget_ratio=0.5,
    )
    inputs, masks, metas, _gt_segments, _gt_labels = _make_time_index_inputs(torch, dense_len=8)

    outputs = selector.forward_test(inputs, masks, metas)
    meta = outputs["metas"][0]

    selected = meta["pc_ot_mras_prebackbone_selected_dense_indices"]
    assert selected == [4, 5, 6, 7]
    assert selected != [0, 1, 2, 3]
    assert torch.equal(outputs["inputs"][0, :, :, 0, 0].mean(dim=0), torch.tensor(selected, dtype=torch.float32))
    assert meta["pc_ot_mras_prebackbone_hard_selection_source"] == "interval_boundary_packet"
    assert "boundary_packet" in meta["pc_ot_mras_prebackbone_selected_roles"]
    assert "interior_action_packet" in meta["pc_ot_mras_prebackbone_selected_roles"]
    packet_meta = meta["pc_ot_mras_prebackbone_interval_packet_metadata"]
    assert packet_meta["boundary_budget"] == 2
    assert packet_meta["interior_budget"] == 2
    assert packet_meta["boundary_positions"] == [4, 7]
    assert packet_meta["interior_positions"] == [5, 6]
    assert set(packet_meta["source_heads"]) >= {
        "actionness_logits",
        "start_logits",
        "end_logits",
        "boundary_logits",
        "uncertainty_logits",
        "redundancy_logits",
    }


def test_interval_boundary_packet_train_eval_hard_indices_match_and_remain_finite():
    torch = _import_torch_or_skip()
    module = _load_prebackbone_selector_module(_IntervalPacketReader())
    selector = module.PCOTMRASPreBackboneFrameSelector(
        reader={"type": "IntervalPacketReader"},
        target_len=4,
        dense_window_size=8,
        descriptor_dim=12,
        selection_strategy="interval_boundary_packet",
        protected_uniform_count=0,
        coverage_guard_count=0,
        interval_boundary_budget_ratio=0.5,
        frame_score_st_surrogate="global_softmax",
    )
    inputs, masks, metas, gt_segments, gt_labels = _make_time_index_inputs(torch, dense_len=8)

    train_outputs = selector.forward_train(inputs, masks, metas, gt_segments, gt_labels)
    test_outputs = selector.forward_test(inputs, masks, [{"split": "test"}])

    train_selected = train_outputs["metas"][0]["pc_ot_mras_prebackbone_selected_dense_indices"]
    test_selected = test_outputs["metas"][0]["pc_ot_mras_prebackbone_selected_dense_indices"]
    assert train_selected == test_selected == [4, 5, 6, 7]
    assert torch.isfinite(train_outputs["inputs"]).all()
    assert torch.isfinite(test_outputs["inputs"]).all()
    assert train_outputs["masks"].tolist() == [[True, True, True, True]]
    assert test_outputs["masks"].tolist() == [[True, True, True, True]]


def test_interval_boundary_packet_fixed384_pair_enumeration_is_topk_bounded():
    torch = _import_torch_or_skip()
    module = _load_prebackbone_selector_module(_IntervalPacketReader())
    selector = module.PCOTMRASPreBackboneFrameSelector(
        reader={"type": "IntervalPacketReader"},
        target_len=384,
        dense_window_size=768,
        descriptor_dim=12,
        selection_strategy="interval_boundary_packet",
        protected_uniform_count=0,
        coverage_guard_count=0,
        interval_boundary_budget_ratio=0.5,
        interval_candidate_topk=24,
        frame_score_st_surrogate="global_softmax",
    )
    inputs, masks, metas, _gt_segments, _gt_labels = _make_time_index_inputs(torch, dense_len=768)

    outputs = selector.forward_test(inputs, masks, metas)

    meta = outputs["metas"][0]
    packet_meta = meta["pc_ot_mras_prebackbone_interval_packet_metadata"]
    assert outputs["masks"].long().sum().item() == 384
    assert packet_meta["interval_candidate_topk"] == 24
    assert packet_meta["interval_pair_limit"] == 24
    assert packet_meta["interval_record_count"] <= 24 * 24
    assert packet_meta["interior_candidate_count"] <= 768


def test_physical_grid_selector_mode_keeps_training_gt_on_dense_axis_when_remap_disabled():
    torch = _import_torch_or_skip()
    module = _load_prebackbone_selector_module(_IntervalPacketReader())
    selector = module.PCOTMRASPreBackboneFrameSelector(
        reader={"type": "IntervalPacketReader"},
        target_len=4,
        dense_window_size=8,
        descriptor_dim=12,
        selection_strategy="frame_score_topk",
        protected_uniform_count=0,
        coverage_guard_count=0,
        remap_gt_to_selected_axis=False,
    )
    inputs, masks, metas, gt_segments, gt_labels = _make_time_index_inputs(torch, dense_len=8)

    outputs = selector.forward_train(inputs, masks, metas, gt_segments, gt_labels)

    assert outputs["gt_segments"] is gt_segments
    assert outputs["gt_labels"] is gt_labels
    assert torch.equal(outputs["gt_segments"][0], torch.tensor([[4.0, 8.0]], dtype=torch.float32))
    assert outputs["metas"][0]["pc_ot_mras_prebackbone_selected_dense_indices"] != [4, 5, 6, 7]


def test_global_rank_surrogate_backpropagates_to_nonlocal_competing_frame():
    torch = _import_torch_or_skip()
    reader = _TrainableGlobalCompetitorReader(torch)
    module = _load_prebackbone_selector_module(reader)
    selector = module.PCOTMRASPreBackboneFrameSelector(
        reader={"type": "TrainableGlobalCompetitorReader"},
        target_len=2,
        dense_window_size=8,
        descriptor_dim=12,
        selection_strategy="frame_score_topk",
        protected_uniform_count=0,
        coverage_guard_count=0,
        frame_score_st_surrogate="global_softmax",
        frame_score_st_temperature=1.0,
    )
    inputs, masks, metas, gt_segments, gt_labels = _make_time_index_inputs(torch, dense_len=8)

    outputs = selector.forward_train(inputs, masks, metas, gt_segments, gt_labels)
    selected = outputs["metas"][0]["pc_ot_mras_prebackbone_selected_dense_indices"]
    assert selected == [0, 1]

    detector_like_loss = outputs["inputs"].sum()
    detector_like_loss.backward()

    assert reader.frame_logits.grad is not None
    assert torch.isfinite(reader.frame_logits.grad).all()
    assert reader.frame_logits.grad[7].abs().item() > 1.0e-6
