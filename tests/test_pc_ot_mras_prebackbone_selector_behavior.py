from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
SELECTOR_PATH = ROOT / "opentad" / "models" / "selectors" / "pc_ot_mras_prebackbone_frame_selector.py"


class _Registry:
    def register_module(self):
        def _decorator(cls):
            return cls

        return _decorator


class _TailBiasedFakeReader(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_features = None

    def forward(self, lowcost_features, valid_mask, time_coords=None):
        self.last_features = lowcost_features.detach()
        batch, time, _dim = lowcost_features.shape
        slots = 8
        matrix = lowcost_features.new_zeros((batch, slots, time))
        tail_positions = torch.tensor([15, 14, 13, 12, 11, 10, 9, 8], device=lowcost_features.device)
        for slot, pos in enumerate(tail_positions):
            matrix[:, slot, pos] = 1.0
        logits = lowcost_features.new_zeros((batch, time))
        logits[:, 3:7] = 3.0
        return {
            "acquisition_matrix": matrix,
            "value_logits": logits,
            "risk_logits": -logits,
            "role_logits": lowcost_features.new_zeros((batch, slots, 6)),
            "regularizers": {"total_regularizer": matrix.sum() * 0.0},
        }


class _SoftAmbiguousFakeReader(nn.Module):
    def forward(self, lowcost_features, valid_mask, time_coords=None):
        batch, time, _dim = lowcost_features.shape
        slots = 4
        matrix = lowcost_features.new_zeros((batch, slots, time))
        preferred = ((1, 2), (3, 4), (5, 6), (7, 0))
        for slot, (top1, top2) in enumerate(preferred):
            matrix[:, slot, top1] = 0.70
            matrix[:, slot, top2] = 0.30
        return {
            "acquisition_matrix": matrix,
            "regularizers": {"total_regularizer": matrix.sum() * 0.0},
        }


class _ResidualSlotsFakeReader(nn.Module):
    def forward(self, lowcost_features, valid_mask, time_coords=None):
        batch, time, _dim = lowcost_features.shape
        slots = 4
        matrix = lowcost_features.new_zeros((batch, slots, time))
        for slot, pos in enumerate((1, 3, 6, 12)):
            matrix[:, slot, pos] = 1.0
        return {
            "acquisition_matrix": matrix,
            "regularizers": {"total_regularizer": matrix.sum() * 0.0},
        }


class _ConflictingLogitsAndProbabilityReader(nn.Module):
    def forward(self, lowcost_features, valid_mask, time_coords=None):
        batch, time, _dim = lowcost_features.shape
        slots = 4
        logits = lowcost_features.new_full((batch, slots, time), -8.0)
        probability = lowcost_features.new_zeros((batch, slots, time))
        for slot, (logit_pos, probability_pos) in enumerate(zip((4, 5, 6, 7), (0, 1, 2, 3))):
            logits[:, slot, logit_pos] = 8.0
            probability[:, slot, probability_pos] = 1.0
        return {
            "slot_logits": logits,
            "acquisition_matrix": probability,
            "regularizers": {"total_regularizer": logits.sum() * 0.0},
        }


class _DuplicateSlotFakeReader(nn.Module):
    def forward(self, lowcost_features, valid_mask, time_coords=None):
        batch, time, _dim = lowcost_features.shape
        slots = 4
        logits = lowcost_features.new_full((batch, slots, time), -8.0)
        for slot, pos in enumerate((1, 1, 1, 7)):
            logits[:, slot, pos] = 8.0
        return {
            "slot_logits": logits,
            "acquisition_matrix": torch.softmax(logits, dim=-1),
            "regularizers": {"total_regularizer": logits.sum() * 0.0},
        }


class _TrainableSoftFakeReader(nn.Module):
    def __init__(self):
        super().__init__()
        logits = torch.full((1, 4, 8), -4.0)
        for slot, pos in enumerate((1, 3, 5, 7)):
            logits[0, slot, pos] = 4.0
        self.logits = nn.Parameter(logits)

    def forward(self, lowcost_features, valid_mask, time_coords=None):
        batch = lowcost_features.shape[0]
        logits = self.logits.expand(batch, -1, -1)
        return {
            "slot_logits": logits,
            "acquisition_matrix": torch.softmax(logits, dim=-1),
            "regularizers": {"total_regularizer": self.logits.sum() * 0.0},
        }


class _RecordingFramePixelReader(nn.Module):
    def __init__(self, num_slots=2):
        super().__init__()
        self.num_slots = int(num_slots)
        self.last_features = None

    def forward(self, lowcost_features, valid_mask, time_coords=None):
        self.last_features = lowcost_features.detach()
        batch, time, _dim = lowcost_features.shape
        matrix = lowcost_features.new_full((batch, self.num_slots, time), -4.0)
        for slot in range(self.num_slots):
            matrix[:, slot, min(slot, time - 1)] = 4.0
        return {
            "slot_logits": matrix,
            "acquisition_matrix": torch.softmax(matrix, dim=-1),
            "regularizers": {"total_regularizer": matrix.sum() * 0.0},
        }


def _load_selector_module(fake_reader):
    for name in (
        "opentad.models.selectors.pc_ot_mras_prebackbone_frame_selector",
        "opentad.models.builder",
    ):
        sys.modules.pop(name, None)

    for package, path in (
        ("opentad", ROOT / "opentad"),
        ("opentad.models", ROOT / "opentad" / "models"),
        ("opentad.models.selectors", ROOT / "opentad" / "models" / "selectors"),
    ):
        module = sys.modules.get(package)
        if module is None:
            module = types.ModuleType(package)
            module.__path__ = [str(path)]
            sys.modules[package] = module

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


def _make_inputs(batch=2, dense_len=16):
    frames = torch.arange(batch * 3 * dense_len * 4 * 4, dtype=torch.float32)
    inputs = frames.view(batch, 3, dense_len, 4, 4) / 255.0
    masks = torch.ones((batch, dense_len), dtype=torch.bool)
    metas = [{"sample_id": f"sample-{idx}"} for idx in range(batch)]
    gt_segments = [
        torch.tensor([[2.0, 5.0], [10.0, 15.5]], dtype=torch.float32),
        torch.tensor([[0.0, 3.5]], dtype=torch.float32),
    ]
    gt_labels = [torch.tensor([1, 2], dtype=torch.long), torch.tensor([3], dtype=torch.long)]
    return inputs, masks, metas, gt_segments, gt_labels


def _make_time_index_inputs(batch=1, dense_len=8):
    values = torch.arange(dense_len, dtype=torch.float32).view(1, 1, dense_len, 1, 1)
    inputs = values.expand(batch, 3, dense_len, 2, 2).contiguous()
    masks = torch.ones((batch, dense_len), dtype=torch.bool)
    metas = [{"sample_id": f"hard-{idx}"} for idx in range(batch)]
    gt_segments = [torch.tensor([[1.0, 7.0]], dtype=torch.float32) for _ in range(batch)]
    gt_labels = [torch.tensor([1], dtype=torch.long) for _ in range(batch)]
    return inputs, masks, metas, gt_segments, gt_labels


def test_compressed_pixel_scout_features_are_direct_downsampled_frame_pixels():
    fake_reader = _RecordingFramePixelReader(num_slots=2)
    module = _load_selector_module(fake_reader)
    selector = module.PCOTMRASPreBackboneFrameSelector(
        reader={"type": "PCOTMRASReader"},
        target_len=2,
        dense_window_size=4,
        descriptor_dim=12,
        scout_feature_source="compressed_pixels",
        scout_spatial_size=2,
        protected_uniform_count=0,
        coverage_guard_count=0,
    )
    inputs = torch.arange(1 * 3 * 4 * 4 * 4, dtype=torch.float32).view(1, 3, 4, 4, 4)
    masks = torch.ones((1, 4), dtype=torch.bool)
    metas = [{"sample_id": "pixels"}]

    outputs = selector.forward_test(inputs, masks, metas)

    expected = torch.nn.functional.interpolate(
        inputs.permute(0, 2, 1, 3, 4).reshape(4, 3, 4, 4),
        size=(2, 2),
        mode="bilinear",
        align_corners=False,
    ).reshape(1, 4, 12)
    assert fake_reader.last_features.shape == (1, 4, 12)
    assert torch.allclose(fake_reader.last_features, expected)
    assert outputs["metas"][0]["pc_ot_mras_prebackbone_scout_feature_source"] == "compressed_pixels"
    assert outputs["metas"][0]["pc_ot_mras_prebackbone_scout_spatial_size"] == [2, 2]


def test_forward_keeps_uniform_anchors_and_remaps_gt_with_padded_descriptor():
    fake_reader = _TailBiasedFakeReader()
    module = _load_selector_module(fake_reader)
    selector = module.PCOTMRASPreBackboneFrameSelector(
        reader={"type": "PCOTMRASReader"},
        target_len=8,
        dense_window_size=16,
        descriptor_dim=14,
        protected_uniform_count=4,
        aux_value_loss_weight=0.01,
        aux_risk_loss_weight=0.01,
        aux_role_entropy_loss_weight=0.01,
    )
    inputs, masks, metas, gt_segments, gt_labels = _make_inputs()

    outputs = selector.forward_train(inputs, masks, metas, gt_segments, gt_labels)

    assert outputs["inputs"].shape == (2, 3, 8, 4, 4)
    assert outputs["masks"].shape == (2, 8)
    assert fake_reader.last_features.shape == (2, 16, 14)
    assert torch.count_nonzero(fake_reader.last_features[..., 12:]).item() == 0

    expected_anchors = {0, 5, 10, 15}
    for meta in outputs["metas"]:
        selected = meta["pc_ot_mras_prebackbone_selected_dense_indices"]
        assert selected == sorted(selected)
        assert len(selected) == len(set(selected)) == 8
        assert min(selected) >= 0
        assert max(selected) < 16
        assert expected_anchors.issubset(set(selected))

    assert outputs["gt_segments"]
    for segments in outputs["gt_segments"]:
        assert torch.all(segments >= 0.0)
        assert torch.all(segments <= 8.0)
        assert torch.all(segments[:, 1] > segments[:, 0])
    assert "selector_value_aux_loss" in outputs["losses"]
    assert "selector_risk_aux_loss" in outputs["losses"]
    assert "selector_role_entropy_loss" in outputs["losses"]


def test_train_forward_uses_hard_top1_frames_not_topk_mixed_frames():
    fake_reader = _SoftAmbiguousFakeReader()
    module = _load_selector_module(fake_reader)
    selector = module.PCOTMRASPreBackboneFrameSelector(
        reader={"type": "PCOTMRASReader"},
        target_len=4,
        dense_window_size=8,
        descriptor_dim=12,
        transport_topk=1,
        eval_transport_topk=1,
        protected_uniform_count=0,
        coverage_guard_count=0,
    )
    inputs, masks, metas, gt_segments, gt_labels = _make_time_index_inputs()

    outputs = selector.forward_train(inputs, masks, metas, gt_segments, gt_labels)

    selected = outputs["metas"][0]["pc_ot_mras_prebackbone_selected_dense_indices"]
    selected_values = outputs["inputs"][0, :, :, 0, 0].mean(dim=0)
    assert selected == [1, 3, 5, 7]
    assert torch.equal(selected_values, torch.tensor(selected, dtype=selected_values.dtype))


@pytest.mark.parametrize("bad_kwargs", [{"transport_topk": 2}, {"eval_transport_topk": 2}])
def test_transport_topk_knobs_fail_closed_to_hard_top1_only(bad_kwargs):
    module = _load_selector_module(_SoftAmbiguousFakeReader())

    with pytest.raises(ValueError, match="hard top-1 transport"):
        module.PCOTMRASPreBackboneFrameSelector(
            reader={"type": "PCOTMRASReader"},
            target_len=4,
            dense_window_size=8,
            descriptor_dim=12,
            protected_uniform_count=0,
            coverage_guard_count=0,
            **bad_kwargs,
        )


def test_c2_residual_slots_are_merged_with_uniform_scaffold_as_real_frames():
    fake_reader = _ResidualSlotsFakeReader()
    module = _load_selector_module(fake_reader)
    selector = module.PCOTMRASPreBackboneFrameSelector(
        reader={"type": "PCOTMRASReader"},
        target_len=8,
        dense_window_size=16,
        descriptor_dim=12,
        protected_uniform_count=4,
        coverage_guard_count=4,
        residual_count=4,
        residual_slot_role="learned_residual",
        selector_support_status="supported",
    )
    inputs, masks, metas, gt_segments, gt_labels = _make_time_index_inputs(dense_len=16)

    outputs = selector.forward_train(inputs, masks, metas, gt_segments, gt_labels)

    selected = outputs["metas"][0]["pc_ot_mras_prebackbone_selected_dense_indices"]
    selected_values = outputs["inputs"][0, :, :, 0, 0].mean(dim=0)
    assert selected == [0, 1, 3, 5, 6, 10, 12, 15]
    assert torch.equal(selected_values, torch.tensor(selected, dtype=selected_values.dtype))
    assert outputs["metas"][0]["pc_ot_mras_prebackbone_residual_count"] == 4
    assert outputs["metas"][0]["pc_ot_mras_prebackbone_selector_support_status"] == "supported"
    roles = outputs["metas"][0]["pc_ot_mras_prebackbone_selected_roles"]
    assert roles.count("uniform_protected") == 4
    assert roles.count("learned_residual") == 4


def test_slot_logits_drive_hard_selection_when_probability_matrix_is_conflicting():
    fake_reader = _ConflictingLogitsAndProbabilityReader()
    module = _load_selector_module(fake_reader)
    selector = module.PCOTMRASPreBackboneFrameSelector(
        reader={"type": "PCOTMRASReader"},
        target_len=4,
        dense_window_size=8,
        descriptor_dim=12,
        protected_uniform_count=0,
        coverage_guard_count=0,
    )
    inputs, masks, metas, gt_segments, gt_labels = _make_time_index_inputs()

    outputs = selector.forward_train(inputs, masks, metas, gt_segments, gt_labels)

    selected = outputs["metas"][0]["pc_ot_mras_prebackbone_selected_dense_indices"]
    selected_values = outputs["inputs"][0, :, :, 0, 0].mean(dim=0)
    assert selected == [4, 5, 6, 7]
    assert torch.equal(selected_values, torch.tensor(selected, dtype=selected_values.dtype))
    assert outputs["metas"][0]["pc_ot_mras_prebackbone_raw_slot_dense_indices"] == [4, 5, 6, 7]


def test_metadata_reports_raw_slot_collapse_before_repair_and_fill():
    fake_reader = _DuplicateSlotFakeReader()
    module = _load_selector_module(fake_reader)
    selector = module.PCOTMRASPreBackboneFrameSelector(
        reader={"type": "PCOTMRASReader"},
        target_len=4,
        dense_window_size=8,
        descriptor_dim=12,
        protected_uniform_count=0,
        coverage_guard_count=0,
    )
    inputs, masks, metas, gt_segments, gt_labels = _make_time_index_inputs()

    outputs = selector.forward_train(inputs, masks, metas, gt_segments, gt_labels)
    meta = outputs["metas"][0]

    assert meta["pc_ot_mras_prebackbone_duplicate_rate"] == 0.0
    assert meta["pc_ot_mras_prebackbone_raw_slot_dense_indices"] == [1, 1, 1, 7]
    assert meta["pc_ot_mras_prebackbone_raw_slot_unique_count"] == 2
    assert meta["pc_ot_mras_prebackbone_raw_slot_duplicate_rate"] == 0.5
    assert meta["pc_ot_mras_prebackbone_reader_fill_count"] == 2
    assert meta["pc_ot_mras_prebackbone_st_active_row_count"] == 2


def test_r_series_metadata_records_valid_len_gap_duplicates_and_boundary_hooks():
    fake_reader = _SoftAmbiguousFakeReader()
    module = _load_selector_module(fake_reader)
    selector = module.PCOTMRASPreBackboneFrameSelector(
        reader={"type": "PCOTMRASReader"},
        target_len=4,
        dense_window_size=8,
        descriptor_dim=12,
        protected_uniform_count=0,
        coverage_guard_count=0,
    )
    inputs, masks, metas, _gt_segments, _gt_labels = _make_time_index_inputs()

    outputs = selector.forward_test(inputs, masks, metas)
    meta = outputs["metas"][0]

    assert meta["pc_ot_mras_prebackbone_selected_dense_indices"] == [1, 3, 5, 7]
    assert meta["pc_ot_mras_prebackbone_valid_len"] == 8
    assert meta["pc_ot_mras_prebackbone_gap"] == [2, 2, 2]
    assert meta["pc_ot_mras_prebackbone_duplicate_rate"] == 0.0
    assert meta["pc_ot_mras_prebackbone_selection_unit"] == 1
    assert meta["pc_ot_mras_prebackbone_boundary_diagnostics"]["status"] == "placeholder"
    assert "score_rank_hook" in meta["pc_ot_mras_prebackbone_boundary_diagnostics"]


def test_detector_loss_backpropagates_to_reader_scores_through_st_transport():
    fake_reader = _TrainableSoftFakeReader()
    module = _load_selector_module(fake_reader)
    selector = module.PCOTMRASPreBackboneFrameSelector(
        reader={"type": "PCOTMRASReader"},
        target_len=4,
        dense_window_size=8,
        descriptor_dim=12,
        transport_topk=1,
        eval_transport_topk=1,
        protected_uniform_count=0,
        coverage_guard_count=0,
        straight_through_detector_loss=True,
    )
    inputs, masks, metas, gt_segments, gt_labels = _make_time_index_inputs()

    outputs = selector.forward_train(inputs, masks, metas, gt_segments, gt_labels)
    selected = outputs["metas"][0]["pc_ot_mras_prebackbone_selected_dense_indices"]
    selected_values = outputs["inputs"][0, :, :, 0, 0].mean(dim=0)
    assert selected == [1, 3, 5, 7]
    assert torch.equal(selected_values, torch.tensor(selected, dtype=selected_values.dtype))

    detector_like_loss = outputs["inputs"].sum()
    detector_like_loss.backward()

    assert fake_reader.logits.grad is not None
    assert torch.isfinite(fake_reader.logits.grad).all()
    assert fake_reader.logits.grad.abs().sum().item() > 0.0


def test_tiny_transformer_reader_emits_slot_logits_and_frame_acquisition_matrix():
    module = _load_selector_module(_TailBiasedFakeReader())
    reader = module.PCOTMRASTinyTransformerFrameScout(
        in_dim=5,
        hidden_dim=16,
        num_slots=3,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
    )
    features = torch.randn(2, 7, 5)
    valid = torch.tensor(
        [
            [True, True, True, True, True, True, True],
            [True, True, True, True, False, False, False],
        ],
        dtype=torch.bool,
    )
    time_coords = torch.linspace(0.0, 1.0, steps=7).unsqueeze(0).expand(2, -1)

    outputs = reader(features, valid, time_coords=time_coords)

    assert outputs["slot_logits"].shape == (2, 3, 7)
    assert outputs["acquisition_matrix"].shape == (2, 3, 7)
    assert torch.all(outputs["acquisition_matrix"][1, :, 4:] == 0.0)
    row_sums = outputs["acquisition_matrix"].sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1.0e-5)


def test_short_valid_windows_are_padded_and_masked_instead_of_crashing():
    fake_reader = _TailBiasedFakeReader()
    module = _load_selector_module(fake_reader)
    selector = module.PCOTMRASPreBackboneFrameSelector(
        reader={"type": "PCOTMRASReader"},
        target_len=8,
        dense_window_size=16,
        descriptor_dim=12,
        protected_uniform_count=4,
    )
    inputs, masks, metas, gt_segments, gt_labels = _make_inputs(batch=1, dense_len=16)
    masks[:, 5:] = False
    gt_segments = [torch.tensor([[0.0, 4.5]], dtype=torch.float32)]

    outputs = selector.forward_train(inputs, masks, metas, gt_segments, gt_labels[:1])

    assert outputs["inputs"].shape == (1, 3, 8, 4, 4)
    assert outputs["masks"].tolist() == [[True, True, True, True, True, False, False, False]]
    selected = outputs["metas"][0]["pc_ot_mras_prebackbone_selected_dense_indices"]
    assert len(selected) == 8
    assert selected[:5] == [0, 1, 2, 3, 4]
    assert selected[5:] == [4, 4, 4]
    assert outputs["metas"][0]["irregular_selected_positions"] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert outputs["metas"][0]["irregular_selected_output_valid_len"] == 5.0
    assert torch.all(outputs["gt_segments"][0] >= 0.0)
    assert torch.all(outputs["gt_segments"][0] <= 5.0)


@pytest.mark.parametrize("descriptor_dim", [6, 20])
def test_descriptor_truncation_and_padding_are_configurable(descriptor_dim):
    fake_reader = _TailBiasedFakeReader()
    module = _load_selector_module(fake_reader)
    selector = module.PCOTMRASPreBackboneFrameSelector(
        reader={"type": "PCOTMRASReader"},
        target_len=8,
        dense_window_size=16,
        descriptor_dim=descriptor_dim,
        protected_uniform_count=2,
    )
    inputs, masks, metas, _gt_segments, _gt_labels = _make_inputs(batch=1)

    outputs = selector.forward_test(inputs, masks, metas)

    assert outputs["inputs"].shape == (1, 3, 8, 4, 4)
    assert fake_reader.last_features.shape == (1, 16, descriptor_dim)
    selected = outputs["metas"][0]["pc_ot_mras_prebackbone_selected_dense_indices"]
    assert selected == sorted(selected)
    assert len(selected) == len(set(selected)) == 8
    assert min(selected) >= 0
    assert max(selected) < 16
