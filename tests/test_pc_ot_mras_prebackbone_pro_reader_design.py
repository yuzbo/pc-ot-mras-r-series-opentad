from __future__ import annotations

import importlib.util
import inspect
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


def _load_prebackbone_selector_module(fake_reader=None):
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
    builder.build_selector = (lambda _cfg: fake_reader) if fake_reader is not None else (lambda cfg: cfg)
    sys.modules["opentad.models.builder"] = builder

    spec = importlib.util.spec_from_file_location(
        "opentad.models.selectors.pc_ot_mras_prebackbone_frame_selector",
        SELECTOR_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _pro_reader_cls(module):
    accepted_names = (
        "PCOTMRASLowResPixelTemporalFrameReader",
        "PCOTMRASLowResolutionPixelTemporalFrameReader",
        "PCOTMRASLowResPixelTemporalFrameScout",
        "PCOTMRASRSeriesHybridFrameScout",
    )
    for name in accepted_names:
        reader_cls = getattr(module, name, None)
        if reader_cls is not None:
            return reader_cls
    raise AssertionError(f"missing Pro low-resolution pixel temporal reader; tried {accepted_names}")


def _instantiate_reader(reader_cls, *, in_dim: int, hidden_dim: int, num_slots: int):
    kwargs = {
        "in_dim": in_dim,
        "hidden_dim": hidden_dim,
        "num_slots": num_slots,
        "num_layers": 2,
        "num_heads": 2,
        "temporal_layers": 2,
        "temporal_kernel_size": 3,
        "kernel_size": 3,
        "dilations": (1, 2),
        "dropout": 0.0,
        "slot_mlp_layers": 2,
        "slot_temperature_init": 1.0,
    }
    signature = inspect.signature(reader_cls.__init__)
    filtered = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return reader_cls(**filtered)


def _make_time_index_inputs(torch, *, batch: int = 1, dense_len: int = 8):
    values = torch.arange(dense_len, dtype=torch.float32).view(1, 1, dense_len, 1, 1)
    inputs = values.expand(batch, 3, dense_len, 2, 2).contiguous()
    masks = torch.ones((batch, dense_len), dtype=torch.bool)
    metas = [{"sample_id": f"pro-reader-{idx}"} for idx in range(batch)]
    return inputs, masks, metas


class _TailClusterReader:
    def __call__(self, lowcost_features, valid_mask, time_coords=None):
        batch, time, _dim = lowcost_features.shape
        slots = 8
        logits = lowcost_features.new_full((batch, slots, time), -9.0)
        preferred = [28, 29, 30, 31, 31, 30, 29, 28]
        for slot, pos in enumerate(preferred):
            logits[:, slot, min(pos, time - 1)] = 9.0
        return {
            "slot_logits": logits,
            "acquisition_matrix": logits.softmax(dim=-1),
            "regularizers": {"total_regularizer": logits.sum() * 0.0},
        }


class _HeadDiagnosticReader:
    def __call__(self, lowcost_features, valid_mask, time_coords=None):
        batch, time, _dim = lowcost_features.shape
        slots = 4
        logits = lowcost_features.new_full((batch, slots, time), -8.0)
        for slot, pos in enumerate((1, 3, 5, 7)):
            logits[:, slot, min(pos, time - 1)] = 8.0
        frame_axis = lowcost_features.new_tensor(range(time)).unsqueeze(0).expand(batch, -1)
        return {
            "slot_logits": logits,
            "acquisition_matrix": logits.softmax(dim=-1),
            "actionness_logits": frame_axis,
            "start_logits": -frame_axis,
            "end_logits": frame_axis.flip(dims=(1,)),
            "uncertainty_logits": lowcost_features.new_zeros((batch, time)),
            "redundancy_logits": lowcost_features.new_zeros((batch, time)),
            "regularizers": {"total_regularizer": logits.sum() * 0.0},
        }


class _ConflictingSlotAndFrameScoreReader:
    def __call__(self, lowcost_features, valid_mask, time_coords=None):
        batch, time, _dim = lowcost_features.shape
        slots = 4
        slot_logits = lowcost_features.new_full((batch, slots, time), -10.0)
        for slot, pos in enumerate((7, 6, 5, 4)):
            slot_logits[:, slot, min(pos, time - 1)] = 10.0
        frame_scores = lowcost_features.new_full((batch, time), -10.0)
        for pos, score in zip((0, 1, 2, 3), (10.0, 9.0, 8.0, 7.0)):
            frame_scores[:, min(pos, time - 1)] = score
        return {
            "slot_logits": slot_logits,
            "acquisition_matrix": slot_logits.softmax(dim=-1),
            "frame_selection_logits": frame_scores,
            "actionness_logits": frame_scores,
            "start_logits": frame_scores,
            "end_logits": frame_scores,
            "uncertainty_logits": lowcost_features.new_zeros((batch, time)),
            "redundancy_logits": lowcost_features.new_zeros((batch, time)),
            "regularizers": {"total_regularizer": lowcost_features.sum() * 0.0},
        }


class _InvalidPaddingHighScoreReader:
    def __call__(self, lowcost_features, valid_mask, time_coords=None):
        batch, time, _dim = lowcost_features.shape
        slots = 4
        slot_logits = lowcost_features.new_zeros((batch, slots, time))
        frame_scores = lowcost_features.new_full((batch, time), -20.0)
        frame_scores[:, :4] = lowcost_features.new_tensor([4.0, 3.0, 2.0, 1.0])
        if time > 6:
            frame_scores[:, 6:] = 100.0
        return {
            "slot_logits": slot_logits,
            "acquisition_matrix": slot_logits.softmax(dim=-1),
            "frame_selection_logits": frame_scores,
            "actionness_logits": frame_scores,
            "start_logits": frame_scores,
            "end_logits": frame_scores,
            "uncertainty_logits": lowcost_features.new_zeros((batch, time)),
            "redundancy_logits": lowcost_features.new_zeros((batch, time)),
            "regularizers": {"total_regularizer": lowcost_features.sum() * 0.0},
        }


class _TrainableFrameScoreReader:
    def __init__(self, torch):
        self.frame_logits = torch.nn.Parameter(
            torch.tensor([10.0, 9.0, 8.0, 7.0, -6.0, -7.0, -8.0, -9.0], dtype=torch.float32)
        )

    def __call__(self, lowcost_features, valid_mask, time_coords=None):
        batch, time, _dim = lowcost_features.shape
        slots = 4
        slot_logits = lowcost_features.new_full((batch, slots, time), -8.0)
        for slot, pos in enumerate((7, 6, 5, 4)):
            slot_logits[:, slot, min(pos, time - 1)] = 8.0
        frame_scores = self.frame_logits[:time].to(device=lowcost_features.device).unsqueeze(0).expand(batch, -1)
        return {
            "slot_logits": slot_logits,
            "acquisition_matrix": slot_logits.softmax(dim=-1),
            "frame_selection_logits": frame_scores,
            "actionness_logits": frame_scores,
            "start_logits": frame_scores,
            "end_logits": frame_scores,
            "uncertainty_logits": lowcost_features.new_zeros((batch, time)),
            "redundancy_logits": lowcost_features.new_zeros((batch, time)),
            "regularizers": {"total_regularizer": frame_scores.sum() * 0.0},
        }


def test_pro_reader_emits_action_start_end_uncertainty_redundancy_heads_and_slot_matrix():
    torch = _import_torch_or_skip()
    module = _load_prebackbone_selector_module()
    reader_cls = _pro_reader_cls(module)
    batch, time, channels, slots = 2, 9, 3 * 4 * 4, 4
    torch.manual_seed(20260624)
    reader = _instantiate_reader(reader_cls, in_dim=channels, hidden_dim=16, num_slots=slots)
    features = torch.randn(batch, time, channels, requires_grad=True)
    valid = torch.tensor(
        [
            [True, True, True, True, True, True, True, True, True],
            [True, True, True, True, True, False, False, False, False],
        ],
        dtype=torch.bool,
    )
    time_coords = torch.linspace(0.0, 1.0, steps=time).unsqueeze(0).expand(batch, -1)

    outputs = reader(features, valid, time_coords=time_coords)

    required_heads = (
        "actionness_logits",
        "start_logits",
        "end_logits",
        "uncertainty_logits",
        "redundancy_logits",
    )
    missing_heads = [key for key in required_heads if key not in outputs]
    assert missing_heads == [], f"missing required Pro reader frame head(s): {missing_heads}"
    assert outputs["slot_logits"].shape == (batch, slots, time)
    assert outputs["acquisition_matrix"].shape == (batch, slots, time)
    assert torch.all(outputs["acquisition_matrix"][1, :, 5:] == 0.0)
    row_sums = outputs["acquisition_matrix"].sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1.0e-5)
    for key in required_heads:
        assert outputs[key].shape == (batch, time), key
        assert torch.isfinite(outputs[key][valid]).all(), key
        assert torch.all(outputs[key][~valid] == 0.0), key
    regularizers = outputs.get("regularizers", {})
    for key in ("soft_order_regularizer", "duplicate_mass_regularizer", "total_regularizer"):
        assert key in regularizers, f"missing learned temporal geometry regularizer {key}"
        assert torch.isfinite(regularizers[key]), key


def test_frame_score_first_selector_ignores_conflicting_slot_logits_for_hard_selection():
    torch = _import_torch_or_skip()
    module = _load_prebackbone_selector_module(_ConflictingSlotAndFrameScoreReader())
    selector = module.PCOTMRASPreBackboneFrameSelector(
        reader={"type": "ConflictReader"},
        target_len=4,
        dense_window_size=8,
        descriptor_dim=12,
        selection_strategy="frame_score_topk",
        protected_uniform_count=0,
        coverage_guard_count=0,
        max_gap=0,
    )
    inputs, masks, metas = _make_time_index_inputs(torch, dense_len=8)

    outputs = selector.forward_test(inputs, masks, metas)

    selected = outputs["metas"][0]["pc_ot_mras_prebackbone_selected_dense_indices"]
    assert selected == [0, 1, 2, 3]
    assert all(role == "frame_score_topk" for role in outputs["metas"][0]["pc_ot_mras_prebackbone_selected_roles"])
    assert outputs["metas"][0]["pc_ot_mras_prebackbone_selection_strategy"] == "frame_score_topk"
    assert outputs["metas"][0]["pc_ot_mras_prebackbone_hard_selection_source"] == "frame_selection_logits"
    assert outputs["metas"][0]["pc_ot_mras_prebackbone_slot_not_hard_source"] is True


def test_frame_score_first_train_and_eval_hard_indices_are_identical():
    torch = _import_torch_or_skip()
    module = _load_prebackbone_selector_module(_ConflictingSlotAndFrameScoreReader())
    selector = module.PCOTMRASPreBackboneFrameSelector(
        reader={"type": "ConflictReader"},
        target_len=4,
        dense_window_size=8,
        descriptor_dim=12,
        selection_strategy="frame_score_topk",
        protected_uniform_count=0,
        coverage_guard_count=0,
        max_gap=0,
    )
    inputs, masks, _metas = _make_time_index_inputs(torch, dense_len=8)

    train_outputs = selector.forward_train(inputs, masks, [{"split": "train"}], gt_segments=None, gt_labels=None)
    test_outputs = selector.forward_test(inputs, masks, [{"split": "test"}])

    train_selected = train_outputs["metas"][0]["pc_ot_mras_prebackbone_selected_dense_indices"]
    test_selected = test_outputs["metas"][0]["pc_ot_mras_prebackbone_selected_dense_indices"]
    assert train_selected == test_selected == [0, 1, 2, 3]


def test_frame_score_first_masks_invalid_padding_even_when_invalid_scores_are_high():
    torch = _import_torch_or_skip()
    module = _load_prebackbone_selector_module(_InvalidPaddingHighScoreReader())
    selector = module.PCOTMRASPreBackboneFrameSelector(
        reader={"type": "InvalidPaddingHighScoreReader"},
        target_len=4,
        dense_window_size=8,
        descriptor_dim=12,
        selection_strategy="frame_score_topk",
        protected_uniform_count=0,
        coverage_guard_count=0,
        max_gap=0,
    )
    inputs, masks, metas = _make_time_index_inputs(torch, dense_len=8)
    masks[:, 6:] = False

    outputs = selector.forward_test(inputs, masks, metas)

    selected = outputs["metas"][0]["pc_ot_mras_prebackbone_selected_dense_indices"]
    assert selected == [0, 1, 2, 3]
    assert max(selected) < 6


def test_frame_score_first_st_surrogate_backpropagates_detector_loss_to_frame_logits():
    torch = _import_torch_or_skip()
    reader = _TrainableFrameScoreReader(torch)
    module = _load_prebackbone_selector_module(reader)
    selector = module.PCOTMRASPreBackboneFrameSelector(
        reader={"type": "TrainableFrameScoreReader"},
        target_len=4,
        dense_window_size=8,
        descriptor_dim=12,
        selection_strategy="frame_score_topk",
        protected_uniform_count=0,
        coverage_guard_count=0,
        max_gap=0,
    )
    inputs, masks, metas = _make_time_index_inputs(torch, dense_len=8)
    inputs = inputs.clone().requires_grad_(True)

    outputs = selector.forward_train(inputs, masks, metas, gt_segments=None, gt_labels=None)
    detector_loss = outputs["inputs"].square().mean()
    detector_loss.backward()

    assert reader.frame_logits.grad is not None
    assert torch.isfinite(reader.frame_logits.grad).all()
    assert reader.frame_logits.grad.abs().sum().item() > 0.0


def test_frame_score_first_auxiliary_gt_loss_trains_frame_scores_not_slot_matrix():
    torch = _import_torch_or_skip()
    module = _load_prebackbone_selector_module()
    selector = module.PCOTMRASPreBackboneFrameSelector.__new__(module.PCOTMRASPreBackboneFrameSelector)
    selector.selection_strategy = "frame_score_topk"
    selector.aux_gt_acquisition_loss_weight = 1.0
    selector.aux_duplicate_cap_loss_weight = 0.0
    selector.aux_value_loss_weight = 0.0
    selector.aux_risk_loss_weight = 0.0
    selector.aux_uncertainty_loss_weight = 0.0
    selector.aux_redundancy_loss_weight = 0.0
    selector.aux_role_entropy_loss_weight = 0.0
    selector.reader_regularizer_loss_weight = 0.0

    frame_scores = torch.zeros((1, 8), dtype=torch.float32, requires_grad=True)
    slot_matrix = torch.full((1, 4, 8), 0.125, dtype=torch.float32, requires_grad=True)
    valid = torch.ones((1, 8), dtype=torch.bool)
    candidate_dense_indices = torch.arange(8, dtype=torch.long).unsqueeze(0)
    gt_segments = [torch.tensor([[0.0, 4.0]], dtype=torch.float32)]

    losses = selector._losses(
        reader_outputs={
            "frame_selection_logits": frame_scores,
            "acquisition_matrix": slot_matrix,
        },
        valid_mask=valid,
        candidate_dense_indices=candidate_dense_indices,
        gt_segments=gt_segments,
    )

    assert "selector_gt_frame_score_loss" in losses
    assert "selector_gt_acquisition_loss" not in losses
    losses["selector_gt_frame_score_loss"].backward()
    assert frame_scores.grad is not None
    assert torch.isfinite(frame_scores.grad).all()
    assert frame_scores.grad.abs().sum().item() > 0.0
    assert slot_matrix.grad is None or slot_matrix.grad.abs().sum().item() == 0.0


def test_selector_max_gap_coverage_guard_keeps_hard_indices_ordered_unique_and_bounded():
    torch = _import_torch_or_skip()
    module = _load_prebackbone_selector_module(_TailClusterReader())
    selector = module.PCOTMRASPreBackboneFrameSelector(
        reader={"type": "ProTailClusterReader"},
        target_len=8,
        dense_window_size=32,
        descriptor_dim=12,
        protected_uniform_count=0,
        coverage_guard_count=2,
        max_gap=5,
    )
    inputs, masks, metas = _make_time_index_inputs(torch, dense_len=32)

    outputs = selector.forward_test(inputs, masks, metas)

    selected = outputs["metas"][0]["pc_ot_mras_prebackbone_selected_dense_indices"]
    assert selected == sorted(selected)
    assert len(selected) == len(set(selected)) == 8
    gaps = [right - left for left, right in zip(selected[:-1], selected[1:])]
    assert max(gaps) <= 5
    selected_values = outputs["inputs"][0, :, :, 0, 0].mean(dim=0)
    assert torch.equal(selected_values, torch.tensor(selected, dtype=selected_values.dtype))


def test_metadata_exposes_action_boundary_uncertainty_redundancy_and_head_diagnostics():
    torch = _import_torch_or_skip()
    module = _load_prebackbone_selector_module(_HeadDiagnosticReader())
    selector = module.PCOTMRASPreBackboneFrameSelector(
        reader={"type": "ProHeadDiagnosticReader"},
        target_len=4,
        dense_window_size=8,
        descriptor_dim=12,
        protected_uniform_count=0,
        coverage_guard_count=0,
    )
    inputs, masks, metas = _make_time_index_inputs(torch, dense_len=8)

    outputs = selector.forward_test(inputs, masks, metas)

    diagnostics = outputs["metas"][0].get("pc_ot_mras_prebackbone_reader_diagnostics")
    assert isinstance(diagnostics, dict), "metadata must expose reader diagnostics"
    for key in ("action", "boundary", "uncertainty", "redundancy", "head"):
        assert key in diagnostics
        assert diagnostics[key].get("available") is True


def test_metadata_records_no_p2_raw_cache_teacher_or_test_gt_protocol_flags():
    torch = _import_torch_or_skip()
    module = _load_prebackbone_selector_module(_HeadDiagnosticReader())
    selector = module.PCOTMRASPreBackboneFrameSelector(
        reader={"type": "ProHeadDiagnosticReader"},
        target_len=4,
        dense_window_size=8,
        descriptor_dim=12,
        protected_uniform_count=0,
        coverage_guard_count=0,
    )
    inputs, masks, metas = _make_time_index_inputs(torch, dense_len=8)

    outputs = selector.forward_test(inputs, masks, metas)

    protocol_flags = outputs["metas"][0].get("pc_ot_mras_prebackbone_protocol_flags")
    assert protocol_flags == {
        "uses_p2": False,
        "uses_raw_prediction_cache": False,
        "uses_teacher": False,
        "uses_test_gt": False,
    }
