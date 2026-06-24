from __future__ import annotations

import importlib.util
import runpy
import subprocess
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SELECTOR_PATH = ROOT / "opentad" / "models" / "selectors" / "pc_ot_mras_prebackbone_frame_selector.py"
C3_PRO_BOUNDARY_CONFIG = (
    ROOT
    / "configs"
    / "adatad"
    / "thumos"
    / "pc_ot_mras_prebackbone_c3_pro_boundary_reader_full_train_candidate_n16r4.py"
)


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


def _make_time_index_inputs(torch, *, dense_len: int = 8, value_scale: float = 1.0):
    values = torch.arange(dense_len, dtype=torch.float32).view(1, 1, dense_len, 1, 1) * float(value_scale)
    inputs = values.expand(1, 3, dense_len, 2, 2).contiguous()
    masks = torch.ones((1, dense_len), dtype=torch.bool)
    metas = [{"sample_id": "c3-pro-stability"}]
    return inputs, masks, metas


class _ExtremeFrameScoreReader:
    def __init__(self, torch, logits):
        self.frame_logits = torch.nn.Parameter(torch.tensor(logits, dtype=torch.float32))

    def __call__(self, lowcost_features, valid_mask, time_coords=None):
        batch, time, _dim = lowcost_features.shape
        slot_logits = lowcost_features.new_zeros((batch, 4, time))
        frame_scores = self.frame_logits[:time].to(device=lowcost_features.device).unsqueeze(0).expand(batch, -1)
        zeros = lowcost_features.new_zeros((batch, time))
        return {
            "slot_logits": slot_logits,
            "acquisition_matrix": slot_logits.softmax(dim=-1),
            "frame_selection_logits": frame_scores,
            "actionness_logits": frame_scores,
            "start_logits": frame_scores,
            "end_logits": frame_scores,
            "uncertainty_logits": zeros,
            "redundancy_logits": zeros,
            "regularizers": {"total_regularizer": frame_scores.sum() * 0.0},
        }


def _c3_stable_selector(module, *, reader):
    return module.PCOTMRASPreBackboneFrameSelector(
        reader={"type": "ExtremeFrameScoreReader"},
        target_len=4,
        dense_window_size=8,
        descriptor_dim=12,
        selection_strategy="frame_score_topk",
        protected_uniform_count=0,
        coverage_guard_count=0,
        max_gap=0,
        frame_score_st_temperature=4.0,
        frame_score_st_local_width=16.0,
        frame_score_st_local_bias_weight=0.25,
        frame_score_st_logit_clamp=12.0,
        frame_score_st_gradient_scale=0.05,
        frame_score_aux_logit_clamp=12.0,
    )


def test_c3_extreme_frame_score_st_and_aux_backward_keeps_gradients_finite_and_bounded():
    torch = _import_torch_or_skip()
    reader = _ExtremeFrameScoreReader(torch, [40.0, 35.0, 30.0, 25.0, -20.0, -25.0, -30.0, -35.0])
    module = _load_prebackbone_selector_module(reader)
    selector = _c3_stable_selector(module, reader=reader)
    inputs, masks, metas = _make_time_index_inputs(torch, dense_len=8, value_scale=0.25)
    inputs = inputs.requires_grad_(True)
    gt_segments = [torch.tensor([[0.0, 4.0]], dtype=torch.float32)]
    gt_labels = [torch.tensor([0], dtype=torch.long)]

    outputs = selector.forward_train(inputs, masks, metas, gt_segments=gt_segments, gt_labels=gt_labels)
    selected = outputs["metas"][0]["pc_ot_mras_prebackbone_selected_dense_indices"]
    assert selected == [0, 1, 2, 3]
    assert torch.isfinite(outputs["losses"]["selector_gt_frame_score_loss"])

    detector_loss = outputs["inputs"].square().mean() * 100.0
    total_loss = detector_loss + sum(outputs["losses"].values())
    total_loss.backward()

    assert reader.frame_logits.grad is not None
    assert torch.isfinite(reader.frame_logits.grad).all()
    assert reader.frame_logits.grad.abs().max().item() < 1.0e3
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


def test_c3_stability_surrogate_does_not_change_hard_frame_score_topk_indices():
    torch = _import_torch_or_skip()
    reader = _ExtremeFrameScoreReader(torch, [1000.0, 999.0, 998.0, 997.0, 1004.0, 1003.0, 1002.0, 1001.0])
    module = _load_prebackbone_selector_module(reader)
    selector = _c3_stable_selector(module, reader=reader)
    inputs, masks, metas = _make_time_index_inputs(torch, dense_len=8)

    outputs = selector.forward_test(inputs, masks, metas)

    selected = outputs["metas"][0]["pc_ot_mras_prebackbone_selected_dense_indices"]
    assert selected == [4, 5, 6, 7]
    assert outputs["metas"][0]["pc_ot_mras_prebackbone_hard_selection_source"] == "frame_selection_logits"
    assert outputs["metas"][0]["pc_ot_mras_prebackbone_slot_not_hard_source"] is True


def test_c3_pro_boundary_config_uses_stability_parameters_and_remains_pro_boundary():
    config_globals = runpy.run_path(str(C3_PRO_BOUNDARY_CONFIG))
    frame_selector = config_globals["model"]["frame_selector"]

    assert config_globals["route_id"] == "pc_ot_mras_prebackbone_c3_pro_boundary_reader"
    assert frame_selector["selection_strategy"] == "frame_score_topk"
    assert frame_selector["reader"]["type"] == "PCOTMRASBoundaryDifficultyTemporalFrameScout"
    assert frame_selector["reader_regularizer_loss_weight"] == 0.0
    assert frame_selector["frame_score_st_temperature"] >= 4.0
    assert frame_selector["frame_score_st_local_width"] >= 12.0
    assert 0.0 <= frame_selector["frame_score_st_local_bias_weight"] <= 0.50
    assert 0.0 < frame_selector["frame_score_st_gradient_scale"] <= 0.25
    assert 0.0 < frame_selector["frame_score_st_logit_clamp"] <= 20.0
    assert 0.0 < frame_selector["frame_score_aux_logit_clamp"] <= 20.0


def test_c3_frame_score_topk_never_uses_slot_transport_as_hard_source():
    torch = _import_torch_or_skip()
    reader = _ExtremeFrameScoreReader(torch, [4.0, 3.0, 2.0, 1.0, -1.0, -2.0, -3.0, -4.0])
    module = _load_prebackbone_selector_module(reader)
    selector = _c3_stable_selector(module, reader=reader)
    inputs, masks, metas = _make_time_index_inputs(torch, dense_len=8)

    outputs = selector.forward_test(inputs, masks, metas)

    selected = outputs["metas"][0]["pc_ot_mras_prebackbone_selected_dense_indices"]
    assert selected == [0, 1, 2, 3]
    assert outputs["metas"][0]["pc_ot_mras_prebackbone_selection_strategy"] == "frame_score_topk"
    assert outputs["metas"][0]["pc_ot_mras_prebackbone_hard_selection_source"] != "slot_transport"
