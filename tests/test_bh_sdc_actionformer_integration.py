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
        if type_name not in self._items:
            raise KeyError(type_name)
        return self._items[type_name](**cfg)


def _ensure_package(name, path):
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    return module


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _SyntheticScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))


class _SyntheticAffineDropPath(_SyntheticScale):
    pass


def _install_runtime():
    for name in (
        "opentad.models.builder",
        "opentad.models.utils.post_processing",
        "opentad.models.backbones",
        "opentad.models.bricks",
        "opentad.models.detectors.base",
        "opentad.models.detectors.single_stage",
        "opentad.models.detectors.actionformer",
        "opentad.models.selectors.bh_sdc_frame_selector",
    ):
        sys.modules.pop(name, None)

    _ensure_package("opentad", ROOT / "opentad")
    _ensure_package("opentad.models", ROOT / "opentad" / "models")
    _ensure_package("opentad.models.detectors", ROOT / "opentad" / "models" / "detectors")
    _ensure_package("opentad.models.selectors", ROOT / "opentad" / "models" / "selectors")
    _ensure_package("opentad.models.utils", ROOT / "opentad" / "models" / "utils")

    post_processing = types.ModuleType("opentad.models.utils.post_processing")
    post_processing.load_predictions = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("no raw cache"))
    post_processing.save_predictions = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("no raw cache"))
    post_processing.batched_nms = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("no nms"))
    post_processing.convert_to_seconds = lambda segments, meta: segments
    sys.modules["opentad.models.utils.post_processing"] = post_processing

    bricks = types.ModuleType("opentad.models.bricks")
    bricks.Scale = _SyntheticScale
    bricks.AffineDropPath = _SyntheticAffineDropPath
    sys.modules["opentad.models.bricks"] = bricks

    builder = types.ModuleType("opentad.models.builder")
    builder.MODELS = _Registry()
    builder.SELECTORS = builder.MODELS
    builder.TOKEN_COMPRESSORS = builder.MODELS
    builder.DETECTORS = builder.MODELS
    builder.PROJECTIONS = builder.MODELS
    builder.HEADS = builder.MODELS
    builder.NECKS = builder.MODELS
    builder.build_detector = lambda cfg: builder.DETECTORS.build(cfg)
    builder.build_backbone = lambda cfg: builder.MODELS.build(cfg)
    builder.build_projection = lambda cfg: builder.PROJECTIONS.build(cfg)
    builder.build_selector = lambda cfg: builder.SELECTORS.build(cfg)
    builder.build_token_compressor = lambda cfg: builder.TOKEN_COMPRESSORS.build(cfg)
    builder.build_neck = lambda cfg: builder.NECKS.build(cfg)
    builder.build_head = lambda cfg: builder.HEADS.build(cfg)
    sys.modules["opentad.models.builder"] = builder

    class TailSensitiveTemporalMixingBackbone(nn.Module):
        def __init__(self, channels=1):
            super().__init__()
            self.channels = int(channels)
            self.calls = []

        def forward(self, inputs, masks=None):
            self.calls.append((int(inputs.shape[-1]), None if masks is None else masks.detach().cpu().tolist()))
            out = inputs.float().clone()
            if masks is None and inputs.shape[-1] > 4:
                out[..., :4] = out[..., :4] + 1000.0
            if masks is not None:
                out = out * masks[:, None, :].to(dtype=out.dtype)
            return out

    class IdentityProjection(nn.Module):
        def __init__(self, max_seq_len=8):
            super().__init__()
            self.n_mha_win_size = 1
            self.arch = (0,)
            self.max_seq_len = int(max_seq_len)

        def forward(self, x, masks):
            return (x,), (masks,)

    class CapturingHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.prior_generator = types.SimpleNamespace(strides=[1])
            self.last_features = None
            self.last_masks = None
            self.last_metas = None

        def forward_train(self, feat_list, mask_list, gt_segments, gt_labels, metas=None):
            self.last_features = feat_list[0].detach().clone()
            self.last_masks = mask_list[0].detach().clone()
            self.last_metas = metas
            return {"detector_loss": feat_list[0].square().mean()}

        def forward_test(self, feat_list, mask_list, metas=None):
            self.last_features = feat_list[0].detach().clone()
            self.last_masks = mask_list[0].detach().clone()
            self.last_metas = metas
            return [torch.empty(0, 2)], [torch.empty(0, 1)]

    builder.MODELS.register_module()(TailSensitiveTemporalMixingBackbone)
    builder.PROJECTIONS.register_module()(IdentityProjection)
    builder.HEADS.register_module()(CapturingHead)

    selector_module = _load_module(
        "opentad.models.selectors.bh_sdc_frame_selector",
        ROOT / "opentad" / "models" / "selectors" / "bh_sdc_frame_selector.py",
    )
    _load_module("opentad.models.detectors.base", ROOT / "opentad" / "models" / "detectors" / "base.py")
    _load_module("opentad.models.detectors.single_stage", ROOT / "opentad" / "models" / "detectors" / "single_stage.py")
    actionformer_module = _load_module(
        "opentad.models.detectors.actionformer",
        ROOT / "opentad" / "models" / "detectors" / "actionformer.py",
    )
    return actionformer_module.ActionFormer, selector_module, builder


ActionFormer, bh_sdc_module, runtime_builder = _install_runtime()


class _ValueScout(nn.Module):
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
            "protocol": bh_sdc_module._deploy_protocol_flags(),
        }


class _FixedBudget(nn.Module):
    def forward(self, scout_out, valid_mask):
        return torch.full((valid_mask.shape[0],), 4, dtype=torch.long), {
            "min_budget": 2,
            "max_budget": 6,
            "protocol": "unit_fixed_budget",
            "uses_gt": False,
            "uses_teacher": False,
            "uses_raw_prediction_cache": False,
        }


def _bh_sdc_model():
    cfg = dict(
        type="ActionFormer",
        backbone=dict(type="TailSensitiveTemporalMixingBackbone", channels=1),
        projection=dict(type="IdentityProjection", max_seq_len=8),
        rpn_head=dict(type="CapturingHead"),
        frame_selector=dict(
            type="PCOTMRASBoundaryHazardSparseDenseFrameSelector",
            input_channels=1,
            dense_window_size=8,
            min_budget=2,
            target_budget=4,
            max_budget=6,
            budget_step=1,
            scout_hidden_dim=4,
            scout_num_layers=1,
            probe_stride=2,
            coverage_ratio=0.25,
            boundary_ratio=0.50,
            max_dense_gap=3,
            aux_hazard_loss_weight=0.0,
            aux_budget_entropy_loss_weight=0.0,
        ),
        token_compressor=dict(
            type="PCOTMRASBoundaryHazardSparseToDenseBridge",
            dense_window_size=8,
            target_len=8,
            interpolation_temperature=1.0,
            refine_layers=0,
            smoothness_loss_weight=0.0,
        ),
    )
    model = runtime_builder.build_detector(cfg)
    model.frame_selector.scout = _ValueScout()
    model.frame_selector.budget_controller = _FixedBudget()
    return model


def test_bh_sdc_build_detector_constructs_selector_bridge_projection_and_head_smoke():
    model = _bh_sdc_model()

    assert type(model.frame_selector).__name__ == "PCOTMRASBoundaryHazardSparseDenseFrameSelector"
    assert type(model.token_compressor).__name__ == "PCOTMRASBoundaryHazardSparseToDenseBridge"
    assert type(model.projection).__name__ == "IdentityProjection"
    assert type(model.rpn_head).__name__ == "CapturingHead"

    inputs = torch.arange(8, dtype=torch.float32).view(1, 1, 8)
    masks = torch.ones(1, 8, dtype=torch.bool)
    metas = [{"sample_id": "bh-sdc-build-detector"}]
    gt_segments = [torch.tensor([[1.0, 6.0]])]
    gt_labels = [torch.tensor([1])]

    losses = model.forward_train(inputs, masks, metas=metas, gt_segments=gt_segments, gt_labels=gt_labels)

    assert torch.isfinite(losses["cost"])
    assert model.backbone.calls == [(4, [[True, True, True, True]])]
    assert model.rpn_head.last_features.shape[-1] == 8
    assert model.rpn_head.last_metas[0]["bh_sdc_completion"]["route_label"] == BH_SDC_ROUTE_LABEL


def test_bh_sdc_actionformer_compacts_before_temporal_mixing_backbone_and_preserves_metadata():
    model = _bh_sdc_model()
    inputs = torch.arange(8, dtype=torch.float32).view(1, 1, 8)
    masks = torch.ones(1, 8, dtype=torch.bool)
    metas = [{"sample_id": "bh-sdc-actionformer", "physical_time_axis": [float(i) * 0.5 for i in range(8)]}]
    gt_segments = [torch.tensor([[1.0, 6.0]])]
    gt_labels = [torch.tensor([1])]

    losses = model.forward_train(inputs, masks, metas=metas, gt_segments=gt_segments, gt_labels=gt_labels)

    assert torch.isfinite(losses["cost"])
    assert model.backbone.calls == [(4, [[True, True, True, True]])]
    captured = model.rpn_head.last_metas[0]
    plan = captured["bh_sdc_acquisition_plan"]
    selected = plan["selected_dense_indices"][: plan["selected_count"]]
    features = model.rpn_head.last_features
    for sparse_col, dense_idx in enumerate(selected):
        assert torch.allclose(features[0, :, dense_idx], inputs[0, :, dense_idx], atol=1e-6)
    assert captured["bh_sdc_completion"]["observed_mask"] == [idx in selected for idx in range(8)]
    assert captured["bh_sdc_completion"]["synthetic_mask"] == [idx not in selected for idx in range(8)]
    assert captured["bh_sdc_detector_metadata"]["route_label"] == BH_SDC_ROUTE_LABEL


def test_bh_sdc_refine_scale_is_in_optimizer_no_decay_group_once():
    model = _bh_sdc_model()
    model.token_compressor.refine = torch.nn.Conv1d(1, 1, kernel_size=1)
    model.token_compressor.refine_scale = torch.nn.Parameter(torch.zeros(()))

    optim_groups = model.get_optim_groups({"weight_decay": 0.05, "lr": 1e-4})

    refine_scale = model.token_compressor.refine_scale
    memberships = [
        group
        for group in optim_groups
        if any(param is refine_scale for param in group["params"])
    ]
    assert len(memberships) == 1
    assert memberships[0]["weight_decay"] == 0.0


def test_non_bh_sdc_actionformer_keeps_existing_backbone_call_contract():
    model = ActionFormer(
        backbone=dict(type="TailSensitiveTemporalMixingBackbone", channels=1),
        projection=dict(type="IdentityProjection", max_seq_len=8),
        rpn_head=dict(type="CapturingHead"),
    )
    inputs = torch.arange(8, dtype=torch.float32).view(1, 1, 8)
    masks = torch.ones(1, 8, dtype=torch.bool)
    gt_segments = [torch.tensor([[1.0, 6.0]])]
    gt_labels = [torch.tensor([1])]

    model.forward_train(inputs, masks, metas=[{}], gt_segments=gt_segments, gt_labels=gt_labels)

    assert model.backbone.calls == [(8, None)]
    assert torch.allclose(model.rpn_head.last_features[0, :, :4], inputs[0, :, :4] + 1000.0)
