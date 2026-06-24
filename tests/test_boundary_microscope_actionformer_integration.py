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
ROUTE_LABEL = "DIVERGENT_INNOVATION_BOUNDARY_MICROSCOPE_DO_NOT_MERGE_WITH_C3"


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

    def forward(self, x):
        return x * self.scale


class _SyntheticAffineDropPath(_SyntheticScale):
    pass


class _SyntheticVideoMAEBackbone(nn.Module):
    def __init__(self, channels=4):
        super().__init__()
        self.channels = int(channels)
        self.calls = []
        self.last_inputs = None

    def forward(self, inputs):
        self.calls.append(tuple(inputs.shape))
        self.last_inputs = inputs.detach().clone()
        if inputs.ndim != 6:
            raise AssertionError(f"synthetic VideoMAE backbone expects [B,N,C,T,H,W], got {tuple(inputs.shape)}")
        features = inputs.to(dtype=torch.float32).mean(dim=(1, 2, 4, 5))[:, None, :]
        return features.expand(-1, self.channels, -1).contiguous()


class _SyntheticIdentityProjection(nn.Module):
    def __init__(self, channels=4, max_seq_len=64):
        super().__init__()
        self.n_mha_win_size = 1
        self.arch = (0,)
        self.max_seq_len = int(max_seq_len)
        self.proj = nn.Conv1d(int(channels), int(channels), kernel_size=1, bias=False)
        nn.init.eye_(self.proj.weight[:, :, 0])

    def forward(self, x, masks):
        return (self.proj(x),), (masks,)


class _SyntheticRPNHead(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        self.prior_generator = types.SimpleNamespace(strides=[1])
        self.score = nn.Conv1d(int(in_channels), 1, kernel_size=1)
        self.last_train_call = None

    def forward_train(self, feat_list, mask_list, gt_segments, gt_labels, metas=None):
        feat = feat_list[0]
        mask = mask_list[0]
        self.last_train_call = {
            "feat_shape": tuple(feat.shape),
            "mask_shape": tuple(mask.shape),
            "mask": mask.detach().clone(),
            "metas": metas,
        }
        valid_feat = feat.masked_fill(~mask[:, None, :], 0.0)
        return {"detector_loss": self.score(valid_feat).square().mean()}


def _install_boundary_actionformer_runtime():
    for name in (
        "opentad.models.builder",
        "opentad.models.bricks",
        "opentad.models.backbones",
        "opentad.models.utils.post_processing",
        "opentad.models.selectors.boundary_microscope_acquisition_route",
        "opentad.models.detectors.base",
        "opentad.models.detectors.single_stage",
        "opentad.models.detectors.actionformer",
    ):
        sys.modules.pop(name, None)

    _ensure_package("opentad", ROOT / "opentad")
    _ensure_package("opentad.models", ROOT / "opentad" / "models")
    _ensure_package("opentad.models.bricks", ROOT / "opentad" / "models" / "bricks")
    _ensure_package("opentad.models.detectors", ROOT / "opentad" / "models" / "detectors")
    _ensure_package("opentad.models.selectors", ROOT / "opentad" / "models" / "selectors")
    _ensure_package("opentad.models.utils", ROOT / "opentad" / "models" / "utils")

    bricks = sys.modules["opentad.models.bricks"]
    bricks.Scale = _SyntheticScale
    bricks.AffineDropPath = _SyntheticAffineDropPath

    backbones = types.ModuleType("opentad.models.backbones")
    backbones.BackboneWrapper = lambda cfg: (_ for _ in ()).throw(
        RuntimeError("Boundary-Microscope synthetic integration must not build mmaction backbone")
    )
    sys.modules["opentad.models.backbones"] = backbones

    post_processing = types.ModuleType("opentad.models.utils.post_processing")
    post_processing.load_predictions = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("Boundary-Microscope integration must not load raw predictions")
    )
    post_processing.save_predictions = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("Boundary-Microscope integration must not save raw predictions")
    )
    post_processing.batched_nms = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("Boundary-Microscope integration must not run NMS")
    )
    post_processing.convert_to_seconds = lambda segments, meta: segments
    sys.modules["opentad.models.utils.post_processing"] = post_processing

    builder = types.ModuleType("opentad.models.builder")
    builder.MODELS = _Registry()
    builder.SELECTORS = builder.MODELS
    builder.PROJECTIONS = builder.MODELS
    builder.HEADS = builder.MODELS
    builder.NECKS = builder.MODELS
    builder.DETECTORS = builder.MODELS
    builder.TOKEN_COMPRESSORS = builder.MODELS
    builder.PRIOR_GENERATORS = builder.MODELS
    builder.ROI_EXTRACTORS = builder.MODELS
    builder.PROPOSAL_GENERATORS = builder.MODELS
    builder.TRANSFORMERS = builder.MODELS
    builder.LOSSES = builder.MODELS
    builder.MATCHERS = builder.MODELS
    builder.build_backbone = lambda cfg: backbones.BackboneWrapper(cfg)
    builder.build_projection = lambda cfg: builder.PROJECTIONS.build(cfg)
    builder.build_selector = lambda cfg: builder.SELECTORS.build(cfg)
    builder.build_token_compressor = lambda cfg: None if cfg is None else builder.TOKEN_COMPRESSORS.build(cfg)
    builder.build_neck = lambda cfg: builder.NECKS.build(cfg)
    builder.build_head = lambda cfg: builder.HEADS.build(cfg)
    sys.modules["opentad.models.builder"] = builder

    builder.PROJECTIONS.register_module()(_SyntheticIdentityProjection)
    builder.HEADS.register_module()(_SyntheticRPNHead)

    _load_module(
        "opentad.models.selectors.boundary_microscope_acquisition_route",
        ROOT / "opentad" / "models" / "selectors" / "boundary_microscope_acquisition_route.py",
    )
    _load_module("opentad.models.detectors.base", ROOT / "opentad" / "models" / "detectors" / "base.py")
    _load_module(
        "opentad.models.detectors.single_stage",
        ROOT / "opentad" / "models" / "detectors" / "single_stage.py",
    )
    actionformer_module = _load_module(
        "opentad.models.detectors.actionformer",
        ROOT / "opentad" / "models" / "detectors" / "actionformer.py",
    )
    return actionformer_module.ActionFormer


def _make_6d_batch(batch=1, dense_len=64):
    signal = torch.zeros(dense_len, dtype=torch.float32)
    signal[8:22] = 4.0
    signal[39:50] = 3.0
    batch_offset = torch.arange(batch, dtype=torch.float32).view(batch, 1, 1, 1, 1, 1) * 1000.0
    view_offset = torch.arange(2, dtype=torch.float32).view(1, 2, 1, 1, 1, 1) * 100.0
    channel_offset = torch.arange(3, dtype=torch.float32).view(1, 1, 3, 1, 1, 1) * 10.0
    height_offset = torch.arange(3, dtype=torch.float32).view(1, 1, 1, 1, 3, 1)
    width_offset = torch.arange(3, dtype=torch.float32).view(1, 1, 1, 1, 1, 3) * 0.1
    inputs = (
        signal.view(1, 1, 1, dense_len, 1, 1)
        + batch_offset
        + view_offset
        + channel_offset
        + height_offset
        + width_offset
    ).contiguous()
    masks = torch.ones((batch, dense_len), dtype=torch.bool)
    metas = [{"sample_id": f"boundary-actionformer-6d-{idx}"} for idx in range(batch)]
    gt_segments = [torch.tensor([[8.0, 22.0], [39.0, 50.0]], dtype=torch.float32) for _idx in range(batch)]
    gt_labels = [torch.tensor([1, 2], dtype=torch.long) for _idx in range(batch)]
    return inputs, masks, metas, gt_segments, gt_labels


def _model(ActionFormer):
    model = ActionFormer(
        projection=dict(type="_SyntheticIdentityProjection", channels=4, max_seq_len=64),
        rpn_head=dict(type="_SyntheticRPNHead", in_channels=4),
        frame_selector=dict(
            type="BoundaryMicroscopeAcquisitionRoute",
            route_label=ROUTE_LABEL,
            meta_key="boundary_microscope_acquisition_plan",
            target_len=32,
            dense_window_size=64,
            microscope_radius=2,
            microscope_stride=1,
            anchor_stride=16,
            max_dense_gap=8,
        ),
    )
    model.backbone = _SyntheticVideoMAEBackbone(channels=4)
    return model


def test_actionformer_calls_boundary_selector_before_backbone_on_videomae_6d_layout():
    ActionFormer = _install_boundary_actionformer_runtime()
    model = _model(ActionFormer)
    inputs, masks, metas, gt_segments, gt_labels = _make_6d_batch()

    losses = model.forward_train(
        inputs,
        masks,
        metas=metas,
        gt_segments=gt_segments,
        gt_labels=gt_labels,
    )

    assert torch.isfinite(losses["cost"]).item()
    train_metas = model.rpn_head.last_train_call["metas"]
    selected = train_metas[0]["boundary_microscope_selected_dense_indices"]
    selected_len = len(selected)
    assert model.backbone.calls == [(1, 2, 3, selected_len, 3, 3)]
    assert selected_len <= 32
    assert model.rpn_head.last_train_call["feat_shape"] == (1, 4, 64)
    assert model.rpn_head.last_train_call["mask_shape"] == (1, 64)
    assert int(model.rpn_head.last_train_call["mask"].sum().item()) == selected_len
    assert "boundary_microscope_acquisition_plan" not in metas[0]

    plan = train_metas[0]["boundary_microscope_acquisition_plan"]
    assert plan["route_label"] == ROUTE_LABEL
    assert plan["selection_surface"] == "pre_backbone_raw_frame"
    assert plan["input_layout"] == "[B,N,C,T,H,W]"
    assert plan["input_temporal_axis"] == 3
    assert plan["uses_gt"] is False
    assert plan["uses_teacher"] is False
    assert plan["uses_raw_prediction_cache"] is False
    expected_backbone_inputs = inputs[:, :, :, selected, :, :]
    assert torch.equal(model.backbone.last_inputs, expected_backbone_inputs)
    assert train_metas[0]["irregular_selected_positions"] == [float(pos) for pos in selected]
    assert train_metas[0]["irregular_selected_count"] == selected_len
    assert train_metas[0]["irregular_selected_output_valid_len"] == float(selected_len)
    assert train_metas[0]["irregular_selected_valid_len"] == 64.0
    assert all(left < right for left, right in zip(selected, selected[1:]))


def test_boundary_microscope_inherited_config_declares_videomae_6d_raw_convention():
    local_config = ROOT / "configs" / "adatad" / "thumos" / "boundary_microscope_acquisition_local_precheck.py"
    full_config = ROOT / "configs" / "adatad" / "thumos" / "boundary_microscope_acquisition_full_train_candidate_n16r4.py"
    inherited_config = ROOT / "configs" / "adatad" / "thumos" / "e2e_thumos_videomae_s_768x1_160_adapter.py"

    local_text = local_config.read_text(encoding="utf-8")
    full_text = full_config.read_text(encoding="utf-8")
    inherited_text = inherited_config.read_text(encoding="utf-8")

    assert '"./e2e_thumos_videomae_s_768x1_160_adapter.py"' in local_text
    assert '"./boundary_microscope_acquisition_local_precheck.py"' in full_text
    assert 'format_shape="NCTHW"' in inherited_text
    assert 'ops="b n c (t1 t) h w -> (b t1) n c t h w"' in inherited_text
    assert 'ops="(b t1) c t -> b c (t1 t)"' in inherited_text
