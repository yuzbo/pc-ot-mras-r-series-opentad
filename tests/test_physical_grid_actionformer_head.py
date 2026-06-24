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
    pytest.skip(
        "torch unavailable in this process: "
        + (
            torch_probe.stderr.strip().splitlines()[-1]
            if torch_probe.stderr.strip()
            else f"exit {torch_probe.returncode}"
        ),
        allow_module_level=True,
    )

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PACKAGE = "physical_grid_actionformer_runtime"


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


class _DummyLoss(nn.Module):
    def forward(self, inputs, targets, reduction="none", **kwargs):
        loss = inputs.sum() * 0.0
        if reduction in {"mean", "sum"}:
            return loss
        return inputs.new_zeros(inputs.shape[:-1])


class _Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init_value)))

    def forward(self, x):
        return x * self.scale


class _ConvModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise RuntimeError("physical-grid head tests use num_convs=0")


def _ensure_package(name, path):
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    return module


def _load_module(name, path):
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _install_head_runtime():
    _ensure_package(RUNTIME_PACKAGE, ROOT / "opentad")
    _ensure_package(f"{RUNTIME_PACKAGE}.models", ROOT / "opentad" / "models")
    _ensure_package(f"{RUNTIME_PACKAGE}.models.dense_heads", ROOT / "opentad" / "models" / "dense_heads")
    _ensure_package(
        f"{RUNTIME_PACKAGE}.models.dense_heads.prior_generator",
        ROOT / "opentad" / "models" / "dense_heads" / "prior_generator",
    )
    _ensure_package(f"{RUNTIME_PACKAGE}.models.utils", ROOT / "opentad" / "models" / "utils")

    builder = types.ModuleType(f"{RUNTIME_PACKAGE}.models.builder")
    builder.HEADS = _Registry()
    builder.PRIOR_GENERATORS = _Registry()
    builder.LOSSES = _Registry()
    builder.build_prior_generator = lambda cfg: builder.PRIOR_GENERATORS.build(cfg)
    builder.build_loss = lambda cfg: _DummyLoss()
    sys.modules[f"{RUNTIME_PACKAGE}.models.builder"] = builder

    bricks = types.ModuleType(f"{RUNTIME_PACKAGE}.models.bricks")
    bricks.ConvModule = _ConvModule
    bricks.Scale = _Scale
    sys.modules[f"{RUNTIME_PACKAGE}.models.bricks"] = bricks

    _load_module(
        f"{RUNTIME_PACKAGE}.models.utils.temporal_grid",
        ROOT / "opentad" / "models" / "utils" / "temporal_grid.py",
    )
    _load_module(
        f"{RUNTIME_PACKAGE}.models.dense_heads.prior_generator.point_generator",
        ROOT / "opentad" / "models" / "dense_heads" / "prior_generator" / "point_generator.py",
    )
    _load_module(
        f"{RUNTIME_PACKAGE}.models.dense_heads.anchor_free_head",
        ROOT / "opentad" / "models" / "dense_heads" / "anchor_free_head.py",
    )
    actionformer_head = _load_module(
        f"{RUNTIME_PACKAGE}.models.dense_heads.actionformer_head",
        ROOT / "opentad" / "models" / "dense_heads" / "actionformer_head.py",
    )
    return actionformer_head.ActionFormerHead


ActionFormerHead = _install_head_runtime()


def _make_head(**kwargs):
    head = ActionFormerHead(
        num_classes=2,
        in_channels=2,
        feat_channels=2,
        num_convs=0,
        prior_generator=dict(type="PointGenerator", strides=[1], regression_range=[(0, 10000)]),
        loss=types.SimpleNamespace(cls_loss=dict(type="DummyLoss"), reg_loss=dict(type="DummyLoss")),
        **kwargs,
    )
    with torch.no_grad():
        head.cls_head.weight.zero_()
        head.cls_head.bias.zero_()
        head.reg_head.weight.zero_()
        head.reg_head.bias.zero_()
        head.scale[0].scale.fill_(1.0)
    return head


def _features_and_mask():
    return [torch.zeros(1, 2, 4)], [torch.ones(1, 4, dtype=torch.bool)]


def _irregular_meta():
    return {
        "video_name": "synthetic",
        "irregular_selected_positions": [0.0, 2.0, 5.0, 9.0],
        "irregular_selected_count": 4,
        "irregular_dense_valid_len": 10.0,
        "irregular_selected_valid_len": 10.0,
        "irregular_native_axis": False,
    }


def test_actionformer_head_physical_grid_opt_in_decodes_on_selected_dense_positions():
    head = _make_head(temporal_grid=dict(enabled=True, temporal_grid_mode="physical", required=True, strict=True))
    feat_list, mask_list = _features_and_mask()
    metas = [_irregular_meta()]

    proposals, scores = head.forward_test(feat_list, mask_list, metas=metas)

    expected_centers = torch.tensor([0.0, 2.0, 5.0, 9.0])
    assert torch.allclose(proposals[0][:, 0], expected_centers)
    assert torch.allclose(proposals[0][:, 1], expected_centers)
    assert not torch.allclose(proposals[0][:, 0], torch.arange(4, dtype=torch.float32))
    assert scores[0].shape == (4, 2)
    assert metas[0]["irregular_native_axis"] is True
    assert metas[0]["physical_grid_actionformer"] is True


def test_actionformer_head_default_decode_keeps_selected_index_axis_with_same_metadata():
    head = _make_head()
    feat_list, mask_list = _features_and_mask()
    metas = [_irregular_meta()]

    proposals, _ = head.forward_test(feat_list, mask_list, metas=metas)

    expected_centers = torch.arange(4, dtype=torch.float32)
    assert torch.allclose(proposals[0][:, 0], expected_centers)
    assert torch.allclose(proposals[0][:, 1], expected_centers)
    assert metas[0]["irregular_native_axis"] is False
    assert "physical_grid_actionformer" not in metas[0]


def test_actionformer_head_batch_physical_points_drive_target_assignment():
    head = _make_head()
    points = [
        torch.tensor(
            [
                [
                    [0.0, 0.0, 10000.0, 1.0],
                    [2.0, 0.0, 10000.0, 1.0],
                    [5.0, 0.0, 10000.0, 1.0],
                    [9.0, 0.0, 10000.0, 1.0],
                ]
            ],
            dtype=torch.float32,
        )
    ]
    gt_segments = [torch.tensor([[1.5, 2.5]], dtype=torch.float32)]
    gt_labels = [torch.tensor([1], dtype=torch.long)]

    gt_cls, gt_reg = head.prepare_targets(points, gt_segments, gt_labels)

    positive = gt_cls[0].sum(dim=1) > 0
    assert positive.tolist() == [False, True, False, False]
    assert torch.allclose(gt_reg[0][1], torch.tensor([0.5, 0.5]))


def test_actionformer_head_physical_grid_rejects_selected_axis_gt_remap_metadata():
    head = _make_head(temporal_grid=dict(enabled=True, temporal_grid_mode="physical", required=True, strict=True))
    feat_list, mask_list = _features_and_mask()
    metas = [_irregular_meta()]
    metas[0]["pc_ot_mras_prebackbone_remap_gt_to_selected_axis"] = True

    with pytest.raises(ValueError, match="dense-axis GT"):
        head.forward_train(
            feat_list,
            mask_list,
            gt_segments=[torch.tensor([[1.0, 2.0]], dtype=torch.float32)],
            gt_labels=[torch.tensor([1], dtype=torch.long)],
            metas=metas,
        )
