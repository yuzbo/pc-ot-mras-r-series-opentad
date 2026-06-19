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
READER_OUTPUTS_META_KEY = "pc_ot_mras_reader_outputs"
OUTPUT_STRIDES = [1, 2, 4, 8, 16, 32]


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
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _SyntheticAffineDropPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, x):
        return x * self.scale


def _install_actionformer_p2_runtime():
    for name in (
        "opentad.ctf_bdi_role_constants",
        "opentad.models.builder",
        "opentad.models.bricks",
        "opentad.models.bricks.conv",
        "opentad.models.bricks.misc",
        "opentad.models.utils.post_processing",
        "opentad.models.utils.sampling_contract",
        "opentad.models.utils.temporal_grid",
        "opentad.models.backbones",
        "opentad.models.selectors.lowcost_acquisition_browser",
        "opentad.models.selectors.pc_ot_mras_reader",
        "opentad.models.necks.pc_ot_mras_detector_bridge",
        "opentad.models.dense_heads.prior_generator.point_generator",
        "opentad.models.dense_heads.native_irregular_area_head_p2",
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
    _ensure_package("opentad.models.necks", ROOT / "opentad" / "models" / "necks")
    _ensure_package("opentad.models.dense_heads", ROOT / "opentad" / "models" / "dense_heads")
    _ensure_package(
        "opentad.models.dense_heads.prior_generator",
        ROOT / "opentad" / "models" / "dense_heads" / "prior_generator",
    )
    _ensure_package("opentad.models.utils", ROOT / "opentad" / "models" / "utils")

    backbones = types.ModuleType("opentad.models.backbones")
    backbones.BackboneWrapper = lambda cfg: (_ for _ in ()).throw(
        RuntimeError("synthetic PC-OT-MRAS ActionFormer smoke must not build a backbone")
    )
    sys.modules["opentad.models.backbones"] = backbones

    post_processing = types.ModuleType("opentad.models.utils.post_processing")
    post_processing.load_predictions = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("synthetic PC-OT-MRAS ActionFormer smoke must not load raw predictions")
    )
    post_processing.save_predictions = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("synthetic PC-OT-MRAS ActionFormer smoke must not save raw predictions")
    )
    post_processing.batched_nms = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("synthetic PC-OT-MRAS ActionFormer smoke must not run NMS")
    )
    post_processing.convert_to_seconds = lambda segments, meta: segments
    sys.modules["opentad.models.utils.post_processing"] = post_processing

    builder = types.ModuleType("opentad.models.builder")
    builder.MODELS = _Registry()
    builder.SELECTORS = _Registry()
    builder.NECKS = _Registry()
    builder.HEADS = _Registry()
    builder.DETECTORS = _Registry()
    builder.PRIOR_GENERATORS = _Registry()
    builder.PROJECTIONS = _Registry()
    builder.TOKEN_COMPRESSORS = _Registry()
    builder.ROI_EXTRACTORS = _Registry()
    builder.PROPOSAL_GENERATORS = _Registry()
    builder.TRANSFORMERS = _Registry()
    builder.LOSSES = _Registry()
    builder.MATCHERS = _Registry()
    builder.build_backbone = lambda cfg: backbones.BackboneWrapper(cfg)
    builder.build_projection = lambda cfg: builder.PROJECTIONS.build(cfg)
    builder.build_selector = lambda cfg: builder.SELECTORS.build(cfg)
    builder.build_token_compressor = lambda cfg: builder.TOKEN_COMPRESSORS.build(cfg)
    builder.build_neck = lambda cfg: builder.NECKS.build(cfg)
    builder.build_head = lambda cfg: builder.HEADS.build(cfg)
    builder.build_prior_generator = lambda cfg: builder.PRIOR_GENERATORS.build(cfg)
    sys.modules["opentad.models.builder"] = builder

    bricks_pkg = sys.modules["opentad.models.bricks"]
    conv_module = _load_module(
        "opentad.models.bricks.conv",
        ROOT / "opentad" / "models" / "bricks" / "conv.py",
    )
    misc_module = _load_module(
        "opentad.models.bricks.misc",
        ROOT / "opentad" / "models" / "bricks" / "misc.py",
    )
    bricks_pkg.ConvModule = conv_module.ConvModule
    bricks_pkg.Scale = misc_module.Scale
    bricks_pkg.AffineDropPath = _SyntheticAffineDropPath

    class SyntheticIdentityProjection(nn.Module):
        def __init__(self, channels=4, max_seq_len=32):
            super().__init__()
            self.n_mha_win_size = 1
            self.arch = (0,)
            self.max_seq_len = int(max_seq_len)
            self.proj = nn.Conv1d(int(channels), int(channels), kernel_size=1, bias=False)
            nn.init.eye_(self.proj.weight[:, :, 0])

        def forward(self, x, masks):
            return (self.proj(x),), (masks,)

    builder.PROJECTIONS.register_module()(SyntheticIdentityProjection)

    _load_module("opentad.ctf_bdi_role_constants", ROOT / "opentad" / "ctf_bdi_role_constants.py")
    _load_module(
        "opentad.models.selectors.lowcost_acquisition_browser",
        ROOT / "opentad" / "models" / "selectors" / "lowcost_acquisition_browser.py",
    )
    reader_module = _load_module(
        "opentad.models.selectors.pc_ot_mras_reader",
        ROOT / "opentad" / "models" / "selectors" / "pc_ot_mras_reader.py",
    )
    bridge_module = _load_module(
        "opentad.models.necks.pc_ot_mras_detector_bridge",
        ROOT / "opentad" / "models" / "necks" / "pc_ot_mras_detector_bridge.py",
    )
    _load_module(
        "opentad.models.utils.sampling_contract",
        ROOT / "opentad" / "models" / "utils" / "sampling_contract.py",
    )
    _load_module(
        "opentad.models.utils.temporal_grid",
        ROOT / "opentad" / "models" / "utils" / "temporal_grid.py",
    )
    _load_module(
        "opentad.models.dense_heads.prior_generator.point_generator",
        ROOT / "opentad" / "models" / "dense_heads" / "prior_generator" / "point_generator.py",
    )
    head_module = _load_module(
        "opentad.models.dense_heads.native_irregular_area_head_p2",
        ROOT / "opentad" / "models" / "dense_heads" / "native_irregular_area_head_p2.py",
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
    return (
        actionformer_module.ActionFormer,
        reader_module.PCOTMRASReader,
        bridge_module.PCOTMRASDetectorBridge,
        head_module.NativeIrregularAreaHeadP2,
    )


ActionFormer, PCOTMRASReader, PCOTMRASDetectorBridge, NativeIrregularAreaHeadP2 = _install_actionformer_p2_runtime()


def _model():
    return ActionFormer(
        projection=dict(type="SyntheticIdentityProjection", channels=4, max_seq_len=32),
        neck=dict(
            type="PCOTMRASDetectorBridge",
            in_channels=4,
            out_channels=4,
            output_strides=OUTPUT_STRIDES,
        ),
        rpn_head=dict(
            type="NativeIrregularAreaHeadP2",
            num_classes=3,
            in_channels=4,
            feat_channels=4,
            num_convs=0,
            prior_generator=dict(
                type="PointGenerator",
                strides=OUTPUT_STRIDES,
                regression_range=[(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10000)],
            ),
            temporal_grid=dict(required=True, decode_axis="dense", strict=True),
            area_head=dict(
                observation_half_width="cell_support",
                max_boundaries_per_side=4,
                max_pairs_per_class=6,
                start_offset_loss_weight=0.25,
                end_offset_loss_weight=0.25,
                uncertainty_loss_weight=0.05,
            ),
        ),
        pc_ot_mras_reader=dict(
            type="PCOTMRASReader",
            in_dim=4,
            hidden_dim=8,
            num_slots=8,
            num_blocks=1,
            num_roles=6,
            dropout=0.0,
        ),
    )


def _inputs():
    torch.manual_seed(20260619)
    inputs = torch.randn(1, 4, 32, requires_grad=True)
    masks = torch.ones(1, 32, dtype=torch.bool)
    metas = [{"sample_id": "pc_ot_mras_actionformer_multiscale_p2|0"}]
    gt_segments = [torch.tensor([[5.0, 21.0]], dtype=torch.float32)]
    gt_labels = [torch.tensor([1], dtype=torch.long)]
    return inputs, masks, metas, gt_segments, gt_labels


def _expected_lengths(start_len, levels):
    lengths = [start_len]
    for _ in range(1, levels):
        lengths.append((lengths[-1] + 1) // 2)
    return lengths


def _assert_prefix(mask):
    valid_count = mask.long().sum(dim=1)
    expected = torch.arange(mask.shape[1], device=mask.device)[None, :] < valid_count[:, None]
    assert torch.equal(mask, expected)


def _assert_multiscale_p2_call(captured, split):
    assert captured["split"] == split
    feat_list = captured["feat_list"]
    mask_list = captured["mask_list"]
    metas = captured["metas"]
    assert len(feat_list) == len(mask_list) == len(OUTPUT_STRIDES)
    assert [feat.shape[-1] for feat in feat_list] == _expected_lengths(8, len(OUTPUT_STRIDES))
    assert [mask.shape[-1] for mask in mask_list] == _expected_lengths(8, len(OUTPUT_STRIDES))
    for feat, mask in zip(feat_list, mask_list):
        assert feat.shape[0] == 1
        assert feat.shape[1] == 4
        assert mask.dtype == torch.bool
        _assert_prefix(mask)
        assert torch.isfinite(feat).all()
    assert isinstance(metas, list)
    assert len(metas) == 1
    assert READER_OUTPUTS_META_KEY in metas[0]
    assert "pc_ot_mras_bridge" in metas[0]
    bridge_meta = metas[0]["pc_ot_mras_bridge"]
    assert bridge_meta["output_strides"] == "(1, 2, 4, 8, 16, 32)"
    assert bridge_meta["uses_hard_gather"] is False
    assert bridge_meta["selected_tokens_source"] == "recomputed_from_acquisition_matrix"
    assert bridge_meta["selected_tokens_source_verified"] is True
    assert "selected_dense_positions" in bridge_meta
    assert "dense_valid_len_tensor" in bridge_meta
    assert bridge_meta["selected_dense_positions"].shape == (8,)
    assert torch.allclose(bridge_meta["dense_valid_len_tensor"], torch.tensor(32.0))
    assert bridge_meta["temporal_tensor_metadata_mode"] == "selected_dense_positions_from_centers"
    assert metas[0]["irregular_native_axis"] is True
    assert metas[0]["irregular_selected_count"] == 8
    assert metas[0]["irregular_dense_valid_len"] == 32
    assert metas[0]["irregular_selected_valid_len"] == 32
    positions = metas[0]["irregular_selected_positions"]
    assert len(positions) == 8
    assert torch.allclose(bridge_meta["selected_dense_positions"], torch.tensor(positions))
    assert all(0.0 <= pos < 32.0 for pos in positions)
    assert all(left < right for left, right in zip(positions, positions[1:]))


def test_actionformer_internal_reader_multiscale_bridge_p2_train_and_test_smoke():
    model = _model()
    inputs, masks, metas, gt_segments, gt_labels = _inputs()
    captured = {}

    original_forward_train = model.rpn_head.forward_train
    original_forward_test = model.rpn_head.forward_test

    def capture_forward_train(feat_list, mask_list, gt_segments, gt_labels, metas=None, **kwargs):
        captured["train"] = {
            "split": "train",
            "feat_list": feat_list,
            "mask_list": mask_list,
            "metas": metas,
        }
        return original_forward_train(
            feat_list,
            mask_list,
            gt_segments=gt_segments,
            gt_labels=gt_labels,
            metas=metas,
            **kwargs,
        )

    def capture_forward_test(feat_list, mask_list, metas=None, **kwargs):
        captured["test"] = {
            "split": "test",
            "feat_list": feat_list,
            "mask_list": mask_list,
            "metas": metas,
        }
        return original_forward_test(feat_list, mask_list, metas=metas, **kwargs)

    model.rpn_head.forward_train = capture_forward_train
    model.rpn_head.forward_test = capture_forward_test

    model.train()
    losses = model.forward_train(
        inputs,
        masks,
        metas=metas,
        gt_segments=gt_segments,
        gt_labels=gt_labels,
    )

    assert set(losses) == {
        "area_loss",
        "start_gap_loss",
        "end_gap_loss",
        "start_offset_loss",
        "end_offset_loss",
        "boundary_uncertainty_loss",
        "cost",
    }
    assert all(torch.isfinite(value).item() for value in losses.values())
    assert READER_OUTPUTS_META_KEY not in metas[0]
    _assert_multiscale_p2_call(captured["train"], "train")

    losses["cost"].backward()

    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert inputs.grad.abs().sum().item() > 0
    assert model.projection.proj.weight.grad is not None
    assert torch.isfinite(model.projection.proj.weight.grad).all()
    assert model.projection.proj.weight.grad.abs().sum().item() > 0
    assert model.pc_ot_mras_reader.input_proj.weight.grad is not None
    assert torch.isfinite(model.pc_ot_mras_reader.input_proj.weight.grad).all()
    assert model.pc_ot_mras_reader.input_proj.weight.grad.abs().sum().item() > 0
    assert model.pc_ot_mras_reader.gate_head.weight.grad is not None
    assert torch.isfinite(model.pc_ot_mras_reader.gate_head.weight.grad).all()
    assert model.neck.token_proj.weight.grad is not None
    assert torch.isfinite(model.neck.token_proj.weight.grad).all()
    assert model.neck.token_proj.weight.grad.abs().sum().item() > 0
    assert model.rpn_head.area_head.weight.grad is not None
    assert torch.isfinite(model.rpn_head.area_head.weight.grad).all()
    assert model.rpn_head.start_gap_head.weight.grad is not None
    assert torch.isfinite(model.rpn_head.start_gap_head.weight.grad).all()

    model.eval()
    test_inputs = inputs.detach()
    with torch.no_grad():
        proposals, scores = model.forward_test(test_inputs, masks, metas=metas)

    _assert_multiscale_p2_call(captured["test"], "test")
    assert READER_OUTPUTS_META_KEY not in metas[0]
    assert len(proposals) == 1
    assert len(scores) == 1
    assert proposals[0].ndim == 2
    assert proposals[0].shape[1] == 2
    assert scores[0].ndim == 2
    assert scores[0].shape[1] == 3
    assert torch.isfinite(proposals[0]).all()
    assert torch.isfinite(scores[0]).all()
    if proposals[0].numel():
        assert torch.all(proposals[0][:, 1] > proposals[0][:, 0])
