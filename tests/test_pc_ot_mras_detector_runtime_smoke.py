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


ROOT = Path(__file__).resolve().parents[1]


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


def _install_minimal_runtime_modules():
    _ensure_package("opentad", ROOT / "opentad")
    _ensure_package("opentad.models", ROOT / "opentad" / "models")
    _ensure_package("opentad.models.detectors", ROOT / "opentad" / "models" / "detectors")
    _ensure_package("opentad.models.selectors", ROOT / "opentad" / "models" / "selectors")
    _ensure_package("opentad.models.necks", ROOT / "opentad" / "models" / "necks")
    _ensure_package("opentad.models.dense_heads", ROOT / "opentad" / "models" / "dense_heads")
    _ensure_package(
        "opentad.models.dense_heads.prior_generator",
        ROOT / "opentad" / "models" / "dense_heads" / "prior_generator",
    )
    _ensure_package("opentad.models.utils", ROOT / "opentad" / "models" / "utils")
    bricks_pkg = _ensure_package("opentad.models.bricks", ROOT / "opentad" / "models" / "bricks")

    backbones = types.ModuleType("opentad.models.backbones")
    backbones.BackboneWrapper = lambda cfg: (_ for _ in ()).throw(
        RuntimeError("synthetic smoke must not build a backbone")
    )
    sys.modules["opentad.models.backbones"] = backbones

    post_processing = types.ModuleType("opentad.models.utils.post_processing")
    post_processing.load_predictions = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("synthetic smoke must not load raw predictions")
    )
    post_processing.save_predictions = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("synthetic smoke must not save raw predictions")
    )
    post_processing.batched_nms = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("synthetic smoke must not run post-processing NMS")
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
    builder.build_neck = lambda cfg: builder.NECKS.build(cfg)
    builder.build_head = lambda cfg: builder.HEADS.build(cfg)
    builder.build_prior_generator = lambda cfg: builder.PRIOR_GENERATORS.build(cfg)
    sys.modules["opentad.models.builder"] = builder

    _load_module("opentad.ctf_bdi_role_constants", ROOT / "opentad" / "ctf_bdi_role_constants.py")
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
    _load_module(
        "opentad.models.detectors.base",
        ROOT / "opentad" / "models" / "detectors" / "base.py",
    )
    detector_module = _load_module(
        "opentad.models.detectors.single_stage",
        ROOT / "opentad" / "models" / "detectors" / "single_stage.py",
    )
    return (
        reader_module.PCOTMRASReader,
        bridge_module.PCOTMRASDetectorBridge,
        head_module.NativeIrregularAreaHeadP2,
        detector_module.SingleStageDetector,
    )


PCOTMRASReader, PCOTMRASDetectorBridge, NativeIrregularAreaHeadP2, SingleStageDetector = (
    _install_minimal_runtime_modules()
)


def _model():
    model = SingleStageDetector()
    model.neck = PCOTMRASDetectorBridge(in_channels=4, out_channels=4)
    model.rpn_head = NativeIrregularAreaHeadP2(
        num_classes=3,
        in_channels=4,
        feat_channels=4,
        num_convs=0,
        prior_generator=dict(type="PointGenerator", strides=[1], regression_range=[(0, 10000)]),
        temporal_grid=dict(required=True, decode_axis="dense", strict=True),
        area_head=dict(
            observation_half_width=0.5,
            boundary_tau=1.0,
            max_boundaries_per_side=4,
            max_pairs_per_class=8,
            start_offset_loss_weight=0.25,
            end_offset_loss_weight=0.25,
            uncertainty_loss_weight=0.05,
        ),
    )
    return model


def _reader_outputs_and_inputs():
    torch.manual_seed(20260618)
    reader = PCOTMRASReader(in_dim=5, hidden_dim=8, num_slots=4, num_blocks=1, num_roles=6)
    lowcost_features = torch.randn(1, 8, 5, requires_grad=True)
    dense_features = torch.randn(1, 4, 8, requires_grad=True)
    valid_mask = torch.ones(1, 8, dtype=torch.bool)
    reader_outputs = reader(lowcost_features, valid_mask)
    metas = [{"sample_id": "pc_ot_runtime_smoke|0", "pc_ot_mras_reader_outputs": reader_outputs}]
    return reader, lowcost_features, dense_features, valid_mask, metas


def test_pc_ot_mras_reader_bridge_p2_detector_train_and_test_runtime_smoke():
    reader, lowcost_features, dense_features, valid_mask, metas = _reader_outputs_and_inputs()
    model = _model()
    inputs = (dense_features,)
    masks = (valid_mask,)
    gt_segments = [torch.tensor([[1.0, 6.5]], dtype=torch.float32)]
    gt_labels = [torch.tensor([1], dtype=torch.long)]

    model.train()
    losses = model.forward_train(inputs, masks, metas=metas, gt_segments=gt_segments, gt_labels=gt_labels)

    expected_keys = {
        "area_loss",
        "start_gap_loss",
        "end_gap_loss",
        "start_offset_loss",
        "end_offset_loss",
        "boundary_uncertainty_loss",
        "cost",
    }
    assert set(losses) == expected_keys
    assert all(torch.isfinite(value).item() for value in losses.values())
    losses["cost"].backward()
    assert reader.input_proj.weight.grad is not None
    assert torch.isfinite(reader.input_proj.weight.grad).all()
    assert reader.input_proj.weight.grad.abs().sum().item() > 0
    assert model.neck.token_proj.weight.grad is not None
    assert torch.isfinite(model.neck.token_proj.weight.grad).all()
    assert dense_features.grad is not None
    assert torch.isfinite(dense_features.grad).all()
    assert lowcost_features.grad is not None
    assert torch.isfinite(lowcost_features.grad).all()

    model.eval()
    with torch.no_grad():
        proposals, scores = model.forward_test(inputs, masks, metas=metas)

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


def test_pc_ot_mras_detector_runtime_missing_reader_outputs_fails_closed():
    model = _model()
    inputs = (torch.randn(1, 4, 8),)
    masks = (torch.ones(1, 8, dtype=torch.bool),)
    gt_segments = [torch.tensor([[1.0, 6.5]], dtype=torch.float32)]
    gt_labels = [torch.tensor([1], dtype=torch.long)]

    with pytest.raises(ValueError, match="pc_ot_mras_reader_outputs"):
        model.forward_train(inputs, masks, metas=[{"sample_id": "missing"}], gt_segments=gt_segments, gt_labels=gt_labels)
