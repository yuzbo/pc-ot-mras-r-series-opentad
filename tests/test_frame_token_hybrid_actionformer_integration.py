from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ROUTE_PATH = ROOT / "opentad" / "models" / "selectors" / "frame_token_hybrid_acquisition_route.py"
ACTIONFORMER_PATH = ROOT / "opentad" / "models" / "detectors" / "actionformer.py"


class _Registry:
    def register_module(self):
        def _decorator(cls):
            return cls

        return _decorator


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
    import torch

    return torch


def _ensure_package(name: str, path: Path):
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    return module


def _load_modules(torch):
    for name in (
        "opentad.models.selectors.frame_token_hybrid_acquisition_route",
        "opentad.models.detectors.actionformer",
    ):
        sys.modules.pop(name, None)
    _ensure_package("opentad", ROOT / "opentad")
    _ensure_package("opentad.models", ROOT / "opentad" / "models")
    _ensure_package("opentad.models.selectors", ROOT / "opentad" / "models" / "selectors")
    _ensure_package("opentad.models.detectors", ROOT / "opentad" / "models" / "detectors")

    builder = types.ModuleType("opentad.models.builder")
    builder.SELECTORS = _Registry()
    builder.DETECTORS = _Registry()
    builder.build_selector = lambda cfg: cfg
    builder.build_token_compressor = lambda cfg: cfg
    sys.modules["opentad.models.builder"] = builder

    bricks = types.ModuleType("opentad.models.bricks")
    bricks.Scale = type("Scale", (torch.nn.Module,), {})
    bricks.AffineDropPath = type("AffineDropPath", (torch.nn.Module,), {})
    sys.modules["opentad.models.bricks"] = bricks

    single_stage = types.ModuleType("opentad.models.detectors.single_stage")

    class SingleStageDetector(torch.nn.Module):
        def __init__(self, backbone=None, projection=None, neck=None, rpn_head=None):
            super().__init__()
            self.backbone = backbone
            self.projection = projection
            self.neck = neck
            self.rpn_head = rpn_head

        @property
        def with_backbone(self):
            return self.backbone is not None

        @property
        def with_projection(self):
            return self.projection is not None

        @property
        def with_neck(self):
            return self.neck is not None

    single_stage.SingleStageDetector = SingleStageDetector
    sys.modules["opentad.models.detectors.single_stage"] = single_stage

    route_spec = importlib.util.spec_from_file_location(
        "opentad.models.selectors.frame_token_hybrid_acquisition_route",
        ROUTE_PATH,
    )
    route_module = importlib.util.module_from_spec(route_spec)
    sys.modules[route_spec.name] = route_module
    route_spec.loader.exec_module(route_module)

    action_spec = importlib.util.spec_from_file_location(
        "opentad.models.detectors.actionformer",
        ACTIONFORMER_PATH,
    )
    action_module = importlib.util.module_from_spec(action_spec)
    sys.modules[action_spec.name] = action_module
    action_spec.loader.exec_module(action_module)
    return route_module, action_module


def test_actionformer_forwards_selector_rewritten_dense_bridge_to_backbone_and_head():
    torch = _import_torch_or_skip()
    route_module, action_module = _load_modules(torch)

    class RecordingBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.received_inputs = None

        def forward(self, inputs):
            self.received_inputs = inputs.detach().clone()
            return inputs.mean(dim=(3, 4))

    class IdentityProjection(torch.nn.Module):
        n_mha_win_size = 1
        arch = (1, 1, 0)
        max_seq_len = 16

        def forward(self, features, masks):
            return [features], [masks]

    class RecordingHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.prior_generator = types.SimpleNamespace(strides=[1])
            self.received_metas = None

        def forward_test(self, feat_list, mask_list, metas=None):
            self.received_metas = metas
            proposals = feat_list[0].new_zeros((feat_list[0].shape[0], 1, 2))
            scores = feat_list[0].new_zeros((feat_list[0].shape[0], 1, 1))
            return proposals, scores

    selector = route_module.FrameTokenHybridAcquisitionRoute(
        dense_window_size=16,
        target_dense_len=16,
        anchor_stride=8,
        boundary_radius=0,
        boundary_epsilon=0.1,
        stable_gap_min_len=6,
        require_preview_signal=True,
    )
    backbone = RecordingBackbone()
    head = RecordingHead()
    detector = action_module.ActionFormer(
        backbone=backbone,
        projection=IdentityProjection(),
        rpn_head=head,
        frame_selector=selector,
    )
    values = torch.zeros(16, dtype=torch.float32)
    values[4] = 1234.0
    values[8] = 80.0
    values[15] = 150.0
    inputs = values.view(1, 1, 16, 1, 1).expand(1, 3, 16, 2, 2).contiguous()
    masks = torch.ones((1, 16), dtype=torch.bool)
    metas = [
        {
            "sample_id": "actionformer-integration",
            "frame_token_hybrid_preview_signal": [0.0] * 16,
            "frame_token_hybrid_preview_positions": list(range(16)),
        }
    ]

    detector.forward_test(inputs, masks, metas=metas)

    assert backbone.received_inputs is not None
    assert not torch.equal(backbone.received_inputs[0, :, 4], inputs[0, :, 4])
    plan = head.received_metas[0]["frame_token_hybrid_acquisition_plan"]
    assert plan["selection_decision_source"] == "deploy_preview_probe_metadata"
    assert plan["compute_accounting"]["actual_decode_saving_in_current_actionformer_pipeline"] is False
    assert head.received_metas[0]["frame_token_hybrid_dense_completion_mask"][4] is True
