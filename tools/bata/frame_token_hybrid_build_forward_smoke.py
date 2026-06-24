from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ROUTE_PATH = ROOT / "opentad" / "models" / "selectors" / "frame_token_hybrid_acquisition_route.py"
ACTIONFORMER_PATH = ROOT / "opentad" / "models" / "detectors" / "actionformer.py"


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


def _load_smoke_modules(torch):
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


def main():
    import torch

    route_module, action_module = _load_smoke_modules(torch)

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
        preview_source_meta_key="frame_token_hybrid_preview_source",
    )
    backbone = RecordingBackbone()
    head = RecordingHead()
    detector = action_module.ActionFormer(
        backbone=backbone,
        projection=IdentityProjection(),
        rpn_head=head,
        frame_selector=selector,
    )
    inputs = torch.zeros((1, 3, 16, 2, 2), dtype=torch.float32)
    inputs[:, :, 4] = 1234.0
    masks = torch.ones((1, 16), dtype=torch.bool)
    metas = [
        {
            "video_name": "frame-token-build-forward-smoke",
            "frame_token_hybrid_preview_signal": [0.0] * 16,
            "frame_token_hybrid_preview_positions": list(range(16)),
            "frame_token_hybrid_preview_source": "frame_inds_temporal_gap_preview_probe",
        }
    ]

    proposals, scores = detector.forward_test(inputs, masks, metas=metas)
    plan = head.received_metas[0]["frame_token_hybrid_acquisition_plan"]
    assert tuple(proposals.shape) == (1, 1, 2)
    assert tuple(scores.shape) == (1, 1, 1)
    assert plan["selection_decision_source"] == "deploy_preview_probe_metadata"
    assert plan["preview_probe"]["source"] == "frame_inds_temporal_gap_preview_probe"
    assert plan["compute_accounting"]["raw_decode_saving_claim_allowed"] is False
    print("FRAME_TOKEN_HYBRID_BUILD_FORWARD_SMOKE_PASS")


if __name__ == "__main__":
    main()
