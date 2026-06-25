from __future__ import annotations

import types
import importlib.util

import pytest


def test_frame_token_hybrid_build_detector_forward_smoke_with_preview_metadata(monkeypatch):
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on local DLL state.
        pytest.skip(f"torch unavailable in this process: {exc}")
    try:
        has_mmaction_registry = importlib.util.find_spec("mmaction.registry") is not None
    except ModuleNotFoundError:
        has_mmaction_registry = False
    if not has_mmaction_registry:
        pytest.skip("full opentad.models import requires mmaction.registry in this environment")
    from opentad.models import build_detector
    from opentad.models import builder
    import opentad.models.detectors.single_stage as single_stage

    class DummyVideoBackbone(torch.nn.Module):
        def forward(self, inputs, masks=None):
            return inputs.mean(dim=(3, 4))

    class DummyProjection(torch.nn.Module):
        n_mha_win_size = 1
        arch = (1, 1, 0)
        max_seq_len = 16

        def forward(self, features, masks):
            return [features], [masks]

    class DummyHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.prior_generator = types.SimpleNamespace(strides=[1])
            self.received_metas = None

        def forward_test(self, feat_list, mask_list, metas=None):
            self.received_metas = metas
            proposals = feat_list[0].new_zeros((feat_list[0].shape[0], 1, 2))
            scores = feat_list[0].new_zeros((feat_list[0].shape[0], 1, 1))
            return proposals, scores

    builder.PROJECTIONS.register_module(name="FrameTokenHybridDummyProjection", module=DummyProjection, force=True)
    builder.HEADS.register_module(name="FrameTokenHybridDummyHead", module=DummyHead, force=True)
    monkeypatch.setattr(single_stage, "build_backbone", lambda _cfg: DummyVideoBackbone())

    detector = build_detector(
        dict(
            type="ActionFormer",
            backbone=dict(type="FrameTokenHybridDummyBackbone"),
            projection=dict(type="FrameTokenHybridDummyProjection"),
            rpn_head=dict(type="FrameTokenHybridDummyHead"),
            frame_selector=dict(
                type="FrameTokenHybridAcquisitionRoute",
                dense_window_size=16,
                target_dense_len=16,
                anchor_stride=8,
                boundary_radius=0,
                boundary_epsilon=0.1,
                stable_gap_min_len=6,
                require_preview_signal=True,
                preview_signal_meta_key="frame_token_hybrid_preview_signal",
                preview_positions_meta_key="frame_token_hybrid_preview_positions",
                preview_source_meta_key="frame_token_hybrid_preview_source",
            ),
        )
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

    assert tuple(proposals.shape) == (1, 1, 2)
    assert tuple(scores.shape) == (1, 1, 1)
    plan = detector.rpn_head.received_metas[0]["frame_token_hybrid_acquisition_plan"]
    assert plan["route_label"] == "DIVERGENT_INNOVATION_FRAME_TOKEN_HYBRID_DO_NOT_MERGE_WITH_C3"
    assert plan["selection_decision_source"] == "deploy_preview_probe_metadata"
    assert plan["preview_probe"]["source"] == "frame_inds_temporal_gap_preview_probe"
    assert plan["compute_accounting"]["raw_decode_saving_claim_allowed"] is False
