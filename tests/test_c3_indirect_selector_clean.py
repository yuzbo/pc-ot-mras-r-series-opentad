import torch

from opentad.models.detectors.base import BaseDetector
from opentad.models.builder import build_selector
from opentad.models.selectors.c3_indirect_frame_selector import PCOTMRASCoarseActionnessFrameScout


def _make_selector(target_len=4, dense_window_size=8, scout_spatial_size=4):
    return build_selector(
        dict(
            type="PCOTMRASIndirectPreBackboneFrameSelector",
            target_len=target_len,
            dense_window_size=dense_window_size,
            selection_unit=1,
            scout_spatial_size=scout_spatial_size,
            strategy="coarse_actionness_uncertainty",
            quotas=dict(uniform=2, action=1, uncertainty=1, change=0, background=0),
            max_gap_guard_count=0,
            st_local_radius=1,
            st_scale=0.5,
            actionness_loss_weight=0.2,
            scout=dict(
                type="PCOTMRASCoarseActionnessFrameScout",
                in_channels=3 * scout_spatial_size * scout_spatial_size,
                hidden_channels=8,
                num_layers=1,
                kernel_size=3,
            ),
        )
    )


def test_coarse_scout_only_outputs_action_logits():
    scout = PCOTMRASCoarseActionnessFrameScout(
        in_channels=3,
        hidden_channels=8,
        num_layers=1,
        kernel_size=3,
    )
    scout_inputs = torch.randn(2, 8, 3)
    scout_masks = torch.ones(2, 8, dtype=torch.bool)

    outputs = scout(scout_inputs, scout_masks)

    assert sorted(outputs.keys()) == ["action_logits"]
    assert outputs["action_logits"].shape == (2, 8)
    for forbidden_name in ["start_head", "end_head", "boundary_head", "risk_head"]:
        assert not hasattr(scout, forbidden_name)


def test_hard_forward_outputs_real_frames_and_remaps_gt():
    selector = _make_selector()
    inputs = torch.arange(1 * 1 * 3 * 8 * 4 * 4, dtype=torch.float32).reshape(1, 1, 3, 8, 4, 4)
    masks = torch.tensor([[True, True, True, True, True, True, False, False]])
    metas = [dict(video_name="video_0", window_size=8, snippet_stride=4, fps=25.0, duration=10.0)]
    gt_segments = [torch.tensor([[1.0, 4.0], [4.5, 6.0]], dtype=torch.float32)]
    gt_labels = [torch.tensor([2, 3], dtype=torch.long)]

    outputs = selector.forward_train(inputs, masks, metas, gt_segments, gt_labels)

    selected = outputs["selected_dense_indices"][0]
    assert outputs["inputs"].shape == (1, 1, 3, 4, 4, 4)
    assert outputs["masks"].shape == (1, 4)
    assert outputs["masks"].all()
    assert torch.all(selected[:-1] <= selected[1:])
    assert selected.unique().numel() == selected.numel()
    assert selected.max().item() < 6

    for slot, dense_idx in enumerate(selected.tolist()):
        assert torch.equal(outputs["inputs"][0, 0, :, slot], inputs[0, 0, :, dense_idx])

    assert outputs["gt_segments"][0].numel() > 0
    assert outputs["gt_segments"][0].min().item() >= 0
    assert outputs["gt_segments"][0].max().item() <= 4
    assert "loss_c3_actionness" in outputs["losses"]
    assert "loss_c3_boundary" not in outputs["losses"]
    assert outputs["metas"][0]["c3_indirect_fixed_axis"] is True
    assert outputs["metas"][0]["c3_indirect_selected_dense_indices"] == selected.tolist()


def test_forward_test_uses_same_selection_policy_without_gt_outputs():
    selector = _make_selector()
    inputs = torch.randn(1, 1, 3, 8, 4, 4)
    masks = torch.ones(1, 8, dtype=torch.bool)
    metas = [dict(video_name="video_0", window_size=8, snippet_stride=4, fps=25.0, duration=10.0)]

    outputs = selector.forward_test(inputs, masks, metas)

    assert outputs["inputs"].shape[-3] == 4
    assert outputs["selected_dense_indices"].shape == (1, 4)
    assert "gt_segments" not in outputs
    assert "losses" not in outputs
    assert outputs["metas"][0]["window_size"] == 4
    assert outputs["metas"][0]["c3_indirect_original_dense_window_size"] == 8


def test_padding_duplicates_are_marked_invalid_when_valid_frames_are_short():
    selector = _make_selector(target_len=6, dense_window_size=8)
    inputs = torch.randn(1, 1, 3, 8, 4, 4)
    masks = torch.tensor([[True, True, True, False, False, False, False, False]])
    metas = [dict(video_name="video_0", window_size=8, snippet_stride=4, fps=25.0, duration=10.0)]
    gt_segments = [torch.tensor([[0.0, 2.5]], dtype=torch.float32)]
    gt_labels = [torch.tensor([1], dtype=torch.long)]

    outputs = selector.forward_train(inputs, masks, metas, gt_segments, gt_labels)

    assert outputs["inputs"].shape[-3] == 6
    assert outputs["selected_dense_indices"][0].unique().numel() == 3
    assert outputs["masks"][0].tolist().count(True) == 3
    assert outputs["masks"][0, 3:].logical_not().all()
    assert outputs["metas"][0]["c3_indirect_selected_valid_len"] == 3


def test_base_detector_postprocesses_with_selector_returned_metas():
    class DummyDetector(BaseDetector):
        def forward_test(self, inputs, masks, metas=None, infer_cfg=None):
            selected_metas = [dict(metas[0], window_size=4, c3_indirect_fixed_axis=True)]
            predictions = (
                torch.tensor([[[0.0, 1.0]]]),
                torch.tensor([[[0.9]]]),
            )
            return dict(predictions=predictions, metas=selected_metas)

        def post_processing(self, predictions, metas, post_cfg, **kwargs):
            assert metas[0]["window_size"] == 4
            assert metas[0]["c3_indirect_fixed_axis"] is True
            return {"used_window_size": metas[0]["window_size"]}

    detector = DummyDetector()
    infer_cfg = type("InferCfg", (), dict(load_from_raw_predictions=False, save_raw_prediction=False))()
    post_cfg = object()
    original_metas = [dict(video_name="video_0", window_size=8)]

    results = detector.forward_detection(
        torch.empty(1),
        torch.empty(1),
        original_metas,
        infer_cfg,
        post_cfg,
    )

    assert results == {"used_window_size": 4}
