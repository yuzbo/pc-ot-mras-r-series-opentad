import torch
import torch.nn as nn

from opentad.models.builder import build_selector
from opentad.models.selectors.c3_indirect_frame_selector import PCOTMRASCADFDensityFrameScout


class StaticCADFScout(nn.Module):
    def __init__(self, action_logits, utility_logits):
        super().__init__()
        self.register_buffer("action_logits", torch.tensor(action_logits, dtype=torch.float32))
        self.register_buffer("utility_logits", torch.tensor(utility_logits, dtype=torch.float32))

    def forward(self, scout_inputs, masks=None):
        batch_size = scout_inputs.shape[0]
        action_logits = self.action_logits[None].expand(batch_size, -1).clone()
        utility_logits = self.utility_logits[None].expand(batch_size, -1).clone()
        if masks is not None:
            action_logits = action_logits.masked_fill(~masks, -20.0)
            utility_logits = utility_logits.masked_fill(~masks, -20.0)
        return {"action_logits": action_logits, "utility_logits": utility_logits}


def _make_cadf_selector(target_len=4, dense_window_size=8, scout_spatial_size=4):
    return build_selector(
        dict(
            type="PCOTMRASIndirectPreBackboneFrameSelector",
            target_len=target_len,
            dense_window_size=dense_window_size,
            selection_unit=1,
            scout_spatial_size=scout_spatial_size,
            strategy="cadf_density_mesh_st",
            density_alpha=1.0,
            density_temperature=1.0,
            density_weights=dict(action=0.0, uncertainty=0.0, change=0.0, utility=1.0, boundary=0.0),
            max_gap_guard_count=0,
            st_local_radius=1,
            st_scale=0.5,
            actionness_loss_weight=0.2,
            boundary_loss_weight=0.0,
            density_entropy_loss_weight=0.0,
            density_repulsion_loss_weight=0.0,
            scout=dict(
                type="PCOTMRASCADFDensityFrameScout",
                in_channels=3 * scout_spatial_size * scout_spatial_size,
                hidden_channels=8,
                num_layers=1,
                kernel_size=3,
                with_boundary_head=False,
            ),
        )
    )


def test_cadf_scout_outputs_action_and_utility_without_detector_heads():
    scout = PCOTMRASCADFDensityFrameScout(
        in_channels=3,
        hidden_channels=8,
        num_layers=1,
        kernel_size=3,
        with_boundary_head=True,
    )
    scout_inputs = torch.randn(2, 8, 3)
    masks = torch.ones(2, 8, dtype=torch.bool)

    outputs = scout(scout_inputs, masks)

    assert sorted(outputs.keys()) == ["action_logits", "boundary_logits", "utility_logits"]
    assert outputs["action_logits"].shape == (2, 8)
    assert outputs["utility_logits"].shape == (2, 8)
    assert outputs["boundary_logits"].shape == (2, 8)
    for forbidden_name in ["start_head", "end_head", "risk_head"]:
        assert not hasattr(scout, forbidden_name)


def test_cadf_density_mesh_selects_inverse_cdf_real_frames_without_quotas():
    selector = _make_cadf_selector()
    selector.scout = StaticCADFScout(
        action_logits=[0, 0, 0, 0, 0, 0, 0, 0],
        utility_logits=[-20, -20, 6, 6, 6, 6, -20, -20],
    )
    inputs = torch.arange(1 * 1 * 3 * 8 * 4 * 4, dtype=torch.float32).reshape(1, 1, 3, 8, 4, 4)
    masks = torch.ones(1, 8, dtype=torch.bool)
    metas = [dict(video_name="video_0", window_size=8, snippet_stride=4)]
    gt_segments = [torch.tensor([[2.0, 6.0]], dtype=torch.float32)]
    gt_labels = [torch.tensor([1], dtype=torch.long)]

    outputs = selector.forward_train(inputs, masks, metas, gt_segments, gt_labels)

    selected = outputs["selected_dense_indices"][0]
    assert selected.tolist() == [2, 3, 4, 5]
    assert not hasattr(selector, "quotas")
    for slot, dense_idx in enumerate(selected.tolist()):
        assert torch.equal(outputs["inputs"][0, 0, :, slot], inputs[0, 0, :, dense_idx])
    assert outputs["metas"][0]["c3_indirect_strategy"] == "cadf_density_mesh_st"
    assert outputs["metas"][0]["c3_density_mesh_alpha"] == 1.0
    assert outputs["metas"][0]["c3_density_mesh_max_gap"] == 1
    assert outputs["metas"][0]["c3_density_mesh_repair_fraction"] == 0.0


def test_cadf_train_and_test_selection_use_same_hard_policy():
    selector = _make_cadf_selector()
    selector.scout = StaticCADFScout(
        action_logits=[0, 0, 0, 0, 0, 0, 0, 0],
        utility_logits=[-20, 5, -20, 5, -20, 5, -20, 5],
    )
    inputs = torch.randn(1, 1, 3, 8, 4, 4)
    masks = torch.ones(1, 8, dtype=torch.bool)
    metas = [dict(video_name="video_0", window_size=8, snippet_stride=4)]
    gt_segments = [torch.tensor([[1.0, 7.0]], dtype=torch.float32)]
    gt_labels = [torch.tensor([1], dtype=torch.long)]

    train_outputs = selector.forward_train(inputs, masks, metas, gt_segments, gt_labels)
    test_outputs = selector.forward_test(inputs, masks, metas)

    assert torch.equal(train_outputs["selected_dense_indices"], test_outputs["selected_dense_indices"])
    assert "loss_c3_actionness" in train_outputs["losses"]
    assert "loss_c3_boundary" not in train_outputs["losses"]
    assert "losses" not in test_outputs


def test_cadf_density_mesh_rejects_category_quota_config():
    try:
        build_selector(
            dict(
                type="PCOTMRASIndirectPreBackboneFrameSelector",
                target_len=4,
                dense_window_size=8,
                selection_unit=1,
                scout_spatial_size=4,
                strategy="cadf_density_mesh_st",
                quotas=dict(uniform=4),
                scout=dict(type="PCOTMRASCADFDensityFrameScout", in_channels=48),
            )
        )
    except ValueError as exc:
        assert "quotas" in str(exc)
    else:
        raise AssertionError("CADF/DensityMesh must reject category quota configs")
