import torch
import torch.nn as nn
import pytest

from opentad.models.builder import build_selector
from opentad.models.selectors.c3_indirect_frame_selector import PCOTMRASCADFDensityFrameScout


class StaticCADFScout(nn.Module):
    def __init__(self, action_logits, utility_logits, boundary_logits=None):
        super().__init__()
        self.register_buffer("action_logits", torch.tensor(action_logits, dtype=torch.float32))
        self.register_buffer("utility_logits", torch.tensor(utility_logits, dtype=torch.float32))
        if boundary_logits is not None:
            self.register_buffer("boundary_logits", torch.tensor(boundary_logits, dtype=torch.float32))
        else:
            self.boundary_logits = None

    def forward(self, scout_inputs, masks=None):
        batch_size = scout_inputs.shape[0]
        action_logits = self.action_logits[None].expand(batch_size, -1).clone()
        utility_logits = self.utility_logits[None].expand(batch_size, -1).clone()
        outputs = {"action_logits": action_logits, "utility_logits": utility_logits}
        if self.boundary_logits is not None:
            outputs["boundary_logits"] = self.boundary_logits[None].expand(batch_size, -1).clone()
        if masks is not None:
            for key, value in outputs.items():
                outputs[key] = value.masked_fill(~masks, -20.0)
        return outputs


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


def test_cadf_inverse_cdf_duplicate_repair_stays_local_instead_of_global_topk():
    selector = _make_cadf_selector(target_len=8, dense_window_size=16)
    valid_idx = torch.arange(16)
    density = torch.ones(16, dtype=torch.float32) * 0.001
    density[7] = 100.0
    density[9:] = torch.tensor([0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96])
    density = density / density.sum()

    selected, repair_mask = selector._inverse_cdf_select_one(density, valid_idx)

    global_topk = set(valid_idx[density.argsort(descending=True)[: selector.target_len]].tolist())
    assert set(selected.tolist()) != global_topk
    assert 6 in selected.tolist()
    assert 8 in selected.tolist()
    assert repair_mask.float().mean().item() > 0.0


def test_cadf_density_is_row_robust_to_logit_scale_and_extreme_finite_values():
    selector = _make_cadf_selector(target_len=4, dense_window_size=8)
    selector.density_weights = dict(action=0.5, uncertainty=0.0, change=0.0, utility=0.5, boundary=0.0)
    masks = torch.ones(2, 8, dtype=torch.bool)
    base_outputs = {
        "action_logits": torch.tensor([[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]]),
        "utility_logits": torch.tensor([[3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0]]),
    }
    scaled_outputs = {
        "action_logits": base_outputs["action_logits"] * 1000.0,
        "utility_logits": base_outputs["utility_logits"] * 0.001,
    }

    base_density, base_logits = selector._cadf_density(base_outputs, masks[:1])
    scaled_density, scaled_logits = selector._cadf_density(scaled_outputs, masks[:1])
    extreme_density, extreme_logits = selector._cadf_density(
        {
            "action_logits": torch.tensor([[10000.0, -10000.0, 5000.0, -5000.0, 0.0, 1.0, -1.0, 2.0]]),
            "utility_logits": torch.tensor([[-10000.0, 10000.0, -5000.0, 5000.0, 0.0, -1.0, 1.0, -2.0]]),
        },
        masks[:1],
    )

    assert torch.allclose(base_density, scaled_density, atol=1e-4, rtol=1e-4)
    assert torch.isfinite(base_density).all()
    assert torch.isfinite(base_logits).all()
    assert torch.isfinite(scaled_density).all()
    assert torch.isfinite(scaled_logits).all()
    assert torch.isfinite(extreme_density).all()
    assert torch.isfinite(extreme_logits).all()


def test_cadf_alpha_zero_and_one_produce_finite_valid_density():
    selector = _make_cadf_selector(target_len=4, dense_window_size=8)
    masks = torch.tensor([[True, True, True, True, True, False, False, False]])
    scout_outputs = {
        "action_logits": torch.tensor([[10000.0, -10000.0, 0.0, 5.0, -5.0, -20.0, -20.0, -20.0]]),
        "utility_logits": torch.tensor([[-5.0, 5.0, 0.0, -10000.0, 10000.0, -20.0, -20.0, -20.0]]),
    }

    selector.density_alpha = 0.0
    uniform_density, _ = selector._cadf_density(scout_outputs, masks)
    selector.density_alpha = 1.0
    learned_density, _ = selector._cadf_density(scout_outputs, masks)

    assert torch.isfinite(uniform_density).all()
    assert torch.isfinite(learned_density).all()
    assert uniform_density[0, :5].sum().item() == pytest.approx(1.0)
    assert learned_density[0, :5].sum().item() == pytest.approx(1.0)
    assert uniform_density[0, 5:].sum().item() == pytest.approx(0.0)
    assert learned_density[0, 5:].sum().item() == pytest.approx(0.0)


def test_cadf_remap_gt_clamps_to_selected_valid_length_when_tail_is_padding():
    selector = _make_cadf_selector(target_len=6, dense_window_size=8)
    selected = torch.tensor([[0, 1, 2, 2, 2, 2]], dtype=torch.long)
    dense_masks = torch.tensor([[True, True, True, False, False, False, False, False]])
    gt_segments = [torch.tensor([[0.0, 7.0]], dtype=torch.float32)]
    gt_labels = [torch.tensor([1], dtype=torch.long)]

    remapped_segments, remapped_labels = selector._remap_gt(selected, dense_masks, gt_segments, gt_labels)

    selected_valid_len = int(selector._selected_masks(dense_masks, selected)[0].sum().item())
    assert selected_valid_len < selector.target_len
    assert remapped_labels[0].tolist() == [1]
    assert remapped_segments[0].max().item() <= selected_valid_len


def test_cadf_remap_metas_marks_nonuniform_average_stride_backend_flags():
    selector = _make_cadf_selector(target_len=4, dense_window_size=8)
    selected = torch.tensor([[1, 2, 5, 7]], dtype=torch.long)
    dense_masks = torch.ones(1, 8, dtype=torch.bool)
    metas = [dict(video_name="video_0", window_size=8, snippet_stride=4)]

    remapped = selector._remap_metas(metas, selected, dense_masks, selection_diagnostics=[dict(max_gap=3, mean_gap=2.0, repair_fraction=0.25)])

    assert remapped[0]["c3_density_mesh_nonuniform_selection"] is True
    assert remapped[0]["c3_backend_uses_average_stride"] is True
    assert remapped[0]["c3_physical_coords_unused_by_backend"] is True
