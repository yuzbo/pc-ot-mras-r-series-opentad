import inspect

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


def _make_cadf_selector(target_len=4, dense_window_size=8, scout_spatial_size=4, **selector_overrides):
    selector_cfg = dict(
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
    selector_cfg.update(selector_overrides)
    return build_selector(
        selector_cfg
    )


def _make_inputs(dense_len=8, scout_spatial_size=4):
    return torch.arange(1 * 1 * 3 * dense_len * scout_spatial_size * scout_spatial_size, dtype=torch.float32).reshape(
        1, 1, 3, dense_len, scout_spatial_size, scout_spatial_size
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


def test_density_window_mass_and_gap_v2_loss_is_differentiable_on_continuous_density():
    selector = _make_cadf_selector(
        density_window_mass_loss_weight=0.0,
        density_max_gap_loss_weight=0.0,
    )
    density = torch.tensor(
        [[0.02, 0.03, 0.05, 0.70, 0.10, 0.10, 0.0, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    masks = torch.tensor([[True, True, True, True, True, True, False, False]])

    loss = selector._density_window_mass_gap_loss(density, masks)
    loss.backward()

    assert torch.isfinite(loss)
    assert density.grad is not None
    assert torch.isfinite(density.grad).all()
    assert density.grad[0, :6].abs().sum().item() > 0.0
    assert density.grad[0, 6:].abs().sum().item() == pytest.approx(0.0)


def test_density_blue_noise_v2_loss_is_finite_and_uses_continuous_density():
    selector = _make_cadf_selector(density_blue_noise_loss_weight=0.0)
    density = torch.tensor(
        [[0.32, 0.30, 0.28, 0.04, 0.03, 0.03, 0.0, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    masks = torch.tensor([[True, True, True, True, True, True, False, False]])

    loss = selector._density_blue_noise_repulsion_loss(density, masks)
    loss.backward()

    assert torch.isfinite(loss)
    assert density.grad is not None
    assert torch.isfinite(density.grad).all()
    assert density.grad[0, :6].abs().sum().item() > 0.0
    assert density.grad[0, 6:].abs().sum().item() == pytest.approx(0.0)


def test_density_weak_target_v2_loss_is_train_only_and_gt_derived():
    selector = _make_cadf_selector(
        density_alpha=1.0,
        actionness_loss_weight=0.0,
        density_weak_target_loss_weight=0.5,
    )
    selector.scout = StaticCADFScout(
        action_logits=[0, 0, 0, 0, 0, 0, 0, 0],
        utility_logits=[0, 0, 0, 0, 0, 0, 0, 0],
    )
    inputs = _make_inputs()
    masks = torch.ones(1, 8, dtype=torch.bool)
    metas = [dict(video_name="video_0", window_size=8, snippet_stride=4)]
    gt_segments = [torch.tensor([[2.0, 5.0]], dtype=torch.float32)]
    gt_labels = [torch.tensor([1], dtype=torch.long)]

    train_outputs = selector.forward_train(inputs, masks, metas, gt_segments, gt_labels)
    test_outputs = selector.forward_test(inputs, masks, metas)

    assert "loss_c3_density_weak_target" in train_outputs["losses"]
    assert torch.isfinite(train_outputs["losses"]["loss_c3_density_weak_target"])
    assert "losses" not in test_outputs
    assert "c3_density_weak_target_loss_enabled" in train_outputs["metas"][0]
    assert "c3_density_weak_target_loss_enabled" not in test_outputs["metas"][0]


def test_cadf_loss_select_distribution_objective_is_finite_and_backpropagates():
    selector = _make_cadf_selector(
        density_distribution_loss_weight=0.3,
        density_distribution_loss_weights=dict(
            smooth=0.4,
            local_cap=0.6,
            large_gap=0.8,
            collapse=0.5,
            target_kl=0.7,
        ),
        density_distribution_train_gt_target_weight=0.25,
    )
    density = torch.tensor(
        [[0.03, 0.05, 0.08, 0.36, 0.18, 0.14, 0.10, 0.06]],
        dtype=torch.float32,
        requires_grad=True,
    )
    masks = torch.ones(1, 8, dtype=torch.bool)
    scout_outputs = {
        "action_logits": torch.tensor([[-4.0, -2.0, 0.0, 1.5, 3.0, 1.0, -1.0, -3.0]]),
        "utility_logits": torch.zeros(1, 8),
    }
    gt_segments = [torch.tensor([[2.0, 6.0]], dtype=torch.float32)]

    loss, parts, target = selector._density_distribution_objective(
        density,
        masks,
        scout_outputs,
        gt_segments=gt_segments,
        return_parts=True,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert set(parts) == {
        "smooth",
        "local_cap",
        "large_gap",
        "collapse",
        "target_kl",
    }
    assert torch.isfinite(torch.stack(list(parts.values()))).all()
    assert torch.isfinite(target).all()
    assert target[0].sum().item() == pytest.approx(1.0)
    assert density.grad is not None
    assert torch.isfinite(density.grad).all()
    assert density.grad.abs().sum().item() > 0.0


def test_cadf_loss_select_penalizes_collapsed_and_large_gap_distributions_more_than_uniform():
    selector = _make_cadf_selector(
        target_len=4,
        dense_window_size=16,
        density_distribution_loss_weights=dict(
            smooth=0.0,
            local_cap=1.0,
            large_gap=1.0,
            collapse=1.0,
            target_kl=0.0,
        ),
    )
    masks = torch.ones(1, 16, dtype=torch.bool)
    scout_outputs = {
        "action_logits": torch.zeros(1, 16),
        "utility_logits": torch.zeros(1, 16),
    }
    uniform = torch.ones(1, 16, dtype=torch.float32) / 16.0
    collapsed = torch.ones(1, 16, dtype=torch.float32) * (0.04 / 15.0)
    collapsed[0, 7] = 0.96

    uniform_loss = selector._density_distribution_objective(uniform, masks, scout_outputs)
    collapsed_loss = selector._density_distribution_objective(collapsed, masks, scout_outputs)

    assert collapsed_loss.item() > uniform_loss.item()


def test_cadf_loss_select_train_only_gt_target_and_test_no_leakage():
    selector = _make_cadf_selector(
        density_distribution_loss_weight=0.2,
        density_distribution_loss_weights=dict(
            smooth=0.2,
            local_cap=0.2,
            large_gap=0.2,
            collapse=0.2,
            target_kl=1.0,
        ),
        density_distribution_train_gt_target_weight=0.5,
    )
    selector.scout = StaticCADFScout(
        action_logits=[-4, -2, 0, 2, 4, 2, 0, -2],
        utility_logits=[0, 0, 0, 0, 0, 0, 0, 0],
    )
    inputs = _make_inputs()
    masks = torch.ones(1, 8, dtype=torch.bool)
    metas = [dict(video_name="video_0", window_size=8, snippet_stride=4)]
    gt_segments = [torch.tensor([[2.0, 6.0]], dtype=torch.float32)]
    gt_labels = [torch.tensor([1], dtype=torch.long)]

    train_outputs = selector.forward_train(inputs, masks, metas, gt_segments, gt_labels)
    test_outputs = selector.forward_test(inputs, masks, metas)

    assert "loss_c3_density_distribution" in train_outputs["losses"]
    assert torch.isfinite(train_outputs["losses"]["loss_c3_density_distribution"])
    assert "losses" not in test_outputs
    assert train_outputs["metas"][0]["c3_density_distribution_loss_enabled"] is True
    assert train_outputs["metas"][0]["c3_density_distribution_train_gt_target_enabled"] is True
    assert "c3_density_distribution_train_gt_target_enabled" not in test_outputs["metas"][0]
    assert torch.equal(train_outputs["selected_dense_indices"], test_outputs["selected_dense_indices"])


def test_cadf_loss_select_diagnostics_report_density_gaps_duplicates_and_scout_support():
    selector = _make_cadf_selector(target_len=6, dense_window_size=12)
    selector.scout = StaticCADFScout(
        action_logits=[-5, -5, -2, 0, 2, 4, 2, 0, -2, -5, -5, -5],
        utility_logits=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    )
    inputs = _make_inputs(dense_len=12)
    masks = torch.ones(1, 12, dtype=torch.bool)
    metas = [dict(video_name="video_0", window_size=12, snippet_stride=4)]

    outputs = selector.forward_test(inputs, masks, metas)
    meta = outputs["metas"][0]

    assert len(meta["c3_density_mesh_density_histogram_8"]) == 8
    assert meta["c3_density_mesh_duplicate_count"] >= 0
    assert 0.0 <= meta["c3_density_mesh_uncertainty_selected_fraction"] <= 1.0
    assert 0.0 <= meta["c3_density_mesh_change_selected_fraction"] <= 1.0
    assert 0.0 <= meta["c3_density_mesh_distribution_target_selected_fraction"] <= 1.0


def test_cadf_loss_select_is_finite_for_extreme_logits_density_and_short_valid_lengths():
    selector = _make_cadf_selector(
        target_len=4,
        dense_window_size=8,
        density_distribution_loss_weight=0.5,
        density_distribution_loss_weights=dict(
            smooth=1.0,
            local_cap=1.0,
            large_gap=1.0,
            collapse=1.0,
            target_kl=1.0,
        ),
    )
    masks = torch.tensor(
        [
            [True, False, False, False, False, False, False, False],
            [True, True, False, False, False, False, False, False],
            [True, True, True, False, False, False, False, False],
        ]
    )
    raw_density = torch.tensor(
        [
            [float("nan"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [float("inf"), -float("inf"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1000.0, 0.0, -1000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    scout_outputs = {
        "action_logits": torch.tensor(
            [
                [float("inf"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [10000.0, -10000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [float("nan"), 10000.0, -10000.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        "utility_logits": torch.zeros(3, 8),
    }

    density = selector._sanitize_density(raw_density, masks)
    loss, parts, target = selector._density_distribution_objective(
        density,
        masks,
        scout_outputs,
        return_parts=True,
    )
    loss.backward()

    assert torch.isfinite(density).all()
    assert torch.isfinite(loss)
    assert torch.isfinite(torch.stack(list(parts.values()))).all()
    assert torch.isfinite(target).all()
    assert raw_density.grad is not None
    assert torch.isfinite(raw_density.grad).all()


def test_cadf_forward_train_sanitizes_extreme_scout_outputs_before_st_and_losses():
    selector = _make_cadf_selector(
        density_distribution_loss_weight=0.2,
        density_distribution_loss_weights=dict(
            smooth=0.2,
            local_cap=0.2,
            large_gap=0.2,
            collapse=0.2,
            target_kl=0.2,
        ),
        actionness_loss_weight=0.2,
        st_local_radius=2,
        st_scale=0.5,
    )
    selector.scout = StaticCADFScout(
        action_logits=[float("inf"), -float("inf"), float("nan"), 10000.0, -10000.0, 0.0, 20.0, -20.0],
        utility_logits=[float("nan"), 10000.0, -10000.0, float("inf"), -float("inf"), 0.0, 20.0, -20.0],
    )
    inputs = _make_inputs()
    masks = torch.ones(1, 8, dtype=torch.bool)
    metas = [dict(video_name="video_0", window_size=8, snippet_stride=4)]
    gt_segments = [torch.tensor([[1.0, 6.0]], dtype=torch.float32)]
    gt_labels = [torch.tensor([1], dtype=torch.long)]

    outputs = selector.forward_train(inputs, masks, metas, gt_segments, gt_labels)

    assert torch.isfinite(outputs["inputs"]).all()
    assert torch.isfinite(outputs["density"]).all()
    assert torch.isfinite(outputs["action_logits"]).all()
    assert all(torch.isfinite(loss) for loss in outputs["losses"].values())


def test_cadf_loss_select_amp_autocast_keeps_selector_outputs_and_losses_finite():
    selector = _make_cadf_selector(
        density_distribution_loss_weight=0.2,
        density_distribution_loss_weights=dict(
            smooth=0.2,
            local_cap=0.2,
            large_gap=0.2,
            collapse=0.2,
            target_kl=0.2,
        ),
        actionness_loss_weight=0.2,
    )
    selector.scout = StaticCADFScout(
        action_logits=[-20.0, -10.0, 0.0, 10.0, 20.0, 10.0, 0.0, -10.0],
        utility_logits=[20.0, -20.0, 10.0, -10.0, 0.0, 5.0, -5.0, 0.0],
    )
    inputs = _make_inputs()
    masks = torch.ones(1, 8, dtype=torch.bool)
    metas = [dict(video_name="video_0", window_size=8, snippet_stride=4)]
    gt_segments = [torch.tensor([[1.0, 6.0]], dtype=torch.float32)]
    gt_labels = [torch.tensor([1], dtype=torch.long)]

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        outputs = selector.forward_train(inputs, masks, metas, gt_segments, gt_labels)

    assert outputs["action_logits"].dtype == torch.float32
    assert outputs["density"].dtype == torch.float32
    assert torch.isfinite(outputs["inputs"]).all()
    assert torch.isfinite(outputs["density"]).all()
    assert all(torch.isfinite(loss.float()) for loss in outputs["losses"].values())


def test_cadf_loss_select_all_padding_density_objective_is_finite_boundary_case():
    selector = _make_cadf_selector(
        target_len=4,
        dense_window_size=8,
        density_distribution_loss_weight=0.2,
    )
    masks = torch.zeros(1, 8, dtype=torch.bool)
    density = torch.full((1, 8), float("nan"), dtype=torch.float32, requires_grad=True)
    scout_outputs = {
        "action_logits": torch.full((1, 8), float("inf")),
        "utility_logits": torch.full((1, 8), -float("inf")),
    }

    safe_density = selector._sanitize_density(density, masks)
    loss, parts, target = selector._density_distribution_objective(
        safe_density,
        masks,
        scout_outputs,
        return_parts=True,
    )
    loss.backward()

    assert torch.isfinite(safe_density).all()
    assert torch.isfinite(target).all()
    assert torch.isfinite(loss)
    assert torch.isfinite(torch.stack(list(parts.values()))).all()
    assert density.grad is not None
    assert torch.isfinite(density.grad).all()


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


def test_cadf_max_gap_guard_handles_compact_scores_on_nonzero_valid_indices():
    selector = _make_cadf_selector(target_len=4, dense_window_size=10)
    selector.max_gap_guard_count = 2
    valid_idx = torch.tensor([2, 3, 5, 7, 9], dtype=torch.long)
    density = torch.zeros(10, dtype=torch.float32)
    density[valid_idx] = torch.tensor([0.35, 0.05, 0.05, 0.05, 0.50])
    density = density / density.sum()

    selected, repair_mask = selector._inverse_cdf_select_one(density, valid_idx)

    assert selected.numel() == selector.target_len
    assert set(selected.tolist()).issubset(set(valid_idx.tolist()))
    assert repair_mask.float().sum().item() > 0.0
    assert repair_mask.float().mean().item() < 1.0


def test_cadf_repair_diagnostics_split_dedupe_guard_add_prune_and_row_fraction():
    selector = _make_cadf_selector(target_len=4, dense_window_size=10)
    selector.max_gap_guard_count = 2
    valid_idx = torch.tensor([2, 3, 5, 7, 9], dtype=torch.long)
    density = torch.zeros(10, dtype=torch.float32)
    density[valid_idx] = torch.tensor([0.35, 0.05, 0.05, 0.05, 0.50])
    density = density / density.sum()

    selected, repair_mask, repair_stats = selector._inverse_cdf_select_one(
        density,
        valid_idx,
        return_diagnostics=True,
    )
    diagnostics = selector._selection_diagnostics(
        selected[None],
        torch.ones(1, 10, dtype=torch.bool),
        density[None],
        repair_mask[None],
        repair_stats=[repair_stats],
    )

    assert repair_stats["dedupe_repair_count"] >= 0
    assert repair_stats["gap_guard_add_count"] > 0
    assert repair_stats["gap_guard_prune_count"] > 0
    assert diagnostics[0]["row_repair_fraction"] == pytest.approx(repair_mask.float().mean().item())
    assert diagnostics[0]["row_repair_fraction"] < 1.0


def test_cadf_fast_cpu_selection_matches_legacy_guarded_selection():
    legacy = _make_cadf_selector(
        target_len=8,
        dense_window_size=24,
        max_gap_guard_count=4,
        fast_cpu_selection=False,
    )
    fast = _make_cadf_selector(
        target_len=8,
        dense_window_size=24,
        max_gap_guard_count=4,
        fast_cpu_selection=True,
    )
    valid_idx = torch.arange(24)
    density = torch.ones(24, dtype=torch.float32) * 0.001
    density[2] = 4.0
    density[21] = 3.5
    density[10:14] = torch.tensor([0.25, 0.35, 0.45, 0.55])
    density = density / density.sum()

    legacy_selected, legacy_repair, legacy_stats = legacy._inverse_cdf_select_one(
        density,
        valid_idx,
        return_diagnostics=True,
    )
    fast_selected, fast_repair, fast_stats = fast._inverse_cdf_select_one(
        density,
        valid_idx,
        return_diagnostics=True,
    )

    assert torch.equal(fast_selected, legacy_selected)
    assert torch.equal(fast_repair, legacy_repair)
    assert fast_stats == legacy_stats


def test_cadf_formal_can_disable_per_iter_selection_diagnostics(monkeypatch):
    selector = _make_cadf_selector(
        emit_selection_diagnostics=False,
        selection_diagnostics_interval=0,
    )
    selector.scout = StaticCADFScout(
        action_logits=[0, 0, 0, 0, 0, 0, 0, 0],
        utility_logits=[-20, -20, 6, 6, 6, 6, -20, -20],
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("formal fast path should not emit per-iteration selection diagnostics")

    monkeypatch.setattr(selector, "_selection_diagnostics", fail_if_called)

    outputs = selector.forward_train(
        _make_inputs(),
        torch.ones(1, 8, dtype=torch.bool),
        [dict(video_name="video_0", window_size=8, snippet_stride=4)],
        [torch.tensor([[2.0, 6.0]], dtype=torch.float32)],
        [torch.tensor([1], dtype=torch.long)],
    )

    meta = outputs["metas"][0]
    assert meta["c3_density_mesh_alpha"] == pytest.approx(1.0)
    assert meta["c3_density_mesh_nonuniform_selection"] is True
    assert "c3_density_mesh_max_gap" not in meta
    assert "c3_density_mesh_repair_fraction" not in meta


def test_cadf_fast_cpu_selection_does_not_call_legacy_gpu_scalar_prune(monkeypatch):
    selector = _make_cadf_selector(
        target_len=4,
        dense_window_size=10,
        max_gap_guard_count=2,
        fast_cpu_selection=True,
    )
    valid_idx = torch.tensor([2, 3, 5, 7, 9], dtype=torch.long)
    density = torch.zeros(10, dtype=torch.float32)
    density[valid_idx] = torch.tensor([0.35, 0.05, 0.05, 0.05, 0.50])
    density = density / density.sum()

    def fail_legacy_prune(*args, **kwargs):
        raise AssertionError("fast CPU selection must not call legacy GPU scalar prune")

    monkeypatch.setattr(selector, "_prune_guarded_selection", fail_legacy_prune)

    selected, repair_mask, repair_stats = selector._inverse_cdf_select_one(
        density,
        valid_idx,
        return_diagnostics=True,
    )

    assert selected.numel() == selector.target_len
    assert repair_mask.numel() == selector.target_len
    assert repair_stats["gap_guard_add_count"] > 0


def test_cadf_robust_normalize_rows_has_no_row_level_scalar_sync():
    source = inspect.getsource(type(_make_cadf_selector())._robust_normalize_rows)

    assert ".item()" not in source
    assert "for row_idx" not in source


def test_cadf_max_gap_guard_reduces_real_gap_for_two_peak_density():
    selector = _make_cadf_selector(target_len=4, dense_window_size=16)
    valid_idx = torch.arange(16)
    density = torch.ones(16, dtype=torch.float32) * 0.001
    density[0] = 1.0
    density[15] = 1.0
    density = density / density.sum()

    selector.max_gap_guard_count = 0
    without_guard, _ = selector._inverse_cdf_select_one(density, valid_idx)
    selector.max_gap_guard_count = 2
    with_guard, _ = selector._inverse_cdf_select_one(density, valid_idx)

    gap_without = int((without_guard[1:] - without_guard[:-1]).max().item())
    gap_with = int((with_guard[1:] - with_guard[:-1]).max().item())
    assert gap_with < gap_without


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


def test_cadf_staged_alpha_schedule_advances_in_train_and_uses_explicit_test_policy():
    selector = build_selector(
        dict(
            type="PCOTMRASIndirectPreBackboneFrameSelector",
            target_len=4,
            dense_window_size=8,
            selection_unit=1,
            scout_spatial_size=4,
            strategy="cadf_density_mesh_st",
            density_alpha=0.7,
            density_alpha_schedule=dict(train_start_alpha=0.0, train_target_alpha=0.7, warmup_iters=2, test_alpha="target"),
            density_weights=dict(action=0.0, uncertainty=0.55, change=0.35, utility=0.10, boundary=0.0),
            scout=dict(type="PCOTMRASCADFDensityFrameScout", in_channels=48, hidden_channels=8, num_layers=1),
        )
    )
    selector.scout = StaticCADFScout(
        action_logits=[-4, -2, 0, 2, 4, 2, 0, -2],
        utility_logits=[0, 0, 0, 0, 0, 0, 0, 0],
    )
    inputs = _make_inputs()
    masks = torch.ones(1, 8, dtype=torch.bool)
    metas = [dict(video_name="video_0", window_size=8, snippet_stride=4)]
    gt_segments = [torch.tensor([[1.0, 7.0]], dtype=torch.float32)]
    gt_labels = [torch.tensor([1], dtype=torch.long)]

    first = selector.forward_train(inputs, masks, metas, gt_segments, gt_labels)
    second = selector.forward_train(inputs, masks, metas, gt_segments, gt_labels)
    test = selector.forward_test(inputs, masks, metas)

    assert first["metas"][0]["c3_density_mesh_alpha"] == pytest.approx(0.0)
    assert second["metas"][0]["c3_density_mesh_alpha"] == pytest.approx(0.35)
    assert test["metas"][0]["c3_density_mesh_alpha"] == pytest.approx(0.7)
    assert test["metas"][0]["c3_density_mesh_test_alpha_policy"] == "target"
    assert test["metas"][0]["c3_density_mesh_alpha_schedule_scope"] == "diagnostic_short_smoke_not_resumable"
    assert test["metas"][0]["c3_density_mesh_alpha_schedule_recoverable"] is False


def test_cadf_default_density_weights_prioritize_uncertainty_and_transition_not_action():
    selector = build_selector(
        dict(
            type="PCOTMRASIndirectPreBackboneFrameSelector",
            target_len=4,
            dense_window_size=8,
            selection_unit=1,
            scout_spatial_size=4,
            strategy="cadf_density_mesh_st",
            scout=dict(type="PCOTMRASCADFDensityFrameScout", in_channels=48, hidden_channels=8, num_layers=1),
        )
    )

    assert selector.density_weights["action"] <= 0.05
    assert selector.density_weights["uncertainty"] > selector.density_weights["action"]
    assert selector.density_weights["change"] > selector.density_weights["action"]


def test_cadf_high_entropy_background_does_not_monopolize_selection():
    selector = _make_cadf_selector(target_len=4, dense_window_size=12)
    selector.density_alpha = 1.0
    selector.density_weights = dict(action=0.0, uncertainty=0.65, change=0.25, utility=0.10, boundary=0.0)
    selector.max_gap_guard_count = 2
    masks = torch.ones(1, 12, dtype=torch.bool)
    # Frames 0-5 are uncertain but very low-action background; frames 7-10 form
    # the weak action/context support that should keep CADF from selecting only noise.
    scout_outputs = {
        "action_logits": torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -4.0, 1.2, 1.5, 1.2, 0.8, -4.0]]),
        "utility_logits": torch.zeros(1, 12),
    }

    density, _ = selector._cadf_density(scout_outputs, masks)
    selected, _ = selector._inverse_cdf_select_one(density[0], masks[0].nonzero(as_tuple=True)[0])

    background_count = sum(1 for idx in selected.tolist() if idx <= 5)
    action_context_count = sum(1 for idx in selected.tolist() if 7 <= idx <= 10)
    assert background_count < selector.target_len
    assert action_context_count >= 1


def test_cadf_transition_density_is_smoothed_and_keeps_action_neighborhood_support():
    selector = _make_cadf_selector(target_len=5, dense_window_size=12)
    selector.density_alpha = 1.0
    selector.density_weights = dict(action=0.0, uncertainty=0.10, change=0.75, utility=0.15, boundary=0.0)
    masks = torch.ones(1, 12, dtype=torch.bool)
    scout_outputs = {
        "action_logits": torch.tensor([[-5.0, -5.0, -4.0, -1.0, 1.0, 2.0, 2.2, 1.0, -1.0, -4.0, -5.0, -5.0]]),
        "utility_logits": torch.zeros(1, 12),
    }

    density, _ = selector._cadf_density(scout_outputs, masks)
    selected, _ = selector._inverse_cdf_select_one(density[0], masks[0].nonzero(as_tuple=True)[0])

    assert torch.isfinite(density).all()
    assert any(3 <= int(idx) <= 5 for idx in selected.tolist())
    assert any(6 <= int(idx) <= 8 for idx in selected.tolist())
    assert density[0, 5].item() > 0.0
    assert density[0, 6].item() > 0.0


def test_cadf_alpha_zero_backend_control_selects_exact_uniform_like_indices_and_contract():
    selector = _make_cadf_selector(target_len=4, dense_window_size=8)
    selector.density_alpha = 0.0
    selector.scout = StaticCADFScout(
        action_logits=[-4, 4, -4, 4, -4, 4, -4, 4],
        utility_logits=[9, -9, 9, -9, 9, -9, 9, -9],
    )
    inputs = _make_inputs()
    masks = torch.ones(1, 8, dtype=torch.bool)
    metas = [dict(video_name="video_0", window_size=8, snippet_stride=4)]

    outputs = selector.forward_test(inputs, masks, metas)

    assert outputs["selected_dense_indices"][0].tolist() == [0, 2, 4, 6]
    assert outputs["metas"][0]["c3_density_mesh_alpha"] == pytest.approx(0.0)
    assert outputs["metas"][0]["c3_backend_uses_average_stride"] is True
    assert outputs["metas"][0]["c3_physical_coords_unused_by_backend"] is True


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

    remapped = selector._remap_metas(
        metas,
        selected,
        dense_masks,
        selection_diagnostics=[
            dict(
                max_gap=3,
                mean_gap=2.0,
                repair_fraction=0.25,
                repair_count=1,
                density_entropy=0.75,
                density_top_positions=[5, 7, 2],
            )
        ],
    )

    assert remapped[0]["c3_density_mesh_nonuniform_selection"] is True
    assert remapped[0]["c3_backend_uses_average_stride"] is True
    assert remapped[0]["c3_physical_coords_unused_by_backend"] is True
    assert remapped[0]["c3_indirect_selected_mask"] == [True, True, True, True]
    assert remapped[0]["c3_density_mesh_repair_count"] == 1
    assert remapped[0]["c3_density_mesh_density_top_positions"] == [5, 7, 2]
    assert remapped[0]["c3_density_mesh_average_stride_restoration_error"] > 0.0


def test_cadf_train_metas_include_gt_remap_ratio_without_test_gt_pollution():
    selector = _make_cadf_selector(target_len=4, dense_window_size=8)
    selector.scout = StaticCADFScout(
        action_logits=[0, 2, 0, -2, 0, 2, 0, -2],
        utility_logits=[0, 0, 0, 0, 0, 0, 0, 0],
    )
    inputs = _make_inputs()
    masks = torch.ones(1, 8, dtype=torch.bool)
    metas = [dict(video_name="video_0", window_size=8, snippet_stride=4)]
    gt_segments = [torch.tensor([[1.0, 7.0], [3.0, 4.0]], dtype=torch.float32)]
    gt_labels = [torch.tensor([1, 2], dtype=torch.long)]

    train_outputs = selector.forward_train(inputs, masks, metas, gt_segments, gt_labels)
    test_outputs = selector.forward_test(inputs, masks, metas)

    assert "c3_train_gt_remap_length_ratio_mean" in train_outputs["metas"][0]
    assert "c3_train_gt_remap_length_ratio_mean" not in test_outputs["metas"][0]
