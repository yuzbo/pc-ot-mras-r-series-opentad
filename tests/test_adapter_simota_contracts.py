from pathlib import Path
import importlib.util
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]


def read(rel_path):
    return (ROOT / rel_path).read_text(encoding="utf-8")


def load_simota_module():
    torch = import_torch_or_skip()
    module_name = "opentad.models.losses.assigner.anchor_free_simota_assigner"
    module_path = ROOT / "opentad/models/losses/assigner/anchor_free_simota_assigner.py"

    modules = {
        "opentad": types.ModuleType("opentad"),
        "opentad.models": types.ModuleType("opentad.models"),
        "opentad.models.losses": types.ModuleType("opentad.models.losses"),
        "opentad.models.losses.assigner": types.ModuleType("opentad.models.losses.assigner"),
        "opentad.models.losses.builder": types.ModuleType("opentad.models.losses.builder"),
        "opentad.models.builder": types.ModuleType("opentad.models.builder"),
        "opentad.models.losses.focal_loss": types.ModuleType("opentad.models.losses.focal_loss"),
    }

    class DummyRegistry:
        def register_module(self):
            return lambda cls: cls

    modules["opentad.models.builder"].LOSSES = DummyRegistry()
    modules["opentad.models.losses.focal_loss"].sigmoid_focal_loss = (
        lambda out_scores, labels: torch.zeros_like(labels, dtype=out_scores.dtype)
    )

    old_modules = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old_module in old_modules.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


def import_torch_or_skip():
    if sys.platform.startswith("win"):
        pytest.skip("torch tensor tests run on the Linux training environment")
    try:
        import torch
    except Exception as exc:
        pytest.skip(f"torch import unavailable in this environment: {exc}")
    return torch


def test_simota_dynamic_k_rejects_unknown_nested_options():
    simota = load_simota_module()

    with pytest.raises(ValueError, match="Unsupported dynamic_k options"):
        simota.AnchorFreeSimOTAAssigner(dynamic_k=dict(type="dynamic_k_matching", typo_option=3))


def test_simota_iou_sum_dynamic_k_clamps_by_min_k_and_candidates():
    torch = import_torch_or_skip()
    simota = load_simota_module()
    assigner = simota.AnchorFreeSimOTAAssigner(
        topk=3,
        min_k=4,
        dynamic_k=dict(type="dynamic_k_matching", mode="iou_sum"),
    )

    cost = torch.tensor(
        [
            [0.1, 5.0],
            [0.2, 0.1],
            [0.3, 0.2],
            [0.4, 0.3],
            [5.0, 0.4],
        ],
        dtype=torch.float32,
    )
    positive_pos = torch.tensor(
        [
            [True, False],
            [True, True],
            [True, True],
            [True, True],
            [False, True],
        ]
    )
    ious = torch.tensor(
        [
            [0.7, 0.0],
            [0.6, 0.1],
            [0.5, 0.1],
            [0.4, 0.1],
            [0.0, 0.1],
        ],
        dtype=torch.float32,
    )

    matrix, min_inds, weights = assigner.dynamic_k_matching(
        cost=cost,
        num_gt=2,
        valid_mask=torch.ones(cost.shape[0], dtype=torch.bool),
        valid_cost_matrix_inds=positive_pos,
        pairwise_ious=ious,
    )

    stats = assigner.get_last_stats()
    assert stats["dynamic_k_mode"] == "iou_sum"
    assert stats["candidate_counts"] == [4, 4]
    assert stats["dynamic_ks"] == [4, 4]
    assert stats["matched_counts"] == [1, 4]
    assert int(matrix.sum().item()) == 5
    assert min_inds.tolist() == [0, 1, 1, 1, 1]
    assert weights.tolist() == [1.0] * 5


def test_simota_iou_sum_handles_zero_candidate_gt():
    torch = import_torch_or_skip()
    simota = load_simota_module()
    assigner = simota.AnchorFreeSimOTAAssigner(
        topk=3,
        min_k=4,
        dynamic_k=dict(type="dynamic_k_matching", mode="iou_sum"),
    )

    cost = torch.tensor(
        [
            [0.1, 5.0],
            [0.2, 5.0],
            [0.3, 5.0],
        ],
        dtype=torch.float32,
    )
    positive_pos = torch.tensor(
        [
            [True, False],
            [True, False],
            [True, False],
        ]
    )
    ious = torch.tensor(
        [
            [0.7, 0.0],
            [0.6, 0.0],
            [0.5, 0.0],
        ],
        dtype=torch.float32,
    )

    matrix, min_inds, weights = assigner.dynamic_k_matching(
        cost=cost,
        num_gt=2,
        valid_mask=torch.ones(cost.shape[0], dtype=torch.bool),
        valid_cost_matrix_inds=positive_pos,
        pairwise_ious=ious,
    )

    stats = assigner.get_last_stats()
    assert stats["candidate_counts"] == [3, 0]
    assert stats["dynamic_ks"] == [3, 0]
    assert stats["matched_counts"] == [3, 0]
    assert matrix[:, 1].sum().item() == 0
    assert min_inds.tolist() == [0, 0, 0]
    assert weights.tolist() == [1.0] * 3


def test_simota_shortest_gt_filter_can_be_disabled_for_overlaps():
    torch = import_torch_or_skip()
    simota = load_simota_module()
    points = torch.tensor([[10.0, 0.0, 100.0, 1.0]])
    gt_segments = torch.tensor([[0.0, 20.0], [5.0, 15.0]])

    shortest = simota.AnchorFreeSimOTAAssigner(
        center_radius=1000.0,
        filter_shortest_gt=True,
    ).within_center(points, gt_segments)
    all_overlaps = simota.AnchorFreeSimOTAAssigner(
        center_radius=1000.0,
        filter_shortest_gt=False,
    ).within_center(points, gt_segments)

    assert shortest.tolist() == [[0.0, 1.0]]
    assert all_overlaps.tolist() == [[1.0, 1.0]]


def test_simota_source_rejects_unknown_dynamic_k_options():
    source = read("opentad/models/losses/assigner/anchor_free_simota_assigner.py")

    assert "Unsupported dynamic_k options" in source
    assert "if dynamic_k:" in source


def test_adapter_simota_plain_config_and_launcher_are_gated_controls():
    plain = read("configs/adatad/thumos/input_random_fixed_50pct_adapter_simota_mink4_w1.py")
    launch_script = read("scripts/run_adapter_simota_iou_sum.sh")

    assert "AnchorFreeSimOTAAssigner" in plain
    assert "topk=9" in plain
    assert "min_k=4" in plain
    assert "filter_shortest_gt=False" in plain
    assert 'mode="iou_sum"' in plain
    assert "assignment_debug=dict(enabled=True)" in plain
    assert "checkpoint_interval=10" in plain
    assert "val_eval_interval=2" in plain
    assert "val_start_epoch=40" in plain
    assert "disable_checkpoint=False" in plain
    assert "input_pdrop=0.2" not in plain

    assert "input_random_fixed_50pct_adapter_simota_mink4_w1.py" in launch_script
    assert "input_random_fixed_50pct_adapter_simota_center25_mink4_w1.py" not in launch_script
    assert "CHECK_ONLY" in launch_script
    assert "EXPECT_BATCH_SIZE" in launch_script
    assert 'assigner.type == "AnchorFreeSimOTAAssigner"' in launch_script
    assert 'assigner.dynamic_k.mode == "iou_sum"' in launch_script
    assert "assignment_debug.enabled" in launch_script
    assert "workflow.checkpoint_interval" in launch_script
    assert "cfg.solver.train.batch_size" in launch_script
    assert "CUDA_VISIBLE_DEVICES" in launch_script


def test_adapter_simota_center25_config_is_labeled_manual_composite_variant():
    center = read("configs/adatad/thumos/input_random_fixed_50pct_adapter_simota_center25_mink4_w1.py")

    assert '_base_ = ["./input_random_fixed_50pct_adapter_fcos_center25.py"]' in center
    assert "Manual composite diagnostic" in center
    assert "not part of the safe backup launcher" in center
