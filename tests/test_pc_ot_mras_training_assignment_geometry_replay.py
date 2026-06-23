import platform

import pytest

if platform.system().lower() == "windows":
    pytest.skip("Windows local torch runtime is not reliable for this repo; run this test on Linux/N16R4.", allow_module_level=True)

try:
    import torch
except Exception as exc:  # pragma: no cover - environment guard
    pytest.skip(f"torch is required for assignment replay tests: {exc}", allow_module_level=True)

from tools.bata.replay_pc_ot_mras_training_assignment_geometry import (
    READY,
    aggregate_batches,
    summarize_loss_dict,
    summarize_area_grid,
    summarize_level_targets,
    summarize_metas,
    tensor_stats,
)


def _area_grid():
    return {
        "obs_center": torch.tensor([[1.0, 3.0, 6.0]]),
        "obs_start": torch.tensor([[0.5, 2.5, 5.5]]),
        "obs_end": torch.tensor([[1.5, 3.5, 6.5]]),
        "obs_width": torch.tensor([[1.0, 1.0, 1.0]]),
        "obs_valid_mask": torch.tensor([[1, 1, 1]], dtype=torch.bool),
        "gap_start": torch.tensor([[0.0, 1.5, 3.5, 6.5]]),
        "gap_end": torch.tensor([[0.5, 2.5, 5.5, 8.0]]),
        "gap_center": torch.tensor([[0.25, 2.0, 4.5, 7.25]]),
        "gap_width": torch.tensor([[0.5, 1.0, 2.0, 1.5]]),
        "gap_valid_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.bool),
        "dense_valid_len": torch.tensor([8.0]),
    }


def _targets():
    return {
        "area": torch.tensor([[[0.0, 0.0], [0.7, 0.0], [0.2, 0.0]]]),
        "start_gap": torch.tensor([[[0.0, 0.0], [0.8, 0.0], [0.1, 0.0], [0.0, 0.0]]]),
        "end_gap": torch.tensor([[[0.0, 0.0], [0.2, 0.0], [0.9, 0.0], [0.0, 0.0]]]),
        "start_offset": torch.zeros((1, 4, 2)),
        "end_offset": torch.zeros((1, 4, 2)),
        "start_offset_weight": torch.tensor([[[0.0, 0.0], [0.8, 0.0], [0.0, 0.0], [0.0, 0.0]]]),
        "end_offset_weight": torch.tensor([[[0.0, 0.0], [0.0, 0.0], [0.9, 0.0], [0.0, 0.0]]]),
    }


def test_summarize_level_targets_counts_valid_positive_assignments():
    summary = summarize_level_targets(0, _targets(), _area_grid())

    assert summary["level"] == 0
    assert summary["area_ge_0p5_count"] == 1
    assert summary["start_ge_0p5_count"] == 1
    assert summary["end_ge_0p5_count"] == 1
    assert summary["start_offset_weight_gt_0_count"] == 1
    assert summary["end_offset_weight_gt_0_count"] == 1
    assert summary["area_positive_mass"] == pytest.approx(0.9)


def test_tensor_stats_broadcasts_bt1_mask_over_btc_targets():
    tensor = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [9.0, 9.0]]])
    mask = torch.tensor([[[1], [1], [0]]], dtype=torch.bool)

    stats = tensor_stats(tensor, mask)

    assert stats["count"] == 4
    assert stats["mean"] == pytest.approx(2.5)


def test_summarize_area_grid_records_dense_axis_geometry():
    summary = summarize_area_grid(0, _area_grid())

    assert summary["obs_valid_count"] == 3
    assert summary["gap_valid_count"] == 4
    assert summary["center_monotonic_violation_count"] == 0
    assert summary["dense_valid_len"]["mean"] == 8.0
    assert summary["center_gap"]["mean"] == pytest.approx(2.5)


def test_summarize_metas_flags_missing_native_axis_and_nonmonotonic_positions():
    metas = [
        {
            "irregular_selected_positions": [1.0, 3.0, 2.5],
            "irregular_selected_valid_len": 8,
            "irregular_selected_count": 3,
        }
    ]

    rows = summarize_metas(metas, ["sample"])

    assert rows[0]["has_irregular_native_axis"] is False
    assert rows[0]["positions_strictly_increasing"] is False
    assert rows[0]["selected_position_count"] == 3


def test_aggregate_batches_preserves_diagnostic_only_interpretation():
    batch_rows = [
        {
            "sample_ids": ["sample"],
            "meta_summary": [
                {
                    "has_irregular_native_axis": True,
                    "positions_strictly_increasing": True,
                }
            ],
            "gt_coverage_summary": [
                {
                    "gt_with_area_ge_0p5": 1,
                    "gt_with_start_ge_0p5": 1,
                    "gt_with_end_ge_0p5": 1,
                }
            ],
            "level_grid_summary": [
                {
                    "center_monotonic_violation_count": 0,
                }
            ],
            "level_target_summary": [
                {
                    "area_positive_mass": 0.9,
                    "start_positive_mass": 0.9,
                    "end_positive_mass": 1.1,
                    "area_ge_0p5_count": 1,
                    "start_ge_0p5_count": 1,
                    "end_ge_0p5_count": 1,
                }
            ],
            "quality_calibration_summary": {
                "rows": [
                    {
                        "candidate_count_sampled": 4,
                        "quality_target": {"mean": 0.6},
                        "rank": {
                            "rank_pair_count": 2,
                            "rank_margin_violation_fraction": 0.5,
                        },
                    }
                ]
            },
            "actual_loss_summary": {
                "area_loss": {"scalar": 0.1},
                "quality_rank_loss": {"scalar": 0.2},
            },
        }
    ]

    aggregate = aggregate_batches(batch_rows)

    assert aggregate["batch_count"] == 1
    assert aggregate["sample_count"] == 1
    assert aggregate["samples_missing_irregular_native_axis"] == 0
    assert aggregate["samples_with_nonmonotonic_positions"] == 0
    assert aggregate["area_positive_mass"]["mean"] == pytest.approx(0.9)
    assert aggregate["quality_candidate_count_sampled"]["mean"] == pytest.approx(4.0)
    assert aggregate["quality_rank_pair_count"]["mean"] == pytest.approx(2.0)
    assert aggregate["actual_loss_scalar"]["quality_rank_loss"]["mean"] == pytest.approx(0.2)


def test_summarize_loss_dict_records_finite_scalars():
    losses = {"area_loss": torch.tensor(0.25), "vector_loss": torch.tensor([1.0, 3.0])}

    summary = summarize_loss_dict(losses)

    assert summary["area_loss"]["scalar"] == pytest.approx(0.25)
    assert summary["area_loss"]["finite"] is True
    assert summary["vector_loss"]["scalar"] is None
    assert summary["vector_loss"]["stats"]["mean"] == pytest.approx(2.0)


def test_ready_constant_marks_success_schema_decision():
    assert READY == "PC_OT_MRAS_TRAINING_ASSIGNMENT_GEOMETRY_REPLAY_READY"
