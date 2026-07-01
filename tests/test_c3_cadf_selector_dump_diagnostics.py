import json
from pathlib import Path

import pytest
from mmengine.config import Config

from tools.dump_c3_cadf_selector_diagnostics import (
    aggregate_summary,
    build_record_from_meta,
    compute_gt_diagnostics,
    enforce_c3_cadf_selector_dump_config,
    write_outputs,
)


ROOT = Path(__file__).resolve().parents[1]
FAST_SAFE_CONFIG = (
    ROOT
    / "configs/adatad/thumos/c3_cadf_densitymesh_original_adatad_32px_loss_select_v2_fast_safe_formal.py"
)


def test_aggregate_summary_keeps_diagnostic_only_flags_and_gap_percentiles():
    records = [
        dict(
            video_name="v1",
            valid_len=768,
            selected_count=384,
            max_gap=4,
            mean_gap=2.0,
            duplicate_count=0,
            repair_count=2,
            repair_fraction=2 / 384,
            density_entropy=0.80,
            boundary_near_rate=0.25,
            gt_remap_kept_count=2,
            gt_remap_length_ratio=0.75,
            action_inside_selected_fraction=0.50,
            action_inside_coverage_fraction=0.60,
            action_inside_max_gap=6,
            action_inside_mean_gap=3.0,
        ),
        dict(
            video_name="v2",
            valid_len=700,
            selected_count=384,
            max_gap=12,
            mean_gap=2.5,
            duplicate_count=1,
            repair_count=6,
            repair_fraction=6 / 384,
            density_entropy=0.60,
            boundary_near_rate=0.50,
            gt_remap_kept_count=1,
            gt_remap_length_ratio=0.40,
            action_inside_selected_fraction=0.25,
            action_inside_coverage_fraction=0.30,
            action_inside_max_gap=14,
            action_inside_mean_gap=7.0,
        ),
    ]

    summary = aggregate_summary(
        records,
        config_path="cfg.py",
        checkpoint_path="epoch_002.pth",
        split="test",
        warnings=["offline GT diagnostics only"],
    )

    assert summary["status"] == "ok"
    assert summary["diagnostic_only"] is True
    assert summary["official_map_claim"] is False
    assert summary["route_labels"] == ["C3_MAINLINE_OPTIMIZATION", "C3_ORIGINAL_OPTIMIZATION_ROUTE"]
    assert summary["record_count"] == 2
    assert summary["max_gap"]["mean"] == 8.0
    assert summary["max_gap"]["p50"] == 8.0
    assert summary["max_gap"]["p90"] == pytest.approx(11.2)
    assert summary["duplicate_count_mean"] == 0.5
    assert summary["repair_count_mean"] == 4.0
    assert summary["density_entropy_mean"] == pytest.approx(0.7)
    assert summary["boundary_near_rate_mean"] == pytest.approx(0.375)
    assert summary["action_inside_selected_fraction_mean"] == pytest.approx(0.375)
    assert summary["action_inside_coverage_fraction_mean"] == pytest.approx(0.45)
    assert summary["action_inside_mean_gap_mean"] == pytest.approx(5.0)
    assert summary["action_inside_max_gap"]["mean"] == pytest.approx(10.0)
    assert summary["action_inside_max_gap"]["p90"] == pytest.approx(13.2)
    assert summary["warnings"] == ["offline GT diagnostics only"]


def test_compute_gt_diagnostics_reports_kept_length_ratio_and_boundary_rate():
    meta = {"window_size": 384, "c3_indirect_valid_len": 12}
    selected = [0, 2, 4, 6, 8, 10]
    gt_segments = [[1.0, 5.0], [9.0, 12.0], [20.0, 24.0]]

    stats = compute_gt_diagnostics(meta, selected, gt_segments, boundary_radius=1)

    assert stats["gt_remap_kept_count"] == 2
    assert stats["gt_remap_total_count"] == 3
    assert stats["gt_remap_kept_fraction"] == pytest.approx(2 / 3)
    assert stats["gt_remap_length_ratio"] == pytest.approx((4.0 + 2.0) / (4.0 + 3.0))
    assert stats["boundary_count"] == 4
    assert stats["boundary_recall_count"] == 3
    assert stats["boundary_recall_rate"] == pytest.approx(3 / 4)
    assert stats["boundary_near_count"] == 6
    assert stats["boundary_near_rate"] == pytest.approx(1.0)
    assert stats["boundary_radius"] == 1


def test_compute_gt_diagnostics_reports_no_action_inside_fields_without_gt():
    meta = {"window_size": 384, "c3_indirect_valid_len": 12}

    stats = compute_gt_diagnostics(meta, selected_dense_indices=[1, 3, 5], gt_segments=None)

    assert stats["action_inside_segment_count"] == 0
    assert stats["action_inside_selected_count"] == 0
    assert stats["action_inside_selected_fraction"] == 0.0
    assert stats["action_inside_coverage_fraction"] == 0.0
    assert stats["action_inside_max_gap"] is None
    assert stats["action_inside_mean_gap"] is None


def test_compute_gt_diagnostics_counts_selected_indices_inside_actions():
    meta = {"window_size": 384, "c3_indirect_valid_len": 12}
    selected = [1, 2, 4, 5, 8, 10]
    gt_segments = [[2.0, 6.0], [8.0, 10.0]]

    stats = compute_gt_diagnostics(meta, selected, gt_segments)

    assert stats["action_inside_segment_count"] == 2
    assert stats["action_inside_selected_count"] == 4
    assert stats["action_inside_selected_fraction"] == pytest.approx(4 / 6)
    assert stats["action_inside_coverage_fraction"] == pytest.approx(4 / 6)
    assert stats["action_inside_max_gap"] == 1
    assert stats["action_inside_mean_gap"] == pytest.approx(1.0)


def test_compute_gt_diagnostics_reports_zero_action_coverage_when_selected_outside_actions():
    meta = {"window_size": 384, "c3_indirect_valid_len": 12}
    selected = [0, 1, 8, 9, 10]
    gt_segments = [[3.0, 7.0]]

    stats = compute_gt_diagnostics(meta, selected, gt_segments)

    assert stats["action_inside_segment_count"] == 1
    assert stats["action_inside_selected_count"] == 0
    assert stats["action_inside_selected_fraction"] == 0.0
    assert stats["action_inside_coverage_fraction"] == 0.0
    assert stats["action_inside_max_gap"] == 4
    assert stats["action_inside_mean_gap"] == pytest.approx(4.0)


def test_compute_gt_diagnostics_reports_large_gap_inside_long_action():
    meta = {"window_size": 384, "c3_indirect_valid_len": 24}
    selected = [2, 3, 18, 20]
    gt_segments = [[2.0, 21.0]]

    stats = compute_gt_diagnostics(meta, selected, gt_segments)

    assert stats["action_inside_segment_count"] == 1
    assert stats["action_inside_selected_count"] == 4
    assert stats["action_inside_selected_fraction"] == 1.0
    assert stats["action_inside_coverage_fraction"] == pytest.approx(4 / 19)
    assert stats["action_inside_max_gap"] == 14
    assert stats["action_inside_mean_gap"] == pytest.approx((14 + 1) / 2)


def test_build_record_from_meta_merges_selector_and_gt_diagnostics():
    meta = {
        "video_name": "video_test_0000001",
        "c3_indirect_valid_len": 12,
        "c3_indirect_selected_dense_indices": [0, 2, 4, 6, 8, 10],
        "c3_density_mesh_max_gap": 2,
        "c3_density_mesh_mean_gap": 2.0,
        "c3_density_mesh_duplicate_count": 0,
        "c3_density_mesh_repair_count": 1,
        "c3_density_mesh_repair_fraction": 1 / 6,
        "c3_density_mesh_dedupe_repair_count": 1,
        "c3_density_mesh_gap_guard_add_count": 0,
        "c3_density_mesh_gap_guard_prune_count": 0,
        "c3_density_mesh_pad_repair_count": 0,
        "c3_density_mesh_entropy": 0.9,
        "c3_density_mesh_selected_density_sum": 0.8,
        "c3_density_mesh_density_top_positions": [2, 4],
        "c3_density_mesh_density_histogram_8": [0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        "c3_density_mesh_uncertainty_selected_fraction": 0.5,
        "c3_density_mesh_change_selected_fraction": 0.25,
        "c3_density_mesh_distribution_target_selected_fraction": 0.7,
    }

    record = build_record_from_meta(meta, gt_segments=[[1.0, 5.0]], include_selected_indices=True)

    assert record["diagnostic_only"] is True
    assert record["official_map_claim"] is False
    assert record["video_name"] == "video_test_0000001"
    assert record["selected_count"] == 6
    assert record["selected_dense_indices"] == [0, 2, 4, 6, 8, 10]
    assert record["dedupe_repair_count"] == 1
    assert record["density_entropy"] == 0.9
    assert record["gt_remap_kept_count"] == 1
    assert record["boundary_near_rate"] > 0.0
    assert record["action_inside_selected_fraction"] == pytest.approx(2 / 6)
    assert record["action_inside_coverage_fraction"] == pytest.approx(2 / 4)


def test_config_gate_accepts_fast_safe_after_forcing_dump_diagnostics():
    cfg = Config.fromfile(FAST_SAFE_CONFIG)

    enforce_c3_cadf_selector_dump_config(cfg)

    assert cfg.model.frame_selector.emit_selection_diagnostics is True
    assert cfg.model.frame_selector.selection_diagnostics_interval == 1
    assert cfg.model.frame_selector.fast_cpu_selection is True
    assert cfg.inference.load_from_raw_predictions is False
    assert cfg.inference.save_raw_prediction is False


def test_config_gate_rejects_forbidden_routes_and_raw_prediction_cache():
    cfg = Config.fromfile(FAST_SAFE_CONFIG)
    cfg.inference.load_from_raw_predictions = True

    with pytest.raises(AssertionError, match="raw prediction"):
        enforce_c3_cadf_selector_dump_config(cfg)

    cfg = Config.fromfile(FAST_SAFE_CONFIG)
    cfg.model.frame_selector.strategy = "coarse_actionness_uncertainty"
    with pytest.raises(AssertionError, match="CADF/loss-select V2"):
        enforce_c3_cadf_selector_dump_config(cfg)


def test_write_outputs_emits_summary_json_and_records_jsonl(tmp_path):
    records = [
        dict(
            video_name="v1",
            valid_len=12,
            selected_count=6,
            max_gap=2,
            mean_gap=2.0,
            duplicate_count=0,
            repair_count=1,
            repair_fraction=1 / 6,
            density_entropy=0.9,
            boundary_near_rate=0.5,
        )
    ]
    summary = aggregate_summary(records, config_path="cfg.py", checkpoint_path="ckpt.pth", split="test")

    summary_path = tmp_path / "summary.json"
    jsonl_path = tmp_path / "records.jsonl"
    write_outputs(summary, records, summary_path=summary_path, jsonl_path=jsonl_path, csv_path=None)

    loaded_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    loaded_record = json.loads(jsonl_path.read_text(encoding="utf-8").strip())

    assert loaded_summary["diagnostic_only"] is True
    assert loaded_summary["official_map_claim"] is False
    assert loaded_record["video_name"] == "v1"
    assert loaded_record["official_map_claim"] is False
