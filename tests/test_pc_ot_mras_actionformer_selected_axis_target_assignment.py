import json

from tools.bata.analyze_actionformer_selected_axis_target_assignment import (
    READY,
    assign_points,
    gt_segments_to_dense_positions,
    run_selected_axis_target_assignment_audit,
)


def _write_json(path, payload):
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _snapshot_row():
    return {
        "schema_version": "pc_ot_mras_reader_snapshot_dump_v0",
        "snapshot_id": "synthetic",
        "sample_ids": ["video_test_0000001"],
        "reader_out": {
            "selected_times": [[0.0, 2.0 / 7.0, 4.0 / 7.0, 1.0]],
            "selected_mask": [[1, 1, 1, 1]],
            "valid_lengths": [8],
            "valid_mask": [[1, 1, 1, 1, 1, 1, 1, 1]],
        },
        "diagnostic_only": True,
        "uses_gt": False,
        "uses_teacher": False,
        "uses_oracle": False,
        "uses_cache": False,
        "uses_raw_prediction": False,
        "metric_claim_allowed": False,
        "paper_claim_allowed": False,
        "runtime_flops_claim_allowed": False,
        "deploy_claim_allowed": False,
    }


def test_assign_points_uses_center_sampling_and_regression_range():
    gt = [{"segment_dense": [3.5, 4.5]}, {"segment_dense": [1.5, 2.5]}]
    fake = assign_points(
        [0.0, 1.0, 2.0, 3.0],
        gt_segments=gt,
        stride=1,
        regression_range=(0.0, 100.0),
        center_sample_radius=1.5,
    )
    physical = assign_points(
        [0.0, 2.0, 4.0, 7.0],
        gt_segments=gt,
        stride=1,
        regression_range=(0.0, 100.0),
        center_sample_radius=1.5,
    )

    assert fake == [None, None, 1, None]
    assert physical == [None, 1, 0, None]


def test_gt_segments_to_dense_positions_maps_seconds_to_dense_index():
    video = {
        "duration": 7.0,
        "annotations": [
            {"index": 0, "segment_seconds": [3.5, 4.5], "label": "A"},
        ],
    }
    converted = gt_segments_to_dense_positions(video, dense_length=8)
    assert converted[0]["segment_dense"] == [3.5, 4.5]
    assert converted[0]["label"] == "A"


def test_selected_axis_target_assignment_audit_outputs_diagnostic_summary(tmp_path):
    snapshot = tmp_path / "snapshot.jsonl"
    annotation = tmp_path / "anno.json"
    output_dir = tmp_path / "out"
    _write_jsonl(snapshot, [_snapshot_row()])
    _write_json(
        annotation,
        {
            "database": {
                "video_test_0000001": {
                    "duration": 7.0,
                    "frame": 28,
                    "subset": "validation",
                    "annotations": [
                        {"segment": [3.5, 4.5], "label": "A"},
                        {"segment": [1.5, 2.5], "label": "B"},
                    ],
                }
            }
        },
    )

    summary = run_selected_axis_target_assignment_audit(
        snapshot_jsonl=[snapshot],
        annotation=annotation,
        output_dir=output_dir,
        strides=(1,),
        regression_ranges=((0.0, 100.0),),
        center_sample_radius=1.5,
    )

    assert summary["decision"] == READY
    assert summary["sample_count"] == 1
    assert summary["sample_level_rows"] == 1
    assert summary["uses_validation_gt_for_offline_diagnostic_only"] is True
    assert summary["exact_training_assignment_replay"] is False
    assert summary["metric_claim_allowed"] is False
    assert summary["paper_claim_allowed"] is False
    assert summary["fake_positive_stats"]["mean"] == 1.0
    assert summary["physical_positive_stats"]["mean"] == 2.0
    assert summary["positive_delta_physical_minus_fake_stats"]["mean"] == 1.0

    csv_text = (output_dir / "per_sample_level_assignment.csv").read_text(encoding="utf-8")
    assert "physical_only" in csv_text
    assert "gt_coverage_delta_physical_minus_fake" in csv_text
