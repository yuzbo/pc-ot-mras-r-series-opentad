import json

from tools.bata.analyze_actionformer_post_nms_overload import (
    READY,
    parse_log_metrics,
    parse_max_seg_num_from_config,
    run_post_nms_overload_audit,
)


def _write_json(path, payload):
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def test_parse_log_metrics_groups_eval_blocks(tmp_path):
    log_path = tmp_path / "train.log"
    log_path.write_text(
        "\n".join(
            [
                "2026-06-23 00:00:00 Train INFO: Number of ground truth instances: 3325",
                "2026-06-23 00:00:00 Train INFO: Number of predictions: 422000",
                "2026-06-23 00:00:00 Train INFO: Average-mAP: 3.68 (%)",
                "2026-06-23 00:30:00 Train INFO: Number of ground truth instances: 3325",
                "2026-06-23 00:30:00 Train INFO: Number of predictions: 422000",
                "2026-06-23 00:30:00 Train INFO: Average-mAP: 4.37 (%)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = parse_log_metrics(log_path)

    assert len(rows) == 2
    assert rows[0]["predictions"] == 422000
    assert rows[1]["average_map_percent"] == 4.37


def test_post_nms_overload_audit_detects_cap_saturation(tmp_path):
    log_path = tmp_path / "train.log"
    config_path = tmp_path / "cfg.py"
    annotation = tmp_path / "anno.json"
    output_dir = tmp_path / "out"
    log_path.write_text(
        "\n".join(
            [
                "2026-06-23 00:00:00 Train INFO: Number of ground truth instances: 3",
                "2026-06-23 00:00:00 Train INFO: Number of predictions: 4000",
                "2026-06-23 00:00:00 Train INFO: Average-mAP: 1.25 (%)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config_path.write_text(
        "post_processing = dict(nms=dict(max_seg_num=2000), save_dict=False)\n",
        encoding="utf-8",
    )
    _write_json(
        annotation,
        {
            "database": {
                "video_a": {
                    "subset": "validation",
                    "duration": 10.0,
                    "annotations": [{"segment": [0.0, 1.0], "label": "A"}],
                },
                "video_b": {
                    "subset": "validation",
                    "duration": 20.0,
                    "annotations": [{"segment": [1.0, 2.0], "label": "A"}],
                },
            }
        },
    )

    payload = run_post_nms_overload_audit(
        train_log=log_path,
        annotation=annotation,
        output_dir=output_dir,
        config=config_path,
        references={"baseline": 2000},
    )

    assert payload["decision"] == READY
    summary = payload["summary"]
    assert summary["expected_dataset_prediction_cap"] == 4000
    assert summary["all_logged_prediction_counts_equal_cap"] is True
    assert summary["latest_predictions_per_video"] == 2000
    assert summary["reference_prediction_count_ratios"]["baseline"] == 2.0
    assert payload["protocol_flags"]["no_model_forward"] is True
    assert payload["protocol_flags"]["metric_claim_allowed"] is False


def test_post_nms_overload_audit_can_count_result_detection_file(tmp_path):
    log_path = tmp_path / "train.log"
    config_path = tmp_path / "cfg.py"
    annotation = tmp_path / "anno.json"
    result_detection = tmp_path / "result_detection.json"
    output_dir = tmp_path / "out"
    log_path.write_text(
        "2026-06-23 00:00:00 Train INFO: Number of predictions: 3\n",
        encoding="utf-8",
    )
    config_path.write_text("post_processing = dict(nms=dict(max_seg_num=2))\n", encoding="utf-8")
    _write_json(
        annotation,
        {
            "database": {
                "video_a": {"subset": "validation", "duration": 1.0, "annotations": []},
                "video_b": {"subset": "validation", "duration": 1.0, "annotations": []},
            }
        },
    )
    _write_json(
        result_detection,
        {
            "results": {
                "video_a": [{"segment": [0.0, 0.5], "label": "A", "score": 0.1}],
                "video_b": [
                    {"segment": [0.0, 0.5], "label": "A", "score": 0.2},
                    {"segment": [0.5, 1.0], "label": "B", "score": 0.3},
                ],
            }
        },
    )

    payload = run_post_nms_overload_audit(
        train_log=log_path,
        annotation=annotation,
        output_dir=output_dir,
        config=config_path,
        result_detection_json=result_detection,
    )

    counts = payload["summary"]["result_detection_counts"]
    assert counts["total_predictions"] == 3
    assert counts["per_video_count_histogram"] == {"1": 1, "2": 1}
    assert payload["summary"]["result_detection_total_matches_latest_log"] is True


def test_parse_max_seg_num_follows_base_config(tmp_path):
    base = tmp_path / "base.py"
    child = tmp_path / "child.py"
    base.write_text("post_processing = dict(nms=dict(max_seg_num=2000))\n", encoding="utf-8")
    child.write_text("_base_ = ['base.py']\n", encoding="utf-8")

    assert parse_max_seg_num_from_config(child) == 2000
