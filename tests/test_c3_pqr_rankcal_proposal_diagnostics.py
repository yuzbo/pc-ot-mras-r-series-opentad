import json

from tools.analyze_c3_pqr_rankcal_proposals import analyze, main


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_pqr_proposal_diagnostic_reports_score_iou_rank_and_overload(tmp_path):
    annotation = tmp_path / "thumos_14_anno.json"
    prediction = tmp_path / "result_detection.json"
    _write_json(
        annotation,
        {
            "database": {
                "video_a": {
                    "subset": "validation",
                    "annotations": [
                        {"segment": [1.0, 3.0], "label": "BaseballPitch"},
                        {"segment": [6.0, 8.0], "label": "Diving"},
                    ],
                },
                "video_b": {
                    "subset": "validation",
                    "annotations": [{"segment": [2.0, 4.0], "label": "Diving"}],
                },
                "video_train": {
                    "subset": "training",
                    "annotations": [{"segment": [0.0, 1.0], "label": "Diving"}],
                },
            }
        },
    )
    _write_json(
        prediction,
        {
            "results": {
                "video_a": [
                    {"segment": [1.0, 3.0], "label": "BaseballPitch", "score": 0.95},
                    {"segment": [0.0, 1.0], "label": "Diving", "score": 0.90},
                    {"segment": [6.1, 8.0], "label": "Diving", "score": 0.80},
                ],
                "video_b": [
                    {"segment": [2.2, 4.1], "label": "Diving", "score": 0.70},
                    {"segment": [8.0, 9.0], "label": "BaseballPitch", "score": 0.10},
                ],
            }
        },
    )

    summary, records = analyze(prediction, annotation, subset="validation", topk_values=(1, 2), thresholds=(0.5,))

    assert summary["status"] == "PASS_DIAGNOSTIC_ANALYSIS"
    assert summary["diagnostic_only"] is True
    assert summary["official_map_claim"] is False
    assert summary["total_predictions"] == 5
    assert summary["total_ground_truth_instances"] == 3
    assert summary["proposal_count_per_video"]["max"] == 3
    assert summary["proposal_count_per_video_label"]["max"] == 2
    assert summary["score_iou_same_label_spearman"] is not None
    assert summary["rank_recall"]["top1_same_label_recall@0.5"] == 2 / 3
    assert summary["rank_recall"]["top2_same_label_recall@0.5"] == 2 / 3
    assert len(summary["score_rank_bins"]) > 0
    assert len(records) == 5


def test_pqr_proposal_diagnostic_fails_closed_when_artifact_is_missing(tmp_path, capsys):
    annotation = tmp_path / "thumos_14_anno.json"
    output = tmp_path / "summary.json"
    _write_json(annotation, {"database": {}})

    exit_code = main(
        [
            "--prediction",
            str(tmp_path / "missing_result_detection.json"),
            "--annotation",
            str(annotation),
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert "MISSING_ARTIFACT" in captured.out
    assert summary["status"] == "MISSING_ARTIFACT"
    assert summary["required_next_dump"] == "enable post_processing.save_dict=True for the next diagnostic validation"


def test_pqr_proposal_diagnostic_discovers_nested_gpu_prediction(tmp_path, capsys):
    annotation = tmp_path / "thumos_14_anno.json"
    run_dir = tmp_path / "run"
    nested_prediction = run_dir / "gpu1_id0" / "result_detection.json"
    output = tmp_path / "summary.json"
    nested_prediction.parent.mkdir(parents=True)
    _write_json(
        annotation,
        {
            "database": {
                "video_a": {
                    "subset": "validation",
                    "annotations": [{"segment": [1.0, 3.0], "label": "Diving"}],
                }
            }
        },
    )
    _write_json(
        nested_prediction,
        {"results": {"video_a": [{"segment": [1.0, 3.0], "label": "Diving", "score": 0.9}]}},
    )

    exit_code = main(
        [
            "--prediction",
            str(run_dir / "result_detection.json"),
            "--annotation",
            str(annotation),
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert "PASS_DIAGNOSTIC_ANALYSIS" in captured.out
    assert summary["status"] == "PASS_DIAGNOSTIC_ANALYSIS"
    assert summary["prediction_path"] == str(nested_prediction)
    assert summary["prediction_discovery"]["status"] == "DISCOVERED_NESTED_GPU_ARTIFACT"
    assert summary["prediction_discovery"]["requested_path"] == str(run_dir / "result_detection.json")


def test_pqr_proposal_diagnostic_accepts_explicit_prediction_json_alias(tmp_path):
    annotation = tmp_path / "thumos_14_anno.json"
    nested_prediction = tmp_path / "run" / "gpu1_id0" / "result_detection.json"
    output = tmp_path / "summary.json"
    nested_prediction.parent.mkdir(parents=True)
    _write_json(
        annotation,
        {
            "database": {
                "video_a": {
                    "subset": "validation",
                    "annotations": [{"segment": [1.0, 3.0], "label": "Diving"}],
                }
            }
        },
    )
    _write_json(
        nested_prediction,
        {"results": {"video_a": [{"segment": [1.0, 3.0], "label": "Diving", "score": 0.9}]}},
    )

    exit_code = main(
        [
            "--prediction-json",
            str(nested_prediction),
            "--annotation",
            str(annotation),
            "--output",
            str(output),
        ]
    )

    summary = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert summary["status"] == "PASS_DIAGNOSTIC_ANALYSIS"
    assert summary["prediction_path"] == str(nested_prediction)


def test_pqr_proposal_diagnostic_fails_closed_for_multiple_nested_predictions(tmp_path):
    annotation = tmp_path / "thumos_14_anno.json"
    run_dir = tmp_path / "run"
    output = tmp_path / "summary.json"
    _write_json(annotation, {"database": {}})
    for gpu_name in ("gpu0_id0", "gpu1_id0"):
        prediction = run_dir / gpu_name / "result_detection.json"
        prediction.parent.mkdir(parents=True)
        _write_json(prediction, {"results": {}})

    exit_code = main(
        [
            "--prediction",
            str(run_dir / "result_detection.json"),
            "--annotation",
            str(annotation),
            "--output",
            str(output),
        ]
    )

    summary = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert summary["status"] == "AMBIGUOUS_PREDICTION_ARTIFACT"
    assert len(summary["candidate_prediction_paths"]) == 2
