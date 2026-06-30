import json
from types import SimpleNamespace

import pytest

import tools.analyze_c3_pqr_rankcal_proposals as analyzer
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


def test_pqr_proposal_diagnostic_writes_sweep_output_and_keeps_summary_output(tmp_path):
    annotation = tmp_path / "thumos_14_anno.json"
    prediction = tmp_path / "result_detection.json"
    output = tmp_path / "summary.json"
    sweep_output = tmp_path / "sweep.json"
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
        prediction,
        {
            "results": {
                "video_a": [
                    {"segment": [1.0, 3.0], "label": "Diving", "score": 0.95},
                    {"segment": [1.1, 3.1], "label": "Diving", "score": 0.90},
                    {"segment": [7.0, 8.0], "label": "Diving", "score": 0.10},
                ]
            }
        },
    )

    exit_code = main(
        [
            "--prediction",
            str(prediction),
            "--annotation",
            str(annotation),
            "--output",
            str(output),
            "--sweep-output",
            str(sweep_output),
        ]
    )

    summary = json.loads(output.read_text(encoding="utf-8"))
    sweep = json.loads(sweep_output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert summary["status"] == "PASS_DIAGNOSTIC_ANALYSIS"
    assert sweep["status"] == "PASS_DIAGNOSTIC_SWEEP"
    assert sweep["diagnostic_only"] is True
    assert sweep["official_map_claim"] is False
    assert sweep["total_input_predictions"] == 3
    assert {"max_per_video_sweep", "per_class_cap_sweep", "min_score_sweep", "hard_nms_sweep"} <= set(sweep["sweeps"])


def test_pqr_proposal_diagnostic_sweep_caps_and_nms_can_change_retained_counts(tmp_path):
    annotation = tmp_path / "thumos_14_anno.json"
    prediction = tmp_path / "result_detection.json"
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
        prediction,
        {
            "results": {
                "video_a": [
                    {"segment": [1.0, 3.0], "label": "Diving", "score": 0.95},
                    {"segment": [1.1, 3.1], "label": "Diving", "score": 0.90},
                    {"segment": [5.0, 6.0], "label": "Diving", "score": 0.80},
                    {"segment": [9.0, 10.0], "label": "BaseballPitch", "score": 0.70},
                ]
            }
        },
    )

    summary, _records, sweep = analyze(prediction, annotation, subset="validation", include_sweep=True)

    assert summary["total_predictions"] == 4
    per_video_one = next(item for item in sweep["sweeps"]["max_per_video_sweep"] if item["parameters"]["max_per_video"] == 1)
    per_class_one = next(item for item in sweep["sweeps"]["per_class_cap_sweep"] if item["parameters"]["per_class_cap"] == 1)
    nms_05 = next(item for item in sweep["sweeps"]["hard_nms_sweep"] if item["parameters"]["nms_iou_threshold"] == 0.5)
    assert per_video_one["retained_predictions"] == 1
    assert per_video_one["videos_capped_ratio"] == 1.0
    assert per_class_one["retained_predictions"] == 2
    assert nms_05["retained_predictions"] < summary["total_predictions"]
    assert per_video_one["retained_fraction"] == 0.25
    assert "top1_same_label_recall@0.5" in per_video_one["rank_recall"]
    assert "top_score_decile_mean_iou_same_label" in per_video_one


def test_pqr_proposal_diagnostic_reports_explicit_cap_hit_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(analyzer, "DEFAULT_PROPOSAL_CAP", 2, raising=False)
    annotation = tmp_path / "thumos_14_anno.json"
    prediction = tmp_path / "result_detection.json"
    _write_json(
        annotation,
        {
            "database": {
                "video_below": {"subset": "validation", "annotations": []},
                "video_at": {"subset": "validation", "annotations": []},
                "video_over": {"subset": "validation", "annotations": []},
            }
        },
    )

    def proposals(count):
        return [
            {"segment": [float(idx), float(idx + 1)], "label": "Diving", "score": 1.0 / (idx + 1)}
            for idx in range(count)
        ]

    _write_json(
        prediction,
        {
            "results": {
                "video_below": proposals(1),
                "video_at": proposals(2),
                "video_over": proposals(3),
            }
        },
    )

    summary, _records, sweep = analyze(prediction, annotation, subset="validation", include_sweep=True)

    cap = summary["proposal_cap_diagnostic"]
    assert cap["cap_value"] == 2
    assert cap["videos_considered"] == 3
    assert cap["videos_at_cap"] == 1
    assert cap["videos_over_cap"] == 1
    assert cap["videos_at_or_over_cap"] == 2
    assert cap["cap_hit_ratio"] == 2 / 3

    per_video_two = next(item for item in sweep["sweeps"]["max_per_video_sweep"] if item["parameters"]["max_per_video"] == 2)
    retained_cap = per_video_two["proposal_cap_diagnostic"]
    source_cap = per_video_two["source_proposal_cap_diagnostic"]
    assert retained_cap["cap_value"] == 2
    assert retained_cap["videos_considered"] == 3
    assert retained_cap["videos_below_cap"] == 1
    assert retained_cap["videos_at_cap"] == 2
    assert retained_cap["videos_over_cap"] == 0
    assert retained_cap["videos_at_or_over_cap"] == 2
    assert retained_cap["cap_hit_ratio"] == 2 / 3
    assert source_cap["cap_value"] == 2
    assert source_cap["videos_below_cap"] == 1
    assert source_cap["videos_at_cap"] == 1
    assert source_cap["videos_over_cap"] == 1
    assert source_cap["videos_at_or_over_cap"] == 2
    assert source_cap["cap_hit_ratio"] == 2 / 3


def test_pqr_proposal_diagnostic_quality_fusion_reports_missing_fields_without_faking(tmp_path):
    annotation = tmp_path / "thumos_14_anno.json"
    prediction = tmp_path / "result_detection.json"
    _write_json(annotation, {"database": {"video_a": {"subset": "validation", "annotations": []}}})
    _write_json(
        prediction,
        {"results": {"video_a": [{"segment": [0.0, 1.0], "label": "Diving", "score": 0.8}]}},
    )

    _summary, _records, sweep = analyze(prediction, annotation, subset="validation", include_sweep=True)

    quality = sweep["quality_fusion"]
    assert quality["quality_fusion_available"] is False
    assert "quality_score" in quality["missing_quality_fields"]
    assert "cls_score" in quality["missing_quality_fields"]
    assert quality["field_coverage"]["records_total"] == 1
    assert quality["field_coverage"]["records_with_both_quality_and_cls_score"] == 0
    assert quality["alpha_sweep"] == []


def test_pqr_proposal_diagnostic_quality_fusion_alpha_sweep_uses_available_fields(tmp_path):
    annotation = tmp_path / "thumos_14_anno.json"
    prediction = tmp_path / "result_detection.json"
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
        prediction,
        {
            "results": {
                "video_a": [
                    {"segment": [1.0, 3.0], "label": "Diving", "score": 0.30, "quality_score": 1.0, "cls_score": 0.20},
                    {"segment": [8.0, 9.0], "label": "Diving", "score": 0.90, "quality": 0.1, "class_score": 0.30},
                ]
            }
        },
    )

    _summary, records, sweep = analyze(prediction, annotation, subset="validation", include_sweep=True)

    quality = sweep["quality_fusion"]
    alpha_zero = next(item for item in quality["alpha_sweep"] if item["parameters"]["score_alpha"] == 0.0)
    alpha_030 = next(item for item in quality["alpha_sweep"] if item["parameters"]["score_alpha"] == 0.3)
    perfect_record = next(item for item in records if item["start"] == 1.0 and item["end"] == 3.0)
    assert quality["quality_fusion_available"] is True
    assert quality["field_coverage"]["records_with_both_quality_and_cls_score"] == 2
    assert alpha_030["parameters"]["score_formula"] == "cls_score * max(quality_score, 0)^alpha"
    assert perfect_record["quality_score"] == 1.0
    assert perfect_record["cls_score"] == 0.20
    assert alpha_zero["retained_predictions"] == 2
    assert alpha_030["rank_recall"]["top1_same_label_recall@0.5"] == 1.0


def test_pqr_proposal_diagnostic_selected_nms_reports_unavailable_without_selected_coords(tmp_path):
    annotation = tmp_path / "thumos_14_anno.json"
    prediction = tmp_path / "result_detection.json"
    _write_json(annotation, {"database": {"video_a": {"subset": "validation", "annotations": []}}})
    _write_json(
        prediction,
        {"results": {"video_a": [{"segment": [0.0, 1.0], "label": "Diving", "score": 0.8}]}},
    )

    _summary, _records, sweep = analyze(prediction, annotation, subset="validation", include_sweep=True)

    selected_nms = sweep["selected_coordinate_nms"]
    assert selected_nms["selected_coordinate_nms_available"] is False
    assert selected_nms["status"] == "UNAVAILABLE_MISSING_SELECTED_COORDINATES"
    assert selected_nms["field_coverage"]["records_with_selected_coordinates"] == 0


def test_pqr_proposal_diagnostic_records_csv_preserves_optional_quality_and_selected_coords(tmp_path):
    annotation = tmp_path / "thumos_14_anno.json"
    prediction = tmp_path / "result_detection.json"
    records_csv = tmp_path / "records.csv"
    _write_json(annotation, {"database": {"video_a": {"subset": "validation", "annotations": []}}})
    _write_json(
        prediction,
        {
            "results": {
                "video_a": [
                    {
                        "segment": [0.0, 1.0],
                        "selected_segment": [2.0, 4.0],
                        "label": "Diving",
                        "score": 0.8,
                        "quality_score": 0.5,
                        "model_score": 0.7,
                    }
                ]
            }
        },
    )

    exit_code = main(
        [
            "--prediction",
            str(prediction),
            "--annotation",
            str(annotation),
            "--records-csv",
            str(records_csv),
        ]
    )

    header, row = records_csv.read_text(encoding="utf-8").splitlines()
    assert exit_code == 0
    assert "quality_score" in header
    assert "cls_score" in header
    assert "selected_start" in header
    assert "selected_end" in header
    assert "0.5" in row
    assert "0.7" in row
    assert "2.0" in row
    assert "4.0" in row


def test_pqr_proposal_diagnostic_reads_qc_v2_geometry_fields(tmp_path):
    annotation = tmp_path / "thumos_14_anno.json"
    prediction = tmp_path / "result_detection.json"
    _write_json(annotation, {"database": {"video_a": {"subset": "validation", "annotations": []}}})
    _write_json(
        prediction,
        {
            "results": {
                "video_a": [
                    {
                        "segment": [0.0, 1.0],
                        "selected_segment": [0.0, 2.0],
                        "label": "Diving",
                        "score": 0.8,
                        "quality_score": 0.5,
                        "cls_score": 0.7,
                        "fused_score": 0.8,
                        "physical_segment": [0.0, 1.0],
                        "selected_length": 2.0,
                        "physical_length": 1.0,
                        "proposal_width": 1.0,
                        "gap_mean": 2.0,
                        "visibility_support": 0.75,
                        "coverage": 0.5,
                        "endpoint_support": 0.8,
                        "level_id": 2,
                        "point_index": 17,
                        "coverage_available": True,
                    }
                ]
            }
        },
    )

    _summary, records = analyze(prediction, annotation, subset="validation")

    record = records[0]
    assert record["fused_score"] == 0.8
    assert record["physical_start"] == 0.0
    assert record["physical_end"] == 1.0
    assert record["selected_length"] == 2.0
    assert record["physical_length"] == 1.0
    assert record["proposal_width"] == 1.0
    assert record["gap_mean"] == 2.0
    assert record["visibility_support"] == 0.75
    assert record["coverage"] == 0.5
    assert record["endpoint_support"] == 0.8
    assert record["level_id"] == 2
    assert record["point_index"] == 17
    assert record["coverage_available"] is True


def test_pqr_proposal_diagnostic_qc_v2_state_distinguishes_failure_modes(tmp_path):
    annotation = tmp_path / "thumos_14_anno.json"
    prediction = tmp_path / "result_detection.json"
    _write_json(annotation, {"database": {"video_a": {"subset": "validation", "annotations": []}}})
    _write_json(
        prediction,
        {
            "results": {
                "video_a": [
                    {
                        "segment": [0.0, 1.0],
                        "selected_segment": [0.0, 2.0],
                        "label": "Diving",
                        "score": 0.8,
                        "quality_score": 0.5,
                        "cls_score": 0.7,
                        "fused_score": 0.8,
                        "physical_segment": [0.0, 1.0],
                        "selected_length": 2.0,
                        "physical_length": 1.0,
                        "proposal_width": 1.0,
                        "gap_mean": 2.0,
                        "visibility_support": 0.75,
                        "endpoint_support": 0.8,
                        "level_id": 0,
                        "point_index": 3,
                        "coverage_available": True,
                    },
                    {
                        "segment": [3.0, 4.0],
                        "label": "Diving",
                        "score": 0.6,
                    },
                ]
            }
        },
    )

    summary, _records = analyze(prediction, annotation, subset="validation")

    state = summary["qc_v2_diagnostic_state"]
    assert state["status"] == "PARTIAL_QC_V2_DIAGNOSTICS"
    assert state["interpretation"]["classification_calibration_check"] == "AVAILABLE"
    assert state["interpretation"]["ranking_geometry_check"] == "AVAILABLE"
    assert state["interpretation"]["localization_geometry_check"] == "PARTIAL"
    assert state["interpretation"]["proposal_cap_overload_check"] == "AVAILABLE"
    assert state["field_coverage"]["records_with_full_qc_v2_geometry"] == 1


def test_single_stage_post_processing_attaches_qc_v2_diagnostic_fields():
    try:
        import torch
        from opentad.models.detectors.single_stage import SingleStageDetector
    except OSError as exc:
        pytest.skip(f"torch import failed in this Windows environment: {exc}")
    except ModuleNotFoundError as exc:
        if exc.name == "nms_1d_cpu":
            pytest.skip(f"local OpenTAD NMS extension is unavailable: {exc}")
        raise

    predictions = (
        [torch.tensor([[0.0, 2.0]])],
        [torch.tensor([[0.8, 0.1]])],
        [
            dict(
                diagnostic_available=True,
                coverage_available=True,
                cls_scores=torch.tensor([[0.8, 0.1]]),
                fused_scores=torch.tensor([[0.4, 0.05]]),
                quality_scores=torch.tensor([0.5]),
                selected_segments=torch.tensor([[0.0, 2.0]]),
                physical_segments=torch.tensor([[0.0, 4.0]]),
                selected_lengths=torch.tensor([2.0]),
                physical_lengths=torch.tensor([4.0]),
                proposal_widths=torch.tensor([4.0]),
                gap_mean=torch.tensor([2.0]),
                visibility_support=torch.tensor([0.75]),
                coverage=torch.tensor([0.5]),
                endpoint_support=torch.tensor([0.8]),
                level_ids=torch.tensor([1]),
                point_indices=torch.tensor([9]),
            )
        ],
    )
    metas = [
        dict(
            video_name="video_a",
            fps=1.0,
            duration=10.0,
            snippet_stride=1,
            offset_frames=0,
            window_start_frame=0,
            irregular_selected_positions=[0.0, 2.0],
            irregular_selected_valid_len=4.0,
            irregular_native_axis=False,
        )
    ]
    post_cfg = SimpleNamespace(pre_nms_thresh=0.0, pre_nms_topk=5, sliding_window=False, nms=None)

    results = SingleStageDetector.post_processing(
        SingleStageDetector.__new__(SingleStageDetector),
        predictions,
        metas,
        post_cfg,
        ext_cls=["Diving", "BaseballPitch"],
    )

    record = results["video_a"][0]
    assert record["cls_score"] == 0.8
    assert record["fused_score"] == 0.4
    assert record["quality_score"] == 0.5
    assert record["selected_segment"] == [0.0, 2.0]
    assert record["physical_segment"] == [0.0, 4.0]
    assert record["selected_length"] == 2.0
    assert record["physical_length"] == 4.0
    assert record["proposal_width"] == 4.0
    assert record["gap_mean"] == 2.0
    assert record["visibility_support"] == 0.75
    assert record["coverage"] == 0.5
    assert record["endpoint_support"] == 0.8
    assert record["level_id"] == 1
    assert record["point_index"] == 9
    assert record["coverage_available"] is True


def test_test_engine_nms_match_preserves_qc_v2_diagnostic_fields():
    try:
        import torch
        from opentad.cores.test_engine import _match_extra_fields_after_nms
    except OSError as exc:
        pytest.skip(f"torch import failed in this Windows environment: {exc}")
    except ModuleNotFoundError as exc:
        if exc.name == "nms_1d_cpu":
            pytest.skip(f"local OpenTAD NMS extension is unavailable: {exc}")
        raise

    source_records = [
        {
            "segment": [0.0, 2.0],
            "label": "Diving",
            "score": 0.8,
            "cls_score": 0.7,
            "quality_score": 0.5,
            "selected_segment": [0.0, 1.0],
            "coverage_available": True,
        },
        {
            "segment": [4.0, 6.0],
            "label": "BaseballPitch",
            "score": 0.6,
            "cls_score": 0.55,
            "quality_score": 0.4,
            "selected_segment": [2.0, 3.0],
            "coverage_available": False,
        },
    ]
    nms_segments = torch.tensor([[0.0, 2.0], [4.0, 6.0]])
    nms_labels = torch.tensor([0, 1])
    extras = _match_extra_fields_after_nms(
        source_records,
        nms_segments,
        nms_labels,
        ["Diving", "BaseballPitch"],
    )

    assert extras[0]["cls_score"] == 0.7
    assert extras[0]["quality_score"] == 0.5
    assert extras[0]["selected_segment"] == [0.0, 1.0]
    assert extras[0]["coverage_available"] is True
    assert extras[1]["cls_score"] == 0.55
    assert extras[1]["quality_score"] == 0.4
    assert extras[1]["selected_segment"] == [2.0, 3.0]
    assert extras[1]["coverage_available"] is False
