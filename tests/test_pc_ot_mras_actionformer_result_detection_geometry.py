import json

import pytest

from tools.bata.analyze_actionformer_result_detection_geometry import (
    READY,
    result_detection_to_proposal_rows,
    run_result_detection_geometry_audit,
)


def _write_json(path, payload):
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def test_actionformer_result_detection_geometry_audit_outputs_join_and_summary(tmp_path):
    annotation = tmp_path / "anno.json"
    class_map = tmp_path / "category_idx.txt"
    result_detection = tmp_path / "result_detection.json"
    output_dir = tmp_path / "out"
    _write_json(
        annotation,
        {
            "database": {
                "video_0001": {
                    "subset": "validation",
                    "duration": 20.0,
                    "annotations": [
                        {"segment": [1.0, 3.0], "label": "A"},
                        {"segment": [8.0, 10.0], "label": "B"},
                    ],
                }
            }
        },
    )
    class_map.write_text("0 A\n1 B\n", encoding="utf-8")
    _write_json(
        result_detection,
        {
            "results": {
                "video_0001": [
                    {"segment": [8.0, 10.0], "label": 1, "score": 0.95},
                    {"segment": [1.1, 2.9], "label": "A", "score": 0.30},
                    {"segment": [12.0, 13.0], "label": "A", "score": 0.10},
                ]
            }
        },
    )

    summary = run_result_detection_geometry_audit(
        result_detection_json=result_detection,
        output_dir=output_dir,
        annotation=annotation,
        class_map=class_map,
        topk=(1, 2),
        iou_thresholds=(0.5,),
        score_bins=2,
        group_by="video",
    )

    assert summary["decision"] == READY
    assert summary["converted_rows"] == 3
    assert summary["no_model_forward"] is True
    assert summary["uses_validation_gt_for_offline_diagnostic_join_only"] is True
    assert summary["metric_claim_allowed"] is False
    assert summary["paper_claim_allowed"] is False

    converted = [
        json.loads(line)
        for line in (output_dir / "converted_result_detection_proposals.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert converted[0]["label"] == "B"
    assert converted[0]["class_id"] == 1
    assert converted[0]["duration_seconds"] == 20.0
    assert "segment" not in converted[0]

    joined = [
        json.loads(line)
        for line in (output_dir / "joined_proposals.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert joined[0]["diagnostic_score_rank_in_group"] == 1
    assert joined[1]["diagnostic_class_aware_best_gt_label"] == "A"
    assert joined[1]["diagnostic_class_aware_max_gt_iou_seconds"] > 0.8

    topk = json.loads((output_dir / "topk_iou_rank_summary.json").read_text(encoding="utf-8"))
    group = topk["groups"][0]
    assert group["top1_iou0p50_gt_recall"] == 0.5
    assert group["top2_iou0p50_gt_recall"] == 1.0

    audit = json.loads((output_dir / "dense_axis_coordinate_audit.json").read_text(encoding="utf-8"))
    assert audit["counts"]["rows"] == 3
    assert audit["counts"]["has_seconds"] == 3
    assert audit["counts"]["has_frames"] == 0
    assert audit["counts"]["outside_video_seconds"] == 0


def test_result_detection_to_proposal_rows_rejects_bad_segments():
    with pytest.raises(ValueError, match="positive duration"):
        result_detection_to_proposal_rows(
            {
                "video_0001": [
                    {"segment": [3.0, 3.0], "label": "A", "score": 0.5},
                ]
            }
        )
