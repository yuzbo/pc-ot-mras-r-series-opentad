import json

from tools.bata.analyze_p2_raw_row_oracle_rerank import READY, build_closeout


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_raw_row_oracle_rerank_closeout_reports_score_oracle_gap(tmp_path):
    joined = tmp_path / "joined.jsonl"
    base = {
        "sample_id": "sample_a",
        "diagnostic_group_key": "sample_a",
        "video_id": "video_0001",
        "diagnostic_class_aware_max_gt_iou_seconds": 0.0,
    }
    _write_jsonl(
        joined,
        [
            {
                **base,
                "final_score": 0.99,
                "diagnostic_max_gt_iou_seconds": 0.10,
            },
            {
                **base,
                "final_score": 0.20,
                "diagnostic_max_gt_iou_seconds": 0.92,
                "diagnostic_class_aware_max_gt_iou_seconds": 0.92,
            },
            {
                **base,
                "final_score": 0.10,
                "diagnostic_max_gt_iou_seconds": 0.80,
                "diagnostic_class_aware_max_gt_iou_seconds": 0.80,
            },
        ],
    )
    output_json = tmp_path / "summary.json"

    payload = build_closeout(
        [("R17", joined)],
        output_json=output_json,
        topk=(1, 2),
        iou_thresholds=(0.7,),
    )

    case = payload["cases"][0]
    group = case["groups"][0]
    assert payload["decision"] == READY
    assert payload["raw_joined_proposal_rows_available"] is True
    assert payload["full_evaluator_facing_oracle_rerank_bound_available"] is False
    assert group["any_top1_iou0p70_score_positive_count"] == 0
    assert group["any_top1_iou0p70_oracle_positive_count"] == 1
    assert group["any_top1_iou0p70_missed_oracle_positive_count"] == 1
    assert group["any_top1_iou0p70_best_positive_score_rank"] == 2
    assert output_json.is_file()
