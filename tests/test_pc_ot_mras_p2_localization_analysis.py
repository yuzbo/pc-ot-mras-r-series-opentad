import json
from pathlib import Path

from tools.bata.analyze_p2_proposal_localization import (
    compute_dense_axis_coordinate_audit,
    run_localization_attribution,
    temporal_iou,
)


def _write_json(path: Path, payload):
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_temporal_iou():
    assert temporal_iou([1.0, 3.0], [2.0, 4.0]) == 1.0 / 3.0
    assert temporal_iou([1.0, 2.0], [3.0, 4.0]) == 0.0


def test_p2_localization_attribution_outputs_join_topk_reliability_and_audit(tmp_path):
    annotation = tmp_path / "anno.json"
    class_map = tmp_path / "category_idx.txt"
    proposals = tmp_path / "proposals.jsonl"
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
    base = {
        "video_id": "video_0001",
        "sample_id": "video_0001|window_start_frame=0",
        "fps": 10.0,
        "snippet_stride": 2.0,
        "window_start_frame": 0.0,
        "offset_frames": 0.0,
        "window_size": 128.0,
        "window_start_seconds": 0.0,
        "window_end_seconds": 25.6,
        "duration_seconds": 20.0,
        "start_score": 0.5,
        "end_score": 0.5,
        "area_integral": 0.5,
        "observed_fraction": 0.25,
        "uncertainty_penalty": 0.9,
        "level": 0,
    }
    _write_jsonl(
        proposals,
        [
            {
                **base,
                "class_id": 1,
                "label": "B",
                "segment": [40.0, 50.0],
                "segment_seconds": [8.0, 10.0],
                "segment_frames": [80.0, 100.0],
                "final_score": 0.90,
                "duration": 10.0,
            },
            {
                **base,
                "class_id": 0,
                "label": "A",
                "segment": [5.5, 14.5],
                "segment_seconds": [1.1, 2.9],
                "segment_frames": [11.0, 29.0],
                "final_score": 0.20,
                "duration": 9.0,
            },
            {
                **base,
                "class_id": 0,
                "label": "A",
                "segment": [60.0, 65.0],
                "segment_seconds": [12.0, 13.0],
                "segment_frames": [120.0, 130.0],
                "final_score": 0.10,
                "duration": 5.0,
            },
        ],
    )

    summary = run_localization_attribution(
        proposal_jsonl=proposals,
        output_dir=output_dir,
        annotation=annotation,
        class_map=class_map,
        topk=(1, 2),
        iou_thresholds=(0.5,),
        score_bins=2,
        group_by="sample",
    )

    assert summary["decision"].endswith("_READY")
    assert summary["joined_rows"] == 3
    assert summary["max_gt_iou_seconds"]["max"] == 1.0

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

    reliability = json.loads((output_dir / "score_iou_reliability.json").read_text(encoding="utf-8"))
    assert reliability["score_bins"] == 2
    assert reliability["bins"]

    audit = json.loads((output_dir / "dense_axis_coordinate_audit.json").read_text(encoding="utf-8"))
    assert audit["counts"]["rows"] == 3
    assert audit["counts"]["outside_window_seconds"] == 0


def test_dense_axis_coordinate_audit_flags_outside_window():
    audit = compute_dense_axis_coordinate_audit(
        [
            {
                "segment": [10.0, 20.0],
                "segment_seconds": [2.0, 4.0],
                "fps": 10.0,
                "snippet_stride": 2.0,
                "window_start_frame": 0.0,
                "offset_frames": 0.0,
                "window_start_seconds": 0.0,
                "window_end_seconds": 3.0,
                "duration_seconds": 20.0,
            }
        ]
    )
    assert audit["counts"]["outside_window_seconds"] == 1
