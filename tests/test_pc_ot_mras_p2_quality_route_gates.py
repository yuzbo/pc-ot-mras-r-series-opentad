import json
from pathlib import Path

from tools.bata.analyze_p2_quality_route_gates import (
    SCORE_FACTORS_VISIBLE_NEED_RAW_BUCKETS,
    SUMMARY_PROXY_FAIL,
    SUMMARY_PROXY_PASS_NEEDS_RAW_ROWS,
    run_analysis,
)


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _stat(mean, count=4):
    return {"count": count, "min": mean, "mean": mean, "max": mean}


def _metric_key(topk: int, iou_suffix: str, metric: str) -> str:
    return f"top{topk}_{iou_suffix}_{metric}"


def _make_case(root: Path, name: str, *, coord_issue: int = 0, top100_gap: float = 0.25):
    case = root / name
    loc = case / "localization_attribution"
    _write_json(case / "proposal_factor_summary.json", {"rows_written": 16000})
    aggregate = {}
    for k in (100, 300, 1000):
        score = 0.34 if k == 100 else 0.62 if k == 300 else 0.80
        oracle = score + top100_gap if k == 100 else score + 0.06 if k == 300 else score
        aggregate[_metric_key(k, "iou0p70", "gt_recall")] = _stat(score)
        aggregate[_metric_key(k, "iou0p70", "oracle_iou_rank_gt_recall")] = _stat(oracle)
        aggregate[_metric_key(k, "iou0p70", "class_aware_gt_recall")] = _stat(score - 0.05)
    _write_json(loc / "topk_iou_rank_summary.json", {"aggregate": aggregate})
    bins = []
    for idx in range(4):
        scale = float(idx + 1)
        bins.append(
            {
                "score": _stat(scale * 0.01),
                "start_score": _stat(scale * 0.02),
                "end_score": _stat(scale * 0.03),
                "area_integral": _stat(scale * 0.04),
                "observed_fraction": _stat(0.25 + idx * 0.01),
                "uncertainty_penalty": _stat(1.0 - idx * 0.01),
                "duration": _stat(10.0 + idx),
                "max_gt_iou": _stat(0.10 + idx * 0.05),
                "class_aware_max_gt_iou": _stat(idx * 0.02),
                "iou0p70_positive_fraction": 0.02 + idx * 0.03,
                "iou0p70_class_aware_positive_fraction": idx * 0.01,
            }
        )
    _write_json(loc / "score_iou_reliability.json", {"bins": bins})
    _write_json(
        loc / "dense_axis_coordinate_audit.json",
        {
            "counts": {
                "nonpositive_dense_duration": 0,
                "nonpositive_seconds_duration": 0,
                "outside_window_seconds": coord_issue,
                "missing_window_bounds": 0,
                "seconds_conversion_residual_gt_1e_4": 0,
            }
        },
    )


def test_quality_route_summary_proxy_marks_e0_and_e1_without_full_gate(tmp_path):
    _make_case(tmp_path, "R17_C0")
    _make_case(tmp_path, "R18_C0", top100_gap=0.20)

    summary = run_analysis(
        input_root=tmp_path,
        output_json=tmp_path / "summary.json",
        output_markdown=tmp_path / "summary.md",
    )

    assert summary["status"].endswith("_READY")
    assert summary["e0_oracle_rerank_bound"]["decision"] == SUMMARY_PROXY_PASS_NEEDS_RAW_ROWS
    assert summary["e0_oracle_rerank_bound"]["summary_proxy_pass"] is True
    assert summary["e0_oracle_rerank_bound"]["full_gate_pass"] is False
    assert summary["e1_score_factor_bucket"]["decision"] == SCORE_FACTORS_VISIBLE_NEED_RAW_BUCKETS
    assert summary["e1_score_factor_bucket"]["full_gate_pass"] is False
    assert summary["raw_joined_proposal_rows_available"] is False
    assert summary["metric_claim_allowed"] is False
    assert summary["paper_claim_allowed"] is False
    assert "full_evaluator_facing_oracle_rerank_bound_available: `False`" in (
        tmp_path / "summary.md"
    ).read_text(encoding="utf-8")


def test_quality_route_coordinate_issue_blocks_e0_summary_proxy(tmp_path):
    _make_case(tmp_path, "R17_C0", coord_issue=1)

    summary = run_analysis(input_root=tmp_path, output_json=tmp_path / "summary.json")

    assert summary["e0_oracle_rerank_bound"]["decision"] == SUMMARY_PROXY_FAIL
    assert summary["e0_oracle_rerank_bound"]["summary_proxy_pass"] is False
    assert summary["aggregate"]["coordinate_issue_count"]["mean"] == 1.0
