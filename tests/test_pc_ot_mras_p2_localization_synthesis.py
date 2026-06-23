import json
from pathlib import Path

from tools.bata.summarize_p2_localization_attribution_results import (
    COORDINATE_RISK,
    SCORE_RANK_WEAK,
    run_synthesis,
)


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _stat(mean, count=4):
    return {"count": count, "min": mean, "mean": mean, "max": mean}


def _make_case(root: Path, name: str, *, coord_issue: int = 0, top100_gap: float = 0.25):
    case = root / name
    loc = case / "localization_attribution"
    _write_json(
        case / "proposal_factor_summary.json",
        {
            "rows_written": 16000,
            "samples_seen": 16,
            "final_score": _stat(0.02, count=16000),
            "observed_fraction": _stat(0.25, count=16000),
            "diagnostic_only": True,
        },
    )
    _write_json(
        loc / "summary.json",
        {
            "joined_rows": 16000,
            "max_gt_iou_seconds": _stat(0.42, count=16000),
            "class_aware_max_gt_iou_seconds": _stat(0.08, count=16000),
            "diagnostic_only": True,
        },
    )
    aggregate = {}
    for k in (100, 300, 1000):
        for suffix, score, oracle in (
            ("iou0p50", 0.62, 0.80),
            ("iou0p70", 0.35, 0.35 + top100_gap),
        ):
            if k == 300:
                score = 0.55 if suffix == "iou0p70" else 0.75
                oracle = score + 0.12
            if k == 1000:
                score = 0.78 if suffix == "iou0p70" else 0.90
                oracle = score
            aggregate[f"top{k}_{suffix}_gt_recall"] = _stat(score)
            aggregate[f"top{k}_{suffix}_oracle_iou_rank_gt_recall"] = _stat(oracle)
            aggregate[f"top{k}_{suffix}_class_aware_gt_recall"] = _stat(max(0.0, score - 0.1))
    _write_json(
        loc / "topk_iou_rank_summary.json",
        {
            "aggregate": aggregate,
            "diagnostic_only": True,
        },
    )
    bins = []
    for idx, (score, iou, pos70) in enumerate(((0.01, 0.20, 0.02), (0.02, 0.24, 0.03), (0.03, 0.27, 0.04))):
        bins.append(
            {
                "bin_index": idx,
                "count": 100,
                "score": _stat(score, count=100),
                "max_gt_iou": _stat(iou, count=100),
                "class_aware_max_gt_iou": _stat(0.01, count=100),
                "iou0p50_positive_fraction": pos70 + 0.1,
                "iou0p50_class_aware_positive_fraction": 0.0,
                "iou0p70_positive_fraction": pos70,
                "iou0p70_class_aware_positive_fraction": 0.0,
            }
        )
    _write_json(loc / "score_iou_reliability.json", {"bins": bins, "diagnostic_only": True})
    _write_json(
        loc / "dense_axis_coordinate_audit.json",
        {
            "counts": {
                "rows": 16000,
                "nonpositive_dense_duration": 0,
                "nonpositive_seconds_duration": 0,
                "outside_window_seconds": coord_issue,
                "missing_window_bounds": 0,
                "seconds_conversion_residual_gt_1e_4": 0,
                "outside_video_seconds": 0,
            },
            "diagnostic_only": True,
        },
    )


def test_p2_localization_synthesis_detects_score_rank_gap(tmp_path):
    _make_case(tmp_path, "R17_C0")
    _make_case(tmp_path, "R18_C0", top100_gap=0.20)

    summary = run_synthesis(
        input_root=tmp_path,
        output_json=tmp_path / "summary.json",
        output_markdown=tmp_path / "summary.md",
    )

    assert summary["status"].endswith("_READY")
    assert summary["decision"] == SCORE_RANK_WEAK
    assert summary["aggregate"]["top100_iou0p70"]["oracle_minus_score_recall"]["mean"] > 0.1
    assert "P2/NIIQ quality-ranking calibration" in summary["recommended_next_steps"][0]
    assert (tmp_path / "summary.md").read_text(encoding="utf-8").startswith("# P2 Localization")


def test_p2_localization_synthesis_coordinate_issue_blocks_score_claim(tmp_path):
    _make_case(tmp_path, "R17_C0", coord_issue=1)

    summary = run_synthesis(
        input_root=tmp_path,
        output_json=tmp_path / "summary.json",
    )

    assert summary["decision"] == COORDINATE_RISK
    assert summary["cases"][0]["coordinate_issue_count"] == 1
