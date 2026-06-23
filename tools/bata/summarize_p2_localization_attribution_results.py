from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "pc_ot_mras_p2_localization_attribution_synthesis_v0"
READY = "P2_LOCALIZATION_ATTRIBUTION_SYNTHESIS_READY"
NO_GO = "P2_LOCALIZATION_ATTRIBUTION_SYNTHESIS_NO_GO"
SCORE_RANK_WEAK = "P2_SCORE_RANK_CALIBRATION_WEAK_GEOMETRIC_PROPOSALS_EXIST"
GEOMETRY_WEAK = "P2_GEOMETRIC_PROPOSAL_RECALL_WEAK"
COORDINATE_RISK = "P2_COORDINATE_AUDIT_HAS_BLOCKING_RISK"
INCONCLUSIVE = "P2_LOCALIZATION_SYNTHESIS_INCONCLUSIVE"


def strict_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json_value(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    return str(value)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(strict_json_value(dict(payload)), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return payload


def finite_values(values: Iterable[Any]) -> list[float]:
    out = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out.append(number)
    return out


def stats(values: Iterable[Any]) -> dict[str, Any]:
    finite = finite_values(values)
    if not finite:
        return {"count": 0, "min": None, "mean": None, "max": None}
    return {
        "count": len(finite),
        "min": min(finite),
        "mean": sum(finite) / float(len(finite)),
        "max": max(finite),
    }


def mean_from_stat(payload: Mapping[str, Any], key: str, default: float | None = None) -> float | None:
    value = payload.get(key)
    if isinstance(value, Mapping):
        mean = value.get("mean")
    elif key == "mean":
        mean = value
    else:
        return default
    try:
        out = float(mean)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def aggregate_mean(topk_summary: Mapping[str, Any], key: str) -> float | None:
    aggregate = topk_summary.get("aggregate")
    if not isinstance(aggregate, Mapping):
        return None
    return mean_from_stat(aggregate, key)


def metric_key(topk: int, iou: float, metric: str) -> str:
    return f"top{int(topk)}_iou{iou:.2f}_{metric}".replace(".", "p")


def spearman_for_ordered_bins(values: Sequence[float]) -> float | None:
    finite = finite_values(values)
    n = len(finite)
    if n < 2:
        return None
    x = list(range(n))
    x_mean = (n - 1) / 2.0
    y_mean = sum(finite) / float(n)
    numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, finite))
    x_var = sum((xi - x_mean) ** 2 for xi in x)
    y_var = sum((yi - y_mean) ** 2 for yi in finite)
    if x_var <= 0.0 or y_var <= 0.0:
        return None
    return numerator / math.sqrt(x_var * y_var)


def summarize_reliability(reliability: Mapping[str, Any], iou_thresholds: Sequence[float]) -> dict[str, Any]:
    bins = reliability.get("bins", [])
    if not isinstance(bins, Sequence):
        bins = []
    score_means = []
    iou_means = []
    class_iou_means = []
    threshold_stats: dict[str, Any] = {}
    for item in bins:
        if not isinstance(item, Mapping):
            continue
        score_means.append(mean_from_stat(item.get("score", {}), "mean"))
        iou_means.append(mean_from_stat(item.get("max_gt_iou", {}), "mean"))
        class_iou_means.append(mean_from_stat(item.get("class_aware_max_gt_iou", {}), "mean"))
    for thr in iou_thresholds:
        suffix = f"iou{float(thr):.2f}".replace(".", "p")
        positives = [
            item.get(f"{suffix}_positive_fraction")
            for item in bins
            if isinstance(item, Mapping)
        ]
        class_positives = [
            item.get(f"{suffix}_class_aware_positive_fraction")
            for item in bins
            if isinstance(item, Mapping)
        ]
        finite_pos = finite_values(positives)
        finite_cls = finite_values(class_positives)
        threshold_stats[suffix] = {
            "positive_fraction_first": finite_pos[0] if finite_pos else None,
            "positive_fraction_last": finite_pos[-1] if finite_pos else None,
            "positive_fraction_last_minus_first": (finite_pos[-1] - finite_pos[0]) if len(finite_pos) >= 2 else None,
            "positive_fraction_spearman_by_score_bin": spearman_for_ordered_bins(finite_pos),
            "class_aware_positive_fraction_first": finite_cls[0] if finite_cls else None,
            "class_aware_positive_fraction_last": finite_cls[-1] if finite_cls else None,
            "class_aware_positive_fraction_last_minus_first": (finite_cls[-1] - finite_cls[0])
            if len(finite_cls) >= 2
            else None,
            "class_aware_positive_fraction_spearman_by_score_bin": spearman_for_ordered_bins(finite_cls),
        }
    finite_iou = finite_values(iou_means)
    finite_class_iou = finite_values(class_iou_means)
    return {
        "bin_count": len([item for item in bins if isinstance(item, Mapping)]),
        "score_mean": stats(score_means),
        "max_gt_iou_mean_first": finite_iou[0] if finite_iou else None,
        "max_gt_iou_mean_last": finite_iou[-1] if finite_iou else None,
        "max_gt_iou_mean_last_minus_first": (finite_iou[-1] - finite_iou[0]) if len(finite_iou) >= 2 else None,
        "max_gt_iou_mean_spearman_by_score_bin": spearman_for_ordered_bins(finite_iou),
        "class_aware_max_gt_iou_mean_first": finite_class_iou[0] if finite_class_iou else None,
        "class_aware_max_gt_iou_mean_last": finite_class_iou[-1] if finite_class_iou else None,
        "class_aware_max_gt_iou_mean_last_minus_first": (finite_class_iou[-1] - finite_class_iou[0])
        if len(finite_class_iou) >= 2
        else None,
        "thresholds": threshold_stats,
    }


def coordinate_issue_count(audit: Mapping[str, Any]) -> int:
    counts = audit.get("counts")
    if not isinstance(counts, Mapping):
        return 10**9
    keys = (
        "nonpositive_dense_duration",
        "nonpositive_seconds_duration",
        "outside_window_seconds",
        "missing_window_bounds",
        "seconds_conversion_residual_gt_1e_4",
    )
    total = 0
    for key in keys:
        try:
            total += int(counts.get(key, 0))
        except (TypeError, ValueError):
            total += 10**6
    return total


def summarize_case(case_dir: Path, topk: Sequence[int], iou_thresholds: Sequence[float]) -> dict[str, Any]:
    loc_dir = case_dir / "localization_attribution"
    summary = read_json(loc_dir / "summary.json")
    topk_summary = read_json(loc_dir / "topk_iou_rank_summary.json")
    reliability = read_json(loc_dir / "score_iou_reliability.json")
    audit = read_json(loc_dir / "dense_axis_coordinate_audit.json")
    proposal_summary = read_json(case_dir / "proposal_factor_summary.json")

    topk_metrics: dict[str, Any] = {}
    for k in topk:
        for thr in iou_thresholds:
            gt_key = metric_key(k, thr, "gt_recall")
            oracle_key = metric_key(k, thr, "oracle_iou_rank_gt_recall")
            class_key = metric_key(k, thr, "class_aware_gt_recall")
            gt_mean = aggregate_mean(topk_summary, gt_key)
            oracle_mean = aggregate_mean(topk_summary, oracle_key)
            class_mean = aggregate_mean(topk_summary, class_key)
            prefix = f"top{int(k)}_iou{float(thr):.2f}".replace(".", "p")
            topk_metrics[prefix] = {
                "gt_recall_mean_by_score_rank": gt_mean,
                "gt_recall_mean_by_oracle_iou_rank": oracle_mean,
                "oracle_minus_score_recall": (oracle_mean - gt_mean)
                if oracle_mean is not None and gt_mean is not None
                else None,
                "class_aware_gt_recall_mean_by_score_rank": class_mean,
            }

    return {
        "case": case_dir.name,
        "proposal_rows": proposal_summary.get("rows_written"),
        "joined_rows": summary.get("joined_rows"),
        "samples_seen": proposal_summary.get("samples_seen"),
        "final_score_mean": mean_from_stat(proposal_summary.get("final_score", {}), "mean"),
        "observed_fraction_mean": mean_from_stat(proposal_summary.get("observed_fraction", {}), "mean"),
        "max_gt_iou_mean": mean_from_stat(summary.get("max_gt_iou_seconds", {}), "mean"),
        "class_aware_max_gt_iou_mean": mean_from_stat(summary.get("class_aware_max_gt_iou_seconds", {}), "mean"),
        "coordinate_issue_count": coordinate_issue_count(audit),
        "outside_video_seconds": audit.get("counts", {}).get("outside_video_seconds")
        if isinstance(audit.get("counts"), Mapping)
        else None,
        "topk": topk_metrics,
        "score_iou_reliability": summarize_reliability(reliability, iou_thresholds),
        "diagnostic_only": True,
        "uses_validation_gt": True,
        "metric_claim_allowed": False,
        "paper_claim_allowed": False,
    }


def collect_case_dirs(root: str | Path) -> list[Path]:
    root_path = Path(root).expanduser()
    if not root_path.exists():
        raise FileNotFoundError(f"input root not found: {root_path}")
    cases = []
    for child in sorted(root_path.iterdir()):
        if not child.is_dir():
            continue
        if (child / "localization_attribution" / "summary.json").exists():
            cases.append(child)
    if not cases:
        raise ValueError(f"no case directories with localization_attribution/summary.json under {root_path}")
    return cases


def choose_decision(case_summaries: Sequence[Mapping[str, Any]]) -> tuple[str, list[str]]:
    coordinate_issues = [int(case.get("coordinate_issue_count", 0)) for case in case_summaries]
    top1000_07 = [
        case.get("topk", {}).get("top1000_iou0p70", {}).get("gt_recall_mean_by_score_rank")
        for case in case_summaries
    ]
    top300_07_gaps = [
        case.get("topk", {}).get("top300_iou0p70", {}).get("oracle_minus_score_recall")
        for case in case_summaries
    ]
    top100_07_gaps = [
        case.get("topk", {}).get("top100_iou0p70", {}).get("oracle_minus_score_recall")
        for case in case_summaries
    ]
    rel_07_gaps = [
        case.get("score_iou_reliability", {})
        .get("thresholds", {})
        .get("iou0p70", {})
        .get("positive_fraction_last_minus_first")
        for case in case_summaries
    ]
    mean_top1000_07 = stats(top1000_07)["mean"]
    mean_top300_gap = stats(top300_07_gaps)["mean"]
    mean_top100_gap = stats(top100_07_gaps)["mean"]
    mean_rel_gap = stats(rel_07_gaps)["mean"]

    reasons = []
    if any(item > 0 for item in coordinate_issues):
        reasons.append("coordinate audit has non-edge blocking issues")
        return COORDINATE_RISK, reasons
    if mean_top1000_07 is not None and mean_top1000_07 < 0.5:
        reasons.append(f"top1000 IoU0.70 GT recall is low: {mean_top1000_07:.4f}")
        return GEOMETRY_WEAK, reasons
    if (mean_top300_gap is not None and mean_top300_gap >= 0.08) or (
        mean_top100_gap is not None and mean_top100_gap >= 0.10
    ):
        reasons.append(f"top300 IoU0.70 oracle-minus-score recall gap: {mean_top300_gap}")
        reasons.append(f"top100 IoU0.70 oracle-minus-score recall gap: {mean_top100_gap}")
        reasons.append(f"score-bin IoU0.70 positive last-minus-first: {mean_rel_gap}")
        return SCORE_RANK_WEAK, reasons
    reasons.append("geometric recall and score-rank gaps do not cross conservative thresholds")
    return INCONCLUSIVE, reasons


def aggregate_cases(case_summaries: Sequence[Mapping[str, Any]], topk: Sequence[int], iou_thresholds: Sequence[float]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {
        "case_count": len(case_summaries),
        "proposal_rows": stats(case.get("proposal_rows") for case in case_summaries),
        "joined_rows": stats(case.get("joined_rows") for case in case_summaries),
        "final_score_mean": stats(case.get("final_score_mean") for case in case_summaries),
        "observed_fraction_mean": stats(case.get("observed_fraction_mean") for case in case_summaries),
        "max_gt_iou_mean": stats(case.get("max_gt_iou_mean") for case in case_summaries),
        "class_aware_max_gt_iou_mean": stats(case.get("class_aware_max_gt_iou_mean") for case in case_summaries),
        "coordinate_issue_count": stats(case.get("coordinate_issue_count") for case in case_summaries),
    }
    for k in topk:
        for thr in iou_thresholds:
            prefix = f"top{int(k)}_iou{float(thr):.2f}".replace(".", "p")
            aggregate[prefix] = {
                "gt_recall_by_score_rank": stats(
                    case.get("topk", {}).get(prefix, {}).get("gt_recall_mean_by_score_rank")
                    for case in case_summaries
                ),
                "gt_recall_by_oracle_iou_rank": stats(
                    case.get("topk", {}).get(prefix, {}).get("gt_recall_mean_by_oracle_iou_rank")
                    for case in case_summaries
                ),
                "oracle_minus_score_recall": stats(
                    case.get("topk", {}).get(prefix, {}).get("oracle_minus_score_recall")
                    for case in case_summaries
                ),
                "class_aware_gt_recall_by_score_rank": stats(
                    case.get("topk", {}).get(prefix, {}).get("class_aware_gt_recall_mean_by_score_rank")
                    for case in case_summaries
                ),
            }
    aggregate["score_iou0p70_positive_fraction_last_minus_first"] = stats(
        case.get("score_iou_reliability", {})
        .get("thresholds", {})
        .get("iou0p70", {})
        .get("positive_fraction_last_minus_first")
        for case in case_summaries
    )
    aggregate["score_bin_max_gt_iou_mean_last_minus_first"] = stats(
        case.get("score_iou_reliability", {}).get("max_gt_iou_mean_last_minus_first")
        for case in case_summaries
    )
    return aggregate


def recommended_next_steps(decision: str) -> list[str]:
    if decision == SCORE_RANK_WEAK:
        return [
            "Prioritize P2/NIIQ quality-ranking calibration over another C2/C3 metadata rerun.",
            "Design a train-only quality target that predicts class-aware IoU or boundary quality from P2 factors and physical-time geometry.",
            "Add an offline oracle-rerank upper-bound diagnostic before any long trainable scorer run.",
            "Keep validation/test GT out of deploy scoring; validation GT is diagnostic-only in this synthesis.",
        ]
    if decision == GEOMETRY_WEAK:
        return [
            "Audit proposal decode/assignment geometry before score calibration.",
            "Do not launch scorer-only recovery until high-IoU geometric proposal recall is restored.",
        ]
    if decision == COORDINATE_RISK:
        return [
            "Fix coordinate conversion or window-bound metadata before model-side score changes.",
            "Rerun localization attribution after coordinate issues are resolved.",
        ]
    return [
        "Treat the current subset evidence as inconclusive.",
        "Run a larger diagnostic sweep or inspect additional score components before training changes.",
    ]


def write_markdown(path: str | Path, summary: Mapping[str, Any]) -> None:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# P2 Localization Attribution Synthesis",
        "",
        f"Decision: `{summary['decision']}`",
        "",
        "| case | rows | top100@0.7 | oracle | gap | top300@0.7 gap | top1000@0.7 | score-bin @0.7 delta | coord issues |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in summary["cases"]:
        top100 = case["topk"].get("top100_iou0p70", {})
        top300 = case["topk"].get("top300_iou0p70", {})
        top1000 = case["topk"].get("top1000_iou0p70", {})
        rel = case["score_iou_reliability"]["thresholds"].get("iou0p70", {})
        lines.append(
            "| {case} | {rows} | {score} | {oracle} | {gap} | {gap300} | {score1000} | {relgap} | {coord} |".format(
                case=case["case"],
                rows=case.get("joined_rows"),
                score=format_float(top100.get("gt_recall_mean_by_score_rank")),
                oracle=format_float(top100.get("gt_recall_mean_by_oracle_iou_rank")),
                gap=format_float(top100.get("oracle_minus_score_recall")),
                gap300=format_float(top300.get("oracle_minus_score_recall")),
                score1000=format_float(top1000.get("gt_recall_mean_by_score_rank")),
                relgap=format_float(rel.get("positive_fraction_last_minus_first")),
                coord=case.get("coordinate_issue_count"),
            )
        )
    lines.extend(
        [
            "",
            "## Decision Reasons",
            "",
            *[f"- {item}" for item in summary["decision_reasons"]],
            "",
            "## Recommended Next Steps",
            "",
            *[f"- {item}" for item in summary["recommended_next_steps"]],
            "",
        ]
    )
    out.write_text("\n".join(lines), encoding="utf-8")


def format_float(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return f"{number:.4f}"


def run_synthesis(
    *,
    input_root: str | Path,
    output_json: str | Path,
    output_markdown: str | Path | None = None,
    topk: Sequence[int] = (100, 300, 1000),
    iou_thresholds: Sequence[float] = (0.5, 0.7),
) -> dict[str, Any]:
    case_dirs = collect_case_dirs(input_root)
    cases = [summarize_case(case_dir, topk=topk, iou_thresholds=iou_thresholds) for case_dir in case_dirs]
    decision, reasons = choose_decision(cases)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "status": READY,
        "input_root": str(Path(input_root).expanduser()),
        "case_count": len(cases),
        "topk": [int(item) for item in topk],
        "iou_thresholds": [float(item) for item in iou_thresholds],
        "cases": cases,
        "aggregate": aggregate_cases(cases, topk=topk, iou_thresholds=iou_thresholds),
        "decision_reasons": reasons,
        "recommended_next_steps": recommended_next_steps(decision),
        "diagnostic_only": True,
        "uses_validation_gt": True,
        "uses_teacher": False,
        "uses_oracle": False,
        "uses_raw_prediction": False,
        "metric_claim_allowed": False,
        "paper_claim_allowed": False,
    }
    write_json(output_json, summary)
    if output_markdown is not None:
        write_markdown(output_markdown, summary)
    return summary


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def error_payload(exc: BaseException) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": NO_GO,
        "error_type": exc.__class__.__name__,
        "error": str(exc),
        "diagnostic_only": True,
        "metric_claim_allowed": False,
        "paper_claim_allowed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize P2 localization attribution diagnostics across cases.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown")
    parser.add_argument("--topk", default="100,300,1000")
    parser.add_argument("--iou-thresholds", default="0.5,0.7")
    args = parser.parse_args(argv)
    try:
        summary = run_synthesis(
            input_root=args.input_root,
            output_json=args.output_json,
            output_markdown=args.output_markdown,
            topk=parse_int_list(args.topk),
            iou_thresholds=parse_float_list(args.iou_thresholds),
        )
    except Exception as exc:  # pragma: no cover - CLI guard
        print(json.dumps(strict_json_value(error_payload(exc)), sort_keys=True))
        return 1
    print(json.dumps(strict_json_value(summary), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
