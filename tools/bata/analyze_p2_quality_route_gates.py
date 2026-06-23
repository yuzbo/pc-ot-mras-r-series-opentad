from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "pc_ot_mras_p2_quality_route_gates_v0"
READY = "P2_QUALITY_ROUTE_GATES_READY"
NO_GO = "P2_QUALITY_ROUTE_GATES_NO_GO"
SUMMARY_PROXY_PASS_NEEDS_RAW_ROWS = "E0_SUMMARY_PROXY_PASS_FULL_EVALUATOR_BOUND_STILL_REQUIRED"
SUMMARY_PROXY_FAIL = "E0_SUMMARY_PROXY_FAIL_DO_NOT_START_SCORER_ROUTE"
SCORE_FACTORS_VISIBLE_NEED_RAW_BUCKETS = "E1_SCORE_FACTORS_VISIBLE_BUT_MISSED_GOOD_BUCKET_REQUIRES_RAW_ROWS"
SCORE_FACTOR_EVIDENCE_WEAK = "E1_SCORE_FACTOR_SUMMARY_EVIDENCE_WEAK"


FACTOR_NAMES = (
    "score",
    "start_score",
    "end_score",
    "area_integral",
    "observed_fraction",
    "uncertainty_penalty",
    "duration",
    "max_gt_iou",
    "class_aware_max_gt_iou",
)


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


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return payload


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(strict_json_value(dict(payload)), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def mean_from_stat(payload: Any) -> float | None:
    if isinstance(payload, Mapping):
        value = payload.get("mean")
    else:
        value = payload
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def metric_key(topk: int, iou: float, metric: str) -> str:
    return f"top{int(topk)}_iou{iou:.2f}_{metric}".replace(".", "p")


def aggregate_mean(topk_summary: Mapping[str, Any], key: str) -> float | None:
    aggregate = topk_summary.get("aggregate")
    if not isinstance(aggregate, Mapping):
        return None
    return mean_from_stat(aggregate.get(key))


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


def factor_series_from_bins(bins: Sequence[Any], factor: str) -> list[float]:
    values = []
    for item in bins:
        if not isinstance(item, Mapping):
            continue
        values.append(mean_from_stat(item.get(factor)))
    return finite_values(values)


def factor_summary_from_bins(bins: Sequence[Any], factor: str) -> dict[str, Any]:
    series = factor_series_from_bins(bins, factor)
    first = series[0] if series else None
    last = series[-1] if series else None
    delta = (last - first) if first is not None and last is not None and len(series) >= 2 else None
    rel_delta = None
    if delta is not None and first is not None and abs(first) > 1e-12:
        rel_delta = delta / abs(first)
    return {
        "bin_count": len(series),
        "first": first,
        "last": last,
        "last_minus_first": delta,
        "relative_last_minus_first": rel_delta,
        "spearman_by_score_bin": spearman_for_ordered_bins(series),
    }


def collect_case_dirs(root: str | Path) -> list[Path]:
    root_path = Path(root).expanduser()
    if not root_path.exists():
        raise FileNotFoundError(f"input root not found: {root_path}")
    cases = []
    for child in sorted(root_path.iterdir()):
        if not child.is_dir():
            continue
        if (child / "localization_attribution" / "topk_iou_rank_summary.json").exists():
            cases.append(child)
    if not cases:
        raise ValueError(f"no case directories found under {root_path}")
    return cases


def summarize_case(case_dir: Path) -> dict[str, Any]:
    loc_dir = case_dir / "localization_attribution"
    topk_summary = read_json(loc_dir / "topk_iou_rank_summary.json")
    reliability = read_json(loc_dir / "score_iou_reliability.json")
    audit = read_json(loc_dir / "dense_axis_coordinate_audit.json")
    proposal = read_json(case_dir / "proposal_factor_summary.json")
    bins = reliability.get("bins", [])
    if not isinstance(bins, Sequence):
        bins = []

    top100_score = aggregate_mean(topk_summary, metric_key(100, 0.7, "gt_recall"))
    top100_oracle = aggregate_mean(topk_summary, metric_key(100, 0.7, "oracle_iou_rank_gt_recall"))
    top300_score = aggregate_mean(topk_summary, metric_key(300, 0.7, "gt_recall"))
    top300_oracle = aggregate_mean(topk_summary, metric_key(300, 0.7, "oracle_iou_rank_gt_recall"))
    top1000_score = aggregate_mean(topk_summary, metric_key(1000, 0.7, "gt_recall"))
    top1000_class_score = aggregate_mean(topk_summary, metric_key(1000, 0.7, "class_aware_gt_recall"))

    factor_summaries = {name: factor_summary_from_bins(bins, name) for name in FACTOR_NAMES}
    iou70_pos = finite_values(
        item.get("iou0p70_positive_fraction") for item in bins if isinstance(item, Mapping)
    )
    iou70_class_pos = finite_values(
        item.get("iou0p70_class_aware_positive_fraction") for item in bins if isinstance(item, Mapping)
    )
    class_iou = factor_summaries["class_aware_max_gt_iou"]
    max_iou = factor_summaries["max_gt_iou"]

    return {
        "case": case_dir.name,
        "rows_written": proposal.get("rows_written"),
        "coordinate_issue_count": coordinate_issue_count(audit),
        "top100_iou0p70_score_recall": top100_score,
        "top100_iou0p70_oracle_recall": top100_oracle,
        "top100_iou0p70_oracle_minus_score": (top100_oracle - top100_score)
        if top100_oracle is not None and top100_score is not None
        else None,
        "top300_iou0p70_score_recall": top300_score,
        "top300_iou0p70_oracle_recall": top300_oracle,
        "top300_iou0p70_oracle_minus_score": (top300_oracle - top300_score)
        if top300_oracle is not None and top300_score is not None
        else None,
        "top1000_iou0p70_score_recall": top1000_score,
        "top1000_iou0p70_class_aware_score_recall": top1000_class_score,
        "score_bin_iou0p70_positive_last_minus_first": (iou70_pos[-1] - iou70_pos[0])
        if len(iou70_pos) >= 2
        else None,
        "score_bin_iou0p70_class_aware_positive_last_minus_first": (iou70_class_pos[-1] - iou70_class_pos[0])
        if len(iou70_class_pos) >= 2
        else None,
        "score_bin_max_gt_iou_last_minus_first": max_iou["last_minus_first"],
        "score_bin_class_aware_iou_last_minus_first": class_iou["last_minus_first"],
        "factor_summaries": factor_summaries,
    }


def aggregate_factor_summaries(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out = {}
    for name in FACTOR_NAMES:
        per_case = [
            case.get("factor_summaries", {}).get(name, {})
            for case in cases
            if isinstance(case.get("factor_summaries", {}).get(name, {}), Mapping)
        ]
        spearman = [item.get("spearman_by_score_bin") for item in per_case]
        delta = [item.get("last_minus_first") for item in per_case]
        rel_delta = [item.get("relative_last_minus_first") for item in per_case]
        finite_delta = finite_values(delta)
        positive_signs = sum(1 for item in finite_delta if item > 0)
        negative_signs = sum(1 for item in finite_delta if item < 0)
        sign_consistency = None
        if finite_delta:
            sign_consistency = max(positive_signs, negative_signs) / float(len(finite_delta))
        out[name] = {
            "spearman_by_score_bin": stats(spearman),
            "last_minus_first": stats(delta),
            "relative_last_minus_first": stats(rel_delta),
            "sign_consistency": sign_consistency,
            "dominant_direction": "positive"
            if positive_signs >= negative_signs
            else "negative",
        }
    return out


def choose_e0_decision(aggregate: Mapping[str, Any]) -> tuple[str, list[str]]:
    coord_mean = mean_from_stat(aggregate["coordinate_issue_count"])
    top100_gap = mean_from_stat(aggregate["top100_iou0p70_oracle_minus_score"])
    top300_gap = mean_from_stat(aggregate["top300_iou0p70_oracle_minus_score"])
    top1000_score = mean_from_stat(aggregate["top1000_iou0p70_score_recall"])
    reasons = [
        f"coordinate_issue_count_mean={coord_mean}",
        f"top100_iou0p70_oracle_minus_score_mean={top100_gap}",
        f"top300_iou0p70_oracle_minus_score_mean={top300_gap}",
        f"top1000_iou0p70_score_recall_mean={top1000_score}",
    ]
    pass_gate = (
        coord_mean == 0.0
        and top1000_score is not None
        and top1000_score >= 0.75
        and (
            (top100_gap is not None and top100_gap >= 0.20)
            or (top300_gap is not None and top300_gap >= 0.04)
        )
    )
    if pass_gate:
        reasons.append("summary-level E0 thresholds pass, but no raw joined rows are local, so this is not a full evaluator-facing oracle-rerank bound")
        return SUMMARY_PROXY_PASS_NEEDS_RAW_ROWS, reasons
    reasons.append("summary-level E0 thresholds do not pass; do not start scorer route from this proxy")
    return SUMMARY_PROXY_FAIL, reasons


def choose_e1_decision(factors: Mapping[str, Any], aggregate: Mapping[str, Any]) -> tuple[str, list[str]]:
    strong = []
    for name, payload in factors.items():
        if not isinstance(payload, Mapping):
            continue
        spearman_mean = mean_from_stat(payload.get("spearman_by_score_bin"))
        sign_consistency = payload.get("sign_consistency")
        if spearman_mean is not None and abs(spearman_mean) >= 0.50 and sign_consistency is not None and sign_consistency >= 0.80:
            strong.append((name, spearman_mean, sign_consistency))
    class_delta = mean_from_stat(aggregate["score_bin_iou0p70_class_aware_positive_last_minus_first"])
    reasons = [
        f"strong_summary_factor_count={len(strong)}",
        "strong_summary_factors="
        + ", ".join(f"{name}:rho={rho:.3f}:sign={sign:.2f}" for name, rho, sign in strong[:6]),
        f"score_bin_iou0p70_class_aware_positive_last_minus_first_mean={class_delta}",
        "raw_missed_good_proposal_rows_available=false",
    ]
    if strong:
        reasons.append("summary bins expose stable score/factor trends, but missed-good suppressor coverage cannot be measured without raw joined proposal rows")
        return SCORE_FACTORS_VISIBLE_NEED_RAW_BUCKETS, reasons
    reasons.append("summary bins do not expose stable factor trends")
    return SCORE_FACTOR_EVIDENCE_WEAK, reasons


def run_analysis(
    *,
    input_root: str | Path,
    output_json: str | Path,
    output_markdown: str | Path | None = None,
) -> dict[str, Any]:
    cases = [summarize_case(case_dir) for case_dir in collect_case_dirs(input_root)]
    aggregate = {
        "case_count": len(cases),
        "rows_written": stats(case.get("rows_written") for case in cases),
        "coordinate_issue_count": stats(case.get("coordinate_issue_count") for case in cases),
        "top100_iou0p70_oracle_minus_score": stats(case.get("top100_iou0p70_oracle_minus_score") for case in cases),
        "top300_iou0p70_oracle_minus_score": stats(case.get("top300_iou0p70_oracle_minus_score") for case in cases),
        "top1000_iou0p70_score_recall": stats(case.get("top1000_iou0p70_score_recall") for case in cases),
        "top1000_iou0p70_class_aware_score_recall": stats(
            case.get("top1000_iou0p70_class_aware_score_recall") for case in cases
        ),
        "score_bin_iou0p70_positive_last_minus_first": stats(
            case.get("score_bin_iou0p70_positive_last_minus_first") for case in cases
        ),
        "score_bin_iou0p70_class_aware_positive_last_minus_first": stats(
            case.get("score_bin_iou0p70_class_aware_positive_last_minus_first") for case in cases
        ),
        "score_bin_max_gt_iou_last_minus_first": stats(case.get("score_bin_max_gt_iou_last_minus_first") for case in cases),
        "score_bin_class_aware_iou_last_minus_first": stats(
            case.get("score_bin_class_aware_iou_last_minus_first") for case in cases
        ),
    }
    factors = aggregate_factor_summaries(cases)
    e0_decision, e0_reasons = choose_e0_decision(aggregate)
    e1_decision, e1_reasons = choose_e1_decision(factors, aggregate)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": READY,
        "input_root": str(Path(input_root).expanduser()),
        "artifact_level": "summary_json_only",
        "full_evaluator_facing_oracle_rerank_bound_available": False,
        "raw_joined_proposal_rows_available": False,
        "e0_oracle_rerank_bound": {
            "decision": e0_decision,
            "reasons": e0_reasons,
            "full_gate_pass": False,
            "summary_proxy_pass": e0_decision == SUMMARY_PROXY_PASS_NEEDS_RAW_ROWS,
        },
        "e1_score_factor_bucket": {
            "decision": e1_decision,
            "reasons": e1_reasons,
            "full_gate_pass": False,
            "summary_proxy_pass": e1_decision == SCORE_FACTORS_VISIBLE_NEED_RAW_BUCKETS,
        },
        "aggregate": aggregate,
        "factor_aggregate": factors,
        "cases": cases,
        "recommended_next_steps": [
            "Use this as local summary-level E0/E1 evidence only.",
            "Do not start trainable scorer or remote precheck from this proxy alone.",
            "For full E0/E1, recover or regenerate bounded raw joined proposal rows and compute missed-good suppressor buckets.",
            "If raw-row E0/E1 remains positive, implement default-off P2-NIIQ-QualityRank-Calibrator-v0 with train-split-only targets.",
        ],
        "diagnostic_only": True,
        "uses_validation_gt": True,
        "uses_validation_gt_for_diagnostic_only": True,
        "uses_deploy_oracle": False,
        "uses_teacher": False,
        "uses_raw_prediction_cache": False,
        "metric_claim_allowed": False,
        "paper_claim_allowed": False,
    }
    write_json(output_json, summary)
    if output_markdown is not None:
        write_markdown(output_markdown, summary)
    return summary


def format_float(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return f"{number:.4f}"


def write_markdown(path: str | Path, summary: Mapping[str, Any]) -> None:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    aggregate = summary["aggregate"]
    lines = [
        "# P2 Quality Route Gates",
        "",
        f"E0 decision: `{summary['e0_oracle_rerank_bound']['decision']}`",
        f"E1 decision: `{summary['e1_score_factor_bucket']['decision']}`",
        "",
        "## Boundary",
        "",
        f"- artifact_level: `{summary['artifact_level']}`",
        f"- full_evaluator_facing_oracle_rerank_bound_available: `{summary['full_evaluator_facing_oracle_rerank_bound_available']}`",
        f"- raw_joined_proposal_rows_available: `{summary['raw_joined_proposal_rows_available']}`",
        f"- metric_claim_allowed: `{summary['metric_claim_allowed']}`",
        f"- paper_claim_allowed: `{summary['paper_claim_allowed']}`",
        "",
        "## Aggregate Signals",
        "",
        "| signal | mean |",
        "| --- | ---: |",
        f"| coordinate_issue_count | {format_float(aggregate['coordinate_issue_count']['mean'])} |",
        f"| top100 IoU0.70 oracle-minus-score | {format_float(aggregate['top100_iou0p70_oracle_minus_score']['mean'])} |",
        f"| top300 IoU0.70 oracle-minus-score | {format_float(aggregate['top300_iou0p70_oracle_minus_score']['mean'])} |",
        f"| top1000 IoU0.70 score recall | {format_float(aggregate['top1000_iou0p70_score_recall']['mean'])} |",
        f"| top1000 IoU0.70 class-aware score recall | {format_float(aggregate['top1000_iou0p70_class_aware_score_recall']['mean'])} |",
        f"| score-bin IoU0.70 positive last-first | {format_float(aggregate['score_bin_iou0p70_positive_last_minus_first']['mean'])} |",
        f"| score-bin class-aware IoU0.70 positive last-first | {format_float(aggregate['score_bin_iou0p70_class_aware_positive_last_minus_first']['mean'])} |",
        "",
        "## E0 Reasons",
        "",
        *[f"- {item}" for item in summary["e0_oracle_rerank_bound"]["reasons"]],
        "",
        "## E1 Reasons",
        "",
        *[f"- {item}" for item in summary["e1_score_factor_bucket"]["reasons"]],
        "",
        "## Top Factor Trends",
        "",
        "| factor | rho mean | delta mean | sign consistency |",
        "| --- | ---: | ---: | ---: |",
    ]
    factor_rows = []
    for name, payload in summary["factor_aggregate"].items():
        factor_rows.append(
            (
                name,
                abs(mean_from_stat(payload.get("spearman_by_score_bin")) or 0.0),
                payload,
            )
        )
    for name, _, payload in sorted(factor_rows, key=lambda item: item[1], reverse=True)[:8]:
        lines.append(
            "| {name} | {rho} | {delta} | {sign} |".format(
                name=name,
                rho=format_float(payload["spearman_by_score_bin"]["mean"]),
                delta=format_float(payload["last_minus_first"]["mean"]),
                sign=format_float(payload.get("sign_consistency")),
            )
        )
    lines.extend(
        [
            "",
            "## Recommended Next Steps",
            "",
            *[f"- {item}" for item in summary["recommended_next_steps"]],
            "",
        ]
    )
    out.write_text("\n".join(lines), encoding="utf-8")


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
    parser = argparse.ArgumentParser(description="Analyze local summary-level P2 quality-route gates.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown")
    args = parser.parse_args(argv)
    try:
        summary = run_analysis(
            input_root=args.input_root,
            output_json=args.output_json,
            output_markdown=args.output_markdown,
        )
    except Exception as exc:  # pragma: no cover - CLI guard
        print(json.dumps(strict_json_value(error_payload(exc)), sort_keys=True))
        return 1
    print(json.dumps(strict_json_value(summary), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
