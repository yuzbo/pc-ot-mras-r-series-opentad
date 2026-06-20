from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.export_pc_ot_mras_hard_positions import (  # noqa: E402
    resolve_pc_ot_mras_dynamic_budget_plan,
    strict_json_value,
    write_json,
)
from tools.bata.validate_pc_ot_mras_dynamic_budget_pipeline import (  # noqa: E402
    validate_pc_ot_mras_dynamic_budget_pipeline,
)


SCHEMA_VERSION = "pc_ot_mras_dynamic_budget_frontier_audit_v0"
READY = "PC_OT_MRAS_DYNAMIC_BUDGET_FRONTIER_AUDIT_READY"
NO_GO = "PC_OT_MRAS_DYNAMIC_BUDGET_FRONTIER_AUDIT_NO_GO"
DIFFICULTY_RANKS = {
    "easy": 0,
    "low": 0,
    "background": 0,
    "redundant": 0,
    "medium": 1,
    "mid": 1,
    "normal": 1,
    "hard": 2,
    "high": 2,
    "difficult": 2,
    "boundary": 2,
    "critical": 2,
}


def _to_plain(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "tolist"):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    return value


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be numeric") from None
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / float(len(values)))


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(item) for item in values)
    index = int(math.ceil(float(q) * float(len(ordered)))) - 1
    index = min(max(index, 0), len(ordered) - 1)
    return float(ordered[index])


def _distribution(values: Sequence[int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in values:
        key = str(int(item))
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda pair: int(pair[0])))


def _batch_values(value: Any, *, name: str, batch_size: int) -> list[Any]:
    data = _to_plain(value)
    if not isinstance(data, list) or len(data) != int(batch_size):
        raise ValueError(f"{name} must be a batch vector of length {int(batch_size)}")
    return data


def _optional_batch_floats(dynamic_plan: Mapping[str, Any], key: str, batch_size: int) -> list[float] | None:
    if key not in dynamic_plan:
        return None
    values = _batch_values(dynamic_plan[key], name=key, batch_size=batch_size)
    return [_finite_float(value, name=f"{key}[{idx}]") for idx, value in enumerate(values)]


def _optional_scalar_float(dynamic_plan: Mapping[str, Any], key: str) -> float | None:
    if key not in dynamic_plan:
        return None
    data = _to_plain(dynamic_plan[key])
    if isinstance(data, list):
        if len(data) != 1:
            return None
        data = data[0]
    return _finite_float(data, name=key)


def _normalize_reference_budgets(
    dynamic_plan: Mapping[str, Any],
    budgets: Sequence[int],
    reference_budgets: Sequence[int] | None,
) -> list[int]:
    source = reference_budgets
    if source is None and "budget_values" in dynamic_plan:
        source = _to_plain(dynamic_plan["budget_values"])
    if source is None:
        source = sorted(set(int(item) for item in budgets))
    try:
        out = sorted({int(item) for item in source})
    except TypeError as exc:
        raise ValueError("reference_budgets must be a sequence of positive integers") from exc
    if not out or any(item <= 0 for item in out):
        raise ValueError("reference_budgets must contain positive integers")
    return out


def _fixed_budget_summary(reference_budgets: Sequence[int], valid_lens: Sequence[int], fixed_reference: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for budget in reference_budgets:
        selected = [min(int(budget), int(valid_len)) for valid_len in valid_lens]
        avg_selected = _mean([float(item) for item in selected])
        rows.append(
            {
                "budget": int(budget),
                "average_selected": avg_selected,
                "clipped_sample_count": int(sum(1 for item in valid_lens if int(item) < int(budget))),
                "savings_vs_fixed_reference": None
                if not fixed_reference
                else float((float(fixed_reference) - float(avg_selected or 0.0)) / float(fixed_reference)),
            }
        )
    return rows


def _pairwise_monotonicity(values: Sequence[float] | None, budgets: Sequence[int], *, name: str) -> dict[str, Any]:
    if values is None:
        return {"name": name, "available": False, "pair_count": 0, "violation_count": 0, "passed": None}
    if len(values) != len(budgets):
        raise ValueError(f"{name} length must equal budget count")
    pair_count = 0
    violation_count = 0
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            if float(values[left]) == float(values[right]):
                continue
            pair_count += 1
            if float(values[left]) < float(values[right]) and int(budgets[left]) > int(budgets[right]):
                violation_count += 1
            if float(values[left]) > float(values[right]) and int(budgets[left]) < int(budgets[right]):
                violation_count += 1
    return {
        "name": name,
        "available": True,
        "pair_count": int(pair_count),
        "violation_count": int(violation_count),
        "passed": bool(violation_count == 0),
    }


def _normalize_difficulty(label: str) -> str:
    normalized = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(label)).strip("_")
    if normalized not in DIFFICULTY_RANKS:
        raise ValueError(f"unknown difficulty label: {label}")
    return normalized


def _difficulty_order(difficulty_labels: Sequence[str] | None, budgets: Sequence[int]) -> dict[str, Any]:
    if difficulty_labels is None:
        return {"available": False, "group_means": {}, "violation_count": 0, "passed": None}
    if len(difficulty_labels) != len(budgets):
        raise ValueError("difficulty_labels length must equal budget count")
    groups: dict[int, list[float]] = {}
    labels_by_rank: dict[int, str] = {}
    for label, budget in zip(difficulty_labels, budgets):
        normalized = _normalize_difficulty(label)
        rank = DIFFICULTY_RANKS[normalized]
        groups.setdefault(rank, []).append(float(budget))
        labels_by_rank.setdefault(rank, normalized)

    group_means = {
        labels_by_rank[rank]: _mean(groups[rank])
        for rank in sorted(groups)
    }
    ranked = [(rank, _mean(groups[rank])) for rank in sorted(groups)]
    violation_count = 0
    for (_left_rank, left_mean), (_right_rank, right_mean) in zip(ranked, ranked[1:]):
        if right_mean is not None and left_mean is not None and float(right_mean) + 1.0e-9 < float(left_mean):
            violation_count += 1
    return {
        "available": True,
        "group_means": group_means,
        "violation_count": int(violation_count),
        "passed": bool(violation_count == 0),
    }


def audit_pc_ot_mras_dynamic_budget_frontier(
    dynamic_plan: Mapping[str, Any],
    *,
    sample_ids: Sequence[str] | None = None,
    difficulty_labels: Sequence[str] | None = None,
    difficulty_scores: Sequence[float] | None = None,
    fixed_budget_reference: int = 384,
    reference_budgets: Sequence[int] | None = None,
    split: str = "val",
    strict_temporal_grid: bool = True,
) -> dict[str, Any]:
    """Audit whether an R22 dynamic-budget plan exposes a useful budget frontier.

    This local-only audit validates the R22 -> R23 -> R24 -> R25 protocol chain
    first, then checks budget distribution, savings, coverage cap, clipping,
    and monotonic budget allocation against deploy-visible budget scores or
    explicit synthetic difficulty labels. It does not run detector training or
    evaluation, read datasets/checkpoints, measure runtime/FLOPs, validate
    scanner quality, or authorize deployment/paper claims.
    """

    if not isinstance(dynamic_plan, Mapping):
        raise ValueError("dynamic_plan must be a mapping")
    if fixed_budget_reference <= 0:
        raise ValueError("fixed_budget_reference must be positive")

    pipeline = validate_pc_ot_mras_dynamic_budget_pipeline(
        dynamic_plan,
        sample_ids=sample_ids,
        split=split,
        strict_temporal_grid=strict_temporal_grid,
    )
    rows = resolve_pc_ot_mras_dynamic_budget_plan(dynamic_plan, sample_ids=sample_ids)

    budgets = [int(row["budget"]) for row in rows]
    valid_lens = [int(row["valid_len"]) for row in rows]
    budget_scores = _optional_batch_floats(dynamic_plan, "budget_scores", len(rows))
    coverage_shares = _optional_batch_floats(dynamic_plan, "coverage_share", len(rows)) or []
    max_coverage_share = _optional_scalar_float(dynamic_plan, "max_coverage_share")
    difficulty_score_values = None if difficulty_scores is None else [
        _finite_float(item, name=f"difficulty_scores[{idx}]")
        for idx, item in enumerate(difficulty_scores)
    ]
    if difficulty_score_values is not None and len(difficulty_score_values) != len(rows):
        raise ValueError("difficulty_scores length must equal budget count")

    reference_budget_values = _normalize_reference_budgets(dynamic_plan, budgets, reference_budgets)
    budget_avg = _mean([float(item) for item in budgets])
    max_config_budget = max(reference_budget_values)
    coverage_cap_violations = 0
    if coverage_shares and max_coverage_share is not None:
        coverage_cap_violations = sum(
            1 for item in coverage_shares if float(item) > float(max_coverage_share) + 1.0e-6
        )

    budget_score_monotonicity = _pairwise_monotonicity(budget_scores, budgets, name="budget_scores")
    difficulty_score_monotonicity = _pairwise_monotonicity(
        difficulty_score_values,
        budgets,
        name="difficulty_scores",
    )
    difficulty_budget_order = _difficulty_order(difficulty_labels, budgets)

    sensitivity_signals = [
        budget_score_monotonicity,
        difficulty_score_monotonicity,
    ]
    any_monotonic_signal_passed = any(item["passed"] is True for item in sensitivity_signals)
    if difficulty_budget_order["passed"] is True:
        any_monotonic_signal_passed = True

    budget_sensitivity_passed = bool(
        min(budgets) < max(budgets)
        and any_monotonic_signal_passed
        and all(item["violation_count"] == 0 for item in sensitivity_signals)
        and int(difficulty_budget_order["violation_count"]) == 0
    )
    hard_protocol_passed = bool(
        int(pipeline["exact_budget_violations"]) == 0
        and bool(pipeline["center_matches_rows"])
        and int(coverage_cap_violations) == 0
    )
    decision = READY if hard_protocol_passed and budget_sensitivity_passed else NO_GO

    return {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "row_count": len(rows),
        "sample_ids": [str(row["sample_id"]) for row in rows],
        "budgets": budgets,
        "budget_distribution": _distribution(budgets),
        "budget_min": min(budgets),
        "budget_max": max(budgets),
        "budget_mean": budget_avg,
        "fixed_budget_reference": int(fixed_budget_reference),
        "savings_vs_fixed_reference": float((float(fixed_budget_reference) - float(budget_avg or 0.0)) / float(fixed_budget_reference)),
        "dense_valid_lens": valid_lens,
        "short_valid_len_clipped_count": int(
            sum(1 for budget, valid_len in zip(budgets, valid_lens) if int(valid_len) < max_config_budget and int(budget) == int(valid_len))
        ),
        "reference_budget_frontier": _fixed_budget_summary(
            reference_budget_values,
            valid_lens,
            int(fixed_budget_reference),
        ),
        "coverage_share_mean": _mean(coverage_shares),
        "coverage_share_max": max(coverage_shares) if coverage_shares else None,
        "coverage_share_p95": _percentile(coverage_shares, 0.95),
        "coverage_cap": max_coverage_share,
        "coverage_cap_violations": int(coverage_cap_violations),
        "budget_score_monotonicity": budget_score_monotonicity,
        "difficulty_score_monotonicity": difficulty_score_monotonicity,
        "difficulty_budget_order": difficulty_budget_order,
        "budget_sensitivity_passed": budget_sensitivity_passed,
        "hard_protocol_passed": hard_protocol_passed,
        "pipeline_summary": pipeline,
        "frontier_audit": True,
        "dynamic_budget_quality_validation": False,
        "dynamic_budget_validation": False,
        "scanner_quality_validation": False,
        "metric_claim_allowed": False,
        "paper_claim_allowed": False,
        "runtime_flops_claim_allowed": False,
        "deploy_claim_allowed": False,
        "remote_sync_allowed": False,
        "remote_precheck_allowed": False,
        "slurm_gpu_allowed": False,
        "tools_train_allowed": False,
        "tools_test_allowed": False,
        "detector_map_allowed": False,
        "dataset_checkpoint_access_allowed": False,
    }


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("dynamic plan JSON must contain one object")
    return payload


def run_json_frontier_audit(
    input_json: str | Path,
    *,
    summary_json: str | Path | None = None,
    sample_ids: Sequence[str] | None = None,
    difficulty_labels: Sequence[str] | None = None,
    difficulty_scores: Sequence[float] | None = None,
    fixed_budget_reference: int = 384,
    reference_budgets: Sequence[int] | None = None,
    split: str = "val",
    strict_temporal_grid: bool = True,
) -> dict[str, Any]:
    summary = audit_pc_ot_mras_dynamic_budget_frontier(
        read_json(input_json),
        sample_ids=sample_ids,
        difficulty_labels=difficulty_labels,
        difficulty_scores=difficulty_scores,
        fixed_budget_reference=int(fixed_budget_reference),
        reference_budgets=reference_budgets,
        split=split,
        strict_temporal_grid=strict_temporal_grid,
    )
    if summary_json is not None:
        write_json(summary_json, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit the local PC-OT-MRAS dynamic-budget frontier without detector execution."
    )
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--summary-json")
    parser.add_argument("--sample-id", action="append", dest="sample_ids")
    parser.add_argument("--difficulty-label", action="append", dest="difficulty_labels")
    parser.add_argument("--difficulty-score", action="append", type=float, dest="difficulty_scores")
    parser.add_argument("--fixed-budget-reference", type=int, default=384)
    parser.add_argument("--reference-budget", action="append", type=int, dest="reference_budgets")
    parser.add_argument("--split", default="val")
    parser.add_argument("--non-strict-temporal-grid", action="store_true")
    args = parser.parse_args(argv)

    try:
        summary = run_json_frontier_audit(
            args.input_json,
            summary_json=args.summary_json,
            sample_ids=args.sample_ids,
            difficulty_labels=args.difficulty_labels,
            difficulty_scores=args.difficulty_scores,
            fixed_budget_reference=int(args.fixed_budget_reference),
            reference_budgets=args.reference_budgets,
            split=args.split,
            strict_temporal_grid=not bool(args.non_strict_temporal_grid),
        )
    except Exception as exc:  # pragma: no cover - CLI guard
        print(json.dumps({"schema_version": SCHEMA_VERSION, "decision": NO_GO, "error": str(exc)}))
        return 1

    print(json.dumps(strict_json_value(summary), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
