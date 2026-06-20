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

from tools.bata.audit_pc_ot_mras_dynamic_budget_frontier import (  # noqa: E402
    audit_pc_ot_mras_dynamic_budget_frontier,
)
from tools.bata.export_pc_ot_mras_hard_positions import (  # noqa: E402
    resolve_pc_ot_mras_dynamic_budget_plan,
    strict_json_value,
    write_json,
)


SCHEMA_VERSION = "pc_ot_mras_synthetic_task_utility_audit_v0"
READY = "PC_OT_MRAS_SYNTHETIC_TASK_UTILITY_AUDIT_READY"
NO_GO = "PC_OT_MRAS_SYNTHETIC_TASK_UTILITY_AUDIT_NO_GO"
TASK_SCHEMA_VERSION = "pc_ot_mras_synthetic_task_spec_v0"
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
FORBIDDEN_TASK_SPEC_TOKENS = (
    "gt",
    "groundtruth",
    "teacher",
    "oracle",
    "cache",
    "featurecache",
    "prediction",
    "predictioncache",
    "rawprediction",
    "checkpoint",
    "ckpt",
    "result",
    "annotation",
    "annfile",
    "dataset",
)
FALSE_ONLY_TASK_SPEC_FLAGS = frozenset(
    {
        "uses_gt",
        "uses_teacher",
        "uses_raw_prediction",
        "uses_checkpoint",
        "uses_cache",
        "uses_oracle",
        "metric_claim_allowed",
        "paper_claim_allowed",
        "deploy_claim_allowed",
        "dynamic_budget_validation",
        "scanner_quality_validation",
    }
)


def _to_plain(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "tolist"):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    return value


def _normalized_key(value: Any) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _has_forbidden_task_fragment(value: Any) -> bool:
    normalized = _normalized_key(value)
    return any(token in normalized for token in FORBIDDEN_TASK_SPEC_TOKENS)


def _validate_task_spec_no_forbidden_payload(value: Any, *, path: str = "task_spec") -> None:
    data = _to_plain(value)
    if isinstance(data, Mapping):
        for key, item in data.items():
            key_text = str(key or "")
            if key_text in FALSE_ONLY_TASK_SPEC_FLAGS:
                if bool(_to_plain(item)):
                    raise ValueError(f"{path}.{key_text} must be false in synthetic task spec")
                continue
            if _has_forbidden_task_fragment(key_text):
                raise ValueError(f"{path}.{key_text}: forbidden real-data/leakage key in synthetic task spec")
            _validate_task_spec_no_forbidden_payload(item, path=f"{path}.{key_text}")
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            _validate_task_spec_no_forbidden_payload(item, path=f"{path}[{idx}]")
    elif isinstance(data, str) and _has_forbidden_task_fragment(data):
        raise ValueError(f"{path}: forbidden real-data/leakage value in synthetic task spec")


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


def _strict_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        out = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer") from None
    if float(out) != float(value):
        raise ValueError(f"{name} must be an integer")
    return out


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / float(len(values)))


def _normalize_difficulty(value: Any) -> str:
    normalized = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value)).strip("_")
    if normalized not in DIFFICULTY_RANKS:
        raise ValueError(f"unknown synthetic difficulty: {value}")
    return normalized


def _as_position_list(value: Any, *, name: str) -> list[int]:
    data = _to_plain(value)
    if data is None:
        return []
    if not isinstance(data, list):
        raise ValueError(f"{name} must be a list of positions")
    return [_strict_int(item, name=f"{name}[{idx}]") for idx, item in enumerate(data)]


def _normalize_task_windows(task_spec: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    _validate_task_spec_no_forbidden_payload(task_spec)
    if bool(_to_plain(task_spec.get("synthetic_task_utility_only"))) is not True:
        raise ValueError("task_spec.synthetic_task_utility_only must be true")
    if bool(_to_plain(task_spec.get("uses_gt", False))):
        raise ValueError("task_spec.uses_gt must be false")
    if bool(_to_plain(task_spec.get("uses_teacher", False))):
        raise ValueError("task_spec.uses_teacher must be false")
    if bool(_to_plain(task_spec.get("uses_raw_prediction", False))):
        raise ValueError("task_spec.uses_raw_prediction must be false")
    if bool(_to_plain(task_spec.get("uses_checkpoint", False))):
        raise ValueError("task_spec.uses_checkpoint must be false")
    if str(task_spec.get("schema_version")) != TASK_SCHEMA_VERSION:
        raise ValueError(f"task_spec.schema_version must be {TASK_SCHEMA_VERSION}")

    windows = _to_plain(task_spec.get("windows"))
    if not isinstance(windows, list) or len(windows) != len(rows):
        raise ValueError("task_spec.windows must align with dynamic-plan rows")

    by_id = {str(row["sample_id"]): row for row in rows}
    normalized: list[dict[str, Any]] = []
    for idx, item in enumerate(windows):
        if not isinstance(item, Mapping):
            raise ValueError(f"task_spec.windows[{idx}] must be an object")
        sample_id = str(item.get("sample_id", rows[idx]["sample_id"]))
        if sample_id not in by_id:
            raise ValueError(f"{sample_id}: synthetic task sample_id not found in dynamic plan")
        row = by_id[sample_id]
        valid_len = int(row["valid_len"])
        start = _strict_int(item.get("start"), name=f"{sample_id}.start")
        end = _strict_int(item.get("end"), name=f"{sample_id}.end")
        if not 0 <= start <= end < valid_len:
            raise ValueError(f"{sample_id}: synthetic start/end must lie inside dense valid length")
        interior = _as_position_list(item.get("interior_peaks", []), name=f"{sample_id}.interior_peaks")
        background = _as_position_list(item.get("background_positions", []), name=f"{sample_id}.background_positions")
        for pos in [*interior, *background]:
            if not 0 <= int(pos) < valid_len:
                raise ValueError(f"{sample_id}: synthetic position {pos} lies outside valid length")
        normalized.append(
            {
                "sample_id": sample_id,
                "valid_len": valid_len,
                "start": start,
                "end": end,
                "interior_peaks": sorted({int(pos) for pos in interior}),
                "background_positions": sorted({int(pos) for pos in background}),
                "difficulty": _normalize_difficulty(item.get("difficulty", "medium")),
            }
        )
    return normalized


def _uniform_positions(valid_len: int, budget: int) -> list[int]:
    count = min(int(valid_len), int(budget))
    if count <= 0:
        raise ValueError("budget must be positive")
    if count >= int(valid_len):
        return list(range(int(valid_len)))
    selected: list[int] = []
    for idx in range(count):
        pos = int(round(float(idx) * float(valid_len - 1) / float(count - 1))) if count > 1 else 0
        if pos not in selected:
            selected.append(pos)
    filler = 0
    while len(selected) < count:
        if filler not in selected:
            selected.append(filler)
        filler += 1
    return sorted(selected[:count])


def _utility_curve(
    task: Mapping[str, Any],
    *,
    boundary_radius: int,
    boundary_weight: float,
    interior_weight: float,
) -> list[float]:
    valid_len = int(task["valid_len"])
    values = [0.0 for _idx in range(valid_len)]
    for center in (int(task["start"]), int(task["end"])):
        for pos in range(max(0, center - boundary_radius), min(valid_len, center + boundary_radius + 1)):
            values[pos] = max(values[pos], float(boundary_weight))
    for center in task["interior_peaks"]:
        values[int(center)] = max(values[int(center)], float(interior_weight))
    return values


def _support(selected: set[int], center: int, radius: int) -> bool:
    return any(abs(int(pos) - int(center)) <= int(radius) for pos in selected)


def _task_metrics(
    *,
    task: Mapping[str, Any],
    selected_positions: Sequence[int],
    boundary_radius: int,
    boundary_weight: float,
    interior_weight: float,
) -> dict[str, Any]:
    selected = {int(pos) for pos in selected_positions}
    utility = _utility_curve(
        task,
        boundary_radius=boundary_radius,
        boundary_weight=boundary_weight,
        interior_weight=interior_weight,
    )
    selected_utility = sum(float(utility[pos]) for pos in selected if 0 <= pos < len(utility))
    top_positions = {
        pos for pos, _score in sorted(
            [(idx, score) for idx, score in enumerate(utility)],
            key=lambda item: (-item[1], item[0]),
        )[: len(selected)]
    }
    oracle_utility = sum(float(utility[pos]) for pos in top_positions)
    uniform = _uniform_positions(int(task["valid_len"]), len(selected))
    uniform_utility = sum(float(utility[pos]) for pos in uniform)
    start_supported = _support(selected, int(task["start"]), boundary_radius)
    end_supported = _support(selected, int(task["end"]), boundary_radius)
    interior = set(int(pos) for pos in task["interior_peaks"])
    background = set(int(pos) for pos in task["background_positions"])
    return {
        "sample_id": str(task["sample_id"]),
        "selected_count": int(len(selected)),
        "boundary_support": float((int(start_supported) + int(end_supported)) / 2.0),
        "zero_boundary_support": bool(not start_supported and not end_supported),
        "interior_peak_recall": None if not interior else float(len(selected.intersection(interior)) / float(len(interior))),
        "background_selected_share": float(len(selected.intersection(background)) / float(max(len(selected), 1))),
        "selected_task_utility": float(selected_utility),
        "oracle_topk_task_utility": float(oracle_utility),
        "uniform_task_utility": float(uniform_utility),
        "selected_vs_oracle_topk_ratio": None if oracle_utility <= 1.0e-12 else float(selected_utility / oracle_utility),
        "utility_gain_vs_uniform": float(selected_utility - uniform_utility),
    }


def _selected_positions(row: Mapping[str, Any]) -> list[int]:
    raw = _to_plain(row.get("selected_positions"))
    if not isinstance(raw, list):
        raise ValueError("resolved row selected_positions must be a list")
    return [int(item) for item in raw[: int(row["budget"])]]


def _difficulty_budget_monotonic(tasks: Sequence[Mapping[str, Any]], budgets: Sequence[int]) -> dict[str, Any]:
    groups: dict[int, list[float]] = {}
    labels: dict[int, str] = {}
    for task, budget in zip(tasks, budgets):
        rank = DIFFICULTY_RANKS[str(task["difficulty"])]
        groups.setdefault(rank, []).append(float(budget))
        labels.setdefault(rank, str(task["difficulty"]))
    ordered = [(rank, _mean(groups[rank])) for rank in sorted(groups)]
    violations = 0
    for (_left_rank, left), (_right_rank, right) in zip(ordered, ordered[1:]):
        if left is not None and right is not None and float(right) + 1.0e-9 < float(left):
            violations += 1
    return {
        "group_means": {labels[rank]: _mean(groups[rank]) for rank in sorted(groups)},
        "violation_count": int(violations),
        "passed": bool(violations == 0),
    }


def audit_pc_ot_mras_synthetic_task_utility(
    dynamic_plan: Mapping[str, Any],
    task_spec: Mapping[str, Any],
    *,
    sample_ids: Sequence[str] | None = None,
    fixed_budget_reference: int = 384,
    reference_budgets: Sequence[int] | None = None,
    split: str = "synthetic",
    strict_temporal_grid: bool = True,
    boundary_radius: int = 0,
    boundary_support_threshold: float = 0.95,
    min_oracle_topk_ratio: float = 0.80,
    min_utility_gain_vs_uniform: float = 1.0,
    max_background_selected_share: float = 0.35,
) -> dict[str, Any]:
    """Audit whether a dynamic plan targets synthetic TAD-sensitive structure.

    This local audit is stricter than R26's budget-frontier check. It requires
    the R25/R26 protocol chain to pass, then checks synthetic action boundary
    support, interior evidence coverage, background suppression, and same-budget
    exact-uniform controls. It does not read real data, GT annotations,
    checkpoints, raw predictions, or detector outputs.
    """

    if str(split) != "synthetic":
        raise ValueError("R27 synthetic task utility audit must use split='synthetic'")
    if int(boundary_radius) < 0:
        raise ValueError("boundary_radius must be non-negative")

    rows = resolve_pc_ot_mras_dynamic_budget_plan(dynamic_plan, sample_ids=sample_ids)
    tasks = _normalize_task_windows(task_spec, rows)
    difficulty_labels = [str(task["difficulty"]) for task in tasks]
    frontier = audit_pc_ot_mras_dynamic_budget_frontier(
        dynamic_plan,
        sample_ids=sample_ids,
        difficulty_labels=difficulty_labels,
        fixed_budget_reference=int(fixed_budget_reference),
        reference_budgets=reference_budgets,
        split=split,
        strict_temporal_grid=strict_temporal_grid,
    )

    task_by_id = {str(task["sample_id"]): task for task in tasks}
    metrics = []
    for row in rows:
        task = task_by_id[str(row["sample_id"])]
        metrics.append(
            _task_metrics(
                task=task,
                selected_positions=_selected_positions(row),
                boundary_radius=int(boundary_radius),
                boundary_weight=2.0,
                interior_weight=1.0,
            )
        )

    boundary_support_mean = _mean([float(item["boundary_support"]) for item in metrics])
    zero_boundary_support_rate = _mean([1.0 if item["zero_boundary_support"] else 0.0 for item in metrics])
    interior_peak_recall_mean = _mean([
        float(item["interior_peak_recall"]) for item in metrics if item["interior_peak_recall"] is not None
    ])
    background_selected_share_mean = _mean([float(item["background_selected_share"]) for item in metrics])
    oracle_topk_ratio_mean = _mean([
        float(item["selected_vs_oracle_topk_ratio"])
        for item in metrics
        if item["selected_vs_oracle_topk_ratio"] is not None
    ])
    utility_gain_vs_uniform_mean = _mean([float(item["utility_gain_vs_uniform"]) for item in metrics])
    difficulty_budget_order = _difficulty_budget_monotonic(tasks, [int(row["budget"]) for row in rows])

    task_utility_passed = bool(
        frontier["decision"].endswith("READY")
        and boundary_support_mean is not None
        and boundary_support_mean >= float(boundary_support_threshold)
        and zero_boundary_support_rate == 0.0
        and oracle_topk_ratio_mean is not None
        and oracle_topk_ratio_mean >= float(min_oracle_topk_ratio)
        and utility_gain_vs_uniform_mean is not None
        and utility_gain_vs_uniform_mean >= float(min_utility_gain_vs_uniform)
        and background_selected_share_mean is not None
        and background_selected_share_mean <= float(max_background_selected_share)
        and difficulty_budget_order["passed"] is True
    )
    decision = READY if task_utility_passed else NO_GO

    return {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "task_schema_version": TASK_SCHEMA_VERSION,
        "row_count": len(rows),
        "sample_ids": [str(row["sample_id"]) for row in rows],
        "budgets": [int(row["budget"]) for row in rows],
        "boundary_radius": int(boundary_radius),
        "boundary_support_mean": boundary_support_mean,
        "zero_boundary_support_rate": zero_boundary_support_rate,
        "interior_peak_recall_mean": interior_peak_recall_mean,
        "background_selected_share_mean": background_selected_share_mean,
        "selected_vs_oracle_topk_ratio_mean": oracle_topk_ratio_mean,
        "utility_gain_vs_uniform_mean": utility_gain_vs_uniform_mean,
        "difficulty_budget_order": difficulty_budget_order,
        "task_rows": metrics,
        "frontier_audit_summary": frontier,
        "synthetic_task_utility_audit": True,
        "task_utility_passed": task_utility_passed,
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
        raise ValueError("JSON file must contain one object")
    return payload


def run_json_synthetic_task_utility_audit(
    input_json: str | Path,
    task_json: str | Path,
    *,
    summary_json: str | Path | None = None,
    sample_ids: Sequence[str] | None = None,
    fixed_budget_reference: int = 384,
    reference_budgets: Sequence[int] | None = None,
    boundary_radius: int = 0,
) -> dict[str, Any]:
    summary = audit_pc_ot_mras_synthetic_task_utility(
        read_json(input_json),
        read_json(task_json),
        sample_ids=sample_ids,
        fixed_budget_reference=int(fixed_budget_reference),
        reference_budgets=reference_budgets,
        boundary_radius=int(boundary_radius),
    )
    if summary_json is not None:
        write_json(summary_json, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit synthetic task-utility alignment for a local PC-OT-MRAS dynamic-budget plan."
    )
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--task-json", required=True)
    parser.add_argument("--summary-json")
    parser.add_argument("--sample-id", action="append", dest="sample_ids")
    parser.add_argument("--fixed-budget-reference", type=int, default=384)
    parser.add_argument("--reference-budget", action="append", type=int, dest="reference_budgets")
    parser.add_argument("--boundary-radius", type=int, default=0)
    args = parser.parse_args(argv)

    try:
        summary = run_json_synthetic_task_utility_audit(
            args.input_json,
            args.task_json,
            summary_json=args.summary_json,
            sample_ids=args.sample_ids,
            fixed_budget_reference=int(args.fixed_budget_reference),
            reference_budgets=args.reference_budgets,
            boundary_radius=int(args.boundary_radius),
        )
    except Exception as exc:  # pragma: no cover - CLI guard
        print(json.dumps({"schema_version": SCHEMA_VERSION, "decision": NO_GO, "error": str(exc)}))
        return 1

    print(json.dumps(strict_json_value(summary), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
