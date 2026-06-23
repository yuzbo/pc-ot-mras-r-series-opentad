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


SCHEMA_VERSION = "p2_raw_row_oracle_rerank_closeout_v0"
READY = "P2_RAW_ROW_ORACLE_RERANK_CLOSEOUT_READY"
NO_GO = "P2_RAW_ROW_ORACLE_RERANK_CLOSEOUT_NO_GO"


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


def read_jsonl(path: str | Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
            if limit is not None and len(rows) >= int(limit):
                break
    return rows


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _stats(values: Sequence[Any]) -> dict[str, Any]:
    finite = [float(value) for value in values if _as_float(value) is not None]
    if not finite:
        return {"count": 0, "min": None, "mean": None, "max": None}
    return {
        "count": len(finite),
        "min": min(finite),
        "mean": sum(finite) / float(len(finite)),
        "max": max(finite),
    }


def _parse_number_list(value: str, *, as_int: bool = False) -> list[Any]:
    out = []
    for item in str(value).split(","):
        stripped = item.strip()
        if not stripped:
            continue
        out.append(int(stripped) if as_int else float(stripped))
    if not out:
        raise argparse.ArgumentTypeError("list must contain at least one value")
    return out


def _group_key(row: Mapping[str, Any], group_by: str) -> str:
    if group_by == "video":
        return str(row.get("video_id", "unknown"))
    return str(row.get("diagnostic_group_key") or row.get("sample_id") or row.get("video_id") or "unknown")


def _ranked_rows(rows: Sequence[Mapping[str, Any]], key: str, *, descending: bool = True) -> list[Mapping[str, Any]]:
    return sorted(
        rows,
        key=lambda row: _as_float(row.get(key)) if _as_float(row.get(key)) is not None else -float("inf"),
        reverse=descending,
    )


def _topk_case_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    topk: Sequence[int],
    iou_thresholds: Sequence[float],
    iou_key: str,
    prefix: str,
) -> dict[str, Any]:
    score_ordered = _ranked_rows(rows, "final_score")
    oracle_ordered = _ranked_rows(rows, iou_key)
    out: dict[str, Any] = {}
    for rank, row in enumerate(score_ordered, start=1):
        row["_raw_row_score_rank"] = rank
    for k in topk:
        k_int = int(k)
        score_subset = score_ordered[:k_int]
        oracle_subset = oracle_ordered[:k_int]
        score_ids = {int(row["_raw_row_index"]) for row in score_subset}
        for thr in iou_thresholds:
            thr_f = float(thr)
            key_prefix = f"{prefix}_top{k_int}_iou{thr_f:.2f}".replace(".", "p")
            all_positive = [row for row in rows if (_as_float(row.get(iou_key)) or 0.0) >= thr_f]
            score_positive = [row for row in score_subset if (_as_float(row.get(iou_key)) or 0.0) >= thr_f]
            oracle_positive = [row for row in oracle_subset if (_as_float(row.get(iou_key)) or 0.0) >= thr_f]
            missed_oracle = [row for row in oracle_positive if int(row["_raw_row_index"]) not in score_ids]
            best_rank = min((int(row.get("_raw_row_score_rank", 10**9)) for row in all_positive), default=None)
            out[f"{key_prefix}_all_positive_count"] = int(len(all_positive))
            out[f"{key_prefix}_score_positive_count"] = int(len(score_positive))
            out[f"{key_prefix}_oracle_positive_count"] = int(len(oracle_positive))
            out[f"{key_prefix}_oracle_minus_score_positive_count"] = int(len(oracle_positive) - len(score_positive))
            out[f"{key_prefix}_missed_oracle_positive_count"] = int(len(missed_oracle))
            out[f"{key_prefix}_best_positive_score_rank"] = best_rank
            out[f"{key_prefix}_positive_score_rank_mean"] = (
                sum(int(row["_raw_row_score_rank"]) for row in all_positive) / float(len(all_positive))
                if all_positive
                else None
            )
            out[f"{key_prefix}_score_positive_fraction_of_all_positive"] = (
                len(score_positive) / float(len(all_positive)) if all_positive else None
            )
    return out


def analyze_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
    topk: Sequence[int] = (100, 300, 1000),
    iou_thresholds: Sequence[float] = (0.3, 0.5, 0.7),
    group_by: str = "sample",
) -> dict[str, Any]:
    indexed_rows = [dict(row, _raw_row_index=idx) for idx, row in enumerate(rows)]
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in indexed_rows:
        groups.setdefault(_group_key(row, group_by), []).append(row)
    group_rows = []
    for group_key, group in sorted(groups.items()):
        item: dict[str, Any] = {
            "group_key": group_key,
            "row_count": int(len(group)),
            "score": _stats([row.get("final_score") for row in group]),
            "max_gt_iou": _stats([row.get("diagnostic_max_gt_iou_seconds") for row in group]),
            "class_aware_max_gt_iou": _stats([row.get("diagnostic_class_aware_max_gt_iou_seconds") for row in group]),
        }
        item.update(
            _topk_case_metrics(
                group,
                topk=topk,
                iou_thresholds=iou_thresholds,
                iou_key="diagnostic_max_gt_iou_seconds",
                prefix="any",
            )
        )
        item.update(
            _topk_case_metrics(
                group,
                topk=topk,
                iou_thresholds=iou_thresholds,
                iou_key="diagnostic_class_aware_max_gt_iou_seconds",
                prefix="class",
            )
        )
        group_rows.append(item)
    numeric_keys = sorted({key for item in group_rows for key, value in item.items() if isinstance(value, (int, float))})
    aggregate = {key: _stats([item.get(key) for item in group_rows]) for key in numeric_keys}
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": READY,
        "label": str(label),
        "row_count": int(len(indexed_rows)),
        "group_count": int(len(group_rows)),
        "group_by": str(group_by),
        "topk": [int(item) for item in topk],
        "iou_thresholds": [float(item) for item in iou_thresholds],
        "groups": group_rows,
        "aggregate": aggregate,
        "raw_joined_proposal_rows_available": True,
        "full_evaluator_facing_oracle_rerank_bound_available": False,
        "result_detection_geometry_close_out_available": False,
        "diagnostic_scope": "raw_joined_proposal_rows_only",
        "diagnostic_only": True,
        "uses_validation_gt": True,
        "uses_teacher": False,
        "uses_oracle_for_analysis_only": True,
        "uses_deploy_oracle": False,
        "metric_claim_allowed": False,
        "paper_claim_allowed": False,
    }


def build_closeout(
    inputs: Sequence[tuple[str, str | Path]],
    *,
    output_json: str | Path | None = None,
    topk: Sequence[int] = (100, 300, 1000),
    iou_thresholds: Sequence[float] = (0.3, 0.5, 0.7),
    group_by: str = "sample",
    limit_rows: int | None = None,
) -> dict[str, Any]:
    if not inputs:
        raise ValueError("at least one input is required")
    cases = []
    for label, path in inputs:
        rows = read_jsonl(path, limit=limit_rows)
        if not rows:
            raise ValueError(f"{label}: no joined proposal rows found")
        case = analyze_rows(rows, label=label, topk=topk, iou_thresholds=iou_thresholds, group_by=group_by)
        case["input_jsonl"] = str(path)
        cases.append(case)
    key_names = sorted({key for case in cases for key in case.get("aggregate", {}).keys()})
    aggregate = {
        key: _stats([case.get("aggregate", {}).get(key, {}).get("mean") for case in cases])
        for key in key_names
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "decision": READY,
        "case_count": int(len(cases)),
        "cases": cases,
        "aggregate_case_means": aggregate,
        "raw_joined_proposal_rows_available": True,
        "full_evaluator_facing_oracle_rerank_bound_available": False,
        "result_detection_geometry_close_out_available": False,
        "diagnostic_scope": "raw_joined_proposal_rows_only",
        "diagnostic_only": True,
        "uses_validation_gt": True,
        "uses_teacher": False,
        "uses_oracle_for_analysis_only": True,
        "uses_deploy_oracle": False,
        "metric_claim_allowed": False,
        "paper_claim_allowed": False,
    }
    if output_json is not None:
        write_json(output_json, payload)
    return payload


def _parse_labeled_input(value: str) -> tuple[str, str]:
    if "=" not in str(value):
        path = str(value)
        return Path(path).stem, path
    label, path = str(value).split("=", 1)
    if not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("--input entries must be LABEL=PATH or PATH")
    return label.strip(), path.strip()


def error_payload(exc: BaseException) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": NO_GO,
        "error_type": exc.__class__.__name__,
        "error": str(exc),
        "diagnostic_only": True,
        "metric_claim_allowed": False,
        "paper_claim_allowed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze P2 joined proposal rows for oracle-rerank close-out.")
    parser.add_argument("--input", action="append", type=_parse_labeled_input, required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--topk", default="100,300,1000")
    parser.add_argument("--iou-thresholds", default="0.3,0.5,0.7")
    parser.add_argument("--group-by", choices=("sample", "video"), default="sample")
    parser.add_argument("--limit-rows", type=int)
    args = parser.parse_args(argv)
    try:
        payload = build_closeout(
            args.input,
            output_json=args.output_json,
            topk=_parse_number_list(args.topk, as_int=True),
            iou_thresholds=_parse_number_list(args.iou_thresholds),
            group_by=args.group_by,
            limit_rows=args.limit_rows,
        )
    except Exception as exc:  # pragma: no cover - CLI guard
        print(json.dumps(strict_json_value(error_payload(exc)), sort_keys=True))
        return 1
    print(json.dumps(strict_json_value(payload), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
