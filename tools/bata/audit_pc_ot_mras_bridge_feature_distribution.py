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


from tools.bata.dump_pc_ot_mras_reader_bridge_diagnostics import (  # noqa: E402
    BRIDGE_OUTPUT_KEYS,
    SELECTED_TOKEN_KEYS,
    SUMMARY_SCHEMA_VERSION as READER_BRIDGE_SUMMARY_SCHEMA_VERSION,
    read_jsonl,
    run_jsonl_diagnostic,
    strict_json_value,
    write_json,
)


SCHEMA_VERSION = "pc_ot_mras_bridge_feature_distribution_audit_v0"
READY = "PC_OT_MRAS_BRIDGE_FEATURE_DISTRIBUTION_AUDIT_READY"
EVIDENCE_INSUFFICIENT = "PC_OT_MRAS_BRIDGE_FEATURE_DISTRIBUTION_EVIDENCE_INSUFFICIENT"
NO_GO = "PC_OT_MRAS_BRIDGE_FEATURE_DISTRIBUTION_AUDIT_NO_GO"


def _stat_count(stats: Mapping[str, Any] | None) -> int:
    if not isinstance(stats, Mapping):
        return 0
    value = stats.get("count", 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _mean_ratio(numerator: Mapping[str, Any] | None, denominator: Mapping[str, Any] | None) -> float | None:
    if not isinstance(numerator, Mapping) or not isinstance(denominator, Mapping):
        return None
    num = _finite_float(numerator.get("mean"))
    den = _finite_float(denominator.get("mean"))
    if num is None or den is None or abs(den) <= 1.0e-12:
        return None
    return float(num / den)


def _row_reader_out(row: Mapping[str, Any]) -> Mapping[str, Any]:
    reader_out = row.get("reader_out", row)
    return reader_out if isinstance(reader_out, Mapping) else {}


def _row_has_selected_tokens(row: Mapping[str, Any]) -> bool:
    reader_out = _row_reader_out(row)
    return any(key in reader_out for key in SELECTED_TOKEN_KEYS)


def _row_has_bridge_output(row: Mapping[str, Any]) -> bool:
    if "bridge_output_norm_hook" in row:
        return True
    if any(key in row for key in BRIDGE_OUTPUT_KEYS):
        return True
    bridge_meta = row.get("pc_ot_mras_bridge")
    if isinstance(bridge_meta, Mapping) and any(key in bridge_meta for key in ("output", "features", "selected_tokens")):
        return True
    return False


def _visible_key_manifest(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    reader_keys: set[str] = set()
    top_level_keys: set[str] = set()
    snapshot_ids: set[str] = set()
    sample_preview: list[str] = []
    selected_rows = 0
    bridge_rows = 0
    for row in rows:
        top_level_keys.update(str(key) for key in row.keys())
        reader_out = _row_reader_out(row)
        reader_keys.update(str(key) for key in reader_out.keys())
        snapshot_id = row.get("snapshot_id")
        if snapshot_id is not None:
            snapshot_ids.add(str(snapshot_id))
        sample_ids = row.get("sample_ids")
        if isinstance(sample_ids, list):
            for sample_id in sample_ids:
                if len(sample_preview) < 8:
                    sample_preview.append(str(sample_id))
        elif row.get("sample_id") is not None and len(sample_preview) < 8:
            sample_preview.append(str(row.get("sample_id")))
        if _row_has_selected_tokens(row):
            selected_rows += 1
        if _row_has_bridge_output(row):
            bridge_rows += 1
    return {
        "row_count": int(len(rows)),
        "snapshot_ids": sorted(snapshot_ids),
        "sample_ids_preview": sample_preview,
        "top_level_keys": sorted(top_level_keys),
        "reader_out_keys": sorted(reader_keys),
        "rows_with_selected_token_values": int(selected_rows),
        "rows_with_bridge_output_values": int(bridge_rows),
        "selected_token_candidate_keys": list(SELECTED_TOKEN_KEYS),
        "bridge_output_candidate_keys": list(BRIDGE_OUTPUT_KEYS),
    }


def _run_summary(input_jsonl: str | Path, *, limit: int | None = None) -> dict[str, Any]:
    summary = run_jsonl_diagnostic(input_jsonl, limit=limit)
    if summary.get("schema_version") != READER_BRIDGE_SUMMARY_SCHEMA_VERSION:
        raise ValueError("unexpected reader/bridge summary schema")
    return summary


def _audit_one(label: str, input_jsonl: str | Path, *, limit: int | None = None) -> dict[str, Any]:
    rows = read_jsonl(input_jsonl)
    if limit is not None:
        rows = rows[: int(limit)]
    summary = _run_summary(input_jsonl, limit=limit)
    aggregate = summary.get("aggregate")
    if not isinstance(aggregate, Mapping):
        raise ValueError(f"{label}: reader/bridge diagnostic summary missing aggregate")
    selected_stats = aggregate.get("selected_token_norm")
    bridge_stats = aggregate.get("bridge_output_norm")
    selected_count = _stat_count(selected_stats)
    bridge_count = _stat_count(bridge_stats)
    missing_reasons = []
    if selected_count <= 0:
        missing_reasons.append("selected_token_values_not_visible_in_jsonl")
    if bridge_count <= 0:
        missing_reasons.append("bridge_output_values_not_visible_in_jsonl")
    return {
        "label": str(label),
        "input_jsonl": str(input_jsonl),
        "reader_bridge_summary_schema_version": str(summary.get("schema_version")),
        "sample_count": int(summary.get("sample_count", 0)),
        "matrix_keys": list(summary.get("matrix_keys", [])),
        "visible_key_manifest": _visible_key_manifest(rows),
        "feature_distribution_evidence_ready": bool(selected_count > 0 and bridge_count > 0),
        "missing_feature_evidence": missing_reasons,
        "selected_token_norm": selected_stats,
        "bridge_output_norm": bridge_stats,
        "bridge_to_selected_token_norm_mean_ratio": _mean_ratio(bridge_stats, selected_stats),
        "reader_temporal_context": {
            "centers_selected_times_abs_offset": aggregate.get("centers_selected_times_abs_offset"),
            "selected_time_delta": aggregate.get("selected_time_delta"),
            "selected_times_nondecreasing_failure_count": aggregate.get("selected_times_nondecreasing_failure_count"),
            "gate": aggregate.get("gate"),
            "acquisition_entropy": aggregate.get("acquisition_entropy"),
            "acquisition_normalized_entropy": aggregate.get("acquisition_normalized_entropy"),
            "acquisition_top1_center_distance_dense": aggregate.get("acquisition_top1_center_distance_dense"),
        },
        "diagnostic_only": True,
        "uses_gt": False,
        "uses_teacher": False,
        "uses_oracle": False,
        "uses_cache": False,
        "uses_raw_prediction": False,
        "metric_claim_allowed": False,
        "paper_claim_allowed": False,
        "runtime_flops_claim_allowed": False,
        "deploy_claim_allowed": False,
        "tools_train_allowed": False,
        "tools_test_allowed": False,
        "slurm_gpu_allowed": False,
    }


def build_audit(
    inputs: Sequence[tuple[str, str | Path]],
    *,
    output_json: str | Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    if not inputs:
        raise ValueError("at least one input JSONL is required")
    if limit is not None and int(limit) <= 0:
        raise ValueError("limit must be positive when provided")
    runs = [_audit_one(label, path, limit=limit) for label, path in inputs]
    ready_runs = [run for run in runs if run["feature_distribution_evidence_ready"]]
    decision = READY if len(ready_runs) == len(runs) else EVIDENCE_INSUFFICIENT
    missing_by_label = {
        str(run["label"]): list(run["missing_feature_evidence"])
        for run in runs
        if run["missing_feature_evidence"]
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "run_count": int(len(runs)),
        "feature_distribution_ready_count": int(len(ready_runs)),
        "missing_feature_evidence_by_label": missing_by_label,
        "runs": runs,
        "conclusion": (
            "selected-token and bridge-output feature distributions are directly auditable"
            if decision == READY
            else "existing JSONL does not expose enough selected-token/bridge-output feature values for a direct feature-distribution verdict"
        ),
        "required_next_evidence": (
            []
            if decision == READY
            else [
                "one bounded diagnostic JSONL row set containing selected_tokens or selected_token_features",
                "one bounded diagnostic JSONL row set containing bridge_output, bridge_features, or bridge_output_norm_hook",
            ]
        ),
        "diagnostic_only": True,
        "uses_gt": False,
        "uses_teacher": False,
        "uses_oracle": False,
        "uses_cache": False,
        "uses_raw_prediction": False,
        "metric_claim_allowed": False,
        "paper_claim_allowed": False,
        "runtime_flops_claim_allowed": False,
        "deploy_claim_allowed": False,
        "tools_train_allowed": False,
        "tools_test_allowed": False,
        "slurm_gpu_allowed": False,
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
    parser = argparse.ArgumentParser(description="Audit visible PC-OT-MRAS selected-token/bridge feature distributions from JSONL.")
    parser.add_argument("--input", action="append", type=_parse_labeled_input, required=True, help="LABEL=PATH or PATH")
    parser.add_argument("--output-json")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)

    try:
        payload = build_audit(args.input, output_json=args.output_json, limit=args.limit)
    except Exception as exc:
        print(json.dumps(strict_json_value(error_payload(exc)), indent=2, sort_keys=True))
        return 1
    print(json.dumps(strict_json_value(payload), indent=2, sort_keys=True))
    return 0 if payload["decision"] in {READY, EVIDENCE_INSUFFICIENT} else 1


if __name__ == "__main__":
    raise SystemExit(main())
