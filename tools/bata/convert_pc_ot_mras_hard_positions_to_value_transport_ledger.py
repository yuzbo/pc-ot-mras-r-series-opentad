from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.export_pc_ot_mras_hard_positions import strict_json_value, write_json  # noqa: E402


_BOUNDARY_ACQUISITION_PATH = ROOT / "opentad" / "datasets" / "transforms" / "boundary_acquisition.py"
_BOUNDARY_SPEC = importlib.util.spec_from_file_location(
    "pc_ot_mras_frontend_boundary_acquisition",
    _BOUNDARY_ACQUISITION_PATH,
)
_BOUNDARY_MODULE = importlib.util.module_from_spec(_BOUNDARY_SPEC)
assert _BOUNDARY_SPEC.loader is not None
sys.modules[_BOUNDARY_SPEC.name] = _BOUNDARY_MODULE
_BOUNDARY_SPEC.loader.exec_module(_BOUNDARY_MODULE)
validate_value_transport_selection_row = _BOUNDARY_MODULE.validate_value_transport_selection_row


INPUT_SCHEMA_VERSION = "pc_ot_mras_hard_positions_v0"
OUTPUT_SCHEMA_VERSION = "pc_ot_mras_frontend_value_transport_ledger_v0"
SUMMARY_SCHEMA_VERSION = "pc_ot_mras_frontend_value_transport_ledger_summary_v0"
READY = "PC_OT_MRAS_FRONTEND_LEDGER_READY"
NO_GO = "PC_OT_MRAS_FRONTEND_LEDGER_NO_GO"
FORBIDDEN_TRUE_FLAGS = (
    "uses_gt",
    "uses_teacher",
    "uses_oracle",
    "uses_cache",
    "uses_prediction_cache",
    "uses_raw_prediction",
    "uses_checkpoint",
    "prediction_uses_gt",
    "training_only",
    "diagnostic_uses_train_utility_for_audit",
)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).expanduser().open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_no}: hard-position row must be a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"hard-position JSONL has no rows: {path}")
    return rows


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(strict_json_value(dict(row)), sort_keys=True) + "\n")


def _is_true(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    return False


def _optional_int(row: Mapping[str, Any], key: str) -> int | None:
    if key not in row or row[key] is None:
        return None
    value = row[key]
    if isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be an integer") from None


def _finite_float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _selected_positions(row: Mapping[str, Any], *, line_no: int) -> list[int]:
    raw = row.get("selected_positions")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"line {line_no}: selected_positions must be a non-empty JSON list")
    positions: list[int] = []
    for idx, value in enumerate(raw):
        if isinstance(value, bool):
            raise ValueError(f"line {line_no}: selected_positions[{idx}] must be an integer")
        try:
            positions.append(int(value))
        except (TypeError, ValueError):
            raise ValueError(f"line {line_no}: selected_positions[{idx}] must be an integer") from None
    if any(pos < 0 for pos in positions):
        raise ValueError(f"line {line_no}: selected_positions must be non-negative")
    if len(set(positions)) != len(positions):
        raise ValueError(f"line {line_no}: selected_positions must be unique")
    if positions != sorted(positions):
        raise ValueError(f"line {line_no}: selected_positions must be sorted ascending")
    return positions


def hard_row_to_value_transport_row(
    row: Mapping[str, Any],
    *,
    line_no: int,
    target_len: int,
    require_selected_count: int | None = None,
    deploy_selection_ledger: bool = False,
    route_variant: str = "pc_ot_mras_frontend_original_adatad_eval",
) -> dict[str, Any]:
    if row.get("schema_version") != INPUT_SCHEMA_VERSION:
        raise ValueError(f"line {line_no}: schema_version must be {INPUT_SCHEMA_VERSION}")
    sample_id = row.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError(f"line {line_no}: sample_id must be a non-empty string")
    for key in FORBIDDEN_TRUE_FLAGS:
        if _is_true(row.get(key, False)):
            raise ValueError(f"line {line_no}: forbidden source flag {key}=true")

    positions = _selected_positions(row, line_no=line_no)
    if require_selected_count is not None and len(positions) != int(require_selected_count):
        raise ValueError(
            f"line {line_no}: selected_count={len(positions)} does not match "
            f"required count {int(require_selected_count)}"
        )
    valid_len = _optional_int(row, "valid_len")
    dense_len = _optional_int(row, "dense_len")
    if valid_len is not None and any(pos >= valid_len for pos in positions):
        raise ValueError(f"line {line_no}: selected_positions exceed valid_len={valid_len}")
    if dense_len is not None and any(pos >= dense_len for pos in positions):
        raise ValueError(f"line {line_no}: selected_positions exceed dense_len={dense_len}")

    diagnostics = {
        "selected_count": len(positions),
        "duplicate_repair_count": int(row.get("duplicate_repair_count", 0)),
        "invalid_repair_count": int(row.get("invalid_repair_count", 0)),
        "repair_fill_count": int(row.get("repair_fill_count", 0)),
    }
    soft_error = _finite_float_or_none(row.get("soft_hard_time_error"))
    if soft_error is not None:
        diagnostics["soft_hard_time_error"] = soft_error

    ledger_row = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "sample_id": sample_id,
        "selected_positions_unit": "local_dense_index",
        "selected_positions": positions,
        "target_len": int(target_len),
        "selected_count": len(positions),
        "valid_len": valid_len,
        "dense_len": dense_len,
        "route": "PC-OT-MRAS",
        "route_variant": str(route_variant),
        "policy": "pc_ot_mras_hard_export_fixed_budget",
        "source_schema_version": INPUT_SCHEMA_VERSION,
        "source_budget": int(row.get("budget", len(positions))),
        "source_batch_index": int(row.get("batch_index", 0)),
        "role_round_metadata": row.get("role_round_metadata", []),
        "diagnostics": diagnostics,
        "deploy_selection_ledger": bool(deploy_selection_ledger),
        "diagnostic_only": not bool(deploy_selection_ledger),
        "training_only": False,
        "diagnostic_uses_train_utility_for_audit": False,
        "uses_gt": False,
        "uses_teacher": False,
        "uses_oracle": False,
        "uses_cache": False,
        "uses_prediction_cache": False,
        "uses_raw_prediction": False,
        "uses_checkpoint": False,
        "prediction_uses_gt": False,
    }
    validate_value_transport_selection_row(
        ledger_row,
        line_no=line_no,
        require_deployable=bool(deploy_selection_ledger),
    )
    return ledger_row


def run_conversion(
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    target_len: int,
    summary_json: str | Path | None = None,
    require_selected_count: int | None = None,
    deploy_selection_ledger: bool = False,
    route_variant: str = "pc_ot_mras_frontend_original_adatad_eval",
) -> dict[str, Any]:
    source_rows = _read_jsonl(input_jsonl)
    out_rows = [
        hard_row_to_value_transport_row(
            row,
            line_no=line_no,
            target_len=int(target_len),
            require_selected_count=require_selected_count,
            deploy_selection_ledger=bool(deploy_selection_ledger),
            route_variant=route_variant,
        )
        for line_no, row in enumerate(source_rows, start=1)
    ]
    sample_ids = [str(row["sample_id"]) for row in out_rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("converted value-transport ledger has duplicate sample_id")

    _write_jsonl(output_jsonl, out_rows)
    counts = [int(row["selected_count"]) for row in out_rows]
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "decision": READY,
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "row_count": len(out_rows),
        "target_len": int(target_len),
        "deploy_selection_ledger": bool(deploy_selection_ledger),
        "route_variant": str(route_variant),
        "min_selected_count": min(counts),
        "max_selected_count": max(counts),
        "sample_ids_with_window_start": sum(1 for sample_id in sample_ids if "|" in sample_id),
        "total_duplicate_repair_count": sum(int(row["diagnostics"]["duplicate_repair_count"]) for row in out_rows),
        "total_repair_fill_count": sum(int(row["diagnostics"]["repair_fill_count"]) for row in out_rows),
    }
    if summary_json is not None:
        write_json(summary_json, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert PC-OT-MRAS hard positions to value-transport ledger rows.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json")
    parser.add_argument("--target-len", type=int, default=384)
    parser.add_argument("--require-selected-count", type=int)
    parser.add_argument("--deploy-selection-ledger", action="store_true")
    parser.add_argument("--route-variant", default="pc_ot_mras_frontend_original_adatad_eval")
    args = parser.parse_args(argv)

    try:
        summary = run_conversion(
            args.input_jsonl,
            args.output_jsonl,
            target_len=int(args.target_len),
            summary_json=args.summary_json,
            require_selected_count=args.require_selected_count,
            deploy_selection_ledger=bool(args.deploy_selection_ledger),
            route_variant=args.route_variant,
        )
    except Exception as exc:  # pragma: no cover - CLI guard
        print(json.dumps({"schema_version": SUMMARY_SCHEMA_VERSION, "decision": NO_GO, "error": str(exc)}))
        return 1

    print(json.dumps(strict_json_value(summary), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
