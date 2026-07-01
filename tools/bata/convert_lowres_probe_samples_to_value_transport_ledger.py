from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.export_pc_ot_mras_hard_positions import strict_json_value, write_json  # noqa: E402


_BOUNDARY_ACQUISITION_PATH = ROOT / "opentad" / "datasets" / "transforms" / "boundary_acquisition.py"
_BOUNDARY_SPEC = importlib.util.spec_from_file_location(
    "pc_ot_mras_lowres_probe_boundary_acquisition",
    _BOUNDARY_ACQUISITION_PATH,
)
_BOUNDARY_MODULE = importlib.util.module_from_spec(_BOUNDARY_SPEC)
assert _BOUNDARY_SPEC.loader is not None
sys.modules[_BOUNDARY_SPEC.name] = _BOUNDARY_MODULE
_BOUNDARY_SPEC.loader.exec_module(_BOUNDARY_MODULE)
validate_value_transport_selection_row = _BOUNDARY_MODULE.validate_value_transport_selection_row


OUTPUT_SCHEMA_VERSION = "pc_ot_mras_frontend_value_transport_ledger_v0"
SUMMARY_SCHEMA_VERSION = "c3_lowres_probe_value_transport_ledger_summary_v0"
READY = "C3_LOWRES_PROBE_LEDGER_READY"
NO_GO = "C3_LOWRES_PROBE_LEDGER_NO_GO"
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
)
WINDOW_SAMPLE_ID_RE = re.compile(r"^[^|\s][^|]*\|(?:0|[1-9][0-9]*)$")
INTEGER_TEXT_RE = re.compile(r"^[+-]?(?:0|[1-9][0-9]*)$")


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).expanduser().open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise ValueError(f"line {line_no}: probe sample row must be a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"probe samples JSONL has no rows: {path}")
    return rows


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    out_path = Path(path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(strict_json_value(dict(row)), sort_keys=True) + "\n")


def _is_true(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    return False


def _strict_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if INTEGER_TEXT_RE.fullmatch(text):
            return int(text)
        raise ValueError(f"{name} must be an integer") from None
    raise ValueError(f"{name} must be an integer")


def _optional_int(row: Mapping[str, Any], key: str) -> int | None:
    if key not in row or row[key] is None:
        return None
    return _strict_int(row[key], name=key)


def _finite_float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _int_positions(value: Any, *, name: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON list")
    out: list[int] = []
    for idx, item in enumerate(value):
        position = _strict_int(item, name=f"{name}[{idx}]")
        if position < 0:
            raise ValueError(f"{name}[{idx}] must be non-negative")
        out.append(position)
    if out != sorted(out):
        raise ValueError(f"{name} must be sorted")
    if len(set(out)) != len(out):
        raise ValueError(f"{name} must be unique")
    return out


def _validate_sample_id(sample_id: str, *, line_no: int, require_window_sample_id: bool) -> None:
    if require_window_sample_id:
        if not WINDOW_SAMPLE_ID_RE.fullmatch(sample_id):
            raise ValueError(
                f"line {line_no}: sample_id must match video_name|non_negative_window_start"
            )
        return
    if "|" in sample_id and not WINDOW_SAMPLE_ID_RE.fullmatch(sample_id):
        raise ValueError(
            f"line {line_no}: sample_id with a window separator must match video_name|non_negative_window_start"
        )


def _uniform_fill_positions(
    selected: Sequence[int],
    *,
    valid_len: int,
    target_count: int,
) -> tuple[list[int], int]:
    selected_set = {int(item) for item in selected if 0 <= int(item) < int(valid_len)}
    if target_count <= len(selected_set):
        return sorted(selected_set)[: int(target_count)], 0
    candidates = [idx for idx in range(max(int(valid_len), 0)) if idx not in selected_set]
    if not candidates:
        return sorted(selected_set), 0
    need = min(int(target_count) - len(selected_set), len(candidates))
    if need <= 0:
        return sorted(selected_set), 0
    if need == 1:
        fill = [candidates[len(candidates) // 2]]
    else:
        fill = []
        last = len(candidates) - 1
        for rank in range(need):
            fill.append(candidates[round(rank * last / float(need - 1))])
    return sorted(selected_set.union(fill)), len(fill)


def _expected_required_count(
    *,
    require_selected_count: int | None,
    valid_len: int,
    dense_len: int | None,
    allow_short_valid_ratio_count: bool,
) -> int | None:
    if require_selected_count is None:
        return None
    required = int(require_selected_count)
    if not allow_short_valid_ratio_count:
        return required
    if dense_len is None or int(dense_len) <= 0:
        return min(required, int(valid_len))
    if int(valid_len) >= int(dense_len):
        return required
    ratio_required = int(math.ceil(float(valid_len) * float(required) / float(dense_len)))
    return max(1, min(required, int(valid_len), ratio_required))


def _selected_positions_from_sample(
    row: Mapping[str, Any],
    *,
    line_no: int,
    strategy: str,
    fallback_to_selected_positions: bool,
) -> list[int]:
    strategy_rows = row.get("strategy_selected_positions")
    if isinstance(strategy_rows, Mapping) and strategy in strategy_rows:
        return _int_positions(strategy_rows[strategy], name=f"line {line_no}: strategy_selected_positions.{strategy}")
    if fallback_to_selected_positions and "selected_positions" in row:
        return _int_positions(row["selected_positions"], name=f"line {line_no}: selected_positions")
    raise ValueError(f"line {line_no}: strategy '{strategy}' is missing from strategy_selected_positions")


def sample_row_to_value_transport_row(
    row: Mapping[str, Any],
    *,
    line_no: int,
    strategy: str,
    target_len: int,
    require_selected_count: int | None = None,
    fill_to_target_count: bool = False,
    allow_short_valid_ratio_count: bool = False,
    fallback_to_selected_positions: bool = False,
    require_window_sample_id: bool = True,
    deploy_selection_ledger: bool = False,
    route_variant: str = "c3_lowres_probe_delta_p_action_original_adatad",
) -> dict[str, Any]:
    sample_id = row.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError(f"line {line_no}: sample_id must be a non-empty string")
    _validate_sample_id(sample_id, line_no=line_no, require_window_sample_id=bool(require_window_sample_id))
    if deploy_selection_ledger and not require_window_sample_id:
        raise ValueError("deploy_selection_ledger requires strict video_name|window_start sample_id keys")
    for key in FORBIDDEN_TRUE_FLAGS:
        if _is_true(row.get(key, False)):
            raise ValueError(f"line {line_no}: forbidden source flag {key}=true")

    dense_len = _optional_int(row, "dense_len")
    valid_len = _optional_int(row, "valid_len")
    if valid_len is None:
        valid_len = dense_len
    if valid_len is None or valid_len <= 0:
        raise ValueError(f"line {line_no}: valid_len or dense_len must be positive")
    if dense_len is not None and valid_len > dense_len:
        raise ValueError(f"line {line_no}: valid_len cannot exceed dense_len")

    selected = _selected_positions_from_sample(
        row,
        line_no=line_no,
        strategy=strategy,
        fallback_to_selected_positions=bool(fallback_to_selected_positions),
    )
    if any(position >= valid_len for position in selected):
        raise ValueError(f"line {line_no}: selected_positions exceed valid_len={valid_len}")
    expected_required_count = _expected_required_count(
        require_selected_count=require_selected_count,
        valid_len=int(valid_len),
        dense_len=dense_len,
        allow_short_valid_ratio_count=bool(allow_short_valid_ratio_count),
    )
    fill_count = 0
    if fill_to_target_count and expected_required_count is not None and len(selected) < int(expected_required_count):
        selected, fill_count = _uniform_fill_positions(
            selected,
            valid_len=int(valid_len),
            target_count=int(expected_required_count),
        )
    if expected_required_count is not None and len(selected) != int(expected_required_count):
        raise ValueError(
            f"line {line_no}: selected_count={len(selected)} does not match required count {int(expected_required_count)}"
        )

    diagnostics = {
        "source_probe_model": row.get("probe_model"),
        "source_tcn_variant": row.get("tcn_variant"),
        "source_spatial_size": row.get("spatial_size"),
        "source_strategy": str(strategy),
        "source_budget": row.get("budget"),
        "source_selected_count": len(_selected_positions_from_sample(
            row,
            line_no=line_no,
            strategy=strategy,
            fallback_to_selected_positions=bool(fallback_to_selected_positions),
        )),
        "uniform_visible_fill_count": int(fill_count),
        "required_selected_count": expected_required_count,
        "allow_short_valid_ratio_count": bool(allow_short_valid_ratio_count),
        "fallback_to_selected_positions": bool(
            fallback_to_selected_positions
            and not (isinstance(row.get("strategy_selected_positions"), Mapping) and strategy in row["strategy_selected_positions"])
        ),
    }
    boundary_support = None if deploy_selection_ledger else _finite_float_or_none(row.get("boundary_support_r1"))
    if boundary_support is not None:
        diagnostics["diagnostic_boundary_support_r1_ignored_by_selection"] = boundary_support

    ledger_row = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "sample_id": sample_id,
        "selected_positions_unit": "local_dense_index",
        "selected_positions": selected,
        "target_len": int(target_len),
        "selected_count": len(selected),
        "valid_len": int(valid_len),
        "dense_len": dense_len,
        "route": "C3_LOWRES_PROBE",
        "route_variant": str(route_variant),
        "policy": f"c3_lowres_probe_{strategy}",
        "source_schema_version": "c3_lowres_probe_samples_jsonl",
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
    strategy: str = "delta_p_action",
    target_len: int = 384,
    summary_json: str | Path | None = None,
    require_selected_count: int | None = None,
    fill_to_target_count: bool = False,
    allow_short_valid_ratio_count: bool = False,
    fallback_to_selected_positions: bool = False,
    require_window_sample_id: bool = True,
    deploy_selection_ledger: bool = False,
    deduplicate_sample_id: bool = False,
    route_variant: str = "c3_lowres_probe_delta_p_action_original_adatad",
) -> dict[str, Any]:
    source_rows = _read_jsonl(input_jsonl)
    if deploy_selection_ledger and not require_window_sample_id:
        raise ValueError("deploy_selection_ledger requires strict video_name|window_start sample_id keys")
    out_rows = [
        sample_row_to_value_transport_row(
            row,
            line_no=line_no,
            strategy=strategy,
            target_len=target_len,
            require_selected_count=require_selected_count,
            fill_to_target_count=fill_to_target_count,
            allow_short_valid_ratio_count=bool(allow_short_valid_ratio_count),
            fallback_to_selected_positions=fallback_to_selected_positions,
            require_window_sample_id=require_window_sample_id,
            deploy_selection_ledger=deploy_selection_ledger,
            route_variant=route_variant,
        )
        for line_no, row in enumerate(source_rows, start=1)
    ]
    duplicate_sample_id_count = 0
    if deduplicate_sample_id:
        deduped_rows: list[dict[str, Any]] = []
        by_sample_id: dict[str, dict[str, Any]] = {}
        for row in out_rows:
            sample_id = str(row["sample_id"])
            previous = by_sample_id.get(sample_id)
            if previous is None:
                by_sample_id[sample_id] = row
                deduped_rows.append(row)
                continue
            duplicate_sample_id_count += 1
            for key in ("selected_positions", "selected_count", "target_len", "valid_len", "dense_len"):
                if previous.get(key) != row.get(key):
                    raise ValueError(f"duplicate sample_id {sample_id} has conflicting {key}")
        out_rows = deduped_rows
    sample_ids = [str(row["sample_id"]) for row in out_rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("converted value-transport ledger has duplicate sample_id")
    _write_jsonl(output_jsonl, out_rows)
    counts = [int(row["selected_count"]) for row in out_rows]
    fill_counts = [int(row["diagnostics"]["uniform_visible_fill_count"]) for row in out_rows]
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "decision": READY,
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "row_count": len(out_rows),
        "strategy": str(strategy),
        "target_len": int(target_len),
        "require_selected_count": require_selected_count,
        "allow_short_valid_ratio_count": bool(allow_short_valid_ratio_count),
        "fill_to_target_count": bool(fill_to_target_count),
        "deploy_selection_ledger": bool(deploy_selection_ledger),
        "deduplicate_sample_id": bool(deduplicate_sample_id),
        "duplicate_sample_id_count": int(duplicate_sample_id_count),
        "route_variant": str(route_variant),
        "min_selected_count": min(counts),
        "max_selected_count": max(counts),
        "total_uniform_visible_fill_count": sum(fill_counts),
        "sample_ids_with_window_start": sum(1 for sample_id in sample_ids if "|" in sample_id),
    }
    if summary_json is not None:
        write_json(summary_json, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert C3 low-res probe samples to value-transport ledger rows.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json")
    parser.add_argument("--strategy", default="delta_p_action")
    parser.add_argument("--target-len", type=int, default=384)
    parser.add_argument("--require-selected-count", type=int)
    parser.add_argument("--fill-to-target-count", action="store_true")
    parser.add_argument("--allow-short-valid-ratio-count", action="store_true")
    parser.add_argument("--fallback-to-selected-positions", action="store_true")
    parser.add_argument("--allow-video-only-sample-id", action="store_true")
    parser.add_argument("--deploy-selection-ledger", action="store_true")
    parser.add_argument("--deduplicate-sample-id", action="store_true")
    parser.add_argument("--route-variant", default="c3_lowres_probe_delta_p_action_original_adatad")
    args = parser.parse_args(argv)

    try:
        if args.allow_video_only_sample_id and args.deploy_selection_ledger:
            raise ValueError("--allow-video-only-sample-id is diagnostic-only and cannot be used with --deploy-selection-ledger")
        summary = run_conversion(
            args.input_jsonl,
            args.output_jsonl,
            strategy=args.strategy,
            target_len=int(args.target_len),
            summary_json=args.summary_json,
            require_selected_count=args.require_selected_count,
            fill_to_target_count=bool(args.fill_to_target_count),
            allow_short_valid_ratio_count=bool(args.allow_short_valid_ratio_count),
            fallback_to_selected_positions=bool(args.fallback_to_selected_positions),
            require_window_sample_id=not bool(args.allow_video_only_sample_id),
            deploy_selection_ledger=bool(args.deploy_selection_ledger),
            deduplicate_sample_id=bool(args.deduplicate_sample_id),
            route_variant=args.route_variant,
        )
    except Exception as exc:  # pragma: no cover - CLI guard
        print(json.dumps({"schema_version": SUMMARY_SCHEMA_VERSION, "decision": NO_GO, "error": str(exc)}))
        return 1
    print(json.dumps(strict_json_value(summary), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
