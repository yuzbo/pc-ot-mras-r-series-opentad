from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.export_pc_ot_mras_hard_positions import (  # noqa: E402
    pc_ot_mras_hard_rows_to_temporal_metas,
    resolve_pc_ot_mras_dynamic_budget_plan,
    strict_json_value,
    write_json,
)


SCHEMA_VERSION = "pc_ot_mras_dynamic_budget_pipeline_validation_v0"
READY = "PC_OT_MRAS_DYNAMIC_BUDGET_PIPELINE_READY"
NO_GO = "PC_OT_MRAS_DYNAMIC_BUDGET_PIPELINE_NO_GO"


def _load_repo_function(rel_path: str, function_name: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(f"pc_ot_mras_r25_{function_name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return getattr(module, function_name)


def _import_torch():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment guard
        raise RuntimeError("R25 dynamic-budget pipeline validation requires torch") from exc
    return torch


def _selected_prefix_mask(rows: Sequence[Mapping[str, Any]]):
    torch = _import_torch()
    budgets = [int(row["budget"]) for row in rows]
    if not budgets:
        raise ValueError("rows must be non-empty")
    max_budget = max(budgets)
    mask = torch.zeros((len(budgets), max_budget), dtype=torch.bool)
    for batch_idx, budget in enumerate(budgets):
        if budget <= 0:
            raise ValueError("all dynamic budgets must be positive")
        mask[batch_idx, :budget] = True
    return mask


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / float(len(values)))


def validate_pc_ot_mras_dynamic_budget_pipeline(
    dynamic_plan: Mapping[str, Any],
    *,
    sample_ids: Sequence[str] | None = None,
    split: str = "val",
    strict_temporal_grid: bool = True,
) -> dict[str, Any]:
    """Validate the local R22 -> R23 -> R24 dynamic-budget protocol chain.

    This is a synthetic/protocol validator. It does not run a detector, read a
    dataset, load checkpoints, validate scanner quality, measure runtime/FLOPs,
    or create metric/paper evidence.
    """

    if not isinstance(dynamic_plan, Mapping):
        raise ValueError("dynamic_plan must be a mapping")

    validate_sampling_contract = _load_repo_function(
        "opentad/models/utils/sampling_contract.py",
        "validate_sampling_contract",
    )
    temporal_grid_from_metas = _load_repo_function(
        "opentad/models/utils/temporal_grid.py",
        "temporal_grid_from_metas",
    )

    rows = resolve_pc_ot_mras_dynamic_budget_plan(dynamic_plan, sample_ids=sample_ids)
    metas = pc_ot_mras_hard_rows_to_temporal_metas(rows)
    selected_mask = _selected_prefix_mask(rows)

    validate_sampling_contract(metas, selected_mask, split=split)
    grid = temporal_grid_from_metas(
        metas,
        selected_mask,
        required=True,
        strict=bool(strict_temporal_grid),
    )

    budgets = [int(row["budget"]) for row in rows]
    dense_valid_lens = [int(row["valid_len"]) for row in rows]
    selected_counts = [int(meta["irregular_selected_count"]) for meta in metas]
    coverage_shares = [
        _float_or_none(row.get("dynamic_budget_plan", {}).get("coverage_share"))
        for row in rows
    ]
    coverage_shares = [item for item in coverage_shares if item is not None]

    torch = _import_torch()
    grid_counts = grid["valid_mask"].long().sum(dim=1).detach().cpu().tolist()
    center_matches_rows = True
    for batch_idx, row in enumerate(rows):
        budget = int(row["budget"])
        expected = torch.tensor(row["selected_positions"], dtype=torch.float32)
        actual = grid["center"][batch_idx, :budget].detach().cpu()
        if not torch.allclose(actual, expected, atol=1.0e-6, rtol=1.0e-6):
            center_matches_rows = False
            break

    exact_budget_violations = sum(
        1
        for budget, selected_count, grid_count in zip(budgets, selected_counts, grid_counts)
        if int(budget) != int(selected_count) or int(budget) != int(grid_count)
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "decision": READY,
        "row_count": len(rows),
        "sample_ids": [str(row["sample_id"]) for row in rows],
        "budgets": budgets,
        "dense_valid_lens": dense_valid_lens,
        "selected_counts": selected_counts,
        "grid_valid_counts": [int(item) for item in grid_counts],
        "budget_min": min(budgets),
        "budget_max": max(budgets),
        "budget_mean": _mean([float(item) for item in budgets]),
        "coverage_share_mean": _mean(coverage_shares),
        "coverage_share_max": max(coverage_shares) if coverage_shares else None,
        "exact_budget_violations": int(exact_budget_violations),
        "center_matches_rows": bool(center_matches_rows),
        "sampling_contract_passed": True,
        "temporal_grid_passed": True,
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


def run_json_validation(
    input_json: str | Path,
    *,
    summary_json: str | Path | None = None,
    sample_ids: Sequence[str] | None = None,
    split: str = "val",
    strict_temporal_grid: bool = True,
) -> dict[str, Any]:
    summary = validate_pc_ot_mras_dynamic_budget_pipeline(
        read_json(input_json),
        sample_ids=sample_ids,
        split=split,
        strict_temporal_grid=strict_temporal_grid,
    )
    if summary_json is not None:
        write_json(summary_json, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the local PC-OT-MRAS R22->R23->R24 dynamic-budget protocol pipeline."
    )
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--summary-json")
    parser.add_argument("--sample-id", action="append", dest="sample_ids")
    parser.add_argument("--split", default="val")
    parser.add_argument("--non-strict-temporal-grid", action="store_true")
    args = parser.parse_args(argv)

    try:
        summary = run_json_validation(
            args.input_json,
            summary_json=args.summary_json,
            sample_ids=args.sample_ids,
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
