from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.audit_pc_ot_mras_tubelet_packed_profile import (  # noqa: E402
    _expand_tubelet_mask,
    _pack_and_scatter,
)
from tools.bata.export_pc_ot_mras_hard_positions import (  # noqa: E402
    resolve_pc_ot_mras_dynamic_budget_plan,
    strict_json_value,
    write_json,
)


SCHEMA_VERSION = "pc_ot_mras_dynamic_tubelet_contract_audit_v0"
READY = "PC_OT_MRAS_DYNAMIC_TUBELET_CONTRACT_AUDIT_READY"
NO_GO = "PC_OT_MRAS_DYNAMIC_TUBELET_CONTRACT_AUDIT_NO_GO"


def _torch():
    import importlib

    return importlib.import_module("torch")


def build_synthetic_dynamic_tubelet_plan() -> dict[str, Any]:
    """Build a deploy-visible variable-budget plan for local contract audits."""

    return {
        "schema_version": "pc_ot_mras_dynamic_budget_plan_v0",
        "controller_family": "synthetic_dynamic_tubelet_contract_audit",
        "uses_gt": False,
        "uses_teacher": False,
        "uses_cache": False,
        "uses_raw_prediction": False,
        "uses_checkpoint": False,
        "dynamic_budget_validation": False,
        "metric_claim_allowed": False,
        "paper_claim_allowed": False,
        "budgets": [4, 5, 6],
        "dense_valid_len": [16, 16, 16],
        "selected_dense_positions": [
            [0, 1, 8, 9, 0, 0],
            [2, 3, 10, 11, 12, 0],
            [4, 5, 6, 12, 13, 14],
        ],
        "selected_mask": [
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1],
        ],
        "budget_scores": [0.15, 0.55, 0.85],
        "coverage_counts": [2, 3, 4],
        "value_counts": [3, 4, 5],
        "coverage_share": [0.25, 0.375, 0.5],
    }


def _positive_int(value: int, *, name: str) -> int:
    out = int(value)
    if out <= 0:
        raise ValueError(f"{name} must be positive")
    return out


def _position_to_tubelet_ids(
    positions: Sequence[int],
    *,
    dense_valid_len: int,
    temporal_tubelets: int,
) -> list[int]:
    valid_len = _positive_int(dense_valid_len, name="dense_valid_len")
    tubelets = _positive_int(temporal_tubelets, name="temporal_tubelets")
    ids: list[int] = []
    for idx, pos in enumerate(positions):
        item = int(pos)
        if item < 0 or item >= valid_len:
            raise ValueError(f"selected_positions[{idx}] must stay inside dense_valid_len")
        ids.append(min(tubelets - 1, (item * tubelets) // valid_len))
    return ids


def _row_to_tubelet_sample(
    row: Mapping[str, Any],
    *,
    row_idx: int,
    temporal_tubelets: int,
    spatial_tokens: int,
) -> dict[str, Any]:
    selected = [int(pos) for pos in row["selected_positions"]]
    tubelet_ids = _position_to_tubelet_ids(
        selected,
        dense_valid_len=int(row["valid_len"]),
        temporal_tubelets=int(temporal_tubelets),
    )
    tubelet_keep_mask = [idx in set(tubelet_ids) for idx in range(int(temporal_tubelets))]
    selected_tubelet_count = int(sum(tubelet_keep_mask))
    dense_token_mask_true_count = selected_tubelet_count * int(spatial_tokens)
    full_spatial_groups_selected = dense_token_mask_true_count % int(spatial_tokens) == 0
    selected_mask_positions = [
        idx for idx, item in enumerate(row["selected_mask"]) if bool(item)
    ]

    return {
        "sample_id": str(row.get("sample_id", f"sample_{row_idx}")),
        "row_index": int(row_idx),
        "budget": int(row["budget"]),
        "dense_valid_len": int(row["valid_len"]),
        "dense_len": int(row["dense_len"]),
        "selected_positions": selected,
        "selected_tubelet_ids": tubelet_ids,
        "selected_unique_tubelet_ids": sorted(set(tubelet_ids)),
        "selected_tubelet_count": selected_tubelet_count,
        "tubelet_keep_mask": tubelet_keep_mask,
        "spatial_tokens_per_tubelet": int(spatial_tokens),
        "dense_token_mask_true_count": dense_token_mask_true_count,
        "full_spatial_groups_selected": full_spatial_groups_selected,
        "hard_row_selected_mask_consistent": selected_mask_positions == selected,
    }


def _build_bucket_summary(
    samples: Sequence[Mapping[str, Any]],
    *,
    selected_tubelet_count: int,
    temporal_tubelets: int,
    spatial_tokens: int,
    channels: int,
) -> dict[str, Any]:
    torch = _torch()
    tubelet_mask = torch.tensor(
        [sample["tubelet_keep_mask"] for sample in samples],
        dtype=torch.bool,
    )
    dense_mask = _expand_tubelet_mask(tubelet_mask, spatial_tokens=int(spatial_tokens))
    batch_size = int(len(samples))
    dense_tokens = int(temporal_tubelets) * int(spatial_tokens)
    token_values = torch.arange(batch_size * dense_tokens * int(channels), dtype=torch.float32)
    tokens = token_values.reshape(batch_size, dense_tokens, int(channels))
    pack = _pack_and_scatter(tokens, dense_mask)

    return {
        "selected_tubelet_count": int(selected_tubelet_count),
        "sample_indices": [int(sample["row_index"]) for sample in samples],
        "sample_ids": [str(sample["sample_id"]) for sample in samples],
        "batch_size": batch_size,
        "rectangular_pack_ready": True,
        "dense_mask_shape": list(dense_mask.shape),
        "packed_token_shape": list(pack["packed"].shape),
        "scatter_shape": list(pack["scattered"].shape),
        "selected_values_preserved": bool(pack["selected_values_preserved"]),
        "unselected_positions_zero": bool(pack["unselected_positions_zero"]),
    }


def audit_pc_ot_mras_dynamic_tubelet_contract(
    dynamic_plan: Mapping[str, Any] | None = None,
    *,
    temporal_tubelets: int = 8,
    spatial_h: int = 2,
    spatial_w: int = 3,
    channels: int = 4,
    bucketed_pack_fallback_allowed: bool = True,
) -> dict[str, Any]:
    """Audit the R22/R23 dynamic-position contract for packed tubelet routing.

    The audit consumes only deploy-visible dynamic-budget plan fields and the
    existing hard-position resolver. It does not run detector training, detector
    evaluation, real data, checkpoints, or production packed forward. Its job is
    to prove that variable selected dense positions can be lifted to complete
    temporal-tubelet groups, and that ragged variable-budget batches have an
    explicit bucketed-pack fallback before R33 can become a trainable candidate.
    """

    plan = dict(dynamic_plan or build_synthetic_dynamic_tubelet_plan())
    tubelets = _positive_int(temporal_tubelets, name="temporal_tubelets")
    spatial_tokens = _positive_int(spatial_h, name="spatial_h") * _positive_int(spatial_w, name="spatial_w")
    channel_count = _positive_int(channels, name="channels")
    rows = resolve_pc_ot_mras_dynamic_budget_plan(plan)
    if not rows:
        raise ValueError("dynamic plan must resolve to at least one hard-position row")

    samples = [
        _row_to_tubelet_sample(
            row,
            row_idx=row_idx,
            temporal_tubelets=tubelets,
            spatial_tokens=spatial_tokens,
        )
        for row_idx, row in enumerate(rows)
    ]
    by_tubelet_count: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        if int(sample["selected_tubelet_count"]) <= 0:
            raise ValueError("each sample must select at least one temporal tubelet")
        by_tubelet_count[int(sample["selected_tubelet_count"])].append(sample)

    bucket_summaries = [
        _build_bucket_summary(
            bucket_samples,
            selected_tubelet_count=count,
            temporal_tubelets=tubelets,
            spatial_tokens=spatial_tokens,
            channels=channel_count,
        )
        for count, bucket_samples in sorted(by_tubelet_count.items())
    ]
    budget_values = sorted({int(row["budget"]) for row in rows})
    selected_tubelet_counts = sorted(by_tubelet_count)
    single_rectangular_pack_ready = len(selected_tubelet_counts) == 1
    bucketed_pack_required = not single_rectangular_pack_ready
    all_bucket_packable = all(
        item["rectangular_pack_ready"]
        and item["selected_values_preserved"]
        and item["unselected_positions_zero"]
        for item in bucket_summaries
    )
    no_spatial_crop = all(bool(sample["full_spatial_groups_selected"]) for sample in samples)
    selected_mask_consistent = all(bool(sample["hard_row_selected_mask_consistent"]) for sample in samples)
    has_variable_budget = len(budget_values) > 1
    passed = (
        no_spatial_crop
        and selected_mask_consistent
        and all_bucket_packable
        and (single_rectangular_pack_ready or bool(bucketed_pack_fallback_allowed))
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "decision": READY if passed else NO_GO,
        "synthetic_or_deploy_visible_plan_only": True,
        "local_protocol_audit_only": True,
        "uses_gt": False,
        "uses_teacher": False,
        "uses_cache": False,
        "uses_raw_prediction": False,
        "uses_checkpoint": False,
        "source_dynamic_plan_schema": str(plan.get("schema_version", "")),
        "source_hard_position_schema": "pc_ot_mras_hard_positions_v0",
        "route_unit": "temporal_tubelet_group",
        "temporal_tubelets": tubelets,
        "spatial_h": int(spatial_h),
        "spatial_w": int(spatial_w),
        "spatial_tokens_per_tubelet": spatial_tokens,
        "channels": channel_count,
        "sample_count": len(samples),
        "budget_values": budget_values,
        "has_variable_budget": has_variable_budget,
        "selected_tubelet_counts": selected_tubelet_counts,
        "single_rectangular_pack_ready": single_rectangular_pack_ready,
        "bucketed_pack_required": bucketed_pack_required,
        "bucketed_pack_fallback_allowed": bool(bucketed_pack_fallback_allowed),
        "fallback_route": "bucket_by_selected_tubelet_count_then_pack_or_dense_no_pack",
        "all_bucket_packable": all_bucket_packable,
        "full_spatial_groups_selected": no_spatial_crop,
        "selected_mask_consistent": selected_mask_consistent,
        "spatial_patch_crop_allowed": False,
        "spatial_filtering_allowed": False,
        "arbitrary_spatial_patch_filtering_allowed": False,
        "production_forward_changed": False,
        "true_packed_compute_enabled": False,
        "packed_attention_executed": False,
        "packed_mlp_executed": False,
        "measured_runtime": False,
        "measured_flops": False,
        "runtime_flops_claim_allowed": False,
        "metric_claim_allowed": False,
        "paper_claim_allowed": False,
        "deploy_claim_allowed": False,
        "dynamic_budget_quality_validation": False,
        "dynamic_budget_validation": False,
        "remote_sync_allowed": False,
        "slurm_gpu_allowed": False,
        "detector_map_allowed": False,
        "sample_summaries": samples,
        "bucket_summaries": bucket_summaries,
    }


def run_json_dynamic_tubelet_contract_audit(
    config_json: Path | None = None,
    *,
    summary_json: Path | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if config_json is not None:
        kwargs = json.loads(Path(config_json).read_text(encoding="utf-8"))
    summary = audit_pc_ot_mras_dynamic_tubelet_contract(**kwargs)
    if summary_json is not None:
        write_json(summary_json, strict_json_value(summary))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-json", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    args = parser.parse_args(argv)
    summary = run_json_dynamic_tubelet_contract_audit(args.config_json, summary_json=args.summary_json)
    print(json.dumps(strict_json_value(summary), indent=2, sort_keys=True))
    return 0 if summary["decision"] == READY else 2


if __name__ == "__main__":
    raise SystemExit(main())
