import json
import subprocess
import sys
from pathlib import Path

import pytest


torch_probe = subprocess.run(
    [sys.executable, "-c", "import torch"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    timeout=30,
    check=False,
)
if torch_probe.returncode != 0:
    pytest.skip("torch unavailable", allow_module_level=True)

from tools.bata.audit_pc_ot_mras_dynamic_tubelet_contract import (  # noqa: E402
    READY,
    audit_pc_ot_mras_dynamic_tubelet_contract,
    build_synthetic_dynamic_tubelet_plan,
    run_json_dynamic_tubelet_contract_audit,
)


def test_dynamic_tubelet_contract_maps_variable_budget_rows_to_bucketed_full_tubelets():
    summary = audit_pc_ot_mras_dynamic_tubelet_contract(
        temporal_tubelets=8,
        spatial_h=2,
        spatial_w=3,
        channels=4,
    )

    assert summary["decision"] == READY
    assert summary["schema_version"] == "pc_ot_mras_dynamic_tubelet_contract_audit_v0"
    assert summary["source_dynamic_plan_schema"] == "pc_ot_mras_dynamic_budget_plan_v0"
    assert summary["source_hard_position_schema"] == "pc_ot_mras_hard_positions_v0"
    assert summary["synthetic_or_deploy_visible_plan_only"] is True
    assert summary["local_protocol_audit_only"] is True
    assert summary["has_variable_budget"] is True
    assert summary["budget_values"] == [4, 5, 6]
    assert summary["selected_tubelet_counts"] == [2, 3, 4]
    assert summary["single_rectangular_pack_ready"] is False
    assert summary["bucketed_pack_required"] is True
    assert summary["bucketed_pack_fallback_allowed"] is True
    assert summary["all_bucket_packable"] is True
    assert summary["full_spatial_groups_selected"] is True
    assert summary["selected_mask_consistent"] is True
    assert summary["spatial_patch_crop_allowed"] is False
    assert summary["true_packed_compute_enabled"] is False
    assert summary["metric_claim_allowed"] is False
    assert summary["paper_claim_allowed"] is False
    assert summary["detector_map_allowed"] is False

    for sample in summary["sample_summaries"]:
        assert sample["full_spatial_groups_selected"] is True
        assert sample["hard_row_selected_mask_consistent"] is True
        assert sample["dense_token_mask_true_count"] == (
            sample["selected_tubelet_count"] * summary["spatial_tokens_per_tubelet"]
        )

    for bucket in summary["bucket_summaries"]:
        assert bucket["rectangular_pack_ready"] is True
        assert bucket["selected_values_preserved"] is True
        assert bucket["unselected_positions_zero"] is True
        assert bucket["packed_token_shape"][1] == (
            bucket["selected_tubelet_count"] * summary["spatial_tokens_per_tubelet"]
        )
        assert bucket["scatter_shape"][1] == summary["temporal_tubelets"] * summary["spatial_tokens_per_tubelet"]


def test_dynamic_tubelet_contract_single_bucket_can_pack_rectangular_batch():
    plan = build_synthetic_dynamic_tubelet_plan()
    plan["budgets"] = [4, 4]
    plan["dense_valid_len"] = [16, 16]
    plan["selected_dense_positions"] = [
        [0, 1, 8, 9],
        [2, 3, 10, 11],
    ]
    plan["selected_mask"] = [
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ]
    plan["budget_scores"] = [0.2, 0.4]
    plan["coverage_counts"] = [2, 2]
    plan["value_counts"] = [3, 3]
    plan["coverage_share"] = [0.25, 0.25]

    summary = audit_pc_ot_mras_dynamic_tubelet_contract(plan)

    assert summary["decision"] == READY
    assert summary["has_variable_budget"] is False
    assert summary["selected_tubelet_counts"] == [2]
    assert summary["single_rectangular_pack_ready"] is True
    assert summary["bucketed_pack_required"] is False
    assert len(summary["bucket_summaries"]) == 1
    assert summary["bucket_summaries"][0]["batch_size"] == 2


def test_dynamic_tubelet_contract_json_roundtrip(tmp_path: Path):
    config_json = tmp_path / "r32_dynamic_tubelet_contract_config.json"
    summary_json = tmp_path / "r32_dynamic_tubelet_contract_summary.json"
    config_json.write_text(
        json.dumps(
            {
                "dynamic_plan": build_synthetic_dynamic_tubelet_plan(),
                "temporal_tubelets": 8,
                "spatial_h": 2,
                "spatial_w": 3,
                "channels": 4,
            }
        ),
        encoding="utf-8",
    )

    summary = run_json_dynamic_tubelet_contract_audit(config_json, summary_json=summary_json)
    loaded = json.loads(summary_json.read_text(encoding="utf-8"))

    assert summary["decision"] == loaded["decision"] == READY
    assert loaded["schema_version"] == "pc_ot_mras_dynamic_tubelet_contract_audit_v0"
    assert loaded["bucketed_pack_required"] is True
    assert loaded["runtime_flops_claim_allowed"] is False
    assert loaded["metric_claim_allowed"] is False
    assert loaded["paper_claim_allowed"] is False


def test_dynamic_tubelet_contract_rejects_forbidden_or_claim_payloads():
    plan = build_synthetic_dynamic_tubelet_plan()
    plan["labels"] = [1, 2, 3]
    with pytest.raises(ValueError, match="forbidden|unsupported"):
        audit_pc_ot_mras_dynamic_tubelet_contract(plan)

    plan = build_synthetic_dynamic_tubelet_plan()
    plan["metric_claim_allowed"] = True
    with pytest.raises(ValueError, match="must be false"):
        audit_pc_ot_mras_dynamic_tubelet_contract(plan)
