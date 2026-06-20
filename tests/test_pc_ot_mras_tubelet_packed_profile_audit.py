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

from tools.bata.audit_pc_ot_mras_tubelet_packed_profile import (  # noqa: E402
    NO_GO,
    READY,
    audit_pc_ot_mras_tubelet_packed_profile,
    run_json_tubelet_packed_profile_audit,
)


def test_tubelet_packed_profile_reports_pack_scatter_and_accounting():
    summary = audit_pc_ot_mras_tubelet_packed_profile(
        mode="deterministic_tubelet_cap",
        keep_ratio=0.5,
        temporal_tubelets=8,
        spatial_h=2,
        spatial_w=3,
        channels=4,
        num_blocks=12,
    )

    assert summary["decision"] == READY
    assert summary["profiler_only"] is True
    assert summary["true_packed_compute_enabled"] is False
    assert summary["packed_attention_executed"] is False
    assert summary["packed_mlp_executed"] is False
    assert summary["measured_runtime"] is False
    assert summary["measured_flops"] is False
    assert summary["runtime_flops_claim_allowed"] is False
    assert summary["route_unit"] == "temporal_tubelet_group"
    assert summary["spatial_patch_crop_allowed"] is False
    assert summary["spatial_filtering_allowed"] is False
    assert summary["selected_values_preserved"] is True
    assert summary["unselected_positions_zero"] is True
    assert summary["dense_mask_shape"] == [2, 48]
    assert summary["packed_token_shape"] == [2, 24, 4]
    assert summary["scatter_shape"] == [2, 48, 4]

    profile = summary["count_profile"]
    assert profile["dense_token_count"] == 48
    assert profile["packed_token_count"] == 24
    assert profile["dense_attention_token_pairs_per_sample"] == 48 * 48 * 12
    assert profile["packed_attention_token_pairs_per_sample"] == 24 * 24 * 12
    assert profile["attention_pair_ratio"] == pytest.approx(0.25)
    assert profile["linear_token_ratio"] == pytest.approx(0.5)


def test_tubelet_packed_profile_identity_has_no_strict_saving():
    summary = audit_pc_ot_mras_tubelet_packed_profile(
        mode="identity",
        keep_ratio=1.0,
        temporal_tubelets=4,
        spatial_h=2,
        spatial_w=2,
        channels=3,
        num_blocks=2,
    )

    assert summary["decision"] == NO_GO
    assert summary["count_profile"]["attention_pair_ratio"] == pytest.approx(1.0)
    assert summary["count_profile"]["linear_token_ratio"] == pytest.approx(1.0)
    assert summary["runtime_flops_claim_allowed"] is False
    assert summary["metric_claim_allowed"] is False
    assert summary["paper_claim_allowed"] is False


def test_tubelet_packed_profile_json_roundtrip(tmp_path):
    config_json = tmp_path / "r29_packed_profile_config.json"
    summary_json = tmp_path / "r29_packed_profile_summary.json"
    config_json.write_text(
        json.dumps(
            {
                "mode": "deterministic_tubelet_cap",
                "keep_ratio": 0.25,
                "temporal_tubelets": 8,
                "spatial_h": 2,
                "spatial_w": 3,
                "channels": 4,
                "num_blocks": 4,
            }
        ),
        encoding="utf-8",
    )

    summary = run_json_tubelet_packed_profile_audit(config_json, summary_json=summary_json)
    loaded = json.loads(summary_json.read_text(encoding="utf-8"))

    assert summary["decision"] == loaded["decision"] == READY
    assert loaded["schema_version"] == "pc_ot_mras_tubelet_packed_profile_audit_v0"
    assert loaded["packed_token_shape"] == [2, 12, 4]
    assert loaded["count_profile"]["dense_token_count"] == 48
    assert loaded["count_profile"]["packed_token_count"] == 12
    assert loaded["count_profile"]["attention_pair_ratio"] == pytest.approx(0.0625)
    assert loaded["runtime_flops_claim_allowed"] is False
    assert loaded["detector_map_allowed"] is False


def test_tubelet_packed_profile_rejects_invalid_block_count():
    with pytest.raises(ValueError, match="num_blocks"):
        audit_pc_ot_mras_tubelet_packed_profile(num_blocks=0)
