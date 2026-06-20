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

from tools.bata.audit_pc_ot_mras_tubelet_packed_runtime import (  # noqa: E402
    NO_GO,
    READY,
    audit_pc_ot_mras_tubelet_packed_runtime,
    run_json_tubelet_packed_runtime_audit,
)


def test_tubelet_packed_runtime_executes_attention_mlp_and_scatter():
    summary = audit_pc_ot_mras_tubelet_packed_runtime(
        mode="deterministic_tubelet_cap",
        keep_ratio=0.5,
        temporal_tubelets=8,
        spatial_h=2,
        spatial_w=3,
        channels=8,
        num_heads=2,
        mlp_ratio=2.0,
        num_blocks=2,
        warmup=0,
        repeat=1,
    )

    assert summary["decision"] == READY
    assert summary["synthetic_only"] is True
    assert summary["local_runtime_proof_only"] is True
    assert summary["production_forward_changed"] is False
    assert summary["true_packed_compute_enabled"] is True
    assert summary["packed_attention_executed"] is True
    assert summary["packed_mlp_executed"] is True
    assert summary["packed_block_impl"] == "vit_adapter.Block(use_adapter=False)"
    assert summary["scatter_back_executed"] is True
    assert summary["measured_runtime"] is True
    assert summary["runtime_measurement_scope"] == "local_synthetic_tiny_block_only"
    assert summary["measured_flops"] is False
    assert summary["runtime_flops_claim_allowed"] is False
    assert summary["metric_claim_allowed"] is False
    assert summary["paper_claim_allowed"] is False
    assert summary["detector_map_allowed"] is False
    assert summary["spatial_patch_crop_allowed"] is False
    assert summary["spatial_filtering_allowed"] is False
    assert summary["dense_token_shape"] == [2, 48, 8]
    assert summary["packed_token_shape"] == [2, 24, 8]
    assert summary["dense_output_shape"] == [2, 48, 8]
    assert summary["packed_output_shape"] == [2, 24, 8]
    assert summary["scatter_shape"] == [2, 48, 8]
    assert summary["has_strict_token_saving"] is True
    assert summary["packed_output_finite"] is True
    assert summary["dense_output_finite"] is True
    assert summary["scattered_output_finite"] is True
    assert summary["packed_attention_forward_count"] == summary["packed_attention_forward_count_expected"] == 2
    assert summary["packed_mlp_forward_count"] == summary["packed_mlp_forward_count_expected"] == 2
    assert summary["selected_outputs_match_reference"] is True
    assert summary["selected_reference_max_abs_error"] <= 1.0e-5
    assert summary["selected_output_changed_from_input"] is True
    assert summary["unselected_positions_zero_after_scatter"] is True
    assert summary["dense_runtime_ms"]["median"] >= 0.0
    assert summary["packed_runtime_ms"]["median"] >= 0.0
    assert summary["packed_over_dense_runtime_ratio_observed"] >= 0.0
    assert summary["count_profile"]["attention_pair_ratio"] == pytest.approx(0.25)
    assert summary["count_profile"]["linear_token_ratio"] == pytest.approx(0.5)


def test_tubelet_packed_runtime_identity_is_no_go_before_runtime_claim():
    with pytest.raises(ValueError, match="R29 packed profile must pass"):
        audit_pc_ot_mras_tubelet_packed_runtime(
            mode="identity",
            keep_ratio=1.0,
            temporal_tubelets=4,
            spatial_h=2,
            spatial_w=2,
            channels=4,
            num_heads=2,
            num_blocks=1,
            warmup=0,
            repeat=1,
        )


def test_tubelet_packed_runtime_json_roundtrip(tmp_path):
    config_json = tmp_path / "r30_packed_runtime_config.json"
    summary_json = tmp_path / "r30_packed_runtime_summary.json"
    config_json.write_text(
        json.dumps(
            {
                "mode": "deterministic_tubelet_cap",
                "keep_ratio": 0.25,
                "temporal_tubelets": 8,
                "spatial_h": 2,
                "spatial_w": 3,
                "channels": 8,
                "num_heads": 2,
                "mlp_ratio": 2.0,
                "num_blocks": 1,
                "warmup": 0,
                "repeat": 1,
            }
        ),
        encoding="utf-8",
    )

    summary = run_json_tubelet_packed_runtime_audit(config_json, summary_json=summary_json)
    loaded = json.loads(summary_json.read_text(encoding="utf-8"))

    assert summary["decision"] == loaded["decision"] == READY
    assert loaded["schema_version"] == "pc_ot_mras_tubelet_packed_runtime_audit_v0"
    assert loaded["packed_token_shape"] == [2, 12, 8]
    assert loaded["true_packed_compute_enabled"] is True
    assert loaded["packed_attention_executed"] is True
    assert loaded["packed_mlp_executed"] is True
    assert loaded["runtime_flops_claim_allowed"] is False
    assert loaded["detector_map_allowed"] is False


def test_tubelet_packed_runtime_rejects_bad_head_divisibility():
    with pytest.raises(ValueError, match="divisible"):
        audit_pc_ot_mras_tubelet_packed_runtime(
            channels=7,
            num_heads=2,
            warmup=0,
            repeat=1,
        )
