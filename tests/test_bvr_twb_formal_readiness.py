import json
from pathlib import Path

import pytest

from opentad.acquisition.bvr_twb.adapter_bridge import ADAPTER_FIXED_LENGTH_PADDED_BRIDGE
from opentad.acquisition.bvr_twb.types import ROUTE_LABEL
from tools.bvr_twb.validate_bvr_twb_shortdiag import REQUIRED_PRETRAIN_PATH
from tools.bvr_twb.validate_bvr_twb_formal_readiness import (
    FormalReadinessError,
    validate_formal_readiness,
    validate_formal_train_log,
    validate_linux_geometry_summary,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/adatad/thumos/input_bvr_twb_dynamic_adapter_irregular_headv3.py"


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _pipeline_summary(path, overrides=None):
    payload = {
        "route_label": ROUTE_LABEL,
        "blocked": False,
        "all_validated": True,
        "sparse_compute_claim": False,
        "no_training": True,
        "no_metric_claim": True,
        "adapter_bridge_mode": ADAPTER_FIXED_LENGTH_PADDED_BRIDGE,
        "adapter_bridge_modes": [ADAPTER_FIXED_LENGTH_PADDED_BRIDGE],
        "adapter_padding_counts_as_valid": False,
        "raw_frame_handoff_stages": ["pre_decode_selected_raw_frames"],
        "selected_raw_frames_before_decode": True,
        "fixed_padded_bridge_sparse_compute_claim": False,
        "adapter_padding_invalid_for_detector": True,
        "preview_sources": ["deploy_visible_metadata_actionness"],
        "scout_sources": ["deploy_visible_raw_or_metadata_scout"],
        "deterministic_preview_fallback_used": False,
        "value_modes": ["deploy_heuristic_voi"],
        "value_labels_used_at_test": False,
    }
    if overrides:
        payload.update(overrides)
    return _write_json(path, payload)


def _linux_geometry_summary(path, overrides=None):
    payload = {
        "validator": "bvr_twb_geometry_contracts",
        "platform_system": "Linux",
        "source_contract": "passed",
        "numpy_bridge_contract": "passed",
        "torch_runtime_contract": "passed",
        "torch_runtime_skipped": False,
        "no_training": True,
        "no_metric_claim": True,
        "full_training_unlocked": False,
    }
    if overrides:
        payload.update(overrides)
    return _write_json(path, payload)


def _formal_log(path, extra_lines=(), omit_pretrain=False, omit_gradient_evidence=False):
    lines = [
        ROUTE_LABEL,
        "bvr_twb_dynamic_subsample",
        ADAPTER_FIXED_LENGTH_PADDED_BRIDGE,
    ]
    if not omit_pretrain:
        lines.extend(
            [
                REQUIRED_PRETRAIN_PATH,
                "Loads checkpoint by local backend from path: " + REQUIRED_PRETRAIN_PATH,
            ]
        )
    lines.extend(
        [
            "[Train]: [000][00001/00002] Loss=2.5000 cls_loss=1.0000 reg_loss=1.5000 boundary_loss=0.1000",
            "[Train][RuntimeDebug]: epoch=0 iter=1 bad_param_name=periodic | "
            "head_v3_regression_head_fp32_enabled=True | "
            "head_v3_regression_loss_fp32_enabled=True | "
            "head_v3_regression_samples_kept_after_filter=12 | "
            "head_v2_reg_points_total=12",
        ]
    )
    if not omit_gradient_evidence:
        lines.append(
            "[bvr_twb_formal_precheck] finite_gradients=true no_skipped_optimizer_step=true "
            "no_skipped_reg_head=true linux_torch_precheck=true pretrain_loaded=true "
            "full_train_unlocked=false sparse_compute_claim=false"
        )
    lines.extend(extra_lines)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_formal_readiness_requires_all_evidence_but_keeps_train_locked(tmp_path):
    pytest.importorskip("mmengine.config")
    result = validate_formal_readiness(
        CONFIG,
        _pipeline_summary(tmp_path / "pipeline.json"),
        _linux_geometry_summary(tmp_path / "geometry.json"),
        _formal_log(tmp_path / "train.log"),
    )

    assert result["gate_pass"] is True
    assert result["formal_readiness_evidence_complete"] is True
    assert result["full_train_unlocked"] is False
    assert result["formal_train_unlocked"] is False
    assert result["sparse_compute_claim"] is False
    assert result["allowed_next_action"] == "FORMAL_REVIEW_ONLY_FULL_TRAIN_STILL_LOCKED"


def test_formal_readiness_rejects_missing_linux_torch_geometry(tmp_path):
    bad = _linux_geometry_summary(
        tmp_path / "geometry.json",
        overrides={"platform_system": "Windows", "torch_runtime_skipped": True, "torch_runtime_contract": None},
    )
    with pytest.raises(FormalReadinessError, match="Linux geometry precheck"):
        validate_linux_geometry_summary(json.loads(bad.read_text(encoding="utf-8")))


def test_formal_readiness_rejects_historical_nonfinite_gradient_marker(tmp_path):
    log = _formal_log(
        tmp_path / "bad_nonfinite.log",
        extra_lines=["[Train]: non-finite gradients detected at epoch=0 iter=0, param=rpn_head.reg_head.weight, skip optimizer step"],
    )
    with pytest.raises(Exception, match="non[-_ ]finite|stop marker"):
        validate_formal_train_log(log)


def test_formal_readiness_rejects_missing_pretrain_load_evidence(tmp_path):
    log = _formal_log(tmp_path / "missing_pretrain.log", omit_pretrain=True)
    with pytest.raises(Exception, match="pretrain"):
        validate_formal_train_log(log)


def test_formal_readiness_rejects_missing_gradient_and_reg_head_evidence(tmp_path):
    missing_gradient = _formal_log(tmp_path / "missing_gradient.log", omit_gradient_evidence=True)
    with pytest.raises(FormalReadinessError, match="finite-gradient"):
        validate_formal_train_log(missing_gradient)

    zero_reg = _formal_log(
        tmp_path / "zero_reg.log",
        extra_lines=["[Train][RuntimeDebug]: epoch=0 iter=1 head_v3_regression_samples_kept_after_filter=0"],
    )
    with pytest.raises(FormalReadinessError, match="zero_regression_samples_kept"):
        validate_formal_train_log(zero_reg)


def test_launch_gate_rejects_padded_bridge_as_sparse_compute_claim(tmp_path):
    pytest.importorskip("mmengine.config")
    summary = _pipeline_summary(tmp_path / "pipeline.json", {"fixed_padded_bridge_sparse_compute_claim": True})
    with pytest.raises(ValueError, match="fixed padded bridge sparse-compute claims"):
        from tools.bvr_twb.validate_bvr_twb_launch_gate import validate_launch_gate

        validate_launch_gate(CONFIG, summary)
