import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from opentad.acquisition.rba_rbr.adapter_bridge import build_adapter_fixed_length_padded_bridge
from opentad.acquisition.rba_rbr.open_tad_bridge import build_rba_rbr_open_tad_selection
from opentad.acquisition.rba_rbr.types import ROUTE_LABEL
from opentad.acquisition.rba_rbr.validators import exact_uniform_positions, validate_rba_rbr_control_bridge_metadata
from tools.rba_rbr.build_synthetic_ledgers import build_ledgers


ROOT = Path(__file__).resolve().parents[1]
FULL_CONTROL_CONFIG = (
    ROOT
    / "configs/adatad/thumos/input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3_control_uniform_full.py"
)
LOW_CONTROL_CONFIG = (
    ROOT
    / "configs/adatad/thumos/input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3_control_uniform_lowbudget.py"
)


def _control_bridge(keep):
    dense_T = 384
    target_frame_num = 192
    feature_stride = 2
    result = build_rba_rbr_open_tad_selection(
        {"video_name": f"uniform_control_{keep}"},
        dense_window=np.arange(dense_T, dtype=np.int64),
        target_frame_num=target_frame_num,
        split="test",
        train_value_labels=False,
        max_keep=target_frame_num,
        feature_stride=feature_stride,
        control_mode="uniform_raw",
        control_keep=keep,
    )
    adapter = build_adapter_fixed_length_padded_bridge(
        selected_positions=result["keep_positions"],
        selected_frame_inds=result["selected_frame_inds"],
        target_frame_num=target_frame_num,
        dense_T=dense_T,
        feature_stride=feature_stride,
    )
    bridge_meta = {
        "irregular_native_axis": True,
        "rba_rbr_raw_selected_positions": result["keep_positions"],
        "rba_rbr_raw_selected_valid_len": float(dense_T),
        "rba_rbr_detector_feature_positions": adapter["detector_feature_positions"],
        "rba_rbr_detector_feature_valid_len": float(dense_T),
        "detector_valid_mask": adapter["detector_valid_mask"],
        "adapter_valid_raw_mask": adapter["adapter_valid_raw_mask"],
    }
    return result, adapter, bridge_meta


def test_forced_uniform_full_control_preserves_raw_and_detector_bridge_positions():
    result, adapter, bridge_meta = _control_bridge(192)
    ledger = result["ledger"]
    expected_positions = exact_uniform_positions(384, 192)

    assert ledger["route_label"] == ROUTE_LABEL
    assert ledger["selector_method"] == "forced_uniform_control_diagnostic"
    assert ledger["rba_rbr_control_mode"] == "uniform_raw"
    assert ledger["valid_k"] == 192
    assert result["keep_positions"].tolist() == expected_positions
    assert result["selected_frame_inds"].tolist() == expected_positions
    assert adapter["adapter_padding_duplicate_count"] == 0
    assert adapter["adapter_valid_raw_mask"].sum() == 192
    assert adapter["detector_feature_valid_k"] == 96
    assert adapter["detector_valid_mask"].sum() == 96
    assert np.allclose(adapter["detector_feature_positions"], np.mean(np.asarray(expected_positions).reshape(-1, 2), axis=1))
    assert ledger["selector_provenance"]["selection_uses_gt"] is False
    assert ledger["selector_provenance"]["selection_uses_teacher"] is False
    assert ledger["selector_provenance"]["selection_uses_prediction_cache"] is False
    assert ledger["no_metric_claim"] is True
    assert ledger["no_runtime_claim"] is True
    assert ledger["no_deploy_claim"] is True
    assert ledger["no_paper_claim"] is True
    assert ledger["no_sparse_compute_claim"] is True
    assert validate_rba_rbr_control_bridge_metadata(ledger, bridge_meta)


def test_matched_low_budget_uniform_control_preserves_valid_prefix_and_padding():
    result, adapter, bridge_meta = _control_bridge(85)
    ledger = result["ledger"]
    expected_positions = exact_uniform_positions(384, 85)

    assert ledger["valid_k"] == 85
    assert ledger["detector_feature_valid_k"] == 43
    assert result["keep_positions"].tolist() == expected_positions
    assert adapter["adapter_padded_positions"][:85].tolist() == expected_positions
    assert adapter["adapter_valid_raw_mask"][:85].all()
    assert not adapter["adapter_valid_raw_mask"][85:].any()
    assert adapter["adapter_padding_duplicate_count"] == 107
    assert adapter["detector_mask_len"] == 96
    assert adapter["detector_valid_mask"].sum() == 43
    assert np.all(np.diff(result["keep_positions"]) > 0)
    assert np.all(np.diff(adapter["detector_feature_positions"]) > 0)
    assert validate_rba_rbr_control_bridge_metadata(ledger, bridge_meta)


def test_control_configs_resolve_and_keep_short_diagnostic_locks():
    mmengine_config = pytest.importorskip("mmengine.config")
    for path, expected_keep in ((FULL_CONTROL_CONFIG, 192), (LOW_CONTROL_CONFIG, 85)):
        cfg = mmengine_config.Config.fromfile(str(path))

        assert cfg.route_label == ROUTE_LABEL
        assert cfg.rba_rbr_control_diagnostic_only is True
        assert cfg.short_diagnostic_only is True
        assert cfg.full_train_unlocked is False
        assert cfg.no_metric_claim is True
        assert cfg.no_runtime_claim is True
        assert cfg.no_deploy_claim is True
        assert cfg.no_paper_claim is True
        assert cfg.no_sparse_compute_claim is True
        assert cfg.workflow.end_epoch == 2
        assert cfg.workflow.disable_checkpoint is True
        for split in ("train", "val", "test"):
            step = cfg.dataset[split].pipeline[2]
            assert step.method == "rba_rbr_recoverable_bracketing"
            assert step.rba_rbr_control_mode == "uniform_raw"
            assert step.rba_rbr_control_keep == expected_keep
            assert step.rba_rbr_train_value_labels is False
            assert step.rba_rbr_feature_stride == 2
            assert step.rba_rbr_adapter_bridge_mode == "adapter_fixed_length_padded_bridge"


def test_launch_gate_audits_control_configs_and_reports_bridge_summary(tmp_path):
    out_dir = tmp_path / ".tmp_rba_rbr_control_gate_pytest"
    build_ledgers(out_dir, overwrite=True, root=tmp_path)
    for path, expected_keep, expected_detector_k in (
        (FULL_CONTROL_CONFIG, 192, 96),
        (LOW_CONTROL_CONFIG, 85, 43),
    ):
        cmd = [
            sys.executable,
            str(ROOT / "tools/rba_rbr/validate_rba_rbr_launch_gate.py"),
            "--config",
            str(path),
            "--precheck-summary",
            str(out_dir / "summary.json"),
        ]
        completed = subprocess.run(cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert completed.returncode == 0, completed.stderr
        payload = json.loads(completed.stdout)
        assert payload["gate_pass"] is True
        assert payload["full_train_unlocked"] is False
        assert payload["allowed_next_action"] == "FINAL_READ_ONLY_REVIEW_THEN_SHORT_DIAGNOSTIC_ONLY"
        assert payload["control_audit"]["raw_valid_k"] == expected_keep
        assert payload["control_audit"]["detector_feature_valid_k"] == expected_detector_k
        assert payload["control_audit"]["detector_mask_len"] == 96
