import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from opentad.acquisition.rba_rbr.metadata import resolve_backbone_time_axis_meta
from opentad.acquisition.rba_rbr.types import ROUTE_LABEL
from tools.rba_rbr.build_synthetic_ledgers import build_ledgers, run_recovery_case
from tools.rba_rbr.validate_rba_rbr_launch_gate import validate_launch_gate


ROOT = Path(__file__).resolve().parents[1]


def test_rba_rbr_config_resolves_and_stays_fail_closed():
    mmengine_config = pytest.importorskip("mmengine.config")
    cfg = mmengine_config.Config.fromfile(
        str(ROOT / "configs/adatad/thumos/input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3.py")
    )
    text = (ROOT / "configs/adatad/thumos/input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3.py").read_text(
        encoding="utf-8"
    )

    assert cfg.route_label == ROUTE_LABEL
    assert cfg.full_train_unlocked is False
    assert cfg.no_metric_claim is True
    assert cfg.no_runtime_claim is True
    assert cfg.no_deploy_claim is True
    assert cfg.no_paper_claim is True
    assert cfg.dataset.train.pipeline[2].method == "rba_rbr_recoverable_bracketing"
    assert cfg.dataset.train.pipeline[2].rba_rbr_split == "train"
    assert cfg.dataset.val.pipeline[2].rba_rbr_train_value_labels is False
    assert cfg.dataset.test.pipeline[2].rba_rbr_train_value_labels is False
    required_meta = {
        "irregular_selected_positions",
        "irregular_selected_valid_len",
        "irregular_native_axis",
        "rba_rbr_raw_selected_positions",
        "rba_rbr_raw_selected_valid_len",
        "rba_rbr_detector_feature_positions",
        "rba_rbr_detector_feature_valid_len",
        "rba_rbr_ledger",
        "rba_rbr_selected_positions",
        "rba_rbr_selected_valid_len",
        "rba_rbr_dense_valid_len",
        "rba_rbr_train_value_labels",
        "rba_rbr_candidate_count",
    }
    for split_name in ("train", "val", "test"):
        collect = cfg.dataset[split_name].pipeline[-1]
        assert required_meta.issubset(set(collect.meta_keys))
    normalized = text.replace(ROUTE_LABEL, "")
    for token in ("CADF", "PQR", "C3_MAINLINE", "C3_ORIGINAL", "BVR result", "ABR result"):
        assert token not in normalized


def test_backbone_time_axis_prefers_rba_raw_positions_over_detector_centers():
    meta = {
        "irregular_selected_positions": np.asarray([10.5, 30.5], dtype=np.float32),
        "irregular_selected_valid_len": 80.0,
        "rba_rbr_raw_selected_positions": np.asarray([10.0, 11.0, 30.0, 31.0], dtype=np.float32),
        "rba_rbr_raw_selected_valid_len": 80.0,
    }
    positions, valid_len, source = resolve_backbone_time_axis_meta(meta)

    assert source == "rba_rbr_raw"
    assert np.allclose(positions, np.asarray([10.0, 11.0, 30.0, 31.0], dtype=np.float32))
    assert valid_len == 80.0


def test_backbone_wrapper_source_uses_rba_raw_axis_helper_and_keeps_generic_for_detector_centers():
    text = (ROOT / "opentad/models/backbones/backbone_wrapper.py").read_text(encoding="utf-8")
    metadata_text = (ROOT / "opentad/acquisition/rba_rbr/metadata.py").read_text(encoding="utf-8")
    loadframes_text = (ROOT / "opentad/datasets/transforms/end_to_end.py").read_text(encoding="utf-8")

    assert "resolve_backbone_time_axis_meta" in text
    assert "has_backbone_time_axis_meta" in text
    assert "rba_rbr_raw_selected_positions" in metadata_text
    assert 'results["irregular_selected_positions"] = detector_feature_positions / scale' in loadframes_text
    assert 'results["rba_rbr_raw_selected_positions"] = keep_positions.astype(np.float32) / scale' in loadframes_text


def test_synthetic_ledger_builder_proves_recovery_over_hard_bracket(tmp_path):
    result, candidates, diagnostic = run_recovery_case()
    assert result.deploy_ledger["route_label"] == ROUTE_LABEL
    assert diagnostic["missed_boundary_position"] not in diagnostic["hard_bracket_positions"]
    assert diagnostic["recovered_boundary_by_rescue"] is True
    assert any(packet.stage == "rescue" for packet in candidates)

    out_dir = tmp_path / ".tmp_rba_rbr_ledgers_pytest"
    summary = build_ledgers(out_dir, overwrite=True, root=tmp_path)
    assert summary["route_label"] == ROUTE_LABEL
    assert summary["all_validated"] is True
    assert summary["recovery_diagnostic"]["recovered_boundary_by_rescue"] is True
    assert summary["no_training"] is True
    assert summary["no_metric_claim"] is True
    assert summary["no_runtime_claim"] is True
    assert summary["no_deploy_claim"] is True
    assert summary["no_paper_claim"] is True
    assert summary["claim_status"] == "rba_rbr_local_precheck_only_no_metric_runtime_deploy_or_paper_claim"


def test_launch_gate_accepts_only_fail_closed_local_precheck(tmp_path):
    out_dir = tmp_path / ".tmp_rba_rbr_gate_pytest"
    build_ledgers(out_dir, overwrite=True, root=tmp_path)
    result = validate_launch_gate(
        ROOT / "configs/adatad/thumos/input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3.py",
        out_dir / "summary.json",
    )
    assert result["gate_pass"] is True
    assert result["full_train_unlocked"] is False
    assert result["deploy_claim_unlocked"] is False
    assert result["paper_claim_unlocked"] is False
    assert result["allowed_next_action"] == "FINAL_READ_ONLY_REVIEW_THEN_LOCAL_PRECHECK_ONLY"

    bad_summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    bad_summary["route_label"] = "DIVERGENT_INNOVATION_BVR_TWB_DO_NOT_MERGE_WITH_C3"
    bad_path = out_dir / "bad_summary.json"
    bad_path.write_text(json.dumps(bad_summary), encoding="utf-8")
    with pytest.raises(ValueError, match="wrong route_label"):
        validate_launch_gate(
            ROOT / "configs/adatad/thumos/input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3.py",
            bad_path,
        )

    claim_error_patterns = {
        "no_deploy_claim": "deployment readiness",
        "no_paper_claim": "paper readiness",
    }
    for locked_claim, error_pattern in claim_error_patterns.items():
        bad_claim_summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
        bad_claim_summary[locked_claim] = False
        bad_claim_path = out_dir / f"bad_{locked_claim}.json"
        bad_claim_path.write_text(json.dumps(bad_claim_summary), encoding="utf-8")
        with pytest.raises(ValueError, match=error_pattern):
            validate_launch_gate(
                ROOT / "configs/adatad/thumos/input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3.py",
                bad_claim_path,
            )


def test_validator_cli_runs_on_synthetic_ledgers(tmp_path):
    out_dir = tmp_path / ".tmp_rba_rbr_cli_pytest"
    cmd = [
        sys.executable,
        str(ROOT / "tools/rba_rbr/build_synthetic_ledgers.py"),
        "--out-dir",
        str(out_dir),
        "--overwrite",
    ]
    completed = subprocess.run(cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert completed.returncode == 0, completed.stderr
    assert "RBA-RBR synthetic ledgers" in completed.stdout

    gate_cmd = [
        sys.executable,
        str(ROOT / "tools/rba_rbr/validate_rba_rbr_launch_gate.py"),
        "--config",
        str(ROOT / "configs/adatad/thumos/input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3.py"),
        "--precheck-summary",
        str(out_dir / "summary.json"),
    ]
    gate = subprocess.run(gate_cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert gate.returncode == 0, gate.stderr
    assert '"gate_pass": true' in gate.stdout


def test_loadframes_dispatch_records_rba_rbr_ledger_if_runtime_available():
    torch_probe = subprocess.run(
        [sys.executable, "-c", "import torch"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if torch_probe.returncode != 0:
        pytest.skip("torch unavailable for LoadFrames runtime dispatch")

    from opentad.datasets.transforms.end_to_end import LoadFrames

    dense_T = 80
    actionness = np.zeros(dense_T, dtype=np.float64) + 0.05
    actionness[18:31] = 0.82
    uncertainty = np.zeros(dense_T, dtype=np.float64) + 0.08
    transition = np.zeros(dense_T, dtype=np.float64) + 0.04
    uncertainty[54:59] = 0.92
    transition[55:58] = 0.95

    transform = LoadFrames(
        num_clips=1,
        method="rba_rbr_recoverable_bracketing",
        method_base="sliding_window",
        target_len=16,
        scale_factor=1,
        remap_gt_to_selected_axis=False,
        rba_rbr_split="val",
        rba_rbr_min_keep=5,
        rba_rbr_max_keep=16,
        rba_rbr_scaffold_k=4,
        rba_rbr_train_value_labels=False,
        rba_rbr_allow_diagnostic_preview_fallback=False,
    )
    out = transform(
        {
            "total_frames": dense_T,
            "avg_fps": 30.0,
            "snippet_stride": 1,
            "window_size": dense_T,
            "feature_start_idx": 0,
            "feature_end_idx": dense_T - 1,
            "video_name": "loadframes_rba",
            "rba_rbr_preview_actionness": actionness,
            "rba_rbr_preview_uncertainty": uncertainty,
            "rba_rbr_preview_transition": transition,
        }
    )
    assert out["rba_rbr_ledger"]["route_label"] == ROUTE_LABEL
    assert out["rba_rbr_ledger"]["method"] == "rba_rbr_recoverable_bracketing"
    assert out["frame_inds"].shape[0] == out["rba_rbr_adapter_input_frame_count"]
    assert out["rba_rbr_ledger"]["selected_positions"] == sorted(set(out["rba_rbr_ledger"]["selected_positions"]))
