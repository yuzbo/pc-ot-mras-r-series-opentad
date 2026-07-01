import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from opentad.acquisition.rba_rbr.metadata import resolve_backbone_time_axis_meta
from opentad.acquisition.rba_rbr.types import ROUTE_LABEL
from tools.rba_rbr.build_synthetic_ledgers import build_ledgers, run_recovery_case
from tools.rba_rbr.validate_rba_rbr_launch_gate import validate_launch_gate


ROOT = Path(__file__).resolve().parents[1]


class _FakeBatch:
    def __init__(self, frames):
        self._frames = np.asarray(frames, dtype=np.uint8)

    def asnumpy(self):
        return self._frames


class _FakeVideoReader:
    def get_batch(self, frame_indices):
        return _FakeBatch([self._frame(idx) for idx in frame_indices])

    @staticmethod
    def _frame(frame_index):
        frame_index = int(frame_index)
        frame = np.zeros((24, 24, 3), dtype=np.uint8)
        pos = int((frame_index * 5) % 18)
        frame[pos : pos + 6, pos : pos + 6, :] = 160 + int(frame_index % 60)
        frame[:, :, 2] = np.clip(frame[:, :, 2] + (frame_index * 11) % 90, 0, 255)
        return frame


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
    assert cfg.evaluation.ground_truth_filename == cfg.annotation_path
    assert "/root/autodl-tmp" not in cfg.evaluation.ground_truth_filename.replace("\\", "/")
    assert cfg.dataset.train.pipeline[2].method == "rba_rbr_recoverable_bracketing"
    assert cfg.dataset.train.pipeline[2].rba_rbr_split == "train"
    assert cfg.dataset.train.pipeline[2].rba_rbr_scout_sample_count == 32
    assert cfg.dataset.train.pipeline[2].rba_rbr_min_keep == 64
    assert cfg.dataset.train.pipeline[2].rba_rbr_min_detector_feature_keep == 32
    assert cfg.dataset.train.pipeline[2].rba_rbr_max_raw_gap == 16
    assert cfg.dataset.train.pipeline[2].rba_rbr_max_detector_gap == 24
    assert cfg.dataset.train.pipeline[2].rba_rbr_feature_stride == 2
    assert cfg.post_processing.pre_nms_topk == 512
    assert cfg.post_processing.rba_rbr_postprocess_guard.enabled is True
    assert cfg.post_processing.rba_rbr_postprocess_guard.require_rba_meta is True
    assert cfg.post_processing.rba_rbr_postprocess_guard.raw_proposal_cap == 1024
    assert cfg.post_processing.rba_rbr_postprocess_guard.per_class_topk == 32
    assert cfg.post_processing.rba_rbr_postprocess_guard.total_candidate_cap == 512
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


def test_rba_rbr_shortdiag_config_is_training_limited_and_claim_locked():
    mmengine_config = pytest.importorskip("mmengine.config")
    cfg = mmengine_config.Config.fromfile(
        str(ROOT / "configs/adatad/thumos/input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3_shortdiag.py")
    )

    assert cfg.route_label == ROUTE_LABEL
    assert cfg.short_diagnostic_only is True
    assert cfg.full_train_unlocked is False
    assert cfg.no_metric_claim is True
    assert cfg.no_runtime_claim is True
    assert cfg.no_deploy_claim is True
    assert cfg.no_paper_claim is True
    assert cfg.workflow.end_epoch == 2
    assert cfg.workflow.val_eval_interval == -1
    assert cfg.workflow.disable_checkpoint is True
    assert cfg.evaluation.ground_truth_filename == cfg.annotation_path
    assert "/root/autodl-tmp" not in cfg.evaluation.ground_truth_filename.replace("\\", "/")
    assert "shortdiag_only" in cfg.work_dir


def test_rba_rbr_evaldiag_config_is_bounded_eval_and_claim_locked():
    mmengine_config = pytest.importorskip("mmengine.config")
    cfg = mmengine_config.Config.fromfile(
        str(ROOT / "configs/adatad/thumos/input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3_evaldiag.py")
    )

    assert cfg.route_label == ROUTE_LABEL
    assert cfg.diagnostic_eval_only is True
    assert cfg.full_train_unlocked is False
    assert cfg.no_metric_claim is True
    assert cfg.no_runtime_claim is True
    assert cfg.no_deploy_claim is True
    assert cfg.no_paper_claim is True
    assert cfg.workflow.end_epoch == 4
    assert cfg.workflow.val_start_epoch == 1
    assert cfg.workflow.val_eval_interval == 2
    assert cfg.workflow.checkpoint_interval == 2
    assert cfg.workflow.disable_checkpoint is False
    assert cfg.evaluation.ground_truth_filename == cfg.annotation_path
    assert "/root/autodl-tmp" not in cfg.evaluation.ground_truth_filename.replace("\\", "/")
    assert "evaldiag_only" in cfg.work_dir


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
        rba_rbr_min_detector_feature_keep=4,
        rba_rbr_max_raw_gap=12,
        rba_rbr_scaffold_k=4,
        rba_rbr_feature_stride=2,
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
    assert out["rba_rbr_ledger"]["post_guard_detector_feature_valid_k"] >= 4
    assert out["rba_rbr_ledger"]["max_raw_gap_after_guard"] <= 12


def test_loadframes_dispatch_builds_raw_scout_when_preview_metadata_is_absent():
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
    transform = LoadFrames(
        num_clips=1,
        method="rba_rbr_recoverable_bracketing",
        method_base="sliding_window",
        target_len=16,
        scale_factor=1,
        remap_gt_to_selected_axis=False,
        rba_rbr_split="test",
        rba_rbr_min_keep=5,
        rba_rbr_max_keep=16,
        rba_rbr_scaffold_k=4,
        rba_rbr_train_value_labels=False,
        rba_rbr_allow_diagnostic_preview_fallback=False,
        rba_rbr_scout_sample_count=8,
    )
    out = transform(
        {
            "total_frames": dense_T,
            "avg_fps": 30.0,
            "snippet_stride": 1,
            "window_size": dense_T,
            "feature_start_idx": 0,
            "feature_end_idx": dense_T - 1,
            "video_name": "loadframes_rba_raw_scout",
            "video_reader": _FakeVideoReader(),
        }
    )
    ledger = out["rba_rbr_ledger"]
    assert ledger["preview_source"] == "raw_rgb_lowres_scout"
    assert ledger["scout_is_deploy_visible"] is True
    assert ledger["diagnostic_preview_fallback_used"] is False
    assert ledger["preview_meta"]["scout_sample_count"] == 8
    assert out["frame_inds"].shape[0] == out["rba_rbr_adapter_input_frame_count"]
    assert ledger["selected_positions"] == sorted(set(ledger["selected_positions"]))


def _require_torch_for_detector_grid():
    torch_probe = subprocess.run(
        [sys.executable, "-c", "import torch"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if torch_probe.returncode != 0:
        pytest.skip("torch unavailable for detector temporal-grid audit")


def test_rba_rbr_detector_grid_audit_uses_route_specific_native_axis_positions(tmp_path):
    _require_torch_for_detector_grid()
    import torch

    from opentad.models.detectors.irregular_actionformer import IrregularActionFormer

    audit_path = tmp_path / "rba_rbr_grid_audit.jsonl"
    old_enabled = os.environ.get("RBA_RBR_GRID_AUDIT")
    old_path = os.environ.get("RBA_RBR_GRID_AUDIT_PATH")
    try:
        os.environ["RBA_RBR_GRID_AUDIT"] = "1"
        os.environ["RBA_RBR_GRID_AUDIT_PATH"] = str(audit_path)
        detector = object.__new__(IrregularActionFormer)
        masks = torch.tensor([[True, True, True, False, False]])
        meta = {
            "video_name": "rba_grid_audit_unit",
            "irregular_native_axis": True,
            "irregular_selected_positions": np.asarray([100.0, 101.0, 102.0], dtype=np.float32),
            "irregular_selected_valid_len": 128.0,
            "rba_rbr_ledger": {"method": "rba_rbr_recoverable_bracketing"},
            "rba_rbr_raw_selected_positions": np.asarray([2.0, 6.0, 18.0, 22.0, 42.0, 46.0], dtype=np.float32),
            "rba_rbr_detector_feature_positions": np.asarray([4.0, 20.0, 44.0], dtype=np.float32),
            "rba_rbr_detector_feature_valid_len": 128.0,
        }
        meta["rba_rbr_ledger"].update(
            {
                "valid_k": 6,
                "dynamic_target_k": 12,
                "budget_stop_reason": "coverage_guard",
                "pre_guard_budget_stop_reason": "regret_saturation",
                "guard_reason": "min_detector_feature_keep+max_raw_gap",
                "guard_addition_count": 2,
                "selection_gap_diagnostics": {"max_gap": 24},
                "max_raw_gap_before_guard": 48,
                "max_raw_gap_after_guard": 24,
                "detector_feature_valid_k": 3,
                "detector_mask_len": 5,
                "detector_mask_true_count": 3,
                "detector_feature_target_k": 3,
                "max_detector_gap_before_guard": 32.0,
                "max_detector_gap_after_guard": 24.0,
                "adapter_padding_duplicate_count": 186,
            }
        )

        grid = detector._temporal_grid_from_metas([meta], masks)
        rows = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]

        assert torch.allclose(grid["center"][0, :3], torch.tensor([4.0, 20.0, 44.0]))
        assert not torch.allclose(grid["center"][0, :3], torch.tensor([100.0, 101.0, 102.0]))
        assert grid["valid_mask"][0].tolist() == [True, True, True, False, False]
        assert rows[0]["audit_type"] == "rba_rbr_detector_temporal_grid"
        assert rows[0]["route_label"] == ROUTE_LABEL
        assert rows[0]["native_axis"] is True
        assert rows[0]["mask_true_count"] == 3
        assert rows[0]["raw_valid_k"] == 6
        assert rows[0]["dynamic_target_k"] == 12
        assert rows[0]["budget_stop_reason"] == "coverage_guard"
        assert rows[0]["pre_guard_budget_stop_reason"] == "regret_saturation"
        assert rows[0]["guard_reason"] == "min_detector_feature_keep+max_raw_gap"
        assert rows[0]["guard_addition_count"] == 2
        assert rows[0]["selected_max_gap"] == 24
        assert rows[0]["selected_max_gap_before_guard"] == 48
        assert rows[0]["selected_max_gap_after_guard"] == 24
        assert rows[0]["meta_detector_feature_position_count"] == 3
        assert rows[0]["detector_feature_valid_k"] == 3
        assert rows[0]["detector_mask_len"] == 5
        assert rows[0]["detector_mask_true_count"] == 3
        assert rows[0]["detector_feature_target_k"] == 3
        assert rows[0]["max_detector_gap_before_guard"] == 32.0
        assert rows[0]["max_detector_gap_after_guard"] == 24.0
        assert rows[0]["adapter_padding_duplicate_count"] == 186
        assert rows[0]["grid_center_prefix"] == [4.0, 20.0, 44.0]
        assert rows[0]["status"] == "PASS_RBA_RBR_NATIVE_AXIS_POSITIONS_ENTERED_MODEL"
    finally:
        if old_enabled is None:
            os.environ.pop("RBA_RBR_GRID_AUDIT", None)
        else:
            os.environ["RBA_RBR_GRID_AUDIT"] = old_enabled
        if old_path is None:
            os.environ.pop("RBA_RBR_GRID_AUDIT_PATH", None)
        else:
            os.environ["RBA_RBR_GRID_AUDIT_PATH"] = old_path


def test_rba_rbr_detector_grid_fails_closed_on_mask_position_mismatch():
    _require_torch_for_detector_grid()
    import torch

    from opentad.models.detectors.irregular_actionformer import IrregularActionFormer

    detector = object.__new__(IrregularActionFormer)
    masks = torch.tensor([[True, True, False, False]])
    meta = {
        "video_name": "rba_grid_mismatch_unit",
        "irregular_native_axis": True,
        "rba_rbr_ledger": {"method": "rba_rbr_recoverable_bracketing"},
        "rba_rbr_detector_feature_positions": np.asarray([4.0, 20.0, 44.0], dtype=np.float32),
        "rba_rbr_detector_feature_valid_len": 128.0,
    }

    with pytest.raises(ValueError, match="RBA-RBR detector temporal grid mask true count"):
        detector._temporal_grid_from_metas([meta], masks)


def test_rba_rbr_postprocess_guard_records_route_audit_and_caps_candidates(tmp_path):
    _require_torch_for_detector_grid()
    import torch

    from opentad.models.detectors.irregular_actionformer import IrregularActionFormer

    audit_path = tmp_path / "rba_rbr_postprocess_audit.jsonl"
    old_enabled = os.environ.get("RBA_RBR_POSTPROCESS_AUDIT")
    old_path = os.environ.get("RBA_RBR_POSTPROCESS_AUDIT_PATH")
    try:
        os.environ["RBA_RBR_POSTPROCESS_AUDIT"] = "1"
        os.environ["RBA_RBR_POSTPROCESS_AUDIT_PATH"] = str(audit_path)
        detector = object.__new__(IrregularActionFormer)
        proposal_count = 120
        num_classes = 5
        starts = torch.arange(float(proposal_count), dtype=torch.float32)
        proposals = torch.stack([starts, starts + 1.0], dim=1)
        scores = torch.full((proposal_count, num_classes), 0.01, dtype=torch.float32)
        scores[:, 2] = torch.linspace(0.02, 0.80, proposal_count)
        metas = [
            {
                "video_name": "rba_postprocess_guard_unit",
                "fps": 30.0,
                "duration": 12.0,
                "snippet_stride": 1,
                "offset_frames": 0,
                "window_start_frame": 0,
                "irregular_native_axis": True,
                "rba_rbr_ledger": {
                    "method": "rba_rbr_recoverable_bracketing",
                    "valid_k": 84,
                    "detector_feature_valid_k": 42,
                },
                "rba_rbr_detector_feature_positions": np.asarray([2.0, 8.0, 16.0], dtype=np.float32),
                "rba_rbr_detector_feature_valid_len": 128.0,
            }
        ]
        post_cfg = SimpleNamespace(
            pre_nms_thresh=0.001,
            pre_nms_topk=20,
            sliding_window=True,
            nms=None,
            rba_rbr_postprocess_guard=dict(
                enabled=True,
                require_rba_meta=True,
                raw_proposal_cap=50,
                per_class_topk=3,
                total_candidate_cap=12,
                min_score=0.001,
            ),
        )

        results = detector.post_processing(([proposals], [scores]), metas, post_cfg, [f"class_{i}" for i in range(num_classes)])
        rows = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]

        assert rows[0]["audit_type"] == "rba_rbr_postprocess_proposal_count"
        assert rows[0]["route_label"] == ROUTE_LABEL
        assert rows[0]["guard_active"] is True
        assert rows[0]["candidate_generation_mode"] == "rba_rbr_guarded_raw_cap_per_class"
        assert rows[0]["raw_proposal_count"] == proposal_count
        assert rows[0]["flattened_candidate_count"] == proposal_count * num_classes
        assert rows[0]["guard_raw_selected_count"] == 50
        assert rows[0]["guard_class_candidate_count_before_global_topk"] <= 3 * num_classes
        assert rows[0]["pre_nms_selected_count"] <= 12
        assert rows[0]["final_result_count"] <= 12
        assert rows[0]["raw_valid_k"] == 84
        assert rows[0]["detector_feature_valid_k"] == 42
        assert getattr(detector, "_last_rba_rbr_postprocess_audit")[0]["route_label"] == ROUTE_LABEL
        assert "rba_postprocess_guard_unit" in results
    finally:
        if old_enabled is None:
            os.environ.pop("RBA_RBR_POSTPROCESS_AUDIT", None)
        else:
            os.environ["RBA_RBR_POSTPROCESS_AUDIT"] = old_enabled
        if old_path is None:
            os.environ.pop("RBA_RBR_POSTPROCESS_AUDIT_PATH", None)
        else:
            os.environ["RBA_RBR_POSTPROCESS_AUDIT_PATH"] = old_path


def test_rba_rbr_postprocess_guard_fails_closed_without_rba_meta():
    _require_torch_for_detector_grid()
    import torch

    from opentad.models.detectors.irregular_actionformer import IrregularActionFormer

    detector = object.__new__(IrregularActionFormer)
    proposals = torch.tensor([[0.0, 1.0], [2.0, 3.0]], dtype=torch.float32)
    scores = torch.full((2, 3), 0.1, dtype=torch.float32)
    metas = [
        {
            "video_name": "non_rba_guard_should_fail",
            "fps": 30.0,
            "duration": 2.0,
            "snippet_stride": 1,
            "offset_frames": 0,
            "window_start_frame": 0,
        }
    ]
    post_cfg = SimpleNamespace(
        pre_nms_thresh=0.001,
        pre_nms_topk=512,
        sliding_window=True,
        nms=None,
        rba_rbr_postprocess_guard=dict(
            enabled=True,
            require_rba_meta=True,
            raw_proposal_cap=1024,
            per_class_topk=32,
            total_candidate_cap=512,
            min_score=0.001,
        ),
    )

    with pytest.raises(ValueError, match="no RBA-RBR metadata"):
        detector.post_processing(([proposals], [scores]), metas, post_cfg, ["a", "b", "c"])


def test_bvr_detector_grid_path_is_not_captured_by_rba_rbr_audit(tmp_path):
    _require_torch_for_detector_grid()
    import torch

    from opentad.models.detectors.irregular_actionformer import IrregularActionFormer

    rba_audit_path = tmp_path / "rba_should_stay_empty.jsonl"
    bvr_audit_path = tmp_path / "bvr_grid_audit.jsonl"
    old_rba_enabled = os.environ.get("RBA_RBR_GRID_AUDIT")
    old_rba_path = os.environ.get("RBA_RBR_GRID_AUDIT_PATH")
    old_bvr_enabled = os.environ.get("BVR_TWB_GRID_AUDIT")
    old_bvr_path = os.environ.get("BVR_TWB_GRID_AUDIT_PATH")
    try:
        os.environ["RBA_RBR_GRID_AUDIT"] = "1"
        os.environ["RBA_RBR_GRID_AUDIT_PATH"] = str(rba_audit_path)
        os.environ["BVR_TWB_GRID_AUDIT"] = "1"
        os.environ["BVR_TWB_GRID_AUDIT_PATH"] = str(bvr_audit_path)
        detector = object.__new__(IrregularActionFormer)
        masks = torch.tensor([[True, True, False, False]])
        meta = {
            "video_name": "bvr_still_uses_bvr_grid",
            "irregular_native_axis": True,
            "bvr_twb_ledger": {"method": "bvr_twb_dynamic_subsample"},
            "bvr_twb_detector_feature_positions": np.asarray([6.0, 30.0], dtype=np.float32),
            "bvr_twb_detector_feature_valid_len": 96.0,
        }

        grid = detector._temporal_grid_from_metas([meta], masks)
        bvr_rows = [json.loads(line) for line in bvr_audit_path.read_text(encoding="utf-8").splitlines()]

        assert torch.allclose(grid["center"][0, :2], torch.tensor([6.0, 30.0]))
        assert bvr_rows[0]["route_label"] == "DIVERGENT_INNOVATION_BVR_TWB_DO_NOT_MERGE_WITH_C3"
        assert bvr_rows[0]["audit_type"] == "bvr_twb_detector_temporal_grid"
        assert not rba_audit_path.exists()
    finally:
        for key, value in (
            ("RBA_RBR_GRID_AUDIT", old_rba_enabled),
            ("RBA_RBR_GRID_AUDIT_PATH", old_rba_path),
            ("BVR_TWB_GRID_AUDIT", old_bvr_enabled),
            ("BVR_TWB_GRID_AUDIT_PATH", old_bvr_path),
        ):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
