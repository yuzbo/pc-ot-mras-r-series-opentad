import json
import importlib.util
import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from opentad.acquisition.mdl_knot import (
    MDLKnotConfig,
    apply_mdl_knot_to_dense_window,
    build_frame_metadata_scout_curve,
    build_raw_frame_motion_scout_curve,
    build_synthetic_scout_curve,
    greedy_mdl_knot_select,
    validate_formal_readiness_evidence,
)
from opentad.acquisition.mdl_knot.diagnostics import FormalReadinessLocked


ROOT = Path(__file__).resolve().parents[1]
ROUTE_LABEL = "DIVERGENT_INNOVATION_MDL_KNOT_DO_NOT_MERGE_WITH_C3"
CONFIG_PATH = ROOT / "configs" / "adatad" / "thumos" / "input_mdl_knot_dynamic_adapter_irregular_headv3.py"
SHORTDIAG_CONFIG_PATH = ROOT / "configs" / "adatad" / "thumos" / "input_mdl_knot_dynamic_adapter_irregular_headv3_shortdiag.py"
SHORTDIAG_VALIDATOR_PATH = ROOT / "tools" / "mdl_knot" / "validate_mdl_knot_shortdiag.py"
PSEUDO_BOUNDARY_PATH = ROOT / "opentad" / "datasets" / "transforms" / "pseudo_boundary.py"


def _dense_raw_frames(length: int) -> list[np.ndarray]:
    frames = []
    for idx in range(int(length)):
        frame = np.zeros((8, 10, 3), dtype=np.uint8)
        frame[:, :, 0] = idx % 255
        frame[:, :, 1] = (idx * 3) % 255
        frame[:, :, 2] = np.arange(10, dtype=np.uint8)[None, :]
        frames.append(frame)
    return frames


class _InMemoryVideoReader:
    def __init__(self, frame_count: int) -> None:
        self.frames = _dense_raw_frames(frame_count)
        self.requested_indices = []

    def __getitem__(self, index: int) -> np.ndarray:
        self.requested_indices.append(int(index))
        return self.frames[int(index)]

    def get_batch(self, indices) -> np.ndarray:
        self.requested_indices.extend([int(index) for index in indices])
        return np.stack([self.frames[int(index)] for index in indices], axis=0)


def test_pipeline_mock_sets_frame_inds_before_decode_and_records_valid_k():
    dense_window = list(range(1000, 1128))
    curve = build_synthetic_scout_curve("short_islands", dense_t=len(dense_window))
    results = {"video_name": "video_test_0001", "total_frames": 2000}
    dense_inputs = _dense_raw_frames(len(dense_window))

    updated = apply_mdl_knot_to_dense_window(
        results=results,
        dense_window=dense_window,
        scout_curve=curve,
        config=MDLKnotConfig(route_label=ROUTE_LABEL, max_k=40),
        dense_inputs=dense_inputs,
    )

    assert updated["frame_inds"] == sorted(updated["frame_inds"])
    assert updated["frame_inds"] == [dense_window[pos] for pos in updated["mdl_knot_selected_positions"]]
    assert updated["mdl_knot_valid_k"] == len(updated["mdl_knot_selected_positions"])
    assert len(updated["masks"]) == updated["mdl_knot_valid_k"]
    assert updated["mdl_knot_sparse_meta"]["position_unit"] == "original_dense_time_index"
    assert updated["mdl_knot_sparse_meta"]["selected_frame_inds_prefix"] == updated["frame_inds"][: updated["mdl_knot_valid_k"]]
    assert updated["mdl_knot_sparse_meta"]["handoff_audit"]["selected_inputs_is_gathered"] is True
    assert updated["mdl_knot_sparse_meta"]["handoff_audit"]["audit_mode"] == "full_raw"
    assert updated["mdl_knot_sparse_meta"]["handoff_audit"]["raw_inputs_retained"] is False
    assert updated["mdl_knot_real_sparse_handoff_validated"] is True
    assert updated["mdl_knot_handoff_audit"]["selected_inputs_is_gathered"] is True
    assert updated["mdl_knot_handoff_audit"]["raw_inputs_retained"] is False


def test_clean_clone_pseudo_boundary_dependency_exists_without_torch_import():
    assert PSEUDO_BOUNDARY_PATH.exists()
    spec = importlib.util.spec_from_file_location("pseudo_boundary_dependency_check", PSEUDO_BOUNDARY_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for name in (
        "load_pseudo_boundary_cache",
        "select_pseudo_boundary_hybrid_positions",
        "select_pseudo_boundary_snap_positions",
    ):
        assert hasattr(module, name)


def test_fixed_adapter_bridge_pads_frame_inds_without_counting_padding_as_valid():
    dense_window = list(range(1000, 1128))
    curve = build_synthetic_scout_curve("short_islands", dense_t=len(dense_window))
    results = {"video_name": "video_test_bridge", "total_frames": 2000}
    dense_inputs = _dense_raw_frames(len(dense_window))

    updated = apply_mdl_knot_to_dense_window(
        results=results,
        dense_window=dense_window,
        scout_curve=curve,
        config=MDLKnotConfig(route_label=ROUTE_LABEL, max_k=40),
        adapter_target_len=64,
        dense_inputs=dense_inputs,
    )

    valid_k = updated["mdl_knot_valid_k"]
    assert len(updated["frame_inds"]) == 64
    assert len(updated["masks"]) == 64
    assert valid_k < 64
    assert sum(bool(v) for v in updated["masks"]) == valid_k
    assert updated["mdl_knot_padding_counts_as_valid"] is False
    assert updated["frame_inds"][:valid_k] == [dense_window[pos] for pos in updated["mdl_knot_selected_positions"]]
    assert updated["frame_inds"][valid_k:] == [updated["frame_inds"][valid_k - 1]] * (64 - valid_k)
    assert updated["mdl_knot_sparse_meta"]["detector_frame_inds_len"] == 64
    assert updated["mdl_knot_sparse_meta"]["handoff_audit"]["detector_padding_repeats_last_selected"] is True


def test_mdl_knot_config_overrides_real_dataset_pipelines_without_dead_standalone_pipelines():
    cfg = runpy.run_path(str(CONFIG_PATH))

    assert "dataset" in cfg
    assert "train_pipeline" not in cfg
    assert "val_pipeline" not in cfg
    assert "test_pipeline" not in cfg

    for split in ("train", "val", "test"):
        pipeline = cfg["dataset"][split]["pipeline"]
        load_frames = [step for step in pipeline if step.get("type") == "LoadFrames"]
        assert len(load_frames) == 1
        assert load_frames[0]["method"] == "mdl_knot_dynamic_subsample"
        assert load_frames[0]["method"] != "random_fixed_subsample"
        assert load_frames[0].get("mdl_knot_no_gt_selector") is True
        assert load_frames[0].get("mdl_knot_no_teacher") is True
        assert load_frames[0].get("mdl_knot_no_prediction_cache") is True
        assert load_frames[0].get("mdl_knot_no_dense_raw_backbone_handoff") is True
        assert "mmaction.DecordInit" in [step.get("type") for step in pipeline]
        assert "mmaction.DecordDecode" in [step.get("type") for step in pipeline]
        assert "ConvertToTensor" in [step.get("type") for step in pipeline]
        assert pipeline.index(load_frames[0]) < [step.get("type") for step in pipeline].index("mmaction.DecordDecode")

    train_load = [step for step in cfg["dataset"]["train"]["pipeline"] if step.get("type") == "LoadFrames"][0]
    assert train_load["method_base"] == "random_trunc"
    assert train_load["target_len"] == cfg["window_size"]
    assert train_load["source_len"] == cfg["dense_window_size"]
    for split in ("val", "test"):
        assert cfg["dataset"][split]["window_size"] == cfg["dense_window_size"]
        load = [step for step in cfg["dataset"][split]["pipeline"] if step.get("type") == "LoadFrames"][0]
        assert load["method_base"] == "sliding_window"
        assert load["target_len"] == cfg["window_size"]

    assert cfg["dataset"]["test"]["pipeline"][-1]["keys"] == ["masks"]
    assert "gt_segments" in cfg["dataset"]["val"]["pipeline"][-1]["keys"]
    assert cfg["mdl_knot_acquisition"]["no_val_test_gt_selector"] is True
    assert cfg["mdl_knot_acquisition"]["deploy_scout_source"] == "raw_frame_motion_scout_with_metadata_fallback"
    assert cfg["mdl_knot_acquisition"]["synthetic_fallback_allowed"] is False
    assert cfg["evaluation"]["ground_truth_filename"] == cfg["annotation_path"]
    assert cfg["dataset"]["train"]["ann_file"] == cfg["annotation_path"]
    assert cfg["dataset"]["val"]["ann_file"] == cfg["annotation_path"]
    assert cfg["dataset"]["test"]["ann_file"] == cfg["annotation_path"]
    assert "/root/autodl-tmp" not in cfg["evaluation"]["ground_truth_filename"]
    assert cfg["formal_train_unlocked"] is False
    assert cfg["full_train_unlocked"] is False
    assert "USER_OVERRIDE" not in cfg["route_status"]
    assert "FORMAL_TRAIN" not in cfg["route_status"]
    assert "QUEUED" not in cfg["route_status"]
    assert cfg["solver"]["amp"] is True
    assert cfg["sparse_compute_claim"] is False
    assert cfg["mdl_knot_acquisition"]["formal_train_unlocked"] is False
    assert cfg["mdl_knot_acquisition"]["full_train_unlocked"] is False
    assert cfg["mdl_knot_acquisition"]["sparse_compute_claim"] is False
    assert cfg["mdl_knot_acquisition"]["fixed_pad_bridge_compute_boundary"]["sparse_compute_claim"] is False


def test_real_loadframes_mdl_knot_branch_sets_sparse_frame_inds_before_decode():
    torch_probe = subprocess.run(
        [sys.executable, "-c", "import torch"],
        text=True,
        capture_output=True,
    )
    if torch_probe.returncode != 0:
        pytest.skip(f"real OpenTAD LoadFrames smoke skipped because torch import fails: {torch_probe.stderr[-240:]}")

    from opentad.datasets.transforms.end_to_end import LoadFrames

    loader = LoadFrames(
        method="mdl_knot_dynamic_subsample",
        method_base="sliding_window",
        scale_factor=1,
        mdl_knot_max_k=40,
        mdl_knot_target_weighted_error=0.01,
        mdl_knot_deploy_scout_source="frame_metadata_scout",
        mdl_knot_allow_synthetic_fallback=False,
    )
    reader = _InMemoryVideoReader(128)
    results = {
        "video_name": "video_test_0002",
        "total_frames": 128,
        "avg_fps": 30.0,
        "snippet_stride": 1,
        "window_size": 128,
        "feature_start_idx": 0,
        "feature_end_idx": 127,
        "video_reader": reader,
    }

    out = loader(results)

    selected = out["mdl_knot_selected_positions"]
    valid_k = out["mdl_knot_valid_k"]
    assert out["frame_inds"].shape[0] == 64
    assert out["frame_inds"][:valid_k].tolist() == selected
    assert out["frame_inds"][valid_k:].tolist() == [out["frame_inds"][valid_k - 1].item()] * (64 - valid_k)
    assert valid_k == len(selected)
    assert out["masks"].shape[0] == 64
    assert int(out["masks"].sum().item()) == valid_k
    assert out["mdl_knot_padding_counts_as_valid"] is False
    assert out["mdl_knot_selector_used_gt"] is False
    assert out["mdl_knot_deploy_scout_source"] == "frame_metadata_scout"
    assert out["mdl_knot_deploy_scout_provenance"]["metadata_only"] is True
    diagnostic = out["mdl_knot_pipeline_diagnostic"]
    assert diagnostic["metadata_fallback_used"] is False
    assert diagnostic["mask_metadata_alignment"]["all_aligned"] is True
    assert diagnostic["frame_handoff_alignment"]["all_aligned"] is True
    assert diagnostic["fixed_pad_bridge_compute_boundary"]["sparse_compute_claim"] is False
    assert out["mdl_knot_real_sparse_handoff_validated"] is True
    assert out["mdl_knot_handoff_audit"]["selected_inputs_is_gathered"] is True
    assert out["mdl_knot_handoff_audit"]["audit_mode"] == "sampled_raw"
    assert out["mdl_knot_handoff_audit"]["dense_window_materialized_for_audit"] is False
    assert len(reader.requested_indices) < 128

    from opentad.datasets.transforms.formatting import Collect

    out["imgs"] = np.zeros((3, 64, 1, 1), dtype=np.float32)
    batch_item = Collect(inputs="imgs", keys=["masks"])(out)
    meta = batch_item["metas"]
    assert meta["irregular_selected_positions"].tolist() == out["irregular_selected_positions"].tolist()
    assert meta["mdl_knot_sparse_meta"]["selected_frame_inds_prefix"] == out["frame_inds"][:valid_k].tolist()
    assert meta["mdl_knot_sparse_meta"]["handoff_audit"]["selected_inputs_is_gathered"] is True
    assert meta["mdl_knot_pipeline_diagnostic"]["frame_handoff_alignment"]["all_aligned"] is True


def test_real_loadframes_sampled_raw_short_dense_window_degrades_full_observation_profiled(monkeypatch):
    torch_probe = subprocess.run(
        [sys.executable, "-c", "import torch"],
        text=True,
        capture_output=True,
    )
    if torch_probe.returncode != 0:
        pytest.skip(f"real OpenTAD LoadFrames smoke skipped because torch import fails: {torch_probe.stderr[-240:]}")

    from opentad.datasets.transforms.end_to_end import LoadFrames

    monkeypatch.setenv("MDL_KNOT_PROFILE", "1")
    reader = _InMemoryVideoReader(4)
    loader = LoadFrames(
        method="mdl_knot_dynamic_subsample",
        method_base="sliding_window",
        scale_factor=1,
        target_len=4,
        mdl_knot_min_k=4,
        mdl_knot_max_k=384,
        mdl_knot_deploy_scout_source="frame_metadata_scout",
        mdl_knot_allow_synthetic_fallback=False,
        mdl_knot_handoff_audit_mode="sampled_raw",
    )

    out = loader(
        {
            "video_name": "video_test_full_observation",
            "total_frames": 4,
            "avg_fps": 30.0,
            "snippet_stride": 1,
            "window_size": 4,
            "feature_start_idx": 0,
            "feature_end_idx": 3,
            "video_reader": reader,
        }
    )

    audit = out["mdl_knot_handoff_audit"]
    assert out["mdl_knot_valid_k"] == 4
    assert out["mdl_knot_selected_positions"] == [0, 1, 2, 3]
    assert audit["audit_mode"] == "sampled_raw_full_observation"
    assert audit["full_observation_no_compression"] is True
    assert audit["sampled_raw_sparse_compute_evidence"] is False
    assert audit["selected_inputs_is_gathered"] is False
    assert audit["sparse_compute_claim"] is False
    assert out["mdl_knot_real_sparse_handoff_validated"] is False
    diagnostic = out["mdl_knot_pipeline_diagnostic"]
    assert diagnostic["frame_handoff_alignment"]["full_observation_no_compression"] is True
    assert diagnostic["frame_handoff_alignment"]["sampled_raw_sparse_compute_evidence"] is False
    assert diagnostic["fixed_pad_bridge_compute_boundary"]["sparse_compute_claim"] is False
    for stage in (
        "scout_build",
        "selector_objective_loop",
        "gap_guard",
        "metadata_build",
        "handoff_metadata_build",
        "handoff_validator",
        "handoff_audit_sampled_raw",
        "handoff_audit_sampled_raw_decode",
    ):
        assert stage in out["mdl_knot_profile"]
        assert out["mdl_knot_profile"][stage]["count"] >= 1
        assert out["mdl_knot_profile"][stage]["total_ms"] >= 0.0


def test_real_loadframes_structural_audit_does_not_read_dense_raw_inputs():
    torch_probe = subprocess.run(
        [sys.executable, "-c", "import torch"],
        text=True,
        capture_output=True,
    )
    if torch_probe.returncode != 0:
        pytest.skip(f"real OpenTAD LoadFrames smoke skipped because torch import fails: {torch_probe.stderr[-240:]}")

    from opentad.datasets.transforms.end_to_end import LoadFrames

    reader = _InMemoryVideoReader(128)
    loader = LoadFrames(
        method="mdl_knot_dynamic_subsample",
        method_base="sliding_window",
        scale_factor=1,
        target_len=64,
        mdl_knot_max_k=40,
        mdl_knot_target_weighted_error=0.01,
        mdl_knot_deploy_scout_source="frame_metadata_scout",
        mdl_knot_allow_synthetic_fallback=False,
        mdl_knot_handoff_audit_mode="structural",
    )
    out = loader(
        {
            "video_name": "video_test_structural",
            "total_frames": 128,
            "avg_fps": 30.0,
            "snippet_stride": 1,
            "window_size": 128,
            "feature_start_idx": 0,
            "feature_end_idx": 127,
            "video_reader": reader,
        }
    )

    assert reader.requested_indices == []
    assert out["mdl_knot_handoff_audit"]["audit_mode"] == "structural"
    assert out["mdl_knot_handoff_audit"]["selected_inputs_is_gathered"] is False
    assert out["mdl_knot_handoff_audit"]["structural_sparse_handoff_validated"] is True
    assert out["mdl_knot_real_sparse_handoff_validated"] is False
    diagnostic = out["mdl_knot_pipeline_diagnostic"]
    assert diagnostic["frame_handoff_alignment"]["audit_mode"] == "structural"
    assert diagnostic["frame_handoff_alignment"]["all_aligned"] is True
    assert diagnostic["frame_handoff_alignment"]["formal_raw_handoff_evidence"] is False


def test_raw_frame_motion_scout_builder_is_deploy_visible_and_non_synthetic():
    frames = []
    for idx in range(6):
        frame = np.zeros((16, 16, 3), dtype=np.uint8)
        frame[:, :, 0] = idx * 20
        frame[4:8, 4:8, 1] = idx * 30
        frames.append(frame)
    curve = build_raw_frame_motion_scout_curve(frames, probe_positions=[0, 3, 6, 9, 12, 15], dense_t=16)

    assert curve.source == "raw_frame_motion_scout"
    assert curve.dense_t == 16
    assert curve.provenance["uses_raw_frame_probe"] is True
    assert curve.provenance["uses_gt"] is False
    assert max(curve.temporal_change) > 0.0


def test_frame_metadata_scout_is_real_but_marked_metadata_only():
    curve = build_frame_metadata_scout_curve(dense_t=16, total_frames=160, duration=5.0, fps=30.0)

    assert curve.source == "frame_metadata_scout"
    assert curve.provenance["metadata_only"] is True
    assert curve.provenance["uses_gt"] is False


def test_launch_gate_locked_until_precheck_passes(tmp_path):
    missing = tmp_path / "missing_summary.json"
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "mdl_knot" / "validate_mdl_knot_launch_gate.py"),
        "--config",
        str(missing),
    ]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    assert proc.returncode != 0
    assert "LOCKED" in proc.stdout


def test_precheck_outputs_validated_json_summary(tmp_path):
    out_dir = tmp_path / "precheck"
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "mdl_knot" / "audit_mdl_knot_pipeline_precheck.py"),
        "--out-dir",
        str(out_dir),
        "--overwrite",
    ]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr

    summary_path = out_dir / "mdl_knot_precheck_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["route_label"] == ROUTE_LABEL
    assert summary["validated"] is True
    assert summary["locked_actions"]["remote_sync"] is True
    assert summary["cases"]["short_islands"]["valid_k"] != summary["cases"]["stable_background"]["valid_k"]
    assert summary["real_video_pipeline_diagnostics"] is None
    assert summary["synthetic_pipeline_diagnostics"]["synthetic_fallback_rejected"] is False
    assert summary["synthetic_pipeline_diagnostics"]["valid_k_distribution"]["nonconstant"] is True


def test_launch_gate_unlocks_only_for_valid_precheck_summary(tmp_path):
    out_dir = tmp_path / "precheck"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "mdl_knot" / "audit_mdl_knot_pipeline_precheck.py"),
            "--out-dir",
            str(out_dir),
            "--overwrite",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    cmd = [
        sys.executable,
        str(ROOT / "tools" / "mdl_knot" / "validate_mdl_knot_launch_gate.py"),
        "--config",
        str(CONFIG_PATH),
        "--precheck-summary",
        str(out_dir / "mdl_knot_precheck_summary.json"),
    ]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr
    assert "PRECHECK_ONLY_REQUEST_ALLOWED" in proc.stdout


def test_launch_gate_rejects_formal_unlock_config_status_and_sparse_claim(tmp_path):
    unlocked_config = tmp_path / "input_mdl_knot_unlocked.py"
    text = CONFIG_PATH.read_text(encoding="utf-8")

    unlocked_config.write_text(text.replace("formal_train_unlocked = False", "formal_train_unlocked = True"), encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "mdl_knot" / "validate_mdl_knot_launch_gate.py"),
            "--config",
            str(unlocked_config),
            "--route-label",
            ROUTE_LABEL,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert proc.returncode != 0
    assert "formal_train_unlocked" in proc.stdout

    unlocked_config.write_text(
        text.replace(
            "LOCAL_FINAL_CODE_CANDIDATE_PRECHECK_ONLY_AFTER_SAMPLED_RAW_EDGE_FIX_NO_METRIC_CLAIMS",
            "LOCAL_FINAL_CODE_CANDIDATE_USER_OVERRIDE_FORMAL_TRAIN_QUEUED_AFTER_PREVIOUS_RUN_NO_METRIC_CLAIMS",
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "mdl_knot" / "validate_mdl_knot_launch_gate.py"),
            "--config",
            str(unlocked_config),
            "--route-label",
            ROUTE_LABEL,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert proc.returncode != 0
    assert "route_status" in proc.stdout

    unlocked_config.write_text(text.replace("sparse_compute_claim = False", "sparse_compute_claim = True"), encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "mdl_knot" / "validate_mdl_knot_launch_gate.py"),
            "--config",
            str(unlocked_config),
            "--route-label",
            ROUTE_LABEL,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert proc.returncode != 0
    assert "sparse_compute_claim" in proc.stdout


def test_launch_gate_rejects_non_mdl_config_evidence_and_forbidden_tokens(tmp_path):
    bad_summary = {
        "route_label": ROUTE_LABEL,
        "validated": True,
        "locked_actions": {
            "remote_sync": True,
            "slurm": True,
            "training": True,
            "evaluation": True,
            "tools_test_py": True,
        },
        "no_claims": {
            "mAP": True,
            "runtime": True,
            "FLOPs": True,
            "deploy": True,
            "paper": True,
            "sparse_compute": True,
        },
        "cases": {"a": {"valid_k": 4}, "b": {"valid_k": 7}},
        "config_evidence": {
            "route_label": ROUTE_LABEL,
            "dataset_pipelines_use_mdl": False,
            "load_methods": {"train": "random_fixed_subsample", "val": "random_fixed_subsample", "test": "random_fixed_subsample"},
            "forbidden_route_token_hits": [],
            "deploy_scout_source": "raw_frame_motion_scout_with_metadata_fallback",
            "synthetic_fallback_allowed": False,
            "no_metric_runtime_deploy_claims": True,
        },
    }
    summary_path = tmp_path / "bad_summary.json"
    summary_path.write_text(json.dumps(bad_summary), encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "mdl_knot" / "validate_mdl_knot_launch_gate.py"),
            "--config",
            str(CONFIG_PATH),
            "--route-label",
            ROUTE_LABEL,
            "--precheck-summary",
            str(summary_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert proc.returncode != 0
    assert "dataset pipelines do not all use MDL" in proc.stdout

    bad_summary["config_evidence"]["dataset_pipelines_use_mdl"] = True
    bad_summary["config_evidence"]["forbidden_route_token_hits"] = ["BVR"]
    summary_path.write_text(json.dumps(bad_summary), encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "mdl_knot" / "validate_mdl_knot_launch_gate.py"),
            "--config",
            str(CONFIG_PATH),
            "--route-label",
            ROUTE_LABEL,
            "--precheck-summary",
            str(summary_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert proc.returncode != 0
    assert "forbidden route tokens" in proc.stdout


def test_launch_gate_rejects_c3_pro_and_globalrank_drift_in_config_text(tmp_path):
    drift_config = tmp_path / "input_mdl_knot_with_drift_comment.py"
    text = CONFIG_PATH.read_text(encoding="utf-8")
    drift_config.write_text(text + "\n# forbidden drift: C3-Pro GlobalRank C3_MAINLINE\n", encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "mdl_knot" / "validate_mdl_knot_launch_gate.py"),
            "--config",
            str(drift_config),
            "--route-label",
            ROUTE_LABEL,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert proc.returncode != 0
    assert "forbidden route drift tokens" in proc.stdout
    assert "C3" in proc.stdout
    assert "GLOBALRANK" in proc.stdout


def test_launch_gate_rejects_missing_tools_test_lock_random_fixed_and_combo(tmp_path):
    safety = {
        split: {
            "no_gt_selector": True,
            "no_teacher": True,
            "no_prediction_cache": True,
            "no_dense_raw_backbone_handoff": True,
            "load_before_decode": True,
        }
        for split in ("train", "val", "test")
    }
    summary = {
        "route_label": ROUTE_LABEL,
        "validated": True,
        "locked_actions": {
            "remote_sync": True,
            "slurm": True,
            "training": True,
            "evaluation": True,
        },
        "no_claims": {
            "mAP": True,
            "runtime": True,
            "FLOPs": True,
            "deploy": True,
            "paper": True,
            "sparse_compute": True,
        },
        "cases": {"a": {"valid_k": 4}, "b": {"valid_k": 7}},
        "config_evidence": {
            "route_label": ROUTE_LABEL,
            "dataset_pipelines_use_mdl": True,
            "load_methods": {
                "train": "mdl_knot_dynamic_subsample",
                "val": "mdl_knot_dynamic_subsample",
                "test": "mdl_knot_dynamic_subsample",
            },
            "bridges": {"train": "fixed_pad", "val": "fixed_pad", "test": "fixed_pad"},
            "safety": safety,
            "forbidden_route_token_hits": [],
            "drift_tokens": [],
            "deploy_scout_source": "raw_frame_motion_scout_with_metadata_fallback",
            "real_scout_unavailable": False,
            "synthetic_fallback_allowed": False,
            "no_metric_runtime_deploy_claims": True,
        },
    }
    summary_path = tmp_path / "gate_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "mdl_knot" / "validate_mdl_knot_launch_gate.py"),
            "--config",
            str(CONFIG_PATH),
            "--route-label",
            ROUTE_LABEL,
            "--precheck-summary",
            str(summary_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert proc.returncode != 0
    assert "tools_test_py" in proc.stdout

    summary["locked_actions"]["tools_test_py"] = True
    summary["config_evidence"]["drift_tokens"] = ["RANDOM_FIXED_SUBSAMPLE"]
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "mdl_knot" / "validate_mdl_knot_launch_gate.py"),
            "--config",
            str(CONFIG_PATH),
            "--route-label",
            ROUTE_LABEL,
            "--precheck-summary",
            str(summary_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert proc.returncode != 0
    assert "random-fixed drift" in proc.stdout

    summary["config_evidence"]["drift_tokens"] = ["COMBO"]
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "mdl_knot" / "validate_mdl_knot_launch_gate.py"),
            "--config",
            str(CONFIG_PATH),
            "--route-label",
            ROUTE_LABEL,
            "--precheck-summary",
            str(summary_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert proc.returncode != 0
    assert "COMBO" in proc.stdout

    summary["config_evidence"]["drift_tokens"] = ["C3-Pro", "GlobalRank"]
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "mdl_knot" / "validate_mdl_knot_launch_gate.py"),
            "--config",
            str(CONFIG_PATH),
            "--route-label",
            ROUTE_LABEL,
            "--precheck-summary",
            str(summary_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert proc.returncode != 0
    assert "C3 drift" in proc.stdout


def test_launch_gate_rejects_precheck_without_sparse_compute_claim_lock(tmp_path):
    safety = {
        split: {
            "no_gt_selector": True,
            "no_teacher": True,
            "no_prediction_cache": True,
            "no_dense_raw_backbone_handoff": True,
            "load_before_decode": True,
        }
        for split in ("train", "val", "test")
    }
    summary = {
        "route_label": ROUTE_LABEL,
        "validated": True,
        "locked_actions": {
            "remote_sync": True,
            "slurm": True,
            "training": True,
            "evaluation": True,
            "tools_test_py": True,
        },
        "no_claims": {"mAP": True, "runtime": True, "FLOPs": True, "deploy": True, "paper": True},
        "cases": {"a": {"valid_k": 4}, "b": {"valid_k": 7}},
        "config_evidence": {
            "route_label": ROUTE_LABEL,
            "dataset_pipelines_use_mdl": True,
            "load_methods": {
                "train": "mdl_knot_dynamic_subsample",
                "val": "mdl_knot_dynamic_subsample",
                "test": "mdl_knot_dynamic_subsample",
            },
            "bridges": {"train": "fixed_pad", "val": "fixed_pad", "test": "fixed_pad"},
            "safety": safety,
            "forbidden_route_token_hits": [],
            "drift_tokens": [],
            "deploy_scout_source": "raw_frame_motion_scout_with_metadata_fallback",
            "real_scout_unavailable": False,
            "synthetic_fallback_allowed": False,
            "no_metric_runtime_deploy_claims": True,
        },
    }
    summary_path = tmp_path / "gate_summary_missing_sparse_compute.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "mdl_knot" / "validate_mdl_knot_launch_gate.py"),
            "--config",
            str(CONFIG_PATH),
            "--route-label",
            ROUTE_LABEL,
            "--precheck-summary",
            str(summary_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert proc.returncode != 0
    assert "sparse_compute" in proc.stdout


def _write_valid_shortdiag_log(tmp_path: Path) -> Path:
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "mdl_knot_shortdiag_one_epoch.log"
    log_path.write_text(
        "\n".join(
            [
                "Epoch [1][1/2] lr: 1.0e-04 Loss 2.0 loss_cls: 0.5",
                "Epoch [1][2/2] loss_bbox=0.1 finite diagnostic train step complete",
                "Training short diagnostic stopped after one epoch without evaluation",
            ]
        ),
        encoding="utf-8",
    )
    return log_path


def _formal_readiness_summary(train_log: str = "logs/mdl_knot_shortdiag_one_epoch.log") -> dict:
    return {
        "route_label": ROUTE_LABEL,
        "validated": True,
        "source_mode": "real_video_pipeline",
        "dry_run_fixture": False,
        "annotation_path": "annotations/thumos_14_anno.json",
        "video_root": ["thumos14/test"],
        "reader_backend": {
            "mode": "annotation_video_root",
            "decord_available": True,
            "real_video_reader_required_for_formal_readiness": True,
        },
        "real_video_reader_window_count": 2,
        "real_video_raw_scout_count": 2,
        "fixture_window_count": 0,
        "formal_train_unlocked": False,
        "full_train_unlocked": False,
        "locked_actions": {
            "remote_sync": True,
            "slurm": True,
            "training": True,
            "evaluation": True,
            "tools_test_py": True,
        },
        "no_claims": {
            "mAP": True,
            "runtime": True,
            "FLOPs": True,
            "deploy": True,
            "paper": True,
            "sparse_compute": True,
        },
        "real_video_pipeline_diagnostics": {
            "window_count": 3,
            "raw_frame_scout_windows": 2,
            "metadata_fallback_windows": 1,
            "synthetic_fallback_windows": 0,
            "synthetic_fallback_rejected": True,
            "valid_k_distribution": {"count": 3, "unique_count": 2, "nonconstant": True},
            "max_gap_distribution": {"count": 3},
            "gap_p95_distribution": {"count": 3},
            "mask_metadata_alignment": {"all_aligned": True},
            "mask_meta_alignment_status": "aligned",
            "frame_handoff_alignment": {"all_aligned": True},
            "frame_handoff_alignment_status": "aligned",
            "handoff_audit_modes": {"full_raw": 3},
            "full_raw_handoff_evidence_windows": 3,
            "sampled_raw_handoff_evidence_windows": 0,
            "structural_handoff_evidence_windows": 0,
            "short_boundary_risk_monitoring": {
                "short_island_total": 2,
                "short_island_uncovered_count": 0,
                "transition_band_total": 2,
                "transition_band_uncovered_count": 0,
                "vanilla_mdl_smoothing_risk_measurable": True,
            },
            "fixed_pad_bridge_compute_boundary": {
                "bridge": "fixed_pad",
                "sparse_compute_claim": False,
            },
        },
        "shortdiag_evidence": {
            "validated": True,
            "formal_train_unlocked": False,
            "no_sparse_compute_claim": True,
            "evidence_scope": "one_epoch_train_log",
            "log_evidence": {
                "train_log": train_log,
                "finite_loss_count": 3,
                "loss_min": 0.1,
                "loss_max": 2.0,
                "epoch_max": 1,
            },
        },
    }


def test_formal_readiness_evidence_rejects_unlocked_actions_claims_and_config_only_shortdiag(tmp_path):
    _write_valid_shortdiag_log(tmp_path)
    valid = _formal_readiness_summary()
    validate_formal_readiness_evidence(valid, evidence_roots=[tmp_path])

    bad = _formal_readiness_summary()
    bad["formal_train_unlocked"] = True
    with pytest.raises(FormalReadinessLocked, match="formal_train_unlocked"):
        validate_formal_readiness_evidence(bad, evidence_roots=[tmp_path])

    bad = _formal_readiness_summary()
    bad["locked_actions"]["training"] = False
    with pytest.raises(FormalReadinessLocked, match="training"):
        validate_formal_readiness_evidence(bad, evidence_roots=[tmp_path])

    bad = _formal_readiness_summary()
    bad["no_claims"]["mAP"] = False
    with pytest.raises(FormalReadinessLocked, match="mAP"):
        validate_formal_readiness_evidence(bad, evidence_roots=[tmp_path])

    bad = _formal_readiness_summary()
    bad["no_claims"]["runtime"] = False
    with pytest.raises(FormalReadinessLocked, match="runtime"):
        validate_formal_readiness_evidence(bad, evidence_roots=[tmp_path])

    bad = _formal_readiness_summary()
    bad["no_claims"]["paper"] = False
    with pytest.raises(FormalReadinessLocked, match="paper"):
        validate_formal_readiness_evidence(bad, evidence_roots=[tmp_path])

    bad = _formal_readiness_summary()
    bad["no_claims"]["sparse_compute"] = False
    with pytest.raises(FormalReadinessLocked, match="sparse_compute"):
        validate_formal_readiness_evidence(bad, evidence_roots=[tmp_path])

    bad = _formal_readiness_summary()
    bad["shortdiag_evidence"]["validated"] = False
    bad["shortdiag_evidence"]["log_evidence"] = None
    bad["shortdiag_evidence"]["evidence_scope"] = "static_config_only"
    with pytest.raises(FormalReadinessLocked, match="shortdiag execution"):
        validate_formal_readiness_evidence(bad, evidence_roots=[tmp_path])


def test_formal_readiness_evidence_rejects_uncovered_short_or_transition_guards(tmp_path):
    _write_valid_shortdiag_log(tmp_path)
    bad = _formal_readiness_summary()
    bad["real_video_pipeline_diagnostics"]["short_boundary_risk_monitoring"]["short_island_uncovered_count"] = 1
    with pytest.raises(FormalReadinessLocked, match="short-island guard"):
        validate_formal_readiness_evidence(bad, evidence_roots=[tmp_path])

    bad = _formal_readiness_summary()
    bad["real_video_pipeline_diagnostics"]["short_boundary_risk_monitoring"]["transition_band_uncovered_count"] = 1
    with pytest.raises(FormalReadinessLocked, match="transition guard"):
        validate_formal_readiness_evidence(bad, evidence_roots=[tmp_path])

    bad = _formal_readiness_summary()
    bad["real_video_pipeline_diagnostics"]["short_transition_guard_coverage"] = {
        "placeholder_or_measured": "measured",
        "short_island_uncovered_count": 1,
        "transition_band_uncovered_count": 0,
    }
    with pytest.raises(FormalReadinessLocked, match="short-island guard"):
        validate_formal_readiness_evidence(bad, evidence_roots=[tmp_path])

    bad = _formal_readiness_summary()
    bad["real_video_pipeline_diagnostics"]["short_transition_guard_coverage"] = {
        "placeholder_or_measured": "measured",
        "short_island_uncovered_count": 0,
        "transition_band_uncovered_count": 1,
    }
    with pytest.raises(FormalReadinessLocked, match="transition guard"):
        validate_formal_readiness_evidence(bad, evidence_roots=[tmp_path])


def test_formal_readiness_launch_gate_requires_shortdiag_execution_evidence(tmp_path):
    _write_valid_shortdiag_log(tmp_path)
    summary = _formal_readiness_summary()
    summary["shortdiag_evidence"]["validated"] = False
    summary["shortdiag_evidence"]["log_evidence"] = None
    summary["shortdiag_evidence"]["evidence_scope"] = "static_config_only"
    summary_path = tmp_path / "config_only_shortdiag_formal.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "mdl_knot" / "validate_mdl_knot_launch_gate.py"),
            "--config",
            str(CONFIG_PATH),
            "--formal-readiness-summary",
            str(summary_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert proc.returncode != 0
    assert "formal training remains locked" in proc.stdout
    assert "shortdiag execution" in proc.stdout


def test_formal_readiness_launch_gate_cannot_use_globalrank_drift_log_evidence(tmp_path):
    log_path = tmp_path / "globalrank_drift.log"
    log_path.write_text(
        "Epoch [1] Loss 1.25 GlobalRank diagnostic drift marker",
        encoding="utf-8",
    )
    shortdiag_proc = subprocess.run(
        [
            sys.executable,
            str(SHORTDIAG_VALIDATOR_PATH),
            "--config",
            str(SHORTDIAG_CONFIG_PATH),
            "--train-log",
            str(log_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert shortdiag_proc.returncode != 0
    assert "route drift" in shortdiag_proc.stdout
    assert "SHORTDIAG_EVIDENCE=" not in shortdiag_proc.stdout

    summary = _formal_readiness_summary(train_log=log_path.name)
    summary["shortdiag_evidence"]["log_evidence"].update(
        {
            "train_log": log_path.name,
            "finite_loss_count": 1,
            "loss_min": 1.25,
            "loss_max": 1.25,
            "epoch_max": 1,
        }
    )
    summary_path = tmp_path / "globalrank_drift_formal_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "mdl_knot" / "validate_mdl_knot_launch_gate.py"),
            "--config",
            str(CONFIG_PATH),
            "--formal-readiness-summary",
            str(summary_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert proc.returncode != 0
    assert "formal training remains locked" in proc.stdout
    assert "route drift" in proc.stdout


def test_formal_readiness_gate_rejects_missing_or_synthetic_only_diagnostics(tmp_path):
    missing = tmp_path / "missing_formal.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "mdl_knot" / "validate_mdl_knot_launch_gate.py"),
            "--config",
            str(CONFIG_PATH),
            "--formal-readiness-summary",
            str(missing),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert proc.returncode != 0
    assert "formal readiness summary" in proc.stdout

    out_dir = tmp_path / "precheck"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "mdl_knot" / "audit_mdl_knot_pipeline_precheck.py"),
            "--out-dir",
            str(out_dir),
            "--overwrite",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "mdl_knot" / "validate_mdl_knot_launch_gate.py"),
            "--config",
            str(CONFIG_PATH),
            "--formal-readiness-summary",
            str(out_dir / "mdl_knot_precheck_summary.json"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert proc.returncode != 0
    assert "formal training remains locked" in proc.stdout
    assert "source_mode" in proc.stdout or "real_video_pipeline_diagnostics" in proc.stdout
