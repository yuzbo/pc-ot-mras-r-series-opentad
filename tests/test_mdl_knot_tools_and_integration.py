import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from opentad.acquisition.mdl_knot import (
    MDLKnotConfig,
    apply_mdl_knot_to_dense_window,
    build_synthetic_scout_curve,
    greedy_mdl_knot_select,
)


ROOT = Path(__file__).resolve().parents[1]
ROUTE_LABEL = "DIVERGENT_INNOVATION_EVENT_SURPRISE_DO_NOT_MERGE_WITH_C3"
CONFIG_PATH = ROOT / "configs" / "adatad" / "thumos" / "input_mdl_knot_dynamic_adapter_irregular_headv3.py"


def test_pipeline_mock_sets_frame_inds_before_decode_and_records_valid_k():
    dense_window = list(range(1000, 1128))
    curve = build_synthetic_scout_curve("short_islands", dense_t=len(dense_window))
    results = {"video_name": "video_test_0001", "total_frames": 2000}

    updated = apply_mdl_knot_to_dense_window(
        results=results,
        dense_window=dense_window,
        scout_curve=curve,
        config=MDLKnotConfig(route_label=ROUTE_LABEL, max_k=40),
    )

    assert updated["frame_inds"] == sorted(updated["frame_inds"])
    assert updated["frame_inds"] == [dense_window[pos] for pos in updated["mdl_knot_selected_positions"]]
    assert updated["mdl_knot_valid_k"] == len(updated["mdl_knot_selected_positions"])
    assert len(updated["masks"]) == updated["mdl_knot_valid_k"]
    assert updated["mdl_knot_sparse_meta"]["position_unit"] == "original_dense_time_index"


def test_fixed_adapter_bridge_pads_frame_inds_without_counting_padding_as_valid():
    dense_window = list(range(1000, 1128))
    curve = build_synthetic_scout_curve("short_islands", dense_t=len(dense_window))
    results = {"video_name": "video_test_bridge", "total_frames": 2000}

    updated = apply_mdl_knot_to_dense_window(
        results=results,
        dense_window=dense_window,
        scout_curve=curve,
        config=MDLKnotConfig(route_label=ROUTE_LABEL, max_k=40),
        adapter_target_len=64,
    )

    valid_k = updated["mdl_knot_valid_k"]
    assert len(updated["frame_inds"]) == 64
    assert len(updated["masks"]) == 64
    assert valid_k < 64
    assert sum(bool(v) for v in updated["masks"]) == valid_k
    assert updated["mdl_knot_padding_counts_as_valid"] is False
    assert updated["frame_inds"][:valid_k] == [dense_window[pos] for pos in updated["mdl_knot_selected_positions"]]
    assert updated["frame_inds"][valid_k:] == [updated["frame_inds"][valid_k - 1]] * (64 - valid_k)


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
    assert cfg["mdl_knot_acquisition"]["deploy_scout_source"] == "fallback_synthetic_precheck_only"


def test_real_loadframes_mdl_knot_branch_sets_sparse_frame_inds_before_decode():
    torch_probe = subprocess.run(
        [sys.executable, "-c", "import torch"],
        text=True,
        capture_output=True,
    )
    if torch_probe.returncode != 0:
        pytest.skip(f"real OpenTAD LoadFrames smoke skipped because torch import fails: {torch_probe.stderr[-240:]}")

    from opentad.datasets.transforms.end_to_end import LoadFrames

    curve = build_synthetic_scout_curve("short_islands", dense_t=128)
    loader = LoadFrames(
        method="mdl_knot_dynamic_subsample",
        method_base="sliding_window",
        scale_factor=1,
        mdl_knot_max_k=40,
        mdl_knot_target_weighted_error=0.01,
    )
    results = {
        "video_name": "video_test_0002",
        "total_frames": 128,
        "avg_fps": 30.0,
        "snippet_stride": 1,
        "window_size": 128,
        "feature_start_idx": 0,
        "feature_end_idx": 127,
        "mdl_knot_scout": {
            "p_action": curve.p_action,
            "uncertainty": curve.uncertainty,
            "temporal_change": curve.temporal_change,
            "persistence": curve.persistence,
            "source": "deploy_scout",
        },
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


def test_launch_gate_locked_until_precheck_passes(tmp_path):
    missing = tmp_path / "missing_summary.json"
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "mdl_knot" / "validate_mdl_knot_launch_gate.py"),
        "--route-label",
        ROUTE_LABEL,
        "--precheck-summary",
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
        "--route-label",
        ROUTE_LABEL,
        "--precheck-summary",
        str(out_dir / "mdl_knot_precheck_summary.json"),
    ]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr
    assert "PRECHECK_ONLY_REQUEST_ALLOWED" in proc.stdout


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
            "stage_commit_push": True,
        },
        "no_claims": {"mAP": True, "runtime": True, "FLOPs": True, "deploy": True, "paper": True},
        "cases": {"a": {"valid_k": 4}, "b": {"valid_k": 7}},
        "config_evidence": {
            "route_label": ROUTE_LABEL,
            "dataset_pipelines_use_mdl": False,
            "load_methods": {"train": "random_fixed_subsample", "val": "random_fixed_subsample", "test": "random_fixed_subsample"},
            "forbidden_route_token_hits": [],
            "deploy_scout_source": "fallback_synthetic_precheck_only",
            "no_metric_runtime_deploy_claims": True,
        },
    }
    summary_path = tmp_path / "bad_summary.json"
    summary_path.write_text(json.dumps(bad_summary), encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "mdl_knot" / "validate_mdl_knot_launch_gate.py"),
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
            "stage_commit_push": True,
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
            "deploy_scout_source": "fallback_synthetic_precheck_only",
            "real_scout_unavailable": True,
            "no_metric_runtime_deploy_claims": True,
        },
    }
    summary_path = tmp_path / "gate_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "mdl_knot" / "validate_mdl_knot_launch_gate.py"),
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
