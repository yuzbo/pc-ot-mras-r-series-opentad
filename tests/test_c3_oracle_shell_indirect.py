import json
from pathlib import Path

import numpy as np
import pytest
import torch
from mmengine.config import Config

from opentad.datasets.transforms.end_to_end import LoadFrames
from opentad.models.utils.post_processing.utils import selected_axis_to_dense_axis
from opentad.datasets.transforms.pseudo_boundary import (
    load_coarse_score_cache,
    select_coarse_oracle_shell_positions,
    validate_coarse_score_cache_manifest,
)
from tools.export_c3_coarse_score_cache_from_selector_checkpoint import ScoreAccumulator, _window_start_snippet
from tools.build_c3_coarse_score_cache_from_records import build_cache
from tools.diagnose_c3_oracle_shell_indirect_distribution import (
    compare_oracle_and_indirect_window,
    iter_annotation_cache_comparisons,
)
from tools.train import _should_run_epoch_event
from tools.validate_c3_oracle_shell_indirect import validate_config


ROOT = Path(__file__).resolve().parents[1]
PRECHECK_CONFIG = ROOT / "configs/adatad/thumos/c3_oracle_shell_indirect_precheck.py"
FULL_CONFIG = ROOT / "configs/adatad/thumos/c3_oracle_shell_indirect_full_train.py"
PRECHECK_LAUNCHER = ROOT / "scripts/run_c3_oracle_shell_indirect_precheck_gpu1.sh"
FULL_LAUNCHER = ROOT / "scripts/run_c3_oracle_shell_indirect_full_train_gpu1.sh"


def _write_cache(tmp_path, *, uses_gt=False, axis="global_snippet_index", video_name="video_test_0000001"):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)
    scores = np.asarray([0.01, 0.03, 0.08, 0.35, 0.80, 0.72, 0.30, 0.06, 0.02, 0.40, 0.86, 0.82], dtype=np.float32)
    np.savez(cache_dir / f"{video_name}.npz", action_score=scores, axis=axis, video_name=video_name)
    manifest = {
        "schema_version": 1,
        "route_labels": ["C3_MAINLINE_OPTIMIZATION", "C3_ORIGINAL_OPTIMIZATION_ROUTE"],
        "score_source": "unit_test_coarse_classifier",
        "uses_gt": uses_gt,
        "axis": axis,
        "videos": {video_name: {"file": f"{video_name}.npz", "num_frames": int(scores.size)}},
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return cache_dir


def _base_results():
    return {
        "video_name": "video_test_0000001",
        "total_frames": 12,
        "avg_fps": 30.0,
        "snippet_stride": 1,
        "gt_segments": np.asarray([[3.0, 7.0], [9.0, 12.0]], dtype=np.float32),
        "gt_labels": np.asarray([1, 2], dtype=np.int32),
    }


def test_coarse_score_cache_manifest_rejects_gt_and_axis_mismatch(tmp_path):
    gt_cache = _write_cache(tmp_path / "gt", uses_gt=True)
    with pytest.raises(AssertionError, match="uses_gt"):
        validate_coarse_score_cache_manifest(gt_cache / "manifest.json", expected_axis="global_snippet_index")

    bad_axis_cache = _write_cache(tmp_path / "axis", axis="selected_axis")
    with pytest.raises(AssertionError, match="axis"):
        validate_coarse_score_cache_manifest(bad_axis_cache / "manifest.json", expected_axis="global_snippet_index")


def test_build_cache_defaults_to_global_snippet_axis(tmp_path):
    input_jsonl = tmp_path / "records.jsonl"
    input_jsonl.write_text(
        json.dumps({"video_name": "video_test_0000001", "action_score": [0.1, 0.9, 0.2]}) + "\n",
        encoding="utf-8",
    )

    manifest = build_cache(input_jsonl, tmp_path / "cache", "unit-test-cache", overwrite=True)

    assert manifest["axis"] == "global_snippet_index"
    loaded, loaded_manifest = load_coarse_score_cache(tmp_path / "cache", "video_test_0000001")
    assert loaded_manifest["axis"] == "global_snippet_index"
    assert loaded["action_score"].tolist() == pytest.approx([0.1, 0.9, 0.2])


def test_score_accumulator_averages_overlapping_global_windows_and_keeps_manifest_contract(tmp_path):
    acc = ScoreAccumulator("video_test_0000001")
    acc.add_window(global_indices=np.asarray([0, 1, 2], dtype=np.int64), action_logits=np.asarray([0.0, 2.0, -2.0]))
    acc.add_window(global_indices=np.asarray([2, 3], dtype=np.int64), action_logits=np.asarray([2.0, 0.0]))
    record = acc.finalize()

    assert record["action_logit"].shape == (4,)
    assert record["observed_fraction"] == 1.0
    assert record["count"].tolist() == [1, 1, 2, 1]
    assert record["action_logit"][2] == pytest.approx(0.0)


def test_window_start_snippet_uses_frame_stride_and_offset():
    meta = {"window_start_frame": 64, "snippet_stride": 16, "offset_frames": 0}
    assert _window_start_snippet(meta) == 4


def test_score_based_group_selection_uses_scores_without_gt_and_returns_sorted_unique_positions():
    action_score = np.asarray([0.01, 0.02, 0.05, 0.52, 0.90, 0.88, 0.47, 0.04, 0.03, 0.45, 0.84, 0.82])

    keep = select_coarse_oracle_shell_positions(
        valid_len=12,
        target_frame_num=6,
        action_score=action_score,
        sample_key="unit-no-gt",
        min_action_score=0.4,
        transition_top_fraction=0.5,
        transition_radius=1,
    )

    assert keep.tolist() == sorted(set(keep.tolist()))
    assert keep.dtype == np.int64
    assert keep.size == 6
    assert keep.min() >= 0
    assert keep.max() < 12
    assert any(pos in keep for pos in [3, 6, 9])


def test_oracle_shell_indirect_matches_oracle_shell_contract_without_gt_selection(tmp_path):
    cache_dir = _write_cache(tmp_path)
    oracle = LoadFrames(
        num_clips=1,
        method="oracle_action_boundary_subsample",
        method_base="random_trunc",
        target_len=6,
        source_len=12,
        keep_ratio=0.5,
        trunc_thresh=0.0,
        crop_ratio=None,
        scale_factor=1,
        oracle_boundary_radius=1,
    )
    indirect = LoadFrames(
        num_clips=1,
        method="coarse_score_oracle_shell_subsample",
        method_base="random_trunc",
        target_len=6,
        source_len=12,
        keep_ratio=0.5,
        trunc_thresh=0.0,
        crop_ratio=None,
        scale_factor=1,
        oracle_boundary_radius=1,
        coarse_score_cache_dir=str(cache_dir),
    )

    oracle_results = oracle(dict(_base_results()))
    indirect_results = indirect(dict(_base_results()))

    assert indirect_results["frame_inds"].shape == oracle_results["frame_inds"].shape == (6,)
    assert indirect_results["masks"].shape == oracle_results["masks"].shape == (6,)
    assert indirect_results["masks"].dtype == oracle_results["masks"].dtype
    assert indirect_results["gt_segments"].ndim == oracle_results["gt_segments"].ndim == 2
    assert indirect_results["gt_segments"].shape[1] == oracle_results["gt_segments"].shape[1] == 2
    assert "irregular_selected_positions" in indirect_results
    assert "irregular_selected_valid_len" in indirect_results
    assert indirect_results["coarse_oracle_shell_uses_gt_for_selection"] is False


def test_diagnostic_oracle_comparison_does_not_mutate_frame_indices(tmp_path):
    cache_dir = _write_cache(tmp_path)
    scores, _manifest = load_coarse_score_cache(cache_dir, "video_test_0000001")
    frame_inds = np.asarray([0, 3, 4, 6, 9, 10], dtype=np.int64)
    before = frame_inds.copy()

    record = compare_oracle_and_indirect_window(
        valid_len=12,
        target_frame_num=6,
        gt_segments=np.asarray([[3.0, 7.0], [9.0, 12.0]], dtype=np.float32),
        indirect_positions=frame_inds,
        action_score=scores["action_score"],
        boundary_radius=1,
        sample_key="diag-only",
    )

    assert np.array_equal(frame_inds, before)
    assert record["diagnostic_only"] is True
    assert record["selection_mutated"] is False
    assert 0.0 <= record["jaccard"] <= 1.0
    assert "oracle_only_distance_to_indirect" in record
    assert "indirect_only_distance_to_oracle" in record


def test_annotation_cache_distribution_diagnostic_is_offline_only(tmp_path):
    cache_dir = _write_cache(tmp_path)
    ann_file = tmp_path / "ann.json"
    ann_file.write_text(
        json.dumps(
            {
                "database": {
                    "video_test_0000001": {
                        "subset": "validation",
                        "duration": 12.0,
                        "frame": 12,
                        "annotations": [
                            {"segment": [3.0, 7.0], "label": "A"},
                            {"segment": [9.0, 11.0], "label": "B"},
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    records = iter_annotation_cache_comparisons(
        ann_file=ann_file,
        cache_dir=cache_dir,
        subset_name=["validation"],
        target_frame_num=6,
        window_size=12,
        window_overlap_ratio=0.0,
        feature_stride=1,
        sample_stride=1,
    )

    assert len(records) == 1
    assert records[0]["diagnostic_only"] is True
    assert records[0]["official_map_claim"] is False
    assert records[0]["video_name"] == "video_test_0000001"
    assert "indirect_boundary_recall" in records[0]


def test_c3_oracle_shell_indirect_configs_pass_validator_and_reject_forbidden_tokens(tmp_path):
    validate_config(PRECHECK_CONFIG)
    validate_config(FULL_CONFIG)

    cfg = Config.fromfile(FULL_CONFIG)
    cfg.c3_route_mixed_note = "forbidden teacher raw_prediction_cache pqr bh_sdc divergent"
    bad_config = tmp_path / "bad_c3_oracle_shell_indirect.py"
    cfg.dump(bad_config)

    with pytest.raises(AssertionError, match="Forbidden"):
        validate_config(bad_config)


def test_full_train_uses_epoch2_then_every5_validation_schedule():
    cfg = Config.fromfile(FULL_CONFIG)

    assert cfg.workflow.val_start_epoch == 2
    assert cfg.workflow.val_eval_epochs == [2]
    assert cfg.workflow.val_eval_interval == 5
    assert cfg.workflow.val_eval_interval_anchor_epoch == 2
    actual_eval_epochs = [
        epoch
        for epoch in range(cfg.workflow.end_epoch)
        if _should_run_epoch_event(
            epoch,
            cfg.workflow.val_eval_interval,
            start_epoch=cfg.workflow.val_start_epoch,
            explicit_epochs=cfg.workflow.val_eval_epochs,
            anchor_epoch=cfg.workflow.val_eval_interval_anchor_epoch,
        )
    ]
    assert actual_eval_epochs == [2, 7, 12, 17, 22, 27, 32, 37, 42, 47, 52, 57]


def test_collect_preserves_oracle_shell_meta_and_test_split_does_not_pass_gt():
    cfg = Config.fromfile(FULL_CONFIG)
    required_meta = {
        "irregular_selected_positions",
        "irregular_selected_valid_len",
        "irregular_native_axis",
        "coarse_oracle_shell_score_source",
        "coarse_oracle_shell_uses_gt_for_selection",
        "coarse_oracle_shell_score_axis",
        "coarse_oracle_shell_selected_positions",
    }
    for split in ("train", "val", "test"):
        collect = next(step for step in cfg.dataset[split].pipeline if step["type"] == "Collect")
        assert required_meta.issubset(set(collect["meta_keys"]))
    test_collect = next(step for step in cfg.dataset.test.pipeline if step["type"] == "Collect")
    assert "gt_segments" not in test_collect["keys"]
    assert "gt_labels" not in test_collect["keys"]


def test_selected_axis_segments_are_mapped_back_to_dense_axis():
    segments = torch.tensor([[0.0, 2.0], [1.0, 3.0]], dtype=torch.float32)
    meta = {
        "irregular_selected_positions": [0.0, 4.0, 8.0],
        "irregular_selected_valid_len": 12.0,
        "irregular_native_axis": False,
    }
    dense = selected_axis_to_dense_axis(segments, meta)
    assert torch.allclose(dense, torch.tensor([[0.0, 8.0], [4.0, 12.0]]))


def test_gpu1_launchers_fail_closed_and_preserve_parent_hold():
    for launcher in [PRECHECK_LAUNCHER, FULL_LAUNCHER]:
        text = launcher.read_text(encoding="utf-8")
        assert "CUDA_VISIBLE_DEVICES=1" in text
        assert "GPU1" in text
        assert "1118197" in text
        assert "scancel" not in text
        assert "release" not in text.lower()
        assert "tools/train.py" in text
        assert "validate_c3_oracle_shell_indirect.py" in text
        assert "C3_COARSE_SCORE_CACHE_DIR" in text
    full = FULL_LAUNCHER.read_text(encoding="utf-8")
    assert "C3_ORACLE_SHELL_INDIRECT_FULLTRAIN_UNLOCK" in full
    assert "CONFIRMED" in full
    assert "exit 2" in full
