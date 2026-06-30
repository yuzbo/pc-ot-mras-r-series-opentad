import numpy as np
import pytest
import subprocess
import sys
from pathlib import Path

from opentad.acquisition.bvr_twb.adapter_bridge import (
    ADAPTER_FIXED_LENGTH_PADDED_BRIDGE,
    build_adapter_fixed_length_padded_bridge,
    build_detector_feature_centers_from_raw,
)

_torch_probe = subprocess.run(
    [sys.executable, "-c", "import torch"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
)

if _torch_probe.returncode == 0:
    import torch
    from opentad.datasets.transforms.end_to_end import LoadFrames
    from opentad.models.detectors.irregular_actionformer import IrregularActionFormer
    from opentad.models.utils.post_processing import convert_to_seconds

    TORCH_IMPORT_ERROR = None
else:
    torch = None
    LoadFrames = None
    IrregularActionFormer = None
    convert_to_seconds = None
    TORCH_IMPORT_ERROR = RuntimeError((_torch_probe.stderr or _torch_probe.stdout).strip().splitlines()[-1])


def _dense_384_results(split="train", with_gt=True):
    dense_len = 384
    x = np.arange(dense_len, dtype=np.float32)
    actionness = (
        0.04
        + 0.58 * np.exp(-((x - 116.0) ** 2) / (2.0 * 12.0**2))
        + 0.47 * np.exp(-((x - 254.0) ** 2) / (2.0 * 19.0**2))
        + 0.11 * np.sin(x / 17.0) ** 2
    )
    out = {
        "video_name": f"bvr_geometry_{split}",
        "data_path": "mock",
        "total_frames": 768,
        "avg_fps": 30.0,
        "fps": 30.0,
        "duration": 768.0 / 30.0,
        "snippet_stride": 1,
        "window_size": dense_len,
        "feature_start_idx": 0,
        "feature_end_idx": dense_len - 1,
        "split": split,
        "bvr_twb_preview_actionness": np.clip(actionness, 0.0, 1.0).astype(np.float32),
    }
    if with_gt:
        out["gt_segments"] = np.asarray([[104.0, 143.0], [238.0, 290.0]], dtype=np.float32)
        out["gt_labels"] = np.asarray([1, 3], dtype=np.int32)
    return out


def _dense_384_loader(split="train", train_value_labels=False):
    return LoadFrames(
        num_clips=1,
        method="bvr_twb_dynamic_subsample",
        method_base="sliding_window",
        keep_ratio=0.5,
        target_len=192,
        scale_factor=1,
        remap_gt_to_selected_axis=False,
        bvr_twb_split=split,
        bvr_twb_min_keep=72,
        bvr_twb_max_keep=160,
        bvr_twb_max_gap=32,
        bvr_twb_scaffold_k=4,
        bvr_twb_train_value_labels=train_value_labels,
        bvr_twb_feature_stride=2,
        bvr_twb_adapter_bridge_mode=ADAPTER_FIXED_LENGTH_PADDED_BRIDGE,
        bvr_twb_scout_source="deploy_visible_raw_or_metadata_scout",
        bvr_twb_require_deploy_visible_scout=True,
        bvr_twb_allow_diagnostic_preview_fallback=False,
        bvr_twb_scout_sample_count=24,
        bvr_twb_value_mode="deploy_heuristic_voi",
    )


def _require_torch():
    if TORCH_IMPORT_ERROR is not None:
        pytest.skip(f"torch/OpenTAD pipeline unavailable locally: {TORCH_IMPORT_ERROR}")


def test_loader_dense384_target192_outputs_detector_feature_axis_and_no_val_labels():
    _require_torch()

    train = _dense_384_loader(split="train", train_value_labels=True)(_dense_384_results("train", with_gt=True))
    ledger = train["bvr_twb_ledger"]
    raw_positions = np.asarray(ledger["raw_selected_positions"], dtype=np.float32)
    detector_positions = np.asarray(train["bvr_twb_detector_feature_positions"], dtype=np.float32)
    expected_detector_positions = build_detector_feature_centers_from_raw(raw_positions, feature_stride=2)

    assert train["frame_inds"].shape == (192,)
    assert ledger["adapter_input_frame_count"] == 192
    assert ledger["dense_T"] == 384
    assert 0 < ledger["valid_k"] < 192
    assert ledger["adapter_padding_duplicate_count"] == 192 - ledger["valid_k"]
    assert ledger["adapter_padding_counts_as_valid"] is False
    assert int(train["masks"].sum().item()) == ledger["detector_feature_valid_k"]
    assert train["masks"].shape[0] == ledger["detector_mask_len"] == 96
    assert detector_positions.shape[0] == ledger["detector_feature_valid_k"]
    assert np.allclose(detector_positions, expected_detector_positions)
    assert np.allclose(train["irregular_selected_positions"], detector_positions)
    assert train["bvr_twb_detector_feature_valid_len"] == 384.0
    assert train["irregular_selected_valid_len"] == 384.0
    assert len(set(np.diff(raw_positions).astype(int).tolist())) > 1
    assert train["frame_inds"][-1] == train["frame_inds"][ledger["valid_k"] - 1]
    assert train["bvr_twb_train_value_labels"]

    val = _dense_384_loader(split="val", train_value_labels=False)(_dense_384_results("val", with_gt=True))
    assert val["bvr_twb_train_value_labels"] == []
    assert val["bvr_twb_ledger"]["value_labels_used_at_test"] is False


def test_detector_bvr_grid_prefers_detector_feature_positions_and_fails_closed():
    _require_torch()
    masks = torch.tensor([[True, True, True, False, False, False]])
    detector = object.__new__(IrregularActionFormer)
    meta = {
        "video_name": "bvr_prefers_detector_axis",
        "irregular_selected_positions": np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        "irregular_selected_valid_len": 6.0,
        "irregular_native_axis": True,
        "bvr_twb_ledger": {"method": "bvr_twb_dynamic_subsample"},
        "bvr_twb_raw_selected_positions": np.asarray([0.0, 7.0, 31.0, 70.0, 129.0, 191.0], dtype=np.float32),
        "bvr_twb_raw_selected_valid_len": 384.0,
        "bvr_twb_detector_feature_positions": np.asarray([3.5, 50.5, 160.0], dtype=np.float32),
        "bvr_twb_detector_feature_valid_len": 384.0,
    }

    grid = detector._temporal_grid_from_metas([meta], masks)

    assert torch.allclose(grid["center"][0, :3], torch.tensor([3.5, 50.5, 160.0]))
    assert grid["valid_mask"][0].tolist() == [True, True, True, False, False, False]
    assert grid["fresh_mask"][0].tolist() == [True, True, True, False, False, False]
    assert float(grid["cell_right"][0, 2]) == pytest.approx(224.0)

    missing = dict(meta)
    missing.pop("bvr_twb_detector_feature_positions")
    with pytest.raises(ValueError, match="bvr_twb_detector_feature_positions"):
        detector._temporal_grid_from_metas([missing], masks)


def test_remap_false_gt_native_axis_matches_bvr_detector_grid_not_selected_index():
    _require_torch()

    transformed = _dense_384_loader(split="train", train_value_labels=False)(_dense_384_results("train", with_gt=True))
    detector = object.__new__(IrregularActionFormer)
    grid = detector._temporal_grid_from_metas([transformed], transformed["masks"][None])

    assert transformed["irregular_native_axis"] is True
    assert np.allclose(transformed["gt_segments"], np.asarray([[104.0, 143.0], [238.0, 290.0]], dtype=np.float32))
    assert transformed["gt_segments"].max() > transformed["bvr_twb_ledger"]["valid_k"]
    assert torch.allclose(
        grid["center"][0, : transformed["masks"].sum()],
        torch.as_tensor(transformed["bvr_twb_detector_feature_positions"], dtype=torch.float32),
    )
    assert grid["center"][0, transformed["masks"].sum() - 1] < transformed["bvr_twb_detector_feature_valid_len"]


def test_bvr_native_axis_postprocessing_seconds_uses_dense_coordinates_directly():
    _require_torch()
    segments = torch.tensor([[96.0, 144.0], [238.0, 290.0]], dtype=torch.float32)
    meta = {
        "video_name": "bvr_seconds",
        "fps": 30.0,
        "duration": 20.0,
        "snippet_stride": 2,
        "offset_frames": 4,
        "window_start_frame": 30,
        "irregular_native_axis": True,
        "irregular_selected_positions": np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
        "irregular_selected_valid_len": 3.0,
        "bvr_twb_detector_feature_positions": np.asarray([96.0, 144.0, 238.0], dtype=np.float32),
        "bvr_twb_detector_feature_valid_len": 384.0,
    }

    seconds = convert_to_seconds(segments.clone(), meta)

    expected = (segments * 2.0 + 30.0 + 4.0) / 30.0
    expected = expected.clamp(min=0.0, max=meta["duration"])
    assert torch.allclose(seconds, expected)


def test_forced_uniform_bvr_bridge_keeps_head_eval_geometry_sane():
    _require_torch()
    raw_positions = np.round(np.linspace(0, 383, 96)).astype(np.int64)
    bridge = build_adapter_fixed_length_padded_bridge(
        selected_positions=raw_positions,
        selected_frame_inds=raw_positions * 2,
        target_frame_num=192,
        dense_T=384,
        feature_stride=2,
    )
    masks = torch.as_tensor(bridge["detector_valid_mask"], dtype=torch.bool)[None]
    detector = object.__new__(IrregularActionFormer)
    meta = {
        "video_name": "bvr_forced_uniform_bridge",
        "irregular_selected_positions": bridge["detector_feature_positions"],
        "irregular_selected_valid_len": 384.0,
        "irregular_native_axis": True,
        "bvr_twb_ledger": {"method": "bvr_twb_dynamic_subsample"},
        "bvr_twb_raw_selected_positions": raw_positions.astype(np.float32),
        "bvr_twb_raw_selected_valid_len": 384.0,
        "bvr_twb_detector_feature_positions": bridge["detector_feature_positions"],
        "bvr_twb_detector_feature_valid_len": 384.0,
    }

    grid = detector._temporal_grid_from_metas([meta], masks)
    seconds = convert_to_seconds(
        torch.tensor([[48.0, 96.0], [192.0, 240.0]], dtype=torch.float32),
        {
            **meta,
            "fps": 30.0,
            "duration": 20.0,
            "snippet_stride": 1,
            "offset_frames": 0,
            "window_start_frame": 0,
        },
    )

    assert bridge["adapter_input_frame_count"] == 192
    assert bridge["adapter_padding_duplicate_count"] == 96
    assert int(masks.sum().item()) == 48
    assert torch.allclose(grid["center"][0, :48], torch.as_tensor(bridge["detector_feature_positions"]))
    assert torch.allclose(seconds, torch.tensor([[1.6, 3.2], [6.4, 8.0]], dtype=torch.float32))


def test_bvr_geometry_source_contracts_are_fail_closed_without_torch():
    root = Path(__file__).resolve().parents[1]
    detector_text = (root / "opentad/models/detectors/irregular_actionformer.py").read_text(encoding="utf-8")
    post_text = (root / "opentad/models/utils/post_processing/utils.py").read_text(encoding="utf-8")

    assert "def _bvr_twb_temporal_grid_from_meta" in detector_text
    assert '"bvr_twb_detector_feature_positions", "bvr_twb_detector_feature_valid_len"' in detector_text
    assert "BVR-TWB detector temporal grid requires" in detector_text
    assert "mask true count must equal detector feature position count" in detector_text
    assert "grids.append(self._bvr_twb_temporal_grid_from_meta(meta, mask))" in detector_text
    assert "bvr_twb_detector_feature_positions" in post_text
    assert "BVR-TWB post-processing requires irregular_native_axis=True" in post_text


def test_bvr_geometry_validator_runs_source_and_numpy_contracts_without_training():
    from tools.bvr_twb.validate_bvr_twb_geometry_contracts import run_geometry_contracts

    summary = run_geometry_contracts(require_torch=False)

    assert summary["validator"] == "bvr_twb_geometry_contracts"
    assert summary["source_contract"] == "passed"
    assert summary["numpy_bridge_contract"] == "passed"
    assert summary["no_training"] is True
    assert summary["no_metric_claim"] is True
    assert summary["full_training_unlocked"] is False
