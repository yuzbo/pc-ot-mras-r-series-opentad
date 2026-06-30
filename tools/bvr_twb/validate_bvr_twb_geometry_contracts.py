import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.acquisition.bvr_twb.adapter_bridge import (  # noqa: E402
    ADAPTER_FIXED_LENGTH_PADDED_BRIDGE,
    build_adapter_fixed_length_padded_bridge,
    build_detector_feature_centers_from_raw,
)


def _probe_torch():
    proc = subprocess.run(
        [sys.executable, "-c", "import torch"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode == 0:
        return None
    return (proc.stderr or proc.stdout).strip().splitlines()[-1]


def _validate_source_contracts():
    detector_text = (ROOT / "opentad/models/detectors/irregular_actionformer.py").read_text(encoding="utf-8")
    post_text = (ROOT / "opentad/models/utils/post_processing/utils.py").read_text(encoding="utf-8")
    required_detector_tokens = [
        "def _bvr_twb_temporal_grid_from_meta",
        '"bvr_twb_detector_feature_positions", "bvr_twb_detector_feature_valid_len"',
        "BVR-TWB detector temporal grid requires",
        "mask true count must equal detector feature position count",
        "grids.append(self._bvr_twb_temporal_grid_from_meta(meta, mask))",
    ]
    required_post_tokens = [
        "bvr_twb_detector_feature_positions",
        "BVR-TWB post-processing requires irregular_native_axis=True",
    ]
    missing = [token for token in required_detector_tokens if token not in detector_text]
    missing += [token for token in required_post_tokens if token not in post_text]
    if missing:
        raise ValueError(f"BVR-TWB geometry source contract missing tokens: {missing}")
    return {"source_contract": "passed"}


def _validate_numpy_bridge_contract():
    raw_positions = np.round(np.linspace(0, 383, 96)).astype(np.int64)
    bridge = build_adapter_fixed_length_padded_bridge(
        selected_positions=raw_positions,
        selected_frame_inds=raw_positions * 2,
        target_frame_num=192,
        dense_T=384,
        feature_stride=2,
    )
    expected_centers = build_detector_feature_centers_from_raw(raw_positions, feature_stride=2)
    if bridge["adapter_bridge_mode"] != ADAPTER_FIXED_LENGTH_PADDED_BRIDGE:
        raise ValueError("adapter bridge mode mismatch")
    if bridge["adapter_input_frame_count"] != 192:
        raise ValueError("adapter bridge did not preserve target input length")
    if int(bridge["adapter_valid_raw_mask"].sum()) != 96:
        raise ValueError("adapter valid raw mask must count only selected raw observations")
    if bool(bridge["adapter_padding_counts_as_valid"]):
        raise ValueError("adapter padding duplicate must remain invalid")
    if int(bridge["detector_valid_mask"].sum()) != int(expected_centers.shape[0]):
        raise ValueError("detector valid mask must count feature centers, not padded raw inputs")
    if not np.allclose(bridge["detector_feature_positions"], expected_centers):
        raise ValueError("detector feature positions must be grouped centers from raw selected positions")
    return {
        "numpy_bridge_contract": "passed",
        "dense_T": 384,
        "target_frame_num": 192,
        "raw_valid_k": 96,
        "detector_feature_valid_k": int(expected_centers.shape[0]),
    }


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
        "video_name": f"bvr_geometry_validator_{split}",
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


def _validate_torch_runtime_contract():
    import torch

    from opentad.datasets.transforms.end_to_end import LoadFrames
    from opentad.models.detectors.irregular_actionformer import IrregularActionFormer
    from opentad.models.utils.post_processing import convert_to_seconds

    loader = LoadFrames(
        num_clips=1,
        method="bvr_twb_dynamic_subsample",
        method_base="sliding_window",
        keep_ratio=0.5,
        target_len=192,
        scale_factor=1,
        remap_gt_to_selected_axis=False,
        bvr_twb_split="train",
        bvr_twb_min_keep=72,
        bvr_twb_max_keep=160,
        bvr_twb_max_gap=32,
        bvr_twb_scaffold_k=4,
        bvr_twb_train_value_labels=False,
        bvr_twb_feature_stride=2,
        bvr_twb_adapter_bridge_mode=ADAPTER_FIXED_LENGTH_PADDED_BRIDGE,
        bvr_twb_scout_source="deploy_visible_raw_or_metadata_scout",
        bvr_twb_require_deploy_visible_scout=True,
        bvr_twb_allow_diagnostic_preview_fallback=False,
        bvr_twb_scout_sample_count=24,
        bvr_twb_value_mode="deploy_heuristic_voi",
    )
    transformed = loader(_dense_384_results("train", with_gt=True))
    detector = object.__new__(IrregularActionFormer)
    grid = detector._temporal_grid_from_metas([transformed], transformed["masks"][None])
    valid_count = int(transformed["masks"].sum().item())
    if not torch.allclose(
        grid["center"][0, :valid_count],
        torch.as_tensor(transformed["bvr_twb_detector_feature_positions"], dtype=torch.float32),
    ):
        raise ValueError("detector did not use BVR-TWB detector feature positions")
    if not np.allclose(transformed["gt_segments"], np.asarray([[104.0, 143.0], [238.0, 290.0]], dtype=np.float32)):
        raise ValueError("remap_gt_to_selected_axis=False did not preserve native GT coordinates")

    seconds = convert_to_seconds(
        torch.tensor([[96.0, 144.0]], dtype=torch.float32),
        {
            **transformed,
            "fps": 30.0,
            "duration": 20.0,
            "snippet_stride": 2,
            "offset_frames": 4,
            "window_start_frame": 30,
        },
    )
    expected = torch.tensor([[(96.0 * 2.0 + 34.0) / 30.0, (144.0 * 2.0 + 34.0) / 30.0]], dtype=torch.float32)
    if not torch.allclose(seconds, expected):
        raise ValueError("BVR-TWB native-axis post-processing seconds conversion mismatch")
    return {
        "torch_runtime_contract": "passed",
        "loader_frame_count": int(transformed["frame_inds"].shape[0]),
        "detector_feature_valid_k": valid_count,
    }


def run_geometry_contracts(require_torch=False):
    summary = {
        "validator": "bvr_twb_geometry_contracts",
        "platform_system": platform.system(),
        "no_training": True,
        "no_video_decode": True,
        "no_metric_claim": True,
        "full_training_unlocked": False,
    }
    summary.update(_validate_source_contracts())
    summary.update(_validate_numpy_bridge_contract())
    torch_error = _probe_torch()
    if torch_error is None:
        summary.update(_validate_torch_runtime_contract())
        summary["torch_runtime_skipped"] = False
    else:
        if require_torch:
            raise RuntimeError(f"torch runtime contract required but torch import failed: {torch_error}")
        summary["torch_runtime_skipped"] = True
        summary["torch_runtime_skip_reason"] = torch_error
    return summary


def main():
    parser = argparse.ArgumentParser(description="Validate BVR-TWB local geometry contracts without training.")
    parser.add_argument("--require-torch", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_geometry_contracts(require_torch=args.require_torch), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
