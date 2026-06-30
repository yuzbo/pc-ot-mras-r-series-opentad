import math

import numpy as np


ADAPTER_FIXED_LENGTH_PADDED_BRIDGE = "adapter_fixed_length_padded_bridge"


def build_detector_feature_centers_from_raw(selected_positions, feature_stride=1):
    positions = np.asarray(selected_positions, dtype=np.float32).reshape(-1)
    feature_stride = int(max(feature_stride, 1))
    if positions.size == 0:
        raise ValueError("RBA-RBR detector feature centers require at least one raw selected position")
    centers = []
    for start in range(0, int(positions.size), feature_stride):
        centers.append(float(np.mean(positions[start : start + feature_stride])))
    return np.asarray(centers, dtype=np.float32)


def build_adapter_fixed_length_padded_bridge(selected_positions, selected_frame_inds, target_frame_num, dense_T, feature_stride=1):
    positions = np.asarray(selected_positions, dtype=np.int64).reshape(-1)
    frame_inds = np.asarray(selected_frame_inds, dtype=np.int64).reshape(-1)
    target = int(target_frame_num)
    dense_T = int(dense_T)
    if positions.size == 0:
        raise ValueError("RBA-RBR adapter bridge requires at least one selected frame")
    if positions.size != frame_inds.size:
        raise ValueError("selected_positions and selected_frame_inds must have the same length")
    if positions.size > target:
        raise ValueError("valid_k cannot exceed target_frame_num for fixed-length adapter bridge")
    if np.any(positions < 0) or np.any(positions >= dense_T):
        raise ValueError("selected_positions out of dense_T range")
    pad_count = int(target - positions.size)
    if pad_count > 0:
        padded_positions = np.concatenate([positions, np.repeat(positions[-1], pad_count)])
        padded_frame_inds = np.concatenate([frame_inds, np.repeat(frame_inds[-1], pad_count)])
    else:
        padded_positions = positions.copy()
        padded_frame_inds = frame_inds.copy()
    valid_raw_mask = np.zeros(target, dtype=np.bool_)
    valid_raw_mask[: positions.size] = True
    feature_stride = int(max(feature_stride, 1))
    detector_mask_len = int(math.ceil(float(target) / float(feature_stride)))
    detector_feature_valid_k = int(math.ceil(float(positions.size) / float(feature_stride)))
    detector_valid_mask = np.zeros(detector_mask_len, dtype=np.bool_)
    detector_valid_mask[:detector_feature_valid_k] = True
    return {
        "adapter_bridge_mode": ADAPTER_FIXED_LENGTH_PADDED_BRIDGE,
        "adapter_target_frame_num": target,
        "adapter_input_frame_count": int(padded_frame_inds.size),
        "adapter_padded_frame_inds": padded_frame_inds.astype(np.int64),
        "adapter_padded_positions": padded_positions.astype(np.int64),
        "adapter_valid_raw_mask": valid_raw_mask,
        "adapter_padding_duplicate_count": pad_count,
        "adapter_padding_counts_as_valid": False,
        "detector_mask_len": detector_mask_len,
        "detector_feature_valid_k": detector_feature_valid_k,
        "detector_feature_positions": build_detector_feature_centers_from_raw(positions, feature_stride=feature_stride),
        "detector_valid_mask": detector_valid_mask,
        "feature_stride": feature_stride,
    }
