import os
import pickle
import torch
import torch.nn.functional as F


def boundary_choose(score):
    mask_high = score > score.max(dim=1, keepdim=True)[0] * 0.5
    mask_peak = score == F.max_pool1d(score, kernel_size=3, stride=1, padding=1)
    mask = mask_peak | mask_high
    return mask


def save_predictions(predictions, metas, folder):
    for idx in range(len(metas)):
        video_name = metas[idx]["video_name"]

        file_path = os.path.join(folder, f"{video_name}.pkl")
        prediction = [data[idx] for data in predictions]
        with open(file_path, "wb") as outfile:
            pickle.dump(prediction, outfile, pickle.HIGHEST_PROTOCOL)


def load_single_prediction(metas, folder):
    """Should not be used for sliding window. Since we saved the files with video name, and sliding window will have multiple files with the same name."""
    predictions = []
    for idx in range(len(metas)):
        video_name = metas[idx]["video_name"]
        file_path = os.path.join(folder, f"{video_name}.pkl")
        with open(file_path, "rb") as infile:
            prediction = pickle.load(infile)
        predictions.append(prediction)

    batched_predictions = []
    for i in range(len(predictions[0])):
        data = torch.stack([prediction[i] for prediction in predictions])
        batched_predictions.append(data)
    return batched_predictions


def load_predictions(metas, infer_cfg):
    if "fuse_list" in infer_cfg.keys():
        predictions = []
        predictions_list = [load_single_prediction(metas, folder) for folder in infer_cfg.fuse_list]
        for i in range(len(predictions_list[0])):
            predictions.append(torch.stack([pred[i] for pred in predictions_list]).mean(dim=0))
        return predictions
    else:
        return load_single_prediction(metas, infer_cfg.folder)


def selected_axis_to_dense_axis(coords, meta):
    positions = meta.get("irregular_selected_positions", None)
    valid_len = meta.get("irregular_selected_valid_len", None)
    if positions is None or valid_len is None or meta.get("irregular_native_axis", False):
        return coords

    positions = torch.as_tensor(positions, dtype=coords.dtype, device=coords.device).reshape(-1)
    if positions.numel() == 0:
        return coords

    xp = torch.arange(positions.numel(), dtype=coords.dtype, device=coords.device)
    xp = torch.cat([xp, xp.new_tensor([float(positions.numel())])], dim=0)
    fp = torch.cat([positions, positions.new_tensor([float(valid_len)])], dim=0)

    coord_shape = coords.shape
    coord_flat = coords.reshape(-1).clamp(min=0.0, max=float(positions.numel()))
    right_idx = torch.searchsorted(xp, coord_flat, right=True).clamp(min=1, max=xp.numel() - 1)
    left_idx = right_idx - 1
    x0 = xp[left_idx]
    x1 = xp[right_idx]
    y0 = fp[left_idx]
    y1 = fp[right_idx]
    weight = (coord_flat - x0) / (x1 - x0).clamp(min=1e-6)
    return (y0 + weight * (y1 - y0)).reshape(coord_shape)


def convert_to_seconds(segments, meta):
    if meta["fps"] == -1:  # resize setting, like in anet / hacs
        segments = segments / meta["resize_length"] * meta["duration"]
    else:  # sliding window / padding setting, like in thumos / ego4d
        snippet_stride = meta["snippet_stride"]
        offset_frames = meta["offset_frames"]
        window_start_frame = meta["window_start_frame"] if "window_start_frame" in meta.keys() else 0
        irregular_positions = meta.get("irregular_selected_positions", None)
        irregular_valid_len = meta.get("irregular_selected_valid_len", None)
        if (
            irregular_positions is not None
            and irregular_valid_len is not None
            and not meta.get("irregular_native_axis", False)
        ):
            segments = selected_axis_to_dense_axis(segments, meta)
        segments = (segments * snippet_stride + window_start_frame + offset_frames) / meta["fps"]

    # truncate all boundaries within [0, duration]
    if segments.shape[0] > 0:
        segments[segments <= 0.0] *= 0.0
        segments[segments >= meta["duration"]] = segments[segments >= meta["duration"]] * 0.0 + meta["duration"]
    return segments
