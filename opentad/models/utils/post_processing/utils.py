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


def _meta_float_tensor(meta, key, *, dtype, device):
    if key not in meta:
        return None
    value = meta[key]
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype).flatten()
    return torch.as_tensor(value, dtype=dtype, device=device).flatten()


def _flag_is_true(value):
    if value is True:
        return True
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().item()) if value.numel() == 1 else False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    return False


def _selected_axis_segments_to_dense_axis(segments, meta):
    if _flag_is_true(meta.get("irregular_native_axis", True)):
        return segments
    selected_positions = _meta_float_tensor(
        meta,
        "irregular_selected_positions",
        dtype=segments.dtype,
        device=segments.device,
    )
    if selected_positions is None:
        return segments
    if selected_positions.numel() == 0:
        raise ValueError("irregular selected-axis post-processing requires non-empty irregular_selected_positions")
    if selected_positions.numel() > 1 and torch.any(selected_positions[1:] < selected_positions[:-1]):
        raise ValueError("irregular_selected_positions must be sorted for selected-axis post-processing")

    valid_len = _meta_float_tensor(
        meta,
        "irregular_selected_valid_len",
        dtype=segments.dtype,
        device=segments.device,
    )
    if valid_len is None or valid_len.numel() == 0:
        valid_value = selected_positions[-1] + 1.0
    else:
        valid_value = valid_len[0]
    target_positions = torch.cat([selected_positions, valid_value.reshape(1)])
    selected_count = int(selected_positions.numel())

    coords = torch.clamp(segments, min=0.0, max=float(selected_count))
    left = torch.floor(coords).to(dtype=torch.long).clamp(min=0, max=max(selected_count - 1, 0))
    right = (left + 1).clamp(max=selected_count)
    frac = coords - left.to(dtype=coords.dtype)
    return target_positions[left] * (1.0 - frac) + target_positions[right] * frac


def convert_to_seconds(segments, meta):
    if meta["fps"] == -1:  # resize setting, like in anet / hacs
        segments = segments / meta["resize_length"] * meta["duration"]
    else:  # sliding window / padding setting, like in thumos / ego4d
        segments = _selected_axis_segments_to_dense_axis(segments, meta)
        snippet_stride = meta["snippet_stride"]
        offset_frames = meta["offset_frames"]
        window_start_frame = meta["window_start_frame"] if "window_start_frame" in meta.keys() else 0
        segments = (segments * snippet_stride + window_start_frame + offset_frames) / meta["fps"]

    # truncate all boundaries within [0, duration]
    if segments.shape[0] > 0:
        segments[segments <= 0.0] *= 0.0
        segments[segments >= meta["duration"]] = segments[segments >= meta["duration"]] * 0.0 + meta["duration"]
    return segments
