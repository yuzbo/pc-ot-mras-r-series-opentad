import torch


FORBIDDEN_FLAG_KEYS = (
    "uses_gt",
    "uses_teacher",
    "uses_cache",
    "uses_prediction_cache",
    "uses_raw_prediction",
    "prediction_uses_gt",
    "remap_gt_to_selected_axis",
)
FORBIDDEN_FLAG_KEY_SET = {key.lower() for key in FORBIDDEN_FLAG_KEYS}

FORBIDDEN_EVAL_PAYLOAD_KEYS = (
    "annotation",
    "annotations",
    "annotation_path",
    "annotations_path",
    "gt",
    "gt_segments",
    "gt_labels",
    "gt_segments_path",
    "gt_labels_path",
    "gt_path",
    "ground_truth",
    "ground_truth_path",
    "teacher",
    "teacher_logits",
    "teacher_scores",
    "teacher_path",
    "teacher_logits_path",
    "teacher_scores_path",
    "raw_prediction",
    "raw_predictions",
    "raw_prediction_path",
    "raw_predictions_path",
    "raw_preds_path",
    "prediction_cache",
    "prediction_cache_path",
    "cache",
    "cache_key",
    "cache_path",
    "hidden_cache",
    "hidden_cache_path",
    "precomputed_prediction",
    "precomputed_predictions",
    "precomputed_prediction_path",
    "precomputed_predictions_path",
    "oracle",
    "oracle_path",
    "oracle_scores",
    "oracle_logits",
    "oracle_segments",
    "label",
    "labels",
    "label_path",
    "labels_path",
    "segment",
    "segments",
    "segment_path",
    "segments_path",
    "result_detection",
    "result_detection_path",
    "nms_prediction",
    "nms_predictions",
    "proposal_cache",
    "proposal_cache_path",
)

FORBIDDEN_EVAL_PAYLOAD_KEY_SET = {key.lower() for key in FORBIDDEN_EVAL_PAYLOAD_KEYS}

FORBIDDEN_EVAL_KEY_TOKENS = (
    "teacher",
    "raw_pred",
    "raw_prediction",
    "prediction_cache",
    "hidden_cache",
    "precomputed_prediction",
    "oracle",
    "proposal_cache",
    "result_detection",
    "nms_prediction",
)

DENSE_AXIS_KEYS = (
    "gt_axis",
    "segments_axis",
    "target_axis",
    "proposal_axis",
    "temporal_axis",
    "decode_axis",
    "gt_segments_unit",
    "segment_unit",
)

DENSE_AXIS_ALLOWED_VALUES = {
    "dense",
    "dense_axis",
    "dense_index",
    "dense_time",
    "physical",
    "physical_time",
    "native_dense",
}


def _as_bool_mask(mask, name):
    if not torch.is_tensor(mask):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if mask.ndim != 2:
        raise ValueError(f"{name} must be 2D [B, T], got shape {tuple(mask.shape)}.")
    mask = mask.bool()
    valid_counts = mask.sum(dim=1)
    if (valid_counts <= 0).any().item():
        raise ValueError(f"{name} must contain at least one valid token per sample.")
    for batch_idx in range(mask.shape[0]):
        valid_count = int(valid_counts[batch_idx].item())
        if not mask[batch_idx, :valid_count].all().item() or mask[batch_idx, valid_count:].any().item():
            raise ValueError(f"{name} must be a contiguous valid prefix for sample {batch_idx}.")
    return mask


def _flag_is_true(value):
    if torch.is_tensor(value):
        if value.numel() != 1:
            return bool(value.any().item())
        return bool(value.item())
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _iter_nested_items(value, path="meta"):
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            item_path = f"{path}.{key_text}"
            yield item_path, key_text, item
            yield from _iter_nested_items(item, item_path)
    elif isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            yield from _iter_nested_items(item, f"{path}[{idx}]")


def _eval_payload_key_is_forbidden(key):
    lower = str(key).lower()
    if lower in FORBIDDEN_EVAL_PAYLOAD_KEY_SET:
        return True
    if lower.startswith("oracle_"):
        return True
    return any(token in lower for token in FORBIDDEN_EVAL_KEY_TOKENS)


def _check_forbidden_flags(meta, batch_idx):
    for path, key, value in _iter_nested_items(meta):
        if str(key).lower() in FORBIDDEN_FLAG_KEY_SET and _flag_is_true(value):
            raise ValueError(f"meta[{batch_idx}] has forbidden true flag '{key}' at {path}.")


def _check_eval_payload_keys(meta, batch_idx):
    for path, key, value in _iter_nested_items(meta):
        if str(key).lower() in FORBIDDEN_FLAG_KEY_SET and not _flag_is_true(value):
            continue
        if _eval_payload_key_is_forbidden(key):
            raise ValueError(f"eval meta[{batch_idx}] contains forbidden payload '{key}' at {path}.")


def _check_dense_native_axis_contract(meta, batch_idx):
    if "remap_gt_to_selected_axis" in meta and _flag_is_true(meta["remap_gt_to_selected_axis"]):
        raise ValueError("P2 forbids remap_gt_to_selected_axis=True; GT/proposals must stay on dense axis.")

    lowered = {str(key).lower(): value for key, value in meta.items()}
    for key in DENSE_AXIS_KEYS:
        if key not in lowered:
            continue
        value = lowered[key]
        if isinstance(value, str):
            axis_value = value.strip().lower()
            if axis_value not in DENSE_AXIS_ALLOWED_VALUES:
                raise ValueError(
                    f"meta[{batch_idx}] axis field '{key}' must describe dense/native physical time, got '{value}'."
                )


def validate_sampling_contract(
    metas,
    masks,
    split,
    positions_key="irregular_selected_positions",
    valid_len_key="irregular_selected_valid_len",
    require_native_axis=True,
    forbid_eval_payload=True,
):
    """Validate deploy-visible sparse sampling metadata against feature masks."""

    mask = _as_bool_mask(masks, "masks")
    if metas is None:
        raise ValueError("metas are required for native irregular sampling.")
    if len(metas) != mask.shape[0]:
        raise ValueError(f"metas length {len(metas)} must match batch size {mask.shape[0]}.")

    split_name = str(split).lower()
    eval_like = split_name in {"val", "valid", "validation", "test", "eval", "inference"}
    valid_counts = mask.sum(dim=1)
    for batch_idx, meta in enumerate(metas):
        if not isinstance(meta, dict):
            raise TypeError(f"meta[{batch_idx}] must be a dict.")

        _check_forbidden_flags(meta, batch_idx)

        if eval_like and forbid_eval_payload:
            _check_eval_payload_keys(meta, batch_idx)

        if require_native_axis and not bool(meta.get("irregular_native_axis", False)):
            raise ValueError("P2 requires irregular_native_axis=True with dense-axis GT/proposals.")
        if require_native_axis:
            _check_dense_native_axis_contract(meta, batch_idx)
        dense_valid_len_key = "irregular_dense_valid_len" if "irregular_dense_valid_len" in meta else valid_len_key
        if positions_key not in meta or dense_valid_len_key not in meta:
            raise ValueError(f"meta[{batch_idx}] must contain '{positions_key}' and '{dense_valid_len_key}'.")
        if "irregular_dense_valid_len" in meta and valid_len_key in meta:
            if float(meta["irregular_dense_valid_len"]) != float(meta[valid_len_key]):
                raise ValueError(
                    f"meta[{batch_idx}] irregular_dense_valid_len must match '{valid_len_key}' alias."
                )

        positions = torch.as_tensor(meta[positions_key], device=mask.device, dtype=torch.float32).flatten()
        valid_count = int(valid_counts[batch_idx].item())
        if "irregular_selected_count" in meta and int(meta["irregular_selected_count"]) != valid_count:
            raise ValueError(
                f"meta[{batch_idx}] irregular_selected_count must equal valid token count {valid_count}."
            )
        if positions.numel() != valid_count:
            raise ValueError(
                f"meta[{batch_idx}]['{positions_key}'] length {positions.numel()} "
                f"must equal valid token count {valid_count}."
            )
        if valid_count > 1 and ((positions[1:] - positions[:-1]) <= 0).any().item():
            raise ValueError(f"meta[{batch_idx}]['{positions_key}'] must be strictly increasing.")

        dense_len = float(meta[dense_valid_len_key])
        if dense_len <= 0:
            raise ValueError(f"meta[{batch_idx}]['{dense_valid_len_key}'] must be positive.")
        if positions.numel() and (positions.min().item() < 0 or positions.max().item() >= dense_len):
            raise ValueError(
                f"meta[{batch_idx}]['{positions_key}'] must be in [0, {dense_len}), "
                f"got min={positions.min().item()}, max={positions.max().item()}."
            )
    return True
