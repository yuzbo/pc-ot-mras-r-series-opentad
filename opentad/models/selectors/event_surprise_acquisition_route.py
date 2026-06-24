from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import SELECTORS


EVENT_SURPRISE_ROUTE_LABEL = "DIVERGENT_INNOVATION_EVENT_SURPRISE_DO_NOT_MERGE_WITH_C3"
EVENT_SURPRISE_META_KEY = "event_surprise_acquisition_plan"

_DEPLOY_META_LEAF_KEYS = frozenset(
    {
        "video_name",
        "video_id",
        "sample_id",
        "data_path",
        "fps",
        "duration",
        "snippet_stride",
        "window_start_frame",
        "resize_length",
        "window_size",
        "offset_frames",
        "feature_start_idx",
        "feature_end_idx",
        "feature_stride",
        "feat_stride",
        "feats_len_ori",
        "frame_count",
        "num_frames",
        "subset",
        "irregular_selected_positions",
        "irregular_selected_valid_len",
        "irregular_selected_output_valid_len",
        "irregular_native_axis",
        "event_surprise_selected_dense_indices",
        "event_surprise_selected_times",
        "event_surprise_remap_gt_to_selected_axis",
    }
)
_EVENT_SURPRISE_SCORE_KEYS = frozenset(
    {
        "selection_score",
        "temporal_surprise_score",
        "motion_score",
        "uncertainty_score",
        "redundancy_score",
    }
)
_EVENT_SURPRISE_PROTOCOL_FLAG_KEYS = frozenset(
    {
        "uses_deploy_visible_inputs_only",
        "uses_test_gt",
        "uses_oracle",
        "uses_teacher",
        "uses_raw_prediction_cache",
        "route_isolated_from_c3",
        "preview_feature_source",
        "decode_saving_claim_allowed",
        "runtime_flops_claim_allowed",
    }
)
_EVENT_SURPRISE_INFERENCE_MAPPING_KEYS = frozenset(
    {
        "selected_axis_to_dense_axis",
        "input_axis",
        "output_axis",
        "interpolation",
        "mapping_applied",
    }
)
_EVENT_SURPRISE_PLAN_KEYS = frozenset(
    {
        "route_label",
        "meta_key",
        "selection_policy",
        "selected_count",
        "target_len",
        "max_gap",
        "coverage_anchor_count",
        "remap_gt_to_selected_axis",
        "uses_deploy_visible_inputs_only",
        "uses_test_gt",
        "uses_oracle",
        "uses_teacher",
        "uses_raw_prediction_cache",
        "selected_axis_to_dense_axis_inference",
        "decode_saving_claim_allowed",
        "runtime_flops_claim_allowed",
    }
)
_DEPLOY_META_NESTED_KEY_ALLOWLIST = {
    "event_surprise_selected_scores": _EVENT_SURPRISE_SCORE_KEYS,
    "event_surprise_protocol_flags": _EVENT_SURPRISE_PROTOCOL_FLAG_KEYS,
    "event_surprise_inference_mapping": _EVENT_SURPRISE_INFERENCE_MAPPING_KEYS,
    EVENT_SURPRISE_META_KEY: _EVENT_SURPRISE_PLAN_KEYS,
}
_ALLOWED_DEPLOY_META_KEYS = _DEPLOY_META_LEAF_KEYS | frozenset(_DEPLOY_META_NESTED_KEY_ALLOWLIST)
_DEPLOY_FALSE_FLAG_KEYS = frozenset(
    {
        "uses_test_gt",
        "uses_oracle",
        "uses_teacher",
        "uses_raw_prediction_cache",
    }
)

_FORBIDDEN_DEPLOY_META_TOKENS = frozenset(
    {
        "gt",
        "oracle",
        "teacher",
        "cache",
        "prediction",
        "predictions",
        "checkpoint",
        "ckpt",
        "result",
    }
)
_FORBIDDEN_DEPLOY_META_PHRASES = frozenset(
    {
        "ground_truth",
        "groundtruth",
        "raw_prediction",
        "raw_predictions",
        "rawprediction",
        "rawpredictions",
        "gtsegment",
        "gtsegments",
        "gtlabel",
        "gtlabels",
    }
)


def _require_finite(tensor: torch.Tensor, name: str) -> None:
    if not torch.is_tensor(tensor):
        raise TypeError(f"{name} must be a tensor")
    if torch.is_complex(tensor):
        raise ValueError(f"{name} must be real-valued")
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must be finite")


def _prefix_mask(mask: torch.Tensor, expected_shape: tuple[int, int]) -> torch.Tensor:
    if mask.ndim != 2:
        raise ValueError(f"valid mask must be [B,T], got {tuple(mask.shape)}")
    if tuple(mask.shape) != tuple(expected_shape):
        raise ValueError(f"valid mask shape mismatch: expected {expected_shape}, got {tuple(mask.shape)}")
    if mask.dtype != torch.bool and not bool(((mask == 0) | (mask == 1)).all().item()):
        raise ValueError("valid mask must be boolean or binary")
    valid = mask.bool()
    valid_count = valid.long().sum(dim=1)
    if bool((valid_count <= 0).any().item()):
        raise ValueError("each sample must contain at least one valid temporal position")
    prefix = torch.arange(valid.shape[1], device=valid.device)[None, :] < valid_count[:, None]
    if not torch.equal(valid, prefix):
        raise ValueError("event-surprise acquisition requires prefix-contiguous valid masks")
    return valid


def _default_time_coords(valid: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    batch, time = valid.shape
    positions = torch.arange(time, device=valid.device, dtype=dtype)[None, :].expand(batch, -1)
    valid_len = valid.long().sum(dim=1).clamp(min=1).to(dtype=dtype)
    denom = (valid_len - 1.0).clamp(min=1.0)
    return (positions / denom[:, None]).masked_fill(~valid, 0.0)


def _normalize_valid(scores: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    masked = scores.float().masked_fill(~valid, 0.0)
    high = masked.masked_fill(~valid, torch.finfo(torch.float32).min).amax(dim=1, keepdim=True)
    low = masked.masked_fill(~valid, torch.finfo(torch.float32).max).amin(dim=1, keepdim=True)
    denom = (high - low).clamp_min(1.0e-6)
    normalized = (masked - low) / denom
    return normalized.masked_fill(~valid, 0.0)


def _normalized_text(value: object) -> str:
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(value or ""))
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", text)
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text)


def _contains_forbidden_deploy_fragment(value: object) -> bool:
    normalized = _normalized_text(value)
    tokens = [token for token in re.split(r"_+", normalized.strip("_")) if token]
    compact = "".join(tokens)
    if any(token in _FORBIDDEN_DEPLOY_META_TOKENS for token in tokens):
        return True
    if "ground" in tokens and "truth" in tokens:
        return True
    if compact.startswith("gt"):
        return True
    if any(phrase in normalized or phrase in compact for phrase in _FORBIDDEN_DEPLOY_META_PHRASES):
        return True
    return any(token != "gt" and token in compact for token in _FORBIDDEN_DEPLOY_META_TOKENS)


def _contains_forbidden_deploy_meta(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            _contains_forbidden_deploy_fragment(key) or _contains_forbidden_deploy_meta(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_deploy_meta(item) for item in value)
    if isinstance(value, str):
        return _contains_forbidden_deploy_fragment(value)
    return False


def _validate_deploy_leaf_value(value: object, *, location: str) -> None:
    if isinstance(value, Mapping):
        raise ValueError(f"{location} must not contain nested deploy metadata")
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_deploy_leaf_value(item, location=f"{location}[{index}]")
        return
    if isinstance(value, str) and _contains_forbidden_deploy_fragment(value):
        raise ValueError(f"{location} contains forbidden deploy meta")


def _validate_deploy_visible_meta(
    value: object,
    *,
    location: str = "metas",
    allowed_keys: frozenset[str] = _ALLOWED_DEPLOY_META_KEYS,
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key or "")
            if key_text not in allowed_keys:
                if _contains_forbidden_deploy_fragment(key_text) or _contains_forbidden_deploy_meta(item):
                    raise ValueError(f"{location}.{key_text} contains forbidden deploy meta")
                raise ValueError(f"{location}.{key_text} unexpected deploy meta key")
            if key_text in _DEPLOY_FALSE_FLAG_KEYS and item is not False:
                raise ValueError(f"{location}.{key_text} must be false for deploy-visible metadata")
            nested_allowed = _DEPLOY_META_NESTED_KEY_ALLOWLIST.get(key_text)
            if nested_allowed is None:
                _validate_deploy_leaf_value(item, location=f"{location}.{key_text}")
            else:
                if not isinstance(item, Mapping):
                    raise ValueError(f"{location}.{key_text} must be a deploy metadata mapping")
                _validate_deploy_visible_meta(
                    item,
                    location=f"{location}.{key_text}",
                    allowed_keys=nested_allowed,
                )
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_deploy_visible_meta(item, location=f"{location}[{index}]", allowed_keys=allowed_keys)
        return
    _validate_deploy_leaf_value(value, location=location)


def _coverage_anchors(valid_len: int, count: int) -> list[int]:
    if valid_len <= 0:
        raise ValueError("valid_len must be positive")
    count = max(1, min(int(count), valid_len))
    if count == 1:
        return [0]
    return sorted({int(round(item)) for item in torch.linspace(0, valid_len - 1, steps=count).tolist()})


def _remove_one_without_breaking_gap(selected: set[int], protected: set[int], max_gap: int) -> int | None:
    ordered = sorted(selected)
    removable = [idx for idx in ordered if idx not in protected]
    for idx in removable:
        trial = sorted(selected - {idx})
        if len(trial) <= 1:
            continue
        gaps = [right - left for left, right in zip(trial[:-1], trial[1:])]
        if max(gaps) <= max_gap:
            return idx
    return removable[0] if removable else None


@SELECTORS.register_module()
class EventSurpriseTemporalAcquisitionSelector(nn.Module):
    """Deploy-visible temporal surprise acquisition candidate.

    The selector scores dense temporal positions by feature change, local motion
    energy, and uncertainty from neighboring-change disagreement. Stable
    redundant spans receive lower priority, while coverage anchors and a max-gap
    repair keep the selected temporal axis usable by the downstream detector.
    """

    forbid_raw_prediction_cache = True

    def __init__(
        self,
        in_dim: int,
        target_len: int = 384,
        max_gap: int = 8,
        coverage_anchor_count: int = 64,
        surprise_topk: int | None = None,
        surprise_weight: float = 1.0,
        motion_weight: float = 0.75,
        uncertainty_weight: float = 0.50,
        redundancy_weight: float = 0.35,
        min_width: float = 0.015,
        input_layout: str = "auto",
        remap_gt_to_selected_axis: bool = True,
        route_label: str = EVENT_SURPRISE_ROUTE_LABEL,
        meta_key: str = EVENT_SURPRISE_META_KEY,
    ) -> None:
        super().__init__()
        if int(in_dim) <= 0:
            raise ValueError("in_dim must be positive")
        if int(target_len) <= 0:
            raise ValueError("target_len must be positive")
        if int(max_gap) <= 0:
            raise ValueError("max_gap must be positive")
        if int(coverage_anchor_count) <= 0:
            raise ValueError("coverage_anchor_count must be positive")
        if float(min_width) <= 0:
            raise ValueError("min_width must be positive")
        self.in_dim = int(in_dim)
        self.target_len = int(target_len)
        self.max_gap = int(max_gap)
        self.coverage_anchor_count = int(coverage_anchor_count)
        self.surprise_topk = None if surprise_topk is None else int(surprise_topk)
        self.surprise_weight = float(surprise_weight)
        self.motion_weight = float(motion_weight)
        self.uncertainty_weight = float(uncertainty_weight)
        self.redundancy_weight = float(redundancy_weight)
        self.min_width = float(min_width)
        self.input_layout = str(input_layout)
        if self.input_layout not in {"auto", "btc", "bct", "bcthw"}:
            raise ValueError("input_layout must be one of auto, btc, bct, or bcthw")
        self.remap_gt_to_selected_axis = bool(remap_gt_to_selected_axis)
        self.route_label = str(route_label)
        self.meta_key = str(meta_key)
        if self.route_label != EVENT_SURPRISE_ROUTE_LABEL:
            raise ValueError("unexpected event-surprise route_label")
        if self.meta_key != EVENT_SURPRISE_META_KEY:
            raise ValueError("unexpected event-surprise meta_key")

    def _validate_inputs(
        self,
        features: torch.Tensor,
        valid_mask: torch.Tensor,
        time_coords: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 3:
            raise ValueError(f"features must be [B,T,C], got {tuple(features.shape)}")
        if int(features.shape[-1]) != self.in_dim:
            raise ValueError(f"expected feature dim {self.in_dim}, got {features.shape[-1]}")
        _require_finite(features, "features")
        valid = _prefix_mask(valid_mask.to(device=features.device), tuple(features.shape[:2]))
        if time_coords is None:
            coords = _default_time_coords(valid, features.dtype)
        else:
            if time_coords.ndim != 2 or tuple(time_coords.shape) != tuple(features.shape[:2]):
                raise ValueError("time_coords must be [B,T] and match features")
            coords = time_coords.to(device=features.device, dtype=features.dtype)
            _require_finite(coords, "time_coords")
            adjacent_valid = valid[:, 1:] & valid[:, :-1]
            if bool(adjacent_valid.any().item()):
                deltas = coords[:, 1:] - coords[:, :-1]
                if not bool((deltas[adjacent_valid] > 0).all().item()):
                    raise ValueError("time_coords valid prefix must be strictly increasing")
            coords = coords.masked_fill(~valid, 0.0)
        return valid, coords

    def _event_scores(self, features: torch.Tensor, valid: torch.Tensor) -> dict[str, torch.Tensor]:
        masked_features = features.masked_fill(~valid.unsqueeze(-1), 0.0)
        delta = masked_features.new_zeros(features.shape[:2])
        pair_valid = valid[:, 1:] & valid[:, :-1]
        frame_delta = (masked_features[:, 1:] - masked_features[:, :-1]).abs().mean(dim=-1)
        delta[:, 1:] = frame_delta.masked_fill(~pair_valid, 0.0)
        motion = F.avg_pool1d(delta[:, None, :], kernel_size=3, stride=1, padding=1).squeeze(1)
        local = F.avg_pool1d(delta[:, None, :], kernel_size=5, stride=1, padding=2).squeeze(1)
        uncertainty = (delta - local).abs()
        redundancy = 1.0 - _normalize_valid(motion + uncertainty, valid)
        surprise = _normalize_valid(delta, valid)
        motion = _normalize_valid(motion, valid)
        uncertainty = _normalize_valid(uncertainty, valid)
        selection_score = (
            self.surprise_weight * surprise
            + self.motion_weight * motion
            + self.uncertainty_weight * uncertainty
            - self.redundancy_weight * redundancy
        ).masked_fill(~valid, torch.finfo(torch.float32).min)
        return {
            "temporal_surprise_score": surprise,
            "motion_score": motion,
            "uncertainty_score": uncertainty,
            "redundancy_score": redundancy.masked_fill(~valid, 0.0),
            "selection_score": selection_score,
        }

    def _select_one(self, scores: torch.Tensor, valid_len: int) -> tuple[list[int], int]:
        target = min(self.target_len, int(valid_len))
        if target <= 0:
            raise ValueError("target selected length must be positive")
        if target > 1 and (target - 1) * self.max_gap < int(valid_len) - 1:
            raise ValueError("target_len and max_gap cannot cover the valid temporal span")
        anchor_count = min(self.coverage_anchor_count, target)
        anchors = set(_coverage_anchors(int(valid_len), anchor_count))
        protected = set(anchors)
        selected = set(anchors)
        surprise_budget = self.surprise_topk if self.surprise_topk is not None else target
        surprise_budget = max(0, min(int(surprise_budget), target))
        ranked = torch.argsort(scores[:valid_len], descending=True).tolist()
        for idx in ranked[:surprise_budget]:
            selected.add(int(idx))
            if len(selected) >= target:
                break
        for idx in ranked:
            if len(selected) >= target:
                break
            selected.add(int(idx))

        while True:
            ordered = sorted(selected)
            if len(ordered) <= 1:
                break
            gap_items = [(right - left, left, right) for left, right in zip(ordered[:-1], ordered[1:])]
            max_observed_gap, left, _right = max(gap_items)
            if max_observed_gap <= self.max_gap:
                break
            candidate = min(int(valid_len) - 1, left + self.max_gap)
            while candidate in selected and candidate > left:
                candidate -= 1
            if candidate <= left or candidate in selected:
                candidate = min(int(valid_len) - 1, left + max(1, max_observed_gap // 2))
            selected.add(candidate)
            protected.add(candidate)
            while len(selected) > target:
                removable = _remove_one_without_breaking_gap(selected, protected, self.max_gap)
                if removable is None:
                    raise ValueError("cannot satisfy event-surprise max-gap repair under current target_len")
                selected.remove(removable)

        for idx in ranked:
            if len(selected) >= target:
                break
            selected.add(int(idx))
        ordered = sorted(selected)
        if len(ordered) > target:
            ordered = ordered[:target]
        if len(ordered) > 1:
            gaps = [right - left for left, right in zip(ordered[:-1], ordered[1:])]
            if max(gaps) > self.max_gap:
                raise ValueError("event-surprise selected indices violate max_gap")
        return ordered, target

    def _build_outputs(
        self,
        features: torch.Tensor,
        valid: torch.Tensor,
        coords: torch.Tensor,
        scores: Mapping[str, torch.Tensor],
    ) -> dict[str, Any]:
        batch, dense_len, channels = features.shape
        indices = torch.zeros((batch, self.target_len), dtype=torch.long, device=features.device)
        selected_mask = torch.zeros((batch, self.target_len), dtype=torch.bool, device=features.device)
        acquisition = features.new_zeros((batch, self.target_len, dense_len))
        selected_tokens = features.new_zeros((batch, self.target_len, channels))
        selected_positions = features.new_zeros((batch, self.target_len))
        selected_times = features.new_zeros((batch, self.target_len))
        widths = features.new_zeros((batch, self.target_len))
        valid_lengths = valid.long().sum(dim=1)

        for batch_idx in range(batch):
            valid_len = int(valid_lengths[batch_idx].item())
            selected, count = self._select_one(scores["selection_score"][batch_idx].detach().cpu(), valid_len)
            selected_tensor = torch.tensor(selected, dtype=torch.long, device=features.device)
            indices[batch_idx, :count] = selected_tensor
            selected_mask[batch_idx, :count] = True
            acquisition[batch_idx, torch.arange(count, device=features.device), selected_tensor] = 1.0
            selected_tokens[batch_idx, :count] = features[batch_idx, selected_tensor]
            selected_positions[batch_idx, :count] = selected_tensor.to(dtype=features.dtype)
            selected_times[batch_idx, :count] = coords[batch_idx, selected_tensor]
            if count > 1:
                gap = selected_tensor.float().diff().median().clamp_min(1.0)
                widths[batch_idx, :count] = max(self.min_width, float(gap.item()) / max(float(valid_len), 1.0))
            else:
                widths[batch_idx, :count] = self.min_width

        gates = selected_mask.to(dtype=features.dtype)
        centers = selected_times.clone()
        outputs: dict[str, Any] = {
            "route_label": self.route_label,
            "meta_key": self.meta_key,
            "valid_mask": valid,
            "valid_lengths": valid_lengths,
            "time_coords": coords,
            "selected_dense_indices": indices,
            "selected_mask": selected_mask,
            "selected_tokens": selected_tokens,
            "selected_positions": selected_positions,
            "selected_times": selected_times,
            "centers": centers,
            "widths": widths,
            "gates": gates,
            "allocation": acquisition,
            "acquisition_matrix": acquisition,
            self.meta_key: {
                "route_label": self.route_label,
                "uses_deploy_visible_inputs_only": True,
                "selection_policy": "event_surprise_with_coverage_anchors_and_max_gap",
                "target_len": self.target_len,
                "max_gap": self.max_gap,
                "coverage_anchor_count": self.coverage_anchor_count,
            },
        }
        outputs.update(scores)
        return outputs

    def _fit_preview_dim(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(f"preview features must be [B,T,C], got {tuple(features.shape)}")
        current_dim = int(features.shape[-1])
        if current_dim <= 0:
            raise ValueError("preview feature dim must be positive")
        if current_dim == self.in_dim:
            return features
        if current_dim > self.in_dim:
            return features[..., : self.in_dim]
        repeat = (self.in_dim + current_dim - 1) // current_dim
        return features.repeat_interleave(repeat, dim=-1)[..., : self.in_dim]

    def _detector_preview_features(self, inputs: torch.Tensor) -> tuple[torch.Tensor, str]:
        if not torch.is_tensor(inputs):
            raise TypeError("inputs must be a tensor")
        _require_finite(inputs, "inputs")
        layout = self.input_layout
        if inputs.ndim == 3:
            if layout == "btc":
                features = inputs
                resolved = "btc"
            elif layout == "bct":
                features = inputs.transpose(1, 2).contiguous()
                resolved = "bct"
            elif layout == "auto":
                if int(inputs.shape[-1]) == self.in_dim and int(inputs.shape[1]) != self.in_dim:
                    features = inputs
                    resolved = "btc"
                else:
                    features = inputs.transpose(1, 2).contiguous()
                    resolved = "bct"
            else:
                raise ValueError(f"input_layout={layout} requires 5D [B,C,T,H,W] inputs")
            return self._fit_preview_dim(features.detach().to(dtype=torch.float32)), resolved

        if inputs.ndim == 5:
            if layout not in {"auto", "bcthw"}:
                raise ValueError("5D inputs require input_layout='auto' or 'bcthw'")
            features = inputs.detach().to(dtype=torch.float32).mean(dim=(3, 4)).transpose(1, 2).contiguous()
            return self._fit_preview_dim(features), "bcthw"

        raise ValueError(f"unsupported detector input shape: {tuple(inputs.shape)}")

    @staticmethod
    def _gather_detector_inputs(
        inputs: torch.Tensor,
        indices: torch.Tensor,
        selected_mask: torch.Tensor,
        layout: str,
    ) -> torch.Tensor:
        batch, target_len = indices.shape
        if layout == "btc":
            if inputs.ndim != 3 or int(inputs.shape[0]) != batch:
                raise ValueError("btc detector inputs must be [B,T,C]")
            gather_index = indices[:, :, None].expand(-1, -1, inputs.shape[-1])
            selected = torch.gather(inputs, dim=1, index=gather_index)
            return selected.masked_fill(~selected_mask[:, :, None], 0.0)
        if layout == "bct":
            if inputs.ndim != 3 or int(inputs.shape[0]) != batch:
                raise ValueError("bct detector inputs must be [B,C,T]")
            gather_index = indices[:, None, :].expand(-1, inputs.shape[1], -1)
            selected = torch.gather(inputs, dim=2, index=gather_index)
            return selected.masked_fill(~selected_mask[:, None, :], 0.0)
        if layout == "bcthw":
            if inputs.ndim != 5 or int(inputs.shape[0]) != batch:
                raise ValueError("bcthw detector inputs must be [B,C,T,H,W]")
            gather_index = indices[:, None, :, None, None].expand(
                -1,
                inputs.shape[1],
                -1,
                inputs.shape[3],
                inputs.shape[4],
            )
            selected = torch.gather(inputs, dim=2, index=gather_index)
            return selected.masked_fill(~selected_mask[:, None, :, None, None], 0.0)
        raise ValueError(f"unsupported resolved layout: {layout}")

    def _normalize_metas_for_batch(
        self,
        metas: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None,
        *,
        batch_size: int,
    ) -> list[dict[str, Any]]:
        if metas is None:
            return [{} for _idx in range(batch_size)]
        if isinstance(metas, Mapping):
            if batch_size != 1:
                raise ValueError("mapping metas are only valid for batch size 1")
            return [dict(metas)]
        if not isinstance(metas, (list, tuple)):
            raise ValueError("metas must be a mapping or sequence of mappings")
        if len(metas) != batch_size:
            raise ValueError(f"metas length mismatch: expected {batch_size}, got {len(metas)}")
        out: list[dict[str, Any]] = []
        for idx, meta in enumerate(metas):
            if not isinstance(meta, Mapping):
                raise ValueError(f"metas[{idx}] must be a mapping")
            out.append(dict(meta))
        return out

    def _write_detector_metas(
        self,
        metas: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None,
        selector_outputs: Mapping[str, Any],
        *,
        preview_feature_source: str,
    ) -> list[dict[str, Any]]:
        indices = selector_outputs["selected_dense_indices"].detach().cpu()
        selected_mask = selector_outputs["selected_mask"].detach().cpu()
        selected_times = selector_outputs["selected_times"].detach().cpu()
        valid_lengths = selector_outputs["valid_lengths"].detach().cpu()
        batch_size = int(indices.shape[0])
        out = self._normalize_metas_for_batch(metas, batch_size=batch_size)
        score_keys = (
            "selection_score",
            "temporal_surprise_score",
            "motion_score",
            "uncertainty_score",
            "redundancy_score",
        )
        for batch_idx, meta in enumerate(out):
            count = int(selected_mask[batch_idx].long().sum().item())
            prefix_indices = [int(item) for item in indices[batch_idx, :count].tolist()]
            prefix_times = [float(item) for item in selected_times[batch_idx, :count].tolist()]
            score_payload = {}
            for key in score_keys:
                dense_scores = selector_outputs.get(key)
                if not torch.is_tensor(dense_scores):
                    continue
                score_payload[key] = [
                    float(dense_scores[batch_idx, pos].detach().cpu().item())
                    for pos in prefix_indices
                ]
            meta["irregular_selected_positions"] = [float(item) for item in prefix_indices]
            meta["irregular_selected_valid_len"] = float(valid_lengths[batch_idx].item())
            meta["irregular_selected_output_valid_len"] = float(count)
            meta["irregular_native_axis"] = False
            meta["event_surprise_selected_dense_indices"] = prefix_indices
            meta["event_surprise_selected_times"] = prefix_times
            meta["event_surprise_selected_scores"] = score_payload
            meta["event_surprise_remap_gt_to_selected_axis"] = bool(self.remap_gt_to_selected_axis)
            meta["event_surprise_protocol_flags"] = {
                "uses_deploy_visible_inputs_only": True,
                "uses_test_gt": False,
                "uses_oracle": False,
                "uses_teacher": False,
                "uses_raw_prediction_cache": False,
                "route_isolated_from_c3": True,
                "preview_feature_source": preview_feature_source,
                "decode_saving_claim_allowed": False,
                "runtime_flops_claim_allowed": False,
            }
            meta["event_surprise_inference_mapping"] = {
                "selected_axis_to_dense_axis": True,
                "input_axis": "selected_detector_axis",
                "output_axis": "dense_window",
                "interpolation": "piecewise_linear_selected_slots_to_dense_indices",
                "mapping_applied": False,
            }
            meta[self.meta_key] = {
                "route_label": self.route_label,
                "meta_key": self.meta_key,
                "selection_policy": "event_surprise_with_coverage_anchors_and_max_gap",
                "selected_count": count,
                "target_len": self.target_len,
                "max_gap": self.max_gap,
                "coverage_anchor_count": self.coverage_anchor_count,
                "remap_gt_to_selected_axis": bool(self.remap_gt_to_selected_axis),
                "uses_test_gt": False,
                "uses_oracle": False,
                "uses_teacher": False,
                "uses_raw_prediction_cache": False,
                "selected_axis_to_dense_axis_inference": True,
                "decode_saving_claim_allowed": False,
                "runtime_flops_claim_allowed": False,
            }
        return out

    def _remap_gt_batch(
        self,
        gt_segments: object,
        gt_labels: object,
        selected_dense_indices: torch.Tensor,
        selected_mask: torch.Tensor,
    ) -> tuple[object, object]:
        if not self.remap_gt_to_selected_axis or gt_segments is None or gt_labels is None:
            return gt_segments, gt_labels
        segments_iter = list(gt_segments) if not torch.is_tensor(gt_segments) else list(gt_segments)
        labels_iter = list(gt_labels) if not torch.is_tensor(gt_labels) else list(gt_labels)
        if len(segments_iter) != int(selected_dense_indices.shape[0]) or len(labels_iter) != len(segments_iter):
            raise ValueError("gt_segments/gt_labels length must match batch size")
        mapped_segments = []
        mapped_labels = []
        for batch_idx, (segments, labels) in enumerate(zip(segments_iter, labels_iter)):
            if not torch.is_tensor(segments):
                raise TypeError("gt_segments items must be tensors")
            if segments.ndim != 2 or int(segments.shape[-1]) != 2:
                raise ValueError("gt_segments items must be [N,2]")
            label_count = self._label_count(labels)
            if int(segments.shape[0]) != label_count:
                raise ValueError(
                    "gt segment/label count mismatch: "
                    f"segments={int(segments.shape[0])}, labels={label_count}"
                )
            count = int(selected_mask[batch_idx].long().sum().item())
            positions = selected_dense_indices[batch_idx, :count].to(device=segments.device, dtype=torch.float32)
            remapped, kept = self._remap_one_gt(segments, positions)
            mapped_segments.append(remapped.to(dtype=segments.dtype))
            mapped_labels.append(self._filter_labels(labels, kept))
        return mapped_segments, mapped_labels

    @staticmethod
    def _label_count(labels: object) -> int:
        if torch.is_tensor(labels):
            if labels.ndim == 0:
                raise ValueError("gt_labels items must have a label axis")
            return int(labels.shape[0])
        if isinstance(labels, Sequence) and not isinstance(labels, (str, bytes)):
            return len(labels)
        raise TypeError("gt_labels items must be tensors or non-string sequences")

    @staticmethod
    def _filter_labels(labels: object, keep: torch.Tensor) -> object:
        if torch.is_tensor(labels):
            return labels[keep.to(device=labels.device)]
        if isinstance(labels, tuple):
            keep_list = [bool(item) for item in keep.detach().cpu().tolist()]
            return tuple(label for label, keep_item in zip(labels, keep_list) if keep_item)
        if isinstance(labels, list):
            keep_list = [bool(item) for item in keep.detach().cpu().tolist()]
            return [label for label, keep_item in zip(labels, keep_list) if keep_item]
        raise TypeError("gt_labels items must be tensors or non-string sequences")

    @staticmethod
    def _remap_one_gt(segments: torch.Tensor, selected_positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not torch.is_tensor(segments):
            raise TypeError("gt_segments items must be tensors")
        if segments.numel() == 0:
            keep = torch.zeros((0,), dtype=torch.bool, device=segments.device)
            return segments.reshape(0, 2), keep
        if segments.ndim != 2 or int(segments.shape[-1]) != 2:
            raise ValueError("gt_segments items must be [N,2]")
        if selected_positions.numel() <= 1:
            keep = torch.zeros((segments.shape[0],), dtype=torch.bool, device=segments.device)
            return segments.new_zeros((0, 2)), keep
        positions = selected_positions.to(device=segments.device, dtype=torch.float32)
        axis = torch.arange(positions.numel(), device=segments.device, dtype=torch.float32)
        endpoints = segments.to(dtype=torch.float32).reshape(-1)
        endpoints = endpoints.clamp(min=float(positions[0].item()), max=float(positions[-1].item()))
        right = torch.searchsorted(positions, endpoints, right=False).clamp(min=1, max=positions.numel() - 1)
        left = right - 1
        denom = (positions[right] - positions[left]).clamp_min(1.0e-6)
        alpha = (endpoints - positions[left]) / denom
        mapped = (axis[left] + alpha).reshape(-1, 2)
        keep = mapped[:, 1] > mapped[:, 0] + 1.0e-4
        return mapped[keep], keep

    @staticmethod
    def selected_axis_segments_to_dense_axis(
        segments: torch.Tensor,
        selected_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Map selected-axis proposal coordinates back to dense window indices."""
        if not torch.is_tensor(segments):
            raise TypeError("segments must be a tensor")
        if not torch.is_tensor(selected_positions):
            raise TypeError("selected_positions must be a tensor")
        if segments.numel() == 0:
            return segments.reshape(0, 2)
        if segments.ndim != 2 or int(segments.shape[-1]) != 2:
            raise ValueError("segments must be [N,2]")
        if selected_positions.ndim != 1:
            raise ValueError("selected_positions must be [K]")
        if selected_positions.numel() <= 0:
            raise ValueError("selected_positions must be non-empty")
        positions = selected_positions.to(device=segments.device, dtype=torch.float32)
        if positions.numel() == 1:
            return segments.new_full(segments.shape, float(positions[0].item()))
        if not bool((positions[1:] > positions[:-1]).all().item()):
            raise ValueError("selected_positions must be strictly increasing")
        axis = torch.arange(positions.numel(), device=segments.device, dtype=torch.float32)
        endpoints = segments.to(dtype=torch.float32).reshape(-1).clamp(min=0.0, max=float(axis[-1].item()))
        right = torch.searchsorted(axis, endpoints, right=False).clamp(min=1, max=positions.numel() - 1)
        left = right - 1
        denom = (axis[right] - axis[left]).clamp_min(1.0e-6)
        alpha = (endpoints - axis[left]) / denom
        mapped = positions[left] + alpha * (positions[right] - positions[left])
        return mapped.reshape_as(segments).to(dtype=segments.dtype)

    def map_selected_axis_predictions_to_dense_axis(
        self,
        proposals: Sequence[torch.Tensor],
        selector_outputs: Mapping[str, Any],
        metas: Sequence[Mapping[str, Any]] | None = None,
    ) -> list[torch.Tensor]:
        if not isinstance(proposals, (list, tuple)):
            raise TypeError("proposals must be a list/tuple of tensors")
        indices = selector_outputs.get("selected_dense_indices")
        selected_mask = selector_outputs.get("selected_mask")
        if not torch.is_tensor(indices) or not torch.is_tensor(selected_mask):
            raise ValueError("selector_outputs must contain selected_dense_indices and selected_mask tensors")
        if len(proposals) != int(indices.shape[0]):
            raise ValueError("proposal batch size must match selector outputs")
        if metas is not None and len(metas) != len(proposals):
            raise ValueError("metas batch size must match proposals")

        mapped: list[torch.Tensor] = []
        for batch_idx, proposal in enumerate(proposals):
            count = int(selected_mask[batch_idx].long().sum().item())
            if count <= 0:
                mapped.append(proposal.reshape(0, 2))
                continue
            selected_positions = indices[batch_idx, :count].to(device=proposal.device, dtype=torch.float32)
            mapped_proposal = self.selected_axis_segments_to_dense_axis(proposal, selected_positions)
            mapped.append(mapped_proposal)
            if metas is not None:
                meta = metas[batch_idx]
                if not isinstance(meta, Mapping):
                    raise ValueError(f"metas[{batch_idx}] must be a mapping")
                mapping = meta.get("event_surprise_inference_mapping")
                if isinstance(mapping, dict):
                    mapping["mapping_applied"] = True
        return mapped

    def _forward_detector(
        self,
        *,
        inputs: torch.Tensor,
        masks: torch.Tensor,
        metas: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
        time_coords: torch.Tensor | None = None,
        gt_segments: object = None,
        gt_labels: object = None,
        mode: str,
    ) -> dict[str, Any]:
        if str(mode) in {"test", "predict", "eval"} and metas is not None:
            _validate_deploy_visible_meta(metas)
        features, layout = self._detector_preview_features(inputs)
        preview_feature_source = (
            "loaded_dense_detector_input_prototype" if layout == "bcthw" else "deploy_visible_detector_feature_tensor"
        )
        valid, coords = self._validate_inputs(features, masks.to(device=features.device), time_coords)
        scores = self._event_scores(features, valid)
        selector_outputs = self._build_outputs(features, valid, coords, scores)
        selected_inputs = self._gather_detector_inputs(
            inputs,
            selector_outputs["selected_dense_indices"],
            selector_outputs["selected_mask"],
            layout,
        )
        output_metas = self._write_detector_metas(
            metas,
            selector_outputs,
            preview_feature_source=preview_feature_source,
        )
        detector_outputs: dict[str, Any] = {
            "inputs": selected_inputs,
            "masks": selector_outputs["selected_mask"].to(device=masks.device),
            "metas": output_metas,
            "losses": {},
            "event_surprise_selector_outputs": selector_outputs,
        }
        if str(mode) not in {"test", "predict", "eval"}:
            new_gt_segments, new_gt_labels = self._remap_gt_batch(
                gt_segments,
                gt_labels,
                selector_outputs["selected_dense_indices"],
                selector_outputs["selected_mask"],
            )
            detector_outputs["gt_segments"] = new_gt_segments
            detector_outputs["gt_labels"] = new_gt_labels
        return detector_outputs

    def forward(
        self,
        features: torch.Tensor,
        valid_mask: torch.Tensor,
        time_coords: torch.Tensor | None = None,
        metas: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
        mode: str = "train",
    ) -> dict[str, Any]:
        if str(mode) in {"test", "predict", "eval"} and metas is not None:
            _validate_deploy_visible_meta(metas)
        valid, coords = self._validate_inputs(features, valid_mask, time_coords)
        scores = self._event_scores(features, valid)
        return self._build_outputs(features, valid, coords, scores)

    def forward_train(
        self,
        features: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        time_coords: torch.Tensor | None = None,
        metas: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
        gt_segments: object = None,
        gt_labels: object = None,
        *,
        inputs: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if inputs is not None or masks is not None:
            if inputs is None or masks is None:
                raise ValueError("inputs and masks must be provided together")
            return self._forward_detector(
                inputs=inputs,
                masks=masks,
                metas=metas,
                time_coords=time_coords,
                gt_segments=gt_segments,
                gt_labels=gt_labels,
                mode="train",
            )
        if features is None or valid_mask is None:
            raise ValueError("features and valid_mask must be provided for standalone forward_train")
        return self.forward(features, valid_mask, time_coords=time_coords, metas=metas, mode="train")

    def forward_test(
        self,
        features: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
        time_coords: torch.Tensor | None = None,
        metas: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
        *,
        inputs: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if inputs is not None or masks is not None:
            if inputs is None or masks is None:
                raise ValueError("inputs and masks must be provided together")
            return self._forward_detector(
                inputs=inputs,
                masks=masks,
                metas=metas,
                time_coords=time_coords,
                mode="test",
            )
        if features is None or valid_mask is None:
            raise ValueError("features and valid_mask must be provided for standalone forward_test")
        return self.forward(features, valid_mask, time_coords=time_coords, metas=metas, mode="test")


__all__ = [
    "EVENT_SURPRISE_META_KEY",
    "EVENT_SURPRISE_ROUTE_LABEL",
    "EventSurpriseTemporalAcquisitionSelector",
]
