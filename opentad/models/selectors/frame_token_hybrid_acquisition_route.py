from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, MutableMapping, Sequence, Tuple

import torch
import torch.nn as nn

from ..builder import SELECTORS


ROUTE_LABEL = "DIVERGENT_INNOVATION_FRAME_TOKEN_HYBRID_DO_NOT_MERGE_WITH_C3"
DEFAULT_META_KEY = "frame_token_hybrid_acquisition_plan"
FORBIDDEN_TEST_META_TOKENS = (
    "gt",
    "ground_truth",
    "teacher",
    "oracle",
    "cache",
    "prediction",
    "raw_prediction",
    "result",
    "checkpoint",
    "ckpt",
)


def _as_bool_prefix_mask(masks: torch.Tensor, *, expected_shape: Tuple[int, int]) -> torch.Tensor:
    if masks.ndim != 2:
        raise ValueError(f"masks must be [B,T], got {tuple(masks.shape)}")
    if tuple(masks.shape) != tuple(expected_shape):
        raise ValueError(f"masks shape mismatch: expected {expected_shape}, got {tuple(masks.shape)}")
    if masks.dtype != torch.bool:
        if not torch.logical_or(masks == 0, masks == 1).all():
            raise ValueError("masks must be boolean or binary")
    valid = masks.bool()
    valid_count = valid.long().sum(dim=1)
    if torch.any(valid_count <= 0):
        raise ValueError("each sample must contain at least one valid position")
    prefix = torch.arange(valid.shape[1], device=valid.device)[None, :] < valid_count[:, None]
    if not torch.equal(valid, prefix):
        raise ValueError("FrameTokenHybridAcquisitionRoute requires prefix-contiguous masks")
    return valid


def _contains_forbidden_key(value: Any) -> str | None:
    if isinstance(value, MutableMapping):
        for key, item in value.items():
            key_text = str(key).lower()
            for token in FORBIDDEN_TEST_META_TOKENS:
                if token in key_text:
                    return str(key)
            nested = _contains_forbidden_key(item)
            if nested is not None:
                return nested
    elif isinstance(value, (list, tuple)):
        for item in value:
            nested = _contains_forbidden_key(item)
            if nested is not None:
                return nested
    elif isinstance(value, str):
        value_text = value.lower()
        for token in FORBIDDEN_TEST_META_TOKENS:
            if token in value_text:
                return value
    return None


@dataclass(frozen=True)
class _SpanToken:
    span_start: int
    span_end: int
    role: str
    visibility: float
    compression_confidence: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "span_start": int(self.span_start),
            "span_end": int(self.span_end),
            "role": self.role,
            "visibility": float(self.visibility),
            "compression_confidence": float(self.compression_confidence),
        }


@SELECTORS.register_module()
class FrameTokenHybridAcquisitionRoute(nn.Module):
    """Frame/token hybrid acquisition route.

    The route keeps deploy-visible raw frame observations at boundary and anchor
    positions, compresses long stable regions into span-token metadata, then
    reconstructs a dense temporal axis for ActionFormer-compatible downstream
    modules.
    """

    def __init__(
        self,
        target_len: int = 384,
        dense_window_size: int = 768,
        target_dense_len: int = 768,
        anchor_stride: int = 24,
        boundary_radius: int = 2,
        boundary_epsilon: float = 0.25,
        stable_gap_min_len: int = 12,
        stable_epsilon: float = 0.02,
        max_span_tokens: int = 64,
        route_label: str = ROUTE_LABEL,
        meta_key: str = DEFAULT_META_KEY,
    ) -> None:
        super().__init__()
        if int(target_len) <= 0:
            raise ValueError("target_len must be positive")
        if int(dense_window_size) <= 0:
            raise ValueError("dense_window_size must be positive")
        if int(target_dense_len) <= 0:
            raise ValueError("target_dense_len must be positive")
        if int(anchor_stride) <= 0:
            raise ValueError("anchor_stride must be positive")
        if int(boundary_radius) < 0:
            raise ValueError("boundary_radius must be non-negative")
        if int(stable_gap_min_len) <= 0:
            raise ValueError("stable_gap_min_len must be positive")
        if int(max_span_tokens) < 0:
            raise ValueError("max_span_tokens must be non-negative")
        self.target_len = int(target_len)
        self.dense_window_size = int(dense_window_size)
        self.target_dense_len = int(target_dense_len)
        self.anchor_stride = int(anchor_stride)
        self.boundary_radius = int(boundary_radius)
        self.boundary_epsilon = float(boundary_epsilon)
        self.stable_gap_min_len = int(stable_gap_min_len)
        self.stable_epsilon = float(stable_epsilon)
        self.max_span_tokens = int(max_span_tokens)
        self.route_label = str(route_label)
        self.meta_key = str(meta_key)

    def forward_train(self, inputs, masks, metas, gt_segments, gt_labels):
        outputs = self._forward_impl(inputs, masks, metas, reject_forbidden_meta=False)
        outputs["gt_segments"] = gt_segments
        outputs["gt_labels"] = gt_labels
        return outputs

    def forward_test(self, inputs, masks, metas=None):
        return self._forward_impl(inputs, masks, metas, reject_forbidden_meta=True)

    def _forward_impl(self, inputs: torch.Tensor, masks: torch.Tensor, metas, *, reject_forbidden_meta: bool):
        if inputs.ndim != 5:
            raise ValueError(f"inputs must be [B,C,T,H,W], got {tuple(inputs.shape)}")
        batch, _channels, dense_len, _height, _width = inputs.shape
        if dense_len != self.target_dense_len:
            raise ValueError(f"target_dense_len={self.target_dense_len} must match input dense axis {dense_len}")
        if dense_len != self.dense_window_size:
            raise ValueError(f"dense_window_size={self.dense_window_size} must match input dense axis {dense_len}")
        valid = _as_bool_prefix_mask(masks, expected_shape=(batch, dense_len))
        if metas is None:
            metas = [{} for _idx in range(batch)]
        if len(metas) != batch:
            raise ValueError(f"metas length mismatch: expected {batch}, got {len(metas)}")
        if reject_forbidden_meta:
            for meta in metas:
                forbidden = _contains_forbidden_key(meta)
                if forbidden is not None:
                    raise ValueError(f"forbidden test-time meta key for frame/token hybrid route: {forbidden}")

        frame_signal = inputs.detach().to(dtype=torch.float32).mean(dim=(1, 3, 4))
        plans = [
            self._build_plan_for_sample(
                valid_len=int(valid[idx].long().sum().item()),
                signal=frame_signal[idx].detach().cpu().tolist(),
            )
            for idx in range(batch)
        ]
        completed = self._dense_complete(inputs, plans)
        output_metas = self._write_metas(metas, plans)
        return {
            "inputs": completed,
            "masks": valid,
            "metas": output_metas,
            "observed_positions": [
                torch.tensor(plan["observed_positions"], dtype=torch.long, device=inputs.device)
                for plan in plans
            ],
            "span_tokens": [plan["span_tokens"] for plan in plans],
        }

    def _build_plan_for_sample(self, *, valid_len: int, signal: Sequence[float]) -> Dict[str, Any]:
        valid_len = int(valid_len)
        observed = set(self._anchor_positions(valid_len))
        boundary_positions = self._boundary_positions(signal[:valid_len])
        for center in boundary_positions:
            for offset in range(-self.boundary_radius, self.boundary_radius + 1):
                position = center + offset
                if 0 <= position < valid_len:
                    observed.add(int(position))
        span_tokens = self._stable_span_tokens(signal[:valid_len], valid_len=valid_len)
        for token in span_tokens:
            observed.add(token.span_start)
            observed.add(token.span_end)

        observed_positions = sorted(observed)
        if len(observed_positions) > self.target_len:
            protected = {0, valid_len - 1}
            protected.update(boundary_positions)
            ranked = sorted(
                observed_positions,
                key=lambda pos: (0 if pos in protected else 1, pos),
            )[: self.target_len]
            observed_positions = sorted(ranked)
        observed_set = set(observed_positions)
        span_dicts = [
            token.as_dict()
            for token in span_tokens
            if token.span_start in observed_set and token.span_end in observed_set
        ]
        return {
            "valid_len": valid_len,
            "observed_positions": observed_positions,
            "boundary_positions": boundary_positions,
            "span_tokens": span_dicts,
        }

    def _anchor_positions(self, valid_len: int) -> Iterable[int]:
        yield 0
        if valid_len > 1:
            yield valid_len - 1
        for position in range(0, valid_len, self.anchor_stride):
            yield int(position)

    def _boundary_positions(self, signal: Sequence[float]) -> List[int]:
        if len(signal) <= 1:
            return []
        diffs = [abs(float(signal[idx]) - float(signal[idx - 1])) for idx in range(1, len(signal))]
        return [idx for idx, value in enumerate(diffs, start=1) if value > self.boundary_epsilon]

    def _stable_span_tokens(self, signal: Sequence[float], *, valid_len: int) -> List[_SpanToken]:
        if valid_len <= 1 or self.max_span_tokens == 0:
            return []
        stable = [False]
        for idx in range(1, valid_len):
            stable.append(abs(float(signal[idx]) - float(signal[idx - 1])) <= self.stable_epsilon)

        spans: List[_SpanToken] = []
        run_start = None
        for idx in range(1, valid_len):
            if stable[idx] and run_start is None:
                run_start = idx - 1
            if (not stable[idx] or idx == valid_len - 1) and run_start is not None:
                run_end = idx if stable[idx] and idx == valid_len - 1 else idx - 1
                span_len = run_end - run_start + 1
                if span_len >= self.stable_gap_min_len:
                    visibility = min(1.0, 2.0 / float(span_len))
                    confidence = max(0.0, min(1.0, 1.0 - self._mean_abs_step(signal, run_start, run_end)))
                    spans.append(
                        _SpanToken(
                            span_start=int(run_start),
                            span_end=int(run_end),
                            role="stable_gap_span_token",
                            visibility=visibility,
                            compression_confidence=confidence,
                        )
                    )
                run_start = None
            if len(spans) >= self.max_span_tokens:
                break
        return spans

    @staticmethod
    def _mean_abs_step(signal: Sequence[float], start: int, end: int) -> float:
        if end <= start:
            return 0.0
        values = [abs(float(signal[idx]) - float(signal[idx - 1])) for idx in range(start + 1, end + 1)]
        return sum(values) / max(1, len(values))

    def _dense_complete(self, inputs: torch.Tensor, plans: Sequence[Dict[str, Any]]) -> torch.Tensor:
        completed = torch.zeros_like(inputs)
        for batch_idx, plan in enumerate(plans):
            observed = sorted(int(pos) for pos in plan["observed_positions"])
            if not observed:
                continue
            valid_len = int(plan["valid_len"])
            observed_set = set(observed)
            for position in observed:
                if 0 <= position < valid_len:
                    completed[batch_idx, :, position] = inputs[batch_idx, :, position]

            if len(observed) == 1:
                completed[batch_idx, :, :valid_len] = inputs[batch_idx, :, observed[0]][:, None]
                continue

            span_tokens = list(plan["span_tokens"])
            for left_pos, right_pos in zip(observed[:-1], observed[1:]):
                if right_pos <= left_pos:
                    continue
                left = inputs[batch_idx, :, left_pos]
                right = inputs[batch_idx, :, right_pos]
                for position in range(left_pos + 1, min(right_pos, valid_len)):
                    if position in observed_set:
                        completed[batch_idx, :, position] = inputs[batch_idx, :, position]
                        continue
                    alpha = float(position - left_pos) / float(right_pos - left_pos)
                    interpolated = left * (1.0 - alpha) + right * alpha
                    span_token = self._span_token_covering_position(span_tokens, position)
                    completed[batch_idx, :, position] = self._condition_dense_completion(
                        interpolated=interpolated,
                        left=left,
                        right=right,
                        span_token=span_token,
                    )
            for position in range(valid_len):
                if position in observed_set:
                    completed[batch_idx, :, position] = inputs[batch_idx, :, position]
        return completed

    @staticmethod
    def _span_token_covering_position(span_tokens: Sequence[Dict[str, Any]], position: int) -> Dict[str, Any] | None:
        for token in span_tokens:
            start = int(token["span_start"])
            end = int(token["span_end"])
            if start < int(position) < end:
                return token
        return None

    @staticmethod
    def _condition_dense_completion(
        *,
        interpolated: torch.Tensor,
        left: torch.Tensor,
        right: torch.Tensor,
        span_token: Dict[str, Any] | None,
    ) -> torch.Tensor:
        if span_token is None or span_token.get("role") != "stable_gap_span_token":
            return interpolated
        visibility = max(0.0, min(1.0, float(span_token["visibility"])))
        confidence = max(0.0, min(1.0, float(span_token["compression_confidence"])))
        span_strength = confidence * (1.0 - visibility)
        stable_level = (left + right) * 0.5
        return interpolated * (1.0 - span_strength) + stable_level * span_strength

    def _write_metas(self, metas: Sequence[Dict[str, Any]], plans: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        output = []
        conditioning_keys = [
            "span_start",
            "span_end",
            "role",
            "visibility",
            "compression_confidence",
        ]
        for meta, plan in zip(metas, plans):
            item = dict(meta)
            observed = [int(pos) for pos in plan["observed_positions"]]
            span_tokens = [dict(token) for token in plan["span_tokens"]]
            item["frame_token_hybrid_observed_raw_positions"] = observed
            item["irregular_selected_positions"] = [float(pos) for pos in observed]
            item["irregular_selected_output_valid_len"] = float(len(observed))
            item["irregular_selected_valid_len"] = float(plan["valid_len"])
            item["frame_token_hybrid_bridge"] = {
                "schema_version": "frame_token_hybrid_dense_completion_v0",
                "dense_completion_conditioning_keys": conditioning_keys,
                "completion_rule": "raw_observed_positions_are_copied; unobserved_valid_positions_are_interpolated_and_span_conditioned; invalid_mask_suffix_is_zero",
                "preserves_observed_raw_positions": True,
                "output_dense_axis_len": self.target_dense_len,
            }
            item[self.meta_key] = {
                "route_label": self.route_label,
                "meta_key": self.meta_key,
                "selection_surface": "frame_token_hybrid_pre_backbone",
                "selection_timing": "online_before_backbone",
                "acquisition_unit": "raw_frame_observation_plus_span_token",
                "strategy": "boundary_anchor_raw_frames_stable_gap_span_tokens_dense_completion",
                "target_len": self.target_len,
                "dense_window_size": self.dense_window_size,
                "output_dense_axis_len": self.target_dense_len,
                "observed_raw_frame_count": len(observed),
                "span_token_count": len(span_tokens),
                "boundary_positions": [int(pos) for pos in plan["boundary_positions"]],
                "span_tokens": span_tokens,
                "uses_gt": False,
                "uses_teacher": False,
                "uses_oracle": False,
                "uses_raw_prediction_cache": False,
                "uses_detector_outputs": False,
                "deploy_time_signals": ["frame_mean", "temporal_difference", "valid_prefix_mask"],
            }
            output.append(item)
        return output
