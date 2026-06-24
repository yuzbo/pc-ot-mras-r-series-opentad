from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, MutableMapping, Sequence, Tuple

import torch
import torch.nn as nn

from ..builder import SELECTORS


ROUTE_LABEL = "DIVERGENT_INNOVATION_BOUNDARY_MICROSCOPE_DO_NOT_MERGE_WITH_C3"
DEFAULT_META_KEY = "boundary_microscope_acquisition_plan"
FORBIDDEN_TEST_META_TOKENS = (
    "gt",
    "ground_truth",
    "teacher",
    "oracle",
    "raw_prediction",
    "raw_predictions",
    "prediction_cache",
    "detector_cache",
    "result_detection",
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
    if torch.any(valid.long().sum(dim=1) <= 0):
        raise ValueError("each sample must contain at least one valid position")
    valid_count = valid.long().sum(dim=1)
    prefix = torch.arange(valid.shape[1], device=valid.device)[None, :] < valid_count[:, None]
    if not torch.equal(valid, prefix):
        raise ValueError("BoundaryMicroscopeAcquisitionRoute requires prefix-contiguous masks")
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


def _normalize_valid(signal: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    masked = signal.masked_fill(~valid, 0.0)
    count = valid.long().sum(dim=1, keepdim=True).clamp_min(1).to(dtype=signal.dtype)
    mean = masked.sum(dim=1, keepdim=True) / count
    centered = (signal - mean).masked_fill(~valid, 0.0)
    max_abs = centered.abs().amax(dim=1, keepdim=True).clamp_min(1e-6)
    return (centered / max_abs).masked_fill(~valid, 0.0)


def _positive_segments(values: Sequence[float], *, threshold: float) -> List[Tuple[int, int]]:
    segments: List[Tuple[int, int]] = []
    start = None
    for idx, value in enumerate(values):
        if value > threshold and start is None:
            start = idx
        elif value <= threshold and start is not None:
            segments.append((start, idx))
            start = None
    if start is not None:
        segments.append((start, len(values)))
    return segments


@dataclass(frozen=True)
class _Candidate:
    position: int
    role: str
    priority: int


@SELECTORS.register_module()
class BoundaryMicroscopeAcquisitionRoute(nn.Module):
    """Boundary-focused sparse acquisition route.

    The route uses only deploy-visible frame content and masks. A cheap global
    scanner estimates action and boundary hazards, microscope packets densify
    likely start/end neighborhoods, and sparse anchors preserve interior and
    background temporal coverage.
    """

    forbid_raw_prediction_cache = True

    def __init__(
        self,
        target_len: int = 384,
        dense_window_size: int = 768,
        microscope_radius: int = 3,
        microscope_stride: int = 1,
        anchor_stride: int = 24,
        max_dense_gap: int = 8,
        max_start_hazards: int = 4,
        max_end_hazards: int = 4,
        action_threshold: float = 0.15,
        hazard_epsilon: float = 1e-6,
        route_label: str = ROUTE_LABEL,
        meta_key: str = DEFAULT_META_KEY,
    ) -> None:
        super().__init__()
        if int(target_len) <= 0:
            raise ValueError("target_len must be positive")
        if int(dense_window_size) <= 0:
            raise ValueError("dense_window_size must be positive")
        if int(microscope_radius) < 0:
            raise ValueError("microscope_radius must be non-negative")
        if int(microscope_stride) <= 0:
            raise ValueError("microscope_stride must be positive")
        if int(anchor_stride) <= 0:
            raise ValueError("anchor_stride must be positive")
        if int(max_dense_gap) <= 0:
            raise ValueError("max_dense_gap must be positive")
        if int(max_start_hazards) < 0:
            raise ValueError("max_start_hazards must be non-negative")
        if int(max_end_hazards) < 0:
            raise ValueError("max_end_hazards must be non-negative")
        if float(hazard_epsilon) < 0.0:
            raise ValueError("hazard_epsilon must be non-negative")
        if str(route_label) != ROUTE_LABEL:
            raise ValueError("BoundaryMicroscopeAcquisitionRoute route_label must preserve the divergent route label")
        if str(meta_key) != DEFAULT_META_KEY:
            raise ValueError("BoundaryMicroscopeAcquisitionRoute meta_key must preserve the route metadata schema")
        self.target_len = int(target_len)
        self.dense_window_size = int(dense_window_size)
        self.microscope_radius = int(microscope_radius)
        self.microscope_stride = int(microscope_stride)
        self.anchor_stride = int(anchor_stride)
        self.max_dense_gap = int(max_dense_gap)
        self.max_start_hazards = int(max_start_hazards)
        self.max_end_hazards = int(max_end_hazards)
        self.action_threshold = float(action_threshold)
        self.hazard_epsilon = float(hazard_epsilon)
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
        if int(dense_len) != self.dense_window_size:
            raise ValueError(
                f"dense_window_size={self.dense_window_size} must match input dense axis {int(dense_len)}"
            )
        valid = _as_bool_prefix_mask(masks, expected_shape=(batch, dense_len))
        if metas is None:
            metas = [{} for _idx in range(batch)]
        if len(metas) != batch:
            raise ValueError(f"metas length mismatch: expected {batch}, got {len(metas)}")
        if reject_forbidden_meta:
            for meta in metas:
                forbidden = _contains_forbidden_key(meta)
                if forbidden is not None:
                    raise ValueError(f"forbidden test-time meta key for boundary microscope route: {forbidden}")

        scores = self._cheap_global_scan(inputs, valid)
        plans = [
            self._build_plan_for_sample(
                valid_len=int(valid[idx].long().sum().item()),
                action_scores=scores["actionness"][idx].detach().cpu().tolist(),
                start_scores=scores["start_hazard"][idx].detach().cpu().tolist(),
                end_scores=scores["end_hazard"][idx].detach().cpu().tolist(),
            )
            for idx in range(batch)
        ]
        max_selected = max(len(plan["indices"]) for plan in plans)
        gather_indices = torch.zeros((batch, max_selected), dtype=torch.long, device=inputs.device)
        selected_masks = torch.zeros((batch, max_selected), dtype=torch.bool, device=inputs.device)
        for idx, plan in enumerate(plans):
            selected = torch.tensor(plan["indices"], dtype=torch.long, device=inputs.device)
            gather_indices[idx, : selected.numel()] = selected
            if selected.numel() < max_selected:
                gather_indices[idx, selected.numel() :] = selected[-1]
            selected_masks[idx, : selected.numel()] = True

        expanded = gather_indices[:, None, :, None, None].expand(
            -1,
            inputs.shape[1],
            -1,
            inputs.shape[3],
            inputs.shape[4],
        )
        selected_inputs = torch.gather(inputs, dim=2, index=expanded)
        output_metas = self._write_metas(metas, plans)
        return {
            "inputs": selected_inputs,
            "masks": selected_masks,
            "metas": output_metas,
            "selected_positions": gather_indices.to(dtype=torch.float32),
            "selected_output_valid_lengths": selected_masks.long().sum(dim=1),
            "scanner_scores": scores,
        }

    def _cheap_global_scan(self, inputs: torch.Tensor, valid: torch.Tensor) -> Dict[str, torch.Tensor]:
        frame_signal = inputs.detach().to(dtype=torch.float32).mean(dim=(1, 3, 4))
        actionness = _normalize_valid(frame_signal, valid).clamp_min(0.0)
        diff = frame_signal.new_zeros(frame_signal.shape)
        diff[:, 1:] = frame_signal[:, 1:] - frame_signal[:, :-1]
        diff = diff.masked_fill(~valid, 0.0)
        start_hazard = diff.clamp_min(0.0).masked_fill(~valid, 0.0)
        end_hazard = (-diff).clamp_min(0.0).masked_fill(~valid, 0.0)
        boundary_hazard = torch.maximum(start_hazard, end_hazard)
        return {
            "actionness": actionness,
            "start_hazard": start_hazard,
            "end_hazard": end_hazard,
            "boundary_hazard": boundary_hazard,
        }

    def _build_plan_for_sample(
        self,
        *,
        valid_len: int,
        action_scores: Sequence[float],
        start_scores: Sequence[float],
        end_scores: Sequence[float],
    ) -> Dict[str, Any]:
        valid_len = int(valid_len)
        if valid_len <= 0:
            raise ValueError("valid_len must be positive")
        if valid_len > self.dense_window_size:
            raise ValueError("valid_len cannot exceed dense_window_size")
        effective_budget = min(self.target_len, valid_len)
        if effective_budget <= 1 and valid_len > 1:
            raise ValueError("target_len cannot preserve temporal coverage for valid_len > 1")
        if effective_budget > 1 and (effective_budget - 1) * self.max_dense_gap < valid_len - 1:
            raise ValueError("target_len and max_dense_gap cannot cover the valid temporal span")
        candidates: Dict[int, _Candidate] = {}
        start_hazards = self._top_positive_positions(start_scores[:valid_len], self.max_start_hazards)
        end_hazards = self._top_positive_positions(end_scores[:valid_len], self.max_end_hazards)

        for center in start_hazards:
            self._add_packet(candidates, center=center, valid_len=valid_len, role="start_microscope_packet", priority=0)
        for center in end_hazards:
            self._add_packet(candidates, center=center, valid_len=valid_len, role="end_microscope_packet", priority=0)

        for position, role in self._anchor_positions(valid_len=valid_len, action_scores=action_scores[:valid_len]):
            self._add_candidate(candidates, position, role, priority=3)

        self._fill_max_gap(candidates, valid_len=valid_len)
        indices, roles = self._finalize_candidates(candidates)
        if len(indices) > self.target_len:
            kept = sorted(
                candidates.values(),
                key=lambda item: (item.priority, item.position),
            )[: self.target_len]
            kept = sorted(kept, key=lambda item: item.position)
            indices = [item.position for item in kept]
            roles = [item.role for item in kept]

        gaps = [right - left for left, right in zip(indices[:-1], indices[1:])]
        if gaps and max(gaps) > self.max_dense_gap:
            raise ValueError("BoundaryMicroscopeAcquisitionRoute selected indices violate max_dense_gap")
        return {
            "indices": indices,
            "roles": roles,
            "start_hazard_positions": start_hazards,
            "end_hazard_positions": end_hazards,
            "max_gap": max(gaps) if gaps else 0,
            "max_gap_satisfied": (max(gaps) if gaps else 0) <= self.max_dense_gap,
            "valid_len": valid_len,
        }

    def _top_positive_positions(self, scores: Sequence[float], limit: int) -> List[int]:
        ranked = [
            (float(score), int(idx))
            for idx, score in enumerate(scores)
            if float(score) > self.hazard_epsilon
        ]
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [idx for _score, idx in ranked[: max(0, int(limit))]]

    def _add_packet(self, candidates: Dict[int, _Candidate], *, center: int, valid_len: int, role: str, priority: int) -> None:
        for offset in range(-self.microscope_radius, self.microscope_radius + 1, self.microscope_stride):
            self._add_candidate(candidates, center + offset, role, priority=priority, valid_len=valid_len)

    def _anchor_positions(self, *, valid_len: int, action_scores: Sequence[float]) -> Iterable[Tuple[int, str]]:
        yield 0, "background_anchor"
        if valid_len > 1:
            yield valid_len - 1, "background_anchor"
        for position in range(0, valid_len, self.anchor_stride):
            score = float(action_scores[position]) if position < len(action_scores) else 0.0
            role = "interior_anchor" if score > self.action_threshold else "background_anchor"
            yield position, role
        segments = _positive_segments(action_scores, threshold=self.action_threshold)
        for start, end in segments:
            if end - start >= 3:
                yield (start + end - 1) // 2, "interior_anchor"

    def _fill_max_gap(self, candidates: Dict[int, _Candidate], *, valid_len: int) -> None:
        if valid_len <= 1:
            return
        self._add_candidate(candidates, 0, "background_anchor", priority=3, valid_len=valid_len)
        self._add_candidate(candidates, valid_len - 1, "background_anchor", priority=3, valid_len=valid_len)
        while True:
            ordered = sorted(candidates)
            gaps = [(right - left, left, right) for left, right in zip(ordered[:-1], ordered[1:])]
            oversized = [(gap, left, right) for gap, left, right in gaps if gap > self.max_dense_gap]
            if not oversized:
                break
            gap, left, right = max(oversized, key=lambda item: (item[0], -item[1]))
            midpoint = (left + right) // 2
            if midpoint in candidates or midpoint <= left or midpoint >= right:
                break
            self._add_candidate(candidates, midpoint, "gap_guard_anchor", priority=2, valid_len=valid_len)
            if len(candidates) >= self.target_len:
                break

    def _add_candidate(
        self,
        candidates: Dict[int, _Candidate],
        position: int,
        role: str,
        *,
        priority: int,
        valid_len: int | None = None,
    ) -> None:
        if valid_len is not None and (position < 0 or position >= valid_len):
            return
        if position < 0:
            return
        existing = candidates.get(int(position))
        candidate = _Candidate(position=int(position), role=str(role), priority=int(priority))
        if existing is None or candidate.priority < existing.priority:
            candidates[int(position)] = candidate

    def _finalize_candidates(self, candidates: Dict[int, _Candidate]) -> Tuple[List[int], List[str]]:
        ordered = [candidates[position] for position in sorted(candidates)]
        return [item.position for item in ordered], [item.role for item in ordered]

    def _write_metas(self, metas: Sequence[Dict[str, Any]], plans: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        output = []
        for meta, plan in zip(metas, plans):
            item = dict(meta)
            indices = [int(pos) for pos in plan["indices"]]
            roles = [str(role) for role in plan["roles"]]
            item["boundary_microscope_selected_dense_indices"] = indices
            item["boundary_microscope_selected_roles"] = roles
            item["irregular_selected_positions"] = [float(pos) for pos in indices]
            item["irregular_selected_output_valid_len"] = float(len(indices))
            item["irregular_selected_valid_len"] = float(plan["valid_len"])
            item["irregular_selected_count"] = int(len(indices))
            item[self.meta_key] = {
                "route_label": self.route_label,
                "meta_key": self.meta_key,
                "selection_surface": "pre_backbone_raw_frame",
                "selection_timing": "online_before_backbone",
                "acquisition_unit": "frame",
                "strategy": "cheap_global_boundary_scanner_dense_microscope_packets_sparse_anchors",
                "budget": len(indices),
                "budget_contract": "up_to_target_len_fail_closed_max_gap",
                "target_len": self.target_len,
                "dense_window_size": self.dense_window_size,
                "microscope_radius": self.microscope_radius,
                "microscope_stride": self.microscope_stride,
                "anchor_stride": self.anchor_stride,
                "max_dense_gap": self.max_dense_gap,
                "observed_max_gap": int(plan["max_gap"]),
                "max_gap_satisfied": bool(plan["max_gap_satisfied"]),
                "start_hazard_positions": [int(pos) for pos in plan["start_hazard_positions"]],
                "end_hazard_positions": [int(pos) for pos in plan["end_hazard_positions"]],
                "uses_gt": False,
                "uses_teacher": False,
                "uses_raw_prediction_cache": False,
                "uses_detector_outputs": False,
                "deploy_time_signals": ["frame_mean", "temporal_difference", "valid_prefix_mask"],
            }
            output.append(item)
        return output
