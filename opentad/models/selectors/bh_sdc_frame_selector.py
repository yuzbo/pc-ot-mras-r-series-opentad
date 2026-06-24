from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import SELECTORS


_MASKED_LOW_LOGIT = -1.0e4
BH_SDC_ROUTE_LABEL = "DIVERGENT_INNOVATION_BH_SDC_DO_NOT_MERGE_WITH_C3"
_FORBIDDEN_TEST_META_TOKENS = frozenset(
    {
        "gt",
        "teacher",
        "oracle",
        "cache",
        "prediction",
        "predictions",
        "target",
        "targets",
        "checkpoint",
        "ckpt",
        "result",
        "results",
    }
)
_FORBIDDEN_TEST_META_PHRASES = frozenset(
    {
        "ground_truth",
        "groundtruth",
        "raw_prediction",
        "raw_predictions",
        "prediction_cache",
        "value_target",
        "value_targets",
        "pc_ot_mras_value_targets",
    }
)


@dataclass
class AcquisitionPlan:
    selected_dense_indices: torch.Tensor
    selected_mask: torch.Tensor
    physical_times: torch.Tensor
    budget: torch.Tensor
    roles: dict[str, torch.Tensor]
    diagnostics: dict[str, torch.Tensor]


def _require_finite(tensor: torch.Tensor, name: str) -> None:
    if torch.is_complex(tensor):
        raise ValueError(f"{name} must be real-valued")
    if torch.is_floating_point(tensor) and not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must be finite")


def _prefix_mask(mask: torch.Tensor, *, expected_shape: tuple[int, int], name: str = "mask") -> torch.Tensor:
    if not torch.is_tensor(mask):
        raise TypeError(f"{name} must be a tensor")
    if mask.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {tuple(mask.shape)}")
    if mask.dtype != torch.bool and not bool(((mask == 0) | (mask == 1)).all().item()):
        raise ValueError(f"{name} must be boolean or binary")
    valid = mask.bool()
    counts = valid.long().sum(dim=1)
    if bool((counts <= 0).any().item()):
        raise ValueError(f"{name} must contain at least one valid temporal position per sample")
    prefix = torch.arange(valid.shape[1], device=valid.device)[None, :] < counts[:, None]
    if not torch.equal(valid, prefix):
        raise ValueError(f"{name} must be a contiguous valid prefix")
    return valid


def _temporal_dim(inputs: torch.Tensor) -> int:
    if inputs.ndim == 3:
        return 2
    if inputs.ndim == 5:
        return 2
    if inputs.ndim == 6:
        return 3
    raise ValueError(
        "BH-SDC frame selector expects [B,C,T], [B,C,T,H,W], or [B,N,C,T,H,W] inputs; "
        f"got {tuple(inputs.shape)}"
    )


def _temporal_len(inputs: torch.Tensor) -> int:
    return int(inputs.shape[_temporal_dim(inputs)])


def _deploy_protocol_flags() -> dict[str, bool | str]:
    return {
        "protocol": "bh_sdc_deploy_visible_v1",
        "route_label": BH_SDC_ROUTE_LABEL,
        "uses_gt": False,
        "uses_teacher": False,
        "uses_oracle": False,
        "uses_cache": False,
        "uses_raw_prediction_cache": False,
    }


def _normalized_text(value: object) -> str:
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(value or ""))
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", text)
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text)


def _forbidden_test_meta_key(value: object) -> bool:
    normalized = _normalized_text(value)
    tokens = [token for token in normalized.strip("_").split("_") if token]
    compact = "".join(tokens)
    if any(token in _FORBIDDEN_TEST_META_TOKENS for token in tokens):
        return True
    if "ground" in tokens and "truth" in tokens:
        return True
    if compact.startswith("gt"):
        return True
    return any(phrase in normalized or phrase in compact for phrase in _FORBIDDEN_TEST_META_PHRASES)


def _reject_forbidden_test_meta(value: object, *, location: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if _forbidden_test_meta_key(key):
                raise ValueError(f"{location}.{key} contains forbidden deploy/test-time payload")
            _reject_forbidden_test_meta(item, location=f"{location}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_forbidden_test_meta(item, location=f"{location}[{index}]")
        return
    if isinstance(value, str) and _forbidden_test_meta_key(value):
        raise ValueError(f"{location} contains forbidden deploy/test-time payload")


def _reject_forbidden_test_metas(metas) -> None:
    if metas is None:
        return
    for index, meta in enumerate(metas):
        _reject_forbidden_test_meta(meta, location=f"metas[{index}]")


def _feature_sequence_for_scout(inputs: torch.Tensor) -> torch.Tensor:
    if inputs.ndim == 3:
        return inputs.float()
    if inputs.ndim == 5:
        return inputs.float().mean(dim=(-1, -2))
    if inputs.ndim == 6:
        return inputs.float().mean(dim=(1, -1, -2))
    raise ValueError(f"unsupported input shape {tuple(inputs.shape)}")


def _gather_temporal(inputs: torch.Tensor, indices: torch.Tensor, selected_mask: torch.Tensor) -> torch.Tensor:
    time_dim = _temporal_dim(inputs)
    if inputs.shape[0] != indices.shape[0]:
        raise ValueError("indices batch size must match inputs")
    safe_indices = indices.to(device=inputs.device, dtype=torch.long).clamp(min=0, max=inputs.shape[time_dim] - 1)
    if inputs.ndim == 3:
        gathered = inputs.gather(2, safe_indices[:, None, :].expand(-1, inputs.shape[1], -1))
        mask_shape = (selected_mask.shape[0], 1, selected_mask.shape[1])
    elif inputs.ndim == 5:
        gathered = inputs.gather(
            2,
            safe_indices[:, None, :, None, None].expand(
                -1,
                inputs.shape[1],
                -1,
                inputs.shape[3],
                inputs.shape[4],
            ),
        )
        mask_shape = (selected_mask.shape[0], 1, selected_mask.shape[1], 1, 1)
    elif inputs.ndim == 6:
        gathered = inputs.gather(
            3,
            safe_indices[:, None, None, :, None, None].expand(
                -1,
                inputs.shape[1],
                inputs.shape[2],
                -1,
                inputs.shape[4],
                inputs.shape[5],
            ),
        )
        mask_shape = (selected_mask.shape[0], 1, 1, selected_mask.shape[1], 1, 1)
    else:
        raise ValueError(f"unsupported input shape {tuple(inputs.shape)}")
    return gathered * selected_mask.to(device=inputs.device, dtype=gathered.dtype).view(mask_shape)


def _valid_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    masked = values.masked_fill(~valid, 0.0)
    denom = valid.to(dtype=values.dtype).sum(dim=1).clamp_min(1.0)
    return masked.sum(dim=1) / denom


def _linspace_anchors(valid_len: int, count: int, *, device: torch.device) -> list[int]:
    if count <= 0:
        return []
    if count >= valid_len:
        return list(range(valid_len))
    raw = torch.linspace(0, valid_len - 1, steps=count, device=device).round().to(torch.long)
    out: list[int] = []
    for item in raw.detach().cpu().tolist():
        idx = int(item)
        if idx not in out:
            out.append(idx)
    return out


@SELECTORS.register_module()
class BoundaryHazardTemporalScout(nn.Module):
    """Local temporal CNN scout for deploy-visible boundary hazard signals."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        kernel_size: int = 5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if int(in_channels) <= 0:
            raise ValueError("in_channels must be positive")
        if int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        if int(num_layers) <= 0:
            raise ValueError("num_layers must be positive")
        if int(kernel_size) <= 0 or int(kernel_size) % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")

        layers: list[nn.Module] = []
        current = int(in_channels)
        for _idx in range(int(num_layers)):
            layers.append(nn.Conv1d(current, int(hidden_dim), kernel_size, padding=kernel_size // 2))
            layers.append(nn.GroupNorm(1, int(hidden_dim)))
            layers.append(nn.GELU())
            if float(dropout) > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            current = int(hidden_dim)
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Conv1d(int(hidden_dim), 7, kernel_size=1)

    def forward(self, features: torch.Tensor, valid_mask: torch.Tensor, metas: Sequence[Mapping[str, Any]] | None = None):
        if features.ndim != 3:
            raise ValueError(f"features must be [B,C,T], got {tuple(features.shape)}")
        batch, _channels, time = features.shape
        valid = _prefix_mask(valid_mask, expected_shape=(batch, time), name="valid_mask").to(device=features.device)
        x = features.float().masked_fill(~valid[:, None, :], 0.0)
        _require_finite(x, "scout features")
        logits = self.head(self.encoder(x))
        keys = (
            "actionness_logits",
            "start_hazard_logits",
            "end_hazard_logits",
            "difficulty_logits",
            "uncertainty_logits",
            "redundancy_logits",
            "boundary_logits",
        )
        outputs = {}
        for idx, key in enumerate(keys):
            value = logits[:, idx, :]
            fill_value = _MASKED_LOW_LOGIT
            if key == "redundancy_logits":
                fill_value = -_MASKED_LOW_LOGIT
            value = value.masked_fill(~valid, fill_value)
            _require_finite(value, key)
            outputs[key] = value
        outputs["frame_selection_logits"] = (
            outputs["actionness_logits"]
            + 0.5 * outputs["boundary_logits"]
            + 0.25 * outputs["difficulty_logits"]
            + 0.25 * outputs["uncertainty_logits"]
            - 0.25 * outputs["redundancy_logits"]
        ).masked_fill(~valid, _MASKED_LOW_LOGIT)
        outputs["valid_mask"] = valid
        outputs["protocol"] = _deploy_protocol_flags()
        return outputs


@SELECTORS.register_module()
class BoundaryHazardDynamicBudgetController(nn.Module):
    """Map deploy-visible scout hazards into per-sample acquisition budgets."""

    def __init__(
        self,
        min_budget: int,
        target_budget: int,
        max_budget: int,
        budget_step: int = 1,
        hazard_weight: float = 2.0,
        difficulty_weight: float = 1.0,
        uncertainty_weight: float = 1.0,
        redundancy_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.min_budget = int(min_budget)
        self.target_budget = int(target_budget)
        self.max_budget = int(max_budget)
        self.budget_step = int(budget_step)
        if self.min_budget <= 0 or self.target_budget <= 0 or self.max_budget <= 0:
            raise ValueError("budgets must be positive")
        if not self.min_budget <= self.target_budget <= self.max_budget:
            raise ValueError("must satisfy min_budget <= target_budget <= max_budget")
        if self.budget_step <= 0:
            raise ValueError("budget_step must be positive")
        self.hazard_weight = float(hazard_weight)
        self.difficulty_weight = float(difficulty_weight)
        self.uncertainty_weight = float(uncertainty_weight)
        self.redundancy_weight = float(redundancy_weight)
        target_fraction = (self.target_budget - self.min_budget) / float(self.max_budget - self.min_budget)
        target_fraction = min(max(target_fraction, 1.0e-4), 1.0 - 1.0e-4)
        self._target_budget_logit_bias = float(torch.logit(torch.tensor(target_fraction)).item())

    def forward(self, scout_out: Mapping[str, torch.Tensor], valid_mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        valid = valid_mask.bool()
        for key in (
            "actionness_logits",
            "start_hazard_logits",
            "end_hazard_logits",
            "boundary_logits",
            "difficulty_logits",
            "uncertainty_logits",
            "redundancy_logits",
        ):
            if key not in scout_out or not torch.is_tensor(scout_out[key]):
                raise ValueError(f"scout_out['{key}'] tensor is required")
            if scout_out[key].shape != valid.shape:
                raise ValueError(f"scout_out['{key}'] shape must match valid_mask")
            _require_finite(scout_out[key], key)

        hazard = (
            torch.sigmoid(scout_out["actionness_logits"])
            + torch.sigmoid(scout_out["start_hazard_logits"])
            + torch.sigmoid(scout_out["end_hazard_logits"])
            + torch.sigmoid(scout_out["boundary_logits"])
        ) * 0.25
        difficulty = torch.sigmoid(scout_out["difficulty_logits"])
        uncertainty = torch.sigmoid(scout_out["uncertainty_logits"])
        redundancy = torch.sigmoid(scout_out["redundancy_logits"])
        risk = (
            self.hazard_weight * _valid_mean(hazard, valid)
            + self.difficulty_weight * _valid_mean(difficulty, valid)
            + self.uncertainty_weight * _valid_mean(uncertainty, valid)
            - self.redundancy_weight * _valid_mean(redundancy, valid)
        )
        normalized = torch.sigmoid(risk - 1.5 + self._target_budget_logit_bias)
        budget_float = self.min_budget + normalized * float(self.max_budget - self.min_budget)
        quantized = torch.round((budget_float - self.min_budget) / self.budget_step) * self.budget_step
        budget = (quantized + self.min_budget).round().long().clamp(self.min_budget, self.max_budget)
        valid_len = valid.long().sum(dim=1)
        budget = torch.minimum(budget.to(device=valid.device), valid_len)
        budget = torch.maximum(budget, torch.ones_like(budget))
        meta = {
            "protocol": "bh_sdc_dynamic_budget_v1",
            "risk_score": risk.detach(),
            "normalized_risk": normalized.detach(),
            "budget": budget.detach(),
            "min_budget": self.min_budget,
            "target_budget": self.target_budget,
            "max_budget": self.max_budget,
            "uses_gt": False,
            "uses_teacher": False,
            "uses_raw_prediction_cache": False,
        }
        return budget, meta


@SELECTORS.register_module()
class BoundaryHazardAcquisitionPolicy(nn.Module):
    """Select monotonic sparse observations from boundary hazard scout signals."""

    def __init__(
        self,
        dense_window_size: int,
        min_budget: int,
        max_budget: int,
        coverage_ratio: float = 0.20,
        boundary_ratio: float = 0.45,
        difficulty_ratio: float = 0.20,
        uncertainty_ratio: float = 0.15,
        max_dense_gap: int = 0,
    ) -> None:
        super().__init__()
        self.dense_window_size = int(dense_window_size)
        self.min_budget = int(min_budget)
        self.max_budget = int(max_budget)
        self.coverage_ratio = float(coverage_ratio)
        self.boundary_ratio = float(boundary_ratio)
        self.difficulty_ratio = float(difficulty_ratio)
        self.uncertainty_ratio = float(uncertainty_ratio)
        self.max_dense_gap = int(max_dense_gap)
        if self.dense_window_size <= 0 or self.min_budget <= 0 or self.max_budget <= 0:
            raise ValueError("dense_window_size and budgets must be positive")
        if self.min_budget > self.max_budget:
            raise ValueError("min_budget must not exceed max_budget")
        for name, value in (
            ("coverage_ratio", self.coverage_ratio),
            ("boundary_ratio", self.boundary_ratio),
            ("difficulty_ratio", self.difficulty_ratio),
            ("uncertainty_ratio", self.uncertainty_ratio),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.max_dense_gap < 0:
            raise ValueError("max_dense_gap must be non-negative")

    def forward(
        self,
        dense_axis: torch.Tensor,
        scout_out: Mapping[str, torch.Tensor],
        budget_per_sample: torch.Tensor,
        valid_mask: torch.Tensor,
        metas: Sequence[Mapping[str, Any]] | None = None,
    ) -> AcquisitionPlan:
        valid = valid_mask.bool()
        batch, time = valid.shape
        if time != self.dense_window_size:
            raise ValueError(f"expected dense_window_size={self.dense_window_size}, got {time}")
        max_out = min(self.max_budget, time)
        selected = torch.zeros(batch, max_out, dtype=torch.long, device=valid.device)
        selected_mask = torch.zeros(batch, max_out, dtype=torch.bool, device=valid.device)
        boundary_role = torch.zeros_like(selected_mask)
        coverage_role = torch.zeros_like(selected_mask)
        difficulty_role = torch.zeros_like(selected_mask)
        uncertainty_role = torch.zeros_like(selected_mask)
        combined_score_rows = []
        max_gap_rows = []

        boundary_score = torch.sigmoid(scout_out["start_hazard_logits"]) + torch.sigmoid(scout_out["end_hazard_logits"])
        boundary_score = boundary_score + torch.sigmoid(scout_out["boundary_logits"])
        difficulty_score = torch.sigmoid(scout_out["difficulty_logits"])
        uncertainty_score = torch.sigmoid(scout_out["uncertainty_logits"])
        action_score = torch.sigmoid(scout_out["actionness_logits"])
        redundancy_score = torch.sigmoid(scout_out["redundancy_logits"])
        combined = action_score + 0.75 * boundary_score + 0.35 * difficulty_score + 0.25 * uncertainty_score
        combined = combined - 0.45 * redundancy_score
        combined = combined.masked_fill(~valid, -1.0e6)

        for batch_idx in range(batch):
            valid_len = int(valid[batch_idx].long().sum().item())
            budget = int(budget_per_sample[batch_idx].item())
            budget = max(1, min(budget, valid_len, max_out))
            coverage_count = min(budget, max(1, int(round(budget * self.coverage_ratio))))
            if self.max_dense_gap > 0 and valid_len > 1:
                required_gap_anchors = int((valid_len - 1 + self.max_dense_gap - 1) // self.max_dense_gap) + 1
                coverage_count = min(budget, max(coverage_count, required_gap_anchors))
            boundary_count = min(budget, max(1, int(round(budget * self.boundary_ratio))))
            difficulty_count = min(budget, int(round(budget * self.difficulty_ratio)))
            uncertainty_count = min(budget, int(round(budget * self.uncertainty_ratio)))

            chosen: list[int] = []
            role_by_idx: dict[int, str] = {}

            def add_idx(idx: int, role: str) -> None:
                if 0 <= idx < valid_len and idx not in chosen:
                    chosen.append(idx)
                    role_by_idx[idx] = role

            for idx in _linspace_anchors(valid_len, coverage_count, device=valid.device):
                add_idx(idx, "coverage")

            for score_tensor, count, role in (
                (boundary_score[batch_idx], boundary_count, "boundary"),
                (difficulty_score[batch_idx], difficulty_count, "difficulty"),
                (uncertainty_score[batch_idx], uncertainty_count, "uncertainty"),
            ):
                if len(chosen) >= budget:
                    break
                order = torch.argsort(score_tensor[:valid_len], descending=True, stable=True).detach().cpu().tolist()
                for raw_idx in order:
                    if sum(1 for item in chosen if role_by_idx.get(item) == role) >= count:
                        break
                    add_idx(int(raw_idx), role)
                    if len(chosen) >= budget:
                        break

            if self.max_dense_gap > 0:
                chosen = sorted(chosen)
                changed = True
                while changed and len(chosen) < budget:
                    changed = False
                    for left, right in list(zip(chosen, chosen[1:])):
                        if right - left <= self.max_dense_gap or len(chosen) >= budget:
                            continue
                        candidates = [idx for idx in range(left + 1, right) if idx not in chosen]
                        if not candidates:
                            continue
                        midpoint = 0.5 * float(left + right)
                        best = min(
                            candidates,
                            key=lambda item: (
                                abs(float(item) - midpoint),
                                -float(combined[batch_idx, item].detach().cpu().item()),
                            ),
                        )
                        add_idx(best, "gap_guard")
                        chosen = sorted(chosen)
                        changed = True
                        break

            order = torch.argsort(combined[batch_idx, :valid_len], descending=True, stable=True).detach().cpu().tolist()
            for raw_idx in order:
                if len(chosen) >= budget:
                    break
                add_idx(int(raw_idx), "score")
            cursor = 0
            while len(chosen) < budget and cursor < valid_len:
                add_idx(cursor, "fallback")
                cursor += 1

            chosen = sorted(chosen[:budget])
            if len(chosen) < budget:
                raise RuntimeError("failed to build BH-SDC acquisition plan")
            out = torch.as_tensor(chosen, dtype=torch.long, device=valid.device)
            selected[batch_idx, :budget] = out
            if budget < max_out:
                selected[batch_idx, budget:] = out[-1]
            selected_mask[batch_idx, :budget] = True
            for col, idx in enumerate(chosen):
                role = role_by_idx.get(idx, "score")
                boundary_role[batch_idx, col] = role == "boundary"
                coverage_role[batch_idx, col] = role == "coverage"
                difficulty_role[batch_idx, col] = role == "difficulty"
                uncertainty_role[batch_idx, col] = role == "uncertainty"
            combined_score_rows.append(combined[batch_idx, out].detach())
            max_gap = 0
            if len(chosen) > 1:
                max_gap = max(right - left for left, right in zip(chosen, chosen[1:]))
            max_gap_rows.append(float(max_gap))

        physical_times = selected.to(dtype=torch.float32)
        diagnostics = {
            "combined_score_mean": combined.masked_fill(~valid, 0.0).sum(dim=1)
            / valid.to(dtype=combined.dtype).sum(dim=1).clamp_min(1.0),
            "max_dense_gap": torch.as_tensor(max_gap_rows, device=valid.device, dtype=torch.float32),
        }
        roles = {
            "boundary_hazard": boundary_role,
            "coverage_anchor": coverage_role,
            "difficulty": difficulty_role,
            "uncertainty": uncertainty_role,
        }
        return AcquisitionPlan(
            selected_dense_indices=selected,
            selected_mask=selected_mask,
            physical_times=physical_times,
            budget=budget_per_sample.to(device=valid.device),
            roles=roles,
            diagnostics=diagnostics,
        )


@SELECTORS.register_module()
class PCOTMRASBoundaryHazardSparseDenseFrameSelector(nn.Module):
    """Boundary-hazard sparse acquisition front-end for ActionFormer.

    The selector emits sparse raw observations and a deploy-visible acquisition
    plan. Dense detector-axis reconstruction is performed by
    PCOTMRASBoundaryHazardSparseToDenseBridge after the backbone.
    """

    def __init__(
        self,
        input_channels: int,
        dense_window_size: int,
        min_budget: int,
        target_budget: int,
        max_budget: int,
        budget_step: int = 1,
        scout_hidden_dim: int = 128,
        scout_num_layers: int = 3,
        scout_kernel_size: int = 5,
        scout_dropout: float = 0.0,
        coverage_ratio: float = 0.20,
        boundary_ratio: float = 0.45,
        difficulty_ratio: float = 0.20,
        uncertainty_ratio: float = 0.15,
        max_dense_gap: int = 0,
        aux_hazard_loss_weight: float = 0.05,
        aux_budget_entropy_loss_weight: float = 0.001,
        meta_key: str = "bh_sdc_acquisition_plan",
    ) -> None:
        super().__init__()
        self.dense_window_size = int(dense_window_size)
        self.min_budget = int(min_budget)
        self.target_budget = int(target_budget)
        self.max_budget = int(max_budget)
        self.meta_key = str(meta_key)
        self.aux_hazard_loss_weight = float(aux_hazard_loss_weight)
        self.aux_budget_entropy_loss_weight = float(aux_budget_entropy_loss_weight)
        self.scout = BoundaryHazardTemporalScout(
            in_channels=int(input_channels),
            hidden_dim=int(scout_hidden_dim),
            num_layers=int(scout_num_layers),
            kernel_size=int(scout_kernel_size),
            dropout=float(scout_dropout),
        )
        self.budget_controller = BoundaryHazardDynamicBudgetController(
            min_budget=int(min_budget),
            target_budget=int(target_budget),
            max_budget=int(max_budget),
            budget_step=int(budget_step),
        )
        self.policy = BoundaryHazardAcquisitionPolicy(
            dense_window_size=int(dense_window_size),
            min_budget=int(min_budget),
            max_budget=int(max_budget),
            coverage_ratio=float(coverage_ratio),
            boundary_ratio=float(boundary_ratio),
            difficulty_ratio=float(difficulty_ratio),
            uncertainty_ratio=float(uncertainty_ratio),
            max_dense_gap=int(max_dense_gap),
        )

    def forward_train(self, inputs, masks, metas, gt_segments, gt_labels):
        outputs = self._select(inputs=inputs, masks=masks, metas=metas)
        losses = self._losses(outputs["scout_out"], outputs["valid_mask"], gt_segments)
        return {
            "inputs": outputs["inputs"],
            "masks": outputs["masks"],
            "metas": outputs["metas"],
            "gt_segments": gt_segments,
            "gt_labels": gt_labels,
            "losses": losses,
        }

    def forward_test(self, inputs, masks, metas=None):
        _reject_forbidden_test_metas(metas)
        outputs = self._select(inputs=inputs, masks=masks, metas=metas)
        return {
            "inputs": outputs["inputs"],
            "masks": outputs["masks"],
            "metas": outputs["metas"],
        }

    def _select(self, *, inputs: torch.Tensor, masks: torch.Tensor, metas):
        batch = int(inputs.shape[0])
        dense_len = _temporal_len(inputs)
        if dense_len != self.dense_window_size:
            raise ValueError(f"expected dense_window_size={self.dense_window_size}, got {dense_len}")
        valid = _prefix_mask(masks, expected_shape=(batch, dense_len), name="masks").to(device=inputs.device)
        sequence = _feature_sequence_for_scout(inputs)
        scout_out = self.scout(sequence, valid, metas=metas)
        budget, budget_meta = self.budget_controller(scout_out, valid)
        dense_axis = torch.arange(dense_len, device=inputs.device, dtype=torch.long)[None, :].expand(batch, -1)
        plan = self.policy(
            dense_axis=dense_axis,
            scout_out=scout_out,
            budget_per_sample=budget,
            valid_mask=valid,
            metas=metas,
        )
        selected_inputs = _gather_temporal(inputs, plan.selected_dense_indices, plan.selected_mask)
        new_metas = self._write_plan_meta(metas, plan, valid, budget_meta)
        return {
            "inputs": selected_inputs,
            "masks": plan.selected_mask,
            "metas": new_metas,
            "scout_out": scout_out,
            "valid_mask": valid,
            "plan": plan,
        }

    def _write_plan_meta(
        self,
        metas,
        plan: AcquisitionPlan,
        valid: torch.Tensor,
        budget_meta: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        if metas is None:
            metas = [{} for _ in range(plan.selected_dense_indices.shape[0])]
        if len(metas) != plan.selected_dense_indices.shape[0]:
            raise ValueError("metas length must match batch size")
        out: list[dict[str, Any]] = []
        for batch_idx, meta in enumerate(metas):
            item = dict(meta)
            selected_count = int(plan.selected_mask[batch_idx].long().sum().item())
            selected = plan.selected_dense_indices[batch_idx, :selected_count].detach().cpu().tolist()
            dense_valid_len = int(valid[batch_idx].long().sum().item())
            role_payload = {
                key: value[batch_idx, :selected_count].detach().cpu().to(torch.long).tolist()
                for key, value in plan.roles.items()
            }
            item[self.meta_key] = {
                "protocol": "bh_sdc_acquisition_plan_v1",
                "route_label": BH_SDC_ROUTE_LABEL,
                "selected_dense_indices": selected,
                "selected_count": selected_count,
                "dense_valid_len": dense_valid_len,
                "dense_window_size": self.dense_window_size,
                "budget": int(plan.budget[batch_idx].item()),
                "roles": role_payload,
                "max_dense_gap": float(plan.diagnostics["max_dense_gap"][batch_idx].item()),
                "budget_protocol": "bh_sdc_dynamic_budget_v1",
                "uses_gt": False,
                "uses_teacher": False,
                "uses_oracle": False,
                "uses_cache": False,
                "uses_raw_prediction_cache": False,
            }
            item["bh_sdc_dynamic_budget"] = {
                "protocol": "bh_sdc_dynamic_budget_v1",
                "route_label": BH_SDC_ROUTE_LABEL,
                "budget": int(plan.budget[batch_idx].item()),
                "min_budget": int(budget_meta["min_budget"]),
                "target_budget": self.target_budget,
                "max_budget": int(budget_meta["max_budget"]),
                "uses_gt": False,
                "uses_teacher": False,
                "uses_raw_prediction_cache": False,
            }
            item["bh_sdc_protocol_flags"] = _deploy_protocol_flags()
            item["irregular_selected_positions"] = selected
            item["irregular_selected_count"] = selected_count
            item["irregular_selected_valid_len"] = dense_valid_len
            item["irregular_dense_valid_len"] = dense_valid_len
            item["irregular_native_axis"] = False
            out.append(item)
        return out

    def _losses(self, scout_out: Mapping[str, torch.Tensor], valid: torch.Tensor, gt_segments) -> dict[str, torch.Tensor]:
        loss = scout_out["actionness_logits"].new_zeros(())
        if self.aux_hazard_loss_weight > 0.0:
            targets = self._build_hazard_targets(valid, gt_segments)
            action_loss = self._masked_bce(scout_out["actionness_logits"], targets["actionness"], valid)
            start_loss = self._masked_bce(scout_out["start_hazard_logits"], targets["start"], valid)
            end_loss = self._masked_bce(scout_out["end_hazard_logits"], targets["end"], valid)
            boundary_loss = self._masked_bce(scout_out["boundary_logits"], targets["boundary"], valid)
            redundancy_loss = self._masked_bce(scout_out["redundancy_logits"], 1.0 - targets["actionness"], valid)
            loss = (action_loss + start_loss + end_loss + boundary_loss + 0.25 * redundancy_loss) * float(
                self.aux_hazard_loss_weight
            )
        entropy_loss = scout_out["actionness_logits"].new_zeros(())
        if self.aux_budget_entropy_loss_weight > 0.0:
            prob = torch.sigmoid(scout_out["frame_selection_logits"]).clamp(1.0e-4, 1.0 - 1.0e-4)
            entropy = -(prob * prob.log() + (1.0 - prob) * (1.0 - prob).log())
            entropy_loss = _valid_mean(entropy, valid).mean() * float(self.aux_budget_entropy_loss_weight)
        return {
            "selector_bh_sdc_hazard_loss": loss,
            "selector_bh_sdc_budget_entropy_loss": entropy_loss,
        }

    @staticmethod
    def _masked_bce(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        selected_logits = logits.masked_select(valid)
        selected_target = target.to(device=logits.device, dtype=logits.dtype).masked_select(valid)
        if selected_logits.numel() == 0:
            return logits.new_zeros(())
        return F.binary_cross_entropy_with_logits(selected_logits, selected_target)

    @staticmethod
    def _build_hazard_targets(valid: torch.Tensor, gt_segments) -> dict[str, torch.Tensor]:
        batch, time = valid.shape
        device = valid.device
        axis = torch.arange(time, device=device, dtype=torch.float32)
        action = torch.zeros(batch, time, device=device)
        start = torch.zeros_like(action)
        end = torch.zeros_like(action)
        sigma = max(1.0, float(time) / 96.0)
        for batch_idx in range(batch):
            segments = gt_segments[batch_idx] if gt_segments is not None else None
            if segments is None or not torch.is_tensor(segments) or segments.numel() == 0:
                continue
            segments = segments.to(device=device, dtype=torch.float32).reshape(-1, 2)
            dense_valid_len = int(valid[batch_idx].long().sum().item())
            for seg in segments:
                left = float(seg[0].item())
                right = float(seg[1].item())
                if right <= left:
                    continue
                left = max(0.0, min(left, dense_valid_len - 1.0))
                right = max(0.0, min(right, dense_valid_len - 1.0))
                action[batch_idx] = torch.maximum(action[batch_idx], ((axis >= left) & (axis <= right)).float())
                start[batch_idx] = torch.maximum(start[batch_idx], torch.exp(-0.5 * ((axis - left) / sigma) ** 2))
                end[batch_idx] = torch.maximum(end[batch_idx], torch.exp(-0.5 * ((axis - right) / sigma) ** 2))
        boundary = torch.maximum(start, end)
        return {
            "actionness": action * valid.to(dtype=action.dtype),
            "start": start * valid.to(dtype=start.dtype),
            "end": end * valid.to(dtype=end.dtype),
            "boundary": boundary * valid.to(dtype=boundary.dtype),
        }


@SELECTORS.register_module()
class PCOTMRASBoundaryHazardSparseToDenseBridge(nn.Module):
    """Complete sparse backbone features back to a dense detector time axis."""

    def __init__(
        self,
        dense_window_size: int,
        target_len: int | None = None,
        interpolation_temperature: float = 4.0,
        refine_channels: int | None = None,
        refine_layers: int = 0,
        smoothness_loss_weight: float = 0.001,
        meta_key: str = "bh_sdc_acquisition_plan",
    ) -> None:
        super().__init__()
        self.dense_window_size = int(dense_window_size)
        self.target_len = int(target_len) if target_len is not None else int(dense_window_size)
        self.interpolation_temperature = float(interpolation_temperature)
        self.smoothness_loss_weight = float(smoothness_loss_weight)
        self.meta_key = str(meta_key)
        if self.dense_window_size <= 0 or self.target_len <= 0:
            raise ValueError("dense_window_size and target_len must be positive")
        if self.target_len != self.dense_window_size:
            raise ValueError("BH-SDC completion bridge target_len must equal dense_window_size for dense inference")
        if self.interpolation_temperature <= 0.0:
            raise ValueError("interpolation_temperature must be positive")
        self.refine: nn.Module | None = None
        self.refine_scale: nn.Parameter | None = None
        if int(refine_layers) > 0:
            if refine_channels is None or int(refine_channels) <= 0:
                raise ValueError("refine_channels must be positive when refine_layers > 0")
            layers: list[nn.Module] = []
            for _idx in range(int(refine_layers)):
                layers.append(nn.Conv1d(int(refine_channels), int(refine_channels), kernel_size=3, padding=1))
                layers.append(nn.GELU())
            layers.append(nn.Conv1d(int(refine_channels), int(refine_channels), kernel_size=1))
            self.refine = nn.Sequential(*layers)
            self.refine_scale = nn.Parameter(torch.zeros(()))

    def forward_train(self, features, masks, metas, gt_segments, gt_labels):
        features, masks, metas, observed_mask = self._complete(features, masks, metas)
        losses = self._losses(features, masks, observed_mask)
        return {
            "features": features,
            "masks": masks,
            "metas": metas,
            "gt_segments": gt_segments,
            "gt_labels": gt_labels,
            "losses": losses,
        }

    def forward_test(self, features, masks, metas=None):
        _reject_forbidden_test_metas(metas)
        features, masks, metas, _observed_mask = self._complete(features, masks, metas)
        return {
            "features": features,
            "masks": masks,
            "metas": metas,
        }

    def _complete(self, features: torch.Tensor, masks: torch.Tensor, metas):
        if features.ndim != 3:
            raise ValueError(f"features must be [B,C,K], got {tuple(features.shape)}")
        batch, channels, sparse_len = features.shape
        sparse_mask = _prefix_mask(masks, expected_shape=(batch, sparse_len), name="sparse masks").to(
            device=features.device
        )
        if metas is None or len(metas) != batch:
            raise ValueError("BH-SDC completion bridge requires one meta dict per sample")
        dense = features.new_zeros(batch, channels, self.dense_window_size)
        dense_mask = torch.zeros(batch, self.dense_window_size, dtype=torch.bool, device=features.device)
        observed_mask = torch.zeros_like(dense_mask)
        dense_axis = torch.arange(self.dense_window_size, device=features.device, dtype=features.dtype)
        new_metas: list[dict[str, Any]] = []

        for batch_idx, meta in enumerate(metas):
            if not isinstance(meta, Mapping) or self.meta_key not in meta:
                raise ValueError(f"meta[{batch_idx}] is missing {self.meta_key}")
            plan = meta[self.meta_key]
            selected_count = int(plan["selected_count"])
            dense_valid_len = int(plan["dense_valid_len"])
            if selected_count <= 0 or selected_count > sparse_len:
                raise ValueError(f"meta[{batch_idx}] selected_count is out of range")
            if dense_valid_len <= 0 or dense_valid_len > self.dense_window_size:
                raise ValueError(f"meta[{batch_idx}] dense_valid_len is out of range")
            if int(sparse_mask[batch_idx].long().sum().item()) < selected_count:
                raise ValueError(f"meta[{batch_idx}] selected_count exceeds sparse feature valid count")
            selected = torch.as_tensor(
                plan["selected_dense_indices"][:selected_count],
                device=features.device,
                dtype=torch.long,
            )
            if selected.numel() != selected_count:
                raise ValueError(f"meta[{batch_idx}] selected_dense_indices length mismatch")
            if bool((selected[1:] <= selected[:-1]).any().item()):
                raise ValueError(f"meta[{batch_idx}] selected_dense_indices must be strictly increasing")
            if int(selected.min().item()) < 0 or int(selected.max().item()) >= dense_valid_len:
                raise ValueError(f"meta[{batch_idx}] selected_dense_indices must lie inside dense valid length")

            sparse = features[batch_idx, :, :selected_count]
            distance = (dense_axis[:, None] - selected.to(dtype=features.dtype)[None, :]).abs()
            weights = torch.softmax(-distance / self.interpolation_temperature, dim=1)
            completed = sparse @ weights.transpose(0, 1)
            completed[:, selected] = sparse
            completed[:, dense_valid_len:] = 0.0
            dense[batch_idx] = completed
            dense_mask[batch_idx, :dense_valid_len] = True
            observed_mask[batch_idx, selected] = True

            item = dict(meta)
            item["bh_sdc_completion"] = {
                "protocol": "bh_sdc_sparse_to_dense_completion_v1",
                "route_label": BH_SDC_ROUTE_LABEL,
                "dense_window_size": self.dense_window_size,
                "dense_valid_len": dense_valid_len,
                "observed_dense_indices": selected.detach().cpu().tolist(),
                "observed_count": selected_count,
                "interpolation_temperature": self.interpolation_temperature,
                "uses_gt": False,
                "uses_teacher": False,
                "uses_raw_prediction_cache": False,
            }
            dense_positions = list(range(dense_valid_len))
            item["irregular_selected_positions"] = dense_positions
            item["irregular_selected_count"] = dense_valid_len
            item["irregular_selected_valid_len"] = dense_valid_len
            item["irregular_dense_valid_len"] = dense_valid_len
            item["irregular_native_axis"] = True
            new_metas.append(item)

        if self.refine is not None and self.refine_scale is not None:
            refined = dense + self.refine_scale * self.refine(dense)
            for batch_idx, meta in enumerate(new_metas):
                selected = torch.as_tensor(
                    meta[self.meta_key]["selected_dense_indices"][: meta[self.meta_key]["selected_count"]],
                    device=features.device,
                    dtype=torch.long,
                )
                refined[batch_idx, :, selected] = dense[batch_idx, :, selected]
            dense = refined.masked_fill(~dense_mask[:, None, :], 0.0)

        return dense, dense_mask, new_metas, observed_mask

    def _losses(self, dense: torch.Tensor, dense_mask: torch.Tensor, observed_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        loss = dense.new_zeros(())
        if self.smoothness_loss_weight > 0.0 and dense.shape[-1] > 1:
            diff = dense[:, :, 1:] - dense[:, :, :-1]
            valid = dense_mask[:, 1:] & dense_mask[:, :-1]
            synthetic = ~(observed_mask[:, 1:] & observed_mask[:, :-1])
            weight = (valid & synthetic).to(dtype=dense.dtype)
            denom = weight.sum().clamp_min(1.0)
            loss = (diff.square() * weight[:, None, :]).sum() / denom
            loss = loss * float(self.smoothness_loss_weight)
        return {"token_bh_sdc_completion_smoothness_loss": loss}
