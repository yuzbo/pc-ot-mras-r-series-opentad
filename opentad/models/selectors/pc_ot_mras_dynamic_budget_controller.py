from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..builder import SELECTORS


_FORBIDDEN_PAYLOAD_TOKENS = frozenset(
    {
        "gt",
        "oracle",
        "teacher",
        "cache",
        "prediction",
        "predictions",
        "pred",
        "checkpoint",
        "ckpt",
        "result",
        "target",
    }
)
_FORBIDDEN_PAYLOAD_PHRASES = frozenset(
    {
        "ground_truth",
        "groundtruth",
        "raw_prediction",
        "raw_predictions",
        "rawprediction",
        "rawpredictions",
        "value_transport",
        "pc_ot_mras_value_targets",
    }
)


@dataclass(frozen=True)
class PCOTMRASDynamicBudgetControllerConfig:
    budget_values: Tuple[int, ...] = (288, 320, 352, 384, 416)
    budget_thresholds: Tuple[float, ...] = (0.20, 0.35, 0.50, 0.65)
    value_weight: float = 1.0
    risk_weight: float = 0.50
    redundancy_weight: float = 0.25
    transport_weight: float = 0.25
    coverage_share: float = 0.35
    max_coverage_share: float = 0.65
    require_value_logits: bool = True


def _normalized_text(value: object) -> str:
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(value or ""))
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", text)
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text)


def _contains_forbidden_fragment(value: object) -> bool:
    normalized = _normalized_text(value)
    tokens = [token for token in re.split(r"_+", normalized.strip("_")) if token]
    if any(token in _FORBIDDEN_PAYLOAD_TOKENS for token in tokens):
        return True
    if "ground" in tokens and "truth" in tokens:
        return True
    compact = "".join(tokens)
    if any(phrase in normalized or phrase in compact for phrase in _FORBIDDEN_PAYLOAD_PHRASES):
        return True
    if compact.startswith("gt"):
        return True
    return any(token != "gt" and token in compact for token in _FORBIDDEN_PAYLOAD_TOKENS)


def _validate_deploy_visible_payload(value: object, *, location: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key or "")
            if _contains_forbidden_fragment(key_text):
                raise ValueError(f"{location}.{key_text} contains forbidden deploy-time payload")
            _validate_deploy_visible_payload(item, location=f"{location}.{key_text}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_deploy_visible_payload(item, location=f"{location}[{index}]")
        return
    if isinstance(value, str) and _contains_forbidden_fragment(value):
        raise ValueError(f"{location} contains forbidden deploy-time payload")


def _require_tensor(mapping: Mapping[str, object], key: str) -> torch.Tensor:
    value = mapping.get(key)
    if not torch.is_tensor(value):
        raise ValueError(f"reader_outputs['{key}'] must be a tensor")
    if torch.is_complex(value):
        raise ValueError(f"reader_outputs['{key}'] must be real-valued")
    if torch.is_floating_point(value) and not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"reader_outputs['{key}'] must be finite")
    return value


def _optional_tensor(
    mapping: Mapping[str, object],
    key: str,
    *,
    expected_shape: torch.Size,
    expected_device: torch.device,
) -> Optional[torch.Tensor]:
    value = mapping.get(key)
    if value is None:
        return None
    if not torch.is_tensor(value):
        raise ValueError(f"reader_outputs['{key}'] must be a tensor when provided")
    if value.shape != expected_shape:
        raise ValueError(f"reader_outputs['{key}'] shape mismatch: {tuple(value.shape)} vs {tuple(expected_shape)}")
    if value.device != expected_device:
        raise ValueError(f"reader_outputs['{key}'] must share device with valid_mask")
    if torch.is_complex(value):
        raise ValueError(f"reader_outputs['{key}'] must be real-valued")
    if torch.is_floating_point(value) and not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"reader_outputs['{key}'] must be finite")
    return value


def _prefix_binary_mask(mask: torch.Tensor, *, name: str) -> torch.Tensor:
    if mask.ndim != 2:
        raise ValueError(f"{name} must be [B,T], got {tuple(mask.shape)}")
    if torch.is_complex(mask):
        raise ValueError(f"{name} must contain binary 0/1 values")
    if torch.is_floating_point(mask) and not bool(torch.isfinite(mask).all().item()):
        raise ValueError(f"{name} must be finite")
    if mask.dtype != torch.bool and not bool(((mask == 0) | (mask == 1)).all().item()):
        raise ValueError(f"{name} must contain binary 0/1 values")
    valid = mask.bool()
    valid_count = valid.long().sum(dim=1)
    if bool((valid_count <= 0).any().item()):
        raise ValueError(f"{name} must contain at least one valid position per sample")
    prefix = torch.arange(valid.shape[1], device=valid.device)[None, :] < valid_count[:, None]
    if not torch.equal(valid, prefix):
        raise ValueError(f"{name} must be a contiguous valid prefix")
    return valid


def _normalize_budget_values(values: Sequence[int]) -> tuple[int, ...]:
    try:
        budgets = tuple(int(value) for value in values)
    except TypeError as exc:
        raise ValueError("budget_values must be a sequence of positive integers") from exc
    if not budgets:
        raise ValueError("budget_values must be non-empty")
    if any(value <= 0 for value in budgets):
        raise ValueError("budget_values must be positive")
    if any(current <= prev for prev, current in zip(budgets, budgets[1:])):
        raise ValueError("budget_values must be strictly increasing")
    return budgets


def _normalize_thresholds(thresholds: Sequence[float], *, expected_len: int) -> tuple[float, ...]:
    try:
        values = tuple(float(value) for value in thresholds)
    except TypeError as exc:
        raise ValueError("budget_thresholds must be a sequence of floats") from exc
    if len(values) != int(expected_len):
        raise ValueError("budget_thresholds length must be len(budget_values) - 1")
    if any(value <= 0.0 or value >= 1.0 for value in values):
        raise ValueError("budget_thresholds must lie inside (0, 1)")
    if any(current <= prev for prev, current in zip(values, values[1:])):
        raise ValueError("budget_thresholds must be strictly increasing")
    return values


def _normalized_valid_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    masked = values.masked_fill(~valid, 0.0)
    denom = valid.to(dtype=values.dtype).sum(dim=1).clamp_min(1.0)
    return masked.sum(dim=1) / denom


def _transport_signal(reader_outputs: Mapping[str, object], valid: torch.Tensor) -> torch.Tensor:
    acquisition = reader_outputs.get("acquisition_matrix")
    if acquisition is None:
        return torch.zeros_like(valid, dtype=torch.float32)
    if not torch.is_tensor(acquisition):
        raise ValueError("reader_outputs['acquisition_matrix'] must be a tensor when provided")
    if acquisition.ndim != 3 or acquisition.shape[0] != valid.shape[0] or acquisition.shape[2] != valid.shape[1]:
        raise ValueError("acquisition_matrix must be [B,K,T] and match valid_mask")
    if acquisition.device != valid.device:
        raise ValueError("acquisition_matrix must share device with valid_mask")
    if torch.is_complex(acquisition) or not bool(torch.isfinite(acquisition).all().item()):
        raise ValueError("acquisition_matrix must be finite and real-valued")
    if bool((acquisition < 0).any().item()):
        raise ValueError("acquisition_matrix must be non-negative")
    invalid_mass = acquisition.masked_select(~valid[:, None, :].expand_as(acquisition))
    if invalid_mass.numel() and bool((invalid_mass.abs() > 1.0e-7).any().item()):
        raise ValueError("acquisition_matrix must not allocate mass to invalid positions")
    col_mass = acquisition.sum(dim=1).to(dtype=torch.float32).masked_fill(~valid, 0.0)
    max_mass = col_mass.amax(dim=1, keepdim=True).clamp_min(1.0e-6)
    return col_mass / max_mass


def _coverage_positions(valid_len: int, count: int, *, device: torch.device) -> list[int]:
    if count <= 0:
        return []
    if count >= valid_len:
        return list(range(valid_len))
    anchors = torch.linspace(0, valid_len - 1, steps=count, device=device).round().to(dtype=torch.long)
    out = []
    for item in anchors.detach().cpu().tolist():
        idx = int(item)
        if idx not in out:
            out.append(idx)
    return out


def _fill_positions(
    *,
    score: torch.Tensor,
    valid_len: int,
    target_budget: int,
    coverage_count: int,
) -> list[int]:
    device = score.device
    selected = _coverage_positions(valid_len, coverage_count, device=device)
    selected_set = set(selected)
    order = torch.argsort(score[:valid_len], descending=True, stable=True).detach().cpu().tolist()
    for raw_idx in order:
        idx = int(raw_idx)
        if idx not in selected_set:
            selected.append(idx)
            selected_set.add(idx)
        if len(selected) >= target_budget:
            break
    idx = 0
    while len(selected) < target_budget and idx < valid_len:
        if idx not in selected_set:
            selected.append(idx)
            selected_set.add(idx)
        idx += 1
    return sorted(selected[:target_budget])


@SELECTORS.register_module()
class PCOTMRASDynamicBudgetController(nn.Module):
    """Convert deploy-visible PC-OT-MRAS value signals into variable-budget plans.

    The controller consumes reader logits and transport tensors only. It rejects
    GT, teacher, cache, prediction, result, checkpoint, and train-only target
    payloads, so its output is a local protocol artifact rather than mAP or
    deployment evidence.
    """

    def __init__(
        self,
        budget_values: Sequence[int] = (288, 320, 352, 384, 416),
        budget_thresholds: Sequence[float] = (0.20, 0.35, 0.50, 0.65),
        value_weight: float = 1.0,
        risk_weight: float = 0.50,
        redundancy_weight: float = 0.25,
        transport_weight: float = 0.25,
        coverage_share: float = 0.35,
        max_coverage_share: float = 0.65,
        require_value_logits: bool = True,
    ) -> None:
        super().__init__()
        budgets = _normalize_budget_values(budget_values)
        thresholds = _normalize_thresholds(budget_thresholds, expected_len=len(budgets) - 1)
        if not 0.0 <= float(coverage_share) <= float(max_coverage_share) <= 1.0:
            raise ValueError("coverage_share and max_coverage_share must satisfy 0 <= coverage <= max <= 1")
        self.cfg = PCOTMRASDynamicBudgetControllerConfig(
            budget_values=budgets,
            budget_thresholds=thresholds,
            value_weight=float(value_weight),
            risk_weight=float(risk_weight),
            redundancy_weight=float(redundancy_weight),
            transport_weight=float(transport_weight),
            coverage_share=float(coverage_share),
            max_coverage_share=float(max_coverage_share),
            require_value_logits=bool(require_value_logits),
        )

    def forward(self, reader_outputs: Mapping[str, object]) -> Dict[str, torch.Tensor | object]:
        if not isinstance(reader_outputs, Mapping):
            raise ValueError("reader_outputs must be a mapping")
        _validate_deploy_visible_payload(reader_outputs, location="reader_outputs")

        valid = _prefix_binary_mask(_require_tensor(reader_outputs, "valid_mask"), name="valid_mask")
        batch, dense_len = valid.shape
        device = valid.device
        if self.cfg.require_value_logits:
            value_logits = _require_tensor(reader_outputs, "value_logits")
        else:
            value_logits = reader_outputs.get("value_logits")
            if value_logits is None:
                value_logits = torch.zeros((batch, dense_len), dtype=torch.float32, device=device)
            elif not torch.is_tensor(value_logits):
                raise ValueError("reader_outputs['value_logits'] must be a tensor when provided")
        if value_logits.shape != valid.shape or value_logits.device != device:
            raise ValueError("value_logits must be [B,T] and share device with valid_mask")
        if torch.is_complex(value_logits) or not bool(torch.isfinite(value_logits).all().item()):
            raise ValueError("value_logits must be finite and real-valued")

        risk_logits = _optional_tensor(
            reader_outputs,
            "risk_logits",
            expected_shape=valid.shape,
            expected_device=device,
        )
        redundancy_logits = _optional_tensor(
            reader_outputs,
            "redundancy_logits",
            expected_shape=valid.shape,
            expected_device=device,
        )

        value = torch.sigmoid(value_logits.to(dtype=torch.float32)).masked_fill(~valid, 0.0)
        risk = torch.sigmoid(risk_logits.to(dtype=torch.float32)).masked_fill(~valid, 0.0) if risk_logits is not None else 0.0
        redundancy = (
            torch.sigmoid(redundancy_logits.to(dtype=torch.float32)).masked_fill(~valid, 0.0)
            if redundancy_logits is not None
            else 0.0
        )
        transport = _transport_signal(reader_outputs, valid)

        utility = (
            self.cfg.value_weight * value
            - self.cfg.risk_weight * risk
            - self.cfg.redundancy_weight * redundancy
            + self.cfg.transport_weight * transport
        ).clamp_min(0.0).masked_fill(~valid, 0.0)

        score = _normalized_valid_mean(utility, valid).clamp(0.0, 1.0)
        thresholds = torch.tensor(self.cfg.budget_thresholds, dtype=score.dtype, device=device)
        budget_index = torch.bucketize(score.contiguous(), thresholds)
        budget_values = torch.tensor(self.cfg.budget_values, dtype=torch.long, device=device)
        dense_valid_len = valid.long().sum(dim=1)
        budgets = budget_values[budget_index].minimum(dense_valid_len)

        max_budget = int(budgets.max().item())
        selected_positions = torch.zeros((batch, max_budget), dtype=torch.long, device=device)
        selected_mask = torch.zeros((batch, max_budget), dtype=torch.bool, device=device)
        coverage_counts = torch.zeros((batch,), dtype=torch.long, device=device)
        value_counts = torch.zeros((batch,), dtype=torch.long, device=device)

        for batch_idx in range(batch):
            budget = int(budgets[batch_idx].item())
            valid_len = int(dense_valid_len[batch_idx].item())
            coverage_count = min(
                budget,
                int(round(float(budget) * min(self.cfg.coverage_share, self.cfg.max_coverage_share))),
            )
            selected = _fill_positions(
                score=utility[batch_idx],
                valid_len=valid_len,
                target_budget=budget,
                coverage_count=coverage_count,
            )
            if len(selected) != budget or len(set(selected)) != budget:
                raise RuntimeError("dynamic budget controller failed to produce exact sorted unique positions")
            selected_tensor = torch.tensor(selected, dtype=torch.long, device=device)
            selected_positions[batch_idx, :budget] = selected_tensor
            selected_mask[batch_idx, :budget] = True
            coverage_counts[batch_idx] = int(coverage_count)
            value_counts[batch_idx] = int(budget - coverage_count)

        coverage_share = coverage_counts.to(dtype=torch.float32) / budgets.to(dtype=torch.float32).clamp_min(1.0)
        if bool((coverage_share > float(self.cfg.max_coverage_share) + 1.0e-6).any().item()):
            raise RuntimeError("coverage share exceeds configured cap")

        return {
            "schema_version": "pc_ot_mras_dynamic_budget_plan_v0",
            "controller_family": "r22_value_to_budget_control",
            "uses_gt": False,
            "uses_teacher": False,
            "uses_cache": False,
            "uses_raw_prediction": False,
            "uses_checkpoint": False,
            "dynamic_budget_validation": False,
            "metric_claim_allowed": False,
            "paper_claim_allowed": False,
            "budget_values": budget_values,
            "budget_scores": score,
            "budgets": budgets,
            "dense_valid_len": dense_valid_len,
            "utility_scores": utility,
            "selected_dense_positions": selected_positions,
            "selected_mask": selected_mask,
            "coverage_counts": coverage_counts,
            "value_counts": value_counts,
            "coverage_share": coverage_share,
            "max_coverage_share": torch.tensor(float(self.cfg.max_coverage_share), dtype=torch.float32, device=device),
        }


ValueToBudgetPCOTMRASController = PCOTMRASDynamicBudgetController

__all__ = ["PCOTMRASDynamicBudgetController", "ValueToBudgetPCOTMRASController"]
