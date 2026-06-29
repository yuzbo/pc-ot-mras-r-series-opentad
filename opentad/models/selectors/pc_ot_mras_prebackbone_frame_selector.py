from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import SELECTORS, build_selector


def _require_finite(tensor: torch.Tensor, name: str, *, error_type: type[Exception] = FloatingPointError) -> torch.Tensor:
    if not torch.is_tensor(tensor):
        raise TypeError(f"{name} must be a tensor")
    if not torch.isfinite(tensor).all():
        raise error_type(f"{name} must be finite")
    return tensor


def _as_bool_prefix_mask(masks: torch.Tensor, *, expected_shape: tuple[int, int]) -> torch.Tensor:
    if masks.ndim != 2:
        raise ValueError(f"masks must be [B,T], got {tuple(masks.shape)}")
    if tuple(masks.shape) != tuple(expected_shape):
        raise ValueError(f"masks shape mismatch: expected {expected_shape}, got {tuple(masks.shape)}")
    if masks.dtype != torch.bool:
        if not bool(torch.logical_or(masks == 0, masks == 1).all().item()):
            raise ValueError("masks must be boolean or binary")
    valid = masks.bool()
    valid_count = valid.long().sum(dim=1)
    if bool((valid_count <= 0).any().item()):
        raise ValueError("each sample must contain at least one valid dense frame")
    prefix = torch.arange(valid.shape[1], device=valid.device)[None, :] < valid_count[:, None]
    if not torch.equal(valid, prefix):
        raise ValueError("prebackbone selector requires prefix-contiguous masks")
    return valid


def _validate_frame_scout_inputs(
    features: torch.Tensor,
    valid: torch.Tensor,
    time_coords: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if features.ndim != 3:
        raise ValueError(f"features must be [B,T,C], got {tuple(features.shape)}")
    _require_finite(features, "features", error_type=ValueError)
    if valid.ndim != 2:
        raise ValueError(f"valid mask must be [B,T], got {tuple(valid.shape)}")
    if tuple(valid.shape) != tuple(features.shape[:2]):
        raise ValueError("valid mask must match feature batch/time axes")
    if valid.dtype != torch.bool:
        if not bool(torch.logical_or(valid == 0, valid == 1).all().item()):
            raise ValueError("valid mask must be boolean or binary")
    valid = valid.to(device=features.device).bool()
    if bool((valid.long().sum(dim=1) <= 0).any().item()):
        raise ValueError("each sample must contain at least one valid frame")
    if time_coords is None:
        time_coords = torch.zeros(features.shape[:2], dtype=features.dtype, device=features.device)
    else:
        if time_coords.ndim != 2 or tuple(time_coords.shape) != tuple(features.shape[:2]):
            raise ValueError("time_coords must match feature batch/time axes")
        time_coords = time_coords.to(device=features.device, dtype=features.dtype)
        _require_finite(time_coords, "time_coords", error_type=ValueError)
    return valid, time_coords


def _masked_slot_transport(slot_logits: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if slot_logits.ndim != 3:
        raise ValueError(f"slot_logits must be [B,K,T], got {tuple(slot_logits.shape)}")
    if valid.ndim != 2 or tuple(valid.shape) != (int(slot_logits.shape[0]), int(slot_logits.shape[2])):
        raise ValueError("valid mask must match slot_logits batch/time axes")
    _require_finite(slot_logits, "slot_logits", error_type=ValueError)
    valid = valid.to(device=slot_logits.device).bool()
    if bool((valid.long().sum(dim=1) <= 0).any().item()):
        raise ValueError("each sample must contain at least one valid frame")
    logits_fp32 = slot_logits.float()
    masked_logits = logits_fp32.masked_fill(~valid[:, None, :], torch.finfo(torch.float32).min)
    acquisition_matrix = F.softmax(masked_logits, dim=-1).masked_fill(~valid[:, None, :], 0.0)
    row_mass = acquisition_matrix.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    acquisition_matrix = acquisition_matrix / row_mass
    _require_finite(acquisition_matrix, "acquisition_matrix", error_type=ValueError)
    return masked_logits, acquisition_matrix


def _masked_frame_logits(logits: torch.Tensor, valid: torch.Tensor, name: str) -> torch.Tensor:
    if logits.ndim != 2 or tuple(logits.shape) != tuple(valid.shape):
        raise ValueError(f"{name} must be [B,T] and match valid mask")
    _require_finite(logits, name, error_type=ValueError)
    return logits.float().masked_fill(~valid.to(device=logits.device).bool(), 0.0)


def _smooth_clamp_logits(logits: torch.Tensor, limit: float, name: str) -> torch.Tensor:
    logits = logits.float()
    _require_finite(logits, name, error_type=ValueError)
    limit = float(limit)
    if limit <= 0.0:
        return logits
    bounded = torch.tanh(logits / limit) * limit
    _require_finite(bounded, f"{name} smooth-clamped", error_type=ValueError)
    return bounded


def _inverse_softplus(value: float) -> float:
    value = float(value)
    if value <= 0.0:
        raise ValueError("inverse softplus input must be positive")
    return math.log(math.expm1(value))


def _resolve_temporal_dilations(
    *,
    num_layers: int,
    dilation_base: int = 1,
    dilations: Sequence[int] | None = None,
) -> list[int]:
    if dilations is not None:
        resolved = [int(item) for item in dilations]
        if len(resolved) != int(num_layers):
            raise ValueError("dilations length must match num_layers")
    else:
        base = int(dilation_base)
        if base <= 0:
            raise ValueError("dilation_base must be positive")
        resolved = [base ** layer_idx for layer_idx in range(int(num_layers))]
    if any(item <= 0 for item in resolved):
        raise ValueError("all temporal dilations must be positive")
    return resolved


class _MaskedTemporalConvStack(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        num_layers: int,
        kernel_size: int,
        dropout: float,
        dilation_base: int = 1,
        dilations: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if int(num_layers) <= 0:
            raise ValueError("num_layers must be positive")
        if int(kernel_size) <= 0 or int(kernel_size) % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        resolved_dilations = _resolve_temporal_dilations(
            num_layers=int(num_layers),
            dilation_base=int(dilation_base),
            dilations=dilations,
        )
        self.convs = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for dilation in resolved_dilations:
            padding = (int(kernel_size) // 2) * int(dilation)
            self.convs.append(
                nn.Conv1d(
                    int(hidden_dim),
                    int(hidden_dim),
                    kernel_size=int(kernel_size),
                    padding=padding,
                    dilation=int(dilation),
                )
            )
            self.dropouts.append(nn.Dropout(float(dropout)))

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        mask = valid[:, None, :].to(device=x.device).bool()
        x = x.masked_fill(~mask, 0.0)
        for conv, dropout in zip(self.convs, self.dropouts):
            x = dropout(F.gelu(conv(x)))
            x = x.masked_fill(~mask, 0.0)
        return x


@SELECTORS.register_module()
class PCOTMRASTinyTransformerFrameScout(nn.Module):
    """Lightweight frame-level scout that emits PC-OT-MRAS-style slot scores."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 96,
        num_slots: int = 384,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if int(in_dim) <= 0 or int(hidden_dim) <= 0 or int(num_slots) <= 0:
            raise ValueError("in_dim, hidden_dim, and num_slots must be positive")
        self.num_slots = int(num_slots)
        self.input_proj = nn.Linear(int(in_dim) + 1, int(hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=int(hidden_dim),
            nhead=int(num_heads),
            dim_feedforward=int(hidden_dim) * 2,
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_layers))
        self.slot_queries = nn.Parameter(torch.randn(self.num_slots, int(hidden_dim)) * 0.02)

    def forward(self, features: torch.Tensor, valid: torch.Tensor, time_coords: torch.Tensor | None = None):
        valid, time_coords = _validate_frame_scout_inputs(features, valid, time_coords)
        x = torch.cat([features, time_coords.to(dtype=features.dtype).unsqueeze(-1)], dim=-1)
        x = x.masked_fill(~valid.unsqueeze(-1), 0.0)
        encoded = self.encoder(self.input_proj(x), src_key_padding_mask=~valid)
        slot_logits = torch.einsum("bth,kh->bkt", encoded, self.slot_queries) * (encoded.shape[-1] ** -0.5)
        slot_logits, acquisition_matrix = _masked_slot_transport(slot_logits, valid)
        return {"slot_logits": slot_logits, "acquisition_matrix": acquisition_matrix}


@SELECTORS.register_module()
class PCOTMRASCNNFrameScout(nn.Module):
    """Small temporal-CNN scout with the same slot-logit contract as TinyTransformer."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_slots: int = 384,
        num_layers: int = 3,
        kernel_size: int = 5,
        dropout: float = 0.10,
        dilation_base: int = 1,
        dilations: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if int(in_dim) <= 0:
            raise ValueError("in_dim must be positive")
        if int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        if int(num_slots) <= 0:
            raise ValueError("num_slots must be positive")
        self.num_slots = int(num_slots)
        self.input_proj = nn.Conv1d(int(in_dim) + 1, int(hidden_dim), kernel_size=1)
        self.blocks = _MaskedTemporalConvStack(
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            kernel_size=int(kernel_size),
            dropout=float(dropout),
            dilation_base=int(dilation_base),
            dilations=dilations,
        )
        self.norm = nn.LayerNorm(int(hidden_dim))
        self.slot_queries = nn.Parameter(torch.randn(self.num_slots, int(hidden_dim)) * 0.02)

    def forward(self, features: torch.Tensor, valid: torch.Tensor, time_coords: torch.Tensor | None = None):
        valid, time_coords = _validate_frame_scout_inputs(features, valid, time_coords)
        x = torch.cat([features, time_coords.to(dtype=features.dtype).unsqueeze(-1)], dim=-1)
        x = x.masked_fill(~valid.unsqueeze(-1), 0.0).transpose(1, 2)
        encoded = self.blocks(self.input_proj(x), valid).transpose(1, 2)
        encoded = self.norm(encoded).masked_fill(~valid.unsqueeze(-1), 0.0)
        slot_logits = torch.einsum("bth,kh->bkt", encoded, self.slot_queries) * (encoded.shape[-1] ** -0.5)
        slot_logits, acquisition_matrix = _masked_slot_transport(slot_logits, valid)
        return {"slot_logits": slot_logits, "acquisition_matrix": acquisition_matrix}


@SELECTORS.register_module()
class PCOTMRASMotionTCNFrameScout(nn.Module):
    """Motion-aware TCN scout using only deploy-visible compressed descriptors."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_slots: int = 384,
        num_layers: int = 3,
        kernel_size: int = 5,
        dropout: float = 0.10,
        dilation_base: int = 2,
        dilations: Sequence[int] | None = None,
        motion_feature_mode: str = "frame_delta_abs",
        motion_delta_stride: int = 1,
        motion_rgb_fusion: str = "descriptor_plus_delta",
    ) -> None:
        super().__init__()
        if int(in_dim) <= 0:
            raise ValueError("in_dim must be positive")
        if int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        if int(num_slots) <= 0:
            raise ValueError("num_slots must be positive")
        if str(motion_feature_mode) != "frame_delta_abs":
            raise ValueError("PCOTMRASMotionTCNFrameScout supports only motion_feature_mode='frame_delta_abs'")
        if int(motion_delta_stride) <= 0:
            raise ValueError("motion_delta_stride must be positive")
        if str(motion_rgb_fusion) != "descriptor_plus_delta":
            raise ValueError("PCOTMRASMotionTCNFrameScout supports only motion_rgb_fusion='descriptor_plus_delta'")
        self.num_slots = int(num_slots)
        self.motion_delta_stride = int(motion_delta_stride)
        motion_dim = int(in_dim) * 3 + 4
        self.input_proj = nn.Conv1d(motion_dim, int(hidden_dim), kernel_size=1)
        self.blocks = _MaskedTemporalConvStack(
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            kernel_size=int(kernel_size),
            dropout=float(dropout),
            dilation_base=int(dilation_base),
            dilations=dilations,
        )
        self.norm = nn.LayerNorm(int(hidden_dim))
        self.slot_queries = nn.Parameter(torch.randn(self.num_slots, int(hidden_dim)) * 0.02)

    def _motion_descriptor(self, features: torch.Tensor, valid: torch.Tensor, time_coords: torch.Tensor) -> torch.Tensor:
        stride = self.motion_delta_stride
        prev_valid = torch.zeros_like(valid)
        prev_valid[:, stride:] = valid[:, :-stride]
        valid_pair = valid & prev_valid
        diff = torch.zeros_like(features)
        diff[:, stride:] = features[:, stride:] - features[:, :-stride]
        diff = diff.masked_fill(~valid_pair.unsqueeze(-1), 0.0)
        abs_diff = diff.abs()
        motion_energy = abs_diff.mean(dim=-1, keepdim=True)
        local_energy = F.avg_pool1d(
            motion_energy.transpose(1, 2),
            kernel_size=3,
            stride=1,
            padding=1,
        ).transpose(1, 2)
        time_delta = torch.zeros_like(time_coords)
        time_delta[:, stride:] = time_coords[:, stride:] - time_coords[:, :-stride]
        time_delta = time_delta.masked_fill(~valid_pair, 0.0).unsqueeze(-1)
        descriptor = torch.cat(
            [
                features,
                diff,
                abs_diff,
                motion_energy,
                local_energy,
                time_coords.unsqueeze(-1),
                time_delta,
            ],
            dim=-1,
        )
        return descriptor.masked_fill(~valid.unsqueeze(-1), 0.0)

    def forward(self, features: torch.Tensor, valid: torch.Tensor, time_coords: torch.Tensor | None = None):
        valid, time_coords = _validate_frame_scout_inputs(features, valid, time_coords)
        features = features.masked_fill(~valid.unsqueeze(-1), 0.0)
        descriptor = self._motion_descriptor(features, valid, time_coords)
        encoded = self.blocks(self.input_proj(descriptor.transpose(1, 2)), valid).transpose(1, 2)
        encoded = self.norm(encoded).masked_fill(~valid.unsqueeze(-1), 0.0)
        slot_logits = torch.einsum("bth,kh->bkt", encoded, self.slot_queries) * (encoded.shape[-1] ** -0.5)
        slot_logits, acquisition_matrix = _masked_slot_transport(slot_logits, valid)
        return {"slot_logits": slot_logits, "acquisition_matrix": acquisition_matrix}


@SELECTORS.register_module()
class PCOTMRASHybridFrameScout(nn.Module):
    """Descriptor-projection plus temporal-TCN scout with learned slot queries."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 96,
        num_slots: int = 384,
        num_layers: int = 3,
        kernel_size: int = 5,
        temporal_layers: int | None = None,
        temporal_kernel_size: int | None = None,
        dropout: float = 0.10,
        dilation_base: int = 2,
        dilations: Sequence[int] | None = None,
        descriptor_hidden_dim: int | None = None,
        slot_mlp_layers: int = 1,
        slot_hidden_dim: int | None = None,
        slot_dropout: float = 0.0,
        slot_temperature_init: float = 1.0,
        local_global_fusion: str = "temporal_cnn_plus_slot_attention",
    ) -> None:
        super().__init__()
        if int(in_dim) <= 0:
            raise ValueError("in_dim must be positive")
        if int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        if int(num_slots) <= 0:
            raise ValueError("num_slots must be positive")
        if temporal_layers is not None:
            num_layers = int(temporal_layers)
        if temporal_kernel_size is not None:
            kernel_size = int(temporal_kernel_size)
        descriptor_hidden_dim = int(descriptor_hidden_dim or hidden_dim)
        if descriptor_hidden_dim <= 0:
            raise ValueError("descriptor_hidden_dim must be positive")
        if int(slot_mlp_layers) <= 0:
            raise ValueError("slot_mlp_layers must be positive")
        slot_hidden_dim = int(slot_hidden_dim or hidden_dim)
        if slot_hidden_dim <= 0:
            raise ValueError("slot_hidden_dim must be positive")
        if float(slot_temperature_init) <= 0:
            raise ValueError("slot_temperature_init must be positive")
        if str(local_global_fusion) != "temporal_cnn_plus_slot_attention":
            raise ValueError("PCOTMRASHybridFrameScout supports only local_global_fusion='temporal_cnn_plus_slot_attention'")
        self.num_slots = int(num_slots)
        self.slot_temperature = float(slot_temperature_init)
        self.descriptor_proj = nn.Sequential(
            nn.LayerNorm(int(in_dim)),
            nn.Linear(int(in_dim), descriptor_hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(descriptor_hidden_dim, int(hidden_dim)),
        )
        self.time_proj = nn.Linear(1, int(hidden_dim))
        self.blocks = _MaskedTemporalConvStack(
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            kernel_size=int(kernel_size),
            dropout=float(dropout),
            dilation_base=int(dilation_base),
            dilations=dilations,
        )
        self.norm = nn.LayerNorm(int(hidden_dim))
        slot_layers: list[nn.Module] = []
        for layer_idx in range(int(slot_mlp_layers) - 1):
            in_features = int(hidden_dim) if layer_idx == 0 else slot_hidden_dim
            slot_layers.extend(
                [
                    nn.Linear(in_features, slot_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(float(slot_dropout)),
                ]
            )
        slot_layers.append(nn.Linear(slot_hidden_dim if int(slot_mlp_layers) > 1 else int(hidden_dim), int(hidden_dim)))
        self.slot_mlp = nn.Sequential(*slot_layers)
        self.slot_dropout = nn.Dropout(float(slot_dropout))
        self.slot_queries = nn.Parameter(torch.randn(self.num_slots, int(hidden_dim)) * 0.02)

    def forward(self, features: torch.Tensor, valid: torch.Tensor, time_coords: torch.Tensor | None = None):
        valid, time_coords = _validate_frame_scout_inputs(features, valid, time_coords)
        features = features.masked_fill(~valid.unsqueeze(-1), 0.0)
        frame_tokens = self.descriptor_proj(features) + self.time_proj(time_coords.unsqueeze(-1))
        frame_tokens = frame_tokens.masked_fill(~valid.unsqueeze(-1), 0.0)
        encoded = self.blocks(frame_tokens.transpose(1, 2), valid).transpose(1, 2)
        encoded = self.norm(encoded).masked_fill(~valid.unsqueeze(-1), 0.0)
        slot_features = self.slot_dropout(self.slot_mlp(encoded)).masked_fill(~valid.unsqueeze(-1), 0.0)
        slot_logits = torch.einsum("bth,kh->bkt", slot_features, self.slot_queries) * (
            slot_features.shape[-1] ** -0.5
        )
        slot_logits = slot_logits / self.slot_temperature
        slot_logits, acquisition_matrix = _masked_slot_transport(slot_logits, valid)
        return {"slot_logits": slot_logits, "acquisition_matrix": acquisition_matrix}


@SELECTORS.register_module()
class PCOTMRASBoundaryDifficultyTemporalFrameScout(nn.Module):
    """Boundary/difficulty-aware scout with dense frame heads and slot transport.

    This reader keeps the pre-backbone hard-frame interface, but exposes the
    Pro-requested dense evidence heads so selection can be diagnosed as a
    task-aware frame acquisition policy instead of an opaque slot allocator.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_slots: int = 384,
        temporal_layers: int = 3,
        temporal_kernel_size: int = 5,
        dilations: Sequence[int] | None = (1, 2, 4),
        dropout: float = 0.05,
        descriptor_hidden_dim: int | None = None,
        slot_temperature_init: float = 1.0,
        geometry_width: float = 0.015,
        geometry_bias_weight: float = 0.75,
        action_bias_weight: float = 0.40,
        boundary_bias_weight: float = 0.60,
        uncertainty_bias_weight: float = 0.25,
        redundancy_bias_weight: float = 0.25,
        slot_logit_clamp: float = 30.0,
        soft_order_regularizer_weight: float = 1.0,
        duplicate_mass_regularizer_weight: float = 0.2,
        duplicate_mass_cap_factor: float = 4.0,
        local_global_fusion: str = "boundary_difficulty_temporal_cnn_slot_attention",
    ) -> None:
        super().__init__()
        if int(in_dim) <= 0:
            raise ValueError("in_dim must be positive")
        if int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        if int(num_slots) <= 0:
            raise ValueError("num_slots must be positive")
        if int(temporal_layers) <= 0:
            raise ValueError("temporal_layers must be positive")
        if float(slot_temperature_init) <= 0.0:
            raise ValueError("slot_temperature_init must be positive")
        if float(geometry_width) <= 0.0:
            raise ValueError("geometry_width must be positive")
        if float(soft_order_regularizer_weight) < 0.0:
            raise ValueError("soft_order_regularizer_weight must be non-negative")
        if float(duplicate_mass_regularizer_weight) < 0.0:
            raise ValueError("duplicate_mass_regularizer_weight must be non-negative")
        if float(duplicate_mass_cap_factor) <= 0.0:
            raise ValueError("duplicate_mass_cap_factor must be positive")
        if str(local_global_fusion) != "boundary_difficulty_temporal_cnn_slot_attention":
            raise ValueError(
                "PCOTMRASBoundaryDifficultyTemporalFrameScout supports only "
                "local_global_fusion='boundary_difficulty_temporal_cnn_slot_attention'"
            )
        self.num_slots = int(num_slots)
        self.slot_temperature = float(slot_temperature_init)
        self.geometry_width = float(geometry_width)
        self.geometry_bias_weight = float(geometry_bias_weight)
        self.action_bias_weight = float(action_bias_weight)
        self.boundary_bias_weight = float(boundary_bias_weight)
        self.uncertainty_bias_weight = float(uncertainty_bias_weight)
        self.redundancy_bias_weight = float(redundancy_bias_weight)
        self.slot_logit_clamp = float(slot_logit_clamp)
        self.soft_order_regularizer_weight = float(soft_order_regularizer_weight)
        self.duplicate_mass_regularizer_weight = float(duplicate_mass_regularizer_weight)
        self.duplicate_mass_cap_factor = float(duplicate_mass_cap_factor)

        descriptor_hidden_dim = int(descriptor_hidden_dim or hidden_dim)
        self.descriptor_proj = nn.Sequential(
            nn.LayerNorm(int(in_dim)),
            nn.Linear(int(in_dim), descriptor_hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(descriptor_hidden_dim, int(hidden_dim)),
        )
        self.time_proj = nn.Linear(1, int(hidden_dim))
        self.temporal = _MaskedTemporalConvStack(
            hidden_dim=int(hidden_dim),
            num_layers=int(temporal_layers),
            kernel_size=int(temporal_kernel_size),
            dropout=float(dropout),
            dilations=dilations,
        )
        self.norm = nn.LayerNorm(int(hidden_dim))
        self.slot_queries = nn.Parameter(torch.randn(self.num_slots, int(hidden_dim)) * 0.02)
        self.register_buffer(
            "base_slot_centers",
            torch.linspace(0.0, 1.0, steps=self.num_slots, dtype=torch.float32),
            persistent=False,
        )
        self.action_head = nn.Linear(int(hidden_dim), 1)
        self.start_head = nn.Linear(int(hidden_dim), 1)
        self.end_head = nn.Linear(int(hidden_dim), 1)
        self.uncertainty_head = nn.Linear(int(hidden_dim), 1)
        self.redundancy_head = nn.Linear(int(hidden_dim), 1)
        self.role_head = nn.Linear(int(hidden_dim), 5)

    def _geometry_bias(self, time_coords: torch.Tensor) -> torch.Tensor:
        width = max(float(self.geometry_width), 1.0 / float(max(1, self.num_slots * 4)))
        delta = time_coords.float()[:, None, :] - self.base_slot_centers.float()[None, :, None]
        geometry_bias = -0.5 * (delta / width).square()
        _require_finite(geometry_bias, "boundary difficulty geometry bias")
        return geometry_bias

    def forward(self, features: torch.Tensor, valid: torch.Tensor, time_coords: torch.Tensor | None = None):
        valid, time_coords = _validate_frame_scout_inputs(features, valid, time_coords)
        features = features.float().masked_fill(~valid.unsqueeze(-1), 0.0)
        encoded = self.descriptor_proj(features) + self.time_proj(time_coords.float().unsqueeze(-1))
        encoded = encoded.masked_fill(~valid.unsqueeze(-1), 0.0)
        encoded = self.temporal(encoded.transpose(1, 2), valid).transpose(1, 2)
        encoded = self.norm(encoded).masked_fill(~valid.unsqueeze(-1), 0.0)
        _require_finite(encoded, "boundary difficulty encoded tokens")

        action_logits = _masked_frame_logits(self.action_head(encoded).squeeze(-1), valid, "action_logits")
        start_logits = _masked_frame_logits(self.start_head(encoded).squeeze(-1), valid, "start_logits")
        end_logits = _masked_frame_logits(self.end_head(encoded).squeeze(-1), valid, "end_logits")
        boundary_logits = torch.maximum(start_logits, end_logits).masked_fill(~valid, 0.0)
        _require_finite(boundary_logits, "boundary_logits", error_type=ValueError)
        uncertainty_logits = _masked_frame_logits(
            self.uncertainty_head(encoded).squeeze(-1),
            valid,
            "uncertainty_logits",
        )
        redundancy_logits = _masked_frame_logits(
            self.redundancy_head(encoded).squeeze(-1),
            valid,
            "redundancy_logits",
        )
        role_logits = self.role_head(encoded).float().masked_fill(~valid.unsqueeze(-1), 0.0)
        _require_finite(role_logits, "role_logits", error_type=ValueError)

        frame_selection_logits = (
            self.action_bias_weight * action_logits
            + self.boundary_bias_weight * boundary_logits
            + self.uncertainty_bias_weight * uncertainty_logits
            - self.redundancy_bias_weight * redundancy_logits
        ).masked_fill(~valid, 0.0)
        _require_finite(frame_selection_logits, "frame_selection_logits", error_type=ValueError)

        content_logits = torch.einsum("bth,kh->bkt", encoded, self.slot_queries.float())
        content_logits = content_logits * (encoded.shape[-1] ** -0.5)
        slot_logits = (
            content_logits.float() / self.slot_temperature
            + self.geometry_bias_weight * self._geometry_bias(time_coords)
            + frame_selection_logits[:, None, :]
        )
        if self.slot_logit_clamp > 0.0:
            slot_logits = slot_logits.clamp(min=-self.slot_logit_clamp, max=self.slot_logit_clamp)
        _require_finite(slot_logits, "boundary difficulty slot logits", error_type=ValueError)
        slot_logits, acquisition_matrix = _masked_slot_transport(slot_logits, valid)
        soft_centers = (acquisition_matrix * time_coords.float()[:, None, :]).sum(dim=-1)
        center_diffs = soft_centers[:, 1:] - soft_centers[:, :-1]
        order_regularizer = (
            F.relu(-center_diffs).square().mean() if center_diffs.numel() else soft_centers.sum() * 0.0
        )
        column_mass = acquisition_matrix.masked_fill(~valid[:, None, :], 0.0).sum(dim=1)
        valid_count = valid.float().sum(dim=1).clamp_min(1.0)
        expected_mass = float(self.num_slots) / valid_count
        duplicate_cap = expected_mass[:, None] * self.duplicate_mass_cap_factor
        duplicate_excess = F.relu(column_mass / duplicate_cap.clamp_min(1.0e-6) - 1.0)
        duplicate_regularizer = duplicate_excess.masked_select(valid).square().mean() if bool(valid.any().item()) else column_mass.sum() * 0.0
        regularizer = (
            self.soft_order_regularizer_weight * order_regularizer
            + self.duplicate_mass_regularizer_weight * duplicate_regularizer
        )
        _require_finite(order_regularizer, "boundary difficulty soft order regularizer")
        _require_finite(duplicate_regularizer, "boundary difficulty duplicate mass regularizer")
        _require_finite(regularizer, "boundary difficulty total regularizer")
        return {
            "slot_logits": slot_logits,
            "acquisition_matrix": acquisition_matrix,
            "action_logits": action_logits,
            "actionness_logits": action_logits,
            "value_logits": action_logits,
            "start_logits": start_logits,
            "end_logits": end_logits,
            "boundary_logits": boundary_logits,
            "risk_logits": boundary_logits,
            "uncertainty_logits": uncertainty_logits,
            "redundancy_logits": redundancy_logits,
            "role_logits": role_logits,
            "frame_selection_logits": frame_selection_logits,
            "regularizers": {
                "soft_order_regularizer": order_regularizer,
                "duplicate_mass_regularizer": duplicate_regularizer,
                "total_regularizer": regularizer,
            },
        }


@SELECTORS.register_module()
class PCOTMRASLowResPixelTemporalFrameScout(PCOTMRASBoundaryDifficultyTemporalFrameScout):
    """Compatibility alias for the Pro boundary/difficulty temporal scout."""


@SELECTORS.register_module()
class PCOTMRASLowResolutionPixelTemporalFrameScout(PCOTMRASBoundaryDifficultyTemporalFrameScout):
    """Compatibility alias for the Pro boundary/difficulty temporal scout."""


@SELECTORS.register_module()
class PCOTMRASLowResPixelTemporalFrameReader(PCOTMRASBoundaryDifficultyTemporalFrameScout):
    """Compatibility alias for the Pro boundary/difficulty temporal scout."""


@SELECTORS.register_module()
class PCOTMRASRSeriesHybridFrameScout(nn.Module):
    """R-series inspired pre-backbone reader for deploy-visible low-res pixels.

    The module keeps the C3 hard-real-frame contract while adding the pieces that
    made the R-series selector diagnostically meaningful: ordered slot geometry,
    action/boundary/difficulty/redundancy frame heads, and gated slot allocation.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_slots: int = 384,
        temporal_layers: int = 3,
        temporal_kernel_size: int = 5,
        dilations: Sequence[int] | None = (1, 2, 4),
        dropout: float = 0.05,
        descriptor_hidden_dim: int | None = None,
        slot_mlp_layers: int = 2,
        slot_hidden_dim: int | None = None,
        slot_temperature_init: float = 1.0,
        center_offset_scale: float = 0.35,
        width_init: float = 0.0125,
        width_min: float = 0.0025,
        width_max: float = 0.08,
        geometry_bias_weight: float = 1.0,
        action_bias_weight: float = 0.35,
        boundary_bias_weight: float = 0.45,
        uncertainty_bias_weight: float = 0.20,
        redundancy_bias_weight: float = 0.25,
        slot_logit_clamp: float = 30.0,
        local_global_fusion: str = "rseries_temporal_geometry_slot_attention",
    ) -> None:
        super().__init__()
        if int(in_dim) <= 0:
            raise ValueError("in_dim must be positive")
        if int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        if int(num_slots) <= 0:
            raise ValueError("num_slots must be positive")
        if int(temporal_layers) <= 0:
            raise ValueError("temporal_layers must be positive")
        if int(slot_mlp_layers) <= 0:
            raise ValueError("slot_mlp_layers must be positive")
        if float(slot_temperature_init) <= 0.0:
            raise ValueError("slot_temperature_init must be positive")
        if str(local_global_fusion) != "rseries_temporal_geometry_slot_attention":
            raise ValueError(
                "PCOTMRASRSeriesHybridFrameScout supports only "
                "local_global_fusion='rseries_temporal_geometry_slot_attention'"
            )
        self.num_slots = int(num_slots)
        self.slot_temperature = float(slot_temperature_init)
        self.center_offset_scale = float(center_offset_scale)
        self.width_min = float(width_min)
        self.width_max = float(width_max)
        self.geometry_bias_weight = float(geometry_bias_weight)
        self.action_bias_weight = float(action_bias_weight)
        self.boundary_bias_weight = float(boundary_bias_weight)
        self.uncertainty_bias_weight = float(uncertainty_bias_weight)
        self.redundancy_bias_weight = float(redundancy_bias_weight)
        self.slot_logit_clamp = float(slot_logit_clamp)

        descriptor_hidden_dim = int(descriptor_hidden_dim or hidden_dim)
        slot_hidden_dim = int(slot_hidden_dim or hidden_dim)
        self.descriptor_proj = nn.Sequential(
            nn.LayerNorm(int(in_dim)),
            nn.Linear(int(in_dim), descriptor_hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(descriptor_hidden_dim, int(hidden_dim)),
        )
        self.time_proj = nn.Linear(1, int(hidden_dim))
        self.temporal = _MaskedTemporalConvStack(
            hidden_dim=int(hidden_dim),
            num_layers=int(temporal_layers),
            kernel_size=int(temporal_kernel_size),
            dropout=float(dropout),
            dilations=dilations,
        )
        self.norm = nn.LayerNorm(int(hidden_dim))

        slot_layers: list[nn.Module] = []
        for layer_idx in range(int(slot_mlp_layers) - 1):
            in_features = int(hidden_dim) if layer_idx == 0 else slot_hidden_dim
            slot_layers.extend([nn.Linear(in_features, slot_hidden_dim), nn.GELU(), nn.Dropout(float(dropout))])
        slot_layers.append(nn.Linear(slot_hidden_dim if int(slot_mlp_layers) > 1 else int(hidden_dim), int(hidden_dim)))
        self.slot_mlp = nn.Sequential(*slot_layers)
        self.slot_queries = nn.Parameter(torch.randn(self.num_slots, int(hidden_dim)) * 0.02)
        base_centers = torch.linspace(0.0, 1.0, steps=self.num_slots, dtype=torch.float32)
        self.register_buffer("base_slot_centers", base_centers, persistent=False)
        self.center_offsets = nn.Parameter(torch.zeros(self.num_slots))
        self.width_logits = nn.Parameter(torch.full((self.num_slots,), _inverse_softplus(float(width_init))))
        self.slot_gate_logits = nn.Parameter(torch.zeros(self.num_slots))

        self.value_head = nn.Linear(int(hidden_dim), 1)
        self.boundary_head = nn.Linear(int(hidden_dim), 1)
        self.uncertainty_head = nn.Linear(int(hidden_dim), 1)
        self.redundancy_head = nn.Linear(int(hidden_dim), 1)
        self.role_head = nn.Linear(int(hidden_dim), 4)

    def _slot_geometry_bias(self, time_coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_offset = self.center_offset_scale / float(max(1, self.num_slots))
        centers = (self.base_slot_centers + torch.tanh(self.center_offsets.float()) * max_offset).clamp(0.0, 1.0)
        widths = (F.softplus(self.width_logits.float()) + self.width_min).clamp(max=self.width_max)
        gates = torch.sigmoid(self.slot_gate_logits.float()).clamp(1.0e-4, 1.0)
        delta = time_coords.float()[:, None, :] - centers[None, :, None]
        geometry_bias = -0.5 * (delta / widths[None, :, None].clamp_min(1.0e-4)).square()
        geometry_bias = geometry_bias + gates.log()[None, :, None]
        _require_finite(geometry_bias, "slot geometry bias")
        _require_finite(centers, "slot centers")
        _require_finite(widths, "slot widths")
        return geometry_bias, centers, widths

    def forward(self, features: torch.Tensor, valid: torch.Tensor, time_coords: torch.Tensor | None = None):
        valid, time_coords = _validate_frame_scout_inputs(features, valid, time_coords)
        features = features.float().masked_fill(~valid.unsqueeze(-1), 0.0)
        frame_tokens = self.descriptor_proj(features) + self.time_proj(time_coords.float().unsqueeze(-1))
        frame_tokens = frame_tokens.masked_fill(~valid.unsqueeze(-1), 0.0)
        _require_finite(frame_tokens, "rseries frame tokens")
        encoded = self.temporal(frame_tokens.transpose(1, 2), valid).transpose(1, 2)
        encoded = self.norm(encoded).masked_fill(~valid.unsqueeze(-1), 0.0)
        _require_finite(encoded, "rseries encoded tokens")
        slot_features = self.slot_mlp(encoded).masked_fill(~valid.unsqueeze(-1), 0.0)
        _require_finite(slot_features, "rseries slot features")

        content_logits = torch.einsum("bth,kh->bkt", slot_features, self.slot_queries.float())
        content_logits = content_logits * (slot_features.shape[-1] ** -0.5)
        _require_finite(content_logits, "rseries content slot logits")
        geometry_bias, centers, widths = self._slot_geometry_bias(time_coords)

        value_logits = _masked_frame_logits(self.value_head(encoded).squeeze(-1), valid, "value_logits")
        boundary_logits = _masked_frame_logits(self.boundary_head(encoded).squeeze(-1), valid, "risk_logits")
        uncertainty_logits = _masked_frame_logits(
            self.uncertainty_head(encoded).squeeze(-1),
            valid,
            "uncertainty_logits",
        )
        redundancy_logits = _masked_frame_logits(
            self.redundancy_head(encoded).squeeze(-1),
            valid,
            "redundancy_logits",
        )
        role_logits = self.role_head(encoded).float().masked_fill(~valid.unsqueeze(-1), 0.0)
        _require_finite(role_logits, "role_logits", error_type=ValueError)

        task_bias = (
            self.action_bias_weight * value_logits[:, None, :]
            + self.boundary_bias_weight * boundary_logits[:, None, :]
            + self.uncertainty_bias_weight * uncertainty_logits[:, None, :]
            - self.redundancy_bias_weight * redundancy_logits[:, None, :]
        )
        slot_logits = (content_logits.float() / self.slot_temperature) + self.geometry_bias_weight * geometry_bias + task_bias
        if self.slot_logit_clamp > 0.0:
            slot_logits = slot_logits.clamp(min=-self.slot_logit_clamp, max=self.slot_logit_clamp)
        _require_finite(slot_logits, "rseries slot logits")
        slot_logits, acquisition_matrix = _masked_slot_transport(slot_logits, valid)
        center_diffs = centers[1:] - centers[:-1]
        order_regularizer = F.relu(-center_diffs).square().mean() if center_diffs.numel() else centers.sum() * 0.0
        width_regularizer = F.relu(widths - self.width_max).square().mean() + F.relu(self.width_min - widths).square().mean()
        _require_finite(order_regularizer, "slot order regularizer")
        _require_finite(width_regularizer, "slot width regularizer")
        return {
            "slot_logits": slot_logits,
            "acquisition_matrix": acquisition_matrix,
            "action_logits": value_logits,
            "value_logits": value_logits,
            "boundary_logits": boundary_logits,
            "risk_logits": boundary_logits,
            "uncertainty_logits": uncertainty_logits,
            "redundancy_logits": redundancy_logits,
            "role_logits": role_logits,
            "regularizers": {
                "order_regularizer": order_regularizer,
                "width_regularizer": width_regularizer,
                "total_regularizer": order_regularizer + width_regularizer,
            },
        }


@SELECTORS.register_module()
class PCOTMRASCoarseActionnessFrameScout(nn.Module):
    """Binary action/background scout for coarse uncertainty-driven sampling."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 96,
        num_slots: int | None = None,
        temporal_layers: int = 3,
        temporal_kernel_size: int = 5,
        dilations: Sequence[int] | None = (1, 2, 4),
        dropout: float = 0.05,
        descriptor_hidden_dim: int | None = None,
        action_bias_weight: float = 1.0,
        uncertainty_bias_weight: float = 0.75,
        change_bias_weight: float = 0.75,
        score_logit_eps: float = 1.0e-4,
        local_global_fusion: str = "coarse_actionness_temporal_cnn",
    ) -> None:
        super().__init__()
        if int(in_dim) <= 0:
            raise ValueError("in_dim must be positive")
        if int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        if int(temporal_layers) <= 0:
            raise ValueError("temporal_layers must be positive")
        if num_slots is not None and int(num_slots) <= 0:
            raise ValueError("num_slots must be positive when provided")
        if float(action_bias_weight) < 0.0:
            raise ValueError("action_bias_weight must be non-negative")
        if float(uncertainty_bias_weight) < 0.0:
            raise ValueError("uncertainty_bias_weight must be non-negative")
        if float(change_bias_weight) < 0.0:
            raise ValueError("change_bias_weight must be non-negative")
        if not 0.0 < float(score_logit_eps) < 0.5:
            raise ValueError("score_logit_eps must lie inside (0, 0.5)")
        if str(local_global_fusion) != "coarse_actionness_temporal_cnn":
            raise ValueError(
                "PCOTMRASCoarseActionnessFrameScout supports only "
                "local_global_fusion='coarse_actionness_temporal_cnn'"
            )
        descriptor_hidden_dim = int(descriptor_hidden_dim or hidden_dim)
        self.action_bias_weight = float(action_bias_weight)
        self.uncertainty_bias_weight = float(uncertainty_bias_weight)
        self.change_bias_weight = float(change_bias_weight)
        self.score_logit_eps = float(score_logit_eps)
        self.descriptor_proj = nn.Sequential(
            nn.LayerNorm(int(in_dim)),
            nn.Linear(int(in_dim), descriptor_hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(descriptor_hidden_dim, int(hidden_dim)),
        )
        self.time_proj = nn.Linear(1, int(hidden_dim))
        self.temporal = _MaskedTemporalConvStack(
            hidden_dim=int(hidden_dim),
            num_layers=int(temporal_layers),
            kernel_size=int(temporal_kernel_size),
            dropout=float(dropout),
            dilations=dilations,
        )
        self.norm = nn.LayerNorm(int(hidden_dim))
        self.action_head = nn.Linear(int(hidden_dim), 1)

    @staticmethod
    def _binary_uncertainty_scores(action_prob: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eps = 1.0e-6
        prob = action_prob.float().clamp(min=eps, max=1.0 - eps)
        entropy_score = -(prob * prob.log() + (1.0 - prob) * (1.0 - prob).log()) / math.log(2.0)
        margin_uncertainty = 1.0 - (2.0 * prob - 1.0).abs()
        uncertainty_score = torch.maximum(entropy_score, margin_uncertainty).clamp(0.0, 1.0)

        valid_pair_prev = valid[:, 1:] & valid[:, :-1]
        prev_change = torch.zeros_like(prob)
        prev_change[:, 1:] = (prob[:, 1:] - prob[:, :-1]).abs().masked_fill(~valid_pair_prev, 0.0)
        next_change = torch.zeros_like(prob)
        next_change[:, :-1] = (prob[:, 1:] - prob[:, :-1]).abs().masked_fill(~valid_pair_prev, 0.0)
        change_score = torch.maximum(prev_change, next_change).clamp(0.0, 1.0)
        return entropy_score, uncertainty_score, change_score

    def forward(self, features: torch.Tensor, valid: torch.Tensor, time_coords: torch.Tensor | None = None):
        valid, time_coords = _validate_frame_scout_inputs(features, valid, time_coords)
        features = features.float().masked_fill(~valid.unsqueeze(-1), 0.0)
        encoded = self.descriptor_proj(features) + self.time_proj(time_coords.float().unsqueeze(-1))
        encoded = encoded.masked_fill(~valid.unsqueeze(-1), 0.0)
        encoded = self.temporal(encoded.transpose(1, 2), valid).transpose(1, 2)
        encoded = self.norm(encoded).masked_fill(~valid.unsqueeze(-1), 0.0)
        _require_finite(encoded, "coarse actionness encoded tokens")

        action_logits = _masked_frame_logits(self.action_head(encoded).squeeze(-1), valid, "action_logits")
        action_prob = torch.sigmoid(action_logits).masked_fill(~valid, 0.0)
        entropy_score, uncertainty_score, change_score = self._binary_uncertainty_scores(action_prob, valid)
        entropy_score = entropy_score.masked_fill(~valid, 0.0)
        uncertainty_score = uncertainty_score.masked_fill(~valid, 0.0)
        change_score = change_score.masked_fill(~valid, 0.0)
        margin_score = (1.0 - (2.0 * action_prob.float().clamp(1.0e-6, 1.0 - 1.0e-6) - 1.0).abs()).clamp(
            0.0,
            1.0,
        )
        margin_score = margin_score.masked_fill(~valid, 0.0)
        background_context_score = ((1.0 - action_prob) * (1.0 - uncertainty_score)).masked_fill(~valid, 0.0)

        normalizer = self.action_bias_weight + self.uncertainty_bias_weight + self.change_bias_weight
        if normalizer <= 0.0:
            raise ValueError("at least one coarse actionness score weight must be positive")
        mixed_score = (
            self.action_bias_weight * action_prob
            + self.uncertainty_bias_weight * uncertainty_score
            + self.change_bias_weight * change_score
        ) / normalizer
        mixed_score = mixed_score.clamp(min=self.score_logit_eps, max=1.0 - self.score_logit_eps)
        frame_selection_logits = torch.logit(mixed_score).masked_fill(~valid, 0.0)
        _require_finite(frame_selection_logits, "coarse actionness frame selection logits")

        return {
            "action_logits": action_logits,
            "actionness_logits": action_logits,
            "value_logits": action_logits,
            "action_prob": action_prob,
            "p_action": action_prob,
            "entropy_score": entropy_score,
            "entropy": entropy_score,
            "margin_score": margin_score,
            "margin": margin_score,
            "uncertainty_score": uncertainty_score,
            "change_score": change_score,
            "p_change": change_score,
            "transition_score": change_score,
            "background_context_score": background_context_score,
            "frame_selection_logits": frame_selection_logits,
            "regularizers": {"total_regularizer": action_logits.sum() * 0.0},
        }


@SELECTORS.register_module()
class PCOTMRASPreBackboneFrameSelector(nn.Module):
    """Online PC-OT-MRAS frame acquisition before the video backbone.

    The selector consumes dense raw frames, emits a fixed-length selected-frame
    tensor for the unchanged AdaTAD/ActionFormer detector, optionally remaps
    training GT to the selected axis, and mutates metas in-place so downstream
    heads can recover the physical temporal geometry.
    """

    forbid_raw_prediction_cache = True

    def __init__(
        self,
        reader: Mapping[str, Any],
        target_len: int = 384,
        dense_window_size: int = 768,
        descriptor_dim: int = 4,
        transport_topk: int = 1,
        eval_transport_topk: int = 1,
        protected_uniform_count: int = 0,
        coverage_guard_count: int = 0,
        scout_feature_source: str = "handcrafted_descriptors",
        scout_spatial_size: int | Sequence[int] = 32,
        selection_unit: int = 1,
        selection_unit2_supported: bool = True,
        residual_count: int | None = None,
        residual_slot_role: str = "learned_residual",
        selector_support_status: str = "supported",
        straight_through_detector_loss: bool = True,
        straight_through_downstream: bool | None = None,
        remap_gt_to_selected_axis: bool = True,
        aux_gt_acquisition_loss_weight: float = 0.05,
        aux_frame_score_boundary_loss_weight: float = 0.0,
        aux_duplicate_cap_loss_weight: float = 0.001,
        aux_duplicate_column_cap: float = 1.5,
        aux_value_loss_weight: float = 0.0,
        aux_risk_loss_weight: float = 0.0,
        aux_uncertainty_loss_weight: float = 0.0,
        aux_redundancy_loss_weight: float = 0.0,
        aux_role_entropy_loss_weight: float = 0.0,
        reader_regularizer_loss_weight: float = 0.01,
        st_surrogate_mode: str = "mean_proxy",
        scout_pixel_normalize: bool = True,
        scout_pixel_clamp: float = 5.0,
        max_dense_gap: int = 0,
        max_gap_guard_count: int = 0,
        max_gap: int | None = None,
        selection_strategy: str = "slot_transport",
        frame_score_st_temperature: float = 1.0,
        frame_score_st_local_width: float = 8.0,
        frame_score_st_local_bias_weight: float = 1.0,
        frame_score_st_surrogate: str = "local_softmax",
        frame_score_st_logit_clamp: float = 0.0,
        frame_score_st_gradient_scale: float = 1.0,
        frame_score_aux_logit_clamp: float = 0.0,
        global_rank_st_temperature: float | None = None,
        global_rank_st_topk: int | None = None,
        global_rank_st_rank_width: float = 1.0,
        interval_boundary_budget_ratio: float = 0.5,
        interval_candidate_topk: int = 16,
        coarse_uniform_count: int = 160,
        coarse_action_count: int = 96,
        coarse_uncertainty_count: int = 80,
        coarse_change_count: int = 32,
        coarse_background_count: int = 16,
        coarse_action_weight: float = 1.0,
        coarse_uncertainty_weight: float = 0.75,
        coarse_change_weight: float = 0.75,
        dynamic_budget: Mapping[str, Any] | None = None,
        meta_source: str = "pc_ot_mras_prebackbone_e2e_frame_selector",
    ) -> None:
        super().__init__()
        if int(target_len) <= 0:
            raise ValueError("target_len must be positive")
        if int(dense_window_size) <= 0:
            raise ValueError("dense_window_size must be positive")
        if int(target_len) > int(dense_window_size):
            raise ValueError("target_len must not exceed dense_window_size")
        if int(descriptor_dim) <= 0:
            raise ValueError("descriptor_dim must be positive")
        if int(transport_topk) <= 0 or int(eval_transport_topk) <= 0:
            raise ValueError("transport_topk and eval_transport_topk must be positive")
        if int(transport_topk) != 1 or int(eval_transport_topk) != 1:
            raise ValueError(
                "PCOTMRASPreBackboneFrameSelector currently supports only hard top-1 transport; "
                f"got transport_topk={transport_topk}, eval_transport_topk={eval_transport_topk}"
            )
        if int(protected_uniform_count) < 0 or int(coverage_guard_count) < 0:
            raise ValueError("uniform/coverage guard counts must be non-negative")
        if str(scout_feature_source) not in ("handcrafted_descriptors", "compressed_pixels"):
            raise ValueError("scout_feature_source must be handcrafted_descriptors or compressed_pixels")
        if isinstance(scout_spatial_size, Sequence) and not isinstance(scout_spatial_size, (str, bytes)):
            if len(scout_spatial_size) != 2:
                raise ValueError("scout_spatial_size sequence must contain height and width")
            scout_height = int(scout_spatial_size[0])
            scout_width = int(scout_spatial_size[1])
        else:
            scout_height = int(scout_spatial_size)
            scout_width = int(scout_spatial_size)
        if scout_height <= 0 or scout_width <= 0:
            raise ValueError("scout_spatial_size must be positive")
        if int(selection_unit) not in (1, 2):
            raise ValueError("selection_unit must be 1 or 2 for the C3 frame-level candidate")
        if int(selection_unit) == 2 and not bool(selection_unit2_supported):
            raise ValueError("selection_unit=2 requires selection_unit2_supported=True")
        if residual_count is not None:
            residual_count = int(residual_count)
            if residual_count <= 0:
                raise ValueError("residual_count must be positive when provided")
            if residual_count > int(target_len):
                raise ValueError("residual_count must not exceed target_len")

        reader_cfg = dict(reader)
        reader_cfg.setdefault("type", "PCOTMRASReader")
        reader_cfg.setdefault("in_dim", int(descriptor_dim))
        reader_cfg.setdefault("num_slots", int(target_len))
        if int(reader_cfg["in_dim"]) != int(descriptor_dim):
            raise ValueError("reader in_dim must match selector descriptor_dim")
        if str(reader_cfg.get("type")) == "PCOTMRASRSeriesHybridFrameScout":
            if str(scout_feature_source) != "compressed_pixels":
                raise ValueError("PCOTMRASRSeriesHybridFrameScout requires scout_feature_source='compressed_pixels'")
            if (scout_height, scout_width) != (32, 32):
                raise ValueError("PCOTMRASRSeriesHybridFrameScout requires 32x32 compressed pixel descriptors")
            if int(descriptor_dim) != 3 * 32 * 32:
                raise ValueError("PCOTMRASRSeriesHybridFrameScout requires descriptor_dim=3072 for RGB 32x32 pixels")
        self.reader = build_selector(reader_cfg)
        self.target_len = int(target_len)
        self.dense_window_size = int(dense_window_size)
        self.descriptor_dim = int(descriptor_dim)
        self.transport_topk = int(transport_topk)
        self.eval_transport_topk = int(eval_transport_topk)
        self.protected_uniform_count = min(int(protected_uniform_count), self.target_len)
        self.coverage_guard_count = min(int(coverage_guard_count), self.target_len)
        self.scout_feature_source = str(scout_feature_source)
        self.scout_spatial_size = (scout_height, scout_width)
        self.selection_unit = int(selection_unit)
        self.selection_unit2_supported = bool(selection_unit2_supported)
        self.residual_count = residual_count
        self.residual_slot_role = str(residual_slot_role)
        self.selector_support_status = str(selector_support_status)
        if straight_through_downstream is None:
            straight_through_downstream = straight_through_detector_loss
        self.straight_through_downstream = bool(straight_through_downstream)
        self.straight_through_detector_loss = self.straight_through_downstream
        self.remap_gt_to_selected_axis = bool(remap_gt_to_selected_axis)
        self.aux_gt_acquisition_loss_weight = float(aux_gt_acquisition_loss_weight)
        self.aux_frame_score_boundary_loss_weight = float(aux_frame_score_boundary_loss_weight)
        self.aux_duplicate_cap_loss_weight = float(aux_duplicate_cap_loss_weight)
        self.aux_duplicate_column_cap = float(aux_duplicate_column_cap)
        self.aux_value_loss_weight = float(aux_value_loss_weight)
        self.aux_risk_loss_weight = float(aux_risk_loss_weight)
        self.aux_uncertainty_loss_weight = float(aux_uncertainty_loss_weight)
        self.aux_redundancy_loss_weight = float(aux_redundancy_loss_weight)
        self.aux_role_entropy_loss_weight = float(aux_role_entropy_loss_weight)
        self.reader_regularizer_loss_weight = float(reader_regularizer_loss_weight)
        if str(st_surrogate_mode) not in ("mean_proxy", "full_flat"):
            raise ValueError("st_surrogate_mode must be 'mean_proxy' or 'full_flat'")
        if max_gap is not None:
            max_dense_gap = int(max_gap)
        if int(max_dense_gap) < 0:
            raise ValueError("max_dense_gap must be non-negative")
        if int(max_gap_guard_count) < 0:
            raise ValueError("max_gap_guard_count must be non-negative")
        if str(selection_strategy) not in (
            "slot_transport",
            "frame_score_topk",
            "frame_score_global_rank_st",
            "interval_boundary_packet",
            "coarse_actionness_uncertainty",
        ):
            raise ValueError(
                "selection_strategy must be 'slot_transport', 'frame_score_topk', "
                "'frame_score_global_rank_st', 'interval_boundary_packet', "
                "or 'coarse_actionness_uncertainty'"
            )
        if float(frame_score_st_temperature) <= 0.0:
            raise ValueError("frame_score_st_temperature must be positive")
        if float(frame_score_st_local_width) <= 0.0:
            raise ValueError("frame_score_st_local_width must be positive")
        if float(frame_score_st_local_bias_weight) < 0.0:
            raise ValueError("frame_score_st_local_bias_weight must be non-negative")
        if str(frame_score_st_surrogate) not in ("local_softmax", "global_softmax", "global_rank_topk"):
            raise ValueError(
                "frame_score_st_surrogate must be 'local_softmax', 'global_softmax', or 'global_rank_topk'"
            )
        if str(selection_strategy) == "frame_score_global_rank_st" and str(frame_score_st_surrogate) != "global_rank_topk":
            raise ValueError("frame_score_global_rank_st requires frame_score_st_surrogate='global_rank_topk'")
        if float(frame_score_st_logit_clamp) < 0.0:
            raise ValueError("frame_score_st_logit_clamp must be non-negative")
        if not 0.0 <= float(frame_score_st_gradient_scale) <= 1.0:
            raise ValueError("frame_score_st_gradient_scale must be in [0, 1]")
        if float(frame_score_aux_logit_clamp) < 0.0:
            raise ValueError("frame_score_aux_logit_clamp must be non-negative")
        if float(aux_frame_score_boundary_loss_weight) < 0.0:
            raise ValueError("aux_frame_score_boundary_loss_weight must be non-negative")
        if global_rank_st_temperature is None:
            global_rank_st_temperature = float(frame_score_st_temperature)
        if float(global_rank_st_temperature) <= 0.0:
            raise ValueError("global_rank_st_temperature must be positive")
        if global_rank_st_topk is None:
            global_rank_st_topk = int(target_len)
        if int(global_rank_st_topk) <= 0:
            raise ValueError("global_rank_st_topk must be positive")
        if int(global_rank_st_topk) > int(target_len):
            raise ValueError("global_rank_st_topk must not exceed target_len")
        if float(global_rank_st_rank_width) <= 0.0:
            raise ValueError("global_rank_st_rank_width must be positive")
        if not 0.0 <= float(interval_boundary_budget_ratio) <= 1.0:
            raise ValueError("interval_boundary_budget_ratio must be in [0, 1]")
        if int(interval_candidate_topk) <= 0:
            raise ValueError("interval_candidate_topk must be positive")
        for name, value in (
            ("coarse_uniform_count", coarse_uniform_count),
            ("coarse_action_count", coarse_action_count),
            ("coarse_uncertainty_count", coarse_uncertainty_count),
            ("coarse_change_count", coarse_change_count),
            ("coarse_background_count", coarse_background_count),
        ):
            if int(value) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name, value in (
            ("coarse_action_weight", coarse_action_weight),
            ("coarse_uncertainty_weight", coarse_uncertainty_weight),
            ("coarse_change_weight", coarse_change_weight),
        ):
            if float(value) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        self.st_surrogate_mode = str(st_surrogate_mode)
        self.scout_pixel_normalize = bool(scout_pixel_normalize)
        self.scout_pixel_clamp = float(scout_pixel_clamp)
        self.max_dense_gap = int(max_dense_gap)
        self.max_gap_guard_count = min(int(max_gap_guard_count), self.target_len)
        self.selection_strategy = str(selection_strategy)
        self.frame_score_st_temperature = float(frame_score_st_temperature)
        self.frame_score_st_local_width = float(frame_score_st_local_width)
        self.frame_score_st_local_bias_weight = float(frame_score_st_local_bias_weight)
        self.frame_score_st_surrogate = str(frame_score_st_surrogate)
        self.frame_score_st_logit_clamp = float(frame_score_st_logit_clamp)
        self.frame_score_st_gradient_scale = float(frame_score_st_gradient_scale)
        self.frame_score_aux_logit_clamp = float(frame_score_aux_logit_clamp)
        self.global_rank_st_temperature = float(global_rank_st_temperature)
        self.global_rank_st_topk = int(global_rank_st_topk)
        self.global_rank_st_rank_width = float(global_rank_st_rank_width)
        self.interval_boundary_budget_ratio = float(interval_boundary_budget_ratio)
        self.interval_candidate_topk = int(interval_candidate_topk)
        self.coarse_uniform_count = int(coarse_uniform_count)
        self.coarse_action_count = int(coarse_action_count)
        self.coarse_uncertainty_count = int(coarse_uncertainty_count)
        self.coarse_change_count = int(coarse_change_count)
        self.coarse_background_count = int(coarse_background_count)
        self.coarse_action_weight = float(coarse_action_weight)
        self.coarse_uncertainty_weight = float(coarse_uncertainty_weight)
        self.coarse_change_weight = float(coarse_change_weight)
        self.dynamic_budget = self._normalize_dynamic_budget_config(dynamic_budget)
        self.meta_source = str(meta_source)
        self._metadata_dump_count = 0

    def forward_train(self, inputs, masks, metas, gt_segments, gt_labels):
        outputs = self._select(inputs=inputs, masks=masks, metas=metas, gt_segments=gt_segments, training=True)
        new_gt_segments, new_gt_labels = self._remap_gt_batch(
            gt_segments=gt_segments,
            gt_labels=gt_labels,
            selected_positions=outputs["selected_positions"],
            valid_lengths=outputs["valid_lengths"],
            selected_output_valid_lengths=outputs["selected_output_valid_lengths"],
        )
        losses = self._losses(
            reader_outputs=outputs["reader_outputs"],
            valid_mask=outputs["valid_mask"],
            candidate_dense_indices=outputs["candidate_dense_indices"],
            gt_segments=gt_segments,
        )
        return {
            "inputs": outputs["inputs"],
            "masks": outputs["masks"],
            "metas": outputs["metas"],
            "gt_segments": new_gt_segments,
            "gt_labels": new_gt_labels,
            "losses": losses,
        }

    def forward_test(self, inputs, masks, metas=None):
        outputs = self._select(inputs=inputs, masks=masks, metas=metas, training=False)
        return {
            "inputs": outputs["inputs"],
            "masks": outputs["masks"],
            "metas": outputs["metas"],
        }

    def _select(self, inputs, masks, metas, *, training: bool, gt_segments=None) -> dict[str, Any]:
        batch, dense_len = self._input_batch_and_time(inputs)
        if dense_len != int(self.dense_window_size):
            raise ValueError(f"expected dense_window_size={self.dense_window_size}, got {dense_len}")
        valid = _as_bool_prefix_mask(masks, expected_shape=(batch, dense_len))
        valid_lengths = valid.long().sum(dim=1)

        descriptors = self._scout_frame_features(inputs, valid)
        candidate_descriptors, candidate_valid, candidate_dense_indices, candidate_time_coords = self._candidate_grid(
            descriptors=descriptors,
            valid=valid,
        )
        reader_outputs = self.reader(candidate_descriptors, candidate_valid, time_coords=candidate_time_coords)
        for name in (
            "slot_logits",
            "acquisition_matrix",
            "allocation",
            "action_logits",
            "value_logits",
            "boundary_logits",
            "risk_logits",
            "start_logits",
            "end_logits",
            "uncertainty_logits",
            "uncertainty_score",
            "change_score",
            "transition_score",
            "background_context_score",
            "redundancy_logits",
            "role_logits",
            "frame_selection_logits",
        ):
            tensor = reader_outputs.get(name)
            if torch.is_tensor(tensor):
                _require_finite(tensor, f"reader output {name}")
        plan = self._sparse_transport_plan(
            reader_outputs,
            valid,
            candidate_valid=candidate_valid,
            candidate_dense_indices=candidate_dense_indices,
            training=training,
        )
        selected_inputs = self._apply_sparse_transport(
            inputs,
            plan["indices"],
            plan["weights"],
            transport_weights=plan.get("transport_weights") if training else None,
        )
        output_axis = torch.arange(self.target_len, device=masks.device)[None, :]
        selected_masks = output_axis < plan["selected_output_valid_lengths"][:, None].to(device=masks.device)
        metas = self._write_selected_axis_meta(
            metas=metas,
            selected_positions=plan["selected_positions"],
            valid_lengths=valid_lengths,
            selected_output_valid_lengths=plan["selected_output_valid_lengths"],
            selected_roles=plan.get("selected_roles"),
            raw_slot_dense_indices=plan.get("raw_slot_dense_indices"),
            raw_slot_duplicate_rates=plan.get("raw_slot_duplicate_rates"),
            raw_slot_unique_counts=plan.get("raw_slot_unique_counts"),
            reader_fill_counts=plan.get("reader_fill_counts"),
            st_active_row_counts=plan.get("st_active_row_counts"),
            dynamic_budget_meta=plan.get("dynamic_budget_meta"),
            max_gap_guard_meta=plan.get("max_gap_guard_meta"),
            interval_packet_metadata=plan.get("interval_packet_metadata"),
            coarse_policy_meta=plan.get("coarse_policy_meta"),
            reader_outputs=reader_outputs,
            candidate_valid=candidate_valid,
            candidate_dense_indices=candidate_dense_indices,
            gt_segments=gt_segments,
            training=training,
        )
        return {
            "inputs": selected_inputs,
            "masks": selected_masks,
            "metas": metas,
            "reader_outputs": reader_outputs,
            "valid_mask": candidate_valid,
            "valid_lengths": valid_lengths,
            "candidate_dense_indices": candidate_dense_indices,
            "selected_positions": plan["selected_positions"],
            "selected_output_valid_lengths": plan["selected_output_valid_lengths"],
        }

    @staticmethod
    def _input_batch_and_time(inputs: torch.Tensor) -> tuple[int, int]:
        if inputs.ndim == 6:
            return int(inputs.shape[0]), int(inputs.shape[3])
        if inputs.ndim == 5:
            return int(inputs.shape[0]), int(inputs.shape[2])
        raise ValueError(f"inputs must be [B,N,C,T,H,W] or [B,C,T,H,W], got {tuple(inputs.shape)}")

    def _raw_frame_descriptors(self, inputs: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        video = self._video_tensor_for_descriptors(inputs).float()
        channel_mean = video.mean(dim=(3, 4)).transpose(1, 2)
        channel_std = video.std(dim=(3, 4), unbiased=False).transpose(1, 2)
        channel_mean = self._fit_descriptor_width(channel_mean, width=3)
        channel_std = self._fit_descriptor_width(channel_std, width=3)

        if video.shape[1] >= 3:
            luma = (
                0.299 * video[:, 0]
                + 0.587 * video[:, 1]
                + 0.114 * video[:, 2]
            ).mean(dim=(2, 3))
        else:
            luma = video.mean(dim=(1, 3, 4))

        motion = torch.zeros_like(luma)
        motion[:, 1:] = torch.abs(video[:, :, 1:] - video[:, :, :-1]).mean(dim=(1, 3, 4))

        grad_h = torch.zeros_like(luma)
        grad_w = torch.zeros_like(luma)
        if video.shape[3] > 1:
            grad_h = torch.abs(video[:, :, :, 1:, :] - video[:, :, :, :-1, :]).mean(dim=(1, 3, 4))
        if video.shape[4] > 1:
            grad_w = torch.abs(video[:, :, :, :, 1:] - video[:, :, :, :, :-1]).mean(dim=(1, 3, 4))
        edge = 0.5 * (grad_h + grad_w)
        edge_delta = torch.zeros_like(edge)
        edge_delta[:, 1:] = torch.abs(edge[:, 1:] - edge[:, :-1])
        motion_delta = torch.zeros_like(motion)
        motion_delta[:, 1:] = torch.abs(motion[:, 1:] - motion[:, :-1])
        coords = self._time_coords(valid, dtype=video.dtype)

        descriptors = torch.cat(
            [
                channel_mean,
                channel_std,
                luma.unsqueeze(-1),
                motion.unsqueeze(-1),
                edge.unsqueeze(-1),
                edge_delta.unsqueeze(-1),
                coords.unsqueeze(-1),
                motion_delta.unsqueeze(-1),
            ],
            dim=-1,
        )
        descriptors = self._fit_descriptor_width(descriptors, width=self.descriptor_dim)
        return descriptors.masked_fill(~valid.unsqueeze(-1), 0.0)

    def _scout_frame_features(self, inputs: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        if self.scout_feature_source == "compressed_pixels":
            return self._compressed_pixel_features(inputs, valid)
        return self._raw_frame_descriptors(inputs, valid)

    def _compressed_pixel_features(self, inputs: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        video = self._video_tensor_for_descriptors(inputs).float()
        video = self._normalize_scout_pixels(video)
        batch, channels, dense_len, height, width = video.shape
        frames = video.permute(0, 2, 1, 3, 4).reshape(batch * dense_len, channels, height, width)
        compressed = F.interpolate(
            frames,
            size=self.scout_spatial_size,
            mode="bilinear",
            align_corners=False,
        )
        features = compressed.reshape(batch, dense_len, channels * self.scout_spatial_size[0] * self.scout_spatial_size[1])
        features = self._fit_descriptor_width(features, width=self.descriptor_dim)
        _require_finite(features, "compressed scout features")
        return features.masked_fill(~valid.unsqueeze(-1), 0.0)

    def _normalize_scout_pixels(self, video: torch.Tensor) -> torch.Tensor:
        if not self.scout_pixel_normalize:
            _require_finite(video, "raw scout pixel tensor")
            return video
        video = video.float()
        _require_finite(video, "raw scout pixel tensor")
        max_abs = video.detach().abs().amax()
        if bool((max_abs > 2.0).item()):
            video = video / 255.0
        mean = video.mean(dim=(-2, -1), keepdim=True)
        std = video.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(1.0e-4)
        video = (video - mean) / std
        if self.scout_pixel_clamp > 0.0:
            video = video.clamp(min=-self.scout_pixel_clamp, max=self.scout_pixel_clamp)
        _require_finite(video, "normalized scout pixel tensor")
        return video

    @staticmethod
    def _video_tensor_for_descriptors(inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 6:
            return inputs.float().mean(dim=1)
        if inputs.ndim == 5:
            return inputs.float()
        raise ValueError(f"unsupported input shape: {tuple(inputs.shape)}")

    @staticmethod
    def _fit_descriptor_width(features: torch.Tensor, *, width: int) -> torch.Tensor:
        current = int(features.shape[-1])
        if current == int(width):
            return features
        if current > int(width):
            return features[..., : int(width)]
        pad_shape = (*features.shape[:-1], int(width) - current)
        pad = features.new_zeros(pad_shape)
        return torch.cat([features, pad], dim=-1)

    @staticmethod
    def _time_coords(valid: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        dense_len = int(valid.shape[1])
        denom = valid.long().sum(dim=1).clamp(min=1).to(dtype=dtype) - 1.0
        denom = denom.clamp_min(1.0)
        coords = torch.arange(dense_len, device=valid.device, dtype=dtype)[None, :] / denom[:, None]
        return coords.masked_fill(~valid, 0.0)

    def _candidate_grid(
        self,
        *,
        descriptors: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, dense_len, descriptor_dim = descriptors.shape
        if self.selection_unit == 1:
            dense_indices = torch.arange(dense_len, device=descriptors.device, dtype=torch.long)[None, :].expand(batch, -1)
            return descriptors, valid, dense_indices, self._time_coords(valid, dtype=descriptors.dtype)

        unit = int(self.selection_unit)
        grid_len = (dense_len + unit - 1) // unit
        padded_len = grid_len * unit
        pad_len = padded_len - dense_len
        if pad_len:
            desc_pad = descriptors.new_zeros((batch, pad_len, descriptor_dim))
            valid_pad = valid.new_zeros((batch, pad_len))
            descriptors_padded = torch.cat([descriptors, desc_pad], dim=1)
            valid_padded = torch.cat([valid, valid_pad], dim=1)
        else:
            descriptors_padded = descriptors
            valid_padded = valid
        grouped_desc = descriptors_padded.view(batch, grid_len, unit, descriptor_dim)
        grouped_valid = valid_padded.view(batch, grid_len, unit)
        candidate_valid = grouped_valid.any(dim=2)
        count = grouped_valid.long().sum(dim=2).clamp(min=1).to(dtype=descriptors.dtype)
        candidate_descriptors = (grouped_desc * grouped_valid.unsqueeze(-1).to(dtype=descriptors.dtype)).sum(dim=2)
        candidate_descriptors = candidate_descriptors / count.unsqueeze(-1)
        starts = torch.arange(grid_len, device=descriptors.device, dtype=torch.long) * unit
        candidate_dense_indices = starts[None, :].expand(batch, -1).clone()
        valid_last = valid.long().sum(dim=1).clamp(min=1)[:, None] - 1
        candidate_dense_indices = torch.minimum(candidate_dense_indices, valid_last)
        denom = valid.long().sum(dim=1).clamp(min=1).to(dtype=descriptors.dtype) - 1.0
        denom = denom.clamp_min(1.0)
        candidate_time_coords = candidate_dense_indices.to(dtype=descriptors.dtype) / denom[:, None]
        candidate_time_coords = candidate_time_coords.masked_fill(~candidate_valid, 0.0)
        return (
            candidate_descriptors.masked_fill(~candidate_valid.unsqueeze(-1), 0.0),
            candidate_valid,
            candidate_dense_indices,
            candidate_time_coords,
        )

    def _sparse_transport_plan(
        self,
        reader_outputs: Mapping[str, torch.Tensor],
        valid: torch.Tensor,
        *,
        candidate_valid: torch.Tensor,
        candidate_dense_indices: torch.Tensor,
        training: bool,
    ) -> dict[str, torch.Tensor]:
        if getattr(self, "selection_strategy", "slot_transport") in (
            "frame_score_topk",
            "frame_score_global_rank_st",
        ):
            return self._frame_score_transport_plan(
                reader_outputs=reader_outputs,
                valid=valid,
                candidate_valid=candidate_valid,
                candidate_dense_indices=candidate_dense_indices,
                training=training,
            )
        if getattr(self, "selection_strategy", "slot_transport") == "coarse_actionness_uncertainty":
            return self._coarse_actionness_uncertainty_transport_plan(
                reader_outputs=reader_outputs,
                valid=valid,
                candidate_valid=candidate_valid,
                candidate_dense_indices=candidate_dense_indices,
                training=training,
            )
        if getattr(self, "selection_strategy", "slot_transport") == "interval_boundary_packet":
            return self._interval_boundary_packet_transport_plan(
                reader_outputs=reader_outputs,
                valid=valid,
                candidate_valid=candidate_valid,
                candidate_dense_indices=candidate_dense_indices,
                training=training,
            )
        matrix = reader_outputs.get("acquisition_matrix")
        if matrix is None:
            matrix = reader_outputs.get("allocation")
        logits = reader_outputs.get("slot_logits")
        if matrix is None and logits is None:
            raise ValueError("prebackbone selector requires slot_logits, acquisition_matrix, or allocation")
        if matrix is None:
            matrix = logits
        if torch.is_tensor(logits):
            _require_finite(logits, "slot_logits", error_type=ValueError)
        if torch.is_tensor(matrix):
            _require_finite(matrix, "acquisition matrix", error_type=ValueError)
        if matrix.ndim != 3:
            raise ValueError(f"acquisition matrix must be [B,K,T], got {tuple(matrix.shape)}")
        batch, slots, candidate_len = matrix.shape
        dense_len = int(valid.shape[1])
        expected_slots = self.target_len if self.residual_count is None else int(self.residual_count)
        if slots != expected_slots or candidate_len != candidate_valid.shape[1]:
            raise ValueError(
                "acquisition matrix shape must match target_len/residual_count and candidate axis"
            )
        if tuple(candidate_dense_indices.shape) != tuple(candidate_valid.shape):
            raise ValueError("candidate_dense_indices must match candidate_valid")

        if logits is not None and logits.shape != matrix.shape:
            raise ValueError("slot_logits shape must match acquisition_matrix/allocation when both are provided")
        candidate_valid = candidate_valid.to(device=matrix.device).bool()
        candidate_dense_indices = candidate_dense_indices.to(device=matrix.device)
        valid = valid.to(device=matrix.device).bool()
        if bool((candidate_valid.long().sum(dim=1) <= 0).any().item()):
            raise ValueError("each sample must contain at least one valid sparse transport candidate")

        matrix_fp32 = matrix.float()
        logits_fp32 = logits.float() if logits is not None else None
        score_source = logits_fp32 if logits_fp32 is not None else matrix_fp32
        min_score = torch.finfo(torch.float32).min
        hard_scores = score_source.masked_fill(~candidate_valid[:, None, :], min_score)
        _require_finite(hard_scores, "masked hard transport scores")
        candidate_indices = hard_scores.argmax(dim=-1)
        selected_positions = candidate_dense_indices.gather(dim=1, index=candidate_indices).to(dtype=torch.float32)
        if logits_fp32 is not None:
            soft_surrogate = F.softmax(hard_scores, dim=-1).masked_fill(~candidate_valid[:, None, :], 0.0)
        else:
            soft_surrogate = matrix_fp32.masked_fill(~candidate_valid[:, None, :], 0.0).clamp_min(0.0)
        soft_surrogate = soft_surrogate / soft_surrogate.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(torch.float32).eps
        )
        _require_finite(soft_surrogate, "soft transport surrogate")
        column_scores = soft_surrogate.sum(dim=1).masked_fill(~candidate_valid, min_score)
        _require_finite(column_scores, "sparse transport column scores")

        topk = 1
        fixed_indices = torch.empty((batch, self.target_len, topk), dtype=torch.long, device=matrix.device)
        fixed_weights = torch.ones((batch, self.target_len, topk), dtype=torch.float32, device=matrix.device)
        fixed_positions = torch.empty((batch, self.target_len), dtype=torch.float32, device=matrix.device)
        transport_weights = torch.zeros((batch, self.target_len, dense_len), dtype=torch.float32, device=matrix.device)
        selected_roles: list[list[str]] = []
        raw_slot_dense_indices: list[list[int]] = []
        raw_slot_duplicate_rates: list[float] = []
        raw_slot_unique_counts: list[int] = []
        reader_fill_counts: list[int] = []
        st_active_row_counts: list[int] = []
        selected_output_valid_lengths = torch.empty((batch,), dtype=torch.long, device=valid.device)
        protected_count = max(self.protected_uniform_count, self.coverage_guard_count)

        for batch_idx in range(batch):
            valid_positions = candidate_dense_indices[batch_idx][candidate_valid[batch_idx]]
            output_valid_len = min(int(valid_positions.numel()), self.target_len)
            selected_output_valid_lengths[batch_idx] = output_valid_len
            raw_positions = [int(pos) for pos in selected_positions[batch_idx].detach().cpu().tolist()]
            raw_slot_dense_indices.append(raw_positions)
            raw_unique_count = len(set(raw_positions))
            raw_slot_unique_counts.append(raw_unique_count)
            raw_slot_duplicate_rates.append(
                1.0 - float(raw_unique_count) / float(max(1, len(raw_positions)))
            )
            if valid_positions.numel() < self.target_len:
                batch_roles = []
                pad_pos = int(valid_positions[-1].item())
                for out_idx in range(self.target_len):
                    if out_idx < int(valid_positions.numel()):
                        pos = int(valid_positions[out_idx].item())
                        role = "valid_prefix"
                    else:
                        pos = pad_pos
                        role = "pad_repeat"
                    fixed_positions[batch_idx, out_idx] = float(pos)
                    fixed_indices[batch_idx, out_idx, 0] = pos
                    fixed_weights[batch_idx, out_idx, 0] = 1.0
                    transport_weights[batch_idx, out_idx, pos] = 1.0
                    batch_roles.append(role)
                selected_roles.append(batch_roles)
                reader_fill_counts.append(0)
                st_active_row_counts.append(0)
                continue
            protected_positions = self._uniform_anchor_positions(
                valid_positions=valid_positions,
                count=min(protected_count, output_valid_len),
            )
            max_gap_positions = self._max_gap_guard_positions(
                valid_positions=valid_positions,
                count=getattr(self, "max_gap_guard_count", 0) or output_valid_len,
                max_gap=getattr(self, "max_dense_gap", 0),
            )
            order = torch.argsort(selected_positions[batch_idx], stable=True)
            used: set[int] = set()
            rows: list[tuple[int, int | None, str]] = []
            for pos_tensor in protected_positions:
                pos = int(pos_tensor.item())
                used.add(pos)
                rows.append((pos, None, "uniform_protected"))
                if len(rows) >= self.target_len:
                    break
            for pos_tensor in max_gap_positions:
                pos = int(pos_tensor.item())
                if pos in used:
                    continue
                used.add(pos)
                rows.append((pos, None, "max_gap_guard"))
                if len(rows) >= self.target_len:
                    break
            for slot_tensor in order:
                slot = int(slot_tensor.item())
                pos = int(selected_positions[batch_idx, slot].item())
                if pos in used or not bool(valid[batch_idx, pos].item()):
                    continue
                used.add(pos)
                rows.append((pos, slot, self.residual_slot_role if self.residual_count is not None else "reader_selected"))
                if len(rows) >= self.target_len:
                    break

            if len(rows) < self.target_len:
                ranked_fill = torch.argsort(column_scores[batch_idx], descending=True, stable=True)
                for candidate_tensor in ranked_fill:
                    candidate_idx = int(candidate_tensor.item())
                    pos = int(candidate_dense_indices[batch_idx, candidate_idx].item())
                    if pos in used or not bool(valid[batch_idx, pos].item()):
                        continue
                    used.add(pos)
                    rows.append((pos, None, "reader_fill"))
                    if len(rows) >= self.target_len:
                        break

            if len(rows) < self.target_len:
                for pos_tensor in valid_positions:
                    pos = int(pos_tensor.item())
                    if pos in used or not bool(valid[batch_idx, pos].item()):
                        continue
                    used.add(pos)
                    rows.append((pos, None, "dense_fill"))
                    if len(rows) >= self.target_len:
                        break

            if len(rows) > self.target_len:
                rows = rows[: self.target_len]

            if len(rows) != self.target_len:
                raise ValueError(
                    "failed to resolve fixed-count sparse transport plan: "
                    f"rows={len(rows)}, target_len={self.target_len}, used={len(used)}, "
                    f"valid_positions={int(valid_positions.numel())}, "
                    f"max_dense_gap={getattr(self, 'max_dense_gap', 0)}"
                )
            rows.sort(key=lambda item: item[0])
            batch_roles = []
            for out_idx, (pos, slot, role) in enumerate(rows):
                fixed_positions[batch_idx, out_idx] = float(pos)
                fixed_indices[batch_idx, out_idx, 0] = pos
                fixed_weights[batch_idx, out_idx, 0] = 1.0
                batch_roles.append(role)
                hard = torch.zeros((dense_len,), dtype=torch.float32, device=matrix.device)
                hard[pos] = 1.0
                if self.straight_through_detector_loss and training and slot is not None:
                    # Straight-through contract: hard + soft_surrogate - detach(soft_surrogate).
                    soft_dense = torch.zeros((dense_len,), dtype=torch.float32, device=matrix.device)
                    soft_dense.scatter_add_(
                        0,
                        candidate_dense_indices[batch_idx].to(device=matrix.device),
                        soft_surrogate[batch_idx, slot],
                    )
                    transport_weights[batch_idx, out_idx] = hard + soft_dense - soft_dense.detach()
                else:
                    transport_weights[batch_idx, out_idx] = hard
            selected_roles.append(batch_roles)
            reader_fill_counts.append(sum(1 for _pos, _slot, role in rows if role == "reader_fill"))
            st_active_row_counts.append(sum(1 for _pos, slot, _role in rows if slot is not None))

        _require_finite(fixed_weights, "sparse transport fixed weights")
        _require_finite(fixed_positions, "sparse transport selected positions")
        _require_finite(transport_weights, "sparse transport weights")
        return {
            "indices": fixed_indices,
            "weights": fixed_weights,
            "transport_weights": transport_weights,
            "selected_positions": fixed_positions,
            "selected_output_valid_lengths": selected_output_valid_lengths,
            "selected_roles": selected_roles,
            "raw_slot_dense_indices": raw_slot_dense_indices,
            "raw_slot_duplicate_rates": raw_slot_duplicate_rates,
            "raw_slot_unique_counts": raw_slot_unique_counts,
            "reader_fill_counts": reader_fill_counts,
            "st_active_row_counts": st_active_row_counts,
        }

    def _dense_head_or_zeros(
        self,
        *,
        reader_outputs: Mapping[str, torch.Tensor],
        candidate_valid: torch.Tensor,
        names: Sequence[str],
        device: torch.device,
    ) -> tuple[torch.Tensor, str]:
        for name in names:
            tensor = reader_outputs.get(name)
            if tensor is None:
                continue
            if not torch.is_tensor(tensor):
                raise TypeError(f"selector dense head {name} must be a tensor")
            if tuple(tensor.shape) != tuple(candidate_valid.shape):
                raise ValueError(
                    f"selector dense head {name} must match candidate axis; "
                    f"got {tuple(tensor.shape)}, expected {tuple(candidate_valid.shape)}"
                )
            _require_finite(tensor, f"selector dense head {name}", error_type=ValueError)
            return tensor.to(device=device).float(), name
        return torch.zeros(candidate_valid.shape, dtype=torch.float32, device=device), "zeros"

    def _rank_transport_candidate_distribution(
        self,
        *,
        scores: torch.Tensor,
        candidate_valid: torch.Tensor,
        candidate_dense_indices: torch.Tensor,
        candidate_idx: int | None,
        hard_rank_position: int | None = None,
        topk_budget: int | None = None,
        global_rank_cache: Mapping[str, torch.Tensor] | None = None,
        name: str,
    ) -> torch.Tensor:
        if scores.ndim != 1 or candidate_valid.ndim != 1 or candidate_dense_indices.ndim != 1:
            raise ValueError("rank transport distribution expects one-dimensional per-sample tensors")
        if tuple(scores.shape) != tuple(candidate_valid.shape) or tuple(scores.shape) != tuple(candidate_dense_indices.shape):
            raise ValueError("rank transport score, valid, and dense-index axes must match")
        _require_finite(scores, f"{name} rank transport scores", error_type=ValueError)
        temperature = float(getattr(self, "frame_score_st_temperature", 1.0))
        min_score = torch.finfo(torch.float32).min
        logits = scores.float() / temperature
        surrogate = getattr(self, "frame_score_st_surrogate", "local_softmax")
        if surrogate == "local_softmax":
            if candidate_idx is None:
                raise ValueError("local_softmax rank transport requires a hard candidate index")
            local_width = float(getattr(self, "frame_score_st_local_width", 8.0))
            local_bias_weight = float(getattr(self, "frame_score_st_local_bias_weight", 1.0))
            center = candidate_dense_indices[int(candidate_idx)].to(dtype=torch.float32)
            distances = candidate_dense_indices.to(dtype=torch.float32) - center
            logits = logits + (-0.5 * (distances / local_width).square() * local_bias_weight)
        elif surrogate == "global_softmax":
            pass
        elif surrogate == "global_rank_topk":
            if hard_rank_position is None:
                raise ValueError("global_rank_topk rank transport requires a hard rank position")
            rank_width = float(getattr(self, "global_rank_st_rank_width", 1.0))
            if global_rank_cache is None:
                global_rank_cache = self._global_rank_topk_cache(
                    scores=scores,
                    candidate_valid=candidate_valid,
                    topk_budget=topk_budget,
                    name=name,
                )
            soft_rank = global_rank_cache["soft_rank"]
            log_membership = global_rank_cache["log_membership"]
            _require_finite(soft_rank, f"{name} global soft ranks")
            rank_center = float(int(hard_rank_position) + 1)
            rank_logits = -0.5 * ((soft_rank - rank_center) / rank_width).square()
            logits = rank_logits + log_membership
        else:
            raise ValueError(f"unknown frame_score_st_surrogate={surrogate}")
        logits = _smooth_clamp_logits(
            logits,
            float(getattr(self, "frame_score_st_logit_clamp", 0.0)),
            f"{name} rank transport logits",
        )
        logits = logits.masked_fill(~candidate_valid.bool(), min_score)
        _require_finite(logits, f"{name} rank transport logits")
        soft_candidate = F.softmax(logits, dim=0).masked_fill(~candidate_valid.bool(), 0.0)
        soft_candidate = soft_candidate / soft_candidate.sum().clamp_min(torch.finfo(torch.float32).eps)
        _require_finite(soft_candidate, f"{name} rank transport distribution")
        return soft_candidate

    def _global_rank_topk_cache(
        self,
        *,
        scores: torch.Tensor,
        candidate_valid: torch.Tensor,
        topk_budget: int | None,
        name: str,
    ) -> dict[str, torch.Tensor]:
        if scores.ndim != 1 or candidate_valid.ndim != 1 or tuple(scores.shape) != tuple(candidate_valid.shape):
            raise ValueError("global_rank_topk cache expects matching one-dimensional score and valid tensors")
        _require_finite(scores, f"{name} global rank cache scores", error_type=ValueError)
        valid_bool = candidate_valid.bool()
        valid_count = int(valid_bool.long().sum().item())
        if valid_count <= 0:
            raise ValueError("global_rank_topk rank transport requires at least one valid candidate")
        topk_limit = int(topk_budget) if topk_budget is not None else int(getattr(self, "global_rank_st_topk", self.target_len))
        topk_limit = max(1, min(topk_limit, valid_count))
        rank_temperature = float(getattr(self, "global_rank_st_temperature", getattr(self, "frame_score_st_temperature", 1.0)))
        rank_width = float(getattr(self, "global_rank_st_rank_width", 1.0))
        rank_scores = _smooth_clamp_logits(
            scores.float(),
            float(getattr(self, "frame_score_st_logit_clamp", 0.0)),
            f"{name} global rank scores",
        )
        rank_scores = rank_scores / rank_temperature
        score_diffs = rank_scores[None, :] - rank_scores[:, None]
        pair_valid = valid_bool[:, None] & valid_bool[None, :]
        eye = torch.eye(scores.shape[0], dtype=torch.bool, device=scores.device)
        higher_prob = torch.sigmoid(score_diffs).masked_fill(~pair_valid | eye, 0.0)
        soft_rank = 1.0 + higher_prob.sum(dim=1)
        membership = torch.sigmoid((float(topk_limit) + 0.5 - soft_rank) / rank_width)
        log_membership = torch.log(membership.clamp_min(torch.finfo(torch.float32).eps))
        soft_rank = soft_rank.masked_fill(~valid_bool, 0.0)
        log_membership = log_membership.masked_fill(~valid_bool, 0.0)
        _require_finite(soft_rank, f"{name} global rank cache soft ranks")
        _require_finite(log_membership, f"{name} global rank cache membership")
        return {"soft_rank": soft_rank, "log_membership": log_membership}

    def _scatter_candidate_distribution_to_dense(
        self,
        *,
        candidate_distribution: torch.Tensor,
        candidate_dense_indices: torch.Tensor,
        dense_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        soft_dense = torch.zeros((int(dense_len),), dtype=torch.float32, device=device)
        soft_dense.scatter_add_(0, candidate_dense_indices.to(device=device), candidate_distribution)
        _require_finite(soft_dense, "rank transport dense distribution")
        return soft_dense

    def _interval_boundary_packet_transport_plan(
        self,
        *,
        reader_outputs: Mapping[str, torch.Tensor],
        valid: torch.Tensor,
        candidate_valid: torch.Tensor,
        candidate_dense_indices: torch.Tensor,
        training: bool,
    ) -> dict[str, Any]:
        if tuple(candidate_dense_indices.shape) != tuple(candidate_valid.shape):
            raise ValueError("candidate_dense_indices must match candidate_valid")
        device = candidate_dense_indices.device
        candidate_valid = candidate_valid.to(device=device).bool()
        candidate_dense_indices = candidate_dense_indices.to(device=device)
        valid = valid.to(device=device).bool()
        if bool((candidate_valid.long().sum(dim=1) <= 0).any().item()):
            raise ValueError("each sample must contain at least one interval_boundary_packet candidate")

        action_scores, action_source = self._dense_head_or_zeros(
            reader_outputs=reader_outputs,
            candidate_valid=candidate_valid,
            names=("actionness_logits", "action_logits", "value_logits", "frame_selection_logits"),
            device=device,
        )
        start_scores, start_source = self._dense_head_or_zeros(
            reader_outputs=reader_outputs,
            candidate_valid=candidate_valid,
            names=("start_logits",),
            device=device,
        )
        end_scores, end_source = self._dense_head_or_zeros(
            reader_outputs=reader_outputs,
            candidate_valid=candidate_valid,
            names=("end_logits",),
            device=device,
        )
        boundary_scores, boundary_source = self._dense_head_or_zeros(
            reader_outputs=reader_outputs,
            candidate_valid=candidate_valid,
            names=("boundary_logits", "risk_logits"),
            device=device,
        )
        uncertainty_scores, uncertainty_source = self._dense_head_or_zeros(
            reader_outputs=reader_outputs,
            candidate_valid=candidate_valid,
            names=("uncertainty_logits",),
            device=device,
        )
        redundancy_scores, redundancy_source = self._dense_head_or_zeros(
            reader_outputs=reader_outputs,
            candidate_valid=candidate_valid,
            names=("redundancy_logits",),
            device=device,
        )
        source_heads = [
            action_source,
            start_source,
            end_source,
            boundary_source,
            uncertainty_source,
            redundancy_source,
        ]
        dense_len = int(valid.shape[1])
        batch = int(candidate_valid.shape[0])
        topk = 1
        fixed_indices = torch.empty((batch, self.target_len, topk), dtype=torch.long, device=device)
        fixed_weights = torch.ones((batch, self.target_len, topk), dtype=torch.float32, device=device)
        fixed_positions = torch.empty((batch, self.target_len), dtype=torch.float32, device=device)
        transport_weights = torch.zeros((batch, self.target_len, dense_len), dtype=torch.float32, device=device)
        selected_output_valid_lengths = torch.empty((batch,), dtype=torch.long, device=device)
        selected_roles: list[list[str]] = []
        raw_dense_indices: list[list[int]] = []
        raw_duplicate_rates: list[float] = []
        raw_unique_counts: list[int] = []
        reader_fill_counts: list[int] = []
        st_active_row_counts: list[int] = []
        interval_packet_metadata: list[dict[str, Any]] = []
        min_score = torch.finfo(torch.float32).min

        boundary_rank_scores = (
            start_scores + end_scores + boundary_scores + 0.25 * uncertainty_scores - 0.25 * redundancy_scores
        ).masked_fill(~candidate_valid, min_score)
        interior_rank_scores = (
            action_scores + 0.25 * boundary_scores + 0.25 * uncertainty_scores - 0.50 * redundancy_scores
        ).masked_fill(~candidate_valid, min_score)
        start_rank_scores = (
            start_scores + 0.50 * boundary_scores + 0.25 * uncertainty_scores - 0.25 * redundancy_scores
        ).masked_fill(~candidate_valid, min_score)
        end_rank_scores = (
            end_scores + 0.50 * boundary_scores + 0.25 * uncertainty_scores - 0.25 * redundancy_scores
        ).masked_fill(~candidate_valid, min_score)
        _require_finite(boundary_rank_scores, "interval boundary rank scores")
        _require_finite(interior_rank_scores, "interval interior rank scores")

        for batch_idx in range(batch):
            valid_candidate_indices = torch.nonzero(candidate_valid[batch_idx], as_tuple=False).flatten()
            output_valid_len = min(int(valid_candidate_indices.numel()), self.target_len)
            selected_output_valid_lengths[batch_idx] = output_valid_len
            if output_valid_len <= 0:
                raise ValueError("interval_boundary_packet found no valid candidates for a sample")
            if output_valid_len < self.target_len:
                batch_roles = []
                valid_positions = candidate_dense_indices[batch_idx][valid_candidate_indices]
                pad_pos = int(valid_positions[-1].item())
                for out_idx in range(self.target_len):
                    if out_idx < int(valid_positions.numel()):
                        pos = int(valid_positions[out_idx].item())
                        role = "valid_prefix"
                    else:
                        pos = pad_pos
                        role = "pad_repeat"
                    fixed_positions[batch_idx, out_idx] = float(pos)
                    fixed_indices[batch_idx, out_idx, 0] = pos
                    transport_weights[batch_idx, out_idx, pos] = 1.0
                    batch_roles.append(role)
                selected_roles.append(batch_roles)
                raw_positions = [int(pos) for pos in valid_positions.detach().cpu().tolist()]
                raw_dense_indices.append(raw_positions)
                raw_unique_counts.append(len(set(raw_positions)))
                raw_duplicate_rates.append(1.0 - float(len(set(raw_positions))) / float(max(1, len(raw_positions))))
                reader_fill_counts.append(0)
                st_active_row_counts.append(0)
                interval_packet_metadata.append(
                    {
                        "enabled": True,
                        "status": "short_valid_prefix",
                        "boundary_budget": 0,
                        "interior_budget": 0,
                        "boundary_positions": [],
                        "interior_positions": [],
                        "interval_candidate_topk": int(self.interval_candidate_topk),
                        "interval_pair_limit": int(valid_candidate_indices.numel()),
                        "interval_record_count": 0,
                        "interior_candidate_count": 0,
                        "source_heads": source_heads,
                    }
                )
                continue

            boundary_budget = int(round(float(output_valid_len) * float(self.interval_boundary_budget_ratio)))
            if output_valid_len >= 2:
                boundary_budget = max(2, boundary_budget)
            boundary_budget = min(output_valid_len, boundary_budget)
            interior_budget = max(0, output_valid_len - boundary_budget)

            start_ranked = torch.argsort(start_rank_scores[batch_idx], descending=True, stable=True)
            start_ranked = start_ranked[candidate_valid[batch_idx].gather(0, start_ranked)]
            end_ranked = torch.argsort(end_rank_scores[batch_idx], descending=True, stable=True)
            end_ranked = end_ranked[candidate_valid[batch_idx].gather(0, end_ranked)]
            pair_limit = min(
                int(valid_candidate_indices.numel()),
                int(self.interval_candidate_topk),
            )
            pair_limit = max(1, pair_limit)
            interval_records: list[dict[str, Any]] = []
            prefix_action = torch.cat(
                [
                    action_scores.new_zeros((1,)),
                    torch.cumsum(action_scores[batch_idx].masked_fill(~candidate_valid[batch_idx], 0.0), dim=0),
                ],
                dim=0,
            )
            prefix_redundancy = torch.cat(
                [
                    redundancy_scores.new_zeros((1,)),
                    torch.cumsum(redundancy_scores[batch_idx].masked_fill(~candidate_valid[batch_idx], 0.0), dim=0),
                ],
                dim=0,
            )
            for start_tensor in start_ranked[:pair_limit]:
                start_idx = int(start_tensor.item())
                for end_tensor in end_ranked[:pair_limit]:
                    end_idx = int(end_tensor.item())
                    if end_idx < start_idx:
                        continue
                    span_count = max(1, end_idx - start_idx + 1)
                    action_mean = (prefix_action[end_idx + 1] - prefix_action[start_idx]) / float(span_count)
                    redundancy_mean = (prefix_redundancy[end_idx + 1] - prefix_redundancy[start_idx]) / float(span_count)
                    score = (
                        start_scores[batch_idx, start_idx]
                        + end_scores[batch_idx, end_idx]
                        + 0.50 * (boundary_scores[batch_idx, start_idx] + boundary_scores[batch_idx, end_idx])
                        + action_mean
                        + 0.25
                        * (uncertainty_scores[batch_idx, start_idx] + uncertainty_scores[batch_idx, end_idx])
                        - 0.25
                        * (redundancy_scores[batch_idx, start_idx] + redundancy_scores[batch_idx, end_idx])
                        - 0.25 * redundancy_mean
                    )
                    interval_records.append(
                        {
                            "score": float(score.detach().cpu().item()),
                            "start_idx": start_idx,
                            "end_idx": end_idx,
                            "start_pos": int(candidate_dense_indices[batch_idx, start_idx].item()),
                            "end_pos": int(candidate_dense_indices[batch_idx, end_idx].item()),
                        }
                    )
            if not interval_records:
                first_idx = int(valid_candidate_indices[0].item())
                interval_records.append(
                    {
                        "score": 0.0,
                        "start_idx": first_idx,
                        "end_idx": first_idx,
                        "start_pos": int(candidate_dense_indices[batch_idx, first_idx].item()),
                        "end_pos": int(candidate_dense_indices[batch_idx, first_idx].item()),
                    }
                )
            interval_records.sort(key=lambda item: item["score"], reverse=True)

            used: set[int] = set()
            rows: list[tuple[int, int | None, str]] = []
            boundary_positions: list[int] = []
            interior_positions: list[int] = []

            def _try_add(candidate_idx: int | None, role: str) -> bool:
                if candidate_idx is None:
                    return False
                pos = int(candidate_dense_indices[batch_idx, int(candidate_idx)].item())
                if pos in used or not bool(valid[batch_idx, pos].item()):
                    return False
                used.add(pos)
                rows.append((pos, int(candidate_idx), role))
                if role == "boundary_packet":
                    boundary_positions.append(pos)
                elif role == "interior_action_packet":
                    interior_positions.append(pos)
                return True

            for record in interval_records:
                for candidate_idx in (int(record["start_idx"]), int(record["end_idx"])):
                    if len(boundary_positions) >= boundary_budget:
                        break
                    _try_add(candidate_idx, "boundary_packet")
                if len(boundary_positions) >= boundary_budget:
                    break
            if len(boundary_positions) < boundary_budget:
                ranked_boundary = torch.argsort(boundary_rank_scores[batch_idx], descending=True, stable=True)
                for candidate_tensor in ranked_boundary:
                    if not bool(candidate_valid[batch_idx, candidate_tensor].item()):
                        continue
                    if _try_add(int(candidate_tensor.item()), "boundary_packet") and len(boundary_positions) >= boundary_budget:
                        break

            interior_candidate_scores: dict[int, float] = {}
            for record in interval_records:
                for candidate_idx in range(int(record["start_idx"]), int(record["end_idx"]) + 1):
                    if not bool(candidate_valid[batch_idx, candidate_idx].item()):
                        continue
                    pos = int(candidate_dense_indices[batch_idx, candidate_idx].item())
                    if pos in used:
                        continue
                    score = float(interior_rank_scores[batch_idx, candidate_idx].detach().cpu().item())
                    previous = interior_candidate_scores.get(candidate_idx)
                    if previous is None or score > previous:
                        interior_candidate_scores[candidate_idx] = score
            interior_candidates = [
                (score, candidate_idx) for candidate_idx, score in interior_candidate_scores.items()
            ]
            interior_candidates.sort(key=lambda item: item[0], reverse=True)
            for _score, candidate_idx in interior_candidates:
                if len(interior_positions) >= interior_budget:
                    break
                _try_add(candidate_idx, "interior_action_packet")

            if len(interior_positions) < interior_budget:
                ranked_action = torch.argsort(interior_rank_scores[batch_idx], descending=True, stable=True)
                for candidate_tensor in ranked_action:
                    if not bool(candidate_valid[batch_idx, candidate_tensor].item()):
                        continue
                    if _try_add(int(candidate_tensor.item()), "interior_action_packet") and len(interior_positions) >= interior_budget:
                        break

            if len(rows) < output_valid_len:
                combined_scores = torch.maximum(boundary_rank_scores[batch_idx], interior_rank_scores[batch_idx])
                ranked_fill = torch.argsort(combined_scores, descending=True, stable=True)
                for candidate_tensor in ranked_fill:
                    if not bool(candidate_valid[batch_idx, candidate_tensor].item()):
                        continue
                    if _try_add(int(candidate_tensor.item()), "interval_rank_fill") and len(rows) >= output_valid_len:
                        break

            if len(rows) < output_valid_len:
                for candidate_tensor in valid_candidate_indices:
                    if _try_add(int(candidate_tensor.item()), "dense_fill") and len(rows) >= output_valid_len:
                        break

            if len(rows) != output_valid_len:
                raise ValueError(
                    "failed to resolve interval_boundary_packet plan: "
                    f"rows={len(rows)}, output_valid_len={output_valid_len}, target_len={self.target_len}"
                )

            raw_positions = [pos for pos, _candidate_idx, _role in rows]
            raw_dense_indices.append(raw_positions)
            raw_unique_count = len(set(raw_positions))
            raw_unique_counts.append(raw_unique_count)
            raw_duplicate_rates.append(1.0 - float(raw_unique_count) / float(max(1, len(raw_positions))))
            rows.sort(key=lambda item: item[0])
            batch_roles: list[str] = []
            active_st_rows = 0
            for out_idx in range(self.target_len):
                if out_idx < output_valid_len:
                    pos, candidate_idx, role = rows[out_idx]
                else:
                    pos = rows[-1][0]
                    candidate_idx = rows[-1][1]
                    role = "pad_repeat"
                fixed_positions[batch_idx, out_idx] = float(pos)
                fixed_indices[batch_idx, out_idx, 0] = pos
                fixed_weights[batch_idx, out_idx, 0] = 1.0
                batch_roles.append(role)
                hard = torch.zeros((dense_len,), dtype=torch.float32, device=device)
                hard[pos] = 1.0
                if (
                    self.straight_through_detector_loss
                    and training
                    and candidate_idx is not None
                    and role in ("boundary_packet", "interior_action_packet", "interval_rank_fill")
                ):
                    rank_scores = (
                        boundary_rank_scores[batch_idx]
                        if role == "boundary_packet"
                        else interior_rank_scores[batch_idx]
                    )
                    soft_candidate = self._rank_transport_candidate_distribution(
                        scores=rank_scores,
                        candidate_valid=candidate_valid[batch_idx],
                        candidate_dense_indices=candidate_dense_indices[batch_idx],
                        candidate_idx=int(candidate_idx),
                        name=f"interval_boundary_packet/{role}",
                    )
                    soft_dense = self._scatter_candidate_distribution_to_dense(
                        candidate_distribution=soft_candidate,
                        candidate_dense_indices=candidate_dense_indices[batch_idx],
                        dense_len=dense_len,
                        device=device,
                    )
                    transport_weights[batch_idx, out_idx] = hard + soft_dense - soft_dense.detach()
                    active_st_rows += 1
                else:
                    transport_weights[batch_idx, out_idx] = hard
            selected_roles.append(batch_roles)
            reader_fill_counts.append(sum(1 for _pos, _candidate_idx, role in rows if role == "interval_rank_fill"))
            st_active_row_counts.append(active_st_rows if training and self.straight_through_detector_loss else 0)
            interval_packet_metadata.append(
                {
                    "enabled": True,
                    "status": "interval_score_first",
                    "boundary_budget": int(boundary_budget),
                    "interior_budget": int(interior_budget),
                    "boundary_positions": sorted(int(pos) for pos in boundary_positions),
                    "interior_positions": sorted(int(pos) for pos in interior_positions),
                    "interval_candidate_topk": int(self.interval_candidate_topk),
                    "interval_pair_limit": int(pair_limit),
                    "interval_record_count": int(len(interval_records)),
                    "interior_candidate_count": int(len(interior_candidates)),
                    "top_intervals": [
                        {
                            "start": int(record["start_pos"]),
                            "end": int(record["end_pos"]),
                            "score": float(record["score"]),
                        }
                        for record in interval_records[: min(4, len(interval_records))]
                    ],
                    "source_heads": source_heads,
                    "uses_gt": False,
                    "uses_teacher": False,
                    "uses_raw_prediction_cache": False,
                    "uses_p2": False,
                }
            )

        _require_finite(fixed_weights, "interval_boundary_packet fixed weights")
        _require_finite(fixed_positions, "interval_boundary_packet selected positions")
        _require_finite(transport_weights, "interval_boundary_packet sparse transport weights")
        return {
            "indices": fixed_indices,
            "weights": fixed_weights,
            "transport_weights": transport_weights,
            "selected_positions": fixed_positions,
            "selected_output_valid_lengths": selected_output_valid_lengths,
            "selected_roles": selected_roles,
            "raw_slot_dense_indices": raw_dense_indices,
            "raw_slot_duplicate_rates": raw_duplicate_rates,
            "raw_slot_unique_counts": raw_unique_counts,
            "reader_fill_counts": reader_fill_counts,
            "st_active_row_counts": st_active_row_counts,
            "interval_packet_metadata": interval_packet_metadata,
        }

    def _frame_score_transport_plan(
        self,
        *,
        reader_outputs: Mapping[str, torch.Tensor],
        valid: torch.Tensor,
        candidate_valid: torch.Tensor,
        candidate_dense_indices: torch.Tensor,
        training: bool,
    ) -> dict[str, torch.Tensor]:
        strategy_name = getattr(self, "selection_strategy", "frame_score_topk")
        plan_name = "frame_score_global_rank_st" if strategy_name == "frame_score_global_rank_st" else "frame_score_topk"
        frame_scores = reader_outputs.get("frame_selection_logits")
        if frame_scores is None and plan_name == "frame_score_global_rank_st":
            raise ValueError("frame_score_global_rank_st selection requires frame_selection_logits")
        if frame_scores is None:
            frame_scores = reader_outputs.get("actionness_logits", reader_outputs.get("action_logits"))
        if frame_scores is None:
            raise ValueError(f"{plan_name} selection requires frame_selection_logits or actionness/action logits")
        if not torch.is_tensor(frame_scores):
            raise TypeError(f"{plan_name} frame scores must be a tensor")
        _require_finite(frame_scores, f"{plan_name} frame scores", error_type=ValueError)
        if tuple(candidate_dense_indices.shape) != tuple(candidate_valid.shape):
            raise ValueError("candidate_dense_indices must match candidate_valid")
        if tuple(frame_scores.shape) != tuple(candidate_valid.shape):
            raise ValueError(
                f"{plan_name} frame scores must match candidate axis; "
                f"got scores={tuple(frame_scores.shape)}, candidate_valid={tuple(candidate_valid.shape)}"
            )

        device = frame_scores.device
        frame_scores = frame_scores.float()
        candidate_valid = candidate_valid.to(device=device).bool()
        candidate_dense_indices = candidate_dense_indices.to(device=device)
        valid = valid.to(device=device).bool()
        if bool((candidate_valid.long().sum(dim=1) <= 0).any().item()):
            raise ValueError(f"each sample must contain at least one valid {plan_name} candidate")

        batch, candidate_len = frame_scores.shape
        dense_len = int(valid.shape[1])
        topk = 1
        fixed_indices = torch.empty((batch, self.target_len, topk), dtype=torch.long, device=device)
        fixed_weights = torch.ones((batch, self.target_len, topk), dtype=torch.float32, device=device)
        fixed_positions = torch.empty((batch, self.target_len), dtype=torch.float32, device=device)
        transport_weights = torch.zeros((batch, self.target_len, dense_len), dtype=torch.float32, device=device)
        selected_output_valid_lengths = torch.empty((batch,), dtype=torch.long, device=device)
        selected_roles: list[list[str]] = []
        raw_dense_indices: list[list[int]] = []
        raw_duplicate_rates: list[float] = []
        raw_unique_counts: list[int] = []
        reader_fill_counts: list[int] = []
        st_active_row_counts: list[int] = []
        max_gap_guard_meta: list[dict[str, Any]] = []

        min_score = torch.finfo(torch.float32).min
        masked_scores = frame_scores.masked_fill(~candidate_valid, min_score)
        _require_finite(masked_scores, f"{plan_name} masked scores")
        dynamic_budget_plan = self._dynamic_budget_plan(
            reader_outputs=reader_outputs,
            frame_scores=frame_scores,
            candidate_valid=candidate_valid,
        )
        dynamic_budgets = dynamic_budget_plan["budgets"] if dynamic_budget_plan is not None else None
        dynamic_budget_meta = dynamic_budget_plan["metadata"] if dynamic_budget_plan is not None else None

        for batch_idx in range(batch):
            valid_candidate_indices = torch.nonzero(candidate_valid[batch_idx], as_tuple=False).flatten()
            configured_budget = (
                int(dynamic_budgets[batch_idx].item()) if torch.is_tensor(dynamic_budgets) else self.target_len
            )
            output_valid_len = min(int(valid_candidate_indices.numel()), configured_budget, self.target_len)
            selected_output_valid_lengths[batch_idx] = output_valid_len
            if output_valid_len <= 0:
                raise ValueError(f"{plan_name} found no valid candidates for a sample")

            ranked_candidate_indices = torch.argsort(masked_scores[batch_idx], descending=True, stable=True)
            ranked_candidate_indices = ranked_candidate_indices[
                candidate_valid[batch_idx].gather(0, ranked_candidate_indices)
            ]
            rank_position_by_candidate = {
                int(candidate.item()): rank
                for rank, candidate in enumerate(ranked_candidate_indices[:output_valid_len])
            }
            raw_topk_positions = [
                int(pos)
                for pos in candidate_dense_indices[batch_idx]
                .gather(0, ranked_candidate_indices[:output_valid_len])
                .detach()
                .cpu()
                .tolist()
            ]
            raw_dense_indices.append(raw_topk_positions)
            raw_unique_count = len(set(raw_topk_positions))
            raw_unique_counts.append(raw_unique_count)
            raw_duplicate_rates.append(
                1.0 - float(raw_unique_count) / float(max(1, len(raw_topk_positions)))
            )

            valid_positions = candidate_dense_indices[batch_idx][candidate_valid[batch_idx]]
            rows: list[tuple[int, int | None, str]] = []
            used: set[int] = set()
            guard_enabled = bool(getattr(self, "max_dense_gap", 0) > 0 and getattr(self, "max_gap_guard_count", 0) > 0)
            max_gap_positions = self._max_gap_guard_positions(
                valid_positions=valid_positions,
                count=min(int(getattr(self, "max_gap_guard_count", 0)), output_valid_len),
                max_gap=int(getattr(self, "max_dense_gap", 0)),
            )
            for pos_tensor in max_gap_positions:
                pos = int(pos_tensor.item())
                if pos in used or not bool(valid[batch_idx, pos].item()):
                    continue
                used.add(pos)
                rows.append((pos, None, "max_gap_guard"))
                if len(rows) >= output_valid_len:
                    break

            if len(rows) < output_valid_len:
                for candidate_tensor in ranked_candidate_indices:
                    candidate_idx = int(candidate_tensor.item())
                    pos = int(candidate_dense_indices[batch_idx, candidate_idx].item())
                    if pos in used or not bool(valid[batch_idx, pos].item()):
                        continue
                    used.add(pos)
                    rows.append((pos, candidate_idx, plan_name))
                    if len(rows) >= output_valid_len:
                        break

            if len(rows) < output_valid_len:
                for candidate_tensor in valid_candidate_indices:
                    candidate_idx = int(candidate_tensor.item())
                    pos = int(candidate_dense_indices[batch_idx, candidate_idx].item())
                    if pos in used or not bool(valid[batch_idx, pos].item()):
                        continue
                    used.add(pos)
                    rows.append((pos, None, "dense_fill"))
                    if len(rows) >= output_valid_len:
                        break

            if len(rows) != output_valid_len:
                raise ValueError(
                    "failed to resolve frame_score_topk dynamic budget plan: "
                    f"rows={len(rows)}, output_valid_len={output_valid_len}, target_len={self.target_len}"
                )
            rows.sort(key=lambda item: item[0])
            prefix_positions = [pos for pos, _candidate_idx, _role in rows]
            prefix_gaps = [right - left for left, right in zip(prefix_positions[:-1], prefix_positions[1:])]
            max_gap_guard_meta.append(
                {
                    "enabled": guard_enabled,
                    "max_dense_gap": int(getattr(self, "max_dense_gap", 0)),
                    "max_gap_guard_count": int(getattr(self, "max_gap_guard_count", 0)),
                    "applied_count": sum(1 for _pos, _candidate_idx, role in rows if role == "max_gap_guard"),
                    "max_gap_after": max(prefix_gaps) if prefix_gaps else 0,
                    "safety_gate_only": True,
                }
            )

            batch_roles: list[str] = []
            global_rank_cache = None
            if (
                self.straight_through_detector_loss
                and training
                and getattr(self, "frame_score_st_surrogate", "local_softmax") == "global_rank_topk"
            ):
                global_rank_cache = self._global_rank_topk_cache(
                    scores=frame_scores[batch_idx],
                    candidate_valid=candidate_valid[batch_idx],
                    topk_budget=output_valid_len,
                    name=plan_name,
                )
            for out_idx in range(self.target_len):
                if out_idx < output_valid_len:
                    pos, candidate_idx, role = rows[out_idx]
                else:
                    pos = rows[-1][0]
                    candidate_idx = rows[-1][1]
                    role = "pad_repeat"

                fixed_positions[batch_idx, out_idx] = float(pos)
                fixed_indices[batch_idx, out_idx, 0] = pos
                fixed_weights[batch_idx, out_idx, 0] = 1.0
                batch_roles.append(role)

                hard = torch.zeros((dense_len,), dtype=torch.float32, device=device)
                hard[pos] = 1.0
                if (
                    self.straight_through_detector_loss
                    and training
                    and role in ("frame_score_topk", "frame_score_global_rank_st")
                    and candidate_idx is not None
                ):
                    soft_candidate = self._rank_transport_candidate_distribution(
                        scores=frame_scores[batch_idx],
                        candidate_valid=candidate_valid[batch_idx],
                        candidate_dense_indices=candidate_dense_indices[batch_idx],
                        candidate_idx=int(candidate_idx),
                        hard_rank_position=rank_position_by_candidate.get(int(candidate_idx)),
                        topk_budget=output_valid_len,
                        global_rank_cache=global_rank_cache,
                        name=plan_name,
                    )
                    soft_dense = self._scatter_candidate_distribution_to_dense(
                        candidate_distribution=soft_candidate,
                        candidate_dense_indices=candidate_dense_indices[batch_idx],
                        dense_len=dense_len,
                        device=device,
                    )
                    st_scale = float(getattr(self, "frame_score_st_gradient_scale", 1.0))
                    transport_weights[batch_idx, out_idx] = hard + st_scale * (soft_dense - soft_dense.detach())
                else:
                    transport_weights[batch_idx, out_idx] = hard
            selected_roles.append(batch_roles)
            reader_fill_counts.append(0)
            st_active_row_counts.append(
                sum(
                    1
                    for _pos, candidate_idx, role in rows
                    if training and role in ("frame_score_topk", "frame_score_global_rank_st") and candidate_idx is not None
                )
                if self.straight_through_detector_loss
                else 0
            )

        _require_finite(fixed_weights, f"{plan_name} fixed weights")
        _require_finite(fixed_positions, f"{plan_name} selected positions")
        _require_finite(transport_weights, f"{plan_name} sparse transport weights")
        return {
            "indices": fixed_indices,
            "weights": fixed_weights,
            "transport_weights": transport_weights,
            "selected_positions": fixed_positions,
            "selected_output_valid_lengths": selected_output_valid_lengths,
            "selected_roles": selected_roles,
            "raw_slot_dense_indices": raw_dense_indices,
            "raw_slot_duplicate_rates": raw_duplicate_rates,
            "raw_slot_unique_counts": raw_unique_counts,
            "reader_fill_counts": reader_fill_counts,
            "st_active_row_counts": st_active_row_counts,
            "dynamic_budget_meta": dynamic_budget_meta,
            "max_gap_guard_meta": max_gap_guard_meta,
        }

    def _coarse_actionness_scores(
        self,
        *,
        reader_outputs: Mapping[str, torch.Tensor],
        candidate_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        action_logits = reader_outputs.get("actionness_logits", reader_outputs.get("action_logits"))
        if not torch.is_tensor(action_logits):
            raise ValueError("coarse_actionness_uncertainty requires actionness_logits or action_logits")
        if tuple(action_logits.shape) != tuple(candidate_valid.shape):
            raise ValueError("coarse actionness logits must match candidate_valid shape")
        _require_finite(action_logits, "coarse actionness logits", error_type=ValueError)

        valid = candidate_valid.to(device=action_logits.device).bool()
        prob = torch.sigmoid(action_logits.float()).masked_fill(~valid, 0.0)
        eps = 1.0e-6
        prob_safe = prob.clamp(min=eps, max=1.0 - eps)
        entropy = (
            -(prob_safe * prob_safe.log() + (1.0 - prob_safe) * (1.0 - prob_safe).log()) / math.log(2.0)
        ).masked_fill(~valid, 0.0)
        margin_uncertainty = (1.0 - (2.0 * prob_safe - 1.0).abs()).clamp(0.0, 1.0).masked_fill(~valid, 0.0)
        uncertainty = torch.maximum(entropy, margin_uncertainty).clamp(0.0, 1.0).masked_fill(~valid, 0.0)

        adjacent_valid = valid[:, 1:] & valid[:, :-1]
        prev_change = torch.zeros_like(prob)
        prev_change[:, 1:] = (prob[:, 1:] - prob[:, :-1]).abs().masked_fill(~adjacent_valid, 0.0)
        next_change = torch.zeros_like(prob)
        next_change[:, :-1] = (prob[:, 1:] - prob[:, :-1]).abs().masked_fill(~adjacent_valid, 0.0)
        change = torch.maximum(prev_change, next_change).clamp(0.0, 1.0).masked_fill(~valid, 0.0)
        background = ((1.0 - prob) * (1.0 - uncertainty)).masked_fill(~valid, 0.0)

        normalizer = self.coarse_action_weight + self.coarse_uncertainty_weight + self.coarse_change_weight
        if normalizer <= 0.0:
            raise ValueError("coarse actionness policy requires at least one positive score weight")
        mixed = (
            self.coarse_action_weight * prob
            + self.coarse_uncertainty_weight * uncertainty
            + self.coarse_change_weight * change
        ) / normalizer
        mixed = mixed.masked_fill(~valid, 0.0)
        scores = {
            "p_action": prob,
            "entropy": entropy,
            "margin": margin_uncertainty,
            "p_change": change,
            "coarse_action": prob,
            "coarse_uncertainty": uncertainty,
            "coarse_change": change,
            "coarse_background": background,
            "coarse_mixed_fill": mixed,
        }
        for name, value in scores.items():
            _require_finite(value, f"{name} score", error_type=ValueError)
        return scores

    @staticmethod
    def _coarse_candidate_eligible_roles(
        *,
        candidate_idx: int,
        ranked_by_score: Mapping[str, torch.Tensor],
        quota: Mapping[str, int],
    ) -> list[str]:
        eligible: list[str] = []
        for role in (
            "coarse_action",
            "coarse_uncertainty",
            "coarse_change",
            "coarse_background",
            "coarse_mixed_fill",
        ):
            count = max(0, int(quota.get(role, 0)))
            if count <= 0:
                continue
            ranked = ranked_by_score.get(role)
            if ranked is None:
                continue
            top = ranked[:count].detach().cpu().tolist()
            if int(candidate_idx) in {int(item) for item in top}:
                eligible.append(role)
        return eligible

    def _coarse_candidate_points_for_batch(
        self,
        *,
        scores: Mapping[str, torch.Tensor],
        candidate_valid: torch.Tensor,
        candidate_dense_indices: torch.Tensor,
        batch_idx: int,
        ranked_by_score: Mapping[str, torch.Tensor],
        quota: Mapping[str, int],
        final_role_by_candidate: Mapping[int, str],
        source_score_role_by_candidate: Mapping[int, str | None],
    ) -> list[dict[str, Any]]:
        valid_mask = candidate_valid[batch_idx].detach().bool()
        dense_indices = candidate_dense_indices[batch_idx].detach()
        points: list[dict[str, Any]] = []
        component_keys = (
            "p_action",
            "entropy",
            "p_change",
            "margin",
            "coarse_uncertainty",
            "coarse_background",
            "coarse_mixed_fill",
        )
        for candidate_idx in range(int(valid_mask.numel())):
            is_valid = bool(valid_mask[candidate_idx].item())
            components = {}
            for key in component_keys:
                value = scores[key][batch_idx, candidate_idx].detach().cpu().item()
                components[key] = float(value)
            points.append(
                {
                    "candidate_idx": int(candidate_idx),
                    "dense_index": int(dense_indices[candidate_idx].detach().cpu().item()),
                    "valid": is_valid,
                    "final_role": final_role_by_candidate.get(candidate_idx),
                    "source_score_role": source_score_role_by_candidate.get(candidate_idx),
                    "eligible_roles": self._coarse_candidate_eligible_roles(
                        candidate_idx=candidate_idx,
                        ranked_by_score=ranked_by_score,
                        quota=quota,
                    )
                    if is_valid
                    else [],
                    "components": components,
                }
            )
        return points

    @staticmethod
    def _scaled_coarse_quota(configured: Mapping[str, int], budget: int) -> dict[str, int]:
        budget = int(budget)
        if budget <= 0:
            return {key: 0 for key in configured}
        clean = {key: max(0, int(value)) for key, value in configured.items()}
        total = sum(clean.values())
        if total <= budget:
            return clean
        raw = {key: (float(value) * float(budget) / float(total)) for key, value in clean.items()}
        quota = {key: int(math.floor(value)) for key, value in raw.items()}
        remainder = budget - sum(quota.values())
        order = sorted(raw, key=lambda key: (raw[key] - quota[key], clean[key]), reverse=True)
        for key in order[:remainder]:
            quota[key] += 1
        return quota

    def _coarse_actionness_uncertainty_transport_plan(
        self,
        *,
        reader_outputs: Mapping[str, torch.Tensor],
        valid: torch.Tensor,
        candidate_valid: torch.Tensor,
        candidate_dense_indices: torch.Tensor,
        training: bool,
    ) -> dict[str, Any]:
        if tuple(candidate_dense_indices.shape) != tuple(candidate_valid.shape):
            raise ValueError("candidate_dense_indices must match candidate_valid")
        scores = self._coarse_actionness_scores(
            reader_outputs=reader_outputs,
            candidate_valid=candidate_valid,
        )
        action_scores = scores["coarse_action"]
        device = action_scores.device
        candidate_valid = candidate_valid.to(device=device).bool()
        candidate_dense_indices = candidate_dense_indices.to(device=device)
        valid = valid.to(device=device).bool()
        if bool((candidate_valid.long().sum(dim=1) <= 0).any().item()):
            raise ValueError("each sample must contain at least one coarse actionness candidate")

        batch, _candidate_len = action_scores.shape
        dense_len = int(valid.shape[1])
        topk = 1
        fixed_indices = torch.empty((batch, self.target_len, topk), dtype=torch.long, device=device)
        fixed_weights = torch.ones((batch, self.target_len, topk), dtype=torch.float32, device=device)
        fixed_positions = torch.empty((batch, self.target_len), dtype=torch.float32, device=device)
        transport_weights = torch.zeros((batch, self.target_len, dense_len), dtype=torch.float32, device=device)
        selected_output_valid_lengths = torch.empty((batch,), dtype=torch.long, device=device)
        selected_roles: list[list[str]] = []
        raw_dense_indices: list[list[int]] = []
        raw_duplicate_rates: list[float] = []
        raw_unique_counts: list[int] = []
        reader_fill_counts: list[int] = []
        st_active_row_counts: list[int] = []
        max_gap_guard_meta: list[dict[str, Any]] = []
        coarse_policy_meta: list[dict[str, Any]] = []

        configured_quota = {
            "coarse_uniform": self.coarse_uniform_count,
            "coarse_action": self.coarse_action_count,
            "coarse_uncertainty": self.coarse_uncertainty_count,
            "coarse_change": self.coarse_change_count,
            "coarse_background": self.coarse_background_count,
        }
        min_score = torch.finfo(torch.float32).min
        mixed_ranked_all = torch.argsort(
            scores["coarse_mixed_fill"].masked_fill(~candidate_valid, min_score),
            dim=1,
            descending=True,
            stable=True,
        )

        for batch_idx in range(batch):
            valid_candidate_indices = torch.nonzero(candidate_valid[batch_idx], as_tuple=False).flatten()
            output_valid_len = min(int(valid_candidate_indices.numel()), self.target_len)
            selected_output_valid_lengths[batch_idx] = output_valid_len
            if output_valid_len <= 0:
                raise ValueError("coarse actionness policy found no valid candidates for a sample")

            valid_positions = candidate_dense_indices[batch_idx][candidate_valid[batch_idx]]
            pos_to_candidate: dict[int, int] = {}
            for candidate_tensor in valid_candidate_indices:
                candidate_idx = int(candidate_tensor.item())
                pos = int(candidate_dense_indices[batch_idx, candidate_idx].item())
                pos_to_candidate.setdefault(pos, candidate_idx)

            quota = self._scaled_coarse_quota(configured_quota, output_valid_len)
            rows: list[tuple[int, int | None, str, str | None]] = []
            used: set[int] = set()

            def add_pos(pos: int, candidate_idx: int | None, role: str, score_key: str | None) -> None:
                if len(rows) >= output_valid_len:
                    return
                if pos in used or not bool(valid[batch_idx, pos].item()):
                    return
                used.add(pos)
                rows.append((pos, candidate_idx, role, score_key))

            for pos_tensor in self._uniform_anchor_positions(
                valid_positions=valid_positions,
                count=quota["coarse_uniform"],
            ):
                pos = int(pos_tensor.item())
                add_pos(pos, pos_to_candidate.get(pos), "coarse_uniform", None)

            guard_enabled = bool(getattr(self, "max_dense_gap", 0) > 0 and getattr(self, "max_gap_guard_count", 0) > 0)
            for pos_tensor in self._max_gap_guard_positions(
                valid_positions=valid_positions,
                count=min(int(getattr(self, "max_gap_guard_count", 0)), output_valid_len),
                max_gap=int(getattr(self, "max_dense_gap", 0)),
            ):
                pos = int(pos_tensor.item())
                add_pos(pos, pos_to_candidate.get(pos), "coarse_max_gap_guard", None)

            ranked_by_score: dict[str, torch.Tensor] = {}
            rank_position_by_score: dict[str, dict[int, int]] = {}
            for score_key in (
                "coarse_action",
                "coarse_uncertainty",
                "coarse_change",
                "coarse_background",
                "coarse_mixed_fill",
            ):
                ranked = torch.argsort(
                    scores[score_key][batch_idx].masked_fill(~candidate_valid[batch_idx], min_score),
                    descending=True,
                    stable=True,
                )
                ranked = ranked[candidate_valid[batch_idx].gather(0, ranked)]
                ranked_by_score[score_key] = ranked
                rank_position_by_score[score_key] = {
                    int(candidate.item()): rank for rank, candidate in enumerate(ranked)
                }

            for score_key, count in (
                ("coarse_action", quota["coarse_action"]),
                ("coarse_uncertainty", quota["coarse_uncertainty"]),
                ("coarse_change", quota["coarse_change"]),
                ("coarse_background", quota["coarse_background"]),
            ):
                for candidate_tensor in ranked_by_score[score_key]:
                    if sum(1 for _pos, _candidate_idx, role, _score_key in rows if role == score_key) >= count:
                        break
                    candidate_idx = int(candidate_tensor.item())
                    pos = int(candidate_dense_indices[batch_idx, candidate_idx].item())
                    add_pos(pos, candidate_idx, score_key, score_key)
                    if len(rows) >= output_valid_len:
                        break

            for candidate_tensor in ranked_by_score["coarse_mixed_fill"]:
                if len(rows) >= output_valid_len:
                    break
                candidate_idx = int(candidate_tensor.item())
                pos = int(candidate_dense_indices[batch_idx, candidate_idx].item())
                add_pos(pos, candidate_idx, "coarse_mixed_fill", "coarse_mixed_fill")

            if len(rows) < output_valid_len:
                for pos_tensor in valid_positions:
                    if len(rows) >= output_valid_len:
                        break
                    pos = int(pos_tensor.item())
                    add_pos(pos, pos_to_candidate.get(pos), "coarse_mixed_fill", "coarse_mixed_fill")

            if len(rows) != output_valid_len:
                raise ValueError(
                    "failed to resolve coarse actionness uncertainty plan: "
                    f"rows={len(rows)}, output_valid_len={output_valid_len}, target_len={self.target_len}"
                )

            rows.sort(key=lambda item: item[0])
            prefix_positions = [pos for pos, _candidate_idx, _role, _score_key in rows]
            prefix_gaps = [right - left for left, right in zip(prefix_positions[:-1], prefix_positions[1:])]
            raw_top_positions = [
                int(candidate_dense_indices[batch_idx, int(candidate.item())].item())
                for candidate in mixed_ranked_all[batch_idx][:output_valid_len]
            ]
            raw_dense_indices.append(raw_top_positions)
            raw_unique_count = len(set(raw_top_positions))
            raw_unique_counts.append(raw_unique_count)
            raw_duplicate_rates.append(
                1.0 - float(raw_unique_count) / float(max(1, len(raw_top_positions)))
            )
            role_counts: dict[str, int] = {}
            for _pos, _candidate_idx, role, _score_key in rows:
                role_counts[role] = role_counts.get(role, 0) + 1
            final_role_by_candidate: dict[int, str] = {}
            source_score_role_by_candidate: dict[int, str | None] = {}
            for _pos, candidate_idx, role, score_key in rows:
                if candidate_idx is None:
                    continue
                final_role_by_candidate[int(candidate_idx)] = role
                source_score_role_by_candidate[int(candidate_idx)] = score_key
            candidate_role_quota = dict(quota)
            candidate_role_quota["coarse_mixed_fill"] = sum(
                1 for _pos, _candidate_idx, role, _score_key in rows if role == "coarse_mixed_fill"
            )
            candidate_points = self._coarse_candidate_points_for_batch(
                scores=scores,
                candidate_valid=candidate_valid,
                candidate_dense_indices=candidate_dense_indices,
                batch_idx=batch_idx,
                ranked_by_score=ranked_by_score,
                quota=candidate_role_quota,
                final_role_by_candidate=final_role_by_candidate,
                source_score_role_by_candidate=source_score_role_by_candidate,
            )
            max_gap_guard_meta.append(
                {
                    "enabled": guard_enabled,
                    "max_dense_gap": int(getattr(self, "max_dense_gap", 0)),
                    "max_gap_guard_count": int(getattr(self, "max_gap_guard_count", 0)),
                    "applied_count": role_counts.get("coarse_max_gap_guard", 0),
                    "max_gap_after": max(prefix_gaps) if prefix_gaps else 0,
                    "safety_gate_only": True,
                }
            )
            coarse_policy_meta.append(
                {
                    "enabled": True,
                    "protocol": "binary_actionness_uncertainty_v0",
                    "configured_quota": dict(configured_quota),
                    "effective_quota": dict(quota),
                    "role_counts": role_counts,
                    "score_weights": {
                        "action": float(self.coarse_action_weight),
                        "uncertainty": float(self.coarse_uncertainty_weight),
                        "change": float(self.coarse_change_weight),
                    },
                    "candidate_points": candidate_points,
                    "mixed_fill_role": "coarse_mixed_fill",
                    "dense_fill_compat_count": 0,
                    "deploy_time_signals": [
                        "actionness_logits",
                        "p_action",
                        "binary_entropy",
                        "binary_margin_uncertainty",
                        "temporal_probability_change",
                    ],
                    "uses_learned_boundary_head": False,
                    "uses_gt": False,
                    "uses_teacher": False,
                    "uses_raw_prediction_cache": False,
                    "uses_p2": False,
                }
            )

            batch_roles: list[str] = []
            st_count = 0
            global_rank_cache_by_key: dict[str, Mapping[str, torch.Tensor]] = {}
            for out_idx in range(self.target_len):
                if out_idx < output_valid_len:
                    pos, candidate_idx, role, score_key = rows[out_idx]
                else:
                    pos, candidate_idx, role, score_key = rows[-1][0], rows[-1][1], "pad_repeat", None

                fixed_positions[batch_idx, out_idx] = float(pos)
                fixed_indices[batch_idx, out_idx, 0] = pos
                fixed_weights[batch_idx, out_idx, 0] = 1.0
                batch_roles.append(role)

                hard = torch.zeros((dense_len,), dtype=torch.float32, device=device)
                hard[pos] = 1.0
                if (
                    self.straight_through_detector_loss
                    and training
                    and score_key is not None
                    and candidate_idx is not None
                ):
                    if (
                        getattr(self, "frame_score_st_surrogate", "local_softmax") == "global_rank_topk"
                        and score_key not in global_rank_cache_by_key
                    ):
                        global_rank_cache_by_key[score_key] = self._global_rank_topk_cache(
                            scores=scores[score_key][batch_idx],
                            candidate_valid=candidate_valid[batch_idx],
                            topk_budget=output_valid_len,
                            name=f"coarse_actionness_uncertainty/{score_key}",
                        )
                    soft_candidate = self._rank_transport_candidate_distribution(
                        scores=scores[score_key][batch_idx],
                        candidate_valid=candidate_valid[batch_idx],
                        candidate_dense_indices=candidate_dense_indices[batch_idx],
                        candidate_idx=int(candidate_idx),
                        hard_rank_position=rank_position_by_score[score_key].get(int(candidate_idx)),
                        topk_budget=output_valid_len,
                        global_rank_cache=global_rank_cache_by_key.get(score_key),
                        name=f"coarse_actionness_uncertainty/{score_key}",
                    )
                    soft_dense = self._scatter_candidate_distribution_to_dense(
                        candidate_distribution=soft_candidate,
                        candidate_dense_indices=candidate_dense_indices[batch_idx],
                        dense_len=dense_len,
                        device=device,
                    )
                    st_scale = float(getattr(self, "frame_score_st_gradient_scale", 1.0))
                    transport_weights[batch_idx, out_idx] = hard + st_scale * (soft_dense - soft_dense.detach())
                    st_count += 1
                else:
                    transport_weights[batch_idx, out_idx] = hard
            selected_roles.append(batch_roles)
            reader_fill_counts.append(role_counts.get("coarse_mixed_fill", 0) + role_counts.get("dense_fill", 0))
            st_active_row_counts.append(st_count if self.straight_through_detector_loss else 0)

        _require_finite(fixed_weights, "coarse actionness fixed weights")
        _require_finite(fixed_positions, "coarse actionness selected positions")
        _require_finite(transport_weights, "coarse actionness sparse transport weights")
        return {
            "indices": fixed_indices,
            "weights": fixed_weights,
            "transport_weights": transport_weights,
            "selected_positions": fixed_positions,
            "selected_output_valid_lengths": selected_output_valid_lengths,
            "selected_roles": selected_roles,
            "raw_slot_dense_indices": raw_dense_indices,
            "raw_slot_duplicate_rates": raw_duplicate_rates,
            "raw_slot_unique_counts": raw_unique_counts,
            "reader_fill_counts": reader_fill_counts,
            "st_active_row_counts": st_active_row_counts,
            "max_gap_guard_meta": max_gap_guard_meta,
            "coarse_policy_meta": coarse_policy_meta,
        }

    def _normalize_dynamic_budget_config(self, cfg: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if cfg is None:
            return None
        cfg = dict(cfg)
        if not bool(cfg.get("enabled", False)):
            return None
        protocol = str(cfg.get("protocol", "marginal_utility_v0"))
        if protocol != "marginal_utility_v0":
            raise ValueError("dynamic_budget.protocol must be 'marginal_utility_v0'")
        min_budget = int(cfg.get("min_budget", cfg.get("min", self.target_len)))
        target_budget = int(cfg.get("target_budget", cfg.get("target", self.target_len)))
        max_budget = int(cfg.get("max_budget", cfg.get("max", self.target_len)))
        average_budget = int(cfg.get("average_budget", cfg.get("average", target_budget)))
        budget_step = int(cfg.get("budget_step", 1))
        if budget_step <= 0:
            raise ValueError("dynamic_budget.budget_step must be positive")
        if not (0 < min_budget <= target_budget <= max_budget <= self.target_len):
            raise ValueError("dynamic_budget requires 0 < min <= target <= max <= target_len")
        if not (min_budget <= average_budget <= max_budget):
            raise ValueError("dynamic_budget.average_budget must be inside [min_budget, max_budget]")
        midpoint = float(cfg.get("score_midpoint", 0.5))
        if not (0.0 < midpoint < 1.0):
            raise ValueError("dynamic_budget.score_midpoint must lie inside (0, 1)")
        return {
            "enabled": True,
            "protocol": protocol,
            "min_budget": min_budget,
            "target_budget": target_budget,
            "max_budget": max_budget,
            "average_budget": average_budget,
            "budget_step": budget_step,
            "score_midpoint": midpoint,
            "actionness_weight": float(cfg.get("actionness_weight", 1.0)),
            "boundary_weight": float(cfg.get("boundary_weight", 0.35)),
            "uncertainty_weight": float(cfg.get("uncertainty_weight", 0.20)),
            "redundancy_weight": float(cfg.get("redundancy_weight", 0.35)),
            "valid_len_weight": float(cfg.get("valid_len_weight", 0.0)),
        }

    def _dynamic_budget_plan(
        self,
        *,
        reader_outputs: Mapping[str, torch.Tensor],
        frame_scores: torch.Tensor,
        candidate_valid: torch.Tensor,
    ) -> dict[str, Any] | None:
        cfg = getattr(self, "dynamic_budget", None)
        if not cfg:
            return None
        for key in reader_outputs.keys():
            key_text = str(key).lower()
            if any(token in key_text for token in ("gt", "teacher", "cache", "raw_prediction", "oracle", "target")):
                raise ValueError(f"dynamic_budget received forbidden deploy-time payload: {key}")

        action_score = self._dynamic_budget_head_mean(
            reader_outputs=reader_outputs,
            candidate_valid=candidate_valid,
            names=("frame_selection_logits", "actionness_logits", "action_logits", "value_logits"),
            fallback=frame_scores,
        )
        boundary_score = self._dynamic_budget_head_mean(
            reader_outputs=reader_outputs,
            candidate_valid=candidate_valid,
            names=("boundary_logits", "start_logits", "end_logits", "risk_logits"),
            fallback=None,
        )
        uncertainty_score = self._dynamic_budget_head_mean(
            reader_outputs=reader_outputs,
            candidate_valid=candidate_valid,
            names=("uncertainty_logits",),
            fallback=None,
        )
        redundancy_score = self._dynamic_budget_head_mean(
            reader_outputs=reader_outputs,
            candidate_valid=candidate_valid,
            names=("redundancy_logits",),
            fallback=None,
        )
        valid_len_score = candidate_valid.float().mean(dim=1)
        utility = (
            float(cfg["actionness_weight"]) * action_score
            + float(cfg["boundary_weight"]) * boundary_score
            + float(cfg["uncertainty_weight"]) * uncertainty_score
            - float(cfg["redundancy_weight"]) * redundancy_score
            + float(cfg["valid_len_weight"]) * valid_len_score
        )
        normalizer = (
            abs(float(cfg["actionness_weight"]))
            + abs(float(cfg["boundary_weight"]))
            + abs(float(cfg["uncertainty_weight"]))
            + abs(float(cfg["redundancy_weight"]))
            + abs(float(cfg["valid_len_weight"]))
        )
        if normalizer <= 0.0:
            raise ValueError("dynamic_budget requires at least one non-zero utility weight")
        utility_score = (utility / normalizer).clamp(0.0, 1.0)

        min_budget = int(cfg["min_budget"])
        target_budget = int(cfg["target_budget"])
        max_budget = int(cfg["max_budget"])
        midpoint = float(cfg["score_midpoint"])
        below = utility_score < midpoint
        lower_span = max(1, target_budget - min_budget)
        upper_span = max(1, max_budget - target_budget)
        lower_ratio = (utility_score / midpoint).clamp(0.0, 1.0)
        upper_ratio = ((utility_score - midpoint) / (1.0 - midpoint)).clamp(0.0, 1.0)
        budget_float = torch.where(
            below,
            float(min_budget) + lower_ratio * float(lower_span),
            float(target_budget) + upper_ratio * float(upper_span),
        )
        step = int(cfg["budget_step"])
        budgets = torch.round(budget_float / float(step)).to(dtype=torch.long) * step
        budgets = budgets.clamp(min=min_budget, max=max_budget)
        valid_counts = candidate_valid.long().sum(dim=1)
        budgets = torch.minimum(budgets.to(device=valid_counts.device), valid_counts)

        metadata: list[dict[str, Any]] = []
        for batch_idx, budget in enumerate(budgets.detach().cpu().tolist()):
            metadata.append(
                {
                    "enabled": True,
                    "protocol": str(cfg["protocol"]),
                    "budget": int(budget),
                    "min_budget": min_budget,
                    "target_budget": target_budget,
                    "max_budget": max_budget,
                    "average_budget": int(cfg["average_budget"]),
                    "budget_step": step,
                    "utility_score": float(utility_score[batch_idx].detach().cpu().item()),
                    "actionness_score": float(action_score[batch_idx].detach().cpu().item()),
                    "boundary_score": float(boundary_score[batch_idx].detach().cpu().item()),
                    "uncertainty_score": float(uncertainty_score[batch_idx].detach().cpu().item()),
                    "redundancy_score": float(redundancy_score[batch_idx].detach().cpu().item()),
                    "valid_len": int(valid_counts[batch_idx].detach().cpu().item()),
                    "deploy_time_signals": [
                        "frame_selection_logits",
                        "actionness_logits",
                        "boundary_logits",
                        "uncertainty_logits",
                        "redundancy_logits",
                        "valid_len",
                    ],
                    "uses_gt": False,
                    "uses_teacher": False,
                    "uses_raw_prediction_cache": False,
                    "uses_p2": False,
                    "safety_gate_only": False,
                    "dynamic_budget_validation": False,
                    "metric_claim_allowed": False,
                    "paper_claim_allowed": False,
                }
            )
        return {"budgets": budgets, "metadata": metadata}

    @staticmethod
    def _dynamic_budget_head_mean(
        *,
        reader_outputs: Mapping[str, torch.Tensor],
        candidate_valid: torch.Tensor,
        names: Sequence[str],
        fallback: torch.Tensor | None,
    ) -> torch.Tensor:
        tensors = []
        for name in names:
            value = reader_outputs.get(name)
            if not torch.is_tensor(value):
                continue
            if tuple(value.shape) != tuple(candidate_valid.shape):
                raise ValueError(f"dynamic_budget reader output {name} must match candidate_valid shape")
            _require_finite(value, f"dynamic_budget reader output {name}", error_type=ValueError)
            tensors.append(torch.sigmoid(value.float()))
        if not tensors and fallback is not None:
            if tuple(fallback.shape) != tuple(candidate_valid.shape):
                raise ValueError("dynamic_budget fallback scores must match candidate_valid shape")
            _require_finite(fallback, "dynamic_budget fallback scores", error_type=ValueError)
            tensors.append(torch.sigmoid(fallback.float()))
        if not tensors:
            return torch.zeros(
                (candidate_valid.shape[0],),
                dtype=torch.float32,
                device=candidate_valid.device,
            )
        stacked = torch.stack(tensors, dim=0).mean(dim=0).masked_fill(~candidate_valid, 0.0)
        denom = candidate_valid.to(dtype=torch.float32).sum(dim=1).clamp_min(1.0)
        return stacked.sum(dim=1) / denom

    @staticmethod
    def _uniform_anchor_positions(*, valid_positions: torch.Tensor, count: int) -> torch.Tensor:
        count = min(int(count), int(valid_positions.numel()))
        if count <= 0:
            return valid_positions.new_empty((0,))
        if count == 1:
            return valid_positions[:1]
        anchor_offsets = torch.linspace(
            0,
            int(valid_positions.numel()) - 1,
            steps=count,
            device=valid_positions.device,
            dtype=torch.float32,
        ).round().to(dtype=torch.long)
        anchors = valid_positions[anchor_offsets]
        if anchors.unique().numel() == anchors.numel():
            return anchors
        repaired = []
        used = set()
        for pos_tensor in anchors:
            pos = int(pos_tensor.item())
            if pos not in used:
                repaired.append(pos_tensor)
                used.add(pos)
        for pos_tensor in valid_positions:
            pos = int(pos_tensor.item())
            if pos in used:
                continue
            repaired.append(pos_tensor)
            used.add(pos)
            if len(repaired) == count:
                break
        return torch.stack(repaired, dim=0)

    @staticmethod
    def _max_gap_guard_positions(
        *,
        valid_positions: torch.Tensor,
        count: int,
        max_gap: int,
    ) -> torch.Tensor:
        count = min(int(count), int(valid_positions.numel()))
        max_gap = int(max_gap)
        if count <= 0 or max_gap <= 0:
            return valid_positions.new_empty((0,))
        if count == 1:
            return valid_positions[:1]
        if int(valid_positions[-1].item()) - int(valid_positions[0].item()) <= max_gap * max(1, count - 1):
            anchor_offsets = torch.linspace(
                0,
                int(valid_positions.numel()) - 1,
                steps=count,
                device=valid_positions.device,
                dtype=torch.float32,
            ).round().to(dtype=torch.long)
            anchors = valid_positions[anchor_offsets]
            if anchors.unique().numel() == anchors.numel():
                return anchors
        repaired = []
        last_pos: int | None = None
        for pos_tensor in valid_positions:
            pos = int(pos_tensor.item())
            if last_pos is None or (pos - last_pos) >= max_gap:
                repaired.append(pos_tensor)
                last_pos = pos
                if len(repaired) == count:
                    break
        if len(repaired) == 0:
            return valid_positions.new_empty((0,))
        return torch.stack(repaired, dim=0)

    @staticmethod
    def _flatten_time(inputs: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...], bool]:
        if inputs.ndim == 6:
            batch, num_views, channels, time, height, width = inputs.shape
            flat = inputs.permute(0, 3, 1, 2, 4, 5).contiguous().view(batch, time, -1)
            return flat, (num_views, channels, height, width), True
        if inputs.ndim == 5:
            batch, channels, time, height, width = inputs.shape
            flat = inputs.permute(0, 2, 1, 3, 4).contiguous().view(batch, time, -1)
            return flat, (channels, height, width), False
        raise ValueError(f"unsupported input shape: {tuple(inputs.shape)}")

    @staticmethod
    def _restore_time(flat: torch.Tensor, shape_tail: tuple[int, ...], has_view_dim: bool) -> torch.Tensor:
        batch, time, _dim = flat.shape
        if has_view_dim:
            num_views, channels, height, width = shape_tail
            return flat.view(batch, time, num_views, channels, height, width).permute(0, 2, 3, 1, 4, 5).contiguous()
        channels, height, width = shape_tail
        return flat.view(batch, time, channels, height, width).permute(0, 2, 1, 3, 4).contiguous()

    def _apply_sparse_transport(
        self,
        inputs: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        transport_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        flat, shape_tail, has_view_dim = self._flatten_time(inputs)
        flat = flat.to(dtype=weights.dtype)
        batch, _dense_len, flat_dim = flat.shape
        _require_finite(flat, "sparse transport dense inputs")
        _require_finite(weights, "sparse transport gather weights")
        if indices.ndim != 3 or indices.shape[:2] != (batch, self.target_len):
            raise ValueError("sparse transport indices must be [B,target_len,K]")
        if tuple(weights.shape) != tuple(indices.shape):
            raise ValueError("sparse transport weights must match indices")
        if bool((indices < 0).any().item()) or bool((indices >= int(_dense_len)).any().item()):
            raise ValueError("sparse transport indices out of dense input range")
        out = torch.zeros((batch, self.target_len, flat_dim), dtype=flat.dtype, device=flat.device)
        for item_idx in range(indices.shape[-1]):
            gather_index = indices[:, :, item_idx].unsqueeze(-1).expand(batch, self.target_len, flat_dim)
            gathered = flat.gather(dim=1, index=gather_index)
            out = out + gathered * weights[:, :, item_idx].unsqueeze(-1)
        if transport_weights is not None:
            transport_weights = transport_weights.to(dtype=flat.dtype)
            if tuple(transport_weights.shape) != (batch, self.target_len, int(_dense_len)):
                raise ValueError("transport_weights must be [B,target_len,dense_len]")
            _require_finite(transport_weights, "sparse transport straight-through weights")
            if self.st_surrogate_mode == "full_flat":
                surrogate = torch.bmm(transport_weights, flat.detach())
            elif self.st_surrogate_mode == "mean_proxy":
                frame_proxy = flat.detach().mean(dim=-1, keepdim=True)
                surrogate = torch.bmm(transport_weights, frame_proxy).expand_as(out)
            else:
                raise ValueError(f"unknown st_surrogate_mode={self.st_surrogate_mode}")
            _require_finite(surrogate, "sparse transport straight-through surrogate")
            out = out + (surrogate - surrogate.detach())
        _require_finite(out, "sparse transport output flat")
        restored = self._restore_time(out, shape_tail, has_view_dim)
        _require_finite(restored, "sparse transport output")
        return restored

    def _write_selected_axis_meta(
        self,
        metas: Sequence[dict[str, Any]] | None,
        selected_positions: torch.Tensor,
        valid_lengths: torch.Tensor,
        selected_output_valid_lengths: torch.Tensor,
        selected_roles: Sequence[Sequence[str]] | None = None,
        raw_slot_dense_indices: Sequence[Sequence[int]] | None = None,
        raw_slot_duplicate_rates: Sequence[float] | None = None,
        raw_slot_unique_counts: Sequence[int] | None = None,
        reader_fill_counts: Sequence[int] | None = None,
        st_active_row_counts: Sequence[int] | None = None,
        dynamic_budget_meta: Sequence[Mapping[str, Any]] | None = None,
        max_gap_guard_meta: Sequence[Mapping[str, Any]] | None = None,
        interval_packet_metadata: Sequence[Mapping[str, Any]] | None = None,
        coarse_policy_meta: Sequence[Mapping[str, Any]] | None = None,
        reader_outputs: Mapping[str, torch.Tensor] | None = None,
        candidate_valid: torch.Tensor | None = None,
        candidate_dense_indices: torch.Tensor | None = None,
        gt_segments=None,
        training: bool = False,
    ) -> list[dict[str, Any]]:
        if metas is None:
            metas = [{} for _ in range(selected_positions.shape[0])]
        metas = list(metas)
        if len(metas) != selected_positions.shape[0]:
            raise ValueError("metas length must match batch size")
        positions_cpu = selected_positions.detach().cpu().tolist()
        valid_cpu = valid_lengths.detach().cpu().tolist()
        selected_valid_cpu = selected_output_valid_lengths.detach().cpu().tolist()
        for idx, meta in enumerate(metas):
            selected_count = int(selected_valid_cpu[idx])
            selected_prefix = positions_cpu[idx][:selected_count]
            dense_indices = [int(pos) for pos in positions_cpu[idx]]
            prefix_indices = [int(pos) for pos in selected_prefix]
            roles = list(selected_roles[idx]) if selected_roles is not None else []
            meta["irregular_selected_positions"] = [float(pos) for pos in selected_prefix]
            meta["irregular_selected_valid_len"] = float(valid_cpu[idx])
            meta["irregular_selected_valid_len_semantics"] = "carried_forward_dense_valid_len_alias"
            meta["irregular_dense_valid_len"] = int(valid_cpu[idx])
            meta["irregular_selected_count"] = int(selected_valid_cpu[idx])
            meta["selected_valid_len"] = int(selected_valid_cpu[idx])
            meta["irregular_selected_output_valid_len"] = float(selected_valid_cpu[idx])
            meta["irregular_native_axis"] = not bool(self.remap_gt_to_selected_axis)
            meta["remap_gt_to_selected_axis"] = bool(self.remap_gt_to_selected_axis)
            meta["pc_ot_mras_prebackbone_selected_dense_indices"] = dense_indices
            meta["pc_ot_mras_prebackbone_valid_len"] = int(valid_cpu[idx])
            meta["pc_ot_mras_prebackbone_gap"] = [
                int(right - left) for left, right in zip(prefix_indices[:-1], prefix_indices[1:])
            ]
            unique_count = len(set(prefix_indices))
            meta["pc_ot_mras_prebackbone_duplicate_rate"] = 1.0 - float(unique_count) / float(max(1, len(prefix_indices)))
            meta["pc_ot_mras_prebackbone_selection_unit"] = int(self.selection_unit)
            meta["pc_ot_mras_prebackbone_remap_gt_to_selected_axis"] = bool(self.remap_gt_to_selected_axis)
            meta["pc_ot_mras_prebackbone_residual_count"] = (
                int(self.residual_count) if self.residual_count is not None else None
            )
            meta["pc_ot_mras_prebackbone_residual_slot_role"] = self.residual_slot_role
            meta["pc_ot_mras_prebackbone_selector_support_status"] = self.selector_support_status
            selection_strategy = getattr(self, "selection_strategy", "slot_transport")
            meta["pc_ot_mras_prebackbone_selection_strategy"] = selection_strategy
            hard_source_by_strategy = {
                "frame_score_topk": "frame_selection_logits",
                "frame_score_global_rank_st": "frame_selection_logits",
                "interval_boundary_packet": "interval_boundary_packet",
                "coarse_actionness_uncertainty": "classification_probability_uncertainty_change",
            }
            meta["pc_ot_mras_prebackbone_hard_selection_source"] = hard_source_by_strategy.get(
                selection_strategy,
                "slot_transport",
            )
            meta["pc_ot_mras_prebackbone_slot_not_hard_source"] = selection_strategy != "slot_transport"
            meta["pc_ot_mras_prebackbone_frame_score_st_surrogate"] = getattr(
                self,
                "frame_score_st_surrogate",
                "local_softmax",
            )
            meta["pc_ot_mras_prebackbone_frame_score_st_logit_clamp"] = float(
                getattr(self, "frame_score_st_logit_clamp", 0.0)
            )
            meta["pc_ot_mras_prebackbone_frame_score_st_gradient_scale"] = float(
                getattr(self, "frame_score_st_gradient_scale", 1.0)
            )
            meta["pc_ot_mras_prebackbone_frame_score_aux_logit_clamp"] = float(
                getattr(self, "frame_score_aux_logit_clamp", 0.0)
            )
            meta["pc_ot_mras_prebackbone_global_rank_st_temperature"] = float(
                getattr(self, "global_rank_st_temperature", getattr(self, "frame_score_st_temperature", 1.0))
            )
            meta["pc_ot_mras_prebackbone_global_rank_st_topk"] = int(
                getattr(self, "global_rank_st_topk", self.target_len)
            )
            meta["pc_ot_mras_prebackbone_global_rank_st_rank_width"] = float(
                getattr(self, "global_rank_st_rank_width", 1.0)
            )
            meta["pc_ot_mras_prebackbone_selected_roles"] = roles
            meta["pc_ot_mras_prebackbone_raw_slot_dense_indices"] = (
                [int(pos) for pos in raw_slot_dense_indices[idx]] if raw_slot_dense_indices is not None else []
            )
            meta["pc_ot_mras_prebackbone_raw_slot_duplicate_rate"] = (
                float(raw_slot_duplicate_rates[idx]) if raw_slot_duplicate_rates is not None else None
            )
            meta["pc_ot_mras_prebackbone_raw_slot_unique_count"] = (
                int(raw_slot_unique_counts[idx]) if raw_slot_unique_counts is not None else None
            )
            meta["pc_ot_mras_prebackbone_reader_fill_count"] = (
                int(reader_fill_counts[idx]) if reader_fill_counts is not None else 0
            )
            meta["pc_ot_mras_prebackbone_st_active_row_count"] = (
                int(st_active_row_counts[idx]) if st_active_row_counts is not None else 0
            )
            meta["pc_ot_mras_prebackbone_dynamic_budget"] = (
                dict(dynamic_budget_meta[idx])
                if dynamic_budget_meta is not None
                else {
                    "enabled": False,
                    "protocol": "fixed",
                    "budget": selected_count,
                    "min_budget": self.target_len,
                    "target_budget": self.target_len,
                    "max_budget": self.target_len,
                    "average_budget": self.target_len,
                    "uses_gt": False,
                    "uses_teacher": False,
                    "uses_raw_prediction_cache": False,
                    "uses_p2": False,
                }
            )
            meta["pc_ot_mras_prebackbone_max_gap_guard"] = (
                dict(max_gap_guard_meta[idx])
                if max_gap_guard_meta is not None
                else {
                    "enabled": bool(getattr(self, "max_dense_gap", 0) > 0 and getattr(self, "max_gap_guard_count", 0) > 0),
                    "max_dense_gap": int(getattr(self, "max_dense_gap", 0)),
                    "max_gap_guard_count": int(getattr(self, "max_gap_guard_count", 0)),
                    "applied_count": 0,
                    "max_gap_after": max(meta["pc_ot_mras_prebackbone_gap"])
                    if meta["pc_ot_mras_prebackbone_gap"]
                    else 0,
                    "safety_gate_only": True,
                }
            )
            meta["pc_ot_mras_prebackbone_interval_packet_metadata"] = (
                dict(interval_packet_metadata[idx])
                if interval_packet_metadata is not None
                else {
                    "enabled": False,
                    "status": "not_interval_boundary_packet",
                    "boundary_budget": 0,
                    "interior_budget": 0,
                    "boundary_positions": [],
                    "interior_positions": [],
                    "source_heads": [],
                    "uses_gt": False,
                    "uses_teacher": False,
                    "uses_raw_prediction_cache": False,
                    "uses_p2": False,
                }
            )
            meta["pc_ot_mras_prebackbone_coarse_actionness_policy"] = (
                dict(coarse_policy_meta[idx])
                if coarse_policy_meta is not None
                else {
                    "enabled": False,
                    "protocol": "not_coarse_actionness_uncertainty",
                    "uses_learned_boundary_head": None,
                    "uses_gt": False,
                    "uses_teacher": False,
                    "uses_raw_prediction_cache": False,
                    "uses_p2": False,
                }
            )
            head_diagnostics = self._reader_head_diagnostics_for_sample(
                reader_outputs=reader_outputs,
                candidate_valid=candidate_valid,
                candidate_dense_indices=candidate_dense_indices,
                batch_idx=idx,
                selected_prefix=prefix_indices,
            )
            meta["pc_ot_mras_prebackbone_reader_head_diagnostics"] = head_diagnostics
            meta["pc_ot_mras_prebackbone_reader_diagnostics"] = self._reader_diagnostic_summary(head_diagnostics)
            meta["pc_ot_mras_prebackbone_protocol_flags"] = {
                "uses_p2": False,
                "uses_gt": False,
                "uses_raw_prediction_cache": False,
                "uses_teacher": False,
                "uses_test_gt": False,
                "uses_learned_boundary_head": False,
            }
            meta["pc_ot_mras_prebackbone_scout_feature_source"] = self.scout_feature_source
            meta["pc_ot_mras_prebackbone_scout_spatial_size"] = [
                int(self.scout_spatial_size[0]),
                int(self.scout_spatial_size[1]),
            ]
            meta["pc_ot_mras_prebackbone_boundary_diagnostics"] = {
                "status": "placeholder"
                if head_diagnostics.get("status") != "available"
                else head_diagnostics.get("status"),
                "boundary_selected_mean": head_diagnostics.get("selected_mean", {}).get("boundary_logits"),
                "boundary_valid_mean": head_diagnostics.get("valid_mean", {}).get("boundary_logits"),
                "score_rank_hook": "not_computed_in_selector_forward",
            }
            meta["pc_ot_mras_prebackbone_selector_source"] = self.meta_source
            self._append_metadata_dump_row(
                meta=meta,
                batch_idx=idx,
                selected_dense_indices=prefix_indices,
                valid_len=int(valid_cpu[idx]),
                gt_segments=gt_segments,
                reader_outputs=reader_outputs,
                candidate_valid=candidate_valid,
                candidate_dense_indices=candidate_dense_indices,
                training=training,
            )
        return metas

    def _append_metadata_dump_row(
        self,
        *,
        meta: Mapping[str, Any],
        batch_idx: int,
        selected_dense_indices: Sequence[int],
        valid_len: int,
        gt_segments,
        reader_outputs: Mapping[str, torch.Tensor] | None,
        candidate_valid: torch.Tensor | None,
        candidate_dense_indices: torch.Tensor | None,
        training: bool,
    ) -> None:
        dump_path = os.environ.get("PC_OT_MRAS_PREBACKBONE_SELECTOR_METADATA_JSONL") or os.environ.get(
            "C3_SELECTOR_METADATA_JSONL"
        )
        if not dump_path:
            return
        if str(os.environ.get("RANK", "0")) != "0":
            return
        max_rows = self._metadata_dump_max_rows()
        written = int(getattr(self, "_metadata_dump_count", 0))
        if max_rows is not None and written >= max_rows:
            return

        row = {
            "schema_version": "pc_ot_mras_prebackbone_selector_metadata_dump_v0",
            "phase": "train" if training else "eval",
            "sample_id": self._metadata_dump_sample_id(meta, batch_idx),
            "video_name": meta.get("video_name"),
            "window_start_frame": meta.get("window_start_frame"),
            "selected_dense_indices": [int(item) for item in selected_dense_indices],
            "valid_len": int(valid_len),
            "gt_segments": self._metadata_dump_segments(gt_segments, batch_idx),
            "selector_scores": self._metadata_dump_scores(
                reader_outputs=reader_outputs,
                candidate_valid=candidate_valid,
                candidate_dense_indices=candidate_dense_indices,
                batch_idx=batch_idx,
                valid_len=int(valid_len),
            ),
            "selector_score_components": self._metadata_dump_score_components(
                reader_outputs=reader_outputs,
                candidate_valid=candidate_valid,
                candidate_dense_indices=candidate_dense_indices,
                batch_idx=batch_idx,
                valid_len=int(valid_len),
            ),
            "selector_candidate_points": list(
                meta.get("pc_ot_mras_prebackbone_coarse_actionness_policy", {}).get("candidate_points", [])
            ),
            "packet_roles": list(meta.get("pc_ot_mras_prebackbone_selected_roles", [])),
            "irregular_selected_positions": list(meta.get("irregular_selected_positions", [])),
            "irregular_dense_valid_len": int(valid_len),
            "irregular_selected_valid_len": int(valid_len),
            "irregular_selected_output_valid_len": meta.get("irregular_selected_output_valid_len"),
            "irregular_native_axis": meta.get("irregular_native_axis"),
            "pc_ot_mras_prebackbone_raw_slot_dense_indices": meta.get(
                "pc_ot_mras_prebackbone_raw_slot_dense_indices",
                [],
            ),
            "pc_ot_mras_prebackbone_raw_slot_duplicate_rate": meta.get(
                "pc_ot_mras_prebackbone_raw_slot_duplicate_rate"
            ),
            "pc_ot_mras_prebackbone_raw_slot_unique_count": meta.get("pc_ot_mras_prebackbone_raw_slot_unique_count"),
            "pc_ot_mras_prebackbone_reader_fill_count": meta.get("pc_ot_mras_prebackbone_reader_fill_count"),
            "pc_ot_mras_prebackbone_st_active_row_count": meta.get("pc_ot_mras_prebackbone_st_active_row_count"),
            "pc_ot_mras_prebackbone_selection_strategy": meta.get("pc_ot_mras_prebackbone_selection_strategy"),
            "pc_ot_mras_prebackbone_hard_selection_source": meta.get(
                "pc_ot_mras_prebackbone_hard_selection_source"
            ),
            "pc_ot_mras_prebackbone_dynamic_budget": meta.get("pc_ot_mras_prebackbone_dynamic_budget"),
            "pc_ot_mras_prebackbone_coarse_actionness_policy": meta.get(
                "pc_ot_mras_prebackbone_coarse_actionness_policy"
            ),
            "pc_ot_mras_prebackbone_protocol_flags": meta.get("pc_ot_mras_prebackbone_protocol_flags"),
            "pc_ot_mras_prebackbone_selector_source": meta.get("pc_ot_mras_prebackbone_selector_source"),
        }
        path = Path(dump_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(self._metadata_dump_jsonable(row), sort_keys=True, allow_nan=False) + "\n")
        self._metadata_dump_count = written + 1

    @staticmethod
    def _metadata_dump_max_rows() -> int | None:
        raw = os.environ.get("PC_OT_MRAS_PREBACKBONE_SELECTOR_METADATA_MAX_ROWS")
        if raw is None or str(raw).strip() == "":
            return None
        max_rows = int(raw)
        if max_rows < 0:
            raise ValueError("PC_OT_MRAS_PREBACKBONE_SELECTOR_METADATA_MAX_ROWS must be non-negative")
        return max_rows

    @staticmethod
    def _metadata_dump_sample_id(meta: Mapping[str, Any], batch_idx: int) -> str:
        explicit = meta.get("sample_id")
        if explicit:
            return str(explicit)
        video_name = meta.get("video_name", meta.get("video_id", f"sample_{batch_idx}"))
        if "window_start_frame" in meta:
            window = meta["window_start_frame"]
            if isinstance(window, float) and float(window).is_integer():
                window = int(window)
            return f"{video_name}|window_start_frame={window}"
        return str(video_name)

    @staticmethod
    def _metadata_dump_segments(gt_segments, batch_idx: int) -> list[list[float]]:
        if gt_segments is None:
            return []
        if torch.is_tensor(gt_segments):
            value = gt_segments[batch_idx] if gt_segments.ndim >= 3 else gt_segments
        elif isinstance(gt_segments, Sequence) and not isinstance(gt_segments, (str, bytes)):
            if batch_idx >= len(gt_segments):
                return []
            value = gt_segments[batch_idx]
        else:
            return []
        data = PCOTMRASPreBackboneFrameSelector._metadata_dump_jsonable(value)
        if not isinstance(data, list):
            return []
        if len(data) == 2 and all(isinstance(item, (int, float)) for item in data):
            data = [data]
        out: list[list[float]] = []
        for item in data:
            if not isinstance(item, list) or len(item) < 2:
                continue
            try:
                start = float(item[0])
                end = float(item[1])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(start) or not math.isfinite(end):
                continue
            out.append([start, end])
        return out

    @staticmethod
    def _metadata_dump_scores(
        *,
        reader_outputs: Mapping[str, torch.Tensor] | None,
        candidate_valid: torch.Tensor | None,
        candidate_dense_indices: torch.Tensor | None,
        batch_idx: int,
        valid_len: int,
    ) -> list[float]:
        if reader_outputs is None:
            return []
        for name in (
            "frame_selection_logits",
            "actionness_logits",
            "action_logits",
            "value_logits",
            "boundary_logits",
            "risk_logits",
        ):
            tensor = reader_outputs.get(name)
            if not torch.is_tensor(tensor) or tensor.ndim != 2 or int(tensor.shape[0]) <= int(batch_idx):
                continue
            scores = tensor[batch_idx].detach().float().cpu()
            if (
                candidate_valid is not None
                and candidate_dense_indices is not None
                and candidate_valid.ndim == 2
                and candidate_dense_indices.ndim == 2
                and int(candidate_valid.shape[0]) > int(batch_idx)
                and int(candidate_dense_indices.shape[0]) > int(batch_idx)
                and int(candidate_valid.shape[1]) == int(scores.numel())
                and int(candidate_dense_indices.shape[1]) == int(scores.numel())
            ):
                dense_scores = [0.0 for _ in range(max(0, int(valid_len)))]
                valid_mask = candidate_valid[batch_idx].detach().cpu().bool()
                dense_indices = candidate_dense_indices[batch_idx].detach().cpu().long()
                for score, is_valid, pos in zip(scores.tolist(), valid_mask.tolist(), dense_indices.tolist()):
                    if is_valid and 0 <= int(pos) < len(dense_scores):
                        dense_scores[int(pos)] = float(score)
                return dense_scores
            return [float(item) for item in scores.tolist()]
        return []

    def _metadata_dump_score_components(
        self,
        *,
        reader_outputs: Mapping[str, torch.Tensor] | None,
        candidate_valid: torch.Tensor | None,
        candidate_dense_indices: torch.Tensor | None,
        batch_idx: int,
        valid_len: int,
    ) -> dict[str, list[float]]:
        if reader_outputs is None:
            return {}

        def to_dense_scores(tensor: torch.Tensor) -> list[float]:
            scores = tensor[batch_idx].detach().float().cpu()
            if (
                candidate_valid is not None
                and candidate_dense_indices is not None
                and candidate_valid.ndim == 2
                and candidate_dense_indices.ndim == 2
                and int(candidate_valid.shape[0]) > int(batch_idx)
                and int(candidate_dense_indices.shape[0]) > int(batch_idx)
                and int(candidate_valid.shape[1]) == int(scores.numel())
                and int(candidate_dense_indices.shape[1]) == int(scores.numel())
            ):
                dense_scores = [0.0 for _ in range(max(0, int(valid_len)))]
                valid_mask = candidate_valid[batch_idx].detach().cpu().bool()
                dense_indices = candidate_dense_indices[batch_idx].detach().cpu().long()
                for score, is_valid, pos in zip(scores.tolist(), valid_mask.tolist(), dense_indices.tolist()):
                    if is_valid and 0 <= int(pos) < len(dense_scores):
                        dense_scores[int(pos)] = float(score)
                return dense_scores
            return [float(item) for item in scores.tolist()]

        components: dict[str, list[float]] = {}
        for name in (
            "frame_selection_logits",
            "actionness_logits",
            "action_logits",
            "value_logits",
            "boundary_logits",
            "risk_logits",
        ):
            tensor = reader_outputs.get(name)
            if torch.is_tensor(tensor) and tensor.ndim == 2 and int(tensor.shape[0]) > int(batch_idx):
                components[name] = to_dense_scores(tensor)

        action_logits = reader_outputs.get("actionness_logits", reader_outputs.get("action_logits"))
        if (
            torch.is_tensor(action_logits)
            and candidate_valid is not None
            and action_logits.ndim == 2
            and candidate_valid.ndim == 2
            and tuple(action_logits.shape) == tuple(candidate_valid.shape)
        ):
            coarse_scores = self._coarse_actionness_scores(
                reader_outputs=reader_outputs,
                candidate_valid=candidate_valid.to(device=action_logits.device),
            )
            for output_key, score_key in (
                ("p_action", "coarse_action"),
                ("entropy", "entropy"),
                ("margin", "margin"),
                ("p_change", "coarse_change"),
                ("uncertainty", "coarse_uncertainty"),
                ("change", "coarse_change"),
                ("background", "coarse_background"),
                ("mixed", "coarse_mixed_fill"),
            ):
                components[output_key] = to_dense_scores(coarse_scores[score_key])
        return components

    @staticmethod
    def _metadata_dump_jsonable(value: Any) -> Any:
        if torch.is_tensor(value):
            return PCOTMRASPreBackboneFrameSelector._metadata_dump_jsonable(value.detach().cpu().tolist())
        if hasattr(value, "item") and not isinstance(value, (str, bytes)):
            try:
                return PCOTMRASPreBackboneFrameSelector._metadata_dump_jsonable(value.item())
            except (TypeError, ValueError):
                pass
        if isinstance(value, Mapping):
            return {str(key): PCOTMRASPreBackboneFrameSelector._metadata_dump_jsonable(item) for key, item in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [PCOTMRASPreBackboneFrameSelector._metadata_dump_jsonable(item) for item in value]
        if isinstance(value, float):
            return float(value) if math.isfinite(value) else None
        if isinstance(value, (bool, int, str)) or value is None:
            return value
        return str(value)

    @staticmethod
    def _reader_head_diagnostics_for_sample(
        *,
        reader_outputs: Mapping[str, torch.Tensor] | None,
        candidate_valid: torch.Tensor | None,
        candidate_dense_indices: torch.Tensor | None,
        batch_idx: int,
        selected_prefix: Sequence[int],
    ) -> dict[str, Any]:
        if reader_outputs is None or candidate_valid is None or candidate_dense_indices is None:
            return {"status": "unavailable", "available_heads": [], "valid_mean": {}, "selected_mean": {}}
        if candidate_valid.ndim != 2 or candidate_dense_indices.ndim != 2:
            return {"status": "unavailable", "available_heads": [], "valid_mean": {}, "selected_mean": {}}
        head_names = (
            "actionness_logits",
            "action_logits",
            "value_logits",
            "start_logits",
            "end_logits",
            "boundary_logits",
            "risk_logits",
            "uncertainty_logits",
            "redundancy_logits",
            "frame_selection_logits",
        )
        available: list[str] = []
        valid_mean: dict[str, float] = {}
        selected_mean: dict[str, float] = {}
        valid_mask_base = candidate_valid[batch_idx].detach().bool()
        candidate_positions_base = candidate_dense_indices[batch_idx].detach()
        for name in head_names:
            tensor = reader_outputs.get(name)
            if not torch.is_tensor(tensor) or tensor.ndim != 2:
                continue
            if int(tensor.shape[0]) <= int(batch_idx) or int(tensor.shape[1]) != int(valid_mask_base.numel()):
                continue
            scores = tensor[batch_idx].detach().float()
            valid_mask = valid_mask_base.to(device=scores.device)
            candidate_positions = candidate_positions_base.to(device=scores.device)
            valid_values = scores[valid_mask]
            if valid_values.numel() == 0:
                continue
            if selected_prefix:
                selected_tensor = torch.tensor(
                    [int(item) for item in selected_prefix],
                    device=scores.device,
                    dtype=candidate_positions.dtype,
                )
                selected_mask = (candidate_positions[None, :] == selected_tensor[:, None]).any(dim=0) & valid_mask
            else:
                selected_mask = torch.zeros_like(valid_mask)
            selected_values = scores[selected_mask]
            available.append(name)
            valid_mean[name] = float(valid_values.mean().item())
            selected_mean[name] = float(selected_values.mean().item()) if selected_values.numel() else None
        status = "available" if available else "unavailable"
        return {
            "status": status,
            "available_heads": available,
            "valid_mean": valid_mean,
            "selected_mean": selected_mean,
        }

    @staticmethod
    def _reader_diagnostic_summary(head_diagnostics: Mapping[str, Any]) -> dict[str, dict[str, bool]]:
        available = set(head_diagnostics.get("available_heads", []))
        return {
            "action": {
                "available": bool(
                    available
                    & {
                        "actionness_logits",
                        "action_logits",
                        "value_logits",
                        "frame_selection_logits",
                    }
                )
            },
            "boundary": {
                "available": bool(
                    available
                    & {
                        "start_logits",
                        "end_logits",
                        "boundary_logits",
                        "risk_logits",
                    }
                )
            },
            "uncertainty": {"available": "uncertainty_logits" in available},
            "redundancy": {"available": "redundancy_logits" in available},
            "head": {"available": bool(available)},
        }

    def _remap_gt_batch(
        self,
        gt_segments,
        gt_labels,
        selected_positions: torch.Tensor,
        valid_lengths: torch.Tensor,
        selected_output_valid_lengths: torch.Tensor,
    ):
        if not self.remap_gt_to_selected_axis:
            return gt_segments, gt_labels
        if gt_segments is None or gt_labels is None:
            return gt_segments, gt_labels
        if torch.is_tensor(gt_segments):
            segments_iter = list(gt_segments)
        else:
            segments_iter = list(gt_segments)
        if torch.is_tensor(gt_labels):
            labels_iter = list(gt_labels)
        else:
            labels_iter = list(gt_labels)
        new_segments = []
        new_labels = []
        for idx, (segments, labels) in enumerate(zip(segments_iter, labels_iter)):
            mapped_segments, mapped_labels = self._remap_one_gt(
                segments=segments,
                labels=labels,
                selected_positions=selected_positions[idx].to(device=segments.device, dtype=segments.dtype),
                valid_len=valid_lengths[idx].to(device=segments.device, dtype=segments.dtype),
                selected_output_valid_len=selected_output_valid_lengths[idx].to(device=segments.device),
            )
            new_segments.append(mapped_segments)
            new_labels.append(mapped_labels)
        return new_segments, new_labels

    @staticmethod
    def _interp_dense_to_selected(coords: torch.Tensor, selected_positions: torch.Tensor, valid_len: torch.Tensor) -> torch.Tensor:
        xp = torch.cat([selected_positions, valid_len.reshape(1)])
        fp = torch.arange(xp.numel(), device=xp.device, dtype=coords.dtype)
        coords = coords.clamp(min=0.0, max=float(valid_len.detach().cpu().item()))
        right = torch.searchsorted(xp, coords, right=False)
        below = right <= 0
        above = right >= xp.numel()
        right = right.clamp(min=1, max=xp.numel() - 1)
        left = right - 1
        denom = (xp[right] - xp[left]).clamp_min(torch.finfo(coords.dtype).eps)
        alpha = (coords - xp[left]) / denom
        mapped = fp[left] * (1.0 - alpha) + fp[right] * alpha
        mapped = torch.where(below, torch.zeros_like(mapped), mapped)
        mapped = torch.where(above, torch.full_like(mapped, float(selected_positions.numel())), mapped)
        return mapped

    def _remap_one_gt(
        self,
        *,
        segments: torch.Tensor,
        labels: torch.Tensor,
        selected_positions: torch.Tensor,
        valid_len: torch.Tensor,
        selected_output_valid_len: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if segments.numel() == 0:
            return segments.new_zeros((0, 2)), labels.new_zeros((0,), dtype=labels.dtype)
        pair_count = min(int(segments.shape[0]), int(labels.shape[0]))
        if pair_count <= 0:
            return segments.new_zeros((0, 2)), labels.new_zeros((0,), dtype=labels.dtype)
        segments = segments[:pair_count]
        labels = labels[:pair_count]
        effective_len = max(1, min(int(selected_output_valid_len.detach().cpu().item()), int(selected_positions.numel())))
        effective_positions = selected_positions[:effective_len]
        starts = self._interp_dense_to_selected(segments[:, 0], effective_positions, valid_len)
        ends = self._interp_dense_to_selected(segments[:, 1], effective_positions, valid_len)
        max_coord = float(effective_len)
        starts = starts.clamp(0.0, max_coord)
        ends = ends.clamp(0.0, max_coord)
        ends = torch.maximum(ends, starts + 1.0e-3)
        ends = ends.clamp(0.0, max_coord)
        keep = ends > starts
        if not bool(keep.any().item()):
            return segments.new_zeros((0, 2)), labels.new_zeros((0,), dtype=labels.dtype)
        return torch.stack([starts[keep], ends[keep]], dim=-1), labels[keep]

    def _losses(
        self,
        *,
        reader_outputs: Mapping[str, torch.Tensor],
        valid_mask: torch.Tensor,
        candidate_dense_indices: torch.Tensor | None = None,
        gt_segments,
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        regularizers = reader_outputs.get("regularizers")
        if isinstance(regularizers, Mapping) and "total_regularizer" in regularizers:
            regularizer_loss = regularizers["total_regularizer"] * self.reader_regularizer_loss_weight
            _require_finite(regularizer_loss, "selector reader regularizer loss")
            losses["selector_reader_regularizer_loss"] = regularizer_loss
        if gt_segments is None:
            return losses
        matrix = reader_outputs.get("acquisition_matrix")
        actionness_logits = reader_outputs.get("actionness_logits", reader_outputs.get("action_logits"))
        value_logits = reader_outputs.get("value_logits", reader_outputs.get("action_logits"))
        risk_logits = reader_outputs.get("risk_logits", reader_outputs.get("boundary_logits"))
        frame_selection_logits = reader_outputs.get("frame_selection_logits")
        uncertainty_logits = reader_outputs.get("uncertainty_logits")
        redundancy_logits = reader_outputs.get("redundancy_logits")
        role_logits = reader_outputs.get("role_logits")
        aux_tensors = (
            matrix,
            actionness_logits,
            value_logits,
            risk_logits,
            frame_selection_logits,
            uncertainty_logits,
            redundancy_logits,
            role_logits,
        )
        if all(not torch.is_tensor(tensor) for tensor in aux_tensors):
            return losses

        dtype = None
        device = valid_mask.device
        for tensor in aux_tensors:
            if torch.is_tensor(tensor):
                _require_finite(tensor, "selector auxiliary tensor")
        for tensor in aux_tensors:
            if torch.is_tensor(tensor):
                dtype = tensor.dtype
                device = tensor.device
                break
        if dtype is None:
            return losses
        action_target, boundary_target = self._dense_gt_targets(
            valid_mask=valid_mask.to(device=device),
            gt_segments=gt_segments,
            dtype=dtype,
            device=device,
            candidate_dense_indices=None
            if candidate_dense_indices is None
            else candidate_dense_indices.to(device=device),
        )
        valid = valid_mask.to(device=device).bool()
        action_target = action_target.float()
        boundary_target = boundary_target.float()
        _require_finite(action_target, "selector action auxiliary target")
        _require_finite(boundary_target, "selector boundary auxiliary target")
        slot_prob = None
        column_mass = None
        if (
            getattr(self, "selection_strategy", "slot_transport") == "coarse_actionness_uncertainty"
            and actionness_logits is not None
            and self.aux_gt_acquisition_loss_weight > 0.0
            and bool(valid.any().item())
        ):
            aux_actionness_logits = _smooth_clamp_logits(
                actionness_logits.float(),
                float(getattr(self, "frame_score_aux_logit_clamp", 0.0)),
                "selector coarse actionness logits",
            )
            actionness_loss = (
                F.binary_cross_entropy_with_logits(aux_actionness_logits[valid], action_target[valid])
                * self.aux_gt_acquisition_loss_weight
            )
            _require_finite(actionness_loss, "selector coarse actionness loss")
            losses["selector_gt_actionness_loss"] = actionness_loss
        if (
            getattr(self, "selection_strategy", "slot_transport") in (
                "frame_score_topk",
                "frame_score_global_rank_st",
            )
            and frame_selection_logits is not None
            and self.aux_gt_acquisition_loss_weight > 0.0
            and bool(valid.any().item())
        ):
            aux_frame_selection_logits = _smooth_clamp_logits(
                frame_selection_logits.float(),
                float(getattr(self, "frame_score_aux_logit_clamp", 0.0)),
                "selector gt frame score logits",
            )
            frame_score_loss = (
                F.binary_cross_entropy_with_logits(aux_frame_selection_logits[valid], action_target[valid])
                * self.aux_gt_acquisition_loss_weight
            )
            _require_finite(frame_score_loss, "selector gt frame score loss")
            losses["selector_gt_frame_score_loss"] = frame_score_loss
        if (
            getattr(self, "selection_strategy", "slot_transport") in (
                "frame_score_topk",
                "frame_score_global_rank_st",
            )
            and frame_selection_logits is not None
            and getattr(self, "aux_frame_score_boundary_loss_weight", 0.0) > 0.0
            and bool(valid.any().item())
        ):
            aux_frame_boundary_logits = _smooth_clamp_logits(
                frame_selection_logits.float(),
                float(getattr(self, "frame_score_aux_logit_clamp", 0.0)),
                "selector gt frame boundary score logits",
            )
            frame_boundary_loss = (
                F.binary_cross_entropy_with_logits(aux_frame_boundary_logits[valid], boundary_target[valid])
                * self.aux_frame_score_boundary_loss_weight
            )
            _require_finite(frame_boundary_loss, "selector gt frame boundary score loss")
            losses["selector_gt_frame_boundary_score_loss"] = frame_boundary_loss
        if (
            matrix is not None
            and getattr(self, "selection_strategy", "slot_transport")
            not in ("frame_score_topk", "frame_score_global_rank_st")
            and self.aux_gt_acquisition_loss_weight > 0.0
            and bool(valid.any().item())
        ):
            slot_prob = matrix.float().masked_fill(~valid[:, None, :], 0.0).clamp(min=0.0, max=1.0)
            _require_finite(slot_prob, "selector acquisition probabilities")
            eps = 1.0e-6
            log_not_selected = torch.log1p(-slot_prob.clamp(max=1.0 - eps)).sum(dim=1)
            union_scores = (-torch.expm1(log_not_selected)).clamp(min=eps, max=1.0 - eps)
            _require_finite(union_scores, "selector union acquisition probabilities")
            union_logits = torch.logit(union_scores)
            _require_finite(union_logits, "selector union acquisition logits")
            loss = (
                F.binary_cross_entropy_with_logits(union_logits[valid], action_target[valid])
                * self.aux_gt_acquisition_loss_weight
            )
            _require_finite(loss, "selector gt acquisition loss")
            losses["selector_gt_acquisition_loss"] = loss
        elif matrix is not None:
            slot_prob = matrix.float().masked_fill(~valid[:, None, :], 0.0).clamp(min=0.0, max=1.0)
            _require_finite(slot_prob, "selector acquisition probabilities")
        if slot_prob is not None:
            column_mass = slot_prob.sum(dim=1).masked_fill(~valid, 0.0)
            _require_finite(column_mass, "selector acquisition column mass")
        if (
            column_mass is not None
            and getattr(self, "aux_duplicate_cap_loss_weight", 0.0) > 0.0
            and bool(valid.any().item())
        ):
            cap = float(self.aux_duplicate_column_cap)
            if cap <= 0.0:
                raise ValueError("aux_duplicate_column_cap must be positive")
            duplicate_cap_loss = F.relu(column_mass[valid] - cap).square().mean()
            duplicate_cap_loss = duplicate_cap_loss * self.aux_duplicate_cap_loss_weight
            _require_finite(duplicate_cap_loss, "selector duplicate column cap loss")
            losses["selector_duplicate_column_cap_loss"] = duplicate_cap_loss
        if value_logits is not None and self.aux_value_loss_weight > 0.0 and bool(valid.any().item()):
            loss = (
                F.binary_cross_entropy_with_logits(value_logits.float()[valid], action_target[valid])
                * self.aux_value_loss_weight
            )
            _require_finite(loss, "selector value auxiliary loss")
            losses["selector_value_aux_loss"] = loss
        if risk_logits is not None and self.aux_risk_loss_weight > 0.0 and bool(valid.any().item()):
            loss = (
                F.binary_cross_entropy_with_logits(risk_logits.float()[valid], boundary_target[valid])
                * self.aux_risk_loss_weight
            )
            _require_finite(loss, "selector risk auxiliary loss")
            losses["selector_risk_aux_loss"] = loss
        uncertainty_target = (0.65 * boundary_target + 0.35 * action_target).clamp(0.0, 1.0)
        _require_finite(uncertainty_target, "selector uncertainty auxiliary target")
        if uncertainty_logits is not None and self.aux_uncertainty_loss_weight > 0.0 and bool(valid.any().item()):
            loss = (
                F.binary_cross_entropy_with_logits(uncertainty_logits.float()[valid], uncertainty_target[valid])
                * self.aux_uncertainty_loss_weight
            )
            _require_finite(loss, "selector uncertainty auxiliary loss")
            losses["selector_uncertainty_aux_loss"] = loss
        redundancy_target = torch.zeros_like(action_target)
        if column_mass is not None:
            mass_denom = column_mass.masked_fill(~valid, 0.0).amax(dim=1, keepdim=True).clamp_min(1.0)
            normalized_mass = (column_mass / mass_denom).clamp(0.0, 1.0)
            duplicate_pressure = F.relu(column_mass - 1.0).clamp(0.0, 1.0)
            redundancy_target = torch.maximum(normalized_mass, duplicate_pressure).detach()
            redundancy_target = redundancy_target.masked_fill(~valid, 0.0)
        _require_finite(redundancy_target, "selector redundancy auxiliary target")
        if redundancy_logits is not None and self.aux_redundancy_loss_weight > 0.0 and bool(valid.any().item()):
            loss = (
                F.binary_cross_entropy_with_logits(redundancy_logits.float()[valid], redundancy_target[valid])
                * self.aux_redundancy_loss_weight
            )
            _require_finite(loss, "selector redundancy auxiliary loss")
            losses["selector_redundancy_aux_loss"] = loss
        if role_logits is not None and self.aux_role_entropy_loss_weight > 0.0 and bool(valid.any().item()):
            if role_logits.shape[-1] < 4:
                raise ValueError("role_logits must contain at least four role classes")
            role_target = torch.zeros(valid.shape, dtype=torch.long, device=device)
            role_target = torch.where(action_target > 0.5, torch.ones_like(role_target), role_target)
            role_target = torch.where(boundary_target > 0.5, torch.full_like(role_target, 2), role_target)
            role_target = torch.where(redundancy_target > 0.5, torch.full_like(role_target, 3), role_target)
            role_target = role_target.masked_fill(~valid, 0)
            loss = F.cross_entropy(role_logits.float()[valid], role_target[valid]) * self.aux_role_entropy_loss_weight
            _require_finite(loss, "selector role auxiliary loss")
            losses["selector_role_aux_loss"] = loss
        return losses

    @staticmethod
    def _dense_gt_targets(
        *,
        valid_mask: torch.Tensor,
        gt_segments,
        dtype: torch.dtype,
        device: torch.device,
        candidate_dense_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_target = torch.zeros(valid_mask.shape, dtype=dtype, device=device)
        boundary_target = torch.zeros(valid_mask.shape, dtype=dtype, device=device)
        segments_iter = list(gt_segments) if not torch.is_tensor(gt_segments) else list(gt_segments)
        if candidate_dense_indices is None:
            candidate_positions = torch.arange(valid_mask.shape[1], device=device, dtype=dtype)[None, :].expand(
                valid_mask.shape[0],
                -1,
            )
        else:
            if tuple(candidate_dense_indices.shape) != tuple(valid_mask.shape):
                raise ValueError("candidate_dense_indices must match valid_mask for selector auxiliary targets")
            candidate_positions = candidate_dense_indices.to(device=device, dtype=dtype)
        for batch_idx, segments in enumerate(segments_iter):
            if segments is None or segments.numel() == 0:
                continue
            seg = segments.to(device=device, dtype=dtype)
            _require_finite(seg, "selector gt segment auxiliary source")
            for start, end in seg:
                positions = candidate_positions[batch_idx]
                action = (positions >= start) & (positions < end)
                action_target[batch_idx] = torch.where(action, torch.ones_like(action_target[batch_idx]), action_target[batch_idx])
                if positions.numel() > 1:
                    positive_delta = (positions[1:] - positions[:-1]).abs()
                    positive_delta = positive_delta[positive_delta > 0]
                    boundary_width = positive_delta.min().clamp_min(1.0) if positive_delta.numel() else positions.new_tensor(1.0)
                else:
                    boundary_width = positions.new_tensor(1.0)
                boundary = (torch.abs(positions - start) <= boundary_width) | (torch.abs(positions - end) <= boundary_width)
                boundary_target[batch_idx] = torch.where(
                    boundary,
                    torch.ones_like(boundary_target[batch_idx]),
                    boundary_target[batch_idx],
                )
        action_target = action_target.masked_fill(~valid_mask.bool(), 0.0)
        boundary_target = boundary_target.masked_fill(~valid_mask.bool(), 0.0)
        _require_finite(action_target, "selector dense action target")
        _require_finite(boundary_target, "selector dense boundary target")
        return action_target, boundary_target


__all__ = [
    "PCOTMRASPreBackboneFrameSelector",
    "PCOTMRASTinyTransformerFrameScout",
    "PCOTMRASCNNFrameScout",
    "PCOTMRASMotionTCNFrameScout",
    "PCOTMRASHybridFrameScout",
    "PCOTMRASBoundaryDifficultyTemporalFrameScout",
    "PCOTMRASLowResPixelTemporalFrameScout",
    "PCOTMRASLowResolutionPixelTemporalFrameScout",
    "PCOTMRASLowResPixelTemporalFrameReader",
    "PCOTMRASRSeriesHybridFrameScout",
    "PCOTMRASCoarseActionnessFrameScout",
]
