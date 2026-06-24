from __future__ import annotations

import math
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
class PCOTMRASPreBackboneFrameSelector(nn.Module):
    """Online PC-OT-MRAS frame acquisition before the video backbone.

    The selector consumes dense raw frames, emits a fixed-length selected-frame
    tensor for the unchanged AdaTAD/ActionFormer detector, remaps training GT to
    the selected axis, and mutates metas in-place so existing post-processing can
    map selected-axis predictions back to physical time.
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
        if str(selection_strategy) not in ("slot_transport", "frame_score_topk"):
            raise ValueError("selection_strategy must be 'slot_transport' or 'frame_score_topk'")
        if float(frame_score_st_temperature) <= 0.0:
            raise ValueError("frame_score_st_temperature must be positive")
        if float(frame_score_st_local_width) <= 0.0:
            raise ValueError("frame_score_st_local_width must be positive")
        if float(frame_score_st_local_bias_weight) < 0.0:
            raise ValueError("frame_score_st_local_bias_weight must be non-negative")
        self.st_surrogate_mode = str(st_surrogate_mode)
        self.scout_pixel_normalize = bool(scout_pixel_normalize)
        self.scout_pixel_clamp = float(scout_pixel_clamp)
        self.max_dense_gap = int(max_dense_gap)
        self.max_gap_guard_count = min(int(max_gap_guard_count), self.target_len)
        self.selection_strategy = str(selection_strategy)
        self.frame_score_st_temperature = float(frame_score_st_temperature)
        self.frame_score_st_local_width = float(frame_score_st_local_width)
        self.frame_score_st_local_bias_weight = float(frame_score_st_local_bias_weight)
        self.meta_source = str(meta_source)

    def forward_train(self, inputs, masks, metas, gt_segments, gt_labels):
        outputs = self._select(inputs=inputs, masks=masks, metas=metas, training=True)
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

    def _select(self, inputs, masks, metas, *, training: bool) -> dict[str, Any]:
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
            reader_outputs=reader_outputs,
            candidate_valid=candidate_valid,
            candidate_dense_indices=candidate_dense_indices,
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
        if getattr(self, "selection_strategy", "slot_transport") == "frame_score_topk":
            return self._frame_score_transport_plan(
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

    def _frame_score_transport_plan(
        self,
        *,
        reader_outputs: Mapping[str, torch.Tensor],
        valid: torch.Tensor,
        candidate_valid: torch.Tensor,
        candidate_dense_indices: torch.Tensor,
        training: bool,
    ) -> dict[str, torch.Tensor]:
        frame_scores = reader_outputs.get("frame_selection_logits")
        if frame_scores is None:
            frame_scores = reader_outputs.get("actionness_logits", reader_outputs.get("action_logits"))
        if frame_scores is None:
            raise ValueError("frame_score_topk selection requires frame_selection_logits or actionness/action logits")
        if not torch.is_tensor(frame_scores):
            raise TypeError("frame_score_topk frame scores must be a tensor")
        _require_finite(frame_scores, "frame_score_topk frame scores", error_type=ValueError)
        if tuple(candidate_dense_indices.shape) != tuple(candidate_valid.shape):
            raise ValueError("candidate_dense_indices must match candidate_valid")
        if tuple(frame_scores.shape) != tuple(candidate_valid.shape):
            raise ValueError(
                "frame_score_topk frame scores must match candidate axis; "
                f"got scores={tuple(frame_scores.shape)}, candidate_valid={tuple(candidate_valid.shape)}"
            )

        device = frame_scores.device
        frame_scores = frame_scores.float()
        candidate_valid = candidate_valid.to(device=device).bool()
        candidate_dense_indices = candidate_dense_indices.to(device=device)
        valid = valid.to(device=device).bool()
        if bool((candidate_valid.long().sum(dim=1) <= 0).any().item()):
            raise ValueError("each sample must contain at least one valid frame_score_topk candidate")

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

        min_score = torch.finfo(torch.float32).min
        masked_scores = frame_scores.masked_fill(~candidate_valid, min_score)
        _require_finite(masked_scores, "frame_score_topk masked scores")
        temperature = float(getattr(self, "frame_score_st_temperature", 1.0))
        local_width = float(getattr(self, "frame_score_st_local_width", 8.0))
        local_bias_weight = float(getattr(self, "frame_score_st_local_bias_weight", 1.0))

        for batch_idx in range(batch):
            valid_candidate_indices = torch.nonzero(candidate_valid[batch_idx], as_tuple=False).flatten()
            output_valid_len = min(int(valid_candidate_indices.numel()), self.target_len)
            selected_output_valid_lengths[batch_idx] = output_valid_len
            if output_valid_len <= 0:
                raise ValueError("frame_score_topk found no valid candidates for a sample")

            ranked_candidate_indices = torch.argsort(masked_scores[batch_idx], descending=True, stable=True)
            ranked_candidate_indices = ranked_candidate_indices[
                candidate_valid[batch_idx].gather(0, ranked_candidate_indices)
            ]
            selected_candidate_indices = ranked_candidate_indices[:output_valid_len]
            selected_dense_positions = candidate_dense_indices[batch_idx].gather(0, selected_candidate_indices)
            order = torch.argsort(selected_dense_positions, stable=True)
            selected_candidate_indices = selected_candidate_indices.gather(0, order)
            selected_dense_positions = selected_dense_positions.gather(0, order)

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

            batch_roles: list[str] = []
            for out_idx in range(self.target_len):
                if out_idx < output_valid_len:
                    candidate_idx = int(selected_candidate_indices[out_idx].item())
                    pos = int(selected_dense_positions[out_idx].item())
                    role = "frame_score_topk"
                else:
                    candidate_idx = int(selected_candidate_indices[-1].item())
                    pos = int(selected_dense_positions[-1].item())
                    role = "pad_repeat"

                fixed_positions[batch_idx, out_idx] = float(pos)
                fixed_indices[batch_idx, out_idx, 0] = pos
                fixed_weights[batch_idx, out_idx, 0] = 1.0
                batch_roles.append(role)

                hard = torch.zeros((dense_len,), dtype=torch.float32, device=device)
                hard[pos] = 1.0
                if self.straight_through_detector_loss and training and role == "frame_score_topk":
                    center = candidate_dense_indices[batch_idx, candidate_idx].to(dtype=torch.float32)
                    distances = candidate_dense_indices[batch_idx].to(dtype=torch.float32) - center
                    local_bias = -0.5 * (distances / local_width).square() * local_bias_weight
                    soft_logits = (frame_scores[batch_idx] / temperature + local_bias).masked_fill(
                        ~candidate_valid[batch_idx],
                        min_score,
                    )
                    _require_finite(soft_logits, "frame_score_topk straight-through logits")
                    soft_candidate = F.softmax(soft_logits, dim=0).masked_fill(~candidate_valid[batch_idx], 0.0)
                    soft_candidate = soft_candidate / soft_candidate.sum().clamp_min(torch.finfo(torch.float32).eps)
                    _require_finite(soft_candidate, "frame_score_topk straight-through distribution")
                    soft_dense = torch.zeros((dense_len,), dtype=torch.float32, device=device)
                    soft_dense.scatter_add_(0, candidate_dense_indices[batch_idx], soft_candidate)
                    transport_weights[batch_idx, out_idx] = hard + soft_dense - soft_dense.detach()
                else:
                    transport_weights[batch_idx, out_idx] = hard
            selected_roles.append(batch_roles)
            reader_fill_counts.append(0)
            st_active_row_counts.append(output_valid_len if training and self.straight_through_detector_loss else 0)

        _require_finite(fixed_weights, "frame_score_topk fixed weights")
        _require_finite(fixed_positions, "frame_score_topk selected positions")
        _require_finite(transport_weights, "frame_score_topk sparse transport weights")
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
        }

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
        reader_outputs: Mapping[str, torch.Tensor] | None = None,
        candidate_valid: torch.Tensor | None = None,
        candidate_dense_indices: torch.Tensor | None = None,
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
            meta["irregular_selected_output_valid_len"] = float(selected_valid_cpu[idx])
            meta["irregular_native_axis"] = False
            meta["pc_ot_mras_prebackbone_selected_dense_indices"] = dense_indices
            meta["pc_ot_mras_prebackbone_valid_len"] = int(valid_cpu[idx])
            meta["pc_ot_mras_prebackbone_gap"] = [
                int(right - left) for left, right in zip(prefix_indices[:-1], prefix_indices[1:])
            ]
            unique_count = len(set(prefix_indices))
            meta["pc_ot_mras_prebackbone_duplicate_rate"] = 1.0 - float(unique_count) / float(max(1, len(prefix_indices)))
            meta["pc_ot_mras_prebackbone_selection_unit"] = int(self.selection_unit)
            meta["pc_ot_mras_prebackbone_residual_count"] = (
                int(self.residual_count) if self.residual_count is not None else None
            )
            meta["pc_ot_mras_prebackbone_residual_slot_role"] = self.residual_slot_role
            meta["pc_ot_mras_prebackbone_selector_support_status"] = self.selector_support_status
            selection_strategy = getattr(self, "selection_strategy", "slot_transport")
            meta["pc_ot_mras_prebackbone_selection_strategy"] = selection_strategy
            meta["pc_ot_mras_prebackbone_hard_selection_source"] = (
                "frame_selection_logits" if selection_strategy == "frame_score_topk" else "slot_transport"
            )
            meta["pc_ot_mras_prebackbone_slot_not_hard_source"] = selection_strategy == "frame_score_topk"
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
                "uses_raw_prediction_cache": False,
                "uses_teacher": False,
                "uses_test_gt": False,
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
        return metas

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
        value_logits = reader_outputs.get("value_logits", reader_outputs.get("action_logits"))
        risk_logits = reader_outputs.get("risk_logits", reader_outputs.get("boundary_logits"))
        frame_selection_logits = reader_outputs.get("frame_selection_logits")
        uncertainty_logits = reader_outputs.get("uncertainty_logits")
        redundancy_logits = reader_outputs.get("redundancy_logits")
        role_logits = reader_outputs.get("role_logits")
        aux_tensors = (
            matrix,
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
            getattr(self, "selection_strategy", "slot_transport") == "frame_score_topk"
            and frame_selection_logits is not None
            and self.aux_gt_acquisition_loss_weight > 0.0
            and bool(valid.any().item())
        ):
            frame_score_loss = (
                F.binary_cross_entropy_with_logits(frame_selection_logits.float()[valid], action_target[valid])
                * self.aux_gt_acquisition_loss_weight
            )
            _require_finite(frame_score_loss, "selector gt frame score loss")
            losses["selector_gt_frame_score_loss"] = frame_score_loss
        elif matrix is not None and self.aux_gt_acquisition_loss_weight > 0.0 and bool(valid.any().item()):
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
]
