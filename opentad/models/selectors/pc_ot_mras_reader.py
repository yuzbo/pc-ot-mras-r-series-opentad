from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...ctf_bdi_role_constants import NUM_CTF_BDI_ROLES, ROLE_ID_TO_ROUND
from ..builder import SELECTORS
from .lowcost_acquisition_browser import DilatedTCNBlock


@dataclass(frozen=True)
class PCOTMRASReaderConfig:
    in_dim: int
    hidden_dim: int = 96
    num_slots: int = 64
    num_blocks: int = 4
    kernel_size: int = 5
    dropout: float = 0.10
    num_process_states: int = 7
    num_roles: int = NUM_CTF_BDI_ROLES
    temperature: float = 1.0
    min_width: float = 0.015
    max_width: float = 0.250
    order_margin: float = 1.0e-3
    column_cap: float = 2.0
    enable_value_heads: bool = False
    emit_pair_distribution: bool = True


def _neg(dtype: torch.dtype) -> float:
    return float(torch.finfo(dtype).min / 4.0)


def _mask_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~mask, _neg(logits.dtype))


def _validate_binary_mask(
    valid_mask: torch.Tensor,
    expected_shape: torch.Size,
    *,
    expected_device: torch.device,
) -> torch.Tensor:
    if valid_mask.ndim != 2:
        raise ValueError(f"valid_mask must be [B,T], got {tuple(valid_mask.shape)}")
    if valid_mask.shape != expected_shape:
        raise ValueError(f"valid_mask shape mismatch: {tuple(valid_mask.shape)} vs {tuple(expected_shape)}")
    if valid_mask.device != expected_device:
        raise ValueError("valid_mask must be on the same device as lowcost_features")
    if torch.is_complex(valid_mask):
        raise ValueError("valid_mask must contain binary 0/1 values")
    if torch.is_floating_point(valid_mask) and not torch.isfinite(valid_mask).all():
        raise ValueError("valid_mask must be finite")
    if not bool(torch.logical_or(valid_mask == 0, valid_mask == 1).all().item()):
        raise ValueError("valid_mask must be binary")
    valid = valid_mask.bool()
    if torch.any(valid.long().sum(dim=1) <= 0):
        raise ValueError("each sample must contain at least one valid position")
    valid_count = valid.long().sum(dim=1)
    prefix = torch.arange(valid.shape[1], device=valid.device)[None, :] < valid_count[:, None]
    if not torch.equal(valid, prefix):
        raise ValueError("valid_mask must be a contiguous valid prefix")
    return valid


def _default_time_coords(valid_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    batch, time = valid_mask.shape
    valid_len = valid_mask.long().sum(dim=1).clamp(min=1).to(dtype=dtype)
    pos = torch.arange(time, device=valid_mask.device, dtype=dtype)[None, :].expand(batch, -1)
    denom = (valid_len - 1.0).clamp(min=1.0)
    coords = pos / denom[:, None]
    return coords.masked_fill(~valid_mask.bool(), 0.0)


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    if mask.shape != logits.shape:
        raise ValueError(f"mask shape mismatch: {tuple(mask.shape)} vs {tuple(logits.shape)}")
    if mask.device != logits.device:
        raise ValueError("mask must be on the same device as logits")
    masked = _mask_logits(logits, mask)
    max_values = masked.max(dim=dim, keepdim=True).values
    exp = torch.exp(masked - max_values).masked_fill(~mask, 0.0)
    denom = exp.sum(dim=dim, keepdim=True).clamp_min(torch.finfo(logits.dtype).eps)
    return exp / denom


def _prob_entropy(prob: torch.Tensor, dim) -> torch.Tensor:
    prob_f = prob.float().clamp_min(1.0e-8)
    return -(prob_f * prob_f.log()).sum(dim=dim)


@SELECTORS.register_module()
class PCOTMRASReader(nn.Module):
    """Differentiable ordered-transport reader for local PC-OT-MRAS gates."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 96,
        num_slots: int = 64,
        num_blocks: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.10,
        num_process_states: int = 7,
        num_roles: int = NUM_CTF_BDI_ROLES,
        temperature: float = 1.0,
        min_width: float = 0.015,
        max_width: float = 0.250,
        order_margin: float = 1.0e-3,
        column_cap: float = 2.0,
        enable_value_heads: bool = False,
        emit_pair_distribution: bool = True,
    ) -> None:
        super().__init__()
        if int(in_dim) <= 0:
            raise ValueError("in_dim must be positive")
        if int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        if int(num_slots) <= 0:
            raise ValueError("num_slots must be positive")
        if int(num_blocks) <= 0:
            raise ValueError("num_blocks must be positive")
        if int(num_process_states) <= 0:
            raise ValueError("num_process_states must be positive")
        if int(num_roles) <= 0:
            raise ValueError("num_roles must be positive")
        if float(temperature) <= 0:
            raise ValueError("temperature must be positive")
        if not (0.0 < float(min_width) <= float(max_width)):
            raise ValueError("width range must satisfy 0 < min_width <= max_width")

        self.cfg = PCOTMRASReaderConfig(
            in_dim=int(in_dim),
            hidden_dim=int(hidden_dim),
            num_slots=int(num_slots),
            num_blocks=int(num_blocks),
            kernel_size=int(kernel_size),
            dropout=float(dropout),
            num_process_states=int(num_process_states),
            num_roles=int(num_roles),
            temperature=float(temperature),
            min_width=float(min_width),
            max_width=float(max_width),
            order_margin=float(order_margin),
            column_cap=float(column_cap),
            enable_value_heads=bool(enable_value_heads),
            emit_pair_distribution=bool(emit_pair_distribution),
        )

        self.input_proj = nn.Linear(self.cfg.in_dim, self.cfg.hidden_dim)
        self.blocks = nn.ModuleList(
            [
                DilatedTCNBlock(
                    dim=self.cfg.hidden_dim,
                    kernel_size=self.cfg.kernel_size,
                    dilation=2**idx,
                    dropout=self.cfg.dropout,
                )
                for idx in range(self.cfg.num_blocks)
            ]
        )
        self.head_norm = nn.LayerNorm(self.cfg.hidden_dim)

        self.key_proj = nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim)
        self.value_proj = nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim)
        self.global_proj = nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim)
        self.query_embed = nn.Parameter(torch.randn(self.cfg.num_slots, self.cfg.hidden_dim) * 0.02)

        self.center_inc_head = nn.Linear(self.cfg.hidden_dim, 1)
        self.center_shift_head = nn.Linear(self.cfg.hidden_dim, 1)
        self.width_head = nn.Linear(self.cfg.hidden_dim, 1)
        self.gate_head = nn.Linear(self.cfg.hidden_dim, 1)
        self.role_head = nn.Linear(self.cfg.hidden_dim, self.cfg.num_roles)
        allocation_signal_dim = 8 if self.cfg.enable_value_heads else 6
        self.role_bias_head = nn.Linear(self.cfg.hidden_dim, allocation_signal_dim)
        self.process_bias_head = nn.Linear(self.cfg.num_process_states, 1)

        self.process_head = nn.Linear(self.cfg.hidden_dim, self.cfg.num_process_states)
        self.start_head = nn.Linear(self.cfg.hidden_dim, 1)
        self.end_head = nn.Linear(self.cfg.hidden_dim, 1)
        self.boundary_head = nn.Linear(self.cfg.hidden_dim, 1)
        self.body_head = nn.Linear(self.cfg.hidden_dim, 1)
        self.uncertainty_head = nn.Linear(self.cfg.hidden_dim, 1)
        self.redundancy_head = nn.Linear(self.cfg.hidden_dim, 1)
        self.value_head = nn.Linear(self.cfg.hidden_dim, 1) if self.cfg.enable_value_heads else None
        self.risk_head = nn.Linear(self.cfg.hidden_dim, 1) if self.cfg.enable_value_heads else None
        self.pair_scorer = nn.Sequential(
            nn.LayerNorm(2 * self.cfg.hidden_dim + 1),
            nn.Linear(2 * self.cfg.hidden_dim + 1, self.cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(self.cfg.hidden_dim, 1),
        )

        role_round = torch.zeros((self.cfg.num_roles,), dtype=torch.long)
        for role_id, round_id in ROLE_ID_TO_ROUND.items():
            if int(role_id) < self.cfg.num_roles:
                role_round[int(role_id)] = int(round_id)
        self.register_buffer("role_id_to_round", role_round, persistent=False)

    def _validate_inputs(
        self,
        lowcost_features: torch.Tensor,
        valid_mask: torch.Tensor,
        time_coords: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if lowcost_features.ndim != 3:
            raise ValueError(f"lowcost_features must be [B,T,C], got {tuple(lowcost_features.shape)}")
        if lowcost_features.shape[-1] != self.cfg.in_dim:
            raise ValueError(f"expected feature dim {self.cfg.in_dim}, got {lowcost_features.shape[-1]}")
        if not torch.isfinite(lowcost_features).all():
            raise ValueError("lowcost_features must be finite")
        valid = _validate_binary_mask(
            valid_mask,
            lowcost_features.shape[:2],
            expected_device=lowcost_features.device,
        )
        if time_coords is None:
            coords = _default_time_coords(valid, dtype=lowcost_features.dtype)
        else:
            if time_coords.shape != valid.shape:
                raise ValueError("time_coords must be [B,T] and match valid_mask")
            if time_coords.device != lowcost_features.device:
                raise ValueError("time_coords must be on the same device as lowcost_features")
            if not torch.isfinite(time_coords).all():
                raise ValueError("time_coords must be finite")
            coords = time_coords.to(dtype=lowcost_features.dtype)
            valid_values = coords[valid]
            eps = torch.finfo(coords.dtype).eps * 8.0
            if valid_values.numel() and not bool(((valid_values >= -eps) & (valid_values <= 1.0 + eps)).all().item()):
                raise ValueError("time_coords valid prefix values must lie in [0, 1]")
            adjacent_valid = valid[:, 1:] & valid[:, :-1]
            if bool(adjacent_valid.any().item()):
                adjacent_delta = coords[:, 1:] - coords[:, :-1]
                if not bool((adjacent_delta[adjacent_valid] > 0).all().item()):
                    raise ValueError("time_coords valid prefix must be strictly increasing")
        coords = coords.masked_fill(~valid, 0.0)
        return valid, coords

    def _slot_state(self, h: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        denom = valid.long().sum(dim=1).clamp(min=1).to(dtype=h.dtype)
        pooled = h.sum(dim=1) / denom[:, None]
        global_state = self.global_proj(pooled)[:, None, :]
        return self.query_embed[None, :, :] + global_state

    def _ordered_centers_and_widths(self, slot_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        step_logits = self.center_inc_head(slot_state).squeeze(-1)
        step_logits = step_logits + 0.25 * torch.tanh(self.center_shift_head(slot_state).squeeze(-1))
        raw_steps = F.softplus(step_logits) + 1.0e-4
        centers = torch.cumsum(raw_steps, dim=1)
        centers = (centers - 0.5 * raw_steps) / centers[:, -1:].clamp_min(1.0e-6)
        widths = self.cfg.min_width + (self.cfg.max_width - self.cfg.min_width) * torch.sigmoid(
            self.width_head(slot_state).squeeze(-1)
        )
        return centers, widths

    def _dense_heads(self, h: torch.Tensor, valid: torch.Tensor) -> Dict[str, torch.Tensor]:
        valid_t = valid
        valid_ts = valid.unsqueeze(-1)
        dense = {
            "process_logits": _mask_logits(self.process_head(h), valid_ts),
            "start_logits": _mask_logits(self.start_head(h).squeeze(-1), valid_t),
            "end_logits": _mask_logits(self.end_head(h).squeeze(-1), valid_t),
            "boundary_logits": _mask_logits(self.boundary_head(h).squeeze(-1), valid_t),
            "body_logits": _mask_logits(self.body_head(h).squeeze(-1), valid_t),
            "uncertainty_logits": _mask_logits(self.uncertainty_head(h).squeeze(-1), valid_t),
            "redundancy_logits": _mask_logits(self.redundancy_head(h).squeeze(-1), valid_t),
        }
        if self.cfg.enable_value_heads:
            dense["value_logits"] = _mask_logits(self.value_head(h).squeeze(-1), valid_t)
            dense["risk_logits"] = _mask_logits(self.risk_head(h).squeeze(-1), valid_t)
        return dense

    def _allocation(
        self,
        h: torch.Tensor,
        valid: torch.Tensor,
        time_coords: torch.Tensor,
        slot_state: torch.Tensor,
        centers: torch.Tensor,
        widths: torch.Tensor,
        dense: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = self.key_proj(h)
        query = F.normalize(slot_state, dim=-1)
        key = F.normalize(key, dim=-1)
        dot = torch.einsum("bkd,btd->bkt", query, key) / self.cfg.temperature
        dist = (time_coords[:, None, :] - centers[:, :, None]) / widths[:, :, None].clamp_min(1.0e-6)
        local = -0.5 * dist.square()
        signal_items = [
            dense["start_logits"],
            dense["end_logits"],
            dense["boundary_logits"],
            dense["body_logits"],
            dense["uncertainty_logits"],
            -dense["redundancy_logits"],
        ]
        if self.cfg.enable_value_heads:
            signal_items.extend([dense["value_logits"], -dense["risk_logits"]])
        signal = torch.stack(signal_items, dim=-1).masked_fill(~valid.unsqueeze(-1), 0.0)
        role_bias = torch.einsum("bks,bts->bkt", torch.tanh(self.role_bias_head(slot_state)), signal)
        process_bias = self.process_bias_head(
            dense["process_logits"].masked_fill(~valid.unsqueeze(-1), 0.0)
        ).squeeze(-1)
        logits = dot + local + 0.1 * role_bias + 0.1 * process_bias[:, None, :]
        mask = valid[:, None, :].expand_as(logits)
        allocation = _masked_softmax(logits, mask, dim=-1)
        return _mask_logits(logits, mask), allocation.masked_fill(~mask, 0.0)

    def _pair_distribution(
        self,
        h: torch.Tensor,
        valid: torch.Tensor,
        time_coords: torch.Tensor,
        dense: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, time, hidden = h.shape
        start_feat = h[:, :, None, :].expand(batch, time, time, hidden)
        end_feat = h[:, None, :, :].expand(batch, time, time, hidden)
        raw_duration = time_coords[:, None, :] - time_coords[:, :, None]
        duration = raw_duration.clamp_min(0.0)
        pair_input = torch.cat([start_feat, end_feat, duration.unsqueeze(-1)], dim=-1)
        learned = self.pair_scorer(pair_input).squeeze(-1)
        logits = dense["start_logits"][:, :, None] + dense["end_logits"][:, None, :] + learned
        pair_mask = valid[:, :, None] & valid[:, None, :] & (raw_duration > 0.0)
        masked_logits = _mask_logits(logits, pair_mask)
        flat_logits = masked_logits.flatten(1)
        flat_mask = pair_mask.flatten(1)
        flat_prob = torch.zeros_like(flat_logits)
        has_pair = flat_mask.any(dim=1)
        if bool(has_pair.any().item()):
            pair_prob = _masked_softmax(flat_logits[has_pair], flat_mask[has_pair], dim=-1)
            flat_prob[has_pair] = pair_prob.to(dtype=flat_prob.dtype)
        pair_prob = flat_prob.view(batch, time, time).masked_fill(~pair_mask, 0.0)
        return masked_logits, pair_prob, pair_mask

    def _regularizers(
        self,
        allocation: torch.Tensor,
        centers: torch.Tensor,
        widths: torch.Tensor,
        gates: torch.Tensor,
        pair_prob: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        gap = centers[:, 1:] - centers[:, :-1]
        order_loss = F.relu(self.cfg.order_margin - gap).mean() if gap.numel() else centers.sum() * 0.0
        center_dist = torch.abs(centers[:, :, None] - centers[:, None, :])
        eye = torch.eye(centers.shape[1], device=centers.device, dtype=torch.bool)[None, :, :]
        diversity_loss = torch.exp(-center_dist / widths.mean(dim=1, keepdim=True)[:, None].clamp_min(1.0e-6))
        diversity_loss = diversity_loss.masked_fill(eye, 0.0).mean()
        entropy = _prob_entropy(allocation, dim=-1).mean()
        column_mass = allocation.sum(dim=1)
        column_cap_loss = F.relu(column_mass - self.cfg.column_cap).square().mean()
        budget_loss = gates.mean()
        width_loss = widths.mean()
        if pair_prob is None:
            pair_entropy = allocation.sum() * 0.0
        else:
            pair_entropy = _prob_entropy(pair_prob, dim=(1, 2)).mean()
        total = order_loss + 0.1 * diversity_loss + 0.01 * entropy + column_cap_loss + 0.01 * budget_loss + 0.01 * width_loss
        return {
            "order_loss": order_loss,
            "diversity_loss": diversity_loss,
            "entropy_loss": entropy,
            "column_cap_loss": column_cap_loss,
            "budget_loss": budget_loss,
            "width_loss": width_loss,
            "pair_entropy_loss": pair_entropy,
            "total_regularizer": total,
        }

    def forward(
        self,
        lowcost_features: torch.Tensor,
        valid_mask: torch.Tensor,
        time_coords: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:
        valid, coords = self._validate_inputs(lowcost_features, valid_mask, time_coords)

        valid_f = valid.unsqueeze(-1)
        x = self.input_proj(lowcost_features).masked_fill(~valid_f, 0.0)
        for block in self.blocks:
            x = block(x.masked_fill(~valid_f, 0.0)).masked_fill(~valid_f, 0.0)
        h = self.head_norm(x).masked_fill(~valid_f, 0.0)

        dense = self._dense_heads(h, valid)
        slot_state = self._slot_state(h, valid)
        centers, widths = self._ordered_centers_and_widths(slot_state)
        gate_logits = self.gate_head(slot_state).squeeze(-1)
        gates = torch.sigmoid(gate_logits)
        role_logits = self.role_head(slot_state)
        role_probs = F.softmax(role_logits, dim=-1)
        role_ids = role_probs.argmax(dim=-1)
        round_ids = self.role_id_to_round.to(device=role_ids.device)[role_ids.clamp(min=0, max=self.cfg.num_roles - 1)]

        allocation_logits, allocation = self._allocation(
            h=h,
            valid=valid,
            time_coords=coords,
            slot_state=slot_state,
            centers=centers,
            widths=widths,
            dense=dense,
        )
        acquisition_matrix = allocation * gates[:, :, None]
        values = self.value_proj(h)
        selected_tokens = torch.einsum("bkt,btd->bkd", acquisition_matrix, values)
        selected_times = torch.einsum("bkt,bt->bk", allocation, coords)
        dense_positions = torch.arange(coords.shape[1], device=coords.device, dtype=coords.dtype)
        acquisition_row_mass = acquisition_matrix.sum(dim=-1).clamp_min(torch.finfo(acquisition_matrix.dtype).eps)
        selected_positions = torch.einsum("bkt,t->bk", acquisition_matrix, dense_positions) / acquisition_row_mass
        pair_logits = pair_prob = pair_valid_mask = None
        if self.cfg.emit_pair_distribution:
            pair_logits, pair_prob, pair_valid_mask = self._pair_distribution(h, valid, coords, dense)
        regularizers = self._regularizers(allocation, centers, widths, gates, pair_prob)
        selected_mask = torch.ones(
            (lowcost_features.shape[0], self.cfg.num_slots),
            dtype=torch.bool,
            device=lowcost_features.device,
        )

        out: Dict[str, torch.Tensor | Dict[str, torch.Tensor]] = {
            "browser_memory": h,
            "valid_mask": valid,
            "valid_lengths": valid.long().sum(dim=1),
            "time_coords": coords,
            "slot_state": slot_state,
            "allocation_logits": allocation_logits,
            "allocation": allocation,
            "acquisition_matrix": acquisition_matrix,
            "selected_tokens": selected_tokens,
            "selected_times": selected_times,
            "selected_positions": selected_positions,
            "selected_mask": selected_mask,
            "centers": centers,
            "widths": widths,
            "gates": gates,
            "gate_logits": gate_logits,
            "role_logits": role_logits,
            "role_probs": role_probs,
            "role_ids": role_ids,
            "round_ids": round_ids,
            "regularizers": regularizers,
        }
        if pair_logits is not None and pair_prob is not None and pair_valid_mask is not None:
            out.update(
                {
                    "pair_logits": pair_logits,
                    "pair_prob": pair_prob,
                    "pair_valid_mask": pair_valid_mask,
                }
            )
        out.update(dense)
        return out


ProcessConditionedOrderedTransportMRASReader = PCOTMRASReader

__all__ = ["PCOTMRASReader", "ProcessConditionedOrderedTransportMRASReader"]
