from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn

from ..builder import SELECTORS


def _neg(dtype: torch.dtype) -> float:
    return float(torch.finfo(dtype).min / 4.0)


def _mask_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~mask, _neg(logits.dtype))


@dataclass(frozen=True)
class LowCostBrowserConfig:
    in_dim: int
    hidden_dim: int = 96
    num_blocks: int = 4
    kernel_size: int = 5
    dropout: float = 0.10
    budget_values: Tuple[int, ...] = (288, 320, 352, 384, 416)
    use_local_attention: bool = False


class DilatedTCNBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        if int(kernel_size) <= 0 or int(kernel_size) % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        padding = (int(kernel_size) - 1) * int(dilation) // 2
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, 2 * dim, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.proj = nn.Conv1d(dim, dim, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.norm(x).transpose(1, 2)
        y = self.conv(y)
        gate_value, gate = y.chunk(2, dim=1)
        y = gate_value * torch.sigmoid(gate)
        y = self.proj(y).transpose(1, 2)
        return residual + self.drop(y)


@SELECTORS.register_module()
class LowCostAcquisitionBrowser(nn.Module):
    """Low-cost temporal browser for latent acquisition decisions.

    The module consumes deploy-visible low-cost features and produces acquisition,
    boundary, and budget logits. It does not run the detector and does not consume
    labels or cached detector outputs.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 96,
        num_blocks: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.10,
        budget_values: Tuple[int, ...] = (288, 320, 352, 384, 416),
        use_local_attention: bool = False,
    ) -> None:
        super().__init__()
        if int(in_dim) <= 0:
            raise ValueError("in_dim must be positive")
        if int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        if int(num_blocks) <= 0:
            raise ValueError("num_blocks must be positive")
        if not tuple(int(value) for value in budget_values):
            raise ValueError("budget_values must be non-empty")
        self_attention_num_heads = 2
        if bool(use_local_attention) and int(hidden_dim) % self_attention_num_heads != 0:
            raise ValueError("hidden_dim must be divisible by 2 when use_local_attention=True")

        self.cfg = LowCostBrowserConfig(
            in_dim=int(in_dim),
            hidden_dim=int(hidden_dim),
            num_blocks=int(num_blocks),
            kernel_size=int(kernel_size),
            dropout=float(dropout),
            budget_values=tuple(int(value) for value in budget_values),
            use_local_attention=bool(use_local_attention),
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
        if self.cfg.use_local_attention:
            # Compatibility flag: this branch uses full-sequence self-attention,
            # not a windowed local-attention implementation.
            self.full_attn_norm = nn.LayerNorm(self.cfg.hidden_dim)
            self.full_attn = nn.MultiheadAttention(
                embed_dim=self.cfg.hidden_dim,
                num_heads=self_attention_num_heads,
                dropout=self.cfg.dropout,
                batch_first=True,
            )
        else:
            self.full_attn_norm = None
            self.full_attn = None

        self.head_norm = nn.LayerNorm(self.cfg.hidden_dim)
        self.acq_head = nn.Linear(self.cfg.hidden_dim, 1)
        self.start_head = nn.Linear(self.cfg.hidden_dim, 1)
        self.end_head = nn.Linear(self.cfg.hidden_dim, 1)
        self.boundary_head = nn.Linear(self.cfg.hidden_dim, 1)
        self.budget_head = nn.Sequential(
            nn.LayerNorm(self.cfg.hidden_dim),
            nn.Linear(self.cfg.hidden_dim, len(self.cfg.budget_values)),
        )

    def forward(self, lowcost_features: torch.Tensor, valid_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        if lowcost_features.ndim != 3:
            raise ValueError(f"lowcost_features must be [B,T,D], got {tuple(lowcost_features.shape)}")
        if valid_mask.ndim != 2:
            raise ValueError(f"valid_mask must be [B,T], got {tuple(valid_mask.shape)}")
        if lowcost_features.shape[:2] != valid_mask.shape:
            raise ValueError(
                "lowcost_features and valid_mask shape mismatch: "
                f"{tuple(lowcost_features.shape[:2])} vs {tuple(valid_mask.shape)}"
            )
        if lowcost_features.shape[-1] != self.cfg.in_dim:
            raise ValueError(f"expected feature dim {self.cfg.in_dim}, got {lowcost_features.shape[-1]}")
        if not torch.isfinite(lowcost_features).all():
            raise ValueError("lowcost_features must be finite")
        if not torch.isfinite(valid_mask).all():
            raise ValueError("valid_mask must be finite")
        if valid_mask.dtype != torch.bool:
            if not torch.logical_or(valid_mask == 0, valid_mask == 1).all():
                raise ValueError("valid_mask must be binary")
        valid = valid_mask.bool()
        if torch.any(valid.long().sum(dim=1) <= 0):
            raise ValueError("each sample must contain at least one valid position")
        valid_count = valid.long().sum(dim=1)
        prefix = torch.arange(valid.shape[1], device=valid.device)[None, :] < valid_count[:, None]
        if not torch.equal(valid, prefix):
            raise ValueError("valid_mask must be a contiguous valid prefix")

        valid_f = valid.unsqueeze(-1)
        x = self.input_proj(lowcost_features).masked_fill(~valid_f, 0.0)
        for block in self.blocks:
            x = block(x.masked_fill(~valid_f, 0.0)).masked_fill(~valid_f, 0.0)

        if self.full_attn is not None and self.full_attn_norm is not None:
            q = self.full_attn_norm(x).masked_fill(~valid_f, 0.0)
            y, _ = self.full_attn(q, q, q, key_padding_mask=~valid, need_weights=False)
            x = (x + y).masked_fill(~valid_f, 0.0)

        h = self.head_norm(x).masked_fill(~valid_f, 0.0)
        acq_logits = _mask_logits(self.acq_head(h).squeeze(-1), valid)
        start_logits = _mask_logits(self.start_head(h).squeeze(-1), valid)
        end_logits = _mask_logits(self.end_head(h).squeeze(-1), valid)
        boundary_logits = _mask_logits(self.boundary_head(h).squeeze(-1), valid)

        pooled = (h * valid.unsqueeze(-1).to(dtype=h.dtype)).sum(dim=1)
        pooled = pooled / valid.long().sum(dim=1, keepdim=True).clamp_min(1).to(dtype=h.dtype)
        budget_logits = self.budget_head(pooled)

        return {
            "acq_logits": acq_logits,
            "start_logits": start_logits,
            "end_logits": end_logits,
            "boundary_logits": boundary_logits,
            "budget_logits": budget_logits,
            "browser_feats": h,
        }
