# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import math
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor, nn
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmcv.cnn.bricks.transformer import FFN, PatchEmbed
from mmengine.registry import MODELS
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import constant_init, trunc_normal_init
from mmaction.utils import ConfigType, OptConfigType
from mmaction.models.backbones.vit_mae import get_sinusoid_encoding


class Adapter(BaseModule):
    def __init__(
        self,
        embed_dims: int,
        mlp_ratio: float = 0.25,
        kernel_size: int = 3,
        dilation: int = 1,
        temporal_size: int = 384,
    ) -> None:
        super().__init__()

        hidden_dims = int(embed_dims * mlp_ratio)

        # temporal depth-wise convolution
        self.temporal_size = temporal_size
        self.dwconv = nn.Conv1d(
            hidden_dims,
            hidden_dims,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size // 2) * dilation,
            dilation=dilation,
            groups=hidden_dims,
        )
        self.conv = nn.Conv1d(hidden_dims, hidden_dims, 1)
        self.dwconv.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / kernel_size))
        self.dwconv.bias.data.zero_()
        self.conv.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / hidden_dims))
        self.conv.bias.data.zero_()

        # adapter projection
        self.down_proj = nn.Linear(embed_dims, hidden_dims)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(hidden_dims, embed_dims)
        self.gamma = nn.Parameter(torch.ones(1))
        trunc_normal_init(self.down_proj, std=0.02, bias=0)
        constant_init(self.up_proj, 0)  # the last projection layer is initialized to 0

    def forward(self, x: Tensor, h: int, w: int) -> Tensor:
        inputs = x

        # down and up projection
        x = self.down_proj(x)
        x = self.act(x)

        # temporal depth-wise convolution
        B, N, C = x.shape  # 48, 8*10*10, 384
        attn = x.reshape(-1, self.temporal_size, h, w, x.shape[-1])  # [b,t,h,w,c]  [1,384,10,10,384]
        attn = attn.permute(0, 2, 3, 4, 1).flatten(0, 2)  # [b*h*w,c,t] [1*10*10,384,384]
        attn = self.dwconv(attn)  # [b*h*w,c,t] [1*10*10,384,384]
        attn = self.conv(attn)  # [b*h*w,c,t] [1*10*10,384,384]
        attn = attn.unflatten(0, (-1, h, w)).permute(0, 4, 1, 2, 3)  # [b,t,h,w,c] [1,384,10,10,384]
        attn = attn.reshape(B, N, C)
        x = x + attn

        x = self.up_proj(x)
        return x * self.gamma + inputs


class PlainAdapter(BaseModule):
    def __init__(
        self,
        embed_dims: int,
        mlp_ratio: float = 0.25,
        **kwargs,
    ) -> None:
        super().__init__()

        hidden_dims = int(embed_dims * mlp_ratio)

        # adapter projection
        self.down_proj = nn.Linear(embed_dims, hidden_dims)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(hidden_dims, embed_dims)
        self.gamma = nn.Parameter(torch.ones(1))
        trunc_normal_init(self.down_proj, std=0.02, bias=0)
        constant_init(self.up_proj, 0)  # the last projection layer is initialized to 0

    def forward(self, x: Tensor, h: int, w: int) -> Tensor:
        inputs = x

        # down and up projection
        x = self.down_proj(x)
        x = self.act(x)
        x = self.up_proj(x)
        return x * self.gamma + inputs


class TubeletTokenRedundancyAux(BaseModule):
    """Local-only tubelet/token redundancy auditor for VideoMAE tokens.

    The first R28 use is deliberately non-destructive: it scores temporal
    tubelet groups using their spatial-token statistics, records a proposed
    tubelet keep mask, and returns the dense token sequence unchanged.
    """

    valid_modes = ("identity", "shadow", "deterministic_tubelet_cap")

    def __init__(
        self,
        enabled: bool = False,
        mode: str = "identity",
        route_unit: str = "temporal_tubelet_group",
        keep_ratio: float = 1.0,
        min_keep_tubelets: int = 1,
        route_pattern: str = "round_linspace",
        spatial_pool: str = "energy_std",
        forbid_spatial_crop: bool = True,
        local_audit_only: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        if mode not in self.valid_modes:
            raise ValueError(f"unsupported tubelet redundancy mode: {mode}")
        if route_unit != "temporal_tubelet_group":
            raise ValueError("R28 v0 only supports temporal_tubelet_group routing")
        if route_pattern != "round_linspace":
            raise ValueError("R28 v0 only supports round_linspace route_pattern")
        keep_ratio = float(keep_ratio)
        if keep_ratio <= 0.0 or keep_ratio > 1.0:
            raise ValueError("keep_ratio must lie in (0, 1]")
        if int(min_keep_tubelets) <= 0:
            raise ValueError("min_keep_tubelets must be positive")
        if not bool(forbid_spatial_crop):
            raise ValueError("R28 v0 forbids spatial patch crop")
        if not bool(local_audit_only):
            raise ValueError("R28 v0 is local_audit_only")

        self.enabled = bool(enabled)
        self.mode = mode
        self.route_unit = route_unit
        self.keep_ratio = keep_ratio
        self.min_keep_tubelets = int(min_keep_tubelets)
        self.route_pattern = route_pattern
        self.spatial_pool = spatial_pool
        self.forbid_spatial_crop = bool(forbid_spatial_crop)
        self.local_audit_only = bool(local_audit_only)
        self.last_summary = None

    @staticmethod
    def _round_linspace_indices(length: int, count: int, device: torch.device) -> Tensor:
        length = int(length)
        count = min(int(count), length)
        if count <= 0 or length <= 0:
            raise ValueError("length and count must be positive")
        if count >= length:
            return torch.arange(length, device=device, dtype=torch.long)
        if count == 1:
            return torch.zeros(1, device=device, dtype=torch.long)

        selected: List[int] = []
        for idx in range(count):
            pos = int(round(float(idx) * float(length - 1) / float(count - 1)))
            if pos not in selected:
                selected.append(pos)
        filler = 0
        while len(selected) < count:
            if filler not in selected:
                selected.append(filler)
            filler += 1
        return torch.tensor(sorted(selected[:count]), device=device, dtype=torch.long)

    def _shape_contract(self, x: Tensor, h: int, w: int) -> Tuple[int, int]:
        if x.ndim != 3:
            raise ValueError("tubelet redundancy aux expects x with shape [B, N, C]")
        spatial_tokens = int(h) * int(w)
        if spatial_tokens <= 0:
            raise ValueError("spatial token count must be positive")
        if int(x.shape[1]) % spatial_tokens != 0:
            raise ValueError("token length must be divisible by h*w")
        temporal_tubelets = int(x.shape[1]) // spatial_tokens
        if temporal_tubelets <= 0:
            raise ValueError("temporal tubelet count must be positive")
        return temporal_tubelets, spatial_tokens

    def summarize(self, x: Tensor, h: int, w: int) -> Dict[str, object]:
        temporal_tubelets, spatial_tokens = self._shape_contract(x, h, w)
        keep_count = max(
            self.min_keep_tubelets,
            int(math.ceil(float(temporal_tubelets) * float(self.keep_ratio))),
        )
        keep_count = min(keep_count, temporal_tubelets)

        token_energy = x.detach().float().pow(2).mean(dim=-1)
        token_energy = token_energy.reshape(int(x.shape[0]), temporal_tubelets, spatial_tokens)
        spatial_energy_mean = token_energy.mean(dim=-1)
        spatial_energy_std = token_energy.std(dim=-1, unbiased=False)
        redundancy_score = 1.0 / (1.0 + spatial_energy_std)

        proposed_mask = torch.zeros(
            (int(x.shape[0]), temporal_tubelets),
            dtype=torch.bool,
            device=x.device,
        )
        if self.mode in ("identity", "shadow"):
            proposed_mask[:] = True
        else:
            keep_idx = self._round_linspace_indices(temporal_tubelets, keep_count, x.device)
            proposed_mask[:, keep_idx] = True

        return {
            "schema_version": "tubelet_token_redundancy_aux_summary_v0",
            "enabled": self.enabled,
            "mode": self.mode,
            "route_unit": self.route_unit,
            "route_pattern": self.route_pattern,
            "local_audit_only": self.local_audit_only,
            "spatial_patch_crop_allowed": False,
            "spatial_crop_forbidden": self.forbid_spatial_crop,
            "dense_output_preserved": True,
            "compute_route_applied": False,
            "runtime_flops_claim_allowed": False,
            "batch_size": int(x.shape[0]),
            "token_length": int(x.shape[1]),
            "channels": int(x.shape[2]),
            "temporal_tubelets": temporal_tubelets,
            "spatial_tokens_per_tubelet": spatial_tokens,
            "keep_ratio": float(self.keep_ratio),
            "proposed_keep_count": int(proposed_mask[0].sum().item()),
            "effective_dense_token_count": int(x.shape[1]),
            "proposed_tubelet_keep_mask": proposed_mask,
            "spatial_energy_mean": spatial_energy_mean,
            "spatial_energy_std": spatial_energy_std,
            "redundancy_score": redundancy_score,
        }

    def forward(self, x: Tensor, h: int, w: int) -> Tensor:
        if not self.enabled:
            self.last_summary = None
            return x
        self.last_summary = self.summarize(x, h, w)
        return x


class PackedTubeletRuntimeRoute(BaseModule):
    """Opt-in packed temporal-tubelet execution inside ViT forward.

    This route is intentionally narrow. It only handles temporal-tubelet groups
    and keeps all spatial patches within a selected tubelet. Packed execution
    is applied inside each transformer block's attention/MLP subpath, then the
    selected outputs are scattered back before Adapter convolution sees the
    tensor. It is disabled by default and exists to make the R30 packed-runtime
    proof executable through the production backbone forward path.
    """

    valid_modes = ("deterministic_tubelet_cap",)
    valid_scatter_modes = ("identity", "zero")

    def __init__(
        self,
        enabled: bool = False,
        mode: str = "deterministic_tubelet_cap",
        route_unit: str = "temporal_tubelet_group",
        keep_ratio: float = 1.0,
        min_keep_tubelets: int = 1,
        route_pattern: str = "round_linspace",
        forbid_spatial_crop: bool = True,
        local_forward_only: bool = True,
        require_no_adapter_blocks: bool = False,
        allow_training_mode: bool = False,
        scatter_unselected: str = "identity",
        **kwargs,
    ) -> None:
        super().__init__()
        if mode not in self.valid_modes:
            raise ValueError(f"unsupported packed tubelet route mode: {mode}")
        if route_unit != "temporal_tubelet_group":
            raise ValueError("packed tubelet route only supports temporal_tubelet_group")
        if route_pattern != "round_linspace":
            raise ValueError("packed tubelet route only supports round_linspace route_pattern")
        keep_ratio = float(keep_ratio)
        if keep_ratio <= 0.0 or keep_ratio > 1.0:
            raise ValueError("keep_ratio must lie in (0, 1]")
        if int(min_keep_tubelets) <= 0:
            raise ValueError("min_keep_tubelets must be positive")
        if not bool(forbid_spatial_crop):
            raise ValueError("packed tubelet route forbids spatial patch crop")
        if not bool(local_forward_only):
            raise ValueError("packed tubelet route is local_forward_only")
        if scatter_unselected not in self.valid_scatter_modes:
            raise ValueError(f"unsupported packed tubelet scatter mode: {scatter_unselected}")

        self.enabled = bool(enabled)
        self.mode = mode
        self.route_unit = route_unit
        self.keep_ratio = keep_ratio
        self.min_keep_tubelets = int(min_keep_tubelets)
        self.route_pattern = route_pattern
        self.forbid_spatial_crop = bool(forbid_spatial_crop)
        self.local_forward_only = bool(local_forward_only)
        self.require_no_adapter_blocks = bool(require_no_adapter_blocks)
        self.allow_training_mode = bool(allow_training_mode)
        self.scatter_unselected = scatter_unselected
        self.last_summary = None

    @staticmethod
    def _shape_contract(x: Tensor, h: int, w: int) -> Tuple[int, int]:
        if x.ndim != 3:
            raise ValueError("packed tubelet route expects x with shape [B, N, C]")
        spatial_tokens = int(h) * int(w)
        if spatial_tokens <= 0:
            raise ValueError("spatial token count must be positive")
        if int(x.shape[1]) % spatial_tokens != 0:
            raise ValueError("token length must be divisible by h*w")
        temporal_tubelets = int(x.shape[1]) // spatial_tokens
        if temporal_tubelets <= 0:
            raise ValueError("temporal tubelet count must be positive")
        return temporal_tubelets, spatial_tokens

    @staticmethod
    def _round_linspace_indices(length: int, count: int, device: torch.device) -> Tensor:
        return TubeletTokenRedundancyAux._round_linspace_indices(length, count, device)

    def _tubelet_mask(self, x: Tensor, h: int, w: int) -> Tuple[Tensor, int, int]:
        temporal_tubelets, spatial_tokens = self._shape_contract(x, h, w)
        keep_count = max(
            self.min_keep_tubelets,
            int(math.ceil(float(temporal_tubelets) * float(self.keep_ratio))),
        )
        keep_count = min(keep_count, temporal_tubelets)
        keep_idx = self._round_linspace_indices(temporal_tubelets, keep_count, x.device)
        tubelet_mask = torch.zeros(
            (int(x.shape[0]), temporal_tubelets),
            dtype=torch.bool,
            device=x.device,
        )
        tubelet_mask[:, keep_idx] = True
        return tubelet_mask, temporal_tubelets, spatial_tokens

    @staticmethod
    def _expand_tubelet_mask(tubelet_mask: Tensor, spatial_tokens: int) -> Tensor:
        if tubelet_mask.ndim != 2:
            raise ValueError("tubelet_mask must have shape [B, T]")
        return tubelet_mask.unsqueeze(-1).expand(-1, -1, int(spatial_tokens)).reshape(
            int(tubelet_mask.shape[0]),
            int(tubelet_mask.shape[1]) * int(spatial_tokens),
        )

    @staticmethod
    def _pack_tokens(x: Tensor, dense_mask: Tensor) -> Tensor:
        if dense_mask.shape != x.shape[:2]:
            raise ValueError("dense_mask must match x batch/token shape")
        per_batch_counts = dense_mask.sum(dim=1)
        if int(per_batch_counts.min().item()) <= 0:
            raise ValueError("packed tubelet route requires at least one selected token per sample")
        if not torch.equal(per_batch_counts, per_batch_counts[:1].expand_as(per_batch_counts)):
            raise ValueError("packed tubelet route requires equal selected-token count across batch")
        return x[dense_mask].reshape(int(x.shape[0]), int(per_batch_counts[0].item()), int(x.shape[2]))

    def _scatter_tokens(self, base: Tensor, packed: Tensor, dense_mask: Tensor) -> Tensor:
        if self.scatter_unselected == "identity":
            scattered = base.clone()
        elif self.scatter_unselected == "zero":
            scattered = packed.new_zeros(base.shape)
        else:
            raise ValueError(f"unsupported packed tubelet scatter mode: {self.scatter_unselected}")
        scattered[dense_mask] = packed.reshape(-1, int(packed.shape[-1]))
        return scattered

    @staticmethod
    def _run_blocks(blocks, x: Tensor, h: int, w: int, dense_mask: Tensor, stats: Dict[str, int]) -> Tensor:
        out = x
        for block in blocks:
            if bool(getattr(block, "use_adapter", False)):
                stats["adapter_forward_count"] += 1
            out = block(out, h, w, packed_dense_mask=dense_mask, packed_stats=stats)
        return out

    def forward(self, x: Tensor, blocks, h: int, w: int, *, training: bool = False) -> Tensor:
        if not self.enabled:
            self.last_summary = None
            return x
        if bool(training) and not self.allow_training_mode:
            raise ValueError("packed tubelet route is local-forward-only and forbids training mode")
        adapter_block_count = sum(1 for block in blocks if bool(getattr(block, "use_adapter", False)))
        if self.require_no_adapter_blocks and adapter_block_count:
            raise ValueError("packed tubelet route requires adapter-free blocks for R31")

        tubelet_mask, temporal_tubelets, spatial_tokens = self._tubelet_mask(x, h, w)
        dense_mask = self._expand_tubelet_mask(tubelet_mask, spatial_tokens)
        packed = self._pack_tokens(x, dense_mask)
        stats = {
            "packed_attention_forward_count": 0,
            "packed_mlp_forward_count": 0,
            "adapter_forward_count": 0,
        }
        out = self._run_blocks(blocks, x, h, w, dense_mask, stats)

        selected_output = out[dense_mask].reshape(int(out.shape[0]), int(packed.shape[1]), int(out.shape[2]))
        selected_output_finite = bool(torch.isfinite(selected_output).all().item())
        scattered_output_finite = bool(torch.isfinite(out).all().item())
        if self.scatter_unselected == "identity" and adapter_block_count == 0:
            unselected_identity = bool(torch.allclose(out[~dense_mask], x[~dense_mask]))
        else:
            unselected_identity = None

        self.last_summary = {
            "schema_version": "packed_tubelet_runtime_route_summary_v0",
            "enabled": True,
            "mode": self.mode,
            "route_unit": self.route_unit,
            "route_pattern": self.route_pattern,
            "local_forward_only": self.local_forward_only,
            "production_forward_changed": True,
            "training_mode_allowed": self.allow_training_mode,
            "adapter_blocks_supported": not self.require_no_adapter_blocks,
            "adapter_block_count": int(adapter_block_count),
            "adapter_dense_contract_preserved": not self.require_no_adapter_blocks,
            "dense_scatter_before_adapter": bool(adapter_block_count and not self.require_no_adapter_blocks),
            "spatial_patch_crop_allowed": False,
            "spatial_filtering_allowed": False,
            "arbitrary_spatial_patch_filtering_allowed": False,
            "scatter_unselected": self.scatter_unselected,
            "dense_token_shape": list(x.shape),
            "packed_token_shape": list(packed.shape),
            "dense_output_shape": list(out.shape),
            "selected_output_shape": list(selected_output.shape),
            "temporal_tubelets": int(temporal_tubelets),
            "spatial_tokens_per_tubelet": int(spatial_tokens),
            "selected_tubelets": int(tubelet_mask[0].sum().item()),
            "selected_dense_tokens": int(dense_mask[0].sum().item()),
            "has_strict_token_saving": bool(packed.shape[1] < x.shape[1]),
            "true_packed_compute_enabled": True,
            "packed_attention_executed_in_forward": True,
            "packed_mlp_executed_in_forward": True,
            "scatter_back_executed": True,
            "packed_attention_forward_count": int(stats["packed_attention_forward_count"]),
            "packed_mlp_forward_count": int(stats["packed_mlp_forward_count"]),
            "adapter_forward_count": int(stats["adapter_forward_count"]),
            "unselected_positions_identity_bypass_without_adapter": unselected_identity,
            "selected_output_finite": selected_output_finite,
            "scattered_output_finite": scattered_output_finite,
            "measured_runtime": False,
            "measured_flops": False,
            "runtime_flops_claim_allowed": False,
            "metric_claim_allowed": False,
            "paper_claim_allowed": False,
        }
        return out


class Attention(BaseModule):
    """Multi-head Self-attention.

    Args:
        embed_dims (int): Dimensions of embedding.
        num_heads (int): Number of parallel attention heads.
        qkv_bias (bool): If True, add a learnable bias to q and v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        attn_drop_rate (float): Dropout ratio of attention weight.
            Defaults to 0.
        drop_rate (float): Dropout ratio of output. Defaults to 0.
        init_cfg (dict or ConfigDict, optional): The Config
            for initialization. Defaults to None.
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop_rate: float = 0.0,
        drop_rate: float = 0.0,
        init_cfg: OptConfigType = None,
        **kwargs,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads

        self.scale = qk_scale or head_embed_dims**-0.5

        if qkv_bias:
            self._init_qv_bias()

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(drop_rate)

    def _init_qv_bias(self) -> None:
        self.q_bias = nn.Parameter(torch.zeros(self.embed_dims))
        self.v_bias = nn.Parameter(torch.zeros(self.embed_dims))

    def forward(self, x: Tensor) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input data with size of (B, N, C).
        Returns:
            Tensor: The output of the attention block, same size as inputs.
        """
        B, N, C = x.shape

        if hasattr(self, "q_bias"):
            k_bias = torch.zeros_like(self.v_bias, requires_grad=False)
            qkv_bias = torch.cat((self.q_bias, k_bias, self.v_bias))
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        else:
            qkv = self.qkv(x)

        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # standard self-attention
        # q = q * self.scale
        # attn = q @ k.transpose(-2, -1)
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        # x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        # fast attention
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
        x = x.transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(BaseModule):
    """The basic block in the Vision Transformer.

    Args:
        embed_dims (int): Dimensions of embedding.
        num_heads (int): Number of parallel attention heads.
        mlp_ratio (int): The ratio between the hidden layer and the
            input layer in the FFN. Defaults to 4.
        qkv_bias (bool): If True, add a learnable bias to q and v.
            Defaults to True.
        qk_scale (float): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        drop_rate (float): Dropout ratio of output. Defaults to 0.
        attn_drop_rate (float): Dropout ratio of attention weight.
            Defaults to 0.
        drop_path_rate (float): Dropout ratio of the residual branch.
            Defaults to 0.
        act_cfg (dict or ConfigDict): Config for activation layer in FFN.
            Defaults to `dict(type='GELU')`.
        norm_cfg (dict or ConfigDict): Config for norm layers.
            Defaults to `dict(type='LN', eps=1e-6)`.
        init_cfg (dict or ConfigDict, optional): The Config
            for initialization. Defaults to None.
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        mlp_ratio: int = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        act_cfg: ConfigType = dict(type="GELU"),
        norm_cfg: ConfigType = dict(type="LN", eps=1e-6),
        init_cfg: OptConfigType = None,
        with_cp: bool = False,
        use_adapter: bool = False,
        adapter_mlp_ratio: float = 0.25,
        temporal_size: int = 384,
        **kwargs,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.with_cp = with_cp
        self.use_adapter = use_adapter

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = Attention(
            embed_dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            drop_rate=drop_rate,
        )

        self.drop_path = nn.Identity()
        if drop_path_rate > 0.0:
            self.drop_path = DropPath(drop_path_rate)
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        mlp_hidden_dim = int(embed_dims * mlp_ratio)
        self.mlp = FFN(
            embed_dims=embed_dims,
            feedforward_channels=mlp_hidden_dim,
            act_cfg=act_cfg,
            ffn_drop=drop_rate,
            add_identity=False,
        )

        if self.use_adapter:
            self.adapter = Adapter(
                embed_dims=embed_dims,
                kernel_size=3,
                dilation=1,
                temporal_size=temporal_size,
                mlp_ratio=adapter_mlp_ratio,
            )

    @staticmethod
    def _pack_selected_tokens(x: Tensor, dense_mask: Tensor) -> Tensor:
        if dense_mask.shape != x.shape[:2]:
            raise ValueError("packed_dense_mask must match x batch/token shape")
        per_batch_counts = dense_mask.sum(dim=1)
        if int(per_batch_counts.min().item()) <= 0:
            raise ValueError("packed block path requires at least one selected token per sample")
        if not torch.equal(per_batch_counts, per_batch_counts[:1].expand_as(per_batch_counts)):
            raise ValueError("packed block path requires equal selected-token count across batch")
        return x[dense_mask].reshape(int(x.shape[0]), int(per_batch_counts[0].item()), int(x.shape[2]))

    def _packed_attention_mlp_forward(
        self,
        x: Tensor,
        dense_mask: Tensor,
        packed_stats: Optional[Dict[str, int]],
    ) -> Tensor:
        selected = self._pack_selected_tokens(x, dense_mask)
        selected = selected + self.drop_path(self.attn(self.norm1(selected)))
        selected = selected + self.drop_path(self.mlp(self.norm2(selected)))
        if packed_stats is not None:
            packed_stats["packed_attention_forward_count"] = int(packed_stats.get("packed_attention_forward_count", 0)) + 1
            packed_stats["packed_mlp_forward_count"] = int(packed_stats.get("packed_mlp_forward_count", 0)) + 1
        out = x.clone()
        out[dense_mask] = selected.reshape(-1, int(selected.shape[-1]))
        return out

    def forward(
        self,
        x: Tensor,
        h,
        w,
        packed_dense_mask: Optional[Tensor] = None,
        packed_stats: Optional[Dict[str, int]] = None,
    ) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input data with size of (B, N, C).
        Returns:
            Tensor: The output of the transformer block, same size as inputs.
        """

        def _inner_forward(x):
            """Forward wrapper for utilizing checkpoint."""
            if packed_dense_mask is None:
                x = x + self.drop_path(self.attn(self.norm1(x)))
                x = x + self.drop_path(self.mlp(self.norm2(x)))
            else:
                x = self._packed_attention_mlp_forward(x, packed_dense_mask, packed_stats)

            if self.use_adapter:
                x = self.adapter(x, h, w)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


@MODELS.register_module()
class VisionTransformerAdapter(BaseModule):
    """Vision Transformer with support for patch or hybrid CNN input stage. An
    impl of `VideoMAE: Masked Autoencoders are Data-Efficient Learners for
    Self-Supervised Video Pre-Training <https://arxiv.org/pdf/2203.12602.pdf>`_

    We add the checkpointing and frozen stage to the original VisionTransformer.

    Args:
        img_size (int or tuple): Size of input image.
            Defaults to 224.
        patch_size (int): Spatial size of one patch. Defaults to 16.
        in_channels (int): The number of channels of he input.
            Defaults to 3.
        embed_dims (int): Dimensions of embedding. Defaults to 768.
        depth (int): number of blocks in the transformer.
            Defaults to 12.
        num_heads (int): Number of parallel attention heads in
            TransformerCoder. Defaults to 12.
        mlp_ratio (int): The ratio between the hidden layer and the
            input layer in the FFN. Defaults to 4.
        qkv_bias (bool): If True, add a learnable bias to q and v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        drop_rate (float): Dropout ratio of output. Defaults to 0.
        attn_drop_rate (float): Dropout ratio of attention weight.
            Defaults to 0.
        drop_path_rate (float): Dropout ratio of the residual branch.
            Defaults to 0.
        norm_cfg (dict or Configdict): Config for norm layers.
            Defaults to `dict(type='LN', eps=1e-6)`.
        num_frames (int): Number of frames in the video. Defaults to 16.
        tubelet_size (int): Temporal size of one patch. Defaults to 2.
        use_mean_pooling (bool): If True, take the mean pooling over all
            positions. Defaults to True.
        pretrained (str, optional): Name of pretrained model. Default: None.
        return_feat_map (bool): If True, return the feature in the shape of
            `[B, C, T, H, W]`. Defaults to False.
        init_cfg (dict or list[dict]): Initialization config dict. Defaults to
            ``[
            dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
            dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
            ]``.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dims: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: int = 4.0,
        qkv_bias: bool = True,
        qk_scale: int = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_cfg: ConfigType = dict(type="LN", eps=1e-6),
        num_frames: int = 16,  # frames per attention
        tubelet_size: int = 2,
        use_mean_pooling: int = True,
        pretrained: Optional[str] = None,
        return_feat_map: bool = False,
        with_cp: bool = False,
        adapter_mlp_ratio: float = 0.25,
        total_frames: int = 768,
        adapter_index: list = [3, 5, 7, 11],
        tubelet_token_redundancy_aux: Optional[Dict] = None,
        tubelet_packed_runtime_route: Optional[Dict] = None,
        init_cfg: Optional[Union[Dict, List[Dict]]] = [
            dict(type="TruncNormal", layer="Linear", std=0.02, bias=0.0),
            dict(type="Constant", layer="LayerNorm", val=1.0, bias=0.0),
        ],
        **kwargs,
    ) -> None:
        if pretrained:
            self.init_cfg = dict(type="Pretrained", checkpoint=pretrained)
        super().__init__(init_cfg=init_cfg)

        self.with_cp = with_cp

        self.embed_dims = embed_dims
        self.patch_size = patch_size
        self.latest_tubelet_token_redundancy_summary = None
        self.latest_tubelet_packed_runtime_summary = None

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type="Conv3d",
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
            padding=(0, 0, 0),
            dilation=(1, 1, 1),
        )

        grid_size = img_size // patch_size
        num_patches = grid_size**2 * (num_frames // tubelet_size)
        self.grid_size = (grid_size, grid_size)

        # sine-cosine positional embeddings
        pos_embed = get_sinusoid_encoding(num_patches, embed_dims)
        self.register_buffer("pos_embed", pos_embed)

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.tubelet_token_redundancy_aux = None
        if tubelet_token_redundancy_aux is not None:
            self.tubelet_token_redundancy_aux = TubeletTokenRedundancyAux(**dict(tubelet_token_redundancy_aux))
        self.tubelet_packed_runtime_route = None
        if tubelet_packed_runtime_route is not None:
            self.tubelet_packed_runtime_route = PackedTubeletRuntimeRoute(**dict(tubelet_packed_runtime_route))

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = ModuleList(
            [
                Block(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[i],
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                    init_cfg=init_cfg,
                    use_adapter=i in adapter_index,
                    adapter_mlp_ratio=adapter_mlp_ratio,
                    temporal_size=total_frames // tubelet_size,
                )
                for i in range(depth)
            ]
        )

        if use_mean_pooling:
            self.norm = nn.Identity()
            self.fc_norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
            self.fc_norm = None

        self.return_feat_map = return_feat_map

        # count the number of parameters in the backbone
        num_vit_param = sum(p.numel() for name, p in self.named_parameters() if "adapter" not in name)
        num_adapter_param = sum(p.numel() for name, p in self.named_parameters() if "adapter" in name)
        ratio = num_adapter_param / num_vit_param * 100
        print("ViT's param: {}, Adapter's params: {}, ratio: {:2.1f}%".format(num_vit_param, num_adapter_param, ratio))

    def forward(self, x: Tensor) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input data.
        Returns:
            Tensor: The feature of the input
                samples extracted by the backbone.
        """
        self._freeze_layers()

        b, _, _, h, w = x.shape
        h //= self.patch_size
        w //= self.patch_size
        x = self.patch_embed(x)[0]
        if self.tubelet_token_redundancy_aux is not None:
            x = self.tubelet_token_redundancy_aux(x, h, w)
            self.latest_tubelet_token_redundancy_summary = self.tubelet_token_redundancy_aux.last_summary

        if (h, w) != self.grid_size:
            pos_embed = self.pos_embed.reshape(-1, *self.grid_size, self.embed_dims)
            pos_embed = pos_embed.permute(0, 3, 1, 2)
            pos_embed = F.interpolate(pos_embed, size=(h, w), mode="bicubic", align_corners=False)
            pos_embed = pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
            pos_embed = pos_embed.reshape(1, -1, self.embed_dims)
        else:
            pos_embed = self.pos_embed

        x = x + pos_embed
        x = self.pos_drop(x)

        if self.tubelet_packed_runtime_route is not None and self.tubelet_packed_runtime_route.enabled:
            x = self.tubelet_packed_runtime_route(x, self.blocks, h, w, training=self.training)
            self.latest_tubelet_packed_runtime_summary = self.tubelet_packed_runtime_route.last_summary
        else:
            self.latest_tubelet_packed_runtime_summary = None
            for blk in self.blocks:
                x = blk(x, h, w)

        x = self.norm(x)

        if self.return_feat_map:
            x = x.reshape(b, -1, h, w, self.embed_dims)
            x = x.permute(0, 4, 1, 2, 3)
            return x

        if self.fc_norm is not None:
            return self.fc_norm(x.mean(1))

        return x[:, 0]

    def _freeze_layers(self):
        """Prevent all the parameters not in the adapters"""

        # freeze patch_embed
        self.patch_embed.eval()
        for m in self.patch_embed.modules():
            for param in m.parameters():
                param.requires_grad = False

        # freeze blocks except the adapter's parameters
        for block in self.blocks:
            for m, n in block.named_children():
                if "adapter" not in m and m != "drop_path":
                    n.eval()
                    for param in n.parameters():
                        param.requires_grad = False
