import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest


torch_probe = subprocess.run(
    [sys.executable, "-c", "import torch"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    timeout=30,
    check=False,
)
if torch_probe.returncode != 0:
    pytest.skip("torch unavailable", allow_module_level=True)

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]


def _install_vit_adapter_import_stubs():
    mmcv = types.ModuleType("mmcv")
    mmcv_cnn = types.ModuleType("mmcv.cnn")

    def build_norm_layer(_cfg, embed_dims):
        return "ln", nn.LayerNorm(embed_dims)

    mmcv_cnn.build_norm_layer = build_norm_layer
    bricks = types.ModuleType("mmcv.cnn.bricks")

    class DropPath(nn.Identity):
        pass

    bricks.DropPath = DropPath
    transformer = types.ModuleType("mmcv.cnn.bricks.transformer")

    class FFN(nn.Module):
        def __init__(self, embed_dims, feedforward_channels, act_cfg=None, ffn_drop=0.0, add_identity=False):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(embed_dims, feedforward_channels),
                nn.GELU(),
                nn.Dropout(ffn_drop),
                nn.Linear(feedforward_channels, embed_dims),
            )

        def forward(self, x):
            return self.layers(x)

    class PatchEmbed(nn.Module):
        def __init__(self, in_channels, embed_dims, conv_type, kernel_size, stride, padding, dilation):
            super().__init__()
            self.proj = nn.Conv3d(
                in_channels,
                embed_dims,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )

        def forward(self, x):
            x = self.proj(x)
            b, c, t, h, w = x.shape
            return x.permute(0, 2, 3, 4, 1).reshape(b, t * h * w, c), (t, h, w)

    transformer.FFN = FFN
    transformer.PatchEmbed = PatchEmbed
    mmengine_registry = types.ModuleType("mmengine.registry")

    class _Models:
        def register_module(self):
            def decorator(cls):
                return cls

            return decorator

    mmengine_registry.MODELS = _Models()
    mmengine_model = types.ModuleType("mmengine.model")

    class BaseModule(nn.Module):
        def __init__(self, *args, init_cfg=None, **kwargs):
            super().__init__()
            self.init_cfg = init_cfg

    mmengine_model.BaseModule = BaseModule
    mmengine_model.ModuleList = nn.ModuleList

    def _noop_init(*_args, **_kwargs):
        return None

    mmengine_model.constant_init = _noop_init
    mmengine_model.trunc_normal_init = _noop_init
    mmengine_weight_init = types.ModuleType("mmengine.model.weight_init")
    mmengine_weight_init.constant_init = _noop_init
    mmengine_weight_init.trunc_normal_init = _noop_init
    mmaction_utils = types.ModuleType("mmaction.utils")
    mmaction_utils.ConfigType = dict
    mmaction_utils.OptConfigType = dict
    mmaction_vit = types.ModuleType("mmaction.models.backbones.vit_mae")

    def get_sinusoid_encoding(num_patches, embed_dims):
        return torch.zeros(1, int(num_patches), int(embed_dims))

    mmaction_vit.get_sinusoid_encoding = get_sinusoid_encoding

    sys.modules.setdefault("mmcv", mmcv)
    sys.modules["mmcv.cnn"] = mmcv_cnn
    sys.modules["mmcv.cnn.bricks"] = bricks
    sys.modules["mmcv.cnn.bricks.transformer"] = transformer
    sys.modules["mmengine.registry"] = mmengine_registry
    sys.modules["mmengine.model"] = mmengine_model
    sys.modules["mmengine.model.weight_init"] = mmengine_weight_init
    sys.modules["mmaction.utils"] = mmaction_utils
    sys.modules["mmaction.models.backbones.vit_mae"] = mmaction_vit


_install_vit_adapter_import_stubs()
spec = importlib.util.spec_from_file_location(
    "r31_vit_adapter_for_test",
    ROOT / "opentad" / "models" / "backbones" / "vit_adapter.py",
)
vit_adapter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = vit_adapter
spec.loader.exec_module(vit_adapter)

PackedTubeletRuntimeRoute = vit_adapter.PackedTubeletRuntimeRoute
Block = vit_adapter.Block
VisionTransformerAdapter = vit_adapter.VisionTransformerAdapter


def _blocks(*, use_adapter=False):
    return nn.ModuleList(
        [
            Block(
                embed_dims=8,
                num_heads=2,
                mlp_ratio=2.0,
                qkv_bias=True,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.0,
                with_cp=False,
                use_adapter=use_adapter,
                temporal_size=8,
                init_cfg=None,
            )
        ]
    )


def test_packed_tubelet_route_executes_blocks_and_scatters_to_dense_shape():
    torch.manual_seed(20260620)
    x = torch.randn(2, 48, 8)
    route = PackedTubeletRuntimeRoute(
        enabled=True,
        mode="deterministic_tubelet_cap",
        keep_ratio=0.5,
        local_forward_only=True,
        require_no_adapter_blocks=False,
    )

    y = route(x, _blocks(use_adapter=False), 2, 3, training=False)
    summary = route.last_summary

    assert y.shape == x.shape
    assert summary["schema_version"] == "packed_tubelet_runtime_route_summary_v0"
    assert summary["production_forward_changed"] is True
    assert summary["route_unit"] == "temporal_tubelet_group"
    assert summary["dense_token_shape"] == [2, 48, 8]
    assert summary["packed_token_shape"] == [2, 24, 8]
    assert summary["dense_output_shape"] == [2, 48, 8]
    assert summary["selected_output_shape"] == [2, 24, 8]
    assert summary["has_strict_token_saving"] is True
    assert summary["true_packed_compute_enabled"] is True
    assert summary["packed_attention_executed_in_forward"] is True
    assert summary["packed_mlp_executed_in_forward"] is True
    assert summary["scatter_back_executed"] is True
    assert summary["packed_attention_forward_count"] == 1
    assert summary["packed_mlp_forward_count"] == 1
    assert summary["adapter_forward_count"] == 0
    assert summary["unselected_positions_identity_bypass_without_adapter"] is True
    assert summary["selected_output_finite"] is True
    assert summary["scattered_output_finite"] is True
    assert summary["adapter_blocks_supported"] is True
    assert summary["adapter_dense_contract_preserved"] is True
    assert summary["measured_runtime"] is False
    assert summary["runtime_flops_claim_allowed"] is False
    assert summary["metric_claim_allowed"] is False
    assert summary["paper_claim_allowed"] is False


def test_packed_tubelet_route_supports_adapter_dense_contract_and_keeps_fail_closed_modes():
    x = torch.randn(2, 48, 8)
    route = PackedTubeletRuntimeRoute(enabled=True, keep_ratio=0.5, require_no_adapter_blocks=False)
    y = route(x, _blocks(use_adapter=True), 2, 3, training=False)
    summary = route.last_summary

    assert y.shape == x.shape
    assert summary["adapter_block_count"] == 1
    assert summary["adapter_forward_count"] == 1
    assert summary["dense_scatter_before_adapter"] is True
    assert summary["adapter_dense_contract_preserved"] is True
    assert summary["packed_attention_forward_count"] == 1
    assert summary["packed_mlp_forward_count"] == 1

    strict_route = PackedTubeletRuntimeRoute(enabled=True, keep_ratio=0.5, require_no_adapter_blocks=True)
    with pytest.raises(ValueError, match="adapter-free blocks"):
        strict_route(x, _blocks(use_adapter=True), 2, 3, training=False)

    with pytest.raises(ValueError, match="forbids training mode"):
        route(x, _blocks(use_adapter=False), 2, 3, training=True)


def test_vit_adapter_forward_optin_returns_dense_feature_map_and_summary():
    torch.manual_seed(20260620)
    model = VisionTransformerAdapter(
        img_size=8,
        patch_size=4,
        in_channels=3,
        embed_dims=8,
        depth=1,
        num_heads=2,
        mlp_ratio=2.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        num_frames=4,
        tubelet_size=2,
        use_mean_pooling=False,
        return_feat_map=True,
        with_cp=False,
        adapter_index=[0],
        total_frames=4,
        tubelet_packed_runtime_route=dict(
            enabled=True,
            mode="deterministic_tubelet_cap",
            keep_ratio=0.5,
            local_forward_only=True,
            require_no_adapter_blocks=False,
            allow_training_mode=False,
            scatter_unselected="identity",
        ),
        init_cfg=None,
    )
    model.eval()
    x = torch.randn(2, 3, 4, 8, 12)

    with torch.no_grad():
        y = model(x)

    summary = model.latest_tubelet_packed_runtime_summary
    assert y.shape == (2, 8, 2, 2, 3)
    assert summary["schema_version"] == "packed_tubelet_runtime_route_summary_v0"
    assert summary["dense_token_shape"] == [2, 12, 8]
    assert summary["packed_token_shape"] == [2, 6, 8]
    assert summary["dense_output_shape"] == [2, 12, 8]
    assert summary["adapter_block_count"] == 1
    assert summary["adapter_forward_count"] == 1
    assert summary["dense_scatter_before_adapter"] is True
    assert summary["adapter_dense_contract_preserved"] is True
    assert summary["scatter_back_executed"] is True
    assert summary["spatial_patch_crop_allowed"] is False
    assert summary["runtime_flops_claim_allowed"] is False


def test_vit_adapter_forward_optin_depth_gt1_does_not_run_dense_block_loop():
    torch.manual_seed(20260621)
    model = VisionTransformerAdapter(
        img_size=8,
        patch_size=4,
        in_channels=3,
        embed_dims=8,
        depth=3,
        num_heads=2,
        mlp_ratio=2.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        num_frames=4,
        tubelet_size=2,
        use_mean_pooling=False,
        return_feat_map=True,
        with_cp=False,
        adapter_index=[1],
        total_frames=4,
        tubelet_packed_runtime_route=dict(
            enabled=True,
            mode="deterministic_tubelet_cap",
            keep_ratio=0.5,
            local_forward_only=True,
            require_no_adapter_blocks=False,
            allow_training_mode=False,
            scatter_unselected="identity",
        ),
        init_cfg=None,
    )
    model.eval()
    counts = {"packed": 0, "dense": 0}
    for block in model.blocks:
        original_forward = block.forward

        def wrapped_forward(x, h, w, *args, _original_forward=original_forward, **kwargs):
            if kwargs.get("packed_dense_mask", None) is None:
                counts["dense"] += 1
            else:
                counts["packed"] += 1
            return _original_forward(x, h, w, *args, **kwargs)

        block.forward = wrapped_forward

    x = torch.randn(2, 3, 4, 8, 12)
    with torch.no_grad():
        y = model(x)

    summary = model.latest_tubelet_packed_runtime_summary
    assert y.shape == (2, 8, 2, 2, 3)
    assert counts["packed"] == 3
    assert counts["dense"] == 0
    assert summary["packed_attention_forward_count"] == 3
    assert summary["packed_mlp_forward_count"] == 3
    assert summary["adapter_block_count"] == 1
    assert summary["adapter_forward_count"] == 1
    assert summary["dense_output_shape"] == [2, 12, 8]
    assert summary["selected_dense_tokens"] == 6


def test_dynamic_selected_positions_map_to_full_tubelet_groups_without_spatial_crop():
    selected_positions = torch.tensor(
        [
            [0, 1, 4, 5],
            [2, 3, 6, 7],
        ],
        dtype=torch.long,
    )
    dense_valid_len = 8
    temporal_tubelets = 4
    spatial_tokens = 3
    tubelet_ids = torch.div(
        selected_positions * temporal_tubelets,
        dense_valid_len,
        rounding_mode="floor",
    ).clamp_(0, temporal_tubelets - 1)
    tubelet_mask = torch.zeros((2, temporal_tubelets), dtype=torch.bool)
    tubelet_mask.scatter_(1, tubelet_ids, True)
    dense_mask = PackedTubeletRuntimeRoute._expand_tubelet_mask(tubelet_mask, spatial_tokens)

    assert tubelet_ids.min().item() >= 0
    assert tubelet_ids.max().item() < temporal_tubelets
    assert torch.equal(tubelet_mask[0], torch.tensor([True, False, True, False]))
    assert torch.equal(tubelet_mask[1], torch.tensor([False, True, False, True]))
    assert dense_mask.shape == (2, temporal_tubelets * spatial_tokens)
    for batch_idx in range(dense_mask.shape[0]):
        for tubelet_idx in range(temporal_tubelets):
            group = dense_mask[batch_idx, tubelet_idx * spatial_tokens : (tubelet_idx + 1) * spatial_tokens]
            assert bool(group.all().item()) == bool(tubelet_mask[batch_idx, tubelet_idx].item())

    values = torch.arange(2 * temporal_tubelets * spatial_tokens, dtype=torch.float32).reshape(
        2, temporal_tubelets * spatial_tokens, 1
    )
    packed = PackedTubeletRuntimeRoute._pack_tokens(values, dense_mask)
    assert torch.equal(packed[0, :, 0], values[0, dense_mask[0], 0])
    assert torch.equal(packed[1, :, 0], values[1, dense_mask[1], 0])

    route = PackedTubeletRuntimeRoute(enabled=True, scatter_unselected="zero")
    restored = route._scatter_tokens(values, packed, dense_mask)
    assert torch.equal(restored[dense_mask], values[dense_mask])
    assert torch.equal(restored[~dense_mask], torch.zeros_like(restored[~dense_mask]))


def test_dynamic_tubelet_pack_rejects_ragged_batch_without_padding():
    values = torch.randn(2, 12, 2)
    ragged_mask = torch.tensor(
        [
            [True, True, True, False, False, False, True, True, True, False, False, False],
            [True, True, True, False, False, False, False, False, False, False, False, False],
        ],
        dtype=torch.bool,
    )
    with pytest.raises(ValueError, match="equal selected-token count"):
        PackedTubeletRuntimeRoute._pack_tokens(values, ragged_mask)


def test_vit_adapter_forward_default_path_has_no_packed_summary():
    model = VisionTransformerAdapter(
        img_size=8,
        patch_size=4,
        in_channels=3,
        embed_dims=8,
        depth=1,
        num_heads=2,
        mlp_ratio=2.0,
        num_frames=4,
        tubelet_size=2,
        use_mean_pooling=False,
        return_feat_map=True,
        adapter_index=[],
        init_cfg=None,
    )
    model.eval()
    x = torch.randn(2, 3, 4, 8, 8)

    with torch.no_grad():
        y = model(x)

    assert y.shape == (2, 8, 2, 2, 2)
    assert model.latest_tubelet_packed_runtime_summary is None
