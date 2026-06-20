import importlib
import json
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

from tools.bata.audit_pc_ot_mras_tubelet_token_redundancy import (
    READY,
    audit_pc_ot_mras_tubelet_token_redundancy,
    run_json_tubelet_token_redundancy_audit,
)


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
    mmengine_model.BaseModule = nn.Module
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
    "r28_vit_adapter_for_test",
    ROOT / "opentad" / "models" / "backbones" / "vit_adapter.py",
)
vit_adapter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = vit_adapter
spec.loader.exec_module(vit_adapter)
TubeletTokenRedundancyAux = vit_adapter.TubeletTokenRedundancyAux


def _tokens(batch=2, tubelets=4, h=2, w=3, channels=5):
    return torch.arange(batch * tubelets * h * w * channels, dtype=torch.float32).reshape(
        batch,
        tubelets * h * w,
        channels,
    )


def test_tubelet_redundancy_aux_identity_preserves_dense_tokens():
    x = _tokens()
    aux = TubeletTokenRedundancyAux(enabled=True, mode="identity")
    y = aux(x, 2, 3)

    assert torch.equal(y, x)
    summary = aux.last_summary
    assert summary["schema_version"] == "tubelet_token_redundancy_aux_summary_v0"
    assert summary["route_unit"] == "temporal_tubelet_group"
    assert summary["temporal_tubelets"] == 4
    assert summary["spatial_tokens_per_tubelet"] == 6
    assert summary["proposed_keep_count"] == 4
    assert summary["dense_output_preserved"] is True
    assert summary["compute_route_applied"] is False
    assert summary["spatial_patch_crop_allowed"] is False
    assert summary["runtime_flops_claim_allowed"] is False


def test_tubelet_redundancy_aux_deterministic_cap_preserves_dense_tokens():
    x = _tokens(tubelets=8)
    aux = TubeletTokenRedundancyAux(
        enabled=True,
        mode="deterministic_tubelet_cap",
        keep_ratio=0.5,
        route_unit="temporal_tubelet_group",
    )
    y = aux(x, 2, 3)
    summary = aux.last_summary

    assert torch.equal(y, x)
    assert summary["temporal_tubelets"] == 8
    assert summary["spatial_tokens_per_tubelet"] == 6
    assert summary["proposed_keep_count"] == 4
    keep_mask = summary["proposed_tubelet_keep_mask"]
    assert keep_mask.shape == (2, 8)
    assert keep_mask[0].sum().item() == 4
    assert summary["effective_dense_token_count"] == x.shape[1]


def test_tubelet_redundancy_aux_rejects_spatial_crop_route_unit():
    with pytest.raises(ValueError, match="temporal_tubelet_group"):
        TubeletTokenRedundancyAux(
            enabled=True,
            mode="deterministic_tubelet_cap",
            route_unit="spatial_patch",
            keep_ratio=0.5,
        )


def test_tubelet_redundancy_audit_json_roundtrip(tmp_path):
    config_json = tmp_path / "r28_audit_config.json"
    summary_json = tmp_path / "r28_audit_summary.json"
    config_json.write_text(
        json.dumps(
            {
                "mode": "deterministic_tubelet_cap",
                "keep_ratio": 0.5,
                "temporal_tubelets": 8,
                "spatial_h": 2,
                "spatial_w": 3,
                "channels": 4,
            }
        ),
        encoding="utf-8",
    )

    summary = run_json_tubelet_token_redundancy_audit(config_json, summary_json=summary_json)
    loaded = json.loads(summary_json.read_text(encoding="utf-8"))

    assert summary["decision"] == loaded["decision"] == READY
    assert loaded["route_unit"] == "temporal_tubelet_group"
    assert loaded["dense_shape_preserved"] is True
    assert loaded["dense_values_preserved"] is True
    assert loaded["spatial_patch_crop_allowed"] is False
    assert loaded["true_packed_compute_enabled"] is False
    assert loaded["aux_summary"]["proposed_keep_count"] == 4


def test_tubelet_redundancy_audit_reports_no_claim_boundaries():
    summary = audit_pc_ot_mras_tubelet_token_redundancy(mode="deterministic_tubelet_cap", keep_ratio=0.75)

    assert summary["decision"] == READY
    assert summary["uses_gt"] is False
    assert summary["uses_teacher"] is False
    assert summary["uses_raw_prediction"] is False
    assert summary["uses_checkpoint"] is False
    assert summary["runtime_flops_claim_allowed"] is False
    assert summary["metric_claim_allowed"] is False
    assert summary["paper_claim_allowed"] is False
    assert summary["remote_sync_allowed"] is False
    assert summary["slurm_gpu_allowed"] is False
    assert summary["detector_map_allowed"] is False
