from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.export_pc_ot_mras_hard_positions import strict_json_value, write_json  # noqa: E402


SCHEMA_VERSION = "pc_ot_mras_tubelet_token_redundancy_audit_v0"
READY = "PC_OT_MRAS_TUBELET_TOKEN_REDUNDANCY_AUDIT_READY"
NO_GO = "PC_OT_MRAS_TUBELET_TOKEN_REDUNDANCY_AUDIT_NO_GO"


def _torch():
    return importlib.import_module("torch")


def _install_vit_adapter_import_stubs() -> None:
    torch = _torch()
    nn = torch.nn

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


def _load_vit_adapter_by_path():
    _install_vit_adapter_import_stubs()
    path = ROOT / "opentad" / "models" / "backbones" / "vit_adapter.py"
    spec = importlib.util.spec_from_file_location("r28_vit_adapter_for_audit", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _aux_class():
    try:
        module = importlib.import_module("opentad.models.backbones.vit_adapter")
    except (ImportError, ModuleNotFoundError):
        module = _load_vit_adapter_by_path()
    return module.TubeletTokenRedundancyAux


def _to_plain(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "tolist"):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    return value


def build_synthetic_tubelet_tokens(
    *,
    batch_size: int = 2,
    temporal_tubelets: int = 8,
    spatial_h: int = 2,
    spatial_w: int = 3,
    channels: int = 4,
):
    torch = _torch()
    if min(batch_size, temporal_tubelets, spatial_h, spatial_w, channels) <= 0:
        raise ValueError("synthetic token dimensions must be positive")
    spatial_tokens = int(spatial_h) * int(spatial_w)
    base = torch.arange(
        int(batch_size) * int(temporal_tubelets) * spatial_tokens * int(channels),
        dtype=torch.float32,
    )
    tokens = base.reshape(int(batch_size), int(temporal_tubelets), spatial_tokens, int(channels))
    tubelet_scale = torch.linspace(1.0, 2.0, int(temporal_tubelets), dtype=torch.float32)
    tokens = tokens / 100.0 + tubelet_scale.reshape(1, int(temporal_tubelets), 1, 1)
    return tokens.reshape(int(batch_size), int(temporal_tubelets) * spatial_tokens, int(channels))


def audit_pc_ot_mras_tubelet_token_redundancy(
    *,
    mode: str = "deterministic_tubelet_cap",
    keep_ratio: float = 0.75,
    temporal_tubelets: int = 8,
    spatial_h: int = 2,
    spatial_w: int = 3,
    channels: int = 4,
) -> dict[str, Any]:
    torch = _torch()
    aux_cls = _aux_class()
    tokens = build_synthetic_tubelet_tokens(
        temporal_tubelets=temporal_tubelets,
        spatial_h=spatial_h,
        spatial_w=spatial_w,
        channels=channels,
    )
    aux = aux_cls(
        enabled=True,
        mode=mode,
        route_unit="temporal_tubelet_group",
        keep_ratio=keep_ratio,
        min_keep_tubelets=1,
        route_pattern="round_linspace",
        forbid_spatial_crop=True,
        local_audit_only=True,
    )
    outputs = aux(tokens, int(spatial_h), int(spatial_w))
    summary = aux.last_summary
    if summary is None:
        raise RuntimeError("tubelet redundancy aux did not produce a summary")

    dense_shape_preserved = tuple(outputs.shape) == tuple(tokens.shape)
    dense_values_preserved = bool(torch.equal(outputs, tokens))
    route_unit_ok = summary.get("route_unit") == "temporal_tubelet_group"
    no_spatial_crop = summary.get("spatial_patch_crop_allowed") is False
    no_compute_claim = summary.get("compute_route_applied") is False and summary.get("runtime_flops_claim_allowed") is False
    proposed_keep_count = int(summary["proposed_keep_count"])
    temporal_count = int(summary["temporal_tubelets"])

    passed = (
        dense_shape_preserved
        and dense_values_preserved
        and route_unit_ok
        and no_spatial_crop
        and no_compute_claim
        and 0 < proposed_keep_count <= temporal_count
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": READY if passed else NO_GO,
        "synthetic_only": True,
        "uses_gt": False,
        "uses_teacher": False,
        "uses_raw_prediction": False,
        "uses_checkpoint": False,
        "route_unit": "temporal_tubelet_group",
        "mode": mode,
        "keep_ratio": float(keep_ratio),
        "dense_shape_preserved": dense_shape_preserved,
        "dense_values_preserved": dense_values_preserved,
        "spatial_patch_crop_allowed": False,
        "true_packed_compute_enabled": False,
        "runtime_flops_claim_allowed": False,
        "metric_claim_allowed": False,
        "paper_claim_allowed": False,
        "remote_sync_allowed": False,
        "slurm_gpu_allowed": False,
        "detector_map_allowed": False,
        "aux_summary": _to_plain(summary),
    }


def run_json_tubelet_token_redundancy_audit(
    config_json: Path | None = None,
    *,
    summary_json: Path | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if config_json is not None:
        kwargs = json.loads(Path(config_json).read_text(encoding="utf-8"))
    summary = audit_pc_ot_mras_tubelet_token_redundancy(**kwargs)
    if summary_json is not None:
        write_json(summary_json, strict_json_value(summary))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-json", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    args = parser.parse_args(argv)
    summary = run_json_tubelet_token_redundancy_audit(args.config_json, summary_json=args.summary_json)
    print(json.dumps(strict_json_value(summary), indent=2, sort_keys=True))
    return 0 if summary["decision"] == READY else 2


if __name__ == "__main__":
    raise SystemExit(main())
