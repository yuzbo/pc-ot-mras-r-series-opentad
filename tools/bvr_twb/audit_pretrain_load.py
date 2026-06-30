import argparse
import contextlib
import io
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _to_plain(value):
    if isinstance(value, dict):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    return value


def _resolve_config(config_path):
    from mmengine.config import Config

    cfg = Config.fromfile(str(config_path))
    backbone_cfg = cfg.model.backbone
    custom_cfg = backbone_cfg.get("custom", {})
    model_cfg = backbone_cfg.get("backbone", {})
    pretrain = custom_cfg.get("pretrain", None)
    return cfg, _to_plain(custom_cfg), _to_plain(model_cfg), pretrain


def _checkpoint_stats(path):
    import torch

    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        keys = sorted(str(key) for key in checkpoint.keys())
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    else:
        keys = [type(checkpoint).__name__]
        state_dict = checkpoint
    state_keys = sorted(str(key) for key in state_dict.keys()) if hasattr(state_dict, "keys") else []
    return {
        "checkpoint_top_keys": keys[:40],
        "state_dict_num_keys": len(state_keys),
        "state_dict_key_sample": state_keys[:20],
        "has_backbone_like_keys": any(
            ("backbone" in key or "patch_embed" in key or "blocks." in key)
            for key in state_keys[:2000]
        ),
    }


def _runtime_build_backbone(backbone_cfg):
    from opentad.models.backbones.backbone_wrapper import BackboneWrapper

    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        wrapper = BackboneWrapper(backbone_cfg)
    return {
        "runtime_build_attempted": True,
        "runtime_build_class": wrapper.__class__.__name__,
        "runtime_stdout": stdout.getvalue().splitlines(),
        "runtime_stderr": stderr.getvalue().splitlines(),
    }


def run_pretrain_audit(config, out_dir=None, require_runtime=False):
    config_path = Path(config)
    cfg, custom_cfg, model_cfg, pretrain = _resolve_config(config_path)
    expected_name = "vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"
    summary = {
        "audit": "bvr_twb_pretrain_load",
        "config": str(config_path),
        "route_label": cfg.get("route_label", None),
        "model_type": cfg.model.get("type", None),
        "backbone_type": model_cfg.get("type", None),
        "backbone_embed_dims": model_cfg.get("embed_dims", None),
        "custom_cfg_keys": sorted(custom_cfg.keys()),
        "resolved_pretrain": pretrain,
        "expected_videomae_s_pretrain_name": expected_name,
        "pretrain_resolves_videomae_s": isinstance(pretrain, str) and expected_name in pretrain,
        "no_training": True,
        "no_video_decode": True,
        "no_metric_claim": True,
        "no_runtime_or_flops_claim": True,
        "full_training_unlocked": False,
    }

    if pretrain is None:
        summary.update(
            {
                "verdict": "BLOCKER_PRETRAIN_MISSING_IN_RESOLVED_CONFIG",
                "blocked": True,
                "blocked_reason": (
                    "Resolved cfg.model.backbone.custom has no pretrain path. "
                    "BackboneWrapper would take its random-init warning path instead of "
                    "loading the VideoMAE-S checkpoint."
                ),
                "runtime_load_attempted": False,
            }
        )
    else:
        pretrain_path = Path(pretrain)
        if not pretrain_path.is_absolute():
            pretrain_path = (ROOT / pretrain_path).resolve()
        summary["resolved_pretrain_abs"] = str(pretrain_path)
        summary["pretrain_file_exists"] = pretrain_path.exists()
        if not pretrain_path.exists():
            summary.update(
                {
                    "verdict": "BLOCKER_PRETRAIN_FILE_MISSING",
                    "blocked": True,
                    "blocked_reason": f"Resolved pretrain path does not exist: {pretrain_path}",
                    "runtime_load_attempted": False,
                }
            )
        else:
            summary.update(_checkpoint_stats(pretrain_path))
            if require_runtime:
                summary.update(_runtime_build_backbone(cfg.model.backbone))
            summary.update(
                {
                    "verdict": "PASS_PRETRAIN_RESOLVED_AND_READABLE_NO_TRAINING",
                    "blocked": False,
                    "runtime_load_attempted": bool(require_runtime),
                }
            )

    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Audit BVR-TWB resolved pretrain path without training.")
    parser.add_argument(
        "--config",
        default="configs/adatad/thumos/input_bvr_twb_dynamic_adapter_irregular_headv3.py",
    )
    parser.add_argument("--out-dir")
    parser.add_argument("--require-runtime", action="store_true")
    args = parser.parse_args()
    summary = run_pretrain_audit(args.config, out_dir=args.out_dir, require_runtime=args.require_runtime)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if summary.get("blocked"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
