from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, Mapping


ROUTE_LABEL = "C3_MAINLINE_OPTIMIZATION"
ROUTE_FAMILY = "C3_ORIGINAL_OPTIMIZATION_ROUTE"


MODEL_MATRIX: list[dict[str, Any]] = [
    {
        "id": "timm_mobilenetv3_large_100_tsm_tcn",
        "family": "image_backbone_temporal_head",
        "backend": "timm",
        "constructor": "mobilenetv3_large_100",
        "tier": "first_wave",
        "default_download": True,
        "compute_class": "low",
        "expected_input": "lowres_rgb_frames_64_or_96px",
        "intended_head": "tsm_or_tcn_frame_actionness",
        "why": "Existing MobileNet evidence is the strongest cheap p_action baseline; adding temporal head tests whether frame context fixes weak boundaries.",
    },
    {
        "id": "timm_tf_efficientnetv2_b0_tcn",
        "family": "image_backbone_temporal_head",
        "backend": "timm",
        "constructor": "tf_efficientnetv2_b0",
        "tier": "first_wave",
        "default_download": True,
        "compute_class": "low_mid",
        "expected_input": "lowres_rgb_frames_64_or_96px",
        "intended_head": "tcn_frame_actionness",
        "why": "EfficientNetV2-B0 is a compact spatial recognizer that may improve action/background separation over MobileNet at modest cost.",
    },
    {
        "id": "timm_convnext_tiny_tcn",
        "family": "image_backbone_temporal_head",
        "backend": "timm",
        "constructor": "convnext_tiny",
        "tier": "first_wave",
        "default_download": True,
        "compute_class": "mid",
        "expected_input": "lowres_rgb_frames_96_or_128px",
        "intended_head": "tcn_or_transformer_frame_actionness",
        "why": "ConvNeXt-Tiny is a stronger modern spatial backbone; useful to test whether coarse-classifier failure is mostly visual-recognition capacity.",
    },
    {
        "id": "timm_resnet18_tcn",
        "family": "image_backbone_temporal_head",
        "backend": "timm",
        "constructor": "resnet18",
        "tier": "first_wave",
        "default_download": True,
        "compute_class": "low_mid",
        "expected_input": "lowres_rgb_frames_64_or_96px",
        "intended_head": "tcn_frame_actionness",
        "why": "ResNet18 is a stable classic baseline for visual capacity and debugging timm/torchvision feature extraction.",
    },
    {
        "id": "timm_vit_tiny_patch16_224_temporal",
        "family": "image_backbone_temporal_head",
        "backend": "timm",
        "constructor": "vit_tiny_patch16_224",
        "tier": "first_wave",
        "default_download": True,
        "compute_class": "mid",
        "expected_input": "resized_rgb_frames_128_or_224px",
        "intended_head": "temporal_transformer_frame_actionness",
        "why": "A small image Transformer tests whether global per-frame context helps actionness more than CNN texture bias.",
    },
    {
        "id": "torchvision_r3d_18",
        "family": "native_video_classifier",
        "backend": "torchvision_video",
        "constructor": "r3d_18",
        "weights_enum": "R3D_18_Weights",
        "tier": "first_wave",
        "default_download": True,
        "compute_class": "mid",
        "expected_input": "short_rgb_clip",
        "intended_head": "sliding_clip_to_frame_actionness",
        "why": "Classic 3D CNN baseline for short-term motion and appearance.",
    },
    {
        "id": "torchvision_r2plus1d_18",
        "family": "native_video_classifier",
        "backend": "torchvision_video",
        "constructor": "r2plus1d_18",
        "weights_enum": "R2Plus1D_18_Weights",
        "tier": "first_wave",
        "default_download": True,
        "compute_class": "mid",
        "expected_input": "short_rgb_clip",
        "intended_head": "sliding_clip_to_frame_actionness",
        "why": "Factorized spatial-temporal convolution is often stronger than plain 3D conv at similar scale.",
    },
    {
        "id": "torchvision_mc3_18",
        "family": "native_video_classifier",
        "backend": "torchvision_video",
        "constructor": "mc3_18",
        "weights_enum": "MC3_18_Weights",
        "tier": "first_wave",
        "default_download": True,
        "compute_class": "mid",
        "expected_input": "short_rgb_clip",
        "intended_head": "sliding_clip_to_frame_actionness",
        "why": "Mixed 2D/3D convolution baseline helps separate whether full 3D modeling is necessary.",
    },
    {
        "id": "torchvision_s3d",
        "family": "native_video_classifier",
        "backend": "torchvision_video",
        "constructor": "s3d",
        "weights_enum": "S3D_Weights",
        "tier": "first_wave",
        "default_download": True,
        "compute_class": "mid",
        "expected_input": "short_rgb_clip",
        "intended_head": "sliding_clip_to_frame_actionness",
        "why": "Separable 3D video model gives a stronger efficient motion-aware baseline.",
    },
    {
        "id": "torchvision_mvit_v2_s",
        "family": "native_video_classifier",
        "backend": "torchvision_video",
        "constructor": "mvit_v2_s",
        "weights_enum": "MViT_V2_S_Weights",
        "tier": "first_wave",
        "default_download": True,
        "compute_class": "mid_high",
        "expected_input": "short_rgb_clip",
        "intended_head": "sliding_clip_to_frame_actionness_or_teacher",
        "why": "Multiscale video Transformer tests whether stronger spatiotemporal context can produce better p_action transitions.",
    },
    {
        "id": "torchvision_swin3d_t",
        "family": "native_video_classifier",
        "backend": "torchvision_video",
        "constructor": "swin3d_t",
        "weights_enum": "Swin3D_T_Weights",
        "tier": "first_wave",
        "default_download": True,
        "compute_class": "mid_high",
        "expected_input": "short_rgb_clip",
        "intended_head": "sliding_clip_to_frame_actionness_or_teacher",
        "why": "Video Swin-T is a compact modern video Transformer for stronger short-clip recognition.",
    },
    {
        "id": "pytorchvideo_x3d_xs",
        "family": "native_video_classifier",
        "backend": "pytorchvideo_hub",
        "constructor": "x3d_xs",
        "tier": "first_wave",
        "default_download": True,
        "compute_class": "low_mid",
        "expected_input": "short_rgb_clip",
        "intended_head": "sliding_clip_to_frame_actionness",
        "why": "X3D-XS is a very efficient video model and a strong candidate for deployable low-cost temporal sensing.",
    },
    {
        "id": "pytorchvideo_x3d_s",
        "family": "native_video_classifier",
        "backend": "pytorchvideo_hub",
        "constructor": "x3d_s",
        "tier": "first_wave",
        "default_download": True,
        "compute_class": "mid",
        "expected_input": "short_rgb_clip",
        "intended_head": "sliding_clip_to_frame_actionness",
        "why": "X3D-S tests the next cost point above X3D-XS for better action/background accuracy.",
    },
    {
        "id": "pytorchvideo_c2d_r50",
        "family": "native_video_classifier",
        "backend": "pytorchvideo_hub",
        "constructor": "c2d_r50",
        "tier": "first_wave",
        "default_download": True,
        "compute_class": "mid",
        "expected_input": "short_rgb_clip",
        "intended_head": "sliding_clip_to_frame_actionness",
        "why": "C2D-R50 separates spatial-recognition strength from true temporal modeling.",
    },
    {
        "id": "pytorchvideo_i3d_r50",
        "family": "native_video_classifier",
        "backend": "pytorchvideo_hub",
        "constructor": "i3d_r50",
        "tier": "first_wave",
        "default_download": True,
        "compute_class": "mid_high",
        "expected_input": "short_rgb_clip",
        "intended_head": "sliding_clip_to_frame_actionness",
        "why": "I3D-R50 is a classic spatiotemporal recognition baseline for boundary-sensitive p_action curves.",
    },
    {
        "id": "pytorchvideo_slowfast_r50",
        "family": "native_video_classifier",
        "backend": "pytorchvideo_hub",
        "constructor": "slowfast_r50",
        "tier": "first_wave",
        "default_download": True,
        "compute_class": "high",
        "expected_input": "slowfast_short_clip",
        "intended_head": "teacher_or_expensive_upper_bound",
        "why": "SlowFast explicitly models fast motion; useful as a high-cost diagnostic for whether motion-aware p_action can cover boundaries.",
    },
    {
        "id": "hf_videomae_small_kinetics",
        "family": "video_transformer_teacher",
        "backend": "hf_snapshot",
        "repo_id": "MCG-NJU/videomae-small-finetuned-kinetics",
        "tier": "first_wave",
        "default_download": True,
        "compute_class": "mid_high",
        "expected_input": "short_rgb_clip",
        "intended_head": "teacher_or_transformers_adapter",
        "why": "VideoMAE-small is a manageable pretrained video Transformer teacher candidate; it may reveal whether stronger video representation fixes p_action.",
    },
    {
        "id": "hf_videomae_base_kinetics",
        "family": "video_transformer_teacher",
        "backend": "hf_snapshot",
        "repo_id": "MCG-NJU/videomae-base-finetuned-kinetics",
        "tier": "second_wave",
        "default_download": False,
        "compute_class": "high",
        "expected_input": "short_rgb_clip",
        "intended_head": "teacher_upper_bound",
        "why": "VideoMAE-base is a stronger but heavier teacher; keep out of default download until first-wave results justify it.",
    },
    {
        "id": "torchvision_swin3d_s",
        "family": "native_video_classifier",
        "backend": "torchvision_video",
        "constructor": "swin3d_s",
        "weights_enum": "Swin3D_S_Weights",
        "tier": "second_wave",
        "default_download": False,
        "compute_class": "high",
        "expected_input": "short_rgb_clip",
        "intended_head": "teacher_upper_bound",
        "why": "Heavier Video Swin scale for upper-bound diagnosis after first-wave triage.",
    },
]


def iter_matrix(*, tier: str = "first_wave", include_optional: bool = False, families: set[str] | None = None):
    for entry in MODEL_MATRIX:
        if families and str(entry["family"]) not in families:
            continue
        if tier != "all" and str(entry["tier"]) != tier:
            if not (include_optional and str(entry["tier"]) == "second_wave"):
                continue
        if not include_optional and not bool(entry.get("default_download", False)):
            continue
        yield entry


def _status(entry: Mapping[str, Any], status: str, **extra: Any) -> dict[str, Any]:
    result = {
        "id": entry["id"],
        "family": entry["family"],
        "backend": entry["backend"],
        "tier": entry["tier"],
        "status": status,
    }
    result.update(extra)
    return result


def _download_timm(entry: Mapping[str, Any], *, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return _status(entry, "dry_run")
    import timm

    model = timm.create_model(str(entry["constructor"]), pretrained=True, num_classes=0)
    param_count = sum(p.numel() for p in model.parameters())
    return _status(entry, "downloaded", param_count=param_count)


def _download_torchvision_video(entry: Mapping[str, Any], *, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return _status(entry, "dry_run")
    from torchvision.models import video

    fn = getattr(video, str(entry["constructor"]))
    weights_enum = getattr(video, str(entry["weights_enum"]))
    weights = weights_enum.DEFAULT
    model = fn(weights=weights)
    param_count = sum(p.numel() for p in model.parameters())
    return _status(entry, "downloaded", weights=str(weights), param_count=param_count)


def _download_pytorchvideo_hub(entry: Mapping[str, Any], *, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return _status(entry, "dry_run")
    import pytorchvideo.models.hub as hub

    fn = getattr(hub, str(entry["constructor"]))
    model = fn(pretrained=True)
    param_count = sum(p.numel() for p in model.parameters())
    return _status(entry, "downloaded", param_count=param_count)


def _download_hf_snapshot(entry: Mapping[str, Any], *, dry_run: bool) -> dict[str, Any]:
    repo_id = str(entry["repo_id"])
    if dry_run:
        return _status(entry, "dry_run", repo_id=repo_id)
    from huggingface_hub import snapshot_download

    local_path = snapshot_download(
        repo_id=repo_id,
        allow_patterns=[
            "config.json",
            "preprocessor_config.json",
            "pytorch_model.bin",
            "model.safetensors",
            "*.txt",
            "README.md",
        ],
    )
    return _status(entry, "downloaded", repo_id=repo_id, local_path=local_path)


def download_entry(entry: Mapping[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
    started = time.time()
    try:
        backend = str(entry["backend"])
        if backend == "timm":
            result = _download_timm(entry, dry_run=dry_run)
        elif backend == "torchvision_video":
            result = _download_torchvision_video(entry, dry_run=dry_run)
        elif backend == "pytorchvideo_hub":
            result = _download_pytorchvideo_hub(entry, dry_run=dry_run)
        elif backend == "hf_snapshot":
            result = _download_hf_snapshot(entry, dry_run=dry_run)
        else:
            result = _status(entry, "unsupported_backend", backend=backend)
    except Exception as exc:  # pragma: no cover - exercised by remote environment differences
        result = _status(entry, "failed", error_type=type(exc).__name__, error=str(exc))
    result["elapsed_sec"] = round(time.time() - started, 3)
    return result


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="C3 coarse classifier model matrix and weight downloader.")
    parser.add_argument("--print-matrix", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--tier", choices=["first_wave", "second_wave", "all"], default="first_wave")
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--family", action="append", default=[])
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.cache_root is not None:
        cache_root = args.cache_root.resolve()
        os.environ.setdefault("TORCH_HOME", str(cache_root / "torch"))
        os.environ.setdefault("HF_HOME", str(cache_root / "hf"))
        os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
        cache_root.mkdir(parents=True, exist_ok=True)

    families = set(args.family) if args.family else None
    entries = list(iter_matrix(tier=args.tier, include_optional=args.include_optional, families=families))
    payload: dict[str, Any] = {
        "schema_version": "c3_coarse_classifier_model_matrix_v1",
        "route_label": ROUTE_LABEL,
        "route_family": ROUTE_FAMILY,
        "diagnostic_only": True,
        "no_detector_training": True,
        "no_detector_eval": True,
        "selected_tier": args.tier,
        "include_optional": bool(args.include_optional),
        "entries": entries,
    }

    if args.download:
        results = [download_entry(entry, dry_run=bool(args.dry_run)) for entry in entries]
        payload["download_results"] = results
        payload["failed_count"] = sum(1 for item in results if item.get("status") == "failed")

    if args.print_matrix or not args.output_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
