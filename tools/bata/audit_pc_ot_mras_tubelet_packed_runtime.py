from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.audit_pc_ot_mras_tubelet_packed_profile import (  # noqa: E402
    READY as R29_READY,
    _expand_tubelet_mask,
    _pack_and_scatter,
    audit_pc_ot_mras_tubelet_packed_profile,
)
from tools.bata.audit_pc_ot_mras_tubelet_token_redundancy import (  # noqa: E402
    _load_vit_adapter_by_path,
    build_synthetic_tubelet_tokens,
)
from tools.bata.export_pc_ot_mras_hard_positions import strict_json_value, write_json  # noqa: E402


SCHEMA_VERSION = "pc_ot_mras_tubelet_packed_runtime_audit_v0"
READY = "PC_OT_MRAS_TUBELET_PACKED_RUNTIME_AUDIT_READY"
NO_GO = "PC_OT_MRAS_TUBELET_PACKED_RUNTIME_AUDIT_NO_GO"


def _torch():
    import importlib

    return importlib.import_module("torch")


def _to_plain(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "tolist"):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    return value


def _finite_float(value: float, *, name: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _cuda_sync_if_needed(tensor) -> None:
    torch = _torch()
    if getattr(tensor, "is_cuda", False) and torch.cuda.is_available():
        torch.cuda.synchronize(tensor.device)


class _ForwardHookCounter:
    def __init__(self, blocks) -> None:
        self.attention_forward_count = 0
        self.mlp_forward_count = 0
        self._handles = []
        for block in blocks:
            self._handles.append(block.attn.register_forward_hook(self._attention_hook))
            self._handles.append(block.mlp.register_forward_hook(self._mlp_hook))

    def _attention_hook(self, _module, _inputs, _output) -> None:
        self.attention_forward_count += 1

    def _mlp_hook(self, _module, _inputs, _output) -> None:
        self.mlp_forward_count += 1

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


def _vit_adapter_module():
    return _load_vit_adapter_by_path()


def _build_blocks(*, channels: int, num_heads: int, mlp_ratio: float, num_blocks: int, seed: int, device):
    torch = _torch()
    if int(num_blocks) <= 0:
        raise ValueError("num_blocks must be positive")
    if int(channels) <= 0:
        raise ValueError("channels must be positive")
    if int(num_heads) <= 0:
        raise ValueError("num_heads must be positive")
    if int(channels) % int(num_heads) != 0:
        raise ValueError("channels must be divisible by num_heads")
    if float(mlp_ratio) <= 0.0:
        raise ValueError("mlp_ratio must be positive")
    vit_adapter = _vit_adapter_module()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        blocks = []
        for _idx in range(int(num_blocks)):
            block = vit_adapter.Block(
                embed_dims=int(channels),
                num_heads=int(num_heads),
                mlp_ratio=float(mlp_ratio),
                qkv_bias=True,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.0,
                with_cp=False,
                use_adapter=False,
                init_cfg=None,
            )
            block.eval()
            block.to(device)
            blocks.append(block)
        return blocks


def _run_blocks(blocks, tokens, *, h: int, w: int):
    out = tokens
    for block in blocks:
        out = block(out, int(h), int(w))
    return out


def _time_blocks(blocks, tokens, *, h: int, w: int, warmup: int, repeat: int, count_hooks: bool = False) -> dict[str, Any]:
    torch = _torch()
    if int(warmup) < 0:
        raise ValueError("warmup must be non-negative")
    if int(repeat) <= 0:
        raise ValueError("repeat must be positive")

    counter = _ForwardHookCounter(blocks) if count_hooks else None
    with torch.no_grad():
        try:
            for _idx in range(int(warmup)):
                _run_blocks(blocks, tokens, h=int(h), w=int(w))
            _cuda_sync_if_needed(tokens)

            timings_ms: list[float] = []
            output = None
            for _idx in range(int(repeat)):
                start = time.perf_counter()
                output = _run_blocks(blocks, tokens, h=int(h), w=int(w))
                _cuda_sync_if_needed(tokens)
                timings_ms.append((time.perf_counter() - start) * 1000.0)
        finally:
            if counter is not None:
                counter.close()

    if output is None:
        raise RuntimeError("timed execution did not produce output")
    median_ms = _finite_float(statistics.median(timings_ms), name="median_runtime_ms")
    min_ms = _finite_float(min(timings_ms), name="min_runtime_ms")
    max_ms = _finite_float(max(timings_ms), name="max_runtime_ms")
    return {
        "output": output,
        "timings_ms": timings_ms,
        "median_ms": median_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "attention_forward_count": 0 if counter is None else int(counter.attention_forward_count),
        "mlp_forward_count": 0 if counter is None else int(counter.mlp_forward_count),
    }


def _run_selected_reference(blocks, packed_tokens, *, h: int, w: int):
    torch = _torch()
    rows = []
    with torch.no_grad():
        for batch_idx in range(int(packed_tokens.shape[0])):
            rows.append(_run_blocks(blocks, packed_tokens[batch_idx : batch_idx + 1], h=int(h), w=int(w)))
    return torch.cat(rows, dim=0)


def audit_pc_ot_mras_tubelet_packed_runtime(
    *,
    mode: str = "deterministic_tubelet_cap",
    keep_ratio: float = 0.5,
    temporal_tubelets: int = 8,
    spatial_h: int = 2,
    spatial_w: int = 3,
    channels: int = 8,
    num_heads: int = 2,
    mlp_ratio: float = 2.0,
    num_blocks: int = 2,
    warmup: int = 1,
    repeat: int = 3,
    seed: int = 20260620,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run a local synthetic packed temporal-tubelet block and scatter back.

    This is the first step after R29's profiler-only proof. It executes a tiny
    transformer-like attention+MLP stack on packed selected temporal-tubelet
    tokens, scatters the packed outputs back to dense token shape, and records
    local synthetic wall-clock timings. It is not a production ViT/Adapter
    forward path and does not authorize runtime/FLOPs, metric, deploy, or paper
    claims.
    """

    torch = _torch()
    requested_device = str(device)
    if requested_device != "cpu" and not torch.cuda.is_available():
        raise ValueError("non-cpu packed runtime audit requires CUDA to be available")
    run_device = torch.device(requested_device)

    r29_summary = audit_pc_ot_mras_tubelet_packed_profile(
        mode=mode,
        keep_ratio=keep_ratio,
        temporal_tubelets=temporal_tubelets,
        spatial_h=spatial_h,
        spatial_w=spatial_w,
        channels=channels,
        num_blocks=max(int(num_blocks), 1),
    )
    if r29_summary["decision"] != R29_READY:
        raise ValueError("R29 packed profile must pass before true packed runtime proof")

    tokens = build_synthetic_tubelet_tokens(
        temporal_tubelets=temporal_tubelets,
        spatial_h=spatial_h,
        spatial_w=spatial_w,
        channels=channels,
    ).to(run_device)
    tubelet_mask = torch.tensor(
        r29_summary["r28_aux_summary"]["proposed_tubelet_keep_mask"],
        dtype=torch.bool,
        device=run_device,
    )
    spatial_tokens = int(r29_summary["spatial_tokens_per_tubelet"])
    dense_mask = _expand_tubelet_mask(tubelet_mask, spatial_tokens=spatial_tokens)
    pack = _pack_and_scatter(tokens, dense_mask)
    packed_tokens = pack["packed"]

    blocks = _build_blocks(
        channels=int(channels),
        num_heads=int(num_heads),
        mlp_ratio=float(mlp_ratio),
        num_blocks=int(num_blocks),
        seed=int(seed),
        device=run_device,
    )

    dense_timed = _time_blocks(blocks, tokens, h=int(spatial_h), w=int(spatial_w), warmup=int(warmup), repeat=int(repeat))
    packed_timed = _time_blocks(
        blocks,
        packed_tokens,
        h=int(spatial_h),
        w=int(spatial_w),
        warmup=int(warmup),
        repeat=int(repeat),
        count_hooks=True,
    )
    dense_output = dense_timed["output"]
    packed_output = packed_timed["output"]
    selected_reference_output = _run_selected_reference(
        blocks,
        packed_tokens,
        h=int(spatial_h),
        w=int(spatial_w),
    )
    reference_delta = (selected_reference_output - packed_output).abs()
    reference_max_abs_error = float(reference_delta.max().item())
    selected_outputs_match_reference = bool(torch.allclose(selected_reference_output, packed_output, atol=1.0e-5, rtol=1.0e-5))

    scattered_output = torch.zeros_like(tokens)
    scattered_output[dense_mask] = packed_output.reshape(-1, int(channels))
    selected_output = scattered_output[dense_mask]
    selected_input = tokens[dense_mask]
    selected_delta = (selected_output - selected_input).abs()
    dense_selected_delta = (dense_output[dense_mask] - selected_output).abs()
    unselected_zero = bool(torch.equal(scattered_output[~dense_mask], torch.zeros_like(scattered_output[~dense_mask])))

    packed_finite = bool(torch.isfinite(packed_output).all().item())
    scattered_finite = bool(torch.isfinite(scattered_output).all().item())
    dense_finite = bool(torch.isfinite(dense_output).all().item())
    selected_changed = bool(selected_delta.max().item() > 0.0)
    packed_token_count = int(packed_tokens.shape[1])
    dense_token_count = int(tokens.shape[1])
    has_strict_saving = packed_token_count < dense_token_count
    expected_hook_count = int(num_blocks) * (int(warmup) + int(repeat))
    runtime_ratio = _finite_float(
        float(packed_timed["median_ms"]) / max(float(dense_timed["median_ms"]), 1.0e-12),
        name="packed_over_dense_runtime_ratio",
    )

    passed = (
        has_strict_saving
        and packed_finite
        and scattered_finite
        and dense_finite
        and selected_changed
        and selected_outputs_match_reference
        and int(packed_timed["attention_forward_count"]) == expected_hook_count
        and int(packed_timed["mlp_forward_count"]) == expected_hook_count
        and unselected_zero
        and r29_summary["spatial_patch_crop_allowed"] is False
        and r29_summary["spatial_filtering_allowed"] is False
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "decision": READY if passed else NO_GO,
        "synthetic_only": True,
        "local_runtime_proof_only": True,
        "production_forward_changed": False,
        "uses_gt": False,
        "uses_teacher": False,
        "uses_raw_prediction": False,
        "uses_checkpoint": False,
        "route_unit": "temporal_tubelet_group",
        "spatial_patch_crop_allowed": False,
        "spatial_filtering_allowed": False,
        "arbitrary_spatial_patch_filtering_allowed": False,
        "true_packed_compute_enabled": True,
        "packed_attention_executed": True,
        "packed_mlp_executed": True,
        "packed_block_impl": "vit_adapter.Block(use_adapter=False)",
        "dense_backbone_output_changed": False,
        "scatter_back_executed": True,
        "measured_runtime": True,
        "runtime_measurement_scope": "local_synthetic_tiny_block_only",
        "measured_flops": False,
        "runtime_flops_claim_allowed": False,
        "metric_claim_allowed": False,
        "paper_claim_allowed": False,
        "deploy_claim_allowed": False,
        "remote_sync_allowed": False,
        "slurm_gpu_allowed": False,
        "detector_map_allowed": False,
        "mode": mode,
        "keep_ratio": float(keep_ratio),
        "temporal_tubelets": int(temporal_tubelets),
        "spatial_tokens_per_tubelet": spatial_tokens,
        "channels": int(channels),
        "num_heads": int(num_heads),
        "mlp_ratio": float(mlp_ratio),
        "num_blocks": int(num_blocks),
        "warmup": int(warmup),
        "repeat": int(repeat),
        "seed": int(seed),
        "device": str(run_device),
        "dense_token_shape": list(tokens.shape),
        "packed_token_shape": list(packed_tokens.shape),
        "dense_output_shape": list(dense_output.shape),
        "packed_output_shape": list(packed_output.shape),
        "scatter_shape": list(scattered_output.shape),
        "packed_token_count": packed_token_count,
        "dense_token_count": dense_token_count,
        "has_strict_token_saving": has_strict_saving,
        "packed_output_finite": packed_finite,
        "dense_output_finite": dense_finite,
        "scattered_output_finite": scattered_finite,
        "packed_attention_forward_count": int(packed_timed["attention_forward_count"]),
        "packed_mlp_forward_count": int(packed_timed["mlp_forward_count"]),
        "packed_attention_forward_count_expected": expected_hook_count,
        "packed_mlp_forward_count_expected": expected_hook_count,
        "selected_outputs_match_reference": selected_outputs_match_reference,
        "selected_reference_max_abs_error": reference_max_abs_error,
        "selected_output_changed_from_input": selected_changed,
        "selected_output_max_abs_delta_from_input": float(selected_delta.max().item()),
        "selected_output_mean_abs_delta_from_dense": float(dense_selected_delta.mean().item()),
        "unselected_positions_zero_after_scatter": unselected_zero,
        "dense_runtime_ms": {
            "median": float(dense_timed["median_ms"]),
            "min": float(dense_timed["min_ms"]),
            "max": float(dense_timed["max_ms"]),
            "samples": [float(item) for item in dense_timed["timings_ms"]],
        },
        "packed_runtime_ms": {
            "median": float(packed_timed["median_ms"]),
            "min": float(packed_timed["min_ms"]),
            "max": float(packed_timed["max_ms"]),
            "samples": [float(item) for item in packed_timed["timings_ms"]],
        },
        "packed_over_dense_runtime_ratio_observed": runtime_ratio,
        "count_profile": _to_plain(r29_summary["count_profile"]),
        "r29_profile_summary": _to_plain(r29_summary),
    }


def run_json_tubelet_packed_runtime_audit(
    config_json: Path | None = None,
    *,
    summary_json: Path | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if config_json is not None:
        kwargs = json.loads(Path(config_json).read_text(encoding="utf-8"))
    summary = audit_pc_ot_mras_tubelet_packed_runtime(**kwargs)
    if summary_json is not None:
        write_json(summary_json, strict_json_value(summary))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-json", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    args = parser.parse_args(argv)
    summary = run_json_tubelet_packed_runtime_audit(args.config_json, summary_json=args.summary_json)
    print(json.dumps(strict_json_value(summary), indent=2, sort_keys=True))
    return 0 if summary["decision"] == READY else 2


if __name__ == "__main__":
    raise SystemExit(main())
