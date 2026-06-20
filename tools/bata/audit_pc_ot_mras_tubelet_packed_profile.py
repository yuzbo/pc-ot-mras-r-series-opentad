from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.audit_pc_ot_mras_tubelet_token_redundancy import (  # noqa: E402
    READY as R28_READY,
    audit_pc_ot_mras_tubelet_token_redundancy,
    build_synthetic_tubelet_tokens,
)
from tools.bata.export_pc_ot_mras_hard_positions import strict_json_value, write_json  # noqa: E402


SCHEMA_VERSION = "pc_ot_mras_tubelet_packed_profile_audit_v0"
READY = "PC_OT_MRAS_TUBELET_PACKED_PROFILE_AUDIT_READY"
NO_GO = "PC_OT_MRAS_TUBELET_PACKED_PROFILE_AUDIT_NO_GO"


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


def _finite_ratio(value: float, *, name: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _expand_tubelet_mask(tubelet_mask, *, spatial_tokens: int):
    torch = _torch()
    if tubelet_mask.ndim != 2:
        raise ValueError("tubelet_mask must have shape [B, T]")
    if int(spatial_tokens) <= 0:
        raise ValueError("spatial_tokens must be positive")
    return tubelet_mask.unsqueeze(-1).expand(-1, -1, int(spatial_tokens)).reshape(
        int(tubelet_mask.shape[0]),
        int(tubelet_mask.shape[1]) * int(spatial_tokens),
    ).to(dtype=torch.bool)


def _pack_and_scatter(tokens, dense_mask):
    torch = _torch()
    if tokens.ndim != 3:
        raise ValueError("tokens must have shape [B, N, C]")
    if dense_mask.shape != tokens.shape[:2]:
        raise ValueError("dense_mask must match [B, N]")
    per_batch_counts = dense_mask.sum(dim=1)
    if int(per_batch_counts.min().item()) <= 0:
        raise ValueError("packed profile requires at least one selected token per sample")
    if not bool(torch.equal(per_batch_counts, per_batch_counts[:1].expand_as(per_batch_counts))):
        raise ValueError("packed profile requires equal selected-token count across batch for rectangular pack")

    packed = tokens[dense_mask].reshape(int(tokens.shape[0]), int(per_batch_counts[0].item()), int(tokens.shape[2]))
    scattered = torch.zeros_like(tokens)
    scattered[dense_mask] = packed.reshape(-1, int(tokens.shape[2]))
    selected_values_preserved = bool(torch.equal(scattered[dense_mask], tokens[dense_mask]))
    unselected_zero = bool(torch.equal(scattered[~dense_mask], torch.zeros_like(scattered[~dense_mask])))
    return {
        "packed": packed,
        "scattered": scattered,
        "selected_values_preserved": selected_values_preserved,
        "unselected_positions_zero": unselected_zero,
    }


def _profile_counts(
    *,
    dense_token_count: int,
    packed_token_count: int,
    num_blocks: int,
) -> dict[str, Any]:
    dense = int(dense_token_count)
    packed = int(packed_token_count)
    blocks = int(num_blocks)
    if min(dense, packed, blocks) <= 0:
        raise ValueError("dense_token_count, packed_token_count, and num_blocks must be positive")
    if packed > dense:
        raise ValueError("packed_token_count must not exceed dense_token_count")

    dense_attention_pairs = dense * dense * blocks
    packed_attention_pairs = packed * packed * blocks
    dense_linear_tokens = dense * blocks
    packed_linear_tokens = packed * blocks
    attention_pair_ratio = _finite_ratio(packed_attention_pairs / dense_attention_pairs, name="attention_pair_ratio")
    linear_token_ratio = _finite_ratio(packed_linear_tokens / dense_linear_tokens, name="linear_token_ratio")
    return {
        "num_blocks": blocks,
        "dense_token_count": dense,
        "packed_token_count": packed,
        "dense_attention_token_pairs_per_sample": int(dense_attention_pairs),
        "packed_attention_token_pairs_per_sample": int(packed_attention_pairs),
        "attention_pair_ratio": attention_pair_ratio,
        "attention_pair_savings": float(1.0 - attention_pair_ratio),
        "dense_linear_tokens_per_sample": int(dense_linear_tokens),
        "packed_linear_tokens_per_sample": int(packed_linear_tokens),
        "linear_token_ratio": linear_token_ratio,
        "linear_token_savings": float(1.0 - linear_token_ratio),
    }


def audit_pc_ot_mras_tubelet_packed_profile(
    *,
    mode: str = "deterministic_tubelet_cap",
    keep_ratio: float = 0.5,
    temporal_tubelets: int = 8,
    spatial_h: int = 2,
    spatial_w: int = 3,
    channels: int = 4,
    num_blocks: int = 12,
) -> dict[str, Any]:
    """Profile a hypothetical packed temporal-tubelet route without running it.

    This audit deliberately stops before real backbone attention/MLP execution.
    It proves that R28's temporal tubelet keep mask can be expanded to a dense
    token mask, packed into a rectangular token tensor, and scattered back to
    dense shape while preserving selected token values. The token-pair counts
    are accounting estimates only and must not be reported as measured runtime
    or FLOPs.
    """

    if int(num_blocks) <= 0:
        raise ValueError("num_blocks must be positive")
    r28_summary = audit_pc_ot_mras_tubelet_token_redundancy(
        mode=mode,
        keep_ratio=keep_ratio,
        temporal_tubelets=temporal_tubelets,
        spatial_h=spatial_h,
        spatial_w=spatial_w,
        channels=channels,
    )
    aux_summary = r28_summary["aux_summary"]
    if r28_summary["decision"] != R28_READY:
        raise ValueError("R28 tubelet redundancy audit must pass before packed profiling")
    if aux_summary["route_unit"] != "temporal_tubelet_group":
        raise ValueError("packed profile only supports temporal_tubelet_group")
    if aux_summary["spatial_patch_crop_allowed"] is not False:
        raise ValueError("packed profile forbids spatial patch crop")

    torch = _torch()
    tokens = build_synthetic_tubelet_tokens(
        temporal_tubelets=temporal_tubelets,
        spatial_h=spatial_h,
        spatial_w=spatial_w,
        channels=channels,
    )
    tubelet_mask = torch.tensor(aux_summary["proposed_tubelet_keep_mask"], dtype=torch.bool)
    spatial_tokens = int(aux_summary["spatial_tokens_per_tubelet"])
    dense_mask = _expand_tubelet_mask(tubelet_mask, spatial_tokens=spatial_tokens)
    pack = _pack_and_scatter(tokens, dense_mask)

    dense_token_count = int(tokens.shape[1])
    packed_token_count = int(pack["packed"].shape[1])
    count_profile = _profile_counts(
        dense_token_count=dense_token_count,
        packed_token_count=packed_token_count,
        num_blocks=num_blocks,
    )
    has_strict_saving = packed_token_count < dense_token_count
    passed = (
        has_strict_saving
        and pack["selected_values_preserved"]
        and pack["unselected_positions_zero"]
        and count_profile["attention_pair_ratio"] < 1.0
        and count_profile["linear_token_ratio"] < 1.0
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "decision": READY if passed else NO_GO,
        "synthetic_only": True,
        "profiler_only": True,
        "uses_gt": False,
        "uses_teacher": False,
        "uses_raw_prediction": False,
        "uses_checkpoint": False,
        "route_unit": "temporal_tubelet_group",
        "spatial_patch_crop_allowed": False,
        "spatial_filtering_allowed": False,
        "arbitrary_spatial_patch_filtering_allowed": False,
        "true_packed_compute_enabled": False,
        "packed_attention_executed": False,
        "packed_mlp_executed": False,
        "dense_backbone_output_changed": False,
        "scatter_back_required_for_next_stage": True,
        "measured_runtime": False,
        "measured_flops": False,
        "runtime_flops_claim_allowed": False,
        "metric_claim_allowed": False,
        "paper_claim_allowed": False,
        "remote_sync_allowed": False,
        "slurm_gpu_allowed": False,
        "detector_map_allowed": False,
        "mode": mode,
        "keep_ratio": float(keep_ratio),
        "temporal_tubelets": int(temporal_tubelets),
        "spatial_tokens_per_tubelet": spatial_tokens,
        "channels": int(channels),
        "selected_tubelets": int(aux_summary["proposed_keep_count"]),
        "dense_mask_shape": list(dense_mask.shape),
        "packed_token_shape": list(pack["packed"].shape),
        "scatter_shape": list(pack["scattered"].shape),
        "selected_values_preserved": pack["selected_values_preserved"],
        "unselected_positions_zero": pack["unselected_positions_zero"],
        "count_profile": count_profile,
        "r28_aux_summary": _to_plain(aux_summary),
    }


def run_json_tubelet_packed_profile_audit(
    config_json: Path | None = None,
    *,
    summary_json: Path | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if config_json is not None:
        kwargs = json.loads(Path(config_json).read_text(encoding="utf-8"))
    summary = audit_pc_ot_mras_tubelet_packed_profile(**kwargs)
    if summary_json is not None:
        write_json(summary_json, strict_json_value(summary))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-json", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    args = parser.parse_args(argv)
    summary = run_json_tubelet_packed_profile_audit(args.config_json, summary_json=args.summary_json)
    print(json.dumps(strict_json_value(summary), indent=2, sort_keys=True))
    return 0 if summary["decision"] == READY else 2


if __name__ == "__main__":
    raise SystemExit(main())
