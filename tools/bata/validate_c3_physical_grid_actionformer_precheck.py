from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SCHEMA_VERSION = "c3_physical_grid_actionformer_precheck_v0"
READY = "C3_PHYSICAL_GRID_ACTIONFORMER_PRECHECK_READY"
NO_GO = "C3_PHYSICAL_GRID_ACTIONFORMER_PRECHECK_NO_GO"
FORBIDDEN_CONFIG_TOKENS = (
    "P2",
    "NativeIrregularAreaHeadP2",
    "raw_prediction",
    "raw_prediction_cache",
    "teacher",
    "test_gt",
    "offline_ledger",
    "DIVERGENT_INNOVATION",
    "BH_SDC",
)


def strict_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json_value(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    return str(value)


def _load_config(config_path: Path):
    from mmengine.config import Config

    return Config.fromfile(str(config_path))


def _flag(cfg: Any, key: str) -> Any:
    flags = getattr(cfg, "protocol_flags", {})
    if isinstance(flags, Mapping):
        return flags.get(key)
    return getattr(flags, key, None)


def _iter_config_token_hits(value: Any, token: str, path: tuple[str, ...] = ()) -> list[str]:
    hits: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            child_path = path + (key_text,)
            key_is_disabled_switch = item is False or item is None
            if path[:1] != ("protocol_flags",) and not key_is_disabled_switch and token in key_text:
                hits.append(".".join(child_path))
            hits.extend(_iter_config_token_hits(item, token, child_path))
        return hits
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            hits.extend(_iter_config_token_hits(item, token, path + (str(index),)))
        return hits
    if isinstance(value, str) and token in value:
        hits.append(".".join(path) or "<root>")
    return hits


def validate_precheck_config(config: str | Path) -> dict[str, Any]:
    config_path = Path(config).expanduser()
    cfg = _load_config(config_path)
    blockers: list[str] = []

    if getattr(cfg, "route_label", None) != "C3_ORIGINAL_OPTIMIZATION_ROUTE":
        blockers.append("route_label must be C3_ORIGINAL_OPTIMIZATION_ROUTE")
    if getattr(cfg, "route_family", None) != "C3_MAINLINE_OPTIMIZATION":
        blockers.append("route_family must be C3_MAINLINE_OPTIMIZATION")
    if cfg.model.type != "ActionFormer":
        blockers.append("model.type must be ActionFormer")
    if cfg.model.rpn_head.type != "ActionFormerHead":
        blockers.append("rpn_head.type must be ActionFormerHead")

    physical = cfg.model.rpn_head.get("physical_grid_actionformer", {})
    if not bool(physical.get("enabled", False)):
        blockers.append("physical_grid_actionformer.enabled must be true")
    if not bool(physical.get("required", False)):
        blockers.append("physical_grid_actionformer.required must be true")
    if not bool(physical.get("strict", False)):
        blockers.append("physical_grid_actionformer.strict must be true")

    required_false_flags = (
        "selector_changed",
        "uses_p2_head",
        "uses_raw_prediction_cache",
        "uses_teacher",
        "uses_test_gt",
        "uses_offline_ledger",
        "tools_test_allowed",
        "tools_train_allowed",
        "remote_sync_allowed",
        "slurm_allowed",
        "metric_claim_allowed",
        "paper_claim_allowed",
    )
    for key in required_false_flags:
        if _flag(cfg, key) is not False:
            blockers.append(f"protocol_flags.{key} must be false")
    if _flag(cfg, "precheck_only") is not True:
        blockers.append("protocol_flags.precheck_only must be true")

    for split in ("train", "val", "test"):
        load_steps = [
            step for step in cfg.dataset[split].pipeline
            if isinstance(step, Mapping) and step.get("type") == "LoadFrames"
        ]
        if len(load_steps) != 1:
            blockers.append(f"dataset.{split}.pipeline must contain exactly one LoadFrames step")
            continue
        if load_steps[0].get("remap_gt_to_selected_axis") is not False:
            blockers.append(f"dataset.{split}.LoadFrames.remap_gt_to_selected_axis must be false")

    cfg_dict = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
    for token in FORBIDDEN_CONFIG_TOKENS:
        hits = _iter_config_token_hits(cfg_dict, token)
        if hits:
            blockers.append(f"forbidden config token {token!r} appears at: {', '.join(hits[:8])}")

    return {
        "schema_version": SCHEMA_VERSION,
        "decision": READY if not blockers else NO_GO,
        "config": str(config_path),
        "blocking_findings": blockers,
        "protocol_flags": {
            "precheck_only": True,
            "runs_training": False,
            "runs_tools_test": False,
            "tools_train_allowed": False,
            "tools_test_allowed": False,
            "remote_sync_allowed": False,
            "slurm_gpu_allowed": False,
            "metric_claim_allowed": False,
            "paper_claim_allowed": False,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fail-closed local precheck for the C3 physical-grid ActionFormer candidate.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        payload = validate_precheck_config(args.config)
    except Exception as exc:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "decision": NO_GO,
            "error_type": exc.__class__.__name__,
            "error": str(exc),
        }
    if args.output:
        Path(args.output).write_text(json.dumps(strict_json_value(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(strict_json_value(payload), sort_keys=True))
    return 0 if payload.get("decision") == READY else 1


if __name__ == "__main__":
    raise SystemExit(main())
