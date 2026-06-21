from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from tools.bata.export_pc_ot_mras_hard_positions import strict_json_value, write_json  # noqa: E402


SCHEMA_VERSION = "pc_ot_mras_reader_snapshot_dump_v0"
SUMMARY_SCHEMA_VERSION = "pc_ot_mras_reader_snapshot_dump_summary_v0"
READY = "PC_OT_MRAS_READER_SNAPSHOT_DUMP_READY"
NO_GO = "PC_OT_MRAS_READER_SNAPSHOT_DUMP_NO_GO"
READER_OUTPUT_KEYS = (
    "acquisition_matrix",
    "allocation",
    "transport_prob",
    "valid_mask",
    "valid_lengths",
    "time_coords",
    "selected_times",
    "selected_mask",
    "centers",
    "widths",
    "gates",
    "role_ids",
    "round_ids",
    "start_logits",
    "end_logits",
    "boundary_logits",
    "body_logits",
    "uncertainty_logits",
    "redundancy_logits",
    "value_logits",
    "risk_logits",
)
FORBIDDEN_META_KEY_TOKENS = (
    "gt",
    "groundtruth",
    "teacher",
    "oracle",
    "label",
    "labels",
    "segment",
    "segments",
    "annotation",
    "annotations",
    "cache",
    "prediction",
    "rawprediction",
    "detectionresult",
    "resultdetection",
)


def _to_jsonable(value: Any, *, float_digits: int) -> Any:
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if torch.is_floating_point(value):
            value = value.to(dtype=torch.float32)
        return _to_jsonable(value.tolist(), float_digits=float_digits)
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item, float_digits=float_digits) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item, float_digits=float_digits) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("reader snapshot contains non-finite float")
        return round(float(value), int(float_digits))
    return str(value)


def _validate_reader_tensor(value: Any, *, key: str) -> None:
    if not torch.is_tensor(value):
        raise ValueError(f"reader output '{key}' must be a tensor")
    if torch.is_complex(value):
        raise ValueError(f"reader output '{key}' must be real-valued")
    if torch.is_floating_point(value) and not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"reader output '{key}' must be finite")


def serialize_reader_outputs(reader_outputs: Mapping[str, Any], *, float_digits: int = 6) -> dict[str, Any]:
    if not isinstance(reader_outputs, Mapping):
        raise ValueError("reader_outputs must be a mapping")
    out: dict[str, Any] = {}
    for key in READER_OUTPUT_KEYS:
        if key not in reader_outputs:
            continue
        value = reader_outputs[key]
        _validate_reader_tensor(value, key=key)
        out[key] = _to_jsonable(value, float_digits=int(float_digits))
    required = {"valid_mask"}
    if not required.issubset(out):
        raise ValueError("reader snapshot missing required valid_mask")
    if not any(key in out for key in ("acquisition_matrix", "allocation", "transport_prob")):
        raise ValueError("reader snapshot needs acquisition_matrix, allocation, or transport_prob")
    return out


def _safe_sample_id(meta: Mapping[str, Any], fallback: str) -> str:
    for key in ("video_name", "video_id", "sample_id", "filename", "name"):
        value = meta.get(key)
        if isinstance(value, str) and value:
            return value
    return fallback


def _validate_meta_for_identifier_only(meta: Any, *, index: int) -> Mapping[str, Any]:
    if not isinstance(meta, Mapping):
        raise ValueError(f"metas[{index}] must be a mapping")
    for key in meta.keys():
        normalized = "".join(ch for ch in str(key).lower() if ch.isalnum())
        if any(token in normalized for token in FORBIDDEN_META_KEY_TOKENS):
            continue
    return meta


def sample_ids_from_metas(metas: Sequence[Mapping[str, Any]], *, seen_count: int = 0) -> list[str]:
    ids: list[str] = []
    for idx, meta in enumerate(metas):
        checked = _validate_meta_for_identifier_only(meta, index=idx)
        ids.append(_safe_sample_id(checked, f"sample_{seen_count + idx}"))
    return ids


def make_snapshot_row(
    *,
    sample_ids: Sequence[str],
    reader_outputs: Mapping[str, Any],
    snapshot_id: str,
    epoch: int | None,
    float_digits: int = 6,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "snapshot_id": str(snapshot_id),
        "epoch": None if epoch is None else int(epoch),
        "sample_ids": [str(item) for item in sample_ids],
        "reader_out": serialize_reader_outputs(reader_outputs, float_digits=int(float_digits)),
        "diagnostic_only": True,
        "uses_gt": False,
        "uses_teacher": False,
        "uses_oracle": False,
        "uses_cache": False,
        "uses_raw_prediction": False,
        "metric_claim_allowed": False,
        "paper_claim_allowed": False,
        "runtime_flops_claim_allowed": False,
        "deploy_claim_allowed": False,
    }


class ReaderOutputHook:
    def __init__(self) -> None:
        self.latest: Mapping[str, Any] | None = None

    def __call__(self, _module, _inputs, output) -> None:
        if not isinstance(output, Mapping):
            raise ValueError("PC-OT-MRAS reader hook expected mapping output")
        self.latest = output

    def pop(self) -> Mapping[str, Any]:
        if self.latest is None:
            raise RuntimeError("PC-OT-MRAS reader hook did not capture output")
        out = self.latest
        self.latest = None
        return out


def _model_reader_module(model: torch.nn.Module) -> torch.nn.Module:
    module = model.module if hasattr(model, "module") else model
    reader = getattr(module, "pc_ot_mras_reader", None)
    if reader is None:
        raise ValueError("model has no pc_ot_mras_reader")
    return reader


def _device_from_arg(device_text: str) -> torch.device:
    if device_text == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_text)


def _strip_module_prefix_from_state_dict(state_dict: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(state_dict, Mapping):
        raise ValueError("checkpoint state_dict must be a mapping")
    if not any(isinstance(key, str) and key.startswith("module.") for key in state_dict.keys()):
        return state_dict
    return {
        key[7:] if isinstance(key, str) and key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def _load_checkpoint_state(model: torch.nn.Module, checkpoint_path: str | Path, *, use_ema: bool | None) -> int | None:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must be a mapping")
    if use_ema is True:
        key = "state_dict_ema"
    elif use_ema is False:
        key = "state_dict"
    else:
        key = "state_dict_ema" if "state_dict_ema" in checkpoint else "state_dict"
    if key not in checkpoint:
        raise ValueError(f"checkpoint missing {key}")
    state_dict = checkpoint[key]
    if not isinstance(state_dict, Mapping):
        raise ValueError(f"checkpoint {key} must be a mapping")
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        stripped_state_dict = _strip_module_prefix_from_state_dict(state_dict)
        if stripped_state_dict is state_dict:
            raise
        try:
            model.load_state_dict(stripped_state_dict)
        except RuntimeError as stripped_error:
            raise RuntimeError(
                f"failed to strictly load {key} from {checkpoint_path} with original "
                "or module-prefix-stripped keys"
            ) from stripped_error
    epoch = checkpoint.get("epoch")
    return None if epoch is None else int(epoch)


def _move_batch_to_device(data_dict: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    out = dict(data_dict)
    for key in ("inputs", "masks"):
        value = out.get(key)
        if not torch.is_tensor(value):
            raise ValueError(f"batch missing tensor '{key}'")
        out[key] = value.to(device, non_blocking=True)
    return out


def dump_reader_snapshots(
    *,
    config: str | Path,
    checkpoint: str | Path,
    output_jsonl: str | Path,
    summary_json: str | Path | None = None,
    split: str = "val",
    limit_batches: int = 1,
    device: str = "auto",
    snapshot_id: str | None = None,
    use_ema: bool | None = None,
    float_digits: int = 6,
    use_amp: bool = False,
) -> dict[str, Any]:
    from mmengine.config import Config
    from opentad.datasets import build_dataloader, build_dataset
    from opentad.models import build_detector
    from opentad.models.utils.pc_ot_mras_raw_prediction_guard import assert_no_raw_prediction_shortcut_for_pc_ot_mras

    if int(limit_batches) <= 0:
        raise ValueError("limit_batches must be positive")
    cfg = Config.fromfile(str(config))
    assert_no_raw_prediction_shortcut_for_pc_ot_mras(cfg)
    if not hasattr(cfg, "dataset") or split not in cfg.dataset:
        raise ValueError(f"config missing dataset.{split}")
    if not hasattr(cfg, "solver") or split not in cfg.solver:
        raise ValueError(f"config missing solver.{split}")

    torch_device = _device_from_arg(str(device))
    dataset = build_dataset(cfg.dataset[split], default_args=dict(logger=None))
    loader = build_dataloader(
        dataset,
        rank=0,
        world_size=1,
        shuffle=False,
        drop_last=False,
        **cfg.solver[split],
    )
    model = build_detector(cfg.model)
    epoch = _load_checkpoint_state(model, checkpoint, use_ema=use_ema)
    model.to(torch_device)
    model.eval()

    hook_state = ReaderOutputHook()
    handle = _model_reader_module(model).register_forward_hook(hook_state)
    output_path = Path(output_jsonl).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    samples_written = 0
    snapshot = snapshot_id or (f"epoch_{epoch}" if epoch is not None else Path(checkpoint).stem)
    try:
        with output_path.open("w", encoding="utf-8") as f:
            for batch_idx, data_dict in enumerate(loader):
                if batch_idx >= int(limit_batches):
                    break
                batch = _move_batch_to_device(data_dict, torch_device)
                metas = data_dict.get("metas")
                if not isinstance(metas, (list, tuple)):
                    raise ValueError("batch metas must be a list/tuple")
                with torch.no_grad():
                    with torch.cuda.amp.autocast(dtype=torch.float16, enabled=bool(use_amp) and torch_device.type == "cuda"):
                        model.forward_test(batch["inputs"], batch["masks"], metas=metas, infer_cfg=cfg.inference)
                reader_outputs = hook_state.pop()
                sample_ids = sample_ids_from_metas(metas, seen_count=samples_written)
                row = make_snapshot_row(
                    sample_ids=sample_ids,
                    reader_outputs=reader_outputs,
                    snapshot_id=snapshot,
                    epoch=epoch,
                    float_digits=int(float_digits),
                )
                f.write(json.dumps(strict_json_value(row), sort_keys=True) + "\n")
                rows_written += 1
                samples_written += len(sample_ids)
    finally:
        handle.remove()

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "decision": READY,
        "config": str(config),
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": epoch,
        "split": str(split),
        "snapshot_id": str(snapshot),
        "output_jsonl": str(output_path),
        "rows_written": int(rows_written),
        "samples_written": int(samples_written),
        "limit_batches": int(limit_batches),
        "diagnostic_only": True,
        "uses_checkpoint": True,
        "uses_gt": False,
        "uses_teacher": False,
        "uses_oracle": False,
        "uses_cache": False,
        "uses_raw_prediction": False,
        "tools_test_allowed": False,
        "detector_map_allowed": False,
        "metric_claim_allowed": False,
        "paper_claim_allowed": False,
        "runtime_flops_claim_allowed": False,
        "deploy_claim_allowed": False,
    }
    if summary_json is not None:
        write_json(summary_json, summary)
    return summary


def _parse_use_ema(value: str) -> bool | None:
    lowered = str(value).lower()
    if lowered in {"auto", "none"}:
        return None
    if lowered in {"1", "true", "yes", "ema"}:
        return True
    if lowered in {"0", "false", "no", "raw"}:
        return False
    raise argparse.ArgumentTypeError("--use-ema must be auto, true, or false")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Dump detached PC-OT-MRAS reader outputs for diagnostic visualization."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--limit-batches", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--snapshot-id")
    parser.add_argument("--use-ema", type=_parse_use_ema, default=None)
    parser.add_argument("--float-digits", type=int, default=6)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args(argv)

    try:
        summary = dump_reader_snapshots(
            config=args.config,
            checkpoint=args.checkpoint,
            output_jsonl=args.output_jsonl,
            summary_json=args.summary_json,
            split=args.split,
            limit_batches=int(args.limit_batches),
            device=args.device,
            snapshot_id=args.snapshot_id,
            use_ema=args.use_ema,
            float_digits=int(args.float_digits),
            use_amp=bool(args.amp),
        )
    except Exception as exc:  # pragma: no cover - CLI guard
        print(json.dumps({"schema_version": SUMMARY_SCHEMA_VERSION, "decision": NO_GO, "error": str(exc)}))
        return 1

    print(json.dumps(strict_json_value(summary), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
