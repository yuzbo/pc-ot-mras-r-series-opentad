from __future__ import annotations

import argparse
import functools
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SCHEMA_VERSION = "native_irregular_area_head_p2_proposal_factor_dump_v0"
SUMMARY_SCHEMA_VERSION = "native_irregular_area_head_p2_proposal_factor_summary_v0"
READY = "NATIVE_IRREGULAR_AREA_HEAD_P2_PROPOSAL_FACTOR_DUMP_READY"
NO_GO = "NATIVE_IRREGULAR_AREA_HEAD_P2_PROPOSAL_FACTOR_DUMP_NO_GO"


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


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(strict_json_value(dict(payload)), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _stats(values: Sequence[float]) -> dict[str, Any]:
    finite = [float(item) for item in values if math.isfinite(float(item))]
    if not finite:
        return {"count": 0, "min": None, "mean": None, "max": None}
    return {
        "count": len(finite),
        "min": min(finite),
        "mean": sum(finite) / float(len(finite)),
        "max": max(finite),
    }


def _to_plain(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "tolist"):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist") and not isinstance(value, (list, tuple, Mapping, str, bytes)):
        try:
            return value.tolist()
        except TypeError:
            pass
    if isinstance(value, Mapping):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    return value


def _segments_from_meta(meta: Mapping[str, Any]) -> list[list[float]]:
    for key in ("gt_segments", "gt_segment", "segments"):
        if key not in meta:
            continue
        value = _to_plain(meta[key])
        if not isinstance(value, list):
            continue
        if value and all(isinstance(item, (int, float)) for item in value) and len(value) == 2:
            start, end = _as_float(value[0]), _as_float(value[1])
            return [] if start is None or end is None else [[start, end]]
        out: list[list[float]] = []
        for item in value:
            if not isinstance(item, list) or len(item) < 2:
                continue
            start, end = _as_float(item[0]), _as_float(item[1])
            if start is not None and end is not None and end > start:
                out.append([start, end])
        return out
    return []


def _max_iou(segment: Sequence[float], gt_segments: Sequence[Sequence[float]]) -> float | None:
    if len(segment) < 2 or not gt_segments:
        return None
    start, end = float(segment[0]), float(segment[1])
    best = 0.0
    for gt in gt_segments:
        if len(gt) < 2:
            continue
        gt_start, gt_end = float(gt[0]), float(gt[1])
        inter = max(0.0, min(end, gt_end) - max(start, gt_start))
        union = max(0.0, end - start) + max(0.0, gt_end - gt_start) - inter
        if union > 0.0:
            best = max(best, inter / union)
    return float(best)


class _RPNInputCapture:
    def __init__(self) -> None:
        self.latest_inputs: tuple[Any, ...] | None = None
        self.latest_kwargs: Mapping[str, Any] = {}

    def __call__(self, _module, inputs, kwargs=None) -> None:
        self.latest_inputs = tuple(inputs)
        self.latest_kwargs = dict(kwargs or {})

    def pop(self) -> tuple[Any, ...]:
        if self.latest_inputs is None:
            raise RuntimeError("RPN head input hook did not capture proposal features")
        inputs = self.latest_inputs
        kwargs = dict(self.latest_kwargs)
        self.latest_inputs = None
        self.latest_kwargs = {}
        return inputs, kwargs


def _module(model: Any) -> Any:
    return model.module if hasattr(model, "module") else model


def _register_rpn_pre_hook(rpn_head: Any, capture: _RPNInputCapture):
    original = rpn_head.forward_test

    @functools.wraps(original)
    def wrapped_forward_test(*args, **kwargs):
        capture(None, args, kwargs)
        return original(*args, **kwargs)

    rpn_head.forward_test = wrapped_forward_test

    class _Handle:
        def remove(self) -> None:
            rpn_head.forward_test = original

    return _Handle()


def run_checkpoint_dump(
    *,
    config: str | Path,
    checkpoint: str | Path,
    output_jsonl: str | Path,
    summary_json: str | Path | None = None,
    split: str = "val",
    limit_batches: int = 1,
    device: str = "auto",
    use_ema: bool | None = None,
    use_amp: bool = False,
    include_gt_iou: bool = True,
    topk_per_sample: int | None = 200,
) -> dict[str, Any]:
    try:
        import torch
        from mmengine.config import Config
        from opentad.datasets import build_dataloader, build_dataset
        from opentad.models import build_detector
        from opentad.models.utils.pc_ot_mras_raw_prediction_guard import assert_no_raw_prediction_shortcut_for_pc_ot_mras
        from tools.bata.dump_pc_ot_mras_reader_snapshots import (
            _device_from_arg,
            _load_checkpoint_state,
            _move_batch_to_device,
            sample_ids_from_metas,
        )
    except Exception as exc:
        raise RuntimeError(f"OpenTAD proposal-factor dump imports failed: {exc}") from None

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
    loader = build_dataloader(dataset, rank=0, world_size=1, shuffle=False, drop_last=False, **cfg.solver[split])
    model = build_detector(cfg.model)
    epoch = _load_checkpoint_state(model, checkpoint, use_ema=use_ema)
    model.to(torch_device)
    model.eval()

    module = _module(model)
    rpn_head = getattr(module, "rpn_head", None)
    if rpn_head is None or not hasattr(rpn_head, "dump_proposal_factors"):
        raise ValueError("model.rpn_head has no dump_proposal_factors method")

    out_path = Path(output_jsonl).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    capture = _RPNInputCapture()
    handle = _register_rpn_pre_hook(rpn_head, capture)
    total_rows = 0
    total_samples = 0
    final_scores: list[float] = []
    observed_fractions: list[float] = []
    max_gt_ious: list[float] = []
    rows_per_sample: dict[str, int] = {}
    try:
        with out_path.open("w", encoding="utf-8") as f:
            for batch_idx, data_dict in enumerate(loader):
                if batch_idx >= int(limit_batches):
                    break
                batch = _move_batch_to_device(data_dict, torch_device)
                metas = data_dict.get("metas")
                if not isinstance(metas, (list, tuple)):
                    raise ValueError("batch metas must be a list/tuple")
                sample_ids = sample_ids_from_metas(metas, seen_count=total_samples)
                with torch.no_grad():
                    with torch.cuda.amp.autocast(
                        dtype=torch.float16,
                        enabled=bool(use_amp) and torch_device.type == "cuda",
                    ):
                        module.forward_test(batch["inputs"], batch["masks"], metas=metas, infer_cfg=cfg.inference)
                captured_inputs, captured_kwargs = capture.pop()
                if len(captured_inputs) < 2:
                    raise RuntimeError("captured RPN input does not contain feature and mask lists")
                feat_list, mask_list = captured_inputs[0], captured_inputs[1]
                head_metas = captured_kwargs.get("metas", metas)
                rows = rpn_head.dump_proposal_factors(feat_list, mask_list, metas=head_metas)
                gt_by_video: dict[str, list[list[float]]] = {}
                if include_gt_iou:
                    for sample_idx, meta in enumerate(metas):
                        video_id = str(meta.get("video_name", sample_ids[sample_idx])) if isinstance(meta, Mapping) else sample_ids[sample_idx]
                        gt_by_video[video_id] = _segments_from_meta(meta) if isinstance(meta, Mapping) else []
                if topk_per_sample is not None:
                    grouped: dict[str, list[Mapping[str, Any]]] = {}
                    for row in rows:
                        grouped.setdefault(str(row.get("video_id", "unknown")), []).append(row)
                    rows = []
                    for video_rows in grouped.values():
                        rows.extend(
                            sorted(
                                video_rows,
                                key=lambda item: float(item.get("final_score", float("-inf"))),
                                reverse=True,
                            )[: int(topk_per_sample)]
                        )
                for row in rows:
                    video_id = str(row.get("video_id", "unknown"))
                    out_row = {
                        "schema_version": SCHEMA_VERSION,
                        "snapshot_id": f"epoch_{epoch}" if epoch is not None else Path(checkpoint).stem,
                        "epoch": epoch,
                        "batch_idx": int(batch_idx),
                        "diagnostic_only": True,
                        "uses_teacher": False,
                        "uses_oracle": False,
                        "uses_cache": False,
                        "uses_raw_prediction": False,
                        "metric_claim_allowed": False,
                        "paper_claim_allowed": False,
                        **dict(row),
                    }
                    if include_gt_iou:
                        iou = _max_iou(row.get("segment", []), gt_by_video.get(video_id, []))
                        out_row["uses_gt_for_diagnostic_iou"] = iou is not None
                        out_row["diagnostic_max_gt_iou"] = iou
                        if iou is not None:
                            max_gt_ious.append(float(iou))
                    score = _as_float(row.get("final_score"))
                    if score is not None:
                        final_scores.append(score)
                    observed = _as_float(row.get("observed_fraction"))
                    if observed is not None:
                        observed_fractions.append(observed)
                    rows_per_sample[video_id] = rows_per_sample.get(video_id, 0) + 1
                    total_rows += 1
                    f.write(json.dumps(strict_json_value(out_row), sort_keys=True) + "\n")
                total_samples += len(sample_ids)
    finally:
        handle.remove()

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "decision": READY,
        "config": str(config),
        "checkpoint": str(checkpoint),
        "split": str(split),
        "limit_batches": int(limit_batches),
        "epoch": epoch,
        "output_jsonl": str(out_path),
        "samples_seen": int(total_samples),
        "rows_written": int(total_rows),
        "rows_per_sample": rows_per_sample,
        "final_score": _stats(final_scores),
        "observed_fraction": _stats(observed_fractions),
        "diagnostic_max_gt_iou": _stats(max_gt_ious),
        "diagnostic_only": True,
        "uses_gt_for_diagnostic_iou": bool(include_gt_iou),
        "metric_claim_allowed": False,
        "paper_claim_allowed": False,
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


def error_payload(exc: BaseException) -> dict[str, Any]:
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "decision": NO_GO,
        "error_type": exc.__class__.__name__,
        "error": str(exc),
        "diagnostic_only": True,
        "metric_claim_allowed": False,
        "paper_claim_allowed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dump read-only NativeIrregularAreaHeadP2 proposal factors.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--limit-batches", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--use-ema", type=_parse_use_ema, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--topk-per-sample", type=int, default=200)
    parser.add_argument("--no-gt-iou", action="store_true")
    args = parser.parse_args(argv)

    try:
        summary = run_checkpoint_dump(
            config=args.config,
            checkpoint=args.checkpoint,
            output_jsonl=args.output_jsonl,
            summary_json=args.summary_json,
            split=args.split,
            limit_batches=int(args.limit_batches),
            device=args.device,
            use_ema=args.use_ema,
            use_amp=bool(args.amp),
            include_gt_iou=not bool(args.no_gt_iou),
            topk_per_sample=args.topk_per_sample,
        )
    except Exception as exc:  # pragma: no cover - CLI guard
        print(json.dumps(strict_json_value(error_payload(exc)), sort_keys=True))
        return 1

    print(json.dumps(strict_json_value(summary), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
