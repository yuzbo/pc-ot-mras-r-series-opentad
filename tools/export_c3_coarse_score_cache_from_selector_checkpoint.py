import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from mmengine.config import Config, DictAction

sys.dont_write_bytecode = True
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


ROUTE_LABELS = ["C3_MAINLINE_OPTIMIZATION", "C3_ORIGINAL_OPTIMIZATION_ROUTE"]


def _to_scalar(value, default=0):
    if value is None:
        return default
    if hasattr(value, "detach"):
        value = value.detach().cpu().item()
    elif hasattr(value, "item"):
        value = value.item()
    return value


def _safe_sigmoid(values):
    values = np.asarray(values, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def _safe_logit(values):
    values = np.clip(np.asarray(values, dtype=np.float32), 1e-6, 1.0 - 1e-6)
    return np.log(values / (1.0 - values)).astype(np.float32)


def _window_start_snippet(meta):
    start_frame = float(_to_scalar(meta.get("window_start_frame", 0.0), 0.0))
    snippet_stride = max(float(_to_scalar(meta.get("snippet_stride", 1.0), 1.0)), 1.0)
    offset_frames = float(_to_scalar(meta.get("offset_frames", 0.0), 0.0))
    return max(0, int(round((start_frame - offset_frames) / snippet_stride)))


class ScoreAccumulator:
    def __init__(self, video_name):
        self.video_name = str(video_name)
        self.logit_sum = np.zeros((0,), dtype=np.float64)
        self.score_sum = np.zeros((0,), dtype=np.float64)
        self.count = np.zeros((0,), dtype=np.int64)

    def _ensure_len(self, length):
        length = int(length)
        if length <= self.count.size:
            return
        pad = length - self.count.size
        self.logit_sum = np.pad(self.logit_sum, (0, pad), mode="constant")
        self.score_sum = np.pad(self.score_sum, (0, pad), mode="constant")
        self.count = np.pad(self.count, (0, pad), mode="constant")

    def add_window(self, global_indices, action_logits=None, action_scores=None):
        global_indices = np.asarray(global_indices, dtype=np.int64).reshape(-1)
        if global_indices.size == 0:
            return
        if action_logits is None and action_scores is None:
            raise ValueError("ScoreAccumulator.add_window requires action_logits or action_scores")
        if action_logits is None:
            action_scores = np.asarray(action_scores, dtype=np.float32).reshape(-1)
            action_logits = _safe_logit(action_scores)
        else:
            action_logits = np.asarray(action_logits, dtype=np.float32).reshape(-1)
            action_scores = _safe_sigmoid(action_logits)
        if action_logits.size != global_indices.size:
            raise ValueError("global_indices and action logits/scores must have the same length")

        keep = global_indices >= 0
        global_indices = global_indices[keep]
        action_logits = action_logits[keep]
        action_scores = action_scores[keep]
        if global_indices.size == 0:
            return
        self._ensure_len(int(global_indices.max()) + 1)
        np.add.at(self.logit_sum, global_indices, action_logits.astype(np.float64))
        np.add.at(self.score_sum, global_indices, action_scores.astype(np.float64))
        np.add.at(self.count, global_indices, 1)

    def finalize(self):
        if self.count.size == 0:
            raise ValueError(f"no scores accumulated for {self.video_name}")
        observed = self.count > 0
        action_logit = np.zeros_like(self.logit_sum, dtype=np.float32)
        action_score = np.full_like(self.score_sum, 0.5, dtype=np.float32)
        action_logit[observed] = (self.logit_sum[observed] / self.count[observed]).astype(np.float32)
        action_score[observed] = (self.score_sum[observed] / self.count[observed]).astype(np.float32)
        return {
            "video_name": self.video_name,
            "action_logit": action_logit,
            "action_score": np.clip(action_score, 0.0, 1.0).astype(np.float32),
            "count": self.count.astype(np.int64),
            "observed_fraction": float(observed.mean()) if observed.size else 0.0,
        }


def _load_checkpoint_state_dict(checkpoint_path, device, use_ema=False):
    import torch

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if use_ema and isinstance(checkpoint, dict) and "state_dict_ema" in checkpoint:
        return checkpoint["state_dict_ema"], checkpoint
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"], checkpoint
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"], checkpoint
    return checkpoint, checkpoint


def _load_state_dict_flex(model, state_dict):
    if not isinstance(state_dict, dict):
        raise AssertionError("checkpoint does not contain a state_dict-like mapping")
    try:
        return model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        stripped = {}
        for key, value in state_dict.items():
            stripped[key[7:] if key.startswith("module.") else key] = value
        return model.load_state_dict(stripped, strict=True)


def _move_batch_to_device(data_dict, device):
    moved = {}
    for key, value in data_dict.items():
        if hasattr(value, "to") and key in {"inputs", "masks"}:
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def _split_cfg_from_source(cfg, split, window_size, window_overlap_ratio, batch_gt=False):
    source = cfg.dataset[split]
    collect_keys = ["masks", "gt_segments", "gt_labels"] if batch_gt else ["masks"]
    tensor_keys = ["imgs", "gt_segments", "gt_labels"] if batch_gt else ["imgs"]
    return dict(
        type="ThumosSlidingDataset",
        ann_file=source.ann_file,
        subset_name=source.subset_name,
        data_path=source.data_path,
        class_map=source.class_map,
        filter_gt=source.get("filter_gt", False),
        class_agnostic=source.get("class_agnostic", False),
        block_list=source.get("block_list", None),
        test_mode=not batch_gt,
        feature_stride=source.feature_stride,
        sample_stride=source.get("sample_stride", 1),
        offset_frames=source.get("offset_frames", 0),
        window_size=int(window_size),
        window_overlap_ratio=float(window_overlap_ratio),
        ioa_thresh=0.0,
        fps=source.get("fps", -1),
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="sliding_window", scale_factor=1),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=tensor_keys),
            dict(
                type="Collect",
                inputs="imgs",
                keys=collect_keys,
                meta_keys=[
                    "video_name",
                    "data_path",
                    "fps",
                    "duration",
                    "snippet_stride",
                    "window_start_frame",
                    "window_size",
                    "offset_frames",
                ],
            ),
        ],
    )


def _extract_action_logits(selector, inputs, masks):
    dense_masks = selector._normalize_dense_masks(masks, inputs)
    scout_inputs = selector._build_scout_inputs(inputs)
    scout_outputs = selector.scout(scout_inputs, dense_masks)
    if hasattr(selector, "_sanitize_scout_outputs"):
        scout_outputs = selector._sanitize_scout_outputs(scout_outputs, dense_masks)
    if "action_logits" not in scout_outputs:
        raise AssertionError("selector scout output does not contain action_logits")
    return scout_outputs["action_logits"].float(), dense_masks


def _write_cache(output_dir, accumulators, args, checkpoint_meta=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    videos = {}
    for video_name in sorted(accumulators):
        record = accumulators[video_name].finalize()
        out_name = f"{video_name}.npz"
        out_path = output_dir / out_name
        if out_path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing score file: {out_path}")
        np.savez(
            out_path,
            action_score=record["action_score"],
            action_logit=record["action_logit"],
            count=record["count"],
            video_name=video_name,
            axis="global_snippet_index",
        )
        videos[video_name] = {
            "file": out_name,
            "num_frames": int(record["action_score"].size),
            "observed_fraction": record["observed_fraction"],
        }

    manifest = {
        "schema_version": 1,
        "route_labels": ROUTE_LABELS,
        "score_source": args.score_source,
        "uses_gt": False,
        "axis": "global_snippet_index",
        "videos": videos,
        "exporter": {
            "config": str(args.config),
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": checkpoint_meta.get("epoch") if isinstance(checkpoint_meta, dict) else None,
            "splits": args.splits,
            "window_size": int(args.window_size),
            "window_overlap_ratio": float(args.window_overlap_ratio),
            "use_ema": bool(args.use_ema),
        },
    }
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing manifest: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def export_score_cache(args):
    import torch

    from opentad.datasets import build_dataloader, build_dataset
    from opentad.models import build_detector
    from opentad.utils import set_seed

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    set_seed(args.seed)

    device = torch.device(args.device)
    model = build_detector(cfg.model).to(device)
    state_dict, checkpoint = _load_checkpoint_state_dict(args.checkpoint, device, use_ema=args.use_ema)
    _load_state_dict_flex(model, state_dict)
    model.eval()
    if not getattr(model, "with_frame_selector", False):
        raise AssertionError("export requires a model.frame_selector")
    selector = model.frame_selector
    selector.eval()
    if not hasattr(selector, "scout") or not hasattr(selector, "_build_scout_inputs"):
        raise AssertionError("export requires a pre-backbone selector with scout and _build_scout_inputs")

    accumulators = {}
    per_split = {}
    with torch.no_grad():
        for split in args.splits:
            split_cfg = _split_cfg_from_source(cfg, split, args.window_size, args.window_overlap_ratio, batch_gt=False)
            dataset = build_dataset(split_cfg, default_args=dict(logger=None))
            solver_cfg = dict(cfg.solver.get("test", {}))
            solver_cfg["batch_size"] = int(args.batch_size)
            solver_cfg["num_workers"] = int(args.num_workers)
            loader = build_dataloader(
                dataset,
                rank=0,
                world_size=1,
                shuffle=False,
                drop_last=False,
                **solver_cfg,
            )
            window_count = 0
            for batch_idx, data_dict in enumerate(loader):
                if args.limit_batches is not None and batch_idx >= args.limit_batches:
                    break
                data_dict = _move_batch_to_device(data_dict, device)
                logits, dense_masks = _extract_action_logits(selector, data_dict["inputs"], data_dict["masks"])
                logits = logits.detach().cpu().numpy()
                valid_lengths = dense_masks.detach().cpu().sum(dim=1).numpy().astype(np.int64)
                for row_idx, meta in enumerate(data_dict.get("metas", [])):
                    video_name = str(meta["video_name"])
                    start = _window_start_snippet(meta)
                    valid_len = int(valid_lengths[row_idx])
                    global_indices = start + np.arange(valid_len, dtype=np.int64)
                    accumulators.setdefault(video_name, ScoreAccumulator(video_name)).add_window(
                        global_indices=global_indices,
                        action_logits=logits[row_idx, :valid_len],
                    )
                    window_count += 1
            per_split[split] = {"windows": int(window_count), "dataset_len": int(len(dataset))}

    manifest = _write_cache(args.output_dir, accumulators, args, checkpoint_meta=checkpoint)
    summary = {
        "status": "ok",
        "route_labels": ROUTE_LABELS,
        "uses_gt": False,
        "axis": "global_snippet_index",
        "videos": len(manifest["videos"]),
        "output_dir": str(args.output_dir),
        "manifest": str(Path(args.output_dir) / "manifest.json"),
        "per_split": per_split,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export deploy-visible C3 coarse action score cache from a trained pre-backbone selector checkpoint."
    )
    parser.add_argument("config", help="source C3/CADF selector config")
    parser.add_argument("checkpoint", help="trained selector/detector checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--score-source", default="c3_coarse_classifier_selector_checkpoint")
    parser.add_argument("--splits", nargs="+", default=["train", "test"], choices=["train", "val", "test"])
    parser.add_argument("--window-size", type=int, default=768)
    parser.add_argument("--window-overlap-ratio", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, help="override settings")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))
    export_score_cache(args)


if __name__ == "__main__":
    main()
