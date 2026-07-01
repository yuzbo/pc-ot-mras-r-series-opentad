import copy
import hashlib
import os
import pickle
import random
import torch
import random
import pandas as pd
import numpy as np

from ..builder import PIPELINES
from torch.nn import functional as F
from .boundary_acquisition import load_value_transport_selection_ledger


def _stable_string_seed(value):
    if value is None:
        value = "unknown"
    if not isinstance(value, str):
        value = str(value)
    digest = hashlib.sha1(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


@PIPELINES.register_module()
class PrepareVideoInfo:
    def __init__(self, format="mp4", modality="RGB", prefix=""):
        self.format = format
        self.modality = modality
        self.prefix = prefix

    def __call__(self, results):
        results["modality"] = self.modality
        results["filename"] = os.path.join(
            results["data_path"],
            self.prefix + results["video_name"] + "." + self.format,
        )
        return results


@PIPELINES.register_module()
class LoadSnippetFrames:
    """Load the snippet frame, the output should follows the format:
    snippet_num x channel x clip_len x height x width
    """

    def __init__(
        self,
        clip_len,
        frame_interval=1,
        method="resize",
        trunc_len=None,
        trunc_thresh=None,
        crop_ratio=None,
    ):
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.method = method  # resize or padding or sliding window
        # todo: support to  change FPS
        # random_trunc settings
        self.trunc_len = trunc_len
        self.trunc_thresh = trunc_thresh
        self.crop_ratio = crop_ratio

    def random_trunc(self, feats, trunc_len, gt_segments, gt_labels, offset=0, max_num_trials=200):
        feat_len = feats.shape[0]
        num_segs = gt_segments.shape[0]

        trunc_len = trunc_len
        if feat_len <= trunc_len:
            if self.crop_ratio == None:  # do nothing
                return feats, gt_segments, gt_labels
            else:  # randomly crop the seq by setting trunc_len to a value in [l, r]
                trunc_len = random.randint(
                    max(round(self.crop_ratio[0] * feat_len), 1),
                    min(round(self.crop_ratio[1] * feat_len), feat_len),
                )
                # corner case
                if feat_len == trunc_len:
                    return feats, gt_segments, gt_labels

        # try a few times till a valid truncation with at least one action
        for _ in range(max_num_trials):
            # sample a random truncation of the video feats
            st = random.randint(0, feat_len - trunc_len)
            ed = st + trunc_len
            window = np.array([st, ed], dtype=np.float32)

            # compute the intersection between the sampled window and all segments
            window = np.repeat(window[None, :], num_segs, axis=0)
            left = np.maximum(window[:, 0] - offset, gt_segments[:, 0])
            right = np.minimum(window[:, 1] + offset, gt_segments[:, 1])
            inter = np.clip(right - left, a_min=0, a_max=None)
            area_segs = np.abs(gt_segments[:, 1] - gt_segments[:, 0])
            inter_ratio = inter / area_segs

            # only select those segments over the thresh
            seg_idx = inter_ratio >= self.trunc_thresh

            # with at least one action
            if seg_idx.sum().item() > 0:
                break

        feats = feats[st:ed]
        gt_segments = np.stack((left[seg_idx], right[seg_idx]), axis=1)  # [N,2] in feature grids
        gt_segments = gt_segments - st  # shift the time stamps due to truncation
        gt_labels = gt_labels[seg_idx]  # [N]
        return feats, gt_segments, gt_labels

    def __call__(self, results):
        assert "total_frames" in results.keys(), "should have total_frames as a key"
        total_frames = results["total_frames"]
        fps = results["avg_fps"]

        if self.method == "resize":
            assert "resize_length" in results.keys(), "should have resize_length as a key"
            snippet_num = results["resize_length"]
            snippet_stride = total_frames / snippet_num
            snippet_center = np.arange(
                snippet_stride / 2 - 0.5,
                total_frames + snippet_stride / 2 - 0.5,
                snippet_stride,
            )
            masks = torch.ones(results["resize_length"]).bool()

            # don't forget to resize the ground truth segments
            if "gt_segments" in results.keys():
                # convert gt seconds to feature grid
                results["gt_segments"] = np.clip(results["gt_segments"] / results["duration"], 0.0, 1.0)
                results["gt_segments"] *= results["resize_length"]

        elif self.method == "random_trunc":
            snippet_num = self.trunc_len
            snippet_center = np.arange(0, total_frames, results["snippet_stride"])

            # trunc the snippet_center
            snippet_center, gt_segments, gt_labels = self.random_trunc(
                snippet_center,
                trunc_len=snippet_num,
                gt_segments=results["gt_segments"],
                gt_labels=results["gt_labels"],
            )

            # update the gt_segments
            results["gt_segments"] = gt_segments
            results["gt_labels"] = gt_labels

            # pad the snippet_center
            if len(snippet_center) < snippet_num:
                valid_len = len(snippet_center)
                snippet_center = np.pad(snippet_center, (0, snippet_num - valid_len), mode="edge")
                masks = torch.cat([torch.ones(valid_len), torch.zeros(snippet_num - valid_len)]).bool()
            else:
                masks = torch.ones(snippet_num).bool()

        elif self.method == "sliding_window":
            snippet_num = results["window_size"]
            snippet_center = np.arange(0, total_frames, results["snippet_stride"])

            start_idx = min(results["feature_start_idx"], len(snippet_center))
            end_idx = min((results["feature_end_idx"] + 1), len(snippet_center))

            snippet_center = snippet_center[start_idx:end_idx]

            if len(snippet_center) < snippet_num:
                valid_len = len(snippet_center)
                snippet_center = np.pad(snippet_center, (0, snippet_num - valid_len), mode="edge")
                masks = torch.cat([torch.ones(valid_len), torch.zeros(snippet_num - valid_len)]).bool()
            else:
                masks = torch.ones(snippet_num).bool()
        elif self.method == "padding":
            raise NotImplementedError

        # extend snippet center to a clip
        clip_idxs = np.arange(-(self.clip_len // 2), self.clip_len // 2)
        frame_idxs = snippet_center[:, None] + self.frame_interval * clip_idxs[None, :]  # [snippet_num, clip_len]

        # truncate to [0, total_frames-1], and round to int
        frame_idxs = np.clip(frame_idxs, 0, total_frames - 1).round()

        assert frame_idxs.shape[0] == snippet_num, "snippet center number should be equal to snippet number"
        assert frame_idxs.shape[1] == self.clip_len, "snippet length should be equal to clip length"

        results["frame_inds"] = frame_idxs.astype(int)
        results["num_clips"] = snippet_num
        results["clip_len"] = self.clip_len
        results["masks"] = masks
        return results


@PIPELINES.register_module()
class LoadFrames:
    def __init__(
        self,
        num_clips=1,
        scale_factor=1,
        method="resize",
        trunc_len=None,
        trunc_thresh=None,
        crop_ratio=None,
        keep_ratio=0.5,
        method_base=None,
        target_len=None,
        source_len=None,
        remap_gt_to_selected_axis=True,
        bata_value_transport_ledger_path=None,
        bata_value_transport_allow_missing_fallback=False,
        bata_value_transport_require_deployable=True,
        bata_value_transport_require_selected_count=None,
        bata_value_transport_allow_short_valid_ratio_count=False,
        bata_value_transport_source="pc_ot_mras_frontend_hard_positions",
        bata_value_transport_config_hash="",
    ):
        self.num_clips = num_clips
        self.scale_factor = scale_factor  # multiply by the frame number, if backbone has downsampling
        self.method = method  # resize or padding or random_trunc or sliding_window or value-transport ledger
        # random_trunc settings
        self.trunc_len = trunc_len
        self.trunc_thresh = trunc_thresh
        self.crop_ratio = crop_ratio
        self.keep_ratio = keep_ratio
        self.method_base = method_base
        self.target_len = target_len
        self.source_len = source_len
        self.remap_gt_to_selected_axis = bool(remap_gt_to_selected_axis)
        self.bata_value_transport_ledger_path = bata_value_transport_ledger_path
        self.bata_value_transport_allow_missing_fallback = bool(bata_value_transport_allow_missing_fallback)
        self.bata_value_transport_require_deployable = bool(bata_value_transport_require_deployable)
        self.bata_value_transport_require_selected_count = bata_value_transport_require_selected_count
        self.bata_value_transport_allow_short_valid_ratio_count = bool(
            bata_value_transport_allow_short_valid_ratio_count
        )
        self.bata_value_transport_source = bata_value_transport_source
        self.bata_value_transport_config_hash = bata_value_transport_config_hash
        self._bata_value_transport_ledger = None

    def random_trunc(self, feats, trunc_len, gt_segments, gt_labels, offset=0, max_num_trials=200):
        feat_len = feats.shape[0]
        num_segs = gt_segments.shape[0]

        trunc_len = trunc_len
        if feat_len <= trunc_len:
            if self.crop_ratio == None:  # do nothing
                return feats, gt_segments, gt_labels
            else:  # randomly crop the seq by setting trunc_len to a value in [l, r]
                trunc_len = random.randint(
                    max(round(self.crop_ratio[0] * feat_len), 1),
                    min(round(self.crop_ratio[1] * feat_len), feat_len),
                )
                # corner case
                if feat_len == trunc_len:
                    return feats, gt_segments, gt_labels

        # try a few times till a valid truncation with at least one action
        for _ in range(max_num_trials):
            # sample a random truncation of the video feats
            st = random.randint(0, feat_len - trunc_len)
            ed = st + trunc_len
            window = np.array([st, ed], dtype=np.float32)

            # compute the intersection between the sampled window and all segments
            window = np.repeat(window[None, :], num_segs, axis=0)
            left = np.maximum(window[:, 0] - offset, gt_segments[:, 0])
            right = np.minimum(window[:, 1] + offset, gt_segments[:, 1])
            inter = np.clip(right - left, a_min=0, a_max=None)
            area_segs = np.abs(gt_segments[:, 1] - gt_segments[:, 0])
            inter_ratio = inter / area_segs

            # only select those segments over the thresh
            seg_idx = inter_ratio >= self.trunc_thresh

            # with at least one action
            if seg_idx.sum().item() > 0:
                break

        feats = feats[st:ed]
        gt_segments = np.stack((left[seg_idx], right[seg_idx]), axis=1)  # [N,2] in feature grids
        gt_segments = gt_segments - st  # shift the time stamps due to truncation
        gt_labels = gt_labels[seg_idx]  # [N]
        return feats, gt_segments, gt_labels

    def _map_coord_to_selected_axis(self, coord, kept_positions, valid_len):
        if kept_positions.size == 0:
            return 0.0
        xp = np.concatenate([kept_positions.astype(np.float32), np.array([float(valid_len)], dtype=np.float32)])
        fp = np.concatenate(
            [np.arange(kept_positions.size, dtype=np.float32), np.array([float(kept_positions.size)], dtype=np.float32)]
        )
        coord = float(np.clip(coord, 0.0, float(valid_len)))
        return float(np.interp(coord, xp, fp))

    def _remap_gt_to_selected_axis(self, gt_segments, gt_labels, kept_positions, valid_len):
        if gt_segments is None or gt_labels is None or len(gt_segments) == 0 or kept_positions.size == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int32)

        remapped_segments = []
        remapped_labels = []
        max_coord = float(kept_positions.size)
        for idx, seg in enumerate(gt_segments):
            start = self._map_coord_to_selected_axis(seg[0], kept_positions, valid_len)
            end = self._map_coord_to_selected_axis(seg[1], kept_positions, valid_len)
            start = float(np.clip(start, 0.0, max_coord))
            end = float(np.clip(end, 0.0, max_coord))
            if end <= start:
                end = min(max_coord, start + 1e-3)
            if end > start:
                remapped_segments.append([start, end])
                remapped_labels.append(int(gt_labels[idx]))

        if len(remapped_segments) == 0:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int32)
        return np.asarray(remapped_segments, dtype=np.float32), np.asarray(remapped_labels, dtype=np.int32)

    def _set_irregular_axis_meta(self, results, kept_positions, valid_len):
        scale = float(max(self.scale_factor, 1))
        selected_positions = np.asarray(kept_positions, dtype=np.float32) / scale
        results["irregular_selected_positions"] = selected_positions
        results["selected_dense_indices"] = selected_positions
        results["selected_valid_len"] = int(len(kept_positions))
        results["irregular_selected_valid_len"] = float(valid_len) / scale
        results["irregular_dense_valid_len"] = float(valid_len) / scale
        results["remap_gt_to_selected_axis"] = bool(self.remap_gt_to_selected_axis)
        results["gt_remapped_to_selected_axis"] = bool(self.remap_gt_to_selected_axis)
        results["irregular_native_axis"] = bool(not self.remap_gt_to_selected_axis)

    def _select_random_fixed_positions(self, valid_len, target_frame_num, sample_key):
        valid_len = int(valid_len)
        target_frame_num = int(target_frame_num)
        if valid_len <= 0 or target_frame_num <= 0:
            return np.zeros((0,), dtype=np.int64)
        if target_frame_num >= valid_len:
            return np.arange(valid_len, dtype=np.int64)
        rng = np.random.RandomState(_stable_string_seed(sample_key))
        return np.sort(rng.choice(valid_len, size=target_frame_num, replace=False)).astype(np.int64)

    def _exact_uniform_dense_positions(self, valid_len, dense_frame_num, frame_num):
        valid_len = int(valid_len)
        dense_frame_num = int(dense_frame_num)
        frame_num = int(frame_num)
        if valid_len <= 0 or dense_frame_num <= 0 or frame_num <= 0:
            return np.zeros((0,), dtype=np.int64)
        selected_valid_len = int(np.ceil(float(valid_len) * float(frame_num) / float(dense_frame_num)))
        selected_valid_len = max(1, min(selected_valid_len, frame_num))
        step = float(dense_frame_num) / float(frame_num)
        positions = [int(round(float(idx) * step)) for idx in range(selected_valid_len)]
        positions = [int(np.clip(pos, 0, valid_len - 1)) for pos in positions]
        return np.asarray(sorted(dict.fromkeys(positions)), dtype=np.int64)

    def _value_transport_ledger(self):
        if self._bata_value_transport_ledger is None:
            self._bata_value_transport_ledger = load_value_transport_selection_ledger(
                self.bata_value_transport_ledger_path,
                require_deployable=self.bata_value_transport_require_deployable,
            )
        return self._bata_value_transport_ledger

    def _value_transport_sample_id(self, results):
        if "window_start_frame" not in results:
            raise ValueError("value_transport_ledger_subsample requires window_start_frame in results")
        return f"{results.get('video_name', 'unknown')}|{int(results['window_start_frame'])}"

    def _lookup_value_transport_positions(self, results, valid_len, dense_frame_num, frame_num):
        sample_id = self._value_transport_sample_id(results)
        row = self._value_transport_ledger().get(sample_id)
        if row is None:
            if not self.bata_value_transport_allow_missing_fallback:
                raise KeyError(f"value-transport ledger missing sample_id={sample_id}")
            positions = self._exact_uniform_dense_positions(valid_len, dense_frame_num, frame_num)
            return positions, dict(sample_id=sample_id, fallback_missing_ledger=True)
        positions = row["selected_positions"].astype(np.int64, copy=False).reshape(-1)
        if positions.size == 0:
            raise ValueError(f"value-transport ledger sample_id={sample_id} selected no positions")
        if positions[0] < 0 or positions[-1] >= int(valid_len):
            raise ValueError(f"value-transport ledger sample_id={sample_id} positions exceed valid dense window")
        return positions, row

    def __call__(self, results):
        assert "total_frames" in results.keys(), "should have total_frames as a key"
        total_frames = results["total_frames"]
        fps = results["avg_fps"]

        if self.method == "resize":
            assert "resize_length" in results.keys(), "should have resize_length as a key"
            frame_num = results["resize_length"] * self.scale_factor
            frame_stride = total_frames / frame_num
            frame_idxs = np.arange(
                frame_stride / 2 - 0.5,
                total_frames + frame_stride / 2 - 0.5,
                frame_stride,
            )
            masks = torch.ones(results["resize_length"]).bool()  # should not multiply by scale_factor

            # don't forget to resize the ground truth segments
            if "gt_segments" in results.keys():
                # convert gt seconds to feature grid
                results["gt_segments"] = np.clip(results["gt_segments"] / results["duration"], 0.0, 1.0)
                results["gt_segments"] *= results["resize_length"]

        elif self.method == "random_trunc":
            assert results["snippet_stride"] >= self.scale_factor, "snippet_stride should be larger than scale_factor"
            assert (
                results["snippet_stride"] % self.scale_factor == 0
            ), "snippet_stride should be divisible by scale_factor"

            frame_num = self.trunc_len * self.scale_factor
            frame_stride = results["snippet_stride"] // self.scale_factor
            frame_idxs = np.arange(0, total_frames, frame_stride)

            # trunc the frame_idxs
            frame_idxs, gt_segments, gt_labels = self.random_trunc(
                frame_idxs,
                trunc_len=frame_num,
                gt_segments=results["gt_segments"] * self.scale_factor,  # gt segment should be mapped to frame level
                gt_labels=results["gt_labels"],
            )
            results["gt_segments"] = gt_segments / self.scale_factor  # convert back to original scale
            results["gt_labels"] = gt_labels

            # pad the frame_idxs
            if len(frame_idxs) < frame_num:
                valid_len = len(frame_idxs) // self.scale_factor
                frame_idxs = np.pad(frame_idxs, (0, frame_num - len(frame_idxs)), mode="edge")
                masks = torch.cat([torch.ones(valid_len), torch.zeros(self.trunc_len - valid_len)]).bool()
            else:
                masks = torch.ones(self.trunc_len).bool()

        elif self.method == "sliding_window":
            assert results["snippet_stride"] >= self.scale_factor, "snippet_stride should be larger than scale_factor"
            assert (
                results["snippet_stride"] % self.scale_factor == 0
            ), "snippet_stride should be divisible by scale_factor"

            window_size = results["window_size"]
            frame_num = window_size * self.scale_factor
            frame_stride = results["snippet_stride"] // self.scale_factor
            frame_idxs = np.arange(0, total_frames, frame_stride)

            start_idx = min(results["feature_start_idx"] * self.scale_factor, len(frame_idxs))
            end_idx = min((results["feature_end_idx"] + 1) * self.scale_factor, len(frame_idxs))

            frame_idxs = frame_idxs[start_idx:end_idx]

            if len(frame_idxs) < frame_num:
                valid_len = len(frame_idxs) // self.scale_factor
                frame_idxs = np.pad(frame_idxs, (0, frame_num - len(frame_idxs)), mode="edge")
                masks = torch.cat([torch.ones(valid_len), torch.zeros(window_size - valid_len)]).bool()
            else:
                masks = torch.ones(window_size).bool()

        elif self.method == "random_fixed_subsample":
            assert results["snippet_stride"] >= self.scale_factor, "snippet_stride should be larger than scale_factor"
            assert (
                results["snippet_stride"] % self.scale_factor == 0
            ), "snippet_stride should be divisible by scale_factor"

            keep_ratio = float(self.keep_ratio)
            frame_stride = results["snippet_stride"] // self.scale_factor
            dense_frame_idxs = np.arange(0, total_frames, frame_stride)
            gt_segments = results["gt_segments"] * self.scale_factor if "gt_segments" in results else None
            gt_labels = results["gt_labels"] if "gt_labels" in results else None

            if self.method_base == "random_trunc":
                if gt_segments is None or gt_labels is None:
                    raise ValueError("random_fixed_subsample with random_trunc requires gt_segments and gt_labels")
                if self.trunc_len is None and self.target_len is None:
                    raise ValueError(
                        "random_fixed_subsample requires trunc_len or target_len when method_base='random_trunc'"
                    )
                target_len = int(self.target_len) if self.target_len is not None else int(self.trunc_len)
                source_len = int(self.source_len) if self.source_len is not None else int(round(target_len / max(keep_ratio, 1e-6)))
                frame_num = target_len * self.scale_factor
                dense_frame_num = source_len * self.scale_factor
                dense_window, gt_segments, gt_labels = self.random_trunc(
                    dense_frame_idxs,
                    trunc_len=dense_frame_num,
                    gt_segments=gt_segments,
                    gt_labels=gt_labels,
                )
            elif self.method_base == "sliding_window":
                if "window_size" not in results:
                    raise ValueError("random_fixed_subsample with sliding_window requires window_size in results")
                dense_window_len = int(results["window_size"])
                target_len = int(self.target_len) if self.target_len is not None else int(round(dense_window_len * keep_ratio))
                frame_num = target_len * self.scale_factor
                dense_frame_num = dense_window_len * self.scale_factor
                start_idx = min(results["feature_start_idx"] * self.scale_factor, len(dense_frame_idxs))
                end_idx = min((results["feature_end_idx"] + 1) * self.scale_factor, len(dense_frame_idxs))
                dense_window = dense_frame_idxs[start_idx:end_idx]
            else:
                raise ValueError("random_fixed_subsample requires method_base='random_trunc' or 'sliding_window'")

            valid_len = int(len(dense_window))
            if valid_len <= 0:
                raise RuntimeError("random_fixed_subsample received an empty dense window")

            sample_key = (
                f"{results.get('video_name', 'unknown')}|random_fixed|"
                f"{int(dense_window[0])}|{int(dense_window[-1])}|{valid_len}|{frame_num}"
            )
            keep_positions = self._select_random_fixed_positions(valid_len, frame_num, sample_key)
            if keep_positions.size == 0:
                keep_positions = np.array([0], dtype=np.int64)

            frame_idxs = dense_window[keep_positions]
            self._set_irregular_axis_meta(results, keep_positions, valid_len)

            if gt_segments is not None and gt_labels is not None:
                if self.remap_gt_to_selected_axis:
                    gt_segments, gt_labels = self._remap_gt_to_selected_axis(
                        gt_segments=gt_segments,
                        gt_labels=gt_labels,
                        kept_positions=keep_positions,
                        valid_len=valid_len,
                    )
                results["gt_segments"] = gt_segments / self.scale_factor
                results["gt_labels"] = gt_labels

            if len(frame_idxs) < frame_num:
                valid_mask_len = min(
                    int(np.ceil(keep_positions.size / max(self.scale_factor, 1))),
                    int(np.ceil(frame_num / max(self.scale_factor, 1))),
                )
                target_mask_len = int(np.ceil(frame_num / self.scale_factor))
                frame_idxs = np.pad(frame_idxs, (0, frame_num - len(frame_idxs)), mode="edge")
                masks = torch.cat([torch.ones(valid_mask_len), torch.zeros(target_mask_len - valid_mask_len)]).bool()
            else:
                masks = torch.ones(int(np.ceil(frame_num / self.scale_factor))).bool()

        elif self.method == "bata_value_transport_ledger_subsample":
            assert results["snippet_stride"] >= self.scale_factor, "snippet_stride should be larger than scale_factor"
            assert (
                results["snippet_stride"] % self.scale_factor == 0
            ), "snippet_stride should be divisible by scale_factor"
            if self.method_base != "sliding_window":
                raise ValueError("bata_value_transport_ledger_subsample currently supports method_base='sliding_window'")
            if "window_size" not in results:
                raise ValueError("bata_value_transport_ledger_subsample requires window_size in results")

            dense_window_len = int(results["window_size"])
            target_len = int(self.target_len) if self.target_len is not None else int(round(dense_window_len * float(self.keep_ratio)))
            frame_num = target_len * self.scale_factor
            dense_frame_num = dense_window_len * self.scale_factor
            frame_stride = results["snippet_stride"] // self.scale_factor
            dense_frame_idxs = np.arange(0, total_frames, frame_stride)
            start_idx = min(results["feature_start_idx"] * self.scale_factor, len(dense_frame_idxs))
            end_idx = min((results["feature_end_idx"] + 1) * self.scale_factor, len(dense_frame_idxs))
            dense_window = dense_frame_idxs[start_idx:end_idx]
            valid_len = int(len(dense_window))
            if valid_len <= 0:
                raise RuntimeError("bata_value_transport_ledger_subsample received an empty dense window")

            keep_positions, ledger_row = self._lookup_value_transport_positions(
                results,
                valid_len=valid_len,
                dense_frame_num=dense_frame_num,
                frame_num=frame_num,
            )
            if keep_positions.size > int(frame_num):
                sample_id = ledger_row.get("sample_id", self._value_transport_sample_id(results))
                raise ValueError(
                    f"value-transport ledger sample_id={sample_id} selects {keep_positions.size} positions "
                    f"but target_len={target_len} allows only {frame_num} frame indices"
                )
            required_count = self.bata_value_transport_require_selected_count
            if required_count is True:
                required_count = int(frame_num)
            elif required_count in (False, None):
                required_count = None
            else:
                required_count = int(required_count)
            if required_count is not None and keep_positions.size != required_count:
                expected_required_count = required_count
                if self.bata_value_transport_allow_short_valid_ratio_count and valid_len < dense_frame_num:
                    expected_required_count = int(
                        np.ceil(float(valid_len) * float(required_count) / float(dense_frame_num))
                    )
                    expected_required_count = max(1, min(expected_required_count, int(required_count), int(valid_len)))
                if keep_positions.size == expected_required_count:
                    required_count = expected_required_count
                else:
                    sample_id = ledger_row.get("sample_id", self._value_transport_sample_id(results))
                    raise ValueError(
                        f"value-transport ledger sample_id={sample_id} selects {keep_positions.size} positions "
                        f"but bata_value_transport_require_selected_count={required_count} "
                        f"(expected={expected_required_count}, valid_len={valid_len}, dense_frame_num={dense_frame_num})"
                    )
            if required_count is not None and keep_positions.size != required_count:
                sample_id = ledger_row.get("sample_id", self._value_transport_sample_id(results))
                raise ValueError(
                    f"value-transport ledger sample_id={sample_id} selects {keep_positions.size} positions "
                    f"but bata_value_transport_require_selected_count={required_count}"
                )

            frame_idxs = dense_window[keep_positions]
            self._set_irregular_axis_meta(results, keep_positions, valid_len)

            gt_segments = results["gt_segments"] * self.scale_factor if "gt_segments" in results else None
            gt_labels = results["gt_labels"] if "gt_labels" in results else None
            if gt_segments is not None and gt_labels is not None:
                if self.remap_gt_to_selected_axis:
                    gt_segments, gt_labels = self._remap_gt_to_selected_axis(
                        gt_segments=gt_segments,
                        gt_labels=gt_labels,
                        kept_positions=keep_positions,
                        valid_len=valid_len,
                    )
                results["gt_segments"] = gt_segments / self.scale_factor
                results["gt_labels"] = gt_labels

            results["bata_selected_dense_indices"] = keep_positions.astype(np.int64)
            results["bata_value_transport_selection_row"] = {
                key: value
                for key, value in dict(ledger_row).items()
                if key
                in {
                    "sample_id",
                    "schema_version",
                    "route",
                    "route_variant",
                    "policy",
                    "valid_len",
                    "dense_len",
                    "target_len",
                    "selected_count",
                    "selected_positions_unit",
                    "diagnostics",
                    "diagnostic_only",
                    "training_only",
                    "diagnostic_uses_train_utility_for_audit",
                    "deploy_selection_ledger",
                    "prediction_uses_gt",
                    "uses_gt",
                    "uses_teacher",
                    "uses_oracle",
                    "uses_cache",
                    "uses_prediction_cache",
                    "uses_raw_prediction",
                    "uses_checkpoint",
                }
            }
            results["bata_score_source"] = self.bata_value_transport_source
            results["bata_value_transport_config_hash"] = self.bata_value_transport_config_hash
            results["bata_diagnostic_only"] = bool(ledger_row.get("diagnostic_only", False))

            if len(frame_idxs) < frame_num:
                valid_mask_len = min(
                    int(np.ceil(keep_positions.size / max(self.scale_factor, 1))),
                    int(np.ceil(frame_num / max(self.scale_factor, 1))),
                )
                target_mask_len = int(np.ceil(frame_num / self.scale_factor))
                frame_idxs = np.pad(frame_idxs, (0, frame_num - len(frame_idxs)), mode="edge")
                masks = torch.cat([torch.ones(valid_mask_len), torch.zeros(target_mask_len - valid_mask_len)]).bool()
            else:
                masks = torch.ones(int(np.ceil(frame_num / self.scale_factor))).bool()

        elif self.method == "padding":
            raise NotImplementedError

        # truncate to [0, total_frames-1], and round to int
        frame_idxs = np.clip(frame_idxs, 0, total_frames - 1).round()

        assert frame_idxs.shape[0] == frame_num, "snippet center number should be equal to snippet number"

        results["frame_inds"] = frame_idxs.astype(int)
        results["num_clips"] = self.num_clips
        results["clip_len"] = frame_num // self.num_clips
        results["masks"] = masks
        return results


@PIPELINES.register_module()
class Interpolate:
    def __init__(self, keys, size=128, mode="linear"):
        self.keys = keys
        self.size = size
        self.mode = mode

    def __call__(self, results):
        for key in self.keys:
            if results[key].shape[2:] != self.size:
                results[key] = F.interpolate(
                    results[key],
                    size=self.size,
                    mode=self.mode,
                    align_corners=False,
                )
        return results
