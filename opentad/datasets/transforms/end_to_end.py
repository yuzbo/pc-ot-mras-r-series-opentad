import copy
import os
import pickle
import random
import torch
import random
import pandas as pd
import numpy as np

from ..builder import PIPELINES
from torch.nn import functional as F
from .pseudo_boundary import (
    load_coarse_score_cache,
    select_coarse_oracle_shell_positions,
    slice_global_scores_for_window,
)


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
        method_base=None,
        keep_ratio=0.5,
        target_len=None,
        source_len=None,
        trunc_len=None,
        trunc_thresh=None,
        crop_ratio=None,
        oracle_boundary_radius=2,
        remap_gt_to_selected_axis=True,
        store_dense_window=False,
        selection_unit=1,
        coarse_score_cache_dir=None,
        coarse_score_allow_missing=False,
        coarse_score_debug_fallback=False,
        coarse_score_min_action_score=0.35,
        coarse_score_transition_top_fraction=0.15,
        coarse_score_transition_top_count=None,
        coarse_score_transition_radius=2,
        coarse_score_uncertainty_weight=0.4,
        coarse_score_change_weight=0.6,
    ):
        self.num_clips = num_clips
        self.scale_factor = scale_factor  # multiply by the frame number, if backbone has downsampling
        self.method = method  # resize or padding or random_trunc or sliding_window
        self.method_base = method_base
        self.keep_ratio = keep_ratio
        self.target_len = target_len
        self.source_len = source_len
        # random_trunc settings
        self.trunc_len = trunc_len
        self.trunc_thresh = trunc_thresh
        self.crop_ratio = crop_ratio
        self.oracle_boundary_radius = int(oracle_boundary_radius)
        self.remap_gt_to_selected_axis = bool(remap_gt_to_selected_axis)
        self.store_dense_window = bool(store_dense_window)
        self.selection_unit = int(selection_unit)
        self.coarse_score_cache_dir = coarse_score_cache_dir
        self.coarse_score_allow_missing = bool(coarse_score_allow_missing)
        self.coarse_score_debug_fallback = bool(coarse_score_debug_fallback)
        self.coarse_score_min_action_score = float(coarse_score_min_action_score)
        self.coarse_score_transition_top_fraction = float(coarse_score_transition_top_fraction)
        self.coarse_score_transition_top_count = coarse_score_transition_top_count
        self.coarse_score_transition_radius = int(coarse_score_transition_radius)
        self.coarse_score_uncertainty_weight = float(coarse_score_uncertainty_weight)
        self.coarse_score_change_weight = float(coarse_score_change_weight)

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

    def _selection_group_size(self):
        return max(int(self.selection_unit), 1) * max(int(self.scale_factor), 1)

    def _num_selection_units(self, valid_len):
        return int(np.ceil(float(valid_len) / float(self._selection_group_size())))

    def _selection_target_count(self, target_frame_num):
        return int(np.ceil(float(target_frame_num) / float(self._selection_group_size())))

    def _positions_to_selection_units(self, positions, valid_len):
        positions = np.asarray(positions, dtype=np.int64)
        positions = positions[(positions >= 0) & (positions < valid_len)]
        if self._selection_group_size() == 1:
            return np.unique(positions)
        return np.unique(positions // self._selection_group_size())

    def _expand_selected_units(self, units, valid_len):
        units = np.asarray(units, dtype=np.int64)
        if self._selection_group_size() == 1:
            return np.unique(units[(units >= 0) & (units < valid_len)]).astype(np.int64)
        positions = []
        group_size = self._selection_group_size()
        for unit in units:
            start = int(unit) * group_size
            end = min(start + group_size, valid_len)
            positions.extend(range(start, end))
        return np.asarray(sorted(set(x for x in positions if 0 <= x < valid_len)), dtype=np.int64)

    def _uniform_pick_indices(self, candidates, count):
        candidates = np.asarray(candidates, dtype=np.int64)
        candidates = np.unique(candidates[candidates >= 0])
        count = min(max(int(count), 0), candidates.size)
        if count <= 0:
            return np.zeros((0,), dtype=np.int64)
        if count >= candidates.size:
            return candidates
        pick = np.linspace(0, candidates.size - 1, num=count)
        return candidates[np.rint(pick).astype(np.int64)]

    def _build_frame_oracle_groups(self, valid_len, gt_segments, profile):
        boundary_positions = set()
        action_positions = set()
        if gt_segments is not None:
            for seg in np.asarray(gt_segments, dtype=np.float32):
                if len(seg) < 2:
                    continue
                start = int(np.floor(np.clip(seg[0], 0, valid_len)))
                end = int(np.ceil(np.clip(seg[1], 0, valid_len)))
                if end <= start:
                    continue
                left = max(0, start)
                right = min(valid_len - 1, max(start, end - 1))
                for boundary in (left, right):
                    lo = max(0, boundary - self.oracle_boundary_radius)
                    hi = min(valid_len, boundary + self.oracle_boundary_radius + 1)
                    boundary_positions.update(range(lo, hi))
                if profile == "action_boundary_dense":
                    action_positions.update(range(left, right + 1))

        boundary = np.array(sorted(boundary_positions), dtype=np.int64)
        action = np.array(sorted(action_positions.difference(boundary_positions)), dtype=np.int64)
        all_positions = np.arange(valid_len, dtype=np.int64)
        used = boundary if profile == "boundary_dense" else np.union1d(boundary, action)
        background = np.setdiff1d(all_positions, used, assume_unique=False)
        return boundary, action, background

    def _build_oracle_groups(self, valid_len, gt_segments, profile):
        boundary, action, background = self._build_frame_oracle_groups(valid_len, gt_segments, profile)
        if self._selection_group_size() == 1:
            return boundary, action, background
        boundary_units = self._positions_to_selection_units(boundary, valid_len)
        action_units = self._positions_to_selection_units(action, valid_len)
        action_units = np.setdiff1d(action_units, boundary_units, assume_unique=False)
        all_units = np.arange(self._num_selection_units(valid_len), dtype=np.int64)
        used_units = boundary_units if profile == "boundary_dense" else np.union1d(boundary_units, action_units)
        background_units = np.setdiff1d(all_units, used_units, assume_unique=False)
        return boundary_units, action_units, background_units

    def _select_oracle_positions(self, valid_len, gt_segments, target_frame_num, profile):
        boundary, action, background = self._build_oracle_groups(valid_len, gt_segments, profile)
        groups = [boundary]
        if profile == "action_boundary_dense":
            groups.append(action)
        groups.append(background)

        selected = []
        target_count = min(self._selection_target_count(target_frame_num), self._num_selection_units(valid_len))
        remaining = int(target_count)
        for group in groups:
            if remaining <= 0:
                break
            chosen = self._uniform_pick_indices(group, min(remaining, group.size))
            if chosen.size > 0:
                selected.append(chosen)
                remaining -= chosen.size
        if selected:
            selected = np.unique(np.concatenate(selected))
        else:
            selected = np.zeros((0,), dtype=np.int64)
        return self._expand_selected_units(np.sort(selected), valid_len)[: int(target_frame_num)]

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
        if not remapped_segments:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int32)
        return np.asarray(remapped_segments, dtype=np.float32), np.asarray(remapped_labels, dtype=np.int32)

    def _set_irregular_axis_meta(self, results, kept_positions, valid_len):
        scale = float(max(self.scale_factor, 1))
        results["irregular_selected_positions"] = np.asarray(kept_positions, dtype=np.float32) / scale
        results["irregular_selected_valid_len"] = float(valid_len) / scale
        results["irregular_native_axis"] = bool(not self.remap_gt_to_selected_axis)

    def _subsample_window_from_positions(self, dense_frame_idxs, gt_segments, gt_labels, target_frame_num, keep_positions):
        valid_len = int(len(dense_frame_idxs))
        if valid_len <= 0:
            raise RuntimeError(f"{self.method} received an empty dense window")
        keep_positions = np.asarray(keep_positions, dtype=np.int64)
        keep_positions = np.unique(keep_positions[(keep_positions >= 0) & (keep_positions < valid_len)])
        if keep_positions.size == 0:
            keep_positions = np.array([0], dtype=np.int64)
        kept_frame_idxs = dense_frame_idxs[keep_positions]
        if self.remap_gt_to_selected_axis and gt_segments is not None and gt_labels is not None:
            out_segments, out_labels = self._remap_gt_to_selected_axis(gt_segments, gt_labels, keep_positions, valid_len)
        else:
            out_segments, out_labels = gt_segments, gt_labels

        frame_num = int(target_frame_num)
        if kept_frame_idxs.shape[0] < frame_num:
            kept_frame_idxs = np.pad(kept_frame_idxs, (0, frame_num - kept_frame_idxs.shape[0]), mode="edge")
        valid_mask_len = min(
            int(np.ceil(keep_positions.size / max(self.scale_factor, 1))),
            int(np.ceil(frame_num / max(self.scale_factor, 1))),
        )
        target_mask_len = int(np.ceil(frame_num / max(self.scale_factor, 1)))
        if valid_mask_len < target_mask_len:
            masks = torch.cat([torch.ones(valid_mask_len), torch.zeros(target_mask_len - valid_mask_len)]).bool()
        else:
            masks = torch.ones(target_mask_len).bool()
        return kept_frame_idxs, out_segments, out_labels, masks, keep_positions.astype(np.float32), valid_len

    def _oracle_subsample_window(self, dense_frame_idxs, gt_segments, gt_labels, target_frame_num, profile):
        keep_positions = self._select_oracle_positions(len(dense_frame_idxs), gt_segments, target_frame_num, profile)
        return self._subsample_window_from_positions(dense_frame_idxs, gt_segments, gt_labels, target_frame_num, keep_positions)

    def _select_coarse_oracle_shell_positions(self, dense_window, frame_stride, frame_num, results):
        video_name = results.get("video_name", "unknown")
        if not self.coarse_score_cache_dir:
            raise ValueError("coarse_score_oracle_shell_subsample requires coarse_score_cache_dir")
        try:
            raw_scores, manifest = load_coarse_score_cache(self.coarse_score_cache_dir, video_name)
        except FileNotFoundError:
            if not (self.coarse_score_allow_missing and self.coarse_score_debug_fallback):
                raise
            manifest = {"score_source": "debug_uniform_fallback", "uses_gt": False}
            raw_scores = {"action_score": np.ones((len(dense_window),), dtype=np.float32) * 0.5}
            results["coarse_oracle_shell_debug_fallback"] = True
        global_indices = np.rint(dense_window / max(frame_stride, 1)).astype(np.int64)
        window_scores = slice_global_scores_for_window(raw_scores, global_indices)
        keep_positions = select_coarse_oracle_shell_positions(
            valid_len=len(dense_window),
            target_frame_num=frame_num,
            action_score=window_scores.get("action_score"),
            sample_key=(
                f"{video_name}|coarse_score_oracle_shell|{int(dense_window[0]) if len(dense_window) else -1}|"
                f"{int(dense_window[-1]) if len(dense_window) else -1}|{len(dense_window)}|{frame_num}"
            ),
            min_action_score=self.coarse_score_min_action_score,
            transition_top_fraction=self.coarse_score_transition_top_fraction,
            transition_top_count=self.coarse_score_transition_top_count,
            transition_radius=self.coarse_score_transition_radius,
            uncertainty_weight=self.coarse_score_uncertainty_weight,
            change_weight=self.coarse_score_change_weight,
            group_size=self._selection_group_size(),
        )
        results["coarse_oracle_shell_score_source"] = manifest.get("score_source", "unknown")
        results["coarse_oracle_shell_uses_gt_for_selection"] = False
        results["coarse_oracle_shell_score_axis"] = manifest.get("axis", "global_snippet_index")
        results["coarse_oracle_shell_selected_positions"] = keep_positions.astype(np.int64)
        return keep_positions

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

        elif self.method in (
            "oracle_boundary_subsample",
            "oracle_action_boundary_subsample",
            "coarse_score_oracle_shell_subsample",
            "coarse_actionness_oracle_shell_subsample",
        ):
            if self.method in ("oracle_boundary_subsample", "oracle_action_boundary_subsample"):
                assert "gt_segments" in results.keys(), f"{self.method} requires gt_segments in the pipeline"
                assert "gt_labels" in results.keys(), f"{self.method} requires gt_labels in the pipeline"
            assert results["snippet_stride"] >= self.scale_factor
            assert results["snippet_stride"] % self.scale_factor == 0

            profile = "boundary_dense" if self.method == "oracle_boundary_subsample" else "action_boundary_dense"
            frame_stride = results["snippet_stride"] // self.scale_factor
            dense_frame_idxs = np.arange(0, total_frames, frame_stride)
            gt_segments = results["gt_segments"] * self.scale_factor if "gt_segments" in results else None
            gt_labels = results["gt_labels"] if "gt_labels" in results else None

            if self.target_len is None:
                raise ValueError(f"{self.method} requires target_len to be set")
            frame_num = int(self.target_len * self.scale_factor)

            if self.method_base == "random_trunc":
                if gt_segments is None or gt_labels is None:
                    raise ValueError(f"{self.method} with random_trunc requires gt_segments and gt_labels")
                source_len = self.source_len
                if source_len is None:
                    source_len = int(round(self.target_len / max(self.keep_ratio, 1e-6)))
                dense_window, gt_segments, gt_labels = self.random_trunc(
                    dense_frame_idxs,
                    trunc_len=int(source_len * self.scale_factor),
                    gt_segments=gt_segments,
                    gt_labels=gt_labels,
                )
            elif self.method_base == "sliding_window":
                start_idx = min(results["feature_start_idx"] * self.scale_factor, len(dense_frame_idxs))
                end_idx = min((results["feature_end_idx"] + 1) * self.scale_factor, len(dense_frame_idxs))
                dense_window = dense_frame_idxs[start_idx:end_idx]
            else:
                raise ValueError(
                    f"{self.method} requires method_base='random_trunc' or 'sliding_window', got {self.method_base}"
                )

            dense_frame_num = int((self.source_len or len(dense_window)) * self.scale_factor)
            if self.store_dense_window:
                oracle_frame_inds = dense_window
                if oracle_frame_inds.shape[0] < dense_frame_num:
                    oracle_frame_inds = np.pad(
                        oracle_frame_inds,
                        (0, dense_frame_num - oracle_frame_inds.shape[0]),
                        mode="edge",
                    )
                else:
                    oracle_frame_inds = oracle_frame_inds[:dense_frame_num]
                results["oracle_dense_frame_inds"] = oracle_frame_inds.astype(int)

            if self.method in ("oracle_boundary_subsample", "oracle_action_boundary_subsample"):
                frame_idxs, gt_segments, gt_labels, masks, keep_positions, valid_len = self._oracle_subsample_window(
                    dense_frame_idxs=dense_window,
                    gt_segments=gt_segments,
                    gt_labels=gt_labels,
                    target_frame_num=frame_num,
                    profile=profile,
                )
            else:
                keep_positions = self._select_coarse_oracle_shell_positions(dense_window, frame_stride, frame_num, results)
                frame_idxs, gt_segments, gt_labels, masks, keep_positions, valid_len = self._subsample_window_from_positions(
                    dense_frame_idxs=dense_window,
                    gt_segments=gt_segments,
                    gt_labels=gt_labels,
                    target_frame_num=frame_num,
                    keep_positions=keep_positions,
                )

            if gt_segments is not None and gt_labels is not None:
                results["gt_segments"] = gt_segments / self.scale_factor
                results["gt_labels"] = gt_labels
            self._set_irregular_axis_meta(results, keep_positions, valid_len)

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
