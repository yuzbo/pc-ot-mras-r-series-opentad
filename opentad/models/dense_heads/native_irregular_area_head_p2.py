import math
from collections.abc import Mapping

import torch
import torch.nn as nn
from torch.nn import functional as F

from ..bricks import ConvModule, Scale
from ..builder import HEADS, build_prior_generator
from ..utils.sampling_contract import validate_sampling_contract
from ..utils.temporal_grid import (
    build_area_time_grid,
    downsample_temporal_grid,
    prepare_area_targets,
    segment_area_integral,
    temporal_grid_from_metas,
    validate_area_time_grid,
    validate_temporal_grid_alignment,
)


@HEADS.register_module()
class NativeIrregularAreaHeadP2(nn.Module):
    """Native irregular-time area head.

    The head keeps the OpenTAD detector interface but does not decode proposals
    as center +/- regressed distance. It predicts action density on observed
    cells, predicts start/end boundaries inside real-time gap cells, then scores
    boundary pairs by integrating observed action evidence over real time.
    """

    def __init__(
        self,
        num_classes,
        in_channels,
        feat_channels,
        num_convs=2,
        prior_generator=None,
        temporal_grid=None,
        area_head=None,
        cls_prior_prob=0.01,
        loss_normalizer=100,
        loss_normalizer_momentum=0.9,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"NativeIrregularAreaHeadP2 got unsupported argument(s): {unknown}")
        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)
        self.feat_channels = int(feat_channels)
        self.num_convs = int(num_convs)
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        if self.num_convs < 0:
            raise ValueError("num_convs must be >= 0.")

        self.prior_generator = build_prior_generator(prior_generator)
        self.temporal_grid_cfg = temporal_grid or {}
        self.grid_positions_key = self.temporal_grid_cfg.get("positions_key", "irregular_selected_positions")
        self.grid_valid_len_key = self.temporal_grid_cfg.get("valid_len_key", "irregular_selected_valid_len")
        self.grid_strict = bool(self.temporal_grid_cfg.get("strict", True))
        self.require_temporal_grid = bool(self.temporal_grid_cfg.get("required", True))
        if self.temporal_grid_cfg.get("decode_axis", "dense") != "dense":
            raise ValueError("NativeIrregularAreaHeadP2 supports only dense decode_axis.")

        cfg = area_head or {}
        raw_observation_half_width = cfg.get("observation_half_width", "cell_support")
        if raw_observation_half_width is None or str(raw_observation_half_width) == "cell_support":
            self.observation_half_width = "cell_support"
        else:
            self.observation_half_width = float(raw_observation_half_width)
        self.observation_support_scale = float(cfg.get("observation_support_scale", 0.25))
        self.boundary_tau = float(cfg.get("boundary_tau", 1.0))
        self.max_boundaries_per_side = int(cfg.get("max_boundaries_per_side", 32))
        self.max_pairs_per_class = int(cfg.get("max_pairs_per_class", 64))
        self.min_pair_duration = float(cfg.get("min_pair_duration", 1e-4))
        self.area_loss_weight = float(cfg.get("area_loss_weight", 1.0))
        self.start_gap_loss_weight = float(cfg.get("start_gap_loss_weight", 0.5))
        self.end_gap_loss_weight = float(cfg.get("end_gap_loss_weight", 0.5))
        self.start_offset_loss_weight = float(cfg.get("start_offset_loss_weight", 0.25))
        self.end_offset_loss_weight = float(cfg.get("end_offset_loss_weight", 0.25))
        self.uncertainty_loss_weight = float(cfg.get("uncertainty_loss_weight", 0.05))
        self.uncertainty_penalty_alpha = float(cfg.get("uncertainty_penalty_alpha", 1.0))
        self.observed_fraction_power = float(cfg.get("observed_fraction_power", 0.5))
        self.use_regression_range_assignment = bool(cfg.get("use_regression_range_assignment", True))
        self.score_fusion_mode = str(cfg.get("score_fusion_mode", "hand_geometric"))
        self.enable_pair_scorer = bool(cfg.get("enable_pair_scorer", False))
        self.pair_scorer_hidden = int(cfg.get("pair_scorer_hidden", self.feat_channels))
        self.pair_scorer_loss_weight = float(cfg.get("pair_scorer_loss_weight", 0.5))
        self.pair_scorer_iou_positive = float(cfg.get("pair_scorer_iou_positive", 0.5))
        self.pair_scorer_sample_topk = int(cfg.get("pair_scorer_sample_topk", 16))
        quality_cfg = cfg.get("quality_calibration", {})
        if quality_cfg is None:
            quality_cfg = {}
        if not isinstance(quality_cfg, Mapping):
            raise TypeError("area_head.quality_calibration must be a mapping when provided.")
        self.enable_quality_calibration = bool(quality_cfg.get("enable", False))
        self.quality_calibration_hidden = int(quality_cfg.get("hidden_dim", self.pair_scorer_hidden))
        self.quality_calibration_loss_weight = float(quality_cfg.get("quality_loss_weight", 0.5))
        self.quality_boundary_loss_weight = float(quality_cfg.get("boundary_loss_weight", 0.25))
        self.quality_rank_loss_weight = float(quality_cfg.get("rank_loss_weight", 0.0))
        self.quality_boundary_tau = float(quality_cfg.get("boundary_tau", 1.0))
        self.quality_rank_positive_iou = float(quality_cfg.get("rank_positive_iou", 0.7))
        self.quality_rank_negative_iou = float(quality_cfg.get("rank_negative_iou", 0.3))
        self.quality_rank_margin = float(quality_cfg.get("rank_margin", 0.25))
        self.quality_rank_sample_size = int(quality_cfg.get("rank_sample_size", 64))
        self.quality_score_beta = float(quality_cfg.get("score_beta", 1.0))
        self.quality_boundary_gamma = float(quality_cfg.get("boundary_gamma", 1.0))
        self.quality_base_delta = float(quality_cfg.get("base_delta", 1.0))
        self.quality_score_eps = float(quality_cfg.get("score_eps", 1e-6))
        self.enable_center_distance_hybrid = bool(cfg.get("enable_center_distance_hybrid", False))
        self.center_distance_loss_weight = float(cfg.get("center_distance_loss_weight", 1.0))
        self.center_distance_max_proposals_per_level = int(
            cfg.get("center_distance_max_proposals_per_level", self.max_pairs_per_class)
        )
        self.loss_normalizer_momentum = float(loss_normalizer_momentum)
        self.register_buffer("loss_normalizer", torch.tensor(float(loss_normalizer)))

        if self.observation_half_width != "cell_support" and self.observation_half_width <= 0:
            raise ValueError("area_head.observation_half_width must be positive.")
        if self.observation_support_scale <= 0:
            raise ValueError("area_head.observation_support_scale must be positive.")
        if self.boundary_tau <= 0:
            raise ValueError("area_head.boundary_tau must be positive.")
        for key, value in (
            ("max_boundaries_per_side", self.max_boundaries_per_side),
            ("max_pairs_per_class", self.max_pairs_per_class),
        ):
            if value <= 0:
                raise ValueError(f"area_head.{key} must be positive.")
        if self.min_pair_duration <= 0:
            raise ValueError("area_head.min_pair_duration must be positive.")
        if self.score_fusion_mode not in ("hand_geometric", "learned_pair", "hybrid_sum"):
            raise ValueError(
                "area_head.score_fusion_mode must be one of "
                "'hand_geometric', 'learned_pair', or 'hybrid_sum'."
            )
        if self.score_fusion_mode in ("learned_pair", "hybrid_sum") and not self.enable_pair_scorer:
            raise ValueError("learned pair score fusion requires area_head.enable_pair_scorer=True.")
        if self.pair_scorer_hidden <= 0:
            raise ValueError("area_head.pair_scorer_hidden must be positive.")
        if self.pair_scorer_loss_weight < 0:
            raise ValueError("area_head.pair_scorer_loss_weight must be non-negative.")
        if self.pair_scorer_sample_topk <= 0:
            raise ValueError("area_head.pair_scorer_sample_topk must be positive.")
        if self.quality_calibration_hidden <= 0:
            raise ValueError("area_head.quality_calibration.hidden_dim must be positive.")
        for key, value in (
            ("quality_loss_weight", self.quality_calibration_loss_weight),
            ("boundary_loss_weight", self.quality_boundary_loss_weight),
            ("rank_loss_weight", self.quality_rank_loss_weight),
        ):
            if value < 0:
                raise ValueError(f"area_head.quality_calibration.{key} must be non-negative.")
        if self.quality_boundary_tau <= 0:
            raise ValueError("area_head.quality_calibration.boundary_tau must be positive.")
        if self.quality_rank_sample_size <= 0:
            raise ValueError("area_head.quality_calibration.rank_sample_size must be positive.")
        if self.quality_rank_negative_iou > self.quality_rank_positive_iou:
            raise ValueError("quality rank negative IoU threshold must not exceed positive threshold.")
        if self.quality_rank_margin < 0:
            raise ValueError("area_head.quality_calibration.rank_margin must be non-negative.")
        if self.quality_score_eps <= 0:
            raise ValueError("area_head.quality_calibration.score_eps must be positive.")
        for key, value in (
            ("score_beta", self.quality_score_beta),
            ("boundary_gamma", self.quality_boundary_gamma),
            ("base_delta", self.quality_base_delta),
        ):
            if value < 0:
                raise ValueError(f"area_head.quality_calibration.{key} must be non-negative.")
        if self.center_distance_loss_weight < 0:
            raise ValueError("area_head.center_distance_loss_weight must be non-negative.")
        if self.center_distance_max_proposals_per_level <= 0:
            raise ValueError("area_head.center_distance_max_proposals_per_level must be positive.")

        self.shared_convs = nn.ModuleList()
        for idx in range(self.num_convs):
            self.shared_convs.append(
                ConvModule(
                    self.in_channels if idx == 0 else self.feat_channels,
                    self.feat_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=dict(type="LN"),
                    act_cfg=dict(type="relu"),
                )
            )
        head_channels = self.in_channels if self.num_convs == 0 else self.feat_channels
        self.area_head = nn.Conv1d(head_channels, self.num_classes, kernel_size=3, padding=1)
        self.start_gap_head = nn.Conv1d(head_channels, self.num_classes, kernel_size=3, padding=1)
        self.end_gap_head = nn.Conv1d(head_channels, self.num_classes, kernel_size=3, padding=1)
        self.start_offset_head = nn.Conv1d(head_channels, self.num_classes, kernel_size=3, padding=1)
        self.end_offset_head = nn.Conv1d(head_channels, self.num_classes, kernel_size=3, padding=1)
        self.gap_uncertainty_head = nn.Conv1d(head_channels, 2, kernel_size=3, padding=1)
        self.pair_scorer_input_dim = head_channels * 3 + 8
        if self.enable_pair_scorer:
            self.pair_scorer = nn.Sequential(
                nn.Linear(self.pair_scorer_input_dim, self.pair_scorer_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(self.pair_scorer_hidden, self.pair_scorer_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(self.pair_scorer_hidden, 1),
            )
        if self.enable_quality_calibration:
            self.quality_calibrator = nn.Sequential(
                nn.Linear(self.pair_scorer_input_dim, self.quality_calibration_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(self.quality_calibration_hidden, self.quality_calibration_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(self.quality_calibration_hidden, 2),
            )
        if self.enable_center_distance_hybrid:
            self.center_cls_head = nn.Conv1d(head_channels, self.num_classes, kernel_size=3, padding=1)
            self.center_reg_head = nn.Conv1d(head_channels, 2, kernel_size=3, padding=1)
            self.center_scale = nn.ModuleList([Scale() for _ in range(len(self.prior_generator.strides))])

        if cls_prior_prob > 0:
            bias_value = -(math.log((1 - cls_prior_prob) / cls_prior_prob))
            nn.init.constant_(self.area_head.bias, bias_value)
            nn.init.constant_(self.start_gap_head.bias, bias_value)
            nn.init.constant_(self.end_gap_head.bias, bias_value)
            if self.enable_center_distance_hybrid:
                nn.init.constant_(self.center_cls_head.bias, bias_value)
        nn.init.zeros_(self.start_offset_head.weight)
        nn.init.zeros_(self.start_offset_head.bias)
        nn.init.zeros_(self.end_offset_head.weight)
        nn.init.zeros_(self.end_offset_head.bias)
        if self.enable_center_distance_hybrid:
            nn.init.zeros_(self.center_reg_head.weight)
            nn.init.zeros_(self.center_reg_head.bias)

    def forward_train(self, feat_list, mask_list, gt_segments, gt_labels, metas=None, **kwargs):
        validate_sampling_contract(
            metas,
            mask_list[0],
            split="train",
            positions_key=self.grid_positions_key,
            valid_len_key=self.grid_valid_len_key,
        )
        preds, area_grids = self._forward_fields(feat_list, mask_list, metas)
        return self.losses(preds, area_grids, gt_segments, gt_labels)

    def forward_test(self, feat_list, mask_list, metas=None, **kwargs):
        if self.enable_quality_calibration:
            self._reject_quality_eval_gt_kwargs(kwargs)
        validate_sampling_contract(
            metas,
            mask_list[0],
            split="test",
            positions_key=self.grid_positions_key,
            valid_len_key=self.grid_valid_len_key,
        )
        preds, area_grids = self._forward_fields(feat_list, mask_list, metas)
        return self.decode_area_proposals(preds, area_grids)

    def _forward_fields(self, feat_list, mask_list, metas):
        level_grids = self._build_level_grids(metas, mask_list)
        area_grids = [
            build_area_time_grid(
                grid,
                observation_half_width=self.observation_half_width,
                observation_support_scale=self.observation_support_scale,
            )
            for grid in level_grids
        ]
        preds = {
            "area_logits": [],
            "start_gap_logits": [],
            "end_gap_logits": [],
            "start_offset": [],
            "end_offset": [],
            "gap_uncertainty_logits": [],
            "obs_features": [],
            "gap_features": [],
        }
        if self.enable_center_distance_hybrid:
            preds["center_cls_logits"] = []
            preds["center_reg"] = []

        for level_idx, (feat, mask, grid) in enumerate(zip(feat_list, mask_list, area_grids)):
            hidden = feat
            out_mask = mask
            for conv in self.shared_convs:
                hidden, out_mask = conv(hidden, out_mask)
            obs_mask = grid["obs_valid_mask"].to(device=out_mask.device)
            hidden = hidden * obs_mask[:, None].to(dtype=hidden.dtype)
            gap_hidden = self._make_gap_features(hidden, obs_mask)
            anchor_scale = self._anchor_scale_for_area_grid(grid, hidden)
            grid["anchor_scale"] = anchor_scale
            preds["area_logits"].append(self.area_head(hidden))
            preds["start_gap_logits"].append(self.start_gap_head(gap_hidden))
            preds["end_gap_logits"].append(self.end_gap_head(gap_hidden))
            preds["start_offset"].append(torch.tanh(self.start_offset_head(gap_hidden)))
            preds["end_offset"].append(torch.tanh(self.end_offset_head(gap_hidden)))
            preds["gap_uncertainty_logits"].append(self.gap_uncertainty_head(gap_hidden))
            preds["obs_features"].append(hidden)
            preds["gap_features"].append(gap_hidden)
            if self.enable_center_distance_hybrid:
                preds["center_cls_logits"].append(self.center_cls_head(hidden))
                preds["center_reg"].append(F.relu(self.center_scale[level_idx](self.center_reg_head(hidden))))
        return preds, area_grids

    def _anchor_scale_for_area_grid(self, area_grid, template):
        obs_width = area_grid["obs_width"].to(device=template.device, dtype=template.dtype)
        if self.observation_half_width == "cell_support":
            scale = obs_width / max(self.observation_support_scale, 1e-6)
        else:
            scale = obs_width
        return scale.clamp_min(1e-4)

    def _build_level_grids(self, metas, mask_list):
        if len(mask_list) != len(self.prior_generator.strides):
            raise ValueError(
                "NativeIrregularAreaHeadP2 expects one mask per prior stride: "
                f"got {len(mask_list)} masks and {len(self.prior_generator.strides)} strides."
            )
        current = temporal_grid_from_metas(
            metas,
            mask_list[0],
            positions_key=self.grid_positions_key,
            valid_len_key=self.grid_valid_len_key,
            required=self.require_temporal_grid,
            strict=self.grid_strict,
        )
        grids = [current]
        validate_temporal_grid_alignment(current, mask_list[0], context="p2_level0_temporal_grid")
        for level in range(1, len(mask_list)):
            target_len = mask_list[level].shape[1]
            current = downsample_temporal_grid(current)
            while current["center"].shape[1] > target_len:
                current = downsample_temporal_grid(current)
            if current["center"].shape[1] != target_len:
                raise ValueError(
                    f"P2 temporal grid level {level} length {current['center'].shape[1]} "
                    f"does not match feature mask length {target_len}."
                )
            validate_temporal_grid_alignment(current, mask_list[level], context=f"p2_level{level}_temporal_grid")
            grids.append(current)
        return grids

    @staticmethod
    def _make_gap_features(cell_features, cell_mask):
        batch, channels, length = cell_features.shape
        NativeIrregularAreaHeadP2._assert_prefix_mask(cell_mask, context="P2 cell_mask")
        gap = cell_features.new_zeros((batch, channels, length + 1))
        gap_valid = cell_mask.new_zeros((batch, length + 1))
        valid_counts = cell_mask.long().sum(dim=1)
        for batch_idx in range(batch):
            count = int(valid_counts[batch_idx].item())
            if count > 0:
                valid_features = cell_features[batch_idx, :, :count]
                gap[batch_idx, :, 0] = valid_features[:, 0]
                if count > 1:
                    gap[batch_idx, :, 1:count] = 0.5 * (valid_features[:, :-1] + valid_features[:, 1:])
                gap[batch_idx, :, count] = valid_features[:, count - 1]
                gap_valid[batch_idx, : count + 1] = True
        return gap * gap_valid[:, None].to(dtype=gap.dtype)

    @staticmethod
    def _assert_prefix_mask(mask, context):
        if mask.ndim != 2:
            raise ValueError(f"{context} must be [B, T], got shape {tuple(mask.shape)}.")
        mask = mask.bool()
        counts = mask.long().sum(dim=1)
        index = torch.arange(mask.shape[1], device=mask.device)[None, :]
        expected = index < counts[:, None]
        if not torch.equal(mask, expected):
            raise ValueError(f"{context} must be a contiguous valid prefix.")

    @staticmethod
    def _flatten_obs(area_grids, key):
        return torch.cat([grid[key] for grid in area_grids], dim=1)

    @staticmethod
    def _flatten_gap(area_grids, key):
        return torch.cat([grid[key] for grid in area_grids], dim=1)

    @staticmethod
    def _flatten_cell_logits(chunks):
        return torch.cat([chunk.permute(0, 2, 1) for chunk in chunks], dim=1)

    def losses(self, preds, area_grids, gt_segments, gt_labels):
        flat_area_logits = self._flatten_cell_logits(preds["area_logits"])
        flat_start_logits = self._flatten_cell_logits(preds["start_gap_logits"])
        flat_end_logits = self._flatten_cell_logits(preds["end_gap_logits"])
        flat_start_offset = self._flatten_cell_logits(preds["start_offset"])
        flat_end_offset = self._flatten_cell_logits(preds["end_offset"])
        flat_unc_logits = torch.cat([chunk.permute(0, 2, 1) for chunk in preds["gap_uncertainty_logits"]], dim=1)

        obs_valid = self._flatten_obs(area_grids, "obs_valid_mask").to(device=flat_area_logits.device)
        gap_valid = self._flatten_gap(area_grids, "gap_valid_mask").to(device=flat_start_logits.device)
        targets = self._prepare_flat_targets(area_grids, gt_segments, gt_labels)
        area_target = targets["area"].to(device=flat_area_logits.device, dtype=flat_area_logits.dtype)
        start_target = targets["start_gap"].to(device=flat_start_logits.device, dtype=flat_start_logits.dtype)
        end_target = targets["end_gap"].to(device=flat_end_logits.device, dtype=flat_end_logits.dtype)
        start_offset_target = targets["start_offset"].to(device=flat_start_offset.device, dtype=flat_start_offset.dtype)
        end_offset_target = targets["end_offset"].to(device=flat_end_offset.device, dtype=flat_end_offset.dtype)
        start_offset_weight = targets["start_offset_weight"].to(
            device=flat_start_offset.device,
            dtype=flat_start_offset.dtype,
        )
        end_offset_weight = targets["end_offset_weight"].to(
            device=flat_end_offset.device,
            dtype=flat_end_offset.dtype,
        )

        pos_count = int((area_target[obs_valid].sum() + start_target[gap_valid].sum() + end_target[gap_valid].sum()).item())
        if self.training:
            with torch.no_grad():
                self.loss_normalizer.mul_(self.loss_normalizer_momentum).add_(
                    (1.0 - self.loss_normalizer_momentum) * max(pos_count, 1)
                )
            normalizer = self.loss_normalizer.clamp_min(1.0)
        else:
            normalizer = flat_area_logits.new_tensor(float(max(pos_count, 1)))

        area_loss = self._masked_bce(flat_area_logits, area_target, obs_valid, normalizer)
        start_gap_loss = self._masked_bce(flat_start_logits, start_target, gap_valid, normalizer)
        end_gap_loss = self._masked_bce(flat_end_logits, end_target, gap_valid, normalizer)
        start_offset_loss = self._weighted_smooth_l1(
            flat_start_offset,
            start_offset_target,
            start_offset_weight,
            gap_valid,
            normalizer,
        )
        end_offset_loss = self._weighted_smooth_l1(
            flat_end_offset,
            end_offset_target,
            end_offset_weight,
            gap_valid,
            normalizer,
        )
        uncertainty_loss = self._uncertainty_loss(flat_unc_logits, area_grids, gap_valid)
        losses = {
            "area_loss": area_loss * self.area_loss_weight,
            "start_gap_loss": start_gap_loss * self.start_gap_loss_weight,
            "end_gap_loss": end_gap_loss * self.end_gap_loss_weight,
            "start_offset_loss": start_offset_loss * self.start_offset_loss_weight,
            "end_offset_loss": end_offset_loss * self.end_offset_loss_weight,
            "boundary_uncertainty_loss": uncertainty_loss * self.uncertainty_loss_weight,
        }
        if self.enable_pair_scorer and self.pair_scorer_loss_weight > 0:
            losses["pair_quality_loss"] = (
                self._pair_scorer_loss(preds, area_grids, gt_segments, gt_labels, normalizer)
                * self.pair_scorer_loss_weight
            )
        if self.enable_quality_calibration:
            losses.update(self._quality_calibration_losses(preds, area_grids, gt_segments, gt_labels, normalizer))
        if self.enable_center_distance_hybrid and self.center_distance_loss_weight > 0:
            center_losses = self._center_distance_losses(preds, area_grids, gt_segments, gt_labels, normalizer)
            for key, value in center_losses.items():
                losses[key] = value * self.center_distance_loss_weight
        return losses

    @staticmethod
    def _masked_bce(logits, target, valid_mask, normalizer):
        if not valid_mask.any().item():
            return logits.sum() * 0.0
        loss = F.binary_cross_entropy_with_logits(logits[valid_mask], target[valid_mask], reduction="sum")
        return loss / normalizer

    @staticmethod
    def _weighted_smooth_l1(pred, target, weight, valid_mask, normalizer):
        valid = valid_mask[:, :, None] & (weight > 0)
        if not valid.any().item():
            return pred.sum() * 0.0
        loss = F.smooth_l1_loss(pred[valid], target[valid], reduction="none")
        return (loss * weight[valid]).sum() / normalizer

    def _uncertainty_loss(self, uncertainty_logits, area_grids, gap_valid):
        if self.uncertainty_loss_weight <= 0:
            return uncertainty_logits.sum() * 0.0
        dense_len = area_grids[0]["dense_valid_len"].to(device=uncertainty_logits.device, dtype=uncertainty_logits.dtype)
        gap_width = self._flatten_gap(area_grids, "gap_width").to(
            device=uncertainty_logits.device,
            dtype=uncertainty_logits.dtype,
        )
        dense = dense_len[:, None].expand_as(gap_width).clamp_min(1.0)
        target = (gap_width / dense).clamp(0.0, 1.0)
        target = torch.stack((target, target), dim=-1)
        if not gap_valid.any().item():
            return uncertainty_logits.sum() * 0.0
        return F.binary_cross_entropy_with_logits(
            uncertainty_logits[gap_valid],
            target[gap_valid],
            reduction="mean",
        )

    def _prepare_flat_targets(self, area_grids, gt_segments, gt_labels):
        area_targets = []
        start_targets = []
        end_targets = []
        start_offset_targets = []
        end_offset_targets = []
        start_offset_weights = []
        end_offset_weights = []
        for level_idx, grid in enumerate(area_grids):
            targets = prepare_area_targets(
                grid,
                gt_segments,
                gt_labels,
                num_classes=self.num_classes,
                boundary_tau=self.boundary_tau,
                duration_range=self._duration_range_for_level(level_idx),
            )
            area_targets.append(targets["area"])
            start_targets.append(targets["start_gap"])
            end_targets.append(targets["end_gap"])
            start_offset_targets.append(targets["start_offset"])
            end_offset_targets.append(targets["end_offset"])
            start_offset_weights.append(targets["start_offset_weight"])
            end_offset_weights.append(targets["end_offset_weight"])
        return {
            "area": torch.cat(area_targets, dim=1),
            "start_gap": torch.cat(start_targets, dim=1),
            "end_gap": torch.cat(end_targets, dim=1),
            "start_offset": torch.cat(start_offset_targets, dim=1),
            "end_offset": torch.cat(end_offset_targets, dim=1),
            "start_offset_weight": torch.cat(start_offset_weights, dim=1),
            "end_offset_weight": torch.cat(end_offset_weights, dim=1),
        }

    def _pair_scorer_loss(self, preds, area_grids, gt_segments, gt_labels, normalizer):
        losses = []
        for level_idx, grid in enumerate(area_grids):
            for batch_idx in range(preds["area_logits"][level_idx].shape[0]):
                candidates = self._collect_area_level_pairs(preds, grid, level_idx, batch_idx)
                if not candidates or candidates["pair_start"].numel() == 0:
                    continue
                gt_segment = gt_segments[batch_idx].to(
                    device=candidates["pair_start"].device,
                    dtype=candidates["pair_start"].dtype,
                )
                gt_label = gt_labels[batch_idx].to(device=candidates["pair_start"].device).long()
                if gt_segment.numel() == 0:
                    target = candidates["pair_start"].new_zeros(candidates["pair_start"].shape)
                else:
                    target = self._pair_iou_quality_target(candidates, gt_segment, gt_label)
                logits = self.pair_scorer(candidates["pair_features"]).squeeze(-1)
                if logits.numel() > self.max_pairs_per_class * self.num_classes:
                    take = min(logits.numel(), self.max_pairs_per_class * self.num_classes)
                    hand_score = candidates["hand_score"].detach().to(dtype=target.dtype)
                    priority = torch.maximum(target, hand_score)
                    _, order = torch.topk(priority, k=take)
                    logits = logits[order]
                    target = target[order]
                losses.append(F.binary_cross_entropy_with_logits(logits, target, reduction="sum"))
        if not losses:
            return preds["area_logits"][0].sum() * 0.0
        return torch.stack(losses).sum() / normalizer

    @staticmethod
    def _pair_iou_quality_target(candidates, gt_segment, gt_label):
        pair_segment = torch.stack((candidates["pair_start"], candidates["pair_end"]), dim=-1)
        cls_idx = candidates["class_id"].long()
        target = pair_segment.new_zeros((pair_segment.shape[0],))
        for gt_idx in range(gt_segment.shape[0]):
            label = int(gt_label[gt_idx].item())
            same_cls = cls_idx == label
            if not same_cls.any().item():
                continue
            iou = NativeIrregularAreaHeadP2._segment_iou_tensor(pair_segment[same_cls], gt_segment[gt_idx])
            iou = iou.to(dtype=target.dtype)
            target[same_cls] = torch.maximum(target[same_cls], iou)
        return target.clamp(0.0, 1.0)

    def _quality_calibration_losses(self, preds, area_grids, gt_segments, gt_labels, normalizer):
        quality_losses = []
        boundary_losses = []
        rank_losses = []
        for level_idx, grid in enumerate(area_grids):
            for batch_idx in range(preds["area_logits"][level_idx].shape[0]):
                candidates = self._collect_area_level_pairs(preds, grid, level_idx, batch_idx)
                if not candidates or candidates["pair_start"].numel() == 0:
                    continue
                gt_segment = gt_segments[batch_idx].to(
                    device=candidates["pair_start"].device,
                    dtype=candidates["pair_start"].dtype,
                )
                gt_label = gt_labels[batch_idx].to(device=candidates["pair_start"].device).long()
                if gt_segment.numel() == 0:
                    quality_target = candidates["pair_start"].new_zeros(candidates["pair_start"].shape)
                    boundary_target = quality_target
                else:
                    with torch.no_grad():
                        quality_target = self._pair_iou_quality_target(candidates, gt_segment, gt_label)
                        boundary_target = self._pair_boundary_quality_target(
                            candidates,
                            gt_segment,
                            gt_label,
                            tau=self.quality_boundary_tau,
                        )
                quality_logit, boundary_logit = self._quality_calibration_logits(candidates)
                quality_logit, boundary_logit, quality_target, boundary_target = self._sample_quality_training_rows(
                    quality_logit,
                    boundary_logit,
                    quality_target,
                    boundary_target,
                    candidates,
                )
                if self.quality_calibration_loss_weight > 0:
                    quality_losses.append(
                        F.binary_cross_entropy_with_logits(quality_logit, quality_target, reduction="sum")
                    )
                if self.quality_boundary_loss_weight > 0:
                    boundary_losses.append(
                        F.binary_cross_entropy_with_logits(boundary_logit, boundary_target, reduction="sum")
                    )
                if self.quality_rank_loss_weight > 0:
                    rank_losses.append(self._quality_rank_loss(quality_logit, quality_target))
        template = preds["area_logits"][0]
        losses = {}
        quality_loss = torch.stack(quality_losses).sum() / normalizer if quality_losses else template.sum() * 0.0
        boundary_loss = torch.stack(boundary_losses).sum() / normalizer if boundary_losses else template.sum() * 0.0
        rank_loss = torch.stack(rank_losses).mean() if rank_losses else template.sum() * 0.0
        if self.quality_calibration_loss_weight > 0:
            losses["quality_calibration_loss"] = quality_loss * self.quality_calibration_loss_weight
        if self.quality_boundary_loss_weight > 0:
            losses["quality_boundary_loss"] = boundary_loss * self.quality_boundary_loss_weight
        if self.quality_rank_loss_weight > 0:
            losses["quality_rank_loss"] = rank_loss * self.quality_rank_loss_weight
        return losses

    @staticmethod
    def _pair_boundary_quality_target(candidates, gt_segment, gt_label, tau=1.0):
        cls_idx = candidates["class_id"].long()
        target = candidates["pair_start"].new_zeros(candidates["pair_start"].shape)
        for gt_idx in range(gt_segment.shape[0]):
            label = int(gt_label[gt_idx].item())
            same_cls = cls_idx == label
            if not same_cls.any().item():
                continue
            gt_start = gt_segment[gt_idx, 0]
            gt_end = gt_segment[gt_idx, 1]
            gt_duration = (gt_end - gt_start).clamp_min(1e-4)
            start_error = (candidates["pair_start"][same_cls] - gt_start).abs()
            end_error = (candidates["pair_end"][same_cls] - gt_end).abs()
            normalized_error = (start_error + end_error) / gt_duration
            quality = torch.exp(-normalized_error / max(float(tau), 1e-6))
            quality = quality.to(dtype=target.dtype)
            target[same_cls] = torch.maximum(target[same_cls], quality)
        return target.clamp(0.0, 1.0)

    def _quality_calibration_logits(self, candidates):
        logits = self.quality_calibrator(candidates["pair_features"])
        return logits[:, 0], logits[:, 1]

    def _sample_quality_training_rows(self, quality_logit, boundary_logit, quality_target, boundary_target, candidates):
        max_rows = self.max_pairs_per_class * self.num_classes
        if quality_logit.numel() <= max_rows:
            return quality_logit, boundary_logit, quality_target, boundary_target
        hand_score = candidates["hand_score"].detach().to(dtype=quality_target.dtype)
        priority = torch.maximum(quality_target, hand_score)
        _, order = torch.topk(priority, k=max_rows)
        return quality_logit[order], boundary_logit[order], quality_target[order], boundary_target[order]

    def _quality_rank_loss(self, quality_logit, quality_target):
        pos = quality_target >= self.quality_rank_positive_iou
        neg = quality_target <= self.quality_rank_negative_iou
        if not pos.any().item() or not neg.any().item():
            return quality_logit.sum() * 0.0
        pos_logit = quality_logit[pos]
        neg_logit = quality_logit[neg]
        if pos_logit.numel() > self.quality_rank_sample_size:
            _, pos_order = torch.topk(-pos_logit.detach(), k=self.quality_rank_sample_size)
            pos_logit = pos_logit[pos_order]
        if neg_logit.numel() > self.quality_rank_sample_size:
            _, neg_order = torch.topk(neg_logit.detach(), k=self.quality_rank_sample_size)
            neg_logit = neg_logit[neg_order]
        diff = pos_logit[:, None] - neg_logit[None, :]
        return F.relu(self.quality_rank_margin - diff).mean()

    def _center_distance_losses(self, preds, area_grids, gt_segments, gt_labels, normalizer):
        cls_chunks = []
        reg_chunks = []
        center_chunks = []
        scale_chunks = []
        valid_chunks = []
        for level_idx, grid in enumerate(area_grids):
            cls_chunks.append(preds["center_cls_logits"][level_idx].permute(0, 2, 1))
            reg_chunks.append(preds["center_reg"][level_idx].permute(0, 2, 1))
            center_chunks.append(grid["obs_center"].to(device=cls_chunks[-1].device, dtype=cls_chunks[-1].dtype))
            scale_chunks.append(grid["anchor_scale"].to(device=cls_chunks[-1].device, dtype=cls_chunks[-1].dtype))
            valid_chunks.append(grid["obs_valid_mask"].to(device=cls_chunks[-1].device))
        cls_logits = torch.cat(cls_chunks, dim=1)
        reg_pred = torch.cat(reg_chunks, dim=1)
        center = torch.cat(center_chunks, dim=1)
        scale = torch.cat(scale_chunks, dim=1).clamp_min(1e-4)
        valid = torch.cat(valid_chunks, dim=1)
        ranges = self._flat_duration_ranges_for_grids(area_grids, center)
        cls_target, reg_target, pos_mask = self._prepare_center_distance_targets(
            center,
            scale,
            valid,
            ranges,
            gt_segments,
            gt_labels,
        )
        cls_loss = self._masked_bce(cls_logits, cls_target.to(dtype=cls_logits.dtype), valid, normalizer)
        if pos_mask.any().item():
            flat_center = center.reshape(-1)
            flat_scale = scale.reshape(-1)
            flat_reg_pred = reg_pred.reshape(-1, 2)
            flat_reg_target = reg_target.reshape(-1, 2)
            flat_pos = pos_mask.reshape(-1)
            pred_segments = torch.stack(
                (
                    flat_center[flat_pos] - flat_reg_pred[flat_pos, 0] * flat_scale[flat_pos],
                    flat_center[flat_pos] + flat_reg_pred[flat_pos, 1] * flat_scale[flat_pos],
                ),
                dim=-1,
            )
            target_segments = torch.stack(
                (
                    flat_center[flat_pos] - flat_reg_target[flat_pos, 0] * flat_scale[flat_pos],
                    flat_center[flat_pos] + flat_reg_target[flat_pos, 1] * flat_scale[flat_pos],
                ),
                dim=-1,
            )
            reg_loss = F.smooth_l1_loss(pred_segments, target_segments, reduction="sum") / normalizer
        else:
            reg_loss = reg_pred.sum() * 0.0
        return {"center_cls_loss": cls_loss, "center_reg_loss": reg_loss}

    def _prepare_center_distance_targets(self, center, scale, valid, ranges, gt_segments, gt_labels):
        cls_target = center.new_zeros((center.shape[0], center.shape[1], self.num_classes))
        reg_target = center.new_zeros((center.shape[0], center.shape[1], 2))
        pos_mask = valid.new_zeros(valid.shape)
        for batch_idx, (gt_segment, gt_label) in enumerate(zip(gt_segments, gt_labels)):
            if gt_segment.numel() == 0:
                continue
            gt_segment = gt_segment.to(device=center.device, dtype=center.dtype)
            gt_label = gt_label.to(device=center.device).long()
            gt_start = gt_segment[:, 0]
            gt_end = gt_segment[:, 1]
            duration = (gt_end - gt_start).clamp_min(1e-4)
            points = center[batch_idx]
            left = points[:, None] - gt_start[None, :]
            right = gt_end[None, :] - points[:, None]
            inside = (left > 0) & (right > 0)
            max_dist = torch.maximum(left, right)
            range_mask = (max_dist >= ranges[:, 0:1]) & (max_dist <= ranges[:, 1:2])
            candidate = inside & range_mask & valid[batch_idx, :, None]
            cost = duration[None, :].expand_as(candidate).clone()
            cost = cost.masked_fill(~candidate, float("inf"))
            min_cost, assignment = cost.min(dim=1)
            assigned = torch.isfinite(min_cost)
            if not assigned.any().item():
                continue
            indices = assigned.nonzero(as_tuple=True)[0]
            gt_indices = assignment[indices]
            labels = gt_label[gt_indices]
            cls_target[batch_idx, indices, labels] = 1.0
            reg_target[batch_idx, indices, 0] = left[indices, gt_indices] / scale[batch_idx, indices]
            reg_target[batch_idx, indices, 1] = right[indices, gt_indices] / scale[batch_idx, indices]
            pos_mask[batch_idx, indices] = True
        return cls_target, reg_target, pos_mask

    def _flat_duration_ranges_for_grids(self, area_grids, template):
        ranges = []
        for level_idx, grid in enumerate(area_grids):
            duration_range = self._duration_range_for_level(level_idx) or (0.0, 10000.0)
            lower, upper = duration_range
            level_len = int(grid["obs_valid_mask"].shape[1])
            ranges.extend([[float(lower), float(upper)]] * level_len)
        if not ranges:
            return template.new_tensor([[0.0, 10000.0]]).expand(template.shape[1], 2)
        range_tensor = template.new_tensor(ranges)
        if range_tensor.shape[0] != template.shape[1]:
            return template.new_tensor([[0.0, 10000.0]]).expand(template.shape[1], 2)
        return range_tensor

    def _duration_range_for_level(self, level_idx):
        if not self.use_regression_range_assignment:
            return None
        ranges = getattr(self.prior_generator, "regression_range", None)
        if ranges is None:
            return None
        if level_idx >= len(ranges):
            raise ValueError(
                f"Missing regression_range for P2 level {level_idx}; "
                f"got {len(ranges)} range entries."
            )
        lower, upper = ranges[level_idx]
        return float(lower), float(upper)

    def decode_area_proposals(self, preds, area_grids):
        if not area_grids:
            return [], []
        batch_proposals = []
        batch_scores = []
        batch_size = preds["area_logits"][0].shape[0]
        for batch_idx in range(batch_size):
            proposals_per_sample = []
            scores_per_sample = []
            for level_idx, grid in enumerate(area_grids):
                level_proposals, level_scores = self._decode_area_level(
                    preds,
                    grid,
                    level_idx,
                    batch_idx,
                )
                if level_proposals.numel() > 0:
                    proposals_per_sample.append(level_proposals)
                    scores_per_sample.append(level_scores)
            if proposals_per_sample:
                batch_proposals.append(torch.cat(proposals_per_sample, dim=0))
                batch_scores.append(torch.cat(scores_per_sample, dim=0))
            else:
                template = preds["area_logits"][0]
                batch_proposals.append(template.new_zeros((0, 2)))
                batch_scores.append(template.new_zeros((0, self.num_classes)))
            if self.enable_center_distance_hybrid:
                center_proposals, center_scores = self._decode_center_distance_sample(preds, area_grids, batch_idx)
                if center_proposals.numel() > 0:
                    batch_proposals[-1] = torch.cat((batch_proposals[-1], center_proposals), dim=0)
                    batch_scores[-1] = torch.cat((batch_scores[-1], center_scores), dim=0)
        return batch_proposals, batch_scores

    def _decode_area_level(self, preds, area_grid, level_idx, batch_idx):
        candidates = self._collect_area_level_pairs(preds, area_grid, level_idx, batch_idx)
        if not candidates or candidates["pair_start"].numel() == 0:
            template = preds["area_logits"][level_idx]
            return template.new_zeros((0, 2)), template.new_zeros((0, self.num_classes))
        pair_score = self._score_pair_candidates(candidates)
        proposals_per_level = []
        scores_per_level = []
        for cls_idx in range(self.num_classes):
            cls_mask = candidates["class_id"] == cls_idx
            if not cls_mask.any().item():
                continue
            cls_score = pair_score[cls_mask]
            take = min(self.max_pairs_per_class, cls_score.numel())
            if take <= 0:
                continue
            score_top, order = torch.topk(cls_score, k=take)
            pair_start = candidates["pair_start"][cls_mask][order]
            pair_end = candidates["pair_end"][cls_mask][order]
            pair_segments = torch.stack((pair_start, pair_end), dim=-1)
            pair_scores = pair_score.new_zeros((take, self.num_classes))
            pair_scores[:, cls_idx] = score_top
            proposals_per_level.append(pair_segments)
            scores_per_level.append(pair_scores)
        if proposals_per_level:
            return torch.cat(proposals_per_level, dim=0), torch.cat(scores_per_level, dim=0)
        template = preds["area_logits"][level_idx]
        return template.new_zeros((0, 2)), template.new_zeros((0, self.num_classes))

    def _collect_area_level_pairs(self, preds, area_grid, level_idx, batch_idx):
        area_score = preds["area_logits"][level_idx].permute(0, 2, 1).sigmoid()
        start_score = preds["start_gap_logits"][level_idx].permute(0, 2, 1).sigmoid()
        end_score = preds["end_gap_logits"][level_idx].permute(0, 2, 1).sigmoid()
        start_offset = preds["start_offset"][level_idx].permute(0, 2, 1)
        end_offset = preds["end_offset"][level_idx].permute(0, 2, 1)
        uncertainty = F.softplus(preds["gap_uncertainty_logits"][level_idx].permute(0, 2, 1))

        obs_start = area_grid["obs_start"].to(device=area_score.device, dtype=area_score.dtype)
        obs_end = area_grid["obs_end"].to(device=area_score.device, dtype=area_score.dtype)
        obs_valid = area_grid["obs_valid_mask"].to(device=area_score.device)
        gap_center = area_grid["gap_center"].to(device=start_score.device, dtype=start_score.dtype)
        gap_width = area_grid["gap_width"].to(device=start_score.device, dtype=start_score.dtype)
        gap_valid = area_grid["gap_valid_mask"].to(device=start_score.device)

        row_chunks = []
        valid_obs = obs_valid[batch_idx]
        valid_gap = gap_valid[batch_idx]
        if not valid_obs.any().item() or int(valid_gap.sum().item()) < 2:
            return {}

        cell_start = obs_start[batch_idx, valid_obs]
        cell_end = obs_end[batch_idx, valid_obs]
        obs_feat = preds["obs_features"][level_idx].permute(0, 2, 1)
        gap_feat = preds["gap_features"][level_idx].permute(0, 2, 1)
        valid_obs_feat = obs_feat[batch_idx, valid_obs]
        dense_len = area_grid["dense_valid_len"].to(device=area_score.device, dtype=area_score.dtype)[batch_idx]
        for cls_idx in range(self.num_classes):
            k = min(self.max_boundaries_per_side, int(valid_gap.sum().item()))
            start_values = start_score[batch_idx, :, cls_idx].masked_fill(~valid_gap, -1.0)
            end_values = end_score[batch_idx, :, cls_idx].masked_fill(~valid_gap, -1.0)
            start_val, start_idx = torch.topk(start_values, k=k)
            end_val, end_idx = torch.topk(end_values, k=k)

            start_coord = self._gap_boundary_coord(gap_center, gap_width, start_offset, batch_idx, start_idx, cls_idx)
            end_coord = self._gap_boundary_coord(gap_center, gap_width, end_offset, batch_idx, end_idx, cls_idx)
            pair_start = start_coord[:, None].expand(k, k).reshape(-1)
            pair_end = end_coord[None, :].expand(k, k).reshape(-1)
            pair_start_score = start_val[:, None].expand(k, k).reshape(-1)
            pair_end_score = end_val[None, :].expand(k, k).reshape(-1)
            pair_start_idx = start_idx[:, None].expand(k, k).reshape(-1)
            pair_end_idx = end_idx[None, :].expand(k, k).reshape(-1)
            pair_start_rank = torch.arange(k, device=pair_start.device)[:, None].expand(k, k).reshape(-1)
            pair_end_rank = torch.arange(k, device=pair_start.device)[None, :].expand(k, k).reshape(-1)
            pair_start_unc = uncertainty[batch_idx, start_idx, 0][:, None].expand(k, k).reshape(-1)
            pair_end_unc = uncertainty[batch_idx, end_idx, 1][None, :].expand(k, k).reshape(-1)
            pair_duration = pair_end - pair_start
            pair_valid = pair_duration > self.min_pair_duration
            duration_range = self._duration_range_for_level(level_idx)
            if duration_range is not None:
                lower, upper = duration_range
                pair_valid = pair_valid & (pair_duration >= lower) & (pair_duration <= upper)
            if not pair_valid.any().item():
                continue

            pair_start = pair_start[pair_valid]
            pair_end = pair_end[pair_valid]
            pair_start_score = pair_start_score[pair_valid]
            pair_end_score = pair_end_score[pair_valid]
            pair_start_idx = pair_start_idx[pair_valid]
            pair_end_idx = pair_end_idx[pair_valid]
            pair_start_rank = pair_start_rank[pair_valid]
            pair_end_rank = pair_end_rank[pair_valid]
            pair_start_unc = pair_start_unc[pair_valid]
            pair_end_unc = pair_end_unc[pair_valid]
            integral = segment_area_integral(
                pair_start,
                pair_end,
                cell_start,
                cell_end,
                area_score[batch_idx, valid_obs, cls_idx],
            )
            inside = integral["score"].clamp_min(1e-6)
            observed = integral["observed_fraction"].clamp_min(1e-6).pow(self.observed_fraction_power)
            duration = pair_duration[pair_valid].clamp_min(self.min_pair_duration)
            uncertainty_penalty = torch.exp(
                -self.uncertainty_penalty_alpha * (pair_start_unc + pair_end_unc) / duration
            )
            hand_score = (
                pair_start_score.clamp_min(1e-6)
                * pair_end_score.clamp_min(1e-6)
                * inside
            ).pow(1.0 / 3.0)
            hand_score = (hand_score * observed * uncertainty_penalty).clamp(0.0, 1.0)
            start_feat = gap_feat[batch_idx, pair_start_idx]
            end_feat = gap_feat[batch_idx, pair_end_idx]
            pooled_feat = self._interval_pool_features(
                pair_start,
                pair_end,
                cell_start,
                cell_end,
                valid_obs_feat,
            )
            numeric = torch.stack(
                (
                    pair_start_score,
                    pair_end_score,
                    inside,
                    integral["observed_fraction"].clamp(0.0, 1.0),
                    uncertainty_penalty.clamp(0.0, 1.0),
                    duration / dense_len.clamp_min(1.0),
                    gap_width[batch_idx, pair_start_idx] / dense_len.clamp_min(1.0),
                    gap_width[batch_idx, pair_end_idx] / dense_len.clamp_min(1.0),
                ),
                dim=-1,
            )
            pair_features = torch.cat((start_feat, end_feat, pooled_feat, numeric), dim=-1)
            row_chunks.append(
                {
                    "pair_start": pair_start,
                    "pair_end": pair_end,
                    "pair_start_score": pair_start_score,
                    "pair_end_score": pair_end_score,
                    "area_integral": inside,
                    "observed_fraction_raw": integral["observed_fraction"].clamp(0.0, 1.0),
                    "observed_factor": observed,
                    "uncertainty_penalty": uncertainty_penalty.clamp(0.0, 1.0),
                    "pair_duration": duration,
                    "hand_score": hand_score,
                    "class_id": pair_start.new_full(pair_start.shape, cls_idx, dtype=torch.long),
                    "level": pair_start.new_full(pair_start.shape, level_idx, dtype=torch.long),
                    "start_idx": pair_start_idx.long(),
                    "end_idx": pair_end_idx.long(),
                    "start_rank": pair_start_rank.long(),
                    "end_rank": pair_end_rank.long(),
                    "pair_features": pair_features,
                }
            )

        if not row_chunks:
            return {}
        merged = {}
        for key in row_chunks[0]:
            merged[key] = torch.cat([chunk[key] for chunk in row_chunks], dim=0)
        return merged

    def _score_pair_candidates(self, candidates):
        hand_score = candidates["hand_score"]
        if self.score_fusion_mode == "hand_geometric" or not self.enable_pair_scorer:
            base_score = hand_score
        else:
            learned = torch.sigmoid(self.pair_scorer(candidates["pair_features"]).squeeze(-1))
            if self.score_fusion_mode == "learned_pair":
                base = (
                    candidates["pair_start_score"].clamp_min(1e-6)
                    * candidates["pair_end_score"].clamp_min(1e-6)
                    * candidates["area_integral"].clamp_min(1e-6)
                ).pow(1.0 / 3.0)
                base_score = (base * learned).clamp(0.0, 1.0)
            elif self.score_fusion_mode == "hybrid_sum":
                base_score = (0.5 * hand_score + 0.5 * learned).clamp(0.0, 1.0)
            else:
                base_score = hand_score
        if not self.enable_quality_calibration:
            return base_score
        quality_logit, boundary_logit = self._quality_calibration_logits(candidates)
        quality = torch.sigmoid(quality_logit)
        boundary_quality = torch.sigmoid(boundary_logit)
        calibrated = base_score.clamp_min(self.quality_score_eps).pow(self.quality_base_delta)
        if self.quality_score_beta > 0:
            calibrated = calibrated * quality.clamp_min(self.quality_score_eps).pow(self.quality_score_beta)
        if self.quality_boundary_gamma > 0:
            calibrated = calibrated * boundary_quality.clamp_min(self.quality_score_eps).pow(
                self.quality_boundary_gamma
            )
        return calibrated.clamp(0.0, 1.0)

    @staticmethod
    def _reject_quality_eval_gt_kwargs(kwargs):
        forbidden = {
            "gt_segments",
            "gt_labels",
            "gt_bboxes",
            "gt_masks",
            "targets",
            "quality_targets",
            "oracle_targets",
        }
        present = sorted(key for key in forbidden if kwargs.get(key) is not None)
        if present:
            raise ValueError(
                "quality_calibration eval/test forward must not receive GT or oracle target kwargs: "
                + ", ".join(present)
            )

    @staticmethod
    def _interval_pool_features(pair_start, pair_end, cell_start, cell_end, cell_features, eps=1e-4):
        if pair_start.numel() == 0:
            return cell_features.new_zeros((0, cell_features.shape[-1]))
        overlap_start = torch.maximum(cell_start[None, :], pair_start[:, None])
        overlap_end = torch.minimum(cell_end[None, :], pair_end[:, None])
        overlap = (overlap_end - overlap_start).clamp_min(0.0)
        weight_sum = overlap.sum(dim=1, keepdim=True).clamp_min(eps)
        return overlap @ cell_features / weight_sum

    def _decode_center_distance_sample(self, preds, area_grids, batch_idx):
        proposals = []
        scores = []
        for level_idx, grid in enumerate(area_grids):
            cls_score = preds["center_cls_logits"][level_idx].permute(0, 2, 1).sigmoid()
            reg = preds["center_reg"][level_idx].permute(0, 2, 1)
            center = grid["obs_center"].to(device=reg.device, dtype=reg.dtype)
            scale = grid["anchor_scale"].to(device=reg.device, dtype=reg.dtype).clamp_min(1e-4)
            valid = grid["obs_valid_mask"].to(device=reg.device)
            valid_idx = valid[batch_idx]
            if not valid_idx.any().item():
                continue
            start = center[batch_idx] - reg[batch_idx, :, 0] * scale[batch_idx]
            end = center[batch_idx] + reg[batch_idx, :, 1] * scale[batch_idx]
            segment = torch.stack((start, end), dim=-1)
            segment = segment[valid_idx]
            score = cls_score[batch_idx, valid_idx]
            valid_segment = segment[:, 1] > segment[:, 0] + self.min_pair_duration
            if not valid_segment.any().item():
                continue
            segment = segment[valid_segment]
            score = score[valid_segment]
            if segment.shape[0] > self.center_distance_max_proposals_per_level:
                top_scores = score.max(dim=1).values
                _, order = torch.topk(top_scores, k=self.center_distance_max_proposals_per_level)
                segment = segment[order]
                score = score[order]
            proposals.append(segment)
            scores.append(score)
        if proposals:
            return torch.cat(proposals, dim=0), torch.cat(scores, dim=0)
        template = preds["area_logits"][0]
        return template.new_zeros((0, 2)), template.new_zeros((0, self.num_classes))

    @staticmethod
    def _proposal_factor_meta_scalar(meta, key):
        if not isinstance(meta, Mapping):
            return None
        value = meta.get(key)
        if hasattr(value, "detach") and hasattr(value, "cpu"):
            value = value.detach().cpu()
        if hasattr(value, "item"):
            value = value.item()
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    @classmethod
    def _proposal_factor_meta_context(cls, video_id, meta, batch_idx):
        fps = cls._proposal_factor_meta_scalar(meta, "fps")
        snippet_stride = cls._proposal_factor_meta_scalar(meta, "snippet_stride")
        window_start_frame = cls._proposal_factor_meta_scalar(meta, "window_start_frame")
        offset_frames = cls._proposal_factor_meta_scalar(meta, "offset_frames")
        window_size = cls._proposal_factor_meta_scalar(meta, "window_size")
        duration_seconds = cls._proposal_factor_meta_scalar(meta, "duration")
        if offset_frames is None:
            offset_frames = 0.0
        sample_id_parts = [str(video_id), f"batch={int(batch_idx)}"]
        if window_start_frame is not None:
            sample_id_parts.append(f"window_start_frame={window_start_frame:.6f}")
        context = {
            "batch_sample_idx": int(batch_idx),
            "sample_id": "|".join(sample_id_parts),
        }
        optional_scalars = {
            "fps": fps,
            "snippet_stride": snippet_stride,
            "window_start_frame": window_start_frame,
            "offset_frames": offset_frames,
            "window_size": window_size,
            "duration_seconds": duration_seconds,
        }
        for key, value in optional_scalars.items():
            if value is not None:
                context[key] = float(value)
        if fps is not None and fps > 0.0 and window_start_frame is not None:
            context["window_start_seconds"] = float((window_start_frame + offset_frames) / fps)
            if snippet_stride is not None and window_size is not None:
                context["window_end_seconds"] = float(
                    (window_start_frame + offset_frames + window_size * snippet_stride) / fps
                )
        return context

    @staticmethod
    def _proposal_factor_segment_context(segment, meta_context):
        out = {}
        snippet_stride = meta_context.get("snippet_stride")
        window_start_frame = meta_context.get("window_start_frame")
        offset_frames = meta_context.get("offset_frames", 0.0)
        fps = meta_context.get("fps")
        if snippet_stride is None or window_start_frame is None:
            return out
        start_frame = window_start_frame + offset_frames + segment[0] * snippet_stride
        end_frame = window_start_frame + offset_frames + segment[1] * snippet_stride
        out["segment_frames"] = [float(start_frame), float(end_frame)]
        if fps is not None and fps > 0.0:
            out["segment_seconds"] = [float(start_frame / fps), float(end_frame / fps)]
        return out

    def dump_proposal_factors(self, feat_list, mask_list, metas=None, label_names=None):
        validate_sampling_contract(
            metas,
            mask_list[0],
            split="test",
            positions_key=self.grid_positions_key,
            valid_len_key=self.grid_valid_len_key,
        )
        preds, area_grids = self._forward_fields(feat_list, mask_list, metas)
        rows = []
        batch_size = preds["area_logits"][0].shape[0]
        for batch_idx in range(batch_size):
            meta = metas[batch_idx] if metas else {}
            video_id = str(meta.get("video_name", batch_idx)) if isinstance(meta, Mapping) else str(batch_idx)
            meta_context = self._proposal_factor_meta_context(video_id, meta, batch_idx)
            for level_idx, grid in enumerate(area_grids):
                candidates = self._collect_area_level_pairs(preds, grid, level_idx, batch_idx)
                if not candidates or candidates["pair_start"].numel() == 0:
                    continue
                final_score = self._score_pair_candidates(candidates)
                for row_idx in range(candidates["pair_start"].numel()):
                    cls_idx = int(candidates["class_id"][row_idx].item())
                    label = str(label_names[cls_idx]) if label_names is not None and cls_idx < len(label_names) else str(cls_idx)
                    segment = [
                        float(candidates["pair_start"][row_idx].detach().cpu().item()),
                        float(candidates["pair_end"][row_idx].detach().cpu().item()),
                    ]
                    rows.append(
                        {
                            "video_id": video_id,
                            **meta_context,
                            "class_id": cls_idx,
                            "label": label,
                            "segment": segment,
                            **self._proposal_factor_segment_context(segment, meta_context),
                            "start_score": float(candidates["pair_start_score"][row_idx].detach().cpu().item()),
                            "end_score": float(candidates["pair_end_score"][row_idx].detach().cpu().item()),
                            "area_integral": float(candidates["area_integral"][row_idx].detach().cpu().item()),
                            "observed_fraction": float(candidates["observed_fraction_raw"][row_idx].detach().cpu().item()),
                            "uncertainty_penalty": float(candidates["uncertainty_penalty"][row_idx].detach().cpu().item()),
                            "final_score": float(final_score[row_idx].detach().cpu().item()),
                            "duration": float(candidates["pair_duration"][row_idx].detach().cpu().item()),
                            "level": int(candidates["level"][row_idx].detach().cpu().item()),
                            "start_rank": int(candidates["start_rank"][row_idx].detach().cpu().item()),
                            "end_rank": int(candidates["end_rank"][row_idx].detach().cpu().item()),
                            "start_coord": float(candidates["pair_start"][row_idx].detach().cpu().item()),
                            "end_coord": float(candidates["pair_end"][row_idx].detach().cpu().item()),
                        }
                    )
        return rows

    @staticmethod
    def _gap_boundary_coord(gap_center, gap_width, offset, batch_idx, indices, cls_idx):
        center = gap_center[batch_idx, indices]
        width = gap_width[batch_idx, indices]
        delta = offset[batch_idx, indices, cls_idx] * 0.5 * width
        return center + delta

    @staticmethod
    def _segment_iou_tensor(segments, gt_segment):
        inter_start = torch.maximum(segments[:, 0], gt_segment[0])
        inter_end = torch.minimum(segments[:, 1], gt_segment[1])
        inter = (inter_end - inter_start).clamp_min(0.0)
        union = (segments[:, 1] - segments[:, 0]).clamp_min(0.0) + (gt_segment[1] - gt_segment[0]).clamp_min(0.0) - inter
        return torch.where(union > 0, inter / union.clamp_min(1e-6), torch.zeros_like(inter))

    def validate_area_grids(self, area_grids):
        for grid in area_grids:
            validate_area_time_grid(grid)
        return True
