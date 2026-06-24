import math
from collections.abc import Mapping

import torch
import torch.nn as nn
from torch.nn import functional as F

from ..builder import HEADS, build_prior_generator, build_loss
from ..bricks import ConvModule, Scale
from ..utils.temporal_grid import downsample_temporal_grid, temporal_grid_from_metas, validate_temporal_grid_alignment


@HEADS.register_module()
class AnchorFreeHead(nn.Module):
    def __init__(
        self,
        num_classes,
        in_channels,
        feat_channels,
        num_convs=3,
        prior_generator=None,
        loss=None,
        loss_normalizer=100,
        loss_normalizer_momentum=0.9,
        center_sample="radius",
        center_sample_radius=1.5,
        label_smoothing=0,
        cls_prior_prob=0.01,
        loss_weight=1.0,
        filter_similar_gt=True,
        temporal_grid=None,
        temporal_grid_mode="selected_index",
        physical_grid_actionformer=False,
    ):
        super(AnchorFreeHead, self).__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.num_convs = num_convs
        self.cls_prior_prob = cls_prior_prob
        self.label_smoothing = label_smoothing
        self.filter_similar_gt = filter_similar_gt
        self._init_temporal_grid_config(
            temporal_grid=temporal_grid,
            temporal_grid_mode=temporal_grid_mode,
            physical_grid_actionformer=physical_grid_actionformer,
        )

        self.loss_weight = loss_weight
        self.center_sample = center_sample
        self.center_sample_radius = center_sample_radius
        self.loss_normalizer_momentum = loss_normalizer_momentum
        self.register_buffer("loss_normalizer", torch.tensor(loss_normalizer))  # save in the state_dict

        # point generator
        self.prior_generator = build_prior_generator(prior_generator)

        self._init_layers()

        self.cls_loss = build_loss(loss.cls_loss)
        self.reg_loss = build_loss(loss.reg_loss)

    def _init_layers(self):
        """Initialize layers of the head."""
        self._init_cls_convs()
        self._init_reg_convs()
        self._init_heads()

    def _init_cls_convs(self):
        """Initialize classification conv layers of the head."""
        self.cls_convs = nn.ModuleList([])
        for i in range(self.num_convs):
            self.cls_convs.append(
                ConvModule(
                    self.in_channels if i == 0 else self.feat_channels,
                    self.feat_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=dict(type="LN"),
                    act_cfg=dict(type="relu"),
                )
            )

    def _init_reg_convs(self):
        """Initialize bbox regression conv layers of the head."""
        self.reg_convs = nn.ModuleList([])
        for i in range(self.num_convs):
            self.reg_convs.append(
                ConvModule(
                    self.in_channels if i == 0 else self.feat_channels,
                    self.feat_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=dict(type="LN"),
                    act_cfg=dict(type="relu"),
                )
            )

    def _init_heads(self):
        """Initialize predictor layers of the head."""
        self.cls_head = nn.Conv1d(self.feat_channels, self.num_classes, kernel_size=3, padding=1)
        self.reg_head = nn.Conv1d(self.feat_channels, 2, kernel_size=3, padding=1)
        self.scale = nn.ModuleList([Scale() for _ in range(len(self.prior_generator.strides))])

        # use prior in model initialization to improve stability
        # this will overwrite other weight init
        if self.cls_prior_prob > 0:
            bias_value = -(math.log((1 - self.cls_prior_prob) / self.cls_prior_prob))
            nn.init.constant_(self.cls_head.bias, bias_value)

    def forward_train(self, feat_list, mask_list, gt_segments, gt_labels, metas=None, **kwargs):
        cls_pred = []
        reg_pred = []

        for l, (feat, mask) in enumerate(zip(feat_list, mask_list)):
            cls_feat = feat
            reg_feat = feat

            for i in range(self.num_convs):
                cls_feat, mask = self.cls_convs[i](cls_feat, mask)
                reg_feat, mask = self.reg_convs[i](reg_feat, mask)

            cls_pred.append(self.cls_head(cls_feat))
            reg_pred.append(F.relu(self.scale[l](self.reg_head(reg_feat))))

        points = self._points_for_feature_geometry(feat_list, mask_list, metas=metas, mark_native_axis=True)

        losses = self.losses(cls_pred, reg_pred, mask_list, points, gt_segments, gt_labels)
        return losses

    def forward_test(self, feat_list, mask_list, metas=None, **kwargs):
        cls_pred = []
        reg_pred = []

        for l, (feat, mask) in enumerate(zip(feat_list, mask_list)):
            cls_feat = feat
            reg_feat = feat

            for i in range(self.num_convs):
                cls_feat, mask = self.cls_convs[i](cls_feat, mask)
                reg_feat, mask = self.reg_convs[i](reg_feat, mask)

            cls_pred.append(self.cls_head(cls_feat))
            reg_pred.append(F.relu(self.scale[l](self.reg_head(reg_feat))))

        points = self._points_for_feature_geometry(feat_list, mask_list, metas=metas, mark_native_axis=True)

        # get refined proposals and scores
        proposals, scores = self.get_valid_proposals_scores(points, reg_pred, cls_pred, mask_list)  # list [T,2]
        return proposals, scores

    def get_refined_proposals(self, points, reg_pred):
        reg_pred = torch.cat(reg_pred, dim=-1).permute(0, 2, 1)  # [B,T,2]

        if self._points_are_batched(points):
            points = torch.cat(points, dim=1)  # [B,T,4]
            if points.shape[0] != reg_pred.shape[0] or points.shape[1] != reg_pred.shape[1]:
                raise ValueError("batched physical points must align with regression predictions.")
            start = points[:, :, 0] - reg_pred[:, :, 0] * points[:, :, 3]
            end = points[:, :, 0] + reg_pred[:, :, 1] * points[:, :, 3]
        else:
            points = torch.cat(points, dim=0)  # [T,4]
            start = points[:, 0][None] - reg_pred[:, :, 0] * points[:, 3][None]
            end = points[:, 0][None] + reg_pred[:, :, 1] * points[:, 3][None]
        proposals = torch.stack((start, end), dim=-1)  # [B,T,2]
        return proposals

    def get_valid_proposals_scores(self, points, reg_pred, cls_pred, mask_list):
        # apply regression to get refined proposals
        proposals = self.get_refined_proposals(points, reg_pred)  # [B,T,2]
        # proposal scores
        scores = torch.cat(cls_pred, dim=-1).permute(0, 2, 1).sigmoid()  # [B,T,num_classes]

        # mask out invalid, and return a list with batch size
        masks = torch.cat(mask_list, dim=1)  # [B,T]
        new_proposals, new_scores = [], []
        for proposal, score, mask in zip(proposals, scores, masks):
            new_proposals.append(proposal[mask])  # [T,2]
            new_scores.append(score[mask])  # [T,num_classes]
        return new_proposals, new_scores

    def losses(self, cls_pred, reg_pred, mask_list, points, gt_segments, gt_labels):
        gt_cls, gt_reg = self.prepare_targets(points, gt_segments, gt_labels)

        # positive mask
        gt_cls = torch.stack(gt_cls)
        valid_mask = torch.cat(mask_list, dim=1)
        pos_mask = torch.logical_and((gt_cls.sum(-1) > 0), valid_mask)
        num_pos = pos_mask.sum().item()

        # maintain an EMA of foreground to stabilize the loss normalizer
        # useful for small mini-batch training
        if self.training:
            self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
                1 - self.loss_normalizer_momentum
            ) * max(num_pos, 1)
            loss_normalizer = self.loss_normalizer
        else:
            loss_normalizer = max(num_pos, 1)

        # 1. classification loss
        cls_pred = [x.permute(0, 2, 1) for x in cls_pred]
        cls_pred = torch.cat(cls_pred, dim=1)[valid_mask]
        gt_target = gt_cls[valid_mask]

        # optional label smoothing
        gt_target *= 1 - self.label_smoothing
        gt_target += self.label_smoothing / (self.num_classes + 1)

        cls_loss = self.cls_loss(cls_pred, gt_target, reduction="sum")
        cls_loss /= loss_normalizer

        # 2. regression using IoU/GIoU/DIOU loss (defined on positive samples)
        split_size = [reg.shape[-1] for reg in reg_pred]
        gt_reg = torch.stack(gt_reg).permute(0, 2, 1).split(split_size, dim=-1)  # [B,2,T]
        pred_segments = self.get_refined_proposals(points, reg_pred)[pos_mask]
        gt_segments = self.get_refined_proposals(points, gt_reg)[pos_mask]
        if num_pos == 0:
            reg_loss = pred_segments.sum() * 0
        else:
            # giou loss defined on positive samples
            reg_loss = self.reg_loss(pred_segments, gt_segments, reduction="sum")
            reg_loss /= loss_normalizer

        if self.loss_weight > 0:
            loss_weight = self.loss_weight
        else:
            loss_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)

        return {"cls_loss": cls_loss, "reg_loss": reg_loss * loss_weight}

    @torch.no_grad()
    def prepare_targets(self, points, gt_segments, gt_labels):
        batched_points = self._points_are_batched(points)
        if batched_points:
            num_pts = sum(level_points.shape[1] for level_points in points)
        else:
            shared_points = torch.cat(points, dim=0)
            num_pts = shared_points.shape[0]
        gt_cls, gt_reg = [], []

        for batch_idx, (gt_segment, gt_label) in enumerate(zip(gt_segments, gt_labels)):
            if batched_points:
                concat_points = torch.cat([level_points[batch_idx] for level_points in points], dim=0)
            else:
                concat_points = shared_points
            num_gts = gt_segment.shape[0]

            # corner case where current sample does not have actions
            if num_gts == 0:
                gt_cls.append(gt_segment.new_full((num_pts, self.num_classes), 0))
                gt_reg.append(gt_segment.new_zeros((num_pts, 2)))
                continue

            # compute the lengths of all segments -> F T x N
            lens = gt_segment[:, 1] - gt_segment[:, 0]
            lens = lens[None, :].repeat(num_pts, 1)

            # compute the distance of every point to each segment boundary
            # auto broadcasting for all reg target-> F T x N x2
            gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)
            left = concat_points[:, 0, None] - gt_segs[:, :, 0]
            right = gt_segs[:, :, 1] - concat_points[:, 0, None]
            reg_targets = torch.stack((left, right), dim=-1)

            if self.center_sample == "radius":
                # center of all segments F T x N
                center_pts = 0.5 * (gt_segs[:, :, 0] + gt_segs[:, :, 1])
                # center sampling based on stride radius
                # compute the new boundaries:
                # concat_points[:, 3] stores the stride
                t_mins = center_pts - concat_points[:, 3, None] * self.center_sample_radius
                t_maxs = center_pts + concat_points[:, 3, None] * self.center_sample_radius
                # prevent t_mins / maxs from over-running the action boundary
                # left: torch.maximum(t_mins, gt_segs[:, :, 0])
                # right: torch.minimum(t_maxs, gt_segs[:, :, 1])
                # F T x N (distance to the new boundary)
                cb_dist_left = concat_points[:, 0, None] - torch.maximum(t_mins, gt_segs[:, :, 0])
                cb_dist_right = torch.minimum(t_maxs, gt_segs[:, :, 1]) - concat_points[:, 0, None]
                # F T x N x 2
                center_seg = torch.stack((cb_dist_left, cb_dist_right), -1)
                # F T x N
                inside_gt_seg_mask = center_seg.min(-1)[0] > 0
            else:
                # inside an gt action
                inside_gt_seg_mask = reg_targets.min(-1)[0] > 0

            # limit the regression range for each location
            max_regress_distance = reg_targets.max(-1)[0]
            # F T x N
            inside_regress_range = torch.logical_and(
                (max_regress_distance >= concat_points[:, 1, None]), (max_regress_distance <= concat_points[:, 2, None])
            )

            # if there are still more than one actions for one moment
            # pick the one with the shortest duration (easiest to regress)
            lens.masked_fill_(inside_gt_seg_mask == 0, float("inf"))
            lens.masked_fill_(inside_regress_range == 0, float("inf"))
            # F T x N -> F T
            min_len, min_len_inds = lens.min(dim=1)

            # corner case: multiple actions with very similar durations (e.g., THUMOS14)
            if self.filter_similar_gt:
                min_len_mask = torch.logical_and((lens <= (min_len[:, None] + 1e-3)), (lens < float("inf")))
            else:
                min_len_mask = lens < float("inf")
            min_len_mask = min_len_mask.to(reg_targets.dtype)

            # cls_targets: F T x C; reg_targets F T x 2
            gt_label_one_hot = F.one_hot(gt_label.long(), self.num_classes).to(reg_targets.dtype)
            cls_targets = min_len_mask @ gt_label_one_hot
            # to prevent multiple GT actions with the same label and boundaries
            cls_targets.clamp_(min=0.0, max=1.0)
            # OK to use min_len_inds
            reg_targets = reg_targets[range(num_pts), min_len_inds]
            # normalization based on stride
            reg_targets /= concat_points[:, 3, None]

            gt_cls.append(cls_targets)
            gt_reg.append(reg_targets)
        return gt_cls, gt_reg

    def _init_temporal_grid_config(self, temporal_grid, temporal_grid_mode, physical_grid_actionformer):
        cfg = dict(temporal_grid or {})
        cfg_mode = cfg.get("temporal_grid_mode", cfg.get("mode", temporal_grid_mode))
        self.temporal_grid_mode = str(cfg_mode or "selected_index")
        mode_key = self.temporal_grid_mode.lower()
        physical_modes = {"physical", "dense", "physical_dense", "dense_physical"}
        explicitly_enabled = bool(physical_grid_actionformer or cfg.get("enabled", False))
        self.physical_grid_actionformer = explicitly_enabled or mode_key in physical_modes
        if not self.physical_grid_actionformer:
            return
        if mode_key not in physical_modes and mode_key != "selected_index":
            raise ValueError(f"unsupported temporal_grid_mode for ActionFormerHead: {self.temporal_grid_mode}")
        decode_axis = cfg.get("decode_axis", "dense")
        if str(decode_axis) != "dense":
            raise ValueError("physical-grid ActionFormerHead requires temporal_grid.decode_axis='dense'.")
        self.temporal_grid_mode = "physical"
        self.temporal_grid_positions_key = cfg.get("positions_key", "irregular_selected_positions")
        self.temporal_grid_valid_len_key = cfg.get("valid_len_key", "irregular_selected_valid_len")
        self.temporal_grid_required = bool(cfg.get("required", True))
        self.temporal_grid_strict = bool(cfg.get("strict", True))

    def _points_for_feature_geometry(self, feat_list, mask_list, metas=None, mark_native_axis=False):
        if not self.physical_grid_actionformer:
            return self.prior_generator(feat_list)

        if not isinstance(feat_list, (tuple, list)) or not isinstance(mask_list, (tuple, list)):
            raise ValueError("physical-grid ActionFormerHead expects feature and mask lists.")
        if len(feat_list) != len(mask_list):
            raise ValueError("physical-grid ActionFormerHead feature/mask level count mismatch.")
        if len(feat_list) == 0:
            raise ValueError("physical-grid ActionFormerHead requires at least one feature level.")

        current_grid = temporal_grid_from_metas(
            metas,
            mask_list[0],
            positions_key=self.temporal_grid_positions_key,
            valid_len_key=self.temporal_grid_valid_len_key,
            required=self.temporal_grid_required,
            strict=self.temporal_grid_strict,
        )
        self._ensure_dense_gt_axis_contract(metas)

        points = []
        for level, (feat, mask) in enumerate(zip(feat_list, mask_list)):
            if level > 0:
                current_grid = downsample_temporal_grid(current_grid)
            validate_temporal_grid_alignment(
                current_grid,
                mask,
                context=f"actionformer_physical_grid_level{level}",
            )
            points.append(self._physical_points_for_level(current_grid, feat, level))

        if mark_native_axis:
            self._mark_physical_grid_native_axis(metas)
        return points

    def _physical_points_for_level(self, temporal_grid, feat, level):
        center = temporal_grid["center"].to(device=feat.device, dtype=torch.float32)
        batch, time = center.shape
        if time != feat.shape[-1]:
            raise ValueError(
                f"physical-grid level {level} length mismatch: grid={time}, feature={feat.shape[-1]}."
            )
        reg_range = torch.as_tensor(
            self.prior_generator.regression_range[level],
            device=feat.device,
            dtype=torch.float32,
        )[None, None, :].expand(batch, time, 2)
        level_scale = temporal_grid["level_scale"].to(device=feat.device, dtype=torch.float32)
        stride = level_scale[:, None, None].expand(batch, time, 1).clamp_min(1.0e-4)
        return torch.cat((center[:, :, None], reg_range, stride), dim=2)

    def _mark_physical_grid_native_axis(self, metas):
        if metas is None:
            return
        if not isinstance(metas, (list, tuple)):
            raise ValueError("physical-grid ActionFormerHead expects metas as a list/tuple.")
        for idx, meta in enumerate(metas):
            if not isinstance(meta, Mapping):
                raise ValueError(f"physical-grid ActionFormerHead metas[{idx}] must be a mapping.")
            meta["irregular_native_axis"] = True
            meta["physical_grid_actionformer"] = True
            meta["temporal_grid_mode"] = "physical"

    def _ensure_dense_gt_axis_contract(self, metas):
        if metas is None:
            return
        if not isinstance(metas, (list, tuple)):
            raise ValueError("physical-grid ActionFormerHead expects metas as a list/tuple.")
        for idx, meta in enumerate(metas):
            if not isinstance(meta, Mapping):
                raise ValueError(f"physical-grid ActionFormerHead metas[{idx}] must be a mapping.")
            if bool(meta.get("pc_ot_mras_prebackbone_remap_gt_to_selected_axis", False)):
                raise ValueError(
                    "physical-grid ActionFormerHead requires dense-axis GT. "
                    "Set frame_selector.remap_gt_to_selected_axis=False for physical-grid configs."
                )

    @staticmethod
    def _points_are_batched(points):
        return len(points) > 0 and points[0].ndim == 3
