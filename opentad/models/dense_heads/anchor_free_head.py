import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from ..builder import HEADS, build_prior_generator, build_loss
from ..bricks import ConvModule, Scale


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
        assignment_debug=None,
        physical_grid_actionformer=None,
    ):
        super(AnchorFreeHead, self).__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.num_convs = num_convs
        self.cls_prior_prob = cls_prior_prob
        self.label_smoothing = label_smoothing
        self.filter_similar_gt = filter_similar_gt
        self.assignment_debug = assignment_debug or {}
        self.assignment_debug_enabled = bool(self.assignment_debug.get("enabled", False))
        self.physical_grid_cfg = {} if physical_grid_actionformer is None else dict(physical_grid_actionformer)
        self.physical_grid_enabled = bool(self.physical_grid_cfg.get("enabled", False))
        self.physical_grid_required = bool(self.physical_grid_cfg.get("required", self.physical_grid_enabled))
        self.physical_grid_strict = bool(self.physical_grid_cfg.get("strict", True))
        self.physical_grid_eps = float(self.physical_grid_cfg.get("eps", 1.0e-6))
        self._physical_grid_debug = {}

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

    def _physical_grid_forbidden_gt_remap(self, meta):
        forbidden_keys = (
            "remap_gt_to_selected_axis",
            "pc_ot_mras_prebackbone_remap_gt_to_selected_axis",
            "gt_remapped_to_selected_axis",
        )
        return any(bool(meta.get(key, False)) for key in forbidden_keys)

    def _validate_physical_grid_train_gt_axis(self, meta):
        if meta.get("irregular_native_axis", None) is not True:
            raise ValueError(
                "physical-grid ActionFormer requires dense-axis GT; "
                "irregular_native_axis must be explicitly True for training."
            )
        if self._physical_grid_forbidden_gt_remap(meta):
            raise ValueError("physical-grid ActionFormer requires dense-axis GT; selected-axis GT remap is forbidden.")

    def _physical_selected_count_from_meta(self, meta, positions):
        count_sources = []
        for key in ("selected_valid_len", "irregular_selected_count"):
            if key in meta and meta[key] is not None:
                count_sources.append((key, int(round(float(meta[key])))))
        if not count_sources:
            return int(positions.numel())

        selected_count = count_sources[0][1]
        for key, value in count_sources[1:]:
            if value != selected_count:
                raise ValueError(
                    "physical-grid ActionFormer selected-count metadata mismatch: "
                    f"{count_sources[0][0]}={selected_count}, {key}={value}."
                )
        if selected_count < 0:
            raise ValueError("physical-grid ActionFormer selected count must be non-negative.")
        if selected_count > int(positions.numel()):
            raise ValueError(
                "physical-grid ActionFormer selected count exceeds physical positions length: "
                f"selected_count={selected_count}, positions={int(positions.numel())}."
            )
        return selected_count

    def _physical_positions_from_meta(self, meta, device, dtype):
        positions = meta.get("irregular_selected_positions", None)
        if positions is None:
            positions = meta.get("selected_dense_indices", None)
        if positions is None:
            if self.physical_grid_required:
                raise ValueError("physical-grid ActionFormer requires irregular_selected_positions or selected_dense_indices.")
            return None, None

        positions = torch.as_tensor(positions, device=device, dtype=dtype).reshape(-1)
        selected_count = self._physical_selected_count_from_meta(meta, positions)
        positions = positions[:selected_count]
        if positions.numel() == 0:
            if self.physical_grid_required:
                raise ValueError("physical-grid ActionFormer requires at least one selected physical position.")
            return None, None

        dense_valid_len = meta.get("irregular_dense_valid_len", meta.get("irregular_selected_valid_len", None))
        if dense_valid_len is None:
            dense_valid_len = float(positions[-1].item()) + 1.0
        dense_valid_len = max(float(dense_valid_len), float(positions[-1].item()) + 1.0)
        return positions, dense_valid_len

    def _selected_axis_to_physical_axis(self, coords, positions, dense_valid_len):
        xp = torch.arange(positions.numel(), dtype=coords.dtype, device=coords.device)
        xp = torch.cat([xp, xp.new_tensor([float(positions.numel())])], dim=0)
        fp = torch.cat([positions, positions.new_tensor([float(dense_valid_len)])], dim=0)
        flat = coords.reshape(-1).clamp(min=0.0, max=float(positions.numel()))
        right_idx = torch.searchsorted(xp, flat, right=True).clamp(min=1, max=xp.numel() - 1)
        left_idx = right_idx - 1
        x0 = xp[left_idx]
        x1 = xp[right_idx]
        y0 = fp[left_idx]
        y1 = fp[right_idx]
        weight = (flat - x0) / (x1 - x0).clamp(min=self.physical_grid_eps)
        return (y0 + weight * (y1 - y0)).reshape(coords.shape)

    def _build_physical_points_and_masks(self, points, mask_list, metas=None, train_mode=False):
        if not self.physical_grid_enabled:
            return points, mask_list
        if metas is None:
            if self.physical_grid_required:
                raise ValueError("physical-grid ActionFormer requires metas.")
            return points, mask_list

        batch_size = mask_list[0].shape[0]
        if len(metas) != batch_size:
            raise ValueError(
                f"physical-grid ActionFormer metas batch mismatch: metas={len(metas)}, batch={batch_size}."
            )

        physical_points = [[] for _ in points]
        physical_masks = [mask.clone().bool() for mask in mask_list]
        debug_centers = []
        debug_axis_delta = []
        valid_points_total = 0
        debug_selected_count = 0
        debug_dense_valid_len = 0.0

        for batch_idx, meta in enumerate(metas):
            if train_mode:
                self._validate_physical_grid_train_gt_axis(meta)

            base_device = points[0].device
            base_dtype = points[0].dtype
            positions, dense_valid_len = self._physical_positions_from_meta(meta, base_device, base_dtype)
            if positions is None:
                return points, mask_list

            selected_count = int(positions.numel())
            debug_selected_count += selected_count
            debug_dense_valid_len = max(debug_dense_valid_len, float(dense_valid_len))
            meta["irregular_native_axis"] = True
            meta["physical_grid_actionformer"] = True
            meta["physical_grid_selected_count"] = selected_count
            meta["physical_grid_dense_valid_len"] = float(dense_valid_len)

            for level_idx, base_point in enumerate(points):
                point = base_point.clone()
                selected_center = point[:, 0].to(dtype=base_dtype, device=base_device)
                slot_ordinal = torch.arange(point.shape[0], dtype=base_dtype, device=base_device)
                nominal_stride = point[:, 3].to(dtype=base_dtype, device=base_device).clamp(min=self.physical_grid_eps)
                physical_center = self._selected_axis_to_physical_axis(selected_center, positions, dense_valid_len)
                physical_prev = self._selected_axis_to_physical_axis(
                    (selected_center - nominal_stride).clamp(min=0.0), positions, dense_valid_len
                )
                physical_next = self._selected_axis_to_physical_axis(
                    selected_center + nominal_stride, positions, dense_valid_len
                )
                physical_stride = ((physical_next - physical_prev) * 0.5).clamp(min=self.physical_grid_eps)
                range_scale = physical_stride / nominal_stride
                point[:, 0] = physical_center
                point[:, 1] = point[:, 1] * range_scale
                point[:, 2] = point[:, 2] * range_scale
                point[:, 3] = physical_stride
                physical_points[level_idx].append(point)

                slot_index = torch.arange(point.shape[0], device=base_device)
                level_valid = slot_index < int(selected_count)
                physical_masks[level_idx][batch_idx] = physical_masks[level_idx][batch_idx] & level_valid
                kept = physical_masks[level_idx][batch_idx]
                if kept.any():
                    kept_centers = physical_center[kept]
                    debug_centers.append(kept_centers.detach())
                    debug_axis_delta.append((kept_centers - slot_ordinal[kept]).abs().detach())
                    valid_points_total += int(kept.sum().item())

        physical_points = [torch.stack(level_points, dim=0) for level_points in physical_points]
        if debug_centers:
            centers = torch.cat(debug_centers)
            axis_delta = torch.cat(debug_axis_delta)
            self._physical_grid_debug = {
                "physical_grid_actionformer_enabled": True,
                "physical_grid_actionformer_valid_points": int(valid_points_total),
                "physical_grid_actionformer_selected_count": int(debug_selected_count),
                "physical_grid_actionformer_dense_valid_len_max": float(debug_dense_valid_len),
                "physical_grid_actionformer_center_min": float(centers.min().item()),
                "physical_grid_actionformer_center_max": float(centers.max().item()),
                "physical_grid_actionformer_axis_delta_reference": "selected_slot_ordinal",
                "physical_grid_actionformer_axis_delta_mean": float(axis_delta.mean().item()),
                "physical_grid_actionformer_axis_delta_max": float(axis_delta.max().item()),
            }
        else:
            self._physical_grid_debug = {
                "physical_grid_actionformer_enabled": True,
                "physical_grid_actionformer_valid_points": 0,
            }
        return physical_points, physical_masks

    def collect_debug_state(self):
        return dict(self._physical_grid_debug)

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

        points = self.prior_generator(feat_list)
        points, mask_list = self._build_physical_points_and_masks(
            points, mask_list, metas=metas, train_mode=True
        )

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

        points = self.prior_generator(feat_list)
        points, mask_list = self._build_physical_points_and_masks(
            points, mask_list, metas=metas, train_mode=False
        )

        # get refined proposals and scores
        proposals, scores = self.get_valid_proposals_scores(points, reg_pred, cls_pred, mask_list)  # list [T,2]
        return proposals, scores

    def get_refined_proposals(self, points, reg_pred):
        points = torch.cat(points, dim=1) if points[0].dim() == 3 else torch.cat(points, dim=0)  # [B,T,4] or [T,4]
        reg_pred = torch.cat(reg_pred, dim=-1).permute(0, 2, 1)  # [B,T,2]

        if points.dim() == 3:
            center = points[:, :, 0]
            stride = points[:, :, 3]
        else:
            center = points[:, 0][None]
            stride = points[:, 3][None]
        start = center - reg_pred[:, :, 0] * stride
        end = center + reg_pred[:, :, 1] * stride
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
        concat_points = torch.cat(points, dim=1) if points[0].dim() == 3 else torch.cat(points, dim=0)
        batched_points = concat_points.dim() == 3
        gt_cls, gt_reg = [], []

        for batch_idx, (gt_segment, gt_label) in enumerate(zip(gt_segments, gt_labels)):
            point = concat_points[batch_idx] if batched_points else concat_points
            num_pts = point.shape[0]
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
            left = point[:, 0, None] - gt_segs[:, :, 0]
            right = gt_segs[:, :, 1] - point[:, 0, None]
            reg_targets = torch.stack((left, right), dim=-1)

            if self.center_sample == "radius":
                # center of all segments F T x N
                center_pts = 0.5 * (gt_segs[:, :, 0] + gt_segs[:, :, 1])
                # center sampling based on stride radius
                # compute the new boundaries:
                # point[:, 3] stores the stride
                t_mins = center_pts - point[:, 3, None] * self.center_sample_radius
                t_maxs = center_pts + point[:, 3, None] * self.center_sample_radius
                # prevent t_mins / maxs from over-running the action boundary
                # left: torch.maximum(t_mins, gt_segs[:, :, 0])
                # right: torch.minimum(t_maxs, gt_segs[:, :, 1])
                # F T x N (distance to the new boundary)
                cb_dist_left = point[:, 0, None] - torch.maximum(t_mins, gt_segs[:, :, 0])
                cb_dist_right = torch.minimum(t_maxs, gt_segs[:, :, 1]) - point[:, 0, None]
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
                (max_regress_distance >= point[:, 1, None]), (max_regress_distance <= point[:, 2, None])
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
            reg_targets /= point[:, 3, None]

            gt_cls.append(cls_targets)
            gt_reg.append(reg_targets)
        return gt_cls, gt_reg
