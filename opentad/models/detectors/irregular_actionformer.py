import json
import os

import torch

from ..builder import DETECTORS, build_backbone, build_projection, build_head, build_neck
from .base import BaseDetector
from ..utils import build_temporal_grid, normalize_temporal_grid_input
from ..utils.post_processing import batched_nms, convert_to_seconds
from ..bricks import Scale, AffineDropPath
import torch.nn as nn


@DETECTORS.register_module()
class IrregularActionFormer(BaseDetector):
    def __init__(self, projection, rpn_head, neck=None, backbone=None, max_seq_len=384):
        super().__init__()
        if backbone is not None:
            self.backbone = build_backbone(backbone)
        if projection is not None:
            self.projection = build_projection(projection)
            max_seq_len = getattr(self.projection, "max_seq_len", max_seq_len)
        if neck is not None:
            self.neck = build_neck(neck)
        if rpn_head is not None:
            self.rpn_head = build_head(rpn_head)

        self.max_seq_len = max_seq_len

    @property
    def with_backbone(self):
        return hasattr(self, "backbone") and self.backbone is not None

    @property
    def with_projection(self):
        return hasattr(self, "projection") and self.projection is not None

    @property
    def with_neck(self):
        return hasattr(self, "neck") and self.neck is not None

    @property
    def with_rpn_head(self):
        return hasattr(self, "rpn_head") and self.rpn_head is not None

    def _pad_temporal_grid(self, temporal_grid, feat_len, max_len, masks):
        if temporal_grid is None:
            return None

        if torch.is_tensor(temporal_grid):
            center = temporal_grid
            fresh_mask = masks[:, :feat_len]
            cell_left = None
            cell_right = None
        else:
            center = temporal_grid["center"]
            fresh_mask = temporal_grid.get("fresh_mask", masks[:, :feat_len])
            cell_left = temporal_grid.get("cell_left", None)
            cell_right = temporal_grid.get("cell_right", None)

        if center.shape[1] == max_len:
            grid = {"center": center, "fresh_mask": fresh_mask}
            if cell_left is not None and cell_right is not None:
                grid["cell_left"] = cell_left
                grid["cell_right"] = cell_right
            return normalize_temporal_grid_input(grid, masks)

        if center.shape[1] == 1:
            gap = center.new_ones(center.shape[0], 1)
        else:
            gap = (center[:, -1:] - center[:, -2:-1]).clamp_min(1e-4)

        extra = max_len - center.shape[1]
        if extra > 0:
            steps = torch.arange(1, extra + 1, device=center.device, dtype=center.dtype)[None]
            pad_center = center[:, -1:] + gap * steps
            center = torch.cat([center, pad_center], dim=1)
            pad_fresh = fresh_mask.new_zeros(fresh_mask.shape[0], extra)
            fresh_mask = torch.cat([fresh_mask, pad_fresh], dim=1)
            if cell_left is not None and cell_right is not None:
                pad_left = cell_left[:, -1:].expand(-1, extra)
                pad_right = cell_right[:, -1:].expand(-1, extra)
                cell_left = torch.cat([cell_left, pad_left], dim=1)
                cell_right = torch.cat([cell_right, pad_right], dim=1)

        grid = {"center": center[:, :max_len], "fresh_mask": fresh_mask[:, :max_len]}
        if cell_left is not None and cell_right is not None:
            grid["cell_left"] = cell_left[:, :max_len]
            grid["cell_right"] = cell_right[:, :max_len]
        return normalize_temporal_grid_input(grid, masks)

    def _build_center_grid_from_positions(self, pos, native_end, mask):
        target_len = mask.shape[0]
        valid_points = int(pos.numel())
        if valid_points == 0:
            center = torch.zeros(target_len, device=mask.device, dtype=torch.float32)
            fresh_mask = torch.zeros(target_len, device=mask.device, dtype=torch.bool)
            return build_temporal_grid(center[None], valid_mask=mask[None], fresh_mask=fresh_mask[None])

        valid_points = min(valid_points, target_len)
        pos = pos[:valid_points].to(device=mask.device, dtype=torch.float32)
        native_end = max(float(native_end), float(pos[-1].item()) + 1.0)

        center = pos
        if valid_points < target_len:
            pad = center[-1:].repeat(target_len - valid_points)
            center = torch.cat([center, pad], dim=0)

        fresh_mask = torch.zeros(target_len, device=mask.device, dtype=torch.bool)
        fresh_mask[:valid_points] = True

        if valid_points == 1:
            left = center.new_ones(target_len)
            right = center.new_ones(target_len)
            right[0] = max(native_end - float(pos[0].item()), 1e-4)
        else:
            delta = (pos[1:] - pos[:-1]).clamp_min(1e-4)
            left = center.new_ones(target_len)
            right = center.new_ones(target_len)
            left[:valid_points] = torch.cat([delta[:1], delta], dim=0)
            right[:valid_points] = torch.cat([delta, pos.new_tensor([native_end - float(pos[-1].item())])], dim=0)

        left = left.clamp_min(1e-4)
        right = right.clamp_min(1e-4)
        if valid_points < target_len:
            left[valid_points:] = left[valid_points - 1]
            right[valid_points:] = right[valid_points - 1]

        return build_temporal_grid(
            center[None],
            valid_mask=mask[None],
            fresh_mask=fresh_mask[None],
            cell_left=left[None],
            cell_right=right[None],
        )

    def _is_bvr_twb_meta(self, meta):
        if meta is None:
            return False
        if "bvr_twb_ledger" in meta:
            return True
        return any(str(key).startswith("bvr_twb_") for key in meta.keys())

    def _is_rba_rbr_meta(self, meta):
        if meta is None:
            return False
        if "rba_rbr_ledger" in meta:
            return True
        return any(str(key).startswith("rba_rbr_") for key in meta.keys())

    def _cfg_get(self, cfg, key, default=None):
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    def _resolve_bvr_twb_postprocess_guard(self, post_cfg, meta, pre_nms_thresh, pre_nms_topk):
        guard_cfg = self._cfg_get(post_cfg, "bvr_twb_postprocess_guard", None)
        if guard_cfg is None or not bool(self._cfg_get(guard_cfg, "enabled", False)):
            return None

        require_bvr_meta = bool(self._cfg_get(guard_cfg, "require_bvr_meta", True))
        if require_bvr_meta and not self._is_bvr_twb_meta(meta):
            raise ValueError("BVR-TWB postprocess guard is enabled but the sample has no BVR-TWB metadata")

        raw_proposal_cap = int(self._cfg_get(guard_cfg, "raw_proposal_cap", 1024))
        per_class_topk = int(self._cfg_get(guard_cfg, "per_class_topk", 32))
        total_candidate_cap = int(self._cfg_get(guard_cfg, "total_candidate_cap", pre_nms_topk))
        min_score = float(self._cfg_get(guard_cfg, "min_score", pre_nms_thresh))

        invalid = []
        if raw_proposal_cap <= 0:
            invalid.append("raw_proposal_cap")
        if per_class_topk <= 0:
            invalid.append("per_class_topk")
        if total_candidate_cap <= 0:
            invalid.append("total_candidate_cap")
        if min_score < 0:
            invalid.append("min_score")
        if invalid:
            raise ValueError(f"Invalid BVR-TWB postprocess guard fields: {invalid}")

        return {
            "raw_proposal_cap": raw_proposal_cap,
            "per_class_topk": per_class_topk,
            "total_candidate_cap": min(total_candidate_cap, int(pre_nms_topk)),
            "score_thresh": max(float(pre_nms_thresh), min_score),
            "require_bvr_meta": require_bvr_meta,
        }

    def _resolve_rba_rbr_postprocess_guard(self, post_cfg, meta, pre_nms_thresh, pre_nms_topk):
        guard_cfg = self._cfg_get(post_cfg, "rba_rbr_postprocess_guard", None)
        if guard_cfg is None or not bool(self._cfg_get(guard_cfg, "enabled", False)):
            return None

        require_rba_meta = bool(self._cfg_get(guard_cfg, "require_rba_meta", True))
        if require_rba_meta and not self._is_rba_rbr_meta(meta):
            raise ValueError("RBA-RBR postprocess guard is enabled but the sample has no RBA-RBR metadata")

        raw_proposal_cap = int(self._cfg_get(guard_cfg, "raw_proposal_cap", 1024))
        per_class_topk = int(self._cfg_get(guard_cfg, "per_class_topk", 32))
        total_candidate_cap = int(self._cfg_get(guard_cfg, "total_candidate_cap", pre_nms_topk))
        min_score = float(self._cfg_get(guard_cfg, "min_score", pre_nms_thresh))

        invalid = []
        if raw_proposal_cap <= 0:
            invalid.append("raw_proposal_cap")
        if per_class_topk <= 0:
            invalid.append("per_class_topk")
        if total_candidate_cap <= 0:
            invalid.append("total_candidate_cap")
        if min_score < 0:
            invalid.append("min_score")
        if invalid:
            raise ValueError(f"Invalid RBA-RBR postprocess guard fields: {invalid}")

        return {
            "raw_proposal_cap": raw_proposal_cap,
            "per_class_topk": per_class_topk,
            "total_candidate_cap": min(total_candidate_cap, int(pre_nms_topk)),
            "score_thresh": max(float(pre_nms_thresh), min_score),
            "require_rba_meta": require_rba_meta,
        }

    def _select_route_guarded_candidates(
        self,
        segments,
        scores,
        num_classes,
        guard,
        candidate_generation_mode,
    ):
        raw_input_count = int(segments.shape[0])
        audit = {
            "guard_active": True,
            "candidate_generation_mode": candidate_generation_mode,
            "guard_raw_proposal_cap": int(guard["raw_proposal_cap"]),
            "guard_per_class_topk": int(guard["per_class_topk"]),
            "guard_total_candidate_cap": int(guard["total_candidate_cap"]),
            "guard_score_thresh": float(guard["score_thresh"]),
            "guard_raw_input_count": raw_input_count,
        }

        if raw_input_count == 0:
            empty_scores = scores.new_zeros((0,))
            empty_labels = torch.zeros(0, dtype=torch.long, device=scores.device)
            audit.update(
                {
                    "guard_raw_selected_count": 0,
                    "guard_above_threshold_before_per_class_cap": 0,
                    "guard_class_candidate_count_before_global_topk": 0,
                }
            )
            return segments, empty_scores, empty_labels, audit

        raw_cap = min(int(guard["raw_proposal_cap"]), raw_input_count)
        if raw_cap < raw_input_count:
            raw_rank = scores.max(dim=1).values
            raw_keep = raw_rank.topk(raw_cap, largest=True, sorted=False).indices
            segments = segments[raw_keep]
            scores = scores[raw_keep]
        audit["guard_raw_selected_count"] = int(segments.shape[0])

        pred_chunks = []
        point_chunks = []
        class_chunks = []
        above_threshold_count = 0
        for class_idx in range(num_classes):
            class_scores = scores[:, class_idx]
            keep = class_scores > float(guard["score_thresh"])
            keep_idxs = keep.nonzero(as_tuple=True)[0]
            above_threshold_count += int(keep_idxs.numel())
            if keep_idxs.numel() == 0:
                continue
            kept_scores = class_scores[keep_idxs]
            class_topk = min(int(guard["per_class_topk"]), int(kept_scores.numel()))
            kept_scores, order = kept_scores.sort(descending=True)
            selected_points = keep_idxs[order[:class_topk]]
            pred_chunks.append(kept_scores[:class_topk])
            point_chunks.append(selected_points)
            class_chunks.append(torch.full_like(selected_points, class_idx))

        audit["guard_above_threshold_before_per_class_cap"] = above_threshold_count
        if not pred_chunks:
            empty_segments = segments[:0]
            empty_scores = scores.new_zeros((0,))
            empty_labels = torch.zeros(0, dtype=torch.long, device=scores.device)
            audit["guard_class_candidate_count_before_global_topk"] = 0
            return empty_segments, empty_scores, empty_labels, audit

        pred_prob = torch.cat(pred_chunks, dim=0)
        pt_idxs = torch.cat(point_chunks, dim=0)
        cls_idxs = torch.cat(class_chunks, dim=0)
        audit["guard_class_candidate_count_before_global_topk"] = int(pred_prob.numel())

        num_topk = min(int(guard["total_candidate_cap"]), int(pred_prob.numel()))
        pred_prob, idxs = pred_prob.sort(descending=True)
        pred_prob = pred_prob[:num_topk].clone()
        pt_idxs = pt_idxs[idxs[:num_topk]].clone()
        cls_idxs = cls_idxs[idxs[:num_topk]].clone()
        return segments[pt_idxs], pred_prob, cls_idxs, audit

    def _select_bvr_twb_guarded_candidates(
        self,
        segments,
        scores,
        num_classes,
        guard,
    ):
        return self._select_route_guarded_candidates(
            segments,
            scores,
            num_classes,
            guard,
            "bvr_twb_guarded_raw_cap_per_class",
        )

    def _select_rba_rbr_guarded_candidates(
        self,
        segments,
        scores,
        num_classes,
        guard,
    ):
        return self._select_route_guarded_candidates(
            segments,
            scores,
            num_classes,
            guard,
            "rba_rbr_guarded_raw_cap_per_class",
        )

    def _append_env_jsonl_audit(self, env_key, row):
        path = os.environ.get(env_key)
        if not path:
            return
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    def _record_bvr_twb_grid_audit(self, metas, masks, grid):
        if os.environ.get("BVR_TWB_GRID_AUDIT", "").lower() not in {"1", "true", "yes"}:
            return
        rows = []
        for idx, meta in enumerate(metas):
            if not self._is_bvr_twb_meta(meta):
                continue
            positions = meta.get("bvr_twb_detector_feature_positions", [])
            native_axis = bool(meta.get("irregular_native_axis", False))
            mask_true = int(masks[idx].bool().sum().item())
            grid_valid_true = int(grid["valid_mask"][idx].bool().sum().item())
            row = {
                "audit_type": "bvr_twb_detector_temporal_grid",
                "route_label": "DIVERGENT_INNOVATION_BVR_TWB_DO_NOT_MERGE_WITH_C3",
                "video_name": meta.get("video_name", "unknown"),
                "dispatch_hit": True,
                "native_axis": native_axis,
                "mask_shape": list(masks[idx].shape),
                "mask_true_count": mask_true,
                "meta_detector_feature_position_count": int(len(positions)),
                "meta_detector_feature_valid_len": float(meta.get("bvr_twb_detector_feature_valid_len", 0.0)),
                "grid_center_prefix": [
                    float(value)
                    for value in grid["center"][idx, : min(mask_true, 8)].detach().cpu().tolist()
                ],
                "grid_valid_mask_true_count": grid_valid_true,
                "grid_fresh_mask_true_count": int(grid["fresh_mask"][idx].bool().sum().item()),
                "status": "PASS_NATIVE_AXIS_POSITIONS_ENTERED_MODEL",
            }
            rows.append(row)
        if rows:
            self._last_bvr_twb_grid_audit = rows
            for row in rows:
                self._append_env_jsonl_audit("BVR_TWB_GRID_AUDIT_PATH", row)

    def _record_rba_rbr_grid_audit(self, metas, masks, grid):
        if os.environ.get("RBA_RBR_GRID_AUDIT", "").lower() not in {"1", "true", "yes"}:
            return
        rows = []
        for idx, meta in enumerate(metas):
            if not self._is_rba_rbr_meta(meta):
                continue
            positions = meta.get("rba_rbr_detector_feature_positions", [])
            ledger = meta.get("rba_rbr_ledger", {}) or {}
            native_axis = bool(meta.get("irregular_native_axis", False))
            mask_true = int(masks[idx].bool().sum().item())
            grid_valid_true = int(grid["valid_mask"][idx].bool().sum().item())
            gap_diag = ledger.get("selection_gap_diagnostics", {}) if isinstance(ledger, dict) else {}
            row = {
                "audit_type": "rba_rbr_detector_temporal_grid",
                "route_label": "DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3",
                "video_name": meta.get("video_name", "unknown"),
                "dispatch_hit": True,
                "native_axis": native_axis,
                "mask_shape": list(masks[idx].shape),
                "mask_true_count": mask_true,
                "raw_valid_k": int(ledger.get("valid_k", len(meta.get("rba_rbr_raw_selected_positions", [])))),
                "dynamic_target_k": int(ledger.get("dynamic_target_k", 0)),
                "budget_stop_reason": ledger.get("budget_stop_reason", "unknown"),
                "pre_guard_budget_stop_reason": ledger.get("pre_guard_budget_stop_reason", "unknown"),
                "guard_reason": ledger.get("guard_reason", "unknown"),
                "guard_addition_count": int(ledger.get("guard_addition_count", 0)),
                "selected_max_gap": int(gap_diag.get("max_gap", ledger.get("max_raw_gap_after_guard", 0))),
                "selected_max_gap_before_guard": int(ledger.get("max_raw_gap_before_guard", 0)),
                "selected_max_gap_after_guard": int(ledger.get("max_raw_gap_after_guard", 0)),
                "meta_detector_feature_position_count": int(len(positions)),
                "meta_detector_feature_valid_len": float(meta.get("rba_rbr_detector_feature_valid_len", 0.0)),
                "detector_feature_valid_k": int(ledger.get("detector_feature_valid_k", mask_true)),
                "detector_mask_len": int(ledger.get("detector_mask_len", masks[idx].shape[0])),
                "detector_mask_true_count": int(ledger.get("detector_mask_true_count", mask_true)),
                "detector_feature_target_k": ledger.get("detector_feature_target_k", None),
                "max_detector_gap_before_guard": float(ledger.get("max_detector_gap_before_guard", 0.0)),
                "max_detector_gap_after_guard": float(ledger.get("max_detector_gap_after_guard", 0.0)),
                "adapter_padding_duplicate_count": int(ledger.get("adapter_padding_duplicate_count", 0)),
                "grid_center_prefix": [
                    float(value)
                    for value in grid["center"][idx, : min(mask_true, 8)].detach().cpu().tolist()
                ],
                "grid_valid_mask_true_count": grid_valid_true,
                "grid_fresh_mask_true_count": int(grid["fresh_mask"][idx].bool().sum().item()),
                "status": "PASS_RBA_RBR_NATIVE_AXIS_POSITIONS_ENTERED_MODEL",
            }
            rows.append(row)
        if rows:
            self._last_rba_rbr_grid_audit = rows
            for row in rows:
                self._append_env_jsonl_audit("RBA_RBR_GRID_AUDIT_PATH", row)

    def _bvr_twb_temporal_grid_from_meta(self, meta, mask):
        required = ("bvr_twb_detector_feature_positions", "bvr_twb_detector_feature_valid_len")
        missing = [key for key in required if key not in meta or meta.get(key) is None]
        if missing:
            raise ValueError(
                "BVR-TWB detector temporal grid requires "
                "bvr_twb_detector_feature_positions and bvr_twb_detector_feature_valid_len; "
                f"missing={missing}"
            )
        if not bool(meta.get("irregular_native_axis", False)):
            raise ValueError(
                "BVR-TWB detector temporal grid requires irregular_native_axis=True "
                "because remap_gt_to_selected_axis=False keeps Head coordinates on the native dense axis."
            )

        pos = torch.as_tensor(
            meta["bvr_twb_detector_feature_positions"],
            device=mask.device,
            dtype=torch.float32,
        ).flatten()
        if pos.numel() == 0:
            raise ValueError("BVR-TWB detector temporal grid received empty bvr_twb_detector_feature_positions")

        mask_valid = int(mask.bool().sum().item())
        if mask_valid != int(pos.numel()):
            raise ValueError(
                "BVR-TWB detector temporal grid mask true count must equal detector feature position count: "
                f"mask_true={mask_valid}, positions={int(pos.numel())}"
            )

        valid_len = max(int(round(float(meta["bvr_twb_detector_feature_valid_len"]))), 1)
        if valid_len <= float(pos.max().item()):
            raise ValueError(
                "BVR-TWB detector temporal grid native valid length must exceed the last detector feature position: "
                f"valid_len={valid_len}, last_position={float(pos.max().item())}"
        )
        return self._build_center_grid_from_positions(pos, valid_len, mask)

    def _rba_rbr_temporal_grid_from_meta(self, meta, mask):
        required = ("rba_rbr_detector_feature_positions", "rba_rbr_detector_feature_valid_len")
        missing = [key for key in required if key not in meta or meta.get(key) is None]
        if missing:
            raise ValueError(
                "RBA-RBR detector temporal grid requires "
                "rba_rbr_detector_feature_positions and rba_rbr_detector_feature_valid_len; "
                f"missing={missing}"
            )
        if not bool(meta.get("irregular_native_axis", False)):
            raise ValueError(
                "RBA-RBR detector temporal grid requires irregular_native_axis=True "
                "because remap_gt_to_selected_axis=False keeps Head coordinates on the native dense axis."
            )

        pos = torch.as_tensor(
            meta["rba_rbr_detector_feature_positions"],
            device=mask.device,
            dtype=torch.float32,
        ).flatten()
        if pos.numel() == 0:
            raise ValueError("RBA-RBR detector temporal grid received empty rba_rbr_detector_feature_positions")

        mask_valid = int(mask.bool().sum().item())
        if mask_valid != int(pos.numel()):
            raise ValueError(
                "RBA-RBR detector temporal grid mask true count must equal detector feature position count: "
                f"mask_true={mask_valid}, positions={int(pos.numel())}"
            )

        valid_len = max(int(round(float(meta["rba_rbr_detector_feature_valid_len"]))), 1)
        if valid_len <= float(pos.max().item()):
            raise ValueError(
                "RBA-RBR detector temporal grid native valid length must exceed the last detector feature position: "
                f"valid_len={valid_len}, last_position={float(pos.max().item())}"
            )
        return self._build_center_grid_from_positions(pos, valid_len, mask)

    def _temporal_grid_from_metas(self, metas, masks):
        if metas is None:
            return None
        if not all(
            (self._is_bvr_twb_meta(meta) or self._is_rba_rbr_meta(meta) or "irregular_selected_positions" in meta)
            for meta in metas
        ):
            return None

        grids = []
        target_len = masks.shape[1]
        for meta, mask in zip(metas, masks):
            if self._is_bvr_twb_meta(meta):
                grids.append(self._bvr_twb_temporal_grid_from_meta(meta, mask))
                continue
            if self._is_rba_rbr_meta(meta):
                grids.append(self._rba_rbr_temporal_grid_from_meta(meta, mask))
                continue

            pos = meta.get("irregular_selected_positions", None)
            valid_len = meta.get("irregular_selected_valid_len", None)
            if pos is None or valid_len is None:
                return None

            pos = torch.as_tensor(pos, device=mask.device, dtype=torch.float32)
            valid_len = max(int(round(float(valid_len))), 1)
            if pos.numel() == 0:
                pos = torch.zeros(1, device=mask.device, dtype=torch.float32)
                valid_len = 1

            if meta.get("irregular_native_axis", False):
                grids.append(self._build_center_grid_from_positions(pos, valid_len, mask))
                continue

            selected_len = max(min(valid_len, target_len), 1)
            pos = pos[:selected_len]
            if pos.numel() < target_len:
                pad = pos[-1:].repeat(target_len - pos.numel())
                pos = torch.cat([pos, pad], dim=0)

            fresh = torch.zeros(target_len, device=mask.device, dtype=torch.bool)
            fresh[:selected_len] = True
            grids.append(normalize_temporal_grid_input({"center": pos[:target_len][None], "fresh_mask": fresh[None]}, mask[None]))

        grid = {
            "center": torch.cat([grid["center"] for grid in grids], dim=0),
            "cell_left": torch.cat([grid["cell_left"] for grid in grids], dim=0),
            "cell_right": torch.cat([grid["cell_right"] for grid in grids], dim=0),
            "valid_mask": torch.cat([grid["valid_mask"] for grid in grids], dim=0),
            "fresh_mask": torch.cat([grid["fresh_mask"] for grid in grids], dim=0),
            "level_scale": torch.cat([grid["level_scale"] for grid in grids], dim=0),
        }
        self._record_bvr_twb_grid_audit(metas, masks, grid)
        self._record_rba_rbr_grid_audit(metas, masks, grid)
        return grid

    def pad_data(self, inputs, masks, temporal_grid=None):
        feat_len = inputs.shape[-1]
        if feat_len == self.max_seq_len:
            return inputs, masks, normalize_temporal_grid_input(temporal_grid, masks)
        if feat_len < self.max_seq_len:
            max_len = self.max_seq_len
        else:
            max_len = feat_len

        padding_size = [0, max_len - feat_len]
        inputs = torch.nn.functional.pad(inputs, padding_size, value=0)
        pad_masks = torch.zeros((inputs.shape[0], max_len), device=masks.device).bool()
        pad_masks[:, :feat_len] = masks
        temporal_grid = self._pad_temporal_grid(temporal_grid, feat_len, max_len, pad_masks)
        return inputs, pad_masks, temporal_grid

    def _project_features(self, x, masks, temporal_grid):
        if self.with_projection:
            feat_list, mask_list, temporal_grid_list = self.projection(x, masks, temporal_grid)
        else:
            temporal_grid = normalize_temporal_grid_input(temporal_grid, masks)
            feat_list = (x,)
            mask_list = (masks.bool(),)
            temporal_grid_list = (temporal_grid,)

        if self.with_neck:
            feat_list, mask_list, temporal_grid_list = self.neck(feat_list, mask_list, temporal_grid_list)
        return feat_list, mask_list, temporal_grid_list

    def forward_train(self, inputs, masks, metas, gt_segments, gt_labels, temporal_grids=None, **kwargs):
        losses = {}
        x = self.backbone(inputs, metas=metas) if self.with_backbone else inputs
        if temporal_grids is None:
            temporal_grids = self._temporal_grid_from_metas(metas, masks)
        x, masks, temporal_grid = self.pad_data(x, masks, temporal_grids)

        feat_list, mask_list, temporal_grid_list = self._project_features(x, masks, temporal_grid)

        losses.update(
            self.rpn_head.forward_train(
                feat_list,
                mask_list,
                temporal_grid_list=temporal_grid_list,
                gt_segments=gt_segments,
                gt_labels=gt_labels,
                **kwargs,
            )
        )
        losses["cost"] = sum(value for value in losses.values())
        return losses

    def set_train_epoch(self, curr_epoch):
        if hasattr(self, "rpn_head") and hasattr(self.rpn_head, "set_train_epoch"):
            self.rpn_head.set_train_epoch(curr_epoch)
        if hasattr(self, "backbone") and hasattr(self.backbone, "set_train_epoch"):
            self.backbone.set_train_epoch(curr_epoch)

    def forward_test(self, inputs, masks, metas=None, infer_cfg=None, temporal_grids=None, **kwargs):
        x = self.backbone(inputs, metas=metas) if self.with_backbone else inputs
        if temporal_grids is None:
            temporal_grids = self._temporal_grid_from_metas(metas, masks)
        x, masks, temporal_grid = self.pad_data(x, masks, temporal_grids)

        feat_list, mask_list, temporal_grid_list = self._project_features(x, masks, temporal_grid)

        return self.rpn_head.forward_test(
            feat_list,
            mask_list,
            temporal_grid_list=temporal_grid_list,
            **kwargs,
        )

    @torch.no_grad()
    def post_processing(self, predictions, metas, post_cfg, ext_cls, **kwargs):
        rpn_proposals, rpn_scores = predictions
        pre_nms_thresh = getattr(post_cfg, "pre_nms_thresh", 0.001)
        pre_nms_topk = getattr(post_cfg, "pre_nms_topk", 2000)
        num_classes = rpn_scores[0].shape[-1]

        results = {}
        bvr_audit_rows = []
        rba_audit_rows = []
        for i in range(len(metas)):
            segments = rpn_proposals[i].detach().cpu()
            scores = rpn_scores[i].detach().cpu()
            raw_proposal_count = int(segments.shape[0])
            raw_score_count = int(scores.numel())
            bvr_guard = self._resolve_bvr_twb_postprocess_guard(post_cfg, metas[i], pre_nms_thresh, pre_nms_topk)
            rba_guard = self._resolve_rba_rbr_postprocess_guard(post_cfg, metas[i], pre_nms_thresh, pre_nms_topk)
            if bvr_guard is not None and rba_guard is not None:
                raise ValueError("Only one sparse route postprocess guard may be enabled for a sample")
            guard = bvr_guard if bvr_guard is not None else rba_guard
            guard_route = "bvr_twb" if bvr_guard is not None else "rba_rbr" if rba_guard is not None else None
            audit_row = None
            audit_env_key = None
            if self._is_bvr_twb_meta(metas[i]) and os.environ.get("BVR_TWB_POSTPROCESS_AUDIT", "").lower() in {
                "1",
                "true",
                "yes",
            }:
                audit_row = {
                    "audit_type": "bvr_twb_postprocess_proposal_count",
                    "route_label": "DIVERGENT_INNOVATION_BVR_TWB_DO_NOT_MERGE_WITH_C3",
                    "video_name": metas[i].get("video_name", "unknown"),
                    "raw_proposal_count": raw_proposal_count,
                    "num_classes": int(num_classes),
                    "raw_score_shape": list(scores.shape),
                    "flattened_candidate_count": raw_score_count,
                    "pre_nms_thresh": float(pre_nms_thresh),
                    "pre_nms_topk": int(pre_nms_topk),
                    "nms_enabled": bool(post_cfg.sliding_window is False and post_cfg.nms is not None),
                    "native_axis": bool(metas[i].get("irregular_native_axis", False)),
                    "guard_active": bool(guard is not None),
                    "candidate_generation_mode": "legacy_flatten_all_classes",
                }
                audit_env_key = "BVR_TWB_POSTPROCESS_AUDIT_PATH"
            elif self._is_rba_rbr_meta(metas[i]) and os.environ.get("RBA_RBR_POSTPROCESS_AUDIT", "").lower() in {
                "1",
                "true",
                "yes",
            }:
                audit_row = {
                    "audit_type": "rba_rbr_postprocess_proposal_count",
                    "route_label": "DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3",
                    "video_name": metas[i].get("video_name", "unknown"),
                    "raw_proposal_count": raw_proposal_count,
                    "num_classes": int(num_classes),
                    "raw_score_shape": list(scores.shape),
                    "flattened_candidate_count": raw_score_count,
                    "pre_nms_thresh": float(pre_nms_thresh),
                    "pre_nms_topk": int(pre_nms_topk),
                    "nms_enabled": bool(post_cfg.sliding_window is False and post_cfg.nms is not None),
                    "native_axis": bool(metas[i].get("irregular_native_axis", False)),
                    "guard_active": bool(guard is not None),
                    "candidate_generation_mode": "legacy_flatten_all_classes",
                    "raw_valid_k": int((metas[i].get("rba_rbr_ledger", {}) or {}).get("valid_k", 0)),
                    "detector_feature_valid_k": int(
                        (metas[i].get("rba_rbr_ledger", {}) or {}).get("detector_feature_valid_k", 0)
                    ),
                }
                audit_env_key = "RBA_RBR_POSTPROCESS_AUDIT_PATH"

            if num_classes == 1:
                scores = scores.squeeze(-1)
                labels = torch.zeros(scores.shape[0]).contiguous()
                if audit_row is not None:
                    audit_row["above_threshold_count"] = int(scores.shape[0])
                    audit_row["pre_nms_selected_count"] = int(scores.shape[0])
            else:
                if guard_route == "bvr_twb":
                    segments, scores, labels, guard_audit = self._select_bvr_twb_guarded_candidates(
                        segments,
                        scores,
                        num_classes,
                        bvr_guard,
                    )
                    if audit_row is not None:
                        audit_row.update(guard_audit)
                        audit_row["above_threshold_count"] = int(
                            guard_audit["guard_class_candidate_count_before_global_topk"]
                        )
                        audit_row["pre_nms_selected_count"] = int(scores.shape[0])
                elif guard_route == "rba_rbr":
                    segments, scores, labels, guard_audit = self._select_rba_rbr_guarded_candidates(
                        segments,
                        scores,
                        num_classes,
                        rba_guard,
                    )
                    if audit_row is not None:
                        audit_row.update(guard_audit)
                        audit_row["above_threshold_count"] = int(
                            guard_audit["guard_class_candidate_count_before_global_topk"]
                        )
                        audit_row["pre_nms_selected_count"] = int(scores.shape[0])
                else:
                    pred_prob = scores.flatten()
                    keep_idxs1 = pred_prob > pre_nms_thresh
                    if audit_row is not None:
                        audit_row["above_threshold_count"] = int(keep_idxs1.sum().item())
                    pred_prob = pred_prob[keep_idxs1]
                    topk_idxs = keep_idxs1.nonzero(as_tuple=True)[0]
                    num_topk = min(pre_nms_topk, topk_idxs.size(0))
                    pred_prob, idxs = pred_prob.sort(descending=True)
                    pred_prob = pred_prob[:num_topk].clone()
                    topk_idxs = topk_idxs[idxs[:num_topk]].clone()
                    pt_idxs = torch.div(topk_idxs, num_classes, rounding_mode="floor")
                    cls_idxs = torch.fmod(topk_idxs, num_classes)
                    segments = segments[pt_idxs]
                    scores = pred_prob
                    labels = cls_idxs
                    if audit_row is not None:
                        audit_row["pre_nms_selected_count"] = int(num_topk)

            if post_cfg.sliding_window is False and post_cfg.nms is not None:
                segments, scores, labels = batched_nms(segments, scores, labels, **post_cfg.nms)
            if audit_row is not None:
                audit_row["post_nms_count"] = int(segments.shape[0])

            video_id = metas[i]["video_name"]
            segments = convert_to_seconds(segments, metas[i])

            if isinstance(ext_cls, list):
                labels = [ext_cls[label.item()] for label in labels]
            else:
                segments, labels, scores = ext_cls(video_id, segments, scores)
            if audit_row is not None:
                audit_row["final_result_count"] = int(len(scores))
                audit_row["explains_large_prediction_count"] = raw_score_count >= 365940
                audit_row["status"] = "PASS_POSTPROCESS_COUNT_AUDITED_NO_METRIC_CLAIM"
                if audit_env_key == "BVR_TWB_POSTPROCESS_AUDIT_PATH":
                    bvr_audit_rows.append(audit_row)
                if audit_env_key == "RBA_RBR_POSTPROCESS_AUDIT_PATH":
                    rba_audit_rows.append(audit_row)
                if audit_env_key is not None:
                    self._append_env_jsonl_audit(audit_env_key, audit_row)

            results_per_video = []
            for segment, label, score in zip(segments, labels, scores):
                results_per_video.append(
                    dict(
                        segment=[round(seg.item(), 2) for seg in segment],
                        label=label,
                        score=round(score.item(), 4),
                    )
                )

            if video_id in results:
                results[video_id].extend(results_per_video)
            else:
                results[video_id] = results_per_video
        if bvr_audit_rows:
            self._last_bvr_twb_postprocess_audit = bvr_audit_rows
        if rba_audit_rows:
            self._last_rba_rbr_postprocess_audit = rba_audit_rows
        return results

    def get_optim_groups(self, cfg):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d)
        blacklist_weight_modules = (nn.LayerNorm, nn.GroupNorm)

        for mn, m in self.named_modules():
            for pn, _ in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn
                if fpn.startswith("backbone"):
                    continue

                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
                elif pn.endswith("in_proj_weight") and isinstance(m, nn.MultiheadAttention):
                    decay.add(fpn)
                elif "output_norms" in fpn and pn in {"weight", "bias"}:
                    no_decay.add(fpn)
                elif pn.endswith("scale") and isinstance(m, (Scale, AffineDropPath)):
                    no_decay.add(fpn)
                elif pn.endswith("rel_pe"):
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters() if not pn.startswith("backbone")}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
        assert len(param_dict.keys() - union_params) == 0, (
            f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay sets!"
        )

        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": cfg["weight_decay"],
                "lr": cfg["lr"],
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
                "lr": cfg["lr"],
            },
        ]
        return optim_groups

    def _tensor_stats(self, tensor, name):
        detached = tensor.detach()
        finite = torch.isfinite(detached)
        finite_count = int(finite.sum().item())
        numel = detached.numel()
        stats_tensor = detached if torch.is_floating_point(detached) or torch.is_complex(detached) else detached.to(torch.float32)
        if finite_count > 0:
            finite_tensor = stats_tensor[finite]
            return {
                f"{name}_shape": tuple(detached.shape),
                f"{name}_dtype": str(detached.dtype),
                f"{name}_numel": int(numel),
                f"{name}_finite_count": finite_count,
                f"{name}_nonfinite_count": int(numel - finite_count),
                f"{name}_min": float(finite_tensor.min().item()),
                f"{name}_max": float(finite_tensor.max().item()),
                f"{name}_mean": float(finite_tensor.mean().item()),
                f"{name}_std": float(finite_tensor.std(unbiased=False).item()),
                f"{name}_absmax": float(finite_tensor.abs().max().item()),
            }
        return {
            f"{name}_shape": tuple(detached.shape),
            f"{name}_dtype": str(detached.dtype),
            f"{name}_numel": int(numel),
            f"{name}_finite_count": 0,
            f"{name}_nonfinite_count": int(numel),
        }

    def collect_runtime_debug(self, data_dict, bad_param_name):
        report = {"bad_param_name": bad_param_name}
        if hasattr(self, "projection") and hasattr(self.projection, "collect_debug_state"):
            report.update(self.projection.collect_debug_state())
        if hasattr(self, "neck") and hasattr(self.neck, "collect_debug_state"):
            report.update(self.neck.collect_debug_state())
        if hasattr(self, "rpn_head") and hasattr(self.rpn_head, "collect_debug_state"):
            report.update(self.rpn_head.collect_debug_state())

        metas = data_dict.get("metas", [])
        if metas is not None:
            report["batch_video_names"] = [meta.get("video_name", "unknown") for meta in metas]
            report["batch_irregular_valid_len"] = [meta.get("irregular_selected_valid_len", None) for meta in metas]

        gt_segments = data_dict.get("gt_segments", None)
        if gt_segments is not None:
            report["batch_num_gt"] = [int(seg.shape[0]) for seg in gt_segments]
            report["batch_gt_absmax"] = [float(seg.abs().max().item()) if seg.numel() > 0 else 0.0 for seg in gt_segments]

        masks = data_dict.get("masks", None)
        if masks is not None:
            report["mask_valid_per_sample"] = [int(mask.sum().item()) for mask in masks]
            report.update(self._tensor_stats(masks.to(torch.float32), "batch_mask_tensor"))

        inputs = data_dict.get("inputs", None)
        if inputs is not None:
            report.update(self._tensor_stats(inputs, "batch_inputs"))
        return report
