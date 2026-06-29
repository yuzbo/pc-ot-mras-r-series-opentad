import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import SELECTORS


@SELECTORS.register_module()
class PCOTMRASCoarseActionnessFrameScout(nn.Module):
    """A small action/background scout over compressed per-frame pixels."""

    def __init__(
        self,
        in_channels,
        hidden_channels=128,
        num_layers=3,
        kernel_size=5,
        dropout=0.0,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        layers = []
        curr_channels = in_channels
        padding = kernel_size // 2
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Conv1d(curr_channels, hidden_channels, kernel_size, padding=padding),
                    nn.GroupNorm(1, hidden_channels),
                    nn.GELU(),
                ]
            )
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            curr_channels = hidden_channels
        self.encoder = nn.Sequential(*layers)
        self.action_head = nn.Conv1d(hidden_channels, 1, kernel_size=1)

    def forward(self, scout_inputs, masks=None):
        # scout_inputs: [B, T, C]
        x = scout_inputs.transpose(1, 2).contiguous()
        if masks is not None:
            x = x * masks.unsqueeze(1).to(dtype=x.dtype)
        feats = self.encoder(x)
        action_logits = self.action_head(feats).squeeze(1)
        if masks is not None:
            action_logits = action_logits.masked_fill(~masks, -20.0)
        return {"action_logits": action_logits}


@SELECTORS.register_module()
class PCOTMRASIndirectPreBackboneFrameSelector(nn.Module):
    """Pre-backbone fixed-budget selector from coarse actionness signals.

    The hard path gathers real frames. A local straight-through path gives the
    detector loss a differentiable surrogate around the selected indices while
    preserving the exact hard frames in forward values.
    """

    def __init__(
        self,
        target_len=384,
        dense_window_size=768,
        selection_unit=1,
        scout_spatial_size=32,
        strategy="coarse_actionness_uncertainty",
        quotas=None,
        max_gap_guard_count=0,
        st_local_radius=2,
        st_scale=1.0,
        actionness_loss_weight=0.05,
        scout=None,
    ):
        super().__init__()
        if strategy != "coarse_actionness_uncertainty":
            raise ValueError(f"Unsupported C3 indirect selection strategy: {strategy}")
        if selection_unit != 1:
            raise ValueError("Clean C3 indirect selector currently supports frame-level selection_unit=1 only")
        if target_len <= 0 or dense_window_size <= 0:
            raise ValueError("target_len and dense_window_size must be positive")
        if target_len > dense_window_size:
            raise ValueError("target_len must be <= dense_window_size")
        if scout is None:
            raise ValueError("scout config is required")

        self.target_len = int(target_len)
        self.dense_window_size = int(dense_window_size)
        self.selection_unit = int(selection_unit)
        self.scout_spatial_size = int(scout_spatial_size)
        self.strategy = strategy
        self.quotas = quotas or dict(uniform=288, action=72, uncertainty=24, change=0, background=0)
        self.max_gap_guard_count = int(max_gap_guard_count)
        self.st_local_radius = int(st_local_radius)
        self.st_scale = float(st_scale)
        self.actionness_loss_weight = float(actionness_loss_weight)

        self.scout = SELECTORS.build(scout)

    def forward_train(self, inputs, masks, metas, gt_segments, gt_labels):
        dense_masks = self._normalize_dense_masks(masks, inputs)
        scout_inputs = self._build_scout_inputs(inputs)
        scout_outputs = self.scout(scout_inputs, dense_masks)
        action_logits = scout_outputs["action_logits"]
        selected = self._select_indices(action_logits, dense_masks)
        selected_inputs, selected_masks = self._gather_inputs(inputs, dense_masks, selected, action_logits)
        selected_gt_segments, selected_gt_labels = self._remap_gt(selected, dense_masks, gt_segments, gt_labels)
        selected_metas = self._remap_metas(metas, selected, dense_masks)

        losses = {}
        if self.actionness_loss_weight > 0:
            target = self._build_actionness_targets(action_logits, dense_masks, gt_segments)
            loss = F.binary_cross_entropy_with_logits(action_logits[dense_masks], target[dense_masks])
            losses["loss_c3_actionness"] = loss * self.actionness_loss_weight

        return dict(
            inputs=selected_inputs,
            masks=selected_masks,
            metas=selected_metas,
            gt_segments=selected_gt_segments,
            gt_labels=selected_gt_labels,
            losses=losses,
            selected_dense_indices=selected,
            action_logits=action_logits,
        )

    def forward_test(self, inputs, masks, metas):
        dense_masks = self._normalize_dense_masks(masks, inputs)
        scout_inputs = self._build_scout_inputs(inputs)
        with torch.no_grad():
            action_logits = self.scout(scout_inputs, dense_masks)["action_logits"]
            selected = self._select_indices(action_logits, dense_masks)
            selected_inputs, selected_masks = self._gather_inputs(inputs, dense_masks, selected, action_logits)
            selected_metas = self._remap_metas(metas, selected, dense_masks)
        return dict(
            inputs=selected_inputs,
            masks=selected_masks,
            metas=selected_metas,
            selected_dense_indices=selected,
            action_logits=action_logits,
        )

    def _normalize_dense_masks(self, masks, inputs):
        if masks is None:
            return torch.ones(inputs.shape[0], inputs.shape[3], device=inputs.device, dtype=torch.bool)
        dense_masks = masks.to(device=inputs.device, dtype=torch.bool)
        if dense_masks.shape[-1] != inputs.shape[3]:
            raise ValueError(f"Mask length {dense_masks.shape[-1]} does not match dense input T={inputs.shape[3]}")
        return dense_masks

    def _build_scout_inputs(self, inputs):
        if inputs.dim() != 6:
            raise ValueError(f"Expected inputs as [B,N,C,T,H,W], got shape {tuple(inputs.shape)}")
        if inputs.shape[1] != 1:
            raise ValueError("Clean C3 indirect selector expects num_clips=1 frame input")

        frames = inputs[:, 0].float().permute(0, 2, 1, 3, 4).contiguous()  # [B,T,C,H,W]
        bsz, dense_len, channels, height, width = frames.shape
        flat_frames = frames.reshape(bsz * dense_len, channels, height, width)
        lowres = F.interpolate(
            flat_frames,
            size=(self.scout_spatial_size, self.scout_spatial_size),
            mode="bilinear",
            align_corners=False,
        )
        return lowres.reshape(bsz, dense_len, -1)

    def _select_indices(self, action_logits, dense_masks):
        scores = self._selection_scores(action_logits, dense_masks)
        selected_rows = []
        for row_idx in range(action_logits.shape[0]):
            valid_idx = dense_masks[row_idx].nonzero(as_tuple=True)[0]
            if valid_idx.numel() == 0:
                raise ValueError("C3 indirect selector received a sample with zero valid frames")
            row_scores = {key: value[row_idx] for key, value in scores.items()}
            row_selected = self._select_one(row_scores, action_logits[row_idx], valid_idx)
            selected_rows.append(row_selected)
        return torch.stack(selected_rows, dim=0)

    def _selection_scores(self, action_logits, dense_masks):
        p_action = action_logits.sigmoid()
        entropy = -(p_action * (p_action + 1e-6).log() + (1.0 - p_action) * (1.0 - p_action + 1e-6).log())
        change = torch.zeros_like(p_action)
        change[:, 1:] = (p_action[:, 1:] - p_action[:, :-1]).abs()
        background = 1.0 - p_action
        invalid = ~dense_masks
        return dict(
            action=p_action.masked_fill(invalid, -1.0),
            uncertainty=entropy.masked_fill(invalid, -1.0),
            change=change.masked_fill(invalid, -1.0),
            background=background.masked_fill(invalid, -1.0),
        )

    def _select_one(self, scores, logits, valid_idx):
        picks = []
        picks.extend(self._uniform_positions(valid_idx, int(self.quotas.get("uniform", 0))).tolist())
        for role in ["action", "uncertainty", "change", "background"]:
            quota = int(self.quotas.get(role, 0))
            if quota <= 0:
                continue
            picks.extend(self._topk_unique(scores[role], valid_idx, quota, picks).tolist())
        if self.max_gap_guard_count > 0:
            picks.extend(self._max_gap_guard(valid_idx, picks, self.max_gap_guard_count).tolist())
        if len(picks) < self.target_len:
            mixed = scores["action"] + 0.5 * scores["uncertainty"] + 0.25 * scores["change"]
            picks.extend(self._topk_unique(mixed, valid_idx, self.target_len - len(picks), picks).tolist())

        if len(picks) == 0:
            picks = [int(valid_idx[0].item())]
        unique = sorted(set(int(p) for p in picks if int(p) in set(valid_idx.tolist())))
        if len(unique) < self.target_len:
            fill = self._uniform_positions(valid_idx, self.target_len).tolist()
            for pos in fill:
                pos = int(pos)
                if pos not in unique:
                    unique.append(pos)
                if len(unique) == self.target_len:
                    break
        if len(unique) < self.target_len:
            unique.extend([unique[-1]] * (self.target_len - len(unique)))
        unique = sorted(unique[: self.target_len])
        return valid_idx.new_tensor(unique)

    def _uniform_positions(self, valid_idx, count):
        if count <= 0:
            return valid_idx.new_empty((0,))
        if valid_idx.numel() == 1:
            return valid_idx.repeat(count)
        steps = torch.linspace(0, valid_idx.numel() - 1, steps=count, device=valid_idx.device)
        return valid_idx[steps.round().long().clamp(0, valid_idx.numel() - 1)]

    def _topk_unique(self, score, valid_idx, count, already):
        if count <= 0:
            return valid_idx.new_empty((0,))
        already_set = set(int(x) for x in already)
        sorted_valid = valid_idx[score[valid_idx].argsort(descending=True)]
        picks = []
        for idx in sorted_valid.tolist():
            if int(idx) not in already_set:
                picks.append(int(idx))
            if len(picks) == count:
                break
        return valid_idx.new_tensor(picks)

    def _max_gap_guard(self, valid_idx, picks, count):
        if count <= 0 or len(picks) < 2:
            return valid_idx.new_empty((0,))
        current = sorted(set(int(p) for p in picks))
        guards = []
        for _ in range(count):
            gaps = [(right - left, left, right) for left, right in zip(current[:-1], current[1:])]
            if not gaps:
                break
            _, left, right = max(gaps)
            mid = int(round((left + right) * 0.5))
            candidates = valid_idx[(valid_idx >= left) & (valid_idx <= right)]
            if candidates.numel() > 0:
                mid = int(candidates[(candidates - mid).abs().argmin()].item())
            if mid in current:
                break
            current.append(mid)
            current.sort()
            guards.append(mid)
        return valid_idx.new_tensor(guards)

    def _gather_inputs(self, inputs, dense_masks, selected, action_logits):
        gather_idx = selected[:, None, None, :, None, None].expand(
            -1, inputs.shape[1], inputs.shape[2], -1, inputs.shape[4], inputs.shape[5]
        )
        hard_inputs = inputs.gather(3, gather_idx)
        selected_masks = self._selected_masks(dense_masks, selected)

        if self.training and self.st_scale > 0 and self.st_local_radius > 0:
            soft_inputs = self._local_soft_inputs(inputs, selected, action_logits)
            selected_inputs = hard_inputs + self.st_scale * (soft_inputs - soft_inputs.detach())
        else:
            selected_inputs = hard_inputs
        return selected_inputs, selected_masks

    def _selected_masks(self, dense_masks, selected):
        gathered = dense_masks.gather(1, selected)
        unique_mask = torch.ones_like(gathered)
        unique_mask[:, 1:] = selected[:, 1:] != selected[:, :-1]
        return gathered & unique_mask

    def _local_soft_inputs(self, inputs, selected, action_logits):
        offsets = torch.arange(
            -self.st_local_radius,
            self.st_local_radius + 1,
            device=inputs.device,
            dtype=selected.dtype,
        )
        local_idx = (selected[:, :, None] + offsets[None, None, :]).clamp(0, inputs.shape[3] - 1)
        local_logits = action_logits.gather(1, local_idx.reshape(action_logits.shape[0], -1)).reshape(local_idx.shape)
        weights = F.softmax(local_logits, dim=-1)
        gather_idx = local_idx[:, None, None, :, :, None, None].expand(
            -1, inputs.shape[1], inputs.shape[2], -1, -1, inputs.shape[4], inputs.shape[5]
        )
        expanded = inputs[:, :, :, None].expand(-1, -1, -1, self.target_len, -1, -1, -1)
        local_frames = expanded.gather(4, gather_idx)
        return (local_frames * weights[:, None, None, :, :, None, None]).sum(dim=4)

    def _build_actionness_targets(self, action_logits, dense_masks, gt_segments):
        target = torch.zeros_like(action_logits)
        for batch_idx, segments in enumerate(gt_segments):
            if segments.numel() == 0:
                continue
            for seg in segments.to(device=action_logits.device, dtype=torch.float32):
                start = int(torch.floor(seg[0]).clamp(0, action_logits.shape[1] - 1).item())
                end = int(torch.ceil(seg[1]).clamp(0, action_logits.shape[1]).item())
                if end > start:
                    target[batch_idx, start:end] = 1.0
        return target * dense_masks.to(dtype=target.dtype)

    def _remap_gt(self, selected, dense_masks, gt_segments, gt_labels):
        remapped_segments = []
        remapped_labels = []
        for batch_idx, (segments, labels) in enumerate(zip(gt_segments, gt_labels)):
            valid_selected = selected[batch_idx][self._selected_masks(dense_masks[batch_idx : batch_idx + 1], selected[batch_idx : batch_idx + 1])[0]]
            if valid_selected.numel() == 0 or segments.numel() == 0:
                remapped_segments.append(segments.new_zeros((0, 2)))
                remapped_labels.append(labels.new_zeros((0,), dtype=labels.dtype))
                continue

            pos = valid_selected.to(device=segments.device, dtype=segments.dtype)
            starts = torch.searchsorted(pos, segments[:, 0].contiguous(), right=False)
            ends = torch.searchsorted(pos, segments[:, 1].contiguous(), right=True)
            starts = starts.clamp(0, self.target_len - 1).to(dtype=segments.dtype)
            ends = ends.clamp(0, self.target_len).to(dtype=segments.dtype)
            new_segments = torch.stack([starts, ends], dim=1)
            keep = new_segments[:, 1] > new_segments[:, 0]
            remapped_segments.append(new_segments[keep])
            remapped_labels.append(labels[keep])
        return remapped_segments, remapped_labels

    def _remap_metas(self, metas, selected, dense_masks):
        new_metas = []
        for batch_idx, meta in enumerate(metas):
            new_meta = copy.deepcopy(meta)
            selected_list = selected[batch_idx].detach().cpu().tolist()
            valid_len = int(dense_masks[batch_idx].sum().item())
            selected_mask = self._selected_masks(dense_masks[batch_idx : batch_idx + 1], selected[batch_idx : batch_idx + 1])[0]
            selected_valid_len = int(selected_mask.sum().item())
            old_stride = float(new_meta.get("snippet_stride", 1.0))
            if selected_valid_len > 0:
                new_meta["snippet_stride"] = old_stride * max(valid_len, 1) / float(selected_valid_len)
            new_meta["window_size"] = self.target_len
            new_meta["c3_indirect_fixed_axis"] = True
            new_meta["c3_indirect_irregular_axis"] = False
            new_meta["c3_indirect_original_dense_window_size"] = self.dense_window_size
            new_meta["c3_indirect_selected_dense_indices"] = selected_list
            new_meta["c3_indirect_valid_len"] = valid_len
            new_meta["c3_indirect_selected_valid_len"] = selected_valid_len
            returnable = new_meta
            new_metas.append(returnable)
        return new_metas
