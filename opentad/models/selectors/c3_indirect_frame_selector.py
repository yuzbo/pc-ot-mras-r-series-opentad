import copy
import math
import numbers

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
class PCOTMRASCADFDensityFrameScout(nn.Module):
    """A small scout that predicts coarse actionness and acquisition utility."""

    def __init__(
        self,
        in_channels,
        hidden_channels=128,
        num_layers=3,
        kernel_size=5,
        dropout=0.0,
        with_boundary_head=False,
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
        self.utility_head = nn.Conv1d(hidden_channels, 1, kernel_size=1)
        self.boundary_head = nn.Conv1d(hidden_channels, 1, kernel_size=1) if with_boundary_head else None

    def forward(self, scout_inputs, masks=None):
        x = scout_inputs.transpose(1, 2).contiguous()
        if masks is not None:
            x = x * masks.unsqueeze(1).to(dtype=x.dtype)
        feats = self.encoder(x)
        outputs = {
            "action_logits": self.action_head(feats).squeeze(1),
            "utility_logits": self.utility_head(feats).squeeze(1),
        }
        if self.boundary_head is not None:
            outputs["boundary_logits"] = self.boundary_head(feats).squeeze(1)
        if masks is not None:
            for key, value in outputs.items():
                outputs[key] = value.masked_fill(~masks, -20.0)
        return outputs


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
        density_alpha=0.65,
        density_alpha_schedule=None,
        density_temperature=1.0,
        density_weights=None,
        density_entropy_floor=0.45,
        density_entropy_loss_weight=0.0,
        density_repulsion_loss_weight=0.0,
        density_window_mass_loss_weight=0.0,
        density_max_gap_loss_weight=0.0,
        density_blue_noise_loss_weight=0.0,
        density_weak_target_loss_weight=0.0,
        density_uniform_floor=0.08,
        density_body_floor_weight=0.08,
        density_context_floor_weight=0.04,
        density_transition_smooth_radius=1,
        diagnostic_prediction_cap=None,
        diagnostic_proposal_cap=None,
        selected_index_aware_postprocess_enabled=False,
        physical_time_postprocess_enabled=False,
        boundary_loss_weight=0.0,
        boundary_target_radius=1,
        max_gap_guard_count=0,
        st_local_radius=2,
        st_scale=1.0,
        actionness_loss_weight=0.05,
        fast_cpu_selection=False,
        emit_selection_diagnostics=True,
        selection_diagnostics_interval=0,
        scout=None,
    ):
        super().__init__()
        supported_strategies = {"coarse_actionness_uncertainty", "cadf_density_mesh_st", "cadf_densitymesh_st"}
        if strategy not in supported_strategies:
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
        self.strategy = "cadf_density_mesh_st" if strategy == "cadf_densitymesh_st" else strategy
        if self.strategy == "coarse_actionness_uncertainty":
            self.quotas = quotas or dict(uniform=288, action=72, uncertainty=24, change=0, background=0)
        elif quotas is not None:
            raise ValueError("CADF/DensityMesh uses continuous acquisition density; category quotas are not allowed")
        self.density_alpha = float(density_alpha)
        self.density_alpha_schedule = self._validate_alpha_schedule(density_alpha_schedule)
        # Diagnostic/short-smoke only: this counter is intentionally not a
        # checkpointed buffer, so resumable long training remains validator-locked.
        self._density_alpha_train_step = 0
        self._last_density_alpha = min(max(self.density_alpha, 0.0), 1.0)
        self._last_density_alpha_policy = "static"
        self.density_temperature = float(density_temperature)
        self.density_weights = density_weights or dict(action=0.02, uncertainty=0.48, change=0.35, utility=0.15, boundary=0.0)
        self._validate_density_config(self.density_weights, scout)
        self.density_entropy_floor = float(density_entropy_floor)
        self.density_entropy_loss_weight = float(density_entropy_loss_weight)
        self.density_repulsion_loss_weight = float(density_repulsion_loss_weight)
        self.density_window_mass_loss_weight = self._validate_nonnegative_weight(
            density_window_mass_loss_weight, "density_window_mass_loss_weight"
        )
        self.density_max_gap_loss_weight = self._validate_nonnegative_weight(
            density_max_gap_loss_weight, "density_max_gap_loss_weight"
        )
        self.density_blue_noise_loss_weight = self._validate_nonnegative_weight(
            density_blue_noise_loss_weight, "density_blue_noise_loss_weight"
        )
        self.density_weak_target_loss_weight = self._validate_nonnegative_weight(
            density_weak_target_loss_weight, "density_weak_target_loss_weight"
        )
        self.density_uniform_floor = min(max(float(density_uniform_floor), 0.0), 0.95)
        self.density_body_floor_weight = min(max(float(density_body_floor_weight), 0.0), 0.95)
        self.density_context_floor_weight = min(max(float(density_context_floor_weight), 0.0), 0.95)
        self.density_transition_smooth_radius = max(int(density_transition_smooth_radius), 0)
        self.diagnostic_prediction_cap = None if diagnostic_prediction_cap is None else int(diagnostic_prediction_cap)
        self.diagnostic_proposal_cap = None if diagnostic_proposal_cap is None else int(diagnostic_proposal_cap)
        self.selected_index_aware_postprocess_enabled = bool(selected_index_aware_postprocess_enabled)
        self.physical_time_postprocess_enabled = bool(physical_time_postprocess_enabled)
        self.boundary_loss_weight = float(boundary_loss_weight)
        self.boundary_target_radius = int(boundary_target_radius)
        self.max_gap_guard_count = int(max_gap_guard_count)
        self.st_local_radius = int(st_local_radius)
        self.st_scale = float(st_scale)
        self.actionness_loss_weight = float(actionness_loss_weight)
        self.fast_cpu_selection = bool(fast_cpu_selection)
        self.emit_selection_diagnostics = bool(emit_selection_diagnostics)
        self.selection_diagnostics_interval = max(int(selection_diagnostics_interval), 0)
        self._selection_call_count = 0

        self.scout = SELECTORS.build(scout)

    def _validate_nonnegative_weight(self, value, name):
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValueError(f"{name} must be numeric, got {value!r}")
        weight = float(value)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"{name} must be finite and non-negative, got {weight}")
        return weight

    def _validate_alpha_schedule(self, schedule):
        if schedule is None:
            return None
        if not isinstance(schedule, dict):
            raise ValueError("density_alpha_schedule must be a dict when provided")

        allowed = {"train_start_alpha", "train_target_alpha", "warmup_iters", "test_alpha"}
        unknown = set(schedule) - allowed
        if unknown:
            raise ValueError(f"density_alpha_schedule contains unsupported keys: {sorted(unknown)}")

        parsed = dict(schedule)
        parsed.setdefault("train_start_alpha", 0.0)
        parsed.setdefault("train_target_alpha", self.density_alpha)
        parsed.setdefault("warmup_iters", 1)
        parsed.setdefault("test_alpha", "target")
        for key in ["train_start_alpha", "train_target_alpha"]:
            value = float(parsed[key])
            if not math.isfinite(value):
                raise ValueError(f"density_alpha_schedule {key} must be finite")
            parsed[key] = min(max(value, 0.0), 1.0)
        parsed["warmup_iters"] = max(int(parsed["warmup_iters"]), 1)
        test_alpha = parsed["test_alpha"]
        if isinstance(test_alpha, str):
            if test_alpha not in {"target", "start", "base"}:
                raise ValueError("density_alpha_schedule test_alpha must be 'target', 'start', 'base', or a numeric alpha")
        elif isinstance(test_alpha, numbers.Real) and not isinstance(test_alpha, bool):
            parsed["test_alpha"] = min(max(float(test_alpha), 0.0), 1.0)
        else:
            raise ValueError("density_alpha_schedule test_alpha must be 'target', 'start', 'base', or a numeric alpha")
        return parsed

    def _resolve_density_alpha(self, mode=None):
        if self.density_alpha_schedule is None:
            self._last_density_alpha_policy = "static"
            return min(max(self.density_alpha, 0.0), 1.0)

        schedule = self.density_alpha_schedule
        start = float(schedule["train_start_alpha"])
        target = float(schedule["train_target_alpha"])
        if mode == "test":
            policy = schedule["test_alpha"]
            self._last_density_alpha_policy = str(policy)
            if policy == "target":
                return target
            if policy == "start":
                return start
            if policy == "base":
                return min(max(self.density_alpha, 0.0), 1.0)
            return float(policy)

        warmup_iters = int(schedule["warmup_iters"])
        progress = min(max(float(self._density_alpha_train_step) / float(warmup_iters), 0.0), 1.0)
        self._last_density_alpha_policy = f"train_linear_warmup_{warmup_iters}_iters"
        return start + (target - start) * progress

    def _advance_density_alpha_schedule(self):
        if self.density_alpha_schedule is not None:
            self._density_alpha_train_step += 1

    def _validate_density_config(self, density_weights, scout):
        allowed_keys = {"action", "uncertainty", "change", "utility", "boundary"}
        unknown_keys = set(density_weights) - allowed_keys
        if unknown_keys:
            raise ValueError(f"density_weights contains unsupported keys: {sorted(unknown_keys)}")

        total_weight = 0.0
        for name, value in density_weights.items():
            if isinstance(value, bool) or not isinstance(value, numbers.Real):
                raise ValueError(f"density_weights must be numeric, got {name}={value!r}")
            weight = float(value)
            if not math.isfinite(weight):
                raise ValueError(f"density_weights must be finite, got {name}={weight}")
            if weight < 0.0:
                raise ValueError(f"density_weights must be non-negative, got {name}={weight}")
            total_weight += weight
        if total_weight <= 0.0:
            raise ValueError("density_weights must sum to a positive value")

        boundary_weight = float(density_weights.get("boundary", 0.0))
        if boundary_weight <= 0.0:
            return
        has_boundary_head = False
        if isinstance(scout, dict):
            has_boundary_head = bool(scout.get("with_boundary_head", False))
        else:
            has_boundary_head = getattr(scout, "boundary_head", None) is not None
        if not has_boundary_head:
            raise ValueError("CADF/DensityMesh boundary density weight requires scout.with_boundary_head=True")

    def forward_train(self, inputs, masks, metas, gt_segments, gt_labels):
        dense_masks = self._normalize_dense_masks(masks, inputs)
        scout_inputs = self._build_scout_inputs(inputs)
        scout_outputs = self.scout(scout_inputs, dense_masks)
        action_logits = scout_outputs["action_logits"]
        selection_outputs = self._select_indices(scout_outputs, dense_masks, mode="train")
        self._advance_density_alpha_schedule()
        selected = selection_outputs["selected"]
        st_logits = selection_outputs.get("st_logits", action_logits)
        selection_diagnostics = selection_outputs.get("diagnostics")
        selected_inputs, selected_masks = self._gather_inputs(inputs, dense_masks, selected, st_logits)
        selected_gt_segments, selected_gt_labels = self._remap_gt(selected, dense_masks, gt_segments, gt_labels)
        selected_metas = self._remap_metas(metas, selected, dense_masks, selection_diagnostics)
        self._attach_train_gt_diagnostics(selected_metas, gt_segments, selected_gt_segments)

        losses = {}
        if self.actionness_loss_weight > 0:
            target = self._build_actionness_targets(action_logits, dense_masks, gt_segments)
            loss = F.binary_cross_entropy_with_logits(action_logits[dense_masks], target[dense_masks])
            losses["loss_c3_actionness"] = loss * self.actionness_loss_weight
        if self.boundary_loss_weight > 0 and "boundary_logits" in scout_outputs:
            target = self._build_boundary_targets(scout_outputs["boundary_logits"], dense_masks, gt_segments)
            loss = F.binary_cross_entropy_with_logits(scout_outputs["boundary_logits"][dense_masks], target[dense_masks])
            losses["loss_c3_boundary"] = loss * self.boundary_loss_weight
        if self.density_entropy_loss_weight > 0 and "density" in selection_outputs:
            entropy_loss = self._density_entropy_floor_loss(selection_outputs["density"], dense_masks)
            losses["loss_c3_density_entropy"] = entropy_loss * self.density_entropy_loss_weight
        if self.density_repulsion_loss_weight > 0:
            repulsion_loss = self._selected_repulsion_loss(selected, dense_masks)
            losses["loss_c3_density_repulsion"] = repulsion_loss * self.density_repulsion_loss_weight
        if (
            (self.density_window_mass_loss_weight > 0 or self.density_max_gap_loss_weight > 0)
            and "density" in selection_outputs
        ):
            mass_loss, gap_loss = self._density_window_mass_gap_loss(
                selection_outputs["density"], dense_masks, return_parts=True
            )
            if self.density_window_mass_loss_weight > 0:
                losses["loss_c3_density_window_mass"] = mass_loss * self.density_window_mass_loss_weight
            if self.density_max_gap_loss_weight > 0:
                losses["loss_c3_density_max_gap"] = gap_loss * self.density_max_gap_loss_weight
        if self.density_blue_noise_loss_weight > 0 and "density" in selection_outputs:
            blue_noise_loss = self._density_blue_noise_repulsion_loss(selection_outputs["density"], dense_masks)
            losses["loss_c3_density_blue_noise"] = blue_noise_loss * self.density_blue_noise_loss_weight
        if self.density_weak_target_loss_weight > 0 and "density" in selection_outputs:
            weak_target = self._build_density_weak_targets(selection_outputs["density"], dense_masks, gt_segments)
            weak_loss = -(weak_target * selection_outputs["density"].clamp_min(1e-6).log()).sum(dim=1).mean()
            losses["loss_c3_density_weak_target"] = weak_loss * self.density_weak_target_loss_weight
            for meta in selected_metas:
                meta["c3_density_weak_target_loss_enabled"] = True

        output = dict(
            inputs=selected_inputs,
            masks=selected_masks,
            metas=selected_metas,
            gt_segments=selected_gt_segments,
            gt_labels=selected_gt_labels,
            losses=losses,
            selected_dense_indices=selected,
            action_logits=action_logits,
        )
        for key in ["utility_logits", "boundary_logits"]:
            if key in scout_outputs:
                output[key] = scout_outputs[key]
        if "density" in selection_outputs:
            output["density"] = selection_outputs["density"]
        return output

    def forward_test(self, inputs, masks, metas):
        dense_masks = self._normalize_dense_masks(masks, inputs)
        scout_inputs = self._build_scout_inputs(inputs)
        with torch.no_grad():
            scout_outputs = self.scout(scout_inputs, dense_masks)
            action_logits = scout_outputs["action_logits"]
            selection_outputs = self._select_indices(scout_outputs, dense_masks, mode="test")
            selected = selection_outputs["selected"]
            st_logits = selection_outputs.get("st_logits", action_logits)
            selection_diagnostics = selection_outputs.get("diagnostics")
            selected_inputs, selected_masks = self._gather_inputs(inputs, dense_masks, selected, st_logits)
            selected_metas = self._remap_metas(metas, selected, dense_masks, selection_diagnostics)
        output = dict(
            inputs=selected_inputs,
            masks=selected_masks,
            metas=selected_metas,
            selected_dense_indices=selected,
            action_logits=action_logits,
        )
        for key in ["utility_logits", "boundary_logits"]:
            if key in scout_outputs:
                output[key] = scout_outputs[key]
        if "density" in selection_outputs:
            output["density"] = selection_outputs["density"]
        return output

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

    def _select_indices(self, scout_outputs, dense_masks, mode=None):
        if self.strategy == "cadf_density_mesh_st":
            return self._select_cadf_density_mesh(scout_outputs, dense_masks, mode=mode)

        action_logits = scout_outputs["action_logits"]
        scores = self._selection_scores(action_logits, dense_masks)
        selected_rows = []
        for row_idx in range(action_logits.shape[0]):
            valid_idx = dense_masks[row_idx].nonzero(as_tuple=True)[0]
            if valid_idx.numel() == 0:
                raise ValueError("C3 indirect selector received a sample with zero valid frames")
            row_scores = {key: value[row_idx] for key, value in scores.items()}
            row_selected = self._select_one(row_scores, action_logits[row_idx], valid_idx)
            selected_rows.append(row_selected)
        selected = torch.stack(selected_rows, dim=0)
        diagnostics = self._selection_diagnostics(selected, dense_masks, None, torch.zeros_like(selected, dtype=torch.float32))
        return {"selected": selected, "st_logits": action_logits, "diagnostics": diagnostics}

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

    def _select_cadf_density_mesh(self, scout_outputs, dense_masks, mode=None):
        density, acquisition_logits = self._cadf_density(scout_outputs, dense_masks, mode=mode)
        selected_rows = []
        repair_rows = []
        repair_stats_rows = []
        emit_diagnostics = self._should_emit_selection_diagnostics()
        for row_idx in range(density.shape[0]):
            valid_idx = dense_masks[row_idx].nonzero(as_tuple=True)[0]
            if valid_idx.numel() == 0:
                raise ValueError("CADF/DensityMesh received a sample with zero valid frames")
            row_selected, row_repair, row_repair_stats = self._inverse_cdf_select_one(
                density[row_idx], valid_idx, return_diagnostics=True
            )
            selected_rows.append(row_selected)
            repair_rows.append(row_repair)
            repair_stats_rows.append(row_repair_stats)
        selected = torch.stack(selected_rows, dim=0)
        diagnostics = None
        if emit_diagnostics:
            repairs = torch.stack(repair_rows, dim=0)
            diagnostics = self._selection_diagnostics(
                selected,
                dense_masks,
                density,
                repairs,
                repair_stats=repair_stats_rows,
            )
        self._selection_call_count += 1
        return {
            "selected": selected,
            "st_logits": acquisition_logits,
            "density": density,
            "diagnostics": diagnostics,
        }

    def _should_emit_selection_diagnostics(self):
        if not self.emit_selection_diagnostics:
            return False
        if self.selection_diagnostics_interval <= 0:
            return True
        return (self._selection_call_count % self.selection_diagnostics_interval) == 0

    def _cadf_density(self, scout_outputs, dense_masks, mode=None):
        action_logits = scout_outputs["action_logits"]
        p_action = action_logits.sigmoid()
        entropy = -(p_action * (p_action + 1e-6).log() + (1.0 - p_action) * (1.0 - p_action + 1e-6).log())
        change = torch.zeros_like(p_action)
        change[:, 1:] = (p_action[:, 1:] - p_action[:, :-1]).abs()
        change = self._smooth_rows(change, dense_masks, self.density_transition_smooth_radius)
        utility_logits = scout_outputs.get("utility_logits", torch.zeros_like(action_logits))
        boundary_logits = scout_outputs.get("boundary_logits", torch.zeros_like(action_logits))

        action_score = self._robust_normalize_rows(action_logits, dense_masks)
        action_floor_prob = action_score.sigmoid() * dense_masks.to(dtype=action_score.dtype)
        action_context = self._smooth_rows(action_floor_prob, dense_masks, radius=2)
        entropy_score = self._robust_normalize_rows(entropy * action_context.clamp_min(0.05), dense_masks)
        change_score = self._robust_normalize_rows(change * action_context.clamp_min(0.05), dense_masks)
        utility_score = self._robust_normalize_rows(utility_logits, dense_masks)
        boundary_score = self._robust_normalize_rows(boundary_logits, dense_masks)

        weights = self.density_weights
        acquisition_logits = (
            float(weights.get("action", 0.0)) * action_score
            + float(weights.get("uncertainty", 0.0)) * entropy_score
            + float(weights.get("change", 0.0)) * change_score
            + float(weights.get("utility", 0.0)) * utility_score
            + float(weights.get("boundary", 0.0)) * boundary_score
        )
        acquisition_logits = acquisition_logits.masked_fill(~dense_masks, -20.0)
        temperature = max(self.density_temperature, 1e-4)
        learned_density = F.softmax(acquisition_logits / temperature, dim=1) * dense_masks.to(dtype=acquisition_logits.dtype)
        learned_density = learned_density / learned_density.sum(dim=1, keepdim=True).clamp_min(1e-6)
        uniform_density = dense_masks.to(dtype=acquisition_logits.dtype)
        uniform_density = uniform_density / uniform_density.sum(dim=1, keepdim=True).clamp_min(1.0)
        floor_mix = self._density_safety_floor(uniform_density, action_floor_prob, action_context, dense_masks)
        learned_density = self._mix_density_with_safety_floor(learned_density, floor_mix)
        alpha = self._resolve_density_alpha(mode=mode)
        self._last_density_alpha = alpha
        density = (1.0 - alpha) * uniform_density + alpha * learned_density
        density = density * dense_masks.to(dtype=density.dtype)
        density = density / density.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return density, acquisition_logits

    def _smooth_rows(self, values, dense_masks, radius):
        if radius <= 0:
            return values * dense_masks.to(dtype=values.dtype)
        kernel = 2 * radius + 1
        row_values = values.masked_fill(~dense_masks, 0.0).unsqueeze(1)
        row_masks = dense_masks.to(dtype=values.dtype).unsqueeze(1)
        numerator = F.avg_pool1d(row_values, kernel_size=kernel, stride=1, padding=radius) * kernel
        denominator = F.avg_pool1d(row_masks, kernel_size=kernel, stride=1, padding=radius) * kernel
        return (numerator / denominator.clamp_min(1.0)).squeeze(1).masked_fill(~dense_masks, 0.0)

    def _normalize_positive_rows(self, values, dense_masks):
        positive = values.clamp_min(0.0) * dense_masks.to(dtype=values.dtype)
        return positive / positive.sum(dim=1, keepdim=True).clamp_min(1e-6)

    def _density_safety_floor(self, uniform_density, p_action, action_context, dense_masks):
        body_floor = self._normalize_positive_rows(p_action, dense_masks)
        context_floor = self._normalize_positive_rows(action_context, dense_masks)
        total = self.density_uniform_floor + self.density_body_floor_weight + self.density_context_floor_weight
        if total <= 0.0:
            return uniform_density
        mixed = (
            self.density_uniform_floor * uniform_density
            + self.density_body_floor_weight * body_floor
            + self.density_context_floor_weight * context_floor
        )
        return mixed / mixed.sum(dim=1, keepdim=True).clamp_min(1e-6)

    def _mix_density_with_safety_floor(self, learned_density, floor_density):
        floor_weight = self.density_uniform_floor + self.density_body_floor_weight + self.density_context_floor_weight
        floor_weight = min(max(float(floor_weight), 0.0), 0.95)
        if floor_weight <= 0.0:
            return learned_density
        mixed = (1.0 - floor_weight) * learned_density + floor_weight * floor_density
        return mixed / mixed.sum(dim=1, keepdim=True).clamp_min(1e-6)

    def _robust_normalize_rows(self, values, dense_masks, clamp_value=5.0):
        if values.numel() == 0:
            return torch.zeros_like(values)
        counts = dense_masks.sum(dim=1)
        median_idx = ((counts.clamp_min(1) - 1) // 2).clamp_min(0)
        inf = torch.tensor(float("inf"), device=values.device, dtype=values.dtype)
        sorted_values = values.masked_fill(~dense_masks, inf).sort(dim=1).values
        center = sorted_values.gather(1, median_idx[:, None])
        deviations = (values - center).abs().masked_fill(~dense_masks, inf)
        sorted_deviations = deviations.sort(dim=1).values
        mad = sorted_deviations.gather(1, median_idx[:, None])
        scale = (mad * 1.4826).clamp_min(1e-6)
        normalized = ((values - center) / scale).clamp(-clamp_value, clamp_value)
        valid = dense_masks & (counts[:, None] > 1)
        return torch.where(valid, normalized, torch.zeros_like(values))

    def _inverse_cdf_select_one(self, density, valid_idx, return_diagnostics=False):
        if self.target_len <= 0:
            raise ValueError("target_len must be positive")
        valid_density = density[valid_idx]
        valid_density = valid_density / valid_density.sum().clamp_min(1e-6)
        if valid_idx.numel() == 1:
            selected = valid_idx.repeat(self.target_len)
            repairs = torch.zeros_like(selected, dtype=torch.float32)
            repair_stats = dict(dedupe_repair_count=0, gap_guard_add_count=0, gap_guard_prune_count=0, pad_repair_count=0)
            if return_diagnostics:
                return selected, repairs, repair_stats
            return selected, repairs

        target_unique = min(self.target_len, int(valid_idx.numel()))
        cdf = valid_density.cumsum(dim=0)
        quantiles = (torch.arange(target_unique, device=density.device, dtype=valid_density.dtype) + 0.5) / float(target_unique)
        selected_pos = torch.searchsorted(cdf, quantiles, right=False).clamp(0, valid_idx.numel() - 1)
        selected = valid_idx[selected_pos]

        if self.fast_cpu_selection:
            selected, repair_mask, repair_stats = self._repair_inverse_cdf_selection_cpu_once(
                selected,
                valid_idx,
                density,
                target_unique,
            )
        else:
            selected, repair_mask = self._dedupe_inverse_cdf_selection(selected, valid_idx, valid_density, target_unique)
            repair_stats = dict(
                dedupe_repair_count=int(repair_mask.float().sum().item()),
                gap_guard_add_count=0,
                gap_guard_prune_count=0,
                pad_repair_count=0,
            )
        if (not self.fast_cpu_selection) and self.max_gap_guard_count > 0 and selected.numel() >= 2:
            pre_guard = [int(x) for x in selected.tolist()]
            pre_mask_by_idx = {
                int(idx): float(mask)
                for idx, mask in zip(selected.detach().cpu().tolist(), repair_mask.detach().cpu().tolist())
            }
            guards = [int(x) for x in self._max_gap_guard(valid_idx, pre_guard, self.max_gap_guard_count).tolist()]
            pre_guard_set = set(pre_guard)
            added_guards = [idx for idx in guards if idx not in pre_guard_set]
            repair_stats["gap_guard_add_count"] = len(set(added_guards))
            guarded = sorted(set(pre_guard + guards))
            before_prune_count = len(guarded)
            if len(guarded) > target_unique:
                guarded = self._prune_guarded_selection(guarded, density, target_unique)
            guarded = sorted(set(int(x) for x in guarded))[:target_unique]
            repair_stats["gap_guard_prune_count"] = max(0, before_prune_count - len(guarded))
            added_guard_set = set(added_guards)
            selected = valid_idx.new_tensor(guarded)
            repair_values = [1.0 if idx in added_guard_set else pre_mask_by_idx.get(idx, 0.0) for idx in guarded]
            repair_mask = valid_density.new_tensor(repair_values)

        if selected.numel() < self.target_len:
            pad = selected[-1:].repeat(self.target_len - selected.numel())
            selected = torch.cat([selected, pad], dim=0)
            pad_mask = torch.ones_like(pad, dtype=torch.float32)
            repair_mask = torch.cat([repair_mask, pad_mask], dim=0)
            repair_stats["pad_repair_count"] = int(pad_mask.numel())
        selected = selected[: self.target_len]
        repair_mask = repair_mask[: self.target_len]
        if return_diagnostics:
            return selected, repair_mask, repair_stats
        return selected, repair_mask

    def _repair_inverse_cdf_selection_cpu_once(self, selected, valid_idx, density, target_unique):
        valid_list = [int(x) for x in valid_idx.detach().cpu().tolist()]
        selected_list = [int(x) for x in selected.detach().cpu().tolist()]
        density_cpu = density.detach().float().cpu()
        valid_positions = {idx: pos for pos, idx in enumerate(valid_list)}
        used = set()
        repaired = []
        repair_mask = []

        for idx in selected_list:
            if idx not in used:
                repaired.append(idx)
                used.add(idx)
                repair_mask.append(0.0)
                continue
            replacement = None
            center_pos = valid_positions.get(idx, 0)
            for radius in range(1, len(valid_list) + 1):
                for cand_pos in (center_pos - radius, center_pos + radius):
                    if cand_pos < 0 or cand_pos >= len(valid_list):
                        continue
                    cand = valid_list[cand_pos]
                    if cand not in used:
                        replacement = cand
                        break
                if replacement is not None:
                    break
            if replacement is None:
                replacement = idx
            repaired.append(replacement)
            used.add(replacement)
            repair_mask.append(1.0)

        if len(repaired) < target_unique:
            for cand in self._uniform_positions_cpu(valid_list, target_unique):
                if cand not in used:
                    repaired.append(cand)
                    used.add(cand)
                    repair_mask.append(1.0)
                if len(repaired) == target_unique:
                    break

        order = sorted(range(len(repaired)), key=lambda i: repaired[i])
        repaired = [repaired[i] for i in order[:target_unique]]
        repair_mask = [repair_mask[i] for i in order[:target_unique]]
        repair_stats = dict(
            dedupe_repair_count=int(sum(repair_mask)),
            gap_guard_add_count=0,
            gap_guard_prune_count=0,
            pad_repair_count=0,
        )

        if self.max_gap_guard_count > 0 and len(repaired) >= 2:
            pre_guard = list(repaired)
            pre_mask_by_idx = {int(idx): float(mask) for idx, mask in zip(repaired, repair_mask)}
            guards = self._max_gap_guard_cpu(valid_list, pre_guard, self.max_gap_guard_count)
            pre_guard_set = set(pre_guard)
            added_guards = [idx for idx in guards if idx not in pre_guard_set]
            repair_stats["gap_guard_add_count"] = len(set(added_guards))
            guarded = sorted(set(pre_guard + guards))
            before_prune_count = len(guarded)
            if len(guarded) > target_unique:
                guarded = self._prune_guarded_selection_cpu(guarded, density_cpu, target_unique)
            guarded = sorted(set(int(x) for x in guarded))[:target_unique]
            repair_stats["gap_guard_prune_count"] = max(0, before_prune_count - len(guarded))
            added_guard_set = set(added_guards)
            repaired = guarded
            repair_mask = [
                1.0 if idx in added_guard_set else pre_mask_by_idx.get(idx, 0.0)
                for idx in guarded
            ]

        return valid_idx.new_tensor(repaired), density.new_tensor(repair_mask, dtype=torch.float32), repair_stats

    def _uniform_positions_cpu(self, valid_list, count):
        if count <= 0:
            return []
        if len(valid_list) == 1:
            return [valid_list[0]] * count
        steps = torch.linspace(0, len(valid_list) - 1, steps=count, device="cpu")
        positions = steps.round().long().clamp(0, len(valid_list) - 1).tolist()
        return [valid_list[int(pos)] for pos in positions]

    def _max_gap_guard_cpu(self, valid_list, picks, count):
        if count <= 0 or len(picks) < 2:
            return []
        current = sorted(set(int(p) for p in picks))
        guards = []
        for _ in range(count):
            gaps = [(right - left, left, right) for left, right in zip(current[:-1], current[1:])]
            if not gaps:
                break
            _, left, right = max(gaps)
            mid = int(round((left + right) * 0.5))
            candidates = [idx for idx in valid_list if left <= idx <= right]
            if candidates:
                mid = min(candidates, key=lambda idx: abs(idx - mid))
            if mid in current:
                break
            current.append(mid)
            current.sort()
            guards.append(mid)
        return guards

    def _prune_guarded_selection_cpu(self, picks, density_cpu, target_unique):
        pruned = sorted(set(int(x) for x in picks))
        while len(pruned) > target_unique:
            best_remove = None
            best_key = None
            for candidate in pruned:
                trial = [idx for idx in pruned if idx != candidate]
                if len(trial) < 2:
                    max_gap = 0
                    mean_gap = 0.0
                else:
                    gaps = [right - left for left, right in zip(trial[:-1], trial[1:])]
                    max_gap = max(gaps)
                    mean_gap = sum(gaps) / float(len(gaps))
                score = float(density_cpu[candidate])
                key = (max_gap, mean_gap, score)
                if best_key is None or key < best_key:
                    best_key = key
                    best_remove = candidate
            pruned.remove(best_remove)
        return pruned

    def _prune_guarded_selection(self, picks, density, target_unique):
        pruned = sorted(set(int(x) for x in picks))
        while len(pruned) > target_unique:
            best_remove = None
            best_key = None
            for candidate in pruned:
                trial = [idx for idx in pruned if idx != candidate]
                if len(trial) < 2:
                    max_gap = 0
                    mean_gap = 0.0
                else:
                    gaps = [right - left for left, right in zip(trial[:-1], trial[1:])]
                    max_gap = max(gaps)
                    mean_gap = sum(gaps) / float(len(gaps))
                score = float(density[candidate].detach().item())
                key = (max_gap, mean_gap, score)
                if best_key is None or key < best_key:
                    best_key = key
                    best_remove = candidate
            pruned.remove(best_remove)
        return pruned

    def _dedupe_inverse_cdf_selection(self, selected, valid_idx, valid_density, target_unique):
        used = set()
        repaired = []
        repair_mask = []
        valid_positions = {int(idx): pos for pos, idx in enumerate(valid_idx.tolist())}
        for idx in selected.tolist():
            idx = int(idx)
            if idx not in used:
                repaired.append(idx)
                used.add(idx)
                repair_mask.append(0.0)
                continue
            replacement = None
            center_pos = valid_positions.get(idx, 0)
            for radius in range(1, int(valid_idx.numel()) + 1):
                candidate_positions = [center_pos - radius, center_pos + radius]
                for cand_pos in candidate_positions:
                    if cand_pos < 0 or cand_pos >= valid_idx.numel():
                        continue
                    cand = int(valid_idx[cand_pos].item())
                    if cand not in used:
                        replacement = cand
                        break
                if replacement is not None:
                    break
            if replacement is None:
                replacement = idx
            repaired.append(replacement)
            used.add(replacement)
            repair_mask.append(1.0)
        if len(repaired) < target_unique:
            for cand in self._uniform_positions(valid_idx, target_unique).tolist():
                cand = int(cand)
                if cand not in used:
                    repaired.append(cand)
                    used.add(cand)
                    repair_mask.append(1.0)
                if len(repaired) == target_unique:
                    break
        order = sorted(range(len(repaired)), key=lambda i: repaired[i])
        repaired = [repaired[i] for i in order[:target_unique]]
        repair_mask = [repair_mask[i] for i in order[:target_unique]]
        return valid_idx.new_tensor(repaired), valid_density.new_tensor(repair_mask)

    def _selection_diagnostics(self, selected, dense_masks, density, repair_mask, repair_stats=None):
        rows = []
        for row_idx in range(selected.shape[0]):
            row_selected = selected[row_idx]
            row_valid = dense_masks[row_idx]
            selected_mask = self._selected_masks(row_valid[None], row_selected[None])[0]
            valid_selected = row_selected[selected_mask]
            if valid_selected.numel() >= 2:
                gaps = valid_selected[1:] - valid_selected[:-1]
                max_gap = int(gaps.max().item())
                mean_gap = float(gaps.float().mean().item())
            else:
                max_gap = 0
                mean_gap = 0.0
            row = {
                "max_gap": max_gap,
                "mean_gap": mean_gap,
                "repair_fraction": float(repair_mask[row_idx].float().mean().item()) if repair_mask.numel() > 0 else 0.0,
                "row_repair_fraction": float(repair_mask[row_idx].float().mean().item()) if repair_mask.numel() > 0 else 0.0,
                "repair_count": int(repair_mask[row_idx].float().sum().item()) if repair_mask.numel() > 0 else 0,
                "alpha": float(self._last_density_alpha),
                "temperature": float(self.density_temperature),
            }
            if repair_stats is not None:
                row_stats = repair_stats[row_idx]
                row["dedupe_repair_count"] = int(row_stats.get("dedupe_repair_count", 0))
                row["gap_guard_add_count"] = int(row_stats.get("gap_guard_add_count", 0))
                row["gap_guard_prune_count"] = int(row_stats.get("gap_guard_prune_count", 0))
                row["pad_repair_count"] = int(row_stats.get("pad_repair_count", 0))
            if density is not None:
                valid_density = density[row_idx][row_valid]
                valid_positions = row_valid.nonzero(as_tuple=True)[0]
                entropy = -(valid_density * (valid_density + 1e-6).log()).sum()
                denom = torch.log(torch.tensor(float(max(valid_density.numel(), 2)), device=density.device, dtype=density.dtype))
                row["density_entropy"] = float((entropy / denom.clamp_min(1e-6)).item())
                row["selected_density_sum"] = float(density[row_idx].gather(0, row_selected).sum().item())
                top_count = min(8, int(valid_density.numel()))
                if top_count > 0:
                    top_pos = valid_positions[valid_density.argsort(descending=True)[:top_count]]
                    row["density_top_positions"] = [int(x) for x in top_pos.detach().cpu().tolist()]
                else:
                    row["density_top_positions"] = []
            rows.append(row)
        return rows

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

    def _build_boundary_targets(self, boundary_logits, dense_masks, gt_segments):
        target = torch.zeros_like(boundary_logits)
        radius = max(self.boundary_target_radius, 0)
        for batch_idx, segments in enumerate(gt_segments):
            if segments.numel() == 0:
                continue
            for seg in segments.to(device=boundary_logits.device, dtype=torch.float32):
                for value in [seg[0], seg[1]]:
                    center = int(torch.round(value).clamp(0, boundary_logits.shape[1] - 1).item())
                    left = max(0, center - radius)
                    right = min(boundary_logits.shape[1], center + radius + 1)
                    target[batch_idx, left:right] = 1.0
        return target * dense_masks.to(dtype=target.dtype)

    def _density_entropy_floor_loss(self, density, dense_masks):
        losses = []
        for row_density, row_mask in zip(density, dense_masks):
            valid_density = row_density[row_mask]
            if valid_density.numel() <= 1:
                continue
            entropy = -(valid_density * (valid_density + 1e-6).log()).sum()
            entropy = entropy / torch.log(valid_density.new_tensor(float(valid_density.numel()))).clamp_min(1e-6)
            losses.append(F.relu(valid_density.new_tensor(self.density_entropy_floor) - entropy))
        if not losses:
            return density.sum() * 0.0
        return torch.stack(losses).mean()

    def _density_window_mass_gap_loss(self, density, dense_masks, return_parts=False):
        mass_losses = []
        gap_losses = []
        for row_density, row_mask in zip(density, dense_masks):
            valid_density = row_density[row_mask]
            valid_len = int(valid_density.numel())
            if valid_len <= 1:
                continue
            valid_density = valid_density / valid_density.sum().clamp_min(1e-6)
            ideal_gap = max(float(valid_len) / float(max(self.target_len, 1)), 1.0)
            window = max(1, min(valid_len, int(math.ceil(ideal_gap))))
            window_masses = []
            for start in range(0, valid_len - window + 1):
                window_masses.append(valid_density[start : start + window].sum())
            if not window_masses:
                continue
            window_masses = torch.stack(window_masses)
            uniform_window_mass = float(window) / float(valid_len)
            mass_floor = valid_density.new_tensor(uniform_window_mass * 0.5)
            mass_losses.append(F.relu(mass_floor - window_masses).mean())
            gap_temperature = valid_density.new_tensor(float(max(self.target_len, 1)))
            gap_losses.append(torch.exp(-window_masses * gap_temperature).mean())
        if not mass_losses:
            zero = density.sum() * 0.0
            return (zero, zero) if return_parts else zero
        mass_loss = torch.stack(mass_losses).mean()
        gap_loss = torch.stack(gap_losses).mean()
        if return_parts:
            return mass_loss, gap_loss
        return mass_loss + gap_loss

    def _density_blue_noise_repulsion_loss(self, density, dense_masks):
        losses = []
        for row_density, row_mask in zip(density, dense_masks):
            valid_density = row_density[row_mask]
            valid_len = int(valid_density.numel())
            if valid_len <= 1:
                continue
            valid_density = valid_density / valid_density.sum().clamp_min(1e-6)
            positions = torch.linspace(0.0, 1.0, steps=valid_len, device=density.device, dtype=density.dtype)
            pair_distance = (positions[:, None] - positions[None, :]).abs()
            eye = torch.eye(valid_len, device=density.device, dtype=torch.bool)
            min_spacing = 1.0 / float(max(self.target_len, 1))
            kernel = torch.exp(-pair_distance / max(min_spacing, 1e-6)).masked_fill(eye, 0.0)
            pair_mass = valid_density[:, None] * valid_density[None, :]
            losses.append((pair_mass * kernel).sum())
        if not losses:
            return density.sum() * 0.0
        return torch.stack(losses).mean()

    def _build_density_weak_targets(self, density, dense_masks, gt_segments):
        targets = torch.zeros_like(density)
        for batch_idx, segments in enumerate(gt_segments):
            row_mask = dense_masks[batch_idx]
            if segments.numel() > 0:
                for seg in segments.to(device=density.device, dtype=torch.float32):
                    start = int(torch.floor(seg[0]).clamp(0, density.shape[1] - 1).item())
                    end = int(torch.ceil(seg[1]).clamp(0, density.shape[1]).item())
                    if end > start:
                        targets[batch_idx, start:end] = 1.0
            if targets[batch_idx][row_mask].sum().item() <= 0:
                targets[batch_idx] = row_mask.to(dtype=density.dtype)
        targets = targets * dense_masks.to(dtype=targets.dtype)
        return targets / targets.sum(dim=1, keepdim=True).clamp_min(1e-6)

    def _selected_repulsion_loss(self, selected, dense_masks):
        losses = []
        for row_selected, row_mask in zip(selected, dense_masks):
            valid_selected = row_selected[self._selected_masks(row_mask[None], row_selected[None])[0]]
            valid_len = row_mask.sum().to(dtype=torch.float32).clamp_min(1.0)
            if valid_selected.numel() <= 1:
                continue
            gaps = (valid_selected[1:] - valid_selected[:-1]).to(dtype=torch.float32) / valid_len
            floor = 1.0 / float(max(valid_selected.numel(), 1))
            losses.append(F.relu(gaps.new_tensor(floor * 0.25) - gaps).mean())
        if not losses:
            return selected.float().sum() * 0.0
        return torch.stack(losses).mean()

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
            selected_valid_len = int(valid_selected.numel())
            starts = starts.clamp(0, selected_valid_len - 1).to(dtype=segments.dtype)
            ends = ends.clamp(0, selected_valid_len).to(dtype=segments.dtype)
            new_segments = torch.stack([starts, ends], dim=1)
            keep = new_segments[:, 1] > new_segments[:, 0]
            remapped_segments.append(new_segments[keep])
            remapped_labels.append(labels[keep])
        return remapped_segments, remapped_labels

    def _attach_train_gt_diagnostics(self, metas, gt_segments, selected_gt_segments):
        for meta, before_segments, after_segments in zip(metas, gt_segments, selected_gt_segments):
            if before_segments.numel() == 0:
                meta["c3_train_gt_remap_length_ratio_mean"] = None
                meta["c3_train_gt_remap_kept_count"] = 0
                continue
            before_lengths = (before_segments[:, 1] - before_segments[:, 0]).clamp_min(1e-6)
            after_lengths = after_segments[:, 1] - after_segments[:, 0] if after_segments.numel() > 0 else before_lengths.new_zeros((0,))
            ratio = float(after_lengths.sum().item() / before_lengths.sum().item())
            meta["c3_train_gt_remap_length_ratio_mean"] = ratio
            meta["c3_train_gt_remap_kept_count"] = int(after_segments.shape[0]) if after_segments.ndim > 1 else 0

    def _average_stride_restoration_error(self, valid_selected, valid_len):
        if valid_selected.numel() <= 1:
            return 0.0
        average_axis = torch.linspace(
            0,
            max(valid_len - 1, 0),
            steps=int(valid_selected.numel()),
            device=valid_selected.device,
            dtype=torch.float32,
        )
        real_axis = valid_selected.to(dtype=torch.float32)
        return float((real_axis - average_axis).abs().mean().item())

    def _remap_metas(self, metas, selected, dense_masks, selection_diagnostics=None):
        new_metas = []
        for batch_idx, meta in enumerate(metas):
            new_meta = copy.deepcopy(meta)
            selected_list = selected[batch_idx].detach().cpu().tolist()
            valid_len = int(dense_masks[batch_idx].sum().item())
            selected_mask = self._selected_masks(dense_masks[batch_idx : batch_idx + 1], selected[batch_idx : batch_idx + 1])[0]
            selected_valid_len = int(selected_mask.sum().item())
            valid_selected = selected[batch_idx][selected_mask]
            old_stride = float(new_meta.get("snippet_stride", 1.0))
            if selected_valid_len > 0:
                new_meta["snippet_stride"] = old_stride * max(valid_len, 1) / float(selected_valid_len)
            new_meta["window_size"] = self.target_len
            new_meta["c3_indirect_fixed_axis"] = True
            new_meta["c3_indirect_irregular_axis"] = False
            new_meta["c3_indirect_original_dense_window_size"] = self.dense_window_size
            new_meta["c3_indirect_selected_dense_indices"] = selected_list
            new_meta["c3_indirect_selected_mask"] = [bool(x) for x in selected_mask.detach().cpu().tolist()]
            new_meta["c3_indirect_valid_len"] = valid_len
            new_meta["c3_indirect_selected_valid_len"] = selected_valid_len
            new_meta["c3_indirect_strategy"] = self.strategy
            if self.strategy == "cadf_density_mesh_st":
                new_meta["c3_density_mesh_alpha"] = float(self._last_density_alpha)
                new_meta["c3_density_mesh_alpha_schedule_enabled"] = self.density_alpha_schedule is not None
                new_meta["c3_density_mesh_alpha_schedule_scope"] = (
                    "diagnostic_short_smoke_not_resumable" if self.density_alpha_schedule is not None else "static"
                )
                new_meta["c3_density_mesh_alpha_schedule_recoverable"] = False if self.density_alpha_schedule is not None else None
                new_meta["c3_density_mesh_test_alpha_policy"] = self._last_density_alpha_policy
                new_meta["c3_density_mesh_temperature"] = self.density_temperature
                new_meta["c3_density_mesh_weights"] = copy.deepcopy(self.density_weights)
                new_meta["c3_density_mesh_action_support_hook"] = "low_res_scout_action_floor_only"
                new_meta["c3_density_mesh_transition_support_hook"] = "low_res_scout_abs_delta_action"
                new_meta["c3_density_mesh_boundary_support_hook"] = "disabled" if self.density_weights.get("boundary", 0.0) <= 0 else "low_res_scout_boundary_head"
                new_meta["c3_density_mesh_nonuniform_selection"] = True
                new_meta["c3_backend_uses_average_stride"] = True
                new_meta["c3_physical_coords_unused_by_backend"] = True
                new_meta["c3_selected_index_aware_postprocess_enabled"] = self.selected_index_aware_postprocess_enabled
                new_meta["c3_physical_time_postprocess_enabled"] = self.physical_time_postprocess_enabled
                new_meta["c3_diagnostic_prediction_cap"] = self.diagnostic_prediction_cap
                new_meta["c3_diagnostic_proposal_cap"] = self.diagnostic_proposal_cap
                new_meta["c3_density_mesh_average_stride_restoration_error"] = self._average_stride_restoration_error(valid_selected, valid_len)
                if selection_diagnostics is not None:
                    diag = selection_diagnostics[batch_idx]
                    new_meta["c3_density_mesh_max_gap"] = diag["max_gap"]
                    new_meta["c3_density_mesh_mean_gap"] = diag["mean_gap"]
                    new_meta["c3_density_mesh_repair_fraction"] = diag["repair_fraction"]
                    new_meta["c3_density_mesh_row_repair_fraction"] = diag.get("row_repair_fraction", diag["repair_fraction"])
                    new_meta["c3_density_mesh_repair_count"] = diag["repair_count"]
                    new_meta["c3_density_mesh_dedupe_repair_count"] = diag.get("dedupe_repair_count", 0)
                    new_meta["c3_density_mesh_gap_guard_add_count"] = diag.get("gap_guard_add_count", 0)
                    new_meta["c3_density_mesh_gap_guard_prune_count"] = diag.get("gap_guard_prune_count", 0)
                    new_meta["c3_density_mesh_pad_repair_count"] = diag.get("pad_repair_count", 0)
                    if "density_entropy" in diag:
                        new_meta["c3_density_mesh_entropy"] = diag["density_entropy"]
                        if "selected_density_sum" in diag:
                            new_meta["c3_density_mesh_selected_density_sum"] = diag["selected_density_sum"]
                        new_meta["c3_density_mesh_density_top_positions"] = diag.get("density_top_positions", [])
            returnable = new_meta
            new_metas.append(returnable)
        return new_metas
