import numpy as np

from .probe_builder import build_scaffold_positions
from .sparse_gather import sparse_gather
from .types import METHOD_NAME, RbaRbrBudgetConfig, RbaRbrSelectionResult, ROUTE_LABEL, sorted_unique_positions
from .validators import (
    LOCAL_PRECHECK_CLAIM_STATUS,
    build_original_time_metadata,
    build_selection_gap_diagnostics,
    validate_rba_rbr_ledger,
)


class DynamicBudgetController:
    """Regret-based recoverable bracketing controller.

    The controller treats brackets as priors. It first keeps a small scaffold,
    then spends dynamic budget on high-regret in-bracket refine probes and
    out-of-bracket rescue probes driven only by deploy-visible risk signals.
    """

    def __init__(self, config: RbaRbrBudgetConfig):
        self.config = config

    def _dynamic_target_k(self, risk_map):
        risk = np.asarray(risk_map.rescue_score, dtype=np.float64)
        peak = float(np.max(risk))
        mean = float(np.mean(risk))
        extra = int(round((0.40 * peak + 0.60 * mean) * max(self.config.max_k - self.config.min_k, 0)))
        return int(np.clip(self.config.min_k + extra, self.config.min_k, self.config.max_k))

    def _detector_feature_valid_k(self, raw_k):
        stride = int(max(self.config.feature_stride, 1))
        return int(np.ceil(float(max(int(raw_k), 0)) / float(stride)))

    def _detector_centers(self, selected):
        positions = np.asarray(selected, dtype=np.float64).reshape(-1)
        if positions.size == 0:
            return np.zeros((0,), dtype=np.float64)
        stride = int(max(self.config.feature_stride, 1))
        centers = []
        for start in range(0, int(positions.size), stride):
            centers.append(float(np.mean(positions[start : start + stride])))
        return np.asarray(centers, dtype=np.float64)

    @staticmethod
    def _max_raw_gap(selected, dense_T):
        positions = sorted_unique_positions(selected, dense_T)
        if not positions:
            return int(dense_T)
        gaps = [positions[0] + 1]
        gaps.extend(int(right - left) for left, right in zip(positions, positions[1:]))
        gaps.append(int(dense_T - positions[-1]))
        return int(max(gaps))

    def _max_detector_gap(self, selected, dense_T):
        centers = self._detector_centers(selected)
        if centers.size == 0:
            return float(dense_T)
        gaps = [float(centers[0] + 1.0)]
        gaps.extend(float(right - left) for left, right in zip(centers, centers[1:]))
        gaps.append(float(dense_T - centers[-1]))
        return float(max(gaps))

    @staticmethod
    def _coverage_risk_score(risk_map, probes):
        score = (
            0.36 * np.asarray(risk_map.rescue_score, dtype=np.float64)
            + 0.24 * np.asarray(risk_map.staleness, dtype=np.float64)
            + 0.18 * np.asarray(risk_map.gap_risk, dtype=np.float64)
            + 0.14 * np.asarray(risk_map.uncertainty, dtype=np.float64)
            + 0.08 * np.asarray(risk_map.transition, dtype=np.float64)
        )
        if score.size == 0:
            return score
        probe_bonus = np.zeros_like(score)
        max_regret = max([float(probe.predicted_regret) for probe in probes] + [1.0])
        for probe in probes:
            for pos in probe.positions:
                if 0 <= int(pos) < probe_bonus.size:
                    role_bonus = 0.16 if probe.stage == "rescue" else 0.08
                    outside_bonus = 0.06 if bool(probe.outside_hard_bracket) else 0.0
                    probe_bonus[int(pos)] = max(
                        probe_bonus[int(pos)],
                        role_bonus + outside_bonus + 0.12 * float(probe.predicted_regret) / max_regret,
                    )
        return np.clip(score + probe_bonus, 0.0, None)

    @staticmethod
    def _largest_raw_gap_interval(selected, dense_T):
        positions = sorted_unique_positions(selected, dense_T)
        if not positions:
            return int(dense_T), 0, int(dense_T) - 1
        intervals = [(positions[0] + 1, 0, positions[0] - 1)]
        for left, right in zip(positions, positions[1:]):
            intervals.append((int(right - left), int(left) + 1, int(right) - 1))
        intervals.append((int(dense_T - positions[-1]), int(positions[-1]) + 1, int(dense_T) - 1))
        intervals = [row for row in intervals if row[1] <= row[2]]
        if not intervals:
            return 0, 0, -1
        return max(intervals, key=lambda row: (row[0], row[2] - row[1]))

    def _largest_detector_gap_interval(self, selected, dense_T):
        positions = sorted_unique_positions(selected, dense_T)
        centers = self._detector_centers(positions)
        if centers.size == 0:
            return int(dense_T), 0, int(dense_T) - 1
        stride = int(max(self.config.feature_stride, 1))
        intervals = []
        first_hi = positions[0] - 1 if positions else int(dense_T) - 1
        intervals.append((float(centers[0] + 1.0), 0, int(first_hi)))
        for idx in range(int(centers.size) - 1):
            boundary = min((idx + 1) * stride, len(positions) - 1)
            left = int(positions[max(boundary - 1, 0)]) + 1
            right = int(positions[min(boundary, len(positions) - 1)]) - 1
            intervals.append((float(centers[idx + 1] - centers[idx]), left, right))
        intervals.append((float(dense_T - centers[-1]), int(positions[-1]) + 1, int(dense_T) - 1))
        intervals = [row for row in intervals if row[1] <= row[2]]
        if not intervals:
            return 0.0, 0, -1
        return max(intervals, key=lambda row: (row[0], row[2] - row[1]))

    @staticmethod
    def _best_guard_position(score, selected, dense_T, lo=None, hi=None, center_fraction=None):
        selected_set = set(int(pos) for pos in selected)
        lo = 0 if lo is None else int(max(lo, 0))
        hi = int(dense_T) - 1 if hi is None else int(min(hi, int(dense_T) - 1))
        if lo > hi:
            return None
        candidates = [pos for pos in range(lo, hi + 1) if pos not in selected_set]
        if not candidates:
            return None
        center = 0.5 * (lo + hi)
        if center_fraction is not None:
            half_width = max(1.0, 0.5 * float(hi - lo + 1) * float(center_fraction))
            center_candidates = [pos for pos in candidates if abs(float(pos) - center) <= half_width]
            if center_candidates:
                candidates = center_candidates
        return max(candidates, key=lambda pos: (float(score[pos]), -abs(float(pos) - center), -pos))

    def _coverage_guard(self, selected, risk_map, probes, rows, initial_stop_reason):
        dense_T = int(risk_map.dense_T)
        selected = sorted_unique_positions(selected, dense_T)
        score = self._coverage_risk_score(risk_map, probes)
        max_k = int(self.config.max_k)
        min_raw_target = int(self.config.min_k)
        detector_target = None
        if self.config.min_detector_feature_k is not None:
            detector_target = int(self.config.min_detector_feature_k)
            raw_for_detector = max(1, (detector_target - 1) * int(max(self.config.feature_stride, 1)) + 1)
            min_raw_target = max(min_raw_target, min(raw_for_detector, max_k))

        before = {
            "valid_k": int(len(selected)),
            "detector_feature_valid_k": int(self._detector_feature_valid_k(len(selected))),
            "max_raw_gap": int(self._max_raw_gap(selected, dense_T)),
            "max_detector_gap": float(self._max_detector_gap(selected, dense_T)),
        }
        guard_additions = []

        def add_position(pos, reason, gap_before=None):
            if pos is None or len(selected) >= max_k or int(pos) in set(selected):
                return False
            before_positions = list(selected)
            selected.append(int(pos))
            selected[:] = sorted_unique_positions(selected, dense_T)
            addition = {
                "position": int(pos),
                "guard_reason": reason,
                "score": float(score[int(pos)]),
                "raw_valid_k_after": int(len(selected)),
                "detector_feature_valid_k_after": int(self._detector_feature_valid_k(len(selected))),
            }
            if gap_before is not None:
                addition["gap_before"] = float(gap_before)
            guard_additions.append(addition)
            rows.append(
                {
                    "route_label": ROUTE_LABEL,
                    "method": METHOD_NAME,
                    "probe_id": None,
                    "probe_stage": "coverage_guard",
                    "probe_role": reason,
                    "center_pos": int(pos),
                    "selected_before_positions": [int(value) for value in before_positions],
                    "selected_after_positions": [int(value) for value in selected],
                    "selected_decision": "selected_coverage_guard_backfill",
                    "predicted_regret": float(score[int(pos)]),
                    "outside_hard_bracket": None,
                    "source_signal": "recoverability_regret_staleness_rescue_guard",
                }
            )
            return True

        while len(selected) < min_raw_target and len(selected) < max_k:
            pos = self._best_guard_position(score, selected, dense_T)
            if not add_position(pos, "min_raw_or_detector_feature_keep"):
                break

        if self.config.max_raw_gap is not None:
            max_raw_gap = int(self.config.max_raw_gap)
            while len(selected) < max_k:
                gap, lo, hi = self._largest_raw_gap_interval(selected, dense_T)
                if int(gap) <= max_raw_gap:
                    break
                pos = self._best_guard_position(score, selected, dense_T, lo=lo, hi=hi, center_fraction=0.50)
                if pos is None:
                    pos = int(round(0.5 * (lo + hi))) if lo <= hi else None
                if not add_position(pos, "max_raw_gap"):
                    break

        if detector_target is not None:
            while self._detector_feature_valid_k(len(selected)) < detector_target and len(selected) < max_k:
                pos = self._best_guard_position(score, selected, dense_T)
                if not add_position(pos, "min_detector_feature_keep"):
                    break

        if self.config.max_detector_gap is not None:
            max_detector_gap = float(self.config.max_detector_gap)
            while len(selected) < max_k:
                gap, lo, hi = self._largest_detector_gap_interval(selected, dense_T)
                if float(gap) <= max_detector_gap:
                    break
                pos = self._best_guard_position(score, selected, dense_T, lo=lo, hi=hi, center_fraction=0.50)
                if pos is None:
                    pos = int(round(0.5 * (lo + hi))) if lo <= hi else None
                if not add_position(pos, "max_detector_gap", gap_before=gap):
                    break

        after = {
            "valid_k": int(len(selected)),
            "detector_feature_valid_k": int(self._detector_feature_valid_k(len(selected))),
            "max_raw_gap": int(self._max_raw_gap(selected, dense_T)),
            "max_detector_gap": float(self._max_detector_gap(selected, dense_T)),
        }
        unmet = []
        if after["valid_k"] < int(self.config.min_k):
            unmet.append("min_raw_keep_unmet_budget_or_candidate_exhausted")
        if detector_target is not None and after["detector_feature_valid_k"] < detector_target:
            unmet.append("min_detector_feature_keep_unmet_budget_or_candidate_exhausted")
        if self.config.max_raw_gap is not None and after["max_raw_gap"] > int(self.config.max_raw_gap):
            unmet.append("max_raw_gap_unmet_budget_or_dense_limit")
        if self.config.max_detector_gap is not None and after["max_detector_gap"] > float(self.config.max_detector_gap):
            unmet.append("max_detector_gap_unmet_budget_or_dense_limit")

        if guard_additions:
            guard_reason = "+".join(sorted({row["guard_reason"] for row in guard_additions}))
            stop_reason = "coverage_guard"
        elif unmet:
            guard_reason = "+".join(unmet)
            stop_reason = initial_stop_reason
        else:
            guard_reason = "not_needed"
            stop_reason = initial_stop_reason
        return selected, stop_reason, {
            "coverage_guard_enabled": True,
            "pre_guard_budget_stop_reason": initial_stop_reason,
            "guard_reason": guard_reason,
            "guard_unmet_constraints": unmet,
            "guard_additions": guard_additions,
            "guard_addition_count": int(len(guard_additions)),
            "pre_guard_valid_k": before["valid_k"],
            "post_guard_valid_k": after["valid_k"],
            "detector_feature_target_k": None if detector_target is None else int(detector_target),
            "pre_guard_detector_feature_valid_k": before["detector_feature_valid_k"],
            "post_guard_detector_feature_valid_k": after["detector_feature_valid_k"],
            "max_raw_gap_before_guard": before["max_raw_gap"],
            "max_raw_gap_after_guard": after["max_raw_gap"],
            "max_detector_gap_before_guard": before["max_detector_gap"],
            "max_detector_gap_after_guard": after["max_detector_gap"],
            "max_raw_gap_target": None if self.config.max_raw_gap is None else int(self.config.max_raw_gap),
            "max_detector_gap_target": None
            if self.config.max_detector_gap is None
            else int(self.config.max_detector_gap),
        }

    def select(
        self,
        risk_map,
        brackets,
        probes,
        scaffold_positions=None,
        video_id="synthetic",
        window_id=0,
        split="synthetic",
        fps=30.0,
        dense_inputs=None,
    ):
        dense_T = int(risk_map.dense_T)
        target_k = self._dynamic_target_k(risk_map)
        scaffold_positions = build_scaffold_positions(dense_T, self.config.scaffold_k) if scaffold_positions is None else scaffold_positions
        selected = []
        selected_probes = []
        rows = []

        def add_probe(probe, decision):
            before = list(selected)
            candidate_positions = sorted_unique_positions(before + list(probe.positions), dense_T)
            if len(candidate_positions) > int(self.config.max_k):
                rows.append(self._row(probe, before, before, "skipped_budget_cap"))
                return False
            if candidate_positions == before:
                rows.append(self._row(probe, before, before, "skipped_duplicate"))
                return False
            selected[:] = candidate_positions
            selected_probes.append(probe)
            rows.append(self._row(probe, before, selected, decision))
            return True

        scaffold_set = set(int(pos) for pos in sorted_unique_positions(scaffold_positions, dense_T))
        scaffold_probes = [probe for probe in probes if any(pos in scaffold_set for pos in probe.positions)]
        for pos in sorted(scaffold_set):
            match = next((probe for probe in scaffold_probes if pos in probe.positions), None)
            if match is None:
                continue
            add_probe(match, "selected_scaffold")
            if len(selected) >= min(self.config.min_k, target_k):
                break

        ranked = sorted(
            [probe for probe in probes if probe not in selected_probes],
            key=lambda probe: (
                probe.stage != "rescue",
                -float(probe.predicted_regret),
                not bool(probe.outside_hard_bracket),
                probe.rank,
                probe.center_pos,
            ),
        )
        rescue_selected = 0
        rescue_quota = max(1, int(round(float(self.config.rescue_quota_fraction) * target_k)))
        stop_reason = None
        for probe in ranked:
            if len(selected) >= target_k:
                break
            if float(probe.predicted_regret) < float(self.config.min_marginal_regret) and len(selected) >= self.config.min_k:
                stop_reason = "regret_saturation"
                rows.append(self._row(probe, selected, selected, "skipped_low_regret"))
                break
            if probe.stage == "rescue" and rescue_selected >= rescue_quota and len(selected) >= self.config.min_k:
                continue
            changed = add_probe(probe, "selected_rescue" if probe.stage == "rescue" else "selected_refine")
            if changed and probe.stage == "rescue":
                rescue_selected += 1

        if len(selected) < self.config.min_k:
            for probe in ranked:
                if len(selected) >= self.config.min_k:
                    break
                add_probe(probe, "selected_min_k_backfill")

        if stop_reason is None:
            if len(selected) >= self.config.max_k:
                stop_reason = "budget_cap"
            elif len(selected) >= target_k:
                stop_reason = "risk_satisfied"
            elif not ranked:
                stop_reason = "candidate_exhausted"
            else:
                stop_reason = "candidate_exhausted"
        selected[:] = sorted_unique_positions(selected, dense_T)
        selected, stop_reason, guard_ledger = self._coverage_guard(
            selected,
            risk_map,
            probes,
            rows,
            initial_stop_reason=stop_reason,
        )
        if dense_inputs is None:
            dense_inputs = np.arange(dense_T, dtype=np.float64).reshape(dense_T, 1)
        _, gather_evidence = sparse_gather(dense_inputs, selected, temporal_dim=0)
        original_time = build_original_time_metadata(dense_T, selected, fps=fps)
        rescue_outside = sum(1 for probe in selected_probes if probe.stage == "rescue" and probe.outside_hard_bracket)
        ledger = {
            "route_label": ROUTE_LABEL,
            "method": METHOD_NAME,
            "selector_method": "regret_based_recoverable_bracketing",
            "split": split,
            "video_id": video_id,
            "window_id": int(window_id),
            "dense_T": dense_T,
            "selected_positions": [int(pos) for pos in selected],
            "valid_k": int(len(selected)),
            "dynamic_target_k": int(target_k),
            "min_k": int(self.config.min_k),
            "max_k": int(self.config.max_k),
            "scaffold_k": int(len(scaffold_set)),
            "budget_stop_reason": stop_reason,
            "selection_gap_diagnostics": build_selection_gap_diagnostics(selected, dense_T),
            "selected_probe_ids": [int(probe.probe_id) for probe in selected_probes],
            "selected_probe_stages": [probe.stage for probe in selected_probes],
            "rescue_outside_hard_bracket_count": int(rescue_outside),
            "soft_bracket_count": int(len(brackets)),
            "hard_bracket_count": int(sum(1 for bracket in brackets if bracket.hard_bracket)),
            "soft_bracket_is_prior_not_mask": True,
            "original_time_metadata": original_time,
            "real_sparse_evidence": gather_evidence,
            "claim_status": LOCAL_PRECHECK_CLAIM_STATUS,
            "no_metric_claim": True,
            "no_runtime_claim": True,
            "no_deploy_claim": True,
            "no_paper_claim": True,
            "full_train_unlocked": False,
            "value_labels_used_at_test": False,
            "selector_provenance": {
                "selection_uses_gt": False,
                "selection_uses_teacher": False,
                "selection_uses_prediction_cache": False,
                "selection_uses_raw_detector_prediction": False,
                "selection_uses_oracle_boundary": False,
                "selection_uses_oracle_residual": False,
            },
        }
        ledger.update(guard_ledger)
        validate_rba_rbr_ledger(ledger)
        return RbaRbrSelectionResult(
            selected_positions=list(selected),
            selected_probes=list(selected_probes),
            selection_rows=rows,
            deploy_ledger=ledger,
            stop_reason=stop_reason,
        )

    def _row(self, probe, before, after, decision):
        return {
            "route_label": ROUTE_LABEL,
            "method": METHOD_NAME,
            "probe_id": int(probe.probe_id),
            "probe_stage": probe.stage,
            "probe_role": probe.role,
            "center_pos": int(probe.center_pos),
            "selected_before_positions": [int(pos) for pos in before],
            "selected_after_positions": [int(pos) for pos in after],
            "selected_decision": decision,
            "predicted_regret": float(probe.predicted_regret),
            "outside_hard_bracket": bool(probe.outside_hard_bracket),
            "source_signal": probe.source_signal,
        }
