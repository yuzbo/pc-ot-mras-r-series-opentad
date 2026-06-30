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
