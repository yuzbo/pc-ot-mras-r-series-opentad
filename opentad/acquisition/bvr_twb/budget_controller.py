from collections import Counter, defaultdict

import numpy as np

from .boundary_belief import (
    required_witness_roles_for_bracket,
    summarize_belief_trace,
    update_boundary_belief_trace,
)
from .scaffold import gap_statistics, repair_positions_for_max_gap
from .sparse_gather import sparse_gather
from .types import BudgetConfig, CandidatePacket, METHOD_NAME, ROUTE_LABEL, SelectionResult, STOP_REASONS, sorted_unique_positions
from .validators import build_original_time_metadata, build_selection_gap_diagnostics, validate_deploy_ledger


class DynamicBudgetController:
    """Value-of-information boundary-belief controller.

    Scaffold and gap anchors are candidates or fail-closed constraints, not a
    mandatory learned contribution. The controller greedily follows marginal
    posterior risk reduction, then stops when posterior belief/gap/regret
    constraints are safe or the candidate/budget frontier is exhausted.
    """

    def __init__(self, config: BudgetConfig):
        self.config = config

    def select(
        self,
        scaffold_packets,
        candidate_packets,
        brackets,
        dense_T,
        fps=30.0,
        video_id="synthetic",
        window_id=0,
        split="synthetic",
        dense_inputs=None,
    ):
        dense_T = int(dense_T)
        scaffold_packets = list(scaffold_packets)
        candidate_packets = list(candidate_packets)
        all_candidates = self._rank_candidates(scaffold_packets + candidate_packets)
        selected_packets = []
        selected_positions = []
        rows = []
        trace_flags = {
            "regret_saturation_seen": False,
            "mandatory_scaffold_first": False,
            "scaffold_candidates_offered_count": int(len(scaffold_packets)),
            "safety_floor_selected_count": 0,
        }

        def add_packet(packet, decision, components=None):
            before_positions = list(selected_positions)
            before_packets = list(selected_packets)
            after_candidate = sorted_unique_positions(before_positions + list(packet.positions), dense_T)
            if len(after_candidate) > self.config.max_k:
                rows.append(
                    self._row(
                        packet,
                        before_positions,
                        before_positions,
                        decision="skipped_budget_cap",
                        dense_T=dense_T,
                        value_components=components,
                        belief_state_before=self._belief_frontier_state(brackets, before_packets),
                        belief_state_after=self._belief_frontier_state(brackets, before_packets),
                    )
                )
                return False
            changed = False
            for pos in packet.positions:
                if int(pos) not in selected_positions:
                    selected_positions.append(int(pos))
                    changed = True
            selected_positions[:] = sorted_unique_positions(selected_positions, dense_T)
            if changed and packet not in selected_packets:
                selected_packets.append(packet)
                if packet.safety_floor:
                    trace_flags["safety_floor_selected_count"] += 1
            after_packets = list(selected_packets)
            rows.append(
                self._row(
                    packet,
                    before_positions,
                    selected_positions,
                    decision=decision,
                    dense_T=dense_T,
                    value_components=components,
                    belief_state_before=self._belief_frontier_state(brackets, before_packets),
                    belief_state_after=self._belief_frontier_state(brackets, after_packets),
                )
            )
            return changed

        if self.config.low_risk_scaffold_mandatory:
            for packet in [p for p in scaffold_packets if p.safety_floor]:
                if len(selected_positions) >= self.config.max_k:
                    break
                add_packet(packet, "selected_safety_floor")

        while len(selected_positions) < min(self.config.min_k, self.config.max_k):
            packet, components = self._best_marginal_packet(all_candidates, selected_positions, selected_packets, brackets, dense_T)
            if packet is None:
                break
            decision = "selected_risk_constraint" if self._posterior_constraints_unsafe(selected_positions, selected_packets, brackets, dense_T) else "selected_min_k"
            if not add_packet(packet, decision, components=components):
                break

        selected_positions, rows = self._repair_max_gap(selected_positions, selected_packets, rows, all_candidates, dense_T)
        if len(selected_positions) >= self.config.max_k:
            post_repair_gap = gap_statistics(selected_positions, dense_T)["max_gap"]
            stop = "gap_guard" if int(post_repair_gap) > int(self.config.max_gap) else "budget_cap"
            return self._finish(
                selected_positions,
                selected_packets,
                rows,
                dense_T,
                fps,
                video_id,
                window_id,
                split,
                brackets,
                stop,
                dense_inputs,
                trace_flags=trace_flags,
            )

        self._satisfy_role_coverage(brackets, all_candidates, selected_positions, selected_packets, rows, add_packet)
        belief_trace = update_boundary_belief_trace(brackets, selected_packets, safe_width=self.config.safe_belief_width)
        if len(selected_positions) >= self.config.min_k and self._all_active_beliefs_safe(brackets, belief_trace):
            return self._finish(
                selected_positions,
                selected_packets,
                rows,
                dense_T,
                fps,
                video_id,
                window_id,
                split,
                brackets,
                "belief_width_safe",
                dense_inputs,
                belief_trace=belief_trace,
                trace_flags=trace_flags,
            )
        if self._risk_constraints_satisfied(selected_positions, selected_packets, brackets, dense_T, belief_trace):
            return self._finish(
                selected_positions,
                selected_packets,
                rows,
                dense_T,
                fps,
                video_id,
                window_id,
                split,
                brackets,
                "risk_constraints_satisfied",
                dense_inputs,
                belief_trace=belief_trace,
                trace_flags=trace_flags,
            )

        stop_reason = None
        while len(selected_positions) < self.config.max_k:
            packet, components = self._best_marginal_packet(all_candidates, selected_positions, selected_packets, brackets, dense_T)
            if packet is None:
                break
            value = self._component_score(components)
            if value < float(self.config.min_marginal_value):
                stop_reason = "regret_saturation"
                trace_flags["regret_saturation_seen"] = True
                rows.append(
                    self._row(
                        packet,
                        selected_positions,
                        selected_positions,
                        decision="skipped_low_value",
                        dense_T=dense_T,
                        value_components=components,
                        belief_state_before=self._belief_frontier_state(brackets, selected_packets),
                        belief_state_after=self._belief_frontier_state(brackets, selected_packets),
                    )
                )
                break
            before = list(selected_positions)
            changed = add_packet(packet, "selected_marginal_voi", components=components)
            if changed:
                belief_trace = update_boundary_belief_trace(brackets, selected_packets, safe_width=self.config.safe_belief_width)
                selected_positions, rows = self._repair_max_gap(selected_positions, selected_packets, rows, all_candidates, dense_T)
                if len(selected_positions) >= self.config.min_k and self._all_active_beliefs_safe(brackets, belief_trace):
                    stop_reason = "belief_width_safe"
                    break
                if self._risk_constraints_satisfied(selected_positions, selected_packets, brackets, dense_T, belief_trace):
                    stop_reason = "risk_constraints_satisfied"
                    break
            if before == selected_positions:
                break

        belief_trace = update_boundary_belief_trace(brackets, selected_packets, safe_width=self.config.safe_belief_width)
        if stop_reason is None:
            stop_reason = self._infer_stop_reason(selected_positions, brackets, all_candidates, dense_T, belief_trace=belief_trace)
        if stop_reason not in STOP_REASONS:
            stop_reason = "candidate_exhausted"
        return self._finish(
            selected_positions,
            selected_packets,
            rows,
            dense_T,
            fps,
            video_id,
            window_id,
            split,
            brackets,
            stop_reason,
            dense_inputs,
            belief_trace=belief_trace,
            trace_flags=trace_flags,
        )

    def _rank_candidates(self, packets):
        return sorted(
            packets,
            key=lambda packet: (
                -float(packet.predicted_value.value_per_cost if packet.predicted_value else 0.0),
                -float(packet.feature_summary.get("gap_risk", 0.0)),
                -int(packet.required_for_role_coverage),
                packet.role,
                packet.positions,
                packet.packet_id,
            ),
        )

    def _repair_max_gap(self, selected_positions, selected_packets, rows, candidates, dense_T):
        repaired, bridges = repair_positions_for_max_gap(selected_positions, dense_T, self.config.max_gap)
        if len(repaired) > self.config.max_k:
            raise ValueError(
                "hard max-gap contract infeasible under budget: "
                f"dense_T={int(dense_T)} max_gap={int(self.config.max_gap)} "
                f"max_k={int(self.config.max_k)} required_k={len(repaired)}"
            )
        for bridge in bridges:
            packet = next((candidate for candidate in candidates if candidate.role == "gap_bridge" and bridge in candidate.positions), None)
            if packet is None:
                packet = CandidatePacket(
                    packet_id=900000 + int(bridge),
                    video_id=selected_packets[0].video_id if selected_packets else "synthetic",
                    window_id=selected_packets[0].window_id if selected_packets else 0,
                    split=selected_packets[0].split if selected_packets else "synthetic",
                    source="gap_repair",
                    role="gap_bridge",
                    positions=[int(bridge)],
                    dense_T=int(dense_T),
                    reason="controller_gap_guard_constraint_repair",
                    max_gap_repair=True,
                    feature_summary={"gap_risk": 1.0, "gap_if_omitted_frames": float(self.config.max_gap + 1)},
                )
            before = list(selected_positions)
            before_state = self._belief_frontier_state([], [])
            if int(bridge) not in selected_positions:
                selected_positions.append(int(bridge))
                selected_positions[:] = sorted_unique_positions(selected_positions, dense_T)
            if packet not in selected_packets:
                selected_packets.append(packet)
            components = self._marginal_components(packet, before, selected_packets, [], dense_T)
            rows.append(
                self._row(
                    packet,
                    before,
                    selected_positions,
                    "selected_gap_guard",
                    dense_T,
                    value_components=components,
                    belief_state_before=before_state,
                    belief_state_after=before_state,
                )
            )
        selected_positions[:] = sorted_unique_positions(selected_positions, dense_T)
        return selected_positions, rows

    def _satisfy_role_coverage(self, brackets, candidates, selected_positions, selected_packets, rows, add_packet):
        coverage = defaultdict(set)
        for packet in selected_packets:
            if packet.bracket_id is not None:
                coverage[packet.bracket_id].add(packet.role)
        active_brackets = [bracket for bracket in brackets if bracket.state == "active"]
        active_brackets = sorted(
            active_brackets,
            key=lambda bracket: (
                -(float(bracket.entropy) + float(bracket.peak_transition) + float(bracket.short_action_risk)),
                bracket.bracket_id,
            ),
        )
        for bracket in active_brackets:
            if len(selected_positions) >= self.config.max_k:
                return
            if float(bracket.entropy) < self.config.safe_entropy and float(bracket.gap_risk) < self.config.safe_risk_mass:
                continue
            needed_roles = required_witness_roles_for_bracket(bracket)
            missing = sorted(needed_roles.difference(coverage.get(bracket.bracket_id, set())))
            for role in missing:
                if len(selected_positions) >= self.config.max_k:
                    return
                match = next((packet for packet in candidates if packet.bracket_id == bracket.bracket_id and packet.role == role), None)
                if match is not None:
                    components = self._marginal_components(match, selected_positions, selected_packets, brackets, match.dense_T)
                    add_packet(match, "selected_role_coverage", components=components)
                    coverage[bracket.bracket_id].add(role)
                    if self._all_active_beliefs_safe(
                        brackets,
                        update_boundary_belief_trace(brackets, selected_packets, safe_width=self.config.safe_belief_width),
                    ):
                        return

    def _infer_stop_reason(self, selected_positions, brackets, candidates, dense_T, belief_trace=None):
        if len(selected_positions) >= self.config.max_k:
            return "budget_cap"
        stats = gap_statistics(selected_positions, int(dense_T))
        if stats["max_gap"] > self.config.max_gap:
            return "gap_guard"
        belief_trace = [] if belief_trace is None else belief_trace
        if self._all_active_beliefs_safe(brackets, belief_trace):
            return "belief_width_safe"
        if self._risk_constraints_satisfied(selected_positions, [], brackets, dense_T, belief_trace):
            return "risk_constraints_satisfied"
        if not candidates:
            return "candidate_exhausted"
        return "candidate_exhausted"

    def _row(
        self,
        packet,
        before,
        after,
        decision,
        dense_T,
        value_components=None,
        belief_state_before=None,
        belief_state_after=None,
    ):
        value = packet.predicted_value
        value_components = self._value_components(packet) if value_components is None else value_components
        belief_state_before = self._empty_belief_state() if belief_state_before is None else belief_state_before
        belief_state_after = self._empty_belief_state() if belief_state_after is None else belief_state_after
        score = self._component_score(value_components)
        return {
            "route_label": ROUTE_LABEL,
            "method": METHOD_NAME,
            "voi_bbc_spec": "Value-of-Information Boundary Belief Controller",
            "video_id": packet.video_id,
            "split": packet.split,
            "window_id": int(packet.window_id),
            "dense_T": int(dense_T),
            "packet_id": int(packet.packet_id),
            "bracket_id": packet.bracket_id,
            "packet_role": packet.role,
            "packet_source": packet.source,
            "packet_positions": list(packet.positions),
            "packet_cost_frames": float(packet.cost_frames),
            "safety_floor": bool(packet.safety_floor),
            "selected_before_positions": list(before),
            "selected_after_positions": list(after),
            "selected_decision": decision,
            "selected_decision_subreason": self._decision_subreason(decision),
            "predicted_regret": 0.0 if value is None else float(value.predicted_regret),
            "expected_belief_reduction": 0.0 if value is None else float(value.expected_belief_reduction),
            "value_uncertainty": 0.0 if value is None else float(value.value_uncertainty),
            "value_per_cost": 0.0 if value is None else float(value.value_per_cost),
            "marginal_value_per_cost": float(score / max(float(packet.cost_frames), 1e-6)),
            "marginal_voi_score": float(score),
            "value_components": {key: float(val) for key, val in value_components.items()},
            "belief_state_before": belief_state_before,
            "belief_state_after": belief_state_after,
            "belief_risk_before": float(belief_state_before["max_posterior_risk_mass"]),
            "belief_risk_after": float(belief_state_after["max_posterior_risk_mass"]),
            "constraint_state": self._constraint_state(before, after, packet, dense_T),
            "marginal_value_threshold": float(self.config.min_marginal_value),
            "valid_k_after": int(len(after)),
            "min_k": int(self.config.min_k),
            "max_k": int(self.config.max_k),
            "stop_reason_if_final": "not_final",
            "provenance": self._clean_provenance(),
        }

    def _finish(
        self,
        selected_positions,
        selected_packets,
        rows,
        dense_T,
        fps,
        video_id,
        window_id,
        split,
        brackets,
        stop_reason,
        dense_inputs,
        belief_trace=None,
        trace_flags=None,
    ):
        selected_positions = sorted_unique_positions(selected_positions, dense_T)
        gap_diagnostics = build_selection_gap_diagnostics(selected_positions, dense_T, self.config.max_gap)
        if gap_diagnostics["coverage_violation"]:
            raise ValueError(
                "hard max-gap contract violated before ledger generation: "
                f"max_gap={gap_diagnostics['max_gap']} max_allowed_gap={gap_diagnostics['max_allowed_gap']}"
            )
        for row in rows:
            row["stop_reason_if_final"] = stop_reason
        if dense_inputs is None:
            dense_inputs = np.arange(int(dense_T), dtype=np.float64).reshape(int(dense_T), 1)
        _, gather_evidence = sparse_gather(dense_inputs, selected_positions, temporal_dim=0, detector_forward_exists=False)
        metadata = build_original_time_metadata(
            dense_T=dense_T,
            selected_positions=selected_positions,
            fps=fps,
            window_start_sec=0.0,
            window_end_sec=float(dense_T) / float(fps),
        )
        role_counts = Counter(packet.role for packet in selected_packets)
        active_brackets = [bracket for bracket in brackets if bracket.state == "active"]
        belief_trace = update_boundary_belief_trace(brackets, selected_packets, safe_width=self.config.safe_belief_width) if belief_trace is None else belief_trace
        active_belief_trace = self._active_belief_trace(brackets, belief_trace)
        posterior_widths = [float(row["posterior_width_p80_frames"]) for row in belief_trace]
        initial_widths = [float(row["initial_width_p80_frames"]) for row in belief_trace]
        trace_summary = summarize_belief_trace(brackets, belief_trace)
        two_sided = sum(1 for row in active_belief_trace if row.get("two_sided_witness_coverage"))
        trace_flags = {} if trace_flags is None else dict(trace_flags)
        deploy_ledger = {
            "route_label": ROUTE_LABEL,
            "method": METHOD_NAME,
            "voi_bbc_spec": "Value-of-Information Boundary Belief Controller",
            "video_id": video_id,
            "split": split,
            "window_id": int(window_id),
            "dense_T": int(dense_T),
            "selected_positions": selected_positions,
            "selected_times_sec": metadata["selected_times_sec"],
            "selected_positions_unit": "original_dense_index",
            "claim_mode": "local_gather_smoke",
            "valid_k": int(len(selected_positions)),
            "scaffold_k": int(role_counts.get("scaffold_anchor", 0)),
            "safety_floor_k": int(sum(1 for packet in selected_packets if packet.safety_floor)),
            "min_k": int(self.config.min_k),
            "max_k": int(self.config.max_k),
            "budget_stop_reason": stop_reason,
            "selection_gap_diagnostics": gap_diagnostics,
            "selected_packet_ids": [int(packet.packet_id) for packet in selected_packets],
            "selected_packet_roles": [packet.role for packet in selected_packets],
            "predicted_regret_values": [
                0.0 if packet.predicted_value is None else float(packet.predicted_value.predicted_regret)
                for packet in selected_packets
            ],
            "expected_belief_reductions": [
                0.0 if packet.predicted_value is None else float(packet.predicted_value.expected_belief_reduction)
                for packet in selected_packets
            ],
            "controller_trace_summary": {
                **trace_flags,
                "stop_reason": stop_reason,
                "num_rows": int(len(rows)),
                "selected_gap_guard_count": int(sum(1 for row in rows if row["selected_decision"] == "selected_gap_guard")),
                "selected_marginal_voi_count": int(sum(1 for row in rows if row["selected_decision"] == "selected_marginal_voi")),
            },
            "bracket_summary": {
                "num_brackets": int(len(brackets)),
                "num_active_brackets": int(len(active_brackets)),
                "mean_belief_entropy": float(np.mean([b.entropy for b in brackets]) if brackets else 0.0),
                "mean_belief_width_p80": float(np.mean(initial_widths) if initial_widths else 0.0),
                "initial_mean_belief_width_p80": float(np.mean(initial_widths) if initial_widths else 0.0),
                "posterior_mean_belief_width_p80": float(np.mean(posterior_widths) if posterior_widths else 0.0),
                "two_sided_witness_coverage_rate": float(two_sided / max(len(active_brackets), 1)),
                "belief_update_trace": belief_trace,
                "active_belief_update_trace": active_belief_trace,
                "all_active_beliefs_updated_and_safe": bool(self._all_active_beliefs_safe(brackets, belief_trace)),
                **trace_summary,
            },
            "original_time_metadata": metadata,
            "real_sparse_evidence": gather_evidence,
            "forbidden_fields_absent": {
                "regret_label_absent": True,
                "gt_fields_absent": True,
                "teacher_fields_absent": True,
                "prediction_cache_absent": True,
            },
            "provenance": self._clean_provenance(),
        }
        validate_deploy_ledger(deploy_ledger)
        return SelectionResult(
            selected_positions=selected_positions,
            selected_packets=selected_packets,
            ledger_rows=rows,
            deploy_ledger=deploy_ledger,
            stop_reason=stop_reason,
        )

    def _value_components(self, packet):
        if packet.predicted_value is not None:
            components = packet.predicted_value.diagnostics.get("value_components", {})
            if components:
                return {key: float(val) for key, val in components.items()}
        return {
            "expected_entropy_reduction": 0.0,
            "expected_width_reduction": 0.0,
            "expected_gap_risk_reduction": 0.0,
            "short_action_value": 0.0,
            "two_sided_witness_value": 0.0,
            "predicted_regret": 0.0,
            "value_per_cost": 0.0,
            "actionness_component": 0.0,
        }

    def _active_belief_trace(self, brackets, belief_trace):
        active_ids = {int(bracket.bracket_id) for bracket in brackets if bracket.state == "active"}
        return [row for row in belief_trace if int(row["bracket_id"]) in active_ids]

    def _all_active_beliefs_safe(self, brackets, belief_trace):
        active_trace = self._active_belief_trace(brackets, belief_trace)
        if not active_trace:
            return False
        return all(bool(row["updated_by_selected_witness"]) and bool(row["belief_width_safe"]) for row in active_trace)

    def _risk_constraints_satisfied(self, selected_positions, selected_packets, brackets, dense_T, belief_trace=None):
        if len(selected_positions) < self.config.min_k:
            return False
        if gap_statistics(selected_positions, int(dense_T))["max_gap"] > self.config.max_gap:
            return False
        belief_trace = update_boundary_belief_trace(brackets, selected_packets, safe_width=self.config.safe_belief_width) if belief_trace is None else belief_trace
        active_trace = self._active_belief_trace(brackets, belief_trace)
        if not active_trace:
            return True
        summary = summarize_belief_trace(brackets, belief_trace)
        if summary["max_posterior_risk_mass"] > self.config.safe_risk_mass:
            return False
        if self.config.require_two_sided_witness and any(
            row["initial_risk_mass"] > self.config.safe_risk_mass and not row["two_sided_witness_coverage"]
            for row in active_trace
        ):
            return False
        return True

    def _posterior_constraints_unsafe(self, selected_positions, selected_packets, brackets, dense_T):
        return not self._risk_constraints_satisfied(
            selected_positions,
            selected_packets,
            brackets,
            dense_T,
            update_boundary_belief_trace(brackets, selected_packets, safe_width=self.config.safe_belief_width),
        )

    def _constraint_state(self, before, after, packet, dense_T):
        before = list(before)
        after = list(after)
        stats_before = gap_statistics(before, int(dense_T)) if before else {"max_gap": int(dense_T)}
        stats_after = gap_statistics(after, int(dense_T)) if after else {"max_gap": int(dense_T)}
        duplicate = all(int(pos) in set(before) for pos in packet.positions)
        return {
            "budget_ok": bool(len(after) <= self.config.max_k),
            "min_budget_met": bool(len(after) >= self.config.min_k),
            "max_gap_ok": bool(int(stats_after["max_gap"]) <= int(self.config.max_gap)),
            "max_gap_before": int(stats_before["max_gap"]),
            "max_gap_after": int(stats_after["max_gap"]),
            "max_allowed_gap": int(self.config.max_gap),
            "duplicate": bool(duplicate),
            "role": packet.role,
            "safety_floor": bool(packet.safety_floor),
            "scaffold_as_candidate": bool(self.config.scaffold_as_candidate),
        }

    def _decision_subreason(self, decision):
        mapping = {
            "selected_safety_floor": "explicit_safety_floor_not_learned_novelty",
            "selected_min_k": "minimum_observation_floor",
            "selected_risk_constraint": "posterior_risk_constraint_frontier",
            "selected_gap_guard": "max_gap_constraint_repair",
            "selected_role_coverage": "high_risk_bracket_witness_coverage",
            "selected_marginal_voi": "marginal_voi_risk_reduction",
            "skipped_duplicate": "duplicate_position",
            "skipped_low_value": "regret_saturation_below_threshold",
            "skipped_budget_cap": "would_exceed_max_budget",
        }
        return mapping.get(decision, "unspecified")

    def _best_marginal_packet(self, candidates, selected_positions, selected_packets, brackets, dense_T):
        best = None
        best_components = None
        best_score = -1e18
        for packet in candidates:
            if any(pos in selected_positions for pos in packet.positions):
                continue
            components = self._marginal_components(packet, selected_positions, selected_packets, brackets, dense_T)
            score = self._component_score(components)
            key = (score, -packet.rank, -packet.packet_id)
            if score > best_score or (score == best_score and best is not None and key > (best_score, -best.rank, -best.packet_id)):
                best = packet
                best_components = components
                best_score = score
        return best, best_components

    def _marginal_components(self, packet, selected_positions, selected_packets, brackets, dense_T):
        components = self._value_components(packet)
        before_trace = update_boundary_belief_trace(brackets, selected_packets, safe_width=self.config.safe_belief_width)
        after_packets = list(selected_packets)
        if packet not in after_packets and packet.bracket_id is not None:
            after_packets.append(packet)
        after_trace = update_boundary_belief_trace(brackets, after_packets, safe_width=self.config.safe_belief_width)
        before_by_id = {int(row["bracket_id"]): row for row in before_trace}
        after_by_id = {int(row["bracket_id"]): row for row in after_trace}
        entropy_delta = 0.0
        width_delta = 0.0
        risk_delta = 0.0
        two_sided_gain = 0.0
        if packet.bracket_id is not None and int(packet.bracket_id) in before_by_id and int(packet.bracket_id) in after_by_id:
            before = before_by_id[int(packet.bracket_id)]
            after = after_by_id[int(packet.bracket_id)]
            entropy_delta = max(0.0, float(before["posterior_entropy"]) - float(after["posterior_entropy"]))
            width_delta = max(0.0, float(before["posterior_width_p80_frames"]) - float(after["posterior_width_p80_frames"])) / float(max(dense_T, 1))
            risk_delta = max(0.0, float(before["posterior_risk_mass"]) - float(after["posterior_risk_mass"]))
            two_sided_gain = float(after["two_sided_witness_coverage"] and not before["two_sided_witness_coverage"])

        before_gap = gap_statistics(selected_positions, int(dense_T))["max_gap"] if selected_positions else int(dense_T)
        after_positions = sorted_unique_positions(list(selected_positions) + list(packet.positions), int(dense_T))
        after_gap = gap_statistics(after_positions, int(dense_T))["max_gap"]
        gap_delta = float(max(0, before_gap - after_gap)) / float(max(self.config.max_gap, 1))
        components["expected_entropy_reduction"] = max(float(components.get("expected_entropy_reduction", 0.0)), entropy_delta)
        components["expected_width_reduction"] = max(float(components.get("expected_width_reduction", 0.0)), width_delta)
        components["expected_gap_risk_reduction"] = max(float(components.get("expected_gap_risk_reduction", 0.0)), gap_delta, risk_delta)
        components["two_sided_witness_value"] = max(float(components.get("two_sided_witness_value", 0.0)), 0.22 * two_sided_gain)
        if packet.role == "short_action_guard":
            components["short_action_value"] = max(float(components.get("short_action_value", 0.0)), 0.20)
        if packet.predicted_value is not None:
            components["predicted_regret"] = max(float(components.get("predicted_regret", 0.0)), float(packet.predicted_value.predicted_regret))
        nearest = min([abs(int(packet.positions[0]) - int(pos)) for pos in selected_positions], default=int(dense_T))
        if nearest <= 1:
            components["expected_width_reduction"] -= 0.10
        components["value_per_cost"] = self._component_score(components) / max(float(packet.cost_frames), 1e-6)
        return components

    def _component_score(self, components):
        if not components:
            return 0.0
        return float(
            components.get("expected_entropy_reduction", 0.0)
            + components.get("expected_width_reduction", 0.0)
            + components.get("expected_gap_risk_reduction", 0.0)
            + components.get("short_action_value", 0.0)
            + components.get("two_sided_witness_value", 0.0)
            + components.get("predicted_regret", 0.0)
            + 0.25 * components.get("actionness_component", 0.0)
        )

    def _belief_frontier_state(self, brackets, selected_packets):
        if not brackets:
            return self._empty_belief_state()
        trace = update_boundary_belief_trace(brackets, selected_packets, safe_width=self.config.safe_belief_width)
        return summarize_belief_trace(brackets, trace)

    def _empty_belief_state(self):
        return {
            "mean_posterior_entropy": 0.0,
            "mean_posterior_width": 0.0,
            "mean_posterior_risk_mass": 0.0,
            "max_posterior_risk_mass": 0.0,
            "two_sided_witness_coverage_rate": 0.0,
            "updated_bracket_rate": 0.0,
        }

    def _clean_provenance(self):
        return {
            "selection_uses_gt": False,
            "selection_uses_teacher": False,
            "selection_uses_prediction_cache": False,
            "selection_uses_raw_detector_prediction": False,
            "selection_uses_oracle_boundary": False,
            "selection_uses_oracle_residual": False,
        }
