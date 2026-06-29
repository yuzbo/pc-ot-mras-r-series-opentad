from collections import Counter, defaultdict

import numpy as np

from .boundary_belief import required_witness_roles_for_bracket, update_boundary_belief_trace
from .scaffold import gap_statistics, repair_positions_for_max_gap
from .sparse_gather import sparse_gather
from .types import BudgetConfig, CandidatePacket, METHOD_NAME, ROUTE_LABEL, SelectionResult, STOP_REASONS, sorted_unique_positions
from .validators import build_original_time_metadata, build_selection_gap_diagnostics, validate_deploy_ledger


class DynamicBudgetController:
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
        packets_by_id = {packet.packet_id: packet for packet in scaffold_packets + candidate_packets}
        selected_packets = []
        selected_positions = []
        rows = []

        def add_packet(packet, decision):
            before = list(selected_positions)
            after_candidate = sorted_unique_positions(before + list(packet.positions), dense_T)
            if len(after_candidate) > self.config.max_k:
                rows.append(self._row(packet, before, before, "skipped_budget_cap", dense_T))
                return False
            changed = False
            for pos in packet.positions:
                if pos not in selected_positions:
                    selected_positions.append(int(pos))
                    changed = True
            selected_positions[:] = sorted_unique_positions(selected_positions, dense_T)
            if changed and packet not in selected_packets:
                selected_packets.append(packet)
            rows.append(self._row(packet, before, selected_positions, decision, dense_T))
            return changed

        for packet in sorted(scaffold_packets, key=lambda p: (p.rank, p.positions, p.packet_id)):
            add_packet(packet, "required_scaffold")
            if len(selected_positions) >= self.config.max_k:
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
                    "budget_cap",
                    dense_inputs,
                )

        sorted_candidates = self._rank_candidates(candidate_packets)

        while len(selected_positions) < min(self.config.min_k, self.config.max_k):
            packet = self._next_addable(sorted_candidates, selected_positions, dense_T, packets_by_id)
            if packet is None:
                break
            add_packet(packet, "selected_min_k")

        selected_positions, rows = self._repair_max_gap(selected_positions, selected_packets, rows, sorted_candidates, dense_T)
        if len(selected_positions) >= self.config.max_k:
            post_repair_gap = gap_statistics(selected_positions, dense_T)["max_gap"]
            stop = "gap_guard" if int(post_repair_gap) > int(self.config.max_gap) else "budget_cap"
            return self._finish(
                selected_positions, selected_packets, rows, dense_T, fps, video_id, window_id, split, brackets, stop, dense_inputs
            )

        self._satisfy_role_coverage(brackets, sorted_candidates, selected_positions, selected_packets, rows, add_packet)
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
            )

        stop_reason = None
        while len(selected_positions) < self.config.max_k:
            packet, components = self._best_marginal_packet(sorted_candidates, selected_positions, selected_packets, brackets, dense_T)
            if packet is None:
                break
            if len(selected_positions) >= self.config.max_k:
                stop_reason = "budget_cap"
                break
            if any(pos in selected_positions for pos in packet.positions):
                rows.append(self._row(packet, selected_positions, selected_positions, "skipped_duplicate", dense_T))
                continue
            value = float(sum(components.values()))
            if value < float(self.config.min_marginal_value):
                stop_reason = "value_saturation"
                rows.append(self._row(packet, selected_positions, selected_positions, "skipped_low_value", dense_T, value_components=components))
                break
            before = list(selected_positions)
            changed = add_packet(packet, "selected")
            if changed and rows:
                rows[-1]["value_components"] = {key: float(val) for key, val in components.items()}
                rows[-1]["marginal_value_per_cost"] = float(value)
                belief_trace = update_boundary_belief_trace(brackets, selected_packets, safe_width=self.config.safe_belief_width)
                if len(selected_positions) >= self.config.min_k and self._all_active_beliefs_safe(brackets, belief_trace):
                    stop_reason = "belief_width_safe"
                    break
            if before == selected_positions:
                break

        belief_trace = update_boundary_belief_trace(brackets, selected_packets, safe_width=self.config.safe_belief_width)
        if stop_reason is None:
            stop_reason = self._infer_stop_reason(selected_positions, brackets, sorted_candidates, dense_T, belief_trace=belief_trace)
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
        )

    def _rank_candidates(self, packets):
        return sorted(
            packets,
            key=lambda packet: (
                -float(packet.predicted_value.value_per_cost if packet.predicted_value else 0.0),
                -int(packet.required_for_role_coverage),
                packet.role,
                packet.positions,
                packet.packet_id,
            ),
        )

    def _next_addable(self, packets, selected_positions, dense_T, packets_by_id):
        for packet in packets:
            if any(pos not in selected_positions for pos in packet.positions):
                return packet
        return None

    def _repair_max_gap(self, selected_positions, selected_packets, rows, candidates, dense_T):
        repaired, bridges = repair_positions_for_max_gap(selected_positions, dense_T, self.config.max_gap)
        if len(repaired) > self.config.max_k:
            raise ValueError(
                "hard max-gap contract infeasible under budget: "
                f"dense_T={int(dense_T)} max_gap={int(self.config.max_gap)} "
                f"max_k={int(self.config.max_k)} required_k={len(repaired)}"
            )
        for bridge in bridges:
            packet = None
            for candidate in candidates:
                if candidate.role == "gap_bridge" and bridge in candidate.positions:
                    packet = candidate
                    break
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
                    reason="controller_hard_max_gap_repair",
                    max_gap_repair=True,
                    feature_summary={"gap_risk": 1.0, "gap_if_omitted_frames": float(self.config.max_gap + 1)},
                )
            if len(repaired) <= self.config.max_k:
                before = list(selected_positions)
                if int(bridge) not in selected_positions:
                    selected_positions.append(int(bridge))
                    selected_positions[:] = sorted_unique_positions(selected_positions, dense_T)
                if packet not in selected_packets:
                    selected_packets.append(packet)
                rows.append(self._row(packet, before, selected_positions, "selected_gap_guard", dense_T))
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
        feasible_role_budget = max(0, self.config.max_k - len(selected_positions) - 1)
        max_brackets = max(1, feasible_role_budget // 3) if feasible_role_budget > 0 else 0
        for bracket in active_brackets[:max_brackets]:
            needed_roles = required_witness_roles_for_bracket(bracket)
            missing = sorted(needed_roles.difference(coverage.get(bracket.bracket_id, set())))
            for role in missing:
                if len(selected_positions) >= self.config.max_k:
                    return
                match = None
                for packet in candidates:
                    if packet.bracket_id == bracket.bracket_id and packet.role == role:
                        match = packet
                        break
                if match is not None:
                    add_packet(match, "selected_role_coverage")
                    coverage[bracket.bracket_id].add(role)

    def _infer_stop_reason(self, selected_positions, brackets, candidates, dense_T, belief_trace=None):
        if len(selected_positions) >= self.config.max_k:
            return "budget_cap"
        stats = gap_statistics(selected_positions, int(dense_T))
        if stats["max_gap"] > self.config.max_gap:
            return "gap_guard"
        belief_trace = [] if belief_trace is None else belief_trace
        active_ids = {bracket.bracket_id for bracket in brackets if bracket.state == "active"}
        if active_ids and self._all_active_beliefs_safe(brackets, belief_trace):
            return "belief_width_safe"
        if not candidates:
            return "candidate_exhausted"
        return "candidate_exhausted"

    def _row(self, packet, before, after, decision, dense_T, value_components=None):
        value = packet.predicted_value
        value_components = self._value_components(packet) if value_components is None else value_components
        return {
            "route_label": ROUTE_LABEL,
            "method": METHOD_NAME,
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
            "selected_before_positions": list(before),
            "selected_after_positions": list(after),
            "selected_decision": decision,
            "selected_decision_subreason": self._decision_subreason(decision),
            "predicted_regret": 0.0 if value is None else float(value.predicted_regret),
            "expected_belief_reduction": 0.0 if value is None else float(value.expected_belief_reduction),
            "value_uncertainty": 0.0 if value is None else float(value.value_uncertainty),
            "value_per_cost": 0.0 if value is None else float(value.value_per_cost),
            "marginal_value_per_cost": float(sum(value_components.values())),
            "value_components": {key: float(val) for key, val in value_components.items()},
            "constraint_state": self._constraint_state(before, after, packet, dense_T),
            "marginal_value_threshold": float(self.config.min_marginal_value),
            "valid_k_after": int(len(after)),
            "min_k": int(self.config.min_k),
            "max_k": int(self.config.max_k),
            "stop_reason_if_final": "not_final",
            "provenance": self._clean_provenance(),
        }

    def _finish(self, selected_positions, selected_packets, rows, dense_T, fps, video_id, window_id, split, brackets, stop_reason, dense_inputs, belief_trace=None):
        selected_positions = sorted_unique_positions(selected_positions, dense_T)
        gap_diagnostics = build_selection_gap_diagnostics(selected_positions, dense_T, self.config.max_gap)
        if gap_diagnostics["coverage_violation"]:
            raise ValueError(
                "hard max-gap contract violated before ledger generation: "
                f"max_gap={gap_diagnostics['max_gap']} "
                f"max_allowed_gap={gap_diagnostics['max_allowed_gap']}"
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
        two_sided = 0
        for bracket in active_brackets:
            roles = {packet.role for packet in selected_packets if packet.bracket_id == bracket.bracket_id}
            if {"transition_before", "transition_after"}.issubset(roles):
                two_sided += 1
        deploy_ledger = {
            "route_label": ROUTE_LABEL,
            "method": METHOD_NAME,
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
            "belief_width_gain": 0.0,
            "role_gain": 0.0,
            "gap_gain": 0.0,
            "short_action_gain": 0.0,
            "redundancy_repulsion_penalty": 0.0,
            "low_actionness_component": 0.0,
        }

    def _active_belief_trace(self, brackets, belief_trace):
        active_ids = {int(bracket.bracket_id) for bracket in brackets if bracket.state == "active"}
        return [row for row in belief_trace if int(row["bracket_id"]) in active_ids]

    def _all_active_beliefs_safe(self, brackets, belief_trace):
        active_trace = self._active_belief_trace(brackets, belief_trace)
        if not active_trace:
            return False
        return all(bool(row["updated_from_selected_witness"]) and bool(row["belief_width_safe"]) for row in active_trace)

    def _constraint_state(self, before, after, packet, dense_T):
        before = list(before)
        after = list(after)
        stats = gap_statistics(after, int(dense_T)) if after else {"max_gap": int(dense_T)}
        duplicate = all(int(pos) in set(before) for pos in packet.positions)
        return {
            "budget_ok": bool(len(after) <= self.config.max_k),
            "min_budget_met": bool(len(after) >= self.config.min_k),
            "max_gap_ok": bool(int(stats["max_gap"]) <= int(self.config.max_gap)),
            "max_gap_after": int(stats["max_gap"]),
            "max_allowed_gap": int(self.config.max_gap),
            "duplicate": bool(duplicate),
            "role": packet.role,
        }

    def _decision_subreason(self, decision):
        mapping = {
            "required_scaffold": "scaffold_anchor_required",
            "selected_min_k": "min_budget_fill",
            "selected_gap_guard": "hard_max_gap_incremental_bridge",
            "selected_role_coverage": "bracket_kind_required_role",
            "selected": "marginal_gain_positive",
            "skipped_duplicate": "duplicate_position",
            "skipped_low_value": "marginal_gain_below_threshold",
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
            score = float(sum(components.values()))
            key = (score, -packet.rank, -packet.packet_id)
            if score > best_score or (score == best_score and best is not None and key > (best_score, -best.rank, -best.packet_id)):
                best = packet
                best_components = components
                best_score = score
        return best, best_components

    def _marginal_components(self, packet, selected_positions, selected_packets, brackets, dense_T):
        components = self._value_components(packet)
        before_gap = gap_statistics(selected_positions, int(dense_T))["max_gap"] if selected_positions else int(dense_T)
        after_positions = sorted_unique_positions(list(selected_positions) + list(packet.positions), int(dense_T))
        after_gap = gap_statistics(after_positions, int(dense_T))["max_gap"]
        components["gap_gain"] = max(float(components.get("gap_gain", 0.0)), float(max(0, before_gap - after_gap)) / float(max(self.config.max_gap, 1)))
        if packet.bracket_id is not None:
            current_roles = {p.role for p in selected_packets if p.bracket_id == packet.bracket_id}
            bracket = next((b for b in brackets if b.bracket_id == packet.bracket_id), None)
            needed = required_witness_roles_for_bracket(bracket) if bracket is not None else set()
            if packet.role in needed and packet.role not in current_roles:
                components["role_gain"] = max(float(components.get("role_gain", 0.0)), 0.18)
        nearest = min([abs(int(packet.positions[0]) - int(pos)) for pos in selected_positions], default=int(dense_T))
        if nearest <= 1:
            components["redundancy_repulsion_penalty"] = float(components.get("redundancy_repulsion_penalty", 0.0)) - 0.12
        return components

    def _clean_provenance(self):
        return {
            "selection_uses_gt": False,
            "selection_uses_teacher": False,
            "selection_uses_prediction_cache": False,
            "selection_uses_raw_detector_prediction": False,
            "selection_uses_oracle_boundary": False,
            "selection_uses_oracle_residual": False,
        }
