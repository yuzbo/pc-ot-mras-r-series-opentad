from collections import Counter, defaultdict

import numpy as np

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
            return self._finish(
                selected_positions, selected_packets, rows, dense_T, fps, video_id, window_id, split, brackets, "gap_guard", dense_inputs
            )

        self._satisfy_role_coverage(brackets, sorted_candidates, selected_positions, selected_packets, rows, add_packet)
        active_widths = [float(bracket.width_p80_frames) for bracket in brackets if bracket.state == "active"]
        if len(selected_positions) >= self.config.min_k and active_widths:
            if float(np.mean(active_widths)) <= float(self.config.safe_belief_width):
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
                )
        if len(selected_positions) >= self.config.min_k and not active_widths:
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
            )

        stop_reason = None
        for packet in sorted_candidates:
            if len(selected_positions) >= self.config.max_k:
                stop_reason = "budget_cap"
                break
            if any(pos in selected_positions for pos in packet.positions):
                rows.append(self._row(packet, selected_positions, selected_positions, "skipped_duplicate", dense_T))
                continue
            value = 0.0 if packet.predicted_value is None else float(packet.predicted_value.value_per_cost)
            if value < float(self.config.min_marginal_value):
                stop_reason = "value_saturation"
                rows.append(self._row(packet, selected_positions, selected_positions, "skipped_low_value", dense_T))
                break
            add_packet(packet, "selected")

        if stop_reason is None:
            stop_reason = self._infer_stop_reason(selected_positions, brackets, sorted_candidates, dense_T)
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
                selected_positions[:] = repaired
                if packet not in selected_packets:
                    selected_packets.append(packet)
                rows.append(self._row(packet, before, selected_positions, "selected_gap_guard", dense_T))
        return selected_positions, rows

    def _satisfy_role_coverage(self, brackets, candidates, selected_positions, selected_packets, rows, add_packet):
        coverage = defaultdict(set)
        for packet in selected_packets:
            if packet.bracket_id is not None:
                coverage[packet.bracket_id].add(packet.role)
        needed_roles = {"transition_before", "transition_center", "transition_after"}
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

    def _infer_stop_reason(self, selected_positions, brackets, candidates, dense_T):
        if len(selected_positions) >= self.config.max_k:
            return "budget_cap"
        stats = gap_statistics(selected_positions, int(dense_T))
        if stats["max_gap"] >= self.config.max_gap:
            return "gap_guard"
        active_widths = [float(b.width_p80_frames) for b in brackets if b.state == "active"]
        if active_widths and float(np.mean(active_widths)) <= float(self.config.safe_belief_width):
            return "belief_width_safe"
        if not candidates:
            return "candidate_exhausted"
        return "candidate_exhausted"

    def _row(self, packet, before, after, decision, dense_T):
        value = packet.predicted_value
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
            "predicted_regret": 0.0 if value is None else float(value.predicted_regret),
            "expected_belief_reduction": 0.0 if value is None else float(value.expected_belief_reduction),
            "value_uncertainty": 0.0 if value is None else float(value.value_uncertainty),
            "value_per_cost": 0.0 if value is None else float(value.value_per_cost),
            "marginal_value_threshold": float(self.config.min_marginal_value),
            "valid_k_after": int(len(after)),
            "min_k": int(self.config.min_k),
            "max_k": int(self.config.max_k),
            "stop_reason_if_final": "not_final",
            "provenance": self._clean_provenance(),
        }

    def _finish(self, selected_positions, selected_packets, rows, dense_T, fps, video_id, window_id, split, brackets, stop_reason, dense_inputs):
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
                "mean_belief_width_p80": float(np.mean([b.width_p80_frames for b in brackets]) if brackets else 0.0),
                "two_sided_witness_coverage_rate": float(two_sided / max(len(active_brackets), 1)),
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

    def _clean_provenance(self):
        return {
            "selection_uses_gt": False,
            "selection_uses_teacher": False,
            "selection_uses_prediction_cache": False,
            "selection_uses_raw_detector_prediction": False,
            "selection_uses_oracle_boundary": False,
            "selection_uses_oracle_residual": False,
        }
