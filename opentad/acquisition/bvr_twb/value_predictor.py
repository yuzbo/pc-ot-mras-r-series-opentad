import numpy as np

from .types import CandidatePacket, PacketValue


ROLE_BONUS = {
    "transition_before": 0.20,
    "transition_center": 0.25,
    "transition_after": 0.20,
    "short_action_guard": 0.22,
    "ambiguity_probe": 0.15,
    "gap_bridge": 0.14,
    "action_core": 0.08,
    "scaffold_anchor": 0.02,
}


class PacketValuePredictor:
    """Deploy-visible heuristic and constant ablation value predictor."""

    def __init__(self, mode="heuristic_fallback", min_non_action_fraction=0.55):
        if mode not in {"deploy_voi_heuristic", "heuristic_fallback", "mock_constant_ablation"}:
            raise ValueError(f"unsupported PacketValuePredictor mode: {mode}")
        self.mode = mode
        self.min_non_action_fraction = float(min_non_action_fraction)

    def score_packet(self, packet: CandidatePacket):
        features = packet.feature_summary
        if self.mode == "mock_constant_ablation":
            components = {
                "expected_entropy_reduction": 0.20,
                "expected_width_reduction": 0.20,
                "expected_gap_risk_reduction": 0.05,
                "short_action_value": 0.0,
                "two_sided_witness_value": ROLE_BONUS.get(packet.role, 0.0) * 0.10,
                "predicted_regret": 0.25,
                "value_per_cost": 0.0,
                "actionness_component": 0.0,
            }
            value = sum(components.values())
            expected_belief = 0.25
            uncertainty = 0.50
        else:
            role_bonus = ROLE_BONUS.get(packet.role, 0.0)
            transition = float(features.get("max_transition", 0.0))
            uncertainty_feat = float(features.get("mean_uncertainty", 0.0))
            bracket_width = float(features.get("bracket_width_frames", 0.0))
            short_risk = float(features.get("short_action_risk", 0.0))
            contrast = float(features.get("two_sided_state_contrast", 0.0))
            gap_risk = float(features.get("gap_risk", features.get("gap_if_omitted_frames", 0.0) / 64.0))
            actionness = float(features.get("mean_actionness", 0.0))
            width_term = float(np.clip(bracket_width / 24.0, 0.0, 1.0))
            entropy = float(features.get("belief_entropy", uncertainty_feat))
            two_sided_role = float(packet.role in {"transition_before", "transition_after", "transition_center", "ambiguity_probe"})
            two_sided = float(features.get("two_sided_state_contrast", 0.0)) * two_sided_role
            predicted_regret = (
                0.18 * transition
                + 0.16 * uncertainty_feat
                + 0.13 * width_term
                + 0.12 * short_risk
                + 0.10 * np.clip(gap_risk, 0.0, 1.0)
                + role_bonus
            )
            components = {
                "expected_entropy_reduction": 0.20 * entropy + 0.10 * uncertainty_feat + 0.08 * transition,
                "expected_width_reduction": 0.18 * width_term + 0.10 * contrast + 0.06 * transition,
                "expected_gap_risk_reduction": 0.14 * np.clip(gap_risk, 0.0, 1.0),
                "short_action_value": 0.17 * short_risk,
                "two_sided_witness_value": 0.14 * two_sided + 0.06 * float(packet.role in {"transition_before", "transition_after"}),
                "predicted_regret": predicted_regret,
                "value_per_cost": 0.0,
                "actionness_component": 0.03 * actionness * float(packet.role == "action_core"),
            }
            if float(features.get("gap_if_omitted_frames", 99.0)) <= 1.0:
                components["expected_gap_risk_reduction"] -= 0.08
            value = sum(components.values())
            expected_belief = float(np.clip(0.35 * transition + 0.30 * uncertainty_feat + 0.20 * contrast + 0.15 * short_risk, 0.0, 1.0))
            uncertainty = float(np.clip(0.55 * uncertainty_feat + 0.25 * (1.0 - contrast) + 0.20 * width_term, 0.0, 1.0))

        cost = max(float(packet.cost_frames), 1e-6)
        value_no_vpc = float(sum(v for k, v in components.items() if k != "value_per_cost"))
        components["value_per_cost"] = float(np.clip(value_no_vpc, 0.0, 2.0) / cost)
        packet.predicted_value = PacketValue(
            predicted_regret=float(np.clip(components.get("predicted_regret", value_no_vpc), 0.0, 2.0)),
            expected_belief_reduction=float(np.clip(expected_belief, 0.0, 1.0)),
            value_uncertainty=float(np.clip(uncertainty, 0.0, 1.0)),
            value_per_cost=float(components["value_per_cost"]),
            diagnostics=self._diagnostics(packet, value_no_vpc, components),
        )
        return packet.predicted_value

    def score_packets(self, packets):
        for packet in packets:
            self.score_packet(packet)
        return packets

    def _diagnostics(self, packet, value, components):
        action_component = abs(float(components.get("actionness_component", 0.0)))
        total_abs = max(sum(abs(float(val)) for val in components.values()), 1e-9)
        non_action = 1.0 - action_component / total_abs
        return {
            "actionness_component_fraction": float(action_component / total_abs),
            "non_action_component_fraction": float(non_action),
            "voi_non_action_component_fraction": float(non_action),
            "value_components": {key: float(val) for key, val in components.items()},
        }

    def anti_actionness_only_diagnostics(self, packets, top_fraction=0.35):
        packets = list(packets)
        if len(packets) < 3:
            return {"passes": True, "top_overlap": 0.0, "mean_non_action_fraction": 1.0}
        for packet in packets:
            if packet.predicted_value is None:
                self.score_packet(packet)
        n_top = max(1, int(round(len(packets) * float(top_fraction))))
        by_value = sorted(
            packets,
            key=lambda packet: (
                -float(packet.predicted_value.value_per_cost),
                packet.role,
                packet.positions,
                packet.packet_id,
            ),
        )[:n_top]
        by_action = sorted(
            packets,
            key=lambda packet: (
                -float(packet.feature_summary.get("mean_actionness", 0.0)),
                packet.role,
                packet.positions,
                packet.packet_id,
            ),
        )[:n_top]
        value_ids = {packet.packet_id for packet in by_value}
        action_ids = {packet.packet_id for packet in by_action}
        overlap = len(value_ids.intersection(action_ids)) / float(n_top)
        non_action_fracs = [
            packet.predicted_value.diagnostics.get("non_action_component_fraction", 0.0)
            for packet in packets
        ]
        action_fracs = [
            packet.predicted_value.diagnostics.get("actionness_component_fraction", 0.0)
            for packet in packets
        ]
        mean_non_action = float(np.mean(non_action_fracs)) if non_action_fracs else 0.0
        max_action_fraction = float(max(action_fracs) if action_fracs else 0.0)
        passes = bool(mean_non_action >= self.min_non_action_fraction and max_action_fraction <= (1.0 - self.min_non_action_fraction))
        return {
            "passes": passes,
            "top_overlap": float(overlap),
            "mean_non_action_fraction": mean_non_action,
            "max_actionness_component_fraction": max_action_fraction,
            "mode": self.mode,
        }

    def assert_not_actionness_only(self, packets):
        diag = self.anti_actionness_only_diagnostics(packets)
        if not diag["passes"]:
            raise ValueError(
                "PacketValuePredictor failed anti-actionness-only gate: "
                f"top_overlap={diag['top_overlap']:.3f}, "
                f"mean_non_action_fraction={diag['mean_non_action_fraction']:.3f}"
            )
        return diag
