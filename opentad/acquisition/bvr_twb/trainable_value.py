import numpy as np

from .types import PacketValue
from .value_predictor import PacketValuePredictor


PACKET_FEATURE_KEYS = (
    "mean_actionness",
    "max_transition",
    "mean_uncertainty",
    "persistence",
    "short_action_risk",
    "two_sided_state_contrast",
    "bracket_width_frames",
    "gap_if_omitted_frames",
    "gap_risk",
    "belief_entropy",
)

PACKET_ROLE_KEYS = (
    "scaffold_anchor",
    "gap_bridge",
    "transition_before",
    "transition_center",
    "transition_after",
    "action_core",
    "ambiguity_probe",
    "short_action_guard",
)


def _require_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    return torch, nn, F


def packet_feature_vector(packet, dense_T=None):
    features = getattr(packet, "feature_summary", {}) or {}
    dense_T = float(dense_T or getattr(packet, "dense_T", 1) or 1)
    values = []
    for key in PACKET_FEATURE_KEYS:
        value = float(features.get(key, 0.0))
        if key in {"bracket_width_frames", "gap_if_omitted_frames"}:
            value = value / max(dense_T, 1.0)
        values.append(value)
    role = getattr(packet, "role", "")
    values.extend([1.0 if role == key else 0.0 for key in PACKET_ROLE_KEYS])
    values.extend(
        [
            float(getattr(packet, "cost_frames", 1.0)) / max(dense_T, 1.0),
            float(getattr(packet, "visibility", 1.0)),
            1.0 if bool(getattr(packet, "required_for_role_coverage", False)) else 0.0,
            1.0 if bool(getattr(packet, "max_gap_repair", False)) else 0.0,
        ]
    )
    arr = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(arr)):
        raise ValueError("BVR-TWB packet feature vector contains non-finite values")
    return arr


def packet_feature_matrix(packets, dense_T=None):
    rows = [packet_feature_vector(packet, dense_T=dense_T) for packet in packets]
    if not rows:
        return np.zeros((0, len(PACKET_FEATURE_KEYS) + len(PACKET_ROLE_KEYS) + 4), dtype=np.float32)
    return np.stack(rows, axis=0).astype(np.float32)


def build_value_training_batch(packets, regret_labels, dense_T=None, device=None):
    torch, _, _ = _require_torch()
    label_by_id = {int(row["packet_id"]): float(row.get("target_omission_regret", row["target_regret"])) for row in regret_labels}
    usable = [packet for packet in packets if int(packet.packet_id) in label_by_id]
    if not usable:
        raise ValueError("BVR-TWB value training batch has no packet/label matches")
    x = torch.as_tensor(packet_feature_matrix(usable, dense_T=dense_T), dtype=torch.float32, device=device)
    y = torch.as_tensor([label_by_id[int(packet.packet_id)] for packet in usable], dtype=torch.float32, device=device)
    return x, y


def build_voi_training_targets(regret_labels, device=None):
    torch, _, _ = _require_torch()
    rows = list(regret_labels)
    if not rows:
        raise ValueError("BVR-TWB VOI training targets require at least one label")
    keys = (
        "target_omission_regret",
        "target_entropy_reduction",
        "target_width_reduction",
        "target_boundary_risk_reduction",
    )
    return {
        key: torch.as_tensor([float(row[key]) for row in rows], dtype=torch.float32, device=device)
        for key in keys
    }


class BVRPacketValueMLP:
    def __new__(cls, input_dim=None, hidden_dim=64, dropout=0.0):
        _, nn, _ = _require_torch()
        input_dim = input_dim or (len(PACKET_FEATURE_KEYS) + len(PACKET_ROLE_KEYS) + 4)

        class _BVRPacketValueMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(int(input_dim), int(hidden_dim)),
                    nn.ReLU(inplace=True),
                    nn.Dropout(float(dropout)),
                    nn.Linear(int(hidden_dim), int(hidden_dim)),
                    nn.ReLU(inplace=True),
                    nn.Linear(int(hidden_dim), 1),
                )

            def forward(self, features):
                torch, _, _ = _require_torch()
                return torch.sigmoid(self.net(features).squeeze(-1))

        return _BVRPacketValueMLP()


def packet_value_loss(predicted_regret, target_regret, reduction="mean"):
    _, _, F = _require_torch()
    return F.smooth_l1_loss(predicted_regret, target_regret, reduction=reduction)


class LearnedPacketValueAdapter:
    def __init__(self, model=None, fallback=None, device=None):
        self.model = model
        self.fallback = fallback or PacketValuePredictor(mode="heuristic_fallback")
        self.device = device

    def score_packet(self, packet):
        if self.model is None:
            return self.fallback.score_packet(packet)
        torch, _, _ = _require_torch()
        self.model.eval()
        with torch.no_grad():
            features = torch.as_tensor(
                packet_feature_vector(packet, dense_T=packet.dense_T)[None],
                dtype=torch.float32,
                device=self.device,
            )
            regret = float(self.model(features).detach().cpu().reshape(-1)[0].item())
        fallback_value = self.fallback.score_packet(packet)
        cost = max(float(packet.cost_frames), 1e-6)
        packet.predicted_value = PacketValue(
            predicted_regret=regret,
            expected_belief_reduction=float(fallback_value.expected_belief_reduction),
            value_uncertainty=float(fallback_value.value_uncertainty),
            value_per_cost=float(regret / cost),
            diagnostics={
                **dict(fallback_value.diagnostics),
                "learned_value_used": True,
                "fallback_predicted_regret": float(fallback_value.predicted_regret),
            },
        )
        return packet.predicted_value

    def score_packets(self, packets):
        for packet in packets:
            self.score_packet(packet)
        return packets
