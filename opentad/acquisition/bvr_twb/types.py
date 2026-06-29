from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


ROUTE_LABEL = "DIVERGENT_INNOVATION_BVR_TWB_DO_NOT_MERGE_WITH_C3"
METHOD_NAME = "BVR-TWB"

STOP_REASONS = {
    "value_saturation",
    "belief_width_safe",
    "gap_guard",
    "budget_cap",
    "candidate_exhausted",
}

PACKET_ROLES = {
    "scaffold_anchor",
    "gap_bridge",
    "transition_before",
    "transition_center",
    "transition_after",
    "action_core",
    "ambiguity_probe",
    "short_action_guard",
}

FORBIDDEN_ROUTE_TOKENS = (
    "BH_SDC",
    "C3",
    "C3-Pro",
    "C3_PRO",
    "GlobalRank",
    "Interval",
    "dynamic-budget-C3",
    "dynamic_budget_C3",
    "physical-grid",
    "physical_grid",
    "COMBO",
)

FORBIDDEN_DEPLOY_KEYS = (
    "gt_segments",
    "gt_labels",
    "teacher_logits",
    "teacher_scores",
    "dense_detector_predictions",
    "raw_detector_prediction",
    "raw_detector_predictions",
    "proposal_cache",
    "prediction_cache",
    "oracle_boundary",
    "oracle_residual",
    "c3_scores",
    "globalrank_scores",
    "train_regret_label",
    "train_counterfactual_delta",
    "train_quality_proxy",
    "train_gt_boundary_band",
    "train_gt_gap_risk",
)


def as_float_array(values: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def sorted_unique_positions(positions: Sequence[int], dense_T: int) -> List[int]:
    dense_T = int(dense_T)
    result = sorted({int(pos) for pos in positions})
    if any(pos < 0 or pos >= dense_T for pos in result):
        raise ValueError(f"positions out of range for dense_T={dense_T}: {result}")
    return result


@dataclass(frozen=True)
class ScoutCurves:
    dense_T: int
    p_action: np.ndarray
    p_background: np.ndarray
    uncertainty: np.ndarray
    transition_score: np.ndarray
    persistence: np.ndarray
    short_action_risk: np.ndarray
    preview_signal: Optional[np.ndarray] = None
    source: str = "deploy_visible_cpu_fallback"
    route_label: str = ROUTE_LABEL

    def __post_init__(self):
        dense_T = int(self.dense_T)
        if dense_T <= 0:
            raise ValueError("dense_T must be positive")
        for name in (
            "p_action",
            "p_background",
            "uncertainty",
            "transition_score",
            "persistence",
            "short_action_risk",
        ):
            arr = np.asarray(getattr(self, name), dtype=np.float64).reshape(-1)
            if arr.shape[0] != dense_T:
                raise ValueError(f"{name} length {arr.shape[0]} != dense_T {dense_T}")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} contains non-finite values")
            object.__setattr__(self, name, np.clip(arr, 0.0, 1.0))
        if self.preview_signal is not None:
            preview = np.asarray(self.preview_signal, dtype=np.float64).reshape(-1)
            if preview.shape[0] != dense_T:
                raise ValueError("preview_signal length must match dense_T")
            object.__setattr__(self, "preview_signal", np.clip(preview, 0.0, 1.0))
        if self.route_label != ROUTE_LABEL:
            raise ValueError(f"invalid route label: {self.route_label}")


@dataclass
class PacketValue:
    predicted_regret: float
    expected_belief_reduction: float
    value_uncertainty: float
    value_per_cost: float
    diagnostics: Dict[str, float] = field(default_factory=dict)


@dataclass
class CandidatePacket:
    packet_id: int
    video_id: str
    window_id: int
    split: str
    source: str
    role: str
    positions: List[int]
    dense_T: int
    bracket_id: Optional[int] = None
    cost_frames: float = 1.0
    visibility: float = 1.0
    rank: int = 0
    reason: str = ""
    feature_summary: Dict[str, float] = field(default_factory=dict)
    predicted_value: Optional[PacketValue] = None
    required_for_scaffold: bool = False
    required_for_role_coverage: bool = False
    max_gap_repair: bool = False
    route_label: str = ROUTE_LABEL

    def __post_init__(self):
        if self.route_label != ROUTE_LABEL:
            raise ValueError(f"invalid packet route label: {self.route_label}")
        if self.role not in PACKET_ROLES:
            raise ValueError(f"unsupported packet role: {self.role}")
        if self.split not in {"train", "val", "test", "deploy", "synthetic"}:
            raise ValueError(f"unsupported split: {self.split}")
        self.positions = sorted_unique_positions(self.positions, self.dense_T)
        if not self.positions:
            raise ValueError("CandidatePacket positions must not be empty")
        if self.cost_frames <= 0:
            raise ValueError("cost_frames must be positive")
        self.visibility = float(np.clip(self.visibility, 0.0, 1.0))

    @property
    def start_pos(self) -> int:
        return int(self.positions[0])

    @property
    def end_pos(self) -> int:
        return int(self.positions[-1])

    def to_ledger_dict(self) -> Dict[str, object]:
        value = self.predicted_value
        return {
            "route_label": self.route_label,
            "method": METHOD_NAME,
            "packet_id": int(self.packet_id),
            "video_id": self.video_id,
            "window_id": int(self.window_id),
            "split": self.split,
            "packet_source": self.source,
            "packet_role": self.role,
            "bracket_id": self.bracket_id,
            "packet_positions": list(self.positions),
            "packet_cost_frames": float(self.cost_frames),
            "predicted_regret": None if value is None else float(value.predicted_regret),
            "expected_belief_reduction": None if value is None else float(value.expected_belief_reduction),
            "value_uncertainty": None if value is None else float(value.value_uncertainty),
            "value_per_cost": None if value is None else float(value.value_per_cost),
            "rank": int(self.rank),
            "reason": self.reason,
        }


@dataclass
class BracketState:
    bracket_id: int
    video_id: str
    window_id: int
    split: str
    kind: str
    left_pos: int
    right_pos: int
    center_pos: int
    dense_T: int
    peak_transition: float
    mean_uncertainty: float
    action_left_mean: float
    action_right_mean: float
    persistence: float
    short_action_risk: float
    gap_risk: float
    entropy: float
    width_p50_frames: float
    width_p80_frames: float
    confidence: float
    two_sided_state_contrast: float
    state: str = "active"
    route_label: str = ROUTE_LABEL

    def __post_init__(self):
        if self.route_label != ROUTE_LABEL:
            raise ValueError(f"invalid bracket route label: {self.route_label}")
        if self.split not in {"train", "val", "test", "deploy", "synthetic"}:
            raise ValueError(f"unsupported split: {self.split}")
        if not (0 <= int(self.left_pos) <= int(self.center_pos) <= int(self.right_pos) < int(self.dense_T)):
            raise ValueError("bracket positions must be ordered original dense indices")
        if self.state not in {"active", "selected", "belief_safe", "gap_guarded", "stale", "rejected"}:
            raise ValueError(f"unsupported bracket state: {self.state}")

    @property
    def width_frames(self) -> int:
        return int(self.right_pos - self.left_pos + 1)


@dataclass
class BudgetConfig:
    min_k: int
    max_k: int
    max_gap: int
    min_marginal_value: float = 0.08
    safe_belief_width: float = 5.0
    require_two_sided_witness: bool = True
    route_label: str = ROUTE_LABEL

    def __post_init__(self):
        if self.route_label != ROUTE_LABEL:
            raise ValueError(f"invalid budget route label: {self.route_label}")
        if self.min_k < 1 or self.max_k < self.min_k:
            raise ValueError("budget requires 1 <= min_k <= max_k")
        if self.max_gap < 1:
            raise ValueError("max_gap must be positive")


@dataclass
class SelectionResult:
    selected_positions: List[int]
    selected_packets: List[CandidatePacket]
    ledger_rows: List[Dict[str, object]]
    deploy_ledger: Dict[str, object]
    stop_reason: str

