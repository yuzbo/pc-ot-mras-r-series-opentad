from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


ROUTE_LABEL = "DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3"
METHOD_NAME = "RBA-RBR"
METHOD_KEY = "rba_rbr_recoverable_bracketing"

STOP_REASONS = {
    "risk_satisfied",
    "regret_saturation",
    "budget_cap",
    "candidate_exhausted",
}

FORBIDDEN_ROUTE_TOKENS = (
    "BH_SDC",
    "BVR_TWB",
    "BVR-TWB",
    "ABR",
    "CADF",
    "PQR",
    "C3",
    "C3-Pro",
    "C3_PRO",
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
    "train_regret_label",
    "train_counterfactual_delta",
    "train_gt_boundary_band",
)

PROBE_STAGES = {"scaffold", "refine", "rescue"}


def as_float_array(values: Sequence[float], name: str, dense_T: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if dense_T is not None and arr.size != int(dense_T):
        raise ValueError(f"{name} length {arr.size} != dense_T {int(dense_T)}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return np.clip(arr, 0.0, 1.0)


def sorted_unique_positions(positions: Sequence[int], dense_T: int) -> List[int]:
    dense_T = int(dense_T)
    result = sorted({int(pos) for pos in positions})
    if any(pos < 0 or pos >= dense_T for pos in result):
        raise ValueError(f"positions out of range for dense_T={dense_T}: {result}")
    return result


@dataclass(frozen=True)
class RbaRbrRiskMap:
    dense_T: int
    actionness: np.ndarray
    uncertainty: np.ndarray
    transition: np.ndarray
    staleness: np.ndarray
    conflict: np.ndarray
    gap_risk: np.ndarray
    soft_bracket_prior: np.ndarray
    rescue_score: np.ndarray
    source: str = "deploy_visible_metadata_preview"
    route_label: str = ROUTE_LABEL

    def __post_init__(self):
        if self.route_label != ROUTE_LABEL:
            raise ValueError(f"invalid route label: {self.route_label}")
        dense_T = int(self.dense_T)
        if dense_T <= 0:
            raise ValueError("dense_T must be positive")
        for name in (
            "actionness",
            "uncertainty",
            "transition",
            "staleness",
            "conflict",
            "gap_risk",
            "soft_bracket_prior",
            "rescue_score",
        ):
            object.__setattr__(self, name, as_float_array(getattr(self, name), name, dense_T=dense_T))


@dataclass(frozen=True)
class SoftBracket:
    bracket_id: int
    left: int
    center: int
    right: int
    dense_T: int
    hard_left: int
    hard_right: int
    hard_bracket: bool
    confidence: float
    risk_mass: float
    source: str
    route_label: str = ROUTE_LABEL

    def __post_init__(self):
        if self.route_label != ROUTE_LABEL:
            raise ValueError(f"invalid route label: {self.route_label}")
        if not (0 <= int(self.left) <= int(self.center) <= int(self.right) < int(self.dense_T)):
            raise ValueError("soft bracket positions must be ordered original dense indices")
        if self.hard_bracket and not (0 <= int(self.hard_left) <= int(self.hard_right) < int(self.dense_T)):
            raise ValueError("hard bracket span out of range")


@dataclass
class ProbeCandidate:
    probe_id: int
    stage: str
    role: str
    center_pos: int
    positions: List[int]
    dense_T: int
    predicted_regret: float
    source_signal: str
    bracket_id: Optional[int] = None
    outside_hard_bracket: bool = False
    rank: int = 0
    reason: str = ""
    components: Dict[str, float] = field(default_factory=dict)
    route_label: str = ROUTE_LABEL

    def __post_init__(self):
        if self.route_label != ROUTE_LABEL:
            raise ValueError(f"invalid route label: {self.route_label}")
        if self.stage not in PROBE_STAGES:
            raise ValueError(f"unsupported probe stage: {self.stage}")
        self.positions = sorted_unique_positions(self.positions, self.dense_T)
        if not self.positions:
            raise ValueError("ProbeCandidate positions must not be empty")
        self.center_pos = int(np.clip(int(self.center_pos), 0, int(self.dense_T) - 1))
        self.predicted_regret = float(max(self.predicted_regret, 0.0))

    def to_ledger_dict(self) -> Dict[str, object]:
        return {
            "route_label": self.route_label,
            "method": METHOD_NAME,
            "probe_id": int(self.probe_id),
            "stage": self.stage,
            "probe_role": self.role,
            "center_pos": int(self.center_pos),
            "positions": [int(pos) for pos in self.positions],
            "dense_T": int(self.dense_T),
            "predicted_regret": float(self.predicted_regret),
            "source_signal": self.source_signal,
            "bracket_id": None if self.bracket_id is None else int(self.bracket_id),
            "outside_hard_bracket": bool(self.outside_hard_bracket),
            "rank": int(self.rank),
            "reason": self.reason,
            "components": {key: float(value) for key, value in self.components.items()},
        }


@dataclass(frozen=True)
class RbaRbrBudgetConfig:
    min_k: int
    max_k: int
    scaffold_k: int = 4
    min_marginal_regret: float = 0.18
    rescue_quota_fraction: float = 0.30
    risk_satisfied_threshold: float = 0.42
    route_label: str = ROUTE_LABEL

    def __post_init__(self):
        if self.route_label != ROUTE_LABEL:
            raise ValueError(f"invalid budget route label: {self.route_label}")
        if int(self.min_k) < 1 or int(self.max_k) < int(self.min_k):
            raise ValueError("budget requires 1 <= min_k <= max_k")
        if int(self.scaffold_k) < 1:
            raise ValueError("scaffold_k must be positive")


@dataclass
class RbaRbrSelectionResult:
    selected_positions: List[int]
    selected_probes: List[ProbeCandidate]
    selection_rows: List[Dict[str, object]]
    deploy_ledger: Dict[str, object]
    stop_reason: str
