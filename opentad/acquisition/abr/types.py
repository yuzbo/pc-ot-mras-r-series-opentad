from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


ABR_ROUTE_LABEL = "DIVERGENT_INNOVATION_ABR_DO_NOT_MERGE_WITH_C3"

DEFAULT_PROVENANCE = {
    "uses_gt": False,
    "uses_teacher": False,
    "uses_prediction_cache": False,
    "uses_detector_feedback": False,
    "dense_raw_backbone_handoff": False,
    "selected_inputs_is_gathered": True,
}


@dataclass(frozen=True)
class ABRConfig:
    k0: int = 12
    k1_cap: int = 16
    k2_cap: int = 4
    max_total_k: int = 64
    max_gap: int = 32
    target_frame_num: Optional[int] = None
    round2_enabled: bool = True
    deadline_ms: float = 80.0
    action_threshold: float = 0.60
    background_threshold: float = 0.35
    uncertainty_band: float = 0.16
    derivative_threshold: float = 0.24
    resolve_width: int = 3
    round2_min_width: int = 6
    outside_witness_offset: int = 2
    stale_decay: float = 0.50
    keep_stale_confidence: float = 0.15
    max_children_per_bracket: int = 2
    scout_cost_ms_per_position: float = 0.015
    acquisition_cost_ms_per_position: float = 0.010
    route_label: str = ABR_ROUTE_LABEL
    allow_diagnostic_fallback_scout: bool = False
    fallback_stage: str = "FORMAL_LOCKED"


@dataclass
class BracketState:
    bracket_id: int
    kind: str
    left: int
    right: int
    parent_id: Optional[int] = None
    round_created: int = 0
    last_updated_round: int = 0
    confidence: float = 0.5
    uncertainty: float = 0.5
    state_left: Optional[str] = None
    state_right: Optional[str] = None
    has_pre_background_witness: bool = False
    has_action_core_witness: bool = False
    has_post_background_witness: bool = False
    priority: float = 0.0
    status: str = "active"

    @property
    def width(self) -> int:
        return max(0, int(self.right) - int(self.left))

    def width_seconds(self, fps: float) -> float:
        return self.width / max(float(fps), 1e-6)

    def to_dict(self, fps: float = 1.0) -> Dict[str, Any]:
        payload = asdict(self)
        payload["width"] = self.width
        payload["width_seconds"] = self.width_seconds(fps)
        return payload


@dataclass
class ABRRoundLedger:
    video_id: str
    window_id: str
    dense_T: int
    fps: float
    round_id: int
    selected_positions: List[int]
    selected_roles: List[str]
    selected_bracket_ids: List[Optional[int]]
    selected_source: List[str]
    selected_cost_ms: List[float]
    selected_source_detail: List[str]
    cumulative_k: int
    cumulative_scout_ms: float
    deadline_ms: float
    stop_reason: str
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, bool] = field(default_factory=lambda: dict(DEFAULT_PROVENANCE))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ABRCostSummary:
    rounds_used: int
    selected_k: int
    mean_selected_fraction: float
    scout_ms: float
    acquisition_wait_ms: float
    detector_forward_count: int
    deadline_ms: float
    total_latency_proxy_ms: float
    stop_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ABRSelectionResult:
    route_label: str
    selected_positions: List[int]
    selected_rounds: List[int]
    selected_roles: List[str]
    selected_bracket_ids: List[int]
    valid_k: int
    dense_T: int
    fps: float
    round_ledgers: List[ABRRoundLedger]
    brackets: List[BracketState]
    cost: ABRCostSummary
    provenance: Dict[str, bool]
    scout_source: str
    diagnostic_fallback_used: bool
    config: ABRConfig

    def to_dict(self) -> Dict[str, Any]:
        return {
            "route_label": self.route_label,
            "selected_positions": list(self.selected_positions),
            "selected_rounds": list(self.selected_rounds),
            "selected_roles": list(self.selected_roles),
            "selected_bracket_ids": list(self.selected_bracket_ids),
            "valid_k": int(self.valid_k),
            "dense_T": int(self.dense_T),
            "fps": float(self.fps),
            "round_ledgers": [ledger.to_dict() for ledger in self.round_ledgers],
            "brackets": [bracket.to_dict(self.fps) for bracket in self.brackets],
            "cost": self.cost.to_dict(),
            "provenance": dict(self.provenance),
            "scout_source": self.scout_source,
            "diagnostic_fallback_used": bool(self.diagnostic_fallback_used),
            "config": asdict(self.config),
        }
