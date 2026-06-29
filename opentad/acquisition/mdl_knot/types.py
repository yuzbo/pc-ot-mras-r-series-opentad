from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np


MDL_KNOT_ROUTE_LABEL = "DIVERGENT_INNOVATION_MDL_KNOT_DO_NOT_MERGE_WITH_C3"

FORBIDDEN_SOURCE_TOKENS = (
    "C3",
    "GLOBALRANK",
    "INTERVAL",
    "TEACHER",
    "CACHE",
    "VAL_GT",
    "TEST_GT",
    "ORACLE",
)

ROLE_TO_ID = {
    "endpoint_anchor": 0,
    "scaffold_anchor": 1,
    "mdl_knot": 2,
    "transition_guard": 3,
    "gap_guard": 4,
    "short_risk_guard": 5,
    "stable_span_knot": 6,
}


def _as_float_list(values: Iterable[float], name: str) -> List[float]:
    out = [float(v) for v in values]
    if len(out) == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(np.asarray(out, dtype=np.float64)).all():
        raise ValueError(f"{name} contains non-finite values")
    return out


def reject_forbidden_source_tokens(value: str, field_name: str = "source") -> None:
    text = str(value or "").upper()
    for token in FORBIDDEN_SOURCE_TOKENS:
        if token in text:
            raise ValueError(f"forbidden route/source token {token} in {field_name}: {value}")


@dataclass
class MDLKnotConfig:
    route_label: str = MDL_KNOT_ROUTE_LABEL
    scout_source: str = "deploy_scout"
    min_k: int = 4
    max_k: int = 384
    min_anchor_k: int = 4
    target_weighted_error: float = 0.02
    min_marginal_gain: float = 1.0e-4
    max_gap: int = 32
    transition_guard_radius: int = 2
    short_island_max_width: int = 10
    short_island_min_knots: int = 2
    island_threshold: float = 0.45
    transition_threshold: float = 0.18
    candidate_limit: int = 96
    residual_candidate_count: int = 24
    signal_candidate_count: int = 16
    lambda_k: float = 2.0e-4
    lambda_seg: float = 1.0e-4
    lambda_gap: float = 0.02
    lambda_duration: float = 0.025
    lambda_transition: float = 0.03
    weight_gradient: float = 1.5
    weight_uncertainty: float = 0.8
    weight_change: float = 1.0
    weight_short: float = 1.2
    bonus_transition_guard: float = 0.015
    bonus_gap_guard: float = 0.010
    bonus_short_risk_guard: float = 0.020
    random_seed: int = 0

    def __post_init__(self) -> None:
        if self.route_label != MDL_KNOT_ROUTE_LABEL:
            raise ValueError(f"route_label must be {MDL_KNOT_ROUTE_LABEL}")
        reject_forbidden_source_tokens(self.scout_source, "scout_source")
        if self.min_k < 2:
            raise ValueError("min_k must be at least 2")
        if self.max_k < self.min_k:
            raise ValueError("max_k must be >= min_k")
        if self.max_gap < 1:
            raise ValueError("max_gap must be positive")


@dataclass
class ScoutCurve:
    p_action: List[float]
    uncertainty: List[float]
    temporal_change: List[float]
    persistence: List[float]
    time_index: List[int]
    source: str = "deploy_scout"
    motion: Optional[List[float]] = None
    provenance: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        reject_forbidden_source_tokens(self.source, "ScoutCurve.source")
        self.p_action = _as_float_list(self.p_action, "p_action")
        self.uncertainty = _as_float_list(self.uncertainty, "uncertainty")
        self.temporal_change = _as_float_list(self.temporal_change, "temporal_change")
        self.persistence = _as_float_list(self.persistence, "persistence")
        self.time_index = [int(v) for v in self.time_index]
        lengths = {
            len(self.p_action),
            len(self.uncertainty),
            len(self.temporal_change),
            len(self.persistence),
            len(self.time_index),
        }
        if self.motion is not None:
            self.motion = _as_float_list(self.motion, "motion")
            lengths.add(len(self.motion))
        if len(lengths) != 1:
            raise ValueError(f"ScoutCurve fields must have equal length, got {sorted(lengths)}")
        default_provenance = {
            "uses_gt": False,
            "uses_teacher": False,
            "uses_prediction_cache": False,
            "dense_raw_backbone_handoff": False,
            "selected_inputs_is_gathered": True,
            "position_unit": "original_dense_time_index",
        }
        default_provenance.update(self.provenance or {})
        self.provenance = default_provenance

    @property
    def dense_t(self) -> int:
        return len(self.p_action)

    def as_matrix(self) -> np.ndarray:
        columns = [
            np.asarray(self.p_action, dtype=np.float64),
            np.asarray(self.uncertainty, dtype=np.float64),
            np.asarray(self.temporal_change, dtype=np.float64),
            np.asarray(self.persistence, dtype=np.float64),
        ]
        if self.motion is not None:
            columns.append(np.asarray(self.motion, dtype=np.float64))
        return np.stack(columns, axis=1)


@dataclass
class MDLObjectiveTerms:
    weighted_reconstruction_error: float
    mean_reconstruction_error: float
    complexity_penalty: float
    long_gap_risk: float
    duration_risk: float
    transition_risk: float
    total_cost: float
    max_gap: int
    gap_p95: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "weighted_reconstruction_error": float(self.weighted_reconstruction_error),
            "mean_reconstruction_error": float(self.mean_reconstruction_error),
            "complexity_penalty": float(self.complexity_penalty),
            "long_gap_risk": float(self.long_gap_risk),
            "duration_risk": float(self.duration_risk),
            "transition_risk": float(self.transition_risk),
            "total_cost": float(self.total_cost),
            "max_gap": int(self.max_gap),
            "gap_p95": float(self.gap_p95),
        }


@dataclass
class SparseTemporalMeta:
    selected_positions: List[int]
    original_dense_t: int
    gap_left: List[float]
    gap_right: List[float]
    visibility_mask: List[bool]
    role_id: List[int]
    valid_k: int
    position_unit: str = "original_dense_time_index"

    def to_dict(self) -> Dict[str, object]:
        return {
            "selected_positions": list(self.selected_positions),
            "original_dense_T": int(self.original_dense_t),
            "gap_left": [float(v) for v in self.gap_left],
            "gap_right": [float(v) for v in self.gap_right],
            "visibility_mask": [bool(v) for v in self.visibility_mask],
            "role_id": [int(v) for v in self.role_id],
            "valid_k": int(self.valid_k),
            "position_unit": self.position_unit,
        }


@dataclass
class KnotLedger:
    video_id: str
    window_id: int
    dense_t: int
    selected_positions: List[int]
    selected_roles: List[str]
    selected_scores: List[float]
    target_k: int
    valid_k: int
    stop_reason: str
    mean_reconstruction_error: float
    weighted_reconstruction_error: float
    max_gap: int
    gap_p95: float
    estimated_islands: List[Dict[str, object]]
    transition_bands: List[Dict[str, object]]
    budget_bin: str
    objective_terms: Dict[str, float]
    provenance: Dict[str, object]
    route_label: str = MDL_KNOT_ROUTE_LABEL
    position_unit: str = "original_dense_time_index"
    control_name: str = "mdl_plus_transition_gap_duration"
    selection_history: List[Dict[str, object]] = field(default_factory=list)

    def to_sparse_meta(self) -> SparseTemporalMeta:
        positions = list(self.selected_positions)
        gap_left = []
        gap_right = []
        for idx, pos in enumerate(positions):
            prev_pos = positions[idx - 1] if idx > 0 else pos
            next_pos = positions[idx + 1] if idx + 1 < len(positions) else pos
            gap_left.append(float(pos - prev_pos))
            gap_right.append(float(next_pos - pos))
        role_id = [ROLE_TO_ID.get(role, ROLE_TO_ID["mdl_knot"]) for role in self.selected_roles]
        return SparseTemporalMeta(
            selected_positions=positions,
            original_dense_t=self.dense_t,
            gap_left=gap_left,
            gap_right=gap_right,
            visibility_mask=[True] * self.valid_k,
            role_id=role_id,
            valid_k=self.valid_k,
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "route_label": self.route_label,
            "video_id": self.video_id,
            "window_id": int(self.window_id),
            "dense_T": int(self.dense_t),
            "selected_positions": list(self.selected_positions),
            "selected_roles": list(self.selected_roles),
            "selected_scores": [float(v) for v in self.selected_scores],
            "target_k": int(self.target_k),
            "actual_k": int(self.valid_k),
            "valid_k": int(self.valid_k),
            "stop_reason": self.stop_reason,
            "mean_reconstruction_error": float(self.mean_reconstruction_error),
            "weighted_reconstruction_error": float(self.weighted_reconstruction_error),
            "max_gap": int(self.max_gap),
            "gap_p95": float(self.gap_p95),
            "estimated_islands": list(self.estimated_islands),
            "transition_bands": list(self.transition_bands),
            "budget_bin": self.budget_bin,
            "objective_terms": dict(self.objective_terms),
            "provenance": dict(self.provenance),
            "position_unit": self.position_unit,
            "control_name": self.control_name,
            "selection_history": list(self.selection_history),
        }
