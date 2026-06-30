import numpy as np

from .types import RbaRbrRiskMap, as_float_array, sorted_unique_positions


def _normalize(values):
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi <= lo + 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _gradient_signal(values):
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size <= 1:
        return np.zeros_like(arr)
    return _normalize(np.abs(np.gradient(arr)))


def _staleness_from_observed(dense_T, observed_positions):
    dense_T = int(dense_T)
    if not observed_positions:
        return np.ones(dense_T, dtype=np.float64)
    observed = np.asarray(sorted_unique_positions(observed_positions, dense_T), dtype=np.int64)
    positions = np.arange(dense_T, dtype=np.int64)
    dist = np.min(np.abs(positions[:, None] - observed[None, :]), axis=1)
    return _normalize(dist)


def build_risk_map(
    actionness,
    uncertainty=None,
    transition=None,
    observed_positions=None,
    source="deploy_visible_metadata_preview",
):
    actionness = as_float_array(actionness, "actionness")
    dense_T = int(actionness.size)
    if uncertainty is None:
        uncertainty = 1.0 - np.abs(actionness - 0.5) * 2.0
    uncertainty = as_float_array(uncertainty, "uncertainty", dense_T=dense_T)
    if transition is None:
        transition = _gradient_signal(actionness)
    transition = as_float_array(transition, "transition", dense_T=dense_T)
    staleness = _staleness_from_observed(dense_T, observed_positions or [])
    conflict = np.clip(uncertainty * (0.35 + actionness) + transition * (1.0 - actionness), 0.0, 1.0)
    gap_risk = np.clip(0.55 * staleness + 0.25 * uncertainty + 0.20 * transition, 0.0, 1.0)
    soft_prior = np.clip(0.55 * actionness + 0.25 * transition + 0.20 * uncertainty, 0.0, 1.0)
    rescue_score = np.clip(
        0.28 * uncertainty + 0.28 * transition + 0.18 * staleness + 0.16 * conflict + 0.10 * gap_risk,
        0.0,
        1.0,
    )
    return RbaRbrRiskMap(
        dense_T=dense_T,
        actionness=actionness,
        uncertainty=uncertainty,
        transition=transition,
        staleness=staleness,
        conflict=conflict,
        gap_risk=gap_risk,
        soft_bracket_prior=soft_prior,
        rescue_score=rescue_score,
        source=source,
    )
