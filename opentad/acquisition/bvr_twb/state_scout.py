import numpy as np

from .types import FORBIDDEN_DEPLOY_KEYS, ScoutCurves, as_float_array


def _minmax01(values):
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return arr
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi <= lo + 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _moving_average(values, radius):
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    radius = int(max(radius, 0))
    if radius == 0 or arr.size <= 1:
        return arr.copy()
    padded = np.pad(arr, (radius, radius), mode="edge")
    kernel = np.ones(2 * radius + 1, dtype=np.float64) / float(2 * radius + 1)
    return np.convolve(padded, kernel, mode="valid")


def _scaled_change(values, scale=0.35):
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size <= 1:
        return np.zeros_like(arr)
    change = np.abs(np.gradient(arr))
    return np.clip(change / float(max(scale, 1e-6)), 0.0, 1.0)


def _reject_forbidden_metadata(metadata):
    if not metadata:
        return
    lower_keys = {str(key).lower() for key in metadata.keys()}
    for forbidden in FORBIDDEN_DEPLOY_KEYS:
        if forbidden.lower() in lower_keys:
            raise ValueError(f"StateScout deploy metadata contains forbidden key: {forbidden}")


def build_scout_from_actionness(
    p_action,
    p_background=None,
    uncertainty=None,
    transition_score=None,
    persistence=None,
    short_action_risk=None,
    motion_signal=None,
    metadata=None,
):
    """Build deterministic deploy-visible scout curves from synthetic preview signals."""

    _reject_forbidden_metadata(metadata)
    action = np.clip(as_float_array(p_action, "p_action"), 0.0, 1.0)
    dense_T = int(action.shape[0])
    motion = None if motion_signal is None else np.clip(as_float_array(motion_signal, "motion_signal"), 0.0, 1.0)
    if motion is not None and motion.shape[0] != dense_T:
        raise ValueError("motion_signal length must match p_action")

    if p_background is None:
        background = 1.0 - action
    else:
        background = np.clip(as_float_array(p_background, "p_background"), 0.0, 1.0)
    if background.shape[0] != dense_T:
        raise ValueError("p_background length must match p_action")

    if uncertainty is None:
        entropy_like = 1.0 - np.abs(2.0 * action - 1.0)
        if motion is not None:
            motion_change = _minmax01(np.abs(np.gradient(motion)))
            entropy_like = np.clip(0.80 * entropy_like + 0.20 * motion_change, 0.0, 1.0)
        uncertainty_arr = entropy_like
    else:
        uncertainty_arr = np.clip(as_float_array(uncertainty, "uncertainty"), 0.0, 1.0)

    if transition_score is None:
        action_change = _scaled_change(action, scale=0.28)
        if motion is not None:
            motion_change = _scaled_change(motion, scale=0.35)
            transition_arr = np.clip(0.70 * action_change + 0.30 * motion_change, 0.0, 1.0)
        else:
            transition_arr = action_change
    else:
        transition_arr = np.clip(as_float_array(transition_score, "transition_score"), 0.0, 1.0)

    if persistence is None:
        smooth = _moving_average(action, radius=max(1, dense_T // 32))
        local_change = _minmax01(np.abs(action - smooth))
        persistence_arr = np.clip(1.0 - local_change, 0.0, 1.0)
    else:
        persistence_arr = np.clip(as_float_array(persistence, "persistence"), 0.0, 1.0)

    if short_action_risk is None:
        smooth3 = _moving_average(action, radius=1)
        smooth_long = _moving_average(action, radius=max(2, dense_T // 16))
        local_peak = np.clip(action - smooth_long, 0.0, 1.0)
        edge = _minmax01(np.abs(np.gradient(smooth3)))
        short_risk_arr = np.clip(0.60 * local_peak + 0.40 * edge * uncertainty_arr, 0.0, 1.0)
    else:
        short_risk_arr = np.clip(as_float_array(short_action_risk, "short_action_risk"), 0.0, 1.0)

    for name, arr in (
        ("uncertainty", uncertainty_arr),
        ("transition_score", transition_arr),
        ("persistence", persistence_arr),
        ("short_action_risk", short_risk_arr),
    ):
        if arr.shape[0] != dense_T:
            raise ValueError(f"{name} length must match p_action")

    return ScoutCurves(
        dense_T=dense_T,
        p_action=action,
        p_background=background,
        uncertainty=uncertainty_arr,
        transition_score=transition_arr,
        persistence=persistence_arr,
        short_action_risk=short_risk_arr,
        preview_signal=motion,
    )
