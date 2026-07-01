import json
import math
from pathlib import Path

import numpy as np


COARSE_SCORE_REQUIRED_ROUTE_LABELS = {"C3_MAINLINE_OPTIMIZATION", "C3_ORIGINAL_OPTIMIZATION_ROUTE"}


def _stable_string_seed(value):
    text = str(value)
    seed = 0
    for ch in text:
        seed = (seed * 131 + ord(ch)) % (2**32 - 1)
    return seed


def _sigmoid(x):
    x = np.asarray(x, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _safe_scores(values, valid_len):
    scores = np.asarray(values, dtype=np.float32).reshape(-1)
    if scores.size < valid_len:
        if scores.size == 0:
            scores = np.zeros((valid_len,), dtype=np.float32)
        else:
            scores = np.pad(scores, (0, valid_len - scores.size), mode="edge")
    return scores[:valid_len]


def _top_positions(scores, count):
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    count = min(max(int(count), 0), scores.size)
    if count <= 0 or scores.size == 0:
        return np.zeros((0,), dtype=np.int64)
    order = np.argsort(-scores, kind="mergesort")[:count]
    return np.sort(order.astype(np.int64))


def _uniform_pick_indices(candidates, count):
    candidates = np.asarray(candidates, dtype=np.int64)
    candidates = np.unique(candidates[candidates >= 0])
    count = min(max(int(count), 0), candidates.size)
    if count <= 0:
        return np.zeros((0,), dtype=np.int64)
    if count >= candidates.size:
        return candidates
    pick = np.linspace(0, candidates.size - 1, num=count)
    return candidates[np.rint(pick).astype(np.int64)]


def _positions_to_units(positions, valid_len, group_size):
    if group_size <= 1:
        return np.asarray(positions, dtype=np.int64)
    positions = np.asarray(positions, dtype=np.int64)
    positions = positions[(positions >= 0) & (positions < valid_len)]
    return np.unique(positions // group_size)


def _expand_units(units, valid_len, group_size):
    units = np.asarray(units, dtype=np.int64)
    if group_size <= 1:
        return np.unique(units[(units >= 0) & (units < valid_len)]).astype(np.int64)
    expanded = []
    for unit in units:
        start = int(unit) * int(group_size)
        end = min(start + int(group_size), int(valid_len))
        expanded.extend(range(start, end))
    return np.asarray(sorted(set(x for x in expanded if 0 <= x < valid_len)), dtype=np.int64)


def _target_unit_count(target_frame_num, group_size):
    return int(math.ceil(float(target_frame_num) / float(max(group_size, 1))))


def derive_coarse_score_fields(action_score=None, action_logit=None, valid_len=None):
    if action_score is None and action_logit is None:
        raise ValueError("coarse score cache requires action_score or action_logit")
    if valid_len is None:
        valid_len = len(action_score) if action_score is not None else len(action_logit)
    if action_score is None:
        p_action = _sigmoid(_safe_scores(action_logit, valid_len))
    else:
        p_action = np.clip(_safe_scores(action_score, valid_len), 0.0, 1.0)
    entropy = -(p_action * np.log(p_action + 1e-6) + (1.0 - p_action) * np.log(1.0 - p_action + 1e-6))
    change = np.zeros_like(p_action, dtype=np.float32)
    if valid_len > 1:
        change[1:] = np.abs(p_action[1:] - p_action[:-1])
    return {
        "action_score": p_action.astype(np.float32),
        "uncertainty": entropy.astype(np.float32),
        "change": change.astype(np.float32),
    }


def validate_coarse_score_cache_manifest(manifest_path, expected_axis="global_snippet_index", allow_gt=False):
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing coarse score manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if bool(manifest.get("uses_gt", False)) and not allow_gt:
        raise AssertionError("coarse score cache manifest uses_gt=true is forbidden for deploy selection")
    axis = manifest.get("axis", None)
    if axis != expected_axis:
        raise AssertionError(f"coarse score cache axis mismatch: expected {expected_axis}, got {axis}")
    route_labels = set(manifest.get("route_labels", []))
    if not COARSE_SCORE_REQUIRED_ROUTE_LABELS.issubset(route_labels):
        raise AssertionError("coarse score cache manifest missing required C3 route labels")
    videos = manifest.get("videos", None)
    if not isinstance(videos, dict):
        raise AssertionError("coarse score cache manifest requires a videos mapping")
    return manifest


def load_coarse_score_cache(cache_dir, video_name, expected_axis="global_snippet_index"):
    cache_dir = Path(cache_dir)
    manifest = validate_coarse_score_cache_manifest(cache_dir / "manifest.json", expected_axis=expected_axis)
    video_entry = manifest.get("videos", {}).get(video_name)
    if video_entry is None:
        raise FileNotFoundError(f"coarse score cache missing video entry: {video_name}")
    score_path = cache_dir / video_entry.get("file", f"{video_name}.npz")
    if not score_path.is_file():
        raise FileNotFoundError(f"coarse score cache missing npz for {video_name}: {score_path}")
    with np.load(score_path, allow_pickle=False) as data:
        raw = {key: data[key] for key in data.files}
    if "action_score" not in raw and "action_logit" not in raw:
        raise AssertionError(f"coarse score cache npz for {video_name} requires action_score or action_logit")
    return raw, manifest


def slice_global_scores_for_window(raw_scores, global_indices):
    global_indices = np.asarray(global_indices, dtype=np.int64)
    if global_indices.size == 0:
        return derive_coarse_score_fields(action_score=np.zeros((0,), dtype=np.float32), valid_len=0)
    max_index = int(global_indices.max()) if global_indices.size else -1
    fields = {}
    for key in ("action_score", "action_logit"):
        if key in raw_scores:
            values = _safe_scores(raw_scores[key], max_index + 1)
            fields[key] = values[np.clip(global_indices, 0, values.size - 1)]
    return derive_coarse_score_fields(
        action_score=fields.get("action_score"),
        action_logit=fields.get("action_logit"),
        valid_len=global_indices.size,
    )


def select_coarse_oracle_shell_positions(
    valid_len,
    target_frame_num,
    action_score=None,
    action_logit=None,
    sample_key="coarse_oracle_shell",
    min_action_score=0.35,
    transition_top_fraction=0.15,
    transition_top_count=None,
    transition_radius=2,
    uncertainty_weight=0.4,
    change_weight=0.6,
    group_size=1,
):
    del sample_key
    valid_len = int(valid_len)
    if valid_len <= 0 or target_frame_num <= 0:
        return np.zeros((0,), dtype=np.int64)
    group_size = max(int(group_size), 1)
    total_units = int(math.ceil(float(valid_len) / float(group_size)))
    target_units = min(_target_unit_count(target_frame_num, group_size), total_units)
    if target_units >= total_units:
        return np.arange(valid_len, dtype=np.int64)

    fields = derive_coarse_score_fields(action_score=action_score, action_logit=action_logit, valid_len=valid_len)
    p_action = fields["action_score"]
    uncertainty = fields["uncertainty"]
    change = fields["change"]
    transition_score = float(uncertainty_weight) * uncertainty + float(change_weight) * change

    if transition_top_count is None:
        transition_top_count = int(math.ceil(valid_len * float(transition_top_fraction)))
    transition_seeds = _top_positions(transition_score, transition_top_count)
    transition_positions = set()
    radius = max(int(transition_radius), 0)
    for seed in transition_seeds:
        lo = max(0, int(seed) - radius)
        hi = min(valid_len, int(seed) + radius + 1)
        transition_positions.update(range(lo, hi))
    transition = np.asarray(sorted(transition_positions), dtype=np.int64)
    action = np.where(p_action >= float(min_action_score))[0].astype(np.int64)
    action = np.setdiff1d(action, transition, assume_unique=False)
    all_positions = np.arange(valid_len, dtype=np.int64)
    background = np.setdiff1d(all_positions, np.union1d(transition, action), assume_unique=False)

    groups = [
        _positions_to_units(transition, valid_len, group_size),
        _positions_to_units(action, valid_len, group_size),
        _positions_to_units(background, valid_len, group_size),
    ]
    selected = []
    remaining = target_units
    for group in groups:
        if remaining <= 0:
            break
        chosen = _uniform_pick_indices(group, min(remaining, group.size))
        if chosen.size > 0:
            selected.append(chosen)
            remaining -= chosen.size
    if selected:
        selected_units = np.unique(np.concatenate(selected)).astype(np.int64)
    else:
        selected_units = np.zeros((0,), dtype=np.int64)
    return _expand_units(np.sort(selected_units), valid_len, group_size)[: int(target_frame_num)].astype(np.int64)
