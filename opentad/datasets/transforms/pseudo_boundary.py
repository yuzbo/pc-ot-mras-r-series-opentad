import hashlib
import json
from pathlib import Path

import numpy as np


def _stable_string_seed(value):
    if value is None:
        value = "unknown"
    if not isinstance(value, str):
        value = str(value)
    digest = hashlib.sha1(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def _as_score_array(boundary_scores):
    if boundary_scores is None:
        return None
    scores = np.asarray(boundary_scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        return None
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    return scores


def _target_unit_count(target_frame_num, group_size):
    target_frame_num = int(max(target_frame_num, 0))
    group_size = int(max(group_size, 1))
    if target_frame_num == 0:
        return 0
    if group_size == 1:
        return target_frame_num
    return max(target_frame_num // group_size, 1)


def _num_units(valid_len, group_size):
    valid_len = int(max(valid_len, 0))
    group_size = int(max(group_size, 1))
    return int(np.ceil(valid_len / float(group_size)))


def _expand_units(units, valid_len, group_size):
    valid_len = int(max(valid_len, 0))
    group_size = int(max(group_size, 1))
    if valid_len == 0:
        return np.zeros((0,), dtype=np.int64)
    units = np.asarray(units, dtype=np.int64).reshape(-1)
    if units.size == 0:
        return np.zeros((0,), dtype=np.int64)
    if group_size == 1:
        positions = units[(units >= 0) & (units < valid_len)]
        return np.sort(np.unique(positions.astype(np.int64)))

    expanded = []
    max_unit = _num_units(valid_len, group_size)
    for unit in np.sort(np.unique(units)):
        if unit < 0 or unit >= max_unit:
            continue
        start = int(unit) * group_size
        end = min(valid_len, start + group_size)
        if start < end:
            expanded.append(np.arange(start, end, dtype=np.int64))
    if len(expanded) == 0:
        return np.zeros((0,), dtype=np.int64)
    return np.concatenate(expanded)


def _positions_to_units(positions, valid_len, group_size):
    valid_len = int(max(valid_len, 0))
    group_size = int(max(group_size, 1))
    positions = np.asarray(positions, dtype=np.int64).reshape(-1)
    if positions.size == 0 or valid_len == 0:
        return np.zeros((0,), dtype=np.int64)
    positions = positions[(positions >= 0) & (positions < valid_len)]
    if positions.size == 0:
        return np.zeros((0,), dtype=np.int64)
    if group_size == 1:
        return np.sort(np.unique(positions.astype(np.int64)))
    units = positions // group_size
    units = units[(units >= 0) & (units < _num_units(valid_len, group_size))]
    return np.sort(np.unique(units.astype(np.int64)))


def _finalize_positions(primary_positions, valid_len, target_frame_num, sample_key, group_size=1):
    valid_len = int(max(valid_len, 0))
    if valid_len == 0:
        return np.zeros((0,), dtype=np.int64)

    group_size = int(max(group_size, 1))
    target_units = min(_target_unit_count(target_frame_num, group_size), _num_units(valid_len, group_size))
    if target_units <= 0:
        return np.zeros((0,), dtype=np.int64)

    primary_units = _positions_to_units(primary_positions, valid_len, group_size)
    selected = []
    seen = set()
    for unit in primary_units.tolist():
        if 0 <= unit < _num_units(valid_len, group_size) and unit not in seen:
            selected.append(unit)
            seen.add(unit)
        if len(selected) >= target_units:
            break

    if len(selected) < target_units:
        fallback = select_random_fixed_positions(valid_len, target_frame_num, sample_key, group_size=group_size)
        fallback_units = _positions_to_units(fallback, valid_len, group_size)
        for unit in fallback_units.tolist():
            if unit not in seen:
                selected.append(unit)
                seen.add(unit)
            if len(selected) >= target_units:
                break

    if len(selected) < target_units:
        for unit in range(_num_units(valid_len, group_size)):
            if unit not in seen:
                selected.append(unit)
                seen.add(unit)
            if len(selected) >= target_units:
                break

    return _expand_units(np.asarray(selected, dtype=np.int64), valid_len, group_size)


def select_random_fixed_positions(valid_len, target_frame_num, sample_key, group_size=1):
    valid_len = int(max(valid_len, 0))
    if valid_len == 0:
        return np.zeros((0,), dtype=np.int64)

    group_size = int(max(group_size, 1))
    unit_count = _num_units(valid_len, group_size)
    target_units = min(_target_unit_count(target_frame_num, group_size), unit_count)
    if target_units <= 0:
        return np.zeros((0,), dtype=np.int64)
    if target_units >= unit_count:
        return np.arange(valid_len, dtype=np.int64)

    rng = np.random.default_rng(_stable_string_seed(sample_key))
    selected_units = np.sort(rng.choice(unit_count, size=target_units, replace=False).astype(np.int64))
    return _expand_units(selected_units, valid_len, group_size)


def load_boundary_scores(cache_dir, video_name):
    if cache_dir is None or str(cache_dir) == "":
        return None

    cache_root = Path(cache_dir)
    manifest_path = cache_root / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        if bool(manifest.get("uses_gt", False)):
            raise ValueError(f"pseudo-boundary cache must not use GT: {manifest_path}")
        if bool(manifest.get("uses_oracle", False)):
            raise ValueError(f"pseudo-boundary cache must not use oracle labels: {manifest_path}")

    npz_path = cache_root / f"{video_name}.npz"
    if not npz_path.exists():
        return None

    with np.load(npz_path) as data:
        for key in ("boundary_score", "boundary_scores", "scores", "score"):
            if key in data:
                return _as_score_array(data[key])
    raise KeyError(f"pseudo-boundary cache missing boundary_score array: {npz_path}")


def slice_global_scores_for_window(boundary_scores, global_indices):
    scores = _as_score_array(boundary_scores)
    if scores is None:
        return None

    indices = np.asarray(list(global_indices), dtype=np.int64).reshape(-1)
    window_scores = np.zeros(indices.shape[0], dtype=np.float64)
    valid = (indices >= 0) & (indices < scores.size)
    if np.any(valid):
        window_scores[valid] = scores[indices[valid]]
    return window_scores


def _rank_boundary_positions(boundary_scores, valid_len, pseudo_min_score):
    scores = _as_score_array(boundary_scores)
    if scores is None:
        return np.zeros((0,), dtype=np.int64)

    valid_len = int(max(valid_len, 0))
    if valid_len == 0:
        return np.zeros((0,), dtype=np.int64)
    local_scores = scores[:valid_len]
    eligible = np.flatnonzero(local_scores >= float(pseudo_min_score))
    if eligible.size == 0:
        return np.zeros((0,), dtype=np.int64)

    order = np.lexsort((eligible, -local_scores[eligible]))
    return eligible[order].astype(np.int64)


def select_pseudo_boundary_hybrid_positions(
    valid_len,
    target_frame_num,
    sample_key,
    boundary_scores,
    pseudo_quota=64,
    pseudo_radius=1,
    pseudo_min_score=0.0,
    fallback="random_fixed",
    group_size=1,
):
    if fallback != "random_fixed":
        raise ValueError(f"Unsupported pseudo-boundary fallback: {fallback}")

    ranked = _rank_boundary_positions(boundary_scores, valid_len, pseudo_min_score)
    if ranked.size == 0 or int(pseudo_quota) <= 0:
        return select_random_fixed_positions(valid_len, target_frame_num, sample_key, group_size=group_size)

    quota = int(max(pseudo_quota, 0))
    radius = int(max(pseudo_radius, 0))
    primary = []
    seen = set()
    for center in ranked.tolist():
        for pos in range(center - radius, center + radius + 1):
            if 0 <= pos < int(valid_len) and pos not in seen:
                primary.append(pos)
                seen.add(pos)
            if len(primary) >= quota:
                break
        if len(primary) >= quota:
            break

    return _finalize_positions(primary, valid_len, target_frame_num, sample_key, group_size=group_size)


def select_pseudo_boundary_snap_positions(
    valid_len,
    target_frame_num,
    sample_key,
    boundary_scores,
    pseudo_quota=64,
    pseudo_snap_distance=2,
    pseudo_min_score=0.0,
    fallback="random_fixed",
    group_size=1,
):
    if fallback != "random_fixed":
        raise ValueError(f"Unsupported pseudo-boundary fallback: {fallback}")

    base = select_random_fixed_positions(valid_len, target_frame_num, sample_key, group_size=group_size)
    ranked = _rank_boundary_positions(boundary_scores, valid_len, pseudo_min_score)
    if ranked.size == 0 or int(pseudo_quota) <= 0:
        return base

    snap_distance = int(max(pseudo_snap_distance, 0))
    selected = set(base.tolist())
    replaced = set()
    for candidate in ranked.tolist():
        if len(replaced) >= int(pseudo_quota):
            break
        if candidate in selected:
            continue
        if len(selected) == 0:
            continue
        source = min(selected, key=lambda pos: (abs(pos - candidate), pos))
        if abs(source - candidate) > snap_distance or source in replaced:
            continue
        selected.remove(source)
        selected.add(int(candidate))
        replaced.add(source)

    return _finalize_positions(sorted(selected), valid_len, target_frame_num, sample_key, group_size=group_size)
