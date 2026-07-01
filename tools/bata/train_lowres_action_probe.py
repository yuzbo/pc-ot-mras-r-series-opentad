from __future__ import annotations

import copy
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_READER_TYPE = "PCOTMRASCoarseActionnessFrameScout"
DEFAULT_PROBE_CONFIG = (
    "configs/adatad/thumos/"
    "pc_ot_mras_a_uniform_scaffold_small_actionness_strict_maxgap_c3_physical_grid_actionformer_n16r4.py"
)
SUPPORTED_C3_READER_TYPES = {
    "PCOTMRASBoundaryDifficultyTemporalFrameScout",
    "PCOTMRASCoarseActionnessFrameScout",
}
SUPPORTED_TCN_VARIANTS = ("lite", "dilated", "multiscale", "motion", "residual", "gated")


def _as_nested_list(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def build_action_targets(valid: Any, gt_segments: Sequence[Any]) -> list[list[float]]:
    """Build frame/snippet action-vs-background targets from window-local GT segments."""

    valid_rows = _as_nested_list(valid)
    gt_rows = [_as_nested_list(row) for row in gt_segments]
    if len(valid_rows) != len(gt_rows):
        raise ValueError("valid and gt_segments batch sizes must match")

    targets: list[list[float]] = []
    for valid_row, segments in zip(valid_rows, gt_rows):
        row: list[float] = []
        for frame_idx, is_valid in enumerate(valid_row):
            if not bool(is_valid):
                row.append(0.0)
                continue
            center = float(frame_idx) + 0.5
            inside = False
            for segment in segments:
                if len(segment) != 2:
                    raise ValueError("each gt segment must contain [start, end]")
                start, end = float(segment[0]), float(segment[1])
                if start <= center <= end:
                    inside = True
                    break
            row.append(1.0 if inside else 0.0)
        targets.append(row)
    return targets


def _flatten_valid(logits: Any, target: Any, valid: Any) -> tuple[list[float], list[int]]:
    logits_rows = _as_nested_list(logits)
    target_rows = _as_nested_list(target)
    valid_rows = _as_nested_list(valid)
    scores: list[float] = []
    labels: list[int] = []
    for logit_row, target_row, valid_row in zip(logits_rows, target_rows, valid_rows):
        if not (len(logit_row) == len(target_row) == len(valid_row)):
            raise ValueError("logits, target, and valid must share batch/time shape")
        for logit, label, is_valid in zip(logit_row, target_row, valid_row):
            if bool(is_valid):
                scores.append(float(logit))
                labels.append(1 if float(label) >= 0.5 else 0)
    if not scores:
        raise ValueError("no valid positions available for metric computation")
    return scores, labels


def _roc_auc(scores: list[float], labels: list[int]) -> float | None:
    positive_count = int(sum(labels))
    negative_count = int(len(labels) - positive_count)
    if positive_count <= 0 or negative_count <= 0:
        return None

    order = sorted(range(len(scores)), key=lambda idx: scores[idx])
    rank_sum_pos = 0.0
    rank = 1
    cursor = 0
    while cursor < len(order):
        next_cursor = cursor + 1
        score = scores[order[cursor]]
        while next_cursor < len(order) and scores[order[next_cursor]] == score:
            next_cursor += 1
        avg_rank = (rank + rank + (next_cursor - cursor) - 1) / 2.0
        for idx in order[cursor:next_cursor]:
            if labels[idx] == 1:
                rank_sum_pos += avg_rank
        rank += next_cursor - cursor
        cursor = next_cursor

    u_stat = rank_sum_pos - positive_count * (positive_count + 1) / 2.0
    return u_stat / float(positive_count * negative_count)


def _average_precision(scores: list[float], labels: list[int]) -> float | None:
    positive_count = sum(labels)
    if positive_count == 0:
        return None
    order = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    hits = 0
    precision_sum = 0.0
    for rank, idx in enumerate(order, start=1):
        if labels[idx] == 1:
            hits += 1
            precision_sum += hits / float(rank)
    return precision_sum / float(positive_count)


def _best_f1(scores: list[float], labels: list[int]) -> tuple[float | None, float | None]:
    positive_count = sum(labels)
    if positive_count == 0:
        return None, None
    best_f1 = 0.0
    best_threshold = None
    true_positive = 0
    false_positive = 0
    order = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    cursor = 0
    while cursor < len(order):
        threshold = float(scores[order[cursor]])
        next_cursor = cursor + 1
        group_positive = 1 if labels[order[cursor]] == 1 else 0
        while next_cursor < len(order) and scores[order[next_cursor]] == threshold:
            group_positive += 1 if labels[order[next_cursor]] == 1 else 0
            next_cursor += 1
        group_size = next_cursor - cursor
        true_positive += group_positive
        false_positive += group_size - group_positive
        fn = positive_count - true_positive
        precision = true_positive / float(max(true_positive + false_positive, 1))
        recall = true_positive / float(max(true_positive + fn, 1))
        denom = precision + recall
        f1 = 0.0 if denom <= 0.0 else 2.0 * precision * recall / denom
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
        cursor = next_cursor
    return best_f1, best_threshold


def compute_binary_action_metrics(logits: Any, target: Any, valid: Any) -> dict[str, Any]:
    scores, labels = _flatten_valid(logits, target, valid)
    best_f1, best_threshold = _best_f1(scores, labels)
    positive_count = int(sum(labels))
    negative_count = int(len(labels) - positive_count)
    predictions = [1 if score >= 0.0 else 0 for score in scores]
    correct_count = sum(1 for pred, label in zip(predictions, labels) if pred == label)
    true_positive = sum(1 for pred, label in zip(predictions, labels) if pred == 1 and label == 1)
    true_negative = sum(1 for pred, label in zip(predictions, labels) if pred == 0 and label == 0)
    positive_recall = None if positive_count <= 0 else true_positive / float(positive_count)
    negative_recall = None if negative_count <= 0 else true_negative / float(negative_count)
    if positive_recall is None and negative_recall is None:
        balanced_accuracy = None
    elif positive_recall is None:
        balanced_accuracy = negative_recall
    elif negative_recall is None:
        balanced_accuracy = positive_recall
    else:
        balanced_accuracy = 0.5 * (positive_recall + negative_recall)
    return {
        "valid_count": int(len(labels)),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_rate": positive_count / float(len(labels)),
        "accuracy": correct_count / float(len(labels)),
        "balanced_accuracy": balanced_accuracy,
        "roc_auc": _roc_auc(scores, labels),
        "average_precision": _average_precision(scores, labels),
        "best_f1": best_f1,
        "best_threshold": best_threshold,
    }


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(item) for item in values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, float(q))) * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _safe_div(numerator: Any, denominator: Any) -> float | None:
    try:
        num = float(numerator)
        den = float(denominator)
    except (TypeError, ValueError):
        return None
    if den == 0.0:
        return None
    return num / den


def _gt_boundaries(segments: Sequence[Any]) -> list[float]:
    boundaries: list[float] = []
    for segment in segments:
        if len(segment) != 2:
            raise ValueError("each gt segment must contain [start, end]")
        boundaries.extend([float(segment[0]), float(segment[1])])
    return boundaries


def _boundary_hit_count(selected: Sequence[int], boundaries: Sequence[float], radius: int | float) -> int:
    selected_positions = [float(idx) for idx in selected]
    radius = float(radius)
    return sum(
        1
        for boundary in boundaries
        if any(abs(position - float(boundary)) <= radius for position in selected_positions)
    )


def _boundary_support(hit_count: int, boundary_count: int) -> float | None:
    if int(boundary_count) <= 0:
        return None
    support = float(hit_count) / float(boundary_count)
    if not 0.0 <= support <= 1.0:
        raise ValueError(f"boundary support must lie in [0, 1], got {support}")
    return support


def _resolve_budget(valid_count: int, *, budget: int | None, budget_fraction: float | None) -> int:
    if valid_count <= 0:
        return 0
    if budget is not None:
        resolved = int(budget)
    else:
        fraction = 0.5 if budget_fraction is None else float(budget_fraction)
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("budget_fraction must lie inside [0, 1]")
        resolved = int(math.ceil(valid_count * fraction))
    return max(0, min(valid_count, resolved))


def _selected_run_lengths(selected: Sequence[int]) -> list[int]:
    if not selected:
        return []
    lengths: list[int] = []
    current_length = 1
    previous = int(selected[0])
    for index in selected[1:]:
        current = int(index)
        if current == previous + 1:
            current_length += 1
        else:
            lengths.append(current_length)
            current_length = 1
        previous = current
    lengths.append(current_length)
    return lengths


def _sample_id_from_batch_row(batch: Mapping[str, Any], batch_idx: int, fallback: str) -> str:
    for key in ("sample_id", "video_name", "video_id", "sample_name"):
        if key in batch:
            value = batch[key]
            values = _as_nested_list(value)
            if isinstance(values, list):
                if batch_idx < len(values):
                    item = values[batch_idx]
                    if item is not None:
                        return str(item)
            elif values is not None:
                return str(values)
    metas = batch.get("metas")
    if isinstance(metas, list) and batch_idx < len(metas):
        meta = metas[batch_idx]
        if isinstance(meta, Mapping):
            for key in ("sample_id", "video_name", "video_id"):
                if key in meta and meta[key] is not None:
                    return str(meta[key])
            if "window_start_frame" in meta and "video_name" in meta:
                return f"{meta['video_name']}|window_start_frame={meta['window_start_frame']}"
    if isinstance(metas, Mapping):
        for key in ("sample_id", "video_name", "video_id"):
            if key in metas and metas[key] is not None:
                return str(metas[key])
    return fallback


def _frame_roles_from_scores(
    *,
    p_action: Sequence[float],
    entropy: Sequence[float],
    p_change: Sequence[float],
    margin: Sequence[float],
    valid: Sequence[bool],
    boundary_radius: int,
) -> list[dict[str, Any]]:
    action_enter = 0.60
    action_exit = 0.40
    transition_entropy = 0.62
    transition_change = 0.20
    roles: list[dict[str, Any]] = []
    for idx, (prob, ent, change, marg, is_valid) in enumerate(zip(p_action, entropy, p_change, margin, valid)):
        if not is_valid:
            roles.append(
                {
                    "frame_idx": idx,
                    "role": "invalid",
                    "candidate_roles": [],
                    "overlap_roles": [],
                    "mixed_fill": False,
                    "state_id": -1,
                }
            )
            continue
        candidate_roles = ["background", "action"]
        if ent >= transition_entropy or change >= transition_change or (action_exit <= prob <= action_enter):
            candidate_roles.append("transition")
        overlap_roles = list(candidate_roles)
        if marg <= 0.35:
            overlap_roles.append("uncertain")
        if prob >= action_enter:
            role = "action"
            state_id = 2
        elif prob <= action_exit:
            role = "background"
            state_id = 0
        elif ent >= transition_entropy or change >= transition_change:
            role = "transition"
            state_id = 1
        else:
            role = "transition"
            state_id = 1
        mixed_fill = bool(
            role == "transition"
            or (action_exit <= prob <= action_enter)
            or (ent >= transition_entropy and change >= transition_change)
            or (marg <= 0.35)
        )
        roles.append(
            {
                "frame_idx": idx,
                "role": role,
                "candidate_roles": candidate_roles,
                "overlap_roles": overlap_roles,
                "mixed_fill": mixed_fill,
                "state_id": state_id,
                "boundary_radius": int(boundary_radius),
            }
        )
    return roles


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _sigmoid(value: float) -> float:
    value = float(value)
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _binary_entropy(prob: float) -> float:
    prob = min(max(float(prob), 0.0), 1.0)
    other = 1.0 - prob
    if prob <= 0.0 or other <= 0.0:
        return 0.0
    return float(-(prob * math.log(prob) + other * math.log(other)) / math.log(2.0))


def _frame_signal_bundle(logit_row: Sequence[float], valid_row: Sequence[bool]) -> dict[str, list[float] | list[int] | list[bool]]:
    logits = [float(value) for value in logit_row]
    valid = [bool(item) for item in valid_row]
    if len(logits) != len(valid):
        raise ValueError("logits and valid must share batch/time shape")

    p_action = [_sigmoid(value) if is_valid else 0.0 for value, is_valid in zip(logits, valid)]
    entropy = [_binary_entropy(prob) if is_valid else 0.0 for prob, is_valid in zip(p_action, valid)]
    margin = [abs(2.0 * prob - 1.0) if is_valid else 0.0 for prob, is_valid in zip(p_action, valid)]
    p_change = [0.0]
    for previous, current, is_valid in zip(p_action, p_action[1:], valid[1:]):
        p_change.append(abs(float(current) - float(previous)) if is_valid else 0.0)

    state_ids: list[int] = []
    state_change: list[bool] = []
    state = 0
    action_enter = 0.60
    action_exit = 0.40
    transition_entropy = 0.62
    transition_change = 0.20
    for idx, (prob, ent, change, is_valid) in enumerate(zip(p_action, entropy, p_change, valid)):
        if not is_valid:
            state_ids.append(-1)
            state_change.append(False)
            continue
        if idx == 0:
            if prob >= action_enter:
                state = 2
            elif prob <= action_exit:
                state = 0
            else:
                state = 1
            state_ids.append(state)
            state_change.append(True)
            continue
        previous_state = state
        if prob >= action_enter:
            state = 2
        elif prob <= action_exit:
            state = 0
        elif ent >= transition_entropy or change >= transition_change:
            state = 1
        state_ids.append(state)
        state_change.append(state != previous_state)

    boundary_score = [
        (float(ent) + float(change) + (1.0 - float(marg)) + (1.0 if bool(changed) else 0.0))
        if is_valid
        else 0.0
        for ent, change, marg, changed, is_valid in zip(entropy, p_change, margin, state_change, valid)
    ]
    action_score = [float(prob) if is_valid else 0.0 for prob, is_valid in zip(p_action, valid)]
    background_score = [1.0 - float(prob) if is_valid else 0.0 for prob, is_valid in zip(p_action, valid)]
    uncertainty = [1.0 - float(marg) if is_valid else 0.0 for marg, is_valid in zip(margin, valid)]
    overlap_eligible = [
        bool(is_valid and (state_id == 1 or (ent >= transition_entropy and change >= transition_change)))
        for state_id, ent, change, is_valid in zip(state_ids, entropy, p_change, valid)
    ]
    mixed_fill = [bool(is_valid and (state_id == 1 or (prob >= action_exit and prob <= action_enter) or eligible))
                   for state_id, prob, eligible, is_valid in zip(state_ids, p_action, overlap_eligible, valid)]
    role_overlap = [1.0 if bool(mixed) else 0.0 for mixed in mixed_fill]

    return {
        "p_action": p_action,
        "entropy": entropy,
        "p_change": p_change,
        "margin": margin,
        "uncertainty": uncertainty,
        "state_id": state_ids,
        "state_change": state_change,
        "boundary_score": boundary_score,
        "action_score": action_score,
        "background_score": background_score,
        "overlap_eligible": overlap_eligible,
        "mixed_fill": mixed_fill,
        "role_overlap": role_overlap,
    }


def _select_top_indices(scores: Sequence[float], valid: Sequence[bool], budget: int) -> list[int]:
    valid_indices = [idx for idx, is_valid in enumerate(valid) if bool(is_valid)]
    if budget <= 0 or not valid_indices:
        return []
    ranked = sorted(valid_indices, key=lambda idx: (float(scores[idx]), -idx), reverse=True)
    return sorted(ranked[: min(int(budget), len(valid_indices))])


def _selection_quality_from_indices(
    *,
    selected_indices: Sequence[Sequence[int]],
    target_rows: Sequence[Sequence[Any]],
    valid_rows: Sequence[Sequence[Any]],
    gt_rows: Sequence[Sequence[Any]],
    budget: int | None,
    requested_budget_fraction: float | None,
    boundary_radius: int,
    strategy_name: str | None = None,
) -> dict[str, Any]:
    selected_counts: list[int] = []
    budget_fractions: list[float] = []
    gaps: list[float] = []
    run_counts: list[int] = []
    longest_runs: list[int] = []
    run_lengths_by_window: list[list[int]] = []
    all_run_lengths: list[int] = []
    all_selected_distances: list[float] = []
    boundary_hits = 0
    boundary_total = 0
    selected_positive = 0
    total_selected = 0
    selected_unique_positive = 0
    total_positive = 0

    for selected_raw, target_row, valid_row, segments in zip(selected_indices, target_rows, valid_rows, gt_rows):
        valid_indices = [idx for idx, is_valid in enumerate(valid_row) if bool(is_valid)]
        valid_set = set(valid_indices)
        selected = sorted(dict.fromkeys(int(idx) for idx in selected_raw if int(idx) in valid_set))
        selected_set = set(selected)
        selected_counts.append(len(selected))
        budget_fractions.append(0.0 if not valid_indices else len(selected) / float(len(valid_indices)))
        total_selected += len(selected)
        run_lengths = _selected_run_lengths(selected)
        run_lengths_by_window.append(run_lengths)
        run_counts.append(len(run_lengths))
        longest_runs.append(0 if not run_lengths else max(run_lengths))
        all_run_lengths.extend(run_lengths)

        for left, right in zip(selected, selected[1:]):
            gaps.append(float(right - left))
        for idx in selected:
            if float(target_row[idx]) >= 0.5:
                selected_positive += 1
        positive_indices = [idx for idx in valid_indices if float(target_row[idx]) >= 0.5]
        total_positive += len(positive_indices)
        selected_unique_positive += sum(1 for idx in positive_indices if idx in selected_set)

        boundaries = _gt_boundaries(segments)
        boundary_total += len(boundaries)
        boundary_hits += _boundary_hit_count(selected, boundaries, boundary_radius)
        for idx in selected:
            distance = min((abs(float(idx) - boundary) for boundary in boundaries), default=None)
            if distance is not None:
                all_selected_distances.append(float(distance))

    support = _boundary_support(boundary_hits, boundary_total)
    metrics = {
        "budget": None if budget is None else int(budget),
        "budget_fraction": sum(budget_fractions) / float(max(len(budget_fractions), 1)),
        "requested_budget_fraction": requested_budget_fraction,
        "selected_count": sum(selected_counts) / float(max(len(selected_counts), 1)),
        "sample_count": int(sum(selected_counts)),
        "selected_indices": [list(map(int, row)) for row in selected_indices],
        "selected_run_count_mean": sum(run_counts) / float(max(len(run_counts), 1)),
        "selected_run_count_p95": _percentile(run_counts, 0.95),
        "longest_selected_run_mean": sum(longest_runs) / float(max(len(longest_runs), 1)),
        "longest_selected_run_p95": _percentile(longest_runs, 0.95),
        "mean_selected_run_length": None if not all_run_lengths else sum(all_run_lengths) / float(len(all_run_lengths)),
        "selected_run_count_by_window": run_counts,
        "longest_selected_run_by_window": longest_runs,
        "selected_run_lengths_by_window": run_lengths_by_window,
        f"boundary_support_r{int(boundary_radius)}": support,
        f"boundary_support@{int(boundary_radius)}": support,
        "zero_support_rate": None if support is None else 1.0 - support,
        "mean_gap": None if not gaps else sum(gaps) / float(len(gaps)),
        "max_gap": None if not gaps else max(gaps),
        "p95_gap": _percentile(gaps, 0.95),
        "mean_selected_distance_to_boundary": None if not all_selected_distances else sum(all_selected_distances) / float(len(all_selected_distances)),
        "action_selected_fraction": None if total_selected <= 0 else selected_positive / float(total_selected),
        "action_positive_recall": None if total_positive <= 0 else selected_unique_positive / float(total_positive),
        "action_positive_coverage": None if total_positive <= 0 else selected_unique_positive / float(total_positive),
    }
    if strategy_name is not None:
        metrics["strategy"] = str(strategy_name)
    return metrics


def _allocate_role_budgets(budget: int, fractions: Sequence[float]) -> list[int]:
    if budget <= 0:
        return [0 for _ in fractions]
    total = sum(float(value) for value in fractions)
    if total <= 0.0:
        fractions = [1.0 for _ in fractions]
        total = float(len(fractions))
    raw = [float(budget) * float(value) / total for value in fractions]
    assigned = [int(math.floor(value)) for value in raw]
    remainder = budget - sum(assigned)
    order = sorted(range(len(raw)), key=lambda idx: (raw[idx] - assigned[idx], -idx), reverse=True)
    for idx in order[: max(0, remainder)]:
        assigned[idx] += 1
    return assigned


def compute_sampling_quality_from_logits(
    *,
    logits: Any,
    target: Any,
    valid: Any,
    gt_segments: Sequence[Any],
    budget: int | None = None,
    budget_fraction: float | None = None,
    boundary_radius: int = 1,
) -> dict[str, Any]:
    """Evaluate top-k logit frame sampling without invoking a detector."""

    logits_rows = _as_nested_list(logits)
    target_rows = _as_nested_list(target)
    valid_rows = _as_nested_list(valid)
    gt_rows = [_as_nested_list(row) for row in gt_segments]
    if not (len(logits_rows) == len(target_rows) == len(valid_rows) == len(gt_rows)):
        raise ValueError("logits, target, valid, and gt_segments batch sizes must match")
    if int(boundary_radius) < 0:
        raise ValueError("boundary_radius must be non-negative")

    selected_indices: list[list[int]] = []
    selected_counts: list[int] = []
    budget_fractions: list[float] = []
    gaps: list[float] = []
    run_counts: list[int] = []
    longest_runs: list[int] = []
    run_lengths_by_window: list[list[int]] = []
    all_run_lengths: list[int] = []
    boundary_hits = 0
    boundary_total = 0
    selected_positive = 0
    total_selected = 0
    selected_unique_positive = 0
    total_positive = 0

    for logit_row, target_row, valid_row, segments in zip(logits_rows, target_rows, valid_rows, gt_rows):
        if not (len(logit_row) == len(target_row) == len(valid_row)):
            raise ValueError("logits, target, and valid must share batch/time shape")
        valid_indices = [idx for idx, is_valid in enumerate(valid_row) if bool(is_valid)]
        resolved_budget = _resolve_budget(len(valid_indices), budget=budget, budget_fraction=budget_fraction)
        ranked = sorted(valid_indices, key=lambda idx: (float(logit_row[idx]), -idx), reverse=True)
        selected = sorted(ranked[:resolved_budget])
        selected_set = set(selected)
        selected_indices.append(selected)
        selected_counts.append(len(selected))
        budget_fractions.append(0.0 if not valid_indices else len(selected) / float(len(valid_indices)))
        total_selected += len(selected)
        run_lengths = _selected_run_lengths(selected)
        run_lengths_by_window.append(run_lengths)
        run_counts.append(len(run_lengths))
        longest_runs.append(0 if not run_lengths else max(run_lengths))
        all_run_lengths.extend(run_lengths)

        for left, right in zip(selected, selected[1:]):
            gaps.append(float(right - left))
        for idx in selected:
            if float(target_row[idx]) >= 0.5:
                selected_positive += 1
        positive_indices = [idx for idx in valid_indices if float(target_row[idx]) >= 0.5]
        total_positive += len(positive_indices)
        selected_unique_positive += sum(1 for idx in positive_indices if idx in selected_set)

        boundaries = _gt_boundaries(segments)
        boundary_total += len(boundaries)
        boundary_hits += _boundary_hit_count(selected, boundaries, boundary_radius)

    support = _boundary_support(boundary_hits, boundary_total)
    mean_selected_count = sum(selected_counts) / float(max(len(selected_counts), 1))
    mean_budget_fraction = sum(budget_fractions) / float(max(len(budget_fractions), 1))
    mean_run_count = sum(run_counts) / float(max(len(run_counts), 1))
    mean_longest_run = sum(longest_runs) / float(max(len(longest_runs), 1))
    action_selected_fraction = None if total_selected <= 0 else selected_positive / float(total_selected)
    action_positive_recall = None if total_positive <= 0 else selected_unique_positive / float(total_positive)
    max_gap = None if not gaps else max(gaps)
    metrics = {
        "budget": None if budget is None else int(budget),
        "budget_fraction": mean_budget_fraction,
        "requested_budget_fraction": budget_fraction,
        "selected_count": mean_selected_count,
        "sample_count": int(sum(selected_counts)),
        "selected_indices": selected_indices,
        "selected_run_count_mean": mean_run_count,
        "selected_run_count_p95": _percentile(run_counts, 0.95),
        "longest_selected_run_mean": mean_longest_run,
        "longest_selected_run_p95": _percentile(longest_runs, 0.95),
        "mean_selected_run_length": None
        if not all_run_lengths
        else sum(all_run_lengths) / float(len(all_run_lengths)),
        "selected_run_count_by_window": run_counts,
        "longest_selected_run_by_window": longest_runs,
        "selected_run_lengths_by_window": run_lengths_by_window,
        f"boundary_support_r{int(boundary_radius)}": support,
        f"boundary_support@{int(boundary_radius)}": support,
        "zero_support_rate": None if support is None else 1.0 - support,
        "mean_gap": None if not gaps else sum(gaps) / float(len(gaps)),
        "max_gap": max_gap,
        "p95_gap": _percentile(gaps, 0.95),
        "action_selected_fraction": action_selected_fraction,
        "action_positive_recall": action_positive_recall,
        "action_positive_coverage": action_positive_recall,
    }
    return metrics


def compute_indirect_selection_quality_from_logits(
    *,
    logits: Any,
    target: Any,
    valid: Any,
    gt_segments: Sequence[Any],
    sample_ids: Sequence[Any] | None = None,
    snapshot_id: str | None = None,
    budget: int | None = None,
    budget_fraction: float | None = None,
    boundary_radius: int = 1,
) -> dict[str, Any]:
    logits_rows = _as_nested_list(logits)
    target_rows = _as_nested_list(target)
    valid_rows = _as_nested_list(valid)
    gt_rows = [_as_nested_list(row) for row in gt_segments]
    if not (len(logits_rows) == len(target_rows) == len(valid_rows) == len(gt_rows)):
        raise ValueError("logits, target, valid, and gt_segments batch sizes must match")
    if sample_ids is None:
        resolved_sample_ids = [f"sample_{sample_idx}" for sample_idx in range(len(logits_rows))]
    else:
        resolved_sample_ids = [str(item) for item in _as_nested_list(sample_ids)]
        if len(resolved_sample_ids) != len(logits_rows):
            raise ValueError("sample_ids batch size must match logits batch size")

    baseline_metrics = compute_sampling_quality_from_logits(
        logits=logits,
        target=target,
        valid=valid,
        gt_segments=gt_segments,
        budget=budget,
        budget_fraction=budget_fraction,
        boundary_radius=boundary_radius,
    )

    selected_indices: list[list[int]] = []
    per_sample_rows: list[dict[str, Any]] = []
    per_sample_baseline: list[list[int]] = baseline_metrics["selected_indices"]
    boundary_total = 0
    boundary_hits = 0
    boundary_selected_distances: list[float] = []
    strategy_selected_indices: dict[str, list[list[int]]] = {
        "topk_action_logit": [],
        "delta_p_action": [],
        "entropy_uncertainty": [],
        "boundary_score": [],
        "weighted_transition_mix": [],
        "state_machine_mix": [],
    }
    selected_role_counts = {"background": 0, "action": 0, "transition": 0, "mixed_fill": 0}
    action_selected_count = 0
    background_selected_count = 0
    overlap_selected_count = 0
    action_total = 0
    background_total = 0
    selected_count_total = 0
    zero_boundary_support_count = 0
    selected_run_lengths_all: list[int] = []
    all_selected_distances: list[float] = []

    for sample_idx, (logit_row, target_row, valid_row, segments) in enumerate(zip(logits_rows, target_rows, valid_rows, gt_rows)):
        if not (len(logit_row) == len(target_row) == len(valid_row)):
            raise ValueError("logits, target, and valid must share batch/time shape")
        sample_id = resolved_sample_ids[sample_idx]
        bundle = _frame_signal_bundle(logit_row, valid_row)
        p_action = bundle["p_action"]
        entropy = bundle["entropy"]
        p_change = bundle["p_change"]
        margin = bundle["margin"]
        valid_mask = [bool(item) for item in valid_row]
        valid_indices = [idx for idx, is_valid in enumerate(valid_mask) if is_valid]
        resolved_budget = _resolve_budget(len(valid_indices), budget=budget, budget_fraction=budget_fraction)

        boundaries = _gt_boundaries(segments)
        boundary_total += len(boundaries)

        boundary_candidates = _select_top_indices(bundle["boundary_score"], valid_mask, resolved_budget)
        action_candidates = _select_top_indices(bundle["action_score"], valid_mask, resolved_budget)
        background_candidates = _select_top_indices(bundle["background_score"], valid_mask, resolved_budget)
        delta_candidates = _select_top_indices(bundle["p_change"], valid_mask, resolved_budget)
        entropy_candidates = _select_top_indices(bundle["entropy"], valid_mask, resolved_budget)
        weighted_transition_score = [
            (
                1.50 * float(bundle["boundary_score"][idx])
                + 1.00 * float(bundle["p_change"][idx])
                + 0.75 * float(bundle["entropy"][idx])
                + 0.50 * float(bundle["uncertainty"][idx])
                + 0.25 * float(bundle["role_overlap"][idx])
            )
            if bool(valid_mask[idx])
            else 0.0
            for idx in range(len(valid_mask))
        ]
        weighted_transition_candidates = _select_top_indices(weighted_transition_score, valid_mask, resolved_budget)
        selected_roles = _frame_roles_from_scores(
            p_action=p_action,
            entropy=entropy,
            p_change=p_change,
            margin=margin,
            valid=valid_mask,
            boundary_radius=boundary_radius,
        )

        role_candidates = {
            "boundary": boundary_candidates,
            "action": action_candidates,
            "background": background_candidates,
        }
        role_budgets = _allocate_role_budgets(
            resolved_budget,
            [
                sum(1.0 for idx in valid_indices if selected_roles[idx]["role"] == "transition"),
                sum(1.0 for idx in valid_indices if selected_roles[idx]["role"] == "action"),
                sum(1.0 for idx in valid_indices if selected_roles[idx]["role"] == "background"),
            ],
        )
        provisional: list[int] = []
        for role_name, role_budget in zip(("boundary", "action", "background"), role_budgets):
            for idx in role_candidates[role_name][: max(0, int(role_budget))]:
                provisional.append(int(idx))

        if len(provisional) < resolved_budget:
            fallback_scores = {idx: float(bundle["boundary_score"][idx]) + float(bundle["action_score"][idx]) + float(bundle["background_score"][idx]) for idx in valid_indices}
            fallback_order = sorted(valid_indices, key=lambda idx: (fallback_scores[idx], -idx), reverse=True)
            for idx in fallback_order:
                if idx not in provisional:
                    provisional.append(int(idx))
                if len(provisional) >= resolved_budget:
                    break

        selected = sorted(dict.fromkeys(provisional) if provisional else [])
        if len(selected) > resolved_budget:
            selected = sorted(selected[:resolved_budget])
        selected_indices.append(selected)
        strategy_selected_indices["topk_action_logit"].append(list(per_sample_baseline[sample_idx]))
        strategy_selected_indices["delta_p_action"].append(delta_candidates)
        strategy_selected_indices["entropy_uncertainty"].append(entropy_candidates)
        strategy_selected_indices["boundary_score"].append(boundary_candidates)
        strategy_selected_indices["weighted_transition_mix"].append(weighted_transition_candidates)
        strategy_selected_indices["state_machine_mix"].append(selected)
        selected_count_total += len(selected)
        selected_run_lengths_all.extend(_selected_run_lengths(selected))

        sample_selected_scores: list[dict[str, Any]] = []
        for idx in selected:
            sample_selected_scores.append(
                {
                    "frame_idx": int(idx),
                    "p_action": float(p_action[idx]),
                    "entropy": float(entropy[idx]),
                    "p_change": float(p_change[idx]),
                    "margin": float(margin[idx]),
                    "role": selected_roles[idx]["role"],
                    "candidate_roles": list(selected_roles[idx]["candidate_roles"]),
                    "overlap_roles": list(selected_roles[idx]["overlap_roles"]),
                    "mixed_fill": bool(selected_roles[idx]["mixed_fill"]),
                    "boundary_score": float(bundle["boundary_score"][idx]),
                    "state_id": int(selected_roles[idx]["role"] == "transition" and 1 or (2 if selected_roles[idx]["role"] == "action" else 0)),
                }
            )
            distance = min((abs(float(idx) - boundary) for boundary in boundaries), default=None)
            if distance is not None:
                all_selected_distances.append(float(distance))
                boundary_selected_distances.append(float(distance))
            if float(target_row[idx]) >= 0.5:
                action_selected_count += 1
            else:
                background_selected_count += 1
            if selected_roles[idx]["mixed_fill"]:
                overlap_selected_count += 1
            role_name = str(selected_roles[idx]["role"])
            selected_role_counts[role_name if role_name in selected_role_counts else "transition"] += 1

        action_total += sum(1 for idx in valid_indices if float(target_row[idx]) >= 0.5)
        background_total += sum(1 for idx in valid_indices if float(target_row[idx]) < 0.5)
        sample_boundary_hits = _boundary_hit_count(selected, boundaries, boundary_radius)
        boundary_hits += sample_boundary_hits
        if boundaries and sample_boundary_hits == 0:
            zero_boundary_support_count += 1

        per_sample_rows.append(
            {
                "sample_id": sample_id,
                "snapshot_id": snapshot_id,
                "dense_len": len(logit_row),
                "valid_len": len(valid_indices),
                "budget": int(resolved_budget),
                "baseline_selected_positions": per_sample_baseline[sample_idx],
                "selected_positions": selected,
                "selected_positions_baseline": per_sample_baseline[sample_idx],
                "selected_sources": {
                    "baseline": "topk_action_logit",
                    "indirect": "boundary_action_background_state_machine",
                    "strategies": list(strategy_selected_indices),
                },
                "strategy_selected_positions": {
                    key: list(value[sample_idx]) for key, value in strategy_selected_indices.items()
                },
                "frame_signals": {
                    "p_action": [float(item) for item in p_action],
                    "entropy": [float(item) for item in entropy],
                    "p_change": [float(item) for item in p_change],
                    "margin": [float(item) for item in margin],
                    "boundary_score": [float(item) for item in bundle["boundary_score"]],
                    "action_score": [float(item) for item in bundle["action_score"]],
                    "background_score": [float(item) for item in bundle["background_score"]],
                    "mixed_fill": [bool(item) for item in bundle["mixed_fill"]],
                    "role_overlap": [float(item) for item in bundle["role_overlap"]],
                },
                "p_action": [float(item) for item in p_action],
                "entropy": [float(item) for item in entropy],
                "p_change": [float(item) for item in p_change],
                "margin": [float(item) for item in margin],
                "mixed_fill": [bool(item) for item in bundle["mixed_fill"]],
                "role_overlap": [float(item) for item in bundle["role_overlap"]],
                "candidate_roles": [list(item["candidate_roles"]) for item in selected_roles],
                "overlap_roles": [list(item["overlap_roles"]) for item in selected_roles],
                "frame_signals": {
                    "p_action": [float(item) for item in p_action],
                    "entropy": [float(item) for item in entropy],
                    "p_change": [float(item) for item in p_change],
                    "margin": [float(item) for item in margin],
                    "boundary_score": [float(item) for item in bundle["boundary_score"]],
                    "action_score": [float(item) for item in bundle["action_score"]],
                    "background_score": [float(item) for item in bundle["background_score"]],
                    "mixed_fill": [bool(item) for item in bundle["mixed_fill"]],
                    "role_overlap": [float(item) for item in bundle["role_overlap"]],
                },
                "roles": selected_roles,
                "selected_role_details": sample_selected_scores,
                "boundary_radius": int(boundary_radius),
                f"boundary_support_r{int(boundary_radius)}": _boundary_support(sample_boundary_hits, len(boundaries)),
                f"baseline_boundary_support_r{int(boundary_radius)}": None
                if not boundaries
                else sum(
                    1
                    for boundary in boundaries
                    if any(abs(float(idx) - boundary) <= float(boundary_radius) for idx in per_sample_baseline[sample_idx])
                )
                / float(len(boundaries)),
                "action_coverage": _safe_div(
                    sum(1 for idx in selected if float(target_row[idx]) >= 0.5),
                    sum(1 for idx in valid_indices if float(target_row[idx]) >= 0.5),
                ),
                "background_coverage": _safe_div(
                    sum(1 for idx in selected if float(target_row[idx]) < 0.5),
                    sum(1 for idx in valid_indices if float(target_row[idx]) < 0.5),
                ),
                "selected_run_length_mean": None
                if not _selected_run_lengths(selected)
                else sum(_selected_run_lengths(selected)) / float(len(_selected_run_lengths(selected))),
            }
        )

    boundary_key = f"boundary_support_r{int(boundary_radius)}"
    boundary_at_key = f"boundary_support@{int(boundary_radius)}"
    indirect_support = _boundary_support(boundary_hits, boundary_total)
    indirect_metrics = {
        "budget": None if budget is None else int(budget),
        "budget_fraction": _safe_div(selected_count_total, sum(sum(1 for is_valid in row if bool(is_valid)) for row in valid_rows)),
        "selected_count": selected_count_total / float(max(len(selected_indices), 1)),
        "selected_indices": selected_indices,
        boundary_key: indirect_support,
        boundary_at_key: indirect_support,
        "zero_support_rate": None if indirect_support is None else 1.0 - indirect_support,
        "mean_selected_run_length": None if not selected_run_lengths_all else sum(selected_run_lengths_all) / float(len(selected_run_lengths_all)),
        "mean_selected_distance_to_boundary": None if not all_selected_distances else sum(all_selected_distances) / float(len(all_selected_distances)),
        "action_selected_fraction": _safe_div(action_selected_count, selected_count_total),
        "action_positive_coverage": _safe_div(action_selected_count, action_total),
        "background_selected_fraction": _safe_div(background_selected_count, selected_count_total),
        "background_positive_coverage": _safe_div(background_selected_count, background_total),
        "selected_role_counts": selected_role_counts,
        "mixed_fill_selected_count": overlap_selected_count,
        "mixed_fill_rate": _safe_div(overlap_selected_count, selected_count_total),
        "sample_count": len(selected_indices),
    }
    strategy_metrics = {
        strategy_name: _selection_quality_from_indices(
            selected_indices=strategy_indices,
            target_rows=target_rows,
            valid_rows=valid_rows,
            gt_rows=gt_rows,
            budget=budget,
            requested_budget_fraction=budget_fraction,
            boundary_radius=boundary_radius,
            strategy_name=strategy_name,
        )
        for strategy_name, strategy_indices in strategy_selected_indices.items()
    }
    boundary_values = {
        strategy_name: metrics.get(boundary_key)
        for strategy_name, metrics in strategy_metrics.items()
    }
    valid_boundary_values = {
        strategy_name: float(value)
        for strategy_name, value in boundary_values.items()
        if value is not None
    }
    best_boundary_strategy = None if not valid_boundary_values else max(valid_boundary_values, key=lambda item: valid_boundary_values[item])

    return {
        "baseline": baseline_metrics,
        "indirect": indirect_metrics,
        "strategy_metrics": strategy_metrics,
        "strategy_comparison": {
            "boundary_support_key": boundary_key,
            "boundary_support_by_strategy": boundary_values,
            f"{boundary_key}_by_strategy": boundary_values,
            "best_boundary_support_strategy": best_boundary_strategy,
        },
        "per_sample_rows": per_sample_rows,
        "delta": {
            boundary_key: None
            if baseline_metrics[boundary_key] is None or indirect_metrics[boundary_key] is None
            else indirect_metrics[boundary_key] - baseline_metrics[boundary_key],
            "action_positive_coverage": None
            if baseline_metrics["action_positive_coverage"] is None or indirect_metrics["action_positive_coverage"] is None
            else indirect_metrics["action_positive_coverage"] - baseline_metrics["action_positive_coverage"],
            "mean_selected_run_length": None
            if baseline_metrics["mean_selected_run_length"] is None or indirect_metrics["mean_selected_run_length"] is None
            else indirect_metrics["mean_selected_run_length"] - baseline_metrics["mean_selected_run_length"],
        },
        "per_sample": per_sample_rows,
    }


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_project_importable() -> None:
    root = str(_project_root())
    if root not in sys.path:
        sys.path.insert(0, root)


def _import_torch():
    import torch
    import torch.nn.functional as F

    return torch, F


def _import_torchvision_mobilenet():
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    return mobilenet_v3_small, MobileNet_V3_Small_Weights


def _load_torch_state_dict(path: str, *, map_location: str = "cpu") -> Mapping[str, Any]:
    torch, _F = _import_torch()
    payload = torch.load(path, map_location=map_location)
    if isinstance(payload, Mapping):
        for key in ("state_dict", "probe_state_dict"):
            if key in payload and isinstance(payload[key], Mapping):
                return payload[key]
        return payload
    raise ValueError(f"expected a mapping state_dict at {path}, got {type(payload).__name__}")


def _load_probe_checkpoint(model: Any, checkpoint_path: str):
    state_dict = _load_torch_state_dict(str(checkpoint_path))
    return model.load_state_dict(state_dict)


def _seed_everything(seed: int) -> None:
    torch, _F = _import_torch()
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed % (2**32))
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker_from_torch(worker_id: int) -> None:
    torch, _F = _import_torch()
    worker_seed = int(torch.initial_seed() % (2**32))
    random.seed(worker_seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(worker_seed)
    except Exception:
        pass
    torch.manual_seed(worker_seed)


def default_reader_cfg(in_dim: int = 3 * 32 * 32, num_slots: int = 384) -> dict[str, Any]:
    return {
        "type": DEFAULT_READER_TYPE,
        "in_dim": int(in_dim),
        "hidden_dim": 96,
        "num_slots": int(num_slots),
        "temporal_layers": 4,
        "temporal_kernel_size": 5,
        "dilations": (1, 2, 4, 8),
        "dropout": 0.10,
    }


class C3LowResActionProbe:
    """Thin diagnostic wrapper around a C3 low-resolution actionness reader."""

    def __init__(self, reader_cfg: Mapping[str, Any] | None = None) -> None:
        _ensure_project_importable()
        from opentad.models.builder import build_selector
        import opentad.models.selectors  # noqa: F401

        if reader_cfg is None:
            reader_cfg = default_reader_cfg()
        reader_cfg = dict(reader_cfg)
        reader_cfg.setdefault("type", DEFAULT_READER_TYPE)
        if reader_cfg["type"] not in SUPPORTED_C3_READER_TYPES:
            raise ValueError(f"action probe expects one of {sorted(SUPPORTED_C3_READER_TYPES)}, got {reader_cfg['type']}")
        self.reader = build_selector(reader_cfg)

    def __call__(self, features: Any, valid: Any, time_coords: Any | None = None):
        reader_outputs = self.reader(features, valid, time_coords)
        return reader_outputs["action_logits"]

    def train(self):
        self.reader.train()
        return self

    def eval(self):
        self.reader.eval()
        return self

    def to(self, *args, **kwargs):
        self.reader.to(*args, **kwargs)
        return self

    def parameters(self):
        return self.reader.parameters()

    def state_dict(self):
        return self.reader.state_dict()

    def load_state_dict(self, state_dict):
        return self.reader.load_state_dict(state_dict)


class C3MobileNetV3ActionProbe:
    """Frame-wise MobileNetV3 action/background probe for diagnostic experiments."""

    def __init__(
        self,
        *,
        pretrained: bool = True,
        variant: str = "small",
        freeze_backbone: bool = False,
        weights_path: str | None = None,
    ) -> None:
        if variant != "small":
            raise ValueError("only MobileNetV3-small is supported for this probe")
        torch, _F = _import_torch()
        nn = getattr(sys.modules.get(__name__), "nn", None)
        if nn is None:
            import torch.nn as nn  # type: ignore

        mobilenet_v3_small, weights_cls = _import_torchvision_mobilenet()
        self.module = nn.Module()
        if weights_path:
            self.backbone = mobilenet_v3_small(weights=None)
            self.backbone.load_state_dict(dict(_load_torch_state_dict(weights_path)))
        else:
            weights = weights_cls.DEFAULT if pretrained else None
            self.backbone = mobilenet_v3_small(weights=weights)
        self.module.backbone = self.backbone if isinstance(self.backbone, nn.Module) else nn.Identity()
        self._external_backbone = None if isinstance(self.backbone, nn.Module) else self.backbone
        self.output_head = None
        if hasattr(self.backbone, "classifier"):
            classifier = getattr(self.backbone, "classifier")
            try:
                last_layer = classifier[-1]
                in_features = int(last_layer.in_features)
                classifier[-1] = nn.Linear(in_features, 1)
                self.output_head = None
            except Exception:
                self.output_head = nn.LazyLinear(1)
                self.module.output_head = self.output_head
        else:
            self.output_head = nn.LazyLinear(1)
            self.module.output_head = self.output_head
        if freeze_backbone:
            for name, param in self.module.named_parameters():
                if not name.startswith("output_head"):
                    param.requires_grad = False
            if isinstance(self.backbone, nn.Module) and hasattr(self.backbone, "classifier"):
                for param in self.backbone.classifier.parameters():
                    param.requires_grad = True

    def __call__(self, frames: Any, valid: Any, time_coords: Any | None = None):
        torch, _F = _import_torch()
        if frames.ndim != 5:
            raise ValueError(f"MobileNetV3 probe expects [B,T,C,H,W], got {tuple(frames.shape)}")
        batch, dense_len, channels, height, width = frames.shape
        if int(channels) != 3:
            raise ValueError("MobileNetV3 probe expects RGB frame tensors with 3 channels")
        flat = frames.float().reshape(batch * dense_len, channels, height, width)
        if bool((flat.detach().abs().amax() > 2.0).item()):
            flat = flat / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=flat.dtype, device=flat.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=flat.dtype, device=flat.device).view(1, 3, 1, 1)
        flat = (flat - mean) / std
        if self._external_backbone is not None:
            out = self._external_backbone(flat)
        else:
            out = self.backbone(flat)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.ndim > 2:
            out = out.flatten(1)
        if self.output_head is not None:
            out = self.output_head(out)
        logits = out.reshape(batch, dense_len, -1)[..., 0]
        if hasattr(valid, "to"):
            valid = valid.to(device=logits.device).bool()
        return logits.masked_fill(~valid, 0.0)

    def train(self):
        self.module.train()
        if isinstance(self.backbone, type(self.module)):
            self.backbone.train()
        return self

    def eval(self):
        self.module.eval()
        if hasattr(self.backbone, "eval"):
            self.backbone.eval()
        return self

    def to(self, *args, **kwargs):
        self.module.to(*args, **kwargs)
        if hasattr(self.backbone, "to"):
            self.backbone.to(*args, **kwargs)
        return self

    def parameters(self):
        return self.module.parameters()

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict):
        return self.module.load_state_dict(state_dict)


class C3TemporalTCNActionProbe:
    """Low-resolution frame-image TCN probe for action/background diagnostics."""

    def __init__(
        self,
        *,
        variant: str = "lite",
        spatial_size: int = 64,
        hidden_dim: int = 96,
        dropout: float = 0.10,
    ) -> None:
        if variant not in SUPPORTED_TCN_VARIANTS:
            raise ValueError(f"unsupported temporal-tcn variant: {variant}")
        torch, _F = _import_torch()
        import torch.nn as nn  # type: ignore

        class ResidualTCNBlock(nn.Module):
            def __init__(self, channels: int, dilation: int, dropout_rate: float) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size=3,
                        padding=int(dilation),
                        dilation=int(dilation),
                        bias=False,
                    ),
                    nn.BatchNorm1d(channels),
                    nn.SiLU(inplace=True),
                    nn.Dropout(float(dropout_rate)),
                    nn.Conv1d(channels, channels, kernel_size=1, bias=False),
                    nn.BatchNorm1d(channels),
                )
                self.act = nn.SiLU(inplace=True)

            def forward(self, x):
                return self.act(x + self.net(x))

        class GatedTCNBlock(nn.Module):
            def __init__(self, channels: int, dilation: int, dropout_rate: float) -> None:
                super().__init__()
                self.conv = nn.Conv1d(
                    channels,
                    channels * 2,
                    kernel_size=3,
                    padding=int(dilation),
                    dilation=int(dilation),
                    bias=False,
                )
                self.norm = nn.BatchNorm1d(channels * 2)
                self.glu = nn.GLU(dim=1)
                self.dropout = nn.Dropout(float(dropout_rate))
                self.project = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

            def forward(self, x):
                gated = self.glu(self.norm(self.conv(x)))
                return x + self.project(self.dropout(gated))

        self.variant = str(variant)
        self.spatial_size = int(spatial_size)
        self.hidden_dim = int(hidden_dim)
        in_channels = 6 if self.variant == "motion" else 3
        stem_dim = 32 if self.variant == "lite" else 48
        temporal_dim = max(32, int(hidden_dim))

        self.module = nn.Module()
        self.spatial_stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(stem_dim, temporal_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(temporal_dim),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.module.spatial_stem = self.spatial_stem

        if self.variant == "lite":
            self.temporal = nn.Sequential(
                nn.Conv1d(temporal_dim, temporal_dim, kernel_size=3, padding=1, groups=temporal_dim, bias=False),
                nn.BatchNorm1d(temporal_dim),
                nn.SiLU(inplace=True),
                nn.Conv1d(temporal_dim, temporal_dim, kernel_size=1, bias=False),
                nn.BatchNorm1d(temporal_dim),
                nn.SiLU(inplace=True),
                nn.Dropout(float(dropout)),
            )
            self.temporal_branches = None
            classifier_in = temporal_dim
        elif self.variant == "dilated":
            blocks = []
            for dilation in (1, 2, 4, 8):
                blocks.extend(
                    [
                        nn.Conv1d(
                            temporal_dim,
                            temporal_dim,
                            kernel_size=3,
                            padding=int(dilation),
                            dilation=int(dilation),
                            bias=False,
                        ),
                        nn.BatchNorm1d(temporal_dim),
                        nn.SiLU(inplace=True),
                        nn.Dropout(float(dropout)),
                    ]
                )
            self.temporal = nn.Sequential(*blocks)
            self.temporal_branches = None
            classifier_in = temporal_dim
        elif self.variant == "residual":
            self.temporal = nn.Sequential(
                *[ResidualTCNBlock(temporal_dim, dilation, float(dropout)) for dilation in (1, 2, 4, 8)]
            )
            self.temporal_branches = None
            classifier_in = temporal_dim
        elif self.variant == "gated":
            self.temporal = nn.Sequential(
                *[GatedTCNBlock(temporal_dim, dilation, float(dropout)) for dilation in (1, 2, 4, 8)]
            )
            self.temporal_branches = None
            classifier_in = temporal_dim
        elif self.variant == "multiscale":
            self.temporal = None
            self.temporal_branches = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv1d(temporal_dim, temporal_dim, kernel_size=kernel, padding=kernel // 2, bias=False),
                        nn.BatchNorm1d(temporal_dim),
                        nn.SiLU(inplace=True),
                    )
                    for kernel in (3, 5, 9)
                ]
            )
            self.module.temporal_branches = self.temporal_branches
            classifier_in = temporal_dim * 3
        else:
            self.temporal = nn.Sequential(
                nn.Conv1d(temporal_dim, temporal_dim, kernel_size=5, padding=2, bias=False),
                nn.BatchNorm1d(temporal_dim),
                nn.SiLU(inplace=True),
                nn.Conv1d(temporal_dim, temporal_dim, kernel_size=3, padding=2, dilation=2, bias=False),
                nn.BatchNorm1d(temporal_dim),
                nn.SiLU(inplace=True),
                nn.Dropout(float(dropout)),
            )
            self.temporal_branches = None
            classifier_in = temporal_dim
        if self.temporal is not None:
            self.module.temporal = self.temporal
        self.classifier = nn.Conv1d(classifier_in, 1, kernel_size=1)
        self.module.classifier = self.classifier

    def __call__(self, frames: Any, valid: Any, time_coords: Any | None = None):
        torch, _F = _import_torch()
        if frames.ndim != 5:
            raise ValueError(f"Temporal TCN probe expects [B,T,C,H,W], got {tuple(frames.shape)}")
        batch, dense_len, channels, height, width = frames.shape
        if int(channels) != 3:
            raise ValueError("Temporal TCN probe expects RGB frame tensors with 3 channels")
        frames = frames.float()
        if bool((frames.detach().abs().amax() > 2.0).item()):
            frames = frames / 255.0
        if self.variant == "motion":
            motion = torch.zeros_like(frames)
            if int(dense_len) > 1:
                motion[:, 1:] = (frames[:, 1:] - frames[:, :-1]).abs()
            frames = torch.cat([frames, motion], dim=2)
            channels = int(channels) * 2
        flat = frames.reshape(batch * dense_len, channels, height, width)
        features = self.spatial_stem(flat).flatten(1).reshape(batch, dense_len, -1).transpose(1, 2)
        if self.temporal_branches is not None:
            features = torch.cat([branch(features) for branch in self.temporal_branches], dim=1)
        elif self.temporal is not None:
            features = self.temporal(features)
        logits = self.classifier(features).squeeze(1)
        if hasattr(valid, "to"):
            valid = valid.to(device=logits.device).bool()
        return logits.masked_fill(~valid, 0.0)

    def train(self):
        self.module.train()
        return self

    def eval(self):
        self.module.eval()
        return self

    def to(self, *args, **kwargs):
        self.module.to(*args, **kwargs)
        return self

    def parameters(self):
        return self.module.parameters()

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict):
        return self.module.load_state_dict(state_dict)


def make_lowres_descriptors(inputs: Any, *, scout_spatial_size: int = 32, normalize: bool = True):
    """Match the C3 compressed-pixel scout descriptor contract: [B,T,3*S*S]."""

    torch, F = _import_torch()
    if inputs.ndim == 6:
        video = inputs.float().mean(dim=1)
    elif inputs.ndim == 5:
        video = inputs.float()
    else:
        raise ValueError(f"unsupported input shape: {tuple(inputs.shape)}")
    if video.ndim != 5:
        raise ValueError(f"video tensor must be [B,C,T,H,W], got {tuple(video.shape)}")
    if normalize:
        max_abs = video.detach().abs().amax()
        if bool((max_abs > 2.0).item()):
            video = video / 255.0
        mean = video.mean(dim=(-2, -1), keepdim=True)
        std = video.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(1.0e-4)
        video = (video - mean) / std
    batch, channels, dense_len, height, width = video.shape
    frames = video.permute(0, 2, 1, 3, 4).reshape(batch * dense_len, channels, height, width)
    compressed = F.interpolate(
        frames,
        size=(int(scout_spatial_size), int(scout_spatial_size)),
        mode="bilinear",
        align_corners=False,
    )
    return compressed.reshape(batch, dense_len, channels * int(scout_spatial_size) * int(scout_spatial_size))


def make_lowres_frame_images(inputs: Any, *, spatial_size: int = 32, normalize: bool = False):
    """Prepare MobileNet probe images with contract [B,T,3,H,W]."""

    _torch, F = _import_torch()
    if inputs.ndim == 6:
        video = inputs.float().mean(dim=1)
    elif inputs.ndim == 5:
        video = inputs.float()
    else:
        raise ValueError(f"unsupported input shape: {tuple(inputs.shape)}")
    if video.ndim != 5:
        raise ValueError(f"video tensor must be [B,C,T,H,W], got {tuple(video.shape)}")
    batch, channels, dense_len, height, width = video.shape
    if int(channels) != 3:
        raise ValueError("MobileNetV3 probe expects RGB frame tensors with 3 channels")
    frames = video.permute(0, 2, 1, 3, 4).reshape(batch * dense_len, channels, height, width)
    resized = F.interpolate(
        frames,
        size=(int(spatial_size), int(spatial_size)),
        mode="bilinear",
        align_corners=False,
    )
    if normalize:
        max_abs = resized.detach().abs().amax()
        if bool((max_abs > 2.0).item()):
            resized = resized / 255.0
    return resized.reshape(batch, dense_len, channels, int(spatial_size), int(spatial_size))


def prepare_probe_inputs(inputs: Any, *, probe_model: str, spatial_size: int):
    if probe_model == "c3-reader":
        return make_lowres_descriptors(inputs, scout_spatial_size=int(spatial_size))
    if probe_model in {"mobilenetv3", "temporal-tcn"}:
        return make_lowres_frame_images(inputs, spatial_size=int(spatial_size))
    raise ValueError(f"unsupported probe_model: {probe_model}")


def _targets_to_torch(valid: Any, gt_segments: Sequence[Any], *, device: Any):
    torch, _F = _import_torch()
    targets = build_action_targets(valid, gt_segments)
    return torch.tensor(targets, dtype=torch.float32, device=device)


def _batch_gt_segments(batch: Mapping[str, Any]) -> Sequence[Any]:
    if "gt_segments" not in batch:
        raise ValueError("batch is missing gt_segments")
    return batch["gt_segments"]


def _batch_inputs(batch: Mapping[str, Any]):
    if "inputs" in batch:
        return batch["inputs"]
    if "imgs" in batch:
        return batch["imgs"]
    raise ValueError("batch is missing inputs/imgs")


def _resolve_sample_ids(batch: Mapping[str, Any], *, batch_idx: int, batch_size: int) -> list[str]:
    def _coerce_sequence(value: Any) -> list[str] | None:
        if isinstance(value, str):
            return [value]
        if isinstance(value, Sequence):
            return ["" if item is None else str(item) for item in value]
        return None

    for key in ("sample_ids", "sample_id", "video_names", "video_name"):
        if key not in batch:
            continue
        resolved = _coerce_sequence(batch[key])
        if resolved is None:
            continue
        if len(resolved) == 1 and batch_size > 1:
            return [resolved[0] for _ in range(batch_size)]
        if len(resolved) == batch_size:
            return resolved
        if len(resolved) > 1:
            return resolved[:batch_size]

    metas = batch.get("metas")
    if isinstance(metas, Sequence) and not isinstance(metas, (str, bytes)):
        resolved: list[str] = []
        for meta in metas[:batch_size]:
            if isinstance(meta, Mapping):
                value = meta.get("sample_id") or meta.get("sample_ids") or meta.get("video_name") or meta.get("video_names")
                if value is not None:
                    resolved.append(str(value))
                    continue
            resolved.append("")
        if any(item for item in resolved):
            return [item if item else f"batch_{int(batch_idx):05d}|sample_{sample_idx:05d}" for sample_idx, item in enumerate(resolved)]

    return [f"batch_{int(batch_idx):05d}|sample_{sample_idx:05d}" for sample_idx in range(max(int(batch_size), 0))]



_LARGE_METRIC_DETAIL_KEYS = {
    "longest_selected_run_by_window",
    "selected_indices",
    "selected_run_count_by_window",
    "selected_run_lengths_by_window",
}


def _compact_metric_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        compact: dict[str, Any] = {}
        for key, item in value.items():
            if key in _LARGE_METRIC_DETAIL_KEYS:
                if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
                    compact[f"{key}_omitted_count"] = len(item)
                else:
                    compact[f"{key}_omitted"] = True
                continue
            compact[str(key)] = _compact_metric_payload(item)
        return compact
    if isinstance(value, list):
        return [_compact_metric_payload(item) for item in value]
    return value


def _emit_progress(progress_path: Path | None, event: str, **payload: Any) -> None:
    record = {
        "event": event,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **_compact_metric_payload(payload),
    }
    line = json.dumps(record, sort_keys=True)
    print(line, flush=True)
    if progress_path is not None:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def train_one_epoch(
    *,
    model: Any,
    dataloader: Iterable[Mapping[str, Any]],
    optimizer: Any,
    device: str,
    scout_spatial_size: int,
    probe_model: str,
    max_batches: int,
    epoch: int,
    total_epochs: int,
    progress_path: Path | None,
    log_every_batches: int,
    tcn_variant: str | None = None,
) -> dict[str, Any]:
    torch, F = _import_torch()
    model.train()
    loss_sum = 0.0
    batch_count = 0
    total_batches = len(dataloader) if hasattr(dataloader, "__len__") else None
    if max_batches > 0 and total_batches is not None:
        total_batches = min(total_batches, max_batches)
    start_time = time.time()
    _emit_progress(
        progress_path,
        "train_epoch_start",
        epoch=epoch,
        total_epochs=total_epochs,
        expected_batches=total_batches,
        tcn_variant=tcn_variant,
    )
    for batch_idx, batch in enumerate(dataloader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        inputs = _batch_inputs(batch).to(device)
        valid = batch["masks"].to(device).bool()
        probe_inputs = prepare_probe_inputs(inputs, probe_model=probe_model, spatial_size=scout_spatial_size)
        target = _targets_to_torch(valid, _batch_gt_segments(batch), device=probe_inputs.device)
        logits = model(probe_inputs, valid)
        loss = F.binary_cross_entropy_with_logits(logits[valid], target[valid])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.detach().cpu().item())
        batch_count += 1
        if log_every_batches > 0 and (batch_count == 1 or batch_count % log_every_batches == 0):
            _emit_progress(
                progress_path,
                "train_batch",
                epoch=epoch,
                batch=batch_count,
                expected_batches=total_batches,
                loss=loss_sum / float(max(batch_count, 1)),
                last_loss=float(loss.detach().cpu().item()),
                tcn_variant=tcn_variant,
            )
    if batch_count <= 0:
        raise ValueError("train_one_epoch processed zero batches; check max_train_batches and the dataloader")
    stats = {
        "loss": loss_sum / float(max(batch_count, 1)),
        "batches": batch_count,
        "seconds": time.time() - start_time,
    }
    _emit_progress(progress_path, "train_epoch_end", epoch=epoch, total_epochs=total_epochs, tcn_variant=tcn_variant, **stats)
    return stats


def evaluate(
    *,
    model: Any,
    dataloader: Iterable[Mapping[str, Any]],
    device: str,
    scout_spatial_size: int,
    probe_model: str,
    max_batches: int,
    epoch: int,
    total_epochs: int,
    progress_path: Path | None,
    log_every_batches: int,
    coverage_budget_fraction: float,
    coverage_budget: int | None,
    boundary_radius: int,
    sample_jsonl_path: Path | None = None,
    tcn_variant: str | None = None,
) -> dict[str, Any]:
    torch, _F = _import_torch()
    model.eval()
    all_logits: list[list[float]] = []
    all_targets: list[list[float]] = []
    all_valid: list[list[bool]] = []
    all_gt_segments: list[Any] = []
    all_sample_ids: list[str] = []
    batch_count = 0
    total_batches = len(dataloader) if hasattr(dataloader, "__len__") else None
    if max_batches > 0 and total_batches is not None:
        total_batches = min(total_batches, max_batches)
    start_time = time.time()
    _emit_progress(
        progress_path,
        "val_epoch_start",
        epoch=epoch,
        total_epochs=total_epochs,
        expected_batches=total_batches,
        tcn_variant=tcn_variant,
    )
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            inputs = _batch_inputs(batch)
            if hasattr(inputs, "to"):
                inputs = inputs.to(device)
            valid = batch["masks"].to(device).bool()
            batch_size = int(valid.shape[0]) if hasattr(valid, "shape") else len(valid)
            all_sample_ids.extend(_resolve_sample_ids(batch, batch_idx=batch_idx, batch_size=batch_size))
            probe_inputs = prepare_probe_inputs(inputs, probe_model=probe_model, spatial_size=scout_spatial_size)
            target = _targets_to_torch(valid, _batch_gt_segments(batch), device=probe_inputs.device)
            logits = model(probe_inputs, valid)
            all_logits.extend(_as_nested_list(logits))
            all_targets.extend(_as_nested_list(target))
            all_valid.extend(_as_nested_list(valid))
            all_gt_segments.extend(_as_nested_list(_batch_gt_segments(batch)))
            batch_count += 1
            if log_every_batches > 0 and (batch_count == 1 or batch_count % log_every_batches == 0):
                _emit_progress(
                    progress_path,
                    "val_batch",
                    epoch=epoch,
                    batch=batch_count,
                    expected_batches=total_batches,
                    tcn_variant=tcn_variant,
                )
    if batch_count <= 0:
        raise ValueError("evaluate processed zero validation batches; check max_val_batches and the dataloader")
    metrics = compute_binary_action_metrics(all_logits, all_targets, all_valid)
    metrics["target_positive_frames"] = metrics["positive_count"]
    metrics["target_negative_frames"] = metrics["negative_count"]
    metrics["sampling_quality"] = compute_sampling_quality_from_logits(
        logits=all_logits,
        target=all_targets,
        valid=all_valid,
        gt_segments=all_gt_segments,
        budget=coverage_budget,
        budget_fraction=float(coverage_budget_fraction),
        boundary_radius=int(boundary_radius),
    )
    if sample_jsonl_path is not None:
        indirect_quality = compute_indirect_selection_quality_from_logits(
            logits=all_logits,
            target=all_targets,
            valid=all_valid,
            gt_segments=all_gt_segments,
            sample_ids=all_sample_ids,
            budget=coverage_budget,
            budget_fraction=float(coverage_budget_fraction),
            boundary_radius=int(boundary_radius),
        )
        sample_rows = []
        for row in indirect_quality["per_sample_rows"]:
            sample_row = dict(row)
            sample_row["probe_model"] = probe_model
            sample_row["tcn_variant"] = tcn_variant
            sample_row["spatial_size"] = int(scout_spatial_size)
            sample_rows.append(sample_row)
        _write_jsonl(sample_jsonl_path, sample_rows)
        compact_indirect_quality = {
            key: value for key, value in indirect_quality.items() if key not in {"per_sample", "per_sample_rows"}
        }
        compact_indirect_quality["sample_count"] = len(sample_rows)
        metrics["indirect_selection_sample_jsonl"] = str(sample_jsonl_path)
        metrics["indirect_selection_quality"] = compact_indirect_quality
        metrics["probe_model"] = probe_model
        metrics["tcn_variant"] = tcn_variant
        metrics["spatial_size"] = int(scout_spatial_size)
        metrics["indirect_selection_baseline"] = compact_indirect_quality.get("baseline")
        metrics["indirect_selection_delta"] = compact_indirect_quality.get("delta")
    metrics["batches"] = batch_count
    metrics["seconds"] = time.time() - start_time
    metrics["tcn_variant"] = tcn_variant
    _emit_progress(progress_path, "val_epoch_end", epoch=epoch, total_epochs=total_epochs, **metrics)
    return metrics


def _load_cfg(config_path: str):
    _ensure_project_importable()
    from mmengine.config import Config

    return Config.fromfile(config_path)


def _get_config_section(container: Any, key: str) -> Any:
    if isinstance(container, Mapping):
        return container[key]
    return getattr(container, key)


def _has_config_section(container: Any, key: str) -> bool:
    if isinstance(container, Mapping):
        return key in container
    return hasattr(container, key)


def apply_dataset_overrides(
    cfg: Any,
    *,
    ann_file: str | None = None,
    class_map: str | None = None,
    train_data_path: str | None = None,
    val_data_path: str | None = None,
    test_data_path: str | None = None,
    train_subset_name: str | None = None,
    val_subset_name: str | None = None,
    test_subset_name: str | None = None,
) -> dict[str, str]:
    """Patch dataset paths in-memory for local diagnostics without editing the config file."""

    dataset = _get_config_section(cfg, "dataset")
    applied: dict[str, str] = {}
    common_paths = {"ann_file": ann_file, "class_map": class_map}
    split_paths = {
        "train": train_data_path,
        "val": val_data_path,
        "test": test_data_path,
    }
    split_subsets = {
        "train": train_subset_name,
        "val": val_subset_name,
        "test": test_subset_name,
    }

    for split_name in ("train", "val", "test"):
        if not _has_config_section(dataset, split_name):
            continue
        split_cfg = _get_config_section(dataset, split_name)
        for key, value in common_paths.items():
            if value:
                split_cfg[key] = value
                applied[key] = value
        split_path = split_paths[split_name]
        if split_path:
            split_cfg["data_path"] = split_path
            applied[f"{split_name}_data_path"] = split_path
        split_subset = split_subsets[split_name]
        if split_subset:
            split_cfg["subset_name"] = split_subset
            applied[f"{split_name}_subset_name"] = split_subset
    return applied


def _find_pipeline_step(pipeline: Sequence[Mapping[str, Any]], step_type: str) -> dict[str, Any]:
    for step in pipeline:
        if step.get("type") == step_type:
            return dict(step)
    raise ValueError(f"Required pipeline step not found: {step_type}")


def apply_fast_lowres_pipeline(
    cfg: Any,
    *,
    spatial_size: int = 32,
    probe_window_size: int | None = None,
) -> dict[str, str]:
    """Rewrite e2e video pipelines to a minimal low-res diagnostic path."""

    dataset = _get_config_section(cfg, "dataset")
    rewrites: dict[str, str] = {}
    for split_name in ("train", "val", "test"):
        if not _has_config_section(dataset, split_name):
            continue
        split_cfg = _get_config_section(dataset, split_name)
        pipeline = list(split_cfg.get("pipeline", []))
        if not pipeline:
            continue

        prepare = _find_pipeline_step(pipeline, "PrepareVideoInfo")
        decord_init = _find_pipeline_step(pipeline, "mmaction.DecordInit")
        load_frames = _find_pipeline_step(pipeline, "LoadFrames")
        collect = _find_pipeline_step(pipeline, "Collect")

        if probe_window_size is not None:
            if load_frames.get("method") == "random_trunc":
                load_frames["trunc_len"] = int(probe_window_size)
            elif load_frames.get("method") == "sliding_window":
                split_cfg["window_size"] = int(probe_window_size)
            if load_frames.get("method") == "random_fixed_subsample":
                load_frames["target_len"] = int(probe_window_size)
                if load_frames.get("method_base") == "random_trunc":
                    load_frames["source_len"] = int(probe_window_size)

        collect_keys = list(collect.get("keys", []))
        tensor_keys = ["imgs"]
        for key in ("gt_segments", "gt_labels"):
            if key in collect_keys:
                tensor_keys.append(key)

        split_cfg["pipeline"] = [
            prepare,
            decord_init,
            load_frames,
            {"type": "mmaction.DecordDecode"},
            {"type": "mmaction.Resize", "scale": (int(spatial_size), int(spatial_size)), "keep_ratio": False},
            {"type": "mmaction.FormatShape", "input_format": "NCTHW"},
            {"type": "ConvertToTensor", "keys": tensor_keys},
            {"type": "Collect", "inputs": collect.get("inputs", "imgs"), "keys": collect_keys},
        ]
        rewrites[split_name] = f"fast_lowres_{int(spatial_size)}"
    return rewrites


def _build_dataloaders(cfg: Any, *, batch_size: int, num_workers: int, seed: int):
    _ensure_project_importable()
    import opentad.datasets  # noqa: F401
    import opentad.datasets.transforms  # noqa: F401
    from opentad.datasets import build_dataset
    from opentad.datasets.builder import collate
    from torch.utils.data import DataLoader
    torch, _F = _import_torch()

    train_dataset = build_dataset(cfg.dataset.train)
    val_cfg = cfg.dataset.val if hasattr(cfg.dataset, "val") else cfg.dataset.train
    val_dataset = build_dataset(val_cfg)
    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        collate_fn=collate,
        drop_last=False,
        generator=train_generator,
        worker_init_fn=None if int(num_workers) <= 0 else _seed_worker_from_torch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=collate,
        drop_last=False,
        worker_init_fn=None if int(num_workers) <= 0 else _seed_worker_from_torch,
    )
    return train_loader, val_loader


def _reader_cfg_from_config(cfg: Any) -> dict[str, Any]:
    selector = cfg.model.get("frame_selector", {})
    reader_cfg = dict(selector.get("reader", default_reader_cfg()))
    reader_cfg.setdefault("type", DEFAULT_READER_TYPE)
    if reader_cfg["type"] not in SUPPORTED_C3_READER_TYPES:
        raise ValueError(
            f"action probe expects one of {sorted(SUPPORTED_C3_READER_TYPES)}, got {reader_cfg['type']}"
        )
    return reader_cfg


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate a C3 low-res action-vs-background frame probe.")
    parser.add_argument(
        "--config",
        default=DEFAULT_PROBE_CONFIG,
        help="C3 config used only for reader settings and data pipeline.",
    )
    parser.add_argument("--out-dir", default="logs/lowres_action_probe", help="Output directory for summary.json.")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--probe-model", choices=("c3-reader", "mobilenetv3", "temporal-tcn"), default="c3-reader")
    parser.add_argument("--scout-spatial-size", type=int, default=32)
    parser.add_argument("--mobilenet-sizes", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--tcn-variants", nargs="+", default=list(SUPPORTED_TCN_VARIANTS))
    parser.add_argument("--mobilenet-weights-path", default=None, help="Optional local MobileNetV3 checkpoint path for offline pretrained loading.")
    parser.add_argument("--probe-checkpoint", default=None, help="Optional full probe checkpoint saved by --save-checkpoint.")
    parser.add_argument("--mobilenet-pretrained", dest="mobilenet_pretrained", action="store_true", default=True)
    parser.add_argument("--no-mobilenet-pretrained", dest="mobilenet_pretrained", action="store_false")
    parser.add_argument("--freeze-backbone", dest="freeze_backbone", action="store_true", default=True)
    parser.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false")
    parser.add_argument("--coverage-only", action="store_true", help="Skip training and only evaluate logit sampling coverage.")
    parser.add_argument("--coverage-budget-fraction", type=float, default=0.5)
    parser.add_argument("--coverage-budget", type=int, default=None)
    parser.add_argument("--boundary-radius", type=int, default=1)
    parser.add_argument("--sample-jsonl", default=None, help="Optional path for per-sample indirect-selection JSONL export.")
    parser.add_argument("--max-train-batches", type=int, default=50, help="0 means no artificial train-batch cap.")
    parser.add_argument("--max-val-batches", type=int, default=50, help="0 means no artificial val-batch cap.")
    parser.add_argument("--log-every-batches", type=int, default=10)
    parser.add_argument("--ann-file", default=None, help="Optional local THUMOS annotation override.")
    parser.add_argument("--class-map", default=None, help="Optional local category_idx.txt override.")
    parser.add_argument("--train-data-path", default=None, help="Optional local train/video-validation mp4 directory.")
    parser.add_argument("--val-data-path", default=None, help="Optional local validation/test mp4 directory.")
    parser.add_argument("--test-data-path", default=None, help="Optional local test mp4 directory.")
    parser.add_argument("--train-subset-name", default=None, help="Optional dataset subset override for train split.")
    parser.add_argument("--val-subset-name", default=None, help="Optional dataset subset override for val split.")
    parser.add_argument("--test-subset-name", default=None, help="Optional dataset subset override for test split.")
    parser.add_argument("--fast-lowres-pipeline", action="store_true", help="Replace video augmentation with 32x32 probe pipeline.")
    parser.add_argument("--probe-window-size", type=int, default=None, help="Optional shorter frame window for fast local diagnostics.")
    parser.add_argument("--save-checkpoint", action="store_true")
    args = parser.parse_args(argv)
    if int(args.max_train_batches) < 0:
        parser.error("--max-train-batches must be >= 0")
    if int(args.max_val_batches) < 0:
        parser.error("--max-val-batches must be >= 0")
    unsupported_tcn_variants = [variant for variant in args.tcn_variants if variant not in SUPPORTED_TCN_VARIANTS]
    if unsupported_tcn_variants:
        parser.error(f"--tcn-variants must be drawn from {list(SUPPORTED_TCN_VARIANTS)}, got {unsupported_tcn_variants}")
    return args


def _active_tcn_variant(args: argparse.Namespace, explicit_variant: str | None = None) -> str | None:
    if args.probe_model != "temporal-tcn":
        return None
    variant = explicit_variant if explicit_variant is not None else getattr(args, "tcn_variant", None)
    if variant is None:
        variants = list(getattr(args, "tcn_variants", []))
        variant = variants[0] if variants else "lite"
    if variant not in SUPPORTED_TCN_VARIANTS:
        raise ValueError(f"unsupported temporal-tcn variant: {variant}")
    return str(variant)


def _build_probe_model(args: argparse.Namespace, cfg: Any, *, spatial_size: int):
    if args.probe_model == "c3-reader":
        reader_cfg = _reader_cfg_from_config(cfg)
        reader_cfg["in_dim"] = int(3 * int(spatial_size) * int(spatial_size))
        return C3LowResActionProbe(reader_cfg=reader_cfg), reader_cfg
    if args.probe_model == "mobilenetv3":
        return (
            C3MobileNetV3ActionProbe(
                pretrained=bool(args.mobilenet_pretrained),
                variant="small",
                freeze_backbone=bool(args.freeze_backbone),
                weights_path=args.mobilenet_weights_path,
            ),
            None,
        )
    if args.probe_model == "temporal-tcn":
        return (
            C3TemporalTCNActionProbe(
                variant=str(_active_tcn_variant(args)),
                spatial_size=int(spatial_size),
            ),
            None,
        )
    raise ValueError(f"unsupported probe_model: {args.probe_model}")


def _probe_out_dir(
    base_out_dir: Path,
    *,
    probe_model: str,
    spatial_size: int,
    multi_size: bool,
    tcn_variant: str | None = None,
    multi_variant: bool = False,
) -> Path:
    if probe_model == "mobilenetv3" and multi_size:
        return base_out_dir / f"mobilenetv3_{int(spatial_size)}"
    if probe_model == "temporal-tcn" and multi_variant:
        if not tcn_variant:
            raise ValueError("tcn_variant is required for temporal-tcn multi-variant output")
        return base_out_dir / f"temporal_tcn_{tcn_variant}_{int(spatial_size)}"
    return base_out_dir


def _combine_multisize_summaries(*, base_summary: Mapping[str, Any], summaries: Sequence[Mapping[str, Any]], args_out_dir: Path) -> dict[str, Any]:
    combined: dict[str, Any] = {
        "schema_version": "lowres_action_probe_multisize_v1",
        "purpose": base_summary.get("purpose", "diagnostic_only_action_vs_background_frame_probe"),
        "probe_model": base_summary.get("probe_model"),
        "diagnostic_only": True,
        "not_connected_to_detector": True,
        "no_detector_training": True,
        "no_detector_eval": True,
        "no_detector_map": True,
        "seed": base_summary.get("seed"),
        "out_dir": str(args_out_dir),
        "summaries": list(summaries),
    }
    summary_by_size: dict[int, Mapping[str, Any]] = {}
    for summary in summaries:
        size = summary.get("spatial_size")
        if size is None:
            continue
        summary_by_size[int(size)] = summary
        combined[f"mobilenetv3_{int(size)}"] = summary
    if 32 in summary_by_size and 64 in summary_by_size:
        final32 = summary_by_size[32].get("final_val", {})
        final64 = summary_by_size[64].get("final_val", {})
        ap32 = final32.get("average_precision")
        ap64 = final64.get("average_precision")
        roc32 = final32.get("roc_auc")
        roc64 = final64.get("roc_auc")
        combined["comparison"] = {
            "mobilenetv3_32": {
                "average_precision": ap32,
                "roc_auc": roc32,
                "balanced_accuracy": final32.get("balanced_accuracy"),
                "boundary_support_r1": final32.get("sampling_quality", {}).get("boundary_support_r1"),
                "action_positive_coverage": final32.get("sampling_quality", {}).get("action_positive_coverage"),
            },
            "mobilenetv3_64": {
                "average_precision": ap64,
                "roc_auc": roc64,
                "balanced_accuracy": final64.get("balanced_accuracy"),
                "boundary_support_r1": final64.get("sampling_quality", {}).get("boundary_support_r1"),
                "action_positive_coverage": final64.get("sampling_quality", {}).get("action_positive_coverage"),
            },
            "average_precision_delta_64_minus_32": None if ap32 is None or ap64 is None else float(ap64) - float(ap32),
            "roc_auc_delta_64_minus_32": None if roc32 is None or roc64 is None else float(roc64) - float(roc32),
            "boundary_support_delta_64_minus_32": None
            if final32.get("sampling_quality", {}).get("boundary_support_r1") is None
            or final64.get("sampling_quality", {}).get("boundary_support_r1") is None
            else float(final64.get("sampling_quality", {}).get("boundary_support_r1"))
            - float(final32.get("sampling_quality", {}).get("boundary_support_r1")),
        }
    return combined


def _combine_tcn_variant_summaries(*, base_summary: Mapping[str, Any], summaries: Sequence[Mapping[str, Any]], args_out_dir: Path) -> dict[str, Any]:
    combined: dict[str, Any] = {
        "schema_version": "lowres_action_probe_tcn_variants_v1",
        "purpose": base_summary.get("purpose", "diagnostic_only_action_vs_background_frame_probe"),
        "probe_model": "temporal-tcn",
        "diagnostic_only": True,
        "not_connected_to_detector": True,
        "no_detector_training": True,
        "no_detector_eval": True,
        "no_detector_map": True,
        "seed": base_summary.get("seed"),
        "out_dir": str(args_out_dir),
        "tcn_variants": [],
        "summaries": list(summaries),
    }
    average_precision_by_variant: dict[str, Any] = {}
    roc_auc_by_variant: dict[str, Any] = {}
    balanced_accuracy_by_variant: dict[str, Any] = {}
    boundary_support_by_variant: dict[str, Any] = {}
    best_indirect_strategy_by_variant: dict[str, Any] = {}
    indirect_boundary_support_by_variant: dict[str, Any] = {}
    for summary in summaries:
        variant = summary.get("tcn_variant")
        if not variant:
            continue
        variant = str(variant)
        combined["tcn_variants"].append(variant)
        combined[f"temporal_tcn_{variant}"] = summary
        final_val = summary.get("final_val", {})
        average_precision_by_variant[variant] = final_val.get("average_precision")
        roc_auc_by_variant[variant] = final_val.get("roc_auc")
        balanced_accuracy_by_variant[variant] = final_val.get("balanced_accuracy")
        boundary_support_by_variant[variant] = final_val.get("sampling_quality", {}).get("boundary_support_r1")
        strategy_comparison = final_val.get("indirect_selection_quality", {}).get("strategy_comparison", {})
        best_indirect_strategy_by_variant[variant] = strategy_comparison.get("best_boundary_support_strategy")
        indirect_boundary_support_by_variant[variant] = (
            strategy_comparison.get("boundary_support_r1_by_strategy")
            or strategy_comparison.get("boundary_support_by_strategy")
            or {}
        )

    valid_ap = {
        variant: float(value)
        for variant, value in average_precision_by_variant.items()
        if value is not None
    }
    best_average_precision_variant = None
    if valid_ap:
        best_average_precision_variant = max(valid_ap, key=lambda variant: valid_ap[variant])
    combined["comparison"] = {
        "average_precision_by_variant": average_precision_by_variant,
        "roc_auc_by_variant": roc_auc_by_variant,
        "balanced_accuracy_by_variant": balanced_accuracy_by_variant,
        "boundary_support_r1_by_variant": boundary_support_by_variant,
        "best_indirect_strategy_by_variant": best_indirect_strategy_by_variant,
        "indirect_boundary_support_r1_by_variant": indirect_boundary_support_by_variant,
        "best_average_precision_variant": best_average_precision_variant,
    }
    return combined


def _run_probe_experiment(
    *,
    args: argparse.Namespace,
    cfg: Any,
    spatial_size: int,
    multi_size: bool,
    seed: int,
    tcn_variant: str | None = None,
    multi_variant: bool = False,
) -> dict[str, Any]:
    torch, _F = _import_torch()
    _seed_everything(seed)
    cfg = copy.deepcopy(cfg)
    dataset_overrides = apply_dataset_overrides(
        cfg,
        ann_file=args.ann_file,
        class_map=args.class_map,
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        test_data_path=args.test_data_path,
        train_subset_name=args.train_subset_name,
        val_subset_name=args.val_subset_name,
        test_subset_name=args.test_subset_name,
    )
    pipeline_rewrites: dict[str, str] = {}
    if args.fast_lowres_pipeline:
        pipeline_rewrites = apply_fast_lowres_pipeline(
            cfg,
            spatial_size=int(spatial_size),
            probe_window_size=args.probe_window_size,
        )
    train_loader, val_loader = _build_dataloaders(cfg, batch_size=args.batch_size, num_workers=args.num_workers, seed=seed)
    active_tcn_variant = _active_tcn_variant(args, tcn_variant)
    run_args = copy.copy(args)
    if active_tcn_variant is not None:
        setattr(run_args, "tcn_variant", active_tcn_variant)
    device = run_args.device
    model, reader_cfg = _build_probe_model(run_args, cfg, spatial_size=spatial_size)
    checkpoint_load_result = None
    if run_args.probe_checkpoint:
        checkpoint_load_result = str(_load_probe_checkpoint(model, run_args.probe_checkpoint))
    model = model.to(device)
    optimizer = None if run_args.coverage_only else torch.optim.AdamW(model.parameters(), lr=float(run_args.lr), weight_decay=0.01)
    out_dir = Path(run_args.out_dir)
    out_dir = _probe_out_dir(
        out_dir,
        probe_model=run_args.probe_model,
        spatial_size=spatial_size,
        multi_size=multi_size,
        tcn_variant=active_tcn_variant,
        multi_variant=multi_variant,
    )
    sample_jsonl_path = Path(run_args.sample_jsonl) if run_args.sample_jsonl and not multi_variant else out_dir / "samples.jsonl"
    progress_path = out_dir / "progress.jsonl"
    if progress_path.exists():
        progress_path.unlink()
    total_epochs = int(args.epochs)
    _emit_progress(
        progress_path,
        "run_start",
        config=str(run_args.config),
        device=device,
        epochs=total_epochs,
        train_batches=len(train_loader) if hasattr(train_loader, "__len__") else None,
        val_batches=len(val_loader) if hasattr(val_loader, "__len__") else None,
        max_train_batches=int(args.max_train_batches),
        max_val_batches=int(args.max_val_batches),
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        seed=int(seed),
        probe_model=run_args.probe_model,
        tcn_variant=active_tcn_variant,
        scout_spatial_size=int(spatial_size),
        spatial_size=int(spatial_size),
        coverage_only=bool(args.coverage_only),
        coverage_budget=args.coverage_budget,
        coverage_budget_fraction=float(args.coverage_budget_fraction),
        boundary_radius=int(args.boundary_radius),
        probe_checkpoint=str(run_args.probe_checkpoint) if run_args.probe_checkpoint else None,
        probe_checkpoint_load_result=checkpoint_load_result,
        dataset_overrides=dataset_overrides,
        pipeline_rewrites=pipeline_rewrites,
    )

    history = []
    val_metrics: dict[str, Any] = {}
    loop_epochs = 1 if args.coverage_only else total_epochs
    for _epoch in range(loop_epochs):
        epoch = _epoch + 1
        if args.coverage_only:
            train_stats = {"loss": None, "batches": 0, "seconds": 0.0, "skipped": "coverage_only"}
        else:
            if optimizer is None:
                raise RuntimeError("optimizer is required when coverage_only is false")
            train_stats = train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                device=device,
                scout_spatial_size=int(spatial_size),
                probe_model=run_args.probe_model,
                max_batches=int(args.max_train_batches),
                epoch=epoch,
                total_epochs=total_epochs,
                progress_path=progress_path,
                log_every_batches=int(args.log_every_batches),
                tcn_variant=active_tcn_variant,
            )
        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            scout_spatial_size=int(spatial_size),
            probe_model=run_args.probe_model,
            max_batches=int(args.max_val_batches),
            epoch=epoch,
            total_epochs=loop_epochs,
            progress_path=progress_path,
            log_every_batches=int(args.log_every_batches),
            coverage_budget_fraction=float(args.coverage_budget_fraction),
            coverage_budget=args.coverage_budget,
            boundary_radius=int(args.boundary_radius),
            sample_jsonl_path=sample_jsonl_path,
            tcn_variant=active_tcn_variant,
        )
        compact_val_metrics = _compact_metric_payload(val_metrics)
        history.append({"epoch": epoch, "train": train_stats, "val": compact_val_metrics})
        _emit_progress(
            progress_path,
            "epoch_summary",
            epoch=epoch,
            total_epochs=loop_epochs,
            train_loss=train_stats.get("loss"),
            train_batches=train_stats.get("batches"),
            val_roc_auc=val_metrics.get("roc_auc"),
            val_average_precision=val_metrics.get("average_precision"),
            val_best_f1=val_metrics.get("best_f1"),
            val_positive_rate=val_metrics.get("positive_rate"),
            val_batches=val_metrics.get("batches"),
            tcn_variant=active_tcn_variant,
        )

    final_val = _compact_metric_payload(val_metrics)
    summary = {
        "schema_version": "c3_lowres_action_probe_v0",
        "purpose": "diagnostic_only_action_vs_background_frame_probe",
        "config": str(run_args.config),
        "dataset_overrides": dataset_overrides,
        "pipeline_rewrites": pipeline_rewrites,
        "reader_type": reader_cfg.get("type") if reader_cfg is not None else None,
        "reader_cfg": reader_cfg,
        "probe_model": run_args.probe_model,
        "tcn_variant": active_tcn_variant,
        "spatial_size": int(spatial_size),
        "out_dir": str(out_dir),
        "diagnostic_only": True,
        "not_connected_to_detector": True,
        "supervision": "train_gt_action_inside_window_only",
        "no_detector_training": True,
        "no_detector_eval": True,
        "no_detector_map": True,
        "no_test_gt_selector_input": True,
        "coverage_only": bool(args.coverage_only),
        "coverage_budget": args.coverage_budget,
        "coverage_budget_fraction": float(args.coverage_budget_fraction),
        "boundary_radius": int(args.boundary_radius),
        "seed": int(seed),
        "mobilenet_weights_path": str(run_args.mobilenet_weights_path) if run_args.probe_model == "mobilenetv3" else None,
        "probe_checkpoint": str(run_args.probe_checkpoint) if run_args.probe_checkpoint else None,
        "probe_checkpoint_load_result": checkpoint_load_result,
        "mobilenet_pretrained": bool(run_args.mobilenet_pretrained) if run_args.probe_model == "mobilenetv3" else None,
        "freeze_backbone": bool(run_args.freeze_backbone) if run_args.probe_model == "mobilenetv3" else None,
        "history": history,
        "final_val": final_val,
    }
    _write_json(out_dir / "summary.json", summary)
    if args.save_checkpoint:
        torch.save({"probe_state_dict": model.state_dict(), "summary": summary}, out_dir / "probe_reader.pth")
    _emit_progress(progress_path, "run_end", final_val=final_val, summary_path=str(out_dir / "summary.json"))
    print(json.dumps(summary["final_val"], indent=2, sort_keys=True))
    return summary


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    torch, _F = _import_torch()
    if args.probe_model in {"c3-reader", "temporal-tcn"}:
        sizes = [int(args.scout_spatial_size)]
    else:
        sizes = [int(size) for size in args.mobilenet_sizes]
    if not sizes:
        raise ValueError("at least one probe spatial size is required")
    tcn_variants = list(args.tcn_variants) if args.probe_model == "temporal-tcn" else [None]
    if args.probe_model == "temporal-tcn" and not tcn_variants:
        raise ValueError("at least one temporal-tcn variant is required")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    if (args.probe_model == "mobilenetv3" and len(sizes) > 1) or (args.probe_model == "temporal-tcn" and len(tcn_variants) > 1):
        _seed_everything(args.seed)
    summaries = []
    if args.probe_model == "temporal-tcn":
        for variant in tcn_variants:
            summaries.append(
                _run_probe_experiment(
                    args=args,
                    cfg=_load_cfg(args.config),
                    spatial_size=sizes[0],
                    multi_size=False,
                    seed=int(args.seed),
                    tcn_variant=str(variant),
                    multi_variant=len(tcn_variants) > 1,
                )
            )
    else:
        summaries = [
            _run_probe_experiment(
                args=args,
                cfg=_load_cfg(args.config),
                spatial_size=size,
                multi_size=len(sizes) > 1,
                seed=int(args.seed),
            )
            for size in sizes
        ]
    if len(summaries) == 1:
        return summaries[0]
    base_summary = {
        "purpose": "diagnostic_only_action_vs_background_frame_probe",
        "probe_model": args.probe_model,
        "seed": int(args.seed),
    }
    if args.probe_model == "temporal-tcn":
        combined = _combine_tcn_variant_summaries(
            base_summary=base_summary,
            summaries=summaries,
            args_out_dir=Path(args.out_dir),
        )
    else:
        combined = _combine_multisize_summaries(
            base_summary=base_summary,
            summaries=summaries,
            args_out_dir=Path(args.out_dir),
        )
    _write_json(Path(args.out_dir) / "summary.json", combined)
    return combined


if __name__ == "__main__":
    main()
