import hashlib

import numpy as np

from .scaffold import build_scaffold_packets
from .sparse_gather import sparse_gather
from .types import METHOD_NAME, ROUTE_LABEL, sorted_unique_positions
from .validators import (
    build_original_time_metadata,
    build_selection_gap_diagnostics,
    exact_uniform_positions,
    validate_deploy_ledger,
)


def _stable_seed(value):
    digest = hashlib.sha1(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def _uniform_same_k(dense_T, k):
    return exact_uniform_positions(dense_T, k)


def _random_same_k(dense_T, k, seed):
    rng = np.random.RandomState(int(seed))
    k = min(int(k), int(dense_T))
    return sorted(rng.choice(np.arange(int(dense_T)), size=k, replace=False).astype(int).tolist())


def _max_gap(positions, dense_T, max_gap):
    return build_selection_gap_diagnostics(positions, dense_T, max_gap)["max_gap"]


def _gap_safe_random_same_k(dense_T, k, seed, max_gap):
    dense_T = int(dense_T)
    k = min(int(k), dense_T)
    rng = np.random.RandomState(int(seed))
    if k <= 0:
        return []
    base = exact_uniform_positions(dense_T, k)
    if _max_gap(base, dense_T, max_gap) > int(max_gap):
        raise ValueError(f"same-K max-gap control infeasible: dense_T={dense_T} k={k} max_gap={max_gap}")
    if k <= 2:
        return base
    jitter = max(1, int(max_gap) // 4)
    for _ in range(128):
        trial = [base[0]]
        for pos in base[1:-1]:
            trial.append(int(np.clip(pos + rng.randint(-jitter, jitter + 1), 0, dense_T - 1)))
        trial.append(base[-1])
        trial = sorted(set(trial))
        while len(trial) < k:
            candidate = int(rng.randint(0, dense_T))
            if candidate not in trial:
                trial.append(candidate)
                trial = sorted(trial)
        if len(trial) == k and _max_gap(trial, dense_T, max_gap) <= int(max_gap):
            return trial
    return base


def _ensure_gap_safe_same_k(positions, dense_T, k, max_gap):
    dense_T = int(dense_T)
    k = min(int(k), dense_T)
    positions = sorted_unique_positions(positions, dense_T)[:k]
    fallback = exact_uniform_positions(dense_T, k)
    if _max_gap(fallback, dense_T, max_gap) > int(max_gap):
        raise ValueError(f"same-K max-gap control infeasible: dense_T={dense_T} k={k} max_gap={max_gap}")
    for pos in fallback:
        if len(positions) >= k:
            break
        if pos not in positions:
            positions.append(pos)
            positions = sorted(positions)
    if len(positions) != k or _max_gap(positions, dense_T, max_gap) > int(max_gap):
        positions = fallback
    return positions


def _build_control_ledger(name, positions, dense_T, fps, video_id, window_id, split, dense_inputs, scaffold_k=0, max_gap=None):
    positions = sorted_unique_positions(positions, dense_T)
    if max_gap is None:
        max_gap = int(dense_T)
    _, evidence = sparse_gather(dense_inputs, positions, temporal_dim=0, detector_forward_exists=False)
    metadata = build_original_time_metadata(
        dense_T=dense_T,
        selected_positions=positions,
        fps=fps,
        window_start_sec=0.0,
        window_end_sec=float(dense_T) / float(fps),
    )
    ledger = {
        "route_label": ROUTE_LABEL,
        "method": f"{METHOD_NAME}:{name}",
        "control_name": name,
        "video_id": video_id,
        "split": split,
        "window_id": int(window_id),
        "dense_T": int(dense_T),
        "selected_positions": positions,
        "selected_times_sec": metadata["selected_times_sec"],
        "selected_positions_unit": "original_dense_index",
        "valid_k": int(len(positions)),
        "scaffold_k": int(scaffold_k),
        "min_k": int(len(positions)),
        "max_k": int(len(positions)),
        "budget_stop_reason": "candidate_exhausted",
        "selection_gap_diagnostics": build_selection_gap_diagnostics(positions, dense_T, max_gap),
        "selected_packet_ids": [],
        "selected_packet_roles": [],
        "predicted_regret_values": [],
        "expected_belief_reductions": [],
        "bracket_summary": {
            "num_brackets": 0,
            "num_active_brackets": 0,
            "mean_belief_entropy": 0.0,
            "mean_belief_width_p80": 0.0,
            "two_sided_witness_coverage_rate": 0.0,
        },
        "original_time_metadata": metadata,
        "real_sparse_evidence": evidence,
        "forbidden_fields_absent": {
            "regret_label_absent": True,
            "gt_fields_absent": True,
            "teacher_fields_absent": True,
            "prediction_cache_absent": True,
        },
        "provenance": {
            "selection_uses_gt": False,
            "selection_uses_teacher": False,
            "selection_uses_prediction_cache": False,
            "selection_uses_raw_detector_prediction": False,
            "selection_uses_oracle_boundary": False,
            "selection_uses_oracle_residual": False,
        },
        "uses_same_gather_time_validator_path": True,
        "dense_handoff_used": False,
    }
    validate_deploy_ledger(ledger)
    return ledger


def build_matched_controls(
    bvr_ledgers,
    candidate_packets_by_video=None,
    dense_inputs_by_video=None,
    fps=30.0,
    random_seed=17,
    mean_k=None,
    scaffold_k=4,
    max_gap=16,
):
    ledgers = list(bvr_ledgers)
    if not ledgers:
        raise ValueError("matched controls require at least one BVR ledger")
    candidate_packets_by_video = {} if candidate_packets_by_video is None else candidate_packets_by_video
    dense_inputs_by_video = {} if dense_inputs_by_video is None else dense_inputs_by_video
    if mean_k is None:
        mean_k = int(round(float(np.mean([row["valid_k"] for row in ledgers]))))

    controls = []
    for row in ledgers:
        dense_T = int(row["dense_T"])
        k = int(row["valid_k"])
        video_id = row["video_id"]
        window_id = int(row.get("window_id", 0))
        split = row.get("split", "synthetic")
        dense_inputs = dense_inputs_by_video.get(video_id)
        if dense_inputs is None:
            dense_inputs = np.arange(dense_T, dtype=np.float64).reshape(dense_T, 1)

        controls.append(
            _build_control_ledger(
                "same_k_uniform",
                _uniform_same_k(dense_T, k),
                dense_T,
                fps,
                video_id,
                window_id,
                split,
                dense_inputs,
                max_gap=max_gap,
            )
        )
        controls.append(
            _build_control_ledger(
                "same_mean_k_exact_uniform",
                _uniform_same_k(dense_T, mean_k),
                dense_T,
                fps,
                video_id,
                window_id,
                split,
                dense_inputs,
                max_gap=max_gap,
            )
        )
        controls.append(
            _build_control_ledger(
                "random_same_k",
                _gap_safe_random_same_k(dense_T, k, random_seed + _stable_seed(video_id), max_gap),
                dense_T,
                fps,
                video_id,
                window_id,
                split,
                dense_inputs,
                max_gap=max_gap,
            )
        )
        scaffold_packets = build_scaffold_packets(
            dense_T=dense_T,
            scaffold_k=min(scaffold_k, max(k, 1)),
            max_gap=max_gap,
            video_id=video_id,
            window_id=window_id,
            split=split,
        )
        scaffold_positions = []
        for packet in scaffold_packets:
            scaffold_positions.extend(packet.positions)
        controls.append(
            _build_control_ledger(
                "scaffold_only",
                _ensure_gap_safe_same_k(scaffold_positions, dense_T, max(1, min(len(scaffold_positions), k)), max_gap),
                dense_T,
                fps,
                video_id,
                window_id,
                split,
                dense_inputs,
                scaffold_k=len(scaffold_positions[: max(1, min(len(scaffold_positions), k))]),
                max_gap=max_gap,
            )
        )
        candidates = list(candidate_packets_by_video.get(video_id, []))
        twb_order = sorted(
            candidates,
            key=lambda packet: (
                packet.role not in {"transition_before", "transition_center", "transition_after", "short_action_guard", "gap_bridge"},
                packet.bracket_id if packet.bracket_id is not None else 10**9,
                packet.positions,
                packet.packet_id,
            ),
        )
        twb_positions = []
        for packet in twb_order:
            for pos in packet.positions:
                if pos not in twb_positions:
                    twb_positions.append(pos)
            if len(twb_positions) >= k:
                break
        if len(twb_positions) < k:
            for pos in _uniform_same_k(dense_T, k):
                if pos not in twb_positions:
                    twb_positions.append(pos)
                if len(twb_positions) >= k:
                    break
        controls.append(
            _build_control_ledger(
                "twb_no_regret",
                _ensure_gap_safe_same_k(twb_positions, dense_T, k, max_gap),
                dense_T,
                fps,
                video_id,
                window_id,
                split,
                dense_inputs,
                max_gap=max_gap,
            )
        )
    return controls
