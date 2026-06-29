import numpy as np

from .types import CandidatePacket, sorted_unique_positions


def endpoint_linspace_positions(dense_T, scaffold_k):
    dense_T = int(dense_T)
    scaffold_k = int(scaffold_k)
    if dense_T <= 0:
        raise ValueError("dense_T must be positive")
    if scaffold_k <= 0:
        return []
    if scaffold_k == 1:
        return [0]
    raw = np.linspace(0, dense_T - 1, num=min(scaffold_k, dense_T))
    return sorted_unique_positions(np.rint(raw).astype(np.int64).tolist(), dense_T)


def repair_positions_for_max_gap(positions, dense_T, max_gap):
    dense_T = int(dense_T)
    max_gap = int(max_gap)
    selected = set(sorted_unique_positions(positions, dense_T))
    if dense_T <= 0 or max_gap < 1:
        raise ValueError("dense_T and max_gap must be positive")
    if not selected:
        selected.add(0)
        if dense_T > 1:
            selected.add(dense_T - 1)

    def largest_gap():
        ordered = sorted(selected)
        edges = [-1] + ordered + [dense_T]
        gaps = []
        for left, right in zip(edges[:-1], edges[1:]):
            gaps.append((right - left, left, right))
        return max(gaps, key=lambda item: (item[0], -item[1]))

    bridges = []
    while True:
        gap, left, right = largest_gap()
        if gap <= max_gap + 1:
            break
        midpoint = int((left + right) // 2)
        midpoint = int(np.clip(midpoint, 0, dense_T - 1))
        if midpoint in selected:
            break
        selected.add(midpoint)
        bridges.append(midpoint)
    return sorted(selected), bridges


def gap_statistics(positions, dense_T):
    selected = sorted_unique_positions(positions, dense_T)
    if not selected:
        return {"max_gap": int(dense_T), "mean_gap": float(dense_T), "gaps": []}
    edges = [-1] + selected + [int(dense_T)]
    gaps = [int(right - left - 1) for left, right in zip(edges[:-1], edges[1:])]
    return {
        "max_gap": int(max(gaps) if gaps else 0),
        "mean_gap": float(np.mean(gaps) if gaps else 0.0),
        "gaps": gaps,
    }


def build_scaffold_packets(
    dense_T,
    scaffold_k,
    max_gap,
    video_id="synthetic",
    window_id=0,
    split="synthetic",
    start_packet_id=0,
):
    anchors = endpoint_linspace_positions(dense_T, scaffold_k)
    repaired, bridges = repair_positions_for_max_gap(anchors, dense_T, max_gap)
    bridge_set = set(bridges)
    packets = []
    for rank, pos in enumerate(repaired):
        is_bridge = int(pos) in bridge_set
        is_endpoint = int(pos) in {0, int(dense_T) - 1}
        packets.append(
            CandidatePacket(
                packet_id=int(start_packet_id + rank),
                video_id=video_id,
                window_id=window_id,
                split=split,
                source="gap_repair" if is_bridge else "scaffold",
                role="gap_bridge" if is_bridge else "scaffold_anchor",
                positions=[int(pos)],
                dense_T=int(dense_T),
                cost_frames=1.0,
                visibility=1.0,
                rank=rank,
                reason="risk_constrained_gap_candidate" if is_bridge else "low_cost_scaffold_safety_candidate",
                feature_summary={
                    "mean_actionness": 0.0,
                    "max_transition": 0.0,
                    "mean_uncertainty": 0.05 if is_endpoint else 0.0,
                    "persistence": 1.0,
                    "short_action_risk": 0.0,
                    "two_sided_state_contrast": 0.0,
                    "bracket_width_frames": 0.0,
                    "gap_if_omitted_frames": float(max_gap if is_bridge else 0),
                    "gap_risk": float(1.0 if is_bridge else 0.15 if is_endpoint else 0.05),
                    "belief_entropy": 0.0,
                    "safety_floor_candidate": 1.0 if is_endpoint else 0.0,
                },
                required_for_scaffold=False,
                safety_floor=bool(is_endpoint),
                max_gap_repair=is_bridge,
            )
        )
    return packets
