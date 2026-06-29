import numpy as np

from .scaffold import gap_statistics
from .types import CandidatePacket, BracketState, ScoutCurves


def _clip_pos(pos, dense_T):
    return int(np.clip(int(pos), 0, int(dense_T) - 1))


def _features_for_pos(scout, bracket, pos, selected_positions=None):
    selected_positions = [] if selected_positions is None else sorted(selected_positions)
    if selected_positions:
        nearest_gap = min(abs(int(pos) - int(sel)) for sel in selected_positions)
    else:
        nearest_gap = int(scout.dense_T)
    return {
        "mean_actionness": float(scout.p_action[int(pos)]),
        "max_transition": float(max(scout.transition_score[int(pos)], bracket.peak_transition)),
        "mean_uncertainty": float(max(scout.uncertainty[int(pos)], bracket.mean_uncertainty)),
        "persistence": float(min(scout.persistence[int(pos)], bracket.persistence)),
        "short_action_risk": float(max(scout.short_action_risk[int(pos)], bracket.short_action_risk)),
        "two_sided_state_contrast": float(bracket.two_sided_state_contrast),
        "bracket_width_frames": float(bracket.width_frames),
        "gap_if_omitted_frames": float(max(nearest_gap, bracket.width_frames)),
        "gap_risk": float(bracket.gap_risk),
        "belief_entropy": float(bracket.entropy),
    }


def _packet(packet_id, role, source, pos, scout, bracket, video_id, window_id, split, rank, reason, required=False):
    return CandidatePacket(
        packet_id=int(packet_id),
        video_id=video_id,
        window_id=window_id,
        split=split,
        source=source,
        role=role,
        positions=[_clip_pos(pos, scout.dense_T)],
        dense_T=int(scout.dense_T),
        bracket_id=int(bracket.bracket_id),
        cost_frames=1.0,
        visibility=float(np.clip(1.0 - 0.35 * bracket.entropy, 0.0, 1.0)),
        rank=int(rank),
        reason=reason,
        feature_summary=_features_for_pos(scout, bracket, _clip_pos(pos, scout.dense_T)),
        required_for_role_coverage=bool(required),
    )


def build_witness_packets(
    scout: ScoutCurves,
    brackets,
    scaffold_positions=None,
    video_id="synthetic",
    window_id=0,
    split="synthetic",
    start_packet_id=1000,
):
    scaffold_positions = [] if scaffold_positions is None else list(scaffold_positions)
    packets = []
    next_id = int(start_packet_id)
    for bracket in brackets:
        if not isinstance(bracket, BracketState):
            raise TypeError("brackets must contain BracketState objects")
        width = max(1, bracket.width_frames)
        offset = max(1, min(width // 3, max(1, scout.dense_T // 32)))
        roles = [
            ("transition_before", bracket.center_pos - offset, "twb", "left transition witness", True),
            ("transition_center", bracket.center_pos, "twb", "center transition witness", True),
            ("transition_after", bracket.center_pos + offset, "twb", "right transition witness", True),
        ]
        if bracket.kind == "action_island":
            roles.append(("short_action_guard", bracket.center_pos, "short_guard", "short action risk guard", True))
        if bracket.kind in {"start_like", "end_like", "action_island"}:
            core_pos = bracket.center_pos + (offset if bracket.kind == "start_like" else -offset if bracket.kind == "end_like" else 0)
            roles.append(("action_core", core_pos, "twb", "nearby action core witness", False))
        if bracket.kind == "unknown_transition" or bracket.entropy > 0.45:
            roles.append(("ambiguity_probe", bracket.center_pos, "ambiguity", "ambiguous transition probe", False))

        seen = set()
        for role, pos, source, reason, required in roles:
            pos = _clip_pos(pos, scout.dense_T)
            key = (role, pos, bracket.bracket_id)
            if key in seen:
                continue
            seen.add(key)
            packets.append(
                _packet(
                    next_id,
                    role,
                    source,
                    pos,
                    scout,
                    bracket,
                    video_id,
                    window_id,
                    split,
                    rank=len(packets),
                    reason=reason,
                    required=required,
                )
            )
            next_id += 1

    stats = gap_statistics(scaffold_positions, scout.dense_T) if scaffold_positions else {"max_gap": scout.dense_T}
    if stats["max_gap"] > max(3, scout.dense_T // 8):
        ordered = sorted(scaffold_positions)
        edges = [-1] + ordered + [int(scout.dense_T)]
        for left, right in zip(edges[:-1], edges[1:]):
            if right - left - 1 <= max(3, scout.dense_T // 8):
                continue
            midpoint = _clip_pos((left + right) // 2, scout.dense_T)
            fake_bracket = BracketState(
                bracket_id=100000 + next_id,
                video_id=video_id,
                window_id=window_id,
                split=split,
                kind="stable_span",
                left_pos=max(0, midpoint - 1),
                center_pos=midpoint,
                right_pos=min(int(scout.dense_T) - 1, midpoint + 1),
                dense_T=int(scout.dense_T),
                peak_transition=float(scout.transition_score[midpoint]),
                mean_uncertainty=float(scout.uncertainty[midpoint]),
                action_left_mean=float(scout.p_action[midpoint]),
                action_right_mean=float(scout.p_action[midpoint]),
                persistence=float(scout.persistence[midpoint]),
                short_action_risk=float(scout.short_action_risk[midpoint]),
                gap_risk=1.0,
                entropy=float(scout.uncertainty[midpoint]),
                width_p50_frames=1.0,
                width_p80_frames=2.0,
                confidence=float(1.0 - scout.uncertainty[midpoint]),
                two_sided_state_contrast=0.0,
            )
            packet = _packet(
                next_id,
                "gap_bridge",
                "gap_repair",
                midpoint,
                scout,
                fake_bracket,
                video_id,
                window_id,
                split,
                rank=len(packets),
                reason="residual_gap_bridge_candidate",
                required=True,
            )
            packet.max_gap_repair = True
            packets.append(packet)
            next_id += 1
    return packets

