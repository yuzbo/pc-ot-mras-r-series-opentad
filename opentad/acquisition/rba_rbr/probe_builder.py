import numpy as np

from .types import ProbeCandidate, sorted_unique_positions


def _inside_any_hard(pos, brackets):
    for bracket in brackets:
        if bracket.hard_bracket and int(bracket.hard_left) <= int(pos) <= int(bracket.hard_right):
            return True
    return False


def _top_positions(score, count, blocked=None, min_distance=2):
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    blocked = set() if blocked is None else set(int(pos) for pos in blocked)
    order = np.argsort(-score)
    chosen = []
    for pos in order.tolist():
        if int(pos) in blocked:
            continue
        if any(abs(int(pos) - int(old)) < int(min_distance) for old in chosen):
            continue
        chosen.append(int(pos))
        if len(chosen) >= int(count):
            break
    return chosen


def build_scaffold_positions(dense_T, scaffold_k):
    dense_T = int(dense_T)
    scaffold_k = int(max(1, min(scaffold_k, dense_T)))
    if scaffold_k == 1:
        return [0]
    return sorted_unique_positions(np.round(np.linspace(0, dense_T - 1, scaffold_k)).astype(np.int64), dense_T)


def _probe(probe_id, stage, role, pos, dense_T, regret, source_signal, bracket_id=None, outside=False, reason=""):
    return ProbeCandidate(
        probe_id=int(probe_id),
        stage=stage,
        role=role,
        center_pos=int(pos),
        positions=[int(pos)],
        dense_T=int(dense_T),
        predicted_regret=float(regret),
        source_signal=source_signal,
        bracket_id=bracket_id,
        outside_hard_bracket=outside,
        reason=reason,
        components={source_signal: float(regret)},
    )


def build_probe_candidates(risk_map, brackets, scaffold_positions=None, video_id="synthetic", split="synthetic"):
    dense_T = int(risk_map.dense_T)
    probes = []
    used = set()
    scaffold_positions = build_scaffold_positions(dense_T, 4) if scaffold_positions is None else sorted_unique_positions(
        scaffold_positions, dense_T
    )
    for pos in scaffold_positions:
        used.add(int(pos))
        probes.append(
            _probe(
                len(probes),
                "scaffold",
                "scaffold_anchor",
                pos,
                dense_T,
                regret=max(0.05, float(risk_map.gap_risk[int(pos)])),
                source_signal="staleness_gap_scaffold",
                reason="coarse recoverable scaffold anchor",
            )
        )

    for bracket in brackets:
        refine_positions = [bracket.left, bracket.center, bracket.right]
        if bracket.hard_bracket:
            refine_positions.extend([bracket.hard_left, bracket.hard_right])
        for pos in sorted_unique_positions(refine_positions, dense_T):
            if pos in used:
                continue
            used.add(int(pos))
            score = float(
                0.45 * risk_map.soft_bracket_prior[pos]
                + 0.25 * risk_map.transition[pos]
                + 0.20 * risk_map.uncertainty[pos]
                + 0.10 * bracket.risk_mass
            )
            probes.append(
                _probe(
                    len(probes),
                    "refine",
                    "in_bracket_refine",
                    pos,
                    dense_T,
                    regret=score,
                    source_signal="soft_bracket_prior",
                    bracket_id=bracket.bracket_id,
                    outside=False,
                    reason="refine soft bracket without hard-masking other regions",
                )
            )

    rescue_scores = np.asarray(risk_map.rescue_score, dtype=np.float64).copy()
    for pos in range(dense_T):
        if _inside_any_hard(pos, brackets):
            rescue_scores[pos] *= 0.20
    for pos in _top_positions(rescue_scores, count=max(8, dense_T // 12), blocked=used, min_distance=2):
        outside = not _inside_any_hard(pos, brackets)
        if not outside and float(risk_map.rescue_score[pos]) < 0.85:
            continue
        role = "out_of_bracket_rescue" if outside else "conflict_rescue"
        source = max(
            (
                ("uncertainty", float(risk_map.uncertainty[pos])),
                ("transition", float(risk_map.transition[pos])),
                ("staleness", float(risk_map.staleness[pos])),
                ("conflict", float(risk_map.conflict[pos])),
                ("gap_risk", float(risk_map.gap_risk[pos])),
            ),
            key=lambda item: item[1],
        )[0]
        regret = float(risk_map.rescue_score[pos] + (0.18 if outside else 0.0))
        probes.append(
            _probe(
                len(probes),
                "rescue",
                role,
                pos,
                dense_T,
                regret=regret,
                source_signal=source,
                bracket_id=None,
                outside=outside,
                reason="recoverable rescue probe outside irreversible hard bracket",
            )
        )
    probes = sorted(probes, key=lambda probe: (-probe.predicted_regret, probe.stage, probe.center_pos, probe.probe_id))
    for rank, probe in enumerate(probes):
        probe.rank = int(rank)
    return probes
