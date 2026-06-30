import numpy as np

from .types import SoftBracket


def _segments_from_mask(mask):
    mask = np.asarray(mask, dtype=np.bool_).reshape(-1)
    spans = []
    start = None
    for idx, value in enumerate(mask.tolist()):
        if value and start is None:
            start = idx
        if start is not None and (not value or idx == mask.size - 1):
            end = idx if value and idx == mask.size - 1 else idx - 1
            spans.append((start, end))
            start = None
    return spans


def _expand(left, right, dense_T, radius):
    return max(0, int(left) - int(radius)), min(int(dense_T) - 1, int(right) + int(radius))


def build_soft_brackets(risk_map, action_threshold=0.55, rescue_threshold=0.72, max_brackets=8):
    dense_T = int(risk_map.dense_T)
    brackets = []
    hard_spans = _segments_from_mask(np.asarray(risk_map.actionness) >= float(action_threshold))
    for left, right in hard_spans:
        soft_left, soft_right = _expand(left, right, dense_T, radius=max(2, int(round((right - left + 1) * 0.25))))
        local = risk_map.soft_bracket_prior[soft_left : soft_right + 1]
        center = int(soft_left + int(np.argmax(local)))
        brackets.append(
            SoftBracket(
                bracket_id=len(brackets),
                left=soft_left,
                center=center,
                right=soft_right,
                dense_T=dense_T,
                hard_left=int(left),
                hard_right=int(right),
                hard_bracket=True,
                confidence=float(np.mean(risk_map.actionness[left : right + 1])),
                risk_mass=float(np.mean(local)),
                source="soft_prior_from_actionness_span",
            )
        )

    rescue_mask = (np.asarray(risk_map.rescue_score) >= float(rescue_threshold)) & (
        np.asarray(risk_map.actionness) < float(action_threshold)
    )
    for left, right in _segments_from_mask(rescue_mask):
        if len(brackets) >= int(max_brackets):
            break
        soft_left, soft_right = _expand(left, right, dense_T, radius=2)
        local = risk_map.rescue_score[soft_left : soft_right + 1]
        center = int(soft_left + int(np.argmax(local)))
        brackets.append(
            SoftBracket(
                bracket_id=len(brackets),
                left=soft_left,
                center=center,
                right=soft_right,
                dense_T=dense_T,
                hard_left=0,
                hard_right=0,
                hard_bracket=False,
                confidence=float(np.max(local)),
                risk_mass=float(np.mean(local)),
                source="recoverable_rescue_prior_outside_action_bracket",
            )
        )

    if not brackets:
        center = int(np.argmax(risk_map.rescue_score))
        left, right = _expand(center, center, dense_T, radius=3)
        brackets.append(
            SoftBracket(
                bracket_id=0,
                left=left,
                center=center,
                right=right,
                dense_T=dense_T,
                hard_left=0,
                hard_right=0,
                hard_bracket=False,
                confidence=float(risk_map.rescue_score[center]),
                risk_mass=float(np.mean(risk_map.rescue_score[left : right + 1])),
                source="fallback_soft_rescue_prior",
            )
        )
    return brackets[: int(max_brackets)]
