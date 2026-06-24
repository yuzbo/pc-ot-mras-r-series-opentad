"""Local pre-backbone selection quality diagnostics.

This tool is diagnostic-only. Ground-truth segments may be passed as an
offline audit input, but they are never a selector runtime dependency.
"""

import argparse
import json
import math
from pathlib import Path


def _as_numbers(values, name):
    if values is None:
        return []
    if not isinstance(values, list):
        raise ValueError(f"{name} must be a list")
    numbers = []
    for value in values:
        if not isinstance(value, (int, float)):
            raise ValueError(f"{name} must contain only numbers")
        numbers.append(value)
    return numbers


def _mean(values):
    if not values:
        return None
    return sum(values) / len(values)


def _percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * q
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def _round_float(value, digits=6):
    if value is None:
        return None
    return round(float(value), digits)


def _pearson(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = _mean(xs)
    y_mean = _mean(ys)
    x_centered = [x - x_mean for x in xs]
    y_centered = [y - y_mean for y in ys]
    numerator = sum(x * y for x, y in zip(x_centered, y_centered))
    x_norm = math.sqrt(sum(x * x for x in x_centered))
    y_norm = math.sqrt(sum(y * y for y in y_centered))
    if x_norm == 0 or y_norm == 0:
        return None
    return numerator / (x_norm * y_norm)


def _diagnose_gaps(indices):
    gaps = [abs(indices[i] - indices[i - 1]) for i in range(1, len(indices))]
    return {
        "count": len(gaps),
        "mean": _round_float(_mean(gaps)),
        "max": max(gaps) if gaps else None,
        "p95": _round_float(_percentile(gaps, 0.95)),
    }


def _coerce_segments(payload):
    segments = payload.get("gt_segments")
    if segments is None:
        segments = payload.get("segments")
    if segments is None:
        return []
    if not isinstance(segments, list):
        raise ValueError("gt_segments must be a list")

    coerced = []
    for segment in segments:
        if not isinstance(segment, list) or len(segment) != 2:
            raise ValueError("each gt segment must be a [start, end] list")
        start, end = segment
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            raise ValueError("gt segment boundaries must be numeric")
        coerced.append((float(start), float(end)))
    return coerced


def _diagnose_gt(indices, payload):
    segments = _coerce_segments(payload)
    if not segments:
        return {"provided": False}

    radius = float(payload.get("boundary_radius", 2))
    boundaries = [point for segment in segments for point in segment]
    supported = 0
    for boundary in boundaries:
        if any(abs(index - boundary) <= radius for index in indices):
            supported += 1

    near_selected = 0
    for index in indices:
        if any(abs(index - boundary) <= radius for boundary in boundaries):
            near_selected += 1

    boundary_count = len(boundaries)
    selected_count = len(indices)
    return {
        "provided": True,
        "boundary_radius": int(radius) if radius.is_integer() else radius,
        "gt_segment_count": len(segments),
        "boundary_count": boundary_count,
        "boundary_support_at_radius": supported,
        "boundary_support_rate": _round_float(supported / boundary_count if boundary_count else None),
        "boundary_near_selected_count": near_selected,
        "boundary_near_rate": _round_float(near_selected / selected_count if selected_count else None),
    }


def _diagnose_proposals(payload):
    proposals = payload.get("proposals")
    if not proposals:
        return {
            "provided": False,
            "score_rank_diagnostics": "placeholder: pass proposals with score and iou for rank diagnostics",
        }
    if not isinstance(proposals, list):
        raise ValueError("proposals must be a list")

    rows = []
    for proposal in proposals:
        if not isinstance(proposal, dict):
            raise ValueError("each proposal must be an object")
        score = proposal.get("score")
        iou = proposal.get("iou")
        if not isinstance(score, (int, float)) or not isinstance(iou, (int, float)):
            raise ValueError("each proposal must include numeric score and iou")
        rows.append({"score": float(score), "iou": float(iou)})

    ranked = sorted(rows, key=lambda row: row["score"], reverse=True)
    top_k = int(payload.get("top_k", min(10, len(ranked))))
    top_k = max(0, min(top_k, len(ranked)))
    positive_iou_threshold = float(payload.get("positive_iou_threshold", 0.5))
    positive_ranks = [
        rank
        for rank, row in enumerate(ranked, start=1)
        if row["iou"] >= positive_iou_threshold
    ]

    return {
        "provided": True,
        "proposal_count": len(rows),
        "score_iou_correlation": _round_float(
            _pearson([row["score"] for row in rows], [row["iou"] for row in rows])
        ),
        "top_k": top_k,
        "top_k_mean_iou": _round_float(_mean([row["iou"] for row in ranked[:top_k]])),
        "positive_iou_threshold": positive_iou_threshold,
        "positive_count": len(positive_ranks),
        "positive_rank_min": min(positive_ranks) if positive_ranks else None,
        "positive_rank_mean": _round_float(_mean(positive_ranks)),
    }


def diagnose_selection_quality(payload):
    """Return selection-risk diagnostics for one pre-backbone selection row."""
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")

    indices = _as_numbers(payload.get("selected_dense_indices"), "selected_dense_indices")
    if "valid_len" not in payload:
        raise ValueError("valid_len is required")
    valid_len = payload["valid_len"]
    if not isinstance(valid_len, int) or valid_len < 0:
        raise ValueError("valid_len must be a non-negative integer")

    selected_count = len(indices)
    unique_count = len(set(indices))
    duplicate_rate = 0.0 if selected_count == 0 else (selected_count - unique_count) / selected_count
    monotonic = all(indices[i] >= indices[i - 1] for i in range(1, selected_count))
    in_range_count = sum(1 for index in indices if 0 <= index < valid_len)

    summary = {
        "schema_version": "prebackbone_selection_quality_diagnostic_v0",
        "sample_id": payload.get("sample_id"),
        "selected_count": selected_count,
        "unique_selected_count": unique_count,
        "valid_len": valid_len,
        "selected_fraction": _round_float(selected_count / valid_len if valid_len else None),
        "monotonic": monotonic,
        "duplicate_rate": _round_float(duplicate_rate),
        "in_range_count": in_range_count,
        "out_of_range_count": selected_count - in_range_count,
        "gap": _diagnose_gaps(indices),
        "gt_diagnostics": _diagnose_gt(indices, payload),
        "proposal_diagnostics": _diagnose_proposals(payload),
        "protocol": {
            "diagnostic_only": True,
            "selector_runtime_uses_gt": False,
            "gt_runtime_allowed": False,
            "gt_usage_note": "GT segments are diagnostic input only and must not be used by selector runtime.",
            "runs_training": False,
            "runs_tools_test": False,
            "uses_remote": False,
        },
    }
    return summary


def diagnose_payload(payload):
    if isinstance(payload, list):
        samples = [diagnose_selection_quality(item) for item in payload]
        return {"schema_version": "prebackbone_selection_quality_batch_v0", "sample_count": len(samples), "samples": samples}
    if isinstance(payload, dict) and isinstance(payload.get("samples"), list):
        samples = [diagnose_selection_quality(item) for item in payload["samples"]]
        return {"schema_version": "prebackbone_selection_quality_batch_v0", "sample_count": len(samples), "samples": samples}
    return diagnose_selection_quality(payload)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose pre-backbone selected_dense_indices quality from JSON."
    )
    parser.add_argument("--input", required=True, help="Input JSON file with one row, a list, or {'samples': [...]} payload.")
    parser.add_argument("--output", help="Optional output JSON path.")
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation for stdout/output.")
    return parser.parse_args()


def main():
    args = _parse_args()
    input_path = Path(args.input)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    summary = diagnose_payload(payload)
    text = json.dumps(summary, indent=args.indent, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
