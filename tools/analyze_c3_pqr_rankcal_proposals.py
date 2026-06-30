import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


DEFAULT_TIOU_THRESHOLDS = (0.3, 0.5, 0.7)
DEFAULT_TOPK = (1, 5, 10, 50, 100)


def _segment_iou(segment, candidates):
    if not candidates:
        return 0.0
    start, end = float(segment[0]), float(segment[1])
    best = 0.0
    for cand_start, cand_end in candidates:
        cand_start = float(cand_start)
        cand_end = float(cand_end)
        intersection = max(0.0, min(end, cand_end) - max(start, cand_start))
        union = max(1e-8, (end - start) + (cand_end - cand_start) - intersection)
        best = max(best, intersection / union)
    return best


def _pearson(xs, ys):
    if len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x <= 0 or den_y <= 0:
        return None
    return num / (den_x * den_y)


def _average_ranks(values):
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + 1 + end) / 2.0
        for pos in range(cursor, end):
            ranks[order[pos]] = rank
        cursor = end
    return ranks


def _spearman(xs, ys):
    if len(xs) < 2:
        return None
    return _pearson(_average_ranks(xs), _average_ranks(ys))


def _quantile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[idx]


def _load_annotations(annotation_path, subset):
    data = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    gt_by_video = {}
    for video_id, info in data["database"].items():
        if info.get("subset") != subset:
            continue
        seen = set()
        annotations = []
        for ann in info.get("annotations", []):
            start = float(ann["segment"][0])
            end = float(ann["segment"][1])
            label = ann["label"]
            if end <= start:
                continue
            key = (round(start, 3), round(end, 3), label)
            if key in seen:
                continue
            seen.add(key)
            annotations.append({"segment": [start, end], "label": label})
        gt_by_video[video_id] = annotations
    return gt_by_video


def _load_predictions(prediction_path):
    data = json.loads(Path(prediction_path).read_text(encoding="utf-8"))
    if "results" not in data:
        raise ValueError("prediction JSON must contain a top-level 'results' field")
    return data["results"]


def _prediction_records(predictions, gt_by_video):
    records = []
    per_video_counts = {}
    per_video_label_counts = defaultdict(lambda: defaultdict(int))

    for video_id, video_predictions in predictions.items():
        sorted_predictions = sorted(video_predictions, key=lambda item: float(item.get("score", 0.0)), reverse=True)
        per_video_counts[video_id] = len(sorted_predictions)
        gt = gt_by_video.get(video_id, [])
        gt_by_label = defaultdict(list)
        any_gt = []
        for ann in gt:
            gt_by_label[ann["label"]].append(ann["segment"])
            any_gt.append(ann["segment"])

        for rank, pred in enumerate(sorted_predictions, start=1):
            label = pred.get("label")
            segment = pred.get("segment", [0.0, 0.0])
            score = float(pred.get("score", 0.0))
            same_label_iou = _segment_iou(segment, gt_by_label.get(label, []))
            any_label_iou = _segment_iou(segment, any_gt)
            per_video_label_counts[video_id][label] += 1
            records.append(
                {
                    "video_id": video_id,
                    "rank": rank,
                    "label": label,
                    "score": score,
                    "start": float(segment[0]),
                    "end": float(segment[1]),
                    "max_iou_same_label": same_label_iou,
                    "max_iou_any_label": any_label_iou,
                    "has_same_label_gt": bool(gt_by_label.get(label, [])),
                    "video_prediction_count": len(sorted_predictions),
                }
            )
    return records, per_video_counts, per_video_label_counts


def _topk_recall(records, gt_by_video, topk_values, thresholds):
    by_video = defaultdict(list)
    for record in records:
        by_video[record["video_id"]].append(record)

    total_gt = sum(len(annotations) for annotations in gt_by_video.values())
    summary = {}
    for k in topk_values:
        topk_records_by_video = {
            video_id: [record for record in sorted(video_records, key=lambda item: item["rank"]) if record["rank"] <= k]
            for video_id, video_records in by_video.items()
        }
        for threshold in thresholds:
            covered = 0
            for video_id, annotations in gt_by_video.items():
                candidates = topk_records_by_video.get(video_id, [])
                for ann in annotations:
                    best = 0.0
                    for pred in candidates:
                        if pred["label"] != ann["label"]:
                            continue
                        best = max(best, _segment_iou(ann["segment"], [[pred["start"], pred["end"]]]))
                    if best >= threshold:
                        covered += 1
            key = f"top{k}_same_label_recall@{threshold:.1f}"
            summary[key] = None if total_gt == 0 else covered / total_gt
    return summary


def _score_bins(records, bins=10):
    if not records:
        return []
    ordered = sorted(records, key=lambda item: item["score"], reverse=True)
    result = []
    for bin_idx in range(bins):
        start = int(len(ordered) * bin_idx / bins)
        end = int(len(ordered) * (bin_idx + 1) / bins)
        chunk = ordered[start:end]
        if not chunk:
            continue
        result.append(
            {
                "score_rank_bin": bin_idx + 1,
                "count": len(chunk),
                "score_min": min(item["score"] for item in chunk),
                "score_max": max(item["score"] for item in chunk),
                "mean_iou_same_label": sum(item["max_iou_same_label"] for item in chunk) / len(chunk),
                "mean_iou_any_label": sum(item["max_iou_any_label"] for item in chunk) / len(chunk),
            }
        )
    return result


def analyze(prediction_path, annotation_path, subset="validation", topk_values=DEFAULT_TOPK, thresholds=DEFAULT_TIOU_THRESHOLDS):
    predictions = _load_predictions(prediction_path)
    gt_by_video = _load_annotations(annotation_path, subset)
    records, per_video_counts, per_video_label_counts = _prediction_records(predictions, gt_by_video)
    same_label_ious = [record["max_iou_same_label"] for record in records]
    any_label_ious = [record["max_iou_any_label"] for record in records]
    scores = [record["score"] for record in records]
    counts = list(per_video_counts.values())
    class_counts = [count for label_counts in per_video_label_counts.values() for count in label_counts.values()]

    summary = {
        "status": "PASS_DIAGNOSTIC_ANALYSIS",
        "diagnostic_only": True,
        "official_map_claim": False,
        "prediction_path": str(prediction_path),
        "annotation_path": str(annotation_path),
        "subset": subset,
        "videos_with_predictions": len(per_video_counts),
        "videos_with_ground_truth": len(gt_by_video),
        "total_predictions": len(records),
        "total_ground_truth_instances": sum(len(annotations) for annotations in gt_by_video.values()),
        "score_iou_same_label_pearson": _pearson(scores, same_label_ious),
        "score_iou_same_label_spearman": _spearman(scores, same_label_ious),
        "score_iou_any_label_pearson": _pearson(scores, any_label_ious),
        "score_iou_any_label_spearman": _spearman(scores, any_label_ious),
        "proposal_count_per_video": {
            "mean": None if not counts else sum(counts) / len(counts),
            "p50": _quantile(counts, 0.50),
            "p90": _quantile(counts, 0.90),
            "p99": _quantile(counts, 0.99),
            "max": None if not counts else max(counts),
        },
        "proposal_count_per_video_label": {
            "p90": _quantile(class_counts, 0.90),
            "p99": _quantile(class_counts, 0.99),
            "max": None if not class_counts else max(class_counts),
        },
        "rank_recall": _topk_recall(records, gt_by_video, topk_values, thresholds),
        "score_rank_bins": _score_bins(records),
    }
    return summary, records


def _find_nested_gpu_predictions(requested_prediction_path):
    requested_prediction_path = Path(requested_prediction_path)
    if requested_prediction_path.name != "result_detection.json":
        return []
    run_dir = requested_prediction_path.parent
    if not run_dir.is_dir():
        return []
    candidates = []
    for path in run_dir.rglob("result_detection.json"):
        parent_name = path.parent.name
        if parent_name.startswith("gpu") and "_id" in parent_name:
            candidates.append(path)
    return sorted(candidates, key=lambda path: str(path))


def _resolve_prediction_path(prediction_path):
    requested_prediction_path = Path(prediction_path)
    if requested_prediction_path.is_file():
        return requested_prediction_path, None, None
    candidates = _find_nested_gpu_predictions(requested_prediction_path)
    if len(candidates) == 1:
        discovery = {
            "status": "DISCOVERED_NESTED_GPU_ARTIFACT",
            "requested_path": str(requested_prediction_path),
            "resolved_path": str(candidates[0]),
        }
        return candidates[0], discovery, None
    if len(candidates) > 1:
        error = {
            "status": "AMBIGUOUS_PREDICTION_ARTIFACT",
            "diagnostic_only": True,
            "official_map_claim": False,
            "prediction_path": str(requested_prediction_path),
            "missing_prediction": True,
            "candidate_prediction_paths": [str(path) for path in candidates],
            "required_next_dump": "pass --prediction-json with the exact gpu*_id*/result_detection.json path",
        }
        return requested_prediction_path, None, error
    return requested_prediction_path, None, None


def _write_missing_artifact(output_path, prediction_path, annotation_path):
    summary = {
        "status": "MISSING_ARTIFACT",
        "diagnostic_only": True,
        "official_map_claim": False,
        "prediction_path": str(prediction_path),
        "annotation_path": str(annotation_path),
        "missing_prediction": not Path(prediction_path).is_file(),
        "missing_annotation": not Path(annotation_path).is_file(),
        "required_next_dump": "enable post_processing.save_dict=True for the next diagnostic validation",
    }
    if output_path is not None:
        Path(output_path).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def _write_prediction_resolution_error(output_path, error_summary, annotation_path):
    summary = dict(error_summary)
    summary["annotation_path"] = str(annotation_path)
    summary["missing_annotation"] = not Path(annotation_path).is_file()
    if output_path is not None:
        Path(output_path).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def _write_records_csv(records, csv_path):
    if csv_path is None:
        return
    fieldnames = [
        "video_id",
        "rank",
        "label",
        "score",
        "start",
        "end",
        "max_iou_same_label",
        "max_iou_any_label",
        "has_same_label_gt",
        "video_prediction_count",
    ]
    with Path(csv_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Diagnostic-only PQR score-IoU/rank/proposal overload analysis")
    parser.add_argument("--prediction", "--prediction-json", dest="prediction", required=True, help="Path to result_detection.json")
    parser.add_argument("--annotation", required=True, help="Path to thumos_14_anno.json")
    parser.add_argument("--subset", default="validation")
    parser.add_argument("--output", help="Path for JSON summary")
    parser.add_argument("--records-csv", help="Optional per-proposal CSV path")
    args = parser.parse_args(argv)

    prediction_path, prediction_discovery, prediction_error = _resolve_prediction_path(args.prediction)
    if prediction_error is not None:
        _write_prediction_resolution_error(args.output, prediction_error, args.annotation)
        return 2

    if not prediction_path.is_file() or not Path(args.annotation).is_file():
        _write_missing_artifact(args.output, args.prediction, args.annotation)
        return 2

    summary, records = analyze(prediction_path, args.annotation, subset=args.subset)
    if prediction_discovery is not None:
        summary["prediction_discovery"] = prediction_discovery
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_records_csv(records, args.records_csv)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
