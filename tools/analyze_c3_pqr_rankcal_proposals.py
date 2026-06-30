import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


DEFAULT_TIOU_THRESHOLDS = (0.3, 0.5, 0.7)
DEFAULT_TOPK = (1, 5, 10, 50, 100)
SWEEP_TIOU_THRESHOLDS = (0.5, 0.7)
SWEEP_MAX_PER_VIDEO = (1, 2, 5, 10, 50, 100, 200, 500)
SWEEP_PER_CLASS_CAP = (1, 2, 5, 10, 20, 50, 100)
SWEEP_MIN_SCORE = (0.0, 0.001, 0.01, 0.05, 0.10, 0.20, 0.30, 0.50)
SWEEP_NMS_THRESHOLDS = (0.3, 0.5, 0.7, 0.9)
SWEEP_SCORE_ALPHA = (0.0, 0.05, 0.10, 0.20, 0.30)
QUALITY_KEYS = ("quality_score", "quality")
CLS_SCORE_KEYS = ("cls_score", "class_score", "model_score")
FUSED_SCORE_KEYS = ("fused_score", "score_fused")
SELECTED_SEGMENT_KEYS = ("selected_segment", "segment_selected")
PHYSICAL_SEGMENT_KEYS = ("physical_segment", "segment_physical")
QC_V2_FLOAT_KEYS = (
    "selected_length",
    "physical_length",
    "proposal_width",
    "gap_mean",
    "visibility_support",
    "coverage",
    "endpoint_support",
)
QC_V2_INT_KEYS = ("level_id", "point_index")


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


def _first_present(mapping, keys):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _as_float_or_none(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _segment_or_none(value):
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    start = _as_float_or_none(value[0])
    end = _as_float_or_none(value[1])
    if start is None or end is None:
        return None
    return [start, end]


def _count_summary(counts):
    return {
        "mean": None if not counts else sum(counts) / len(counts),
        "p50": _quantile(counts, 0.50),
        "p90": _quantile(counts, 0.90),
        "p99": _quantile(counts, 0.99),
        "max": None if not counts else max(counts),
    }


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
            segment = _segment_or_none(pred.get("segment")) or [0.0, 0.0]
            selected_segment = _segment_or_none(_first_present(pred, SELECTED_SEGMENT_KEYS))
            physical_segment = _segment_or_none(_first_present(pred, PHYSICAL_SEGMENT_KEYS))
            score = float(pred.get("score", 0.0))
            quality_score = _as_float_or_none(_first_present(pred, QUALITY_KEYS))
            cls_score = _as_float_or_none(_first_present(pred, CLS_SCORE_KEYS))
            fused_score = _as_float_or_none(_first_present(pred, FUSED_SCORE_KEYS))
            same_label_iou = _segment_iou(segment, gt_by_label.get(label, []))
            any_label_iou = _segment_iou(segment, any_gt)
            per_video_label_counts[video_id][label] += 1
            record = {
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
                "quality_score": quality_score,
                "cls_score": cls_score,
                "fused_score": fused_score,
                "selected_start": None if selected_segment is None else selected_segment[0],
                "selected_end": None if selected_segment is None else selected_segment[1],
                "physical_start": None if physical_segment is None else physical_segment[0],
                "physical_end": None if physical_segment is None else physical_segment[1],
                "coverage_available": bool(pred.get("coverage_available", False)),
            }
            for key in QC_V2_FLOAT_KEYS:
                record[key] = _as_float_or_none(pred.get(key))
            for key in QC_V2_INT_KEYS:
                value = _as_float_or_none(pred.get(key))
                record[key] = None if value is None else int(value)
            records.append(record)
    return records, per_video_counts, per_video_label_counts


def _count_records(records):
    per_video_counts = defaultdict(int)
    per_video_label_counts = defaultdict(lambda: defaultdict(int))
    for record in records:
        per_video_counts[record["video_id"]] += 1
        per_video_label_counts[record["video_id"]][record["label"]] += 1
    return per_video_counts, per_video_label_counts


def _rerank_records(records):
    by_video = defaultdict(list)
    for record in records:
        by_video[record["video_id"]].append(dict(record))
    reranked = []
    for video_id, video_records in by_video.items():
        sorted_records = sorted(video_records, key=lambda item: float(item.get("score", 0.0)), reverse=True)
        for rank, record in enumerate(sorted_records, start=1):
            record["rank"] = rank
            record["video_prediction_count"] = len(sorted_records)
            reranked.append(record)
    return reranked


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


def _top_score_decile_mean_iou(records, iou_key):
    if not records:
        return None
    ordered = sorted(records, key=lambda item: item["score"], reverse=True)
    decile_count = max(1, int(math.ceil(len(ordered) / 10.0)))
    top_decile = ordered[:decile_count]
    if not top_decile:
        return None
    return sum(item[iou_key] for item in top_decile) / len(top_decile)


def _analysis_summary_from_records(
    records,
    gt_by_video,
    per_video_counts,
    per_video_label_counts,
    topk_values,
    thresholds,
):
    same_label_ious = [record["max_iou_same_label"] for record in records]
    any_label_ious = [record["max_iou_any_label"] for record in records]
    scores = [record["score"] for record in records]
    counts = list(per_video_counts.values())
    class_counts = [count for label_counts in per_video_label_counts.values() for count in label_counts.values()]

    return {
        "videos_with_predictions": len(per_video_counts),
        "videos_with_ground_truth": len(gt_by_video),
        "total_predictions": len(records),
        "total_ground_truth_instances": sum(len(annotations) for annotations in gt_by_video.values()),
        "score_iou_same_label_pearson": _pearson(scores, same_label_ious),
        "score_iou_same_label_spearman": _spearman(scores, same_label_ious),
        "score_iou_any_label_pearson": _pearson(scores, any_label_ious),
        "score_iou_any_label_spearman": _spearman(scores, any_label_ious),
        "proposal_count_per_video": _count_summary(counts),
        "proposal_count_per_video_label": _count_summary(class_counts),
        "rank_recall": _topk_recall(records, gt_by_video, topk_values, thresholds),
        "score_rank_bins": _score_bins(records),
        "qc_v2_diagnostic_state": _qc_v2_diagnostic_state(records),
    }


def _availability(records, predicate):
    if not records:
        return "MISSING"
    count = sum(1 for record in records if predicate(record))
    if count == len(records):
        return "AVAILABLE"
    if count > 0:
        return "PARTIAL"
    return "MISSING"


def _availability_any(records, predicate):
    if not records:
        return "MISSING"
    return "AVAILABLE" if any(predicate(record) for record in records) else "MISSING"


def _qc_v2_diagnostic_state(records):
    total = len(records)
    has_selected = lambda record: record.get("selected_start") is not None and record.get("selected_end") is not None
    has_physical = lambda record: record.get("physical_start") is not None and record.get("physical_end") is not None
    has_geometry = lambda record: all(record.get(key) is not None for key in QC_V2_FLOAT_KEYS if key != "coverage")
    has_level_point = lambda record: record.get("level_id") is not None and record.get("point_index") is not None
    full_geometry = lambda record: has_selected(record) and has_physical(record) and has_geometry(record) and has_level_point(record)

    coverage = {
        "records_total": total,
        "records_with_quality_score": sum(1 for record in records if record.get("quality_score") is not None),
        "records_with_cls_score": sum(1 for record in records if record.get("cls_score") is not None),
        "records_with_fused_score": sum(1 for record in records if record.get("fused_score") is not None),
        "records_with_selected_coordinates": sum(1 for record in records if has_selected(record)),
        "records_with_physical_coordinates": sum(1 for record in records if has_physical(record)),
        "records_with_geometry_support": sum(1 for record in records if has_geometry(record)),
        "records_with_level_point_index": sum(1 for record in records if has_level_point(record)),
        "records_with_full_qc_v2_geometry": sum(1 for record in records if full_geometry(record)),
    }
    if total == 0 or coverage["records_with_full_qc_v2_geometry"] == 0:
        status = "MISSING_QC_V2_DIAGNOSTICS"
    elif coverage["records_with_full_qc_v2_geometry"] == total:
        status = "PASS_QC_V2_DIAGNOSTICS"
    else:
        status = "PARTIAL_QC_V2_DIAGNOSTICS"

    return {
        "status": status,
        "diagnostic_only": True,
        "official_map_claim": False,
        "field_coverage": coverage,
        "interpretation": {
            "localization_geometry_check": _availability(records, full_geometry),
            "classification_calibration_check": _availability_any(
                records,
                lambda record: record.get("cls_score") is not None and record.get("quality_score") is not None,
            ),
            "ranking_geometry_check": _availability_any(
                records,
                lambda record: record.get("fused_score") is not None and has_selected(record) and has_geometry(record),
            ),
            "proposal_cap_overload_check": "AVAILABLE",
        },
    }


def _nms_coordinate(record, coordinate_space):
    if coordinate_space == "selected":
        if record.get("selected_start") is None or record.get("selected_end") is None:
            return None
        return [record["selected_start"], record["selected_end"]]
    return [record["start"], record["end"]]


def _hard_nms(records, threshold, coordinate_space="physical"):
    if threshold is None:
        return list(records)
    kept = []
    by_video_label = defaultdict(list)
    for record in sorted(records, key=lambda item: float(item.get("score", 0.0)), reverse=True):
        coord = _nms_coordinate(record, coordinate_space)
        if coord is None:
            kept.append(record)
            continue
        key = (record["video_id"], record["label"])
        overlaps = []
        for kept_record in by_video_label[key]:
            kept_coord = _nms_coordinate(kept_record, coordinate_space)
            if kept_coord is not None:
                overlaps.append(kept_coord)
        if _segment_iou(coord, overlaps) <= threshold:
            kept.append(record)
            by_video_label[key].append(record)
    return kept


def _apply_offline_filters(
    records,
    max_per_video=None,
    per_class_cap=None,
    min_score=None,
    nms_iou_threshold=None,
    nms_coordinate_space="physical",
):
    filtered = [record for record in records if min_score is None or record["score"] >= min_score]
    filtered = _hard_nms(filtered, nms_iou_threshold, coordinate_space=nms_coordinate_space)

    by_video = defaultdict(list)
    for record in filtered:
        by_video[record["video_id"]].append(record)

    capped = []
    for video_id, video_records in by_video.items():
        sorted_records = sorted(video_records, key=lambda item: float(item.get("score", 0.0)), reverse=True)
        if per_class_cap is not None:
            label_counts = defaultdict(int)
            class_capped = []
            for record in sorted_records:
                if label_counts[record["label"]] >= per_class_cap:
                    continue
                label_counts[record["label"]] += 1
                class_capped.append(record)
            sorted_records = class_capped
        if max_per_video is not None:
            sorted_records = sorted_records[:max_per_video]
        capped.extend(sorted_records)
    return _rerank_records(capped)


def _candidate_diagnostic(records, gt_by_video, total_input_predictions, source_per_video_counts, parameters):
    per_video_counts, per_video_label_counts = _count_records(records)
    summary = _analysis_summary_from_records(
        records,
        gt_by_video,
        per_video_counts,
        per_video_label_counts,
        DEFAULT_TOPK,
        SWEEP_TIOU_THRESHOLDS,
    )
    videos_seen = len(source_per_video_counts)
    reduced_videos = sum(
        1
        for video_id, input_count in source_per_video_counts.items()
        if per_video_counts.get(video_id, 0) < input_count
    )
    return {
        "status": "PASS_CANDIDATE_DIAGNOSTIC",
        "diagnostic_only": True,
        "official_map_claim": False,
        "parameters": parameters,
        "retained_predictions": len(records),
        "retained_fraction": None if total_input_predictions == 0 else len(records) / total_input_predictions,
        "videos_capped_ratio": None if videos_seen == 0 else reduced_videos / videos_seen,
        "proposal_count_per_video": summary["proposal_count_per_video"],
        "proposal_count_per_video_label": summary["proposal_count_per_video_label"],
        "score_iou_same_label_pearson": summary["score_iou_same_label_pearson"],
        "score_iou_same_label_spearman": summary["score_iou_same_label_spearman"],
        "score_iou_any_label_pearson": summary["score_iou_any_label_pearson"],
        "score_iou_any_label_spearman": summary["score_iou_any_label_spearman"],
        "rank_recall": summary["rank_recall"],
        "top_score_decile_mean_iou_same_label": _top_score_decile_mean_iou(records, "max_iou_same_label"),
        "top_score_decile_mean_iou_any_label": _top_score_decile_mean_iou(records, "max_iou_any_label"),
    }


def _build_filter_candidate(records, gt_by_video, total_input_predictions, source_per_video_counts, parameters):
    retained = _apply_offline_filters(
        records,
        max_per_video=parameters.get("max_per_video"),
        per_class_cap=parameters.get("per_class_cap"),
        min_score=parameters.get("min_score"),
        nms_iou_threshold=parameters.get("nms_iou_threshold"),
        nms_coordinate_space=parameters.get("nms_coordinate_space", "physical"),
    )
    return _candidate_diagnostic(retained, gt_by_video, total_input_predictions, source_per_video_counts, parameters)


def _quality_fusion_sweep(records, gt_by_video, source_per_video_counts):
    records_with_quality = sum(1 for record in records if record.get("quality_score") is not None)
    records_with_cls = sum(1 for record in records if record.get("cls_score") is not None)
    records_with_both = sum(
        1 for record in records if record.get("quality_score") is not None and record.get("cls_score") is not None
    )
    has_quality = records_with_quality > 0
    has_cls = records_with_cls > 0
    coverage = {
        "records_total": len(records),
        "records_with_quality_score": records_with_quality,
        "records_with_cls_score": records_with_cls,
        "records_with_both_quality_and_cls_score": records_with_both,
    }
    missing = []
    if not has_quality:
        missing.append("quality_score")
    if not has_cls:
        missing.append("cls_score")
    if missing:
        return {
            "status": "UNAVAILABLE_MISSING_QUALITY_OR_CLASS_SCORE",
            "quality_fusion_available": False,
            "missing_quality_fields": missing,
            "field_coverage": coverage,
            "alpha_sweep": [],
        }

    alpha_sweep = []
    for alpha in SWEEP_SCORE_ALPHA:
        fused_records = []
        for record in records:
            fused = dict(record)
            quality_score = fused.get("quality_score")
            cls_score = fused.get("cls_score")
            if quality_score is not None and cls_score is not None:
                fused["score"] = cls_score * max(quality_score, 0.0) ** alpha
            fused_records.append(fused)
        reranked = _rerank_records(fused_records)
        alpha_sweep.append(
            _candidate_diagnostic(
                reranked,
                gt_by_video,
                len(records),
                source_per_video_counts,
                {"score_alpha": alpha, "score_formula": "cls_score * max(quality_score, 0)^alpha"},
            )
        )
    return {
        "status": "PASS_QUALITY_FUSION_DIAGNOSTIC",
        "quality_fusion_available": True,
        "missing_quality_fields": [],
        "field_coverage": coverage,
        "alpha_sweep": alpha_sweep,
    }


def _selected_coordinate_nms_sweep(records, gt_by_video, source_per_video_counts):
    records_with_selected = sum(
        1 for record in records if record.get("selected_start") is not None and record.get("selected_end") is not None
    )
    has_selected = records_with_selected > 0
    coverage = {
        "records_total": len(records),
        "records_with_selected_coordinates": records_with_selected,
    }
    if not has_selected:
        return {
            "status": "UNAVAILABLE_MISSING_SELECTED_COORDINATES",
            "selected_coordinate_nms_available": False,
            "field_coverage": coverage,
            "nms_comparison": [],
        }
    comparison = []
    for threshold in SWEEP_NMS_THRESHOLDS:
        comparison.append(
            {
                "nms_iou_threshold": threshold,
                "physical": _build_filter_candidate(
                    records,
                    gt_by_video,
                    len(records),
                    source_per_video_counts,
                    {"nms_iou_threshold": threshold, "nms_coordinate_space": "physical"},
                ),
                "selected": _build_filter_candidate(
                    records,
                    gt_by_video,
                    len(records),
                    source_per_video_counts,
                    {"nms_iou_threshold": threshold, "nms_coordinate_space": "selected"},
                ),
            }
        )
    return {
        "status": "PASS_SELECTED_COORDINATE_NMS_DIAGNOSTIC",
        "selected_coordinate_nms_available": True,
        "field_coverage": coverage,
        "nms_comparison": comparison,
    }


def _diagnostic_sweep(prediction_path, annotation_path, subset, records, gt_by_video, per_video_counts):
    total_input_predictions = len(records)
    sweeps = {
        "max_per_video_sweep": [
            _build_filter_candidate(
                records,
                gt_by_video,
                total_input_predictions,
                per_video_counts,
                {"max_per_video": cap, "nms_coordinate_space": "physical"},
            )
            for cap in SWEEP_MAX_PER_VIDEO
        ],
        "per_class_cap_sweep": [
            _build_filter_candidate(
                records,
                gt_by_video,
                total_input_predictions,
                per_video_counts,
                {"per_class_cap": cap, "nms_coordinate_space": "physical"},
            )
            for cap in SWEEP_PER_CLASS_CAP
        ],
        "min_score_sweep": [
            _build_filter_candidate(
                records,
                gt_by_video,
                total_input_predictions,
                per_video_counts,
                {"min_score": min_score, "nms_coordinate_space": "physical"},
            )
            for min_score in SWEEP_MIN_SCORE
        ],
        "hard_nms_sweep": [
            _build_filter_candidate(
                records,
                gt_by_video,
                total_input_predictions,
                per_video_counts,
                {"nms_iou_threshold": threshold, "nms_coordinate_space": "physical"},
            )
            for threshold in SWEEP_NMS_THRESHOLDS
        ],
    }
    return {
        "status": "PASS_DIAGNOSTIC_SWEEP",
        "diagnostic_only": True,
        "official_map_claim": False,
        "prediction_path": str(prediction_path),
        "annotation_path": str(annotation_path),
        "subset": subset,
        "total_input_predictions": total_input_predictions,
        "sweep_note": "Offline proposal filtering diagnostics on result_detection only; not official mAP.",
        "sweeps": sweeps,
        "quality_fusion": _quality_fusion_sweep(records, gt_by_video, per_video_counts),
        "selected_coordinate_nms": _selected_coordinate_nms_sweep(records, gt_by_video, per_video_counts),
    }


def analyze(
    prediction_path,
    annotation_path,
    subset="validation",
    topk_values=DEFAULT_TOPK,
    thresholds=DEFAULT_TIOU_THRESHOLDS,
    include_sweep=False,
):
    predictions = _load_predictions(prediction_path)
    gt_by_video = _load_annotations(annotation_path, subset)
    records, per_video_counts, per_video_label_counts = _prediction_records(predictions, gt_by_video)

    summary = {
        "status": "PASS_DIAGNOSTIC_ANALYSIS",
        "diagnostic_only": True,
        "official_map_claim": False,
        "prediction_path": str(prediction_path),
        "annotation_path": str(annotation_path),
        "subset": subset,
    }
    summary.update(
        _analysis_summary_from_records(
            records,
            gt_by_video,
            per_video_counts,
            per_video_label_counts,
            topk_values,
            thresholds,
        )
    )
    if include_sweep:
        return summary, records, _diagnostic_sweep(prediction_path, annotation_path, subset, records, gt_by_video, per_video_counts)
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
        "quality_score",
        "cls_score",
        "fused_score",
        "selected_start",
        "selected_end",
        "physical_start",
        "physical_end",
        "selected_length",
        "physical_length",
        "proposal_width",
        "gap_mean",
        "visibility_support",
        "coverage",
        "endpoint_support",
        "level_id",
        "point_index",
        "coverage_available",
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
    parser.add_argument("--sweep-output", help="Optional diagnostic-only offline sweep JSON path")
    parser.add_argument("--records-csv", help="Optional per-proposal CSV path")
    args = parser.parse_args(argv)

    prediction_path, prediction_discovery, prediction_error = _resolve_prediction_path(args.prediction)
    if prediction_error is not None:
        _write_prediction_resolution_error(args.output, prediction_error, args.annotation)
        return 2

    if not prediction_path.is_file() or not Path(args.annotation).is_file():
        _write_missing_artifact(args.output, args.prediction, args.annotation)
        return 2

    if args.sweep_output:
        summary, records, sweep_summary = analyze(prediction_path, args.annotation, subset=args.subset, include_sweep=True)
    else:
        summary, records = analyze(prediction_path, args.annotation, subset=args.subset)
        sweep_summary = None
    if prediction_discovery is not None:
        summary["prediction_discovery"] = prediction_discovery
        if sweep_summary is not None:
            sweep_summary["prediction_discovery"] = prediction_discovery
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if args.sweep_output:
        Path(args.sweep_output).write_text(json.dumps(sweep_summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_records_csv(records, args.records_csv)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
