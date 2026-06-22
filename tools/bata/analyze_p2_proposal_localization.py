from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SCHEMA_VERSION = "native_irregular_area_head_p2_localization_attribution_v0"
READY = "NATIVE_IRREGULAR_AREA_HEAD_P2_LOCALIZATION_ATTRIBUTION_READY"
NO_GO = "NATIVE_IRREGULAR_AREA_HEAD_P2_LOCALIZATION_ATTRIBUTION_NO_GO"


def strict_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json_value(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    return str(value)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(strict_json_value(dict(payload)), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> int:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(strict_json_value(dict(row)), sort_keys=True) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path, limit_rows: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_idx}: {exc}") from None
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row at {path}:{line_idx} is not an object")
            rows.append(row)
            if limit_rows is not None and len(rows) >= int(limit_rows):
                break
    return rows


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _as_int(value: Any) -> int | None:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out


def _segment(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    start = _as_float(value[0])
    end = _as_float(value[1])
    if start is None or end is None:
        return None
    return [start, end]


def _stats(values: Sequence[float | None]) -> dict[str, Any]:
    finite = [float(item) for item in values if item is not None and math.isfinite(float(item))]
    if not finite:
        return {"count": 0, "min": None, "mean": None, "max": None}
    return {
        "count": len(finite),
        "min": min(finite),
        "mean": sum(finite) / float(len(finite)),
        "max": max(finite),
    }


def temporal_iou(segment: Sequence[float], gt_segment: Sequence[float]) -> float:
    start, end = float(segment[0]), float(segment[1])
    gt_start, gt_end = float(gt_segment[0]), float(gt_segment[1])
    inter = max(0.0, min(end, gt_end) - max(start, gt_start))
    union = max(0.0, end - start) + max(0.0, gt_end - gt_start) - inter
    return float(inter / union) if union > 0.0 else 0.0


def _overlaps(a: Sequence[float], b: Sequence[float]) -> bool:
    return min(float(a[1]), float(b[1])) > max(float(a[0]), float(b[0]))


def parse_number_list(value: str, *, as_int: bool = False) -> list[Any]:
    out = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item) if as_int else float(item))
    if not out:
        raise argparse.ArgumentTypeError("list argument must contain at least one value")
    return out


def resolve_maybe_relative(path: str | Path | None, base: str | Path | None = None) -> Path | None:
    if path is None:
        return None
    out = Path(path).expanduser()
    if out.is_absolute():
        return out
    repo_relative = ROOT / out
    if repo_relative.exists() or str(out).replace("\\", "/").startswith("data/"):
        return repo_relative
    if base is not None:
        return Path(base).expanduser().resolve().parent / out
    return repo_relative


def infer_dataset_paths_from_config(config: str | Path | None, split: str) -> dict[str, Path | None]:
    if config is None:
        return {"annotation": None, "class_map": None}
    try:
        from mmengine.config import Config
    except Exception as exc:
        raise RuntimeError(f"mmengine Config import failed while reading config paths: {exc}") from None
    cfg_path = Path(config).expanduser().resolve()
    cfg = Config.fromfile(str(cfg_path))
    if not hasattr(cfg, "dataset") or split not in cfg.dataset:
        raise ValueError(f"config missing dataset.{split}")
    dataset = cfg.dataset[split]
    annotation = dataset.get("ann_file", None)
    class_map = dataset.get("class_map", None)
    return {
        "annotation": resolve_maybe_relative(annotation, cfg_path),
        "class_map": resolve_maybe_relative(class_map, cfg_path),
    }


def load_class_map(path: str | Path | None) -> tuple[dict[str, int], list[str]]:
    if path is None:
        return {}, []
    class_path = Path(path).expanduser()
    label_to_id: dict[str, int] = {}
    id_to_label: list[str] = []
    with class_path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            label: str
            idx: int
            if len(parts) >= 2 and parts[0].lstrip("-").isdigit():
                idx = int(parts[0])
                label = " ".join(parts[1:])
            elif len(parts) >= 2 and parts[-1].lstrip("-").isdigit():
                idx = int(parts[-1])
                label = " ".join(parts[:-1])
            else:
                idx = len(id_to_label)
                label = stripped
            while len(id_to_label) <= idx:
                id_to_label.append("")
            id_to_label[idx] = label
            label_to_id[label] = idx
    return label_to_id, id_to_label


def load_annotations(
    path: str | Path,
    *,
    split: str = "validation",
    class_map: Mapping[str, int] | None = None,
    include_ambiguous: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    database = payload.get("database", payload)
    if not isinstance(database, Mapping):
        raise ValueError("annotation file must contain a database object or be a database object")
    out: dict[str, list[dict[str, Any]]] = {}
    for video_id, item in database.items():
        if not isinstance(item, Mapping):
            continue
        subset = str(item.get("subset", ""))
        if split and subset and subset.lower() != split.lower():
            continue
        duration = _as_float(item.get("duration"))
        annotations = item.get("annotations", [])
        if not isinstance(annotations, Sequence):
            continue
        rows = []
        for ann_idx, ann in enumerate(annotations):
            if not isinstance(ann, Mapping):
                continue
            label = str(ann.get("label", ""))
            if not include_ambiguous and label.lower() == "ambiguous":
                continue
            segment = _segment(ann.get("segment"))
            if segment is None or segment[1] <= segment[0]:
                continue
            class_id = class_map.get(label) if class_map is not None else None
            rows.append(
                {
                    "video_id": str(video_id),
                    "gt_index": int(ann_idx),
                    "label": label,
                    "class_id": class_id,
                    "segment_seconds": segment,
                    "duration_seconds": duration,
                }
            )
        out[str(video_id)] = rows
    return out


def proposal_segment_seconds(row: Mapping[str, Any]) -> list[float] | None:
    direct = _segment(row.get("segment_seconds"))
    if direct is not None:
        return direct
    start_sec = _as_float(row.get("proposal_start_seconds"))
    end_sec = _as_float(row.get("proposal_end_seconds"))
    if start_sec is not None and end_sec is not None:
        return [start_sec, end_sec]
    dense = _segment(row.get("segment"))
    if dense is None:
        return None
    fps = _as_float(row.get("fps"))
    snippet_stride = _as_float(row.get("snippet_stride"))
    window_start_frame = _as_float(row.get("window_start_frame"))
    offset_frames = _as_float(row.get("offset_frames")) or 0.0
    if fps is None or snippet_stride is None or window_start_frame is None or fps <= 0.0:
        return None
    return [
        (window_start_frame + offset_frames + dense[0] * snippet_stride) / fps,
        (window_start_frame + offset_frames + dense[1] * snippet_stride) / fps,
    ]


def proposal_segment_frames(row: Mapping[str, Any]) -> list[float] | None:
    direct = _segment(row.get("segment_frames"))
    if direct is not None:
        return direct
    dense = _segment(row.get("segment"))
    if dense is None:
        return None
    snippet_stride = _as_float(row.get("snippet_stride"))
    window_start_frame = _as_float(row.get("window_start_frame"))
    offset_frames = _as_float(row.get("offset_frames")) or 0.0
    if snippet_stride is None or window_start_frame is None:
        return None
    return [
        window_start_frame + offset_frames + dense[0] * snippet_stride,
        window_start_frame + offset_frames + dense[1] * snippet_stride,
    ]


def row_label_matches_gt(row: Mapping[str, Any], gt: Mapping[str, Any]) -> bool:
    row_label = row.get("label")
    gt_label = gt.get("label")
    if row_label is not None and gt_label is not None and str(row_label) == str(gt_label):
        return True
    row_class = _as_int(row.get("class_id"))
    gt_class = _as_int(gt.get("class_id"))
    return row_class is not None and gt_class is not None and row_class == gt_class


def best_gt_match(
    segment: Sequence[float] | None,
    gt_rows: Sequence[Mapping[str, Any]],
    *,
    row: Mapping[str, Any] | None = None,
    class_aware: bool = False,
) -> dict[str, Any]:
    if segment is None:
        return {"max_iou": None, "gt_index": None, "gt_label": None, "gt_segment_seconds": None}
    best_iou = 0.0
    best: Mapping[str, Any] | None = None
    for gt in gt_rows:
        if class_aware and row is not None and not row_label_matches_gt(row, gt):
            continue
        gt_segment = gt.get("segment_seconds")
        if not isinstance(gt_segment, Sequence) or len(gt_segment) < 2:
            continue
        iou = temporal_iou(segment, gt_segment)
        if iou > best_iou:
            best_iou = iou
            best = gt
    if best is None:
        return {"max_iou": 0.0, "gt_index": None, "gt_label": None, "gt_segment_seconds": None}
    gt_segment = list(best["segment_seconds"])
    return {
        "max_iou": float(best_iou),
        "gt_index": best.get("gt_index"),
        "gt_label": best.get("label"),
        "gt_segment_seconds": gt_segment,
        "center_error_seconds": ((segment[0] + segment[1]) * 0.5) - ((gt_segment[0] + gt_segment[1]) * 0.5),
        "start_error_seconds": segment[0] - gt_segment[0],
        "end_error_seconds": segment[1] - gt_segment[1],
        "duration_error_seconds": (segment[1] - segment[0]) - (gt_segment[1] - gt_segment[0]),
    }


def _sample_key(row: Mapping[str, Any], group_by: str) -> str:
    if group_by == "video":
        return str(row.get("video_id", "unknown"))
    sample_id = row.get("sample_id")
    if sample_id is not None:
        return str(sample_id)
    video_id = str(row.get("video_id", "unknown"))
    window_start = row.get("window_start_frame")
    if window_start is not None:
        return f"{video_id}|window_start_frame={window_start}"
    return video_id


def assign_score_ranks(rows: list[dict[str, Any]], *, group_by: str) -> None:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_sample_key(row, group_by), []).append(row)
    for key, group_rows in groups.items():
        ordered = sorted(
            group_rows,
            key=lambda item: _as_float(item.get("final_score")) if _as_float(item.get("final_score")) is not None else -1.0,
            reverse=True,
        )
        denom = max(1, len(ordered) - 1)
        for rank, row in enumerate(ordered, start=1):
            row["diagnostic_group_key"] = key
            row["diagnostic_score_rank_in_group"] = rank
            row["diagnostic_score_percentile_in_group"] = 1.0 - float(rank - 1) / float(denom)


def join_proposals_with_gt(
    proposal_rows: Sequence[Mapping[str, Any]],
    gt_by_video: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    group_by: str = "sample",
    require_seconds: bool = True,
) -> list[dict[str, Any]]:
    joined: list[dict[str, Any]] = []
    missing_seconds = 0
    for row in proposal_rows:
        out = dict(row)
        video_id = str(out.get("video_id", "unknown"))
        segment_seconds = proposal_segment_seconds(out)
        segment_frames = proposal_segment_frames(out)
        if segment_seconds is None:
            missing_seconds += 1
            if require_seconds:
                raise ValueError(
                    "proposal row has no segment_seconds and lacks fps/snippet_stride/window_start_frame metadata"
                )
        out["diagnostic_segment_seconds"] = segment_seconds
        out["diagnostic_segment_frames"] = segment_frames
        out["diagnostic_group_key"] = _sample_key(out, group_by)
        gt_rows = gt_by_video.get(video_id, [])
        best_any = best_gt_match(segment_seconds, gt_rows, row=out, class_aware=False)
        best_cls = best_gt_match(segment_seconds, gt_rows, row=out, class_aware=True)
        out["uses_validation_gt_for_diagnostic_join"] = segment_seconds is not None
        out["diagnostic_max_gt_iou_seconds"] = best_any["max_iou"]
        out["diagnostic_best_gt_index"] = best_any["gt_index"]
        out["diagnostic_best_gt_label"] = best_any["gt_label"]
        out["diagnostic_best_gt_segment_seconds"] = best_any["gt_segment_seconds"]
        out["diagnostic_class_aware_max_gt_iou_seconds"] = best_cls["max_iou"]
        out["diagnostic_class_aware_best_gt_index"] = best_cls["gt_index"]
        out["diagnostic_class_aware_best_gt_label"] = best_cls["gt_label"]
        out["diagnostic_center_error_seconds"] = best_any.get("center_error_seconds")
        out["diagnostic_start_error_seconds"] = best_any.get("start_error_seconds")
        out["diagnostic_end_error_seconds"] = best_any.get("end_error_seconds")
        out["diagnostic_duration_error_seconds"] = best_any.get("duration_error_seconds")
        joined.append(out)
    assign_score_ranks(joined, group_by=group_by)
    if missing_seconds:
        for row in joined:
            row["diagnostic_missing_seconds_rows"] = int(missing_seconds)
    return joined


def _gt_for_group(rows: Sequence[Mapping[str, Any]], gt_by_video: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[Mapping[str, Any]]:
    if not rows:
        return []
    video_id = str(rows[0].get("video_id", "unknown"))
    all_gt = list(gt_by_video.get(video_id, []))
    window = _segment([rows[0].get("window_start_seconds"), rows[0].get("window_end_seconds")])
    if window is None:
        return all_gt
    return [gt for gt in all_gt if _overlaps(window, gt["segment_seconds"])]


def compute_topk_iou_rank_summary(
    joined_rows: Sequence[Mapping[str, Any]],
    gt_by_video: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    topk: Sequence[int],
    iou_thresholds: Sequence[float],
    group_by: str = "sample",
) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in joined_rows:
        groups.setdefault(_sample_key(row, group_by), []).append(row)
    per_group = []
    aggregate: dict[str, dict[str, float]] = {}
    for group_key, rows in sorted(groups.items()):
        ordered = sorted(rows, key=lambda item: int(item.get("diagnostic_score_rank_in_group", 10**9)))
        gt_rows = _gt_for_group(ordered, gt_by_video)
        group_out: dict[str, Any] = {
            "group_key": group_key,
            "video_id": ordered[0].get("video_id") if ordered else None,
            "proposal_count": len(ordered),
            "gt_count_in_window": len(gt_rows),
        }
        for k in topk:
            k_int = int(k)
            subset = ordered[:k_int]
            oracle_subset = sorted(
                ordered,
                key=lambda item: _as_float(item.get("diagnostic_max_gt_iou_seconds")) or 0.0,
                reverse=True,
            )[:k_int]
            for thr in iou_thresholds:
                thr_f = float(thr)
                prefix = f"top{k_int}_iou{thr_f:.2f}".replace(".", "p")
                positives = [
                    row
                    for row in subset
                    if (_as_float(row.get("diagnostic_max_gt_iou_seconds")) or 0.0) >= thr_f
                ]
                class_positives = [
                    row
                    for row in subset
                    if (_as_float(row.get("diagnostic_class_aware_max_gt_iou_seconds")) or 0.0) >= thr_f
                ]
                proposal_count = max(1, len(subset))
                matched_gt = 0
                class_matched_gt = 0
                oracle_matched_gt = 0
                for gt in gt_rows:
                    best_any = 0.0
                    best_cls = 0.0
                    best_oracle = 0.0
                    for row in subset:
                        seg = row.get("diagnostic_segment_seconds")
                        if isinstance(seg, Sequence) and len(seg) >= 2:
                            iou = temporal_iou(seg, gt["segment_seconds"])
                            best_any = max(best_any, iou)
                            if row_label_matches_gt(row, gt):
                                best_cls = max(best_cls, iou)
                    for row in oracle_subset:
                        seg = row.get("diagnostic_segment_seconds")
                        if isinstance(seg, Sequence) and len(seg) >= 2:
                            best_oracle = max(best_oracle, temporal_iou(seg, gt["segment_seconds"]))
                    matched_gt += int(best_any >= thr_f)
                    class_matched_gt += int(best_cls >= thr_f)
                    oracle_matched_gt += int(best_oracle >= thr_f)
                gt_count = max(1, len(gt_rows))
                group_out[f"{prefix}_proposal_positive_rate"] = len(positives) / float(proposal_count)
                group_out[f"{prefix}_class_aware_proposal_positive_rate"] = len(class_positives) / float(proposal_count)
                group_out[f"{prefix}_gt_recall"] = matched_gt / float(gt_count) if gt_rows else None
                group_out[f"{prefix}_class_aware_gt_recall"] = class_matched_gt / float(gt_count) if gt_rows else None
                group_out[f"{prefix}_oracle_iou_rank_gt_recall"] = oracle_matched_gt / float(gt_count) if gt_rows else None
        per_group.append(group_out)

    numeric_keys = sorted({key for item in per_group for key, value in item.items() if isinstance(value, (int, float))})
    for key in numeric_keys:
        values = [float(item[key]) for item in per_group if isinstance(item.get(key), (int, float))]
        if values:
            aggregate[key] = _stats(values)
    return {
        "schema_version": SCHEMA_VERSION,
        "diagnostic": "top_k_iou_rank_dump",
        "group_by": group_by,
        "groups": per_group,
        "aggregate": aggregate,
        "diagnostic_only": True,
        "uses_validation_gt": True,
        "metric_claim_allowed": False,
        "paper_claim_allowed": False,
    }


def compute_score_iou_reliability(
    joined_rows: Sequence[Mapping[str, Any]],
    *,
    score_bins: int,
    iou_thresholds: Sequence[float],
) -> dict[str, Any]:
    rows = [row for row in joined_rows if _as_float(row.get("final_score")) is not None]
    rows.sort(key=lambda item: _as_float(item.get("final_score")) or 0.0)
    bins = []
    n = len(rows)
    bin_count = max(1, int(score_bins))
    component_keys = ("start_score", "end_score", "area_integral", "observed_fraction", "uncertainty_penalty", "duration")
    for bin_idx in range(bin_count):
        start = int(round(bin_idx * n / bin_count))
        end = int(round((bin_idx + 1) * n / bin_count))
        subset = rows[start:end]
        if not subset:
            continue
        scores = [_as_float(row.get("final_score")) for row in subset]
        ious = [_as_float(row.get("diagnostic_max_gt_iou_seconds")) for row in subset]
        class_ious = [_as_float(row.get("diagnostic_class_aware_max_gt_iou_seconds")) for row in subset]
        item: dict[str, Any] = {
            "bin_index": bin_idx,
            "count": len(subset),
            "score": _stats(scores),
            "max_gt_iou": _stats(ious),
            "class_aware_max_gt_iou": _stats(class_ious),
        }
        for key in component_keys:
            item[key] = _stats([_as_float(row.get(key)) for row in subset])
        for thr in iou_thresholds:
            thr_f = float(thr)
            suffix = f"iou{thr_f:.2f}".replace(".", "p")
            item[f"{suffix}_positive_fraction"] = sum((iou or 0.0) >= thr_f for iou in ious) / float(len(subset))
            item[f"{suffix}_class_aware_positive_fraction"] = sum(
                (iou or 0.0) >= thr_f for iou in class_ious
            ) / float(len(subset))
        bins.append(item)
    return {
        "schema_version": SCHEMA_VERSION,
        "diagnostic": "score_vs_iou_reliability",
        "score_bins": int(score_bins),
        "bins": bins,
        "diagnostic_only": True,
        "uses_validation_gt": True,
        "metric_claim_allowed": False,
        "paper_claim_allowed": False,
    }


def compute_dense_axis_coordinate_audit(joined_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = {
        "rows": 0,
        "has_seconds": 0,
        "has_frames": 0,
        "nonpositive_dense_duration": 0,
        "nonpositive_seconds_duration": 0,
        "outside_video_seconds": 0,
        "outside_window_seconds": 0,
        "missing_window_bounds": 0,
        "seconds_conversion_residual_gt_1e_4": 0,
    }
    dense_durations = []
    seconds_durations = []
    conversion_residuals = []
    outliers = []
    for row_idx, row in enumerate(joined_rows):
        counts["rows"] += 1
        dense = _segment(row.get("segment"))
        seconds = proposal_segment_seconds(row)
        frames = proposal_segment_frames(row)
        if frames is not None:
            counts["has_frames"] += 1
        if seconds is not None:
            counts["has_seconds"] += 1
            seconds_duration = seconds[1] - seconds[0]
            seconds_durations.append(seconds_duration)
            if seconds_duration <= 0:
                counts["nonpositive_seconds_duration"] += 1
                outliers.append({"row_index": row_idx, "reason": "nonpositive_seconds_duration", "segment_seconds": seconds})
        if dense is not None:
            dense_duration = dense[1] - dense[0]
            dense_durations.append(dense_duration)
            if dense_duration <= 0:
                counts["nonpositive_dense_duration"] += 1
                outliers.append({"row_index": row_idx, "reason": "nonpositive_dense_duration", "segment": dense})
        video_duration = _as_float(row.get("duration_seconds"))
        if video_duration is not None and seconds is not None:
            if seconds[0] < -1e-4 or seconds[1] > video_duration + 1e-4:
                counts["outside_video_seconds"] += 1
                outliers.append(
                    {
                        "row_index": row_idx,
                        "reason": "outside_video_seconds",
                        "segment_seconds": seconds,
                        "duration_seconds": video_duration,
                    }
                )
        window = _segment([row.get("window_start_seconds"), row.get("window_end_seconds")])
        if seconds is not None:
            if window is None:
                counts["missing_window_bounds"] += 1
            elif seconds[0] < window[0] - 1e-4 or seconds[1] > window[1] + 1e-4:
                counts["outside_window_seconds"] += 1
                outliers.append(
                    {
                        "row_index": row_idx,
                        "reason": "outside_window_seconds",
                        "segment_seconds": seconds,
                        "window_seconds": window,
                    }
                )
        row_seconds = _segment(row.get("segment_seconds"))
        recomputed = proposal_segment_seconds({key: value for key, value in row.items() if key != "segment_seconds"})
        if row_seconds is not None and recomputed is not None:
            residual = max(abs(row_seconds[0] - recomputed[0]), abs(row_seconds[1] - recomputed[1]))
            conversion_residuals.append(residual)
            if residual > 1e-4:
                counts["seconds_conversion_residual_gt_1e_4"] += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "diagnostic": "dense_axis_coordinate_audit",
        "counts": counts,
        "dense_duration": _stats(dense_durations),
        "seconds_duration": _stats(seconds_durations),
        "seconds_conversion_residual": _stats(conversion_residuals),
        "outliers_head": outliers[:100],
        "diagnostic_only": True,
        "uses_validation_gt": False,
        "metric_claim_allowed": False,
        "paper_claim_allowed": False,
    }


def run_localization_attribution(
    *,
    proposal_jsonl: str | Path,
    output_dir: str | Path,
    annotation: str | Path | None = None,
    config: str | Path | None = None,
    class_map: str | Path | None = None,
    split: str = "validation",
    dataset_split_key: str = "val",
    topk: Sequence[int] = (100, 300, 1000),
    iou_thresholds: Sequence[float] = (0.3, 0.5, 0.7),
    score_bins: int = 10,
    group_by: str = "sample",
    require_seconds: bool = True,
    include_ambiguous: bool = False,
    limit_rows: int | None = None,
) -> dict[str, Any]:
    inferred = infer_dataset_paths_from_config(config, dataset_split_key) if config is not None else {}
    ann_path = resolve_maybe_relative(annotation, config) if annotation is not None else inferred.get("annotation")
    class_map_path = resolve_maybe_relative(class_map, config) if class_map is not None else inferred.get("class_map")
    if ann_path is None:
        raise ValueError("annotation path is required; pass --annotation or --config")
    if not ann_path.exists():
        raise FileNotFoundError(f"annotation not found: {ann_path}")
    label_to_id, id_to_label = load_class_map(class_map_path) if class_map_path is not None and class_map_path.exists() else ({}, [])
    gt_by_video = load_annotations(
        ann_path,
        split=split,
        class_map=label_to_id,
        include_ambiguous=include_ambiguous,
    )
    proposals = read_jsonl(proposal_jsonl, limit_rows=limit_rows)
    joined = join_proposals_with_gt(proposals, gt_by_video, group_by=group_by, require_seconds=require_seconds)
    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    joined_path = out_dir / "joined_proposals.jsonl"
    topk_path = out_dir / "topk_iou_rank_summary.json"
    reliability_path = out_dir / "score_iou_reliability.json"
    audit_path = out_dir / "dense_axis_coordinate_audit.json"
    summary_path = out_dir / "summary.json"
    written = write_jsonl(joined_path, joined)
    topk_summary = compute_topk_iou_rank_summary(
        joined,
        gt_by_video,
        topk=topk,
        iou_thresholds=iou_thresholds,
        group_by=group_by,
    )
    reliability = compute_score_iou_reliability(joined, score_bins=score_bins, iou_thresholds=iou_thresholds)
    coordinate_audit = compute_dense_axis_coordinate_audit(joined)
    write_json(topk_path, topk_summary)
    write_json(reliability_path, reliability)
    write_json(audit_path, coordinate_audit)
    ious = [_as_float(row.get("diagnostic_max_gt_iou_seconds")) for row in joined]
    class_ious = [_as_float(row.get("diagnostic_class_aware_max_gt_iou_seconds")) for row in joined]
    scores = [_as_float(row.get("final_score")) for row in joined]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "decision": READY,
        "proposal_jsonl": str(Path(proposal_jsonl).expanduser()),
        "annotation": str(ann_path),
        "class_map": str(class_map_path) if class_map_path is not None else None,
        "config": str(config) if config is not None else None,
        "split": split,
        "dataset_split_key": dataset_split_key,
        "group_by": group_by,
        "proposal_rows": len(proposals),
        "joined_rows": written,
        "videos_with_gt": len(gt_by_video),
        "class_count": len(id_to_label),
        "topk": [int(item) for item in topk],
        "iou_thresholds": [float(item) for item in iou_thresholds],
        "score": _stats(scores),
        "max_gt_iou_seconds": _stats(ious),
        "class_aware_max_gt_iou_seconds": _stats(class_ious),
        "joined_proposals_jsonl": str(joined_path),
        "topk_iou_rank_summary_json": str(topk_path),
        "score_iou_reliability_json": str(reliability_path),
        "dense_axis_coordinate_audit_json": str(audit_path),
        "diagnostic_only": True,
        "uses_validation_gt": True,
        "uses_teacher": False,
        "uses_oracle": False,
        "uses_cache": False,
        "uses_raw_prediction": False,
        "metric_claim_allowed": False,
        "paper_claim_allowed": False,
    }
    write_json(summary_path, summary)
    return summary


def error_payload(exc: BaseException) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": NO_GO,
        "error_type": exc.__class__.__name__,
        "error": str(exc),
        "diagnostic_only": True,
        "metric_claim_allowed": False,
        "paper_claim_allowed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Join P2 proposal factors to val GT and audit localization attribution.")
    parser.add_argument("--proposal-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--annotation")
    parser.add_argument("--config")
    parser.add_argument("--class-map")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--dataset-split-key", default="val")
    parser.add_argument("--topk", default="100,300,1000")
    parser.add_argument("--iou-thresholds", default="0.3,0.5,0.7")
    parser.add_argument("--score-bins", type=int, default=10)
    parser.add_argument("--group-by", choices=("sample", "video"), default="sample")
    parser.add_argument("--allow-missing-seconds", action="store_true")
    parser.add_argument("--include-ambiguous", action="store_true")
    parser.add_argument("--limit-rows", type=int)
    args = parser.parse_args(argv)

    try:
        summary = run_localization_attribution(
            proposal_jsonl=args.proposal_jsonl,
            output_dir=args.output_dir,
            annotation=args.annotation,
            config=args.config,
            class_map=args.class_map,
            split=args.split,
            dataset_split_key=args.dataset_split_key,
            topk=parse_number_list(args.topk, as_int=True),
            iou_thresholds=parse_number_list(args.iou_thresholds),
            score_bins=args.score_bins,
            group_by=args.group_by,
            require_seconds=not bool(args.allow_missing_seconds),
            include_ambiguous=bool(args.include_ambiguous),
            limit_rows=args.limit_rows,
        )
    except Exception as exc:  # pragma: no cover - CLI guard
        print(json.dumps(strict_json_value(error_payload(exc)), sort_keys=True))
        return 1
    print(json.dumps(strict_json_value(summary), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
