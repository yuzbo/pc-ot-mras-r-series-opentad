from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "actionformer_post_nms_overload_audit_v0"
READY = "ACTIONFORMER_POST_NMS_OVERLOAD_AUDIT_READY"
NO_GO = "ACTIONFORMER_POST_NMS_OVERLOAD_AUDIT_NO_GO"


_GT_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?"
    r"Number of ground truth instances:\s*(?P<value>\d+)"
)
_PRED_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?"
    r"Number of predictions:\s*(?P<value>\d+)"
)
_AVG_MAP_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?"
    r"Average-mAP:\s*(?P<value>[-+]?\d+(?:\.\d+)?)\s*\(%\)"
)


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_validation_annotation_stats(path: str | Path, *, split: str = "validation") -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    database = payload.get("database", payload)
    if not isinstance(database, Mapping):
        raise ValueError("annotation file must contain a database object or be a database object")

    video_count = 0
    gt_count = 0
    duration_seconds: list[float] = []
    for _video_id, item in database.items():
        if not isinstance(item, Mapping):
            continue
        subset = str(item.get("subset", ""))
        if split and subset and subset.lower() != split.lower():
            continue
        video_count += 1
        anns = item.get("annotations", [])
        if isinstance(anns, Sequence) and not isinstance(anns, (str, bytes)):
            gt_count += len(anns)
        duration = _as_float(item.get("duration"))
        if duration is not None:
            duration_seconds.append(duration)

    return {
        "split": split,
        "video_count": video_count,
        "annotation_gt_count_raw": gt_count,
        "duration_seconds_min": min(duration_seconds) if duration_seconds else None,
        "duration_seconds_max": max(duration_seconds) if duration_seconds else None,
        "duration_seconds_mean": statistics.fmean(duration_seconds) if duration_seconds else None,
    }


def parse_log_metrics(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current: dict[str, Any] = {}

    for raw_line in Path(path).expanduser().read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.replace("\r", "")
        gt_match = _GT_RE.search(line)
        if gt_match:
            if current:
                rows.append(current)
            current = {
                "timestamp": gt_match.group("timestamp"),
                "ground_truth_instances": int(gt_match.group("value")),
            }
            continue

        pred_match = _PRED_RE.search(line)
        if pred_match:
            if not current:
                current = {"timestamp": pred_match.group("timestamp")}
            current["timestamp"] = current.get("timestamp") or pred_match.group("timestamp")
            current["predictions"] = int(pred_match.group("value"))
            continue

        avg_map_match = _AVG_MAP_RE.search(line)
        if avg_map_match:
            if not current:
                current = {"timestamp": avg_map_match.group("timestamp")}
            current["timestamp"] = current.get("timestamp") or avg_map_match.group("timestamp")
            current["average_map_percent"] = float(avg_map_match.group("value"))
            rows.append(current)
            current = {}

    if current:
        rows.append(current)
    return [row for row in rows if "predictions" in row or "average_map_percent" in row]


def _parse_base_config_paths(path: Path, text: str) -> list[Path]:
    match = re.search(r"_base_\s*=\s*(?P<value>\[[\s\S]*?\]|[\"'][^\"']+[\"'])", text)
    if not match:
        return []
    value = match.group("value")
    paths = re.findall(r"[\"']([^\"']+)[\"']", value)
    return [(path.parent / item).resolve() for item in paths]


def parse_max_seg_num_from_config(path: str | Path | None, *, _seen: set[Path] | None = None) -> int | None:
    if path is None:
        return None
    cfg_path = Path(path).expanduser().resolve()
    seen = _seen or set()
    if cfg_path in seen:
        return None
    seen.add(cfg_path)

    text = cfg_path.read_text(encoding="utf-8", errors="ignore")
    matches: list[int] = []
    for base_path in _parse_base_config_paths(cfg_path, text):
        base_value = parse_max_seg_num_from_config(base_path, _seen=seen) if base_path.exists() else None
        if base_value is not None:
            matches.append(int(base_value))
    matches.extend(int(match.group(1)) for match in re.finditer(r"\bmax_seg_num\s*=\s*(\d+)", text))
    if matches:
        return matches[-1]
    return None


def parse_reference_counts(items: Sequence[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"reference count must use name=count format: {item}")
        name, value = item.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"reference count name is empty: {item}")
        out[name] = int(value.strip())
    return out


def load_result_detection_counts(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    results = payload.get("results", payload) if isinstance(payload, Mapping) else None
    if not isinstance(results, Mapping):
        raise ValueError("result_detection JSON must contain a results object or be a video-to-results object")

    counts = {str(video_id): len(items or []) for video_id, items in results.items()}
    values = list(counts.values())
    if not values:
        return {
            "result_detection_json": str(Path(path).expanduser()),
            "video_count": 0,
            "total_predictions": 0,
            "per_video_min": 0,
            "per_video_max": 0,
            "per_video_mean": 0,
            "per_video_count_histogram": {},
        }
    return {
        "result_detection_json": str(Path(path).expanduser()),
        "video_count": len(values),
        "total_predictions": sum(values),
        "per_video_min": min(values),
        "per_video_max": max(values),
        "per_video_mean": statistics.fmean(values),
        "per_video_median": statistics.median(values),
        "per_video_count_histogram": {str(k): v for k, v in sorted(Counter(values).items())},
    }


def summarize_overload(
    *,
    log_metrics: Sequence[Mapping[str, Any]],
    annotation_stats: Mapping[str, Any],
    max_seg_num: int | None,
    reference_counts: Mapping[str, int],
    result_detection_counts: Mapping[str, Any] | None,
) -> dict[str, Any]:
    prediction_values = [int(row["predictions"]) for row in log_metrics if "predictions" in row]
    avg_map_values = [float(row["average_map_percent"]) for row in log_metrics if "average_map_percent" in row]
    video_count = int(annotation_stats.get("video_count", 0) or 0)
    expected_cap = video_count * max_seg_num if max_seg_num is not None else None
    saturation_flags = [
        expected_cap is not None and int(value) == int(expected_cap)
        for value in prediction_values
    ]

    reference_ratios = {}
    latest_predictions = prediction_values[-1] if prediction_values else None
    for name, count in reference_counts.items():
        reference_ratios[name] = (latest_predictions / count) if latest_predictions is not None and count else None

    result_file_matches_log = None
    if result_detection_counts is not None and latest_predictions is not None:
        result_file_matches_log = int(result_detection_counts["total_predictions"]) == int(latest_predictions)

    return {
        "log_eval_count": len(log_metrics),
        "prediction_values": prediction_values,
        "prediction_unique_values": sorted(set(prediction_values)),
        "latest_predictions": latest_predictions,
        "average_map_percent_values": avg_map_values,
        "latest_average_map_percent": avg_map_values[-1] if avg_map_values else None,
        "annotation_validation_video_count": video_count,
        "annotation_gt_count_raw": annotation_stats.get("annotation_gt_count_raw"),
        "post_processing_max_seg_num": max_seg_num,
        "expected_dataset_prediction_cap": expected_cap,
        "all_logged_prediction_counts_equal_cap": bool(saturation_flags) and all(saturation_flags),
        "cap_saturated_eval_count": sum(1 for flag in saturation_flags if flag),
        "cap_saturation_ratio": (sum(1 for flag in saturation_flags if flag) / len(saturation_flags))
        if saturation_flags
        else None,
        "latest_predictions_per_video": (latest_predictions / video_count) if latest_predictions and video_count else None,
        "reference_prediction_count_ratios": reference_ratios,
        "result_detection_counts": result_detection_counts,
        "result_detection_total_matches_latest_log": result_file_matches_log,
        "interpretation": (
            "logged prediction count exactly equals validation_video_count * post_processing.max_seg_num"
            if saturation_flags and all(saturation_flags)
            else "logged prediction count is not fully explained by the dataset-level post-NMS cap"
        ),
    }


def run_post_nms_overload_audit(
    *,
    train_log: str | Path,
    annotation: str | Path,
    output_dir: str | Path,
    config: str | Path | None = None,
    result_detection_json: str | Path | None = None,
    split: str = "validation",
    references: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    log_metrics = parse_log_metrics(train_log)
    annotation_stats = load_validation_annotation_stats(annotation, split=split)
    max_seg_num = parse_max_seg_num_from_config(config)
    result_counts = load_result_detection_counts(result_detection_json)
    summary = summarize_overload(
        log_metrics=log_metrics,
        annotation_stats=annotation_stats,
        max_seg_num=max_seg_num,
        reference_counts=references or {},
        result_detection_counts=result_counts,
    )
    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "decision": READY,
        "diagnostic": "actionformer_post_nms_overload_audit",
        "train_log": str(Path(train_log).expanduser()),
        "config": str(Path(config).expanduser()) if config is not None else None,
        "annotation": str(Path(annotation).expanduser()),
        "split": split,
        "summary": summary,
        "log_metrics": list(log_metrics),
        "protocol_flags": {
            "diagnostic_only": True,
            "no_model_forward": True,
            "no_training": True,
            "no_checkpoint_read": True,
            "no_tools_test": True,
            "uses_existing_train_log": True,
            "uses_existing_result_detection_predictions": result_detection_json is not None,
            "uses_validation_gt": True,
            "uses_validation_gt_for_count_normalization_only": True,
            "uses_teacher": False,
            "uses_oracle": False,
            "uses_raw_prediction": False,
            "metric_claim_allowed": False,
            "paper_claim_allowed": False,
            "runtime_or_deployment_claim_allowed": False,
        },
    }
    write_json(out_dir / "summary.json", payload)
    write_json(out_dir / "log_metrics.json", {"rows": list(log_metrics)})
    return payload


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
    parser = argparse.ArgumentParser(
        description="Read-only audit for ActionFormer post-NMS proposal-count overload from logs/results."
    )
    parser.add_argument("--train-log", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config")
    parser.add_argument("--result-detection-json")
    parser.add_argument("--split", default="validation")
    parser.add_argument(
        "--reference-count",
        action="append",
        default=[],
        help="Reference prediction count in name=count format; can be repeated.",
    )
    args = parser.parse_args(argv)

    try:
        summary = run_post_nms_overload_audit(
            train_log=args.train_log,
            annotation=args.annotation,
            output_dir=args.output_dir,
            config=args.config,
            result_detection_json=args.result_detection_json,
            split=args.split,
            references=parse_reference_counts(args.reference_count),
        )
    except Exception as exc:  # pragma: no cover - CLI guard
        print(json.dumps(error_payload(exc), sort_keys=True))
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
