from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from tools.bata.analyze_p2_proposal_localization import (  # noqa: E402
    infer_dataset_paths_from_config,
    load_class_map,
    parse_number_list,
    resolve_maybe_relative,
    run_localization_attribution,
    strict_json_value,
    write_json,
    write_jsonl,
)


SCHEMA_VERSION = "actionformer_result_detection_geometry_audit_v0"
READY = "ACTIONFORMER_RESULT_DETECTION_GEOMETRY_AUDIT_READY"
NO_GO = "ACTIONFORMER_RESULT_DETECTION_GEOMETRY_AUDIT_NO_GO"


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _segment_seconds(value: Any, *, context: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise ValueError(f"{context} segment must be a two-value list")
    start = _as_float(value[0])
    end = _as_float(value[1])
    if start is None or end is None:
        raise ValueError(f"{context} segment contains non-finite values")
    if end <= start:
        raise ValueError(f"{context} segment must have positive duration")
    return [float(start), float(end)]


def _score(value: Any, *, context: str) -> float:
    score = _as_float(value)
    if score is None:
        raise ValueError(f"{context} score must be finite")
    return float(score)


def load_video_durations(path: str | Path, *, split: str = "validation") -> dict[str, float]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    database = payload.get("database", payload)
    if not isinstance(database, Mapping):
        raise ValueError("annotation file must contain a database object or be a database object")
    out: dict[str, float] = {}
    for video_id, item in database.items():
        if not isinstance(item, Mapping):
            continue
        subset = str(item.get("subset", ""))
        if split and subset and subset.lower() != split.lower():
            continue
        duration = _as_float(item.get("duration"))
        if duration is not None:
            out[str(video_id)] = float(duration)
    return out


def load_result_detection(path: str | Path) -> dict[str, list[Mapping[str, Any]]]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    results = payload.get("results", payload) if isinstance(payload, Mapping) else None
    if not isinstance(results, Mapping):
        raise ValueError("result_detection JSON must contain a results object or be a video-to-results object")
    out: dict[str, list[Mapping[str, Any]]] = {}
    for video_id, items in results.items():
        if items is None:
            out[str(video_id)] = []
            continue
        if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
            raise ValueError(f"results for video {video_id} must be a list")
        rows: list[Mapping[str, Any]] = []
        for idx, item in enumerate(items):
            if not isinstance(item, Mapping):
                raise ValueError(f"results for video {video_id} item {idx} must be an object")
            rows.append(item)
        out[str(video_id)] = rows
    return out


def _label_and_class_id(
    item: Mapping[str, Any],
    *,
    label_to_id: Mapping[str, int],
    id_to_label: Sequence[str],
) -> tuple[str | None, int | None]:
    raw_label = item.get("label", item.get("class", item.get("category")))
    raw_class_id = item.get("class_id", item.get("category_id", item.get("label_id")))
    class_id = _as_int(raw_class_id)
    label: str | None = None

    raw_label_int = _as_int(raw_label)
    if raw_label_int is not None and not isinstance(raw_label, str):
        class_id = raw_label_int if class_id is None else class_id
        if 0 <= raw_label_int < len(id_to_label) and id_to_label[raw_label_int]:
            label = str(id_to_label[raw_label_int])
        else:
            label = str(raw_label_int)
    elif raw_label is not None:
        label = str(raw_label)
        if class_id is None and label in label_to_id:
            class_id = int(label_to_id[label])

    if label is None and class_id is not None:
        if 0 <= class_id < len(id_to_label) and id_to_label[class_id]:
            label = str(id_to_label[class_id])
        else:
            label = str(class_id)
    return label, class_id


def result_detection_to_proposal_rows(
    result_detection: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    label_to_id: Mapping[str, int] | None = None,
    id_to_label: Sequence[str] = (),
    duration_by_video: Mapping[str, float] | None = None,
    limit_rows: int | None = None,
) -> list[dict[str, Any]]:
    labels = label_to_id or {}
    durations = duration_by_video or {}
    rows: list[dict[str, Any]] = []
    for video_id, items in sorted(result_detection.items()):
        for proposal_index, item in enumerate(items):
            context = f"video {video_id} result {proposal_index}"
            segment = _segment_seconds(item.get("segment", item.get("segment_seconds")), context=context)
            score = _score(item.get("score", item.get("final_score")), context=context)
            label, class_id = _label_and_class_id(item, label_to_id=labels, id_to_label=id_to_label)
            row: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "source_format": "opentad_result_detection_json",
                "video_id": str(video_id),
                "sample_id": str(video_id),
                "proposal_index": int(proposal_index),
                "segment_seconds": segment,
                "result_detection_segment_seconds": segment,
                "final_score": score,
                "score": score,
                "label": label,
                "class_id": class_id,
                "diagnostic_only": True,
                "uses_postprocessed_result_detection": True,
            }
            duration = durations.get(str(video_id))
            if duration is not None:
                row["duration_seconds"] = float(duration)
            rows.append(row)
            if limit_rows is not None and len(rows) >= int(limit_rows):
                return rows
    return rows


def run_result_detection_geometry_audit(
    *,
    result_detection_json: str | Path,
    output_dir: str | Path,
    annotation: str | Path | None = None,
    config: str | Path | None = None,
    class_map: str | Path | None = None,
    split: str = "validation",
    dataset_split_key: str = "val",
    topk: Sequence[int] = (100, 300, 1000),
    iou_thresholds: Sequence[float] = (0.3, 0.5, 0.7),
    score_bins: int = 10,
    group_by: str = "video",
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
    label_to_id, id_to_label = (
        load_class_map(class_map_path) if class_map_path is not None and class_map_path.exists() else ({}, [])
    )
    durations = load_video_durations(ann_path, split=split)
    result_detection = load_result_detection(result_detection_json)
    proposal_rows = result_detection_to_proposal_rows(
        result_detection,
        label_to_id=label_to_id,
        id_to_label=id_to_label,
        duration_by_video=durations,
        limit_rows=limit_rows,
    )
    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    converted_path = out_dir / "converted_result_detection_proposals.jsonl"
    write_jsonl(converted_path, proposal_rows)

    base_summary = run_localization_attribution(
        proposal_jsonl=converted_path,
        output_dir=out_dir,
        annotation=ann_path,
        class_map=class_map_path,
        split=split,
        dataset_split_key=dataset_split_key,
        topk=topk,
        iou_thresholds=iou_thresholds,
        score_bins=score_bins,
        group_by=group_by,
        require_seconds=True,
        include_ambiguous=include_ambiguous,
    )
    summary = dict(base_summary)
    summary.update(
        {
            "schema_version": SCHEMA_VERSION,
            "decision": READY,
            "diagnostic": "actionformer_result_detection_geometry_audit",
            "result_detection_json": str(Path(result_detection_json).expanduser()),
            "converted_result_detection_proposals_jsonl": str(converted_path),
            "result_detection_videos": len(result_detection),
            "converted_rows": len(proposal_rows),
            "source_segments_are_seconds": True,
            "group_by": group_by,
            "no_model_forward": True,
            "no_training": True,
            "uses_existing_result_detection_predictions": True,
            "uses_validation_gt": True,
            "uses_validation_gt_for_offline_diagnostic_join_only": True,
            "uses_teacher": False,
            "uses_oracle_for_training_or_test_protocol": False,
            "uses_cache": False,
            "metric_claim_allowed": False,
            "paper_claim_allowed": False,
            "runtime_or_deployment_claim_allowed": False,
        }
    )
    write_json(out_dir / "summary.json", summary)
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
    parser = argparse.ArgumentParser(
        description="Join ActionFormer result_detection.json proposals to validation GT for read-only geometry audit."
    )
    parser.add_argument("--result-detection-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--annotation")
    parser.add_argument("--config")
    parser.add_argument("--class-map")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--dataset-split-key", default="val")
    parser.add_argument("--topk", default="100,300,1000")
    parser.add_argument("--iou-thresholds", default="0.3,0.5,0.7")
    parser.add_argument("--score-bins", type=int, default=10)
    parser.add_argument("--group-by", choices=("sample", "video"), default="video")
    parser.add_argument("--include-ambiguous", action="store_true")
    parser.add_argument("--limit-rows", type=int)
    args = parser.parse_args(argv)

    try:
        summary = run_result_detection_geometry_audit(
            result_detection_json=args.result_detection_json,
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
