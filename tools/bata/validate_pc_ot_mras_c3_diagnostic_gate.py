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

from tools.bata.export_pc_ot_mras_hard_positions import strict_json_value, write_json  # noqa: E402


SCHEMA_VERSION = "pc_ot_mras_c3_diagnostic_gate_v0"
READY = "PC_OT_MRAS_C3_DIAGNOSTIC_GATE_READY"
NO_GO = "PC_OT_MRAS_C3_DIAGNOSTIC_GATE_NO_GO"
PROPOSAL_RANKING_READY_DECISIONS = {
    "NATIVE_IRREGULAR_AREA_HEAD_P2_LOCALIZATION_ATTRIBUTION_READY",
}
PROPOSAL_CAP_READY_DECISIONS = {
    "ACTIONFORMER_POST_NMS_OVERLOAD_AUDIT_READY",
}
SELECTOR_READY_DECISION = "PC_OT_MRAS_SELECTOR_POSTTRAIN_DIAGNOSTIC_READY"


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def _nested(mapping: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _status(missing: Sequence[str]) -> str:
    return "PASS" if not missing else "NO_GO"


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and int(value) > 0


def _non_null(value: Any) -> bool:
    return value is not None


def _norm_path(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).replace("\\", "/").rstrip("/")
    return text or None


def _histogram_count(histogram: Mapping[str, Any], key: int) -> int | None:
    raw = histogram.get(str(int(key)))
    if raw is None:
        raw = histogram.get(int(key))
    if raw is None:
        return 0
    if isinstance(raw, int) and not isinstance(raw, bool):
        return int(raw)
    return None


def _load_inline_or_path(summary: Mapping[str, Any], *, inline_key: str, path_key: str) -> tuple[dict[str, Any] | None, str | None]:
    inline = summary.get(inline_key)
    if isinstance(inline, Mapping):
        return dict(inline), "inline"
    path = summary.get(path_key)
    if isinstance(path, str) and path:
        path_obj = Path(path).expanduser()
        if path_obj.exists():
            return _load_json(path_obj), str(path_obj)
    return None, None


def _selector_dump_gate(selector_summary: Mapping[str, Any], *, min_selector_samples: int) -> dict[str, Any]:
    missing: list[str] = []
    decision = str(selector_summary.get("decision", ""))
    if decision != SELECTOR_READY_DECISION:
        missing.append("selector diagnostic decision must be strict READY")

    non_finite = selector_summary.get("non_finite")
    non_finite_count = None
    if isinstance(non_finite, Mapping):
        raw_count = non_finite.get("count")
        if isinstance(raw_count, int) and not isinstance(raw_count, bool):
            non_finite_count = int(raw_count)
    if non_finite_count is None:
        missing.append("selector non_finite count")
    elif non_finite_count > 0:
        missing.append("selector dump must not contain NaN/Inf values")

    sample_count = selector_summary.get("sample_count")
    if not isinstance(sample_count, int) or sample_count < int(min_selector_samples):
        missing.append(f"selector sample_count must be >= {int(min_selector_samples)}")

    samples = selector_summary.get("samples")
    if not isinstance(samples, list) or not samples:
        missing.append("selector samples with selected_dense_indices")
    else:
        if not all(isinstance(item, Mapping) and item.get("selected_dense_indices") for item in samples):
            missing.append("every selector sample must contain selected_dense_indices")

    aggregate = selector_summary.get("aggregate")
    if not isinstance(aggregate, Mapping):
        missing.append("selector aggregate")
        aggregate = {}

    required_aggregate_paths = (
        ("duplicate_rate_mean", "selector duplicate/repeat rate"),
        ("gap_max_p95", "selector gap/max-gap p95"),
        ("boundary.near_selected_rate", "boundary-near selected rate"),
        ("slot_transport.raw_slot_duplicate_rate_mean", "raw_slot_duplicate rate"),
        ("slot_transport.reader_fill_count_mean", "reader_fill count"),
        ("slot_transport.st_active_row_count_mean", "ST active rows"),
    )
    for dotted, label in required_aggregate_paths:
        if _nested(aggregate, dotted.split(".")) is None:
            missing.append(label)

    metadata_inconsistent_count = _nested(aggregate, ("metadata_consistency", "inconsistent_sample_count"))
    if not isinstance(metadata_inconsistent_count, int) or isinstance(metadata_inconsistent_count, bool):
        missing.append("selector metadata consistency count")
    elif int(metadata_inconsistent_count) > 0:
        missing.append("selector metadata consistency must be clean")

    return {
        "status": _status(missing),
        "missing": missing,
        "observed": {
            "sample_count": sample_count,
            "decision": decision,
            "non_finite_count": non_finite_count,
            "metadata_inconsistent_sample_count": metadata_inconsistent_count,
            "duplicate_rate_mean": aggregate.get("duplicate_rate_mean"),
            "gap_max_p95": aggregate.get("gap_max_p95"),
            "boundary_near_selected_rate": _nested(aggregate, ("boundary", "near_selected_rate")),
            "raw_slot_duplicate_rate_mean": _nested(aggregate, ("slot_transport", "raw_slot_duplicate_rate_mean")),
            "reader_fill_count_mean": _nested(aggregate, ("slot_transport", "reader_fill_count_mean")),
            "st_active_row_count_mean": _nested(aggregate, ("slot_transport", "st_active_row_count_mean")),
        },
    }


def _proposal_ranking_gate(proposal_summary: Mapping[str, Any] | None) -> dict[str, Any]:
    missing: list[str] = []
    observed: dict[str, Any] = {}
    if proposal_summary is None:
        return {"status": "NO_GO", "missing": ["proposal ranking summary"], "observed": observed}

    decision = str(proposal_summary.get("decision", ""))
    if decision not in PROPOSAL_RANKING_READY_DECISIONS:
        missing.append("proposal ranking diagnostic decision must be a recognized READY decision")
    if not _positive_int(proposal_summary.get("proposal_rows")):
        missing.append("proposal_rows > 0")
    if not _positive_int(proposal_summary.get("joined_rows")):
        missing.append("joined_rows > 0")

    topk, topk_source = _load_inline_or_path(
        proposal_summary,
        inline_key="topk_iou_rank_summary",
        path_key="topk_iou_rank_summary_json",
    )
    reliability, reliability_source = _load_inline_or_path(
        proposal_summary,
        inline_key="score_iou_reliability",
        path_key="score_iou_reliability_json",
    )
    if topk is None:
        missing.append("top-k IoU rank summary")
    elif topk.get("diagnostic") != "top_k_iou_rank_dump":
        missing.append("top-k IoU rank summary diagnostic marker")
    if reliability is None:
        missing.append("score-IoU reliability summary")
    elif reliability.get("diagnostic") != "score_vs_iou_reliability":
        missing.append("score-IoU reliability diagnostic marker")

    observed.update(
        {
            "proposal_rows": proposal_summary.get("proposal_rows"),
            "joined_rows": proposal_summary.get("joined_rows"),
            "topk_iou_rank_summary": topk_source,
            "score_iou_reliability": reliability_source,
        }
    )
    return {"status": _status(missing), "missing": missing, "observed": observed}


def _proposal_cap_gate(overload_summary: Mapping[str, Any] | None) -> dict[str, Any]:
    missing: list[str] = []
    observed: dict[str, Any] = {}
    if overload_summary is None:
        return {"status": "NO_GO", "missing": ["proposal cap / post-NMS overload summary"], "observed": observed}

    decision = str(overload_summary.get("decision", ""))
    if decision not in PROPOSAL_CAP_READY_DECISIONS:
        missing.append("proposal cap diagnostic decision must be a recognized READY decision")
    summary = overload_summary.get("summary")
    if not isinstance(summary, Mapping):
        missing.append("proposal cap summary")
        summary = {}
    required = (
        ("latest_predictions", "latest prediction count"),
        ("annotation_validation_video_count", "validation video count"),
        ("post_processing_max_seg_num", "post-processing max_seg_num"),
        ("expected_dataset_prediction_cap", "expected dataset proposal cap"),
        ("all_logged_prediction_counts_equal_cap", "whether every logged prediction count equals proposal cap"),
        ("cap_saturation_ratio", "proposal cap saturation ratio"),
        ("latest_predictions_per_video", "latest predictions per video"),
    )
    for key, label in required:
        if not _non_null(summary.get(key)):
            missing.append(label)
    observed.update({key: summary.get(key) for key, _label in required})

    if summary.get("all_logged_prediction_counts_equal_cap") is not True:
        missing.append("every logged prediction count must equal proposal cap")
    cap_saturation_ratio = summary.get("cap_saturation_ratio")
    if (
        isinstance(cap_saturation_ratio, bool)
        or not isinstance(cap_saturation_ratio, (int, float))
        or not math.isclose(float(cap_saturation_ratio), 1.0, rel_tol=0.0, abs_tol=1.0e-9)
    ):
        missing.append("proposal cap saturation ratio must be 1.0")

    result_counts = summary.get("result_detection_counts")
    max_seg_num = summary.get("post_processing_max_seg_num")
    video_count = summary.get("annotation_validation_video_count")
    if not isinstance(result_counts, Mapping):
        missing.append("per-video result_detection proposal counts")
    else:
        count_video_count = result_counts.get("video_count")
        histogram = result_counts.get("per_video_count_histogram")
        if not isinstance(histogram, Mapping) or not histogram:
            missing.append("per-video result_detection proposal count histogram")
            histogram = {}
        if (
            isinstance(count_video_count, int)
            and isinstance(video_count, int)
            and count_video_count != video_count
        ):
            missing.append("result_detection video_count must match validation video count")
        if summary.get("result_detection_total_matches_latest_log") is not True:
            missing.append("result_detection total must match latest logged prediction count")
        if isinstance(max_seg_num, int) and not isinstance(max_seg_num, bool):
            cap_hit_count = _histogram_count(histogram, int(max_seg_num))
        else:
            cap_hit_count = None
        if cap_hit_count is None:
            missing.append("per-video proposal cap hit count")
        elif isinstance(video_count, int) and not isinstance(video_count, bool) and cap_hit_count != video_count:
            missing.append("every result_detection video must hit proposal cap")
        observed.update(
            {
                "result_detection_video_count": count_video_count,
                "result_detection_total_predictions": result_counts.get("total_predictions"),
                "result_detection_per_video_min": result_counts.get("per_video_min"),
                "result_detection_per_video_max": result_counts.get("per_video_max"),
                "result_detection_per_video_mean": result_counts.get("per_video_mean"),
                "result_detection_per_video_count_histogram": dict(histogram),
                "result_detection_total_matches_latest_log": summary.get("result_detection_total_matches_latest_log"),
                "result_detection_cap_hit_video_count": cap_hit_count,
                "result_detection_cap_hit_video_ratio": (
                    float(cap_hit_count) / float(video_count)
                    if isinstance(cap_hit_count, int) and isinstance(video_count, int) and video_count > 0
                    else None
                ),
                "result_detection_all_videos_at_cap": (
                    cap_hit_count == video_count
                    if isinstance(cap_hit_count, int) and isinstance(video_count, int)
                    else None
                ),
            }
        )
    return {"status": _status(missing), "missing": missing, "observed": observed}


def _provenance_gate(
    *,
    selector_summary: Mapping[str, Any],
    proposal_summary: Mapping[str, Any] | None,
    overload_summary: Mapping[str, Any] | None,
    expected_run_root: str | None = None,
    expected_work_dir: str | None = None,
    expected_train_stdout: str | None = None,
    expected_result_detection_json: str | None = None,
) -> dict[str, Any]:
    expected = {
        "run_root": _norm_path(expected_run_root),
        "work_dir": _norm_path(expected_work_dir),
        "train_stdout": _norm_path(expected_train_stdout),
        "result_detection_json": _norm_path(expected_result_detection_json),
    }
    expected = {key: value for key, value in expected.items() if value is not None}
    if not expected:
        return {"status": "PASS", "missing": [], "observed": {"expected": {}}}

    missing: list[str] = []
    observed: dict[str, Any] = {"expected": expected, "summaries": {}}
    summaries: tuple[tuple[str, Mapping[str, Any] | None], ...] = (
        ("selector", selector_summary),
        ("proposal", proposal_summary),
        ("overload", overload_summary),
    )
    for name, summary in summaries:
        if summary is None:
            missing.append(f"{name} summary provenance")
            continue
        provenance = summary.get("provenance")
        if not isinstance(provenance, Mapping):
            missing.append(f"{name} summary provenance")
            observed["summaries"][name] = None
            continue
        normalized = {key: _norm_path(provenance.get(key)) for key in expected}
        observed["summaries"][name] = normalized
        for key, expected_value in expected.items():
            if normalized.get(key) != expected_value:
                if key == "run_root":
                    missing.append("run_root must match current C3 run")
                elif key == "result_detection_json":
                    missing.append("result_detection_json must match current C3 result file")
                else:
                    missing.append(f"{key} must match current C3 run")

    if expected.get("result_detection_json") is not None and isinstance(overload_summary, Mapping):
        result_detection_json = _nested(
            overload_summary,
            ("summary", "result_detection_counts", "result_detection_json"),
        )
        observed["overload_result_detection_json"] = _norm_path(result_detection_json)
        if _norm_path(result_detection_json) != expected["result_detection_json"]:
            missing.append("result_detection_json must match current C3 result file")

    # Keep the report compact when the same missing item is found in all three summaries.
    deduped_missing = list(dict.fromkeys(missing))
    return {"status": _status(deduped_missing), "missing": deduped_missing, "observed": observed}


def validate_c3_diagnostic_gate_payloads(
    *,
    selector_summary: Mapping[str, Any],
    proposal_summary: Mapping[str, Any] | None,
    overload_summary: Mapping[str, Any] | None,
    min_selector_samples: int = 1,
    expected_run_root: str | None = None,
    expected_work_dir: str | None = None,
    expected_train_stdout: str | None = None,
    expected_result_detection_json: str | None = None,
) -> dict[str, Any]:
    selector_gate = _selector_dump_gate(selector_summary, min_selector_samples=int(min_selector_samples))
    proposal_gate = _proposal_ranking_gate(proposal_summary)
    cap_gate = _proposal_cap_gate(overload_summary)
    provenance_gate = _provenance_gate(
        selector_summary=selector_summary,
        proposal_summary=proposal_summary,
        overload_summary=overload_summary,
        expected_run_root=expected_run_root,
        expected_work_dir=expected_work_dir,
        expected_train_stdout=expected_train_stdout,
        expected_result_detection_json=expected_result_detection_json,
    )
    gates = {
        "selector_dump": selector_gate,
        "proposal_ranking": proposal_gate,
        "proposal_cap": cap_gate,
        "provenance": provenance_gate,
    }
    blockers = [
        f"{name}: {', '.join(item['missing'])}"
        for name, item in gates.items()
        if item["status"] != "PASS"
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": READY if not blockers else NO_GO,
        "gate": gates,
        "blocking_findings": blockers,
        "interpretation_boundary": (
            "PASS means diagnostic evidence is complete enough to distinguish selector selection quality "
            "from downstream selected-axis geometry/ranking failure. It is not a metric or paper claim."
        ),
        "protocol_flags": {
            "diagnostic_only": True,
            "runs_training": False,
            "runs_tools_test": False,
            "tools_train_allowed": False,
            "tools_test_allowed": False,
            "remote_sync_allowed": False,
            "remote_deploy_allowed": False,
            "slurm_gpu_allowed": False,
            "checkpoint_write_allowed": False,
            "metric_claim_allowed": False,
            "paper_claim_allowed": False,
            "runtime_flops_claim_allowed": False,
            "deploy_claim_allowed": False,
        },
    }


def run_c3_diagnostic_gate(
    *,
    selector_summary_path: str | Path,
    proposal_summary_path: str | Path | None = None,
    overload_summary_path: str | Path | None = None,
    output_json: str | Path | None = None,
    min_selector_samples: int = 1,
    expected_run_root: str | None = None,
    expected_work_dir: str | None = None,
    expected_train_stdout: str | None = None,
    expected_result_detection_json: str | None = None,
) -> dict[str, Any]:
    payload = validate_c3_diagnostic_gate_payloads(
        selector_summary=_load_json(selector_summary_path),
        proposal_summary=_load_json(proposal_summary_path) if proposal_summary_path is not None else None,
        overload_summary=_load_json(overload_summary_path) if overload_summary_path is not None else None,
        min_selector_samples=int(min_selector_samples),
        expected_run_root=expected_run_root,
        expected_work_dir=expected_work_dir,
        expected_train_stdout=expected_train_stdout,
        expected_result_detection_json=expected_result_detection_json,
    )
    if output_json is not None:
        write_json(output_json, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate that C3/C3+ selector and proposal diagnostics are complete before result attribution."
    )
    parser.add_argument("--selector-summary", required=True)
    parser.add_argument("--proposal-summary")
    parser.add_argument("--overload-summary")
    parser.add_argument("--output")
    parser.add_argument("--min-selector-samples", type=int, default=1)
    parser.add_argument("--expected-run-root")
    parser.add_argument("--expected-work-dir")
    parser.add_argument("--expected-train-stdout")
    parser.add_argument("--expected-result-detection-json")
    args = parser.parse_args(argv)
    try:
        payload = run_c3_diagnostic_gate(
            selector_summary_path=args.selector_summary,
            proposal_summary_path=args.proposal_summary,
            overload_summary_path=args.overload_summary,
            output_json=args.output,
            min_selector_samples=int(args.min_selector_samples),
            expected_run_root=args.expected_run_root,
            expected_work_dir=args.expected_work_dir,
            expected_train_stdout=args.expected_train_stdout,
            expected_result_detection_json=args.expected_result_detection_json,
        )
    except Exception as exc:
        payload = {"schema_version": SCHEMA_VERSION, "decision": NO_GO, "error": str(exc)}
        print(json.dumps(strict_json_value(payload), sort_keys=True))
        return 1
    print(json.dumps(strict_json_value(payload), sort_keys=True))
    return 0 if payload["decision"] == READY else 1


if __name__ == "__main__":
    raise SystemExit(main())
