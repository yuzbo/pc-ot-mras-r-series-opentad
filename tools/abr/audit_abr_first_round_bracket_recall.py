from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from opentad.acquisition.abr import ABRConfig, ABR_ROUTE_LABEL, select_active_bracket_refinement
from opentad.acquisition.abr.types import BracketState
from opentad.acquisition.abr.validators import (
    ABRValidationError,
    assert_no_forbidden_route_tokens,
    assert_no_forbidden_selection_inputs,
)


CLAIM_LOCKS = {
    "formal_full_train_unlocked": False,
    "tools_test_allowed": False,
    "official_map_claim": False,
    "runtime_flops_claim": False,
    "deploy_claim": False,
    "sparse_compute_claim": False,
    "paper_claim": False,
}

FORBIDDEN_SCOUT_KEYS = {
    "gt_segments",
    "gt_labels",
    "teacher_logits",
    "teacher_features",
    "prediction_cache",
    "prediction_cache_path",
    "raw_predictions",
    "raw_detector_outputs",
    "detector_outputs",
    "detector_predictions",
    "detector_feedback",
    "dense_backbone_features",
    "dense_backbone_handoff",
    "dense_raw_backbone_handoff",
    "gt_scout_curve",
    "gt_frame_signal",
    "teacher_scout_curve",
    "prediction_scout_curve",
    "detector_scout_curve",
    "globalrank_tokens",
}

SCOUT_CURVE_KEYS = (
    "abr_scout_curve",
    "deploy_visible_scout_curve",
    "low_cost_scout_curve",
    "scout_curve",
    "abr_frame_signal",
    "raw_frame_signal",
    "frame_signal",
    "low_cost_frame_signal",
)


@dataclass(frozen=True)
class GTInstance:
    video_id: str
    label: str
    start: float
    end: float
    duration: float


@dataclass(frozen=True)
class WindowCase:
    video_id: str
    window_id: str
    video_info: Mapping[str, Any]
    duration: float
    fps: float
    dense_t: int
    scout_curve: list[float] | None
    scout_source: str
    window_start_seconds: float
    window_end_seconds: float
    selector_payload: dict[str, Any]


def run_audit(
    annotation_json: Path,
    scout_json: Path | None = None,
    subset: str = "validation",
    max_videos: int | None = None,
    max_windows: int | None = None,
    window_size: int = 0,
    window_overlap_ratio: float = 0.5,
    allow_diagnostic_fallback_scout: bool = False,
    fallback_stage: str = "DIAGNOSTIC_ONLY",
    min_first_round_bracket_recall: float = 0.95,
    min_first_round_transition_coverage: float = 0.95,
    abr_config: ABRConfig | None = None,
) -> dict[str, Any]:
    ann = _load_json(annotation_json)
    videos = _load_annotation_videos(ann, subset=subset, max_videos=max_videos)
    scout_records = _load_scout_records(scout_json) if scout_json is not None else {}
    cfg = abr_config or ABRConfig(
        k0=96,
        k1_cap=160,
        k2_cap=64,
        max_total_k=384,
        max_gap=24,
        target_frame_num=384,
        round2_enabled=True,
        route_label=ABR_ROUTE_LABEL,
        allow_diagnostic_fallback_scout=allow_diagnostic_fallback_scout,
        fallback_stage=fallback_stage,
    )
    if cfg.route_label != ABR_ROUTE_LABEL:
        raise ABRValidationError("ABR first-round audit requires the ABR route label")
    assert_no_forbidden_route_tokens({"route_label": ABR_ROUTE_LABEL, "method": "abr_first_round_bracket_recall"})

    cases = _build_window_cases(
        videos,
        scout_records,
        cfg,
        window_size=window_size,
        window_overlap_ratio=window_overlap_ratio,
        max_windows=max_windows,
        allow_diagnostic_fallback_scout=allow_diagnostic_fallback_scout,
        fallback_stage=fallback_stage,
    )
    if not cases:
        raise ABRValidationError("LOCKED: no auditable ABR windows were built")

    totals = _empty_totals()
    window_summaries = []
    all_widths = []
    all_width_seconds = []
    scout_sources = set()
    fallback_used = False
    selector_gt_visible = False
    selector_payload_gt_keys_stripped = True
    video_ids = set()

    for case in cases:
        assert_selector_payload_is_deploy_visible(case.selector_payload)
        if any(key in case.selector_payload for key in ("gt_segments", "gt_labels")):
            selector_gt_visible = True
            selector_payload_gt_keys_stripped = False

        selection = select_active_bracket_refinement(
            dense_t=case.dense_t,
            fps=case.fps,
            video_id=case.video_id,
            window_id=case.window_id,
            scout_curve=case.scout_curve,
            scout_source=case.scout_source,
            config=cfg,
        )
        fallback_used = fallback_used or bool(selection.diagnostic_fallback_used)
        scout_sources.add(str(selection.scout_source))
        video_ids.add(case.video_id)
        round0 = _round0_brackets(selection.brackets)
        gt_instances, ambiguous_transition_count = _gt_instances_for_window(
            case.video_id, case.video_info, case.window_start_seconds, case.window_end_seconds
        )
        score = _score_window(case, round0, gt_instances, ambiguous_transition_count)
        _accumulate(totals, score, case, round0, gt_instances)
        all_widths.extend([bracket.width for bracket in round0])
        all_width_seconds.extend([bracket.width_seconds(case.fps) for bracket in round0])
        if score["missed_transition_count"] > 0:
            window_summaries.append(
                {
                    "video_id": case.video_id,
                    "window_id": case.window_id,
                    "transition_count": score["transition_count"],
                    "missed_transition_count": score["missed_transition_count"],
                    "missed_transitions": score["missed_transitions"][:16],
                    "scout_source": selection.scout_source,
                }
            )

    transition_count = totals["transition_count"]
    bracketed_transition_count = totals["bracketed_transition_count"]
    missed_transition_count = totals["missed_transition_count"]
    action_instance_count = totals["action_instance_count"]
    fully_bracketed_action_count = totals["fully_bracketed_action_count"]
    temporal_coverage_fraction = (
        totals["covered_dense_positions"] / float(totals["dense_positions"]) if totals["dense_positions"] > 0 else 0.0
    )
    first_round_bracket_recall = _safe_fraction(bracketed_transition_count, transition_count)
    first_round_transition_coverage = _safe_fraction(fully_bracketed_action_count, action_instance_count)
    false_positive_bracket_density = (
        totals["false_positive_bracket_count"] / float(totals["dense_positions"])
        if totals["dense_positions"] > 0
        else 0.0
    )
    formal_thresholds = {
        "min_first_round_bracket_recall": float(min_first_round_bracket_recall),
        "min_first_round_transition_coverage": float(min_first_round_transition_coverage),
    }
    real_evidence = (
        not fallback_used
        and not selector_gt_visible
        and transition_count > 0
        and len(cases) > 0
        and all(not src.startswith("diagnostic_fallback:") for src in scout_sources)
    )
    formal_gate_passed = (
        real_evidence
        and missed_transition_count == 0
        and first_round_bracket_recall >= formal_thresholds["min_first_round_bracket_recall"]
        and first_round_transition_coverage >= formal_thresholds["min_first_round_transition_coverage"]
    )
    allowed_next_action = (
        "FORMAL_REVIEW_PACKET_ONLY_WITH_REAL_SCOUT_RECALL_EVIDENCE"
        if formal_gate_passed
        else _locked_next_action(
            transition_count=transition_count,
            fallback_used=fallback_used,
            selector_gt_visible=selector_gt_visible,
            real_evidence=real_evidence,
        )
    )

    payload = {
        "route_label": ABR_ROUTE_LABEL,
        "method": "abr_first_round_bracket_recall_diagnostic",
        "diagnostic_only": True,
        "no_detector": True,
        "no_training": True,
        "selector_gt_visible": bool(selector_gt_visible),
        "selector_payload_gt_keys_stripped": bool(selector_payload_gt_keys_stripped),
        "real_deploy_visible_recall_evidence": bool(real_evidence),
        "formal_thresholds": formal_thresholds,
        "formal_gate_passed": bool(formal_gate_passed),
        "scout_sources": sorted(scout_sources),
        "video_count": int(len(video_ids)),
        "window_count": int(len(cases)),
        "transition_count": int(transition_count),
        "bracketed_transition_count": int(bracketed_transition_count),
        "missed_transition_count": int(missed_transition_count),
        "first_round_bracket_recall": first_round_bracket_recall,
        "first_round_transition_coverage": first_round_transition_coverage,
        "temporal_coverage_fraction": float(temporal_coverage_fraction),
        "bracket_width_stats": _stats(all_widths),
        "bracket_width_seconds_stats": _stats(all_width_seconds),
        "false_positive_bracket_density": float(false_positive_bracket_density),
        "false_positive_bracket_count": int(totals["false_positive_bracket_count"]),
        "short_action_stratified_recall": _short_action_recall(totals["short_action_bins"]),
        "class_misses": dict(sorted(totals["class_misses"].items())),
        "video_misses": dict(sorted(totals["video_misses"].items())),
        "window_misses": window_summaries[:64],
        "ambiguous_transition_count": int(totals["ambiguous_transition_count"]),
        "zero_transition_pseudo_perfect_rejected": bool(transition_count == 0),
        "diagnostic_fallback_used": bool(fallback_used),
        "allowed_next_action": allowed_next_action,
        **CLAIM_LOCKS,
    }
    assert_no_forbidden_route_tokens(payload)
    return payload


def assert_selector_payload_is_deploy_visible(payload: Mapping[str, Any]) -> None:
    _assert_no_forbidden_keys_recursive(payload)
    assert_no_forbidden_selection_inputs(payload, allow_gt_after_selection=False)
    assert_no_forbidden_route_tokens(payload)


def build_selector_payload(record: Mapping[str, Any], video_id: str, window_id: str, dense_t: int) -> dict[str, Any]:
    _assert_no_forbidden_keys_recursive(record)
    payload = {
        "video_name": video_id,
        "window_id": window_id,
        "total_frames": int(record.get("total_frames", record.get("frame", dense_t))),
        "avg_fps": float(record.get("avg_fps", record.get("fps", 25.0))),
        "abr_first_round_audit": True,
    }
    for key in SCOUT_CURVE_KEYS:
        if key in record:
            payload[key] = record[key]
            break
    assert_selector_payload_is_deploy_visible(payload)
    return payload


def _load_annotation_videos(payload: Mapping[str, Any], subset: str, max_videos: int | None) -> dict[str, dict[str, Any]]:
    database = payload.get("database", payload)
    if not isinstance(database, Mapping):
        raise ABRValidationError("LOCKED: annotation JSON must contain a database mapping")
    selected = {}
    for video_id, info in database.items():
        if not isinstance(info, Mapping):
            continue
        if subset and str(info.get("subset", "")).lower() != subset.lower():
            continue
        duration = float(info.get("duration", 0.0))
        frame = int(info.get("frame", info.get("num_frames", info.get("total_frames", 0))))
        if duration <= 0.0:
            raise ABRValidationError(f"LOCKED: video {video_id} has missing/invalid duration")
        annotations = info.get("annotations", [])
        if not isinstance(annotations, Sequence):
            raise ABRValidationError(f"LOCKED: video {video_id} annotations must be a sequence")
        selected[str(video_id)] = {"duration": duration, "frame": frame, "annotations": list(annotations)}
        if max_videos is not None and len(selected) >= int(max_videos):
            break
    if not selected:
        raise ABRValidationError(f"LOCKED: no videos found for subset={subset!r}")
    return selected


def _load_scout_records(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ABRValidationError("LOCKED: scout JSON is empty")
    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        records = json.loads(text)
    if isinstance(records, Mapping):
        if "videos" in records and isinstance(records["videos"], Mapping):
            records = records["videos"]
        elif "records" in records and isinstance(records["records"], Sequence):
            records = records["records"]
        elif "database" in records and isinstance(records["database"], Mapping):
            records = records["database"]
    result: dict[str, Any] = {}
    if isinstance(records, Mapping):
        for video_id, record in records.items():
            if not isinstance(record, Mapping):
                raise ABRValidationError(f"LOCKED: scout record for {video_id} must be a mapping")
            _assert_no_forbidden_keys_recursive(record)
            merged = dict(record)
            merged.setdefault("video_id", video_id)
            result[str(video_id)] = merged
        return result
    if isinstance(records, Sequence):
        for idx, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise ABRValidationError(f"LOCKED: scout record #{idx} must be a mapping")
            _assert_no_forbidden_keys_recursive(record)
            video_id = str(record.get("video_id", record.get("video_name", "")))
            if not video_id:
                raise ABRValidationError(f"LOCKED: scout record #{idx} missing video_id")
            result[video_id] = dict(record)
        return result
    raise ABRValidationError("LOCKED: unsupported scout record format")


def _build_window_cases(
    videos: Mapping[str, Mapping[str, Any]],
    scout_records: Mapping[str, Any],
    cfg: ABRConfig,
    window_size: int,
    window_overlap_ratio: float,
    max_windows: int | None,
    allow_diagnostic_fallback_scout: bool,
    fallback_stage: str,
) -> list[WindowCase]:
    cases: list[WindowCase] = []
    for video_id, info in videos.items():
        scout_record = scout_records.get(video_id, {})
        if scout_record:
            _assert_no_forbidden_keys_recursive(scout_record)
        windows = _explicit_windows(scout_record)
        if windows:
            for idx, window in enumerate(windows):
                cases.append(_case_from_record(video_id, info, window, idx, cfg, allow_diagnostic_fallback_scout, fallback_stage))
                if max_windows is not None and len(cases) >= int(max_windows):
                    return cases
            continue

        curve = _extract_curve(scout_record)
        if curve is None:
            if not allow_diagnostic_fallback_scout:
                raise ABRValidationError(
                    f"LOCKED: video {video_id} has no deploy-visible scout curve; "
                    "use --allow-diagnostic-fallback-scout only for diagnostic/precheck non-evidence"
                )
            dense_t = max(int(info.get("frame", 0)), 1)
            curve = None
        else:
            dense_t = len(curve)

        if window_size and curve is not None and len(curve) > int(window_size):
            for idx, (start, end) in enumerate(_sliding_dense_windows(len(curve), window_size, window_overlap_ratio)):
                record = dict(scout_record)
                record["scout_curve"] = curve[start:end]
                record["window_id"] = f"window{idx:04d}"
                record["window_start_seconds"] = _dense_to_seconds(start, len(curve), float(info["duration"]))
                record["window_end_seconds"] = _dense_to_seconds(end - 1, len(curve), float(info["duration"]))
                cases.append(_case_from_record(video_id, info, record, idx, cfg, allow_diagnostic_fallback_scout, fallback_stage))
                if max_windows is not None and len(cases) >= int(max_windows):
                    return cases
        else:
            record = dict(scout_record)
            if curve is not None:
                record["scout_curve"] = curve
            record.setdefault("window_id", "full_video")
            record.setdefault("window_start_seconds", 0.0)
            record.setdefault("window_end_seconds", float(info["duration"]))
            if curve is None:
                record.setdefault("dense_t", dense_t)
            cases.append(_case_from_record(video_id, info, record, 0, cfg, allow_diagnostic_fallback_scout, fallback_stage))
            if max_windows is not None and len(cases) >= int(max_windows):
                return cases
    return cases


def _case_from_record(
    video_id: str,
    info: Mapping[str, Any],
    record: Mapping[str, Any],
    idx: int,
    cfg: ABRConfig,
    allow_diagnostic_fallback_scout: bool,
    fallback_stage: str,
) -> WindowCase:
    curve = _extract_curve(record)
    dense_t = int(record.get("dense_t", len(curve) if curve is not None else max(int(info.get("frame", 0)), 1)))
    if dense_t <= 0:
        raise ABRValidationError(f"LOCKED: video {video_id} window #{idx} has non-positive dense_t")
    source = _scout_source(record, allow_diagnostic_fallback_scout, fallback_stage)
    window_id = str(record.get("window_id", f"window{idx:04d}"))
    duration = float(info["duration"])
    window_start = float(record.get("window_start_seconds", record.get("start_seconds", 0.0)))
    window_end = float(record.get("window_end_seconds", record.get("end_seconds", duration)))
    if not (0.0 <= window_start < window_end <= duration + 1e-6):
        raise ABRValidationError(f"LOCKED: video {video_id} window {window_id} has invalid temporal bounds")
    fps = float(record.get("fps", record.get("avg_fps", info.get("frame", 0) / duration if info.get("frame", 0) else 25.0)))
    selector_payload = build_selector_payload(record, video_id, window_id, dense_t)
    assert_selector_payload_is_deploy_visible(selector_payload)
    return WindowCase(
        video_id=video_id,
        window_id=window_id,
        video_info=info,
        duration=duration,
        fps=fps,
        dense_t=dense_t,
        scout_curve=curve,
        scout_source=source,
        window_start_seconds=window_start,
        window_end_seconds=window_end,
        selector_payload=selector_payload,
    )


def _gt_instances_for_window(
    video_id: str,
    info: Mapping[str, Any],
    window_start: float,
    window_end: float,
) -> tuple[list[GTInstance], int]:
    instances = []
    ambiguous_transition_count = 0
    for ann in info.get("annotations", []):
        if not isinstance(ann, Mapping):
            raise ABRValidationError(f"LOCKED: video {video_id} has non-mapping annotation")
        label = str(ann.get("label", ""))
        seg = ann.get("segment")
        if not isinstance(seg, Sequence) or len(seg) != 2:
            raise ABRValidationError(f"LOCKED: video {video_id} has annotation with missing segment")
        start, end = float(seg[0]), float(seg[1])
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            raise ABRValidationError(f"LOCKED: video {video_id} has invalid annotation segment")
        if label == "Ambiguous":
            if start < window_end and end > window_start:
                ambiguous_transition_count += 2
            continue
        clipped_start = max(start, window_start)
        clipped_end = min(end, window_end)
        if clipped_end <= clipped_start:
            continue
        instances.append(GTInstance(video_id, label, clipped_start, clipped_end, clipped_end - clipped_start))
    return instances, ambiguous_transition_count


def _score_window(
    case: WindowCase,
    brackets: Sequence[BracketState],
    gt_instances: Sequence[GTInstance],
    ambiguous_transition_count: int,
) -> dict[str, Any]:
    transitions = []
    missed = []
    bracketed = 0
    fully_bracketed_action_count = 0
    span = max(case.window_end_seconds - case.window_start_seconds, 1e-6)
    for inst in gt_instances:
        start_pos = _seconds_to_local_dense(inst.start, case.window_start_seconds, span, case.dense_t)
        end_pos = _seconds_to_local_dense(inst.end, case.window_start_seconds, span, case.dense_t)
        start_hit = _covered_by_any_bracket(start_pos, brackets)
        end_hit = _covered_by_any_bracket(end_pos, brackets)
        if start_hit and end_hit:
            fully_bracketed_action_count += 1
        for kind, pos, hit in (("start", start_pos, start_hit), ("end", end_pos, end_hit)):
            item = {
                "video_id": case.video_id,
                "window_id": case.window_id,
                "label": inst.label,
                "kind": kind,
                "position": int(pos),
                "time_seconds": float(inst.start if kind == "start" else inst.end),
                "action_duration_seconds": float(inst.duration),
            }
            transitions.append(item)
            if hit:
                bracketed += 1
            else:
                missed.append(item)
    covered_dense = _bracket_union_covered_positions(brackets, case.dense_t)
    false_positive = 0
    for bracket in brackets:
        if not any(bracket.left <= transition["position"] <= bracket.right for transition in transitions):
            false_positive += 1
    return {
        "transition_count": len(transitions),
        "bracketed_transition_count": bracketed,
        "missed_transition_count": len(missed),
        "missed_transitions": missed,
        "fully_bracketed_action_count": fully_bracketed_action_count,
        "action_instance_count": len(gt_instances),
        "covered_dense_positions": covered_dense,
        "dense_positions": case.dense_t,
        "false_positive_bracket_count": false_positive,
        "ambiguous_transition_count": int(ambiguous_transition_count),
    }


def _accumulate(
    totals: dict[str, Any],
    score: Mapping[str, Any],
    case: WindowCase,
    brackets: Sequence[BracketState],
    gt_instances: Sequence[GTInstance],
) -> None:
    for key in (
        "transition_count",
        "bracketed_transition_count",
        "missed_transition_count",
        "fully_bracketed_action_count",
        "action_instance_count",
        "covered_dense_positions",
        "dense_positions",
        "false_positive_bracket_count",
        "ambiguous_transition_count",
    ):
        totals[key] += int(score[key])
    for missed in score["missed_transitions"]:
        label = str(missed["label"])
        totals["class_misses"][label] = totals["class_misses"].get(label, 0) + 1
        totals["video_misses"][case.video_id] = totals["video_misses"].get(case.video_id, 0) + 1
    for inst in gt_instances:
        bin_name = _short_bin(inst.duration)
        bucket = totals["short_action_bins"].setdefault(bin_name, {"transition_count": 0, "bracketed_transition_count": 0})
        span = max(case.window_end_seconds - case.window_start_seconds, 1e-6)
        for boundary in (inst.start, inst.end):
            pos = _seconds_to_local_dense(boundary, case.window_start_seconds, span, case.dense_t)
            bucket["transition_count"] += 1
            if _covered_by_any_bracket(pos, brackets):
                bucket["bracketed_transition_count"] += 1


def _empty_totals() -> dict[str, Any]:
    return {
        "transition_count": 0,
        "bracketed_transition_count": 0,
        "missed_transition_count": 0,
        "fully_bracketed_action_count": 0,
        "action_instance_count": 0,
        "covered_dense_positions": 0,
        "dense_positions": 0,
        "false_positive_bracket_count": 0,
        "ambiguous_transition_count": 0,
        "class_misses": {},
        "video_misses": {},
        "short_action_bins": {},
    }


def _round0_brackets(brackets: Sequence[BracketState]) -> list[BracketState]:
    return [bracket for bracket in brackets if int(bracket.round_created) == 0]


def _covered_by_any_bracket(position: int, brackets: Sequence[BracketState]) -> bool:
    return any(int(bracket.left) <= int(position) <= int(bracket.right) for bracket in brackets)


def _bracket_union_covered_positions(brackets: Sequence[BracketState], dense_t: int) -> int:
    covered = set()
    for bracket in brackets:
        left = max(0, int(bracket.left))
        right = min(dense_t - 1, int(bracket.right))
        if right >= left:
            covered.update(range(left, right + 1))
    return len(covered)


def _seconds_to_local_dense(seconds: float, window_start: float, window_duration: float, dense_t: int) -> int:
    if dense_t <= 1:
        return 0
    local = min(max(float(seconds) - float(window_start), 0.0), float(window_duration))
    return int(round(local / max(float(window_duration), 1e-6) * float(dense_t - 1)))


def _dense_to_seconds(position: int, dense_t: int, duration: float) -> float:
    if dense_t <= 1:
        return 0.0
    return float(position) / float(dense_t - 1) * float(duration)


def _safe_fraction(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if int(denominator) > 0 else 0.0


def _stats(values: Sequence[float]) -> dict[str, float | int | None]:
    clean = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not clean:
        return {"count": 0, "min": None, "max": None, "mean": None, "p50": None, "p95": None}
    return {
        "count": len(clean),
        "min": clean[0],
        "max": clean[-1],
        "mean": float(mean(clean)),
        "p50": _percentile(clean, 0.50),
        "p95": _percentile(clean, 0.95),
    }


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    idx = min(max(int(round((len(values) - 1) * q)), 0), len(values) - 1)
    return float(values[idx])


def _short_bin(duration: float) -> str:
    if duration <= 1.0:
        return "short_le_1s"
    if duration <= 2.0:
        return "short_1_2s"
    if duration <= 5.0:
        return "medium_2_5s"
    return "long_gt_5s"


def _short_action_recall(bins: Mapping[str, Mapping[str, int]]) -> dict[str, dict[str, float | int]]:
    order = ("short_le_1s", "short_1_2s", "medium_2_5s", "long_gt_5s")
    out: dict[str, dict[str, float | int]] = {}
    for name in order:
        data = bins.get(name, {"transition_count": 0, "bracketed_transition_count": 0})
        total = int(data.get("transition_count", 0))
        covered = int(data.get("bracketed_transition_count", 0))
        out[name] = {
            "transition_count": total,
            "bracketed_transition_count": covered,
            "recall": _safe_fraction(covered, total),
        }
    return out


def _locked_next_action(
    transition_count: int,
    fallback_used: bool,
    selector_gt_visible: bool,
    real_evidence: bool = False,
) -> str:
    if selector_gt_visible:
        return "LOCKED_SELECTOR_GT_VISIBLE_INVALID"
    if fallback_used:
        return "LOCKED_DIAGNOSTIC_FALLBACK_NOT_REAL_RECALL_EVIDENCE"
    if transition_count <= 0:
        return "LOCKED_ZERO_TRANSITION_NO_REAL_RECALL_EVIDENCE"
    if real_evidence:
        return "LOCKED_REAL_SCOUT_RECALL_BELOW_FORMAL_GATE_REVISE_BRACKET_POLICY_OR_SCOUT"
    return "LOCKED_DIAGNOSTIC_ONLY_REVIEW_REQUIRED"


def _explicit_windows(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    windows = record.get("windows", [])
    if windows is None:
        return []
    if not isinstance(windows, Sequence) or isinstance(windows, (str, bytes)):
        raise ABRValidationError("LOCKED: scout windows must be a sequence")
    for item in windows:
        if not isinstance(item, Mapping):
            raise ABRValidationError("LOCKED: each scout window must be a mapping")
        _assert_no_forbidden_keys_recursive(item)
    return list(windows)


def _sliding_dense_windows(dense_t: int, window_size: int, overlap_ratio: float) -> Iterable[tuple[int, int]]:
    window_size = min(max(int(window_size), 1), int(dense_t))
    overlap_ratio = min(max(float(overlap_ratio), 0.0), 0.95)
    stride = max(int(round(window_size * (1.0 - overlap_ratio))), 1)
    start = 0
    while start < dense_t:
        end = min(start + window_size, dense_t)
        yield start, end
        if end >= dense_t:
            break
        start += stride


def _extract_curve(record: Mapping[str, Any]) -> list[float] | None:
    for key in SCOUT_CURVE_KEYS:
        if key in record:
            value = record[key]
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                raise ABRValidationError(f"LOCKED: {key} must be a numeric sequence")
            curve = [float(v) for v in value]
            if not curve:
                raise ABRValidationError(f"LOCKED: {key} is empty")
            if not all(math.isfinite(v) for v in curve):
                raise ABRValidationError(f"LOCKED: {key} contains non-finite values")
            return curve
    return None


def _scout_source(record: Mapping[str, Any], allow_fallback: bool, fallback_stage: str) -> str:
    explicit = str(record.get("scout_source", "")).strip()
    if explicit:
        return explicit
    for key in SCOUT_CURVE_KEYS:
        if key in record:
            return f"{key}:deploy_visible"
    if allow_fallback:
        return f"diagnostic_fallback:{fallback_stage}"
    return "missing_deploy_visible_scout"


def _assert_no_forbidden_keys_recursive(payload: Any, prefix: str = "") -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            child = str(key) if not prefix else f"{prefix}.{key}"
            if str(key) in FORBIDDEN_SCOUT_KEYS:
                raise ABRValidationError(f"LOCKED: forbidden deploy-time selector/scout key {child}")
            _assert_no_forbidden_keys_recursive(value, child)
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for idx, value in enumerate(payload):
            _assert_no_forbidden_keys_recursive(value, f"{prefix}[{idx}]")


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise ABRValidationError(f"LOCKED: missing input file {path}") from exc
    except json.JSONDecodeError as exc:
        raise ABRValidationError(f"LOCKED: invalid JSON in {path}: {exc}") from exc


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="ABR selector-only/no-detector/no-training first-round bracket recall diagnostic."
    )
    parser.add_argument("--annotation-json", required=True)
    parser.add_argument("--scout-json")
    parser.add_argument("--subset", default="validation")
    parser.add_argument("--out-json")
    parser.add_argument("--max-videos", type=int)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--window-size", type=int, default=0)
    parser.add_argument("--window-overlap-ratio", type=float, default=0.5)
    parser.add_argument("--allow-diagnostic-fallback-scout", action="store_true")
    parser.add_argument("--fallback-stage", default="DIAGNOSTIC_ONLY")
    parser.add_argument("--k0", type=int, default=96)
    parser.add_argument("--k1-cap", type=int, default=160)
    parser.add_argument("--k2-cap", type=int, default=64)
    parser.add_argument("--max-total-k", type=int, default=384)
    parser.add_argument("--max-gap", type=int, default=24)
    parser.add_argument("--min-first-round-bracket-recall", type=float, default=0.95)
    parser.add_argument("--min-first-round-transition-coverage", type=float, default=0.95)
    args = parser.parse_args(argv)

    cfg = ABRConfig(
        k0=args.k0,
        k1_cap=args.k1_cap,
        k2_cap=args.k2_cap,
        max_total_k=args.max_total_k,
        max_gap=args.max_gap,
        target_frame_num=args.max_total_k,
        route_label=ABR_ROUTE_LABEL,
        allow_diagnostic_fallback_scout=bool(args.allow_diagnostic_fallback_scout),
        fallback_stage=str(args.fallback_stage),
    )
    try:
        payload = run_audit(
            annotation_json=Path(args.annotation_json),
            scout_json=Path(args.scout_json) if args.scout_json else None,
            subset=args.subset,
            max_videos=args.max_videos,
            max_windows=args.max_windows,
            window_size=args.window_size,
            window_overlap_ratio=args.window_overlap_ratio,
            allow_diagnostic_fallback_scout=bool(args.allow_diagnostic_fallback_scout),
            fallback_stage=str(args.fallback_stage),
            min_first_round_bracket_recall=float(args.min_first_round_bracket_recall),
            min_first_round_transition_coverage=float(args.min_first_round_transition_coverage),
            abr_config=cfg,
        )
    except Exception as exc:
        payload = {
            "route_label": ABR_ROUTE_LABEL,
            "method": "abr_first_round_bracket_recall_diagnostic",
            "diagnostic_only": True,
            "no_detector": True,
            "no_training": True,
            "selector_gt_visible": False,
            "real_deploy_visible_recall_evidence": False,
            "status": "LOCKED",
            "error": str(exc),
            "allowed_next_action": "LOCKED_FIX_INPUT_ARTIFACTS_OR_SELECTOR_LEAKAGE",
            **CLAIM_LOCKS,
        }
        if args.out_json:
            Path(args.out_json).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    status_locked = (
        payload["allowed_next_action"].startswith("LOCKED")
        or payload["selector_gt_visible"]
        or payload["diagnostic_fallback_used"]
        or payload["transition_count"] <= 0
    )
    payload["status"] = "LOCKED" if status_locked else "PASS_DIAGNOSTIC_ONLY_REAL_SCOUT_RECALL_EVIDENCE"
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if status_locked else 0


if __name__ == "__main__":
    raise SystemExit(main())
