from __future__ import annotations

import math
import re
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable, Mapping, Sequence

from .types import MDL_KNOT_ROUTE_LABEL, KnotLedger
from .validators import validate_knot_ledger


class FormalReadinessLocked(ValueError):
    """Raised when MDL-Knot evidence is not enough for formal train readiness."""


FORBIDDEN_ROUTE_DRIFT = (
    "C3",
    "C3_PRO",
    "C3-PRO",
    "C3 PRO",
    "C3_ORIGINAL",
    "C3_MAINLINE",
    "EVENT_SURPRISE",
    "GLOBALRANK",
    "GLOBAL_RANK",
    "GLOBAL-RANK",
    "GLOBAL RANK",
    "INTERVAL",
    "ORACLE",
    "BVR",
    "ABR",
    "COMBO",
)

FATAL_LOG_PATTERNS = (
    r"\btraceback\b",
    r"\bruntimeerror\b",
    r"\bcuda\s+out\s+of\s+memory\b",
    r"\bout\s+of\s+memory\b",
    r"\boom\b",
    r"\bkilled\b",
    r"\bno\s+space\s+left\b",
    r"\bno\s+gpu\b",
    r"\bno\s+cuda\b",
)

EVAL_OR_CLAIM_PATTERNS = (
    r"\btools/test\.py\b",
    r"\bresult_detection\.json\b",
    r"\bmap(@|\b)",
    r"\beval(?:uate|uation)?\b(?!\s*(?:locked|disabled|off|false|not|no|without))",
    r"\bcheckpoint\b(?!\s*(?:locked|disabled|off|false|not|no|without))",
    r"\bfull[_ -]?train[_ -]?unlocked\s*[:=]\s*true\b",
    r"\bformal[_ -]?full[_ -]?train\b",
    r"\bdeploy(?:ment)?\s+claim\b",
    r"\bpaper\s+claim\b",
    r"\bruntime\s+claim\b",
    r"\bsparse[_ -]?compute[_ -]?claim\s*[:=]\s*true\b",
    r"\bmetric[_ -]?claim\s*[:=]\s*true\b",
)

LOSS_RE = re.compile(
    r"(?<![a-z0-9_])(?:loss|loss_[a-z0-9_]*|cost)(?![a-z0-9_])\s*[:=]?\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)",
    re.IGNORECASE,
)


def _ledger_dict(ledger) -> dict:
    if isinstance(ledger, KnotLedger):
        return ledger.to_dict()
    if isinstance(ledger, Mapping):
        return dict(ledger)
    raise TypeError(f"unsupported ledger type: {type(ledger)!r}")


def _percentile(values: Sequence[float], pct: float) -> float:
    vals = sorted(float(v) for v in values)
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    rank = (len(vals) - 1) * float(pct) / 100.0
    lo = int(rank)
    hi = min(lo + 1, len(vals) - 1)
    weight = rank - lo
    return vals[lo] * (1.0 - weight) + vals[hi] * weight


def _stats(values: Iterable[float]) -> dict:
    vals = [float(v) for v in values]
    if not vals:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "p50": 0.0, "p95": 0.0}
    return {
        "count": len(vals),
        "min": min(vals),
        "max": max(vals),
        "mean": float(mean(vals)),
        "std": float(pstdev(vals)) if len(vals) > 1 else 0.0,
        "p50": _percentile(vals, 50.0),
        "p95": _percentile(vals, 95.0),
    }


def _mask_values(mask) -> list[bool]:
    if mask is None:
        return []
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().tolist()
    elif hasattr(mask, "tolist"):
        mask = mask.tolist()
    return [bool(v) for v in mask]


def _is_metadata_fallback(source: str, provenance: Mapping[str, object]) -> bool:
    policy = str(provenance.get("scout_policy", ""))
    return source == "frame_metadata_scout" and policy == "raw_frame_motion_scout_with_metadata_fallback"


def _is_synthetic(source: str, provenance: Mapping[str, object]) -> bool:
    return bool(provenance.get("synthetic_precheck_only", False)) or "synthetic" in source.lower()


def _has_pattern(patterns: tuple[str, ...], text: str) -> str | None:
    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return pattern
    return None


def _resolve_evidence_path(path_arg: object, evidence_roots: Sequence[Path | str] | None = None) -> Path:
    raw = str(path_arg or "").strip()
    if not raw:
        raise FormalReadinessLocked("validated shortdiag evidence is missing train_log path")
    path = Path(raw)
    if path.is_absolute():
        return path
    roots = [Path(root) for root in (evidence_roots or ()) if root]
    roots.append(Path.cwd())
    for root in roots:
        candidate = (root / path).resolve()
        if candidate.exists():
            return candidate
    return (roots[0] / path).resolve()


def validate_shortdiag_train_log_content(
    train_log: object,
    *,
    evidence_roots: Sequence[Path | str] | None = None,
) -> dict:
    """Revalidate one-epoch MDL-Knot short diagnostic train-log content."""

    log_path = _resolve_evidence_path(train_log, evidence_roots)
    if not log_path.exists():
        raise FormalReadinessLocked(f"missing train log: {log_path}")
    text = log_path.read_text(encoding="utf-8", errors="replace")
    normalized = text.replace(MDL_KNOT_ROUTE_LABEL, "")
    lower = normalized.lower()

    fatal = _has_pattern(FATAL_LOG_PATTERNS, lower)
    if fatal:
        raise FormalReadinessLocked(f"fatal train-log marker: {fatal}")
    if re.search(r"(?<![a-z0-9_])(?:nan|\+?inf|-inf|infinity)(?![a-z0-9_])", lower):
        raise FormalReadinessLocked("non-finite train-log numeric marker")
    route_hits = [token for token in FORBIDDEN_ROUTE_DRIFT if token in normalized.upper()]
    if route_hits:
        raise FormalReadinessLocked(f"route drift tokens in train log: {route_hits}")

    marker_text = lower
    for allowed_phrase in (
        "without evaluation",
        "no evaluation",
        "evaluation locked",
        "evaluation disabled",
        "without checkpoint",
        "no checkpoint",
        "checkpoint locked",
        "checkpoint disabled",
    ):
        marker_text = marker_text.replace(allowed_phrase, "")
    marker = _has_pattern(EVAL_OR_CLAIM_PATTERNS, marker_text)
    if marker:
        raise FormalReadinessLocked(f"evaluation/checkpoint/claim marker in train log: {marker}")

    losses = [float(match.group(1)) for match in LOSS_RE.finditer(text)]
    finite_losses = [value for value in losses if math.isfinite(value)]
    if not finite_losses:
        raise FormalReadinessLocked("train log provided but no finite Loss value was found")
    if len(finite_losses) != len(losses):
        raise FormalReadinessLocked("train log contains non-finite loss values")

    epoch_numbers = [int(value) for value in re.findall(r"epoch\s*\[?(\d+)", lower)]
    if epoch_numbers and max(epoch_numbers) > 1:
        raise FormalReadinessLocked(f"short diagnostic log exceeds one epoch: {max(epoch_numbers)}")

    return {
        "train_log": str(log_path),
        "finite_loss_count": len(finite_losses),
        "loss_min": min(finite_losses),
        "loss_max": max(finite_losses),
        "epoch_max": max(epoch_numbers) if epoch_numbers else 1,
    }


def _guard_coverage(ledger_data: Mapping[str, object]) -> dict:
    roles = [str(role) for role in ledger_data.get("selected_roles", [])]
    islands = list(ledger_data.get("estimated_islands", []))
    bands = list(ledger_data.get("transition_bands", []))
    positions = [int(v) for v in ledger_data.get("selected_positions", [])]

    short_islands = [item for item in islands if bool(item.get("short_risk", False))]
    uncovered_short = 0
    for island in short_islands:
        start, end = int(island["start"]), int(island["end"])
        if not any(start <= pos <= end for pos, role in zip(positions, roles) if role == "short_risk_guard"):
            uncovered_short += 1

    uncovered_transition = 0
    for band in bands:
        start, end = int(band["start"]), int(band["end"])
        if not any(start <= pos <= end for pos, role in zip(positions, roles) if role == "transition_guard"):
            uncovered_transition += 1

    return {
        "short_island_total": len(short_islands),
        "short_island_guard_count": roles.count("short_risk_guard"),
        "short_island_uncovered_count": uncovered_short,
        "transition_band_total": len(bands),
        "transition_guard_count": roles.count("transition_guard"),
        "transition_band_uncovered_count": uncovered_transition,
        "vanilla_mdl_smoothing_risk_measurable": bool(short_islands or bands),
    }


def build_pipeline_diagnostic(
    *,
    ledger,
    sparse_meta: Mapping[str, object],
    masks,
    scout_source: str,
    scout_provenance: Mapping[str, object],
    bridge: str,
    adapter_target_len: int,
) -> dict:
    ledger_data = _ledger_dict(ledger)
    validate_knot_ledger(ledger_data)
    meta = dict(sparse_meta)
    mask_values = _mask_values(masks)
    valid_k = int(ledger_data["valid_k"])
    selected = [int(v) for v in ledger_data["selected_positions"]]
    selected_gaps = [float(right - left) for left, right in zip(selected, selected[1:])]
    meta_selected = [int(v) for v in meta.get("selected_positions", [])]
    meta_frame_prefix = [int(v) for v in meta.get("selected_frame_inds_prefix", [])]
    handoff_audit = dict(meta.get("handoff_audit", {}))
    audit_mode = str(handoff_audit.get("audit_mode", handoff_audit.get("validation_mode", "structural")))
    if audit_mode == "full_dense":
        audit_mode = "full_raw"
    if audit_mode == "selected_only":
        audit_mode = "sampled_raw"
    visible_count = sum(mask_values[:valid_k]) if mask_values else 0
    source = str(scout_source)
    provenance = dict(scout_provenance or {})

    mask_metadata_alignment = {
        "checked": True,
        "selected_positions_match": selected == meta_selected,
        "valid_k_match": int(meta.get("valid_k", -1)) == valid_k,
        "mask_true_count_matches_valid_k": visible_count == valid_k,
        "mask_prefix_visible": mask_values[:valid_k] == [True] * valid_k if mask_values else False,
        "padding_masked_out": all(not flag for flag in mask_values[valid_k:]) if mask_values else False,
        "position_unit_match": meta.get("position_unit") == ledger_data.get("position_unit"),
    }
    mask_metadata_alignment["all_aligned"] = all(mask_metadata_alignment.values())
    frame_handoff_alignment = {
        "checked": True,
        "audit_mode": audit_mode,
        "structural_sparse_handoff_validated": bool(
            handoff_audit.get("structural_sparse_handoff_validated", False)
        ),
        "real_sparse_handoff_validated": bool(handoff_audit.get("selected_inputs_is_gathered", False)),
        "raw_frame_values_compared": bool(handoff_audit.get("raw_frame_values_compared", False)),
        "full_raw_dense_comparison": bool(handoff_audit.get("full_raw_dense_comparison", False)),
        "bounded_raw_sample_comparison": bool(handoff_audit.get("bounded_raw_sample_comparison", False)),
        "formal_raw_handoff_evidence": bool(handoff_audit.get("formal_raw_handoff_evidence", False)),
        "dense_window_materialized_for_audit": bool(handoff_audit.get("dense_window_materialized_for_audit", False)),
        "raw_audit_frame_count": int(handoff_audit.get("raw_audit_frame_count", 0)),
        "selected_len_matches_valid_k": int(handoff_audit.get("selected_len", -1)) == valid_k,
        "dense_len_matches_dense_T": int(handoff_audit.get("dense_len", -1)) == int(ledger_data["dense_T"]),
        "raw_inputs_not_retained": handoff_audit.get("raw_inputs_retained") is False,
        "selected_frame_inds_prefix_present": len(meta_frame_prefix) == valid_k,
        "selected_positions_prefix_match": [
            int(v) for v in handoff_audit.get("selected_positions_prefix", [])
        ] == selected,
        "selected_frame_inds_prefix_match": [
            int(v) for v in handoff_audit.get("selected_frame_inds_prefix", [])
        ] == meta_frame_prefix,
        "detector_frame_inds_len_matches_adapter_target": int(
            handoff_audit.get("detector_frame_inds_len", adapter_target_len)
        )
        == int(adapter_target_len),
        "detector_frame_prefix_is_sparse": handoff_audit.get("detector_frame_inds_prefix_is_sparse") is True,
        "detector_padding_repeats_last_selected": handoff_audit.get("detector_padding_repeats_last_selected") is True,
    }
    common_alignment_keys = (
        "structural_sparse_handoff_validated",
        "selected_len_matches_valid_k",
        "dense_len_matches_dense_T",
        "raw_inputs_not_retained",
        "selected_frame_inds_prefix_present",
        "selected_positions_prefix_match",
        "selected_frame_inds_prefix_match",
        "detector_frame_inds_len_matches_adapter_target",
        "detector_frame_prefix_is_sparse",
        "detector_padding_repeats_last_selected",
    )
    frame_handoff_alignment["all_aligned"] = all(bool(frame_handoff_alignment[key]) for key in common_alignment_keys)

    synthetic_used = _is_synthetic(source, provenance)
    guard = _guard_coverage(ledger_data)
    return {
        "route_label": ledger_data.get("route_label"),
        "video_id": ledger_data.get("video_id"),
        "window_id": int(ledger_data.get("window_id", 0)),
        "scout_source": source,
        "raw_frame_scout_used": bool(provenance.get("uses_raw_frame_probe", False)),
        "metadata_fallback_used": _is_metadata_fallback(source, provenance),
        "metadata_only": bool(provenance.get("metadata_only", False)),
        "synthetic_fallback_used": synthetic_used,
        "synthetic_fallback_rejected": not synthetic_used,
        "valid_k": valid_k,
        "adapter_target_len": int(adapter_target_len),
        "dense_T": int(ledger_data["dense_T"]),
        "max_gap": int(ledger_data["max_gap"]),
        "gap_p95": float(ledger_data["gap_p95"]),
        "selected_gaps": selected_gaps,
        "selected_gap_stats": _stats(selected_gaps),
        "mask_metadata_alignment": mask_metadata_alignment,
        "frame_handoff_alignment": frame_handoff_alignment,
        "short_boundary_risk": guard,
        "fixed_pad_bridge_compute_boundary": {
            "bridge": str(bridge),
            "detector_input_len": int(adapter_target_len),
            "dynamic_valid_k": valid_k,
            "padding_counts_as_valid": False,
            "sparse_compute_claim": False,
            "claim": "fixed_pad preserves Adapter tensor length; it is not sparse-compute evidence",
        },
    }


def summarize_pipeline_diagnostics(diagnostics: Sequence[Mapping[str, object]]) -> dict:
    items = [dict(item) for item in diagnostics]
    raw = sum(1 for item in items if item.get("raw_frame_scout_used") is True)
    metadata = sum(1 for item in items if item.get("metadata_fallback_used") is True)
    synthetic = sum(1 for item in items if item.get("synthetic_fallback_used") is True)
    audit_modes = {}
    raw_audit_frame_counts = []
    for item in items:
        handoff = dict(item.get("frame_handoff_alignment", {}))
        mode = str(handoff.get("audit_mode", "unknown"))
        audit_modes[mode] = int(audit_modes.get(mode, 0)) + 1
        raw_audit_frame_counts.append(float(handoff.get("raw_audit_frame_count", 0)))
    align_failures = [
        item.get("video_id", "unknown")
        for item in items
        if not dict(item.get("mask_metadata_alignment", {})).get("all_aligned", False)
    ]
    handoff_failures = [
        item.get("video_id", "unknown")
        for item in items
        if not dict(item.get("frame_handoff_alignment", {})).get("all_aligned", False)
    ]
    valid_ks = [int(item.get("valid_k", 0)) for item in items]
    short_total = sum(int(dict(item.get("short_boundary_risk", {})).get("short_island_total", 0)) for item in items)
    short_uncovered = sum(
        int(dict(item.get("short_boundary_risk", {})).get("short_island_uncovered_count", 0)) for item in items
    )
    transition_total = sum(
        int(dict(item.get("short_boundary_risk", {})).get("transition_band_total", 0)) for item in items
    )
    transition_uncovered = sum(
        int(dict(item.get("short_boundary_risk", {})).get("transition_band_uncovered_count", 0)) for item in items
    )
    count = len(items)
    selected_gaps = []
    for item in items:
        if item.get("selected_gaps"):
            selected_gaps.extend(float(value) for value in item.get("selected_gaps", []))
            continue
        gap_stats = dict(item.get("selected_gap_stats", {}))
        if gap_stats.get("count", 0):
            selected_gaps.extend([float(gap_stats.get("mean", 0.0))] * int(gap_stats.get("count", 0)))
    return {
        "route_label": MDL_KNOT_ROUTE_LABEL,
        "validation_scope": "real_video_pipeline_diagnostics_required_for_formal_train",
        "synthetic": False,
        "window_count": count,
        "raw_frame_scout_windows": raw,
        "raw_frame_scout_count": raw,
        "metadata_fallback_windows": metadata,
        "metadata_fallback_count": metadata,
        "synthetic_fallback_windows": synthetic,
        "synthetic_fallback_count": synthetic,
        "raw_frame_scout_ratio": raw / count if count else 0.0,
        "metadata_fallback_ratio": metadata / count if count else 0.0,
        "synthetic_fallback_ratio": synthetic / count if count else 0.0,
        "synthetic_fallback_rejected": synthetic == 0 and count > 0,
        "valid_k_distribution": {
            **_stats(valid_ks),
            "unique_count": len(set(valid_ks)),
            "nonconstant": len(set(valid_ks)) > 1,
        },
        "max_gap_distribution": _stats([float(item.get("max_gap", 0)) for item in items]),
        "gap_p95_distribution": _stats([float(item.get("gap_p95", 0.0)) for item in items]),
        "selected_gap_stats": _stats(selected_gaps),
        "mask_metadata_alignment": {
            "checked_windows": count,
            "failure_count": len(align_failures),
            "failed_video_ids": align_failures[:20],
            "all_aligned": count > 0 and not align_failures,
        },
        "mask_meta_alignment_status": "aligned" if count > 0 and not align_failures else "failed",
        "frame_handoff_alignment": {
            "checked_windows": count,
            "failure_count": len(handoff_failures),
            "failed_video_ids": handoff_failures[:20],
            "all_aligned": count > 0 and not handoff_failures,
        },
        "frame_handoff_alignment_status": "aligned" if count > 0 and not handoff_failures else "failed",
        "handoff_audit_modes": audit_modes,
        "full_raw_handoff_evidence_windows": int(audit_modes.get("full_raw", 0)),
        "sampled_raw_handoff_evidence_windows": int(audit_modes.get("sampled_raw", 0)),
        "structural_handoff_evidence_windows": int(audit_modes.get("structural", 0)),
        "raw_audit_frame_count_distribution": _stats(raw_audit_frame_counts),
        "short_boundary_risk_monitoring": {
            "short_island_total": short_total,
            "short_island_uncovered_count": short_uncovered,
            "short_island_guard_coverage_ratio": 1.0 - short_uncovered / short_total if short_total else 1.0,
            "transition_band_total": transition_total,
            "transition_band_uncovered_count": transition_uncovered,
            "transition_guard_coverage_ratio": 1.0 - transition_uncovered / transition_total if transition_total else 1.0,
            "vanilla_mdl_smoothing_risk_measurable": any(
                dict(item.get("short_boundary_risk", {})).get("vanilla_mdl_smoothing_risk_measurable", False)
                for item in items
            ),
        },
        "short_transition_guard_coverage": {
            "placeholder_or_measured": "measured",
            "short_island_uncovered_count": short_uncovered,
            "transition_band_uncovered_count": transition_uncovered,
            "short_guard_coverage_ratio": 1.0 - short_uncovered / short_total if short_total else 1.0,
            "transition_guard_coverage_ratio": 1.0 - transition_uncovered / transition_total if transition_total else 1.0,
        },
        "fixed_pad_bridge_compute_boundary": {
            "bridge": "fixed_pad",
            "sparse_compute_claim": False,
            "statement": "dynamic valid_k is padded to fixed Adapter length; report acquisition evidence only",
        },
        "fixed_pad_sparse_compute_claim_locked": True,
        "route_drift_claim_lock": {
            "locked": True,
            "tokens": [],
            "claim_markers": [],
            "no_mAP": True,
            "no_tools_test": True,
            "no_train": True,
        },
    }


def validate_formal_readiness_evidence(
    summary: Mapping[str, object],
    *,
    evidence_roots: Sequence[Path | str] | None = None,
) -> None:
    if summary.get("route_label") != MDL_KNOT_ROUTE_LABEL:
        raise FormalReadinessLocked("formal readiness route_label mismatch")
    if summary.get("formal_train_unlocked") is not False:
        raise FormalReadinessLocked("formal_train_unlocked must remain false in readiness evidence")
    if summary.get("full_train_unlocked", False) is not False:
        raise FormalReadinessLocked("full_train_unlocked must remain false in readiness evidence")
    locked_actions = summary.get("locked_actions")
    if not isinstance(locked_actions, Mapping):
        raise FormalReadinessLocked("missing locked_actions in formal readiness summary")
    for key in ("remote_sync", "slurm", "training", "evaluation", "tools_test_py"):
        if locked_actions.get(key) is not True:
            raise FormalReadinessLocked(f"formal readiness summary does not keep {key} locked")
    for key in ("synthetic", "no_mAP", "no_tools_test", "no_train"):
        if key in summary and summary.get(key) is not (False if key == "synthetic" else True):
            raise FormalReadinessLocked(f"formal readiness summary has invalid {key} marker")
    source_mode = summary.get("source_mode")
    if source_mode != "real_video_pipeline":
        raise FormalReadinessLocked(f"real-video formal readiness requires source_mode=real_video_pipeline, got {source_mode}")
    if summary.get("dry_run_fixture") is not False:
        raise FormalReadinessLocked("dry-run fixture summary cannot satisfy real-video formal readiness")
    if int(summary.get("fixture_window_count", 0)) != 0:
        raise FormalReadinessLocked("fixture windows cannot satisfy real-video formal readiness")
    if int(summary.get("real_video_reader_window_count", 0)) <= 0:
        raise FormalReadinessLocked("real-video formal readiness requires real video reader windows")
    reader_backend = summary.get("reader_backend")
    if not isinstance(reader_backend, Mapping):
        raise FormalReadinessLocked("missing reader/backend provenance in real-video summary")
    if reader_backend.get("mode") == "fixture":
        raise FormalReadinessLocked("fixture reader backend cannot satisfy real-video formal readiness")
    if reader_backend.get("real_video_reader_required_for_formal_readiness") is not True:
        raise FormalReadinessLocked("reader backend provenance does not require real video reader evidence")
    if not str(summary.get("annotation_path", "")).strip():
        raise FormalReadinessLocked("missing annotation_path in real-video summary")
    video_roots = summary.get("video_root")
    if not isinstance(video_roots, Sequence) or isinstance(video_roots, (str, bytes)) or not video_roots:
        raise FormalReadinessLocked("missing video_root in real-video summary")
    no_claims = summary.get("no_claims")
    if not isinstance(no_claims, Mapping):
        raise FormalReadinessLocked("missing no_claims in formal readiness summary")
    for key in ("mAP", "runtime", "FLOPs", "deploy", "paper", "sparse_compute"):
        if no_claims.get(key) is not True:
            raise FormalReadinessLocked(f"formal readiness summary does not explicitly lock {key} claims")

    diag = summary.get("real_video_pipeline_diagnostics")
    if not isinstance(diag, Mapping):
        raise FormalReadinessLocked("missing real_video_pipeline_diagnostics")
    if diag.get("synthetic", False) is not False:
        raise FormalReadinessLocked("real diagnostics must be synthetic=false")
    if int(diag.get("window_count", 0)) <= 0:
        raise FormalReadinessLocked("real diagnostics must include at least one real pipeline window")
    raw_count = int(diag.get("raw_frame_scout_count", diag.get("raw_frame_scout_windows", 0)))
    if raw_count <= 0:
        raise FormalReadinessLocked("real diagnostics have no raw-frame scout usage")
    synthetic_count = int(diag.get("synthetic_fallback_count", diag.get("synthetic_fallback_windows", 0)))
    if diag.get("synthetic_fallback_rejected") is not True or synthetic_count != 0:
        raise FormalReadinessLocked("formal diagnostics must reject synthetic fallback")
    valid_k = dict(diag.get("valid_k_distribution", {}))
    if valid_k.get("nonconstant") is not True or int(valid_k.get("unique_count", 0)) <= 1:
        raise FormalReadinessLocked("valid_k distribution is not dynamic in real diagnostics")
    if dict(diag.get("mask_metadata_alignment", {})).get("all_aligned") is not True:
        raise FormalReadinessLocked("mask/metadata alignment diagnostics did not pass")
    if diag.get("mask_meta_alignment_status", "aligned") != "aligned":
        raise FormalReadinessLocked("mask/meta alignment status is not aligned")
    if dict(diag.get("frame_handoff_alignment", {})).get("all_aligned") is not True:
        raise FormalReadinessLocked("raw-frame sparse handoff alignment diagnostics did not pass")
    if diag.get("frame_handoff_alignment_status", "aligned") != "aligned":
        raise FormalReadinessLocked("raw-frame sparse handoff alignment status is not aligned")
    if int(diag.get("full_raw_handoff_evidence_windows", 0)) <= 0:
        raise FormalReadinessLocked("formal readiness requires at least one full_raw handoff audit window")
    if int(diag.get("structural_handoff_evidence_windows", 0)) >= int(diag.get("window_count", 0)):
        raise FormalReadinessLocked("structural-only handoff diagnostics cannot unlock formal readiness")
    if "max_gap_distribution" not in diag or "gap_p95_distribution" not in diag:
        raise FormalReadinessLocked("missing max_gap or gap_p95 diagnostic distributions")
    if "selected_gap_stats" in diag:
        selected_gap_stats = dict(diag.get("selected_gap_stats", {}))
        if int(selected_gap_stats.get("count", 0)) <= 0:
            raise FormalReadinessLocked("missing selected gap diagnostics")
    drift_lock = diag.get("route_drift_claim_lock")
    if isinstance(drift_lock, Mapping):
        if drift_lock.get("locked") is not True:
            raise FormalReadinessLocked("route drift/claim lock is not locked")
        drift_tokens = [str(token) for token in drift_lock.get("tokens", [])]
        if drift_tokens:
            raise FormalReadinessLocked(f"route drift tokens in real diagnostics: {drift_tokens}")
        claim_markers = [str(marker) for marker in drift_lock.get("claim_markers", [])]
        if claim_markers:
            raise FormalReadinessLocked(f"claim markers in real diagnostics: {claim_markers}")
        for key in ("no_mAP", "no_tools_test", "no_train"):
            if drift_lock.get(key) is not True:
                raise FormalReadinessLocked(f"route drift/claim lock missing {key}")
    shortdiag = summary.get("shortdiag_evidence")
    if not isinstance(shortdiag, Mapping) or shortdiag.get("validated") is not True:
        raise FormalReadinessLocked("missing validated shortdiag execution evidence")
    if shortdiag.get("formal_train_unlocked") is not False:
        raise FormalReadinessLocked("shortdiag evidence must keep formal_train_unlocked false")
    if shortdiag.get("no_sparse_compute_claim") is not True:
        raise FormalReadinessLocked("shortdiag evidence must keep sparse-compute claims locked")
    for key in ("metric_claim", "sparse_compute_claim", "runtime_claim", "deploy_claim", "paper_claim"):
        if shortdiag.get(key, False) is True:
            raise FormalReadinessLocked(f"shortdiag evidence opens forbidden {key}")
    log_evidence = shortdiag.get("log_evidence")
    if not isinstance(log_evidence, Mapping):
        raise FormalReadinessLocked("validated shortdiag evidence requires real train-log evidence")
    if not str(log_evidence.get("train_log", "")).strip():
        raise FormalReadinessLocked("validated shortdiag evidence is missing train_log path")
    revalidated_log = validate_shortdiag_train_log_content(
        log_evidence.get("train_log"),
        evidence_roots=evidence_roots,
    )
    for key in ("finite_loss_count", "epoch_max"):
        if int(log_evidence.get(key, revalidated_log[key])) != int(revalidated_log[key]):
            raise FormalReadinessLocked(f"shortdiag log_evidence {key} does not match revalidated train log")
    for key in ("loss_min", "loss_max"):
        if key in log_evidence and abs(float(log_evidence[key]) - float(revalidated_log[key])) > 1e-8:
            raise FormalReadinessLocked(f"shortdiag log_evidence {key} does not match revalidated train log")
    if int(revalidated_log.get("finite_loss_count", 0)) <= 0:
        raise FormalReadinessLocked("validated shortdiag evidence has no finite train loss")
    risk = dict(diag.get("short_boundary_risk_monitoring", {}))
    if risk.get("vanilla_mdl_smoothing_risk_measurable") is not True:
        raise FormalReadinessLocked("short-action/boundary smoothing risk is not measurable")
    guard_coverage = diag.get("short_transition_guard_coverage")
    if isinstance(guard_coverage, Mapping):
        coverage_scope = guard_coverage.get("placeholder_or_measured")
        if coverage_scope not in ("placeholder", "measured"):
            raise FormalReadinessLocked("short/transition guard coverage scope is missing")
        for key in ("short_island_uncovered_count", "transition_band_uncovered_count"):
            if key not in guard_coverage:
                raise FormalReadinessLocked(f"short/transition guard coverage missing {key}")
        if int(guard_coverage.get("short_island_uncovered_count", 0)) != 0:
            raise FormalReadinessLocked("short-island guard has uncovered risky islands")
        if int(guard_coverage.get("transition_band_uncovered_count", 0)) != 0:
            raise FormalReadinessLocked("transition guard has uncovered transition bands")
    else:
        if int(risk.get("short_island_uncovered_count", 0)) != 0:
            raise FormalReadinessLocked("short-island guard has uncovered risky islands")
        if int(risk.get("transition_band_uncovered_count", 0)) != 0:
            raise FormalReadinessLocked("transition guard has uncovered transition bands")
    boundary = dict(diag.get("fixed_pad_bridge_compute_boundary", {}))
    if boundary.get("sparse_compute_claim") is not False:
        raise FormalReadinessLocked("fixed_pad bridge must not make sparse-compute claims")
    if diag.get("fixed_pad_sparse_compute_claim_locked", True) is not True:
        raise FormalReadinessLocked("fixed_pad sparse-compute claim lock is open")
