from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


FALSE_FLAG_KEYS = (
    "prediction_uses_gt",
    "uses_gt",
    "uses_gt_annotation",
    "uses_gt_annotations",
    "uses_teacher",
    "uses_oracle",
    "uses_cache",
    "uses_hidden_cache",
    "uses_validation_gt",
    "uses_test_gt",
    "uses_detector_prediction",
    "uses_raw_detector_prediction",
    "uses_prediction_cache",
    "uses_raw_prediction",
    "uses_raw_predictions",
    "uses_detector_outputs",
    "uses_external_labels",
    "uses_supervision",
    "uses_saved_outputs",
    "uses_checkpoint_features",
    "uses_checkpoint_derived_features",
    "uses_result_json",
    "uses_detector_features",
    "uses_precomputed_detector_features",
)

CLAIM_FLAG_KEYS = (
    "deploy_claim_allowed",
    "deployable_claim_allowed",
    "dynamic_budget_claim_allowed",
    "remote_sync_allowed",
    "remote_precheck_allowed",
    "runtime_flops_claim_allowed",
    "runtime_or_flops_claim_allowed",
    "remote_or_training_claim_allowed",
    "slurm_training_allowed",
    "paper_claim_allowed",
    "detector_map_run",
    "detector_map_allowed",
    "detector_metric_claim_allowed",
    "scanner_quality_claim_allowed",
    "deployable_selection_evidence",
)

FORBIDDEN_SOURCE_TOKENS = (
    "gt",
    "ground_truth",
    "truth",
    "annotation_gt",
    "gt_annotation",
    "gt_annotations",
    "validation_gt",
    "test_gt",
    "oracle",
    "pred",
    "prediction",
    "detector_prediction",
    "raw_detector_prediction",
    "teacher",
    "cache",
    "hidden_cache",
    "raw_prediction",
    "raw_predictions",
    "prediction_cache",
    "detector_output",
    "detector_outputs",
    "detector_feature",
    "detector_features",
    "saved_output",
    "saved_outputs",
    "checkpoint",
    "ckpt",
    "checkpoint_derived",
    "checkpoint_feature",
    "checkpoint_features",
    "result_json",
    "result_detection",
    "result_artifact",
    "result_artifacts",
    "result_file",
    "result_files",
    "result_path",
    "result_paths",
    "result_output",
    "result_outputs",
    "result_dump",
    "result_dumps",
    "results_json",
    "results_file",
    "results_files",
    "results_path",
    "results_paths",
    "detection_result",
    "detection_results",
    "detector_result",
    "detector_results",
    "detector_artifact",
    "detector_artifacts",
    "prediction_artifact",
    "prediction_artifacts",
    "prediction_dump",
    "prediction_dumps",
    "proposal_artifact",
    "proposal_artifacts",
    "proposal_dump",
    "proposal_dumps",
    "evidence_artifact",
    "evidence_artifacts",
    "evidence_file",
    "evidence_files",
    "evidence_path",
    "evidence_paths",
    "selection_evidence",
    "saved_artifact",
    "saved_artifacts",
    "saved_result",
    "saved_results",
    "output_artifact",
    "output_artifacts",
    "output_result",
    "output_results",
    "artifact_path",
    "artifact_paths",
    "result_source",
    "map_result",
    "map_results",
    "metric_result",
    "metric_results",
    "precomputed_detector_feature",
    "precomputed_detector_features",
    "diagnostic_target",
    "bca",
    "bce",
)

SOURCE_LIKE_KEYS = (
    "source",
    "origin",
    "provenance",
    "source_role",
    "score_source",
    "candidate_source",
    "score_provenance",
    "process_signal_source",
    "proxy_source",
    "prediction_source",
    "detector_source",
    "raw_prediction_source",
    "cache_source",
)

RECURSIVE_FALSE_FLAG_KEYS = (
    *FALSE_FLAG_KEYS,
    "uses_gt_annotations_for_selection",
    "full_detector_forward_per_round",
    "hidden_cache",
    "hidden_cache_used",
    "validation_gt_used",
    "validation_gt_annotation",
    "validation_gt_annotations",
    "test_gt_used",
    "test_gt_annotation",
    "test_gt_annotations",
    "detector_prediction",
    "detector_predictions",
    "detector_prediction_used",
    "raw_detector_prediction",
    "raw_detector_predictions",
    "raw_detector_prediction_used",
    "prediction_cache",
    "prediction_cache_used",
    "raw_prediction",
    "raw_predictions",
    "raw_prediction_used",
    "gt_annotation",
    "gt_annotations",
    "gt_annotation_used",
    "diagnostic_target",
    "diagnostic_targets",
    "diagnostic_target_used",
)

RECURSIVE_CLAIM_FLAG_KEYS = CLAIM_FLAG_KEYS

FORBIDDEN_KEY_PHRASES = (
    "hidden_cache",
    "validation_gt",
    "test_gt",
    "ground_truth",
    "raw_prediction",
    "raw_pred",
    "raw_detector_prediction",
    "raw_detector_pred",
    "prediction_cache",
    "pred_cache",
    "detector_prediction",
    "detector_pred",
    "annotation_gt",
    "gt_annotation",
    "diagnostic_target",
    "detector_output",
    "detector_outputs",
    "detector_feature",
    "detector_features",
    "saved_output",
    "saved_outputs",
    "checkpoint",
    "ckpt",
    "checkpoint_derived",
    "checkpoint_feature",
    "checkpoint_features",
    "result_json",
    "result_detection",
    "result_artifact",
    "result_artifacts",
    "result_file",
    "result_files",
    "result_path",
    "result_paths",
    "result_output",
    "result_outputs",
    "result_dump",
    "result_dumps",
    "results_json",
    "results_file",
    "results_files",
    "results_path",
    "results_paths",
    "detection_result",
    "detection_results",
    "detector_result",
    "detector_results",
    "detector_artifact",
    "detector_artifacts",
    "prediction_artifact",
    "prediction_artifacts",
    "prediction_dump",
    "prediction_dumps",
    "proposal_artifact",
    "proposal_artifacts",
    "proposal_dump",
    "proposal_dumps",
    "evidence_artifact",
    "evidence_artifacts",
    "evidence_file",
    "evidence_files",
    "evidence_path",
    "evidence_paths",
    "selection_evidence",
    "saved_artifact",
    "saved_artifacts",
    "saved_result",
    "saved_results",
    "output_artifact",
    "output_artifacts",
    "output_result",
    "output_results",
    "artifact_path",
    "artifact_paths",
    "result_source",
    "map_result",
    "map_results",
    "metric_result",
    "metric_results",
    "precomputed_detector_feature",
    "precomputed_detector_features",
)

FORBIDDEN_KEY_TOKENS = (
    "gt",
    "truth",
    "oracle",
    "teacher",
    "cache",
    "prediction",
    "pred",
    "bca",
    "bce",
)

FORBIDDEN_CLAIM_KEY_PHRASES = (
    "allow_detector_map",
    "allow_remote_precheck",
    "allow_remote_sync",
    "allow_sbatch",
    "allow_slurm",
    "allow_sync",
    "deploy_allowed",
    "deploy_claim",
    "deploy_claim_allowed",
    "deployable_claim",
    "deployable_claim_allowed",
    "detector_map",
    "detector_metric_claim",
    "detector_metric_claim_allowed",
    "dynamic_budget_allowed",
    "dynamic_budget_claim_allowed",
    "flops_allowed",
    "paper_claim",
    "paper_allowed",
    "remote_precheck",
    "remote_sync_allowed",
    "remote_sync",
    "remote_or_training_claim",
    "remote_or_training_claim_allowed",
    "remote_precheck_allowed",
    "runtime_allowed",
    "runtime_flops_claim",
    "runtime_flops_claim_allowed",
    "runtime_flops_allowed",
    "runtime_or_flops_claim",
    "runtime_or_flops_claim_allowed",
    "runtime_or_flops_allowed",
    "sbatch_allowed",
    "scanner_quality_claim",
    "scanner_quality_claim_allowed",
    "slurm_training",
    "slurm_training_allowed",
    "slurm_allowed",
    "paper_claim_allowed",
    "detector_map_run",
    "detector_map_allowed",
    "precheck_allowed",
    "sync_allowed",
    "training_allowed",
)

CLAIM_DOMAIN_PHRASES = (
    "deploy",
    "detector_map",
    "detector_metric",
    "dynamic_budget",
    "flops",
    "paper",
    "precheck",
    "remote",
    "runtime",
    "sbatch",
    "scanner_quality",
    "slurm",
    "sync",
    "training",
)

CLAIM_UNLOCK_PHRASES = (
    "allow",
    "allowed",
    "claim",
    "evidence",
    "precheck",
    "run",
    "unlock",
)

EVIDENCE_ARTIFACT_DOMAIN_PHRASES = (
    "checkpoint",
    "ckpt",
    "detection",
    "detections",
    "detector",
    "eval",
    "evaluation",
    "evidence",
    "map",
    "metric",
    "metrics",
    "output",
    "pred",
    "prediction",
    "predictions",
    "proposal",
    "proposals",
    "result",
    "results",
    "saved",
    "selected",
    "selection",
)

EVIDENCE_ARTIFACT_CARRIER_PHRASES = (
    "artifact",
    "artifacts",
    "dir",
    "dirs",
    "directory",
    "directories",
    "dump",
    "dumps",
    "file",
    "files",
    "folder",
    "folders",
    "json",
    "path",
    "paths",
    "result",
    "results",
    "uri",
    "uris",
    "url",
    "urls",
)


def is_explicit_false(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return float(value) == 0.0
    if isinstance(value, str):
        return value.strip().lower() in {"", "0", "false", "no", "none", "null", "n"}
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return not bool(value)


def is_strict_zero_detector_forward_count(value: Any) -> bool:
    if value is False:
        return True
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return int(value) == 0
    if isinstance(value, float):
        return float(value) == 0.0
    if isinstance(value, str):
        return value.strip() == "0"
    return False


def normalize_alias_text(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def normalized_has_phrase_alias(normalized: str, phrases: tuple[str, ...]) -> bool:
    compact = normalized.replace("_", "")
    return any(phrase in normalized or phrase.replace("_", "") in compact for phrase in phrases)


def normalized_matches_alias(normalized: str, aliases: tuple[str, ...]) -> bool:
    compact = normalized.replace("_", "")
    return any(normalized == alias or compact == alias.replace("_", "") for alias in aliases)


def normalized_has_forbidden_evidence_artifact_family(normalized: str) -> bool:
    if not normalized:
        return False
    return normalized_has_phrase_alias(normalized, EVIDENCE_ARTIFACT_DOMAIN_PHRASES) and normalized_has_phrase_alias(
        normalized,
        EVIDENCE_ARTIFACT_CARRIER_PHRASES,
    )


def source_has_forbidden_token(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    normalized = normalize_alias_text(value)
    compact = normalized.replace("_", "")
    if normalized_has_forbidden_evidence_artifact_family(normalized):
        return True
    return any(
        token in text
        or token in normalized
        or token.replace("_", "") in compact
        for token in FORBIDDEN_SOURCE_TOKENS
    )


def key_is_source_like(key: Any) -> bool:
    text = str(key or "").strip().lower()
    if text in SOURCE_LIKE_KEYS:
        return True
    return any(token in text for token in ("source", "provenance", "origin"))


def key_has_forbidden_evidence_alias(key: Any) -> bool:
    normalized = normalize_alias_text(key)
    if not normalized:
        return False
    if normalized_has_phrase_alias(normalized, FORBIDDEN_KEY_PHRASES):
        return True
    if normalized_has_forbidden_evidence_artifact_family(normalized):
        return True
    tokens = set(part for part in normalized.split("_") if part)
    return any(token in tokens for token in FORBIDDEN_KEY_TOKENS)


def key_has_forbidden_claim_alias(key: Any) -> bool:
    normalized = normalize_alias_text(key)
    if not normalized:
        return False
    if normalized_matches_alias(normalized, RECURSIVE_CLAIM_FLAG_KEYS):
        return True
    if normalized_has_phrase_alias(normalized, FORBIDDEN_CLAIM_KEY_PHRASES):
        return True
    if normalized_has_phrase_alias(normalized, CLAIM_DOMAIN_PHRASES) and normalized_has_phrase_alias(
        normalized, CLAIM_UNLOCK_PHRASES
    ):
        return True
    return False


def value_has_forbidden_claim_token(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = normalize_alias_text(value)
    if not normalized:
        return False
    if normalized_matches_alias(normalized, RECURSIVE_CLAIM_FLAG_KEYS):
        return True
    if normalized_has_phrase_alias(normalized, FORBIDDEN_CLAIM_KEY_PHRASES):
        return True
    if normalized_has_phrase_alias(normalized, CLAIM_DOMAIN_PHRASES) and normalized_has_phrase_alias(
        normalized, CLAIM_UNLOCK_PHRASES
    ):
        return True
    return False


def value_indicates_forbidden_alias_presence(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return float(value) != 0.0
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "none", "null", "n"}
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    return bool(value)


def validate_claim_boundaries_all_false(value: Any, *, sample_id: str, location: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{sample_id}: {location} must be an object with explicit false values")
    for key, item in value.items():
        child_location = f"{location}.{str(key or '').strip().lower()}"
        if isinstance(item, Mapping):
            validate_claim_boundaries_all_false(item, sample_id=sample_id, location=child_location)
        elif not is_explicit_false(item):
            raise ValueError(f"{sample_id}: claim/deploy unlock metadata is forbidden at {child_location}")


def validate_no_recursive_contamination(
    value: Any,
    *,
    sample_id: str,
    location: str,
    strict_string_values: bool = False,
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_raw = str(key or "").strip()
            key_text = key_raw.lower()
            key_normalized = normalize_alias_text(key_raw)
            child_location = f"{location}.{key_text}" if key_text else location
            if normalized_matches_alias(key_normalized, ("claim_boundaries",)):
                validate_claim_boundaries_all_false(item, sample_id=sample_id, location=child_location)
            if key_normalized == "detector_forward_count":
                if not is_strict_zero_detector_forward_count(item):
                    raise ValueError(f"{sample_id}: {child_location} must have detector_forward_count=0")
            elif key_normalized == "forward_count":
                raise ValueError(f"{sample_id}: {child_location} is ambiguous; use detector_forward_count=0")
            if normalized_matches_alias(key_normalized, RECURSIVE_FALSE_FLAG_KEYS) and not is_explicit_false(item):
                raise ValueError(f"{sample_id}: forbidden detector-side provenance flag {child_location}")
            if normalized_matches_alias(key_normalized, RECURSIVE_CLAIM_FLAG_KEYS) and not is_explicit_false(item):
                raise ValueError(f"{sample_id}: claim/deploy unlock metadata is forbidden at {child_location}")
            if key_is_source_like(key_text) and isinstance(item, str) and source_has_forbidden_token(item):
                raise ValueError(f"{sample_id}: {child_location} contains forbidden source token")
            if key_has_forbidden_evidence_alias(key_raw) and value_indicates_forbidden_alias_presence(item):
                raise ValueError(f"{sample_id}: forbidden detector-side provenance key {child_location}")
            if key_has_forbidden_claim_alias(key_raw) and value_indicates_forbidden_alias_presence(item):
                raise ValueError(f"{sample_id}: claim/deploy unlock metadata is forbidden at {child_location}")
            validate_no_recursive_contamination(
                item,
                sample_id=sample_id,
                location=child_location,
                strict_string_values=strict_string_values,
            )
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            validate_no_recursive_contamination(
                item,
                sample_id=sample_id,
                location=f"{location}[{index}]",
                strict_string_values=strict_string_values,
            )
        return
    if strict_string_values and isinstance(value, str) and source_has_forbidden_token(value):
        raise ValueError(f"{sample_id}: {location} contains forbidden source token")
    if strict_string_values and isinstance(value, str) and value_has_forbidden_claim_token(value):
        raise ValueError(f"{sample_id}: {location} contains forbidden claim token")
