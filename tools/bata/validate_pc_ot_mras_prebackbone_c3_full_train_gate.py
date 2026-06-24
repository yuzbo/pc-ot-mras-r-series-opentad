import argparse
import hashlib
import json
from pathlib import Path


ALLOWED_DECISION = "ALLOW_PC_OT_MRAS_PREBACKBONE_C3_RS_HYBRID_ST_FULL_TRAIN_FIXED50"
ALLOWED_ROUTE = "pc_ot_mras_prebackbone_c3_rs_hybrid_st_original_adatad"
ALLOWED_VARIANT = "C3-RS-Hybrid-ST-OriginalAdaTAD"
ALLOWED_READER = "PCOTMRASRSeriesHybridFrameScout"
ALLOWED_READER_FAMILY = "RSeriesHybrid"
ALLOWED_SELECTOR_GRADIENT = "st_hard_real_frames_with_full_flat_soft_transport_surrogate"
ALLOWED_ROBUST_AUX_OBJECTIVE = "gt_duplicate_value_risk_uncertainty_redundancy_role"

REQUIRED_EXACT = {
    "variant_id": ALLOWED_VARIANT,
    "execution_mode": "train",
    "selection_surface": "pre_backbone_raw_frame",
    "selection_timing": "online_before_backbone",
    "acquisition_unit": "frame",
    "claim_tier": "no_deploy_until_runtime_audit",
    "selection_unit": 1,
    "scout_feature_source": "compressed_pixels",
    "scout_spatial_size": 32,
    "selector_reader": ALLOWED_READER,
    "reader_family": ALLOWED_READER_FAMILY,
    "selector_gradient": ALLOWED_SELECTOR_GRADIENT,
    "robust_aux_objective": ALLOWED_ROBUST_AUX_OBJECTIVE,
    "st_surrogate_mode": "full_flat",
    "scout_pixel_clamp": 5.0,
    "aux_gt_acquisition_loss_weight": 0.05,
    "aux_duplicate_cap_loss_weight": 0.001,
    "aux_duplicate_column_cap": 1.25,
    "aux_value_loss_weight": 0.02,
    "aux_risk_loss_weight": 0.02,
    "aux_uncertainty_loss_weight": 0.01,
    "aux_redundancy_loss_weight": 0.01,
    "aux_role_entropy_loss_weight": 0.001,
    "reader_regularizer_loss_weight": 0.01,
}

REQUIRED_TRUE_KEYS = (
    "allow_remote_sync",
    "allow_precheck_only",
    "allow_slurm",
    "allow_gpu",
    "single_gpu",
    "allow_prebackbone_frame_selector",
    "allow_joint_selector_detector_training",
    "allow_tools_train",
    "allow_dataset_access",
    "allow_pretrained_initialization",
    "allow_checkpoint_write",
    "allow_train_validation_map",
    "allow_long_training",
    "reader_trainable",
    "st_hard_real_frames",
    "train_loop_finite_fail_fast",
    "nan_fail_fast",
    "robust_aux_enabled",
    "scout_pixel_normalize",
)

REQUIRED_FALSE_KEYS = (
    "uses_p2",
    "uses_offline_ledger",
    "uses_teacher",
    "uses_test_gt",
    "uses_raw_prediction_cache",
    "st_off",
    "tools_test",
    "allow_tools_test",
    "direct_tools_test",
    "allow_detector_map",
)

FORBIDDEN_TRUE_KEYS = (
    "tools_test",
    "allow_tools_test",
    "direct_tools_test",
    "detector_map",
    "allow_detector_map",
    "formal_eval",
    "allow_formal_eval",
    "checkpoint_load",
    "allow_checkpoint_load",
    "resume",
    "allow_resume",
    "load_from",
    "allow_load_from",
    "offline_ledger",
    "allow_offline_ledger",
    "frozen_reader_ledger",
    "allow_frozen_reader_ledger",
    "post_projection_bridge",
    "allow_post_projection_bridge",
    "raw_prediction",
    "allow_raw_prediction",
    "raw_predictions",
    "allow_raw_predictions",
    "raw_prediction_cache",
    "allow_raw_prediction_cache",
    "prediction_cache",
    "allow_prediction_cache",
    "load_from_raw_predictions",
    "allow_load_from_raw_predictions",
    "save_raw_prediction",
    "allow_save_raw_prediction",
    "save_raw_predictions",
    "allow_save_raw_predictions",
    "uses_p2",
    "uses_teacher",
    "uses_oracle",
    "uses_test_gt",
    "uses_raw_prediction",
    "st_off",
    "metric_claim",
    "allow_metric_claim",
    "metric_claim_allowed",
    "paper_claim",
    "allow_paper_claim",
    "paper_claim_allowed",
    "runtime_claim",
    "allow_runtime_claim",
    "flops_claim",
    "allow_flops_claim",
    "runtime_flops_claim",
    "allow_runtime_flops_claim",
    "runtime_flops_claim_allowed",
    "deploy_claim",
    "allow_deploy_claim",
    "deploy_claim_allowed",
)

CONTROL_KEYS = (
    "decision",
    "route",
    "active_sha256_manifest_sha256",
    "expected_active_sha256_manifest_sha256",
    "resolved_config_sha256",
    "expected_resolved_config_sha256",
    "pretrained_sha256",
    "budget",
    "dense_window_size",
    "max_epochs",
    "checkpoint_interval",
    "val_start_epoch",
    "val_eval_interval",
    *REQUIRED_EXACT.keys(),
)

HARMLESS_METADATA_KEYS = ("note", "review_id", "run_tag")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_present(payload, keys):
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _allowed_payload_keys():
    return (
        set(CONTROL_KEYS)
        | set(REQUIRED_TRUE_KEYS)
        | set(REQUIRED_FALSE_KEYS)
        | set(FORBIDDEN_TRUE_KEYS)
        | set(HARMLESS_METADATA_KEYS)
    )


def _require_exact(payload, key, expected):
    if payload.get(key) != expected:
        raise ValueError(f"C3 full train gate must preserve {key}={expected}: {payload.get(key)}")


def _require_sha(payload, key, expected):
    value = payload.get(key)
    if not value:
        raise ValueError(f"C3 full train gate must bind {key}")
    if value != expected:
        raise ValueError(f"C3 full train gate {key} mismatch: expected={expected} actual={value}")


def validate_gate_payload(
    payload,
    active_manifest_sha256,
    resolved_config_sha256,
    pretrained_sha256,
    budget,
    dense_window_size,
):
    if payload.get("decision") != ALLOWED_DECISION:
        raise ValueError(f"C3 full train gate decision is not allowed: {payload.get('decision')}")
    if payload.get("route") != ALLOWED_ROUTE:
        raise ValueError(f"C3 full train gate route mismatch: {payload.get('route')}")

    expected_manifest = _first_present(
        payload,
        ("active_sha256_manifest_sha256", "expected_active_sha256_manifest_sha256"),
    )
    if not expected_manifest:
        raise ValueError("C3 full train gate must bind active_sha256_manifest_sha256")
    if expected_manifest != active_manifest_sha256:
        raise ValueError(
            "C3 full train gate active manifest sha256 mismatch: "
            f"expected={expected_manifest} actual={active_manifest_sha256}"
        )

    expected_resolved = _first_present(
        payload,
        ("resolved_config_sha256", "expected_resolved_config_sha256"),
    )
    if not expected_resolved:
        raise ValueError("C3 full train gate must bind resolved_config_sha256")
    if expected_resolved != resolved_config_sha256:
        raise ValueError(
            "C3 full train gate resolved config sha256 mismatch: "
            f"expected={expected_resolved} actual={resolved_config_sha256}"
        )

    _require_sha(payload, "pretrained_sha256", pretrained_sha256)
    _require_exact(payload, "budget", int(budget))
    _require_exact(payload, "dense_window_size", int(dense_window_size))
    _require_exact(payload, "max_epochs", 60)
    _require_exact(payload, "checkpoint_interval", 60)
    _require_exact(payload, "val_start_epoch", 40)
    _require_exact(payload, "val_eval_interval", 2)
    for key, value in REQUIRED_EXACT.items():
        _require_exact(payload, key, value)

    for key in REQUIRED_TRUE_KEYS:
        if payload.get(key) is not True:
            raise ValueError(f"C3 full train gate must set {key}=true")

    for key in REQUIRED_FALSE_KEYS:
        if payload.get(key) is not False:
            raise ValueError(f"C3 full train gate must set {key}=false")

    for key in FORBIDDEN_TRUE_KEYS:
        if key in payload and payload[key] is not False:
            raise ValueError(f"C3 full train gate must keep {key}=false/absent; got {payload[key]!r}")

    for key in payload:
        if key not in _allowed_payload_keys():
            raise ValueError(f"C3 full train gate contains unknown or unallowlisted key: {key}")

    return True


def validate_gate_file(
    gate_json,
    gate_sha256,
    active_manifest_sha256,
    resolved_config_sha256,
    pretrained_sha256,
    budget,
    dense_window_size,
):
    gate_path = Path(gate_json)
    if not gate_path.is_file():
        raise ValueError(f"missing C3 full train gate JSON: {gate_json}")
    actual_sha256 = sha256_file(gate_path)
    if actual_sha256 != gate_sha256:
        raise ValueError(f"C3 full train gate sha256 mismatch: expected={gate_sha256} actual={actual_sha256}")
    payload = json.loads(gate_path.read_text(encoding="utf-8"))
    validate_gate_payload(
        payload,
        active_manifest_sha256=active_manifest_sha256,
        resolved_config_sha256=resolved_config_sha256,
        pretrained_sha256=pretrained_sha256,
        budget=int(budget),
        dense_window_size=int(dense_window_size),
    )
    return payload


def main():
    parser = argparse.ArgumentParser(description="Validate a PC-OT-MRAS C3-RS-Hybrid-ST full-train gate.")
    parser.add_argument("--gate-json", required=True)
    parser.add_argument("--gate-sha256", required=True)
    parser.add_argument("--active-manifest-sha256", required=True)
    parser.add_argument("--resolved-config-sha256", required=True)
    parser.add_argument("--pretrained-sha256", required=True)
    parser.add_argument("--budget", type=int, default=384)
    parser.add_argument("--dense-window-size", type=int, default=768)
    args = parser.parse_args()

    validate_gate_file(
        gate_json=args.gate_json,
        gate_sha256=args.gate_sha256,
        active_manifest_sha256=args.active_manifest_sha256,
        resolved_config_sha256=args.resolved_config_sha256,
        pretrained_sha256=args.pretrained_sha256,
        budget=int(args.budget),
        dense_window_size=int(args.dense_window_size),
    )
    print("PC_OT_MRAS_PREBACKBONE_C3_RS_HYBRID_ST_FULL_TRAIN_GATE_VALIDATION_PASS")


if __name__ == "__main__":
    main()
