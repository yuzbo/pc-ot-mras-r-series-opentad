from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ALLOWED_DECISION = "ALLOW_FRAME_TOKEN_HYBRID_PRECHECK_ONLY"
ALLOWED_ROUTE = "frame_token_hybrid_acquisition"
ALLOWED_ROUTE_LABEL = "DIVERGENT_INNOVATION_FRAME_TOKEN_HYBRID_DO_NOT_MERGE_WITH_C3"

REQUIRED_TRUE_KEYS = ("allow_precheck_only",)

REQUIRED_FALSE_KEYS = (
    "allow_tools_train",
    "allow_tools_test",
    "allow_remote_sync",
    "allow_slurm",
    "allow_gpu",
    "allow_full_train",
    "load_from_raw_predictions",
    "save_raw_prediction",
    "uses_gt_at_test",
    "uses_teacher",
    "uses_oracle",
    "uses_raw_prediction_cache",
    "metric_claim_allowed",
    "paper_claim_allowed",
)

FORBIDDEN_TRUE_KEYS = (
    "tools_train",
    "allow_tools_train",
    "tools_test",
    "allow_tools_test",
    "direct_tools_test",
    "remote_sync",
    "allow_remote_sync",
    "slurm",
    "allow_slurm",
    "sbatch",
    "allow_sbatch",
    "gpu_train",
    "allow_gpu_train",
    "full_train",
    "allow_full_train",
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
    "metric",
    "metric_claim",
    "allow_metric_claim",
    "metric_claim_allowed",
    "paper_claim",
    "allow_paper_claim",
    "paper_claim_allowed",
    "runtime_claim",
    "allow_runtime_claim",
    "deploy_claim",
    "allow_deploy_claim",
    "uses_gt_at_test",
    "uses_test_gt",
    "uses_teacher",
    "uses_oracle",
)

CONTROL_KEYS = (
    "decision",
    "route",
    "route_label",
    "active_sha256_manifest_sha256",
    "expected_active_sha256_manifest_sha256",
    "resolved_config_sha256",
    "expected_resolved_config_sha256",
    "budget",
    "dense_window_size",
    "target_dense_len",
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
        raise ValueError(f"Frame/token hybrid gate must preserve {key}={expected}: {payload.get(key)}")


def validate_gate_payload(
    payload,
    active_manifest_sha256,
    resolved_config_sha256,
    budget,
    dense_window_size,
    target_dense_len,
):
    if payload.get("decision") != ALLOWED_DECISION:
        raise ValueError(f"Frame/token hybrid gate decision is not allowed: {payload.get('decision')}")
    if payload.get("route") != ALLOWED_ROUTE:
        raise ValueError(f"Frame/token hybrid gate route mismatch: {payload.get('route')}")
    if payload.get("route_label") != ALLOWED_ROUTE_LABEL:
        raise ValueError(f"Frame/token hybrid gate route_label mismatch: {payload.get('route_label')}")

    expected_manifest = _first_present(
        payload,
        ("active_sha256_manifest_sha256", "expected_active_sha256_manifest_sha256"),
    )
    if not expected_manifest:
        raise ValueError("Frame/token hybrid gate must bind active_sha256_manifest_sha256")
    if expected_manifest != active_manifest_sha256:
        raise ValueError(
            "Frame/token hybrid gate active manifest sha256 mismatch: "
            f"expected={expected_manifest} actual={active_manifest_sha256}"
        )

    expected_resolved = _first_present(
        payload,
        ("resolved_config_sha256", "expected_resolved_config_sha256"),
    )
    if not expected_resolved:
        raise ValueError("Frame/token hybrid gate must bind resolved_config_sha256")
    if expected_resolved != resolved_config_sha256:
        raise ValueError(
            "Frame/token hybrid gate resolved config sha256 mismatch: "
            f"expected={expected_resolved} actual={resolved_config_sha256}"
        )

    _require_exact(payload, "budget", int(budget))
    _require_exact(payload, "dense_window_size", int(dense_window_size))
    _require_exact(payload, "target_dense_len", int(target_dense_len))

    for key in REQUIRED_TRUE_KEYS:
        if payload.get(key) is not True:
            raise ValueError(f"Frame/token hybrid gate must set {key}=true")
    for key in REQUIRED_FALSE_KEYS:
        if payload.get(key) is not False:
            raise ValueError(f"Frame/token hybrid gate must set {key}=false")
    for key in FORBIDDEN_TRUE_KEYS:
        if key in payload and payload[key] is not False:
            raise ValueError(f"Frame/token hybrid gate must keep {key}=false/absent; got {payload[key]!r}")
    for key in payload:
        if key not in _allowed_payload_keys():
            raise ValueError(f"Frame/token hybrid gate contains unknown or unallowlisted key: {key}")
    return True


def validate_gate_file(
    gate_json,
    gate_sha256,
    active_manifest_sha256,
    resolved_config_sha256,
    budget,
    dense_window_size,
    target_dense_len,
):
    gate_path = Path(gate_json)
    if not gate_path.is_file():
        raise ValueError(f"missing Frame/token hybrid gate JSON: {gate_json}")
    actual_sha256 = sha256_file(gate_path)
    if actual_sha256 != gate_sha256:
        raise ValueError(f"Frame/token hybrid gate sha256 mismatch: expected={gate_sha256} actual={actual_sha256}")
    payload = json.loads(gate_path.read_text(encoding="utf-8-sig"))
    validate_gate_payload(
        payload,
        active_manifest_sha256=active_manifest_sha256,
        resolved_config_sha256=resolved_config_sha256,
        budget=int(budget),
        dense_window_size=int(dense_window_size),
        target_dense_len=int(target_dense_len),
    )
    return payload


def main():
    parser = argparse.ArgumentParser(description="Validate a fail-closed Frame/Token Hybrid Acquisition gate.")
    parser.add_argument("--gate-json", required=True)
    parser.add_argument("--gate-sha256", required=True)
    parser.add_argument("--active-manifest-sha256", required=True)
    parser.add_argument("--resolved-config-sha256", required=True)
    parser.add_argument("--budget", type=int, default=384)
    parser.add_argument("--dense-window-size", type=int, default=768)
    parser.add_argument("--target-dense-len", type=int, default=768)
    args = parser.parse_args()

    validate_gate_file(
        gate_json=args.gate_json,
        gate_sha256=args.gate_sha256,
        active_manifest_sha256=args.active_manifest_sha256,
        resolved_config_sha256=args.resolved_config_sha256,
        budget=int(args.budget),
        dense_window_size=int(args.dense_window_size),
        target_dense_len=int(args.target_dense_len),
    )
    print("FRAME_TOKEN_HYBRID_GATE_VALIDATION_PASS")


if __name__ == "__main__":
    main()
