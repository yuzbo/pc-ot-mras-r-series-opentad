import argparse
import hashlib
import json
from pathlib import Path


ALLOWED_DECISION = "PASS_ALLOW_BOUNDED_P2QR_SLURM_SHORT_SMOKE_ONLY"

REQUIRED_TRUE_KEYS = (
    "allow_slurm",
    "allow_gpu",
    "allow_tools_train",
    "single_gpu",
    "allow_dataset_access",
    "allow_pretrained_initialization",
    "disable_checkpoint",
)

FORBIDDEN_TRUE_KEYS = (
    "eval",
    "allow_eval",
    "formal_eval",
    "allow_formal_eval",
    "train_validation_map",
    "allow_train_validation_map",
    "tools_test_map",
    "allow_tools_test_map",
    "tools_test",
    "allow_tools_test",
    "detector_map",
    "allow_detector_map",
    "long_training",
    "allow_long_training",
    "checkpoint_access",
    "allow_checkpoint_access",
    "checkpoint_write",
    "allow_checkpoint_write",
    "checkpoint_load",
    "allow_checkpoint_load",
    "load_checkpoint",
    "allow_load_checkpoint",
    "checkpoint",
    "allow_checkpoint",
    "resume",
    "allow_resume",
    "resume_from",
    "allow_resume_from",
    "load_from",
    "allow_load_from",
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
    "runtime_or_flops_claim_allowed",
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
    "max_epochs",
    "max_train_iters",
)

HARMLESS_METADATA_KEYS = (
    "note",
    "review_id",
)


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
    return set(CONTROL_KEYS) | set(REQUIRED_TRUE_KEYS) | set(FORBIDDEN_TRUE_KEYS) | set(HARMLESS_METADATA_KEYS)


def validate_gate_payload(payload, active_manifest_sha256, resolved_config_sha256):
    """Validate the P2QR bounded execution gate for one two-iteration smoke run."""
    if payload.get("decision") != ALLOWED_DECISION:
        raise ValueError(f"P2QR execution gate decision is not allowed: {payload.get('decision')}")

    if payload.get("route") not in (None, "CTF-BDI/PC-OT-MRAS"):
        raise ValueError(f"P2QR execution gate route mismatch: {payload.get('route')}")

    expected_manifest = _first_present(
        payload,
        ("active_sha256_manifest_sha256", "expected_active_sha256_manifest_sha256"),
    )
    if not expected_manifest:
        raise ValueError("P2QR execution gate must bind active_sha256_manifest_sha256")
    if expected_manifest != active_manifest_sha256:
        raise ValueError(
            "P2QR execution gate active manifest sha256 mismatch: "
            f"expected={expected_manifest} actual={active_manifest_sha256}"
        )

    expected_resolved = _first_present(
        payload,
        ("resolved_config_sha256", "expected_resolved_config_sha256"),
    )
    if not expected_resolved:
        raise ValueError("P2QR execution gate must bind resolved_config_sha256")
    if expected_resolved != resolved_config_sha256:
        raise ValueError(
            "P2QR execution gate resolved config sha256 mismatch: "
            f"expected={expected_resolved} actual={resolved_config_sha256}"
        )

    if payload.get("max_epochs") != 1:
        raise ValueError(f"P2QR execution gate must preserve max_epochs=1: {payload.get('max_epochs')}")
    if payload.get("max_train_iters") != 2:
        raise ValueError(f"P2QR execution gate must preserve max_train_iters=2: {payload.get('max_train_iters')}")

    for key in REQUIRED_TRUE_KEYS:
        if payload.get(key) is not True:
            raise ValueError(f"P2QR execution gate must set {key}=true")

    for key in FORBIDDEN_TRUE_KEYS:
        if key in payload and payload[key] is not False:
            raise ValueError(f"P2QR execution gate must keep {key}=false/absent; got {payload[key]!r}")

    for key in payload:
        if key not in _allowed_payload_keys():
            raise ValueError(f"P2QR execution gate contains unknown or unallowlisted key: {key}")

    return True


def validate_gate_file(gate_json, gate_sha256, active_manifest_sha256, resolved_config_sha256):
    gate_path = Path(gate_json)
    if not gate_path.is_file():
        raise ValueError(f"missing P2QR execution gate JSON: {gate_json}")
    actual_sha256 = sha256_file(gate_path)
    if actual_sha256 != gate_sha256:
        raise ValueError(f"P2QR execution gate sha256 mismatch: expected={gate_sha256} actual={actual_sha256}")
    payload = json.loads(gate_path.read_text(encoding="utf-8"))
    validate_gate_payload(payload, active_manifest_sha256, resolved_config_sha256)
    return payload


def main():
    parser = argparse.ArgumentParser(description="Validate a P2QR bounded short-smoke execution gate JSON.")
    parser.add_argument("--gate-json", required=True)
    parser.add_argument("--gate-sha256", required=True)
    parser.add_argument("--active-manifest-sha256", required=True)
    parser.add_argument("--resolved-config-sha256", required=True)
    args = parser.parse_args()

    validate_gate_file(
        gate_json=args.gate_json,
        gate_sha256=args.gate_sha256,
        active_manifest_sha256=args.active_manifest_sha256,
        resolved_config_sha256=args.resolved_config_sha256,
    )
    print("P2QR_SHORT_SMOKE_EXECUTION_GATE_VALIDATION_PASS")


if __name__ == "__main__":
    main()
