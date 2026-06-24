import argparse
import hashlib
import json
from pathlib import Path


ALLOWED_DECISION = "ALLOW_PC_OT_MRAS_FRONTEND_R18_FROZEN_READER_ADATAD_RETRAIN"
ALLOWED_PREDUMP_DECISION = "ALLOW_PC_OT_MRAS_FRONTEND_R18_FROZEN_READER_LEDGER_GENERATION"
ALLOWED_ROUTE = "pc_ot_mras_frontend_original_adatad"

REQUIRED_TRUE_KEYS = (
    "allow_slurm",
    "allow_gpu",
    "single_gpu",
    "allow_reader_dump",
    "allow_hard_position_export",
    "allow_value_transport_ledger",
    "allow_tools_train",
    "allow_dataset_access",
    "allow_pretrained_initialization",
    "allow_checkpoint_write",
    "allow_train_validation_map",
    "allow_long_training",
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
    "uses_gt",
    "uses_teacher",
    "uses_oracle",
    "uses_raw_prediction",
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
    "pc_ot_mras_checkpoint_sha256",
    "pretrained_sha256",
    "train_ledger_sha256",
    "val_ledger_sha256",
    "test_ledger_sha256",
    "budget",
    "dense_window_size",
    "max_epochs",
    "val_start_epoch",
    "val_eval_interval",
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
        | set(FORBIDDEN_TRUE_KEYS)
        | set(HARMLESS_METADATA_KEYS)
    )


def _require_exact(payload, key, expected):
    if payload.get(key) != expected:
        raise ValueError(f"frontend retrain gate must preserve {key}={expected}: {payload.get(key)}")


def _require_sha(payload, key, expected):
    value = payload.get(key)
    if not value:
        raise ValueError(f"frontend retrain gate must bind {key}")
    if value != expected:
        raise ValueError(f"frontend retrain gate {key} mismatch: expected={expected} actual={value}")


def validate_gate_payload(
    payload,
    active_manifest_sha256,
    resolved_config_sha256,
    pc_ot_mras_checkpoint_sha256,
    pretrained_sha256,
    train_ledger_sha256,
    val_ledger_sha256,
    test_ledger_sha256,
    budget,
    dense_window_size,
):
    if payload.get("decision") != ALLOWED_DECISION:
        raise ValueError(f"frontend retrain gate decision is not allowed: {payload.get('decision')}")

    if payload.get("route") != ALLOWED_ROUTE:
        raise ValueError(f"frontend retrain gate route mismatch: {payload.get('route')}")

    expected_manifest = _first_present(
        payload,
        ("active_sha256_manifest_sha256", "expected_active_sha256_manifest_sha256"),
    )
    if not expected_manifest:
        raise ValueError("frontend retrain gate must bind active_sha256_manifest_sha256")
    if expected_manifest != active_manifest_sha256:
        raise ValueError(
            "frontend retrain gate active manifest sha256 mismatch: "
            f"expected={expected_manifest} actual={active_manifest_sha256}"
        )

    expected_resolved = _first_present(
        payload,
        ("resolved_config_sha256", "expected_resolved_config_sha256"),
    )
    if not expected_resolved:
        raise ValueError("frontend retrain gate must bind resolved_config_sha256")
    if expected_resolved != resolved_config_sha256:
        raise ValueError(
            "frontend retrain gate resolved config sha256 mismatch: "
            f"expected={expected_resolved} actual={resolved_config_sha256}"
        )

    _require_sha(payload, "pc_ot_mras_checkpoint_sha256", pc_ot_mras_checkpoint_sha256)
    _require_sha(payload, "pretrained_sha256", pretrained_sha256)
    _require_sha(payload, "train_ledger_sha256", train_ledger_sha256)
    _require_sha(payload, "val_ledger_sha256", val_ledger_sha256)
    _require_sha(payload, "test_ledger_sha256", test_ledger_sha256)
    _require_exact(payload, "budget", int(budget))
    _require_exact(payload, "dense_window_size", int(dense_window_size))
    _require_exact(payload, "max_epochs", 60)
    _require_exact(payload, "val_start_epoch", 40)
    _require_exact(payload, "val_eval_interval", 2)

    for key in REQUIRED_TRUE_KEYS:
        if payload.get(key) is not True:
            raise ValueError(f"frontend retrain gate must set {key}=true")

    for key in FORBIDDEN_TRUE_KEYS:
        if key in payload and payload[key] is not False:
            raise ValueError(f"frontend retrain gate must keep {key}=false/absent; got {payload[key]!r}")

    for key in payload:
        if key not in _allowed_payload_keys():
            raise ValueError(f"frontend retrain gate contains unknown or unallowlisted key: {key}")

    return True


def validate_predump_gate_payload(
    payload,
    active_manifest_sha256,
    resolved_config_sha256,
    pc_ot_mras_checkpoint_sha256,
    pretrained_sha256,
    budget,
    dense_window_size,
):
    if payload.get("decision") != ALLOWED_PREDUMP_DECISION:
        raise ValueError(f"frontend retrain predump gate decision is not allowed: {payload.get('decision')}")

    if payload.get("route") != ALLOWED_ROUTE:
        raise ValueError(f"frontend retrain predump gate route mismatch: {payload.get('route')}")

    expected_manifest = _first_present(
        payload,
        ("active_sha256_manifest_sha256", "expected_active_sha256_manifest_sha256"),
    )
    if not expected_manifest:
        raise ValueError("frontend retrain predump gate must bind active_sha256_manifest_sha256")
    if expected_manifest != active_manifest_sha256:
        raise ValueError(
            "frontend retrain predump gate active manifest sha256 mismatch: "
            f"expected={expected_manifest} actual={active_manifest_sha256}"
        )

    expected_resolved = _first_present(
        payload,
        ("resolved_config_sha256", "expected_resolved_config_sha256"),
    )
    if not expected_resolved:
        raise ValueError("frontend retrain predump gate must bind resolved_config_sha256")
    if expected_resolved != resolved_config_sha256:
        raise ValueError(
            "frontend retrain predump gate resolved config sha256 mismatch: "
            f"expected={expected_resolved} actual={resolved_config_sha256}"
        )

    _require_sha(payload, "pc_ot_mras_checkpoint_sha256", pc_ot_mras_checkpoint_sha256)
    _require_sha(payload, "pretrained_sha256", pretrained_sha256)
    _require_exact(payload, "budget", int(budget))
    _require_exact(payload, "dense_window_size", int(dense_window_size))

    for key in (
        "allow_slurm",
        "allow_gpu",
        "single_gpu",
        "allow_reader_dump",
        "allow_hard_position_export",
        "allow_value_transport_ledger",
        "allow_dataset_access",
    ):
        if payload.get(key) is not True:
            raise ValueError(f"frontend retrain predump gate must set {key}=true")

    for key in FORBIDDEN_TRUE_KEYS + ("allow_tools_train",):
        if key in payload and payload[key] is not False:
            raise ValueError(
                f"frontend retrain predump gate must keep {key}=false/absent; got {payload[key]!r}"
            )

    allowed_keys = (
        {
            "decision",
            "route",
            "active_sha256_manifest_sha256",
            "expected_active_sha256_manifest_sha256",
            "resolved_config_sha256",
            "expected_resolved_config_sha256",
            "pc_ot_mras_checkpoint_sha256",
            "pretrained_sha256",
            "budget",
            "dense_window_size",
            "allow_slurm",
            "allow_gpu",
            "single_gpu",
            "allow_reader_dump",
            "allow_hard_position_export",
            "allow_value_transport_ledger",
            "allow_dataset_access",
            "allow_tools_train",
        }
        | set(FORBIDDEN_TRUE_KEYS)
        | set(HARMLESS_METADATA_KEYS)
    )
    for key in payload:
        if key not in allowed_keys:
            raise ValueError(f"frontend retrain predump gate contains unknown or unallowlisted key: {key}")

    return True


def validate_gate_file(
    gate_json,
    gate_sha256,
    active_manifest_sha256,
    resolved_config_sha256,
    pc_ot_mras_checkpoint_sha256,
    pretrained_sha256,
    train_ledger_sha256,
    val_ledger_sha256,
    test_ledger_sha256,
    budget,
    dense_window_size,
):
    gate_path = Path(gate_json)
    if not gate_path.is_file():
        raise ValueError(f"missing frontend retrain gate JSON: {gate_json}")
    actual_sha256 = sha256_file(gate_path)
    if actual_sha256 != gate_sha256:
        raise ValueError(
            f"frontend retrain gate sha256 mismatch: expected={gate_sha256} actual={actual_sha256}"
        )
    payload = json.loads(gate_path.read_text(encoding="utf-8"))
    validate_gate_payload(
        payload,
        active_manifest_sha256=active_manifest_sha256,
        resolved_config_sha256=resolved_config_sha256,
        pc_ot_mras_checkpoint_sha256=pc_ot_mras_checkpoint_sha256,
        pretrained_sha256=pretrained_sha256,
        train_ledger_sha256=train_ledger_sha256,
        val_ledger_sha256=val_ledger_sha256,
        test_ledger_sha256=test_ledger_sha256,
        budget=int(budget),
        dense_window_size=int(dense_window_size),
    )
    return payload


def validate_predump_gate_file(
    gate_json,
    gate_sha256,
    active_manifest_sha256,
    resolved_config_sha256,
    pc_ot_mras_checkpoint_sha256,
    pretrained_sha256,
    budget,
    dense_window_size,
):
    gate_path = Path(gate_json)
    if not gate_path.is_file():
        raise ValueError(f"missing frontend retrain predump gate JSON: {gate_json}")
    actual_sha256 = sha256_file(gate_path)
    if actual_sha256 != gate_sha256:
        raise ValueError(
            f"frontend retrain predump gate sha256 mismatch: expected={gate_sha256} actual={actual_sha256}"
        )
    payload = json.loads(gate_path.read_text(encoding="utf-8"))
    validate_predump_gate_payload(
        payload,
        active_manifest_sha256=active_manifest_sha256,
        resolved_config_sha256=resolved_config_sha256,
        pc_ot_mras_checkpoint_sha256=pc_ot_mras_checkpoint_sha256,
        pretrained_sha256=pretrained_sha256,
        budget=int(budget),
        dense_window_size=int(dense_window_size),
    )
    return payload


def main():
    parser = argparse.ArgumentParser(description="Validate a frontend R18 frozen-reader AdaTAD retrain gate.")
    parser.add_argument("--mode", choices=("entrypoint", "predump"), default="entrypoint")
    parser.add_argument("--gate-json", required=True)
    parser.add_argument("--gate-sha256", required=True)
    parser.add_argument("--active-manifest-sha256", required=True)
    parser.add_argument("--resolved-config-sha256", required=True)
    parser.add_argument("--pc-ot-mras-checkpoint-sha256", required=True)
    parser.add_argument("--pretrained-sha256", required=True)
    parser.add_argument("--train-ledger-sha256")
    parser.add_argument("--val-ledger-sha256")
    parser.add_argument("--test-ledger-sha256")
    parser.add_argument("--budget", type=int, default=384)
    parser.add_argument("--dense-window-size", type=int, default=768)
    args = parser.parse_args()

    if args.mode == "predump":
        validate_predump_gate_file(
            gate_json=args.gate_json,
            gate_sha256=args.gate_sha256,
            active_manifest_sha256=args.active_manifest_sha256,
            resolved_config_sha256=args.resolved_config_sha256,
            pc_ot_mras_checkpoint_sha256=args.pc_ot_mras_checkpoint_sha256,
            pretrained_sha256=args.pretrained_sha256,
            budget=int(args.budget),
            dense_window_size=int(args.dense_window_size),
        )
        print("PC_OT_MRAS_FRONTEND_RETRAIN_PREDUMP_GATE_VALIDATION_PASS")
        return

    for name in ("train_ledger_sha256", "val_ledger_sha256", "test_ledger_sha256"):
        if getattr(args, name) is None:
            raise SystemExit(f"--{name.replace('_', '-')} is required in entrypoint mode")
    validate_gate_file(
        gate_json=args.gate_json,
        gate_sha256=args.gate_sha256,
        active_manifest_sha256=args.active_manifest_sha256,
        resolved_config_sha256=args.resolved_config_sha256,
        pc_ot_mras_checkpoint_sha256=args.pc_ot_mras_checkpoint_sha256,
        pretrained_sha256=args.pretrained_sha256,
        train_ledger_sha256=args.train_ledger_sha256,
        val_ledger_sha256=args.val_ledger_sha256,
        test_ledger_sha256=args.test_ledger_sha256,
        budget=int(args.budget),
        dense_window_size=int(args.dense_window_size),
    )
    print("PC_OT_MRAS_FRONTEND_RETRAIN_GATE_VALIDATION_PASS")


if __name__ == "__main__":
    main()
