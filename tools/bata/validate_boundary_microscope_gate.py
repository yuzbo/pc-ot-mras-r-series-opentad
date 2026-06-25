from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PRECHECK_ALLOWED_DECISION = "ALLOW_BOUNDARY_MICROSCOPE_PRECHECK_ONLY"
FULL_TRAIN_ALLOWED_DECISION = "ALLOW_BOUNDARY_MICROSCOPE_FULL_TRAIN"
ALLOWED_DECISION = PRECHECK_ALLOWED_DECISION
ALLOWED_ROUTE = "boundary_microscope_acquisition"
ALLOWED_ROUTE_LABEL = "DIVERGENT_INNOVATION_BOUNDARY_MICROSCOPE_DO_NOT_MERGE_WITH_C3"
FULL_TRAIN_USER_OVERRIDE_STATEMENT = "USER_EXPLICITLY_REQUESTED_BOUNDARY_MICROSCOPE_FULL_TRAIN_CANDIDATE_ON_2026-06-24"

PRECHECK_REQUIRED_TRUE_KEYS = ("allow_precheck_only",)

PRECHECK_REQUIRED_FALSE_KEYS = (
    "allow_tools_train",
    "allow_tools_test",
    "allow_remote_sync",
    "allow_slurm",
    "allow_gpu",
    "allow_full_train",
    "allow_raw_prediction",
    "load_from_raw_predictions",
    "save_raw_prediction",
    "test_time_gt_allowed",
    "teacher_allowed",
    "raw_prediction_cache_allowed",
    "uses_gt_at_test",
    "uses_teacher",
    "uses_raw_prediction_cache",
    "metric_claim_allowed",
    "paper_claim_allowed",
)

PRECHECK_FORBIDDEN_TRUE_KEYS = (
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

FULL_TRAIN_REQUIRED_TRUE_KEYS = (
    "allow_tools_train",
    "allow_slurm",
    "allow_gpu",
    "allow_full_train",
)

FULL_TRAIN_REQUIRED_FALSE_KEYS = (
    "allow_precheck_only",
    "allow_tools_test",
    "allow_remote_sync",
    "allow_raw_prediction",
    "load_from_raw_predictions",
    "save_raw_prediction",
    "test_time_gt_allowed",
    "teacher_allowed",
    "raw_prediction_cache_allowed",
    "uses_gt_at_test",
    "uses_teacher",
    "uses_raw_prediction_cache",
    "metric_claim_allowed",
    "paper_claim_allowed",
)

FULL_TRAIN_FORBIDDEN_TRUE_KEYS = (
    "tools_train",
    "tools_test",
    "allow_tools_test",
    "direct_tools_test",
    "remote_sync",
    "allow_remote_sync",
    "sbatch",
    "allow_sbatch",
    "gpu_train",
    "allow_gpu_train",
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
    "user_override_statement",
    "budget",
    "dense_window_size",
)

HARMLESS_METADATA_KEYS = ("note", "review_id", "run_tag")

FORBIDDEN_ATTRIBUTION_TOKENS = (
    "bh-sdc",
    "bhsdc",
    "event surprise",
    "event-surprise",
    "event_surprise",
    "framesurprise",
    "frame-token",
    "frame_token",
    "frame token",
    "frame-token-hybrid",
    "combo",
    "combined_route",
    "c3-pro",
    "c3_rs",
    "pc_ot_mras_prebackbone_frame_selector",
    "pcotmrasprebackboneframeselector",
)

CONFIG_REQUIRED_FALSE_KEYS = (
    "allow_tools_test",
    "allow_remote_sync",
    "allow_slurm",
    "allow_gpu",
    "allow_full_train",
    "allow_raw_prediction",
    "metric_claim_allowed",
    "paper_claim_allowed",
)

BOUNDARY_ENTRYPOINT_GATE_ENVS = dict(
    gate_json_env="OPENTAD_BOUNDARY_MICROSCOPE_ENTRYPOINT_GATE_JSON",
    gate_sha256_env="OPENTAD_BOUNDARY_MICROSCOPE_ENTRYPOINT_GATE_SHA256",
    active_manifest_sha256_env="OPENTAD_BOUNDARY_MICROSCOPE_ACTIVE_MANIFEST_SHA256",
    resolved_config_sha256_env="OPENTAD_BOUNDARY_MICROSCOPE_RESOLVED_CONFIG_SHA256",
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


def _gate_schema(action):
    if action == "precheck":
        return (
            PRECHECK_ALLOWED_DECISION,
            PRECHECK_REQUIRED_TRUE_KEYS,
            PRECHECK_REQUIRED_FALSE_KEYS,
            PRECHECK_FORBIDDEN_TRUE_KEYS,
        )
    if action == "full-train":
        return (
            FULL_TRAIN_ALLOWED_DECISION,
            FULL_TRAIN_REQUIRED_TRUE_KEYS,
            FULL_TRAIN_REQUIRED_FALSE_KEYS,
            FULL_TRAIN_FORBIDDEN_TRUE_KEYS,
        )
    raise ValueError(f"unknown Boundary microscope gate action: {action}")


def _allowed_payload_keys(action):
    _, required_true_keys, required_false_keys, forbidden_true_keys = _gate_schema(action)
    return (
        set(CONTROL_KEYS)
        | set(required_true_keys)
        | set(required_false_keys)
        | set(forbidden_true_keys)
        | set(HARMLESS_METADATA_KEYS)
    )


def _contains_forbidden_attribution(value: Any) -> str | None:
    if isinstance(value, dict):
        for key, item in value.items():
            nested = _contains_forbidden_attribution(str(key))
            if nested is not None:
                return nested
            nested = _contains_forbidden_attribution(item)
            if nested is not None:
                return nested
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            nested = _contains_forbidden_attribution(item)
            if nested is not None:
                return nested
    elif isinstance(value, str):
        normalized = value.lower()
        for token in FORBIDDEN_ATTRIBUTION_TOKENS:
            if token in normalized:
                return token
    return None


def _reject_forbidden_attribution(payload):
    for key, value in payload.items():
        if key == "route_label":
            continue
        token = _contains_forbidden_attribution(value)
        if token is not None:
            raise ValueError(f"Boundary microscope gate attribution drift is forbidden: {key} contains {token}")


def _require_exact(payload, key, expected):
    if payload.get(key) != expected:
        raise ValueError(f"Boundary microscope gate must preserve {key}={expected}: {payload.get(key)}")


def _require_gate_bound_full_train_config(gate):
    if gate.get("allowed_decision") != FULL_TRAIN_ALLOWED_DECISION:
        raise ValueError("Boundary microscope full-train config gate must bind ALLOW_BOUNDARY_MICROSCOPE_FULL_TRAIN")
    if gate.get("allow_precheck_only") is not False:
        raise ValueError("Boundary microscope full-train config gate must keep allow_precheck_only=False")
    if gate.get("allow_tools_train") is not True:
        raise ValueError("Boundary microscope full-train config gate must set allow_tools_train=True")
    if tuple(gate.get("allowed_entrypoints", ())) != ("tools/train.py",):
        raise ValueError("Boundary microscope full-train config gate must allow only tools/train.py")

    context = gate.get("entrypoint_gate_context")
    if not isinstance(context, dict):
        raise ValueError("Boundary microscope full-train config must define entrypoint_gate_context")
    if context.get("required") is not True:
        raise ValueError("Boundary microscope full-train entrypoint gate context must be required")
    for key, expected in BOUNDARY_ENTRYPOINT_GATE_ENVS.items():
        if context.get(key) != expected:
            raise ValueError(f"Boundary microscope full-train entrypoint gate context must set {key}={expected}")
    if tuple(context.get("allowed_decisions", ())) != (FULL_TRAIN_ALLOWED_DECISION,):
        raise ValueError("Boundary microscope full-train entrypoint gate must allow only full-train decision")
    if context.get("strict_payload_validation") is not True:
        raise ValueError("Boundary microscope full-train entrypoint gate must use strict payload validation")

    exact = context.get("required_exact_values", {})
    for key, expected in (
        ("route", ALLOWED_ROUTE),
        ("route_label", ALLOWED_ROUTE_LABEL),
        ("budget", 384),
        ("dense_window_size", 768),
        ("user_override_statement", FULL_TRAIN_USER_OVERRIDE_STATEMENT),
    ):
        if exact.get(key) != expected:
            raise ValueError(f"Boundary microscope full-train entrypoint gate must bind {key}={expected}")

    required_true = set(context.get("required_true_keys", ()))
    for key in FULL_TRAIN_REQUIRED_TRUE_KEYS:
        if key not in required_true:
            raise ValueError(f"Boundary microscope full-train entrypoint gate must require {key}=true")
    required_false = set(context.get("required_false_keys", ()))
    for key in FULL_TRAIN_REQUIRED_FALSE_KEYS:
        if key not in required_false:
            raise ValueError(f"Boundary microscope full-train entrypoint gate must require {key}=false")
    forbidden_true = set(context.get("forbidden_true_keys", ()))
    for key in FULL_TRAIN_FORBIDDEN_TRUE_KEYS:
        if key not in forbidden_true:
            raise ValueError(f"Boundary microscope full-train entrypoint gate must forbid {key}=true")
    if context.get("unknown_key_policy") != "reject_unknown_except_explicit_harmless_metadata":
        raise ValueError("Boundary microscope full-train entrypoint gate must reject unknown payload keys")


def validate_gate_payload(
    payload,
    active_manifest_sha256,
    resolved_config_sha256,
    budget,
    dense_window_size,
    action="precheck",
    run_tag=None,
):
    expected_decision, required_true_keys, required_false_keys, forbidden_true_keys = _gate_schema(action)
    if payload.get("decision") != expected_decision:
        raise ValueError(f"Boundary microscope gate decision is not allowed: {payload.get('decision')}")
    if payload.get("route") != ALLOWED_ROUTE:
        raise ValueError(f"Boundary microscope gate route mismatch: {payload.get('route')}")
    if payload.get("route_label") != ALLOWED_ROUTE_LABEL:
        raise ValueError(f"Boundary microscope gate route_label mismatch: {payload.get('route_label')}")
    _reject_forbidden_attribution(payload)

    expected_manifest = _first_present(
        payload,
        ("active_sha256_manifest_sha256", "expected_active_sha256_manifest_sha256"),
    )
    if not expected_manifest:
        raise ValueError("Boundary microscope gate must bind active_sha256_manifest_sha256")
    if expected_manifest != active_manifest_sha256:
        raise ValueError(
            "Boundary microscope gate active manifest sha256 mismatch: "
            f"expected={expected_manifest} actual={active_manifest_sha256}"
        )

    expected_resolved = _first_present(
        payload,
        ("resolved_config_sha256", "expected_resolved_config_sha256"),
    )
    if not expected_resolved:
        raise ValueError("Boundary microscope gate must bind resolved_config_sha256")
    if expected_resolved != resolved_config_sha256:
        raise ValueError(
            "Boundary microscope gate resolved config sha256 mismatch: "
            f"expected={expected_resolved} actual={resolved_config_sha256}"
        )

    _require_exact(payload, "budget", int(budget))
    _require_exact(payload, "dense_window_size", int(dense_window_size))
    if action == "full-train":
        if not run_tag:
            raise ValueError("Boundary microscope full-train gate requires --run-tag")
        _require_exact(payload, "run_tag", run_tag)
        _require_exact(payload, "user_override_statement", FULL_TRAIN_USER_OVERRIDE_STATEMENT)

    for key in required_true_keys:
        if payload.get(key) is not True:
            raise ValueError(f"Boundary microscope gate must set {key}=true")
    for key in required_false_keys:
        if payload.get(key) is not False:
            raise ValueError(f"Boundary microscope gate must set {key}=false")
    for key in forbidden_true_keys:
        if key in payload and payload[key] is not False:
            raise ValueError(f"Boundary microscope gate must keep {key}=false/absent; got {payload[key]!r}")
    for key in payload:
        if key not in _allowed_payload_keys(action):
            raise ValueError(f"Boundary microscope gate contains unknown or unallowlisted key: {key}")
    return True


def validate_gate_file(
    gate_json,
    gate_sha256,
    active_manifest_sha256,
    resolved_config_sha256,
    budget,
    dense_window_size,
    action="precheck",
    run_tag=None,
):
    gate_path = Path(gate_json)
    if not gate_path.is_file():
        raise ValueError(f"missing Boundary microscope gate JSON: {gate_json}")
    actual_sha256 = sha256_file(gate_path)
    if actual_sha256 != gate_sha256:
        raise ValueError(f"Boundary microscope gate sha256 mismatch: expected={gate_sha256} actual={actual_sha256}")
    payload = json.loads(gate_path.read_text(encoding="utf-8-sig"))
    validate_gate_payload(
        payload,
        active_manifest_sha256=active_manifest_sha256,
        resolved_config_sha256=resolved_config_sha256,
        budget=int(budget),
        dense_window_size=int(dense_window_size),
        action=action,
        run_tag=run_tag,
    )
    return payload


def _load_plain_python_config(config_path, seen=None):
    path = Path(config_path).resolve()
    if seen is None:
        seen = set()
    if path in seen:
        raise ValueError(f"cyclic Boundary microscope config _base_ reference: {path}")
    seen.add(path)
    namespace = {"__file__": str(path), "__name__": "__boundary_microscope_config__"}
    code = compile(path.read_text(encoding="utf-8"), str(path), "exec")
    exec(code, namespace)
    merged = {}
    for base in namespace.get("_base_", ()):
        base_path = Path(base)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        if base_path.suffix == ".py" and base_path.is_file():
            merged.update(_load_plain_python_config(base_path, seen=seen))
    merged.update({key: value for key, value in namespace.items() if not key.startswith("__")})
    return merged


def validate_config_file(config_path):
    path = Path(config_path)
    if not path.is_file():
        raise ValueError(f"missing Boundary microscope config: {config_path}")
    namespace = _load_plain_python_config(path)
    route = namespace.get("route_id")
    route_label = namespace.get("route_label")
    if route != ALLOWED_ROUTE:
        raise ValueError(f"Boundary microscope config route mismatch: {route}")
    if route_label != ALLOWED_ROUTE_LABEL:
        raise ValueError(f"Boundary microscope config route_label mismatch: {route_label}")

    text = path.read_text(encoding="utf-8")
    forbidden = _contains_forbidden_attribution(text)
    if forbidden is not None:
        raise ValueError(f"Boundary microscope config attribution drift is forbidden: {forbidden}")

    gate = namespace.get("boundary_microscope_gate")
    if not isinstance(gate, dict):
        raise ValueError("Boundary microscope config must define boundary_microscope_gate dict")
    if gate.get("route") != ALLOWED_ROUTE:
        raise ValueError(f"Boundary microscope config gate route mismatch: {gate.get('route')}")
    if gate.get("route_label") != ALLOWED_ROUTE_LABEL:
        raise ValueError(f"Boundary microscope config gate route_label mismatch: {gate.get('route_label')}")
    is_full_train_candidate = gate.get("stage") == "full_train_candidate_n16r4"
    if is_full_train_candidate:
        _require_gate_bound_full_train_config(gate)
    else:
        if gate.get("allow_precheck_only") is not True:
            raise ValueError("Boundary microscope config gate must keep allow_precheck_only=True")
        if gate.get("allow_tools_train") is not False:
            raise ValueError("Boundary microscope precheck config gate must keep allow_tools_train=False")
        if tuple(gate.get("allowed_entrypoints", ())) != ():
            raise ValueError("Boundary microscope precheck config gate must keep allowed_entrypoints empty")
    for key in CONFIG_REQUIRED_FALSE_KEYS:
        if gate.get(key) is not False:
            raise ValueError(f"Boundary microscope config gate must keep {key}=False")
    _reject_forbidden_attribution({key: value for key, value in gate.items() if key != "route_label"})

    inference = namespace.get("inference", {})
    if isinstance(inference, dict):
        if inference.get("load_from_raw_predictions") is not False:
            raise ValueError("Boundary microscope config must keep inference.load_from_raw_predictions=False")
        if inference.get("save_raw_prediction") is not False:
            raise ValueError("Boundary microscope config must keep inference.save_raw_prediction=False")

    return {
        "status": "BOUNDARY_MICROSCOPE_CONFIG_GATE_VALIDATION_PASS",
        "config": str(path),
        "config_sha256": sha256_file(path),
        "route": ALLOWED_ROUTE,
        "route_label": ALLOWED_ROUTE_LABEL,
        "stage": gate.get("stage"),
        "allowed_decision": gate.get("allowed_decision", ALLOWED_DECISION),
        "allows_precheck_only": bool(gate.get("allow_precheck_only")),
        "allows_tools_train": bool(gate.get("allow_tools_train")),
        "allows_tools_test": bool(gate.get("allow_tools_test")),
        "allows_remote_sync": bool(gate.get("allow_remote_sync")),
        "allows_slurm": bool(gate.get("allow_slurm")),
        "allows_gpu": bool(gate.get("allow_gpu")),
        "allows_full_train": bool(gate.get("allow_full_train")),
        "allows_raw_prediction": bool(gate.get("allow_raw_prediction")),
        "metric_claim_allowed": bool(gate.get("metric_claim_allowed")),
        "paper_claim_allowed": bool(gate.get("paper_claim_allowed")),
        "requires_entrypoint_gate": bool(gate.get("entrypoint_gate_context", {}).get("required")),
        "future_full_train_requires_separate_decision": True,
        "current_gate_cannot_authorize_remote_sync_or_full_train": True,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate a fail-closed Boundary Microscope Acquisition gate.")
    parser.add_argument("config", nargs="?")
    parser.add_argument("--json", action="store_true", dest="emit_json")
    parser.add_argument("--action", choices=("precheck", "full-train"), default="precheck")
    parser.add_argument("--gate-json")
    parser.add_argument("--gate-sha256")
    parser.add_argument("--active-manifest-sha256")
    parser.add_argument("--resolved-config-sha256")
    parser.add_argument("--run-tag")
    parser.add_argument("--budget", type=int, default=384)
    parser.add_argument("--dense-window-size", type=int, default=768)
    args = parser.parse_args()

    if args.config is not None:
        report = validate_config_file(args.config)
        if args.emit_json:
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        else:
            print(report["status"])
        return

    required = {
        "--gate-json": args.gate_json,
        "--gate-sha256": args.gate_sha256,
        "--active-manifest-sha256": args.active_manifest_sha256,
        "--resolved-config-sha256": args.resolved_config_sha256,
    }
    if args.action == "full-train":
        required["--run-tag"] = args.run_tag
    missing = [flag for flag, value in required.items() if value is None]
    if missing:
        parser.error("the following arguments are required for gate JSON mode: " + ", ".join(missing))

    payload = validate_gate_file(
        gate_json=args.gate_json,
        gate_sha256=args.gate_sha256,
        active_manifest_sha256=args.active_manifest_sha256,
        resolved_config_sha256=args.resolved_config_sha256,
        budget=int(args.budget),
        dense_window_size=int(args.dense_window_size),
        action=args.action,
        run_tag=args.run_tag,
    )
    if args.emit_json:
        status = "BOUNDARY_MICROSCOPE_GATE_VALIDATION_PASS"
        if args.action == "full-train":
            status = "BOUNDARY_MICROSCOPE_FULL_TRAIN_GATE_VALIDATION_PASS"
        print(
            json.dumps(
                {
                    "status": status,
                    "decision": payload["decision"],
                    "route": payload["route"],
                    "route_label": payload["route_label"],
                    "run_tag": payload.get("run_tag"),
                    "allows_tools_train": bool(payload.get("allow_tools_train")),
                    "allows_slurm": bool(payload.get("allow_slurm")),
                    "allows_gpu": bool(payload.get("allow_gpu")),
                    "allows_full_train": bool(payload.get("allow_full_train")),
                    "metric_claim_allowed": bool(payload.get("metric_claim_allowed")),
                    "paper_claim_allowed": bool(payload.get("paper_claim_allowed")),
                    "current_gate_cannot_authorize_remote_sync_or_full_train": args.action != "full-train",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    else:
        if args.action == "full-train":
            print("BOUNDARY_MICROSCOPE_FULL_TRAIN_GATE_VALIDATION_PASS")
        else:
            print("BOUNDARY_MICROSCOPE_GATE_VALIDATION_PASS")


if __name__ == "__main__":
    main()
