from __future__ import annotations

import argparse
import hashlib
import json
import runpy
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

CONFIG_REQUIRED_TRUE_KEYS = (
    "allow_precheck_only",
    "requires_deploy_preview_probe_signal",
)

CONFIG_REQUIRED_FALSE_KEYS = REQUIRED_FALSE_KEYS + (
    "actual_decode_saving_in_current_pipeline",
    "raw_decode_saving_claim_allowed",
    "pre_decode_loader_hook_reviewed",
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


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _load_python_config(path):
    config_path = Path(path)
    if not config_path.is_file():
        raise ValueError(f"missing Frame/token hybrid config: {path}")
    namespace = runpy.run_path(str(config_path))
    return namespace


def _config_gate_payload(config_path):
    namespace = _load_python_config(config_path)
    gate = dict(namespace.get("frame_token_hybrid_gate") or {})
    scope = dict(namespace.get("experiment_scope") or {})
    model = dict(namespace.get("model") or {})
    frame_selector = dict(model.get("frame_selector") or {})
    route = namespace.get("route_id", gate.get("route", scope.get("route")))
    route_label = namespace.get("route_label", gate.get("route_label", scope.get("route_label")))
    payload = {
        "decision": gate.get("allowed_decision", ALLOWED_DECISION),
        "route": route,
        "route_label": route_label,
        "stage": gate.get("stage", namespace.get("stage_id", scope.get("stage"))),
        "allowed_entrypoints": _as_list(gate.get("allowed_entrypoints")),
        "forbidden_entrypoints": _as_list(gate.get("forbidden_entrypoints")),
        "allow_precheck_only": gate.get("allow_precheck_only"),
        "allow_tools_train": gate.get("allow_tools_train"),
        "allow_tools_test": gate.get("allow_tools_test"),
        "allow_remote_sync": gate.get("allow_remote_sync"),
        "allow_slurm": gate.get("allow_slurm"),
        "allow_gpu": gate.get("allow_gpu"),
        "allow_full_train": gate.get("allow_full_train"),
        "load_from_raw_predictions": gate.get("load_from_raw_predictions"),
        "save_raw_prediction": gate.get("save_raw_prediction"),
        "uses_gt_at_test": scope.get("test_time_gt_allowed"),
        "uses_teacher": scope.get("teacher_allowed"),
        "uses_oracle": scope.get("oracle_allowed"),
        "uses_raw_prediction_cache": scope.get("raw_prediction_cache_allowed"),
        "metric_claim_allowed": gate.get("metric_claim_allowed"),
        "paper_claim_allowed": gate.get("paper_claim_allowed"),
        "actual_decode_saving_in_current_pipeline": scope.get("actual_decode_saving_in_current_pipeline"),
        "raw_decode_saving_claim_allowed": scope.get("raw_decode_saving_claim_allowed"),
        "pre_decode_loader_hook_reviewed": scope.get("pre_decode_loader_hook_reviewed"),
        "requires_deploy_preview_probe_signal": scope.get("requires_deploy_preview_probe_signal"),
        "frame_selector_type": frame_selector.get("type"),
        "frame_selector_require_preview_signal": frame_selector.get("require_preview_signal"),
        "preview_signal_meta_key": frame_selector.get("preview_signal_meta_key"),
        "preview_positions_meta_key": frame_selector.get("preview_positions_meta_key"),
    }
    validate_config_payload(payload)
    return payload


def validate_config_payload(payload):
    if payload.get("decision") != ALLOWED_DECISION:
        raise ValueError(f"Frame/token hybrid config decision is not allowed: {payload.get('decision')}")
    if payload.get("route") != ALLOWED_ROUTE:
        raise ValueError(f"Frame/token hybrid config route mismatch: {payload.get('route')}")
    if payload.get("route_label") != ALLOWED_ROUTE_LABEL:
        raise ValueError(f"Frame/token hybrid config route_label mismatch: {payload.get('route_label')}")
    if payload.get("frame_selector_type") != "FrameTokenHybridAcquisitionRoute":
        raise ValueError("Frame/token hybrid config must use FrameTokenHybridAcquisitionRoute")
    if payload.get("frame_selector_require_preview_signal") is not True:
        raise ValueError("Frame/token hybrid config must require deploy-visible preview/probe signal")

    for key in CONFIG_REQUIRED_TRUE_KEYS:
        if payload.get(key) is not True:
            raise ValueError(f"Frame/token hybrid config must set {key}=true")
    for key in CONFIG_REQUIRED_FALSE_KEYS:
        if payload.get(key) is not False:
            raise ValueError(f"Frame/token hybrid config must set {key}=false")

    allowed_entrypoints = [str(item) for item in payload.get("allowed_entrypoints", [])]
    if any("train.py" in item or "test.py" in item or item in {"sbatch", "scp", "rsync"} for item in allowed_entrypoints):
        raise ValueError(f"Frame/token hybrid config allowed_entrypoints are not precheck-only: {allowed_entrypoints}")
    if any("precheck" not in item and "validate_frame_token_hybrid_gate.py" not in item for item in allowed_entrypoints):
        raise ValueError(f"Frame/token hybrid config allowed_entrypoints must be empty or precheck-only: {allowed_entrypoints}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Validate a fail-closed Frame/Token Hybrid Acquisition gate.")
    parser.add_argument("config", nargs="?", help="Optional route config to summarize and validate.")
    parser.add_argument("--json", action="store_true", help="Emit a JSON fail-closed config summary.")
    parser.add_argument("--gate-json")
    parser.add_argument("--gate-sha256")
    parser.add_argument("--active-manifest-sha256")
    parser.add_argument("--resolved-config-sha256")
    parser.add_argument("--budget", type=int, default=384)
    parser.add_argument("--dense-window-size", type=int, default=768)
    parser.add_argument("--target-dense-len", type=int, default=768)
    args = parser.parse_args()

    if args.config is not None:
        payload = _config_gate_payload(args.config)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        else:
            print("FRAME_TOKEN_HYBRID_CONFIG_GATE_PRECHECK_ONLY")
        return

    missing = [
        name
        for name in (
            "gate_json",
            "gate_sha256",
            "active_manifest_sha256",
            "resolved_config_sha256",
        )
        if getattr(args, name) is None
    ]
    if missing:
        parser.error("the following arguments are required for gate JSON validation: " + ", ".join(f"--{name.replace('_', '-')}" for name in missing))

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
