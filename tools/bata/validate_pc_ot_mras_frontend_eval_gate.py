from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ALLOWED_DECISION = "PASS_ALLOW_PC_OT_MRAS_FRONTEND_ADATAD_EVAL_ONLY"
ALLOWED_ROUTE = "pc_ot_mras_frontend_original_adatad"

REQUIRED_TRUE_KEYS = (
    "allow_slurm",
    "allow_gpu",
    "single_gpu",
    "allow_detector_checkpoint_eval",
    "allow_reader_dump",
    "allow_checkpoint_access",
    "allow_dataset_access",
    "allow_hard_position_export",
    "allow_value_transport_ledger",
    "allow_tools_test",
    "allow_detector_map",
)

FORBIDDEN_TRUE_KEYS = (
    "tools_train",
    "direct_tools_train",
    "allow_tools_train",
    "long_training",
    "allow_long_training",
    "train_validation_map",
    "allow_train_validation_map",
    "raw_prediction_cache",
    "prediction_cache",
    "load_from_raw_predictions",
    "save_raw_prediction",
    "metric_claim",
    "metric_claim_allowed",
    "paper_claim",
    "paper_claim_allowed",
    "runtime_flops_claim",
    "runtime_flops_claim_allowed",
    "deploy_claim",
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
    "adatad_checkpoint_sha256",
    "budget",
    "require_selected_count",
)

HARMLESS_METADATA_KEYS = (
    "note",
    "review_id",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_present(payload: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _allowed_payload_keys() -> set[str]:
    return set(CONTROL_KEYS) | set(REQUIRED_TRUE_KEYS) | set(FORBIDDEN_TRUE_KEYS) | set(HARMLESS_METADATA_KEYS)


def validate_gate_payload(
    payload: Mapping[str, Any],
    *,
    active_manifest_sha256: str,
    resolved_config_sha256: str,
    pc_ot_mras_checkpoint_sha256: str,
    adatad_checkpoint_sha256: str,
    budget: int,
    require_selected_count: int,
) -> bool:
    """Validate the formal hard-frontend -> original AdaTAD eval execution gate."""

    if payload.get("decision") != ALLOWED_DECISION:
        raise ValueError(f"frontend eval gate decision is not allowed: {payload.get('decision')}")
    if payload.get("route") != ALLOWED_ROUTE:
        raise ValueError(f"frontend eval gate route mismatch: {payload.get('route')}")

    expected_manifest = _first_present(
        payload,
        ("active_sha256_manifest_sha256", "expected_active_sha256_manifest_sha256"),
    )
    if not expected_manifest:
        raise ValueError("frontend eval gate must bind active_sha256_manifest_sha256")
    if expected_manifest != active_manifest_sha256:
        raise ValueError(
            "frontend eval gate active manifest sha256 mismatch: "
            f"expected={expected_manifest} actual={active_manifest_sha256}"
        )

    expected_resolved = _first_present(
        payload,
        ("resolved_config_sha256", "expected_resolved_config_sha256"),
    )
    if not expected_resolved:
        raise ValueError("frontend eval gate must bind resolved_config_sha256")
    if expected_resolved != resolved_config_sha256:
        raise ValueError(
            "frontend eval gate resolved config sha256 mismatch: "
            f"expected={expected_resolved} actual={resolved_config_sha256}"
        )

    if payload.get("pc_ot_mras_checkpoint_sha256") != pc_ot_mras_checkpoint_sha256:
        raise ValueError("frontend eval gate PC-OT-MRAS checkpoint sha256 mismatch")
    if payload.get("adatad_checkpoint_sha256") != adatad_checkpoint_sha256:
        raise ValueError("frontend eval gate AdaTAD checkpoint sha256 mismatch")

    if payload.get("budget") != int(budget):
        raise ValueError(f"frontend eval gate must preserve budget={int(budget)}")
    if payload.get("require_selected_count") != int(require_selected_count):
        raise ValueError(
            f"frontend eval gate must preserve require_selected_count={int(require_selected_count)}"
        )

    for key in REQUIRED_TRUE_KEYS:
        if payload.get(key) is not True:
            raise ValueError(f"frontend eval gate must set {key}=true")

    for key in FORBIDDEN_TRUE_KEYS:
        if key in payload and payload[key] is not False:
            raise ValueError(f"frontend eval gate must keep {key}=false/absent; got {payload[key]!r}")

    for key in payload:
        if key not in _allowed_payload_keys():
            raise ValueError(f"frontend eval gate contains unknown or unallowlisted key: {key}")

    return True


def validate_gate_file(
    *,
    gate_json: str | Path,
    gate_sha256: str,
    active_manifest_sha256: str,
    resolved_config_sha256: str,
    pc_ot_mras_checkpoint_sha256: str,
    adatad_checkpoint_sha256: str,
    budget: int,
    require_selected_count: int,
) -> Mapping[str, Any]:
    gate_path = Path(gate_json)
    if not gate_path.is_file():
        raise ValueError(f"missing frontend eval gate JSON: {gate_json}")
    actual_sha256 = sha256_file(gate_path)
    if actual_sha256 != gate_sha256:
        raise ValueError(f"frontend eval gate sha256 mismatch: expected={gate_sha256} actual={actual_sha256}")
    payload = json.loads(gate_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("frontend eval gate JSON must be an object")
    validate_gate_payload(
        payload,
        active_manifest_sha256=active_manifest_sha256,
        resolved_config_sha256=resolved_config_sha256,
        pc_ot_mras_checkpoint_sha256=pc_ot_mras_checkpoint_sha256,
        adatad_checkpoint_sha256=adatad_checkpoint_sha256,
        budget=int(budget),
        require_selected_count=int(require_selected_count),
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a hard-frontend AdaTAD eval execution gate JSON.")
    parser.add_argument("--gate-json", required=True)
    parser.add_argument("--gate-sha256", required=True)
    parser.add_argument("--active-manifest-sha256", required=True)
    parser.add_argument("--resolved-config-sha256", required=True)
    parser.add_argument("--pc-ot-mras-checkpoint-sha256", required=True)
    parser.add_argument("--adatad-checkpoint-sha256", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--require-selected-count", type=int, required=True)
    args = parser.parse_args()

    validate_gate_file(
        gate_json=args.gate_json,
        gate_sha256=args.gate_sha256,
        active_manifest_sha256=args.active_manifest_sha256,
        resolved_config_sha256=args.resolved_config_sha256,
        pc_ot_mras_checkpoint_sha256=args.pc_ot_mras_checkpoint_sha256,
        adatad_checkpoint_sha256=args.adatad_checkpoint_sha256,
        budget=int(args.budget),
        require_selected_count=int(args.require_selected_count),
    )
    print("PC_OT_MRAS_FRONTEND_EVAL_GATE_VALIDATION_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
