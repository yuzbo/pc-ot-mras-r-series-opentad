import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_SCHEMA = "pc_ot_mras_reader_disabled_c1_gate_binding_draft_v0"
FORMAL_SCHEMA = "pc_ot_mras_reader_disabled_c1_formal_gate_v0"
TARGETS = ("R17", "R18")
FORBIDDEN_TRUE_KEYS = (
    "tools_train",
    "raw_prediction_cache",
    "paper_claim",
    "runtime_flops_claim",
    "deploy_claim",
    "dynamic_budget_claim",
)


class GateDraftError(ValueError):
    pass


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GateDraftError(f"invalid JSON draft: {exc}") from exc


def _require(condition, message):
    if not condition:
        raise GateDraftError(message)


def validate_gate_draft(draft):
    _require(draft.get("schema_version") == EXPECTED_SCHEMA, "unexpected draft schema_version")
    _require(draft.get("status") == "DRAFT_ONLY_NOT_EXECUTABLE", "draft status must be DRAFT_ONLY_NOT_EXECUTABLE")
    unlock = draft.get("execution_unlock") or {}
    for key in (
        "remote_sync_allowed",
        "slurm_allowed",
        "tools_test_allowed",
        "detector_map_claim_allowed",
        "metric_claim_allowed",
        "paper_claim_allowed",
    ):
        _require(unlock.get(key) is False, f"draft execution_unlock.{key} must be false")

    remote = draft.get("remote_read_only_evidence") or {}
    _require(remote.get("result_detection_json_found") is False, "draft must not record result_detection.json evidence")

    targets = draft.get("targets") or {}
    _require(set(targets) == set(TARGETS), "draft must contain exactly R17 and R18 targets")
    for target in TARGETS:
        item = targets[target]
        decision = f"ALLOW_{target}_READER_DISABLED_EVAL"
        _require(item.get("decision_if_formalized") == decision, f"{target} decision mismatch")
        _require(item.get("checkpoint"), f"{target} checkpoint path is missing")
        _require(len(str(item.get("checkpoint_sha256", ""))) == 64, f"{target} checkpoint sha256 is invalid")
        _require(item.get("checkpoint_size_bytes", 0) > 0, f"{target} checkpoint size is invalid")
        _require("Training Over" in str(item.get("training_over_evidence", "")), f"{target} lacks Training Over evidence")
        _require(len(str(item.get("local_resolved_config_sha256_draft", ""))) == 64, f"{target} resolved SHA is invalid")
        _require(len(str(item.get("local_active_manifest_sha256_draft", ""))) == 64, f"{target} manifest SHA is invalid")
        required = item.get("required_formal_gate_fields") or {}
        _require(required.get("decision") == decision, f"{target} required decision mismatch")
        _require(required.get("completed_training_evidence") is True, f"{target} must require completed training")
        _require(required.get("reader_disabled_eval") is True, f"{target} must require reader_disabled_eval")
        _require(required.get("same_head_control") is True, f"{target} must require same_head_control")
        _require(required.get("exact_uniform_reader_override") is True, f"{target} must require exact_uniform_reader_override")
        for key in FORBIDDEN_TRUE_KEYS:
            _require(required.get(key) is False, f"{target} required {key} must be false")
    return draft


def build_formal_gate(
    draft,
    target,
    *,
    remote_active_manifest_sha256,
    remote_resolved_config_sha256,
    review_evidence,
    remote_sync_evidence,
    generated_at,
):
    validate_gate_draft(draft)
    target = target.upper()
    _require(target in TARGETS, "target must be R17 or R18")
    for name, value in (
        ("remote_active_manifest_sha256", remote_active_manifest_sha256),
        ("remote_resolved_config_sha256", remote_resolved_config_sha256),
    ):
        _require(len(str(value)) == 64, f"{name} must be a 64-char sha256")
    _require(review_evidence, "review evidence is required before formal gate generation")
    _require(remote_sync_evidence, "remote sync evidence is required before formal gate generation")

    item = draft["targets"][target]
    required = item["required_formal_gate_fields"]
    payload = {
        "schema_version": FORMAL_SCHEMA,
        "generated_at": generated_at,
        "source_draft_schema_version": draft["schema_version"],
        "source_draft_timestamp": draft.get("timestamp"),
        "target": target,
        "decision": required["decision"],
        "route": draft.get("route"),
        "config": item["config"],
        "checkpoint": item["checkpoint"],
        "checkpoint_sha256": item["checkpoint_sha256"],
        "checkpoint_size_bytes": item["checkpoint_size_bytes"],
        "training_over_evidence": item["training_over_evidence"],
        "completed_training_evidence": True,
        "reader_disabled_eval": True,
        "same_head_control": True,
        "exact_uniform_reader_override": True,
        "active_sha256_manifest_sha256": remote_active_manifest_sha256,
        "resolved_config_sha256": remote_resolved_config_sha256,
        "review_evidence": review_evidence,
        "remote_sync_evidence": remote_sync_evidence,
        "tools_train": False,
        "raw_prediction_cache": False,
        "paper_claim": False,
        "runtime_flops_claim": False,
        "deploy_claim": False,
        "dynamic_budget_claim": False,
        "metric_claim_allowed": False,
        "paper_claim_allowed": False,
        "runtime_flops_claim_allowed": False,
        "deploy_claim_allowed": False,
        "notes": (
            "Formal gate permits only the reviewed reader-disabled exact-uniform tools/test.py "
            "diagnostic path. It does not permit training, raw prediction cache, runtime, deploy, "
            "metric, dynamic-budget, or paper claims."
        ),
    }
    return payload


def _write_json(path, payload):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Validate or formalize a PC-OT-MRAS C1 reader-disabled gate draft.")
    parser.add_argument("--draft", required=True, help="Path to the non-executable C1 binding draft JSON.")
    parser.add_argument("--target", choices=TARGETS, default="R17", help="Gate target to validate or formalize.")
    parser.add_argument("--write-gate-json", default=None, help="Output path for a formal executable gate JSON.")
    parser.add_argument("--allow-executable-gate", action="store_true", help="Allow writing a formal gate JSON.")
    parser.add_argument("--remote-active-manifest-sha256", default=None)
    parser.add_argument("--remote-resolved-config-sha256", default=None)
    parser.add_argument("--review-evidence", default=None)
    parser.add_argument("--remote-sync-evidence", default=None)
    parser.add_argument("--generated-at", default="UNSET_GENERATED_AT")
    parser.add_argument("--expect-draft-sha256", default=None, help="Optional SHA256 expected for the draft file.")
    return parser.parse_args()


def main():
    args = parse_args()
    draft_path = Path(args.draft)
    if args.expect_draft_sha256:
        actual = _sha256_file(draft_path)
        if actual != args.expect_draft_sha256:
            raise SystemExit(f"draft sha256 mismatch: expected {args.expect_draft_sha256} got {actual}")

    draft = validate_gate_draft(_load_json(draft_path))
    target = args.target.upper()
    if args.write_gate_json and not args.allow_executable_gate:
        raise SystemExit("--write-gate-json requires --allow-executable-gate")
    if not args.write_gate_json:
        print(
            json.dumps(
                {
                    "status": "DRAFT_VALID_NOT_EXECUTABLE",
                    "target": target,
                    "decision_if_formalized": draft["targets"][target]["decision_if_formalized"],
                    "checkpoint_sha256": draft["targets"][target]["checkpoint_sha256"],
                    "formal_gate_written": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    payload = build_formal_gate(
        draft,
        target,
        remote_active_manifest_sha256=args.remote_active_manifest_sha256,
        remote_resolved_config_sha256=args.remote_resolved_config_sha256,
        review_evidence=args.review_evidence,
        remote_sync_evidence=args.remote_sync_evidence,
        generated_at=args.generated_at,
    )
    output_path = _write_json(args.write_gate_json, payload)
    output_sha256 = _sha256_file(output_path)
    print(
        json.dumps(
            {
                "status": "FORMAL_GATE_WRITTEN",
                "target": target,
                "output": str(output_path),
                "output_sha256": output_sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
