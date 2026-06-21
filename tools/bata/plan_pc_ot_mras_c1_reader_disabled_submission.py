import argparse
import json
from pathlib import Path


SCHEMA_VERSION = "pc_ot_mras_c1_reader_disabled_submission_plan_v0"
EXPECTED_DRAFT_SCHEMA = "pc_ot_mras_reader_disabled_c1_gate_binding_draft_v0"
EXPECTED_SYNC_PLAN_SCHEMA = "pc_ot_mras_c1_remote_sync_plan_v0"
TARGETS = ("R17", "R18")


class SubmissionPlanError(ValueError):
    pass


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _require(condition, message):
    if not condition:
        raise SubmissionPlanError(message)


def _target_env(target, item, *, launcher, gate_json_placeholder, gate_sha_placeholder):
    prefix = target.upper()
    target_lower = prefix.lower()
    return {
        "PCOTMRAS_READER_DISABLED_TARGET": prefix,
        f"ALLOW_{prefix}_READER_DISABLED_EVAL": "1",
        "PRECHECK_ONLY": "0",
        "CONFIG": item["config"],
        f"{prefix}_READER_DISABLED_EVAL_GATE_JSON": gate_json_placeholder,
        f"{prefix}_READER_DISABLED_EVAL_GATE_SHA256": gate_sha_placeholder,
        f"{prefix}_EVAL_CHECKPOINT": item["checkpoint"],
        f"{prefix}_EVAL_CHECKPOINT_SHA256": item["checkpoint_sha256"],
        "RUN_TAG": f"ctf_bdi_pc_ot_mras_{target_lower}_reader_disabled_eval_FORMAL_GATE",
        "ALLOW_PCOTMRAS_READER_DISABLED_TOOLS_TRAIN": "0",
        "ALLOW_PCOTMRAS_READER_DISABLED_RAW_PREDICTION_CACHE": "0",
        "ALLOW_PCOTMRAS_READER_DISABLED_RUNTIME_OR_DEPLOY_CLAIM": "0",
        "SBATCH_SCRIPT": launcher,
    }


def build_submission_plan(
    gate_draft,
    remote_sync_plan,
    *,
    generated_at="UNSET_GENERATED_AT",
    review_evidence_placeholder="REQUIRED_REVIEW_OR_USER_APPROVED_REPLACEMENT",
    remote_sync_evidence_placeholder="REQUIRED_POST_SYNC_SHA_VERIFICATION",
    formal_gate_root="logs/pc_ot_mras_c1_reader_disabled_gates",
):
    _require(gate_draft.get("schema_version") == EXPECTED_DRAFT_SCHEMA, "unexpected gate draft schema")
    _require(gate_draft.get("status") == "DRAFT_ONLY_NOT_EXECUTABLE", "gate draft must remain non-executable")
    draft_unlock = gate_draft.get("execution_unlock") or {}
    for key in ("remote_sync_allowed", "slurm_allowed", "tools_test_allowed", "metric_claim_allowed", "paper_claim_allowed"):
        _require(draft_unlock.get(key) is False, f"gate draft execution_unlock.{key} must be false")

    _require(remote_sync_plan.get("schema_version") == EXPECTED_SYNC_PLAN_SCHEMA, "unexpected remote sync plan schema")
    _require(remote_sync_plan.get("status") == "PLAN_READY_NOT_EXECUTED", "remote sync plan must not be executed")
    sync_unlock = remote_sync_plan.get("execution_unlock") or {}
    for key in ("remote_sync_allowed", "slurm_allowed", "tools_test_allowed", "detector_map_reporting_allowed"):
        _require(sync_unlock.get(key) is False, f"remote sync plan execution_unlock.{key} must be false")

    targets = gate_draft.get("targets") or {}
    _require(set(targets) == set(TARGETS), "gate draft must contain R17 and R18 targets")
    launcher = "scripts/run_ctf_bdi_pc_ot_mras_reader_disabled_eval_n16r4.sbatch"
    remote_root = str(remote_sync_plan.get("remote_root") or gate_draft.get("remote_read_only_evidence", {}).get("remote_root"))
    _require(remote_root, "remote root is required")

    formal_gate_commands = []
    submission_commands = []
    for target in TARGETS:
        item = targets[target]
        target_lower = target.lower()
        formal_gate_json = f"{formal_gate_root}/{target_lower}_reader_disabled_eval_gate.json"
        formal_gate_sha = f"{formal_gate_root}/{target_lower}_reader_disabled_eval_gate.sha256"
        formal_gate_commands.append(
            {
                "target": target,
                "program": "python",
                "args": [
                    "tools/bata/finalize_pc_ot_mras_reader_disabled_gate.py",
                    "--draft",
                    "research-wiki/experiments/CTF_BDI_PC_OT_MRAS_READER_DISABLED_C1_GATE_BINDING_DRAFT_20260622.json",
                    "--target",
                    target,
                    "--allow-executable-gate",
                    "--write-gate-json",
                    formal_gate_json,
                    "--remote-active-manifest-sha256",
                    f"REMOTE_{target}_ACTIVE_MANIFEST_SHA256_AFTER_PRECHECK",
                    "--remote-resolved-config-sha256",
                    f"REMOTE_{target}_RESOLVED_CONFIG_SHA256_AFTER_PRECHECK",
                    "--review-evidence",
                    review_evidence_placeholder,
                    "--remote-sync-evidence",
                    remote_sync_evidence_placeholder,
                    "--generated-at",
                    "GENERATED_AT_AFTER_UNLOCK",
                ],
                "writes_local_gate": True,
                "requires_unlock": True,
            }
        )
        env = _target_env(
            target,
            item,
            launcher=launcher,
            gate_json_placeholder=formal_gate_json,
            gate_sha_placeholder=formal_gate_sha,
        )
        submission_commands.append(
            {
                "target": target,
                "remote_cwd": remote_root,
                "program": "sbatch",
                "env": env,
                "args": [launcher],
                "expected_artifacts": [
                    f"logs/ctf_bdi_pc_ot_mras_{target_lower}_reader_disabled_eval_*/eval_workdir/gpu1_id0/result_detection.json",
                    f"logs/ctf_bdi_pc_ot_mras_{target_lower}_reader_disabled_eval_*/*reader_disabled_eval_summary.json",
                ],
                "writes_remote": True,
                "requires_slurm": True,
                "requires_unlock": True,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "PLAN_READY_NOT_EXECUTED",
        "route": "CTF-BDI/PC-OT-MRAS",
        "purpose": "Plan the post-sync formal-gate and sbatch submission steps for R17/R18 reader-disabled diagnostics.",
        "remote_root": remote_root,
        "source_gate_draft_schema": gate_draft["schema_version"],
        "source_remote_sync_plan_schema": remote_sync_plan["schema_version"],
        "formal_gate_commands": formal_gate_commands,
        "submission_commands": submission_commands,
        "execution_unlock": {
            "formal_gate_generation_allowed": False,
            "remote_sync_allowed": False,
            "slurm_allowed": False,
            "tools_test_allowed": False,
            "detector_map_reporting_allowed": False,
            "metric_claim_allowed": False,
            "paper_claim_allowed": False,
        },
        "required_before_execution": [
            "required_review_or_user_approved_explicit_replacement",
            "remote_sync_plan_executed_and_verified",
            "remote_precheck_generates_active_and_resolved_sha_for_each_target",
            "formal_gate_jsons_generated_and_sha_recorded",
            "tracker_manifest_unlocked_for_reader_disabled_tools_test",
            "operator_deliberately_submits_sbatch_commands",
        ],
        "non_unlocks": [
            "remote_sync",
            "formal_gate_generation",
            "slurm_submission",
            "tools_test",
            "result_detection_json_creation",
            "detector_map_reporting",
            "full_training_launch",
            "runtime_flops_claim",
            "deploy_claim",
            "metric_claim",
            "paper_claim",
        ],
        "pass": True,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Plan, but do not submit, PC-OT-MRAS C1 reader-disabled diagnostics.")
    parser.add_argument("--gate-draft", required=True)
    parser.add_argument("--remote-sync-plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--generated-at", default="UNSET_GENERATED_AT")
    parser.add_argument("--review-evidence-placeholder", default="REQUIRED_REVIEW_OR_USER_APPROVED_REPLACEMENT")
    parser.add_argument("--remote-sync-evidence-placeholder", default="REQUIRED_POST_SYNC_SHA_VERIFICATION")
    parser.add_argument("--formal-gate-root", default="logs/pc_ot_mras_c1_reader_disabled_gates")
    return parser.parse_args()


def main():
    args = parse_args()
    payload = build_submission_plan(
        _load_json(args.gate_draft),
        _load_json(args.remote_sync_plan),
        generated_at=args.generated_at,
        review_evidence_placeholder=args.review_evidence_placeholder,
        remote_sync_evidence_placeholder=args.remote_sync_evidence_placeholder,
        formal_gate_root=args.formal_gate_root,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
