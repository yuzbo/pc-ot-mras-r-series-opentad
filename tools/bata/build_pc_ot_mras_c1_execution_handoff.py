import argparse
import json
from pathlib import Path


SCHEMA_VERSION = "pc_ot_mras_c1_execution_handoff_v0"
READY = "HANDOFF_READY_NOT_EXECUTABLE"
FAILED = "HANDOFF_FAILED"

EXPECTED_UNLOCK_SCHEMA = "pc_ot_mras_c1_unlock_candidate_v0"
EXPECTED_SYNC_PLAN_SCHEMA = "pc_ot_mras_c1_remote_sync_plan_v0"
EXPECTED_SUBMISSION_PLAN_SCHEMA = "pc_ot_mras_c1_reader_disabled_submission_plan_v0"

EXECUTION_UNLOCK_KEYS = (
    "remote_sync_allowed",
    "formal_gate_generation_allowed",
    "slurm_allowed",
    "tools_test_allowed",
    "detector_map_reporting_allowed",
    "metric_claim_allowed",
    "paper_claim_allowed",
    "runtime_flops_claim_allowed",
    "deploy_claim_allowed",
)

HANDOFF_REQUIRED_BEFORE_EXECUTION = (
    "explicit_user_approved_replacement_or_unlock_for_cancelled_pro_major_gate",
    "operator_runs_remote_sync_plan_with_remote_write_enabled",
    "post_sync_remote_sha_verification_saved",
    "remote_precheck_active_manifest_sha256",
    "remote_precheck_resolved_config_sha256",
    "formal_gate_json_r17",
    "formal_gate_json_r18",
    "formal_gate_sha256_r17",
    "formal_gate_sha256_r18",
    "tracker_manifest_unlocked_for_c1_remote_sync",
    "tracker_manifest_unlocked_for_reader_disabled_tools_test",
    "deliberate_sbatch_submission",
    "slurm_job_ids_recorded",
)


class HandoffError(ValueError):
    pass


def _load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HandoffError(f"invalid JSON: {path}: {exc}") from exc


def _execution_unlock_false_payload():
    return {key: False for key in EXECUTION_UNLOCK_KEYS}


def _unlock_offenders(label, payload, keys=EXECUTION_UNLOCK_KEYS):
    unlock = payload.get("execution_unlock") or {}
    offenders = []
    for key in keys:
        if key in unlock and unlock.get(key) is not False:
            offenders.append(f"{label}.execution_unlock.{key}={unlock.get(key)!r}")
    return offenders


def _check(name, passed, detail):
    return {"name": name, "pass": bool(passed), "detail": detail}


def _targets_from_commands(commands):
    return sorted(str(item.get("target")) for item in commands or [])


def _copy_command_count(remote_sync_plan):
    commands = remote_sync_plan.get("commands") or {}
    return len(commands.get("copy") or [])


def _mkdir_command_count(remote_sync_plan):
    commands = remote_sync_plan.get("commands") or {}
    return len(commands.get("mkdir") or [])


def _has_verify_after_sync(remote_sync_plan):
    commands = remote_sync_plan.get("commands") or {}
    verify = commands.get("verify_after_sync")
    return bool(verify)


def _submission_env_ok(command):
    env = command.get("env") or {}
    target = str(command.get("target"))
    return (
        env.get("PRECHECK_ONLY") == "0"
        and env.get("PCOTMRAS_READER_DISABLED_TARGET") == target
        and env.get(f"ALLOW_{target}_READER_DISABLED_EVAL") == "1"
        and env.get("ALLOW_PCOTMRAS_READER_DISABLED_TOOLS_TRAIN") == "0"
        and env.get("ALLOW_PCOTMRAS_READER_DISABLED_RAW_PREDICTION_CACHE") == "0"
        and env.get("ALLOW_PCOTMRAS_READER_DISABLED_RUNTIME_OR_DEPLOY_CLAIM") == "0"
        and command.get("requires_unlock") is True
        and command.get("requires_slurm") is True
        and command.get("writes_remote") is True
    )


def _submission_env_offenders(submission_plan):
    offenders = []
    for command in submission_plan.get("submission_commands") or []:
        if not _submission_env_ok(command):
            offenders.append(str(command.get("target")))
    return offenders


def _stage_payload(name, order, description, required_before, executable_now=False, command_refs=None):
    return {
        "order": order,
        "name": name,
        "description": description,
        "executable_now": bool(executable_now),
        "required_before": list(required_before),
        "command_refs": command_refs or [],
    }


def build_execution_handoff(
    *,
    unlock_candidate,
    remote_sync_plan,
    submission_plan,
    generated_at="UNSET_GENERATED_AT",
):
    sync_commands = remote_sync_plan.get("commands") or {}
    formal_gate_commands = submission_plan.get("formal_gate_commands") or []
    submission_commands = submission_plan.get("submission_commands") or []
    remote_root = remote_sync_plan.get("remote_root")
    submission_remote_root = submission_plan.get("remote_root")

    checks = [
        _check(
            "unlock_candidate_ready_not_executable",
            unlock_candidate.get("schema_version") == EXPECTED_UNLOCK_SCHEMA
            and unlock_candidate.get("status") == "UNLOCK_CANDIDATE_READY_NOT_EXECUTABLE"
            and unlock_candidate.get("pass") is True
            and not _unlock_offenders("unlock_candidate", unlock_candidate)
            and (unlock_candidate.get("diagnostic_scope") or {}).get("reader_disabled_detector_control_ready_to_unlock")
            is True,
            {
                "schema_version": unlock_candidate.get("schema_version"),
                "status": unlock_candidate.get("status"),
                "pass": unlock_candidate.get("pass"),
                "unlock_offenders": _unlock_offenders("unlock_candidate", unlock_candidate),
            },
        ),
        _check(
            "remote_sync_plan_ready_with_commands",
            remote_sync_plan.get("schema_version") == EXPECTED_SYNC_PLAN_SCHEMA
            and remote_sync_plan.get("status") == "PLAN_READY_NOT_EXECUTED"
            and remote_sync_plan.get("pass") is True
            and not _unlock_offenders("remote_sync_plan", remote_sync_plan)
            and _copy_command_count(remote_sync_plan) == int(remote_sync_plan.get("file_count") or -1)
            and _mkdir_command_count(remote_sync_plan) >= 1
            and _has_verify_after_sync(remote_sync_plan),
            {
                "schema_version": remote_sync_plan.get("schema_version"),
                "status": remote_sync_plan.get("status"),
                "file_count": remote_sync_plan.get("file_count"),
                "copy_command_count": _copy_command_count(remote_sync_plan),
                "mkdir_command_count": _mkdir_command_count(remote_sync_plan),
                "has_verify_after_sync": _has_verify_after_sync(remote_sync_plan),
                "unlock_offenders": _unlock_offenders("remote_sync_plan", remote_sync_plan),
            },
        ),
        _check(
            "submission_plan_ready_with_r17_r18_commands",
            submission_plan.get("schema_version") == EXPECTED_SUBMISSION_PLAN_SCHEMA
            and submission_plan.get("status") == "PLAN_READY_NOT_EXECUTED"
            and submission_plan.get("pass") is True
            and not _unlock_offenders("submission_plan", submission_plan)
            and _targets_from_commands(formal_gate_commands) == ["R17", "R18"]
            and _targets_from_commands(submission_commands) == ["R17", "R18"]
            and not _submission_env_offenders(submission_plan),
            {
                "schema_version": submission_plan.get("schema_version"),
                "status": submission_plan.get("status"),
                "formal_gate_targets": _targets_from_commands(formal_gate_commands),
                "submission_targets": _targets_from_commands(submission_commands),
                "submission_env_offenders": _submission_env_offenders(submission_plan),
                "unlock_offenders": _unlock_offenders("submission_plan", submission_plan),
            },
        ),
        _check(
            "remote_roots_consistent",
            bool(remote_root) and remote_root == submission_remote_root,
            {
                "remote_sync_plan_remote_root": remote_root,
                "submission_plan_remote_root": submission_remote_root,
            },
        ),
    ]

    passed = all(item["pass"] for item in checks)
    missing = sorted(
        set(HANDOFF_REQUIRED_BEFORE_EXECUTION)
        | set(unlock_candidate.get("missing_before_execution") or [])
        | set(remote_sync_plan.get("required_before_execution") or [])
        | set(submission_plan.get("required_before_execution") or [])
    )

    stages = [
        _stage_payload(
            "review_replacement_or_user_unlock",
            1,
            "Record the explicit replacement/unlock for the cancelled Pro major gate before any remote write.",
            ["explicit_user_approved_replacement_or_unlock_for_cancelled_pro_major_gate"],
        ),
        _stage_payload(
            "remote_sync_after_unlock",
            2,
            "Run the reviewed native OpenSSH mkdir/scp source sync plan and save the post-sync SHA evidence.",
            [
                "tracker_manifest_unlocked_for_c1_remote_sync",
                "operator_runs_remote_sync_plan_with_remote_write_enabled",
                "post_sync_remote_sha_verification_saved",
            ],
            command_refs=[
                {"name": "mkdir", "count": len(sync_commands.get("mkdir") or [])},
                {"name": "copy", "count": len(sync_commands.get("copy") or [])},
                {"name": "verify_after_sync", "count": 1 if sync_commands.get("verify_after_sync") else 0},
            ],
        ),
        _stage_payload(
            "remote_precheck_after_sync",
            3,
            "Run target-specific precheck to generate active-manifest and resolved-config SHA evidence.",
            [
                "remote_precheck_active_manifest_sha256",
                "remote_precheck_resolved_config_sha256",
            ],
            command_refs=[
                {"target": item.get("target"), "precheck_only": "1", "launcher": item.get("args", [None])[0]}
                for item in submission_commands
            ],
        ),
        _stage_payload(
            "formal_gate_generation_after_precheck",
            4,
            "Generate R17/R18 formal reader-disabled gate JSONs and record their SHA256 values.",
            [
                "formal_gate_json_r17",
                "formal_gate_json_r18",
                "formal_gate_sha256_r17",
                "formal_gate_sha256_r18",
            ],
            command_refs=[
                {"target": item.get("target"), "program": item.get("program"), "requires_unlock": item.get("requires_unlock")}
                for item in formal_gate_commands
            ],
        ),
        _stage_payload(
            "slurm_reader_disabled_eval_submission",
            5,
            "Submit the R17/R18 reader-disabled diagnostic tools/test jobs and record job ids plus expected artifacts.",
            [
                "tracker_manifest_unlocked_for_reader_disabled_tools_test",
                "deliberate_sbatch_submission",
                "slurm_job_ids_recorded",
            ],
            command_refs=[
                {
                    "target": item.get("target"),
                    "program": item.get("program"),
                    "requires_unlock": item.get("requires_unlock"),
                    "expected_artifacts": item.get("expected_artifacts"),
                }
                for item in submission_commands
            ],
        ),
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": str(generated_at),
        "status": READY if passed else FAILED,
        "pass": passed,
        "route": "CTF-BDI/PC-OT-MRAS",
        "purpose": (
            "Build a single non-executable handoff package for the C1 reader-disabled "
            "remote sync, formal-gate, and Slurm diagnostic sequence."
        ),
        "targets": ["R17", "R18"],
        "remote_root": remote_root,
        "checks": checks,
        "failed_checks": [item for item in checks if not item["pass"]],
        "satisfied_conditions": [item["name"] for item in checks if item["pass"]],
        "execution_allowed_now": False,
        "execution_unlock": _execution_unlock_false_payload(),
        "required_before_execution": missing,
        "handoff_stages": stages,
        "source_refs": {
            "unlock_candidate_status": unlock_candidate.get("status"),
            "remote_sync_plan_status": remote_sync_plan.get("status"),
            "submission_plan_status": submission_plan.get("status"),
        },
        "diagnostic_scope": {
            "reader_disabled_detector_control_handoff_ready": passed,
            "detector_map_available": False,
            "same_head_control_available": False,
            "remote_sync_executed": False,
            "formal_gate_generated": False,
            "slurm_submitted": False,
            "tools_test_executed": False,
        },
        "non_unlocks": [
            "remote_sync",
            "formal_gate_generation",
            "slurm_submission",
            "tools_test",
            "result_detection_json_creation",
            "detector_map_reporting",
            "new_full_training_launch",
            "runtime_flops_claim",
            "deploy_claim",
            "metric_claim",
            "paper_claim",
        ],
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a non-executable C1 reader-disabled execution handoff package."
    )
    parser.add_argument("--unlock-candidate", required=True)
    parser.add_argument("--remote-sync-plan", required=True)
    parser.add_argument("--submission-plan", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--generated-at", default="UNSET_GENERATED_AT")
    return parser.parse_args()


def main():
    args = parse_args()
    payload = build_execution_handoff(
        unlock_candidate=_load_json(args.unlock_candidate),
        remote_sync_plan=_load_json(args.remote_sync_plan),
        submission_plan=_load_json(args.submission_plan),
        generated_at=args.generated_at,
    )
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(text, end="")
    raise SystemExit(0 if payload["pass"] else 1)


if __name__ == "__main__":
    main()
