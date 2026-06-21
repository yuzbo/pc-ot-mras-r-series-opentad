import argparse
import json
import subprocess
from pathlib import Path


SCHEMA_VERSION = "pc_ot_mras_c1_remote_sync_executor_v0"
DRYRUN_STATUS = "SYNC_EXECUTOR_DRYRUN_NO_REMOTE_WRITE"
BLOCKED_STATUS = "SYNC_EXECUTOR_BLOCKED"
EXECUTED_STATUS = "SYNC_EXECUTOR_EXECUTED"

EXPECTED_SYNC_PLAN_SCHEMA = "pc_ot_mras_c1_remote_sync_plan_v0"
EXPECTED_CHECKLIST_SCHEMA = "pc_ot_mras_c1_unlock_checklist_v0"

REQUIRED_COMPLETED_FOR_REMOTE_WRITE = (
    "explicit_user_approved_replacement_or_unlock_for_cancelled_pro_major_gate",
    "tracker_manifest_unlocked_for_c1_remote_sync",
)

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


class RemoteSyncExecutionError(ValueError):
    pass


def _load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RemoteSyncExecutionError(f"invalid JSON: {path}: {exc}") from exc


def _execution_unlock_false_payload():
    return {key: False for key in EXECUTION_UNLOCK_KEYS}


def _unlock_offenders(label, payload):
    unlock = payload.get("execution_unlock") or {}
    offenders = []
    for key in EXECUTION_UNLOCK_KEYS:
        if key in unlock and unlock.get(key) is not False:
            offenders.append(f"{label}.execution_unlock.{key}={unlock.get(key)!r}")
    return offenders


def _check(name, passed, detail):
    return {"name": name, "pass": bool(passed), "detail": detail}


def _checklist_statuses(checklist):
    return {str(item.get("key")): str(item.get("status")) for item in checklist.get("checklist") or []}


def _completed_checklist_items(checklist):
    statuses = _checklist_statuses(checklist)
    return sorted(key for key, value in statuses.items() if value in {"satisfied", "complete", "completed"})


def _missing_remote_write_prereqs(checklist):
    completed = set(_completed_checklist_items(checklist))
    return [key for key in REQUIRED_COMPLETED_FOR_REMOTE_WRITE if key not in completed]


def _command_groups(remote_sync_plan):
    commands = remote_sync_plan.get("commands") or {}
    mkdir_commands = list(commands.get("mkdir") or [])
    copy_commands = list(commands.get("copy") or [])
    verify_command = commands.get("verify_after_sync") or {}
    return mkdir_commands, copy_commands, verify_command


def _command_summary(remote_sync_plan):
    mkdir_commands, copy_commands, verify_command = _command_groups(remote_sync_plan)
    return {
        "mkdir_count": len(mkdir_commands),
        "copy_count": len(copy_commands),
        "has_verify_after_sync": bool(verify_command),
        "file_count": remote_sync_plan.get("file_count"),
    }


def _plan_commands_valid(remote_sync_plan):
    mkdir_commands, copy_commands, verify_command = _command_groups(remote_sync_plan)
    file_count = int(remote_sync_plan.get("file_count") or -1)
    writes = [item.get("writes_remote") for item in mkdir_commands + copy_commands]
    return (
        file_count > 0
        and len(copy_commands) == file_count
        and len(mkdir_commands) >= 1
        and bool(verify_command)
        and all(value is True for value in writes)
        and verify_command.get("writes_remote") is False
    )


def _run_command(command, runner):
    result = runner(
        [command["program"]] + list(command.get("args") or []),
        input=command.get("stdin"),
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "program": command["program"],
        "purpose": command.get("purpose"),
        "returncode": int(result.returncode),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _execute_plan_commands(remote_sync_plan, runner):
    mkdir_commands, copy_commands, verify_command = _command_groups(remote_sync_plan)
    records = []
    for command in mkdir_commands + copy_commands + [verify_command]:
        record = _run_command(command, runner)
        records.append(record)
        if record["returncode"] != 0:
            break
    return records


def build_remote_sync_execution(
    *,
    remote_sync_plan,
    unlock_checklist,
    allow_remote_write=False,
    user_unlock_evidence=None,
    generated_at="UNSET_GENERATED_AT",
    runner=subprocess.run,
):
    evidence_path = Path(user_unlock_evidence) if user_unlock_evidence else None
    missing_remote_write_prereqs = _missing_remote_write_prereqs(unlock_checklist)
    checks = [
        _check(
            "remote_sync_plan_ready_not_executed",
            remote_sync_plan.get("schema_version") == EXPECTED_SYNC_PLAN_SCHEMA
            and remote_sync_plan.get("status") == "PLAN_READY_NOT_EXECUTED"
            and remote_sync_plan.get("pass") is True
            and not _unlock_offenders("remote_sync_plan", remote_sync_plan),
            {
                "schema_version": remote_sync_plan.get("schema_version"),
                "status": remote_sync_plan.get("status"),
                "pass": remote_sync_plan.get("pass"),
                "unlock_offenders": _unlock_offenders("remote_sync_plan", remote_sync_plan),
            },
        ),
        _check(
            "unlock_checklist_ready_no_execution",
            unlock_checklist.get("schema_version") == EXPECTED_CHECKLIST_SCHEMA
            and unlock_checklist.get("status") == "UNLOCK_CHECKLIST_READY_NO_EXECUTION"
            and unlock_checklist.get("pass") is True
            and unlock_checklist.get("execution_allowed_now") is False
            and not _unlock_offenders("unlock_checklist", unlock_checklist),
            {
                "schema_version": unlock_checklist.get("schema_version"),
                "status": unlock_checklist.get("status"),
                "pass": unlock_checklist.get("pass"),
                "execution_allowed_now": unlock_checklist.get("execution_allowed_now"),
                "unlock_offenders": _unlock_offenders("unlock_checklist", unlock_checklist),
            },
        ),
        _check(
            "remote_sync_commands_bound",
            _plan_commands_valid(remote_sync_plan),
            _command_summary(remote_sync_plan),
        ),
    ]

    base_pass = all(item["pass"] for item in checks)
    remote_write_prereqs_pass = (
        allow_remote_write
        and not missing_remote_write_prereqs
        and evidence_path is not None
        and evidence_path.exists()
        and evidence_path.is_file()
    )

    execution_records = []
    if base_pass and remote_write_prereqs_pass:
        execution_records = _execute_plan_commands(remote_sync_plan, runner)
        executed_ok = bool(execution_records) and all(item["returncode"] == 0 for item in execution_records)
        status = EXECUTED_STATUS if executed_ok else BLOCKED_STATUS
        pass_value = executed_ok
    elif allow_remote_write:
        status = BLOCKED_STATUS
        pass_value = False
    else:
        status = DRYRUN_STATUS if base_pass else BLOCKED_STATUS
        pass_value = base_pass

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": str(generated_at),
        "status": status,
        "pass": pass_value,
        "route": "CTF-BDI/PC-OT-MRAS",
        "purpose": (
            "Fail-closed executor for the C1 reader-disabled native OpenSSH remote sync plan. "
            "Default mode performs no remote write."
        ),
        "allow_remote_write_requested": bool(allow_remote_write),
        "remote_write_executed": status == EXECUTED_STATUS,
        "checks": checks,
        "failed_checks": [item for item in checks if not item["pass"]],
        "missing_remote_write_prereqs": missing_remote_write_prereqs,
        "user_unlock_evidence": str(evidence_path) if evidence_path else None,
        "user_unlock_evidence_exists": bool(evidence_path and evidence_path.exists()),
        "command_summary": _command_summary(remote_sync_plan),
        "execution_records": execution_records,
        "execution_allowed_now": False,
        "execution_unlock": _execution_unlock_false_payload(),
        "diagnostic_scope": {
            "remote_sync_executed": status == EXECUTED_STATUS,
            "post_sync_sha_evidence_available": status == EXECUTED_STATUS,
            "formal_gate_generated": False,
            "slurm_submitted": False,
            "tools_test_executed": False,
            "detector_map_available": False,
        },
        "non_unlocks": [
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
    parser = argparse.ArgumentParser(description="Fail-closed executor for the PC-OT-MRAS C1 remote sync plan.")
    parser.add_argument("--remote-sync-plan", required=True)
    parser.add_argument("--unlock-checklist", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--generated-at", default="UNSET_GENERATED_AT")
    parser.add_argument("--allow-remote-write", action="store_true")
    parser.add_argument("--user-unlock-evidence", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    payload = build_remote_sync_execution(
        remote_sync_plan=_load_json(args.remote_sync_plan),
        unlock_checklist=_load_json(args.unlock_checklist),
        allow_remote_write=args.allow_remote_write,
        user_unlock_evidence=args.user_unlock_evidence,
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
