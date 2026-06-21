import argparse
import json
from pathlib import Path


SCHEMA_VERSION = "pc_ot_mras_c1_unlock_checklist_v0"
READY = "UNLOCK_CHECKLIST_READY_NO_EXECUTION"
FAILED = "UNLOCK_CHECKLIST_FAILED"

EXPECTED_HANDOFF_SCHEMA = "pc_ot_mras_c1_execution_handoff_v0"
EXPECTED_HANDOFF_STATUS = "HANDOFF_READY_NOT_EXECUTABLE"

EXPECTED_STAGE_NAMES = (
    "review_replacement_or_user_unlock",
    "remote_sync_after_unlock",
    "remote_precheck_after_sync",
    "formal_gate_generation_after_precheck",
    "slurm_reader_disabled_eval_submission",
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


class ChecklistError(ValueError):
    pass


def _load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ChecklistError(f"invalid JSON: {path}: {exc}") from exc


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


def _stage_names(handoff):
    return [str(stage.get("name")) for stage in handoff.get("handoff_stages") or []]


def _stage_executable_offenders(handoff):
    return [
        str(stage.get("name"))
        for stage in handoff.get("handoff_stages") or []
        if stage.get("executable_now") is not False
    ]


def _checklist_item(key, stage, status, evidence_needed, unlocks_if_completed=None):
    return {
        "key": key,
        "stage": stage,
        "status": status,
        "evidence_needed": evidence_needed,
        "unlocks_if_completed": unlocks_if_completed or [],
    }


def build_unlock_checklist(*, handoff, generated_at="UNSET_GENERATED_AT"):
    checks = [
        _check(
            "handoff_ready_not_executable",
            handoff.get("schema_version") == EXPECTED_HANDOFF_SCHEMA
            and handoff.get("status") == EXPECTED_HANDOFF_STATUS
            and handoff.get("pass") is True
            and handoff.get("execution_allowed_now") is False,
            {
                "schema_version": handoff.get("schema_version"),
                "status": handoff.get("status"),
                "pass": handoff.get("pass"),
                "execution_allowed_now": handoff.get("execution_allowed_now"),
            },
        ),
        _check(
            "all_execution_unlocks_false",
            not _unlock_offenders("handoff", handoff),
            {
                "unlock_offenders": _unlock_offenders("handoff", handoff),
            },
        ),
        _check(
            "expected_stage_sequence_present",
            tuple(_stage_names(handoff)) == EXPECTED_STAGE_NAMES,
            {
                "expected": list(EXPECTED_STAGE_NAMES),
                "actual": _stage_names(handoff),
            },
        ),
        _check(
            "all_handoff_stages_non_executable",
            not _stage_executable_offenders(handoff),
            {
                "executable_stage_offenders": _stage_executable_offenders(handoff),
            },
        ),
    ]
    passed = all(item["pass"] for item in checks)

    items = [
        _checklist_item(
            "explicit_user_approved_replacement_or_unlock_for_cancelled_pro_major_gate",
            "review_replacement_or_user_unlock",
            "missing",
            "A local tracker/log entry or user message explicitly replacing/unlocking the cancelled Pro major gate for C1 only.",
            ["tracker_manifest_unlocked_for_c1_remote_sync"],
        ),
        _checklist_item(
            "tracker_manifest_unlocked_for_c1_remote_sync",
            "remote_sync_after_unlock",
            "missing",
            "Manifest and Sparse/BATA tracker state set to permit C1 remote sync, with non-claim boundaries retained.",
            ["operator_runs_remote_sync_plan_with_remote_write_enabled"],
        ),
        _checklist_item(
            "post_sync_remote_sha_verification_saved",
            "remote_sync_after_unlock",
            "missing",
            "Saved output from the planned native OpenSSH post-sync SHA verification for all copied C1 source files.",
            ["remote_precheck_active_manifest_sha256", "remote_precheck_resolved_config_sha256"],
        ),
        _checklist_item(
            "remote_precheck_active_and_resolved_sha_r17_r18",
            "remote_precheck_after_sync",
            "missing",
            "R17 and R18 precheck summaries containing active-manifest SHA256 and resolved-config SHA256.",
            ["formal_gate_json_r17", "formal_gate_json_r18"],
        ),
        _checklist_item(
            "formal_gate_json_and_sha_r17_r18",
            "formal_gate_generation_after_precheck",
            "missing",
            "Formal R17/R18 reader-disabled gate JSON files plus SHA256 files generated by the finalizer.",
            ["tracker_manifest_unlocked_for_reader_disabled_tools_test"],
        ),
        _checklist_item(
            "tracker_manifest_unlocked_for_reader_disabled_tools_test",
            "slurm_reader_disabled_eval_submission",
            "missing",
            "Manifest and tracker state explicitly permitting C1 reader-disabled diagnostic tools/test execution only.",
            ["deliberate_sbatch_submission"],
        ),
        _checklist_item(
            "deliberate_sbatch_submission_and_job_ids",
            "slurm_reader_disabled_eval_submission",
            "missing",
            "Operator submits R17/R18 sbatch commands and records job IDs before monitoring.",
            ["result_detection_json_and_reader_disabled_summary_collection"],
        ),
        _checklist_item(
            "result_detection_json_and_reader_disabled_summary_collection",
            "post_slurm_collection",
            "not_available_before_execution",
            "R17/R18 result_detection.json and reader_disabled_eval_summary.json after completed Slurm diagnostics.",
            [],
        ),
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": str(generated_at),
        "status": READY if passed else FAILED,
        "pass": passed,
        "route": "CTF-BDI/PC-OT-MRAS",
        "targets": ["R17", "R18"],
        "purpose": (
            "Create a machine-checkable unlock checklist for C1 reader-disabled "
            "remote sync and diagnostic execution without performing any execution."
        ),
        "checks": checks,
        "failed_checks": [item for item in checks if not item["pass"]],
        "current_blocking_item": items[0]["key"],
        "checklist": items,
        "all_checklist_items_satisfied": False,
        "execution_allowed_now": False,
        "execution_unlock": _execution_unlock_false_payload(),
        "source_refs": {
            "handoff_status": handoff.get("status"),
            "handoff_generated_at": handoff.get("generated_at"),
            "handoff_remote_root": handoff.get("remote_root"),
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
        description="Build a non-executable C1 reader-disabled unlock checklist."
    )
    parser.add_argument("--handoff", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--generated-at", default="UNSET_GENERATED_AT")
    return parser.parse_args()


def main():
    args = parse_args()
    payload = build_unlock_checklist(
        handoff=_load_json(args.handoff),
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
