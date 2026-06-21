import argparse
import json
from pathlib import Path


SCHEMA_VERSION = "pc_ot_mras_c1_unlock_candidate_v0"
READY = "UNLOCK_CANDIDATE_READY_NOT_EXECUTABLE"
FAILED = "UNLOCK_CANDIDATE_FAILED"

EXPECTED_PREFLIGHT_SCHEMA = "pc_ot_mras_c1_acquisition_preflight_v0"
EXPECTED_SYNC_PLAN_SCHEMA = "pc_ot_mras_c1_remote_sync_plan_v0"
EXPECTED_SUBMISSION_PLAN_SCHEMA = "pc_ot_mras_c1_reader_disabled_submission_plan_v0"
EXPECTED_EXECUTION_MATRIX_SCHEMA = "pc_ot_mras_execution_matrix_verifier_v0"
REQUIRED_GEMINI_VERDICT = "PASS_LOCAL_REVIEW_ONLY"

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

REQUIRED_MISSING_BEFORE_EXECUTION = (
    "explicit_user_approved_replacement_or_unlock_for_cancelled_pro_major_gate",
    "remote_sync_execution_and_sha_verification",
    "remote_precheck_active_manifest_sha256",
    "remote_precheck_resolved_config_sha256",
    "formal_gate_json_r17",
    "formal_gate_json_r18",
    "tracker_manifest_unlocked_for_c1_remote_sync",
    "tracker_manifest_unlocked_for_reader_disabled_tools_test",
    "deliberate_sbatch_submission",
)


class UnlockCandidateError(ValueError):
    pass


def _load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise UnlockCandidateError(f"invalid JSON: {path}: {exc}") from exc


def _read_text(path):
    data = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "utf-16", "utf-16-le", "utf-16-be"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _read_exitcode(path):
    text = _read_text(path).strip()
    try:
        return int(text)
    except ValueError as exc:
        raise UnlockCandidateError(f"invalid Gemini exit code file: {path}") from exc


def _check(name, passed, detail):
    return {"name": name, "pass": bool(passed), "detail": detail}


def _execution_unlock_false_payload():
    return {key: False for key in EXECUTION_UNLOCK_KEYS}


def _unlock_offenders(label, payload, keys=EXECUTION_UNLOCK_KEYS):
    unlock = payload.get("execution_unlock") or {}
    offenders = []
    for key in keys:
        if key in unlock and unlock.get(key) is not False:
            offenders.append(f"{label}.execution_unlock.{key}={unlock.get(key)!r}")
    return offenders


def _submission_targets(submission_plan):
    return sorted(str(item.get("target")) for item in submission_plan.get("submission_commands") or [])


def build_unlock_candidate(
    *,
    preflight,
    gemini_stdout,
    gemini_exitcode,
    gemini_report_path,
    remote_sync_plan,
    submission_plan,
    execution_matrix,
    generated_at="UNSET_GENERATED_AT",
):
    gemini_report = Path(gemini_report_path)
    checks = [
        _check(
            "c1_local_preflight_pass_execution_locked",
            preflight.get("schema_version") == EXPECTED_PREFLIGHT_SCHEMA
            and preflight.get("status") == "LOCAL_PREFLIGHT_PASS_EXECUTION_LOCKED"
            and preflight.get("pass") is True
            and preflight.get("local_preflight_pass") is True
            and preflight.get("execution_allowed") is False
            and preflight.get("remote_sync_allowed") is False
            and preflight.get("slurm_allowed") is False
            and preflight.get("tools_test_allowed") is False
            and preflight.get("detector_map_reporting_allowed") is False,
            {
                "schema_version": preflight.get("schema_version"),
                "status": preflight.get("status"),
                "pass": preflight.get("pass"),
                "execution_allowed": preflight.get("execution_allowed"),
            },
        ),
        _check(
            "gemini_cli_review_pass_local_only",
            int(gemini_exitcode) == 0
            and REQUIRED_GEMINI_VERDICT in str(gemini_stdout)
            and gemini_report.exists()
            and gemini_report.is_file(),
            {
                "exitcode": int(gemini_exitcode),
                "required_verdict": REQUIRED_GEMINI_VERDICT,
                "verdict_found": REQUIRED_GEMINI_VERDICT in str(gemini_stdout),
                "report_path": str(gemini_report),
                "report_exists": gemini_report.exists(),
            },
        ),
        _check(
            "remote_sync_plan_ready_not_executed",
            remote_sync_plan.get("schema_version") == EXPECTED_SYNC_PLAN_SCHEMA
            and remote_sync_plan.get("status") == "PLAN_READY_NOT_EXECUTED"
            and remote_sync_plan.get("pass") is True
            and not _unlock_offenders("remote_sync_plan", remote_sync_plan),
            {
                "schema_version": remote_sync_plan.get("schema_version"),
                "status": remote_sync_plan.get("status"),
                "file_count": remote_sync_plan.get("file_count"),
                "unlock_offenders": _unlock_offenders("remote_sync_plan", remote_sync_plan),
            },
        ),
        _check(
            "submission_plan_ready_not_executed",
            submission_plan.get("schema_version") == EXPECTED_SUBMISSION_PLAN_SCHEMA
            and submission_plan.get("status") == "PLAN_READY_NOT_EXECUTED"
            and submission_plan.get("pass") is True
            and _submission_targets(submission_plan) == ["R17", "R18"]
            and not _unlock_offenders("submission_plan", submission_plan),
            {
                "schema_version": submission_plan.get("schema_version"),
                "status": submission_plan.get("status"),
                "targets": _submission_targets(submission_plan),
                "unlock_offenders": _unlock_offenders("submission_plan", submission_plan),
            },
        ),
        _check(
            "execution_matrix_pass_no_new_training_unlocked",
            execution_matrix.get("schema_version") == EXPECTED_EXECUTION_MATRIX_SCHEMA
            and execution_matrix.get("pass") is True
            and execution_matrix.get("current_execution_conclusion") == "no_new_full_training_unlocked"
            and execution_matrix.get("full_training_worthy_now") == ["R17", "R18"]
            and not [
                key
                for key, value in (execution_matrix.get("c1_unlock_flags") or {}).items()
                if value is not False
            ],
            {
                "schema_version": execution_matrix.get("schema_version"),
                "pass": execution_matrix.get("pass"),
                "current_execution_conclusion": execution_matrix.get("current_execution_conclusion"),
                "full_training_worthy_now": execution_matrix.get("full_training_worthy_now"),
                "c1_unlock_flags": execution_matrix.get("c1_unlock_flags"),
            },
        ),
    ]

    passed = all(item["pass"] for item in checks)
    missing = sorted(
        set(REQUIRED_MISSING_BEFORE_EXECUTION)
        | set(preflight.get("missing_before_execution") or [])
        | set(remote_sync_plan.get("required_before_execution") or [])
        | set(submission_plan.get("required_before_execution") or [])
    )
    satisfied = [item["name"] for item in checks if item["pass"]]
    failed_checks = [item for item in checks if not item["pass"]]

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": str(generated_at),
        "status": READY if passed else FAILED,
        "pass": passed,
        "route": "CTF-BDI/PC-OT-MRAS",
        "targets": ["R17", "R18"],
        "purpose": (
            "Aggregate local C1 reader-disabled evidence into a machine-checkable "
            "unlock candidate while preserving all execution locks."
        ),
        "checks": checks,
        "satisfied_conditions": satisfied,
        "failed_checks": failed_checks,
        "missing_before_execution": missing,
        "execution_unlock": _execution_unlock_false_payload(),
        "source_refs": {
            "preflight_status": preflight.get("status"),
            "gemini_verdict": REQUIRED_GEMINI_VERDICT if REQUIRED_GEMINI_VERDICT in str(gemini_stdout) else None,
            "gemini_report": str(gemini_report),
            "remote_sync_plan_status": remote_sync_plan.get("status"),
            "submission_plan_status": submission_plan.get("status"),
            "execution_matrix_conclusion": execution_matrix.get("current_execution_conclusion"),
        },
        "diagnostic_scope": {
            "reader_out_snapshots_ready": True,
            "selection_heatmaps_timelines_ready": True,
            "exact_uniform_matched_compare_ready": True,
            "reader_disabled_detector_control_ready_to_unlock": passed,
            "detector_map_available": False,
            "same_head_control_available": False,
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
        description="Build a non-executable C1 reader-disabled unlock-candidate evidence package."
    )
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--gemini-stdout", required=True)
    parser.add_argument("--gemini-exitcode", required=True)
    parser.add_argument("--gemini-report", required=True)
    parser.add_argument("--remote-sync-plan", required=True)
    parser.add_argument("--submission-plan", required=True)
    parser.add_argument("--execution-matrix", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--generated-at", default="UNSET_GENERATED_AT")
    return parser.parse_args()


def main():
    args = parse_args()
    payload = build_unlock_candidate(
        preflight=_load_json(args.preflight),
        gemini_stdout=_read_text(args.gemini_stdout),
        gemini_exitcode=_read_exitcode(args.gemini_exitcode),
        gemini_report_path=args.gemini_report,
        remote_sync_plan=_load_json(args.remote_sync_plan),
        submission_plan=_load_json(args.submission_plan),
        execution_matrix=_load_json(args.execution_matrix),
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
