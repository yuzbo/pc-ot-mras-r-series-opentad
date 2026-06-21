import argparse
import hashlib
import json
from pathlib import Path


SCHEMA_VERSION = "pc_ot_mras_c1_acquisition_preflight_v0"

DEFAULT_TRAINABILITY_JSON = (
    "research-wiki/experiments/CTF_BDI_PC_OT_MRAS_TRAINABILITY_MANIFEST_VERIFY_20260622.json"
)
DEFAULT_SYNC_PACKAGE_JSON = (
    "research-wiki/experiments/CTF_BDI_PC_OT_MRAS_C1_SYNC_PACKAGE_DRYRUN_20260622.json"
)
DEFAULT_UNLOCK_PACKET_JSON = (
    "research-wiki/experiments/CTF_BDI_PC_OT_MRAS_C1_EVIDENCE_ACQUISITION_UNLOCK_PACKET_20260622.json"
)
DEFAULT_GATE_DRAFT_JSON = (
    "research-wiki/experiments/CTF_BDI_PC_OT_MRAS_READER_DISABLED_C1_GATE_BINDING_DRAFT_20260622.json"
)
DEFAULT_FINAL_FIGURE_DIR = "figures/pcotmras_selection_diag_20260622_final"

EXECUTION_UNLOCK_FALSE_KEYS = (
    "remote_sync_allowed",
    "slurm_allowed",
    "tools_test_allowed",
    "detector_map_reporting_allowed",
    "detector_map_claim_allowed",
    "metric_claim_allowed",
    "paper_claim_allowed",
)

REQUIRED_AVAILABLE_FLAGS = (
    "real_checkpoint_reader_out_snapshots",
    "selection_heatmaps_and_timelines",
    "exact_uniform_matched_compare",
    "reader_disabled_local_override",
    "reader_disabled_configs",
    "reader_disabled_launcher",
    "binding_draft",
    "gate_finalizer_dryrun",
)


class C1PreflightError(ValueError):
    pass


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise C1PreflightError(f"invalid JSON: {path}: {exc}") from exc


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _line_count(path):
    with open(path, "r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _check(name, passed, detail):
    return {"name": name, "pass": bool(passed), "detail": detail}


def _execution_unlocks_all_false(*payloads):
    offenders = []
    for label, payload in payloads:
        unlock = payload.get("execution_unlock") or {}
        for key in EXECUTION_UNLOCK_FALSE_KEYS:
            if key in unlock and unlock.get(key) is not False:
                offenders.append(f"{label}.execution_unlock.{key}={unlock.get(key)!r}")
    return offenders


def _validate_gate_draft(draft):
    targets = draft.get("targets") or {}
    problems = []
    if draft.get("schema_version") != "pc_ot_mras_reader_disabled_c1_gate_binding_draft_v0":
        problems.append("unexpected schema_version")
    if draft.get("status") != "DRAFT_ONLY_NOT_EXECUTABLE":
        problems.append("status is not DRAFT_ONLY_NOT_EXECUTABLE")
    if set(targets) != {"R17", "R18"}:
        problems.append("targets must be exactly R17 and R18")
    for target in ("R17", "R18"):
        item = targets.get(target) or {}
        required = item.get("required_formal_gate_fields") or {}
        if item.get("decision_if_formalized") != f"ALLOW_{target}_READER_DISABLED_EVAL":
            problems.append(f"{target} decision mismatch")
        if len(str(item.get("checkpoint_sha256", ""))) != 64:
            problems.append(f"{target} checkpoint_sha256 missing or invalid")
        if item.get("checkpoint_size_bytes", 0) <= 0:
            problems.append(f"{target} checkpoint_size_bytes invalid")
        if "Training Over" not in str(item.get("training_over_evidence", "")):
            problems.append(f"{target} Training Over evidence missing")
        for key in ("completed_training_evidence", "reader_disabled_eval", "same_head_control"):
            if required.get(key) is not True:
                problems.append(f"{target} required_formal_gate_fields.{key} must be true")
        for key in ("tools_train", "raw_prediction_cache", "paper_claim", "runtime_flops_claim", "deploy_claim"):
            if required.get(key) is not False:
                problems.append(f"{target} required_formal_gate_fields.{key} must be false")
    return problems


def _selector_artifact_summary(workspace_root, final_figure_dir):
    figure_dir = Path(workspace_root) / final_figure_dir
    summary_path = (
        figure_dir
        / "pcotmras_selection_learning_final_epoch59_matched_compare_20260622"
        / "final_epoch59_exact_uniform_matched_comparison_summary.json"
    )
    per_sample_path = (
        figure_dir
        / "pcotmras_selection_learning_final_epoch59_matched_compare_20260622"
        / "final_epoch59_exact_uniform_matched_comparison_per_sample.csv"
    )
    targets = {}
    for target in ("R17", "R18"):
        lower = target.lower()
        snapshot_dir = figure_dir / f"{target}_final_{lower}_epoch59_final_reader_snapshot"
        snapshot_path = snapshot_dir / "snapshot.jsonl"
        viz_dir = snapshot_dir / "viz"
        targets[target] = {
            "snapshot": str(snapshot_path),
            "snapshot_exists": snapshot_path.exists(),
            "snapshot_rows": _line_count(snapshot_path) if snapshot_path.exists() else 0,
            "viz_dir": str(viz_dir),
            "svg_count": len(list(viz_dir.glob("*.svg"))) if viz_dir.exists() else 0,
        }
    return {
        "final_figure_dir": str(figure_dir),
        "matched_summary": str(summary_path),
        "matched_summary_exists": summary_path.exists(),
        "matched_per_sample_csv": str(per_sample_path),
        "matched_per_sample_csv_exists": per_sample_path.exists(),
        "targets": targets,
    }


def _current_source_matches_sync_package(repo, sync_package):
    records = []
    mismatches = []
    repo = Path(repo)
    for item in sync_package.get("files") or []:
        rel_path = item.get("path")
        path = repo / rel_path
        exists = path.exists() and path.is_file()
        actual_sha256 = _sha256_file(path) if exists else None
        match = exists and actual_sha256 == item.get("sha256")
        record = {
            "path": rel_path,
            "exists": exists,
            "expected_sha256": item.get("sha256"),
            "actual_sha256": actual_sha256,
            "match": match,
        }
        records.append(record)
        if not match:
            mismatches.append(record)
    return records, mismatches


def build_c1_acquisition_preflight(
    workspace_root,
    repo,
    *,
    trainability_json=DEFAULT_TRAINABILITY_JSON,
    sync_package_json=DEFAULT_SYNC_PACKAGE_JSON,
    unlock_packet_json=DEFAULT_UNLOCK_PACKET_JSON,
    gate_draft_json=DEFAULT_GATE_DRAFT_JSON,
    final_figure_dir=DEFAULT_FINAL_FIGURE_DIR,
    generated_at="UNSET_GENERATED_AT",
):
    workspace_root = Path(workspace_root).resolve()
    repo = Path(repo).resolve()
    trainability = _read_json(workspace_root / trainability_json)
    sync_package = _read_json(workspace_root / sync_package_json)
    unlock_packet = _read_json(workspace_root / unlock_packet_json)
    gate_draft = _read_json(workspace_root / gate_draft_json)

    selector_artifacts = _selector_artifact_summary(workspace_root, final_figure_dir)
    sync_records, sync_mismatches = _current_source_matches_sync_package(repo, sync_package)
    unlock_offenders = _execution_unlocks_all_false(
        ("sync_package", sync_package),
        ("unlock_packet", unlock_packet),
        ("gate_draft", gate_draft),
    )
    gate_draft_problems = _validate_gate_draft(gate_draft)

    available = unlock_packet.get("already_available") or {}
    missing_available_flags = [key for key in REQUIRED_AVAILABLE_FLAGS if available.get(key) is not True]
    selector_missing = []
    if not selector_artifacts["matched_summary_exists"]:
        selector_missing.append("final_exact_uniform_matched_summary_json")
    if not selector_artifacts["matched_per_sample_csv_exists"]:
        selector_missing.append("final_exact_uniform_matched_per_sample_csv")
    for target, item in selector_artifacts["targets"].items():
        if item["snapshot_rows"] <= 0:
            selector_missing.append(f"{target}_reader_snapshot_jsonl")
        if item["svg_count"] <= 0:
            selector_missing.append(f"{target}_selection_svg_visualization")

    checks = [
        _check(
            "trainability_manifest_current",
            trainability.get("pass") is True
            and trainability.get("full_training_worthy_now") == ["R17", "R18"]
            and trainability.get("unfinished_full_training_worthy") == [],
            {
                "pass": trainability.get("pass"),
                "full_training_worthy_now": trainability.get("full_training_worthy_now"),
                "unfinished_full_training_worthy": trainability.get("unfinished_full_training_worthy"),
                "current_execution_conclusion": trainability.get("current_execution_conclusion"),
            },
        ),
        _check(
            "c1_sync_package_dryrun_current",
            sync_package.get("pass") is True
            and sync_package.get("status") == "DRYRUN_ONLY_NOT_SYNCABLE"
            and not sync_mismatches,
            {
                "status": sync_package.get("status"),
                "file_count": sync_package.get("file_count"),
                "mismatch_count": len(sync_mismatches),
                "mismatches": sync_mismatches,
            },
        ),
        _check(
            "c1_unlock_packet_non_executable",
            unlock_packet.get("status") == "UNLOCK_PACKET_READY_NON_EXECUTABLE"
            and not missing_available_flags,
            {
                "status": unlock_packet.get("status"),
                "missing_available_flags": missing_available_flags,
            },
        ),
        _check(
            "c1_gate_draft_valid_non_executable",
            not gate_draft_problems,
            {"status": gate_draft.get("status"), "problems": gate_draft_problems},
        ),
        _check(
            "selector_final_artifacts_available",
            not selector_missing,
            {"missing": selector_missing, "artifacts": selector_artifacts},
        ),
        _check(
            "all_execution_unlocks_remain_false",
            not unlock_offenders,
            {"offenders": unlock_offenders},
        ),
    ]
    local_preflight_pass = all(item["pass"] for item in checks)
    execution_allowed = False
    missing_before_execution = sorted(
        set(sync_package.get("missing_before_remote_sync") or [])
        | set(unlock_packet.get("missing_before_execution") or [])
        | {
            "required_review_or_explicit_replacement",
            "remote_sync_after_review",
            "formal_gate_json_r17",
            "formal_gate_json_r18",
        }
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "LOCAL_PREFLIGHT_PASS_EXECUTION_LOCKED" if local_preflight_pass else "LOCAL_PREFLIGHT_FAILED",
        "workspace_root": str(workspace_root),
        "repo": str(repo),
        "local_preflight_pass": local_preflight_pass,
        "execution_allowed": execution_allowed,
        "remote_sync_allowed": False,
        "slurm_allowed": False,
        "tools_test_allowed": False,
        "detector_map_reporting_allowed": False,
        "metric_claim_allowed": False,
        "paper_claim_allowed": False,
        "checks": checks,
        "source_fingerprint_records": sync_records,
        "selector_artifacts": selector_artifacts,
        "missing_before_execution": missing_before_execution,
        "current_allowed_commands": unlock_packet.get("current_allowed_commands") or [],
        "decision": (
            "C1 local acquisition evidence path is prepared but execution remains locked"
            if local_preflight_pass
            else "C1 acquisition path has stale or missing local evidence"
        ),
        "non_unlocks": sorted(
            set(sync_package.get("non_unlocks") or [])
            | set(unlock_packet.get("non_unlocks") or [])
            | {"remote_sync", "slurm_submission", "tools_test", "detector_map_reporting", "metric_claim", "paper_claim"}
        ),
        "pass": local_preflight_pass,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Check local C1 reader-disabled evidence-acquisition readiness.")
    parser.add_argument("--workspace-root", default="..")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--output", default=None)
    parser.add_argument("--generated-at", default="UNSET_GENERATED_AT")
    return parser.parse_args()


def main():
    args = parse_args()
    payload = build_c1_acquisition_preflight(
        args.workspace_root,
        args.repo,
        generated_at=args.generated_at,
    )
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")
    raise SystemExit(0 if payload["pass"] else 1)


if __name__ == "__main__":
    main()
