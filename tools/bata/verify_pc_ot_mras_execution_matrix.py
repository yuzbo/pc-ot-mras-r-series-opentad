import argparse
import json
from pathlib import Path


SCHEMA_VERSION = "pc_ot_mras_execution_matrix_verifier_v0"
REQUIRED_STAGES = ["R16A", "R16B"] + [f"R{i}" for i in range(17, 36)]
REQUIRED_MODEL_FIELDS = (
    "stage",
    "experiment_model",
    "changed_surface",
    "innovation_purpose",
    "implementation_summary",
    "full_training_worthy_now",
    "remote_sync_status",
    "training_status",
    "next_gate",
)
EXPECTED_FULL_TRAINING_WORTHY = ("R17", "R18")


class ExecutionMatrixError(ValueError):
    pass


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _nonempty_string(value):
    return isinstance(value, str) and bool(value.strip())


def _model_execution_class(model):
    stage = model["stage"]
    training_status = str(model.get("training_status", ""))
    remote_sync_status = str(model.get("remote_sync_status", ""))
    if model.get("full_training_worthy_now"):
        return "completed_full_training_worthy" if training_status.startswith("completed") else "unfinished_full_training_worthy"
    if stage == "R16A":
        return "completed_smoke_only"
    if stage == "R16B":
        return "gate_plan_not_model"
    if "planned" in remote_sync_status or stage in {"R33", "R34", "R35"}:
        return "planned_or_locked_future_candidate"
    if "local" in remote_sync_status or "reviewed" in remote_sync_status:
        return "local_or_review_context_only"
    return "locked_not_full_training"


def verify_execution_matrix(manifest_path):
    manifest = _load_json(manifest_path)
    models = manifest.get("models") or []
    model_by_stage = {item.get("stage"): item for item in models}
    missing_stages = [stage for stage in REQUIRED_STAGES if stage not in model_by_stage]
    extra_stages = sorted(stage for stage in model_by_stage if stage not in REQUIRED_STAGES)

    row_errors = []
    rows = []
    for stage in REQUIRED_STAGES:
        model = model_by_stage.get(stage)
        if not model:
            continue
        missing_fields = [field for field in REQUIRED_MODEL_FIELDS if field not in model]
        if missing_fields:
            row_errors.append({"stage": stage, "error": "missing_fields", "fields": missing_fields})
            continue
        for field in ("experiment_model", "innovation_purpose", "implementation_summary", "remote_sync_status", "training_status", "next_gate"):
            if not _nonempty_string(model.get(field)):
                row_errors.append({"stage": stage, "error": "empty_field", "field": field})
        changed_surface = model.get("changed_surface")
        if not isinstance(changed_surface, list) or not changed_surface or not all(_nonempty_string(item) for item in changed_surface):
            row_errors.append({"stage": stage, "error": "invalid_changed_surface"})
        if not isinstance(model.get("full_training_worthy_now"), bool):
            row_errors.append({"stage": stage, "error": "full_training_worthy_now_not_bool"})
        rows.append(
            {
                "stage": stage,
                "experiment_model": model["experiment_model"],
                "changed_surface": changed_surface,
                "innovation_purpose": model["innovation_purpose"],
                "implementation_summary": model["implementation_summary"],
                "full_training_worthy_now": model["full_training_worthy_now"],
                "remote_sync_status": model["remote_sync_status"],
                "training_status": model["training_status"],
                "next_gate": model["next_gate"],
                "execution_class": _model_execution_class(model),
            }
        )

    full_training_worthy = tuple(row["stage"] for row in rows if row["full_training_worthy_now"])
    unexpected_full_training_worthy = sorted(set(full_training_worthy) - set(EXPECTED_FULL_TRAINING_WORTHY))
    missing_expected_full_training = sorted(set(EXPECTED_FULL_TRAINING_WORTHY) - set(full_training_worthy))
    unfinished_full_training = sorted(
        row["stage"]
        for row in rows
        if row["full_training_worthy_now"] and not str(row["training_status"]).startswith("completed")
    )

    analysis = manifest.get("analysis_status") or {}
    c1_controls = {
        "remote_sync_plan": (analysis.get("c1_remote_sync_plan") or {}).get("status"),
        "submission_plan": (analysis.get("c1_reader_disabled_submission_plan") or {}).get("status"),
    }
    c1_unlock_flags = {
        "remote_sync_allowed": bool((analysis.get("c1_remote_sync_plan") or {}).get("remote_sync_allowed")),
        "slurm_allowed": bool((analysis.get("c1_reader_disabled_submission_plan") or {}).get("slurm_allowed")),
        "tools_test_allowed": bool((analysis.get("c1_reader_disabled_submission_plan") or {}).get("tools_test_allowed")),
        "detector_map_reporting_allowed": bool((analysis.get("c1_reader_disabled_submission_plan") or {}).get("detector_map_reporting_allowed")),
    }
    c1_unexpected_unlocks = sorted(key for key, value in c1_unlock_flags.items() if value)

    active_jobs = manifest.get("active_full_training_jobs") or []
    completed_full_training_jobs = sorted(
        item.get("stage")
        for item in active_jobs
        if str(item.get("state", "")).startswith("COMPLETED")
    )

    errors = {
        "missing_stages": missing_stages,
        "extra_stages": extra_stages,
        "row_errors": row_errors,
        "unexpected_full_training_worthy": unexpected_full_training_worthy,
        "missing_expected_full_training": missing_expected_full_training,
        "unfinished_full_training": unfinished_full_training,
        "c1_unexpected_unlocks": c1_unexpected_unlocks,
    }
    passed = not any(errors.values())

    return {
        "schema_version": SCHEMA_VERSION,
        "manifest": str(manifest_path),
        "manifest_status": manifest.get("manifest_status"),
        "model_count": len(models),
        "required_stage_count": len(REQUIRED_STAGES),
        "rows": rows,
        "full_training_worthy_now": list(full_training_worthy),
        "expected_full_training_worthy": list(EXPECTED_FULL_TRAINING_WORTHY),
        "completed_full_training_jobs": completed_full_training_jobs,
        "current_execution_conclusion": (
            "no_new_full_training_unlocked"
            if full_training_worthy == EXPECTED_FULL_TRAINING_WORTHY and not unfinished_full_training
            else "execution_matrix_requires_attention"
        ),
        "c1_control_status": c1_controls,
        "c1_unlock_flags": c1_unlock_flags,
        "errors": errors,
        "pass": passed,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Verify and export the PC-OT-MRAS model execution matrix.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    payload = verify_execution_matrix(args.manifest)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(text, end="")
    raise SystemExit(0 if payload["pass"] else 1)


if __name__ == "__main__":
    main()
