import argparse
import json
import re
from pathlib import Path


REQUIRED_CURRENT_STAGES = ["R16A", "R16B"] + [f"R{i}" for i in range(17, 36)]
def _stage_from_name(name):
    if "synthetic_local" in name:
        return "SYNTHETIC_LOCAL"
    match = re.search(r"_r(\d+)([a-z]?)_", f"_{name}")
    if not match:
        return None
    number = int(match.group(1))
    suffix = match.group(2).upper()
    if number == 16 and "gpu_smoke" in name:
        return "R16A"
    if suffix:
        return f"R{number}{suffix}"
    return f"R{number}"


def _scan_files(repo):
    repo = Path(repo)
    config_dir = repo / "configs" / "adatad" / "thumos"
    script_dir = repo / "scripts"
    configs = sorted(config_dir.glob("ctf_bdi_pc_ot_mras*.py"))
    launchers = sorted(script_dir.glob("run_ctf_bdi_pc_ot_mras*.sbatch"))
    tools = sorted((repo / "tools" / "bata").glob("*pc_ot_mras*.py"))
    tests = sorted((repo / "tests").glob("test_pc_ot_mras*.py"))
    return {
        "configs": configs,
        "launchers": launchers,
        "tools": tools,
        "tests": tests,
    }


def _classify_launcher(path):
    name = path.name
    if "formal_train" in name:
        return "formal_full_training"
    if "gpu_smoke" in name:
        return "smoke_only"
    if "precheck" in name:
        return "precheck_only"
    if "post_train_eval" in name or "reader_disabled_eval" in name:
        return "diagnostic_eval_only"
    if "selection_learning_diagnostic" in name:
        return "diagnostic_visualization_only"
    return "unknown"


def verify_trainability_manifest(manifest_path, repo_path):
    manifest_path = Path(manifest_path)
    repo_path = Path(repo_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    models = manifest.get("models") or []
    model_by_stage = {item["stage"]: item for item in models}
    manifest_stages = set(model_by_stage)

    files = _scan_files(repo_path)
    config_stage_map = {
        str(path.relative_to(repo_path)).replace("\\", "/"): _stage_from_name(path.name)
        for path in files["configs"]
    }
    launcher_map = {
        str(path.relative_to(repo_path)).replace("\\", "/"): {
            "stage": _stage_from_name(path.name),
            "kind": _classify_launcher(path),
        }
        for path in files["launchers"]
    }

    missing_required_stages = [stage for stage in REQUIRED_CURRENT_STAGES if stage not in manifest_stages]
    historical_or_local_config_stages = sorted(
        {
            stage
            for stage in config_stage_map.values()
            if stage and stage not in manifest_stages
        }
    )
    unknown_configs = sorted(path for path, stage in config_stage_map.items() if stage is None)
    unknown_launchers = sorted(path for path, item in launcher_map.items() if item["kind"] == "unknown")
    formal_launchers = sorted(path for path, item in launcher_map.items() if item["kind"] == "formal_full_training")

    full_training_worthy = sorted(stage for stage, item in model_by_stage.items() if item.get("full_training_worthy_now"))
    unfinished_full_training_worthy = sorted(
        stage
        for stage, item in model_by_stage.items()
        if item.get("full_training_worthy_now")
        and not str(item.get("training_status", "")).startswith("completed")
    )
    formal_launcher_stages = sorted(
        {item["stage"] for item in launcher_map.values() if item["kind"] == "formal_full_training" and item["stage"]}
    )
    unregistered_formal_launcher_stages = sorted(stage for stage in formal_launcher_stages if stage not in manifest_stages)

    result = {
        "schema_version": "pc_ot_mras_trainability_manifest_verifier_v0",
        "manifest": str(manifest_path),
        "repo": str(repo_path),
        "manifest_model_count": len(models),
        "required_current_stages": REQUIRED_CURRENT_STAGES,
        "missing_required_stages": missing_required_stages,
        "full_training_worthy_now": full_training_worthy,
        "unfinished_full_training_worthy": unfinished_full_training_worthy,
        "formal_full_training_launchers": formal_launchers,
        "formal_full_training_launcher_stages": formal_launcher_stages,
        "unregistered_formal_launcher_stages": unregistered_formal_launcher_stages,
        "historical_or_local_config_stages_outside_manifest": historical_or_local_config_stages,
        "unknown_configs": unknown_configs,
        "unknown_launchers": unknown_launchers,
        "counts": {
            "configs": len(files["configs"]),
            "launchers": len(files["launchers"]),
            "tools": len(files["tools"]),
            "tests": len(files["tests"]),
        },
        "current_execution_conclusion": "no_new_full_training_unlocked",
        "pass": not (
            missing_required_stages
            or unfinished_full_training_worthy
            or unregistered_formal_launcher_stages
            or unknown_launchers
        ),
    }
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Verify PC-OT-MRAS trainability manifest coverage.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    result = verify_trainability_manifest(args.manifest, args.repo)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
