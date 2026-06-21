import argparse
import hashlib
import json
from pathlib import Path


SCHEMA_VERSION = "pc_ot_mras_c1_sync_package_dryrun_v0"

C1_SYNC_FILES = (
    "opentad/models/detectors/actionformer.py",
    "configs/adatad/thumos/ctf_bdi_pc_ot_mras_r17_reader_disabled_eval_candidate.py",
    "configs/adatad/thumos/ctf_bdi_pc_ot_mras_r18_reader_disabled_eval_candidate.py",
    "scripts/run_ctf_bdi_pc_ot_mras_reader_disabled_eval_n16r4.sbatch",
    "tools/bata/finalize_pc_ot_mras_reader_disabled_gate.py",
    "tests/test_pc_ot_mras_actionformer_forward_selector.py",
    "tests/test_pc_ot_mras_r17_r18_post_train_eval_gate.py",
    "tests/test_pc_ot_mras_reader_disabled_gate_finalizer.py",
)

FORBIDDEN_SUFFIXES = {
    ".ckpt",
    ".mp4",
    ".npy",
    ".npz",
    ".pkl",
    ".pth",
    ".pt",
    ".tar",
    ".tgz",
    ".zip",
}

FORBIDDEN_PARTS = {"checkpoint", "checkpoints", "exps", "figures", "logs", "outputs", "work_dirs"}


class SyncPackageError(ValueError):
    pass


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition, message):
    if not condition:
        raise SyncPackageError(message)


def _is_relative_to(path, parent):
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
    except ValueError:
        return False
    return True


def _normalize_repo_path(repo, rel_path):
    repo = Path(repo).resolve()
    rel = Path(rel_path)
    _require(not rel.is_absolute(), f"sync path must be relative: {rel_path}")
    _require(".." not in rel.parts, f"sync path must not escape repo: {rel_path}")
    resolved = (repo / rel).resolve()
    _require(_is_relative_to(resolved, repo), f"sync path escapes repo: {rel_path}")
    return rel.as_posix(), resolved


def build_sync_package(repo, *, files=None, generated_at="UNSET_GENERATED_AT"):
    repo = Path(repo).resolve()
    files = tuple(files or C1_SYNC_FILES)
    records = []
    total_size = 0
    for rel_path in files:
        rel_posix, path = _normalize_repo_path(repo, rel_path)
        parts = set(Path(rel_posix).parts)
        _require(path.exists(), f"missing sync file: {rel_posix}")
        _require(path.is_file(), f"sync path is not a file: {rel_posix}")
        _require(path.suffix.lower() not in FORBIDDEN_SUFFIXES, f"forbidden artifact suffix in sync file: {rel_posix}")
        _require(not (parts & FORBIDDEN_PARTS), f"forbidden artifact directory in sync file: {rel_posix}")
        size = path.stat().st_size
        total_size += size
        records.append(
            {
                "path": rel_posix,
                "sha256": _sha256_file(path),
                "size_bytes": size,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "DRYRUN_ONLY_NOT_SYNCABLE",
        "repo": str(repo),
        "purpose": "Prepare exact C1 reader-disabled source file bindings for a later reviewed remote sync.",
        "files": records,
        "file_count": len(records),
        "total_size_bytes": total_size,
        "execution_unlock": {
            "remote_sync_allowed": False,
            "slurm_allowed": False,
            "tools_test_allowed": False,
            "detector_map_reporting_allowed": False,
            "metric_claim_allowed": False,
            "paper_claim_allowed": False,
        },
        "missing_before_remote_sync": [
            "required_review_or_explicit_replacement",
            "tracker_manifest_unlocked_for_c1_remote_sync",
            "remote_target_path_decision",
            "post_sync_remote_active_manifest_sha256",
            "post_sync_remote_resolved_config_sha256",
            "formal_gate_json_generation",
        ],
        "non_unlocks": [
            "remote_sync",
            "slurm_submission",
            "tools_test",
            "result_detection_json_creation",
            "detector_map_reporting",
            "runtime_flops_claim",
            "deploy_claim",
            "metric_claim",
            "paper_claim",
        ],
        "pass": True,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Build a dry-run C1 reader-disabled sync package manifest.")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--generated-at", default="UNSET_GENERATED_AT")
    return parser.parse_args()


def main():
    args = parse_args()
    payload = build_sync_package(args.repo, generated_at=args.generated_at)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
