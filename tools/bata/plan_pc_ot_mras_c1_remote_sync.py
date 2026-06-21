import argparse
import json
from pathlib import Path


SCHEMA_VERSION = "pc_ot_mras_c1_remote_sync_plan_v0"
EXPECTED_SYNC_SCHEMA = "pc_ot_mras_c1_sync_package_dryrun_v0"
EXPECTED_GAP_SCHEMA = "pc_ot_mras_c1_remote_sync_gap_audit_v0"

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
ALLOWED_REMOTE_PREFIXES = (
    "/data/run01/sczc063/yuzibo/",
    "/home/sczc063/run/yuzibo/",
    "$HOME/run/yuzibo/",
    "~/run/yuzibo/",
)


class RemoteSyncPlanError(ValueError):
    pass


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _require(condition, message):
    if not condition:
        raise RemoteSyncPlanError(message)


def _is_relative_to(path, parent):
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
    except ValueError:
        return False
    return True


def _validate_rel_path(path):
    rel = Path(path)
    _require(not rel.is_absolute(), f"sync path must be relative: {path}")
    _require(".." not in rel.parts, f"sync path must not escape repo: {path}")
    _require(rel.suffix.lower() not in FORBIDDEN_SUFFIXES, f"forbidden artifact suffix in sync path: {path}")
    _require(not (set(rel.parts) & FORBIDDEN_PARTS), f"forbidden artifact directory in sync path: {path}")
    return rel.as_posix()


def _validate_remote_root(remote_root):
    root = str(remote_root).rstrip("/")
    normalized = root + "/"
    _require(
        any(normalized.startswith(prefix) for prefix in ALLOWED_REMOTE_PREFIXES),
        f"remote_root is outside allowed yuzibo workspace: {remote_root}",
    )
    _require(".." not in Path(root).parts, f"remote_root must not contain path escape: {remote_root}")
    return root


def _native_openssh_common_args(*, ssh_host, ssh_user, ssh_port, identity_file):
    return [
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "PubkeyAcceptedAlgorithms=+ssh-rsa",
        "-o",
        "HostkeyAlgorithms=+ssh-rsa",
        "-i",
        identity_file,
        "-p",
        str(ssh_port),
        "-l",
        ssh_user,
        ssh_host,
    ]


def _native_openssh_scp_args(*, ssh_user, ssh_port, identity_file):
    return [
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "PubkeyAcceptedAlgorithms=+ssh-rsa",
        "-o",
        "HostkeyAlgorithms=+ssh-rsa",
        "-o",
        f"User={ssh_user}",
        "-i",
        identity_file,
        "-P",
        str(ssh_port),
    ]


def _remote_join(remote_root, rel_path):
    return f"{remote_root}/{rel_path}"


def _remote_dir(remote_root, rel_path):
    return str(Path(_remote_join(remote_root, rel_path)).parent).replace("\\", "/")


def build_remote_sync_plan(
    sync_package,
    remote_gap_audit,
    *,
    generated_at="UNSET_GENERATED_AT",
    remote_root=None,
    ssh_host="ssh.cn-zhongwei-1.paracloud.com",
    ssh_user="sczc063@BSCC-N16R4",
    ssh_port=22,
    identity_file="C:\\Users\\skywalker\\.ssh\\id_rsa",
):
    _require(sync_package.get("schema_version") == EXPECTED_SYNC_SCHEMA, "unexpected sync package schema")
    _require(sync_package.get("status") == "DRYRUN_ONLY_NOT_SYNCABLE", "sync package must remain dry-run only")
    _require(sync_package.get("pass") is True, "sync package did not pass")
    unlock = sync_package.get("execution_unlock") or {}
    for key in ("remote_sync_allowed", "slurm_allowed", "tools_test_allowed", "detector_map_reporting_allowed"):
        _require(unlock.get(key) is False, f"sync package execution_unlock.{key} must be false")

    _require(remote_gap_audit.get("schema_version") == EXPECTED_GAP_SCHEMA, "unexpected remote gap schema")
    _require(
        remote_gap_audit.get("status") == "REMOTE_SYNC_GAP_CONFIRMED_EXECUTION_LOCKED",
        "remote gap audit must confirm locked sync gap",
    )
    gap_unlock = remote_gap_audit.get("execution_unlock") or {}
    for key in ("remote_sync_allowed", "slurm_allowed", "tools_test_allowed", "detector_map_reporting_allowed"):
        _require(gap_unlock.get(key) is False, f"remote gap execution_unlock.{key} must be false")

    gap_check = remote_gap_audit.get("remote_check") or {}
    target_root = _validate_remote_root(remote_root or gap_check.get("remote_root"))
    repo = Path(sync_package.get("repo", ".")).resolve()
    _require(repo.exists(), f"local repo does not exist: {repo}")

    common = _native_openssh_common_args(
        ssh_host=ssh_host,
        ssh_user=ssh_user,
        ssh_port=ssh_port,
        identity_file=identity_file,
    )
    scp_common = _native_openssh_scp_args(
        ssh_user=ssh_user,
        ssh_port=ssh_port,
        identity_file=identity_file,
    )
    files = []
    mkdir_commands = []
    copy_commands = []
    expected_sha_lines = []
    for item in sync_package.get("files") or []:
        rel_path = _validate_rel_path(item["path"])
        local_path = (repo / rel_path).resolve()
        _require(_is_relative_to(local_path, repo), f"local sync path escapes repo: {rel_path}")
        _require(local_path.exists(), f"local sync file missing: {rel_path}")
        remote_path = _remote_join(target_root, rel_path)
        remote_dir = _remote_dir(target_root, rel_path)
        files.append(
            {
                "path": rel_path,
                "local_path": str(local_path),
                "remote_path": remote_path,
                "sha256": item["sha256"],
                "size_bytes": item["size_bytes"],
            }
        )
        mkdir_commands.append(
            {
                "program": "C:\\Windows\\System32\\OpenSSH\\ssh.exe",
                "args": common + [f"mkdir -p {json.dumps(remote_dir)}"],
                "writes_remote": True,
                "purpose": f"create remote directory for {rel_path}",
            }
        )
        copy_commands.append(
            {
                "program": "C:\\Windows\\System32\\OpenSSH\\scp.exe",
                "args": scp_common
                + [
                    str(local_path),
                    f"{ssh_host}:{remote_path}",
                ],
                "writes_remote": True,
                "purpose": f"copy {rel_path}",
            }
        )
        expected_sha_lines.append(f"{item['sha256']}  {remote_path}")

    _require(files, "sync package contains no files")
    verify_script = "set -e\n" + "\n".join(f"test -f {json.dumps(row['remote_path'])}" for row in files)
    verify_script += "\n" + "\n".join(f"sha256sum {json.dumps(row['remote_path'])}" for row in files)
    expected_sha256_manifest = "\n".join(expected_sha_lines) + "\n"

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "PLAN_READY_NOT_EXECUTED",
        "route": "CTF-BDI/PC-OT-MRAS",
        "purpose": "Machine-auditable native OpenSSH plan for a later unlocked C1 reader-disabled source sync.",
        "source_sync_package_schema": sync_package["schema_version"],
        "source_remote_gap_schema": remote_gap_audit["schema_version"],
        "remote_root": target_root,
        "ssh": {
            "host": ssh_host,
            "user": ssh_user,
            "port": int(ssh_port),
            "identity_file": identity_file,
            "transport": "Windows native OpenSSH",
        },
        "files": files,
        "file_count": len(files),
        "commands": {
            "mkdir": mkdir_commands,
            "copy": copy_commands,
            "verify_after_sync": {
                "program": "C:\\Windows\\System32\\OpenSSH\\ssh.exe",
                "args": common + ["bash -s"],
                "stdin": verify_script,
                "expected_sha256_manifest": expected_sha256_manifest,
                "writes_remote": False,
            },
        },
        "execution_unlock": {
            "remote_sync_allowed": False,
            "slurm_allowed": False,
            "tools_test_allowed": False,
            "detector_map_reporting_allowed": False,
            "metric_claim_allowed": False,
            "paper_claim_allowed": False,
        },
        "required_before_execution": [
            "required_review_or_user_approved_explicit_replacement",
            "tracker_manifest_unlocked_for_c1_remote_sync",
            "operator_runs_this_plan_with_remote_write_enabled",
            "post_sync_remote_sha_verification",
            "formal_gate_json_generation",
        ],
        "non_unlocks": [
            "remote_sync",
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
    parser = argparse.ArgumentParser(description="Plan, but do not execute, a PC-OT-MRAS C1 remote source sync.")
    parser.add_argument("--sync-package", required=True)
    parser.add_argument("--remote-gap-audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--generated-at", default="UNSET_GENERATED_AT")
    parser.add_argument("--remote-root", default=None)
    parser.add_argument("--ssh-host", default="ssh.cn-zhongwei-1.paracloud.com")
    parser.add_argument("--ssh-user", default="sczc063@BSCC-N16R4")
    parser.add_argument("--ssh-port", type=int, default=22)
    parser.add_argument("--identity-file", default="C:\\Users\\skywalker\\.ssh\\id_rsa")
    return parser.parse_args()


def main():
    args = parse_args()
    payload = build_remote_sync_plan(
        _load_json(args.sync_package),
        _load_json(args.remote_gap_audit),
        generated_at=args.generated_at,
        remote_root=args.remote_root,
        ssh_host=args.ssh_host,
        ssh_user=args.ssh_user,
        ssh_port=args.ssh_port,
        identity_file=args.identity_file,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
