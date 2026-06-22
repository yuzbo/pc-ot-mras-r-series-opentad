import argparse
import ast
import hashlib
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = "pc_ot_mras_local_lowmem_eval_gate_prep_v0"
GATE_SCHEMA_VERSION = "pc_ot_mras_local_lowmem_eval_gate_v0"
DEFAULT_WSL_LOG_ROOT = "/home/skywalker/tad_local_runs"

DEFAULT_MANIFEST_PATHS = (
    "tools/test.py",
    "opentad/cores/test_engine.py",
    "opentad/utils/training_guard.py",
    "opentad/models/detectors/actionformer.py",
    "opentad/models/selectors/pc_ot_mras_reader.py",
    "opentad/models/necks/pc_ot_mras_detector_bridge.py",
    "opentad/models/dense_heads/native_irregular_area_head_p2.py",
    "opentad/models/utils/temporal_grid.py",
)

FORBIDDEN_TRUE_KEYS = (
    "tools_train",
    "direct_tools_train",
    "raw_prediction_cache",
    "paper_claim",
    "runtime_flops_claim",
    "deploy_claim",
    "dynamic_budget_claim",
)


class LocalLowmemEvalGateError(RuntimeError):
    def __init__(self, code, message, *, details=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

    def to_payload(self):
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "FAILED",
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            },
        }


def sha256_file(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _repo_root():
    return Path(__file__).resolve().parents[2]


def _posix_rel(path, root):
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return str(Path(path).resolve()).replace("\\", "/")


def _require_file(path, label):
    path = Path(path)
    if not path.is_file():
        raise LocalLowmemEvalGateError(
            "MISSING_FILE",
            f"missing required {label}: {path}",
            details={"label": label, "path": str(path)},
        )
    return path


def _parse_cfg_options(items):
    result = {}
    for item in items or []:
        if "=" not in item:
            raise LocalLowmemEvalGateError(
                "INVALID_CFG_OPTION",
                f"--cfg-options item must be key=value: {item}",
                details={"item": item},
            )
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise LocalLowmemEvalGateError(
                "INVALID_CFG_OPTION",
                f"--cfg-options item has an empty key: {item}",
                details={"item": item},
            )
        try:
            parsed = ast.literal_eval(value)
        except Exception:
            parsed = value
        result[key] = parsed
    return result


def _load_mmengine_config_cls():
    try:
        from mmengine.config import Config
    except ImportError as exc:
        raise LocalLowmemEvalGateError(
            "MISSING_MMENGINE",
            "mmengine is required to resolve and dump config text, but it is not importable.",
            details={"missing_package": "mmengine", "hint": "Install project requirements or run inside the OpenTAD env."},
        ) from exc
    return Config


def _resolved_config_text(config_path, cfg_options, *, config_cls=None):
    Config = config_cls or _load_mmengine_config_cls()
    cfg = Config.fromfile(str(config_path))
    parsed_options = _parse_cfg_options(cfg_options)
    if parsed_options:
        cfg.merge_from_dict(parsed_options)
    return cfg.pretty_text.rstrip() + "\n", parsed_options


def _config_dependency_files(config_path):
    seen = []

    def visit(path):
        path = Path(path).resolve()
        if path in seen:
            return
        _require_file(path, "config dependency")
        seen.append(path)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(target, ast.Name) and target.id == "_base_" for target in node.targets):
                continue
            try:
                value = ast.literal_eval(node.value)
            except Exception as exc:
                raise LocalLowmemEvalGateError(
                    "UNREADABLE_CONFIG_BASE",
                    f"could not parse _base_ in config: {path}",
                    details={"path": str(path), "reason": str(exc)},
                ) from exc
            bases = [value] if isinstance(value, str) else list(value)
            for base in bases:
                visit(path.parent / base)

    visit(config_path)
    return seen


def _manifest_entries(repo, config_path, checkpoint_path, resolved_config_path, extra_paths=()):
    repo = Path(repo).resolve()
    paths = []
    paths.extend(_config_dependency_files(config_path))
    paths.append(Path(checkpoint_path).resolve())
    paths.append(Path(resolved_config_path).resolve())
    for rel_path in DEFAULT_MANIFEST_PATHS:
        paths.append(repo / rel_path)
    for rel_or_abs in extra_paths or ():
        path = Path(rel_or_abs)
        paths.append(path if path.is_absolute() else repo / path)

    unique = []
    seen = set()
    for path in paths:
        path = Path(path).resolve()
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        _require_file(path, "manifest input")
        unique.append(path)

    entries = []
    for path in unique:
        entries.append(
            {
                "path": _posix_rel(path, repo),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return entries


def _write_manifest(path, entries):
    lines = [f"{item['sha256']}  {item['path']}" for item in entries]
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    return text


def _to_wsl_path(path):
    text = str(path).replace("\\", "/")
    if len(text) >= 3 and text[1] == ":" and text[2] == "/":
        drive = text[0].lower()
        return f"/mnt/{drive}{text[2:]}"
    return text


def _shell_export_line(key, value):
    return f"export {key}={shlex.quote(str(value))}"


def _build_env_text(gate_path, gate_sha256, manifest_sha256, resolved_config_sha256):
    lines = (
        "# Source this file inside WSL before running tools/test.py.",
        _shell_export_line("OPENTAD_PCOTMRAS_ENTRYPOINT_GATE_JSON", _to_wsl_path(gate_path)),
        _shell_export_line("OPENTAD_PCOTMRAS_ENTRYPOINT_GATE_SHA256", gate_sha256),
        _shell_export_line("OPENTAD_PCOTMRAS_ACTIVE_MANIFEST_SHA256", manifest_sha256),
        _shell_export_line("OPENTAD_PCOTMRAS_RESOLVED_CONFIG_SHA256", resolved_config_sha256),
    )
    return "\n".join(lines) + "\n"


def _format_cfg_options_for_command(cfg_options):
    if not cfg_options:
        return ""
    return " \\\n  --cfg-options " + " ".join(shlex.quote(item) for item in cfg_options)


def _build_run_command(
    *,
    repo,
    config_path,
    checkpoint_path,
    env_path,
    wsl_log_root,
    run_name,
    eval_id,
    seed,
    master_port,
    cfg_options,
):
    log_root = f"{str(wsl_log_root).rstrip('/')}/{run_name}"
    stdout_path = f"{log_root}/eval_stdout.log"
    work_dir = f"{log_root}/eval_workdir"
    command_cfg_options = list(cfg_options or []) + [f"work_dir={work_dir}"]
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"cd {shlex.quote(_to_wsl_path(repo))}",
            f"mkdir -p {shlex.quote(log_root)} {shlex.quote(work_dir)}",
            f"source {shlex.quote(_to_wsl_path(env_path))}",
            "# Logs stay on WSL ext4; copy a summary back to the repo after the run if needed.",
            "torchrun --nproc_per_node=1 --master_port=${MASTER_PORT:-"
            + shlex.quote(str(master_port))
            + "} \\",
            f"  tools/test.py {shlex.quote(_to_wsl_path(config_path))} --checkpoint {shlex.quote(_to_wsl_path(checkpoint_path))} --id {int(eval_id)} --seed {int(seed)}"
            + _format_cfg_options_for_command(command_cfg_options)
            + " \\",
            f"  2>&1 | tee {shlex.quote(stdout_path)}",
        ]
    ) + "\n"


def _gate_payload(
    *,
    decision,
    config_path,
    checkpoint_path,
    checkpoint_sha256,
    checkpoint_size,
    resolved_config_path,
    resolved_config_sha256,
    manifest_path,
    manifest_sha256,
    cfg_options_dict,
    generated_at,
):
    return {
        "schema_version": GATE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "decision": decision,
        "local_lowmem_eval": True,
        "tools_test": True,
        "tools_train": False,
        "direct_tools_train": False,
        "raw_prediction_cache": False,
        "paper_claim": False,
        "runtime_flops_claim": False,
        "deploy_claim": False,
        "dynamic_budget_claim": False,
        "metric_claim_allowed": False,
        "paper_claim_allowed": False,
        "runtime_flops_claim_allowed": False,
        "deploy_claim_allowed": False,
        "completed_training_evidence": True,
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_size_bytes": checkpoint_size,
        "resolved_config": str(resolved_config_path),
        "resolved_config_sha256": resolved_config_sha256,
        "active_sha256_manifest": str(manifest_path),
        "active_sha256_manifest_sha256": manifest_sha256,
        "cfg_options": cfg_options_dict,
        "forbidden_true_keys": list(FORBIDDEN_TRUE_KEYS),
        "notes": (
            "Local WSL low-memory tools/test.py preparation gate only. This file does not "
            "approve training, raw-prediction cache use, metric/paper/runtime/deploy claims, "
            "or any model/config/launcher change."
        ),
    }


def prepare_local_lowmem_eval_gate(
    *,
    config,
    checkpoint,
    run_dir,
    decision,
    cfg_options=(),
    repo=None,
    wsl_log_root=DEFAULT_WSL_LOG_ROOT,
    eval_id=0,
    seed=42,
    master_port=29517,
    generated_at=None,
    config_cls=None,
):
    repo = Path(repo or _repo_root()).resolve()
    config_path = _require_file(Path(config), "config").resolve()
    checkpoint_path = _require_file(Path(checkpoint), "checkpoint").resolve()
    if not str(config_path).lower().startswith(str(repo).lower()):
        raise LocalLowmemEvalGateError(
            "CONFIG_OUTSIDE_REPO",
            f"config must be inside repo: {config_path}",
            details={"repo": str(repo), "config": str(config_path)},
        )
    if not decision:
        raise LocalLowmemEvalGateError("MISSING_DECISION", "decision must be non-empty")

    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    generated_at = generated_at or datetime.now(timezone.utc).isoformat()

    resolved_config_path = run_dir / "resolved_config.py"
    manifest_path = run_dir / "active_sha256_manifest.txt"
    gate_path = run_dir / "gate.json"
    env_path = run_dir / "eval_gate_env.sh"
    run_command_path = run_dir / "suggested_wsl_run_command.sh"

    resolved_text, cfg_options_dict = _resolved_config_text(config_path, cfg_options, config_cls=config_cls)
    resolved_config_path.write_text(resolved_text, encoding="utf-8")
    resolved_config_sha256 = sha256_file(resolved_config_path)

    entries = _manifest_entries(repo, config_path, checkpoint_path, resolved_config_path)
    _write_manifest(manifest_path, entries)
    manifest_sha256 = sha256_file(manifest_path)

    checkpoint_sha256 = sha256_file(checkpoint_path)
    gate = _gate_payload(
        decision=decision,
        config_path=_posix_rel(config_path, repo),
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_size=checkpoint_path.stat().st_size,
        resolved_config_path=resolved_config_path,
        resolved_config_sha256=resolved_config_sha256,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        cfg_options_dict=cfg_options_dict,
        generated_at=generated_at,
    )
    gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    gate_sha256 = sha256_file(gate_path)

    env_path.write_text(
        _build_env_text(gate_path, gate_sha256, manifest_sha256, resolved_config_sha256),
        encoding="utf-8",
    )
    run_command_text = _build_run_command(
        repo=repo,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        env_path=env_path,
        wsl_log_root=wsl_log_root,
        run_name=run_dir.name,
        eval_id=eval_id,
        seed=seed,
        master_port=master_port,
        cfg_options=list(cfg_options or ()),
    )
    run_command_path.write_text(run_command_text, encoding="utf-8")

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PREPARED_NOT_LAUNCHED",
        "default_prepare_only": True,
        "launch_executed": False,
        "repo": str(repo),
        "decision": decision,
        "outputs": {
            "gate_json": str(gate_path),
            "gate_json_sha256": gate_sha256,
            "resolved_config": str(resolved_config_path),
            "resolved_config_sha256": resolved_config_sha256,
            "active_sha256_manifest": str(manifest_path),
            "active_sha256_manifest_sha256": manifest_sha256,
            "env_file": str(env_path),
            "suggested_wsl_run_command": str(run_command_path),
        },
        "manifest_entries": entries,
        "suggested_wsl": {
            "log_root": f"{str(wsl_log_root).rstrip('/')}/{run_dir.name}",
            "stdout_log": f"{str(wsl_log_root).rstrip('/')}/{run_dir.name}/eval_stdout.log",
            "run_command": run_command_text,
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare a local WSL low-memory PC-OT-MRAS eval gate.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--decision", required=True)
    parser.add_argument("--cfg-options", nargs="*", default=())
    parser.add_argument("--repo", default=None)
    parser.add_argument("--wsl-log-root", default=DEFAULT_WSL_LOG_ROOT)
    parser.add_argument("--eval-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--master-port", type=int, default=29517)
    parser.add_argument("--generated-at", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        payload = prepare_local_lowmem_eval_gate(
            config=args.config,
            checkpoint=args.checkpoint,
            run_dir=args.run_dir,
            decision=args.decision,
            cfg_options=args.cfg_options,
            repo=args.repo,
            wsl_log_root=args.wsl_log_root,
            eval_id=args.eval_id,
            seed=args.seed,
            master_port=args.master_port,
            generated_at=args.generated_at,
        )
    except LocalLowmemEvalGateError as exc:
        print(json.dumps(exc.to_payload(), indent=2, sort_keys=True))
        raise SystemExit(1)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
