from __future__ import annotations

import argparse
import importlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


SCHEMA_VERSION = "pc_ot_mras_p2_quality_rank_calibrator_full_env_precheck_v0"
PASS_STATUS = "P2_QUALITY_RANK_CALIBRATOR_FULL_ENV_PRECHECK_PASS"
HOLD_STATUS = "P2_QUALITY_RANK_CALIBRATOR_FULL_ENV_PRECHECK_HOLD"

DEFAULT_CONFIG = "configs/adatad/thumos/ctf_bdi_pc_ot_mras_p2_quality_rank_calibrator_v0_local.py"
DEFAULT_RUNTIME_SMOKE = "tests/test_pc_ot_mras_p2_quality_calibration_runtime_smoke.py"

DENIED_GATE_FLAGS = (
    "allow_detector_training",
    "allow_remote_sync",
    "allow_precheck_only",
    "allow_slurm",
    "allow_gpu",
    "allow_tools_train",
    "allow_tools_test",
    "allow_detector_map",
    "launch_gate_passed",
    "metric_claim_allowed",
    "paper_claim_allowed",
    "runtime_flops_claim_allowed",
    "deploy_claim_allowed",
)

REQUIRED_IMPORTS = (
    "torch",
    "mmcv.cnn",
    "mmengine.config",
    "mmaction.registry",
    "opentad.models.dense_heads.native_irregular_area_head_p2",
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def ensure_repo_on_path() -> None:
    repo = str(REPO_ROOT)
    if repo not in sys.path:
        sys.path.insert(0, repo)


def strict_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json_value(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    return str(value)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(strict_json_value(dict(payload)), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def check_imports(
    modules: Sequence[str] = REQUIRED_IMPORTS,
    importer: Callable[[str], Any] = importlib.import_module,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    all_pass = True
    for name in modules:
        try:
            module = importer(name)
        except Exception as exc:  # noqa: BLE001 - diagnostic should preserve all import failures.
            checks[name] = {
                "pass": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            all_pass = False
            continue
        checks[name] = {
            "pass": True,
            "version": str(getattr(module, "__version__", "")),
        }
    return {"pass": all_pass, "checks": checks}


def parse_config(config_path: str | Path) -> dict[str, Any]:
    try:
        from mmengine.config import Config

        cfg = Config.fromfile(str(config_path))
    except Exception as exc:  # noqa: BLE001 - precheck must fail closed with error detail.
        return {
            "pass": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    gate = cfg.get("p2_quality_rank_calibrator_v0_gate", None)
    quality_cfg = cfg.get("model", {}).get("rpn_head", {}).get("area_head", {}).get("quality_calibration", None)
    if not isinstance(gate, Mapping):
        return {"pass": False, "error": "missing p2_quality_rank_calibrator_v0_gate"}
    if not isinstance(quality_cfg, Mapping):
        return {"pass": False, "error": "missing model.rpn_head.area_head.quality_calibration"}

    denied_flags = {key: bool(gate.get(key, True)) for key in DENIED_GATE_FLAGS}
    denied_flags_pass = all(value is False for value in denied_flags.values())
    quality_pass = bool(quality_cfg.get("enable", False)) is True
    allowed_checks = tuple(gate.get("allowed_checks", ()))
    forbidden_checks = tuple(gate.get("forbidden_checks", ()))
    config_pass = denied_flags_pass and quality_pass and bool(gate.get("local_synthetic_gate_only", False))
    return {
        "pass": config_pass,
        "work_dir": str(cfg.get("work_dir", "")),
        "gate": {
            "stage": str(gate.get("stage", "")),
            "local_synthetic_gate_only": bool(gate.get("local_synthetic_gate_only", False)),
            "denied_flags_pass": denied_flags_pass,
            "denied_flags": denied_flags,
            "allowed_checks": list(allowed_checks),
            "forbidden_checks": list(forbidden_checks),
        },
        "quality_calibration": {
            "enable": bool(quality_cfg.get("enable", False)),
            "hidden_dim": quality_cfg.get("hidden_dim"),
            "quality_loss_weight": quality_cfg.get("quality_loss_weight"),
            "boundary_loss_weight": quality_cfg.get("boundary_loss_weight"),
            "rank_loss_weight": quality_cfg.get("rank_loss_weight"),
        },
    }


def run_runtime_smoke(
    test_path: str | Path,
    *,
    run: bool,
    command_prefix: Sequence[str] | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    if not run:
        return {
            "pass": False,
            "ran": False,
            "decision": "RUNTIME_SMOKE_NOT_REQUESTED",
        }
    command = list(command_prefix or [sys.executable, "-m", "pytest"])
    command.extend([str(test_path), "-q"])
    proc = runner(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return {
        "pass": proc.returncode == 0,
        "ran": True,
        "command": command,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def run_precheck(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    runtime_smoke_path: str | Path = DEFAULT_RUNTIME_SMOKE,
    run_smoke: bool = False,
    runtime_command_prefix: Sequence[str] | None = None,
) -> dict[str, Any]:
    ensure_repo_on_path()
    imports = check_imports()
    config = parse_config(config_path)
    runtime = run_runtime_smoke(
        runtime_smoke_path,
        run=run_smoke,
        command_prefix=runtime_command_prefix,
    )
    full_pass = bool(imports["pass"]) and bool(config["pass"]) and bool(runtime["pass"])
    blockers = []
    if not imports["pass"]:
        blockers.append("FULL_IMPORT_DEPENDENCY_CHECK_FAILED")
    if not config["pass"]:
        blockers.append("CONFIG_OR_GATE_CHECK_FAILED")
    if not runtime["pass"]:
        blockers.append("RUNTIME_SMOKE_FAILED_OR_NOT_RUN")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": PASS_STATUS if full_pass else HOLD_STATUS,
        "pass": full_pass,
        "blockers": blockers,
        "python": sys.executable,
        "config_path": str(config_path),
        "runtime_smoke_path": str(runtime_smoke_path),
        "imports": imports,
        "config": config,
        "runtime_smoke": runtime,
        "permissions": {
            "remote_sync_allowed": False,
            "remote_precheck_allowed": False,
            "slurm_allowed": False,
            "tools_train_allowed": False,
            "tools_test_allowed": False,
            "metric_claim_allowed": False,
            "paper_claim_allowed": False,
            "deploy_claim_allowed": False,
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--runtime-smoke", default=DEFAULT_RUNTIME_SMOKE)
    parser.add_argument("--run-runtime-smoke", action="store_true")
    parser.add_argument(
        "--runtime-command-prefix",
        nargs="+",
        default=None,
        help="Command prefix before <runtime-smoke> -q, e.g. python -m pytest.",
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--require-pass", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_precheck(
        config_path=args.config,
        runtime_smoke_path=args.runtime_smoke,
        run_smoke=args.run_runtime_smoke,
        runtime_command_prefix=args.runtime_command_prefix,
    )
    if args.output_json:
        write_json(args.output_json, result)
    print(json.dumps(strict_json_value(result), indent=2, sort_keys=True))
    return 0 if result["pass"] or not args.require_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
