from __future__ import annotations

import argparse
import json
import math
import re
import runpy
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.acquisition.mdl_knot import MDL_KNOT_ROUTE_LABEL  # noqa: E402
from tools.mdl_knot.validate_mdl_knot_launch_gate import (  # noqa: E402
    FORBIDDEN_ROUTE_DRIFT,
    _validate_config as _validate_precheck_config,
)

FATAL_LOG_PATTERNS = (
    r"\btraceback\b",
    r"\bruntimeerror\b",
    r"\bcuda\s+out\s+of\s+memory\b",
    r"\bout\s+of\s+memory\b",
    r"\boom\b",
    r"\bkilled\b",
    r"\bno\s+space\s+left\b",
    r"\bno\s+gpu\b",
    r"\bno\s+cuda\b",
)

EVAL_OR_CLAIM_PATTERNS = (
    r"\btools/test\.py\b",
    r"\bresult_detection\.json\b",
    r"\bmap(@|\b)",
    r"\beval(?:uate|uation)?\b(?!\s*(?:locked|disabled|off|false|not|no|without))",
    r"\bcheckpoint\b(?!\s*(?:locked|disabled|off|false|not|no|without))",
    r"\bfull[_ -]?train[_ -]?unlocked\s*[:=]\s*true\b",
    r"\bformal[_ -]?full[_ -]?train\b",
    r"\bdeploy(?:ment)?\s+claim\b",
    r"\bpaper\s+claim\b",
    r"\bruntime\s+claim\b",
    r"\bsparse[_ -]?compute[_ -]?claim\s*[:=]\s*true\b",
    r"\bmetric[_ -]?claim\s*[:=]\s*true\b",
)

ALLOWED_LOG_SCHEDULE_PATTERNS = (
    r"\bcheckpoint[_ -]?interval\b\s*[:=]?\s*-?\d+",
    r"\beval[_ -]?interval\b\s*[:=]?\s*-?\d+",
    r"\bevaluation[_ -]?interval\b\s*[:=]?\s*-?\d+",
    r"\bval[_ -]?eval[_ -]?interval\b\s*[:=]?\s*-?\d+",
    r"\bval[_ -]?loss[_ -]?interval\b\s*[:=]?\s*-?\d+",
    r"\blogging[_ -]?interval\b\s*[:=]?\s*-?\d+",
)

LOSS_RE = re.compile(
    r"(?<![a-z0-9_])(?:loss|loss_[a-z0-9_]*|cost)(?![a-z0-9_])\s*[:=]?\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail-closed MDL-Knot SHORT_DIAGNOSTIC_ONLY gate.")
    parser.add_argument("--config", required=True, help="MDL-Knot short diagnostic config to validate.")
    parser.add_argument("--route-label", default=MDL_KNOT_ROUTE_LABEL)
    parser.add_argument(
        "--train-log",
        default=None,
        help="One-epoch diagnostic train log to validate as execution evidence. Omit for static config check only.",
    )
    return parser.parse_args()


def _locked(message: str, code: int = 2) -> int:
    print(f"LOCKED: {message}")
    print(
        "Still locked: full training, evaluation, checkpoints, tools/test.py, "
        "mAP/runtime/FLOPs/deploy/paper/sparse-compute claims"
    )
    return code


def _as_path(path_arg: str) -> Path:
    path = Path(path_arg)
    if not path.is_absolute():
        path = ROOT / path
    return path


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if key.startswith("__"):
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config_with_base(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_cfg = runpy.run_path(str(config_path))
    bases = raw_cfg.get("_base_", [])
    if isinstance(bases, str):
        bases = [bases]

    merged: dict[str, Any] = {}
    for base in bases:
        base_path = (config_path.parent / base).resolve()
        base_merged, _ = _load_config_with_base(base_path)
        merged = _deep_merge(merged, base_merged)
    merged = _deep_merge(merged, raw_cfg)
    return merged, raw_cfg


def _validate_shortdiag_config(config_path: Path, cfg: dict[str, Any], raw_cfg: dict[str, Any]) -> tuple[int, dict[str, Any] | None]:
    if raw_cfg.get("_base_") != ["./input_mdl_knot_dynamic_adapter_irregular_headv3.py"]:
        return _locked("shortdiag config must extend the existing MDL-Knot config"), None
    if cfg.get("route_label") != MDL_KNOT_ROUTE_LABEL:
        return _locked(f"route_label mismatch: {cfg.get('route_label')}"), None

    config_text = config_path.read_text(encoding="utf-8").replace(MDL_KNOT_ROUTE_LABEL, "")
    route_hits = [token for token in FORBIDDEN_ROUTE_DRIFT if token in config_text.upper()]
    if route_hits:
        return _locked(f"route drift tokens in shortdiag config: {route_hits}"), None

    precheck_status, precheck_evidence = _validate_precheck_config(config_path, cfg)
    if precheck_status != 0:
        return _locked("base MDL-Knot launch gate did not pass for merged shortdiag config"), None

    gate = cfg.get("shortdiag_gate", {})
    required_false = ("full_train_unlocked", "metric_claim", "sparse_compute_claim")
    for key in required_false:
        if gate.get(key) is not False or cfg.get(key) is not False:
            return _locked(f"shortdiag lock must be false: {key}"), None
    required_true = (
        "diagnostic_only",
        "evaluation_locked",
        "checkpoint_locked",
        "tools_test_py_locked",
        "map_claim_locked",
        "claim_locked",
        "no_eval",
        "no_checkpoint",
        "no_tools_test_py",
        "no_map",
        "no_metric_claim",
        "no_runtime_claim",
        "no_deploy_claim",
        "no_paper_claim",
    )
    for key in required_true:
        if gate.get(key) is not True:
            return _locked(f"shortdiag gate flag must be true: {key}"), None
    if cfg.get("diagnostic_only") is not True:
        return _locked("top-level diagnostic_only must be true"), None
    if gate.get("max_epochs") != 1 or cfg.get("total_epochs") != 1 or cfg.get("max_epochs") != 1:
        return _locked("shortdiag must be exactly one epoch"), None
    if cfg.get("workflow") != [("train", 1)]:
        return _locked("shortdiag workflow must be train-only for one epoch"), None

    acq = cfg.get("mdl_knot_acquisition", {})
    if acq.get("deploy_scout_source") != "raw_frame_motion_scout_with_metadata_fallback":
        return _locked("shortdiag must use raw_frame_motion_scout_with_metadata_fallback"), None
    if acq.get("synthetic_fallback_allowed") is not False:
        return _locked("synthetic formal fallback must be disabled"), None
    for key in ("diagnostic_only",):
        if acq.get(key) is not True:
            return _locked(f"acquisition flag must be true: {key}"), None
    for key in required_false:
        if acq.get(key) is not False:
            return _locked(f"acquisition claim lock must be false: {key}"), None

    evaluation = cfg.get("evaluation", {})
    checkpoint = cfg.get("checkpoint", {})
    if evaluation.get("shortdiag_disabled") is not True:
        return _locked("evaluation must be shortdiag-disabled")
    if checkpoint.get("shortdiag_disabled") is not True or checkpoint.get("save_last") is not False:
        return _locked("checkpoint saving must be disabled")

    evidence = {
        "config_path": str(config_path),
        "route_label": cfg.get("route_label"),
        "route_status": cfg.get("route_status"),
        "diagnostic_only": cfg.get("diagnostic_only"),
        "full_train_unlocked": cfg.get("full_train_unlocked"),
        "metric_claim": cfg.get("metric_claim"),
        "sparse_compute_claim": cfg.get("sparse_compute_claim"),
        "max_epochs": cfg.get("max_epochs"),
        "workflow": cfg.get("workflow"),
        "deploy_scout_source": acq.get("deploy_scout_source"),
        "synthetic_fallback_allowed": acq.get("synthetic_fallback_allowed"),
        "base_precheck_evidence": precheck_evidence,
    }
    return 0, evidence


def _has_pattern(patterns: tuple[str, ...], text: str) -> str | None:
    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return pattern
    return None


def _strip_allowed_log_schedule_terms(text: str) -> str:
    cleaned = text
    for pattern in ALLOWED_LOG_SCHEDULE_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    return cleaned


def _validate_train_log(train_log_arg: str) -> tuple[int, dict[str, Any] | None]:
    log_path = _as_path(train_log_arg)
    if not log_path.exists():
        return _locked(f"missing train log: {log_path}"), None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    normalized = text.replace(MDL_KNOT_ROUTE_LABEL, "")
    lower = normalized.lower()

    fatal = _has_pattern(FATAL_LOG_PATTERNS, lower)
    if fatal:
        return _locked(f"fatal train-log marker: {fatal}"), None
    if re.search(r"(?<![a-z0-9_])(?:nan|\+?inf|-inf|infinity)(?![a-z0-9_])", lower):
        return _locked("non-finite train-log numeric marker"), None
    safety_text = _strip_allowed_log_schedule_terms(normalized)
    route_hits = [token for token in FORBIDDEN_ROUTE_DRIFT if token in safety_text.upper()]
    if route_hits:
        return _locked(f"route drift tokens in train log: {route_hits}"), None
    marker_text = _strip_allowed_log_schedule_terms(lower)
    for allowed_phrase in (
        "without evaluation",
        "no evaluation",
        "evaluation locked",
        "evaluation disabled",
        "without checkpoint",
        "no checkpoint",
        "checkpoint locked",
        "checkpoint disabled",
    ):
        marker_text = marker_text.replace(allowed_phrase, "")
    marker = _has_pattern(EVAL_OR_CLAIM_PATTERNS, marker_text)
    if marker:
        return _locked(f"evaluation/checkpoint/claim marker in train log: {marker}"), None

    losses = [float(match.group(1)) for match in LOSS_RE.finditer(text)]
    finite_losses = [value for value in losses if math.isfinite(value)]
    if not finite_losses:
        return _locked("train log provided but no finite Loss value was found"), None
    if len(finite_losses) != len(losses):
        return _locked("train log contains non-finite loss values"), None

    epoch_numbers = [int(value) for value in re.findall(r"epoch\s*\[?(\d+)", lower)]
    if epoch_numbers and max(epoch_numbers) > 1:
        return _locked(f"short diagnostic log exceeds one epoch: {max(epoch_numbers)}"), None

    return 0, {
        "train_log": str(log_path),
        "finite_loss_count": len(finite_losses),
        "loss_min": min(finite_losses),
        "loss_max": max(finite_losses),
        "epoch_max": max(epoch_numbers) if epoch_numbers else 1,
    }


def main() -> int:
    args = parse_args()
    if args.route_label != MDL_KNOT_ROUTE_LABEL:
        return _locked(f"route label mismatch: {args.route_label}")

    config_path = _as_path(args.config)
    if not config_path.exists():
        return _locked(f"missing config: {config_path}")
    try:
        cfg, raw_cfg = _load_config_with_base(config_path)
    except Exception as exc:
        return _locked(f"cannot load config: {exc}")

    status, evidence = _validate_shortdiag_config(config_path, cfg, raw_cfg)
    if status != 0:
        return status

    log_evidence = None
    if args.train_log:
        status, log_evidence = _validate_train_log(args.train_log)
        if status != 0:
            return status

    evidence = dict(evidence or {})
    evidence["validated"] = log_evidence is not None
    evidence["formal_train_unlocked"] = False
    evidence["no_sparse_compute_claim"] = True
    evidence["log_evidence"] = log_evidence
    evidence["execution_evidence_required_for_formal_readiness"] = log_evidence is None
    evidence["evidence_scope"] = "one_epoch_train_log" if log_evidence is not None else "static_config_only"
    if log_evidence is None:
        print("SHORT_DIAGNOSTIC_CONFIG_STATIC_CHECK_ALLOWED")
    else:
        print("SHORT_DIAGNOSTIC_ONLY_REQUEST_ALLOWED")
    print("SHORTDIAG_EVIDENCE=" + json.dumps(evidence, sort_keys=True))
    print(
        "Still locked: full training, evaluation, checkpoints, tools/test.py, "
        "mAP and sparse-compute claims remain locked"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
