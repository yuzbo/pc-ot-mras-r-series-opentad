import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.acquisition.bvr_twb.adapter_bridge import ADAPTER_FIXED_LENGTH_PADDED_BRIDGE  # noqa: E402
from opentad.acquisition.bvr_twb.types import ROUTE_LABEL  # noqa: E402


EXPECTED_COMMIT = "478325af8da10646f747f955a54378d53fffd3ef"
REQUIRED_PRETRAIN_PATH = "pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"
SHORTDIAG_CONFIG = "configs/adatad/thumos/input_bvr_twb_dynamic_adapter_irregular_headv3_shortdiag.py"

FORBIDDEN_ROUTE_TOKENS = ("C3", "C3-Pro", "C3_PRO", "ABR", "MDL")
FAIL_LOG_PATTERNS = (
    (re.compile(r"no pretrain path is provided", re.IGNORECASE), "missing_pretrain_warning"),
    (re.compile(r"\bNaN\b|cost\s*[:=]\s*nan", re.IGNORECASE), "nan_marker"),
    (re.compile(r"(?<![a-z])Inf(?:inity)?(?![a-z])|cost\s*[:=]\s*inf", re.IGNORECASE), "inf_marker"),
    (re.compile(r"non[- ]finite", re.IGNORECASE), "non_finite_marker"),
    (re.compile(r"Traceback \(most recent call last\)|RuntimeError", re.IGNORECASE), "python_runtime_error"),
    (re.compile(r"CUDA out of memory|CUDA OOM", re.IGNORECASE), "cuda_oom"),
    (re.compile(r"\bKilled\b|No space left on device|no GPU|No CUDA GPUs", re.IGNORECASE), "resource_failure"),
)
EVAL_OR_CLAIM_PATTERNS = (
    (re.compile(r"\bmAP\b|Average\s+mAP|Avg[-_ ]?mAP|result_detection\.json", re.IGNORECASE), "metric_claim"),
    (re.compile(r"\beval_one_epoch\b|\[Eval\]|Running evaluation|tools/test\.py", re.IGNORECASE), "eval_marker"),
    (re.compile(r"sparse[-_ ]?compute\s+claim|FLOPs\s+claim|latency\s+claim", re.IGNORECASE), "sparse_compute_claim"),
)
FULL_TRAIN_UNLOCK_PATTERNS = (
    re.compile(r"full_train_unlocked\s*[:=]\s*true", re.IGNORECASE),
    re.compile(r"formal\s+full\s+train\s+(allowed|unlocked|approved)", re.IGNORECASE),
    re.compile(r"paper\s+claim|deploy\s+claim|deployment\s+claim", re.IGNORECASE),
)
FINITE_LOSS_RE = re.compile(r"\bLoss=([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\b")


class ShortDiagValidationError(ValueError):
    pass


def _read_text(path):
    return Path(path).read_text(encoding="utf-8", errors="replace")


def _normalize_route_label(text):
    return text.replace(ROUTE_LABEL, "ROUTE_LABEL")


def _git_head(repo_root):
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise ShortDiagValidationError(f"cannot read git HEAD: {(proc.stderr or proc.stdout).strip()}")
    return proc.stdout.strip()


def _load_config(config_path):
    try:
        from mmengine.config import Config
    except Exception as exc:
        raise ShortDiagValidationError(f"mmengine Config is required for shortdiag validation: {exc}") from exc
    try:
        return Config.fromfile(str(config_path))
    except Exception as exc:
        raise ShortDiagValidationError(f"cannot resolve config {config_path}: {exc}") from exc


def _get_nested(mapping, *keys, default=None):
    value = mapping
    for key in keys:
        if hasattr(value, "get"):
            value = value.get(key, default)
        else:
            value = getattr(value, key, default)
        if value is default:
            return default
    return value


def _require_no_forbidden_route_tokens(text, source):
    normalized = _normalize_route_label(text)
    for token in FORBIDDEN_ROUTE_TOKENS:
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])", normalized, re.IGNORECASE):
            raise ShortDiagValidationError(f"{source} contains forbidden route token: {token}")


def validate_shortdiag_config(config_path, repo_root=ROOT, expected_commit=None, require_pretrain_file=True):
    config_path = Path(config_path)
    repo_root = Path(repo_root)
    text = _read_text(config_path)
    _require_no_forbidden_route_tokens(text, "config")

    cfg = _load_config(config_path)
    if expected_commit is not None:
        head = _git_head(repo_root)
        if head != expected_commit:
            raise ShortDiagValidationError(f"git HEAD mismatch: got {head}, expected {expected_commit}")
    else:
        head = None

    marker = _get_nested(cfg, "shortdiag_expected_base_commit")
    if marker != EXPECTED_COMMIT:
        raise ShortDiagValidationError(f"config commit marker mismatch: got {marker!r}, expected {EXPECTED_COMMIT!r}")
    if _get_nested(cfg, "route_label") != ROUTE_LABEL:
        raise ShortDiagValidationError("shortdiag config must preserve the BVR-TWB route label")
    if not bool(_get_nested(cfg, "diagnostic_only")):
        raise ShortDiagValidationError("shortdiag config must set diagnostic_only=True")
    if bool(_get_nested(cfg, "full_train_unlocked", default=False)):
        raise ShortDiagValidationError("shortdiag config must keep full_train_unlocked=False")
    if bool(_get_nested(cfg, "metric_claim", default=False)):
        raise ShortDiagValidationError("shortdiag config must keep metric_claim=False")
    if bool(_get_nested(cfg, "sparse_compute_claim", default=False)):
        raise ShortDiagValidationError("shortdiag config must keep sparse_compute_claim=False")

    pretrain = _get_nested(cfg, "model", "backbone", "custom", "pretrain")
    if pretrain != REQUIRED_PRETRAIN_PATH:
        raise ShortDiagValidationError(
            f"resolved pretrain mismatch: got {pretrain!r}, expected {REQUIRED_PRETRAIN_PATH!r}"
        )
    pretrain_path = repo_root / REQUIRED_PRETRAIN_PATH
    if require_pretrain_file and not pretrain_path.is_file():
        raise ShortDiagValidationError(f"required VideoMAE-S pretrain file is missing: {pretrain_path}")

    workflow = cfg.workflow
    if int(workflow.end_epoch) != 1:
        raise ShortDiagValidationError(f"shortdiag workflow.end_epoch must be 1, got {workflow.end_epoch!r}")
    if not bool(workflow.disable_checkpoint):
        raise ShortDiagValidationError("shortdiag workflow.disable_checkpoint must be True")
    if int(workflow.val_start_epoch) <= int(workflow.end_epoch):
        raise ShortDiagValidationError("shortdiag val_start_epoch must be beyond end_epoch")
    if int(workflow.val_loss_interval) != -1 or int(workflow.val_eval_interval) != -1:
        raise ShortDiagValidationError("shortdiag validation loss/eval intervals must both be -1")
    if int(workflow.logging_interval) > 5 or int(workflow.runtime_debug_interval) > 5:
        raise ShortDiagValidationError("shortdiag logging/runtime debug intervals must stay small")

    if bool(cfg.solver.amp) or bool(cfg.solver.fp16_compress):
        raise ShortDiagValidationError("shortdiag must keep AMP and fp16 compression disabled")
    head_cfg = cfg.model.rpn_head
    expected_head = {
        "type": "IrregularActionFormerHeadV3",
        "max_reg_log_distance": 6.0,
        "regression_head_fp32": True,
        "regression_loss_fp32": True,
        "filter_invalid_regression_samples": False,
        "min_regression_segment_length": 1e-6,
    }
    for key, expected in expected_head.items():
        got = head_cfg.get(key) if hasattr(head_cfg, "get") else getattr(head_cfg, key)
        if isinstance(expected, float):
            if not math.isclose(float(got), expected, rel_tol=0.0, abs_tol=1e-12):
                raise ShortDiagValidationError(f"HeadV3 stability flag {key} mismatch: {got!r}")
        elif got != expected:
            raise ShortDiagValidationError(f"HeadV3 stability flag {key} mismatch: {got!r}")

    pretty = cfg.pretty_text
    required_tokens = (
        "bvr_twb_dynamic_subsample",
        ADAPTER_FIXED_LENGTH_PADDED_BRIDGE,
        "remap_gt_to_selected_axis=False",
        "bvr_twb_require_deploy_visible_scout=True",
        "bvr_twb_allow_diagnostic_preview_fallback=False",
        "bvr_twb_value_mode='deploy_heuristic_voi'",
    )
    compact_pretty = pretty.replace('"', "'")
    for token in required_tokens:
        if token not in compact_pretty and token not in text:
            raise ShortDiagValidationError(f"shortdiag resolved config missing BVR token: {token}")
    if "diagnostic_only" not in cfg.work_dir:
        raise ShortDiagValidationError("shortdiag work_dir must be diagnostic-only")

    return {
        "config_valid": True,
        "git_head": head,
        "resolved_pretrain": pretrain,
        "pretrain_exists": pretrain_path.is_file(),
        "workflow_end_epoch": int(workflow.end_epoch),
        "checkpoint_disabled": bool(workflow.disable_checkpoint),
        "eval_disabled": True,
        "route_label": ROUTE_LABEL,
    }


def validate_train_log(train_log):
    text = _read_text(train_log)
    _require_no_forbidden_route_tokens(text, "train_log")
    for pattern, reason in FAIL_LOG_PATTERNS:
        if pattern.search(text):
            raise ShortDiagValidationError(f"train log contains stop marker: {reason}")
    for pattern, reason in EVAL_OR_CLAIM_PATTERNS:
        if pattern.search(text):
            raise ShortDiagValidationError(f"train log contains forbidden eval/claim marker: {reason}")
    for pattern in FULL_TRAIN_UNLOCK_PATTERNS:
        if pattern.search(text):
            raise ShortDiagValidationError("train log contains forbidden full-train/deploy/paper claim")
    for token in (ROUTE_LABEL, "bvr_twb_dynamic_subsample", ADAPTER_FIXED_LENGTH_PADDED_BRIDGE):
        if token not in text:
            raise ShortDiagValidationError(f"train log missing BVR route token: {token}")
    if REQUIRED_PRETRAIN_PATH not in text:
        raise ShortDiagValidationError("train log missing required VideoMAE-S pretrain path")
    load_markers = (
        "Loads checkpoint by local backend from path",
        "load checkpoint",
        "Loaded checkpoint",
        "load pretrained",
        "load model from",
    )
    lowered = text.lower()
    if not any(marker.lower() in lowered for marker in load_markers):
        raise ShortDiagValidationError("train log does not show a real pretrained checkpoint load marker")

    finite_losses = []
    for match in FINITE_LOSS_RE.finditer(text):
        value = float(match.group(1))
        if math.isfinite(value):
            finite_losses.append(value)
    if not finite_losses:
        raise ShortDiagValidationError("train log has no finite Loss=... line")
    return {
        "train_log_valid": True,
        "finite_loss_count": len(finite_losses),
        "last_finite_loss": finite_losses[-1],
        "pretrain_load_marker_found": True,
    }


def validate_shortdiag(config_path, train_log=None, repo_root=ROOT, expected_commit=None, require_pretrain_file=True):
    result = {
        "validator": "bvr_twb_shortdiag",
        "route_label": ROUTE_LABEL,
        "expected_commit": EXPECTED_COMMIT,
        "full_train_unlocked": False,
        "metric_claim": False,
        "sparse_compute_claim": False,
        "allowed_next_action": "SHORT_DIAGNOSTIC_ONLY_REVIEW_EVIDENCE",
    }
    result.update(
        validate_shortdiag_config(
            config_path,
            repo_root=repo_root,
            expected_commit=expected_commit,
            require_pretrain_file=require_pretrain_file,
        )
    )
    if train_log is not None:
        result.update(validate_train_log(train_log))
    return result


def main():
    parser = argparse.ArgumentParser(description="Validate BVR-TWB short diagnostic config and optional train log.")
    parser.add_argument("--config", default=SHORTDIAG_CONFIG)
    parser.add_argument("--train-log")
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--expected-commit")
    parser.add_argument("--allow-missing-pretrain", action="store_true")
    args = parser.parse_args()
    try:
        result = validate_shortdiag(
            args.config,
            train_log=args.train_log,
            repo_root=args.repo_root,
            expected_commit=args.expected_commit,
            require_pretrain_file=not args.allow_missing_pretrain,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "gate_pass": False,
                    "validator": "bvr_twb_shortdiag",
                    "route_label": ROUTE_LABEL,
                    "full_train_unlocked": False,
                    "metric_claim": False,
                    "sparse_compute_claim": False,
                    "reason": str(exc),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        raise SystemExit(1)
    result["gate_pass"] = True
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
