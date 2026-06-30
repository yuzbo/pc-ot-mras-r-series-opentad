import argparse
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.acquisition.bvr_twb.adapter_bridge import ADAPTER_FIXED_LENGTH_PADDED_BRIDGE
from opentad.acquisition.bvr_twb.types import ROUTE_LABEL
from tools.bvr_twb.validate_bvr_twb_launch_gate import validate_launch_gate
from tools.bvr_twb.validate_bvr_twb_shortdiag import REQUIRED_PRETRAIN_PATH, validate_train_log


FINITE_REG_LOSS_RE = re.compile(r"\breg_loss=([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\b")
KEY_VALUE_RE = re.compile(r"([A-Za-z0-9_.]+)=([^| ]+)")
GRADIENT_EVIDENCE_TOKENS = (
    "finite_gradients=true",
    "no_skipped_optimizer_step=true",
    "no_skipped_reg_head=true",
)
FORMAL_STOP_PATTERNS = (
    (re.compile(r"non[- ]finite gradients detected", re.IGNORECASE), "non_finite_gradient_marker"),
    (re.compile(r"skip optimizer step", re.IGNORECASE), "skipped_optimizer_step_marker"),
    (re.compile(r"reg(?:ression)?[_ -]?head.*skipp?ed|skipp?ed.*reg(?:ression)?[_ -]?head", re.IGNORECASE), "skipped_reg_head_marker"),
    (re.compile(r"head_v3_regression_samples_kept_after_filter=0(?![0-9])"), "zero_regression_samples_kept"),
    (re.compile(r"head_v2_reg_points_total=0(?![0-9])"), "zero_regression_points"),
)


class FormalReadinessError(ValueError):
    pass


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _require(condition, message):
    if not condition:
        raise FormalReadinessError(message)


def _parse_runtime_debug_lines(text):
    rows = []
    for line in text.splitlines():
        if "[Train][RuntimeDebug]" not in line:
            continue
        row = {match.group(1): match.group(2).strip(",") for match in KEY_VALUE_RE.finditer(line)}
        rows.append(row)
    return rows


def _bool_value(row, key):
    value = str(row.get(key, "")).lower()
    if value == "true":
        return True
    if value == "false":
        return False
    return None


def _int_value(row, key):
    try:
        return int(float(str(row.get(key, ""))))
    except ValueError:
        return None


def validate_linux_geometry_summary(summary):
    _require(summary.get("validator") == "bvr_twb_geometry_contracts", "geometry summary has wrong validator")
    _require(str(summary.get("platform_system", "")).lower() == "linux", "formal readiness requires Linux geometry precheck")
    _require(summary.get("source_contract") == "passed", "geometry source contract did not pass")
    _require(summary.get("numpy_bridge_contract") == "passed", "geometry numpy bridge contract did not pass")
    _require(summary.get("torch_runtime_skipped") is False, "formal readiness rejects skipped torch runtime geometry")
    _require(summary.get("torch_runtime_contract") == "passed", "formal readiness requires torch runtime geometry pass")
    _require(summary.get("no_training") is True, "geometry precheck must be no-training")
    _require(summary.get("no_metric_claim") is True, "geometry precheck must be no-metric")
    _require(summary.get("full_training_unlocked") is False, "geometry precheck must not unlock full training")
    return True


def validate_formal_train_log(train_log):
    text = Path(train_log).read_text(encoding="utf-8", errors="replace")
    validate_train_log(train_log)
    for pattern, reason in FORMAL_STOP_PATTERNS:
        for line in text.splitlines():
            if "[bvr_twb_formal_precheck]" in line.lower():
                continue
            if pattern.search(line):
                raise FormalReadinessError(f"formal readiness log contains stop marker: {reason}")

    reg_losses = [float(match.group(1)) for match in FINITE_REG_LOSS_RE.finditer(text)]
    if not reg_losses or not all(math.isfinite(value) for value in reg_losses):
        raise FormalReadinessError("formal readiness requires finite reg_loss evidence")

    runtime_rows = _parse_runtime_debug_lines(text)
    if not runtime_rows:
        raise FormalReadinessError("formal readiness requires [Train][RuntimeDebug] HeadV3 evidence")
    accepted_runtime_row = None
    for row in runtime_rows:
        if _bool_value(row, "head_v3_regression_head_fp32_enabled") is not True:
            continue
        if _bool_value(row, "head_v3_regression_loss_fp32_enabled") is not True:
            continue
        if (_int_value(row, "head_v3_regression_samples_kept_after_filter") or 0) <= 0:
            continue
        if (_int_value(row, "head_v2_reg_points_total") or 0) <= 0:
            continue
        accepted_runtime_row = row
        break
    if accepted_runtime_row is None:
        raise FormalReadinessError("formal readiness requires HeadV3 runtime evidence with nonzero kept regression samples")

    gradient_evidence_line = None
    for line in text.splitlines():
        normalized = line.lower()
        if "[bvr_twb_formal_precheck]" in normalized and all(token in normalized for token in GRADIENT_EVIDENCE_TOKENS):
            gradient_evidence_line = line
            break
    if gradient_evidence_line is None:
        raise FormalReadinessError(
            "formal readiness requires explicit finite-gradient/no-skipped-reg-head evidence line"
        )

    return {
        "train_log_valid": True,
        "finite_reg_loss_count": len(reg_losses),
        "last_finite_reg_loss": reg_losses[-1],
        "runtime_debug_rows": len(runtime_rows),
        "pretrain_load_marker_found": REQUIRED_PRETRAIN_PATH in text,
        "gradient_evidence_line_found": True,
    }


def validate_formal_readiness(config_path, pipeline_summary_path, geometry_summary_path, train_log):
    launch_gate = validate_launch_gate(config_path, pipeline_summary_path)
    geometry_summary = _load_json(geometry_summary_path)
    validate_linux_geometry_summary(geometry_summary)
    train_log_result = validate_formal_train_log(train_log)
    return {
        "validator": "bvr_twb_formal_readiness",
        "gate_pass": True,
        "route_label": ROUTE_LABEL,
        "formal_readiness_evidence_complete": True,
        "allowed_next_action": "FORMAL_REVIEW_ONLY_FULL_TRAIN_STILL_LOCKED",
        "full_train_unlocked": False,
        "formal_train_unlocked": False,
        "remote_sync_unlocked_by_formal_readiness": False,
        "sparse_compute_claim": False,
        "metric_claim": False,
        "runtime_or_flops_claim": False,
        "deploy_claim": False,
        "paper_claim": False,
        "adapter_bridge_mode": ADAPTER_FIXED_LENGTH_PADDED_BRIDGE,
        "pretrain_path": REQUIRED_PRETRAIN_PATH,
        "launch_gate_allowed_next_action": launch_gate["allowed_next_action"],
        **train_log_result,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate BVR-TWB formal-readiness evidence without unlocking train.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--pipeline-summary", required=True)
    parser.add_argument("--geometry-summary", required=True)
    parser.add_argument("--train-log", required=True)
    args = parser.parse_args()
    try:
        result = validate_formal_readiness(
            args.config,
            args.pipeline_summary,
            args.geometry_summary,
            args.train_log,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "gate_pass": False,
                    "validator": "bvr_twb_formal_readiness",
                    "route_label": ROUTE_LABEL,
                    "full_train_unlocked": False,
                    "formal_train_unlocked": False,
                    "sparse_compute_claim": False,
                    "reason": str(exc),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        raise SystemExit(1)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
