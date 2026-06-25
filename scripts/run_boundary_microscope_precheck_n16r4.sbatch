#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -J bm_precheck
#SBATCH -o logs/%x-%j.out

set -eo pipefail

source /etc/profile >/dev/null 2>&1 || true
set -u
if command -v module >/dev/null 2>&1; then
  module load cuda/11.8 >/dev/null 2>&1 || true
  module load miniforge3/24.11 >/dev/null 2>&1 || true
fi

YUZIBO_ROOT="${YUZIBO_ROOT:-$HOME/run/yuzibo}"
if [ "${BOUNDARY_MICROSCOPE_SKIP_CONDA:-0}" != "1" ]; then
  source activate "$YUZIBO_ROOT/conda_envs/opentad"
fi

REPO="${OPENTAD_BOUNDARY_MICROSCOPE_ROOT:-$(pwd)}"
if [ ! -d "$REPO" ]; then
  echo "[boundary-microscope-precheck] $(date -Is) ERROR missing repo: $REPO" >&2
  exit 2
fi
cd "$REPO"

CONFIG_REVIEWED="configs/adatad/thumos/boundary_microscope_acquisition_full_train_candidate_n16r4.py"
CONFIG="${CONFIG:-$CONFIG_REVIEWED}"
PRECHECK_ONLY="${PRECHECK_ONLY:-1}"
ALLOW_BOUNDARY_MICROSCOPE_FULL_TRAIN="${ALLOW_BOUNDARY_MICROSCOPE_FULL_TRAIN:-0}"
BOUNDARY_MICROSCOPE_FULL_TRAIN_GATE_JSON="${BOUNDARY_MICROSCOPE_FULL_TRAIN_GATE_JSON:-}"
BOUNDARY_MICROSCOPE_FULL_TRAIN_GATE_SHA256="${BOUNDARY_MICROSCOPE_FULL_TRAIN_GATE_SHA256:-}"
EXPECTED_GIT_BRANCH="${EXPECTED_GIT_BRANCH:-codex/boundary-microscope-pro-fix-20260625}"
RUN_TAG="${RUN_TAG:-boundary_microscope_precheck_${SLURM_JOB_ID:-nojob}_$(date '+%Y%m%d_%H%M%S')}"

case "$RUN_TAG" in
  *..*|*/*|*\\*|*[[:space:]]*|*[\;\&\|\>\<\`\"\'$]*)
    echo "[boundary-microscope-precheck] $(date -Is) ERROR RUN_TAG must be path-safe; got $RUN_TAG" >&2
    exit 2
    ;;
esac

LOG_DIR="$REPO/logs"
RUN_ROOT="$LOG_DIR/$RUN_TAG"
RUN_LOG="$RUN_ROOT/${RUN_TAG}.log"
SUMMARY_JSON="$RUN_ROOT/boundary_microscope_precheck_summary.json"
VALIDATOR_JSON="$RUN_ROOT/config_gate_report.json"
RESOLVED_CONFIG_DUMP="$RUN_ROOT/resolved_config.py"
SHA_MANIFEST="$RUN_ROOT/active_sha256_manifest.txt"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_LOG") 2>&1

log() {
  echo "[boundary-microscope-precheck] $(date -Is) $*"
}

fail() {
  log "ERROR $*"
  exit 2
}

write_summary() {
  local status="$1"
  local decision="$2"
  local reason="$3"
  python - "$SUMMARY_JSON" "$status" "$decision" "$reason" "$RUN_ROOT" "$CONFIG" <<'PY'
import json
import os
import socket
import sys
from pathlib import Path

summary_path, status, decision, reason, run_root, config = sys.argv[1:7]
payload = {
    "status": status,
    "decision": decision,
    "reason": reason,
    "route": "boundary_microscope_acquisition",
    "route_label": "DIVERGENT_INNOVATION_BOUNDARY_MICROSCOPE_DO_NOT_MERGE_WITH_C3",
    "stage": "deployment_precheck_n16r4",
    "host": socket.gethostname(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "run_root": run_root,
    "config": config,
    "claim_boundaries": {
        "precheck_only": True,
        "remote_precheck_only": True,
        "no_detector_training": True,
        "no_detector_eval": True,
        "no_raw_prediction_cache": True,
        "no_metric_claim": True,
        "full_train_locked": True,
    },
}
Path(summary_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  log "summary_json=$SUMMARY_JSON"
}

on_exit() {
  local status="$?"
  if [ "$status" -ne 0 ] && [ ! -f "$SUMMARY_JSON" ]; then
    write_summary "FAIL" "BOUNDARY_MICROSCOPE_PRECHECK_FAILED" "launcher exited before successful precheck completion"
  fi
}
trap on_exit EXIT

case "$PRECHECK_ONLY" in
  0|1) ;;
  *) fail "PRECHECK_ONLY must be exactly 0 or 1; got $PRECHECK_ONLY" ;;
esac
case "$ALLOW_BOUNDARY_MICROSCOPE_FULL_TRAIN" in
  0|1) ;;
  *) fail "ALLOW_BOUNDARY_MICROSCOPE_FULL_TRAIN must be exactly 0 or 1; got $ALLOW_BOUNDARY_MICROSCOPE_FULL_TRAIN" ;;
esac
if [ "$CONFIG" != "$CONFIG_REVIEWED" ]; then
  fail "launcher is reviewed only for $CONFIG_REVIEWED; got $CONFIG"
fi
if [ -z "${SLURM_JOB_ID:-}" ] && [ "${ALLOW_LOGIN_NODE_DEBUG:-0}" != "1" ]; then
  fail "refusing direct non-Slurm execution unless ALLOW_LOGIN_NODE_DEBUG=1 for lightweight PRECHECK_ONLY"
fi
if [ -z "${SLURM_JOB_ID:-}" ] && [ "${ALLOW_LOGIN_NODE_DEBUG:-0}" = "1" ] && [ "$PRECHECK_ONLY" != "1" ]; then
  fail "login-node debug must keep PRECHECK_ONLY=1; refusing training outside Slurm"
fi
if [ "$PRECHECK_ONLY" = "1" ]; then
  case "$ALLOW_BOUNDARY_MICROSCOPE_FULL_TRAIN" in
    0) ;;
    *) fail "PRECHECK_ONLY=1 must not set ALLOW_BOUNDARY_MICROSCOPE_FULL_TRAIN=1" ;;
  esac
else
  case "$ALLOW_BOUNDARY_MICROSCOPE_FULL_TRAIN" in
    1) ;;
    *) fail "PRECHECK_ONLY=0 requires ALLOW_BOUNDARY_MICROSCOPE_FULL_TRAIN=1" ;;
  esac
  if [ -z "$BOUNDARY_MICROSCOPE_FULL_TRAIN_GATE_JSON" ] || [ -z "$BOUNDARY_MICROSCOPE_FULL_TRAIN_GATE_SHA256" ]; then
    fail "PRECHECK_ONLY=0 requires BOUNDARY_MICROSCOPE_FULL_TRAIN_GATE_JSON and BOUNDARY_MICROSCOPE_FULL_TRAIN_GATE_SHA256"
  fi
fi
if [ -n "${CHECKPOINT:-}" ] || [ -n "${LOAD_FROM:-}" ] || [ -n "${RESUME:-}" ]; then
  fail "external checkpoint/load/resume environment shortcuts are forbidden"
fi
if [ -n "${RAW_PREDICTION_CACHE:-}" ] || [ -n "${PREDICTION_CACHE:-}" ] || [ -n "${LOAD_FROM_RAW_PREDICTIONS:-}" ]; then
  fail "raw prediction/cache environment shortcuts are forbidden"
fi
if [ -n "${CFG_OPTIONS:-}" ] || [ -n "${EXTRA_CFG_OPTIONS:-}" ]; then
  fail "arbitrary cfg-options are forbidden for Boundary Microscope deployment precheck"
fi

GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)" || fail "repo must be a git checkout"
GIT_HEAD="$(git rev-parse HEAD 2>/dev/null)" || fail "repo HEAD unavailable"
if [ "$GIT_BRANCH" != "$EXPECTED_GIT_BRANCH" ]; then
  fail "expected branch $EXPECTED_GIT_BRANCH; got $GIT_BRANCH"
fi
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  fail "tracked repo files are modified; refusing deployment precheck"
fi

log "host=$(hostname) job=${SLURM_JOB_ID:-none}"
log "repo=$REPO"
log "git_branch=$GIT_BRANCH git_head=$GIT_HEAD"
log "config=$CONFIG"
log "run_root=$RUN_ROOT"
log "precheck_only=$PRECHECK_ONLY allow_boundary_microscope_full_train=$ALLOW_BOUNDARY_MICROSCOPE_FULL_TRAIN"
log "route_label=DIVERGENT_INNOVATION_BOUNDARY_MICROSCOPE_DO_NOT_MERGE_WITH_C3"

test -f "$CONFIG" || fail "missing reviewed config: $CONFIG"
test -f "tools/bata/validate_boundary_microscope_gate.py" || fail "missing Boundary Microscope validator"
test -f "opentad/models/selectors/boundary_microscope_acquisition_route.py" || fail "missing Boundary Microscope selector"

log "validating fail-closed config gate"
python tools/bata/validate_boundary_microscope_gate.py "$CONFIG" --json | tee "$VALIDATOR_JSON"

log "dumping resolved config"
python - "$CONFIG" "$RESOLVED_CONFIG_DUMP" <<'PY'
import sys
from pathlib import Path

from mmengine.config import Config


def require(condition, message):
    if not condition:
        raise SystemExit(str(message))


cfg = Config.fromfile(sys.argv[1])
Path(sys.argv[2]).write_text(cfg.pretty_text + "\n", encoding="utf-8")
gate = cfg.boundary_microscope_gate
require(cfg.route_label == "DIVERGENT_INNOVATION_BOUNDARY_MICROSCOPE_DO_NOT_MERGE_WITH_C3", cfg.route_label)
require(cfg.route_id == "boundary_microscope_acquisition", cfg.route_id)
require(cfg.model.frame_selector.type == "BoundaryMicroscopeAcquisitionRoute", cfg.model.frame_selector)
require(int(cfg.model.frame_selector.target_len) == 384, cfg.model.frame_selector)
require(int(cfg.model.frame_selector.dense_window_size) == 768, cfg.model.frame_selector)
require(gate.allowed_decision == "ALLOW_BOUNDARY_MICROSCOPE_FULL_TRAIN", gate)
require(gate.requires_gate_json is True, gate)
require(gate.allow_precheck_only is False, gate)
require(gate.allow_tools_train is True, gate)
require(tuple(gate.allowed_entrypoints) == ("tools/train.py",), gate)
requires_entrypoint_gate = bool(gate.entrypoint_gate_context.required)
require(requires_entrypoint_gate is True, gate)
require(gate.entrypoint_gate_context.gate_json_env == "OPENTAD_BOUNDARY_MICROSCOPE_ENTRYPOINT_GATE_JSON", gate)
require(gate.entrypoint_gate_context.gate_sha256_env == "OPENTAD_BOUNDARY_MICROSCOPE_ENTRYPOINT_GATE_SHA256", gate)
require(gate.entrypoint_gate_context.active_manifest_sha256_env == "OPENTAD_BOUNDARY_MICROSCOPE_ACTIVE_MANIFEST_SHA256", gate)
require(gate.entrypoint_gate_context.resolved_config_sha256_env == "OPENTAD_BOUNDARY_MICROSCOPE_RESOLVED_CONFIG_SHA256", gate)
require(tuple(gate.entrypoint_gate_context.allowed_decisions) == ("ALLOW_BOUNDARY_MICROSCOPE_FULL_TRAIN",), gate)
for key in (
    "allow_tools_test",
    "allow_remote_sync",
    "allow_slurm",
    "allow_gpu",
    "allow_full_train",
    "allow_raw_prediction",
    "metric_claim_allowed",
    "paper_claim_allowed",
):
    require(getattr(gate, key) is False, f"{key} must remain False")
require(cfg.inference.load_from_raw_predictions is False, cfg.inference)
require(cfg.inference.save_raw_prediction is False, cfg.inference)
print("boundary_microscope_resolved_config_preflight=PASS")
PY
RESOLVED_CONFIG_SHA256="$(sha256sum "$RESOLVED_CONFIG_DUMP" | awk '{print $1}')"
log "resolved_config_sha256=$RESOLVED_CONFIG_SHA256"

log "writing active SHA256 manifest"
sha256sum \
  "$CONFIG" \
  "$RESOLVED_CONFIG_DUMP" \
  scripts/run_boundary_microscope_precheck_n16r4.sbatch \
  tools/bata/validate_boundary_microscope_gate.py \
  configs/adatad/thumos/boundary_microscope_acquisition_local_precheck.py \
  opentad/models/selectors/boundary_microscope_acquisition_route.py \
  opentad/models/selectors/__init__.py \
  tests/test_boundary_microscope_config_gate.py \
  docs/en/boundary_microscope_acquisition_route_review_context_20260624.md \
  > "$SHA_MANIFEST"
ACTIVE_MANIFEST_SHA256="$(sha256sum "$SHA_MANIFEST" | awk '{print $1}')"
log "active_sha256_manifest_sha256=$ACTIVE_MANIFEST_SHA256"

log "static py_compile"
python -m py_compile \
  "$CONFIG" \
  configs/adatad/thumos/boundary_microscope_acquisition_local_precheck.py \
  tools/bata/validate_boundary_microscope_gate.py \
  opentad/models/selectors/boundary_microscope_acquisition_route.py \
  tests/test_boundary_microscope_config_gate.py

log "focused Boundary Microscope config gate pytest"
python -m pytest tests/test_boundary_microscope_config_gate.py -q -rs

if [ "$PRECHECK_ONLY" = "1" ]; then
  write_summary "PASS" "BOUNDARY_MICROSCOPE_PRECHECK_ONLY_PASS_NO_TRAIN" "PRECHECK_ONLY=1; launcher exited before tools/train.py, detector eval, data access, or metric claim"
  log "PRECHECK_ONLY=1; exiting before tools/train.py"
  exit 0
fi

test -f "$BOUNDARY_MICROSCOPE_FULL_TRAIN_GATE_JSON" || fail "missing Boundary Microscope full-train gate JSON: $BOUNDARY_MICROSCOPE_FULL_TRAIN_GATE_JSON"
ACTUAL_GATE_SHA256="$(sha256sum "$BOUNDARY_MICROSCOPE_FULL_TRAIN_GATE_JSON" | awk '{print $1}')"
if [ "$ACTUAL_GATE_SHA256" != "$BOUNDARY_MICROSCOPE_FULL_TRAIN_GATE_SHA256" ]; then
  fail "Boundary Microscope full-train gate sha256 mismatch: expected=$BOUNDARY_MICROSCOPE_FULL_TRAIN_GATE_SHA256 actual=$ACTUAL_GATE_SHA256"
fi
fail "full train remains locked in Boundary Microscope deployment-precheck launcher; a separate code-owner full-train gate is required before tools/train.py"
