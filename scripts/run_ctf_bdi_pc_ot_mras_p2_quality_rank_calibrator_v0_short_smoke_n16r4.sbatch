#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -J p2qr_smk
#SBATCH -o logs/%x-%j.out

set -eo pipefail

source /etc/profile >/dev/null 2>&1 || true
set -u

YUZIBO_ROOT="${YUZIBO_ROOT:-$HOME/run/yuzibo}"
P2QR_LAUNCHER_TEST_MODE="${P2QR_LAUNCHER_TEST_MODE:-0}"
if [ "$P2QR_LAUNCHER_TEST_MODE" != "1" ]; then
  module load cuda/11.8
  module load miniforge3/24.11
  source activate "$YUZIBO_ROOT/conda_envs/opentad"
fi

CLEAN_REPO_NAME="${OPENTAD_PCOTMRAS_CLEAN_REPO_NAME:-OpenTAD_PCOTMRAS_R16_R18_R20_Clean_20260619_1730}"
REPO="${OPENTAD_PCOTMRAS_CLEAN_ROOT:-$YUZIBO_ROOT/$CLEAN_REPO_NAME}"
case "$REPO" in
  *OpenTAD_BATA_Clean*)
    echo "[p2qr-short-smoke] $(date -Is) ERROR refusing dirty historical OpenTAD_BATA_Clean path: $REPO" >&2
    exit 2
    ;;
esac
if [ ! -d "$REPO" ]; then
  echo "[p2qr-short-smoke] $(date -Is) ERROR missing clean PC-OT-MRAS repo: $REPO" >&2
  exit 2
fi
cd "$REPO"

CONFIG_REVIEWED="configs/adatad/thumos/ctf_bdi_pc_ot_mras_p2_quality_rank_calibrator_v0_short_smoke.py"
CONFIG="${CONFIG:-$CONFIG_REVIEWED}"
PRECHECK_ONLY="${PRECHECK_ONLY:-1}"
ALLOW_P2QR_SHORT_SMOKE="${ALLOW_P2QR_SHORT_SMOKE:-0}"
ALLOW_P2QR_SLURM="${ALLOW_P2QR_SLURM:-0}"
ALLOW_P2QR_GPU="${ALLOW_P2QR_GPU:-0}"
P2QR_SHORT_SMOKE_ONLY="${P2QR_SHORT_SMOKE_ONLY:-1}"
ALLOW_P2QR_TOOLS_TEST="${ALLOW_P2QR_TOOLS_TEST:-0}"
ALLOW_P2QR_DETECTOR_MAP="${ALLOW_P2QR_DETECTOR_MAP:-0}"
ALLOW_P2QR_LONG_TRAINING="${ALLOW_P2QR_LONG_TRAINING:-0}"
P2QR_SHORT_SMOKE_GATE_JSON="${P2QR_SHORT_SMOKE_GATE_JSON:-}"
P2QR_SHORT_SMOKE_GATE_SHA256="${P2QR_SHORT_SMOKE_GATE_SHA256:-}"
TRAIN_ID="${TRAIN_ID:-0}"
RUN_TAG="${RUN_TAG:-p2qr_short_smoke_scaffold_${SLURM_JOB_ID:-nojob}_$(date '+%Y%m%d_%H%M%S')}"

case "$RUN_TAG" in
  *..*|*/*|*\\*|*[[:space:]]*|*[\;\&\|\>\<\`\"\'$]*)
    echo "[p2qr-short-smoke] $(date -Is) ERROR RUN_TAG must be path-safe; got $RUN_TAG" >&2
    exit 2
    ;;
esac

LOG_DIR="$REPO/logs"
RUN_ROOT="$LOG_DIR/$RUN_TAG"
RUN_LOG="$RUN_ROOT/${RUN_TAG}.log"
SHA_MANIFEST="$RUN_ROOT/active_sha256_manifest.txt"
RESOLVED_CONFIG_DUMP="$RUN_ROOT/resolved_config.py"
SUMMARY_JSON="$RUN_ROOT/p2qr_short_smoke_scaffold_summary.json"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export YUZIBO_ROOT
export OPENTAD_PCOTMRAS_CLEAN_ROOT="$REPO"

mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_LOG") 2>&1

log() {
  echo "[p2qr-short-smoke] $(date -Is) $*"
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
    "stage": "P2QR_SHORT_SMOKE_LAUNCH_SCAFFOLD_ONLY",
    "route": "CTF-BDI/PC-OT-MRAS",
    "host": socket.gethostname(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "run_root": run_root,
    "config": config,
    "claim_boundaries": {
        "launch_scaffold_only": True,
        "precheck_only_default": True,
        "max_train_iters_future_limit": 2,
        "slurm_gpu_requires_future_gate": True,
        "no_tools_train_execution_this_version": True,
        "no_tools_test_execution": True,
        "no_detector_map": True,
        "no_dataset_access": True,
        "no_checkpoint_access": True,
        "no_long_training": True,
        "no_runtime_flops_deploy_metric_paper_claims": True,
    },
    "claim_flags": {
        "tools_train_run": False,
        "tools_test_run": False,
        "detector_map_run": False,
        "metric_claim_allowed": False,
        "paper_claim_allowed": False,
        "runtime_or_flops_claim_allowed": False,
        "deploy_claim_allowed": False,
    },
}
Path(summary_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  log "summary_json=$SUMMARY_JSON"
}

on_exit() {
  local status="$?"
  if [ "$status" -ne 0 ] && [ ! -f "$SUMMARY_JSON" ]; then
    write_summary "FAIL" "P2QR_SHORT_SMOKE_SCAFFOLD_FAILED" "launcher scaffold exited before successful completion"
  fi
}
trap on_exit EXIT

case "$PRECHECK_ONLY" in
  0|1) ;;
  *) fail "PRECHECK_ONLY must be exactly 0 or 1; got $PRECHECK_ONLY" ;;
esac
case "$P2QR_SHORT_SMOKE_ONLY" in
  1) ;;
  *) fail "P2QR_SHORT_SMOKE_ONLY must remain 1" ;;
esac
case "$ALLOW_P2QR_TOOLS_TEST" in
  0) ;;
  *) fail "tools/test.py is not approved; ALLOW_P2QR_TOOLS_TEST must remain 0" ;;
esac
case "$ALLOW_P2QR_SLURM" in
  0|1) ;;
  *) fail "ALLOW_P2QR_SLURM must be exactly 0 or 1; got $ALLOW_P2QR_SLURM" ;;
esac
case "$ALLOW_P2QR_GPU" in
  0|1) ;;
  *) fail "ALLOW_P2QR_GPU must be exactly 0 or 1; got $ALLOW_P2QR_GPU" ;;
esac
case "$ALLOW_P2QR_DETECTOR_MAP" in
  0) ;;
  *) fail "detector mAP is not approved; ALLOW_P2QR_DETECTOR_MAP must remain 0" ;;
esac
case "$ALLOW_P2QR_LONG_TRAINING" in
  0) ;;
  *) fail "long training is not approved; ALLOW_P2QR_LONG_TRAINING must remain 0" ;;
esac
if [ "$CONFIG" != "$CONFIG_REVIEWED" ]; then
  fail "launcher scaffold is reviewed only for $CONFIG_REVIEWED; got $CONFIG"
fi
if [ -z "${SLURM_JOB_ID:-}" ] && [ "${ALLOW_LOGIN_NODE_DEBUG:-0}" != "1" ]; then
  fail "refusing direct non-Slurm execution unless ALLOW_LOGIN_NODE_DEBUG=1 for lightweight PRECHECK_ONLY"
fi
if [ -z "${SLURM_JOB_ID:-}" ] && [ "${ALLOW_LOGIN_NODE_DEBUG:-0}" = "1" ] && [ "$PRECHECK_ONLY" != "1" ]; then
  fail "login-node debug must keep PRECHECK_ONLY=1; refusing training outside Slurm"
fi
if [ "$PRECHECK_ONLY" = "1" ]; then
  case "$ALLOW_P2QR_SHORT_SMOKE" in
    0) ;;
    *) fail "PRECHECK_ONLY=1 must not set ALLOW_P2QR_SHORT_SMOKE=1" ;;
  esac
  case "$ALLOW_P2QR_SLURM" in
    0) ;;
    *) fail "PRECHECK_ONLY=1 must not set ALLOW_P2QR_SLURM=1" ;;
  esac
  case "$ALLOW_P2QR_GPU" in
    0) ;;
    *) fail "PRECHECK_ONLY=1 must not set ALLOW_P2QR_GPU=1" ;;
  esac
  if [ -n "${SLURM_JOB_ID:-}" ] && [ "${ALLOW_P2QR_PRECHECK_SLURM:-0}" != "1" ]; then
    fail "PRECHECK_ONLY scaffold is not approved for Slurm allocation; use local review only or a separate Pro/Gemini gate"
  fi
else
  case "$ALLOW_P2QR_SHORT_SMOKE" in
    1) ;;
    *) fail "PRECHECK_ONLY=0 requires ALLOW_P2QR_SHORT_SMOKE=1 after a separate Pro/Gemini execution gate" ;;
  esac
  if [ -z "${SLURM_JOB_ID:-}" ]; then
    fail "PRECHECK_ONLY=0 requires a Slurm allocation bound to a future execution gate"
  fi
  case "$ALLOW_P2QR_SLURM" in
    1) ;;
    *) fail "PRECHECK_ONLY=0 requires ALLOW_P2QR_SLURM=1 from a separate Pro/Gemini execution gate" ;;
  esac
  case "$ALLOW_P2QR_GPU" in
    1) ;;
    *) fail "PRECHECK_ONLY=0 requires ALLOW_P2QR_GPU=1 from a separate Pro/Gemini execution gate" ;;
  esac
  if [ -z "$P2QR_SHORT_SMOKE_GATE_JSON" ] || [ -z "$P2QR_SHORT_SMOKE_GATE_SHA256" ]; then
    fail "PRECHECK_ONLY=0 requires P2QR_SHORT_SMOKE_GATE_JSON and P2QR_SHORT_SMOKE_GATE_SHA256"
  fi
fi
if [ -n "${CHECKPOINT:-}" ] || [ -n "${CHECKPOINT_SHA256:-}" ] || [ -n "${LOAD_FROM:-}" ] || [ -n "${RESUME:-}" ]; then
  fail "checkpoint/load/resume shortcuts are forbidden for P2QR launch scaffold"
fi
if [ -n "${LOAD_FROM_RAW_PREDICTIONS:-}" ] || [ -n "${RAW_PREDICTION_CACHE:-}" ] || [ -n "${PREDICTION_CACHE:-}" ]; then
  fail "raw prediction/cache shortcuts are forbidden for P2QR launch scaffold"
fi
if [ -n "${CFG_OPTIONS:-}" ] || [ -n "${EXTRA_CFG_OPTIONS:-}" ]; then
  fail "arbitrary cfg-options are forbidden for P2QR launch scaffold"
fi

log "host=$(hostname) job=${SLURM_JOB_ID:-none}"
log "repo=$REPO"
log "config=$CONFIG"
log "run_root=$RUN_ROOT"
log "precheck_only=$PRECHECK_ONLY allow_p2qr_short_smoke=$ALLOW_P2QR_SHORT_SMOKE allow_p2qr_slurm=$ALLOW_P2QR_SLURM allow_p2qr_gpu=$ALLOW_P2QR_GPU"
log "scope=LOCAL_LAUNCH_SCAFFOLD_ONLY no_train=true no_test=true no_map=true no_data=true no_claims=true"
log "python=$(python -c 'import sys; print(sys.executable)')"
log "git_head=$(git rev-parse HEAD 2>/dev/null || echo unknown)"

test -f "$CONFIG" || fail "missing reviewed config: $CONFIG"

log "writing active SHA256 manifest=$SHA_MANIFEST"
sha256sum \
  "$CONFIG" \
  scripts/run_ctf_bdi_pc_ot_mras_p2_quality_rank_calibrator_v0_short_smoke_n16r4.sbatch \
  tools/train.py \
  tools/bata/validate_p2qr_short_smoke_gate.py \
  tests/test_train_engine_max_train_iters.py \
  opentad/cores/train_engine.py \
  opentad/utils/training_guard.py \
  opentad/models/dense_heads/native_irregular_area_head_p2.py \
  tests/test_pc_ot_mras_p2_quality_launch_scaffold.py \
  > "$SHA_MANIFEST"
ACTIVE_MANIFEST_SHA256="$(sha256sum "$SHA_MANIFEST" | awk '{print $1}')"
log "active_sha256_manifest_sha256=$ACTIVE_MANIFEST_SHA256"
export P2QR_ACTIVE_MANIFEST_SHA256="$ACTIVE_MANIFEST_SHA256"

log "dumping and auditing resolved config=$RESOLVED_CONFIG_DUMP"
python - "$CONFIG" "$RESOLVED_CONFIG_DUMP" <<'PY'
import sys
from pathlib import Path

from mmengine.config import Config


def require(condition, message):
    if not condition:
        raise SystemExit(str(message))


cfg = Config.fromfile(sys.argv[1])
Path(sys.argv[2]).write_text(cfg.pretty_text + "\n", encoding="utf-8")

require(cfg.p2_quality_rank_calibrator_v0_gate is None, "local P2QR gate must be cleared")
gate = cfg.p2_quality_rank_calibrator_v0_short_smoke_gate
require(gate.stage == "P2_NIIQ_QualityRank_Calibrator_v0_short_smoke_scaffold", gate)
require(gate.launch_scaffold_only is True, gate)
require(gate.smoke_only is True, gate)
require(gate.allow_detector_training is False, gate)
require(gate.launch_gate_passed is False, gate)
require(gate.allow_tools_train is False, gate)
require(gate.allow_tools_test is False, gate)
require(gate.allow_detector_map is False, gate)
require(gate.allow_slurm is False, gate)
require(gate.allow_gpu is False, gate)
require(gate.allow_long_training is False, gate)
require(int(gate.max_train_iters) == 2, gate)
require(int(cfg.workflow.max_train_iters) == 2, cfg.workflow)
require(int(cfg.workflow.end_epoch) == 1, cfg.workflow)
require(int(cfg.workflow.val_eval_interval) == -1, cfg.workflow)
require(int(cfg.workflow.val_loss_interval) == -1, cfg.workflow)
require(cfg.inference.load_from_raw_predictions is False, cfg.inference)
require(cfg.inference.save_raw_prediction is False, cfg.inference)
require(cfg.model.rpn_head.area_head.quality_calibration.enable is True, cfg.model.rpn_head.area_head)
print("p2qr_short_smoke_scaffold_config_preflight=PASS")
PY
RESOLVED_CONFIG_SHA256="$(sha256sum "$RESOLVED_CONFIG_DUMP" | awk '{print $1}')"
log "resolved_config_sha256=$RESOLVED_CONFIG_SHA256"
export P2QR_RESOLVED_CONFIG_SHA256="$RESOLVED_CONFIG_SHA256"

if [ "$PRECHECK_ONLY" = "0" ]; then
  test -f "$P2QR_SHORT_SMOKE_GATE_JSON" || fail "missing P2QR execution gate JSON: $P2QR_SHORT_SMOKE_GATE_JSON"
  ACTUAL_GATE_SHA256="$(sha256sum "$P2QR_SHORT_SMOKE_GATE_JSON" | awk '{print $1}')"
  if [ "$ACTUAL_GATE_SHA256" != "$P2QR_SHORT_SMOKE_GATE_SHA256" ]; then
    fail "P2QR execution gate sha256 mismatch: expected=$P2QR_SHORT_SMOKE_GATE_SHA256 actual=$ACTUAL_GATE_SHA256"
  fi
  python tools/bata/validate_p2qr_short_smoke_gate.py \
    --gate-json "$P2QR_SHORT_SMOKE_GATE_JSON" \
    --gate-sha256 "$P2QR_SHORT_SMOKE_GATE_SHA256" \
    --active-manifest-sha256 "$ACTIVE_MANIFEST_SHA256" \
    --resolved-config-sha256 "$RESOLVED_CONFIG_SHA256"
fi

log "static py_compile"
python -m py_compile \
  "$CONFIG" \
  tools/bata/validate_p2qr_short_smoke_gate.py \
  opentad/cores/train_engine.py \
  opentad/utils/training_guard.py \
  opentad/models/dense_heads/native_irregular_area_head_p2.py \
  tests/test_pc_ot_mras_p2_quality_launch_scaffold.py \
  tests/test_train_engine_max_train_iters.py

log "focused local scaffold pytest"
python -m pytest \
  tests/test_pc_ot_mras_p2_quality_launch_scaffold.py \
  tests/test_train_engine_max_train_iters.py \
  -q -rs

if [ "$PRECHECK_ONLY" = "1" ]; then
  write_summary "PASS" "P2QR_SHORT_SMOKE_LAUNCH_SCAFFOLD_PRECHECK_PASS_NO_TRAIN" "PRECHECK_ONLY=1; exiting before any tools/train.py execution"
  log "PRECHECK_ONLY=1; exiting before any P2QR tools/train execution"
  exit 0
fi

fail "P2QR short-smoke execution is still denied in this scaffold; build a reviewed execution launcher after Pro/Gemini approval"
