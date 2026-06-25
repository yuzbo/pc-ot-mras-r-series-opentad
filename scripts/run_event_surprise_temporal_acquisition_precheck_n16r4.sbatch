#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH -J evsur_pre
#SBATCH -o logs/%x-%j.out

set -eo pipefail

source /etc/profile >/dev/null 2>&1 || true
set -u
module load cuda/11.8
module load miniforge3/24.11

YUZIBO_ROOT="${YUZIBO_ROOT:-$HOME/run/yuzibo}"
REPO="${EVENT_SURPRISE_REPO:-$YUZIBO_ROOT/OpenTAD_EventSurprise_PrecheckDeploy_20260624_126e04e}"
PRECHECK_ONLY="${PRECHECK_ONLY:-1}"
RUN_TAG="${RUN_TAG:-event_surprise_precheck_${SLURM_JOB_ID:-nojob}_$(date '+%Y%m%d_%H%M%S')}"

case "$RUN_TAG" in
  *..*|*/*|*\\*|*[[:space:]]*|*[\;\&\|\>\<\`\"\'$]*)
    echo "[event-surprise-precheck] $(date -Is) ERROR RUN_TAG must be a simple path-safe name; got $RUN_TAG" >&2
    exit 2
    ;;
esac

if [ ! -d "$REPO" ]; then
  echo "[event-surprise-precheck] $(date -Is) ERROR missing Event-Surprise repo: $REPO" >&2
  exit 2
fi

RUN_ROOT="$REPO/logs/$RUN_TAG"
RUN_LOG="$RUN_ROOT/${RUN_TAG}.log"
SUMMARY_JSON="$RUN_ROOT/event_surprise_precheck_summary.json"
PRECHECK_CONFIG="configs/adatad/thumos/event_surprise_temporal_acquisition_local_precheck.py"
FULL_CONFIG="configs/adatad/thumos/event_surprise_temporal_acquisition_full_train_candidate_n16r4.py"
VALIDATOR="tools/bata/validate_event_surprise_gate.py"
RESOLVED_PRECHECK_DUMP="$RUN_ROOT/resolved_event_surprise_precheck_config.py"
RESOLVED_FULL_DUMP="$RUN_ROOT/resolved_event_surprise_full_candidate_config.py"
SHA_MANIFEST="$RUN_ROOT/event_surprise_active_sha256_manifest.txt"

mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_LOG") 2>&1

log() {
  echo "[event-surprise-precheck] $(date -Is) $*"
}

fail() {
  log "ERROR $*"
  exit 2
}

write_summary() {
  local status="$1"
  local decision="$2"
  local reason="$3"
  python - "$SUMMARY_JSON" "$status" "$decision" "$reason" "$RUN_ROOT" "$PRECHECK_ONLY" <<'PY'
import json
import os
import socket
import sys
from pathlib import Path

summary_path, status, decision, reason, run_root, precheck_only = sys.argv[1:7]
payload = {
    "status": status,
    "decision": decision,
    "reason": reason,
    "stage": "EVENT_SURPRISE_DEPLOYMENT_PRECHECK",
    "route": "event_surprise_temporal_acquisition",
    "route_label": "DIVERGENT_INNOVATION_EVENT_SURPRISE_DO_NOT_MERGE_WITH_C3",
    "host": socket.gethostname(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "run_root": run_root,
    "precheck_only": precheck_only,
    "claim_boundaries": {
        "precheck_only": precheck_only == "1",
        "tools_train": False,
        "tools_test": False,
        "detector_map": False,
        "checkpoint_access": False,
        "raw_prediction_cache": False,
        "runtime_or_flops_claim_allowed": False,
        "deploy_claim_allowed": False,
        "paper_claim_allowed": False,
    },
}
Path(summary_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  log "summary_json=$SUMMARY_JSON"
}

on_exit() {
  local status="$?"
  if [ "$status" -ne 0 ] && [ ! -f "$SUMMARY_JSON" ]; then
    write_summary "FAIL" "EVENT_SURPRISE_PRECHECK_FAILED" "launcher exited before successful completion"
  fi
}
trap on_exit EXIT

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export YUZIBO_ROOT
export EVENT_SURPRISE_REPO="$REPO"

cd "$REPO"

if [ -z "${SLURM_JOB_ID:-}" ] && [ "${ALLOW_LOGIN_NODE_DEBUG:-0}" != "1" ]; then
  fail "refusing direct non-Slurm execution unless ALLOW_LOGIN_NODE_DEBUG=1 for lightweight PRECHECK_ONLY"
fi

case "$PRECHECK_ONLY" in
  1|true|TRUE|yes|YES) PRECHECK_ONLY=1 ;;
  0|false|FALSE|no|NO) PRECHECK_ONLY=0 ;;
  *) fail "PRECHECK_ONLY must be 1 for precheck or 0 only after explicit full-train gate" ;;
esac

case "$REPO" in
  "$YUZIBO_ROOT"/OpenTAD_EventSurprise_PrecheckDeploy_*|"$YUZIBO_ROOT"/OpenTAD_EventSurprise_*)
    ;;
  *)
    fail "refusing non-route-owned repo path: $REPO"
    ;;
esac

if [ -n "${CHECKPOINT:-}" ] || [ -n "${CHECKPOINT_SHA256:-}" ] || [ -n "${LOAD_FROM:-}" ] || [ -n "${RESUME:-}" ]; then
  fail "checkpoint/load/resume shortcuts are forbidden for Event-Surprise precheck"
fi
if [ -n "${CFG_OPTIONS:-}" ] || [ -n "${EXTRA_CFG_OPTIONS:-}" ]; then
  fail "arbitrary cfg-options are forbidden for Event-Surprise precheck"
fi
if [ -n "${LOAD_FROM_RAW_PREDICTIONS:-}" ] || [ -n "${RAW_PREDICTION_CACHE:-}" ] || [ -n "${PREDICTION_CACHE:-}" ]; then
  fail "raw prediction/cache environment shortcuts are forbidden for Event-Surprise precheck"
fi

EXPECTED_GIT_BRANCH="${EVENT_SURPRISE_EXPECTED_BRANCH:-codex/event-surprise-pro-fix-20260625}"
GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)" || fail "Event-Surprise repo must be a git checkout"
GIT_HEAD="$(git rev-parse HEAD 2>/dev/null)" || fail "Event-Surprise repo HEAD is unavailable"
if [ "$GIT_BRANCH" != "$EXPECTED_GIT_BRANCH" ]; then
  fail "expected branch $EXPECTED_GIT_BRANCH; got $GIT_BRANCH"
fi
if [ -n "${EVENT_SURPRISE_EXPECTED_HEAD:-}" ] && [ "$GIT_HEAD" != "$EVENT_SURPRISE_EXPECTED_HEAD" ]; then
  fail "expected HEAD $EVENT_SURPRISE_EXPECTED_HEAD; got $GIT_HEAD"
fi
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  fail "tracked Event-Surprise repo files are modified; refusing precheck"
fi

log "host=$(hostname) job=${SLURM_JOB_ID:-none}"
log "repo=$REPO"
log "git_branch=$GIT_BRANCH git_head=$GIT_HEAD"
log "precheck_only=$PRECHECK_ONLY"
log "run_root=$RUN_ROOT"
log "scope=Event-Surprise PRECHECK_ONLY no_train=true no_test=true no_map=true no_raw_predictions=true no_claims=true"

source activate "$YUZIBO_ROOT/conda_envs/opentad"
log "python=$(python -c 'import sys; print(sys.executable)')"

test -f "$PRECHECK_CONFIG" || fail "missing precheck config: $PRECHECK_CONFIG"
test -f "$FULL_CONFIG" || fail "missing full candidate config: $FULL_CONFIG"
test -f "$VALIDATOR" || fail "missing validator: $VALIDATOR"

if [ "$PRECHECK_ONLY" = "0" ]; then
  GATE_JSON="${EVENT_SURPRISE_FULL_TRAIN_GATE_JSON:-}"
  GATE_SHA256="${EVENT_SURPRISE_FULL_TRAIN_GATE_SHA256:-}"
  OVERRIDE="${EVENT_SURPRISE_COORDINATOR_OVERRIDE:-}"
  if [ -z "$GATE_JSON" ] || [ -z "$GATE_SHA256" ]; then
    fail "PRECHECK_ONLY=0 requires EVENT_SURPRISE_FULL_TRAIN_GATE_JSON and EVENT_SURPRISE_FULL_TRAIN_GATE_SHA256"
  fi
  if [ "$OVERRIDE" != "ALLOW_EVENT_SURPRISE_FULL_TRAIN_CANDIDATE" ]; then
    fail "PRECHECK_ONLY=0 requires EVENT_SURPRISE_COORDINATOR_OVERRIDE=ALLOW_EVENT_SURPRISE_FULL_TRAIN_CANDIDATE"
  fi
  test -f "$GATE_JSON" || fail "missing full-train gate JSON: $GATE_JSON"
  ACTUAL_GATE_SHA256="$(sha256sum "$GATE_JSON" | awk '{print $1}')"
  if [ "$ACTUAL_GATE_SHA256" != "$GATE_SHA256" ]; then
    fail "full-train gate JSON sha256 mismatch: expected $GATE_SHA256 got $ACTUAL_GATE_SHA256"
  fi
  python "$VALIDATOR" --config "$FULL_CONFIG" --action full-train --gate-json "$GATE_JSON"
  if [ "${EVENT_SURPRISE_ENABLE_TOOLS_TRAIN_AFTER_GATE:-0}" != "1" ]; then
    fail "full-train gate validated, but EVENT_SURPRISE_ENABLE_TOOLS_TRAIN_AFTER_GATE=1 is also required"
  fi
  write_summary "READY" "EVENT_SURPRISE_FULL_TRAIN_GATE_PASS_REQUIRES_SEPARATE_COORDINATOR_LAUNCH" "full-train gate validated; this owner did not start training"
  log "EVENT_SURPRISE_FULL_TRAIN_GATE_PASS_NO_TRAIN_STARTED_BY_PRECHECK_OWNER"
  exit 0
fi

log "python/torch/cuda probe"
python - <<'PY'
import json
import torch

payload = {
    "torch_version": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": bool(torch.cuda.is_available()),
    "device_count": int(torch.cuda.device_count()),
}
print(json.dumps(payload, sort_keys=True))
PY
nvidia-smi || fail "nvidia-smi failed"

log "event-surprise validator precheck action"
python "$VALIDATOR" --config "$PRECHECK_CONFIG" --action precheck-only

log "event-surprise precheck/full candidate config validation"
python "$VALIDATOR" --config "$PRECHECK_CONFIG"
python "$VALIDATOR" --config "$FULL_CONFIG"

log "asserting full-train remains fail-closed without external gate JSON"
if python "$VALIDATOR" --config "$FULL_CONFIG" --action full-train; then
  fail "full-train action unexpectedly passed without gate JSON"
else
  log "full_train_without_gate=EXPECTED_FAIL"
fi

log "dumping resolved configs"
python - "$PRECHECK_CONFIG" "$FULL_CONFIG" "$RESOLVED_PRECHECK_DUMP" "$RESOLVED_FULL_DUMP" <<'PY'
import sys
from pathlib import Path
from mmengine.config import Config

precheck, full, precheck_dump, full_dump = sys.argv[1:5]
Path(precheck_dump).write_text(Config.fromfile(precheck).pretty_text + "\n", encoding="utf-8")
Path(full_dump).write_text(Config.fromfile(full).pretty_text + "\n", encoding="utf-8")
print("resolved_event_surprise_configs=PASS")
PY

log "writing active SHA256 manifest=$SHA_MANIFEST"
sha256sum \
  "$PRECHECK_CONFIG" \
  "$FULL_CONFIG" \
  "$VALIDATOR" \
  scripts/run_event_surprise_temporal_acquisition_precheck_n16r4.sbatch \
  opentad/models/selectors/event_surprise_acquisition_route.py \
  opentad/models/selectors/__init__.py \
  tests/test_event_surprise_config_gate.py \
  tests/test_event_surprise_acquisition_route.py \
  "$RESOLVED_PRECHECK_DUMP" \
  "$RESOLVED_FULL_DUMP" \
  > "$SHA_MANIFEST"
ACTIVE_MANIFEST_SHA256="$(sha256sum "$SHA_MANIFEST" | awk '{print $1}')"
log "active_sha256_manifest_sha256=$ACTIVE_MANIFEST_SHA256"

log "static py_compile"
python -m py_compile \
  "$VALIDATOR" \
  "$PRECHECK_CONFIG" \
  "$FULL_CONFIG" \
  opentad/models/selectors/event_surprise_acquisition_route.py \
  tests/test_event_surprise_config_gate.py \
  tests/test_event_surprise_acquisition_route.py

log "focused Event-Surprise launcher/gate pytest"
unset EVENT_SURPRISE_ENTRYPOINT_GATE_JSON
unset EVENT_SURPRISE_ENTRYPOINT_GATE_SHA256
unset EVENT_SURPRISE_ACTIVE_MANIFEST_SHA256
unset EVENT_SURPRISE_RESOLVED_CONFIG_SHA256
unset EVENT_SURPRISE_RUN_TAG
python -m pytest \
  tests/test_event_surprise_config_gate.py \
  -q

write_summary "PASS" "EVENT_SURPRISE_PRECHECK_ONLY_PASS_NO_TRAIN_NO_DATA_NO_MAP" "Event-Surprise validator/import/pytest precheck completed without detector training or evaluation"
log "EVENT_SURPRISE_PRECHECK_ONLY_PASS_NO_TRAIN_NO_DATA_NO_MAP"
