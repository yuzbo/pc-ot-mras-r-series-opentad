#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --time=3-00:00:00
#SBATCH -J evsur_full
#SBATCH -o logs/%x-%j.out

set -eo pipefail

source /etc/profile >/dev/null 2>&1 || true
set -u
module load cuda/11.8
module load miniforge3/24.11

YUZIBO_ROOT="${YUZIBO_ROOT:-$HOME/run/yuzibo}"
REPO="${EVENT_SURPRISE_REPO:-$YUZIBO_ROOT/OpenTAD_EventSurprise_PrecheckDeploy_20260624_126e04e}"
PRECHECK_ONLY="${PRECHECK_ONLY:-1}"
RUN_TAG="${RUN_TAG:-event_surprise_full_train_candidate_${SLURM_JOB_ID:-nojob}_$(date '+%Y%m%d_%H%M%S')}"
TRAIN_ID="${TRAIN_ID:-0}"

case "$RUN_TAG" in
  *..*|*/*|*\\*|*[[:space:]]*|*[\;\&\|\>\<\`\"\'$]*)
    echo "[event-surprise-full] $(date -Is) ERROR RUN_TAG must be a simple path-safe name; got $RUN_TAG" >&2
    exit 2
    ;;
esac

if [ ! -d "$REPO" ]; then
  echo "[event-surprise-full] $(date -Is) ERROR missing Event-Surprise repo: $REPO" >&2
  exit 2
fi

RUN_ROOT="$REPO/logs/$RUN_TAG"
RUN_LOG="$RUN_ROOT/${RUN_TAG}.log"
SUMMARY_JSON="$RUN_ROOT/event_surprise_full_train_summary.json"
FULL_CONFIG="configs/adatad/thumos/event_surprise_temporal_acquisition_full_train_candidate_n16r4.py"
VALIDATOR="tools/bata/validate_event_surprise_gate.py"
FULL_LAUNCHER="scripts/run_event_surprise_temporal_acquisition_full_train_n16r4.sbatch"
PRECHECK_LAUNCHER="scripts/run_event_surprise_temporal_acquisition_precheck_n16r4.sbatch"
RESOLVED_CONFIG_DUMP="$RUN_ROOT/resolved_event_surprise_full_candidate_config.py"
SHA_MANIFEST="$RUN_ROOT/event_surprise_active_sha256_manifest.txt"
TRAIN_STDOUT="$RUN_ROOT/${RUN_TAG}.train.stdout.log"

mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_LOG") 2>&1

log() {
  echo "[event-surprise-full] $(date -Is) $*"
}

fail() {
  log "ERROR $*"
  exit 2
}

write_summary() {
  local status="$1"
  local decision="$2"
  local reason="$3"
  python - "$SUMMARY_JSON" "$status" "$decision" "$reason" "$RUN_ROOT" "$PRECHECK_ONLY" "$RUN_TAG" <<'PY'
import json
import os
import socket
import sys
from pathlib import Path

summary_path, status, decision, reason, run_root, precheck_only, run_tag = sys.argv[1:8]
payload = {
    "status": status,
    "decision": decision,
    "reason": reason,
    "stage": "EVENT_SURPRISE_FULL_TRAIN_GATE",
    "route": "event_surprise_temporal_acquisition",
    "route_label": "DIVERGENT_INNOVATION_EVENT_SURPRISE_DO_NOT_MERGE_WITH_C3",
    "host": socket.gethostname(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "run_root": run_root,
    "run_tag": run_tag,
    "precheck_only": precheck_only,
    "claim_boundaries": {
        "full_train_candidate": precheck_only == "0",
        "direct_eval_entrypoint": False,
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
    write_summary "FAIL" "EVENT_SURPRISE_FULL_TRAIN_GATE_FAILED" "launcher exited before successful completion"
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
  *) fail "PRECHECK_ONLY must be 1 for precheck or 0 for gated full train" ;;
esac

case "$REPO" in
  "$YUZIBO_ROOT"/OpenTAD_EventSurprise_PrecheckDeploy_*|"$YUZIBO_ROOT"/OpenTAD_EventSurprise_*)
    ;;
  *)
    fail "refusing non-route-owned repo path: $REPO"
    ;;
esac

if [ -n "${CHECKPOINT:-}" ] || [ -n "${CHECKPOINT_SHA256:-}" ] || [ -n "${LOAD_FROM:-}" ] || [ -n "${RESUME:-}" ]; then
  fail "checkpoint/load/resume shortcuts are forbidden for Event-Surprise full train"
fi
if [ -n "${CFG_OPTIONS:-}" ] || [ -n "${EXTRA_CFG_OPTIONS:-}" ]; then
  fail "arbitrary cfg-options are forbidden for Event-Surprise full train"
fi
if [ -n "${LOAD_FROM_RAW_PREDICTIONS:-}" ] || [ -n "${RAW_PREDICTION_CACHE:-}" ] || [ -n "${PREDICTION_CACHE:-}" ]; then
  fail "raw prediction/cache environment shortcuts are forbidden for Event-Surprise full train"
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
  fail "tracked Event-Surprise repo files are modified; refusing full train"
fi

log "host=$(hostname) job=${SLURM_JOB_ID:-none}"
log "repo=$REPO"
log "git_branch=$GIT_BRANCH git_head=$GIT_HEAD"
log "precheck_only=$PRECHECK_ONLY"
log "run_tag=$RUN_TAG"
log "run_root=$RUN_ROOT"
log "scope=Event-Surprise independent route no_c3_merge=true no_raw_predictions=true no_claims=true"

source activate "$YUZIBO_ROOT/conda_envs/opentad"
log "python=$(python -c 'import sys; print(sys.executable)')"

test -f "$FULL_CONFIG" || fail "missing full candidate config: $FULL_CONFIG"
test -f "$VALIDATOR" || fail "missing validator: $VALIDATOR"
test -f "$FULL_LAUNCHER" || fail "missing full train launcher: $FULL_LAUNCHER"
test -f "$PRECHECK_LAUNCHER" || fail "missing precheck launcher: $PRECHECK_LAUNCHER"

log "dumping resolved full candidate config"
python - "$FULL_CONFIG" "$RESOLVED_CONFIG_DUMP" <<'PY'
import sys
from pathlib import Path
from mmengine.config import Config

config_path, resolved_dump = sys.argv[1:3]
Path(resolved_dump).write_text(Config.fromfile(config_path).pretty_text + "\n", encoding="utf-8")
print("resolved_event_surprise_full_candidate_config=PASS")
PY
RESOLVED_CONFIG_SHA256="$(sha256sum "$RESOLVED_CONFIG_DUMP" | awk '{print $1}')"
log "resolved_config_sha256=$RESOLVED_CONFIG_SHA256"

log "writing active SHA256 manifest=$SHA_MANIFEST"
sha256sum \
  "$FULL_CONFIG" \
  "$VALIDATOR" \
  "$FULL_LAUNCHER" \
  "$PRECHECK_LAUNCHER" \
  opentad/models/selectors/event_surprise_acquisition_route.py \
  opentad/models/selectors/__init__.py \
  tests/test_event_surprise_config_gate.py \
  tests/test_event_surprise_acquisition_route.py \
  "$RESOLVED_CONFIG_DUMP" \
  > "$SHA_MANIFEST"
ACTIVE_MANIFEST_SHA256="$(sha256sum "$SHA_MANIFEST" | awk '{print $1}')"
log "active_sha256_manifest_sha256=$ACTIVE_MANIFEST_SHA256"

log "event-surprise gate-bound full config validation"
python "$VALIDATOR" --config "$FULL_CONFIG"

log "static py_compile"
python -m py_compile \
  "$VALIDATOR" \
  "$FULL_CONFIG" \
  opentad/models/selectors/event_surprise_acquisition_route.py \
  tests/test_event_surprise_config_gate.py \
  tests/test_event_surprise_acquisition_route.py

log "bash syntax check"
bash -n "$FULL_LAUNCHER"
bash -n "$PRECHECK_LAUNCHER"

if [ "$PRECHECK_ONLY" = "1" ]; then
  unset EVENT_SURPRISE_ENTRYPOINT_GATE_JSON
  unset EVENT_SURPRISE_ENTRYPOINT_GATE_SHA256
  unset EVENT_SURPRISE_ACTIVE_MANIFEST_SHA256
  unset EVENT_SURPRISE_RESOLVED_CONFIG_SHA256
  unset EVENT_SURPRISE_RUN_TAG
  log "asserting full-train remains fail-closed without external gate JSON"
  if python "$VALIDATOR" --config "$FULL_CONFIG" --action full-train; then
    fail "full-train action unexpectedly passed without gate JSON"
  else
    log "full_train_without_gate=EXPECTED_FAIL"
  fi
  write_summary "PASS" "EVENT_SURPRISE_FULL_TRAIN_PRECHECK_ONLY_PASS_NO_TRAIN" "full-train launcher precheck completed without detector training"
  log "EVENT_SURPRISE_FULL_TRAIN_PRECHECK_ONLY_PASS_NO_TRAIN"
  exit 0
fi

if [ "${ALLOW_EVENT_SURPRISE_FULL_TRAIN:-0}" != "1" ]; then
  fail "PRECHECK_ONLY=0 requires ALLOW_EVENT_SURPRISE_FULL_TRAIN=1"
fi
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

log "event-surprise full-train gate validation"
python "$VALIDATOR" \
  --config "$FULL_CONFIG" \
  --action full-train \
  --gate-json "$GATE_JSON" \
  --gate-sha256 "$GATE_SHA256" \
  --active-manifest-sha256 "$ACTIVE_MANIFEST_SHA256" \
  --resolved-config-sha256 "$RESOLVED_CONFIG_SHA256" \
  --run-tag "$RUN_TAG"

export EVENT_SURPRISE_ENTRYPOINT_GATE_JSON="$GATE_JSON"
export EVENT_SURPRISE_ENTRYPOINT_GATE_SHA256="$GATE_SHA256"
export EVENT_SURPRISE_ACTIVE_MANIFEST_SHA256="$ACTIVE_MANIFEST_SHA256"
export EVENT_SURPRISE_RESOLVED_CONFIG_SHA256="$RESOLVED_CONFIG_SHA256"
export EVENT_SURPRISE_RUN_TAG="$RUN_TAG"

write_summary "RUNNING" "EVENT_SURPRISE_FULL_TRAIN_CANDIDATE_STARTED" "full-train gate validated and training command is starting"
log "starting Event-Surprise full-train candidate"
python -m torch.distributed.run --standalone --nproc_per_node=1 tools/train.py "$FULL_CONFIG" --id "$TRAIN_ID" 2>&1 | tee "$TRAIN_STDOUT"
write_summary "PASS" "EVENT_SURPRISE_FULL_TRAIN_CANDIDATE_COMMAND_FINISHED" "training command exited successfully"
log "EVENT_SURPRISE_FULL_TRAIN_CANDIDATE_COMMAND_FINISHED"
