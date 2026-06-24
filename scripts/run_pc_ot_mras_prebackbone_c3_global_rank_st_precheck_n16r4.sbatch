#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH -J pcot_c3grst_pre
#SBATCH -o logs/%x-%j.out

set -eo pipefail

source /etc/profile >/dev/null 2>&1 || true
set -u
module load cuda/11.8
module load miniforge3/24.11

YUZIBO_ROOT="${YUZIBO_ROOT:-$HOME/run/yuzibo}"
source activate "$YUZIBO_ROOT/conda_envs/opentad"

ROUTE_FAMILY="C3_ORIGINAL_OPTIMIZATION_ROUTE"
ROUTE_ID="pc_ot_mras_prebackbone_c3_global_rank_st"
VARIANT_ID="C3-GlobalRankST-BoundaryDifficulty-OriginalAdaTAD"
CONFIG_REVIEWED="configs/adatad/thumos/pc_ot_mras_prebackbone_c3_global_rank_st_full_train_candidate_n16r4.py"
CONFIG="${CONFIG:-$CONFIG_REVIEWED}"
PRECHECK_ONLY="${PRECHECK_ONLY:-1}"
ALLOW_C3_GLOBAL_RANK_ST_TRAIN="${ALLOW_C3_GLOBAL_RANK_ST_TRAIN:-0}"
ALLOW_C3_GLOBAL_RANK_ST_GPU_DEBUG="${ALLOW_C3_GLOBAL_RANK_ST_GPU_DEBUG:-0}"
ALLOW_C3_GLOBAL_RANK_ST_RAW_PREDICTION_CACHE="${ALLOW_C3_GLOBAL_RANK_ST_RAW_PREDICTION_CACHE:-0}"
RUN_TAG="${RUN_TAG:-c3_global_rank_st_precheck_${SLURM_JOB_ID:-nojob}_$(date '+%Y%m%d_%H%M%S')}"

case "$RUN_TAG" in
  *..*|*/*|*\\*|*[[:space:]]*|*[\;\&\|\>\<\`\"\'$]*)
    echo "[c3-global-rank-st] $(date -Is) ERROR RUN_TAG must be path-safe; got $RUN_TAG" >&2
    exit 2
    ;;
esac

CLEAN_REPO_NAME="${OPENTAD_PCOTMRAS_CLEAN_REPO_NAME:-OpenTAD_C3GlobalRankST_Worktree_20260624}"
REPO="${OPENTAD_PCOTMRAS_CLEAN_ROOT:-$YUZIBO_ROOT/$CLEAN_REPO_NAME}"
if [ ! -d "$REPO" ]; then
  echo "[c3-global-rank-st] $(date -Is) ERROR missing repo: $REPO" >&2
  exit 2
fi
cd "$REPO"

LOG_DIR="$REPO/logs"
RUN_ROOT="$LOG_DIR/$RUN_TAG"
SUMMARY_JSON="$RUN_ROOT/c3_global_rank_st_precheck_summary.json"
mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_ROOT/${RUN_TAG}.log") 2>&1

log() {
  echo "[c3-global-rank-st] $(date -Is) $*"
}

fail() {
  log "ERROR $*"
  exit 2
}

write_summary() {
  local status="$1"
  local decision="$2"
  local reason="$3"
  python - "$SUMMARY_JSON" "$status" "$decision" "$reason" "$ROUTE_FAMILY" "$ROUTE_ID" "$VARIANT_ID" "$CONFIG" <<'PY'
import json
import os
import socket
import sys
from pathlib import Path

summary_path, status, decision, reason, route_family, route, variant, config = sys.argv[1:9]
payload = {
    "status": status,
    "decision": decision,
    "reason": reason,
    "route_family": route_family,
    "route": route,
    "variant_id": variant,
    "stage": "C3_GLOBAL_RANK_ST_PRECHECK_ONLY",
    "host": socket.gethostname(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "config": config,
    "claim_boundaries": {
        "hard_selection": "real dense frames from frame_selection_logits",
        "st_surrogate": "global soft rank plus top-k membership",
        "p2": False,
        "teacher": False,
        "test_gt": False,
        "raw_prediction_cache": False,
        "detector_head": "original_ActionFormerHead",
        "post_processing": "unchanged",
        "formal_training_allowed": False,
        "metric_claim_allowed": False,
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
    write_summary "FAIL" "C3_GLOBAL_RANK_ST_PRECHECK_FAILED" "launcher exited before successful completion"
  fi
}
trap on_exit EXIT

case "$PRECHECK_ONLY" in
  1) ;;
  0) fail "PRECHECK_ONLY=0 remains locked for C3_GLOBAL_RANK_ST; no formal training approval exists" ;;
  *) fail "PRECHECK_ONLY must be 1 for this fail-closed precheck launcher; got $PRECHECK_ONLY" ;;
esac
case "$ALLOW_C3_GLOBAL_RANK_ST_TRAIN" in 0) ;; *) fail "ALLOW_C3_GLOBAL_RANK_ST_TRAIN must remain 0" ;; esac
case "$ALLOW_C3_GLOBAL_RANK_ST_GPU_DEBUG" in 0|1) ;; *) fail "ALLOW_C3_GLOBAL_RANK_ST_GPU_DEBUG must be 0 or 1" ;; esac
case "$ALLOW_C3_GLOBAL_RANK_ST_RAW_PREDICTION_CACHE" in
  0) ;;
  *) fail "raw prediction/cache is forbidden for C3_GLOBAL_RANK_ST" ;;
esac

if [ "$CONFIG" != "$CONFIG_REVIEWED" ]; then
  fail "reviewed only for $CONFIG_REVIEWED; got $CONFIG"
fi
if [ -n "${RAW_PREDICTION_CACHE:-}" ] || [ -n "${PREDICTION_CACHE:-}" ] || [ -n "${LOAD_FROM_RAW_PREDICTIONS:-}" ]; then
  fail "raw prediction/cache is forbidden for C3_GLOBAL_RANK_ST"
fi
if [ -n "${P2_CHECKPOINT:-}" ] || [ -n "${TEACHER_CHECKPOINT:-}" ] || [ -n "${TEST_GT_PATH:-}" ]; then
  fail "P2/teacher/test-GT shortcuts are forbidden for C3_GLOBAL_RANK_ST"
fi

log "route_family=$ROUTE_FAMILY route=$ROUTE_ID variant=$VARIANT_ID"
log "config=$CONFIG"
log "precheck_only=$PRECHECK_ONLY"
test -f "$CONFIG" || fail "missing reviewed config: $CONFIG"

python - "$CONFIG" <<'PY'
import sys
from mmengine.config import Config

cfg = Config.fromfile(sys.argv[1])

def require(condition, message):
    if not condition:
        raise SystemExit(str(message))

scope = cfg.experiment_scope
gate = cfg.pc_ot_mras_prebackbone_e2e_acquisition_gate
selector = cfg.model.frame_selector
require(scope.route_family == "C3_ORIGINAL_OPTIMIZATION_ROUTE", scope)
require(scope.route == "pc_ot_mras_prebackbone_c3_global_rank_st", scope)
require(scope.selection_strategy == "frame_score_global_rank_st", scope)
require(scope.rank_transport_surrogate == "global_rank_topk", scope)
require(scope.uses_p2 is False, scope)
require(scope.uses_teacher is False, scope)
require(scope.uses_test_gt is False, scope)
require(scope.uses_raw_prediction_cache is False, scope)
require(gate.formal_train_candidate is False, gate)
require(gate.launch_gate_passed is False, gate)
require(gate.allow_tools_train is False, gate)
require(gate.allow_tools_test is False, gate)
require(gate.allow_detector_map is False, gate)
require(gate.allow_long_training is False, gate)
require(tuple(gate.allowed_entrypoints) == (), gate)
require(selector.type == "PCOTMRASPreBackboneFrameSelector", selector)
require(selector.selection_strategy == "frame_score_global_rank_st", selector)
require(selector.frame_score_st_surrogate == "global_rank_topk", selector)
require(int(selector.global_rank_st_topk) == 384, selector)
require(int(selector.target_len) == 384, selector)
require(int(selector.dense_window_size) == 768, selector)
require(int(selector.max_dense_gap) == 0, selector)
require(int(selector.max_gap_guard_count) == 0, selector)
require(selector.reader.type == "PCOTMRASBoundaryDifficultyTemporalFrameScout", selector.reader)
require(cfg.model.rpn_head.type == "ActionFormerHead", cfg.model.rpn_head)
require("PCOTMRASDetectorBridge" not in repr(cfg.model), cfg.model)
require(cfg.inference.load_from_raw_predictions is False, cfg.inference)
require(cfg.inference.save_raw_prediction is False, cfg.inference)
print("C3_GLOBAL_RANK_ST_CONFIG_PRECHECK_PASS")
PY

python -m py_compile \
  "$CONFIG" \
  opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py \
  tests/test_pc_ot_mras_prebackbone_c3_global_rank_st.py

python -m pytest tests/test_pc_ot_mras_prebackbone_c3_global_rank_st.py -q

write_summary "PASS" "C3_GLOBAL_RANK_ST_PRECHECK_ONLY_PASS_NO_TRAIN" "PRECHECK_ONLY=1; formal training and detector evaluation remain locked"
log "C3_GLOBAL_RANK_ST_PRECHECK_ONLY_PASS_NO_TRAIN"
