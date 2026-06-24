#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH -J c3diag1
#SBATCH -o logs/%x-%j.out

set -eo pipefail

source /etc/profile >/dev/null 2>&1 || true
set -u
module load cuda/11.8
module load miniforge3/24.11

YUZIBO_ROOT="${YUZIBO_ROOT:-$HOME/run/yuzibo}"
source activate "$YUZIBO_ROOT/conda_envs/opentad"

REPO="${OPENTAD_C3PRO_BOUNDARY_STABILITY_ROOT:-$YUZIBO_ROOT/OpenTAD_C3PRO_BOUNDARY_STABILITY_20260624_513cb84}"
if [ ! -d "$REPO" ]; then
  echo "[c3-pro-boundary-gradient-diag] $(date -Is) ERROR missing repo: $REPO" >&2
  exit 2
fi
cd "$REPO"

CONFIG="configs/adatad/thumos/pc_ot_mras_prebackbone_c3_pro_boundary_gradient_diag_smoke_n16r4.py"
TRAIN_ID="${TRAIN_ID:-0}"
SEED="${SEED:-20260624}"
PRETRAINED_PATH_REVIEWED="pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"
PRETRAINED_PATH="${PRETRAINED_PATH:-$PRETRAINED_PATH_REVIEWED}"
SLURM_JOB_ID_FOR_PORT="${SLURM_JOB_ID:-0}"
MASTER_PORT="${MASTER_PORT:-$((41700 + (SLURM_JOB_ID_FOR_PORT % 20000)))}"
RUN_TAG="${RUN_TAG:-c3pro_boundary_gradient_diag_$(git rev-parse --short HEAD 2>/dev/null || echo nogit)_smoke1_${SLURM_JOB_ID:-nojob}_$(date '+%Y%m%d_%H%M%S')}"

case "$RUN_TAG" in
  *..*|*/*|*\\*|*[[:space:]]*|*[\;\&\|\>\<\`\"\'$]*)
    echo "[c3-pro-boundary-gradient-diag] $(date -Is) ERROR RUN_TAG must be path-safe; got $RUN_TAG" >&2
    exit 2
    ;;
esac

LOG_DIR="$REPO/logs"
RUN_ROOT="$LOG_DIR/$RUN_TAG"
RUN_LOG="$RUN_ROOT/${RUN_TAG}.log"
TRAIN_STDOUT="$RUN_ROOT/${RUN_TAG}.train.stdout.log"
SHA_MANIFEST="$RUN_ROOT/active_sha256_manifest.txt"
RESOLVED_CONFIG_DUMP="$RUN_ROOT/resolved_config.py"
SUMMARY_JSON="$RUN_ROOT/c3_pro_boundary_gradient_diag_summary.json"
WORK_DIR_BASE="$RUN_ROOT/train_workdir"
WORK_DIR="$WORK_DIR_BASE/gpu1_id${TRAIN_ID}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export YUZIBO_ROOT
export OPENTAD_C3PRO_BOUNDARY_STABILITY_ROOT="$REPO"
export THUMOS14_ANNOTATION_PATH="${THUMOS14_ANNOTATION_PATH:-$YUZIBO_ROOT/thumos14/annotations/thumos_14_anno.json}"
export THUMOS14_CLASS_MAP="${THUMOS14_CLASS_MAP:-$YUZIBO_ROOT/thumos14/annotations/category_idx.txt}"
export THUMOS14_TRAIN_DATA_PATH="${THUMOS14_TRAIN_DATA_PATH:-$YUZIBO_ROOT/thumos14/train}"
export THUMOS14_TEST_DATA_PATH="${THUMOS14_TEST_DATA_PATH:-$YUZIBO_ROOT/thumos14/test}"
export PC_OT_MRAS_PREBACKBONE_C3_PRETRAINED_PATH="$PRETRAINED_PATH"

mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_LOG") 2>&1

log() {
  echo "[c3-pro-boundary-gradient-diag] $(date -Is) $*"
}

fail() {
  log "ERROR $*"
  exit 2
}

write_summary() {
  local status="$1"
  local decision="$2"
  local reason="$3"
  python - "$SUMMARY_JSON" "$status" "$decision" "$reason" "$RUN_ROOT" "$CONFIG" "$WORK_DIR" "$TRAIN_STDOUT" <<'PY'
import json
import os
import socket
import sys
from pathlib import Path

summary_path, status, decision, reason, run_root, config, work_dir, train_stdout = sys.argv[1:9]
payload = {
    "status": status,
    "decision": decision,
    "reason": reason,
    "stage": "C3_PRO_BOUNDARY_GRADIENT_DIAG_SMOKE",
    "host": socket.gethostname(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "run_root": run_root,
    "config": config,
    "work_dir": work_dir,
    "train_stdout": train_stdout,
    "claim_boundaries": {
        "diagnostic_only": True,
        "max_train_iters": 1,
        "single_gpu": True,
        "fixed_budget": "384/768",
        "selection_surface": "pre_backbone_raw_frame",
        "selection_timing": "online_before_backbone",
        "detector_head": "original_ActionFormerHead",
        "no_tools_test_execution": True,
        "no_detector_map": True,
        "no_checkpoint": True,
        "no_long_training": True,
        "no_raw_prediction_cache": True,
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
    write_summary "FAIL" "C3_PRO_BOUNDARY_GRADIENT_DIAG_FAILED" "launcher exited before successful completion; inspect train stdout/slurm log"
  fi
}
trap on_exit EXIT

for numeric_name in TRAIN_ID SEED MASTER_PORT; do
  value="${!numeric_name}"
  case "$value" in ''|*[!0-9]*) fail "$numeric_name must be numeric; got $value" ;; esac
done
if [ -z "${SLURM_JOB_ID:-}" ] && [ "${ALLOW_LOGIN_NODE_DEBUG:-0}" != "1" ]; then
  fail "refusing direct non-Slurm execution unless ALLOW_LOGIN_NODE_DEBUG=1 for lightweight checks"
fi
if [ "$PRETRAINED_PATH" != "$PRETRAINED_PATH_REVIEWED" ]; then
  fail "PRETRAINED_PATH override is forbidden for this diagnostic"
fi
if [ -n "${LOAD_FROM_RAW_PREDICTIONS:-}" ] || [ -n "${RAW_PREDICTION_CACHE:-}" ] || [ -n "${PREDICTION_CACHE:-}" ]; then
  fail "raw prediction/cache environment shortcuts are forbidden"
fi

EXPECTED_GIT_BRANCH="${OPENTAD_C3PRO_BOUNDARY_EXPECTED_BRANCH:-codex/c3-pro-boundary-stability-20260624}"
GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)" || fail "repo must be a git checkout"
GIT_HEAD="$(git rev-parse HEAD 2>/dev/null)" || fail "repo HEAD unavailable"
if [ "$GIT_BRANCH" != "$EXPECTED_GIT_BRANCH" ]; then
  fail "expected branch $EXPECTED_GIT_BRANCH; got $GIT_BRANCH"
fi
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  fail "tracked repo files are modified; refusing execution"
fi

log "host=$(hostname) job=${SLURM_JOB_ID:-none}"
log "repo=$REPO"
log "git_branch=$GIT_BRANCH git_head=$GIT_HEAD"
log "config=$CONFIG"
log "run_root=$RUN_ROOT"
log "work_dir=$WORK_DIR"
log "scope=diagnostic_only max_train_iters=1 no_validation no_checkpoint no_claims"
log "python=$(python -c 'import sys; print(sys.executable)')"

test -f "$CONFIG" || fail "missing config: $CONFIG"
test -f "$PRETRAINED_PATH" || fail "missing pretrained file: $PRETRAINED_PATH"
test -f "$THUMOS14_ANNOTATION_PATH" || fail "missing annotation: $THUMOS14_ANNOTATION_PATH"
test -f "$THUMOS14_CLASS_MAP" || fail "missing class map: $THUMOS14_CLASS_MAP"
test -d "$THUMOS14_TRAIN_DATA_PATH" || fail "missing train data dir: $THUMOS14_TRAIN_DATA_PATH"
test -d "$THUMOS14_TEST_DATA_PATH" || fail "missing test data dir: $THUMOS14_TEST_DATA_PATH"

python - "$CONFIG" "$RESOLVED_CONFIG_DUMP" <<'PY'
import sys
from pathlib import Path

from mmengine.config import Config
from opentad.utils.training_guard import assert_detector_training_allowed

cfg = Config.fromfile(sys.argv[1])
Path(sys.argv[2]).write_text(cfg.pretty_text + "\n", encoding="utf-8")
gate = cfg.pc_ot_mras_prebackbone_e2e_acquisition_gate
if gate.smoke_only is not True:
    raise SystemExit("diagnostic config must set smoke_only=True")
if gate.allow_tools_train is not True or gate.allow_tools_test is not False:
    raise SystemExit("diagnostic config must allow only tools/train.py")
if gate.allow_detector_map is not False or gate.allow_long_training is not False:
    raise SystemExit("diagnostic config must forbid detector map and long training")
if int(gate.max_epochs) != 1 or int(gate.max_train_iters) != 1:
    raise SystemExit("diagnostic gate must bind one epoch and one train iteration")
if int(cfg.workflow.end_epoch) != 1 or int(cfg.workflow.max_train_iters) != 1:
    raise SystemExit("diagnostic workflow must bind one epoch and one train iteration")
if cfg.workflow.disable_checkpoint is not True:
    raise SystemExit("diagnostic workflow must disable checkpoint writes")
if int(cfg.workflow.val_eval_interval) > 0 or int(cfg.workflow.val_loss_interval) > 0:
    raise SystemExit("diagnostic workflow must forbid validation/eval")
if cfg.model.frame_selector.reader.type != "PCOTMRASBoundaryDifficultyTemporalFrameScout":
    raise SystemExit("diagnostic must keep C3-Pro boundary reader")
if int(cfg.model.frame_selector.target_len) != 384 or int(cfg.model.frame_selector.dense_window_size) != 768:
    raise SystemExit("diagnostic must keep fixed 384/768 contract")
if cfg.model.rpn_head.type != "ActionFormerHead":
    raise SystemExit("diagnostic must keep original ActionFormerHead")
assert_detector_training_allowed(cfg, entrypoint="tools/train.py")
try:
    assert_detector_training_allowed(cfg, entrypoint="tools/test.py")
except RuntimeError:
    pass
else:
    raise SystemExit("diagnostic config must reject tools/test.py")
print("c3_pro_boundary_gradient_diag_config_preflight=PASS")
PY

RESOLVED_CONFIG_SHA256="$(sha256sum "$RESOLVED_CONFIG_DUMP" | awk '{print $1}')"
log "resolved_config_sha256=$RESOLVED_CONFIG_SHA256"

{
  sha256sum "$CONFIG"
  printf '%s  resolved_config.py\n' "$RESOLVED_CONFIG_SHA256"
  sha256sum \
    scripts/run_pc_ot_mras_prebackbone_c3_pro_boundary_gradient_diag_smoke_n16r4.sbatch \
    tools/train.py \
    opentad/cores/train_engine.py \
    opentad/utils/training_guard.py \
    opentad/models/detectors/actionformer.py \
    opentad/models/dense_heads/actionformer_head.py \
    opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py
} > "$SHA_MANIFEST"
ACTIVE_MANIFEST_SHA256="$(sha256sum "$SHA_MANIFEST" | awk '{print $1}')"
log "active_sha256_manifest_sha256=$ACTIVE_MANIFEST_SHA256"

python -m py_compile \
  "$CONFIG" \
  tools/train.py \
  opentad/cores/train_engine.py \
  opentad/utils/training_guard.py \
  opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py

python -m pytest \
  tests/test_train_engine_max_train_iters.py \
  tests/test_pc_ot_mras_prebackbone_c3_pro_stability.py \
  tests/test_pc_ot_mras_prebackbone_c3_reader_variants.py \
  tests/test_pc_ot_mras_prebackbone_c3_pro_gradient_diag_smoke.py \
  -q

mkdir -p "$WORK_DIR_BASE"
log "launching one-step tools/train.py gradient diagnostic"
torchrun --nproc_per_node=1 --master_port="$MASTER_PORT" \
  tools/train.py "$CONFIG" --id "$TRAIN_ID" --seed "$SEED" \
  --cfg-options \
  work_dir="$WORK_DIR_BASE" \
  dataset.train.ann_file="$THUMOS14_ANNOTATION_PATH" \
  dataset.val.ann_file="$THUMOS14_ANNOTATION_PATH" \
  dataset.test.ann_file="$THUMOS14_ANNOTATION_PATH" \
  dataset.train.class_map="$THUMOS14_CLASS_MAP" \
  dataset.val.class_map="$THUMOS14_CLASS_MAP" \
  dataset.test.class_map="$THUMOS14_CLASS_MAP" \
  dataset.train.data_path="$THUMOS14_TRAIN_DATA_PATH" \
  dataset.val.data_path="$THUMOS14_TEST_DATA_PATH" \
  dataset.test.data_path="$THUMOS14_TEST_DATA_PATH" \
  evaluation.ground_truth_filename="$THUMOS14_ANNOTATION_PATH" \
  2>&1 | tee "$TRAIN_STDOUT"

if find "$WORK_DIR_BASE" -maxdepth 5 -type f \( -name 'epoch_*.pth' -o -name 'best.pth' -o -name 'last.pth' -o -name 'result_detection.json' \) | grep -q .; then
  fail "diagnostic produced forbidden checkpoint or metric artifact under $WORK_DIR_BASE"
fi
grep -q "max_train_iters=1 reached" "$TRAIN_STDOUT" || fail "train stdout does not show max_train_iters=1 early stop"
grep -q "Training Over" "$TRAIN_STDOUT" || fail "train stdout does not show clean Training Over"
if grep -Eq "Testing Starts|Average-mAP|mAP at tIoU|result_detection" "$TRAIN_STDOUT"; then
  fail "train stdout contains forbidden testing/mAP/result marker"
fi

write_summary "PASS" "C3_PRO_BOUNDARY_GRADIENT_DIAG_PASS_FIRST_STEP_FINITE" "one-step train-entry gradient diagnostic completed with finite gradients"
log "C3-Pro boundary gradient diagnostic complete; no detector mAP, runtime/FLOPs, deploy, metric, or paper claim is unlocked"
