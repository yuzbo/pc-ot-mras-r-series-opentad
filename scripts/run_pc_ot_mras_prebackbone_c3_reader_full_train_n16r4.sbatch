#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --time=72:00:00
#SBATCH -J pcot_c3reader
#SBATCH -o logs/%x-%j.out

set -eo pipefail

source /etc/profile >/dev/null 2>&1 || true
set -u
module load cuda/11.8
module load miniforge3/24.11

YUZIBO_ROOT="${YUZIBO_ROOT:-$HOME/run/yuzibo}"
source activate "$YUZIBO_ROOT/conda_envs/opentad"

CLEAN_REPO_NAME="${OPENTAD_PCOTMRAS_CLEAN_REPO_NAME:-OpenTAD_PCOTMRAS_R16_R18_R20_Clean_20260619_1730}"
REPO="${OPENTAD_PCOTMRAS_CLEAN_ROOT:-$YUZIBO_ROOT/$CLEAN_REPO_NAME}"
if [ ! -d "$REPO" ]; then
  echo "[pcot-c3-reader-full] $(date -Is) ERROR missing clean repo: $REPO" >&2
  exit 2
fi
cd "$REPO"

READER_VARIANT="${READER_VARIANT:-cnn_lite}"
case "$READER_VARIANT" in
  cnn_lite)
    CONFIG="configs/adatad/thumos/pc_ot_mras_prebackbone_c3_cnn_lite_reader_full_train_candidate_n16r4.py"
    READER_TYPE="PCOTMRASCNNFrameScout"
    ;;
  motion_tcn)
    CONFIG="configs/adatad/thumos/pc_ot_mras_prebackbone_c3_motion_tcn_reader_full_train_candidate_n16r4.py"
    READER_TYPE="PCOTMRASMotionTCNFrameScout"
    ;;
  hybrid)
    CONFIG="configs/adatad/thumos/pc_ot_mras_prebackbone_c3_hybrid_reader_full_train_candidate_n16r4.py"
    READER_TYPE="PCOTMRASHybridFrameScout"
    ;;
  *)
    echo "[pcot-c3-reader-full] $(date -Is) ERROR unknown READER_VARIANT=$READER_VARIANT" >&2
    exit 2
    ;;
esac

PRECHECK_ONLY="${PRECHECK_ONLY:-0}"
ALLOW_PC_OT_MRAS_PREBACKBONE_C3_READER_FULL_TRAIN="${ALLOW_PC_OT_MRAS_PREBACKBONE_C3_READER_FULL_TRAIN:-1}"
ALLOW_PREBACKBONE_C3_READER_TOOLS_TEST="${ALLOW_PREBACKBONE_C3_READER_TOOLS_TEST:-0}"
ALLOW_PREBACKBONE_C3_READER_CHECKPOINT_LOAD="${ALLOW_PREBACKBONE_C3_READER_CHECKPOINT_LOAD:-0}"
ALLOW_PREBACKBONE_C3_READER_OFFLINE_LEDGER="${ALLOW_PREBACKBONE_C3_READER_OFFLINE_LEDGER:-0}"
ALLOW_PREBACKBONE_C3_READER_RAW_PREDICTION_CACHE="${ALLOW_PREBACKBONE_C3_READER_RAW_PREDICTION_CACHE:-0}"
TRAIN_ID="${TRAIN_ID:-0}"
SEED="${SEED:-20260624}"
BUDGET="${BUDGET:-384}"
DENSE_WINDOW_SIZE="${DENSE_WINDOW_SIZE:-768}"
SELECTION_UNIT="${SELECTION_UNIT:-1}"
SCOUT_SPATIAL_SIZE="${SCOUT_SPATIAL_SIZE:-32}"
PRETRAINED_PATH_REVIEWED="pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"
PRETRAINED_PATH="${PRETRAINED_PATH:-$PRETRAINED_PATH_REVIEWED}"
SLURM_JOB_ID_FOR_PORT="${SLURM_JOB_ID:-0}"
MASTER_PORT="${MASTER_PORT:-$((41300 + (SLURM_JOB_ID_FOR_PORT % 20000)))}"
RUN_TAG="${RUN_TAG:-prebackbone_c3_reader_${READER_VARIANT}_${SLURM_JOB_ID:-nojob}_$(date '+%Y%m%d_%H%M%S')}"

case "$RUN_TAG" in
  *..*|*/*|*\\*|*[[:space:]]*|*[\;\&\|\>\<\`\"\'$]*)
    echo "[pcot-c3-reader-full] $(date -Is) ERROR RUN_TAG must be path-safe; got $RUN_TAG" >&2
    exit 2
    ;;
esac

LOG_DIR="$REPO/logs"
RUN_ROOT="$LOG_DIR/$RUN_TAG"
RUN_LOG="$RUN_ROOT/${RUN_TAG}.log"
TRAIN_STDOUT="$RUN_ROOT/${RUN_TAG}.train.stdout.log"
SHA_MANIFEST="$RUN_ROOT/active_sha256_manifest.txt"
RESOLVED_CONFIG_DUMP="$RUN_ROOT/resolved_config.py"
SUMMARY_JSON="$RUN_ROOT/prebackbone_c3_reader_full_train_summary.json"
WORK_DIR="$RUN_ROOT/train_workdir"
FINAL_WORK_DIR="$WORK_DIR/gpu1_id${TRAIN_ID}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export YUZIBO_ROOT
export OPENTAD_PCOTMRAS_CLEAN_ROOT="$REPO"
export THUMOS14_ANNOTATION_PATH="${THUMOS14_ANNOTATION_PATH:-$YUZIBO_ROOT/thumos14/annotations/thumos_14_anno.json}"
export THUMOS14_CLASS_MAP="${THUMOS14_CLASS_MAP:-$YUZIBO_ROOT/thumos14/annotations/category_idx.txt}"
export THUMOS14_TRAIN_DATA_PATH="${THUMOS14_TRAIN_DATA_PATH:-$YUZIBO_ROOT/thumos14/train}"
export THUMOS14_TEST_DATA_PATH="${THUMOS14_TEST_DATA_PATH:-$YUZIBO_ROOT/thumos14/test}"
export PC_OT_MRAS_PREBACKBONE_C3_PRETRAINED_PATH="$PRETRAINED_PATH"

mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_LOG") 2>&1

log() {
  echo "[pcot-c3-reader-full] $(date -Is) $*"
}

fail() {
  log "ERROR $*"
  exit 2
}

write_summary() {
  local status="$1"
  local decision="$2"
  local reason="$3"
  python - "$SUMMARY_JSON" "$status" "$decision" "$reason" "$RUN_ROOT" "$CONFIG" "$FINAL_WORK_DIR" "$READER_VARIANT" <<'PY'
import json
import os
import socket
import sys
from pathlib import Path

summary_path, status, decision, reason, run_root, config, work_dir, reader_variant = sys.argv[1:9]
payload = {
    "status": status,
    "decision": decision,
    "reason": reason,
    "stage": "PC_OT_MRAS_PREBACKBONE_C3_READER_FULL_TRAIN",
    "reader_variant": reader_variant,
    "host": socket.gethostname(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "run_root": run_root,
    "config": config,
    "work_dir": work_dir,
    "claim_boundaries": {
        "selection_surface": "pre_backbone_raw_frame",
        "selection_timing": "online_before_backbone",
        "detector_head": "original_ActionFormerHead",
        "p2": False,
        "offline_ledger": False,
        "teacher": False,
        "test_gt": False,
        "direct_tools_test": False,
        "raw_prediction_cache": False,
        "metric_claim_allowed": False,
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
    write_summary "FAIL" "PREBACKBONE_C3_READER_FULL_TRAIN_FAILED" "launcher exited before successful completion"
  fi
}
trap on_exit EXIT

case "$PRECHECK_ONLY" in 0|1) ;; *) fail "PRECHECK_ONLY must be 0 or 1; got $PRECHECK_ONLY" ;; esac
case "$ALLOW_PREBACKBONE_C3_READER_TOOLS_TEST" in 0) ;; *) fail "direct tools/test.py is forbidden" ;; esac
case "$ALLOW_PREBACKBONE_C3_READER_CHECKPOINT_LOAD" in 0) ;; *) fail "checkpoint load/resume is forbidden for first launch" ;; esac
case "$ALLOW_PREBACKBONE_C3_READER_OFFLINE_LEDGER" in 0) ;; *) fail "offline ledger is forbidden" ;; esac
case "$ALLOW_PREBACKBONE_C3_READER_RAW_PREDICTION_CACHE" in 0) ;; *) fail "raw prediction/cache is forbidden" ;; esac

for numeric_name in TRAIN_ID SEED BUDGET DENSE_WINDOW_SIZE SELECTION_UNIT SCOUT_SPATIAL_SIZE MASTER_PORT; do
  value="${!numeric_name}"
  case "$value" in ''|*[!0-9]*) fail "$numeric_name must be numeric; got $value" ;; esac
done
if [ "$BUDGET" -ne 384 ]; then fail "reviewed budget is 384; got $BUDGET"; fi
if [ "$DENSE_WINDOW_SIZE" -ne 768 ]; then fail "reviewed dense window size is 768; got $DENSE_WINDOW_SIZE"; fi
if [ "$SELECTION_UNIT" -ne 1 ]; then fail "reviewed selection unit is 1; got $SELECTION_UNIT"; fi
if [ "$SCOUT_SPATIAL_SIZE" -ne 32 ]; then fail "reviewed scout spatial size is 32; got $SCOUT_SPATIAL_SIZE"; fi
if [ "$PRETRAINED_PATH" != "$PRETRAINED_PATH_REVIEWED" ]; then fail "PRETRAINED_PATH override is forbidden"; fi
if [ -z "${SLURM_JOB_ID:-}" ] && [ "${ALLOW_LOGIN_NODE_DEBUG:-0}" != "1" ]; then
  fail "refusing direct non-Slurm execution unless ALLOW_LOGIN_NODE_DEBUG=1 for lightweight PRECHECK_ONLY"
fi
if [ "$PRECHECK_ONLY" = "0" ] && [ "$ALLOW_PC_OT_MRAS_PREBACKBONE_C3_READER_FULL_TRAIN" != "1" ]; then
  fail "PRECHECK_ONLY=0 requires ALLOW_PC_OT_MRAS_PREBACKBONE_C3_READER_FULL_TRAIN=1"
fi

EXPECTED_GIT_BRANCH="${OPENTAD_PCOTMRAS_EXPECTED_BRANCH:-codex/prebackbone-pcotmras-scout-selector}"
GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)" || fail "repo must be a git checkout"
GIT_HEAD="$(git rev-parse HEAD 2>/dev/null)" || fail "repo HEAD unavailable"
if [ "$GIT_BRANCH" != "$EXPECTED_GIT_BRANCH" ]; then fail "expected branch $EXPECTED_GIT_BRANCH; got $GIT_BRANCH"; fi
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then fail "tracked clean repo files are modified; refusing execution"; fi

log "host=$(hostname) job=${SLURM_JOB_ID:-none}"
log "repo=$REPO"
log "git_branch=$GIT_BRANCH git_head=$GIT_HEAD"
log "reader_variant=$READER_VARIANT reader_type=$READER_TYPE"
log "config=$CONFIG"
log "run_root=$RUN_ROOT"
log "work_dir=$WORK_DIR"
test -f "$CONFIG" || fail "missing config: $CONFIG"

python - "$CONFIG" "$RESOLVED_CONFIG_DUMP" "$READER_TYPE" <<'PY'
import sys
from pathlib import Path

from mmengine.config import Config
from opentad.utils.training_guard import assert_detector_training_allowed

cfg = Config.fromfile(sys.argv[1])
Path(sys.argv[2]).write_text(cfg.pretty_text + "\n", encoding="utf-8")
reader_type = sys.argv[3]
gate = cfg.pc_ot_mras_prebackbone_e2e_acquisition_gate
if gate.formal_train_candidate is not True:
    raise SystemExit("reader full train config must be formal_train_candidate=True")
if gate.allow_tools_train is not True or gate.allow_tools_test is not False:
    raise SystemExit("reader full train config must allow train and forbid test")
if gate.allow_long_training is not True:
    raise SystemExit("reader full train config must allow long training")
if cfg.model.frame_selector.reader.type != reader_type:
    raise SystemExit(f"reader type mismatch: {cfg.model.frame_selector.reader.type} != {reader_type}")
if int(cfg.model.frame_selector.target_len) != 384:
    raise SystemExit("target_len must be 384")
if int(cfg.model.frame_selector.dense_window_size) != 768:
    raise SystemExit("dense_window_size must be 768")
if int(cfg.model.frame_selector.selection_unit) != 1:
    raise SystemExit("selection_unit must be 1")
if cfg.model.rpn_head.type != "ActionFormerHead":
    raise SystemExit("detector head must stay ActionFormerHead")
if int(cfg.workflow.end_epoch) != 60:
    raise SystemExit("end_epoch must be 60")
for split in ("train", "val", "test"):
    text = repr(cfg.dataset[split].pipeline).lower()
    for forbidden in ("bata_value_transport_ledger_subsample", "hard_positions", "teacher", "oracle", "raw_prediction"):
        if forbidden in text:
            raise SystemExit(f"{split} pipeline contains forbidden token {forbidden}")
assert_detector_training_allowed(cfg, entrypoint="tools/train.py")
try:
    assert_detector_training_allowed(cfg, entrypoint="tools/test.py")
except RuntimeError:
    pass
else:
    raise SystemExit("reader full train config must reject direct tools/test.py")
print("pc_ot_mras_prebackbone_c3_reader_full_train_config_preflight=PASS")
PY

RESOLVED_CONFIG_SHA256="$(sha256sum "$RESOLVED_CONFIG_DUMP" | awk '{print $1}')"
log "resolved_config_sha256=$RESOLVED_CONFIG_SHA256"

{
  sha256sum "$CONFIG"
  printf '%s  resolved_config.py\n' "$RESOLVED_CONFIG_SHA256"
  sha256sum \
    scripts/run_pc_ot_mras_prebackbone_c3_reader_full_train_n16r4.sbatch \
    tools/train.py \
    opentad/cores/train_engine.py \
    opentad/utils/training_guard.py \
    opentad/models/detectors/base.py \
    opentad/models/detectors/actionformer.py \
    opentad/models/dense_heads/actionformer_head.py \
    opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py
} > "$SHA_MANIFEST"
ACTIVE_MANIFEST_SHA256="$(sha256sum "$SHA_MANIFEST" | awk '{print $1}')"
log "active_sha256_manifest_sha256=$ACTIVE_MANIFEST_SHA256"

python -m py_compile "$CONFIG" opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py
python -m pytest tests/test_pc_ot_mras_prebackbone_c3_reader_variants.py -q

if [ "$PRECHECK_ONLY" = "1" ]; then
  write_summary "PASS" "PREBACKBONE_C3_READER_FULL_TRAIN_PRECHECK_ONLY_PASS_NO_TRAIN" "PRECHECK_ONLY=1; no tools/train.py execution"
  exit 0
fi

test -f "$PRETRAINED_PATH" || fail "missing pretrained file: $PRETRAINED_PATH"
test -f "$THUMOS14_ANNOTATION_PATH" || fail "missing annotation: $THUMOS14_ANNOTATION_PATH"
test -f "$THUMOS14_CLASS_MAP" || fail "missing class map: $THUMOS14_CLASS_MAP"
test -d "$THUMOS14_TRAIN_DATA_PATH" || fail "missing train data dir: $THUMOS14_TRAIN_DATA_PATH"
test -d "$THUMOS14_TEST_DATA_PATH" || fail "missing test data dir: $THUMOS14_TEST_DATA_PATH"

mkdir -p "$WORK_DIR"
log "launching reader full train tools/train.py"
torchrun --nproc_per_node=1 --master_port="$MASTER_PORT" \
  tools/train.py "$CONFIG" --id "$TRAIN_ID" --seed "$SEED" \
  --cfg-options \
  work_dir="$WORK_DIR" \
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

grep -q "Training Over" "$TRAIN_STDOUT" || fail "train stdout does not show clean Training Over"
if grep -Eiq "nan|\\binf\\b|load_from_raw_predictions|RAW_PREDICTION_CACHE|PREDICTION_CACHE" "$TRAIN_STDOUT"; then
  fail "train stdout contains NaN/Inf or raw prediction/cache marker"
fi

write_summary "PASS" "PREBACKBONE_C3_READER_FULL_TRAIN_PASS_NO_DIRECT_TEST_NO_CLAIMS" "tools/train.py completed full 60-epoch schedule"
log "PREBACKBONE_C3_READER_FULL_TRAIN_PASS_NO_DIRECT_TEST_NO_CLAIMS"
