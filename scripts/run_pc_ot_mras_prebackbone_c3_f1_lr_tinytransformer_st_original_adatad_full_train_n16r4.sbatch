#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --time=72:00:00
#SBATCH -J pcot_c3rsst
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
  echo "[pcot-c3-full] $(date -Is) ERROR missing clean PC-OT-MRAS repo: $REPO" >&2
  exit 2
fi
cd "$REPO"

CONFIG_REVIEWED="configs/adatad/thumos/pc_ot_mras_prebackbone_c3_f1_lr_tinytransformer_st_original_adatad_full_train_candidate_n16r4.py"
CONFIG="${CONFIG:-$CONFIG_REVIEWED}"
PRECHECK_ONLY="${PRECHECK_ONLY:-1}"
ALLOW_PC_OT_MRAS_PREBACKBONE_C3_FULL_TRAIN="${ALLOW_PC_OT_MRAS_PREBACKBONE_C3_FULL_TRAIN:-0}"
PREBACKBONE_C3_GATE_JSON="${PREBACKBONE_C3_GATE_JSON:-}"
PREBACKBONE_C3_GATE_SHA256="${PREBACKBONE_C3_GATE_SHA256:-}"
ALLOW_PREBACKBONE_C3_TOOLS_TEST="${ALLOW_PREBACKBONE_C3_TOOLS_TEST:-0}"
ALLOW_PREBACKBONE_C3_CHECKPOINT_LOAD="${ALLOW_PREBACKBONE_C3_CHECKPOINT_LOAD:-0}"
ALLOW_PREBACKBONE_C3_OFFLINE_LEDGER="${ALLOW_PREBACKBONE_C3_OFFLINE_LEDGER:-0}"
ALLOW_PREBACKBONE_C3_RAW_PREDICTION_CACHE="${ALLOW_PREBACKBONE_C3_RAW_PREDICTION_CACHE:-0}"
TRAIN_ID="${TRAIN_ID:-0}"
SEED="${SEED:-20260624}"
BUDGET="${BUDGET:-384}"
DENSE_WINDOW_SIZE="${DENSE_WINDOW_SIZE:-768}"
SELECTION_UNIT="${SELECTION_UNIT:-1}"
SCOUT_SPATIAL_SIZE="${SCOUT_SPATIAL_SIZE:-32}"
PRETRAINED_PATH_REVIEWED="pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"
PRETRAINED_PATH="${PRETRAINED_PATH:-$PRETRAINED_PATH_REVIEWED}"
SLURM_JOB_ID_FOR_PORT="${SLURM_JOB_ID:-0}"
MASTER_PORT="${MASTER_PORT:-$((39700 + (SLURM_JOB_ID_FOR_PORT % 20000)))}"
RUN_TAG="${RUN_TAG:-prebackbone_c3_full_train_${SLURM_JOB_ID:-nojob}_$(date '+%Y%m%d_%H%M%S')}"

case "$RUN_TAG" in
  *..*|*/*|*\\*|*[[:space:]]*|*[\;\&\|\>\<\`\"\'$]*)
    echo "[pcot-c3-full] $(date -Is) ERROR RUN_TAG must be path-safe; got $RUN_TAG" >&2
    exit 2
    ;;
esac

LOG_DIR="$REPO/logs"
RUN_ROOT="$LOG_DIR/$RUN_TAG"
RUN_LOG="$RUN_ROOT/${RUN_TAG}.log"
TRAIN_STDOUT="$RUN_ROOT/${RUN_TAG}.train.stdout.log"
SHA_MANIFEST="$RUN_ROOT/active_sha256_manifest.txt"
RESOLVED_CONFIG_DUMP="$RUN_ROOT/resolved_config.py"
SUMMARY_JSON="$RUN_ROOT/prebackbone_c3_full_train_summary.json"
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
  echo "[pcot-c3-full] $(date -Is) $*"
}

fail() {
  log "ERROR $*"
  exit 2
}

write_summary() {
  local status="$1"
  local decision="$2"
  local reason="$3"
  python - "$SUMMARY_JSON" "$status" "$decision" "$reason" "$RUN_ROOT" "$CONFIG" "$FINAL_WORK_DIR" <<'PY'
import json
import os
import socket
import sys
from pathlib import Path

summary_path, status, decision, reason, run_root, config, work_dir = sys.argv[1:8]
payload = {
    "status": status,
    "decision": decision,
    "reason": reason,
    "stage": "PC_OT_MRAS_PREBACKBONE_C3_RS_HYBRID_ST_FULL_TRAIN",
    "route": "pc_ot_mras_prebackbone_c3_rs_hybrid_st_original_adatad",
    "variant_id": "C3-RS-Hybrid-ST-OriginalAdaTAD",
    "host": socket.gethostname(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "run_root": run_root,
    "config": config,
    "work_dir": work_dir,
    "claim_boundaries": {
        "selection_surface": "pre_backbone_raw_frame",
        "selection_timing": "online_before_backbone",
        "selector_reader": "PCOTMRASRSeriesHybridFrameScout",
        "selector_gradient": "ST hard real frames + full_flat soft transport surrogate",
        "detector_head": "original_ActionFormerHead",
        "nan_fail_fast": True,
        "robust_aux_objective": "gt_duplicate_value_risk_uncertainty_redundancy_role",
        "st_surrogate_mode": "full_flat",
        "scout_pixel_normalize": True,
        "scout_pixel_clamp": 5.0,
        "p2": False,
        "offline_ledger": False,
        "teacher": False,
        "test_gt": False,
        "direct_tools_test": False,
        "raw_prediction_cache": False,
        "checkpoint_load_or_resume": False,
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
    write_summary "FAIL" "PREBACKBONE_C3_RS_HYBRID_ST_FULL_TRAIN_FAILED" "launcher exited before successful completion"
  fi
}
trap on_exit EXIT

case "$PRECHECK_ONLY" in 0|1) ;; *) fail "PRECHECK_ONLY must be 0 or 1; got $PRECHECK_ONLY" ;; esac
case "$ALLOW_PREBACKBONE_C3_TOOLS_TEST" in 0) ;; *) fail "direct tools/test.py is forbidden for this launcher" ;; esac
case "$ALLOW_PREBACKBONE_C3_CHECKPOINT_LOAD" in 0) ;; *) fail "checkpoint load/resume is forbidden for first launch" ;; esac
case "$ALLOW_PREBACKBONE_C3_OFFLINE_LEDGER" in 0) ;; *) fail "offline ledger is forbidden for prebackbone C3" ;; esac
case "$ALLOW_PREBACKBONE_C3_RAW_PREDICTION_CACHE" in 0) ;; *) fail "raw prediction/cache is forbidden for prebackbone C3" ;; esac

for numeric_name in TRAIN_ID SEED BUDGET DENSE_WINDOW_SIZE SELECTION_UNIT SCOUT_SPATIAL_SIZE MASTER_PORT; do
  value="${!numeric_name}"
  case "$value" in ''|*[!0-9]*) fail "$numeric_name must be numeric; got $value" ;; esac
done
if [ "$BUDGET" -ne 384 ]; then fail "reviewed budget is 384; got $BUDGET"; fi
if [ "$DENSE_WINDOW_SIZE" -ne 768 ]; then fail "reviewed dense window size is 768; got $DENSE_WINDOW_SIZE"; fi
if [ "$SELECTION_UNIT" -ne 1 ]; then fail "reviewed selection unit is 1; got $SELECTION_UNIT"; fi
if [ "$SCOUT_SPATIAL_SIZE" -ne 32 ]; then fail "reviewed scout spatial size is 32; got $SCOUT_SPATIAL_SIZE"; fi
if [ "$CONFIG" != "$CONFIG_REVIEWED" ]; then fail "reviewed only for $CONFIG_REVIEWED; got $CONFIG"; fi
if [ "$PRETRAINED_PATH" != "$PRETRAINED_PATH_REVIEWED" ]; then fail "PRETRAINED_PATH override is forbidden"; fi
if [ -z "${SLURM_JOB_ID:-}" ] && [ "${ALLOW_LOGIN_NODE_DEBUG:-0}" != "1" ]; then
  fail "refusing direct non-Slurm execution unless ALLOW_LOGIN_NODE_DEBUG=1 for lightweight PRECHECK_ONLY"
fi
if [ -z "${SLURM_JOB_ID:-}" ] && [ "${ALLOW_LOGIN_NODE_DEBUG:-0}" = "1" ] && [ "$PRECHECK_ONLY" != "1" ]; then
  fail "login-node debug must keep PRECHECK_ONLY=1; refusing training outside Slurm"
fi
if [ "$PRECHECK_ONLY" = "0" ]; then
  case "$ALLOW_PC_OT_MRAS_PREBACKBONE_C3_FULL_TRAIN" in
    1) ;;
    *) fail "PRECHECK_ONLY=0 requires ALLOW_PC_OT_MRAS_PREBACKBONE_C3_FULL_TRAIN=1 after Pro/subagent review gate" ;;
  esac
  if [ -z "$PREBACKBONE_C3_GATE_JSON" ] || [ -z "$PREBACKBONE_C3_GATE_SHA256" ]; then
    fail "PRECHECK_ONLY=0 requires PREBACKBONE_C3_GATE_JSON and PREBACKBONE_C3_GATE_SHA256"
  fi
else
  case "$ALLOW_PC_OT_MRAS_PREBACKBONE_C3_FULL_TRAIN" in 0) ;; *) fail "PRECHECK_ONLY=1 must not enable training" ;; esac
fi
if [ -n "${CHECKPOINT:-}" ] || [ -n "${LOAD_FROM:-}" ] || [ -n "${RESUME:-}" ]; then
  fail "external checkpoint/load/resume env shortcuts are forbidden"
fi
if [ -n "${PC_OT_MRAS_FRONTEND_TRAIN_LEDGER_PATH:-}" ] || [ -n "${PC_OT_MRAS_FRONTEND_VAL_LEDGER_PATH:-}" ] || [ -n "${PC_OT_MRAS_FRONTEND_TEST_LEDGER_PATH:-}" ]; then
  fail "offline frontend ledger envs are forbidden"
fi
if [ -n "${RAW_PREDICTION_CACHE:-}" ] || [ -n "${PREDICTION_CACHE:-}" ] || [ -n "${LOAD_FROM_RAW_PREDICTIONS:-}" ]; then
  fail "raw prediction/cache env shortcuts are forbidden"
fi

EXPECTED_GIT_BRANCH="${OPENTAD_PCOTMRAS_EXPECTED_BRANCH:-codex/prebackbone-pcotmras-scout-selector}"
GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)" || fail "repo must be a git checkout"
GIT_HEAD="$(git rev-parse HEAD 2>/dev/null)" || fail "repo HEAD unavailable"
if [ "$GIT_BRANCH" != "$EXPECTED_GIT_BRANCH" ]; then fail "expected branch $EXPECTED_GIT_BRANCH; got $GIT_BRANCH"; fi
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then fail "tracked clean repo files are modified; refusing execution"; fi

log "host=$(hostname) job=${SLURM_JOB_ID:-none}"
log "repo=$REPO"
log "git_branch=$GIT_BRANCH git_head=$GIT_HEAD"
log "config=$CONFIG"
log "run_root=$RUN_ROOT"
log "work_dir=$WORK_DIR"
log "seed=$SEED train_id=$TRAIN_ID master_port=$MASTER_PORT"
log "required_gate_decision=ALLOW_PC_OT_MRAS_PREBACKBONE_C3_RS_HYBRID_ST_FULL_TRAIN_FIXED50"
test -f "$CONFIG" || fail "missing reviewed config: $CONFIG"

log "dumping resolved config"
python - "$CONFIG" "$RESOLVED_CONFIG_DUMP" <<'PY'
import sys
from pathlib import Path

from mmengine.config import Config
from opentad.utils.training_guard import assert_detector_training_allowed


def require(condition, message):
    if not condition:
        raise SystemExit(str(message))


cfg = Config.fromfile(sys.argv[1])
Path(sys.argv[2]).write_text(cfg.pretty_text + "\n", encoding="utf-8")
gate = cfg.pc_ot_mras_prebackbone_e2e_acquisition_gate
require(gate.route == "pc_ot_mras_prebackbone_c3_rs_hybrid_st_original_adatad", gate)
require(gate.stage == "c3_rs_hybrid_st_original_adatad_full_train", gate)
require(gate.selection_surface == "pre_backbone_raw_frame", gate)
require(gate.selection_timing == "online_before_backbone", gate)
require(gate.allow_tools_train is True, gate)
require(gate.allow_tools_test is False, gate)
require(gate.allow_detector_map is False, gate)
require(gate.allow_long_training is True, gate)
require(cfg.experiment_scope.variant_id == "C3-RS-Hybrid-ST-OriginalAdaTAD", cfg.experiment_scope)
require(cfg.experiment_scope.route == "pc_ot_mras_prebackbone_c3_rs_hybrid_st_original_adatad", cfg.experiment_scope)
require(cfg.experiment_scope.stage == "c3_rs_hybrid_st_original_adatad_full_train", cfg.experiment_scope)
require(cfg.experiment_scope.selector_reader == "PCOTMRASRSeriesHybridFrameScout", cfg.experiment_scope)
require(cfg.experiment_scope.reader_family == "RSeriesHybrid", cfg.experiment_scope)
require(cfg.experiment_scope.selector_gradient == "st_hard_real_frames_with_full_flat_soft_transport_surrogate", cfg.experiment_scope)
require(cfg.experiment_scope.st_hard_real_frames is True, cfg.experiment_scope)
require(cfg.experiment_scope.st_surrogate_mode == "full_flat", cfg.experiment_scope)
require(cfg.experiment_scope.train_loop_finite_fail_fast is True, cfg.experiment_scope)
require(cfg.experiment_scope.nan_fail_fast is True, cfg.experiment_scope)
require(cfg.experiment_scope.robust_aux_objective == "gt_duplicate_value_risk_uncertainty_redundancy_role", cfg.experiment_scope)
require(cfg.experiment_scope.scout_pixel_normalize is True, cfg.experiment_scope)
require(float(cfg.experiment_scope.scout_pixel_clamp) == 5.0, cfg.experiment_scope)
require(cfg.experiment_scope.free_frame_level_selector is True, cfg.experiment_scope)
require(cfg.experiment_scope.protected_scaffold is False, cfg.experiment_scope)
require(cfg.experiment_scope.uses_p2 is False, cfg.experiment_scope)
require(cfg.experiment_scope.uses_offline_ledger is False, cfg.experiment_scope)
require(cfg.experiment_scope.uses_teacher is False, cfg.experiment_scope)
require(cfg.experiment_scope.uses_test_gt is False, cfg.experiment_scope)
require(cfg.experiment_scope.uses_raw_prediction_cache is False, cfg.experiment_scope)
require(cfg.model.frame_selector.type == "PCOTMRASPreBackboneFrameSelector", cfg.model.frame_selector)
require(int(cfg.model.frame_selector.target_len) == 384, cfg.model.frame_selector)
require(int(cfg.model.frame_selector.dense_window_size) == 768, cfg.model.frame_selector)
require(int(cfg.model.frame_selector.selection_unit) == 1, cfg.model.frame_selector)
require(int(cfg.model.frame_selector.descriptor_dim) == 3072, cfg.model.frame_selector)
require(int(cfg.model.frame_selector.protected_uniform_count) == 0, cfg.model.frame_selector)
require(int(cfg.model.frame_selector.coverage_guard_count) == 0, cfg.model.frame_selector)
require(cfg.model.frame_selector.scout_feature_source == "compressed_pixels", cfg.model.frame_selector)
require(int(cfg.model.frame_selector.scout_spatial_size) == 32, cfg.model.frame_selector)
require(int(cfg.model.frame_selector.transport_topk) == 1, cfg.model.frame_selector)
require(int(cfg.model.frame_selector.eval_transport_topk) == 1, cfg.model.frame_selector)
require(cfg.model.frame_selector.straight_through_downstream is True, cfg.model.frame_selector)
require(cfg.model.frame_selector.st_surrogate_mode == "full_flat", cfg.model.frame_selector)
require(cfg.model.frame_selector.scout_pixel_normalize is True, cfg.model.frame_selector)
require(float(cfg.model.frame_selector.scout_pixel_clamp) == 5.0, cfg.model.frame_selector)
require(float(cfg.model.frame_selector.aux_gt_acquisition_loss_weight) == 0.05, cfg.model.frame_selector)
require(float(cfg.model.frame_selector.aux_duplicate_cap_loss_weight) == 0.001, cfg.model.frame_selector)
require(float(cfg.model.frame_selector.aux_duplicate_column_cap) == 1.25, cfg.model.frame_selector)
require(float(cfg.model.frame_selector.aux_value_loss_weight) == 0.02, cfg.model.frame_selector)
require(float(cfg.model.frame_selector.aux_risk_loss_weight) == 0.02, cfg.model.frame_selector)
require(float(cfg.model.frame_selector.aux_uncertainty_loss_weight) == 0.01, cfg.model.frame_selector)
require(float(cfg.model.frame_selector.aux_redundancy_loss_weight) == 0.01, cfg.model.frame_selector)
require(float(cfg.model.frame_selector.aux_role_entropy_loss_weight) == 0.001, cfg.model.frame_selector)
require(float(cfg.model.frame_selector.reader_regularizer_loss_weight) == 0.01, cfg.model.frame_selector)
require(cfg.model.frame_selector.reader.type == "PCOTMRASRSeriesHybridFrameScout", cfg.model.frame_selector.reader)
require(int(cfg.model.frame_selector.reader.in_dim) == 3072, cfg.model.frame_selector.reader)
require(int(cfg.model.frame_selector.reader.num_slots) == 384, cfg.model.frame_selector.reader)
require(int(cfg.model.frame_selector.reader.temporal_layers) == 3, cfg.model.frame_selector.reader)
require(float(cfg.model.frame_selector.reader.dropout) == 0.05, cfg.model.frame_selector.reader)
require(cfg.model.frame_selector.reader.local_global_fusion == "rseries_temporal_geometry_slot_attention", cfg.model.frame_selector.reader)
require("pc_ot_mras_reader" not in cfg.model, cfg.model)
require(cfg.model.rpn_head.type == "ActionFormerHead", cfg.model.rpn_head)
require(cfg.model.neck.type != "PCOTMRASDetectorBridge", cfg.model.neck)
require(int(cfg.model.backbone.backbone.total_frames) == 384, cfg.model.backbone)
require(int(cfg.model.projection.max_seq_len) == 384, cfg.model.projection)
require(int(cfg.workflow.end_epoch) == 60, cfg.workflow)
require(int(cfg.workflow.checkpoint_interval) == 60, cfg.workflow)
require(int(cfg.workflow.val_start_epoch) == 40, cfg.workflow)
require(int(cfg.workflow.val_eval_interval) == 2, cfg.workflow)
require("window_size" not in cfg.dataset.train, cfg.dataset.train)
train_loadframes = next(
    (step for step in cfg.dataset.train.pipeline if step.get("type") == "LoadFrames"),
    None,
)
require(train_loadframes is not None, cfg.dataset.train.pipeline)
require(int(train_loadframes.trunc_len) == 768, train_loadframes)
for split in ("val", "test"):
    require(int(cfg.dataset[split].window_size) == 768, cfg.dataset[split])
for split in ("train", "val", "test"):
    text = repr(cfg.dataset[split].pipeline).lower()
    for forbidden in ("bata_value_transport_ledger_subsample", "hard_positions", "teacher", "oracle", "raw_prediction"):
        require(forbidden not in text, f"{split} pipeline contains forbidden token {forbidden}")
try:
    assert_detector_training_allowed(cfg, entrypoint="tools/test.py")
except RuntimeError:
    pass
else:
    raise SystemExit("C3 full train config must reject direct tools/test.py")
print("pc_ot_mras_prebackbone_c3_full_train_config_preflight=PASS")
PY
RESOLVED_CONFIG_SHA256="$(sha256sum "$RESOLVED_CONFIG_DUMP" | awk '{print $1}')"
log "resolved_config_sha256=$RESOLVED_CONFIG_SHA256"

log "writing active SHA256 manifest"
{
  sha256sum "$CONFIG"
  printf '%s  resolved_config.py\n' "$RESOLVED_CONFIG_SHA256"
  sha256sum \
    scripts/run_pc_ot_mras_prebackbone_c3_f1_lr_tinytransformer_st_original_adatad_full_train_n16r4.sbatch \
    tools/train.py \
    tools/bata/validate_pc_ot_mras_prebackbone_c3_full_train_gate.py \
    opentad/cores/train_engine.py \
    opentad/utils/training_guard.py \
    opentad/models/detectors/base.py \
    opentad/models/detectors/actionformer.py \
    opentad/models/dense_heads/actionformer_head.py \
    opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py \
    opentad/models/selectors/pc_ot_mras_reader.py \
    tests/test_pc_ot_mras_prebackbone_c3_full_train_execution.py \
    tests/test_pc_ot_mras_prebackbone_c3_frame_level.py \
    tests/test_pc_ot_mras_prebackbone_selector_behavior.py
} > "$SHA_MANIFEST"
ACTIVE_MANIFEST_SHA256="$(sha256sum "$SHA_MANIFEST" | awk '{print $1}')"
log "active_sha256_manifest_sha256=$ACTIVE_MANIFEST_SHA256"

log "static py_compile and focused tests"
python -m py_compile \
  "$CONFIG" \
  tools/bata/validate_pc_ot_mras_prebackbone_c3_full_train_gate.py \
  opentad/cores/train_engine.py \
  opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py \
  opentad/models/detectors/base.py \
  tests/test_pc_ot_mras_prebackbone_c3_full_train_execution.py \
  tests/test_pc_ot_mras_prebackbone_c3_reader_variants.py \
  tests/test_pc_ot_mras_prebackbone_nan_guards.py
python -m pytest \
  tests/test_pc_ot_mras_prebackbone_c3_full_train_execution.py \
  tests/test_pc_ot_mras_prebackbone_c3_reader_variants.py \
  tests/test_pc_ot_mras_prebackbone_nan_guards.py \
  tests/test_pc_ot_mras_prebackbone_c3_frame_level.py \
  tests/test_pc_ot_mras_prebackbone_selector_behavior.py \
  -q

if [ "$PRECHECK_ONLY" = "1" ]; then
  write_summary "PASS" "PREBACKBONE_C3_RS_HYBRID_ST_FULL_TRAIN_PRECHECK_ONLY_PASS_NO_TRAIN" "PRECHECK_ONLY=1; no tools/train.py execution"
  exit 0
fi

test -f "$PRETRAINED_PATH" || fail "missing pretrained file: $PRETRAINED_PATH"
test -f "$THUMOS14_ANNOTATION_PATH" || fail "missing annotation: $THUMOS14_ANNOTATION_PATH"
test -f "$THUMOS14_CLASS_MAP" || fail "missing class map: $THUMOS14_CLASS_MAP"
test -d "$THUMOS14_TRAIN_DATA_PATH" || fail "missing train data dir: $THUMOS14_TRAIN_DATA_PATH"
test -d "$THUMOS14_TEST_DATA_PATH" || fail "missing test data dir: $THUMOS14_TEST_DATA_PATH"
PRETRAINED_SHA256="$(sha256sum "$PRETRAINED_PATH" | awk '{print $1}')"
log "pretrained_sha256=$PRETRAINED_SHA256"

python tools/bata/validate_pc_ot_mras_prebackbone_c3_full_train_gate.py \
  --gate-json "$PREBACKBONE_C3_GATE_JSON" \
  --gate-sha256 "$PREBACKBONE_C3_GATE_SHA256" \
  --active-manifest-sha256 "$ACTIVE_MANIFEST_SHA256" \
  --resolved-config-sha256 "$RESOLVED_CONFIG_SHA256" \
  --pretrained-sha256 "$PRETRAINED_SHA256" \
  --budget "$BUDGET" \
  --dense-window-size "$DENSE_WINDOW_SIZE"

mkdir -p "$WORK_DIR"
export OPENTAD_PCOTMRAS_PREBACKBONE_C3_GATE_JSON="$PREBACKBONE_C3_GATE_JSON"
export OPENTAD_PCOTMRAS_PREBACKBONE_C3_GATE_SHA256="$PREBACKBONE_C3_GATE_SHA256"
export OPENTAD_PCOTMRAS_PREBACKBONE_C3_ACTIVE_MANIFEST_SHA256="$ACTIVE_MANIFEST_SHA256"
export OPENTAD_PCOTMRAS_PREBACKBONE_C3_RESOLVED_CONFIG_SHA256="$RESOLVED_CONFIG_SHA256"

log "launching C3 full train tools/train.py"
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
if grep -Eiq "load_from_raw_predictions|RAW_PREDICTION_CACHE|PREDICTION_CACHE" "$TRAIN_STDOUT"; then
  fail "train stdout contains raw prediction/cache marker"
fi

write_summary "PASS" "PREBACKBONE_C3_RS_HYBRID_ST_FULL_TRAIN_PASS_NO_DIRECT_TEST_NO_CLAIMS" "tools/train.py completed full 60-epoch schedule; use final scheduled validation as performance"
log "PREBACKBONE_C3_RS_HYBRID_ST_FULL_TRAIN_PASS_NO_DIRECT_TEST_NO_CLAIMS"
