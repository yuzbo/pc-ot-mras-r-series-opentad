#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --time=72:00:00
#SBATCH -J c3phys_full
#SBATCH -o logs/%x-%j.out

set -eo pipefail

source /etc/profile >/dev/null 2>&1 || true
set -u
module load cuda/11.8
module load miniforge3/24.11

YUZIBO_ROOT="${YUZIBO_ROOT:-$HOME/run/yuzibo}"
source activate "$YUZIBO_ROOT/conda_envs/opentad"

ROUTE_FAMILY="C3_ORIGINAL_OPTIMIZATION_ROUTE"
ROUTE_ID="pc_ot_mras_prebackbone_c3_physical_grid_actionformer"
VARIANT_ID="C3-PhysicalGridActionFormer-PreBackbone-OriginalAdaTAD"
STAGE_ID="c3_physical_grid_actionformer_full_train_n16r4"
CONFIG_REVIEWED="configs/adatad/thumos/pc_ot_mras_prebackbone_c3_physical_grid_actionformer_full_train_n16r4.py"
CONFIG_REVIEWED_COARSE="configs/adatad/thumos/pc_ot_mras_coarse_actionness_uncertainty_c3_physical_grid_actionformer_n16r4.py"
CONFIG_REVIEWED_EXACT_UNIFORM="configs/adatad/thumos/pc_ot_mras_exact_uniform_c3_physical_grid_actionformer_n16r4.py"
CONFIG_REVIEWED_UNIFORM_BIAS="configs/adatad/thumos/pc_ot_mras_uniform_biased_coarse_actionness_c3_physical_grid_actionformer_n16r4.py"
CONFIG_REVIEWED_A="configs/adatad/thumos/pc_ot_mras_a_uniform_scaffold_small_actionness_strict_maxgap_c3_physical_grid_actionformer_n16r4.py"
CONFIG="${CONFIG:-$CONFIG_REVIEWED_A}"
PRECHECK_ONLY="${PRECHECK_ONLY:-1}"
ALLOW_C3_PHYSICAL_GRID_FULL_TRAIN="${ALLOW_C3_PHYSICAL_GRID_FULL_TRAIN:-0}"
ALLOW_C3_PHYSICAL_GRID_RAW_PREDICTION_CACHE="${ALLOW_C3_PHYSICAL_GRID_RAW_PREDICTION_CACHE:-0}"
ALLOW_C3_PHYSICAL_GRID_CHECKPOINT_LOAD="${ALLOW_C3_PHYSICAL_GRID_CHECKPOINT_LOAD:-0}"
ALLOW_C3_PHYSICAL_GRID_OFFLINE_LEDGER="${ALLOW_C3_PHYSICAL_GRID_OFFLINE_LEDGER:-0}"
TRAIN_ID="${TRAIN_ID:-0}"
SEED="${SEED:-20260625}"
SLURM_JOB_ID_FOR_PORT="${SLURM_JOB_ID:-0}"
MASTER_PORT="${MASTER_PORT:-$((43100 + (SLURM_JOB_ID_FOR_PORT % 20000)))}"
PRETRAINED_PATH_REVIEWED="pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"
PRETRAINED_PATH="${C3_PRETRAINED_PATH:-${PRETRAINED_PATH:-$PRETRAINED_PATH_REVIEWED}}"
RUN_TAG="${RUN_TAG:-c3_physical_grid_full_${SLURM_JOB_ID:-nojob}_$(date '+%Y%m%d_%H%M%S')}"

case "$RUN_TAG" in
  *..*|*/*|*\\*|*[[:space:]]*|*[\;\&\|\>\<\`\"\'$]*)
    echo "[c3-physical-grid-full] $(date -Is) ERROR RUN_TAG must be path-safe; got $RUN_TAG" >&2
    exit 2
    ;;
esac

CLEAN_REPO_NAME="${OPENTAD_C3_PHYSICAL_GRID_REPO_NAME:-OpenTAD_C3PhysicalGridHead_Yuzbo_Worktree_20260625}"
REPO="${OPENTAD_C3_PHYSICAL_GRID_ROOT:-$YUZIBO_ROOT/$CLEAN_REPO_NAME}"
if [ ! -d "$REPO" ]; then
  echo "[c3-physical-grid-full] $(date -Is) ERROR missing repo: $REPO" >&2
  exit 2
fi
cd "$REPO"

LOG_DIR="$REPO/logs"
RUN_ROOT="$LOG_DIR/$RUN_TAG"
RUN_LOG="$RUN_ROOT/${RUN_TAG}.log"
TRAIN_STDOUT="$RUN_ROOT/${RUN_TAG}.train.stdout.log"
RESOLVED_CONFIG_DUMP="$RUN_ROOT/resolved_config.py"
SUMMARY_JSON="$RUN_ROOT/c3_physical_grid_full_train_summary.json"
SHA_MANIFEST="$RUN_ROOT/active_sha256_manifest.txt"
GATE_JSON="$RUN_ROOT/c3_physical_grid_full_train_gate.json"
WORK_DIR="$RUN_ROOT/train_workdir"
FINAL_WORK_DIR="$WORK_DIR/gpu1_id${TRAIN_ID}"
POSTTRAIN_DIAGNOSTIC_GATE_REQUIRED="${POSTTRAIN_DIAGNOSTIC_GATE_REQUIRED:-1}"
C3_SELECTOR_METADATA_JSONL="$RUN_ROOT/selector_metadata.jsonl"
C3_SELECTOR_METADATA_MAX_ROWS="${C3_SELECTOR_METADATA_MAX_ROWS:-32768}"
C3_SELECTOR_SUMMARY_JSON="${C3_SELECTOR_SUMMARY_JSON:-$RUN_ROOT/c3_selector_posttrain_diagnostic_summary.json}"
C3_RESULT_DETECTION_GEOMETRY_DIR="${C3_RESULT_DETECTION_GEOMETRY_DIR:-$RUN_ROOT/c3_result_detection_geometry}"
C3_RESULT_DETECTION_PROPOSALS_JSONL="$C3_RESULT_DETECTION_GEOMETRY_DIR/converted_result_detection_proposals.jsonl"
C3_PROPOSAL_DIAGNOSTIC_DIR="${C3_PROPOSAL_DIAGNOSTIC_DIR:-$RUN_ROOT/c3_proposal_localization_diagnostic}"
C3_PROPOSAL_SUMMARY_JSON="${C3_PROPOSAL_SUMMARY_JSON:-$C3_PROPOSAL_DIAGNOSTIC_DIR/summary.json}"
C3_OVERLOAD_DIAGNOSTIC_DIR="${C3_OVERLOAD_DIAGNOSTIC_DIR:-$RUN_ROOT/c3_post_nms_overload_diagnostic}"
C3_OVERLOAD_SUMMARY_JSON="${C3_OVERLOAD_SUMMARY_JSON:-$C3_OVERLOAD_DIAGNOSTIC_DIR/summary.json}"
C3_DIAGNOSTIC_GATE_JSON="${C3_DIAGNOSTIC_GATE_JSON:-$RUN_ROOT/c3_posttrain_diagnostic_gate_summary.json}"
C3_DIAGNOSTIC_GATE_MIN_SELECTOR_SAMPLES="${C3_DIAGNOSTIC_GATE_MIN_SELECTOR_SAMPLES:-1}"
C3_RESULT_DETECTION_JSON="$FINAL_WORK_DIR/result_detection.json"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export YUZIBO_ROOT
export THUMOS14_ANNOTATION_PATH="${THUMOS14_ANNOTATION_PATH:-$YUZIBO_ROOT/thumos14/annotations/thumos_14_anno.json}"
export THUMOS14_CLASS_MAP="${THUMOS14_CLASS_MAP:-$YUZIBO_ROOT/thumos14/annotations/category_idx.txt}"
export THUMOS14_TRAIN_DATA_PATH="${THUMOS14_TRAIN_DATA_PATH:-$YUZIBO_ROOT/thumos14/train}"
export THUMOS14_TEST_DATA_PATH="${THUMOS14_TEST_DATA_PATH:-$YUZIBO_ROOT/thumos14/test}"
export PC_OT_MRAS_PREBACKBONE_SELECTOR_METADATA_JSONL="$C3_SELECTOR_METADATA_JSONL"
export PC_OT_MRAS_PREBACKBONE_SELECTOR_METADATA_MAX_ROWS="$C3_SELECTOR_METADATA_MAX_ROWS"

mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_LOG") 2>&1

log() {
  echo "[c3-physical-grid-full] $(date -Is) $*"
}

fail() {
  log "ERROR $*"
  exit 2
}

write_summary() {
  local status="$1"
  local decision="$2"
  local reason="$3"
  python - "$SUMMARY_JSON" "$status" "$decision" "$reason" "$RUN_ROOT" "$CONFIG" "$FINAL_WORK_DIR" "$ROUTE_ID" "$VARIANT_ID" "$STAGE_ID" <<'PY'
import json
import os
import socket
import sys
from pathlib import Path

summary_path, status, decision, reason, run_root, config, work_dir, route, variant, stage_id = sys.argv[1:11]
payload = {
    "status": status,
    "decision": decision,
    "reason": reason,
    "stage": stage_id,
    "route_family": "C3_ORIGINAL_OPTIMIZATION_ROUTE",
    "route": route,
    "variant_id": variant,
    "host": socket.gethostname(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "run_root": run_root,
    "config": config,
    "work_dir": work_dir,
    "claim_boundaries": {
        "selection_surface": "pre_backbone_raw_frame",
        "detector_head": "ActionFormerHead with physical_grid_actionformer",
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
    write_summary "FAIL" "C3_PHYSICAL_GRID_FULL_TRAIN_FAILED" "launcher exited before successful completion"
  fi
}
trap on_exit EXIT

case "$PRECHECK_ONLY" in 0|1) ;; *) fail "PRECHECK_ONLY must be 0 or 1; got $PRECHECK_ONLY" ;; esac
case "$ALLOW_C3_PHYSICAL_GRID_RAW_PREDICTION_CACHE" in 0) ;; *) fail "raw prediction/cache is forbidden" ;; esac
case "$ALLOW_C3_PHYSICAL_GRID_CHECKPOINT_LOAD" in 0) ;; *) fail "checkpoint load/resume is forbidden" ;; esac
case "$ALLOW_C3_PHYSICAL_GRID_OFFLINE_LEDGER" in 0) ;; *) fail "offline ledger is forbidden" ;; esac
if [ -n "${P2_CHECKPOINT:-}" ] || [ -n "${TEACHER_CHECKPOINT:-}" ] || [ -n "${TEST_GT_PATH:-}" ]; then
  fail "P2/teacher/test-GT shortcuts are forbidden"
fi
if [ -n "${RAW_PREDICTION_CACHE:-}" ] || [ -n "${PREDICTION_CACHE:-}" ] || [ -n "${LOAD_FROM_RAW_PREDICTIONS:-}" ]; then
  fail "raw prediction/cache is forbidden"
fi
if [ "$PRECHECK_ONLY" = "0" ] && [ "$ALLOW_C3_PHYSICAL_GRID_FULL_TRAIN" != "1" ]; then
  fail "PRECHECK_ONLY=0 requires ALLOW_C3_PHYSICAL_GRID_FULL_TRAIN=1"
fi
if [ -z "${SLURM_JOB_ID:-}" ]; then
  if [ "${ALLOW_LOGIN_NODE_DEBUG:-0}" = "1" ] && [ "$PRECHECK_ONLY" = "1" ]; then
    log "ALLOW_LOGIN_NODE_DEBUG=1 permits lightweight PRECHECK_ONLY=1"
  else
    fail "refusing non-Slurm execution; use PRECHECK_ONLY=1 with ALLOW_LOGIN_NODE_DEBUG=1 only for lightweight checks"
  fi
fi
if [ "$PRECHECK_ONLY" = "0" ] && [ -z "${SLURM_JOB_ID:-}" ]; then
  fail "full training requires a Slurm GPU allocation"
fi
case "$POSTTRAIN_DIAGNOSTIC_GATE_REQUIRED" in 0|1) ;; *) fail "POSTTRAIN_DIAGNOSTIC_GATE_REQUIRED must be 0 or 1";; esac
if [ "$PRECHECK_ONLY" = "0" ] && [ "$POSTTRAIN_DIAGNOSTIC_GATE_REQUIRED" != "1" ]; then
  fail "full training requires POSTTRAIN_DIAGNOSTIC_GATE_REQUIRED=1"
fi

for numeric_name in TRAIN_ID SEED MASTER_PORT; do
  value="${!numeric_name}"
  case "$value" in ''|*[!0-9]*) fail "$numeric_name must be numeric; got $value" ;; esac
done
if [ "$MASTER_PORT" -lt 1024 ] || [ "$MASTER_PORT" -gt 65535 ]; then
  fail "MASTER_PORT must be between 1024 and 65535; got $MASTER_PORT"
fi
case "$CONFIG" in
  "$CONFIG_REVIEWED"|"$CONFIG_REVIEWED_COARSE"|"$CONFIG_REVIEWED_EXACT_UNIFORM"|"$CONFIG_REVIEWED_UNIFORM_BIAS"|"$CONFIG_REVIEWED_A") ;;
  *) fail "reviewed only for $CONFIG_REVIEWED, $CONFIG_REVIEWED_COARSE, $CONFIG_REVIEWED_EXACT_UNIFORM, $CONFIG_REVIEWED_UNIFORM_BIAS, or $CONFIG_REVIEWED_A; got $CONFIG" ;;
esac

EXPECTED_GIT_BRANCH="${OPENTAD_C3_PHYSICAL_GRID_EXPECTED_BRANCH:-codex/c3-physical-grid-head-20260625}"
GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)" || fail "repo must be a git checkout"
GIT_HEAD="$(git rev-parse HEAD 2>/dev/null)" || fail "repo HEAD unavailable"
if [ "$GIT_BRANCH" != "$EXPECTED_GIT_BRANCH" ]; then fail "expected branch codex/c3-physical-grid-head-20260625; got $GIT_BRANCH"; fi
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then fail "tracked clean repo files are modified; refusing execution"; fi

log "route_family=$ROUTE_FAMILY route=$ROUTE_ID variant=$VARIANT_ID"
log "repo=$REPO"
log "git_branch=$GIT_BRANCH git_head=$GIT_HEAD"
log "config=$CONFIG"
log "run_root=$RUN_ROOT"
test -f "$CONFIG" || fail "missing reviewed config: $CONFIG"

PRETRAINED_RESOLVED_PATH="$(readlink -f "$PRETRAINED_PATH" 2>/dev/null || true)"
if [ -z "$PRETRAINED_RESOLVED_PATH" ]; then
  case "$PRETRAINED_PATH" in
    /*) PRETRAINED_RESOLVED_PATH="$PRETRAINED_PATH" ;;
    *) PRETRAINED_RESOLVED_PATH="$REPO/$PRETRAINED_PATH" ;;
  esac
fi
test -f "$PRETRAINED_RESOLVED_PATH" || fail "missing pretrained file: $PRETRAINED_RESOLVED_PATH (from PRETRAINED_PATH=$PRETRAINED_PATH)"
export PC_OT_MRAS_PREBACKBONE_C3_PRETRAINED_PATH="$PRETRAINED_RESOLVED_PATH"
log "resolved_pretrained_path=$PRETRAINED_RESOLVED_PATH"

read VARIANT_ID ROUTE_ID STAGE_ID <<EOF
$(python - "$CONFIG" <<'PY'
import sys
from mmengine.config import Config

cfg = Config.fromfile(sys.argv[1])
print(cfg.variant_id, cfg.route_id, cfg.stage_id)
PY
)
EOF
log "resolved_identity route=$ROUTE_ID variant=$VARIANT_ID stage=$STAGE_ID"

python - "$CONFIG" "$RESOLVED_CONFIG_DUMP" <<'PY'
import sys
from pathlib import Path
from mmengine.config import Config

cfg = Config.fromfile(sys.argv[1])
Path(sys.argv[2]).write_text(cfg.pretty_text + "\n", encoding="utf-8")
print("c3_physical_grid_resolved_config_dump=PASS")
PY
RESOLVED_CONFIG_SHA256="$(sha256sum "$RESOLVED_CONFIG_DUMP" | awk '{print $1}')"

{
  sha256sum "$CONFIG"
  printf '%s  resolved_config.py\n' "$RESOLVED_CONFIG_SHA256"
  sha256sum \
    scripts/run_pc_ot_mras_prebackbone_c3_physical_grid_actionformer_full_train_n16r4.sbatch \
    tools/bata/validate_pc_ot_mras_prebackbone_c3_physical_grid_full_train_gate.py \
    tools/train.py \
    opentad/cores/train_engine.py \
    opentad/utils/training_guard.py \
    opentad/models/detectors/actionformer.py \
    opentad/models/dense_heads/anchor_free_head.py \
    opentad/models/dense_heads/actionformer_head.py \
    opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py \
    tools/bata/analyze_pc_ot_mras_selector_posttrain_diagnostics.py \
    tools/bata/analyze_actionformer_result_detection_geometry.py \
    tools/bata/analyze_p2_proposal_localization.py \
    tools/bata/analyze_actionformer_post_nms_overload.py \
    tools/bata/validate_pc_ot_mras_c3_diagnostic_gate.py
} > "$SHA_MANIFEST"
ACTIVE_MANIFEST_SHA256="$(sha256sum "$SHA_MANIFEST" | awk '{print $1}')"

python tools/bata/validate_pc_ot_mras_prebackbone_c3_physical_grid_full_train_gate.py --config "$CONFIG"
python -m py_compile \
  "$CONFIG" \
  opentad/models/dense_heads/anchor_free_head.py \
  opentad/models/detectors/actionformer.py \
  opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py \
  tools/bata/validate_pc_ot_mras_prebackbone_c3_physical_grid_full_train_gate.py \
  tools/bata/analyze_pc_ot_mras_selector_posttrain_diagnostics.py \
  tools/bata/analyze_actionformer_result_detection_geometry.py \
  tools/bata/analyze_p2_proposal_localization.py \
  tools/bata/analyze_actionformer_post_nms_overload.py \
  tools/bata/validate_pc_ot_mras_c3_diagnostic_gate.py
python -m pytest -q \
  tests/test_c3_physical_grid_actionformer_candidate.py \
  tests/test_c3_physical_grid_round_trip.py \
  tests/test_c3_physical_grid_full_train_execution.py

if [ "$PRECHECK_ONLY" = "1" ]; then
  write_summary "PASS" "C3_PHYSICAL_GRID_FULL_TRAIN_PRECHECK_ONLY_PASS_NO_TRAIN" "PRECHECK_ONLY=1; no tools/train.py execution"
  exit 0
fi

test -f opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py || fail "missing pre-backbone selector runtime: opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py"
test -f tools/bata/analyze_pc_ot_mras_selector_posttrain_diagnostics.py || fail "missing selector diagnostic producer"
test -f tools/bata/validate_pc_ot_mras_c3_diagnostic_gate.py || fail "missing C3 posttrain diagnostic gate"
test -f "$THUMOS14_ANNOTATION_PATH" || fail "missing annotation: $THUMOS14_ANNOTATION_PATH"
test -f "$THUMOS14_CLASS_MAP" || fail "missing class map: $THUMOS14_CLASS_MAP"
test -d "$THUMOS14_TRAIN_DATA_PATH" || fail "missing train data dir: $THUMOS14_TRAIN_DATA_PATH"
test -d "$THUMOS14_TEST_DATA_PATH" || fail "missing test data dir: $THUMOS14_TEST_DATA_PATH"

PRETRAINED_SHA256="$(sha256sum "$PRETRAINED_RESOLVED_PATH" | awk '{print $1}')"
python - "$GATE_JSON" "$ACTIVE_MANIFEST_SHA256" "$RESOLVED_CONFIG_SHA256" "$PRETRAINED_SHA256" "$PRETRAINED_RESOLVED_PATH" "$ROUTE_ID" "$VARIANT_ID" "$STAGE_ID" <<'PY'
import json
import sys
from pathlib import Path

path, manifest, resolved, pretrained, pretrained_resolved, route, variant, stage = sys.argv[1:9]
payload = {
    "decision": "ALLOW_C3_PHYSICAL_GRID_FULL_TRAIN",
    "route": route,
    "variant_id": variant,
    "stage": stage,
    "execution_mode": "train",
    "selection_surface": "pre_backbone_raw_frame",
    "selection_timing": "online_before_backbone",
    "acquisition_unit": "frame",
    "budget": 384,
    "dense_window_size": 768,
    "allow_tools_train": True,
    "allow_tools_test": False,
    "direct_tools_test": False,
    "allow_slurm": True,
    "allow_gpu": True,
    "single_gpu": True,
    "allow_precheck_only": True,
    "allow_prebackbone_frame_selector": True,
    "allow_train_validation_map": True,
    "allow_long_training": True,
    "allow_pretrained_initialization": True,
    "allow_checkpoint_write": True,
    "allow_checkpoint_load": False,
    "allow_resume": False,
    "allow_offline_ledger": False,
    "allow_raw_prediction_cache": False,
    "allow_detector_training": True,
    "uses_p2": False,
    "uses_offline_ledger": False,
    "uses_teacher": False,
    "uses_test_gt": False,
    "uses_raw_prediction_cache": False,
    "checkpoint_load": False,
    "resume": False,
    "offline_ledger": False,
    "raw_prediction_cache": False,
    "load_from_raw_predictions": False,
    "save_raw_prediction": False,
    "metric_claim": False,
    "metric_claim_allowed": False,
    "paper_claim": False,
    "paper_claim_allowed": False,
    "runtime_flops_claim": False,
    "runtime_flops_claim_allowed": False,
    "deploy_claim": False,
    "deploy_claim_allowed": False,
    "active_sha256_manifest_sha256": manifest,
    "resolved_config_sha256": resolved,
    "pretrained_sha256": pretrained,
    "pretrained_resolved_path": pretrained_resolved,
}
Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
GATE_SHA256="$(sha256sum "$GATE_JSON" | awk '{print $1}')"
python tools/bata/validate_pc_ot_mras_prebackbone_c3_physical_grid_full_train_gate.py \
  --config "$CONFIG" \
  --gate-json "$GATE_JSON" \
  --gate-sha256 "$GATE_SHA256" \
  --active-manifest-sha256 "$ACTIVE_MANIFEST_SHA256" \
  --resolved-config-sha256 "$RESOLVED_CONFIG_SHA256" \
  --pretrained-sha256 "$PRETRAINED_SHA256"
export OPENTAD_C3_PHYSICAL_GRID_GATE_JSON="$GATE_JSON"
export OPENTAD_C3_PHYSICAL_GRID_GATE_SHA256="$GATE_SHA256"
export OPENTAD_C3_PHYSICAL_GRID_ACTIVE_MANIFEST_SHA256="$ACTIVE_MANIFEST_SHA256"
export OPENTAD_C3_PHYSICAL_GRID_RESOLVED_CONFIG_SHA256="$RESOLVED_CONFIG_SHA256"

mkdir -p "$WORK_DIR"
rm -f "$C3_SELECTOR_METADATA_JSONL"
log "launching C3 PhysicalGrid full train"
if command -v ss >/dev/null 2>&1; then
  if ss -ltn | awk '{print $4}' | grep -Eq "[:.]${MASTER_PORT}$"; then
    fail "MASTER_PORT=$MASTER_PORT is already in use"
  fi
fi
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
  model.backbone.custom.pretrain="$PRETRAINED_RESOLVED_PATH" \
  2>&1 | tee "$TRAIN_STDOUT"

grep -q "Training Over" "$TRAIN_STDOUT" || fail "train stdout does not show clean Training Over"
if grep -Eq "load_from_raw_predictions[\"']?[[:space:]]*[:=][[:space:]]*True|save_raw_prediction[\"']?[[:space:]]*[:=][[:space:]]*True|(^|[[:space:]])RAW_PREDICTION_CACHE[[:space:]]*=|(^|[[:space:]])PREDICTION_CACHE[[:space:]]*=" "$TRAIN_STDOUT"; then
  fail "train stdout contains raw prediction/cache marker"
fi

if [ "$POSTTRAIN_DIAGNOSTIC_GATE_REQUIRED" = "1" ]; then
  test -s "$C3_SELECTOR_METADATA_JSONL" || fail "missing current selector metadata dump: $C3_SELECTOR_METADATA_JSONL"
  test -f "$C3_RESULT_DETECTION_JSON" || fail "missing current result_detection json: $C3_RESULT_DETECTION_JSON"
  mkdir -p "$C3_RESULT_DETECTION_GEOMETRY_DIR" "$C3_PROPOSAL_DIAGNOSTIC_DIR" "$C3_OVERLOAD_DIAGNOSTIC_DIR"

  python tools/bata/analyze_pc_ot_mras_selector_posttrain_diagnostics.py \
    --input "$C3_SELECTOR_METADATA_JSONL" \
    --output "$C3_SELECTOR_SUMMARY_JSON" \
    --provenance-run-root "$RUN_ROOT" \
    --provenance-work-dir "$FINAL_WORK_DIR" \
    --provenance-train-stdout "$TRAIN_STDOUT" \
    --provenance-result-detection-json "$C3_RESULT_DETECTION_JSON"

  python tools/bata/analyze_actionformer_result_detection_geometry.py \
    --result-detection-json "$C3_RESULT_DETECTION_JSON" \
    --output-dir "$C3_RESULT_DETECTION_GEOMETRY_DIR" \
    --annotation "$THUMOS14_ANNOTATION_PATH" \
    --class-map "$THUMOS14_CLASS_MAP" \
    --config "$CONFIG" \
    --group-by video
  test -s "$C3_RESULT_DETECTION_PROPOSALS_JSONL" || fail "missing converted proposals"

  python tools/bata/analyze_p2_proposal_localization.py \
    --proposal-jsonl "$C3_RESULT_DETECTION_PROPOSALS_JSONL" \
    --output-dir "$C3_PROPOSAL_DIAGNOSTIC_DIR" \
    --annotation "$THUMOS14_ANNOTATION_PATH" \
    --class-map "$THUMOS14_CLASS_MAP" \
    --config "$CONFIG" \
    --group-by video \
    --provenance-run-root "$RUN_ROOT" \
    --provenance-work-dir "$FINAL_WORK_DIR" \
    --provenance-train-stdout "$TRAIN_STDOUT" \
    --provenance-result-detection-json "$C3_RESULT_DETECTION_JSON"

  python tools/bata/analyze_actionformer_post_nms_overload.py \
    --train-log "$TRAIN_STDOUT" \
    --annotation "$THUMOS14_ANNOTATION_PATH" \
    --output-dir "$C3_OVERLOAD_DIAGNOSTIC_DIR" \
    --config "$CONFIG" \
    --result-detection-json "$C3_RESULT_DETECTION_JSON" \
    --provenance-run-root "$RUN_ROOT" \
    --provenance-work-dir "$FINAL_WORK_DIR" \
    --provenance-train-stdout "$TRAIN_STDOUT" \
    --provenance-result-detection-json "$C3_RESULT_DETECTION_JSON"

  python tools/bata/validate_pc_ot_mras_c3_diagnostic_gate.py \
    --selector-summary "$C3_SELECTOR_SUMMARY_JSON" \
    --proposal-summary "$C3_PROPOSAL_SUMMARY_JSON" \
    --overload-summary "$C3_OVERLOAD_SUMMARY_JSON" \
    --min-selector-samples "$C3_DIAGNOSTIC_GATE_MIN_SELECTOR_SAMPLES" \
    --expected-run-root "$RUN_ROOT" \
    --expected-work-dir "$FINAL_WORK_DIR" \
    --expected-train-stdout "$TRAIN_STDOUT" \
    --expected-result-detection-json "$C3_RESULT_DETECTION_JSON" \
    --output "$C3_DIAGNOSTIC_GATE_JSON"
fi

write_summary "PASS" "C3_PHYSICAL_GRID_FULL_TRAIN_PASS_WITH_DIAGNOSTIC_GATE" "training completed and current-run selector/proposal diagnostic gate passed"
log "C3_PHYSICAL_GRID_FULL_TRAIN_PASS_WITH_DIAGNOSTIC_GATE"
