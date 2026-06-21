#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH -J pcot_selviz
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
case "$REPO" in
  *OpenTAD_BATA_Clean*)
    echo "[pc-ot-mras-selection-learning] $(date -Is) ERROR refusing dirty historical OpenTAD_BATA_Clean path: $REPO" >&2
    exit 2
    ;;
esac
if [ ! -d "$REPO" ]; then
  echo "[pc-ot-mras-selection-learning] $(date -Is) ERROR missing clean PC-OT-MRAS repo: $REPO" >&2
  exit 2
fi
cd "$REPO"

PRECHECK_ONLY="${PRECHECK_ONLY:-1}"
ALLOW_PCOTMRAS_SELECTION_DIAGNOSTIC="${ALLOW_PCOTMRAS_SELECTION_DIAGNOSTIC:-0}"
MODEL_STAGE="${MODEL_STAGE:-}"
CHECKPOINT_TIER="${CHECKPOINT_TIER:-}"
SNAPSHOT_ID="${SNAPSHOT_ID:-}"
PCOTMRAS_SELECTION_CONFIG="${PCOTMRAS_SELECTION_CONFIG:-}"
PCOTMRAS_SELECTION_CONFIG_SHA256="${PCOTMRAS_SELECTION_CONFIG_SHA256:-}"
PCOTMRAS_SELECTION_CHECKPOINT="${PCOTMRAS_SELECTION_CHECKPOINT:-}"
PCOTMRAS_SELECTION_CHECKPOINT_SHA256="${PCOTMRAS_SELECTION_CHECKPOINT_SHA256:-}"
PCOTMRAS_SELECTION_GATE_JSON="${PCOTMRAS_SELECTION_GATE_JSON:-}"
PCOTMRAS_SELECTION_GATE_SHA256="${PCOTMRAS_SELECTION_GATE_SHA256:-}"
SPLIT="${SPLIT:-val}"
LIMIT_BATCHES="${LIMIT_BATCHES:-4}"
VIS_LIMIT="${VIS_LIMIT:-64}"
USE_EMA="${USE_EMA:-auto}"
USE_AMP="${USE_AMP:-0}"
DEVICE="${DEVICE:-auto}"
RUN_TAG="${RUN_TAG:-ctf_bdi_pc_ot_mras_selection_learning_${SLURM_JOB_ID:-nojob}_$(date '+%Y%m%d_%H%M%S')}"

case "$RUN_TAG" in
  *..*|*/*|*\\*|*[[:space:]]*|*[\;\&\|\>\<\`\"\'$]*)
    echo "[pc-ot-mras-selection-learning] $(date -Is) ERROR RUN_TAG must be path-safe; got $RUN_TAG" >&2
    exit 2
    ;;
esac

LOG_DIR="$REPO/logs"
OUTPUT_ROOT="${OUTPUT_ROOT:-$LOG_DIR/pcotmras_selection_learning_$RUN_TAG}"
OUTPUT_ROOT_ABS="$(readlink -m "$OUTPUT_ROOT")"
LOG_DIR_ABS="$(readlink -m "$LOG_DIR")"
case "$OUTPUT_ROOT_ABS" in
  "$LOG_DIR_ABS"/*) ;;
  *) echo "[pc-ot-mras-selection-learning] $(date -Is) ERROR OUTPUT_ROOT must stay under $LOG_DIR_ABS; got $OUTPUT_ROOT_ABS" >&2; exit 2 ;;
esac
RUN_LOG="$OUTPUT_ROOT_ABS/${RUN_TAG}.log"
SUMMARY_JSON="$OUTPUT_ROOT_ABS/selection_learning_diagnostic_summary.json"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export YUZIBO_ROOT
export OPENTAD_PCOTMRAS_CLEAN_ROOT="$REPO"
export THUMOS14_ANNOTATION_PATH="${THUMOS14_ANNOTATION_PATH:-$YUZIBO_ROOT/thumos14/annotations/thumos_14_anno.json}"
export THUMOS14_CLASS_MAP="${THUMOS14_CLASS_MAP:-$YUZIBO_ROOT/thumos14/annotations/category_idx.txt}"
export THUMOS14_TRAIN_DATA_PATH="${THUMOS14_TRAIN_DATA_PATH:-$YUZIBO_ROOT/thumos14/train}"
export THUMOS14_TEST_DATA_PATH="${THUMOS14_TEST_DATA_PATH:-$YUZIBO_ROOT/thumos14/test}"

mkdir -p "$OUTPUT_ROOT_ABS"
exec > >(tee -a "$RUN_LOG") 2>&1

log() {
  echo "[pc-ot-mras-selection-learning] $(date -Is) $*"
}

write_summary() {
  local status="$1"
  local decision="$2"
  local reason="$3"
  local snapshot_jsonl="${4:-}"
  local snapshot_summary="${5:-}"
  local visualization_summary="${6:-}"
  local visualization_dir="${7:-}"
  python - "$SUMMARY_JSON" "$status" "$decision" "$reason" "$MODEL_STAGE" "$CHECKPOINT_TIER" "$SNAPSHOT_ID" "$PCOTMRAS_SELECTION_CONFIG" "$PCOTMRAS_SELECTION_CHECKPOINT" "$OUTPUT_ROOT_ABS" "$snapshot_jsonl" "$snapshot_summary" "$visualization_summary" "$visualization_dir" <<'PY'
import json
import os
import socket
import sys

(
    summary_path,
    status,
    decision,
    reason,
    model_stage,
    checkpoint_tier,
    snapshot_id,
    config,
    checkpoint,
    output_root,
    snapshot_jsonl,
    snapshot_summary,
    visualization_summary,
    visualization_dir,
) = sys.argv[1:15]
payload = {
    "status": status,
    "decision": decision,
    "reason": reason,
    "stage": "PC_OT_MRAS_SELECTION_LEARNING_DIAGNOSTIC",
    "route": "CTF-BDI/PC-OT-MRAS",
    "host": socket.gethostname(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "model_stage": model_stage,
    "checkpoint_tier": checkpoint_tier,
    "snapshot_id": snapshot_id,
    "config": config,
    "checkpoint": checkpoint,
    "output_root": output_root,
    "snapshot_jsonl": snapshot_jsonl,
    "snapshot_summary_json": snapshot_summary,
    "visualization_summary_json": visualization_summary,
    "visualization_dir": visualization_dir,
    "claim_boundaries": {
        "diagnostic_only": True,
        "tools_train": False,
        "tools_test": False,
        "detector_map": False,
        "raw_prediction_cache": False,
        "metric_claim_allowed": False,
        "runtime_or_flops_claim_allowed": False,
        "deploy_claim_allowed": False,
        "paper_claim_allowed": False,
    },
}
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
    f.write("\n")
PY
  log "summary_json=$SUMMARY_JSON"
}

fail() {
  local message="$1"
  write_summary "FAIL" "PC_OT_MRAS_SELECTION_LEARNING_DIAGNOSTIC_NO_GO" "$message"
  log "ERROR $message"
  exit 2
}

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

require_file_sha() {
  local path="$1"
  local expected="$2"
  local label="$3"
  [ -n "$path" ] || fail "$label path is required"
  [ -f "$path" ] || fail "$label file not found: $path"
  [ -n "$expected" ] || fail "$label SHA256 is required"
  local actual
  actual="$(sha256_file "$path")"
  if [ "$actual" != "$expected" ]; then
    fail "$label SHA256 mismatch: expected $expected got $actual"
  fi
}

case "$PRECHECK_ONLY" in
  0|1) ;;
  *) fail "PRECHECK_ONLY must be exactly 0 or 1; got $PRECHECK_ONLY" ;;
esac
case "$ALLOW_PCOTMRAS_SELECTION_DIAGNOSTIC" in
  0|1) ;;
  *) fail "ALLOW_PCOTMRAS_SELECTION_DIAGNOSTIC must be 0 or 1; got $ALLOW_PCOTMRAS_SELECTION_DIAGNOSTIC" ;;
esac
case "$USE_AMP" in
  0|1) ;;
  *) fail "USE_AMP must be 0 or 1; got $USE_AMP" ;;
esac
case "$USE_EMA" in
  auto|none|0|1|true|false|yes|no|ema|raw) ;;
  *) fail "USE_EMA must be auto, true, or false compatible; got $USE_EMA" ;;
esac
case "$SPLIT" in
  val) ;;
  *) fail "selection-learning diagnostic is reviewed only for SPLIT=val; got $SPLIT" ;;
esac
case "$LIMIT_BATCHES" in
  ''|*[!0-9]*) fail "LIMIT_BATCHES must be numeric; got $LIMIT_BATCHES" ;;
esac
if [ "$LIMIT_BATCHES" -lt 1 ] || [ "$LIMIT_BATCHES" -gt 16 ]; then
  fail "LIMIT_BATCHES must be in [1,16]; got $LIMIT_BATCHES"
fi
case "$VIS_LIMIT" in
  ''|*[!0-9]*) fail "VIS_LIMIT must be numeric; got $VIS_LIMIT" ;;
esac
if [ "$VIS_LIMIT" -lt 1 ] || [ "$VIS_LIMIT" -gt 256 ]; then
  fail "VIS_LIMIT must be in [1,256]; got $VIS_LIMIT"
fi

if [ -n "${LOAD_FROM_RAW_PREDICTIONS:-}" ] || [ -n "${RAW_PREDICTION_CACHE:-}" ] || [ -n "${PREDICTION_CACHE:-}" ] || [ -n "${RESULT_DETECTION_JSON:-}" ]; then
  fail "raw prediction/cache/result-detection environment shortcuts are forbidden"
fi
if [ -n "${LOAD_FROM:-}" ] || [ -n "${RESUME:-}" ] || [ -n "${CHECKPOINT:-}" ]; then
  fail "generic checkpoint/load/resume environment shortcuts are forbidden; use PCOTMRAS_SELECTION_CHECKPOINT with SHA"
fi
if [ -n "${ALLOW_TOOLS_TRAIN:-}" ] || [ -n "${ALLOW_TOOLS_TEST:-}" ]; then
  fail "tools/train.py and tools/test.py are not part of this diagnostic launcher"
fi
if [ -z "${SLURM_JOB_ID:-}" ] && [ "${ALLOW_LOGIN_NODE_DEBUG:-0}" != "1" ]; then
  fail "refusing direct non-Slurm execution unless ALLOW_LOGIN_NODE_DEBUG=1 for lightweight PRECHECK_ONLY"
fi
if [ -z "${SLURM_JOB_ID:-}" ] && [ "${ALLOW_LOGIN_NODE_DEBUG:-0}" = "1" ] && [ "$PRECHECK_ONLY" != "1" ]; then
  fail "login-node debug must keep PRECHECK_ONLY=1; refusing checkpoint inference outside Slurm"
fi

for required_tool in \
  tools/bata/dump_pc_ot_mras_reader_snapshots.py \
  tools/bata/visualize_pc_ot_mras_selection.py; do
  [ -f "$required_tool" ] || fail "missing required diagnostic tool: $required_tool"
done

if [ "$PRECHECK_ONLY" = "1" ]; then
  if [ "$ALLOW_PCOTMRAS_SELECTION_DIAGNOSTIC" != "0" ]; then
    fail "PRECHECK_ONLY=1 must not set ALLOW_PCOTMRAS_SELECTION_DIAGNOSTIC=1"
  fi
  write_summary "PASS" "PC_OT_MRAS_SELECTION_LEARNING_DIAGNOSTIC_PRECHECK_PASS_NO_CHECKPOINT_READ" "precheck validated launcher boundaries only"
  exit 0
fi

if [ "$ALLOW_PCOTMRAS_SELECTION_DIAGNOSTIC" != "1" ]; then
  fail "PRECHECK_ONLY=0 requires ALLOW_PCOTMRAS_SELECTION_DIAGNOSTIC=1"
fi
case "$MODEL_STAGE" in
  R17|R18) ;;
  *) fail "MODEL_STAGE must be R17 or R18; got $MODEL_STAGE" ;;
esac
case "$CHECKPOINT_TIER" in
  early|middle|final) ;;
  *) fail "CHECKPOINT_TIER must be early, middle, or final; got $CHECKPOINT_TIER" ;;
esac
case "$SNAPSHOT_ID" in
  ""|*..*|*/*|*\\*|*[[:space:]]*|*[\;\&\|\>\<\`\"\'$]*)
    fail "SNAPSHOT_ID must be a non-empty path-safe label; got $SNAPSHOT_ID"
    ;;
esac
case "$MODEL_STAGE:$PCOTMRAS_SELECTION_CONFIG" in
  R17:configs/adatad/thumos/ctf_bdi_pc_ot_mras_r17_formal_train_candidate.py) ;;
  R18:configs/adatad/thumos/ctf_bdi_pc_ot_mras_r18_aux_formal_train_candidate.py) ;;
  *) fail "config does not match reviewed model stage: MODEL_STAGE=$MODEL_STAGE CONFIG=$PCOTMRAS_SELECTION_CONFIG" ;;
esac

require_file_sha "$PCOTMRAS_SELECTION_CONFIG" "$PCOTMRAS_SELECTION_CONFIG_SHA256" "config"
require_file_sha "$PCOTMRAS_SELECTION_CHECKPOINT" "$PCOTMRAS_SELECTION_CHECKPOINT_SHA256" "checkpoint"
require_file_sha "$PCOTMRAS_SELECTION_GATE_JSON" "$PCOTMRAS_SELECTION_GATE_SHA256" "selection diagnostic gate"

GATE_DECISION="$(python - "$PCOTMRAS_SELECTION_GATE_JSON" "$MODEL_STAGE" "$CHECKPOINT_TIER" "$PCOTMRAS_SELECTION_CONFIG_SHA256" "$PCOTMRAS_SELECTION_CHECKPOINT_SHA256" <<'PY'
import json
import sys

gate_path, model_stage, checkpoint_tier, config_sha, checkpoint_sha = sys.argv[1:6]
with open(gate_path, "r", encoding="utf-8") as f:
    gate = json.load(f)
if gate.get("decision") != "ALLOW_PCOTMRAS_SELECTION_LEARNING_DIAGNOSTIC":
    raise SystemExit("bad decision")
if gate.get("model_stage") != model_stage:
    raise SystemExit("model_stage mismatch")
if gate.get("checkpoint_tier") != checkpoint_tier:
    raise SystemExit("checkpoint_tier mismatch")
if gate.get("config_sha256") != config_sha:
    raise SystemExit("config_sha256 mismatch")
if gate.get("checkpoint_sha256") != checkpoint_sha:
    raise SystemExit("checkpoint_sha256 mismatch")
for key in (
    "tools_train",
    "tools_test",
    "detector_map",
    "raw_prediction_cache",
    "metric_claim",
    "paper_claim",
    "runtime_flops_claim",
    "deploy_claim",
):
    if gate.get(key) not in (False, 0, None):
        raise SystemExit(f"{key} must be false")
print(gate["decision"])
PY
)" || fail "selection diagnostic gate payload validation failed"
if [ "$GATE_DECISION" != "ALLOW_PCOTMRAS_SELECTION_LEARNING_DIAGNOSTIC" ]; then
  fail "unexpected gate decision: $GATE_DECISION"
fi

DIAG_DIR="$OUTPUT_ROOT_ABS/${MODEL_STAGE}_${CHECKPOINT_TIER}_${SNAPSHOT_ID}"
SNAPSHOT_JSONL="$DIAG_DIR/snapshot.jsonl"
SNAPSHOT_SUMMARY_JSON="$DIAG_DIR/snapshot_summary.json"
VISUALIZATION_SUMMARY_JSON="$DIAG_DIR/visualization_summary.json"
VISUALIZATION_DIR="$DIAG_DIR/viz"
ANALYSIS_MD="$DIAG_DIR/analysis.md"
mkdir -p "$DIAG_DIR" "$VISUALIZATION_DIR"

AMP_ARGS=()
if [ "$USE_AMP" = "1" ]; then
  AMP_ARGS=(--amp)
fi

log "running reader snapshot dump for $MODEL_STAGE $CHECKPOINT_TIER $SNAPSHOT_ID"
python tools/bata/dump_pc_ot_mras_reader_snapshots.py \
  --config "$PCOTMRAS_SELECTION_CONFIG" \
  --checkpoint "$PCOTMRAS_SELECTION_CHECKPOINT" \
  --output-jsonl "$SNAPSHOT_JSONL" \
  --summary-json "$SNAPSHOT_SUMMARY_JSON" \
  --split "$SPLIT" \
  --limit-batches "$LIMIT_BATCHES" \
  --device "$DEVICE" \
  --snapshot-id "$SNAPSHOT_ID" \
  --use-ema "$USE_EMA" \
  "${AMP_ARGS[@]}"

log "running selection visualization for $MODEL_STAGE $CHECKPOINT_TIER $SNAPSHOT_ID"
python tools/bata/visualize_pc_ot_mras_selection.py \
  --input-jsonl "$SNAPSHOT_JSONL" \
  --output-dir "$VISUALIZATION_DIR" \
  --summary-json "$VISUALIZATION_SUMMARY_JSON" \
  --snapshot-label "$SNAPSHOT_ID" \
  --limit "$VIS_LIMIT"

cat > "$ANALYSIS_MD" <<EOF
# PC-OT-MRAS Selection Learning Diagnostic

Status: diagnostic artifacts generated.

- Model: $MODEL_STAGE
- Checkpoint tier: $CHECKPOINT_TIER
- Snapshot id: $SNAPSHOT_ID
- Config: $PCOTMRAS_SELECTION_CONFIG
- Config SHA256: $PCOTMRAS_SELECTION_CONFIG_SHA256
- Checkpoint: $PCOTMRAS_SELECTION_CHECKPOINT
- Checkpoint SHA256: $PCOTMRAS_SELECTION_CHECKPOINT_SHA256
- Snapshot JSONL: $SNAPSHOT_JSONL
- Snapshot summary: $SNAPSHOT_SUMMARY_JSON
- Visualization summary: $VISUALIZATION_SUMMARY_JSON
- Visualization directory: $VISUALIZATION_DIR

This diagnostic does not run training, tools/test.py, detector mAP, raw prediction/cache flows, runtime/FLOPs claims, deployment claims, metric claims, or paper claims.
EOF

write_summary "PASS" "PC_OT_MRAS_SELECTION_LEARNING_DIAGNOSTIC_PASS_NO_MAP_NO_CLAIMS" "snapshot and visualization generated" "$SNAPSHOT_JSONL" "$SNAPSHOT_SUMMARY_JSON" "$VISUALIZATION_SUMMARY_JSON" "$VISUALIZATION_DIR"
