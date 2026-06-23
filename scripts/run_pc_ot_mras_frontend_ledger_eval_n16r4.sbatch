#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -J pcot_front_eval
#SBATCH -o logs/%x-%j.out

set -eo pipefail

source /etc/profile >/dev/null 2>&1 || true
set -u
module load cuda/11.8
module load miniforge3/24.11

YUZIBO_ROOT="${YUZIBO_ROOT:-$HOME/run/yuzibo}"
source activate "$YUZIBO_ROOT/conda_envs/opentad"

REPO="${OPENTAD_PCOTMRAS_CLEAN_ROOT:-$YUZIBO_ROOT/OpenTAD_PCOTMRAS_R16_R18_R20_Clean_20260619_1730}"
cd "$REPO"

CONFIG="${CONFIG:-configs/adatad/thumos/pc_ot_mras_frontend_hard_ledger_fixed50_adapter_n16r4.py}"
PC_OT_MRAS_CONFIG="${PC_OT_MRAS_CONFIG:-configs/adatad/thumos/ctf_bdi_pc_ot_mras_r18_aux_diag_candidate.py}"
PC_OT_MRAS_CHECKPOINT="${PC_OT_MRAS_CHECKPOINT:-}"
ADATAD_CHECKPOINT="${ADATAD_CHECKPOINT:-}"
PC_OT_MRAS_CHECKPOINT_SHA256="${PC_OT_MRAS_CHECKPOINT_SHA256:-}"
ADATAD_CHECKPOINT_SHA256="${ADATAD_CHECKPOINT_SHA256:-}"
SPLIT="${SPLIT:-test}"
BUDGET="${BUDGET:-384}"
REQUIRE_SELECTED_COUNT="${REQUIRE_SELECTED_COUNT:-$BUDGET}"
RUN_ID="${RUN_ID:-0}"
SEED="${SEED:-20260623}"
MASTER_PORT="${MASTER_PORT:-$((24000 + SLURM_JOB_ID % 20000))}"
RUN_TAG="${RUN_TAG:-pc_ot_mras_frontend_eval_$(date '+%Y%m%d_%H%M%S')}"
PRECHECK_ONLY="${PRECHECK_ONLY:-1}"
ALLOW_FRONTEND_ADATAD_EVAL="${ALLOW_FRONTEND_ADATAD_EVAL:-0}"
DUMP_LIMIT_BATCHES="${DUMP_LIMIT_BATCHES:-999999}"
DUMP_USE_AMP="${DUMP_USE_AMP:-1}"

if [[ -z "$RUN_TAG" || "$RUN_TAG" == *"/"* || "$RUN_TAG" == *"\\"* || "$RUN_TAG" == *".."* || ! "$RUN_TAG" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "[pcot-frontend-eval] ERROR unsafe RUN_TAG=$RUN_TAG" >&2
  exit 2
fi

LOG_DIR="$REPO/logs"
RUN_ROOT="$LOG_DIR/$RUN_TAG"
RUN_LOG="$RUN_ROOT/${RUN_TAG}.log"
WORK_DIR="$RUN_ROOT/test_workdir"
SNAPSHOT_JSONL="$RUN_ROOT/pc_ot_mras_reader_snapshots.jsonl"
SNAPSHOT_SUMMARY="$RUN_ROOT/pc_ot_mras_reader_snapshots.summary.json"
HARD_JSONL="$RUN_ROOT/pc_ot_mras_hard_positions.jsonl"
HARD_SUMMARY="$RUN_ROOT/pc_ot_mras_hard_positions.summary.json"
LEDGER_JSONL="$RUN_ROOT/pc_ot_mras_frontend_value_transport_ledger.jsonl"
LEDGER_SUMMARY="$RUN_ROOT/pc_ot_mras_frontend_value_transport_ledger.summary.json"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export YUZIBO_ROOT
export OPENTAD_PCOTMRAS_CLEAN_ROOT="$REPO"
export THUMOS14_ANNOTATION_PATH="${THUMOS14_ANNOTATION_PATH:-$YUZIBO_ROOT/thumos14/annotations/thumos_14_anno.json}"
export THUMOS14_CLASS_MAP="${THUMOS14_CLASS_MAP:-$YUZIBO_ROOT/thumos14/annotations/category_idx.txt}"
export THUMOS14_TEST_DATA_PATH="${THUMOS14_TEST_DATA_PATH:-$YUZIBO_ROOT/thumos14/test}"
export PC_OT_MRAS_FRONTEND_LEDGER_PATH="$LEDGER_JSONL"

mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_LOG") 2>&1

log() {
  echo "[pcot-frontend-eval] $(date -Is) $*"
}

fail() {
  log "ERROR $*"
  exit 2
}

require_file_sha() {
  local path="$1"
  local expected="$2"
  local label="$3"
  if [ -z "$expected" ]; then
    fail "${label} SHA256 must be provided explicitly before frontend AdaTAD evaluation"
  fi
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [ "$actual" != "$expected" ]; then
    fail "${label} SHA256 mismatch: expected=$expected actual=$actual"
  fi
  log "${label}_sha256=$actual"
}

if [ -z "$PC_OT_MRAS_CHECKPOINT" ]; then
  fail "PC_OT_MRAS_CHECKPOINT must be provided explicitly"
fi
if [ -z "$ADATAD_CHECKPOINT" ]; then
  fail "ADATAD_CHECKPOINT must be provided explicitly"
fi

log "host=$(hostname) job=${SLURM_JOB_ID:-none}"
log "repo=$REPO"
log "config=$CONFIG"
log "pc_ot_mras_config=$PC_OT_MRAS_CONFIG"
log "pc_ot_mras_checkpoint=$PC_OT_MRAS_CHECKPOINT"
log "adatad_checkpoint=$ADATAD_CHECKPOINT"
log "ledger=$PC_OT_MRAS_FRONTEND_LEDGER_PATH"
log "split=$SPLIT budget=$BUDGET require_selected_count=$REQUIRE_SELECTED_COUNT"
log "precheck_only=$PRECHECK_ONLY allow_frontend_adatad_eval=$ALLOW_FRONTEND_ADATAD_EVAL"
log "run_root=$RUN_ROOT"
log "work_dir=$WORK_DIR"
log "seed=$SEED master_port=$MASTER_PORT"
log "python=$(python -c 'import sys; print(sys.executable)')"
log "git_head=$(git rev-parse HEAD 2>/dev/null || echo unknown)"

test -f "$CONFIG" || fail "missing config: $CONFIG"
test -f "$PC_OT_MRAS_CONFIG" || fail "missing PC-OT-MRAS config: $PC_OT_MRAS_CONFIG"
test -f "$PC_OT_MRAS_CHECKPOINT" || fail "missing PC-OT-MRAS checkpoint: $PC_OT_MRAS_CHECKPOINT"
test -f "$ADATAD_CHECKPOINT" || fail "missing AdaTAD checkpoint: $ADATAD_CHECKPOINT"
test -f "$THUMOS14_ANNOTATION_PATH" || fail "missing annotation: $THUMOS14_ANNOTATION_PATH"
test -f "$THUMOS14_CLASS_MAP" || fail "missing class map: $THUMOS14_CLASS_MAP"
test -d "$THUMOS14_TEST_DATA_PATH" || fail "missing test data dir: $THUMOS14_TEST_DATA_PATH"

if command -v ss >/dev/null 2>&1; then
  if ss -ltn | awk '{print $4}' | grep -Eq "[:.]${MASTER_PORT}$"; then
    fail "MASTER_PORT=$MASTER_PORT is already in use"
  fi
else
  log "WARN ss not available; skipping master port preflight"
fi

log "static py_compile"
python -m py_compile \
  tools/bata/dump_pc_ot_mras_reader_snapshots.py \
  tools/bata/export_pc_ot_mras_hard_positions.py \
  tools/bata/convert_pc_ot_mras_hard_positions_to_value_transport_ledger.py \
  opentad/datasets/transforms/boundary_acquisition.py \
  opentad/datasets/transforms/end_to_end.py \
  opentad/models/utils/post_processing/utils.py \
  "$CONFIG"

log "focused pytest"
python -m pytest \
  tests/test_pc_ot_mras_frontend_ledger.py \
  tests/test_pc_ot_mras_hard_export_consistency.py \
  tests/test_bata_post_processing_selected_axis.py \
  -q

log "config contract preflight"
python - "$CONFIG" <<'PY'
import sys
from mmengine.config import Config

cfg = Config.fromfile(sys.argv[1])
assert int(cfg.window_size) == 384
assert int(cfg.dense_window_size) == 768
assert int(cfg.model.backbone.backbone.total_frames) == 384
assert int(cfg.model.projection.max_seq_len) == 384
assert "pc_ot_mras_reader" not in cfg.model
assert bool(cfg.inference.load_from_raw_predictions) is False
assert bool(cfg.inference.save_raw_prediction) is False
for split in ("val", "test"):
    lf = cfg.dataset[split].pipeline[2]
    assert lf.method == "bata_value_transport_ledger_subsample"
    assert lf.method_base == "sliding_window"
    assert int(lf.target_len) == 384
    assert bool(lf.bata_value_transport_allow_missing_fallback) is False
    assert bool(lf.bata_value_transport_require_deployable) is False
    assert lf.bata_value_transport_source == "pc_ot_mras_frontend_hard_positions"
print("pc_ot_mras_frontend_eval_config_preflight=PASS")
PY

if [ "$PRECHECK_ONLY" = "1" ]; then
  log "PRECHECK_ONLY=1; stopping before reader dump, ledger generation, and detector mAP"
  exit 0
fi

if [ "$ALLOW_FRONTEND_ADATAD_EVAL" != "1" ]; then
  fail "ALLOW_FRONTEND_ADATAD_EVAL=1 is required before dump, ledger generation, and detector mAP"
fi
require_file_sha "$PC_OT_MRAS_CHECKPOINT" "$PC_OT_MRAS_CHECKPOINT_SHA256" "pc_ot_mras_checkpoint"
require_file_sha "$ADATAD_CHECKPOINT" "$ADATAD_CHECKPOINT_SHA256" "adatad_checkpoint"

log "dumping PC-OT-MRAS reader snapshots"
DUMP_ARGS=(
  --config "$PC_OT_MRAS_CONFIG"
  --checkpoint "$PC_OT_MRAS_CHECKPOINT"
  --output-jsonl "$SNAPSHOT_JSONL"
  --summary-json "$SNAPSHOT_SUMMARY"
  --split "$SPLIT"
  --limit-batches "$DUMP_LIMIT_BATCHES"
  --budget "$BUDGET"
)
if [ "$DUMP_USE_AMP" = "1" ]; then
  DUMP_ARGS+=(--use-amp)
fi
python tools/bata/dump_pc_ot_mras_reader_snapshots.py "${DUMP_ARGS[@]}"

log "resolving hard PC-OT-MRAS positions"
python tools/bata/export_pc_ot_mras_hard_positions.py \
  --input-jsonl "$SNAPSHOT_JSONL" \
  --output-jsonl "$HARD_JSONL" \
  --summary-json "$HARD_SUMMARY" \
  --budget "$BUDGET"

log "converting hard positions to value-transport ledger"
CONVERT_ARGS=(
  --input-jsonl "$HARD_JSONL"
  --output-jsonl "$LEDGER_JSONL"
  --summary-json "$LEDGER_SUMMARY"
  --target-len "$BUDGET"
  --route-variant "pc_ot_mras_frontend_original_adatad_eval"
)
if [ -n "$REQUIRE_SELECTED_COUNT" ]; then
  CONVERT_ARGS+=(--require-selected-count "$REQUIRE_SELECTED_COUNT")
fi
python tools/bata/convert_pc_ot_mras_hard_positions_to_value_transport_ledger.py "${CONVERT_ARGS[@]}"

log "launching original AdaTAD detector evaluation with PC-OT-MRAS frontend ledger"
torchrun --nproc_per_node=1 --master_port="$MASTER_PORT" \
  tools/test.py "$CONFIG" \
  --checkpoint "$ADATAD_CHECKPOINT" \
  --id "$RUN_ID" \
  --seed "$SEED" \
  --cfg-options work_dir="$WORK_DIR"

log "PC-OT-MRAS frontend original-AdaTAD eval finished"
