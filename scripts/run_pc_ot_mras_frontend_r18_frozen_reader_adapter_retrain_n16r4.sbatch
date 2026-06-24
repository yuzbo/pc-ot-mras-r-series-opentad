#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --time=72:00:00
#SBATCH -J pcot_front_train
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
    echo "[pcot-frontend-retrain] $(date -Is) ERROR refusing dirty historical OpenTAD_BATA_Clean path: $REPO" >&2
    exit 2
    ;;
esac
if [ ! -d "$REPO" ]; then
  echo "[pcot-frontend-retrain] $(date -Is) ERROR missing clean PC-OT-MRAS repo: $REPO" >&2
  exit 2
fi
cd "$REPO"

CONFIG_REVIEWED="configs/adatad/thumos/pc_ot_mras_frontend_r18_frozen_reader_adapter_retrain_candidate_n16r4.py"
PC_OT_MRAS_CONFIG_REVIEWED="configs/adatad/thumos/pc_ot_mras_frontend_r18_sliding_reader_dump_n16r4.py"
CONFIG="${CONFIG:-$CONFIG_REVIEWED}"
PC_OT_MRAS_CONFIG="${PC_OT_MRAS_CONFIG:-$PC_OT_MRAS_CONFIG_REVIEWED}"
PC_OT_MRAS_CHECKPOINT="${PC_OT_MRAS_CHECKPOINT:-}"
PC_OT_MRAS_CHECKPOINT_SHA256="${PC_OT_MRAS_CHECKPOINT_SHA256:-}"
PRECHECK_ONLY="${PRECHECK_ONLY:-1}"
ALLOW_FRONTEND_RETRAIN="${ALLOW_FRONTEND_RETRAIN:-0}"
ALLOW_FRONTEND_RETRAIN_TOOLS_TEST="${ALLOW_FRONTEND_RETRAIN_TOOLS_TEST:-0}"
ALLOW_FRONTEND_RETRAIN_CHECKPOINT_LOAD="${ALLOW_FRONTEND_RETRAIN_CHECKPOINT_LOAD:-0}"
ALLOW_FRONTEND_RETRAIN_RAW_PREDICTION_CACHE="${ALLOW_FRONTEND_RETRAIN_RAW_PREDICTION_CACHE:-0}"
TRAIN_ID="${TRAIN_ID:-0}"
SEED="${SEED:-20260623}"
BUDGET="${BUDGET:-384}"
DENSE_WINDOW_SIZE="${DENSE_WINDOW_SIZE:-768}"
REQUIRE_SELECTED_COUNT="${REQUIRE_SELECTED_COUNT:-$BUDGET}"
DUMP_TRAIN_LIMIT_BATCHES="${DUMP_TRAIN_LIMIT_BATCHES:-0}"
DUMP_VAL_LIMIT_BATCHES="${DUMP_VAL_LIMIT_BATCHES:-0}"
DUMP_TEST_LIMIT_BATCHES="${DUMP_TEST_LIMIT_BATCHES:-0}"
DUMP_USE_AMP="${DUMP_USE_AMP:-1}"
PRETRAINED_PATH_REVIEWED="pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"
PRETRAINED_PATH="${PRETRAINED_PATH:-$PRETRAINED_PATH_REVIEWED}"
SLURM_JOB_ID_FOR_PORT="${SLURM_JOB_ID:-0}"
DEFAULT_MASTER_PORT=$((39000 + (SLURM_JOB_ID_FOR_PORT % 20000)))
REQUESTED_MASTER_PORT="${MASTER_PORT:-}"
if [ -n "$REQUESTED_MASTER_PORT" ]; then
  MASTER_PORT="$REQUESTED_MASTER_PORT"
  MASTER_PORT_SOURCE="environment"
else
  MASTER_PORT="$DEFAULT_MASTER_PORT"
  MASTER_PORT_SOURCE="default_from_slurm_job_id"
fi
RUN_TAG="${RUN_TAG:-frontend_retrain_r18_frozen_reader_${SLURM_JOB_ID:-nojob}_$(date '+%Y%m%d_%H%M%S')}"

case "$RUN_TAG" in
  *..*|*/*|*\\*|*[[:space:]]*|*[\;\&\|\>\<\`\"\'$]*)
    echo "[pcot-frontend-retrain] $(date -Is) ERROR RUN_TAG must be a simple path-safe name; got $RUN_TAG" >&2
    exit 2
    ;;
esac

LOG_DIR="$REPO/logs"
RUN_ROOT="$LOG_DIR/$RUN_TAG"
RUN_LOG="$RUN_ROOT/${RUN_TAG}.log"
TRAIN_STDOUT="$RUN_ROOT/${RUN_TAG}.train.stdout.log"
SHA_MANIFEST="$RUN_ROOT/active_sha256_manifest.txt"
RESOLVED_CONFIG_DUMP="$RUN_ROOT/resolved_config.py"
SUMMARY_JSON="$RUN_ROOT/frontend_retrain_summary.json"
FRONTEND_RETRAIN_PREDUMP_GATE_JSON="$RUN_ROOT/frontend_retrain_predump_gate.json"
FRONTEND_RETRAIN_GATE_JSON="$RUN_ROOT/frontend_retrain_entrypoint_gate.json"
WORK_DIR_BASE="$RUN_ROOT/train_workdir"
WORK_DIR="$WORK_DIR_BASE/gpu1_id${TRAIN_ID}"

TRAIN_SNAPSHOT_JSONL="$RUN_ROOT/train_reader_snapshots.jsonl"
TRAIN_SNAPSHOT_SUMMARY="$RUN_ROOT/train_reader_snapshots.summary.json"
TRAIN_HARD_JSONL="$RUN_ROOT/train_hard_positions.jsonl"
TRAIN_HARD_SUMMARY="$RUN_ROOT/train_hard_positions.summary.json"
TRAIN_LEDGER_JSONL="$RUN_ROOT/train_value_transport_ledger.jsonl"
TRAIN_LEDGER_SUMMARY="$RUN_ROOT/train_value_transport_ledger.summary.json"

VAL_SNAPSHOT_JSONL="$RUN_ROOT/val_reader_snapshots.jsonl"
VAL_SNAPSHOT_SUMMARY="$RUN_ROOT/val_reader_snapshots.summary.json"
VAL_HARD_JSONL="$RUN_ROOT/val_hard_positions.jsonl"
VAL_HARD_SUMMARY="$RUN_ROOT/val_hard_positions.summary.json"
VAL_LEDGER_JSONL="$RUN_ROOT/val_value_transport_ledger.jsonl"
VAL_LEDGER_SUMMARY="$RUN_ROOT/val_value_transport_ledger.summary.json"

TEST_SNAPSHOT_JSONL="$RUN_ROOT/test_reader_snapshots.jsonl"
TEST_SNAPSHOT_SUMMARY="$RUN_ROOT/test_reader_snapshots.summary.json"
TEST_HARD_JSONL="$RUN_ROOT/test_hard_positions.jsonl"
TEST_HARD_SUMMARY="$RUN_ROOT/test_hard_positions.summary.json"
TEST_LEDGER_JSONL="$RUN_ROOT/test_value_transport_ledger.jsonl"
TEST_LEDGER_SUMMARY="$RUN_ROOT/test_value_transport_ledger.summary.json"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export YUZIBO_ROOT
export OPENTAD_PCOTMRAS_CLEAN_ROOT="$REPO"
export THUMOS14_ANNOTATION_PATH="${THUMOS14_ANNOTATION_PATH:-$YUZIBO_ROOT/thumos14/annotations/thumos_14_anno.json}"
export THUMOS14_CLASS_MAP="${THUMOS14_CLASS_MAP:-$YUZIBO_ROOT/thumos14/annotations/category_idx.txt}"
export THUMOS14_TRAIN_DATA_PATH="${THUMOS14_TRAIN_DATA_PATH:-$YUZIBO_ROOT/thumos14/train}"
export THUMOS14_TEST_DATA_PATH="${THUMOS14_TEST_DATA_PATH:-$YUZIBO_ROOT/thumos14/test}"
export PC_OT_MRAS_FRONTEND_TRAIN_LEDGER_PATH="$TRAIN_LEDGER_JSONL"
export PC_OT_MRAS_FRONTEND_VAL_LEDGER_PATH="$VAL_LEDGER_JSONL"
export PC_OT_MRAS_FRONTEND_TEST_LEDGER_PATH="$TEST_LEDGER_JSONL"

mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_LOG") 2>&1

log() {
  echo "[pcot-frontend-retrain] $(date -Is) $*"
}

fail() {
  log "ERROR $*"
  exit 2
}

write_summary() {
  local status="$1"
  local decision="$2"
  local reason="$3"
  python - "$SUMMARY_JSON" "$status" "$decision" "$reason" "$RUN_ROOT" "$CONFIG" "$PC_OT_MRAS_CONFIG" "$WORK_DIR" <<'PY'
import json
import os
import socket
import sys
from pathlib import Path

summary_path, status, decision, reason, run_root, config, reader_config, work_dir = sys.argv[1:9]
payload = {
    "status": status,
    "decision": decision,
    "reason": reason,
    "stage": "PC_OT_MRAS_FRONTEND_R18_FROZEN_READER_ADATAD_RETRAIN",
    "route": "pc_ot_mras_frontend_original_adatad",
    "host": socket.gethostname(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "run_root": run_root,
    "config": config,
    "reader_dump_config": reader_config,
    "work_dir": work_dir,
    "claim_boundaries": {
        "frozen_reader_selector_only": True,
        "detector_retrained": True,
        "selector_gradient": False,
        "fixed_sliding_train_windows": True,
        "tools_train_only": True,
        "train_time_validation_map_allowed": True,
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
    write_summary "FAIL" "FRONTEND_RETRAIN_FAILED" "launcher exited before successful completion"
  fi
}
trap on_exit EXIT

require_file_sha() {
  local path="$1"
  local expected="$2"
  local label="$3"
  if [ -z "$expected" ]; then
    fail "${label} SHA256 must be provided explicitly"
  fi
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [ "$actual" != "$expected" ]; then
    fail "${label} SHA256 mismatch: expected=$expected actual=$actual"
  fi
  log "${label}_sha256=$actual"
}

case "$PRECHECK_ONLY" in
  0|1) ;;
  *) fail "PRECHECK_ONLY must be exactly 0 or 1; got $PRECHECK_ONLY" ;;
esac
case "$ALLOW_FRONTEND_RETRAIN_TOOLS_TEST" in
  0) ;;
  *) fail "direct tools/test.py is not part of this frontend retrain launcher" ;;
esac
case "$ALLOW_FRONTEND_RETRAIN_CHECKPOINT_LOAD" in
  0) ;;
  *) fail "checkpoint load/resume is not part of this first frontend retrain launch" ;;
esac
case "$ALLOW_FRONTEND_RETRAIN_RAW_PREDICTION_CACHE" in
  0) ;;
  *) fail "raw prediction/cache shortcuts are forbidden for frontend retrain" ;;
esac
case "$DUMP_USE_AMP" in
  0|1) ;;
  *) fail "DUMP_USE_AMP must be exactly 0 or 1; got $DUMP_USE_AMP" ;;
esac
for numeric_name in TRAIN_ID SEED BUDGET DENSE_WINDOW_SIZE REQUIRE_SELECTED_COUNT DUMP_TRAIN_LIMIT_BATCHES DUMP_VAL_LIMIT_BATCHES DUMP_TEST_LIMIT_BATCHES MASTER_PORT; do
  value="${!numeric_name}"
  case "$value" in
    ''|*[!0-9]*) fail "$numeric_name must be numeric; got $value" ;;
  esac
done
if [ "$BUDGET" -ne 384 ]; then
  fail "reviewed frontend retrain budget is 384; got $BUDGET"
fi
if [ "$DENSE_WINDOW_SIZE" -ne 768 ]; then
  fail "reviewed frontend retrain dense window size is 768; got $DENSE_WINDOW_SIZE"
fi
if [ "$REQUIRE_SELECTED_COUNT" -ne "$BUDGET" ]; then
  fail "REQUIRE_SELECTED_COUNT must equal BUDGET for fixed-50 retrain; got $REQUIRE_SELECTED_COUNT vs $BUDGET"
fi
if [ "$MASTER_PORT" -lt 1024 ] || [ "$MASTER_PORT" -gt 65535 ]; then
  if [ "$MASTER_PORT_SOURCE" = "environment" ]; then
    log "WARN ignoring inherited MASTER_PORT=$MASTER_PORT outside [1024,65535]; using default $DEFAULT_MASTER_PORT"
    MASTER_PORT="$DEFAULT_MASTER_PORT"
  else
    fail "MASTER_PORT must be between 1024 and 65535; got $MASTER_PORT"
  fi
fi
if [ "$CONFIG" != "$CONFIG_REVIEWED" ]; then
  fail "launcher is reviewed only for $CONFIG_REVIEWED; got $CONFIG"
fi
if [ "$PC_OT_MRAS_CONFIG" != "$PC_OT_MRAS_CONFIG_REVIEWED" ]; then
  fail "launcher is reviewed only for reader dump config $PC_OT_MRAS_CONFIG_REVIEWED; got $PC_OT_MRAS_CONFIG"
fi
if [ -z "${SLURM_JOB_ID:-}" ] && [ "${ALLOW_LOGIN_NODE_DEBUG:-0}" != "1" ]; then
  fail "refusing direct non-Slurm execution unless ALLOW_LOGIN_NODE_DEBUG=1 for lightweight PRECHECK_ONLY"
fi
if [ -z "${SLURM_JOB_ID:-}" ] && [ "${ALLOW_LOGIN_NODE_DEBUG:-0}" = "1" ] && [ "$PRECHECK_ONLY" != "1" ]; then
  fail "login-node debug must keep PRECHECK_ONLY=1; refusing training outside Slurm"
fi
if [ "$PRECHECK_ONLY" = "0" ]; then
  case "$ALLOW_FRONTEND_RETRAIN" in
    1) ;;
    *) fail "PRECHECK_ONLY=0 requires ALLOW_FRONTEND_RETRAIN=1 after review gate" ;;
  esac
  if [ "$DUMP_TRAIN_LIMIT_BATCHES" -ne 0 ] || [ "$DUMP_VAL_LIMIT_BATCHES" -ne 0 ] || [ "$DUMP_TEST_LIMIT_BATCHES" -ne 0 ]; then
    fail "DUMP_TRAIN_LIMIT_BATCHES/DUMP_VAL_LIMIT_BATCHES/DUMP_TEST_LIMIT_BATCHES must all be 0 for formal frontend retrain ledger generation"
  fi
else
  case "$ALLOW_FRONTEND_RETRAIN" in
    0) ;;
    *) fail "PRECHECK_ONLY=1 must not set ALLOW_FRONTEND_RETRAIN=1" ;;
  esac
fi
if [ -n "${CHECKPOINT:-}" ] || [ -n "${CHECKPOINT_SHA256:-}" ] || [ -n "${LOAD_FROM:-}" ] || [ -n "${RESUME:-}" ]; then
  fail "external checkpoint/load/resume environment shortcuts are forbidden for frontend retrain"
fi
if [ "$PRETRAINED_PATH" != "$PRETRAINED_PATH_REVIEWED" ]; then
  fail "PRETRAINED_PATH override is forbidden for frontend retrain; reviewed path is $PRETRAINED_PATH_REVIEWED"
fi
if [ -n "${LOAD_FROM_RAW_PREDICTIONS:-}" ] || [ -n "${RAW_PREDICTION_CACHE:-}" ] || [ -n "${PREDICTION_CACHE:-}" ]; then
  fail "raw prediction/cache environment shortcuts are forbidden for frontend retrain"
fi
if [ -n "${CFG_OPTIONS:-}" ] || [ -n "${EXTRA_CFG_OPTIONS:-}" ]; then
  fail "arbitrary cfg-options are forbidden for frontend retrain"
fi

EXPECTED_GIT_BRANCH="${OPENTAD_PCOTMRAS_EXPECTED_BRANCH:-codex/pcotmras-r16-r18-r20-clean}"
GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)" || fail "clean repo must be a git checkout"
GIT_HEAD="$(git rev-parse HEAD 2>/dev/null)" || fail "clean repo HEAD is unavailable"
if [ "$GIT_BRANCH" != "$EXPECTED_GIT_BRANCH" ]; then
  fail "expected branch $EXPECTED_GIT_BRANCH; got $GIT_BRANCH"
fi
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  fail "tracked clean repo files are modified; refusing execution"
fi

log "host=$(hostname) job=${SLURM_JOB_ID:-none}"
log "repo=$REPO"
log "git_branch=$GIT_BRANCH git_head=$GIT_HEAD"
log "config=$CONFIG"
log "reader_dump_config=$PC_OT_MRAS_CONFIG"
log "pc_ot_mras_checkpoint=$PC_OT_MRAS_CHECKPOINT"
log "run_root=$RUN_ROOT"
log "work_dir=$WORK_DIR"
log "train_ledger=$TRAIN_LEDGER_JSONL"
log "val_ledger=$VAL_LEDGER_JSONL"
log "test_ledger=$TEST_LEDGER_JSONL"
log "seed=$SEED train_id=$TRAIN_ID master_port=$MASTER_PORT"
log "precheck_only=$PRECHECK_ONLY allow_frontend_retrain=$ALLOW_FRONTEND_RETRAIN"
log "scope=frozen_reader_selector detector_retrain fixed_sliding_train_windows direct_tools_test=false no_claims=true"
log "python=$(python -c 'import sys; print(sys.executable)')"

test -f "$CONFIG" || fail "missing reviewed config: $CONFIG"
test -f "$PC_OT_MRAS_CONFIG" || fail "missing reviewed reader dump config: $PC_OT_MRAS_CONFIG"

log "dumping and auditing resolved config=$RESOLVED_CONFIG_DUMP"
python - "$CONFIG" "$PC_OT_MRAS_CONFIG" "$RESOLVED_CONFIG_DUMP" <<'PY'
import os
import sys
from pathlib import Path

from mmengine.config import Config
from opentad.utils.training_guard import assert_detector_training_allowed


def require(condition, message):
    if not condition:
        raise SystemExit(str(message))


cfg = Config.fromfile(sys.argv[1])
reader_cfg = Config.fromfile(sys.argv[2])
Path(sys.argv[3]).write_text(cfg.pretty_text + "\n", encoding="utf-8")

require(int(cfg.window_size) == 384, cfg.window_size)
require(int(cfg.dense_window_size) == 768, cfg.dense_window_size)
require("pc_ot_mras_reader" not in cfg.model, cfg.model)
require(cfg.model.rpn_head.type == "ActionFormerHead", cfg.model.rpn_head)
require(int(cfg.model.backbone.backbone.total_frames) == 384, cfg.model.backbone.backbone)
require(int(cfg.model.projection.max_seq_len) == 384, cfg.model.projection)
gate = cfg.pc_ot_mras_frontend_retrain_gate
require(gate.launch_gate_passed is True, gate)
require(gate.allow_detector_training is True, gate)
require(gate.allow_tools_train is True, gate)
require(gate.allow_tools_test is False, gate)
require(gate.allow_detector_map is False, gate)
require(gate.allow_train_validation_map is True, gate)
require(gate.allow_long_training is True, gate)
require(gate.entrypoint_gate_context.required is True, gate)
require(tuple(gate.allowed_entrypoints) == ("tools/train.py",), gate)
require(int(cfg.workflow.end_epoch) == 60, cfg.workflow)
require(int(cfg.workflow.val_start_epoch) == 40, cfg.workflow)
require(int(cfg.workflow.val_eval_interval) == 2, cfg.workflow)
require(cfg.inference.load_from_raw_predictions is False, cfg.inference)
require(cfg.inference.save_raw_prediction is False, cfg.inference)
for split, env_key in (
    ("train", "PC_OT_MRAS_FRONTEND_TRAIN_LEDGER_PATH"),
    ("val", "PC_OT_MRAS_FRONTEND_VAL_LEDGER_PATH"),
    ("test", "PC_OT_MRAS_FRONTEND_TEST_LEDGER_PATH"),
):
    ds = cfg.dataset[split]
    lf = ds.pipeline[2]
    require(ds.type == "ThumosSlidingDataset", ds)
    require(int(ds.window_size) == 768, ds)
    require(lf.method == "bata_value_transport_ledger_subsample", lf)
    require(lf.method_base == "sliding_window", lf)
    require(int(lf.target_len) == 384, lf)
    require(lf.bata_value_transport_ledger_path == os.environ[env_key], lf)
    require(bool(lf.bata_value_transport_allow_missing_fallback) is False, lf)
    require(bool(lf.bata_value_transport_require_deployable) is True, lf)
    require(int(lf.bata_value_transport_require_selected_count) == 384, lf)
for split in ("train", "val", "test"):
    ds = reader_cfg.dataset[split]
    require(ds.type == "ThumosSlidingDataset", ds)
    require(int(ds.window_size) == 768, ds)
    require(ds.pipeline[2].method == "sliding_window", ds.pipeline)
require(int(reader_cfg.model.pc_ot_mras_reader.num_slots) == 384, reader_cfg.model.pc_ot_mras_reader)
require(int(reader_cfg.model.projection.max_seq_len) == 768, reader_cfg.model.projection)
try:
    assert_detector_training_allowed(cfg, entrypoint="tools/test.py")
except RuntimeError:
    pass
else:
    raise SystemExit("frontend retrain config must reject direct tools/test.py")
for key in (
    "OPENTAD_PCOTMRAS_FRONTEND_RETRAIN_GATE_JSON",
    "OPENTAD_PCOTMRAS_FRONTEND_RETRAIN_GATE_SHA256",
    "OPENTAD_PCOTMRAS_FRONTEND_RETRAIN_ACTIVE_MANIFEST_SHA256",
    "OPENTAD_PCOTMRAS_FRONTEND_RETRAIN_RESOLVED_CONFIG_SHA256",
):
    os.environ.pop(key, None)
try:
    assert_detector_training_allowed(cfg, entrypoint="tools/train.py")
except RuntimeError as exc:
    if "missing required entrypoint gate env" not in str(exc):
        raise
else:
    raise SystemExit("frontend retrain config must require entrypoint gate envs")
print("pc_ot_mras_frontend_retrain_config_preflight=PASS")
PY
RESOLVED_CONFIG_SHA256="$(sha256sum "$RESOLVED_CONFIG_DUMP" | awk '{print $1}')"
log "resolved_config_sha256=$RESOLVED_CONFIG_SHA256"

mapfile -t CONFIG_DEP_FILES < <(python - "$CONFIG" "$PC_OT_MRAS_CONFIG" <<'PY'
import ast
import sys
from pathlib import Path

seen = []


def visit(path):
    path = Path(path).resolve()
    if path in seen:
        return
    seen.append(path)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "_base_" for target in node.targets):
            continue
        value = ast.literal_eval(node.value)
        bases = [value] if isinstance(value, str) else list(value)
        for base in bases:
            visit(path.parent / base)


for item in sys.argv[1:]:
    visit(Path(item))
for path in seen:
    print(path.relative_to(Path.cwd()).as_posix())
PY
)
log "resolved_config_dependency_count=${#CONFIG_DEP_FILES[@]}"
log "writing active SHA256 manifest=$SHA_MANIFEST"
sha256sum \
  "${CONFIG_DEP_FILES[@]}" \
  "$RESOLVED_CONFIG_DUMP" \
  scripts/run_pc_ot_mras_frontend_r18_frozen_reader_adapter_retrain_n16r4.sbatch \
  tools/train.py \
  tools/test.py \
  tools/bata/dump_pc_ot_mras_reader_snapshots.py \
  tools/bata/export_pc_ot_mras_hard_positions.py \
  tools/bata/convert_pc_ot_mras_hard_positions_to_value_transport_ledger.py \
  tools/bata/validate_pc_ot_mras_frontend_retrain_gate.py \
  opentad/datasets/transforms/boundary_acquisition.py \
  opentad/datasets/transforms/end_to_end.py \
  opentad/models/utils/post_processing/utils.py \
  opentad/utils/training_guard.py \
  tests/test_pc_ot_mras_frontend_retrain.py \
  tests/test_pc_ot_mras_frontend_ledger.py \
  > "$SHA_MANIFEST"
ACTIVE_MANIFEST_SHA256="$(sha256sum "$SHA_MANIFEST" | awk '{print $1}')"
log "active_sha256_manifest_sha256=$ACTIVE_MANIFEST_SHA256"

log "static py_compile"
python -m py_compile \
  "$CONFIG" \
  "$PC_OT_MRAS_CONFIG" \
  tools/train.py \
  tools/test.py \
  tools/bata/dump_pc_ot_mras_reader_snapshots.py \
  tools/bata/export_pc_ot_mras_hard_positions.py \
  tools/bata/convert_pc_ot_mras_hard_positions_to_value_transport_ledger.py \
  tools/bata/validate_pc_ot_mras_frontend_retrain_gate.py \
  opentad/datasets/transforms/boundary_acquisition.py \
  opentad/datasets/transforms/end_to_end.py \
  opentad/models/utils/post_processing/utils.py \
  opentad/utils/training_guard.py \
  tests/test_pc_ot_mras_frontend_retrain.py \
  tests/test_pc_ot_mras_frontend_ledger.py

log "focused frontend retrain tests"
python -m pytest \
  tests/test_pc_ot_mras_frontend_retrain.py \
  tests/test_pc_ot_mras_frontend_ledger.py \
  -q

if [ "$PRECHECK_ONLY" = "1" ]; then
  write_summary "PASS" "FRONTEND_RETRAIN_PRECHECK_ONLY_PASS_NO_TRAIN" "PRECHECK_ONLY=1; exiting before checkpoint/data access, ledger generation, and tools/train.py"
  log "PRECHECK_ONLY=1; exiting before frontend retrain execution"
  exit 0
fi

if [ -z "${SLURM_JOB_ID:-}" ]; then
  fail "PRECHECK_ONLY=0 requires a Slurm allocation"
fi
if [ -z "$PC_OT_MRAS_CHECKPOINT" ]; then
  fail "PC_OT_MRAS_CHECKPOINT must be provided explicitly"
fi
test -f "$PC_OT_MRAS_CHECKPOINT" || fail "missing PC-OT-MRAS checkpoint: $PC_OT_MRAS_CHECKPOINT"
test -f "$PRETRAINED_PATH" || fail "missing reviewed pretrained initialization file: $PRETRAINED_PATH"
test -f "$THUMOS14_ANNOTATION_PATH" || fail "missing annotation: $THUMOS14_ANNOTATION_PATH"
test -f "$THUMOS14_CLASS_MAP" || fail "missing class map: $THUMOS14_CLASS_MAP"
test -d "$THUMOS14_TRAIN_DATA_PATH" || fail "missing train data dir: $THUMOS14_TRAIN_DATA_PATH"
test -d "$THUMOS14_TEST_DATA_PATH" || fail "missing test data dir: $THUMOS14_TEST_DATA_PATH"
if command -v ss >/dev/null 2>&1; then
  if ss -ltn | awk '{print $4}' | grep -Eq "[:.]${MASTER_PORT}$"; then
    fail "MASTER_PORT=$MASTER_PORT is already in use"
  fi
else
  log "WARN ss not available; skipping master port preflight"
fi
require_file_sha "$PC_OT_MRAS_CHECKPOINT" "$PC_OT_MRAS_CHECKPOINT_SHA256" "pc_ot_mras_checkpoint"
PRETRAINED_SHA256="$(sha256sum "$PRETRAINED_PATH" | awk '{print $1}')"
log "pretrained_sha256=$PRETRAINED_SHA256"

log "writing frontend retrain predump gate=$FRONTEND_RETRAIN_PREDUMP_GATE_JSON"
python - "$FRONTEND_RETRAIN_PREDUMP_GATE_JSON" "$ACTIVE_MANIFEST_SHA256" "$RESOLVED_CONFIG_SHA256" "$PC_OT_MRAS_CHECKPOINT_SHA256" "$PRETRAINED_SHA256" "$RUN_TAG" <<'PY'
import json
import sys
from pathlib import Path

out_path, active_manifest, resolved_config, pcot_ckpt, pretrained, run_tag = sys.argv[1:7]
payload = {
    "decision": "ALLOW_PC_OT_MRAS_FRONTEND_R18_FROZEN_READER_LEDGER_GENERATION",
    "route": "pc_ot_mras_frontend_original_adatad",
    "active_sha256_manifest_sha256": active_manifest,
    "resolved_config_sha256": resolved_config,
    "pc_ot_mras_checkpoint_sha256": pcot_ckpt,
    "pretrained_sha256": pretrained,
    "budget": 384,
    "dense_window_size": 768,
    "allow_slurm": True,
    "allow_gpu": True,
    "single_gpu": True,
    "allow_reader_dump": True,
    "allow_hard_position_export": True,
    "allow_value_transport_ledger": True,
    "allow_dataset_access": True,
    "allow_tools_train": False,
    "tools_test": False,
    "allow_tools_test": False,
    "direct_tools_test": False,
    "detector_map": False,
    "allow_detector_map": False,
    "checkpoint_load": False,
    "allow_checkpoint_load": False,
    "resume": False,
    "allow_resume": False,
    "raw_prediction_cache": False,
    "allow_raw_prediction_cache": False,
    "load_from_raw_predictions": False,
    "save_raw_prediction": False,
    "uses_gt": False,
    "uses_teacher": False,
    "uses_oracle": False,
    "uses_raw_prediction": False,
    "metric_claim": False,
    "metric_claim_allowed": False,
    "paper_claim": False,
    "paper_claim_allowed": False,
    "runtime_flops_claim": False,
    "runtime_flops_claim_allowed": False,
    "deploy_claim": False,
    "deploy_claim_allowed": False,
    "run_tag": run_tag,
}
Path(out_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
FRONTEND_RETRAIN_PREDUMP_GATE_SHA256="$(sha256sum "$FRONTEND_RETRAIN_PREDUMP_GATE_JSON" | awk '{print $1}')"
python tools/bata/validate_pc_ot_mras_frontend_retrain_gate.py \
  --mode predump \
  --gate-json "$FRONTEND_RETRAIN_PREDUMP_GATE_JSON" \
  --gate-sha256 "$FRONTEND_RETRAIN_PREDUMP_GATE_SHA256" \
  --active-manifest-sha256 "$ACTIVE_MANIFEST_SHA256" \
  --resolved-config-sha256 "$RESOLVED_CONFIG_SHA256" \
  --pc-ot-mras-checkpoint-sha256 "$PC_OT_MRAS_CHECKPOINT_SHA256" \
  --pretrained-sha256 "$PRETRAINED_SHA256" \
  --budget "$BUDGET" \
  --dense-window-size "$DENSE_WINDOW_SIZE"

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
if not payload["cuda_available"] or payload["device_count"] < 1:
    raise SystemExit("frontend retrain requires at least one visible CUDA device")
PY
nvidia-smi || fail "nvidia-smi failed"

generate_split_ledger() {
  local split_name="$1"
  local limit_batches="$2"
  local snapshot_jsonl="$3"
  local snapshot_summary="$4"
  local hard_jsonl="$5"
  local hard_summary="$6"
  local ledger_jsonl="$7"
  local ledger_summary="$8"
  log "generate_split_ledger $split_name"
  local dump_args=(
    --config "$PC_OT_MRAS_CONFIG"
    --checkpoint "$PC_OT_MRAS_CHECKPOINT"
    --output-jsonl "$snapshot_jsonl"
    --summary-json "$snapshot_summary"
    --split "$split_name"
    --limit-batches "$limit_batches"
    --budget "$BUDGET"
    --dense-window-size "$DENSE_WINDOW_SIZE"
    --reader-dump-gate-mode frontend_retrain_predump
    --reader-dump-gate-json "$FRONTEND_RETRAIN_PREDUMP_GATE_JSON"
    --reader-dump-gate-sha256 "$FRONTEND_RETRAIN_PREDUMP_GATE_SHA256"
    --active-manifest-sha256 "$ACTIVE_MANIFEST_SHA256"
    --resolved-config-sha256 "$RESOLVED_CONFIG_SHA256"
    --pc-ot-mras-checkpoint-sha256 "$PC_OT_MRAS_CHECKPOINT_SHA256"
    --pretrained-sha256 "$PRETRAINED_SHA256"
    --require-selected-count "$REQUIRE_SELECTED_COUNT"
  )
  if [ "$DUMP_USE_AMP" = "1" ]; then
    dump_args+=(--use-amp)
  fi
  python tools/bata/dump_pc_ot_mras_reader_snapshots.py "${dump_args[@]}"
  python tools/bata/export_pc_ot_mras_hard_positions.py \
    --input-jsonl "$snapshot_jsonl" \
    --output-jsonl "$hard_jsonl" \
    --summary-json "$hard_summary" \
    --budget "$BUDGET"
  python tools/bata/convert_pc_ot_mras_hard_positions_to_value_transport_ledger.py \
    --input-jsonl "$hard_jsonl" \
    --output-jsonl "$ledger_jsonl" \
    --summary-json "$ledger_summary" \
    --target-len "$BUDGET" \
    --require-selected-count "$REQUIRE_SELECTED_COUNT" \
    --deploy-selection-ledger \
    --route-variant "pc_ot_mras_frontend_r18_frozen_reader_retrain_${split_name}"
}

generate_split_ledger train "$DUMP_TRAIN_LIMIT_BATCHES" "$TRAIN_SNAPSHOT_JSONL" "$TRAIN_SNAPSHOT_SUMMARY" "$TRAIN_HARD_JSONL" "$TRAIN_HARD_SUMMARY" "$TRAIN_LEDGER_JSONL" "$TRAIN_LEDGER_SUMMARY"
generate_split_ledger val "$DUMP_VAL_LIMIT_BATCHES" "$VAL_SNAPSHOT_JSONL" "$VAL_SNAPSHOT_SUMMARY" "$VAL_HARD_JSONL" "$VAL_HARD_SUMMARY" "$VAL_LEDGER_JSONL" "$VAL_LEDGER_SUMMARY"
generate_split_ledger test "$DUMP_TEST_LIMIT_BATCHES" "$TEST_SNAPSHOT_JSONL" "$TEST_SNAPSHOT_SUMMARY" "$TEST_HARD_JSONL" "$TEST_HARD_SUMMARY" "$TEST_LEDGER_JSONL" "$TEST_LEDGER_SUMMARY"

TRAIN_LEDGER_SHA256="$(sha256sum "$TRAIN_LEDGER_JSONL" | awk '{print $1}')"
VAL_LEDGER_SHA256="$(sha256sum "$VAL_LEDGER_JSONL" | awk '{print $1}')"
TEST_LEDGER_SHA256="$(sha256sum "$TEST_LEDGER_JSONL" | awk '{print $1}')"
log "train_ledger_sha256=$TRAIN_LEDGER_SHA256"
log "val_ledger_sha256=$VAL_LEDGER_SHA256"
log "test_ledger_sha256=$TEST_LEDGER_SHA256"

log "writing frontend retrain entrypoint gate=$FRONTEND_RETRAIN_GATE_JSON"
python - "$FRONTEND_RETRAIN_GATE_JSON" "$ACTIVE_MANIFEST_SHA256" "$RESOLVED_CONFIG_SHA256" "$PC_OT_MRAS_CHECKPOINT_SHA256" "$PRETRAINED_SHA256" "$TRAIN_LEDGER_SHA256" "$VAL_LEDGER_SHA256" "$TEST_LEDGER_SHA256" "$RUN_TAG" <<'PY'
import json
import sys
from pathlib import Path

(
    out_path,
    active_manifest,
    resolved_config,
    pcot_ckpt,
    pretrained,
    train_ledger,
    val_ledger,
    test_ledger,
    run_tag,
) = sys.argv[1:10]
payload = {
    "decision": "ALLOW_PC_OT_MRAS_FRONTEND_R18_FROZEN_READER_ADATAD_RETRAIN",
    "route": "pc_ot_mras_frontend_original_adatad",
    "active_sha256_manifest_sha256": active_manifest,
    "resolved_config_sha256": resolved_config,
    "pc_ot_mras_checkpoint_sha256": pcot_ckpt,
    "pretrained_sha256": pretrained,
    "train_ledger_sha256": train_ledger,
    "val_ledger_sha256": val_ledger,
    "test_ledger_sha256": test_ledger,
    "budget": 384,
    "dense_window_size": 768,
    "max_epochs": 60,
    "val_start_epoch": 40,
    "val_eval_interval": 2,
    "allow_slurm": True,
    "allow_gpu": True,
    "single_gpu": True,
    "allow_reader_dump": True,
    "allow_hard_position_export": True,
    "allow_value_transport_ledger": True,
    "allow_tools_train": True,
    "allow_dataset_access": True,
    "allow_pretrained_initialization": True,
    "allow_checkpoint_write": True,
    "allow_train_validation_map": True,
    "allow_long_training": True,
    "tools_test": False,
    "allow_tools_test": False,
    "direct_tools_test": False,
    "detector_map": False,
    "allow_detector_map": False,
    "checkpoint_load": False,
    "allow_checkpoint_load": False,
    "resume": False,
    "allow_resume": False,
    "raw_prediction_cache": False,
    "allow_raw_prediction_cache": False,
    "load_from_raw_predictions": False,
    "save_raw_prediction": False,
    "uses_gt": False,
    "uses_teacher": False,
    "uses_oracle": False,
    "uses_raw_prediction": False,
    "metric_claim": False,
    "metric_claim_allowed": False,
    "paper_claim": False,
    "paper_claim_allowed": False,
    "runtime_flops_claim": False,
    "runtime_flops_claim_allowed": False,
    "deploy_claim": False,
    "deploy_claim_allowed": False,
    "run_tag": run_tag,
}
Path(out_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
FRONTEND_RETRAIN_GATE_SHA256="$(sha256sum "$FRONTEND_RETRAIN_GATE_JSON" | awk '{print $1}')"
log "frontend_retrain_gate_sha256=$FRONTEND_RETRAIN_GATE_SHA256"

python tools/bata/validate_pc_ot_mras_frontend_retrain_gate.py \
  --gate-json "$FRONTEND_RETRAIN_GATE_JSON" \
  --gate-sha256 "$FRONTEND_RETRAIN_GATE_SHA256" \
  --active-manifest-sha256 "$ACTIVE_MANIFEST_SHA256" \
  --resolved-config-sha256 "$RESOLVED_CONFIG_SHA256" \
  --pc-ot-mras-checkpoint-sha256 "$PC_OT_MRAS_CHECKPOINT_SHA256" \
  --pretrained-sha256 "$PRETRAINED_SHA256" \
  --train-ledger-sha256 "$TRAIN_LEDGER_SHA256" \
  --val-ledger-sha256 "$VAL_LEDGER_SHA256" \
  --test-ledger-sha256 "$TEST_LEDGER_SHA256" \
  --budget "$BUDGET" \
  --dense-window-size "$DENSE_WINDOW_SIZE"

mkdir -p "$WORK_DIR"
export OPENTAD_PCOTMRAS_FRONTEND_RETRAIN_GATE_JSON="$FRONTEND_RETRAIN_GATE_JSON"
export OPENTAD_PCOTMRAS_FRONTEND_RETRAIN_GATE_SHA256="$FRONTEND_RETRAIN_GATE_SHA256"
export OPENTAD_PCOTMRAS_FRONTEND_RETRAIN_ACTIVE_MANIFEST_SHA256="$ACTIVE_MANIFEST_SHA256"
export OPENTAD_PCOTMRAS_FRONTEND_RETRAIN_RESOLVED_CONFIG_SHA256="$RESOLVED_CONFIG_SHA256"
export PC_OT_MRAS_FRONTEND_RETRAIN_PCOT_CKPT_PATH="$PC_OT_MRAS_CHECKPOINT"
export PC_OT_MRAS_FRONTEND_RETRAIN_PRETRAINED_PATH="$PRETRAINED_PATH"
log "launching frontend retrain tools/train.py"
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
RAW_CACHE_STDOUT_PATTERN="['\"]?load_from_raw_predictions['\"]?[[:space:]]*[:=][[:space:]]*(true|1|yes)|(^|[[:space:]])(RAW_PREDICTION_CACHE|PREDICTION_CACHE)[[:space:]]*="
if grep -Eiq "$RAW_CACHE_STDOUT_PATTERN" "$TRAIN_STDOUT"; then
  fail "train stdout contains raw prediction/cache marker"
fi

write_summary "PASS" "FRONTEND_RETRAIN_TRAIN_PASS_NO_DIRECT_TEST_NO_CLAIMS" "tools/train.py completed with frozen-reader ledgers and train-time validation only"
log "FRONTEND_RETRAIN_TRAIN_PASS_NO_DIRECT_TEST_NO_CLAIMS"
