#!/bin/bash
# BH-SDC N16R4 full-train candidate launcher.
# Route: DIVERGENT_INNOVATION_BH_SDC_DO_NOT_MERGE_WITH_C3
# Decision: ALLOW_BH_SDC_N16R4_SYNC_AND_FULL_TRAIN_CANDIDATE_V1
#
# This script is fail-closed for formal training. It refuses training unless a
# reviewed gate JSON is supplied and bound to the active manifest plus resolved
# config hash.
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH -J bhsdc_tr
#SBATCH -o logs/%x-%j.out

set -eo pipefail

log() {
  echo "[bh-sdc-n16r4] $(date -Is) $*"
}

fail() {
  log "ERROR $*"
  exit 64
}

PRECHECK_ONLY="${PRECHECK_ONLY:-1}"
ALLOW_BH_SDC_FULL_TRAIN="${ALLOW_BH_SDC_FULL_TRAIN:-0}"
ALLOW_LOGIN_NODE_DEBUG="${ALLOW_LOGIN_NODE_DEBUG:-0}"
GATE_JSON="${OPENTAD_BH_SDC_GATE_JSON:-${BH_SDC_GATE_JSON:-}}"
GATE_SHA256="${OPENTAD_BH_SDC_GATE_SHA256:-${BH_SDC_GATE_SHA256:-}}"
EXPECTED_THUMOS14_TRAIN_COUNT="${EXPECTED_THUMOS14_TRAIN_COUNT:-200}"
EXPECTED_THUMOS14_TEST_COUNT="${EXPECTED_THUMOS14_TEST_COUNT:-211}"
EXPECTED_SYNC_COMMAND="REMOTE_SYNC_TO_N16R4:~/run/yuzibo/OpenTAD_Back_check"

case "$PRECHECK_ONLY" in
  0|1) ;;
  *) fail "PRECHECK_ONLY must be exactly 0 or 1; got $PRECHECK_ONLY" ;;
esac

if [ "$PRECHECK_ONLY" = "0" ]; then
  [ "$ALLOW_BH_SDC_FULL_TRAIN" = "1" ] || fail "PRECHECK_ONLY=0 requires ALLOW_BH_SDC_FULL_TRAIN=1"
  [ -n "$GATE_JSON" ] || fail "refusing training: OPENTAD_BH_SDC_GATE_JSON is required"
  [ -n "$GATE_SHA256" ] || fail "refusing training: OPENTAD_BH_SDC_GATE_SHA256 is required"
else
  [ "$ALLOW_BH_SDC_FULL_TRAIN" = "0" ] || fail "PRECHECK_ONLY=1 must keep ALLOW_BH_SDC_FULL_TRAIN=0"
fi

source /etc/profile >/dev/null 2>&1 || true
set -u

YUZIBO_ROOT="${YUZIBO_ROOT:-$HOME/run/yuzibo}"
REPO_ROOT="${REPO_ROOT:-$YUZIBO_ROOT/OpenTAD_Back_check}"
CONFIG_REVIEWED="configs/adatad/thumos/bh_sdc_boundary_hazard_sparse_dense_full_train_candidate_n16r4.py"
CONFIG="${CONFIG:-$CONFIG_REVIEWED}"
TRAIN_ID="${TRAIN_ID:-0}"
SEED="${SEED:-20260624}"
SLURM_JOB_ID_FOR_PORT="${SLURM_JOB_ID:-0}"
MASTER_PORT="${MASTER_PORT:-$((33000 + (SLURM_JOB_ID_FOR_PORT % 20000)))}"
RUN_TAG="${RUN_TAG:-bh_sdc_full_train_${SLURM_JOB_ID:-nojob}_$(date '+%Y%m%d_%H%M%S')}"
TRAIN_COMMAND_REVIEWED="python tools/train.py $CONFIG_REVIEWED --id 0"

case "$RUN_TAG" in
  *..*|*/*|*\\*|*[[:space:]]*|*[\;\&\|\>\<\`\"\'$]*)
    fail "RUN_TAG must be a simple path-safe name; got $RUN_TAG"
    ;;
esac
case "$TRAIN_ID" in
  0) ;;
  *) fail "BH-SDC launch gate is reviewed only for TRAIN_ID=0; got $TRAIN_ID" ;;
esac
case "$SEED" in
  ''|*[!0-9]*) fail "SEED must be numeric; got $SEED" ;;
esac
case "$MASTER_PORT" in
  ''|*[!0-9]*) fail "MASTER_PORT must be numeric; got $MASTER_PORT" ;;
esac
case "$EXPECTED_THUMOS14_TRAIN_COUNT" in
  ''|*[!0-9]*) fail "EXPECTED_THUMOS14_TRAIN_COUNT must be numeric; got $EXPECTED_THUMOS14_TRAIN_COUNT" ;;
esac
case "$EXPECTED_THUMOS14_TEST_COUNT" in
  ''|*[!0-9]*) fail "EXPECTED_THUMOS14_TEST_COUNT must be numeric; got $EXPECTED_THUMOS14_TEST_COUNT" ;;
esac
if [ "$MASTER_PORT" -lt 1024 ] || [ "$MASTER_PORT" -gt 65535 ]; then
  fail "MASTER_PORT must be between 1024 and 65535; got $MASTER_PORT"
fi
if [ "$EXPECTED_THUMOS14_TRAIN_COUNT" -lt 1 ] || [ "$EXPECTED_THUMOS14_TEST_COUNT" -lt 1 ]; then
  fail "expected THUMOS14 train/test counts must be positive"
fi
if [ "$CONFIG" != "$CONFIG_REVIEWED" ]; then
  fail "launcher is reviewed only for $CONFIG_REVIEWED; got $CONFIG"
fi
case "$REPO_ROOT" in
  "$YUZIBO_ROOT"/OpenTAD_Back_check|.) ;;
  *) fail "REPO_ROOT must be the BH-SDC N16R4 workspace $YUZIBO_ROOT/OpenTAD_Back_check; got $REPO_ROOT" ;;
esac
if [ -z "${SLURM_JOB_ID:-}" ] && [ "$ALLOW_LOGIN_NODE_DEBUG" != "1" ]; then
  fail "refusing direct non-Slurm execution unless ALLOW_LOGIN_NODE_DEBUG=1 for lightweight local checks"
fi
if [ -z "${SLURM_JOB_ID:-}" ] && [ "$ALLOW_LOGIN_NODE_DEBUG" = "1" ] && [ "$PRECHECK_ONLY" != "1" ]; then
  fail "login-node debug must keep PRECHECK_ONLY=1; refusing training outside Slurm"
fi
if [ -n "${LOAD_FROM:-}" ] || [ -n "${RESUME:-}" ] || [ -n "${CHECKPOINT:-}" ]; then
  fail "checkpoint/load/resume environment shortcuts are forbidden for BH-SDC"
fi
if [ -n "${RAW_PREDICTION_CACHE:-}" ] || [ -n "${PREDICTION_CACHE:-}" ] || [ -n "${LOAD_FROM_RAW_PREDICTIONS:-}" ]; then
  fail "raw prediction/cache environment shortcuts are forbidden for BH-SDC"
fi
if [ -n "${CFG_OPTIONS:-}" ] || [ -n "${EXTRA_CFG_OPTIONS:-}" ]; then
  fail "arbitrary cfg-options are forbidden for BH-SDC"
fi

if [ "${BH_SDC_SKIP_MODULES:-0}" != "1" ]; then
  module load cuda/11.8
  module load miniforge3/24.11
  source activate "$YUZIBO_ROOT/conda_envs/opentad"
fi

cd "$REPO_ROOT"
test -f "$CONFIG_REVIEWED" || fail "missing reviewed config: $CONFIG_REVIEWED"
test -f "tools/bata/validate_bh_sdc_full_train_gate.py" || fail "missing BH-SDC validator"
if [ "$PRECHECK_ONLY" = "0" ]; then
  test -f "$GATE_JSON" || fail "OPENTAD_BH_SDC_GATE_JSON does not exist: $GATE_JSON"
fi

LOG_DIR="$REPO_ROOT/logs"
RUN_ROOT="$LOG_DIR/$RUN_TAG"
RUN_LOG="$RUN_ROOT/${RUN_TAG}.log"
TRAIN_STDOUT="$RUN_ROOT/${RUN_TAG}.train.stdout.log"
SHA_MANIFEST="$RUN_ROOT/active_sha256_manifest.txt"
RESOLVED_CONFIG_DUMP="$RUN_ROOT/resolved_config.py"
SUMMARY_JSON="$RUN_ROOT/bh_sdc_full_train_summary.json"
WORK_DIR="$RUN_ROOT/train_workdir/gpu1_id0"

mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_LOG") 2>&1

write_summary() {
  local status="$1"
  local decision="$2"
  local reason="$3"
  python - "$SUMMARY_JSON" "$status" "$decision" "$reason" "$RUN_ROOT" "$WORK_DIR" <<'PY'
import json
import os
import socket
import sys
from pathlib import Path

summary_path, status, decision, reason, run_root, work_dir = sys.argv[1:7]
payload = {
    "status": status,
    "decision": decision,
    "reason": reason,
    "route_label": "DIVERGENT_INNOVATION_BH_SDC_DO_NOT_MERGE_WITH_C3",
    "host": socket.gethostname(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "run_root": run_root,
    "work_dir": work_dir,
    "claims_allowed": False,
    "direct_eval_entrypoint_allowed": False,
    "raw_prediction_cache_allowed": False,
}
Path(summary_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  log "summary_json=$SUMMARY_JSON"
}

on_exit() {
  local status="$?"
  if [ "$status" -ne 0 ] && [ -n "${SUMMARY_JSON:-}" ] && [ ! -f "$SUMMARY_JSON" ]; then
    write_summary "FAIL" "BH_SDC_LAUNCHER_FAILED" "launcher exited before clean completion"
  fi
}
trap on_exit EXIT

log "route_label=DIVERGENT_INNOVATION_BH_SDC_DO_NOT_MERGE_WITH_C3"
log "decision=ALLOW_BH_SDC_N16R4_SYNC_AND_FULL_TRAIN_CANDIDATE_V1"
log "repo_root=$REPO_ROOT"
log "config=$CONFIG_REVIEWED"
log "run_root=$RUN_ROOT"
log "precheck_only=$PRECHECK_ONLY allow_full_train=$ALLOW_BH_SDC_FULL_TRAIN"
log "expected_sync_command=$EXPECTED_SYNC_COMMAND"
log "train_command_reviewed=$TRAIN_COMMAND_REVIEWED"

python - "$CONFIG_REVIEWED" "$RESOLVED_CONFIG_DUMP" <<'PY'
import sys
from pathlib import Path

from mmengine.config import Config

cfg = Config.fromfile(sys.argv[1])
Path(sys.argv[2]).write_text(cfg.pretty_text + "\n", encoding="utf-8")
gate = cfg.bh_sdc_gate
if gate.launch_gate_passed is not True:
    raise SystemExit("BH-SDC full candidate must be entrypoint-gated with launch_gate_passed=True")
if tuple(gate.allowed_entrypoints) != ("tools/train.py",):
    raise SystemExit("BH-SDC full candidate must allow only the train entrypoint")
if gate.allow_tools_test is not False or gate.allow_detector_map is not False:
    raise SystemExit("BH-SDC full candidate must reject direct evaluation and detector-claim entrypoints")
if cfg.inference.load_from_raw_predictions is not False or cfg.inference.save_raw_prediction is not False:
    raise SystemExit("BH-SDC full candidate must keep raw prediction cache disabled")
print("bh_sdc_static_config_preflight=PASS")
PY

RESOLVED_CONFIG_SHA256="$(sha256sum "$RESOLVED_CONFIG_DUMP" | awk '{print $1}')"
log "resolved_config_sha256=$RESOLVED_CONFIG_SHA256"

mapfile -t CONFIG_DEP_FILES < <(python - "$CONFIG_REVIEWED" <<'PY'
import ast
import sys
from pathlib import Path

seen = []
root = Path(sys.argv[1]).resolve()

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

visit(root)
for path in seen:
    print(path.relative_to(Path.cwd()).as_posix())
PY
)

sha256sum \
  "${CONFIG_DEP_FILES[@]}" \
  "$RESOLVED_CONFIG_DUMP" \
  scripts/run_bh_sdc_full_train_n16r4.sbatch \
  tools/bata/validate_bh_sdc_full_train_gate.py \
  tools/train.py \
  opentad/utils/training_guard.py \
  opentad/models/detectors/actionformer.py \
  opentad/models/selectors/bh_sdc_frame_selector.py \
  tests/test_bh_sdc_config_gate.py \
  tests/test_bh_sdc_launcher_gate.py \
  docs/en/bh_sdc_route_original_purpose_and_drift_check_20260624.md \
  > "$SHA_MANIFEST"
ACTIVE_MANIFEST_SHA256="$(sha256sum "$SHA_MANIFEST" | awk '{print $1}')"
log "active_sha256_manifest_sha256=$ACTIVE_MANIFEST_SHA256"

python -m py_compile tools/bata/validate_bh_sdc_full_train_gate.py

if [ "$PRECHECK_ONLY" = "1" ]; then
  write_summary "PASS" "BH_SDC_STATIC_PRECHECK_ONLY_PASS_NO_TRAIN" "PRECHECK_ONLY=1; no gate payload consumed"
  log "PRECHECK_ONLY=1; exiting before training. Remote sync, Slurm full train, and metric claims remain unstarted."
  exit 0
fi

python tools/bata/validate_bh_sdc_full_train_gate.py "$CONFIG_REVIEWED" \
  --gate-json "$GATE_JSON" \
  --gate-sha256 "$GATE_SHA256" \
  --active-manifest-sha256 "$ACTIVE_MANIFEST_SHA256" \
  --resolved-config-sha256 "$RESOLVED_CONFIG_SHA256" \
  --requested-action slurm_full_train_candidate \
  --requested-command "$TRAIN_COMMAND_REVIEWED" \
  --json

export OPENTAD_BH_SDC_GATE_JSON="$GATE_JSON"
export OPENTAD_BH_SDC_GATE_SHA256="$GATE_SHA256"
export OPENTAD_BH_SDC_ACTIVE_MANIFEST_SHA256="$ACTIVE_MANIFEST_SHA256"
export OPENTAD_BH_SDC_RESOLVED_CONFIG_SHA256="$RESOLVED_CONFIG_SHA256"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export YUZIBO_ROOT
export THUMOS14_ANNOTATION_PATH="${THUMOS14_ANNOTATION_PATH:-$YUZIBO_ROOT/thumos14/annotations/thumos_14_anno.json}"
export THUMOS14_CLASS_MAP="${THUMOS14_CLASS_MAP:-$YUZIBO_ROOT/thumos14/annotations/category_idx.txt}"
export THUMOS14_TRAIN_DATA_PATH="${THUMOS14_TRAIN_DATA_PATH:-$YUZIBO_ROOT/thumos14/train}"
export THUMOS14_TEST_DATA_PATH="${THUMOS14_TEST_DATA_PATH:-$YUZIBO_ROOT/thumos14/test}"

test -f "$THUMOS14_ANNOTATION_PATH" || fail "missing annotation: $THUMOS14_ANNOTATION_PATH"
test -f "$THUMOS14_CLASS_MAP" || fail "missing class map: $THUMOS14_CLASS_MAP"
test -d "$THUMOS14_TRAIN_DATA_PATH" || fail "missing train data dir: $THUMOS14_TRAIN_DATA_PATH"
test -d "$THUMOS14_TEST_DATA_PATH" || fail "missing validation/test data dir: $THUMOS14_TEST_DATA_PATH"

count_mp4_files() {
  find "$1" -maxdepth 1 \( -type f -o -type l \) -iname '*.mp4' | wc -l | awk '{print $1}'
}

train_count="$(count_mp4_files "$THUMOS14_TRAIN_DATA_PATH")"
test_count="$(count_mp4_files "$THUMOS14_TEST_DATA_PATH")"
log "thumos14_train_mp4_count=$train_count expected=$EXPECTED_THUMOS14_TRAIN_COUNT"
log "thumos14_test_mp4_count=$test_count expected=$EXPECTED_THUMOS14_TEST_COUNT"
if [ "$train_count" -ne "$EXPECTED_THUMOS14_TRAIN_COUNT" ]; then
  fail "THUMOS14 train data count mismatch: expected $EXPECTED_THUMOS14_TRAIN_COUNT mp4 files, found $train_count in $THUMOS14_TRAIN_DATA_PATH"
fi
if [ "$test_count" -ne "$EXPECTED_THUMOS14_TEST_COUNT" ]; then
  fail "THUMOS14 validation/test data count mismatch: expected $EXPECTED_THUMOS14_TEST_COUNT mp4 files, found $test_count in $THUMOS14_TEST_DATA_PATH"
fi

if command -v ss >/dev/null 2>&1; then
  if ss -ltn | awk '{print $4}' | grep -Eq "[:.]${MASTER_PORT}$"; then
    fail "MASTER_PORT=$MASTER_PORT is already in use"
  fi
fi

mkdir -p "$WORK_DIR"
log "launching BH-SDC full-train candidate"
torchrun --nproc_per_node=1 --master_port="$MASTER_PORT" \
  tools/train.py "$CONFIG_REVIEWED" --id 0 --seed "$SEED" \
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
if grep -Eiq "load_from_raw_predictions[[:space:]=:]+True|save_raw_prediction[[:space:]=:]+True|RAW_PREDICTION_CACHE|PREDICTION_CACHE" "$TRAIN_STDOUT"; then
  fail "train stdout contains raw prediction/cache marker"
fi

write_summary "PASS" "BH_SDC_FULL_TRAIN_PASS_NO_CLAIMS" "training completed; no direct evaluation entrypoint or paper/runtime/deploy claim authorized"
log "BH_SDC_FULL_TRAIN_PASS_NO_CLAIMS"
