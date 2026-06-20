#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -J pcot_r16smk
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
    echo "[pc-ot-mras-r16a-gpu-smoke] $(date -Is) ERROR refusing dirty historical OpenTAD_BATA_Clean path for clean PC-OT-MRAS run: $REPO" >&2
    exit 2
    ;;
esac
if [ ! -d "$REPO" ]; then
  echo "[pc-ot-mras-r16a-gpu-smoke] $(date -Is) ERROR missing clean PC-OT-MRAS repo: $REPO" >&2
  exit 2
fi
cd "$REPO"

CONFIG_REVIEWED="configs/adatad/thumos/ctf_bdi_pc_ot_mras_r16_gpu_smoke_candidate.py"
CONFIG="${CONFIG:-$CONFIG_REVIEWED}"
PRECHECK_ONLY="${PRECHECK_ONLY:-1}"
ALLOW_R16A_GPU_SMOKE="${ALLOW_R16A_GPU_SMOKE:-0}"
R16A_GPU_SMOKE_ONLY="${R16A_GPU_SMOKE_ONLY:-1}"
ALLOW_R16A_LONG_TRAINING="${ALLOW_R16A_LONG_TRAINING:-0}"
ALLOW_R16A_TOOLS_TEST="${ALLOW_R16A_TOOLS_TEST:-0}"
ALLOW_R16A_DETECTOR_MAP="${ALLOW_R16A_DETECTOR_MAP:-0}"
R16A_GPU_SMOKE_GATE_JSON="${R16A_GPU_SMOKE_GATE_JSON:-}"
R16A_GPU_SMOKE_GATE_SHA256="${R16A_GPU_SMOKE_GATE_SHA256:-}"
TRAIN_ID="${TRAIN_ID:-0}"
SEED="${SEED:-20260619}"
SLURM_JOB_ID_FOR_PORT="${SLURM_JOB_ID:-0}"
DEFAULT_MASTER_PORT=$((31000 + (SLURM_JOB_ID_FOR_PORT % 20000)))
REQUESTED_MASTER_PORT="${MASTER_PORT:-}"
MASTER_PORT_SOURCE="default_from_slurm_job_id"
if [ -n "$REQUESTED_MASTER_PORT" ]; then
  MASTER_PORT="$REQUESTED_MASTER_PORT"
  MASTER_PORT_SOURCE="environment"
else
  MASTER_PORT="$DEFAULT_MASTER_PORT"
fi
RUN_TAG="${RUN_TAG:-ctf_bdi_pc_ot_mras_r16a_gpu_smoke_${SLURM_JOB_ID:-nojob}_$(date '+%Y%m%d_%H%M%S')}"

case "$RUN_TAG" in
  *..*|*/*|*\\*|*[[:space:]]*|*[\;\&\|\>\<\`\"\'$]*)
    echo "[pc-ot-mras-r16a-gpu-smoke] $(date -Is) ERROR RUN_TAG must be a simple path-safe name; got $RUN_TAG" >&2
    exit 2
    ;;
esac

LOG_DIR="$REPO/logs"
RUN_ROOT="$LOG_DIR/$RUN_TAG"
RUN_LOG="$RUN_ROOT/${RUN_TAG}.log"
TRAIN_STDOUT="$RUN_ROOT/${RUN_TAG}.train.stdout.log"
SHA_MANIFEST="$RUN_ROOT/active_sha256_manifest.txt"
RESOLVED_CONFIG_DUMP="$RUN_ROOT/resolved_config.py"
SUMMARY_JSON="$RUN_ROOT/r16a_gpu_smoke_summary.json"
PRETRAINED_PATH_REVIEWED="pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"
PRETRAINED_PATH="${PRETRAINED_PATH:-$PRETRAINED_PATH_REVIEWED}"
WORK_DIR_BASE="$RUN_ROOT/train_workdir"
WORK_DIR="$WORK_DIR_BASE/gpu1_id${TRAIN_ID}"
CHECKPOINT_PATH="$WORK_DIR/checkpoint/epoch_0.pth"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export YUZIBO_ROOT
export OPENTAD_PCOTMRAS_CLEAN_ROOT="$REPO"
export THUMOS14_ANNOTATION_PATH="${THUMOS14_ANNOTATION_PATH:-$YUZIBO_ROOT/thumos14/annotations/thumos_14_anno.json}"
export THUMOS14_CLASS_MAP="${THUMOS14_CLASS_MAP:-$YUZIBO_ROOT/thumos14/annotations/category_idx.txt}"
export THUMOS14_TRAIN_DATA_PATH="${THUMOS14_TRAIN_DATA_PATH:-$YUZIBO_ROOT/thumos14/train}"
export THUMOS14_TEST_DATA_PATH="${THUMOS14_TEST_DATA_PATH:-$YUZIBO_ROOT/thumos14/test}"

mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_LOG") 2>&1

log() {
  echo "[pc-ot-mras-r16a-gpu-smoke] $(date -Is) $*"
}

fail() {
  log "ERROR $*"
  exit 2
}

write_summary() {
  local status="$1"
  local decision="$2"
  local reason="$3"
  python - "$SUMMARY_JSON" "$status" "$decision" "$reason" "$RUN_ROOT" "$CONFIG" "$WORK_DIR" "$CHECKPOINT_PATH" <<'PY'
import json
import os
import socket
import sys
from pathlib import Path

summary_path, status, decision, reason, run_root, config, work_dir, checkpoint = sys.argv[1:9]
payload = {
    "status": status,
    "decision": decision,
    "reason": reason,
    "stage": "R16A_GPU_SMOKE_ONLY",
    "route": "CTF-BDI/PC-OT-MRAS",
    "host": socket.gethostname(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "run_root": run_root,
    "config": config,
    "work_dir": work_dir,
    "checkpoint_path": checkpoint,
    "checkpoint_for_recovery_only": True,
    "claim_boundaries": {
        "gpu_smoke_only": True,
        "max_train_iters": 2,
        "single_gpu": True,
        "no_tools_test_execution": True,
        "no_detector_map": True,
        "no_long_training": True,
        "no_raw_prediction_cache": True,
        "no_runtime_flops_deploy_metric_paper_claims": True,
    },
    "claim_flags": {
        "detector_map_run": False,
        "metric_claim_allowed": False,
        "paper_claim_allowed": False,
        "runtime_or_flops_claim_allowed": False,
        "deploy_claim_allowed": False,
        "dynamic_budget_claim_allowed": False,
        "scanner_quality_claim_allowed": False,
    },
}
Path(summary_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  log "summary_json=$SUMMARY_JSON"
}

on_exit() {
  local status="$?"
  if [ "$status" -ne 0 ] && [ ! -f "$SUMMARY_JSON" ]; then
    write_summary "FAIL" "R16A_GPU_SMOKE_FAILED" "launcher exited before successful completion"
  fi
}
trap on_exit EXIT

case "$PRECHECK_ONLY" in
  0|1) ;;
  *) fail "PRECHECK_ONLY must be exactly 0 or 1; got $PRECHECK_ONLY" ;;
esac
case "$R16A_GPU_SMOKE_ONLY" in
  1) ;;
  *) fail "R16A_GPU_SMOKE_ONLY must remain 1" ;;
esac
case "$ALLOW_R16A_LONG_TRAINING" in
  0) ;;
  *) fail "long training is not approved; ALLOW_R16A_LONG_TRAINING must remain 0" ;;
esac
case "$ALLOW_R16A_TOOLS_TEST" in
  0) ;;
  *) fail "tools/test.py is not approved for R16A; ALLOW_R16A_TOOLS_TEST must remain 0" ;;
esac
case "$ALLOW_R16A_DETECTOR_MAP" in
  0) ;;
  *) fail "detector mAP is not approved for R16A; ALLOW_R16A_DETECTOR_MAP must remain 0" ;;
esac
case "$MASTER_PORT" in
  ''|*[!0-9]*)
    if [ "$MASTER_PORT_SOURCE" = "environment" ]; then
      log "WARN ignoring invalid inherited MASTER_PORT=$MASTER_PORT; using default $DEFAULT_MASTER_PORT"
      MASTER_PORT="$DEFAULT_MASTER_PORT"
      MASTER_PORT_SOURCE="default_after_invalid_environment"
    else
      fail "MASTER_PORT must be numeric; got $MASTER_PORT"
    fi
    ;;
esac
case "$MASTER_PORT" in
  ''|*[!0-9]*) fail "normalized MASTER_PORT must be numeric; got $MASTER_PORT" ;;
esac
if [ "$MASTER_PORT" -lt 1024 ] || [ "$MASTER_PORT" -gt 65535 ]; then
  if [ "$MASTER_PORT_SOURCE" = "environment" ]; then
    log "WARN ignoring inherited MASTER_PORT=$MASTER_PORT outside [1024,65535]; using default $DEFAULT_MASTER_PORT"
    MASTER_PORT="$DEFAULT_MASTER_PORT"
    MASTER_PORT_SOURCE="default_after_invalid_environment"
  else
    fail "MASTER_PORT must be between 1024 and 65535; got $MASTER_PORT"
  fi
fi
if [ "$MASTER_PORT" -lt 1024 ] || [ "$MASTER_PORT" -gt 65535 ]; then
  fail "normalized MASTER_PORT must be between 1024 and 65535; got $MASTER_PORT"
fi
case "$TRAIN_ID" in
  ''|*[!0-9]*) fail "TRAIN_ID must be numeric; got $TRAIN_ID" ;;
esac
case "$SEED" in
  ''|*[!0-9]*) fail "SEED must be numeric; got $SEED" ;;
esac
if [ "$CONFIG" != "$CONFIG_REVIEWED" ]; then
  fail "launcher is reviewed only for $CONFIG_REVIEWED; got $CONFIG"
fi
if [ -z "${SLURM_JOB_ID:-}" ] && [ "${ALLOW_LOGIN_NODE_DEBUG:-0}" != "1" ]; then
  fail "refusing direct non-Slurm execution unless ALLOW_LOGIN_NODE_DEBUG=1 for lightweight PRECHECK_ONLY"
fi
if [ -z "${SLURM_JOB_ID:-}" ] && [ "${ALLOW_LOGIN_NODE_DEBUG:-0}" = "1" ] && [ "$PRECHECK_ONLY" != "1" ]; then
  fail "login-node debug must keep PRECHECK_ONLY=1; refusing training outside Slurm"
fi
if [ "$PRECHECK_ONLY" = "0" ]; then
  case "$ALLOW_R16A_GPU_SMOKE" in
    1) ;;
    *) fail "PRECHECK_ONLY=0 requires ALLOW_R16A_GPU_SMOKE=1 after a separate Pro/Gemini execution-package gate" ;;
  esac
  if [ -z "$R16A_GPU_SMOKE_GATE_JSON" ] || [ -z "$R16A_GPU_SMOKE_GATE_SHA256" ]; then
    fail "PRECHECK_ONLY=0 requires explicit R16A_GPU_SMOKE_GATE_JSON and R16A_GPU_SMOKE_GATE_SHA256"
  fi
else
  case "$ALLOW_R16A_GPU_SMOKE" in
    0) ;;
    *) fail "PRECHECK_ONLY=1 must not set ALLOW_R16A_GPU_SMOKE=1" ;;
  esac
fi
if [ -n "${CHECKPOINT:-}" ] || [ -n "${CHECKPOINT_SHA256:-}" ] || [ -n "${LOAD_FROM:-}" ] || [ -n "${RESUME:-}" ]; then
  fail "external checkpoint/load/resume environment shortcuts are forbidden for R16A smoke"
fi
if [ "$PRETRAINED_PATH" != "$PRETRAINED_PATH_REVIEWED" ]; then
  fail "PRETRAINED_PATH override is forbidden for R16A smoke; reviewed path is $PRETRAINED_PATH_REVIEWED"
fi
if [ -n "${LOAD_FROM_RAW_PREDICTIONS:-}" ] || [ -n "${RAW_PREDICTION_CACHE:-}" ] || [ -n "${PREDICTION_CACHE:-}" ]; then
  fail "raw prediction/cache environment shortcuts are forbidden for R16A smoke"
fi
if [ -n "${CFG_OPTIONS:-}" ] || [ -n "${EXTRA_CFG_OPTIONS:-}" ]; then
  fail "arbitrary cfg-options are forbidden for R16A smoke"
fi

log "host=$(hostname) job=${SLURM_JOB_ID:-none}"
log "repo=$REPO"
log "config=$CONFIG"
log "run_root=$RUN_ROOT"
log "work_dir=$WORK_DIR"
log "checkpoint_path=$CHECKPOINT_PATH"
log "seed=$SEED train_id=$TRAIN_ID master_port=$MASTER_PORT master_port_source=$MASTER_PORT_SOURCE default_master_port=$DEFAULT_MASTER_PORT"
log "precheck_only=$PRECHECK_ONLY allow_r16a_gpu_smoke=$ALLOW_R16A_GPU_SMOKE"
log "scope=GPU_SMOKE_ONLY no_tools_test=true no_map=true no_long_training=true no_claims=true"
log "python=$(python -c 'import sys; print(sys.executable)')"
log "git_head=$(git rev-parse HEAD 2>/dev/null || echo unknown)"

test -f "$CONFIG" || fail "missing reviewed config: $CONFIG"
test -f "$PRETRAINED_PATH" || fail "missing reviewed pretrained initialization file: $PRETRAINED_PATH"
test -f "$THUMOS14_ANNOTATION_PATH" || fail "missing annotation: $THUMOS14_ANNOTATION_PATH"
test -f "$THUMOS14_CLASS_MAP" || fail "missing class map: $THUMOS14_CLASS_MAP"
test -d "$THUMOS14_TRAIN_DATA_PATH" || fail "missing train data dir: $THUMOS14_TRAIN_DATA_PATH"
test -d "$THUMOS14_TEST_DATA_PATH" || fail "missing test data dir: $THUMOS14_TEST_DATA_PATH"

log "writing active SHA256 manifest=$SHA_MANIFEST"
sha256sum \
  "$CONFIG" \
  "$PRETRAINED_PATH" \
  scripts/run_ctf_bdi_pc_ot_mras_r16a_gpu_smoke_n16r4.sbatch \
  tools/train.py \
  tools/test.py \
  opentad/cores/train_engine.py \
  opentad/utils/training_guard.py \
  opentad/models/detectors/actionformer.py \
  opentad/models/dense_heads/__init__.py \
  opentad/models/dense_heads/native_irregular_area_head_p2.py \
  opentad/models/necks/pc_ot_mras_detector_bridge.py \
  opentad/models/selectors/pc_ot_mras_reader.py \
  opentad/models/utils/pc_ot_mras_raw_prediction_guard.py \
  tests/test_pc_ot_mras_r16_gpu_smoke_guard.py \
  tests/test_train_engine_max_train_iters.py \
  > "$SHA_MANIFEST"
ACTIVE_MANIFEST_SHA256="$(sha256sum "$SHA_MANIFEST" | awk '{print $1}')"
log "active_sha256_manifest_sha256=$ACTIVE_MANIFEST_SHA256"

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
    raise SystemExit("R16A GPU smoke requires at least one visible CUDA device")
PY
nvidia-smi || fail "nvidia-smi failed"

if [ "$PRECHECK_ONLY" = "0" ]; then
  test -f "$R16A_GPU_SMOKE_GATE_JSON" || fail "missing R16A execution gate JSON: $R16A_GPU_SMOKE_GATE_JSON"
  RESOLVED_GATE_JSON="$(realpath -e "$R16A_GPU_SMOKE_GATE_JSON")" || fail "R16A execution gate realpath failed: $R16A_GPU_SMOKE_GATE_JSON"
  case "$RESOLVED_GATE_JSON" in
    "$REPO"/*) ;;
    *) fail "R16A execution gate JSON must be under reviewed repo $REPO; got $RESOLVED_GATE_JSON" ;;
  esac
  ACTUAL_GATE_SHA256="$(sha256sum "$R16A_GPU_SMOKE_GATE_JSON" | awk '{print $1}')"
  if [ "$ACTUAL_GATE_SHA256" != "$R16A_GPU_SMOKE_GATE_SHA256" ]; then
    fail "R16A execution gate sha256 mismatch: expected=$R16A_GPU_SMOKE_GATE_SHA256 actual=$ACTUAL_GATE_SHA256"
  fi
  python - "$R16A_GPU_SMOKE_GATE_JSON" "$ACTIVE_MANIFEST_SHA256" <<'PY'
import json
import sys
from pathlib import Path

gate = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
active_manifest_sha256 = sys.argv[2]
allowed = {
    "PASS_ALLOW_R16A_BOUNDED_GPU_SMOKE_ONLY",
    "PASS_ALLOW_R16A_GPU_SMOKE_EXECUTION_ONLY",
}
if gate.get("decision") not in allowed:
    raise SystemExit(f"R16A execution gate decision does not allow GPU smoke: {gate.get('decision')}")
expected_manifest_sha256 = gate.get("active_sha256_manifest_sha256") or gate.get(
    "expected_active_sha256_manifest_sha256"
)
if not expected_manifest_sha256:
    raise SystemExit("R16A execution gate must bind active_sha256_manifest_sha256")
if expected_manifest_sha256 != active_manifest_sha256:
    raise SystemExit(
        "R16A execution gate active manifest sha256 mismatch: "
        f"expected={expected_manifest_sha256} actual={active_manifest_sha256}"
    )
if gate.get("route") not in (None, "CTF-BDI/PC-OT-MRAS"):
    raise SystemExit(f"R16A execution gate route mismatch: {gate.get('route')}")
if gate.get("max_train_iters") not in (None, 2):
    raise SystemExit(f"R16A execution gate must preserve max_train_iters=2: {gate.get('max_train_iters')}")
if gate.get("single_gpu") not in (None, True):
    raise SystemExit(f"R16A execution gate must preserve single_gpu=true: {gate.get('single_gpu')}")
for key in ("allow_tools_test", "allow_detector_map", "allow_long_training", "allow_metric_claim", "allow_paper_claim"):
    if gate.get(key) is True:
        raise SystemExit(f"R16A execution gate must not set {key}=true")
print("ctf_bdi_pc_ot_mras_r16a_execution_gate=PASS_BOUND_TO_ACTIVE_MANIFEST")
PY
fi

if command -v ss >/dev/null 2>&1; then
  if ss -ltn | awk '{print $4}' | grep -Eq "[:.]${MASTER_PORT}$"; then
    fail "MASTER_PORT=$MASTER_PORT is already in use"
  fi
else
  log "WARN ss not available; skipping master port preflight"
fi

log "dumping and auditing resolved config=$RESOLVED_CONFIG_DUMP"
python - "$CONFIG" "$RESOLVED_CONFIG_DUMP" <<'PY'
import sys
from pathlib import Path

from mmengine.config import Config


def require(condition, message):
    if not condition:
        raise SystemExit(str(message))


cfg = Config.fromfile(sys.argv[1])
Path(sys.argv[2]).write_text(cfg.pretty_text + "\n", encoding="utf-8")

gate = cfg.r16_pc_ot_mras_gpu_smoke_gate
require(cfg.get("r14_pc_ot_mras_train_candidate_gate", "missing") is None, "R14 gate must be cleared only in R16 smoke config")
require(gate.stage == "R16_gpu_slurm_smoke_candidate", gate)
require(gate.smoke_only is True, gate)
require(gate.launch_gate_passed is True, gate)
require(gate.allow_detector_training is True, gate)
require(gate.allow_tools_train is True, gate)
require(gate.allow_tools_test is False, gate)
require(gate.allow_detector_map is False, gate)
require(gate.allow_long_training is False, gate)
require(int(gate.max_epochs) == 1, gate)
require(int(gate.max_train_iters) == 2, gate)
require(tuple(gate.allowed_entrypoints) == ("tools/train.py",), gate)

require(int(cfg.workflow.end_epoch) == 1, cfg.workflow)
require(int(cfg.workflow.max_train_iters) == 2, cfg.workflow)
require(int(cfg.workflow.val_start_epoch) == 999, cfg.workflow)
require(int(cfg.workflow.val_eval_interval) == -1, cfg.workflow)
require(int(cfg.workflow.val_loss_interval) == -1, cfg.workflow)
require(int(cfg.workflow.checkpoint_interval) == 1, cfg.workflow)
require(int(cfg.solver.train.batch_size) == 1, cfg.solver.train)
require(int(cfg.solver.val.batch_size) == 1, cfg.solver.val)
require(int(cfg.solver.test.batch_size) == 1, cfg.solver.test)
require(cfg.inference.load_from_raw_predictions is False, cfg.inference)
require(cfg.inference.save_raw_prediction is False, cfg.inference)
require(cfg.model.pc_ot_mras_reader.type == "PCOTMRASReader", cfg.model.pc_ot_mras_reader)
require(cfg.model.neck.type == "PCOTMRASDetectorBridge", cfg.model.neck)
require(int(cfg.model.pc_ot_mras_reader_feature_level) == 0, cfg.model)
require(int(cfg.model.neck.source_feature_level) == 0, cfg.model.neck)
require(cfg.model.rpn_head.type == "NativeIrregularAreaHeadP2", cfg.model.rpn_head)

for split in ("train", "val", "test"):
    pipeline_text = repr(cfg.dataset[split].pipeline).lower()
    for forbidden in ("raw_prediction", "prediction_cache", "teacher", "oracle"):
        require(forbidden not in pipeline_text, f"{split} pipeline contains forbidden token {forbidden}")

print("ctf_bdi_pc_ot_mras_r16a_gpu_smoke_config_preflight=PASS")
PY

log "static py_compile"
python -m py_compile \
  "$CONFIG" \
  tools/train.py \
  tools/test.py \
  opentad/cores/train_engine.py \
  opentad/utils/training_guard.py \
  opentad/models/detectors/actionformer.py \
  opentad/models/dense_heads/__init__.py \
  opentad/models/dense_heads/native_irregular_area_head_p2.py \
  opentad/models/necks/pc_ot_mras_detector_bridge.py \
  opentad/models/selectors/pc_ot_mras_reader.py \
  opentad/models/utils/pc_ot_mras_raw_prediction_guard.py \
  tests/test_pc_ot_mras_r16_gpu_smoke_guard.py \
  tests/test_train_engine_max_train_iters.py

log "focused R16A smoke guard pytest"
python -m pytest \
  tests/test_train_engine_max_train_iters.py \
  tests/test_pc_ot_mras_train_local_only_guard.py \
  tests/test_pc_ot_mras_r14_trainable_candidate_config.py \
  tests/test_pc_ot_mras_r16_gpu_smoke_guard.py \
  -q -rs

if [ "$PRECHECK_ONLY" = "1" ]; then
  write_summary "PASS" "R16A_GPU_SMOKE_PRECHECK_ONLY_PASS_NO_TRAIN" "precheck-only launcher checks completed without tools/train execution"
  log "PRECHECK_ONLY=1; exiting before R16A tools/train GPU smoke"
  exit 0
fi

log "launching bounded R16A train-entry GPU smoke"
TRAIN_ARGS=(
  tools/train.py "$CONFIG"
  --id "$TRAIN_ID"
  --seed "$SEED"
  --cfg-options
  work_dir="$WORK_DIR_BASE"
  dataset.train.ann_file="$THUMOS14_ANNOTATION_PATH"
  dataset.train.class_map="$THUMOS14_CLASS_MAP"
  dataset.train.data_path="$THUMOS14_TRAIN_DATA_PATH"
  dataset.val.ann_file="$THUMOS14_ANNOTATION_PATH"
  dataset.val.class_map="$THUMOS14_CLASS_MAP"
  dataset.val.data_path="$THUMOS14_TEST_DATA_PATH"
  dataset.test.ann_file="$THUMOS14_ANNOTATION_PATH"
  dataset.test.class_map="$THUMOS14_CLASS_MAP"
  dataset.test.data_path="$THUMOS14_TEST_DATA_PATH"
  evaluation.ground_truth_filename="$THUMOS14_ANNOTATION_PATH"
)
torchrun --nproc_per_node=1 --master_port="$MASTER_PORT" "${TRAIN_ARGS[@]}" 2>&1 | tee "$TRAIN_STDOUT"

log "validating smoke outputs"
test -f "$CHECKPOINT_PATH" || fail "missing bounded recovery checkpoint: $CHECKPOINT_PATH"
if find "$WORK_DIR" -maxdepth 3 -type f \( -name 'result_detection.json' -o -name '*mAP*' \) | grep -q .; then
  fail "R16A smoke produced forbidden detector metric artifacts under $WORK_DIR"
fi
grep -q "max_train_iters=2 reached" "$TRAIN_STDOUT" || fail "train stdout does not show max_train_iters early stop"
grep -q "Training Over" "$TRAIN_STDOUT" || fail "train stdout does not show clean Training Over"
if grep -Eq "Testing Starts|mAP|result_detection" "$TRAIN_STDOUT"; then
  fail "train stdout contains forbidden testing/mAP/result marker"
fi

write_summary "PASS" "R16A_GPU_SMOKE_PASS_NO_MAP_NO_LONG_TRAINING" "bounded two-iteration train-entry smoke completed"
log "R16A GPU smoke complete; no tools/test.py, detector mAP, long training, runtime/FLOPs, deploy, metric, or paper claim is unlocked"
