#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-/root/autodl-tmp/OpenTAD_Back_check}"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

QUEUE_LOG="$LOG_DIR/adapter_quality_rescore_driver.log"
GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
TORCHRUN="${TORCHRUN:-/root/miniconda3/bin/torchrun}"
BASE_PORT="${BASE_PORT:-30610}"
GPU_FREE_MIB="${GPU_FREE_MIB:-900}"
CHECK_ONLY="${CHECK_ONLY:-0}"

CONFIG="configs/adatad/thumos/input_random_fixed_50pct_adapter_quality_rescore_detached.py"
NAME="input_random_fixed_50pct_adapter_quality_rescore_detached"
WORK_LOG="$ROOT_DIR/exps/thumos/adatad/${NAME}/gpu1_id0/log.json"

log_msg() {
  echo "$(date '+%F %T') $*" | tee -a "$QUEUE_LOG"
}

wait_for_gpu() {
  while true; do
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n "$((GPU_ID + 1))p")
    if [[ -n "${used}" && "${used}" -lt "$GPU_FREE_MIB" ]]; then
      break
    fi
    log_msg "waiting for GPU ${GPU_ID}, memory.used=${used:-unknown} MiB, threshold=${GPU_FREE_MIB} MiB"
    sleep 60
  done
}

check_cfg() {
  log_msg "checking ${CONFIG}"
  cd "$ROOT_DIR"
  "$PYTHON_BIN" - <<PY
import sys
sys.path.insert(0, "$ROOT_DIR")
from mmengine.config import Config

cfg = Config.fromfile("$CONFIG")
assert cfg.model.type == "ActionFormer", cfg.model.type
assert cfg.model.backbone.backbone.type == "VisionTransformerAdapter", cfg.model.backbone.backbone
assert cfg.model.projection.type == "Conv1DTransformerProj", cfg.model.projection
assert cfg.model.rpn_head.type == "ActionFormerHead", cfg.model.rpn_head
assert cfg.model.neck.type == "FPNIdentity", cfg.model.neck
assert abs(float(cfg.model.projection.get("input_pdrop", 0.0))) < 1e-9, cfg.model.projection
assert int(cfg.model.backbone.backbone.total_frames) == 384, cfg.model.backbone.backbone
assert int(cfg.model.projection.max_seq_len) == 384, cfg.model.projection

quality = cfg.model.rpn_head.quality_head_cfg
assert cfg.model.rpn_head.quality_head_cfg.enabled, quality
assert int(quality.kernel_size) == 3, quality
assert abs(float(quality.loss_weight) - 0.10) < 1e-9, quality
assert abs(float(quality.score_alpha) - 0.25) < 1e-9, quality

train = cfg.dataset.train
val = cfg.dataset.val
test = cfg.dataset.test
assert int(train.sample_stride) == 1, train
assert int(val.sample_stride) == 1, val
assert int(test.sample_stride) == 1, test
assert int(val.window_size) == 768, val
assert int(test.window_size) == 768, test
train_load = next(x for x in train.pipeline if x.type == "LoadFrames")
val_load = next(x for x in val.pipeline if x.type == "LoadFrames")
test_load = next(x for x in test.pipeline if x.type == "LoadFrames")
assert train_load.method == "random_fixed_subsample", train_load
assert train_load.method_base == "random_trunc", train_load
assert val_load.method == "random_fixed_subsample", val_load
assert val_load.method_base == "sliding_window", val_load
assert test_load.method == "random_fixed_subsample", test_load
assert test_load.method_base == "sliding_window", test_load

workflow = cfg.workflow
assert int(workflow.checkpoint_interval) == 10, workflow
assert not bool(workflow.get("disable_checkpoint", False)), workflow
assert int(workflow.val_start_epoch) == 40, workflow
assert int(workflow.val_eval_interval) == 2, workflow
assert int(workflow.end_epoch) == 60, workflow

print("work_dir=", cfg.work_dir)
print("model=", cfg.model.type)
print("backbone=", cfg.model.backbone.backbone.type)
print("input_pdrop=", cfg.model.projection.get("input_pdrop", 0.0))
print("quality_loss_weight=", quality.loss_weight)
print("quality_score_alpha=", quality.score_alpha)
print("train_load=", train_load.method, train_load.method_base)
print("val_load=", val_load.method, val_load.method_base)
print("checkpoint_interval=", workflow.checkpoint_interval)
print("disable_checkpoint=", workflow.get("disable_checkpoint", False))
PY
}

run_exp() {
  local timestamp log_file supervisor_log monitor_log train_pid supervisor_pid status
  train_pid=""
  supervisor_pid=""
  timestamp="$(date '+%Y%m%d_%H%M%S')"
  log_file="$LOG_DIR/${NAME}_${timestamp}.log"
  supervisor_log="$LOG_DIR/${NAME}_supervisor_${timestamp}.md"
  monitor_log="$log_file"
  log_msg "starting ${NAME}"
  cd "$ROOT_DIR"

  cleanup() {
    if [[ -n "${supervisor_pid}" ]]; then
      kill "$supervisor_pid" 2>/dev/null || true
    fi
    if [[ -n "${train_pid}" ]]; then
      kill "$train_pid" 2>/dev/null || true
    fi
  }

  trap 'cleanup; exit 130' INT TERM

  set +e
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$TORCHRUN" --master_port="$BASE_PORT" --nproc_per_node=1 \
    tools/train.py "$CONFIG" --id 0 > >(tee "$log_file") 2>&1 &
  train_pid="$!"

  for _ in {1..60}; do
    if [[ -f "$WORK_LOG" ]]; then
      monitor_log="$WORK_LOG"
      break
    fi
    sleep 1
  done

  PROCESS_PID="$train_pid" \
    PROCESS_PATTERN="tools/train.py.*${CONFIG}" \
    INTERVAL_SECONDS="${SUPERVISE_INTERVAL_SECONDS:-1800}" \
    bash scripts/supervise_adapter_quality.sh "${NAME}_${GPU_ID}" "$monitor_log" "$supervisor_log" &
  supervisor_pid="$!"

  wait "$train_pid"
  status="$?"
  sleep "${SUPERVISE_FINAL_FLUSH_SECONDS:-1}"
  kill "$supervisor_pid" 2>/dev/null || true
  trap - INT TERM
  set -e
  log_msg "finished ${NAME} status=${status} supervisor_log=${supervisor_log}"
  return "$status"
}

log_msg "adapter quality rescore queue start"
df -h /root/autodl-tmp | tee -a "$QUEUE_LOG"
check_cfg

if [[ "$CHECK_ONLY" == "1" ]]; then
  log_msg "adapter quality rescore check-only complete"
  exit 0
fi

wait_for_gpu
run_exp
log_msg "adapter quality rescore queue complete"
