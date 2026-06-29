#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-${1:-/root/autodl-tmp/OpenTAD_Back_check}}"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

QUEUE_LOG="$LOG_DIR/adapter_quality_rescore_driver.log"
GPU_ID="${GPU_ID:-0}"
RUN_ID="${RUN_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
TORCHRUN="${TORCHRUN:-/root/miniconda3/bin/torchrun}"
BASE_PORT="${BASE_PORT:-30610}"
GPU_FREE_MIB="${GPU_FREE_MIB:-900}"
CHECK_ONLY="${CHECK_ONLY:-0}"
STORAGE_CHECK_PATH="${STORAGE_CHECK_PATH:-$ROOT_DIR}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
ALLOW_RESUME_CHECKPOINT="${ALLOW_RESUME_CHECKPOINT:-0}"
SKIP_GPU_WAIT="${SKIP_GPU_WAIT:-0}"
KEEP_CUDA_VISIBLE_DEVICES="${KEEP_CUDA_VISIBLE_DEVICES:-0}"

CONFIG="${CONFIG:-configs/adatad/thumos/input_random_fixed_50pct_adapter_quality_rescore_detached.py}"
NAME="${NAME:-input_random_fixed_50pct_adapter_quality_rescore_detached}"
WORK_LOG="$ROOT_DIR/exps/thumos/adatad/${NAME}/gpu1_id${RUN_ID}/log.json"
EXPECT_QUALITY_LOSS_WEIGHT="${EXPECT_QUALITY_LOSS_WEIGHT:-0.10}"
EXPECT_QUALITY_SCORE_ALPHA="${EXPECT_QUALITY_SCORE_ALPHA:-0.25}"
EXPECT_QUALITY_TARGET_MODE="${EXPECT_QUALITY_TARGET_MODE:-assigned_iou}"
EXPECT_QUALITY_POSITIVE_WEIGHT="${EXPECT_QUALITY_POSITIVE_WEIGHT:-1.0}"
EXPECT_QUALITY_NEGATIVE_WEIGHT="${EXPECT_QUALITY_NEGATIVE_WEIGHT:-1.0}"
EXPECT_QUALITY_LOSS_NORMALIZER="${EXPECT_QUALITY_LOSS_NORMALIZER:-valid}"
EXPECT_QUALITY_KEEP_ZERO_GRAPH="${EXPECT_QUALITY_KEEP_ZERO_GRAPH:-0}"
EXPECT_BATCH_SIZE="${EXPECT_BATCH_SIZE:-2}"

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
assert abs(float(quality.get("bias_init", 0.0)) - 4.59511985013459) < 1e-9, quality
assert abs(float(quality.get("weight_init", 1.0))) < 1e-12, quality
assert abs(float(quality.loss_weight) - float("$EXPECT_QUALITY_LOSS_WEIGHT")) < 1e-9, quality
assert abs(float(quality.score_alpha) - float("$EXPECT_QUALITY_SCORE_ALPHA")) < 1e-9, quality
assert quality.get("target_mode", "assigned_iou") == "$EXPECT_QUALITY_TARGET_MODE", quality
assert abs(float(quality.get("positive_weight", 1.0)) - float("$EXPECT_QUALITY_POSITIVE_WEIGHT")) < 1e-9, quality
assert abs(float(quality.get("negative_weight", 1.0)) - float("$EXPECT_QUALITY_NEGATIVE_WEIGHT")) < 1e-9, quality
assert quality.get("loss_normalizer", "valid") == "$EXPECT_QUALITY_LOSS_NORMALIZER", quality
expected_keep_zero_graph = bool(int("$EXPECT_QUALITY_KEEP_ZERO_GRAPH"))
assert bool(quality.get("keep_loss_graph_when_weight_zero", False)) == expected_keep_zero_graph, quality

from opentad.models.detectors.actionformer import ActionFormer
import inspect
grad_clip_source = inspect.getsource(ActionFormer.grad_clip_parameters)
assert "exclude_quality_head" in grad_clip_source, grad_clip_source
assert 'name.startswith("rpn_head.quality_head.")' in grad_clip_source, grad_clip_source

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

solver = cfg.solver
expected_batch_size = int("$EXPECT_BATCH_SIZE")
assert int(solver.train.batch_size) == expected_batch_size, solver
assert int(solver.val.batch_size) == expected_batch_size, solver
assert int(solver.test.batch_size) == expected_batch_size, solver

print("work_dir=", cfg.work_dir)
print("model=", cfg.model.type)
print("backbone=", cfg.model.backbone.backbone.type)
print("input_pdrop=", cfg.model.projection.get("input_pdrop", 0.0))
print("quality_bias_init=", quality.bias_init)
print("quality_weight_init=", quality.weight_init)
print("quality_loss_weight=", quality.loss_weight)
print("quality_score_alpha=", quality.score_alpha)
print("quality_target_mode=", quality.get("target_mode", "assigned_iou"))
print("quality_negative_weight=", quality.get("negative_weight", 1.0))
print("quality_loss_normalizer=", quality.get("loss_normalizer", "valid"))
print("quality_keep_zero_graph=", quality.get("keep_loss_graph_when_weight_zero", False))
print("train_load=", train_load.method, train_load.method_base)
print("val_load=", val_load.method, val_load.method_base)
print("batch_size=", solver.train.batch_size, solver.val.batch_size, solver.test.batch_size)
print("checkpoint_interval=", workflow.checkpoint_interval)
print("disable_checkpoint=", workflow.get("disable_checkpoint", False))
PY
}

run_exp() {
  local timestamp log_file supervisor_log monitor_log train_pid supervisor_pid status
  local -a train_args
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
  train_args=(tools/train.py "$CONFIG" --id "$RUN_ID")
  if [[ -n "$RESUME_CHECKPOINT" ]]; then
    train_args+=(--resume "$RESUME_CHECKPOINT")
    log_msg "resuming ${NAME} from ${RESUME_CHECKPOINT}"
  fi
  if [[ "$KEEP_CUDA_VISIBLE_DEVICES" == "1" ]]; then
    "$TORCHRUN" --master_port="$BASE_PORT" --nproc_per_node=1 \
      "${train_args[@]}" > >(tee "$log_file") 2>&1 &
  else
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$TORCHRUN" --master_port="$BASE_PORT" --nproc_per_node=1 \
      "${train_args[@]}" > >(tee "$log_file") 2>&1 &
  fi
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
df -h "$STORAGE_CHECK_PATH" | tee -a "$QUEUE_LOG"

if [[ -n "$RESUME_CHECKPOINT" && "$ALLOW_RESUME_CHECKPOINT" != "1" ]]; then
  log_msg "refusing RESUME_CHECKPOINT without ALLOW_RESUME_CHECKPOINT=1: ${RESUME_CHECKPOINT}"
  exit 2
fi

check_cfg

if [[ "$CHECK_ONLY" == "1" ]]; then
  log_msg "adapter quality rescore check-only complete"
  exit 0
fi

if [[ "$SKIP_GPU_WAIT" == "1" ]]; then
  log_msg "skipping GPU wait because allocation is managed by the scheduler"
else
  wait_for_gpu
fi
run_exp
log_msg "adapter quality rescore queue complete"
