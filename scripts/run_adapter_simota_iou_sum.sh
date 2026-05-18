#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-/root/autodl-tmp/OpenTAD_Back_check}"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

QUEUE_LOG="$LOG_DIR/adapter_simota_iou_sum_driver.log"
GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
TORCHRUN="${TORCHRUN:-/root/miniconda3/bin/torchrun}"
BASE_PORT="${BASE_PORT:-30610}"
GPU_FREE_MIB="${GPU_FREE_MIB:-900}"
CHECK_ONLY="${CHECK_ONLY:-0}"
EXPECT_BATCH_SIZE="${EXPECT_BATCH_SIZE:-2}"

CONFIGS=(
  "configs/adatad/thumos/input_random_fixed_50pct_adapter_simota_mink4_w1.py"
)

NAMES=(
  "input_random_fixed_50pct_adapter_simota_mink4_w1"
)

START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-$((${#CONFIGS[@]} - 1))}"

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
  local config_path="$1"
  log_msg "checking ${config_path}"
  cd "$ROOT_DIR"
  "$PYTHON_BIN" - <<PY
import sys
sys.path.insert(0, "$ROOT_DIR")
from mmengine.config import Config

cfg = Config.fromfile("$config_path")
assert cfg.model.type == "ActionFormer", cfg.model.type
assert cfg.model.backbone.backbone.type == "VisionTransformerAdapter", cfg.model.backbone.backbone
assert cfg.model.projection.type == "Conv1DTransformerProj", cfg.model.projection
assert cfg.model.rpn_head.type == "ActionFormerHead", cfg.model.rpn_head
assert cfg.model.neck.type == "FPNIdentity", cfg.model.neck
assert not bool(getattr(cfg.model.rpn_head, "quality_head_cfg", {}).get("enabled", False)), cfg.model.rpn_head
assert abs(float(cfg.model.projection.get("input_pdrop", 0.0))) < 1e-9, cfg.model.projection
assert int(cfg.model.backbone.backbone.total_frames) == 384, cfg.model.backbone.backbone
assert int(cfg.model.projection.max_seq_len) == 384, cfg.model.projection

assigner = cfg.model.rpn_head.assigner
assert assigner.type == "AnchorFreeSimOTAAssigner", assigner
assert abs(float(assigner.cls_weight) - 1.0) < 1e-9, assigner
assert abs(float(assigner.iou_weight) - 3.0) < 1e-9, assigner
assert abs(float(assigner.center_radius) - 2.5) < 1e-9, assigner
assert abs(float(assigner.keep_percent) - 0.65) < 1e-9, assigner
assert abs(float(assigner.confuse_weight) - 1.0) < 1e-9, assigner
assert int(assigner.topk) == 9, assigner
assert int(assigner.min_k) == 4, assigner
assert not bool(assigner.filter_shortest_gt), assigner
assert assigner.dynamic_k.type == "dynamic_k_matching", assigner
assert assigner.dynamic_k.mode == "iou_sum", assigner
assert abs(float(assigner.dynamic_k.min_candidate_iou) - 0.05) < 1e-9, assigner
assert bool(cfg.model.rpn_head.assignment_debug.enabled), cfg.model.rpn_head

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
solver = cfg.solver
assert int(workflow.val_start_epoch) == 40, workflow
assert int(workflow.val_eval_interval) == 2, workflow
assert int(workflow.end_epoch) == 60, workflow
assert int(workflow.checkpoint_interval) == 10, workflow
assert not bool(workflow.get("disable_checkpoint", False)), workflow
assert int(workflow.runtime_debug_interval) == 1, workflow
expected_batch_size = int("$EXPECT_BATCH_SIZE")
assert int(cfg.solver.train.batch_size) == expected_batch_size, solver
assert int(cfg.solver.val.batch_size) == expected_batch_size, solver
assert int(cfg.solver.test.batch_size) == expected_batch_size, solver

print("work_dir=", cfg.work_dir)
print("assigner=", assigner.type)
print("dynamic_k_mode=", assigner.dynamic_k.mode)
print("min_candidate_iou=", assigner.dynamic_k.min_candidate_iou)
print("min_k=", assigner.min_k)
print("filter_shortest_gt=", assigner.filter_shortest_gt)
print("train_load=", train_load.method, train_load.method_base)
print("batch_size=", solver.train.batch_size, solver.val.batch_size, solver.test.batch_size)
PY
}

run_exp() {
  local config_path="$1"
  local name="$2"
  local port="$3"
  local log_file="$LOG_DIR/${name}_$(date '+%Y%m%d_%H%M%S').log"
  wait_for_gpu
  log_msg "starting ${name}"
  cd "$ROOT_DIR"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$TORCHRUN" --master_port="$port" --nproc_per_node=1 \
    tools/train.py "$config_path" --id 0 2>&1 | tee "$log_file"
  log_msg "finished ${name}"
}

log_msg "adapter SimOTA iou-sum queue start"
log_msg "run slice START_INDEX=${START_INDEX} END_INDEX=${END_INDEX}"
df -h /root/autodl-tmp | tee -a "$QUEUE_LOG"

for cfg in "${CONFIGS[@]}"; do
  check_cfg "$cfg"
done

if [[ "$CHECK_ONLY" == "1" ]]; then
  log_msg "adapter SimOTA iou-sum check-only complete"
  exit 0
fi

for i in "${!CONFIGS[@]}"; do
  if (( i < START_INDEX || i > END_INDEX )); then
    continue
  fi
  run_exp "${CONFIGS[$i]}" "${NAMES[$i]}" "$((BASE_PORT + i))"
done

log_msg "adapter SimOTA iou-sum queue complete"
