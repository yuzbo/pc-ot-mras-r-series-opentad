#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-/root/autodl-tmp/OpenTAD_Back_check}"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

QUEUE_LOG="$LOG_DIR/adapter_pseudo_boundary_snap_pair_driver.log"
GPU_ID="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
TORCHRUN="${TORCHRUN:-/root/miniconda3/bin/torchrun}"
BASE_PORT="${BASE_PORT:-30820}"
GPU_FREE_MIB="${GPU_FREE_MIB:-900}"
CHECK_ONLY="${CHECK_ONLY:-0}"
SKIP_CACHE_BUILD="${SKIP_CACHE_BUILD:-0}"
FORCE_CACHE_BUILD="${FORCE_CACHE_BUILD:-0}"
TEACHER_CKPT="${TEACHER_CKPT:-$ROOT_DIR/exps/thumos/adatad/input_random_fixed_50pct_adapter_virtual_baseline/gpu1_id0/checkpoint/epoch_59.pth}"
ANN_PATH="${ANN_PATH:-/root/autodl-tmp/annotations/thumos_14_anno.json}"
CACHE_ROOT="${CACHE_ROOT:-$ROOT_DIR/pseudo_boundary_cache/adapter_virtual_baseline}"

TEACHER_TRAIN_CFG="configs/adatad/thumos/input_random_fixed_50pct_adapter_teacher_cache_train.py"
TEACHER_VAL_CFG="configs/adatad/thumos/input_random_fixed_50pct_adapter_teacher_cache_val.py"

CONFIGS=(
  "configs/adatad/thumos/input_random_fixed_50pct_adapter_pseudo_boundary_snap_q32.py"
  "configs/adatad/thumos/input_random_fixed_50pct_adapter_pseudo_boundary_snap_q64.py"
)

NAMES=(
  "input_random_fixed_50pct_adapter_pseudo_boundary_snap_q32"
  "input_random_fixed_50pct_adapter_pseudo_boundary_snap_q64"
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
workflow = cfg.workflow
assert cfg.model.type == "ActionFormer", cfg.model.type
assert cfg.model.backbone.backbone.type == "VisionTransformerAdapter", cfg.model.backbone.backbone
assert cfg.model.rpn_head.type == "ActionFormerHead", cfg.model.rpn_head
assert int(cfg.model.backbone.backbone.total_frames) == 384, cfg.model.backbone.backbone
assert int(cfg.model.projection.max_seq_len) == 384, cfg.model.projection
assert abs(float(cfg.model.projection.get("input_pdrop", 0.0))) < 1e-9, cfg.model.projection
assert int(workflow.checkpoint_interval) == 10, workflow
assert not bool(workflow.get("disable_checkpoint", False)), workflow
assert int(workflow.val_start_epoch) == 40, workflow
for split_name, split in (("train", cfg.dataset.train), ("val", cfg.dataset.val), ("test", cfg.dataset.test)):
    load = next(x for x in split.pipeline if x.type == "LoadFrames")
    collect = next(x for x in split.pipeline if x.type == "Collect")
    collect_keys = collect["keys"]
    if isinstance(collect_keys, str):
        collect_keys = [collect_keys]
    assert load.method == "pseudo_boundary_snap_subsample", (split_name, load)
    assert int(load.target_len) == 384, (split_name, load)
    assert load.pseudo_boundary_fallback == "random_fixed", (split_name, load)
    assert int(load.pseudo_boundary_snap_distance) == 2, (split_name, load)
    if split_name == "test":
        assert "gt_segments" not in collect_keys, collect
        assert "gt_labels" not in collect_keys, collect
    else:
        assert "gt_segments" in collect_keys, collect
        assert "gt_labels" in collect_keys, collect
print("work_dir=", cfg.work_dir)
print("quota=", next(x for x in cfg.dataset.train.pipeline if x.type == "LoadFrames").pseudo_boundary_quota)
PY
}

check_cache_manifest() {
  local cache_dir="$1"
  local expected_subset="$2"
  log_msg "checking pseudo-boundary cache manifest: ${cache_dir}"
  "$PYTHON_BIN" - <<PY
import json
from pathlib import Path

manifest_path = Path("$cache_dir") / "manifest.json"
if not manifest_path.is_file():
    raise SystemExit(f"missing pseudo-boundary cache manifest: {manifest_path}")
with manifest_path.open("r", encoding="utf-8") as f:
    manifest = json.load(f)
if bool(manifest.get("uses_gt", False)):
    raise SystemExit(f"pseudo-boundary cache must not use GT: {manifest_path}")
axis = manifest.get("axis", "global_snippet_index")
if axis != "global_snippet_index":
    raise SystemExit(f"unsupported pseudo-boundary cache axis: {axis}")
subset = manifest.get("subset")
if subset != "$expected_subset":
    raise SystemExit(f"unexpected pseudo-boundary cache subset: got={subset!r} expected='$expected_subset'")
source = manifest.get("source", "")
if source != "postprocessed_teacher_detections":
    raise SystemExit(f"unexpected pseudo-boundary cache source: {source!r}")
written = int(manifest.get("videos_written", 0))
if written <= 0:
    raise SystemExit(f"pseudo-boundary cache manifest has no written videos: {manifest_path}")
print("manifest=", manifest_path)
print("uses_gt=", manifest.get("uses_gt", False))
print("subset=", subset)
print("source=", source)
print("videos_written=", written)
PY
}

run_teacher_export() {
  local cfg="$1"
  local name="$2"
  local result_path="$ROOT_DIR/exps/thumos/adatad/${name}/gpu1_id0/result_detection.json"
  if [[ "$FORCE_CACHE_BUILD" != "1" && -s "$result_path" ]]; then
    log_msg "teacher export exists: $result_path"
    return
  fi
  if [[ ! -s "$TEACHER_CKPT" ]]; then
    log_msg "missing TEACHER_CKPT=$TEACHER_CKPT"
    exit 1
  fi
  wait_for_gpu
  log_msg "exporting teacher detections: $cfg"
  cd "$ROOT_DIR"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$TORCHRUN" --master_port="$BASE_PORT" --nproc_per_node=1 \
    tools/test.py "$cfg" --checkpoint "$TEACHER_CKPT" --id 0 --not_eval 2>&1 | tee "$LOG_DIR/${name}_export_$(date '+%Y%m%d_%H%M%S').log"
}

build_cache() {
  local subset="$1"
  local pred="$2"
  local out_dir="$3"
  if [[ "$FORCE_CACHE_BUILD" != "1" && -s "$out_dir/manifest.json" ]]; then
    log_msg "pseudo-boundary cache exists: $out_dir"
    return
  fi
  log_msg "building pseudo-boundary cache for subset=$subset"
  cd "$ROOT_DIR"
  "$PYTHON_BIN" scripts/build_pseudo_boundary_cache.py \
    --ann "$ANN_PATH" \
    --pred "$pred" \
    --out "$out_dir" \
    --subset "$subset" \
    --snippet-stride 4 \
    --max-detections 2000 \
    --min-score 0.001 \
    --source-config "$TEACHER_VAL_CFG" \
    --source-checkpoint "$TEACHER_CKPT" 2>&1 | tee -a "$QUEUE_LOG"
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

log_msg "adapter pseudo-boundary snap pair queue start"
log_msg "run slice START_INDEX=${START_INDEX} END_INDEX=${END_INDEX}"
df -h /root/autodl-tmp | tee -a "$QUEUE_LOG"

for cfg in "${CONFIGS[@]}"; do
  check_cfg "$cfg"
done

if [[ "$SKIP_CACHE_BUILD" == "1" || "$CHECK_ONLY" == "1" ]]; then
  check_cache_manifest "$CACHE_ROOT/train" "training"
  check_cache_manifest "$CACHE_ROOT/validation" "validation"
fi

if [[ "$CHECK_ONLY" == "1" ]]; then
  log_msg "adapter pseudo-boundary snap pair check-only complete"
  exit 0
fi

if [[ "$SKIP_CACHE_BUILD" != "1" ]]; then
  run_teacher_export "$TEACHER_TRAIN_CFG" "input_random_fixed_50pct_adapter_teacher_cache_train"
  run_teacher_export "$TEACHER_VAL_CFG" "input_random_fixed_50pct_adapter_teacher_cache_val"
  build_cache "training" "$ROOT_DIR/exps/thumos/adatad/input_random_fixed_50pct_adapter_teacher_cache_train/gpu1_id0/result_detection.json" "$CACHE_ROOT/train"
  build_cache "validation" "$ROOT_DIR/exps/thumos/adatad/input_random_fixed_50pct_adapter_teacher_cache_val/gpu1_id0/result_detection.json" "$CACHE_ROOT/validation"
  check_cache_manifest "$CACHE_ROOT/train" "training"
  check_cache_manifest "$CACHE_ROOT/validation" "validation"
fi

for i in "${!CONFIGS[@]}"; do
  if (( i < START_INDEX || i > END_INDEX )); then
    continue
  fi
  run_exp "${CONFIGS[$i]}" "${NAMES[$i]}" "$((BASE_PORT + 10 + i))"
done

log_msg "adapter pseudo-boundary snap pair queue complete"
