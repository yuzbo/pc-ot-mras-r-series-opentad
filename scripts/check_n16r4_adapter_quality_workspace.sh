#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-$HOME/run/yuzibo}"
ROOT_DIR="${ROOT_DIR:-$WORKSPACE/OpenTAD_Back_check}"
THUMOS_ROOT="${THUMOS_ROOT:-$WORKSPACE/thumos14}"
CONDA_ENV="${CONDA_ENV:-$WORKSPACE/conda_envs/opentad}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$WORKSPACE/conda_pkgs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$WORKSPACE/pip_cache}"
CONFIG="${CONFIG:-configs/adatad/thumos/input_random_fixed_50pct_adapter_quality_rescore_detached_n16r4.py}"
EXPECTED_TRAIN_MP4="${EXPECTED_TRAIN_MP4:-1010}"
EXPECTED_TEST_MP4="${EXPECTED_TEST_MP4:-1574}"

case "$WORKSPACE" in
  "$HOME"/run/yuzibo) ;;
  *)
    echo "refusing to use WORKSPACE outside ~/run/yuzibo: $WORKSPACE" >&2
    exit 2
    ;;
esac

cd "$ROOT_DIR"
mkdir -p logs

missing=0
for path in \
  "$THUMOS_ROOT/annotations/thumos_14_anno.json" \
  "$THUMOS_ROOT/annotations/category_idx.txt" \
  "$THUMOS_ROOT/train" \
  "$THUMOS_ROOT/test"; do
  if [[ ! -e "$path" ]]; then
    echo "missing: $path" >&2
    missing=1
  fi
done

if [[ "$missing" != "0" ]]; then
  exit 1
fi

for dir in "$THUMOS_ROOT/train" "$THUMOS_ROOT/test"; do
  if [[ ! -d "$dir" ]]; then
    echo "not a directory: $dir" >&2
    exit 1
  fi
done

train_count=$(find "$THUMOS_ROOT/train" -maxdepth 1 -type f -name '*.mp4' | wc -l)
test_count=$(find "$THUMOS_ROOT/test" -maxdepth 1 -type f -name '*.mp4' | wc -l)
if [[ "$train_count" -ne "$EXPECTED_TRAIN_MP4" ]]; then
  echo "train mp4 count mismatch: got=$train_count expected=$EXPECTED_TRAIN_MP4" >&2
  exit 1
fi
if [[ "$test_count" -ne "$EXPECTED_TEST_MP4" ]]; then
  echo "test mp4 count mismatch: got=$test_count expected=$EXPECTED_TEST_MP4" >&2
  exit 1
fi

if ! command -v module >/dev/null 2>&1; then
  set +u
  source /etc/profile
  set -u
fi

module load cuda/11.8
module load miniforge3/24.11
source activate "$CONDA_ENV"

ROOT_DIR="$ROOT_DIR" CONFIG="$CONFIG" "$CONDA_ENV/bin/python" - <<'PY'
import os
from pathlib import Path
from mmengine.config import Config

root_dir = Path(os.environ["ROOT_DIR"])
cfg = Config.fromfile(os.environ["CONFIG"])
pretrain = cfg.model.backbone.custom.get("pretrain")
if not pretrain:
    raise SystemExit("missing cfg.model.backbone.custom.pretrain")
pretrain_path = Path(pretrain)
if not pretrain_path.is_absolute():
    pretrain_path = root_dir / pretrain_path
if not pretrain_path.exists():
    raise SystemExit(f"missing pretrained checkpoint from config: {pretrain_path}")
print("pretrain_checkpoint=", pretrain_path)
print("work_dir=", cfg.work_dir)
PY

ROOT_DIR="$ROOT_DIR" \
THUMOS_ROOT="$THUMOS_ROOT" \
STORAGE_CHECK_PATH="$WORKSPACE" \
CONFIG="$CONFIG" \
PYTHON_BIN="$(command -v python)" \
TORCHRUN="$(command -v torchrun)" \
CHECK_ONLY=1 \
bash scripts/run_adapter_quality_rescore.sh "$ROOT_DIR"
