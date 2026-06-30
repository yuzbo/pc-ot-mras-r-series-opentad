#!/usr/bin/env bash
set -euo pipefail

if [[ "${CUDA_VISIBLE_DEVICES:-}" != "1" ]]; then
  echo "ERROR: C3 CADF loss-select V2 train-iter profile is GPU1-only. Set CUDA_VISIBLE_DEVICES=1." >&2
  exit 2
fi

echo "GPU1 C3 CADF loss-select V2 train-iter speed profile; no eval, no checkpoint, no final claim"
export LOCAL_RANK="${LOCAL_RANK:-0}"
export RANK="${RANK:-0}"
export WORLD_SIZE="${WORLD_SIZE:-1}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-${CADF_LOSS_SELECT_V2_PROFILE_MASTER_PORT:-30036}}"
python tools/train.py \
  configs/adatad/thumos/c3_cadf_densitymesh_original_adatad_32px_loss_select_v2_fast_safe_train_iter_profile.py \
  --id "${CADF_LOSS_SELECT_V2_PROFILE_RUN_ID:-0}"
