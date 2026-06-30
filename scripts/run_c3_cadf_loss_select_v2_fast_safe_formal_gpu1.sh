#!/usr/bin/env bash
set -euo pipefail

if [[ "${CUDA_VISIBLE_DEVICES:-}" != "1" ]]; then
  echo "ERROR: C3 CADF loss-select V2 fast-safe formal candidate is GPU1-only. Set CUDA_VISIBLE_DEVICES=1." >&2
  exit 2
fi

if [[ "${CADF_LOSS_SELECT_V2_FAST_SAFE_FORMAL_UNLOCK:-}" != "CONFIRMED" ]]; then
  echo "ERROR: fast-safe formal fulltrain candidate is locked. Export CADF_LOSS_SELECT_V2_FAST_SAFE_FORMAL_UNLOCK=CONFIRMED only after speed smoke passes." >&2
  exit 2
fi

if [[ ! -f "${CADF_LOSS_SELECT_V2_UNLOCK_EVIDENCE:-}" ]]; then
  echo "ERROR: CADF_LOSS_SELECT_V2_UNLOCK_EVIDENCE must point to a local evidence file." >&2
  exit 2
fi

echo "GPU1 C3 CADF loss-select V2 fast-safe formal candidate"
export LOCAL_RANK="${LOCAL_RANK:-0}"
export RANK="${RANK:-0}"
export WORLD_SIZE="${WORLD_SIZE:-1}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-${CADF_LOSS_SELECT_V2_MASTER_PORT:-30034}}"
python tools/train.py \
  configs/adatad/thumos/c3_cadf_densitymesh_original_adatad_32px_loss_select_v2_fast_safe_formal.py \
  --id "${CADF_LOSS_SELECT_V2_RUN_ID:-0}"
