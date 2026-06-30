#!/usr/bin/env bash
set -euo pipefail

if [[ "${CUDA_VISIBLE_DEVICES:-}" != "1" ]]; then
  echo "ERROR: C3 CADF loss-select V2 precheck is GPU1-only. Set CUDA_VISIBLE_DEVICES=1." >&2
  exit 2
fi

echo "GPU1 C3 CADF loss-select V2 precheck"
python tools/train.py \
  configs/adatad/thumos/c3_cadf_densitymesh_original_adatad_32px_loss_select_v2_precheck.py \
  --id "${CADF_LOSS_SELECT_V2_RUN_ID:-0}"
