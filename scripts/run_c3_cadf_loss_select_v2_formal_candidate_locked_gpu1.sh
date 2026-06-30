#!/usr/bin/env bash
set -euo pipefail

if [[ "${CUDA_VISIBLE_DEVICES:-}" != "1" ]]; then
  echo "ERROR: C3 CADF loss-select V2 formal candidate is GPU1-only. Set CUDA_VISIBLE_DEVICES=1." >&2
  exit 2
fi

if [[ "${CADF_LOSS_SELECT_V2_FORMAL_UNLOCK:-}" != "CONFIRMED" ]]; then
  echo "ERROR: formal fulltrain candidate is locked. Export CADF_LOSS_SELECT_V2_FORMAL_UNLOCK=CONFIRMED only after user unlock and local gates." >&2
  exit 2
fi

if [[ ! -f "${CADF_LOSS_SELECT_V2_UNLOCK_EVIDENCE:-}" ]]; then
  echo "ERROR: CADF_LOSS_SELECT_V2_UNLOCK_EVIDENCE must point to a local evidence file." >&2
  exit 2
fi

echo "GPU1 C3 CADF loss-select V2 formal candidate"
python tools/train.py \
  configs/adatad/thumos/c3_cadf_densitymesh_original_adatad_32px_loss_select_v2_formal_candidate_locked.py \
  --id "${CADF_LOSS_SELECT_V2_RUN_ID:-0}"
