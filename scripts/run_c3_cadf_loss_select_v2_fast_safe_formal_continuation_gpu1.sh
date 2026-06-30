#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/adatad/thumos/c3_cadf_densitymesh_original_adatad_32px_loss_select_v2_fast_safe_formal.py"

if [[ "${CUDA_VISIBLE_DEVICES:-}" != "1" ]]; then
  echo "ERROR: C3 CADF loss-select V2 modified-validation-schedule continuation is GPU1-only. Set CUDA_VISIBLE_DEVICES=1." >&2
  echo "This launcher is not clean full train; it resumes an existing checkpoint under a new continuation run id." >&2
  exit 2
fi

if [[ "${CADF_LOSS_SELECT_V2_CONTINUATION_UNLOCK:-}" != "CONFIRMED" ]]; then
  echo "ERROR: modified-validation-schedule continuation is locked." >&2
  echo "Export CADF_LOSS_SELECT_V2_CONTINUATION_UNLOCK=CONFIRMED only after recording continuation evidence." >&2
  echo "This launcher is not clean full train." >&2
  exit 2
fi

if [[ -z "${CADF_LOSS_SELECT_V2_CONTINUATION_EVIDENCE:-}" ]]; then
  echo "ERROR: CADF_LOSS_SELECT_V2_CONTINUATION_EVIDENCE must point to a local evidence file." >&2
  echo "This modified-validation-schedule continuation is not clean full train." >&2
  exit 2
fi

if [[ ! -f "$CADF_LOSS_SELECT_V2_CONTINUATION_EVIDENCE" ]]; then
  echo "ERROR: CADF_LOSS_SELECT_V2_CONTINUATION_EVIDENCE does not exist: $CADF_LOSS_SELECT_V2_CONTINUATION_EVIDENCE" >&2
  echo "This modified-validation-schedule continuation is not clean full train." >&2
  exit 2
fi

if [[ -z "${CADF_LOSS_SELECT_V2_RESUME_CHECKPOINT:-}" ]]; then
  echo "ERROR: CADF_LOSS_SELECT_V2_RESUME_CHECKPOINT must point to the existing checkpoint to resume." >&2
  echo "This modified-validation-schedule continuation is not clean full train." >&2
  exit 2
fi

if [[ ! -f "$CADF_LOSS_SELECT_V2_RESUME_CHECKPOINT" ]]; then
  echo "ERROR: CADF_LOSS_SELECT_V2_RESUME_CHECKPOINT does not exist: $CADF_LOSS_SELECT_V2_RESUME_CHECKPOINT" >&2
  echo "This modified-validation-schedule continuation is not clean full train." >&2
  exit 2
fi

if [[ -z "${CADF_LOSS_SELECT_V2_CONTINUATION_RUN_ID:-}" ]]; then
  echo "ERROR: CADF_LOSS_SELECT_V2_CONTINUATION_RUN_ID must be set to a new continuation run id." >&2
  echo "This launcher intentionally has no default run id and is not clean full train." >&2
  exit 2
fi

if ! [[ "$CADF_LOSS_SELECT_V2_CONTINUATION_RUN_ID" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: CADF_LOSS_SELECT_V2_CONTINUATION_RUN_ID must be a positive integer and must not be 0." >&2
  echo "Use a new continuation id so this resumed run cannot be confused with gpu1_id0 clean/full-train outputs." >&2
  exit 2
fi

echo "GPU1 C3 CADF loss-select V2 modified-validation-schedule continuation; not clean full train"
echo "Config: $CONFIG"
echo "Resume checkpoint: $CADF_LOSS_SELECT_V2_RESUME_CHECKPOINT"
echo "Continuation evidence: $CADF_LOSS_SELECT_V2_CONTINUATION_EVIDENCE"
echo "Continuation run id: $CADF_LOSS_SELECT_V2_CONTINUATION_RUN_ID"

export LOCAL_RANK="${LOCAL_RANK:-0}"
export RANK="${RANK:-0}"
export WORLD_SIZE="${WORLD_SIZE:-1}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-${CADF_LOSS_SELECT_V2_MASTER_PORT:-30035}}"
python tools/train.py \
  "$CONFIG" \
  --id "$CADF_LOSS_SELECT_V2_CONTINUATION_RUN_ID" \
  --resume "$CADF_LOSS_SELECT_V2_RESUME_CHECKPOINT"
