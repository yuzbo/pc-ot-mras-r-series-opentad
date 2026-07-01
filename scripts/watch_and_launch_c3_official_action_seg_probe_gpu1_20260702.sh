#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/run01/sczc063/yuzibo/OpenTAD_C3TCNCoarseProbe_20260701}"
PARENT_JOB_ID="${PARENT_JOB_ID:-1118197}"
NODE_NAME="${NODE_NAME:-g0030}"
OUT_DIR="${OUT_DIR:-/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/outputs/c3_official_action_seg_probe_gpu1_$(date +%Y%m%d_%H%M%S_%z)}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs}"
WAIT_SECONDS="${WAIT_SECONDS:-120}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-0}"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_DIR}"

start_ts=$(date +%s)
echo "OFFICIAL_ACTION_SEG_WATCH_START $(date -Iseconds) PROJECT_DIR=${PROJECT_DIR} OUT_DIR=${OUT_DIR}"
echo "PARENT_JOB_ID=${PARENT_JOB_ID} NODE_NAME=${NODE_NAME} WAIT_SECONDS=${WAIT_SECONDS} MAX_WAIT_SECONDS=${MAX_WAIT_SECONDS}"

while true; do
  running=$(
    {
      squeue --steps -j "${PARENT_JOB_ID}" 2>/dev/null || true
      ps -u "${USER}" -o pid,ppid,stat,etime,cmd 2>/dev/null || true
    } | grep -E 'c3_tcn_g1|launch_c3_tcn|train_lowres_action_probe.py.*temporal-tcn|official_action_seg_g1|c3_asfdl|c3_asfdlt_g1|c3_asformer_delta_ledger|run_c3_asformer_delta_ledger_adatad_full_train_gpu1' | grep -v grep || true
  )
  if [[ -z "${running}" ]]; then
    break
  else
    echo "Waiting for active C3 child at $(date -Iseconds): ${running}"
  fi
  if [[ "${MAX_WAIT_SECONDS}" != "0" ]]; then
    now_ts=$(date +%s)
    if (( now_ts - start_ts > MAX_WAIT_SECONDS )); then
      echo "Timed out waiting for GPU1/C3 child release." >&2
      exit 77
    fi
  fi
  sleep "${WAIT_SECONDS}"
done

echo "OFFICIAL_ACTION_SEG_GPU1_READY $(date -Iseconds)"

srun --jobid="${PARENT_JOB_ID}" --overlap -N1 -n1 -w "${NODE_NAME}" --cpus-per-task=8 -J official_action_seg_g1 bash -lc "
  set -euo pipefail
  cd '${PROJECT_DIR}'
  export CUDA_VISIBLE_DEVICES=1
  unset SLURM_STEP_GPUS
  unset SLURM_JOB_GPUS
  export PROJECT_DIR='${PROJECT_DIR}'
  export OUT_DIR='${OUT_DIR}'
  echo OFFICIAL_ACTION_SEG_CHILD_START \$(date --iso-8601=seconds) CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES OUT_DIR=\$OUT_DIR
  bash scripts/run_c3_official_action_seg_probe_gpu1_20260702.sh
"

echo "OFFICIAL_ACTION_SEG_WATCH_DONE $(date -Iseconds)"
