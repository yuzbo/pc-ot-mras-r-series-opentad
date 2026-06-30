#!/usr/bin/env bash
set -euo pipefail

# ABR short diagnostic child-context wrapper only.
# Do not call this script with sbatch. It never submits Slurm jobs, cancels jobs,
# releases parent holds, evaluates checkpoints, or runs tools/test.py.

MODE="${1:-verify}"
CONFIG="configs/adatad/thumos/input_abr_active_bracket_refinement_adapter_irregular_headv3_shortdiag.py"
VALIDATOR="tools/abr/validate_abr_shortdiag.py"
LOG_DIR="${ABR_SHORTDIAG_LOG_DIR:-logs/abr_shortdiag}"
RUN_ID="${ABR_SHORTDIAG_ID:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

case "${MODE}" in
  verify|train)
    ;;
  *)
    echo "Usage: $0 [verify|train]" >&2
    exit 2
    ;;
esac

python -m py_compile \
  "${CONFIG}" \
  "${VALIDATOR}" \
  tests/test_abr_shortdiag.py

python -m pytest \
  tests/test_abr_shortdiag.py \
  tests/test_abr_core.py \
  tests/test_abr_pipeline_and_gate.py \
  -q

python tools/abr/validate_abr_launch_gate.py --config configs/adatad/thumos/input_abr_active_bracket_refinement_adapter_irregular_headv3.py
python "${VALIDATOR}" --config "${CONFIG}"

if [[ "${MODE}" == "verify" ]]; then
  echo "PASS: ABR short diagnostic local gate verified; training remains locked unless invoked in child GPU context."
  exit 0
fi

if [[ -z "${SLURM_JOB_ID:-}" && "${ABR_CHILD_GPU_CONTEXT:-0}" != "1" ]]; then
  echo "LOCKED: train mode requires an already allocated child GPU context (SLURM_JOB_ID or ABR_CHILD_GPU_CONTEXT=1)." >&2
  exit 1
fi

if [[ "${ABR_SHORTDIAG_ACK:-0}" != "1" ]]; then
  echo "LOCKED: set ABR_SHORTDIAG_ACK=1 to acknowledge one-epoch/no-eval/no-checkpoint/no-claim semantics." >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
TRAIN_LOG="${LOG_DIR}/abr_shortdiag_train_${RUN_ID}.log"

torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" \
  tools/train.py "${CONFIG}" --id "${RUN_ID}" --not_eval \
  2>&1 | tee "${TRAIN_LOG}"

python "${VALIDATOR}" --config "${CONFIG}" --train-log "${TRAIN_LOG}"
echo "PASS: ABR short diagnostic produced finite train loss only; full train, evaluation, checkpoint claims, sparse-compute claims, deploy claims, and paper claims remain locked."
