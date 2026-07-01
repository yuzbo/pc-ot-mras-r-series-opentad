#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/run01/sczc063/yuzibo/OpenTAD_C3TCNCoarseProbe_20260701}"
OUT_ROOT="${OUT_ROOT:-/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/ledger_exports}"
EXPORT_TAG="${EXPORT_TAG:-c3_lowres_probe_ledger_export_$(date '+%Y%m%d_%H%M%S_%z')}"
EXPORT_SPLIT="${EXPORT_SPLIT:-val}"
PROBE_MODEL="${PROBE_MODEL:-temporal-tcn}"
TCN_VARIANT="${TCN_VARIANT:-gated}"
MATRIX_MODEL_ID="${MATRIX_MODEL_ID:-}"
OFFICIAL_ACTION_SEG_BACKEND="${OFFICIAL_ACTION_SEG_BACKEND:-}"
PROBE_CHECKPOINT="${PROBE_CHECKPOINT:-}"
SELECTION_STRATEGY="${SELECTION_STRATEGY:-delta_p_action}"
SCOUT_SPATIAL_SIZE="${SCOUT_SPATIAL_SIZE:-64}"
DENSE_WINDOW_SIZE="${DENSE_WINDOW_SIZE:-768}"
TARGET_LEN="${TARGET_LEN:-384}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-0}"
BOUNDARY_RADIUS="${BOUNDARY_RADIUS:-1}"
CONFIG="${CONFIG:-configs/adatad/thumos/pc_ot_mras_a_uniform_scaffold_small_actionness_strict_maxgap_c3_physical_grid_actionformer_n16r4.py}"

ANN_FILE="${ANN_FILE:-/data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json}"
CLASS_MAP="${CLASS_MAP:-/data/run01/sczc063/yuzibo/thumos14/annotations/category_idx.txt}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-/data/run01/sczc063/yuzibo/raw/Validation Data/validation}"
TEST_DATA_PATH="${TEST_DATA_PATH:-/data/run01/sczc063/yuzibo/raw/Test Data/TH14_test_set_mp4}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
if [[ "${CUDA_VISIBLE_DEVICES}" != "1" ]]; then
  echo "Refusing to start: CUDA_VISIBLE_DEVICES must be exactly 1 for C3 mainline GPU1, got '${CUDA_VISIBLE_DEVICES}'." >&2
  exit 44
fi
echo "SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
if [[ -z "${PROBE_CHECKPOINT}" ]]; then
  echo "PROBE_CHECKPOINT is required; use the selected coarse-classifier probe_reader.pth." >&2
  exit 47
fi

case "${EXPORT_SPLIT}" in
  train)
    VAL_DATA_PATH="${VAL_DATA_PATH:-${TRAIN_DATA_PATH}}"
    VAL_SUBSET_NAME="${VAL_SUBSET_NAME:-training}"
    EVAL_WINDOW_OVERLAP_RATIO="${EVAL_WINDOW_OVERLAP_RATIO:-0.25}"
    ;;
  val|test)
    VAL_DATA_PATH="${VAL_DATA_PATH:-${TEST_DATA_PATH}}"
    VAL_SUBSET_NAME="${VAL_SUBSET_NAME:-validation}"
    if [[ "${EXPORT_SPLIT}" == "test" ]]; then
      EVAL_WINDOW_OVERLAP_RATIO="${EVAL_WINDOW_OVERLAP_RATIO:-0.5}"
    else
      EVAL_WINDOW_OVERLAP_RATIO="${EVAL_WINDOW_OVERLAP_RATIO:-0.25}"
    fi
    ;;
  *)
    echo "EXPORT_SPLIT must be train, val, or test; got '${EXPORT_SPLIT}'." >&2
    exit 48
    ;;
esac

RUN_DIR="${OUT_ROOT}/${EXPORT_TAG}/${EXPORT_SPLIT}"
SAMPLES_JSONL="${RUN_DIR}/samples.jsonl"
LEDGER_JSONL="${RUN_DIR}/value_transport_ledger_${SELECTION_STRATEGY}_${TARGET_LEN}.jsonl"
LEDGER_SUMMARY="${RUN_DIR}/value_transport_ledger_${SELECTION_STRATEGY}_${TARGET_LEN}.summary.json"
mkdir -p "${RUN_DIR}"

cd "${PROJECT_DIR}"
if command -v module >/dev/null 2>&1; then
  module load cuda/11.8
  module load miniforge3/24.11
else
  echo "module command not found; using existing environment paths."
fi
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1

echo "START $(date -Iseconds)"
echo "EXPORT_SPLIT=${EXPORT_SPLIT}"
echo "PROBE_MODEL=${PROBE_MODEL}"
echo "TCN_VARIANT=${TCN_VARIANT}"
echo "MATRIX_MODEL_ID=${MATRIX_MODEL_ID}"
echo "OFFICIAL_ACTION_SEG_BACKEND=${OFFICIAL_ACTION_SEG_BACKEND}"
echo "PROBE_CHECKPOINT=${PROBE_CHECKPOINT}"
echo "SELECTION_STRATEGY=${SELECTION_STRATEGY}"
echo "DENSE_WINDOW_SIZE=${DENSE_WINDOW_SIZE}"
echo "TARGET_LEN=${TARGET_LEN}"
echo "EVAL_WINDOW_OVERLAP_RATIO=${EVAL_WINDOW_OVERLAP_RATIO}"
echo "RUN_DIR=${RUN_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

PROBE_ARGS=(
  --config "${CONFIG}"
  --out-dir "${RUN_DIR}/probe_eval"
  --device cuda
  --epochs 1
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --seed 0
  --probe-model "${PROBE_MODEL}"
  --scout-spatial-size "${SCOUT_SPATIAL_SIZE}"
  --max-train-batches 0
  --max-val-batches "${MAX_VAL_BATCHES}"
  --log-every-batches 25
  --fast-lowres-pipeline
  --probe-window-size "${DENSE_WINDOW_SIZE}"
  --eval-window-overlap-ratio "${EVAL_WINDOW_OVERLAP_RATIO}"
  --eval-include-all-windows
  --coverage-only
  --coverage-budget-fraction 0.5
  --boundary-radius "${BOUNDARY_RADIUS}"
  --probe-checkpoint "${PROBE_CHECKPOINT}"
  --sample-jsonl "${SAMPLES_JSONL}"
  --ann-file "${ANN_FILE}"
  --class-map "${CLASS_MAP}"
  --train-data-path "${TRAIN_DATA_PATH}"
  --val-data-path "${VAL_DATA_PATH}"
  --test-data-path "${TEST_DATA_PATH}"
  --val-subset-name "${VAL_SUBSET_NAME}"
)

case "${PROBE_MODEL}" in
  temporal-tcn)
    PROBE_ARGS+=(--tcn-variants "${TCN_VARIANT}")
    ;;
  matrix-zoo)
    if [[ -z "${MATRIX_MODEL_ID}" ]]; then
      echo "MATRIX_MODEL_ID is required for PROBE_MODEL=matrix-zoo." >&2
      exit 49
    fi
    PROBE_ARGS+=(--matrix-model-ids "${MATRIX_MODEL_ID}")
    ;;
  official-action-seg)
    if [[ -z "${OFFICIAL_ACTION_SEG_BACKEND}" ]]; then
      echo "OFFICIAL_ACTION_SEG_BACKEND is required for PROBE_MODEL=official-action-seg." >&2
      exit 50
    fi
    PROBE_ARGS+=(--official-action-seg-backends "${OFFICIAL_ACTION_SEG_BACKEND}")
    ;;
esac

python -u tools/bata/train_lowres_action_probe.py "${PROBE_ARGS[@]}"

python tools/bata/convert_lowres_probe_samples_to_value_transport_ledger.py \
  --input-jsonl "${SAMPLES_JSONL}" \
  --output-jsonl "${LEDGER_JSONL}" \
  --summary-json "${LEDGER_SUMMARY}" \
  --strategy "${SELECTION_STRATEGY}" \
  --target-len "${TARGET_LEN}" \
  --require-selected-count "${TARGET_LEN}" \
  --allow-short-valid-ratio-count \
  --fill-to-target-count \
  --deduplicate-sample-id \
  --deploy-selection-ledger \
  --route-variant "c3_lowres_${PROBE_MODEL}_${TCN_VARIANT}${MATRIX_MODEL_ID}${OFFICIAL_ACTION_SEG_BACKEND}_${SELECTION_STRATEGY}_dense${DENSE_WINDOW_SIZE}_to_${TARGET_LEN}_${EXPORT_SPLIT}"

echo "LEDGER_JSONL=${LEDGER_JSONL}"
echo "LEDGER_SUMMARY=${LEDGER_SUMMARY}"
echo "DONE $(date -Iseconds)"
