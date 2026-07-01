#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "[C3_ASFORMER_DELTA_FULLTRAIN][FAIL] $*" >&2
  exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PRECHECK_ONLY="${PRECHECK_ONLY:-1}"
ALLOW_C3_ASFORMER_DELTA_LEDGER_FULLTRAIN="${ALLOW_C3_ASFORMER_DELTA_LEDGER_FULLTRAIN:-0}"
CONFIG="${CONFIG:-configs/adatad/thumos/c3_official_asformer_delta_ledger_original_adatad_full_train.py}"
EXEC_CONFIG="${EXEC_CONFIG:-configs/adatad/thumos/c3_official_asformer_delta_ledger_original_adatad_full_train_exec.py}"
VALIDATOR="${VALIDATOR:-tools/bata/validate_c3_asformer_delta_ledger_full_train.py}"
RUN_TAG="${RUN_TAG:-c3_asformer_delta_ledger_adatad_full_train_gpu1_$(date +%Y%m%d_%H%M%S_%z)}"
RUN_ID="${RUN_ID:-0}"
SEED="${SEED:-0}"
MASTER_PORT="${MASTER_PORT:-30217}"

BASE="${BASE:-/data/run01/sczc063/yuzibo}"
export HOME="${HOME:-${BASE}/tmp/home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${BASE}/tmp/xdg_cache}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-${BASE}/tmp/xdg_config}"
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" logs

export YUZIBO_ROOT="${YUZIBO_ROOT:-${BASE}}"
export THUMOS14_ANNOTATION_PATH="${THUMOS14_ANNOTATION_PATH:-${BASE}/thumos14/annotations/thumos_14_anno.json}"
export THUMOS14_CLASS_MAP="${THUMOS14_CLASS_MAP:-${BASE}/thumos14/annotations/category_idx.txt}"
export THUMOS14_TRAIN_DATA_PATH="${THUMOS14_TRAIN_DATA_PATH:-${BASE}/raw/Validation Data/validation}"
export THUMOS14_TEST_DATA_PATH="${THUMOS14_TEST_DATA_PATH:-${BASE}/raw/Test Data/TH14_test_set_mp4}"

LEDGER_ROOT="${LEDGER_ROOT:-${BASE}/projects/c3_lowres_action_probe/ledger_exports/c3_official_asformer_delta_ledgers_20260702_052357_+0800}"
export C3_ASFORMER_DELTA_TRAIN_LEDGER_PATH="${C3_ASFORMER_DELTA_TRAIN_LEDGER_PATH:-${LEDGER_ROOT}/train/value_transport_ledger_delta_p_action_384.jsonl}"
export C3_ASFORMER_DELTA_VAL_LEDGER_PATH="${C3_ASFORMER_DELTA_VAL_LEDGER_PATH:-${LEDGER_ROOT}/val/value_transport_ledger_delta_p_action_384.jsonl}"
export C3_ASFORMER_DELTA_TEST_LEDGER_PATH="${C3_ASFORMER_DELTA_TEST_LEDGER_PATH:-${LEDGER_ROOT}/test/value_transport_ledger_delta_p_action_384.jsonl}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
if [[ "${CUDA_VISIBLE_DEVICES}" != "1" ]]; then
  fail "C3 mainline full train must use GPU1; got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

require_file() {
  local path="$1"
  [[ -f "${path}" ]] || fail "required file missing: ${path}"
}

require_file "${CONFIG}"
require_file "${EXEC_CONFIG}"
require_file "${VALIDATOR}"
require_file "${THUMOS14_ANNOTATION_PATH}"
require_file "${THUMOS14_CLASS_MAP}"
require_file "${C3_ASFORMER_DELTA_TRAIN_LEDGER_PATH}"
require_file "${C3_ASFORMER_DELTA_VAL_LEDGER_PATH}"
require_file "${C3_ASFORMER_DELTA_TEST_LEDGER_PATH}"

module load cuda/11.8 >/dev/null 2>&1 || true
module load miniforge3/24.11 >/dev/null 2>&1 || true
PYTHON="${PYTHON:-/data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python}"
[[ -x "${PYTHON}" ]] || fail "python not executable: ${PYTHON}"

echo "[C3_ASFORMER_DELTA_FULLTRAIN] repo=${REPO_ROOT}"
echo "[C3_ASFORMER_DELTA_FULLTRAIN] head=$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
echo "[C3_ASFORMER_DELTA_FULLTRAIN] gpu=${CUDA_VISIBLE_DEVICES}"
echo "[C3_ASFORMER_DELTA_FULLTRAIN] slurm_step_gpus=${SLURM_STEP_GPUS:-none} slurm_job_gpus=${SLURM_JOB_GPUS:-none}"
echo "[C3_ASFORMER_DELTA_FULLTRAIN] precheck_only=${PRECHECK_ONLY} unlock=${ALLOW_C3_ASFORMER_DELTA_LEDGER_FULLTRAIN}"

bash -n "${BASH_SOURCE[0]}"
"${PYTHON}" -m py_compile \
  tools/train.py \
  opentad/utils/train_schedule.py \
  opentad/datasets/transforms/end_to_end.py \
  "${VALIDATOR}"
"${PYTHON}" "${VALIDATOR}" --config "${CONFIG}" --require-ledger-files
"${PYTHON}" "${VALIDATOR}" --config "${EXEC_CONFIG}" --require-ledger-files --allow-launch-unlocked
"${PYTHON}" -m pytest \
  tests/test_pc_ot_mras_frontend_ledger.py \
  tests/test_c3_asformer_delta_ledger_full_train.py \
  -q

if [[ "${PRECHECK_ONLY}" == "1" ]]; then
  echo "[C3_ASFORMER_DELTA_FULLTRAIN] PRECHECK_ONLY complete"
  exit 0
fi

if [[ "${ALLOW_C3_ASFORMER_DELTA_LEDGER_FULLTRAIN}" != "1" ]]; then
  fail "ALLOW_C3_ASFORMER_DELTA_LEDGER_FULLTRAIN=1 is required for formal full train"
fi

RUN_DIR="${RUN_DIR:-logs/${RUN_TAG}}"
WORK_DIR="${WORK_DIR:-exps/thumos/adatad/c3_official_asformer_delta_ledger_original_adatad_full_train/${RUN_TAG}}"
mkdir -p "${RUN_DIR}" "${WORK_DIR}"

echo "[C3_ASFORMER_DELTA_FULLTRAIN] run_dir=${RUN_DIR}"
echo "[C3_ASFORMER_DELTA_FULLTRAIN] work_dir=${WORK_DIR}"

"${PYTHON}" -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port="${MASTER_PORT}" \
  tools/train.py \
  "${EXEC_CONFIG}" \
  --id "${RUN_ID}" \
  --seed "${SEED}" \
  --cfg-options "work_dir=${WORK_DIR}" \
  2>&1 | tee "${RUN_DIR}/srun.out"
