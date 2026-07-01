#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/run01/sczc063/yuzibo/OpenTAD_C3TCNCoarseProbe_20260701}"
BASE_DIR="${BASE_DIR:-/data/run01/sczc063/yuzibo}"
CACHE_ROOT="${CACHE_ROOT:-${BASE_DIR}/model_zoo_cache/c3_coarse_classifier}"
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/logs/c3_coarse_classifier_model_zoo_download_$(date +%Y%m%d_%H%M%S_%z)}"
TIER="${TIER:-first_wave}"
INCLUDE_OPTIONAL="${INCLUDE_OPTIONAL:-0}"
DRY_RUN="${DRY_RUN:-0}"

cd "${PROJECT_DIR}"
mkdir -p "${OUT_DIR}" "${CACHE_ROOT}" "${BASE_DIR}/tmp/home" "${BASE_DIR}/tmp/xdg_cache" "${BASE_DIR}/tmp/xdg_config" "${BASE_DIR}/hf_cache"

if command -v module >/dev/null 2>&1; then
  module load cuda/11.8
  module load miniforge3/24.11
else
  echo "[C3_MODEL_ZOO_DOWNLOAD] module command unavailable; using existing conda env path."
fi
source "${BASE_DIR}/conda_envs/opentad/bin/activate"

export http_proxy="${http_proxy:-http://u-MtfrT7:vH5orjDV@10.244.6.36:3128}"
export https_proxy="${https_proxy:-${http_proxy}}"
export HTTP_PROXY="${HTTP_PROXY:-${http_proxy}}"
export HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy}}"
export HOME="${HOME:-${BASE_DIR}/tmp/home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${BASE_DIR}/tmp/xdg_cache}"
export XDG_CONFIG_HOME="${XDG_CONFIG_HOME:-${BASE_DIR}/tmp/xdg_config}"
export HF_HOME="${HF_HOME:-${BASE_DIR}/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""

echo "START $(date -Iseconds)"
echo "HOST $(hostname)"
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "CACHE_ROOT=${CACHE_ROOT}"
echo "OUT_DIR=${OUT_DIR}"
echo "TIER=${TIER}"
echo "INCLUDE_OPTIONAL=${INCLUDE_OPTIONAL}"
echo "DRY_RUN=${DRY_RUN}"
df -h "${BASE_DIR}" || true

args=(
  --tier "${TIER}"
  --cache-root "${CACHE_ROOT}"
  --output-json "${OUT_DIR}/c3_coarse_classifier_model_matrix_download_status.json"
  --download
)

if [[ "${INCLUDE_OPTIONAL}" == "1" ]]; then
  args+=(--include-optional)
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  args+=(--dry-run)
fi

python -u tools/bata/c3_coarse_classifier_model_matrix.py "${args[@]}" 2>&1 | tee "${OUT_DIR}/download_stdout.log"

echo "END $(date -Iseconds)"
df -h "${BASE_DIR}" || true
