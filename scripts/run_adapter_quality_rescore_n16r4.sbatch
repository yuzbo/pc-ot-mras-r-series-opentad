#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH -J adapter_quality
#SBATCH --chdir=/data/home/sczc063/run/yuzibo/OpenTAD_Back_check
#SBATCH -o logs/%x-%j.out

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$HOME/run/yuzibo/OpenTAD_Back_check}"
THUMOS_ROOT="${THUMOS_ROOT:-$HOME/run/yuzibo/thumos14}"
WORKSPACE="${WORKSPACE:-$HOME/run/yuzibo}"
CONDA_ENV="${CONDA_ENV:-$WORKSPACE/conda_envs/opentad}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$WORKSPACE/conda_pkgs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$WORKSPACE/pip_cache}"

cd "$ROOT_DIR"
mkdir -p logs

if ! command -v module >/dev/null 2>&1; then
  set +u
  source /etc/profile
  set -u
fi

module load cuda/11.8
module load miniforge3/24.11
source activate "$CONDA_ENV"

echo "ROOT_DIR=$ROOT_DIR"
echo "THUMOS_ROOT=$THUMOS_ROOT"
echo "CONDA_ENV=$CONDA_ENV"
nvidia-smi -L

export ROOT_DIR
export THUMOS_ROOT
export STORAGE_CHECK_PATH="${STORAGE_CHECK_PATH:-$WORKSPACE}"
export CONFIG="${CONFIG:-configs/adatad/thumos/input_random_fixed_50pct_adapter_quality_rescore_detached_n16r4.py}"
export NAME="${NAME:-input_random_fixed_50pct_adapter_quality_rescore_detached_n16r4}"
export PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
export TORCHRUN="${TORCHRUN:-$(command -v torchrun)}"
export RUN_ID="${RUN_ID:-0}"
if [[ -z "${BASE_PORT:-}" ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    export BASE_PORT="$((30000 + SLURM_JOB_ID % 1000))"
  else
    export BASE_PORT=30610
  fi
fi
export SKIP_GPU_WAIT=1
export KEEP_CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export SUPERVISE_INTERVAL_SECONDS="${SUPERVISE_INTERVAL_SECONDS:-1800}"

echo "CONFIG=$CONFIG"
echo "NAME=$NAME"
echo "RUN_ID=$RUN_ID"
echo "BASE_PORT=$BASE_PORT"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"

bash scripts/run_adapter_quality_rescore.sh "$ROOT_DIR"
