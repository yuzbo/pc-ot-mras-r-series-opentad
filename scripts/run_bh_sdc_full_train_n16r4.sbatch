#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -J bh_sdc_full_candidate
#SBATCH -o logs/%x-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/run/yuzibo/OpenTAD_PCOTMRAS_R16_R18_R20_Clean_20260619_1730}"
CONFIG="${CONFIG:-configs/adatad/thumos/bh_sdc_boundary_hazard_sparse_dense_full_train_candidate_n16r4.py}"
PRECHECK_ONLY="${PRECHECK_ONLY:-1}"
ALLOW_BH_SDC_FULL_TRAIN="${ALLOW_BH_SDC_FULL_TRAIN:-0}"

cd "$REPO_ROOT"
mkdir -p logs

module load cuda/11.8
module load miniforge3/24.11
source activate "${CONDA_ENV:-$HOME/run/yuzibo/conda_envs/opentad}"

python tools/bata/validate_bh_sdc_full_train_gate.py "$CONFIG"

if [ "$PRECHECK_ONLY" != "0" ]; then
    echo "BH-SDC precheck-only mode complete. Full training remains locked."
    exit 0
fi

echo "Refusing BH-SDC full training: this launcher is locked to precheck-only until a new reviewed gate is implemented."
echo "ALLOW_BH_SDC_FULL_TRAIN=${ALLOW_BH_SDC_FULL_TRAIN} is intentionally ignored in this locked candidate."
exit 64
