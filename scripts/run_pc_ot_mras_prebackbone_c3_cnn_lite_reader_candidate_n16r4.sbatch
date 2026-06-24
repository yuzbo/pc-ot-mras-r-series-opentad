#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH -J pcot_c3cnn_gate
#SBATCH -o logs/%x-%j.out

set -eo pipefail
set -u

source /etc/profile >/dev/null 2>&1 || true
module load cuda/11.8
module load miniforge3/24.11

YUZIBO_ROOT="${YUZIBO_ROOT:-$HOME/run/yuzibo}"
source activate "$YUZIBO_ROOT/conda_envs/opentad"

CLEAN_REPO_NAME="${OPENTAD_PCOTMRAS_CLEAN_REPO_NAME:-OpenTAD_PCOTMRAS_R16_R18_R20_Clean_20260619_1730}"
REPO="${OPENTAD_PCOTMRAS_CLEAN_ROOT:-$YUZIBO_ROOT/$CLEAN_REPO_NAME}"
CONFIG="configs/adatad/thumos/pc_ot_mras_prebackbone_c3_cnn_lite_reader_candidate_n16r4.py"
VALIDATOR="tools/bata/validate_pc_ot_mras_prebackbone_c3_reader_improvement_candidates.py"

if [ "${ALLOW_PC_OT_MRAS_C3_READER_TRAIN:-0}" != "0" ]; then
  echo "[pcot-c3-cnn-lite-gate] ERROR this candidate launcher is precheck-only and must not launch training" >&2
  exit 2
fi
if [ "${ALLOW_PC_OT_MRAS_C3_READER_TOOLS_TEST:-0}" != "0" ]; then
  echo "[pcot-c3-cnn-lite-gate] ERROR tools/test.py is forbidden for this candidate gate" >&2
  exit 2
fi

cd "$REPO"
mkdir -p logs
python -m py_compile "$CONFIG" "$VALIDATOR"
python "$VALIDATOR" --candidate cnn_lite --require-registered-readers
echo "[pcot-c3-cnn-lite-gate] PC_OT_MRAS_C3_CNN_LITE_READER_CANDIDATE_GATE_PASS_NO_TRAIN"
