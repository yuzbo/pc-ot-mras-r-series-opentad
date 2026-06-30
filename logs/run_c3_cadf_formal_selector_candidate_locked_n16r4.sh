#!/bin/bash
source /etc/profile
set -eo pipefail

if [ "${CADF_COMBO_OLD_WINDOW_PASS:-}" != "CONFIRMED" ]; then
  echo "LOCKED: set CADF_COMBO_OLD_WINDOW_PASS=CONFIRMED only after the ST+actionness combo gate has passed the old NaN window." >&2
  exit 2
fi

if [ -z "${CADF_COMBO_OLD_WINDOW_EVIDENCE:-}" ] || [ ! -f "${CADF_COMBO_OLD_WINDOW_EVIDENCE}" ]; then
  echo "LOCKED: CADF_COMBO_OLD_WINDOW_EVIDENCE must point to the recorded combo pass evidence file." >&2
  exit 2
fi

module load cuda/11.8
module load miniforge3/24.11
source activate /data/home/sczc063/run/yuzibo/conda_envs/opentad

cd /data/home/sczc063/run/yuzibo/OpenTAD_C3CADFStageFix_Precheck_20260629

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export CUDA_VISIBLE_DEVICES="${CADF_VISIBLE_DEVICES:-1}"
export C3_CADF_ROUTE_LABEL=C3_MAINLINE_OPTIMIZATION
export C3_CADF_DIAGNOSTIC=formal_selector_candidate_locked

CONFIG=configs/adatad/thumos/c3_cadf_densitymesh_original_adatad_32px_formal_selector_candidate_locked.py

echo "C3_CADF_FORMAL_SELECTOR_CANDIDATE_START $(date '+%F %T %z')"
echo "CONFIG=$CONFIG"
echo "COMBO_EVIDENCE=$CADF_COMBO_OLD_WINDOW_EVIDENCE"
echo "PWD=$PWD"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python -m torch.distributed.run \
  --master_addr="${CADF_MASTER_ADDR:-127.0.0.1}" \
  --master_port="${CADF_MASTER_PORT:-29862}" \
  --nproc_per_node=1 \
  tools/train.py \
  "$CONFIG" \
  --id 0
echo "C3_CADF_FORMAL_SELECTOR_CANDIDATE_END $(date '+%F %T %z')"
