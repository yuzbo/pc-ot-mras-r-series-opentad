#!/bin/bash
source /etc/profile
set -eo pipefail

if [ -z "${CADF_PROBE_CONFIG:-}" ] || [ -z "${CADF_PROBE_NAME:-}" ]; then
  echo "CADF_PROBE_CONFIG and CADF_PROBE_NAME are required" >&2
  exit 2
fi

module load cuda/11.8
module load miniforge3/24.11
source activate /data/home/sczc063/run/yuzibo/conda_envs/opentad

cd /data/home/sczc063/run/yuzibo/OpenTAD_C3CADFStageFix_Precheck_20260629

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export CUDA_VISIBLE_DEVICES="${CADF_VISIBLE_DEVICES:-1}"
export C3_CADF_ROUTE_LABEL=C3_MAINLINE_OPTIMIZATION
export C3_CADF_DIAGNOSTIC="${CADF_PROBE_NAME}"

echo "C3_CADF_ALPHA0_SINGLE_FACTOR_PROBE_START $(date '+%F %T %z')"
echo "PROBE_NAME=$CADF_PROBE_NAME"
echo "PROBE_CONFIG=$CADF_PROBE_CONFIG"
echo "PWD=$PWD"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python -m torch.distributed.run \
  --master_addr="${CADF_MASTER_ADDR:-127.0.0.1}" \
  --master_port="${CADF_MASTER_PORT:-29851}" \
  --nproc_per_node=1 \
  tools/train.py \
  "$CADF_PROBE_CONFIG" \
  --id 0
echo "C3_CADF_ALPHA0_SINGLE_FACTOR_PROBE_END $(date '+%F %T %z')"
