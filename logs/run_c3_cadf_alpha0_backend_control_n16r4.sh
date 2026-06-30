#!/bin/bash
set -eo pipefail

source /etc/profile
module load cuda/11.8
module load miniforge3/24.11
source activate /data/home/sczc063/run/yuzibo/conda_envs/opentad

cd /data/home/sczc063/run/yuzibo/OpenTAD_C3CADFStageFix_Precheck_20260629

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=1
export C3_CADF_ROUTE_LABEL=C3_MAINLINE_OPTIMIZATION
export C3_CADF_DIAGNOSTIC=alpha0_backend_control

python -m torch.distributed.run \
  --master_port=29831 \
  --nproc_per_node=1 \
  tools/train.py \
  configs/adatad/thumos/c3_cadf_densitymesh_original_adatad_32px_alpha0_backend_control.py \
  --id 0
