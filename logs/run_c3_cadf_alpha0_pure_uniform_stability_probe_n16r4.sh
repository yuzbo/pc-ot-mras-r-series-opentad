#!/bin/bash
source /etc/profile
set -eo pipefail

module load cuda/11.8
module load miniforge3/24.11
source activate /data/home/sczc063/run/yuzibo/conda_envs/opentad

cd /data/home/sczc063/run/yuzibo/OpenTAD_C3CADFStageFix_Precheck_20260629

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=1
export C3_CADF_ROUTE_LABEL=C3_MAINLINE_OPTIMIZATION
export C3_CADF_DIAGNOSTIC=alpha0_pure_uniform_stability_probe

echo "C3_CADF_ALPHA0_STABILITY_PROBE_START $(date '+%F %T %z')"
echo "PWD=$PWD"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python -m torch.distributed.run \
  --master_addr=127.0.0.1 \
  --master_port=29841 \
  --nproc_per_node=1 \
  tools/train.py \
  configs/adatad/thumos/c3_cadf_densitymesh_original_adatad_32px_alpha0_pure_uniform_stability_probe.py \
  --id 0
echo "C3_CADF_ALPHA0_STABILITY_PROBE_END $(date '+%F %T %z')"
