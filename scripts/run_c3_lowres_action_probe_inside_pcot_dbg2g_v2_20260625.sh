#!/bin/bash
set -euo pipefail

PROJECT_DIR="/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/code"
OUT_DIR="/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/outputs/c3_lowres_action_probe_20260625_pcot_dbg2g_v2_progress"

cd "${PROJECT_DIR}"

module load cuda/11.8
module load miniforge3/24.11
source /data/home/sczc063/run/yuzibo/conda_envs/opentad/bin/activate

export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

echo "START $(date -Iseconds)"
echo "HOST $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_STEP_ID=${SLURM_STEP_ID:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi || true
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(), "device_count", torch.cuda.device_count(), flush=True)
PY

python -u tools/bata/train_lowres_action_probe.py \
  --config configs/adatad/thumos/pc_ot_mras_a_uniform_scaffold_small_actionness_strict_maxgap_c3_physical_grid_actionformer_n16r4.py \
  --out-dir "${OUT_DIR}" \
  --device cuda \
  --epochs 10 \
  --batch-size 4 \
  --num-workers 4 \
  --lr 1.0e-4 \
  --seed 0 \
  --probe-model mobilenetv3 \
  --mobilenet-sizes 32 64 \
  --mobilenet-weights-path /data/home/sczc063/run/yuzibo/projects/c3_lowres_action_probe/logs/mobilenet_v3_small-047dcff4.pth \
  --no-freeze-backbone \
  --max-train-batches 50 \
  --max-val-batches 20 \
  --log-every-batches 10 \
  --fast-lowres-pipeline \
  --probe-window-size 384 \
  --ann-file /data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json \
  --class-map /data/run01/sczc063/yuzibo/thumos14/annotations/category_idx.txt \
  --train-data-path "/data/run01/sczc063/yuzibo/raw/Validation Data/validation" \
  --val-data-path "/data/run01/sczc063/yuzibo/raw/Test Data/TH14_test_set_mp4" \
  --test-data-path "/data/run01/sczc063/yuzibo/raw/Test Data/TH14_test_set_mp4" \
  --save-checkpoint
