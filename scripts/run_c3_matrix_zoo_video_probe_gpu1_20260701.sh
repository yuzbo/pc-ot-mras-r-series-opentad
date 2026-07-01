#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/run01/sczc063/yuzibo/OpenTAD_C3TCNCoarseProbe_20260701}"
OUT_DIR="${OUT_DIR:-/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/outputs/c3_matrix_zoo_video_probe_gpu1_20260701}"
MODEL_IDS="${MODEL_IDS:-torchvision_r3d_18 torchvision_r2plus1d_18 torchvision_mc3_18 torchvision_s3d pytorchvideo_x3d_xs pytorchvideo_x3d_s pytorchvideo_c2d_r50 pytorchvideo_i3d_r50}"
SPATIAL_SIZE="${SPATIAL_SIZE:-112}"
EPOCHS="${EPOCHS:-6}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-40}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-20}"
VIDEO_CLIP_LEN="${VIDEO_CLIP_LEN:-16}"
VIDEO_ANCHOR_STRIDE="${VIDEO_ANCHOR_STRIDE:-8}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
if [[ "${CUDA_VISIBLE_DEVICES}" != "1" ]]; then
  echo "Refusing to start: CUDA_VISIBLE_DEVICES must be exactly 1 for C3 mainline GPU1, got '${CUDA_VISIBLE_DEVICES}'." >&2
  exit 44
fi

cd "${PROJECT_DIR}"

module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1
export TORCH_HOME="${TORCH_HOME:-/data/run01/sczc063/yuzibo/model_zoo_cache/c3_coarse_classifier/torch}"
export HF_HOME="${HF_HOME:-/data/run01/sczc063/yuzibo/hf_cache}"

echo "START $(date -Iseconds)"
echo "HOST $(hostname)"
echo "OUT_DIR=${OUT_DIR}"
echo "MODEL_IDS=${MODEL_IDS}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi || true

python -u tools/bata/train_lowres_action_probe.py \
  --config configs/adatad/thumos/pc_ot_mras_a_uniform_scaffold_small_actionness_strict_maxgap_c3_physical_grid_actionformer_n16r4.py \
  --out-dir "${OUT_DIR}" \
  --device cuda \
  --epochs "${EPOCHS}" \
  --batch-size 1 \
  --num-workers 4 \
  --lr 5.0e-5 \
  --seed 0 \
  --probe-model matrix-zoo \
  --scout-spatial-size "${SPATIAL_SIZE}" \
  --matrix-model-ids ${MODEL_IDS} \
  --matrix-video-clip-len "${VIDEO_CLIP_LEN}" \
  --matrix-video-anchor-stride "${VIDEO_ANCHOR_STRIDE}" \
  --matrix-freeze-backbone \
  --matrix-continue-on-model-error \
  --max-train-batches "${MAX_TRAIN_BATCHES}" \
  --max-val-batches "${MAX_VAL_BATCHES}" \
  --log-every-batches 5 \
  --fast-lowres-pipeline \
  --probe-window-size 256 \
  --ann-file /data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json \
  --class-map /data/run01/sczc063/yuzibo/thumos14/annotations/category_idx.txt \
  --train-data-path "/data/run01/sczc063/yuzibo/raw/Validation Data/validation" \
  --val-data-path "/data/run01/sczc063/yuzibo/raw/Test Data/TH14_test_set_mp4" \
  --test-data-path "/data/run01/sczc063/yuzibo/raw/Test Data/TH14_test_set_mp4" \
  --save-checkpoint

echo "END $(date -Iseconds)"
