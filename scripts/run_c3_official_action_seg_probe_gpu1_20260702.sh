#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/run01/sczc063/yuzibo/OpenTAD_C3TCNCoarseProbe_20260701}"
OUT_DIR="${OUT_DIR:-/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/outputs/c3_official_action_seg_probe_gpu1_20260702}"
OFFICIAL_BACKENDS="${OFFICIAL_BACKENDS:-official_ms_tcn2 official_asformer official_fact official_video_mamba_asformer}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EPOCHS="${EPOCHS:-10}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-50}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-20}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
if [[ "${CUDA_VISIBLE_DEVICES}" != "1" ]]; then
  echo "Refusing to start: CUDA_VISIBLE_DEVICES must be exactly 1 for C3 mainline GPU1, got '${CUDA_VISIBLE_DEVICES}'." >&2
  exit 44
fi

if [[ -n "${SLURM_STEP_GPUS:-}" && "${SLURM_STEP_GPUS}" != "1" ]]; then
  echo "Refusing to start: SLURM_STEP_GPUS must be GPU1 when set, got '${SLURM_STEP_GPUS}'." >&2
  exit 45
fi

if [[ -n "${SLURM_JOB_GPUS:-}" && "${SLURM_JOB_GPUS}" != *"1"* ]]; then
  echo "Refusing to start: SLURM_JOB_GPUS does not include GPU1, got '${SLURM_JOB_GPUS}'." >&2
  exit 46
fi

cd "${PROJECT_DIR}"

module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1

echo "START $(date -Iseconds)"
echo "HOST $(hostname)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_STEP_ID=${SLURM_STEP_ID:-}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}"
echo "SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "OFFICIAL_BACKENDS=${OFFICIAL_BACKENDS}"
nvidia-smi || true

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(), "device_count", torch.cuda.device_count(), flush=True)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("Refusing to start: expected exactly one visible CUDA device after CUDA_VISIBLE_DEVICES=1.")
PY

python - <<'PY'
import os
import torch
from tools.bata import train_lowres_action_probe as probe

backends = list(os.environ["OFFICIAL_BACKENDS"].split())
print("OFFICIAL_BACKEND_AVAILABILITY", [(b, probe.official_action_seg_backend_available(b)) for b in backends], flush=True)
frames = torch.rand(1, 8, 3, 64, 64, device="cuda")
valid = torch.ones(1, 8, dtype=torch.bool, device="cuda")
for backend in backends:
    if not probe.official_action_seg_backend_available(backend):
        raise SystemExit(f"official backend unavailable before launch: {backend}")
    model = probe.C3OfficialActionSegmentationProbe(
        backend=backend,
        spatial_size=64,
        hidden_dim=16,
        num_layers=1,
    ).to("cuda").eval()
    with torch.no_grad():
        logits = model(frames, valid)
    if tuple(logits.shape) != (1, 8) or not torch.isfinite(logits).all():
        raise SystemExit(f"official backend tiny CUDA forward failed: {backend} shape={tuple(logits.shape)}")
    print("OFFICIAL_BACKEND_TINY_FORWARD_OK", backend, float(logits.mean().detach().cpu()), flush=True)
print("OFFICIAL_BACKEND_TINY_FORWARD_ALL_OK", flush=True)
PY

python -u tools/bata/train_lowres_action_probe.py \
  --config configs/adatad/thumos/pc_ot_mras_a_uniform_scaffold_small_actionness_strict_maxgap_c3_physical_grid_actionformer_n16r4.py \
  --out-dir "${OUT_DIR}" \
  --device cuda \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers 4 \
  --lr 1.0e-4 \
  --seed 0 \
  --probe-model official-action-seg \
  --scout-spatial-size 64 \
  --official-action-seg-backends ${OFFICIAL_BACKENDS} \
  --max-train-batches "${MAX_TRAIN_BATCHES}" \
  --max-val-batches "${MAX_VAL_BATCHES}" \
  --log-every-batches 10 \
  --fast-lowres-pipeline \
  --probe-window-size 384 \
  --ann-file /data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json \
  --class-map /data/run01/sczc063/yuzibo/thumos14/annotations/category_idx.txt \
  --train-data-path "/data/run01/sczc063/yuzibo/raw/Validation Data/validation" \
  --val-data-path "/data/run01/sczc063/yuzibo/raw/Test Data/TH14_test_set_mp4" \
  --test-data-path "/data/run01/sczc063/yuzibo/raw/Test Data/TH14_test_set_mp4" \
  --save-checkpoint
