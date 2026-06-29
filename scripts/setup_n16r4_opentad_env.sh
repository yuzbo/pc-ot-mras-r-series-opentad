#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-$HOME/run/yuzibo}"
ROOT_DIR="${ROOT_DIR:-$WORKSPACE/OpenTAD_Back_check}"
ENV_PREFIX="${CONDA_ENV:-$WORKSPACE/conda_envs/opentad}"
SENTINEL="$ENV_PREFIX/.opentad_n16r4_env_ok"
TORCH_SENTINEL="$ENV_PREFIX/.opentad_n16r4_torch_ok"
SETUP_TORCH_ONLY="${SETUP_TORCH_ONLY:-0}"
OPENTAD_MAMBA_OFFLINE="${OPENTAD_MAMBA_OFFLINE:-0}"

case "$WORKSPACE" in
  "$HOME"/run/yuzibo) ;;
  *)
    echo "refusing to use WORKSPACE outside ~/run/yuzibo: $WORKSPACE" >&2
    exit 2
    ;;
esac

mkdir -p \
  "$WORKSPACE/conda_envs" \
  "$WORKSPACE/conda_pkgs" \
  "$WORKSPACE/pip_cache" \
  "$WORKSPACE/.cache/mim" \
  "$WORKSPACE/cache" \
  "$ROOT_DIR/logs"

export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$WORKSPACE/conda_pkgs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$WORKSPACE/pip_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$WORKSPACE/cache}"
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"

if ! command -v module >/dev/null 2>&1; then
  set +u
  source /etc/profile
  set -u
fi

module load cuda/11.8
module load miniforge3/24.11

if [[ -f "$SENTINEL" ]]; then
  source activate "$ENV_PREFIX"
  python - <<'PY'
import torch
import mmcv
import mmengine
import mmaction
import decord
print("python_env_ok")
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("mmcv", mmcv.__version__)
print("mmengine", mmengine.__version__)
print("mmaction", mmaction.__version__)
print("decord", decord.__version__)
PY
  exit 0
fi

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  mamba create -y -p "$ENV_PREFIX" \
    python=3.10 \
    pytorch=2.0.1 \
    torchvision=0.15.2 \
    pytorch-cuda=11.8 \
    -c pytorch \
    -c nvidia
fi

source activate "$ENV_PREFIX"

mamba_extra_args=()
if [[ "$OPENTAD_MAMBA_OFFLINE" == "1" ]]; then
  mamba_extra_args+=(--offline)
fi

mamba install "${mamba_extra_args[@]}" -y -p "$ENV_PREFIX" \
  pytorch=2.0.1 \
  torchvision=0.15.2 \
  pytorch-cuda=11.8 \
  "mkl<2024" \
  -c pytorch \
  -c nvidia

python - <<'PY'
import torch
import torchvision

print("torch_env_ok")
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("torchvision", torchvision.__version__)
assert torch.version.cuda == "11.8", torch.version.cuda
PY
touch "$TORCH_SENTINEL"

if [[ "$SETUP_TORCH_ONLY" == "1" ]]; then
  exit 0
fi

python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy==1.23.5 ninja
python -m pip install openmim mmengine==0.10.3
HOME="$WORKSPACE" mim install mmcv==2.0.1
HOME="$WORKSPACE" mim install mmaction2==1.1.0
python -m pip install --no-build-isolation -r "$ROOT_DIR/requirements.txt"
python -m pip install decord==0.6.0 timm==0.6.13

python - <<'PY'
import torch
import torchvision
import mmcv
import mmengine
import mmaction
import decord

print("python_env_ok")
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("cuda_available", torch.cuda.is_available())
assert torch.version.cuda == "11.8", torch.version.cuda
print("mmcv", mmcv.__version__)
print("mmengine", mmengine.__version__)
print("mmaction", mmaction.__version__)
print("decord", decord.__version__)
PY
touch "$SENTINEL"
