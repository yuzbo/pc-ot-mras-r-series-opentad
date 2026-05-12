#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH -J setup_opentad
#SBATCH --chdir=/data/home/sczc063/run/yuzibo/OpenTAD_Back_check
#SBATCH -o logs/setup-opentad-%j.out

set -euo pipefail

cd /data/home/sczc063/run/yuzibo/OpenTAD_Back_check
export SETUP_TORCH_ONLY="${SETUP_TORCH_ONLY:-1}"
export OPENTAD_MAMBA_OFFLINE="${OPENTAD_MAMBA_OFFLINE:-1}"
bash scripts/setup_n16r4_opentad_env.sh
