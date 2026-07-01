#!/usr/bin/env bash
set -euo pipefail

# C3 oracle-shell indirect full train launcher for N16R4 GPU1 inside protected parent hold 1118197.
# This script never cancels Slurm jobs and never changes the parent allocation policy.

REPO_DIR="${C3_ORACLE_SHELL_INDIRECT_REPO:-$HOME/run/yuzibo/OpenTAD_C3OracleShellIndirect_Worktree_20260701}"
CONFIG="configs/adatad/thumos/c3_oracle_shell_indirect_full_train.py"

if [[ "${C3_ORACLE_SHELL_INDIRECT_FULLTRAIN_UNLOCK:-}" != "CONFIRMED" ]]; then
  echo "Full train is locked. Set C3_ORACLE_SHELL_INDIRECT_FULLTRAIN_UNLOCK=CONFIRMED only in the main deployment process." >&2
  exit 2
fi

if [[ -z "${C3_COARSE_SCORE_CACHE_DIR:-}" ]]; then
  echo "C3_COARSE_SCORE_CACHE_DIR must point to manifest.json + per-video npz cache" >&2
  exit 2
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "1" ]]; then
  echo "GPU1 fail-closed: CUDA_VISIBLE_DEVICES must be 1, got ${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES=1

cd "$REPO_DIR"
python tools/validate_c3_oracle_shell_indirect.py "$CONFIG"
export LOCAL_RANK="${LOCAL_RANK:-0}"
export RANK="${RANK:-0}"
export WORLD_SIZE="${WORLD_SIZE:-1}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-${C3_ORACLE_SHELL_INDIRECT_MASTER_PORT:-30042}}"
python tools/train.py "$CONFIG" --id 0
