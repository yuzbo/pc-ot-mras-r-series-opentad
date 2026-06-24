#!/bin/bash
# BH-SDC locked static precheck launcher.
# Route: DIVERGENT_INNOVATION_BH_SDC_DO_NOT_MERGE_WITH_C3
#
# This file intentionally does not request GPUs and does not call training or
# evaluation entrypoints. Full training remains locked until the second Pro review,
# subagent final review, and an explicit launch gate produce non-empty
# allowed_entrypoints in a future reviewed config.
# Current allowed_entrypoints contract: empty tuple/list only.
#SBATCH -J bh_sdc_static_precheck
#SBATCH -o logs/%x-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
CONFIG="${CONFIG:-configs/adatad/thumos/bh_sdc_boundary_hazard_sparse_dense_full_train_candidate_n16r4.py}"
PRECHECK_ONLY="${PRECHECK_ONLY:-1}"

cd "$REPO_ROOT"

if [ "$PRECHECK_ONLY" != "1" ]; then
    echo "Refusing BH-SDC full training: PRECHECK_ONLY must remain 1 in this locked candidate."
    echo "allowed_entrypoints is intentionally empty until Pro + subagent + explicit gate approval."
    exit 64
fi

python tools/bata/validate_bh_sdc_full_train_gate.py "$CONFIG" --json

echo "BH-SDC static precheck complete. Full training, remote sync, Slurm GPU use, training, and evaluation remain locked."
exit 0
