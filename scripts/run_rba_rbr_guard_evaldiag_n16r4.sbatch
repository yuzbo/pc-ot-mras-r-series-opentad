#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH -J rba_guarddiag
#SBATCH --exclude=g0030

set -euo pipefail

DIR=${DIR:-/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_GuardDiag_20260701_e6de60e9_bundle}
PY=${PY:-/data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python}
CONFIG=${CONFIG:-configs/adatad/thumos/input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3_evaldiag.py}
RUN_ID=${RUN_ID:-0}
RUN_TAG=${RUN_TAG:-rba_rbr_guard_evaldiag_torchrun_${SLURM_JOB_ID:-manual}}
LOGDIR=${LOGDIR:-$DIR/logs/$RUN_TAG}

cd "$DIR"
mkdir -p "$LOGDIR"

echo "RBA_GUARD_DIAG_START $(date '+%F %T %z')"
echo "HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES_INITIAL=${CUDA_VISIBLE_DEVICES:-unset}"
echo "HEAD=$(git rev-parse --short=8 HEAD)"
echo "CONFIG=$CONFIG"
echo "LOGDIR=$LOGDIR"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export RBA_RBR_GRID_AUDIT=1
export RBA_RBR_GRID_AUDIT_PATH="$LOGDIR/rba_rbr_grid_audit.jsonl"
export LOGDIR

"$PY" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=1 \
  tools/train.py "$CONFIG" --id "$RUN_ID" --disable_deterministic \
  2>&1 | tee "$LOGDIR/train.log"

"$PY" - <<'PY'
import json
import os
from collections import Counter
from pathlib import Path

logdir = Path(os.environ["LOGDIR"])
path = logdir / "rba_rbr_grid_audit.jsonl"
summary_path = logdir / "grid_audit_summary.txt"
rows = []
bad = 0
if path.exists():
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            bad += 1

labels = Counter(str(row.get("route_label", "missing")) for row in rows)
statuses = Counter(str(row.get("status", "missing")) for row in rows)
guards = Counter(str(row.get("guard_reason", "missing")) for row in rows)


def vals(key):
    out = []
    for row in rows:
        try:
            out.append(float(row.get(key)))
        except Exception:
            pass
    return out


def stat_line(key):
    xs = vals(key)
    if not xs:
        return f"{key}=missing"
    return f"{key}_min={min(xs):.4f} {key}_max={max(xs):.4f} {key}_avg={sum(xs) / len(xs):.4f}"


lines = [
    f"rows={len(rows)} parse_bad={bad}",
    f"route_labels={dict(labels)}",
    f"statuses={dict(statuses)}",
    f"guard_reasons={dict(guards)}",
    stat_line("raw_valid_k"),
    stat_line("mask_true_count"),
    stat_line("meta_detector_feature_position_count"),
    stat_line("guard_addition_count"),
    stat_line("selected_max_gap_before_guard"),
    stat_line("selected_max_gap_after_guard"),
    stat_line("max_detector_gap_before_guard"),
    stat_line("max_detector_gap_after_guard"),
]
summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
PY

echo "RBA_GUARD_DIAG_DONE $(date '+%F %T %z')"
