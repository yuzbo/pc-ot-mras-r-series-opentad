#!/usr/bin/env bash
set -euo pipefail

WT=${WT:-/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_GuardDiag_20260701_e6de60e9_bundle}
PY=${PY:-/data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python}
CONFIG=${CONFIG:-configs/adatad/thumos/input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3_evaldiag.py}
RUN_ID=${RUN_ID:-0}
LOGDIR=${RBA_RBR_GUARD_LOGDIR:?RBA_RBR_GUARD_LOGDIR must be set by watcher}

cd "$WT"
mkdir -p "$LOGDIR"

module load cuda/11.8 >/dev/null 2>&1 || true
module load miniforge3/24.11 >/dev/null 2>&1 || true

export CUDA_VISIBLE_DEVICES=${RBA_RBR_HOLD_CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export PYTHONNOUSERSITE=1
export RBA_RBR_ROUTE_LABEL=DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3
export RBA_RBR_GRID_AUDIT=1
export RBA_RBR_GRID_AUDIT_PATH="$LOGDIR/rba_rbr_grid_audit.jsonl"
export LOGDIR

echo "RBA_GUARD_HOLD_G0_START $(date '+%F %T %z')"
echo "HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES_INITIAL=${CUDA_VISIBLE_DEVICES:-unset}"
echo "HEAD=$(git rev-parse --short=8 HEAD)"
echo "CONFIG=$CONFIG"
echo "LOGDIR=$LOGDIR"
echo "SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-unset}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
if [ "${CUDA_VISIBLE_DEVICES:-}" != "0" ]; then
  echo "ERROR_RBA_RBR_HOLD_GPU_BOUNDARY expected CUDA_VISIBLE_DEVICES=0 got ${CUDA_VISIBLE_DEVICES:-unset}" >&2
  exit 8
fi
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader || true

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
            value = row.get(key)
            if value is not None:
                out.append(float(value))
        except Exception:
            pass
    return out


def stat_line(key):
    xs = sorted(vals(key))
    if not xs:
        return f"{key}=missing"

    def pct(p):
        idx = min(len(xs) - 1, max(0, int(round((len(xs) - 1) * p))))
        return xs[idx]

    return (
        f"{key}_n={len(xs)} {key}_min={xs[0]:.4f} {key}_p05={pct(0.05):.4f} "
        f"{key}_p50={pct(0.50):.4f} {key}_p95={pct(0.95):.4f} "
        f"{key}_max={xs[-1]:.4f} {key}_avg={sum(xs) / len(xs):.4f}"
    )


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

echo "RBA_GUARD_HOLD_G0_DONE $(date '+%F %T %z')"
