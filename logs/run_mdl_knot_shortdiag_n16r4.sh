#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$ROOT_DIR"

SHORTDIAG_CONFIG="configs/adatad/thumos/input_mdl_knot_dynamic_adapter_irregular_headv3_shortdiag.py"
BASE_MDL_CONFIG="configs/adatad/thumos/input_mdl_knot_dynamic_adapter_irregular_headv3.py"
SHORTDIAG_VALIDATOR="tools/mdl_knot/validate_mdl_knot_shortdiag.py"
BASE_GATE="tools/mdl_knot/validate_mdl_knot_launch_gate.py"
TRAIN_LOG="${TRAIN_LOG:-logs/mdl_knot_shortdiag_train_${SLURM_JOB_ID:-manual}.log}"

echo "MDL-Knot SHORT_DIAGNOSTIC_ONLY gate"
echo "Route: DIVERGENT_INNOVATION_MDL_KNOT_DO_NOT_MERGE_WITH_C3"
echo "Scope: already allocated child GPU context only; no scheduler submission or parent-hold release."

python -m py_compile \
  "$SHORTDIAG_CONFIG" \
  "$SHORTDIAG_VALIDATOR" \
  tests/test_mdl_knot_shortdiag.py

python -m pytest \
  tests/test_mdl_knot_shortdiag.py \
  tests/test_mdl_knot_core.py \
  tests/test_mdl_knot_tools_and_integration.py \
  -q

python "$BASE_GATE" --config "$BASE_MDL_CONFIG"
python "$SHORTDIAG_VALIDATOR" --config "$SHORTDIAG_CONFIG"

if [[ "${RUN_SHORTDIAG_TRAIN:-0}" != "1" ]]; then
  echo "Gate-only mode complete. Set RUN_SHORTDIAG_TRAIN=1 inside an existing GPU allocation to run the one-epoch diagnostic."
  exit 0
fi

if [[ -z "${SLURM_JOB_ID:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "LOCKED: RUN_SHORTDIAG_TRAIN=1 requires an already allocated child GPU context." >&2
  exit 2
fi

mkdir -p "$(dirname "$TRAIN_LOG")"
echo "Starting one-epoch diagnostic train; output will be tee'd to $TRAIN_LOG"
python -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE:-1}" \
  tools/train.py "$SHORTDIAG_CONFIG" --id "${RUN_ID:-0}" 2>&1 | tee "$TRAIN_LOG"

python "$SHORTDIAG_VALIDATOR" --config "$SHORTDIAG_CONFIG" --train-log "$TRAIN_LOG"
echo "SHORT_DIAGNOSTIC_ONLY complete; no evaluation, no checkpoint claim, no mAP/runtime/deploy/paper/sparse-compute claim."
