#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "python or python3 is required for PRECHECK_ONLY" >&2
    exit 127
  fi
fi
PRECHECK_ONLY="${PRECHECK_ONLY:-1}"
CONFIG="configs/adatad/thumos/input_random_fixed_50pct_c3_physical_grid_actionformer_precheck.py"

if [[ "$PRECHECK_ONLY" != "1" ]]; then
  echo "C3 physical-grid ActionFormer candidate is fail-closed: set PRECHECK_ONLY=1 only." >&2
  exit 3
fi

cd "$ROOT_DIR"

"$PYTHON_BIN" -m py_compile \
  opentad/models/dense_heads/anchor_free_head.py \
  opentad/models/detectors/actionformer.py \
  opentad/datasets/transforms/formatting.py \
  tools/bata/validate_c3_physical_grid_actionformer_precheck.py

"$PYTHON_BIN" -m pytest tests/test_c3_physical_grid_actionformer_candidate.py -q
"$PYTHON_BIN" -m pytest tests/test_c3_physical_grid_round_trip.py -q

"$PYTHON_BIN" tools/bata/validate_c3_physical_grid_actionformer_precheck.py --config "$CONFIG"

echo "C3 physical-grid ActionFormer PRECHECK_ONLY complete; no train, no evaluator shortcut, no remote sync, no Slurm."
