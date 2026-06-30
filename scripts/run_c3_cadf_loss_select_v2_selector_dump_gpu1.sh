#!/usr/bin/env bash
set -euo pipefail

if [[ "${CUDA_VISIBLE_DEVICES:-}" != "1" ]]; then
  echo "ERROR: selector dump is C3/CADF mainline diagnostic and must run on physical GPU1." >&2
  echo "Set CUDA_VISIBLE_DEVICES=1 explicitly before launching." >&2
  exit 2
fi

CONFIG="${CONFIG:-configs/adatad/thumos/c3_cadf_densitymesh_original_adatad_32px_loss_select_v2_fast_safe_formal.py}"
CHECKPOINT="${1:?usage: CUDA_VISIBLE_DEVICES=1 bash scripts/run_c3_cadf_loss_select_v2_selector_dump_gpu1.sh /path/to/checkpoint.pth [output_dir]}"
OUT_DIR="${2:-logs/c3_cadf_loss_select_v2_selector_dump}"

mkdir -p "${OUT_DIR}"

python tools/dump_c3_cadf_selector_diagnostics.py \
  "${CONFIG}" \
  "${CHECKPOINT}" \
  --split test \
  --device cuda:0 \
  --summary-json "${OUT_DIR}/selector_dump_summary.json" \
  --records-jsonl "${OUT_DIR}/selector_dump_records.jsonl" \
  --records-csv "${OUT_DIR}/selector_dump_records.csv" \
  --include-selected-indices
