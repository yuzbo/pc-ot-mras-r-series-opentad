#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -J p2_loc_attr
#SBATCH -o logs/%x-%j.out

set -euo pipefail

REPO="${REPO:-$HOME/run/yuzibo/OpenTAD_PCOTMRAS_CPU_PRERUN_20260620_1418}"
cd "$REPO"

module load cuda/11.8
module load miniforge3/24.11
source activate "$HOME/run/yuzibo/conda_envs/opentad"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

HEAD_SHORT="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${P2LOC_OUT_ROOT:-logs/pcotmras_p2_localization_attribution_${HEAD_SHORT}_${STAMP}}"
LIMIT_BATCHES="${P2LOC_LIMIT_BATCHES:-8}"
TOPK_PER_SAMPLE="${P2LOC_TOPK_PER_SAMPLE:-1000}"
TOPK_LIST="${P2LOC_TOPK_LIST:-100,300,1000}"
IOU_THRESHOLDS="${P2LOC_IOU_THRESHOLDS:-0.3,0.5,0.7}"
SCORE_BINS="${P2LOC_SCORE_BINS:-10}"
DEVICE="${P2LOC_DEVICE:-cuda}"

R17_CKPT="${R17_CKPT:-$REPO/logs/ctf_bdi_pc_ot_mras_r17_formal_6d15388_gatefix_20260621_185815/train_workdir/gpu1_id0/gpu1_id0/checkpoint/epoch_59.pth}"
R18_CKPT="${R18_CKPT:-$REPO/logs/ctf_bdi_pc_ot_mras_r18_aux_formal_6d15388_gatefix_20260621_185815/train_workdir/gpu1_id0/gpu1_id0/checkpoint/epoch_59.pth}"

mkdir -p "$OUT_ROOT"

cat > "$OUT_ROOT/manifest.json" <<MANIFEST
{
  "diagnostic": "p2_localization_attribution",
  "diagnostic_only": true,
  "uses_validation_gt": true,
  "metric_claim_allowed": false,
  "paper_claim_allowed": false,
  "repo": "$REPO",
  "git_head": "$HEAD_SHORT",
  "limit_batches": $LIMIT_BATCHES,
  "topk_per_sample": $TOPK_PER_SAMPLE,
  "topk_list": "$TOPK_LIST",
  "iou_thresholds": "$IOU_THRESHOLDS",
  "score_bins": $SCORE_BINS
}
MANIFEST

CASES=(
  "R17_C0"
  "R17_C2"
  "R17_C3"
  "R18_C0"
  "R18_C2"
  "R18_C3"
)
CONFIGS=(
  "configs/adatad/thumos/ctf_bdi_pc_ot_mras_r17_post_train_eval_local_lowmem_candidate.py"
  "configs/adatad/thumos/ctf_bdi_pc_ot_mras_r17_c2_selected_times_eval_local_lowmem_candidate.py"
  "configs/adatad/thumos/ctf_bdi_pc_ot_mras_r17_c3_no_gate_scale_eval_local_lowmem_candidate.py"
  "configs/adatad/thumos/ctf_bdi_pc_ot_mras_r18_post_train_eval_local_lowmem_candidate.py"
  "configs/adatad/thumos/ctf_bdi_pc_ot_mras_r18_c2_selected_times_eval_local_lowmem_candidate.py"
  "configs/adatad/thumos/ctf_bdi_pc_ot_mras_r18_c3_no_gate_scale_eval_local_lowmem_candidate.py"
)
CKPTS=(
  "$R17_CKPT"
  "$R17_CKPT"
  "$R17_CKPT"
  "$R18_CKPT"
  "$R18_CKPT"
  "$R18_CKPT"
)

for index in "${!CASES[@]}"; do
  CASE="${CASES[$index]}"
  CONFIG="${CONFIGS[$index]}"
  CKPT="${CKPTS[$index]}"
  CASE_DIR="$OUT_ROOT/$CASE"
  mkdir -p "$CASE_DIR"
  echo "=== $CASE ==="
  echo "config=$CONFIG"
  echo "checkpoint=$CKPT"
  test -f "$CONFIG"
  test -f "$CKPT"
  python tools/bata/dump_native_irregular_p2_proposal_factors.py \
    --config "$CONFIG" \
    --checkpoint "$CKPT" \
    --output-jsonl "$CASE_DIR/proposal_factors.jsonl" \
    --summary-json "$CASE_DIR/proposal_factor_summary.json" \
    --split val \
    --limit-batches "$LIMIT_BATCHES" \
    --device "$DEVICE" \
    --use-ema auto \
    --amp \
    --topk-per-sample "$TOPK_PER_SAMPLE"
  python tools/bata/analyze_p2_proposal_localization.py \
    --proposal-jsonl "$CASE_DIR/proposal_factors.jsonl" \
    --output-dir "$CASE_DIR/localization_attribution" \
    --config "$CONFIG" \
    --dataset-split-key val \
    --split validation \
    --topk "$TOPK_LIST" \
    --iou-thresholds "$IOU_THRESHOLDS" \
    --score-bins "$SCORE_BINS" \
    --group-by sample
done

python - "$OUT_ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
case_summaries = {}
for summary_path in sorted(root.glob("*/localization_attribution/summary.json")):
    case = summary_path.parents[1].name
    case_summaries[case] = json.loads(summary_path.read_text(encoding="utf-8"))
bundle = {
    "diagnostic": "p2_localization_attribution_bundle",
    "diagnostic_only": True,
    "uses_validation_gt": True,
    "metric_claim_allowed": False,
    "paper_claim_allowed": False,
    "cases": case_summaries,
}
(root / "bundle_summary.json").write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(bundle, sort_keys=True))
PY

echo "P2 localization attribution diagnostics complete: $OUT_ROOT"
