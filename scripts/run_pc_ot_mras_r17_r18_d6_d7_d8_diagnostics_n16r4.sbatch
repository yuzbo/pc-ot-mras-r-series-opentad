#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -J r17r18_d678
#SBATCH -o logs/%x-%j.out

set -euo pipefail

YUZIBO_ROOT="${YUZIBO_ROOT:-$HOME/run/yuzibo}"
REPO="${OPENTAD_PCOTMRAS_CLEAN_ROOT:-$YUZIBO_ROOT/OpenTAD_PCOTMRAS_R16_R18_R20_Clean_20260619_1730}"
CPU_PRERUN_REPO="${CPU_PRERUN_REPO:-$YUZIBO_ROOT/OpenTAD_PCOTMRAS_CPU_PRERUN_20260620_1418}"
cd "$REPO"

if command -v module >/dev/null 2>&1; then
  module load cuda/11.8
  module load miniforge3/24.11
fi

PYTHON="${PYTHON:-$YUZIBO_ROOT/conda_envs/opentad/bin/python}"
if [ ! -x "$PYTHON" ]; then
  if command -v conda >/dev/null 2>&1; then
    source activate "$YUZIBO_ROOT/conda_envs/opentad"
    PYTHON="python"
  else
    echo "No executable Python found at $PYTHON and conda is unavailable" >&2
    exit 2
  fi
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

PRECHECK_ONLY="${PRECHECK_ONLY:-1}"
ALLOW_DIAG="${ALLOW_PCOTMRAS_D6_D7_D8_DIAG:-0}"
DEVICE="${DEVICE:-cuda}"
USE_EMA="${USE_EMA:-auto}"
USE_AMP="${USE_AMP:-1}"
RUN_TAG="${RUN_TAG:-}"
REQUIRE_CHECKPOINT_SHA256="${REQUIRE_CHECKPOINT_SHA256:-1}"
D7_LIMIT_BATCHES="${D7_LIMIT_BATCHES:-2}"
D7_FEATURE_PREVIEW_DIMS="${D7_FEATURE_PREVIEW_DIMS:-8}"
D7_FEATURE_MAX_SLOTS_PER_SAMPLE="${D7_FEATURE_MAX_SLOTS_PER_SAMPLE:-64}"
D8_LIMIT_BATCHES="${D8_LIMIT_BATCHES:-2}"
D6_TOPK_LIST="${D6_TOPK_LIST:-100,300,1000}"
D6_IOU_THRESHOLDS="${D6_IOU_THRESHOLDS:-0.3,0.5,0.7}"
D6_REGENERATE_RAW_ROWS="${D6_REGENERATE_RAW_ROWS:-0}"
D6_LIMIT_BATCHES="${D6_LIMIT_BATCHES:-8}"
D6_TOPK_PER_SAMPLE="${D6_TOPK_PER_SAMPLE:-1000}"
D6_SCORE_BINS="${D6_SCORE_BINS:-10}"

if [ -n "$RUN_TAG" ] && [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_TAG must contain only letters, numbers, dot, underscore, or dash" >&2
  exit 2
fi

HEAD_SHORT="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
STAMP="$(date +%Y%m%d_%H%M%S)"
if [ -n "$RUN_TAG" ]; then
  OUT_ROOT="${OUT_ROOT:-logs/pcotmras_r17_r18_d6_d7_d8_${RUN_TAG}_${HEAD_SHORT}_${STAMP}}"
else
  OUT_ROOT="${OUT_ROOT:-logs/pcotmras_r17_r18_d6_d7_d8_${HEAD_SHORT}_${STAMP}}"
fi

R17_TRAIN_CONFIG="${R17_TRAIN_CONFIG:-configs/adatad/thumos/ctf_bdi_pc_ot_mras_r17_formal_train_candidate.py}"
R18_TRAIN_CONFIG="${R18_TRAIN_CONFIG:-configs/adatad/thumos/ctf_bdi_pc_ot_mras_r18_aux_formal_train_candidate.py}"
R17_EVAL_CONFIG="${R17_EVAL_CONFIG:-configs/adatad/thumos/ctf_bdi_pc_ot_mras_r17_post_train_eval_local_lowmem_candidate.py}"
R18_EVAL_CONFIG="${R18_EVAL_CONFIG:-configs/adatad/thumos/ctf_bdi_pc_ot_mras_r18_post_train_eval_local_lowmem_candidate.py}"
R17_CHECKPOINT="${R17_CHECKPOINT:-$CPU_PRERUN_REPO/logs/ctf_bdi_pc_ot_mras_r17_formal_train_6d15388_gatefix_20260621_185815/train_workdir/gpu1_id0/gpu1_id0/checkpoint/epoch_59.pth}"
R18_CHECKPOINT="${R18_CHECKPOINT:-$CPU_PRERUN_REPO/logs/ctf_bdi_pc_ot_mras_r18_aux_formal_6d15388_gatefix_20260621_185815/train_workdir/gpu1_id0/gpu1_id0/checkpoint/epoch_59.pth}"
R17_CHECKPOINT_SHA256="${R17_CHECKPOINT_SHA256:-11ab60a89a5af1968d0d21e67844001c058e4f9f087e46fad8b9b75c28cea2ba}"
R18_CHECKPOINT_SHA256="${R18_CHECKPOINT_SHA256:-9ce808746a828a6484e5d7cb8ae40f78c6ad38674bd607b5be4ab7aa1ffe21b0}"
D6_R17_JOINED_JSONL="${D6_R17_JOINED_JSONL:-$CPU_PRERUN_REPO/logs/pcotmras_p2_localization_attribution_ef29cb6_20260622_225701/R17_C0/localization_attribution/joined_proposals.jsonl}"
D6_R18_JOINED_JSONL="${D6_R18_JOINED_JSONL:-$CPU_PRERUN_REPO/logs/pcotmras_p2_localization_attribution_ef29cb6_20260622_225701/R18_C0/localization_attribution/joined_proposals.jsonl}"

echo "=== R17/R18 D6-D7-D8 diagnostics precheck ==="
echo "repo=$REPO"
echo "head=$HEAD_SHORT"
echo "precheck_only=$PRECHECK_ONLY"
echo "out_root=$OUT_ROOT"

"$PYTHON" -m py_compile \
  tools/bata/analyze_p2_raw_row_oracle_rerank.py \
  tools/bata/analyze_p2_proposal_localization.py \
  tools/bata/dump_native_irregular_p2_proposal_factors.py \
  tools/bata/dump_pc_ot_mras_reader_bridge_diagnostics.py \
  tools/bata/audit_pc_ot_mras_bridge_feature_distribution.py \
  tools/bata/replay_pc_ot_mras_training_assignment_geometry.py \
  tests/test_p2_raw_row_oracle_rerank.py \
  tests/test_pc_ot_mras_reader_bridge_diagnostic_dump.py \
  tests/test_pc_ot_mras_training_assignment_geometry_replay.py \
  tests/test_pc_ot_mras_r17_r18_d6_d7_d8_diagnostics_launcher.py
"$PYTHON" -m pytest \
  tests/test_p2_raw_row_oracle_rerank.py \
  tests/test_pc_ot_mras_reader_bridge_diagnostic_dump.py \
  tests/test_pc_ot_mras_training_assignment_geometry_replay.py \
  tests/test_pc_ot_mras_r17_r18_d6_d7_d8_diagnostics_launcher.py \
  -q

if [ "$PRECHECK_ONLY" = "1" ]; then
  echo "PRECHECK_ONLY=1; static checks passed and no checkpoint/data access was attempted."
  exit 0
fi

if [ "$ALLOW_DIAG" != "1" ]; then
  echo "Set ALLOW_PCOTMRAS_D6_D7_D8_DIAG=1 to run diagnostic-only GPU forwards." >&2
  exit 2
fi

if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "Full diagnostics must run inside a Slurm allocation or sbatch job." >&2
  exit 2
fi

verify_file() {
  local label="$1"
  local path="$2"
  local expected_sha="$3"
  test -f "$path" || { echo "$label missing: $path" >&2; exit 2; }
  if [ "$REQUIRE_CHECKPOINT_SHA256" = "1" ] && [ -z "$expected_sha" ]; then
    echo "$label SHA256 is required; set ${label}_SHA256 or REQUIRE_CHECKPOINT_SHA256=0." >&2
    exit 2
  fi
  if [ -n "$expected_sha" ]; then
    local actual_sha
    actual_sha="$(sha256sum "$path" | awk '{print $1}')"
    if [ "$actual_sha" != "$expected_sha" ]; then
      echo "$label SHA256 mismatch: expected $expected_sha, got $actual_sha" >&2
      exit 2
    fi
  fi
}

test -f "$R17_TRAIN_CONFIG"
test -f "$R18_TRAIN_CONFIG"
test -f "$R17_EVAL_CONFIG"
test -f "$R18_EVAL_CONFIG"
verify_file R17_CHECKPOINT "$R17_CHECKPOINT" "$R17_CHECKPOINT_SHA256"
verify_file R18_CHECKPOINT "$R18_CHECKPOINT" "$R18_CHECKPOINT_SHA256"

mkdir -p "$OUT_ROOT"/{D6_raw_row_oracle,D7_bridge_features,D8_training_assignment}
cat > "$OUT_ROOT/manifest.json" <<MANIFEST
{
  "schema_version": "pc_ot_mras_r17_r18_d6_d7_d8_diagnostic_manifest_v0",
  "diagnostic": "D6_D7_D8_R17_R18_mechanism_closeout",
  "diagnostic_only": true,
  "uses_train_gt_for_d8_assignment_replay_only": true,
  "uses_validation_gt_for_d6_oracle_join_only": true,
  "uses_test_gt": false,
  "metric_claim_allowed": false,
  "paper_claim_allowed": false,
  "training_or_eval_entrypoints_used": false,
  "repo": "$REPO",
  "git_head": "$HEAD_SHORT",
  "r17_checkpoint": "$R17_CHECKPOINT",
  "r18_checkpoint": "$R18_CHECKPOINT",
  "d6_regenerate_raw_rows": "$D6_REGENERATE_RAW_ROWS",
  "d7_limit_batches": $D7_LIMIT_BATCHES,
  "d8_limit_batches": $D8_LIMIT_BATCHES
}
MANIFEST

AMP_FLAG=()
if [ "$USE_AMP" = "1" ]; then
  AMP_FLAG+=(--amp)
fi

ensure_d6_rows() {
  local label="$1"
  local config="$2"
  local checkpoint="$3"
  local joined_path="$4"
  local case_dir="$OUT_ROOT/D6_raw_row_oracle/$label"
  if [ -f "$joined_path" ]; then
    echo "$joined_path"
    return
  fi
  if [ "$D6_REGENERATE_RAW_ROWS" != "1" ]; then
    echo "D6 joined rows missing for $label: $joined_path" >&2
    echo "Set D6_REGENERATE_RAW_ROWS=1 to regenerate bounded diagnostic raw rows." >&2
    exit 2
  fi
  mkdir -p "$case_dir"
  "$PYTHON" tools/bata/dump_native_irregular_p2_proposal_factors.py \
    --config "$config" \
    --checkpoint "$checkpoint" \
    --output-jsonl "$case_dir/proposal_factors.jsonl" \
    --summary-json "$case_dir/proposal_factor_summary.json" \
    --split val \
    --limit-batches "$D6_LIMIT_BATCHES" \
    --device "$DEVICE" \
    --use-ema "$USE_EMA" \
    --topk-per-sample "$D6_TOPK_PER_SAMPLE" \
    "${AMP_FLAG[@]}" > "$case_dir/proposal_factor_stdout.json"
  "$PYTHON" tools/bata/analyze_p2_proposal_localization.py \
    --proposal-jsonl "$case_dir/proposal_factors.jsonl" \
    --output-dir "$case_dir/localization_attribution" \
    --config "$config" \
    --dataset-split-key val \
    --split validation \
    --topk "$D6_TOPK_LIST" \
    --iou-thresholds "$D6_IOU_THRESHOLDS" \
    --score-bins "$D6_SCORE_BINS" \
    --group-by sample > "$case_dir/localization_attribution_stdout.json"
  echo "$case_dir/localization_attribution/joined_proposals.jsonl"
}

R17_D6_ROWS="$(ensure_d6_rows R17_C0 "$R17_EVAL_CONFIG" "$R17_CHECKPOINT" "$D6_R17_JOINED_JSONL")"
R18_D6_ROWS="$(ensure_d6_rows R18_C0 "$R18_EVAL_CONFIG" "$R18_CHECKPOINT" "$D6_R18_JOINED_JSONL")"

"$PYTHON" tools/bata/analyze_p2_raw_row_oracle_rerank.py \
  --input "R17_C0=$R17_D6_ROWS" \
  --input "R18_C0=$R18_D6_ROWS" \
  --output-json "$OUT_ROOT/D6_raw_row_oracle/summary.json" \
  --topk "$D6_TOPK_LIST" \
  --iou-thresholds "$D6_IOU_THRESHOLDS" \
  --group-by sample

run_d7() {
  local label="$1"
  local config="$2"
  local checkpoint="$3"
  local case_dir="$OUT_ROOT/D7_bridge_features/$label"
  mkdir -p "$case_dir"
  "$PYTHON" tools/bata/dump_pc_ot_mras_reader_bridge_diagnostics.py \
    --mode checkpoint \
    --config "$config" \
    --checkpoint "$checkpoint" \
    --output-json "$case_dir/summary.json" \
    --snapshot-jsonl "$case_dir/snapshots.jsonl" \
    --feature-jsonl "$case_dir/bridge_features.jsonl" \
    --split val \
    --limit-batches "$D7_LIMIT_BATCHES" \
    --device "$DEVICE" \
    --use-ema "$USE_EMA" \
    --feature-preview-dims "$D7_FEATURE_PREVIEW_DIMS" \
    --feature-max-slots-per-sample "$D7_FEATURE_MAX_SLOTS_PER_SAMPLE" \
    "${AMP_FLAG[@]}"
}

run_d8() {
  local label="$1"
  local config="$2"
  local checkpoint="$3"
  local case_dir="$OUT_ROOT/D8_training_assignment/$label"
  mkdir -p "$case_dir"
  "$PYTHON" tools/bata/replay_pc_ot_mras_training_assignment_geometry.py \
    --config "$config" \
    --checkpoint "$checkpoint" \
    --output-dir "$case_dir" \
    --split train \
    --limit-batches "$D8_LIMIT_BATCHES" \
    --device "$DEVICE" \
    --use-ema "$USE_EMA" \
    "${AMP_FLAG[@]}"
}

run_d7 R17_final "$R17_TRAIN_CONFIG" "$R17_CHECKPOINT"
run_d7 R18_final "$R18_TRAIN_CONFIG" "$R18_CHECKPOINT"
run_d8 R17_final "$R17_TRAIN_CONFIG" "$R17_CHECKPOINT"
run_d8 R18_final "$R18_TRAIN_CONFIG" "$R18_CHECKPOINT"

"$PYTHON" - "$OUT_ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

bundle = {
    "schema_version": "pc_ot_mras_r17_r18_d6_d7_d8_bundle_v0",
    "diagnostic": "D6_D7_D8_R17_R18_mechanism_closeout",
    "diagnostic_only": True,
    "uses_train_gt_for_d8_assignment_replay_only": True,
    "uses_validation_gt_for_d6_oracle_join_only": True,
    "metric_claim_allowed": False,
    "paper_claim_allowed": False,
    "D6_raw_row_oracle": read(root / "D6_raw_row_oracle" / "summary.json"),
    "D7_bridge_features": {
        case.name: read(case / "summary.json")
        for case in sorted((root / "D7_bridge_features").iterdir())
        if (case / "summary.json").exists()
    },
    "D8_training_assignment": {
        case.name: read(case / "summary.json")
        for case in sorted((root / "D8_training_assignment").iterdir())
        if (case / "summary.json").exists()
    },
}
(root / "bundle_summary.json").write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(bundle, sort_keys=True))
PY

echo "R17/R18 D6-D7-D8 diagnostics complete: $OUT_ROOT"
