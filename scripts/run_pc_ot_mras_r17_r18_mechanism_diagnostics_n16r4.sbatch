#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -J r17r18_mech_diag
#SBATCH -o logs/%x-%j.out

set -euo pipefail

YUZIBO_ROOT="${YUZIBO_ROOT:-$HOME/run/yuzibo}"
REPO="${OPENTAD_PCOTMRAS_CLEAN_ROOT:-$YUZIBO_ROOT/OpenTAD_PCOTMRAS_R16_R18_R20_Clean_20260619_1730}"
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
ALLOW_DIAG="${ALLOW_PCOTMRAS_R17_R18_MECH_DIAG:-0}"
LIMIT_BATCHES="${LIMIT_BATCHES:-4}"
TOPK_PER_SAMPLE="${TOPK_PER_SAMPLE:-1000}"
SCORE_BINS="${SCORE_BINS:-10}"
BUDGET="${BUDGET:-384}"
DEVICE="${DEVICE:-cuda}"
USE_EMA="${USE_EMA:-auto}"
USE_AMP="${USE_AMP:-1}"
RUN_TAG="${RUN_TAG:-}"
REQUIRE_CHECKPOINT_SHA256="${REQUIRE_CHECKPOINT_SHA256:-1}"

if [ -n "$RUN_TAG" ] && [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_TAG must contain only letters, numbers, dot, underscore, or dash" >&2
  exit 2
fi

HEAD_SHORT="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
STAMP="$(date +%Y%m%d_%H%M%S)"
if [ -n "$RUN_TAG" ]; then
  OUT_ROOT="${OUT_ROOT:-logs/pcotmras_r17_r18_mechanism_diagnostics_${RUN_TAG}_${HEAD_SHORT}_${STAMP}}"
else
  OUT_ROOT="${OUT_ROOT:-logs/pcotmras_r17_r18_mechanism_diagnostics_${HEAD_SHORT}_${STAMP}}"
fi

R17_CONFIG="${R17_CONFIG:-configs/adatad/thumos/ctf_bdi_pc_ot_mras_r17_formal_train_candidate.py}"
R18_CONFIG="${R18_CONFIG:-configs/adatad/thumos/ctf_bdi_pc_ot_mras_r18_aux_formal_train_candidate.py}"
R17_MECH_CHECKPOINT="${R17_MECH_CHECKPOINT:-$REPO/logs/ctf_bdi_pc_ot_mras_r17_formal_train_6d15388_gatefix_20260621_185815/train_workdir/gpu1_id0/gpu1_id0/checkpoint/epoch_59.pth}"
R18_MECH_CHECKPOINT="${R18_MECH_CHECKPOINT:-$REPO/logs/ctf_bdi_pc_ot_mras_r18_aux_formal_6d15388_gatefix_20260621_185815/train_workdir/gpu1_id0/gpu1_id0/checkpoint/epoch_59.pth}"
R17_MECH_CHECKPOINT_SHA256="${R17_MECH_CHECKPOINT_SHA256:-}"
R18_MECH_CHECKPOINT_SHA256="${R18_MECH_CHECKPOINT_SHA256:-}"

echo "=== R17/R18 mechanism diagnostic precheck ==="
echo "repo=$REPO"
echo "head=$HEAD_SHORT"
echo "precheck_only=$PRECHECK_ONLY"
echo "out_root=$OUT_ROOT"
"$PYTHON" -m py_compile \
  tools/bata/diagnose_pc_ot_mras_r17_r18_mechanisms.py \
  tests/test_pc_ot_mras_r17_r18_mechanism_diagnostics.py
"$PYTHON" -m pytest tests/test_pc_ot_mras_r17_r18_mechanism_diagnostics.py -q

if [ "$PRECHECK_ONLY" = "1" ]; then
  echo "PRECHECK_ONLY=1; static checks passed and no checkpoint/data access was attempted."
  exit 0
fi

if [ "$ALLOW_DIAG" != "1" ]; then
  echo "Set ALLOW_PCOTMRAS_R17_R18_MECH_DIAG=1 to run diagnostic-only GPU forwards." >&2
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

test -f "$R17_CONFIG"
test -f "$R18_CONFIG"
verify_file R17_MECH_CHECKPOINT "$R17_MECH_CHECKPOINT" "$R17_MECH_CHECKPOINT_SHA256"
verify_file R18_MECH_CHECKPOINT "$R18_MECH_CHECKPOINT" "$R18_MECH_CHECKPOINT_SHA256"

mkdir -p "$OUT_ROOT"
cat > "$OUT_ROOT/manifest.json" <<MANIFEST
{
  "schema_version": "pc_ot_mras_r17_r18_mechanism_diagnostic_manifest_v0",
  "diagnostic": "D1_D2_D3_R17_R18_first_wave",
  "diagnostic_only": true,
  "uses_validation_gt_for_diagnostic_iou_only": true,
  "tools_train": false,
  "tools_test": false,
  "metric_claim_allowed": false,
  "paper_claim_allowed": false,
  "repo": "$REPO",
  "git_head": "$HEAD_SHORT",
  "r17_config": "$R17_CONFIG",
  "r18_config": "$R18_CONFIG",
  "r17_checkpoint": "$R17_MECH_CHECKPOINT",
  "r18_checkpoint": "$R18_MECH_CHECKPOINT",
  "limit_batches": $LIMIT_BATCHES,
  "topk_per_sample": $TOPK_PER_SAMPLE,
  "score_bins": $SCORE_BINS,
  "budget": $BUDGET
}
MANIFEST

AMP_FLAG=()
if [ "$USE_AMP" = "1" ]; then
  AMP_FLAG+=(--amp)
fi

run_case() {
  local case_name="$1"
  local config="$2"
  local checkpoint="$3"
  echo "=== $case_name ==="
  echo "config=$config"
  echo "checkpoint=$checkpoint"
  "$PYTHON" tools/bata/diagnose_pc_ot_mras_r17_r18_mechanisms.py \
    --case-name "$case_name" \
    --config "$config" \
    --checkpoint "$checkpoint" \
    --output-dir "$OUT_ROOT/$case_name" \
    --split val \
    --limit-batches "$LIMIT_BATCHES" \
    --device "$DEVICE" \
    --use-ema "$USE_EMA" \
    --budget "$BUDGET" \
    --topk-per-sample "$TOPK_PER_SAMPLE" \
    --score-bins "$SCORE_BINS" \
    "${AMP_FLAG[@]}"
}

run_case D1_D2_D3_R17_final "$R17_CONFIG" "$R17_MECH_CHECKPOINT"
run_case D1_D2_D3_R18_final "$R18_CONFIG" "$R18_MECH_CHECKPOINT"

"$PYTHON" - "$OUT_ROOT" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
cases = {}
for summary_path in sorted(root.glob("*/D1_D2_D3_*_mechanism_summary.json")):
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    cases[payload["case_name"]] = {
        "summary_json": str(summary_path),
        "status": payload.get("status"),
        "verdict": payload.get("verdict"),
        "primary_failure_mode": payload.get("primary_failure_mode"),
        "has_gt": payload.get("has_gt"),
        "has_meta": payload.get("has_meta"),
        "sample_count": payload.get("sample_count"),
        "proposal_row_count": payload.get("proposal_row_count"),
        "D1": payload.get("D1_reader_acquisition_utility"),
        "D2": payload.get("D2_temporal_geometry_contract"),
        "D3": payload.get("D3_proposal_survival_light"),
    }
bundle = {
    "schema_version": "pc_ot_mras_r17_r18_mechanism_diagnostic_bundle_v0",
    "diagnostic": "D1_D2_D3_R17_R18_first_wave",
    "diagnostic_only": True,
    "metric_claim_allowed": False,
    "paper_claim_allowed": False,
    "cases": cases,
}
(root / "bundle_summary.json").write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(bundle, sort_keys=True))
PY

echo "R17/R18 D1-D2-D3 mechanism diagnostics complete: $OUT_ROOT"
