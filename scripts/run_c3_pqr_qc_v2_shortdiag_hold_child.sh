#!/usr/bin/env bash
set -euo pipefail

EXPECTED_HEAD="${PQR_EXPECTED_HEAD:-9b3859d1b2d43b1be8b860a99fe37ee9169ff83a}"

fail_root_guard() {
  echo "PQR_QCV2_ROOT_GUARD_FAIL $*"
  exit 41
}

if [[ -z "${PQR_ROOT:-}" ]]; then
  fail_root_guard "PQR_ROOT must be explicitly set to the current prechecked QC V2 clone; expected HEAD=$EXPECTED_HEAD. Refusing to default to an old RankCal clone."
fi

ROOT="$PQR_ROOT"
if [[ ! -d "$ROOT" ]]; then
  fail_root_guard "PQR_ROOT does not exist: $ROOT; set PQR_ROOT to the current prechecked QC V2 clone."
fi

if ! git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  fail_root_guard "PQR_ROOT is not a git repository: $ROOT; set PQR_ROOT to the current prechecked QC V2 clone."
fi

ACTUAL_HEAD="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null)" || fail_root_guard "cannot read git HEAD from PQR_ROOT: $ROOT"
if [[ "$ACTUAL_HEAD" != "$EXPECTED_HEAD" ]]; then
  echo "PQR_QCV2_HEAD_GUARD_FAIL PQR_ROOT=$ROOT expected=$EXPECTED_HEAD actual=$ACTUAL_HEAD"
  exit 41
fi
echo "PQR_QCV2_ROOT_GUARD_PASS root=$ROOT head=$ACTUAL_HEAD expected=$EXPECTED_HEAD"

if [[ "${CUDA_VISIBLE_DEVICES:-}" != "1" ]]; then
  echo "PQR_QCV2_GPU_GUARD_FAIL expected CUDA_VISIBLE_DEVICES=1 for C3/PQR/QC V2 GPU1-only diagnostics, got ${CUDA_VISIBLE_DEVICES:-unset}"
  exit 42
fi

CONFIG_REL="configs/adatad/thumos/c3_indirect_original_adatad_32px_a_pqr_rankcal_v1_sparse_irregular_qc_v2_shortdiag.py"
CONFIG="$ROOT/$CONFIG_REL"
STAMP="${PQR_STAMP:-$(date +%Y%m%d_%H%M%S_%z)}"
RUN_NAME="${PQR_RUN_NAME:-pqr_qc_v2_shortdiag_${STAMP}}"
RUN_DIR="$ROOT/exps/thumos/adatad/${RUN_NAME}"
LOG_DIR="$ROOT/logs/${RUN_NAME}"
LOG="$LOG_DIR/${RUN_NAME}.log"
TMP_CONFIG="$LOG_DIR/${RUN_NAME}_runtime_config.py"
ANNOTATION_PATH="/data/home/sczc063/run/yuzibo/thumos14/annotations/thumos_14_anno.json"
CLASS_MAP="/data/home/sczc063/run/yuzibo/thumos14/annotations/category_idx.txt"
TRAIN_DATA_PATH="/data/home/sczc063/run/yuzibo/thumos14/train"
TEST_DATA_PATH="/data/home/sczc063/run/yuzibo/thumos14/test"

mkdir -p "$LOG_DIR" "$RUN_DIR"
cd "$ROOT"

cat > "$TMP_CONFIG" <<EOF
_base_ = ["../../$CONFIG_REL"]
work_dir = "$RUN_DIR"
post_processing = dict(
    save_dict=True,
    qc_v2_diagnostic_dump=True,
)
pqr_rankcal_v1 = dict(
    diagnostic_only=True,
    formal_fulltrain=False,
    user_override_fulltrain=False,
    claim_map_improvement=False,
    official_map_claim=False,
    remote_launch_locked=True,
    use_teacher=False,
    use_test_gt=False,
    use_raw_prediction_cache=False,
    backend_control="random_fixed_adapter_50pct",
)
EOF

{
  echo "BEGIN $(date '+%F %T %Z')"
  echo "diagnostic_only=true"
  echo "formal_fulltrain=false"
  echo "official_map_claim=false"
  echo "claim_map_improvement=false"
  echo "config=$CONFIG"
  echo "runtime_config=$TMP_CONFIG"
  echo "run_dir=$RUN_DIR"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

  module load cuda/11.8
  module load miniforge3/24.11
  source activate /data/home/sczc063/run/yuzibo/conda_envs/opentad

  export PYTHONNOUSERSITE=1
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
  export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  export MASTER_PORT="${MASTER_PORT:-30941}"
  export WORLD_SIZE="${WORLD_SIZE:-1}"
  export RANK="${RANK:-0}"
  export LOCAL_RANK="${LOCAL_RANK:-0}"

  python -m py_compile tools/analyze_c3_pqr_rankcal_proposals.py tools/validate_c3_pqr_rankcal_v1_config.py
  python tools/validate_c3_pqr_rankcal_v1_config.py "$CONFIG"
  python tools/validate_c3_pqr_rankcal_v1_config.py "$TMP_CONFIG"

  python tools/train.py "$TMP_CONFIG" --id 0 --cfg-options \
    annotation_path="$ANNOTATION_PATH" \
    class_map="$CLASS_MAP" \
    train_data_path="$TRAIN_DATA_PATH" \
    test_data_path="$TEST_DATA_PATH" \
    dataset.train.ann_file="$ANNOTATION_PATH" \
    dataset.train.class_map="$CLASS_MAP" \
    dataset.train.data_path="$TRAIN_DATA_PATH" \
    dataset.val.ann_file="$ANNOTATION_PATH" \
    dataset.val.class_map="$CLASS_MAP" \
    dataset.val.data_path="$TEST_DATA_PATH" \
    dataset.test.ann_file="$ANNOTATION_PATH" \
    dataset.test.class_map="$CLASS_MAP" \
    dataset.test.data_path="$TEST_DATA_PATH"

  RESULT_JSON="$RUN_DIR/result_detection.json"
  if [[ ! -f "$RESULT_JSON" ]]; then
    mapfile -t CANDIDATE_RESULTS < <(find "$RUN_DIR" -path "*/gpu*_id*/result_detection.json" -type f | sort)
    if [[ "${#CANDIDATE_RESULTS[@]}" -eq 1 ]]; then
      RESULT_JSON="${CANDIDATE_RESULTS[0]}"
      echo "PQR_QCV2_RESULT_DISCOVERED $RESULT_JSON"
    elif [[ "${#CANDIDATE_RESULTS[@]}" -gt 1 ]]; then
      echo "PQR_QCV2_RESULT_AMBIGUOUS requested=$RUN_DIR/result_detection.json candidates=${CANDIDATE_RESULTS[*]}"
      python tools/analyze_c3_pqr_rankcal_proposals.py \
        --prediction "$RUN_DIR/result_detection.json" \
        --annotation "$ANNOTATION_PATH" \
        --output "$RUN_DIR/pqr_qc_v2_proposal_diagnostic_missing.json" || true
      exit 44
    fi
  fi

  if [[ ! -f "$RESULT_JSON" ]]; then
    echo "PQR_QCV2_RESULT_MISSING expected $RUN_DIR/result_detection.json or one nested gpu*_id*/result_detection.json"
    python tools/analyze_c3_pqr_rankcal_proposals.py \
      --prediction "$RUN_DIR/result_detection.json" \
      --annotation "$ANNOTATION_PATH" \
      --output "$RUN_DIR/pqr_qc_v2_proposal_diagnostic_missing.json" || true
    exit 43
  fi

  python tools/analyze_c3_pqr_rankcal_proposals.py \
    --prediction "$RESULT_JSON" \
    --annotation "$ANNOTATION_PATH" \
    --output "$RUN_DIR/pqr_qc_v2_proposal_diagnostic.json" \
    --sweep-output "$RUN_DIR/pqr_qc_v2_proposal_diagnostic_sweep.json" \
    --records-csv "$RUN_DIR/pqr_qc_v2_proposal_diagnostic_records.csv"

  echo "END $(date '+%F %T %Z')"
} 2>&1 | tee "$LOG"
