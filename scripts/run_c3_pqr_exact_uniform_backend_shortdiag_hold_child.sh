#!/usr/bin/env bash
set -euo pipefail

ROOT="${PQR_ROOT:-/data/home/sczc063/run/yuzibo/OpenTAD_C3PQRRankCal_Precheck_20260630/github_clean_c3_pqr_rankcal_v1}"
CONFIG_REL="configs/adatad/thumos/c3_indirect_original_adatad_32px_a_exact_uniform_backend_control_pqr_rankcal_v1_shortdiag.py"
CONFIG="$ROOT/$CONFIG_REL"
STAMP="${PQR_STAMP:-$(date +%Y%m%d_%H%M%S_%z)}"
RUN_NAME="${PQR_RUN_NAME:-pqr_exact_uniform_backend_shortdiag_${STAMP}}"
RUN_DIR="$ROOT/exps/thumos/adatad/${RUN_NAME}"
LOG_DIR="$ROOT/logs"
LOG="$LOG_DIR/${RUN_NAME}.log"
TMP_CONFIG="$LOG_DIR/${RUN_NAME}_runtime_config.py"

mkdir -p "$LOG_DIR" "$RUN_DIR"
cd "$ROOT"

cat > "$TMP_CONFIG" <<EOF
_base_ = ["../$CONFIG_REL"]
work_dir = "$RUN_DIR"
post_processing = dict(save_dict=True)
pqr_rankcal_v1 = dict(
    diagnostic_only=True,
    claim_map_improvement=False,
    official_map_claim=False,
    remote_launch_locked=True,
    backend_control="adapter_stride2_uniform_50pct",
)
EOF

{
  echo "BEGIN $(date '+%F %T %Z')"
  echo "diagnostic_only=true"
  echo "official_map_claim=false"
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
  export MASTER_PORT="${MASTER_PORT:-30931}"
  export WORLD_SIZE="${WORLD_SIZE:-1}"
  export RANK="${RANK:-0}"
  export LOCAL_RANK="${LOCAL_RANK:-0}"

  python -m py_compile tools/analyze_c3_pqr_rankcal_proposals.py tools/validate_c3_pqr_rankcal_v1_config.py
  python tools/validate_c3_pqr_rankcal_v1_config.py "$CONFIG"
  python tools/train.py "$TMP_CONFIG" --id 0 --cfg-options \
    annotation_path=/data/home/sczc063/run/yuzibo/thumos14/annotations/thumos_14_anno.json \
    class_map=/data/home/sczc063/run/yuzibo/thumos14/annotations/category_idx.txt \
    train_data_path=/data/home/sczc063/run/yuzibo/thumos14/train \
    test_data_path=/data/home/sczc063/run/yuzibo/thumos14/test \
    dataset.train.ann_file=/data/home/sczc063/run/yuzibo/thumos14/annotations/thumos_14_anno.json \
    dataset.train.class_map=/data/home/sczc063/run/yuzibo/thumos14/annotations/category_idx.txt \
    dataset.train.data_path=/data/home/sczc063/run/yuzibo/thumos14/train \
    dataset.val.ann_file=/data/home/sczc063/run/yuzibo/thumos14/annotations/thumos_14_anno.json \
    dataset.val.class_map=/data/home/sczc063/run/yuzibo/thumos14/annotations/category_idx.txt \
    dataset.val.data_path=/data/home/sczc063/run/yuzibo/thumos14/test \
    dataset.test.ann_file=/data/home/sczc063/run/yuzibo/thumos14/annotations/thumos_14_anno.json \
    dataset.test.class_map=/data/home/sczc063/run/yuzibo/thumos14/annotations/category_idx.txt \
    dataset.test.data_path=/data/home/sczc063/run/yuzibo/thumos14/test

  if [[ -f "$RUN_DIR/result_detection.json" ]]; then
    python tools/analyze_c3_pqr_rankcal_proposals.py \
      --prediction "$RUN_DIR/result_detection.json" \
      --annotation /data/home/sczc063/run/yuzibo/thumos14/annotations/thumos_14_anno.json \
      --output "$RUN_DIR/pqr_score_iou_rank_proposal_diagnostic.json" \
      --records-csv "$RUN_DIR/pqr_score_iou_rank_proposal_records.csv"
  else
    python tools/analyze_c3_pqr_rankcal_proposals.py \
      --prediction "$RUN_DIR/result_detection.json" \
      --annotation /data/home/sczc063/run/yuzibo/thumos14/annotations/thumos_14_anno.json \
      --output "$RUN_DIR/pqr_score_iou_rank_proposal_diagnostic_missing.json" || true
  fi

  echo "END $(date '+%F %T %Z')"
} 2>&1 | tee "$LOG"
