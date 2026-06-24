#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH -J c3_interval_score_first_precheck
#SBATCH -o logs/%x-%j.out

set -eo pipefail
set -u

YUZIBO_ROOT="${YUZIBO_ROOT:-$HOME/run/yuzibo}"
REPO="${OPENTAD_PCOTMRAS_CLEAN_ROOT:-$YUZIBO_ROOT/OpenTAD_Back_check}"
CONFIG="configs/adatad/thumos/pc_ot_mras_prebackbone_c3_interval_score_first_precheck_n16r4.py"
FULL_CONFIG="configs/adatad/thumos/pc_ot_mras_prebackbone_c3_interval_score_first_full_train_candidate_n16r4.py"
PRECHECK_ONLY="${PRECHECK_ONLY:-1}"
ALLOW_C3_INTERVAL_SCORE_FIRST_FULL_TRAIN="${ALLOW_C3_INTERVAL_SCORE_FIRST_FULL_TRAIN:-0}"
ALLOW_C3_INTERVAL_SCORE_FIRST_RAW_PREDICTION_CACHE="${ALLOW_C3_INTERVAL_SCORE_FIRST_RAW_PREDICTION_CACHE:-0}"
ALLOW_C3_INTERVAL_SCORE_FIRST_TEACHER="${ALLOW_C3_INTERVAL_SCORE_FIRST_TEACHER:-0}"
ALLOW_C3_INTERVAL_SCORE_FIRST_TEST_GT="${ALLOW_C3_INTERVAL_SCORE_FIRST_TEST_GT:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

log() {
  echo "[c3-interval-score-first-precheck] $(date -Is) $*"
}

fail() {
  log "ERROR $*"
  exit 2
}

case "$PRECHECK_ONLY" in
  0|1) ;;
  *) fail "PRECHECK_ONLY must be exactly 0 or 1; got $PRECHECK_ONLY" ;;
esac
case "$ALLOW_C3_INTERVAL_SCORE_FIRST_RAW_PREDICTION_CACHE" in
  0) ;;
  *) fail "raw_prediction/cache is forbidden for C3 interval-score-first" ;;
esac
case "$ALLOW_C3_INTERVAL_SCORE_FIRST_TEACHER" in
  0) ;;
  *) fail "teacher signals are forbidden for C3 interval-score-first" ;;
esac
case "$ALLOW_C3_INTERVAL_SCORE_FIRST_TEST_GT" in
  0) ;;
  *) fail "test-GT visibility is forbidden for C3 interval-score-first" ;;
esac

if [ "$PRECHECK_ONLY" = "0" ]; then
  case "$ALLOW_C3_INTERVAL_SCORE_FIRST_FULL_TRAIN" in
    1) fail "full training remains locked; use a separately reviewed launcher after read-only review" ;;
    *) fail "PRECHECK_ONLY=0 requires a separate reviewed full-train gate" ;;
  esac
fi

if [ ! -d "$REPO" ]; then
  fail "missing repo: $REPO"
fi
cd "$REPO"

test -f "$CONFIG" || fail "missing precheck config: $CONFIG"
test -f "$FULL_CONFIG" || fail "missing candidate config: $FULL_CONFIG"

"$PYTHON_BIN" - "$CONFIG" "$FULL_CONFIG" <<'PY'
import sys
from mmengine.config import Config

precheck = Config.fromfile(sys.argv[1])
candidate = Config.fromfile(sys.argv[2])

for name, cfg in (("precheck", precheck), ("candidate", candidate)):
    selector = cfg.model.frame_selector
    gate = cfg.pc_ot_mras_prebackbone_e2e_acquisition_gate
    scope = cfg.experiment_scope
    if selector.selection_strategy != "interval_score_first_packet":
        raise SystemExit(f"{name}: selector strategy drift")
    if selector.frame_score_st_surrogate != "global_softmax":
        raise SystemExit(f"{name}: surrogate must stay global_softmax")
    if int(selector.target_len) != 384 or int(selector.dense_window_size) != 768:
        raise SystemExit(f"{name}: fixed 384/768 contract drift")
    if int(selector.max_dense_gap) != 0 or int(selector.max_gap_guard_count) != 0:
        raise SystemExit(f"{name}: coverage/max-gap guard must stay disabled by default")
    if cfg.model.rpn_head.type != "ActionFormerHead":
        raise SystemExit(f"{name}: detector head drift")
    if "PCOTMRASDetectorBridge" in repr(cfg.model):
        raise SystemExit(f"{name}: detector bridge is forbidden")
    if gate.allow_tools_train or gate.allow_tools_test or gate.allow_detector_map:
        raise SystemExit(f"{name}: train/test/map gates must be locked")
    if gate.allow_long_training or gate.allow_slurm or gate.allow_gpu:
        raise SystemExit(f"{name}: long run gates must be locked")
    if gate.allowed_entrypoints != ():
        raise SystemExit(f"{name}: allowed_entrypoints must be empty")
    for key in ("uses_teacher", "uses_test_gt", "uses_raw_prediction_cache"):
        if bool(scope.get(key, False)):
            raise SystemExit(f"{name}: {key} must stay false")
    if not bool(scope.changes_input_sampling):
        raise SystemExit(f"{name}: input sampling change must be explicit")
    for key in ("changes_detector_head", "changes_neck", "changes_loss_assignment", "changes_post_processing"):
        if bool(scope.get(key, False)):
            raise SystemExit(f"{name}: {key} must stay false")
    if bool(cfg.inference.load_from_raw_predictions) or bool(cfg.inference.save_raw_prediction):
        raise SystemExit(f"{name}: raw prediction inference flags must stay false")

print("C3_INTERVAL_SCORE_FIRST_PRECHECK_CONFIG_PASS")
PY

"$PYTHON_BIN" -m py_compile \
  "$CONFIG" \
  "$FULL_CONFIG" \
  opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py

"$PYTHON_BIN" -m pytest tests/test_pc_ot_mras_prebackbone_c3_interval_score_first.py -q

if [ "$PRECHECK_ONLY" = "1" ]; then
  log "C3_INTERVAL_SCORE_FIRST_PRECHECK_ONLY_PASS_NO_TRAIN_NO_TEST"
  exit 0
fi
