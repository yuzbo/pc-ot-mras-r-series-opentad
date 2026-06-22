#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH -J pcot_rd_eval
#SBATCH -o logs/%x-%j.out

set -eo pipefail

source /etc/profile >/dev/null 2>&1 || true
set -u
module load cuda/11.8
module load miniforge3/24.11

YUZIBO_ROOT="${YUZIBO_ROOT:-$HOME/run/yuzibo}"
source activate "$YUZIBO_ROOT/conda_envs/opentad"

CLEAN_REPO_NAME="${OPENTAD_PCOTMRAS_CLEAN_REPO_NAME:-OpenTAD_PCOTMRAS_R16_R18_R20_Clean_20260619_1730}"
REPO="${OPENTAD_PCOTMRAS_CLEAN_ROOT:-$YUZIBO_ROOT/$CLEAN_REPO_NAME}"
case "$REPO" in
  *OpenTAD_BATA_Clean*)
    echo "[pc-ot-mras-reader-disabled-eval] $(date -Is) ERROR refusing dirty historical OpenTAD_BATA_Clean path: $REPO" >&2
    exit 2
    ;;
esac
if [ ! -d "$REPO" ]; then
  echo "[pc-ot-mras-reader-disabled-eval] $(date -Is) ERROR missing clean PC-OT-MRAS repo: $REPO" >&2
  exit 2
fi
cd "$REPO"

TARGET_RAW="${PCOTMRAS_READER_DISABLED_TARGET:-${TARGET:-R17}}"
TARGET_UPPER="$(printf '%s' "$TARGET_RAW" | tr '[:lower:]' '[:upper:]')"
case "$TARGET_UPPER" in
  R17)
    CONFIG_REVIEWED="configs/adatad/thumos/ctf_bdi_pc_ot_mras_r17_reader_disabled_eval_candidate.py"
    GATE_NAME="r17_pc_ot_mras_reader_disabled_eval_gate"
    DISABLED_OLD_GATE_NAME="r17_pc_ot_mras_post_train_eval_gate"
    DECISION="ALLOW_R17_READER_DISABLED_EVAL"
    ALLOW_READER_DISABLED_EVAL="${ALLOW_R17_READER_DISABLED_EVAL:-0}"
    GATE_JSON="${R17_READER_DISABLED_EVAL_GATE_JSON:-}"
    GATE_SHA256="${R17_READER_DISABLED_EVAL_GATE_SHA256:-}"
    EVAL_CHECKPOINT="${R17_EVAL_CHECKPOINT:-}"
    EVAL_CHECKPOINT_SHA256="${R17_EVAL_CHECKPOINT_SHA256:-}"
    DEFAULT_PORT_BASE=36000
    ;;
  R18)
    CONFIG_REVIEWED="configs/adatad/thumos/ctf_bdi_pc_ot_mras_r18_reader_disabled_eval_candidate.py"
    GATE_NAME="r18_pc_ot_mras_reader_disabled_eval_gate"
    DISABLED_OLD_GATE_NAME="r18_pc_ot_mras_post_train_eval_gate"
    DECISION="ALLOW_R18_READER_DISABLED_EVAL"
    ALLOW_READER_DISABLED_EVAL="${ALLOW_R18_READER_DISABLED_EVAL:-0}"
    GATE_JSON="${R18_READER_DISABLED_EVAL_GATE_JSON:-}"
    GATE_SHA256="${R18_READER_DISABLED_EVAL_GATE_SHA256:-}"
    EVAL_CHECKPOINT="${R18_EVAL_CHECKPOINT:-}"
    EVAL_CHECKPOINT_SHA256="${R18_EVAL_CHECKPOINT_SHA256:-}"
    DEFAULT_PORT_BASE=37000
    ;;
  *)
    echo "[pc-ot-mras-reader-disabled-eval] $(date -Is) ERROR PCOTMRAS_READER_DISABLED_TARGET must be R17 or R18; got $TARGET_RAW" >&2
    exit 2
    ;;
esac
TARGET_LOWER="$(printf '%s' "$TARGET_UPPER" | tr '[:upper:]' '[:lower:]')"
CONFIG="${CONFIG:-$CONFIG_REVIEWED}"
PRECHECK_ONLY="${PRECHECK_ONLY:-1}"
ALLOW_TOOLS_TRAIN="${ALLOW_PCOTMRAS_READER_DISABLED_TOOLS_TRAIN:-0}"
ALLOW_RAW_PREDICTION_CACHE="${ALLOW_PCOTMRAS_READER_DISABLED_RAW_PREDICTION_CACHE:-0}"
ALLOW_RUNTIME_OR_DEPLOY_CLAIM="${ALLOW_PCOTMRAS_READER_DISABLED_RUNTIME_OR_DEPLOY_CLAIM:-0}"
EVAL_ID="${EVAL_ID:-0}"
SEED="${SEED:-20260619}"
SLURM_JOB_ID_FOR_PORT="${SLURM_JOB_ID:-0}"
DEFAULT_MASTER_PORT=$((DEFAULT_PORT_BASE + (SLURM_JOB_ID_FOR_PORT % 1000)))
REQUESTED_MASTER_PORT="${MASTER_PORT:-}"
MASTER_PORT_SOURCE="default_from_slurm_job_id"
if [ -n "$REQUESTED_MASTER_PORT" ]; then
  MASTER_PORT="$REQUESTED_MASTER_PORT"
  MASTER_PORT_SOURCE="environment"
else
  MASTER_PORT="$DEFAULT_MASTER_PORT"
fi
RUN_TAG="${RUN_TAG:-ctf_bdi_pc_ot_mras_${TARGET_LOWER}_reader_disabled_eval_${SLURM_JOB_ID:-nojob}_$(date '+%Y%m%d_%H%M%S')}"

case "$RUN_TAG" in
  *..*|*/*|*\\*|*[[:space:]]*|*[\;\&\|\>\<\`\"\'$]*)
    echo "[pc-ot-mras-reader-disabled-eval] $(date -Is) ERROR RUN_TAG must be a simple path-safe name; got $RUN_TAG" >&2
    exit 2
    ;;
esac

RUN_ROOT="$REPO/logs/$RUN_TAG"
RUN_LOG="$RUN_ROOT/${RUN_TAG}.log"
EVAL_STDOUT="$RUN_ROOT/${RUN_TAG}.tools_test.stdout.log"
SHA_MANIFEST="$RUN_ROOT/active_sha256_manifest.txt"
RESOLVED_CONFIG_DUMP="$RUN_ROOT/resolved_config.py"
SUMMARY_JSON="$RUN_ROOT/${TARGET_LOWER}_reader_disabled_eval_summary.json"
WORK_DIR_BASE="$RUN_ROOT/eval_workdir"
WORK_DIR="$WORK_DIR_BASE"
FINAL_WORK_DIR="$WORK_DIR/gpu1_id${EVAL_ID}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export YUZIBO_ROOT
export OPENTAD_PCOTMRAS_CLEAN_ROOT="$REPO"
export THUMOS14_ANNOTATION_PATH="${THUMOS14_ANNOTATION_PATH:-$YUZIBO_ROOT/thumos14/annotations/thumos_14_anno.json}"
export THUMOS14_CLASS_MAP="${THUMOS14_CLASS_MAP:-$YUZIBO_ROOT/thumos14/annotations/category_idx.txt}"
export THUMOS14_TEST_DATA_PATH="${THUMOS14_TEST_DATA_PATH:-$YUZIBO_ROOT/thumos14/test}"

mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_LOG") 2>&1

log() {
  echo "[pc-ot-mras-${TARGET_LOWER}-reader-disabled-eval] $(date -Is) $*"
}

fail() {
  log "ERROR $*"
  exit 2
}

write_summary() {
  local status="$1"
  local decision="$2"
  local reason="$3"
  python - "$SUMMARY_JSON" "$status" "$decision" "$reason" "$RUN_ROOT" "$CONFIG" "$WORK_DIR" "$FINAL_WORK_DIR" "$EVAL_CHECKPOINT" "$TARGET_UPPER" <<'PY'
import json
import os
import socket
import sys
from pathlib import Path

summary_path, status, decision, reason, run_root, config, work_dir, final_work_dir, checkpoint, target = sys.argv[1:11]
payload = {
    "status": status,
    "decision": decision,
    "reason": reason,
    "stage": f"{target}_READER_DISABLED_EVAL",
    "route": "CTF-BDI/PC-OT-MRAS",
    "host": socket.gethostname(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "run_root": run_root,
    "config": config,
    "work_dir": work_dir,
    "final_work_dir": final_work_dir,
    "checkpoint": checkpoint,
    "claim_boundaries": {
        "tools_test_only": True,
        "post_train_detector_map_reporting_allowed": True,
        "reader_disabled_same_head_control": True,
        "exact_uniform_reader_override": True,
        "tools_train": False,
        "raw_prediction_cache": False,
        "runtime_or_flops_claim_allowed": False,
        "deploy_claim_allowed": False,
        "paper_claim_allowed": False,
    },
}
Path(summary_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  log "summary_json=$SUMMARY_JSON"
}

on_exit() {
  local status="$?"
  if [ "$status" -ne 0 ] && [ ! -f "$SUMMARY_JSON" ]; then
    write_summary "FAIL" "${TARGET_UPPER}_READER_DISABLED_EVAL_FAILED" "launcher exited before successful completion"
  fi
}
trap on_exit EXIT

case "$PRECHECK_ONLY" in
  0|1) ;;
  *) fail "PRECHECK_ONLY must be exactly 0 or 1; got $PRECHECK_ONLY" ;;
esac
case "$ALLOW_TOOLS_TRAIN" in
  0) ;;
  *) fail "tools/train.py is not part of this reader-disabled eval launcher" ;;
esac
case "$ALLOW_RAW_PREDICTION_CACHE" in
  0) ;;
  *) fail "raw prediction/cache shortcuts are forbidden for reader-disabled eval" ;;
esac
case "$ALLOW_RUNTIME_OR_DEPLOY_CLAIM" in
  0) ;;
  *) fail "runtime/FLOPs/deploy claims are not approved for this launcher" ;;
esac
case "$MASTER_PORT" in
  ''|*[!0-9]*)
    if [ "$MASTER_PORT_SOURCE" = "environment" ]; then
      log "WARN ignoring invalid inherited MASTER_PORT=$MASTER_PORT; using default $DEFAULT_MASTER_PORT"
      MASTER_PORT="$DEFAULT_MASTER_PORT"
    else
      fail "MASTER_PORT must be numeric; got $MASTER_PORT"
    fi
    ;;
esac
if [ "$MASTER_PORT" -lt 1024 ] || [ "$MASTER_PORT" -gt 65535 ]; then
  log "WARN ignoring MASTER_PORT=$MASTER_PORT outside [1024,65535]; using default $DEFAULT_MASTER_PORT"
  MASTER_PORT="$DEFAULT_MASTER_PORT"
fi
case "$EVAL_ID" in
  ''|*[!0-9]*) fail "EVAL_ID must be numeric; got $EVAL_ID" ;;
esac
case "$SEED" in
  ''|*[!0-9]*) fail "SEED must be numeric; got $SEED" ;;
esac
if [ "$CONFIG" != "$CONFIG_REVIEWED" ]; then
  fail "launcher is reviewed only for $CONFIG_REVIEWED; got $CONFIG"
fi
if [ -z "${SLURM_JOB_ID:-}" ] && [ "${ALLOW_LOGIN_NODE_DEBUG:-0}" != "1" ]; then
  fail "refusing direct non-Slurm execution unless ALLOW_LOGIN_NODE_DEBUG=1 for lightweight PRECHECK_ONLY"
fi
if [ -z "${SLURM_JOB_ID:-}" ] && [ "${ALLOW_LOGIN_NODE_DEBUG:-0}" = "1" ] && [ "$PRECHECK_ONLY" != "1" ]; then
  fail "login-node debug must keep PRECHECK_ONLY=1; refusing eval outside Slurm"
fi
if [ "$PRECHECK_ONLY" = "0" ]; then
  case "$ALLOW_READER_DISABLED_EVAL" in
    1) ;;
    *) fail "PRECHECK_ONLY=0 requires ALLOW_${TARGET_UPPER}_READER_DISABLED_EVAL=1 after review" ;;
  esac
  if [ -z "$GATE_JSON" ] || [ -z "$GATE_SHA256" ]; then
    fail "PRECHECK_ONLY=0 requires ${TARGET_UPPER}_READER_DISABLED_EVAL_GATE_JSON and ${TARGET_UPPER}_READER_DISABLED_EVAL_GATE_SHA256"
  fi
  if [ -z "$EVAL_CHECKPOINT" ] || [ -z "$EVAL_CHECKPOINT_SHA256" ]; then
    fail "PRECHECK_ONLY=0 requires ${TARGET_UPPER}_EVAL_CHECKPOINT and ${TARGET_UPPER}_EVAL_CHECKPOINT_SHA256"
  fi
else
  case "$ALLOW_READER_DISABLED_EVAL" in
    0) ;;
    *) fail "PRECHECK_ONLY=1 must not set ALLOW_${TARGET_UPPER}_READER_DISABLED_EVAL=1" ;;
  esac
fi
if [ -n "${CHECKPOINT:-}" ] || [ -n "${LOAD_FROM:-}" ] || [ -n "${RESUME:-}" ]; then
  fail "generic CHECKPOINT/LOAD_FROM/RESUME env shortcuts are forbidden; use ${TARGET_UPPER}_EVAL_CHECKPOINT"
fi
if [ -n "${LOAD_FROM_RAW_PREDICTIONS:-}" ] || [ -n "${RAW_PREDICTION_CACHE:-}" ] || [ -n "${PREDICTION_CACHE:-}" ]; then
  fail "raw prediction/cache environment shortcuts are forbidden for reader-disabled eval"
fi
if [ -n "${CFG_OPTIONS:-}" ] || [ -n "${EXTRA_CFG_OPTIONS:-}" ]; then
  fail "arbitrary cfg-options are forbidden for reader-disabled eval"
fi
EXPECTED_GIT_BRANCH="${OPENTAD_PCOTMRAS_EXPECTED_BRANCH:-codex/pcotmras-r16-r18-r20-clean}"
GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)" || fail "clean repo must be a git checkout"
GIT_HEAD="$(git rev-parse HEAD 2>/dev/null)" || fail "clean repo HEAD is unavailable"
if [ "$GIT_BRANCH" != "$EXPECTED_GIT_BRANCH" ]; then
  fail "expected branch $EXPECTED_GIT_BRANCH; got $GIT_BRANCH"
fi
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  fail "tracked clean repo files are modified; refusing reader-disabled eval"
fi

log "host=$(hostname) job=${SLURM_JOB_ID:-none}"
log "target=$TARGET_UPPER"
log "repo=$REPO"
log "git_branch=$GIT_BRANCH git_head=$GIT_HEAD"
log "config=$CONFIG"
log "run_root=$RUN_ROOT"
log "work_dir=$WORK_DIR"
log "final_work_dir=$FINAL_WORK_DIR"
log "scope=${TARGET_UPPER}_READER_DISABLED_EVAL tools_test=true tools_train=false detector_map_reporting=true same_head_control=true exact_uniform_reader_override=true no_claims=true"

test -f "$CONFIG" || fail "missing reviewed config: $CONFIG"
test -f "$THUMOS14_ANNOTATION_PATH" || fail "missing annotation: $THUMOS14_ANNOTATION_PATH"
test -f "$THUMOS14_CLASS_MAP" || fail "missing class map: $THUMOS14_CLASS_MAP"
test -d "$THUMOS14_TEST_DATA_PATH" || fail "missing test data dir: $THUMOS14_TEST_DATA_PATH"

log "dumping and auditing resolved config=$RESOLVED_CONFIG_DUMP"
python - "$CONFIG" "$RESOLVED_CONFIG_DUMP" "$TARGET_UPPER" "$GATE_NAME" "$DISABLED_OLD_GATE_NAME" "$DECISION" <<'PY'
import sys
from pathlib import Path

from mmengine.config import Config
from opentad.utils.training_guard import assert_detector_training_allowed

config_path, dump_path, target, gate_name, disabled_old_gate_name, decision = sys.argv[1:7]
cfg = Config.fromfile(config_path)
Path(dump_path).write_text(cfg.pretty_text + "\n", encoding="utf-8")
assert cfg.get(disabled_old_gate_name) is None
gate = cfg.get(gate_name)
assert gate.stage == f"{target}_reader_disabled_eval_candidate", gate
assert gate.reader_disabled_eval_candidate is True, gate
assert gate.post_train_eval_candidate is True, gate
assert gate.allow_tools_train is False, gate
assert gate.allow_tools_test is True, gate
assert gate.allow_detector_map is True, gate
assert tuple(gate.allowed_entrypoints) == ("tools/test.py",), gate
assert tuple(gate.entrypoint_gate_context.allowed_decisions) == (decision,), gate
assert "reader_disabled_exact_uniform_eval_override" in gate.allowed_checks, gate
assert cfg.inference.load_from_raw_predictions is False
assert cfg.inference.save_raw_prediction is False
override = cfg.model.pc_ot_mras_reader_eval_override
assert override.enabled is True, override
assert override.mode == "exact_uniform", override
assert override.num_slots == 384, override
assert cfg.model.pc_ot_mras_reader.type == "PCOTMRASReader"
assert cfg.model.neck.type == "PCOTMRASDetectorBridge"
if target == "R18":
    assert cfg.model.pc_ot_mras_reader_aux_loss.enabled is True
try:
    assert_detector_training_allowed(cfg, entrypoint="tools/train.py")
except RuntimeError:
    pass
else:
    raise SystemExit(f"{target} reader-disabled eval config must reject tools/train.py")
print(f"ctf_bdi_pc_ot_mras_{target.lower()}_reader_disabled_eval_config_preflight=PASS")
PY
RESOLVED_CONFIG_SHA256="$(sha256sum "$RESOLVED_CONFIG_DUMP" | awk '{print $1}')"
log "resolved_config_sha256=$RESOLVED_CONFIG_SHA256"

mapfile -t CONFIG_DEP_FILES < <(python - "$CONFIG" <<'PY'
import ast
import sys
from pathlib import Path

seen = []
def visit(path):
    path = Path(path).resolve()
    if path in seen:
        return
    seen.append(path)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_base_" for t in node.targets):
            value = ast.literal_eval(node.value)
            for base in ([value] if isinstance(value, str) else list(value)):
                visit(path.parent / base)
visit(Path(sys.argv[1]))
for path in seen:
    print(path.relative_to(Path.cwd()).as_posix())
PY
)
sha256sum \
  "${CONFIG_DEP_FILES[@]}" \
  "$RESOLVED_CONFIG_DUMP" \
  scripts/run_ctf_bdi_pc_ot_mras_reader_disabled_eval_n16r4.sbatch \
  tools/test.py \
  opentad/cores/test_engine.py \
  opentad/utils/training_guard.py \
  opentad/models/detectors/actionformer.py \
  opentad/models/selectors/pc_ot_mras_reader.py \
  opentad/models/necks/pc_ot_mras_detector_bridge.py \
  opentad/models/dense_heads/native_irregular_area_head_p2.py \
  opentad/models/utils/pc_ot_mras_raw_prediction_guard.py \
  tests/test_pc_ot_mras_r17_r18_post_train_eval_gate.py \
  > "$SHA_MANIFEST"
ACTIVE_MANIFEST_SHA256="$(sha256sum "$SHA_MANIFEST" | awk '{print $1}')"
log "active_sha256_manifest_sha256=$ACTIVE_MANIFEST_SHA256"

python -m py_compile tools/test.py opentad/cores/test_engine.py opentad/utils/training_guard.py tests/test_pc_ot_mras_r17_r18_post_train_eval_gate.py
python -m pytest tests/test_pc_ot_mras_r17_r18_post_train_eval_gate.py -q

if [ "$PRECHECK_ONLY" = "1" ]; then
  write_summary "PASS" "${TARGET_UPPER}_READER_DISABLED_EVAL_PRECHECK_ONLY_PASS_NO_TEST" "PRECHECK_ONLY=1; exiting before tools/test.py"
  exit 0
fi

test -f "$EVAL_CHECKPOINT" || fail "missing ${TARGET_UPPER} checkpoint: $EVAL_CHECKPOINT"
ACTUAL_CKPT_SHA256="$(sha256sum "$EVAL_CHECKPOINT" | awk '{print $1}')"
if [ "$ACTUAL_CKPT_SHA256" != "$EVAL_CHECKPOINT_SHA256" ]; then
  fail "${TARGET_UPPER} checkpoint SHA256 mismatch: expected $EVAL_CHECKPOINT_SHA256 got $ACTUAL_CKPT_SHA256"
fi
ACTUAL_GATE_SHA256="$(sha256sum "$GATE_JSON" | awk '{print $1}')"
if [ "$ACTUAL_GATE_SHA256" != "$GATE_SHA256" ]; then
  fail "${TARGET_UPPER} reader-disabled eval gate SHA256 mismatch: expected $GATE_SHA256 got $ACTUAL_GATE_SHA256"
fi

python - "$GATE_JSON" "$ACTIVE_MANIFEST_SHA256" "$RESOLVED_CONFIG_SHA256" "$EVAL_CHECKPOINT_SHA256" "$DECISION" "$TARGET_UPPER" <<'PY'
import json
import sys

gate = json.load(open(sys.argv[1], "r", encoding="utf-8"))
if gate.get("decision") != sys.argv[5]:
    raise SystemExit(f"{sys.argv[6]} gate decision does not allow reader-disabled eval")
if gate.get("active_sha256_manifest_sha256") != sys.argv[2]:
    raise SystemExit(f"{sys.argv[6]} gate active manifest sha256 mismatch")
expected_resolved = gate.get("resolved_config_sha256") or gate.get("expected_resolved_config_sha256")
if expected_resolved != sys.argv[3]:
    raise SystemExit(f"{sys.argv[6]} gate resolved config sha256 mismatch")
if gate.get("checkpoint_sha256") != sys.argv[4]:
    raise SystemExit(f"{sys.argv[6]} gate checkpoint sha256 mismatch")
if not (gate.get("completed_training_evidence") is True or gate.get("recorded_stop_continue_decision") is True):
    raise SystemExit(f"{sys.argv[6]} gate requires completed training evidence or recorded stop/continue decision")
if gate.get("reader_disabled_eval") is not True:
    raise SystemExit(f"{sys.argv[6]} gate requires reader_disabled_eval=true")
if gate.get("same_head_control") is not True:
    raise SystemExit(f"{sys.argv[6]} gate requires same_head_control=true")
if gate.get("exact_uniform_reader_override") is not True:
    raise SystemExit(f"{sys.argv[6]} gate requires exact_uniform_reader_override=true")
for key in ("tools_train", "raw_prediction_cache", "paper_claim", "runtime_flops_claim", "deploy_claim", "dynamic_budget_claim"):
    if gate.get(key) is True:
        raise SystemExit(f"{sys.argv[6]} gate must not enable {key}")
print(f"ctf_bdi_pc_ot_mras_{sys.argv[6].lower()}_reader_disabled_eval_gate=PASS")
PY

mkdir -p "$WORK_DIR"
export OPENTAD_PCOTMRAS_ENTRYPOINT_GATE_JSON="$GATE_JSON"
export OPENTAD_PCOTMRAS_ENTRYPOINT_GATE_SHA256="$GATE_SHA256"
export OPENTAD_PCOTMRAS_ACTIVE_MANIFEST_SHA256="$ACTIVE_MANIFEST_SHA256"
export OPENTAD_PCOTMRAS_RESOLVED_CONFIG_SHA256="$RESOLVED_CONFIG_SHA256"
torchrun --nproc_per_node=1 --master_port="$MASTER_PORT" \
  tools/test.py "$CONFIG" --checkpoint "$EVAL_CHECKPOINT" --id "$EVAL_ID" --seed "$SEED" \
  --cfg-options \
  work_dir="$WORK_DIR" \
  dataset.test.ann_file="$THUMOS14_ANNOTATION_PATH" \
  dataset.test.class_map="$THUMOS14_CLASS_MAP" \
  dataset.test.data_path="$THUMOS14_TEST_DATA_PATH" \
  evaluation.ground_truth_filename="$THUMOS14_ANNOTATION_PATH" \
  2>&1 | tee "$EVAL_STDOUT"

grep -q "Testing Over" "$EVAL_STDOUT" || fail "${TARGET_UPPER} reader-disabled eval stdout does not show Testing Over"
if grep -Eiq "Average-mAP|mAP at tIoU|average_mAP|mAP@" "$EVAL_STDOUT"; then
  log "eval_metric_stdout=FOUND"
else
  fail "${TARGET_UPPER} reader-disabled eval stdout does not contain evaluator mAP evidence"
fi
RESULT_DETECTION_JSON="$FINAL_WORK_DIR/result_detection.json"
if [ -f "$RESULT_DETECTION_JSON" ]; then
  log "result_detection_json=$RESULT_DETECTION_JSON"
else
  log "result_detection_json=ABSENT_EXPECTED_WHEN_post_processing.save_dict_FALSE; using evaluator stdout mAP as artifact"
fi
write_summary "PASS" "${TARGET_UPPER}_READER_DISABLED_EVAL_PASS_MAP_REPORTING_ONLY_NO_CLAIMS" "tools/test.py completed with exact-uniform reader-disabled payload; Testing Over and detector mAP were found in stdout/log with no paper/deploy claims"
log "${TARGET_UPPER}_READER_DISABLED_EVAL_PASS_MAP_REPORTING_ONLY_NO_CLAIMS"
