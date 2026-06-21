#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH -J pcot_r18eval
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
    echo "[pc-ot-mras-r18-post-eval] $(date -Is) ERROR refusing dirty historical OpenTAD_BATA_Clean path: $REPO" >&2
    exit 2
    ;;
esac
if [ ! -d "$REPO" ]; then
  echo "[pc-ot-mras-r18-post-eval] $(date -Is) ERROR missing clean PC-OT-MRAS repo: $REPO" >&2
  exit 2
fi
cd "$REPO"

CONFIG_REVIEWED="configs/adatad/thumos/ctf_bdi_pc_ot_mras_r18_post_train_eval_candidate.py"
CONFIG="${CONFIG:-$CONFIG_REVIEWED}"
PRECHECK_ONLY="${PRECHECK_ONLY:-1}"
ALLOW_R18_POST_TRAIN_EVAL="${ALLOW_R18_POST_TRAIN_EVAL:-0}"
ALLOW_R18_TOOLS_TRAIN="${ALLOW_R18_TOOLS_TRAIN:-0}"
ALLOW_R18_RAW_PREDICTION_CACHE="${ALLOW_R18_RAW_PREDICTION_CACHE:-0}"
ALLOW_R18_RUNTIME_OR_DEPLOY_CLAIM="${ALLOW_R18_RUNTIME_OR_DEPLOY_CLAIM:-0}"
R18_POST_TRAIN_EVAL_GATE_JSON="${R18_POST_TRAIN_EVAL_GATE_JSON:-}"
R18_POST_TRAIN_EVAL_GATE_SHA256="${R18_POST_TRAIN_EVAL_GATE_SHA256:-}"
R18_EVAL_CHECKPOINT="${R18_EVAL_CHECKPOINT:-}"
R18_EVAL_CHECKPOINT_SHA256="${R18_EVAL_CHECKPOINT_SHA256:-}"
EVAL_ID="${EVAL_ID:-0}"
SEED="${SEED:-20260619}"
SLURM_JOB_ID_FOR_PORT="${SLURM_JOB_ID:-0}"
DEFAULT_MASTER_PORT=$((35000 + (SLURM_JOB_ID_FOR_PORT % 20000)))
REQUESTED_MASTER_PORT="${MASTER_PORT:-}"
MASTER_PORT_SOURCE="default_from_slurm_job_id"
if [ -n "$REQUESTED_MASTER_PORT" ]; then
  MASTER_PORT="$REQUESTED_MASTER_PORT"
  MASTER_PORT_SOURCE="environment"
else
  MASTER_PORT="$DEFAULT_MASTER_PORT"
fi
RUN_TAG="${RUN_TAG:-ctf_bdi_pc_ot_mras_r18_post_train_eval_${SLURM_JOB_ID:-nojob}_$(date '+%Y%m%d_%H%M%S')}"

case "$RUN_TAG" in
  *..*|*/*|*\\*|*[[:space:]]*|*[\;\&\|\>\<\`\"\'$]*)
    echo "[pc-ot-mras-r18-post-eval] $(date -Is) ERROR RUN_TAG must be a simple path-safe name; got $RUN_TAG" >&2
    exit 2
    ;;
esac

RUN_ROOT="$REPO/logs/$RUN_TAG"
RUN_LOG="$RUN_ROOT/${RUN_TAG}.log"
EVAL_STDOUT="$RUN_ROOT/${RUN_TAG}.tools_test.stdout.log"
SHA_MANIFEST="$RUN_ROOT/active_sha256_manifest.txt"
RESOLVED_CONFIG_DUMP="$RUN_ROOT/resolved_config.py"
SUMMARY_JSON="$RUN_ROOT/r18_post_train_eval_summary.json"
WORK_DIR_BASE="$RUN_ROOT/eval_workdir"
WORK_DIR="$WORK_DIR_BASE/gpu1_id${EVAL_ID}"

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
  echo "[pc-ot-mras-r18-post-eval] $(date -Is) $*"
}

fail() {
  log "ERROR $*"
  exit 2
}

write_summary() {
  local status="$1"
  local decision="$2"
  local reason="$3"
  python - "$SUMMARY_JSON" "$status" "$decision" "$reason" "$RUN_ROOT" "$CONFIG" "$WORK_DIR" "$R18_EVAL_CHECKPOINT" <<'PY'
import json
import os
import socket
import sys
from pathlib import Path

summary_path, status, decision, reason, run_root, config, work_dir, checkpoint = sys.argv[1:9]
payload = {
    "status": status,
    "decision": decision,
    "reason": reason,
    "stage": "R18_POST_TRAIN_EVAL",
    "route": "CTF-BDI/PC-OT-MRAS",
    "host": socket.gethostname(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "run_root": run_root,
    "config": config,
    "work_dir": work_dir,
    "checkpoint": checkpoint,
    "claim_boundaries": {
        "tools_test_only": True,
        "post_train_detector_map_reporting_allowed": True,
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
    write_summary "FAIL" "R18_POST_TRAIN_EVAL_FAILED" "launcher exited before successful completion"
  fi
}
trap on_exit EXIT

case "$PRECHECK_ONLY" in
  0|1) ;;
  *) fail "PRECHECK_ONLY must be exactly 0 or 1; got $PRECHECK_ONLY" ;;
esac
case "$ALLOW_R18_TOOLS_TRAIN" in
  0) ;;
  *) fail "tools/train.py is not part of this R18 post-train eval launcher" ;;
esac
case "$ALLOW_R18_RAW_PREDICTION_CACHE" in
  0) ;;
  *) fail "raw prediction/cache shortcuts are forbidden for R18 post-train eval" ;;
esac
case "$ALLOW_R18_RUNTIME_OR_DEPLOY_CLAIM" in
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
  case "$ALLOW_R18_POST_TRAIN_EVAL" in
    1) ;;
    *) fail "PRECHECK_ONLY=0 requires ALLOW_R18_POST_TRAIN_EVAL=1 after review" ;;
  esac
  if [ -z "$R18_POST_TRAIN_EVAL_GATE_JSON" ] || [ -z "$R18_POST_TRAIN_EVAL_GATE_SHA256" ]; then
    fail "PRECHECK_ONLY=0 requires R18_POST_TRAIN_EVAL_GATE_JSON and R18_POST_TRAIN_EVAL_GATE_SHA256"
  fi
  if [ -z "$R18_EVAL_CHECKPOINT" ] || [ -z "$R18_EVAL_CHECKPOINT_SHA256" ]; then
    fail "PRECHECK_ONLY=0 requires R18_EVAL_CHECKPOINT and R18_EVAL_CHECKPOINT_SHA256"
  fi
else
  case "$ALLOW_R18_POST_TRAIN_EVAL" in
    0) ;;
    *) fail "PRECHECK_ONLY=1 must not set ALLOW_R18_POST_TRAIN_EVAL=1" ;;
  esac
fi
if [ -n "${CHECKPOINT:-}" ] || [ -n "${LOAD_FROM:-}" ] || [ -n "${RESUME:-}" ]; then
  fail "generic CHECKPOINT/LOAD_FROM/RESUME env shortcuts are forbidden; use R18_EVAL_CHECKPOINT"
fi
if [ -n "${LOAD_FROM_RAW_PREDICTIONS:-}" ] || [ -n "${RAW_PREDICTION_CACHE:-}" ] || [ -n "${PREDICTION_CACHE:-}" ]; then
  fail "raw prediction/cache environment shortcuts are forbidden for R18 post-train eval"
fi
if [ -n "${CFG_OPTIONS:-}" ] || [ -n "${EXTRA_CFG_OPTIONS:-}" ]; then
  fail "arbitrary cfg-options are forbidden for R18 post-train eval"
fi
EXPECTED_GIT_BRANCH="${OPENTAD_PCOTMRAS_EXPECTED_BRANCH:-codex/pcotmras-r16-r18-r20-clean}"
GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)" || fail "clean repo must be a git checkout"
GIT_HEAD="$(git rev-parse HEAD 2>/dev/null)" || fail "clean repo HEAD is unavailable"
if [ "$GIT_BRANCH" != "$EXPECTED_GIT_BRANCH" ]; then
  fail "expected branch $EXPECTED_GIT_BRANCH; got $GIT_BRANCH"
fi
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  fail "tracked clean repo files are modified; refusing eval"
fi

log "host=$(hostname) job=${SLURM_JOB_ID:-none}"
log "repo=$REPO"
log "git_branch=$GIT_BRANCH git_head=$GIT_HEAD"
log "config=$CONFIG"
log "run_root=$RUN_ROOT"
log "work_dir=$WORK_DIR"
log "scope=R18_POST_TRAIN_EVAL tools_test=true tools_train=false detector_map_reporting=true no_claims=true"

test -f "$CONFIG" || fail "missing reviewed config: $CONFIG"
test -f "$THUMOS14_ANNOTATION_PATH" || fail "missing annotation: $THUMOS14_ANNOTATION_PATH"
test -f "$THUMOS14_CLASS_MAP" || fail "missing class map: $THUMOS14_CLASS_MAP"
test -d "$THUMOS14_TEST_DATA_PATH" || fail "missing test data dir: $THUMOS14_TEST_DATA_PATH"

log "dumping and auditing resolved config=$RESOLVED_CONFIG_DUMP"
python - "$CONFIG" "$RESOLVED_CONFIG_DUMP" <<'PY'
import sys
from pathlib import Path

from mmengine.config import Config
from opentad.utils.training_guard import assert_detector_training_allowed

cfg = Config.fromfile(sys.argv[1])
Path(sys.argv[2]).write_text(cfg.pretty_text + "\n", encoding="utf-8")
assert cfg.r18_pc_ot_mras_aux_diag_gate is None
gate = cfg.r18_pc_ot_mras_post_train_eval_gate
assert gate.stage == "R18_post_train_eval_candidate", gate
assert gate.allow_tools_train is False, gate
assert gate.allow_tools_test is True, gate
assert gate.allow_detector_map is True, gate
assert tuple(gate.allowed_entrypoints) == ("tools/test.py",), gate
assert cfg.model.pc_ot_mras_reader_aux_loss.enabled is True
assert cfg.inference.load_from_raw_predictions is False
assert cfg.inference.save_raw_prediction is False
try:
    assert_detector_training_allowed(cfg, entrypoint="tools/train.py")
except RuntimeError:
    pass
else:
    raise SystemExit("R18 post-train eval config must reject tools/train.py")
print("ctf_bdi_pc_ot_mras_r18_post_train_eval_config_preflight=PASS")
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
  scripts/run_ctf_bdi_pc_ot_mras_r18_post_train_eval_n16r4.sbatch \
  tools/test.py \
  opentad/cores/test_engine.py \
  opentad/utils/training_guard.py \
  opentad/models/detectors/actionformer.py \
  opentad/models/losses/pc_ot_mras_auxiliary_losses.py \
  opentad/models/utils/pc_ot_mras_raw_prediction_guard.py \
  tests/test_pc_ot_mras_r17_r18_post_train_eval_gate.py \
  > "$SHA_MANIFEST"
ACTIVE_MANIFEST_SHA256="$(sha256sum "$SHA_MANIFEST" | awk '{print $1}')"
log "active_sha256_manifest_sha256=$ACTIVE_MANIFEST_SHA256"

python -m py_compile tools/test.py opentad/cores/test_engine.py opentad/utils/training_guard.py tests/test_pc_ot_mras_r17_r18_post_train_eval_gate.py
python -m pytest tests/test_pc_ot_mras_r17_r18_post_train_eval_gate.py -q

if [ "$PRECHECK_ONLY" = "1" ]; then
  write_summary "PASS" "R18_POST_TRAIN_EVAL_PRECHECK_ONLY_PASS_NO_TEST" "PRECHECK_ONLY=1; exiting before tools/test.py"
  exit 0
fi

test -f "$R18_EVAL_CHECKPOINT" || fail "missing R18 checkpoint: $R18_EVAL_CHECKPOINT"
ACTUAL_CKPT_SHA256="$(sha256sum "$R18_EVAL_CHECKPOINT" | awk '{print $1}')"
if [ "$ACTUAL_CKPT_SHA256" != "$R18_EVAL_CHECKPOINT_SHA256" ]; then
  fail "R18 checkpoint SHA256 mismatch: expected $R18_EVAL_CHECKPOINT_SHA256 got $ACTUAL_CKPT_SHA256"
fi
ACTUAL_GATE_SHA256="$(sha256sum "$R18_POST_TRAIN_EVAL_GATE_JSON" | awk '{print $1}')"
if [ "$ACTUAL_GATE_SHA256" != "$R18_POST_TRAIN_EVAL_GATE_SHA256" ]; then
  fail "R18 post-train eval gate SHA256 mismatch: expected $R18_POST_TRAIN_EVAL_GATE_SHA256 got $ACTUAL_GATE_SHA256"
fi

python - "$R18_POST_TRAIN_EVAL_GATE_JSON" "$ACTIVE_MANIFEST_SHA256" "$RESOLVED_CONFIG_SHA256" "$R18_EVAL_CHECKPOINT_SHA256" <<'PY'
import json
import sys

gate = json.load(open(sys.argv[1], "r", encoding="utf-8"))
if gate.get("decision") != "ALLOW_R18_POST_TRAIN_EVAL":
    raise SystemExit("R18 gate decision does not allow post-train eval")
if gate.get("active_sha256_manifest_sha256") != sys.argv[2]:
    raise SystemExit("R18 gate active manifest sha256 mismatch")
expected_resolved = gate.get("resolved_config_sha256") or gate.get("expected_resolved_config_sha256")
if expected_resolved != sys.argv[3]:
    raise SystemExit("R18 gate resolved config sha256 mismatch")
if gate.get("checkpoint_sha256") != sys.argv[4]:
    raise SystemExit("R18 gate checkpoint sha256 mismatch")
if not (gate.get("completed_training_evidence") is True or gate.get("recorded_stop_continue_decision") is True):
    raise SystemExit("R18 gate requires completed training evidence or recorded stop/continue decision")
for key in ("tools_train", "raw_prediction_cache", "paper_claim", "runtime_flops_claim", "deploy_claim", "dynamic_budget_claim"):
    if gate.get(key) is True:
        raise SystemExit(f"R18 gate must not enable {key}")
print("ctf_bdi_pc_ot_mras_r18_post_train_eval_gate=PASS")
PY

mkdir -p "$WORK_DIR"
export OPENTAD_PCOTMRAS_ENTRYPOINT_GATE_JSON="$R18_POST_TRAIN_EVAL_GATE_JSON"
export OPENTAD_PCOTMRAS_ENTRYPOINT_GATE_SHA256="$R18_POST_TRAIN_EVAL_GATE_SHA256"
export OPENTAD_PCOTMRAS_ACTIVE_MANIFEST_SHA256="$ACTIVE_MANIFEST_SHA256"
export OPENTAD_PCOTMRAS_RESOLVED_CONFIG_SHA256="$RESOLVED_CONFIG_SHA256"
torchrun --nproc_per_node=1 --master_port="$MASTER_PORT" \
  tools/test.py "$CONFIG" --checkpoint "$R18_EVAL_CHECKPOINT" --id "$EVAL_ID" --seed "$SEED" \
  --cfg-options \
  work_dir="$WORK_DIR" \
  dataset.test.ann_file="$THUMOS14_ANNOTATION_PATH" \
  dataset.test.class_map="$THUMOS14_CLASS_MAP" \
  dataset.test.data_path="$THUMOS14_TEST_DATA_PATH" \
  evaluation.ground_truth_filename="$THUMOS14_ANNOTATION_PATH" \
  2>&1 | tee "$EVAL_STDOUT"

grep -q "Testing Over" "$EVAL_STDOUT" || fail "R18 post-train eval stdout does not show Testing Over"
test -f "$WORK_DIR/result_detection.json" || fail "missing expected post-train eval artifact: $WORK_DIR/result_detection.json"
write_summary "PASS" "R18_POST_TRAIN_EVAL_PASS_MAP_REPORTING_ONLY_NO_CLAIMS" "tools/test.py completed with result_detection.json as post-train eval artifact"
log "R18_POST_TRAIN_EVAL_PASS_MAP_REPORTING_ONLY_NO_CLAIMS"
