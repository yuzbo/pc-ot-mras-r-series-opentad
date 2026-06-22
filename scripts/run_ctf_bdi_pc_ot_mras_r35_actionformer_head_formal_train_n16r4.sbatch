#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH -J pcot_r35afh
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
    echo "[pc-ot-mras-r35-actionformer-head] $(date -Is) ERROR refusing dirty historical OpenTAD_BATA_Clean path: $REPO" >&2
    exit 2
    ;;
esac
if [ ! -d "$REPO" ]; then
  echo "[pc-ot-mras-r35-actionformer-head] $(date -Is) ERROR missing clean PC-OT-MRAS repo: $REPO" >&2
  exit 2
fi
cd "$REPO"

CONFIG_REVIEWED="configs/adatad/thumos/ctf_bdi_pc_ot_mras_r35_actionformer_head_formal_train_candidate.py"
CONFIG="${CONFIG:-$CONFIG_REVIEWED}"
PRECHECK_ONLY="${PRECHECK_ONLY:-1}"
ALLOW_R35_ACTIONFORMER_HEAD_FORMAL_TRAIN="${ALLOW_R35_ACTIONFORMER_HEAD_FORMAL_TRAIN:-0}"
ALLOW_R35_TOOLS_TEST="${ALLOW_R35_TOOLS_TEST:-0}"
ALLOW_R35_DETECTOR_MAP_CLAIM="${ALLOW_R35_DETECTOR_MAP_CLAIM:-0}"
ALLOW_R35_RAW_PREDICTION_CACHE="${ALLOW_R35_RAW_PREDICTION_CACHE:-0}"
R35_ACTIONFORMER_HEAD_FORMAL_GATE_JSON="${R35_ACTIONFORMER_HEAD_FORMAL_GATE_JSON:-}"
R35_ACTIONFORMER_HEAD_FORMAL_GATE_SHA256="${R35_ACTIONFORMER_HEAD_FORMAL_GATE_SHA256:-}"
TRAIN_ID="${TRAIN_ID:-0}"
SEED="${SEED:-20260622}"
SLURM_JOB_ID_FOR_PORT="${SLURM_JOB_ID:-0}"
DEFAULT_MASTER_PORT=$((34000 + (SLURM_JOB_ID_FOR_PORT % 20000)))
REQUESTED_MASTER_PORT="${MASTER_PORT:-}"
MASTER_PORT_SOURCE="default_from_slurm_job_id"
if [ -n "$REQUESTED_MASTER_PORT" ]; then
  MASTER_PORT="$REQUESTED_MASTER_PORT"
  MASTER_PORT_SOURCE="environment"
else
  MASTER_PORT="$DEFAULT_MASTER_PORT"
fi
RUN_TAG="${RUN_TAG:-ctf_bdi_pc_ot_mras_r35_actionformer_head_formal_${SLURM_JOB_ID:-nojob}_$(date '+%Y%m%d_%H%M%S')}"

case "$RUN_TAG" in
  *..*|*/*|*\\*|*[[:space:]]*|*[\;\&\|\>\<\`\"\'$]*)
    echo "[pc-ot-mras-r35-actionformer-head] $(date -Is) ERROR RUN_TAG must be a simple path-safe name; got $RUN_TAG" >&2
    exit 2
    ;;
esac

LOG_DIR="$REPO/logs"
RUN_ROOT="$LOG_DIR/$RUN_TAG"
RUN_LOG="$RUN_ROOT/${RUN_TAG}.log"
TRAIN_STDOUT="$RUN_ROOT/${RUN_TAG}.train.stdout.log"
SHA_MANIFEST="$RUN_ROOT/active_sha256_manifest.txt"
RESOLVED_CONFIG_DUMP="$RUN_ROOT/resolved_config.py"
SUMMARY_JSON="$RUN_ROOT/r35_actionformer_head_formal_train_summary.json"
PRETRAINED_PATH_REVIEWED="pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"
PRETRAINED_PATH="${PRETRAINED_PATH:-$PRETRAINED_PATH_REVIEWED}"
WORK_DIR_BASE="$RUN_ROOT/train_workdir"
WORK_DIR="$WORK_DIR_BASE/gpu1_id${TRAIN_ID}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export YUZIBO_ROOT
export OPENTAD_PCOTMRAS_CLEAN_ROOT="$REPO"
export THUMOS14_ANNOTATION_PATH="${THUMOS14_ANNOTATION_PATH:-$YUZIBO_ROOT/thumos14/annotations/thumos_14_anno.json}"
export THUMOS14_CLASS_MAP="${THUMOS14_CLASS_MAP:-$YUZIBO_ROOT/thumos14/annotations/category_idx.txt}"
export THUMOS14_TRAIN_DATA_PATH="${THUMOS14_TRAIN_DATA_PATH:-$YUZIBO_ROOT/thumos14/train}"
export THUMOS14_TEST_DATA_PATH="${THUMOS14_TEST_DATA_PATH:-$YUZIBO_ROOT/thumos14/test}"

mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_LOG") 2>&1

log() {
  echo "[pc-ot-mras-r35-actionformer-head] $(date -Is) $*"
}

fail() {
  log "ERROR $*"
  exit 2
}

write_summary() {
  local status="$1"
  local decision="$2"
  local reason="$3"
  python - "$SUMMARY_JSON" "$status" "$decision" "$reason" "$RUN_ROOT" "$CONFIG" "$WORK_DIR" <<'PY'
import json
import os
import socket
import sys
from pathlib import Path

summary_path, status, decision, reason, run_root, config, work_dir = sys.argv[1:8]
payload = {
    "status": status,
    "decision": decision,
    "reason": reason,
    "stage": "R35_ACTIONFORMER_HEAD_FORMAL_TRAIN",
    "route": "CTF-BDI/PC-OT-MRAS",
    "host": socket.gethostname(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "run_root": run_root,
    "config": config,
    "work_dir": work_dir,
    "claim_boundaries": {
        "tools_train_only": True,
        "train_time_validation_map_allowed": True,
        "tools_test_direct_entrypoint": False,
        "raw_prediction_cache": False,
        "metric_claim_allowed": False,
        "runtime_or_flops_claim_allowed": False,
        "deploy_claim_allowed": False,
        "paper_claim_allowed": False,
        "attribution_control_only": True,
    },
}
Path(summary_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  log "summary_json=$SUMMARY_JSON"
}

on_exit() {
  local status="$?"
  if [ "$status" -ne 0 ] && [ ! -f "$SUMMARY_JSON" ]; then
    write_summary "FAIL" "R35_ACTIONFORMER_HEAD_FORMAL_FAILED" "launcher exited before successful completion"
  fi
}
trap on_exit EXIT

case "$PRECHECK_ONLY" in
  0|1) ;;
  *) fail "PRECHECK_ONLY must be exactly 0 or 1; got $PRECHECK_ONLY" ;;
esac
case "$ALLOW_R35_TOOLS_TEST" in
  0) ;;
  *) fail "direct tools/test.py is not part of this R35 launcher" ;;
esac
case "$ALLOW_R35_DETECTOR_MAP_CLAIM" in
  0) ;;
  *) fail "detector mAP claims are not approved for this R35 launcher" ;;
esac
case "$ALLOW_R35_RAW_PREDICTION_CACHE" in
  0) ;;
  *) fail "raw prediction/cache shortcuts are forbidden for R35 formal training" ;;
esac
case "$MASTER_PORT" in
  ''|*[!0-9]*)
    if [ "$MASTER_PORT_SOURCE" = "environment" ]; then
      log "WARN ignoring invalid inherited MASTER_PORT=$MASTER_PORT; using default $DEFAULT_MASTER_PORT"
      MASTER_PORT="$DEFAULT_MASTER_PORT"
      MASTER_PORT_SOURCE="default_after_invalid_environment"
    else
      fail "MASTER_PORT must be numeric; got $MASTER_PORT"
    fi
    ;;
esac
case "$MASTER_PORT" in
  ''|*[!0-9]*) fail "normalized MASTER_PORT must be numeric; got $MASTER_PORT" ;;
esac
if [ "$MASTER_PORT" -lt 1024 ] || [ "$MASTER_PORT" -gt 65535 ]; then
  if [ "$MASTER_PORT_SOURCE" = "environment" ]; then
    log "WARN ignoring inherited MASTER_PORT=$MASTER_PORT outside [1024,65535]; using default $DEFAULT_MASTER_PORT"
    MASTER_PORT="$DEFAULT_MASTER_PORT"
    MASTER_PORT_SOURCE="default_after_invalid_environment"
  else
    fail "MASTER_PORT must be between 1024 and 65535; got $MASTER_PORT"
  fi
fi
if [ "$MASTER_PORT" -lt 1024 ] || [ "$MASTER_PORT" -gt 65535 ]; then
  fail "normalized MASTER_PORT must be between 1024 and 65535; got $MASTER_PORT"
fi
case "$TRAIN_ID" in
  ''|*[!0-9]*) fail "TRAIN_ID must be numeric; got $TRAIN_ID" ;;
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
  fail "login-node debug must keep PRECHECK_ONLY=1; refusing training outside Slurm"
fi
if [ "$PRECHECK_ONLY" = "0" ]; then
  case "$ALLOW_R35_ACTIONFORMER_HEAD_FORMAL_TRAIN" in
    1) ;;
    *) fail "PRECHECK_ONLY=0 requires ALLOW_R35_ACTIONFORMER_HEAD_FORMAL_TRAIN=1 after a separate execution gate" ;;
  esac
  if [ -z "$R35_ACTIONFORMER_HEAD_FORMAL_GATE_JSON" ] || [ -z "$R35_ACTIONFORMER_HEAD_FORMAL_GATE_SHA256" ]; then
    fail "PRECHECK_ONLY=0 requires explicit R35_ACTIONFORMER_HEAD_FORMAL_GATE_JSON and R35_ACTIONFORMER_HEAD_FORMAL_GATE_SHA256"
  fi
else
  case "$ALLOW_R35_ACTIONFORMER_HEAD_FORMAL_TRAIN" in
    0) ;;
    *) fail "PRECHECK_ONLY=1 must not set ALLOW_R35_ACTIONFORMER_HEAD_FORMAL_TRAIN=1" ;;
  esac
fi
if [ -n "${CHECKPOINT:-}" ] || [ -n "${CHECKPOINT_SHA256:-}" ] || [ -n "${LOAD_FROM:-}" ] || [ -n "${RESUME:-}" ]; then
  fail "external checkpoint/load/resume environment shortcuts are forbidden for R35"
fi
if [ "$PRETRAINED_PATH" != "$PRETRAINED_PATH_REVIEWED" ]; then
  fail "PRETRAINED_PATH override is forbidden for R35; reviewed path is $PRETRAINED_PATH_REVIEWED"
fi
if [ -n "${LOAD_FROM_RAW_PREDICTIONS:-}" ] || [ -n "${RAW_PREDICTION_CACHE:-}" ] || [ -n "${PREDICTION_CACHE:-}" ]; then
  fail "raw prediction/cache environment shortcuts are forbidden for R35"
fi
if [ -n "${CFG_OPTIONS:-}" ] || [ -n "${EXTRA_CFG_OPTIONS:-}" ]; then
  fail "arbitrary cfg-options are forbidden for R35"
fi
EXPECTED_GIT_BRANCH="${OPENTAD_PCOTMRAS_EXPECTED_BRANCH:-codex/pcotmras-r16-r18-r20-clean}"
GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)" || fail "clean repo must be a git checkout"
GIT_HEAD="$(git rev-parse HEAD 2>/dev/null)" || fail "clean repo HEAD is unavailable"
if [ "$GIT_BRANCH" != "$EXPECTED_GIT_BRANCH" ]; then
  fail "expected branch $EXPECTED_GIT_BRANCH; got $GIT_BRANCH"
fi
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  fail "tracked clean repo files are modified; refusing execution"
fi

log "host=$(hostname) job=${SLURM_JOB_ID:-none}"
log "repo=$REPO"
log "git_branch=$GIT_BRANCH git_head=$GIT_HEAD"
log "config=$CONFIG"
log "run_root=$RUN_ROOT"
log "work_dir=$WORK_DIR"
log "seed=$SEED train_id=$TRAIN_ID master_port=$MASTER_PORT master_port_source=$MASTER_PORT_SOURCE default_master_port=$DEFAULT_MASTER_PORT"
log "precheck_only=$PRECHECK_ONLY allow_r35_actionformer_head_formal_train=$ALLOW_R35_ACTIONFORMER_HEAD_FORMAL_TRAIN"
log "scope=R35_ACTIONFORMER_HEAD tools_train=true train_validation_map=true direct_tools_test=false no_claims=true"
log "python=$(python -c 'import sys; print(sys.executable)')"

test -f "$CONFIG" || fail "missing reviewed config: $CONFIG"
test -f "$PRETRAINED_PATH" || fail "missing reviewed pretrained initialization file: $PRETRAINED_PATH"
test -f "$THUMOS14_ANNOTATION_PATH" || fail "missing annotation: $THUMOS14_ANNOTATION_PATH"
test -f "$THUMOS14_CLASS_MAP" || fail "missing class map: $THUMOS14_CLASS_MAP"
test -d "$THUMOS14_TRAIN_DATA_PATH" || fail "missing train data dir: $THUMOS14_TRAIN_DATA_PATH"
test -d "$THUMOS14_TEST_DATA_PATH" || fail "missing test data dir: $THUMOS14_TEST_DATA_PATH"

log "python/torch/cuda probe"
python - <<'PY'
import json
import torch

payload = {
    "torch_version": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": bool(torch.cuda.is_available()),
    "device_count": int(torch.cuda.device_count()),
}
print(json.dumps(payload, sort_keys=True))
if not payload["cuda_available"] or payload["device_count"] < 1:
    raise SystemExit("R35 formal training/precheck requires at least one visible CUDA device")
PY
nvidia-smi || fail "nvidia-smi failed"

if command -v ss >/dev/null 2>&1; then
  if ss -ltn | awk '{print $4}' | grep -Eq "[:.]${MASTER_PORT}$"; then
    fail "MASTER_PORT=$MASTER_PORT is already in use"
  fi
else
  log "WARN ss not available; skipping master port preflight"
fi

log "dumping and auditing resolved config=$RESOLVED_CONFIG_DUMP"
python - "$CONFIG" "$RESOLVED_CONFIG_DUMP" <<'PY'
import sys
from pathlib import Path

from mmengine.config import Config
from opentad.utils.training_guard import assert_detector_training_allowed


def require(condition, message):
    if not condition:
        raise SystemExit(str(message))


cfg = Config.fromfile(sys.argv[1])
Path(sys.argv[2]).write_text(cfg.pretty_text + "\n", encoding="utf-8")

require(cfg.r17_pc_ot_mras_formal_train_gate is None, "R17 gate must be cleared in R35 formal config")
gate = cfg.r35_pc_ot_mras_actionformer_head_gate
require(gate.stage == "R35_actionformer_head_formal_train_candidate", gate)
require(gate.launch_gate_passed is True, gate)
require(gate.allow_detector_training is True, gate)
require(gate.allow_tools_train is True, gate)
require(gate.allow_tools_test is False, gate)
require(gate.allow_train_validation_map is True, gate)
require(gate.allow_long_training is True, gate)
require(gate.metric_claim_allowed is False, gate)
require(gate.paper_claim_allowed is False, gate)
require(tuple(gate.allowed_entrypoints) == ("tools/train.py",), gate)
require(int(cfg.workflow.end_epoch) == 60, cfg.workflow)
require(int(cfg.workflow.val_start_epoch) == 40, cfg.workflow)
require(int(cfg.workflow.val_eval_interval) == 2, cfg.workflow)
require(int(cfg.workflow.checkpoint_interval) == 2, cfg.workflow)
require(cfg.inference.load_from_raw_predictions is False, cfg.inference)
require(cfg.inference.save_raw_prediction is False, cfg.inference)
require(cfg.model.pc_ot_mras_reader.type == "PCOTMRASReader", cfg.model.pc_ot_mras_reader)
require("pc_ot_mras_reader_aux_loss" not in cfg.model, cfg.model)
require(cfg.model.neck.type == "PCOTMRASDetectorBridge", cfg.model.neck)
require(cfg.model.rpn_head.type == "ActionFormerHead", cfg.model.rpn_head)
require("temporal_grid" not in cfg.model.rpn_head, cfg.model.rpn_head)
require("area_head" not in cfg.model.rpn_head, cfg.model.rpn_head)
require(list(cfg.model.rpn_head.prior_generator.strides) == [1, 2, 4, 8, 16, 32], cfg.model.rpn_head)
require(
    [tuple(item) for item in cfg.model.rpn_head.prior_generator.regression_range]
    == [(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10000)],
    cfg.model.rpn_head,
)
try:
    assert_detector_training_allowed(cfg, entrypoint="tools/test.py")
except RuntimeError:
    pass
else:
    raise SystemExit("R35 formal config must reject direct tools/test.py")

for split in ("train", "val", "test"):
    pipeline_text = repr(cfg.dataset[split].pipeline).lower()
    for forbidden in ("raw_prediction", "prediction_cache", "teacher", "oracle"):
        require(forbidden not in pipeline_text, f"{split} pipeline contains forbidden token {forbidden}")

print("ctf_bdi_pc_ot_mras_r35_actionformer_head_formal_config_preflight=PASS")
PY
RESOLVED_CONFIG_SHA256="$(sha256sum "$RESOLVED_CONFIG_DUMP" | awk '{print $1}')"
log "resolved_config_sha256=$RESOLVED_CONFIG_SHA256"

mapfile -t CONFIG_DEP_FILES < <(python - "$CONFIG" <<'PY'
import ast
import sys
from pathlib import Path

seen = []
root = Path(sys.argv[1]).resolve()

def visit(path):
    path = Path(path).resolve()
    if path in seen:
        return
    seen.append(path)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "_base_" for target in node.targets):
            continue
        value = ast.literal_eval(node.value)
        if isinstance(value, str):
            bases = [value]
        else:
            bases = list(value)
        for base in bases:
            visit(path.parent / base)

visit(root)
for path in seen:
    print(path.relative_to(Path.cwd()).as_posix())
PY
)
log "resolved_config_dependency_count=${#CONFIG_DEP_FILES[@]}"
log "writing active SHA256 manifest=$SHA_MANIFEST"
sha256sum \
  "${CONFIG_DEP_FILES[@]}" \
  "$RESOLVED_CONFIG_DUMP" \
  "$PRETRAINED_PATH" \
  scripts/run_ctf_bdi_pc_ot_mras_r35_actionformer_head_formal_train_n16r4.sbatch \
  tools/train.py \
  tools/test.py \
  opentad/cores/train_engine.py \
  opentad/utils/training_guard.py \
  opentad/models/detectors/actionformer.py \
  opentad/models/dense_heads/__init__.py \
  opentad/models/dense_heads/actionformer_head.py \
  opentad/models/necks/pc_ot_mras_detector_bridge.py \
  opentad/models/selectors/pc_ot_mras_reader.py \
  opentad/models/utils/pc_ot_mras_raw_prediction_guard.py \
  tests/test_pc_ot_mras_r35_actionformer_head_formal_train_config.py \
  > "$SHA_MANIFEST"
ACTIVE_MANIFEST_SHA256="$(sha256sum "$SHA_MANIFEST" | awk '{print $1}')"
log "active_sha256_manifest_sha256=$ACTIVE_MANIFEST_SHA256"

log "static py_compile"
python -m py_compile \
  tools/train.py \
  tools/test.py \
  opentad/cores/train_engine.py \
  opentad/utils/training_guard.py \
  opentad/models/detectors/actionformer.py \
  opentad/models/dense_heads/__init__.py \
  opentad/models/dense_heads/actionformer_head.py \
  opentad/models/necks/pc_ot_mras_detector_bridge.py \
  opentad/models/selectors/pc_ot_mras_reader.py \
  tests/test_pc_ot_mras_r35_actionformer_head_formal_train_config.py

log "focused R35 formal tests"
python -m pytest \
  tests/test_pc_ot_mras_r35_actionformer_head_formal_train_config.py \
  tests/test_pc_ot_mras_train_local_only_guard.py \
  -q

if [ "$PRECHECK_ONLY" = "1" ]; then
  write_summary "PASS" "R35_ACTIONFORMER_HEAD_FORMAL_PRECHECK_ONLY_PASS_NO_TRAIN" "PRECHECK_ONLY=1; exiting before tools/train.py"
  log "PRECHECK_ONLY=1; exiting before R35 tools/train formal run"
  exit 0
fi

ACTUAL_GATE_SHA256="$(sha256sum "$R35_ACTIONFORMER_HEAD_FORMAL_GATE_JSON" | awk '{print $1}')"
if [ "$ACTUAL_GATE_SHA256" != "$R35_ACTIONFORMER_HEAD_FORMAL_GATE_SHA256" ]; then
  fail "R35 execution gate SHA256 mismatch: expected $R35_ACTIONFORMER_HEAD_FORMAL_GATE_SHA256 got $ACTUAL_GATE_SHA256"
fi

python - "$R35_ACTIONFORMER_HEAD_FORMAL_GATE_JSON" "$ACTIVE_MANIFEST_SHA256" "$RESOLVED_CONFIG_SHA256" <<'PY'
import json
import sys

gate = json.load(open(sys.argv[1], "r", encoding="utf-8"))
expected_manifest = sys.argv[2]
if gate.get("decision") != "ALLOW_R35_ACTIONFORMER_HEAD_FORMAL_TRAIN":
    raise SystemExit("R35 execution gate decision does not allow ActionFormer-head formal train")
if gate.get("active_sha256_manifest_sha256") != expected_manifest:
    raise SystemExit("R35 execution gate active manifest sha256 mismatch")
expected_resolved = gate.get("resolved_config_sha256") or gate.get("expected_resolved_config_sha256")
if expected_resolved != sys.argv[3]:
    raise SystemExit("R35 execution gate resolved config sha256 mismatch")
for key in ("tools_test", "direct_tools_test", "detector_map", "metric_claim", "paper_claim", "runtime_flops_claim", "deploy_claim", "raw_prediction_cache"):
    if gate.get(key) is True:
        raise SystemExit(f"R35 execution gate must not enable {key}")
print("ctf_bdi_pc_ot_mras_r35_execution_gate=PASS_BOUND_TO_ACTIVE_MANIFEST")
PY

mkdir -p "$WORK_DIR"
export OPENTAD_PCOTMRAS_ENTRYPOINT_GATE_JSON="$R35_ACTIONFORMER_HEAD_FORMAL_GATE_JSON"
export OPENTAD_PCOTMRAS_ENTRYPOINT_GATE_SHA256="$R35_ACTIONFORMER_HEAD_FORMAL_GATE_SHA256"
export OPENTAD_PCOTMRAS_ACTIVE_MANIFEST_SHA256="$ACTIVE_MANIFEST_SHA256"
export OPENTAD_PCOTMRAS_RESOLVED_CONFIG_SHA256="$RESOLVED_CONFIG_SHA256"
log "launching R35 ActionFormerHead formal tools/train.py"
torchrun --nproc_per_node=1 --master_port="$MASTER_PORT" \
  tools/train.py "$CONFIG" --id "$TRAIN_ID" --seed "$SEED" \
  --cfg-options \
  work_dir="$WORK_DIR" \
  dataset.train.ann_file="$THUMOS14_ANNOTATION_PATH" \
  dataset.val.ann_file="$THUMOS14_ANNOTATION_PATH" \
  dataset.test.ann_file="$THUMOS14_ANNOTATION_PATH" \
  dataset.train.class_map="$THUMOS14_CLASS_MAP" \
  dataset.val.class_map="$THUMOS14_CLASS_MAP" \
  dataset.test.class_map="$THUMOS14_CLASS_MAP" \
  dataset.train.data_path="$THUMOS14_TRAIN_DATA_PATH" \
  dataset.val.data_path="$THUMOS14_TEST_DATA_PATH" \
  dataset.test.data_path="$THUMOS14_TEST_DATA_PATH" \
  evaluation.ground_truth_filename="$THUMOS14_ANNOTATION_PATH" \
  2>&1 | tee "$TRAIN_STDOUT"

grep -q "Training Over" "$TRAIN_STDOUT" || fail "train stdout does not show clean Training Over"
if grep -Eq "load_from_raw_predictions|RAW_PREDICTION_CACHE|PREDICTION_CACHE" "$TRAIN_STDOUT"; then
  fail "train stdout contains raw prediction/cache marker"
fi
if find "$WORK_DIR" "$RUN_ROOT" -maxdepth 4 -type f -name 'result_detection.json' -print -quit | grep -q .; then
  fail "unexpected result_detection.json found; direct tools/test.py artifact is outside this launcher"
fi

write_summary "PASS" "R35_ACTIONFORMER_HEAD_FORMAL_TRAIN_PASS_NO_DIRECT_TEST_NO_CLAIMS" "tools/train.py completed with train-time validation allowed; no direct tools/test.py or claims"
log "R35_ACTIONFORMER_HEAD_FORMAL_TRAIN_PASS_NO_DIRECT_TEST_NO_CLAIMS"
