#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH -J p2qr_exec
#SBATCH -o logs/%x-%j.out

set -eo pipefail

source /etc/profile >/dev/null 2>&1 || true
set -u

YUZIBO_ROOT="${YUZIBO_ROOT:-$HOME/run/yuzibo}"
P2QR_LAUNCHER_TEST_MODE="${P2QR_LAUNCHER_TEST_MODE:-0}"
if [ "$P2QR_LAUNCHER_TEST_MODE" != "1" ]; then
  module load cuda/11.8
  module load miniforge3/24.11
  source activate "$YUZIBO_ROOT/conda_envs/opentad"
fi

CLEAN_REPO_NAME="${OPENTAD_PCOTMRAS_CLEAN_REPO_NAME:-OpenTAD_PCOTMRAS_R16_R18_R20_Clean_20260619_1730}"
REPO="${OPENTAD_PCOTMRAS_CLEAN_ROOT:-$YUZIBO_ROOT/$CLEAN_REPO_NAME}"
case "$REPO" in
  *OpenTAD_BATA_Clean*)
    echo "[p2qr-short-smoke-exec] $(date -Is) ERROR refusing dirty historical OpenTAD_BATA_Clean path: $REPO" >&2
    exit 2
    ;;
esac
if [ ! -d "$REPO" ]; then
  echo "[p2qr-short-smoke-exec] $(date -Is) ERROR missing clean PC-OT-MRAS repo: $REPO" >&2
  exit 2
fi
cd "$REPO"

CONFIG_REVIEWED="configs/adatad/thumos/ctf_bdi_pc_ot_mras_p2_quality_rank_calibrator_v0_short_smoke_exec_candidate.py"
CONFIG="${CONFIG:-$CONFIG_REVIEWED}"
PRECHECK_ONLY="${PRECHECK_ONLY:-1}"
ALLOW_P2QR_SHORT_SMOKE="${ALLOW_P2QR_SHORT_SMOKE:-0}"
ALLOW_P2QR_SLURM="${ALLOW_P2QR_SLURM:-0}"
ALLOW_P2QR_GPU="${ALLOW_P2QR_GPU:-0}"
P2QR_SHORT_SMOKE_ONLY="${P2QR_SHORT_SMOKE_ONLY:-1}"
ALLOW_P2QR_TOOLS_TEST="${ALLOW_P2QR_TOOLS_TEST:-0}"
ALLOW_P2QR_DETECTOR_MAP="${ALLOW_P2QR_DETECTOR_MAP:-0}"
ALLOW_P2QR_LONG_TRAINING="${ALLOW_P2QR_LONG_TRAINING:-0}"
ALLOW_P2QR_CHECKPOINT_WRITE="${ALLOW_P2QR_CHECKPOINT_WRITE:-0}"
P2QR_SHORT_SMOKE_GATE_JSON="${P2QR_SHORT_SMOKE_GATE_JSON:-}"
P2QR_SHORT_SMOKE_GATE_SHA256="${P2QR_SHORT_SMOKE_GATE_SHA256:-}"
TRAIN_ID="${TRAIN_ID:-0}"
SEED="${SEED:-20260623}"
SLURM_JOB_ID_FOR_PORT="${SLURM_JOB_ID:-0}"
DEFAULT_MASTER_PORT=$((36000 + (SLURM_JOB_ID_FOR_PORT % 20000)))
REQUESTED_MASTER_PORT="${MASTER_PORT:-}"
MASTER_PORT_SOURCE="default_from_slurm_job_id"
if [ -n "$REQUESTED_MASTER_PORT" ]; then
  MASTER_PORT="$REQUESTED_MASTER_PORT"
  MASTER_PORT_SOURCE="environment"
else
  MASTER_PORT="$DEFAULT_MASTER_PORT"
fi
RUN_TAG="${RUN_TAG:-p2qr_short_smoke_exec_${SLURM_JOB_ID:-nojob}_$(date '+%Y%m%d_%H%M%S')}"

case "$RUN_TAG" in
  *..*|*/*|*\\*|*[[:space:]]*|*[\;\&\|\>\<\`\"\'$]*)
    echo "[p2qr-short-smoke-exec] $(date -Is) ERROR RUN_TAG must be path-safe; got $RUN_TAG" >&2
    exit 2
    ;;
esac

LOG_DIR="$REPO/logs"
RUN_ROOT="$LOG_DIR/$RUN_TAG"
RUN_LOG="$RUN_ROOT/${RUN_TAG}.log"
TRAIN_STDOUT="$RUN_ROOT/${RUN_TAG}.train.stdout.log"
SHA_MANIFEST="$RUN_ROOT/active_sha256_manifest.txt"
RESOLVED_CONFIG_DUMP="$RUN_ROOT/resolved_config.py"
SUMMARY_JSON="$RUN_ROOT/p2qr_short_smoke_execution_summary.json"
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
  echo "[p2qr-short-smoke-exec] $(date -Is) $*"
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
    "stage": "P2QR_SHORT_SMOKE_EXECUTION_CANDIDATE",
    "route": "CTF-BDI/PC-OT-MRAS",
    "host": socket.gethostname(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "run_root": run_root,
    "config": config,
    "work_dir": work_dir,
    "claim_boundaries": {
        "tools_train_only": True,
        "max_epochs": 1,
        "max_train_iters": 2,
        "checkpoint_disabled": True,
        "tools_test_direct_entrypoint": False,
        "detector_map": False,
        "train_validation_map": False,
        "raw_prediction_cache": False,
        "metric_claim_allowed": False,
        "runtime_or_flops_claim_allowed": False,
        "deploy_claim_allowed": False,
        "paper_claim_allowed": False,
    },
    "claim_flags": {
        "tools_train_run": decision == "P2QR_SHORT_SMOKE_EXECUTION_PASS_NO_EVAL_NO_CLAIMS",
        "tools_test_run": False,
        "detector_map_run": False,
        "metric_claim_allowed": False,
        "paper_claim_allowed": False,
        "runtime_or_flops_claim_allowed": False,
        "deploy_claim_allowed": False,
    },
}
Path(summary_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  log "summary_json=$SUMMARY_JSON"
}

on_exit() {
  local status="$?"
  if [ "$status" -ne 0 ] && [ ! -f "$SUMMARY_JSON" ]; then
    write_summary "FAIL" "P2QR_SHORT_SMOKE_EXECUTION_FAILED" "launcher exited before successful completion"
  fi
}
trap on_exit EXIT

case "$PRECHECK_ONLY" in
  0|1) ;;
  *) fail "PRECHECK_ONLY must be exactly 0 or 1; got $PRECHECK_ONLY" ;;
esac
case "$P2QR_SHORT_SMOKE_ONLY" in
  1) ;;
  *) fail "P2QR_SHORT_SMOKE_ONLY must remain 1" ;;
esac
case "$ALLOW_P2QR_TOOLS_TEST" in
  0) ;;
  *) fail "tools/test.py is not approved; ALLOW_P2QR_TOOLS_TEST must remain 0" ;;
esac
case "$ALLOW_P2QR_DETECTOR_MAP" in
  0) ;;
  *) fail "detector mAP is not approved; ALLOW_P2QR_DETECTOR_MAP must remain 0" ;;
esac
case "$ALLOW_P2QR_LONG_TRAINING" in
  0) ;;
  *) fail "long training is not approved; ALLOW_P2QR_LONG_TRAINING must remain 0" ;;
esac
case "$ALLOW_P2QR_CHECKPOINT_WRITE" in
  0) ;;
  *) fail "checkpoint writing is disabled for this bounded short smoke" ;;
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
if [ "$PRECHECK_ONLY" = "1" ]; then
  case "$ALLOW_P2QR_SHORT_SMOKE" in
    0) ;;
    *) fail "PRECHECK_ONLY=1 must not set ALLOW_P2QR_SHORT_SMOKE=1" ;;
  esac
  case "$ALLOW_P2QR_SLURM" in
    0) ;;
    *) fail "PRECHECK_ONLY=1 must not set ALLOW_P2QR_SLURM=1" ;;
  esac
  case "$ALLOW_P2QR_GPU" in
    0) ;;
    *) fail "PRECHECK_ONLY=1 must not set ALLOW_P2QR_GPU=1" ;;
  esac
  if [ -n "${SLURM_JOB_ID:-}" ] && [ "${ALLOW_P2QR_PRECHECK_SLURM:-0}" != "1" ]; then
    fail "PRECHECK_ONLY=1 is local/static only unless ALLOW_P2QR_PRECHECK_SLURM=1"
  fi
else
  case "$ALLOW_P2QR_SHORT_SMOKE" in
    1) ;;
    *) fail "PRECHECK_ONLY=0 requires ALLOW_P2QR_SHORT_SMOKE=1 from the reviewed execution gate" ;;
  esac
  if [ -z "${SLURM_JOB_ID:-}" ]; then
    fail "PRECHECK_ONLY=0 requires a Slurm allocation"
  fi
  case "$ALLOW_P2QR_SLURM" in
    1) ;;
    *) fail "PRECHECK_ONLY=0 requires ALLOW_P2QR_SLURM=1 from the reviewed execution gate" ;;
  esac
  case "$ALLOW_P2QR_GPU" in
    1) ;;
    *) fail "PRECHECK_ONLY=0 requires ALLOW_P2QR_GPU=1 from the reviewed execution gate" ;;
  esac
  if [ -z "$P2QR_SHORT_SMOKE_GATE_JSON" ] || [ -z "$P2QR_SHORT_SMOKE_GATE_SHA256" ]; then
    fail "PRECHECK_ONLY=0 requires P2QR_SHORT_SMOKE_GATE_JSON and P2QR_SHORT_SMOKE_GATE_SHA256"
  fi
fi
if [ -n "${CHECKPOINT:-}" ] || [ -n "${CHECKPOINT_SHA256:-}" ] || [ -n "${LOAD_FROM:-}" ] || [ -n "${RESUME:-}" ]; then
  fail "checkpoint/load/resume shortcuts are forbidden for P2QR short smoke"
fi
if [ -n "${LOAD_FROM_RAW_PREDICTIONS:-}" ] || [ -n "${RAW_PREDICTION_CACHE:-}" ] || [ -n "${PREDICTION_CACHE:-}" ]; then
  fail "raw prediction/cache shortcuts are forbidden for P2QR short smoke"
fi
if [ -n "${CFG_OPTIONS:-}" ] || [ -n "${EXTRA_CFG_OPTIONS:-}" ]; then
  fail "arbitrary cfg-options are forbidden for P2QR short smoke"
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
log "precheck_only=$PRECHECK_ONLY allow_p2qr_short_smoke=$ALLOW_P2QR_SHORT_SMOKE allow_p2qr_slurm=$ALLOW_P2QR_SLURM allow_p2qr_gpu=$ALLOW_P2QR_GPU"
log "scope=P2QR_SHORT_SMOKE tools_train_max_2_iters=true direct_tools_test=false detector_map=false checkpoint_disabled=true no_claims=true"
log "python=$(python -c 'import sys; print(sys.executable)')"

test -f "$CONFIG" || fail "missing reviewed config: $CONFIG"

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

gate = cfg.p2_quality_rank_calibrator_v0_short_smoke_gate
require(gate.stage == "P2_NIIQ_QualityRank_Calibrator_v0_short_smoke_execution_candidate", gate)
require(gate.execution_candidate is True, gate)
require(gate.launch_scaffold_only is False, gate)
require(gate.smoke_only is True, gate)
require(gate.allow_detector_training is True, gate)
require(gate.launch_gate_passed is True, gate)
require(gate.allow_tools_train is True, gate)
require(gate.allow_tools_test is False, gate)
require(gate.allow_detector_map is False, gate)
require(gate.allow_train_validation_map is False, gate)
require(gate.allow_slurm is True, gate)
require(gate.allow_gpu is True, gate)
require(gate.allow_long_training is False, gate)
require(gate.allow_dataset_access is True, gate)
require(gate.allow_pretrained_initialization is True, gate)
require(gate.allow_checkpoint_access is False, gate)
require(gate.allow_checkpoint_write is False, gate)
require(gate.disable_checkpoint is True, gate)
require(gate.entrypoint_gate_context.required is True, gate)
require(tuple(gate.allowed_entrypoints) == ("tools/train.py",), gate)
require(int(gate.max_train_iters) == 2, gate)
require(int(cfg.workflow.max_train_iters) == 2, cfg.workflow)
require(int(cfg.workflow.end_epoch) == 1, cfg.workflow)
require(int(cfg.workflow.val_eval_interval) == -1, cfg.workflow)
require(int(cfg.workflow.val_loss_interval) == -1, cfg.workflow)
require(cfg.workflow.disable_checkpoint is True, cfg.workflow)
require(cfg.inference.load_from_raw_predictions is False, cfg.inference)
require(cfg.inference.save_raw_prediction is False, cfg.inference)
require(cfg.model.rpn_head.area_head.quality_calibration.enable is True, cfg.model.rpn_head.area_head)
try:
    assert_detector_training_allowed(cfg, entrypoint="tools/test.py")
except RuntimeError:
    pass
else:
    raise SystemExit("P2QR short-smoke execution config must reject direct tools/test.py")
print("p2qr_short_smoke_execution_config_preflight=PASS")
PY
RESOLVED_CONFIG_SHA256="$(sha256sum "$RESOLVED_CONFIG_DUMP" | awk '{print $1}')"
log "resolved_config_sha256=$RESOLVED_CONFIG_SHA256"
export P2QR_RESOLVED_CONFIG_SHA256="$RESOLVED_CONFIG_SHA256"

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
  scripts/run_ctf_bdi_pc_ot_mras_p2_quality_rank_calibrator_v0_short_smoke_execution_candidate_n16r4.sbatch \
  tools/train.py \
  tools/bata/validate_p2qr_short_smoke_execution_gate.py \
  opentad/cores/train_engine.py \
  opentad/utils/training_guard.py \
  opentad/models/detectors/actionformer.py \
  opentad/models/dense_heads/__init__.py \
  opentad/models/dense_heads/native_irregular_area_head_p2.py \
  opentad/models/necks/pc_ot_mras_detector_bridge.py \
  opentad/models/selectors/pc_ot_mras_reader.py \
  tests/test_pc_ot_mras_p2_quality_short_smoke_execution_candidate.py \
  > "$SHA_MANIFEST"
ACTIVE_MANIFEST_SHA256="$(sha256sum "$SHA_MANIFEST" | awk '{print $1}')"
log "active_sha256_manifest_sha256=$ACTIVE_MANIFEST_SHA256"
export P2QR_ACTIVE_MANIFEST_SHA256="$ACTIVE_MANIFEST_SHA256"

log "static py_compile"
python -m py_compile \
  "$CONFIG" \
  tools/train.py \
  tools/bata/validate_p2qr_short_smoke_execution_gate.py \
  opentad/cores/train_engine.py \
  opentad/utils/training_guard.py \
  opentad/models/detectors/actionformer.py \
  opentad/models/dense_heads/__init__.py \
  opentad/models/dense_heads/native_irregular_area_head_p2.py \
  opentad/models/necks/pc_ot_mras_detector_bridge.py \
  opentad/models/selectors/pc_ot_mras_reader.py \
  tests/test_pc_ot_mras_p2_quality_short_smoke_execution_candidate.py

log "focused local execution-candidate pytest"
python -m pytest \
  tests/test_pc_ot_mras_p2_quality_short_smoke_execution_candidate.py \
  tests/test_pc_ot_mras_p2_quality_launch_scaffold.py \
  tests/test_train_engine_max_train_iters.py \
  -q -rs

if [ "$PRECHECK_ONLY" = "1" ]; then
  write_summary "PASS" "P2QR_SHORT_SMOKE_EXECUTION_PRECHECK_PASS_NO_TRAIN" "PRECHECK_ONLY=1; exiting before tools/train.py and before data/pretrained access"
  log "PRECHECK_ONLY=1; exiting before any P2QR tools/train execution"
  exit 0
fi

test -f "$P2QR_SHORT_SMOKE_GATE_JSON" || fail "missing P2QR execution gate JSON: $P2QR_SHORT_SMOKE_GATE_JSON"
ACTUAL_GATE_SHA256="$(sha256sum "$P2QR_SHORT_SMOKE_GATE_JSON" | awk '{print $1}')"
if [ "$ACTUAL_GATE_SHA256" != "$P2QR_SHORT_SMOKE_GATE_SHA256" ]; then
  fail "P2QR execution gate sha256 mismatch: expected=$P2QR_SHORT_SMOKE_GATE_SHA256 actual=$ACTUAL_GATE_SHA256"
fi
python tools/bata/validate_p2qr_short_smoke_execution_gate.py \
  --gate-json "$P2QR_SHORT_SMOKE_GATE_JSON" \
  --gate-sha256 "$P2QR_SHORT_SMOKE_GATE_SHA256" \
  --active-manifest-sha256 "$ACTIVE_MANIFEST_SHA256" \
  --resolved-config-sha256 "$RESOLVED_CONFIG_SHA256"

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
    raise SystemExit("P2QR short smoke requires at least one visible CUDA device")
PY
nvidia-smi || fail "nvidia-smi failed"

if command -v ss >/dev/null 2>&1; then
  if ss -ltn | awk '{print $4}' | grep -Eq "[:.]${MASTER_PORT}$"; then
    fail "MASTER_PORT=$MASTER_PORT is already in use"
  fi
else
  log "WARN ss not available; skipping master port preflight"
fi

mkdir -p "$WORK_DIR"
export OPENTAD_PCOTMRAS_ENTRYPOINT_GATE_JSON="$P2QR_SHORT_SMOKE_GATE_JSON"
export OPENTAD_PCOTMRAS_ENTRYPOINT_GATE_SHA256="$P2QR_SHORT_SMOKE_GATE_SHA256"
export OPENTAD_PCOTMRAS_ACTIVE_MANIFEST_SHA256="$ACTIVE_MANIFEST_SHA256"
export OPENTAD_PCOTMRAS_RESOLVED_CONFIG_SHA256="$RESOLVED_CONFIG_SHA256"
log "launching bounded P2QR tools/train.py short smoke"
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
grep -q "max_train_iters=2 reached" "$TRAIN_STDOUT" || fail "train stdout does not show max_train_iters=2 early stop"
if grep -Eq "load_from_raw_predictions|RAW_PREDICTION_CACHE|PREDICTION_CACHE" "$TRAIN_STDOUT"; then
  fail "train stdout contains raw prediction/cache marker"
fi
if find "$WORK_DIR" "$RUN_ROOT" -maxdepth 4 -type f -name 'result_detection.json' -print -quit | grep -q .; then
  fail "unexpected result_detection.json found; tools/test.py or detector mAP artifact is outside this smoke"
fi
if find "$WORK_DIR" -maxdepth 4 -type f \( -name '*.pth' -o -name '*.pt' \) -print -quit | grep -q .; then
  fail "unexpected checkpoint artifact found despite workflow.disable_checkpoint=True"
fi

write_summary "PASS" "P2QR_SHORT_SMOKE_EXECUTION_PASS_NO_EVAL_NO_CLAIMS" "tools/train.py completed a two-iteration smoke; no tools/test.py, mAP, raw prediction cache, or checkpoint artifact"
log "P2QR_SHORT_SMOKE_EXECUTION_PASS_NO_EVAL_NO_CLAIMS"
