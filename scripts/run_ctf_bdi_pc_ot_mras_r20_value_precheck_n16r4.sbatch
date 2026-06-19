#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH -J pcot_r20chk
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
    echo "[pc-ot-mras-r20-value-precheck] $(date -Is) ERROR refusing dirty historical OpenTAD_BATA_Clean path: $REPO" >&2
    exit 2
    ;;
esac
if [ ! -d "$REPO" ]; then
  echo "[pc-ot-mras-r20-value-precheck] $(date -Is) ERROR missing clean PC-OT-MRAS repo: $REPO" >&2
  exit 2
fi
cd "$REPO"

CONFIG_MAIN="configs/adatad/thumos/ctf_bdi_pc_ot_mras_r20_value_distill_candidate.py"
CONFIG_VALUE_ONLY="configs/adatad/thumos/ctf_bdi_pc_ot_mras_r20_value_only_control.py"
CONFIG="${CONFIG:-$CONFIG_MAIN}"
PRECHECK_ONLY="${PRECHECK_ONLY:-1}"
RUN_TAG="${RUN_TAG:-ctf_bdi_pc_ot_mras_r20_value_precheck_${SLURM_JOB_ID:-nojob}_$(date '+%Y%m%d_%H%M%S')}"

case "$RUN_TAG" in
  *..*|*/*|*\\*|*[[:space:]]*|*[\;\&\|\>\<\`\"\'$]*)
    echo "[pc-ot-mras-r20-value-precheck] $(date -Is) ERROR RUN_TAG must be a simple path-safe name; got $RUN_TAG" >&2
    exit 2
    ;;
esac

RUN_ROOT="$REPO/logs/$RUN_TAG"
RUN_LOG="$RUN_ROOT/${RUN_TAG}.log"
SUMMARY_JSON="$RUN_ROOT/r20_value_precheck_summary.json"
RESOLVED_CONFIG_DUMP="$RUN_ROOT/resolved_config.py"
SHA_MANIFEST="$RUN_ROOT/active_sha256_manifest.txt"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export YUZIBO_ROOT
export OPENTAD_PCOTMRAS_CLEAN_ROOT="$REPO"

mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_LOG") 2>&1

log() {
  echo "[pc-ot-mras-r20-value-precheck] $(date -Is) $*"
}

fail() {
  log "ERROR $*"
  exit 2
}

write_summary() {
  local status="$1"
  local decision="$2"
  local reason="$3"
  python - "$SUMMARY_JSON" "$status" "$decision" "$reason" "$RUN_ROOT" "$CONFIG" <<'PY'
import json
import os
import socket
import sys
from pathlib import Path

summary_path, status, decision, reason, run_root, config = sys.argv[1:7]
payload = {
    "status": status,
    "decision": decision,
    "reason": reason,
    "stage": "R20_VALUE_PRECHECK_ONLY",
    "route": "CTF-BDI/PC-OT-MRAS",
    "host": socket.gethostname(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "run_root": run_root,
    "config": config,
    "claim_boundaries": {
        "precheck_only": True,
        "tools_train": False,
        "tools_test": False,
        "detector_map": False,
        "real_dataset": False,
        "checkpoint_access": False,
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
    write_summary "FAIL" "R20_VALUE_PRECHECK_FAILED" "launcher exited before successful completion"
  fi
}
trap on_exit EXIT

case "$PRECHECK_ONLY" in
  1) ;;
  *) fail "R20 value launcher is precheck-only; PRECHECK_ONLY must remain 1" ;;
esac
case "$CONFIG" in
  "$CONFIG_MAIN"|"$CONFIG_VALUE_ONLY") ;;
  *) fail "reviewed only for $CONFIG_MAIN or $CONFIG_VALUE_ONLY; got $CONFIG" ;;
esac
if [ -z "${SLURM_JOB_ID:-}" ] && [ "${ALLOW_LOGIN_NODE_DEBUG:-0}" != "1" ]; then
  fail "refusing direct non-Slurm execution unless ALLOW_LOGIN_NODE_DEBUG=1 for lightweight PRECHECK_ONLY"
fi
if [ -n "${CHECKPOINT:-}" ] || [ -n "${CHECKPOINT_SHA256:-}" ] || [ -n "${LOAD_FROM:-}" ] || [ -n "${RESUME:-}" ]; then
  fail "checkpoint/load/resume shortcuts are forbidden for R20 precheck"
fi
if [ -n "${CFG_OPTIONS:-}" ] || [ -n "${EXTRA_CFG_OPTIONS:-}" ]; then
  fail "arbitrary cfg-options are forbidden for R20 precheck"
fi
if [ -n "${LOAD_FROM_RAW_PREDICTIONS:-}" ] || [ -n "${RAW_PREDICTION_CACHE:-}" ] || [ -n "${PREDICTION_CACHE:-}" ]; then
  fail "raw prediction/cache environment shortcuts are forbidden for R20 precheck"
fi
EXPECTED_GIT_BRANCH="${OPENTAD_PCOTMRAS_EXPECTED_BRANCH:-codex/pcotmras-r16-r18-r20-clean}"
GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)" || fail "clean repo must be a git checkout"
GIT_HEAD="$(git rev-parse HEAD 2>/dev/null)" || fail "clean repo HEAD is unavailable"
if [ "$GIT_BRANCH" != "$EXPECTED_GIT_BRANCH" ]; then
  fail "expected branch $EXPECTED_GIT_BRANCH; got $GIT_BRANCH"
fi
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  fail "tracked clean repo files are modified; refusing precheck"
fi

log "host=$(hostname) job=${SLURM_JOB_ID:-none}"
log "repo=$REPO"
log "git_branch=$GIT_BRANCH git_head=$GIT_HEAD"
log "config=$CONFIG"
log "run_root=$RUN_ROOT"
log "scope=R20_VALUE_PRECHECK_ONLY no_train=true no_test=true no_map=true no_data=true no_claims=true"
log "python=$(python -c 'import sys; print(sys.executable)')"

test -f "$CONFIG" || fail "missing reviewed config: $CONFIG"

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
PY
nvidia-smi || fail "nvidia-smi failed"

log "dumping and auditing resolved R20 config=$RESOLVED_CONFIG_DUMP"
python - "$CONFIG" "$RESOLVED_CONFIG_DUMP" <<'PY'
import sys
from pathlib import Path

from mmengine.config import Config
from opentad.utils.training_guard import assert_detector_training_allowed

cfg = Config.fromfile(sys.argv[1])
Path(sys.argv[2]).write_text(cfg.pretty_text + "\n", encoding="utf-8")

if "r20_pc_ot_mras_value_distill_gate" in cfg:
    gate = cfg.r20_pc_ot_mras_value_distill_gate
    expected_stage = "R20A_train_only_value_of_information_distillation_candidate"
else:
    gate = cfg.r20_pc_ot_mras_value_only_control_gate
    expected_stage = "R20_value_only_attribution_control"

assert gate.stage == expected_stage, gate
assert gate.launch_gate_passed is False, gate
assert gate.allow_tools_train is False, gate
assert gate.allow_tools_test is False, gate
assert gate.allow_slurm is False, gate
assert gate.allow_gpu is False, gate
assert gate.allow_real_dataset is False, gate
assert gate.allow_checkpoint is False, gate
assert gate.metric_claim_allowed is False, gate
assert gate.paper_claim_allowed is False, gate
assert cfg.model.pc_ot_mras_reader.enable_value_heads is True
assert cfg.model.pc_ot_mras_reader_value_loss.enabled is True
assert cfg.model.pc_ot_mras_reader_value_loss.require_targets is True
assert cfg.model.pc_ot_mras_reader_value_loss.allow_teacher_targets is False
assert cfg.inference.load_from_raw_predictions is False
assert cfg.inference.save_raw_prediction is False

try:
    assert_detector_training_allowed(cfg, entrypoint="tools/train.py")
except RuntimeError:
    pass
else:
    raise SystemExit("R20 value config must remain training-blocked")

print("ctf_bdi_pc_ot_mras_r20_value_precheck_config=PASS")
PY
RESOLVED_CONFIG_SHA256="$(sha256sum "$RESOLVED_CONFIG_DUMP" | awk '{print $1}')"
log "resolved_config_sha256=$RESOLVED_CONFIG_SHA256"

mapfile -t CONFIG_DEP_FILES < <(python - "$CONFIG_MAIN" "$CONFIG_VALUE_ONLY" <<'PY'
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

for arg in sys.argv[1:]:
    visit(Path(arg))
for path in seen:
    print(path.relative_to(Path.cwd()).as_posix())
PY
)
log "resolved_config_dependency_count=${#CONFIG_DEP_FILES[@]}"
log "writing active SHA256 manifest=$SHA_MANIFEST"
sha256sum \
  "${CONFIG_DEP_FILES[@]}" \
  "$RESOLVED_CONFIG_DUMP" \
  scripts/run_ctf_bdi_pc_ot_mras_r20_value_precheck_n16r4.sbatch \
  tools/train.py \
  tools/test.py \
  opentad/utils/training_guard.py \
  opentad/models/detectors/actionformer.py \
  opentad/models/losses/pc_ot_mras_value_distillation_losses.py \
  opentad/models/selectors/pc_ot_mras_reader.py \
  opentad/models/utils/pc_ot_mras_raw_prediction_guard.py \
  tests/test_pc_ot_mras_r20_value_only_control_config.py \
  tests/test_pc_ot_mras_value_distillation_losses.py \
  > "$SHA_MANIFEST"
ACTIVE_MANIFEST_SHA256="$(sha256sum "$SHA_MANIFEST" | awk '{print $1}')"
log "active_sha256_manifest_sha256=$ACTIVE_MANIFEST_SHA256"

log "static py_compile"
python -m py_compile \
  opentad/models/detectors/actionformer.py \
  opentad/models/losses/pc_ot_mras_value_distillation_losses.py \
  opentad/models/selectors/pc_ot_mras_reader.py \
  tests/test_pc_ot_mras_r20_value_only_control_config.py \
  tests/test_pc_ot_mras_value_distillation_losses.py

log "focused R20 value tests"
python -m pytest \
  tests/test_pc_ot_mras_r20_value_only_control_config.py \
  tests/test_pc_ot_mras_value_distillation_losses.py \
  tests/test_pc_ot_mras_actionformer_forward_selector.py \
  -q

write_summary "PASS" "R20_VALUE_PRECHECK_ONLY_PASS_NO_TRAIN_NO_DATA_NO_MAP" "R20 config/value-loss precheck completed without training or data access"
log "R20_VALUE_PRECHECK_ONLY_PASS_NO_TRAIN_NO_DATA_NO_MAP"
