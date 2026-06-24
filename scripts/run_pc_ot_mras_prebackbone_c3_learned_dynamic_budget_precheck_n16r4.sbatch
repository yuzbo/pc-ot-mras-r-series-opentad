#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH -J c3dyn_precheck
#SBATCH -o logs/%x-%j.out

set -eo pipefail

source /etc/profile >/dev/null 2>&1 || true
set -u
module load cuda/11.8
module load miniforge3/24.11

YUZIBO_ROOT="${YUZIBO_ROOT:-$HOME/run/yuzibo}"
source activate "$YUZIBO_ROOT/conda_envs/opentad"

CLEAN_REPO_NAME="${OPENTAD_PCOTMRAS_CLEAN_REPO_NAME:-OpenTAD_C3DynamicBudget_Worktree_20260624}"
REPO="${OPENTAD_PCOTMRAS_CLEAN_ROOT:-$YUZIBO_ROOT/$CLEAN_REPO_NAME}"
CONFIG="configs/adatad/thumos/pc_ot_mras_prebackbone_c3_learned_dynamic_budget_candidate_n16r4.py"
EXPECTED_GIT_BRANCH="${OPENTAD_PCOTMRAS_EXPECTED_BRANCH:-codex/c3-dynamic-budget-20260624}"
PRECHECK_ONLY="${PRECHECK_ONLY:-1}"
RUN_TAG="${RUN_TAG:-c3_learned_dynamic_budget_precheck_${SLURM_JOB_ID:-nojob}_$(date '+%Y%m%d_%H%M%S')}"

if [ "$PRECHECK_ONLY" != "1" ]; then
  echo "[c3-dynamic-budget-precheck] $(date -Is) ERROR PRECHECK_ONLY must stay 1" >&2
  exit 2
fi
if [ ! -d "$REPO" ]; then
  echo "[c3-dynamic-budget-precheck] $(date -Is) ERROR missing repo: $REPO" >&2
  exit 2
fi

cd "$REPO"
LOG_DIR="$REPO/logs"
RUN_ROOT="$LOG_DIR/$RUN_TAG"
SUMMARY_JSON="$RUN_ROOT/c3_learned_dynamic_budget_precheck_summary.json"
mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_ROOT/precheck.log") 2>&1

log() {
  echo "[c3-dynamic-budget-precheck] $(date -Is) $*"
}

fail() {
  log "ERROR $*"
  exit 2
}

GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)" || fail "repo must be a git checkout"
GIT_HEAD="$(git rev-parse HEAD 2>/dev/null)" || fail "repo HEAD unavailable"
if [ "$GIT_BRANCH" != "$EXPECTED_GIT_BRANCH" ]; then
  fail "expected branch $EXPECTED_GIT_BRANCH; got $GIT_BRANCH"
fi
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  fail "tracked repo files are modified; refusing precheck"
fi
test -f "$CONFIG" || fail "missing config: $CONFIG"

log "repo=$REPO"
log "git_branch=$GIT_BRANCH git_head=$GIT_HEAD"
log "config=$CONFIG"
log "run_root=$RUN_ROOT"

python - "$CONFIG" <<'PY'
import sys
from mmengine.config import Config

cfg = Config.fromfile(sys.argv[1])
gate = cfg.pc_ot_mras_prebackbone_e2e_acquisition_gate
selector = cfg.model.frame_selector
if cfg.experiment_scope.route != "pc_ot_mras_prebackbone_c3_learned_dynamic_budget":
    raise SystemExit("wrong route")
if cfg.experiment_scope.backend != "OriginalAdaTAD":
    raise SystemExit("backend must stay OriginalAdaTAD")
if gate.allow_precheck_only is not True:
    raise SystemExit("precheck must be allowed")
for key in ("allow_tools_train", "allow_tools_test", "allow_detector_map"):
    if getattr(gate, key):
        raise SystemExit(f"{key} must be false")
if gate.allow_slurm_precheck is not True or gate.allow_gpu_precheck is not True:
    raise SystemExit("Slurm/GPU are allowed only for this bounded precheck environment")
if gate.allow_slurm_long_training is not False or gate.allow_gpu_long_training is not False:
    raise SystemExit("Slurm/GPU long training must stay forbidden")
if selector.dynamic_budget.protocol != "learned_dynamic_budget":
    raise SystemExit("dynamic budget protocol mismatch")
if tuple(selector.dynamic_budget.budget_bins) != (256, 320, 384):
    raise SystemExit("budget bins mismatch")
if selector.target_len != 384 or selector.reader.num_slots != 384:
    raise SystemExit("max output capacity must be 384")
PY

python -m py_compile \
  opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py \
  tests/test_pc_ot_mras_prebackbone_c3_pro_dynamic_budget.py

python -m pytest -q \
  tests/test_pc_ot_mras_prebackbone_c3_pro_dynamic_budget.py \
  tests/test_pc_ot_mras_prebackbone_c3_reader_variants.py

python - "$SUMMARY_JSON" "$CONFIG" "$GIT_HEAD" <<'PY'
import json
import sys
from pathlib import Path

summary_path, config, git_head = sys.argv[1:4]
payload = {
    "status": "PASS",
    "decision": "PREBACKBONE_C3_LEARNED_DYNAMIC_BUDGET_PRECHECK_PASS",
    "config": config,
    "git_head": git_head,
    "claim_boundaries": {
        "selection_surface": "pre_backbone_raw_frame",
        "budget_policy": "learned_dynamic_budget",
        "p2": False,
        "offline_ledger": False,
        "teacher": False,
        "test_gt": False,
        "raw_prediction_cache": False,
        "metric_claim_allowed": False,
        "runtime_or_flops_claim_allowed": False,
        "deploy_claim_allowed": False,
        "paper_claim_allowed": False,
    },
}
Path(summary_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

log "PREBACKBONE_C3_LEARNED_DYNAMIC_BUDGET_PRECHECK_PASS summary=$SUMMARY_JSON"
