#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -J ft_hybrid_pre
#SBATCH -o logs/%x-%j.out

set -euo pipefail

source /etc/profile >/dev/null 2>&1 || true
if command -v module >/dev/null 2>&1; then
  module load cuda/11.8
  module load miniforge3/24.11
fi

YUZIBO_ROOT="${YUZIBO_ROOT:-$HOME/run/yuzibo}"
REPO="${FRAME_TOKEN_HYBRID_REPO:-$YUZIBO_ROOT/OpenTAD_FrameToken_PrecheckDeploy_20260624_17d958e}"
PYTHON="${PYTHON:-$YUZIBO_ROOT/conda_envs/opentad/bin/python}"

if [ ! -d "$REPO" ]; then
  echo "[frame-token-hybrid-precheck] ERROR missing repo: $REPO" >&2
  exit 2
fi
cd "$REPO"

if [ ! -x "$PYTHON" ]; then
  if command -v conda >/dev/null 2>&1; then
    source activate "$YUZIBO_ROOT/conda_envs/opentad"
    PYTHON="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON="python"
  else
    echo "[frame-token-hybrid-precheck] ERROR no Python interpreter found" >&2
    exit 2
  fi
fi

PRECHECK_ONLY="${PRECHECK_ONLY:-1}"
ALLOW_FULL_TRAIN="${ALLOW_FRAME_TOKEN_HYBRID_FULL_TRAIN:-0}"
ALLOW_TOOLS_TRAIN="${ALLOW_FRAME_TOKEN_HYBRID_TOOLS_TRAIN:-0}"
ALLOW_TOOLS_TEST="${ALLOW_FRAME_TOKEN_HYBRID_TOOLS_TEST:-0}"
ALLOW_RAW_PREDICTION_CACHE="${ALLOW_FRAME_TOKEN_HYBRID_RAW_PREDICTION_CACHE:-0}"

CONFIG="${FRAME_TOKEN_HYBRID_CONFIG:-configs/adatad/thumos/frame_token_hybrid_acquisition_full_train_candidate_n16r4.py}"
LOCAL_CONFIG="configs/adatad/thumos/frame_token_hybrid_acquisition_local_precheck.py"
VALIDATOR="tools/bata/validate_frame_token_hybrid_gate.py"
LAUNCHER="scripts/run_frame_token_hybrid_acquisition_precheck_n16r4.sbatch"
SELECTOR="opentad/models/selectors/frame_token_hybrid_acquisition_route.py"
CONFIG_TEST="tests/test_frame_token_hybrid_config_gate.py"
ROUTE_LABEL="DIVERGENT_INNOVATION_FRAME_TOKEN_HYBRID_DO_NOT_MERGE_WITH_C3"

RUN_TAG="${RUN_TAG:-frame_token_hybrid_precheck_$(git rev-parse --short HEAD 2>/dev/null || echo nogit)_$(date '+%Y%m%d_%H%M%S')}"
case "$RUN_TAG" in
  *..*|*/*|*\\*|*[[:space:]]*|*[\;\&\|\>\<\`\"\'$]*)
    echo "[frame-token-hybrid-precheck] ERROR RUN_TAG must be path-safe; got $RUN_TAG" >&2
    exit 2
    ;;
esac

RUN_ROOT="${RUN_ROOT:-logs/$RUN_TAG}"
RUN_LOG="$RUN_ROOT/${RUN_TAG}.log"
RESOLVED_CONFIG_DUMP="$RUN_ROOT/resolved_config.py"
CONFIG_GATE_SUMMARY="$RUN_ROOT/config_gate_summary.json"
GATE_JSON="$RUN_ROOT/frame_token_hybrid_gate.json"
SHA_MANIFEST="$RUN_ROOT/active_sha256_manifest.txt"
SUMMARY_JSON="$RUN_ROOT/frame_token_hybrid_precheck_summary.json"

mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_LOG") 2>&1

log() {
  echo "[frame-token-hybrid-precheck] $(date -Is) $*"
}

fail() {
  log "ERROR $*"
  write_summary "FAIL" "FRAME_TOKEN_HYBRID_PRECHECK_FAILED" "$*"
  exit 2
}

write_summary() {
  local status="$1"
  local decision="$2"
  local reason="$3"
  "$PYTHON" - "$SUMMARY_JSON" "$status" "$decision" "$reason" "$RUN_ROOT" "$CONFIG" "$ROUTE_LABEL" <<'PY'
import json
import os
import socket
import sys
from pathlib import Path

summary_path, status, decision, reason, run_root, config, route_label = sys.argv[1:8]
payload = {
    "status": status,
    "decision": decision,
    "reason": reason,
    "stage": "FRAME_TOKEN_HYBRID_DEPLOYMENT_PRECHECK_ONLY",
    "route": "frame_token_hybrid_acquisition",
    "route_label": route_label,
    "host": socket.gethostname(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "repo": os.getcwd(),
    "git_branch": os.popen("git rev-parse --abbrev-ref HEAD 2>/dev/null").read().strip(),
    "git_head": os.popen("git rev-parse HEAD 2>/dev/null").read().strip(),
    "run_root": run_root,
    "config": config,
    "precheck_only": os.environ.get("PRECHECK_ONLY", "1"),
    "full_train_locked": True,
    "tools_train": False,
    "tools_test": False,
    "metric_claim_allowed": False,
    "paper_claim_allowed": False,
}
Path(summary_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  log "summary_json=$SUMMARY_JSON"
}

case "$PRECHECK_ONLY" in
  0|1) ;;
  *) fail "PRECHECK_ONLY must be 0 or 1; got $PRECHECK_ONLY" ;;
esac
case "$ALLOW_TOOLS_TRAIN" in 0) ;; *) fail "training entrypoint is forbidden during deployment precheck" ;; esac
case "$ALLOW_TOOLS_TEST" in 0) ;; *) fail "test entrypoint is forbidden during deployment precheck" ;; esac
case "$ALLOW_RAW_PREDICTION_CACHE" in 0) ;; *) fail "raw prediction/cache is forbidden for this route precheck" ;; esac

GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)" || fail "repo must be a git checkout"
GIT_HEAD="$(git rev-parse HEAD 2>/dev/null)" || fail "repo HEAD unavailable"
EXPECTED_BRANCH="${FRAME_TOKEN_HYBRID_EXPECTED_BRANCH:-codex/frame-token-precheck-deploy-20260624}"
if [ "$GIT_BRANCH" != "$EXPECTED_BRANCH" ]; then
  fail "expected branch $EXPECTED_BRANCH; got $GIT_BRANCH"
fi
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  fail "tracked repo files are modified; refusing precheck"
fi

log "repo=$REPO"
log "git_branch=$GIT_BRANCH git_head=$GIT_HEAD"
log "config=$CONFIG"
log "precheck_only=$PRECHECK_ONLY"
log "run_root=$RUN_ROOT"

test -f "$CONFIG" || fail "missing config: $CONFIG"
test -f "$LOCAL_CONFIG" || fail "missing local config: $LOCAL_CONFIG"
test -f "$VALIDATOR" || fail "missing validator: $VALIDATOR"
test -f "$LAUNCHER" || fail "missing launcher: $LAUNCHER"
test -f "$SELECTOR" || fail "missing selector: $SELECTOR"
test -f "$CONFIG_TEST" || fail "missing config test: $CONFIG_TEST"

"$PYTHON" -m py_compile "$VALIDATOR" "$CONFIG" "$LOCAL_CONFIG" "$SELECTOR" "$CONFIG_TEST"

"$PYTHON" "$VALIDATOR" "$CONFIG" --json > "$CONFIG_GATE_SUMMARY"
"$PYTHON" - "$CONFIG_GATE_SUMMARY" "$ROUTE_LABEL" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
route_label = sys.argv[2]
if payload.get("route_label") != route_label:
    raise SystemExit("route label mismatch")
for key in (
    "allow_tools_train",
    "allow_tools_test",
    "allow_remote_sync",
    "allow_slurm",
    "allow_gpu",
    "allow_full_train",
    "metric_claim_allowed",
    "paper_claim_allowed",
):
    if payload.get(key) is not False:
        raise SystemExit(f"{key} must remain false")
if payload.get("allow_precheck_only") is not True:
    raise SystemExit("allow_precheck_only must be true")
if payload.get("allowed_entrypoints") != []:
    raise SystemExit("allowed_entrypoints must remain empty")
PY

"$PYTHON" - "$CONFIG" "$RESOLVED_CONFIG_DUMP" "$ROUTE_LABEL" <<'PY'
import sys
from pathlib import Path

from mmengine.config import Config

config_path, dump_path, route_label = sys.argv[1:4]
cfg = Config.fromfile(config_path)
if cfg.route_label != route_label:
    raise SystemExit("route_label mismatch")
gate = cfg.frame_token_hybrid_gate
if gate.allowed_decision != "ALLOW_FRAME_TOKEN_HYBRID_PRECHECK_ONLY":
    raise SystemExit("unexpected gate decision")
if gate.allow_precheck_only is not True:
    raise SystemExit("allow_precheck_only must be true")
for key in (
    "allow_tools_train",
    "allow_tools_test",
    "allow_remote_sync",
    "allow_slurm",
    "allow_gpu",
    "allow_full_train",
    "metric_claim_allowed",
    "paper_claim_allowed",
):
    if getattr(gate, key) is not False:
        raise SystemExit(f"{key} must remain false")
selector = cfg.model.frame_selector
if selector.type != "FrameTokenHybridAcquisitionRoute":
    raise SystemExit("selector type mismatch")
if selector.require_preview_signal is not True:
    raise SystemExit("preview signal must be required")
if selector.target_len != 384 or selector.target_dense_len != 768:
    raise SystemExit("unexpected frame/token budget geometry")
Path(dump_path).write_text(cfg.pretty_text + "\n", encoding="utf-8")
print("FRAME_TOKEN_HYBRID_RESOLVED_CONFIG_PRECHECK_PASS")
PY

RESOLVED_CONFIG_SHA256="$(sha256sum "$RESOLVED_CONFIG_DUMP" | awk '{print $1}')"
{
  sha256sum "$CONFIG"
  sha256sum "$LOCAL_CONFIG"
  sha256sum "$VALIDATOR"
  sha256sum "$LAUNCHER"
  sha256sum "$SELECTOR"
  sha256sum "$CONFIG_TEST"
  printf '%s  resolved_config.py\n' "$RESOLVED_CONFIG_SHA256"
} > "$SHA_MANIFEST"
ACTIVE_MANIFEST_SHA256="$(sha256sum "$SHA_MANIFEST" | awk '{print $1}')"

"$PYTHON" - "$GATE_JSON" "$ACTIVE_MANIFEST_SHA256" "$RESOLVED_CONFIG_SHA256" <<'PY'
import json
import sys
from pathlib import Path

gate_path, manifest_sha, config_sha = sys.argv[1:4]
payload = {
    "decision": "ALLOW_FRAME_TOKEN_HYBRID_PRECHECK_ONLY",
    "route": "frame_token_hybrid_acquisition",
    "route_label": "DIVERGENT_INNOVATION_FRAME_TOKEN_HYBRID_DO_NOT_MERGE_WITH_C3",
    "active_sha256_manifest_sha256": manifest_sha,
    "resolved_config_sha256": config_sha,
    "budget": 384,
    "dense_window_size": 768,
    "target_dense_len": 768,
    "allow_precheck_only": True,
    "allow_tools_train": False,
    "allow_tools_test": False,
    "allow_remote_sync": False,
    "allow_slurm": False,
    "allow_gpu": False,
    "allow_full_train": False,
    "load_from_raw_predictions": False,
    "save_raw_prediction": False,
    "uses_gt_at_test": False,
    "uses_teacher": False,
    "uses_oracle": False,
    "uses_raw_prediction_cache": False,
    "metric_claim_allowed": False,
    "paper_claim_allowed": False,
}
Path(gate_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
GATE_SHA256="$(sha256sum "$GATE_JSON" | awk '{print $1}')"

"$PYTHON" "$VALIDATOR" \
  --gate-json "$GATE_JSON" \
  --gate-sha256 "$GATE_SHA256" \
  --active-manifest-sha256 "$ACTIVE_MANIFEST_SHA256" \
  --resolved-config-sha256 "$RESOLVED_CONFIG_SHA256" \
  --budget 384 \
  --dense-window-size 768 \
  --target-dense-len 768

"$PYTHON" -m pytest "$CONFIG_TEST" -q

if [ "$PRECHECK_ONLY" = "1" ]; then
  write_summary "PASS" "FRAME_TOKEN_HYBRID_PRECHECK_ONLY_PASS_NO_TRAIN" "PRECHECK_ONLY=1; gate/config/tests passed; no training command executed"
  log "FRAME_TOKEN_HYBRID_PRECHECK_ONLY_PASS_NO_TRAIN"
  exit 0
fi

if [ "$ALLOW_FULL_TRAIN" != "1" ]; then
  fail "PRECHECK_ONLY=0 requires ALLOW_FRAME_TOKEN_HYBRID_FULL_TRAIN=1 and a reviewed full-train gate"
fi

"$PYTHON" - "$CONFIG_GATE_SUMMARY" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if payload.get("allow_full_train") is not True or payload.get("allow_tools_train") is not True:
    raise SystemExit("full train gate is still locked")
PY

fail "this deployment-precheck launcher is not a full-training launcher"
