#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -J ft_hybrid_full
#SBATCH -o logs/%x-%j.out

set -eo pipefail

source /etc/profile >/dev/null 2>&1 || true
set -u
if command -v module >/dev/null 2>&1; then
  module load cuda/11.8
  module load miniforge3/24.11
fi

YUZIBO_ROOT="${YUZIBO_ROOT:-$HOME/run/yuzibo}"
REPO="${FRAME_TOKEN_HYBRID_REPO:-$YUZIBO_ROOT/OpenTAD_FrameToken_PrecheckDeploy_20260624_ecb7c27}"
PYTHON="${PYTHON:-$YUZIBO_ROOT/conda_envs/opentad/bin/python}"

if [ ! -d "$REPO" ]; then
  echo "[frame-token-hybrid-full] ERROR missing repo: $REPO" >&2
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
    echo "[frame-token-hybrid-full] ERROR no Python interpreter found" >&2
    exit 2
  fi
fi

PRECHECK_ONLY="${PRECHECK_ONLY:-1}"
ALLOW_FULL_TRAIN="${ALLOW_FRAME_TOKEN_HYBRID_FULL_TRAIN:-0}"
ALLOW_TOOLS_TEST="${ALLOW_FRAME_TOKEN_HYBRID_TOOLS_TEST:-0}"
ALLOW_RAW_PREDICTION_CACHE="${ALLOW_FRAME_TOKEN_HYBRID_RAW_PREDICTION_CACHE:-0}"

CONFIG="${FRAME_TOKEN_HYBRID_CONFIG:-configs/adatad/thumos/frame_token_hybrid_acquisition_full_train_candidate_n16r4.py}"
LOCAL_CONFIG="configs/adatad/thumos/frame_token_hybrid_acquisition_local_precheck.py"
VALIDATOR="tools/bata/validate_frame_token_hybrid_gate.py"
FULL_LAUNCHER="scripts/run_frame_token_hybrid_acquisition_full_train_n16r4.sbatch"
SELECTOR="opentad/models/selectors/frame_token_hybrid_acquisition_route.py"
CONFIG_TEST="tests/test_frame_token_hybrid_config_gate.py"
ROUTE_LABEL="DIVERGENT_INNOVATION_FRAME_TOKEN_HYBRID_DO_NOT_MERGE_WITH_C3"

RUN_TAG="${RUN_TAG:-frame_token_hybrid_full_train_$(git rev-parse --short HEAD 2>/dev/null || echo nogit)_$(date '+%Y%m%d_%H%M%S')}"
case "$RUN_TAG" in
  *..*|*/*|*\\*|*[[:space:]]*|*[\;\&\|\>\<\`\"\'$]*)
    echo "[frame-token-hybrid-full] ERROR RUN_TAG must be path-safe; got $RUN_TAG" >&2
    exit 2
    ;;
esac
export RUN_TAG

RUN_ROOT="${RUN_ROOT:-logs/$RUN_TAG}"
RUN_LOG="$RUN_ROOT/${RUN_TAG}.log"
RESOLVED_CONFIG_DUMP="$RUN_ROOT/resolved_config.py"
CONFIG_GATE_SUMMARY="$RUN_ROOT/config_gate_summary.json"
PRECHECK_GATE_JSON="$RUN_ROOT/frame_token_hybrid_precheck_gate.json"
SHA_MANIFEST="$RUN_ROOT/active_sha256_manifest.txt"
SUMMARY_JSON="$RUN_ROOT/frame_token_hybrid_full_train_summary.json"

mkdir -p "$RUN_ROOT"
exec > >(tee -a "$RUN_LOG") 2>&1

log() {
  echo "[frame-token-hybrid-full] $(date -Is) $*"
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
    "stage": "FRAME_TOKEN_HYBRID_FULL_TRAIN_CANDIDATE_N16R4",
    "route": "frame_token_hybrid_acquisition",
    "route_label": route_label,
    "host": socket.gethostname(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "repo": os.getcwd(),
    "git_branch": os.popen("git rev-parse --abbrev-ref HEAD 2>/dev/null").read().strip(),
    "git_head": os.popen("git rev-parse HEAD 2>/dev/null").read().strip(),
    "run_tag": os.environ.get("RUN_TAG"),
    "run_root": run_root,
    "config": config,
    "precheck_only": os.environ.get("PRECHECK_ONLY", "1"),
    "active_manifest_sha256": os.environ.get("FRAME_TOKEN_HYBRID_ACTIVE_MANIFEST_SHA256"),
    "resolved_config_sha256": os.environ.get("FRAME_TOKEN_HYBRID_RESOLVED_CONFIG_SHA256"),
    "full_train_gate_json": os.environ.get("FRAME_TOKEN_HYBRID_FULL_TRAIN_GATE_JSON"),
    "full_train_gate_sha256": os.environ.get("FRAME_TOKEN_HYBRID_FULL_TRAIN_GATE_SHA256"),
    "tools_test": False,
    "metric_claim_allowed": False,
    "paper_claim_allowed": False,
}
Path(summary_path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  log "summary_json=$SUMMARY_JSON"
}

fail() {
  log "ERROR $*"
  write_summary "FAIL" "FRAME_TOKEN_HYBRID_FULL_TRAIN_GATE_FAILED_CLOSED" "$*"
  exit 2
}

case "$PRECHECK_ONLY" in
  0|1) ;;
  *) fail "PRECHECK_ONLY must be 0 or 1; got $PRECHECK_ONLY" ;;
esac
case "$ALLOW_TOOLS_TEST" in 0) ;; *) fail "test entrypoint is forbidden for this route gate" ;; esac
case "$ALLOW_RAW_PREDICTION_CACHE" in 0) ;; *) fail "raw prediction/cache is forbidden for this route gate" ;; esac
if [ "$PRECHECK_ONLY" = "0" ] && [ -z "${SLURM_JOB_ID:-}" ]; then
  fail "PRECHECK_ONLY=0 full train must run inside Slurm"
fi

GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null)" || fail "repo must be a git checkout"
GIT_HEAD="$(git rev-parse HEAD 2>/dev/null)" || fail "repo HEAD unavailable"
EXPECTED_BRANCH="${FRAME_TOKEN_HYBRID_EXPECTED_BRANCH:-codex/frame-token-precheck-deploy-20260624}"
if [ "$GIT_BRANCH" != "$EXPECTED_BRANCH" ]; then
  fail "expected branch $EXPECTED_BRANCH; got $GIT_BRANCH"
fi
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  fail "tracked repo files are modified; refusing full-train gate"
fi

log "repo=$REPO"
log "git_branch=$GIT_BRANCH git_head=$GIT_HEAD"
log "config=$CONFIG"
log "precheck_only=$PRECHECK_ONLY"
log "run_tag=$RUN_TAG"
log "run_root=$RUN_ROOT"

test -f "$CONFIG" || fail "missing config: $CONFIG"
test -f "$LOCAL_CONFIG" || fail "missing local config: $LOCAL_CONFIG"
test -f "$VALIDATOR" || fail "missing validator: $VALIDATOR"
test -f "$FULL_LAUNCHER" || fail "missing full-train launcher: $FULL_LAUNCHER"
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
    "allow_tools_test",
    "allow_remote_sync",
    "metric_claim_allowed",
    "paper_claim_allowed",
):
    if payload.get(key) is not False:
        raise SystemExit(f"{key} must remain false in config default")
for key in ("allow_tools_train", "allow_slurm", "allow_gpu", "allow_full_train"):
    if payload.get(key) is not True:
        raise SystemExit(f"{key} must be true in gate-bound full-train config")
if payload.get("allow_precheck_only") is not True:
    raise SystemExit("allow_precheck_only must be true")
if payload.get("allowed_entrypoints") != ["tools/train.py"]:
    raise SystemExit("allowed_entrypoints must be tools/train.py only")
if payload.get("entrypoint_gate_required") is not True:
    raise SystemExit("entrypoint gate context must be required")
if payload.get("entrypoint_gate_allowed_decisions") != ["ALLOW_FRAME_TOKEN_HYBRID_FULL_TRAIN"]:
    raise SystemExit("entrypoint gate must bind only ALLOW_FRAME_TOKEN_HYBRID_FULL_TRAIN")
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
if gate.allowed_decision != "ALLOW_FRAME_TOKEN_HYBRID_FULL_TRAIN":
    raise SystemExit("config must be a full-train gate-bound candidate")
if gate.allow_precheck_only is not True:
    raise SystemExit("allow_precheck_only must be true")
for key in (
    "allow_tools_test",
    "allow_remote_sync",
    "metric_claim_allowed",
    "paper_claim_allowed",
):
    if getattr(gate, key) is not False:
        raise SystemExit(f"{key} must remain false in config default")
for key in ("allow_tools_train", "allow_slurm", "allow_gpu", "allow_full_train"):
    if getattr(gate, key) is not True:
        raise SystemExit(f"{key} must be true in gate-bound full-train config")
if tuple(gate.allowed_entrypoints) != ("tools/train.py",):
    raise SystemExit("allowed_entrypoints must be tools/train.py only")
context = gate.entrypoint_gate_context
if context.required is not True:
    raise SystemExit("entrypoint gate context must be required")
if tuple(context.allowed_decisions) != ("ALLOW_FRAME_TOKEN_HYBRID_FULL_TRAIN",):
    raise SystemExit("entrypoint gate must bind only ALLOW_FRAME_TOKEN_HYBRID_FULL_TRAIN")
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
  sha256sum "$FULL_LAUNCHER"
  sha256sum "$SELECTOR"
  sha256sum "$CONFIG_TEST"
  printf '%s  resolved_config.py\n' "$RESOLVED_CONFIG_SHA256"
} > "$SHA_MANIFEST"
ACTIVE_MANIFEST_SHA256="$(sha256sum "$SHA_MANIFEST" | awk '{print $1}')"
export FRAME_TOKEN_HYBRID_ACTIVE_MANIFEST_SHA256="$ACTIVE_MANIFEST_SHA256"
export FRAME_TOKEN_HYBRID_RESOLVED_CONFIG_SHA256="$RESOLVED_CONFIG_SHA256"

"$PYTHON" - "$PRECHECK_GATE_JSON" "$ACTIVE_MANIFEST_SHA256" "$RESOLVED_CONFIG_SHA256" "$RUN_TAG" <<'PY'
import json
import sys
from pathlib import Path

gate_path, manifest_sha, config_sha, run_tag = sys.argv[1:5]
payload = {
    "decision": "ALLOW_FRAME_TOKEN_HYBRID_PRECHECK_ONLY",
    "route": "frame_token_hybrid_acquisition",
    "route_label": "DIVERGENT_INNOVATION_FRAME_TOKEN_HYBRID_DO_NOT_MERGE_WITH_C3",
    "active_sha256_manifest_sha256": manifest_sha,
    "resolved_config_sha256": config_sha,
    "run_tag": run_tag,
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
PRECHECK_GATE_SHA256="$(sha256sum "$PRECHECK_GATE_JSON" | awk '{print $1}')"

"$PYTHON" "$VALIDATOR" \
  --action precheck \
  --gate-json "$PRECHECK_GATE_JSON" \
  --gate-sha256 "$PRECHECK_GATE_SHA256" \
  --active-manifest-sha256 "$ACTIVE_MANIFEST_SHA256" \
  --resolved-config-sha256 "$RESOLVED_CONFIG_SHA256" \
  --budget 384 \
  --dense-window-size 768 \
  --target-dense-len 768

log "running focused config-gate pytest with full-train gate env isolated"
env \
  -u ALLOW_FRAME_TOKEN_HYBRID_FULL_TRAIN \
  -u FRAME_TOKEN_HYBRID_FULL_TRAIN_GATE_JSON \
  -u FRAME_TOKEN_HYBRID_FULL_TRAIN_GATE_SHA256 \
  -u FRAME_TOKEN_HYBRID_ACTIVE_MANIFEST_SHA256 \
  -u FRAME_TOKEN_HYBRID_RESOLVED_CONFIG_SHA256 \
  -u RUN_TAG \
  "$PYTHON" -m pytest "$CONFIG_TEST" -q

if [ "$PRECHECK_ONLY" = "1" ]; then
  write_summary "PASS" "FRAME_TOKEN_HYBRID_PRECHECK_ONLY_PASS_NO_TRAIN" "PRECHECK_ONLY=1; manifest/config/gate/tests passed; no training command executed"
  log "FRAME_TOKEN_HYBRID_PRECHECK_ONLY_PASS_NO_TRAIN"
  exit 0
fi

if [ "$ALLOW_FULL_TRAIN" != "1" ]; then
  fail "PRECHECK_ONLY=0 requires ALLOW_FRAME_TOKEN_HYBRID_FULL_TRAIN=1"
fi
if [ -z "${FRAME_TOKEN_HYBRID_FULL_TRAIN_GATE_JSON:-}" ]; then
  fail "PRECHECK_ONLY=0 requires FRAME_TOKEN_HYBRID_FULL_TRAIN_GATE_JSON"
fi
if [ -z "${FRAME_TOKEN_HYBRID_FULL_TRAIN_GATE_SHA256:-}" ]; then
  fail "PRECHECK_ONLY=0 requires FRAME_TOKEN_HYBRID_FULL_TRAIN_GATE_SHA256"
fi

"$PYTHON" "$VALIDATOR" \
  --action full-train \
  --gate-json "$FRAME_TOKEN_HYBRID_FULL_TRAIN_GATE_JSON" \
  --gate-sha256 "$FRAME_TOKEN_HYBRID_FULL_TRAIN_GATE_SHA256" \
  --active-manifest-sha256 "$ACTIVE_MANIFEST_SHA256" \
  --resolved-config-sha256 "$RESOLVED_CONFIG_SHA256" \
  --run-tag "$RUN_TAG" \
  --budget 384 \
  --dense-window-size 768 \
  --target-dense-len 768

export FRAME_TOKEN_HYBRID_FULL_TRAIN_GATE_JSON
export FRAME_TOKEN_HYBRID_FULL_TRAIN_GATE_SHA256

TRAIN_ID="${FRAME_TOKEN_HYBRID_TRAIN_ID:-0}"
log "FRAME_TOKEN_HYBRID_FULL_TRAIN_GATE_VALIDATION_PASS"
log "launching_train=$PYTHON tools/train.py $CONFIG --id $TRAIN_ID"
write_summary "PASS" "FRAME_TOKEN_HYBRID_FULL_TRAIN_GATE_PASS_TRAIN_LAUNCHED" "full-train gate validated; launching tools/train.py"

set +e
"$PYTHON" tools/train.py "$CONFIG" --id "$TRAIN_ID"
TRAIN_RC="$?"
set -e
if [ "$TRAIN_RC" -ne 0 ]; then
  fail "tools/train.py exited with code $TRAIN_RC"
fi

write_summary "PASS" "FRAME_TOKEN_HYBRID_FULL_TRAIN_FINISHED" "tools/train.py exited 0"
log "FRAME_TOKEN_HYBRID_FULL_TRAIN_FINISHED"
