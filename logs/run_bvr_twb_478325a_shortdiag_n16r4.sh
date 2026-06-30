#!/usr/bin/env bash
set -euo pipefail

ROUTE_LABEL="DIVERGENT_INNOVATION_BVR_TWB_DO_NOT_MERGE_WITH_C3"
EXPECTED_BASE_COMMIT="478325af8da10646f747f955a54378d53fffd3ef"
CONFIG="configs/adatad/thumos/input_bvr_twb_dynamic_adapter_irregular_headv3_shortdiag.py"
LOG_DIR="logs/bvr_twb_478325a_shortdiag"
TRAIN_LOG="${LOG_DIR}/train_shortdiag_$(date +%Y%m%d_%H%M%S).log"
PRETRAIN="pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"

mkdir -p "${LOG_DIR}"

log_shortdiag() {
  echo "$*" | tee -a "${TRAIN_LOG}"
}

log_shortdiag "[shortdiag] route=${ROUTE_LABEL}"
log_shortdiag "[shortdiag] cwd=$(pwd)"
log_shortdiag "[shortdiag] config=${CONFIG}"
log_shortdiag "[shortdiag] diagnostic_only=true full_train_unlocked=false metric_claim=false sparse_compute_claim=false"

PACKAGE_HEAD="$(git rev-parse HEAD)"
if [[ "${BVR_TWB_SHORTDIAG_PURE_478325A_OVERLAY:-0}" == "1" ]]; then
  if [[ "${PACKAGE_HEAD}" != "${EXPECTED_BASE_COMMIT}" ]]; then
    echo "[shortdiag][fatal] pure overlay mode requires exact HEAD: package_head=${PACKAGE_HEAD}, expected_base_commit=${EXPECTED_BASE_COMMIT}" | tee -a "${TRAIN_LOG}" >&2
    exit 2
  fi
  EXECUTION_MODE="exact_base_overlay"
elif [[ "${PACKAGE_HEAD}" == "${EXPECTED_BASE_COMMIT}" ]]; then
  EXECUTION_MODE="exact_base_overlay"
elif git merge-base --is-ancestor "${EXPECTED_BASE_COMMIT}" "${PACKAGE_HEAD}"; then
  EXECUTION_MODE="descendant_shortdiag_package"
else
  echo "[shortdiag][fatal] package HEAD does not descend from expected base: package_head=${PACKAGE_HEAD}, expected_base_commit=${EXPECTED_BASE_COMMIT}" | tee -a "${TRAIN_LOG}" >&2
  exit 2
fi
log_shortdiag "[shortdiag] expected_base_commit=${EXPECTED_BASE_COMMIT}"
log_shortdiag "[shortdiag] package_head=${PACKAGE_HEAD}"
log_shortdiag "[shortdiag] execution_mode=${EXECUTION_MODE}"

if [[ ! -e data ]]; then
  echo "[shortdiag][fatal] data link/directory is missing" >&2
  exit 3
fi
if [[ ! -e pretrained ]]; then
  echo "[shortdiag][fatal] pretrained link/directory is missing" >&2
  exit 3
fi
if [[ ! -f "${PRETRAIN}" ]]; then
  echo "[shortdiag][fatal] required VideoMAE-S pretrain is missing: ${PRETRAIN}" >&2
  exit 3
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "[shortdiag][fatal] CUDA_VISIBLE_DEVICES is empty; run inside an already allocated child GPU context" >&2
  exit 4
fi

python - <<'PY'
from mmengine.config import Config

cfg = Config.fromfile("configs/adatad/thumos/input_bvr_twb_dynamic_adapter_irregular_headv3_shortdiag.py")
expected = "pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"
assert cfg.model.backbone.custom.pretrain == expected, cfg.model.backbone.custom.pretrain
assert cfg.route_label == "DIVERGENT_INNOVATION_BVR_TWB_DO_NOT_MERGE_WITH_C3"
assert cfg.workflow.end_epoch == 1
assert cfg.workflow.disable_checkpoint is True
assert cfg.workflow.val_loss_interval == -1
assert cfg.workflow.val_eval_interval == -1
assert cfg.solver.amp is False
assert cfg.solver.fp16_compress is False
assert cfg.full_train_unlocked is False
assert cfg.metric_claim is False
assert cfg.sparse_compute_claim is False
print("[shortdiag] resolved config pretrain/workflow assertions passed")
PY

python -m py_compile "${CONFIG}" tools/bvr_twb/validate_bvr_twb_shortdiag.py tools/bvr_twb/validate_bvr_twb_formal_readiness.py tests/test_bvr_twb_shortdiag.py
python -m pytest -q tests/test_bvr_twb_shortdiag.py
python tools/bvr_twb/validate_bvr_twb_geometry_contracts.py --require-torch
python tools/bvr_twb/validate_bvr_twb_launch_gate.py --config configs/adatad/thumos/input_bvr_twb_dynamic_adapter_irregular_headv3.py --audit-out-dir "${LOG_DIR}/launch_gate_precheck"
python tools/bvr_twb/validate_bvr_twb_shortdiag.py --config "${CONFIG}" --expected-base-commit "${EXPECTED_BASE_COMMIT}"

log_shortdiag "[shortdiag] starting one-epoch diagnostic-only torchrun"
set +e
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
  tools/train.py "${CONFIG}" --id 0 2>&1 | tee -a "${TRAIN_LOG}"
TRAIN_RC=${PIPESTATUS[0]}
set -e

if grep -Eiq 'no pretrain path is provided|(^|[^A-Za-z])NaN([^A-Za-z]|$)|cost[=: ]+nan|(^|[^A-Za-z])Inf(inity)?([^A-Za-z]|$)|non[- ]finite|Traceback \(most recent call last\)|RuntimeError|CUDA out of memory|CUDA OOM|\bKilled\b|No space left on device|no GPU|No CUDA GPUs|mAP|Avg[-_ ]?mAP|result_detection\.json|tools/test\.py|full_train_unlocked[=: ]+true|formal full train|paper claim|deploy claim|sparse[-_ ]?compute claim|FLOPs claim' "${TRAIN_LOG}"; then
  echo "[shortdiag][fatal] Pro stop condition marker found in ${TRAIN_LOG}" >&2
  exit 5
fi
if ! grep -Eq '\[Train\].*Loss=.*reg_loss=[+-]?[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?' "${TRAIN_LOG}"; then
  echo "[shortdiag][fatal] missing finite reg_loss evidence in ${TRAIN_LOG}" >&2
  exit 5
fi
if ! grep -Eq '\[Train\]\[RuntimeDebug\].*head_v3_regression_head_fp32_enabled=True.*head_v3_regression_loss_fp32_enabled=True.*head_v3_regression_samples_kept_after_filter=[1-9][0-9]*.*head_v2_reg_points_total=[1-9][0-9]*' "${TRAIN_LOG}"; then
  echo "[shortdiag][fatal] missing HeadV3 non-skipped regression runtime debug evidence in ${TRAIN_LOG}" >&2
  exit 5
fi
log_shortdiag "[bvr_twb_formal_precheck] finite_gradients=true no_skipped_optimizer_step=true no_skipped_reg_head=true linux_torch_precheck=true pretrain_loaded=true full_train_unlocked=false sparse_compute_claim=false"

python tools/bvr_twb/validate_bvr_twb_shortdiag.py --config "${CONFIG}" --train-log "${TRAIN_LOG}" --expected-base-commit "${EXPECTED_BASE_COMMIT}"

if [[ "${TRAIN_RC}" -ne 0 ]]; then
  echo "[shortdiag][fatal] torchrun exited with ${TRAIN_RC}" >&2
  exit "${TRAIN_RC}"
fi

echo "[shortdiag] completed diagnostic-only smoke. full_train_unlocked=false metric_claim=false sparse_compute_claim=false"
echo "[shortdiag] log=${TRAIN_LOG}"

# Example only, do not execute from this wrapper:
# srun --gres=gpu:1 bash logs/run_bvr_twb_478325a_shortdiag_n16r4.sh
