# DIVERGENT BVR-TWB 478325a Short Diagnostic Plan

Route label: `DIVERGENT_INNOVATION_BVR_TWB_DO_NOT_MERGE_WITH_C3`

Status: diagnostic-only setup implemented for the BVR-TWB route. This is not a long-train package, not a metric claim, not a sparse-compute/FLOPs claim, not a deploy claim, and not a paper claim. Formal full training remains locked.

## 2026-06-30 Execution Update

- Short diagnostic branch: `codex/divergent-bvr-twb-shortdiag-20260630`.
- Short diagnostic commit: `41c6a2703bce3e33a10a5141c0c6661cd6d2dcc7`.
- GitHub branch: `https://github.com/yuzbo/pc-ot-mras-r-series-opentad/tree/codex/divergent-bvr-twb-shortdiag-20260630`.
- Local verification completed:
  - `python -m py_compile configs/adatad/thumos/input_bvr_twb_dynamic_adapter_irregular_headv3_shortdiag.py tools/bvr_twb/validate_bvr_twb_shortdiag.py tests/test_bvr_twb_shortdiag.py`
  - `python -m pytest tests/test_bvr_twb_shortdiag.py -q` -> `5 passed`
  - `python tools/bvr_twb/validate_bvr_twb_shortdiag.py --config configs/adatad/thumos/input_bvr_twb_dynamic_adapter_irregular_headv3_shortdiag.py --allow-missing-pretrain` -> `gate_pass=true`, `full_train_unlocked=false`, `metric_claim=false`, `sparse_compute_claim=false`
  - pseudo good-log validator check -> `gate_pass=true`, `train_log_valid=true`, finite loss counted
  - `python -m pytest tests/test_bvr_twb_shortdiag.py tests/test_bvr_twb_validators.py tests/test_bvr_twb_opentad_pipeline.py tests/test_bvr_twb_geometry_contracts.py -q` -> `31 passed, 12 skipped`
  - `git diff --check` -> clean
  - `bash -n logs/run_bvr_twb_478325a_shortdiag_n16r4.sh` -> pass
- Contamination handling: one writable worker accidentally created the six short-diagnostic files in the shared root. The coordinator copied the exact files into this route-owned worktree, verified matching content, removed only those exact shared-root copies, and did not stage/commit/push from the shared root.
- Resource state: formal/diagnostic BVR launch remains blocked. Protected parent hold `1118197 pcot_dbg2g` must not be released. Current C3 child step `1118197.376 cadf_combo_g0` is still running with `AllocTRES=gres/gpu=2`, so GPU1 is not Slurm-safe for BVR even if it appears physically idle.

## Decision Context

- Valid Pro verdict: `REQUIRE_SHORT_DIAGNOSTIC_SMOKE_FIRST`.
- Pro evidence branch/commit: `125efd8`.
- Launch package evidence: `6d32639`.
- Exact code base required for this short diagnostic: `478325af8da10646f747f955a54378d53fffd3ef`.
- Resource caveat: `.376` may block GPU availability. Do not launch until the coordinator rechecks resources and confirms an already allocated child GPU context.

## Diagnostic Purpose

This package is bounded to one epoch and is meant to verify:

- resolved VideoMAE-S pretrain path and real load evidence;
- `data` and `pretrained` runtime resources;
- BVR route identity and absence of C3/C3-Pro/ABR/MDL route drift;
- native-axis geometry, mask semantics, adapter fixed-length padded bridge, and HeadV3 finite loss/gradient stability;
- no GT/teacher/cache/test-time leakage;
- no validation/evaluation, no `tools/test.py`, no mAP interpretation, and no full-train unlock.

## Changed Surface

- `configs/adatad/thumos/input_bvr_twb_dynamic_adapter_irregular_headv3_shortdiag.py`
  - Extends the formal BVR-TWB HeadV3 config.
  - Keeps the route label and BVR method unchanged.
  - Sets `diagnostic_only=True`, `full_train_unlocked=False`, `metric_claim=False`, `sparse_compute_claim=False`.
  - Sets one-epoch workflow, checkpoint disabled, validation loss/eval disabled, and small logging/runtime debug intervals.
  - Preserves AMP off, fp16 compression off, required VideoMAE-S pretrain, and HeadV3 stability flags.
- `tools/bvr_twb/validate_bvr_twb_shortdiag.py`
  - Validates resolved config and optional train log.
  - Fails closed on wrong commit marker/HEAD, missing pretrain file, resolved pretrain mismatch, no-pretrain warning, NaN/Inf/non-finite/cost nan, Python/runtime/CUDA/storage/resource failures, missing finite `Loss=...`, route drift tokens, eval/mAP markers, and full-train/deploy/paper claim words.
  - Always reports `full_train_unlocked=false`, `metric_claim=false`, and `sparse_compute_claim=false`.
- `tests/test_bvr_twb_shortdiag.py`
  - Covers config inheritance, pretrain, HeadV3 stability flags, one-epoch/checkpoint/eval locks, and validator rejection of missing finite loss, no-pretrain warning, and NaN.
- `logs/run_bvr_twb_478325a_shortdiag_n16r4.sh`
  - Intended for an already allocated child GPU context inside a route clean tree.
  - Does not call `sbatch`, `srun`, `scancel`, SSH, `tools/test.py`, or any hold release/cancel action.
  - Asserts exact HEAD unless `BVR_TWB_SHORTDIAG_ALLOW_HEAD_OVERRIDE=1`.
  - Asserts `data`, `pretrained`, required pretrain file, and `CUDA_VISIBLE_DEVICES`.
  - Runs py_compile, focused pytest, geometry contract validator, formal launch gate precheck, shortdiag config validator, one-epoch torchrun, stop-condition grep, and post-run log validator.

## Launch Boundary

Allowed next action after local verification and coordinator resource check:

`SHORT_DIAGNOSTIC_ONLY_REVIEW_EVIDENCE`

Still locked:

- formal full training;
- validation/test evaluation;
- `tools/test.py`;
- mAP/Avg-mAP/high-IoU claims;
- sparse compute/FLOPs/latency claims;
- deployment or paper claims;
- C3/C3-Pro/ABR/MDL mixing.

## Expected Evidence From Remote Short Diagnostic

The remote run must preserve:

- exact HEAD `478325af8da10646f747f955a54378d53fffd3ef`, unless the coordinator explicitly records an override;
- real pretrain file present at `pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth`;
- train log containing the required pretrain path and a checkpoint load marker;
- at least one finite `Loss=...` line;
- no NaN/Inf/non-finite/cost-nan/Traceback/RuntimeError/CUDA OOM/Killed/no-space/no-GPU markers;
- no mAP/eval/result JSON/test.py/claim/unlock markers.

No result from this diagnostic may be promoted into a method claim without a separate valid gate.
