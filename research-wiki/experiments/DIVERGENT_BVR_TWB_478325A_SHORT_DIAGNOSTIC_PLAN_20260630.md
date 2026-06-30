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
  - `python -m pytest tests/test_bvr_twb_shortdiag.py tests/test_bvr_twb_validators.py tests/test_bvr_twb_opentad_pipeline.py tests/test_bvr_twb_geometry_contracts.py -q` -> superseded by the 2026-06-30 evidence-freeze rerun below: `35 passed, 12 skipped`
  - `git diff --check` -> clean
  - `bash -n logs/run_bvr_twb_478325a_shortdiag_n16r4.sh` -> pass
- Contamination handling: one writable worker accidentally created the six short-diagnostic files in the shared root. The coordinator copied the exact files into this route-owned worktree, verified matching content, removed only those exact shared-root copies, and did not stage/commit/push from the shared root.
- Resource state: formal/diagnostic BVR launch remains blocked. Protected parent hold `1118197 pcot_dbg2g` must not be released. Current C3 child step `1118197.376 cadf_combo_g0` is still running with `AllocTRES=gres/gpu=2`, so GPU1 is not Slurm-safe for BVR even if it appears physically idle.

## 2026-06-30 Execution Boundary Fix

- Fix branch: `codex/divergent-bvr-twb-shortdiag-execfix-20260630`.
- Fix worktree: `OpenTAD_BVR_TWB_ShortDiagExecFix_Worktree_20260630`.
- Audit-triggered ambiguity: committed shortdiag package branches advance HEAD beyond the expected code base `478325af8da10646f747f955a54378d53fffd3ef`. A strict `HEAD == 478325a` check is ambiguous for a committed shortdiag package branch and should only be used for pure overlay mode.
- Approved execution modes:
  - `exact_base_overlay`: run the shortdiag files as an overlay on a clean tree whose current HEAD is exactly `478325af8da10646f747f955a54378d53fffd3ef`.
  - `descendant_shortdiag_package`: run a committed package branch whose current package HEAD descends from `478325af8da10646f747f955a54378d53fffd3ef`.
- Rejected execution mode: any package HEAD that neither equals nor descends from `478325af8da10646f747f955a54378d53fffd3ef`.
- Wrapper behavior: records both `expected_base_commit` and `package_head` in the train log before torchrun. It requires exact HEAD only when `BVR_TWB_SHORTDIAG_PURE_478325A_OVERLAY=1`; otherwise it accepts exact-base overlay or descendant shortdiag package mode.
- Validator behavior: reports `expected_base_commit`, `package_head`, `package_descends_from_expected_base`, and `shortdiag_execution_mode`. The backward-compatible `--expected-commit` argument is treated as an expected base commit, not an impossible package HEAD equality requirement.
- Claim state after this fix: still diagnostic-only; formal full training, validation/test evaluation, sparse-compute claims, deployment claims, paper claims, and metric claims remain locked.

## 2026-06-30 Evidence Freeze Rerun

Timestamp: `2026-06-30 09:22:41 +08:00`.

Scope: evidence/documentation-only rerun in owned worktree `OpenTAD_BVR_TWB_ShortDiagExecFix_Worktree_20260630` on branch `codex/divergent-bvr-twb-shortdiag-execfix-20260630`. No training, no `tools/test.py`, no evaluation, no remote sync, no Slurm, no Pro, and no shared-repository write was performed.

Frozen local non-GPU evidence:

- `python -m py_compile configs/adatad/thumos/input_bvr_twb_dynamic_adapter_irregular_headv3_shortdiag.py tools/bvr_twb/validate_bvr_twb_shortdiag.py tests/test_bvr_twb_shortdiag.py` -> pass.
- `python -m pytest tests/test_bvr_twb_shortdiag.py tests/test_bvr_twb_validators.py tests/test_bvr_twb_opentad_pipeline.py tests/test_bvr_twb_geometry_contracts.py -q` -> `35 passed, 12 skipped in 5.82s`.
- `python tools/bvr_twb/validate_bvr_twb_shortdiag.py --config configs/adatad/thumos/input_bvr_twb_dynamic_adapter_irregular_headv3_shortdiag.py --allow-missing-pretrain` -> `gate_pass=true`, `config_valid=true`, `checkpoint_disabled=true`, `eval_disabled=true`, `workflow_end_epoch=1`, `allowed_next_action=SHORT_DIAGNOSTIC_ONLY_REVIEW_EVIDENCE`, `full_train_unlocked=false`, `metric_claim=false`, `sparse_compute_claim=false`.
- Validator package/base evidence from that rerun: `expected_base_commit=478325af8da10646f747f955a54378d53fffd3ef`, `package_head=0314c0c73b30a55a600269498d27cc294b8032f3`, `package_descends_from_expected_base=true`, `shortdiag_execution_mode=descendant_shortdiag_package`.
- Local pretrain evidence from the same `--allow-missing-pretrain` validator run: `resolved_pretrain=pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth`, `pretrain_exists=false`. This is allowed only for local no-GPU verification and does not prove remote runtime pretrain availability or load.
- `bash --version` -> GNU bash `5.0.17(1)-release`; `bash -n logs/run_bvr_twb_478325a_shortdiag_n16r4.sh` -> pass.
- `git diff --check` -> clean.
- `git merge-base --is-ancestor 478325af8da10646f747f955a54378d53fffd3ef HEAD` -> pass (`BASE_IS_ANCESTOR=true`).

This rerun freezes the focused pytest target count as `35 passed, 12 skipped`. The previous `31 passed, 12 skipped` count is superseded by this rerun. Any later documentation-only commit may advance `package_head`; the execution boundary remains descendant shortdiag package mode unless pure overlay mode is explicitly requested.

Claim state remains unchanged: no mAP, no runtime/FLOPs/latency, no deploy, no paper, no sparse-compute, and no formal-train claim is unlocked. Formal train remains locked.

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
  - Records `shortdiag_expected_base_commit=478325af8da10646f747f955a54378d53fffd3ef`, `shortdiag_execution_boundary=expected_base_or_descendant_shortdiag_package`, and the audit package HEAD `2b4d48be7b6b44c8e4f20946005b508bb9e62587`.
  - Sets one-epoch workflow, checkpoint disabled, validation loss/eval disabled, and small logging/runtime debug intervals.
  - Preserves AMP off, fp16 compression off, required VideoMAE-S pretrain, and HeadV3 stability flags.
- `tools/bvr_twb/validate_bvr_twb_shortdiag.py`
  - Validates resolved config and optional train log.
  - Fails closed on wrong base commit marker, wrong package ancestry, missing pretrain file, resolved pretrain mismatch, no-pretrain warning, NaN/Inf/non-finite/cost nan, Python/runtime/CUDA/storage/resource failures, missing finite `Loss=...`, route drift tokens, eval/mAP markers, and full-train/deploy/paper claim words.
  - Accepts either exact-base overlay mode or descendant shortdiag package mode, and reports both current package HEAD and expected base commit.
  - Always reports `full_train_unlocked=false`, `metric_claim=false`, and `sparse_compute_claim=false`.
- `tests/test_bvr_twb_shortdiag.py`
  - Covers config inheritance, pretrain, HeadV3 stability flags, one-epoch/checkpoint/eval locks, exact-base mode, descendant package mode, wrong-base rejection, exact-only overlay rejection, and validator rejection of missing finite loss, no-pretrain warning, and NaN.
- `logs/run_bvr_twb_478325a_shortdiag_n16r4.sh`
  - Intended for an already allocated child GPU context inside a route clean tree.
  - Does not call `sbatch`, `srun`, `scancel`, SSH, `tools/test.py`, or any hold release/cancel action.
  - Asserts either exact-base overlay or descendant package ancestry. It asserts exact HEAD only when `BVR_TWB_SHORTDIAG_PURE_478325A_OVERLAY=1`.
  - Writes `expected_base_commit`, `package_head`, and `execution_mode` to the shortdiag train log.
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

- either exact HEAD `478325af8da10646f747f955a54378d53fffd3ef` in pure overlay mode, or a package HEAD that descends from `478325af8da10646f747f955a54378d53fffd3ef` in descendant shortdiag package mode;
- train log lines recording both `expected_base_commit` and current `package_head`;
- real pretrain file present at `pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth`;
- train log containing the required pretrain path and a checkpoint load marker;
- at least one finite `Loss=...` line;
- no NaN/Inf/non-finite/cost-nan/Traceback/RuntimeError/CUDA OOM/Killed/no-space/no-GPU markers;
- no mAP/eval/result JSON/test.py/claim/unlock markers.

No result from this diagnostic may be promoted into a method claim without a separate valid gate.
