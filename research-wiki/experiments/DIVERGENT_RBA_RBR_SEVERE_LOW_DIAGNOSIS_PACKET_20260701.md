# DIVERGENT RBA-RBR Severe-Low Diagnosis Packet - 2026-07-01

## Decision Context

- Route label: `DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3`.
- Route name: `RBA-RBR: Regret-Based Recoverable Bracketing`.
- Purpose: recoverable active sparse-frame acquisition, where a cheap first-pass bracket is only a soft prior and later probes can rescue high-regret / high-uncertainty boundary regions outside the first bracket.
- Current stage: bounded eval diagnostic only.
- Not a formal full train.
- No mAP/runtime/FLOPs/deploy/paper claim is unlocked.
- C3/C3-Pro results, selectors, labels, and attribution are not mixed into this route.

## Current Code And Branch State

- Local route-owned worktree: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_RBA_RBR_Worktree_20260701`.
- Local branch: `codex/divergent-rba-rbr-20260701`.
- Local evidence commit after first validation: `a8d533d`.
- Local severe-low packet commit before final eval completion: `769415d`.
- Remote N16R4 route-owned worktree: `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_Shortdiag_20260701_9311f49`.
- Remote evidence commit after first validation: `152ffed`.
- Remote severe-low packet commit before final eval completion: `0704fd9`.
- GitHub branch exists, but the local/remote route-owned worktrees are ahead of the tracked GitHub branch. Before asking Pro to inspect GitHub, the branch must be synchronized.

## Key Files For Review

- `configs/adatad/thumos/input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3.py`
- `configs/adatad/thumos/input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3_evaldiag.py`
- `opentad/acquisition/rba_rbr/open_tad_bridge.py`
- `opentad/acquisition/rba_rbr/risk_map.py`
- `opentad/acquisition/rba_rbr/soft_bracket.py`
- `opentad/acquisition/rba_rbr/probe_builder.py`
- `opentad/acquisition/rba_rbr/budget_controller.py`
- `opentad/acquisition/rba_rbr/regret_scorer.py`
- `opentad/acquisition/rba_rbr/validators.py`
- `opentad/acquisition/rba_rbr/metadata.py`
- `opentad/datasets/transforms/end_to_end.py`
- `opentad/models/backbones/backbone_wrapper.py`
- `tests/test_rba_rbr_core.py`
- `tests/test_rba_rbr_integration.py`
- `tools/rba_rbr/validate_rba_rbr_launch_gate.py`

## Remote Run Evidence

- Protected parent hold: `1118197 pcot_dbg2g`.
- Parent hold status: protected; not released, cancelled, replaced, or modified.
- GPU binding: GPU0 only, `CUDA_VISIBLE_DEVICES=0`.
- Child step: `1118197.539`.
- Job name: `rba_rbr_eval_g0`.
- Log directory: `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_Shortdiag_20260701_9311f49/logs/rba_rbr_evaldiag_564a6f3_gpu0_20260701_043542_+0800`.
- Log file: `srun-1118197.out`.

## Launch And Load Evidence

- Remote route-owned tests before launch: `20 passed`.
- Launch gate before launch: `gate_pass=true`, `full_train_unlocked=false`.
- Eval diagnostic config:
  - `end_epoch=4`.
  - `val_start_epoch=1`.
  - `val_eval_interval=2`.
  - `disable_checkpoint=False`.
  - N16R4 annotation path: `/data/home/sczc063/run/yuzibo/thumos14/annotations/thumos_14_anno.json`.
- Pretrained loading:
  - Log contains: `Loads checkpoint by local backend from path: pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth`.
  - Unexpected keys: classification head weights/bias.
  - Missing keys: adapter parameters, expected because adapter layers are newly added/trainable.
  - Therefore current evidence does not support the simple hypothesis that no pretraining was loaded.

## First Validation Result

- First validation completed all `1645/1645` windows.
- Bad-pattern count for Traceback, RuntimeError, OOM, killed, NaN, non-finite, ValueError, PermissionError, and FileNotFoundError: `0`.
- First diagnostic validation metric after epoch 1:
  - `Average-mAP=0.12%`.
  - `mAP@0.30=0.34%`.
  - `mAP@0.40=0.18%`.
  - `mAP@0.50=0.06%`.
  - `mAP@0.60=0.02%`.
  - `mAP@0.70=0.01%`.
- The child continued into epoch 2; no hard failure was observed at the time of this packet draft.

## Final Bounded Diagnostic Result

- Child step `1118197.539` completed normally.
- Slurm state: `COMPLETED|0:0`.
- Elapsed: `01:57:09`.
- Bad-pattern count for Traceback, RuntimeError, OOM, killed, NaN, non-finite, ValueError, PermissionError, and FileNotFoundError: `0`.
- Checkpoints/logs observed:
  - `.../input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3_evaldiag_only/gpu1_id0/checkpoint/epoch_1.pth`
  - `.../input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3_evaldiag_only/gpu1_id0/checkpoint/epoch_3.pth`
  - `.../input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3_evaldiag_only/gpu1_id0/log.json`
- Final bounded diagnostic metric after epoch 3:
  - `Average-mAP=4.42%`.
  - `mAP@0.30=10.92%`.
  - `mAP@0.40=6.35%`.
  - `mAP@0.50=3.16%`.
  - `mAP@0.60=1.28%`.
  - `mAP@0.70=0.37%`.
- Interpretation: execution is stable and pretraining is loaded, but the detector-health metric remains failure-scale severe-low. This triggers the repository severe-result gate before any RBA-RBR full train or route-level performance interpretation.

## Current Suspicion List

This is a diagnosis checklist, not a conclusion.

- The raw RGB low-resolution scout is deploy-visible but weak: it uses 32 sampled frames and derives actionness/uncertainty/transition from motion, contrast, and mean-change.
- Train/validation selector distribution may be mismatched:
  - train path: `method_base="random_trunc"`, `rba_rbr_train_value_labels=True`.
  - validation/test path: `method_base="sliding_window"`, `rba_rbr_train_value_labels=False`.
- The selected-frame axis and detector physical-time axis may still be inconsistent despite metadata plumbing:
  - generic `irregular_selected_positions` is detector-feature positions.
  - RBA-specific `rba_rbr_raw_selected_positions` is used for backbone time-axis metadata.
  - `remap_gt_to_selected_axis=False`.
- The adapter bridge is fixed-length padded back to `target_len=192`, so the method may currently behave more like a distorted sparse-to-fixed bridge than a clean recoverable detector input.
- The final bounded diagnostic recovered from `0.12%` to `4.42%`, but remains severe-low with normal loss and no hard runtime failure. This suggests a mechanism/geometry/protocol issue is more likely than simple launch failure.

## Questions For Pro / Severe Diagnosis

Ask GPT-5.5 Pro to inspect the synchronized GitHub branch and answer code-grounded, not speculative:

1. Does the current RBA-RBR implementation actually preserve the original recoverable-bracketing idea, or did the fixed-length adapter bridge / weak raw-scout turn it into an unstable sparse sampler?
2. Is there any coordinate, mask, physical-time, selected-axis, or post-processing bug that could explain near-zero mAP despite finite loss?
3. Is `remap_gt_to_selected_axis=False` correct for this route given `rba_rbr_feature_stride=2`, detector-feature centers, and raw-frame selected positions?
4. Does the validation/test pipeline feed the detector with the same temporal geometry assumed by the Adapter/Head?
5. Are the `selected_frame_inds`, `irregular_selected_positions`, `rba_rbr_raw_selected_positions`, and `rba_rbr_detector_feature_positions` internally consistent?
6. Is the deploy-visible raw-scout actionness/uncertainty/transition too weak or badly normalized for THUMOS, and what minimal replacement would remain deploy-visible?
7. What concrete diagnostic experiments should be run next before any formal long train?
8. Provide key code changes if there is an obvious blocker.

## Launch Decision Before Pro

- Do not launch RBA-RBR formal full training.
- Do not interpret the bounded diagnostic metric as a final route result or paper claim.
- Synchronize GitHub before any Pro request if using repository URLs, because local and N16R4 route-owned worktrees are ahead of the tracked GitHub branch.
- Send this packet plus current files/logs to Pro/Oracle for severe-result diagnosis before any further RBA-RBR long run or route-level conclusion.

## Local Sparse-Forward Precheck Audit - 2026-07-01 07:23:37 +08:00

Purpose:

- This update adds a minimal detector temporal-grid audit for the severe-low follow-up.
- It is intended to diagnose whether RBA-RBR detector feature positions are actually handed to the irregular Head on the native dense time axis.
- It is not a performance claim, not a metric interpretation, and not a full-train unlock.

Changed behavior:

- `IrregularActionFormer` now detects RBA-RBR metadata independently via `rba_rbr_*`.
- RBA-RBR grid construction uses `rba_rbr_detector_feature_positions` / `rba_rbr_detector_feature_valid_len`, even when generic `irregular_selected_positions` is present.
- RBA-RBR grid construction fails closed unless `irregular_native_axis=True`.
- RBA-RBR grid construction fails closed when the input mask true count does not equal the number of detector feature positions.
- If `RBA_RBR_GRID_AUDIT=1` and `RBA_RBR_GRID_AUDIT_PATH` are set, the detector appends JSONL rows labeled `DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3`.
- Existing BVR grid audit remains independently labeled and was covered by a regression test.

Verification:

- `python -m py_compile opentad\models\detectors\irregular_actionformer.py tests\test_rba_rbr_integration.py` passed.
- `python -m pytest tests\test_rba_rbr_integration.py -q` passed: `8 passed, 5 skipped`.
- `python -m pytest tests\test_rba_rbr_core.py tests\test_rba_rbr_integration.py -q` passed: `18 passed, 5 skipped`.
- A wildcard attempt `python -m pytest tests\test_rba_rbr*.py -q` did not run because PowerShell did not expand the path for pytest; the explicit file-list command above is the valid suite result.

Still locked:

- No training, remote sync, Slurm, Pro submission, or GPU job was run for this audit update.
- Local staging/commit and GitHub API sync were performed only after the focused tests passed, to keep the severe-result evidence visible for later Pro diagnosis.
- The optional route-owned CPU ledger diagnostic tool was deferred to keep this stage minimal and avoid broadening the implementation.
- Formal full training remains locked.
- Pro transport remains `INCOMPLETE`; no valid Pro severe-result diagnosis has been harvested.
- No final mAP, runtime/FLOPs, deployment, or paper claim is unlocked.

## Grid-Audit GitHub Evidence Branch - 2026-07-01 07:39:58 +08:00

Because ordinary `git push` could not safely update the existing GitHub branch, the grid-audit evidence was synchronized without force-pushing:

- Existing GitHub branch `codex/divergent-rba-rbr-20260701` currently points to remote-only commit `1b26de8a7c606d9305cfa228b7f87e920529aa6a`.
- A new branch was created from that remote commit: `codex/divergent-rba-rbr-grid-audit-aadf9708-20260701`.
- The branch was updated with the five local grid-audit files from commit `aadf9708`.
- Confirmed GitHub ref after Contents API sync: `0a4bf5747de24f822ae23183e89f5cce234e69a8`.
- Review URL: `https://github.com/yuzbo/pc-ot-mras-r-series-opentad/tree/codex/divergent-rba-rbr-grid-audit-aadf9708-20260701`.

This branch is intended for Pro/Oracle severe-result diagnosis and remote sparse-forward precheck context only. It does not unlock full training or any metric/runtime/FLOPs/deploy/paper claim.

## Grid-Audit Remote PRECHECK_ONLY - 2026-07-01 07:48:48 +08:00

The grid-audit update was also checked in a Linux/OpenTAD no-GPU environment:

- Remote route-owned precheck copy: `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_GridAuditPrecheck_20260701_600fc8f2`.
- Source tree: copied from `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_Shortdiag_20260701_9311f49`, then overlaid with the five grid-audit files from the evidence branch.
- Package: `/data/run01/sczc063/yuzibo/rba_rbr_grid_audit_600fc8f2.tar`.
- `python -m py_compile opentad/models/detectors/irregular_actionformer.py tests/test_rba_rbr_integration.py` passed.
- `python -m pytest tests/test_rba_rbr_core.py tests/test_rba_rbr_integration.py -q` passed: `23 passed in 50.13s`.
- Logs:
  - `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_GridAuditPrecheck_20260701_600fc8f2/logs/rba_rbr_grid_audit_precheck_600fc8f2/py_compile.log`.
  - `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_GridAuditPrecheck_20260701_600fc8f2/logs/rba_rbr_grid_audit_precheck_600fc8f2/pytest.log`.

This verifies that the sparse-forward temporal-grid audit is runnable on N16R4/Linux. It does not diagnose or fix the severe-low mAP by itself, does not run training, does not run `tools/test.py`, and does not unlock formal full training or any claim.

## Coverage-Guard First Validation Severe-Low - 2026-07-01 13:39:19 +08:00

Run context:

- Corrected coverage-guard branch commit deployed remotely: `e6de60e9`.
- GitHub evidence branch with this first-validation packet and Pro prompt: `codex/divergent-rba-rbr-guard-severe-firsteval-15303a3c-20260701`.
- GitHub URL: `https://github.com/yuzbo/pc-ot-mras-r-series-opentad/tree/codex/divergent-rba-rbr-guard-severe-firsteval-15303a3c-20260701`.
- Protected parent hold: `1118197 pcot_dbg2g`; not released, cancelled, replaced, or modified.
- Child step: `1118197.560`.
- Job name: `rba_guard_g0`.
- GPU binding: protected hold GPU0 only; GPU1/C3 not touched.
- Log directory: `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_GuardDiag_20260701_e6de60e9_bundle/logs/rba_rbr_guard_evaldiag_manual_parallel_holdg0_20260701_123213_+0800`.
- Evidence files: `srun-1118197.out`, `train.log`, `rba_rbr_grid_audit.jsonl`.

First validation metric:

- Completed at `2026-07-01 13:39:19 +08:00`.
- `Average-mAP=0.22%`.
- `mAP@0.3/0.4/0.5/0.6/0.7 = 0.66/0.30/0.09/0.04/0.01`.
- `3325` ground-truth instances.
- `411700` predictions.

Health evidence:

- Epoch 0 final: `[000][00199/00199] Loss=2.3574 ... mem=9344MB`.
- Epoch 1 final: `[001][00199/00199] Loss=1.8465 ... mem=9344MB`.
- Training continued after first validation into epoch 2; latest inspected line: `[002][00120/00199] Loss=1.6674 ... mem=9344MB`.
- No Traceback/OOM/NaN/non-finite pattern was observed in the inspected log.

Coverage-guard audit at inspection time:

- Rows: `2183`.
- Labels: only `DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3`.
- Statuses: only `PASS_RBA_RBR_NATIVE_AXIS_POSITIONS_ENTERED_MODEL`.
- `raw_valid_k`: min `60`, p05 `78`, p50 `85`, p95 `91`, max `94`, avg `84.56`.
- `mask_true_count`: min `30`, p05 `39`, p50 `43`, p95 `46`, max `47`, avg `42.53`.
- `meta_detector_feature_position_count`: min `30`, p05 `39`, p50 `43`, p95 `46`, max `47`, avg `42.53`.
- `selected_max_gap_after_guard`: min `1`, p05 `14`, p50 `16`, p95 `16`, max `16`, avg `15.58`.
- `max_detector_gap_after_guard`: min `2.0`, p05 `20.5`, p50 `23.0`, p95 `24.0`, max `24.0`, avg `22.52`.
- `guard_addition_count`: min `3`, p05 `13`, p50 `29`, p95 `49`, max `54`, avg `29.88`.
- `mask_lt32=1`, `rawgap_gt16_after=0`, `detgap_gt24_after=0`.

Interpretation:

- The guard restored detector feature count and raw/detector gap constraints.
- The first-validation metric is still failure-scale severe-low.
- Therefore the current evidence does not support the narrower hypothesis that the collapse was only caused by large temporal holes or too few detector feature positions.
- The remaining suspected causes are deeper selector quality, sparse-to-fixed adapter bridge semantics, train/validation geometry mismatch, coordinate/loss/postprocess alignment, or raw-scout weakness. These remain suspicions, not conclusions.

Pro/Oracle severe-result gate:

- Rosetta probing failed on all currently requested/known ports:
  - `rosetta probe --port 9223 --host 127.0.0.1`: connection refused.
  - `rosetta probe --port 9333 --host 127.0.0.1`: connection refused.
  - `rosetta probe --port 9222 --host 127.0.0.1`: connection refused.
- Oracle provider check failed for Pro models because the OpenAI provider is not ready and `OPENAI_API_KEY` is missing.
- This is `INCOMPLETE/UNAVAILABLE`, not an accepted Pro diagnosis.

Launch decision:

- Let the current `SHORT_DIAGNOSTIC_ONLY` run continue to its scheduled epoch-3/final diagnostic if it remains finite.
- Do not launch RBA-RBR formal/full training.
- Do not launch a new RBA-RBR long follow-up run.
- Do not make metric/runtime/FLOPs/sparse-compute/deploy/paper claims.
- Prepare a severe-result Pro prompt/context package for when Rosetta/Oracle transport becomes available.

## Coverage-Guard Final Diagnostic Severe-Low - 2026-07-01 14:47:42 +08:00

Run context:

- Route label: `DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3`.
- Corrected coverage-guard branch commit deployed remotely: `e6de60e9`.
- Latest GitHub evidence branch for Pro diagnosis: `codex/divergent-rba-rbr-guard-complete-39073521-20260701`.
- Latest GitHub URL: `https://github.com/yuzbo/pc-ot-mras-r-series-opentad/tree/codex/divergent-rba-rbr-guard-complete-39073521-20260701`.
- Protected parent hold: `1118197 pcot_dbg2g`; not released, cancelled, replaced, or modified by this evidence update.
- Completed child step: `1118197.560`.
- Job name: `rba_guard_g0`.
- GPU binding: protected hold GPU0 only; GPU1/C3 not touched.
- Log directory: `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_GuardDiag_20260701_e6de60e9_bundle/logs/rba_rbr_guard_evaldiag_manual_parallel_holdg0_20260701_123213_+0800`.
- Evidence files: `srun-1118197.out`, `train.log`, `rba_rbr_grid_audit.jsonl`.

Metric evidence:

- First observed Average-mAP line at `2026-07-01 13:39:19 +08:00`: `0.22%`.
- Final observed Average-mAP line at `2026-07-01 14:47:42 +08:00`: `7.45%`.
- Final `mAP@0.3/0.4/0.5/0.6/0.7 = 17.34/11.47/5.69/2.09/0.65`.
- Final evaluator counts: `411700` predictions and `3325` GT instances.
- `Training Over` count: `1`.
- Hard error count: `0`.

Final coverage-guard audit:

- Rows: `4090`.
- Labels: only `DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3`.
- Statuses: all `PASS_RBA_RBR_NATIVE_AXIS_POSITIONS_ENTERED_MODEL`.
- `raw_valid_k` avg: `84.6235`.
- `mask_true_count` avg: `42.5601`.
- `selected_max_gap_after_guard`: max `16`, avg `15.5897`.
- `max_detector_gap_after_guard`: max `24`, avg `22.5460`.

Interpretation:

- The guard diagnostic completed normally and produced a better final detector-health number than its first validation, but `7.45%` Average-mAP remains failure-scale relative to the RBA-RBR baseline targets and known fixed-budget references.
- The final audit confirms that RBA-RBR native-axis detector positions entered the model and that the coverage/gap guard held through the full diagnostic.
- Because coverage/gap constraints are restored while detector performance remains severely low, the route should be diagnosed as a deeper RBA-RBR mechanism, geometry, bridge, assignment, or post-processing failure rather than as a simple missing-audit or launch failure.

Decision:

- Formal/full RBA-RBR training remains locked.
- No mAP/runtime/FLOPs/sparse-compute/deploy/paper claim is unlocked.
- No route conclusion is unlocked.
- A valid GPT-5.5 Pro severe-result diagnosis is required before any long RBA-RBR follow-up or route-level conclusion, unless the user gives an explicit same-scope override.

## Local Postprocess Candidate Guard Repair - 2026-07-01 14:09:53 +08:00

Motivation:

- The corrected coverage guard restored detector feature count and gap constraints, but first validation still emitted `411700` predictions and `Average-mAP=0.22%`.
- This makes inherited post-processing candidate explosion a concrete next diagnostic variable.

Local repair:

- `IrregularActionFormer` now has an RBA-specific `rba_rbr_postprocess_guard` resolver and guarded candidate selector.
- The guard fails closed unless `require_rba_meta=True` is satisfied by RBA-RBR metadata.
- The guarded path caps raw proposals by deploy-visible detector scores, then caps per-class candidates, then caps total candidates.
- `RBA_RBR_POSTPROCESS_AUDIT=1` plus `RBA_RBR_POSTPROCESS_AUDIT_PATH` records JSONL rows labeled `DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3`.
- The RBA config sets `pre_nms_topk=512`, `raw_proposal_cap=1024`, `per_class_topk=32`, `total_candidate_cap=512`, and `min_score=0.001`.
- The RBA launch gate now requires these guard fields.

Verification:

- `python -m py_compile opentad\models\detectors\irregular_actionformer.py tools\rba_rbr\validate_rba_rbr_launch_gate.py tests\test_rba_rbr_integration.py configs\adatad\thumos\input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3.py` -> pass.
- `python -m pytest tests\test_rba_rbr_integration.py -q` -> `8 passed, 7 skipped`.
- `python tools\rba_rbr\build_synthetic_ledgers.py --out-dir tools\rba_rbr\.tmp_rba_rbr_guard_fix_ledgers --overwrite` -> pass.
- `python tools\rba_rbr\validate_rba_rbr_launch_gate.py --config configs\adatad\thumos\input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3.py --precheck-summary tools\rba_rbr\.tmp_rba_rbr_guard_fix_ledgers\summary.json` -> `gate_pass=true`, `full_train_unlocked=false`.
- `git diff --check` on changed files -> pass with LF/CRLF warnings only.

Decision:

- This repair does not prove RBA-RBR works and does not unlock formal/full training.
- It prepares a next `SHORT_DIAGNOSTIC_ONLY` run to test whether candidate explosion is a dominant failure mode after coverage/gap repair.
- Final mAP, runtime/FLOPs, sparse-compute, deploy, and paper claims remain locked pending valid Pro diagnosis or explicit user override.

GitHub evidence:

- Ordinary `git push` failed with `Recv failure: Connection was reset`.
- Evidence branch: `codex/divergent-rba-rbr-postprocess-guard-546ed615-20260701`.
- Local commit: `546ed6159c2155633981050ae102547de03ea6bc`.
- GitHub synced ref: `6ba723abe4f5b2eca7c38f71c5b14872d637dbaa`.
- URL: `https://github.com/yuzbo/pc-ot-mras-r-series-opentad/tree/codex/divergent-rba-rbr-postprocess-guard-546ed615-20260701`.

## Non-GPU Coordinate/Budget Audit Tool - 2026-07-01 15:12:35 +08:00

Purpose:

- Added a CPU-only diagnostic CLI, `tools/rba_rbr/audit_coordinate_budget.py`, for the severe-low `7.45%` RBA-RBR result.
- It consumes either existing RBA-RBR detector grid-audit JSONL rows or built-in synthetic rows.
- It summarizes `raw_valid_k`, detector valid count, raw/detector gap statistics, route/status counts, and whether the effective detector density is far below the fixed 384/192 50% reference.
- It also runs a synthetic coordinate-closure check: raw selected native-axis positions -> detector feature centers -> synthetic native-axis segment coverage. This uses only synthetic segments and explicitly does not use validation/test GT.

Claim and gate status:

- Non-GPU only; no training, remote sync, Slurm, `tools/test.py`, evaluator change, postprocess runtime change, mAP/runtime/FLOPs/deploy/paper claim, or full-train unlock.
- Claim lock wording is fixed as `rba_rbr_non_gpu_coordinate_budget_audit_only_no_metric_runtime_deploy_or_paper_claim`.
- Pro gate is not waived: `valid_gpt_5_5_pro_severe_result_diagnosis_required_before_long_followup_or_route_conclusion`.

How this informs next steps:

- `postprocess-guard shortdiag`: if the audit confirms detector density is still far below the 50% reference while postprocess candidates remain huge, the already prepared postprocess guard short diagnostic is the lowest-risk next GPU check, but it remains diagnostic-only and still needs the severe-result Pro context before any long follow-up.
- `matched low-budget uniform`: if RBA-RBR detector count is about `42` instead of the 50% detector reference `96`, run a matched low-budget uniform diagnostic with comparable detector valid count to separate RBA mechanism failure from simply operating at a much lower effective detector density.
- `K=192 forced coverage`: force raw selected `K=192` with the same detector/adapter path to test whether the failure is mainly density/coverage, or whether the adapter bridge / coordinate / loss / postprocess semantics remain broken even at the standard fixed 50% raw budget.
- `coordinate closure`: use the synthetic closure result plus any real grid-audit JSONL summary to focus Pro review on native-axis alignment, detector feature centers, assignment/loss coordinates, and proposal time conversion without touching validation/test GT.

Verification:

- `python -m py_compile tools\rba_rbr\audit_coordinate_budget.py tests\test_rba_rbr_diagnostics.py` passed.
- `python -m pytest tests\test_rba_rbr_diagnostics.py -q` passed: `5 passed`.

Still locked:

- Formal/full RBA-RBR training remains locked.
- Severe-result GPT-5.5 Pro diagnosis remains required before any long RBA-RBR follow-up or route-level conclusion.
- No metric/runtime/FLOPs/sparse-compute/deploy/paper claim is unlocked.
