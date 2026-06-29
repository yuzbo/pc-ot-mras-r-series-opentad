# DIVERGENT ABR Full-Code Implementation 20260629

**Timestamp**: 2026-06-29 22:19:23 +08:00  
**Implementation worktree**: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_ABR_FullCode_Worktree_20260629`  
**Branch**: `codex/divergent-abr-fullcode-20260629`  
**Base commit**: `588b2729d7570732b4dc281aa4cd3a2dd7e784ac`  
**Route label**: `DIVERGENT_INNOVATION_ABR_DO_NOT_MERGE_WITH_C3`

## Supersession

The old minimal worktree `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_ABR_Worktree_20260629` is superseded and is now read-only evidence only. The implementation target is the full OpenTAD source tree in `OpenTAD_ABR_FullCode_Worktree_20260629`.

## Changed Surfaces

- `opentad/acquisition/abr/**`: ABR route-local package with config/types, `BracketState`, `ABRRoundLedger`, selector, policy, provenance validators, real sparse handoff validator, and OpenTAD integration helper.
- `opentad/datasets/transforms/end_to_end.py`: real `LoadFrames` integration for `method="abr_active_bracket_refinement"` before `DecordDecode`. It reuses existing random-trunc/sliding-window dense-window construction and sets sparse `frame_inds`.
- `opentad/datasets/transforms/formatting.py`: real `Collect` default metadata now includes ABR ledger and original-time irregular metadata.
- `configs/adatad/thumos/input_abr_active_bracket_refinement_adapter_irregular_headv3.py`: full-code candidate config inheriting the existing irregular headv3 config.
- `tools/abr/audit_abr_pipeline_precheck.py`: synthetic selector/handoff precheck plus fail-closed torch/OpenTAD gate.
- `tools/abr/validate_abr_launch_gate.py`: route-label/no-forbidden-token/validated-precheck gate.
- `tests/test_abr_core.py`, `tests/test_abr_pipeline_and_gate.py`: route identity, no-leakage, selector, bracket state, round ledger, dynamic K, handoff, real `LoadFrames` static/subprocess smoke, and gate tests.

## Self-Check

- Changed surface category: input sampling/acquisition policy and dataloader handoff only.
- Not changed: Adapter internals, detector head logic, loss/assignment, test-time post-processing, BVR, MDL, C3 modules, remote launchers, Slurm scripts.
- Protocol: ABR selection uses only current video/window-derived low-cost scout or deterministic fallback. It rejects validation/test GT, teacher logits/features, prediction cache, raw detector feedback, and dense backbone handoff.
- Train boundary: train `random_trunc` may use train GT for existing windowing/supervision, but ABR selection receives only the resulting dense window and does not use GT values for selected positions.
- Sparse handoff: `frame_inds = dense_window[keep_positions]`; selected positions are sorted unique original dense-time coordinates; padding duplicates are invalid and do not count in `abr_selected_valid_k` or mask sum.
- Cost ledger: one detector forward is recorded; scout/acquisition proxy costs, per-round selected K, deadline, and stop reason are included.
- Claim status: no mAP, runtime, FLOPs, deployment, or paper claim.

## Verification

- `python -m pytest tests/test_abr_core.py tests/test_abr_pipeline_and_gate.py -q`
  - Result: `10 passed, 1 skipped in 1.23s`.
  - Skip reason: real `LoadFrames` subprocess smoke is locked by local torch DLL import failure; static integration check still passed.
- `python -m py_compile opentad\acquisition\__init__.py opentad\acquisition\abr\__init__.py opentad\acquisition\abr\types.py opentad\acquisition\abr\policy.py opentad\acquisition\abr\selector.py opentad\acquisition\abr\validators.py opentad\acquisition\abr\integration.py opentad\datasets\transforms\end_to_end.py opentad\datasets\transforms\formatting.py tools\abr\audit_abr_pipeline_precheck.py tools\abr\validate_abr_launch_gate.py configs\adatad\thumos\input_abr_active_bracket_refinement_adapter_irregular_headv3.py`
  - Result: pass.
- `python tools\abr\audit_abr_pipeline_precheck.py --out-dir .tmp_abr_mock_precheck --overwrite --mock-only`
  - Result: pass as `PASS_MOCK_PRECHECK_ONLY`, `dynamic_k_nonconstant=true`, `real_sparse_handoff_ok=true`, `detector_forward_count=1`, easy/rich K = `13/24`.
- `python tools\abr\audit_abr_pipeline_precheck.py --out-dir .tmp_abr_real_precheck --overwrite`
  - Result: fail-closed `status=LOCKED`; local torch import fails with `OSError: [WinError 1114] ... c10.dll`.
- `python tools\abr\validate_abr_launch_gate.py --precheck-json .tmp_abr_mock_precheck\abr_precheck_summary.json`
  - Result: locked, because mock-only status is not `PASS_PRECHECK_ONLY`.
- `python tools\abr\validate_abr_launch_gate.py --precheck-json .tmp_abr_real_precheck\abr_precheck_summary.json`
  - Result: locked, because real precheck status is `LOCKED`.
- `git diff --check`
  - Result: no whitespace errors; Windows line-ending warnings only for edited transform files.

## Still Locked

- No remote sync, Slurm, training, evaluation, `tools/test.py`, stage, commit, or push has been run.
- Remote `PRECHECK_ONLY` remains locked until a real torch/OpenTAD environment produces `PASS_PRECHECK_ONLY`.
- Required post-implementation external review gates are not completed.

## Huygens Blocker Fix Snapshot

**Timestamp**: 2026-06-30 00:16:21 +08:00  
**Reviewer gate**: Huygens final read-only review returned `FAIL`; this snapshot fixes the blocking findings locally only.  
**Delegation note**: direct fix by the ABR single writable code owner, because the user explicitly assigned this owner to complete the route fix stage.

Fixes:

- Coordinate contract: ABR handoff now distinguishes `selected_positions_window_local`, `selected_positions_original_dense`, and `frame_inds_raw`. The validator requires `selected_positions_original_dense == dense_window[selected_positions_window_local]` and `frame_inds_raw == selected_positions_original_dense` for the valid prefix.
- Metadata: integration records both local and original/global positions. `abr_selected_positions` remains local for backward compatibility, while `abr_selected_positions_original_dense` and `abr_frame_inds_raw` preserve raw/original frame coordinates.
- GT protection: real `LoadFrames` strips `gt_segments` and `gt_labels` from the selector payload before calling ABR. Train config may keep `abr_allow_gt_after_selection=True` for existing post-selection annotation handling, but val/test config now sets `abr_allow_gt_after_selection=False`.
- Precheck: mock precheck now covers a nonzero dense window `[120, 215]`, validates local/global/raw frame relationships, checks val/test GT rejection, and records those fields in JSON.
- Launch gate: `PASS_PRECHECK_ONLY` now requires `nonzero_window_ok=True` and `val_test_gt_rejection_ok=True`; mock-only summaries remain locked.

Verification after fixes:

- `python -m pytest tests/test_abr_core.py tests/test_abr_pipeline_and_gate.py -q`
  - Result: `13 passed, 1 skipped in 1.31s`.
- `python -m py_compile opentad\acquisition\abr\types.py opentad\acquisition\abr\integration.py opentad\acquisition\abr\validators.py opentad\datasets\transforms\end_to_end.py configs\adatad\thumos\input_abr_active_bracket_refinement_adapter_irregular_headv3.py tools\abr\audit_abr_pipeline_precheck.py tools\abr\validate_abr_launch_gate.py tests\test_abr_core.py tests\test_abr_pipeline_and_gate.py`
  - Result: pass.
- `python tools\abr\audit_abr_pipeline_precheck.py --out-dir .tmp_abr_mock_precheck --overwrite --mock-only`
  - Result: `PASS_MOCK_PRECHECK_ONLY`; `nonzero_window_ok=true`, `val_test_gt_rejection_ok=true`, `nonzero_window_start=120`, `nonzero_window_end=215`, `detector_forward_count=1`.
- `python tools\abr\audit_abr_pipeline_precheck.py --out-dir .tmp_abr_real_precheck --overwrite`
  - Result: fail-closed `LOCKED`; local torch import fails with `OSError: [WinError 1114] ... c10.dll`.
- `python tools\abr\validate_abr_launch_gate.py --precheck-json .tmp_abr_mock_precheck\abr_precheck_summary.json`
  - Result: locked because status is `PASS_MOCK_PRECHECK_ONLY`, not `PASS_PRECHECK_ONLY`.
- `python tools\abr\validate_abr_launch_gate.py --precheck-json .tmp_abr_real_precheck\abr_precheck_summary.json`
  - Result: locked because status is `LOCKED`.
- `git diff --check`
  - Result: no whitespace errors; Windows line-ending warnings only for edited transform files.

Still locked after blocker fix:

- No remote sync, Slurm, training, evaluation, `tools/test.py`, stage, commit, or push.
- No mAP/runtime/FLOPs/deploy/paper claim.
- Remote `PRECHECK_ONLY` requires a real torch/OpenTAD environment to produce `PASS_PRECHECK_ONLY`.

## Final Review And Main-Process Reverification

**Timestamp**: 2026-06-30 00:25:51 +08:00  
**Final read-only review**: Anscombe / `019f142e-23ec-7651-a976-8b0de80041e1` returned `PASS_SUBAGENT_FINAL_REVIEW_ONLY` with no blockers.

Main-process reverification recorded:

- `python -m pytest tests/test_abr_core.py tests/test_abr_pipeline_and_gate.py -q`
  - Result: `13 passed, 1 skipped in 1.27s`.
- `python -m py_compile` on ABR files/config/tools
  - Result: pass.
- Mock precheck
  - Result: `PASS_MOCK_PRECHECK_ONLY`.
  - Required summary fields: `nonzero_window_ok=true`, `val_test_gt_rejection_ok=true`, `detector_forward_count=1`, `dynamic_k_nonconstant=true`.
- Real precheck
  - Result: `LOCKED` due local Windows torch `c10.dll` / WinError 1114.
- Launch gate
  - Result: mock summary locked and real summary locked, both as expected.
- `git diff --check`
  - Result: no whitespace error; only line-ending warnings.

Allowed next action:

- Linux/N16R4 `PRECHECK_ONLY` only, after main process commit/push and any required coordination.

Still locked:

- Full train, mAP claims, deploy claims, paper claims, runtime claims, and sparse-compute claims remain locked.

## Linux PRECHECK Import Fix

**Timestamp**: 2026-06-30 00:55:52 +08:00
**Remote attempt**: protected hold `1118197` GPU1, `PRECHECK_ONLY`, logdir `/data/home/sczc063/run/yuzibo/abr_precheck_logs/abr_gpu1_precheck_retry3_20260630_004513`.
**Commit tested remotely**: `48ed384`.
**Remote failure**: real pytest reached ABR local tests but failed before ABR selector logic because the clean clone lacked `opentad/datasets/transforms/pseudo_boundary.py`, while `opentad/datasets/transforms/end_to_end.py` imports that shared transform helper.

Fix:

- Added `opentad/datasets/transforms/pseudo_boundary.py` to the ABR full-code worktree using the compatible shared transform helper lineage from `OpenTAD_Back`.
- Kept it as a shared transform dependency, not as BVR/MDL/C3 route logic and not as an ABR algorithm change.
- Added a path-level clean-clone dependency test in `tests/test_abr_pipeline_and_gate.py` that imports the helper without requiring torch and checks `load_pseudo_boundary_cache`, `select_pseudo_boundary_hybrid_positions`, and `select_pseudo_boundary_snap_positions`.

Verification:

- RED check before helper add: `python -m pytest tests/test_abr_pipeline_and_gate.py -q`
  - Result: failed exactly at `test_pseudo_boundary_helper_exists_for_clean_clone_import_dependency` because `pseudo_boundary.py` did not exist.
- GREEN focused check: `python -m pytest tests/test_abr_pipeline_and_gate.py -q`
  - Result: `7 passed, 1 skipped in 1.03s`.
- Full ABR check: `python -m pytest tests/test_abr_core.py tests/test_abr_pipeline_and_gate.py -q`
  - Result: `14 passed, 1 skipped in 1.06s`.
- `python -m py_compile opentad\datasets\transforms\pseudo_boundary.py opentad\datasets\transforms\end_to_end.py opentad\acquisition\abr\types.py opentad\acquisition\abr\integration.py opentad\acquisition\abr\validators.py tools\abr\audit_abr_pipeline_precheck.py tools\abr\validate_abr_launch_gate.py configs\adatad\thumos\input_abr_active_bracket_refinement_adapter_irregular_headv3.py tests\test_abr_core.py tests\test_abr_pipeline_and_gate.py`
  - Result: pass.
- `python tools\abr\audit_abr_pipeline_precheck.py --out-dir .tmp_abr_mock_precheck --overwrite --mock-only`
  - Result: `PASS_MOCK_PRECHECK_ONLY`; `nonzero_window_ok=true`, `val_test_gt_rejection_ok=true`, `detector_forward_count=1`, `dynamic_k_nonconstant=true`, `real_sparse_handoff_ok=true`.
- `python tools\abr\audit_abr_pipeline_precheck.py --out-dir .tmp_abr_real_precheck --overwrite`
  - Result: fail-closed `LOCKED`; local Windows torch import still fails on `c10.dll` with WinError 1114.
- `python tools\abr\validate_abr_launch_gate.py --precheck-json .tmp_abr_mock_precheck\abr_precheck_summary.json`
  - Result: locked, because status is `PASS_MOCK_PRECHECK_ONLY`, not `PASS_PRECHECK_ONLY`.
- `python tools\abr\validate_abr_launch_gate.py --precheck-json .tmp_abr_real_precheck\abr_precheck_summary.json`
  - Result: locked, because status is `LOCKED`.

Still locked:

- No remote sync, Slurm, training, evaluation, `tools/test.py`, stage, commit, or push was run by this owner.
- Linux/N16R4 `PRECHECK_ONLY` may be retried by the main process after commit/push. Full train, mAP claims, deploy claims, paper claims, runtime claims, and sparse-compute claims remain locked.

## Linux/N16R4 PRECHECK_ONLY Pass

**Timestamp**: 2026-06-30 01:02:47 +08:00
**Remote logdir**: `/data/home/sczc063/run/yuzibo/abr_precheck_logs/abr_gpu1_precheck_0bab835_20260630_010247`
**Remote clean clone**: `/data/home/sczc063/run/yuzibo/OpenTAD_ABR_Precheck_20260630_0bab835`
**Commit tested**: `0bab835 Fix ABR clean-clone transform dependency`
**Protected hold**: `1118197 pcot_dbg2g`, node `g0030`, `CUDA_VISIBLE_DEVICES=1`; parent hold was not released or cancelled.

Result:

- Real Linux/N16R4 ABR `PRECHECK_ONLY` passed after the clean-clone `pseudo_boundary.py` dependency fix.
- Test result: `15 passed in 10.62s`.
- Precheck summary: `status=PASS_PRECHECK_ONLY`, `precheck_validated=true`, route label now corrected to `DIVERGENT_INNOVATION_ABR_DO_NOT_MERGE_WITH_C3`.

## 2026-06-30T03:53:23+08:00 - ABR route identity and real scout/probe repair

**Decision**: direct implementation by the ABR single writable code owner, as explicitly assigned by the user for this stage.

**Changes**:
- Corrected ABR route identity to `DIVERGENT_INNOVATION_ABR_DO_NOT_MERGE_WITH_C3` in code, config, tests, validator, and active route documentation.
- Added deploy-visible scout inputs for `abr_scout_curve`, deploy-visible frame signals, and low-cost frame features; formal missing-scout selection now fails closed.
- Kept deterministic fallback only behind explicit `allow_diagnostic_fallback_scout=True` plus `fallback_stage="PRECHECK_ONLY"` / diagnostic tags.
- Added metadata for `abr_scout_source` and `abr_diagnostic_fallback_used` into ABR handoff and collection.
- Extended launch gate validation to accept the ABR config path and verify precheck-only lock semantics.

**Stage status**: local/precheck candidate only. No remote sync, full training, mAP claim, runtime claim, deploy claim, or paper claim is unlocked.

**Local verification**:
- `python -m pytest tests/test_abr_core.py tests/test_abr_pipeline_and_gate.py -q`: `15 passed, 1 skipped in 1.30s`.
- `python tools/abr/validate_abr_launch_gate.py --config configs/adatad/thumos/input_abr_active_bracket_refinement_adapter_irregular_headv3.py`: passed with `allowed_next_action=LOCAL_PRECHECK_ONLY_VALIDATION` and `still_locked=REMOTE_SYNC_FULL_TRAIN_MAPPAPER_CLAIM`.
- `python tools/abr/audit_abr_pipeline_precheck.py --out-dir .tmp_abr_required_precheck --overwrite --mock-only`: `PASS_MOCK_PRECHECK_ONLY`; summary recorded `deploy_visible_scout_or_explicit_fallback_ok=true`, `formal_missing_scout_rejected=true`, and `diagnostic_fallback_source=diagnostic_fallback:PRECHECK_ONLY`.
- `python tools/abr/audit_abr_pipeline_precheck.py --out-dir .tmp_abr_required_real_precheck --overwrite`: locked locally before ABR logic because torch import fails on `c10.dll` with Windows `WinError 1114`.
- Summary fields: `detector_forward_count=1`, `dynamic_k_nonconstant=true`, `nonzero_window_ok=true`, `val_test_gt_rejection_ok=true`, `real_sparse_handoff_ok=true`, `forbidden_inputs_ok=true`, `pipeline_valid_k=24`, `easy_valid_k=13`, `rich_valid_k=24`.
- Launch gate output: `allowed_next_action=REMOTE_PRECHECK_ONLY_REQUEST`, `still_locked=TRAIN_EVAL_SYNC_STAGE_COMMIT_PUSH`.

Interpretation:

- Linux/N16R4 `PRECHECK_ONLY` is now passed for commit `0bab835`.
- This does not unlock full train, evaluation, `tools/test.py`, mAP/runtime/FLOPs/deploy/paper claims, or sparse-compute claims.
- No additional remote action, Slurm action, training, evaluation, `tools/test.py`, stage, commit, or push was performed by this owner in this documentation-only update.

## 2026-06-30 Route-Identity Repair And Remote PRECHECK_ONLY

Recorded 2026-06-30T04:16:12+08:00.

Current commit: `8a03456382dbd73b8239de71aa2a5dbbdad1fce5`.
GitHub branch: `https://github.com/yuzbo/pc-ot-mras-r-series-opentad/tree/codex/divergent-abr-fullcode-20260629`.
Remote clone: `/data/home/sczc063/run/yuzibo/OpenTAD_ABR_Final_20260630_8a03456`.
Remote log dir: `logs/precheck_20260630_abr_8a03456/`.

Changes since the earlier `0bab835` evidence:

- Route identity is now exactly `DIVERGENT_INNOVATION_ABR_DO_NOT_MERGE_WITH_C3`; the previous Boundary-Microscope label was removed from current ABR code/config/tests/docs.
- Formal missing-scout path fails closed.
- Diagnostic fallback remains allowed only for explicit PRECHECK/diagnostic use.
- Sparse handoff keeps sorted unique selected observations, prefix valid masks, raw selected `frame_inds`, and no dense raw backbone handoff claim.

Verification:

- Local tests: `15 passed, 1 skipped`.
- Local launch gate: `allowed_next_action=LOCAL_PRECHECK_ONLY_VALIDATION`, `still_locked=REMOTE_SYNC_FULL_TRAIN_MAPPAPER_CLAIM`.
- Final read-only review: `PASS_SUBAGENT_FINAL_REVIEW_ONLY`.
- N16R4 login-node PRECHECK_ONLY:
  - `python tools/abr/validate_abr_launch_gate.py --config configs/adatad/thumos/input_abr_active_bracket_refinement_adapter_irregular_headv3.py`: passed with route label `DIVERGENT_INNOVATION_ABR_DO_NOT_MERGE_WITH_C3`.
  - `python tools/abr/audit_abr_pipeline_precheck.py --out-dir logs/precheck_20260630_abr_8a03456/audit --overwrite`: `status=PASS_PRECHECK_ONLY`, `precheck_validated=true`, `formal_missing_scout_rejected=true`, `real_sparse_handoff_ok=true`, `val_test_gt_rejection_ok=true`.

Still locked:

- Slurm, `tools/train.py`, `tools/test.py`, evaluation, detector mAP, runtime/FLOPs, deployment readiness, paper claims, and any C3/combo mixing.
