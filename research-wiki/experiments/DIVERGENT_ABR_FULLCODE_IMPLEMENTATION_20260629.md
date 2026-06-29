# DIVERGENT ABR Full-Code Implementation 20260629

**Timestamp**: 2026-06-29 22:19:23 +08:00  
**Implementation worktree**: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_ABR_FullCode_Worktree_20260629`  
**Branch**: `codex/divergent-abr-fullcode-20260629`  
**Base commit**: `588b2729d7570732b4dc281aa4cd3a2dd7e784ac`  
**Route label**: `DIVERGENT_INNOVATION_BOUNDARY_MICROSCOPE_DO_NOT_MERGE_WITH_C3`

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
