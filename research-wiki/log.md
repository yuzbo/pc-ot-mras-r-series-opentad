# Research Log

## 2026-06-29T22:19:23+08:00 - ABR full-code implementation snapshot

- Superseded old minimal ABR worktree `OpenTAD_ABR_Worktree_20260629`; full implementation target is now `OpenTAD_ABR_FullCode_Worktree_20260629`.
- Implemented ABR on branch `codex/divergent-abr-fullcode-20260629` from full OpenTAD base commit `588b2729d7570732b4dc281aa4cd3a2dd7e784ac`.
- Added route-local ABR package, tests, tools, full-code candidate config, and real `LoadFrames`/`Collect` integration.
- Verification: ABR pytest passed with `10 passed, 1 skipped`; py_compile passed; mock precheck passed as `PASS_MOCK_PRECHECK_ONLY`.
- Real precheck remains locked because local torch import fails on `c10.dll` with WinError 1114; launch gate rejected both mock-only and locked real precheck JSON.
- No remote sync, Slurm, training, evaluation, `tools/test.py`, stage, commit, push, or metric claim.

## 2026-06-30T00:16:21+08:00 - ABR Huygens blocker fixes

- Fixed coordinate mismatch blocker by separating window-local selected positions, original/global dense positions, and raw `frame_inds` in the ABR handoff.
- Strengthened validator to require `global == dense_window[local]` and `frame_inds_raw == global`, rejecting missing or mismatched coordinate relationships.
- Real `LoadFrames` now strips `gt_segments`/`gt_labels` before selector invocation; ABR selection remains no-GT even when train annotation handling continues after selection.
- Candidate config now keeps val/test `abr_allow_gt_after_selection=False`; train remains `True` only for post-selection annotation handling.
- Precheck now covers a nonzero dense window and val/test GT rejection, and launch gate requires these checks for `PASS_PRECHECK_ONLY`.
- Verification: `pytest` passed with `13 passed, 1 skipped`; py_compile passed; mock precheck passed as `PASS_MOCK_PRECHECK_ONLY`; real precheck stayed `LOCKED` due local torch DLL failure; both launch-gate checks stayed locked.
- No remote sync, Slurm, training, evaluation, `tools/test.py`, stage, commit, push, or metric claim.

## 2026-06-30T00:25:51+08:00 - ABR final review pass and main-process reverification

- Main-process reverification recorded: `python -m pytest tests/test_abr_core.py tests/test_abr_pipeline_and_gate.py -q` passed with `13 passed, 1 skipped in 1.27s`.
- `python -m py_compile` on ABR files/config/tools passed.
- Mock precheck result: `PASS_MOCK_PRECHECK_ONLY` with `nonzero_window_ok=true`, `val_test_gt_rejection_ok=true`, `detector_forward_count=1`, and `dynamic_k_nonconstant=true`.
- Real precheck stayed `LOCKED` because local Windows torch import fails on `c10.dll` with WinError 1114.
- Launch gate on both mock summary and real summary remained locked as expected.
- `git diff --check` had no whitespace error, only line-ending warnings.
- Final read-only review agent Anscombe / `019f142e-23ec-7651-a976-8b0de80041e1` returned `PASS_SUBAGENT_FINAL_REVIEW_ONLY` with no blockers.
- Only Linux/N16R4 `PRECHECK_ONLY` is allowed next; full train, mAP, deploy, paper, runtime, and sparse-compute claims remain locked.

## 2026-06-30T00:55:52+08:00 - ABR Linux PRECHECK pseudo-boundary import fix

- Remote Linux `PRECHECK_ONLY` attempt on protected hold `1118197` GPU1 failed before ABR selector logic because commit `48ed384` clean clone lacked `opentad/datasets/transforms/pseudo_boundary.py`; logdir: `/data/home/sczc063/run/yuzibo/abr_precheck_logs/abr_gpu1_precheck_retry3_20260630_004513`.
- Added the missing shared transform helper `opentad/datasets/transforms/pseudo_boundary.py` from compatible repository lineage and kept it as a shared transform dependency, not C3/BVR/MDL route logic.
- Added a path-level torch-independent clean-clone dependency test for `load_pseudo_boundary_cache`, `select_pseudo_boundary_hybrid_positions`, and `select_pseudo_boundary_snap_positions`.
- TDD evidence: focused test failed before helper add because the file was missing; after the fix `python -m pytest tests/test_abr_pipeline_and_gate.py -q` passed with `7 passed, 1 skipped in 1.03s`.
- Full verification: `python -m pytest tests/test_abr_core.py tests/test_abr_pipeline_and_gate.py -q` passed with `14 passed, 1 skipped in 1.06s`; py_compile on relevant ABR files/config/tools/tests including `pseudo_boundary.py` passed.
- Mock precheck remained `PASS_MOCK_PRECHECK_ONLY` with `nonzero_window_ok=true`, `val_test_gt_rejection_ok=true`, `detector_forward_count=1`, `dynamic_k_nonconstant=true`, and `real_sparse_handoff_ok=true`.
- Local real precheck remained fail-closed `LOCKED` due Windows torch `c10.dll` WinError 1114; launch gate remained locked for both mock-only and locked-real summaries.
- No remote sync, Slurm, training, evaluation, `tools/test.py`, stage, commit, push, or metric/runtime/deploy/paper claim.
