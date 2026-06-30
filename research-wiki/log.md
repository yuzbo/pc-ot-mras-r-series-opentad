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

## 2026-06-30T01:02:47+08:00 - ABR Linux/N16R4 PRECHECK_ONLY passed

- Remote clean clone `/data/home/sczc063/run/yuzibo/OpenTAD_ABR_Precheck_20260630_0bab835` tested commit `0bab835 Fix ABR clean-clone transform dependency`.
- Remote logdir: `/data/home/sczc063/run/yuzibo/abr_precheck_logs/abr_gpu1_precheck_0bab835_20260630_010247`.
- Protected hold: `1118197 pcot_dbg2g`, node `g0030`, `CUDA_VISIBLE_DEVICES=1`; parent hold was not released or cancelled.
- Result: `15 passed in 10.62s`.
- Precheck summary: `status=PASS_PRECHECK_ONLY`, `precheck_validated=true`, `route_label=DIVERGENT_INNOVATION_ABR_DO_NOT_MERGE_WITH_C3`. Correction note: an earlier local record used the stale Boundary Microscope label; the route-owned code/config/tests and current formal-gate evidence use the ABR label, and this log entry is corrected to prevent Pro-packet route drift.
- Required fields passed: `detector_forward_count=1`, `dynamic_k_nonconstant=true`, `nonzero_window_ok=true`, `val_test_gt_rejection_ok=true`, `real_sparse_handoff_ok=true`, `forbidden_inputs_ok=true`, `pipeline_valid_k=24`, `easy_valid_k=13`, `rich_valid_k=24`.
- Gate output: `allowed_next_action=REMOTE_PRECHECK_ONLY_REQUEST`, `still_locked=TRAIN_EVAL_SYNC_STAGE_COMMIT_PUSH`.
- Interpretation: Linux/N16R4 `PRECHECK_ONLY` passed after the clean-clone dependency fix, but full train, evaluation, `tools/test.py`, mAP/runtime/FLOPs/deploy/paper claims, and sparse-compute claims remain locked.

## 2026-06-30T18:24:34+08:00 - ABR scout policy repair remains formal-locked

- Worktree `OpenTAD_ABR_ScoutPolicyRepair_Worktree_20260630`, branch `codex/divergent-abr-scout-policy-repair-20260630`, route `DIVERGENT_INNOVATION_ABR_DO_NOT_MERGE_WITH_C3`.
- Fixed first-round audit scoring to use the saved round-0 bracket snapshot and added deploy-visible event-train envelopes, risk-gap micro bridges, silent-gap sentinels, start-edge guards, and max single-bracket width validation.
- Final real 5-video deploy-visible audit improved from the reproduced `first_round_bracket_recall=0.16129032258064516` to `0.7661290322580645`, with `first_round_transition_coverage=0.7258064516129032`, `temporal_coverage_fraction=0.6567708333333333`, `max_bracket_width_fraction=0.296875`, `fallback=false`, and `selector_gt_visible=false`.
- The audit still failed closed with `status=LOCKED`, `missed_transition_count=29`, and `allowed_next_action=LOCKED_REAL_SCOUT_RECALL_BELOW_FORMAL_GATE_REVISE_BRACKET_POLICY_OR_SCOUT`.
- Verification passed: `34 passed` for ABR pytest, py_compile passed, and formal config validator returned `formal_config_ok=true`, `full_train_unlocked=false`.
- No remote sync, Slurm, training, evaluation, `tools/test.py`, or mAP/runtime/deploy/paper claim was performed. ABR formal/full train remains locked; only further local precheck/short diagnostic policy work is justified.

## 2026-06-30T23:13:46+08:00 - ABR recall repair 3 robust local-change brackets

- Worktree `OpenTAD_ABR_RecallRepair3_Worktree_20260630`, branch `codex/divergent-abr-recall-repair3-20260630`, route `DIVERGENT_INNOVATION_ABR_DO_NOT_MERGE_WITH_C3`.
- Added deploy-visible-only robust local-change/extrema round-0 bracket proposals in `opentad/acquisition/abr/policy.py`, using scout-curve smoothing, local contrast, median/MAD salience thresholds, curvature, and extrema prominence; no GT/teacher/detector-cache/dense-backbone input was introduced.
- Updated first-round diagnostics to list `robust_local_change_extrema_brackets`.
- Added TDD coverage for a subthreshold short raw-video graydiff event between sparse scaffold points; the test failed before the fix with `first_round_bracket_recall=0.0` and now passes with full boundary/action coverage under bounded width/density checks.
- Verification passed: focused new test `1 passed`; ABR suite `50 passed, 1 skipped`; formal config validator stayed `full_train_unlocked=false`; launch gate stayed `allowed_next_action=LOCAL_PRECHECK_ONLY_VALIDATION`.
- No remote sync, Slurm, GPU training/evaluation, `tools/test.py`, stage, commit, push, mAP/runtime/sparse-compute/deploy/paper claim, or formal unlock was performed. Real full raw-video diagnostic evidence remains for the main process to harvest.

## 2026-06-30T23:36:21+08:00 - ABR recall repair 3 remote short diagnostic launched

- Main process committed and pushed ABR recall repair 3 as `ccf7db3 DIVERGENT_INNOVATION_ABR_DO_NOT_MERGE_WITH_C3 repair robust scout brackets` on branch `codex/divergent-abr-recall-repair3-20260630`.
- Remote clone synced to `/data/home/sczc063/run/yuzibo/OpenTAD_ABR_RecallRepair3_20260630_ccf7db3` with `data -> /data/run01/sczc063/yuzibo/thumos14` and `pretrained -> /data/home/sczc063/run/yuzibo/pretrained`.
- Final read-only review returned `PASS_SUBAGENT_FINAL_REVIEW_ONLY`; main-process ABR suite stayed `50 passed, 1 skipped`; formal gate stayed `full_train_unlocked=false`.
- Launched CPU/no-GPU real deploy-visible scout recall diagnostic under protected hold `1118197 pcot_dbg2g`, child step `1118197.534 abr_r3diag`, node `g0030`.
- Diagnostic log dir: `/data/home/sczc063/run/yuzibo/OpenTAD_ABR_RecallRepair3_20260630_ccf7db3/logs/abr_repair3_real_scout_recall_max32_gpu0safe_20260630_233530_+0800`.
- Scope is validation `--max-videos 32`, `curve_len=384`, `resize=64`, `CUDA_VISIBLE_DEVICES=EMPTY`; parent hold was not released/cancelled/replaced.
- ABR formal training, `tools/test.py`, mAP/runtime/FLOPs/sparse-compute/deploy/paper claims remain locked.
