# Research Log

## 2026-07-01 01:22:40 +08:00 - RBA-RBR local implementation

- Route label: `DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3`.
- Worktree/branch: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_RBA_RBR_Worktree_20260701`, `codex/divergent-rba-rbr-20260701`.
- Implemented first complete local RBA-RBR acquisition route: risk map, soft brackets, refine/rescue probes, train-only regret labels, dynamic budget controller, validators, OpenTAD LoadFrames dispatch, fail-closed config, synthetic ledger builder, launch gate validator, and focused tests.
- Verification passed: py_compile; `python -m pytest tests/test_rba_rbr*.py -q` as expanded by PowerShell -> `10 passed, 1 skipped`; synthetic ledger builder -> `recovery=True`; launch gate -> `gate_pass=true`, `full_train_unlocked=false`; `git diff --check` -> pass with Windows line-ending warning only.
- No remote sync, SSH, Slurm, training, evaluation, Pro review, stage, commit, push, mAP claim, runtime claim, deploy claim, or paper claim was performed.
- Next allowed action: final read-only review, then local/precheck-only decision; full training remains locked.

## 2026-07-01 01:33:16 +08:00 - RBA-RBR selector-facing leakage blocker fix

- Fixed coordinator blocker in `build_rba_rbr_open_tad_selection()`: val/test/deploy no longer validate the full OpenTAD `results` dict, so ordinary downstream `gt_segments` / `gt_labels` payloads are accepted when `train_value_labels=False`.
- Leakage validation now applies to selector-facing metadata/provenance only, while still rejecting teacher/cache/oracle/raw-prediction fields and `selection_uses_gt=True` provenance.
- Added focused tests proving val/test GT payload acceptance without regret labels, selector-facing leakage rejection, and train-only regret-label behavior.
- Verification passed: py_compile; `python -m pytest tests/test_rba_rbr*.py -q` as expanded by PowerShell -> `12 passed, 1 skipped`; synthetic ledger builder -> `recovery=True`; launch gate -> `gate_pass=true`, `full_train_unlocked=false`; `git diff --check` -> pass with Windows line-ending warning only.
- No remote sync, SSH, Slurm, training, evaluation, Pro review, stage, commit, push, mAP claim, runtime claim, deploy claim, or paper claim was performed.

## 2026-07-01 01:50:39 +08:00 - RBA-RBR final-review metadata/backbone blocker fix

- Fixed final-review blocker: RBA config train/val/test `Collect` transforms now explicitly preserve RBA raw/detector/ledger metadata in `metas`.
- Added RBA metadata helper and patched `BackboneWrapper` so backbone irregular time embedding prefers `rba_rbr_raw_selected_positions` / `rba_rbr_raw_selected_valid_len`, while detector/head generic `irregular_selected_positions` remains detector feature centers.
- Updated tests to prove config metadata propagation, raw-axis preference for backbone, and generic detector-center metadata preservation.
- Cleaned `tools/rba_rbr/.tmp_*` generated artifacts and changed test/verify output to pytest/system temp paths.
- Verification passed: py_compile; `python -m pytest tests/test_rba_rbr*.py -q` as expanded by PowerShell -> `14 passed, 1 skipped`; synthetic ledger builder in system temp -> `recovery=True`; launch gate -> `gate_pass=true`, `full_train_unlocked=false`; `git diff --check` -> pass with Windows line-ending warnings only; `git status --untracked-files=all` -> no `tools/rba_rbr/.tmp_*` artifacts.
- No remote sync, SSH, Slurm, training, evaluation, `tools/test.py`, Pro review, stage, commit, push, mAP claim, runtime claim, deploy claim, or paper claim was performed.

## 2026-07-01 02:02:55 +08:00 - RBA-RBR third-round claim-lock/hygiene blocker fix

- Fixed third-round final-review blocker: config now explicitly sets `no_paper_claim=True`, and launch gate requires both `no_deploy_claim=True` and `no_paper_claim=True`.
- Added deploy/paper claim locks to RBA-RBR deploy ledgers, synthetic summary, ledger validator, launch-gate output, and integration tests; `claim_status` is now `rba_rbr_local_precheck_only_no_metric_runtime_deploy_or_paper_claim`.
- Verification passed: py_compile; `python -m pytest tests/test_rba_rbr*.py -q` as expanded by PowerShell -> `14 passed, 1 skipped`; synthetic ledger builder in system temp -> `recovery=True`; launch gate -> `gate_pass=true`, `full_train_unlocked=false`, `deploy_claim_unlocked=false`, `paper_claim_unlocked=false`; `git diff --check` -> pass with Windows line-ending warnings only; owned-worktree `__pycache__` cleanup deleted 7 generated dirs and final count is `PYCACHE_DIR_COUNT=0`; `git status --untracked-files=all` -> no `.tmp_*` or `__pycache__` artifacts.
- No remote sync, SSH, Slurm, training, evaluation, `tools/test.py`, Pro review, stage, commit, push, mAP claim, runtime claim, deploy claim, or paper claim was performed.

## 2026-07-01 02:27:37 +08:00 - RBA-RBR remote sync and non-GPU precheck

- Synced `DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3` from GitHub branch `codex/divergent-rba-rbr-20260701` at commit `4072d43` into N16R4 route-owned precheck worktree `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_Precheck_20260701_4072d43`.
- Used existing remote clean clone `/data/run01/sczc063/yuzibo/OpenTAD_Back_clean_20260629_588b272` as the Git object source to avoid a full new clone after the first clone attempt hit `Disk quota exceeded`.
- Remote precheck passed without GPU or Slurm: `python -m pytest tests/test_rba_rbr_core.py tests/test_rba_rbr_integration.py -q` -> `15 passed in 19.55s`; py_compile -> pass; RBA-RBR launch gate -> `gate_pass=true`, `full_train_unlocked=false`, `deploy_claim_unlocked=false`, `paper_claim_unlocked=false`, `remote_sync_unlocked_by_local_gate=false`, `sparse_compute_claim=false`.
- Remote audit summary path: `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_Precheck_20260701_4072d43/logs/rba_rbr_precheck_4072d43/rba_rbr_audit/summary.json`; gate result path: `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_Precheck_20260701_4072d43/logs/rba_rbr_precheck_4072d43/gate_result.json`.
- Resource boundary: BVR child `1118197.519` was still running on GPU0 and C3 child `1118197.528` was still running on GPU1; RBA-RBR did not occupy either GPU and did not modify/release/cancel protected parent hold `1118197 pcot_dbg2g`.
- Current next action: RBA-RBR may proceed to Pro/GitHub review or later short diagnostic scheduling when GPU0 is free; full training and all mAP/runtime/deploy/paper claims remain locked.

## 2026-07-01 02:40:43 +08:00 - RBA-RBR Pro review transport incomplete

- Created Pro review prompt `research-wiki/experiments/DIVERGENT_RBA_RBR_PRO_REVIEW_PROMPT_20260701.md` for GitHub branch `codex/divergent-rba-rbr-20260701` at commit `87eae60`.
- Rosetta/Oracle transport attempts are recorded in `research-wiki/experiments/DIVERGENT_RBA_RBR_PRO_REVIEW_TRANSPORT_20260701.md`.
- No valid GPT-5.5 Pro answer was harvested: Rosetta inline failed with stuck send pipeline; Rosetta attachment failed with focus timeout; Oracle attach/copy-profile/persistent/cookie attempts all failed before submission or model selection.
- Pro state is `INCOMPLETE_PRO_DECISION`: no GitHub inspection, no code-grounded Pro blocker, and no Pro approval exists.
- This is transport failure only, not a technical rejection of RBA-RBR. Remote sync and non-GPU precheck remain valid; full training and all mAP/runtime/deploy/paper claims remain locked.

## 2026-07-01 03:05:00 +08:00 - RBA-RBR GitHub API sync and shortdiag gate fix

- Advanced GitHub branch `codex/divergent-rba-rbr-20260701` by API because ordinary HTTPS `git push` kept failing locally; remote branch moved from `87eae60369bacf4f58cbe870170a6eecd1fa57c3` to `52ce1ef71142dc5c08c91755451d10deb0b24494` with 25 synced Pro transport evidence files.
- Fixed RBA-RBR inherited evaluation path risk after BVR failed at epoch-41 evaluation on legacy `/root/autodl-tmp/annotations/thumos_14_anno.json`: RBA config now explicitly sets `evaluation.ground_truth_filename=annotation_path`, and launch gate rejects legacy evaluation paths.
- Added `input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3_shortdiag.py` as a two-epoch, no-eval, no-checkpoint, claim-locked short diagnostic config.
- Verification passed: `python -m pytest tests/test_rba_rbr_core.py tests/test_rba_rbr_integration.py -q` -> `15 passed, 1 skipped`; PowerShell-expanded py_compile -> pass; RBA launch gate -> `gate_pass=true`, `full_train_unlocked=false`, `deploy_claim_unlocked=false`, `paper_claim_unlocked=false`.
- Resource state: protected hold `1118197` unchanged; BVR child disappeared after evaluator failure and MDL child `1118197.535 mdl_formal_g0` started on GPU0; C3 child `1118197.528` remains on GPU1. RBA-RBR short diagnostic is staged but not running.

## 2026-07-01 03:08:44 +08:00 - RBA-RBR short diagnostic staged behind MDL

- Created N16R4 route-owned shortdiag worktree `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_Shortdiag_20260701_9311f49` from GitHub branch `codex/divergent-rba-rbr-20260701` at commit `9311f490f19e86f2ec3f7cc8a9c3233b4c1d4ee6`.
- Remote preflight passed: `python -m pytest tests/test_rba_rbr_core.py tests/test_rba_rbr_integration.py -q` -> `16 passed`; RBA launch gate -> `gate_pass=true`; shortdiag config parse -> `end_epoch=2`, `val_eval_interval=-1`, `disable_checkpoint=true`, local N16R4 annotation path.
- Started watcher `/data/run01/sczc063/yuzibo/route_watchers/rba_rbr_after_mdl_1118197_535_20260701_0310_+0800`, PID `2477396`, waiting for MDL child step `1118197.535` to finish before launching `rba_rbr_short_g0` on GPU0 with `CUDA_VISIBLE_DEVICES=0`.
- RBA-RBR is queued/staged only, not running yet. Protected parent hold `1118197 pcot_dbg2g` was not released/cancelled/replaced. Formal full training and all metric/runtime/deploy/paper claims remain locked.

## 2026-07-01 03:47:32 +08:00 - RBA-RBR deploy-visible raw scout repair

- RBA-RBR short diagnostic child `1118197.537` failed before training because the formal path required preview metadata while the real THUMOS pipeline only had `video_reader` after `DecordInit`.
- Added a formal deploy-visible `raw_rgb_lowres_scout` path that samples a small number of raw frames from `video_reader` and derives actionness, uncertainty, and transition curves without GT, teacher, cache, detector predictions, or diagnostic fallback.
- Updated LoadFrames/config/launch gate/tests for `rba_rbr_scout_sample_count=32`; formal path still fails closed if both explicit preview metadata and `video_reader` are absent.
- Verification passed: `python -m pytest tests/test_rba_rbr_core.py tests/test_rba_rbr_integration.py -q` -> `17 passed, 2 skipped`; RBA launch gate -> `gate_pass=true`, `full_train_unlocked=false`, no metric/runtime/deploy/paper claim unlocked.
- Next action: read-only final review, GitHub/remote sync, then relaunch RBA-RBR `SHORT_DIAGNOSTIC_ONLY` on GPU0 if memory is safe. Protected parent hold `1118197` remains untouched.

## 2026-07-01 04:07:58 +08:00 - RBA-RBR short diagnostic relaunched with raw scout

- Read-only final review returned `PASS_SUBAGENT_FINAL_REVIEW_ONLY`; no blocker found for the deploy-visible raw scout repair.
- Synced repaired files to N16R4 route-owned worktree `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_Shortdiag_20260701_9311f49` and created remote commit `c0574d58015dbe76f99a75644acfee654b45ff74`.
- Remote verification passed: `python -m pytest tests/test_rba_rbr_core.py tests/test_rba_rbr_integration.py -q` -> `19 passed`; RBA launch gate -> `gate_pass=true`, `full_train_unlocked=false`.
- Relaunched RBA-RBR `SHORT_DIAGNOSTIC_ONLY` on protected hold `1118197` GPU0 as child `1118197.538`, logdir `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_Shortdiag_20260701_9311f49/logs/rba_rbr_shortdiag_rawscout_c0574d5_gpu0_20260701_040627_+0800`.
- Startup sanity passed: no preview/Traceback/OOM/NaN/non-finite pattern; first observed training line `[000][00020/00199] Loss=2.0044 cls_loss=0.2843 reg_loss=0.2525 boundary_loss=1.4675 mem=9344MB`.
- Parent hold `1118197 pcot_dbg2g` remains protected and untouched. This is short diagnostic only; formal full training and all metric/runtime/deploy/paper claims remain locked.

## 2026-07-01 04:20:30 +08:00 - RBA-RBR raw scout short diagnostic completed

- RBA-RBR child `1118197.538` completed successfully: `COMPLETED|0:0`, elapsed `00:11:40`.
- Log path: `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_Shortdiag_20260701_9311f49/logs/rba_rbr_shortdiag_rawscout_c0574d5_gpu0_20260701_040627_+0800/srun-1118197.out`.
- Bad-pattern count remained `0`; 24 finite loss lines were observed; final line `[001][00199/00199] Loss=1.9425 cls_loss=0.6359 reg_loss=0.6360 boundary_loss=0.6706 mem=9344MB`; log contains `Training Over...`.
- This confirms the deploy-visible raw scout repair fixed the earlier `.537` launch blocker. It does not provide mAP/runtime/FLOPs/deploy/paper evidence, and formal full training remains locked.

## 2026-07-01 04:29:33 +08:00 - RBA-RBR bounded eval diagnostic config

- Added `configs/adatad/thumos/input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3_evaldiag.py` for a bounded four-epoch RBA-RBR diagnostic with validation at epochs 2 and 4.
- The config stays under `DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3`, keeps `full_train_unlocked=False`, and keeps metric/runtime/deploy/paper claim locks true. Any mAP emitted by this diagnostic is detector-health evidence only, not a final route result.
- Updated `tests/test_rba_rbr_integration.py` to verify evaldiag schedule, checkpoint behavior, N16R4 annotation path, and claim locks.
- Verification passed locally: `python -m pytest tests/test_rba_rbr_core.py tests/test_rba_rbr_integration.py -q` -> `18 passed, 2 skipped`; RBA formal launch gate remains `gate_pass=true`, `full_train_unlocked=false`.
- Subagent tooling limitation recorded for this continuation: only spawn/close interfaces were available, with no usable wait/harvest result path, so the main process proceeded with self-check for this bounded config-only diagnostic. Formal full training remains locked.

## 2026-07-01 04:37:45 +08:00 - RBA-RBR bounded eval diagnostic launched on GPU0

- Synced the eval diagnostic config/test/docs to N16R4 route-owned RBA worktree `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_Shortdiag_20260701_9311f49` and committed remote `564a6f3`.
- Remote verification passed: `python -m pytest tests/test_rba_rbr_core.py tests/test_rba_rbr_integration.py -q` -> `20 passed`; formal RBA launch gate remains `gate_pass=true`, `full_train_unlocked=false`; evaldiag resolves to `end_epoch=4`, `val_start_epoch=1`, `val_eval_interval=2`, `disable_checkpoint=False`, and the N16R4 annotation path.
- Launched child `1118197.539 rba_rbr_eval_g0` on protected hold `1118197` GPU0 with `CUDA_VISIBLE_DEVICES=0`.
- Logdir: `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_Shortdiag_20260701_9311f49/logs/rba_rbr_evaldiag_564a6f3_gpu0_20260701_043542_+0800`.
- Startup sanity passed: child state `RUNNING|0:0`, no bad pattern for Traceback/OOM/NaN/non-finite/ValueError, and first loss lines reached `[000][00020/00199] Loss=2.0044 ... mem=9344MB` and `[000][00040/00199] Loss=2.4596 ... mem=9344MB`.
- This remains diagnostic-only detector-health evidence, not a formal full train or final mAP/runtime/deploy/paper claim. Parent hold remains protected and untouched.
