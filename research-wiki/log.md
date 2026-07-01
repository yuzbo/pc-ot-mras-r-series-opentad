# Research Log

## 2026-07-01 14:47:42 +08:00 - RBA-RBR guard diagnostic completed severe-low

- Route label: `DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3`.
- Protected-hold child `1118197.560 rba_guard_g0` completed its corrected coverage-guard diagnostic in `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_GuardDiag_20260701_e6de60e9_bundle/logs/rba_rbr_guard_evaldiag_manual_parallel_holdg0_20260701_123213_+0800`.
- Metric evidence: first observed Average-mAP line `0.22%` at `2026-07-01 13:39:19 +08:00`; final observed Average-mAP line `7.45%` at `2026-07-01 14:47:42 +08:00`; final `mAP@0.3/0.4/0.5/0.6/0.7 = 17.34/11.47/5.69/2.09/0.65`; evaluator saw `411700` predictions and `3325` GT instances.
- Health evidence: `Training Over` count `1`, hard error count `0`.
- Guard audit evidence: `4090` rows, all `PASS_RBA_RBR_NATIVE_AXIS_POSITIONS_ENTERED_MODEL`; `raw_valid_k` avg `84.6235`; `mask_true_count` avg `42.5601`; `selected_max_gap_after_guard` max `16`, avg `15.5897`; `max_detector_gap_after_guard` max `24`, avg `22.5460`.
- Latest GitHub evidence branch for Pro diagnosis: `https://github.com/yuzbo/pc-ot-mras-r-series-opentad/tree/codex/divergent-rba-rbr-guard-complete-39073521-20260701`.
- Decision: final diagnostic remains severe-low. Formal/full RBA-RBR training, all metric/runtime/FLOPs/sparse-compute/deploy/paper claims, and any route conclusion remain locked until a valid GPT-5.5 Pro severe-result diagnosis is available or the user gives an explicit same-scope override.

## 2026-07-01 13:39:19 +08:00 - RBA-RBR guard diagnostic first validation severe-low

- Route label: `DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3`.
- Protected-hold child `1118197.560 rba_guard_g0` reached first validation in `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_GuardDiag_20260701_e6de60e9_bundle/logs/rba_rbr_guard_evaldiag_manual_parallel_holdg0_20260701_123213_+0800`.
- Metric evidence: `Average-mAP=0.22%`, `mAP@0.3/0.4/0.5/0.6/0.7 = 0.66/0.30/0.09/0.04/0.01`, `3325` GT instances, `411700` predictions.
- Health evidence: no Traceback/OOM/NaN/non-finite pattern was observed; the child continued into epoch 2 with finite loss.
- Guard audit evidence: `2183` rows, all RBA label/status PASS; detector feature count is restored (`mask_true_count` avg `42.53`, p50 `43`), raw gap max is `16`, detector gap max is `24`, and only one row was below the detector feature floor.
- Interpretation: the coverage guard is functioning, but it did not rescue first-validation detector health. The severe-result gate remains active; formal/full RBA-RBR training and all mAP/runtime/FLOPs/sparse-compute/deploy/paper claims stay locked.
- Pro transport state: Rosetta CDP ports `9223/9333/9222` refused connection and Oracle Pro provider was not ready because `OPENAI_API_KEY` is missing. This is not a completed Pro review.
- GitHub evidence branch for later Pro diagnosis: `codex/divergent-rba-rbr-guard-severe-firsteval-15303a3c-20260701`, URL `https://github.com/yuzbo/pc-ot-mras-r-series-opentad/tree/codex/divergent-rba-rbr-guard-severe-firsteval-15303a3c-20260701`.
- Resource boundary: parent hold `1118197 pcot_dbg2g` was not released, cancelled, replaced, or modified; GPU1/C3 was not touched.

## 2026-07-01 12:34:06 +08:00 - RBA-RBR guard diagnostic launched in parallel on protected GPU0

- Route label: `DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3`.
- BVR child `1118197.542 bvr_twb_fix2_g0` remained `RUNNING` on protected hold GPU0, but its current log showed normal finite training and only about `1454MB` model memory; a hold-overlap GPU query showed `CUDA_VISIBLE_DEVICES=0` with about `2441/24564 MiB` used before RBA launch.
- To avoid waiting on Slurm priority while GPU0 had enough memory, cancelled only the RBA public pending job after rechecking the exact record `1132718|rba_guarddiag|PENDING`, then stopped only the route watcher PID `1840205`.
- Started RBA-RBR corrected guard `SHORT_DIAGNOSTIC_ONLY` as protected-hold child `1118197.560 rba_guard_g0` with logdir `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_GuardDiag_20260701_e6de60e9_bundle/logs/rba_rbr_guard_evaldiag_manual_parallel_holdg0_20260701_123213_+0800`.
- Startup evidence: launcher printed `CUDA_VISIBLE_DEVICES_INITIAL=0`, `SLURM_STEP_GPUS=2`, `HEAD=e6de60e9`; torch distributed init succeeded; training reached `[000][00040/00199] Loss=2.5124 ... mem=9344MB`; `rba_rbr_grid_audit.jsonl` reached `46` rows during the startup window.
- No Traceback/OOM/NaN/non-finite pattern was observed in the startup window. Parent hold `1118197 pcot_dbg2g` was not released, cancelled, replaced, or modified; GPU1/C3 work was not touched.
- This is still `SHORT_DIAGNOSTIC_ONLY`; formal/full RBA-RBR training and all metric/runtime/FLOPs/sparse-compute/deploy/paper claims remain locked until guard audit and metric evidence are reviewed.

## 2026-07-01 12:20:35 +08:00 - RBA-RBR guard watcher safety hardening applied

- Route label: `DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3`.
- Accepted read-only review result: `PASS_SUBAGENT_FINAL_REVIEW_ONLY_FOR_RBA_RBR_GUARD_WATCHER_SHORT_DIAGNOSTIC_ONLY`; blockers: none.
- Applied non-blocking safety hardening in the route-owned watcher/hold launcher scripts: cancel public job only after rechecking `jobid|name|state` equals `1132718|rba_guarddiag|PENDING/CONFIGURING`; BVR wait uses a second check after 10 seconds before treating `1118197.542` as ended; hold launcher prints `SLURM_STEP_GPUS`/`SLURM_JOB_GPUS` and fails closed unless `CUDA_VISIBLE_DEVICES=0`.
- Uploaded hardened scripts to `/data/run01/sczc063/yuzibo/route_watchers/rba_rbr_guard_after_bvr_1118197_542_public1132718_20260701/`; remote `bash -n` passed for both.
- Restarted watcher: old PID `1670139` stopped, hardened watcher PID `1840205` is alive and logged `waiting_for_bvr step=1118197.542 public_state=PENDING`.
- Current remote state: public corrected job `1132718` remains `PENDING`, reason `Priority`; BVR child `1118197.542` remains `RUNNING` on protected hold GPU0. Parent hold `1118197 pcot_dbg2g` was not released, cancelled, replaced, or modified.

## 2026-07-01 12:12:00 +08:00 - RBA-RBR corrected guard fallback watcher staged

- Route label: `DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3`.
- Corrected public guard job `1132718 rba_guarddiag` remained `PENDING`, reason `Priority`, with no node assigned.
- BVR child `1118197.542 bvr_twb_fix2_g0` remained `RUNNING` on protected hold GPU0, so RBA-RBR did not launch a hold child immediately.
- Added route-owned watcher scripts: `scripts/watch_rba_rbr_guard_after_bvr_n16r4.sh` and `scripts/launch_rba_rbr_guarddiag_hold_g0_n16r4.sh`.
- Uploaded both scripts to `/data/run01/sczc063/yuzibo/route_watchers/rba_rbr_guard_after_bvr_1118197_542_public1132718_20260701/`; remote `bash -n` passed for both.
- Started watcher PID `1670139`. First log lines: watcher started for parent `1118197`, wait step `1118197.542`, public job `1132718`, then `waiting_for_bvr ... public_state=PENDING`.
- Safety behavior: if `1132718` starts or completes first, watcher exits; if BVR ends while `1132718` is still pending/configuring, watcher cancels only that pending public RBA job to avoid duplicate GPU use and launches the same corrected guard diagnostic on protected hold GPU0. Terminal public failure stops for manual review.
- Resource boundary: parent hold `1118197 pcot_dbg2g` was not released, cancelled, replaced, or modified. Formal/full RBA-RBR training and all claims remain locked.

## 2026-07-01 11:59:27 +08:00 - RBA-RBR guard launcher torchrun fix and relaunch

- Route label: `DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3`.
- Public guard job `1132641 rba_guarddiag` started on `g0024` but failed after `00:00:46` with `KeyError: 'LOCAL_RANK'` because the sbatch launcher called `python tools/train.py` directly. No train iteration, grid audit row, mAP, runtime, FLOPs, deploy, sparse-compute, or paper evidence was produced by that failed launch.
- Added route-owned launcher `scripts/run_rba_rbr_guard_evaldiag_n16r4.sbatch`, using single-GPU `python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1` while preserving the guard evaldiag config and `RBA_RBR_GRID_AUDIT_PATH` summary.
- GitHub evidence branch pushed without force: `codex/divergent-rba-rbr-guard-launcher-843b4948-20260701`; URL `https://github.com/yuzbo/pc-ot-mras-r-series-opentad/tree/codex/divergent-rba-rbr-guard-launcher-843b4948-20260701`.
- Uploaded the launcher to `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_GuardDiag_20260701_e6de60e9_bundle/scripts/run_rba_rbr_guard_evaldiag_n16r4.sbatch`; remote `bash -n` passed.
- Relaunched the coverage-guard `SHORT_DIAGNOSTIC_ONLY` job as `1132718 rba_guarddiag`, logdir `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_GuardDiag_20260701_e6de60e9_bundle/logs/rba_rbr_guard_evaldiag_torchrun_20260701_1200_0800/`. Initial state: `PENDING`, reason `Priority`, no node assigned.
- Resource boundary: BVR child `1118197.542 bvr_twb_fix2_g0` remained `RUNNING` on protected hold GPU0, so no RBA child was launched on the hold. Parent hold `1118197 pcot_dbg2g` was not released, cancelled, replaced, or modified. Formal/full RBA-RBR training and all claims remain locked.

## 2026-07-01 10:58:56 +08:00 - RBA-RBR guard GPU0 fallback watcher staged

- Route label: `DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3`.
- Public Slurm guard job `1132641 rba_guarddiag` remained `PENDING`, reason `Priority`, with no node assigned and no train/audit log.
- BVR child `1118197.542 bvr_twb_fix2_g0` was still running on protected hold GPU0, so no RBA child was launched immediately.
- Staged a conservative fallback watcher at `/data/run01/sczc063/yuzibo/route_watchers/rba_rbr_guard_after_bvr_1118197_542_20260701_1055_+0800`, PID `509961`.
- Watcher behavior: wait for BVR step `1118197.542` to disappear; if public guard job `1132641` is already running/completed, exit without launching; if it is still pending, cancel only that pending public RBA job to avoid duplicate consumption and launch the same guard `SHORT_DIAGNOSTIC_ONLY` on protected hold GPU0 as child `rba_guard_g0`.
- Resource boundary: the protected parent hold `1118197 pcot_dbg2g` was not modified, released, cancelled, or replaced. C3 GPU1 remains out of scope. Formal/full RBA-RBR training and all metric/runtime/FLOPs/sparse-compute/deploy/paper claims remain locked.

## 2026-07-01 10:38:15 +08:00 - RBA-RBR guard/grid-audit monitor update

- Route label: `DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3`.
- Read-only Slurm/log scan confirmed `1132462 rba_grid_audit` remained `RUNNING|0:0` on `g0053`; this is the old pre-guard grid-audit diagnostic.
- Old pre-guard evidence: finite losses continued through epoch 3, first validation remained severe-low at `Average-mAP=0.16%`, and `rba_rbr_grid_audit.jsonl` reached `3962` rows. This is diagnostic evidence for the pre-guard collapse, not evidence against the new coverage guard.
- New guard diagnostic `1132641 rba_guarddiag` remained `PENDING|0:0`, reason `Priority`, with no node assigned. Its logdir still had only the sbatch script and no train/audit output yet.
- Resource boundary: both RBA jobs use `--exclude=g0030`; protected parent hold `1118197 pcot_dbg2g` was not modified, released, cancelled, or reused.
- Decision: formal/full RBA-RBR training and all mAP/runtime/FLOPs/sparse-compute/deploy/paper claims remain locked. Next action is to let the old audit finish and wait for the guard short diagnostic to start, then compare raw/detector count and gap evidence.

## 2026-07-01 10:47:49 +08:00 - RBA-RBR old pre-guard grid-audit completed severe-low

- Route label: `DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3`.
- Slurm job `1132462 rba_grid_audit` completed normally on non-protected node `g0053`: `COMPLETED|0:0`, elapsed `01:50:24`.
- Final diagnostic metric: `Average-mAP=5.10%`, with `mAP@0.3/0.4/0.5/0.6/0.7 = 12.66/7.33/3.56/1.48/0.47`. This remains severe-low diagnostic evidence, not a route success claim.
- Grid audit result: `4090` rows; every row had `PASS_RBA_RBR_NATIVE_AXIS_POSITIONS_ENTERED_MODEL`, `native_axis=True`, and `dispatch_hit=True`.
- Detector coverage diagnosis: `mask_true_count` mean `27.65`, min `9`, p50 `28`, p95 `36`; `3102/4090` rows were below the new coverage-guard floor of `32`.
- Attribution boundary: this is the old pre-guard branch result. New guard job `1132641 rba_guarddiag` remains `PENDING`, reason `Priority`, no node/log yet.
- Decision: old audit supports the coverage-guard fix hypothesis: the detector received native-axis metadata, but the effective detector-side budget/spacing was often too sparse. Formal/full RBA-RBR training and all claims remain locked until the guard diagnostic verifies count/gap restoration.

## 2026-07-01 10:29:01 +08:00 - RBA-RBR guard diagnostic queued

- Route label: `DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3`.
- Local guard fix was committed as `e6de60e9 DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3 coverage guard`.
- GitHub evidence branch pushed without force: `codex/divergent-rba-rbr-coverage-guard-e6de60e9-20260701`.
- GitHub URL: `https://github.com/yuzbo/pc-ot-mras-r-series-opentad/tree/codex/divergent-rba-rbr-coverage-guard-e6de60e9-20260701`.
- Final read-only review agent `019f1b74-27c5-72c2-8a96-fd862dcd7ac6` returned `PASS_SUBAGENT_FINAL_REVIEW_ONLY_FOR_PRECHECK_AND_SHORT_DIAGNOSTIC_ONLY`; no blocking findings. Non-blocking gate hardening was accepted immediately.
- Accepted hardening: `tools/rba_rbr/validate_rba_rbr_launch_gate.py` now requires `rba_rbr_min_detector_feature_keep=32`, `rba_rbr_max_raw_gap=16`, and `rba_rbr_max_detector_gap=24` in the formal RBA config.
- Local verification after hardening: `python -m pytest tests/test_rba_rbr_core.py tests/test_rba_rbr_integration.py -q` -> `21 passed, 5 skipped in 6.02s`; py_compile for RBA implementation and launch gate -> pass; RBA launch gate -> `gate_pass=true`, `full_train_unlocked=false`, `deploy_claim_unlocked=false`, `paper_claim_unlocked=false`; `git diff --check` -> pass with LF/CRLF warnings only.
- N16R4 GitHub clone hit a transient proxy error (`Empty reply from server`), so a local git bundle was uploaded instead. New route-owned remote tree: `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_GuardDiag_20260701_e6de60e9_bundle`, HEAD `e6de60e9`, with `data -> ../OpenTAD_Back_check/data` and `pretrained -> ../pretrained`.
- Remote PRECHECK_ONLY passed: `python -m pytest tests/test_rba_rbr_core.py tests/test_rba_rbr_integration.py -q` -> `26 passed in 89.21s`; py_compile -> pass; launch gate -> `gate_pass=true`, `full_train_unlocked=false`.
- Guard `SHORT_DIAGNOSTIC_ONLY` Slurm job submitted as `1132641 rba_guarddiag` with `--exclude=g0030`; initial status was `PENDING`, no node assigned. Logdir: `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_GuardDiag_20260701_e6de60e9_bundle/logs/rba_rbr_guard_evaldiag_e6de60e9_20260701/`.
- Protected hold `1118197 pcot_dbg2g` was not modified, released, cancelled, or reused. Formal/full training and all mAP/runtime/FLOPs/sparse-compute/deploy/paper claims remain locked.

## 2026-07-01 10:07:14 +08:00 - RBA-RBR recoverable coverage guard local fix

- Route label: `DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3`.
- Trigger evidence: active remote grid-audit diagnostic job `1132462 rba_grid_audit` produced severe-low first validation (`Average-mAP=0.16%`) while grid audit rows showed native-axis RBA dispatch was active and `mask_true_count` could be as low as `9` despite config `rba_rbr_min_keep=64`, `rba_rbr_max_keep=192`, `rba_rbr_feature_stride=2`.
- Local worktree/branch: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_RBA_RBR_Worktree_20260701`, `codex/divergent-rba-rbr-20260701`.
- Changed surface: RBA-RBR budget controller/types/OpenTAD bridge, RBA `LoadFrames` parameter wiring, RBA grid-audit fields in `IrregularActionFormer`, main RBA config, and focused RBA tests.
- Behavior: regret/scaffold/refine/rescue ranking remains primary; after regret selection, a deploy-visible non-GT/non-teacher coverage guard backfills by recoverability/regret/staleness/rescue risk to satisfy configurable raw min, detector-feature min, raw max-gap, and detector max-gap where possible within `max_k`. Gap closure uses the largest uncovered interval center band before risk ranking so it cannot keep piling points on one risk peak.
- Config guard values now inherited by train/val/test/evaldiag: `rba_rbr_min_keep=64`, `rba_rbr_min_detector_feature_keep=32`, `rba_rbr_max_raw_gap=16`, `rba_rbr_max_detector_gap=24`, `rba_rbr_feature_stride=2`, `rba_rbr_max_keep=192`.
- Ledger/audit: records pre/post guard raw valid count, detector target/valid count, max raw/detector gaps before/after, guard additions, `pre_guard_budget_stop_reason`, `guard_reason`, adapter padding duplicate count, detector mask length, and detector mask true count.
- Verification passed locally: `python -m pytest tests/test_rba_rbr_core.py tests/test_rba_rbr_integration.py -q` -> `21 passed, 5 skipped in 6.07s`; `python -m py_compile opentad/acquisition/rba_rbr/types.py opentad/acquisition/rba_rbr/budget_controller.py opentad/acquisition/rba_rbr/open_tad_bridge.py opentad/datasets/transforms/end_to_end.py opentad/models/detectors/irregular_actionformer.py` -> pass; `git diff --check` -> pass with Windows LF/CRLF warnings only.
- No remote sync, no Slurm, no `tools/test.py`, no Pro, no staging/commit/push performed. Formal/full training and all metric/runtime/FLOPs/deploy/paper claims remain locked pending review and severe-result diagnosis.

## 2026-07-01 09:31:39 +08:00 - RBA-RBR grid-audit live sync/status

- Existing branch push to `codex/divergent-rba-rbr-20260701` was rejected as non-fast-forward; no force push or overwrite was attempted.
- New evidence branch pushed successfully and then fast-forwarded with the live-sync records: `codex/divergent-rba-rbr-status-efbb4eea-20260701`.
- GitHub URL: `https://github.com/yuzbo/pc-ot-mras-r-series-opentad/tree/codex/divergent-rba-rbr-status-efbb4eea-20260701`.
- Active job: `1132462 rba_grid_audit`, still `RUNNING` on `g0053` with `--exclude=g0030`; protected hold `1118197 pcot_dbg2g` was not modified.
- Audit file reached `1387` rows; latest inspected row retained `DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3`, `PASS_RBA_RBR_NATIVE_AXIS_POSITIONS_ENTERED_MODEL`, `native_axis=True`, and matched mask/position counts.
- Training portion completed two bounded diagnostic epochs with finite losses and entered validation/audit; progress tail was around `1010/1645` windows.
- This remains `SHORT_DIAGNOSTIC_ONLY`; no mAP/runtime/FLOPs/deploy/paper/sparse-compute claim or full-train unlock.

## 2026-07-01 08:53:47 +08:00 - RBA-RBR grid-audit diagnostic launched off protected hold

- Route label: `DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3`.
- Synced local grid-audit route files to the N16R4 route-owned worktree `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_Shortdiag_20260701_9311f49` and committed remote `a64530e`.
- Remote verification passed before launch: py_compile passed; `python -m pytest tests/test_rba_rbr_core.py tests/test_rba_rbr_integration.py -q` -> `23 passed in 48.59s`; launch gate returned `gate_pass=true`, `full_train_unlocked=false`, `deploy_claim_unlocked=false`, `paper_claim_unlocked=false`, `sparse_compute_claim=false`.
- Read-only final review agent returned `PASS_SUBAGENT_FINAL_REVIEW_ONLY_FOR_SHORT_DIAGNOSTIC_ONLY` with no blockers.
- A first new Slurm job `1132461` started on `g0030` and was cancelled after `00:01:23` because it risked overlapping the protected hold node. This was not the protected parent hold and did not release/cancel/replace `1118197`.
- Relaunched as Slurm job `1132462 rba_grid_audit` with `--exclude=g0030`; it is running on `g0053` with 1 GPU and 4 CPU. Logdir: `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_Shortdiag_20260701_9311f49/logs/rba_rbr_grid_audit_evaldiag_sbatch_3cab4121_20260701_0853_exclude_g0030_+0800/`.
- This is `SHORT_DIAGNOSTIC_ONLY` / sparse-forward grid-audit evidence. It does not unlock formal/full long training, official mAP, runtime/FLOPs, deploy, paper, or sparse-compute claims.

## 2026-07-01 08:58:37 +08:00 - RBA-RBR grid-audit startup evidence

- Slurm job `1132462 rba_grid_audit` is running on `g0053`, not on the protected hold node `g0030`.
- Startup log reached epoch 0 iteration 140 with finite loss, e.g. `[000][00140/00199] Loss=2.5149`.
- `rba_rbr_grid_audit.jsonl` exists and had at least 169 sampled rows at inspection time.
- Audit sample: all sampled rows have route label `DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3`; first row status is `PASS_RBA_RBR_NATIVE_AXIS_POSITIONS_ENTERED_MODEL`; `native_axis=True`; `mask_true_count=23` matches `meta_detector_feature_position_count=23`.
- Interpretation: the detector-side native-axis grid audit is active and RBA detector feature positions are entering `IrregularActionFormer`. This is still diagnostic-only evidence, not metric/runtime/deploy/paper evidence.

## 2026-07-01 07:48:48 +08:00 - RBA-RBR grid-audit remote PRECHECK_ONLY passed

- Route label: `DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3`.
- Created route-owned remote precheck copy `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_GridAuditPrecheck_20260701_600fc8f2` from the existing RBA N16R4 tree plus the five grid-audit files from the new GitHub evidence branch.
- Verification passed without GPU or Slurm: py_compile passed; `python -m pytest tests/test_rba_rbr_core.py tests/test_rba_rbr_integration.py -q` -> `23 passed in 50.13s`.
- Logs: `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_GridAuditPrecheck_20260701_600fc8f2/logs/rba_rbr_grid_audit_precheck_600fc8f2/`.
- This proves only Linux/OpenTAD audit launchability. It does not resolve the severe-low `4.42%` diagnostic, does not unlock formal full training, and creates no mAP/runtime/FLOPs/deploy/paper claim.

## 2026-07-01 07:39:58 +08:00 - RBA-RBR grid-audit GitHub API branch synced

- Route label: `DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3`.
- Ordinary `git push` to `codex/divergent-rba-rbr-20260701` was rejected because GitHub currently points that branch at remote-only commit `1b26de8a7c606d9305cfa228b7f87e920529aa6a`; ordinary HTTPS fetch/push also hit local GitHub connectivity failures.
- To avoid force push or overwriting that remote-only evidence branch, created a new GitHub branch from `1b26de8`: `codex/divergent-rba-rbr-grid-audit-aadf9708-20260701`.
- Synced the five grid-audit files from local commit `aadf9708` using the GitHub Contents API: detector grid audit, RBA integration tests, severe-low diagnosis packet, route-owned tracker mirror, and route-owned log.
- Confirmed branch ref after sync: `0a4bf5747de24f822ae23183e89f5cce234e69a8`.
- GitHub URL: `https://github.com/yuzbo/pc-ot-mras-r-series-opentad/tree/codex/divergent-rba-rbr-grid-audit-aadf9708-20260701`.
- This is evidence synchronization only. It does not constitute a Pro verdict, remote precheck, Slurm launch, training, `tools/test.py`, mAP/runtime/FLOPs/deploy/paper claim, or full-train unlock.

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

## 2026-07-01 05:34:45 +08:00 - RBA-RBR eval diagnostic first validation severe-low signal

- RBA-RBR child `1118197.539 rba_rbr_eval_g0` remained `RUNNING|0:0` on GPU0; protected parent hold `1118197 pcot_dbg2g` was not modified, released, cancelled, or replaced.
- First validation completed `1645/1645` windows with bad-pattern count `0`, then entered evaluator aggregation successfully.
- First diagnostic validation metric after epoch 1: `Average-mAP=0.12%`, `mAP@0.30=0.34%`, `mAP@0.40=0.18%`, `mAP@0.50=0.06%`, `mAP@0.60=0.02%`, `mAP@0.70=0.01%`.
- Interpretation: validation/evaluator chain is alive, but the early detector-health signal is severe-low. This is not a final route result and not a metric claim; formal full training, runtime/FLOPs, deploy, and paper claims remain locked. Because no hard failure occurred, the child was allowed to continue into epoch 2 for the scheduled bounded diagnostic follow-up.

## 2026-07-01 06:33:57 +08:00 - RBA-RBR bounded eval diagnostic completed severe-low

- RBA-RBR child `1118197.539 rba_rbr_eval_g0` completed normally on protected hold `1118197` GPU0: `COMPLETED|0:0`, elapsed `01:57:09`.
- Logdir: `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_Shortdiag_20260701_9311f49/logs/rba_rbr_evaldiag_564a6f3_gpu0_20260701_043542_+0800`.
- Bad-pattern count remained `0`; no Traceback/OOM/NaN/non-finite/permission/path failure was found.
- Final bounded diagnostic metric after epoch 3: `Average-mAP=4.42%`, `mAP@0.30=10.92%`, `mAP@0.40=6.35%`, `mAP@0.50=3.16%`, `mAP@0.60=1.28%`, `mAP@0.70=0.37%`.
- Decision: `SEVERE_RESULT_GATE_TRIGGERED`. This is stable execution but failure-scale detector performance, so RBA-RBR formal training remains locked pending sparse-forward/coordinate/postprocess diagnosis and Pro/Oracle review. Parent hold was not modified or released.

## 2026-07-01 06:45:00 +08:00 - RBA-RBR GitHub API sync after final diagnostic

- Ordinary HTTPS `git push` still failed with `Recv failure: Connection was reset`.
- Used authenticated `gh api` Git Data API sync for branch `codex/divergent-rba-rbr-20260701`.
- GitHub parent before sync: `b9d18aab922426823cf52369fa6fbde677251ec3`.
- New GitHub commit and confirmed branch ref: `88e498cdbba62e981d042a6e3fee22b7edb70f50`.
- Synced 35 RBA files, including configs, acquisition/transform code, tests, launch gate, Pro transport evidence, severe-low diagnosis packet, route-owned tracker mirror, and log.
- External review URL: `https://github.com/yuzbo/pc-ot-mras-r-series-opentad/tree/codex/divergent-rba-rbr-20260701`.
- RBA-RBR formal full training remains locked pending severe-result diagnosis.

## 2026-07-01 07:23:37 +08:00 - RBA-RBR sparse-forward precheck audit

- Added a route-specific RBA-RBR detector temporal-grid audit in `IrregularActionFormer`.
- The detector now recognizes `rba_rbr_*` metadata independently from BVR, uses `rba_rbr_detector_feature_positions` / `rba_rbr_detector_feature_valid_len` for native-axis grid construction, and fails closed on missing native-axis semantics or mask/position count mismatch.
- Added focused tests proving the RBA audit path, the fail-closed mismatch behavior, and that BVR grid auditing remains separately labeled.
- Verification passed: `python -m py_compile opentad\models\detectors\irregular_actionformer.py tests\test_rba_rbr_integration.py`; `python -m pytest tests\test_rba_rbr_core.py tests\test_rba_rbr_integration.py -q` -> `18 passed, 5 skipped`; `git diff --check` passed with line-ending warnings only.
- This is severe-result diagnosis infrastructure only. No training, remote sync, Slurm, Pro decision, mAP/runtime/FLOPs/deploy/paper claim, or full-train unlock was produced.

## 2026-07-01 14:09:53 +08:00 - RBA-RBR postprocess candidate guard repair

- Trigger: corrected guard diagnostic child `1118197.560 rba_guard_g0` produced severe-low first validation (`Average-mAP=0.22%`) with `411700` predictions despite restored RBA grid-audit coverage/gap constraints.
- Added RBA-specific postprocess guard/audit in `IrregularActionFormer`: `rba_rbr_postprocess_guard` fails closed without RBA metadata, caps raw proposals/per-class/global candidates, and writes `RBA_RBR_POSTPROCESS_AUDIT_PATH` rows labeled `DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3`.
- Updated RBA config and launch gate to require `pre_nms_topk=512`, `raw_proposal_cap=1024`, `per_class_topk=32`, and `total_candidate_cap=512`.
- Verification: py_compile passed; `python -m pytest tests\test_rba_rbr_integration.py -q` -> `8 passed, 7 skipped`; synthetic ledger build plus launch gate -> `gate_pass=true`, `full_train_unlocked=false`; `git diff --check` passed with LF/CRLF warnings only.
- No remote sync, Slurm launch, mAP/runtime/FLOPs/sparse-compute/deploy/paper claim, or formal/full-train unlock was produced by this repair. Next allowed step is a route-owned `SHORT_DIAGNOSTIC_ONLY` postprocess-guard run after current `.560` evidence is harvested or if a user override chooses to replace it.

## 2026-07-01 14:18:00 +08:00 - RBA-RBR postprocess guard GitHub evidence sync

- Ordinary `git push` failed with `Recv failure: Connection was reset`.
- Created a new evidence branch without force push: `codex/divergent-rba-rbr-postprocess-guard-546ed615-20260701`.
- GitHub Contents API returned 404 for branch refs, so synchronization used Git Data API blobs/trees/commits.
- Local commit: `546ed6159c2155633981050ae102547de03ea6bc`; GitHub base commit: `1cbdb50422f9b1c968ef92e9c036d2787d434c79`; GitHub synced ref: `6ba723abe4f5b2eca7c38f71c5b14872d637dbaa`.
- URL: `https://github.com/yuzbo/pc-ot-mras-r-series-opentad/tree/codex/divergent-rba-rbr-postprocess-guard-546ed615-20260701`.
- Synced 8 RBA postprocess-guard files for later Pro/severe-result review. No remote precheck, Slurm launch, training result, final mAP/runtime/FLOPs/sparse-compute/deploy/paper claim, or formal/full-train unlock was produced.

## 2026-07-01 15:12:35 +08:00 - RBA-RBR non-GPU coordinate/budget audit tooling

- Added `tools/rba_rbr/audit_coordinate_budget.py` in the route-owned diagnostics worktree for `DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3`.
- The CLI is CPU-only and consumes either detector grid-audit JSONL or built-in synthetic rows; it summarizes `raw_valid_k`, detector valid count, raw/detector gap stats, route/status counts, and whether detector density is far below the 384/192 fixed 50% reference.
- Added synthetic coordinate closure: raw selected native-axis positions -> detector feature centers -> synthetic native-axis segment coverage, with `uses_validation_or_test_gt=False`.
- Added `tests/test_rba_rbr_diagnostics.py` covering synthetic summaries, route label/C3 mixing rejection, CLI JSONL input, and claim-lock wording.
- Verification passed: `python -m py_compile tools\rba_rbr\audit_coordinate_budget.py tests\test_rba_rbr_diagnostics.py`; `python -m pytest tests\test_rba_rbr_diagnostics.py -q` -> `5 passed`.
- This does not run GPU, remote sync, Slurm, `tools/test.py`, or training; it does not modify evaluator/postprocess runtime behavior; it does not unlock mAP/runtime/FLOPs/sparse-compute/deploy/paper claims or formal/full RBA-RBR training.
- Next diagnostic planning remains: postprocess-guard shortdiag, matched low-budget uniform, `K=192` forced coverage, and coordinate-closure Pro diagnosis. Valid GPT-5.5 Pro severe-result diagnosis remains required before long follow-up or route-level conclusion.
