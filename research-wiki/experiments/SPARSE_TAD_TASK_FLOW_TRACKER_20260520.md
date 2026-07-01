# Sparse TAD Task Flow Tracker

Route-owned mirror for `DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3`.

This route worktree did not contain the shared tracker file. The shared/main worktree is read-only under the divergent-route rules, so this route-owned mirror records RBA-RBR state without writing to the shared repository.

## All Required Experiments and Current Status

| Timestamp (+08:00) | Experiment / config | Changed surface | Status | Review / gate state | Deployment / result state | Next action |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-07-01 04:20:30 | `input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3_shortdiag.py` | Raw RGB low-res scout acquisition and adapter bridge launchability | Completed diagnostic | Launch gate pass; full train locked | Child `1118197.538` completed, finite loss, no eval/mAP | Use only as launchability evidence |
| 2026-07-01 06:33:57 | `input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3_evaldiag.py` | Bounded eval diagnostic for detector-health signal | Completed diagnostic, severe-low | `SEVERE_RESULT_GATE_TRIGGERED`; formal full train locked | Child `1118197.539` completed; final diagnostic `Average-mAP=4.42%`, vector `10.92/6.35/3.16/1.28/0.37`; bad count `0` | Preserve evidence, update diagnosis packet, run Pro/Oracle diagnosis before any RBA long train |
| 2026-07-01 06:45:00 | GitHub API sync | Evidence/code synchronization for external diagnosis | Completed sync | Ordinary push failed; GitHub API sync succeeded | Branch `codex/divergent-rba-rbr-20260701` updated to `88e498cdbba62e981d042a6e3fee22b7edb70f50` with 35 files | Use GitHub URL for Pro/Oracle severe-result diagnosis |
| 2026-07-01 07:23:37 | RBA-RBR sparse-forward precheck audit | Detector temporal-grid audit only | Local implementation complete | Self-check passed; Pro transport remains `INCOMPLETE`; formal full train locked | Added env-gated `RBA_RBR_GRID_AUDIT_PATH` JSONL audit and fail-closed tests; local commit/GitHub evidence sync only after tests; no training, remote sync, Slurm, or Pro | Use audit in next sparse-forward precheck; do not treat as performance evidence |
| 2026-07-01 07:39:58 | RBA-RBR grid-audit GitHub API branch | Evidence/code synchronization for Pro/Oracle diagnosis | Completed sync | Ordinary git push blocked by remote divergence/network; GitHub Contents API sync succeeded without force push | New branch `codex/divergent-rba-rbr-grid-audit-aadf9708-20260701` created from remote `1b26de8` and updated to `0a4bf5747de24f822ae23183e89f5cce234e69a8` with 5 grid-audit files | Use branch URL for Pro severe-result diagnosis and remote sparse-forward precheck; no train unlock |
| 2026-07-01 07:48:48 | RBA-RBR grid-audit remote PRECHECK_ONLY | Linux/OpenTAD no-GPU sparse-forward audit precheck | Passed | Pro remains `INCOMPLETE`; full train locked | Remote route-owned copy `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_GridAuditPrecheck_20260701_600fc8f2`; py_compile passed; focused pytest `23 passed in 50.13s`; no GPU/Slurm/train/test.py/mAP | Use as audit launchability evidence only; wait for Pro or run bounded sparse-forward diagnostics when GPU0 frees |
| 2026-07-01 08:53:47 | RBA-RBR grid-audit eval diagnostic, `input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3_evaldiag.py` | Detector sparse-forward grid audit under bounded diagnostic train/eval path | Running on new Slurm job | Remote py_compile passed; focused pytest `23 passed in 48.59s`; launch gate locked full train; read-only final review passed `PASS_SUBAGENT_FINAL_REVIEW_ONLY_FOR_SHORT_DIAGNOSTIC_ONLY`; startup grid audit active | Remote code committed as `a64530e`; first non-protected job `1132461` cancelled after landing on `g0030`; active job `1132462 rba_grid_audit` runs on `g0053` with `--exclude=g0030`; audit sample has route label correct, `native_axis=True`, status `PASS_RBA_RBR_NATIVE_AXIS_POSITIONS_ENTERED_MODEL`, and matched `mask_true_count/meta_detector_feature_position_count`; no mAP yet | Continue short diagnostic; use grid-audit JSONL to diagnose severe-low coordinate/handoff path; formal/full train and all claims remain locked |
| 2026-07-01 09:31:39 | RBA-RBR grid-audit live sync/status | GitHub evidence sync plus bounded eval-audit monitoring | Running, evidence branch synced | Ordinary push to existing branch rejected as non-fast-forward; no force push; formal/full train locked | New GitHub branch `codex/divergent-rba-rbr-status-efbb4eea-20260701` pushed and fast-forwarded with the live-sync records; job `1132462` still running on `g0053`; audit file reached `1387` rows and latest row still has `status=PASS_RBA_RBR_NATIVE_AXIS_POSITIONS_ENTERED_MODEL`, `native_axis=True`, and matched mask/position counts; validation progress tail around `1010/1645` | Let job finish, then summarize full grid-audit JSONL and update severe-low diagnosis packet; no long train or claims |
| 2026-07-01 10:07:14 | `input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3.py` coverage guard fix | RBA budget controller/types/bridge, LoadFrames wiring, RBA grid-audit fields, focused tests | Local implementation complete | Local pytest and py_compile passed; no Pro/subagent review yet; full train locked | No remote sync, Slurm, `tools/test.py`, staging, commit, or push; severe-low first validation from job `1132462` remains diagnostic evidence only | Run required read-only review / severe-result diagnosis before any sync or new RBA long run; if accepted, sync as route-owned precheck only |
| 2026-07-01 10:29:01 | RBA-RBR guard eval diagnostic, `input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3_evaldiag.py` | Recoverable coverage guard, detector feature floor, raw/detector gap guard, RBA grid-audit guard fields | Queued as short diagnostic | Final read-only review passed `PASS_SUBAGENT_FINAL_REVIEW_ONLY_FOR_PRECHECK_AND_SHORT_DIAGNOSTIC_ONLY`; local/remote launch gates keep full train locked | Commit `e6de60e9`; GitHub branch `codex/divergent-rba-rbr-coverage-guard-e6de60e9-20260701`; remote tree `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_GuardDiag_20260701_e6de60e9_bundle`; remote precheck `26 passed in 89.21s`; Slurm job `1132641 rba_guarddiag` submitted with `--exclude=g0030`, initial state `PENDING` | Monitor launch, confirm non-g0030 node, then inspect guard audit summary; no formal train or claims |
| 2026-07-01 10:38:15 | RBA-RBR monitor update for old grid-audit and new guard diagnostic | Read-only Slurm/log monitoring only | Old pre-guard grid-audit running; new guard diagnostic still queued | Formal/full train locked; no claim unlock; monitor command used Slurm/log read-only evidence | Job `1132462 rba_grid_audit` remains `RUNNING` on `g0053` with finite losses through epoch 3 and first validation `Average-mAP=0.16%`; `rba_rbr_grid_audit.jsonl` reached `3962` rows. New guard job `1132641 rba_guarddiag` remains `PENDING`, reason `Priority`, no node assigned and no train log yet. Both use `--exclude=g0030`; protected hold `1118197` unchanged. | Let old audit finish for diagnosis; wait for guard job to start, then compare guard audit counts/gaps before any RBA full training |

## Timeline

### 2026-07-01 10:07:14 +08:00 - RBA-RBR recoverable coverage guard local fix

- Trigger: severe-low first validation from remote job `1132462 rba_grid_audit` (`Average-mAP=0.16%`) plus grid-audit evidence that RBA native-axis metadata reached the detector but detector effective counts could fall below the intended 32-feature floor.
- Changed files: `opentad/acquisition/rba_rbr/types.py`, `opentad/acquisition/rba_rbr/budget_controller.py`, `opentad/acquisition/rba_rbr/open_tad_bridge.py`, `opentad/datasets/transforms/end_to_end.py`, `opentad/models/detectors/irregular_actionformer.py`, `configs/adatad/thumos/input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3.py`, `tests/test_rba_rbr_core.py`, `tests/test_rba_rbr_integration.py`.
- Strict protocol: route label stayed `DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3`; no C3/BVR/MDL behavior was edited; selector guard uses deploy-visible preview/risk only and does not inspect GT, teacher outputs, raw predictions, or caches.
- Behavior: original scaffold/refine/rescue regret ranking is still the first selection pass. A post-selection coverage guard backfills within `max_k` to meet raw selected count, detector feature count, raw max gap, and detector max gap constraints when possible. Backfill ranking uses recoverability/regret/staleness/rescue risk, with center-band restriction only for hard gap closure.
- Config: RBA main config now sets `rba_rbr_min_detector_feature_keep=32`, `rba_rbr_max_raw_gap=16`, `rba_rbr_max_detector_gap=24` in addition to existing `rba_rbr_min_keep=64`, `rba_rbr_max_keep=192`, and `rba_rbr_feature_stride=2`.
- Verification: `python -m pytest tests/test_rba_rbr_core.py tests/test_rba_rbr_integration.py -q` -> `21 passed, 5 skipped in 6.07s`; py_compile for the five requested implementation files -> pass; `git diff --check` -> pass with LF/CRLF warnings only.
- Decision: local implementation is ready for read-only review, not for formal training. No remote sync, Slurm, `tools/test.py`, Pro, staging, commit, or push was performed. No final mAP, runtime/FLOPs, sparse-compute, deploy, or paper claim is unlocked.

### 2026-07-01 10:29:01 +08:00 - RBA-RBR guard diagnostic queued

- Local commit: `e6de60e9 DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3 coverage guard`.
- GitHub evidence branch: `codex/divergent-rba-rbr-coverage-guard-e6de60e9-20260701`.
- GitHub URL: `https://github.com/yuzbo/pc-ot-mras-r-series-opentad/tree/codex/divergent-rba-rbr-coverage-guard-e6de60e9-20260701`.
- Final read-only review: agent `019f1b74-27c5-72c2-8a96-fd862dcd7ac6` returned `PASS_SUBAGENT_FINAL_REVIEW_ONLY_FOR_PRECHECK_AND_SHORT_DIAGNOSTIC_ONLY`; blockers: none.
- Accepted non-blocking hardening: launch gate now explicitly requires `rba_rbr_min_detector_feature_keep=32`, `rba_rbr_max_raw_gap=16`, and `rba_rbr_max_detector_gap=24`.
- Verification after hardening: local pytest `21 passed, 5 skipped`; local py_compile -> pass; local launch gate -> `gate_pass=true`, `full_train_unlocked=false`; remote route-owned precheck pytest `26 passed in 89.21s`, py_compile -> pass, launch gate -> pass.
- Remote sync path: GitHub clone on N16R4 failed with `Empty reply from server`, so a local git bundle was uploaded and cloned into `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_GuardDiag_20260701_e6de60e9_bundle`.
- Slurm: submitted `1132641 rba_guarddiag` with `--exclude=g0030`; initial `sacct` state was `PENDING|0:0|None assigned`.
- Logdir: `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_GuardDiag_20260701_e6de60e9_bundle/logs/rba_rbr_guard_evaldiag_e6de60e9_20260701/`.
- Decision: this is `SHORT_DIAGNOSTIC_ONLY` to check guard-restored raw/detector counts, gap closure, native-axis grid audit, and diagnostic mAP health. It is not formal training and unlocks no mAP/runtime/FLOPs/sparse-compute/deploy/paper claim.

### 2026-07-01 10:38:15 +08:00 - RBA-RBR short diagnostic monitor update

- Read-only Slurm/log scan: `1132462 rba_grid_audit` remained `RUNNING|0:0` on `g0053`; `1132641 rba_guarddiag` remained `PENDING|0:0`, reason `Priority`, with no node assigned.
- Old pre-guard evidence: the grid-audit run reached finite training through epoch 3, retained the severe-low first validation signal `Average-mAP=0.16%`, and `rba_rbr_grid_audit.jsonl` reached `3962` rows. This evidence belongs to the pre-guard branch and must not be interpreted as the new coverage guard result.
- New guard evidence: the guard logdir still contains only the sbatch script, with no `slurm-1132641.out`, no `train.log`, and no guard audit summary yet because the job has not started.
- Resource boundary: both RBA jobs use `--exclude=g0030`; protected parent hold `1118197 pcot_dbg2g` was not modified, released, cancelled, or reused.
- Decision: keep formal/full training and all mAP/runtime/FLOPs/sparse-compute/deploy/paper claims locked. Let old audit finish for diagnosis and wait for the guard diagnostic to start before judging whether the recoverable coverage guard fixes the collapse.

### 2026-07-01 08:53:47 +08:00 - RBA-RBR grid-audit diagnostic launched off protected hold

- Config: `configs/adatad/thumos/input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3_evaldiag.py`.
- Remote route worktree: `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_Shortdiag_20260701_9311f49`.
- Remote code sync commit: `a64530e DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3_grid_audit_sync`.
- Verification before launch:
  - py_compile passed for detector/test/gate files.
  - `python -m pytest tests/test_rba_rbr_core.py tests/test_rba_rbr_integration.py -q` -> `23 passed in 48.59s`.
  - Launch gate output: `gate_pass=true`, `full_train_unlocked=false`, `deploy_claim_unlocked=false`, `paper_claim_unlocked=false`, `sparse_compute_claim=false`.
- Read-only final review: agent `019f1b23-9b5a-7793-9373-002984174ef5` returned `PASS_SUBAGENT_FINAL_REVIEW_ONLY_FOR_SHORT_DIAGNOSTIC_ONLY` and no blockers.
- Resource safety:
  - First new Slurm job `1132461` landed on `g0030` and was cancelled after `00:01:23` because it could overlap the protected hold node.
  - This cancellation affected only the new RBA Slurm job, not the protected parent hold `1118197 pcot_dbg2g`.
  - Relaunched job `1132462 rba_grid_audit` with `--exclude=g0030`; it started on `g0053`, allocated 1 GPU and 4 CPU.
- Active logdir: `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_Shortdiag_20260701_9311f49/logs/rba_rbr_grid_audit_evaldiag_sbatch_3cab4121_20260701_0853_exclude_g0030_+0800/`.
- Current mAP evidence: none from this job yet.
- Decision: this is a `SHORT_DIAGNOSTIC_ONLY` sparse-forward grid audit. It may diagnose whether native-axis detector feature positions and masks enter the detector correctly. It does not unlock formal/full long training, official mAP, runtime/FLOPs, deploy, paper, or sparse-compute claims.

### 2026-07-01 08:58:37 +08:00 - RBA-RBR grid-audit startup evidence

- Job state: `1132462 rba_grid_audit` running on `g0053`.
- Training startup: reached epoch 0 iteration 140 with finite loss; latest sampled line at inspection time was `[000][00140/00199] Loss=2.5149`.
- Audit file: `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_Shortdiag_20260701_9311f49/logs/rba_rbr_grid_audit_evaldiag_sbatch_3cab4121_20260701_0853_exclude_g0030_+0800/rba_rbr_grid_audit.jsonl`.
- Audit sample: 169 rows sampled; all sampled route labels are `DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3`; first sampled row has `status=PASS_RBA_RBR_NATIVE_AXIS_POSITIONS_ENTERED_MODEL`, `native_axis=True`, `mask_true_count=23`, and `meta_detector_feature_position_count=23`.
- Interpretation: the detector-side grid audit is active and native-axis RBA detector feature positions enter the model. This narrows the severe-low diagnosis: the immediate failure is less likely to be "RBA metadata never reaches detector grid" and still may lie in selector quality, adapter bridge semantics, coordinate scaling, loss/assignment alignment, or post-processing.
- Claims: no final mAP, runtime/FLOPs, sparse-compute, deploy, or paper claim unlocked.

### 2026-07-01 09:31:39 +08:00 - RBA-RBR grid-audit live status and evidence branch sync

- Existing branch push: ordinary `git push` to `codex/divergent-rba-rbr-20260701` was rejected as non-fast-forward, so no force push or overwrite was attempted.
- Evidence branch sync: pushed current local route-owned state to `codex/divergent-rba-rbr-status-efbb4eea-20260701`; after recording tracker/log evidence, the branch was fast-forwarded with those live-sync records.
- GitHub URL: `https://github.com/yuzbo/pc-ot-mras-r-series-opentad/tree/codex/divergent-rba-rbr-status-efbb4eea-20260701`.
- Job state: `1132462 rba_grid_audit` remains `RUNNING` on `g0053`, not protected hold node `g0030`.
- Audit evidence: `rba_rbr_grid_audit.jsonl` reached `1387` rows; latest inspected row still had `route_label=DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3`, `status=PASS_RBA_RBR_NATIVE_AXIS_POSITIONS_ENTERED_MODEL`, `native_axis=True`, and matched `mask_true_count=32` / `meta_detector_feature_position_count=32`.
- Train/eval evidence: short diagnostic training completed two epochs with finite losses, then entered validation/audit; progress tail showed around `1010/1645` validation windows.
- Decision: continue this `SHORT_DIAGNOSTIC_ONLY` job to completion, then summarize the full JSONL and update the severe-low diagnosis packet. Formal/full long training, `tools/test.py` result claims, runtime/FLOPs, deploy, paper, and sparse-compute claims remain locked.

### 2026-07-01 06:33:57 +08:00 - RBA-RBR bounded eval diagnostic completed

- Config: `configs/adatad/thumos/input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3_evaldiag.py`.
- Remote route worktree: `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_Shortdiag_20260701_9311f49`.
- Slurm child: `1118197.539 rba_rbr_eval_g0`.
- GPU: protected hold `1118197` GPU0 only, `CUDA_VISIBLE_DEVICES=0`.
- State: `COMPLETED|0:0`, elapsed `01:57:09`.
- Logdir: `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_Shortdiag_20260701_9311f49/logs/rba_rbr_evaldiag_564a6f3_gpu0_20260701_043542_+0800`.
- Bad-pattern count: `0`.
- Epoch-1 diagnostic: `Average-mAP=0.12%`, vector `0.34/0.18/0.06/0.02/0.01`.
- Epoch-3/final bounded diagnostic: `Average-mAP=4.42%`, vector `10.92/6.35/3.16/1.28/0.37`.
- Interpretation: stable execution but severe-low detector performance. No final route result, no deploy claim, no paper claim, no runtime/FLOPs claim.
- Decision: formal RBA-RBR long training remains locked. Next required action is severe-result diagnosis of sparse-forward raw-frame handoff, coordinate metadata, adapter bridge, and postprocess/evaluator alignment.

### 2026-07-01 06:45:00 +08:00 - RBA-RBR GitHub API sync completed

- Ordinary `git push` failed with network reset.
- Used authenticated GitHub Git Data API sync.
- GitHub parent before sync: `b9d18aab922426823cf52369fa6fbde677251ec3`.
- New GitHub commit and confirmed branch ref: `88e498cdbba62e981d042a6e3fee22b7edb70f50`.
- Synced file count: `35`.
- External review URL: `https://github.com/yuzbo/pc-ot-mras-r-series-opentad/tree/codex/divergent-rba-rbr-20260701`.

### 2026-07-01 07:23:37 +08:00 - RBA-RBR detector temporal-grid audit added

- Changed surface: `opentad/models/detectors/irregular_actionformer.py` and `tests/test_rba_rbr_integration.py`.
- Purpose: severe-low follow-up for true sparse raw-frame handoff / sparse-forward precheck, specifically proving RBA-RBR detector feature positions enter the native-axis temporal grid.
- Audit behavior: RBA-RBR metadata is detected independently from BVR via `rba_rbr_*`; RBA grid construction requires `irregular_native_axis=True`, requires mask true count to equal `rba_rbr_detector_feature_positions` length, and writes JSONL only when `RBA_RBR_GRID_AUDIT=1` and `RBA_RBR_GRID_AUDIT_PATH` are set.
- Verification: `python -m py_compile opentad\models\detectors\irregular_actionformer.py tests\test_rba_rbr_integration.py` passed; explicit RBA suite `python -m pytest tests\test_rba_rbr_core.py tests\test_rba_rbr_integration.py -q` -> `18 passed, 5 skipped`.
- Non-action: no training, no remote sync, no Slurm, no Pro submission, and no GPU job. The optional CPU ledger diagnostic tool was deferred to keep this stage minimal.
- Decision: this is audit/precheck evidence only, not a mAP/runtime/FLOPs/deploy/paper claim. Formal RBA-RBR full training remains locked; prior Pro transport remains `INCOMPLETE`.

### 2026-07-01 07:39:58 +08:00 - RBA-RBR grid-audit GitHub API branch synced

- Ordinary `git push` to `codex/divergent-rba-rbr-20260701` was rejected because the GitHub branch currently points to remote-only commit `1b26de8a7c606d9305cfa228b7f87e920529aa6a`; ordinary HTTPS fetch/push also hit local GitHub connectivity failures.
- To avoid force-pushing or overwriting the remote-only commit, created a new GitHub branch from `1b26de8`: `codex/divergent-rba-rbr-grid-audit-aadf9708-20260701`.
- Synced the five files from local commit `aadf9708` using the GitHub Contents API:
  - `opentad/models/detectors/irregular_actionformer.py`.
  - `tests/test_rba_rbr_integration.py`.
  - `research-wiki/experiments/DIVERGENT_RBA_RBR_SEVERE_LOW_DIAGNOSIS_PACKET_20260701.md`.
  - `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`.
  - `research-wiki/log.md`.
- Confirmed branch ref after sync: `0a4bf5747de24f822ae23183e89f5cce234e69a8`.
- GitHub URL: `https://github.com/yuzbo/pc-ot-mras-r-series-opentad/tree/codex/divergent-rba-rbr-grid-audit-aadf9708-20260701`.
- This is evidence synchronization only. No Pro verdict, remote precheck, Slurm job, training, `tools/test.py`, metric/runtime/deploy/paper claim, or full-train unlock was produced by this sync.

### 2026-07-01 07:48:48 +08:00 - RBA-RBR grid-audit remote PRECHECK_ONLY passed

- Created route-owned remote precheck copy: `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_GridAuditPrecheck_20260701_600fc8f2`.
- Source: existing RBA route-owned N16R4 tree `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_Shortdiag_20260701_9311f49` plus the five grid-audit files from the GitHub evidence branch.
- Uploaded package: `/data/run01/sczc063/yuzibo/rba_rbr_grid_audit_600fc8f2.tar`.
- Verification:
  - `python -m py_compile opentad/models/detectors/irregular_actionformer.py tests/test_rba_rbr_integration.py` passed.
  - `python -m pytest tests/test_rba_rbr_core.py tests/test_rba_rbr_integration.py -q` -> `23 passed in 50.13s`.
- Logs:
  - `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_GridAuditPrecheck_20260701_600fc8f2/logs/rba_rbr_grid_audit_precheck_600fc8f2/py_compile.log`.
  - `/data/run01/sczc063/yuzibo/OpenTAD_RBA_RBR_GridAuditPrecheck_20260701_600fc8f2/logs/rba_rbr_grid_audit_precheck_600fc8f2/pytest.log`.
- Non-actions: no GPU, no Slurm child, no `tools/train.py`, no `tools/test.py`, no official evaluation, no mAP/runtime/FLOPs/deploy/paper claim, and no protected-hold release/cancel/replacement.
- Decision: this proves the sparse-forward detector-grid audit is runnable in the Linux/OpenTAD environment only. It does not resolve the RBA-RBR severe-low result, and formal full training remains locked pending Pro/diagnostic decision.
