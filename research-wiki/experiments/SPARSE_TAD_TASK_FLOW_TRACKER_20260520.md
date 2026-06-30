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

## Timeline

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
