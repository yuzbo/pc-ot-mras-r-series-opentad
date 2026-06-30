# Sparse TAD Task Flow Tracker

Route-owned mirror for `DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3`.

This route worktree did not contain the shared tracker file. The shared/main worktree is read-only under the divergent-route rules, so this route-owned mirror records RBA-RBR state without writing to the shared repository.

## All Required Experiments and Current Status

| Timestamp (+08:00) | Experiment / config | Changed surface | Status | Review / gate state | Deployment / result state | Next action |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-07-01 04:20:30 | `input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3_shortdiag.py` | Raw RGB low-res scout acquisition and adapter bridge launchability | Completed diagnostic | Launch gate pass; full train locked | Child `1118197.538` completed, finite loss, no eval/mAP | Use only as launchability evidence |
| 2026-07-01 06:33:57 | `input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3_evaldiag.py` | Bounded eval diagnostic for detector-health signal | Completed diagnostic, severe-low | `SEVERE_RESULT_GATE_TRIGGERED`; formal full train locked | Child `1118197.539` completed; final diagnostic `Average-mAP=4.42%`, vector `10.92/6.35/3.16/1.28/0.37`; bad count `0` | Preserve evidence, update diagnosis packet, run Pro/Oracle diagnosis before any RBA long train |
| 2026-07-01 06:45:00 | GitHub API sync | Evidence/code synchronization for external diagnosis | Completed sync | Ordinary push failed; GitHub API sync succeeded | Branch `codex/divergent-rba-rbr-20260701` updated to `88e498cdbba62e981d042a6e3fee22b7edb70f50` with 35 files | Use GitHub URL for Pro/Oracle severe-result diagnosis |

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
