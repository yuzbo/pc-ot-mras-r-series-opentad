# Sparse TAD Task Flow Tracker

Route-owned mirror for `DIVERGENT_INNOVATION_BVR_TWB_DO_NOT_MERGE_WITH_C3`.

This BVR route worktree did not contain the shared tracker file. The shared/main worktree remains read-only for divergent-route work, so this route-owned mirror records BVR state without writing to the shared repository.

## All Required Experiments and Current Status

| Timestamp (+08:00) | Experiment / config | Changed surface | Status | Review / gate state | Deployment / result state | Next action |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-07-01 14:31:06 | `input_bvr_twb_dynamic_adapter_irregular_headv3.py` | BVR-TWB / VOI-BBC dynamic acquisition, native-axis bridge/eval path fix | Running on GPU0 | User override run; severe-result gate remains active for any collapse; no final claim | Child `1118197.542 bvr_twb_fix2_g0` still running; intermediate Average-mAP sequence `21.78 -> 22.08 -> 22.55 -> 22.86`; no `Training Over` yet | Continue material-event monitoring only; do not stop for low interim mAP alone; trigger severe-result diagnosis only for hard failure or final/severe collapse |
| 2026-07-01 07:02:05 | `input_bvr_twb_dynamic_adapter_irregular_headv3.py` | BVR-TWB / VOI-BBC dynamic acquisition, native-axis bridge/eval path fix | Running on GPU0 | User override for path-fix restart after tests/validator; severe-result gate remains active for future collapse | Child `1118197.542 bvr_twb_fix2_g0` running; pretraining loaded; first loss finite; no startup bad pattern | Monitor until validation around epoch 40; trigger severe-result diagnosis if collapse recurs |

## Timeline

### 2026-07-01 14:31:06 +08:00 - BVR-TWB intermediate validation trend observed

- Remote route worktree: `/data/run01/sczc063/yuzibo/OpenTAD_BVR_TWB_Final_20260630_92ec024`.
- Remote commit: `5d11ffd`.
- Config: `configs/adatad/thumos/input_bvr_twb_dynamic_adapter_irregular_headv3.py`.
- Child: `1118197.542 bvr_twb_fix2_g0`.
- GPU: protected hold `1118197` GPU0 only, `CUDA_VISIBLE_DEVICES=0`.
- Logdir: `/data/run01/sczc063/yuzibo/OpenTAD_BVR_TWB_Final_20260630_92ec024/logs/bvr_twb_pathfix_restart2_gpu0_5d11ffd_20260701_070013_+0800`.
- Intermediate Average-mAP sequence from the current run: `21.78%` at `2026-07-01 11:38:51`, `22.08%` at `2026-07-01 12:26:20`, `22.55%` at `2026-07-01 13:24:22`, and `22.86%` at `2026-07-01 14:31:06`.
- This is not final evidence: no `Training Over`, no final `result_detection.json`, no runtime/FLOPs evidence, no deploy claim, no paper claim, and no true sparse-compute claim.
- Current decision: keep monitoring material events only. Do not stop this route for low interim mAP alone; short/interim mAP is directional evidence and not a final route judgment under the project rule.

### 2026-07-01 07:02:05 +08:00 - BVR-TWB GPU0 path-fix restart running

- Remote route worktree: `/data/run01/sczc063/yuzibo/OpenTAD_BVR_TWB_Final_20260630_92ec024`.
- Remote commit: `5d11ffd`.
- Config: `configs/adatad/thumos/input_bvr_twb_dynamic_adapter_irregular_headv3.py`.
- Child: `1118197.542 bvr_twb_fix2_g0`.
- GPU: protected hold `1118197` GPU0 only, `CUDA_VISIBLE_DEVICES=0`.
- Logdir: `/data/run01/sczc063/yuzibo/OpenTAD_BVR_TWB_Final_20260630_92ec024/logs/bvr_twb_pathfix_restart2_gpu0_5d11ffd_20260701_070013_+0800`.
- Startup evidence: pretraining loaded, no bad pattern, first loss finite.
- Current status: running; no final mAP yet.
