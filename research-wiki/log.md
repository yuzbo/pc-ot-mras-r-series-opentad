# BVR-TWB Route Log

Route-owned log mirror for `DIVERGENT_INNOVATION_BVR_TWB_DO_NOT_MERGE_WITH_C3`.

## 2026-07-01 14:31:06 +08:00 - BVR-TWB intermediate validation trend observed

- Current child `1118197.542 bvr_twb_fix2_g0` is still running on protected hold `1118197` GPU0 only.
- Logdir: `/data/run01/sczc063/yuzibo/OpenTAD_BVR_TWB_Final_20260630_92ec024/logs/bvr_twb_pathfix_restart2_gpu0_5d11ffd_20260701_070013_+0800`.
- Intermediate Average-mAP sequence so far: `21.78% -> 22.08% -> 22.55% -> 22.86%`.
- Latest observed point: `2026-07-01 14:31:06 Train INFO: Average-mAP: 22.86 (%)`.
- No `Training Over` or final result exists yet. This is not a final success/failure judgment and does not unlock runtime/FLOPs, deploy, paper, or true sparse-compute claims.
- Next action: continue material-event monitoring; do not stop for low interim mAP alone.

## 2026-07-01 07:02:05 +08:00 - BVR-TWB GPU0 path-fix restart running

- Previous BVR child `1118197.519` reached epoch 41 and then failed in evaluation because the evaluator still referenced `/root/autodl-tmp/annotations/thumos_14_anno.json`.
- Current remote route-owned worktree `/data/run01/sczc063/yuzibo/OpenTAD_BVR_TWB_Final_20260630_92ec024` resolves the annotation and evaluation path to `/data/home/sczc063/run/yuzibo/thumos14/annotations/thumos_14_anno.json`.
- Remote focused tests passed: `36 passed in 35.45s`.
- Remote geometry validator passed with `torch_runtime_contract=passed`.
- First restart child `1118197.541` failed only because `LOCAL_RANK` was missing from the launch environment.
- Relaunched as child `1118197.542 bvr_twb_fix2_g0` on protected hold `1118197` GPU0 only, `CUDA_VISIBLE_DEVICES=0`.
- Logdir: `/data/run01/sczc063/yuzibo/OpenTAD_BVR_TWB_Final_20260630_92ec024/logs/bvr_twb_pathfix_restart2_gpu0_5d11ffd_20260701_070013_+0800`.
- Startup check: running, pretraining loaded, no bad patterns, first training line `[000][00050/00199] Loss=2.5199 cls_loss=0.5256 reg_loss=0.4425 boundary_loss=1.5518 mem=1454MB`.
- Next action: monitor normally until the first validation around epoch 40, unless NaN/OOM/Traceback/protocol failure appears earlier.
