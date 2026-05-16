# Agent Notes

## Current Remote Training

- Use the old AutoDL servers for the active THUMOS14 Adapter + ActionFormer runs:
  - `ssh -p 35407 root@connect.cqa1.seetacloud.com`
  - `ssh -p 25876 root@connect.cqa1.seetacloud.com`
- Active remote project path: `/root/autodl-tmp/OpenTAD_Back_check`.
- Active config on both servers:
  `configs/adatad/thumos/input_random_fixed_50pct_adapter_quality_rescore_detached.py`.
- Do not stop or replace the active remote runs before the first eval unless the user explicitly authorizes it.
- Small isolated `non-finite gradients` events are expected training noise for this run. Record them, but do not treat a single event as a blocker if loss continues normally.

## Review Policy

- Use Claude CLI for direction discussion and code review.
- Do not use the `claude-review` MCP path unless the user explicitly changes this rule.
- After code review and local fixes, run local checks before any deployment.

## Current Direction

- Current active experiment: detached quality reranking on Adapter + ActionFormer while keeping random-fixed 50% input unchanged.
- Historical anchors:
  - random-fixed Adapter baseline: about 61.95 Average-mAP.
  - NMS/EMA audited baseline: about 63.75-63.85 Average-mAP.
  - oracle boundary dense Adapter: about 75.50 Average-mAP.
- Next candidate ablation prepared locally but not deployed:
  `configs/adatad/thumos/input_random_fixed_50pct_adapter_quality_positive_maxiou_posonly.py`.
  This uses `target_mode="positive_max_iou"`, `negative_weight=0.0`, and `loss_normalizer="positive"` to avoid dense negative BCE pressure on the quality head.
