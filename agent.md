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
- Before launching any innovative experiment, first complete a Claude CLI direction-and-implementation discussion and write the discussion record here.
- The discussion must include the agent's own reasoning and reflection, not just Claude's recommendation. Claude's advice is an input to challenge and refine, not a command to follow blindly.
- A valid discussion record must separate:
  - Claude's concrete claims or objections.
  - The agent's own agreement, disagreement, or partial acceptance, grounded in observed metrics, training contracts, tensor/label semantics, and leakage risk.
  - The final decision rule and the falsifiable metric gate for the next run.
  - What will not be changed in the next experiment, so one-variable ablations remain clean.
- Do not mark a Claude exchange as complete if the agent only copies Claude's conclusion without independent critique.
- If Claude CLI returns a transient error such as API 429, do not treat the discussion as complete and do not launch the innovative experiment until a later successful discussion.

## Current Direction

- Current active experiment: detached quality reranking on Adapter + ActionFormer while keeping random-fixed 50% input unchanged.
- Historical anchors:
  - random-fixed Adapter baseline: about 61.95 Average-mAP.
  - NMS/EMA audited baseline: about 63.75-63.85 Average-mAP.
  - oracle boundary dense Adapter: about 75.50 Average-mAP.
- Next candidate ablation prepared locally but not deployed:
  `configs/adatad/thumos/input_random_fixed_50pct_adapter_quality_positive_maxiou_posonly.py`.
  This uses `target_mode="positive_max_iou"`, `negative_weight=0.0`, and `loss_normalizer="positive"` to avoid dense negative BCE pressure on the quality head.

## Claude CLI Discussion Record

- A formal Claude CLI discussion is required before launching `input_random_fixed_50pct_adapter_quality_positive_maxiou_posonly.py` or any other new innovative experiment.
- 2026-05-16: three Claude CLI attempts were made for the positive-max-IoU direction discussion:
  - full direction prompt: API 429 peak-load rejection.
  - shortened direction prompt: API 429 peak-load rejection.
  - minimal availability probe: socket connection closed unexpectedly.
- Because no successful Claude CLI discussion completed, the next innovative experiment must remain pending. Continue supervising the already-running assigned-IoU quality experiments only.

### 2026-05-16 Successful Quality-Target Discussion

- Later Claude CLI discussion completed successfully.
- Claude's main objection: `positive_max_iou` with positive-only supervision and bias-initialized quality near 0.99 can leave background locations high-quality, making quality fusion inert or poorly calibrated.
- Agent reflection:
  - This objection is valid for background calibration: if negatives are not supervised, score fusion may not suppress background proposals.
  - However, dense max-IoU on all valid points is also not automatically correct, because negative classification locations can decode proposals that overlap GT; forcing all such locations through dense quality BCE may conflict with the existing assignment/classification contract.
  - The cleaner next training ablation is therefore not to switch to dense max-IoU immediately, but to keep the assigned-IoU target and reduce negative pressure with `negative_weight=0.25` plus a weighted normalizer. This changes one variable: negative quality supervision strength.
- Final gate:
  - If current detached-quality run is below the random-fixed Adapter baseline (~61.95 Average-mAP), first re-evaluate the same checkpoint with `score_alpha=0` to separate training harm from inference fusion harm.
  - If it is between ~61.95 and the NMS/EMA audited baseline (~63.75-63.85), launch the reduced-negative assigned-IoU training ablation.
  - If it reaches ~63.85-65.20, run inference-only `score_alpha` sweeps before any new training.
  - If it exceeds ~65.20, lock the config and measure seed variance.
- Current decision: do not deploy `positive_max_iou_posonly` yet. Continue the two active runs until first eval, then apply the gate above.
