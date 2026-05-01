---
type: experiment
node_id: exp:R013
title: "Gumbel + Soft Gating + BCE=1.0"
date: 2026-04-06
status: completed
config: v3_gumbel_softgate_scorer.py
---

# R013: Gumbel + Soft Gating + BCE=1.0

## Purpose
加入 boundary BCE 直接监督 scorer。

## Key Config
- mode=gumbel, soft_gating=True, lambda_scorer=1.0

## Results
- Avg-mAP ≈ 53% ❌

## Conclusion
BCE 监督 scorer 选边界帧，但 det loss 需要的不一定是边界帧，两个目标冲突。加 BCE 反而更差。
