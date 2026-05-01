---
type: experiment
node_id: exp:R014
title: "Gumbel + Soft Gating + BCE + Aux"
date: 2026-04-06
status: completed
config: v3_gumbel_softgate_scorer_aux.py
---

# R014: Gumbel + Soft Gating + BCE + Aux

## Purpose
加入辅助分类 loss 提供短路径梯度。

## Key Config
- mode=gumbel, soft_gating=True, lambda_scorer=1.0, lambda_aux=0.5

## Results
- Avg-mAP = 53.73% ❌

## Conclusion
多任务拉扯加剧，三个 loss 目标不一致，scorer 无法收敛到有意义的策略。
