---
type: experiment
node_id: exp:PROJ_SOFT_BIAS
title: "Projection Soft Attention Bias"
date: 2026-04-09
status: running
config: v3_uniform_50pct_soft_attn_bias.py / v3_hard_scorer_aux_soft_attn_bias.py
---

# Projection Soft Attention Bias

## Purpose
对 dropped token 在 attention 中施加软负偏置，而不是完全屏蔽或完全等同对待，尝试在保留时序连续性的同时抑制 stale token 干扰。

## Status
- 配置已运行
- 当前服务器上有对应 hard_scorer_aux_soft_attn_bias 训练正在推进

## Hypothesis
这是 projection-aware 改造里最小侵入、最值得验证的一条线。
