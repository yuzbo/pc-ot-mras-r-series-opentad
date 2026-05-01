---
type: idea
node_id: idea:007
title: "Soft Attention Bias for Projection"
stage: proposed
outcome: pending
created_at: 2026-04-09T13:00:00Z
updated_at: 2026-04-09T13:00:00Z
---

# Soft Attention Bias

## Core Idea
在 Projection 的 MaskedMHCA 中，对 dropped 位置的 K/V 加可学习负偏置 (init=-2.0)，降低 stale 特征的 attention 权重但不完全屏蔽。

## Design
- att = att + stale_kv.float() * self.stale_bias
- stale_bias 可学习，初始 -2.0
- 不丢信息，模型自己决定降多少权重

## Target Gap
G4: Projection 不感知稀疏性

## Config
- v3_uniform_50pct_soft_attn_bias.py
- v3_hard_scorer_aux_soft_attn_bias.py

## Status
代码已实现，配置已就绪，待运行。
