---
type: idea
node_id: idea:010
title: "V3 Ablation Studies (attn_scale / mask_adapter)"
stage: proposed
outcome: pending
created_at: 2026-04-09T13:00:00Z
updated_at: 2026-04-09T13:00:00Z
---

# V3 Ablation Studies

## Core Idea
验证 V3 关键设计决策的贡献。

## Ablations
1. **attn_scale**: v3_ablation_attn_scale.py — 启用可学习 attn_scale，预期性能下降
2. **mask_adapter**: v3_ablation_no_mask_adapter.py — 关闭 adapter mask，预期 stale 特征污染

## Target
验证 R008 的设计决策是否最优。

## Status
配置已就绪，待运行。
