---
type: idea
node_id: idea:004
title: "Projection Sparse Adaptation"
stage: proposed
outcome: pending
created_at: 2026-04-09T00:00:00Z
updated_at: 2026-04-09T12:00:00Z
---

# Projection 层稀疏适配

## Core Idea
ActionFormer Projection 层假设 dense 连续时间轴输入，但 V3 输出包含 50% stale 特征。通过 indicator embedding 或 soft attention bias 让 Projection 感知稀疏性。

## Design Options (递进)
1. **indicator_embed** (已实现): Embedding(2, C) 区分 kept/dropped，零成本
2. **soft attention bias**: Attention 中对 dropped 位置加可学习负偏置 (init=-2.0)
3. **partial conv**: confidence-weighted Conv1d

## Target Gap
G4: ActionFormer Projection 不感知稀疏性

## Why This Matters
R008 (63.74%) 比 stride=1 全帧 (68.97%) 低 5.2%。其中一部分损失可能来自 Projection Conv1d 混合 kept+stale 特征。indicator_embed 消融可以零成本验证。

## Status
indicator_embed 代码已实现 (`use_indicator_embed=True`)，只需配置启用。
