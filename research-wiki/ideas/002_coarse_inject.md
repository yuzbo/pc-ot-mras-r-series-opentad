---
type: idea
node_id: idea:002
title: "Coarse Feature Injection into Backbone Adapter"
stage: proposed
outcome: pending
created_at: 2026-04-08T00:00:00Z
updated_at: 2026-04-09T12:00:00Z
---

# Coarse Feature Injection

## Core Idea
将 scorer 的 coarse_feat 直接注入 backbone 每层 adapter，为 dropped 位置提供真实信号替代 stale 特征。

## Design
- coarse_feat (B, 128, T) → Linear(128, 384) + LN → token 空间
- chain 模式: 第一层注入 coarse_proj，后续层接收上一层 adapter 输出
- 每层可学习 α 控制注入强度 (初始 0.01)
- 新梯度路径: L_det → adapter → (x + α·coarse_proj) → scorer

## Target Gap
G3: Dropped 位置特征质量差

## Status
代码已实现并通过审查。原方案依赖 soft gating (已判死)，需改造为 hard mode 版本。

## Risk
- 依赖 soft gating 的原方案已取消
- Hard mode 版本需要配合 REINFORCE 或两阶段训练
