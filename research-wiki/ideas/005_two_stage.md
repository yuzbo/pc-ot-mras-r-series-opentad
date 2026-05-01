---
type: idea
node_id: idea:005
title: "Two-Stage Training (Frozen Backbone + Scorer)"
stage: proposed
outcome: pending
created_at: 2026-04-09T00:00:00Z
updated_at: 2026-04-09T12:00:00Z
---

# 两阶段训练

## Core Idea
解耦 backbone 训练和 scorer 训练，避免 scorer 训练破坏 backbone 特征质量。

## Design
- Stage 1: 加载 R008 checkpoint (63.74%)
- Stage 2: 冻结 backbone + projection + rpn_head，只训 scorer (BCE + coarse_det)
- Stage 3 (可选): 用训好的 scorer 生成 mask，解冻全模型 finetune

## Target Gap
G2: Learned scorer 无法超越 uniform 采样

## Why This Might Work
- 完全避免 scorer 训练破坏 backbone
- Scorer 在稳定的特征空间上学习选帧策略
- 类似 AdaFocus 的"backbone 不变只改输入"原则

## Risk
- Stage 2 中 scorer 只能通过 BCE/coarse_det 学习，无 det loss 信号
- 冻结 backbone 后 scorer 的选帧可能无法适应 backbone 的特征分布
