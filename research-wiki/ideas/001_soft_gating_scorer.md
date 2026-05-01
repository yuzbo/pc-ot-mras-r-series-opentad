---
type: idea
node_id: idea:001
title: "Soft Gating Learned Scorer (DynamicViT-style)"
stage: failed
outcome: negative
created_at: 2026-04-06T00:00:00Z
updated_at: 2026-04-09T12:00:00Z
---

# Soft Gating Learned Scorer

## Core Idea
用 DynamicViT-style soft gating 训练 scorer：训练时 soft mask 门控 (O(N²))，推理时 hard sparse (O(K²))。通过 Gumbel-Softmax 实现可微采样，det loss 端到端传梯度到 scorer。

## Target Gap
G2: Learned scorer 无法超越 uniform 采样

## Tested By
- exp:R012 (纯 E2E, ~52-55%) ❌
- exp:R013 (+ BCE=1.0, ~53%) ❌
- exp:R014 (+ BCE + aux, 53.73%) ❌
- exp:R021 (最小对照, 53.98%) ❌ **判定实验**

## Failure Analysis
**Root cause: Train-test gap 致命**
- 训练: soft mask × full attention (O(N²), 连续 mask)
- 推理: hard gather/scatter (O(K²), 离散 mask)
- 12 层累积的 soft mask 缩放导致特征幅度系统性偏移
- R021 证明即使最小改动 (极低 BCE=0.05)，soft gating 仍导致 ~10% 性能损失
- 与 loss 配比无关，是 soft gating 训练方式本身的结构性缺陷

## Lesson Learned
DynamicViT soft gating 适用于分类 (全局池化容忍偏移)，不适用于 TAD 密集预测。TAD 中训练推理必须一致 (都用 hard mode)。
