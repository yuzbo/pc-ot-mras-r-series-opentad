---
type: paper
node_id: paper:rao2021_dynamicvit
title: "DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification"
authors: ["Yongming Rao", "Wenliang Zhao", "Benlin Liu", "Jiwen Lu", "Jie Zhou", "Cho-Jui Hsieh"]
year: 2021
venue: NeurIPS
tags: [token_pruning, ViT, soft_gating, efficiency]
relevance: core
created_at: 2026-04-09T12:00:00Z
updated_at: 2026-04-09T12:00:00Z
---

# DynamicViT: 训练时 soft gating + 推理时 hard pruning 的 token 稀疏化

## Problem / Gap

ViT 计算量与 token 数量平方成正比，需要动态剪枝不重要的 token。

## Method

- 训练时: soft mask gating (Gumbel-Softmax)，全序列 dense attention + 输出门控
- 推理时: hard top-k pruning，只保留重要 token
- 梯度通过 Gumbel-Softmax straight-through estimator 回传到 scorer

## Key Results

- ImageNet 分类: 保留 70% token 时精度损失 <0.5%

## Limitations / Failure Modes

- **Train-test gap**: 训练时 soft mask (连续值) ≠ 推理时 hard mask (离散值)
- 在图像分类上 gap 可接受，但在 TAD 密集预测任务中 gap 被放大

## Relevance to This Project

V3 soft gating 设计直接借鉴 DynamicViT。但 R012-R014, R021 实验证明 soft gating 在 TAD 中导致约 10% 性能损失，train-test gap 是致命问题。**此路线已被判死。**
