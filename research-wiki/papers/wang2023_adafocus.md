---
type: paper
node_id: paper:wang2023_adafocus
title: "Uni-AdaFocus: Spatial-Temporal Dynamic Computation for Video Recognition"
authors: ["Yulin Wang", "Zhaoxi Chen", "Haojun Jiang", "Shiji Song", "Yizeng Han", "Gao Huang"]
year: 2023
venue: TPAMI
tags: [adaptive_inference, frame_selection, video_classification, policy_gradient]
relevance: core
created_at: 2026-04-09T12:00:00Z
updated_at: 2026-04-09T12:00:00Z
---

# Uni-AdaFocus: 视频理解中的自适应时空计算

## Problem / Gap

视频理解中大量帧是冗余的，需要自适应选择信息密度高的帧和空间区域。

## Method

- 轻量 scorer (GFV global feature volume) 预测帧重要性
- Policy gradient (REINFORCE) 训练选帧策略
- Backbone 不变，只改输入帧选择
- 多路 loss: 分类 loss + policy gradient + 辅助 loss

## Key Results

- 视频分类任务上成功: 减少 50% 计算量，精度损失 <1%

## Limitations / Failure Modes

- 核心假设"全局特征足以指导选帧"在 TAD 密集预测中不成立
- TAD 需要密集时序预测，选帧策略需要更精细的时序感知
- "Backbone 不变，只改输入"原则在 V3 backbone 级 masking 中被违反

## Relevance to This Project

多路 loss 设计 (L_det, L_scorer, L_aux, L_coarse_det) 借鉴了 AdaFocus。REINFORCE policy gradient (R020) 也源自此工作。但 AdaFocus 的成功场景 (视频分类) 与 TAD (密集检测) 有本质差异。
