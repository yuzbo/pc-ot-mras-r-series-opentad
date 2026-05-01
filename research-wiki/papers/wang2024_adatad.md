---
type: paper
node_id: paper:wang2024_adatad
title: "AdaTAD: An Adaptive Temporal Action Detection Framework"
authors: ["Shuming Liu", "Chen-Lin Zhang", "Chen Zhao", "Bernard Ghanem"]
year: 2024
venue: CVPR
tags: [TAD, VideoMAE, ActionFormer, end-to-end]
relevance: core
created_at: 2026-04-09T12:00:00Z
updated_at: 2026-04-09T12:00:00Z
---

# AdaTAD: 端到端时序动作检测框架，VideoMAE + ActionFormer

## Problem / Gap

传统 TAD 方法使用预提取特征，无法端到端优化。AdaTAD 将 VideoMAE backbone 与 ActionFormer 检测头联合训练。

## Method

- VideoMAE-S (ViT-Small, 384 dim, 12 layers) 作为 backbone
- Temporal Adapter: 每层 ViT block 后加 temporal depthwise Conv1d adapter
- ActionFormer: Conv1d Projection + Transformer + FPN + Detection Head
- 端到端训练: backbone + adapter + detector 联合优化

## Key Results

- THUMOS14: stride=1 Avg-mAP=68.97%, mAP@0.5=72.20%
- 本项目的 baseline 基准

## Reusable Ingredients

- VideoMAE-S pretrained weights
- Temporal Adapter 设计 (down_proj → dwconv → conv → up_proj, 内部残差)
- ActionFormer 检测头完整实现

## Relevance to This Project

本项目的基础框架。所有 V3 实验都基于 AdaTAD 的 backbone + adapter + ActionFormer 架构。
