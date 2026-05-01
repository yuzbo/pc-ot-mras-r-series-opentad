---
type: experiment
node_id: exp:R012
title: "Gumbel + Soft Gating (纯 E2E)"
date: 2026-04-06
status: completed
config: v3_gumbel_softgate_50pct.py
---

# R012: Gumbel + Soft Gating (纯端到端)

## Purpose
验证 DynamicViT-style soft gating 能否端到端训练 scorer。

## Key Config
- mode=gumbel, soft_gating=True, lambda_scorer=0.0

## Results
- Avg-mAP ≈ 52-55% ❌ (远低于 R008 的 63.74%)

## Conclusion
Soft gating 破坏 backbone 特征分布，train-test gap 致命。
