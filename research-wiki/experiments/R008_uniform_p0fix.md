---
type: experiment
node_id: exp:R008
title: "V3 Uniform 50% P0fix"
date: 2026-04-06
status: completed
config: v3_uniform_50pct_p0fix.py
---

# R008: V3 Uniform 50% P0fix ⭐ 当前最佳

## Purpose
修复 P0 级 bug 后的 uniform 采样基准。

## Key Config
- mode=uniform, keep_ratio=0.5
- use_attn_scale=False, mask_adapter=True
- soft_gating=False (训练推理都用 hard)

## Results
- **Avg-mAP = 63.74%, mAP@0.5 = 66.72%**
- 训练完全稳定，零 NaN

## Significance
所有 learned scorer 实验必须超过此值才有意义。
