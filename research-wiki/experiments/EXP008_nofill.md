---
type: experiment
node_id: exp:EXP008
title: "V3 无零填充 50%"
date: 2026-04-04
status: completed
config: v3_nofill_50pct_fixed.py
---

# EXP-008: V3 无零填充 50%

## Purpose
V3 架构验证，dropped 位置保留上一层 scatter 值。

## Results
- Avg-mAP = 60.68%, mAP@0.5 = 63.33%

## Conclusion
✅ 用 50% 帧反超 V3 baseline (55.99%) +4.69%。scatter 保留上一层值是关键设计。采样可能起到正则化/去噪作用。
