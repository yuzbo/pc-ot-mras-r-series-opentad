---
type: experiment
node_id: exp:EXP005
title: "V3 Baseline (keep_ratio=1.0)"
date: 2026-04-04
status: completed
config: baseline_no_sampling.py
---

# EXP-005: V3 Baseline (keep_ratio=1.0)

## Purpose
建立 V3 框架下不采样的对比基准。

## Key Config
- keep_ratio=1.0, 60 epochs, V3 框架 (含 adapter 等组件)

## Results
- Avg-mAP = 55.99%, mAP@0.5 = 58.72%

## Conclusion
低于 AdaTAD stride=1 (68.97%)，因为 V3 框架引入了额外组件，训练设置不同。后续所有 V3 实验以此为基准。
