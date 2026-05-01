---
type: experiment
node_id: exp:R018
title: "Hard + 纯 Coarse Detection"
date: 2026-04-07
status: completed
config: v3_hard_coarse_det_only.py
---

# R018: Hard + 纯 Coarse Detection

## Purpose
去掉 BCE，只用 coarse_det 训练 scorer。

## Key Config
- mode=hard, lambda_scorer=0.0, lambda_coarse_det=1.0

## Results
- Avg-mAP = 47.68% ❌

## Conclusion
Coarse_det 单独不足以训好 scorer。信号太弱。
