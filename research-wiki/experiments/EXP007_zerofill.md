---
type: experiment
node_id: exp:EXP007
title: "V3 零填充 50%"
date: 2026-04-04
status: completed
config: v3_masked_vit_50pct.py
---

# EXP-007: V3 零填充 50%

## Purpose
V3 架构验证，dropped 位置填零。

## Results
- Avg-mAP = 47.89%, mAP@0.5 = 50.47%

## Conclusion
零填充干扰 adapter Conv1d 时序建模。与无零填充 (EXP-008: 60.68%) 差距 12.8%。
