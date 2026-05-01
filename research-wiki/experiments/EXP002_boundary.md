---
type: experiment
node_id: exp:EXP002
title: "边界感知采样诊断"
date: 2026-03-30
status: completed
config: null
---

# EXP-002: 边界感知采样诊断

## Purpose
验证"stride=2 错过边界"的假设。

## Results
- 边界命中率: stride=1 100%, stride=2 100% (假设被推翻)
- 平均距离: 0.033s → 0.066s (翻倍)
- 边界覆盖率: 8.63%

## Conclusion
❌ 原假设被推翻。stride=2 并未错过边界，性能下降源于整体时序分辨率降低。
