---
type: experiment
node_id: exp:DADBUG_B1
title: "DenseAdapter B1: stride-aware center fix (方案α)"
config: input_random_fixed_50pct_irregular_actionformer_y_dense_grid_sanity_fix_alpha.py
server: Server1 (24013)
status: completed (stopped)
created_at: 2026-04-30
updated_at: 2026-04-30T18:30:00+08:00
---

# DADBUG-B1: 方案α Stride-Aware Center Fix

## 目标
修复 DenseAdapter 的 `_build_dense_reference_grid` 使 center=[0, s, 2s, ..., s*(L-1)] 与 PointGenerator 坐标空间一致。

## 结果

| Epoch | Avg-mAP | mAP@0.3 | mAP@0.5 | mAP@0.7 |
|-------|---------|---------|---------|---------|
| 39 | **15.46%** | 32.52% | 13.83% | 2.09% |

## 对比

| 版本 | Avg-mAP | diff |
|------|---------|------|
| 旧 DenseAdapter (无 fix) | 13.86% | — |
| **B1 stride fix** | **15.46%** | +1.6% |

## 根因分析

Coordinate fix 只对齐了 target grid center 与 PointGenerator，但 `linear_interpolate_features` 中的 `searchsorted(source_center=physical[0,768), target_center=selected[0,s*L))` 仍然是**跨轴插值**。

## Codex 全链审计结论

> 核心bug: `searchsorted` 在物理坐标轴中搜索 selected-axis 坐标值。source grid 是 physical [0,768), target grid 是 selected [0,s*L), 两个不同坐标空间被当作同一空间处理。这不是缩放错误，是坐标轴混用。

## 结论

方案α (stride fix) 方向正确但不够，跨轴插值需要被完全消除。催生 A0 no-interp 诊断实验。
