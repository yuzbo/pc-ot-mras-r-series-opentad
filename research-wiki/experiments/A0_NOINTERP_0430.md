---
type: experiment
node_id: exp:A0_NOINTERP_0430
title: "A0 no-interp DenseAdapter: 消除跨轴插值bug"
config: input_random_fixed_50pct_irregular_actionformer_y_dense_grid_sanity_a0_nointerp.py
server: Server1 (24013)
status: completed
created_at: 2026-04-30
updated_at: 2026-05-01
---

# A0 No-Interp DenseAdapter

## 目标
验证消除 `linear_interpolate_features` 跨轴插值 bug 后 DenseAdapter 能否恢复。

## 方法
跳过 DenseAdapter 的 `linear_interpolate_features` 调用, IrregularFPN 特征原样传给 ActionFormerHead, 使用 selected/rank-time 语义合同。

## 结果

| Epoch | Avg-mAP | mAP@0.5 | mAP@0.7 |
|-------|---------|---------|---------|
| 39 | 48.86% | 51.90% | 22.91% |
| 44 | 49.70% | 52.62% | 24.27% |
| 49 | 50.18% | 53.59% | 24.79% |
| 59 | **51.28%** | **54.17%** | **26.21%** |

## 对比

| 版本 | Avg-mAP | diff |
|------|---------|------|
| 旧 DenseAdapter (跨轴bug) | 13.86% | — |
| B1 stride fix | 15.46% | +1.6% |
| **A0 no-interp** | **51.28%** | **+35.8%** |

## 结论
- 跨轴插值 bug 是 14% 的主因 — 消除后恢复至 51.28%
- "物理特征 → selected-axis 回归" 合同可学习
- A0 是所有 "irregular FPN → dense head" 路径的最强结果
- 仍差 dense baseline (63.12%) 约 12 个点
