---
type: experiment
node_id: exp:PROJ_PARTIAL_CONV
title: "Projection Partial Conv"
date: 2026-04-09
status: prepared
config: v3_uniform_50pct_partial_conv.py
---

# Projection Partial Conv

## Purpose
用 confidence-aware / partial convolution 的思路降低 dropped token 对局部 1D Conv 的污染，显式处理 K-D-K 模式下的局部邻域失配。

## Status
- 配置已准备
- 结果尚未正式写回

## Risk
实现复杂度更高，也更容易引入效率开销，因此优先级低于 indicator / soft bias。
