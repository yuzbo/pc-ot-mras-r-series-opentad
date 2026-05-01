---
type: experiment
node_id: exp:PROJ_INDICATOR
title: "Projection Indicator Embed"
date: 2026-04-09
status: running
config: v3_uniform_50pct_indicator_embed.py
---

# Projection Indicator Embed

## Purpose
给 projection 前的 token 显式标记 kept / dropped 身份，让 projection 知道哪些位置是 stale / unreliable，而不是把 dropped 特征完全当正常 dense token 处理。

## Status
- 配置已落地
- 正在作为 projection-aware 优化候选线推进

## Hypothesis
若 indicator 有效，应该在不破坏时序连续性的前提下，优于完全不标记 dropped token 的处理方式。
