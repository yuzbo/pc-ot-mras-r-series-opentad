---
type: experiment
node_id: exp:INDICATOR
title: "Hard + scorer + indicator embedding"
config: v3_hard_scorer_aux_indicator.py
server: Server 1
date: 2026-04-09
status: completed
created_at: 2026-04-09T17:00:00Z
updated_at: 2026-04-09T17:00:00Z
---

# Indicator Embed: Projection 稀疏感知

## 目的
在 Projection 输入加 Embedding(2, C) 区分 kept/dropped 位置。

## 结果
| Metric | Value |
|--------|-------|
| Avg-mAP | 56.13% |
| mAP@0.5 | 59.51% |
| mAP@0.7 | 29.93% |

## 分析
- 低于 R017 (57.55%)，indicator embed 没有带来提升
- 可能原因：在 hard+scorer 基础上加 indicator，scorer 本身选帧差，embed 帮助有限
- 需要在 uniform 底座上单独验证 indicator 的贡献
