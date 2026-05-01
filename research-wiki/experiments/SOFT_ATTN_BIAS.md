---
type: experiment
node_id: exp:SOFT_ATTN_BIAS
title: "Hard + scorer + soft attention bias"
config: v3_hard_scorer_aux_soft_attn_bias.py
server: Server 1
date: 2026-04-09
status: running
created_at: 2026-04-09T17:00:00Z
updated_at: 2026-04-09T17:00:00Z
---

# Soft Attn Bias: Projection Attention 稀疏偏置

## 目的
在 Projection 的 Attention 中对 dropped 位置加可学习负偏置。

## 结果 (截至 Epoch ~52，未完成)
| Metric | Value |
|--------|-------|
| Avg-mAP | 57.25% (持续上升中) |
| mAP@0.5 | 60.67% |
| mAP@0.7 | 31.24% |

## 分析
- 与 R017 (57.55%) 接近，但仍在训练中
- 比 indicator_embed (56.13%) 好，说明 soft_attn_bias 方向更有效
- 但仍远低于 R015 uniform (64.51%)
