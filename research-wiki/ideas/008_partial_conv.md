---
type: idea
node_id: idea:008
title: "Partial Conv1d for Projection"
stage: proposed
outcome: pending
created_at: 2026-04-09T13:00:00Z
updated_at: 2026-04-09T13:00:00Z
---

# Partial Conv1d

## Core Idea
用 confidence-weighted Conv1d 替代标准 Conv1d：
y[t] = Σ w[k] * x[t+k] * c[t+k] / (Σ c[t+k])
其中 c[t] = 1.0 (kept) 或 α (dropped), α 可学习。

## Target Gap
G4: Projection Conv1d 混合 kept+stale

## Config
v3_uniform_50pct_partial_conv.py

## Status
代码已实现，配置已就绪，待运行。侵入性较高，建议在 indicator_embed 和 soft_attn_bias 验证有效后再考虑。
