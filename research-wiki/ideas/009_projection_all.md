---
type: idea
node_id: idea:009
title: "Projection All-in-One (indicator + soft_attn_bias + partial_conv)"
stage: proposed
outcome: pending
created_at: 2026-04-09T13:00:00Z
updated_at: 2026-04-09T13:00:00Z
---

# Projection All-in-One

## Core Idea
同时启用 indicator_embed + soft_attn_bias + partial_conv 三层 Projection 稀疏适配。

## Config
- v3_uniform_50pct_projection_all.py
- v3_hard_scorer_aux_projection_all.py

## Target Gap
G4: Projection 不感知稀疏性

## Risk
三层同时改动，无法归因。建议先逐个消融，确认各自贡献后再组合。
