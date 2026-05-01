---
type: experiment
node_id: exp:EXP006
title: "V2 架构全路线 (NaN 崩溃)"
date: 2026-04-03
status: completed
config: multiple (compressed/bridge/sparse_head/hybrid/glance-focus)
---

# EXP-006: V2 架构全路线

## Purpose
验证 V2 (q=kept, k/v=full) 架构。

## Attempted Routes
- compressed, bridge, sparse_head, hybrid, glance-focus

## Results
- 全线 NaN 崩溃 ❌

## Root Cause (Codex confirmed)
1. 结构失配: kept q + full k/v 破坏 softmax 质量守恒
2. Coarse cls 冷启动脆弱: cls_head.bias=-4.595 使正样本 logit 系统性过负
3. AMP fp16 放大热点通道溢出

## Conclusion
V2 架构存在根本性设计缺陷 → 催生 V3 (q=k=v=kept)
