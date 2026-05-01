---
type: idea
node_id: idea:013
title: "Query-based Sparse Detector (DETR-style)"
stage: tested
outcome: testing
created_at: 2026-05-01T02:00:00+08:00
updated_at: 2026-05-01T10:00:00+08:00
---

# Query-based Sparse Detector (Plan A)

## Core Idea
替换 FCOS PointGenerator+FPN 范式为 DETR-style query detector: 可学习 query vectors 通过 Transformer decoder 直接 cross-attend 稀疏 backbone 特征, 预测 [start, end, class]。无需 FPN, 无需 PointGenerator。

## Why This Idea Exists
- 所有 FCOS 补丁方法 (Bridge/NativePhysical/DenseAdapter) 均无法超越 51%
- 核心洞察: 不规则几何信息反而破坏均匀网格假设
- 需要全新检测范式, 而非在 FCOS 上打补丁

## Architecture
```
VideoMAE → Conv1d(384→256) → Transformer Decoder(3层)
  N=30 learnable queries → Self-Attn → Cross-Attn(features) → FFN
  → cls_head(20类) + box_head([center,duration])
  → Hungarian matching loss
```

## Plan B (Physical Segment Proposal)
Parallel approach: per-point MLP + Transformer on physical coordinates [0,768), GT remap=False.
Status: queued, waiting for Plan A results.
