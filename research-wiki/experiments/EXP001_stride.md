---
type: experiment
node_id: exp:EXP001
title: "Stride 降采样完整训练"
date: 2026-03-30
status: completed
config: null
---

# EXP-001: Stride 降采样完整训练

## Purpose
验证不同 stride 下采样率对 TAD 性能的影响，确认时序冗余存在性。

## Setup
- Model: AdaTAD (VideoMAE-S + ActionFormer)
- Dataset: THUMOS14
- Variable: sample_stride = 1, 2, 4, 8

## Results

| Stride | Avg-mAP | mAP@0.5 | mAP@0.7 | 相对损失 |
|--------|---------|---------|---------|---------|
| 1 | 68.97% | 72.20% | 47.46% | baseline |
| 2 | 65.20% | 68.42% | 43.01% | -5.5% |
| 4 | 56.53% | 59.45% | 34.49% | -18.0% |
| 8 | 42.74% | 44.40% | 20.92% | -38.0% |

## Conclusion
✅ TAD 存在适度时序冗余。stride=1→2 仅降 5.5%，非线性下降。高 tIoU 更敏感。
