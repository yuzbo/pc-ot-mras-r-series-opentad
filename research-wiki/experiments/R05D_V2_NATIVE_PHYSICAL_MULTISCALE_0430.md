---
type: experiment
node_id: exp:R05D_V2
title: "R05D v2: NativePhysicalMultiScaleHead full THUMOS14 pilot"
config: input_random_fixed_50pct_native_physical_multiscale_full_pilot.py
server: Server1 (24013)
status: completed
created_at: 2026-04-29
updated_at: 2026-04-30
---

# R05D v2: NativePhysicalMultiScaleHead Full Pilot

## 目标

验证 Candidate A (NativePhysicalMultiScaleHead) 在 full THUMOS14 上的表现，目标超过 nearest completion baseline (54.03%)。

## 配置

- **架构**: sparse 384帧 → IrregularActionFormerProjection (6级FPN) → IrregularFPN (top-down) → NativePhysicalMultiScaleHead
- **物理坐标**: remap_gt_to_selected_axis=False, GT 在 [0, 768)
- **regression_range**: [(0,4), (1,4), (1,4), (1,4), (1,4), (1,12)]
- **strides**: [1, 2, 4, 8, 16, 32]
- **loss**: FocalLoss + DIOULoss
- **训练**: 60 epochs, val_start_epoch=30, eval_interval=5

## 结果

| Epoch | Avg-mAP | mAP@0.3 | mAP@0.5 | mAP@0.7 |
|-------|---------|---------|---------|---------|
| 35 | 44.16% | 66.91% | 46.59% | 16.36% |
| 39 | 45.21% | — | — | — |
| 44 | 45.82% | — | — | — |
| 54 | 46.58% | — | — | — |
| 59 | **47.04%** | — | — | — |

## Per-Level Positive Count 分析

| Level | stride | pos_total 范围 | 状态 |
|-------|--------|---------------|------|
| L0 | 1 | 6-199 | ✅ |
| L1 | 2 | 15-190 | ✅ |
| L2 | 4 | 7-51 | ✅ |
| L3 | 8 | 0-50 | ⚠️ |
| L4 | 16 | 0-26 | 🔴 |
| L5 | 32 | **0** | 🔴 全部为0 |

L5 无正样本原因: reg_min = 1 * point_scale(≈128) = 128, 大多数 GT 的 max(left,right) < 128。

## 关键观察

1. **分类强、定位差**: mAP@0.3=66.91% vs mAP@0.7=16.36%，regression 精度是主要瓶颈
2. **收敛趋势放缓**: epoch 35→59 只提升 +2.88%，增速递减
3. **未达目标**: 47.04% 远低于 nearest completion 54.03%，差距 6.99%

## 结论

NativePhysicalMultiScaleHead 在 full THUMOS14 上比 Bridge (43.44%) 好 3.6%，但远低于 nearest completion (54.03%)。问题不在 regression_range（v3 修改后未改善），而是更深层的结构性问题。
