---
type: claim
node_id: claim:C14
title: "DenseAdapter 14% 根因是 linear_interpolate_features 中的跨轴插值 bug"
status: supported
evidence_for: [exp:CODEX_REVIEW_0430, exp:DADBUG_B1, exp:DENSEADAPTER_SANITY]
evidence_against: []
created_at: 2026-04-30T18:30:00+08:00
updated_at: 2026-04-30T18:30:00+08:00
---

# Claim C14: DenseAdapter 跨轴插值 bug

## 声明

DenseAdapter 的 14-15% 根因是 `linear_interpolate_features(feat, source_grid=physical, target_grid=selected)` 在不同坐标轴之间做 searchsorted——source center 在 physical [0,768), target center 在 selected [0,s*L), 两个坐标空间被当作同一空间处理。

## 支撑证据

1. **B0 坐标验证**: L1-L5 dense_center 范围与 PointGenerator 不匹配 (修复后对齐)
2. **B1 stride fix**: 坐标修复仅从 13.86% → 15.46% (+1.6%)
3. **Codex 全链审计**: 逐行确认为跨轴插值 bug
4. **NativePhysical=47%**: 说明 IrregularFPN 特征可用, 进一步印证 bug 在 DenseAdapter

## 正确处理方式

- A0: 跳过 interp, 保持 selected/rank-time 语义合同
- 正确 DenseAdapter: target selected → physical 映射后再插值 (`selected_to_physical(center)`)

## 待验证

A0 no-interp 实验 (running) 将最终验证此 claim
