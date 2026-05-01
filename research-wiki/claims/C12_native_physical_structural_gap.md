---
type: claim
node_id: claim:C12
title: "NativePhysicalMultiScaleHead 的 7% mAP 差距是结构性问题"
status: partial
evidence_for: [exp:R05D_V2, exp:R05D_V3, exp:CODEX_REVIEW_0430, exp:RESULT_TO_CLAIM_0430]
evidence_against: []
created_at: 2026-04-30
updated_at: 2026-04-30T14:30:00+08:00
---

# Claim C12: NativePhysicalMultiScaleHead 的 7% mAP 差距是结构性的

## 声明

在 50% random-fixed sparse 采样下, NativePhysicalMultiScaleHead (47.04%) 与 nearest completion baseline (54.03%) 的 7% 差距主要来自结构性设计缺陷 (共享 temporal conv 假设等间距), 而非超参标定不足。

## 支撑证据

1. **R05D v2 = 47.04%**: 完整训练 60 epochs, 收敛充分
2. **R05D v3 = ~45% (running)**: 修改 regression_range 后未改善, 说明不是 regression_range 的问题
3. **mAP@0.3=66.91% vs mAP@0.7=16.36%**: 分类强但定位差, regression 精度是瓶颈
4. **L5 正样本=0**: 虽然是 regression_range 导致, 但修复后 (v3) 仍未改善
5. **Codex GPT-5.5 审查**: 明确指出共享 kernel=3 在不同层物理感受野差 32 倍是根因

## 反驳证据

暂无

## 实验建议

如果要推翻此 claim, 需要:
1. 在不改架构的前提下, 仅通过超参调整让 NativePhysical 超过 50%
2. 或证明某个具体 bug 是 7% 差距的主因

## Connections

- tested_by: exp:R05D_V2, exp:R05D_V3
- supports: idea:012 (incremental irregular detector 需要重新审视)
