---
type: claim
node_id: claim:C13
title: "DenseAdapter 14% 是坐标/配置 bug, 不是 IrregularFPN 特征质量问题"
status: supported
evidence_for: [exp:DENSEADAPTER_SANITY, exp:CODEX_REVIEW_0430, exp:RESULT_TO_CLAIM_0430]
evidence_against: []
created_at: 2026-04-30
updated_at: 2026-04-30T14:30:00+08:00
---

# Claim C13: DenseAdapter 14% 是 Bug

## 声明

IrregularFPNDenseAdapter + 标准 ActionFormerHead 只有 14% 的原因是坐标语义/配置错配 bug, 不是 IrregularFPN 本身产生低质量特征。

## 支撑证据

1. NativePhysical = 47% 说明 IrregularFPN 特征可用
2. Bridge = 43% 说明 sparse + dense head 也可用
3. 14% 远低于两者, 暗示坐标/监督链断了
4. Codex 审查: "更像是坐标语义/配置错配或 adapter bug"

## 待验证

1. DenseAdapter 输出的 center 坐标与 head 的 PointGenerator stride 是否匹配
2. `remap_gt_to_selected_axis=True` 是否与 DenseAdapter 均匀网格语义冲突
3. 需要对照实验: 不用 DenseAdapter, 直接把 IrregularFPN 输出喂给标准 head (用 remap)

## 影响

如果此 claim 被证实, 修好 bug 后 DenseAdapter 方案有望大幅提升, 可能接近 54%。这是当前最有价值的排查方向。
