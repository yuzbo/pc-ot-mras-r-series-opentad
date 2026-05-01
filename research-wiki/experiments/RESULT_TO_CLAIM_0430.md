---
type: experiment
node_id: exp:RESULT_TO_CLAIM_0430
title: "Result-to-Claim 裁决: NativePhysical 路线关闭, DenseAdapter bug 优先"
status: completed
created_at: 2026-04-30T14:30:00+08:00
updated_at: 2026-04-30T14:30:00+08:00
---

# Result-to-Claim 裁决 (2026-04-30)

## 裁决来源

Codex GPT-5.4 (xhigh reasoning), 基于 R05D_V2/V3 + Codex 审查 + DenseAdapter sanity 完整证据链.

## 裁决结果

### C12: NativePhysicalMultiScaleHead 结构性差距

- **Verdict**: `partial`
- **Revision**: 弱化为 "剩余的 NativePhysical 差距不能由 `regression_range` 解释, 且与 FCOS 风格不规则物理轴的定位不匹配一致"
- **理由**: R05D_V3 排除了 regression_range 假说, 但尚未通过直接干预实验证明 "共享 kernel=3" 是确切因果

### C13: DenseAdapter 14% 是 Bug

- **Verdict**: `yes` (narrowed)
- **Revision**: "14% 高度可能是坐标/配置/监督链 bug, 而非 IrregularFPN 特征质量导致的"
- **理由**: NativePhysical=47% 说明 IrregularFPN 可用, Bridge=43% 说明 sparse+dense head 可用; 14% 远低于两者

## 路线裁决

1. ✅ **正式关闭 NativePhysicalMultiScaleHead 主线** — 停止 v3 训练
2. ✅ **跳过 Candidate A M1 overfit** — 不解决已观察到的架构天花板
3. ✅ **最高优先级: 排查 DenseAdapter 14% bug**
4. ⏳ **备选: Query-based detector / deformable temporal sampling (Candidate B)** — DenseAdapter 修不好时启用

## DenseAdapter Bug 排查步骤

1. 检查 point timestamps, strides, masks, regression targets 与 dense ActionFormer 对比
2. 单视频/单实例 DenseAdapter + ActionFormerHead overfit
3. Identity/nearest-completion adapter sanity: 喂 dense 形状特征通过同一 head path
4. 修好后目标: ~52-54%
5. 仍不行 → Candidate B (query/deformable detector)

## 相关 Claims

- `claim:C12` → `partial`
- `claim:C13` → `supported`
