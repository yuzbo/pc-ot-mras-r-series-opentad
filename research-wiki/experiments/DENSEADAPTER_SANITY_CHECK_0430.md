---
type: experiment
node_id: exp:DENSEADAPTER_SANITY
title: "IrregularFPNDenseAdapter + 标准 ActionFormerHead 诊断"
config: input_random_fixed_50pct_irregular_actionformer_y_dense_grid_sanity_check.py
server: Server1 (24013)
status: completed
created_at: 2026-04-17
updated_at: 2026-04-30
---

# DenseAdapter Sanity Check

## 背景

这是方案C的实现: IrregularFPN 在不均匀坐标上做 top-down fusion, 然后用 `IrregularFPNDenseAdapter` 把特征插值回均匀网格, 最后用标准 `ActionFormerHead` 检测。

## 配置

- **projection**: IrregularConvTransformerProj (kernel=1)
- **neck**: IrregularFPNDenseAdapter (不均匀→均匀插值)
- **rpn_head**: 标准 ActionFormerHead + PointGenerator
- **数据**: remap_gt_to_selected_axis=True (GT 在 [0, 384) feature index 坐标)

## 结果

| Epoch | Avg-mAP |
|-------|---------|
| ~ | 13.84% |
| ~ | 13.97% |
| ~ | 14.08% |
| ~ | 14.04% |
| ~ | 13.86% |

**最高 14.08%, 最终 13.86%**

## 对比

| 方法 | mAP | 差异 |
|------|-----|------|
| Dense baseline | 63.12% | — |
| Nearest completion | 54.03% | — |
| NativePhysical v2 | 47.04% | — |
| Bridge (sparse→dense head, 无FPN) | 43.44% | — |
| **DenseAdapter** | **14%** | 🔴 异常低 |

## Codex GPT-5.5 审查结论 (2026-04-30)

> 更像是坐标语义/配置错配或 adapter bug，不应先归因于 IrregularFPN 本身。

理由:
1. NativePhysical=47% 说明 IrregularFPN 特征可用
2. Bridge=43% 说明 sparse+dense head 也可用
3. 14% 指向 "不均匀特征→均匀网格→标准head" 的坐标/监督链断了

## 待排查

1. DenseAdapter 输出 center=[0,1,...,383], 但 head 的 stride/regression_range/GT remap 是否按 384 空间解释?
2. `remap_gt_to_selected_axis=True` 是否与 DenseAdapter 的均匀网格语义冲突?
3. linear_interpolate_features 从不均匀→均匀的信息损失有多大?

## 价值

如果 14% 是 bug 导致的, 修好后有望大幅提升。这是突破 47% 最可能的路径之一。
