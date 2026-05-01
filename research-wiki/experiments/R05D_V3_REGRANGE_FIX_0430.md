---
type: experiment
node_id: exp:R05D_V3
title: "R05D v3: regression_range 修正实验"
config: input_random_fixed_50pct_native_physical_multiscale_full_pilot_v3.py
server: Server1 (24013)
status: completed (stopped early)
created_at: 2026-04-30
updated_at: 2026-04-30T14:30:00+08:00
---

# R05D v3: Regression Range 修正

## 目标

测试 regression_range 从 `[(0,4),(1,4),...,(1,12)]` 修改为 `[(0,5),(0,5),...,(0,6)]`（所有层 reg_min=0）是否改善 v2 的 L3-L5 正样本不足问题。

## 改动

唯一变量: `regression_range=[(0,5),(0,5),(0,5),(0,5),(0,5),(0,6)]`

修改理由: v2 中 L5 reg_min = 1 * 128 = 128, 过滤掉了所有短于 256 物理单位的 GT。

## 最终结果 (epoch 44, 训练停止)

| Epoch | Avg-mAP | mAP@0.3 | mAP@0.5 | mAP@0.7 |
|-------|---------|---------|---------|---------|
| 34 | 43.82% | — | — | — |
| 39 | 44.99% | 67.79% | 47.12% | 17.55% |
| 44 | **45.80%** | 68.29% | 48.45% | 18.49% |

## 结论

❌ regression_range 修改未改善性能。v3 @ epoch 44 = 45.80% **低于** v2 @ epoch 44 = 45.82%。

L5 获得正样本 (6-10个) 但整体 mAP 反而下降, 说明问题不在 regression_range。

**按 Codex result-to-claim 裁决 (2026-04-30)**: NativePhysicalMultiScaleHead 路线正式关闭。v3 提前停止。
