---
type: idea
node_id: idea:012
title: "Incremental Irregular Detector: 从 dense baseline 逐步适配"
stage: proposed
outcome: pending
created_at: 2026-04-12T23:00:00+08:00
updated_at: 2026-04-12T23:00:00+08:00
---

# Incremental Irregular Detector

## Core Idea
不再一次性重建整个 irregular detector，而是从 dense ActionFormer + random-fixed = 63.12 出发，逐步替换单个模块（head conv → assignment → CKConv → PE → FPN），每步独立验证。

## 与 idea:011 的区别
- idea:011 是"一次性全改"策略，结果 mAP 1-4，结构性失败
- idea:012 是"增量适配"策略，每次只改一个模块，保持其他模块为 dense 原版

## 关键技术方案

### Phase 1: k=1 Head
- 去掉 head 中所有 temporal conv (k=3 → k=1)
- 消除 irregular spacing 在 head 中的影响
- 最简单的改动，用于定位瓶颈

### Phase 2: Voronoi Assignment
- 每个 kept token 拥有 Voronoi cell [midpoint(t_{i-1}, t_i), midpoint(t_i, t_{i+1})]
- GT 按 cell 重叠分配，替代 level-based hard regression range
- 几何上最自然的 assignment 方式

### Phase 3: CKConv Head
- 连续核卷积：K(Δt) = MLP(Δt)
- 天然适配非均匀时间网格
- 文献：CKConv (Romero et al., ICLR 2022)

### Phase 4-5: Time PE + Irregular FPN
- 连续正弦 PE 编码真实时间戳
- FPN top-down 用时间坐标感知插值

## 成功条件
- Phase 1 > 50 mAP（确认 head conv 是瓶颈）
- 最终组合 > 63.12（超过 dense detector on random-fixed）
- 理想目标 > 65.09（接近 uniform stride-2）

## 失败条件
- Phase 1-5 全部完成后仍 < 63.12
- 此时结论：irregular-aware 改造对 ActionFormer 类 detector 无价值

## 详细计划
见 `IRREGULAR_DETECTOR_PLAN.md`

## Target Gaps
- G4: dense ActionFormer prior 与 irregular timeline 失配

## Related
- idea:011 (前序，一次性全改，已失败)
- exp:INPUT_RANDOMFIX_0410 (baseline = 63.12)
- exp:IRREGULAR_TIMELINE_0412 (idea:011 的实验结果)
- claim:C11

## Connections
AUTO-GENERATED from graph/edges.jsonl - do not edit manually
