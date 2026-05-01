---
type: idea
node_id: idea:011
title: "Irregular-Aware TAD Detector"
stage: tested
outcome: negative
created_at: 2026-04-11T12:30:00+08:00
updated_at: 2026-04-12T22:20:00+08:00
---

# Irregular-Aware TAD Detector

## Core Idea
不再把 irregular timeline 硬塞进原始 dense ActionFormer。新建并行 detector 分支，把 projection、pyramid、point generator、assign、regression target、decode 全部改成显式基于真实时间坐标与局部 cell 几何。

## Target Gaps
- G3: dropped / stale positions 在 projection / adapter 前后的语义不清晰
- G4: dense ActionFormer prior 与 irregular timeline 失配

## Why This Idea Exists
- `INPUT_RANDOMFIX_0410 = 63.12` 说明非均匀采样本身不是灾难性失败
- `INPUT_ORACLE_BOUNDARY_DENSE_0411 = 66.01 > 65.09` 说明边界优先的非均匀采样可以有效
- 因此当前主问题更像 detector 时间语义不匹配，而不是“智能选帧 TAD 天然不成立”

## Explicit Review: What We Must Not Repeat
- 不再把 soft gating / REINFORCE / ST-topk 当作主突破口
- 不再依赖 dropped zero-fill 或 `mask_adapter=True`
- 不再把 compressed / compact timeline 直接当成 detector 时间轴
- 不再用旧 sparse attention / kept-q 结构
- 不再把 indicator / soft bias / partial conv 当作主修复，而只作为辅助增强

## Minimal Design Commitments
- 第一阶段只做 fresh-feature detector-only 验证，不接 V3 stale-token
- 新 detector 保持单阶段 anchor-free，不引入 DETR set head
- 统一 `temporal_grid` 接口：
  - `center`
  - `cell_left`
  - `cell_right`
  - `valid_mask`
  - `fresh_mask`
  - `level_scale`
- assign / regression / decode 全部在原时间轴定义

## Validation Plan
- `input_stride2_uniform_baseline`
- `input_random_fixed_50pct`
- `input_oracle_boundary_dense_50pct`

成功条件：
- random-fixed 相对 uniform 的掉点明显缩小
- boundary-dense 不被新 detector 破坏

失败解释：
- 若 fresh-feature 下仍几乎无改善，则“只改 detector 语义”不足以解释当前失败
- 若 boundary-dense 也被破坏，则说明新 detector 实现本身不自洽

## Related Existing Evidence
- exp:INPUT_STRIDE2_0410
- exp:INPUT_RANDOMFIX_0410
- exp:INPUT_ORACLE_BOUNDARY_DENSE_0411
- exp:IRREGULAR_TIMELINE_0412
- claim:C11

## Connections
AUTO-GENERATED from graph/edges.jsonl - do not edit manually
