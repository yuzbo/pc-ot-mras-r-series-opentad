---
type: idea
node_id: idea:006
title: "Oracle 50% 上界测试"
stage: proposed
outcome: pending
created_at: 2026-04-09T00:00:00Z
updated_at: 2026-04-09T12:00:00Z
---

# Oracle 50% 上界测试

## Core Idea
用 GT segments 生成 oracle mask (50%)，优先保留边界附近 tubelet，测试"完美选帧"的性能上界。

## Design
- 用 GT segments 标记边界帧，优先保留
- 保持 R008 的 backbone 设置 (hard gather/scatter, mask_adapter=True)
- 训练推理都用 oracle mask

## Target Gap
G5: 智能选帧的理论上界未知

## Why This Is Critical
- 如果 Oracle ≤ 64% (≈ uniform): 智能选帧对 TAD 无意义，整个方向需重新评估
- 如果 Oracle 65-68%: 选帧有一定价值，值得继续优化 scorer
- 如果 Oracle > 68% (接近 stride=1): 选帧有巨大价值，scorer 训练是唯一瓶颈

## Status
配置 `v3_oracle_50pct.py` 已就绪，待运行
