---
type: experiment
node_id: exp:R022_MOBILENET
title: "MobileNet scorer + temporal conv + coarse inject"
config: v3_mobilenet_temporal_coarse_inject.py
server: Server 2
date: 2026-04-09
status: completed
created_at: 2026-04-09T17:00:00Z
updated_at: 2026-04-09T17:00:00Z
---

# R022-mobilenet: MobileNet Scorer + Coarse Inject

## 目的
用预训练 MobileNetV3-Small 替代轻量 CNN scorer，加时序 conv 和 coarse inject。

## 结果
| Metric | Value |
|--------|-------|
| Avg-mAP | 58.49% |
| mAP@0.5 | 61.94% |
| mAP@0.7 | 33.93% |

## 分析
- 所有 learned scorer 实验中**最高**的 58.49%
- 但仍远低于 uniform R015 (64.51%)
- MobileNet 预训练特征比随机 CNN 好，但根本问题 (scorer 学不到有效策略) 未解决
