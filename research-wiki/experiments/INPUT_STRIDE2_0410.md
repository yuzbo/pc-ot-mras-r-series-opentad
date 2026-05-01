---
type: experiment
node_id: exp:INPUT_STRIDE2_0410
title: "OpenTAD_Back 输入侧等间隔 50% 基线"
config: configs/adatad/thumos/input_stride2_uniform_baseline.py
server: Server 1 (24013)
date: 2026-04-10
status: completed
created_at: 2026-04-10T18:38:29+08:00
updated_at: 2026-04-10T18:38:29+08:00
---

# INPUT_STRIDE2_0410: OpenTAD_Back 输入侧等间隔 50% 基线

## 目的
用最干净的输入侧均匀 50% 抽帧，验证：

1. 在不引入非规则时间几何的前提下，50% 输入预算下的检测性能基线是多少。
2. 后续随机/语义偏置输入侧采样应该与哪个基线对比。

## 配置
- 仓库：`OpenTAD_Back_check`
- 配置：[input_stride2_uniform_baseline.py](E:\DeskTop\TAD\temrefuse-tad\OpenTAD_Back\configs\adatad\thumos\input_stride2_uniform_baseline.py)
- 输入侧采样：规则等间隔，`768 -> 384`
- 特征语义：全部是 fresh features，不存在 stale token / scatter 近似
- 检测头：保持 AdaTAD / ActionFormer 原始 dense 语义

## 结果

| Metric | Value |
|--------|-------|
| Avg-mAP | **65.09** |
| mAP@0.30 | 80.94 |
| mAP@0.40 | 76.41 |
| mAP@0.50 | 68.71 |
| mAP@0.60 | 57.38 |
| mAP@0.70 | 42.00 |

## 结论
- 50% 输入预算下，规则均匀输入侧抽帧仍能保持较高性能。
- 这条实验是后续所有输入侧 irregular / oracle / weighted-random 对照的主基线。
- 它说明“减少输入帧数”本身不是灾难性操作；真正更难的是如何在减少输入的同时不破坏时间几何。

## 日志
- 远端日志：`/root/autodl-tmp/OpenTAD/logs/stride2_uniform_input_baseline_fix0410.log`

