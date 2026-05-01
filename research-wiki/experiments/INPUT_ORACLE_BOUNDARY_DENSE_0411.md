---
type: experiment
node_id: exp:INPUT_ORACLE_BOUNDARY_DENSE_0411
title: "OpenTAD_Back 输入侧边界优先稠密采样 50%"
config: configs/adatad/thumos/input_oracle_boundary_dense_50pct.py
server: Server 1 (24013)
date: 2026-04-11
status: completed
created_at: 2026-04-11T09:30:00+08:00
updated_at: 2026-04-11T09:30:00+08:00
---

# INPUT_ORACLE_BOUNDARY_DENSE_0411: OpenTAD_Back 输入侧边界优先稠密采样 50%

## 目的
验证在 50% 输入预算下，只把更高采样密度分配到动作边界附近，是否能优于 uniform 50%。

## 配置
- 仓库：`OpenTAD_Back_check`
- 配置：[input_oracle_boundary_dense_50pct.py](E:\DeskTop\TAD\temrefuse-tad\OpenTAD_Back\configs\adatad\thumos\input_oracle_boundary_dense_50pct.py)
- 采样语义：边界附近高密度，远背景低密度
- 预算：50% 输入帧
- 特征语义：全部是 fresh features，不存在 stale token / scatter 近似
- 检测头：保持原始 AdaTAD / ActionFormer dense 语义

## 结果

| Metric | First Eval | Best / Final |
|--------|------------|--------------|
| Avg-mAP | 64.51 | **66.01** |
| mAP@0.30 | - | 77.19 |
| mAP@0.40 | - | 73.34 |
| mAP@0.50 | - | 69.42 |
| mAP@0.60 | - | 60.39 |
| mAP@0.70 | - | 49.74 |

## 相对输入侧 50% 对照

| Experiment | Avg-mAP | Delta |
|------------|---------|-------|
| Uniform stride-2 50% | 65.09 | -0.92 |
| Random-fixed 50% | 63.12 | +2.89 |
| Oracle boundary-dense 50% | **66.01** | 0.00 |

## 结论
- 这条结果直接否定了“非均匀采样天然更差”。
- 在输入侧、fresh-feature、无 stale token 干扰的前提下，边界优先的 50% 非均匀采样可以超过 uniform 50%。
- 当前最强的输入侧证据支持：对 TAD 来说，边界附近的时间分辨率比动作内部更值得保留。

## 日志
- 远端日志：`/root/autodl-tmp/OpenTAD_Back_check/logs/input_oracle_boundary_dense_50pct_resume39_evalfix2_20260411.log`
- checkpoint：`/root/autodl-tmp/OpenTAD_Back_check/exps/thumos/adatad/input_oracle_boundary_dense_50pct/gpu1_id0/checkpoint/epoch_59.pth`
