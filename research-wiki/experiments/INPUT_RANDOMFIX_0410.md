---
type: experiment
node_id: exp:INPUT_RANDOMFIX_0410
title: "OpenTAD_Back 输入侧随机固定 50% 对照"
config: configs/adatad/thumos/input_random_fixed_50pct.py
server: Server 2 (25876)
date: 2026-04-10
status: completed
created_at: 2026-04-10T18:38:29+08:00
updated_at: 2026-04-10T18:38:29+08:00
---

# INPUT_RANDOMFIX_0410: OpenTAD_Back 输入侧随机固定 50% 对照

## 目的
验证“输入侧非均匀随机采样”是否天然不可用，重点回答：

1. 非均匀采样是否必然导致性能灾难性下降。
2. 如果下降，原因更像 coverage 崩塌，还是时间几何与 dense detector prior 的失配。

## 配置
- 仓库：`OpenTAD_Back_check`
- 配置：[input_random_fixed_50pct.py](E:\DeskTop\TAD\temrefuse-tad\OpenTAD_Back\configs\adatad\thumos\input_random_fixed_50pct.py)
- 输入侧采样：随机但固定，`768 -> 384`
- 随机种子：由 `video_name + dense_window range + valid_len + frame_num` 稳定构造
- 特征语义：全部是 fresh features，不存在 stale token / scatter 近似
- 检测头：保持 AdaTAD / ActionFormer 原始 dense 语义

## 结果

| Metric | Value |
|--------|-------|
| Avg-mAP | **63.12** |
| mAP@0.30 | 80.06 |
| mAP@0.40 | 74.36 |
| mAP@0.50 | 66.61 |
| mAP@0.60 | 54.79 |
| mAP@0.70 | 39.78 |

## 相对均匀基线

| Metric | Random-fixed 50% | Uniform stride-2 50% | Delta |
|--------|------------------|----------------------|-------|
| Avg-mAP | 63.12 | 65.09 | -1.97 |
| mAP@0.50 | 66.61 | 68.71 | -2.10 |
| mAP@0.70 | 39.78 | 42.00 | -2.22 |

## 采样分布观察
- 可视化文件：[fig_input_random_fixed_distribution.png](E:\DeskTop\TAD\temrefuse-tad\figures\fig_input_random_fixed_distribution.png)
- 逐位置保留概率均值：`0.5000`
- 最小 / 最大逐位置保留概率：`0.4443 / 0.5518`
- 四个 quarter 的平均保留帧数：`96.20 / 95.87 / 95.60 / 96.33`
- 最长连续 dropped run：中位数 `8`，95 分位 `12`

## 结论
- 非均匀随机输入侧采样不是灾难性失败，只比均匀 50% 低 `1.97 mAP`。
- 它的覆盖率仍然很高，并没有塌缩成只看局部。
- 因此它掉点更像是“时间几何不规则”导致的 Projection / FPN / point generator / regression 语义失配，而不是“随机采样漏看了大段视频”。

## 日志
- 远端日志：`/root/autodl-tmp/OpenTAD_retry/OpenTAD/logs/input_random_fixed_50pct_fix0410.log`

