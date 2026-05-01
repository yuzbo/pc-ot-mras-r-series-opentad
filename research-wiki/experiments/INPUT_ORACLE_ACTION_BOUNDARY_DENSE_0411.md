---
type: experiment
node_id: exp:INPUT_ORACLE_ACTION_BOUNDARY_DENSE_0411
title: "OpenTAD_Back 输入侧动作+边界优先稠密采样 50%"
config: configs/adatad/thumos/input_oracle_action_boundary_dense_50pct.py
server: Server 3 (25876)
date: 2026-04-11
status: completed
created_at: 2026-04-11T09:30:00+08:00
updated_at: 2026-04-11T09:30:00+08:00
---

# INPUT_ORACLE_ACTION_BOUNDARY_DENSE_0411: OpenTAD_Back 输入侧动作+边界优先稠密采样 50%

## 目的
验证“边界最高、动作内部中等、背景最低”的更强语义偏置，是否优于单纯的边界优先。

## 配置
- 仓库：`OpenTAD_Back_check`
- 配置：[input_oracle_action_boundary_dense_50pct.py](E:\DeskTop\TAD\temrefuse-tad\OpenTAD_Back\configs\adatad\thumos\input_oracle_action_boundary_dense_50pct.py)
- 采样语义：边界最高密度，动作内部中等密度，背景最低密度
- 预算：50% 输入帧
- 特征语义：全部是 fresh features，不存在 stale token / scatter 近似
- 检测头：保持原始 AdaTAD / ActionFormer dense 语义

## 结果

| Metric | First Eval | Best / Final |
|--------|------------|--------------|
| Avg-mAP | 60.73 | **62.04** |
| mAP@0.30 | - | 77.86 |
| mAP@0.40 | - | 73.30 |
| mAP@0.50 | - | 64.03 |
| mAP@0.60 | - | 53.78 |
| mAP@0.70 | - | 41.21 |

## 相对输入侧 50% 对照

| Experiment | Avg-mAP | Delta |
|------------|---------|-------|
| Uniform stride-2 50% | 65.09 | +3.05 |
| Random-fixed 50% | 63.12 | +1.08 |
| Oracle boundary-dense 50% | 66.01 | +3.97 |
| Oracle action+boundary-dense 50% | **62.04** | 0.00 |

## 结论
- “动作内部也加密”并没有继续提升，反而明显差于只做边界优先。
- 这说明在 50% 预算下，把更多采样密度分配给动作内部，会挤占边界附近最关键的时间分辨率。
- 当前最强输入侧语义采样证据支持“边界优先”而不是“动作全段优先”。

## 日志
- 远端日志：`/root/autodl-tmp/OpenTAD_Back_check/logs/input_oracle_action_boundary_dense_50pct_resume39_evalfix2_20260411.log`
