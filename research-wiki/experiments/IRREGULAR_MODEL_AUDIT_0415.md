---
type: experiment
node_id: exp:IRREGULAR_MODEL_AUDIT_0415
title: "Irregular ActionFormer model-line audit: trusted runs, invalid runs, deployed runs, and missing deployments"
date: 2026-04-15
status: audited
created_at: 2026-04-15T10:37:32+08:00
updated_at: 2026-04-16T16:55:00+08:00
---

# IRREGULAR_MODEL_AUDIT_0415

## 2026-04-16 upstream audit correction

This inventory page remains useful as a run-status ledger, but two entries that were previously treated as trustworthy controls should now be read as attribution-unsafe:

- `input_random_fixed_50pct_irregular_actionformer_step0_dense_points_hard_shell_exact = 53.72`
  - invalid as a shell-exact baseline because the config inherits `remap_gt_to_selected_axis=True`
- `input_random_fixed_50pct_irregular_actionformer_headv3_oabs_full_x_fixsingleton = 52.88`
  - invalid as a clean singleton-fix ablation because the config only changed `work_dir`

One more semantic correction also applies:

- the `x` line in this repo is currently a **dense-trunk + irregular-head** carrier, not evidence for a fully irregular-aware trunk

For the corrected code-level interpretation, use:

- `exp:IRREGULAR_UPSTREAM_AUDIT_0416`

## Scope

本页只审计 `OpenTAD_Back/configs/adatad/thumos/` 下 `input_random_fixed_50pct_irregular_actionformer*.py` 这一整条模型实验线。

判定规则：

- `Training Over` 且与当前本地配置一一对应：记为“完整跑完，结果可信”
- 只有 `Test INFO` 的 eval-only 日志，但 checkpoint / config 对应明确：记为“可信的推理消融”
- 中途终止、控制语义错误、调试任务、或仅有 scratch / duplicate log：记为“不可信 / 不纳入正式比较”
- 已同步到服务器并进入等待队列，但尚未开始或尚未结束：记为“已部署未完成”
- 本地存在 runnable config，但服务器上没有对应部署痕迹：记为“未部署”

## Snapshot

- 完整跑完且可信的训练实验：**17**
- 可信但仅 eval-only 的推理消融：**2**
- 已跑过但不可信 / 不应纳入正式对照：**4** 个具名条目，外加若干 duplicate / scratch logs
- 已部署但还没有完整结果：**4**
- 本地存在但尚未部署：**1**

当前模型主线中，最可信的 strongest result 仍然是：

- `headv3_x = 52.60`

当前最重要的 unfinished items 是：

- `headv3_oabs_visible_x`
- `headv3_oabs_full_x`
- `step0_dense_points_hard_shell_exact`
- `step0b_dense_points_soft_sym`

## 2026-04-16 Addendum

This audit snapshot was created on 2026-04-15 and is no longer the latest summary of the line.

New completed trusted runs after this snapshot:

- `input_random_fixed_50pct_irregular_actionformer_headv3_oabs_visible_x = 51.49`
- `input_random_fixed_50pct_irregular_actionformer_headv3_oabs_full_x = 53.16`
- `input_random_fixed_50pct_irregular_actionformer_headv3_oabs_full_oaa_x = 53.20`
- `input_random_fixed_50pct_irregular_actionformer_headv3_oabs_full_x_fixsingleton = 52.88`
- `input_random_fixed_50pct_irregular_actionformer_step0_dense_points_hard_shell_exact = 53.72`
- `input_random_fixed_50pct_irregular_actionformer_step0b_dense_points_soft_sym = 47.52`

As of 2026-04-16:

- the strongest trusted irregular-shell result is `headv3_oabs_full_oaa_x = 53.20`
- the major newly closed diagnosis is `step0b = 47.52`, which shows soft assignment is already harmful under dense points
- the current authoritative structural reading has moved to `exp:IRREGULAR_ROOTCAUSE_UPDATE_0416`

## A. 完整跑完且结果可信

| Experiment | Server | Avg-mAP | Status | Why trusted |
|---|---:|---:|---|---|
| `input_random_fixed_50pct_irregular_actionformer` | 25876 | **39.04** | done | 审计后的旧 irregular detector 负结果，日志完整 |
| `input_random_fixed_50pct_irregular_actionformer_remap_no_regrange` | 25876 | **3.92** | done | `Training Over`，native irregular semantic collapse |
| `input_random_fixed_50pct_irregular_actionformer_remap_overlap` | 25876 | **2.12** | done | `Training Over`，native irregular semantic collapse |
| `input_random_fixed_50pct_irregular_actionformer_headv2_x` | 24013 | **51.04** | done | 使用最终成功日志 `0037` |
| `input_random_fixed_50pct_irregular_actionformer_headv2_y` | 25876 | **49.00** | done | resume 后完整结束 |
| `input_random_fixed_50pct_irregular_actionformer_headv2_denseexact` | 25876 | **51.04** | done | fair dense passthrough control，关键 C 类残差对照 |
| `input_random_fixed_50pct_irregular_actionformer_headv2_x_topk1` | 25876 | **47.19** | done | 使用最终成功日志 `173856` |
| `input_random_fixed_50pct_irregular_actionformer_headv2_x_k3` | 24013 | **49.53** | done | final predictor `k=3` ablation |
| `input_random_fixed_50pct_irregular_actionformer_headv3_x` | 24013 | **52.60** | done | 当前最强完整训练结果 |
| `input_random_fixed_50pct_irregular_actionformer_headv3_y` | 25876 | **49.37** | done | Y 线完整训练 |
| `input_random_fixed_50pct_irregular_actionformer_headv3_timeembed_x` | 24013 | **50.39** | done | backbone time-embed X |
| `input_random_fixed_50pct_irregular_actionformer_headv3_timeembed_y` | 25876 | **47.06** | done | backbone time-embed Y |
| `input_random_fixed_50pct_irregular_actionformer_step0_densehead_remap` | 24013 | **53.72** | done | remapped pseudo-dense control，bridge Step0_fixed |
| `input_random_fixed_50pct_irregular_actionformer_step1_irregular_points_hard_sym` | 25876 | **37.10** | done | bridge Step1 |
| `input_random_fixed_50pct_irregular_actionformer_step2_irregular_points_hard_asym` | 24013 | **34.41** | done | bridge Step2 |
| `input_random_fixed_50pct_irregular_actionformer_step3_irregular_points_soft_sym` | 25876 | **49.86** | done | bridge Step3 |
| `input_random_fixed_50pct_irregular_actionformer_step4_irregular_points_soft_asym` | 25876 | **49.79** | done | bridge Step4 |

## B. 可信，但属于 eval-only 推理消融

| Experiment | Server | Avg-mAP | Status | Why trusted |
|---|---:|---:|---|---|
| `input_random_fixed_50pct_irregular_actionformer_headv3_x_boundaryaware` | 24013 | **50.55** | eval-only | `Test INFO` 日志清楚，对应 checkpoint / config 明确 |
| `input_random_fixed_50pct_irregular_actionformer_headv3_y_boundaryaware` | 24013 | **47.97** | eval-only | `Test INFO` 日志清楚，对应 checkpoint / config 明确 |

说明：

- 这两条是“推理阶段 boundary-aware ablation”，不是独立重训 run
- 因而它们是可信结果，但不应与完整训练 run 完全等权并列表达

## C. 已跑过，但结果不可信或不应纳入正式比较

| Experiment | Server | Last visible result | Problem | Judgment |
|---|---:|---:|---|---|
| `input_random_fixed_50pct_irregular_actionformer_step0_densehead` | 24013 | `13.79` | 没有 `Training Over`，停在 `Epoch 52`；同时这条是已知无效 control | invalid |
| `input_random_fixed_50pct_irregular_actionformer_remap` | 25876 | none | `ChildFailedError` 崩溃 | invalid |
| `input_random_fixed_50pct_irregular_actionformer_diag` | 25876 | none | 被 `SIGTERM` 中断 | invalid |
| `input_random_fixed_50pct_irregular_actionformer_diag_epoch1` | 25876 | no eval | 1-epoch debug only，存在 non-finite gradient 诊断信息，但不是可比性能实验 | debug-only |

此外还有若干不应纳入正式比较的 duplicate / scratch logs：

- `headv2_x_0032`
- `headv2_x_0035`
- `headv2_x_topk1_141449`
- `input_random_fixed_50pct_irregular_actionformer_.log`
- `run.log`
- `remap_v2 / remap_v3 / remap_v4`

这些条目可以作为排错痕迹保留，但不应再进入论文级结果表。

## D. 已部署，但还没有完整结果

| Experiment | Server | Current state | Latest visible result |
|---|---:|---|---:|
| `input_random_fixed_50pct_irregular_actionformer_headv3_oabs_visible_x` | 24013 | running | `49.20` |
| `input_random_fixed_50pct_irregular_actionformer_headv3_oabs_full_x` | 25876 | running | `49.80` |
| `input_random_fixed_50pct_irregular_actionformer_step0_dense_points_hard_shell_exact` | 24013 | queued after OABS-visible | none |
| `input_random_fixed_50pct_irregular_actionformer_step0b_dense_points_soft_sym` | 25876 | queued after OABS-full | none |

说明：

- `step0_dense_points_hard_shell_exact` 用于回答：`DensePassthrough proj/neck + ActionFormerHead` 在 `IrregularActionFormer` shell 下是否仍能复现 dense semantics
- `step0b_dense_points_soft_sym` 用于回答：`dense points + soft assignment` 本身会损失多少

## E. 本地存在，但还没有部署

| Experiment | Deployment status | Note |
|---|---|---|
| `input_random_fixed_50pct_irregular_actionformer_no_neck` | not deployed | 本地有 runnable config，但两台服务器都没有对应日志 / exp / queue 痕迹 |

## F. 不应被算作“未部署实验”的文件

以下文件是 base config，不是独立实验，不应计入“未部署实验”：

- `input_random_fixed_50pct_irregular_actionformer_denseexact_base.py`
- `input_random_fixed_50pct_irregular_actionformer_headv2_base.py`
- `input_random_fixed_50pct_irregular_actionformer_headv3_base.py`
- `input_random_fixed_50pct_irregular_actionformer_headv3_oabs_x_base.py`

## Bottom Line

当前这条模型线的 authoritative set 已经足够清晰：

1. `headv2_x = 51.04`，`headv3_x = 52.60`，主线有效但仍显著落后于 dense baseline
2. `headv2_denseexact = headv2_x = 51.04`，说明主残差不在 passthrough proj/neck fairness，而在 head-side semantics bundle
3. Step0_fixed 到 Step4 的 bridge decomposition 已经完整闭环，支持当前关于 hard ownership vs soft support assignment 的结构性判断
4. 现在最关键的未完成项只剩两类：
   - `OABS-visible / OABS-full`
   - `step0_shell_exact / step0b_dense_soft`

## Connections

AUTO-GENERATED from graph/edges.jsonl - do not edit manually
