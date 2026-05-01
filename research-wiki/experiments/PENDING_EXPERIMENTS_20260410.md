---
type: experiment
node_id: exp:PENDING20260410
title: "Unrun / Unfinished Experiment Matrix (2026-04-10)"
date: 2026-04-10
status: active
config: mixed
created_at: 2026-04-10T18:59:30+08:00
updated_at: 2026-04-13T22:55:59+08:00
---

# 未进行实验总表（2026-04-10）

## 统计口径
- 本页只统计“有明确研究问题、已经写成配置、但尚未形成最终结论”的实验。
- 不统计 `diag_*`、`*_small64x32.py`、`*_fp32.py`、`*_bs1*.py`、`*_diag*.py` 这类纯诊断或极小样本配置。

## A. V3 / ActionFormer 主线未完成实验

| 实验名 | 配置路径 | 改动层级 | 当前状态 | 主要验证问题 | 备注 |
|---|---|---|---|---|---|
| Oracle 50% 上界 | `OpenTAD/configs/frame_sampling/thumos/v3_oracle_50pct.py` | sampler / detector 对照 | 未运行 | 在当前 V3 backbone + detector 下，完美 50% 采样的理论上界有多高 | 直接回答 `G2` |
| Random 50%（legacy lower bound） | `OpenTAD/configs/frame_sampling/thumos/v3_random_50pct.py` | sampler | 未运行 | 随机 50% 相比 uniform 50% 到底差多少 | 已被 random-fixed 思路部分替代 |
| Random-varying 50% | `OpenTAD/configs/frame_sampling/thumos/v3_random_varying_50pct.py` | sampler | 未运行 | 每个 batch 更换 mask 时，性能是否进一步恶化；mask 一致性是否关键 | 对照对象是 `v3_random_fixed_50pct.py` |
| R016 attn_scale 消融 | `OpenTAD/configs/frame_sampling/thumos/v3_ablation_attn_scale.py` | backbone | 未运行 | 恢复可学习 `attn_scale` 后是否会稳定拉低 uniform 50% | 属于 P0 fix 消融缺口 |
| Uniform + indicator embed | `OpenTAD/configs/frame_sampling/thumos/v3_uniform_50pct_indicator_embed.py` | projection | 结果未回收 | 显式标记 kept / dropped 是否提升 uniform 50% | 仍无可信最终结论 |
| Uniform + soft attention bias | `OpenTAD/configs/frame_sampling/thumos/v3_uniform_50pct_soft_attn_bias.py` | projection | 结果未回收 | projection attention 中软抑制 dropped K/V 是否优于完全不标记 | |
| Uniform + irregular PE | `OpenTAD/configs/frame_sampling/thumos/v3_uniform_50pct_irregular_pe.py` | projection | 未运行 | gap-to-nearest-kept 位置编码是否改善 projection 对 stale token 的感知 | |
| Uniform + partial conv | `OpenTAD/configs/frame_sampling/thumos/v3_uniform_50pct_partial_conv.py` | projection | 未运行 | confidence-aware partial conv 是否比零化 dropped 更合理 | |
| Uniform + projection all | `OpenTAD/configs/frame_sampling/thumos/v3_uniform_50pct_projection_all.py` | projection | 未运行 | indicator + soft bias + partial conv 组合是否可叠加收益 | |
| Hard scorer + projection all | `OpenTAD/configs/frame_sampling/thumos/v3_hard_scorer_aux_projection_all.py` | sampler + projection | 未运行 | projection-aware 优化能否帮助 `R017` 这条 hard scorer 线 | |
| Tubelet rank pretrain | `OpenTAD/configs/frame_sampling/thumos/v3_hard_tubelet_rank_pretrain.py` | sampler training | 未运行 | 真正 hard top-k 前先让 scorer 学到 tubelet 排序能力，能否缓解冷启动 | Stage A |
| Tubelet rank finetune | `OpenTAD/configs/frame_sampling/thumos/v3_hard_tubelet_rank_finetune.py` | sampler training | 未运行 | warm-started scorer + small rank / REINFORCE / coarse_det 能否超过 uniform | 依赖 pretrain checkpoint |

## B. 输入侧验证实验（OpenTAD_Back 主实现）

| 实验名 | 配置路径 | 改动层级 | 当前状态 | 主要验证问题 | 备注 |
|---|---|---|---|---|---|
| Input Oracle Boundary Dense 50% | `OpenTAD_Back/configs/adatad/thumos/input_oracle_boundary_dense_50pct.py` | data / input sampling | 已完成 66.01 | 边界附近更密集是否优于 uniform / random-fixed | 已证明边界优先 50% 可超过 uniform 50% |
| Input Oracle Action+Boundary Dense 50% | `OpenTAD_Back/configs/adatad/thumos/input_oracle_action_boundary_dense_50pct.py` | data / input sampling | 已完成 62.04 | “边界最高、动作中等、背景最低”是否更合理 | 已证伪该假设，低于单纯边界优先 |
| Input Weighted Random Boundary 50% | `OpenTAD_Back/configs/adatad/thumos/input_weighted_random_boundary_50pct.py` | data / input sampling | 已完成 61.33 | 语义偏置是否必须依赖硬优先选择，还是概率偏置也有效 | 低于 random-fixed 63.12 与 oracle boundary 66.01 |
| Input Weighted Random Action+Boundary 50% | `OpenTAD_Back/configs/adatad/thumos/input_weighted_random_action_boundary_50pct.py` | data / input sampling | 已完成 59.24 | 保持随机性的同时，动作/边界偏置能否稳定优于 uniform | 未优于 uniform 或 random-fixed，动作内部加权仍有害 |
| Input Random-fixed Tubelet2 50% | `OpenTAD/configs/frame_sampling/thumos/input_random_fixed_tubelet2_50pct.py` | data / input sampling | 未运行 | 2 帧 tubelet 随机 50% 是否比逐帧随机更符合 VideoMAE tubelet 语义 | |
| Input Oracle Boundary Dense Tubelet2 50% | `OpenTAD_Back/configs/adatad/thumos/input_oracle_boundary_dense_tubelet2_50pct.py` | data / input sampling | 未运行 | 边界优先在 tubelet 级选择时是否更稳 | 逐帧 vs tubelet 对照 |
| Input Oracle Action+Boundary Dense Tubelet2 50% | `OpenTAD_Back/configs/adatad/thumos/input_oracle_action_boundary_dense_tubelet2_50pct.py` | data / input sampling | 未运行 | 动作+边界优先在 tubelet 级是否更合理 | |
| Input Weighted Random Boundary Tubelet2 50% | `OpenTAD_Back/configs/adatad/thumos/input_weighted_random_boundary_tubelet2_50pct.py` | data / input sampling | 未运行 | 加权随机 + tubelet 选择能否减少逐帧随机抖动 | |
| Input Weighted Random Action+Boundary Tubelet2 50% | `OpenTAD_Back/configs/adatad/thumos/input_weighted_random_action_boundary_tubelet2_50pct.py` | data / input sampling | 已完成 58.54 | 语义偏置随机在 tubelet 级是否能兼顾覆盖与时间结构 | tubelet2 未救回 action+boundary 线，甚至低于逐帧版 59.24 |

## C. 当前最该优先回收的结果
1. `v3_oracle_50pct`
2. `v3_hard_tubelet_rank_pretrain`
3. `input_weighted_random_boundary_50pct`
4. `input_weighted_random_action_boundary_50pct`

原因：
- 前两条直接决定 V3 主线还有没有真实上界空间，以及 hard scorer 是否还有救。
- 后两条继续回答“边界优先必须是硬 top-k，还是概率偏置也够用”。
