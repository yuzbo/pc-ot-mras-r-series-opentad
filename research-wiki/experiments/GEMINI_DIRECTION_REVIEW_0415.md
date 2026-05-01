---
type: review
node_id: review:GEMINI_DIRECTION_0415
title: "Gemini direction review on irregular-head line after OABS-full"
date: 2026-04-15
status: completed
created_at: 2026-04-15T16:41:57+08:00
updated_at: 2026-04-15T16:41:57+08:00
---

# GEMINI_DIRECTION_REVIEW_0415

## Scope

本页记录一次基于当前“已完整训练且可信”实验结果的 Gemini 方向审阅。目标不是让 Gemini 重新总结已有实验，而是让它回答：

1. 当前主线是否应该继续押注 `observation-aware supervision / target`
2. 剩余约 `10 mAP` 残差最可信的根因排序是什么
3. 在当前代码与工程约束下，下一轮最有信息量、最小但足够强的实验是什么

本次审阅显式排除了仍在运行的实验，不把 `step0_shell_exact / step0b_dense_points_soft_sym` 当作证据。

## Review Context Sent to Gemini

发送给 Gemini 的关键结果包括：

- dense baseline `63.12`
- oracle boundary dense `66.01`
- old irregular `39.04`
- `remap_no_regrange = 3.92`
- `remap_overlap = 2.12`
- `headv2_x = 51.04`
- `headv2_y = 49.00`
- `headv2_denseexact = 51.04`
- `headv2_x_topk1 = 47.19`
- `headv2_x_k3 = 49.53`
- `headv3_x = 52.60`
- `headv3_y = 49.37`
- `headv3_timeembed_x = 50.39`
- `headv3_timeembed_y = 47.06`
- `headv3_x_boundaryaware = 50.55`
- `headv3_y_boundaryaware = 47.97`
- `headv3_oabs_visible_x = 51.49`
- `headv3_oabs_full_x = 53.16`
- `step0_densehead_remap = 53.72`
- `step1 = 37.10`
- `step2 = 34.41`
- `step3 = 49.86`
- `step4 = 49.79`

## Round 1: Gemini Main Judgment

Gemini 第一轮核心判断：

1. 认同当前最该押注的是 `observation-aware supervision / target`，而不是 frozen backbone irregularization 或 post-hoc boundary-aware inference。
2. 认为 OABS-full `53.16` 说明“观测感知监督”是当前最有效方向，但它只解决了一部分问题，还没有触及 dense baseline `63.12` 的主上限。
3. 对剩余残差，Gemini 的表述偏向：
   - `Temporal Context Degradation`
   - `Supervision Misalignment`
   - `Inference-time Geometry Gap`
4. Gemini 同时建议不要继续优先投入：
   - frozen backbone + time embedding
   - boundary-aware post-hoc inference

## Round 2: Constraint-tightened Follow-up

我们在第二轮对 Gemini 明确了两个约束：

1. `headv2_denseexact = 51.04` 已经是关键公平对照，不应再把问题归结为 proj/neck 公平性或实现异常。
2. 当前工程优先级应收缩到：
   - 观测感知的 target / assignment / decoding 契约一致性
   - 避免立即跳向大规模 teacher-student 蒸馏
   - 暂不继续优先做 backbone irregularization 与 post-hoc boundary inference

在这个约束下，Gemini 将建议重新收紧为以 `OABS-full = 53.16` 为中心的三个最小实验。

## Gemini Final Recommendations

### 1. OAA: Observation-Aware Assignment

建议内容：

- 将正负样本指派逻辑从标准时间重叠，改为只在可观测区域上计算的 assignment 依据
- Gemini 给出的抽象形式是“可见重叠 / visible overlap / visible IoU”

Gemini 想验证的因果命题：

- 当前残差的一部分来自“指派与观测不一致”
- 标准 assignment 在稀疏输入下会制造虚假正样本或错误 ownership

Gemini 对结果的判读：

- 如果显著提升，说明当前瓶颈已经从 `target definition` 延伸到了 `assignment contract`
- 如果不提升，说明 OABS-full 对 target 的修正已足够吸收这类噪声，主矛盾不在 assignment

### 2. Learnable OABS Weighting

建议内容：

- 在 `OABS-full` 基础上，将当前基于可观测性的手工 uncertainty weighting，换成一个轻量可学习权重模块
- 例如输入 proposal 级可见比例 / 长度统计量，预测一个 loss weight

Gemini 想验证的因果命题：

- 手工规则是否过粗，模型能否学习更细的“观测-不确定性”映射

Gemini 对结果的判读：

- 如果提升，说明 OABS 方向应继续从启发式走向数据驱动
- 如果不提升，说明在当前特征质量下，简单启发式比复杂学习更稳

### 3. OAD: Observation-Aware Decoding

建议内容：

- 在推理 / NMS 阶段引入与训练阶段一致的 observation-aware overlap 度量
- 只建议在 OAA 证明有效后再做，作为闭环最后一环

Gemini 想验证的因果命题：

- 当前 inference contract 与训练目标不一致
- 标准 NMS 可能错误压制“在可见区域互补”的 proposal

Gemini 对结果的判读：

- 如果提升，说明需要从 assignment-target-decoding 做完整 observation-aware 闭环
- 如果不提升，说明 decoding 不是主要矛盾，训练阶段才是主战场

## Priority Order Given by Gemini

Gemini 给出的优先级排序：

1. `OAA / observation-aware assignment`
2. `Learnable OABS weighting`
3. `OAD / observation-aware decoding`

其理由是：

- OAA 信息量最高，直接回答“监督一致性是否还需要向前扩一层”
- Learnable OABS 是对当前最佳点 `53.16` 最自然、最小的强化
- OAD 只有在 OAA 成立时，才值得作为系统闭环补齐

## What We Accept

当前最值得吸收的 Gemini 结论：

1. 主线应继续收缩到 `observation-aware consistency`
2. `OABS-full = 53.16` 说明“观测感知 supervision”已被正向支持
3. 最有信息量的下一步，不是更复杂的 backbone 或 inference trick，而是：
   - assignment 是否也必须 observation-aware
   - OABS weighting 是否值得从启发式升级为轻量学习

## What We Treat Cautiously

Gemini 第一轮中“剩余残差首先来自 temporal context degradation”的说法，目前更像一个合理假说，而不是已被当前实验直接证明的结论。

当前实验更直接支持的是：

- 主残差在 `head-side semantics bundle`
- 其中已被明确证伪或削弱的方向包括：
  - final `k=3`
  - frozen backbone time embedding
  - post-hoc boundary-aware inference
  - asymmetric regression 作为主增益来源

因此，Gemini 关于“时序上下文退化”的判断，可以保留为后续高层解释候选，但当前不应越级取代已经由 bridge/OABS 结果支持的更直接结论。

## Bottom Line

Gemini 审阅后的收敛判断是：

1. 当前路线总体正确，但应继续收缩，而不是扩散
2. 最有价值的主线不是“更 irregular 的模块堆叠”，而是“让 assignment / target / decoding 与可观测性达成一致”
3. 当前最推荐的下一步不是新 backbone，也不是边界后处理，而是：
   - `OAA`
   - `Learnable OABS`
   - 在 OAA 成功后再做 `OAD`

## Thread Reference

- Reviewer: `gemini-2.5-pro`
- Gemini thread id: `4f4e7391075a4fb09d47ba403732505a`

