---
type: experiment
node_id: exp:SPARSE_HEAD_SUMMARY_0414
title: "Sparse / irregular TAD head design summary and current decomposition status"
date: 2026-04-14
status: synthesized
created_at: 2026-04-14T17:25:00+08:00
updated_at: 2026-04-16T16:55:00+08:00
---

# SPARSE_HEAD_SUMMARY_0414

## 2026-04-16 upstream audit correction

This summary still captures the head-side trajectory, but three older readings are superseded:

- the current `x` line is a **dense-trunk + irregular-head** carrier, not a full irregular-aware trunk
- `step0_shell_exact = 53.72` is not a valid shell-exact baseline because it inherits a remap config
- `fixsingleton = 52.88` is not a clean config-level singleton ablation

For the corrected code-comparison interpretation, use:

- `exp:IRREGULAR_UPSTREAM_AUDIT_0416`

## Purpose

整理当前所有“稀疏头 / irregular head”相关实验，统一给出：

1. 实验演化路径  
2. 关键结果与对应结论  
3. 当前真正卡住的问题  
4. 下一步最干净的解决路线  

本文只讨论 **detector/head 侧的稀疏适应设计**。输入侧结果只作为参照上界与动机。

## Input-side reference

这些结果不是稀疏头实验本身，但它们定义了当前 detector 侧工作的参照系：

| Reference | Avg-mAP | Meaning |
|---|---:|---|
| `input_random_fixed_50pct` | **63.12** | 当前 random-fixed 50% 输入下的 dense detector 基线 |
| `input_oracle_boundary_dense_50pct` | **66.01** | 当前最强输入侧 oracle，上界来自 boundary 优先覆盖 |

因此，当前 irregular / sparse head 的直接目标不是先超过 66.01，而是先解释为什么在 **同样 random-fixed 50% 输入** 上，dense head 能到 **63.12**，而 irregular-aware head 目前只有 **51~53**。

---

## 1. 实验历程

### 阶段 A：旧 irregular detector 回溯与“语义崩溃”确认

| Experiment | Avg-mAP | Delta vs dense baseline | Readout |
|---|---:|---:|---|
| `input_random_fixed_50pct_irregular_actionformer` | **39.04** | -24.08 | 审计后的旧版 transitional irregular detector |
| `..._remap_no_regrange` | **4.13** | -58.99 | native irregular 语义基本崩溃 |

关键结论：

- 之前流传的 `64.62` 旧 irregular 高分 **不成立**；原始日志只支持 `39.04`
- 从 `39.04 -> 4.13` 说明真正的问题不是简单 NaN，而是 **head + point semantics** 的严重错配

对应文档：

- `exp:IRREGULAR_TIMELINE_0412`

### 阶段 B：HeadV2 把 detector 从“崩溃”拉回“可学”

HeadV2 的三个核心修正：

- cost-based top-k soft assignment
- asymmetric `log1p(left / cell_left), log1p(right / cell_right)`
- branch conv 保持 `k=3`，仅 final predictor 用 `k=1`

| Experiment | Avg-mAP | mAP@0.7 | Delta vs dense baseline | Main meaning |
|---|---:|---:|---:|---|
| `headv2_x` | **51.04** | **25.04** | -12.08 | dense-like bridge proj/neck + HeadV2 |
| `headv2_y` | **49.00** | **22.71** | -14.12 | full irregular proj/neck + HeadV2 |

关键结论：

1. HeadV2 确实修复了最大块的 supervision collapse  
   - `4.13 -> 51.04` 不是小修，而是 detector 重新变得可学
2. 但 residual 仍然很大  
   - 相对 dense baseline 仍差 `12.08 mAP`
3. `Y < X` 只差 `2.04`  
   - 说明 irregular proj/neck 不是主要残差来源，但目前也没有带来净收益

对应文档：

- `exp:HEADV2_0413`

### 阶段 C：HeadV2 内部 ablation，收紧结论

#### C1. `topk=1` 对照

| Experiment | Avg-mAP | Delta vs `headv2_x` | Conclusion |
|---|---:|---:|---|
| `headv2_x_topk1` | **47.19** | **-3.85** | 单 GT 单点支持更差 |

严谨解读：

- 这 **不等于** dense ActionFormer 的 hard assignment
- 它只说明：在当前 **cost-based candidate family** 里，`topk=9` 比 `topk=1` 更好
- 因而当前支持的是：
  - **multi-positive support 是正贡献**
- 但当前还 **不能**据此断言：
  - “soft quality weighting 一定比 uniform weight 更好”
  - “dense hard assignment 一定更差”

对应文档：

- `exp:HEADV2_X_TOPK1`

#### C2. final predictor `k=3` 对照

| Experiment | Avg-mAP | Delta vs `headv2_x` | Conclusion |
|---|---:|---:|---|
| `headv2_x_k3` | **49.53** | **-1.51** | final `k=3` 比 `k=1` 更差 |

严谨解读：

- 在 irregular timeline 上，final layer 的 geometry-agnostic local mixing 是有害的
- `k=1` 不是当前残差来源，反而是 **正确保留的设计**

### 阶段 D：denseexact 公平对照，排除 proj/neck 残差

实现：

- `DensePassthroughConv1DTransformerProj`
- `DensePassthroughFPNIdentity`

目的：

- 在 **相同 detector / temporal_grid contract** 下，排除 passthrough proj/neck 与原先 `headv2_x` 之间的工程差异

| Experiment | Avg-mAP | Delta vs `headv2_x` | Conclusion |
|---|---:|---:|---|
| `headv2_denseexact` | **51.04** | **0.00** | 与 `headv2_x` 完全一致 |

这条结果非常关键：

- 当前 `63.12 -> 51.04` 的残差，不来自这类 proj/neck passthrough 差异
- 更准确地说，主残差位于 **head-side semantics bundle**

也就是：

- points 如何定义
- assignment 如何分配
- regression 如何编码/解码
- center-based decoding 如何使用这些信号

### 阶段 E：HeadV3 扩展 —— geometry modulation + boundary auxiliary

HeadV3 新增：

- geometry-aware FiLM modulation
- boundary auxiliary branch（训练时）

| Experiment | Avg-mAP | mAP@0.7 | Delta vs parent | Readout |
|---|---:|---:|---:|---|
| `headv3_x` | **52.60** | **26.52** | `+1.56` vs `headv2_x` | X 线上有稳定正收益 |
| `headv3_y` | **49.37** | **22.87** | `+0.37` vs `headv2_y` | Y 线上仅小幅改善 |

关键结论：

1. geometry modulation + boundary aux **不是错方向**
2. 但增益有限
3. 说明它们更像是 **二级修正项**，无法解释主残差

### 阶段 F：backbone-side irregular time embedding

在冻结 VideoMAE backbone 的前提下，只训练新加的 `time_embed` 参数。

| Experiment | Avg-mAP | Delta vs parent | Conclusion |
|---|---:|---:|---|
| `headv3_timeembed_x` | **50.39** | **-2.21** vs `headv3_x` | X 线有害 |
| `headv3_timeembed_y` | **47.06** | **-2.31** vs `headv3_y` | Y 线有害 |

关键结论：

- 当前这版“冻结 backbone + 后加 time MLP”的方案 **不成立**
- 更可能是：
  - 随机初始化 time embedding 扰乱了预训练特征分布
  - 冻结 backbone 无法共同适配这个新信号

因此：

- 当前阶段 **不应继续沿着 naive frozen time-embed 方向堆实验**

### 阶段 G：boundary-aware inference 完整版

已经完整实现：

- `_select_level_boundary_mask`
- `_collect_boundary_bank`
- `_select_boundary_subset`
- `_boundary_pair_refine`
- `_boundary_aware_inference`

评测方式：

- 在已训练好的 `HeadV3` checkpoint 上，直接用 boundary-aware inference 替换原 center-based inference

| Experiment | Avg-mAP | Delta vs parent | Conclusion |
|---|---:|---:|---|
| `headv3_x_boundaryaware` | **50.55** | **-2.05** vs `headv3_x` | 有害 |
| `headv3_y_boundaryaware` | **47.97** | **-1.40** vs `headv3_y` | 有害 |

关键结论：

- 当前 boundary 分支 **作为训练辅助** 可能有帮助
- 但当前 boundary prediction **还不能直接拿来做 inference 主决策**

最合理的解释是：

- **train / inference mismatch**
- 训练时 boundary branch 学的是 soft proximity
- 推理时却被拿去做 boundary pairing + snapping + rescoring

这两者不是同一个任务

---

## 2. 当前结果总表

### 2.1 稀疏头主线实验总表

| Family | Experiment | Avg-mAP | Delta vs 63.12 | Direct readout |
|---|---|---:|---:|---|
| old irregular | old audited irregular | 39.04 | -24.08 | 旧版能学，但很弱 |
| native irregular | remap_no_regrange | 4.13 | -58.99 | 语义崩溃 |
| HeadV2 | `headv2_x` | 51.04 | -12.08 | 主要恢复 |
| HeadV2 | `headv2_y` | 49.00 | -14.12 | irregular path 仍略弱 |
| HeadV2 ablation | `headv2_x_topk1` | 47.19 | -15.93 | `topk=1` 更差 |
| HeadV2 ablation | `headv2_x_k3` | 49.53 | -13.59 | final `k=3` 更差 |
| HeadV2 fairness | `headv2_denseexact` | 51.04 | -12.08 | passthrough proj/neck 残差约 0 |
| HeadV3 | `headv3_x` | **52.60** | **-10.52** | 当前最好 irregular-head |
| HeadV3 | `headv3_y` | 49.37 | -13.75 | Y 线改善有限 |
| HeadV3 + backbone time | `headv3_timeembed_x` | 50.39 | -12.73 | time embed 有害 |
| HeadV3 + backbone time | `headv3_timeembed_y` | 47.06 | -16.06 | time embed 有害 |
| HeadV3 + boundary inference | `headv3_x_boundaryaware` | 50.55 | -12.57 | boundary-aware inference 有害 |
| HeadV3 + boundary inference | `headv3_y_boundaryaware` | 47.97 | -15.15 | boundary-aware inference 有害 |

### 2.2 当前最稳的定量判断

1. **HeadV2 修复了大部分 semantic collapse**
2. **HeadV3 带来小幅正收益，但不足以解释主残差**
3. **denseexact = headv2_x**
   - 当前 residual 不是 passthrough proj/neck 差异
4. **topk1 < topk9**
   - 当前 cost-based family 里 multi-positive support 是正贡献
5. **k3 < k1**
   - final predictor 的局部混合不是救命项，反而有害
6. **naive frozen backbone time embedding 有害**
7. **直接 boundary-aware inference 有害**

---

## 3. 这条实验历程说明了什么

### 3.1 已经被支持的判断

#### 判断 A：最大问题确实在 detector 语义，不在输入本身

证据链：

- dense random-fixed 50% 仍可达 `63.12`
- old irregular `39.04`
- native irregular `4.13`
- HeadV2 一次性拉回 `51.04`

支持的结论：

- 当前瓶颈主要来自 **irregular detector 的 head-side semantics bundle**
- 而不是“random-fixed 这种输入天生不可做”

#### 判断 B：projection / neck 不是主残差

证据：

- `headv2_denseexact = 51.04 = headv2_x`

支持的结论：

- 当前 12 分残差不应再优先归因给 passthrough proj/neck 差异

#### 判断 C：HeadV3 方向有一定价值，但只是二级修正

证据：

- `headv3_x = 52.60 > 51.04`
- 但提升只有 `+1.56`

支持的结论：

- geometry-aware modulation / boundary auxiliary 可以继续保留为局部增强
- 但它们不是当前主问题的根因修复

### 3.2 已被削弱或暂时不支持的判断

#### 判断 D：继续堆 backbone-side frozen time embedding

当前证据不支持。

原因：

- `headv3_timeembed_x = 50.39 < 52.60`
- `headv3_timeembed_y = 47.06 < 49.37`

#### 判断 E：把 boundary aux 直接升级为 inference 主逻辑

当前证据不支持。

原因：

- `headv3_x_boundaryaware = 50.55 < 52.60`
- `headv3_y_boundaryaware = 47.97 < 49.37`

更准确的说法是：

- **boundary 方向本身未被否定**
- 被否定的是 **当前这版 train/inference contract**

---

## 4. 当前真正的问题是什么

### 问题 1：仍有约 `10.5~12.1 mAP` 主残差没有解决

最关键对照：

- dense baseline: `63.12`
- best irregular-head: `headv3_x = 52.60`
- residual: `10.52`

这说明：

- 即便 HeadV2/HeadV3 已经修复了“崩溃”
- 当前 head-side semantics bundle 仍然显著弱于 dense head

### 问题 2：当前“更 irregular-aware 的设计”并没有自动带来更强性能

最典型证据：

- `headv2_denseexact = headv2_x = 51.04`
- `headv3_x = 52.60`
- dense baseline 仍是 `63.12`

这意味着：

- 在当前 random-fixed 50% 这种 **温和 irregular** 输入上
- dense head 的强归纳偏置（hard ownership、binary positives、简单稳定回归）仍然更有效
- 而我们当前的 irregular 语义改写虽然更灵活，但也更难优化、正则更弱

### 问题 3：boundary 分支和 backbone time-embed 都存在明显的 contract mismatch

表现：

- boundary branch：训练时学 proximity，推理时做 pairing / snapping / rescoring
- time-embed：训练新时间信号，但 backbone 主体冻结，无法协同适配

### 问题 4：我们仍然没有把 12 分残差拆干净

虽然已经排除了 passthrough proj/neck 差异，但还有更关键的未分解量：

1. **只把 points 放到 irregular axis 上，会损失多少？**
2. **soft assignment 相对 hard assignment，真正造成了多少损失？**
3. **asymmetric log1p regression 相对 symmetric linear，真正造成了多少损失？**

这也是为什么当前最重要的是正在跑的 Step0-4 bridge matrix。

---

## 5. 我们准备如何解决

### 主线：从 dense semantics 出发，逐步替换 irregular 组件

已经实现：

- `IrregularActionFormerBridgeHead`

当前正在运行的 clean decomposition 矩阵：

| Step | Semantics | Status |
|---|---|---|
| `Step0` | dense points + hard assignment + symmetric linear reg | 运行中 |
| `Step1` | irregular points + hard assignment + symmetric linear reg | 运行中 |
| `Step2` | irregular points + hard assignment + asymmetric log1p reg | 排队中 |
| `Step3` | irregular points + soft assignment + symmetric linear reg | 排队中 |
| `Step4` | irregular points + soft assignment + asymmetric log1p reg | 排队中 |

当前队列：

- Server 1 (`24013`)
  - 正在跑：`step0_densehead`
  - 排队：`step2_irregular_points_hard_asym`
- Server 3 (`25876`)
  - 正在跑：`step1_irregular_points_hard_sym`
  - 排队：`step3_irregular_points_soft_sym`
  - 排队：`step4_irregular_points_soft_asym`

截至 `2026-04-14 17:15 +08:00`：

- `Step0`、`Step1` 都刚进入早期训练，首个 eval 尚未出来
- 两条都在 epoch 0~2 阶段出现过少量 non-finite gradient skip，但训练仍继续推进

### 为什么这组 Step0-4 是当前最高优先级

因为它能直接回答：

1. **irregular points 本身是否有害**
2. **soft assignment 是否是主要损失来源**
3. **asymmetric regression 是否是主要损失来源**

其中最关键的是 `Step1`：

- 如果 `Step1 ≈ Step0 ≈ 63`
  - irregular points 本身几乎无害
  - 主损失来自 assignment / regression 改写
- 如果 `Step1` 已经明显掉到 `55` 以下
  - 那么只把 dense semantics 搬到 irregular points 上就已经不成立
  - 问题更深，可能是 center-point family 与 irregular axis 的根本不匹配

### Step0-4 之后的决策规则

#### 情况 A：`Step1` 很高，但 `Step3/4` 明显掉分

结论：

- 主问题在 **assignment / regression 语义改写**

下一步：

- 回到更接近 dense semantics 的半规则化设计
- 例如：
  - hard ownership + soft fallback
  - hybrid regression range
  - single-best regression target + softer cls supervision

#### 情况 B：`Step1` 已经很差

结论：

- 问题不只是 soft/asym
- 而是 **center-based point detector family 本身** 在 irregular timeline 上就不稳

下一步：

- boundary-first / proposal-refinement / set-based 方向优先级上升

#### 情况 C：`Step4` 约等于 `headv2_x`

结论：

- BridgeHead 实验矩阵和 HeadV2 主线是对齐的
- 后续所有结论都可以基于这组 clean decomposition 做因果归因

---

## 6. 当前最准确的总判断

一句话概括：

- **我们已经证明：irregular detector 不再是“完全学不会”，但当前稀疏头主残差仍牢牢卡在 head-side semantics bundle，而不是输入、backbone、或 passthrough proj/neck。**

更细一点：

1. `HeadV2` 修复了崩溃  
2. `HeadV3` 只提供了小幅增益  
3. `denseexact` 排除了 proj/neck fairness 问题  
4. `topk1` 和 `k3` 说明当前默认 `topk9 + final k=1` 是更好的局部选择  
5. `time_embed` 和 `boundary-aware inference` 都暴露了新的 contract mismatch  
6. 因此当前最重要的不是继续堆模块，而是完成 `Step0-4` 的 clean semantic decomposition

---

## 2026-04-15 Update: 5-Step Causal Decomposition

| Step | Experiment | GT axis / semantics | Points | Assignment | Regression | Avg-mAP | Delta vs previous | Causal readout |
|---|---|---|---|---|---|---:|---:|---|
| Step0_fixed | `step0_densehead_remap` | remapped selected axis, pseudo-dense semantics | dense points | hard | symmetric linear | **53.72** | - | once the sparse axis is remapped back into dense semantics, a dense-style head can train again |
| Step1 | `step1_irregular_points_hard_sym` | native irregular axis | irregular points | hard | symmetric linear | **37.10** | **-16.62** | moving only the points onto the native irregular timeline while keeping dense hard ownership causes a major semantic break |
| Step2 | `step2_irregular_points_hard_asym` | native irregular axis | irregular points | hard | asymmetric log1p | **34.41** | **-2.69** | asymmetric regression does not rescue hard ownership; under this setup it makes optimization even less stable |
| Step3 | `step3_irregular_points_soft_sym` | native irregular axis | irregular points | soft top-k | symmetric linear | **49.86** | **+15.45** vs Step2 / **+12.76** vs Step1 | the main recovery comes from replacing hard ownership with soft support assignment on the irregular axis |
| Step4 | `step4_irregular_points_soft_asym` | native irregular axis | irregular points | soft top-k | asymmetric log1p | **49.79** | **-0.07** | once assignment is fixed, asymmetric log1p regression is not the dominant residual source |

**Interpretation**

- `step0_densehead = 13.79` was an invalid control, because it kept dense-head semantics directly on the native irregular axis without remap. The correct bridge baseline is `step0_densehead_remap = 53.72`.
- The main causal break is `Step0_fixed -> Step1`: the dense-style hard level ownership assumption does not survive the move to native irregular points.
- The main causal recovery is `Step1 -> Step3`: soft multi-support assignment is the key ingredient that pulls the detector back from semantic collapse.
- `Step2` shows that asymmetric regression is not a standalone fix. Under hard ownership it is actually less stable; earlier training logs also showed non-finite gradient warnings on this line.
- `Step3 -> Step4` is nearly flat, so the current 50-ish plateau is not primarily explained by asymmetric regression. The dominant residual is still the head-side assignment / supervision semantics bundle.

## 2026-04-16 Update: the remaining open diagnosis gap is now closed

Two newly completed results materially change the reading of this whole line:

| Experiment | Avg-mAP | Direct implication |
|---|---:|---|
| `step0b_dense_points_soft_sym` | **47.52** | soft assignment is directly harmful even under dense points |
| `headv3_oabs_full_oaa_x` | **53.20** | OAA is only a tiny gain over OABS-full |
| `headv3_oabs_full_x_fixsingleton` | **52.88** | the OABS singleton bug was real, but not the main bottleneck |

### New causal tightening

1. `63.12 -> 53.72` is now the clearest shell / path / contract overhead.
2. `53.72 -> 47.52` shows that soft assignment is itself a major negative factor, not merely a rescue tool for native irregular points.
3. `37.10 / 34.41` still show that naive hard semantics on the native irregular axis collapse badly.
4. Therefore the correct pivot is no longer "find a slightly better soft-assignment irregular head."
5. The cleaner mainline is now:
   - sparse-to-dense completion
   - unchanged dense detector path

### Updated priority

The highest-value next experiment is no longer another shell-local OABS / OAA refinement. It is:

1. oracle completion upper bound into the original dense detector
2. minimal learned completion module into the original dense detector

This is the first pivot that directly attacks both confirmed structural losses at once:

- the irregular-shell overhead
- the soft-assignment overhead

## Key file references

- `research-wiki/experiments/IRREGULAR_TIMELINE_0412.md`
- `research-wiki/experiments/HEADV2_0413.md`
- `research-wiki/experiments/HEADV2_X_TOPK1_20260413.md`
- `research-wiki/experiments/HEADV3_0413.md`
- `OpenTAD_Back/opentad/models/dense_heads/irregular_actionformer_head_v3.py`
- `OpenTAD_Back/opentad/models/dense_heads/irregular_actionformer_bridge_head.py`
- `OpenTAD_Back/configs/adatad/thumos/input_random_fixed_50pct_irregular_actionformer_headv2_denseexact.py`
- `OpenTAD_Back/configs/adatad/thumos/input_random_fixed_50pct_irregular_actionformer_step0_densehead.py`
- `OpenTAD_Back/configs/adatad/thumos/input_random_fixed_50pct_irregular_actionformer_step1_irregular_points_hard_sym.py`
- `OpenTAD_Back/configs/adatad/thumos/input_random_fixed_50pct_irregular_actionformer_step2_irregular_points_hard_asym.py`
- `OpenTAD_Back/configs/adatad/thumos/input_random_fixed_50pct_irregular_actionformer_step3_irregular_points_soft_sym.py`
- `OpenTAD_Back/configs/adatad/thumos/input_random_fixed_50pct_irregular_actionformer_step4_irregular_points_soft_asym.py`

## Connections

AUTO-GENERATED from graph/edges.jsonl - do not edit manually
