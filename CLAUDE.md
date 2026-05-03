# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.
本项目不允许进行任何工程简化，按照要求和计划严格落实与实现
如果过程中有不确定的实验细节，请你充分的向我提问题，然后再进行代码实现
每次进行代码修改后必须进行git提交，尽可能不在原始代码上进行修改，以防出现错误无法回档
---

## Project Overview

**TAD时序信息冗余研究项目** - 基于OpenTAD框架，研究时序动作检测中的视频时序信息冗余问题。

## Research Status

- **Stage 1**: ✅ Idea Discovery (评审7.5/10)
- **Stage 2**: ✅ Research Refine (评分7.0/10)
- **Stage 3**: ✅ 冗余验证实验已完成 (EXP-001~004)
- **Stage 4**: ✅ MVP实现完成 — V1(废弃) → V2(废弃) → V3(当前)
- **Stage 5**: 🔄 Learned scorer 训练中 — 目标: 超越 uniform 50% (63.75%)

## 核心结论

✅ **TAD任务存在适度时序冗余**: stride=1→2 采样率减半，性能仅降5.5%
⚠️ **OpenTAD中stride=1已是4×下采样**: THUMOS14 30fps, feature_stride=4 → 实际7.5fps

## 当前方法: MaskedViT V3 + Multi-Path Loss

```
原始视频帧(768帧) → 轻量scorer(56x56) → tubelet mask(2帧粒度, 50%)
→ MaskedViT V3(q=k=v=kept, hard gather/scatter) → dense特征(adapter平滑)
→ ActionFormer → 检测
```

### V3 关键设计 (P0fix 版)
- Attention: q=k=v=kept（真正省计算，语义自洽）
- use_attn_scale=False (禁用 attn_scale，防止 12 层累积压制)
- mask_adapter=True (adapter conv 前零化 dropped 位置，阻断污染)
- LayerNorm 在 full sequence 上做，再 gather kept tokens
- Scatter 每层回 full sequence，dropped 位置保留上一层值
- Adapter 在 full sequence 上运行（temporal conv 平滑差异）
- 采样率 50%, 计算节省: Attention O(K²) vs O(N²), MLP O(K) vs O(N)

### 多路 Loss 设计 (借鉴 Uni-AdaFocus)
- L_det: 主检测loss (长路径, 穿过整个backbone)
- L_scorer: boundary BCE 直接监督 (短路径)
- L_coarse_det: scorer coarse_feat → frozen projection+rpn_head (短路径, det级信号)
- L_aux: weighted_pool(backbone_feat, scorer_weights) → 分类 (短路径)
- L_entropy: entropy 正则化 (仅 gumbel 模式有效, hard 模式下 soft_weights=None)
- L_reinforce: REINFORCE policy gradient (hard 模式专用, 用 det loss 作 reward)

### ⚠️ 已证实失败的方向
- Soft Gating + 高权重 BCE (R012-R014): 52-55%, BCE 与 det loss 目标冲突
- Hard + BCE + coarse_det (R017): 57.55%, scorer 无 det loss 信号, 停留近 uniform
- Hard + 纯 coarse_det (R018): 47.68%, coarse_det 单独不足以训好 scorer
- mask_proj_input / compressed / sparse_head: 改变表示链, 无法与 R008 公平对比

### 当前核心问题诊断

**Learned scorer 全部不如 uniform (63.75%)**

根因分析 (按可能性排序):
1. **soft_gating 本身破坏特征分布?** — R012 (scorer=0.0, 纯 det loss) 仍然 52-55%
2. **BCE 与 det loss 目标冲突?** — R013 (scorer=1.0) 53%, R017 (scorer=1.0+coarse) 57.55%
3. **scorer 冷启动困难?** — 随机初始化 scorer 的选择可能比 uniform 差很多
4. **hard mode 下 scorer 无 det loss 信号** — frame_scores.detach() 切断梯度

待验证假设:
- R021 (gumbel+softgate, scorer=0.05): 如果 > 63.75%, 说明 det loss E2E 有效
- R021 如果仍 < 63.75%, 说明 soft_gating 本身有问题, 需要换方向

## 实验历史

### Phase 1 (R001-R007): Baseline + V3 基础
- ✅ Baseline stride=1: 68.97%, stride=2: 65.20%
- ✅ V3 nofill 50%: 60.68%, V3 nofill 25%: 43.07%

### Phase 2 (R008-R014): P0fix + Soft Gating
- ✅ **R008 uniform 50% P0fix: 63.75%** ← 当前最佳 (vs stride=2 差 1.45%)
- ❌ R012 gumbel+softgate (scorer=0.0): ~52-55%
- ❌ R013 gumbel+softgate+scorer=1.0: ~53%
- ❌ R014 gumbel+softgate+scorer=1.0+aux=0.5: 53.73%

### Phase 3 (R015-R018): Learned Scorer 训练
- ❌ R017 hard+scorer+coarse_det: 57.55% (scorer entropy=0.62, 近uniform)
- ❌ R018 coarse_det_only (scorer=0.0, coarse=1.0): 47.68%

### Phase 4 (R019-R021): 最小对照 + 新梯度方案
- 📋 R019 gumbel+softgate+coarse_det: 配置就绪 (多路径版)
- 📋 R020 hard+REINFORCE: 配置就绪 (policy gradient 版)
- 📋 **R021 gumbel+softgate+minimal**: 配置就绪 ← **第一优先级**
  - 基于 R008 底座, 只改 sampler policy
  - 单一主导: det loss E2E (通过 soft_gating)
  - 极低 BCE (0.05) 仅防冷启动

## 下一步实验优先级

1. **R021** (最小 learned-scorer 对照) — 回答"智能选帧能否超过 uniform"
2. **R020** (REINFORCE) — 如果 R021 失败, 尝试 hard mode + policy gradient
3. **R019** (gumbel+coarse_det) — 多路径版, 如果 R021 成功则加强

## 研究方向总览

| 方向 | 已有实验 | 状态 | 缺什么 |
|------|---------|------|--------|
| learned scorer | R012-R014, R017-R018 | ❌ 全部 < uniform | 需要 R021 验证 soft_gating + 极低 BCE |
| dense-timeline mismatch | — | 未开始 | sparse-native assign, 正负样本分配重写 |
| regression target | uniform_adaptive_compressed, sparse_head | 初步尝试 | 系统验证原时间轴 target, sparse-aware regression |
| coarse-to-fine | glance_focus_v1 | 原型 | proposal 级负样本, 分数校准, 完整可比版本 |

## V3 核心文件

### Backbone
- `OpenTAD/opentad/models/backbones/masked_vit_v3.py` — MaskedBlockV3 + MaskedVisionTransformerV3

### Detector
- `OpenTAD/opentad/models/detectors/frame_sampling_detector.py` — FrameSamplingActionFormerV3 + 多路Loss + REINFORCE

### Scorer
- `OpenTAD/opentad/models/projections/frame_sampler.py` — LightweightFrameScorer + TubeletMaskGenerator + Oracle
- `OpenTAD/opentad/models/projections/mobilenet_scorer.py` — MobileNetFrameScorer (预训练, 就绪未部署)

### 配置
- `OpenTAD/configs/frame_sampling/thumos/v3_uniform_50pct_p0fix.py` — **最佳** (R008, 63.75%)
- `OpenTAD/configs/frame_sampling/thumos/v3_gumbel_softgate_minimal.py` — **R021** (第一优先级)
- `OpenTAD/configs/frame_sampling/thumos/v3_hard_reinforce.py` — R020
- `OpenTAD/configs/frame_sampling/thumos/v3_gumbel_softgate_coarse_det.py` — R019

### 文档
- `EXPERIMENT_LOG.md` — 实验日志
- `UNI_ADAFOCUS_COMPARISON_REPORT.md` — Uni-AdaFocus 对照分析

## Environment

### WSL Environment
- Platform: WSL2
- Project: `/mnt/e/DeskTop/TAD/temrefuse-tad`
- Conda: `conda activate open-tad`

### Remote Server

**Server**: `ssh -p 35407 root@connect.cqa1.seetacloud.com` (RTX 4080 SUPER 32GB)

**配置**:
- 工作目录: `/root/autodl-tmp/OpenTAD_Back_check`
- Screen: `screen -dmS <name> bash -c '...'`
- Torchrun: `/root/miniconda3/bin/torchrun --nproc_per_node=1`
- 训练数据: `/root/autodl-tmp/train/`, 测试数据: `/root/autodl-tmp/test/`
- 标注: `/root/autodl-tmp/annotations/thumos_14_anno.json`
- 训练命令: `torchrun --nproc_per_node=1 tools/train.py <config>`

### 快速查看实验状态
```bash
ssh -p 35407 root@connect.cqa1.seetacloud.com "screen -ls"
ssh -p 35407 root@connect.cqa1.seetacloud.com "tail -100 /root/autodl-tmp/OpenTAD_Back_check/logs/<exp>.log | grep -i loss"
ssh -p 35407 root@connect.cqa1.seetacloud.com "grep 'mAP' /root/autodl-tmp/OpenTAD_Back_check/logs/<exp>.log | tail -10"
```

## Mandatory: Use RTK for All Bash Commands

All Bash tool calls MUST use `rtk` prefix to reduce token consumption. This is not optional.

```bash
rtk ls .                    # NOT: ls
rtk read file.py            # NOT: cat file.py
rtk grep "pattern" .        # NOT: grep or rg
rtk git status              # NOT: git status
rtk git diff                # NOT: git diff
rtk git log                 # NOT: git log
```

Check savings with `rtk gain`. Target: 60%+ token savings per session.

## Critical: Reflective Verification Protocol

1. **Periodic self-reflection**: After every major result or code change, pause and ask: "Does this make sense? Is this consistent with prior evidence?"

2. **Question all results**: Never accept a metric at face value. Cross-check against:
   - Prior runs with known-good configurations
   - Expected ranges from the literature or prior experiments
   - Sanity checks (e.g., prediction file size, gradient norms, loss trajectory shape)

3. **When results significantly violate expectations** (e.g., mAP drops to 0, loss explodes, metric improves implausibly):
   - STOP. Do not proceed to the next experiment.
   - Carefully and comprehensively audit the code for bugs: check the full diff of recent changes, verify data paths, confirm parameter values in `training_config.json`, inspect prediction outputs.
   - Compare against the last known-good state.
   - Document the diagnosis in `refine-logs/` before moving on.

4. **Git commit as safety net (MANDATORY, not optional)**:
   - **Verify repo exists**: At session start, run `git rev-parse --git-dir`. If no repo → `git init` immediately.
   - **Commit triggers** (any one → commit):
     - Before launching any training run (`torchrun tools/train.py`)
     - After any code edit to `src/` or `opentad/` (even single-line)
     - After updating `refine-logs/` or `research-wiki/`
     - After a new experimental result is confirmed
   - **Commit message format**: `<type>: <description>`
     - `feat:` — new feature (e.g., `feat: add END logit masking`)
     - `fix:` — bug fix (e.g., `fix: END<START strict mask caused Infinity loss`)
     - `exp:` — experiment launch/result (e.g., `exp: R039 always-on mask launched`)
     - `audit:` — code review findings (e.g., `audit: full codebase pass, no bugs found`)
     - `wiki:` — research wiki updates
   - **Post-commit verify**: Run `git log --oneline -1` to confirm. If commit fails, fix and retry — do NOT proceed without committing.
   - **Never skip hooks**: No `--no-verify`, no `--no-gpg-sign`.
   - When in doubt, commit first — a bad commit can be reverted, lost work cannot.

5. **Never change multiple things at once**: One variable per experiment. If training fails after changing both the scheduler and the feature normalization, you cannot diagnose which caused the failure.

## Rules

- 使用中文进行文档输出
- **严格执行：所有 bash 命令都必须加 rtk 前缀**（包括 ssh、scp、grep、tail、find、sleep 等所有命令，无一例外）
- **代码实现必须调用 Codex MCP (GPT) 完成**，Claude 只负责思路把控、代码审核和结果分析，不直接编写实现代码
- 需要定时反思，质疑结果和结论；当结果显著违反预期时，应当仔细、全面地检查当前代码是否存在错误
- 合理使用 git commit 存档，避免代码改崩
- 每次实验变更只改一个变量，禁止同时改多个因素
- 实验结果记录在 `EXPERIMENT_LOG.md`
- **实验完成后必须使用 research-wiki skill 记录实验进度**（调用 `/research-wiki` 更新 wiki index、实验页面和 claims）
- **代码核验流程**: 代码编写完成后，上传或实验开始前，必须使用 Codex MCP 进行交叉检查
- **实验验证流程**: 结果出现后提交 Codex 审查 → 代码验证 → 假设反思 → 确定性结论
- **Codex 审查重点**: 正确性、梯度路径、tensor维度、与框架兼容性
- **Codex MCP 超时处理**: 调用 Codex MCP 时设置合理 timeout（建议 120s），若长时间无应答则中断重试，不阻塞整体流程

## Codex 审查记录 (最新)

**MobileNet scorer 审查 (GPT-5.4)**:
- [FIX-1] 加入 ImageNet mean/std 归一化 ✅
- [FIX-2] 冻结层 BN 统计量保持 eval 模式 ✅
- [FIX-3] temporal_conv 默认关闭, BN→GroupNorm ✅
- [FIX-4] frame_diff 改为独立分支, 不修改预训练 conv ✅
- 核心建议: MobileNet 不应优先部署, 先跑 R018/Oracle 确认瓶颈在哪
- lambda_entropy 在 hard 模式下是死参数 (soft_weights=None) ✅ 已修复
