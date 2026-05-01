---
type: document
node_id: doc:CODE_AUDIT_0430
title: "全面代码审计报告 (2026-04-30)"
created_at: 2026-04-30T20:00:00+08:00
updated_at: 2026-04-30T20:00:00+08:00
---

# 全面代码审计报告 — OpenTAD_Back

**审计日期**: 2026-04-30
**审计范围**: 数据集管线、训练引擎、backbone、projection、FPN neck、所有 head 变体、损失函数、配置继承链
**审计方法**: 4 个并行 Agent 逐文件逐行审查 + Codex GPT-5.5 验证

---

## 🔴 CRITICAL 问题 (影响当前实验)

### C-DS-1: `_remap_gt_to_selected_axis` 右边界 GT 静默丢弃
**文件**: `opentad/datasets/transforms/end_to_end.py:545-549`
**状态**: 存疑 (Codex 认为需要 start==max_coord 才触发, 不常见)
**影响**: 当 GT segment 刚好映射到右边界时可能静默丢失

### C-TR-1: Cosine 调度器 horizon 不匹配
**文件**: `tools/train.py:141-145`
**状态**: CONFIRMED — 系统性问题
**影响**: 所有 60 epoch 实验都用 100-epoch cosine 曲线。最终 LR ≈ `eta_min + 0.38*(base_lr - eta_min)`, 非预期的最低 LR。不影响同配置的相对比较, 但绝对 mAP 可能偏低。

### C-MD-1: AMP 下 value stream NaN 未防护
**文件**: `opentad/models/projections/irregular_actionformer_proj.py:154-164`
**状态**: CONFIRMED — 稳定性风险
**影响**: 注意力块的 `nan_to_num` 只防护 attention weight, 不防护 value tensor。在 AMP float16 下可能导致 NaN 传播。

### C-CF-1: `remap=False` + `regression_range` 假设 remap=True
**文件**: `configs/adatad/thumos/input_random_fixed_50pct_irregular_actionformer.py` 及 ~30 个子 config
**状态**: 存疑 (Codex 认为不解释 ~50% 天花板)
**影响**: regression_range 在物理坐标空间应该更大 (约 2x), 但对 soft assignment 为主的 head 影响有限。

### C-HD-1: `NativePhysicalMultiScaleHead` 重复注册
**文件**: `native_physical_multiscale_head.py` 和 `native_physical_multi_scale_head.py`
**状态**: CONFIRMED — registry 覆盖问题
**影响**: 两个不同实现用同一类名, 后 import 的覆盖前一个。R05D_V2/V3 用的哪个不确定。

---

## 🟡 MAJOR 问题

### M-CF-1: work_dir 冲突
`denseexact_base.py`, `headv2_base.py`, `headv3_base.py`, `headv3_oabs_x_base.py` 未设置 work_dir, 写入父 config 的目录。

### M-HD-1: Loss normalizer 单位不一致
`AnchorFreeHead` 用 `num_pos`(整数), `HeadV2/V3/Bridge` 用 `pos_mass`(分数)。跨 head 类型的 loss 比较不可靠。

### M-TR-1: pre-backward 非有限 cost 时 `scaler.update()` 误调用
`train_engine.py:109` — 在 backward 之前就调用 scaler.update(), 侵蚀 AMP scale。

### M-HD-2: Different assignment semantics across heads
AnchorFreeHead 用最短 duration 分配; IrregularActionFormerHead 用 scale-cost 分配。跨 head 类型实验的分配差异是混杂变量。

### M-DS-1: random_fixed_subsample mask 长度不一致
`end_to_end.py:805,809` — valid_mask_len 用 `max(scale_factor,1)` 但 target_mask_len 用 `scale_factor`。当前 scale_factor=1-8 时未触发。

---

## 🟢 MINOR 问题 (汇总)

- `optimizer.py:54`: `raise` on string literal → TypeError
- `scheduler.py:109`: warmup_epoch=1 时除零
- `backbone_wrapper.py:237`: mask `.detach()` 切断 learned-scorer 梯度
- `formatting.py:141`: torchvision_align 精度损失 (float32→float16→float64)
- `loading.py:64-66`: bare except 吞所有异常
- `checkpoint.py:19`: `os.mkdir` 不用 `os.makedirs`
- `train_engine.py:198`: `max_memory_allocated` 从不 reset
- DIoU loss 对长 segment 的 center penalty 近乎消失
- FocalLoss 对 soft target 的 p_t 计算使 modulating factor 永不归零
- `IrregularPointGenerator` V1 的 left_scale=right_scale (丢失不对称性)

---

## Codex 修复优先级

1. **DenseAdapter 坐标轴修复** (当前最高优先级 — A0 实验中)
2. **C-MD-1 + AMP skip 处理**: 给 v_window 加 NaN 防护, 移除 pre-backward scaler.update()
3. **C-TR-1 调度器 horizon**: 新实验匹配 end_epoch 或训练 100 epochs
4. **C-CF-1 config linting**: 添加坐标轴一致性 assert
5. **数据集边界修复 (C-DS-1/C-DS-2)** + 添加 dropped GT 计数
6. **NativePhysical 重复注册 / work_dir / normalizer 清理**

## 已确认不影响实验结果的发现

- Codex 判断 C-CF-1 (remap+reg_range) 不解释 headv2/headv3 的 ~50% 天花板
- Codex 判断 C-DS-1 (GT drop) 触发条件罕见
- Codex 判断 C-TR-1 不影响同配置下的相对比较
- DenseAdapter 的 cross-axis interpolation bug 仍然是 14-15% 的主因
