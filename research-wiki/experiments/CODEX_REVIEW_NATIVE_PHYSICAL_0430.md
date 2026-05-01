---
type: experiment
node_id: exp:CODEX_REVIEW_0430
title: "Codex GPT-5.5 深度审查: NativePhysicalMultiScaleHead 47% 根因"
status: completed
created_at: 2026-04-30
updated_at: 2026-04-30
---

# Codex GPT-5.5 审查: 为何 NativePhysicalMultiScaleHead 只有 47%

## 审查范围

- IrregularPointGenerator (regression_range + point_scale 计算)
- NativePhysicalMultiScaleHead (prepare_targets + get_refined_proposals)
- IrregularFPN (top-down pathway)
- linear_interpolate_features

## 审查结论

### 1. Regression 解码一致性: ✅ 无 factor-of-2 错误

- DIOU loss 端到端学习是正确的
- 最优解等价于 `reg_target = physical_distance / point_scale`
- 各层 scale 差 32 倍会导致梯度不均衡，但不是致命问题
- 建议: 监控 per-level reg_pred/grad, 必要时加 per-level Scale 或 log/exp 编码

### 2. 根因诊断: 🔴 共享 temporal conv 假设与不均匀时间轴不兼容

> **最可能根因: head 的共享 temporal conv 假设各层采样间隔一致, 但 sparse 物理时间轴不均匀且跨层尺度差异巨大**

具体问题:
- 共享 `kernel=3` 在 L0/L5 对应完全不同物理感受野 (L0≈12, L5≈384 物理单位)
- 不均匀相邻点让普通 conv 把远近不同的帧当等距邻居混合
- L5 无监督特征经 top-down 插值传递, 可能放大噪声

### 3. 方案判断: 7% 差距是结构性的, 不只是超参

> A 的 7% 差距更像结构性问题, 不只是超参

- FCOS head 默认等间距、固定 stride、卷积平移等价性
- 不均匀轴破坏正负样本分配、中心度、尺度归属和卷积平移等价性
- nearest-completion 先恢复均匀网格, 标准 ActionFormer 的等距假设成立

### 4. DenseAdapter 14% 异常低: 🔴 坐标/配置错配 bug

> 不应归因于 IrregularFPN 本身。NativePhysical=47 说明 IrregularFPN 可用; Bridge=43 说明 sparse+dense head 可用。14% 指向坐标/监督链断了。

## 推荐下一步

1. **排查 DenseAdapter 14% bug** — 修好后可能大幅提升
2. **方案C (IrregularFPN + DenseAdapter + 标准 dense head)** — 如果修好 bug, 有望接近 54%
3. **方案D (query-based detector / deformable temporal sampling)** — 避免 dense FCOS 假设
4. 如果坚持 NativePhysicalHead, 需要改用 geometry-aware conv (如 deformable conv) 替代共享 kernel=3
