# Gap Map

## G1. Learned sampler still loses badly to uniform 50%
- Status: unresolved
- Evidence: R008 (uniform 50%) = 63.74, while R017/R018/R021 all trail clearly.
- Question: is the bottleneck scorer optimization, train-test gap, or detector/backbone mismatch?

## G2. Missing oracle upper bound / policy headroom estimate
- Status: unresolved
- Evidence: without oracle 50%, we still cannot separate “sampler did not learn” from “50% intelligent sampling has no real headroom”.
- Needed: oracle mask experiment with the same V3 backbone and detector stack.

## G3. Dropped-token stale feature pollution in projection / adapter
- Status: active
- Evidence: coarse-inject and projection-aware lines were proposed because dropped positions are currently stale or weakly marked before projection.
- Needed: indicator / bias / injection style projection-aware experiments on top of the strongest stable baseline.

## G4. Dense ActionFormer prior mismatches sparse / irregular timelines
- Status: active
- Evidence: hole / compressed / adaptive-compressed / sparse-head lines repeatedly produced bad learning or NaN.
- New evidence: input-side fresh-feature controls now show `uniform stride-2 50% = 65.09` vs `random-fixed 50% = 63.12`. The random-fixed mask still has high global coverage, so the gap is more consistent with irregular time geometry mismatch than coverage collapse.
- Correction from upstream audit (2026-04-16): the current `x`-line `GridAware...` path in `OpenTAD_Back` is still feature-dense enough that it cannot be used as evidence for a truly irregular-aware trunk. This means part of the previous "irregular trunk plateau" reading was an interpretation error rather than a settled negative result.
- Needed: sparse-aware projection and detector semantics instead of directly reusing dense assumptions.

## G5. Efficiency evidence is incomplete
- Status: unresolved
- Evidence: current comparisons emphasize mAP more than real FLOPs / latency / throughput.
- Needed: a small, consistent efficiency audit on the main surviving lines.

## G6. mask_adapter 设计方向需修正
- Status: resolved (R015)
- Evidence: R015 (mask_adapter=False) = 64.51% > R008 (mask_adapter=True) = 63.74%. 零化 dropped token 反而有害。
- Resolution: 后续实验默认 mask_adapter=False。

## G7. 所有可微梯度方案在 ~54% 收敛，瓶颈不在梯度
- Status: active
- Evidence: Gumbel-ST/REINFORCE/ST-TopK 四种方案全部收敛到 53-55%，远低于 uniform 64.51%。
- Question: 瓶颈到底是 scorer 能力不足、TAD 任务特性、还是训练耦合？Oracle 实验是关键。

## G8. MobileNet scorer 提升有限
- Status: active
- Evidence: R022-mobilenet (58.49%) 比轻量CNN scorer (57.55%) 仅高 ~1%，仍远低于 uniform。
- Implication: 预训练特征不能根本解决选帧问题。

## G9. NativePhysicalMultiScaleHead 结构性 7% 差距
- Status: active (2026-04-30)
- Evidence: R05D_V2 = 47.04%, nearest completion = 54.03%, 差距 6.99%
- Codex GPT-5.5 审查: 共享 temporal conv (kernel=3) 在不均匀物理时间轴上假设等间距, 破坏 FCOS 平移等价性
- regression_range 修改 (v3) 未改善, 确认问题在 conv/head 架构假设
- Needed: geometry-aware conv, deformable temporal sampling, 或 query-based detector

## G10. DenseAdapter 14% 异常 — 坐标链 bug 待排查
- Status: active (2026-04-30)
- Evidence: IrregularFPNDenseAdapter + ActionFormerHead = 14%, 远低于 Bridge=43% 和 NativePhysical=47%
- Codex 判断: 坐标语义/配置错配 bug, 不是 IrregularFPN 特征问题
- Needed: 排查 DenseAdapter 的坐标映射、GT remap 与 head 的 stride/regression_range 是否匹配
- Priority: **最高** — 修好后可能直接突破 47% 天花板

## G11. DenseAdapter 跨轴插值 bug (2026-04-30)
- Status: **confirmed**
- Evidence: B0 坐标验证 + B1 stride fix = 15.46% (+1.6%) + Codex 全链审计
- Root cause: `linear_interpolate_features(source_grid=physical[0,768), target_grid=selected[0,s*L))` — searchsorted 在不同坐标轴之间搜索
- Fix: A0 no-interp (running) 或 correct selected→physical remap before interpolation
- Implication: 任何跨轴特征重采样在概念上都是错误的; 需要保持坐标轴一致
