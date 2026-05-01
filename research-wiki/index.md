# Research Wiki Index

## Project

- Topic: temporal redundancy reduction / intelligent frame selection / irregular timeline adaptation for TAD
- Main codebase: `OpenTAD_Back`
- Active route separation rule: `OpenTAD_Back` is the main execution repo; `OpenTAD` is a separate frame-sampling / scorer reference repo
- Cross-repo guardrail: compare metrics within the same repository first; only use cross-repo numbers as context, not as direct mainline scoreboards
- Unverified cross-repo anchor: `exp:R015 = 64.51` is still not remotely revalidated and must not be treated as authoritative
- Strongest input-side 50% result so far: `INPUT_ORACLE_BOUNDARY_DENSE_0411 = 66.01`
- Strongest dense random-fixed baseline: `INPUT_RANDOMFIX_0410 = 63.12`
- Strongest current irregular-head result: `HEADV3_OABS_FULL_X_LEGACYSINGLETON = 53.48`
- Current authoritative code-level correction: `IRREGULAR_UPSTREAM_AUDIT_0416`

## Papers

- `paper:wang2024_adatad`
- `paper:rao2021_dynamicvit`
- `paper:wang2023_adafocus`

## Ideas

- `idea:001` soft gating scorer
- `idea:002` coarse inject
- `idea:003` REINFORCE
- `idea:004` projection sparse adaptation
- `idea:005` two-stage training
- `idea:006` oracle upper bound
- `idea:007` soft attention bias
- `idea:008` partial conv
- `idea:009` projection all-in-one
- `idea:010` ablation / diagnosis
- `idea:011` irregular-aware detector
- `idea:012` incremental irregular detector

## Key Experiments

### OpenTAD_Back authoritative mainline

- `exp:INPUT_STRIDE2_0410` uniform input-side 50% baseline: `65.09`
- `exp:INPUT_RANDOMFIX_0410` random-fixed input-side 50%: `63.12`
- `exp:INPUT_ORACLE_BOUNDARY_DENSE_0411` boundary oracle 50%: `66.01`
- `exp:INPUT_ORACLE_ACTION_BOUNDARY_DENSE_0411` action+boundary oracle 50%: `62.04`
- `exp:IRREGULAR_TIMELINE_0412` audited old irregular detector: `39.04`; native remap failure: `4.13`
- `exp:HEADV2_0413` HeadV2 recovery split: `headv2_x = 51.04`, `headv2_y = 49.00`
- `exp:HEADV2_X_TOPK1` single-best assignment ablation: `47.19`
- `exp:HEADV3_0413` HeadV3 line: `headv3_x = 52.60`, `headv3_y = 49.37`; first stable positive V3 recovery over HeadV2
- `exp:SPARSE_HEAD_SUMMARY_0414` consolidated sparse-head trajectory and current Step0-4 decomposition status
- `exp:IRREGULAR_MODEL_AUDIT_0415` full local-vs-server audit table for the `irregular_actionformer` model line: trusted runs, invalid runs, deployed queues, and missing deployments
- `exp:IRREGULAR_COMPLETION_UPDATE_0415B` legacy completed-batch note for `OABS-visible = 51.49` and `OABS-full = 53.16`; its old `step0_shell_exact` reading is later corrected by `IRREGULAR_UPSTREAM_AUDIT_0416`
- `exp:IRREGULAR_ROOTCAUSE_UPDATE_0416` historical root-cause page for `step0b = 47.52` and `OABS-full+OAA = 53.20`; its old shell-overhead and `fixsingleton` readings are later corrected by `IRREGULAR_UPSTREAM_AUDIT_0416`
- `exp:IRREGULAR_UPSTREAM_AUDIT_0416` fresh upstream-vs-local code audit: upstream ActionFormer path is intact, `x` line is currently dense-trunk + irregular-head, and `step0_shell_exact / fixsingleton` are not clean causal controls
- `exp:IRREGULAR_REPAIR_QUEUE_0416` repaired queue final tracker: `step0_dense_head_baseline = 53.72`, `step0b_repaired = 47.52`, clean singleton A/B closes at `53.48 -> 52.88`, and `y_dense_grid_sanity_check = 13.86`
- `exp:TEMPORALGRIDFIX_FULL_QUEUE_0423` completed formal rerun audit for the repaired `temporal_grid.py` on `25876`: final read `52.40`, the config inheritance chain is confirmed clean, and this carrier still validates `BUG-1/2` only rather than the separate `BUG-3` weighting story
- `review:GEMINI_DIRECTION_0417` corrected Gemini direction review after `y_dense_grid_sanity_check = 13.86`: strongest supported diagnosis is now `true irregular y-trunk` vs current dense-head decoding contract mismatch
- `review:GEMINI_DIRECTION_0418` reset review after `10.34 / 44.06`: stop polishing middle-state hybrids, build a stupid dense-semantics anchor, simplify the native `y` line, and use oracle diagnostics before more module growth
- `exp:Y_RECOVERY_QUEUE_0417` closed recovery queue read: `y_dense_grid_sanity_check_norm_ampoff = 10.34` is complete and negative, `y_pointset_softsym_sanity = 44.06`, and `y_pointset_softsym_sanity_ampoff = 44.07`, which rules out AMP as the main blocker
- `exp:RESET_QUEUE_0418_STATUS` current 0418 reset ledger: `R001 = 44.07`, `R002 = 54.03`, `R003` interrupted at `52.56`, old `R004/R005` invalid, and the next live priority is `R006 -> R008 -> R007` plus repaired overfit `R009/R010`
- `exp:SOFT_BRIDGE_MAINLINE_0420` historical full-train verification page for the native `y` soft bridge: it froze the weaker `topk1` mainline at `36.59` vs `37.00` and motivated the later `topk9` transfer pair
- `exp:SOFT_MULTIGT_DIAG_QUEUE_0420` historical toy / multigt diagnostic ladder that established the failure boundary at `2 GT` and motivated the later full-train transfer tests
- `exp:SOFT_BRIDGE_SUPERVISION_CLOSURE_0423` now fully closes the native-`y` soft-bridge branch: supervision closure freezes `topk9 + binary-cls = 43.44`, while the final `0424` transfer pair reads `regwarm10 = 43.05` and `detach-cls = 38.56`, so bridge-side polishing is no longer justified on this carrier
- `exp:INPUT_WEIGHTED_RANDOM_ACTION_BOUNDARY_TUBELET2_0413` tubelet2 weighted random action+boundary: `58.54`
- `exp:PENDING20260410` current pending / unfinished matrix
- `exp:REPO_ROUTE_SEPARATION_0424` authoritative route-separation note: `OpenTAD_Back` is mainline, `OpenTAD` is reference-only, and `R015` remains unverified

### NativePhysicalMultiScaleHead line (2026-04-29 ~ 04-30)

- `exp:R05D_V2` NativePhysicalMultiScaleHead full pilot: **47.04%** (60 epochs, completed)
  - Per-level positive count: L0-L2 healthy, L3 partial, L4 sparse, L5=0
  - mAP@0.3=66.91% (分类强) vs mAP@0.7=16.36% (定位差)
- `exp:R05D_V3` regression_range 修正实验: **45.80%** at epoch 44 (stopped early, 未改善)
  - 修改 regression_range 从 (1,4)→(0,5), 结果反而更低, 说明不是 regression_range 的问题
- `exp:DENSEADAPTER_SANITY` IrregularFPNDenseAdapter + 标准 ActionFormerHead: **14%** (🔴 异常低, 疑似 bug)
- `exp:CODEX_REVIEW_0430` Codex GPT-5.5 审查结论:
  - 47% 差距是结构性的: 共享 temporal conv 假设等间距, 不均匀轴破坏 FCOS 假设
  - DenseAdapter 14% 更像坐标/配置 bug, 不是 IrregularFPN 特征质量问题
  - 推荐: 排查 DenseAdapter bug → 方案C (IrregularFPN + DenseAdapter 修好 + dense head)
- `exp:RESULT_TO_CLAIM_0430` Codex result-to-claim 裁决 (2026-04-30):
  - C12 (NativePhysical 结构性差距) → **partial**: regression_range 排除, 但共享 kernel=3 因果未直接验证
  - C13 (DenseAdapter 14% 是 bug) → **supported**: 一致判定为坐标链 bug
  - **NativePhysicalMultiScaleHead 路线正式关闭**, R05D_V3 提前停止
  - **最高优先级: 排查 DenseAdapter 14% bug** → 修好后目标 ~52-54%
  - 备选: Candidate B (query-based / deformable detector)

### OpenTAD reference line

- `exp:R015` current V3 best: `64.51` (`OpenTAD`, still unverified)
- `exp:FRAME_SAMPLING_BUGFIX_QUEUE_0423` completed `OpenTAD` bugfix summary: `v3_hard_scorer_aux_bugfix0420 = 56.97`, `v3_gumbel_softgate_minimal_bugfix0420 = 54.22`, `v3_gumbel_softgate_coarse_det_bugfix0420 = 53.17`

## Claims

- `claim:C11` non-uniform sampling is not inherently doomed; boundary-focused allocation is useful, action-interior densification is not supported
- `claim:C12` NativePhysicalMultiScaleHead 7% 差距是结构性问题 — **partial**
- `claim:C13` DenseAdapter 14% 是坐标/配置 bug — **supported**
- `claim:C14` DenseAdapter 跨轴插值 bug — **supported** (Codex 审计确认)
- `claim:C15` 跨轴bug确认 + 不规则几何信息有害 (43-51% < 63%) — **supported** (A0=51.28% confirmed)

## Scope Note

- Unless a line explicitly says `OpenTAD`, the judgments below default to `OpenTAD_Back`.

## Current Structural Judgment

1. Non-uniform sampling itself is not the main enemy; random-fixed 50% is only `1.97 mAP` below uniform.
2. Boundary-focused budget allocation is the strongest input-side signal; boundary oracle even beats uniform.
3. Action-interior densification is consistently weak:
   - oracle action+boundary: `62.04`
   - weighted-random action+boundary: `59.24`
   - weighted-random action+boundary tubelet2: `58.54`
4. Tubelet-aware selection does not by itself rescue the action+boundary line, so the issue is more likely semantic budget allocation than frame-vs-tubelet mismatch.
5. HeadV2 repaired the main semantic collapse, and the current best completed repaired head-side carrier is `53.48`, but this best `x` result is still on a **dense-trunk + irregular-head** path rather than a full irregular-aware trunk.
6. `headv2_denseexact = headv2_x = 51.04` now has a corrected reading: under the current code, the `x`-line `GridAware...` path is feature-dense enough that it matches the dense passthrough control.
7. repaired `step0_dense_head_baseline = 53.72` restores the remapped selected-axis dense-head bridge after fixing the GT-axis mismatch bug, but it still should **not** be used as no-remap shell-overhead evidence.
8. repaired `step0b_dense_points_soft_sym_repaired = 47.52` stays `6.20 mAP` below repaired Step0, so soft assignment remains strongly negative inside that remapped control family even though the final no-remap paired control is still missing.
9. `step1 = 37.10` and `step2 = 34.41` still show that naive hard semantics on the native irregular axis are unstable, but that conclusion is restricted to the bridge-head family that was actually run.
10. The clean singleton A/B is now closed at `53.48 -> 52.88`; singleton repair is slightly harmful on the current carrier and is no longer a serious root-cause candidate.
11. `y_dense_grid_sanity_check` has now finished at only `13.86`, so the strongest current structural conclusion is no longer provisional: the true irregular `y` trunk is severely incompatible with dense-head decoding under the current dense-grid adapter path.
12. This sharply separates carrier families: remapped selected-axis dense-like controls can still reach `53.72`, while the true irregular trunk collapses to `13.86`; the main unresolved issue is therefore trunk/head semantic mismatch, not a small head-side ablation.
13. The latest corrected Gemini review now ranks the current root causes as:
   - `y`-trunk vs dense-head semantic mismatch
   - `y`-output distribution / optimization instability
   - `x` line being the wrong carrier for proving irregular benefit
   - soft assignment as a secondary but still real negative factor
   - bridge / remap overhead as unresolved rather than established
14. The 0417 recovery queue has now already closed one branch decisively: `y_dense_grid_sanity_check_norm_ampoff = 10.34` is even worse than `13.86`, so simple dense-grid normalization is not a meaningful recovery path on the true `y` trunk.
15. The first completed point-decoder recovery pair is now `44.06` and `44.07`, so the dense-grid decoder mismatch is confirmed while the AMP-rescue hypothesis is effectively ruled out.
16. The stupid dense-semantics nearest-completion anchor has now landed at `54.03`: directionally better than current irregular-head results, but still far from `60+`.
17. The first malformed tiny-overfit ladder is now only historical context; the later branch-decoupled and multigt diagnostics already clarified the real failure source more cleanly.
18. The 0418 Gemini reset review still stands in its broad message: stop polishing middle-state hybrids before establishing clean dense-semantics-preserving anchors.
19. The best sparse-to-dense anchor currently on record is still `completion_nearest = 54.03`: better than current irregular-head / bridge carriers, but still clearly below `random_fixed dense = 63.12`.
20. Completion therefore remains a stronger practical pivot than more bridge tuning, but it still needs cleaner upper-bound and learned-completion evidence before it can be treated as the main recovery answer.
21. The native-`y` soft-bridge verification line is no longer active; it is now a closed historical branch after the negative `0424` transfer pair.
22. The earlier reg-warm implementation-stability observation remains valid as a code-path fact, but it did not convert into a better final result on the stronger `topk9` carrier.
23. Operational priority has therefore shifted away from maintaining bridge runs and toward comparing dense-semantics-preserving pivots against any future native irregular decoder redesign.
24. The multigt diagnostics should now be read as causal evidence rather than an active queue: they established the failure boundary at `2 GT` and motivated the transfer tests that have since closed negative.
25. The key next question is no longer how to tweak `topk1/topk9` warmup schedules; it is whether to preserve dense semantics end-to-end or replace the current decoder / support contract.
26. The old `36.59 -> 37.00` `topk1` warmup gain is now just historical context; once transferred onto the stronger `topk9` carrier it collapsed to a near-neutral `43.44 -> 43.05`.
27. The formal `headv3_x_temporalgridfix` rerun is now complete: `52.40` does not beat the old `52.60`, so repaired temporal-grid semantics alone do not lift the current `x` carrier.
28. That completed run should still be interpreted narrowly: it validates `BUG-1/2` on the current `GridAware...` path, but not the separate `BUG-3` question, because this carrier still uses dense `MaxPool1d` for feature downsampling rather than `IrregularConvTransformerProj.downsample_features`.
29. `OpenTAD:` the completed minimal bugfix rerun now closes at `54.22` after peaking at `54.32`; this is a small upward correction over old `R021 = 53.98`, but it still leaves the soft-gating minimal family in the same broadly weak regime.
30. `OpenTAD:` the strongest completed bugfix branch is now `v3_hard_scorer_aux_bugfix0420 = 54.42 -> 55.11 -> 55.76 -> 56.00 -> 56.53 -> 56.94 -> 56.87 -> 57.00 -> 56.97`; this is materially better than the minimal soft-gating rerun, but it is still well below the `uniform 50%` reference.
31. `OpenTAD:` the full `coarse_det` branch now closes at `53.17`; despite recovering from a very weak opening, it still underperforms the simpler minimal rerun `54.22` and therefore remains a negative direction in this bugfix family.
32. `headv3_x_temporalgridfix` closes with five sparse skipped reg-head steps but still completes at `52.40`, so numeric instability existed but was not the sole cause of the weak final result.
33. The native-`y` soft-bridge supervision closure is now also complete: `topk9 + binary-cls = 43.44` beats `topk3 + soft-cls = 42.26`, while both `topk1 + soft-cls` and the improved `topk1 + binary-cls` stop at `38.98`; support size is therefore the dominant full-train lever on this carrier.
34. Toy closure sharpens the causal story: original soft `reg-only` fails (`0.6121 / 0.5840`), but oracle-point, hard assignment, `topk1`, and `topk2` reg-only all fit (`0.0009 / 0.0103 / 0.0079 / 0.0128 / 0.0161`), so raw regression capacity is not the blocker; diffuse soft support is.
35. The first catastrophic failure boundary is now explicit: `topk1 + binary-cls joint single_instance = 0.0037`, but `single_video joint = 1.0141`, and `keep2gt joint = 1.0141` already fails at two GTs; multi-GT joint competition is the causal break.
36. `detach-cls` and `regwarm` are both toy-proven rescues: `single_video joint + detach-cls = 0.0480`, `keep2gt joint + detach-cls = 0.0480`, `regwarm100 = 0.0055`, while pure cls-weight reduction alone is weaker (`0.4783`).
37. The `0424` topk9 repair pair is now closed and negative: `regwarm10 = 43.05` is nearly neutral relative to `43.44`, while `detach-cls = 38.56` is clearly harmful.
38. Support size remains the only full-train lever that clearly transferred on the current native-`y` bridge carrier; toy-proven joint rescues did not materially survive full-train deployment.
39. The native-`y` soft-bridge line should therefore be treated as closed at a practical ceiling of `43.44`, not as an active polishing target.
40. If native irregular decoding is revisited, the next method must change the decoder / support contract itself; otherwise the better use of time remains sparse-to-dense completion anchors or stronger non-bridge carriers.
41. The immediate post-bridge execution order is now completion-first: `oracle_upper_ampoff`, then `full_oracle_dense_control`, then `learned_minimal_ampoff`.
42. The key route gate is no longer "can bridge tuning improve?" but "does completion have real headroom above `54.03`, and can a minimal learned completion close a meaningful fraction of that oracle gap?"
43. `NativePhysicalMultiScaleHead` (Candidate A) now has a completed full-train result: `47.04%`, which is `+3.6` over bridge `43.44` but still `-6.99` below nearest-completion `54.03`.
44. Codex GPT-5.5 审查 (2026-04-30) 判断: 7% 差距是**结构性的**, 不只是超参。共享 temporal conv (kernel=3) 在不均匀物理时间轴上假设等间距, 破坏了 FCOS 的平移等价性、正负样本分配和尺度归属。
45. `regression_range` 修改 (v3) 未改善, 进一步证实问题不在 assignment range, 而在 conv/head 架构假设。
46. `DenseAdapter` sanity check = `14%` 极度异常, Codex 判断为坐标/配置 bug 而非 IrregularFPN 特征质量问题。**排查此 bug 是当前最有价值的实验方向**。
47. **NativePhysicalMultiScaleHead 路线正式关闭** (result-to-claim 0430): R05D_V3 @ epoch 44 = 45.80% 低于 v2's 47.04%, 确认 regression_range 不是瓶颈; Candidate A M1 overfit 暂不执行。
48. claim:C12 → **partial**: regression_range 假设被排除, 但共享 kernel=3 的因果机制尚未通过直接干预实验验证。
49. claim:C13 → **supported**: 14% 一致判定为坐标/配置链 bug, 非 IrregularFPN 特征质量问题。
50. claim:C14 → **supported**: Codex 全链审计确认根因为 `searchsorted(physical, selected)` 跨轴插值——DenseAdapter 在不同坐标轴之间做特征重采样。
51. **A0 no-interp 最终 51.28% (epoch 59)**: 消除跨轴插值 bug 后从 14%→51.28% (+37%)。是所有 irregular→dense 路径中最强结果。mAP@0.5=54.17%, @0.7=26.21%。
52. **全面代码审计完成** (doc:CODE_AUDIT_0430): 26 问题 (5C/7M/14m), 6 已修复并 commit。
53. **关键洞察**: 忽略不规则性 (dense baseline 63.12%) > 使用不规则信息 (43-51%)。问题在 IrregularConvTransformerProj/IrregularFPN 的几何感知破坏均匀假设。
54. **Plan A (Query-based Sparse Detector)** v5 训练中 (Server1): DETR-style, 目标超越 dense baseline。Plan B (Physical Segment Proposal) 排队。
55. **最终隔离矩阵完成 (2026-05-02)**: TadTR_ConvSingleProj=30.82%, TadTR_DensePassthrough=28.45%, DenseProj+IrregularFPN=37.79%。DETR架构被证实不适合稀疏TAD。
56. **核心结论**: 几何感知Projection和Neck各自有害。DensePassthrough+FPNIdentity=63.12%是唯一接近目标的路径。
57. **下一步**: 基于完整结果矩阵撰写分析论文 "TAD时序冗余边界研究"。
54. **当前实验优先级**: (1) A0 no-interp 结果; (2) >25% → 正确 DenseAdapter; (3) <25% → Candidate B + 调度器/AMP 修复。
