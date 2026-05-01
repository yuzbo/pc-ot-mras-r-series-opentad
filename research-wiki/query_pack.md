# Query Pack

## Project direction
Target: determine whether intelligent sparse frame selection for TAD can approach full-frame performance, and separate sampler failure from dense-detector incompatibility with irregular timelines.
Repository split rule: `OpenTAD_Back` is the active mainline for input / irregular-detector / completion work; `OpenTAD` is a separate frame-sampling / scorer reference line. Do not mix their scoreboards by default, and treat `R015 = 64.51` as unverified.

## Top gaps
1. `G4`: sparse-input TAD still lacks a clean path that preserves dense-detector semantics on a truly irregular-aware detector path.
2. `G2`: still missing a fully trusted oracle completion upper bound for the sparse-to-dense pivot.
3. `G1`: learned sampler lines still lag behind strong fixed-input controls.
4. `G5`: efficiency evidence is still incomplete; most conclusions remain accuracy-driven.
5. current diagnosis still lacks a clean no-remap shell-exact control and a matching no-remap dense-points soft-assignment control; the repaired remap controls are now stable but do not replace them.

## Failed / weakened lines
- soft gating / Gumbel lines plateaued around low-to-mid 50s
- latest bugfix rerun is slightly stronger than the early snapshot: `v3_gumbel_softgate_minimal_bugfix0420` has now completed at `52.43 -> 53.17 -> 53.28 -> 53.02 -> 53.75 -> 54.32 -> 54.07 -> 53.87 -> 53.73 -> 54.22`, so the fixes exposed some extra headroom but still did not rescue the line out of the low-to-mid `50s` regime
- REINFORCE did not break the learned-sampler ceiling
- old irregular detector collapsed or stayed weak (`39.04`, `4.13`)
- repaired `step0_dense_head_baseline = 53.72` proves the old `13.79` collapse was a GT-axis mismatch bug, but it is still only a remapped selected-axis bridge control
- repaired `step0b_dense_points_soft_sym_repaired = 47.52` sits `6.20` below repaired Step0, so soft assignment remains a real negative factor inside that remapped control family
- `y_dense_grid_sanity_check = 13.86` closes the strongest open sanity check in the worst possible way: the true irregular trunk remains extremely weak even after dense-grid adaptation
- corrected Gemini 0417 review now makes the strongest diagnosis explicit: the main failure is `true irregular y-trunk` vs current dense-head decoding contract mismatch, not another small head-side ablation
- `y_dense_grid_sanity_check_norm_ampoff = 10.34` is now complete and negative, so simple dense-grid output normalization is not a meaningful rescue path
- Gemini 0418 reset review adds a stricter process conclusion: stop polishing middle-state hybrids before establishing a stupid but clean dense-semantics-preserving anchor
- action-interior densification is repeatedly weak:
  - oracle action+boundary `62.04`
  - weighted-random action+boundary `59.24`
  - weighted-random action+boundary tubelet2 `58.54`
- `headv2_x_topk1 = 47.19` shows that shrinking each GT's support from `top9` to `top1` hurts the current irregular detector
- OABS / OAA only move the current dense-trunk irregular-head carrier from `52.60` to about `53.2`, so they are positive but not mainline breakthroughs

## Top experiments
- `OpenTAD_Back authoritative`: `INPUT_ORACLE_BOUNDARY_DENSE_0411 = 66.01`, `INPUT_STRIDE2_0410 = 65.09`, `INPUT_RANDOMFIX_0410 = 63.12`, `completion_nearest_dense_baseline = 54.03`, `HEADV3_OABS_FULL_X_LEGACYSINGLETON = 53.48`
- `OpenTAD_Back route note`: `IRREGULAR_UPSTREAM_AUDIT_0416`, `REPO_ROUTE_SEPARATION_0424`
- `OpenTAD reference-only`: `v3_hard_scorer_aux_bugfix0420 = 56.97`
- `OpenTAD unverified`: `R015 = 64.51`

## Active chains
- random-fixed 50% only drops `1.97` from uniform, so non-uniformity is not inherently fatal
- boundary oracle beats uniform, so "which frames to keep" matters
- upstream code audit shows the original dense ActionFormer path is intact; the main issue is semantic drift in the irregular branch, not global baseline breakage
- the current `x` line is a dense-trunk + irregular-head carrier, because `GridAware...` currently propagates grid metadata more than it changes the feature path
- `headv2_denseexact = headv2_x = 51.04` is therefore explained by shared feature behavior, not by a clean irregular-vs-dense trunk comparison
- action+boundary remains worse than boundary-only even under tubelet-aware sampling, but this still mixes two effects:
  - budget may be spent in the wrong temporal regions
  - tubelet granularity may blur boundary precision
- repaired `step0_dense_head_baseline = 53.72` restores the historical remapped bridge line and confirms the old `13.79` was invalid
- this does not overturn the audit: repaired Step0 is still a remapped selected-axis control, not a no-remap shell-exact control
- repaired `step0b = 47.52` remains `-6.20` vs repaired Step0, so soft assignment is still strongly negative in the remapped control family
- clean singleton A/B now closes at `53.48 -> 52.88`, so singleton repair is not a useful lever on the current `x`-line carrier
- `step1 = 37.10` shows naive hard semantics on the native irregular axis collapse badly, so the clean pivot is neither "keep soft assignment" nor "go back to hard assignment on irregular points"
- the best current shell-local fix is still only `53.48`, so OABS/OAA-like improvements remain small carrier-side gains rather than a route back to `63+`
- `y_dense_grid_sanity_check = 13.86` now makes one conclusion explicit rather than tentative: the true irregular trunk is severely incompatible with dense-head decoding under the current dense-grid adapter path
- corrected Gemini priority order has now tightened again:
  - relaunch oracle interface diagnostics
  - rerun a valid tiny-overfit ladder on the native `y` decoder
  - keep completion as a parallel mainline because the stupid nearest baseline already reached `54.03`
  - stop expecting fp32 alone to rescue the current `y` point decoder
- first live `y`-recovery deployment on 0417 immediately showed AMP instability on both new sanity lines, so future `y` conclusions must separate method effect from fp16 numeric instability
- first completed `y` recovery result is now `y_pointset_softsym_sanity = 44.06`, which is far above `y_dense_grid_sanity_check = 13.86`; this makes dense-grid decoder mismatch a confirmed major failure source rather than a hypothesis

## Current priorities
1. **Plan A (Query-based Sparse Detector)** v5 训练中 (Server1) — DETR-style, 目标超越 63.12%
2. Plan B (Physical Segment Proposal) 排队 — per-point MLP + physical coords
3. **A0 no-interp 最终 51.28%** — 最强 irregular→dense, 但确认天花板 ~51%
4. **关键洞察**: 忽略不规则性 (63.12%) > 使用不规则信息 (43-51%) — FCOS 补丁已达极限
5. **NativePhysicalMultiScaleHead 关闭** — 共享 conv 结构性缺陷

## Latest live read
- `NativePhysical R05D_V2 = 47.04%` (completed)
- `NativePhysical R05D_V3 = 45.80%@e44` (stopped early, regression_range fix did NOT improve)
- `completion_oracle_upper = 58.76`, `completion_full_oracle_dense = 59.22`, `completion_learned_minimal = 53.51`, `completion_scatter = 52.55`
- **NativePhysicalMultiScaleHead 关闭** (result-to-claim 0430)
- **A0 no-interp = 51.28%** — DenseAdapter 跨轴bug已修复, +37% vs bug版本
- **Plan A v5 训练中** (Query-based Sparse Detector), Plan B 排队
- Server 2 (25876) 不可达
- closed `0420` supervision results:
  - `topk1 + binary-cls = 38.98`
  - `topk1 + binary-cls + regwarm10 = 37.64`
  - `topk1 + soft-cls = 38.98`
  - `topk3 + soft-cls = 42.26`
  - `topk9 + binary-cls = 43.44`
- closed toy joint diagnosis:
  - original soft `reg-only` fails at `0.6121 / 0.5840`
  - `oracle-point reg-only = 0.0009`
  - `hardjoint single_instance = 0.0103`
  - `hardjoint single_video = 0.0079`
  - `single_video joint = 1.0141`
  - `single_video joint + detach-cls = 0.0480`
  - `keep2gt joint = 1.0141`
  - `keep2gt joint + detach-cls = 0.0480`
  - `regwarm100 = 0.0055`
- `temporalgridfix_full_s3`: finished at `52.40`; total skipped non-finite reg-head steps observed: `5`
- `OpenTAD bugfix queue`: `v3_gumbel_softgate_minimal_bugfix0420` finished with peak `54.32` and final `54.22`; `v3_gumbel_softgate_coarse_det_bugfix0420` finished at `53.17`; `v3_hard_scorer_aux_bugfix0420` finished at `56.97` with peak `57.00`
- current `OpenTAD_Back` queue note:
  - `completion_oracle_upper_ampoff` running on `24013`
  - `completion_learned_minimal_ampoff` queued behind it on `24013`
  - `completion_full_oracle_dense_control_ampoff` running on `25876`
  - `completion_scatter_dense_baseline_ampoff` queued in a detached wait screen on `25876`
- current interpretation:
  - point-wise irregular decoding is a real recovery over the dense-grid sanity path
  - simple dense-grid output normalization does not help
  - the current bridge-style point decoder still trails the established `~49` `y`-line heads, and fp32 does not rescue it
  - full-train bottleneck on this carrier is now dominated by support size first, then by joint optimization
  - toy diagnostics show cls-to-trunk interference is causal, and reg-only failure is support-driven rather than capacity-driven
  - the completed `topk9` transfer pair closes the bridge line: support-size strengthening transfers, but toy rescue imports (`regwarm10`, `detach-cls`) do not materially lift the strongest full-train carrier
  - the next high-information decision is no longer another bridge or `x`-line run; it is whether oracle and learned completion can move materially above the current `54.03` nearest anchor
  - the new `headv3_x_temporalgridfix` rerun is a clean check for repaired temporal-grid semantics on the current `x` line, but it still does not exercise the separate feature/grid weighting mismatch because `GridAware...` uses dense `MaxPool1d`
  - the latest OpenTAD bugfix reruns strengthen a different conclusion: soft-gating did not fail only because of the already-fixed implementation bugs; after the fixes, the minimal line is somewhat stronger but still sub-baseline, the harder scorer branch is better yet still far from `63+`, the added coarse-det branch stays negative, and the formal temporal-grid fix does not lift the current `x` carrier above its old level
  - the next decisive `OpenTAD_Back` work should therefore move away from bridge polishing and toward either sparse-to-dense completion or a genuinely stronger native irregular decoder contract

## Open unknowns
- what the true no-remap shell-overhead number is once a genuinely no-remap control is run
- how strong the stupid sparse-to-dense + unchanged dense detector baseline is
- how high the detector can go if sparse inputs are oracle-completed before entering the original dense path
- how much of the `63.12 -> 53.20` gap is recoverable by learned completion alone
- whether any residual gap remains once the main path stops relying on the current irregular shell semantics
- how best to encode observation geometry for completion: mask-only, local gaps, visible-span support, or boundary-aware support
- whether boundary-first sparse evidence transfers more naturally into completion than into direct irregular-shell detection
