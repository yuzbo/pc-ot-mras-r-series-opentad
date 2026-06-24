# Idea-Creator Read-Only Follow-Up

Generated: 2026-06-24 12:35 Asia/Shanghai

Scope: read-only idea-creator follow-up from a parallel agent. No code edits, no remote commands, no training, no evaluation, and no metric claims.

Current code context:

- Branch: `codex/prebackbone-pcotmras-scout-selector`
- Latest pushed commit observed by main process: `471897edf7ea9871a4e7f8eeacbedeba178ef989`
- Current C3-Pro route: frame-score-first hard top-k from `frame_selection_logits`, local softmax ST for backward, compressed-pixel scout input, original AdaTAD backend.

## Situation Verdict

The current C3-Pro route is no longer the old slot-query hard source. Hard selection comes from dense per-frame `frame_selection_logits`, and train/eval hard indices are intended to match. This is closer to the user's requirement than the earlier slot-led allocation.

The remaining weakness is structural: single-scalar frame ranking can still over-concentrate around peaks; the local ST surrogate is not a global differentiable top-k relaxation; and the original ActionFormer backend still needs stronger physical-time / irregular-grid treatment to protect high-IoU localization.

## Candidate Ideas

### 1. Interval-Score-First Boundary Packet Acquisition

Method: predict dense start/end/action/risk/redundancy fields, form candidate boundary intervals, then allocate selected frames as boundary packets plus interior evidence packets.

Hypothesis: TAD benefits more from coherent boundary and interval evidence than from isolated high-scoring frames.

Minimum experiment: fixed `384/768`; compare exact-uniform, current C3-Pro frame top-k, and interval packet selection. Track boundary support within 2 dense indices, zero-boundary-support windows, mAP@0.7, and score-IoU ranking correlation.

Risk: medium. Highest method potential, but larger implementation surface than a pure ST change.

### 2. Global Differentiable Top-k Rank Transport

Method: keep the same hard top-k forward and same test-time selected indices, but replace per-selected-frame local softmax ST with a global differentiable ranking surrogate such as SoftSort, NeuralSort, sparsemax/entmax top-k, or a bounded OT-rank transport.

Hypothesis: a global ranking surrogate better matches the actual top-k decision boundary and provides more useful selector gradients than independent local windows.

Minimum experiment: same reader and same hard selected frames, only swap the ST surrogate. Track gradient norm reaching frame logits, ST-active frame count, hard-index stability, finite-loss stability, and mAP@0.7.

Risk: medium-high. Good next implementation after current C3-Pro because it changes training signal without changing deploy-time inputs.

### 3. Dynamic Marginal-Utility Budget Controller

Method: estimate marginal utility from boundary risk, redundancy density, and uncertainty; choose a per-video/window budget such as `288/320/352/384/416` instead of always `384`.

Hypothesis: the final contribution should come from spending fewer frames on easy/redundant windows and more on boundary-sensitive or difficult ones.

Minimum experiment: first do offline frozen-score frontier with current C3-Pro scores, then implement train-time budget if the frontier is meaningful. Report average selected frames, budget distribution, Avg-mAP, mAP@0.6, and mAP@0.7.

Risk: medium. Strong alignment with the final dynamic-acquisition objective.

### 4. Boundary-Dense / Interior-Sparse Role Quota

Method: predict per-frame roles (`start`, `end`, `interior`, `background`, `repair`) and assign deterministic quotas per role, then rank within each role.

Hypothesis: explicit role quotas prevent actionness peaks from consuming all boundary evidence.

Minimum experiment: fixed `384/768`; compare no quota, start/end quota, and start/end/uncertainty quota. Track boundary hit rate and high-IoU mAP.

Risk: medium. Easier than interval packets, less elegant than interval packets.

### 5. Uncertainty-Triggered Uniform Fallback

Method: when score entropy is high, boundary confidence is low, or selected max gap is unsafe, route part of the budget to exact-uniform or max-gap repair.

Hypothesis: fail-closed fallback can prevent selector collapse while the learned reader is still immature.

Minimum experiment: no fallback, partial fallback, full fallback. Track collapse/gap metrics and mAP@0.7.

Risk: low-medium. Useful safety path, but not the main novelty.

### 6. Temporal Diversity Top-k

Method: add a distance repulsion or facility-location surrogate to the frame score objective, selecting frames that are both useful and not overly clustered.

Hypothesis: current frame-score-first may over-focus on local peaks; diversity helps preserve temporal evidence without returning to pure uniform coverage.

Minimum experiment: compare max gap, p95 gap, action coverage, boundary support, and high-IoU mAP.

Risk: medium. Must avoid becoming a uniform sampler in disguise.

### 7. Physical-Grid ActionFormer Lite

Method: teach ActionFormer assignment/regression/postprocess to use real selected positions and cell widths instead of selected-index pseudo-time.

Hypothesis: high-IoU collapse under sparse irregular input is partly caused by a detector head that still assumes uniform selected-axis geometry.

Minimum experiment: keep selected frames fixed; compare original ActionFormer geometry versus physical-grid-lite geometry.

Risk: high. Important but not a same-day full-train modification.

### 8. Proposal-Quality-Aware Score Calibration

Method: use train-only GT to shape frame/interval scores toward future proposal quality or score-IoU ranking, without test-time GT or teacher leakage.

Hypothesis: actionness alone is not enough; acquisition should optimize downstream proposal quality and ranking.

Minimum experiment: add a ranking auxiliary loss and track proposal score-IoU correlation plus mAP@0.7.

Risk: medium-high. Needs careful leakage and attribution controls.

### 9. One-Forward Scout-Then-Refine

Method: use a cheap scout to find boundary-sensitive bands, then densify only those bands before one final detector forward.

Hypothesis: multi-stage acquisition can improve evidence placement without multiple detector passes.

Minimum experiment: precheck-only one-forward contract plus support/gap/budget diagnostics before training.

Risk: medium. Larger engineering surface.

### 10. Token-Frame Hybrid Compression

Method: vary both selected frames and token density, keeping more tokens near boundary/difficult regions and fewer in redundant regions.

Hypothesis: dynamic acquisition should eventually control token budget, not only frame count.

Minimum experiment: fixed frame count with variable token budget frontier.

Risk: high. Not appropriate before C3-Pro evidence is understood.

## Ranked Top Ideas

1. **Interval-Score-First Boundary Packet Acquisition**
   - Best fit to TAD mechanism and CVPR-level method story.
   - Replaces isolated frame peaks with boundary/interval evidence.

2. **Global Differentiable Top-k Rank Transport**
   - Best immediate upgrade to the current C3-Pro route.
   - Keeps train/test hard selection aligned while improving gradient quality.

3. **Dynamic Marginal-Utility Budget Controller**
   - Best bridge to the final dynamic-budget objective.
   - Should start as frozen-score frontier before full training.

## What Not To Do Immediately

Do not make tonight's complete experiment depend on:

- full physical-grid ActionFormer rewrite;
- token-frame hybrid compression;
- train-only utility distillation;
- full scout-then-refine system;
- large Sinkhorn/OT matrices at detector scale;
- any new launcher/protocol that needs another long review cycle before the current complete run finishes.

Immediate priority remains: keep current complete C3/C3-reader experiments running and diagnose them. The next design wave should be interval-first or global-rank-ST, not another slot-led allocation variant.
