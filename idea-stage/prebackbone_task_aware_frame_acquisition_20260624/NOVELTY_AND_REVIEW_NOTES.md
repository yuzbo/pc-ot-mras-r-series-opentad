# Novelty And Critical Review Notes

## Core Claims To Test

### Claim 1: TAD-specific acquisition is different from generic video frame selection.

- Novelty signal: medium-high.
- Closest prior: Flexible Frame Selector and other VideoQA/VLM frame selection methods optimize reasoning or query-conditioned understanding, not high-IoU temporal boundary localization.
- Required differentiation: show that action/start/end/boundary/uncertainty/redundancy signals protect TAD localization better than generic actionness/motion/uniform selection.
- Strongest objection: "This is just frame selection applied to TAD."
- Response needed: boundary support, mAP@0.7, and selected-frame diagnostics that generic selectors do not target.

### Claim 2: Dynamic budget matters more than fixed 50% recovery.

- Novelty signal: medium.
- Closest prior: adaptive frame/token selection and dynamic inference work in VideoQA/VLMs; TAD model compression reduces compute by dropping blocks.
- Required differentiation: report budget distribution by window/action difficulty and show a Pareto frontier, not only one 384/768 point.
- Strongest objection: "A fixed-budget selector is a weak engineering ablation."
- Response needed: average selected frames, budget histogram, latency/FLOPs, Avg-mAP, mAP@0.7, and failure cases.

### Claim 3: Non-uniform temporal acquisition requires detector geometry adaptation.

- Novelty signal: high if implemented cleanly.
- Closest prior: ActionFormer assumes a regular temporal feature grid; prior TAD methods such as RCL and graph/segment methods handle temporal coordinates in different ways, but not as a direct selected-frame physical grid for pre-backbone dynamic acquisition.
- Required differentiation: demonstrate that original selected-index ActionFormer fails or underperforms under non-uniform selected positions, and that a physical temporal grid fixes attribution and high-IoU localization.
- Strongest objection: "This is a detector modification, not an acquisition method."
- Response needed: separate selector-only, detector-only, and combo tables.

### Claim 4: ST frame-score learning can produce deployable utility without test-time teacher/cache/GT.

- Novelty signal: medium but fragile.
- Closest prior: differentiable or learnable frame/token selection methods; adaptive token sampling; keyframe-centric token pruning.
- Required differentiation: prove ST gradients actually affect `frame_selection_logits` and improve discrete selected positions, not only auxiliary losses.
- Strongest objection: "Straight-through gradients are an unstable proxy; top-k decisions are not truly optimized."
- Response needed: gradient-flow tests, score-rank reliability, selected-position evolution, and hard export mAP.

### Claim 5: Boundary-aware acquisition is the right TAD-specific primitive.

- Novelty signal: medium-high.
- Closest prior: boundary-aware TAD heads and refinement methods, but mostly detector-side; oracle-boundary evidence in this project suggests large input-side upside.
- Required differentiation: no-GT deployable boundary candidates must improve selected boundary support and high-IoU mAP.
- Strongest objection: "Boundary heads trained from GT are only a weak proxy and may overfit THUMOS14."
- Response needed: cross-split leakage audit, duration-stratified analysis, and ideally another dataset pilot.

## Closest Prior Work Matrix

| Prior work | What it does | Overlap | Key gap for this project |
|---|---|---|---|
| AdaTAD | Scales end-to-end TAD with temporal-informative adapters and long inputs | Same TAD/AdaTAD ecosystem | Does not solve deployable adaptive acquisition |
| AdaTAD++ | Decouples temporal/spatial adapters and improves scaling | Strong detector-side competitor | Not an input budget controller |
| Progressive Block Drop | Compresses TAD model compute by dropping blocks | Efficiency for TAD | Model compression, not task-aware frame acquisition |
| Flexible Frame Selector | Learnable adaptive frame selection for video reasoning | Learnable frame selection | Different task; not high-IoU TAD localization |
| LGTTP | Language-guided temporal token pruning for VideoLLMs | Adaptive temporal token density | Query/VLM setting, not detector-boundary acquisition |
| EVAD | Keyframe-centric token pruning for video action detection | Token pruning for action detection | Spatiotemporal detection setting, not long untrimmed TAD |
| ActionFormer | Anchor-free temporal localization with multiscale feature grid | Detector backend | Assumes regular temporal grid unless modified |

## Review Synthesis

The most defensible paper story is unlikely to be "C3-Pro frame-score top-k recovers 50% performance." That would be a controlled experiment, not a final contribution. A stronger story should combine three pieces:

1. Deployable TAD-specific acquisition evidence: boundary/action/uncertainty/redundancy aware.
2. Dynamic budget or compute frontier: average budget and high-IoU preservation.
3. Irregular temporal geometry handling: either prove original ActionFormer is sufficient or adapt it explicitly.

Most important next Pro question:

Does C3-Pro deserve a fixed-budget mAP run as a fast gate, or should the route first pivot to physical-grid ActionFormer / dynamic-budget controller before spending full training?

## Red-Team Failure Interpretations

- If C3-Pro matches exact-uniform but no better:
  - Interpretation: useful safety fallback; not a paper contribution.
- If C3-Pro beats exact-uniform mainly at low IoU but not mAP@0.7:
  - Interpretation: actionness/classification improved, boundary localization not solved.
- If mAP collapses while losses look normal:
  - Interpretation: suspect coordinate remap, selected-axis geometry, postprocess, or hard/soft mismatch before blaming reader semantics.
- If dynamic budget saves compute but loses mAP@0.7:
  - Interpretation: budget controller is under-spending on boundary-sensitive windows.
- If physical-grid ActionFormer helps all selectors:
  - Interpretation: main contribution may shift from selector to irregular-aware TAD interface.
