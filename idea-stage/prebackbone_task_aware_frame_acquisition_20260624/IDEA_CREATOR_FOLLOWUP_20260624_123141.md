# IDEA_CREATOR_FOLLOWUP_20260624_123141

Generated: 2026-06-24 12:31:41 Asia/Shanghai

Repository: `E:/DeskTop/TAD/temrefuse-tad/OpenTAD_PCOTMRAS_R16_R18_R20_Clean_20260619_1730`

Observed branch: `codex/prebackbone-pcotmras-scout-selector`

Observed HEAD: `471897edf7ea9871a4e7f8eeacbedeba178ef989`

Scope lock: documentation-only idea-creator follow-up. No code edit, no upload, no deletion, no training, no evaluation run, no remote command. This document reviews the latest C3-Pro frame-score-first implementation and extends the idea-stage package with more divergent but executable routes.

## 0. Idea-Creator Workflow Compliance

This follow-up uses the idea-creator structure:

1. Landscape: current local implementation, existing idea-stage package, and related literature signals.
2. Divergent ideas: 10 route candidates, each with method, hypothesis, minimal code surface, minimum experiment, diagnostics, contribution type, risk, effort, likely failure, and key code needs.
3. Feasibility gate: keep/defer/reject decisions before any implementation or launch.
4. Ranked top ideas: practical priority for innovation and attribution.
5. Pilot/full experiment order: designed sequence only; no experiment was launched here.

Local paper scan note: `papers/` and `literature/` directories are absent in the inspected clean repo. The local idea-stage package already contains earlier landscape and novelty notes. External context was limited to high-signal public references:

- [ActionFormer](https://arxiv.org/abs/2202.07925) and [ECCV PDF](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136640485.pdf)
- [AdaTAD CVPR 2024 PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_End-to-End_Temporal_Action_Detection_with_1B_Parameters_Across_1000_Frames_CVPR_2024_paper.pdf)
- [Flexible Frame Selection for Efficient Video Reasoning, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Buch_Flexible_Frame_Selection_for_Efficient_Video_Reasoning_CVPR_2025_paper.pdf)
- [VideoBrain adaptive frame sampling](https://arxiv.org/html/2602.04094v2)
- [A.I.R. adaptive frame selection](https://arxiv.org/html/2510.04428v1)
- [CoSeLECT adaptive frame selection](https://openreview.net/pdf/85bab755c7254aed0b86d31707b4fac0a92d777d.pdf)
- [Language-Guided Temporal Token Pruning for Efficient VideoLLM](https://aclanthology.org/2025.emnlp-main.451.pdf)
- [EVAD token dropout and context refinement](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_Efficient_Video_Action_Detection_with_Token_Dropout_and_Context_Refinement_ICCV_2023_paper.pdf)

## 1. Landscape

### 1.1 Current C3-Pro Implementation Snapshot

Latest pushed core `471897e` is no longer a slot-led selector. The inspected path is:

- `opentad/models/transformer_adapter/pc_ot_mras_prebackbone_selector.py`
- `configs/adatad/thumos/pc_ot_mras_prebackbone_c3_pro_boundary_reader_full_train_candidate_n16r4.py`
- `scripts/run_pc_ot_mras_prebackbone_c3_reader_full_train_n16r4.sbatch`
- `tests/test_pc_ot_mras_prebackbone_pro_reader_design.py`
- `tests/test_pc_ot_mras_prebackbone_c3_variants.py`
- `tests/test_pc_ot_mras_prebackbone_nan_guards.py`

Observed strengths:

- The selected hard indices for train/eval are driven by `frame_selection_logits`, not by slot logits.
- The reader consumes compressed low-resolution pixels and emits dense task signals: action, start, end, boundary, uncertainty, redundancy, and frame-selection logits.
- Hard top-k selects real frames. Slot logits may exist as diagnostics or auxiliary structure but are not the hard source under `selection_strategy="frame_score_topk"`.
- Straight-through transport sends detector-side gradient to `frame_selection_logits`.
- Tests explicitly check that invalid padding is masked, conflicting slot logits do not dominate hard selection, ST gradient reaches frame score logits, and the metadata excludes P2, teacher, raw-cache, and test-GT shortcuts.
- The launch gate now defaults to `READER_VARIANT=pro_boundary`, forbids direct test shortcut paths, and greps for finite-value hazards in logs.

Remaining mechanism gaps:

- The dense reader heads are richer than their explicit supervision. In the inspected loss path, frame-score-first auxiliary GT loss mainly anchors `frame_selection_logits` to action targets; boundary, uncertainty, and redundancy heads may become decorative unless configured or coupled carefully.
- The ST surrogate path can use `mean_proxy`, which gives a narrow gradient carrier. This may be stable but could under-train semantic ranking quality.
- The downstream detector remains close to the original regular-grid ActionFormer/AdaTAD assumption. Sparse selected positions are recorded, but assignment, temporal distance, pyramid behavior, and post-processing are not yet fully irregular-time native.
- Fixed 384 top-k is a useful controlled implementation target, but it is not the final dynamic-acquisition objective.
- Frame-score-first is better aligned with the user's intended route than slot-led transport, yet it can still collapse into uniform-like coverage, action-center bias, boundary starvation, or redundancy clusters if its objective is not sharpened.

### 1.2 Literature Gap

Recent frame selection work in video reasoning increasingly supports adaptive counts, semantic retrieval, query-aware pruning, or controller tokens. These systems show that uniform sampling is not the only viable interface. However, they usually optimize VQA/reasoning or keyframe understanding, not high-overlap temporal localization under a detector head. Efficient action detection work such as EVAD prunes tokens around keyframes, but it is not a pre-backbone low-res acquisition controller for temporal action detection. ActionFormer and AdaTAD provide strong detector machinery, but they inherit a mostly regular temporal feature-grid world.

Therefore the local opportunity is not just "make frame selection run." The stronger contribution is:

> A deployable pre-backbone acquisition policy that reads cheap pixels, predicts task-aware frame utility, selects a variable sparse set of real frames, and makes the detector understand the resulting physical time geometry.

This follow-up does not assume frame-score-first is correct. Slot transport, dynamic budget, and irregular ActionFormer modification are all open to criticism. The purpose is to create routes that can falsify or strengthen the current direction.

## 2. Divergent Candidate Ideas

### Idea 1. Calibrated PixelRanker: make C3-Pro heads train as real mechanisms

One-sentence method: Keep frame-score-first top-k, but add explicit calibration and train-only losses so boundary, uncertainty, and redundancy heads become measurable causes of ranking rather than unused decorations.

- Core hypothesis: Current C3-Pro may already have the right interface, but its rich heads need sharper supervision and diagnostics before the selector can protect boundaries and suppress redundancy.
- Minimal code surface: selector loss block in `pc_ot_mras_prebackbone_selector.py`; config weights in C3-Pro config; tests in `test_pc_ot_mras_prebackbone_pro_reader_design.py`.
- Minimum experiment: first run synthetic/unit diagnostics only, then a short controlled local smoke when allowed; compare head-target correlations and selected-role composition without claiming detector quality.
- Diagnostic metrics: boundary-head peak alignment, redundancy-head correlation with temporal similarity, uncertainty concentration near ambiguous transitions, selected boundary support, selected duplicate rate, max physical gap, finite logits/gradients.
- Contribution type: mechanism completion and attribution.
- Risk level: Medium.
- Estimated effort: 0.5 to 1 day for code and tests; longer only if target construction touches dataset metadata.
- Most likely failure mode: handcrafted targets overfit the train annotations and turn into brittle boundary labels that do not help deploy-time acquisition.
- Key code demand: loss target construction for start/end/boundary/redundancy/uncertainty, metadata logging for every head, config toggles to turn each target on/off.

### Idea 2. ST-gradient bandwidth audit: replace narrow mean proxy when needed

One-sentence method: Treat the straight-through path itself as a research object by comparing `mean_proxy`, local soft windows, and full-feature surrogate gradients.

- Core hypothesis: If detector gradient reaches `frame_selection_logits` through only a scalar proxy, the selector may learn "which rows get any loss" but not "which visual evidence improves localization."
- Minimal code surface: existing `_apply_sparse_transport` surrogate mode options; config variants; finite-gradient tests.
- Minimum experiment: no full run first; inspect gradient norm, gradient entropy across selected frames, and whether top-k ordering changes after a few controlled steps when allowed.
- Diagnostic metrics: nonzero gradient ratio on frame scores, gradient concentration by rank, local-window entropy, score update magnitude, NaN/Inf count, GPU memory delta for full surrogate.
- Contribution type: optimization mechanism and collapse prevention.
- Risk level: Low to Medium.
- Estimated effort: 0.5 day if modes already exist; 1 day if adding safer logging.
- Most likely failure mode: full-feature surrogate causes memory pressure or unstable gradients; mean proxy may be stable enough and hard to beat.
- Key code demand: `st_surrogate_mode`, `_frame_score_transport_plan`, `_apply_sparse_transport`, selector metadata dump.

### Idea 3. Boundary packet allocator instead of isolated frame top-k

One-sentence method: Convert frame scores into small boundary/action packets, selecting local temporal neighborhoods around predicted starts/ends plus sparse action interiors.

- Core hypothesis: High-overlap localization fails when a selector chooses isolated high-action frames but misses transition evidence; packetized selection gives boundaries temporal context without reverting to uniform coverage.
- Minimal code surface: new `selection_strategy` branch after `frame_selection_logits`; config for packet radius and interior quota; tests for budget, ordering, and invalid mask safety.
- Minimum experiment: synthetic videos with known action intervals; verify starts/ends receive local packet support and budget remains fixed before any detector run.
- Diagnostic metrics: start/end packet hit rate, selected-frame adjacency distribution, boundary-to-interior ratio, max physical gap, duplicate rate, padding mask violations, selected index monotonicity.
- Contribution type: acquisition policy.
- Risk level: Medium.
- Estimated effort: 1 to 2 days.
- Most likely failure mode: packetization becomes a disguised fixed coverage rule, wasting budget on false boundary peaks.
- Key code demand: packet assembly, tie-breaking, budget fill after packet reservation, metadata roles such as `start_packet`, `end_packet`, `interior`, `fallback`.

### Idea 4. Dynamic budget with abstention and fallback

One-sentence method: Predict a per-video or per-window budget bucket from cheap reader signals, with uncertainty-triggered fallback to a conservative sampler.

- Core hypothesis: The final problem is not fixed 384. Easy redundant windows should spend fewer frames, while boundary-dense or uncertain windows should spend more.
- Minimal code surface: budget controller module in selector; config bucket list; metadata for selected budget; launch/test guards that prevent hidden test-time GT.
- Minimum experiment: budget-only dry run over training metadata or cached cheap-reader outputs when available; inspect distribution and fallback rate before detector training.
- Diagnostic metrics: selected budget histogram, budget versus action density, budget versus boundary count, fallback frequency, empty-action behavior, high-risk window budget inflation, compute proxy.
- Contribution type: final-objective reframing and compute-performance frontier.
- Risk level: High.
- Estimated effort: 2 to 4 days for robust implementation, longer for credible experiments.
- Most likely failure mode: controller learns dataset priors or length shortcuts instead of content difficulty; variable length breaks detector contracts.
- Key code demand: budget logits, differentiable or policy-gradient-free training target, selector padding/mask contract, detector mask propagation, reporting of per-sample budget.

### Idea 5. Physical-time ActionFormer micro-adaptation

One-sentence method: Keep the selector fixed, but make ActionFormer consume selected physical coordinates through time-distance embeddings, mask-aware assignment, and coordinate-faithful decoding.

- Core hypothesis: Even a good selector can look bad if the detector still interprets sparse frames as a regular stride axis.
- Minimal code surface: temporal grid utilities, detector head routing, point generation, assignment/loss coordinate conversion, post-processing decode.
- Minimum experiment: coordinate audit only first; feed known selected axes and verify decoded proposal times remain physically consistent.
- Diagnostic metrics: selected-index-to-time roundtrip error, point-to-GT physical distance, mask-consistent assignment count, proposal boundary drift, empty/duplicate selected-axis handling, finite loss under irregular gaps.
- Contribution type: detector adaptation and irregular temporal geometry.
- Risk level: High.
- Estimated effort: 3 to 6 days because the change cuts through shared detector assumptions.
- Most likely failure mode: partial irregular adaptation creates worse attribution than no adaptation; silent off-by-stride errors may appear healthy in loss.
- Key code demand: temporal-grid metadata, ActionFormer point generator, loss assigner, decode/post-processing, config gate to isolate detector-only adaptation.

### Idea 6. Listwise frame-ranking loss instead of pointwise action BCE

One-sentence method: Train `frame_selection_logits` with pairwise/listwise utility targets so boundary-critical frames outrank redundant action interiors rather than merely matching actionness.

- Core hypothesis: Pointwise action supervision makes action centers too attractive and cannot express "this boundary frame is more valuable than the fifth redundant interior frame."
- Minimal code surface: ranking target builder in selector loss; optional differentiable top-k/listwise loss; tests for stable finite ranking loss.
- Minimum experiment: train-only diagnostic on batches with GT intervals; report rank ordering quality and selected role distribution.
- Diagnostic metrics: boundary frame percentile rank, action-interior redundancy rank, pairwise violation count, rank entropy, selected unique temporal bins, NaN/Inf count.
- Contribution type: objective design.
- Risk level: Medium.
- Estimated effort: 1 to 2 days.
- Most likely failure mode: ranking targets are too heuristic and conflict with detector gradients.
- Key code demand: physical-frame target labels, pair sampler, loss normalization by video length/action count, metadata export.

### Idea 7. Redundancy contrast reader

One-sentence method: Add a cheap temporal contrast objective that teaches the reader which low-res frames are near-duplicates before top-k selection.

- Core hypothesis: Redundancy is easier to learn from cheap pixels than action semantics; reducing sustained duplication may free budget for boundaries.
- Minimal code surface: reader feature projection, temporal similarity target or self-supervised contrast term, redundancy-head loss path.
- Minimum experiment: no detector needed first; compare redundancy logits against low-res feature similarity and selected cluster rate.
- Diagnostic metrics: local similarity versus redundancy score, selected cluster length, duplicate dense-index rate, max gap, redundancy score among selected versus rejected frames.
- Contribution type: reader pretraining/regularization.
- Risk level: Medium.
- Estimated effort: 1 to 2 days.
- Most likely failure mode: visual redundancy is not the same as temporal detection redundancy; low-res changes may mark camera motion instead of semantic novelty.
- Key code demand: low-res descriptor cache inside batch, contrast target generation, redundancy loss weight, selector metadata.

### Idea 8. One-forward scout-refine acquisition

One-sentence method: Use the cheap reader in two internal passes: coarse scan finds candidate regions, refine pass reallocates frames near uncertain boundaries, then the detector still runs once.

- Core hypothesis: Single-pass top-k may be too myopic; a two-stage reader can spend budget after seeing its own uncertainty map while preserving deployable one-detector-forward behavior.
- Minimal code surface: reader forward scheduling, second-pass score refinement, budget split configuration, metadata for round-0 and round-1 roles.
- Minimum experiment: synthetic uncertainty maps and dry-run selection; confirm the second pass changes selected frames near boundaries rather than adding uniform fill.
- Diagnostic metrics: changed-frame ratio after refine, boundary uncertainty reduction proxy, round-specific budget use, selected-role histogram, finite logits after two passes.
- Contribution type: acquisition architecture.
- Risk level: High.
- Estimated effort: 3 to 5 days.
- Most likely failure mode: added complexity produces little beyond a wider temporal conv reader; round coupling may make debugging hard.
- Key code demand: multi-round reader API, no detector-feedback shortcut, strict metadata proving detector is invoked once.

### Idea 9. Frame-token hybrid allocator

One-sentence method: Allocate both how many real frames are acquired and how many backbone/tubelet tokens each selected region keeps.

- Core hypothesis: Some windows need more temporal frames; others need fewer frames but richer local tokens. A fixed frame count ignores this tradeoff.
- Minimal code surface: selector budget metadata, backbone/token compression hook, config for frame-token budget frontier.
- Minimum experiment: code-level contract audit first; then tiny smoke to ensure variable token masks do not break the backbone.
- Diagnostic metrics: token count per window, frame count per window, token mask validity, latency/FLOP proxy, boundary-region token density, finite feature stats.
- Contribution type: system-level efficiency frontier.
- Risk level: Very High.
- Estimated effort: 1 to 2 weeks if backbone internals need real modification.
- Most likely failure mode: engineering blast radius overwhelms attribution; detector improvement cannot be assigned to frame acquisition or token compression.
- Key code demand: low-res selector output schema, backbone token mask support, adapter mask propagation, logging of two-dimensional budget.

### Idea 10. Selector failure sentinel and automatic conservative fallback

One-sentence method: Add train/test deploy-safe sentinels that detect dangerous selector states and switch to a declared conservative sampler.

- Core hypothesis: Innovation routes will fail often; a deployable acquisition system needs a visible safety valve for NaN, mask collapse, over-clustering, or boundary starvation.
- Minimal code surface: selector diagnostics, fallback sampler branch, launch guard, tests for each sentinel.
- Minimum experiment: unit tests only first; intentionally feed invalid scores, all-padding windows, clustered peaks, and extreme uncertainty.
- Diagnostic metrics: fallback reason counts, invalid mask rate, finite logits, duplicate/cluster thresholds, max gap threshold, selected valid-count preservation.
- Contribution type: robustness and deployability.
- Risk level: Low.
- Estimated effort: 0.5 to 1 day.
- Most likely failure mode: fallback triggers too often and hides selector weaknesses; papers cannot claim innovation from fallback behavior.
- Key code demand: sentinel thresholds, metadata reasons, config to enable/disable fallback, clear reporting that separates selector and fallback selections.

## 3. Feasibility Gate

| Idea | Gate | Reason | Pre-implementation blocker |
| --- | --- | --- | --- |
| 1. Calibrated PixelRanker | Keep | Directly strengthens C3-Pro without changing detector contracts | Need target definitions that avoid test leakage and over-annotation |
| 2. ST-gradient bandwidth audit | Keep | Low-cost way to test whether current ST path is too weak | Need memory guard for full-feature surrogate |
| 3. Boundary packet allocator | Keep | Attacks high-overlap boundary starvation while staying frame-score-first | Need proof it is not just hidden uniform coverage |
| 4. Dynamic budget fallback | Keep | Best aligned with final objective | Need variable-length detector contract or padded sparse contract |
| 5. Physical-time ActionFormer | Keep but staged | Necessary if sparse inputs are truly irregular | Needs careful coordinate audit before any detector run |
| 6. Listwise ranking loss | Keep | More expressive than pointwise action targets | Need robust ranking target normalization |
| 7. Redundancy contrast reader | Keep | Makes redundancy head trainable from cheap pixels | Need separate visual redundancy from semantic importance |
| 8. One-forward scout-refine | Defer | Innovative but higher coupling and debugging cost | Need simpler reader-head calibration first |
| 9. Frame-token hybrid | Defer | Potentially strong but too broad for immediate attribution | Need stable frame-only route and mask contract |
| 10. Failure sentinel | Keep as safety layer | Prevents NaN/collapse from contaminating formal runs | Must not become a hidden performance crutch |

Rejected for now:

- Test-time detector-feedback active acquisition, because it risks turning into a multi-forward or raw-prediction shortcut.
- Validation/test teacher scoring, because it violates deploy-time evidence boundaries.
- Pure RL budget control as first dynamic-budget route, because variance and attribution cost are too high before reader diagnostics are mature.

## 4. Ranked Top Ideas

### Rank 1. Calibrated PixelRanker + listwise ranking

Combine Ideas 1 and 6. This is the most direct repair for current C3-Pro: it keeps the frame-score-first architecture but forces the reader heads and ranking objective to mean something. It is also the cleanest route for deciding whether the C3-Pro concept is under-trained or fundamentally weak.

Why first: small code surface, strong diagnostic value, and directly addresses boundary/ranking quality without detector surgery.

### Rank 2. Dynamic budget with visible fallback

Combine Ideas 4 and 10. Fixed 384 is only a controlled target. The next serious innovation should expose a budget distribution, not a single budget. The fallback sentinel is important because dynamic budget will create empty-window, clustered-selection, and uncertainty spikes.

Why second: closest to the final research objective, but should follow reader/rank diagnostics to avoid compounding unknowns.

### Rank 3. Physical-time ActionFormer micro-adaptation

Idea 5 is the strongest detector-side route. It should be implemented only as a staged micro-adaptation with coordinate audits first, because irregular geometry bugs can silently poison attribution.

Why third: necessary for the full sparse-input story, but risky enough to require a separate detector-only attribution lane.

### Rank 4. Boundary packet allocator

Idea 3 is a pragmatic middle ground between isolated frame top-k and full irregular detector modification. It can protect boundaries before a larger ActionFormer change is ready.

Why fourth: high practical value, but must prove it is not just uniform coverage with a new name.

### Rank 5. ST-gradient bandwidth audit

Idea 2 should be run as a gate before over-interpreting any failed selector training. If ST gradient is too narrow, many more sophisticated reader ideas will be misdiagnosed.

Why fifth: not a paper route alone, but it protects all other routes from a false-negative implementation conclusion.

## 5. Pilot and Full Experiment Order

No pilots or full experiments were launched by this follow-up. The order below is a proposed queue after code review and user approval.

### Pilot 0. Static and unit-only reader audit

- Purpose: verify current C3-Pro has no hidden slot dominance, invalid-padding selection, or non-finite score path.
- Inputs: existing tests plus new head-diagnostic unit tests if implemented.
- Stop condition: any invalid frame selected, non-finite score, or hard source not equal to frame score.

### Pilot 1. Reader-head calibration dry run

- Purpose: check whether boundary/uncertainty/redundancy heads correlate with their train-only targets.
- Changes: Idea 1 plus optional Idea 6.
- Outputs: head correlations, selected-role histogram, boundary support, redundancy cluster rate, score-gradient stats.
- Stop condition: boundary head learns nothing, redundancy head selects clusters, or ranking loss dominates detector loss scale.

### Pilot 2. ST surrogate bandwidth gate

- Purpose: decide whether current ST carrier is too narrow.
- Changes: Idea 2 only.
- Outputs: gradient ratio, gradient entropy, score update distribution, memory delta.
- Stop condition: full surrogate is unstable or mean proxy gives vanishing frame-score gradient.

### Pilot 3. Boundary packet selector

- Purpose: test whether packetizing boundary neighborhoods improves temporal evidence allocation before detector changes.
- Changes: Idea 3.
- Outputs: packet hit rate, boundary/interior budget ratio, gap distribution, duplicate rate.
- Stop condition: packet selector degenerates into uniform-like coverage or spends budget on false peaks.

### Pilot 4. Dynamic budget dry run

- Purpose: inspect budget-controller behavior without claiming detector performance.
- Changes: Idea 4 plus Idea 10.
- Outputs: budget histogram, fallback reasons, budget versus action density, budget versus uncertainty.
- Stop condition: controller collapses to one budget, uses video length shortcut only, or fallback dominates.

### Full Lane A. Fixed-budget reader route

- Sequence: calibrated PixelRanker -> listwise rank -> ST bandwidth -> boundary packets.
- Attribution: input sampling and reader objective only.
- Detector changes: none, except metadata propagation already required by current selector.

### Full Lane B. Dynamic-budget acquisition route

- Sequence: fixed-budget calibrated selector -> budget controller -> fallback sentinel -> budget frontier reporting.
- Attribution: dynamic acquisition policy.
- Detector changes: only mask/padding contract unless Full Lane C is enabled.

### Full Lane C. Irregular-detector route

- Sequence: coordinate audit -> physical-time point generation -> assignment/loss conversion -> decode/postprocess conversion.
- Attribution: detector adaptation to non-uniform sparse input.
- Required isolation: run against a fixed selector first so geometry effects are not mixed with reader learning.

## 6. Critical Risks to Carry Forward

1. Reader-head decoration risk: a head exists in the architecture but does not meaningfully train or affect selected frames.
2. Slot shadow risk: even with frame-score-first hard selection, slot logits can still influence auxiliary objectives or diagnostics in confusing ways.
3. Uniform-cover mimicry: any boundary packet or budget fill can quietly become a fixed coverage rule.
4. ST false-negative risk: weak surrogate gradient may make a good selector look untrainable.
5. Dynamic-budget shortcut risk: budget controller can learn video length or dataset priors instead of content difficulty.
6. Irregular-coordinate risk: detector loss can look finite while decoded times are physically wrong.
7. High-overlap boundary risk: actionness-centered ranking may improve coarse action evidence while starving transition frames.
8. NaN/collapse risk: top-k with masked logits, soft local ST windows, variable budgets, and duplicate fills all need explicit finite checks and fallback metadata.

## 7. Immediate Recommendation

Do not jump directly from C3-Pro to a large irregular ActionFormer rewrite. The next strongest step is:

1. Turn the C3-Pro reader heads into measurable mechanisms through calibrated head losses and listwise ranking diagnostics.
2. Audit ST-gradient bandwidth so failed learning is not misattributed to the idea.
3. Add a dynamic-budget controller only after fixed-budget ranking quality is interpretable.
4. Build irregular ActionFormer adaptation as a separate detector-only lane with coordinate audits before any long run.

This keeps the current frame-score-first implementation open to criticism: if calibrated ranking and ST audit still show poor boundary support, the route should pivot toward packetized acquisition or detector-side physical-time adaptation rather than defending the current design by default.
