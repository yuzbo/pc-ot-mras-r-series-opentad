# Idea Candidates And Discussion

Direction: pre-backbone task-aware frame acquisition for TAD / C3-Pro frame-score-first / dynamic budget / irregular ActionFormer adaptation.

## External Advisory Status

GlobalAI/Gemini was asked for a divergent critique. It returned partial but useful signal: it strongly objected to treating isolated `frame_selection_logits + hard top-k + ST` as a final contribution and recommended more structured transport, segment-level, predictive, and detector-aware acquisition routes. The response was truncated twice, so it is advisory only.

MiniMax and local LLM reviewer calls were attempted but unavailable because API keys were not set. They are not counted as completed reviews.

## Candidate Ideas

### Idea 1: Boundary-Risk Frame Score Selector

- Method: Keep C3-Pro frame-score-first, but make the score explicitly a calibrated boundary-risk density with action interior, start/end slope, uncertainty, and redundancy terms supervised by train-only GT-derived dense targets.
- Core hypothesis: C3-Pro only becomes meaningful if `frame_selection_logits` are interpretable as TAD utility density rather than generic actionness.
- Minimum experiment: fixed 384/768 on THUMOS14; compare uniform, C3 slot, C3-Pro raw frame-score, and calibrated boundary-risk score; report Avg-mAP, mAP@0.6, mAP@0.7, boundary support@2, selected-frame redundancy.
- Contribution type: method plus diagnostic.
- Risk level: medium.
- Engineering effort: 2-4 days because most code path exists.
- Key code needs: selector main file, C3-Pro config, dense target loss, selected-axis GT remapping, tests for score calibration and finite guards.

### Idea 2: Differentiable Temporal Transport Selector

- Method: Replace hard top-k during training with a Sinkhorn/entropic optimal transport plan from dense 768 positions to selected 384 positions, then export hard monotone selections at inference.
- Core hypothesis: TAD acquisition is a global sequence allocation problem; optimal transport can reduce duplicate/cluster collapse better than independent frame logits.
- Minimum experiment: synthetic monotone boundary-support test, then THUMOS14 fixed 384/768 against C3-Pro and exact-uniform.
- Contribution type: new selection mechanism.
- Risk level: medium-high due memory and OT temperature sensitivity.
- Engineering effort: 1-2 weeks.
- Key code needs: frame selector transport path, ST transport weights, dense-to-selected remap, AMP finite tests, runtime profile.

### Idea 3: Segment-First Scout Instead Of Frame-First Scout

- Method: The low-resolution reader predicts variable-length action/boundary support intervals, then fills each interval with local dense samples plus a global coverage guard.
- Core hypothesis: Actions are segments; frame-only top-k tends to over-select peaks and under-support boundaries/interiors.
- Minimum experiment: fixed average 384 frames with interval proposal outputs; compare boundary support, action interior coverage, and high-IoU mAP versus C3-Pro.
- Contribution type: paradigm shift from frame selection to temporal evidence packet acquisition.
- Risk level: medium.
- Engineering effort: 1-2 weeks.
- Key code needs: reader outputs, interval-to-frame materialization, metadata, GT remap, selected-axis postprocess.

### Idea 4: Scout-Then-Refine Multi-Round Acquisition With One Detector Forward

- Method: Use a cheap scout pass to propose candidate boundary/action/uncertainty bands, allocate additional packets around those bands, merge into one final detector input, and run the detector once.
- Core hypothesis: Dynamic acquisition needs staged evidence gathering, but detector inference must remain one-forward to stay deployable.
- Minimum experiment: local precheck measuring support@2, zero-support rate, coverage share, selected-count budget, then a 384 fixed-budget detector mAP smoke.
- Contribution type: deployable acquisition algorithm.
- Risk level: medium.
- Engineering effort: 1 week if built on existing prebackbone selector, 2+ weeks if cleaner new module.
- Key code needs: scout descriptor path, candidate band ledger, merge/order logic, launch gates, no-cache/no-teacher tests.

### Idea 5: Dynamic Budget Controller From Boundary Risk And Redundancy

- Method: Predict budget per video/window/action-risk case, e.g. 288/320/352/384/416, using calibrated boundary uncertainty and redundancy density, then report compute-performance frontier.
- Core hypothesis: Fixed 50% cannot be the final contribution; easy/background windows should spend less and boundary-sensitive windows should spend more.
- Minimum experiment: use frozen C3-Pro scores to simulate dynamic budget choices; train only controller or use rule-based controller first; report average budget, budget distribution, Avg-mAP, mAP@0.7.
- Contribution type: dynamic inference policy.
- Risk level: medium-high because budget changes confound accuracy.
- Engineering effort: 4-7 days for controller audit; 1-2 weeks for full training.
- Key code needs: dynamic budget controller, hard export, metadata, frontier audit, launcher summary JSON.

### Idea 6: Irregular-Time ActionFormer With Physical Temporal Grid

- Method: Keep selection input-side, but modify ActionFormer point generation, regression targets, assignment, and post-processing to consume physical selected-frame coordinates/cell widths.
- Core hypothesis: Non-uniform acquisition cannot be judged fairly if ActionFormer treats selected frames as a uniform index axis.
- Minimum experiment: exact-uniform and C3-Pro selections with and without physical-grid ActionFormer; check mAP@0.7 and coordinate consistency.
- Contribution type: detector adaptation enabling sparse irregular acquisition.
- Risk level: high due head/assignment/postprocess coupling.
- Engineering effort: 2-4 weeks.
- Key code needs: `temporal_grid.py`, ActionFormer detector, ActionFormer head, post-NMS, selected-axis tests, dense coordinate audit.

### Idea 7: Dual-Stream Selector With Cheap Pixel Scout And Feature-Space Self-Distillation

- Method: Use low-res pixels for deploy-time scout but train it with a feature-space utility target distilled only from train split detector/backbone gradients or loss sensitivity, never validation/test teacher.
- Core hypothesis: Pixel-only frame scores may be too weak; train-only utility distillation can teach task relevance without deploy-time leakage.
- Minimum experiment: train split only utility labels from ablation/gradient sensitivity; compare C3-Pro score correlation with future detector utility and fixed-budget mAP.
- Contribution type: training signal / utility learning.
- Risk level: high due leakage and expensive label generation.
- Engineering effort: 2+ weeks.
- Key code needs: split boundary proof, teacher/utility generation script, no validation/test leakage guard, score-utility audit.

### Idea 8: Redundancy-Aware Token-Frame Hybrid Acquisition

- Method: Combine temporal frame selection with tubelet/token redundancy compression inside selected frames, spending saved compute on boundary-sensitive frames.
- Core hypothesis: Temporal redundancy and spatial/token redundancy should be traded jointly; fixed frame count alone may waste compute on redundant patches.
- Minimum experiment: keep 384 selected frames but reduce token budget in low-risk frames; compare runtime/FLOPs and mAP to pure frame selection.
- Contribution type: compute frontier method.
- Risk level: medium-high because runtime proof and implementation are nontrivial.
- Engineering effort: 2-3 weeks.
- Key code needs: tubelet redundancy aux, packed forward path, backbone adapter, runtime profile, FLOPs estimator.

### Idea 9: Acquisition-Detector Consistency Loss

- Method: Penalize mismatch between selected-frame evidence distribution and detector proposal/boundary confidence distribution, using train-time consistency between reader scores and detector intermediate outputs.
- Core hypothesis: Selector failure comes from optimizing a proxy not aligned with detector localization; consistency can align acquisition with downstream utility.
- Minimum experiment: add lightweight consistency on train split only; compare selector rank reliability, top-k boundary support, and mAP@0.7.
- Contribution type: loss/assignment alignment.
- Risk level: medium due possible self-reinforcing detector errors.
- Engineering effort: 1 week.
- Key code needs: ActionFormer intermediate outputs, selector loss path, metadata injection, no raw prediction cache guard.

### Idea 10: Uncertainty-Triggered Fallback Selector

- Method: Let the reader abstain when its confidence is low and fall back to exact-uniform or coverage-protective sampling for the uncertain portion.
- Core hypothesis: A deployable selector should fail closed; avoiding catastrophic bad selections may matter more than aggressive gains.
- Minimum experiment: compare no fallback, uniform fallback, and partial fallback on hard windows; report collapse rate, mAP@0.7, and budget cost.
- Contribution type: robustness mechanism.
- Risk level: low-medium.
- Engineering effort: 3-5 days.
- Key code needs: selector roles, metadata diagnostics, fallback gates, per-window failure audit.

### Idea 11: Boundary-Dense Action-Interior Sparse Allocation

- Method: Explicitly allocate high density near predicted start/end bands and lower density inside stable action interiors/background, with learnable band widths and duration priors.
- Core hypothesis: High-IoU TAD cares most about boundaries, while interiors need only enough evidence for class/actionness.
- Minimum experiment: oracle-free predicted band route versus uniform and current frame-score top-k; measure boundary support@2, mAP@0.7, and selected-count distribution by action duration.
- Contribution type: TAD-specific acquisition policy.
- Risk level: medium.
- Engineering effort: 1 week.
- Key code needs: start/end logits, boundary band allocator, duration-aware metrics, selected positions export.

### Idea 12: Proposal-Aware Reader Without Detector Raw Cache

- Method: Train a tiny pre-backbone reader to predict a coarse actionness/proposal density directly from low-res pixels, but never consume detector predictions at test time; use it only to decide acquisition before detector forward.
- Core hypothesis: C3-Pro needs a richer task abstraction than frame logits; a proposal-density reader is still deployable if it runs before the detector.
- Minimum experiment: low-res proposal density pretraining on train GT, then joint detector training; compare to frame-score-first and interval-first.
- Contribution type: reader architecture.
- Risk level: medium-high because low-res pixels may be insufficient.
- Engineering effort: 1-2 weeks.
- Key code needs: low-res pixel descriptor path, reader proposal heads, GT dense target builder, config inheritance, no-test-GT guard.

## Ranking For Pro Discussion

Top routes to ask Pro to judge first:

1. Idea 6: Irregular-Time ActionFormer With Physical Temporal Grid.
   - Reason: if detector geometry is wrong, all selector gains/losses are hard to interpret.
2. Idea 5: Dynamic Budget Controller.
   - Reason: aligns directly with final objective and compute frontier.
3. Idea 4 + Idea 11: Scout-refine acquisition with boundary-dense allocation.
   - Reason: strongest TAD-specific acquisition story.
4. Idea 2: Differentiable Temporal Transport.
   - Reason: principled replacement for brittle independent top-k/ST.
5. Idea 1: C3-Pro calibrated boundary-risk selector.
   - Reason: fastest continuation of current code, but likely a stepping stone rather than final paper story.

## Current C3-Pro Risk Register

1. ST gradient mismatch.
   - Hard selected frames determine detector input, while gradients flow through a soft local surrogate. The detector may train the score surface in a direction that does not match the discrete top-k decision boundary.

2. Frame-score collapse.
   - A single scalar `frame_selection_logits` may over-focus on action peaks, short high-motion bursts, or boundary-like noise, starving action interiors or coverage.

3. Boundary target leakage confusion.
   - Train-only GT-derived auxiliary targets are allowed, but the prompt to Pro must force inspection that no val/test GT, teacher, raw cache, or hidden prediction shortcut enters deployment.

4. Geometry mismatch.
   - If selected positions are non-uniform but ActionFormer assignment/regression/postprocess use a uniform selected index axis, high-IoU metrics may collapse or become misleading.

5. Fixed-budget pseudo-progress.
   - Recovering uniform-like 384/768 performance is useful as a control, not enough for the final dynamic acquisition claim.

6. NaN/AMP risk.
   - Masked logits, half precision, softmax under invalid positions, clamp ranges, and ST surrogate bmm need finite checks under extreme reader outputs.

7. Redundancy head ambiguity.
   - If redundancy target is derived from the selector's own duplicate pressure, it may regularize artifacts instead of real temporal redundancy.

8. Reader too weak for low-res pixels.
   - Low-resolution pixel descriptors may not contain enough semantic/action-boundary evidence; the selector could learn dataset priors rather than task utility.

9. Attribution ambiguity.
   - Gains could come from altered GT remapping, selected-axis postprocess, launch protocol, or hidden config inheritance instead of selector intelligence.

10. No compute frontier yet.
   - A fixed 384/768 route does not prove dynamic budget, latency, FLOPs, or deployability.

## Minimal Experiment Tree

Stage 0: code and protocol audit.

- Confirm no P2, no teacher, no raw cache, no val/test GT, no direct `tools/test.py` shortcut.
- Confirm selected positions, selected masks, GT remap, and protocol flags in metadata.
- Confirm finite behavior under extreme logits and AMP.

Stage 1: local selector geometry diagnostics.

- Boundary support@2 / support@4.
- Zero boundary support rate.
- Selected-frame redundancy and max gap.
- Score rank reliability versus train-only action/boundary targets.

Stage 2: fixed-budget smoke.

- Run C3-Pro 384/768 against exact-uniform and current best fixed-budget baselines.
- Main metrics: Avg-mAP, mAP@0.6, mAP@0.7.
- Stop gate: severe collapse or mAP@0.7 near zero triggers Pro severe-result diagnosis before long follow-ups.

Stage 3: mechanism ablations.

- Remove boundary term.
- Remove uncertainty term.
- Remove redundancy term.
- Replace frame_score_topk with slot transport.
- Add fallback uniform guard.

Stage 4: dynamic frontier pilot.

- Budgets 288/320/352/384/416 or continuous budget with cap.
- Report average selected frames, distribution by video/window/action duration/difficulty, Avg-mAP, mAP@0.7, FLOPs/latency if available.

Stage 5: irregular detector gate.

- Compare original ActionFormer versus physical-grid ActionFormer under the same selected positions.
- If physical-grid improves high-IoU, selector-only claims must be downgraded and combo route considered.

## Required Code For Review

Minimum code packet for Pro or any serious reviewer:

- `opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py`
- `configs/adatad/thumos/pc_ot_mras_prebackbone_c3_pro_boundary_reader_full_train_candidate_n16r4.py`
- `configs/adatad/thumos/pc_ot_mras_prebackbone_c3_hybrid_reader_full_train_candidate_n16r4.py`
- `configs/adatad/thumos/pc_ot_mras_prebackbone_c3_f1_lr_tinytransformer_st_original_adatad_full_train_candidate_n16r4.py`
- `scripts/run_pc_ot_mras_prebackbone_c3_reader_full_train_n16r4.sbatch`
- `opentad/models/detectors/actionformer.py`
- `opentad/models/dense_heads/actionformer_head.py`
- `opentad/models/utils/temporal_grid.py`
- `opentad/cores/train_engine.py`
- `opentad/utils/training_guard.py`
- `tests/test_pc_ot_mras_prebackbone_pro_reader_design.py`
- `tests/test_pc_ot_mras_prebackbone_c3_reader_variants.py`
- `tests/test_pc_ot_mras_prebackbone_nan_guards.py`
- `tests/test_pc_ot_mras_prebackbone_selector_behavior.py`
- `tests/test_pc_ot_mras_actionformer_selected_axis_point_geometry.py`
- `tests/test_pc_ot_mras_actionformer_selected_axis_target_assignment.py`
- `tests/test_pc_ot_mras_actionformer_result_detection_geometry.py`
- `PC_OT_MRAS_R16_R18_R20_CLEAN_MANIFEST.md`
