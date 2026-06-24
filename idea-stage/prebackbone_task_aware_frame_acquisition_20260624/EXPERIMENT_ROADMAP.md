# Claim-Driven Experiment Roadmap

This roadmap is planning material only. It does not authorize remote launch by itself.

## Claim Map

| Claim | Why it matters | Minimum convincing evidence | Linked blocks |
|---|---|---|---|
| C1: C3-Pro frame-score-first is a valid fixed-budget gate | Tests whether current implementation has any useful signal | Fixed 384/768 improves or matches strong controls without leakage, NaN, or geometry contradictions | B0, B1, B2 |
| C2: Boundary-aware acquisition protects high-IoU localization | Core TAD-specific claim | mAP@0.7 and boundary support improve over uniform/slot controls | B1, B2, B3 |
| C3: Dynamic budget is necessary for final goal | Avoids fixed-50% pseudo-claim | Pareto frontier with budget distribution and preserved Avg-mAP/mAP@0.7 | B4 |
| C4: Irregular temporal geometry is handled correctly | Makes non-uniform acquisition attributable | physical-grid or equivalent geometry audit distinguishes selector failure from detector mismatch | B5 |

## Experiment Blocks

### B0: Protocol And Finite Gate

- Claim tested: no-leakage and numerical safety.
- Dataset/split: local static and synthetic tests only.
- Compared systems: not applicable.
- Metrics: pass/fail for no P2, no teacher, no raw cache, no test GT; finite logits/matrix/ST; mask/shape invariants.
- Success criterion: all key tests pass and launcher forbids unreviewed shortcuts.
- Failure interpretation: do not launch full training.
- Priority: MUST-RUN.

### B1: Local Selection Geometry Audit

- Claim tested: selected frames support boundaries and avoid redundancy.
- Dataset/split: train/validation metadata only; no detector mAP required.
- Compared systems: exact-uniform, slot transport C3, C3-Pro frame-score, random-fixed.
- Metrics: boundary support@2/@4, zero-support rate, max dense gap, duplicate rate, selected roles, score-rank reliability.
- Success criterion: C3-Pro is not worse than exact-uniform on boundary support and does not produce pathological clustering.
- Failure interpretation: fix selector or add fallback/transport before full mAP.
- Priority: MUST-RUN.

### B2: Fixed 384/768 Detector Gate

- Claim tested: C3-Pro is at least a viable fixed-budget controlled implementation.
- Dataset/split: THUMOS14 standard training/evaluation protocol.
- Compared systems: exact-uniform, uniform stride-2, C3 slot/hybrid, C3-Pro.
- Metrics: Avg-mAP, mAP@0.6, mAP@0.7, loss trend, selected diagnostics.
- Success criterion: no severe collapse; useful signal if near or above exact-uniform and mAP@0.7 is preserved.
- Failure interpretation: if loss normal but eval collapses, escalate to coordinate/geometry diagnosis before more selector variants.
- Priority: MUST-RUN after B0/B1.

### B3: Mechanism Ablation

- Claim tested: action/boundary/uncertainty/redundancy heads matter.
- Compared systems: full C3-Pro, no boundary term, no uncertainty term, no redundancy term, frame-score with uniform fallback, slot transport.
- Metrics: Avg-mAP, mAP@0.7, boundary support, redundancy, selected budget.
- Success criterion: boundary/difficulty terms have measurable high-IoU or support effect.
- Failure interpretation: route may be pseudo-complex; simplify to fallback or transport.
- Priority: MUST-RUN only if B2 is viable.

### B4: Dynamic Budget Frontier Pilot

- Claim tested: dynamic acquisition is the real contribution.
- Compared systems: exact-uniform at fixed budgets, C3-Pro fixed 384, C3-Pro/dynamic controller.
- Metrics: average selected frames, budget distribution, FLOPs/latency if available, Avg-mAP, mAP@0.6, mAP@0.7.
- Success criterion: meaningful compute reduction at comparable high-IoU, or improved high-IoU at similar average budget.
- Failure interpretation: fixed-budget route remains a control; dynamic controller needs boundary-risk calibration.
- Priority: HIGH after fixed gate.

### B5: Irregular-Time ActionFormer Gate

- Claim tested: selected physical coordinates must be respected by detector head/assignment/postprocess.
- Compared systems: original ActionFormer selected-axis, physical-grid ActionFormer, exact-uniform, non-uniform selector.
- Metrics: coordinate audit pass/fail, mAP@0.7, proposal boundary error, assignment consistency.
- Success criterion: either original ActionFormer is proven safe under selected-axis protocol, or physical-grid route produces clear improvement/diagnosis.
- Failure interpretation: selector-only claims are not attributable.
- Priority: HIGH, especially if B2 underperforms.

## First Three Practical Actions

1. Run B0 locally and package evidence for Pro.
2. Run B1 local selection geometry audit before any full detector claim.
3. Ask Pro to choose between launching B2 fixed-budget mAP versus implementing B5 physical-grid gate first.

## Locked Actions Until Evidence Exists

- Full paper claim.
- Deploy claim.
- Runtime/FLOPs claim without measurement.
- Dynamic-budget claim from fixed 384/768 only.
- Selector superiority claim if physical geometry remains unaudited.
- More long STGA/SAPM/SAHM/SAN/combo runs if a severe collapse appears before Pro severe-result discussion.
