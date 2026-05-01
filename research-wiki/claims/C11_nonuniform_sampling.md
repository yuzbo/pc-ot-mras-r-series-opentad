---
type: claim
node_id: claim:C11
title: "Non-uniform input-side sampling is not inherently doomed; boundary-focused allocation can beat uniform, but action-interior densification is not supported"
status: partial
created_at: 2026-04-10T18:38:29+08:00
updated_at: 2026-04-13T22:55:59+08:00
---

# C11: Non-uniform input-side sampling

## Statement

On THUMOS-14 with the current OpenTAD_Back / ActionFormer-style dense detector family, non-uniform input-side sampling is not inherently impossible. It can work when the budget is allocated toward temporally critical regions, especially boundaries. However, spending extra budget inside the action interior is not supported by current evidence and often hurts performance.

## Evidence

| Experiment | Avg-mAP | Readout |
|---|---:|---|
| `exp:INPUT_STRIDE2_0410` | 65.09 | uniform 50% baseline |
| `exp:INPUT_RANDOMFIX_0410` | 63.12 | non-uniform random-fixed is only `-1.97` below uniform |
| `exp:INPUT_ORACLE_BOUNDARY_DENSE_0411` | 66.01 | boundary-focused non-uniform sampling beats uniform |
| `exp:INPUT_ORACLE_ACTION_BOUNDARY_DENSE_0411` | 62.04 | adding action-interior density is worse than boundary-only |
| `exp:INPUT_WEIGHTED_RANDOM_ACTION_BOUNDARY_TUBELET2_0413` | 58.54 | tubelet-aware action+boundary weighted random still underperforms badly |

## Current Interpretation

1. Non-uniform sampling itself is not the core problem.
2. Boundary density matters more than action-interior density for TAD.
3. The best current evidence supports "boundary-focused while preserving broad coverage", not "densify the whole action region".
4. The weak action+boundary results now hold at both frame-level and tubelet-level, so the failure is unlikely to be explained mainly by frame-vs-tubelet mismatch.
5. The remaining difficulty is more consistent with detector time-semantics mismatch:
   - unstable physical gap between neighboring kept tokens
   - projection / point / regression still carrying dense-grid assumptions
   - supervision and local temporal geometry becoming misaligned

## Supported

- "non-uniform input-side sampling is automatically bad" is not supported
- "boundary-focused budget allocation can be useful" is supported
- "action-interior densification is a reliable improvement" is not supported

## What Is Still Not Proven

- that all non-uniform strategies will fail
- that intelligent frame selection for TAD is impossible in principle
- that tubelet-aware selection alone is enough to close the detector mismatch

## Next Checks

1. Continue boundary-oriented input-side validation on the remaining tubelet2 oracle lines.
2. Transfer the strong boundary-side evidence into sparse-aware detector / projection experiments.
3. Keep irregular-detector work focused on true timeline semantics rather than only changing the sampler.

## Related

- exp:INPUT_STRIDE2_0410
- exp:INPUT_RANDOMFIX_0410
- exp:INPUT_ORACLE_BOUNDARY_DENSE_0411
- exp:INPUT_ORACLE_ACTION_BOUNDARY_DENSE_0411
- exp:INPUT_WEIGHTED_RANDOM_ACTION_BOUNDARY_TUBELET2_0413

## Connections

AUTO-GENERATED from graph/edges.jsonl - do not edit manually
