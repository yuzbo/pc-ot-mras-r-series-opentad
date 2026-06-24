# Boundary Microscope Acquisition Route Review Context

Route label:

```text
DIVERGENT_INNOVATION_BOUNDARY_MICROSCOPE_DO_NOT_MERGE_WITH_C3
```

This route is a divergent innovation candidate. It is not part of the original
C3/C3-Pro optimization route, and the current gate does not allow it to be
mixed with C3-Pro, BoundaryDifficulty scout, BH-SDC, event-surprise,
frame-token-hybrid, or combo attribution. It must not be mixed with those
families under the current Boundary Microscope gate.

## Purpose

Boundary Microscope is an input-side acquisition route for high-IoU temporal
localization. It uses a deploy-visible cheap global scanner over dense raw
frame content to estimate coarse actionness and start/end hazard positions.
Likely boundary neighborhoods receive dense microscope packets, while action
interiors and background retain sparse anchors and gap guards.

The route is intended to test whether boundary-first acquisition can preserve
or improve localization under an up-to-384 selected-frame budget from a
768-frame dense window. It is not a detector-head, loss, assignment,
post-processing, token-compression, or C3-reader change.

## Tensor And Geometry Contract

`BoundaryMicroscopeAcquisitionRoute` runs before the backbone. It supports both
plain 5D raw-frame tensors `[B,C,T,H,W]` and inherited VideoMAE-style 6D
raw-frame tensors `[B,N,C,T,H,W]`. The selector gathers along the true temporal
axis only: `dim=2` for 5D and `dim=3` for 6D. It preserves the non-temporal
`N`, `C`, `H`, and `W` axes.

The selector returns prefix masks over the selected temporal axis and writes the
same geometry into metadata:

- `irregular_selected_positions`: selected dense-frame indices;
- `irregular_selected_output_valid_len`: number of selected output frames;
- `irregular_selected_valid_len`: original valid dense-frame prefix length;
- `selected_output_valid_lengths`: tensor equivalent of the selected valid
  output length.

These fields describe selected raw-frame coordinates before backbone feature
extraction. They are not detector predictions, teacher targets, GT-derived
plans, token-compression outputs, or post-processing shortcuts.

## Current Changed Surface

- Input sampling: changed by `BoundaryMicroscopeAcquisitionRoute`.
- Dynamic budget policy: up to `target_len`; actual selected count may be lower
  when hazard and coverage constraints need fewer frames.
- Token compression: unchanged.
- Adapter/backbone/neck/head/loss/assignment/post-processing: unchanged.
- Launcher/gate logic: fail-closed local precheck and full-train-candidate
  configs remain locked against train/test/remote/Slurm/GPU actions.

## Protocol Boundary

Current status is implementation, local gate validation, and
deployment-precheck validation only. The deployment-precheck stage is limited
to a route-owned N16R4 checkout plus `PRECHECK_ONLY=1` launcher execution.

Allowed:

- local unit tests;
- local config parsing;
- local fail-closed gate validator;
- route-owned remote sync into a new Boundary Microscope precheck path;
- route-owned `PRECHECK_ONLY=1` launcher execution that exits before training.

Locked:

- any shared-repo write, branch switch, staging, commit, push, prompt, or log;
- full-training Slurm submission;
- GPU training;
- `tools/train.py`;
- `tools/test.py`;
- raw-prediction load/save;
- detector-mAP claim;
- runtime, deploy, or paper claim.

The current accepted decision string remains:

```text
ALLOW_BOUNDARY_MICROSCOPE_PRECHECK_ONLY
```

This decision cannot authorize remote sync, Slurm, `tools/train.py`, full
training, evaluation, metric claims, runtime claims, deploy claims, or paper
claims. A future full-training route would require a separate explicit decision
string and separate validator logic; it must not reuse the precheck-only gate.
The current deployment-precheck update does not unlock full training,
evaluation, metric claim, deploy claim, or paper claim.

## Negative Attribution Guard

Boundary Microscope must remain separate from C3/C3-Pro, BH-SDC,
Event-Surprise, frame-token routes, and mixed route proposals. Configs and gate
payloads must not relabel this route as any of those families. The review text
may mention those labels only to state that they are excluded from attribution.

## Required Review Questions

Reviewers should check:

1. Whether the implementation really follows cheap global scanner + boundary
   microscope packets + sparse action/background anchors.
2. Whether the route is independent from C3-Pro and cannot be attributed to C3
   reader/head/loss/post-processing changes.
3. Whether test-time GT, teacher, oracle, raw-prediction cache, checkpoint, or
   result JSON shortcuts are rejected.
4. Whether selected indices are sorted, unique, inside the valid prefix, and
   fail closed when the configured budget cannot satisfy the max-gap guard.
5. Whether the configs and validator keep full training and remote execution
   locked until an explicit future gate changes that decision.
