# Boundary Microscope Acquisition Route Review Context

Route label:

```text
DIVERGENT_INNOVATION_BOUNDARY_MICROSCOPE_DO_NOT_MERGE_WITH_C3
```

This route is a divergent innovation candidate. It is not part of the original
C3/C3-Pro optimization route, and it must not be mixed with C3-Pro,
BoundaryDifficulty scout, BH-SDC, event-surprise, or frame-token-hybrid
attribution unless a future combo gate explicitly allows that merge.

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

## Current Changed Surface

- Input sampling: changed by `BoundaryMicroscopeAcquisitionRoute`.
- Dynamic budget policy: up to `target_len`; actual selected count may be lower
  when hazard and coverage constraints need fewer frames.
- Token compression: unchanged.
- Adapter/backbone/neck/head/loss/assignment/post-processing: unchanged.
- Launcher/gate logic: fail-closed local precheck and full-train-candidate
  configs remain locked against train/test/remote/Slurm/GPU actions.

## Protocol Boundary

Current status is implementation and local gate validation only.

Allowed:

- local unit tests;
- local config parsing;
- local fail-closed gate validator;
- GPT-5.5 Pro / Rosetta review using a bounded context packet.

Locked:

- remote sync;
- Slurm submission;
- GPU training;
- `tools/train.py`;
- `tools/test.py`;
- raw-prediction load/save;
- detector-mAP claim;
- runtime, deploy, or paper claim.

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
