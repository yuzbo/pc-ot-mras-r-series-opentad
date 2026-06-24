# Frame/Token Hybrid Acquisition Route Review Context

Route label:
`DIVERGENT_INNOVATION_FRAME_TOKEN_HYBRID_DO_NOT_MERGE_WITH_C3`

This is a divergent Frame/Token Hybrid acquisition route. It is not part of the
C3 optimization line and must not be attributed to C3-Pro, BH-SDC, Event, or
Boundary routes.

## Purpose

The route tests whether a detector-compatible dense temporal input can be built
from two deployable acquisition units:

- raw frame observations kept at boundary-sensitive positions and periodic
  anchors;
- span tokens representing stable long gaps, carrying `span_start`, `span_end`,
  `role`, `visibility`, and `compression_confidence`.

The selector writes a dense-completion bridge that preserves observed raw frame
positions exactly, interpolates unobserved valid positions from observed
anchors, and conditions stable-gap completion on the span token metadata. Invalid
mask suffix positions are zeroed instead of forwarded from the raw dense input.

## Changed Surface

- Input sampling and acquisition policy.
- Token compression metadata for stable temporal gaps.
- Pre-backbone dense completion bridge.
- Local fail-closed gate and validator.
- Route-specific tests and configs.

This route does not change Adapter internals, detector head logic,
loss/assignment, or test-time post-processing.

## Protocol Boundary

Current status is implementation/precheck preparation only.

- Remote sync: locked.
- Slurm or full training: locked.
- `tools/train.py` and `tools/test.py`: locked by config gate.
- Metric, runtime, deploy, and paper claims: locked.
- Test-time GT, teacher, oracle, detector-output, raw-prediction-cache, result,
  and checkpoint payloads: forbidden by the route and validator.

## Local Verification Targets

The route-specific tests check:

- span-token schema and route label;
- no forbidden GT/teacher/oracle/cache/result/checkpoint payloads at test time;
- exact preservation of observed raw frame positions;
- no raw frame forwarding at unobserved valid positions;
- invalid mask suffix zeroing;
- ActionFormer-compatible dense axis shape;
- config parseability and fail-closed gate flags;
- selector export through `opentad.models.selectors`;
- validator rejection of train, remote, Slurm, GPU, full-train, and claim flags;
- UTF-8 BOM gate JSON compatibility.

The validator command remains precheck-only and accepts only
`ALLOW_FRAME_TOKEN_HYBRID_PRECHECK_ONLY`.
