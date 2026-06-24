# Frame/Token Hybrid Acquisition Route Review Context

Route label:
`DIVERGENT_INNOVATION_FRAME_TOKEN_HYBRID_DO_NOT_MERGE_WITH_C3`

This is a divergent Frame/Token Hybrid acquisition route. It is not part of the
C3 optimization line and must not be attributed to C3-Pro, BH-SDC, Event, or
Boundary routes.

## Purpose

The route tests whether a detector-compatible dense temporal input can be built
from preview/probe-visible selection metadata and two acquisition units:

- raw frame observations kept at boundary-sensitive positions and periodic
  anchors;
- span tokens representing stable long gaps, carrying `span_start`, `span_end`,
  `role`, `visibility`, and `compression_confidence`.

The selector writes a dense-completion bridge that preserves observed raw frame
positions exactly, interpolates unobserved valid positions from observed
anchors, and conditions stable-gap completion on the span token metadata. Invalid
mask suffix positions are zeroed instead of forwarded from the raw dense input.

Important scope downgrade after the Pro review: in the current OpenTAD
ActionFormer pipeline, dense `LoadFrames`/`DecordDecode` still happens upstream
of this selector. Therefore this commit must be interpreted as a
preview/probe-visible post-decode, pre-backbone compression and dense-completion
route. It does not yet prove raw decode/read saving. Raw acquisition, deploy,
runtime/FLOPs, metric, and paper claims remain locked until a reviewed pre-decode
loader hook or equivalent deploy path exists.

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
- `sbatch`, `scp`, and `rsync`: locked by the full-train-candidate gate.
- Metric, runtime, deploy, and paper claims: locked.
- Test-time GT, teacher, oracle, detector-output, raw-prediction-cache, result,
  and checkpoint payloads: forbidden by the route and validator.

## Local Verification Targets

The route-specific tests check:

- span-token schema and route label;
- no forbidden GT/teacher/oracle/cache/result/checkpoint payloads at test time;
- exact preservation of observed raw frame positions;
- no raw frame forwarding at unobserved valid positions;
- preview/probe metadata, not the dense input tensor, drives the gated selector
  when `require_preview_signal=True`;
- missing preview/probe metadata fails closed;
- observed raw frame masks, span-derived masks, dense-completion masks, and
  compute-accounting metadata distinguish what is observed, compressed, and
  synthetically completed;
- ActionFormer forwards the selector-rewritten dense bridge and metadata to the
  backbone/head path;
- invalid mask suffix zeroing;
- ActionFormer-compatible dense axis shape;
- config parseability and fail-closed gate flags;
- selector export through `opentad.models.selectors`;
- validator rejection of train, remote, Slurm, GPU, full-train, and claim flags;
- validator JSON summary for config-only precheck remains precheck-only and has
  empty allowed entrypoints;
- UTF-8 BOM gate JSON compatibility.

The validator command remains precheck-only and accepts only
`ALLOW_FRAME_TOKEN_HYBRID_PRECHECK_ONLY`.
