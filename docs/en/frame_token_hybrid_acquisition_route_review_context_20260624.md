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

Current status is implementation/precheck deployment only.

- Route-owned remote sync/precheck: allowed only for this deployment-precheck
  branch, a new Frame/Token-owned remote path, and `PRECHECK_ONLY=1`.
- Slurm full training: locked by default.
- The N16R4 precheck launcher is fail-closed. It validates the gate/config,
  writes bounded precheck artifacts, and exits before any training command when
  `PRECHECK_ONLY=1`.
- The N16R4 full-train launcher is also locked by default. With
  `PRECHECK_ONLY=0`, it may run `tools/train.py` only when
  `ALLOW_FRAME_TOKEN_HYBRID_FULL_TRAIN=1` and an external full-train gate JSON
  passes exact SHA256, active manifest SHA256, resolved config SHA256, RUN_TAG,
  route-label, user-override, and coordinator-override checks.
- `tools/train.py` and `tools/test.py`: locked by config gate.
- Full train remains locked unless a later reviewed gate explicitly opens both
  the config and launcher.
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
- full-train gate JSON missing/invalid cases fail closed;
- valid full-train gate JSON authorizes train only when RUN_TAG, manifest,
  resolved config, route label, and override statements match exactly;
- UTF-8 BOM gate JSON compatibility.
- N16R4 deployment-precheck launcher defaulting to `PRECHECK_ONLY=1`, refusing
  training/test entrypoints, and containing no training command.
- N16R4 full-train launcher defaulting to `PRECHECK_ONLY=1` and binding
  full-train execution to the external full-train gate.

The config validator remains precheck-only by default. Gate JSON validation now
has two explicit actions: `precheck` accepts only
`ALLOW_FRAME_TOKEN_HYBRID_PRECHECK_ONLY`, and `full-train` accepts only
`ALLOW_FRAME_TOKEN_HYBRID_FULL_TRAIN` with the additional fail-closed
authorization fields.

## Deployment Precheck Status

2026-06-24T20:01:33+08:00: deployment-precheck owner prepared the route-owned
N16R4 precheck launcher
`scripts/run_frame_token_hybrid_acquisition_precheck_n16r4.sbatch`. This does
not unlock full training, Slurm training, metric claims, runtime claims, deploy
claims, or paper claims.
