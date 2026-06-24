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

## 2026-06-25 Pro-Finding Fix Status

Status:
`FIXED_FOR_LOCAL_SMOKE_PENDING_FOLLOWUP_PRO`

The current Frame/Token fix supplies a deploy-visible preview metadata source
for the normal train/val/test pipeline by adding
`FrameTokenHybridPreviewProbe` after `LoadFrames` and before decode. The hook
uses only planned `frame_inds` plus prefix masks and writes:

- `frame_token_hybrid_preview_signal`
- `frame_token_hybrid_preview_positions`
- `frame_token_hybrid_preview_source`

The Frame/Token configs collect these keys into `metas`, and the selector now
records the source in `plan["preview_probe"]`. This closes the local metadata
contract that the Pro review flagged: `require_preview_signal=True` can be
satisfied by the normal config path without GT, teacher, oracle, detector
outputs, raw-prediction caches, result JSON, or checkpoints.

The raw-observation/span-token/dense-bridge path remains buildable and locally
smoke-tested:

- observed raw positions are preserved exactly;
- stable gaps are represented as span tokens with `span_start`, `span_end`,
  `role`, `visibility`, and `compression_confidence`;
- dense completion uses masks that distinguish observed, span-derived, and
  completed valid positions;
- invalid mask suffix positions are zeroed;
- ActionFormer receives the selector-rewritten dense bridge and metadata.

Claim boundary remains locked:

- `actual_decode_saving_in_current_actionformer_pipeline=False`
- `raw_decode_saving_claim_allowed=False`
- `pre_decode_loader_hook_reviewed=False`
- `allow_tools_train=False`
- `allow_slurm=False`
- `allow_gpu=False`
- `allow_full_train=False`
- `metric_claim_allowed=False`
- `paper_claim_allowed=False`

The selector package import was narrowed in this route worktree so it no longer
imports PC-OT/MRAS/C3 route classes. This is a route-isolation measure for
`DIVERGENT_INNOVATION_FRAME_TOKEN_HYBRID_DO_NOT_MERGE_WITH_C3`, not a C3
change.

Local verification recorded in the route report:

```text
python -m py_compile ...
PASS

python tools/bata/validate_frame_token_hybrid_gate.py configs/adatad/thumos/frame_token_hybrid_acquisition_full_train_candidate_n16r4.py --json
PASS: fail-closed precheck-only JSON.

C:/Users/skywalker/.conda/envs/torch_1/python.exe -m pytest tests/test_frame_token_hybrid*.py -q
PASS: 31 passed, 1 skipped.

C:/Users/skywalker/.conda/envs/torch_1/python.exe tools/bata/frame_token_hybrid_build_forward_smoke.py
PASS: FRAME_TOKEN_HYBRID_BUILD_FORWARD_SMOKE_PASS.
```

Allowed next action is follow-up Pro review plus local static/smoke checks.
Remote sync, Slurm, long training, `tools/train.py`, `tools/test.py`, mAP,
runtime/FLOPs, deploy/raw-decode-saving, paper claims, and C3/combo attribution
remain locked.
