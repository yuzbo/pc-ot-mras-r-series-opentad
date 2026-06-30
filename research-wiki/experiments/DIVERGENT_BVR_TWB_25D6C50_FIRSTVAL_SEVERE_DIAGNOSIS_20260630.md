# DIVERGENT BVR-TWB 25d6c50 First Validation Severe Diagnosis

## Route Boundary

- Route label: `DIVERGENT_INNOVATION_BVR_TWB_DO_NOT_MERGE_WITH_C3`
- Branch: `codex/divergent-bvr-twb-formalgate-20260630`
- Active run commit: `25d6c50595407a98c6c957e3a488b98aab5c1f4f`
- Remote tree: `/data/home/sczc063/run/yuzibo/OpenTAD_BVR_TWB_FormalGate_20260630_7d66885`
- Child: `1118197.467 bvr_twb_g0_r4`
- Parent hold: `1118197 pcot_dbg2g` protected, preserved
- GPU boundary: BVR used GPU0; GPU1 remains reserved for C3

No C3, CADF, PQR, ABR, MDL, GlobalRank-ST, or combo attribution is mixed into this route.

## Trigger

The first validation for child `1118197.467` produced:

```text
Average-mAP: 22.88 (%)
mAP@0.30: 42.23
mAP@0.40: 33.32
mAP@0.50: 21.24
mAP@0.60: 12.14
mAP@0.70: 5.48
predictions: 365940
ground truth: 3325
```

The run continued after that point with finite losses into epoch 43. Hard-error scan found no Traceback, RuntimeError, CUDA OOM, Killed, No space, NaN/cost NaN, non-finite, ChildFailedError, FileNotFoundError, or PermissionError markers in the inspected log evidence.

This is a severe result relative to the recorded references: random-fixed Adapter `63.77`, strict EMA `63.85`, stratified `64.64`, uniform stride-2 `65.09`, exact-uniform `65.57/65.73`, and oracle-boundary Adapter about `77.62`.

## Pro Submission Evidence

- Prompt: `research-wiki/experiments/DIVERGENT_BVR_TWB_25D6C50_FIRSTVAL_SEVERE_PRO_PROMPT_20260630.md`
- Context package: `logs/bvr_twb_25d6c50_firstval_severe_pro_20260630/bvr_twb_25d6c50_firstval_severe_context_20260630.zip`
- Context package SHA256: `e3d888cc0218f00014918b704435f728301b52ec8ff2eac52454dcecc4bff177`
- Prompt SHA256: `b45df5d81bb83158457abd8f1e1af0f8c536513bf555cf88564582f745dd1d9f`
- Rosetta recall: `bvr-twb-25d6c50-firstval-severe-20260630`
- Chrome CDP: `127.0.0.1:9223`; protocol-preferred `9333` was not reachable in this thread, and prior user instruction allowed 9223.
- Model evidence from stderr: `gpt-5-5-pro`
- Conversation: `6a4380c9-95d4-83ee-9c65-cc7a6635ca79`
- Message: `ae802a73-5273-402d-b45d-c7fcecbce335`
- Elapsed: `657887ms` from Rosetta stderr; wrapper process duration `663.125s`
- Stdout: `logs/bvr_twb_25d6c50_firstval_severe_pro_20260630/rosetta_bvr_25d6c50_firstval_severe.stdout.txt`
- Stderr: `logs/bvr_twb_25d6c50_firstval_severe_pro_20260630/rosetta_bvr_25d6c50_firstval_severe.stderr.txt`
- Process metadata: `logs/bvr_twb_25d6c50_firstval_severe_pro_20260630/rosetta_bvr_25d6c50_firstval_severe.process.json`

Transport note: the wrapper exit code file records `-1`, but stdout is substantive and structured, stderr contains model/conversation/message evidence, and `process.json` records `threads_has_recall=true`. This is accepted as a valid Pro diagnosis with a wrapper-exit anomaly, not as an empty or file-blind transport failure.

## Accepted Pro Verdict

Machine-readable Pro conclusion:

```text
CONTEXT_VISIBLE_FOR_SEVERE_DIAGNOSIS
VERDICT=SEVERE_IMPLEMENTATION_OR_PROTOCOL_FAILURE_NOT_IDEA_REJECTION
CHILD_DECISION=STOP_CHILD_PRESERVE_PARENT
PARENT_HOLD=DO_NOT_RELEASE
GPU1=DO_NOT_USE
NEXT_ALLOWED=READONLY_BOUNDED_DIAGNOSTIC_ONLY
FOLLOWUP_LONG_RUN=LOCKED
```

Main interpretation:

- `22.88` is a severe implementation/protocol failure, not evidence that BVR-TWB / VOI-BBC as an idea is dead.
- The improvement from old `cf662dd` first validation `1.40` to current `25d6c50` first validation `22.88` suggests the pretrain/geometry repairs helped, but the full detector-facing path is still not baseline-recovering.
- The curve shape, especially `mAP@0.3=42.23` versus `mAP@0.7=5.48`, is most consistent with high-IoU localization failure or detector/bridge/postprocess calibration failure rather than a pure classification failure.

## Ranked Root-Cause Hypotheses Accepted

1. Detector-facing native temporal geometry, point grid, or post-processing conversion still has a runtime-level mismatch.
2. Padded bridge, mask, valid length, and backbone-output feature time axis may still be semantically mismatched even if source-level fail-closed checks pass.
3. The current selector path is closer to `deploy_heuristic_voi` than a fully learned VOI/BVR policy; selector/scout quality may be insufficient, but selector quality alone is unlikely to explain a collapse to `22.88`.
4. Proposal count, score calibration, NMS, and proposal cap may bury the few high-IoU proposals that exist.
5. Pretrain loading must be audited from the actual runtime log; it is not the leading hypothesis if the runtime log proves real loading.
6. Continuing blind to the next validation has low information value.

## Stop Evidence

Accepted action was executed as child-only cancellation:

```text
before:
1118197.467|bvr_twb_g0_r4|RUNNING|03:45:13|0:0|g0030
1118197 pcot_dbg2g RUNNING 5-11:25:27 g0030

command:
scancel 1118197.467

after:
1118197.467|bvr_twb_g0_r4|CANCELLED by 1258|03:45:14|0:9|g0030
1118197 pcot_dbg2g RUNNING 5-11:25:30 g0030
```

The protected parent hold was not released, cancelled, requeued, replaced, or modified.

## Minimum Next Diagnostic Tree

Pro recommended bounded diagnostics only:

1. Runtime-log audit for actual pretrain loading, checkpoint lines, missing/unexpected keys, and any hidden `no pretrain path is provided` evidence.
2. Small validation-window ledger/bridge dump: `valid_k`, `detector_feature_valid_k`, `max_gap`, selected positions, detector feature positions, mask count, dense length, preview source, budget stop reason, exact-uniform overlap, and offline GT-only coverage statistics.
3. Forced-uniform-through-BVR-bridge sanity gate using the same detector/head/postprocess path. If this also stays near `20-30`, the main fault is bridge/detector/postprocess rather than selector quality.
4. Single-batch geometry round-trip dump from native GT segments through selected raw positions, detector feature centers, positive assignment, decoded native segments, seconds conversion, and evaluator input.
5. Proposal/NMS/score diagnostic: pre-NMS proposals, post-NMS proposals, per-class top-k, top-k recall, and score-IoU correlation.

## Locked State

Still locked:

- releasing or cancelling parent hold `1118197 pcot_dbg2g`
- using GPU1
- any new BVR formal/full/long training
- any blind BVR follow-up long run
- official `tools/test.py` result claim
- final metric, runtime/FLOPs, deployment, paper, or sparse-compute claim
- C3/CADF/PQR/ABR/MDL/GlobalRank-ST route mixing or attribution transfer

Allowed:

- read-only evidence harvest
- bounded local/remote diagnostics that do not train a new long run
- route-owned code-owner implementation of the diagnostic tree after explicit assignment
