# GPT-5.5 Pro Prompt: BVR-TWB / VOI-BBC 92ec024 Launch Gate

请使用 GPT-5.5 Pro，高推理强度，中文回答。请不要把这个任务当作普通摘要；这是 Temporal Action Detection 研究路线的 formal full-train gate / short-diagnostic gate 决策。

## Decision Needed

Route label: `DIVERGENT_INNOVATION_BVR_TWB_DO_NOT_MERGE_WITH_C3`

请判断：commit `92ec024d1821897245a56f993ed040d97e894831` 加上 Linux/N16R4 torch geometry PRECHECK_ONLY 证据，是否足以允许 BVR-TWB / VOI-BBC 进入 formal full train；还是必须先做 short diagnostic-only run；或者仍需先修复代码/配置/协议问题。

## Current Branch / Commit Evidence

Owned worktree branch:

```text
codex/divergent-bvr-twb-launchgate-92ec024-20260630
```

Base implementation commit under review:

```text
92ec024d1821897245a56f993ed040d97e894831
```

Commit subject:

```text
DIVERGENT_INNOVATION_BVR_TWB_DO_NOT_MERGE_WITH_C3 geometry contract fix
```

GitHub branch placeholder after package push:

```text
https://github.com/yuzbo/pc-ot-mras-r-series-opentad/tree/codex/divergent-bvr-twb-launchgate-92ec024-20260630
```

Base commit URL:

```text
https://github.com/yuzbo/pc-ot-mras-r-series-opentad/commit/92ec024d1821897245a56f993ed040d97e894831
```

## Coordinator-Provided N16R4 Precheck Evidence

Remote clean precheck tree:

```text
/data/home/sczc063/run/yuzibo/OpenTAD_BVR_TWB_GeometryPrecheck_20260630_92ec024_20260630_062143
```

Remote log dir:

```text
/data/home/sczc063/run/yuzibo/bvr_twb_precheck_logs_20260630_92ec024_retry_20260630_062312
```

Remote state:

```text
branch codex/divergent-bvr-twb-final-20260630
HEAD 92ec024
tracked status clean
```

Key log lines:

```text
torch_runtime_contract passed
full_training_unlocked false
no_metric_claim true
no_training true
no_video_decode true
PRECHECK_EXIT=0
```

Main tracker row `1.998407` records this as `PRECHECK_ONLY` and still locked.

Interpretation: Linux torch geometry/runtime contract passed, but it was explicitly no-training, no-video-decode, no-metric, and full-train locked.

## Severe Result Context

Old BVR-TWB / VOI-BBC branch `cf662dd` had a severe result:

- Average-mAP around `1.40`.
- Training loss looked finite.

Accepted prior GPT-5.5 Pro severe diagnosis boundary:

- The result looked more like detector-facing temporal geometry, mask, native-axis, or post-processing seconds-conversion integration failure than a proof that BVR-TWB / VOI-BBC is invalid.
- No full train, mAP claim, runtime claim, deploy claim, paper claim, or true sparse-compute claim was unlocked.
- Required before serious new training: prove detector temporal grid uses native/detector feature positions; mask/valid length are consistent; `remap_gt_to_selected_axis=False` preserves native dense coordinates; post-processing seconds conversion is correct; forced-uniform BVR bridge/head/eval geometry does not collapse independent of selector quality.

## What 92ec024 Claims To Fix

Commit `92ec024` targets the prior failure class only:

- `IrregularActionFormer` identifies BVR-TWB metadata and uses `bvr_twb_detector_feature_positions` plus `bvr_twb_detector_feature_valid_len` for detector temporal grid.
- BVR path fail-closes instead of falling back to raw selected positions, legacy irregular selected positions, or arange.
- BVR detector grid requires `irregular_native_axis=True` because `remap_gt_to_selected_axis=False` keeps Head coordinates on native dense axis.
- Mask true count must equal detector feature position count.
- Native valid length must exceed the last detector feature position.
- `convert_to_seconds` fail-closes if BVR detector feature metadata exists but native-axis is false.
- `tools/bvr_twb/validate_bvr_twb_geometry_contracts.py --require-torch` checks source tokens, forced-uniform adapter bridge sanity, loader/runtime metadata, detector feature centers, native-axis GT preservation, and seconds conversion.

This does not claim mAP or selector quality.

## Candidate Config Under Review

Config:

```text
configs/adatad/thumos/input_bvr_twb_dynamic_adapter_irregular_headv3.py
```

Key properties:

```text
route_label = "DIVERGENT_INNOVATION_BVR_TWB_DO_NOT_MERGE_WITH_C3"
dense_window_size = 384
window_size = 192
method = "bvr_twb_dynamic_subsample"
keep_ratio = 0.5
remap_gt_to_selected_axis = False
bvr_twb_adapter_bridge_mode = "adapter_fixed_length_padded_bridge"
bvr_twb_scout_source = "deploy_visible_raw_or_metadata_scout"
bvr_twb_require_deploy_visible_scout = True
bvr_twb_allow_diagnostic_preview_fallback = False
bvr_twb_value_mode = "deploy_heuristic_voi"
solver.amp = False
solver.fp16_compress = False
model.rpn_head.max_reg_log_distance = 6.0
model.rpn_head.regression_head_fp32 = True
model.rpn_head.regression_loss_fp32 = True
model.rpn_head.filter_invalid_regression_samples = False
```

Train split uses train-only VOI/regret labels. Val/test explicitly disable those labels. The bridge keeps Adapter fixed-length compatibility but records padding duplicates as invalid; true sparse-compute claim remains locked.

## Gate Conditions

Launch gate file:

```text
tools/bvr_twb/validate_bvr_twb_launch_gate.py
```

Gate requires clean BVR route label, dynamic subsample method, adapter fixed-length padded bridge metadata, train-only value labels only in train, deploy-visible scout, no diagnostic preview fallback, `deploy_heuristic_voi`, no learned value model fallback, correct route precheck summary, `all_validated=true`, padding duplicates invalid, no test value labels, `sparse_compute_claim=false`, `no_training=true`, and `no_metric_claim=true`.

Even a passing gate returns:

```text
full_train_unlocked = false
remote_sync_unlocked_by_local_gate = false
```

Geometry validator file:

```text
tools/bvr_twb/validate_bvr_twb_geometry_contracts.py
```

The N16R4 precheck evidence says `torch_runtime_contract passed` and `PRECHECK_EXIT=0`.

## Known Remaining Uncertainties

Please explicitly judge whether each uncertainty blocks formal full train or only requires a prelaunch check:

1. Pretrained/checkpoint path is not resolved in this packet. The candidate inherits Adapter-family pretrained settings; the exact file on the future remote clean train tree must be confirmed.
2. Exact future remote clean full-train tree is not named yet. The precheck tree exists and is clean, but the formal train tree must be explicitly confirmed before launch.
3. GPU1 plan is not finalized. This packet does not authorize any protected-hold change.
4. Formal command is not finalized. No `tools/train.py` command is approved here.
5. Stop/continue rules are not finalized. Pro should recommend short diagnostic vs formal train and immediate stop triggers.

## Strict Claim Locks

The following must remain locked unless you explicitly say otherwise and justify it:

- no mAP claim;
- no validation/test metric claim;
- no runtime/FLOPs/latency claim;
- no deploy-readiness claim;
- no paper claim;
- no true sparse-compute claim;
- no `tools/test.py`;
- no long training until valid Pro decision or explicit user override;
- no C3/C3-Pro/ABR/MDL/combo mixing;
- no protected hold release or Slurm action from this packet.

## Files To Inspect

Primary:

```text
configs/adatad/thumos/input_bvr_twb_dynamic_adapter_irregular_headv3.py
opentad/models/detectors/irregular_actionformer.py
opentad/models/utils/post_processing/utils.py
tools/bvr_twb/validate_bvr_twb_geometry_contracts.py
tools/bvr_twb/validate_bvr_twb_launch_gate.py
tests/test_bvr_twb_geometry_contracts.py
research-wiki/experiments/DIVERGENT_BVR_TWB_GEOMETRY_CONTRACT_FIX_20260630.md
research-wiki/experiments/DIVERGENT_BVR_TWB_92EC024_LAUNCH_GATE_PACKET_20260630.md
```

Context if needed:

```text
docs/DIVERGENT_BVR_TWB_LOCAL_IMPLEMENTATION_20260629.md
opentad/acquisition/bvr_twb/
tests/test_bvr_twb_opentad_pipeline.py
tests/test_bvr_twb_validators.py
tests/test_bvr_twb_voi_bbc.py
tests/test_bvr_twb_regression_stability.py
```

## Questions

1. Is `92ec024` plus the N16R4 torch PRECHECK_ONLY evidence enough to allow formal full training after pretrained path and command are resolved, or should there be a short diagnostic-only run first?
2. Are native-axis, detector feature positions, masks, GT coordinates, and seconds conversion now internally consistent enough to avoid the `cf662dd` collapse class?
3. Does `adapter_fixed_length_padded_bridge` preserve BVR route attribution while keeping true sparse-compute claims locked?
4. Is there any leakage: validation/test GT, train-only labels at test, teacher/cache shortcut, raw prediction shortcut, or hidden post-processing shortcut?
5. Is there route drift toward C3/C3-Pro/ABR/MDL/combo/uniform fallback?
6. Is unresolved pretrained/checkpoint path a hard blocker before full train?
7. What exact next action is allowed?

## Required Answer Format

Please answer with:

```text
Context verdict:
Model evidence:
Inspected materials:
Verdict:
Blocking findings:
Non-blocking findings:
Required fixes or next experiments:
Accepted launch/sync/review/Slurm decision:
Claim locks still active:
```

Use one of these verdict labels if possible:

```text
PASS_ALLOW_FORMAL_FULL_TRAIN_AFTER_PRETRAIN_REMOTE_COMMAND_RESOLVED
REQUIRE_SHORT_DIAGNOSTIC_ONLY_BEFORE_FULL_TRAIN
FIX_BEFORE_ANY_TRAIN
CONTEXT_INSUFFICIENT_GITHUB_NOT_INSPECTED
REJECT_ROUTE_DRIFT_OR_LEAKAGE
```
