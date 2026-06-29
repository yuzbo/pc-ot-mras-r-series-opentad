# BVR-TWB / VOI-BBC 92ec024 Launch-Readiness and Pro Gate Packet

Recorded: 2026-06-30T06:35:10+08:00

Route label: `DIVERGENT_INNOVATION_BVR_TWB_DO_NOT_MERGE_WITH_C3`

Stage: Pro gate / launch-readiness evidence package only.

This package is documentation-only. It does not change implementation code, configs, tests, launchers, validators, remote state, Slurm state, checkpoints, or model behavior. It is intended to be sent to GPT-5.5 Pro later for a formal full-train gate decision or a short-diagnostic-first decision.

## Scope and Ownership

- Owned worktree: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_BVR_TWB_LaunchGate92_Worktree_20260630`
- Owned branch: `codex/divergent-bvr-twb-launchgate-92ec024-20260630`
- Base implementation commit under review: `92ec024d1821897245a56f993ed040d97e894831`
- Commit subject: `DIVERGENT_INNOVATION_BVR_TWB_DO_NOT_MERGE_WITH_C3 geometry contract fix`
- Single-owner statement: for this stage, this agent is the only writable owner for BVR launch/Pro evidence package docs in this branch.
- C3/combo boundary: no C3, ABR, MDL, combo, GlobalRank, PhysicalGrid, or other route mixing is claimed or allowed by this packet.

Allowed writes for this stage were limited to:

- `research-wiki/experiments/DIVERGENT_BVR_TWB_92EC024_LAUNCH_GATE_PACKET_20260630.md`
- `logs/bvr_twb_92ec024_pro_launch_prompt_20260630.md`
- `logs/bvr_twb_92ec024_launch_gate_filelist_20260630.txt`
- `research-wiki/experiments/DIVERGENT_BVR_TWB_92EC024_PACKAGE_SELF_CHECK_20260630.md`

## Local Git Evidence

Local command evidence from the owned worktree:

```text
git status --short --branch
## codex/divergent-bvr-twb-launchgate-92ec024-20260630
```

The absence of short-status entries means the owned worktree was locally clean before this package was written.

```text
git rev-parse HEAD
92ec024d1821897245a56f993ed040d97e894831

git branch --show-current
codex/divergent-bvr-twb-launchgate-92ec024-20260630
```

Known remotes:

```text
origin     https://github.com/sming256/OpenTAD.git
pcot-yuzbo https://github.com/yuzbo/pc-ot-mras-r-series-opentad.git
```

`git show --name-status HEAD -1` shows that `92ec024` changed only the geometry-contract repair surface:

```text
M opentad/models/detectors/irregular_actionformer.py
M opentad/models/utils/post_processing/utils.py
A research-wiki/experiments/DIVERGENT_BVR_TWB_GEOMETRY_CONTRACT_FIX_20260630.md
A tests/test_bvr_twb_geometry_contracts.py
A tools/bvr_twb/validate_bvr_twb_geometry_contracts.py
```

## Coordinator-Provided N16R4 Evidence

This packet records the following coordinator-provided evidence exactly as route evidence. These files and tracker rows were not edited by this package stage.

- Remote clean precheck tree: `/data/home/sczc063/run/yuzibo/OpenTAD_BVR_TWB_GeometryPrecheck_20260630_92ec024_20260630_062143`
- Remote log dir: `/data/home/sczc063/run/yuzibo/bvr_twb_precheck_logs_20260630_92ec024_retry_20260630_062312`
- Remote branch: `codex/divergent-bvr-twb-final-20260630`
- Remote HEAD: `92ec024`
- Remote tracked status: clean
- Key log lines:
  - `torch_runtime_contract passed`
  - `full_training_unlocked false`
  - `no_metric_claim true`
  - `no_training true`
  - `no_video_decode true`
  - `PRECHECK_EXIT=0`
- Main tracker row `1.998407` records this as `PRECHECK_ONLY` and still locked.

Interpretation: the Linux/N16R4 torch geometry precheck passed for commit `92ec024`, but it explicitly remained no-training, no-video-decode, no-metric, and full-training-locked.

## Previous Severe Result Context

Earlier BVR-TWB / VOI-BBC evidence on old stable branch `cf662dd` produced a severe failure-scale result:

- Average-mAP around `1.40`.
- Training loss was finite.
- The failure triggered the severe-result escalation gate.

Accepted old GPT-5.5 Pro severe diagnosis boundary:

- The `cf662dd` result was more consistent with detector-facing temporal geometry, mask, native-axis, or post-processing seconds-conversion integration failure than with BVR-TWB / VOI-BBC being invalid as a research route.
- Pro did not approve long training, mAP claims, runtime claims, deployment claims, paper claims, or true sparse-compute claims.
- Pro required minimum proof before any new serious training: detector temporal grid must use native/detector feature positions; mask and valid length must match; `remap_gt_to_selected_axis=False` must preserve native dense coordinates; post-processing seconds conversion must be valid under native axis; and forced-uniform BVR bridge sanity must show bridge/head/eval geometry does not collapse independent of selector quality.

## Why 92ec024 Addresses the Severe Geometry Failure Class

Commit `92ec024` is targeted at the old Pro severe-diagnosis boundary. It does not claim selector quality or detector performance. It specifically addresses geometry/mask/native-axis/post-processing risks:

- `IrregularActionFormer` now recognizes BVR-TWB metadata and uses `bvr_twb_detector_feature_positions` plus `bvr_twb_detector_feature_valid_len` for the detector temporal grid.
- BVR metadata no longer silently falls back to raw selected positions, legacy `irregular_selected_positions`, or `arange` when required detector-feature metadata is missing.
- BVR detector grid requires `irregular_native_axis=True`, because `remap_gt_to_selected_axis=False` keeps Head coordinates on the native dense axis.
- BVR detector grid fail-closes when detector feature positions are empty, when `mask.sum()` does not equal detector feature position count, or when native valid length does not exceed the last detector feature position.
- `convert_to_seconds` fail-closes if BVR detector-feature metadata exists but `irregular_native_axis` is false, preventing selected-axis interpolation from contaminating seconds conversion.
- `tools/bvr_twb/validate_bvr_twb_geometry_contracts.py --require-torch` validates source contracts, a forced-uniform adapter bridge sanity case, loader/runtime metadata, detector feature centers, native-axis GT preservation, and post-processing seconds conversion.
- The N16R4 precheck evidence above reports `torch_runtime_contract passed` and `PRECHECK_EXIT=0` while keeping all claim locks.

Important boundary: this is evidence that the prior geometry failure class has a targeted fix and Linux torch precheck. It is not a detector mAP result.

## Final Candidate Config Under Review

Final BVR candidate config:

```text
configs/adatad/thumos/input_bvr_twb_dynamic_adapter_irregular_headv3.py
```

Key route/config properties observed locally:

- Inherits: `./input_random_fixed_50pct_adapter_irregular_headv3_x.py`
- Route label: `DIVERGENT_INNOVATION_BVR_TWB_DO_NOT_MERGE_WITH_C3`
- Dense window size: `384`
- Adapter target window size: `192`
- Input method: `bvr_twb_dynamic_subsample`
- Train method base: `random_trunc`
- Val/test method base: `sliding_window`
- `keep_ratio=0.5` remains a controlled first trainable target, not the final dynamic-acquisition claim.
- `remap_gt_to_selected_axis=False`
- `bvr_twb_adapter_bridge_mode="adapter_fixed_length_padded_bridge"`
- Train split: `bvr_twb_train_value_labels=True`
- Val/test split: `bvr_twb_train_value_labels=False`
- Formal scout/value gate:
  - `bvr_twb_scout_source="deploy_visible_raw_or_metadata_scout"`
  - `bvr_twb_require_deploy_visible_scout=True`
  - `bvr_twb_allow_diagnostic_preview_fallback=False`
  - `bvr_twb_value_mode="deploy_heuristic_voi"`
- Stability config:
  - `solver.amp=False`
  - `solver.fp16_compress=False`
  - `model.rpn_head.max_reg_log_distance=6.0`
  - `model.rpn_head.regression_head_fp32=True`
  - `model.rpn_head.regression_loss_fp32=True`
  - `model.rpn_head.filter_invalid_regression_samples=False`
  - `model.rpn_head.min_regression_segment_length=1e-6`
- Work dir: `exps/thumos/adatad/input_bvr_twb_dynamic_adapter_irregular_headv3`

## Gate Conditions Known From Existing Code

Launch gate:

```text
tools/bvr_twb/validate_bvr_twb_launch_gate.py
```

Known fail-closed checks include:

- Config must contain `method="bvr_twb_dynamic_subsample"`.
- Config must contain the exact BVR route label.
- Config text must reject forbidden route tokens after excluding only the route label and benign `checkpoint_interval`.
- Config must include `adapter_fixed_length_padded_bridge`.
- Train config must enable train-only value labels; val/test config must explicitly disable them.
- Formal scout/value path must use deploy-visible scout and `deploy_heuristic_voi`.
- Diagnostic deterministic fallback and silent learned-value fallback are rejected.
- Precheck summary must have correct route label, `blocked=false`, `all_validated=true`, adapter bridge evidence, padding duplicates invalid, formal preview/scout/value modes, `value_labels_used_at_test=false`, `sparse_compute_claim=false`, `no_training=true`, and `no_metric_claim=true`.
- Gate output still returns `full_train_unlocked=false` and `remote_sync_unlocked_by_local_gate=false`.

Geometry contract validator:

```text
tools/bvr_twb/validate_bvr_twb_geometry_contracts.py --require-torch
```

Known proof obligations include:

- Source code contains BVR detector temporal-grid and BVR post-processing fail-closed tokens.
- Forced-uniform bridge sanity has `dense_T=384`, `target_frame_num=192`, `raw_valid_k=96`, and detector feature centers derived from raw selected positions with feature stride `2`.
- Adapter padding duplicates remain invalid.
- Torch runtime contract imports OpenTAD `LoadFrames`, `IrregularActionFormer`, and `convert_to_seconds`.
- Runtime validates detector feature positions, native-axis GT preservation, and seconds conversion.
- Summary remains `no_training=true`, `no_video_decode=true`, `no_metric_claim=true`, and `full_training_unlocked=false`.

## Remaining Uncertainties Before Full Train

These remain unresolved by this package and must be answered by Pro, coordinator, or explicit user override before formal long training:

1. Pretrained/checkpoint path:
   - The config inherits the Adapter family and likely depends on the inherited pretrained path.
   - This package did not verify the exact checkpoint file on the target remote clean tree.
   - A full-train command must not launch until the inherited `pretrain`/resume/checkpoint path is confirmed present and intended.

2. Exact remote clean tree for full train:
   - The passed precheck tree is `/data/home/sczc063/run/yuzibo/OpenTAD_BVR_TWB_GeometryPrecheck_20260630_92ec024_20260630_062143`.
   - It is known to be clean at branch `codex/divergent-bvr-twb-final-20260630`, HEAD `92ec024`.
   - The future formal-train tree should be explicitly named and confirmed clean before launch; do not assume an old mixed runtime tree is acceptable.

3. GPU1 plan:
   - Earlier diagnostics used GPU1 under protected hold policies, but this packet does not authorize or modify any hold.
   - Any GPU1 formal or diagnostic plan must state exact allocation, job id if applicable, CUDA device, log path, and no protected-hold release action.

4. Command:
   - This packet does not define a final `tools/train.py` command.
   - The eventual command must include the exact config, work dir/run id, environment activation, inherited pretrained/checkpoint verification, and log capture.

5. Stop/continue rules:
   - The next launch decision must specify whether to run a short diagnostic first or formal full training.
   - If short diagnostic is required, define expected duration/epoch count and blockers such as import failure, data/pretrain missing, shape/mask mismatch, NaN/Inf, non-finite gradients, mAP collapse-scale symptoms, or post-processing errors.
   - If formal full training is allowed, define immediate stop triggers and evidence to record before any interpretation.

## Questions For GPT-5.5 Pro

Please answer in Chinese and inspect the GitHub branch/commit plus this packet before deciding.

1. Is commit `92ec024` plus the N16R4 `torch_runtime_contract passed` PRECHECK_ONLY evidence enough to allow formal full training, or should BVR require a short diagnostic-only run first?
2. Please review native-axis handling: `remap_gt_to_selected_axis=False`, BVR detector feature positions, detector temporal-grid centers, valid length, cell widths, and GT/native coordinate consistency.
3. Please review detector feature position semantics: are `bvr_twb_detector_feature_positions` correctly separated from raw selected positions and Adapter padded input positions?
4. Please review mask semantics: does the detector valid mask count feature centers only, not Adapter padding duplicates, and does the fail-closed gate catch mismatch?
5. Please review post-processing seconds conversion: under BVR native-axis, are proposals correctly treated as native detector-feature coordinates and converted by snippet stride, offset, window start, and fps?
6. Is there any validation/test GT leakage, train-only label leakage, teacher/cache shortcut, raw-prediction shortcut, or hidden post-processing shortcut?
7. Has this route drifted toward C3/C3-Pro, ABR, MDL, combo, uniform fallback, or another route in a way that invalidates BVR-TWB / VOI-BBC attribution?
8. Is the `adapter_fixed_length_padded_bridge` compatible with a deployable BVR route while keeping the true sparse-compute claim locked?
9. Are the current strict claim locks sufficient: no mAP/runtime/FLOPs/deploy/paper/true sparse-compute claim, no `tools/test.py`, and no long train until Pro or explicit override?
10. Is the unresolved pretrained/checkpoint path a blocker before any formal full train? If yes, what exact prelaunch verification is minimally required?
11. What should be the allowed next action: formal full train, short diagnostic-only run, fix-before-run, or context insufficient?

Requested Pro output schema:

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

Preferred verdict labels:

- `PASS_ALLOW_FORMAL_FULL_TRAIN_AFTER_PRETRAIN_REMOTE_COMMAND_RESOLVED`
- `REQUIRE_SHORT_DIAGNOSTIC_ONLY_BEFORE_FULL_TRAIN`
- `FIX_BEFORE_ANY_TRAIN`
- `CONTEXT_INSUFFICIENT_GITHUB_NOT_INSPECTED`
- `REJECT_ROUTE_DRIFT_OR_LEAKAGE`

## Strict Claim Locks

The following remain locked:

- No mAP claim.
- No validation/test metric claim.
- No runtime/FLOPs/latency claim.
- No deployment-readiness claim.
- No paper claim.
- No true sparse-compute claim.
- No claim that BVR selector quality improves TAD.
- No claim that the `cf662dd` severe failure is fully explained beyond the geometry/mask/native-axis/post-processing diagnosis boundary.
- No `tools/test.py`.
- No long training until a valid Pro decision or explicit user override.
- No remote sync, SSH, Slurm, protected hold action, or training action is authorized by this documentation package.

The current positive evidence is limited to implementation/code inspection, local source evidence, local Git evidence, and coordinator-provided N16R4 geometry PRECHECK_ONLY evidence at `92ec024`.

## File List And GitHub URL Placeholders

Primary review files:

- `configs/adatad/thumos/input_bvr_twb_dynamic_adapter_irregular_headv3.py`
- `opentad/models/detectors/irregular_actionformer.py`
- `opentad/models/utils/post_processing/utils.py`
- `tools/bvr_twb/validate_bvr_twb_geometry_contracts.py`
- `tools/bvr_twb/validate_bvr_twb_launch_gate.py`
- `tests/test_bvr_twb_geometry_contracts.py`
- `research-wiki/experiments/DIVERGENT_BVR_TWB_GEOMETRY_CONTRACT_FIX_20260630.md`
- `research-wiki/experiments/DIVERGENT_BVR_TWB_92EC024_LAUNCH_GATE_PACKET_20260630.md`
- `logs/bvr_twb_92ec024_pro_launch_prompt_20260630.md`
- `logs/bvr_twb_92ec024_launch_gate_filelist_20260630.txt`
- `research-wiki/experiments/DIVERGENT_BVR_TWB_92EC024_PACKAGE_SELF_CHECK_20260630.md`

GitHub placeholders after package push:

- Branch URL: `https://github.com/yuzbo/pc-ot-mras-r-series-opentad/tree/codex/divergent-bvr-twb-launchgate-92ec024-20260630`
- Base implementation commit URL: `https://github.com/yuzbo/pc-ot-mras-r-series-opentad/commit/92ec024d1821897245a56f993ed040d97e894831`
- Package commit URL: `<fill after package commit/push>`
