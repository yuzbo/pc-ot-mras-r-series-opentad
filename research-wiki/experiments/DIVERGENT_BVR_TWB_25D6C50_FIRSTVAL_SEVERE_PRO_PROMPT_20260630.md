# DIVERGENT_BVR_TWB_25D6C50 First-Validation Severe Result Diagnosis Prompt

Decision requested: severe-result diagnosis and stop/continue gate for the active BVR-TWB / VOI-BBC run.

Required answer language: Chinese.

Required answer sections:

1. `Context verdict`
2. `Model evidence`
3. `Inspected materials`
4. `Verdict`
5. `Most likely root causes, ranked`
6. `Stop / continue decision for child 1118197.467`
7. `Minimum next diagnostic tree`
8. `Required fixes before any follow-up long run`
9. `Allowed next action`
10. `Still locked`

If the attached repository files or this prompt are not visible, answer `INCOMPLETE_CONTEXT_NOT_VISIBLE` and do not give a launch or stop/continue verdict.

## Route Boundary

Route label: `DIVERGENT_INNOVATION_BVR_TWB_DO_NOT_MERGE_WITH_C3`.

Do not mix this route with C3, GlobalRank-ST, CADF, PQR, ABR, MDL, or any C3 mainline attribution. This is a divergent BVR-TWB / VOI-BBC route.

The current active run is not a paper claim, deployment claim, runtime/FLOPs claim, sparse-compute claim, or final metric claim.

## Core Idea To Preserve

BVR-TWB / VOI-BBC is intended to select frames by value of information / regret risk:

- do not directly learn a brittle boundary head as the only selector signal;
- estimate where the final detector would regret not observing raw frames;
- combine deploy-visible preview/scout/value signals with train-only regret supervision;
- protect action/background transitions, uncertain candidate regions, and long unobserved spans;
- feed irregular selected observations to a detector path that respects native temporal geometry.

The design is not merely a uniform scaffold or engineering fallback. If the result is bad, please distinguish:

- the BVR idea is flawed;
- the current implementation fails the idea;
- the detector-facing geometry/post-processing path is still wrong;
- the pretrain/config/launcher/training protocol is wrong;
- the run should continue because first validation is misleading.

## Current Run Evidence

Remote tree:

`/data/home/sczc063/run/yuzibo/OpenTAD_BVR_TWB_FormalGate_20260630_7d66885`

GitHub branch:

`https://github.com/yuzbo/pc-ot-mras-r-series-opentad/tree/codex/divergent-bvr-twb-formalgate-20260630`

Active commit:

`25d6c50595407a98c6c957e3a488b98aab5c1f4f`

Important branch ancestry:

- `25d6c50` contains `478325a` pretrain gate fix.
- It also contains the prior geometry/native-axis contract fix lineage from `92ec024`.
- Therefore this is not simply the older broken `cf662dd` branch.

Remote config:

`configs/adatad/thumos/input_bvr_twb_dynamic_adapter_irregular_headv3_n16r4.py`

Runtime log:

`/data/home/sczc063/run/yuzibo/OpenTAD_BVR_TWB_FormalGate_20260630_7d66885/logs/bvr_twb_formal_gpu0_retry4_20260630_130539_+0800/srun_stdout_stderr.log`

Slurm:

`1118197.467|bvr_twb_g0_r4|RUNNING|03:31:20|0:0|g0030`

Resource boundary:

- Parent hold `1118197 pcot_dbg2g` is protected and must not be released.
- This BVR child uses GPU0.
- GPU1 is reserved/used by C3 and must not be used by this route.

First validation:

```text
2026-06-30 16:29:20 Train INFO: Number of ground truth instances: 3325
2026-06-30 16:29:20 Train INFO: Number of predictions: 365940
2026-06-30 16:29:20 Train INFO: Average-mAP: 22.88 (%)
2026-06-30 16:29:20 Train INFO: mAP at tIoU 0.30 is 42.23%
2026-06-30 16:29:20 Train INFO: mAP at tIoU 0.40 is 33.32%
2026-06-30 16:29:20 Train INFO: mAP at tIoU 0.50 is 21.24%
2026-06-30 16:29:20 Train INFO: mAP at tIoU 0.60 is 12.14%
2026-06-30 16:29:20 Train INFO: mAP at tIoU 0.70 is 5.48%
```

After the first validation, the run continued with finite losses:

```text
2026-06-30 16:30:21 [042][00050/00199] Loss=1.0762 cls_loss=0.2355 reg_loss=0.4461 boundary_loss=0.3945
2026-06-30 16:31:22 [042][00100/00199] Loss=1.1123 cls_loss=0.2548 reg_loss=0.4560 boundary_loss=0.4015
2026-06-30 16:32:23 [042][00150/00199] Loss=1.1180 cls_loss=0.2622 reg_loss=0.4474 boundary_loss=0.4084
2026-06-30 16:33:22 [042][00199/00199] Loss=1.1347 cls_loss=0.2645 reg_loss=0.4498 boundary_loss=0.4204
2026-06-30 16:34:21 [043][00050/00199] Loss=1.1106 cls_loss=0.2721 reg_loss=0.4433 boundary_loss=0.3951
```

Hard-error scan found no obvious crash markers for Traceback, RuntimeError, CUDA OOM, Killed, No space, NaN/cost NaN, non-finite, ChildFailedError, FileNotFoundError, or PermissionError.

## Baseline References

Relevant THUMOS14 anchors:

- random-fixed Adapter baseline: `63.77` Avg-mAP
- strict EMA reference: `63.85`
- stratified sampling best: `64.64`
- uniform stride-2 50% reference: `65.09`
- exact-uniform family: `65.57 / 65.73`
- residual64 fallback: `65.46`
- oracle residual64: `66.61`
- oracle-boundary Adapter: about `76-78`, recorded `77.62`

Thus `22.88` is a severe result relative to all relevant references.

## Prior Severe-Result Context

An older BVR stability branch `cf662dd` produced first validation `Average-mAP 1.40`. GPT-5.5 Pro diagnosed that as likely detector-facing geometry/mask/post-processing integration failure, not proof against the BVR idea. The old child was stopped and a geometry/native-axis/pretrain repair sequence followed.

The current `25d6c50` run was intended to include those repairs. Yet first validation is still far too low, although better than `1.40`.

## What I Need From You

Please diagnose, grounded in the attached code/config/docs, whether the current `22.88` is most likely caused by:

- BVR-TWB / VOI-BBC idea failure;
- still-broken detector temporal grid / native-axis / post-processing conversion;
- pretrain not actually loading or ineffective;
- padded bridge / mask / valid length mismatch;
- mismatch between selected frames, detector features, GT coordinates, and evaluator;
- too many predictions / proposal cap / NMS or score calibration issue;
- short/first validation being misleading and worth continuing;
- some other concrete issue visible in files.

Please give a concrete stop/continue verdict for child `1118197.467`:

- `STOP_CHILD_PRESERVE_PARENT`
- `CONTINUE_TO_NEXT_VALIDATION`
- `CONTINUE_TO_FINAL_ONLY_IF...`
- `INCONCLUSIVE_NEED_ONE_QUICK_READONLY_AUDIT`

Do not recommend releasing or cancelling parent hold `1118197`.

Also give the minimum next diagnostic tree. It should be small and high-information. Prefer diagnostics that can distinguish selector quality from detector geometry/post-processing failure without launching another blind long run.

## Attached Materials

The context package should include:

- this prompt;
- active BVR config;
- BVR acquisition implementation files;
- BVR loader / bridge / detector / post-processing files;
- BVR validators and focused tests;
- key BVR route reports;
- latest tracker/log rows.

If GitHub inspection is possible, inspect the branch URL above as well as the attached materials.
