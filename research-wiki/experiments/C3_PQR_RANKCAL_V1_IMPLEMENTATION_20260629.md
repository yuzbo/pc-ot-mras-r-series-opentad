# C3 PQR RankCal V1 Implementation - 2026-06-29

## Scope

Route: `C3_MAINLINE_OPTIMIZATION` / `C3_ORIGINAL_OPTIMIZATION_ROUTE`.

Variant: `C3_PQR_RankCalV1_MaxIoU`.

Changed surface: detector-head proposal ranking calibration through the existing
`AnchorFreeHead` quality-head path.

Important boundary after read-only review fix: this clean `588b272` snapshot
does not contain `PCOTMRASIndirectPreBackboneFrameSelector`, and
`ActionFormer`/`SingleStageDetector` do not consume `model.frame_selector`.
Therefore this implementation is a real, buildable Adapter + ActionFormer
backend ranking-calibration control, not a C3 selector input experiment. A C3
selector-input version must be moved later into a tree that actually contains
the C3 selector implementation and detector wiring.

This route does not change CADF selector logic, sampler code, backbone,
projection, neck, primary assignment, primary regression, NMS parameters, raw
prediction cache use, teacher use, test-time GT use, dynamic budgeting, or
physical-time post-processing.

## Implemented

- Added fail-closed PQR RankCal V1 backend-calibration configs:
  - `configs/adatad/thumos/c3_indirect_original_adatad_32px_a_pqr_rankcal_v1_precheck.py`
  - `configs/adatad/thumos/c3_indirect_original_adatad_32px_a_pqr_rankcal_v1_shortdiag.py`
  - `configs/adatad/thumos/c3_indirect_original_adatad_32px_a_exact_uniform_backend_control_pqr_rankcal_v1_shortdiag.py`
- The first two configs inherit from the real
  `input_random_fixed_50pct_adapter.py` Adapter backend.
- The third config is a real stride-2 uniform Adapter backend control. It does
  not claim to alter any selector quota.
- Reused the existing `AnchorFreeHead` quality-head framework:
  - `target_mode="max_iou"`
  - `loss_weight=0.03`
  - `score_alpha=0.10`
  - neutral quality init with `weight_init=0.0` and `bias_init=4.59511985013459`
- Added `tools/validate_c3_pqr_rankcal_v1_config.py`.
- Added focused config and quality-head tests.

## Safety Gates

The validator requires:

- route label `C3_MAINLINE_OPTIMIZATION`;
- route family `C3_ORIGINAL_OPTIMIZATION_ROUTE`;
- no route labels containing `BH`, `BH-SDC`, `BH_SDC`, `DIVERGENT`, or `CADF`;
- raw prediction cache disabled;
- teacher and test-GT flags disabled;
- physical-time postprocess claim disabled;
- `max_seg_num=2000`;
- quality target `max_iou`;
- quality `loss_weight` in `[0.02, 0.05]`;
- quality `score_alpha` in `[0.05, 0.15]`;
- no `model.frame_selector`;
- no unconsumed top-level `cfg.model` keys relative to the current
  `ActionFormer.__init__` signature;
- either random-fixed Adapter 50% backend or stride-2 uniform Adapter 50%
  backend;
- no selector-quota claim in the exact/uniform backend control.

## Read-Only Review Fix

Blocking issue 1 was valid: the earlier configs parsed but were not buildable
in this snapshot because they introduced `model.frame_selector` without a
registered selector or detector consumer.

Blocking issue 2 was valid: the first validator checked the synthetic
`frame_selector` fields and could pass a config that the current detector would
ignore or reject.

Fix applied:

- removed all `model.frame_selector` blocks from PQR configs;
- changed the main PQR configs to inherit from current buildable Adapter +
  ActionFormer configs;
- changed the exact-uniform control into a real stride-2 uniform Adapter backend
  control;
- added validator static signature guard for `ActionFormer.__init__`;
- added tests that reject any unconsumed `frame_selector`.

## Remote PRECHECK_ONLY Follow-Up

Remote PRECHECK_ONLY on Linux exposed one test-fixture bug after the config and
static validator gates passed:

- remote py_compile: pass;
- three config validator invocations: pass;
- focused pytest: `12 passed, 3 failed`;
- all three failures were torch-backed quality-head behavior tests.

Root cause: the test helper instantiated `AnchorFreeHead(loss=...)` with a
plain Python `dict`, while the current production constructor expects the same
attribute-style loss config used by real configs (`loss.cls_loss` and
`loss.reg_loss`). This was a test fixture/config-object mismatch, not a PQR
production behavior requirement.

Fix applied:

- changed the quality-head test helper to pass an `mmengine.config.ConfigDict`
  loss config with `cls_loss` and `reg_loss`;
- added a focused fixture test so future edits do not silently reintroduce a
  plain dict;
- did not change production `AnchorFreeHead` behavior for this mismatch.

The optional no-data build-only check remains locked by baseline clean-snapshot
import dependencies rather than by PQR-added fields. In the current `588b272`
tree, detector/config build first exposes missing `Rearrange` transform
registration, and after that is patched externally it exposes missing
`opentad.datasets.transforms.pseudo_boundary`. The PQR validator and configs now
record this as
`build_only_status="locked_by_baseline_import_dependencies"` and limit the local
PRECHECK_ONLY scope to config validation plus quality-head unit tests.

## Runtime Gate Fix - 2026-06-30

GPT-5.5 Pro accepted the route alignment and no-leakage boundary but returned
`FIX_BEFORE_RUNTIME` because `workflow.max_train_iters=2` was only a config
field. The standard `tools/train.py` / `train_one_epoch` path did not consume
it, so the planned 2-iteration smoke was not enforceable.

Fix applied in the same route-owned worktree:

- `tools/train.py` now reads and normalizes `cfg.workflow.max_train_iters`.
  Unset or `<=0` means default unchanged behavior. Positive `N` becomes a
  global per-process train-iteration cap for the whole run.
- `tools/train.py` tracks completed train iterations across epochs, passes the
  remaining budget to `train_one_epoch`, and exits the training loop before
  checkpoint/validation/evaluation once the gate is reached.
- `opentad/cores/train_engine.py` now accepts optional `max_train_iters`, hard
  stops the epoch loop after the requested number of attempted train batches,
  logs the event, and returns the completed iteration count.
- `tools/validate_c3_pqr_rankcal_v1_config.py` now fail-closed checks that the
  standard launcher and epoch loop consume the runtime gate.
- `tests/test_c3_pqr_rankcal_v1_config.py` now includes fake-runtime tests
  proving `max_train_iters=2` stops at 2 and the unset default executes a full
  fake epoch.

Evidence report:
`research-wiki/experiments/c3_pqr_rankcal_v1_runtime_gate_fix_20260630.md`.

This fix does not alter CADF selector logic, BH-SDC, evaluator,
post-processing, input sampling, PQR quality/ranking math, teacher/cache/GT
use, or mAP claims.

## 2026-06-30 Clean Clone Dependency Completeness Fix

Remote clean clone PRECHECK later failed at branch HEAD
`30ea4f1a4970416ad744a2d5429b8b37fddb547d` with:

```text
ModuleNotFoundError: No module named 'opentad.models.backbones.time_aligned_rasterizer'
```

The error came from `opentad/models/backbones/vit_adapter.py:18` while running
the three torch-backed tests in
`tests/test_c3_pqr_rankcal_v1_quality_head.py`. The root cause was dependency
completeness: `vit_adapter.py` imports `TimeAlignedRasterizer`, and the module
exists in the local mainline implementation, but it had not been included in
the PQR branch/clean clone.

Fix applied in the same route-owned worktree:

- added `opentad/models/backbones/time_aligned_rasterizer.py` from the existing
  mainline local implementation;
- kept `vit_adapter.py` as a hard import because this is a real Adapter TARA
  dependency, not an optional diagnostic-only hook;
- changed no PQR scoring math, quality-head target/fusion code, CADF selector,
  BH-SDC, sampler, evaluator, post-processing, config inheritance, or launcher
  behavior.

This is a clean-clone dependency completeness fix only. It does not create a
model-result claim, selector claim, smoke claim, or mAP claim.

## Pseudo-Boundary Dependency Fix - 2026-06-30

After the rasterizer dependency fix, the remote clean clone at branch HEAD
`8cb6b64f83c1b9e86d887978accf1cabfe6f7d34` passed Linux PRECHECK but failed the
2-iteration runtime smoke before the training loop with:

```text
ModuleNotFoundError: No module named 'opentad.datasets.transforms.pseudo_boundary'
```

The error came from the hard import in
`opentad/datasets/transforms/end_to_end.py`. The root cause was another
clean-clone dependency completeness gap: the real pseudo-boundary transform
helper existed in repository history and matched the current tests/API, but was
missing from this PQR branch.

Fix applied in the same route-owned worktree:

- added `opentad/datasets/transforms/pseudo_boundary.py`;
- kept `end_to_end.py` as a hard import instead of hiding the issue behind an
  optional import;
- added a PQR validator/test dependency guard for the helper file and required
  API symbols;
- updated the three PQR config metadata fields to state that pseudo_boundary is
  locally restored and remote PRECHECK plus 2-iter smoke evidence is still
  required.

This fix does not alter CADF selector logic, BH-SDC, evaluator,
post-processing, sampler, PQR scoring math, quality-head math, detector head
logic, loss/assignment, training launcher behavior, runtime gate behavior, or
mAP claims.

Evidence report:
`research-wiki/experiments/c3_pqr_rankcal_v1_pseudo_boundary_dependency_fix_20260630.md`.

## Local Verification

Commands run in
`E:\DeskTop\TAD\temrefuse-tad\OpenTAD_C3PQRRankCal_Worktree_20260629`:

```powershell
python -m pytest tests/test_c3_pqr_rankcal_v1_config.py tests/test_c3_pqr_rankcal_v1_quality_head.py -q
```

Result after the PRECHECK_ONLY fixture fix: `13 passed, 3 skipped, 1 warning`.
The 3 skipped tests are torch-backed quality-head behavior tests. Local Windows
torch import fails with
`[WinError 1114] ... c10.dll`, so the tests record that environment blocker
instead of pretending the torch path passed locally.

```powershell
python -m py_compile opentad\models\dense_heads\anchor_free_head.py tools\validate_c3_pqr_rankcal_v1_config.py tests\test_c3_pqr_rankcal_v1_config.py tests\test_c3_pqr_rankcal_v1_quality_head.py configs\adatad\thumos\c3_indirect_original_adatad_32px_a_pqr_rankcal_v1_precheck.py configs\adatad\thumos\c3_indirect_original_adatad_32px_a_pqr_rankcal_v1_shortdiag.py configs\adatad\thumos\c3_indirect_original_adatad_32px_a_exact_uniform_backend_control_pqr_rankcal_v1_shortdiag.py
```

Result: pass.

```powershell
python tools\validate_c3_pqr_rankcal_v1_config.py configs\adatad\thumos\c3_indirect_original_adatad_32px_a_pqr_rankcal_v1_precheck.py
python tools\validate_c3_pqr_rankcal_v1_config.py configs\adatad\thumos\c3_indirect_original_adatad_32px_a_pqr_rankcal_v1_shortdiag.py
python tools\validate_c3_pqr_rankcal_v1_config.py configs\adatad\thumos\c3_indirect_original_adatad_32px_a_exact_uniform_backend_control_pqr_rankcal_v1_shortdiag.py
```

Result: all three print `PASS_C3_PQR_RANKCAL_V1_CONFIG`.

```powershell
git diff --check
```

Result: pass.

Additional untracked-file whitespace check:

```powershell
$files = git ls-files --others --exclude-standard
foreach ($f in $files) { git diff --check --no-index -- $empty $f }
```

Result: pass, with only LF-to-CRLF warnings from Git on Windows.

## Remaining Risk

- Local torch-backed behavioral tests could not execute because of the Windows
  torch DLL failure. They should be rerun in a working torch environment before
  remote PRECHECK_ONLY.
- This is a ranking-calibration diagnostic, not an mAP-gain claim.
- Remote sync, Slurm, training, Pro, and Gemini were not started.

## 2026-06-30 Status Update

Remote/Linux gate status has advanced beyond the original local-only state:

- PQR QC V2 remote PRECHECK R3 passed at commit
  `f827c50538c6cda68756c3d37cd11a53c5b3e314` in fresh clone
  `/data/home/sczc063/run/yuzibo/OpenTAD_C3PQR_QCV2_Precheck_f827c50_20260630_20260630_141515_+0800`.
  The R3 focused Linux pytest result was `45 passed in 27.89s`, with validator,
  py_compile, and `git diff --check` also passing. No Slurm, GPU, training, or
  evaluation was run for R3.
- Matched PQR backend-control diagnostic child `1118197.459 pqr_ctrl_g0_r5`
  completed with `COMPLETED 0:0`. It produced interim Average-mAP `15.15%`,
  final diagnostic Average-mAP `24.95%`, `Training Over`, `result_detection.json`,
  and diagnostic JSON files. This is a short diagnostic result only and must not
  be used as final route-quality evidence.
- Current GPU ownership blocks immediate follow-up: GPU0 is occupied by
  `1118197.467 bvr_twb_g0_r4` and remains reserved for divergent innovation;
  GPU1 is occupied by CADF formal child `1118197.433 cadf_formal_g1`, which was
  still running with finite loss and no validation result at the latest
  read-only check.

Current mechanism interpretation:

- PVR-QC, PQR V1, and QC V2 are separate. PVR-QC diagnoses proposal/ranking/cap
  overload; PQR V1 is a max-IoU quality/ranking calibration control; QC V2 adds
  sparse/irregular geometry-aware quality target and proposal diagnostics.
- Existing evidence more strongly supports the failure mode "high-IoU proposals
  can exist but are poorly ranked/calibrated and buried under cap overload" than
  "the model produces no usable proposals at all". This is not proof that the
  proposal generator is healthy; it only identifies the most visible bottleneck.
- The next high-information step is QC V2 bounded proposal-dump diagnostics on
  GPU1 only after CADF releases GPU1. The purpose is to verify `quality_score`,
  `cls_score`, selected/physical segments, gap, coverage, and endpoint-support
  dumps for score-vs-quality-vs-IoU analysis. It is not a final mAP route
  judgment.
