# PC-OT-MRAS R16/R17/R18/R19/R20/R21/R22/R23/R24/R25/R26/R27/R28/R29/R30/R31 Clean Implementation Manifest

Timestamp: 2026-06-20T23:48:35+08:00

This repository was created from the manually downloaded clean OpenTAD source and initialized as a new git repository. The clean baseline is commit `f19492b` (`Import clean OpenTAD baseline`).

Current branch: `codex/pcotmras-r16-r18-r20-clean`

Current commits:

```text
f19492b Import clean OpenTAD baseline
2301aaa50810d6fe7d8d88760bd2e789896e93d8 Implement clean PC-OT-MRAS R16 R18 R20
5ca29aba79ca152b9070905f97c93b29948258ac Add R20 value-only attribution control
73fc7ba Fix clean PC-OT-MRAS launch guards
709bf9783d0e12104e352e3b863ce6373c2b441d Update clean PC-OT-MRAS manifest after guard fixes
21f8dbabfc61158ac21c000066b8fbae24866148 Remove stale checkpoint audit dependency from clean PC-OT-MRAS
1a95068 Update clean PC-OT-MRAS purity manifest
96eaaeb Add R18 and R20 confirmation candidates
4a4c92a Record R18 R20 confirmation candidate manifest
b5b4448 Update R18 R20 review package manifest
f18c708b188464235f7732b71cfac205f47d4be3 Harden PC-OT-MRAS R18 R20 gates
4a0864b24ce44ac46e4ceb9fddba8c1f36ee8cea Add PC-OT-MRAS R19 soft hard consistency candidate
96f431c Record PC-OT-MRAS R19 clean manifest
a7fa6bc Add PC-OT-MRAS R21 tensor temporal grid path
719f928 Record PC-OT-MRAS R21 clean manifest
9cc3c82 Add PC-OT-MRAS R22 dynamic budget controller
85d64d5 Record PC-OT-MRAS R22 clean manifest
a09a247 Fix PC-OT-MRAS bridge neck registration
6753ba8 Record PC-OT-MRAS R16 bridge registry fix
87d3404 Add clean PC-OT-MRAS R17 formal launcher
6262129 Record PC-OT-MRAS R17 clean launcher manifest
05e7388 Add PC-OT-MRAS R23 dynamic budget hard export
678d907 Record PC-OT-MRAS R23 clean manifest
3f466ee Add PC-OT-MRAS R24 temporal metadata contract
b86012e Record PC-OT-MRAS R24 clean manifest
3ffec40 Fix PC-OT-MRAS R24 hard row validation
c3c1ab2 Record PC-OT-MRAS R24 validation fix
dc972e1 Record PC-OT-MRAS R24 Gemini fix review
da59fe3 Add PC-OT-MRAS R25 pipeline validation
15bca1c Record PC-OT-MRAS R25 clean manifest
c01bae7 Fix PC-OT-MRAS R25 dynamic plan gate
1c1bf65 Record PC-OT-MRAS R25 gate fix
70e7ca3 Fix PC-OT-MRAS Slurm master port guard
42b9ab7c94389aa3532481c520138017f442b22c Fix PC-OT-MRAS native area head registration
d546ca877c644eb4a209a004c592f24aa3ff3d86 Record PC-OT-MRAS native head registry fix
8effdb01da43e7641f172806a3d326552a73eca4 Bind PC-OT-MRAS native head files in launch manifests
7f3fb178dce78f2f9f61522ddde5c2a75a424d48 Record PC-OT-MRAS native head manifest binding
01d41a6b1976dd2ca6d0e538c1fdead47df67180 Add PC-OT-MRAS R26 dynamic budget frontier audit
834fb49cf4b1466a3c21ff1dc73337ba823772f7 Record PC-OT-MRAS R26 clean manifest
c63d3e0538b26c7504a0cd44ba2a6c389d15ec68 add r27 synthetic task utility audit
a1cb470b8e23e2dc78f3d966d2cfb946e7beff1b Add PC-OT-MRAS R28 tubelet token redundancy audit
9744e6fb0f45a1db2702d45e034f95e98b29cb11 Fix PC-OT-MRAS AMP mask sentinels
d0478cc Record PC-OT-MRAS R28 clean manifest
3493337 Record PC-OT-MRAS AMP sentinel fix
fa216a9 Add PC-OT-MRAS R29 tubelet packed profile audit
36687dc Add PC-OT-MRAS R30 tubelet packed runtime proof
a5e2507847f0ea1ef32b6903323ed087751e2934 Add PC-OT-MRAS R31 packed forward opt-in
b83897048434c51688cb10dd012c92195cab330c Refine PC-OT-MRAS R31 adapter packed forward
7d590c305519ccef4c594ee8f16c5c8942ff8990 Fix PC-OT-MRAS pair softmax AMP dtype
990aef92affc2a30e17273a847c68e519b79ab30 Fix PC-OT-MRAS pair entropy AMP finite
```

## Objective

Implement only the PC-OT-MRAS continuous route stages on a clean OpenTAD baseline:

- R16 bounded GPU smoke candidate.
- R17 reader-only formal training candidate.
- R18 train-only reader auxiliary diagnostic candidate.
- R19 train-only soft/hard consistency candidate.
- R20 train-only value-of-information distillation candidate.
- R21 tensor-native temporal coordinate path.
- R22 value-to-dynamic-budget controller candidate.
- R23 dynamic-budget hard-position export candidate.
- R24 dynamic-budget temporal metadata and detector geometry contract candidate.
- R25 dynamic-budget pipeline validation candidate.
- R26 dynamic-budget frontier audit candidate.
- R27 synthetic task-utility audit candidate.
- R28 tubelet/token redundancy auxiliary audit candidate.
- R29 profiler-only temporal-tubelet packed-profile audit candidate.
- R30 local synthetic true packed temporal-tubelet runtime proof candidate.
- R31 local-only packed temporal-tubelet production-forward opt-in candidate.

The implementation intentionally does not copy unrelated BATA/UGIT/MFCSD/DBAC/ITMI experiment families from the dirty working tree.

## R26 Dynamic-Budget Frontier Audit

Commit `01d41a6b1976dd2ca6d0e538c1fdead47df67180` adds a local-only R26
frontier audit layer:

- `tools/bata/audit_pc_ot_mras_dynamic_budget_frontier.py` reuses the R25
  pipeline validator before computing budget distribution, average selected
  count, savings versus a fixed-budget reference, coverage-cap violations,
  short-valid-len clipping, budget-score monotonicity, and optional difficulty
  ordering. It returns `PC_OT_MRAS_DYNAMIC_BUDGET_FRONTIER_AUDIT_READY` only
  when the hard protocol passes and the plan is budget-sensitive.
- `configs/adatad/thumos/ctf_bdi_pc_ot_mras_r26_dynamic_budget_frontier_audit.py`
  is launch-blocked and local-synthetic-only. It explicitly keeps detector
  training, `tools/train.py`, `tools/test.py`, detector mAP, remote sync,
  Slurm/GPU, real data/checkpoints, runtime/FLOPs, deployment, dynamic-budget
  quality validation, scanner-quality validation, metric claims, and paper
  claims disabled.
- `tests/test_pc_ot_mras_dynamic_budget_frontier_audit.py` covers budget
  differentiation, JSON roundtrip, forbidden payload rejection, uniform-budget
  no-go behavior, and short valid-length clipping accounting.
- `tests/test_pc_ot_mras_r26_dynamic_budget_frontier_config.py` covers config
  parse and training/test entrypoint fail-closed behavior.

R26 is a calibration/frontier audit only. It does not produce detector mAP,
runtime/FLOPs evidence, scanner-quality evidence, deployment proof, or a paper
claim.

R26 verification in the local Windows `torch_1` environment:

```text
py_compile changed R26 files: pass
R26 focused pytest: 6 passed in 6.98s
R22-R26 dynamic-budget regression pytest: 65 passed in 14.17s
full PC-OT-MRAS plus train-engine pytest: 269 passed in 62.65s
git diff --check: pass
```

## R27 Synthetic Task-Utility Audit

Commit `c63d3e0538b26c7504a0cd44ba2a6c389d15ec68` adds a local-only
synthetic task-utility audit above R22-R26:

- `tools/bata/audit_pc_ot_mras_synthetic_task_utility.py` reuses R26/R25
  protocol validation and then checks synthetic start/end boundary support,
  interior peak recall, background selected share, same-budget exact-uniform
  control, synthetic oracle top-k utility ratio, and difficulty-budget
  ordering.
- `configs/adatad/thumos/ctf_bdi_pc_ot_mras_r27_synthetic_task_utility_audit.py`
  is launch-blocked and local-synthetic-only. It keeps detector training,
  `tools/train.py`, `tools/test.py`, detector mAP, remote sync, Slurm/GPU,
  real data/checkpoints, runtime/FLOPs, dynamic-budget quality validation,
  scanner-quality validation, metric claims, and paper claims disabled.
- `tests/test_pc_ot_mras_synthetic_task_utility_audit.py` and
  `tests/test_pc_ot_mras_r27_synthetic_task_utility_config.py` cover task
  utility pass/fail behavior, JSON roundtrip, leakage-key rejection, and
  launch-gate behavior.

R27 verification in the local Windows `torch_1` environment:

```text
py_compile changed R27 files: pass
R27 focused pytest: 7 passed
R22-R27 dynamic-budget regression pytest: 72 passed
full PC-OT-MRAS plus train-engine pytest: 276 passed
```

## R28 Tubelet/Token Redundancy Auxiliary Audit

Commit `a1cb470b8e23e2dc78f3d966d2cfb946e7beff1b` adds the first local
space/token redundancy bridge for the PC-OT-MRAS route:

- `opentad/models/backbones/vit_adapter.py` now includes
  `TubeletTokenRedundancyAux`, a default-off, local-only auxiliary auditor
  attached immediately after VideoMAE patch embedding and before positional
  encoding. It scores temporal tubelet groups from spatial-token energy
  statistics, records a proposed tubelet keep mask, forbids arbitrary spatial
  patch crop, and returns the dense token sequence unchanged.
- `configs/adatad/thumos/ctf_bdi_pc_ot_mras_r28_tubelet_token_redundancy_aux.py`
  inherits R27, configures the auxiliary as disabled by default, and keeps
  detector training, `tools/train.py`, `tools/test.py`, detector mAP, remote
  sync, Slurm/GPU, real data/checkpoints, runtime/FLOPs, spatial redundancy
  claims, metric claims, and paper claims disabled.
- `tools/bata/audit_pc_ot_mras_tubelet_token_redundancy.py` runs a synthetic
  local audit that verifies dense shape/value preservation, temporal
  tubelet-group routing, no spatial crop, no GT/teacher/cache/raw-prediction
  inputs, and no runtime/FLOPs or metric claim.
- `tests/test_pc_ot_mras_tubelet_token_redundancy_aux.py` and
  `tests/test_pc_ot_mras_r28_tubelet_token_redundancy_config.py` cover dense
  output parity, deterministic keep-mask contract, spatial-crop rejection,
  JSON audit roundtrip, no-claim boundaries, config parse, and training/test
  fail-closed behavior.

R28 is not true packed compute yet. It is the local contract layer that makes
the tubelet/token spatial redundancy route reviewable before any backbone
runtime, remote sync, Slurm/GPU, mAP, runtime/FLOPs, deployment, or paper
claim.

R28 verification in the local Windows `torch_1` environment:

```text
py_compile changed R28 files: pass
R28 focused pytest: 6 passed in 4.82s
R28 synthetic audit smoke: PC_OT_MRAS_TUBELET_TOKEN_REDUNDANCY_AUDIT_READY
full PC-OT-MRAS plus train-engine pytest: 282 passed in 68.18s
git diff --check: pass
```

## R29 Tubelet Packed-Profile Audit

R29 adds the first profiler-only proof for the R28 tubelet/token route. It does
not change `VisionTransformerAdapter.forward()`, does not run packed
attention/MLP, and does not touch detector training or evaluation paths.

- `tools/bata/audit_pc_ot_mras_tubelet_packed_profile.py` reuses the R28
  synthetic tubelet redundancy summary, expands the temporal tubelet keep mask
  to dense token space, packs selected temporal-tubelet groups into a
  rectangular token tensor, scatters selected values back into dense shape, and
  computes hypothetical attention-token-pair and linear-token accounting.
- `configs/adatad/thumos/ctf_bdi_pc_ot_mras_r29_tubelet_packed_profile_audit.py`
  inherits R28 and remains launch-blocked. It keeps detector training,
  `tools/train.py`, `tools/test.py`, detector mAP, remote sync, Slurm/GPU,
  real data/checkpoints, measured runtime/FLOPs, spatial-redundancy claims,
  metric claims, and paper claims disabled.
- `tests/test_pc_ot_mras_tubelet_packed_profile_audit.py` covers pack/scatter
  bookkeeping, selected-value preservation, no-go behavior when identity has
  no strict saving, JSON roundtrip, and no-claim boundaries.
- `tests/test_pc_ot_mras_r29_tubelet_packed_profile_config.py` covers config
  parse and training/test entrypoint fail-closed behavior.

R29 is not true packed compute. It proves only that the temporal-tubelet mask
can support a future packed-compute implementation and that the accounting
frontier is non-trivial under synthetic inputs.

R29 verification in the local Windows `torch_1` environment:

```text
py_compile changed R29 files: pass
R29 focused pytest: 5 passed in 9.78s
R28/R29 focused regression pytest: 11 passed in 9.59s
R29 synthetic audit smoke: PC_OT_MRAS_TUBELET_PACKED_PROFILE_AUDIT_READY
full PC-OT-MRAS plus train-engine pytest: 289 passed in 68.79s
git diff --check: pass
```

## R30 Tubelet Packed Runtime Proof

R30 adds the first local true packed-compute proof for the R28/R29 tubelet
route. It still does not change `VisionTransformerAdapter.forward()`, does not
train or evaluate a detector, and does not touch remote execution paths.

- `tools/bata/audit_pc_ot_mras_tubelet_packed_runtime.py` reuses R29's
  temporal tubelet mask expansion and rectangular pack/scatter bookkeeping,
  then executes real `vit_adapter.Block(use_adapter=False)` attention and MLP
  modules on the packed token tensor. It records packed-path attention/MLP
  forward-hook counts, compares packed batch output against a selected-only
  per-sample reference, scatters packed outputs back to dense token shape, and
  records local synthetic wall-clock timings.
- `configs/adatad/thumos/ctf_bdi_pc_ot_mras_r30_tubelet_packed_runtime_proof.py`
  inherits R29 and remains launch-blocked. It keeps detector training,
  `tools/train.py`, `tools/test.py`, detector mAP, remote sync, precheck,
  Slurm/GPU, real data/checkpoints, deployment runtime/FLOPs claims,
  spatial-redundancy claims, metric claims, and paper claims disabled.
- `tests/test_pc_ot_mras_tubelet_packed_runtime_audit.py` covers true packed
  attention/MLP execution, hook counts, scatter-back shape, selected-only
  reference agreement, JSON roundtrip, no-go behavior for no-saving identity
  input, and invalid head divisibility.
- `tests/test_pc_ot_mras_r30_tubelet_packed_runtime_config.py` covers config
  parse and training/test entrypoint fail-closed behavior.

Default R30 synthetic smoke reports:

```text
decision: PC_OT_MRAS_TUBELET_PACKED_RUNTIME_AUDIT_READY
packed_block_impl: vit_adapter.Block(use_adapter=False)
dense_token_count: 48
packed_token_count: 24
attention_pair_ratio: 0.25
linear_token_ratio: 0.5
packed_attention_forward_count: 8
packed_mlp_forward_count: 8
selected_outputs_match_reference: true
selected_reference_max_abs_error: 0.0
packed_over_dense_runtime_ratio_observed: 0.806544477497649
runtime_measurement_scope: local_synthetic_tiny_block_only
runtime_flops_claim_allowed: false
detector_map_allowed: false
```

R30 is not production packed ViT execution. It proves only that the selected
temporal-tubelet route can run real attention/MLP on packed tokens and scatter
the result back under a synthetic local contract. Any production forward change,
real-data precheck, detector mAP, deployment runtime/FLOPs claim, spatial
redundancy claim, or paper claim still requires a separate review and launch
gate.

R30 verification in the local Windows `torch_1` environment:

```text
py_compile changed R28/R29/R30 files: pass
R28/R29/R30 focused pytest: 16 passed in 9.21s
R30 synthetic runtime smoke: PC_OT_MRAS_TUBELET_PACKED_RUNTIME_AUDIT_READY
full PC-OT-MRAS pytest: 291 passed in 73.16s
git diff --check: pass with LF/CRLF warning only
```

## R31 Packed Tubelet Forward Opt-In

R31 connects the R30 packed temporal-tubelet runtime proof to the production
`VisionTransformerAdapter.forward()` path as an explicit local-only opt-in. It
does not enable detector training, remote sync, precheck, Slurm/GPU, real
datasets/checkpoints, detector mAP, measured deploy runtime/FLOPs, deployment,
spatial-redundancy, metric, or paper claims.

- `opentad/models/backbones/vit_adapter.py` adds
  `PackedTubeletRuntimeRoute`. The route builds a temporal-tubelet keep mask
  after patch embedding and positional encoding, then passes the mask into each
  transformer `Block`.
- `Block.forward()` now has an optional packed subpath for attention/MLP only:
  selected temporal-tubelet tokens are packed, processed, and scattered back to
  dense shape inside the block. Adapter convolution then receives dense tokens,
  preserving the existing Adapter temporal-grid contract.
- The route keeps unselected tokens on an identity bypass before Adapter. It is
  disabled by default and still rejects training mode unless explicitly allowed
  by a future gate.
- `configs/adatad/thumos/ctf_bdi_pc_ot_mras_r31_packed_forward_optin_local.py`
  inherits R30 and remains launch-blocked. The config keeps the packed route
  default-off and records only local synthetic forward checks as allowed.
- `tests/test_pc_ot_mras_packed_tubelet_forward_route.py` covers direct route
  execution, dense scatter-back shape, finite selected outputs, Adapter dense
  contract preservation, strict adapter-free fail-closed mode, training-mode
  fail-closed behavior, full `VisionTransformerAdapter.forward()` opt-in
  behavior, and default-path preservation.
- `tests/test_pc_ot_mras_r31_packed_forward_config.py` covers config parse,
  default-off route config, and training/test entrypoint fail-closed behavior.

R31 verification in the local Windows `torch_1` environment:

```text
py_compile changed R31 files:
  pass

R31 focused pytest:
  5 passed in 5.18s

R28/R29/R30/R31 focused regression:
  21 passed in 13.27s

full PC-OT-MRAS pytest plus train-engine max-train-iter selection:
  299 passed in 69.59s

git diff --check:
  pass with LF/CRLF warning only
```

## R16A Pair-Distribution AMP Dtype Repair

The second R16A bounded GPU smoke job `1106714` reached AMP training on N16R4
and failed during launch sanity, before any detector mAP or training conclusion:

```text
RuntimeError: expected scalar type Float but found Half
opentad/models/selectors/pc_ot_mras_reader.py:327
flat_prob[has_pair] = _masked_softmax(flat_logits[has_pair], flat_mask[has_pair], dim=-1)
```

The failure was in `_pair_distribution()`: AMP/autocast can make the local
masked-softmax result and the preallocated flattened probability buffer use
different floating dtypes. Commit `7d590c3` casts the softmax result to
`flat_prob.dtype` before assignment. This is a launch-sanity dtype-contract
repair only. It does not change pair validity, role allocation semantics, input
sampling, dynamic-budget policy, Adapter/backbone routing, detector head logic,
loss/assignment, post-processing, GT/teacher/cache boundaries, or any metric
claim.

Verification in the local Windows `torch_1` environment:

```text
py_compile:
  pass

focused reader pair-distribution and half-head tests:
  11 passed in 6.34s

full PC-OT-MRAS pytest plus train-engine max-train-iter selection:
  300 passed in 70.36s

git diff --check:
  pass with LF/CRLF warning only
```

## Source Provenance

The implementation was assembled by white-list overlays:

1. R16 execution package:
   `logs/ctf_bdi_pc_ot_mras_r16a_gpu_smoke_execution_package_20260619_0911/runtime_repo`
2. R18 reviewed extract:
   `logs/_ctf_bdi_pc_ot_mras_r18_aux_diag_pro_review_20260619_1445_verify_extract/OpenTAD_BATA_Clean`
3. R20 focused files from the current PC-OT-MRAS source tree:
   - `configs/adatad/thumos/ctf_bdi_pc_ot_mras_r20_value_distill_candidate.py`
   - `opentad/models/detectors/actionformer.py`
   - `opentad/models/selectors/pc_ot_mras_reader.py`
   - `opentad/models/losses/pc_ot_mras_value_distillation_losses.py`
   - `tests/test_pc_ot_mras_value_distillation_losses.py`
   - `tests/test_pc_ot_mras_actionformer_forward_selector.py`
   - `tests/test_pc_ot_mras_reader_shapes.py`

## Resolved Config Contract

Parsed with `mmengine.Config.fromfile`:

| Config | Reader | Aux loss | Soft/hard loss | Value loss | Raw prediction cache |
| --- | --- | --- | --- | --- | --- |
| `ctf_bdi_pc_ot_mras_r16_gpu_smoke_candidate.py` | present | absent | absent | absent | disabled |
| `ctf_bdi_pc_ot_mras_r17_formal_train_candidate.py` | present | absent | absent | absent | disabled |
| `ctf_bdi_pc_ot_mras_r18_aux_diag_candidate.py` | present | present | absent | absent | disabled |
| `ctf_bdi_pc_ot_mras_r18_aux_formal_train_candidate.py` | present | present | absent | absent | disabled |
| `ctf_bdi_pc_ot_mras_r19_soft_hard_consistency_candidate.py` | present | present | present | absent | disabled |
| `ctf_bdi_pc_ot_mras_r20_value_distill_candidate.py` | present, `enable_value_heads=True` | present | absent | present | disabled |

R20 is therefore an `R18 semantic aux + R20 value` combo mainline, matching the saved R20A design. It is not a value-only attribution control.

## Attribution Control

A separate value-only attribution control is provided at:

```text
configs/adatad/thumos/ctf_bdi_pc_ot_mras_r20_value_only_control.py
```

It inherits R17 directly, clears the R17 formal-train gate, enables R20 value heads and `pc_ot_mras_reader_value_loss`, and does not define or inherit `pc_ot_mras_reader_aux_loss`. This config is launch-blocked and local-only until a separate review gate approves execution.

## Pro v2 Blocker Fix

GPT-5.5 Pro review package v2 returned `FAIL_FIX_REQUIRED_BEFORE_GEMINI`.

Accepted blocking fixes in commit `73fc7ba`:

- `scripts/run_ctf_bdi_pc_ot_mras_r16a_gpu_smoke_n16r4.sbatch` now defaults to the clean repo path `OpenTAD_PCOTMRAS_R16_R18_R20_Clean_20260619_1730`, supports explicit `OPENTAD_PCOTMRAS_CLEAN_ROOT`, exports `OPENTAD_PCOTMRAS_CLEAN_ROOT`, and rejects any path containing `OpenTAD_BATA_Clean`.
- `opentad/utils/training_guard.py` now applies `allowed_entrypoints`, `allow_tools_train`, `allow_tools_test`, and `allow_detector_map` checks to all explicit gates, not only smoke gates.
- Regression tests cover the clean launcher provenance guard and R17 formal-train `tools/test.py` fail-closed behavior.

## Verification

Commands run from this repository:

```powershell
$files = git ls-files -m -o --exclude-standard | Where-Object { $_ -like '*.py' }
C:\Users\skywalker\.conda\envs\torch_1\python.exe -m py_compile @files
git diff --check
$tests = @(Get-ChildItem -LiteralPath tests -Filter 'test_pc_ot_mras_*.py' -File | ForEach-Object { $_.FullName }) + @((Resolve-Path 'tests/test_train_engine_max_train_iters.py').Path)
C:\Users\skywalker\.conda\envs\torch_1\python.exe -m pytest -q @tests
```

Results:

- `py_compile_files=65`
- `git diff --check`: pass
- `pytest`: `180 passed in 47.23s`

Additional verification after R20 value-only control and Pro v2 blocker fixes:

- config parse: R16 reader-only, R17 reader-only, R18 reader+aux, R20 combo reader+aux+value, R20 value-only reader+value without aux; raw prediction cache disabled for all checked configs.
- focused Pro-blocker tests: `24 passed in 2.00s`
- `py_compile`: pass for changed Python files
- `git diff --check`: pass
- `bash -n scripts/run_ctf_bdi_pc_ot_mras_r16a_gpu_smoke_n16r4.sbatch`: exit code `0` with local WSL warning noise
- full PC-OT-MRAS relevant pytest set: `183 passed in 39.57s`

## Local R20 Fix

During clean-repo verification, the R20 value-target loss rejected a non-contiguous `valid_mask` through the earlier sum-mismatch branch. The validation order was corrected so non-contiguous masks report `value target valid_mask must be a contiguous valid prefix` before the redundant sum check. This is a local R20 correctness fix in the clean implementation.

## Purity Recheck and Shared-Entrypoint Fix

The clean baseline commit `f19492b` was rechecked against the manually extracted `OpenTAD-main` tree:

```text
git_file_count=408
manual_file_count=408
only_in_git=0
only_in_manual=0
hash_mismatch=0
```

This confirms that the baseline commit is the manually downloaded clean OpenTAD source, not a filtered copy of the dirty historical experiment tree.

During the same recheck, `opentad/utils/__init__.py` was found to import a non-existent `checkpoint_key_audit.py`, and `tools/train.py` still contained a `--load_from` / checkpoint-key-audit path from the historical dirty tree. This was not needed by R16/R18/R20 and would break `import opentad.utils` before training. Commit `21f8dbabfc61158ac21c000066b8fbae24866148` removes the stale dependency and the unrelated `--load_from` branch instead of copying the old ITMI/BATA checkpoint-audit file into the clean route.

Post-fix verification:

```text
opentad.utils import: PASS, save_checkpoint present, stale validate_incompatible_checkpoint_keys absent
config parse: R16/R17 reader-only, R18 reader+aux, R20 combo reader+aux+value, R20 value-only reader+value without aux
raw prediction cache: disabled for all checked R16/R17/R18/R20/R20-control configs
git diff --check: pass, with LF/CRLF warnings only
py_compile: pass for tracked Python files
pytest: 184 passed in 40.82s
ignored cache cleanup: 72 cache directories removed inside this clean repo
```

Direct `tools.train/tools.test` import in the local Windows `torch_1` environment still reaches the upstream OpenTAD optional dependency issue `ModuleNotFoundError: No module named 'mmaction.registry'` after the stale `opentad.utils` error is removed. This is an environment/dependency boundary of the local Windows env, not evidence that the PC-OT-MRAS stale dependency remains.

## R18/R20 Confirmation Candidates

Commit `96eaaeb` adds the next confirmation candidate layer requested after R16
restart:

- `configs/adatad/thumos/ctf_bdi_pc_ot_mras_r18_aux_formal_train_candidate.py`
  defines an aux-on R18 formal-training candidate. It inherits the reviewed R18
  aux diagnostic config, keeps train-only reader auxiliary losses enabled,
  permits only `tools/train.py`, rejects direct `tools/test.py`, and keeps
  metric/paper/runtime/deploy claims disabled.
- `scripts/run_ctf_bdi_pc_ot_mras_r18_aux_formal_train_n16r4.sbatch` is a
  fail-closed launcher. It defaults to `PRECHECK_ONLY=1`; formal training
  requires `ALLOW_R18_AUX_FORMAL_TRAIN=1` plus an explicit gate JSON/SHA bound
  to the active manifest. It rejects dirty `OpenTAD_BATA_Clean` paths,
  checkpoint/load/resume shortcuts, raw-prediction caches, and arbitrary
  cfg-options.
- `scripts/run_ctf_bdi_pc_ot_mras_r20_value_precheck_n16r4.sbatch` is
  precheck-only for R20 main and value-only control. It runs static/config/value
  focused checks only and intentionally never calls `tools/train.py` or
  `tools/test.py`.

Verification for this layer:

```text
py_compile changed candidate files: pass
R18/R20 launcher/config tests: 8 passed, 1 warning
git diff --check: pass
bash -n R18/R20 sbatch launchers: exit code 0, with local WSL warning noise
torch_1 R20 value/action tests: 31 passed
torch_1 R17/R18/guard tests: 16 passed
torch_1 all PC-OT-MRAS tests plus train-engine max-iter test: 189 passed in 51.75s
```

Known local environment boundary:

```text
Default Python value/action tests skipped because user-site torch is unavailable.
Default Python guard bundle has one expected import-smoke failure caused by
user-site torch c10.dll loading failure. The torch_1 environment passes the
relevant checks above.
```

Review package prepared but not submitted:

```text
logs/ctf_bdi_pc_ot_mras_r18_r20_confirmation_candidates_pro_review_20260619_2050.zip
SHA256: ad336011b0aa7dac8e0cceb20d7f027561456c2f88b4bc8763cc9f26f8c2ee7e
entries: 32
bad_backslash_entries: 0
sha_missing: 0
sha_bad: 0
```

Boundary: this layer is ready for GPT-5.5 Pro read-only review only. It does
not authorize remote sync, remote PRECHECK execution, Slurm/GPU execution,
`tools/train.py`, `tools/test.py`, detector mAP, dataset/checkpoint access,
runtime/FLOPs, deployment, metric claims, or paper claims.

## R18/R20 Complete-Package Pro Fix Layer

The complete R18/R20 package follow-up returned `FAIL_FIX_BEFORE_GEMINI`. The
package was visible and complete, but Pro rejected Gemini for this package
because the direct entrypoint and launch-context boundary was still too soft.

Accepted blocker fixes in this layer:

- `tools/train.py` and `tools/test.py` now call
  `assert_safe_cfg_options_for_gated_config()` before merging `--cfg-options`,
  so PC-OT-MRAS gate, workflow, checkpoint, raw-prediction, metric, and claim
  fields cannot be changed through direct CLI overrides.
- `opentad/utils/training_guard.py` now supports optional entrypoint gate
  context validation: gate JSON path, gate SHA256, active manifest SHA256,
  resolved config SHA256, allowed decisions, and forbidden true keys.
- `ctf_bdi_pc_ot_mras_r18_aux_formal_train_candidate.py` keeps R18 aux-on
  training as a confirmation candidate but disables train-time detector mAP for
  this package and requires launcher-provided entrypoint gate context before
  `tools/train.py` is allowed.
- R18/R20 N16R4 launchers now check the expected clean branch, reject tracked
  dirty files, write resolved config dumps, include recursively discovered
  `_base_` config dependencies in the active manifest, and bind resolved config
  SHA256 into the R18 execution gate and entrypoint environment.

Verification:

```text
py_compile affected files: pass
bash -n R18/R20 sbatch launchers: pass, with local WSL warning noise
focused tests: 17 passed in 5.75s
full PC-OT-MRAS tests plus train-engine max-iter: 190 passed in 48.85s
git diff --check: pass, with LF/CRLF warnings only
```

Boundary: this is a local hardening fix after Pro rejection. It does not
authorize Gemini, remote sync, remote PRECHECK execution, Slurm/GPU execution,
`tools/train.py`, `tools/test.py`, detector mAP, dataset/checkpoint access,
runtime/FLOPs, deployment, metric claims, or paper claims. A fresh complete Pro
package from the new HEAD is required next.

## R19 Soft/Hard Consistency Candidate

Commit `4a0864b24ce44ac46e4ceb9fddba8c1f36ee8cea` adds the R19 local candidate
layer requested by the long-term PC-OT-MRAS plan:

- `opentad/models/losses/pc_ot_mras_soft_hard_consistency_losses.py` defines
  train-only consistency losses between the differentiable reader allocation
  and detached hard anchors derived from that allocation. It does not read
  hard-export JSON, GT, teacher outputs, raw predictions, checkpoints, or
  caches.
- `ActionFormer.forward_train()` can optionally merge the R19 losses after
  reader-output injection. `forward_test()` does not call the R19 loss path.
- `configs/adatad/thumos/ctf_bdi_pc_ot_mras_r19_soft_hard_consistency_candidate.py`
  inherits the R18 aux formal candidate, enables the soft/hard loss, and keeps
  the config launch-blocked/local-synthetic-only.
- Focused tests cover loss gradients, bad reader contracts, config parsing,
  guard denial for `tools/train.py`/`tools/test.py`, and the ActionFormer
  train-only/test-time boundary.

Verification:

```text
py_compile changed R19 files: pass
focused R19 pytest: 22 passed in 9.81s
full PC-OT-MRAS pytest plus train-engine max-iter: 197 passed in 44.21s
git diff --check: pass, with LF/CRLF warnings only
```

Boundary: R19 is a local implementation candidate only. It does not authorize
Pro/Gemini completion, remote sync, remote PRECHECK execution, Slurm/GPU,
`tools/train.py`, `tools/test.py`, detector mAP, dataset/checkpoint access,
runtime/FLOPs, deployment, dynamic-budget/scanner-quality validation, metric
claims, or paper claims. The next gate is a complete GPT-5.5 Pro read-only
implementation review package for R19, then Gemini CLI only if Pro allows it.

## R21 Tensor-Native Temporal Coordinate Path

Commit `a7fa6bc46aa88890208bae5f25c39aade0c3815a` adds the R21 local
infrastructure layer for tensor-native PC-OT-MRAS temporal coordinates:

- `PCOTMRASDetectorBridge` now emits tensor temporal metadata in
  `pc_ot_mras_bridge`: `selected_dense_positions`, `dense_valid_len_tensor`,
  and `temporal_tensor_metadata_mode`. Legacy
  `irregular_selected_positions` / `irregular_dense_valid_len` fields are still
  written for compatibility.
- `temporal_grid_from_metas()` now prefers the bridge tensor payload when it is
  present, validates the bridge selected-mask prefix, and fail-closes if the
  tensor payload disagrees with legacy sidecar aliases.
- `build_temporal_grid()` and the fixed-width branch of
  `build_area_time_grid()` now avoid the in-place operations that broke
  autograd once temporal coordinates became tensor-derived.
- Tests cover tensor payload emission, legacy alias consistency, gradient flow
  from temporal grid centers back to tensor positions, detector-facing bridge
  metadata, and ActionFormer/P2 smoke metadata.

Verification:

```text
py_compile changed R21 files: pass
focused R21 pytest: 64 passed in 12.63s
targeted detector runtime smoke after autograd fix: 1 passed in 5.56s
full PC-OT-MRAS pytest plus train-engine max-iter: 199 passed in 44.69s
git diff --check: pass, with LF/CRLF warnings only
```

Known local environment boundary:

```text
Default Anaconda base Python cannot import user-site torch because c10.dll
fails to initialize. Verification used the working torch_1 conda environment
at C:\Users\skywalker\.conda\envs\torch_1\python.exe, torch 2.3.0+cu121.
```

Boundary: R21 is a local implementation candidate only. It does not authorize
Pro/Gemini completion, remote sync, remote PRECHECK execution, Slurm/GPU,
`tools/train.py`, `tools/test.py`, detector mAP, dataset/checkpoint access,
runtime/FLOPs, deployment, dynamic-budget/scanner-quality validation, metric
claims, or paper claims. Because it changes the temporal-coordinate protocol,
the next gate is GPT-5.5 Pro read-only implementation review, then Gemini CLI
read-only review only if Pro allows it.

## R22 Value-to-Dynamic-Budget Controller Candidate

Commit `9cc3c8291a80eeb5188beffcb31bd08d8e15e52a` adds the R22 local protocol
candidate that turns R20 reader value signals into a variable-budget dense
position plan:

- `opentad/models/selectors/pc_ot_mras_dynamic_budget_controller.py` defines
  `PCOTMRASDynamicBudgetController`, which consumes deploy-visible reader
  tensors only: `valid_mask`, `value_logits`, optional `risk_logits`,
  optional `redundancy_logits`, and optional `acquisition_matrix`.
- The controller rejects GT, oracle, teacher, cache, raw-prediction,
  checkpoint, result, train-only value-target, and legacy value-transport
  payloads before computing any budget plan.
- The output is a local protocol artifact with exact per-sample budgets,
  sorted unique dense positions, prefix `selected_mask`, coverage/value counts,
  and explicit `dynamic_budget_validation=False`, `metric_claim_allowed=False`,
  and `paper_claim_allowed=False` fields.
- `configs/adatad/thumos/ctf_bdi_pc_ot_mras_r22_dynamic_budget_control.py`
  inherits the R20 value-only control to keep value-head attribution isolated,
  adds a launch-blocked R22 gate, and keeps train/test, remote sync, Slurm/GPU,
  detector mAP, dataset/checkpoint, runtime/FLOPs, deployment, scanner-quality,
  dynamic-budget validation, metric, and paper claims disabled.
- `opentad/models/selectors/__init__.py` was added so local PC-OT-MRAS selector
  modules have an explicit registry import path.

Verification:

```text
py_compile changed R22 files: pass
focused R22/R20 pytest: 23 passed in 8.63s
full PC-OT-MRAS pytest plus train-engine max-iter: 205 passed in 47.07s
git diff --check: pass, with LF/CRLF warnings only
```

Known local environment boundary:

```text
`import opentad.models` in the local torch_1 environment is still blocked by
the repository's missing optional `mmaction.registry` dependency. This is an
existing full-package dependency issue; R22 verification used the same
lightweight PC-OT-MRAS module-loading path as the existing clean-repo tests.
```

Boundary: R22 is a local implementation candidate only. It does not authorize
Pro/Gemini completion, remote sync, remote PRECHECK execution, Slurm/GPU,
`tools/train.py`, `tools/test.py`, detector mAP, dataset/checkpoint access,
runtime/FLOPs, deployment, dynamic-budget/scanner-quality validation, metric
claims, or paper claims. Because it adds a dynamic-budget protocol surface, the
next gate is GPT-5.5 Pro read-only implementation review, then Gemini CLI
read-only review only if Pro allows it.

## R16 Bridge Registry Fix

Commit `a09a2476f79a4a78dba590f55e855e2a2f5fcd1b` fixes the registry/import
path that caused remote R16 smoke job `1105455` to fail during model build:

```text
KeyError: 'PCOTMRASDetectorBridge is not in the opentad::models registry'
```

Accepted fix:

- `opentad/models/necks/__init__.py` now imports and exports
  `PCOTMRASDetectorBridge` and
  `ProcessConditionedOrderedTransportMRASDetectorBridge`, so the normal
  OpenTAD package import path registers the bridge before `build_neck()`.
- `tests/test_pc_ot_mras_config_integration.py` now includes a regression test
  for the package-level `opentad.models.necks` import path and verifies that
  `builder.build_neck({"type": "PCOTMRASDetectorBridge", ...})` succeeds.

Verification:

```text
py_compile changed files: pass
tests/test_pc_ot_mras_config_integration.py: 3 passed in 6.17s
full PC-OT-MRAS pytest plus train-engine max-iter: 206 passed in 47.82s
git diff --check: pass, with LF/CRLF warnings only
```

Remote state refreshed at `2026-06-20T02:32:59+08:00`:

```text
R16 1105455: FAILED 1:0 due PCOTMRASDetectorBridge registry miss.
R17 1105456: PENDING, dependency-blocked by failed R16.
R20 1105574: COMPLETED 0:0, R20_VALUE_PRECHECK_ONLY_PASS_NO_TRAIN_NO_DATA_NO_MAP.
R18 1105575: FAILED 2:0 in PRECHECK_ONLY due missing reviewed pretrained file.
```

Boundary: this is an infrastructure registration fix only. It does not change
input sampling, dynamic budget policy, token compression, Adapter/backbone
internals, detector head logic, loss/assignment, post-processing, or launcher
gate logic. It does not authorize remote sync, remote PRECHECK execution,
Slurm/GPU, `tools/train.py`, `tools/test.py`, detector mAP,
dataset/checkpoint access, runtime/FLOPs, deployment, dynamic-budget or
scanner-quality validation, metric claims, or paper claims. A focused Gemini
CLI read-only review is required before using this fix for a clean remote
R16/R17 requeue.

Focused Gemini CLI read-only review result:

```text
gemini-3-pro-preview exitcode: 0
elapsed: 34.77s
verdict: PASS_ALLOW_CLEAN_R16_REQUEUE_PREPARATION_ONLY
blocking findings: none
stdout: logs/gemini3_pro_preview_ctf_bdi_pc_ot_mras_r16_bridge_registry_fix_20260620_0233.txt
```

The Gemini pass allows only clean R16/R17 requeue preparation. It does not
authorize remote sync, remote PRECHECK execution, Slurm/GPU, `tools/train.py`,
`tools/test.py`, detector mAP, dataset/checkpoint access, runtime/FLOPs,
deployment, dynamic-budget/scanner-quality validation, metric claims, or paper
claims.

## R17 Clean Formal Launcher Gate

Commit `87d3404a371b3edf7a98d43f0e84089bfb7d534a` adds the missing clean-repo
R17 formal training launcher and tightens the R17 config entrypoint boundary.

Accepted implementation details:

- `configs/adatad/thumos/ctf_bdi_pc_ot_mras_r17_formal_train_candidate.py`
  now requires launcher-provided entrypoint gate context before
  `tools/train.py` is accepted. The required gate binds
  `OPENTAD_PCOTMRAS_ENTRYPOINT_GATE_JSON`,
  `OPENTAD_PCOTMRAS_ENTRYPOINT_GATE_SHA256`,
  `OPENTAD_PCOTMRAS_ACTIVE_MANIFEST_SHA256`, and
  `OPENTAD_PCOTMRAS_RESOLVED_CONFIG_SHA256`.
- `scripts/run_ctf_bdi_pc_ot_mras_r17_formal_train_n16r4.sbatch` is scoped to
  the clean repo name `OpenTAD_PCOTMRAS_R16_R18_R20_Clean_20260619_1730` and
  refuses any path containing `OpenTAD_BATA_Clean`.
- The launcher defaults to `PRECHECK_ONLY=1`. Formal `tools/train.py`
  execution requires `PRECHECK_ONLY=0`, `ALLOW_R17_FORMAL_TRAIN=1`, and an
  explicit `R17_FORMAL_GATE_JSON`/`R17_FORMAL_GATE_SHA256` pair bound to the
  active SHA manifest and resolved config SHA.
- Direct `tools/test.py`, raw prediction caches, checkpoint/load/resume
  shortcuts, arbitrary `CFG_OPTIONS`, metric/paper/runtime/deploy claims, and
  dirty tracked clean-repo files remain rejected.
- R17 remains reader-only: it does not add R18 aux loss, R19 soft/hard loss,
  R20 value loss, R21 tensor temporal-coordinate changes, or R22 dynamic budget
  logic. Train-time validation can be produced by `tools/train.py`, but it is
  not a metric or paper claim without later audit.

Verification:

```text
py_compile changed R17 config/test files: pass
focused R17/R18/R20/guard pytest: 19 passed in 5.78s
bash -n R17 sbatch launcher: exit code 0, with local WSL warning noise
git diff --check: pass, with LF/CRLF warnings only
full PC-OT-MRAS pytest plus train-engine max-iter: 207 passed in 47.80s
```

Boundary: this is local clean-repo launcher/gate preparation. It does not
authorize remote sync, remote PRECHECK execution, Slurm/GPU execution,
`tools/train.py`, `tools/test.py`, detector mAP, dataset/checkpoint access,
runtime/FLOPs, deployment, dynamic-budget/scanner-quality validation, metric
claims, or paper claims. Because it changes launcher/gate logic, the next gate
is a complete GPT-5.5 Pro read-only review package, then Gemini CLI read-only
review only if Pro allows it.

## R23 Dynamic Budget Hard Export

Commit `05e738861997022631364babd75ed866d617aec9` adds the local-only R23
dynamic-budget hard export protocol layer. This closes the immediate gap after
R22 by converting a deploy-visible R22 budget plan into validated hard-position
rows for later detector-geometry plumbing.

Accepted implementation details:

- `tools/bata/export_pc_ot_mras_hard_positions.py` now exposes
  `resolve_pc_ot_mras_dynamic_budget_plan(...)`.
- The resolver consumes only dynamic-plan fields such as `budgets`,
  `dense_valid_len`, `selected_dense_positions`, and `selected_mask`; it does
  not read GT, teacher, cache, raw prediction, checkpoint, result, or
  train-only value-target payloads.
- It validates exact per-sample variable budgets, prefix selected masks,
  sorted unique selected dense positions, dense selected-mask serialization,
  and false-only claim/safety flags.
- The output stays in the existing `pc_ot_mras_hard_positions_v0` row schema
  and records `pc_ot_mras_dynamic_budget_hard_export_resolver_v0` as its
  generation source.
- `configs/adatad/thumos/ctf_bdi_pc_ot_mras_r23_dynamic_budget_hard_export.py`
  is launch-blocked and inherits the R22 controller without enabling detector
  training, `tools/train.py`, `tools/test.py`, mAP, remote sync, Slurm/GPU,
  raw-prediction cache, checkpoint access, or claims.

Verification:

```text
py_compile changed R23 files: pass
R22/R23 focused pytest in torch_1: 11 passed in 8.15s
hard-export/controller pytest in torch_1: 39 passed in 6.16s
git diff --check: pass, with LF/CRLF warnings only
full PC-OT-MRAS pytest plus train-engine max-iter in torch_1: 212 passed in 47.70s
```

Boundary: R23 is local implementation evidence only. It is not
dynamic-budget quality validation, detector mAP evidence, runtime/FLOPs proof,
deployment evidence, or a paper claim. Because it extends the dynamic-budget
protocol surface, the next required gate is GPT-5.5 Pro read-only
implementation review, followed by Gemini CLI read-only review only if Pro
returns `PASS_ALLOW_GEMINI_REVIEW_ONLY`.

## R24 Dynamic Budget Temporal Metadata Contract

Commit `3f466eeb2c53a34807c17159b96f63ff90fa0c32` adds the local-only R24
protocol layer that connects R23 hard-position rows to the detector temporal
metadata contract.

Accepted implementation details:

- `tools/bata/export_pc_ot_mras_hard_positions.py` now exposes
  `pc_ot_mras_hard_rows_to_temporal_metas(...)`.
- The converter validates `pc_ot_mras_hard_positions_v0` rows before producing
  metadata: exact budget, sorted unique selected positions, selected-mask
  consistency, dense valid length bounds, forbidden payload rejection, and
  false-only claim/safety flags.
- The produced metadata carries
  `irregular_selected_positions`, `irregular_dense_valid_len`,
  `irregular_selected_valid_len`, `irregular_selected_count`, and
  `irregular_native_axis=True`, plus dense/native axis tags for GT, targets,
  proposals, temporal decode, and segments.
- The metadata is designed for `validate_sampling_contract(...)` and
  `temporal_grid_from_metas(...)` and records
  `pc_ot_mras_hard_rows_to_temporal_metadata_v0` as its generation source.
- `configs/adatad/thumos/ctf_bdi_pc_ot_mras_r24_dynamic_budget_temporal_metadata.py`
  is launch-blocked and inherits R23 without enabling detector training,
  `tools/train.py`, `tools/test.py`, mAP, remote sync, Slurm/GPU,
  raw-prediction cache, checkpoint access, dynamic-budget validation, or
  claims.

Verification:

```text
py_compile changed R24 files: pass
focused R24 pytest in torch_1: 4 passed in 4.51s
R22/R23/R24 temporal metadata regression in torch_1: 20 passed in 10.38s
git diff --check: pass, with LF/CRLF warnings only
full PC-OT-MRAS pytest plus train-engine max-iter in torch_1: 216 passed in 48.22s
```

Focused GPT-5.5 Pro same-thread recall review for the single-file inline R24
context returned `PASS_ALLOW_GEMINI_REVIEW_ONLY`. Gemini CLI read-only review
then returned `REJECT_REQUIRES_FIXES` for two R24 hard-row validation blockers:

```text
Gemini CLI stdout: logs/gemini3_pro_preview_ctf_bdi_pc_ot_mras_r24_20260620_0429.txt
exitcode: 0
verdict: REJECT_REQUIRES_FIXES
blocking finding 1: _as_int_list silently truncated non-integer float positions
blocking finding 2: forbidden generic string values could carry deploy-invisible teacher/oracle/cache/provenance terms
```

Commit `3ffec40` fixes both accepted Gemini blockers:

- `tools/bata/export_pc_ot_mras_hard_positions.py` now routes
  `selected_positions` through `_strict_int_scalar(...)`, rejecting bools,
  non-integer floats, and other non-integer scalars instead of silently using
  `int(...)`.
- `_validate_no_forbidden_jsonl_keys(...)` now rejects forbidden
  deploy-invisible fragments in generic string values, not only in JSON object
  keys.
- `tests/test_pc_ot_mras_dynamic_budget_temporal_metadata.py` adds regression
  coverage for non-integer float selected positions and generic forbidden
  provenance strings.

Post-fix verification:

```text
py_compile changed R24 files: pass
focused R24 pytest in torch_1: 4 passed in 4.46s
R22/R23/R24 temporal metadata regression in torch_1: 20 passed in 10.02s
full PC-OT-MRAS pytest plus train-engine max-iter in torch_1: 216 passed in 48.83s
git diff --check: pass, with LF/CRLF warnings only
```

Focused Gemini CLI read-only review of the blocker fix returned pass:

```text
prompt: logs/gemini3_pro_preview_ctf_bdi_pc_ot_mras_r24_fix_prompt_20260620_0438.md
stdout: logs/gemini3_pro_preview_ctf_bdi_pc_ot_mras_r24_fix_20260620_0438.txt
stderr: logs/gemini3_pro_preview_ctf_bdi_pc_ot_mras_r24_fix_20260620_0438.err.txt
exitcode: logs/gemini3_pro_preview_ctf_bdi_pc_ot_mras_r24_fix_20260620_0438.exitcode.txt
verdict: PASS_R24_FIX_GEMINI_REVIEW_ONLY
blocking findings: none
accepted decision: GEMINI_R24_FIX_REVIEW=YES; all remote sync, remote PRECHECK,
Slurm/GPU, tools/train.py, tools/test.py, detector mAP, dataset/checkpoint,
runtime/FLOPs, deployment, dynamic-budget/scanner-quality validation, metric
claim, and paper claim remain NO.
```

Review caveat: stderr contains Gemini terminal-color warnings and two internal
`run_shell_command` tool-not-found messages. The process still exited `0` and
stdout gave a substantive file/function/test-level review. This caveat is
recorded so the review is not later described as a noise-free tool run.

Boundary: R24 is local protocol/geometry evidence only. It is not
dynamic-budget quality validation, detector mAP evidence, runtime/FLOPs proof,
deployment evidence, scanner-quality validation, or a paper claim. The R24
blocker-fix review gate is complete for local source status only. It does not
authorize remote sync, remote PRECHECK execution, Slurm/GPU, `tools/train.py`,
`tools/test.py`, detector mAP, dataset/checkpoint access, runtime/FLOPs,
deployment, dynamic-budget/scanner-quality validation, metric claims, or paper
claims.

## R25 Dynamic Budget Pipeline Validation

Commit `da59fe334f667977ec04e6a09d350c21c0f17d76` adds the local-only R25
protocol validation layer. It turns the existing R22 -> R23 -> R24 inline
contract checks into a reusable function/CLI and a launch-blocked config for
future review/precheck packages.

Accepted implementation details:

- `tools/bata/validate_pc_ot_mras_dynamic_budget_pipeline.py` exposes
  `validate_pc_ot_mras_dynamic_budget_pipeline(...)` and `run_json_validation(...)`.
- The validator chains `resolve_pc_ot_mras_dynamic_budget_plan(...)`,
  `pc_ot_mras_hard_rows_to_temporal_metas(...)`,
  `validate_sampling_contract(...)`, and `temporal_grid_from_metas(...)`.
- The summary reports row count, sample ids, per-sample budgets, dense valid
  lengths, selected counts, temporal-grid valid counts, budget min/max/mean,
  coverage share summary, exact-budget violations, and center-row alignment.
- All summary claim/execution flags remain false: no remote sync, remote
  PRECHECK, Slurm/GPU, `tools/train.py`, `tools/test.py`, detector mAP,
  dataset/checkpoint, runtime/FLOPs, deployment, dynamic-budget validation,
  scanner-quality validation, metric claim, or paper claim.
- `configs/adatad/thumos/ctf_bdi_pc_ot_mras_r25_dynamic_budget_pipeline_validation.py`
  inherits R24 and is launch-blocked.
- Tests cover function-level synthetic validation, JSON roundtrip, forbidden
  generic provenance-string rejection, and config/guard closure.

Verification:

```text
py_compile changed R25 files: pass
focused R25 pytest in torch_1: 4 passed in 4.41s
R22/R23/R24/R25 regression in torch_1: 19 passed in 11.42s
full PC-OT-MRAS pytest plus train-engine max-iter in torch_1: 220 passed in 53.22s
git diff --check: pass
```

Boundary: R25 is local protocol validation evidence only. It is not
dynamic-budget quality validation, detector mAP evidence, runtime/FLOPs proof,
deployment evidence, scanner-quality validation, or a paper claim. Because it
adds a new local validation tool/config, it requires the normal post-
implementation review chain before it can be used for any later remote package
or execution gate.

## R25 Pro Blocker Fix

The first GPT-5.5 Pro R25 package review was harvested from the browser page
after Rosetta local stdout/stderr harvest stalled. The model-visible page
contained the uploaded R25 zip and prompt and returned:

```text
Verdict: FIX_BEFORE_GEMINI
Gemini CLI read-only review: DENY for this R25 package
```

Accepted blockers:

- The dynamic-budget plan validator still accepted deploy-invisible or
  permission/claim fields such as value targets, labels, segments,
  annotation paths, `allow_*` execution flags, and non-approved claim or
  validation flags.
- The review zip omitted `tests/pc_ot_mras_test_utils.py`, so included tests
  could not be collected from the overlay package alone.

Commit `c01bae7` fixes the source blocker:

- `tools/bata/export_pc_ot_mras_hard_positions.py` now uses an explicit
  allowlist for deploy-visible dynamic-budget plan keys.
- Unknown dynamic-plan keys fail closed, while explicit execution/claim,
  annotation/label/segment/target, teacher/oracle/cache/raw-prediction,
  result, and checkpoint fragments remain forbidden.
- Existing false-only flags must stay false.
- R23 and R25 tests now cover the Pro-listed forbidden payloads and permission
  / claim flags, plus an unknown side-channel key.

Post-fix verification:

```text
py_compile changed R25 fix files: pass
focused R23/R25 dynamic-budget pytest in torch_1: 47 passed in 5.95s
R22/R23/R24/R25 regression in torch_1: 59 passed in 10.08s
full PC-OT-MRAS pytest plus train-engine max-iter in torch_1: 260 passed in 51.36s
git diff --check: pass, with LF/CRLF warnings only
```

Boundary: this closes only the local R25 blocker in source. It does not
authorize Gemini, remote sync, remote PRECHECK execution, Slurm/GPU,
`tools/train.py`, `tools/test.py`, detector mAP, dataset/checkpoint access,
runtime/FLOPs, deployment, dynamic-budget/scanner-quality validation, metric
claim, or paper claim. A rebuilt R25 fix Pro package must include
`tests/pc_ot_mras_test_utils.py` and request only whether Gemini CLI read-only
review may run.

## R16/R17/R18 Slurm Master-Port Guard Fix

After clean-head formal queue submission, R16A job `1105984 pcot_r16run`
failed before model training because the Slurm environment provided an invalid
`MASTER_PORT=415984`. The old launchers accepted any numeric `MASTER_PORT`, so
`torchrun` failed while parsing rendezvous endpoint `127.0.0.1:415984`.

This fix hardens the clean repo launchers:

- `scripts/run_ctf_bdi_pc_ot_mras_r16a_gpu_smoke_n16r4.sbatch`
- `scripts/run_ctf_bdi_pc_ot_mras_r17_formal_train_n16r4.sbatch`
- `scripts/run_ctf_bdi_pc_ot_mras_r18_aux_formal_train_n16r4.sbatch`

Each launcher now computes a safe default port from the Slurm job id, records
`MASTER_PORT_SOURCE`, and falls back to the default if an inherited
environment `MASTER_PORT` is non-numeric or outside `[1024, 65535]`. The port
source and default are printed in the launch log so future failures can be
audited. `tests/test_pc_ot_mras_launcher_master_port.py` covers the static
contract for all three launchers.

Verification in `torch_1`:

```text
pytest tests/test_pc_ot_mras_launcher_master_port.py \
  tests/test_pc_ot_mras_r16a_gpu_smoke_launcher.py \
  tests/test_pc_ot_mras_r17_formal_train_config.py \
  tests/test_pc_ot_mras_r18_aux_formal_train_config.py \
  tests/test_pc_ot_mras_train_local_only_guard.py -q
  25 passed in 3.94s

py_compile tests/test_pc_ot_mras_launcher_master_port.py: pass
git diff --check: pass, with LF/CRLF warnings only
full PC-OT-MRAS pytest plus train-engine max-iter in torch_1:
  262 passed in 51.16s
```

Boundary: this is a launcher robustness fix only. It does not produce detector
mAP, runtime/FLOPs, deployment, dynamic-budget/scanner-quality validation,
metric claims, or paper claims.

## NativeIrregularAreaHeadP2 Registry Fix

Rootfix R16A Slurm job `1106086` reached `tools/train.py`, built the dataset,
loaded the reviewed pretrained initialization, and then failed during model
construction because `NativeIrregularAreaHeadP2` was not registered through the
normal OpenTAD package import path:

```text
KeyError: 'NativeIrregularAreaHeadP2 is not in the opentad::models registry'
```

Commit `42b9ab7c94389aa3532481c520138017f442b22c` fixes this clean-repo
runtime registration gap:

- `opentad/models/dense_heads/__init__.py` imports and exports
  `NativeIrregularAreaHeadP2`, so `from opentad.models import *` registers the
  PC-OT-MRAS native irregular-area head before `build_head(...)`.
- `tests/test_pc_ot_mras_config_integration.py` adds a package-import
  regression test that imports `opentad.models.dense_heads`, verifies the
  registered class, and builds a minimal `NativeIrregularAreaHeadP2` through
  `builder.build_head(...)`.

Verification in `torch_1`:

```text
py_compile opentad/models/dense_heads/__init__.py \
  tests/test_pc_ot_mras_config_integration.py: pass

focused registry/R16/R17/R18 pytest:
  22 passed in 9.00s

full PC-OT-MRAS pytest plus train-engine max-iter:
  263 passed in 53.18s

git diff --check:
  pass, with LF/CRLF warnings only
```

Boundary: this is a package-import/registry fix only. It does not change input
sampling, dynamic budget policy, losses, assignment, post-processing, detector
mAP, runtime/FLOPs, deployment, or any metric/paper claim.

## Native Head Active-Manifest Binding Fix

Commit `8effdb01da43e7641f172806a3d326552a73eca4` extends the active SHA
manifest generated by the R16A, R17, and R18 Slurm launchers to include the
runtime files that register and define the native irregular-area head:

```text
opentad/models/dense_heads/__init__.py
opentad/models/dense_heads/native_irregular_area_head_p2.py
```

This ensures the next execution gates bind the head registry fix that R16A
needed to proceed past `build_head(...)`. Static launcher tests now assert
that these files are present in the manifest list for R16A, R17, and R18.

Verification in `torch_1`:

```text
focused launcher/registry pytest:
  17 passed in 9.27s

full PC-OT-MRAS pytest plus train-engine max-iter:
  263 passed in 53.61s

git diff --check:
  pass, with LF/CRLF warnings only
```

Boundary: this is gate-manifest hardening only. It does not authorize
`tools/test.py`, detector mAP, runtime/FLOPs, deployment, metric claims, or
paper claims.

## R27 Synthetic Task-Utility Audit

R27 adds a local-only synthetic task-utility audit above the R22-R26 dynamic
budget protocol chain. Unlike R26, which only checks budget distribution,
savings, monotonicity, and coverage-cap behavior, R27 asks whether a dynamic
budget plan selects TAD-sensitive synthetic structure under the same resolved
hard-position protocol:

- start/end boundary support;
- interior evidence peak recall;
- redundant/background suppression;
- same-budget exact-uniform control comparison;
- synthetic oracle top-k utility ratio;
- hard/medium/easy difficulty budget ordering.

Changed files:

```text
tools/bata/audit_pc_ot_mras_synthetic_task_utility.py
configs/adatad/thumos/ctf_bdi_pc_ot_mras_r27_synthetic_task_utility_audit.py
tests/test_pc_ot_mras_synthetic_task_utility_audit.py
tests/test_pc_ot_mras_r27_synthetic_task_utility_config.py
PC_OT_MRAS_R16_R18_R20_CLEAN_MANIFEST.md
```

Verification in `torch_1`:

```text
py_compile R27 tool/config/tests:
  pass

focused R27 pytest:
  7 passed in 4.38s

R22-R27 dynamic-budget regression:
  72 passed in 16.22s

full PC-OT-MRAS pytest plus train-engine max-train-iter test:
  276 passed in 63.17s
```

Boundary: R27 is a synthetic local utility audit only. It does not read real
annotations, datasets, checkpoints, raw predictions, teacher outputs, or result
caches. It produces no detector mAP, runtime/FLOPs, deployment evidence,
dynamic-budget quality validation, scanner-quality validation, metric claim, or
paper claim. It should be used to decide whether further local protocol-only
audits are becoming low-information: after R27, the next high-information step
is a valid read-only review package and then a real no-GT detector/precheck
gate, not another budget-mechanics-only diagnostic.

## R16A AMP-Safe Mask Sentinel Fix

Remote R16A GPU smoke job `1106493 pcot_r16smk` reached `tools/train.py`,
entered AMP training, and failed in
`opentad/models/selectors/pc_ot_mras_reader.py::_dense_heads` with:

```text
RuntimeError: value cannot be converted to type at::Half without overflow
```

Commit `9744e6fb0f45a1db2702d45e034f95e98b29cb11` fixes the dtype mismatch
that caused this failure. Under autocast, `h` can remain float32 while the
individual linear head outputs are half tensors. The previous `_dense_heads`
implementation computed the negative mask sentinel from `h.dtype`, so it could
try to write a float32-range sentinel into a half logits tensor. The fix adds
dtype-local `_mask_logits(...)` helpers and always derives the sentinel from
the logits tensor being masked.

Changed files:

```text
opentad/models/selectors/pc_ot_mras_reader.py
opentad/models/selectors/lowcost_acquisition_browser.py
tests/test_pc_ot_mras_reader_shapes.py
PC_OT_MRAS_R16_R18_R20_CLEAN_MANIFEST.md
```

The same output-dtype sentinel fix is applied to
`lowcost_acquisition_browser.py` because it had the same AMP risk pattern:
`h.dtype` could differ from the acquisition/head logits dtype.

Verification in `torch_1`:

```text
py_compile changed files:
  pass

focused reader/browser sentinel pytest:
  7 passed in 4.10s

PC-OT-MRAS pytest plus train-engine max-train-iter test:
  284 passed in 69.62s

git diff --check:
  pass, with LF/CRLF warnings only

static negative-sentinel search:
  no remaining selector/loss pattern of h.dtype-derived negative mask fill
```

Boundary: this is an AMP launch-path bug fix for invalid/padded logits only.
It does not change valid-position logits, input sampling, dynamic budget
policy, detector head semantics, loss targets, assignment, post-processing,
GT/teacher/raw-prediction boundaries, detector mAP, runtime/FLOPs, deployment
evidence, metric claims, or paper claims. R17/R18 dependency jobs from the
failed `1106493` wave are dependency-dead and must be cancelled/requeued after
the fixed head is synced and checked.

## R16A AMP-Safe Probability Entropy Fix

Remote R16A GPU smoke job `1107077 pcot_r16smk` reached epoch 0 and failed
before any detector mAP with:

```text
ValueError: pc_ot_mras_reader_outputs.regularizers.pair_entropy_loss must be finite
```

The dependent jobs `1107078 pcot_r17tr` and `1107079 pcot_r18aux` became
`DependencyNeverSatisfied` and were cancelled before preparing a fresh wave.

Commit `990aef92affc2a30e17273a847c68e519b79ab30` fixes the entropy
calculation in `opentad/models/selectors/pc_ot_mras_reader.py`. The failure
was an AMP half-precision zero-probability issue: the old formula clamped
probabilities with `1.0e-8`, but that value underflows to zero in `float16`;
`0 * log(0)` then becomes NaN and the detector bridge correctly rejects the
reader output tree as non-finite. The fix adds `_prob_entropy(...)`, casts the
probability tensor to float32 before clamp/log, and uses it for both allocation
entropy and pair entropy. The detector bridge finite gate is not weakened.

Changed files:

```text
opentad/models/selectors/pc_ot_mras_reader.py
tests/test_pc_ot_mras_reader_pair_distribution.py
PC_OT_MRAS_R16_R18_R20_CLEAN_MANIFEST.md
```

Verification in the local Windows `torch_1` environment:

```text
py_compile changed reader/test files:
  pass

focused reader pair-distribution pytest:
  5 passed in 20.17s

full PC-OT-MRAS pytest plus train-engine max-train-iter test:
  301 passed in 74.30s

git diff --check:
  pass, with LF/CRLF warnings only
```

Boundary: this is a numerical finite-value repair for reader regularizer
reporting under AMP. It does not change input sampling, dynamic-budget policy,
Adapter/backbone packed-token routing, detector head semantics, loss targets,
assignment, post-processing, GT/teacher/raw-prediction boundaries, detector
mAP, runtime/FLOPs, deployment evidence, metric claims, or paper claims.

Gemini CLI read-only review for this focused fix completed with exitcode `0`
and verdict `PASS_ALLOW_REMOTE_SYNC_REQUEUE_ONLY`:

```text
stdout: logs/gemini3_pro_preview_ctf_bdi_pc_ot_mras_pair_entropy_finite_fix_20260621.txt
stderr: logs/gemini3_pro_preview_ctf_bdi_pc_ot_mras_pair_entropy_finite_fix_20260621.err.txt
process: logs/gemini3_pro_preview_ctf_bdi_pc_ot_mras_pair_entropy_finite_fix_20260621.process.json
```

This allows only remote sync of the clean repo and a fresh R16A/R17/R18
dependency-wave requeue. It does not authorize detector mAP, runtime/FLOPs,
deployment, dynamic-budget validation, spatial-redundancy validation, metric
claims, or paper claims.

## R16A Smoke Output Audit Narrowing And R31 Contract Regressions

Remote R16A GPU smoke job `1111913 pcot_r16smk` ran the latest
`b162d4a` code, reached `tools/train.py`, built the model/datasets, completed
the bounded train-entry smoke, printed finite epoch-0 loss, hit
`max_train_iters=2`, and printed `Training Over`. It then failed only in the
launcher post-run stdout audit:

```text
ERROR train stdout contains forbidden testing/mAP/result marker
```

The failure was a launcher audit false positive, not the previous AMP
Float/Half or pair-entropy NaN failures. The R16A config text printed in the
training stdout contains phrases such as "detector mAP is not approved"; the
old post-run guard searched for bare `mAP` anywhere in stdout, so it rejected a
valid no-test smoke. The launcher now checks only real detector-evaluation
markers: `Testing Starts`, `Average-mAP`, `mAP at tIoU`, and
`result_detection`.

This update also adds two R23-R31 contract regressions before moving R23+
toward remote precheck:

1. A production `VisionTransformerAdapter.forward()` opt-in test with
   `depth=3` verifies that packed route execution is mutually exclusive with
   the dense block loop. It asserts packed block calls equal depth, dense block
   calls are zero, packed attention/MLP counts equal depth, and adapter count
   remains one.
2. A dynamic selected-position to temporal-tubelet contract test verifies that
   selected dense positions map to valid temporal tubelet ids, expand only to
   complete spatial groups, preserve stable pack/scatter order, and fail closed
   on ragged batch selected-token counts until padding/grouping is explicitly
   implemented.

Changed files:

```text
scripts/run_ctf_bdi_pc_ot_mras_r16a_gpu_smoke_n16r4.sbatch
tests/test_pc_ot_mras_r16a_gpu_smoke_launcher.py
tests/test_pc_ot_mras_packed_tubelet_forward_route.py
PC_OT_MRAS_R16_R18_R20_CLEAN_MANIFEST.md
```

Verification in the local Windows `torch_1` environment:

```text
focused R16A/R31 pytest:
  5 passed, 1 skipped in 0.97s

R22-R31 focused regression set:
  16 passed, 7 skipped in 8.94s

R16A sbatch bash -n:
  pass

git diff --check:
  pass, with LF/CRLF warnings only
```

Boundary: this is launcher audit hardening and local/remote-precheck test
coverage for R23-R31 contracts. It does not change model training semantics,
input sampling, dynamic-budget policy, detector head semantics, loss targets,
assignment, post-processing, GT/teacher/raw-prediction boundaries, detector
mAP, runtime/FLOPs, deployment evidence, metric claims, spatial-redundancy
claims, dynamic-budget validation claims, or paper claims. R23-R31 remain
precheck/review evidence only until a valid execution gate explicitly unlocks
the next stage.
