# PC-OT-MRAS R16/R18/R20 Clean Implementation Manifest

Timestamp: 2026-06-19T18:34+08:00

This repository was created from the manually downloaded clean OpenTAD source and initialized as a new git repository. The clean baseline is commit `f19492b` (`Import clean OpenTAD baseline`).

Current branch: `codex/pcotmras-r16-r18-r20-clean`

Current commits:

```text
f19492b Import clean OpenTAD baseline
2301aaa50810d6fe7d8d88760bd2e789896e93d8 Implement clean PC-OT-MRAS R16 R18 R20
5ca29aba79ca152b9070905f97c93b29948258ac Add R20 value-only attribution control
73fc7ba Fix clean PC-OT-MRAS launch guards
```

## Objective

Implement only the PC-OT-MRAS continuous route stages on a clean OpenTAD baseline:

- R16 bounded GPU smoke candidate.
- R18 train-only reader auxiliary diagnostic candidate.
- R20 train-only value-of-information distillation candidate.

The implementation intentionally does not copy unrelated BATA/UGIT/MFCSD/DBAC/ITMI experiment families from the dirty working tree.

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

| Config | Reader | Aux loss | Value loss | Raw prediction cache |
| --- | --- | --- | --- | --- |
| `ctf_bdi_pc_ot_mras_r16_gpu_smoke_candidate.py` | present | absent | absent | disabled |
| `ctf_bdi_pc_ot_mras_r17_formal_train_candidate.py` | present | absent | absent | disabled |
| `ctf_bdi_pc_ot_mras_r18_aux_diag_candidate.py` | present | present | absent | disabled |
| `ctf_bdi_pc_ot_mras_r20_value_distill_candidate.py` | present, `enable_value_heads=True` | present | present | disabled |

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
