# PC-OT-MRAS R16/R18/R20 Clean Implementation Manifest

Timestamp: 2026-06-19T18:44+08:00

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
