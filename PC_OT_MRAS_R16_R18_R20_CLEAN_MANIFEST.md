# PC-OT-MRAS R16/R18/R20 Clean Implementation Manifest

Timestamp: 2026-06-19T22:10+08:00

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
96eaaeb Add R18 and R20 confirmation candidates
b5b4448 Update R18 R20 review package manifest
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
| `ctf_bdi_pc_ot_mras_r18_aux_formal_train_candidate.py` | present | present | absent | disabled |
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
