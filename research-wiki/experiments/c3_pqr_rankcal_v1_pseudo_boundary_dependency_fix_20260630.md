# C3 PQR RankCal V1 Pseudo-Boundary Dependency Fix - 2026-06-30

Route: `C3_MAINLINE_OPTIMIZATION` / `C3_ORIGINAL_OPTIMIZATION_ROUTE`.

Scope: local clean-clone runtime dependency completeness fix only. No SSH,
Slurm, remote PRECHECK, remote smoke, training, evaluation, or remote write was
performed.

## Failure Evidence

Remote clean clone:
`/data/home/sczc063/run/yuzibo/OpenTAD_C3PQRRankCal_Precheck_20260630/github_clean_c3_pqr_rankcal_v1`.

Remote branch HEAD at failure:
`8cb6b64f83c1b9e86d887978accf1cabfe6f7d34`.

Linux PRECHECK before the smoke had already passed: py_compile pass, three PQR
validators pass, and focused pytest `19 passed in 19.27s`.

The 2-iteration runtime smoke failed before the training loop started:

```text
ModuleNotFoundError: No module named 'opentad.datasets.transforms.pseudo_boundary'
```

The import path was `opentad.datasets.transforms.end_to_end`.

## Root Cause

`opentad/datasets/transforms/end_to_end.py` hard-imports
`opentad.datasets.transforms.pseudo_boundary`, and the PQR branch did not include
that real helper module in a clean clone.

This was a branch dependency completeness gap, not a PQR sparse-head/ranking
calibration failure and not a selector, sampler, evaluator, post-processing, or
scoring-math issue.

## Fix

- Added `opentad/datasets/transforms/pseudo_boundary.py` from the existing local
  history implementation matching the `end_to_end.py` API.
- Kept `end_to_end.py` as a hard import. This avoids masking a true dependency
  and preserves the ordinary end-to-end transform import path.
- Added a PQR validator/test guard for the clean-clone transform dependency and
  required API symbols.
- Updated the three PQR config metadata fields from "pseudo_boundary missing" to
  "dependency restored locally; remote PRECHECK and 2-iter smoke evidence still
  required".

## Changed Surface

- Input-side helper dependency restoration only:
  `opentad/datasets/transforms/pseudo_boundary.py`.
- PQR validator/test metadata guard only:
  `tools/validate_c3_pqr_rankcal_v1_config.py` and
  `tests/test_c3_pqr_rankcal_v1_config.py`.
- PQR config metadata only:
  three `c3_indirect_original_adatad_32px_a_*pqr_rankcal_v1*.py` configs.

No CADF selector, BH-SDC, evaluator, post-process, sampler, PQR scoring math,
quality-head math, detector head logic, loss/assignment, training launcher, or
runtime gate behavior was changed.

## Local Verification

Commands run in
`E:\DeskTop\TAD\temrefuse-tad\OpenTAD_C3PQRRankCal_Worktree_20260629`:

```powershell
python -m pytest tests/test_c3_pqr_rankcal_v1_config.py::test_clean_clone_transform_dependencies_are_present_for_runtime_imports -q
```

RED before adding the module: failed with `missing clean-clone transform
dependency: opentad/datasets/transforms/pseudo_boundary.py`.

GREEN after adding the module: `1 passed, 1 warning`.

```powershell
python -m py_compile opentad\datasets\transforms\pseudo_boundary.py opentad\datasets\transforms\end_to_end.py tools\validate_c3_pqr_rankcal_v1_config.py tests\test_c3_pqr_rankcal_v1_config.py tests\test_c3_pqr_rankcal_v1_quality_head.py configs\adatad\thumos\c3_indirect_original_adatad_32px_a_pqr_rankcal_v1_precheck.py configs\adatad\thumos\c3_indirect_original_adatad_32px_a_pqr_rankcal_v1_shortdiag.py configs\adatad\thumos\c3_indirect_original_adatad_32px_a_exact_uniform_backend_control_pqr_rankcal_v1_shortdiag.py
```

Result: pass.

```powershell
python -m pytest tests/test_c3_pqr_rankcal_v1_config.py tests/test_c3_pqr_rankcal_v1_quality_head.py -q
```

Result: `17 passed, 3 skipped, 1 warning`. The three skipped tests are
torch-backed quality-head tests affected by the local Windows torch DLL import
failure; Linux PRECHECK must rerun them.

```powershell
python tools\validate_c3_pqr_rankcal_v1_config.py configs\adatad\thumos\c3_indirect_original_adatad_32px_a_pqr_rankcal_v1_precheck.py
python tools\validate_c3_pqr_rankcal_v1_config.py configs\adatad\thumos\c3_indirect_original_adatad_32px_a_pqr_rankcal_v1_shortdiag.py
python tools\validate_c3_pqr_rankcal_v1_config.py configs\adatad\thumos\c3_indirect_original_adatad_32px_a_exact_uniform_backend_control_pqr_rankcal_v1_shortdiag.py
```

Result: all three printed `PASS_C3_PQR_RANKCAL_V1_CONFIG`.

```powershell
python -m pytest tests/test_adapter_safety_contracts.py::test_pseudo_boundary_hybrid_selects_window_local_teacher_boundaries tests/test_adapter_safety_contracts.py::test_pseudo_boundary_hybrid_falls_back_to_random_fixed_without_cache tests/test_adapter_safety_contracts.py::test_pseudo_boundary_snap_keeps_random_fixed_distribution_local tests/test_adapter_safety_contracts.py::test_pseudo_boundary_snap_falls_back_to_random_fixed_without_cache tests/test_adapter_safety_contracts.py::test_pseudo_boundary_loader_rejects_gt_manifest -q
```

Result: `5 passed in 0.16s`.

## Review Attempt

Required read-only final review was attempted but no valid review output was
available:

- Claude review synchronous call: invalid, `Claude CLI did not return JSON
  output`.
- Claude review background job
  `8dbe1a95a2e74617882ff305d0a6e3fe`: failed with the same error.
- GPT-4o fallback via `llm_chat`: unavailable, `LLM_API_KEY environment
  variable not set`.
- MiniMax fallback: unavailable, `MINIMAX_API_KEY environment variable not set`.

These attempts are not counted as a PASS review.

## Next Action

After commit and push, a separate remote agent should rerun Linux PRECHECK and
then the 2-iteration runtime smoke. This local owner must not start remote
training, SSH, Slurm, `tools/test.py`, mAP evaluation, or any long run.

8-epoch short diagnostic, formal/full training, mAP claims, checkpoint claims,
and paper/deploy claims remain locked until a valid remote PRECHECK and
2-iteration runtime smoke pass.
