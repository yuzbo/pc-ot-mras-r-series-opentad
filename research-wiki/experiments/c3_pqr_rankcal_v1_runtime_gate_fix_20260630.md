# C3 PQR RankCal V1 Runtime Gate Fix - 2026-06-30

Timestamp: 2026-06-30 03:48:16 +08:00.

Route: `C3_MAINLINE_OPTIMIZATION` / `C3_ORIGINAL_OPTIMIZATION_ROUTE`.

Scope: local runtime gate implementation only. No SSH, Slurm, remote smoke,
`tools/train.py` real training run, `tools/test.py`, checkpoint claim, metric
claim, or N16R4 long-held allocation action was performed.

## Pro Blocker Fixed

Valid GPT-5.5 Pro verdict was `FIX_BEFORE_RUNTIME` because
`workflow.max_train_iters=2` was present in the precheck config but was not
consumed by the standard training runtime path.

This local fix makes the field executable:

- `tools/train.py` reads `cfg.workflow.max_train_iters`, normalizes values
  `None` or `<=0` to default-off behavior, tracks completed train iterations
  across the full run, and passes the remaining iteration budget to
  `train_one_epoch`.
- `opentad/cores/train_engine.py` accepts optional `max_train_iters`, hard
  stops the epoch loop after that many attempted train batches, logs the gate,
  and returns completed iteration count to the launcher.
- When the launcher reaches the configured global run limit, it logs the event
  and exits the train loop before checkpoint, validation loss, or evaluation.
- `tools/validate_c3_pqr_rankcal_v1_config.py` now fail-closed checks that both
  standard launcher and train loop consume the runtime gate.
- `tests/test_c3_pqr_rankcal_v1_config.py` includes fake-runtime tests proving
  `max_train_iters=2` executes exactly 2 iterations and default-off executes the
  full fake epoch.

## Semantics

- Unset `workflow.max_train_iters`: unchanged full epoch behavior.
- `workflow.max_train_iters <= 0`: unchanged full epoch behavior.
- `workflow.max_train_iters = N > 0`: each process executes at most `N`
  attempted train iterations for the whole run. The final gated epoch exits
  before checkpoint, validation, or evaluation branches.
- This is a runtime smoke limiter only. It does not alter PQR ranking logic,
  input sampling, CADF selector logic, BH-SDC, evaluator, post-processing,
  losses/assignment, or detector inference behavior.

## Local Verification

Commands run from
`E:\DeskTop\TAD\temrefuse-tad\OpenTAD_C3PQRRankCal_Worktree_20260629`:

```powershell
$env:PYTHONPATH = (Get-Location).Path
pytest -q tests/test_c3_pqr_rankcal_v1_config.py -k "max_train_iters or hard_stops or full_epoch"
```

Result: `3 passed, 12 deselected, 1 warning`.

```powershell
$env:PYTHONPATH = (Get-Location).Path
pytest -q tests/test_c3_pqr_rankcal_v1_config.py tests/test_c3_pqr_rankcal_v1_quality_head.py
```

Result: `16 passed, 3 skipped, 1 warning`. The 3 skipped tests are the existing
torch-backed quality-head behavior tests skipped because local Windows torch DLL
initialization fails with `[WinError 1114] ... c10.dll`.

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m py_compile tools/train.py opentad/cores/train_engine.py tools/validate_c3_pqr_rankcal_v1_config.py tests/test_c3_pqr_rankcal_v1_config.py
```

Result: pass, exit code 0.

```powershell
$env:PYTHONPATH = (Get-Location).Path
python tools/validate_c3_pqr_rankcal_v1_config.py configs/adatad/thumos/c3_indirect_original_adatad_32px_a_pqr_rankcal_v1_precheck.py
python tools/validate_c3_pqr_rankcal_v1_config.py configs/adatad/thumos/c3_indirect_original_adatad_32px_a_pqr_rankcal_v1_shortdiag.py
python tools/validate_c3_pqr_rankcal_v1_config.py configs/adatad/thumos/c3_indirect_original_adatad_32px_a_exact_uniform_backend_control_pqr_rankcal_v1_shortdiag.py
```

Result: all three print `PASS_C3_PQR_RANKCAL_V1_CONFIG`.

## Gate State After Fix

Local implementation-level Pro blocker is fixed. Still locked before launch:

- required read-only subagent final review;
- remote PRECHECK rerun in a working torch/Linux environment;
- remote 2-iteration runtime smoke after review and PRECHECK pass;
- 8-epoch short diagnostic;
- formal/full training;
- `tools/test.py`, metric claims, checkpoint claims, paper/deploy claims.

Current mAP evidence: none.

