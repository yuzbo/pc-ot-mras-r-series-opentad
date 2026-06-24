# Boundary Microscope Pro Fix Self-Check 2026-06-25

## Status

- Timestamp: `2026-06-25T00:51:58+08:00`.
- Route label: `DIVERGENT_INNOVATION_BOUNDARY_MICROSCOPE_DO_NOT_MERGE_WITH_C3`.
- Owner worktree: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_BoundaryMicroscope_ProFix_Worktree_20260625`.
- Owner branch: `codex/boundary-microscope-pro-fix-20260625`.
- Decision status: `FIXED_FOR_LOCAL_SMOKE_PENDING_FOLLOWUP_PRO`.
- Full-train / Slurm / remote sync / `tools/test.py` mAP: still locked.

## Boundary Pro Findings Addressed

- Fixed the selector package import/registry blocker by removing unconditional imports of Event-Surprise and PC-OT/C3 selector classes from the Boundary branch package surface. Boundary can now register from its own selector file without depending on other divergent route files.
- Enabled the Boundary route to use the ActionFormer physical temporal-grid head path through config inheritance. The head receives `irregular_selected_positions` as original dense/window coordinates and marks `irregular_native_axis=True` before seconds conversion.
- Added an executable coordinate roundtrip test for a start microscope packet: selected dense positions include the synthetic boundary start/end, physical-grid head points use those dense positions, a proposal roundtrips to `[8, 22]` dense time, and post-processing converts it to seconds via the original window metadata.
- Kept the full-train candidate fail-closed. Gate validation reports `allows_tools_train=false`, `allows_slurm=false`, `allows_full_train=false`, and `future_full_train_requires_separate_decision=true`.

## Changed Files

- `opentad/models/selectors/__init__.py`
- `opentad/models/selectors/boundary_microscope_acquisition_route.py`
- `configs/adatad/thumos/boundary_microscope_acquisition_local_precheck.py`
- `tests/test_boundary_microscope_registry_build.py`
- `tests/test_boundary_microscope_config_gate.py`
- `tests/test_boundary_microscope_coordinate_roundtrip.py`
- `research-wiki/experiments/BOUNDARY_MICROSCOPE_PRO_FIX_SELF_CHECK_20260625.md`

## Contract Check

- Input sampling changed: yes, Boundary selector remains pre-backbone raw-frame sparse acquisition with microscope packets around deploy-visible start/end hazards.
- Dynamic budget policy changed: no new dynamic controller; target remains fixed local smoke/fail-closed candidate budget.
- Token compression changed: no.
- Adapter/backbone internals changed: no.
- Detector head logic changed: no production code change; config now explicitly activates existing `physical_grid_actionformer` temporal-grid behavior.
- Losses/assignment changed: no.
- Test-time post-processing changed: no production code change; existing `irregular_native_axis=True` path is now exercised by the physical-grid head and tested.
- Strict random-fixed 50% status: Boundary route remains an up-to-384-from-768 sparse acquisition candidate, not a claim of final dynamic-budget success.
- GT/teacher/cache leakage risk: tests cover no GT/teacher/raw-prediction/cache metadata in selector plans and forward-test shortcut rejection.
- C3 mixing risk: Boundary package export and configs are route-isolated from Event-Surprise, PC-OT/C3 reader/selector, Frame-Token, BH-SDC, and combo attribution.

## Verification

All verification used the local working Torch environment:
`C:\Users\skywalker\.conda\envs\torch_1\python.exe` (`torch 2.3.0+cu121`).
The default Anaconda base Python still has the known local `torch\lib\c10.dll` initialization failure and was not used for Torch smoke evidence.

Commands and results:

```powershell
C:\Users\skywalker\.conda\envs\torch_1\python.exe -m pytest tests/test_boundary_microscope_registry_build.py::test_boundary_microscope_selector_package_import_is_route_isolated tests/test_boundary_microscope_coordinate_roundtrip.py::test_boundary_microscope_physical_grid_head_uses_selected_dense_positions_for_roundtrip tests/test_boundary_microscope_config_gate.py::test_boundary_microscope_local_config_is_parseable_and_fail_closed -q
```

Result: `3 passed in 3.79s`.

```powershell
C:\Users\skywalker\.conda\envs\torch_1\python.exe -m pytest tests\test_boundary_microscope_acquisition_route.py tests\test_boundary_microscope_actionformer_integration.py tests\test_boundary_microscope_registry_build.py tests\test_boundary_microscope_config_gate.py tests\test_boundary_microscope_coordinate_roundtrip.py -q -rs
```

Result: `26 passed, 1 skipped in 26.91s`.
The single skip is the existing canonical OpenTAD registry build test requiring unavailable local mmaction/mmcv runtime dependencies; the synthetic no-data ActionFormer forward smoke and coordinate roundtrip tests passed.

```powershell
C:\Users\skywalker\.conda\envs\torch_1\python.exe -m py_compile opentad\models\selectors\boundary_microscope_acquisition_route.py opentad\models\selectors\__init__.py opentad\models\detectors\actionformer.py opentad\models\dense_heads\anchor_free_head.py configs\adatad\thumos\boundary_microscope_acquisition_local_precheck.py configs\adatad\thumos\boundary_microscope_acquisition_full_train_candidate_n16r4.py tests\test_boundary_microscope_acquisition_route.py tests\test_boundary_microscope_actionformer_integration.py tests\test_boundary_microscope_registry_build.py tests\test_boundary_microscope_config_gate.py tests\test_boundary_microscope_coordinate_roundtrip.py
```

Result: pass.

```powershell
C:\Users\skywalker\.conda\envs\torch_1\python.exe tools\bata\validate_boundary_microscope_gate.py configs\adatad\thumos\boundary_microscope_acquisition_full_train_candidate_n16r4.py --json
```

Result: `BOUNDARY_MICROSCOPE_CONFIG_GATE_VALIDATION_PASS`, with `allows_tools_train=false`, `allows_slurm=false`, `allows_full_train=false`.

```powershell
git diff --check
```

Result: pass; only Windows LF-to-CRLF warnings.

## Still Locked

- Follow-up Pro review is still pending.
- No remote sync, SCP/rsync, Slurm, GPU allocation, long training, `tools/train.py`, `tools/test.py`, mAP, runtime/FLOPs, deployment, metric claim, or paper claim is unlocked by this local fix.
