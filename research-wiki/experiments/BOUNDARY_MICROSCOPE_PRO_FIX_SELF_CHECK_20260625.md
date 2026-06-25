# Boundary Microscope Pro Fix Self-Check 2026-06-25

## Status

- Timestamp: `2026-06-25T13:03:16+08:00`.
- Route label: `DIVERGENT_INNOVATION_BOUNDARY_MICROSCOPE_DO_NOT_MERGE_WITH_C3`.
- Owner worktree: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_BoundaryMicroscope_ProFix_Worktree_20260625`.
- Owner branch: `codex/boundary-microscope-pro-fix-20260625`.
- Decision status: `LOCAL_PRECHECK_GATE_BOUND_ENTRYPOINT_FIX_READY_FOR_COMMIT`.
- Remote failure addressed: remote bundle commit `b6ce770` failed `PRECHECK_ONLY` before active manifest generation because `scripts/run_boundary_microscope_precheck_n16r4.sbatch` still asserted `allowed_entrypoints == ()` after the full-train candidate config had become gate-bound train-only with `allowed_entrypoints=("tools/train.py",)`.
- Fix summary: both Boundary N16R4 launchers now audit the gate-bound train-only config as `allow_tools_train=True`, `allowed_entrypoints=("tools/train.py",)`, and `requires_entrypoint_gate=True`, while retaining `allow_tools_test=False`, raw-prediction disabled, metric/paper claims disabled, and `PRECHECK_ONLY=1` exit-before-train behavior.
- Remote sync, push, Slurm, `tools/train.py`, `tools/test.py`, mAP, runtime/FLOPs, metric claim, and paper claim: not performed and still outside this local code-owner action.
- Tracker/log note: `research-wiki/log.md` and `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md` are absent in this owned worktree, so no shared-main-worktree tracker/log files were touched.

## 2026-06-25 Gate-Bound Entrypoint PRECHECK Fix

- Changed files:
  - `scripts/run_boundary_microscope_precheck_n16r4.sbatch`
  - `scripts/run_boundary_microscope_full_train_n16r4.sbatch`
  - `tests/test_boundary_microscope_config_gate.py`
  - `research-wiki/experiments/BOUNDARY_MICROSCOPE_PRO_FIX_SELF_CHECK_20260625.md`
- Root cause: the resolved-config audit had contradictory assertions for the same full-train candidate config, first requiring `("tools/train.py",)` and then requiring `()`.
- Precheck-only safety: `PRECHECK_ONLY=1` still rejects `ALLOW_BOUNDARY_MICROSCOPE_FULL_TRAIN=1`, writes `BOUNDARY_MICROSCOPE_PRECHECK_ONLY_PASS_NO_TRAIN`, and exits before full-train gate JSON checks or any `tools/train.py` execution.
- Train-only safety: when a future full-train launcher is used, the resolved-config audit permits only `tools/train.py`, requires the entrypoint gate, and still rejects `tools/test.py`, raw prediction/cache shortcuts, metric claims, and paper claims.
- Initial RED test:
  - `python -m pytest tests/test_boundary_microscope_config_gate.py::test_boundary_microscope_precheck_launcher_accepts_gate_bound_train_only_config_audit tests/test_boundary_microscope_config_gate.py::test_boundary_microscope_n16r4_full_train_launcher_is_locked_by_default_and_runs_train_after_gate -q`
  - Result before implementation: `2 failed`; both failures showed the launcher audit did not yet expose the required gate-bound entrypoint contract and still had the stale empty-entrypoint assertion path.
- Focused GREEN test:
  - `python -m pytest tests/test_boundary_microscope_config_gate.py::test_boundary_microscope_precheck_launcher_accepts_gate_bound_train_only_config_audit tests/test_boundary_microscope_config_gate.py::test_boundary_microscope_n16r4_full_train_launcher_is_locked_by_default_and_runs_train_after_gate -q`
  - Result after implementation: `2 passed in 0.02s`.
- Final focused config/gate test:
  - `python -m pytest tests/test_boundary_microscope_config_gate.py -q -rs`
  - Result: `16 passed in 2.12s`.
- Static compile:
  - `python -m py_compile tools/bata/validate_boundary_microscope_gate.py configs/adatad/thumos/boundary_microscope_acquisition_local_precheck.py configs/adatad/thumos/boundary_microscope_acquisition_full_train_candidate_n16r4.py tests/test_boundary_microscope_config_gate.py opentad/utils/training_guard.py opentad/models/selectors/boundary_microscope_acquisition_route.py`
  - Result: pass.
- Launcher syntax:
  - `Get-Content -Raw scripts/run_boundary_microscope_precheck_n16r4.sbatch | ForEach-Object { $_ -replace "\`r", "" } | bash -n -s`
  - `Get-Content -Raw scripts/run_boundary_microscope_full_train_n16r4.sbatch | ForEach-Object { $_ -replace "\`r", "" } | bash -n -s`
  - Result: both pass.
- Diff hygiene:
  - `git diff --check`
  - Result: pass; only Windows LF-to-CRLF warnings.

- Timestamp: `2026-06-25T12:45:44+08:00`.
- Route label: `DIVERGENT_INNOVATION_BOUNDARY_MICROSCOPE_DO_NOT_MERGE_WITH_C3`.
- Owner worktree: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_BoundaryMicroscope_ProFix_Worktree_20260625`.
- Owner branch: `codex/boundary-microscope-pro-fix-20260625`.
- Decision status: `PRO_FIX_LOCAL_COMMIT_READY_FOR_PRECHECK_ONLY_DEPLOYMENT`.
- Remote sync, push, Slurm, `tools/test.py`, mAP, runtime/FLOPs, metric claim, and paper claim: not performed and still outside this local code-owner action.
- Full train: no longer blocked by a stale `training_guard` double gate after a valid external full-train gate, but still requires explicit external gate JSON/SHA, active manifest SHA, resolved config SHA, matching `RUN_TAG`, and user override statement before `tools/train.py` can start.

## Boundary Pro Findings Addressed

- Fixed the selector package import/registry blocker by removing unconditional imports of Event-Surprise and PC-OT/C3 selector classes from the Boundary branch package surface. Boundary can now register from its own selector file without depending on other divergent route files.
- Enabled the Boundary route to use the ActionFormer physical temporal-grid head path through config inheritance. The head receives `irregular_selected_positions` as original dense/window coordinates and marks `irregular_native_axis=True` before seconds conversion.
- Added an executable coordinate roundtrip test for a start microscope packet: selected dense positions include the synthetic boundary start/end, physical-grid head points use those dense positions, a proposal roundtrips to `[8, 22]` dense time, and post-processing converts it to seconds via the original window metadata.
- Updated both Boundary N16R4 launchers so `EXPECTED_GIT_BRANCH` defaults to `codex/boundary-microscope-pro-fix-20260625`, not the old `codex/boundary-microscope-precheck-deploy-20260624`.
- Converted the full-train candidate config from stale hard-blocked `allow_tools_train=False` to gate-bound train-only: config validation now reports `allows_tools_train=true`, `requires_entrypoint_gate=true`, `allows_slurm=false`, `allows_gpu=false`, and `allows_full_train=false`.
- The full-train launcher now exports `OPENTAD_BOUNDARY_MICROSCOPE_ENTRYPOINT_GATE_JSON`, `OPENTAD_BOUNDARY_MICROSCOPE_ENTRYPOINT_GATE_SHA256`, `OPENTAD_BOUNDARY_MICROSCOPE_ACTIVE_MANIFEST_SHA256`, and `OPENTAD_BOUNDARY_MICROSCOPE_RESOLVED_CONFIG_SHA256` immediately before `tools/train.py`, so `training_guard` verifies the same external gate that the launcher validated.

## Changed Files

- `opentad/models/selectors/__init__.py`
- `opentad/models/selectors/boundary_microscope_acquisition_route.py`
- `configs/adatad/thumos/boundary_microscope_acquisition_local_precheck.py`
- `configs/adatad/thumos/boundary_microscope_acquisition_full_train_candidate_n16r4.py`
- `scripts/run_boundary_microscope_precheck_n16r4.sbatch`
- `scripts/run_boundary_microscope_full_train_n16r4.sbatch`
- `tools/bata/validate_boundary_microscope_gate.py`
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
- GT/teacher/cache leakage risk: full-train gate validation and `training_guard` tests require `test_time_gt_allowed=False`, `teacher_allowed=False`, `raw_prediction_cache_allowed=False`, `metric_claim_allowed=False`, and `paper_claim_allowed=False`.
- C3 mixing risk: Boundary package export and configs are route-isolated from Event-Surprise, PC-OT/C3 reader/selector, Frame-Token, BH-SDC, and combo attribution.
- `training_guard` double-layer risk: fixed without changing the generic guard. The Boundary full-train config now uses the existing entrypoint-gate mechanism; without env-bound gate evidence `tools/train.py` is rejected, with valid gate evidence only `tools/train.py` is allowed, and `tools/test.py` remains rejected.

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

Historical initial self-check result before the launcher gate-readiness fix:
`BOUNDARY_MICROSCOPE_CONFIG_GATE_VALIDATION_PASS`, with `allows_tools_train=false`, `allows_slurm=false`, `allows_full_train=false`.

Additional 2026-06-25 pro-fix verification:

```powershell
python -m pytest tests/test_boundary_microscope_config_gate.py -q -rs
```

Initial RED result before implementation: `6 failed, 9 passed in 1.80s`.
Expected failures covered the old launcher branch defaults and the stale `allow_tools_train=False` second-layer `training_guard` block.

Final result after implementation: `15 passed in 1.55s`.

```powershell
python -m py_compile tools/bata/validate_boundary_microscope_gate.py configs/adatad/thumos/boundary_microscope_acquisition_local_precheck.py configs/adatad/thumos/boundary_microscope_acquisition_full_train_candidate_n16r4.py tests/test_boundary_microscope_config_gate.py opentad/utils/training_guard.py opentad/models/selectors/boundary_microscope_acquisition_route.py
```

Result: pass.

```powershell
Get-Content -Raw scripts/run_boundary_microscope_precheck_n16r4.sbatch | ForEach-Object { $_ -replace "`r", "" } | bash -n -s
Get-Content -Raw scripts/run_boundary_microscope_full_train_n16r4.sbatch | ForEach-Object { $_ -replace "`r", "" } | bash -n -s
```

Result: both pass.

```powershell
python tools/bata/validate_boundary_microscope_gate.py configs/adatad/thumos/boundary_microscope_acquisition_full_train_candidate_n16r4.py --json
```

Result: `BOUNDARY_MICROSCOPE_CONFIG_GATE_VALIDATION_PASS`, with `allowed_decision=ALLOW_BOUNDARY_MICROSCOPE_FULL_TRAIN`, `allows_tools_train=true`, `requires_entrypoint_gate=true`, `allows_slurm=false`, `allows_gpu=false`, `allows_full_train=false`, `metric_claim_allowed=false`, and `paper_claim_allowed=false`.

```powershell
python tools/bata/validate_boundary_microscope_gate.py --action full-train --gate-json <temp> --gate-sha256 <temp_sha> --active-manifest-sha256 manifest-sha --resolved-config-sha256 resolved-sha --run-tag boundary_microscope_full_train_fixed_run_tag --json
```

Result: `BOUNDARY_MICROSCOPE_FULL_TRAIN_GATE_VALIDATION_PASS`, with `allows_tools_train=true`, `allows_slurm=true`, `allows_gpu=true`, `allows_full_train=true`, `metric_claim_allowed=false`, and `paper_claim_allowed=false`.

```powershell
git diff --check
```

Result: pass; only Windows LF-to-CRLF warnings.

```powershell
git diff --check
```

Result: pass; only Windows LF-to-CRLF warnings.

## Still Locked

- No Pro/Gemini/Claude review was launched in this action by user instruction.
- No remote sync, SCP/rsync, push, Slurm, GPU allocation, long training, `tools/train.py`, `tools/test.py`, mAP, runtime/FLOPs, deployment, metric claim, or paper claim was executed by this local fix.
- Main process may use this committed pro-fix branch for remote `PRECHECK_ONLY=1` deployment checks after deciding to sync/push externally; full train still additionally requires the external full-train gate JSON/SHA/run_tag/user authorization bundle.
