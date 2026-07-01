# DIVERGENT MDL-Knot Handoff Speed Fix

Timestamp: 2026-07-01 11:16:42 +08:00 Asia/Shanghai

Route label: `DIVERGENT_INNOVATION_MDL_KNOT_DO_NOT_MERGE_WITH_C3`

Owned worktree: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_MDLKnot_HandoffSpeedFix_Worktree_20260701`

Owned branch: `codex/divergent-mdl-knot-handoff-speedfix-20260701`

## Scope

This stage addresses the slow MDL-Knot short diagnostic handoff path without reducing raw scout coverage. It is local implementation only.

Forbidden actions remained locked: remote sync, Slurm, training, evaluation, `tools/test.py`, Pro/Oracle/Claude/Gemini, mAP/runtime/FLOPs/deploy/paper claims, and C3/BVR/RBA/ABR/combo mixing.

## Root Cause Hypothesis

The selector objective itself is fast. The slow region was likely inflated by two effects:

- profiling grouped raw scout decode, selector, structural handoff, and downstream decode under a broad handoff label;
- fixed-pad MDL handoff writes a 384-frame `frame_inds` tensor whose invalid tail repeats the last selected sparse frame, so plain `mmaction.DecordDecode` can still decode repeated padding requests instead of decoding unique sparse indices once.

This fix does not reduce `mdl_knot_scout_max_frames` or coarse raw scout coverage.

## Fix

- Added sample-local raw scout probe frame cache in `LoadFrames`.
- Added `MDLKnotDecordDecode`, a route-local decoder that:
  - consumes only the final sparse/padded `frame_inds`;
  - decodes unique requested frame indices once;
  - reuses raw scout probe frames when selected sparse indices match probe indices;
  - still outputs the fixed adapter target length with invalid padding masked out;
  - records sparse decode request, unique, cache-hit, and duplicate-avoidance counts.
- Preserved decoder-side `mdl_knot_profile` in metadata and backfilled `mdl_knot_pipeline_diagnostic.profile`
  after sparse decode, so remote short diagnostics can inspect sparse decode time and cache behavior.
- Split profile fields into:
  - `raw_scout_decode_s`;
  - `raw_scout_curve_build_s`;
  - `selector_s`;
  - `structural_handoff_s`;
  - `handoff_validation_s`;
  - `selector_and_structural_handoff_s`;
  - `sparse_decode_s`.
- Updated MDL config and launch/precheck gates to require `MDLKnotDecordDecode`.
- Hardened the short-diagnostic log validator and formal-readiness train-log revalidation against false
  `INTERVAL` route-drift positives from ordinary schedule keys such as `checkpoint_interval`,
  `eval interval`, and `logging interval`, while still rejecting real Interval-route drift text.
- Added focused regression coverage for no scout-budget reduction, raw scout provenance, no dense raw handoff, profile separation, cache/dedup decode behavior, and selected-position/mask preservation.

## Changed Files

- `opentad/datasets/transforms/end_to_end.py`
- `opentad/acquisition/mdl_knot/handoff.py`
- `opentad/acquisition/mdl_knot/diagnostics.py`
- `configs/adatad/thumos/input_mdl_knot_dynamic_adapter_irregular_headv3.py`
- `tools/mdl_knot/validate_mdl_knot_launch_gate.py`
- `tools/mdl_knot/validate_mdl_knot_shortdiag.py`
- `tools/mdl_knot/audit_mdl_knot_pipeline_precheck.py`
- `tests/test_mdl_knot_tools_and_integration.py`
- `tests/test_mdl_knot_shortdiag.py`
- `opentad/datasets/transforms/formatting.py`
- `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`
- `research-wiki/log.md`
- `research-wiki/experiments/DIVERGENT_MDL_KNOT_HANDOFF_SPEEDFIX_20260701.md`

## Verification

- `python -m py_compile opentad/datasets/transforms/end_to_end.py opentad/acquisition/mdl_knot/*.py tools/mdl_knot/*.py`
  - Exit code: `1`.
  - Output: `[Errno 22] Invalid argument: 'opentad/acquisition/mdl_knot/*.py'`.
  - Interpretation: PowerShell did not expand wildcard paths for this Python invocation.
- Expanded equivalent py_compile command:
  - Command: `$files = @('opentad/datasets/transforms/end_to_end.py') + (Get-ChildItem opentad/acquisition/mdl_knot/*.py | ForEach-Object { $_.FullName }) + (Get-ChildItem tools/mdl_knot/*.py | ForEach-Object { $_.FullName }); python -m py_compile @files`
  - Exit code: `0`.
  - Output: no output.
- `python -m pytest tests/test_mdl_knot_core.py tests/test_mdl_knot_tools_and_integration.py tests/test_mdl_knot_shortdiag.py -q`
  - First run exit code: `1`.
  - Failure: local Windows PyTorch DLL import failed with `OSError: [WinError 1114] ... c10.dll`.
  - Fix: added the same torch-import skip guard already used by existing OpenTAD dataset smoke tests.
  - Final run exit code: `0`.
  - Output: `43 passed, 2 skipped in 23.13s`.
- `python tools/mdl_knot/validate_mdl_knot_launch_gate.py --config configs/adatad/thumos/input_mdl_knot_dynamic_adapter_irregular_headv3.py`
  - Exit code: `0`.
  - Output begins: `PRECHECK_ONLY_REQUEST_ALLOWED`.
  - Evidence includes: `"decoder_type": "MDLKnotDecordDecode"`, `"deploy_scout_source": "raw_frame_motion_scout_with_metadata_fallback"`, and all safety flags true.
  - Output ends with locks for remote sync, Slurm, training, evaluation, `tools/test.py`, and mAP/runtime/FLOPs/deploy/paper claims.

## Follow-up Interval Allowlist Revalidation

Timestamp: 2026-07-01 14:37:21 +08:00 Asia/Shanghai

- Scope: local validator/formal-readiness safety fix only; no remote sync, Slurm, training, evaluation, Pro/Oracle/Rosetta, or `tools/test.py`.
- `python -m py_compile opentad/acquisition/mdl_knot/diagnostics.py tools/mdl_knot/validate_mdl_knot_shortdiag.py tests/test_mdl_knot_shortdiag.py`
  - Exit code: `0`.
- `python -m pytest tests/test_mdl_knot_shortdiag.py -q`
  - Exit code: `0`.
  - Output: `17 passed in 3.93s`.
- `python tools/mdl_knot/validate_mdl_knot_shortdiag.py --config configs/adatad/thumos/input_mdl_knot_dynamic_adapter_irregular_headv3_shortdiag.py`
  - Exit code: `0`.
  - Output begins: `SHORT_DIAGNOSTIC_CONFIG_STATIC_CHECK_ALLOWED`.
  - Evidence keeps `validated=false`, `execution_evidence_required_for_formal_readiness=true`, and `formal_train_unlocked=false`.
- `python -m pytest tests/test_mdl_knot_core.py tests/test_mdl_knot_tools_and_integration.py tests/test_mdl_knot_shortdiag.py -q`
  - Exit code: `0`.
  - Output: `47 passed, 2 skipped in 24.75s`.
- `python tools/mdl_knot/validate_mdl_knot_shortdiag.py --config configs/adatad/thumos/input_mdl_knot_dynamic_adapter_irregular_headv3_shortdiag.py --train-log <temp allowed interval log>`
  - Exit code: `0`.
  - Output begins: `SHORT_DIAGNOSTIC_ONLY_REQUEST_ALLOWED`.
  - Evidence keeps `formal_train_unlocked=false`, `full_train_unlocked=false`, `metric_claim=false`, `sparse_compute_claim=false`.
- Added focused regression coverage that `checkpoint/eval/evaluation/val_eval/val_loss/logging interval` schedule text is stripped before route-drift and eval/checkpoint claim checks in both the CLI validator and `validate_shortdiag_train_log_content()`.
- Real `INTERVAL selector route drift` remains rejected in both paths.
- Git push attempt to `pcot-yuzbo` failed once with GitHub port 443 connectivity timeout; no retry was made.
- Status remains `PRECHECK/SHORT_DIAGNOSTIC_ONLY`; formal full training is still locked.

## GitHub Evidence Sync

Timestamp: 2026-07-01 15:34:10 +08:00 Asia/Shanghai

- Retried GitHub synchronization from the route-owned worktree.
- Successfully pushed local commit `493ed2a850e15c67cd0be1dfffe9a1792c239065` to branch `codex/divergent-mdl-knot-handoff-speedfix-493ed2a-20260701`.
- GitHub URL: `https://github.com/yuzbo/pc-ot-mras-r-series-opentad/tree/codex/divergent-mdl-knot-handoff-speedfix-493ed2a-20260701`.
- This sync does not change the claim state: no N16R4 remote sync, Slurm, training, evaluation, `tools/test.py`, mAP/runtime/FLOPs/deploy/paper claim, or sparse-compute claim was produced.
- Next allowed action is bounded remote `SHORT_DIAGNOSTIC_ONLY` resync/rerun only; formal/full training remains locked.

## Claim State

No mAP claim exists.

No runtime or FLOPs claim exists.

No deployment claim exists.

No paper claim exists.

No sparse-compute claim exists.

No formal or full training claim exists.

Formal/full training remains locked.

## Sparse Handoff Short-Window Fix

Timestamp: 2026-07-01 17:40:04 +08:00 Asia/Shanghai

- Scope: local MDL-Knot sparse handoff repair in owned worktree
  `OpenTAD_MDLKnot_SparseHandoffFix_Worktree_20260701` only.
- Route label: `DIVERGENT_INNOVATION_MDL_KNOT_DO_NOT_MERGE_WITH_C3`.
- No remote sync, Slurm, training, evaluation, `tools/test.py`, Pro/Oracle/Rosetta, or C3/CADF/PQR file change was performed.
- Crash reproduced locally with a deterministic short dense window:
  `dense_t=4`, `min_k=4`, `max_k=384`, `target_weighted_error=0.0`, `max_gap=1`.
  Before the fix, selector returned `valid_k=4` and positions `[0, 1, 2, 3]`, so selected-only sparse audit rejected it as dense passthrough.
- Root cause: selector capped selection by `min(max_k, dense_T)`, allowing `valid_k == dense_T` on short real windows. Fixed-pad `frame_inds` then also made it easy for audits to confuse padded detector input length with valid selected sparse length.
- Fix:
  - `opentad/acquisition/mdl_knot/selector.py`: effective sparse cap is now `min(max_k, dense_T - 1)` and `dense_T < 3` fails closed because it cannot preserve two endpoints while remaining sparse.
  - `opentad/acquisition/mdl_knot/validators.py`: selected-only audit now explicitly rejects `valid_k >= dense_T` with a clear diagnostic.
  - `tests/test_mdl_knot_core.py` and `tests/test_mdl_knot_tools_and_integration.py`: added regression coverage for short-window cap, dense passthrough/`valid_k >= dense_T`, fixed-pad length confusion, route-label drift, mask/meta alignment, and selected-only audit slicing.
- Verification:
  - Red tests before fix:
    - `python -m pytest tests/test_mdl_knot_core.py::test_selector_keeps_short_windows_strictly_sparse_under_large_cap -q`
      failed with `assert 4 < 4`.
    - `python -m pytest tests/test_mdl_knot_tools_and_integration.py::test_mdl_knot_handoff_caps_short_windows_before_sparse_audit -q`
      failed in `validate_real_sparse_handoff` with dense-passthrough rejection.
  - After fix:
    - `python -m pytest tests/test_mdl_knot_core.py::test_selector_keeps_short_windows_strictly_sparse_under_large_cap tests/test_mdl_knot_core.py::test_real_sparse_handoff_requires_gathered_inputs_and_valid_mask -q`
      exit code `0`, `2 passed in 0.26s`.
    - `python -m pytest tests/test_mdl_knot_tools_and_integration.py::test_fixed_adapter_bridge_pads_frame_inds_without_counting_padding_as_valid tests/test_mdl_knot_tools_and_integration.py::test_mdl_knot_handoff_caps_short_windows_before_sparse_audit -q`
      exit code `0`, `2 passed in 0.97s`.
    - `python -m pytest tests/test_mdl_knot_core.py tests/test_mdl_knot_tools_and_integration.py tests/test_mdl_knot_shortdiag.py -q`
      exit code `0`, `49 passed, 2 skipped in 25.12s`.
    - `python tools/mdl_knot/audit_mdl_knot_pipeline_precheck.py --out-dir tmp_mdl_knot_precheck_fix --overwrite`
      exit code `0`, produced `VALIDATED_PRECHECK_SUMMARY=tmp_mdl_knot_precheck_fix\mdl_knot_precheck_summary.json`; the temporary output directory was then removed after path verification inside the owned worktree.
- Claim state is unchanged: no mAP, runtime, FLOPs, deployment, paper, sparse-compute, formal-training, or full-training claim exists.
- Next allowed action after required review/permission is bounded local/static review or explicitly authorized short diagnostic resync/rerun. Formal/full training remains locked.
