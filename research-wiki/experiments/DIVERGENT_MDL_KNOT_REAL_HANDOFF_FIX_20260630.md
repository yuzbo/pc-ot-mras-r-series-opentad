# DIVERGENT MDL-Knot True Sparse Handoff Fix

Timestamp: 2026-06-30 18:41:45 +08:00 Asia/Shanghai

Route label: `DIVERGENT_INNOVATION_MDL_KNOT_DO_NOT_MERGE_WITH_C3`

Owned worktree: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_MDLKnot_RealDiag_Worktree_20260630`

Owned branch: `codex/divergent-mdl-knot-realdiag-20260630`

## Failure Evidence

User-provided N16R4 evidence: short diagnostic child `1118197.488 mdl_shortdiag_g0` failed with exit `1:0`. It was not an OOM. Loss stayed finite, with two non-finite gradient skip events before the fatal blocker.

Fatal blocker:

```text
ValueError: selected_inputs must be a real sparse gather, not dense passthrough
```

Stack:

```text
opentad/datasets/transforms/end_to_end.py
  -> apply_mdl_knot_to_dense_window
  -> validate_real_sparse_handoff
```

## Root Cause

`apply_mdl_knot_to_dense_window` selected sparse frame indices correctly for `frame_inds`, but then audited the handoff with `selected_inputs=frame_inds[:valid_k]` and `dense_inputs=list(dense_window)`. That made the validator inspect frame-index lists instead of gathered raw frame samples.

This was a contract bug in the MDL input-side transform/handoff boundary. The detector still received sparse `frame_inds`, but the fail-closed validator correctly rejected the audit because it was not a true raw-frame gather.

## Fix

Changed surface:

- Input sampling / handoff: yes.
- Dynamic budget policy: unchanged selection policy, but dynamic `valid_k` handoff is now audited correctly.
- Token compression: no.
- Adapter/backbone internals: no.
- Detector head logic: no.
- Loss/assignment: no.
- Test-time post-processing/evaluator: no.

Implementation summary:

- `opentad/acquisition/mdl_knot/handoff.py`
  - `apply_mdl_knot_to_dense_window` now requires `dense_inputs` for true handoff validation.
  - It gathers `selected_inputs = dense_inputs[selected_positions]` and passes true raw samples plus original dense samples to `validate_real_sparse_handoff`.
  - It records only small audit metadata in `results`; raw dense audit inputs are not retained for the backbone.
- `opentad/acquisition/mdl_knot/validators.py`
  - Validator still rejects dense passthrough and identity-object handoff.
  - Validator now also rejects frame-index inputs and verifies selected raw samples equal `dense_inputs` gathered at ledger `selected_positions`.
  - Length mismatch remains fail-closed.
- `opentad/datasets/transforms/end_to_end.py`
  - MDL `LoadFrames` now reads a dense raw audit window from the Decord/video reader before `DecordDecode`, passes it into the handoff validator, then keeps only sparse `frame_inds` for downstream decode.
- `tools/mdl_knot/audit_mdl_knot_pipeline_precheck.py`
  - Synthetic precheck now uses tiny raw-frame arrays for handoff validation rather than integer frame IDs.
- `tools/mdl_knot/collect_mdl_knot_real_video_diagnostics.py`
  - Torch-free equivalent loader now also requires a reader and true dense raw inputs for handoff validation.

## Validator Safety

The validator was not weakened. It is stricter than before:

- dense passthrough remains rejected;
- identity object remains rejected;
- length mismatch remains rejected;
- scalar/index lists are now rejected as not raw frame/tensor samples;
- wrong gathered values are now rejected against `ledger.selected_positions`.

The `no_dense_raw_backbone_handoff` safety flag remains meaningful: dense raw frames are used only for fail-closed audit, not retained in `results` as detector inputs.

## Local Verification

Commands run in the owned worktree:

```powershell
python -m pytest tests/test_mdl_knot_core.py tests/test_mdl_knot_tools_and_integration.py tests/test_mdl_knot_realdiag.py tests/test_mdl_knot_shortdiag.py -q
```

Result: `51 passed, 1 skipped in 74.73s`.

```powershell
python -m py_compile opentad/acquisition/mdl_knot/handoff.py opentad/acquisition/mdl_knot/validators.py opentad/acquisition/mdl_knot/types.py opentad/acquisition/mdl_knot/scout.py opentad/datasets/transforms/end_to_end.py tools/mdl_knot/audit_mdl_knot_pipeline_precheck.py tools/mdl_knot/collect_mdl_knot_real_video_diagnostics.py tools/mdl_knot/validate_mdl_knot_launch_gate.py tools/mdl_knot/validate_mdl_knot_shortdiag.py configs/adatad/thumos/input_mdl_knot_dynamic_adapter_irregular_headv3.py configs/adatad/thumos/input_mdl_knot_dynamic_adapter_irregular_headv3_shortdiag.py
```

Result: exit code `0`.

```powershell
python tools/mdl_knot/audit_mdl_knot_pipeline_precheck.py --out-dir logs/mdl_knot_real_handoff_precheck_20260630 --overwrite
```

Result: exit code `0`, `VALIDATED_PRECHECK_SUMMARY=logs\mdl_knot_real_handoff_precheck_20260630\mdl_knot_precheck_summary.json`. The generated ignored local output directory was removed after verification to keep the allowed write scope clean.

```powershell
python tools/mdl_knot/validate_mdl_knot_launch_gate.py --config configs/adatad/thumos/input_mdl_knot_dynamic_adapter_irregular_headv3.py
```

Result: exit code `0`, `PRECHECK_ONLY_REQUEST_ALLOWED`. Still locked: remote sync, Slurm, training, evaluation, `tools/test.py`, mAP/runtime/FLOPs/deploy/paper claims.

```powershell
python tools/mdl_knot/validate_mdl_knot_shortdiag.py --config configs/adatad/thumos/input_mdl_knot_dynamic_adapter_irregular_headv3_shortdiag.py
```

Result: exit code `0`, `SHORT_DIAGNOSTIC_CONFIG_STATIC_CHECK_ALLOWED`. Still locked: full training, evaluation, checkpoints, `tools/test.py`, mAP and sparse-compute claims.

```powershell
python tools/mdl_knot/validate_mdl_knot_launch_gate.py --config configs/adatad/thumos/input_mdl_knot_dynamic_adapter_irregular_headv3.py --precheck-summary logs/mdl_knot_real_handoff_precheck_20260630/mdl_knot_precheck_summary.json
```

Result: exit code `0`, `PRECHECK_ONLY_REQUEST_ALLOWED`.

```powershell
git diff --check
```

Result: exit code `0`; only Windows LF/CRLF working-copy warnings, no whitespace errors.

One extra collector command was attempted with `--shortdiag-log logs/mdl_knot_shortdiag_verification_20260630.txt`; it correctly failed because that verification text contains a forbidden `tools/test.py` marker and is not a shortdiag train log. This was not accepted as a route failure or success signal.

## Next Action

After commit/push and required review acceptance, the technical blocker for true sparse raw-frame handoff is fixed for local `PRECHECK_ONLY` and bounded `SHORT_DIAGNOSTIC_ONLY` rerun consideration.

No remote sync, Slurm, GPU rerun, training, evaluation, `tools/test.py`, mAP claim, runtime/FLOPs claim, deployment claim, paper claim, or parent-hold action was performed by this fix.

## 2026-06-30 Follow-up Confirmation After Remote Failure Evidence

Additional user-provided N16R4 evidence:

- Remote clone: `/data/home/sczc063/run/yuzibo/OpenTAD_MDLKnot_RealDiag_20260630_fd2977f`
- Remote HEAD: `5ef35d2698742db9a90f670b3f7e2926214a8e2c`
- Old shortdiag run directory: `logs/mdl_knot_shortdiag_gpu0_r7_20260630_172858_+0800`
- Old child run: `1118197.488 mdl_shortdiag_g0`
- Runtime before failure: `58m51`
- Failure: `ValueError: selected_inputs must be a real sparse gather, not dense passthrough`
- Stability warning before the fatal handoff failure: non-finite gradients were reported and optimizer steps skipped at epoch 0 iterations 20 and 36 on `module.rpn_head.reg_head.weight`.
- Rehandfix launch directory `logs/mdl_knot_shortdiag_gpu0_rehandfix_launch_20260630_185712_+0800` contains only `launcher_srun.log`; no `srun_stdout_stderr.log` or train log exists, so it is not valid post-fix shortdiag execution evidence.

Interpretation:

- The old failed run is accepted as evidence that the pre-`5ef35d2` handoff audit was correctly fail-closed against dense passthrough.
- It is not evidence for final MDL-Knot route quality, mAP, runtime, sparse compute, or deployability.
- The non-finite gradient skip events are a stability diagnostic risk for the next bounded shortdiag rerun. They are not route-level success/failure evidence unless repeated after the handoff fix or escalated to NaN/OOM/protocol failure.

Current local code inspection confirms that `5ef35d2` fixed the dense-passthrough bug:

- `apply_mdl_knot_to_dense_window` now raises if `dense_inputs` is missing.
- It gathers `selected_inputs` from true dense raw samples at `ledger.selected_positions`.
- `validate_real_sparse_handoff` still rejects dense passthrough, frame-index inputs, and forged raw values.
- `mdl_knot_sparse_meta` now also records the selected detector-frame prefix and handoff audit, so the later `Collect` metadata can prove sparse `frame_inds` and raw gather audit reached detector metadata.

Additional local verification in the owned worktree after the metadata/audit hardening:

```powershell
python -m py_compile opentad/acquisition/mdl_knot/handoff.py opentad/acquisition/mdl_knot/diagnostics.py tools/mdl_knot/collect_mdl_knot_real_video_diagnostics.py tools/mdl_knot/validate_mdl_knot_launch_gate.py tools/mdl_knot/validate_mdl_knot_shortdiag.py tests/test_mdl_knot_core.py tests/test_mdl_knot_tools_and_integration.py tests/test_mdl_knot_realdiag.py
```

Result: exit code `0`.

```powershell
python -m pytest tests/test_mdl_knot_core.py tests/test_mdl_knot_tools_and_integration.py tests/test_mdl_knot_realdiag.py tests/test_mdl_knot_shortdiag.py -q
```

Result: `52 passed, 1 skipped in 100.78s`.

```powershell
python tools/mdl_knot/validate_mdl_knot_launch_gate.py --config configs/adatad/thumos/input_mdl_knot_dynamic_adapter_irregular_headv3.py
```

Result: exit code `0`, `PRECHECK_ONLY_REQUEST_ALLOWED`; all remote sync, Slurm, training, evaluation, `tools/test.py`, mAP/runtime/FLOPs/deploy/paper locks preserved.

```powershell
python tools/mdl_knot/validate_mdl_knot_shortdiag.py --config configs/adatad/thumos/input_mdl_knot_dynamic_adapter_irregular_headv3_shortdiag.py
```

Result: exit code `0`, `SHORT_DIAGNOSTIC_CONFIG_STATIC_CHECK_ALLOWED`; `validated=false`, no train-log evidence yet.

```powershell
$out = Join-Path $env:TEMP 'mdl_knot_realdiag_fixture_summary_20260630.json'
python tools/mdl_knot/collect_mdl_knot_real_video_diagnostics.py --config configs/adatad/thumos/input_mdl_knot_dynamic_adapter_irregular_headv3.py --out $out --dry-run-fixture --window-count 4
python tools/mdl_knot/validate_mdl_knot_launch_gate.py --config configs/adatad/thumos/input_mdl_knot_dynamic_adapter_irregular_headv3.py --formal-readiness-summary $out
```

Result: collector succeeded; formal gate correctly returned `LOCKED` because fixture-only schema evidence is not real-video formal readiness evidence.

```powershell
$out = Join-Path $env:TEMP 'mdl_knot_real_handoff_precheck_20260630'
python tools/mdl_knot/audit_mdl_knot_pipeline_precheck.py --out-dir $out --overwrite
python tools/mdl_knot/validate_mdl_knot_launch_gate.py --config configs/adatad/thumos/input_mdl_knot_dynamic_adapter_irregular_headv3.py --precheck-summary (Join-Path $out 'mdl_knot_precheck_summary.json')
```

Result: exit code `0`, `PRECHECK_ONLY_REQUEST_ALLOWED`.

Bounded shortdiag rerun commands prepared for an already allocated child GPU context only:

```bash
cd /data/home/sczc063/run/yuzibo/OpenTAD_MDLKnot_RealDiag_20260630_fd2977f
git rev-parse HEAD
python tools/mdl_knot/validate_mdl_knot_launch_gate.py --config configs/adatad/thumos/input_mdl_knot_dynamic_adapter_irregular_headv3.py
python tools/mdl_knot/validate_mdl_knot_shortdiag.py --config configs/adatad/thumos/input_mdl_knot_dynamic_adapter_irregular_headv3_shortdiag.py
RUN_SHORTDIAG_TRAIN=1 TRAIN_LOG=logs/mdl_knot_shortdiag_train_rehandfix_$(date +%Y%m%d_%H%M%S_%z).log bash logs/run_mdl_knot_shortdiag_n16r4.sh
```

Bounded rerun remains `SHORT_DIAGNOSTIC_ONLY`: one epoch, no evaluation, no checkpoint claim, no `tools/test.py`, no mAP/runtime/FLOPs/deploy/paper/sparse-compute claim. If non-finite gradients recur after the handoff fix, record them as stability diagnostics and inspect loss/gradient health before any formal/full candidate decision.
