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

## 2026-06-30 N16R4 Post-Fix Shortdiag Launch Evidence

Remote GitHub synchronization was performed on N16R4 with the platform academic
proxy. The existing remote clone
`/data/home/sczc063/run/yuzibo/OpenTAD_MDLKnot_RealDiag_20260630_fd2977f`
fast-forwarded to branch `codex/divergent-mdl-knot-realdiag-20260630`, commit
`797d05c`.

Static gates were rerun remotely:

- `validate_mdl_knot_launch_gate.py --config configs/adatad/thumos/input_mdl_knot_dynamic_adapter_irregular_headv3.py`
  returned `PRECHECK_ONLY_REQUEST_ALLOWED`, with `formal_train_unlocked=false`.
- `validate_mdl_knot_shortdiag.py --config configs/adatad/thumos/input_mdl_knot_dynamic_adapter_irregular_headv3_shortdiag.py`
  returned `SHORT_DIAGNOSTIC_CONFIG_STATIC_CHECK_ALLOWED`, with
  `validated=false` because execution evidence is still required.

After confirming GPU0 was free and GPU1 was not used by divergent routes, the
post-fix one-epoch short diagnostic was launched inside the protected parent
hold as child `1118197.499 mdl_shortfix3_g0`, with explicit
`CUDA_VISIBLE_DEVICES=0`.

Run directory:

`/data/home/sczc063/run/yuzibo/OpenTAD_MDLKnot_RealDiag_20260630_fd2977f/logs/mdl_knot_shortdiag_gpu0_postfix_797d05c_20260630_200355_+0800_r3`

Early finite-loss evidence:

```text
2026-06-30 20:08:25 [000][00001/00199] Loss=2.4086 cls_loss=0.5088 reg_loss=0.5807 boundary_loss=1.3191
2026-06-30 20:10:20 [000][00002/00199] Loss=2.1510 cls_loss=0.3920 reg_loss=0.4437 boundary_loss=1.3153
2026-06-30 20:10:53 [000][00003/00199] Loss=2.4525 cls_loss=0.5212 reg_loss=0.5954 boundary_loss=1.3359
```

Hard-marker scan was empty for Traceback, RuntimeError, CUDA OOM, killed/no
space, `ValueError`, `Loss=nan`, `cost=nan`, and the previous
`selected_inputs must be a real sparse gather, not dense passthrough` failure.

Interpretation:

- The post-fix branch has entered actual training and passed the immediate
  dense-passthrough validator failure point seen in old child `1118197.488`.
- This is still `SHORT_DIAGNOSTIC_ONLY`, not a route-success result.
- MDL formal/full training, `tools/test.py`, official evaluation, mAP,
  runtime/FLOPs, deployment, paper, and sparse-compute claims remain locked
  until the one-epoch shortdiag completes and its validator passes.

## 2026-06-30 Pro-Guided Speed Root-Cause Patch

Timestamp: `2026-06-30 21:41:02 +08:00`.

The old post-fix short diagnostic child later failed in the handoff validator
path, and GPT-5.5 Pro identified the dominant speed risk as full dense raw
handoff audit in the training hot path. This patch keeps the original scout
coverage and MDL budget semantics while separating audit depth:

- `structural`: index/mask/metadata contract audit only.
- `sampled_raw`: reads and validates only selected raw frames; this is the
  short diagnostic and training hot path.
- `full_raw`: reads the full dense raw window for bounded realdiag/formal
  readiness evidence only.

The short diagnostic parameters were explicitly restored and gate-checked:

- `scout_stride=8`
- `scout_max_frames=96`
- `dense_window_size=768`
- `window_size=384`
- `max_k=384`

`MDL_KNOT_PROFILE=1` now enables timing output for scout construction, raw
probe decode, selector/structural handoff, and sampled/full raw audit decode.
It is a diagnostic flag only and does not change selection or detector
semantics.

Local verification in the route-owned worktree:

```text
python tools/mdl_knot/validate_mdl_knot_shortdiag.py --config configs/adatad/thumos/input_mdl_knot_dynamic_adapter_irregular_headv3_shortdiag.py
SHORT_DIAGNOSTIC_CONFIG_STATIC_CHECK_ALLOWED

python tools/mdl_knot/validate_mdl_knot_launch_gate.py --config configs/adatad/thumos/input_mdl_knot_dynamic_adapter_irregular_headv3.py
PRECHECK_ONLY_REQUEST_ALLOWED

python -m pytest tests/test_mdl_knot_core.py tests/test_mdl_knot_tools_and_integration.py tests/test_mdl_knot_realdiag.py tests/test_mdl_knot_shortdiag.py -q
54 passed, 2 skipped in 96.93s

git diff --check
exit code 0; LF/CRLF warnings only
```

Final read-only subagent review returned
`PASS_SUBAGENT_FINAL_REVIEW_ONLY`: no blockers. The reviewer confirmed that
the training/shortdiag default no longer materializes full dense raw windows,
`full_raw` remains reserved for realdiag/formal evidence, no scout/window/K
parameters were reduced, and gates remain fail-closed.

Non-blocking reviewer note: the equivalent real-video collector exposes
`--handoff-audit-mode sampled_raw`, but its fallback path is structural-like
and does not itself read selected raw frames. This does not affect the real
`LoadFrames` training/shortdiag path, which does validate selected raw samples.
Do not use the fallback collector's sampled mode as formal raw-handoff
evidence unless it is later hardened.

Current unlock state:

- Local final-code candidate for `SHORT_DIAGNOSTIC_ONLY`: yes.
- Remote sync / GPU0 short diagnostic: allowed only after route-owned commit
  and coordinator deployment check.
- Formal/full train, `tools/test.py`, official evaluation, mAP, runtime/FLOPs,
  deploy, paper, and sparse-compute claims: still locked.

## 2026-06-30 GPU0 Speedfix Short Diagnostic Deployment

Timestamp: `2026-06-30 21:47:33 +08:00`.

## 2026-07-01 03:30 +08:00 formal-run path and AMP repair

Route label: `DIVERGENT_INNOVATION_MDL_KNOT_DO_NOT_MERGE_WITH_C3`

Owned worktree: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_MDLKnot_RealDiag_Worktree_20260630`

Owned branch: `codex/divergent-mdl-knot-realdiag-20260630`

Remote evidence motivating the repair:

- Active child step `1118197.535` is running as `mdl_formal_g0` on GPU0.
- Its log path is
  `/data/home/sczc063/run/yuzibo/OpenTAD_MDLKnot_RealDiag_20260630_fd2977f/logs/mdl_knot_user_override_formal_after_bvr_db2f650_20260701_025709_+0800/srun-1118197.out`.
- At `2026-07-01 03:26 +08:00`, the log had no `Loss=` line and had not
  grown since `2026-07-01 02:58:31 +08:00`.
- The active log also showed inherited evaluator pollution:
  `ground_truth_filename='/root/autodl-tmp/annotations/thumos_14_anno.json'`.

Code repair:

- `configs/adatad/thumos/input_mdl_knot_dynamic_adapter_irregular_headv3.py`
  now has a route-status string that remains a local final-code candidate
  without mentioning other divergent route names.
- The same config explicitly sets
  `evaluation = dict(ground_truth_filename=annotation_path)`.
- `solver.amp` is now `True`. This changes numeric precision for training
  speed/memory only; it does not change scout stride, scout max frames, dense
  window size, selected-frame budget, or MDL-Knot acquisition logic.
- `tools/mdl_knot/validate_mdl_knot_launch_gate.py` now fails closed unless
  `evaluation.ground_truth_filename` and all split `ann_file` entries equal
  `annotation_path`, and rejects legacy `/root/autodl-tmp` evaluator paths.
- `tools/mdl_knot/audit_mdl_knot_pipeline_precheck.py` now applies the same
  workflow-field drift whitelist as the launch gate so `*_interval` fields do
  not falsely trip the `INTERVAL` route-drift token.
- `tests/test_mdl_knot_tools_and_integration.py` now checks the evaluator path
  override and AMP flag.

Local command evidence:

```powershell
python -m pytest tests/test_mdl_knot_tools_and_integration.py tests/test_mdl_knot_shortdiag.py -q
```

Result: `30 passed, 2 skipped in 21.03s`.

```powershell
python tools/mdl_knot/validate_mdl_knot_launch_gate.py --config configs/adatad/thumos/input_mdl_knot_dynamic_adapter_irregular_headv3.py
```

Result: `PRECHECK_ONLY_REQUEST_ALLOWED`; all remote sync, Slurm, training,
evaluation, `tools/test.py`, mAP, runtime/FLOPs, deploy, and paper claims remain
locked by the gate output.

```powershell
python tools/mdl_knot/audit_mdl_knot_pipeline_precheck.py --out-dir .tmp_mdl_debug_precheck --overwrite
python tools/mdl_knot/validate_mdl_knot_launch_gate.py --config configs/adatad/thumos/input_mdl_knot_dynamic_adapter_irregular_headv3.py --precheck-summary .tmp_mdl_debug_precheck/mdl_knot_precheck_summary.json
```

Result: precheck summary was generated and accepted as
`PRECHECK_ONLY_REQUEST_ALLOWED`.

Interpretation:

- Current step `1118197.535` should not be used as healthy formal-training
  evidence because it was launched with the polluted evaluator path and still
  has no first finite loss line after the launch sanity window.
- This repair prepares the next MDL-Knot restart/queue candidate only.
- It does not create a new mAP result, runtime claim, sparse-compute claim,
  deploy claim, or paper claim.

Code commit deployed:

- GitHub branch: `codex/divergent-mdl-knot-realdiag-20260630`
- Commit: `be65889`
- Commit message:
  `DIVERGENT_INNOVATION_MDL_KNOT_DO_NOT_MERGE_WITH_C3 speed audit handoff modes`

Remote clone:

`/data/home/sczc063/run/yuzibo/OpenTAD_MDLKnot_RealDiag_20260630_fd2977f`

Deployment scope:

- Parent hold: `1118197 pcot_dbg2g`, preserved.
- Node: `g0030`.
- GPU binding: `CUDA_VISIBLE_DEVICES=0`.
- GPU1 was not used.
- Run tier: `SHORT_DIAGNOSTIC_ONLY`.
- No evaluation, no `tools/test.py`, no checkpoint/mAP/runtime/FLOPs/deploy/
  paper/sparse-compute claim.

Run directory:

`/data/home/sczc063/run/yuzibo/OpenTAD_MDLKnot_RealDiag_20260630_fd2977f/logs/mdl_knot_shortdiag_gpu0_speedfix_be65889_20260630_214733_+0800`

Launch command used `srun --overlap --jobid=1118197` with job name
`mdl_speedfix_g0`.

Startup health check:

- `parajobs` still shows parent hold `1118197 pcot_dbg2g` running on `g0030`.
- `launcher_srun.log` entered `run_mdl_knot_shortdiag_n16r4.sh` and printed
  the MDL route label and short diagnostic scope.
- The run was in the script's focused pytest/gate phase at first check; no
  immediate environment crash, CUDA OOM, old dense-passthrough validator error,
  or traceback was observed in the startup window.

Next monitoring policy:

Do not refresh logs frequently. Recheck near expected first training loss or on
user request; record only material state changes such as crash, finite loss,
short diagnostic validation pass/fail, or a next-action-changing blocker.

## 2026-07-01 sampled_raw full-observation edge repair

Timestamp: `2026-07-01 07:53:15 +08:00`.

Route label: `DIVERGENT_INNOVATION_MDL_KNOT_DO_NOT_MERGE_WITH_C3`

Owned worktree: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_MDLKnot_RealDiag_Worktree_20260630`

Owned branch: `codex/divergent-mdl-knot-realdiag-20260630`

Remote failure evidence accepted for this repair:

- N16R4 formal child: `1118197.535`.
- Failure: `ValueError: selected_inputs must be shorter than dense_T for sparse selected-only audit`.
- Interpretation: sampled_raw selected-only validation was too strict for legitimate `valid_k == dense_T` / short dense-window cases. The old run is failed evidence only; it is not mAP, runtime, sparse-compute, deploy, or paper evidence.

Changed surface:

- Input sampling / handoff: yes, sampled_raw/full_raw validator edge-case handling and audit metadata.
- Dynamic budget policy: no policy change; short/easy windows may still produce `valid_k == dense_T`.
- Token compression: no.
- Adapter/backbone internals: no.
- Detector head logic: no.
- Loss/assignment: no.
- Test-time post-processing/evaluator: no.

Fix summary:

- `validate_sampled_sparse_handoff` and `validate_real_sparse_handoff` now allow the legal full-observation/no-compression case when `selected_len == valid_k == dense_T`.
- The same case is explicitly marked with `full_observation_no_compression=true`, `sampled_raw_sparse_compute_evidence=false`, `raw_sparse_compute_evidence=false`, `sparse_compute_claim=false`, and `no_sparse_compute_claim=true`.
- True dense passthrough where `selected_len == dense_len` but `selected_len != valid_k` remains rejected as dense passthrough.
- `LoadFrames` sampled_raw audit writes `sampled_raw_full_observation` for full-observation windows and sets `mdl_knot_real_sparse_handoff_validated=false`, so the edge case cannot be counted as sparse selected-only/raw sparse compute evidence.
- Pipeline diagnostics now count `full_observation_no_compression_windows`, `sampled_raw_full_observation_windows`, and `full_raw_full_observation_windows` separately from sampled/full raw sparse evidence windows.
- `MDL_KNOT_PROFILE=1` now gives finer timing inside the previous `selector_and_structural_handoff` block: `selector_initialization`, `selector_objective_loop`, `selector_candidate_pool`, `gap_guard`, `metadata_build`, `handoff_metadata_build`, and `handoff_validator`, alongside existing scout and sampled/full raw decode stages.

Local verification in the owned worktree:

```powershell
python -m py_compile opentad/acquisition/mdl_knot/validators.py opentad/acquisition/mdl_knot/selector.py opentad/acquisition/mdl_knot/handoff.py opentad/acquisition/mdl_knot/diagnostics.py opentad/datasets/transforms/end_to_end.py tools/mdl_knot/audit_mdl_knot_pipeline_precheck.py tools/mdl_knot/collect_mdl_knot_real_video_diagnostics.py tools/mdl_knot/validate_mdl_knot_launch_gate.py tools/mdl_knot/validate_mdl_knot_shortdiag.py configs/adatad/thumos/input_mdl_knot_dynamic_adapter_irregular_headv3.py configs/adatad/thumos/input_mdl_knot_dynamic_adapter_irregular_headv3_shortdiag.py tests/test_mdl_knot_core.py tests/test_mdl_knot_tools_and_integration.py tests/test_mdl_knot_realdiag.py tests/test_mdl_knot_shortdiag.py
```

Result: exit code `0`.

```powershell
python -m pytest tests/test_mdl_knot_core.py tests/test_mdl_knot_shortdiag.py tests/test_mdl_knot_tools_and_integration.py tests/test_mdl_knot_realdiag.py -q
```

Result: `55 passed, 3 skipped in 96.63s`.

```powershell
python tools/mdl_knot/validate_mdl_knot_launch_gate.py --config configs/adatad/thumos/input_mdl_knot_dynamic_adapter_irregular_headv3.py
```

Result: `PRECHECK_ONLY_REQUEST_ALLOWED`; still locked for remote sync, Slurm, training, evaluation, `tools/test.py`, mAP/runtime/FLOPs/deploy/paper claims by gate output.

```powershell
python tools/mdl_knot/validate_mdl_knot_shortdiag.py --config configs/adatad/thumos/input_mdl_knot_dynamic_adapter_irregular_headv3_shortdiag.py
```

Result: `SHORT_DIAGNOSTIC_CONFIG_STATIC_CHECK_ALLOWED`, `validated=false`; still requires execution evidence for formal readiness and keeps full training, evaluation, checkpoints, `tools/test.py`, mAP, and sparse-compute claims locked.

Current launch interpretation:

- Remote `PRECHECK_ONLY`: allowed after normal coordinator deployment/sync decision.
- One-epoch `SHORT_DIAGNOSTIC_ONLY`: allowed after remote precheck passes; no evaluation, no checkpoint claim, no mAP/runtime/FLOPs/deploy/paper/sparse-compute claim.
- Formal/full long training: still prohibited/locked until remote precheck plus one-epoch shortdiag pass and a separate explicit formal/full-train decision exists.

## 2026-07-01 formal/full train lock blocker fix

Timestamp: `2026-07-01 08:06:19 +08:00`.

Final read-only review returned `BLOCKED` because the main formal config still
contained stale user-override formal-run state after child `1118197.535` failed.
The stale fields were inconsistent with the current sampled_raw edge-fix
decision, which permits only remote `PRECHECK_ONLY` and then one-epoch
`SHORT_DIAGNOSTIC_ONLY`.

Fix summary:

- `configs/adatad/thumos/input_mdl_knot_dynamic_adapter_irregular_headv3.py`
  is fail-closed again:
  - `formal_train_unlocked=False`;
  - `full_train_unlocked=False`;
  - `mdl_knot_acquisition.formal_train_unlocked=False`;
  - `mdl_knot_acquisition.full_train_unlocked=False`;
  - `sparse_compute_claim=False`;
  - route status changed to
    `LOCAL_FINAL_CODE_CANDIDATE_PRECHECK_ONLY_AFTER_SAMPLED_RAW_EDGE_FIX_NO_METRIC_CLAIMS`;
  - work dir no longer contains user-override/formal-train wording.
- `tools/mdl_knot/validate_mdl_knot_launch_gate.py` now fails closed if the
  config or acquisition dict tries to set formal/full train unlocked or
  sparse-compute claim true. It also rejects route-status tokens
  `USER_OVERRIDE`, `FORMAL_TRAIN`, `FULL_TRAIN`, `QUEUED`, and `UNLOCKED`.
- The precheck summary gate now requires `no_claims.sparse_compute=True`; a
  sparse-compute claim lock omission is no longer accepted.
- Tests now assert that the main config remains formal/full-train locked and
  that the launch gate rejects stale unlock/status/sparse-claim states.

Local verification in the owned worktree:

```powershell
python -m py_compile configs/adatad/thumos/input_mdl_knot_dynamic_adapter_irregular_headv3.py configs/adatad/thumos/input_mdl_knot_dynamic_adapter_irregular_headv3_shortdiag.py tools/mdl_knot/validate_mdl_knot_launch_gate.py tools/mdl_knot/validate_mdl_knot_shortdiag.py tests/test_mdl_knot_tools_and_integration.py
```

Result: exit code `0`.

```powershell
python tools/mdl_knot/validate_mdl_knot_launch_gate.py --config configs/adatad/thumos/input_mdl_knot_dynamic_adapter_irregular_headv3.py
```

Result: `PRECHECK_ONLY_REQUEST_ALLOWED`, with `formal_train_unlocked=false`.

```powershell
python tools/mdl_knot/validate_mdl_knot_shortdiag.py --config configs/adatad/thumos/input_mdl_knot_dynamic_adapter_irregular_headv3_shortdiag.py
```

Result: `SHORT_DIAGNOSTIC_CONFIG_STATIC_CHECK_ALLOWED`, `validated=false`.

```powershell
python -m pytest tests/test_mdl_knot_tools_and_integration.py -q
```

Result: `19 passed, 3 skipped in 19.18s`.

Current decision:

- Historical user-override formal child `1118197.535` is failed evidence only.
- This edge fix and lock fix do not create full-train permission.
- Next action remains remote `PRECHECK_ONLY` first, then one-epoch
  `SHORT_DIAGNOSTIC_ONLY` only after remote precheck passes.
- Formal/full long training, `tools/test.py`, evaluation, checkpoints,
  mAP/runtime/FLOPs/deploy/paper/sparse-compute claims remain locked.

## 2026-07-01 precheck sparse-compute claim-lock schema repair

Timestamp: `2026-07-01 08:24:45 +08:00`.

Remote `PRECHECK_ONLY` feedback:

- Remote `py_compile`, `validate_mdl_knot_launch_gate.py`, and
  `validate_mdl_knot_shortdiag.py` passed.
- Remote pytest failed:
  `tests/test_mdl_knot_tools_and_integration.py::test_launch_gate_unlocks_only_for_valid_precheck_summary`.
- Root cause: launch gate correctly required `no_claims.sparse_compute=True`,
  but the precheck summary generator schema could still be interpreted as an
  older format that did not explicitly lock sparse-compute claims.

Fix summary:

- `tools/mdl_knot/audit_mdl_knot_pipeline_precheck.py` now centralizes
  precheck no-claim locks with `sparse_compute=True`.
- Generated `mdl_knot_precheck_summary.json` now also includes explicit
  top-level sparse-compute claim locks:
  `no_sparse_compute_claim=True`, `sparse_compute_claim=False`, and
  `fixed_pad_sparse_compute_claim_locked=True`.
- `config_evidence` now records
  `no_sparse_compute_claim=True` and `sparse_compute_claim=False`.
- Tests now assert that a valid generated precheck summary contains these
  sparse-compute locks and that missing or false `no_claims.sparse_compute`
  remains rejected by the launch gate.

Local verification in the owned worktree:

```powershell
python -m py_compile tools/mdl_knot/audit_mdl_knot_pipeline_precheck.py tools/mdl_knot/validate_mdl_knot_launch_gate.py tests/test_mdl_knot_tools_and_integration.py
```

Result: exit code `0`.

```powershell
python -m pytest tests/test_mdl_knot_tools_and_integration.py::test_launch_gate_unlocks_only_for_valid_precheck_summary -q
```

Result: `1 passed in 4.04s`.

```powershell
$out = Join-Path $env:TEMP 'mdl_knot_precheck_sparse_lock_local'
python tools/mdl_knot/audit_mdl_knot_pipeline_precheck.py --out-dir $out --overwrite
python tools/mdl_knot/validate_mdl_knot_launch_gate.py --config configs/adatad/thumos/input_mdl_knot_dynamic_adapter_irregular_headv3.py --precheck-summary (Join-Path $out 'mdl_knot_precheck_summary.json')
```

Result: generator emitted `VALIDATED_PRECHECK_SUMMARY`; launch gate returned
`PRECHECK_ONLY_REQUEST_ALLOWED`.

```powershell
python -m pytest tests/test_mdl_knot_tools_and_integration.py -q
```

Result: `19 passed, 3 skipped in 19.96s`.

Current decision:

- Remote `PRECHECK_ONLY` remains the only allowed remote next step.
- One-epoch `SHORT_DIAGNOSTIC_ONLY` may follow only after remote precheck passes.
- Formal/full long training, `tools/test.py`, evaluation, checkpoints, and all
  mAP/runtime/FLOPs/deploy/paper/sparse-compute claims remain locked.

## 2026-07-01 remote PRECHECK_ONLY rerun passed

Timestamp: `2026-07-01 08:36:33 +08:00`.

Remote route-owned precheck copy:

`/data/run01/sczc063/yuzibo/OpenTAD_MDLKnot_SampledRawFix_Precheck_20260701_f5fe3a7`

Remote log directory:

`/data/run01/sczc063/yuzibo/OpenTAD_MDLKnot_SampledRawFix_Precheck_20260701_f5fe3a7/logs/mdl_knot_sampledraw_fix_precheck_f5fe3a7/`

Remote verification:

- `py_compile.log`: empty/pass.
- `launch_gate.log`: `PRECHECK_ONLY_REQUEST_ALLOWED`.
- `launch_gate_with_summary.log`: `PRECHECK_ONLY_REQUEST_ALLOWED` after consuming the generated precheck summary.
- `shortdiag_gate.log`: `SHORT_DIAGNOSTIC_CONFIG_STATIC_CHECK_ALLOWED`, `validated=false`.
- `pytest.log`: `60 passed in 204.98s (0:03:24)`.

Current decision:

- The sampled_raw/full-observation edge repair and sparse-compute claim-lock schema fix now pass remote `PRECHECK_ONLY`.
- This unlocks only a one-epoch `SHORT_DIAGNOSTIC_ONLY` attempt when GPU0 is free.
- It does not unlock formal/full long training, evaluation, checkpoints,
  `tools/test.py`, mAP/runtime/FLOPs/deploy/paper claims, or any sparse-compute
  claim.

## 2026-07-01 short diagnostic deployment

Timestamp: `2026-07-01 09:09:10 +08:00`.

Final read-only review returned:

`PASS_SUBAGENT_FINAL_REVIEW_ONLY_FOR_SHORT_DIAGNOSTIC_ONLY`

Allowed action:

- One bounded one-epoch `SHORT_DIAGNOSTIC_ONLY` only.
- No evaluation, `tools/test.py`, checkpoint claim, mAP/runtime/FLOPs/deploy/paper claim, or sparse-compute claim.
- Formal/full training remains locked.

Remote evidence:

- Runtime worktree: `/data/run01/sczc063/yuzibo/OpenTAD_MDLKnot_SampledRawFix_Precheck_20260701_f5fe3a7`.
- Remote code commit: `9183fb7` (`DIVERGENT_INNOVATION_MDL_KNOT_DO_NOT_MERGE_WITH_C3 remote shortdiag sync`).
- Active Slurm job: `1132502` (`mdl_shortdiag`).
- Slurm constraints: `--gpus=1`, `--cpus-per-task=4`, `--exclude=g0030`.
- Initial clean allocation: `RUNNING` on `g0032`, not the protected hold node `g0030`.
- Clean logdir: `/data/run01/sczc063/yuzibo/OpenTAD_MDLKnot_SampledRawFix_Precheck_20260701_f5fe3a7/logs/mdl_knot_shortdiag_sbatch_9183fb7_20260701_091642_+0800_torchrun_port29683_exclude_g0030`.
- Superseded launch attempts: `1132480` was cancelled after a stale sbatch argument made `launch_gate.log` invalid; `1132482` was cancelled after `torch.distributed.run` hit the shared-node default port `29400`; `1132483` was cancelled after direct Python exposed the required `LOCAL_RANK` DDP entrypoint contract. These are deployment-entry corrections, not model failure evidence.
- Startup evidence for `1132502`: clean py_compile, base launch gate, and shortdiag static gate passed; pretrained VideoMAE checkpoint loaded; AMP and EMA enabled; `Epoch 0 started`; first sampled_raw `MDL_KNOT_PROFILE` line emitted before the first loss.
- First training evidence: iter 1 had one skipped non-finite gradient in `module.rpn_head.reg_head.weight`, followed by finite losses at iter 2-7: `1.5851`, `2.1502`, `1.9394`, `1.8892`, `2.0425`, `1.9420`.
- Speed evidence: sampled_raw `selector_and_structural_handoff` can take about `84-105s` for long videos in this shortdiag; this is a serious throughput issue to diagnose, but not a launch crash.

Current decision:

- Monitor only for launch sanity, finite loss, hard errors, and post-shortdiag validator output.
- Do not treat this diagnostic as mAP or sparse-compute evidence.
- Do not unlock formal/full training without a new explicit gate decision.

## 2026-07-01 evidence-branch sync and live shortdiag status

Timestamp: `2026-07-01 09:31:39 +08:00`.

GitHub sync:

- Ordinary push to existing branch `codex/divergent-mdl-knot-realdiag-20260630` was rejected as non-fast-forward.
- No force push or overwrite was attempted.
- Pushed current route-owned local state to a new evidence branch, then fast-forwarded it with live-sync records after recording tracker/log evidence:
  `codex/divergent-mdl-knot-status-410d499f-20260701`.
- GitHub URL:
  `https://github.com/yuzbo/pc-ot-mras-r-series-opentad/tree/codex/divergent-mdl-knot-status-410d499f-20260701`.

Live shortdiag status:

- Job `1132502 mdl_shortdiag` remains `RUNNING` on `g0032`, not protected hold node `g0030`.
- The inspected training tail reached iter 15 with finite losses after the initial iter-1 skipped non-finite gradient.
- No new Traceback or RuntimeError was observed in the inspected tail.
- The sampled_raw `selector_and_structural_handoff` throughput issue remains serious on long videos.

Current decision:

- Continue this bounded shortdiag to natural completion/failure.
- Use it only for stability and speed diagnosis.
- Formal/full long training and all evaluation/runtime/FLOPs/deploy/paper/sparse-compute claims remain locked.
