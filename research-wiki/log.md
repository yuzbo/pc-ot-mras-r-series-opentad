# Research Log

## 2026-07-01 09:09:10 +08:00

MDL-Knot bounded `SHORT_DIAGNOSTIC_ONLY` was deployed after final read-only review passed.

- Review verdict: `PASS_SUBAGENT_FINAL_REVIEW_ONLY_FOR_SHORT_DIAGNOSTIC_ONLY`.
- Remote runtime worktree: `/data/run01/sczc063/yuzibo/OpenTAD_MDLKnot_SampledRawFix_Precheck_20260701_f5fe3a7`.
- Remote code evidence commit: `9183fb7`.
- Active clean Slurm job: `1132502` (`mdl_shortdiag`), submitted with `--exclude=g0030`; initial state `RUNNING` on `g0032`.
- Clean logdir: `/data/run01/sczc063/yuzibo/OpenTAD_MDLKnot_SampledRawFix_Precheck_20260701_f5fe3a7/logs/mdl_knot_shortdiag_sbatch_9183fb7_20260701_091642_+0800_torchrun_port29683_exclude_g0030`.
- Superseded launch attempts: `1132480` was cancelled after a stale launch-gate argument; `1132482` was cancelled after `torch.distributed.run` hit default port `29400`; `1132483` was cancelled after direct Python exposed the required `LOCAL_RANK` DDP contract. These are deployment-entry correction evidence, not route model failure evidence.
- Startup evidence for `1132502`: clean gates passed, pretrained checkpoint loaded, AMP and EMA enabled, `Epoch 0 started`, and first sampled_raw `MDL_KNOT_PROFILE` line emitted before first loss.
- First training evidence: iter 1 had one skipped non-finite gradient in `module.rpn_head.reg_head.weight`; iter 2-7 then produced finite losses from `1.5851` to `2.1502` with memory about `9750MB`.
- Throughput warning: sampled_raw `selector_and_structural_handoff` remains very slow for some videos, roughly `84-105s`, so this shortdiag is also serving as a speed/root-cause probe.
- Claim state remains locked: no mAP/runtime/FLOPs/deploy/paper/sparse-compute claim; formal/full training, evaluation, `tools/test.py`, and checkpoint result claims remain locked.

## 2026-07-01 08:36:33 +08:00

MDL-Knot remote PRECHECK_ONLY rerun passed after the sparse-compute claim-lock schema fix.

- Remote route-owned precheck copy: `/data/run01/sczc063/yuzibo/OpenTAD_MDLKnot_SampledRawFix_Precheck_20260701_f5fe3a7`.
- Logs: `/data/run01/sczc063/yuzibo/OpenTAD_MDLKnot_SampledRawFix_Precheck_20260701_f5fe3a7/logs/mdl_knot_sampledraw_fix_precheck_f5fe3a7/`.
- Remote verification passed: py_compile log is empty/pass; launch gate returned `PRECHECK_ONLY_REQUEST_ALLOWED`; launch gate with generated precheck summary also returned `PRECHECK_ONLY_REQUEST_ALLOWED`; short diagnostic static gate returned `SHORT_DIAGNOSTIC_CONFIG_STATIC_CHECK_ALLOWED`; focused remote pytest returned `60 passed in 204.98s`.
- Gate state remains conservative: remote precheck unlocks only one bounded `SHORT_DIAGNOSTIC_ONLY` attempt when GPU0 is free. Formal/full long training, evaluation, checkpoints, `tools/test.py`, mAP/runtime/FLOPs/deploy/paper/sparse-compute claims remain locked.

## 2026-07-01 07:53:15 +08:00

MDL-Knot route-owned repair for `DIVERGENT_INNOVATION_MDL_KNOT_DO_NOT_MERGE_WITH_C3`.

- Recorded remote formal child `1118197.535` failure: `ValueError: selected_inputs must be shorter than dense_T for sparse selected-only audit`.
- Fixed sampled_raw/full_raw full-observation edge handling so `valid_k == dense_T` no longer crashes DataLoader but is marked `full_observation_no_compression` and cannot count as sampled/raw sparse-compute evidence.
- Added finer `MDL_KNOT_PROFILE` stages for selector objective loop, gap guard, metadata build, and handoff validator.
- Local verification passed: py_compile relevant files; full requested MDL-Knot pytest suite `55 passed, 3 skipped in 96.63s`.
- Gate state remains locked for formal/full training: local `PRECHECK_ONLY_REQUEST_ALLOWED`, shortdiag static allowed with `validated=false`, no mAP/runtime/FLOPs/deploy/paper/sparse-compute claim.

## 2026-07-01 08:06:19 +08:00

MDL-Knot route-owned blocker fix after final review.

- Fixed stale formal/full train unlock state in `input_mdl_knot_dynamic_adapter_irregular_headv3.py`: top-level and acquisition `formal_train_unlocked/full_train_unlocked` are false, route status is precheck-only after sampled_raw edge fix, and no sparse-compute claim is open.
- Hardened `validate_mdl_knot_launch_gate.py` to reject stale user-override/formal/full/queued/unlocked route statuses, config/acquisition train unlocks, and precheck summaries missing `no_claims.sparse_compute=True`.
- Updated tests to assert formal/full train remains locked and to reject stale unlock/status/sparse-claim cases.
- Verification passed: py_compile relevant files, launch gate `PRECHECK_ONLY_REQUEST_ALLOWED`, shortdiag static gate `SHORT_DIAGNOSTIC_CONFIG_STATIC_CHECK_ALLOWED` with `validated=false`, and `tests/test_mdl_knot_tools_and_integration.py` passed `19 passed, 3 skipped in 19.18s`.
- Historical user-override child `1118197.535` remains failed evidence only. Next allowed action is remote `PRECHECK_ONLY`, then one-epoch shortdiag after precheck; formal/full long training remains locked.

## 2026-07-01 08:24:45 +08:00

MDL-Knot route-owned precheck schema fix after remote PRECHECK_ONLY pytest failure.

- Remote feedback: py_compile, launch gate, and shortdiag gate passed, but valid-precheck pytest failed because the generated `mdl_knot_precheck_summary.json` lacked an explicit sparse-compute claim lock.
- Updated `audit_mdl_knot_pipeline_precheck.py` so generated summaries include `no_claims.sparse_compute=True`, `no_sparse_compute_claim=True`, `sparse_compute_claim=False`, and `fixed_pad_sparse_compute_claim_locked=True`; config evidence also records sparse-compute claim locked.
- Updated tests to assert valid generated precheck summaries carry sparse-compute locks and that missing/false `no_claims.sparse_compute` is rejected.
- Verification passed: py_compile relevant files, targeted valid-precheck test `1 passed in 4.04s`, generated precheck plus launch gate `PRECHECK_ONLY_REQUEST_ALLOWED`, and `tests/test_mdl_knot_tools_and_integration.py` `19 passed, 3 skipped in 19.96s`.
- No remote action, training, evaluation, or claims were run locally. Next allowed action remains remote `PRECHECK_ONLY`; shortdiag only after remote precheck; formal/full train remains locked.
