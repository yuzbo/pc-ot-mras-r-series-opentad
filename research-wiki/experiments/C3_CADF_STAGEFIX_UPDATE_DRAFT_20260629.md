# C3 CADF Stagefix Update Draft 2026-06-29

Route labels: `C3_MAINLINE_OPTIMIZATION`, `C3_ORIGINAL_OPTIMIZATION_ROUTE`.

Scope: local-only implementation draft in `OpenTAD_C3CADFStageFix_Worktree_20260629`.
No remote sync, Slurm, training, shared tracker write, or formal metric claim is included.

## Implemented Local Contracts

- Fixed the CADF inverse-CDF max-gap guard dense/compact score indexing bug.
- Added `density_alpha_schedule` support with train warmup and explicit test alpha policy.
- `density_alpha_schedule` is explicitly diagnostic/short-smoke only. Its train-step counter is not checkpoint-resumable by design, and validator requires `c3_alpha_schedule_recoverable=False`, `c3_alpha_schedule_scope="diagnostic_short_smoke_not_resumable"`, and `c3_long_train_resume_claim_locked=True` for schedule configs.
- Added alpha-0 backend-control config to isolate Original AdaTAD average-stride backend behavior from learned CADF density.
- Reduced direct action density to a tiny default (`action=0.02`); uncertainty and transition are the dominant density terms.
- Added transition smoothing plus uniform, weak body, and context safety floors.
- Added diagnostics in metas for selected indices/masks, selected valid length, max/mean gap, repair count/fraction, density entropy, alpha, temperature, density top positions, action/transition/boundary support hooks, train-only GT remap ratio, average-stride restoration error, and diagnostic prediction/proposal caps.
- Kept selected-index-aware and physical-time postprocess paths metadata-only/default-off.

## Fail-Closed Claim State

- Existing 32px and 64px full-train configs are diagnostic-only.
- `c3_full_train_claim_unlocked=False` is required by the validator.
- `c3_physical_time_postprocess_enabled=False` is required by the validator.
- CADF validator forbids route-mix tokens including BH-SDC, DIVERGENT, PVR-QC, RF50, PhysicalGrid, teacher/oracle/raw-cache surfaces, and selected-index-aware/physical-time postprocess enablement.
- Additional route-mix spellings are explicitly blocked: `divergent`, `pvr_qc`, `pvr-qc`, and `pvrqc`.
- `c3_backend_uses_average_stride=True` and `c3_physical_coords_unused_by_backend=True` remain known geometry-risk diagnostics for Original AdaTAD.

## Configs

- `c3_cadf_densitymesh_original_adatad_32px_alpha0_backend_control.py`: alpha-0 exact-uniform-like backend control.
- `c3_cadf_densitymesh_original_adatad_32px_alpha0_only_st_stability_probe.py`: single-factor ST soft path probe, alpha0/fp32/no AMP/no EMA/actionness off.
- `c3_cadf_densitymesh_original_adatad_32px_alpha0_only_actionness_stability_probe.py`: single-factor actionness aux-loss probe, alpha0/fp32/no AMP/no EMA/ST off.
- `c3_cadf_densitymesh_original_adatad_32px_alpha0_st_actionness_combo_gate.py`: minimal combo gate, alpha0 uniform backend control with ST soft path plus actionness aux loss, fp32/no AMP/no EMA, diagnostic-only.
- `c3_cadf_densitymesh_original_adatad_32px_staged_diagnostic.py`: staged alpha warmup with early validation.
- `c3_cadf_densitymesh_original_adatad_32px_full_train_candidate_diagnostic.py`: full-length candidate schedule, still diagnostic-only.
- `c3_cadf_densitymesh_original_adatad_32px_formal_selector_candidate_locked.py`: full-train formal selector candidate, positive CADF density selector with ST soft path and actionness aux loss, fp32/no AMP/no EMA, locked until combo old-window pass evidence is recorded.

## Suggested Next Gate

If local review accepts this worktree, the next allowed action is remote `PRECHECK_ONLY` or a short smoke proposal by the user. This draft does not authorize remote deployment, Slurm, long training, or mAP claims.

## Local Command Evidence

- `conda run -n torch_1 python -m pytest tests/test_c3_cadf_densitymesh_selector.py tests/test_c3_cadf_densitymesh_config.py -q`: `42 passed`.
- `conda run -n torch_1 python -m pytest tests/test_c3_indirect_selector_clean.py tests/test_c3_indirect_clean_config.py -q`: `10 passed`.
- `python -m py_compile opentad\models\selectors\c3_indirect_frame_selector.py tools\validate_c3_indirect_clean_config.py tests\test_c3_cadf_densitymesh_selector.py tests\test_c3_cadf_densitymesh_config.py`: pass.
- `python tools\validate_c3_indirect_clean_config.py <CADF config>`: pass for 32px smoke, 32px full, 32px alpha0 backend control, 32px staged diagnostic, 32px full-train candidate diagnostic, 64px smoke, and 64px full.
- `git diff --check`: exit 0, with Windows LF/CRLF warnings only.

## Review Gate State

- Subagent review attempt 1: `mcp__claude_review.review` returned `Claude CLI did not return JSON output`.
- Subagent review attempt 2: job `6c799365552e488ca7e9b262a4d50a3a` failed with API 402 insufficient balance.
- No valid `PASS_SUBAGENT_FINAL_REVIEW_ONLY` has been obtained in this draft.

## Remote Alpha0 Backend-Control Launch Update - 2026-06-29 22:18 +0800

- Scope remains route-owned CADF stagefix only. No PVR-QC, BH-SDC, DIVERGENT, RF50, PhysicalGrid, shared repo, checkpoint deletion, or protected parent hold release was performed.
- Local config repair: `c3_cadf_densitymesh_original_adatad_32px_alpha0_backend_control.py` was changed from inherited short-smoke behavior to a long diagnostic backend control: `density_alpha=0.0`, density entropy/repulsion losses disabled, fixed 384/768, `workflow.max_train_iters=None`, `workflow.disable_checkpoint=False`, `workflow.val_start_epoch=2`, `workflow.val_eval_interval=1`, and `c3_claim_status="backend_control"`.
- Local verification on Windows: `python -m py_compile opentad/models/selectors/c3_indirect_frame_selector.py tools/validate_c3_indirect_clean_config.py` passed; alpha0 and staged configs passed `tools/validate_c3_indirect_clean_config.py`; local `pytest tests/test_c3_cadf_densitymesh_config.py tests/test_c3_cadf_densitymesh_selector.py` was blocked before test collection by Windows torch DLL initialization error loading `c10.dll`.
- Remote sync target: `/data/home/sczc063/run/yuzibo/OpenTAD_C3CADFStageFix_Precheck_20260629`. Existing resource links were preserved: `data -> ../OpenTAD_C3IndirectClean_5a4722d_20260629/data`, `pretrained -> ../pretrained`; no historical dirty code tree was used as source code.
- Remote verification after sync: in `/data/home/sczc063/run/yuzibo/OpenTAD_C3CADFStageFix_Precheck_20260629`, `python -m py_compile opentad/models/selectors/c3_indirect_frame_selector.py tools/validate_c3_indirect_clean_config.py` passed; alpha0 and staged validators passed; `python -m pytest tests/test_c3_cadf_densitymesh_config.py tests/test_c3_cadf_densitymesh_selector.py` reported `42 passed in 51.91s`.
- Protected hold check: parent hold `1118197 pcot_dbg2g` remained RUNNING on `g0030`. No release/cancel/replace was issued. At 22:08 +0800, `squeue --steps -j 1118197` showed only `1118197.batch` and `1118197.extern`, `nvidia-smi` showed no compute apps, and CADF/train process search was empty.
- Old CADF child status: no active old CADF child was found and no child was stopped.
- Stale watcher status: PID `736917` is a pre-existing login-node watcher for `watch_c3_cadf_alpha0_backend_control_diag20_gpu0_20260629_204000_then_staged.sh`. Its monitored alpha0 launcher log shows an earlier failure, `module: command not found`, and its watch log only contains `WATCH_START`; no staged launch flag or staged child was found. It was not stopped because it was idle and had not launched a child.
- Failed alpha0 launch attempts preserved as evidence:
  - `logs/c3_cadf_alpha0_backend_control_20260629_2210_stdout.log`: launcher failed under `set -u` while sourcing `/etc/profile`.
  - `logs/c3_cadf_alpha0_backend_control_20260629_2213_stdout.log`: launcher entered `tools/train.py` but failed with missing `LOCAL_RANK`.
  - `logs/c3_cadf_alpha0_backend_control_20260629_2216_stdout.log`: launcher entered distributed init but failed with missing `MASTER_ADDR`.
- Active alpha0 launch: `srun --jobid=1118197 --overlap -N1 -n1 -w g0030 /data/home/sczc063/run/yuzibo/OpenTAD_C3CADFStageFix_Precheck_20260629/logs/run_c3_cadf_alpha0_backend_control_n16r4.sh` started child step `1118197.293 run_c3_c` at 22:16:57 +0800. The launcher sets `CUDA_VISIBLE_DEVICES=1`, `LOCAL_RANK=0`, `RANK=0`, `WORLD_SIZE=1`, `MASTER_ADDR=127.0.0.1`, and `MASTER_PORT=29831`.
- Active stdout: `/data/home/sczc063/run/yuzibo/OpenTAD_C3CADFStageFix_Precheck_20260629/logs/c3_cadf_alpha0_backend_control_20260629_2217_stdout.log`.
- Active work dir: `/data/home/sczc063/run/yuzibo/OpenTAD_C3CADFStageFix_Precheck_20260629/exps/thumos/adatad/c3_cadf_densitymesh_original_adatad_32px_alpha0_backend_control/gpu1_id0/`.
- First finite loss evidence: at 2026-06-29 22:18:44 +0800, stdout reported `[Train]: [000][00020/00099] Loss=1.7420 loss_c3_actionness=0.0318 cls_loss=0.9995 reg_loss=0.7107`, with no NaN/Inf in the checked first-loss line.
- Current decision: continue alpha0 backend-control long diagnostic and wait for early validation from epoch 2. Do not launch `staged_diagnostic` until alpha0 validation is healthy enough or explicit instruction is given. If alpha0 hard-NaNs or first validation collapses, stop CADF formal route and move to backend geometry/ranking/postprocess diagnosis.

## Local ST+Actionness Combo Gate Preparation - 2026-06-30 05:53 +0800

- Scope: CADF/C3 only in `OpenTAD_C3CADFStageFix_Worktree_20260629`. No PQR, BH-SDC, Event-Surprise, Boundary Microscope, Frame/Token Hybrid, DIVERGENT route, evaluator/postprocess, or ranking edits.
- New config: `configs/adatad/thumos/c3_cadf_densitymesh_original_adatad_32px_alpha0_st_actionness_combo_gate.py`.
- New launcher wrapper: `logs/run_c3_cadf_alpha0_st_actionness_combo_gate_n16r4.sh`, using `CADF_PROBE_NAME=alpha0_st_actionness_combo_gate`, config above, and default `CADF_MASTER_PORT=29854`.
- Combo semantics: inherits pure-uniform stability probe, keeps `density_alpha=0.0`, no `density_alpha_schedule`, entropy/repulsion losses off, max-gap guard off, fixed 384 selected from 768 dense frames, `st_local_radius=2`, `st_scale=0.5`, `actionness_loss_weight=0.05`, `solver.amp=False`, `solver.fp16_compress=False`, `solver.ema=False`, `diagnostic_only`, no checkpoint/eval in the 4-epoch old-NaN-window gate.
- Validator update: `tools/validate_c3_indirect_clean_config.py` now enforces combo gate AMP/fp16/EMA off, alpha0, ST enabled, and actionness loss enabled when `c3_alpha0_combo_gate="st_soft_path_plus_actionness_fp32_no_amp_no_ema"`.
- Test update: `tests/test_c3_cadf_densitymesh_config.py` now includes the combo config in the CADF config matrix and adds a negative validator check for combo with AMP/fp16/EMA enabled.
- Verification:
  - `python -m py_compile configs\adatad\thumos\c3_cadf_densitymesh_original_adatad_32px_alpha0_st_actionness_combo_gate.py tools\validate_c3_indirect_clean_config.py tests\test_c3_cadf_densitymesh_config.py`: pass.
  - `python tools\validate_c3_indirect_clean_config.py configs\adatad\thumos\c3_cadf_densitymesh_original_adatad_32px_alpha0_only_st_stability_probe.py`: pass.
  - `python tools\validate_c3_indirect_clean_config.py configs\adatad\thumos\c3_cadf_densitymesh_original_adatad_32px_alpha0_only_actionness_stability_probe.py`: pass.
  - `python tools\validate_c3_indirect_clean_config.py configs\adatad\thumos\c3_cadf_densitymesh_original_adatad_32px_alpha0_st_actionness_combo_gate.py`: pass.
  - Direct combo semantic assert: pass, including no PQR/BH-SDC/DIVERGENT tokens in rendered config text.
  - Negative validator check with a temporary combo config overriding `amp=True`, `fp16_compress=True`, and `ema=True`: pass; validator rejected it with the expected combo AMP/fp16/EMA message.
  - Launcher LF/content check: pass.
  - Local pytest caveat: `python -m pytest tests\test_c3_cadf_densitymesh_config.py::test_cadf_densitymesh_alpha0_st_actionness_combo_gate_is_fp32_no_amp_no_ema -q` is blocked before collection by Windows torch DLL initialization failure loading `c10.dll`; this is not counted as a test pass.
- Deployment state: not synced and not launched. No SSH, Slurm, remote process, remote training, mAP, or formal claim was produced.
- Suggested later remote action: sync this worktree delta to `/data/home/sczc063/run/yuzibo/OpenTAD_C3CADFStageFix_Precheck_20260629`, run the same validator on the remote Python env, then run `bash logs/run_c3_cadf_alpha0_st_actionness_combo_gate_n16r4.sh` only inside an already authorized GPU allocation or via the remote agent's approved Slurm/hold policy.

## Local Formal Selector Candidate Preparation - 2026-06-30 06:25 +0800

- Scope: CADF/C3 only in `OpenTAD_C3CADFStageFix_Worktree_20260629`. No remote sync, SSH, Slurm, training, evaluator/postprocess edit, ranking edit, or formal mAP claim was performed.
- New locked config: `configs/adatad/thumos/c3_cadf_densitymesh_original_adatad_32px_formal_selector_candidate_locked.py`.
- New fail-closed launcher: `logs/run_c3_cadf_formal_selector_candidate_locked_n16r4.sh`.
- Candidate semantics: full 60-epoch CADF formal selector candidate, positive `density_alpha=0.65`, no `density_alpha_schedule`, ST soft path preserved with `st_local_radius=2` and `st_scale=0.5`, actionness aux loss preserved with `actionness_loss_weight=0.05`, density entropy loss kept at `0.005`, max-gap guard `12`, fixed 384/768 contract, and deploy/test raw prediction loading/saving disabled.
- Precision/stability choice: `solver.amp=False`, `solver.fp16_compress=False`, and `solver.ema=False`. AMP/EMA can only be reconsidered after stronger combo evidence; this candidate intentionally does not inherit the older AMP/EMA full-train setting.
- Lock semantics: `c3_claim_status="formal_selector_candidate_locked"`, `c3_formal_selector_candidate=True`, `launch_locked_until_combo_pass=True`, required combo config is `c3_cadf_densitymesh_original_adatad_32px_alpha0_st_actionness_combo_gate.py`, required combo status remains `old_nan_window_pass_pending`, evidence is `PENDING`, and user-reported combo child `.376` has not yet passed the old window.
- Launcher semantics: exits with code `2` before reaching `tools/train.py` unless `CADF_COMBO_OLD_WINDOW_PASS=CONFIRMED` and `CADF_COMBO_OLD_WINDOW_EVIDENCE` points to an existing evidence file.
- Validator update: `tools/validate_c3_indirect_clean_config.py` now accepts only the locked formal candidate status and rejects formal configs that unlock launch, enable AMP/fp16/EMA, use alpha0 backend control, include a diagnostic alpha schedule, lose ST/actionness semantics, or contain PQR route-mix tokens.
- Test update: `tests/test_c3_cadf_densitymesh_config.py` now includes formal candidate config/launcher tests and negative validator tests for unlocked launch, AMP/fp16/EMA, alpha0 backend, and PQR route-mix tokens.
- Verification:
  - `python -m py_compile configs\adatad\thumos\c3_cadf_densitymesh_original_adatad_32px_formal_selector_candidate_locked.py tools\validate_c3_indirect_clean_config.py tests\test_c3_cadf_densitymesh_config.py`: pass.
  - `python tools\validate_c3_indirect_clean_config.py configs\adatad\thumos\c3_cadf_densitymesh_original_adatad_32px_formal_selector_candidate_locked.py`: pass.
  - CADF validator matrix over 13 CADF configs including combo and formal candidate: pass.
  - Direct formal semantic harness: pass, including no PQR/BH-SDC/DIVERGENT/teacher/oracle/raw-cache tokens in rendered formal config text.
  - Negative validator harness: pass for unlocked formal launch, AMP/fp16/EMA enabled, alpha0 formal backend, and PQR token injection.
  - `bash -n logs/run_c3_cadf_formal_selector_candidate_locked_n16r4.sh`: pass.
  - Local pytest caveat: focused formal pytest remains blocked before collection by Windows torch DLL initialization failure loading `c10.dll`; this is not counted as a test pass.
- Review gate state: required final read-only subagent review remains incomplete. Claude review job `35e433c4bf4742c8b73d57af8df38b5a` failed with `Claude CLI did not return JSON output`; `llm_chat` fallback failed because `LLM_API_KEY` is not set; MiniMax fallback failed because `MINIMAX_API_KEY` is not set. No valid `PASS_SUBAGENT_FINAL_REVIEW_ONLY` was obtained, so this local prep is not cleared for sync, remote launch, or formal claim.
