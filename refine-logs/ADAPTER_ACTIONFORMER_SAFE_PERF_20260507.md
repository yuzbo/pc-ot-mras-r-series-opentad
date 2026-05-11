# Adapter + ActionFormer Safe Performance Push

Date: 2026-05-07
Last updated: 2026-05-10 08:42 +08:00

## Current Evidence

| Run | Avg-mAP | mAP@0.7 | Status / meaning |
|---|---:|---:|---|
| `input_random_fixed_50pct_adapter` | 63.77 | 42.19 | Strong Adapter + ActionFormer baseline. |
| `input_random_fixed_50pct_adapter_virtual_baseline` | 63.84 | 41.91 | Metadata plumbing is healthy. |
| `input_oracle_boundary_dense_50pct_adapter` | 77.62 | 66.91 | Large headroom if boundary/input allocation improves. |
| `input_stride2_uniform_50pct_adapter_frozen_recipe` | 42.82 | 16.44 | First eval only; stopped. Frozen-style stride-2 recipe does not rescue Adapter. |
| `input_random_fixed_50pct_adapter_regloss15` | 37.64 | 13.11 | First eval only; stopped. Regression-loss reweighting is destructive. |
| `input_random_fixed_50pct_adapter_simota_mink4_w1` | 62.92 | 41.34 | SimOTA does not improve this adapter setting. |
| `input_random_fixed_50pct_adapter_late_linear` | 49.75 | 24.44 | Non-gated time embedding is destructive. |
| `input_random_fixed_50pct_adapter_late_linear_zero` | 49.57 | 24.38 | Zero initial scale alone is insufficient; the branch can still learn into a bad region. |
| `input_random_fixed_50pct_adapter_pdrop02_control` | 48.41 | 22.99 | `input_pdrop=0.2` alone is destructive. |
| `input_random_fixed_50pct_adapter_irregular_headv3_x_pdrop0` | 42.29 | 16.39 | Removing pdrop does not rescue HeadV3. |
| `input_random_fixed_50pct_adapter_irregular_dense_control_pdrop0` | crashed | crashed | DDP reentrant checkpoint issue, not a metric result. |

## Diagnosis

1. The dominant failure mode is violation of the pretrained Adapter + ActionFormer feature contract.
2. `input_pdrop=0.2` is not a harmless regularizer in this setting; it destroys the strong adapter baseline.
3. HeadV3 / IrregularActionFormer is currently too far from the baseline contract and too unstable to be the primary route to 65+.
4. The oracle-boundary result says the real headroom is boundary/input-allocation related, but model-side changes must be introduced as strict residuals.
5. The DDP crash is rooted in reentrant activation checkpointing with DDP/static graph; `use_reentrant=False` is required.

Claude CLI agreed with the first-order diagnosis and recommended adding a later fixed boundary-enhanced sampling control. That is useful as a next diagnostic, but the two experiments below keep the requested Adapter + ActionFormer model path.

## Implemented Changes

| Area | Files | Contract |
|---|---|---|
| DDP checkpoint fix | `opentad/models/backbones/vit_adapter.py` | `cp.checkpoint(..., use_reentrant=False)` prevents the adapter gamma "marked ready twice" crash. |
| Safe adapter time gate | `vit_adapter.py`, `backbone_wrapper.py`, `actionformer.py` | Time embedding starts at exact zero, is forced to zero for 5 epochs, then tanh-bounded by `time_embed_scale_max=0.03`; epoch is propagated to the backbone. |
| Safe ActionFormer head residual | `opentad/models/dense_heads/anchor_free_head.py`, `actionformer.py` | Optional depthwise temporal cls-logit residual is gated by `cls_residual_scale=0.0`, so logits are baseline-identical at init. |
| Adapter experiment config | `configs/adatad/thumos/input_random_fixed_50pct_adapter_timegate_safe.py` | Conservative time embedding, `time_embed` LR `2e-5`, no `input_pdrop`. |
| ActionFormer experiment config | `configs/adatad/thumos/input_random_fixed_50pct_adapter_head_clsres_safe.py` | Zero-gated head residual, no `input_pdrop`. |
| Driver | `scripts/run_adapter_safe_perf_pair.sh` | Check-only validation and sliced launches for the two new configs. |

## Verification

| Scope | Result |
|---|---|
| Local contract tests | `python -m pytest tests/test_adapter_safety_contracts.py -q`: `8 passed`. |
| Local Python syntax | `py_compile` on modified Python/config files: PASS. |
| Local whitespace check | `git diff --check`: PASS. |
| Local bash syntax | Windows Bash service returned `E_ACCESSDENIED`; checked on Linux servers instead. |
| 35407 remote syntax/check-only | `py_compile`, `bash -n`, `CHECK_ONLY=1 bash scripts/run_adapter_safe_perf_pair.sh`: PASS. |
| 25876 remote syntax/check-only | `py_compile`, `bash -n`, `CHECK_ONLY=1 bash scripts/run_adapter_safe_perf_pair.sh`: PASS. |
| Remote contract tests | Pure-Python invocation of test functions on both servers: PASS. |
| Model/optimizer build | 35407 built both configs and called `model.get_optim_groups`; head config reports `cls_residual_scale 0.0`. |

## Deployment

| Server | Screen | Config | Status |
|---|---|---|---|
| `35407` | `adapter_timegate_safe_35407` | `input_random_fixed_50pct_adapter_timegate_safe.py` | Running. Log: `logs/input_random_fixed_50pct_adapter_timegate_safe_20260507_224015.log`. Epoch 3 finished and epoch 4 started. Loss trajectory: `2.1545 -> 1.3540 -> 0.9436 -> 0.8773`; latest `cls_loss=0.5251`, `reg_loss=0.3521`, `mem=6335MB`. |
| `25876` | `adapter_head_clsres_safe_25876` | `input_random_fixed_50pct_adapter_head_clsres_safe.py` | Running. Log: `logs/input_random_fixed_50pct_adapter_head_clsres_safe_20260507_224015.log`. Epoch 3 finished and epoch 4 started. Loss trajectory: `2.2576 -> 1.5276 -> 0.9876 -> 0.8835`; latest `cls_loss=0.5232`, `reg_loss=0.3603`, `mem=6334MB`. |

Both new runs had the usual epoch-0 `cls_head.weight` non-finite-gradient skip at iter 4, which was already observed in baseline-family runs. No DDP "marked ready twice" crash was observed after the checkpoint fix.

## Next Readout

First meaningful validation remains epoch 40+ under the inherited schedule. The key decision gate is whether either safe residual path beats `63.77` and especially whether it crosses `65.0` Avg-mAP without hurting `mAP@0.7`.

## Next-Round Implementation: Adapter / ActionFormer Split

Added two stricter baseline-preserving variants for the next two-server run:

| Variant | Files | Contract |
|---|---|---|
| `input_random_fixed_50pct_adapter_multiscale_safe` | `vit_adapter.py`, `input_random_fixed_50pct_adapter_multiscale_safe.py` | Adds optional Adapter multi-scale temporal branches with dilations `[2, 4]`; `multiscale_scale` starts at `0.0` and is tanh-bounded by `scale_max=0.20`, so the branch is baseline-equivalent at init. |
| `input_random_fixed_50pct_adapter_head_regres_safe` | `anchor_free_head.py`, `actionformer.py`, `input_random_fixed_50pct_adapter_head_regres_safe.py` | Adds optional depthwise regression residual before ActionFormer `Scale/ReLU`; `reg_residual_scale` starts at `0.0` and is excluded from detector weight decay. |

Driver: `scripts/run_adapter_next_perf_pair.sh`.

Local verification:

| Check | Result |
|---|---|
| Contract tests | `python -m pytest tests/test_adapter_safety_contracts.py -q`: `10 passed`. |
| Python syntax | `py_compile` on modified Python/config files: PASS. |
| Whitespace | `git diff --check` on modified files: PASS. |
| GPT-5.5 xhigh static review: Adapter multiscale | No findings. Reviewer confirmed zero-gate baseline equivalence, `[B,N,C] -> [group,T,H,W,C]` reshape consistency, checkpoint safety, and config plumbing into `Adapter`. |
| GPT-5.5 xhigh static review: ActionFormer reg residual | No findings. Reviewer confirmed zero-gate baseline equivalence, train/test consistency, `Scale/ReLU` placement, and `reg_residual_scale` no-decay handling. |
| Config load / model build | Remote `CHECK_ONLY=1 bash scripts/run_adapter_next_perf_pair.sh`: PASS on both 35407 and 25876. Local environment still lacks `mmengine` and has broken local torch DLL loading. |
| Linux bash syntax | Remote `bash -n scripts/run_adapter_next_perf_pair.sh`: PASS on both 35407 and 25876. Local Windows Bash still returns `E_ACCESSDENIED`. |
| GPT-5.5 xhigh static review: Adapter multiscale post-fix | No blocking findings. Reviewer confirmed zero-gate baseline equivalence, RNG restore, DDP/checkpoint safety, shape consistency with TARA off, and noted `time_aligned_rasterizer.py` must be synced because `vit_adapter.py` imports it unconditionally. |
| GPT-5.5 xhigh static review: ActionFormer reg residual post-fix | No blocking findings. Reviewer confirmed zero-gate baseline equivalence, train/test consistency, `Scale/ReLU` placement, and `reg_residual_scale` no-decay handling. |
| Claude/Gemini MCP review retry | Both background review jobs failed before writing a final result, so no conclusions were used from them. |
| Gemini CLI retry | CLI exists, but direct `gemini` is blocked by PowerShell `.ps1` execution policy and `gemini.cmd --version` fails with `spawn EPERM`; no usable Gemini CLI review was produced. |

Deployment status: DNS/SSH recovered. Files were synced to both servers and remote check-only validation passed.

Review fixes applied after the second GPT-5.5 xhigh pass:

1. `_time_embed_scale_value` now uses `scale = scale * 0.0` during warmup instead of `scale.detach().new_zeros(())`, keeping `time_embed_scale` in the DDP static graph with zero gradient.
2. Adapter multiscale branch initialization now saves/restores `torch` RNG state, so enabling `adapter_multiscale_cfg` does not perturb baseline adapter initialization order.

Post-fix local verification:

| Check | Result |
|---|---|
| Contract tests | `python -m pytest tests/test_adapter_safety_contracts.py -q`: `10 passed`. |
| Python syntax | `py_compile` on modified Python/config files: PASS. |
| Whitespace | `git diff --check` on modified files: PASS. |

## Next-Round Remote Deployment: 2026-05-08

Synced file set to both `/root/autodl-tmp/OpenTAD_Back_check` servers:

- `opentad/models/backbones/vit_adapter.py`
- `opentad/models/backbones/time_aligned_rasterizer.py`
- `opentad/models/backbones/backbone_wrapper.py`
- `opentad/models/dense_heads/anchor_free_head.py`
- `opentad/models/detectors/actionformer.py`
- `opentad/models/losses/assigner/anchor_free_simota_assigner.py`
- `configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py`
- `configs/adatad/thumos/input_random_fixed_50pct_adapter.py`
- `configs/adatad/thumos/input_random_fixed_50pct_adapter_timegate_safe.py`
- `configs/adatad/thumos/input_random_fixed_50pct_adapter_head_clsres_safe.py`
- `configs/adatad/thumos/input_random_fixed_50pct_adapter_multiscale_safe.py`
- `configs/adatad/thumos/input_random_fixed_50pct_adapter_head_regres_safe.py`
- `tests/test_adapter_safety_contracts.py`
- `scripts/run_adapter_next_perf_pair.sh`
- `scripts/monitor_adapter_run.sh`
- `scripts/supervise_adapter_quality.sh`
- `scripts/watch_adapter_result.sh`

Remote checks passed on both servers:

| Server | Check evidence |
|---|---|
| `35407` | `py_compile`, `bash -n`, and `CHECK_ONLY=1 bash scripts/run_adapter_next_perf_pair.sh` passed. Config load printed `ActionFormer`, `projection_input_pdrop=0.0`, `checkpoint_interval=10`, `disable_checkpoint=False`, `end_epoch=60` for both new configs. |
| `25876` | Same `py_compile`, `bash -n`, and check-only validation passed with the same config evidence. |

Final run status:

| Server | Screen | Config | Status at 2026-05-09 01:09 +08 |
|---|---|---|---|
| `35407` | `adapter_multiscale_safe_35407` | `input_random_fixed_50pct_adapter_multiscale_safe.py` | Completed. `screen -ls` shows no sockets; GPU memory is `0/32760 MiB`. Final train log: `logs/input_random_fixed_50pct_adapter_multiscale_safe_20260508_080251.log`. Final metric: `Average-mAP=51.35`, `mAP@0.7=26.79`; delta vs `63.77` baseline is `-12.42`, delta vs `65.00` target is `-13.65`. |
| `25876` | `adapter_head_regres_safe_25876` | `input_random_fixed_50pct_adapter_head_regres_safe.py` | Completed. `screen -ls` shows no sockets; GPU memory is `0/32760 MiB`. Final train log: `logs/input_random_fixed_50pct_adapter_head_regres_safe_20260508_080258.log`. Final metric: `Average-mAP=51.64`, `mAP@0.7=26.11`; delta vs `63.77` baseline is `-12.13`, delta vs `65.00` target is `-13.36`. |

Driver logs:

- `35407`: `/root/autodl-tmp/OpenTAD_Back_check/logs/adapter_next_perf_pair_driver.log`
- `25876`: `/root/autodl-tmp/OpenTAD_Back_check/logs/adapter_next_perf_pair_driver.log`

Monitor screens and logs:

| Server | Monitor screen | Monitor log | Purpose |
|---|---|---|---|
| `35407` | `monitor_multiscale_safe_35407` | `/root/autodl-tmp/OpenTAD_Back_check/logs/monitor_multiscale_safe_35407.log` | Polls the multiscale Adapter training log every 15 minutes and records epoch/loss/mAP/error lines. |
| `25876` | `monitor_head_regres_safe_25876` | `/root/autodl-tmp/OpenTAD_Back_check/logs/monitor_head_regres_safe_25876.log` | Polls the regression-residual head training log every 15 minutes and records epoch/loss/mAP/error lines. |

Long-run quality supervisor logs:

| Server | Supervisor screen | Supervisor log | Final decision |
|---|---|---|---|
| `35407` | `supervisor_multiscale_safe_35407` | `/root/autodl-tmp/OpenTAD_Back_check/logs/supervisor_multiscale_safe_35407.log` | Ended after training completion. Final result is below baseline; no crash after startup. |
| `25876` | `supervisor_head_regres_safe_25876` | `/root/autodl-tmp/OpenTAD_Back_check/logs/supervisor_head_regres_safe_25876.log` | Ended after training completion. Final result is below baseline; no crash after startup. |

Result watcher logs:

| Server | Result watcher screen | Result watcher log | Final note |
|---|---|---|---|
| `35407` | `result_watch_multiscale_safe_35407` | `/root/autodl-tmp/OpenTAD_Back_check/logs/result_watch_multiscale_safe_35407.log` | Watcher ended at 15:15 and last recorded `49.89/25.26`, but the train log later recorded the final `51.35/26.79` and `Training Over`. Use the train log as source of truth. |
| `25876` | `result_watch_head_regres_safe_25876` | `/root/autodl-tmp/OpenTAD_Back_check/logs/result_watch_head_regres_safe_25876.log` | Watcher ended at 15:40 and last recorded `50.45/25.01`, but the train log later recorded the final `51.64/26.11` and `Training Over`. Use the train log as source of truth. |

Notification status: `~/.codex/feishu.json` is absent, so Feishu/Lark push notifications are disabled. Long-run supervision is file-log based through the remote screen watchers above.

Final metric curves:

| Run | Eval sequence: Avg-mAP / mAP@0.7 | Best / final verdict |
|---|---|---|
| `input_random_fixed_50pct_adapter_multiscale_safe` | `38.32/12.90`, `39.65/14.19`, `41.11/15.71`, `42.62/17.57`, `44.03/18.78`, `45.66/20.71`, `47.29/22.17`, `48.63/23.96`, `49.89/25.26`, `51.35/26.79` | Best is final `51.35/26.79`; below baseline by `-12.42` Avg-mAP. |
| `input_random_fixed_50pct_adapter_head_regres_safe` | `39.35/14.08`, `40.41/15.19`, `41.82/16.50`, `43.06/17.78`, `44.54/19.02`, `46.09/20.78`, `47.59/22.11`, `49.25/23.74`, `50.45/25.01`, `51.64/26.11` | Best is final `51.64/26.11`; below baseline by `-12.13` Avg-mAP. |

Completion notes:

- Both runs showed the known baseline-family epoch 0 iter 4 non-finite `module.rpn_head.cls_head.weight` gradient skip, then continued to complete epoch 0.
- No DDP static graph crash, no reentrant checkpoint "marked ready twice" crash, and no import/config failure was observed during startup.
- Both runs completed the inherited 60-epoch schedule and logged `Training Over`.
- No current GPU exposure remains on either server from these two runs.
- No new JSON result files were produced for these runs; training logs are the evidence source.
- Performance target was not reached. The best new result is the ActionFormer regression-residual run at `51.64` Avg-mAP.

Historical old-run progress before authorized stop, checked at 2026-05-08 08:00 +08:

| Server | Active old config | Latest evidence | Implication |
|---|---|---|---|
| `35407` | `input_random_fixed_50pct_adapter_virtual_baseline_native_axis.py` | Epoch 43 started. Epoch 41 validation: `Average-mAP=10.51`, `mAP@0.7=1.32`. | Very poor intermediate validation; roughly 17 epochs remain before the new multiscale Adapter queue can start naturally. |
| `25876` | `input_random_fixed_50pct_adapter_irregular_headv3_y_pdrop0.py` | Epoch 15 started. Latest epoch 14 loss: `1.3040` with `boundary_loss=0.6307`. | Long remaining runtime; this older Irregular/HeadV3 route is not the preferred mainline and blocks the new ActionFormer regression-residual queue. |

## Completion Audit

Objective restated as concrete deliverables:

1. Diagnose why current Adapter + ActionFormer variants underperform and identify a safe route toward 65+ Avg-mAP.
2. Use external model review/discussion where available.
3. Implement separate Adapter-side and ActionFormer-side improvements without switching detector family.
4. Verify implementation locally and remotely.
5. Deploy to both SSH servers.
6. Start experiments on both servers.

Prompt-to-artifact checklist:

| Requirement | Evidence | Status |
|---|---|---|
| Analyze failure causes | Diagnosis section above; bad-route table shows `pdrop`, `late_linear`, HeadV3/Irregular failures; oracle boundary result shows headroom. | Done |
| Plan safe improvement route | Next-round variants use zero-gated residual/multiscale branches and preserve Adapter + ActionFormer. | Done |
| Adapter-side implementation | `vit_adapter.py` plus `input_random_fixed_50pct_adapter_multiscale_safe.py`; zero gate, bounded scale, RNG restore. | Done |
| ActionFormer-side implementation | `anchor_free_head.py`, `actionformer.py`, `input_random_fixed_50pct_adapter_head_regres_safe.py`; regression residual before existing `Scale/ReLU`, no-decay gate. | Done |
| GPT-5.5 xhigh review | Two GPT-5.5 xhigh reviews returned no blocking findings; Adapter review explicitly checked RNG, DDP, shapes, sync dependency. | Done |
| Gemini/Claude retry | Gemini CLI and MCP/Claude MCP retries were attempted but failed without usable review output; failures recorded above. | Attempted, unavailable |
| Local verification | `pytest` 10 passed, `py_compile` PASS, `git diff --check` PASS. | Done |
| Remote verification | Both servers passed `py_compile`, `bash -n`, and `CHECK_ONLY=1 bash scripts/run_adapter_next_perf_pair.sh`. | Done |
| Remote deployment | Synced all required model/config/test/script files to both `/root/autodl-tmp/OpenTAD_Back_check` servers. | Done |
| Experiment startup | User authorized stopping old screens; both new configs entered `torchrun`, completed epoch 0, and started epoch 1. | Done |
| Performance validation | Both experiments completed. Adapter multiscale ended at `51.35/26.79`; ActionFormer regression residual ended at `51.64/26.11`; both are below the `63.77` baseline and `65.0` target. | Done, negative |

Current completion verdict: implementation, verification, deployment, startup, supervision, and final performance validation are complete. The target was not achieved; the two safe residual model-side variants underperformed the known Adapter + ActionFormer baseline by about 12 Avg-mAP. This strengthens the current diagnosis that the next credible path should prioritize boundary/input allocation while keeping the baseline Adapter + ActionFormer feature contract intact.

## Next Iteration: Train-Only Boundary Sampling, 2026-05-09

The model-side residual experiments were negative, so the next iteration moves the optimization pressure to input/boundary allocation while keeping the Adapter + ActionFormer model contract unchanged.

Cross-review summary:

| Reviewer | Conclusion |
|---|---|
| GPT-5.5 xhigh reviewer 1 | No evidence of a hard shape/mask bug in the failed residual runs. The likely issue is that zero-gated residuals still learn into a harmful region. Recommended train-only boundary/action-boundary weighted sampling and explicitly warned not to use GT-boundary sampling in val/test. |
| GPT-5.5 xhigh reviewer 2 | No hard implementation error found in safe residual code. The bottleneck is likely compact selected-axis assignment/regression under sparse input. Recommended train-only action-boundary weighted sampling and, as a later ActionFormer-side option, SimOTA center25/mink4/w1. |

Implemented new configs:

| Config | Change | Fairness guard |
|---|---|---|
| `input_random_fixed_50pct_adapter_train_boundary_weighted.py` | Train pipeline uses `weighted_random_boundary_subsample` with boundary/action/background weights `4.0/1.0/1.0`. | Inherits baseline val/test `random_fixed_subsample`; no GT-boundary eval leakage. |
| `input_random_fixed_50pct_adapter_train_action_boundary_weighted.py` | Train pipeline uses `weighted_random_action_boundary_subsample` with boundary/action/background weights `4.0/2.0/1.0`. | Inherits baseline val/test `random_fixed_subsample`; no GT-boundary eval leakage. |
| `scripts/run_adapter_boundary_train_pair.sh` | Two-run queue with config checks for `ActionFormer`, `VisionTransformerAdapter`, `input_pdrop=0.0`, train weighted sampling, and val/test random-fixed sampling. | `CHECK_ONLY=1` passed on both servers before launch. |

Verification:

| Scope | Result |
|---|---|
| Local contract tests | `python -m pytest tests/test_adapter_safety_contracts.py -q`: `11 passed`. |
| Local Python syntax | `py_compile` on the two new configs: PASS. |
| Local whitespace check | `git diff --check` on the new config/script/test files: PASS. |
| Remote syntax/check-only | Both `35407` and `25876` passed `py_compile`, `bash -n scripts/run_adapter_boundary_train_pair.sh`, and `CHECK_ONLY=1 bash scripts/run_adapter_boundary_train_pair.sh`. |
| Remote sample contract | On `35407`, both train configs produced `inputs` shape `(1, 3, 384, 160, 160)`, `masks.sum()=384`, valid `gt_segments` within `0..384`, and monotonic `irregular_selected_positions` of length `384` over a `768` dense window. |

Launch status:

| Server | Screen | Config | Log | Current status |
|---|---|---|---|---|
| `35407` | `adapter_train_boundary_35407` | `input_random_fixed_50pct_adapter_train_boundary_weighted.py` | `logs/input_random_fixed_50pct_adapter_train_boundary_weighted_20260509_011529.log` | Running. Epoch 0 finished at 01:19:57 with `Loss=2.2874`, `cls_loss=1.2917`, `reg_loss=0.9957`; epoch 1 started. GPU memory about `8.25 GiB`. |
| `25876` | `adapter_train_action_boundary_25876` | `input_random_fixed_50pct_adapter_train_action_boundary_weighted.py` | `logs/input_random_fixed_50pct_adapter_train_action_boundary_weighted_20260509_011529.log` | Running. Epoch 0 finished at 01:20:04 with `Loss=2.2905`, `cls_loss=1.2953`, `reg_loss=0.9952`; epoch 1 started. GPU memory about `8.25 GiB`. |

Monitor screens:

| Server | Monitor | Supervisor | Result watcher |
|---|---|---|---|
| `35407` | `monitor_train_boundary_35407` | `supervisor_train_boundary_35407` | `result_watch_train_boundary_35407` |
| `25876` | `monitor_train_action_boundary_25876` | `supervisor_train_action_boundary_25876` | `result_watch_train_action_boundary_25876` |

Startup note: both runs showed the known baseline-family epoch-0 iter-4 non-finite `module.rpn_head.cls_head.weight` gradient skip and then continued. No hard failure, OOM, DDP static-graph crash, or config/import failure is visible so far.

Decision gate: first meaningful validation remains epoch 40+. If either run plateaus near the previous failed residual band around `50-52` Avg-mAP, stop investing in that sampling ratio. If either approaches or exceeds the `63.77` baseline, follow with ratio/radius ablations and a matched baseline rerun.

## First Eval Gate: Train-Only Boundary Sampling, 2026-05-09

Both train-only boundary sampling runs reached the first inherited validation gate around epoch 41/42.

| Server | Config | First eval time | Avg-mAP | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | Delta vs `63.77` baseline | Delta vs `65.00` target | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `35407` | `input_random_fixed_50pct_adapter_train_boundary_weighted.py` | 2026-05-09 04:30:32 +08 | 37.90 | 60.84 | 50.94 | 38.94 | 25.88 | 12.89 | -25.87 | -27.10 | Below baseline |
| `25876` | `input_random_fixed_50pct_adapter_train_action_boundary_weighted.py` | 2026-05-09 04:35:08 +08 | 34.69 | 58.44 | 46.55 | 34.85 | 22.58 | 11.02 | -29.08 | -30.31 | Below baseline |

Current runtime state at the first-eval check:

| Server | Screen state | GPU state | Latest progress |
|---|---|---|---|
| `35407` | `adapter_train_boundary_35407` and monitor/supervisor/result watcher screens still detached | About `8265 / 32760 MiB`, `0%` util at poll instant | Epoch 43 started after first eval |
| `25876` | `adapter_train_action_boundary_25876` and monitor/supervisor/result watcher screens still detached | About `8263 / 32760 MiB`, `0%` util at poll instant | Epoch 43 started after first eval |

Interpretation:

- The train-only boundary weighting does not produce an early positive signal. It is far below the Adapter + ActionFormer random-fixed baseline (`63.77/42.19`) and also below the first-eval band of the failed safe-residual model-side runs (`38.32/12.90` and `39.35/14.08`).
- The action+boundary weighting is worse than boundary-only (`34.69` vs `37.90` Avg-mAP), so increasing action-frame bias does not rescue this sampling strategy.
- The most likely failure mode is train/test sampling distribution mismatch: training sees GT-biased boundary/action selections, while validation still uses random-fixed sparse sampling. That improves neither the deployed evaluation protocol nor the ActionFormer regression/classification calibration on randomly selected sparse positions.
- The low high-IoU numbers (`11.02-12.89 @0.7`) indicate boundary localization is not improved; the model is not merely trading low-IoU recall for sharper localization.
- Continuing these exact ratios is unlikely to reach the `65+` goal. At best they may climb with later epochs, similar to the previous residual runs, but the first gate is too far below baseline to justify more GPU unless the goal is diagnostic curve collection.

Recommended decision:

- Stop both train-only weighted sampling runs to release GPUs, unless a full negative curve is needed for documentation.
- Do not pursue stronger GT-biased train-only ratios such as `boundary/action/background = 8/2/1`; the current evidence says the mismatch itself is the problem.
- Next credible route should keep train and eval sampling distributions aligned. Options worth testing are: matched non-GT boundary-aware sampling that is computable at eval time, a mild deterministic/random hybrid that preserves random-fixed coverage, or ActionFormer-side assignment/loss changes that do not alter the Adapter feature contract.

## Next Launch: Uniform 50% Adapter + ActionFormer, 2026-05-09

After the train-only GT-weighted sampling failure, the next run returns to an aligned train/eval geometry. The new primary hypothesis is that a regular stride-2 input protocol preserves the dense ActionFormer temporal contract better than random-fixed sparse positions, while still staying inside the Adapter + ActionFormer family.

New files:

| File | Purpose |
|---|---|
| `configs/adatad/thumos/input_stride2_uniform_50pct_adapter.py` | Adapter + ActionFormer, `sample_stride=2`, `window_size=384`, train `random_trunc`, val/test `sliding_window`. |
| `configs/adatad/thumos/input_stride2_uniform_50pct_adapter_center25.py` | Same stride-2 Adapter run, but with ActionFormer `center_sample_radius=2.5` and assignment diagnostics enabled. |
| `scripts/run_adapter_uniform_pair.sh` | Two-config queue and strict `CHECK_ONLY` guard for model family, stride/window, `t1=24`, projection/head/neck type, augmentation chain, and zero `input_pdrop`. |
| `tests/test_adapter_safety_contracts.py` | Extended contract tests for the uniform stride-2 Adapter configs and launch script. |

Cross-review:

| Reviewer | Result |
|---|---|
| GPT-5.5 xhigh model-logic review | Supported `input_stride2_uniform_50pct_adapter.py` as the cleanest high-probability Adapter + ActionFormer route. Warned that the historical `65.09` was from a non-adapter/frozen family, so the valid claim must be against the Adapter random-fixed `63.77/42.19` baseline. Treated center25 as exploratory, not proof. |
| GPT-5.5 xhigh code/config review | Found no experiment-invalidating issue. Suggested strengthening `CHECK_ONLY` with projection/head/neck type, `t1=24`, and augmentation assertions; those checks were added before deployment. |
| Gemini CLI | Retry failed with `Background review worker exited before writing a final result`; no usable Gemini critique was available for this iteration. |

Verification:

| Scope | Result |
|---|---|
| Local pytest | `python -m pytest tests/test_adapter_safety_contracts.py -q`: `12 passed`. |
| Local py_compile | New configs passed `python -m py_compile`. |
| Local whitespace | `git diff --check` passed for the new configs, launch script, and test file. |
| Remote `35407` | `/root/miniconda3/bin/python -m py_compile`, `bash -n scripts/run_adapter_uniform_pair.sh`, and `CHECK_ONLY=1 bash scripts/run_adapter_uniform_pair.sh` passed. Config evidence: `ActionFormer`, `VisionTransformerAdapter`, `sample_stride=2`, train `random_trunc`, val `sliding_window`, `center_sample_radius=1.5/2.5`, `input_pdrop=0.0`, `disable_checkpoint=True`, `end_epoch=60`. |
| Remote `25876` | Same remote checks passed. Disk state before launch: `173G/200G` used, `28G` free. |

Launch:

| Server | Screen | Config | Log | Watchers |
|---|---|---|---|---|
| `35407` | `adapter_uniform_35407` | `input_stride2_uniform_50pct_adapter.py` | `logs/input_stride2_uniform_50pct_adapter_20260509_083003.log` | `supervisor_uniform_35407`, `result_watch_uniform_35407` |
| `25876` | `adapter_uniform_center25_25876` | `input_stride2_uniform_50pct_adapter_center25.py` | `logs/input_stride2_uniform_50pct_adapter_center25_20260509_083004.log` | `supervisor_uniform_center25_25876`, `result_watch_uniform_center25_25876` |

Startup status:

- Both screens entered training and started epoch 0.
- GPU memory after startup was about `8.25 GiB` on both servers.
- Both runs showed the known baseline-family epoch-0 iter-4 non-finite `module.rpn_head.cls_head.weight` gradient skip. This is tracked but not considered a hard failure because prior baseline-family runs continued after the same signature.
- First meaningful eval is expected at the inherited validation gate around epoch 39/40. Result watcher baselines are set to `63.77` Avg-mAP and target `65.00`.

Decision gate:

- If `input_stride2_uniform_50pct_adapter.py` beats `63.77`, it becomes the new Adapter + ActionFormer mainline and should be rerun once for variance.
- If it reaches or exceeds `65.00`, it supports the user's 65+ target under the strict Adapter + ActionFormer constraint.
- If center25 helps over the stride-2 default, follow with a matched random-fixed center25 control to isolate whether the gain comes from assignment or from uniform geometry.

## Completion: Uniform 50% Adapter + ActionFormer, 2026-05-09

Both uniform stride-2 Adapter experiments completed and released GPU memory. The train logs are the source of truth; result watcher logs missed the final eval because the training screen ended before the next polling interval.

| Server | Config | Final Avg-mAP | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | Delta vs Adapter random-fixed `63.77/42.19` | Delta vs target `65.00` | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `35407` | `input_stride2_uniform_50pct_adapter.py` | 54.34 | 73.68 | 67.30 | 56.96 | 44.75 | 29.03 | -9.43 Avg / -13.16 @0.7 | -10.66 | Completed, negative |
| `25876` | `input_stride2_uniform_50pct_adapter_center25.py` | 53.60 | 73.09 | 65.73 | 56.99 | 44.32 | 27.89 | -10.17 Avg / -14.30 @0.7 | -11.40 | Completed, negative |

Metric curves:

| Run | Eval sequence: Avg-mAP / mAP@0.7 |
|---|---|
| `input_stride2_uniform_50pct_adapter` | `41.10/15.42`, `44.27/17.87`, `47.75/21.49`, `51.19/25.06`, `54.34/29.03` |
| `input_stride2_uniform_50pct_adapter_center25` | `40.02/14.75`, `43.05/17.27`, `46.71/20.60`, `50.65/24.99`, `53.60/27.89` |

Runtime closure:

| Server | Screen | GPU state | Evidence |
|---|---|---|---|
| `35407` | No sockets | `0 / 32760 MiB`, `0%` util | `Training Over...` at 2026-05-09 14:12:44 +08 |
| `25876` | No sockets | `0 / 32760 MiB`, `0%` util | `Training Over...` at 2026-05-09 14:25:02 +08 |

Interpretation:

- Uniform stride-2 is not a win under the current Adapter training recipe. It is better than the disastrous train-only GT weighted first evals, but still far below the strong Adapter random-fixed baseline (`63.77/42.19`).
- Center radius `2.5` slightly hurts the final result (`53.60` vs `54.34`) and high-IoU localization (`27.89` vs `29.03`). This does not support widening FCOS center assignment as the next ActionFormer-side lever.
- The historical `input_stride2_uniform_baseline = 65.09` is not directly transferable to Adapter. That prior used the frozen `VisionTransformerCP` family with its own projection/post-processing recipe; the current run uses `VisionTransformerAdapter`, adapter fine-tuning, and no `input_pdrop=0.2`.
- The failure mode is therefore not "regular 50% sampling is bad." It is more specific: the Adapter fine-tuning contract that works for random-fixed selected-rank input does not recover the frozen stride-2 behavior when the dataset-level `sample_stride` is changed to 2.
- The result narrows the credible path: the best known Adapter + ActionFormer mainline is still `input_random_fixed_50pct_adapter = 63.77/42.19`; the 65+ path likely requires improving that exact random-fixed protocol or using an eval-computable boundary proxy, not switching to dataset-level stride-2.

Recommended next step:

- Do not continue center-radius widening on stride-2.
- Treat frozen stride-2 `65.09` as an external reference only, not as a direct Adapter target.
- Next experiments should keep the Adapter random-fixed protocol intact and target one of two narrower hypotheses:
  1. improve ActionFormer calibration on random-fixed selected-rank inputs without changing sampling, or
  2. add an eval-computable boundary proxy to sampling so train/eval remain aligned and no GT boundary information leaks.

## Postmortem Follow-Up: NMS Calibration and Frozen-Recipe Control, 2026-05-09

Two GPT-5.5 xhigh postmortem agents reviewed the negative stride-2 result.

| Reviewer focus | Recommendation |
|---|---|
| ActionFormer-side low-risk control | Run post-processing-only NMS calibration on the strongest checkpointed Adapter baseline proxy. |
| Stride-2 postmortem | Run one stride-2 control that keeps Adapter + ActionFormer but matches the frozen stride-2 data/post-processing recipe more closely. |

New files:

| File | Purpose |
|---|---|
| `configs/adatad/thumos/input_random_fixed_50pct_adapter_virtual_baseline_nms_sigma05_minscore001.py` | Test-only NMS calibration on the checkpointed `virtual_baseline` proxy: `sigma=0.5`, `min_score=0.001`, `pre_nms_topk=2000`. |
| `configs/adatad/thumos/input_stride2_uniform_50pct_adapter_frozen_recipe.py` | Stride-2 Adapter control with frozen-style train crop (`Resize(-1,160)` + `CenterCrop(160)`) and frozen-style NMS (`sigma=0.5`, `min_score=0.001`). |
| `scripts/run_adapter_postmortem_next.sh` | Shared launcher with strict `CHECK_ONLY` guards and sliced execution for NMS eval vs frozen-recipe training. |
| `tests/test_adapter_safety_contracts.py` | Extended to 13 tests covering the new controls. |

Verification:

| Scope | Result |
|---|---|
| Local pytest | `python -m pytest tests/test_adapter_safety_contracts.py -q`: `13 passed`. |
| Local py_compile | New configs passed `python -m py_compile`. |
| Local whitespace | `git diff --check` passed for new configs, launcher, and tests. |
| Remote `35407` | `py_compile`, `bash -n`, and `CHECK_ONLY=1 bash scripts/run_adapter_postmortem_next.sh` passed. |
| Remote `25876` | Same remote checks passed. |

NMS-only result on `35407`:

| Config | Checkpoint | Avg-mAP | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | Delta vs same `virtual_baseline` final `63.84/41.91` | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `input_random_fixed_50pct_adapter_virtual_baseline_nms_sigma05_minscore001.py` | `input_random_fixed_50pct_adapter_virtual_baseline/gpu1_id0/checkpoint/epoch_59.pth` | 63.75 | 80.40 | 74.53 | 66.24 | 56.33 | 41.27 | -0.09 Avg / -0.64 @0.7 | No gain |

NMS interpretation:

- Tighter Soft-NMS (`sigma=0.5`) plus explicit `min_score=0.001` does not explain the gap to 65+. It slightly raises loose-IoU AP (`0.3/0.4`) but hurts high-IoU localization and does not improve Avg-mAP.
- This rules out easy post-processing-only gain as the main route. Keep the original Adapter post-processing for now unless a broader NMS grid is run from saved raw predictions.
- The exact `input_random_fixed_50pct_adapter` run did not save a checkpoint (`disable_checkpoint=True`), so this was run on `virtual_baseline`, which is a checkpointed proxy with nearly identical final score (`63.84` vs `63.77`).

Frozen-recipe control launch:

| Server | Screen | Config | Log | Current status at 2026-05-09 21:36 +08 |
|---|---|---|---|---|
| `25876` | `stride2_frozen_recipe` | `input_stride2_uniform_50pct_adapter_frozen_recipe.py` | `logs/input_stride2_uniform_50pct_adapter_frozen_recipe_20260509_211652.log` | Running, epoch 6 started, latest logged loss `0.7720`, one epoch-0 non-finite gradient skip but no crash. |

Decision gate:

- If frozen-recipe stride-2 still lands near `54/29`, fixed uniform stride-2 should be retired for Adapter.
- If it recovers toward `63+`, the previous stride-2 failure was mostly recipe/post-processing confounding.
- If it exceeds `65`, it becomes the new strict Adapter + ActionFormer candidate and should be rerun for variance.

Frozen-recipe first-eval result and stop:

| Server | Config | First eval Avg-mAP | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | Action |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `25876` | `input_stride2_uniform_50pct_adapter_frozen_recipe.py` | 42.82 | 63.35 | 53.57 | 42.43 | 38.32 | 16.44 | Stopped after first eval; GPU released. |

Interpretation:

- Frozen-style crop/NMS does not recover the historical non-adapter stride-2 `65.09` behavior under `VisionTransformerAdapter`.
- Uniform stride-2 should now be treated as a negative control for Adapter, not a main route to 65+.
- The failure remains input/contract specific: changing the dataset-level geometry away from random-fixed selected-rank inputs damages the Adapter + ActionFormer path.

## ActionFormer-Only Follow-Up: Regression Loss Calibration, 2026-05-09

Because NMS-only calibration did not improve the checkpointed random-fixed Adapter proxy, the next ActionFormer-side experiment keeps the successful random-fixed Adapter protocol intact and changes only the ActionFormer regression/classification balance.

New files:

| File | Purpose |
|---|---|
| `configs/adatad/thumos/input_random_fixed_50pct_adapter_regloss15.py` | Inherits `input_random_fixed_50pct_adapter.py`, sets `rpn_head.loss_weight=1.5`, and enables checkpoints. |
| `scripts/run_adapter_actionformer_regloss.sh` | Single-config launcher with `CHECK_ONLY` guards for Adapter + ActionFormer contract, random-fixed train/val/test sampling, zero `input_pdrop`, and checkpointing. |
| `tests/test_adapter_safety_contracts.py` | Extended to 14 tests covering the regression-loss calibration config and launcher. |

Verification:

| Scope | Result |
|---|---|
| Local pytest | `python -m pytest tests/test_adapter_safety_contracts.py -q`: `14 passed`. |
| Local py_compile | `input_random_fixed_50pct_adapter_regloss15.py` passed `python -m py_compile`. |
| Local whitespace | `git diff --check` passed for the new config, launcher, and test file. |
| Remote `35407` | `py_compile`, `bash -n`, and `CHECK_ONLY=1 bash scripts/run_adapter_actionformer_regloss.sh` passed. Evidence: `ActionFormer`, `VisionTransformerAdapter`, `ActionFormerHead`, `loss_weight=1.5`, train `random_fixed_subsample/random_trunc`, val `random_fixed_subsample/sliding_window`, `checkpoint_interval=10`, `disable_checkpoint=False`. |

Launch:

| Server | Screen | Config | Log | Initial status |
|---|---|---|---|---|
| `35407` | `adapter_regloss15_35407` | `input_random_fixed_50pct_adapter_regloss15.py` | `logs/input_random_fixed_50pct_adapter_regloss15_20260509_221634.log` | Running, epoch 0 started, GPU about `8243 MiB`. |

Decision gate:

- First useful signal is at epoch 40, comparable to the baseline's first eval `61.95/39.03`.
- A positive signal should preserve or improve Avg-mAP while lifting high-IoU localization (`mAP@0.7`).
- If it underperforms the baseline at the first gate, do not escalate to larger `loss_weight=2.0` without diagnosing whether classification recall collapsed.

First-eval result and stop:

| Server | Config | Eval trigger | Avg-mAP | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | Delta vs baseline first eval `61.95/39.03` | Action |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `35407` | `input_random_fixed_50pct_adapter_regloss15.py` | After epoch 41 | 37.64 | 60.32 | 50.53 | 38.25 | 26.00 | 13.11 | -24.31 Avg / -25.92 @0.7 | Stopped after first eval; GPU released. |

Interpretation:

- `rpn_head.loss_weight=1.5` is not a safe ActionFormer-only calibration lever in this recipe. It preserves train stability but collapses first validation quality.
- The result is worse than the baseline first-eval band and even near the failed train-only GT-boundary sampling band, so larger regression weights such as `2.0` should not be launched.
- The high-IoU collapse (`13.11` vs baseline first-eval `39.03`) suggests the regression/classification balance is degrading proposal ranking or localization calibration rather than fixing the boundary-evidence gap.

## Input Strategy Review: Is Random-Fixed the Right Final Input?, 2026-05-09

The user asked whether the current random input is the most reasonable route, and what input strategy should ultimately be implemented. Three GPT-5.5 xhigh reviewers were dispatched with separate roles: sampling theory, code feasibility, and experimental route critique.

| Reviewer | Focus | Main conclusion |
|---|---|---|
| Euler | Sampling theory | `random-fixed` is the strongest current non-oracle mainline, but not theoretically final. The final method should be train/eval-consistent fixed-budget input with global coverage plus non-GT boundary densification. |
| Raman | Experimental route critique | Do not treat random as the final answer, but also do not chase oracle or train-only GT boundary sampling. Main route should be pseudo-boundary hybrid sampling; fallback should be stratified random-fixed. |
| Lagrange | Code feasibility | Uniform stride-2 is the simplest leakage-free implementation in the current code, and oracle/weighted boundary samplers are not valid final methods because they use GT. This is a code-safety view, not an empirical recommendation. |

Synthesis:

- `random-fixed` is currently the correct fair baseline and the safest non-oracle production candidate because it does not use GT for point selection, keeps train/eval aligned, and preserves the Adapter + ActionFormer feature contract.
- `random-fixed` is not the ideal final input. The oracle boundary result (`77.62/66.91`) shows large headroom specifically around boundary evidence, while NMS-only calibration (`63.75/41.27`) shows this is not mainly a post-processing problem.
- Fixed uniform stride-2 is code-simple and leakage-free, but the completed Adapter result (`54.34/29.03`) is too negative to use as the main route unless the currently running frozen-recipe control recovers strongly.
- GT-driven oracle/weighted boundary samplers must remain upper-bound or diagnostic tools. They construct sampling groups from `gt_segments`, and some historical configs even set test pipelines to include GT; they are not valid final input methods.
- Train-only GT boundary weighting is empirically invalid under the current setup because it creates train/eval input mismatch and produced very low first-eval scores (`37.90/12.89`, `34.69/11.02`).

Recommended final input direction:

1. **Primary final target: pseudo-boundary hybrid fixed-budget sampling.**
   - Keep the same 50% budget.
   - Allocate part of the budget to global/random or stratified coverage.
   - Allocate part of the budget to boundary candidates generated without GT leakage, e.g. from a frozen first-stage model, a lightweight boundary/actionness predictor, or cached baseline predictions.
   - Use the same sampler at train/val/test, with the same coordinate remapping and mask logic.

2. **Fallback if pseudo-boundaries are too noisy or expensive: stratified random-fixed.**
   - Keep random selection, but force temporal bucket coverage to reduce accidental boundary misses.
   - No GT, no learned first stage, and minimal code surface.
   - This is the most conservative input-only implementation if pseudo-boundary generation is not ready.

3. **Retire or demote: uniform stride-2, center25, train-only GT boundary weighting, and NMS-only tuning.**
   - Uniform stride-2 remains only a control pending the frozen-recipe result.
   - Center25 and train-only boundary weighting are already negative.
   - NMS-only has been ruled out as a main source of 65+ gain.

Minimum next input experiment after current runs:

| Candidate | Why | Success gate | Failure gate |
|---|---|---|---|
| Pseudo-boundary hybrid sampler | Tests the real oracle gap without GT leakage | Final at least `65.5/44.0`, or first eval clearly above baseline first eval `61.95/39.03` | Below `63.0/41.0`, or any val/test GT use invalidates the run |
| Stratified random-fixed sampler | Conservative no-leak fallback if pseudo-boundary source is not ready | Beats or preserves `63.77/42.19` while improving `@0.7` | Falls toward stride-2 band (`55/30`) or shows worse boundary coverage |

Practical decision:

- `stride2_frozen_recipe` first eval was negative (`42.82/16.44`) and was stopped; do not spend more effort on uniform stride-2 as the main Adapter route.
- `regloss15` first eval was negative (`37.64/13.11`) and was stopped; do not escalate regression-loss weights.
- For the next input-method implementation, prefer pseudo-boundary hybrid if a non-GT boundary score source can be produced cheaply; otherwise implement stratified random-fixed first as the smallest no-leak input improvement.

## Next Input Experiment: Stratified Random-Fixed, 2026-05-10

The next implementation follows the conservative fallback from the input-strategy review. It keeps the same 50% budget, Adapter + ActionFormer model contract, compact selected-axis GT remapping, and train/val/test alignment, but changes the random-fixed selector from unconstrained global random sampling to bucketed random sampling.

New implementation:

| File | Purpose |
|---|---|
| `opentad/datasets/transforms/end_to_end.py` | Adds `stratified_random_fixed_subsample` and `_select_stratified_random_fixed_positions`. The existing `random_fixed_subsample` sample key remains `random_fixed`, preserving the known `63.77/42.19` baseline protocol. |
| `configs/adatad/thumos/input_random_fixed_50pct_adapter_stratified.py` | Frame-level stratified random-fixed Adapter + ActionFormer run. Train uses `random_trunc`; val/test use `sliding_window`; all splits use `stratified_random_fixed_subsample`. |
| `configs/adatad/thumos/input_random_fixed_50pct_adapter_stratified_tubelet2.py` | Same experiment, but selects `tubelet` units of size 2 before expansion. |
| `scripts/run_adapter_stratified_pair.sh` | Two-config launcher with strict `CHECK_ONLY` guards for Adapter, ActionFormer, `input_pdrop=0.0`, train/val/test sampler alignment, checkpointing, and split slicing. |
| `tests/test_adapter_safety_contracts.py` | Extended to 15 tests, including no-GT stratified sampler/config/launcher contracts. |

Review and verification:

| Check | Result |
|---|---|
| GPT-5.5 xhigh explorer review | No blocking findings. Confirmed the minimal design: no GT input to selector, deterministic per video/window/method/unit key, one random unit per temporal bucket, sorted/unique positions, compact-axis GT remapping, and preservation of the old `random_fixed` sample key. |
| Gemini/Claude MCP review retry | Both background jobs failed before writing final results, so no conclusions were used from them. |
| Local TDD gate | Added failing contract test first; initial failure was missing config/implementation, then implementation made the test pass. |
| Local pytest | `python -m pytest tests/test_adapter_safety_contracts.py -q`: `15 passed`. |
| Local py_compile | `end_to_end.py`, `input_random_fixed_50pct_adapter_stratified.py`, and `input_random_fixed_50pct_adapter_stratified_tubelet2.py`: PASS. |
| Local bash syntax | `bash -n scripts/run_adapter_stratified_pair.sh`: PASS. |
| Local whitespace | `git diff --check` for touched files: PASS, with only Git LF/CRLF warnings. |
| Remote `35407` | Synced minimal file set. `py_compile`, `bash -n`, and `CHECK_ONLY=1 bash scripts/run_adapter_stratified_pair.sh` passed. Evidence: train/val/test all use `stratified_random_fixed_subsample`; frame config uses default frame selection; tubelet config uses `selection_unit=tubelet`, size 2. |
| Remote `25876` | Same sync and checks passed. Disk had `28G` free under `/root/autodl-tmp`, enough for logs/checkpoints but should be watched. |
| Remote selector behavior smoke | Passed on both servers. The smoke instantiated `LoadFrames`, verified frame-level `768 -> 384` deterministic stratified selection with one point per bucket, and verified tubelet2 expansion to 384 sorted frame positions in consecutive pairs. |

Launch status:

| Server | Screen | Config | Log | Initial status |
|---|---|---|---|---|
| `35407` | `adapter_stratified_frame_35407` | `input_random_fixed_50pct_adapter_stratified.py` | `logs/input_random_fixed_50pct_adapter_stratified_20260510_082744.log` | Running. Epoch 2 completed with `Loss=0.9250`, then epoch 3 started. GPU about `8255 MiB`. |
| `25876` | `adapter_stratified_tubelet2_25876` | `input_random_fixed_50pct_adapter_stratified_tubelet2.py` | `logs/input_random_fixed_50pct_adapter_stratified_tubelet2_20260510_082744.log` | Running. Epoch 2 completed with `Loss=0.9242`, then epoch 3 started. GPU about `8255 MiB`. |

Startup note:

- Both runs showed the known baseline-family epoch 0 iter 4 non-finite `module.rpn_head.cls_head.weight` gradient skip, then continued normally to epoch 1.
- No config import error, DDP startup crash, CUDA OOM, or missing-file failure was observed during startup.

Monitor screens:

| Server | Monitor screen | Monitor log | Cadence |
|---|---|---|---|
| `35407` | `monitor_stratified_frame_35407` | `logs/monitor_stratified_frame_35407.log` | Every 15 minutes until the training screen exits. |
| `25876` | `monitor_stratified_tubelet2_25876` | `logs/monitor_stratified_tubelet2_25876.log` | Every 15 minutes until the training screen exits. |

Decision gate:

- First useful eval should occur after the inherited epoch-40/41 validation gate.
- Compare first eval against baseline first eval `61.95/39.03@0.7`.
- If either run is clearly below `63.0/41.0` after the first validation, stop it rather than spending the full schedule.
- If frame-level stratified preserves baseline but tubelet2 hurts, keep frame-level as the only stratified branch.
- If either exceeds or approaches `65`, rerun once for variance and then move to pseudo-boundary hybrid using the same train/eval-aligned contract.

Pseudo-boundary feasibility note:

- Existing raw predictions were found under `input_random_fixed_50pct_adapter_virtual_baseline_nms_sigma05_minscore001/gpu1_id0/outputs`: `211` `.pkl` files, about `8MB` total.
- A sample file contains `[rpn_proposals, rpn_scores]` with shapes `[500, 2]` and `[500, 20]`, so the tensor format is suitable as a boundary-candidate source.
- However, `save_predictions` writes `f"{video_name}.pkl"` and the code comment says it should not be used for sliding-window because multiple windows for the same video overwrite the same file. Therefore this cache is not a complete pseudo-boundary source for train/eval-aligned sampling.
- The same config has `post_processing.save_dict=False`, so no `result_detection.json` exists. A proper pseudo-boundary route should first export post-processed predictions or change raw prediction saving to include window identity, then build a no-GT sampler from that cache.

## Pseudo-Boundary Snap Failure and Dense-Contract Relaunch, 2026-05-11

The pseudo-boundary snap branch was intended as a safer variant of direct boundary clustering: start from random-fixed positions, then locally snap a limited number of samples to nearby teacher-predicted boundaries. It still failed decisively.

Final snap results:

| Server | Config | Final observed Avg-mAP | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `35407` | `input_random_fixed_50pct_adapter_pseudo_boundary_snap_q32.py` | 51.23 | 70.31 | 63.84 | 54.18 | 41.83 | 25.97 | Training exited; GPU released. |
| `25876` | `input_random_fixed_50pct_adapter_pseudo_boundary_snap_q64.py` | 51.13 | 71.90 | 64.22 | 53.50 | 40.72 | 25.32 | Training exited; GPU released. |

Comparison:

- Random-fixed Adapter baseline remains about `63.77 Avg / 42.19@0.7`.
- Snap q32/q64 underperform by about `-12.5 Avg` and `-16.5@0.7`.
- Direct pseudo-boundary clustered q64/q96 was even worse (`~40-42 Avg`), so both hard clustering and local snapping are negative under the current input-sampling formulation.

Cross-review and discussion:

| Source | Status | Main useful conclusion |
|---|---|---|
| GPT-5.5 xhigh read-only review | Completed | Pseudo-boundary `train/val/test` all use teacher-conditioned frame selection, so evaluation itself is driven by teacher endpoint noise. The cache is post-NMS detection endpoints, not a calibrated dense boundary heatmap. Stop using pseudo-boundary scores to select validation/test inputs. |
| Gemini CLI `gemini-3-pro-preview` | Partially completed, then crashed with `ECONNRESET` while fetching Cloud Code experiments | Useful theoretical point: non-uniform/irregular sampling breaks ActionFormer's translation-invariant dense grid assumptions. However, the CLI session was unstable and cannot be treated as a clean review artifact. |
| Claude CLI method discussion | Completed as a plan file | Suggested QFL + asymmetric-boundary SimOTA, but this conflicts with current evidence that aligned SimOTA was slightly below baseline. Treat as a longer-term ActionFormer-head idea, not the immediate GPU route. |
| Claude CLI pre-launch review | Attempted twice | First run wrote only a plan and requested narrower scope; second narrowed read-only run returned no printed findings. No positive go/no-go decision was taken from Claude for launch. |

Root-cause synthesis:

- The oracle GT-boundary result (`77.62/66.91`) proves boundary evidence is valuable, but this does not imply that noisy teacher endpoints should drive sparse input selection.
- Current pseudo-boundary caches come from post-processed detection endpoints. They are sparse, NMS-biased, and score-calibrated for proposals, not for frame selection.
- Even with coordinate remapping, ActionFormer still learns over a dense selected-rank grid. Teacher-conditioned non-uniform selection changes the temporal support and context distribution, so localization and ranking degrade badly.
- Therefore, the next route should preserve the standard dense Adapter + ActionFormer contract and seek small gains over the stable `63.77` baseline, rather than further changing validation/test input geometry.

Protocol updates:

| Commit | Purpose |
|---|---|
| `be4a25c` | Added `AGENTS.md` with the Claude discussion/review protocol. |
| `ad0ed85` | Added the two current SSH server endpoints and remote work directory to `AGENTS.md`. |

New dense-contract experiments launched:

| Server | Screen | Config | Log | Monitor | Initial status |
|---|---|---|---|---|---|
| `35407` | `adapter_stride2_35407` | `configs/adatad/thumos/input_stride2_uniform_50pct_adapter.py` | `logs/input_stride2_uniform_50pct_adapter_20260511_1100.log` | `monitor_stride2_35407` | Running. Epoch 0 started. GPU about `8247 MiB`. |
| `25876` | `adapter_multiscale_25876` | `configs/adatad/thumos/input_random_fixed_50pct_adapter_multiscale_safe.py` | `logs/input_random_fixed_50pct_adapter_multiscale_safe_20260511_1100.log` | `monitor_multiscale_25876` | Running. Epoch 0 started. GPU about `8255 MiB`. |

Launch checks:

- Local `py_compile` passed for `input_stride2_uniform_50pct_adapter.py`, `input_random_fixed_50pct_adapter_multiscale_safe.py`, `input_random_fixed_50pct_adapter_head_regres_safe.py`, `vit_adapter.py`, and `anchor_free_head.py`.
- Static grep found no `input_pdrop=0.2`, Irregular/HeadV3 route, or pseudo-boundary sampler in the launched configs.
- `stride2_uniform` is a dense contract variant using `sample_stride=2`, train `random_trunc`, and val/test `sliding_window`; it is no-GT at test.
- `multiscale_safe` inherits the known random-fixed Adapter baseline and enables `adapter_multiscale_cfg` with `init_scale=0.0`, so the branch is zero-gated at initialization.
- Both runs again showed the known baseline-family epoch 0 iter 4 non-finite `module.rpn_head.cls_head.weight` gradient skip, then remained alive. Treat this as a watch item, not a crash.

Decision gates:

| Experiment | Continue if first eval | Stop/switch if first eval |
|---|---|---|
| `stride2_uniform_50pct_adapter` | Near or above baseline first gate: `>=62.0 Avg` and `@0.7` not clearly below `39.0`; strong signal if `>=64.5 Avg`. | Below `61.0 Avg` or `@0.7 < 37.0`; if it repeats prior stride-2 negative behavior, demote stride-2 to control only. |
| `multiscale_safe` | At least preserves baseline first gate and trends toward `64+`; promising if it improves `@0.7` without lowering Avg. | More than `0.5 Avg` below baseline first gate; switch server to `input_random_fixed_50pct_adapter_head_regres_safe.py` only if the failure is not a startup/code issue. |

First-eval results and stop actions:

| Server | Config | Eval time | Avg-mAP | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | Delta vs baseline first gate `61.95/39.03` | Action |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `35407` | `input_stride2_uniform_50pct_adapter.py` | 2026-05-11 14:11:02 +08 | 41.09 | 62.91 | 53.99 | 43.33 | 29.87 | 15.36 | -20.86 Avg / -23.67 @0.7 | Stopped immediately; GPU released. |
| `25876` | `input_random_fixed_50pct_adapter_multiscale_safe.py` | 2026-05-11 14:22:57 +08 | 38.19 | 61.20 | 51.53 | 39.14 | 26.46 | 12.62 | -23.76 Avg / -26.41 @0.7 | Stopped immediately; GPU released. |

Interpretation:

- `stride2_uniform_50pct_adapter` again falls in the negative stride-2/geometry-change band, not near the `65+` historical non-adapter behavior. It should remain a control only.
- `multiscale_safe` was already known from an earlier full run to finish at `51.35/26.79`; the new first eval (`38.19/12.62`) matches that negative trajectory. Do not relaunch it or stack it.
- The planned fallback `input_random_fixed_50pct_adapter_head_regres_safe.py` was also already completed earlier at `51.64/26.11`, so it should not be relaunched as a fallback.
- The common pattern is now stronger: small Adapter body branches, head residual branches, simple regression loss reweighting, SimOTA, NMS-only tuning, train-only boundary weighting, stratification, stride-2, and pseudo-boundary input selection all fail to improve the robust random-fixed Adapter baseline.
- Two GPT-5.5 xhigh read-only reviewers were dispatched after these stops to identify directions not yet falsified by the completed matrix.

## Checkpoint Audit and Current Route, 2026-05-11

After the dense-contract relaunch failures, the next no-code audit checked whether the existing strong `input_random_fixed_50pct_adapter_virtual_baseline` checkpoint had a hidden raw/EMA or checkpoint-selection gain before spending another full training cycle.

Remote audit setup:

| Server | Audit | Config | Checkpoint | EMA setting | Status |
|---|---|---|---|---|---|
| `35407` | `audit_ema59` | `input_random_fixed_50pct_adapter_virtual_baseline.py` | `epoch_59.pth` | `solver.ema=True` | Completed; GPU released. |
| `25876` | `audit_raw59` | `input_random_fixed_50pct_adapter_virtual_baseline.py` | `epoch_59.pth` | `solver.ema=False` | Completed; GPU released. |

Audit result:

| Checkpoint view | Avg-mAP | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| EMA epoch 59 | 63.85 | 79.96 | 74.24 | 66.38 | 56.68 | 41.99 | Matches the known virtual baseline and remains the strongest stable non-oracle result. |
| Raw epoch 59 | 62.48 | 78.51 | 72.77 | 64.65 | 55.21 | 41.25 | Worse than EMA; do not switch to raw weights. |

Conclusion:

- The checkpoint audit did not reveal a free `65+` result.
- EMA is materially better than raw for this run and should remain enabled for future Adapter + ActionFormer experiments.
- Schedule-only or raw/EMA checkpoint selection is not a convincing main route unless another epoch already shows `>=64.3 Avg` or `>=42.5@0.7`.
- The current best real route remains the baseline-preserving family around `63.8 Avg`, not any sampler/body/head geometry change tested so far.

Reviewer synthesis after the failures:

| Reviewer | Status | Main conclusion |
|---|---|---|
| GPT-5.5 xhigh `Averroes` | Completed | The common failure is disturbance of the stable Adapter + ActionFormer temporal contract. Retire pdrop, late-linear, multiscale, head-regression, regloss, SimOTA, NMS-only, train-only GT weighting, stratified, pseudo-boundary, and stride-2 as main routes. Prefer dense-contract quality ranking or small training-recipe lifts. |
| GPT-5.5 xhigh `Hilbert` | Completed | Before new training, run the raw/EMA checkpoint audit. Primary next experiment should be a detached class-agnostic localization-quality head that changes proposal ranking only, keeps random-fixed input intact, and requires `alpha=0` to be baseline-equivalent. |
| Claude/Gemini MCP retry | Attempted | Jobs were interrupted/aborted during the session and did not produce a clean usable review artifact. Do not treat them as approval or rejection. |

Recommended next implementation, pending user approval:

| Item | Design |
|---|---|
| Name | Detached Quality Rescore |
| Code surface | `opentad/models/dense_heads/anchor_free_head.py`, one new config, one launcher, focused contract tests. |
| Input contract | Inherit `input_random_fixed_50pct_adapter.py`; keep train/val/test `random_fixed_subsample`; no GT, pseudo-boundary, weighted, or stride sampler in val/test. |
| Quality target | Train-only GT-derived decoded-proposal IoU for positive points, detached from the regression target computation. |
| Gradient isolation | Feed detached regression features to the quality head so the quality loss updates only quality-head parameters. |
| Test-time fusion | `final_score = cls_score * quality_score ** alpha`; `alpha=0` must exactly reproduce baseline scores for the same checkpoint. |
| First config | Conservative `quality_loss_weight` and `alpha=0.25`, with checkpoints every 10 epochs for post-hoc alpha sweeps. |

Minimum verification before deployment:

- Add tests that `alpha=0` is baseline-equivalent on proposals/scores.
- Add tests that quality-loss gradients reach only quality-head parameters.
- Add config/launcher tests proving `ActionFormer`, `VisionTransformerAdapter`, `ActionFormerHead`, train/val/test random-fixed alignment, `input_pdrop=0.0`, no pseudo/oracle/weighted sampler, and no SimOTA assigner.
- Run `python -m pytest tests/test_adapter_safety_contracts.py -q`.
- Run `python -m py_compile` on changed Python/config files and launcher syntax checks.
- Run read-only Claude code review after implementation, apply accepted fixes, rerun verification, then commit before remote sync.

Stop/continue gates:

| Stage | Continue | Stop |
|---|---|---|
| First eval | At least near baseline first gate: `>=61.5 Avg` and `>=39.0@0.7`; strong if `>=63 Avg` and `>=41@0.7`. | `<61 Avg` or `<38@0.7`; hard stop if it enters the `50-52` final-failure trajectory. |
| Final | Continue alpha sweep/rerun if `>=64.5 Avg` or `>=43.5@0.7`. Treat `>=65 Avg` with non-worse `@0.7` as a real go signal. | Stop if `<63 Avg` or `@0.7 <42.0`; do not stack additional branches on a negative run. |

## Claude CLI Discussion and Review-Tool Status, 2026-05-11

The user requested a Claude CLI method discussion for the current direction, route, and solution. The direct CLI call:

```powershell
claude.cmd -p --permission-mode plan --effort xhigh --output-format text "<Adapter + ActionFormer route discussion prompt>"
```

timed out after 360 seconds, but it wrote a plan file under `C:\Users\skywalker\.claude\plans\we-are-optimizing-adapter-nested-valley.md` and returned a usable summary before timeout.

Claude CLI conclusions:

| Topic | Claude CLI conclusion |
|---|---|
| Root cause | Failures are dominated by fragile per-level regression calibration and shifted positive/feature geometry. Geometry-changing routes collapse high-IoU ranking faster than low-IoU recall. |
| Next route | `Detached Quality Rescore` is the best next route because it structurally isolates gradients and preserves an exact `alpha=0` baseline-equivalence contract. |
| Head design | Use a class-agnostic `Conv1d(C -> 1, kernel=3)` quality head fed from `reg_feat.detach()`, not `cls_feat`. |
| Target | Predict decoded proposal IoU to matched GT; Claude recommended including negatives as target 0 for a GFL/QFL-like all-valid-point BCE variant, rather than positive-only supervision. |
| Fusion | `score = cls_score * sigmoid(quality_logit) ** alpha`; start with `alpha=0`, then sweep `{0.25, 0.5, 0.75, 1.0, 1.5, 2.0}` after checkpoints exist. |
| Risks | Uniform quality collapse, target-gradient leakage, `loss_normalizer` contamination, alpha/NMS non-monotonicity, and EMA handling of new-head weights. |
| Gates | Pre-train tests: alpha-0 equivalence and gradient isolation. Epoch-41 continue if `Avg >= 62.0` and `@0.7 >= 39.0`; hard stop if `Avg < 61.0` or `@0.7 < 37.0`. Final go signal: `>=65.0 Avg` with `@0.7 >=42.0`. |
| Backup | If quality rescore is exhausted, consider Distribution Focal Loss on regression or a longer EMA-decay recipe. Do not stack sampler/body/head residual changes. |

Implementation adjustment from Claude:

- The earlier GPT-5.5 route described "positive points only" quality targets. Claude's stronger recommendation is safer for ranking: supervise all valid points, with positives using IoU target and negatives using target 0. This directly trains the quality head to suppress false high-classification background points at test time.
- To avoid disturbing the baseline, keep quality loss on its own small weight and do not use its targets to alter assignment, classification loss, regression loss, sampling, or masks.

`claude-review.review_start` MCP status:

| Check | Evidence | Interpretation |
|---|---|---|
| Tool entry | `review_start` returned `jobId=1c93f44766004000802760a06cb0a2cb`, status `queued`. | MCP entrypoint and tool schema are present. |
| Status polling | `review_status` returned `aborted`; job JSON stayed at `running` with no final result. | The async bridge is unreliable on this Windows setup. |
| Root-cause inspection | `C:\Users\skywalker\.codex\mcp-servers\claude-review\server.py` uses `os.kill(pid, 0)` inside `is_pid_alive()`. | This is a POSIX liveness-check idiom but unsafe on Windows, where signal `0` maps to a control event rather than a pure no-op liveness probe. Polling can interrupt the background worker/Claude process and leave stale `running` jobs. |
| Current protocol | Use direct `claude.cmd -p ...` for method discussion and code review until the MCP bridge is patched. | Do not treat MCP `aborted` as a model-level review result. |

Recommended MCP fix before relying on `claude-review` again:

- Replace `os.kill(pid, 0)` in `is_pid_alive()` with a Windows-safe liveness check, e.g. `psutil.pid_exists(pid)` plus status handling, or a native Windows process-open query.
- Preserve stderr/stdout for background workers in per-job logs so Claude CLI parse failures, timeouts, and auth prompts are inspectable.
- Retest with a minimal prompt before using it for code review.

## Detached Quality Rescore Implementation, 2026-05-11 23:32 +08

Implementation status:

- Code commits: `e4dee3a` in `OpenTAD_Back` (`add detached quality rescore adapter run`), followed by `5c9dcf9` (`fix quality rescore amp target dtype`).
- Added an optional class-agnostic quality head to `AnchorFreeHead`.
- The quality head is enabled only through `quality_head_cfg`; default behavior remains disabled.
- The quality head consumes `reg_feat.detach()`.
- Quality targets are train-GT derived decoded proposal IoU for positive points and `0` for valid negatives.
- Quality loss is added as `quality_loss`, without changing assignment, classification targets, regression targets, or the foreground `loss_normalizer`.
- Test-time fusion is `cls_score * sigmoid(quality_logit) ** alpha`; `alpha <= 0` skips fusion and preserves baseline scores.
- Added `ActionFormer.grad_clip_parameters()` and made `train_engine` use it so global grad clipping excludes `rpn_head.quality_head.*`. This fixes the GPT-reviewed concern that quality-head gradients could otherwise rescale the base Adapter/ActionFormer gradients through global clipping.
- Added `configs/adatad/thumos/input_random_fixed_50pct_adapter_quality_rescore_detached.py`.
- Added `scripts/run_adapter_quality_rescore.sh` with config contract checks for Adapter + ActionFormer, random-fixed train/val/test alignment, `input_pdrop=0.0`, and checkpointing.
- Added `tests/test_adapter_quality_rescore_contracts.py`.

Verification:

| Check | Result |
|---|---|
| `python -m pytest tests/test_adapter_quality_rescore_contracts.py -q` | `4 passed` |
| `python -m pytest tests/test_adapter_safety_contracts.py -q` | `21 passed` |
| Combined targeted pytest | `25 passed` |
| `python -m py_compile` on changed Python/config files | Passed |
| `C:\Program Files\Git\bin\bash.exe -n scripts/run_adapter_quality_rescore.sh` | Passed |
| Local `Config.fromfile` dynamic check | Blocked locally because Windows Python lacks `mmengine`; must be done remotely via `CHECK_ONLY=1 scripts/run_adapter_quality_rescore.sh`. |

Remote smoke fix:

- Initial launch on both servers failed at epoch 0 before any optimizer step with `RuntimeError: expected scalar type Float but found Half`.
- Root cause: AMP produced Half decoded IoU while `quality_target` was Float in `_quality_loss`.
- Fixed in `5c9dcf9` by casting `quality_pred` to float32 before constructing quality logits/targets.

External review:

| Reviewer | Status | Finding / action |
|---|---|---|
| GPT-5.5 xhigh implementation review | Completed | Found high-risk global grad clipping confound; fixed by excluding quality-head parameters from clipping. |
| GPT-5.5 xhigh model-logic review | Completed | No val/test GT leakage found; warned all-valid BCE can over-suppress recall and must be gated. |
| GPT-5.5 xhigh follow-up review | Completed | No blocking findings after grad-clipping fix; proceed to Claude review/deployment. |
| Claude CLI | Attempted twice after implementation | Both attempts returned Claude API `429` high-load rejection. Not counted as a successful Claude review. Retry later when service is available. |

Deployment gates:

- Remote preflight must run `CHECK_ONLY=1 scripts/run_adapter_quality_rescore.sh`.
- First eval continue only if `Avg >= 61.5` and `mAP@0.7 >= 39.0`.
- Stop immediately if `Avg < 61.0` or `mAP@0.7 < 37.0`.
- If final score is below `63.0` or `mAP@0.7 < 42.0`, do not stack additional sampler/body changes on this branch.
- If final score is `>=64.5` or `mAP@0.7 >=43.5`, run alpha sweep / variance check.
- Treat `>=65.0 Avg` with non-worse `mAP@0.7` as a real go signal, then rerun for variance before any SOTA claim.

## Supervision and Claude CLI Protocol Update, 2026-05-12 00:40 +08

Repository protocol update:

- Top-level commit `8cdd043` records the rule that Claude discussions/reviews must use `claude.cmd` CLI, not the `claude-review` MCP channel.
- Rationale: `claude-review.review_start` previously crashed/aborted on this Windows setup and must not be mistaken for a completed Claude review.

Supervision implementation updates:

- `OpenTAD_Back` commit `3fe3921` improves `scripts/supervise_adapter_quality.sh` and `scripts/run_adapter_quality_rescore.sh`.
- `OpenTAD_Back` commit `ad8ac8a` changes the driver supervisor source from the launch tee log to the real work-dir log: `exps/thumos/adatad/input_random_fixed_50pct_adapter_quality_rescore_detached/gpu1_id0/log.json`.
- Supervisor now supports `PROCESS_PID` plus optional `PROCESS_PATTERN`, with pattern-only fallback documented as best effort.
- Fatal stop signatures: traceback, `RuntimeError`, CUDA OOM, graph-change errors, and loss `nan/inf`.
- `non-finite gradients detected ... skip optimizer step` is treated as `CONTINUE` plus anomaly reporting, because the train engine skips the bad step and current runs continued normally afterward.

Review and verification:

| Check | Result |
|---|---|
| Claude CLI monitor review | Completed via `claude.cmd`; found process-pattern false positives, pgrep/pipefail risk, missing caller/docs, stale launch-log monitoring, and hard-error priority. Accepted fixes were applied. |
| Claude CLI final narrow retry | Failed with `API Error: 400 ... organization has been disabled`; not counted as a successful review. |
| GPT-5.5 xhigh monitor review | Completed; flagged broad `PROCESS_PATTERN` false positives. Fixed by adding `PROCESS_PID`-first monitoring. |
| `bash -n` on both scripts | Passed after fixes. |
| Local supervisor smoke tests | Passed for live process `CONTINUE`, eval-ended `REVIEW_RESULT`, crash-after-eval `STOP_REVIEW`, `Loss=nan/inf` `STOP_REVIEW`, and `loss-info` non-match. |

Remote supervision state:

| Server | Train PID | Supervisor PID | Monitor source | Latest observed state |
|---|---:|---:|---|---|
| `35407` | `647893` | `732707` | work-dir `log.json` | epoch 11 started at `2026-05-12 00:33:52`, recent loss `0.7094`, no eval yet. |
| `25876` | `44225` | `128581` | work-dir `log.json` | epoch 11 started at `2026-05-12 00:34:02`, recent loss `0.7089`, no eval yet. |

Important correction:

- The launch logs under `logs/input_random_fixed_50pct_adapter_quality_rescore_detached_20260511_234804.log` stopped after epoch 1, but training was not stuck.
- The active training metrics are in work-dir `log.json`; supervisors were restarted with that source.
- Both runs had one early skipped non-finite gradient around epoch 1 iter 18 on `rpn_head.reg_head.weight`, then continued with normal decreasing losses through epoch 11. Continue monitoring until the first eval gate.
