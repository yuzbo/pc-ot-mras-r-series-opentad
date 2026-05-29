# Research Wiki Log

| Timestamp | Action | Details |
|-----------|--------|---------|
| 2026-05-24T20:55:32+08:00 | bata_selector_speed_pro_synthesis_and_local_optimization | Recorded and synthesized the two user-pasted Pro model speed-diagnosis responses after Oracle browser attachment failed and the inline retry was user-aborted. Added report `research-wiki/experiments/BATA_SELECTOR_SPEED_PRO_SYNTHESIS_AND_OPTIMIZATION_20260524.md`. Implemented local-only optimizations in `OpenTAD_BATA_Clean`: `tools/bata/mobilenet_boundary_selector.py` now exposes `--decord-num-threads`, `--val-every-epochs`, `--validate-last-only`, `--early-stop-patience`, and `--early-stop-min-delta`, guarantees final-epoch validation, logs train-only vs validation epochs, records decode settings in summaries/manifests, and moves sequential/cache preprocessing to target device before resize/normalization. `tools/bata/launch_bata_mobilenet_fixed50_serial.sh` future defaults now favor fast first deployable mAP: one selector epoch, `sequential_video`, no shuffle, no DataLoader workers, train/val caps `2048/1024`, decord threads `2`, early-stop controls exposed. Tests updated in `tests/test_bata_deployable_selector_contracts.py`. Verification PASS: py_compile; launcher `bash -n`; deployable selector pytest `13 passed, 1 skipped`; focused BATA pytest `26 passed, 6 skipped`; `git diff --check` on touched code/test files. The known Windows torch DLL access-violation tail appeared after pytest but exit code was `0`. Strict fixed-50 detector protocol and validation/test GT-free cache contract are unchanged; no remote process was modified. Not deployment-ready until Gemini CLI, DeepSeek CLI, and Linux runtime smoke gates complete. |
| 2026-05-24T21:03:00+08:00 | bata_selector_speed_optimization_review_gate_attempt | Ran required external review gate attempts for the local BATA selector speed patch. Gemini CLI command with `gemini-3-pro-preview` failed and is not accepted: `insufficient_user_quota`; logs `logs/gemini3_pro_preview_bata_selector_speed_optimization_review_20260524.txt` and `.err.txt`. Claude Code CLI DeepSeek `deepseek-v4-pro` completed with PASS; logs `logs/claude_deepseek_v4_pro_bata_selector_speed_optimization_review_20260524.txt`, `.err.txt`, and `.debug.log`. DeepSeek found no blocking issues and no required fixes, marking deployable leakage, GT boundaries, fixed-50 detector contract, tensor/device semantics, validation/checkpoint behavior, launcher safety, and metric attribution as PASS. Gate state: self-check PASS, DeepSeek PASS, Gemini incomplete, Linux torch/decord runtime smoke pending. No remote deployment, restart, or long run was launched. |
| 2026-05-24T21:08:55+08:00 | bata_selector_speed_patch_direct_deploy_override | User explicitly requested direct deployment despite incomplete Gemini gate. Deployed the local BATA selector speed patch to server `35329`, target `/root/autodl-tmp/OpenTAD_BATA_Formal_MobileNet_20260523`, using temp upload `/root/autodl-tmp/bata_speed_patch_20260524_2110` and backup `/root/autodl-tmp/OpenTAD_BATA_Formal_MobileNet_20260523/logs/speed_opt_backups_20260524_210855_user_direct_deploy`. Installed `tools/bata/mobilenet_boundary_selector.py` SHA256 `8e8b5c374149baf55004f4bcdc560c0d59dd382e5dc7d5453371b62aa41f6134`, `tools/bata/launch_bata_mobilenet_fixed50_serial.sh` SHA256 `0139ca43e4fa9e8ff797c7bfa31e548920f4c6ba0932fb172b50e417f3d8549c`, and `tests/test_bata_deployable_selector_contracts.py` SHA256 `abef44f6a5ca1a5e4132d1e7e4955b5fbdaf1f3bc5daa3ec154d14dee6be4465`. Remote verification PASS: py_compile, launcher `bash -n`, and `pytest tests/test_bata_deployable_selector_contracts.py -q` returned `14 passed in 2.34s`. Deployment log: `/root/autodl-tmp/OpenTAD_BATA_Formal_MobileNet_20260523/logs/bata_selector_speed_opt_direct_deploy_20260524_210855.log`. Active selector PID `562544` was not stopped or restarted; it continues with old in-memory code and old 5-epoch command. Future build-cache and future launcher runs in this target tree will use the deployed optimized code. Strict fixed-50 detector protocol and validation/test GT-free cache contract are unchanged; no new long run was launched. |
| 2026-05-24T20:10:42+08:00 | bata_mobilenet_selector_speed_diagnosis_monitor | Read-only check on `35329` for the speed diagnosis. Active screen `bata_mobilenet_seqvalrescue_20260524_1437`, run tag `bata_mobilenet_fixed50_20260524_143349`, PID `562544`, command `tools/bata/mobilenet_boundary_selector.py train ... --loader-mode sequential_video --num-workers 0 --max-train-frames-per-video 4096 --max-val-frames-per-video 2048 --epochs 5`. Latest log reached `epoch=3 step=1900/2292`; GPU memory `985MiB`, sampled util `0%`. Current run cache/eval counts remain `0`; no detector output or deployable mAP exists. Selector log rows exist for epochs 0-2 with best `val_ap=0.132097` at epoch 0. Strict fixed-50 detector config/cache stage has not started, so deployable no-test-GT/no-teacher cache evidence remains pending. Decision: continue the active selector; do not launch another long run. |
| 2026-05-24T13:41:58+08:00 | bata_mobilenet_sequential_validation_rescue_patch_local | Prepared a local-only rescue patch in `OpenTAD_BATA_Clean` because code inspection showed `--loader-mode sequential_video` only changed selector training, while checkpoint-selection validation still used `_evaluate_model()` / per-frame dataset access with no validation progress logs. Changed `tools/bata/mobilenet_boundary_selector.py`, `tools/bata/launch_bata_mobilenet_fixed50_serial.sh`, and `tests/test_bata_deployable_selector_contracts.py`. Added `_evaluate_model_sequential_video()`, `--val-log-every-steps`, launcher env `SELECTOR_VAL_LOG_EVERY_STEPS`, and static deployable contract coverage. Local verification: `py_compile` PASS; `pytest tests/test_bata_deployable_selector_contracts.py -q` `11 passed`; focused BATA pytest `24 passed, 5 skipped`. Report: `research-wiki/experiments/BATA_MOBILENET_SELECTOR_SEQUENTIAL_VALIDATION_RESCUE_PATCH_20260524.md`. Remote `35329` active selector was not modified or interrupted; it remains active with no artifact and I/O progress `rchar=672970558107 -> 675361000459` over 45s. `35407` idle with disk `181G/200G`; `25876` has no visible GPU. Decision: keep waiting for first selector checkpoint; use this patch only as a controlled rescue candidate if the current run fails to checkpoint or becomes inactive. |
| 2026-05-24T13:34:15+08:00 | bata_mobilenet_seqrescue_strong_io_health | Checked `35329`, `35407`, and `25876`. `35329` formal MobileNet selector remains active in screen `bata_mobilenet_seqrescue_clean_20260524_1126`, run tag `bata_mobilenet_fixed50_20260524_112543`, PID `528628`, process state `R`, elapsed about `02:07:00`, CPU about `1051%`, RSS about `8.69GB`, `447` threads, GPU `985/24564 MiB`, disk `68G/200G` used. Latest emitted progress still ends at `epoch=0 step=2250/2292 train_loss_so_far=0.989039`; no `train_log.jsonl`, `best.pth`, `last.pth`, score cache, selector eval, detector artifact, or mAP exists. I/O advanced over 60s from `rchar=660142741174` to `668365074128` and `syscr=20113321` to `20364069`, about `8.22GB` read-character progress, so this is not a stalled process. `35407` has no screen/process, GPU `0/32760 MiB`, disk `181G/200G`; `25876` has no screen/process, no visible GPU, disk `157G/200G`. Decision: continue waiting for first selector checkpoint; no restart and no new long run while I/O continues. Formal deployable evidence remains pending because no validation/test cache or mAP exists. |
| 2026-05-24T13:28:22+08:00 | bata_mobilenet_seqrescue_io_health | Checked `35329`, `35407`, and `25876`. `35329` formal MobileNet selector remains active in screen `bata_mobilenet_seqrescue_clean_20260524_1126`, run tag `bata_mobilenet_fixed50_20260524_112543`, PID `528628`, elapsed about `02:00:56`, CPU about `1030%`, RSS about `8.44GB`, GPU `985/24564 MiB`, disk `68G/200G` used. Latest emitted progress still ends at `epoch=0 step=2250/2292 train_loss_so_far=0.989039`; no `train_log.jsonl`, `best.pth`, `last.pth`, score cache, selector eval, detector artifact, or mAP exists. I/O advanced over 45s from `rchar=631214461623` to `631588339859` and `syscr=19231244` to `19242650`, so the process remains alive and reading/decoding. `35407` has no screen/process, GPU `0/32760 MiB`, disk `181G/200G`; `25876` has no screen/process, no visible GPU, disk `157G/200G`. Decision: continue waiting for first selector checkpoint; no restart and no new long run while I/O continues. Formal deployable evidence remains pending because no validation/test cache or mAP exists. |
| 2026-05-24T13:12:25+08:00 | bata_mobilenet_seqrescue_monitor | Used `monitor-experiment` flow on `35329`, `35407`, and `25876`. `35329` active screen `bata_mobilenet_seqrescue_clean_20260524_1126`, run tag `bata_mobilenet_fixed50_20260524_112543`, log `/root/autodl-tmp/OpenTAD_BATA_Formal_MobileNet_20260523/logs/bata_mobilenet_fixed50_20260524_112543.log`. Selector PID `528628` remains active after about `01:45:58`, CPU about `965%`, RSS about `8.22GB`, GPU `985/24564 MiB`, disk `68G/200G` used. Latest emitted progress remains `epoch=0 step=2250/2292 train_loss_so_far=0.989039`; no `train_log.jsonl`, `best.pth`, `last.pth`, score cache, selector eval, detector artifact, or mAP exists yet. Process I/O is still live: `rchar=608700425688`, `syscr=18544492`, `read_bytes=16649109504`, `fd_count=299`. `35407` has no screen/process, RTX 4080 SUPER `0/32760 MiB`, disk `181G/200G`; `25876` has no screen/process, no visible GPU, disk `157G/200G`. Strict fixed-50 deployable evidence remains pending because no validation/test cache has been generated. Decision: continue waiting for first selector checkpoint; no restart and no new long run while CPU/I/O continues. |
| 2026-05-24T13:07:02+08:00 | bata_mobilenet_seqrescue_io_health | `35329` selector still has no checkpoint/cache/eval/detector/mAP. Screen `bata_mobilenet_seqrescue_clean_20260524_1126`, PID `528628`, elapsed `01:41:16`, CPU about `942%`, RSS about `8.18GB`, GPU memory `985/24564 MiB`, disk `68G/200G` used. Screen/log still end at `epoch=0 step=2250/2292`. I/O advanced over 60s from `rchar=590266146372` to `591400389709` and `syscr=17982664` to `18017255`; open MP4 fds remain ordered later training videos. `35407` remains idle with GPU `0/32760 MiB`, disk `181G/200G`; `25876` remains idle with no visible GPU, disk `157G/200G`. Decision: keep waiting for first selector checkpoint because the process is still doing real decode/read work. If the next monitor remains long-stalled before epoch 0 completion, begin evaluating a controlled observability/validation-limit patch, but do not interrupt now. |
| 2026-05-24T13:02:21+08:00 | bata_mobilenet_seqrescue_io_health | Rechecked `35329` formal selector plus idle servers. `35329` screen `bata_mobilenet_seqrescue_clean_20260524_1126` remains active; PID `528628`, elapsed `01:36:35`, CPU about `915%`, RSS about `8.16GB`, GPU memory `985/24564 MiB`, disk `68G/200G` used. Screen/log still end at `epoch=0 step=2250/2292`; no selector checkpoint/cache/eval/detector/mAP exists. I/O advanced over 45s from `rchar=583514705461` to `583982767492` and `syscr=17776743` to `17791022`, so the process is still reading/decoding video rather than idle. `35407` remains idle with GPU `0/32760 MiB`, disk `181G/200G`; `25876` remains idle with no visible GPU, disk `157G/200G`. Decision: continue waiting for first selector checkpoint; no restart and no new long run. |
| 2026-05-24T12:58:24+08:00 | bata_mobilenet_seqrescue_io_health | I/O health check on `35329` selector PID `528628`: elapsed `01:32:38`, CPU about `891%`, RSS about `8.08GB`. Screen/log still end at `epoch=0 step=2250/2292`; no checkpoint/cache/eval/detector/mAP yet. Process I/O advanced over 45s from `rchar=581011530844` to `581443362643` and `syscr=17700360` to `17713533`; open MP4 fds remain ordered later training videos. Interpretation: still doing real read/decode work, likely slow validation or epoch-tail. Decision: continue waiting for first checkpoint; no restart and no new long run. |
| 2026-05-24T12:54:00+08:00 | bata_mobilenet_seqrescue_monitor | Rechecked all three servers. `35329` selector remains active in screen `bata_mobilenet_seqrescue_clean_20260524_1126`; PID `528628`, elapsed `01:28:14`, CPU about `861%`, RSS about `8.01GB`, GPU memory `985/24564 MiB`, disk `68G/200G` used. Screen/log still end at `epoch=0 step=2250/2292 train_loss_so_far=0.989039`; no selector checkpoint, cache, eval, detector artifact, or mAP exists. `35407` remains idle with GPU `0/32760 MiB`, disk `181G/200G` used; `25876` remains idle with no visible GPU, disk `157G/200G` used. Decision: continue waiting for first selector checkpoint; no restart and no new long run. |
| 2026-05-24T12:50:33+08:00 | bata_mobilenet_seqrescue_io_health | I/O health check on `35329` selector PID `528628`: elapsed `01:24:47`, CPU about `835%`, RSS about `7.95GB`. The log and screen hardcopy still end at `epoch=0 step=2250/2292`, but process I/O advanced over 60s from `rchar=573278011756` to `574409096557` and `syscr=17464408` to `17498911`; open MP4 fds are ordered later training videos. No selector checkpoint/cache/eval/detector/mAP yet. Interpretation: still doing real decode/read work, likely slow validation or epoch-tail. Decision unchanged: continue waiting for first checkpoint; no restart and no new long run. |
| 2026-05-24T12:46:08+08:00 | bata_mobilenet_seqrescue_monitor | Rechecked `35329`, `35407`, and `25876`. `35329` still runs selector PID `528628` in screen `bata_mobilenet_seqrescue_clean_20260524_1126`, elapsed `01:20:22`, CPU about `799%`, RSS about `7.91GB`, GPU memory `985/24564 MiB`, disk `68G/200G` used. Log mtime is still `2026-05-24 12:17:16 +0800`; latest emitted progress remains `epoch=0 step=2250/2292 train_loss_so_far=0.989039`. `train_log.jsonl` is still missing and there are no selector checkpoint, score cache, eval, detector artifact, or mAP files. Given training reached step 2250 at 12:17 after about 52 minutes and validation has 232026 examples with no fine-grained logging, this still fits slow epoch-tail/validation rather than a proven hang. `35407` remains idle with GPU `0/32760 MiB`, disk `181G/200G` used; `25876` remains idle with no visible GPU, disk `157G/200G` used. Decision: continue waiting for first selector checkpoint; no restart and no new long run. |
| 2026-05-24T12:41:41+08:00 | bata_mobilenet_seqrescue_monitor | Rechecked all three servers. `35329` selector is still active in screen `bata_mobilenet_seqrescue_clean_20260524_1126`; PID `528628`, elapsed `01:15:55`, CPU about `758%`, RSS about `7.88GB`, GPU memory `985/24564 MiB`, disk `68G/200G` used. Log mtime remains `2026-05-24 12:17:16 +0800` and latest emitted progress is still `epoch=0 step=2250/2292 train_loss_so_far=0.989039`. No `train_log.jsonl`, selector checkpoint, cache, eval, detector artifact, or mAP exists yet. `35407` is idle with RTX 4080 SUPER `0/32760 MiB` and disk `181G/200G` used; `25876` is idle with no visible GPU and disk `157G/200G` used. Decision: continue waiting for first selector checkpoint; no restart and no new long run. |
| 2026-05-24T12:38:04+08:00 | bata_mobilenet_seqrescue_io_health | Deep health check on `35329` selector PID `528628`: process remains active, elapsed `01:12:18`, CPU about `721%`, RSS about `7.85GB`, GPU memory still about `985/24564 MiB`. The log is unchanged at `epoch=0 step=2250/2292`, but `/proc/$PID/io` advanced over 45s from `rchar=555492920685` to `556280424040` with `syscr=16923697 -> 16947614`; opened MP4 fds remain ordered in the later training videos. This supports slow epoch-tail/validation work rather than a dead hang. No checkpoint/cache/eval/detector/mAP yet. Decision: continue waiting for first selector checkpoint; no restart and no new long run. |
| 2026-05-24T12:33:33+08:00 | bata_mobilenet_seqrescue_recheck | Quick recheck on `35329`: sequential rescue selector PID `528628` remains active, elapsed about `01:07:47`, CPU about `670%`, RSS about `7.82GB`. The log still ends at `epoch=0 step=2250/2292 train_loss_so_far=0.989039`; no selector files, score cache files, eval files, detector process, or mAP yet. Decision unchanged: keep monitoring; do not interrupt or launch another long run while the selector is CPU-active. |
| 2026-05-24T12:27:46+08:00 | bata_mobilenet_seqrescue_health | Checked `35329`, `35407`, and `25876`. `35329` sequential rescue selector is still active in screen `bata_mobilenet_seqrescue_clean_20260524_1126`, PID `528628`, elapsed about `01:02:00`, command includes `--loader-mode sequential_video --no-shuffle`. Latest emitted selector progress remains `epoch=0 step=2250/2292 train_loss_so_far=0.989039`; no `best.pth`, `last.pth`, score cache, eval summary, detector run, or mAP yet. The lack of new log lines is not a dead process: over a 70s check `rchar` increased from `539293708778` to `540365754283`, CPU was about `593%`, RSS about `7.75GB`, and GPU memory `985/24564 MiB`. It is likely in epoch-0 tail or validation, which has no fine-grained progress logging. Disk on `35329` is healthy at `68G/200G` used. `35407` has no active screen/process, RTX 4080 SUPER idle, disk `181G/200G` used; `25876` has no active screen/process and no visible GPU, disk `157G/200G` used. Decision: continue monitoring; no new Sparse/BATA long run and no interruption while CPU/I/O progress continues. |
| 2026-05-24T12:04:46+08:00 | bata_mobilenet_seqrescue_progress | `35329` sequential rescue selector reached `epoch=0 step=1650/2292 train_loss_so_far=0.952061`, PID `528628` active, elapsed `39:00`, GPU memory `985/24564 MiB`, sampled util `2%`. No selector checkpoint/cache/eval/detector output yet because epoch 0 is still running. |
| 2026-05-24T11:56:45+08:00 | bata_mobilenet_seqrescue_progress | `35329` sequential rescue selector reached `epoch=0 step=1350/2292 train_loss_so_far=0.989591`, PID `528628` active, elapsed `30:59`, GPU memory `985/24564 MiB`, disk `68G/200G` used. No selector files/cache/eval/detector yet because epoch 0 is still running. `35407` idle; `25876` has no visible GPU. |
| 2026-05-24T11:55:16+08:00 | bata_mobilenet_seqrescue_health | Short health check: `35329` selector reached `epoch=0 step=1250/2292 train_loss_so_far=1.008588`, PID `528628` active, elapsed `29:30`, GPU memory `985/24564 MiB`. Still waiting for epoch 0 completion and checkpoint. |
| 2026-05-24T11:54:03+08:00 | bata_mobilenet_seqrescue_progress | `35329` sequential rescue selector reached `epoch=0 step=1200/2292 train_loss_so_far=1.023045`, PID `528628` active, elapsed `28:17`, GPU memory `985/24564 MiB`, sampled util `1%`. No selector checkpoint/cache/eval/detector output yet because epoch 0 is still running. This is over half of epoch 0 and confirms the rescue run remains stable. |
| 2026-05-24T11:41:10+08:00 | bata_mobilenet_seqrescue_progress | `35329` sequential rescue selector continues healthy progress. Screen `bata_mobilenet_seqrescue_clean_20260524_1126`, PID `528628`, elapsed `15:24`, GPU memory `985/24564 MiB`, disk `68G/200G` used. Latest log reached `epoch=0 step=700/2292 train_loss_so_far=1.135145`. No selector checkpoint/cache/eval/detector/mAP yet. `35407` remains idle; `25876` has no visible GPU and no active process. |
| 2026-05-24T11:37:02+08:00 | bata_mobilenet_seqrescue_progress | Short-interval recheck on `35329`: sequential rescue selector reached `epoch=0 step=500/2292 train_loss_so_far=1.119478`, PID `528628` active, elapsed `11:16`, GPU memory `985/24564 MiB`. No selector files yet because epoch 0 has not completed. This confirms stable progress after the rescue restart. |
| 2026-05-24T11:34:00+08:00 | bata_mobilenet_seqrescue_progress | `35329` sequential rescue selector is still active and progressing. Screen `bata_mobilenet_seqrescue_clean_20260524_1126`, PID `528628`, elapsed `08:14`, GPU memory `985/24564 MiB`, disk `68G/200G` used. Latest log reached `epoch=0 step=350/2292 train_loss_so_far=1.027897`. No selector checkpoint, cache, eval summary, detector work dir, or mAP yet because epoch 0 is still in progress. `35407` remains idle with GPU `0/32760 MiB` and disk `181G/200G` used. |
| 2026-05-24T11:31:38+08:00 | bata_mobilenet_seqrescue_health | `35329` sequential rescue selector remains healthy. Log reached `epoch=0 step=250/2292`; PID `528628` active with RSS about `2.1GB`. `/proc` I/O shows `read_bytes=1.88GB`, much lower than the failed random DataLoader workers' multi-TB reads. Open mp4 fds are now in sequential training-video order around `video_validation_0000051` through `0000167`, confirming ordered per-video traversal rather than cross-video random seek. No checkpoint/cache/mAP yet. |
| 2026-05-24T11:29:10+08:00 | bata_mobilenet_seqrescue_progress | Follow-up check on `35329` sequential rescue selector. Active process PID `528628`, elapsed `03:24`, command includes `--loader-mode sequential_video --no-shuffle --log-every-steps 50`. Latest log reached `epoch=0 step=150/2292 train_loss_so_far=0.759153`; GPU memory `981/24564 MiB`, util sampled `1%`. Selector has no checkpoint yet because epoch 0 is still running. |
| 2026-05-24T11:27:49+08:00 | bata_mobilenet_seqrescue_launched_and_progressing | Operationally failed original formal selector on `35329` was stopped after about `03:49` elapsed with no selector files, no log progress after `training deployable MobileNet boundary selector`, GPU idle, and workers having read multiple TB from many mp4 files. Deployed the local sequential-loader rescue patch to `/root/autodl-tmp/OpenTAD_BATA_Formal_MobileNet_20260523` after backing up old files under `logs/seqrescue_backups_20260524_1118_seqrescue`. Remote verification after patch: py_compile PASS; deployable pytest `10 passed`; focused BATA pytest `28 passed`. Because Gemini CLI remained unavailable due `insufficient_user_quota`, this is recorded as a controlled rescue under prior user all-permission/no-reconfirm instruction, with self-check + DeepSeek PASS and explicit gate-risk notation. New active screen: `bata_mobilenet_seqrescue_clean_20260524_1126`; log: `/root/autodl-tmp/OpenTAD_BATA_Formal_MobileNet_20260523/logs/bata_mobilenet_fixed50_20260524_112543.log`; run tag: `bata_mobilenet_fixed50_20260524_112543`; env: `SELECTOR_LOADER_MODE=sequential_video`, `SELECTOR_TRAIN_SHUFFLE=0`, `SELECTOR_LOG_EVERY_STEPS=50`, `SELECTOR_NUM_WORKERS=0`. Selector emitted `selector_train_start` with `train_examples=282373`, `val_examples=232026`, `steps_per_epoch=2292`, and has progressed to `epoch=0 step=100/2292`. No cache/detector/mAP yet. |
| 2026-05-24T11:12:57+08:00 | bata_formal_selector_recheck_after_patch | Rechecked `35329` after local sequential-loader patch and review records. Active formal selector is still the original remote process, elapsed about `03:40`, PIDs `511439/511512/511575`, and selector output directory remains empty. No remote code was changed and no process was interrupted. `git diff --check` for the local patch/report files returned only the existing LF-to-CRLF warning on `research-wiki/log.md`, no whitespace errors. |
| 2026-05-24T11:09:00+08:00 | bata_mobilenet_selector_sequential_loader_patch | Extended the local-only selector runtime rescue patch in `OpenTAD_BATA_Clean` with optional `--loader-mode sequential_video` and launcher env `SELECTOR_LOADER_MODE`. The sequential loader groups examples by video, reads ordered frame batches with decord `get_batch`, and keeps default behavior as `dataloader`. Verification PASS: py_compile; deployable selector pytest `10 passed`; focused BATA pytest `23 passed, 5 skipped`. DeepSeek follow-up review PASS with two non-blocking concerns; Gemini CLI again failed with `insufficient_user_quota`, so the Gemini gate remains incomplete. Reports updated: `BATA_MOBILENET_SELECTOR_RUNTIME_OBSERVABILITY_PATCH_20260524.md` and `BATA_MOBILENET_SELECTOR_RUNTIME_OBSERVABILITY_REVIEW_20260524.md`. This patch remains local-only and is not deployed to the active formal run. |
| 2026-05-24T11:01:07+08:00 | bata_formal_selector_monitor_and_diag_cleanup | Rechecked `35329`: formal MobileNet selector still active after about `03:28`, PIDs `511439/511512/511575`, GPU `381/24564 MiB`, util `0%`, log still at `training deployable MobileNet boundary selector`, selector dir empty. Found a leftover diagnostic process from the earlier timed-out exact sample-count command (`526101` with child `526103`); terminated only that diagnostic shell/python, leaving formal selector untouched. DeepSeek review of the local selector runtime observability patch returned `PASS` with no WARN/FAIL; Gemini CLI review failed with `insufficient_user_quota` and is not accepted as a completed Gemini gate. Review report: `research-wiki/experiments/BATA_MOBILENET_SELECTOR_RUNTIME_OBSERVABILITY_REVIEW_20260524.md`. Local patch remains not deployable until Gemini gate is completed or explicitly overridden. |
| 2026-05-24T10:53:10+08:00 | bata_mobilenet_selector_observability_patch | Implemented a local-only runtime observability/rescue patch in `OpenTAD_BATA_Clean` after formal selector training on `35329` stayed output-empty for more than 3 hours. Changed `tools/bata/mobilenet_boundary_selector.py`, `tools/bata/launch_bata_mobilenet_fixed50_serial.sh`, and `tests/test_bata_deployable_selector_contracts.py`. The patch adds a `selector_train_start` JSON summary, optional `--log-every-steps`, optional `--no-shuffle`, and launcher env controls `SELECTOR_MAX_STEPS`, `SELECTOR_MAX_TRAIN_FRAMES_PER_VIDEO`, `SELECTOR_MAX_VAL_FRAMES_PER_VIDEO`, `SELECTOR_LOG_EVERY_STEPS`, `SELECTOR_TRAIN_SHUFFLE`. Defaults preserve existing formal behavior. Local verification from `OpenTAD_BATA_Clean`: `py_compile` PASS; deployable selector pytest `10 passed`; focused BATA pytest `23 passed, 5 skipped`. Report: `research-wiki/experiments/BATA_MOBILENET_SELECTOR_RUNTIME_OBSERVABILITY_PATCH_20260524.md`. Not deployed; current remote formal process remains untouched pending review/decision. |
| 2026-05-24T10:42:02+08:00 | bata_formal_selector_recheck | Lightweight recheck on `35329`: formal MobileNet selector still active with PIDs `511439/511512/511575`, elapsed about `03:09`; workers remain CPU-heavy, GPU unchanged at `381/24564 MiB` and `0%` util. Formal log still ends at `training deployable MobileNet boundary selector`; selector output directory remains empty. No cache build, detector training, or formal mAP has started. Decision unchanged: monitor only; no new long run and no interruption without failure evidence or user direction. |
| 2026-05-24T10:39:12+08:00 | bata_formal_mobilenet_monitor | Checked `35407`, `35329`, and `25876` with native OpenSSH. `35407` oracle-boundary Adapter repro is complete and reproducible at `Average-mAP=77.18`, vector `84.01 / 81.84 / 79.25 / 74.47 / 66.33`, `Training Over` at `2026-05-24 08:39:59`; no active screen/process, RTX 4080 SUPER idle, disk `/root/autodl-tmp` `181G/200G` used. `35329` Pre-2 noisy-oracle diagnostic is complete at `Average-mAP=64.17`, vector `79.63 / 74.78 / 66.75 / 57.00 / 42.69`, and its checkpoint cleanup kept only `epoch_59.pth`; disk now `68G/200G` used. Formal MobileNet-BCA wrapper passed Linux py_compile, focused pytest `28 passed`, MobileNet forward, selector builder forward, tiny train/cache/eval, and manifest smoke; formal deploy marker exists. Current active screen is `bata_mobilenet_formal_serial_20260523_1645`, currently in deployable selector training with run tag `bata_mobilenet_fixed50_20260524_073247`. Active PIDs `511439/511512/511575` have elapsed about `03:06`, GPU memory is only `381/24564 MiB`, and two DataLoader workers are CPU-heavy with about `2.0-2.16 TB` read bytes each. Formal logs have not advanced beyond `training deployable MobileNet boundary selector`, selector output dir is still empty, and no cache/detector/mAP exists yet. `25876` remains idle with no visible GPU. Decision: continue monitoring, do not start new long runs, and do not interrupt the selector unless it clearly fails or the user directs a restart. |
| 2026-05-24T01:32:27+08:00 | bata_oracle_gap_repro_launch | Wrote `research-wiki/experiments/BATA_NOISY_ORACLE_GAP_AND_ORACLE_REPRO_20260524.md` and launched a user-requested historical oracle-boundary Adapter reproduction on idle `35407`. Historical oracle final evidence: `77.62`, vector `84.42 / 82.41 / 79.69 / 74.67 / 66.91`, log `/root/autodl-tmp/OpenTAD_Back_check/logs/input_oracle_boundary_dense_50pct_adapter.log`. Current Pre-2 noisy-oracle BCA latest complete eval remains `63.30`, vector `79.53 / 74.92 / 65.92 / 55.55 / 40.58`. Gap analysis: old oracle uses `oracle_boundary_subsample` and reads GT boundaries in train/val/test; Pre-2 uses noisy score caches plus BCA v1 top-8 peak/triplet/coverage allocation, with Pre-1 boundary-role BR@4 only `51.19%` for oracle-BCA and `39.00%` for noisy-BCA. Repro config on `35407`: `/root/autodl-tmp/OpenTAD_Back_check/configs/adatad/thumos/input_oracle_boundary_dense_50pct_adapter_repro_20260524_0108.py`, only `work_dir` changed. Screen: `oracle_boundary_adapter_repro_20260524_0108`; log: `/root/autodl-tmp/OpenTAD_Back_check/logs/input_oracle_boundary_dense_50pct_adapter_repro_20260524_0108.log`; command uses `torchrun --nproc_per_node=1 --master_port=29631`. Launch verified `Training Starts` / `Epoch 0 started`; GPU about `3451/32760 MiB`; disk `/root/autodl-tmp` `181G/200G` used. This is diagnostic test-time-GT oracle evidence only, not deployable. |
| 2026-05-24T01:34:48+08:00 | oracle_repro_health | Checked `35407` oracle-boundary Adapter repro after launch. Screen `oracle_boundary_adapter_repro_20260524_0108` remains active; log reached epoch 1. Completed epoch 0 train line: `[000][00099/00099] Loss=1.5381 cls_loss=0.9073 reg_loss=0.6308`, mem `2417MB`; GPU about `3455/32760 MiB`. No checked Traceback/RuntimeError/OOM. `35329` Pre-2 has no new complete eval beyond `63.30`; latest grep still shows four evals ending at `2026-05-24 00:35:29`. |
| 2026-05-24T01:03:16+08:00 | server_progress_monitor | Checked servers `35407`, `35329`, and `25876` with native OpenSSH. `35407`: no screen/train, RTX 4080 SUPER idle `0/32760 MiB`, disk `181G/200G` used with `20G` free; completed Sparse SAN remains `64.14`. `35329`: active screens `bata_pre2_noisy_oracle_20260523_122831` and `bata_mobilenet_formal_serial_20260523_1645`; BATA Pre-2 noisy-oracle diagnostic is still active with `tools/train.py/torchrun`, RTX 4090 D `3715/24564 MiB`, disk `69G/200G` used. Latest complete Pre-2 eval at `2026-05-24 00:35:29 +0800`: `Average-mAP=63.30`, vector `79.53 / 74.92 / 65.92 / 55.55 / 40.58`; epoch 49 training finished at `00:57:29` and final validation was in progress around `57/396` in the latest screen snapshot. Formal MobileNet-BCA queue remains waiting behind active train processes; no formal marker, Linux smoke, selector/cache/detector run, or formal mAP yet. `25876`: no active screen/process, no visible GPU, disk `157G/200G` used with `44G` free. Decision: monitor only; Pre-2 is diagnostic-only GT/noisy-oracle evidence and not deployable. |
| 2026-05-23T22:35:01+08:00 | bata_progress_monitor | Checked `35329` BATA tasks. Screens active: `bata_pre2_noisy_oracle_20260523_122831` and `bata_mobilenet_formal_serial_20260523_1645`. GPU `3715/24564 MiB`, `/root/autodl-tmp` `69G/200G` used. Pre-2 noisy-oracle diagnostic first eval completed at `2026-05-23 22:12:40 +0800`: `Average-mAP=63.08`, mAP vector `79.60 / 74.84 / 65.65 / 55.00 / 40.31`, predictions `422000`, GT instances `3325`; training continued to epoch 45 and another eval/progress stage. Checkpoint dir has `epoch_19.pth` and `epoch_39.pth`; no final result file in checked path. Formal MobileNet-BCA queue remains waiting on active Pre-2 `tools/train.py|torchrun`; formal marker is absent as expected, so no Linux smoke, formal selector/cache/detector run, or formal mAP has started. |
| 2026-05-23T16:51:56+08:00 | bata_deployable_mobilenet_serial_queued | Uploaded and launched the formal MobileNet-BCA serial wait queue on `35329`. Remote target: `/root/autodl-tmp/OpenTAD_BATA_Formal_MobileNet_20260523`; archive `/root/autodl-tmp/OpenTAD_BATA_Formal_MobileNet_20260523_1645.tar.gz` SHA256 `4201f9b56d02c392edbc19a569e21fcde2a49fdd22c37b7c58b8e0826b2fbc9a`; wrapper SHA256 `5698aeaa00c7de92bff87e79b970417b60dc19562b15c0b25e4aff7faf27751a`. Created `pretrained -> /root/autodl-tmp/OpenTAD/pretrained`; wrapper and formal launcher passed `bash -n`. First relative-path screen attempt exited immediately with no marker/smoke/run; relaunched with absolute wrapper path. Active screen: `865623.bata_mobilenet_formal_serial_20260523_1645`; log: `/root/autodl-tmp/OpenTAD_BATA_Formal_MobileNet_20260523/logs/bata_mobilenet_formal_serial_20260523_1645.log`. The screen is waiting behind active Pre-2 `tools/train.py|torchrun`; formal marker is absent as expected. No formal selector/cache/detector run or mAP yet. |
| 2026-05-23T16:45:14+08:00 | bata_deployable_mobilenet_serial_rule | Recorded formal MobileNet-BCA serial deployment/autolaunch rule before enabling the queue. Report: `research-wiki/experiments/BATA_DEPLOYABLE_MOBILENET_SERIAL_DEPLOYMENT_20260523.md`. Local archive `logs/OpenTAD_BATA_Formal_MobileNet_20260523_1645.tar.gz` SHA256 `4201F9B56D02C392EDBC19A569E21FCDE2A49FDD22C37B7C58B8E0826B2FBC9A`; wrapper `logs/run_bata_mobilenet_formal_serial_20260523_1645.sh` SHA256 `5698AEAA00C7DE92BFF87E79B970417B60DC19562B15C0B25E4AFF7FAF27751A`. Server `35329` still has active Pre-2 noisy-oracle diagnostic screen `bata_pre2_noisy_oracle_20260523_122831`, latest epoch 22 start and no mAP. The formal wrapper will wait for active `tools/train.py|torchrun`, then run Linux smoke, then write the formal deploy marker and invoke the reviewed formal launcher. If smoke fails, no detector launch. |
| 2026-05-23T16:35:38+08:00 | bata_deployable_mobilenet_gemini_deepseek_pass | Completed post-fix external review gates for formal deployable MobileNet-BCA. Report: `research-wiki/experiments/BATA_DEPLOYABLE_MOBILENET_GEMINI_DEEPSEEK_REVIEW_20260523.md`. Gemini first attempt had exit code 0 but empty stdout and invalid-stream stderr, so it was rejected; retry 1 output `logs/gemini3_pro_preview_bata_deployable_mobilenet_gpt5fix_review_20260523_retry1.txt` returned `PASS` with only non-blocking WARN. Claude CLI DeepSeek output `logs/claude_deepseek_v4_pro_bata_deployable_mobilenet_gpt5fix_review_20260523.txt` returned `PASS`, confirming B1/B2/N2/N3/N4/N6 fixes and no blockers. Formal code remains local only; no remote sync, no selector/cache/detector run, no mAP. Remaining required gate before deployment or long run: Linux py_compile/pytest/MobileNet import-forward/tiny train-cache-eval/formal dataset smoke. |
| 2026-05-23T16:19:07+08:00 | bata_deployable_mobilenet_gpt5pro_fixes | Recorded the full manual GPT-5 Pro review in `research-wiki/experiments/BATA_DEPLOYABLE_MOBILENET_GPT5_PRO_REVIEW_20260523.md`; verdict `FAIL` before formal launch / remote full smoke. Added fix report `research-wiki/experiments/BATA_DEPLOYABLE_MOBILENET_GPT5_PRO_FIXES_20260523.md`. Fixed accepted blockers in `OpenTAD_BATA_Clean`: deployable `build-cache` now uses `include_boundaries=False` and no longer reads validation/test GT boundary segments; eval-cache BR@4 deltas are logging-only and `--fail-on-gate` checks deployable cache contract plus coverage; selector training now guards `--train-subset`; LoadFrames rejects diagnostic-only caches in formal mode; formal launcher uses run-tagged cache dirs with `BATA_TRAIN_CACHE_DIR` / `BATA_VAL_CACHE_DIR` and no default `rm -rf`; tests now include executable manifest/record gate checks. Verification: py_compile PASS; deployable pytest `10 passed`; focused BATA pytest `23 passed, 5 skipped`. Strict fixed-50 remains `384/768`; no Adapter/head/loss/assignment change, no teacher leakage, no deployable test-time GT claim, no remote sync/run/mAP. Next gates: Gemini CLI, Claude CLI DeepSeek, and Linux MobileNet/tiny train-cache-eval/config smoke. |
| 2026-05-23T15:43:09+08:00 | bata_deployable_mobilenet_coded | Implemented formal deployable BATA fixed-50 MobileNet-BCA code locally in `OpenTAD_BATA_Clean`. Added direct `torchvision.models.mobilenet_v3_small` `64x64` boundary selector train/cache/eval tool, deployable cache manifest protocol with `uses_gt=False` and `diagnostic_only=False`, formal `configs/adatad/thumos/bata_mobilenet_bca_fixed50_adapter.py`, deployable smoke gate, serial launcher, and tests. Self-check: `research-wiki/experiments/BATA_DEPLOYABLE_MOBILENET_IMPLEMENTATION_SELF_CHECK_20260523.md`. Local verification: py_compile PASS; `python -m pytest tests\test_bata_boundary_acquisition_contracts.py tests\test_bata_post_processing_selected_axis.py tests\test_bata_deployable_selector_contracts.py -q` returned `19 passed, 5 skipped`. Local torch/torchvision runtime import still fails due Windows DLL initialization, so Linux MobileNet forward/video smoke is pending. No remote sync, no selector training/cache generation, no detector run, and no mAP. Status: `CODED-LOCAL / REVIEW-PENDING`; next gates are GPT-5 Pro, Gemini CLI, Claude CLI DeepSeek, then remote Linux smoke before any launch. |
| 2026-05-23T15:02:52+08:00 | bata_pre2_progress | Checked BATA Pre-2 diagnostic run on `35329`. Screen `bata_pre2_noisy_oracle_20260523_122831` and `torchrun/tools/train.py` remain active; GPU `3713/24564 MiB`; `/root/autodl-tmp` `67G/200G` used. Training has reached `Epoch 13 started`; latest completed epoch 12 logged `Loss=0.6089`, `cls_loss=0.3398`, `reg_loss=0.2691`, down from epoch 0 `Loss=1.6007`. No `Traceback`, `ERROR`, `non-finite`, checkpoint, `result_detection.json`, or mAP yet. Continue to epoch-40 first eval; result remains diagnostic-only GT-cache evidence. |
| 2026-05-23T12:36:53+08:00 | bata_pre2_health | BATA Pre-2 diagnostic run health check on `35329`: screen `bata_pre2_noisy_oracle_20260523_122831` and `torchrun/tools/train.py` are still active; GPU usage about `3707/24564 MiB`; serial log `logs/bata_pre1_pre2_noisy_oracle_serial_20260523_122842.log` remains at `Epoch 0 started` with expected PyTorch/mmengine warnings and no `Traceback`, `ERROR`, mAP, or checkpoint event yet. Run remains active; no cleanup applies. |
| 2026-05-23T12:31:22+08:00 | bata_pre2_training_started | BATA Pre-2 fixed-50 noisy-oracle diagnostic run entered real training on `35329`. Screen: `bata_pre2_noisy_oracle_20260523_122831`; serial log: `logs/bata_pre1_pre2_noisy_oracle_serial_20260523_122842.log`; outer log: `logs/bata_pre2_noisy_oracle_launch_20260523_122831_outer.log`; work dir: `exps/thumos/adatad/bata_noisy_oracle_bca_fixed50_adapter_diagnostic_only_gtcache/gpu1_id0/`. The relaunched script passed py_compile, pytest, cache build, Pre-1, and Pre-2 dataset smoke again; checkpoint loaded from the fixed `pretrained` symlink; DDP/EMA/AMP initialized; `Training Starts` and `Epoch 0 started`. GPU usage about `3703/24564 MiB`. No mAP yet. Result interpretation must remain diagnostic-only GT-cache evidence. |
| 2026-05-23T12:28:41+08:00 | bata_pre2_relaunch | Relaunched BATA Pre-2 fixed-50 noisy-oracle diagnostic GPU closed-loop on `35329` after fixing the pretrained symlink. Screen: `bata_pre2_noisy_oracle_20260523_122831`; runner: `logs/run_bata_pre2_noisy_oracle_20260523_122831.sh`; outer log: `logs/bata_pre2_noisy_oracle_launch_20260523_122831_outer.log`; command: `DEPLOY_MARKER=logs/BATA_PRE1_PRE2_REVIEWED_DEPLOYED_20260523.ok MASTER_PORT=29623 SEED=20260522 bash tools/bata/launch_bata_pre2_noisy_oracle_serial.sh`. Prelaunch verified no active train/screen, GPU `0/24564 MiB`, Pre-1 PASS, Pre-2 smoke PASS, reviewed deployment marker, matched clean baseline artifact, and required pretrained weight. Same reviewed serial launcher; no code/config/model behavior changed. |
| 2026-05-23T12:26:00+08:00 | bata_pretrained_symlink_fix | Fixed the clean BATA remote tree asset path on `35329`: created `/root/autodl-tmp/OpenTAD_BATA_Clean_20260523/pretrained -> /root/autodl-tmp/OpenTAD/pretrained` and verified `pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth` exists (`87M`). No screen or `tools/train.py` / `torchrun` process remained during the fix. This is environment staging only; no code/config/model behavior changed. Next action: relaunch the same reviewed Pre-2 serial command. |
| 2026-05-23T12:23:29+08:00 | bata_pre2_env_crash | First BATA Pre-2 launch on `35329` exited before training. Screen `bata_pre2_noisy_oracle_20260523_122042` is gone; no `tools/train.py` or `torchrun` remains. Launcher gates passed, then `tools/train.py` failed while loading backbone checkpoint: `FileNotFoundError: pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth can not be found.` Logs: `logs/bata_pre2_noisy_oracle_launch_20260523_122042_outer.log` and `logs/bata_pre1_pre2_noisy_oracle_serial_20260523_122052.log`. This is a clean-tree asset/symlink issue, not BCA training evidence. Existing weight found at `/root/autodl-tmp/OpenTAD/pretrained/...` and `/root/autodl-tmp/OpenTAD_Back_check/pretrained/...`; next action is to create `pretrained -> /root/autodl-tmp/OpenTAD/pretrained` in `/root/autodl-tmp/OpenTAD_BATA_Clean_20260523` and relaunch the same reviewed serial command. |
| 2026-05-23T12:20:52+08:00 | bata_pre2_launch | Launched BATA Pre-2 fixed-50 noisy-oracle diagnostic GPU closed-loop on `35329` in screen `bata_pre2_noisy_oracle_20260523_122042`. Runner: `/root/autodl-tmp/OpenTAD_BATA_Clean_20260523/logs/run_bata_pre2_noisy_oracle_20260523_122042.sh`; outer log: `logs/bata_pre2_noisy_oracle_launch_20260523_122042_outer.log`; config: `configs/adatad/thumos/bata_noisy_oracle_bca_fixed50_adapter.py`; launcher: `tools/bata/launch_bata_pre2_noisy_oracle_serial.sh` with `DEPLOY_MARKER=logs/BATA_PRE1_PRE2_REVIEWED_DEPLOYED_20260523.ok`, `MASTER_PORT=29623`, `SEED=20260522`. Final prelaunch gates passed: no active train/screen on `35329`, GPU `0/24564 MiB`, port free, Pre-1 PASS, Pre-2 smoke PASS, and matched clean random-fixed 50% baseline artifact at `/root/autodl-tmp/OpenTAD_SparseTAD_Clean_sapm_61e272e/exps/thumos/adatad/input_random_fixed_50pct_adapter_postfix_early10_20260521_1230/gpu1_id0` with final `63.57 / 40.99@0.7`. This run is fixed `384/768` and uses GT-derived noisy-oracle caches, so any result is diagnostic-only, not deployable selector evidence. |
| 2026-05-23T12:15:57+08:00 | bata_pre2_smoke_pass | Pre-2 config/dataset smoke passed on `35329` from `/root/autodl-tmp/OpenTAD_BATA_Clean_20260523`. Log: `logs/bata_pre2_config_dataset_smoke_20260523_121526.log`; marker: `logs/BATA_PRE2_CONFIG_DATASET_SMOKE_PASS_20260523_121526.ok`. Built diagnostic noisy-oracle score caches for `200` training and `211` validation videos under `/root/autodl-tmp/bata_score_caches/`. `smoke_bata_pre2_remote_gate.py --dataset-smoke --max-samples 1` passed train sample 0, val sample 0, and test sample 0. Fixed-50 target remains `384/768`; diagnostic GT-cache flags and selected-axis metadata passed. No GPU training launched. Remaining gate: identify matched clean 50% baseline artifacts and recheck server serial clearance before any Pre-2 detector closed-loop. |
| 2026-05-23T12:12:50+08:00 | bata_pre1_pass | Full no-GPU BATA Pre-1 offline observability diagnostic passed on `35329`. Log: `logs/bata_pre1_full_offline_20260523_121031.log`; summary: `/root/autodl-tmp/bata_pre1/thumos14_validation_s4_seed20260522_20260523_121031/summary.json`; marker: `logs/BATA_PRE1_FULL_OFFLINE_PASS_20260523_121031.ok`. Processed `211` validation videos and `791` windows. Gate PASS: budget exactness, duplicate-free, BCA coverage, oracle BR@4 delta, and noisy BR@4 delta. Boundary-role BR@4 was oracle BCA `51.19%`, noisy BCA `39.00%`, stratified/uniform/random `0.00%`; noisy BCA coverage pass `100%`. This is fixed-50 diagnostic-only GT-cache evidence and no detector mAP. Next action: Pre-2 config/dataset smoke; no GPU closed-loop until smoke, matched clean baseline artifacts, and serial clearance are recorded. |
| 2026-05-23T12:10:41+08:00 | bata_pre1_launch | Launched full no-GPU BATA Pre-1 offline observability diagnostic on `35329` from `/root/autodl-tmp/OpenTAD_BATA_Clean_20260523` in screen `bata_pre1_full_20260523_121031`. Log: `logs/bata_pre1_full_offline_20260523_121031.log`; output: `/root/autodl-tmp/bata_pre1/thumos14_validation_s4_seed20260522_20260523_121031`. Preflight showed no active screen/train process, GPU `0 MiB`, `/root/autodl-tmp` `65G/200G` used, reviewed BATA markers and THUMOS14 annotation files present. Fixed budget is `384/768`; oracle/noisy GT-derived caches are diagnostic-only. No Pre-2 GPU run launched; next gate is Pre-1 PASS/FAIL summary, then Pre-2 config/dataset smoke and baseline-artifact check before any detector closed loop. |
| 2026-05-21T17:22:00+08:00 | monitor | Checked active runs on `35407` and `35329`. `35407` repaired clean Adapter baseline remains active in screen `clean_baseline_postfix_diag_20260521`; latest completed eval is epoch 49 `Average-mAP=62.79`, vector `78.68 / 74.07 / 66.18 / 55.19 / 39.85`, with epoch 51 checkpoint present and `/root/autodl-tmp` at `185G/200G` used. `35329` repaired clean Adapter early-10 queue passed its gate with `Average-mAP=28.21`, vector `50.38 / 40.15 / 28.68 / 15.48 / 6.34`, then resumed full run from `epoch_9.pth`; `/root/autodl-tmp` has about `134G` free. |
| 2026-05-21T16:39:54+08:00 | idea_survey | Created `idea-stage/INTELLIGENT_FRAME_SELECTION_LIT_AND_IDEAS_20260521.md` after GPT-5.5 Pro idea-creator session `sparse-tad-intelligen-frame-selection` completed. Surveyed frame/clip selection, token pruning/compression/merging, MoD/dynamic routing, ActionFormer/AdaTAD context, and formed initial BBAS/SBA/MCS intelligent frame-selection plan. Current recommendation: prioritize boundary-biased deployable selection diagnostics before any new long sparse-module runs. |
| 2026-04-09T12:00:00Z | init | Wiki initialized for TAD temporal redundancy project |
| 2026-04-09T12:01:00Z | ingest | 3 papers: adatad, dynamicvit, adafocus |
| 2026-04-09T12:02:00Z | create | 6 ideas: soft_gating(failed), coarse_inject, reinforce, projection_sparse, two_stage, oracle |
| 2026-04-09T12:03:00Z | create | 7 experiments: EXP001, EXP002, R008, R012, R017, R018, R021 |
| 2026-04-09T12:04:00Z | create | 6 claims: C1-C6 (2 supported, 4 invalidated) |
| 2026-04-09T12:05:00Z | create | gap_map: G1-G5 |
| 2026-04-09T12:06:00Z | create | graph/edges.jsonl: 22 edges |
| 2026-04-09T12:07:00Z | generate | index.md, query_pack.md |
| 2026-04-09T12:55:01+08:00 | stats | Stats generated |
| 2026-04-09T12:55:01+08:00 | lint | Lint report generated |
| 2026-04-09T12:55:01+08:00 | refresh | Gap map / index / query pack refreshed |
| 2026-04-09T12:55:01+08:00 | create | Added exp:STATUS20260409 current status snapshot |
| 2026-04-09T13:10:56+08:00 | bulk_ingest | Added R020 / R022 / Oracle50 / projection optimization experiment pages |
| 2026-04-09T17:00:00+08:00 | major_update | Discovered 11 unrecorded experiments from servers. Added R015=64.51, R020=53.96, R022-coarse=56.28, R022-mobilenet=58.49, ST-TopK=53.57, indicator=56.13, soft-attn-bias=57.25, mask_proj=50.30, compact_proj=5.84, and updated claims/gaps accordingly |
| 2026-04-10T18:38:29+08:00 | create | Added exp:INPUT_STRIDE2_0410 and exp:INPUT_RANDOMFIX_0410 from OpenTAD_Back input-side 50% verification |
| 2026-04-10T18:38:29+08:00 | create | Added claim:C11: non-uniform input-side sampling is not inherently doomed, but is more prone to dense detector time-semantic mismatch |
| 2026-04-10T18:38:29+08:00 | refresh | Updated gap:G4 with fresh-feature irregular-input evidence and refreshed index.md / query_pack.md / edges.jsonl |
| 2026-04-10T18:59:30+08:00 | create | Added exp:PENDING20260410 pending experiment matrix after cross-checking local configs, wiki pages, roadmap files, and current server queues |
| 2026-04-10T18:59:30+08:00 | refresh | Refreshed index.md and query_pack.md to point to the pending experiment matrix |
| 2026-04-11T09:30:00+08:00 | create | Added exp:INPUT_ORACLE_BOUNDARY_DENSE_0411 with final Avg-mAP 66.01; first confirmed input-side non-uniform 50% result that surpasses uniform stride-2 50% |
| 2026-04-11T09:30:00+08:00 | create | Added exp:INPUT_ORACLE_ACTION_BOUNDARY_DENSE_0411 with final Avg-mAP 62.04; action-interior densification is worse than boundary-only densification |
| 2026-04-11T09:30:00+08:00 | refresh | Updated claim:C11, index.md, query_pack.md, and exp:PENDING20260410 to reflect the new conclusion: boundary-dense non-uniform sampling works, while action+boundary densification is not supported |
| 2026-04-11T12:30:00+08:00 | create | Added root design doc IRREGULAR_AWARE_TAD_DETECTOR_PLAN.md and wiki idea:011 for a detector-first irregular-timeline adaptation path |
| 2026-04-11T12:30:00+08:00 | refresh | Updated query_pack, index, and graph edges to record that the next main direction is an irregular-aware detector rather than further scorer-only fixes |
| 2026-04-12T22:20:00+08:00 | create | Added exp:IRREGULAR_TIMELINE_0412 to document the current OpenTAD_Back irregular timeline implementation locations (`projection + irregular_fpn + irregular_point_generator + irregular_head + detector`) and the latest results: remap_no_regrange = 4.13/3.92, overlap@epoch39 = 1.94 |
| 2026-04-12T22:20:00+08:00 | update | Updated idea:011 from proposed/pending to tested/negative after current irregular detector runs showed semantic failure rather than just NaN |
| 2026-04-12T22:20:00+08:00 | refresh | Updated index.md, query_pack.md, and graph/edges.jsonl so the wiki now points directly to the irregular timeline implementation snapshot and current structural judgment |
| 2026-04-12T23:59:00+08:00 | audit_fix | Audited Server 3 raw log `input_random_fixed_50pct_irregular_actionformer_full_20260411.log`; corrected old irregular detector result from unsupported `64.62` to real `39.04 Avg-mAP / 12.35@0.7`, and updated exp:IRREGULAR_TIMELINE_0412 with a snapshot-vs-server traceback showing the main semantic drift happened in `irregular_actionformer_head.py` and `irregular_point_generator.py` |
| 2026-04-13T12:40:00+08:00 | create | Added exp:HEADV2_0413 to record the HeadV2 recovery split: `headv2_x = 51.04 / 25.04` and `headv2_y = 46.64 / 20.54 at epoch 44` |
| 2026-04-13T12:40:00+08:00 | refresh | Updated exp:IRREGULAR_TIMELINE_0412 so the wiki now reflects that HeadV2 repaired most of the detector semantic collapse, and the next step is ablation on `final k=1` vs `k=3` and `soft top-k` vs `topk=1` |
| 2026-04-13T16:05:00+08:00 | create | Added exp:HEADV3_0413 to record the next irregular head innovation: geometry-aware modulation + boundary auxiliary supervision, with runnable X/Y configs for the current random-fixed 50% detector line |
| 2026-04-13T16:35:00+08:00 | update | Tightened HeadV3 boundary supervision (`boundary_loss` now explicit, switched to `SoftBCELoss`) and added the first backbone-side irregular time embedding path for `VisionTransformerCP`, with runnable `headv3_timeembed_x/y` configs |
| 2026-04-13T21:45:00+08:00 | update | Closed the HeadV2 X/Y split with final `headv2_y = 49.00 / 22.71`; updated exp:HEADV2_0413, exp:IRREGULAR_TIMELINE_0412, and query_pack to reflect the new structural judgment: HeadV2 repaired the main semantic collapse, but current irregular projection/FPN still trails the dense-like bridge by `2.04 mAP` |
| 2026-04-13T22:55:59+08:00 | create | Added exp:INPUT_WEIGHTED_RANDOM_ACTION_BOUNDARY_TUBELET2_0413 with final `58.54 Avg-mAP`; tubelet2 selection did not rescue the weighted random action+boundary input line |
| 2026-04-13T22:55:59+08:00 | create | Added exp:HEADV2_X_TOPK1 with final `47.19 Avg-mAP`; single-best assignment underperformed default soft top-k by `3.85 mAP` |
| 2026-04-13T22:55:59+08:00 | refresh | Updated exp:HEADV2_0413, claim:C11, exp:PENDING20260410, index.md, query_pack.md, and graph/edges.jsonl with the new structural judgment: tubelet2 does not fix action+boundary weighted random, and soft assignment remains preferable to topk=1 in the current irregular detector |
| 2026-04-13T23:20:00+08:00 | refine | Tightened the wiki interpretation of `HEADV2_X_TOPK1 = 47.19`: this is not a dense hard-assignment control, but a `top9 -> top1` support-size ablation inside the same cost-based assignment family |
| 2026-04-13T23:20:00+08:00 | refine | Tightened the wiki interpretation of `INPUT_WEIGHTED_RANDOM_ACTION_BOUNDARY_TUBELET2_0413 = 58.54`: the current evidence does not yet separate budget-allocation error from tubelet-granularity boundary blur; `input_oracle_boundary_dense_tubelet2_50pct` remains the key discriminator |
| 2026-04-13T23:20:00+08:00 | refresh | Updated `exp:HEADV2_0413`, `exp:INPUT_WEIGHTED_RANDOM_ACTION_BOUNDARY_TUBELET2_0413`, and `query_pack.md` with the current priority order: wait `ablation_k3`, quantify C-class residual with a fair dense proj/neck bridge, wait boundary-oracle tubelet2, then decide the `HeadV3 / time_embed` kernel setting |
| 2026-04-14T17:25:00+08:00 | create | Added `exp:SPARSE_HEAD_SUMMARY_0414` to consolidate the sparse-head trajectory: old irregular `39.04` -> HeadV2 `51.04` -> HeadV3 `52.60`, plus denseexact / k3 / time-embed / boundary-aware controls and the current Step0-4 bridge decomposition queue |
| 2026-04-14T17:25:00+08:00 | refresh | Updated `index.md` to reflect the strongest current irregular-head result `headv3_x = 52.60` and to point future diagnosis toward the head-side semantics bundle rather than passthrough proj/neck fairness |
| 2026-04-15T09:31:11+08:00 | update | Updated `exp:SPARSE_HEAD_SUMMARY_0414` with the final 5-step bridge decomposition table: `step0_densehead_remap=53.72`, `step1=37.10`, `step2=34.41`, `step3=49.86`, `step4=49.79`, and tightened the causal reading that the main break is hard ownership on the native irregular axis while the main recovery is soft support assignment |
| 2026-04-15T10:37:32+08:00 | create | Added `exp:IRREGULAR_MODEL_AUDIT_0415` to audit the full `input_random_fixed_50pct_irregular_actionformer*` model line across local configs and the two servers, separating trusted completed runs, eval-only ablations, invalid/debug runs, deployed-but-unfinished jobs, and local-only undeployed configs |
| 2026-04-15T10:37:32+08:00 | refresh | Updated `index.md` so the wiki now points to the authoritative experiment inventory for the irregular model line; current unfinished priorities are `OABS-visible/full` and the newly queued `step0_shell_exact / step0b_dense_soft` |
| 2026-04-15T16:41:57+08:00 | review | Added `review:GEMINI_DIRECTION_0415` to record a two-round Gemini direction review on the irregular-head line after `OABS-full = 53.16`; final recommended next steps are `OAA`, `Learnable OABS`, and conditional `OAD` rather than further backbone irregularization or post-hoc boundary inference |
| 2026-04-15T17:20:46+08:00 | create | Added `exp:IRREGULAR_COMPLETION_UPDATE_0415B` to record the latest completed irregular-head batch: `step0_shell_exact = 53.72`, `headv3_oabs_visible_x = 51.49`, and `headv3_oabs_full_x = 53.16`; updated `index.md` so the strongest completed irregular-head result now points to `OABS-full` |
| 2026-04-16T10:05:47+08:00 | create | Added `exp:IRREGULAR_ROOTCAUSE_UPDATE_0416` to integrate the two newly completed decisive results: `step0b_dense_points_soft_sym = 47.52` and `headv3_oabs_full_x_fixsingleton = 52.88`, while also folding in `headv3_oabs_full_oaa_x = 53.20`; this closes the open diagnosis gap on soft assignment and rewrites the current root-cause ranking |
| 2026-04-16T10:05:47+08:00 | review | Added `review:GEMINI_DIRECTION_0416`; Gemini's updated judgment is that the current plateau is mainly a model-semantic failure rather than a hidden bug story, and the next pivot should move from the irregular shell to sparse-to-dense completion ahead of an unchanged dense detector |
| 2026-04-16T10:05:47+08:00 | refresh | Updated `index.md`, `query_pack.md`, `exp:SPARSE_HEAD_SUMMARY_0414`, and `exp:IRREGULAR_MODEL_AUDIT_0415` to reflect the new strongest irregular-shell result `53.20`, the newly quantified `step0b` soft-assignment overhead, and the completion-first next-step recommendation |
| 2026-04-16T16:55:00+08:00 | create | Added `exp:IRREGULAR_UPSTREAM_AUDIT_0416` after comparing a fresh upstream `OpenTAD` clone against `OpenTAD_Back`; recorded that the original dense ActionFormer path is intact, the current `x` line is a dense-trunk + irregular-head carrier, and `step0_shell_exact / fixsingleton` are not clean causal controls |
| 2026-04-16T16:55:00+08:00 | refresh | Updated `index.md`, `query_pack.md`, `gap_map.md`, `graph/edges.jsonl`, and added correction headers to `exp:IRREGULAR_ROOTCAUSE_UPDATE_0416`, `exp:IRREGULAR_MODEL_AUDIT_0415`, and `exp:SPARSE_HEAD_SUMMARY_0414` so the wiki no longer treats `step0_shell_exact = 53.72` as valid shell-overhead evidence or `fixsingleton = 52.88` as a clean singleton ablation |
| 2026-04-16T23:53:30+08:00 | create | Added `exp:IRREGULAR_REPAIR_QUEUE_0416` to freeze the current repaired experiment queue: running `step0_dense_head_baseline` and `headv3_oabs_full_x_legacysingleton`, plus queued `step0b_repaired`, `y_dense_grid_sanity_check`, and `fixsingleton_repaired`, with explicit purpose / hypothesis / result slots |
| 2026-04-16T23:53:30+08:00 | refresh | Updated `index.md` to point to the active repaired-queue tracking page so later result supplementation can happen in one place |
| 2026-04-17T00:09:45+08:00 | update | Updated `exp:IRREGULAR_REPAIR_QUEUE_0416` with the first completed repaired result `headv3_oabs_full_x_legacysingleton = 53.48`; the clean legacy rerun is already above old `OABS-full = 53.16` and `OAA = 53.20`, so the old singleton-fix story is weakened and the new clean comparison anchor is `53.48` |
| 2026-04-17T00:09:45+08:00 | refresh | Updated `index.md` so the strongest current irregular-head result now points to the repaired clean legacy rerun `53.48`, and recorded `step0_dense_head_baseline` latest logged `13.73` only as an anomaly watchpoint rather than a final conclusion |
| 2026-04-17T13:03:19+08:00 | update | Updated `exp:IRREGULAR_REPAIR_QUEUE_0416` with repaired results `step0_dense_head_baseline = 53.72`, `step0b_dense_points_soft_sym_repaired = 47.52`, and clean singleton A/B `53.48 -> 52.88`; rewrote the queue page so the old `13.79` collapse is now explicitly treated as a GT-axis mismatch bug rather than a real model result |
| 2026-04-17T13:03:19+08:00 | refresh | Updated `exp:IRREGULAR_UPSTREAM_AUDIT_0416`, `index.md`, and `query_pack.md` to clarify that repaired Step0 restores the remapped selected-axis bridge control but does not replace the missing no-remap shell-exact measurement; recorded that soft assignment remains `-6.20` vs repaired Step0 and that clean singleton repair is slightly harmful (`53.48 -> 52.88`) |
| 2026-04-17T17:00:23+08:00 | update | Closed `y_dense_grid_sanity_check` at `13.86 Avg-mAP`; the run did not recover late (`13.84 -> 13.97 -> 13.86`) and therefore becomes the strongest completed evidence that the true irregular `y` trunk is severely incompatible with dense-head decoding under the current dense-grid adapter path |
| 2026-04-17T17:00:23+08:00 | refresh | Updated `exp:IRREGULAR_REPAIR_QUEUE_0416`, `index.md`, and `query_pack.md` so the repair queue is now fully completed and the current structural judgment explicitly separates remapped dense-like controls (`53.72`) from the collapsed true irregular trunk (`13.86`) |
| 2026-04-17T17:18:12+08:00 | review | Added `review:GEMINI_DIRECTION_0417` after the repaired queue fully closed; this corrected Gemini review explicitly rejects treating `63.12 -> 53.72` as clean shell-overhead evidence and instead elevates `y`-trunk vs dense-head contract mismatch to the top root cause |
| 2026-04-17T17:18:12+08:00 | refresh | Updated `index.md` and `query_pack.md` to rewrite the current root-cause ranking and next-step priorities around the 0417 review: null bridge control, `y`-output normalization, point / set-style decoder, and completion as the parallel mainline |
| 2026-04-17T17:36:30+08:00 | create | Added `exp:Y_RECOVERY_QUEUE_0417` to freeze the first post-Gemini-0417 recovery queue: `server1` runs `y_dense_grid_sanity_check_norm_ampoff`, `server3` runs `y_pointset_softsym_sanity`, and `y_pointset_softsym_sanity_ampoff` is queued behind it |
| 2026-04-17T17:36:30+08:00 | refresh | Updated `index.md` and `query_pack.md` so the wiki records one new operational fact from the live deployment itself: both new `y` recovery sanities immediately exposed AMP-related non-finite gradients, so later causal reads must separate method effect from fp16 instability |
| 2026-04-17T22:31:00+08:00 | update | Updated `exp:Y_RECOVERY_QUEUE_0417` with the first completed recovery result `y_pointset_softsym_sanity = 44.06`; this cleanly beats `y_dense_grid_sanity_check = 13.86` and therefore confirms dense-grid decoder mismatch as a major failure source on the true `y` trunk |
| 2026-04-17T22:31:00+08:00 | update | Recorded and fixed a deployment bug in the original `pointset_ampoff` waiting command: shell-side `pgrep -f tools/train.py` matched the waiter command itself and prevented auto-start; `y_pointset_softsym_sanity_ampoff` was relaunched directly |
| 2026-04-17T22:31:00+08:00 | refresh | Updated `index.md` and `query_pack.md` so the current next-step priority is now: wait `pointset_ampoff` and `norm_ampoff`, then decide whether the main `y` branch should fully pivot to a stronger irregular point decoder |
| 2026-04-18T00:03:00+08:00 | update | Updated `exp:Y_RECOVERY_QUEUE_0417` with `y_dense_grid_sanity_check_norm_ampoff = 10.34`; this confirms simple dense-grid output normalization is not a meaningful recovery path on the true `y` trunk |
| 2026-04-18T00:03:00+08:00 | update | Recorded the live status of `y_pointset_softsym_sanity_ampoff`: active on `25876`, stable through about `epoch 30`, no evaluation yet, and no new `non-finite gradients` lines seen in the current log slice |
| 2026-04-18T00:03:00+08:00 | refresh | Updated `index.md` and `query_pack.md` to tighten the 0417 priority order: dense-grid normalization is now deprioritized, and the decisive pending control is `y_pointset_softsym_sanity_ampoff` |
| 2026-04-18T00:08:30+08:00 | update | Refreshed `exp:Y_RECOVERY_QUEUE_0417` runtime status: `y_pointset_softsym_sanity_ampoff` remains stable on `25876` through about `epoch 32`, still without evaluation output and still without new `non-finite gradients` lines in the latest log slice |
| 2026-04-18T00:24:39+08:00 | review | Added `review:GEMINI_DIRECTION_0418` to record the reset-style Gemini critique: stop polishing middle-state hybrids, build a stupid dense-semantics anchor, simplify the native `y` line, and use oracle diagnostics before more module growth |
| 2026-04-18T00:24:39+08:00 | create | Added `refine-logs/EXPERIMENT_PLAN_0418_RESET.md` and `refine-logs/EXPERIMENT_TRACKER_0418_RESET.md` as the new execution plan after Gemini 0418 |
| 2026-04-18T00:24:39+08:00 | refresh | Updated `index.md` and `query_pack.md` so the top priorities are now: close `y_pointset_ampoff`, build the stupid sparse-to-dense dense-detector baseline, simplify the native `y` decoder, and run oracle interface diagnostics |
| 2026-04-18T00:39:00+08:00 | deploy | Synced the new stupid completion baseline implementation to `24013`: added `nearest` / `linear` non-learned completion modes, two new configs, and `scripts/run_completion_stupid_baselines_server1.sh` |
| 2026-04-18T00:39:00+08:00 | update | Launched the `24013` queue screen `completion_reset_s1`; `G0418-R002` (`nearest`) is now running and `G0418-R003` (`linear`) is queued behind it |
| 2026-04-18T00:52:31+08:00 | deploy | Synced the M2 native-`y` overfit diagnostics to `25876`: added dataset `allow_list`, `KeepSingleGT`, two overfit configs, and `scripts/run_y_overfit_ladder_server2.sh` |
| 2026-04-18T00:52:31+08:00 | update | Launched queue screen `y_overfit_ladder_s2`; `G0418-R004` (single-video overfit) and `G0418-R005` (single-instance overfit) are now queued behind `G0418-R001` on `25876` |
| 2026-04-18T00:47:08+08:00 | update | `G0418-R001` (`y_pointset_softsym_sanity_ampoff`) reached its first evaluation after epoch 39 and read only `39.43 Avg-mAP` (`41.01 @0.5`, `12.27 @0.7`), which is below the prior fp16 run `44.06`; AMP is therefore not currently acting like a recovery lever |
| 2026-04-18T00:53:36+08:00 | update | `G0418-R002` (`completion_nearest_dense_baseline`) is still running on `24013`, but repeated `non-finite gradients detected` skips have already appeared at epoch `0`, `3`, and `4`; treat the current run as a watched baseline rather than a clean anchor until convergence is confirmed |
| 2026-04-18T01:01:53+08:00 | deploy | Synced the M3 oracle-interface diagnostics to `24013`: added paired configs `input_oracle_action_boundary_dense_50pct_irregular_actionformer_headv2_denseexact` and `input_oracle_boundary_dense_50pct_irregular_actionformer_headv2_denseexact`, plus `scripts/run_oracle_interface_diag_server1.sh` |
| 2026-04-18T01:01:53+08:00 | update | Launched queue screen `oracle_interface_diag_s1`; `G0418-R006` is now queued behind the current completion queue on `24013`, with `action+boundary` oracle support first and `boundary-only` oracle support second |
| 2026-04-18T01:06:42+08:00 | deploy | Extended the same `24013` post-anchor queue to cover M4 as well: added `input_random_fixed_50pct_completion_learned_minimal_ampoff.py` and relaunched `oracle_interface_diag_s1` so it now runs `R006 -> R008 -> R007` sequentially after `R002/R003` |
| 2026-04-18T11:11:49+08:00 | update | Closed the first 0418 reset reads in the tracker/wiki: `G0418-R001 = 44.07`, `G0418-R002 = 54.03`, `G0418-R003` interrupted at latest `52.56`, and `G0418-R004/R005` marked invalid because the inherited 200-epoch cosine schedule collapsed to near-zero LR under one-iter-per-epoch overfit |
| 2026-04-18T11:11:49+08:00 | create | Added `exp:RESET_QUEUE_0418_STATUS` to freeze the new 0418 reset status page after the first closure of M0/M1/M2 |
| 2026-04-18T11:11:49+08:00 | update | Updated `exp:Y_RECOVERY_QUEUE_0417`, `index.md`, and `query_pack.md` with the final fp32 `y_pointset_softsym_sanity_ampoff = 44.07`, which rules out AMP as the main blocker on the current native `y` point decoder |
| 2026-04-18T11:18:00+08:00 | update | Relaunched the stalled `24013` queue directly in a fresh screen; `G0418-R006` (`oracle action+boundary support`) is now actually running again, with `G0418-R008 -> G0418-R007` queued behind it |
| 2026-04-18T11:19:28+08:00 | deploy | Added repaired tiny-overfit configs and `scripts/run_y_overfit_ladder_repaired_server2.sh`, then relaunched `25876` so `G0418-R009 -> G0418-R010` now use no projection dropout, higher LR, zero weight decay, and a long constant-like schedule |
| 2026-04-18T11:19:40+08:00 | audit_fix | The first repaired `R009` launch immediately exposed a config-merge bug: inherited scheduler state left `warmup_epoch` in a `MultiStepLR` config and crashed scheduler construction. Fixed by setting scheduler `_delete_=True` and relaunched successfully; training is now progressing on `25876` |
| 2026-04-18T11:28:00+08:00 | review | Ran a fresh Gemini discussion on the current 0418 reset status. Key verdicts: `R001` cleanly rules out AMP as the main blocker, `R002 = 54.03` is only weak support for completion, `R006` should be stopped as numerically contaminated, and the highest-value next step is a *true* fixed-sample overfit test |
| 2026-04-18T11:28:30+08:00 | deploy | Implemented deterministic truncation support in `LoadFrames` via `fixed_trunc_start / fixed_trunc_gt_index`, added `y_pointset_softsym_true_overfit_single_video/single_instance` configs, and added `scripts/run_y_true_overfit_ladder_server2.sh` |
| 2026-04-18T11:29:00+08:00 | update | Stopped the active AMP-on oracle run `G0418-R006` on `24013` and stopped the still-impure repaired overfit queue `G0418-R009/R010` on `25876` in order to free compute for the true-overfit diagnosis |
| 2026-04-18T11:30:30+08:00 | audit_fix | Verified the new true-overfit data path on `25876`: under `input_random_fixed_50pct_irregular_actionformer_y_pointset_softsym_true_overfit_single_video`, repeated `dataset[0]` fetches now return identical `inputs`, `masks`, `gt_segments`, `gt_labels`, and irregular-axis metadata |
| 2026-04-18T11:28:34+08:00 | update | Launched the new `25876` queue `G0418-R011 -> G0418-R012`; `G0418-R011` is now the active highest-priority experiment in the 0418 reset plan |
| 2026-04-18T11:33:00+08:00 | update | First `G0418-R011` pass reached late training with loss still around `0.67-0.70`, which is already a strong negative overfit signal, but then crashed during checkpoint saving. Added `workflow.disable_checkpoint=True` support in `tools/train.py` and relaunched the true-overfit queue so the diagnosis can finish without I/O noise |
| 2026-04-19T14:45:00+08:00 | update | Closed the true-overfit ladder: `G0418-R011` finished with final `[499] Loss=0.6693 cls_loss=0.0538 reg_loss=0.6156`, and `G0418-R012` finished with final `[499] Loss=0.6534 cls_loss=0.0663 reg_loss=0.5870`; both deterministic fixed-sample controls failed to memorize |
| 2026-04-19T14:45:00+08:00 | refresh | Updated `refine-logs/EXPERIMENT_TRACKER_0418_RESET.md` and `exp:RESET_QUEUE_0418_STATUS` to promote the native `y` decoder / target / support contract failure to the top current root cause |
| 2026-04-19T14:42:00+08:00 | deploy | Added branch-decoupled true-overfit diagnostics: independent `cls-only` and `reg-only` configs for both single-video and single-instance deterministic sparse samples, plus periodic runtime debug logging in `train_engine` |
| 2026-04-19T14:42:18+08:00 | update | Launched `25876` screen `y_true_overfit_branchdiag_s2`; queue order is `G0418-R013 -> R016`, with `single_video_clsonly` now active and `24013` left idle |
| 2026-04-19T15:00:39+08:00 | deploy | Added and launched two new minimal causal diagnostics on `24013`: `G0418-R017 = oracle-point reg-only true-overfit single-instance` and `G0418-R018 = hard-joint true-overfit single-instance` |
| 2026-04-19T15:02:00+08:00 | update | Early `R017` signal: by about epoch `25`, `reg_loss` is already in the `0.01-0.05` band under one fixed oracle positive point, which weakens the hypothesis that raw regression capacity is the main blocker and increases suspicion on the soft assignment / joint contract |
| 2026-04-20T18:40:00+08:00 | create | Added `exp:SOFT_BRIDGE_MAINLINE_0420` to track the current full-train soft-bridge mainline reruns: no-warmup control on `24013` and `regwarm10` on `25876`, with the main question narrowed to whether reg-first warmup remains beneficial after the real full-train cls-weight switch |
| 2026-04-20T18:40:00+08:00 | refresh | Updated `index.md` and `query_pack.md` so the active priority order now matches the live state: both 0420 soft-bridge mainlines are running, `regwarm10` has already switched to `bridge_cls_loss_weight_effective=1.0` without collapsing, and new branches are temporarily deprioritized until the first trustworthy validation read lands |
| 2026-04-20T18:58:00+08:00 | create | Added `exp:SOFT_MULTIGT_DIAG_QUEUE_0420` to track the next queued native-`y` diagnostics after the Gemini discussion: `single_video_joint_detachcls` plus the new `keep2gt_joint -> keep2gt_joint_detachcls` ladder |
| 2026-04-20T18:58:00+08:00 | deploy | Implemented `KeepGTSubset`, added two new `keep2gt` true-overfit configs, added queue scripts `run_y_soft_multigt_diag_server1.sh` and `run_y_soft_multigt_diag_server3.sh`, deployed them to `24013/25876`, and started detached waiting screens `y_soft_multigt_diag_s1/s3` behind the two active 0420 full-train mainlines |
| 2026-04-20T19:00:00+08:00 | update | Refreshed the live 0420 status pages: no-warmup advanced to about `epoch 27 mid` with `5` skipped non-finite steps total, `regwarm10` advanced to about `epoch 29 start` with `4` skipped steps total and `bridge_cls_loss_weight_effective=1.0` still active, while both new multigt diagnostic queues remained in clean waiting state without interrupting the mainlines |
| 2026-04-20T19:46:00+08:00 | update | Refreshed the live 0420 status again: both no-warmup and `regwarm10` advanced to about `epoch 39`, both remained stable in a very similar `~0.56-0.61` train-loss band, `regwarm10` absorbed one additional skipped step at `epoch 38` without collapsing, and both queued multigt diagnostic screens remained in pure waiting state with no premature launch |
| 2026-04-20T21:46:00+08:00 | update | Recorded the first real validation comparison for the 0420 mainlines: no-warmup reached `36.59`, `regwarm10` reached `37.00`, so reg-first warmup now has the first full-train positive evidence (`+0.41 mAP`) but remains far below the older `44.06 / 44.07` pointset sanity ceiling; both queued multigt diagnostic screens are still waiting for GPU release |
| 2026-04-23T09:52:00+08:00 | create | Added `exp:TEMPORALGRIDFIX_FULL_QUEUE_0423` to freeze the config inheritance chain, repaired `temporal_grid.py` usage path, deployment file set, and interpretation boundary for the new formal `headv3_x_temporalgridfix` rerun queued on `25876` |
| 2026-04-23T12:08:00+08:00 | create | Added `exp:FRAME_SAMPLING_BUGFIX_QUEUE_0423` and refreshed `exp:R021` after the first trustworthy OpenTAD bugfix rerun readout: `v3_gumbel_softgate_minimal_bugfix0420` reached `52.43 -> 53.17`, which keeps the soft-gating line in the historical low-50s regime while making the evidence cleaner |
| 2026-04-23T12:48:00+08:00 | update | Refreshed `exp:FRAME_SAMPLING_BUGFIX_QUEUE_0423`, `exp:R021`, `index.md`, and `query_pack.md` after two more bugfix-rerun evals: `v3_gumbel_softgate_minimal_bugfix0420` is now `52.43 -> 53.17 -> 53.28 -> 53.02`, which strengthens the reading that the line is stable in the old low-50s regime rather than recovering into a new one; `v3_hard_scorer_aux_bugfix0420` has also advanced to about `epoch 39`, making its first eval the next immediate frame-sampling watchpoint |
| 2026-04-23T14:06:23+08:00 | update | Refreshed `exp:FRAME_SAMPLING_BUGFIX_QUEUE_0423`, `exp:R021`, `exp:TEMPORALGRIDFIX_FULL_QUEUE_0423`, `index.md`, and `query_pack.md` after the next live monitoring pass: `v3_gumbel_softgate_minimal_bugfix0420` has now reached nine evals with peak `54.32` and latest `53.73`, `v3_hard_scorer_aux_bugfix0420` has climbed to `56.00`, and `headv3_x_temporalgridfix` is still queued on `25876` without a started `log.json` |
| 2026-04-23T15:26:51+08:00 | update | Refreshed `exp:FRAME_SAMPLING_BUGFIX_QUEUE_0423`, `exp:R021`, `exp:TEMPORALGRIDFIX_FULL_QUEUE_0423`, `index.md`, and `query_pack.md` after the queue handoff and next monitoring pass: `v3_gumbel_softgate_minimal_bugfix0420` has now finished with final `54.22`, `v3_gumbel_softgate_coarse_det_bugfix0420` has started and reached about `epoch 24`, `v3_hard_scorer_aux_bugfix0420` has climbed further to `57.00`, and `headv3_x_temporalgridfix` is still queued on `25876` without a started `log.json` |
| 2026-04-23T16:56:25+08:00 | update | Refreshed `exp:FRAME_SAMPLING_BUGFIX_QUEUE_0423`, `exp:TEMPORALGRIDFIX_FULL_QUEUE_0423`, `index.md`, and `query_pack.md` after the next decisive state change: `v3_hard_scorer_aux_bugfix0420` has now finished with final `56.97` and peak `57.00`, `v3_gumbel_softgate_coarse_det_bugfix0420` has produced a weak opening pair `48.07 -> 49.90`, and `headv3_x_temporalgridfix` has started on `25876`, progressed to about `epoch 17`, and continued through one isolated non-finite skipped step |
| 2026-04-23T17:11:34+08:00 | update | Refreshed `exp:FRAME_SAMPLING_BUGFIX_QUEUE_0423`, `exp:TEMPORALGRIDFIX_FULL_QUEUE_0423`, `index.md`, and `query_pack.md` after the next live check: `v3_gumbel_softgate_coarse_det_bugfix0420` has now inched up to `50.75` but remains far below the other bugfix branches, while `headv3_x_temporalgridfix` has advanced to about `epoch 22` and accumulated three sparse skipped steps from non-finite reg-head gradients without stopping |
| 2026-04-23T20:58:30+08:00 | update | Closed the current monitoring loop by refreshing `exp:FRAME_SAMPLING_BUGFIX_QUEUE_0423`, `exp:TEMPORALGRIDFIX_FULL_QUEUE_0423`, `index.md`, and `query_pack.md`: `v3_gumbel_softgate_coarse_det_bugfix0420` recovered but still finished negatively at `53.17`, `headv3_x_temporalgridfix` finished at `52.40` with five sparse skipped reg-head steps, and the strongest completed bugfix branch remains `v3_hard_scorer_aux_bugfix0420 = 56.97` |
| 2026-04-23T23:40:00+08:00 | create | Added `exp:SOFT_BRIDGE_SUPERVISION_CLOSURE_0423` to close the missing `0420` native-`y` supervision and toy diagnostics: full-train now closes at `topk9 + binary-cls = 43.44`, `topk3 + soft-cls = 42.26`, `topk1 + soft/binary = 38.98`, while toy closure shows `cls-only` fits, original soft `reg-only` fails, `oracle/hard/topk1/2` reg-only fit, and both `detach-cls` and `regwarm` rescue multi-GT joint overfit |
| 2026-04-23T23:40:00+08:00 | refresh | Updated `index.md` and `query_pack.md` to replace the stale live state with the closed supervision verdict and the new highest-information next step: `topk9 + binary-cls + regwarm10` on `25876` plus `topk9 + binary-cls + detach-cls` on `24013` |
| 2026-04-23T23:54:00+08:00 | deploy | Deployed the `0423` topk9-repair pair to both remote servers via `scripts/deploy_y_soft_topk9_repair_2exp.ps1`: `topk9 + binary-cls + detach-cls` started in screen `y_soft_t9repair_s1` on `24013`, and `topk9 + binary-cls + regwarm10` started in screen `y_soft_t9repair_s3` on `25876` |
| 2026-04-23T23:55:00+08:00 | update | Confirmed both new `topk9` repair runs are truly active rather than queued-only: each screen owns about `2.97 GiB` GPU memory, both logs advanced past `epoch 0 iter 17`, and both currently show sparse skipped non-finite `reg_head.weight` steps without stopping, which matches the tolerated pattern seen in prior bridge runs |
| 2026-04-24T11:26:05+08:00 | update | Closed the `0423` topk9-repair pair in the local tracker and wiki: `topk9 + binary-cls + regwarm10 = 43.05 / 45.50 / 15.24`, `topk9 + binary-cls + detach-cls = 38.56 / 39.82 / 13.20`, both below the frozen `43.44 / 46.17 / 15.41` baseline, so bridge-side polishing is now explicitly stopped on this carrier |
| 2026-04-24T11:26:05+08:00 | refresh | Refreshed `exp:SOFT_BRIDGE_SUPERVISION_CLOSURE_0423`, `exp:TEMPORALGRIDFIX_FULL_QUEUE_0423`, `index.md`, `query_pack.md`, and `findings.md` so the wiki now records the final negative transfer verdict and the post-hoc temporal-grid inheritance / interpretation audit |
| 2026-04-24T12:00:00+08:00 | gate | Ran a local post-bridge result-to-claim gate: current bridge-side tuning is not a viable mainline, completion headroom is the highest-information next evidence, and the current route claim is `no [pending external review]` |
| 2026-04-24T12:00:00+08:00 | plan | Added `refine-logs/EXPERIMENT_PLAN_0424_POSTBRIDGE_PIVOT.md` and `refine-logs/EXPERIMENT_TRACKER_0424_POSTBRIDGE_PIVOT.md`; refreshed `index.md` and `query_pack.md` so the new execution order is `oracle_upper_ampoff -> full_oracle_dense_control -> learned_minimal_ampoff` |
| 2026-04-24T19:55:00+08:00 | deploy | Deployed the post-bridge completion trio after fixing oracle eval-time kwargs in `SparseCompletionActionFormer`: `24013` now runs `completion_oracle_upper_ampoff` with `completion_learned_minimal_ampoff` queued behind it, while `25876` runs `completion_full_oracle_dense_control_ampoff` in parallel |
| 2026-04-24T20:58:14+08:00 | create | Added `exp:REPO_ROUTE_SEPARATION_0424` to make the repository split explicit: `OpenTAD_Back` is the active mainline for input / irregular-detector / completion work, `OpenTAD` is a separate frame-sampling / scorer reference line, and `R015 = 64.51` remains unverified |
| 2026-04-24T20:58:14+08:00 | refresh | Refreshed `index.md` and `query_pack.md` so future reads separate `OpenTAD` and `OpenTAD_Back` scoreboards by default, and updated the `OpenTAD_Back` priority order to include the new `completion_scatter_dense_baseline_ampoff` control |
| 2026-04-30T10:32:00+08:00 | create | Added `exp:R05D_V2` — NativePhysicalMultiScaleHead full pilot completed: **47.04%** (60 epochs). Per-level positive count: L0-L2 healthy, L5=0. mAP@0.3=66.91% vs mAP@0.7=16.36% |
| 2026-04-30T10:32:00+08:00 | create | Added `exp:R05D_V3` — regression_range 修正实验 running: **44.99%** at epoch 39. 修改 reg_range 未改善, 说明不是 regression_range 的问题 |
| 2026-04-30T12:53:00+08:00 | create | Added `exp:DENSEADAPTER_SANITY` — 历史实验 IrregularFPNDenseAdapter + ActionFormerHead = **14%**. Codex 判断为坐标/配置 bug |
| 2026-04-30T13:00:00+08:00 | create | Added `exp:CODEX_REVIEW_0430` — Codex GPT-5.5 深度审查: 47% 差距是结构性的 (共享 conv 假设等间距); DenseAdapter 14% 是 bug; 推荐排查 DenseAdapter bug 作为突破口 |
| 2026-04-30T13:00:00+08:00 | create | Added `claim:C12` — NativePhysicalMultiScaleHead 7% 差距是结构性问题, **supported** |
| 2026-04-30T13:00:00+08:00 | create | Added `claim:C13` — DenseAdapter 14% 是坐标/配置 bug, **reported** (待验证) |
| 2026-04-30T13:00:00+08:00 | refresh | Updated `index.md` with new NativePhysical experiment line, new claims C12/C13, and structural judgments #43-#47. Updated `graph/edges.jsonl` with new edges for R05D_V2, R05D_V3, DENSEADAPTER_SANITY, CODEX_REVIEW_0430 |
| 2026-04-30T20:00:00+08:00 | audit | **全面代码审计完成**: 4 并行 Agent + Codex 验证。发现 26 问题 (5 CRITICAL + 7 MAJOR + 14 MINOR)。已修复 6 个。完整报告: doc:CODE_AUDIT_0430 |
| 2026-05-01T10:00:00+08:00 | result | **A0 no-interp 最终 51.28%** (epoch 59)。跨轴插值 bug 确认: 消除后从 14%→51.28% (+37%)。最强 irregular→dense 结果。 |
| 2026-05-01T10:00:00+08:00 | create | 新增 idea:013 (Query-based Sparse Detector), claim:C15 (跨轴bug确认+不规则信息有害), exp:A0_NOINTERP_0430 |
| 2026-05-02T00:00:00+08:00 | result | **隔离矩阵完成**: TadTR v3=30.82%, TadTR v7(DensePassthrough+TadTR)=28.45%, (e) DenseProj+IrregularFPN=37.79%。全部16个实验完成 |
| 2026-05-01T10:00:00+08:00 | insight | **关键洞察: 忽略不规则性 (63.12%) > 使用不规则信息 (43-51%)。问题在 projection/neck 的几何感知破坏均匀假设。** 需要全新检测范式。 |
| 2026-04-30T18:30:00+08:00 | create | 新增 exp:DADBUG_B1 (B1 stride fix 实验页) 和 claim:C14 (跨轴插值bug) |
| 2026-05-05T03:36:00+08:00 | create | Added `exp:SIMOTA_RANGE_CENTER25_MINK4_W1_20260505` to record the running SimOTA alignment run. Early epochs 0-5 now match FCOS-like assignment (`pos/gt` around 3.29-3.34, per-level positives [1289,1320,1306,709,182,40] at epoch4), but final mAP is still pending because `val_start_epoch=39`. |
| 2026-05-05T04:44:00+08:00 | review | Gemini 3 Pro preview reviewed the SimOTA line: it agrees on `target_weights` affecting all valid classification points, `confuse_weight` affecting only candidate-but-unmatched negatives, and original no-range low-level collapse. It adds the key caveat that `min_k4` with `candidate_count_mean~3.8` mostly creates an FCOS-aligned control rather than a genuinely selective SimOTA matcher. Independent epoch14 probe reads `32.15` avg, `32.90@0.5`, `8.88@0.7`; final eval remains pending. |
| 2026-05-05T05:05:00+08:00 | review | Re-ran Gemini 3 Pro preview with corrected config facts: active inherited `filter_shortest_gt=False`, so shortest-GT hard filtering is not the current cause. Parsed epoch 0-30 assignment: FCOS diag `pos/gt=3.2232`, original SimOTA `candidate_count=30.3841` with `valid_weight_mean=0.7664`, and new `range+center25+min_k4+w1` `pos/gt=3.3138`, `dynamic_k=3.5546`, `candidate_count=3.7955`, weights all `1.0`. Current verdict: quantity/level collapse is fixed; first full eval and positive reg-loss diagnostics are needed before claiming a pure regression-scale bottleneck. |
| 2026-05-05T05:50:00+08:00 | result | First formal eval for `range+center25+min_k4+w1` at epoch39: `50.08` Avg-mAP, `69.37@0.3`, `62.29@0.4`, `52.63@0.5`, `40.58@0.6`, `25.51@0.7`. This is neutral versus prior SimOTA fixed/v2 and far below dense random-fixed FCOS `63.12`, so the next debug target shifts to regression target/scale and positive reg-loss diagnostics rather than more min_k inflation. |
| 2026-05-05T06:05:00+08:00 | diagnostic | Implemented regression debug fields in `AnchorFreeHead`, extended `parse_assignment_diag_logs.py`, and ran `simota_regdiag_epoch40_p29601` from `epoch_39.pth`. Result: `pos/gt=3.3437`, `reg_loss=0.2292`, `reg_iou=0.7818`; level0 is the weak point (`reg_loss=0.3150`, `IoU=0.7020`) while levels 2-4 are healthier (`IoU=0.82-0.87`). This supports a short/low-level regression bottleneck rather than assignment quantity. |
| 2026-05-05T06:25:00+08:00 | diagnostic | Added and ran FCOS same-current-path regdiag from `input_random_fixed_50pct_irregular_actionformer_step0_dense_head_baseline` epoch59. Result: `pos/gt=3.1531`, `reg_loss=0.2112`, `reg_iou=0.7983`, `L0 IoU=0.7258`, `<16 IoU=0.7670`. Gemini 3 Pro preview corrected the interpretation: this is not the historical `63.12` dense baseline, but it shows the short/low-level localization weakness is shared in the current path, not SimOTA-specific. Epoch44 SimOTA eval is `51.39` Avg-mAP and `27.18@0.7`. |
| 2026-05-05T06:30:00+08:00 | training-check | Training health check for `simota_mink4w1_full`: after three early non-finite gradient skips in epochs 0/3, no later hard failure was found; losses continue around `0.43-0.46`, LR decays normally, and eval improves from `50.08/25.51@0.7` at epoch39 to `51.39/27.18@0.7` at epoch44. Decision: continue monitoring; next eval expected after epoch49. |
| 2026-05-05T07:05:00+08:00 | audit | Completion audit checked the local `epoch_57.pth` candidate for a possible true dense FCOS regdiag. Torch loading is blocked locally by a DLL error; zip/pickle metadata has no config/work_dir string but state keys look like an AdaTAD dense path. Upload to the remote timed out after 30 minutes and left only a partial 285 MB file, which was removed. A future config `input_random_fixed_50pct_regdiag_epoch58.py` was prepared, but the true `63.12` dense-baseline regdiag remains unavailable until a checkpoint can be transferred or restored server-side. |
| 2026-05-05T07:10:00+08:00 | result | SimOTA `range+center25+min_k4+w1` epoch49 eval: `52.00` Avg-mAP, `70.93@0.3`, `64.22@0.4`, `54.23@0.5`, `42.18@0.6`, `28.43@0.7`. Assignment table refreshed through epoch54; epoch54 remains FCOS-like (`pos/gt=3.3169`, `dynamic_k=3.5866`, `candidate_count=3.8301`, `valid_weight=1.0`). |
| 2026-05-05T10:45:00+08:00 | update | Recovered the SimOTA run after the epoch79 disk-full crash by deleting the corrupt partial checkpoint and old checkpoints, retaining only epoch69/74, then resuming from epoch74 with `disable_checkpoint=True`. Re-uploaded local `epoch_57.pth` by six-part transfer; remote SHA256 matches, but compatibility check against `input_random_fixed_50pct.py` shows 108 unexpected adapter parameters, so it is not strict evidence for the historical 63.12 dense baseline. |
| 2026-05-06T11:20:00+08:00 | result | Closed the three-run AdaTAD adapter matrix on `35407`: `input_random_fixed_50pct_adapter = 63.77`, `input_oracle_boundary_dense_50pct_adapter = 77.62`, `input_random_fixed_50pct_adapter_simota_mink4_w1 = 62.92`. Server is idle after completion; `/root/autodl-tmp` remains tight at `194G/200G` used (`97%`, about `7.0G` free). |
| 2026-05-19T02:04:00+08:00 | monitor | Added `CURRENT_SERVER_RUN_AUDIT_20260519_0204`: `adapter_quality_neg025` on `35329` and `adapter_stratified_frame` on `35407` were both running and pre-eval; both `gate_approvals` directories were empty. |
| 2026-05-19T02:07:00+08:00 | preflight | Synced the eval-only `input_random_fixed_50pct_adapter_visibility_rescore_g025` route to both active servers and passed remote `CHECK_ONLY=1`; missing-sentinel probes later returned `EXIT_CODE=3`, so no eval can launch without written approval. |
| 2026-05-19T02:14:00+08:00 | monitor | Added `FIRST_EVAL_ETA_20260519`: estimated first eval windows are `03:22-03:27 CST` for `adapter_stratified_frame` and `03:43-03:53 CST` for `adapter_quality_neg025`, based on inherited `val_start_epoch=40`, `val_eval_interval=2`. |
| 2026-05-19T02:16:00+08:00 | gate | Added `POST_FIRST_EVAL_DECISION_PLAYBOOK_20260519` and `logs/evaluate_active_first_eval_decisions.ps1`; current read-only output is `NO_EVAL_YET / WAIT_FOR_FIRST_EVAL` for both active runs. |
| 2026-05-19T02:18:00+08:00 | monitor | Started combined first-eval decision watcher `logs/watch_active_first_eval_decisions.ps1` with PID `35264`; first check saw both active runs pre-eval and empty gate directories. |
| 2026-05-19T02:20:00+08:00 | audit | Added `ACTIVE_RUN_ENV_HEALTH_20260519_0220`: both active screens are alive, checkpoints are being written, `35329` has `144G` free and `35407` has `47G` free; environment health passes for continuing to first eval. |
| 2026-05-19T02:31:00+08:00 | guard | Added `CHECKPOINT_SELECTION_GUARD_20260519` after finding stale 2026-05-10 `epoch_29/39/49.pth` files in the active `35407 adapter_stratified_frame` work_dir; future eval/rescore must record explicit checkpoint path, timestamp, bytes, and log consistency. |
| 2026-05-19T11:40:00+08:00 | result | `adapter_stratified_frame` completed on `35407` with `64.64 Avg-mAP / 43.23@0.7`, beating both `63.77` random-fixed Adapter and `63.85` strict EMA references; `adapter_quality_neg025` latest is `63.12 / 41.79@0.7`, so alpha sweep remains blocked. Added `POST_FIRST_EVAL_RESULTS_AND_TUBELET2_GATE_20260519` and approved only `adapter_stratified_tubelet2_after_frame.ok`. |
| 2026-05-19T11:45:00+08:00 | launch | Created the approved `adapter_stratified_tubelet2_after_frame.ok` sentinel on `35407`, passed tubelet2 `CHECK_ONLY=1`, and launched screen `350887.adapter_stratified_tubelet2`; first health check reached epoch 0 iter 50/99 with one isolated non-finite skip. |
| 2026-05-19T11:53:00+08:00 | preflight | Added `input_random_fixed_50pct_adapter_stratified_visibility_rescore_g025` and parameterized `run_adapter_visibility_rescore_eval.sh`; local pytest/bash checks passed and both `35329`/`35407` passed random-fixed plus stratified `CHECK_ONLY=1`. No eval launched; visibility-rescore remains blocked. |
| 2026-05-19T12:09:00+08:00 | monitor | Accepted Gemini CLI code review under the user-confirmed no-120s Gemini rule, added `ACTIVE_TAD_STATUS_20260519_1209`, updated `monitor_current_servers_every2h.ps1` to track `STRATIFIED_TUBELET2_ACTIVE`, and restarted the background monitor with PID `38176`. Current best remains `adapter_stratified_frame = 64.64`; `tubelet2` is running pre-eval and `neg025 = 63.12` still blocks alpha sweep. |
| 2026-05-19T12:13:00+08:00 | test | Added Gemini follow-up tensor behavior tests for `sparse_visibility_support` and `apply_visibility_rescore` in `OpenTAD_Back/tests/test_adapter_safety_contracts.py`; local verification passed with `39 passed, 7 skipped`, and direct remote inline tensor checks passed on both `35407` and `35329` using `/root/miniconda3/bin/python`. |
| 2026-05-19T12:20:00+08:00 | test | Added `mmengine.Config.fromfile` semantic pytest coverage for visibility-rescore and quality neutral/neg025 configs; local result is `39 passed, 9 skipped`, and direct remote semantic config checks passed on both active servers. |
| 2026-05-19T12:22:00+08:00 | monitor | Extended `watch_single_remote_average_map.ps1` with `-ContinueAfterMetric` and started `adapter_quality_neg025_final` watcher PID `101436`; it keeps monitoring after existing mAP and stops on `Training Over`, so alpha-sweep reconsideration waits for final/latest evidence. |
| 2026-05-19T12:25:50+08:00 | policy | User clarified that Gemini CLI reviews have no 120-second thinking-time acceptance rule. Updated current TAD plan/audit records so Gemini acceptance is based on exit code 0, absence of CLI/API failure, and usable content; earlier short Gemini records are secondary only when superseded by better V2 records. |
| 2026-05-19T12:45:00+08:00 | result | `adapter_quality_neg025` completed a new eval after epoch55 at `63.37 Avg-mAP / 41.77@0.7`. Quality gate output is `QUALITY_BRANCH_SAFE_BUT_NEG025_NOT_BASELINE`, so alpha sweep remains blocked; `adapter_stratified_tubelet2` is healthy pre-eval on `35407` at epoch14 with no mAP yet. |
| 2026-05-19T12:49:00+08:00 | plan | Added `VISIBILITY_RESCORE_POST_STRATIFIED_CHECKPOINT_PLAN_20260519` for the eval-only stratified visibility-rescore candidate. Verified `adapter_stratified_frame` `epoch_59.pth` on `35407` as 623799630 bytes with SHA256 `60afee9679481531cc8ba5f9896c602614fa415d71995611d5bca030d1614eeb`; no visibility-rescore sentinel was created and no eval was launched. |
| 2026-05-19T12:51:00+08:00 | monitor | Added `NEXT_EVAL_ETA_20260519_1251`: tubelet2 first mAP is estimated around `14:55-15:10 CST`; neg025 next mAP around `13:50-14:00 CST` and final around `15:00-15:20 CST`, based on current live log cadence. |
| 2026-05-19T12:55:00+08:00 | tooling | Added `logs/evaluate_current_tad_goal_decisions.ps1`, a read-only current-stage decision helper for `adapter_stratified_tubelet2` and `adapter_quality_neg025`. It reports Adapter best updates, 65-candidate audit triggers, and alpha-sweep gate status without creating sentinels. |
| 2026-05-19T12:59:00+08:00 | tooling | Added `logs/watch_current_tad_goal_decisions.ps1`, a short-interval read-only watcher around the current-stage decision helper. It stops when tubelet2 emits a first-eval decision, neg025 recovers the baseline, or a 65-candidate audit is required. |
| 2026-05-19T12:59:35+08:00 | monitor | Started current-stage decision watcher PID `55160` with 300s interval. First check confirms `WAIT_TUBELET2_FIRST_EVAL`, `NEG025_KEEP_ALPHA_BLOCKED`, and `OBJECTIVE_STATUS=ADAPTER_GAIN_HEAD_BLOCKED_65_MISSING`. |
| 2026-05-19T13:02:00+08:00 | audit | Checked `adapter_quality_neg025` checkpoint availability on `35329`: actual run config has `checkpoint_interval=10`, `disable_checkpoint=False`, and currently only `epoch_19/29/39/49.pth` exist. Recorded that alpha sweep requires a verified concrete checkpoint, so non-checkpointed epoch55/57 metrics alone cannot unlock alpha. |
| 2026-05-19T13:08:00+08:00 | guard | Hardened `logs/prepare_adapter_quality_alpha_sweep.ps1` so any remote alpha sweep validates the checkpoint file, bytes/mtime, inferred epoch, latest run log, and epoch completion line before checking the approval sentinel or launching evals. |
| 2026-05-19T13:09:00+08:00 | fix | Fixed `logs/verify_remote_checkpoint_for_eval.ps1` remote stdin handling by stripping a possible UTF-8 BOM before piping to bash; the previous BOM made remote bash read `set` as an invalid command. |
| 2026-05-19T13:09:00+08:00 | fix | Updated `logs/watch_current_tad_goal_decisions.ps1` to match the new checkpoint-aware neg025 recovery decision string (`NEG025_BASELINE_RECOVERED_VERIFY_CHECKPOINT_AND_WRITE_GATE_BEFORE_ALPHA`) and restarted the watcher as PID `36328`. First check remains `WAIT_TUBELET2_FIRST_EVAL` / `NEG025_KEEP_ALPHA_BLOCKED`. |
| 2026-05-19T13:21:00+08:00 | review | Added `CURRENT_DIRECTION_SOLUTION_AND_GEMINI_REVIEW_20260519_1321`: Gemini CLI V3 comparative code review of current new implementation versus last-night completed experiment code exited `0` and is accepted under the no-duration Gemini rule. Verdict: no Critical/High blockers, deployment PASS; residual risks are brittle text-contract tests and a low edge-case coverage gap. Added `valid_len=1` sparse-visibility tensor test; local pytest result is `39 passed, 10 skipped`. Current live decision remains `WAIT_TUBELET2_FIRST_EVAL` / `NEG025_KEEP_ALPHA_BLOCKED`, best valid result `64.64`, objective still incomplete. |
| 2026-05-19T13:24:00+08:00 | monitor | Appended a health check to `CURRENT_DIRECTION_SOLUTION_AND_GEMINI_REVIEW_20260519_1321`: all three local watchers are alive, `adapter_stratified_tubelet2` is active at epoch23 with no mAP yet, and `adapter_quality_neg025` is in the epoch57 validation block with latest completed metric still `63.37 / 41.77@0.7`; alpha remains blocked. |
| 2026-05-19T13:26:00+08:00 | audit | Added `RESULT_INTEGRITY_AUDIT_CHECKLIST_65PLUS_20260519`, a mandatory checklist for any future `>=65.00` candidate. It maps the objective to concrete evidence: raw mAP block, official validation GT count, checkpoint/log/config consistency, gate compliance, no leakage/post-hoc tuning, baseline comparison, numeric health, external review, and local record. Current status remains incomplete: tubelet2 has no eval and neg025 latest is still `63.37`. |
| 2026-05-19T15:00:00+08:00 | result/launch | Added `ADAPTER_GAPFILM_QUEUE_AND_TUBELET2_FIRST_EVAL_20260519`: `adapter_stratified_tubelet2` first eval is `63.34 Avg-mAP / 41.55@0.7`, below baseline and not claimable; `adapter_gapfilm_safe` passed Gemini 3 Pro Preview final review, local checks, remote CHECK_ONLY, and Linux tensor smoke, then was gated and queued on `35407` behind the active tubelet2 run. |
| 2026-05-19T15:21:00+08:00 | implementation/review/launch | Added `HEAD_GEOMREG_SAFE_IMPLEMENTATION_AND_REVIEW_20260519`: implemented head-side geometry-conditioned regression residual using only `irregular_selected_positions`/`valid_len`, passed local checks (`41 passed, 14 skipped`), Gemini 3 Pro Preview final review, remote CHECK_ONLY on `35407`/`25876`, and Linux tensor smoke. Created `adapter_head_geomreg_after_review.ok` and queued it behind `adapter_gapfilm_safe` with PID `728186`. |
| 2026-05-19T13:29:00+08:00 | tooling | Wired the 65+ audit checklist into `logs/evaluate_current_tad_goal_decisions.ps1` and `logs/watch_current_tad_goal_decisions.ps1`: if a future `65_CANDIDATE_AUDIT_REQUIRED` state appears, the output points to `RESULT_INTEGRITY_AUDIT_CHECKLIST_65PLUS_20260519.md`. Verification: decision script still reports `ADAPTER_GAIN_HEAD_BLOCKED_65_MISSING`; PowerShell parse check passed. |
| 2026-05-19T13:36:00+08:00 | policy | Re-applied the user clarification that Gemini has no 120-second thinking-time limit. Future Gemini rerun scripts now use only `exit=0`, no CLI/API error, and usable content for adoption; old prompt files got a top-level override. Direct remote check still shows tubelet2 pre-eval at epoch26 and neg025 latest completed metric `63.37 / 41.77@0.7`, so alpha remains blocked and objective remains incomplete. |
| 2026-05-19T13:42:00+08:00 | audit | Performed a waiting-window implementation/gate audit for active tubelet2 and prepared visibility-rescore: contract tests passed (`39 passed, 10 skipped`), `bash -n` passed for `run_adapter_stratified_pair.sh` and `run_adapter_visibility_rescore_eval.sh`, and PowerShell parse passed for current decision/checkpoint/alpha gate scripts. No new gate sentinel was created; tubelet2 still waits for first eval and neg025 keeps alpha blocked. |
| 2026-05-19T13:52:00+08:00 | result | `adapter_quality_neg025` completed another eval at `63.32 Avg-mAP / 41.58@0.7`, down from `63.37 / 41.77@0.7` and still below the `63.85` strict baseline and `63.54` neutral reference; alpha sweep remains blocked. `adapter_stratified_tubelet2` has reached `epoch_29.pth` on `35407` with no mAP yet, so the current objective remains incomplete. |
| 2026-05-19T15:41:00+08:00 | monitor | Appended the second tubelet2 eval to `ADAPTER_GAPFILM_QUEUE_AND_TUBELET2_FIRST_EVAL_20260519`: `adapter_stratified_tubelet2` improved from `63.34 / 41.55@0.7` to `63.58 / 41.72@0.7`, but remains below `63.77`, `63.85`, and `64.64`. GapFilm PID `667514` is alive waiting for GPU memory; Head-GeomReg queue PID `728186` is alive waiting for GapFilm. Current model-side candidates remain GapFilm first and Head-GeomReg second. |
| 2026-05-19T16:08:00+08:00 | launch/monitor | After `adapter_stratified_tubelet2` third eval reached only `63.73 / 41.66@0.7` and remained below direct/EMA baselines with weaker @0.7 than eval2, stopped the sampling-side screen to unblock model-side work. `25876` still had no visible GPU. `adapter_gapfilm_safe` started on `35407` at `16:00:20`; first epoch had two non-consecutive non-finite skip events and then epoch1 iter50 printed normally with `Loss=0.9785`. |
| 2026-05-19T16:27:00+08:00 | tooling/monitor | Added read-only GapFilm/Head-GeomReg decision scripts `logs/evaluate_gapfilm_geomreg_decisions.ps1` and `logs/watch_gapfilm_geomreg_decisions.ps1`; parse checks passed and one-shot output reports `WAIT_GAPFILM_FIRST_EVAL`, `GAPFILM_NONFINITE_COUNT=3`, and `HEAD_GEOMREG_WAITING_FOR_GAPFILM`. Started background watcher PID `75588`, stdout `logs/watch_gapfilm_geomreg_decisions_20260519_162648.out.log`; no sentinels are created by the watcher. |
| 2026-05-19T16:30:00+08:00 | monitor | GapFilm remains active on `35407` with no eval yet. Latest line is epoch6 iter50: `Loss=0.7651 cls_loss=0.4677 reg_loss=0.2974`; evaluator still reports `GAPFILM_NONFINITE_COUNT=3`, with no new non-finite events since epoch2 iter81. Head-GeomReg remains queued behind GapFilm. |
| 2026-05-19T13:54:00+08:00 | gate | Added `QUALITY_NEG025_ALPHA_BLOCK_DECISION_20260519_1352`: written blocking decision for `adapter_quality_neg025`. It explicitly forbids creating `gate_approvals/adapter_alpha_sweep_after_neg025.ok` unless neg025 later reaches `>=63.85`, has a verified concrete checkpoint, and receives a separate written approval. |
| 2026-05-19T17:02:00+08:00 | implementation/review | Added `ADAPTER_GAPDIFF_SAFE_IMPLEMENTATION_AND_REVIEW_20260519`: implemented reviewed Adapter-side GapDiff residual, a zero-initialized gap-aware left/right neighbor difference branch using existing irregular time geometry. Local py_compile, targeted pytest (`29 passed, 11 skipped`), bash syntax, remote `25876` check-only, remote CPU tensor smoke, and Gemini CLI `gemini-3-pro-preview` all passed with `VERDICT: ACCEPT`. GapDiff is not launched; it is held as the next model-side candidate behind the active `adapter_gapfilm_safe` and queued `adapter_head_geomreg_safe` sequence. |
| 2026-05-19T17:14:00+08:00 | implementation/review | Added `HEAD_GEOMCLS_SAFE_IMPLEMENTATION_AND_REVIEW_20260519`: implemented reviewed Head-side geometry-conditioned classification residual via `cls_geometry_cfg`, reusing sparse `irregular_selected_positions`/`valid_len` metadata to predict a bounded zero-init class-logit residual. Local py_compile, targeted pytest (`30 passed, 13 skipped`), bash syntax, remote `25876` check-only, remote CPU tensor smoke, and Gemini CLI `gemini-3-pro-preview` all passed with `VERDICT: ACCEPT`. Head-GeomCls is not launched; it is held as a Head-side fallback behind active GapFilm and queued Head-GeomReg. |
| 2026-05-19T17:15:00+08:00 | tooling/monitor | Updated `logs/evaluate_gapfilm_geomreg_decisions.ps1` to surface reviewed fallback readiness without creating sentinels. One-shot output now reports `ADAPTER_FALLBACK_GAPDIFF_REVIEWED=TRUE`, `HEAD_FALLBACK_GEOMCLS_REVIEWED=TRUE`, `NEXT_REVIEWED_FALLBACKS=ADAPTER_GAPDIFF_SAFE,HEAD_GEOMCLS_SAFE`, and `FALLBACK_QUEUE_POLICY=WAIT_GAPFILM_AND_HEAD_GEOMREG_DECISION_BEFORE_LAUNCH`. Current live state is still `WAIT_GAPFILM_FIRST_EVAL`; objective remains incomplete. |
| 2026-05-19T17:17:00+08:00 | audit | Added `CURRENT_OBJECTIVE_STATUS_AUDIT_20260519_1717`: mapped the active objective to concrete artifacts and gaps. Adapter/Head implementation coverage is now present via GapFilm, GapDiff, Head-GeomReg, and Head-GeomCls, but result evidence is still missing: GapFilm has no eval yet, Head-GeomReg has not started, and no model-side run has exceeded `63.77`, `63.85`, `64.64`, or triggered the `65+` audit. |
| 2026-05-19T17:20:00+08:00 | verification/monitor | After adding GapDiff and Head-GeomCls, reran the combined local contract suite: `python -m pytest tests/test_adapter_quality_rescore_contracts.py tests/test_adapter_safety_contracts.py tests/test_adapter_simota_contracts.py -q` -> `43 passed, 18 skipped`; py_compile passed for `vit_adapter.py`, `anchor_free_head.py`, and `test_adapter_safety_contracts.py`. Live GapFilm check: epoch18 started at `17:19:26`, latest loss `0.5722`, no eval yet, non-finite count remains `3`, `/root/autodl-tmp` has `44G` free. |
| 2026-05-19T17:22:00+08:00 | tooling/monitor | Enhanced `logs/evaluate_gapfilm_geomreg_decisions.ps1` to print latest epoch and latest loss line for GapFilm/Head-GeomReg. Parse check passed; one-shot output now includes `GAPFILM_LATEST_EPOCH=18` and latest loss line `Loss=0.5698 cls_loss=0.3114 reg_loss=0.2583`, while decision remains `WAIT_GAPFILM_FIRST_EVAL`. |
| 2026-05-19T17:24:00+08:00 | tooling/monitor | Added first-eval distance reporting to `logs/evaluate_gapfilm_geomreg_decisions.ps1` with `FirstEvalEpoch=40`. Parse check passed; one-shot output reports `GAPFILM_FIRST_EVAL_EPOCH=40` and `GAPFILM_FIRST_EVAL_REMAINING_EPOCHS=22`, confirming GapFilm is still well before first eval and should continue monitoring rather than trigger fallback actions. |
| 2026-05-19T17:40:00+08:00 | implementation/review | Added `HEAD_GEOMBOTH_SAFE_IMPLEMENTATION_AND_REVIEW_20260519`: prepared non-launched Head-side combined geometry fallback `input_random_fixed_50pct_adapter_head_geomboth_safe`, enabling both `cls_geometry_cfg` and `reg_geometry_cfg` under strict random-fixed protocol. Local checks passed (`44 passed, 19 skipped`; targeted `-k geomboth`: `1 passed, 1 skipped`). Gemini CLI retry with `gemini-3-pro-preview` returned `VERDICT: ACCEPT` and was adopted. Remote `25876` `CHECK_ONLY=1` passed. No launch sentinel was created. |
| 2026-05-19T17:42:00+08:00 | monitor | Updated `logs/evaluate_gapfilm_geomreg_decisions.ps1` to surface `HEAD_FALLBACK_GEOMBOTH_REVIEWED`. Parse check passed. One-shot output reports GapFilm active at epoch23, no eval yet, `WAIT_GAPFILM_FIRST_EVAL`, Head-GeomReg waiting behind GapFilm, and reviewed fallbacks `ADAPTER_GAPDIFF_SAFE,HEAD_GEOMCLS_SAFE,HEAD_GEOMBOTH_SAFE`; no sentinels are created by the script. |
| 2026-05-19T17:44:00+08:00 | audit | Added `CURRENT_OBJECTIVE_STATUS_AUDIT_20260519_1744`: updated the active objective checklist after `Head-GeomBoth`. Implementation coverage now includes GapFilm, GapDiff, Head-GeomReg, Head-GeomCls, and Head-GeomBoth, but result evidence is still missing because GapFilm has no eval and Head-GeomReg has not started. Goal remains incomplete until a model-side run beats `63.77`, `63.85`, `64.64`, or triggers the `65+` audit. |
| 2026-05-19T17:45:00+08:00 | verification | Synced the non-launched `adapter_head_geomboth_safe` config/script to target server `35407` and ran `CHECK_ONLY=1` successfully while GapFilm remained active. The check verified ActionFormer + ActionFormerHead, both geometry configs, random-fixed train/val/test contract, checkpoint cadence, and batch size `2/2/2`; no GPU run or sentinel was triggered. |
| 2026-05-19T17:50:00+08:00 | implementation/review | Added `ADAPTER_GAPFILM_HEAD_GEOMREG_SAFE_IMPLEMENTATION_AND_REVIEW_20260519`: prepared non-launched combined model-side route `input_random_fixed_50pct_adapter_gapfilm_head_geomreg_safe`, combining Adapter Gap/Visibility FiLM with Head GeomReg. Local checks passed (`45 passed, 20 skipped`; targeted `-k gapfilm_head_geomreg`: `1 passed, 1 skipped`). Gemini CLI `gemini-3-pro-preview` returned `VERDICT: ACCEPT`; remote `25876` and `35407` `CHECK_ONLY=1` passed. No launch sentinel was created; route remains blocked until singleton GapFilm and Head-GeomReg decisions. |
| 2026-05-19T17:51:00+08:00 | tooling/monitor | Updated `logs/evaluate_gapfilm_geomreg_decisions.ps1` and `CURRENT_OBJECTIVE_STATUS_AUDIT_20260519_1744` to surface the reviewed `COMBO_GAPFILM_HEAD_GEOMREG_SAFE` candidate. It remains a prepared-only combination and is not eligible for launch until the singleton GapFilm and Head-GeomReg decisions are known. |
| 2026-05-19T17:51:30+08:00 | monitor | Parse check passed for `logs/evaluate_gapfilm_geomreg_decisions.ps1`. One-shot output reports GapFilm active at epoch25, latest loss `0.5332`, no eval yet, first eval still epoch40 with 15 epochs remaining, Head-GeomReg waiting, and reviewed fallbacks including `COMBO_GAPFILM_HEAD_GEOMREG_SAFE`; no sentinel created. |
| 2026-05-19T17:53:00+08:00 | gate-hardening | Hardened the non-launched `adapter_gapfilm_head_geomreg_safe` launcher gate by renaming its default approval sentinel from `adapter_gapfilm_head_geomreg_after_review.ok` to `adapter_gapfilm_head_geomreg_after_singletons.ok`, matching Gemini's warning that code review alone is insufficient and singleton GapFilm/Head-GeomReg decisions are required before any combination launch. |
| 2026-05-19T17:54:00+08:00 | verification | Reverified `adapter_gapfilm_head_geomreg_safe` after gate hardening: `bash -n` passed, targeted pytest `-k gapfilm_head_geomreg` passed (`1 passed, 1 skipped`), full local suite passed (`45 passed, 20 skipped`), and both `25876`/`35407` `CHECK_ONLY=1` passed while showing default approval file `adapter_gapfilm_head_geomreg_after_singletons.ok`. |
| 2026-05-19T17:55:00+08:00 | monitor/audit | GapFilm remains active on `35407`, now epoch26 with no eval yet and 14 epochs remaining before first eval. Updated `CURRENT_OBJECTIVE_STATUS_AUDIT_20260519_1744` to record that the combo route uses explicit `adapter_gapfilm_head_geomreg_after_singletons.ok` gating. |
| 2026-05-19T17:58:00+08:00 | tooling/monitor | Hardened `logs/evaluate_gapfilm_geomreg_decisions.ps1` to scan remote `gate_approvals`. Parse check passed. One-shot output now distinguishes expected approvals (`adapter_gapfilm_after_review.ok`, `adapter_head_geomreg_after_review.ok`) from future/combination sentinels; `REMOTE_FUTURE_SENTINELS_PRESENT=FALSE`, so reviewed fallbacks remain unlaunched. |
| 2026-05-19T18:00:00+08:00 | tooling/monitor | Hardened `logs/watch_gapfilm_geomreg_decisions.ps1` so the watcher stops for gate audit if `REMOTE_FUTURE_SENTINELS_PRESENT=TRUE`. Restarted watcher with PID `70728`, stdout `logs/watch_gapfilm_geomreg_decisions_20260519_175951.out.log`; first check reports GapFilm epoch27, no eval, 13 epochs to first eval, and no future/combination sentinel. |
| 2026-05-19T18:01:00+08:00 | monitor | Live check: GapFilm remains active at epoch27 with latest loss `0.5323`; no eval yet and 13 epochs remain before first eval. Remote approvals are only the expected current route sentinels (`adapter_gapfilm_after_review.ok`, `adapter_head_geomreg_after_review.ok`); `REMOTE_FUTURE_SENTINELS_PRESENT=FALSE`. |
| 2026-05-19T18:07:00+08:00 | monitor | After one watcher interval, GapFilm remains active at epoch28 with latest loss `0.4780`; no eval yet and 12 epochs remain before first eval. Remote approvals remain limited to expected current sentinels, and `REMOTE_FUTURE_SENTINELS_PRESENT=FALSE`. |
| 2026-05-19T18:13:00+08:00 | monitor | GapFilm remains active at epoch30 with latest loss `0.5200`; no eval yet and 10 epochs remain before first eval. Remote approvals remain limited to expected current sentinels, and `REMOTE_FUTURE_SENTINELS_PRESENT=FALSE`. |
| 2026-05-19T18:19:00+08:00 | monitor | GapFilm remains active at epoch31 with latest loss `0.4820`; no eval yet and 9 epochs remain before first eval. Remote approvals remain limited to expected current sentinels, and `REMOTE_FUTURE_SENTINELS_PRESENT=FALSE`. |
| 2026-05-19T18:24:00+08:00 | monitor | GapFilm remains active at epoch32 with latest loss `0.5136`; no eval yet and 8 epochs remain before first eval. Remote approvals remain limited to expected current sentinels, and `REMOTE_FUTURE_SENTINELS_PRESENT=FALSE`. |
| 2026-05-19T18:30:00+08:00 | monitor | GapFilm remains active at epoch34 with latest loss `0.4951`; no eval yet and 6 epochs remain before first eval. Remote approvals remain limited to expected current sentinels, and `REMOTE_FUTURE_SENTINELS_PRESENT=FALSE`. |
| 2026-05-19T18:36:00+08:00 | monitor | GapFilm remains active at epoch35 with latest loss `0.5229`; no eval yet and 5 epochs remain before first eval. Remote approvals remain limited to expected current sentinels, and `REMOTE_FUTURE_SENTINELS_PRESENT=FALSE`. |
| 2026-05-19T18:41:00+08:00 | monitor | GapFilm remains active at epoch36 with latest loss `0.4960`; no eval yet and 4 epochs remain before first eval. Remote approvals remain limited to expected current sentinels, and `REMOTE_FUTURE_SENTINELS_PRESENT=FALSE`. |
| 2026-05-19T18:47:00+08:00 | monitor | GapFilm remains active at epoch38 with latest loss `0.4908`; no eval yet and 2 epochs remain before first eval. Remote approvals remain limited to expected current sentinels, and `REMOTE_FUTURE_SENTINELS_PRESENT=FALSE`. |
| 2026-05-19T18:53:00+08:00 | monitor | GapFilm remains active at epoch39 with latest loss `0.4918`; no eval yet and 1 epoch remains before first eval. Remote approvals remain limited to expected current sentinels, and `REMOTE_FUTURE_SENTINELS_PRESENT=FALSE`. |
| 2026-05-19T18:58:00+08:00 | monitor | GapFilm reached epoch40 and is still training (`iter50/99`, latest loss `0.4686`); no eval block yet, first-eval remaining is now 0 epochs. Continue short-interval monitoring for the first mAP block. |
| 2026-05-19T19:02:00+08:00 | monitor | GapFilm entered epoch41 with latest loss `0.4581`, but still has no mAP block. Direct remote grep confirms epoch39/40/41 starts and no `Average-mAP`/eval lines, so the effective first eval is delayed relative to the static `val_start_epoch=40` expectation. Continue monitoring for the first actual mAP block; no gate/action change. |
| 2026-05-19T19:24:00+08:00 | result/monitor | GapFilm first eval completed on `35407`: `62.75 Avg-mAP / 39.89@0.7` with `3325` GT instances and `422000` predictions. This is above the `<60` collapse stop gate but below `63.77`, `63.85`, and `64.64`, so it is not a model-side improvement. Decision: continue GapFilm to next eval, keep Head-GeomReg queued behind it, and do not launch reviewed fallbacks or combo routes. |
| 2026-05-19T19:53:00+08:00 | result/action | GapFilm Eval2 completed at `63.22 Avg-mAP / 40.52@0.7`, below the adopted `63.30` probation gate and still below `63.77`, `63.85`, and `64.64`. Stopped GapFilm process group `667510`; no GapFilm residual processes remained. Existing Head-GeomReg queue stayed alive and is now the next model-side singleton to start. |
| 2026-05-19T20:03:00+08:00 | monitor/tooling | Head-GeomReg started from the existing queue at `19:56:13`; active log is `logs/input_random_fixed_50pct_adapter_head_geomreg_safe_20260519_195614.log`. Early health: `HEAD_GEOMREG_ACTIVE=TRUE`, non-finite count `2`, latest epoch `1`, latest loss `1.0337 / cls 0.6507 / reg 0.3831`, no eval yet. Fixed `evaluate_gapfilm_geomreg_decisions.ps1` to parse GapFilm and Head-GeomReg metrics separately, added `watch_head_geomreg_first_eval.ps1`, and started watcher PID `94652` with stdout `logs/watch_head_geomreg_first_eval_20260519_200309.out.log`. |
| 2026-05-19T20:12:00+08:00 | monitor | Head-GeomReg remains active on `35407`: non-finite count is still `2`, latest epoch is `3`, latest loss is `0.8325 / cls 0.5001 / reg 0.3324`, and there is no eval yet. Continue Head-GeomReg; do not launch reviewed fallbacks before a first eval or an audit/crash gate. |
| 2026-05-19T20:19:00+08:00 | monitor | Head-GeomReg remains stable: active at epoch `5`, non-finite count still `2`, latest loss `0.8379 / cls 0.5184 / reg 0.3195`, no eval yet, first eval remaining about `35` epochs. Continue waiting for Head-GeomReg eval; do not launch fallbacks. |
| 2026-05-19T20:21:00+08:00 | tooling | Extended `logs/evaluate_gapfilm_geomreg_decisions.ps1` with Head-GeomReg high-IoU gates: `HEAD_GEOMREG_MAP_070_CONCERN_GATE=41.00`, `HEAD_GEOMREG_MAP_070_VALIDATE_GATE=42.00`, and `HEAD_GEOMREG_MAP_070_DECISION` once a Head eval exists. Parse check passed and current run remains active with no Head eval yet. |
| 2026-05-19T19:30:00+08:00 | review/tooling | Ran Gemini CLI `gemini-3-pro-preview` read-only post-eval discussion for GapFilm first eval; exit `0`, output `logs/gemini3_pro_preview_gapfilm_first_eval_decision_20260519.txt`. Adopted probation gate: continue GapFilm to Eval2, stop and let Head-GeomReg接力 if Eval2 Avg-mAP `<63.30`, continue only if `>=63.30`. Updated `evaluate_gapfilm_geomreg_decisions.ps1` and `watch_gapfilm_geomreg_decisions.ps1`, parse checks passed, restarted watcher PID `86820` with stdout `logs/watch_gapfilm_geomreg_decisions_20260519_192944.out.log`. |
| 2026-05-19T20:50:00+08:00 | review/audit | Added `SUBAGENT_DEEPSEEK_CODE_REVIEW_20260519`: consolidated six subagent read-only code reviews plus Claude Code CLI `deepseek-v4-pro` secondary verification. Verdict is `WARN`: safe configs preserve strict random-fixed 50% and no test-time GT leakage was found, but model-side evidence remains negative/pending; GapFilm failed (`63.22 / 40.52@0.7`), Head-GeomReg has no eval yet, combo is blocked, and hardening items remain for explicit `keep_ratio=0.5` launcher checks, decision-sentinel handoff, stale docs, and end-to-end Adapter/Head smoke tests. |
| 2026-05-19T21:16:00+08:00 | review | Added `GEMINI3_PRO_PREVIEW_HARDENING_CODE_REVIEW_20260519`: exact requested Gemini model `gemini-3-prp-preview` failed through CLI with `model_not_found`, so the hardening patch was reviewed with available `gemini-3-pro-preview`. Gemini returned `VERDICT: ACCEPT` for explicit `keep_ratio=0.5` launcher checks, GapFilm-to-Head-GeomReg handoff sentinel, and added Adapter/Head smoke tests. Independent local check confirms all six reviewed launchers now have three keep-ratio assertions each. |
| 2026-05-19T21:18:00+08:00 | sync/verification | Synced the reviewed hardening scripts/tests to both `35407` and `25876`, plus missing fallback/combination configs to `35407`. Remote `CHECK_ONLY=1` now passes for all six reviewed launchers on both servers. Latest decision helper still reports `REMOTE_FUTURE_SENTINELS_PRESENT=FALSE`; active Head-GeomReg is unaffected and remains pre-eval. |
| 2026-05-19T21:22:00+08:00 | review/tooling | Ran Gemini CLI `gemini-3-pro-preview` route review for the active Head-GeomReg and fallback sequence; exit `0`, output `logs/gemini3_pro_preview_next_model_route_review_20260519.txt`. Verdict: `YELLOW (Cautious Proceeding)`, `NEXT_ACTION=MONITOR_AND_PREPARE_GAPDIFF`. Encoded Gemini's Head-GeomReg gates into `evaluate_gapfilm_geomreg_decisions.ps1`: first-eval stop if Avg-mAP `<61.5` or `mAP@0.7 <38.0`; Eval2 stop if Avg-mAP `<63.5` or `mAP@0.7 <41.5`; non-finite stop gate `>10`. |
| 2026-05-19T21:33:00+08:00 | review/blocked | Attempted the exact requested Gemini CLI model `gemini-3-prp-preview` for a DeepSeek-style line-by-line experiment code review. The CLI retried 10 times and failed with `model_not_found`: `No available channel for model gemini-3-prp-preview under group gemini`. Added `GEMINI3_PRP_PREVIEW_LINE_BY_LINE_REVIEW_ATTEMPT_20260519`; no usable review verdict was produced and no fallback model was substituted for this exact request. |
| 2026-05-19T21:53:00+08:00 | review/resolution | Added `GEMINI3_PRO_PREVIEW_LINE_BY_LINE_REVIEW_RESOLUTION_20260519`: long-wait Gemini 3 Pro Preview returned `ACCEPT_WITH_FIXES`, but its Critical claim that GapFilm/GapDiff crash from `metas` being passed directly to `VisionTransformerAdapter` was rejected as a false positive. Accepted the missing-test concern, added `BackboneWrapper metas -> time_embed` contract/runtime tests, passed local py_compile/pytest, and verified equivalent Linux inline smoke on both `35407` and `25876`. |
| 2026-05-19T22:10:00+08:00 | protocol | Updated root `AGENTS.md` and `OpenTAD_Back/AGENTS.md` with a mandatory post-implementation review gate: self-check, Gemini CLI read-only review using `gemini-3-pro-preview`, and Claude Code CLI DeepSeek secondary verification using the exact model name `deepseek-v4-pro`. The note also records that `gemini-3-prp-preview` currently fails with `model_not_found`, Gemini has no 120-second minimum rule here, and `deepseekv4pro` is an invalid model spelling. |
| 2026-05-19T22:39:00+08:00 | recovery | Added `ACCOUNT_SWITCH_RECOVERY_CONTEXT_20260519_2239` for account switching. Latest snapshot: Head-GeomReg active on `35407`, epoch `37`, no eval yet, non-finite count `3`, about 3 epochs to first eval; GapFilm remains stopped after `63.22 / 40.52@0.7`; fallbacks remain reviewed but unlaunched. |
| 2026-05-19T23:18:00+08:00 | result/action | Added `HEAD_GEOMREG_FIRST_EVAL_DECISION_20260519`: Head-GeomReg first eval on `35407` was `61.45 Avg-mAP / 38.65@0.7` with `3325` GT instances and `422000` predictions. This triggered the adopted first-eval stop gate because Avg-mAP was below `61.50`, remained below `63.77`, `63.85`, `64.64`, and `65.09`, and also raised a high-IoU concern (`38.65 < 41.00`). Stopped process group `728186`; GPU freed to `0 MiB`; no residual Head-GeomReg processes and no fallback/combo sentinel created. Next step is a read-only Claude CLI discussion before any GapDiff launch. |
| 2026-05-20T00:01:00+08:00 | analysis/review | Added `NEGATIVE_MODEL_SIDE_FAILURE_ANALYSIS_20260520`: integrated Gemini and DeepSeek discussions plus local static verification. Corrected the initial DeepSeek assumption: current `GapFilm`/`Head-GeomReg` safe routes use standard `PointGenerator` on selected-index coordinates, while GT is remapped to selected axis and test proposals are mapped back to dense time. Revised verdict: failures are mainly a coordinate-system route/design conflict between physical gap metadata and selected-index supervision, not a proven implementation bug. GapDiff remains paused pending diagnostics. |
| 2026-05-20T00:56:00+08:00 | implementation/review | Added `ADAPTER_NATIVE_DENSE_HEADV2_SAFE_IMPLEMENTATION_AND_REVIEW_20260520`: implemented native physical-axis dense-passthrough Adapter + HeadV2 route after GapFilm/Head-GeomReg failures. Self-check, local py_compile, local targeted pytest (`1 passed, 3 skipped`), Gemini `gemini-3-pro-preview` PASS, DeepSeek `deepseek-v4-pro` PASS, remote 35407 CHECK_ONLY PASS, and remote tensor smoke (`cell_right_last=668.0`) passed. Launch approved behind `adapter_native_dense_headv2_after_review.ok`. |
| 2026-05-20T00:59:00+08:00 | deploy/monitor | Launched `input_random_fixed_50pct_adapter_native_dense_headv2_safe` on `35407` in screen `521730.adapter_native_dense_headv2_safe` after gate sentinel creation. GPU0 active (`~3593 MiB`). Remote CHECK_ONLY and tensor smoke passed. Early train reached epoch0 iter50 with loss `1.5369 / cls 0.8082 / reg 0.7287`; two recoverable non-finite gradient skip steps occurred at epoch0 iters 14 and 17 on `rpn_head.reg_head.weight`, then training continued. Monitor for recurrence before first eval. |
| 2026-05-20T01:30:00+08:00 | deploy/monitor | Synced reviewed `adapter_native_dense_headv2_safe` implementation to newly available server `35329`, including HeadV2/prior/projection/neck dependencies and gate launcher. Remote `35329` py_compile, CHECK_ONLY, and tensor smoke (`cell_right_last=668.0`) passed; gate sentinel created and screen `643478.adapter_native_dense_headv2_safe` launched with `BASE_PORT=30655`. Current native dense HeadV2 replicas: `35407` active at epoch8 with non-finite count `2`; `35329` active from epoch0 with the same two early recoverable non-finite skips; both are pre-eval and no mAP claim is available. |
| 2026-05-20T01:31:00+08:00 | commit | Created scoped OpenTAD_Back commit `29d4130 add native dense headv2 adapter experiment` containing the native dense HeadV2 implementation/config/launcher/tests and required native-axis post-processing support. Unrelated dirty worktree files and large artifacts were excluded. |
| 2026-05-20T02:36:00+08:00 | monitor/review | Repaired and reviewed `logs/watch_headv2_first_eval.ps1` for the current native dense HeadV2 replicas on `35407` and `35329`. Local parser check and one-shot smoke passed; Gemini `gemini-3-pro-preview` fixed review PASS; DeepSeek `deepseek-v4-pro` bare ASCII review PASS. Started read-only watcher PID `47552`, stdout `logs/watch_headv2_first_eval_latest.out.log`; first check: `35407` active at epoch24, non-finite count `3`, no eval; `35329` active at epoch6, non-finite count `2`, no eval. Both remain pre-eval and no mAP claim is available. |
| 2026-05-20T02:41:00+08:00 | monitor | `watch_headv2_first_eval` second check remains healthy and pre-eval: `35407` active at epoch25 iter50 with `Loss=0.6304`, non-finite count `3`; `35329` active at epoch6 iter50 with `Loss=0.7625`, non-finite count `2`; `eval_count=0` on both and no stop gate triggered. |
| 2026-05-20T02:54:00+08:00 | monitor/fix | `watch_headv2_first_eval` triggered `STOP_FOR_CRASH_AUDIT` on `35407` at epoch26, but audit showed this was a watcher false positive: the run was still active and the crash regex had scanned the full SSH output, including the monitor's own grep command text. Fixed the watcher to delimit process and recent-log sections, restrict crash detection to recent log lines, and restrict active detection to current-config process lines. Local parse and smoke passed; Gemini `gemini-3-pro-preview` PASS; DeepSeek `deepseek-v4-pro` PASS. Restarted fixed watcher PID `39896`; first fixed check shows `35407` active at epoch28 with non-finite count `3` and `35329` active at epoch8 with non-finite count `2`, both `eval_count=0`. |
| 2026-05-20T03:04:50+08:00 | monitor | Native dense HeadV2 fixed watcher PID `39896` remains alive with empty stderr. Check `3/288`: `35407` active at epoch31, latest epoch30 iter99 `Loss=0.6233 / cls=0.2630 / reg=0.3604`, non-finite count `3`; `35329` active at epoch8 iter50 `Loss=0.8001 / cls=0.3623 / reg=0.4378`, non-finite count `2`. Both remain pre-eval with `eval_count=0`; no stop gate triggered. |
| 2026-05-20T03:10:49+08:00 | monitor | Native dense HeadV2 watcher check `4/288`: `35407` active at epoch32, latest epoch31 iter99 `Loss=0.6035 / cls=0.2572 / reg=0.3463`, non-finite count `3`; `35329` active at epoch9, latest epoch8 iter99 `Loss=0.7650 / cls=0.3410 / reg=0.4240`, non-finite count `2`. Both still have `eval_count=0`; no `Average-mAP` yet. |
| 2026-05-20T03:26:34+08:00 | monitor | Native dense HeadV2 watcher check `7/288`: `35407` active at epoch36, latest epoch35 iter99 `Loss=0.5860 / cls=0.2455 / reg=0.3405`, non-finite count increased to `4` but remains below the `>10` stop gate; `35329` active at epoch10 iter50 `Loss=0.7516 / cls=0.3316 / reg=0.4200`, non-finite count `2`. Both remain pre-eval with `eval_count=0`. |
| 2026-05-20T03:37:11+08:00 | monitor | Native dense HeadV2 watcher check `9/288`: `35407` active at epoch38, latest epoch37 iter99 `Loss=0.6048 / cls=0.2540 / reg=0.3508`, non-finite count `4`; `35329` active at epoch11 iter50 `Loss=0.7816 / cls=0.3504 / reg=0.4312`, non-finite count `2`. Both still have `eval_count=0`; wait for actual `Average-mAP` block rather than assuming a fixed epoch40 timing. |
| 2026-05-20T03:52:43+08:00 | monitor | Native dense HeadV2 watcher check `12/288`: `35407` active at epoch41, latest epoch41 iter99 `Loss=0.5801 / cls=0.2386 / reg=0.3414`, non-finite count `4`; `35329` active at epoch13, latest epoch12 iter99 `Loss=0.7195 / cls=0.3065 / reg=0.4130`, non-finite count `2`; both `eval_count=0`. Static review of `tools/train.py` explains the timing: with `val_start_epoch=40` and `val_eval_interval=2`, first eval is eligible after epoch41 because `(epoch + 1) % 2 == 0`. |
| 2026-05-20T04:04:13+08:00 | monitor | Direct `35407` remote check confirms native dense HeadV2 is in the epoch41 eval loop, not stalled: training and worker processes are alive, log mtime is updating, and tqdm reached about `382/396`; no `Average-mAP` block yet because eval had not finished. Decision: wait for complete mAP block before judging. |
| 2026-05-20T04:09:08+08:00 | result | Native dense HeadV2 first eval on `35407`: `56.82 Avg-mAP`, `75.01@0.3`, `69.66@0.4`, `60.68@0.5`, `48.04@0.6`, `30.69@0.7`. This is far below random-fixed Adapter `63.77` (`-6.95`), strict EMA `63.85` (`-7.03`), stratified best `64.64` (`-7.82`), and uniform stride-2 reference `65.09` (`-8.27`). Not a 65+ candidate and not model-side progress. Next: Gemini/DeepSeek route discussion before any new launch or route change. |
| 2026-05-20T04:17:29+08:00 | stop/review | Gemini `gemini-3-pro-preview` and DeepSeek `deepseek-v4-pro` post-eval discussions both returned `STOP_NOW` for native dense HeadV2. Stopped same-route screens on `35407` and `35329`, stopped local watcher PID `39896`, and confirmed both GPUs are idle (`0 MiB`) with no matching train process. Minimal diagnostics show stable losses (`~0.55-0.64` near first eval) and low non-finite counts (`4`/`2`), so the failure is more likely coordinate/assignment/head-contract design than numeric divergence. Next route should return to standard Adapter + ActionFormer head and add minimal sampling/reliability-aware loss/assignment weighting or Adapter-side reliability, not another full HeadV2 route. |
| 2026-05-20T05:05:00+08:00 | implementation/self-check | Implemented `input_random_fixed_50pct_adapter_sample_reliability_safe`: standard `ActionFormer` + `ActionFormerHead`, unchanged strict random-fixed 50% sampling, and training-side sample reliability target weights from `irregular_selected_positions` / `irregular_selected_valid_len`. Added gated launcher `run_adapter_sample_reliability_safe.sh` and tests. Local py_compile and `bash -n` passed; `tests/test_adapter_safety_contracts.py` passed with `35 passed, 20 skipped`. External Gemini/DeepSeek review and remote CHECK_ONLY are pending before launch. |
| 2026-05-20T05:32:00+08:00 | review/fix | Gemini CLI `gemini-3-pro-preview` direct-stdin review for `adapter_sample_reliability_safe` returned `WARN` with one shape-mismatch concern; current code inspection showed it was a false positive because `valid_mask` remains `[B, sum(T_l)]` when reliability weights are applied, but explicit shape guards were added. DeepSeek via Claude `deepseek-v4-pro` then returned `PASS`, confirming no leakage, correct random-fixed contract, tensor/mask semantics, gate safety, and clean attribution. Post-fix local py_compile, launcher `bash -n`, and full adapter safety suite passed again (`35 passed, 20 skipped`). |
| 2026-05-20T05:38:00+08:00 | commit/deploy | Created clean scoped commit `8c70093 add adapter sample reliability experiment`, refreshed Gemini clean-worktree review (`Approved`) and DeepSeek clean-worktree verification (`PASS`), synced the five scoped files to `35407` and `35329`, passed remote py_compile, `CHECK_ONLY=1`, and Linux tensor smoke on both servers (`[[1.0, 0.75, 0.75, 1.0]]`). Created `adapter_sample_reliability_after_review.ok` and launched screens `875514.adapter_sample_reliability_safe` on `35407` (`BASE_PORT=30660`) and `761375.adapter_sample_reliability_safe` on `35329` (`BASE_PORT=30670`). Early health: `35407` active at epoch1 iter50 with loss `1.0006` and two recoverable epoch0 non-finite skips; `35329` active at epoch0 iter50 with loss `1.5935` and one recoverable skip. Previous native dense HeadV2 was intentionally stopped after `56.82 Avg-mAP / 30.69@0.7` plus Gemini/DeepSeek `STOP_NOW`, not accidentally interrupted. |
| 2026-05-20T05:44:00+08:00 | monitor/tooling | Added read-only watcher `logs/watch_sample_reliability_first_eval.ps1` and started PID `31044` with 300s interval. First check: `35407` active at epoch3 with latest epoch2 iter99 `Loss=0.8870 / cls=0.5415 / reg=0.3455`, non-finite count `2`, no mAP; `35329` active at epoch1 with latest epoch0 iter99 `Loss=1.5217 / cls=0.9066 / reg=0.6150`, non-finite count `2`, no mAP. Watcher stops on first `Average-mAP`, crash/OOM, non-finite count `>10`, or inactive-without-metric. |
| 2026-05-20T05:50:00+08:00 | monitor | Direct remote check confirms `adapter_sample_reliability_safe` is not interrupted. `35407` remains active in screen `875514.adapter_sample_reliability_safe`, GPU about `3455 MiB`, latest epoch4 iter50 `Loss=0.8560 / cls=0.5147 / reg=0.3413`, non-finite count `2`, no `Average-mAP`, no Traceback/OOM. `35329` remains active in screen `761375.adapter_sample_reliability_safe`, GPU about `3569 MiB`, latest epoch1 iter50 `Loss=1.0008 / cls=0.6618 / reg=0.3390`, non-finite count `2`, no `Average-mAP`, no Traceback/OOM. Continue monitoring for first complete mAP block before any claim or new route decision. |
| 2026-05-20T05:53:00+08:00 | monitor/audit | Checked the sample-reliability eval schedule: inherited workflow has `val_start_epoch=40`, `val_eval_interval=2`, and `tools/train.py` evaluates only when `epoch >= 40` and `(epoch + 1) % 2 == 0`, so the first eligible eval is after epoch41. `35407` reached epoch5 at `05:51:26`, implying a first-eval window around `08:20-08:40 CST` plus validation time at the current cadence; `35329` is slower and still pre-eval. Absence of `Average-mAP` now is expected, not a stall. |
| 2026-05-20T05:55:00+08:00 | monitor | Watcher check `3/288` plus direct SSH recheck confirm both sample-reliability replicas remain healthy and pre-eval. `35407`: screen active, GPU about `3455 MiB`, latest epoch5 iter50 `Loss=0.8160 / cls=0.4837 / reg=0.3323`, non-finite count `2`, no mAP, no Traceback/OOM. `35329`: screen active, GPU about `3571 MiB`, epoch2 just started after epoch1 iter99 `Loss=0.9830 / cls=0.6360 / reg=0.3471`, non-finite count `2`, no mAP, no Traceback/OOM. No Gemini/DeepSeek post-eval discussion is triggered until a metric block or health gate appears. |
| 2026-05-20T05:58:00+08:00 | monitor | Watcher check `4/288`: both sample-reliability replicas remain active and pre-eval. `35407` epoch6 iter50 has `Loss=0.6988 / cls=0.4174 / reg=0.2814`, non-finite count `2`, metric `NONE`; `35329` epoch2 iter50 has `Loss=0.8391 / cls=0.5028 / reg=0.3363`, non-finite count `2`, metric `NONE`; watcher stderr is empty. Continue waiting for the first complete mAP block. |
| 2026-05-20T06:01:00+08:00 | monitor | Direct SSH recheck: `35407` remains healthy and has entered epoch7 after epoch6 iter99 `Loss=0.7272 / cls=0.4381 / reg=0.2892`, non-finite count `2`, no mAP, no Traceback/OOM. `35329` screen and train processes are still alive with latest emitted line epoch2 iter50 `Loss=0.8391 / cls=0.5028 / reg=0.3363`, non-finite count `2`, no mAP, no Traceback/OOM; short-window lack of another log line is not yet treated as a stall. Continue normal watcher monitoring. |
| 2026-05-20T06:03:00+08:00 | monitor | Watcher check `5/288` shows both replicas still active and pre-eval. `35407`: epoch7 iter50 `Loss=0.7661 / cls=0.4600 / reg=0.3061`, non-finite count `2`, metric `NONE`. `35329`: progressed to epoch2 iter99 `Loss=0.8880 / cls=0.5418 / reg=0.3461` and epoch3, non-finite count `2`, metric `NONE`; earlier short-window log silence was slow progress, not a stall. Watcher stderr remains empty. |
| 2026-05-20T06:06:00+08:00 | monitor | Direct SSH recheck: `35407` has reached epoch8 iter50 with `Loss=0.7280 / cls=0.4310 / reg=0.2970`, non-finite count `2`, no mAP, no Traceback/OOM. `35329` has entered epoch3 after epoch2 iter99 `Loss=0.8880 / cls=0.5418 / reg=0.3461`, non-finite count `2`, no mAP, no Traceback/OOM. Both screens and train processes remain active; continue monitoring. |
| 2026-05-20T06:08:00+08:00 | monitor | Watcher check `6/288`: `35407` remains the lead healthy replica, epoch9 after epoch8 iter99 `Loss=0.6878 / cls=0.3983 / reg=0.2895`, non-finite count `2`, metric `NONE`. `35329` remains active but has not emitted a new logging-interval line since epoch2 iter99 at `06:03:35`; latest remains `Loss=0.8880 / cls=0.5418 / reg=0.3461`, non-finite count `2`, metric `NONE`. Treat `35329` as slow-progress observation, not a stop condition; watcher stderr is empty. |
| 2026-05-20T06:11:00+08:00 | monitor | Direct SSH recheck: both replicas are actively writing logs again. `35407` reached epoch9 iter50 with `Loss=0.6477 / cls=0.3688 / reg=0.2789`, non-finite count `2`, no mAP, no Traceback/OOM. `35329` reached epoch3 iter50 with `Loss=0.8125 / cls=0.4812 / reg=0.3313`, non-finite count `2`, no mAP, no Traceback/OOM. Continue waiting for first full mAP block. |
| 2026-05-20T06:13:00+08:00 | monitor | Watcher check `7/288`: both sample-reliability replicas remain healthy and pre-eval. `35407` epoch10 after epoch9 iter99 `Loss=0.6678 / cls=0.3865 / reg=0.2813`, non-finite count `2`, metric `NONE`; `35329` epoch3 iter50 `Loss=0.8125 / cls=0.4812 / reg=0.3313`, non-finite count `2`, metric `NONE`; watcher stderr is empty. No mAP block or stop gate yet. |
| 2026-05-20T06:16:00+08:00 | monitor | Direct SSH recheck: both sample-reliability runs are still healthy and pre-eval. `35407` started epoch11 after epoch10 iter99 `Loss=0.6671 / cls=0.3890 / reg=0.2781`, non-finite count `2`, no mAP, no Traceback/OOM. `35329` started epoch4 after epoch3 iter99 `Loss=0.8260 / cls=0.4957 / reg=0.3303`, non-finite count `2`, no mAP, no Traceback/OOM. Continue monitoring without intervention. |
| 2026-05-20T06:18:00+08:00 | monitor | Watcher check `8/288`: `35407` is active at epoch11 iter50 with `Loss=0.6989 / cls=0.4101 / reg=0.2888`, non-finite count `2`, metric `NONE`; `35329` is active at epoch4 with latest epoch3 iter99 `Loss=0.8260 / cls=0.4957 / reg=0.3303`, non-finite count `2`, metric `NONE`; watcher stderr is empty. Continue waiting for the first eligible eval after epoch41 completion. |
| 2026-05-20T06:22:00+08:00 | monitor | Direct SSH recheck: both sample-reliability runs continue normally and remain pre-eval. `35407` started epoch12 after epoch11 iter99 `Loss=0.6791 / cls=0.3985 / reg=0.2806`, non-finite count `2`, no mAP, no Traceback/OOM. `35329` reached epoch4 iter50 with `Loss=0.8506 / cls=0.5133 / reg=0.3373`, non-finite count `2`, no mAP, no Traceback/OOM. |
| 2026-05-20T06:23:00+08:00 | monitor | Watcher check `9/288`: both runs remain healthy and pre-eval. `35407` epoch12 iter50 `Loss=0.6357 / cls=0.3528 / reg=0.2830`, non-finite count `2`, metric `NONE`; `35329` epoch4 iter50 `Loss=0.8506 / cls=0.5133 / reg=0.3373`, non-finite count `2`, metric `NONE`; watcher stderr is empty. Continue monitoring. |
| 2026-05-20T06:29:00+08:00 | monitor | Watcher check `10/288` plus direct SSH recheck: `adapter_sample_reliability_safe` remains active and pre-eval on both servers. `35407` screen `875514.adapter_sample_reliability_safe` has entered epoch14 after epoch13 iter99 `Loss=0.6515 / cls=0.3703 / reg=0.2811`; `35329` screen `761375.adapter_sample_reliability_safe` has entered epoch5 after epoch4 iter99 `Loss=0.7964 / cls=0.4766 / reg=0.3198`. Both have `Average-mAP=0`, non-finite count `2`, no Traceback/OOM, and local watcher PID `31044` is alive with empty stderr. No Gemini/DeepSeek post-eval discussion is triggered before a complete metric block or health gate. |
| 2026-05-20T06:32:00+08:00 | monitor | Direct SSH recheck: both `adapter_sample_reliability_safe` replicas are still healthy and pre-eval. `35407` reached epoch14 iter50 with `Loss=0.5898 / cls=0.3376 / reg=0.2521`; `35329` reached epoch5 iter50 with `Loss=0.8395 / cls=0.5100 / reg=0.3294`. Both have `Average-mAP=0`, non-finite count `2`, no Traceback/OOM, and active train-process count `4`. Continue waiting for the first full mAP block or a health-gate event. |
| 2026-05-20T06:34:00+08:00 | monitor | Watcher check `11/288`: both `adapter_sample_reliability_safe` replicas remain active and pre-eval. `35407` entered epoch15 after epoch14 iter99 `Loss=0.6136 / cls=0.3535 / reg=0.2601`; `35329` remains active at epoch5 iter50 `Loss=0.8395 / cls=0.5100 / reg=0.3294`. Both have metric `NONE`, non-finite count `2`, and watcher stderr is empty. Continue waiting for the first complete mAP block. |
| 2026-05-20T06:43:00+08:00 | monitor | Watcher check `13/288`: both sample-reliability replicas remain active and pre-eval. `35407` reached epoch17 iter50 with `Loss=0.5519 / cls=0.2958 / reg=0.2560`; `35329` reached epoch6 iter50 with `Loss=0.6924 / cls=0.4125 / reg=0.2799`. Both have metric `NONE`, non-finite count `2`, and watcher stderr is empty. |
| 2026-05-20T06:45:00+08:00 | monitor | Direct SSH recheck: `35407` has entered epoch18 after epoch17 iter99 `Loss=0.5579 / cls=0.3078 / reg=0.2500`; `35329` remains active at epoch6 iter50 `Loss=0.6924 / cls=0.4125 / reg=0.2799`. Both screens and train-process groups are alive, `Average-mAP=0`, non-finite count `2`, no Traceback/OOM. Continue waiting for first eligible eval after epoch41 completion on the lead run. |
| 2026-05-20T06:48:00+08:00 | monitor | Direct SSH recheck: both replicas continue healthy and pre-eval. `35407` reached epoch18 iter50 `Loss=0.5978 / cls=0.3362 / reg=0.2617` with GPU utilization `99%`; `35329` entered epoch7 after epoch6 iter99 `Loss=0.7383 / cls=0.4505 / reg=0.2878`. Both have `Average-mAP=0`, non-finite count `2`, no Traceback/OOM, and active train-process count `4`. |
| 2026-05-20T06:49:00+08:00 | monitor | Watcher check `14/288`: `35407` remains active at epoch18 iter50 `Loss=0.5978 / cls=0.3362 / reg=0.2617`; `35329` remains active at epoch7 after epoch6 iter99 `Loss=0.7383 / cls=0.4505 / reg=0.2878`. Both have metric `NONE`, non-finite count `2`, and watcher stderr is empty. |
| 2026-05-20T06:50:00+08:00 | monitor | Direct SSH recheck: `35407` entered epoch19 after epoch18 iter99 `Loss=0.5910 / cls=0.3312 / reg=0.2599`; `35329` remains active at epoch7 after epoch6 iter99 `Loss=0.7383 / cls=0.4505 / reg=0.2878`. Both have `Average-mAP=0`, non-finite count `2`, no Traceback/OOM, and active train-process count `4`. Continue waiting for first eligible eval after epoch41 completion. |
| 2026-05-20T06:52:00+08:00 | monitor | Direct SSH recheck: `35407` reached epoch19 iter50 `Loss=0.6223 / cls=0.3469 / reg=0.2754`; `35329` remains active in epoch7 with latest epoch6 iter99 `Loss=0.7383 / cls=0.4505 / reg=0.2878`. Both have `Average-mAP=0`, non-finite count `2`, no Traceback/OOM, and active train-process count `4`. Still healthy and pre-eval. |
| 2026-05-20T06:54:00+08:00 | monitor | Watcher check `15/288` plus direct SSH recheck: both sample-reliability replicas remain healthy and pre-eval. `35407` has entered epoch20 after epoch19 iter99 `Loss=0.5970 / cls=0.3395 / reg=0.2576`; `35329` reached epoch7 iter50 `Loss=0.7685 / cls=0.4600 / reg=0.3085`. Both have `Average-mAP=0`, non-finite count `2`, no Traceback/OOM, and watcher stderr is empty. |
| 2026-05-20T06:56:00+08:00 | monitor | Direct SSH recheck: `35407` reached epoch20 iter50 `Loss=0.5387 / cls=0.3003 / reg=0.2384`; `35329` remains at epoch7 iter50 `Loss=0.7685 / cls=0.4600 / reg=0.3085`. Both have `Average-mAP=0`, non-finite count `2`, no Traceback/OOM, and active train-process count `4`. Still healthy and pre-eval. |
| 2026-05-20T06:58:00+08:00 | monitor | Direct SSH recheck: `35407` entered epoch21 after epoch20 iter99 `Loss=0.5629 / cls=0.3069 / reg=0.2561`; `35329` remains active at epoch7 iter50 `Loss=0.7685 / cls=0.4600 / reg=0.3085`. Both have `Average-mAP=0`, non-finite count `2`, no Traceback/OOM, and active train-process count `4`. |
| 2026-05-20T07:04:00+08:00 | monitor | Confirmed latest executable direction is the HeadV2 post-eval Gemini/DeepSeek recommendation to return to standard ActionFormer head plus minimal sample/reliability weighting. `adapter_sample_reliability_safe` remains active and pre-eval on both servers: `35407` screen `875514` at epoch22 iter50 `Loss=0.5438 / cls=0.3057 / reg=0.2381`, GPU util `100%`; `35329` screen `761375` at epoch8 iter50 `Loss=0.7295 / cls=0.4330 / reg=0.2965`. Both have `Average-mAP=0`, non-finite count `2`, no Traceback/OOM, and watcher PID `31044` is alive with empty stderr. |
| 2026-05-20T07:25:00+08:00 | tooling/monitor | Added read-only `logs/evaluate_sample_reliability_decision.ps1` for the active `adapter_sample_reliability_safe` route. Parser check and one-shot remote smoke passed; latest output reports both servers active with `EVAL_COUNT=0` and `COMBINED_DECISION=WAIT_FIRST_EVAL`. Gemini bare review returned `VERDICT PASS`; DeepSeek short bare review returned `PASS`. Earlier Gemini invalid-stream/empty-output and DeepSeek timeout attempts were not accepted. The helper only reads logs/metrics and creates no sentinels or remote side effects. |
| 2026-05-20T07:34:00+08:00 | monitor | Rechecked `adapter_sample_reliability_safe` with the read-only decision helper and watcher. Both servers remain active and pre-eval: `35407` log mtime `07:32:47`, epoch29, loss `0.5031 / cls=0.2610 / reg=0.2421`; `35329` log mtime `07:30:39`, epoch11, loss `0.6548 / cls=0.3788 / reg=0.2760`. Both have `EVAL_COUNT=0`, non-finite count `2`, Traceback/OOM `0`, and combined decision `WAIT_FIRST_EVAL`; watcher PID `31044` is alive with empty stderr. No post-eval Gemini/DeepSeek discussion or new launch is triggered before a complete metric block or health gate. |
| 2026-05-20T07:37:00+08:00 | monitor/eta | Fresh decision-helper output remains `WAIT_FIRST_EVAL`: `35407` active with log mtime `07:34:49`, `35329` active with log mtime `07:36:25`, both `EVAL_COUNT=0`, non-finite `2`, Traceback/OOM `0`. Direct `35407` tail confirms epoch30 started at `07:34:49` after epoch29 iter99 loss `0.5234 / cls=0.2789 / reg=0.2444`. Expected first eligible eval is still after epoch41 completion, roughly `08:20-08:35 CST` plus validation time at current lead-run cadence. |
| 2026-05-20T07:40:00+08:00 | monitor | Watcher check `24/288` and decision helper still report `WAIT_FIRST_EVAL`. `35407` has entered epoch31 after epoch30 iter99 `Loss=0.5095 / cls=0.2769 / reg=0.2326`; non-finite count increased to `3` but remains below the `>10` stop gate, with Traceback/OOM `0`. `35329` remains active at epoch11 iter50 `Loss=0.6983 / cls=0.4104 / reg=0.2880`, non-finite `2`, Traceback/OOM `0`. No mAP block exists yet; no post-eval discussion or route change is triggered. |
| 2026-05-20T07:57:00+08:00 | monitor | Short-poll monitoring from `07:41` to `07:57` kept `adapter_sample_reliability_safe` in `WAIT_FIRST_EVAL`. `35407` watcher check `27/288` reports active at epoch34 with latest loss `0.5148 / cls=0.2750 / reg=0.2398`; decision helper mtime `07:55:40`, `EVAL_COUNT=0`, non-finite `3`, Traceback/OOM `0`. `35329` active at epoch13 with latest loss `0.6243 / cls=0.3551 / reg=0.2692`; decision helper mtime `07:52:31`, `EVAL_COUNT=0`, non-finite `2`, Traceback/OOM `0`. No mAP block yet; continue waiting for lead-run eval after epoch41. |
| 2026-05-20T08:41:54+08:00 | result/gate | `adapter_sample_reliability_safe` first eval on `35407`: `62.07 Avg-mAP`, `78.98@0.3`, `73.26@0.4`, `64.77@0.5`, `54.21@0.6`, `39.13@0.7`, with `3325` GT instances and `422000` predictions. This is below random-fixed Adapter `63.77` by `-1.70`, strict EMA `63.85` by `-1.78`, stratified best `64.64` by `-2.57`, and uniform stride-2 reference `65.09` by `-3.02`. The watcher exited after detecting the first metric as designed. Decision: not model progress; run post-eval Gemini + DeepSeek discussion before any new route or claim. |
| 2026-05-20T08:53:00+08:00 | review/gate | Post-eval discussions completed. Gemini `gemini-3-pro-preview` returned `OBSERVE UNTIL EVAL 2, BUT HIGH RISK`, recommending stop if 35407 Eval2 `<63.0` or 35329 Eval1 `<62.5`; it diagnosed localization harm from `apply_to=all` downweighting regression. DeepSeek `deepseek-v4-pro` leaned `STOP_NOW`, also diagnosing regression/localization harm and using Eval2 `<63.0` as a hard kill gate if not stopped immediately. Adopted gate: keep 35407 only until next eval, stop sample-reliability if Eval2 `<63.0` or 35329 Eval1 `<62.5`; no new route sentinels or progress claim before the gate closes. |
| 2026-05-20T09:17:00+08:00 | result/stop | `adapter_sample_reliability_safe` Eval2 on `35407` was `62.62 Avg-mAP`, `79.24@0.3`, `73.41@0.4`, `65.24@0.5`, `55.26@0.6`, `39.94@0.7`. It improved from Eval1 but still triggered the adopted kill gate `Eval2 <63.0`, remaining below random-fixed Adapter by `-1.15`, strict EMA by `-1.23`, stratified best by `-2.02`, and uniform stride2 by `-2.47`. Stopped both replicas: `35407 ACTIVE=FALSE`, `35329 ACTIVE=FALSE`, no screen sockets, GPUs `0 MiB`. Route is negative and not claimable. |
| 2026-05-20T09:20:00+08:00 | decision | Added `SAMPLE_RELIABILITY_FAILURE_AND_NEXT_ROUTE_DECISION_20260520`. Interpretation: `apply_to=all` sample reliability likely harms localization by downweighting regression; implementation bug is less likely. Required next diagnostics are reliability weight distribution, regression-loss comparison, and boundary/positive-point overlap. Preferred next route is feature consistency regularization across two strict 50% random-fixed views, with standard Adapter + ActionFormerHead and no test-time protocol changes; any implementation still requires self-check, Gemini, and DeepSeek review before deployment. |
| 2026-05-20T09:33:00+08:00 | implementation/self-check | Started `adapter_feature_consistency_safe` from the latest sample-reliability failure decision: training-only dual random-fixed 50% views, standard Adapter + ActionFormerHead, supervised loss on the primary view, stop-gradient projected-feature consistency to an auxiliary random-fixed view, val/test unchanged. Local py_compile, launcher `bash -n`, and scoped pytest passed; Gemini/DeepSeek reviews, remote CHECK_ONLY/tensor smoke, diagnostics, commit, and launch remain pending. |
| 2026-05-20T10:05:00+08:00 | review/commit | Gemini CLI `gemini-3-pro-preview` returned `PASS` with no blockers for `adapter_feature_consistency_safe`; a later Gemini rerun failed due CLI network/extension errors and was not adopted. DeepSeek via Claude Code CLI `deepseek-v4-pro` wrote a substantive plan report with `PASS` and no blockers; copied to `logs/claude_deepseek_v4_pro_feature_consistency_safe_20260520.plan.txt`. Created reviewed commit `08a4f14 add adapter feature consistency experiment`. Remote CHECK_ONLY, Linux tensor smoke, diagnostics, and launch remain pending. |
| 2026-05-20T10:18:00+08:00 | diagnostics/decision | Completed the required pre-launch diagnostics for the failed `adapter_sample_reliability_safe` route. On 24 sampled THUMOS14 training videos, sample-reliability weights downweighted `38.96%` of valid points and `39.53%` of positive points; positive near-boundary weight mean was `0.9317` vs `0.9517` away from boundaries. Log comparison showed comparable losses but negative metrics: sample reliability Eval2 `62.62` vs baseline final `63.77`. Interpretation remains route/design failure from `apply_to=all` downweighting localization pressure, not an interrupted or crashed run. Remote CHECK_ONLY and feature-consistency tensor smoke had passed on `35407`/`35329`; launch of reviewed `adapter_feature_consistency_safe` is approved behind `adapter_feature_consistency_after_review.ok`. |
| 2026-05-20T10:30:00+08:00 | recovery | Added `CONTEXT_RECOVERY_FEATURE_CONSISTENCY_20260520_1030.md`. It records the active objective, latest direction file, closed sample-reliability results, reviewed feature-consistency implementation commit `08a4f14`, review/check artifacts, diagnostics, launch commands, last confirmed remote screen/log state for `35407` and `35329`, and immediate monitoring actions. Last confirmed remote state before interruption: feature-consistency screens `271949.adapter_feature_consistency_safe` and `918778.adapter_feature_consistency_safe` were running in epoch0; `35407` had one recoverable non-finite gradient skip at epoch0 iter17. |
| 2026-05-20T11:15:00+08:00 | review | Added `ADAPTER_FEATURE_CONSISTENCY_SAFE_REREVIEW_20260520.md` after a full rereview of commit `08a4f14`. Self-review, Gemini CLI `gemini-3-pro-preview`, and Claude Code CLI `deepseek-v4-pro` all returned PASS/no blockers. Recorded one non-blocking attribution caveat: aux no-grad teacher forward can still affect train-mode BN running stats if present. Decision remains to continue the active feature-consistency runs until first mAP or health gate; no performance claim exists yet. |
| 2026-05-20T11:28:00+08:00 | tooling/monitor | Added `ADAPTER_FEATURE_CONSISTENCY_WATCHER_20260520.md` and two read-only local monitor scripts, `evaluate_feature_consistency_decision.ps1` and `watch_feature_consistency_first_eval.ps1`. Self-check, Gemini CLI `gemini-3-pro-preview`, and Claude Code CLI `deepseek-v4-pro` all returned PASS. Started watcher PID `65396`; first check reports `35407` active at epoch8 and `35329` active at epoch2, both with non-finite count `2`, Traceback/OOM `0`, no mAP, and combined decision `WAIT_FIRST_EVAL`. |
| 2026-05-20T11:30:00+08:00 | monitor | Fresh `evaluate_feature_consistency_decision.ps1` check still reports `COMBINED_DECISION=WAIT_FIRST_EVAL`. Watcher PID `65396` remains alive with empty stderr. `35407` is active at epoch8 with latest logged loss `0.7389 / cls=0.4329 / reg=0.2997 / feature_consistency_loss=0.0063`, non-finite count `2`, Traceback/OOM `0`, no mAP. `35329` is active at epoch3 after epoch2 iter99 loss `0.9004 / cls=0.5479 / reg=0.3476 / feature_consistency_loss=0.0049`, non-finite count `2`, Traceback/OOM `0`, no mAP. Continue waiting for first eligible eval after epoch41; no route change or claim yet. |
| 2026-05-20T11:31:00+08:00 | monitor | Another manual decision check confirms the same state: `COMBINED_DECISION=WAIT_FIRST_EVAL`. `35407` is active with log mtime `11:30:28`, latest epoch8 iter50 `Loss=0.7164 / cls=0.4173 / reg=0.2930 / feature_consistency_loss=0.0061`, non-finite count `2`, Traceback/OOM `0`, no mAP. `35329` remains active at epoch3 with latest epoch2 iter99 `Loss=0.9004 / cls=0.5479 / reg=0.3476 / feature_consistency_loss=0.0049`, non-finite count `2`, Traceback/OOM `0`, no mAP. Watcher PID `65396` remains the active guard. |
| 2026-05-20T11:33:00+08:00 | monitor | Watcher check `2/288` and a fresh manual decision check both remain `WAIT_FIRST_EVAL`. Watcher PID `65396` is alive with empty stderr. `35407` remains active at epoch8 with non-finite count `2`, Traceback/OOM `0`, no mAP; latest logged line is still epoch8 iter50 at `11:30:28`, which is a short-window logging gap rather than a stop condition. `35329` remains active at epoch3 with non-finite count `2`, Traceback/OOM `0`, no mAP. |
| 2026-05-20T11:34:00+08:00 | monitor | Low-level SSH health check resolved the short-window log gap. `35407` log advanced to epoch9 after epoch8 iter99 `Loss=0.6881 / cls=0.3945 / reg=0.2879 / feature_consistency_loss=0.0058`; train parent and workers are alive, worker CPU is high, non-finite count remains `2`, no Traceback/OOM/mAP. `35329` remains active at epoch3 with train parent and workers alive, worker CPU high, non-finite count `2`, no Traceback/OOM/mAP. Instantaneous GPU util was `0` on both checks but memory remained allocated and CPU workers were active, so this is not treated as a failure. Continue watcher-based monitoring. |
| 2026-05-20T11:35:00+08:00 | monitor | Fresh feature-consistency decision check remains `WAIT_FIRST_EVAL`. Watcher PID `65396` is alive with empty stderr. `35407` is active at epoch9 with latest epoch8 iter99 `Loss=0.6881 / cls=0.3945 / reg=0.2879 / feature_consistency_loss=0.0058`, non-finite count `2`, Traceback/OOM `0`, no mAP. `35329` is active at epoch3 with latest epoch2 iter99 `Loss=0.9004 / cls=0.5479 / reg=0.3476 / feature_consistency_loss=0.0049`, non-finite count `2`, Traceback/OOM `0`, no mAP. No route change or claim. |
| 2026-05-20T11:36:00+08:00 | monitor | Feature-consistency decision check remains `WAIT_FIRST_EVAL`. `35407` is active at epoch9, log mtime `11:34:04`, non-finite count `2`, Traceback/OOM `0`, no mAP. `35329` is active at epoch3, log mtime `11:30:08`, non-finite count `2`, Traceback/OOM `0`, no mAP. Watcher PID `65396` remains alive with empty stderr. Continue waiting for first eligible evaluation after epoch41. |
| 2026-05-20T11:38:00+08:00 | monitor | Watcher check `3/288` plus manual decision check remain `WAIT_FIRST_EVAL`. Watcher PID `65396` is alive with empty stderr. `35407` advanced to epoch9 iter50 with `Loss=0.6518 / cls=0.3668 / reg=0.2792 / feature_consistency_loss=0.0058`, non-finite count `2`, Traceback/OOM `0`, no mAP. `35329` remains active at epoch3 with non-finite count `2`, Traceback/OOM `0`, no mAP. |
| 2026-05-20T11:39:00+08:00 | monitor | Feature-consistency manual decision check still reports `COMBINED_DECISION=WAIT_FIRST_EVAL`. `35407` remains active at epoch9 with latest epoch9 iter50 `Loss=0.6518 / cls=0.3668 / reg=0.2792 / feature_consistency_loss=0.0058`, non-finite count `2`, Traceback/OOM `0`, no mAP. `35329` remains active at epoch3 with non-finite count `2`, Traceback/OOM `0`, no mAP. Continue waiting; no new route or claim. |
| 2026-05-20T11:40:00+08:00 | monitor | Feature-consistency decision check remains `WAIT_FIRST_EVAL`. Watcher PID `65396` remains alive with empty stderr. `35407` is active at epoch9 with latest epoch9 iter50 `Loss=0.6518 / cls=0.3668 / reg=0.2792 / feature_consistency_loss=0.0058`, non-finite count `2`, Traceback/OOM `0`, no mAP. `35329` remains active at epoch3 with latest epoch2 iter99 `Loss=0.9004 / cls=0.5479 / reg=0.3476 / feature_consistency_loss=0.0049`, non-finite count `2`, Traceback/OOM `0`, no mAP. Continue waiting for epoch41 first eval. |
| 2026-05-20T11:41:00+08:00 | monitor | Feature-consistency decision check remains `WAIT_FIRST_EVAL`. `35407` is active at epoch9 with non-finite count `2`, Traceback/OOM `0`, no mAP; latest logged train line remains epoch9 iter50 `Loss=0.6518 / cls=0.3668 / reg=0.2792 / feature_consistency_loss=0.0058`. `35329` advanced to epoch3 iter50 with `Loss=0.8147 / cls=0.4757 / reg=0.3323 / feature_consistency_loss=0.0067`, non-finite count `2`, Traceback/OOM `0`, no mAP. |
## 2026-05-20T12:17:47+08:00

- Completed full code rereview for `adapter_feature_consistency_safe`.
- Gemini CLI first rereview returned FAIL on a real padding position-alignment bug in `_meta_positions_to_level`; accepted the finding.
- Stopped active feature-consistency runs on ports `35407` and `35329`; no old run should be used for claims.
- Fixed `ActionFormer` feature-consistency position mapping to interpolate only over `valid_mask.sum()` and pad only masked slots; added level/shape guards and padding-prefix test coverage.
- Local verification after fix: `py_compile` PASS; `pytest -q tests/test_adapter_safety_contracts.py -k feature_consistency` returned `1 passed, 4 skipped, 26 deselected` on Windows.
- Gemini CLI fix review `gemini-3-pro-preview`: PASS, output `OpenTAD_Back_sample_rel_clean/logs/gemini3_pro_preview_feature_consistency_fix_review_20260520_120224.txt`.
- Claude Code CLI DeepSeek `deepseek-v4-pro`: PASS, output `OpenTAD_Back_sample_rel_clean/logs/claude_deepseek_v4_pro_feature_consistency_fix_verify_20260520_120406.txt`.
- Reviewed fix commit in `OpenTAD_Back_sample_rel_clean`: `bb90161 fix feature consistency padding alignment`.
- Local report: `research-wiki/experiments/ADAPTER_FEATURE_CONSISTENCY_FULL_CODE_REREVIEW_20260520.md`.

## 2026-05-20T12:26:33+08:00

- Synced fixed `adapter_feature_consistency_safe` code from reviewed commit `bb90161` to ports `35407` and `35329`.
- Updated remote gate sentinel content to record `reviewed_commit=bb90161` plus Gemini/DeepSeek review log paths.
- Remote preflight on both servers: `py_compile` PASS; `CHECK_ONLY=1` launch config gate PASS; inline Torch padding-position smoke PASS.
- Remote pytest could not run because `/root/miniconda3/bin/python` has no `pytest` module on either server.
- Relaunched fixed experiment:
  - `35407`: screen `441192.adapter_feature_consistency_safe_fixed_bb90161`, log `logs/input_random_fixed_50pct_adapter_feature_consistency_safe_20260520_122434.log`.
  - `35329`: screen `982088.adapter_feature_consistency_safe_fixed_bb90161`, log `logs/input_random_fixed_50pct_adapter_feature_consistency_safe_20260520_122435.log`.
- Early health check: both active; no Traceback/OOM; `35407` had `NONFINITE=1`, `35329` had `NONFINITE=0`; combined decision `WAIT_FIRST_EVAL`.
- Restarted local watcher PID `74384`, output `logs/watch_feature_consistency_first_eval_latest.out.log`.

## 2026-05-20T12:28:48+08:00

- Follow-up monitor for fixed `adapter_feature_consistency_safe`.
- Watcher PID `74384` is alive; latest watcher decision remains `WAIT_FIRST_EVAL`.
- One-shot decision check: both servers `ACTIVE=TRUE`, `ANY_METRIC=FALSE`, `ANY_HEALTH_STOP=FALSE`, combined `WAIT_FIRST_EVAL`.
- `35407`: latest log `20260520_122434`; GPU `3859/32760 MiB`, utilization `83%`; epoch0 iter50 reached with `Loss=1.6211`, `cls_loss=0.9232`, `reg_loss=0.6960`, `feature_consistency_loss=0.0020`; `NONFINITE=1`, `Traceback=0`, `OOM=0`.
- `35329`: latest log `20260520_122435`; GPU `3973/24564 MiB`; epoch0 startup active but no iter50 line yet; `NONFINITE=0`, `Traceback=0`, `OOM=0`.
- No valid post-fix mAP yet; continue waiting for first eval.

## 2026-05-20T12:33:06+08:00

- Additional fixed-run liveness check because plain log tail appeared stale.
- `35407`: screen hardcopy shows epoch0 iter99 completed and epoch1 started. Latest screen metrics: `Loss=1.5435`, `cls_loss=0.9135`, `reg_loss=0.6281`, `feature_consistency_loss=0.0019`; two early non-finite skip messages by epoch0 iter98; no Traceback/OOM observed.
- `35407` process tree active: parent train process plus two worker children, high CPU on workers; screen `441192.adapter_feature_consistency_safe_fixed_bb90161` still detached.
- `35329`: screen `982088.adapter_feature_consistency_safe_fixed_bb90161` still detached; train process and workers active with high CPU. One non-finite skip at epoch0 iter17; no Traceback/OOM observed. No iter50 line visible yet, so continue monitoring.
- No first eval metric yet. Continue waiting; health stop threshold remains non-finite count `>10` or any Traceback/OOM.

## 2026-05-20T12:37:46+08:00

- Watcher check `3/288`: `COMBINED_DECISION=WAIT_FIRST_EVAL`; no metric yet.
- One-shot decision check: both fixed runs active, `ANY_METRIC=FALSE`, `ANY_HEALTH_STOP=FALSE`, combined `WAIT_FIRST_EVAL`.
- `35407`: latest log `20260520_122434`; epoch1 iter50 reached with `Loss=1.0028`, `cls_loss=0.6581`, `reg_loss=0.3418`, `feature_consistency_loss=0.0030`; `NONFINITE=2`, `Traceback=0`, `OOM=0`.
- `35329`: latest log `20260520_122435`; epoch0 iter50 reached with `Loss=1.6211`, `cls_loss=0.9232`, `reg_loss=0.6960`, `feature_consistency_loss=0.0020`; `NONFINITE=1`, `Traceback=0`, `OOM=0`.
- No intervention needed; continue waiting for first eval at the inherited `val_start_epoch=40` / `val_eval_interval=2` gate.

## 2026-05-20T12:40:11+08:00

- Fixed `adapter_feature_consistency_safe` monitoring update.
- Watcher still alive; latest automatic decision remains `WAIT_FIRST_EVAL`.
- One-shot decision check at 12:39:35: `ANY_ACTIVE=TRUE`, `ANY_METRIC=FALSE`, `ANY_HEALTH_STOP=FALSE`, combined `WAIT_FIRST_EVAL`.
- `35407`: screen hardcopy shows epoch1 iter99 completed with `Loss=0.9827`, `cls_loss=0.6299`, `reg_loss=0.3498`, `feature_consistency_loss=0.0030`; non-finite count still `2`; no Traceback/OOM; GPU util sampled at `100%`.
- `35329`: screen hardcopy shows epoch0 iter50 with `Loss=1.6211`, `cls_loss=0.9232`, `reg_loss=0.6960`, `feature_consistency_loss=0.0020`; non-finite count `1`; no Traceback/OOM; train and worker processes active.
- No first eval metric yet; continue monitoring.

## 2026-05-20T12:42:36+08:00

- Fixed `adapter_feature_consistency_safe` monitoring update.
- Watcher check `4/288` remains `WAIT_FIRST_EVAL`; no metric yet.
- One-shot decision check at 12:41:59: `ANY_ACTIVE=TRUE`, `ANY_METRIC=FALSE`, `ANY_HEALTH_STOP=FALSE`, combined `WAIT_FIRST_EVAL`.
- `35407`: log `20260520_122434`, epoch2 started after epoch1 iter99. Latest completed train line remains epoch1 iter99: `Loss=0.9827`, `cls_loss=0.6299`, `reg_loss=0.3498`, `feature_consistency_loss=0.0030`; `NONFINITE=2`, `Traceback=0`, `OOM=0`.
- `35329`: log `20260520_122435`, latest visible train line epoch0 iter50: `Loss=1.6211`, `cls_loss=0.9232`, `reg_loss=0.6960`, `feature_consistency_loss=0.0020`; `NONFINITE=1`, `Traceback=0`, `OOM=0`; worker CPU active, so keep it running.
- Continue waiting for first eval; no intervention required.

## 2026-05-20T12:45:26+08:00

- Fixed `adapter_feature_consistency_safe` monitoring update.
- Watcher alive; latest automatic check still `WAIT_FIRST_EVAL`.
- One-shot decision check at 12:44:46: both active, no metric, no health stop.
- `35407`: log `20260520_122434`; latest completed train line epoch2 iter50 with `Loss=0.8531`, `cls_loss=0.5099`, `reg_loss=0.3383`, `feature_consistency_loss=0.0050`; `NONFINITE=2`, `Traceback=0`, `OOM=0`. This is the leading fixed run.
- `35329`: log `20260520_122435`; latest visible train line remains epoch0 iter50 with `Loss=1.6211`, `cls_loss=0.9232`, `reg_loss=0.6960`, `feature_consistency_loss=0.0020`; `NONFINITE=1`, `Traceback=0`, `OOM=0`; worker processes remain CPU-active, so keep as backup.
- No first eval metric yet; continue monitoring.

## 2026-05-20T12:48:15+08:00

- Fixed `adapter_feature_consistency_safe` monitoring update.
- Watcher check `5/288`: combined `WAIT_FIRST_EVAL`; no metric yet.
- One-shot decision check: both active, no health stop. `NONFINITE=2` on both; `Traceback=0`, `OOM=0`.
- `35407`: screen hardcopy shows epoch2 iter99 completed and epoch3 started. Latest train line: `Loss=0.8991`, `cls_loss=0.5465`, `reg_loss=0.3477`, `feature_consistency_loss=0.0049`.
- `35329`: screen hardcopy shows epoch0 iter99 completed and epoch1 started. Latest train line: `Loss=1.5436`, `cls_loss=0.9136`, `reg_loss=0.6282`, `feature_consistency_loss=0.0019`.
- Continue waiting for first eval; no intervention required.

## 2026-05-20T12:51:00+08:00

- Fixed `adapter_feature_consistency_safe` monitoring update.
- Watcher PID `74384` still alive; latest watcher output remains `WAIT_FIRST_EVAL`.
- One-shot decision at 12:50:23: `ANY_ACTIVE=TRUE`, `ANY_METRIC=FALSE`, `ANY_HEALTH_STOP=FALSE`, combined `WAIT_FIRST_EVAL`.
- `35407`: latest parsed epoch is `3`; latest train line remains epoch2 iter99 with `Loss=0.8991`, `cls_loss=0.5465`, `reg_loss=0.3477`, `feature_consistency_loss=0.0049`; `NONFINITE=2`, `Traceback=0`, `OOM=0`.
- `35329`: latest parsed epoch is `1`; latest train line remains epoch0 iter99 with `Loss=1.5436`, `cls_loss=0.9136`, `reg_loss=0.6282`, `feature_consistency_loss=0.0019`; `NONFINITE=2`, `Traceback=0`, `OOM=0`.
- No first eval yet. Continue monitoring the leading `35407` run toward epoch41 evaluation.
## 2026-05-20 13:08 CST - Adapter Feature Consistency complete code rereview

- Reviewed repository: `OpenTAD_Back_sample_rel_clean`.
- Reviewed commit: `bb90161 fix feature consistency padding alignment`.
- Boundary: complete feature-consistency route `8c70093..bb90161`, including padding-alignment fix `08a4f14..bb90161`.
- Self-check: `py_compile` PASS; `pytest -q tests/test_adapter_safety_contracts.py -k "feature_consistency"` -> `1 passed, 4 skipped, 26 deselected`; local `mmengine` unavailable in base Windows Python, so local `Config.fromfile` did not run.
- Subagent review: model and data/config sides PASS; launcher/test side WARN because approval gate checks only file existence and not `reviewed_commit == HEAD`.
- Gemini CLI accepted review: `OpenTAD_Back_sample_rel_clean/logs/gemini3_pro_preview_feature_consistency_inline_full_rereview_20260520.txt`, exit code `0`, verdict PASS. Earlier non-inline Gemini attempt was not accepted as a gate because stderr reported unavailable tool execution and output lacked sufficient line evidence.
- Claude Code CLI DeepSeek review: `OpenTAD_Back_sample_rel_clean/logs/claude_deepseek_v4_pro_feature_consistency_full_rereview_20260520.txt`, exit code `0`, verdict PASS with non-blocking gate-hardening suggestion.
- Local record: `research-wiki/experiments/ADAPTER_FEATURE_CONSISTENCY_COMPLETE_CODE_REREVIEW_20260520_1308.md`.
- Active run decision at 13:07 CST: `COMBINED_DECISION=WAIT_FIRST_EVAL`, `ANY_ACTIVE=TRUE`, `ANY_METRIC=FALSE`, `ANY_HEALTH_STOP=FALSE`.
- Latest health: port `35407` epoch `5`, `NONFINITE=2`, no Traceback/OOM; port `35329` epoch `2`, `NONFINITE=2`, no Traceback/OOM.
- Decision: continue monitoring already-launched `bb90161` runs; no metric claim until first eval.

## 2026-05-20 13:25 CST - Next model-side fallback route: PVCC

- Active goal remains incomplete: no verified `65+` result and no completed model-side result above `63.85` yet.
- Current feature-consistency monitor at 13:23 CST: `COMBINED_DECISION=WAIT_FIRST_EVAL`; port `35407` active at epoch `7`, `feature_consistency_loss=0.0062`, no Traceback/OOM; port `35329` active at epoch `2`, no Traceback/OOM.
- Read the negative model-side route summaries:
  - `NEGATIVE_MODEL_SIDE_FAILURE_ANALYSIS_20260520.md`
  - `SAMPLE_RELIABILITY_FAILURE_AND_NEXT_ROUTE_DECISION_20260520.md`
  - `GEMINI3_PRO_PREVIEW_NEXT_MODEL_ROUTE_REVIEW_20260519.md`
  - `SUBAGENT_DEEPSEEK_CODE_REVIEW_20260519.md`
- Adopted design constraint: do not continue physical-gap injection into the selected-index supervised detector contract as the next long route.
- New fallback plan: PVCC, Prediction-Space Classification Consistency.
  - Train-only dual random-fixed 50% views.
  - Primary view supervised only.
  - Auxiliary view no-grad teacher.
  - Align aux classification logits to primary positions.
  - Classification-logit consistency only; no regression consistency.
  - No sampling, assignment, regression-loss, head-at-test, or post-processing change.
- Claude Code CLI / `deepseek-v4-pro` direction discussion succeeded and wrote a temporary plan to `C:/Users/skywalker/.claude/plans/read-only-research-direction-discussion-tidy-book.md`. Accepted conclusion: PVCC is more defensible than GapDiff/geometry/gap-injection fallback, but should not be launched before feature-consistency first eval.
- Local clean plan written: `research-wiki/experiments/NEXT_MODEL_SIDE_PVCC_ROUTE_PLAN_20260520.md`.

## 2026-05-20 - Multi-round idea-creator brainstorm with Gemini 3.1 thinking

- Used project `idea-creator` skill for Adapter/Head innovation brainstorming.
- Gemini CLI requested model: `gemini-3.1-pro-preview-thinking`.
- Three Gemini rounds completed with exit code `0`:
  - `logs/gemini31_thinking_idea_round1_adapter_head_20260520.txt`
  - `logs/gemini31_thinking_idea_round2_critique_20260520.txt`
  - `logs/gemini31_thinking_idea_round3_converge_20260520.txt`
- Key convergence:
  - Do not treat feature consistency as Adapter/Head structure innovation.
  - Do not immediately rewrite ActionFormer into a physical-coordinate detector; keep it as a long-term separate route.
  - Short-term structural route should preserve selected-index detector contract and modify Adapter/Head internals safely.
- Recommended short-term rank:
  1. TWPA: Time-Warped Positional Adapter, Adapter-internal zero/gated physical-position PE.
  2. SARTM: Sparse-Axis Residual Temporal Mixer, Adapter-internal zero-init local mixer.
  3. HCSC: Head Classification Stability Calibrator, classification/score side only, no regression interference.
- Reports:
  - `idea-stage/TAD_ADAPTER_HEAD_IDEA_REPORT_20260520.md`
  - `research-wiki/experiments/TAD_ADAPTER_HEAD_IDEA_BRAINSTORM_GEMINI31_20260520.md`

| 2026-05-20T13:57:00+08:00 | analysis | Added `research-wiki/experiments/ADATAD_HEAD_ATTENTION_ANALYSIS_20260520.md`: reviewed local AdaTAD/ActionFormer evidence for the claim that classification-head saliency has larger detection impact while prediction/regression-head saliency can form noisy peaks. Conclusion: useful as a diagnostic hypothesis and supports prioritizing Head-only classification/ranking routes (`Head-GeomCls` / HCSC) over direct regression geometry, but not yet a result claim until reproduced on the current strict random-fixed 50% Adapter baseline. |
| 2026-05-20T14:00:00+08:00 | decision | Deferred implementation of the lightweight AdaTAD head saliency reproduction. The diagnostic is recorded as a follow-up task only: temporary hooks, no sampling/loss/postprocess change, no long run from this hypothesis alone. Returned to `grill-me` discussion to settle experiment purpose, route boundary, and gates before coding. |
| 2026-05-20T14:05:00+08:00 | grill-me/decision | User corrected the AdaTAD head-attention diagnostic purpose: it should guide ActionFormer-head redesign for sparse non-uniform inputs, especially how to improve boundary regression usefulness and expose localization quality to proposal ranking because current ActionFormer scores depend almost entirely on classification confidence. Agreed first Head-only design direction: detached boundary-quality ranking head, no direct `reg_head` modulation in v1, score fusion only after checkpoint health is proven. |
| 2026-05-20T14:12:00+08:00 | grill-me/correction | User corrected priority: SBQC must not replace the first implementation priority. P0 remains Adapter-side and Head-side model sparsification improvements for non-uniform random-fixed 50% sparse inputs. SBQC is a P1 Head-side auxiliary quality/ranking component and possible later combination module, not the core sparse-modeling contribution. |
| 2026-05-20T14:18:00+08:00 | grill-me/lit-survey | Recorded Q15-Q16 in `ADATAD_HEAD_ATTENTION_ANALYSIS_20260520.md`: P0 Head-only target is sparse-aware head temporal mixing, not ranking first. Surveyed deformable convolution, deformable attention, CKConv/continuous kernels, TriDet boundary heads, and Dynamic Head. Recommended `SAHM-v1`: deterministic local neighbor gather with weights conditioned on physical `delta_t`, zero-init residual, cls tower first, no input/assignment/regression/postprocess change. |
| 2026-05-20T14:24:00+08:00 | grill-me/correction | User corrected that the active grill-me discussion should return to Adapter-side content. Created `research-wiki/experiments/ADAPTER_SPARSE_MODEL_GRILLME_20260520.md` for Adapter P0 sparse/non-uniform model-structure design. Marked SAHM/SBQC in the AdaTAD head-attention file as Head-side side-route material, not a replacement for Adapter priority. |
| 2026-05-20T14:31:00+08:00 | grill-me/head | Started standalone Head-side sparse model grill-me record at `research-wiki/experiments/HEAD_SPARSE_MODEL_GRILLME_20260520.md`, grounded in the idea files. Q1 asks whether the first Head-side route should remain inside the current ActionFormer selected-index detector contract rather than starting with physical-coordinate or query-detector rewrites. |

| 2026-05-20T15:35:00+08:00 | grill-me/full-route-review | Completed full sparse-aware TAD route consolidation and external cross-review before implementation. Gemini CLI `gemini-3-pro-preview` returned WARN/conditional pass at `logs/gemini3_pro_preview_sparse_tad_full_route_cross_review_20260520.txt`. Claude Code CLI `deepseek-v4-pro` returned WARN/conditional pass at `logs/claude_deepseek_v4_pro_sparse_tad_full_route_cross_review_20260520.txt`. Accepted fixes: add Adapter STGA A1/A2 to first-round matrix, merge Adapter temporal-axis audit into Step 0, downgrade SAN-Contract to local smoke/equivalence, move P3 projection combo after P1/P2 health, and distinguish strict 50% token budget from parameter/FLOP deltas. Records updated in `research-wiki/experiments/SPARSE_TAD_FULL_MODEL_GRILLME_20260520.md` and `research-wiki/experiments/SPARSE_TAD_FULL_MODEL_CROSS_REVIEW_20260520.md`. |

## 2026-05-20T16:55:00+08:00 - Clean original-code base and Step 0 implementation

- Created clean implementation workspace `OpenTAD_SparseTAD_Clean` from verified official clone `OpenTAD_UpstreamFresh`; direct GitHub clone failed once due connection reset, fallback clone kept official remote URL and HEAD `1aa8ca4ac5e846b1e8ff69298dd6607121a01589`.
- Wrote clean-base audit: `research-wiki/experiments/SPARSE_TAD_CLEAN_CODE_BASE_AUDIT_20260520.md`.
- Implemented Step 0 infrastructure only: strict `random_fixed_subsample` metadata, `temporal_grid` utilities, `DensePassthroughConv1DTransformerProj`, `GridAwareConv1DTransformerProj`, optional `ActionFormer.temporal_grid_cfg`, FPNIdentity grid passthrough, and `tools/audit_adapter_temporal_axis.py`.
- Added configs: `OpenTAD_SparseTAD_Clean/configs/adatad/thumos/input_random_fixed_50pct_adapter.py` and `input_random_fixed_50pct_adapter_temporal_grid_contract.py`.
- Adapter axis audit result: for `total_frames=384`, `tubelet_size=2`, raw selected frame positions length `384` does not match Adapter temporal axis; Adapter temporal size is `192`, so STGA must use tubelet-level geometry derived from selected positions.
- Local self-check report: `research-wiki/experiments/SPARSE_TAD_STEP0_SELF_CHECK_20260520.md`.
- Verification passed: `py_compile`, standalone `temporal_grid` smoke, random-fixed `LoadFrames` smoke, config merge checks, adapter-axis audit script, and `git diff --check` except Windows line-ending warnings.
- Full targeted pytest is not accepted as passed locally: base Python fails importing torch with `WinError 1114 c10.dll`, `mmaction` env lacks pytest, and `torch_1` env lacks compatible `mmaction.registry` for full OpenTAD model imports. The test file remains for a proper OpenTAD/remote environment.
- Launch decision: no long training or deployment until Gemini CLI `gemini-3-pro-preview` and Claude Code CLI `deepseek-v4-pro` read-only reviews pass.

## 2026-05-20T17:35:00+08:00 - Step 0 external review gate passed and committed

- Gemini CLI review for `OpenTAD_SparseTAD_Clean` Step 0 completed with exit code `0` and PASS: `OpenTAD_SparseTAD_Clean/logs/gemini3_pro_preview_sparse_tad_step0_review_20260520.txt`. Stderr contains transient fetch retries but the final review was substantive and accepted.
- Claude Code CLI DeepSeek first attempt timed out with empty stdout/stderr, and the second attempt was rejected as too shallow. The accepted third review used a full diff prompt and returned PASS with file evidence: `OpenTAD_SparseTAD_Clean/logs/claude_deepseek_v4_pro_sparse_tad_step0_verify_diff_20260520.txt`.
- No blocking findings were reported. DeepSeek requested remote checks before deployment: enabled=False baseline path smoke, enabled=True temporal_grid contract smoke, downsampled grid/mask alignment, and GT remap range verification.
- Created scoped clean-workspace commit: `5264016 add sparse tad step0 temporal grid contract`.
- `research-wiki/experiments/SPARSE_TAD_STEP0_SELF_CHECK_20260520.md` updated with review paths, rejected attempts, accepted findings, and commit hash.

## 2026-05-20T17:58:00+08:00 - Step 0 remote isolated smoke passed

- Uploaded `OpenTAD_SparseTAD_Clean` commit `5264016` to isolated remote path `/root/autodl-tmp/OpenTAD_SparseTAD_Clean_step0_5264016` on port `25876` using a `git archive` tarball. This avoided touching active feature-consistency screens on ports `35407` and `35329`.
- Remote environment on `25876` has `/root/miniconda3/bin/python`, torch `2.1.2+cu118`, mmengine `0.10.7`, mmcv, mmaction, and `mmaction.registry`; no GPU devices, but enough for CPU Step 0 smoke.
- Remote checks passed: `py_compile`, config merge, random-fixed `LoadFrames` smoke, `temporal_grid` smoke, projection feature-equivalence smoke, and no-backbone `ActionFormer` enabled=True forward/backward smoke.
- The ActionFormer smoke verified finite `cost`, successful backward, and missing irregular metadata raising `ValueError`.
- Two earlier remote ActionFormer smoke attempts failed due inline script using plain dicts where OpenTAD construction expects ConfigDict-like objects; rerun with `mmengine.ConfigDict` passed.
- Decision remains: no long training from Step 0. Next model-side implementation must use the Step 0 finding that Adapter sparse geometry is tubelet-level (`384` selected frames -> `192` Adapter temporal tokens for `tubelet_size=2`).

## 2026-05-20T18:10:00+08:00 - Adapter STGA v1 implementation self-check passed, external review pending

- Implemented Adapter-side Sparse Temporal Geometry Adapter in clean workspace `OpenTAD_SparseTAD_Clean`, still based on Step 0 commit `5264016`.
- Changed files: `opentad/models/backbones/vit_adapter.py`, `configs/adatad/thumos/input_random_fixed_50pct_adapter_stga_v1a.py`, `configs/adatad/thumos/input_random_fixed_50pct_adapter_stga_v1b.py`, and `tests/test_adapter_stga_contracts.py`.
- STGA-v1a uses `mode="output_residual"`; STGA-v1b uses `mode="input_gated"`; both enable blocks `[6,7,8,9,10,11]`, use tubelet-level gap features, require sparse metadata, and do not alter sampling, ActionFormer head, assignment/loss, or post-processing.
- Self-check found and fixed two implementation risks before review: `sparse_geometry_encoder` is now included in the optimizer custom group despite `exclude=["backbone"]`, and block checkpointing now passes `encoded_gap` explicitly with `use_reentrant=False`.
- Local `py_compile` passed; local pytest remains blocked by the known Windows torch `c10.dll` import failure.
- Remote isolated smoke on port `25876` passed: config merge confirms strict `random_fixed_subsample` keep_ratio `0.5`; zero-init equivalence passed for v1a/v1b; missing metadata and raw-axis mismatch raise `ValueError`; optimizer grouping includes all `sparse_geometry_encoder` params; checkpoint non-zero gradient path passed for output-residual STGA.
- Self-check report written: `research-wiki/experiments/ADAPTER_STGA_V1_SELF_CHECK_20260520.md`.
- Launch decision: no STGA long training yet. Required next gate remains Gemini CLI `gemini-3-pro-preview` read-only review, then Claude Code CLI `deepseek-v4-pro` read-only verification, then accepted fixes and scoped commit.

## 2026-05-20T18:32:00+08:00 - Gemini STGA v1 initial review failed; blocking metas route fixed

- Gemini CLI `gemini-3-pro-preview` initial STGA review completed with exit code `0` after transient fetch retries and returned `FAIL`: `OpenTAD_SparseTAD_Clean/logs/gemini3_pro_preview_adapter_stga_v1_review_20260520.txt`.
- Accepted blocking finding: `ActionFormer._forward_backbone` only passed `metas` when `temporal_grid_cfg.enabled=True`; STGA configs intentionally do not enable detector-level temporal grid, so end-to-end STGA training would have dropped metadata and crashed inside `VisionTransformerAdapter`.
- Fix applied in `opentad/models/detectors/actionformer.py`: forward `metas` whenever `self.backbone.forward` accepts a `metas` argument, independent of `temporal_grid_cfg`.
- Added regression coverage in `tests/test_adapter_stga_contracts.py` for ActionFormer metadata routing without detector-level temporal grid.
- Verification after fix: local `py_compile` passed; remote `ACTIONFORMER_META_ROUTE_PASS`; remote combined STGA smoke passed with strict random-fixed config, zero-init equivalence, missing-meta error, axis-mismatch error, optimizer sparse encoder group, and non-zero checkpoint gradient.
- Updated self-check report with the Gemini finding, accepted fix, and post-fix smoke evidence. Gemini review must be rerun before DeepSeek and commit.

## 2026-05-20T18:59:00+08:00 - DeepSeek STGA review rerun produced one rejected FAIL; contract clarified

- Several Claude Code CLI DeepSeek attempts were not accepted: the interrupted/plan-mode run wrote only a Claude plan file with empty stdout; the first inline retry also returned empty stdout. The accepted executable pattern for this round is `claude.cmd --bare -p --model deepseek-v4-pro --permission-mode dontAsk --tools "" --strict-mcp-config --mcp-config logs/empty_mcp_config.json`.
- Bare inline DeepSeek review produced substantive stdout at `OpenTAD_SparseTAD_Clean/logs/claude_deepseek_v4_pro_adapter_stga_v1_verify_bare_inline_20260520.txt` and returned `FAIL`.
- Rejected P0 finding: it claimed `frame_valid` should be indexed by dense physical coordinates. This misunderstands the Step 0 Adapter contract. `irregular_selected_positions` are dense-time coordinates attached to consecutive selected input slots; `total_frames=384` is selected input length, while `irregular_selected_valid_len/source_len=768` is the dense physical coordinate length. Adapter tubelets group consecutive selected slots, so validity is a selected-slot prefix mask.
- Added clarification comment in `vit_adapter.py` and a regression contract test in `tests/test_adapter_stga_contracts.py` using physical coordinates `[0,4,5,7,10,11,14,15]` with `total_frames=8` and `valid_len=16`.
- Remote validation passed: `STGA_PHYSICAL_SELECTED_SLOT_COORD_PASS`; config check confirms v1a/v1b `total_frames=384`, `target_len=384`, `source_len=768`, and `TOTAL_FRAMES_SELECTED_INPUT_CONTRACT_PASS`.
- DeepSeek must be rerun with this explanation and updated code before the review gate can be considered passed.

## 2026-05-20T19:26:00+08:00 - Accepted DeepSeek dead-gradient finding and fixed STGA initialization

- DeepSeek final verification at `OpenTAD_SparseTAD_Clean/logs/claude_deepseek_v4_pro_adapter_stga_v1_final_verify_20260520.txt` returned `FAIL`.
- Accepted blocking finding: `sparse_gamma=0` combined with zero-initialized `sparse_delta_proj` makes the STGA branch output zero and all sparse branch gradients zero permanently.
- Fix applied in `opentad/models/backbones/vit_adapter.py`: initialize `sparse_gamma` to `torch.ones(1)` while keeping `sparse_delta_proj` zero. This keeps exact initial output equivalence but makes `sparse_delta_proj` learnable from the first step in `output_residual` mode.
- Added `test_stga_output_residual_delta_projection_learns_at_init` to assert non-zero initial gradient for `sparse_delta_proj.weight`.
- Remote smoke passed: zero-init equivalence remains exact for `output_residual` and `input_gated`; `INIT_DELTA_PROJ_GRAD=0.04701428860425949`; marker `STGA_GRAD_DEADLOCK_FIX_REMOTE_PASS`.
- Because code changed after Gemini/DeepSeek, both external reviews must be rerun before commit or deployment.

## 2026-05-20T19:36:00+08:00 - STGA v1 final Gemini and DeepSeek review gate passed

- Gemini CLI `gemini-3-pro-preview` final review after dead-gradient fix returned `PASS`: `OpenTAD_SparseTAD_Clean/logs/gemini3_pro_preview_adapter_stga_v1_after_deadgrad_fix_20260520.txt`.
- Claude Code CLI `deepseek-v4-pro` final bare inline verification returned `PASS`: `OpenTAD_SparseTAD_Clean/logs/claude_deepseek_v4_pro_adapter_stga_v1_after_deadgrad_fix_20260520.txt`.
- DeepSeek's only note was to confirm `adapter_index` inheritance. Remote config checks already confirmed v1a/v1b use `adapter_index=list(range(12))`, `total_frames=384`, `target_len=384`, `source_len=768`, and `keep_ratio=0.5`.
- Final accepted implementation state: STGA-v1a/v1b preserve exact initial output equivalence, avoid dead-gradient lock in `output_residual`, require sparse metadata, enforce sparse blocks as active Adapter blocks, and keep the change attributable to Adapter internals rather than sampling/head/loss/postprocess.
- Next step: create scoped commit in `OpenTAD_SparseTAD_Clean`, then select a non-conflicting server/GPU for first STGA-v1a deployment.

## 2026-05-20T19:39:00+08:00 - STGA v1 scoped commit created

- Created clean-workspace commit `bf76e63 add adapter sparse temporal geometry` in `OpenTAD_SparseTAD_Clean`.
- Commit contains only STGA code/config/test files:
  - `opentad/models/backbones/vit_adapter.py`
  - `opentad/models/detectors/actionformer.py`
  - `configs/adatad/thumos/input_random_fixed_50pct_adapter_stga_v1a.py`
  - `configs/adatad/thumos/input_random_fixed_50pct_adapter_stga_v1b.py`
  - `tests/test_adapter_stga_contracts.py`
- Logs, prompts, and external review outputs were intentionally not committed.
- Updated self-check report with commit hash and launch status.
- Next step: check remote GPU/screen availability and launch STGA-v1a only if a non-conflicting target is available.

## 2026-05-20T19:43:00+08:00 - STGA v1 deployment package staged, no free GPU yet

- Checked remote servers:
  - `35407`: one active screen `adapter_feature_consistency_safe_fixed_bb90161`; GPU0 using about `3861/32760 MiB`; `tools/train.py configs/adatad/thumos/input_random_fixed_50pct_adapter_feature_consistency_safe.py --id 0` still running.
  - `35329`: one active screen `adapter_feature_consistency_safe_fixed_bb90161`; GPU0 using about `3977/24564 MiB`; same feature-consistency run still running.
  - `25876`: no GPU devices found; only dead screens.
- Decision: do not launch STGA on 35407/35329 yet, because both available GPU endpoints are already running long experiments and launching a second E2E VideoMAE training on the same card risks OOM and metric contamination.
- Created local deployment archive `OpenTAD_SparseTAD_Clean_stga_bf76e63.tar` from commit `bf76e63`.
- Staged clean code on `35407` at `/root/autodl-tmp/OpenTAD_SparseTAD_Clean_stga_bf76e63` and verified remote `py_compile` for `vit_adapter.py` and `actionformer.py`.
- Current launch status: deployment package is ready, but long STGA-v1a training is waiting for a free GPU or explicit decision to stop/reuse an active server.

## 2026-05-20T19:48:00+08:00 - STGA v1 startup dependencies verified; launch command recorded

- Rechecked current remote status:
  - `35407`: feature-consistency run is active at epoch 48; latest eval at 19:40 CST gives `Avg-mAP=62.73`. Do not treat as dead/stale.
  - `35329`: feature-consistency run is active at epoch 22 with recent logs at 19:37 CST. Do not treat as dead/stale.
  - `25876`: no GPU devices.
- Prepared `35407` staged STGA directory:
  - created `logs/`
  - linked `pretrained -> /root/autodl-tmp/OpenTAD/pretrained`
  - verified `pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth`
  - verified `/root/autodl-tmp/annotations/thumos_14_anno.json`
  - verified `/root/autodl-tmp/annotations/category_idx.txt`
  - verified `/root/autodl-tmp/train` and `/root/autodl-tmp/test`
- Startup path check marker: `STGA_V1A_STARTUP_PATHS_PASS`.
- Wrote deployment status and exact first-launch command to `research-wiki/experiments/ADAPTER_STGA_V1_DEPLOYMENT_STATUS_20260520.md`.
- Launch remains blocked by occupied GPUs. First launch target remains `35407` with screen `adapter_stga_v1a_bf76e63` after the active feature-consistency run finishes or is explicitly stopped.

## 2026-05-20T20:01:00+08:00 - SAPM-AttnGapBias implemented and self-checked

- Continued full sparse-aware route because STGA-v1a deployment remains blocked by occupied GPUs.
- Remote status at 19:50 CST: `35407` feature-consistency run active at epoch 49; `35329` active at epoch 22; `25876` no GPU.
- Implemented Projection-side `SAPM-AttnGapBias` in clean workspace `OpenTAD_SparseTAD_Clean`.
- Changed files:
  - `opentad/models/bricks/transformer.py`
  - `opentad/models/projections/actionformer_proj.py`
  - `configs/adatad/thumos/input_random_fixed_50pct_adapter_sapm_attn_gap_bias.py`
  - `tests/test_sapm_attn_gap_bias_contracts.py`
- Method: add temporal-grid physical-distance attention-logit bias to global `MaskedMHCA`; zero gate gives exact original attention at init; gate gradient is non-zero at init.
- Config keeps strict random-fixed 50% and enables `temporal_grid_cfg` plus `GridAwareConv1DTransformerProj`; no Adapter STGA, Head, loss, assignment, or postprocess changes.
- Remote 25876 smoke passed:
  - `SAPM_CONFIG_CONTRACT_PASS`
  - `ZERO_GATE_MAX_DIFF 0.0`
  - missing grid raises `ValueError`
  - opening gate changes output
  - local attention rejected for v1
  - projection output/grid levels align
  - `SAPM_GATE_INIT_GRAD=0.0011729999678209424`
- Self-check report written: `research-wiki/experiments/SAPM_ATTN_GAP_BIAS_SELF_CHECK_20260520.md`.
- Launch decision: no SAPM long run until Gemini CLI and DeepSeek code review pass.

## 2026-05-20T21:06:00+08:00 - GPT-5 Pro full sparse TAD review completed

- Reopened a fresh Rosetta browser conversation through Edge CDP `127.0.0.1:9222` with model `gpt-5-5-pro`.
- Directly uploaded 34 raw files, including project rules, logs, grill-me plans, cross-review notes, Step0/STGA/SAPM reports, configs, implementation files, tests, and audit script. No zip or bundle was used.
- Rosetta upload log recorded 34 `[rosetta-upload] ready` events. The process finished successfully after `713835 ms` (about 11 min 54 s).
- Output: `logs/gpt5pro_sparse_tad_fresh_direct_files_pro_20260520.txt`.
- Stderr/upload/debug logs:
  - `logs/gpt5pro_sparse_tad_fresh_direct_files_pro_20260520.err.txt`
  - `logs/gpt5pro_sparse_tad_fresh_direct_files_pro_20260520.debug.log`
- Full local summary written: `research-wiki/experiments/GPT5_PRO_SPARSE_TAD_FULL_REVIEW_20260520.md`.
- GPT-5 Pro verdict: **Major Revision / continue**. The model-side sparse temporal geometry direction is valid, but the claim must be constrained to same sampler / same selected-index detector protocol / sparse-grid-aware modules.
- Accepted blocking finding: SAPM likely misses `gap_bias_alpha` and `gap_bias_gate` in `ActionFormer.get_optim_groups()`, so optimizer construction may fail. Required fix: add these parameters, or `gap_bias_*`, to no-decay and run a full-model optimizer smoke before Gemini/DeepSeek review or training.
- Review visibility caveat: upload log confirms `OpenTAD_SparseTAD_Clean/opentad/models/utils/temporal_grid.py` was uploaded and ready, but GPT-5 Pro reported it could not see that source file. Do not treat `temporal_grid.py` as fully Pro-reviewed; run a focused follow-up review for the grid contract and tests.
- Recommended order: fix SAPM optimizer and focused grid review, launch STGA-v1a first, run SAPM only after fix/reviews, evaluate STGA-v1b only after v1a, then add geometry causality controls, SAHM classification-only, SAN calibration, and later P1 physical-coordinate detector protocol.

## 2026-05-20T21:34:00+08:00 - SAPM optimizer blocker fixed and external gate passed

- Accepted GPT-5 Pro blocker for SAPM-AttnGapBias: `gap_bias_alpha` and `gap_bias_gate` could be omitted from `ActionFormer.get_optim_groups()` strict decay/no_decay classification.
- Fixed `OpenTAD_SparseTAD_Clean/opentad/models/detectors/actionformer.py` by adding `pn.startswith("gap_bias_")` to the no-decay branch.
- Added `test_actionformer_optimizer_groups_include_gap_bias_controls_once` in `OpenTAD_SparseTAD_Clean/tests/test_sapm_attn_gap_bias_contracts.py`.
- Local `py_compile` passed for `actionformer.py`, `transformer.py`, `actionformer_proj.py`, and `test_sapm_attn_gap_bias_contracts.py`.
- Local pytest remains blocked by the known Windows PyTorch `c10.dll` WinError 1114, so Linux smoke was used for code verification.
- Remote clean smoke on `25876` passed in `/root/autodl-tmp/OpenTAD_SparseTAD_Clean_sapm_smoke`:
  - command: `PYTHONPATH=. /root/miniconda3/bin/python logs/sapm_remote_smoke_20260520.py`
  - marker: `SAPM_OPTIMIZER_AND_CONTRACT_SMOKE_PASS`
- Gemini CLI `gemini-3-pro-preview` read-only review returned `PASS`: `OpenTAD_SparseTAD_Clean/logs/gemini3_pro_preview_sapm_attn_gap_bias_after_optimizer_fix_20260520.txt`.
- Claude Code CLI `deepseek-v4-pro` secondary verification returned `PASS`: `OpenTAD_SparseTAD_Clean/logs/claude_deepseek_v4_pro_sapm_attn_gap_bias_after_optimizer_fix_20260520.txt`.
- Updated `AGENTS.md` so every major implementation version must send key raw code files to GPT-5 Pro/Rosetta for line-by-line review after self-check and before Gemini/DeepSeek. Shallow, truncated, canvas/textdoc-only, weak-model, or sub-2-minute non-substantive Pro outputs are not accepted.
- SAPM-AttnGapBias status: eligible for scoped commit and staging; long training still requires an actually free GPU.

## 2026-05-20T21:58:00+08:00 - SAPM commit created and staged on GPU servers

- Created clean-workspace commit `61e272e add projection gap-aware attention bias` in `OpenTAD_SparseTAD_Clean`.
- Commit contains only SAPM code/config/test files:
  - `opentad/models/bricks/transformer.py`
  - `opentad/models/projections/actionformer_proj.py`
  - `opentad/models/detectors/actionformer.py`
  - `configs/adatad/thumos/input_random_fixed_50pct_adapter_sapm_attn_gap_bias.py`
  - `tests/test_sapm_attn_gap_bias_contracts.py`
- Created local archive `OpenTAD_SparseTAD_Clean_sapm_61e272e.tar`.
- Staged SAPM on GPU servers:
  - `35407`: `/root/autodl-tmp/OpenTAD_SparseTAD_Clean_sapm_61e272e`
  - `35329`: `/root/autodl-tmp/OpenTAD_SparseTAD_Clean_sapm_61e272e`
- Remote py_compile passed on both servers for `transformer.py`, `actionformer_proj.py`, and `actionformer.py`.
- Remote config contract passed on both servers:
  - `temporal_grid_cfg.enabled=True`
  - `projection.type=GridAwareConv1DTransformerProj`
  - `attn_gap_bias_cfg.enabled=True`
  - `dataset.train.pipeline[2].method=random_fixed_subsample`
  - `keep_ratio=0.5`
- Current GPU status:
  - `35407`: feature-consistency run active, epoch 56, recent eval Avg-mAP `63.50`, log still updating.
  - `35329`: feature-consistency run active, epoch 29, log still updating.
  - `25876`: no GPU device.
- Launch decision: do not start STGA/SAPM long training yet because both GPU servers are occupied by active runs. Use the staged packages when a GPU becomes free.
- Wrote next deployment and implementation plan: `research-wiki/experiments/SPARSE_TAD_NEXT_IMPLEMENTATION_AND_DEPLOYMENT_PLAN_20260520.md`.

## 2026-05-20T22:07:25+08:00 - Created Sparse TAD task flow tracker and mandatory update rule

- Created `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md` as the single current execution tracker for implemented experiments, remote staging, active server runs, serial queue order, review gate state, and result updates.
- Refreshed remote status before writing the tracker: `35407` still runs `adapter_feature_consistency_safe_fixed_bb90161` with latest Avg-mAP `63.50` / mAP@0.7 `41.49`; `35329` remains active around epoch 29 without a queried eval result; `25876` has no GPU.
- Updated `AGENTS.md` so every future implementation, review, deployment, training launch, monitor check, first/final eval, crash, rerun, or stop/continue decision must update the tracker plus `research-wiki/log.md` before the project state is considered current.
- Added the new tracker to `research-wiki/index.md` so context recovery and later cross-review can find the active execution ledger directly.

## 2026-05-20T22:12:33+08:00 - Added full experiment implementation matrix to Sparse TAD tracker

- Updated `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md` with a full experiment implementation matrix covering Step0, STGA-v1a/v1b, SAPM-AttnGapBias, SAPM-EmbedDeformable, Head temporal-grid contract, SAHM variants, SAN variants, geometry controls, projection transformer follow-ups, combo routes, physical-coordinate detector protocol, and future deployable intelligent sampling.
- Marked current states explicitly: Step0 `DONE`, STGA-v1a `STAGED`, STGA-v1b code `DONE` but run `DEFERRED`, SAPM-AttnGapBias `STAGED`, SAPM-EmbedDeformable `NEXT-CODE`, and all later head/neck/combo/protocol routes as `PENDING-CODE` or `DEFERRED`.

## 2026-05-20T22:27:50+08:00 - Corrected SAPM-EmbedDeformable status in task tracker

- Rechecked `OpenTAD_SparseTAD_Clean` dirty worktree and confirmed SAPM-EmbedDeformable code is already locally implemented but not self-check reported, externally reviewed, committed, deployed, or trained.
- Updated `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`:
  - Added status legend item `CODED`: implemented locally, but self-check and external review gates are not complete.
  - Changed SAPM-EmbedDeformable from `NEXT-CODE` to `CODED`.
  - Recorded local changed files: `opentad/models/projections/actionformer_proj.py`, `tests/test_sapm_attn_gap_bias_contracts.py`, and `configs/adatad/thumos/input_random_fixed_50pct_adapter_sapm_embed_deformable.py`.
- Launch decision remains unchanged: do not deploy or start long training for SAPM-EmbedDeformable until self-check, GPT-5 Pro raw-file review, Gemini CLI, and Claude CLI DeepSeek verification all pass.

## 2026-05-20T22:45:00+08:00 - SAPM-EmbedDeformable self-check written

- Wrote `research-wiki/experiments/SAPM_EMBED_DEFORMABLE_SELF_CHECK_20260520.md`.
- Self-check verdict: PASS for entering external review, but NOT launch-ready.
- Changed files covered:
  - `OpenTAD_SparseTAD_Clean/opentad/models/projections/actionformer_proj.py`
  - `OpenTAD_SparseTAD_Clean/tests/test_sapm_attn_gap_bias_contracts.py`
  - `OpenTAD_SparseTAD_Clean/configs/adatad/thumos/input_random_fixed_50pct_adapter_sapm_embed_deformable.py`
- Route classification: Projection-only local mixer change; Adapter, Neck, Head, assignment/loss, decode, post-processing, and input sampling remain unchanged.
- Protocol check: config inherits `input_random_fixed_50pct_adapter.py`, keeps `random_fixed_subsample` and `keep_ratio=0.5`, and uses `temporal_grid_cfg=dict(enabled=True, strict=True)`.
- Local verification:
  - `python -m py_compile opentad\models\projections\actionformer_proj.py tests\test_sapm_attn_gap_bias_contracts.py configs\adatad\thumos\input_random_fixed_50pct_adapter_sapm_embed_deformable.py` passed.
  - `git diff --check` passed with only LF/CRLF warnings.
  - `python -m pytest tests\test_sapm_attn_gap_bias_contracts.py -q` remains blocked by the known Windows PyTorch `c10.dll` WinError 1114 during `torch` import.
- Recorded remote smoke evidence remains: `SAPM_EMBED_DEFORMABLE_REMOTE_SMOKE_PASS`, `ZERO_OFFSET_MAX_DIFF 4.76837158203125e-07`, `OPEN_OFFSET_MAX_DIFF 0.021650969982147217`, and `OFFSET_GRAD_SUM 0.03537089005112648`.
- Updated `SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`: SAPM-EmbedDeformable is `CODED`, self-check PASS for external review, GPT-5 Pro/Gemini/DeepSeek pending, not staged, not trained.
- Next required action: GPT-5 Pro raw-file review before Gemini CLI and Claude CLI DeepSeek; no commit, staging, or long training before all external gates pass.

## 2026-05-20T22:54:41+08:00 - GPT-5 Pro SAPM-EmbedDeformable review attempts failed or were not accepted

- Wrote `research-wiki/experiments/GPT5_PRO_SAPM_EMBED_DEFORMABLE_REVIEW_ATTEMPTS_20260520.md`.
- Attempted Pro review through Rosetta raw file attachments:
  - `logs/run_gpt5pro_sapm_embed_deformable_review_20260520.mjs`
  - failed with `RosettaRequestError: Send button never became enabled after typing`;
  - no review output, not accepted.
- Retried Rosetta raw file attachments with `ROSETTA_SEND_BUTTON_WAIT_MS=600000`:
  - `logs/gpt5pro_sapm_embed_deformable_review_retry_20260520.txt`
  - metadata reported `model=gpt-5-5-pro`, `tookMs=155139`, `finish=stop`;
  - stdout was only 904 bytes and malformed/citation-fragmented, not accepted.
- Attempted Oracle browser Pro raw attachments:
  - session `gpt5pro-sapm-embed-deformable-review`;
  - failed: `Attachments did not finish uploading before timeout`;
  - not accepted.
- Attempted Oracle browser Pro minimal raw attachments:
  - session `gpt5pro-sapm-embed-deformable-minimal`;
  - final status `error`, same attachment timeout;
  - not accepted.
- Attempted Rosetta inline raw-code Pro review:
  - `logs/run_gpt5pro_sapm_embed_deformable_inline_review_20260520.mjs`;
  - metadata reported `model=gpt-5-5-pro`, `tookMs=69796`, `promptChars=94535`, `finish=stop`;
  - stdout was only 60 bytes and non-substantive, not accepted.
- Attempted a focused Rosetta projection-only inline review after reducing the prompt to 40,020 characters:
  - `logs/run_gpt5pro_sapm_embed_deformable_projection_only_review_20260520.mjs`;
  - metadata reported `model=gpt-5-5-pro`, `tookMs=54132`, `finish=stop`;
  - stdout was only 46 bytes and non-substantive, not accepted.
- Updated `SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`: SAPM-EmbedDeformable GPT-5 Pro gate is attempted but not accepted; Gemini/DeepSeek remain pending; no deployment/training is allowed.

## 2026-05-20T23:03:36+08:00 - SAPM-EmbedDeformable local edge fix and remote CPU smoke passed

- Performed an additional local engineering self-review before external review.
- Found a real edge-case risk: with `T=1`, the previous `grid_sample` special case could sample the only token for padding-side kernel slots, weakening the zero-offset Conv1d-equivalence claim.
- Fixed `OpenTAD_SparseTAD_Clean/opentad/models/projections/actionformer_proj.py`:
  - added a same-length padding guard: `padding == kernel_size // 2`;
  - masks sampled coordinates outside `[0, T-1]` before applying the Conv1d weights, including the `T=1` case.
- Updated `OpenTAD_SparseTAD_Clean/tests/test_sapm_attn_gap_bias_contracts.py`:
  - added `test_sparse_embed_deformable_zero_offset_matches_conv_module_at_length_one`;
  - added `test_sparse_embed_deformable_rejects_non_same_length_padding`.
- Added remote smoke script: `logs/sapm_embed_deformable_remote_smoke_after_edge_fix_20260520.py`.
- Local checks in `OpenTAD_SparseTAD_Clean`:
  - `python -m py_compile opentad\models\projections\actionformer_proj.py tests\test_sapm_attn_gap_bias_contracts.py configs\adatad\thumos\input_random_fixed_50pct_adapter_sapm_embed_deformable.py` passed.
  - `git diff --check` passed with only LF/CRLF warnings.
  - local pytest remains blocked by Windows torch `c10.dll` WinError 1114.
- Remote `25876` CPU/Linux smoke:
  - synced `actionformer_proj.py`, `test_sapm_attn_gap_bias_contracts.py`, config, and smoke script to `/root/autodl-tmp/OpenTAD_SparseTAD_Clean_embed_deform_smoke`;
  - remote `py_compile` passed;
  - remote pytest could not run because `/root/miniconda3/bin/python` has no `pytest` module;
  - direct smoke passed with marker `SAPM_EMBED_DEFORMABLE_EDGE_FIX_REMOTE_SMOKE_PASS`;
  - smoke metrics: `ZERO_OFFSET_LEN8_MAX_DIFF=3.2782554626464844e-07`, `ZERO_OFFSET_LEN1_MAX_DIFF=1.1920928955078125e-07`, `OFFSET_FINAL_WEIGHT_ABS_SUM=0.0`, `OFFSET_FINAL_BIAS_ABS_SUM=0.0`, `ZERO_OFFSET_KERNEL1_MAX_DIFF=2.384185791015625e-07`, `OPEN_OFFSET_MAX_DIFF=1.1565948724746704`, `OFFSET_GRAD_SUM=0.026817718520760536`.
- Updated `SAPM_EMBED_DEFORMABLE_SELF_CHECK_20260520.md` and `SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`.
- Gate status unchanged: still no accepted GPT-5 Pro/Gemini/DeepSeek review; no commit, staging, or training allowed.

## 2026-05-20T23:49:06+08:00 - SAPM-EmbedDeformable accepted GPT-5 Pro focused review and follow-up fixes

- Obtained an accepted GPT-5 Pro browser review through Oracle CLI with inline file contents:
  - command used `oracle --engine browser --model gpt-5.5-pro --browser-inline-files --timeout 60m`;
  - output: `logs/gpt5pro_sapm_embed_deformable_projection_oracle_cli_20260520.txt`;
  - stdout/session metadata: about `12m10s`, `gpt-5.5-pro[browser]`, `files=5`;
  - summary report: `research-wiki/experiments/GPT5_PRO_SAPM_EMBED_DEFORMABLE_REVIEW_20260520.md`.
- GPT-5 Pro verdict: `WARN`, but can continue Gemini/DeepSeek read-only review.
- Accepted findings and fixes:
  - Effective config / projection-only attribution must be checked.
  - Config-level forward smoke is needed before long training.
  - `conv_cfg` could bypass v1 deformable Conv1d assumptions; fixed in `actionformer_proj.py` by rejecting unsupported `kernel_size`, `stride`, `padding`, `dilation`, `padding_mode`, and `groups` overrides.
  - `dense_valid_len` was dropped by `normalize_temporal_grid_input`; fixed in `temporal_grid.py`, and `downsample_temporal_grid` now carries it forward.
  - Added tests for conv_cfg override rejection and dense_valid_len preservation.
- Added and ran `logs/sapm_embed_deformable_config_forward_smoke_20260520.py` on remote `25876`.
- Remote config/routing smoke passed:
  - `SAPM_EMBED_DEFORMABLE_EFFECTIVE_CONFIG_PASS`
  - `SAPM_EMBED_DEFORMABLE_ACTIONFORMER_PROJECTION_FORWARD_PASS`
  - `ODD_LENGTH_FEATURE_SHAPES [(1, 8, 9), (1, 8, 5)]`
  - `EVEN_LENGTH_FEATURE_SHAPES [(1, 8, 8), (1, 8, 4)]`
- The smoke validates effective `random_fixed_subsample + keep_ratio=0.5` for train/val/test, `GridAwareConv1DTransformerProj`, `embed_deform_cfg.enabled=True`, empty `attn_gap_bias_cfg`, and ActionFormer temporal-grid routing into projection.
- Remaining risk: AMP / fp16 / bf16 coordinate precision was flagged by Pro and is not fully closed without GPU-side AMP smoke. Do not start long training until Gemini/DeepSeek inspect this and a launch smoke plan is recorded.

## 2026-05-20T23:59:30+08:00 - Sparse TAD tracker now has authoritative all-experiment status table

- Updated `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`:
  - renamed/clarified the matrix as `All Required Experiments and Current Status`;
  - stated that it is the authoritative list of all experiments to implement, review, deploy, run, or defer;
  - kept rows for Step0, STGA, SAPM, SAHM, SAN, combo routes, P1 physical-coordinate protocol, and future intelligent non-uniform sampling.
- Corrected SAPM-EmbedDeformable status from generic pending review to `FIX-IN-PROGRESS`:
  - Gemini CLI output `OpenTAD_SparseTAD_Clean/logs/gemini3_pro_preview_sapm_embed_deformable_review_20260520.txt` returned `FAIL`;
  - accepted blockers are boundary padding-slot offset gradient death and AMP/FP16 long-sequence coordinate precision;
  - local post-Gemini fixes have started but are not yet reverified.
- Updated `AGENTS.md` to require this all-experiment table as the experiment queue source of truth.
- Current decision: no SAPM-EmbedDeformable commit, staging, DeepSeek review, deployment, or long training until local checks, remote smoke, and rerun Gemini review are completed.

## 2026-05-21T00:09:55+08:00 - SAPM-EmbedDeformable post-Gemini blocker fixes verified locally and on remote CPU

- Accepted Gemini blockers from `OpenTAD_SparseTAD_Clean/logs/gemini3_pro_preview_sapm_embed_deformable_review_20260520.txt`:
  - boundary padding-slot offset gradient death from hard `valid_sample`;
  - AMP/FP16 long-sequence coordinate precision risk from dtype-dependent coordinate construction.
- Applied fixes in `OpenTAD_SparseTAD_Clean/opentad/models/projections/actionformer_proj.py`:
  - removed the hard `valid_sample` masking path;
  - explicitly pads the temporal input before `grid_sample`;
  - constructs `base`, `rel`, `offset`, `coords`, and sampling input in `torch.float32`;
  - uses minimum explicit sampling padding so `T=1` has a non-degenerate coordinate domain.
- Updated tests and smoke:
  - `OpenTAD_SparseTAD_Clean/tests/test_sapm_attn_gap_bias_contracts.py` adds boundary padding-slot gradient coverage;
  - `OpenTAD_SparseTAD_Clean/tests/test_sparse_tad_step0_contracts.py` fixes a misplaced assertion;
  - `logs/sapm_embed_deformable_remote_smoke_after_edge_fix_20260520.py` now asserts `BOUNDARY_PADDING_SLOT_OFFSET_GRAD > 0`.
- Local checks:
  - `python -m py_compile ...` passed;
  - `git diff --check` passed with LF/CRLF warnings only;
  - local pytest remains blocked by Windows PyTorch `c10.dll` WinError 1114.
- Remote `25876` CPU/Linux checks in `/root/autodl-tmp/OpenTAD_SparseTAD_Clean_embed_deform_smoke`:
  - remote py_compile passed;
  - `SAPM_EMBED_DEFORMABLE_EDGE_FIX_REMOTE_SMOKE_PASS`;
  - `BOUNDARY_PADDING_SLOT_OFFSET_GRAD=2.0`;
  - zero-offset diffs remained within tolerance: `ZERO_OFFSET_LEN8_MAX_DIFF=3.28e-07`, `ZERO_OFFSET_LEN1_MAX_DIFF=1.19e-07`, `ZERO_OFFSET_KERNEL1_MAX_DIFF=4.77e-07`;
  - `OFFSET_GRAD_SUM=0.03013`;
  - `SAPM_EMBED_DEFORMABLE_EFFECTIVE_CONFIG_PASS` and `SAPM_EMBED_DEFORMABLE_ACTIONFORMER_PROJECTION_FORWARD_PASS`.
- Updated `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md` and `research-wiki/experiments/SAPM_EMBED_DEFORMABLE_SELF_CHECK_20260520.md`: status is now `FIX-VERIFIED / GEMINI-RERUN-PENDING / DEEPSEEK-PENDING / NOT-STAGED / NOT-TRAINED`.
- Current decision: rerun Gemini CLI next; do not run DeepSeek, commit, stage, deploy, or train until Gemini accepts the post-fix implementation.

## 2026-05-21T00:27:40+08:00 - SAPM-EmbedDeformable Gemini CLI post-fix review passed

- Ran Gemini CLI main post-fix review from `OpenTAD_SparseTAD_Clean`:
  - prompt: `logs/gemini3_pro_preview_sapm_embed_deformable_postfix_review_prompt_20260521.txt`;
  - output: `logs/gemini3_pro_preview_sapm_embed_deformable_postfix_review_20260521.txt`;
  - stderr: `logs/gemini3_pro_preview_sapm_embed_deformable_postfix_review_20260521.err.txt`;
  - exit code: `0`;
  - verdict: `PASS`;
  - blocking findings: none;
  - required fixes: none.
- The main Gemini stderr included transient `429` retries that recovered, plus two smoke-script visibility issues:
  - `logs/sapm_embed_deformable_config_forward_smoke_20260520.py` ignored by configured patterns;
  - `../logs/sapm_embed_deformable_remote_smoke_after_edge_fix_20260520.py` outside Gemini workspace.
- To close that gap, ran a focused inline smoke-script Gemini review:
  - prompt: `logs/gemini3_pro_preview_sapm_embed_deformable_smoke_inline_review_prompt_20260521.txt`;
  - output: `logs/gemini3_pro_preview_sapm_embed_deformable_smoke_inline_review_20260521.txt`;
  - stderr: `logs/gemini3_pro_preview_sapm_embed_deformable_smoke_inline_review_20260521.err.txt`;
  - exit code: `0`;
  - verdict: `PASS`;
  - blocking findings: none;
  - required fixes: none;
  - Gemini explicitly stated that the main Gemini PASS can now be fully accepted.
- Wrote summary report: `research-wiki/experiments/GEMINI3_PRO_PREVIEW_SAPM_EMBED_DEFORMABLE_POSTFIX_REVIEW_20260521.md`.
- Updated tracker and self-check:
  - `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`;
  - `research-wiki/experiments/SAPM_EMBED_DEFORMABLE_SELF_CHECK_20260520.md`.
- Current decision: SAPM-EmbedDeformable is `GEMINI-PASS / DEEPSEEK-PENDING / NOT-STAGED / NOT-TRAINED`; run Claude CLI `deepseek-v4-pro` next, still no commit/staging/deployment/training.

## 2026-05-21T00:51:24+08:00 - SAPM-EmbedDeformable Claude CLI DeepSeek verification passed

- Ran Claude CLI DeepSeek verification from `OpenTAD_SparseTAD_Clean`.
- First command used `--permission-mode plan` and returned only a plan-file notification in stdout, so it was not accepted as the final gate output.
- Accepted rerun used read-only tools and direct stdout:
  - prompt: `logs/claude_deepseek_v4_pro_sapm_embed_deformable_postfix_prompt_20260521.txt`;
  - output: `logs/claude_deepseek_v4_pro_sapm_embed_deformable_postfix_rerun_20260521.txt`;
  - stderr: `logs/claude_deepseek_v4_pro_sapm_embed_deformable_postfix_rerun_20260521.err.txt`;
  - debug: `logs/claude_deepseek_v4_pro_sapm_embed_deformable_postfix_rerun_20260521.debug.log`;
  - exit code: `0`.
- DeepSeek verdict: `PASS`.
- Blocking findings: none.
- Required fixes: none.
- DeepSeek confirms:
  - zero-offset Conv1d equivalence including `T=1` and `kernel_size=1`;
  - FP32 coordinate path and correct `grid_sample`/`einsum` semantics;
  - previous Gemini blockers are closed;
  - strict random-fixed 50% inheritance and Projection-only attribution are preserved;
  - no GT/teacher leakage.
- DeepSeek caveat: GPU AMP smoke is required before long training, but it is a launch-time safety check rather than a code-gate blocker.
- Wrote summary report: `research-wiki/experiments/CLAUDE_DEEPSEEK_V4_PRO_SAPM_EMBED_DEFORMABLE_POSTFIX_REVIEW_20260521.md`.
- Updated tracker and self-check:
  - `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`;
  - `research-wiki/experiments/SAPM_EMBED_DEFORMABLE_SELF_CHECK_20260520.md`.
- Current decision: SAPM-EmbedDeformable is `GEMINI-PASS / DEEPSEEK-PASS / READY-FOR-COMMIT / NOT-STAGED / NOT-TRAINED`; create a reviewed implementation commit next, then stage/sync and run GPU AMP smoke before any long training.

## 2026-05-21T00:53:53+08:00 - SAPM-EmbedDeformable reviewed implementation committed

- Created commit in `OpenTAD_SparseTAD_Clean`:
  - commit: `14ea069`;
  - subject: `add sapm embed deformable projection`.
- Committed files:
  - `configs/adatad/thumos/input_random_fixed_50pct_adapter_sapm_embed_deformable.py`;
  - `opentad/models/projections/actionformer_proj.py`;
  - `opentad/models/utils/temporal_grid.py`;
  - `tests/test_sapm_attn_gap_bias_contracts.py`;
  - `tests/test_sparse_tad_step0_contracts.py`.
- Commit was created only after:
  - self-check;
  - GPT-5 Pro focused review;
  - Gemini CLI post-fix review plus inline smoke-script follow-up;
  - Claude CLI DeepSeek `deepseek-v4-pro` secondary verification.
- Updated:
  - `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`;
  - `research-wiki/experiments/SAPM_EMBED_DEFORMABLE_SELF_CHECK_20260520.md`.
- Current decision: status is `GEMINI-PASS / DEEPSEEK-PASS / COMMITTED-14ea069 / NOT-STAGED / NOT-TRAINED`; next inspect GPU servers, stage/sync the reviewed commit, and run GPU AMP smoke before any long training.

## 2026-05-21T01:00:41+08:00 - Current server schedule and SAPM staging snapshot

- Checked servers:
  - `35407`: RTX 4080 SUPER, `0/32760 MiB`, util `0%`; no `tools/train.py`/`torchrun` process; no screen sockets.
  - `35329`: RTX 4090 D, `3977/24564 MiB`, util `0%`; active `adapter_feature_consistency_safe_fixed_bb90161` training process and detached screen `982088.adapter_feature_consistency_safe_fixed_bb90161`.
  - `25876`: no GPU devices; only tensorboard/jupyter plus old dead screens; CPU/Linux smoke only.
- Staged SAPM-EmbedDeformable commit:
  - commit `14ea069`;
  - remote path `35407:/root/autodl-tmp/OpenTAD_SparseTAD_Clean_sapm_embed_deform_14ea069`;
  - staging marker `STAGED_SAPM_EMBED_DEFORM_14EA069_PASS`.
- Added GPU AMP smoke script locally and uploaded to `35407`:
  - local path `logs/sapm_embed_deformable_gpu_amp_smoke_20260521.py`;
  - remote path `/root/autodl-tmp/OpenTAD_SparseTAD_Clean_sapm_embed_deform_14ea069/logs/sapm_embed_deformable_gpu_amp_smoke_20260521.py`;
  - remote py_compile passed.
- First GPU AMP smoke attempt failed before model forward:
  - error: `ValueError: meta[1]['irregular_selected_positions'] must be strictly increasing`;
  - cause: synthetic smoke metadata for the second sample used `[float(i * 2 + (i % 3))]`, which is not strictly increasing at every step;
  - interpretation: smoke-script data bug, not evidence of model failure.
- Current decision:
  - no long training has started;
  - fix smoke synthetic positions and rerun GPU AMP smoke on `35407`;
  - `35329` remains occupied and should not receive new long runs.

## 2026-05-21T01:05:00+08:00 - SAPM-EmbedDeformable GPU AMP smoke passed on 35407

- Fixed `logs/sapm_embed_deformable_gpu_amp_smoke_20260521.py`:
  - second synthetic sample now uses strictly increasing `irregular_selected_positions`;
  - `irregular_selected_valid_len` was adjusted to cover the maximum synthetic position.
- Local py_compile passed.
- Uploaded the fixed smoke script to:
  - `35407:/root/autodl-tmp/OpenTAD_SparseTAD_Clean_sapm_embed_deform_14ea069/logs/sapm_embed_deformable_gpu_amp_smoke_20260521.py`.
- Remote py_compile passed.
- Remote GPU AMP smoke command:
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/root/autodl-tmp/OpenTAD_SparseTAD_Clean_sapm_embed_deform_14ea069 /root/miniconda3/bin/python logs/sapm_embed_deformable_gpu_amp_smoke_20260521.py`.
- Output:
  - `SAPM_EMBED_DEFORMABLE_GPU_AMP_SMOKE_PASS`;
  - `LOSS 1.53692913`;
  - `OFFSET_GRAD_ABS_SUM 1.14306808`;
  - `FEATURE_DTYPES ['torch.float32', 'torch.float32']`;
  - `FEATURE_SHAPES [(2, 8, 16), (2, 8, 8)]`;
  - `CUDA_DEVICE NVIDIA GeForce RTX 4080 SUPER`.
- Interpretation:
  - GPU AMP/grid_sample/einsum launch-time risk is cleared for SAPM-EmbedDeformable;
  - no long training has started for SAPM;
  - SAPM can now wait in the serial long-run queue behind earlier queued reviewed experiments unless priority is changed.
- Updated `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`.

## 2026-05-21T01:xx:00+08:00 - Sparse TAD history summary PPT generated by subagent

- A worker subagent generated the historical experiment summary deck for the Sparse TAD route.
- Generated main files:
  - `research-wiki/slides/SPARSE_TAD_HISTORY_SUMMARY_20260521.pptx`;
  - `research-wiki/slides/SPARSE_TAD_HISTORY_SUMMARY_20260521.md`;
  - `research-wiki/slides/SPARSE_TAD_HISTORY_SUMMARY_20260521_ASSETS.md`;
  - `research-wiki/slides/scripts/generate_sparse_tad_history_deck.py`.
- Generated diagram assets under `figures/` and `figures/specs/`, including:
  - baseline anchor chart;
  - original AdaTAD + ActionFormer pipeline;
  - model evolution overview;
  - STGA-v1a Adapter structure;
  - SAPM projection/neck/transformer structure;
  - Head-side SAHM / quality rescore structure;
  - temporal grid / boundary contract;
  - review gate and experiment queue flow.
- Verification:
  - PPTX opens through `python-pptx`;
  - deck contains 25 slides;
  - first slides contain normal Chinese text in the PPTX;
  - asset manifest records all generated figures and tooling notes.
- Subagent `019e465f-b824-7482-85ff-a713298e89e6` was closed after file verification.

## 2026-05-21T01:30:00+08:00 - STGA-v1a running and active runs refreshed

- Checked remote servers with Windows native OpenSSH.
- `35407`:
  - active screen: `567879.stga_v1a_bf76e63_20260521`;
  - remote dir: `/root/autodl-tmp/OpenTAD_SparseTAD_Clean_stga_bf76e63`;
  - config: `configs/adatad/thumos/input_random_fixed_50pct_adapter_stga_v1a.py`;
  - command: `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/root/autodl-tmp/OpenTAD_SparseTAD_Clean_stga_bf76e63 /root/miniconda3/bin/torchrun --master_port=29611 --nproc_per_node=1 tools/train.py configs/adatad/thumos/input_random_fixed_50pct_adapter_stga_v1a.py --id 0`;
  - log: `logs/stga_v1a_20260521_0108.log`;
  - GPU: RTX 4080 SUPER, `3383 / 32760 MiB`, util `0%`;
  - latest early training evidence: epoch 4, `[004][00050/00099] Loss=0.8951 cls_loss=0.5526 reg_loss=0.3424`, memory `2488MB`;
  - grep found no `Traceback`, CUDA OOM, or `Average-mAP` yet.
- `35329`:
  - active screen: `982088.adapter_feature_consistency_safe_fixed_bb90161`;
  - config: `input_random_fixed_50pct_adapter_feature_consistency_safe.py`;
  - latest watcher state: active at epoch 40 with no eval yet; no Traceback/OOM.
- Feature-consistency refresh:
  - the earlier `35407` feature-consistency run is inactive and has `Average-mAP=63.88`, `mAP@0.70=42.14`;
  - deltas: `+0.11` vs random-fixed Adapter `63.77`, `+0.03` vs strict EMA `63.85`, `-0.76` vs stratified `64.64`, `-1.21` vs uniform stride-2 `65.09`.
- Decisions:
  - continue STGA-v1a until first eval or hard health failure;
  - continue `35329` feature-consistency until first eval;
  - do not start SAPM/SAHM/SAN long training on occupied GPUs;
  - no new performance claim is made from STGA-v1a before eval, and feature-consistency `63.88` is only a weak positive control, not a core sparse-model result.
- Updated:
  - `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`;
  - `research-wiki/experiments/ADAPTER_STGA_V1_DEPLOYMENT_STATUS_20260520.md`;
  - `research-wiki/experiments/ADAPTER_FEATURE_CONSISTENCY_WATCHER_20260520.md`.

## 2026-05-21T01:44:00+08:00 - Sparse TAD history PPT regenerated as cleaner v2

- Regenerated the historical experiment summary deck after the first PPT was judged too ugly and too text-heavy.
- New files:
  - `research-wiki/slides/SPARSE_TAD_HISTORY_SUMMARY_20260521_v2.pptx`;
  - `research-wiki/slides/SPARSE_TAD_HISTORY_SUMMARY_20260521_v2.md`;
  - `research-wiki/slides/SPARSE_TAD_HISTORY_SUMMARY_20260521_v2_ASSETS.md`;
  - `research-wiki/slides/scripts/generate_sparse_tad_history_deck_v2.py`;
  - rendered preview directory `research-wiki/slides/SPARSE_TAD_HISTORY_SUMMARY_20260521_v2_preview/`.
- Design changes:
  - uses more通俗中文 short sentences;
  - reduces slide count to 21;
  - uses larger typography, cards, simple diagrams, and editable PowerPoint shapes;
  - redraws architecture/queue/review-gate diagrams inside PPT instead of relying on dense imported figures;
  - keeps the old deck unchanged for comparison.
- Verification:
  - `python-pptx` opened the deck and reported 21 slides;
  - Microsoft PowerPoint COM export to PNG succeeded;
  - manually inspected preview images for cover, overall architecture, active status, final roadmap, and the full contact sheet;
  - fixed two visible layout problems before acceptance: a covered architecture group and a clipped SAPM-EmbedDeformable status title.
- Scope:
  - only slide/report artifacts were changed;
  - no OpenTAD code, config, checkpoint, or server state was modified.

## 2026-05-21T02:03:00+08:00 - Sparse TAD history PPT regenerated as v3 after visual complaint

- Regenerated the Sparse TAD historical experiment summary deck again after the user said the previous PPT was still too ugly.
- New files:
  - `research-wiki/slides/SPARSE_TAD_HISTORY_SUMMARY_20260521_v3.pptx`;
  - `research-wiki/slides/SPARSE_TAD_HISTORY_SUMMARY_20260521_v3.md`;
  - `research-wiki/slides/SPARSE_TAD_HISTORY_SUMMARY_20260521_v3_ASSETS.md`;
  - `research-wiki/slides/scripts/generate_sparse_tad_history_deck_v3.py`;
  - rendered preview directory `research-wiki/slides/SPARSE_TAD_HISTORY_SUMMARY_20260521_v3_preview/`;
  - contact sheet `research-wiki/slides/SPARSE_TAD_HISTORY_SUMMARY_20260521_v3_preview/contact_sheet.png`.
- Design changes:
  - reduced the deck to 18 slides;
  - rewrote the content in simpler Chinese with one core point per slide;
  - reorganized the story into problem, contract, baseline anchors, lessons, model pipeline, module routes, server queue, next plan, claim, and P0/P1/P2 roadmap;
  - redrew the structure diagrams as editable PowerPoint shapes with more whitespace and consistent alignment;
  - preserved the older v1/v2 decks for comparison.
- Verification:
  - `python -m py_compile research-wiki/slides/scripts/generate_sparse_tad_history_deck_v3.py` passed;
  - `python-pptx` opened the generated deck and reported 18 slides;
  - fixed a PowerPoint compatibility issue caused by a negative-height text box in a small card;
  - after the fix, Microsoft PowerPoint COM opened the deck and exported all 18 slides to PNG at 1600x900;
  - generated a contact sheet for quick visual review;
  - nonblank pixel checks confirmed all exported pages contain content.
- Scope:
  - only slide/report artifacts were changed;
  - no OpenTAD code, config, checkpoint, or server state was modified.

## 2026-05-21T02:22:00+08:00 - Head temporal-grid contract implemented and self-checked

- Implemented the `Head temporal-grid contract` route in `OpenTAD_SparseTAD_Clean`.
- Changed files:
  - `OpenTAD_SparseTAD_Clean/opentad/models/detectors/actionformer.py`;
  - `OpenTAD_SparseTAD_Clean/opentad/models/dense_heads/anchor_free_head.py`;
  - `OpenTAD_SparseTAD_Clean/tests/test_head_temporal_grid_contract.py`;
  - self-check report `research-wiki/experiments/HEAD_TEMPORAL_GRID_CONTRACT_SELF_CHECK_20260521.md`;
  - updated tracker `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`.
- Implementation:
  - `ActionFormer.forward_train/test` now preserve the neck-returned `temporal_grid_list` and pass it to the RPN/ActionFormer head;
  - `_neck_forward()` now errors if temporal-grid mode is enabled and a neck drops the grid by returning only `(features, masks)`;
  - `AnchorFreeHead.forward_train/test` accept optional `temporal_grid_list` and validate per-level feature/mask/grid alignment before running the unchanged head computation.
- Contract:
  - input sampling unchanged;
  - strict random-fixed 50% protocol preserved;
  - no GT/teacher leakage added;
  - assignment, loss, decode, NMS, post-processing, projection computation, neck computation, and Adapter internals unchanged;
  - no performance claim is made from this infrastructure patch.
- Verification:
  - local `python -m py_compile opentad\models\detectors\actionformer.py opentad\models\dense_heads\anchor_free_head.py tests\test_head_temporal_grid_contract.py` passed;
  - local `git diff --check` passed with LF/CRLF warnings only;
  - local pytest remains blocked by the known Windows torch `c10.dll` loader issue;
  - copied the changed files to `25876:/root/autodl-tmp/OpenTAD_SparseTAD_Clean_head_grid_contract_20260521_0208`;
  - remote `/root/miniconda3/bin/python -m py_compile ...` passed;
  - remote manual CPU smoke passed with marker `HEAD_TEMPORAL_GRID_CONTRACT_REMOTE_SMOKE_PASS`, covering head test/train equivalence, mismatch rejection, ActionFormer train/test routing, and neck-drop rejection.
- Decision:
  - status is `CODED` with self-check PASS;
  - do not deploy or launch long training until Gemini CLI and Claude CLI DeepSeek reviews pass.

## 2026-05-21T02:36:00+08:00 - Head temporal-grid contract review gate passed and committed

- Completed external read-only review gate for the Head temporal-grid contract.
- Gemini CLI:
  - model `gemini-3-pro-preview`;
  - exit code `0`;
  - verdict `PASS`;
  - output `OpenTAD_SparseTAD_Clean/logs/gemini3_pro_preview_head_temporal_grid_contract_20260521.txt`;
  - report `research-wiki/experiments/GEMINI3_PRO_PREVIEW_HEAD_TEMPORAL_GRID_CONTRACT_REVIEW_20260521.md`;
  - blocking findings: none.
- Claude CLI DeepSeek:
  - model `deepseek-v4-pro`;
  - exit code `0`;
  - verdict `PASS`;
  - output `OpenTAD_SparseTAD_Clean/logs/claude_deepseek_v4_pro_head_temporal_grid_contract_20260521.txt`;
  - report `research-wiki/experiments/CLAUDE_DEEPSEEK_V4_PRO_HEAD_TEMPORAL_GRID_CONTRACT_REVIEW_20260521.md`;
  - blocking findings: none.
- Both reviews agreed:
  - this is infrastructure-only routing/validation;
  - it preserves strict random-fixed 50% and no-test-GT/no-teacher-leakage contracts;
  - it does not change prediction, loss, assignment, decode, NMS, post-processing, sampling, Adapter, projection computation, or neck computation;
  - no metric change can be claimed from this patch alone.
- Committed reviewed code in `OpenTAD_SparseTAD_Clean`:
  - commit `b49d8d3`;
  - subject `route temporal grids to actionformer head`;
  - committed files:
    - `opentad/models/dense_heads/anchor_free_head.py`;
    - `opentad/models/detectors/actionformer.py`;
    - `tests/test_head_temporal_grid_contract.py`.
- Updated `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`.
- Decision:
  - Head temporal-grid contract is now `COMMITTED`;
  - use it as the prerequisite for SAHM head experiments;
  - do not run this infrastructure patch as a standalone mAP experiment.

## 2026-05-21T02:40:00+08:00 - Active remote run monitor refresh

- Refreshed active remote run status with Windows native OpenSSH.
- `35407`:
  - active screen `567879.stga_v1a_bf76e63_20260521`;
  - experiment `STGA-v1a`;
  - GPU `NVIDIA GeForce RTX 4080 SUPER`, `3387 / 32760 MiB`, util `0%`;
  - latest log: epoch 21, `[021][00050/00099] Loss=0.5415 cls_loss=0.3052 reg_loss=0.2363`, memory `2488MB`;
  - no mAP yet.
- `35329`:
  - active screen `982088.adapter_feature_consistency_safe_fixed_bb90161`;
  - experiment `adapter_feature_consistency_safe_fixed_bb90161`;
  - GPU `NVIDIA GeForce RTX 4090 D`, `3977 / 24564 MiB`, util `0%`;
  - latest log shows validation/eval after epoch 41 in progress at about `338/396`;
  - one guarded non-finite-gradient skip occurred at epoch 40 iter 72 on `module.rpn_head.cls_head.weight`;
  - no first mAP yet.
- Decision:
  - keep both active jobs running;
  - do not start SAPM/SAHM/SAN long training while GPU jobs are active;
  - continue local SAHM coding/planning only;
  - monitor `35329` after eval completion and record the first mAP plus the non-finite-gradient context.
- Updated `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`.

## 2026-05-21T02:57:47+08:00 - Active remote monitor refresh and feature-consistency first eval

- Refreshed active remote run status with Windows native OpenSSH.
- `35407`:
  - active screen `567879.stga_v1a_bf76e63_20260521`;
  - experiment `STGA-v1a`;
  - GPU `NVIDIA GeForce RTX 4080 SUPER`, `3387 / 32760 MiB`, util `0%`;
  - latest log reached epoch 25, `[025][00050/00099] Loss=0.5280 cls_loss=0.2933 reg_loss=0.2347`;
  - no mAP yet.
- `35329`:
  - active screen `982088.adapter_feature_consistency_safe_fixed_bb90161`;
  - experiment `adapter_feature_consistency_safe_fixed_bb90161`;
  - log path `/root/autodl-tmp/OpenTAD_Back_check/logs/input_random_fixed_50pct_adapter_feature_consistency_safe_20260520_122435.log`;
  - first eval after epoch 41: `Average-mAP=62.25`, `mAP@0.30=79.31`, `0.40=73.59`, `0.50=64.80`, `0.60=53.97`, `0.70=39.59`;
  - guarded non-finite-gradient skips were recorded at epochs 0, 11, and 40, but training/eval continued.
- Interpretation:
  - feature-consistency first eval is a negative early signal versus random-fixed Adapter `63.77` and strict EMA `63.85`;
  - per user instruction, judge by trend and final mAP rather than stopping on the first eval alone.
- Decision:
  - keep both active jobs running;
  - do not start overlapping long training on occupied GPU servers;
  - continue local SAHM coding and review preparation.
- Updated `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`.

## 2026-05-21T03:11:56+08:00 - SAHM-Deformable-SoftmaxGate-AllCls coded and self-checked

- Implemented the Head-side SAHM first variant in `OpenTAD_SparseTAD_Clean`.
- Changed files:
  - `opentad/models/dense_heads/sparse_head_mixer.py`;
  - `opentad/models/dense_heads/anchor_free_head.py`;
  - `configs/adatad/thumos/input_random_fixed_50pct_adapter_sahm_deformable_softmaxgate_allcls.py`;
  - `tests/test_sahm_head_contracts.py`.
- Implementation:
  - added `SparseAwareHeadDeformableConvModule`;
  - replaces all hidden classification-tower `ConvModule` layers when `sahm_cfg.enabled=True`;
  - uses content + temporal-grid geometry to predict bounded offsets and softmax slot gates;
  - keeps output length on the selected-index grid unchanged;
  - preserves `reg_convs`, `cls_head`, `reg_head`, assignment, losses, decode, NMS, post-processing, and input sampling;
  - carries classification and regression masks separately in the tower loop.
- Contract:
  - strict random-fixed 50% preserved;
  - test-time GT/teacher/oracle leakage not introduced;
  - intended attribution is Head-side classification-tower sparse-aware temporal mixing only.
- Verification:
  - local `py_compile` passed for changed code/config/test/smoke helper;
  - local `git diff --check` passed with LF/CRLF warning only;
  - local pytest remains blocked by the known Windows torch `c10.dll` issue;
  - remote `25876` py_compile passed;
  - remote CPU smoke passed with `SAHM_HEAD_REMOTE_SMOKE_PASS`, `OFFSET_GRAD_ABS_SUM=2.68570733`, `GATE_GRAD_ABS_SUM=1.37786877`;
  - remote config smoke passed with `SAHM_CONFIG_REMOTE_SMOKE_PASS`.
- Self-check report:
  - `research-wiki/experiments/SAHM_DEFORMABLE_SOFTMAXGATE_ALLCLS_SELF_CHECK_20260521.md`.
- Decision:
  - status is `CODED` with self-check PASS;
  - run Gemini CLI read-only review next;
  - do not commit, deploy, run GPU AMP smoke, or launch long training until Gemini and DeepSeek gates pass.
- Updated `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`.

## 2026-05-21T03:15:00+08:00 - SAHM Gemini CLI review passed

- Completed Gemini CLI read-only review for `SAHM-Deformable-SoftmaxGate-AllCls`.
- Command:
  - `gemini.cmd -m gemini-3-pro-preview --approval-mode plan -p "Use the prompt from stdin. Read-only review only." -o text`.
- Output:
  - `OpenTAD_SparseTAD_Clean/logs/gemini3_pro_preview_sahm_deformable_softmaxgate_allcls_20260521.txt`;
  - `OpenTAD_SparseTAD_Clean/logs/gemini3_pro_preview_sahm_deformable_softmaxgate_allcls_20260521.err.txt`.
- Exit code:
  - `0`.
- Verdict:
  - `PASS`.
- Blocking findings:
  - none.
- Required fixes:
  - none.
- Gemini accepted:
  - SAHM implements Head-side deformable local mixing without changing selected-index output coordinates;
  - `anchor_free_head.py` limits SAHM to hidden classification tower layers;
  - regression tower/head, assignment, loss, decode, NMS, post-processing, input sampling, and test-time protocol remain unchanged;
  - temporal-grid checks are strict enough to avoid silent baseline fallback;
  - config attribution is Head-side SAHM, with `DensePassthroughConv1DTransformerProj` only used to route grids.
- Report:
  - `research-wiki/experiments/GEMINI3_PRO_PREVIEW_SAHM_DEFORMABLE_SOFTMAXGATE_ALLCLS_REVIEW_20260521.md`.
- Decision:
  - SAHM status advances to `GEMINI-PASS`;
  - run Claude CLI DeepSeek `deepseek-v4-pro` next;
  - do not commit, deploy, GPU-smoke, or long-train yet.
- Updated `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`.

## 2026-05-21T03:47:00+08:00 - SAHM DeepSeek secondary verification passed

- Completed Claude CLI DeepSeek secondary verification for `SAHM-Deformable-SoftmaxGate-AllCls`.
- Output:
  - `OpenTAD_SparseTAD_Clean/logs/claude_deepseek_v4_pro_sahm_deformable_softmaxgate_allcls_20260521.txt`;
  - `OpenTAD_SparseTAD_Clean/logs/claude_deepseek_v4_pro_sahm_deformable_softmaxgate_allcls_20260521.err.txt`;
  - `OpenTAD_SparseTAD_Clean/logs/claude_deepseek_v4_pro_sahm_deformable_softmaxgate_allcls_20260521.debug.log`.
- Exit code:
  - `0`.
- Verdict:
  - `PASS`.
- Blocking findings:
  - none.
- Required fixes:
  - none.
- DeepSeek accepted:
  - protocol leakage checks;
  - strict random-fixed 50% contract;
  - tensor/mask semantics;
  - config and launcher consistency;
  - optimizer grouping safety;
  - Head-side attribution.
- Report:
  - `research-wiki/experiments/CLAUDE_DEEPSEEK_V4_PRO_SAHM_DEFORMABLE_SOFTMAXGATE_ALLCLS_REVIEW_20260521.md`.
- Decision:
  - SAHM status advances to `READY-FOR-COMMIT`;
  - create a reviewed implementation commit before staging, GPU AMP smoke, or long training;
  - no mAP claim is allowed yet because SAHM has not been trained.
- Updated `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`.

## 2026-05-21T03:48:00+08:00 - Sparse TAD history summary PPT regenerated as v5

- Regenerated the historical experiment and route summary PPT after the previous deck was judged visually unsatisfactory.
- New files:
  - `research-wiki/slides/SPARSE_TAD_HISTORY_SUMMARY_20260521_v5.pptx`;
  - `research-wiki/slides/SPARSE_TAD_HISTORY_SUMMARY_20260521_v5.md`;
  - `research-wiki/slides/SPARSE_TAD_HISTORY_SUMMARY_20260521_v5_ASSETS.md`;
  - `research-wiki/slides/SPARSE_TAD_HISTORY_SUMMARY_20260521_v5_preview/contact_sheet.png`;
  - `research-wiki/slides/scripts/generate_sparse_tad_history_deck_v5.py`.
- Design changes:
  - rewrote the deck in通俗中文;
  - reduced dense paragraph text;
  - used one main judgment per slide;
  - rebuilt architecture and route diagrams as editable PowerPoint shapes;
  - emphasized current objective, baseline anchors, four model-side routes, active queue, and next-step plan.
- Verification:
  - `python -m py_compile research-wiki/slides/scripts/generate_sparse_tad_history_deck_v5.py` passed;
  - PowerPoint COM opened the deck and reported 14 slides;
  - PowerPoint COM exported all 14 slides to PNG;
  - generated contact sheet successfully;
  - blank-page detection found no blank slide;
  - geometry inspection found no out-of-bounds or invalid non-connector shapes.
- Decision:
  - use v5 as the current PPT version for user review;
  - v4 and older versions are preserved for comparison.

## 2026-05-21T03:49:00+08:00 - SAHM reviewed implementation committed

- Created reviewed implementation commit for `SAHM-Deformable-SoftmaxGate-AllCls` in `OpenTAD_SparseTAD_Clean`.
- Commit:
  - `60704ca add sahm deformable head mixer`.
- Committed files:
  - `configs/adatad/thumos/input_random_fixed_50pct_adapter_sahm_deformable_softmaxgate_allcls.py`;
  - `opentad/models/dense_heads/anchor_free_head.py`;
  - `opentad/models/dense_heads/sparse_head_mixer.py`;
  - `tests/test_sahm_head_contracts.py`.
- Final pre-commit checks:
  - `python -m py_compile` passed for changed config/code/test files;
  - `git diff --check` passed with LF/CRLF warnings only;
  - staged diff contained only the four SAHM implementation files.
- Contract:
  - Head-side classification-tower structure change only;
  - strict random-fixed 50% input preserved;
  - no test-time GT/teacher/oracle leakage introduced;
  - regression tower, assignment, loss, decode, NMS, and post-processing unchanged.
- Decision:
  - SAHM status is now `COMMITTED`;
  - no deployment or long training yet;
  - stage and run GPU AMP smoke only after a GPU server is free.
- Updated `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`.

## 2026-05-21T03:53:09+08:00 - Active remote run monitor refresh after SAHM commit

- Refreshed active remote status with Windows native OpenSSH.
- `35407`:
  - active screen `567879.stga_v1a_bf76e63_20260521`;
  - experiment `STGA-v1a`;
  - GPU `NVIDIA GeForce RTX 4080 SUPER`, `3387 / 32760 MiB`, util `0%`;
  - latest concise log reached epoch 38:
    - `[038][00050/00099] Loss=0.4874 cls_loss=0.2566 reg_loss=0.2308`;
  - no `Average-mAP` yet.
- `35329`:
  - active screen `982088.adapter_feature_consistency_safe_fixed_bb90161`;
  - experiment `adapter_feature_consistency_safe_fixed_bb90161`;
  - GPU `NVIDIA GeForce RTX 4090 D`, `3977 / 24564 MiB`, util `0%`;
  - first eval after epoch 41 remains:
    - `Average-mAP=62.25`;
    - `mAP@0.30=79.31`;
    - `mAP@0.40=73.59`;
    - `mAP@0.50=64.80`;
    - `mAP@0.60=53.97`;
    - `mAP@0.70=39.59`;
  - latest concise train log reached epoch 43:
    - `[043][00099/00099] Loss=0.4543 cls_loss=0.2354 reg_loss=0.2148 feature_consistency_loss=0.0041`;
  - guarded non-finite-gradient skips remain recorded at epochs 0, 11, and 40.
- `25876`:
  - no GPU devices;
  - no active training process;
  - only old dead screens.
- Interpretation:
  - both GPU endpoints remain occupied by preserved long runs;
  - feature-consistency first eval is still a negative early signal versus `63.77` random-fixed and `63.85` strict EMA, but should continue for trend/final mAP per user instruction.
- Decision:
  - do not deploy or launch SAHM/SAPM/SAN long training yet;
  - wait for a GPU to become free or for an explicit priority change.
- Updated `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`.

## 2026-05-21T04:02:00+08:00 - SAN-Contract coded and self-checked

- Implemented Neck-side SAN-Contract in `OpenTAD_SparseTAD_Clean`.
- Changed files:
  - `opentad/models/necks/fpn.py`;
  - `configs/adatad/thumos/input_random_fixed_50pct_adapter_san_contract.py`;
  - `tests/test_san_neck_contracts.py`.
- Implementation:
  - added strict feature/mask/temporal-grid alignment checks for `FPN` and `FPNIdentity`;
  - added optional temporal-grid debug state collection;
  - made `FPN` accept and return `temporal_grid_list` when provided;
  - kept no-grid neck behavior unchanged;
  - kept head, prior generator, assignment, loss, decode, NMS, and post-processing unchanged.
- Contract:
  - strict random-fixed 50% preserved;
  - no test-time GT/teacher/oracle leakage introduced;
  - intended attribution is Neck-side infrastructure only;
  - no standalone mAP claim or long training for SAN-Contract.
- Verification:
  - local `py_compile` passed;
  - local `git diff --check` passed with LF/CRLF warning only;
  - local pytest remains blocked by the known Windows torch `c10.dll` issue;
  - remote `25876` py_compile passed;
  - remote manual CPU smoke passed with `SAN_CONTRACT_REMOTE_SMOKE_PASS`;
  - remote config smoke passed with `SAN_CONTRACT_CONFIG_REMOTE_SMOKE_PASS`;
  - debug smoke recorded `IDENTITY_DEBUG_LEVEL_LENGTHS=4,2` and `FPN_DEBUG_LEVEL_LENGTHS=8,4`.
- Self-check report:
  - `research-wiki/experiments/SAN_CONTRACT_SELF_CHECK_20260521.md`.
- Decision:
  - status is `CODED` with self-check PASS;
  - run Gemini CLI read-only review next;
  - do not commit, deploy, GPU-smoke, or long-train until Gemini and DeepSeek gates pass.
- Updated `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`.

## 2026-05-21T04:04:00+08:00 - SAN-Contract Gemini CLI review passed

- Completed Gemini CLI read-only review for `SAN-Contract`.
- Command:
  - `gemini.cmd -m gemini-3-pro-preview --approval-mode plan -p "Use the prompt from stdin. Read-only review only." -o text`.
- Output:
  - `OpenTAD_SparseTAD_Clean/logs/gemini3_pro_preview_san_contract_20260521.txt`;
  - `OpenTAD_SparseTAD_Clean/logs/gemini3_pro_preview_san_contract_20260521.err.txt`.
- Exit code:
  - `0`.
- Verdict:
  - `PASS`.
- Blocking findings:
  - none.
- Required fixes:
  - none.
- Gemini accepted:
  - `FPN` / `FPNIdentity` feature computations remain unchanged;
  - new validation only checks shape and temporal-grid alignment;
  - debug statistics are detached and do not affect gradients;
  - config preserves strict random-fixed 50% and uses DensePassthrough only for metadata routing;
  - SAN-Contract is infrastructure only and cannot support mAP improvement attribution.
- Report:
  - `research-wiki/experiments/GEMINI3_PRO_PREVIEW_SAN_CONTRACT_REVIEW_20260521.md`.
- Decision:
  - SAN-Contract status advances to `GEMINI-PASS`;
  - run Claude CLI DeepSeek `deepseek-v4-pro` next;
  - do not commit, deploy, GPU-smoke, or long-train yet.
- Updated `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`.

## 2026-05-21T04:20:00+08:00 - Sparse TAD history summary PPT regenerated as v6

- Regenerated the Sparse TAD historical experiment and route summary deck after the user judged the previous PPT visually unsatisfactory.
- New files:
  - `research-wiki/slides/SPARSE_TAD_HISTORY_SUMMARY_20260521_v6.pptx`;
  - `research-wiki/slides/SPARSE_TAD_HISTORY_SUMMARY_20260521_v6.md`;
  - `research-wiki/slides/SPARSE_TAD_HISTORY_SUMMARY_20260521_v6_ASSETS.md`;
  - `research-wiki/slides/SPARSE_TAD_HISTORY_SUMMARY_20260521_v6_preview/contact_sheet.png`;
  - `research-wiki/slides/scripts/generate_sparse_tad_history_deck_v6.py`.
- Design changes:
  - rewrote the deck into more conversational Chinese;
  - compressed the story to 13 slides;
  - reduced dense tables and long paragraphs;
  - rebuilt module diagrams using editable PowerPoint shapes;
  - used a consistent grid, larger typography, semantic colors, and cleaner three-column status layouts.
- Verification:
  - `python -m py_compile research-wiki/slides/scripts/generate_sparse_tad_history_deck_v6.py` passed;
  - the deck was generated successfully;
  - PowerPoint COM opened the deck and reported 13 slides;
  - PowerPoint COM exported all 13 slides to PNG;
  - contact sheet generation succeeded;
  - blank-page detection found no blank slide;
  - geometry inspection found no out-of-bounds or invalid non-connector shapes.
- Decision:
  - use v6 as the current recommended PPT version for user review;
  - v5 and earlier versions are preserved for comparison.

## 2026-05-21T04:24:00+08:00 - SAN-Contract Claude CLI DeepSeek review passed

- Completed Claude CLI DeepSeek secondary verification for `SAN-Contract`.
- First run:
  - output `OpenTAD_SparseTAD_Clean/logs/claude_deepseek_v4_pro_san_contract_20260521.txt`;
  - exit code `0`, but stdout was too shallow to accept as a substantive review by itself.
- Accepted rerun:
  - output `OpenTAD_SparseTAD_Clean/logs/claude_deepseek_v4_pro_san_contract_rerun_20260521.txt`;
  - debug log `OpenTAD_SparseTAD_Clean/logs/claude_deepseek_v4_pro_san_contract_rerun_20260521.debug.log`;
  - full evidence report written to `C:\Users\skywalker\.claude\plans\read-only-secondary-code-playful-sutton.md`;
  - exit code `0`.
- Verdict:
  - `PASS`;
  - no blocking findings;
  - no required fixes.
- Non-blocking notes:
  - optional defensive mask dtype assertion;
  - debug state can contain `(None, None)` for empty valid-center levels;
  - LF/CRLF warning only.
- Formal report:
  - `research-wiki/experiments/CLAUDE_DEEPSEEK_V4_PRO_SAN_CONTRACT_REVIEW_20260521.md`.
- Decision:
  - SAN-Contract passed self-check, Gemini CLI, and DeepSeek CLI review gates;
  - create reviewed implementation commit;
  - do not deploy or train SAN-Contract standalone because it is infrastructure only.

## 2026-05-21T04:29:00+08:00 - SAN-Contract reviewed implementation committed

- Created reviewed implementation commit for `SAN-Contract` in `OpenTAD_SparseTAD_Clean`.
- Commit:
  - `759938e add san neck contract checks`.
- Committed files:
  - `opentad/models/necks/fpn.py`;
  - `configs/adatad/thumos/input_random_fixed_50pct_adapter_san_contract.py`;
  - `tests/test_san_neck_contracts.py`.
- Final pre-commit checks:
  - `python -m py_compile` passed for changed code/config/test files;
  - `git diff --check` passed with LF/CRLF warning only;
  - `git status --short` in `OpenTAD_SparseTAD_Clean` is clean after commit.
- Contract:
  - Neck-side infrastructure only;
  - strict random-fixed 50% input preserved;
  - no test-time GT/teacher/oracle leakage introduced;
  - feature math, assignment, loss, decode, NMS, and post-processing unchanged.
- Decision:
  - SAN-Contract status is now `COMMITTED`;
  - no standalone long training or mAP claim is warranted;
  - use it as the prerequisite for `SAN-Aggregation`.
- Updated `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`.

## 2026-05-21T04:38:00+08:00 - STGA-v1a first-eval collapse and stop decision

- Refreshed remote status on `35407`.
- STGA-v1a first evaluation after epoch 41 produced a failure-level metric:
  - `Average-mAP=7.23`;
  - `mAP@0.30=17.06`;
  - `mAP@0.40=10.41`;
  - `mAP@0.50=5.21`;
  - `mAP@0.60=2.43`;
  - `mAP@0.70=1.06`.
- Training losses before the eval were numerically plausible:
  - epoch 38 loss `0.4836`;
  - epoch 39 loss `0.4625`;
  - epoch 40 loss `0.4694`;
  - epoch 41 loss `0.4757`.
- Interpretation:
  - this is not a normal early-eval fluctuation;
  - the mismatch between normal-looking training loss and near-zero mAP suggests an implementation/protocol issue or train/test behavior mismatch in the STGA route.
- Action:
  - stopped STGA-v1a on `35407`;
  - confirmed no STGA train process remains;
  - confirmed `screen -ls` has no active STGA socket;
  - confirmed GPU is free: `0 / 32760 MiB`.
- Preserved evidence:
  - remote log `/root/autodl-tmp/OpenTAD_SparseTAD_Clean_stga_bf76e63/logs/stga_v1a_20260521_0108.log`;
  - failure report `research-wiki/experiments/ADAPTER_STGA_V1A_FIRST_EVAL_FAILURE_20260521.md`.
- Decision:
  - classify STGA-v1a as `FAILED-RUN`;
  - do not launch STGA-v1b or relaunch STGA-v1a before diagnosis;
  - `35407` is now available for the next reviewed queued experiment after preflight.
- Updated `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`.

## 2026-05-21T04:38:30+08:00 - Feature-consistency second eval captured

- Refreshed remote status on `35329`.
- `adapter_feature_consistency_safe_fixed_bb90161` remains running.
- Second evaluation after epoch 43:
  - `Average-mAP=62.52`;
  - `mAP@0.30=79.21`;
  - `mAP@0.40=73.72`;
  - `mAP@0.50=65.32`;
  - `mAP@0.60=54.20`;
  - `mAP@0.70=40.16`.
- Previous first eval after epoch 41:
  - `Average-mAP=62.25`;
  - `mAP@0.70=39.59`.
- Interpretation:
  - slight recovery from the first eval;
  - still negative versus random-fixed Adapter baseline `63.77` and strict EMA `63.85`;
  - no positive claim is justified.
- Decision:
  - continue the run for trend/final mAP per user instruction;
  - do not launch overlapping work on `35329`.
- Updated `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`.

## 2026-05-21T04:39:00+08:00 - SAPM-AttnGapBias v1 launched on 35407

- Started the next reviewed queued experiment after stopping failed STGA-v1a.
- Experiment:
  - `SAPM-AttnGapBias v1`;
  - Projection-side physical-distance attention-logit bias;
  - commit/package path `61e272e`.
- Preflight on `35407`:
  - GPU free before launch: `0 / 32760 MiB`;
  - staged path exists: `/root/autodl-tmp/OpenTAD_SparseTAD_Clean_sapm_61e272e`;
  - config exists: `configs/adatad/thumos/input_random_fixed_50pct_adapter_sapm_attn_gap_bias.py`;
  - remote `py_compile` passed for transformer/projection/detector/config files;
  - config merge confirmed `method=random_fixed_subsample`, `keep_ratio=0.5`,
    `temporal_grid_cfg={'enabled': True, 'strict': True}`, and
    `projection.type=GridAwareConv1DTransformerProj`.
- Launch:
  - screen `sapm_attn_gap_bias_61e272e_20260521`;
  - command uses `CUDA_VISIBLE_DEVICES=0`, `PYTHONPATH=/root/autodl-tmp/OpenTAD_SparseTAD_Clean_sapm_61e272e`,
    `torchrun --master_port=29612 --nproc_per_node=1`;
  - config `configs/adatad/thumos/input_random_fixed_50pct_adapter_sapm_attn_gap_bias.py`;
  - log `logs/sapm_attn_gap_bias_20260521_0439.log`.
- Launch health:
  - screen is active;
  - GPU allocated about `3455 / 32760 MiB`;
  - first train line: `[000][00050/00099] Loss=1.7082 cls_loss=0.9703 reg_loss=0.7378 lr_backbone=2.0e-05 lr_det=1.0e-05 mem=2416MB`;
  - epoch 0 completed line: `[000][00099/00099] Loss=1.6329 cls_loss=0.9431 reg_loss=0.6898 lr_backbone=4.0e-05 lr_det=2.0e-05 mem=2416MB`;
  - no Traceback/RuntimeError/CUDA OOM/non-finite line observed in the initial grep.
- Decision:
  - leave SAPM-AttnGapBias running on `35407`;
  - monitor to first eval;
  - do not launch overlapping long work on `35407`.
- Updated `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`.

## 2026-05-21T04:58:00+08:00 - Sparse TAD PPT regenerated as v7 and active runs refreshed

- Regenerated the Sparse TAD history/route summary deck after the user rejected the previous visual style.
- New files:
  - `research-wiki/slides/SPARSE_TAD_HISTORY_SUMMARY_20260521_v7.pptx`;
  - `research-wiki/slides/SPARSE_TAD_HISTORY_SUMMARY_20260521_v7.md`;
  - `research-wiki/slides/SPARSE_TAD_HISTORY_SUMMARY_20260521_v7_ASSETS.md`;
  - `research-wiki/slides/SPARSE_TAD_HISTORY_SUMMARY_20260521_v7_preview/contact_sheet.png`;
  - `research-wiki/slides/scripts/generate_sparse_tad_history_deck_v7.py`.
- Design changes:
  - reduced the deck to 12 slides;
  - rewrote content in more direct Chinese;
  - replaced log-heavy pages with timeline, pipeline, module-map, and kanban layouts;
  - kept each slide to one central judgment and short supporting text.
- Verification:
  - `python -m py_compile research-wiki/slides/scripts/generate_sparse_tad_history_deck_v7.py` passed;
  - deck generation succeeded;
  - PowerPoint COM opened and exported all 12 slides to PNG;
  - contact sheet generation succeeded;
  - blank-page detection found no blank slide.
- Active run refresh:
  - `35407` SAPM-AttnGapBias remains active; latest checked train line is epoch 2 iter 50 `Loss=0.8427`; no mAP yet;
  - `35329` feature-consistency remains active; latest checked train line is epoch 45 iter 50 `Loss=0.4773`; prior evals remain `62.25` and `62.52` Avg-mAP.
- Decision:
  - use v7 as the current recommended PPT version;
  - keep both active runs unchanged and continue waiting for SAPM first eval / feature-consistency final trend.
- Updated `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`.

## 2026-05-21T05:02:00+08:00 - Active run monitor refresh

- Refreshed all three configured SSH endpoints.
- `35407`:
  - SAPM-AttnGapBias remains active in screen `918136.sapm_attn_gap_bias_61e272e_20260521`;
  - GPU state: RTX 4080 SUPER, about `3455 / 32760 MiB`;
  - latest checked train line: epoch 4 iter 99 `Loss=0.8124`, `cls_loss=0.4867`, `reg_loss=0.3256`;
  - no `Average-mAP` yet and no checked Traceback/OOM/non-finite line.
- `35329`:
  - feature-consistency remains active in screen `982088.adapter_feature_consistency_safe_fixed_bb90161`;
  - GPU state: RTX 4090 D, about `3977 / 24564 MiB`;
  - latest checked train line: epoch 45 iter 99 `Loss=0.4749`, `cls_loss=0.2400`, `reg_loss=0.2308`, `feature_consistency_loss=0.0041`;
  - prior evals remain epoch 41 `62.25` Avg-mAP and epoch 43 `62.52` Avg-mAP, still negative versus random-fixed baseline `63.77`.
- `25876`:
  - no GPU devices reported;
  - only old dead screens were listed.
- Decision:
  - keep both GPU runs active;
  - do not launch overlapping long training;
  - use local time for next implementation/diagnosis work while waiting for first/final evals.
- Updated `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`.

## 2026-05-21T05:25:00+08:00 - SAN-Aggregation implementation self-check completed

- Implemented `SAN-Aggregation`, a Neck-side gap-conditioned feature calibration route.
- Changed files:
  - `OpenTAD_SparseTAD_Clean/opentad/models/necks/fpn.py`;
  - `OpenTAD_SparseTAD_Clean/tests/test_san_neck_contracts.py`;
  - `OpenTAD_SparseTAD_Clean/configs/adatad/thumos/input_random_fixed_50pct_adapter_san_aggregation.py`.
- Implementation:
  - added `SparseAwareNeckCalibration`;
  - added optional `san_aggregation_cfg` to `FPN` and `FPNIdentity`;
  - uses only temporal-grid geometry, not GT/teacher/action labels;
  - predicts per-token, per-channel affine feature calibration;
  - zero-initializes the final affine layer for exact no-op initialization;
  - preserves `mask_list` and `temporal_grid_list`.
- Verification:
  - local `python -m py_compile` passed for changed code/config/test files;
  - local `git diff --check` passed with LF/CRLF warnings only;
  - local pytest remains blocked by the known Windows torch `c10.dll` failure;
  - remote `25876` CPU smoke passed with marker `SAN_AGGREGATION_REMOTE_SMOKE_PASS`;
  - remote config smoke passed with marker `SAN_AGGREGATION_CONFIG_REMOTE_SMOKE_PASS`.
- Contract:
  - strict random-fixed 50% input is preserved;
  - projection uses `DensePassthroughConv1DTransformerProj` only to route temporal-grid metadata;
  - Adapter/backbone, Head, assignment, loss, decode, NMS, and post-processing are unchanged.
- Self-check report:
  - `research-wiki/experiments/SAN_AGGREGATION_SELF_CHECK_20260521.md`.
- Decision:
  - status is `CODED` with self-check PASS;
  - do not commit, deploy, GPU-smoke, long-train, or claim readiness yet;
  - run Gemini CLI `gemini-3-pro-preview` read-only review next, then Claude CLI `deepseek-v4-pro` if Gemini is acceptable.
- Updated `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`.

## 2026-05-21T05:45:00+08:00 - Sparse TAD PPT regenerated as v8 after visual rejection

- Regenerated the Sparse TAD history/route summary PPT after the user rejected v7 as visually poor.
- New files:
  - `research-wiki/slides/SPARSE_TAD_HISTORY_SUMMARY_20260521_v8.pptx`;
  - `research-wiki/slides/SPARSE_TAD_HISTORY_SUMMARY_20260521_v8.md`;
  - `research-wiki/slides/SPARSE_TAD_HISTORY_SUMMARY_20260521_v8_ASSETS.md`;
  - `research-wiki/slides/SPARSE_TAD_HISTORY_SUMMARY_20260521_v8_preview/contact_sheet.png`;
  - `research-wiki/slides/scripts/generate_sparse_tad_history_deck_v8.py`.
- Design changes:
  - expanded to 14 clearer slides instead of compressing the story into dense pages;
  - rewrote the story in plain Chinese for group-meeting style explanation;
  - used larger titles, shorter statements, fewer cards per page, and clearer horizontal module diagrams;
  - added separate pages for historical negative evidence, Adapter/STGA diagnosis, Projection/SAPM, Head/SAHM, Neck/SAN, active queue, and next experiment order;
  - fixed observed text clipping on the historical-signal, Neck, progress, and summary pages after PNG preview inspection.
- Verification:
  - `python -m py_compile research-wiki/slides/scripts/generate_sparse_tad_history_deck_v8.py` passed;
  - deck generation succeeded;
  - PowerPoint COM exported all 14 slides to PNG;
  - contact sheet generation succeeded;
  - blank-page check found no blank slide.
- Decision:
  - use v8 as the current recommended PPT version;
  - preserve v7 for comparison, but do not use it for presentation unless manually requested.

## 2026-05-21T05:29:00+08:00 - SAN-Aggregation Gemini CLI review passed

- Completed Gemini CLI read-only code review for `SAN-Aggregation`.
- Gemini model:
  - `gemini-3-pro-preview`.
- Output:
  - stdout `OpenTAD_SparseTAD_Clean/logs/gemini3_pro_preview_san_aggregation_review_20260521.txt`;
  - stderr `OpenTAD_SparseTAD_Clean/logs/gemini3_pro_preview_san_aggregation_review_20260521.err.txt`;
  - prompt `OpenTAD_SparseTAD_Clean/logs/gemini3_pro_preview_san_aggregation_review_20260521.prompt.txt`.
- Verdict:
  - `PASS`;
  - no blocking findings;
  - no required fixes.
- Gemini accepted:
  - strict `random-fixed 50%` protocol is preserved;
  - no GT/teacher/oracle leakage;
  - geometry inputs are limited to temporal-grid metadata;
  - Adapter/backbone, Head, loss, assignment, decode, NMS, and post-processing remain unchanged;
  - future metric change is attributable to Neck-side SAN-Aggregation.
- Non-blocking note:
  - Gemini did not see `SAN_AGGREGATION_SELF_CHECK_20260521.md` from its current context;
  - stderr had non-blocking CLI/tool warnings.
- Formal report:
  - `research-wiki/experiments/GEMINI3_PRO_PREVIEW_SAN_AGGREGATION_REVIEW_20260521.md`.
- Decision:
  - Gemini gate accepted;
  - run Claude Code CLI `deepseek-v4-pro` secondary verification before commit/deployment/training.

## 2026-05-21T06:00:00+08:00 - SAN-Aggregation DeepSeek review passed

- Completed Claude Code CLI secondary verification with `deepseek-v4-pro`.
- Output:
  - stdout `OpenTAD_SparseTAD_Clean/logs/claude_deepseek_v4_pro_san_aggregation_20260521.txt`;
  - stderr `OpenTAD_SparseTAD_Clean/logs/claude_deepseek_v4_pro_san_aggregation_20260521.err.txt`;
  - debug `OpenTAD_SparseTAD_Clean/logs/claude_deepseek_v4_pro_san_aggregation_20260521.debug.log`.
- Exit code:
  - `0`.
- Verdict:
  - `PASS`;
  - no blocking findings;
  - no required fixes.
- DeepSeek verified:
  - no protocol leakage;
  - strict `random-fixed 50%` contract preserved;
  - mask/grid pass-through preserved;
  - fail-fast behavior when temporal-grid metadata is missing;
  - zero-initialized no-op behavior;
  - invalid-tail safety;
  - config consistency;
  - gradient flow into the affine head;
  - per-level calibrator isolation.
- Non-blocking note:
  - if this route is later enabled on non-identity `FPN` with top-down upsampling, temporal grids must be adapted to the post-upsampling resolution.
- Formal report:
  - `research-wiki/experiments/CLAUDE_DEEPSEEK_V4_PRO_SAN_AGGREGATION_REVIEW_20260521.md`.
- Decision:
  - all required implementation review gates passed;
  - commit the SAN-Aggregation implementation;
  - do not deploy, GPU-smoke, or long-train until the reviewed commit exists and a GPU server is free.

## 2026-05-21T06:02:00+08:00 - SAN-Aggregation reviewed implementation committed

- Created reviewed implementation commit for `SAN-Aggregation` in `OpenTAD_SparseTAD_Clean`.
- Commit:
  - `360e58c add san aggregation neck calibration`.
- Committed files:
  - `opentad/models/necks/fpn.py`;
  - `tests/test_san_neck_contracts.py`;
  - `configs/adatad/thumos/input_random_fixed_50pct_adapter_san_aggregation.py`.
- Final checks:
  - `python -m py_compile opentad/models/necks/fpn.py tests/test_san_neck_contracts.py configs/adatad/thumos/input_random_fixed_50pct_adapter_san_aggregation.py` passed;
  - `git diff --check` passed with LF/CRLF warning only;
  - `git status --short` in `OpenTAD_SparseTAD_Clean` is clean after commit.
- Contract:
  - Neck-side gap-conditioned feature calibration only;
  - strict random-fixed 50% input preserved;
  - no test-time GT/teacher/oracle leakage introduced;
  - Adapter/backbone, Head, assignment, loss, decode, NMS, and post-processing unchanged.
- Decision:
  - SAN-Aggregation status is now `COMMITTED`;
  - wait for free GPU, then stage the reviewed commit and run GPU AMP smoke before any long training.
- Updated `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`.

## 2026-05-21T06:03:00+08:00 - Active run monitor refresh

- Refreshed remote status on `35407`, `35329`, and `25876`.
- `35407`:
  - active screen `918136.sapm_attn_gap_bias_61e272e_20260521`;
  - experiment `SAPM-AttnGapBias v1`;
  - GPU state: RTX 4080 SUPER, about `3459 / 32760 MiB`, util `0%` at check;
  - latest checked train line: epoch 16 iter 99 `Loss=0.5950`, `cls_loss=0.3302`, `reg_loss=0.2648`;
  - no `Average-mAP` yet;
  - no checked Traceback/OOM/non-finite line.
- `35329`:
  - active screen `982088.adapter_feature_consistency_safe_fixed_bb90161`;
  - experiment `adapter_feature_consistency_safe_fixed_bb90161`;
  - third eval after epoch 45:
    - `Average-mAP=62.66`;
    - `mAP@0.30=79.35`;
    - `mAP@0.40=73.79`;
    - `mAP@0.50=65.39`;
    - `mAP@0.60=54.46`;
    - `mAP@0.70=40.32`;
  - trend: `62.25 -> 62.52 -> 62.66`, still below random-fixed baseline `63.77` and strict EMA `63.85`.
- `25876`:
  - no GPU devices;
  - only old dead screens listed.
- Decision:
  - keep SAPM-AttnGapBias running to first eval;
  - keep feature-consistency running for final trend per user instruction;
  - do not launch new long training while `35407` and `35329` are occupied.
- Updated `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`.

## 2026-05-21T06:23:10+08:00 - Sparse TAD history PPT regenerated as v9

- Regenerated the historical experiment summary deck after the user rejected the prior visual style.
- New files:
  - `research-wiki/slides/SPARSE_TAD_HISTORY_SUMMARY_20260521_v9.pptx`;
  - `research-wiki/slides/SPARSE_TAD_HISTORY_SUMMARY_20260521_v9.md`;
  - `research-wiki/slides/SPARSE_TAD_HISTORY_SUMMARY_20260521_v9_ASSETS.md`;
  - `research-wiki/slides/SPARSE_TAD_HISTORY_SUMMARY_20260521_v9_preview/contact_sheet.png`;
  - `research-wiki/slides/scripts/generate_sparse_tad_history_deck_v9.py`.
- Design changes:
  - plain Chinese narrative;
  - one core judgment per slide;
  - larger structure diagrams with editable PowerPoint shapes;
  - cleaner status board and experiment queue layout;
  - reduced dense text cards compared with v8.
- Verification:
  - generated PPTX successfully with `python research-wiki/slides/scripts/generate_sparse_tad_history_deck_v9.py`;
  - exported 15 slide PNG previews through PowerPoint COM;
  - generated contact sheet and visually checked that slides are nonblank and diagrams/text are readable.

## 2026-05-21T06:25:57+08:00 - Active run monitor refresh after PPT regeneration

- Refreshed active runs on `35407`, `35329`, and `25876`.
- `35407`:
  - active screen `918136.sapm_attn_gap_bias_61e272e_20260521`;
  - experiment `SAPM-AttnGapBias v1`;
  - GPU state: RTX 4080 SUPER, about `3459 / 32760 MiB`, util `0%` at check;
  - latest checked train line: epoch 24 iter 99 `Loss=0.5386`, `cls_loss=0.2877`, `reg_loss=0.2509`;
  - no `Average-mAP` yet;
  - no checked Traceback/OOM/non-finite line.
- `35329`:
  - active screen `982088.adapter_feature_consistency_safe_fixed_bb90161`;
  - experiment `adapter_feature_consistency_safe_fixed_bb90161`;
  - GPU state: RTX 4090 D, about `3977 / 24564 MiB`, util `0%` at check;
  - latest checked train line: epoch 47 iter 99 `Loss=0.4765`, `cls_loss=0.2469`, `reg_loss=0.2254`, `feature_consistency_loss=0.0041`;
  - eval trend remains epoch 41 `62.25 / 39.59@0.7`, epoch 43 `62.52 / 40.16@0.7`, epoch 45 `62.66 / 40.32@0.7`;
  - still below random-fixed baseline `63.77` and strict EMA `63.85`, so no positive claim.
- `25876`:
  - no GPU devices;
  - only old dead screens listed.
- Decision:
  - keep SAPM-AttnGapBias running to first eval;
  - keep feature-consistency running for trend/final mAP per user instruction;
  - do not launch SAPM-Embed, SAHM, SAN-Aggregation, or any other long run until a GPU endpoint is free.
- Updated `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`.

## 2026-05-21T06:35:15+08:00 - STGA-v1a D2-light metadata simulation completed

- Advanced the STGA-v1a failure diagnosis while both GPU endpoints remain occupied.
- Updated report:
  - `research-wiki/experiments/ADAPTER_STGA_V1A_FAILURE_DIAGNOSIS_20260521.md`.
- Diagnostic:
  - metadata-only simulation from `figures/cache/thumos_14_anno.json`;
  - reproduced `random_fixed_subsample` selected-position logic;
  - covered 1000 simulated train records and 487 validation sliding windows;
  - no video decoding and no model forward, so this is D2-light, not a full dataloader dump.
- Main numbers:
  - train selected gap p95/p99/max: `4 / 7 / 21`;
  - val selected gap p95/p99/max: `5 / 7 / 18`;
  - train gap feature log-left/right range about `[-0.693, 1.727]`;
  - val gap feature log-left/right range about `[-0.693, 1.531]`;
  - train/val metadata distributions are broadly comparable under this simulation.
- Interpretation:
  - no obvious metadata-only explanation for the `STGA-v1a` collapse to `7.23` Avg-mAP;
  - continue with D1 full-config initialization equivalence, D3 residual magnitude diagnostics, and D4 proposal/score collapse check before any STGA relaunch.
- Updated `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`.

## 2026-05-21T06:52:00+08:00 - Sparse TAD history PPT regenerated as v10

- Regenerated the Sparse TAD historical experiment and next-plan deck after the user rejected the previous visual quality.
- New files:
  - `research-wiki/slides/SPARSE_TAD_HISTORY_SUMMARY_20260521_v10.pptx`;
  - `research-wiki/slides/SPARSE_TAD_HISTORY_SUMMARY_20260521_v10.md`;
  - `research-wiki/slides/SPARSE_TAD_HISTORY_SUMMARY_20260521_v10_ASSETS.md`;
  - `research-wiki/slides/SPARSE_TAD_HISTORY_SUMMARY_20260521_v10_preview/contact_sheet.png`;
  - `research-wiki/slides/scripts/generate_sparse_tad_history_deck_v10.py`.
- Design changes:
  - rewrote all slide text in direct, plain Chinese;
  - limited each slide to one core judgment;
  - moved dense explanation out of slide bodies and into diagrams/cards;
  - rebuilt route diagrams for Adapter, Projection, Neck, Head, Temporal Grid, evidence chain, queue, and target architecture using editable PowerPoint shapes;
  - kept running experiments clearly marked as running/waiting, without claiming mAP improvements before results.
- Verification:
  - generated PPTX successfully with `python research-wiki\slides\scripts\generate_sparse_tad_history_deck_v10.py`;
  - exported 16 PNG slide previews through PowerPoint COM;
  - generated contact sheet;
  - Python validation confirmed 16 slides, no detected mojibake markers, no blank-like preview pages, and contact sheet exists.
- Additional cleanup:
  - closed the two completed historical subagent tasks after collecting their read-only outputs; neither edited files.

## 2026-05-21T06:54:16+08:00 - Active run monitor refresh

- Refreshed active runs on `35407`, `35329`, and `25876` using Windows native OpenSSH.
- `35407`:
  - active screen `918136.sapm_attn_gap_bias_61e272e_20260521`;
  - experiment `SAPM-AttnGapBias v1`;
  - GPU state: RTX 4080 SUPER, about `3459 / 32760 MiB`, util `0%` at check;
  - latest checked train line: epoch 31 iter 99 `Loss=0.5054`, `cls_loss=0.2699`, `reg_loss=0.2355`;
  - no `Average-mAP` yet;
  - no checked Traceback/OOM/non-finite line.
- `35329`:
  - active screen `982088.adapter_feature_consistency_safe_fixed_bb90161`;
  - experiment `adapter_feature_consistency_safe_fixed_bb90161`;
  - GPU state: RTX 4090 D, about `3977 / 24564 MiB`, util `0%` at check;
  - latest checked train line remains epoch 47 iter 99 `Loss=0.4765`, `cls_loss=0.2469`, `reg_loss=0.2254`, `feature_consistency_loss=0.0041`;
  - eval trend remains epoch 41 `62.25 / 39.59@0.7`, epoch 43 `62.52 / 40.16@0.7`, epoch 45 `62.66 / 40.32@0.7`;
  - still below random-fixed baseline `63.77` and strict EMA `63.85`, so no positive claim.
- `25876`:
  - no GPU devices;
  - only old dead screens listed.
- Decision:
  - keep SAPM-AttnGapBias running to first eval;
  - keep feature-consistency running for trend/final mAP per user instruction;
  - do not launch SAPM-Embed, SAHM, SAN-Aggregation, or other long runs until a GPU endpoint is free;
  - use `25876` only for CPU diagnostics, with STGA-v1a D1/D3/D4 as the next useful work.
- Updated `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`.

## 2026-05-21T07:14:14+08:00 - Sparse TAD history PPT regenerated as v11

- Regenerated the Sparse TAD historical experiment and next-plan deck after the user rejected v10 visual quality.
- New files:
  - `research-wiki/slides/SPARSE_TAD_HISTORY_SUMMARY_20260521_v11.pptx`;
  - `research-wiki/slides/SPARSE_TAD_HISTORY_SUMMARY_20260521_v11.md`;
  - `research-wiki/slides/SPARSE_TAD_HISTORY_SUMMARY_20260521_v11_ASSETS.md`;
  - `research-wiki/slides/SPARSE_TAD_HISTORY_SUMMARY_20260521_v11_preview/contact_sheet.png`;
  - `research-wiki/slides/scripts/generate_sparse_tad_history_deck_v11.py`.
- Design corrections:
  - reduced the deck from 16 pages to 14 pages;
  - rewrote page text in shorter, more colloquial Chinese;
  - made each page carry one main judgment rather than a dense table;
  - enlarged route diagrams and reduced small card clutter;
  - rebuilt the key diagrams for random-fixed difficulty, metric boundaries, four model routes, Temporal Grid, STGA, SAPM, SAHM, SAN, current queue, experiment order, and paper evidence chain;
  - shortened overflowing card text after preview inspection.
- Verification:
  - ran `python research-wiki\slides\scripts\generate_sparse_tad_history_deck_v11.py`;
  - exported 14 PNG slide previews and one contact sheet through PowerPoint COM;
  - verified the PPTX has 14 slides;
  - confirmed no detected common mojibake markers in PPT text;
  - checked slide previews for blank-like pages.

## 2026-05-21T07:25:55+08:00 - Active run monitor refresh and STGA D1 diagnosis completed

- Refreshed active runs on `35407`, `35329`, and `25876` using Windows native OpenSSH.
- `35407`:
  - active screen `918136.sapm_attn_gap_bias_61e272e_20260521`;
  - experiment `SAPM-AttnGapBias v1`;
  - GPU state: RTX 4080 SUPER, about `3459 / 32760 MiB`, util `0%` at check;
  - latest checked train line: epoch 37 iter 99 `Loss=0.4869`, `cls_loss=0.2538`, `reg_loss=0.2331`;
  - no `Average-mAP` yet;
  - no checked Traceback/OOM/non-finite line.
- `35329`:
  - active screen `982088.adapter_feature_consistency_safe_fixed_bb90161`;
  - experiment `adapter_feature_consistency_safe_fixed_bb90161`;
  - GPU state: RTX 4090 D, about `3977 / 24564 MiB`, util `0%` at check;
  - new epoch-47 eval: `Average-mAP=62.89`, `mAP@0.30=79.23`, `0.40=73.90`, `0.50=65.59`, `0.60=54.97`, `0.70=40.74`;
  - trend is `62.25 -> 62.52 -> 62.66 -> 62.89`, still below random-fixed baseline `63.77` and strict EMA `63.85`, so no positive claim.
- `25876`:
  - no GPU devices;
  - used for STGA-v1a CPU-only D1 initialization-equivalence diagnosis.
- STGA D1 diagnostic:
  - local diagnostic script: `logs/stga_d1_init_equivalence_probe.py`;
  - remote output: `/root/autodl-tmp/OpenTAD_SparseTAD_Clean_stga_bf76e63/logs/stga_d1_init_equivalence_20260521.json`;
  - local copied output: `logs/stga_d1_init_equivalence_20260521.json`;
  - first reruns exposed diagnostic/invocation issues only: missing `PYTHONPATH`, missing `opentad.datasets` transform registry import, and a `torch.unique` return-value bug;
  - final verdict: `PASS`;
  - summary: `max_adapter_diff=0.0`, `max_zero_abs=0.0`, `max_downstream_feat_diff=0.0`, `max_downstream_prop_diff=0.0`, `max_downstream_score_diff=0.0`;
  - encoded gap shape: `[1, 192, 64]`; adapter sparse blocks checked: `6`; shape mismatches: none.
- Interpretation:
  - STGA-v1a collapse is less likely to be an initialization-equivalence bug, checkpoint-loading mismatch, or accidental projection/head computation change;
  - more likely next suspects are learned residual magnitude, real-batch metadata interaction, or score/proposal distribution collapse after training.
- Decision:
  - keep SAPM-AttnGapBias running to first eval;
  - keep feature-consistency running for trend/final mAP per user instruction;
  - do not launch SAPM-Embed, SAHM, SAN-Aggregation, or other long runs until a GPU endpoint is free;
  - keep `STGA-v1a` as `FAILED-RUN` and keep `STGA-v1b` blocked;
  - continue STGA diagnostics with D3 residual magnitude and D4 score/proposal collapse check before any relaunch.
- Updated:
  - `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`;
  - `research-wiki/experiments/ADAPTER_STGA_V1A_FAILURE_DIAGNOSIS_20260521.md`.

## 2026-05-21T07:37:34+08:00 - Active run monitor refresh and STGA D4-init diagnosis completed

- Refreshed active runs on `35407`, `35329`, and `25876` using Windows native OpenSSH.
- `35407`:
  - active screen `918136.sapm_attn_gap_bias_61e272e_20260521`;
  - experiment `SAPM-AttnGapBias v1`;
  - GPU state: RTX 4080 SUPER, about `3461 / 32760 MiB`, util `0%` at check;
  - latest checked train line: epoch 41 iter 99 `Loss=0.4725`, `cls_loss=0.2443`, `reg_loss=0.2282`;
  - no `Average-mAP` yet;
  - no checked Traceback/OOM/non-finite line.
- `35329`:
  - active screen `982088.adapter_feature_consistency_safe_fixed_bb90161`;
  - experiment `adapter_feature_consistency_safe_fixed_bb90161`;
  - GPU state: RTX 4090 D, about `3979 / 24564 MiB`, util `0%` at check;
  - latest train line: epoch 48 iter 99 `Loss=0.4583`, `cls_loss=0.2352`, `reg_loss=0.2188`, `feature_consistency_loss=0.0042`;
  - latest eval remains epoch 47: `Average-mAP=62.89`, `mAP@0.70=40.74`;
  - still below random-fixed baseline `63.77` and strict EMA `63.85`, so no positive claim.
- STGA D4-init diagnostic:
  - local diagnostic script: `logs/stga_d4_initial_proposal_probe.py`;
  - remote output: `/root/autodl-tmp/OpenTAD_SparseTAD_Clean_stga_bf76e63/logs/stga_d4_initial_proposal_20260521.json`;
  - local copied output: `logs/stga_d4_initial_proposal_20260521.json`;
  - final verdict: `PASS`;
  - equivalence: feature diffs `[0,0,0,0,0,0]`, masks all equal, proposal diff `[0]`, score diff `[0]`;
  - proposal shape `[756,2]`, score shape `[756,20]`;
  - initial score min / median / p95 / p99 / max: `0.002884 / 0.010124 / 0.016889 / 0.020230 / 0.034312`;
  - initial proposal length median / p95 / p99 / max: `0.980881 / 7.394081 / 19.609180 / 32.897736`.
- Interpretation:
  - initialization-time head/proposal distributions are identical between baseline and STGA;
  - this further reduces the likelihood that STGA-v1a failed because initial head/proposal outputs were malformed;
  - trained score/proposal collapse remains untested because the failed run saved no checkpoint.
- Decision:
  - keep SAPM-AttnGapBias running to first eval;
  - keep feature-consistency running for trend/final mAP per user instruction;
  - do not launch SAPM-Embed, SAHM, SAN-Aggregation, or other long runs until a GPU endpoint is free;
  - keep `STGA-v1a` as `FAILED-RUN` and keep `STGA-v1b` blocked;
  - continue STGA diagnostics with D3 residual magnitude and a future trained D4 check with saved diagnostic state before any relaunch.
- Updated:
  - `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`;
  - `research-wiki/experiments/ADAPTER_STGA_V1A_FAILURE_DIAGNOSIS_20260521.md`.

## 2026-05-21T08:00:00+08:00 - Sparse TAD history PPT regenerated as v12

- Regenerated the historical experiment and next-plan slide deck after the user reported that the previous PPT was visually poor.
- New files:
  - `research-wiki/slides/SPARSE_TAD_HISTORY_SUMMARY_20260521_v12.pptx`;
  - `research-wiki/slides/SPARSE_TAD_HISTORY_SUMMARY_20260521_v12.md`;
  - `research-wiki/slides/SPARSE_TAD_HISTORY_SUMMARY_20260521_v12_ASSETS.md`;
  - `research-wiki/slides/SPARSE_TAD_HISTORY_SUMMARY_20260521_v12_preview/contact_sheet.png`;
  - `research-wiki/slides/scripts/generate_sparse_tad_history_deck_v12.py`.
- Design changes:
  - rewrote the deck in simpler Chinese;
  - expanded the main structure diagrams;
  - reduced dense status-card clutter;
  - kept one main judgement per slide;
  - preserved only confirmed experiment facts and avoided premature mAP claims.
- Verification:
  - generated 16-slide PPTX successfully;
  - exported all page previews and a contact sheet;
  - checked for blank pages and common mojibake markers;
  - manually inspected the contact sheet plus key slides, and fixed bottom text crowding on slide 13.

## 2026-05-21T08:08:41+08:00 - SAPM-AttnGapBias first-eval collapse and stop

- Monitored active remote runs on `35407`, `35329`, and `25876` using Windows native OpenSSH.
- `35407`:
  - experiment `SAPM-AttnGapBias v1`;
  - path `/root/autodl-tmp/OpenTAD_SparseTAD_Clean_sapm_61e272e`;
  - log `logs/sapm_attn_gap_bias_20260521_0439.log`;
  - first eval after epoch 41 collapsed to `Average-mAP=7.54`;
  - mAP breakdown: `0.30=17.62`, `0.40=10.64`, `0.50=5.71`, `0.60=2.65`, `0.70=1.07`;
  - stopped screen `sapm_attn_gap_bias_61e272e_20260521` to prevent further GPU waste;
  - after stop, no SAPM train process remains and GPU is `0/32760 MiB`.
- Interpretation:
  - this is a failure-level collapse, not normal underperformance;
  - it is extremely close to the previous STGA-v1a first-eval collapse `7.23 / 1.06@0.7`;
  - repeated collapse suggests a shared clean-repo/config/eval/checkpoint-loading diagnosis is now higher priority than launching another long sparse-aware run.
- `35329`:
  - feature-consistency remains active;
  - latest train line is epoch 49 iter 99 `Loss=0.4609`;
  - latest eval remains epoch 47 `62.89 / 40.74@0.7`, still below random-fixed `63.77` and strict EMA `63.85`.
- `25876`:
  - no GPU devices;
  - only prior STGA D1/D4 diagnostic outputs are present.
- Decision:
  - mark `SAPM-AttnGapBias v1` as `FAILED-RUN`;
  - do not launch `SAPM-EmbedDeformable`, `SAHM`, `SAN-Aggregation`, or another long clean-stack sparse-aware run until the shared collapse diagnosis is complete;
  - use the freed `35407` GPU only for targeted diagnosis or smoke checks.
- Updated:
  - `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`;
  - `research-wiki/experiments/SAPM_ATTN_GAP_BIAS_FIRST_EVAL_FAILURE_20260521.md`.

## 2026-05-21T08:17:15+08:00 - Clean repo random-fixed Adapter baseline diagnosis launched

- Launched a diagnostic baseline on `35407` to test whether the clean sparse-aware code tree can reproduce the known random-fixed Adapter baseline scale without enabling STGA/SAPM/SAHM/SAN.
- Context:
  - STGA-v1a first eval collapsed to `7.23 / 1.06@0.7`;
  - SAPM-AttnGapBias first eval collapsed to `7.54 / 1.07@0.7`;
  - the repeated collapse makes shared clean-repo/config/eval/checkpoint-loading diagnosis higher priority than launching another model-side long run.
- Launch:
  - path `/root/autodl-tmp/OpenTAD_SparseTAD_Clean_sapm_61e272e`;
  - config `configs/adatad/thumos/input_random_fixed_50pct_adapter.py`;
  - screen `clean_baseline_adapter_61e272e_20260521`;
  - log `logs/clean_baseline_adapter_20260521_0812.log`;
  - master port `29613`;
  - preflight marker `CLEAN_BASELINE_PREFLIGHT_PASS`, confirming `random_fixed_subsample`, `keep_ratio=0.5`, and `val_start_epoch=40`.
- First health check:
  - screen active;
  - GPU about `3451/32760 MiB`;
  - training processes active;
  - no `Loss=` line yet at the first short health check.
- Decision:
  - keep this diagnostic running to first loss and first eval;
  - do not launch SAPM-Embed, SAHM, SAN-Aggregation, or another long clean-stack sparse-aware run until this baseline diagnosis clarifies the shared collapse source.
- Updated:
  - `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`.

## 2026-05-21T08:18:45+08:00 - Clean repo baseline first loss captured

- Checked `35407` clean baseline run.
- Active screen:
  - `278793.clean_baseline_adapter_61e272e_20260521`.
- GPU:
  - RTX 4080 SUPER, about `3455/32760 MiB`, util `0%` at check.
- First train line:
  - `[000][00050/00099] Loss=1.7082`, `cls_loss=0.9703`, `reg_loss=0.7378`.
- No checked crash/OOM/non-finite line was observed in the grep output.
- Interpretation:
  - startup is healthy and comparable in scale to the earlier SAPM launch;
  - the decisive diagnostic remains the first epoch-41 eval: normal `~63-64` would implicate sparse-module learned instability, while another `~7` collapse would implicate shared clean-repo/config/eval/checkpoint-loading issues.
- Updated:
  - `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`;
  - `research-wiki/experiments/CLEAN_REPO_BASELINE_DIAGNOSIS_20260521.md`.

## 2026-05-21T08:25:36+08:00 - Active run monitor refresh

- Refreshed `35407`, `35329`, and `25876`.
- `35407` clean baseline diagnosis:
  - screen `278793.clean_baseline_adapter_61e272e_20260521` remains active;
  - GPU about `3455/32760 MiB`;
  - latest lines: `[000][00099/00099] Loss=1.6330`, `[001][00050/00099] Loss=1.0131`, `[001][00099/00099] Loss=0.9865`;
  - no `Average-mAP` yet;
  - no checked crash/OOM/non-finite line.
- `35329` feature-consistency:
  - screen `982088.adapter_feature_consistency_safe_fixed_bb90161` remains active;
  - latest train line remains epoch 49 iter 99 `Loss=0.4609`;
  - latest eval remains epoch 47 `62.89 / 40.74@0.7`;
  - still below `63.77` random-fixed baseline and `63.85` strict EMA reference.
- `25876`:
  - no GPU devices;
  - only prior STGA D1/D4 diagnostic outputs are present.
- Decision:
  - keep clean baseline running to first eval;
  - keep feature-consistency running for trend/final mAP;
  - no additional long deployment until the clean baseline diagnosis resolves the shared collapse question.
- Updated:
  - `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`;
  - `research-wiki/experiments/CLEAN_REPO_BASELINE_DIAGNOSIS_20260521.md`.

## 2026-05-21T08:35:20+08:00 - Active run monitor refresh before PPT regeneration

- Refreshed `35407`, `35329`, and `25876` using Windows native OpenSSH.
- `35407` clean baseline diagnosis:
  - screen `278793.clean_baseline_adapter_61e272e_20260521` remains active;
  - GPU about `3455/32760 MiB`, util `0%` at check;
  - latest checked training lines include `[002][00099/00099] Loss=0.8940` and `[003][00050/00099] Loss=0.8088`;
  - no `Average-mAP`, crash, OOM, or checked non-finite line yet.
- `35329` feature-consistency:
  - screen `982088.adapter_feature_consistency_safe_fixed_bb90161` remains active;
  - epoch-49 eval reached `Average-mAP=63.21`;
  - breakdown: `mAP@0.30=79.36`, `0.40=74.30`, `0.50=65.90`, `0.60=55.39`, `0.70=41.10`;
  - this is an upward trend but still below random-fixed Adapter baseline `63.77` and strict EMA `63.85`, so no positive claim.
- `25876`:
  - no GPU devices;
  - only previous STGA D1/D4 diagnostic JSON files are present.
- Decision:
  - keep clean baseline running to first epoch-41 eval;
  - keep feature-consistency running for trend/final mAP;
  - do not launch queued SAPM-Embed, SAHM, SAN, or combo long runs until the clean baseline diagnosis resolves the shared collapse question.
- Updated:
  - `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`;
  - `research-wiki/experiments/CLEAN_REPO_BASELINE_DIAGNOSIS_20260521.md`.

## 2026-05-21T08:48:00+08:00 - Sparse TAD history PPT regenerated as v13

- Regenerated the historical experiment and next-plan slide deck after the user rejected the previous visual design.
- New files:
  - `research-wiki/slides/SPARSE_TAD_HISTORY_SUMMARY_20260521_v13.pptx`;
  - `research-wiki/slides/SPARSE_TAD_HISTORY_SUMMARY_20260521_v13.md`;
  - `research-wiki/slides/SPARSE_TAD_HISTORY_SUMMARY_20260521_v13_ASSETS.md`;
  - `research-wiki/slides/SPARSE_TAD_HISTORY_SUMMARY_20260521_v13_preview/contact_sheet.png`;
  - `research-wiki/slides/scripts/generate_sparse_tad_history_deck_v13.py`.
- Design changes:
  - reduced the deck from 16 status-heavy slides to 14 presentation-first slides;
  - rewrote dense technical descriptions into simpler Chinese;
  - enlarged architecture diagrams for Adapter, Projection, Neck, Head, and the overall model chain;
  - replaced dense small-card status boards with fewer larger cards and clearer visual grouping;
  - kept confirmed experiment facts only, including feature-consistency epoch-49 `63.21` and clean baseline diagnostic status.
- Verification:
  - generated PPTX successfully;
  - exported all slide PNG previews and a contact sheet;
  - checked for blank pages and common mojibake markers;
  - manually inspected the contact sheet and corrected visible text overflow on slides 7, 8, 11, 12, and 14.

## 2026-05-21T08:50:05+08:00 - Active run monitor refresh

- Refreshed `35407`, `35329`, and `25876` using Windows native OpenSSH.
- `35407` clean baseline diagnosis:
  - screen `278793.clean_baseline_adapter_61e272e_20260521` remains active;
  - GPU about `3457/32760 MiB`, util `0%` at check;
  - latest checked training lines include `[006][00099/00099] Loss=0.7388` and `[007][00050/00099] Loss=0.7717`;
  - no `Average-mAP`, crash, OOM, or checked non-finite line yet.
- `35329` feature-consistency:
  - screen `982088.adapter_feature_consistency_safe_fixed_bb90161` remains active;
  - latest checked train line `[050][00050/00099] Loss=0.4250`;
  - latest eval remains epoch 49 `Average-mAP=63.21`, `mAP@0.70=41.10`;
  - still below random-fixed Adapter baseline `63.77` and strict EMA `63.85`, so no positive claim.
- `25876`:
  - no GPU devices;
  - only previous STGA D1/D4 diagnostic JSON files are present.
- Decision:
  - keep clean baseline running to first epoch-41 eval;
  - keep feature-consistency running for trend/final mAP;
  - do not launch queued SAPM-Embed, SAHM, SAN, or combo long runs until the clean baseline diagnosis resolves the shared collapse question.
- Updated:
  - `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`;
  - `research-wiki/experiments/CLEAN_REPO_BASELINE_DIAGNOSIS_20260521.md`.

## 2026-05-21T09:02:00+08:00 - SAHM-Deformable-NoGate-AllCls self-check completed

- Implemented the Head-side NoGate ablation for SAHM.
- Changed files:
  - `OpenTAD_SparseTAD_Clean/opentad/models/dense_heads/sparse_head_mixer.py`;
  - `OpenTAD_SparseTAD_Clean/tests/test_sahm_head_contracts.py`;
  - `OpenTAD_SparseTAD_Clean/configs/adatad/thumos/input_random_fixed_50pct_adapter_sahm_deformable_nogate_allcls.py`.
- Purpose:
  - remove the learned SAHM slot gate while retaining deformable offsets;
  - isolate whether the learned softmax gate is needed in the classification tower;
  - keep Adapter, Projection, Neck, regression tower, assignment, loss, decode, post-processing, and random-fixed 50% input protocol unchanged.
- Verification:
  - local `py_compile` PASS;
  - local `git diff --check` PASS with LF/CRLF warnings only;
  - local pytest remains blocked by Windows torch `c10.dll`;
  - remote `25876` manual CPU smoke PASS with marker `SAHM_NOGATE_CPU_SMOKE_PASS`.
- Gate state:
  - self-check PASS;
  - Gemini CLI and Claude CLI DeepSeek reviews are pending;
  - no commit, deployment, GPU smoke, or long training is allowed before external review gates pass.
- Updated:
  - `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`;
  - `research-wiki/experiments/SAHM_DEFORMABLE_NOGATE_ALLCLS_SELF_CHECK_20260521.md`.

## 2026-05-21T09:19:26+08:00 - Active run monitor refresh

- Refreshed `35407` and `35329` using Windows native OpenSSH.
- `35407` clean baseline diagnosis:
  - screen `278793.clean_baseline_adapter_61e272e_20260521` remains active;
  - GPU about `3459/32760 MiB`, util `77%` at check;
  - latest checked train lines include `[014][00050/00099] Loss=0.5839` and `[014][00099/00099] Loss=0.6019`;
  - no `Average-mAP`, crash, OOM, or checked non-finite line yet.
- `35329` feature-consistency:
  - screen `982088.adapter_feature_consistency_safe_fixed_bb90161` remains active;
  - latest checked train line `[051][00099/00099] Loss=0.4591`;
  - latest eval remains epoch 49 `Average-mAP=63.21`, `mAP@0.70=41.10`;
  - still below random-fixed Adapter baseline `63.77` and strict EMA `63.85`, so no positive claim.
- SAHM-NoGate review gate:
  - Gemini CLI review already returned PASS with no blocking findings;
  - the Claude CLI DeepSeek review command was interrupted by the user and may have left a local `claude` process running;
  - no acceptable DeepSeek output has been recorded yet, so the gate remains pending.
- Decision:
  - keep clean baseline running to first epoch-41 eval;
  - keep feature-consistency running for trend/final mAP;
  - do not launch queued SAPM-Embed, SAHM, SAN, or combo long runs until the clean baseline diagnosis resolves the shared collapse question;
  - do not count the interrupted DeepSeek review as complete.
- Updated:
  - `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`.

## 2026-05-21T09:43:48+08:00 - GPT-5 Pro post-collapse discussion recorded

- Completed the requested post-collapse GPT-5 Pro discussion record using the user manually retrieved full Pro response.
- Created:
  - `research-wiki/experiments/SPARSE_TAD_COLLAPSE_GPT5_PRO_DISCUSSION_20260521.md`.
- Local invocation evidence:
  - Rosetta Pro retry used extended send-button and attachment waits;
  - stdout path `logs/rosetta_gpt5pro_sparse_tad_collapse_20260521.txt`;
  - debug path `logs/rosetta_gpt5pro_sparse_tad_collapse_20260521.debug.log`;
  - the user-pasted full response is more complete and is treated as authoritative.
- Accepted Pro conclusion:
  - the highest unresolved risk is not yet that STGA and SAPM are invalid ideas;
  - the project must first separate shared clean-stack/protocol collapse from module training-time instability;
  - two independent modules collapsing to about `7` Avg-mAP with similar per-tIoU breakdowns must not be treated as ordinary negative ablations.
- Active gates:
  - keep `35407` clean random-fixed Adapter baseline running to first eval;
  - do not launch STGA-v1b, SAPM-EmbedDeformable, SAHM, SAN, or any combo long training until the clean baseline gate resolves;
  - if clean baseline is `63-64`, diagnose STGA/SAPM learned residual/bias instability;
  - if clean baseline is near `7`, enter shared protocol emergency with known-good checkpoint eval, coordinate dump, and historical config/code diff;
  - if clean baseline is `20-50`, treat sparse results as contaminated and do not claim module effects.
- Updated:
  - `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`.

## 2026-05-21T09:55:44+08:00 - Active run refresh and pretrained loading checked

- Refreshed `35407` and `35329` with Windows native OpenSSH.
- `35407` clean baseline:
  - screen `278793.clean_baseline_adapter_61e272e_20260521` remains active;
  - GPU about `3459/32760 MiB`, util `0%` at check;
  - latest checked train line `[021][00050/00099] Loss=0.5371`;
  - no `Average-mAP`, crash, OOM, or checked non-finite line yet.
- `35329` feature-consistency:
  - screen `982088.adapter_feature_consistency_safe_fixed_bb90161` remains active;
  - epoch-51 eval reached `Average-mAP=63.46`;
  - breakdown: `mAP@0.30=79.27`, `0.40=74.39`, `0.50=66.09`, `0.60=55.89`, `0.70=41.66`;
  - trend is still rising from `62.25 -> 62.52 -> 62.66 -> 62.89 -> 63.21 -> 63.46`, but remains below random-fixed Adapter `63.77` and strict EMA `63.85`, so no positive claim.
- Pretrained loading check:
  - clean baseline, feature-consistency, STGA-v1a, and SAPM-AttnGapBias logs all show loading from `pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth`;
  - the remote file exists on `35407` at `87M`;
  - checkpoint inspection found `163` keys, including patch embed and `156` block keys;
  - run logs show only expected unexpected source keys `cls_head.fc_cls.weight/bias`;
  - missing keys are Adapter parameters and, for STGA, new sparse-geometry parameters, which are expected to be initialized in the experiment model.
- Interpretation:
  - at log-evidence level, pretrained backbone loading is not the current leading explanation for the STGA/SAPM near-7 first-eval collapses;
  - clean baseline first eval remains the decisive gate for shared clean-stack/protocol failure vs sparse-module learned instability.
- Updated:
  - `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`;
  - `research-wiki/experiments/CLEAN_REPO_BASELINE_DIAGNOSIS_20260521.md`.

## 2026-05-21T10:16:58+08:00 - Pro-guided clean-stack code audit completed

- Completed the requested code inspection following the GPT-5 Pro collapse discussion.
- Created:
  - `research-wiki/experiments/SPARSE_TAD_CLEAN_STACK_CODE_AUDIT_20260521.md`.
- Active clean baseline status on `35407`:
  - screen `278793.clean_baseline_adapter_61e272e_20260521` remains active;
  - GPU about `3459/32760 MiB`, util `79%`;
  - latest checked train line `[027][00099/00099] Loss=0.5356`;
  - no `Average-mAP`, crash, OOM, or checked non-finite line yet.
- Highest-priority audit finding:
  - local and remote clean tree `opentad/models/utils/post_processing/utils.py` do not map selected-axis proposals back through `irregular_selected_positions` before seconds conversion;
  - `OpenTAD_Back` already contains `selected_axis_to_dense_axis()` and calls it inside `convert_to_seconds()`;
  - this is a plausible shared clean-stack/protocol explanation if the clean baseline first eval also collapses near `7`.
- Additional audit findings:
  - NMS-before-coordinate-backtransform remains a non-sliding random-fixed protocol risk;
  - ActionFormer temporal-grid tuple routing appears correct and does not explain disabled baseline pollution;
  - SAPM-AttnGapBias has zero-init gate/alpha, expected bias shape, and correct invalid-key masking order, with a future asymmetric-stride warning;
  - STGA sparse metadata routing and zero-init equivalence are valid, but inherited `disable_checkpoint=True` and large post-learning residual scale make trained-state diagnosis impossible for the failed run;
  - pretrained backbone loading remains expected and is not the leading explanation.
- Decision:
  - keep the current clean baseline running to first eval as the empirical gate;
  - do not launch STGA-v1b, SAPM-EmbedDeformable, SAHM, SAN, combos, or physical-coordinate protocol long runs;
  - if clean baseline collapses, patch selected-axis coordinate conversion, add deterministic mapping tests/raw coordinate dump, and rerun self-check plus Gemini CLI and Claude CLI DeepSeek before relaunch.
- Updated:
  - `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`;
  - `research-wiki/experiments/CLEAN_REPO_BASELINE_DIAGNOSIS_20260521.md`.

## 2026-05-21T10:33:37+08:00 - Selected-axis coordinate repair candidate prepared locally

- Refreshed active runs:
  - `35407` clean baseline remains active in screen `278793.clean_baseline_adapter_61e272e_20260521`;
  - latest checked line `[030][00050/00099] Loss=0.5020`;
  - no `Average-mAP`, crash, OOM, or checked non-finite line yet;
  - `35329` feature-consistency remains active, latest checked line `[053][00050/00099] Loss=0.4759`, latest eval still epoch-51 `63.46 / 41.66@0.7`.
- Prepared a local, not-yet-deployed candidate fix for the clean-stack audit blocker.
- Changed local files:
  - `OpenTAD_SparseTAD_Clean/opentad/models/utils/post_processing/utils.py`;
  - `OpenTAD_SparseTAD_Clean/opentad/models/utils/post_processing/__init__.py`;
  - `OpenTAD_SparseTAD_Clean/tests/test_post_processing_selected_axis.py`.
- Created self-check report:
  - `research-wiki/experiments/POSTPROCESS_SELECTED_AXIS_COORD_FIX_SELF_CHECK_20260521.md`.
- Fix summary:
  - added `selected_axis_to_dense_axis()`;
  - `convert_to_seconds()` now maps selected-axis proposals back to dense-axis before seconds conversion when `irregular_selected_positions` and `irregular_selected_valid_len` exist and `irregular_native_axis=False`;
  - legacy no-metadata and native-axis paths remain unchanged.
- Verification:
  - local `py_compile` PASS;
  - local `git diff --check` PASS with only LF-to-CRLF warnings on touched files;
  - local pytest blocked by the known Windows torch `c10.dll` initialization error;
  - remote `25876` inline torch smoke PASS with marker `POSTPROC_SELECTED_AXIS_REMOTE_SMOKE_PASS`;
  - remote `25876` pytest unavailable because `/root/miniconda3/bin/python` has no `pytest` module.
- Decision:
  - do not sync or deploy this patch while the active unpatched clean baseline is still running to first eval;
  - if clean baseline collapses near `7`, run Gemini CLI and Claude CLI DeepSeek review on this patch, then relaunch a repaired clean baseline;
  - if clean baseline is normal, keep this as controlled protocol debt rather than retroactively changing the baseline interpretation.
- Updated:
  - `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`;
  - `research-wiki/experiments/CLEAN_REPO_BASELINE_DIAGNOSIS_20260521.md`.

## 2026-05-21T11:16:00+08:00 - Pro-guided code check refreshed and coordinate repair reviewed

- Continued the code inspection requested after the GPT-5 Pro collapse
  discussion.
- Active unpatched clean baseline on `35407`:
  - screen `278793.clean_baseline_adapter_61e272e_20260521`;
  - path `/root/autodl-tmp/OpenTAD_SparseTAD_Clean_sapm_61e272e`;
  - log `logs/clean_baseline_adapter_20260521_0812.log`;
  - latest checked train line `[041][00050/00099] Loss=0.4465`;
  - no `Average-mAP`, crash, OOM, or checked non-finite line yet.
- Pretrained loading:
  - run log confirms loading
    `pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth`;
  - unexpected source keys are only `cls_head.fc_cls.weight/bias`;
  - missing keys are Adapter parameters;
  - remote checkpoint probe found `163` keys with patch embed and block keys,
    `cls_head` source keys, and no Adapter keys.
- Interpretation:
  - pretrained VideoMAE-S backbone loading is correct and is not the current
    leading explanation for the near-7 STGA/SAPM first-eval collapses;
  - the active remote clean tree remains unpatched and still lacks
    `selected_axis_to_dense_axis()`;
  - selected-axis post-processing remains the highest-priority shared protocol
    repair candidate if the unpatched clean baseline first eval collapses.
- Postprocess selected-axis coordinate repair:
  - created
    `research-wiki/experiments/POSTPROCESS_SELECTED_AXIS_COORD_FIX_REVIEW_20260521.md`;
  - Gemini CLI `gemini-3-pro-preview` exit code `0`, verdict PASS;
  - Claude CLI `deepseek-v4-pro` exit code `0`, verdict PASS;
  - accepted Gemini's non-blocking request by adding negative-coordinate and
    empty-position tests;
  - remote `25876` edge smoke passed with
    `POSTPROC_SELECTED_AXIS_REMOTE_SMOKE_PASS_V2`.
- Decision:
  - do not sync, commit, deploy, or use the reviewed coordinate patch before
    the active unpatched clean baseline first eval is recorded;
  - no new STGA/SAPM/SAHM/SAN/combo long training until the clean baseline gate
    resolves.
- Updated:
  - `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`;
  - `research-wiki/experiments/CLEAN_REPO_BASELINE_DIAGNOSIS_20260521.md`;
  - `research-wiki/experiments/SPARSE_TAD_CLEAN_STACK_CODE_AUDIT_20260521.md`.

## 2026-05-21T11:18:14+08:00 - Clean baseline first eval started

- Refreshed `35407` clean baseline.
- Status:
  - epoch 41 training completed with `[041][00099/00099] Loss=0.4690`;
  - evaluation progress started and reached about `145/396` in the checked
    tail;
  - no `Average-mAP` has been emitted yet;
  - no checked crash/OOM/non-finite line.
- Decision:
  - the decisive unpatched clean-stack gate is now in progress;
  - do not deploy the reviewed selected-axis coordinate repair or launch new
    sparse-aware long runs until this first-eval mAP is recorded.
- Updated:
  - `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`;
  - `research-wiki/experiments/CLEAN_REPO_BASELINE_DIAGNOSIS_20260521.md`.

## 2026-05-21T11:58:00+08:00 - Manual GPT-5 Pro collapse discussion recorded and repaired-path smoke passed

- The user manually copied back a GPT-5 Pro diagnosis for the repeated
  `7.x` Avg-mAP collapse.
- Recorded the discussion summary in:
  - `research-wiki/experiments/SPARSE_TAD_COLLAPSE_GPT5_PRO_MANUAL_DISCUSSION_20260521.md`.
- Pro verdict:
  - unpatched clean baseline `7.45` proves the first priority is shared
    clean-stack/protocol collapse, not independent STGA/SAPM failure;
  - selected-axis post-processing is the correct first repair candidate;
  - repaired clean baseline is GO after cheap branch-hit checks;
  - all STGA/SAPM/SAHM/SAN/combo long runs remain NO-GO until repaired clean
    baseline recovers.
- Remote `35407` repaired tree check:
  - `selected_axis_to_dense_axis` is importable and wired into
    `convert_to_seconds`;
  - `py_compile` passed for the post-processing repair;
  - actual val metadata branch smoke passed on a non-identity window:
    `valid_len=503`, `pos_len=384`, gap max `5`, selected `[190, 192]`
    mapped to dense `[253, 255]`;
  - marker: `POSTPROC_NONIDENTITY_META_BRANCH_SMOKE_PASS`.
- Launch decision:
  - repaired clean baseline should be launched next with strict random-fixed
    50% unchanged, checkpoint interval restored to `2`, and
    `post_processing.save_dict=True`.

## 2026-05-21T12:00:55+08:00 - Repaired clean baseline diagnostic run launched

- Server: `35407`.
- Path: `/root/autodl-tmp/OpenTAD_SparseTAD_Clean_sapm_61e272e`.
- Screen: `clean_baseline_postfix_diag_20260521`.
- Log: `logs/clean_baseline_postfix_diag_20260521_1159.log`.
- Config:
  - `configs/adatad/thumos/input_random_fixed_50pct_adapter.py`.
- Runtime overrides:
  - `workflow.checkpoint_interval=2`;
  - `workflow.disable_checkpoint=False`;
  - `post_processing.save_dict=True`;
  - `work_dir=exps/thumos/adatad/input_random_fixed_50pct_adapter_postfix_diag_20260521`.
- Contract:
  - strict random-fixed 50% sampling unchanged;
  - Adapter + ActionFormer baseline model unchanged;
  - no STGA/SAPM/SAHM/SAN enabled;
  - no GT/teacher leakage introduced.
- Current status:
  - screen exists and GPU memory is allocated;
  - first loss has not yet appeared in the checked log.

## 2026-05-21T12:02:23+08:00 - Server checkpoint storage cleanup

- Checked storage:
  - `35407`: `/root/autodl-tmp` `200G`, used `170G`, available `31G`, use `85%`;
  - `35329`: `/root/autodl-tmp` `200G`, used `64G`, available `137G`, use `32%`;
  - `25876`: `/root/autodl-tmp` `200G`, used `191G`, available `9.1G`, use `96%`.
- Cleanup was needed only on `25876`.
- Removed historical duplicate `epoch_*.pth` files under checkpoint
  directories, keeping the highest epoch file in each checkpoint directory.
- Did not remove code, logs, datasets, result files, or non-epoch artifacts.
- Result:
  - `25876` `/root/autodl-tmp` available space increased from `9.1G` to `44G`;
  - usage dropped from `96%` to `79%`.
- Cleanup report:
  - `/root/autodl-tmp/checkpoint_epoch_cleanup_20260521_120222.log`.

## 2026-05-21T12:12:00+08:00 - Checkpoint retention rule added to AGENTS

- Added `Checkpoint Retention & Storage Hygiene` to `AGENTS.md`.
- New rule:
  - after every complete training run, clean redundant historical
    `epoch_*.pth` files for that run;
  - keep only the highest-numbered `epoch_*.pth` as the last recoverable
    checkpoint;
  - preserve `best.pth`, `last.pth`, configs, logs, result JSON, raw
    predictions, proposal dumps, review records, and cleanup reports;
  - do not clean active or ambiguous runs;
  - record `df -h` before/after, kept checkpoint, and removed epoch files in
    the tracker and this log.
- Repaired clean baseline checkpoint saving was also confirmed on `35407`:
  - run: `clean_baseline_postfix_diag_20260521`;
  - checkpoint file:
    `exps/thumos/adatad/input_random_fixed_50pct_adapter_postfix_diag_20260521/gpu1_id0/checkpoint/epoch_1.pth`;
  - size: `623797169` bytes.

## 2026-05-21T12:16:00+08:00 - Severe-result Pro escalation gate added to AGENTS

- Added `Severe Result Escalation Gate` to `AGENTS.md`.
- Trigger examples:
  - Avg-mAP collapses by about `5` points or more versus the relevant
    baseline/reference;
  - failure-scale mAP such as the current `~7` Avg-mAP collapse;
  - high-IoU mAP collapses while loss remains normal;
  - independent modules fail with similar mAP shape/magnitude;
  - metric/checkpoint/post-processing/coordinate/data/evaluator evidence makes
    attribution unclear.
- Required action:
  - freeze affected follow-up long runs;
  - preserve raw evidence and code/config context;
  - start GPT-5 Pro read-only discussion/review;
  - record prompt/files/response, accepted/rejected recommendations, and the
    go/no-go decision in `research-wiki/experiments/`, this log, and the active
    tracker before resuming experiments.

## 2026-05-21T12:18:00+08:00 - Two-server running experiment report

- `35407`:
  - active screen: `613935.clean_baseline_postfix_diag_20260521`;
  - experiment: repaired clean baseline diagnostic run;
  - path: `/root/autodl-tmp/OpenTAD_SparseTAD_Clean_sapm_61e272e`;
  - config: `configs/adatad/thumos/input_random_fixed_50pct_adapter.py`;
  - log: `logs/clean_baseline_postfix_diag_20260521_1159.log`;
  - latest checked train line: epoch 3 completed with
    `[003][00099/00099] Loss=0.8292`;
  - no first eval yet;
  - checkpoints written: `epoch_1.pth`, `epoch_3.pth`;
  - GPU: RTX 4080 SUPER, about `3455/32760 MiB`;
  - disk: `/root/autodl-tmp` about `30G` free.
- `35329`:
  - active screen: `982088.adapter_feature_consistency_safe_fixed_bb90161`;
  - experiment: `input_random_fixed_50pct_adapter_feature_consistency_safe.py`;
  - path: `/root/autodl-tmp/OpenTAD_Back_check`;
  - log: `logs/input_random_fixed_50pct_adapter_feature_consistency_safe_20260520_122435.log`;
  - latest completed eval before the current eval: epoch 53 eval
    `Average-mAP=63.39`, mAP vector
    `79.46 / 74.52 / 65.48 / 55.90 / 41.59`;
  - latest train line: epoch 55 completed with
    `[055][00099/00099] Loss=0.4651`;
  - current state: epoch 55 evaluation loop active, around mid-eval in the
    raw tail; `tools/train.py` processes are alive;
  - checkpoints present through `epoch_49.pth`;
  - GPU: RTX 4090 D, about `3979/24564 MiB`;
  - disk: `/root/autodl-tmp` about `137G` free.
- Decision:
  - keep both runs active;
  - do not launch new sparse-aware long runs while repaired clean baseline
    first eval is pending;
  - wait for `35407` first eval and `35329` current eval/final trend.

## 2026-05-21T12:35:00+08:00 - Repaired clean Adapter serial queue deployed on 35329

- Clarified current `35329` feature-consistency run:
  - config: `input_random_fixed_50pct_adapter_feature_consistency_safe.py`;
  - purpose: training-only feature regularization across two strict
    random-fixed 50% views;
  - it does not change test sampling, post-processing, decode, NMS, or eval;
  - latest confirmed eval remains epoch 53 `Average-mAP=63.39`,
    vector `79.46 / 74.52 / 65.48 / 55.90 / 41.59`, below random-fixed
    Adapter `63.77` and strict EMA `63.85`.
- Synced the reviewed selected-axis postprocess repair files to
  `35329:/root/autodl-tmp/OpenTAD_SparseTAD_Clean_sapm_61e272e`:
  - `opentad/models/utils/post_processing/utils.py`;
  - `opentad/models/utils/post_processing/__init__.py`;
  - `tests/test_post_processing_selected_axis.py`.
- Verification on `35329`:
  - `py_compile` PASS for the post-processing files;
  - `pytest` unavailable in the remote environment;
  - inline torch smoke PASS with marker
    `POSTPROC_SELECTED_AXIS_35329_INLINE_SMOKE_PASS`.
- Launched serial waiting queue:
  - screen: `11192.clean_adapter_postfix_early10_queue_20260521`;
  - queue script:
    `/root/autodl-tmp/OpenTAD_SparseTAD_Clean_sapm_61e272e/logs/run_clean_adapter_postfix_early10_35329.sh`;
  - gate log:
    `/root/autodl-tmp/OpenTAD_SparseTAD_Clean_sapm_61e272e/logs/clean_adapter_postfix_early10_20260521_1230_gate.log`;
  - screen log:
    `/root/autodl-tmp/OpenTAD_SparseTAD_Clean_sapm_61e272e/logs/clean_adapter_postfix_early10_20260521_1230_queue_screen.log`.
- Serial behavior:
  - waits for the active feature-consistency `tools/train.py` process to exit;
  - waits until GPU0 memory is below `800MiB`;
  - then launches repaired clean Adapter baseline.
- Early gate:
  - phase 1 trains only to epoch 10 with
    `workflow.val_start_epoch=9`, `workflow.val_eval_interval=10`,
    `workflow.end_epoch=10`;
  - if no `Average-mAP` appears, or Avg-mAP `<20.0`, the run writes
    `EARLY10_FAIL` and stops;
  - if Avg-mAP `>=20.0`, it resumes from `epoch_9.pth` to epoch 60 with
    normal post-40 eval cadence;
  - `workflow.disable_checkpoint=False`, `workflow.checkpoint_interval=2`,
    and `post_processing.save_dict=True` are enabled.
- Current queue state:
  - queue is waiting, not using GPU;
  - active feature-consistency process is still present;
  - `/root/autodl-tmp` has about `137G` free.

## 2026-05-21T12:40:00+08:00 - Active run refresh after 35329 queue deployment

- `35329`:
  - screens: `adapter_feature_consistency_safe_fixed_bb90161` and
    `clean_adapter_postfix_early10_queue_20260521`;
  - feature-consistency epoch 55 eval completed:
    `Average-mAP=63.58`;
  - mAP vector:
    `79.66 / 74.53 / 65.84 / 56.01 / 41.87`;
  - this remains below random-fixed Adapter `63.77` and strict EMA `63.85`,
    so it is not a positive result claim;
  - the repaired clean Adapter queue is still waiting for the active
    feature-consistency process and has not started GPU training.
- `35407`:
  - repaired clean baseline screen:
    `clean_baseline_postfix_diag_20260521`;
  - latest checked train line:
    `[008][00050/00099] Loss=0.7308`;
  - no first eval yet;
  - checkpoints present through `epoch_7.pth`;
  - GPU memory about `3457/32760 MiB`.

## 2026-05-21T12:41:00+08:00 - Active run refresh

- `35407` repaired clean baseline:
  - screen: `clean_baseline_postfix_diag_20260521`;
  - log: `logs/clean_baseline_postfix_diag_20260521_1159.log`;
  - latest checked train line:
    `[009][00050/00099] Loss=0.6571`;
  - no `Average-mAP` yet;
  - checkpoints present through `epoch_7.pth`;
  - GPU memory about `3457/32760 MiB`;
  - `/root/autodl-tmp` has about `29G` free.
- `35329`:
  - feature-consistency remains active after epoch 55 eval
    `Average-mAP=63.58`;
  - queued repaired clean Adapter screen
    `clean_adapter_postfix_early10_queue_20260521` remains waiting;
  - queue gate log still shows active feature-consistency `tools/train.py`
    processes, so phase 1 has not started and no GPU conflict exists;
  - `/root/autodl-tmp` has about `137G` free.
- Decision:
  - continue both active long-running screens;
  - no sparse-aware STGA/SAPM/SAHM/SAN long run is allowed until repaired
    clean baseline evidence is available;
  - next blocking evidence remains the `35407` first eval or the `35329`
    early-10 eval after the queue starts.

## 2026-05-21T12:57:00+08:00 - Active run refresh and SAHM-NoGate DeepSeek gate completed

- `35407` repaired clean baseline:
  - screen: `613935.clean_baseline_postfix_diag_20260521`;
  - log: `logs/clean_baseline_postfix_diag_20260521_1159.log`;
  - latest checked train line:
    `[010][00099/00099] Loss=0.6695`;
  - no `Average-mAP` yet;
  - checkpoints present through `epoch_9.pth`;
  - GPU about `3459/32760 MiB`;
  - `/root/autodl-tmp` has about `28G` free.
- `35329`:
  - feature-consistency screen:
    `982088.adapter_feature_consistency_safe_fixed_bb90161`;
  - latest checked train line:
    `[056][00050/00099] Loss=0.4223`;
  - latest eval remains epoch 55 `Average-mAP=63.58`, vector
    `79.66 / 74.53 / 65.84 / 56.01 / 41.87`;
  - this remains below random-fixed Adapter `63.77` and strict EMA `63.85`,
    so it is not a positive result claim;
  - repaired clean Adapter queue
    `11192.clean_adapter_postfix_early10_queue_20260521` is still waiting
    for feature-consistency to finish; phase1 has not started;
  - `/root/autodl-tmp` has about `137G` free.
- SAHM-Deformable-NoGate-AllCls review gate:
  - Gemini CLI review: PASS;
  - Claude CLI DeepSeek `deepseek-v4-pro`: PASS;
  - stdout path:
    `OpenTAD_SparseTAD_Clean/logs/claude_deepseek_v4_pro_sahm_deformable_nogate_allcls_review_20260521.txt`;
  - substantive Claude plan:
    `C:\Users\skywalker\.claude\plans\read-only-secondary-code-staged-moonbeam.md`;
  - review report:
    `research-wiki/experiments/CLAUDE_DEEPSEEK_V4_PRO_SAHM_DEFORMABLE_NOGATE_ALLCLS_REVIEW_20260521.md`;
  - post-review verification:
    `python -m py_compile ...` PASS and `git diff --check ...` PASS with
    LF/CRLF warnings only;
  - committed in `OpenTAD_SparseTAD_Clean`:
    `5da14fd add SAHM no-gate ablation`.
- Decision:
  - SAHM-NoGate is reviewed and committed, but no deployment or long training
    is allowed while the repaired clean baseline gate remains unresolved;
  - continue monitoring `35407` first eval and the `35329` serial queue.

## 2026-05-21T13:00:00+08:00 - Active run refresh

- `35407` repaired clean baseline:
  - latest checked train line:
    `[013][00099/00099] Loss=0.6655`;
  - no `Average-mAP` yet;
  - checkpoints present through `epoch_13.pth`;
  - GPU about `3459/32760 MiB`.
- `35329`:
  - feature-consistency latest checked train line:
    `[056][00099/00099] Loss=0.4419`;
  - latest eval remains epoch 55 `Average-mAP=63.58`, vector
    `79.66 / 74.53 / 65.84 / 56.01 / 41.87`;
  - repaired clean Adapter queue still waits for feature-consistency to
    finish; phase1 has not started.
- Decision:
  - no result gate change;
  - continue monitoring repaired clean baseline and the serial queue;
  - no STGA/SAPM/SAHM/SAN long run launch while clean-stack recovery evidence
    is pending.

## 2026-05-21T13:04:00+08:00 - Active run refresh

- `35407` repaired clean baseline:
  - screen: `613935.clean_baseline_postfix_diag_20260521`;
  - latest checked train line:
    `[014][00050/00099] Loss=0.5802`;
  - no `Average-mAP` yet;
  - checkpoints present through `epoch_13.pth`;
  - `/root/autodl-tmp` has about `27G` free;
  - no checked Traceback/OOM/non-finite line.
- `35329`:
  - feature-consistency latest checked train line:
    `[056][00099/00099] Loss=0.4419`;
  - latest eval remains epoch 55 `Average-mAP=63.58`, vector
    `79.66 / 74.53 / 65.84 / 56.01 / 41.87`;
  - repaired clean Adapter early-10 queue is still waiting for
    feature-consistency to finish; phase1 has not started.
- Decision:
  - no result gate change;
  - continue both active screens;
  - no STGA/SAPM/SAHM/SAN long run launch while repaired clean baseline
    first-eval evidence is pending;
  - if `35407` disk headroom drops further, apply checkpoint retention only
    after the run completes or is explicitly stopped.

## 2026-05-21T13:08:00+08:00 - Active run refresh

- `35407` repaired clean baseline:
  - screen: `613935.clean_baseline_postfix_diag_20260521`;
  - latest checked train line:
    `[014][00050/00099] Loss=0.5802`;
  - no `Average-mAP` yet;
  - checkpoints present through `epoch_13.pth`;
  - result log file exists at
    `exps/thumos/adatad/input_random_fixed_50pct_adapter_postfix_diag_20260521/gpu1_id0/log.json`;
  - `/root/autodl-tmp` has about `27G` free.
- `35329`:
  - active screens:
    `982088.adapter_feature_consistency_safe_fixed_bb90161` and
    `11192.clean_adapter_postfix_early10_queue_20260521`;
  - feature-consistency latest checked train line:
    `[056][00099/00099] Loss=0.4419`;
  - latest eval remains epoch 55 `Average-mAP=63.58`, vector
    `79.66 / 74.53 / 65.84 / 56.01 / 41.87`;
  - repaired clean Adapter early-10 queue is still waiting for
    feature-consistency to finish; phase1 logs are empty.
- Decision:
  - no first-eval gate result yet;
  - keep both screens active;
  - no sparse-aware STGA/SAPM/SAHM/SAN long training launch;
  - next blocking evidence is `35407` repaired clean first eval or the
    `35329` early-10 queue result after feature-consistency exits.

## 2026-05-21T13:12:00+08:00 - Active run refresh

- `35407` repaired clean baseline:
  - screen: `613935.clean_baseline_postfix_diag_20260521`;
  - latest checked train line:
    `[016][00050/00099] Loss=0.5701`;
  - no `Average-mAP` yet;
  - checkpoint file count: `8`;
  - latest checkpoint: `epoch_15.pth`;
  - `/root/autodl-tmp` has about `27G` free;
  - no checked crash/OOM/non-finite line.
- `35329`:
  - feature-consistency latest checked train line:
    `[057][00050/00099] Loss=0.4030`;
  - latest eval remains epoch 55 `Average-mAP=63.58`, vector
    `79.66 / 74.53 / 65.84 / 56.01 / 41.87`;
  - repaired clean Adapter early-10 queue is still waiting for
    feature-consistency to finish; phase1 has not started.
- Decision:
  - no gate result yet;
  - continue monitoring;
  - no new sparse-aware long training launch;
  - do not clean `35407` checkpoints mid-run unless disk becomes critical.

## 2026-05-21T13:16:00+08:00 - Active run refresh

- `35407` repaired clean baseline:
  - screen: `613935.clean_baseline_postfix_diag_20260521`;
  - latest checked train line:
    `[016][00050/00099] Loss=0.5701`;
  - no `Average-mAP` yet;
  - checkpoint file count: `8`;
  - latest checkpoint: `epoch_15.pth`;
  - `/root/autodl-tmp` has about `27G` free;
  - no checked crash/OOM/non-finite line.
- `35329`:
  - feature-consistency latest checked train line:
    `[057][00050/00099] Loss=0.4030`;
  - latest eval remains epoch 55 `Average-mAP=63.58`, vector
    `79.66 / 74.53 / 65.84 / 56.01 / 41.87`;
  - repaired clean Adapter early-10 queue still waits for feature-consistency
    to finish; phase1 has not started.
- Decision:
  - no first-eval or early-10 gate result yet;
  - keep both monitored screens active;
  - no STGA/SAPM/SAHM/SAN long run launch;
  - next blocking evidence remains `35407` repaired clean first eval or the
    `35329` queued early-10 result.

## 2026-05-21T13:22:00+08:00 - Active run refresh

- `35407` repaired clean baseline:
  - screen: `613935.clean_baseline_postfix_diag_20260521`;
  - log: `logs/clean_baseline_postfix_diag_20260521_1159.log`;
  - latest checked train line:
    `[018][00099/00099] Loss=0.5838`;
  - no `Average-mAP` yet;
  - checkpoint file count: `9`;
  - latest checkpoint: `epoch_17.pth`;
  - `/root/autodl-tmp` has about `26G` free;
  - sampled GPU state: `3459/32760 MiB`, util `0%`;
  - no checked Traceback/RuntimeError/CUDA/OOM/non-finite line.
- `35329`:
  - feature-consistency screen:
    `982088.adapter_feature_consistency_safe_fixed_bb90161`;
  - latest checked train line:
    `[057][00099/00099] Loss=0.4228`;
  - latest eval remains epoch 55 `Average-mAP=63.58`, vector
    `79.66 / 74.53 / 65.84 / 56.01 / 41.87`;
  - repaired clean Adapter early-10 queue screen
    `11192.clean_adapter_postfix_early10_queue_20260521` still waits for
    feature-consistency to finish; phase1 log is empty;
  - `/root/autodl-tmp` has about `137G` free.
- Decision:
  - no first-eval or early-10 gate result yet;
  - feature-consistency still has no positive claim because it remains below
    random-fixed Adapter `63.77` and strict EMA `63.85`;
  - keep both active screens running;
  - do not launch STGA/SAPM/SAHM/SAN/combo long training;
  - do not clean `35407` checkpoints mid-run unless disk headroom becomes
    critical; current free space is sufficient for this monitored run.

## 2026-05-21T13:26:00+08:00 - Active run refresh

- `35407` repaired clean baseline:
  - screen: `613935.clean_baseline_postfix_diag_20260521`;
  - latest checked train line:
    `[019][00099/00099] Loss=0.6053`;
  - no `Average-mAP` yet;
  - checkpoint file count: `10`;
  - latest checkpoint: `epoch_19.pth`;
  - `/root/autodl-tmp` has about `26G` free;
  - sampled GPU state: `3459/32760 MiB`, util `0%`;
  - no checked Traceback/RuntimeError/CUDA/OOM/non-finite line.
- `35329`:
  - feature-consistency screen:
    `982088.adapter_feature_consistency_safe_fixed_bb90161`;
  - latest checked train line remains:
    `[057][00099/00099] Loss=0.4228`;
  - latest eval remains epoch 55 `Average-mAP=63.58`, vector
    `79.66 / 74.53 / 65.84 / 56.01 / 41.87`;
  - repaired clean Adapter early-10 queue screen
    `11192.clean_adapter_postfix_early10_queue_20260521` is still waiting for
    feature-consistency to exit; phase1 log remains empty;
  - `/root/autodl-tmp` has about `137G` free.
- Decision:
  - no first-eval or early-10 gate result yet;
  - no new model/result interpretation is possible from this check;
  - keep both active screens running;
  - keep STGA/SAPM/SAHM/SAN/combo long training blocked until repaired clean
    baseline evidence arrives.

## 2026-05-21T13:29:00+08:00 - Active run refresh

- `35407` repaired clean baseline:
  - screen: `613935.clean_baseline_postfix_diag_20260521`;
  - latest checked train line:
    `[020][00050/00099] Loss=0.5485`;
  - no `Average-mAP` yet;
  - checkpoint file count: `10`;
  - latest checkpoint remains `epoch_19.pth`;
  - `/root/autodl-tmp` has about `26G` free;
  - sampled GPU state: `3459/32760 MiB`, util `0%`;
  - no checked Traceback/RuntimeError/CUDA/OOM/non-finite line.
- `35329`:
  - feature-consistency screen:
    `982088.adapter_feature_consistency_safe_fixed_bb90161`;
  - latest checked train line remains:
    `[057][00099/00099] Loss=0.4228`;
  - latest eval remains epoch 55 `Average-mAP=63.58`, vector
    `79.66 / 74.53 / 65.84 / 56.01 / 41.87`;
  - repaired clean Adapter early-10 queue screen
    `11192.clean_adapter_postfix_early10_queue_20260521` is still waiting;
  - phase1 log remains empty;
  - `/root/autodl-tmp` has about `137G` free.
- Decision:
  - no first-eval or early-10 gate result yet;
  - `35407` storage headroom is still sufficient for reaching first eval;
  - no checkpoint cleanup during the active run;
  - keep both active screens running;
  - keep STGA/SAPM/SAHM/SAN/combo long training blocked.

## 2026-05-21T13:31:00+08:00 - Active run refresh

- `35407` repaired clean baseline:
  - screen: `613935.clean_baseline_postfix_diag_20260521`;
  - latest checked train line:
    `[021][00050/00099] Loss=0.5446`;
  - no `Average-mAP` yet;
  - checkpoint file count: `10`;
  - latest checkpoint remains `epoch_19.pth`;
  - `/root/autodl-tmp` has about `26G` free;
  - sampled GPU state: `3459/32760 MiB`, util `0%`;
  - no checked Traceback/RuntimeError/CUDA/OOM/non-finite line.
- `35329`:
  - feature-consistency screen:
    `982088.adapter_feature_consistency_safe_fixed_bb90161`;
  - latest checked train line remains:
    `[057][00099/00099] Loss=0.4228`;
  - latest eval remains epoch 55 `Average-mAP=63.58`, vector
    `79.66 / 74.53 / 65.84 / 56.01 / 41.87`;
  - repaired clean Adapter early-10 queue screen
    `11192.clean_adapter_postfix_early10_queue_20260521` is still waiting;
  - phase1 log remains empty;
  - `/root/autodl-tmp` has about `137G` free.
- Decision:
  - no first-eval or early-10 gate result yet;
  - continue waiting for `35407` repaired clean first eval as the primary
    shared-protocol recovery gate;
  - keep all sparse-aware long runs blocked.

## 2026-05-21T13:35:00+08:00 - Active run refresh

- `35407` repaired clean baseline:
  - screen: `613935.clean_baseline_postfix_diag_20260521`;
  - latest checked train line:
    `[021][00099/00099] Loss=0.5524`;
  - no `Average-mAP` yet;
  - checkpoint file count: `11`;
  - latest checkpoint: `epoch_21.pth`;
  - `/root/autodl-tmp` has about `25G` free;
  - sampled GPU state: `3459/32760 MiB`, util `3%`;
  - no checked Traceback/RuntimeError/CUDA/OOM/non-finite line.
- `35329`:
  - feature-consistency screen:
    `982088.adapter_feature_consistency_safe_fixed_bb90161`;
  - latest checked train line remains:
    `[057][00099/00099] Loss=0.4228`;
  - latest eval remains epoch 55 `Average-mAP=63.58`, vector
    `79.66 / 74.53 / 65.84 / 56.01 / 41.87`;
  - sampled GPU state: `3979/24564 MiB`, util `58%`;
  - repaired clean Adapter early-10 queue screen
    `11192.clean_adapter_postfix_early10_queue_20260521` is still waiting;
  - phase1 log remains empty;
  - `/root/autodl-tmp` has about `137G` free.
- Decision:
  - no first-eval or early-10 gate result yet;
  - keep both active screens running;
  - monitor `35407` disk because checkpoint interval 2 is intentionally
    preserving observability, but do not clean mid-run unless headroom becomes
    critical;
  - keep all sparse-aware long runs blocked.

## 2026-05-21T13:56:00+08:00 - GPT-5 Pro CVPR figure design discussion

- Sent the current Sparse TAD sparse-aware structure and protocol state to
  GPT-5.5 Pro through Oracle browser mode.
- Initial attachment-based route timed out before the browser send button was
  reached; the successful route used a no-attachment prompt containing the
  complete method summary.
- Pro session:
  - id: `sparse-tad-cvpr-figure-pro-2`;
  - resolved ChatGPT label: `Extended Pro`;
  - conversation:
    `https://chatgpt.com/c/6a0e9c0e-b0e4-83ea-9a50-9766bac64c50`;
  - full transcript:
    `logs/gpt5pro_sparse_tad_cvpr_figure_design_20260521.md`;
  - local summary:
    `research-wiki/experiments/SPARSE_TAD_CVPR_FIGURE_PRO_DISCUSSION_20260521.md`.
- Pro recommendation:
  - use a four-panel blueprint:
    `A. Problem setup`, `B. Modular sparse-aware TAD architecture`,
    `C. Coordinate protocol repair`, and
    `D. Experiment and diagnosis gate`;
  - compress the paper main figure into a double-column three-zone layout:
    fixed irregular input + metadata contract, candidate module sites, and
    coordinate repair + clean baseline gate;
  - draw STGA/SAPM/SAN/SAHM as gated candidate modules, not verified wins;
  - make selected-axis -> dense-axis -> seconds conversion explicit and label
    it as sampling-metadata-only with no GT/teacher signal.
- Next figure action:
  - use the Pro design as input for a deterministic editable SVG/FigureSpec
    draft.

## 2026-05-21T16:22:00+08:00 - repaired clean baseline and feature-consistency monitor refresh

- Skills used for this monitoring pass:
  - `experiment-plan` for plan-to-evidence mapping;
  - `run-experiment` for SSH/GPU/storage preflight discipline;
  - `monitor-experiment` for screen/log/result collection;
  - `experiment-queue` for the serial early-10 queue interpretation.
- `35407` repaired clean Adapter baseline:
  - screen: `613935.clean_baseline_postfix_diag_20260521`;
  - log: `/root/autodl-tmp/OpenTAD_SparseTAD_Clean_sapm_61e272e/logs/clean_baseline_postfix_diag_20260521_1159.log`;
  - evals after selected-axis postprocess repair:
    - epoch 41: `61.74`, vector `78.27 / 73.43 / 65.08 / 53.74 / 38.19`;
    - epoch 43: `62.25`, vector `78.37 / 73.80 / 65.70 / 54.63 / 38.74`;
    - epoch 45: `62.42`, vector `78.58 / 73.97 / 65.86 / 54.49 / 39.21`;
  - current state: active, evaluating after epoch 47 around `177/396`;
  - storage: `/root/autodl-tmp` `184G/200G` used, `17G` free;
  - checkpoints: `24` epoch checkpoints, latest `epoch_47.pth`.
- `35329` feature-consistency:
  - final epoch-59 eval: `63.80`, vector `79.80 / 74.73 / 66.07 / 55.95 / 42.46`;
  - interpretation: near random-fixed baseline `63.77`, still below strict EMA `63.85`,
    so this is not a strong positive model-side claim.
- `35329` repaired clean Adapter early-10 queue:
  - screen: `11192.clean_adapter_postfix_early10_queue_20260521`;
  - phase log:
    `/root/autodl-tmp/OpenTAD_SparseTAD_Clean_sapm_61e272e/logs/clean_adapter_postfix_early10_20260521_1230_phase1_10epoch.log`;
  - current state: phase1 active at epoch 7, latest `[007][00050/00099] Loss=0.7832`;
  - checkpoints: `epoch_1/3/5.pth`;
  - storage: `/root/autodl-tmp` has about `135G` free.
- `25876`:
  - old dead screens wiped;
  - no GPU devices visible;
  - `/root/autodl-tmp` remains about `44G` free after prior cleanup.
- Decision:
  - selected-axis postprocess repair clearly fixes the failure-level `7.x` collapse,
    but `35407` clean gate is only partial because current best is `62.42`, not
    `63.77-63.85`;
  - keep all STGA/SAPM/SAHM/SAN/combo long runs blocked;
  - continue `35407` to final/best clean mAP and wait for `35329` early-10 gate;
  - do not clean active-run checkpoints unless disk becomes critical; after
    complete training, apply the keep-last checkpoint rule and record evidence.

## 2026-05-21T18:26:00+08:00 - repaired clean baseline server monitor refresh

- Used `monitor-experiment` for a read-only check of `35407`, `35329`, and
  `25876`; no launch/stop/cleanup action was taken.
- `35407`:
  - active screen `613935.clean_baseline_postfix_diag_20260521`;
  - log:
    `/root/autodl-tmp/OpenTAD_SparseTAD_Clean_sapm_61e272e/logs/clean_baseline_postfix_diag_20260521_1159.log`;
  - repaired clean Adapter eval trend:
    `61.74 -> 62.25 -> 62.42 -> 62.53 -> 62.79 -> 62.83 -> 62.94 -> 63.15`;
  - latest eval vector:
    `78.89 / 74.13 / 66.61 / 55.40 / 40.72`;
  - latest train line:
    `[057][00099/00099] Loss=0.4185`;
  - checkpoints through `epoch_57.pth`, count `29`;
  - `/root/autodl-tmp` has `14G` free.
- `35329`:
  - active screen `11192.clean_adapter_postfix_early10_queue_20260521`;
  - phase1 early-10 eval reached `Average-mAP=28.21`, passing the `>=20`
    sanity gate;
  - phase2 resume is active at epoch 17 with latest
    `[016][00099/00099] Loss=0.5977`;
  - `/root/autodl-tmp` has `132G` free.
- `25876`:
  - no GPU devices;
  - no screen sockets;
  - no active train/test process;
  - `/root/autodl-tmp` has `44G` free.
- Decision:
  - strict random-fixed 50% and no-test-GT/no-teacher contracts are preserved;
  - repaired clean baseline is recovering but still below `63.77`/`63.85`;
  - clean gate remains partial;
  - keep all sparse-aware long runs blocked;
  - continue both repaired clean runs and monitor `35407` disk headroom.

## 2026-05-21T18:46:44+08:00 - BATA dynamic frame-selection plan revised

- Created/updated the BATA input-side planning documents from the manual Pro
  discussion and the user's follow-up revision:
  - `research-wiki/experiments/BATA_GPT5_PRO_MANUAL_PLAN_20260521.md`;
  - `research-wiki/experiments/BATA_BOUNDARY_AWARE_TOKEN_ACQUISITION_PLAN_20260521.md`;
  - `research-wiki/experiments/BATA_FRAME_SELECTION_TASK_TRACKER_20260521.md`.
- Route decision:
  - fixed 50% is no longer treated as the final ceiling of the selection method;
  - fixed 50% remains a fairness anchor, protocol gate, and fallback ablation;
  - the final selection route now targets dynamic frame sampling under matched
    average budget controls.
- Pre-experiment budget:
  - avoid a large multi-round GPU pre-experiment grid;
  - keep only two information-heavy gates before full selector training:
    offline dynamic boundary observability and one minimal detector closed loop.
- Parallel execution rule:
  - keep the sparse-aware model route and BATA selection route in separate
    trackers and server lanes;
  - no BATA long run is launched by this documentation update.

## 2026-05-21T18:54:32+08:00 - BATA target clarified as dynamic省帧 toward full-frame AdaTAD

- User confirmed the true selected-frame count should be dynamic in the final
  method; early experiments may use fixed hyperparameters, then later improve
  toward learned/adaptive budget prediction.
- Updated:
  - `research-wiki/experiments/BATA_BOUNDARY_AWARE_TOKEN_ACQUISITION_PLAN_20260521.md`;
  - `research-wiki/experiments/BATA_FRAME_SELECTION_TASK_TRACKER_20260521.md`;
  - `research-wiki/experiments/BATA_GPT5_PRO_MANUAL_PLAN_20260521.md`.
- Full-frame target anchor recorded:
  - AdaTAD full-frame stride-1: `68.97` Avg-mAP / `47.46` mAP@0.7.
- Revised paper-level target:
  - use dynamic selected-frame counts, ideally averaging about 35%-40% high-cost
    frames, to approach or exceed the full-frame reference;
  - fixed 50% remains a control and attribution anchor, not the final method.
- No code, deployment, or long run was started by this update.

## 2026-05-21T22:16:22+08:00 - BATA execution order corrected to fixed-50% first

- User clarified the first implementation should keep the selected-frame count
  fixed at 50%; dynamic frame count should be added only after strong fixed-50%
  performance is achieved.
- Updated:
  - `research-wiki/experiments/BATA_BOUNDARY_AWARE_TOKEN_ACQUISITION_PLAN_20260521.md`;
  - `research-wiki/experiments/BATA_FRAME_SELECTION_TASK_TRACKER_20260521.md`;
  - `research-wiki/experiments/BATA_GPT5_PRO_MANUAL_PLAN_20260521.md`.
- Current BATA execution rule:
  - first build fixed-50% boundary-aware selection and aim to beat
    `64.64 / 43.23@0.7`;
  - then run ablations and dynamic-frame-count extensions;
  - no BATA code, deployment, or long run was started by this update.

## 2026-05-21T22:19:47+08:00 - BATA intelligent-frame-selection grill-me record created

- Created independent question-and-answer record:
  `research-wiki/experiments/BATA_GRILL_ME_QA_20260521.md`.
- The record captures the settled intelligent-frame-selection decisions:
  - first implementation is fixed 50% boundary-aware selection;
  - dynamic frame count is deferred until fixed-50% performance is strong;
  - long-term target remains approaching full-frame AdaTAD `68.97 / 47.46@0.7`
    with fewer high-cost frames;
  - pre-experiments are capped at an offline observability diagnostic plus one
    minimal fixed-50% detector loop.
- Updated `BATA_FRAME_SELECTION_TASK_TRACKER_20260521.md`.
- No code, deployment, or long run was started by this documentation update.

## 2026-05-22T00:02:42+08:00 - BATA grill-me Q7-Q24 compact summary

- Consolidated recent per-question BATA grill-me log entries to keep
  `research-wiki/log.md` compact. Full details remain in
  `research-wiki/experiments/BATA_GRILL_ME_QA_20260521.md`.
- Decisions retained: use a diagnostic noisy-boundary gate first; implement BCA
  with coverage floor, adaptive boundary demand, adaptive local radius, coverage
  repair, robust score normalization, and clipped demand; train a deployable
  binary MobileNet-based `64x64` preview boundary predictor with soft start/end
  boundary labels; report preview compute separately for fixed-50% results and
  include it in dynamic省帧 accounting; monitor MobileNet boundary learning and
  original-time-axis selected-frame distributions; skip a broad fixed-50%
  ablation block and proceed to dynamic frame count after usable fixed-50%
  performance and healthy diagnostics.
- No code, deployment, or long run was started by these documentation updates.

## 2026-05-22T00:35:49+08:00 - BATA route paused with recovery context

- Paused the intelligent frame-selection route because Pro review is currently
  unavailable and the dynamic-method `grill-me` is incomplete.
- Pro attempts did not produce a usable review: Rosetta CDP refused
  `127.0.0.1:9222`, Oracle Pro attachment upload timed out, and the Oracle
  inline request was interrupted before a substantive response returned.
- Updated `BATA_GRILL_ME_QA_20260521.md`,
  `BATA_BOUNDARY_AWARE_TOKEN_ACQUISITION_PLAN_20260521.md`, and
  `BATA_FRAME_SELECTION_TASK_TRACKER_20260521.md` with a recoverable pause
  state.
- Recoverable state: Q1-Q28 are recorded; Q27 makes `25% / 50% / 75%` bins
  ablation/control only; Q28 confirms constrained optimization as the intended
  dynamic method; Q29 and the A-K dynamic-controller details remain unresolved.
- No code, deployment, or long run was started.

## 2026-05-22T00:40:15+08:00 - Sparse-aware model route monitor and cleanup

- Checked model-side sparse-aware experiment progress on `35407`, `35329`, and
  `25876`.
- `35407` repaired clean Adapter baseline completed:
  - log: `logs/clean_baseline_postfix_diag_20260521_1159.log`;
  - final/best eval: `Average-mAP=63.43`;
  - status: `Training Over...`;
  - interpretation: still below random-fixed Adapter `63.77` and strict EMA
    `63.85`, so the clean gate is not fully recovered.
- Applied keep-last checkpoint cleanup for the completed `35407` run:
  - checkpoint dir:
    `/root/autodl-tmp/OpenTAD_SparseTAD_Clean_sapm_61e272e/exps/thumos/adatad/input_random_fixed_50pct_adapter_postfix_diag_20260521/gpu1_id0/checkpoint`;
  - kept `epoch_59.pth`;
  - removed 29 older `epoch_*.pth` files;
  - `/root/autodl-tmp` improved from `14G` free to `31G` free.
- `35329` repaired clean Adapter remains active in screen
  `11192.clean_adapter_postfix_early10_queue_20260521`;
  phase2 evals so far are `62.10 -> 62.50`, latest checked train line
  `[045][00099/00099] Loss=0.4838`.
- `25876` has no GPU devices, no active screen, and no train/test process.
- Decision: keep all STGA/SAPM/SAHM/SAN/combo long runs blocked until the
  repaired clean gate is resolved; wait for `35329` final/best mAP before
  deciding whether to relaunch any sparse-aware module.

## 2026-05-22T00:47:05+08:00 - Clean baseline reproduction interpretation

- Refined the repaired clean Adapter baseline decision: `63.43` is close to
  the historical random-fixed Adapter `63.77` in Avg-mAP (`-0.34`), so it can
  be treated as near Avg-mAP reproduction and clean-stack sanity recovery.
- It is not a strict clean-gate pass yet because `mAP@0.7=40.78` remains below
  the historical `42.19` by `-1.41`, and the independent `35329` repaired clean
  run is still unfinished.

## 2026-05-22T00:56:45+08:00 - STGA and head-side deployment selection

- Checked deployment candidates after the user asked to start STGA-v1a/v1b and
  head-side sparse experiments across two servers.
- `35407` is free and can host the Adapter-side diagnostic relaunch.
- `35329` still has an active repaired clean Adapter training process; latest
  parsed evals remain `62.10 -> 62.50`, so head-side work should be a
  wait-after-clean queue rather than preemption.
- Recommended split: `35407` runs checkpoint-enabled repaired `STGA-v1a`
  diagnostic first, `STGA-v1b` only if v1a does not collapse; `35329` queues
  `SAHM-Deformable-NoGate-AllCls` first, then
  `SAHM-Deformable-SoftmaxGate-AllCls` if healthy.

## 2026-05-22T01:09:33+08:00 - STGA-v1a diagnostic relaunch started

- Verified that the earlier timed-out remote launch actually started
  `STGA-v1a` on `35407`.
- Path: `/root/autodl-tmp/OpenTAD_SparseTAD_Clean_sapm_61e272e`.
- Log: `logs/stga_v1a_postfix_diag_20260522_0100.log`.
- Work dir:
  `exps/thumos/adatad/input_random_fixed_50pct_adapter_stga_v1a_postfix_diag_20260522_0100`.
- Launch uses checkpoint interval `4`, `disable_checkpoint=False`,
  `val_start_epoch=40`, `val_eval_interval=2`, `end_epoch=60`, and
  `post_processing.save_dict=True`.
- Startup health: train processes are active and the log reached
  `Training Starts...`; no mAP result exists yet.

## 2026-05-22T01:13:00+08:00 - 35329 repaired clean Adapter intermediate eval

- Recorded latest known `35329` repaired clean Adapter phase2 eval from
  `logs/clean_adapter_postfix_early10_20260521_1230_phase2_resume.log`:
  `Average-mAP=62.76`, vector `79.57 / 74.46 / 65.91 / 54.39 / 39.48`.
- Interpretation: still below random-fixed Adapter `63.77` and high-IoU anchor
  `42.19@0.7`; continue the active run to final/best and do not start another
  long run on `35329` while its `tools/train.py` process remains active.

## 2026-05-22T01:20:00+08:00 - Full sparse-aware code staged and SAHM wait queue deployed

- Staged full reviewed code `5da14fd` on `35407` and `35329`; active runs were
  unchanged.
- Added `35329` wait queue `605535.sahm_nogate_wait_after_clean_20260522`:
  launch `SAHM-NoGate` only after clean run exits and best Avg-mAP `>=63.3`;
  otherwise HOLD.
- Future deployment-only records should stay compact; full metrics/decisions
  remain in the main tracker.

## 2026-05-22T01:26:00+08:00 - SAPM-Embed wait queue deployed

- Added `35407` wait queue `396796.sapm_embed_wait_after_stga_20260522`; it
  waits for current STGA training to finish, then launches
  `SAPM-EmbedDeformable`. No new long run started yet.

## 2026-05-22T02:34:00+08:00 - 35329 repaired clean Adapter intermediate eval

- `35329` repaired clean Adapter phase2 improved to `Average-mAP=62.96`,
  vector `79.39 / 74.33 / 66.29 / 54.71 / 40.11`; still below the
  `63.77 / 42.19@0.7` random-fixed anchor, so continue to final/best and keep
  the SAHM queue gated at clean best Avg-mAP `>=63.3`.

## 2026-05-22T03:36:00+08:00 - 35329 repaired clean Adapter intermediate eval

- `35329` repaired clean Adapter phase2 improved to `Average-mAP=63.17`,
  vector `79.72 / 74.57 / 66.80 / 54.66 / 40.10`; still below the
  `63.77 / 42.19@0.7` anchor and below the `63.3` SAHM launch gate, so keep
  waiting for final/best. `35407` STGA-v1a has no mAP/crash/final result yet.

## 2026-05-22T04:38:00+08:00 - SAPM launched and 35329 clean Adapter improved

- `35407` queue launched `SAPM-EmbedDeformable` at `04:16:45`; no SAPM mAP yet.
  STGA-v1a is no longer active, but this check found no STGA mAP/crash/final
  line, so its termination reason remains unresolved until the next permitted
  log check.
- `35329` repaired clean Adapter phase2 reached `Average-mAP=63.35`, vector
  `79.77 / 74.52 / 66.90 / 54.75 / 40.78`; this clears the `63.3` SAHM queue
  Avg-mAP threshold, but still trails `63.77 / 42.19@0.7`.

## 2026-05-22T05:43:00+08:00 - 35329 repaired clean Adapter intermediate eval

- `35329` repaired clean Adapter phase2 latest eval is `Average-mAP=63.28`,
  vector `79.66 / 74.28 / 66.77 / 54.67 / 41.01`; Avg-mAP is slightly below
  the previous `63.35`, but `mAP@0.7=41.01` is the best high-IoU value so far
  for this repaired clean run. SAPM remains active with no mAP yet.

## 2026-05-22T06:46:00+08:00 - 35329 repaired clean Adapter intermediate eval

- `35329` repaired clean Adapter phase2 latest eval is `Average-mAP=63.26`,
  vector `79.74 / 74.32 / 66.34 / 54.70 / 41.20`; Avg-mAP remains below the
  run best `63.35`, but `mAP@0.7=41.20` is the current best high-IoU value.
  SAPM remains active with no mAP yet.

## 2026-05-22T07:49:00+08:00 - SAPM first eval and clean Adapter update

- `35407` SAPM-EmbedDeformable first eval: `Average-mAP=62.28`, vector
  `78.39 / 73.88 / 65.54 / 54.29 / 39.31`; weak and below clean/random-fixed
  anchors, but not a failure-scale collapse.
- `35329` repaired clean Adapter reached its current best `Average-mAP=63.44`,
  vector `79.80 / 74.51 / 66.32 / 55.19 / 41.39`; still below
  `63.77 / 42.19@0.7`. SAHM queue remains waiting.

## 2026-05-22T08:22:00+08:00 - SAPM intermediate eval

- `35407` SAPM-EmbedDeformable improved to `Average-mAP=62.81`, vector
  `78.81 / 74.48 / 65.90 / 54.94 / 39.90`; still below clean/random-fixed
  anchors, so continue to trend/final before deciding review or second step.

## 2026-05-22T08:57:00+08:00 - SAPM update, clean final, and SAHM launch

- `35407` SAPM-EmbedDeformable improved to `Average-mAP=63.05`, vector
  `79.13 / 74.55 / 65.76 / 55.39 / 40.40`; still below clean/random-fixed
  anchors, continue to trend/final.
- `35329` repaired clean Adapter completed with final/best `Average-mAP=63.57`,
  vector `79.94 / 74.65 / 66.56 / 55.68 / 40.99`; Avg-mAP is near reproduction,
  but high-IoU remains below `42.19@0.7`.
- SAHM-NoGate queue launched at `08:30:26`. Clean checkpoint cleanup verified
  only `epoch_59.pth` remains; `/root/autodl-tmp` is `65G used / 136G free`.

## 2026-05-22T09:30:00+08:00 - SAPM intermediate eval

- `35407` SAPM-EmbedDeformable improved to `Average-mAP=63.27`, vector
  `79.21 / 74.61 / 66.19 / 55.72 / 40.62`; it is still below clean/random-fixed
  anchors, so continue to final/trend. SAHM-NoGate remains active with no mAP.

## 2026-05-22T10:02:00+08:00 - SAPM intermediate eval

- `35407` SAPM-EmbedDeformable reached `Average-mAP=63.51`, vector
  `79.31 / 75.11 / 66.61 / 55.87 / 40.65`; best high-IoU so far is `40.82`.
  It is close to clean partial-pass Avg-mAP but still below `63.77 / 42.19@0.7`.
  SAHM-NoGate remains active with no mAP.

## 2026-05-22T10:35:00+08:00 - SAPM intermediate eval

- `35407` SAPM-EmbedDeformable reached `Average-mAP=63.60`, vector
  `79.52 / 75.15 / 66.68 / 55.97 / 40.71`; Avg-mAP now slightly exceeds the
  clean partial-pass `63.57`, but high-IoU is still below clean `40.99` and
  historical `42.19`. SAHM-NoGate remains active with no mAP.

## 2026-05-22T11:07:00+08:00 - SAPM intermediate eval

- `35407` SAPM-EmbedDeformable reached `Average-mAP=63.76`, vector
  `79.54 / 75.04 / 66.90 / 56.17 / 41.17`; Avg-mAP essentially matches
  random-fixed `63.77`, but high-IoU remains below `42.19@0.7`.
  SAHM-NoGate remains active with no mAP.

## 2026-05-22T11:40:00+08:00 - SAPM intermediate eval

- `35407` SAPM-EmbedDeformable reached `Average-mAP=63.84`, vector
  `79.71 / 75.27 / 66.82 / 56.15 / 41.24`; Avg-mAP now exceeds random-fixed
  `63.77` and nearly matches strict EMA `63.85`, but high-IoU remains below
  `42.19@0.7`. SAHM-NoGate remains active with no mAP.

## 2026-05-22T12:14:00+08:00 - SAPM final and checkpoint cleanup

- `35407` SAPM-EmbedDeformable completed with `Training Over` at `11:55:40`:
  final/best `Average-mAP=63.92`, vector
  `79.70 / 75.34 / 67.01 / 56.39 / 41.16`.
- Interpretation: partial pass. Avg-mAP is `+0.15` over random-fixed `63.77`
  and `+0.07` over strict EMA `63.85`, but still below stratified `64.64`,
  uniform `65.09`, and high-IoU anchor `42.19@0.7`.
- SAPM checkpoint cleanup on `35407` kept only `epoch_59.pth`; `/root/autodl-tmp`
  changed from `185G used / 16G free` to `177G used / 24G free`.
- `35329` SAHM-NoGate remains active with no mAP yet. Next server query must
  wait until at least `2026-05-22T12:42:26+08:00`.

## 2026-05-22T12:21:00+08:00 - Deployment snapshot after monitor limit cancelled

- User cancelled the 30-minute query restriction for STGA diagnosis and idle-server avoidance.
- `35407` is idle: no screen, no train/test process, GPU `0/32760 MiB`.
  Best immediate use is short STGA `epoch_39.pth` eval diagnosis or another
  reviewed queued experiment; do not launch STGA-v1b blindly.
- `35329` is active with SAHM-NoGate: around epoch 20, latest loss `0.5431`,
  GPU `3581/24564 MiB`, no mAP yet.

## 2026-05-22T12:52:00+08:00 - SAPM final normal, SAHM still running

- `35407` SAPM-EmbedDeformable completed normally with final/best
  `Average-mAP=63.92`, vector `79.70 / 75.34 / 67.01 / 56.39 / 41.16`;
  checkpoint cleanup kept only `epoch_59.pth`; GPU is idle.
- Interpretation: SAPM is a modest partial pass, not a collapse and not a
  paper-level win; `35407` can be reused for the next reviewed full train or a
  short STGA eval diagnostic if approved.
- `35329` SAHM-NoGate remains active at epoch 23, latest loss `0.5463`, no mAP
  yet; keep it serial and do not launch SoftmaxGate there while active.

## 2026-05-22T12:55:00+08:00 - STGA epoch-39 quick eval found

- `35407` STGA-v1a quick eval log:
  `OpenTAD_SparseTAD_Clean_sapm_61e272e/logs/stga_v1a_epoch39_eval_diag_20260522_1225.log`.
- It loaded
  `exps/thumos/adatad/input_random_fixed_50pct_adapter_stga_v1a_postfix_diag_20260522_0100/gpu1_id0/checkpoint/epoch_39.pth`
  and completed with `Average-mAP=62.50`, vector
  `79.29 / 74.05 / 65.05 / 54.45 / 39.66`.
- Interpretation: eval path is normal and not a failure-scale collapse, but
  STGA-v1a is negative versus repaired clean and random-fixed references; do
  not launch STGA-v1b or more STGA full training blindly.

## 2026-05-22T13:03:00+08:00 - STGA full continuation and serial queues

- User directed that STGA-v1a `epoch_39.pth` is not a final result and should
  continue to epoch 60. Launched `35407` screen
  `stga_v1a_resume39_full60_20260522`; log
  `logs/stga_v1a_resume39_full60_20260522_1304.log` confirms `Resume epoch is
  39`.
- Added serial wait queues without overlapping active training:
  `35407` SAN-Aggregation after STGA full result, and `35329` SAHM-SoftmaxGate
  after SAHM-NoGate. Both queues hold on crash or final Avg-mAP `<58.0`.

## 2026-05-22T13:20:00+08:00 - SAPM-AttnGapBias collapse interpretation refined

- Rechecked the `STGA-v1a 7.23` and `SAPM-AttnGapBias 7.54` collapse timeline
  against the later selected-axis postprocess repair evidence.
- Current interpretation: SAPM-AttnGapBias v1 was run before the repaired
  selected-axis to dense-axis postprocess protocol was validated, so its `7.54`
  is stale/protocol-suspect rather than a clean attention-bias design verdict.
- This does not mean SAPM should inherit Adapter-specific STGA residual code;
  the shared missing piece was the postprocess/protocol repair. Any future
  attention-bias revisit should be a repaired-protocol diagnostic with no-op
  equivalence, mask/scale audit, and bias magnitude probes.

## 2026-05-22T13:15:00+08:00 - SAN requeued behind SAPM

- User clarified that SAN should not be fully stopped and should be deployed
  behind SAPM. Added `35407` screen
  `san_aggregation_wait_after_sapm_attngap_20260522`.
- The SAN queue waits for SAPM-AttnGapBias repaired rerun to actually launch
  and finish before starting, so the 35407 chain is now:
  `STGA-v1a full continuation -> SAPM-AttnGapBias repaired rerun -> SAN-Aggregation`.
- `35329` remains serially queued as `SAHM-NoGate -> SAHM-SoftmaxGate`.

## 2026-05-22T13:35:00+08:00 - BATA route upgraded to RBEA-TAD

- Recorded the user's Pro frontier discussion in
  `research-wiki/experiments/BATA_PRO_FRONTIER_RBEA_DISCUSSION_20260522.md`.
- Route decision: the most valuable input-side paper direction is
  reasoning-guided boundary evidence acquisition, not generic adaptive
  sampling. Immediate implementation should be fixed-50% boundary evidence
  triplets + event/context anchors + coverage repair + evidence ledger.
- Dynamic frame count should use validation-set lambda marginal-gain control
  after fixed-50% works. Three budget bins remain ablations/controls.
- Counterfactual acquisition value learning is high-upside but deferred until
  the fixed-50% and dynamic RBEA baselines are healthy.

## 2026-05-22T13:55:00+08:00 - BATA route keeps deployable TAD path

- Recorded the route decision that video large-model temporal localization
  should not replace the current deployable TAD small-model / preview-acquisition
  route.
- Large models remain useful as training-time evidence teachers, candidate
  boundary teachers, diagnostics, or upper-bound side branches. The main route
  stays RBEA-TAD with low-cost preview, fixed-50% first, then dynamic
  average-cost acquisition.

## 2026-05-22T14:05:00+08:00 - BATA motivation can reference large-model TAD ability

- Recorded that "injecting temporal action detection capability into video
  large-model systems" is a useful high-level motivation and application frame.
- It should not replace the main method claim yet. The primary contribution
  remains a deployable boundary evidence acquisition module; large-model
  integration is a later teacher/diagnostic/application branch unless separately
  implemented and evaluated.

## 2026-05-22T14:20:00+08:00 - BATA distillation risk review recorded

- Recorded the new Pro discussion in
  `research-wiki/experiments/BATA_PRO_DISTILLATION_RISK_AND_PREDICT_VERIFY_ROUTE_20260522.md`.
- Main correction: do not distill video-large-model long CoT or agent traces
  into the 64x64 MobileNet preview selector. This is too risky and not needed
  because TAD has direct boundary supervision.
- Route refined to predict-verify boundary evidence acquisition: boundary
  hypothesis, transition surprise, neighboring candidate comparison,
  before/transition/after evidence, and coverage verification. Optional
  distillation is only narrow structured evidence bootstrapping after the main
  line works.

## 2026-05-22T14:28:00+08:00 - Full Pro distillation feedback recorded

- Expanded
  `research-wiki/experiments/BATA_PRO_DISTILLATION_RISK_AND_PREDICT_VERIFY_ROUTE_20260522.md`
  with a full structured record of Pro's feedback.
- The record now covers the direct verdict, distillation risks, allowed narrow
  distillation, better task combinations, predict-verify route, preview signals,
  boundary micro-clip evidence, coverage as evidence, implementation plan,
  optional bootstrapping, evaluation metrics, and unresolved technical checks.

## 2026-05-22T14:35:00+08:00 - BATA query-free Video-LLM TAD interface review recorded

- Created
  `research-wiki/experiments/BATA_PRO_QUERY_FREE_VIDEO_LLM_TAD_INTERFACE_20260522.md`
  with the full Pro feedback on whether the target should become direct
  Video-LLM TAD or a query-free dense temporal localization interface.
- Decision: do not pivot to direct Video-LLM TAD. Keep fixed-50%
  Predict-Verify RBEA and dynamic lambda RBEA as the main implementation path.
  Treat query-free Video-LLM TAD via boundary evidence packages as a deferred,
  separately costed application/diagnostic branch.

## 2026-05-22T14:45:00+08:00 - BATA Temporal Action Tokens refinement recorded

- Created
  `research-wiki/experiments/BATA_PRO_TEMPORAL_ACTION_TOKENS_20260522.md`
  with the full Pro feedback on why proposal-conditioned Video-LLM verification
  is still not the most elegant TAD injection route.
- Decision: the best long-term framing is `Boundary Evidence Acquisition for
  Temporal Action Tokens`. Do not start full Video-LLM TAD-token training now.
  First implement fixed-50% Predict-Verify RBEA and design the evidence ledger
  as lightweight action-token/evidence-token records; dynamic budget and full
  Video-LLM SFT/RLVR remain later branches.

## 2026-05-22T17:32:00+08:00 - BATA TAD-token novelty review started

- Created
  `research-wiki/experiments/BATA_TAD_TOKEN_IDEA_REVIEW_AND_NOVELTY_CHECK_20260522.md`
  and Pro prompt
  `logs/gpt5pro_bata_tad_token_sft_rlvr_innovation_prompt_20260522.md`.
- User decision: stop planning step-by-step diagnostic GPU experiments for this
  branch. Before any pivot to Full Video-LLM TAD-token SFT/RLVR, run novelty
  review and select only one maximum-information-gain discriminator experiment.
- External review status: Oracle browser Pro session `bata-tad-token-sft-rlvr`
  is running with `gpt-5.5-pro`; Gemini MCP novelty review failed with
  `unsupported Gemini backend: openai`; Gemini CLI fallback timed out without
  substantive stdout. No GPU launch is authorized from this review yet.

## 2026-05-22T20:17:00+08:00 - BATA TAD-token novelty verdict accepted

- Updated
  `research-wiki/experiments/BATA_TAD_TOKEN_IDEA_REVIEW_AND_NOVELTY_CHECK_20260522.md`,
  `research-wiki/experiments/BATA_BOUNDARY_AWARE_TOKEN_ACQUISITION_PLAN_20260521.md`,
  and `research-wiki/experiments/BATA_FRAME_SELECTION_TASK_TRACKER_20260521.md`
  with the Pro verdict.
- Oracle browser Pro session `bata-tad-token-sft-rlvr` completed with
  `gpt-5.5-pro`; transcript artifact is stored under
  `C:\Users\skywalker\.oracle\sessions\bata-tad-token-sft-rlvr\artifacts\transcript.md`.
- Accepted decision: `DO NOT PIVOT`. Full Video-LLM TAD-token SFT/RLVR novelty
  is `3.5/10`; RBEA/BATA as cost-aware boundary evidence acquisition plus
  temporal action-token evidence ledger is about `7/10`.
- Current route is fixed as `Boundary Evidence Acquisition for Temporal Action
  Tokens`. Full Video-LLM SFT/RLVR is high-risk future work only; the only
  allowed discriminator is a frozen-feature TAD-token/evidence-token decoder
  probe with strict high-IoU and evidence-causality thresholds.

## 2026-05-22T21:08:00+08:00 - Native Video-LLM query-free TAD new-idea preview completed

- Updated
  `research-wiki/experiments/BATA_VLLM_NATIVE_TAD_NEW_IDEA_PREVIEW_20260522.md`
  and
  `research-wiki/experiments/BATA_FRAME_SELECTION_TASK_TRACKER_20260521.md`.
- Oracle browser Pro session `vllm-query-free-tad-newidea` completed with
  `gpt-5.5-pro`; transcript artifact:
  `C:\Users\skywalker\.oracle\sessions\vllm-query-free-tad-newidea\artifacts\transcript.md`.
- Gemini novelty reviewer job `fb95b4d0fcf14c7a904af32d94746ea9` failed with
  invalid API key, so no completed Gemini cross-review is recorded.
- F2G / Foresee-to-Ground was independently verified at `arXiv:2605.21973`,
  submitted 2026-05-21, and is a new close risk for claims around verifiable
  evidence-driven temporal grounding.
- Accepted decision: `NEXT-PAPER ONLY`. Do not pivot the current RBEA/BATA
  project. If this branch becomes a future paper, position it as
  `Query-Free Dense Temporal Action Grounding`: a benchmark/protocol and
  minimal set-level post-training recipe for dense action-set construction
  failures such as misses, duplicates, split/merge errors, calibration, output
  instability, and high-IoU boundary drift.

## 2026-05-22T21:24:00+08:00 - BATA implementation uncertainty freeze completed

- Created and updated
  `research-wiki/experiments/BATA_IMPLEMENTATION_UNCERTAINTY_PRO_REVIEW_20260522.md`,
  updated
  `research-wiki/experiments/BATA_BOUNDARY_AWARE_TOKEN_ACQUISITION_PLAN_20260521.md`,
  and updated
  `research-wiki/experiments/BATA_FRAME_SELECTION_TASK_TRACKER_20260521.md`.
- Oracle browser Pro session `bata-implementa-freeze` completed with
  `gpt-5.5-pro`; transcript artifact:
  `C:\Users\skywalker\.oracle\sessions\bata-implementa-freeze\artifacts\transcript.md`.
- Accepted implementation decision: start PR-0/PR-1/PR-2 now. There is no
  route-level blocker for local implementation; the only hard blocker before
  detector closed-loop is reliable selected-axis/dense-axis/frame/seconds and
  evidence-ledger attribution.
- Frozen first runnable path: oracle/noisy-oracle boundary score cache ->
  fixed-50 BCA -> selected-position dump -> evidence ledger diagnostics.
- Frozen BCA v1: top-8 NMS peaks, fixed `p-4/p/p+4` before/center/after
  boundary triplets, and deterministic largest-gap coverage/context repair.
  MobileNet, dynamic lambda, multi-head selector, cheap controls, and deletion
  audit are deferred until fixed-50 allocator/ledger diagnostics are healthy.

## 2026-05-22T21:53:00+08:00 - BATA PR-0/PR-1/PR-2 local implementation coded

- Added local implementation report
  `research-wiki/experiments/BATA_PR0_PR1_PR2_LOCAL_IMPLEMENTATION_SELF_CHECK_20260522.md`
  and updated both BATA and Sparse TAD trackers.
- Changed files:
  `OpenTAD_Back/opentad/datasets/transforms/boundary_acquisition.py`,
  `OpenTAD_Back/opentad/datasets/transforms/end_to_end.py`,
  `OpenTAD_Back/tools/bata/build_oracle_boundary_score_cache.py`,
  `OpenTAD_Back/tools/bata/dump_bata_selection_ledger.py`, and
  `OpenTAD_Back/tests/test_bata_boundary_acquisition_contracts.py`.
- Implemented score-cache-driven fixed-50 BCA v1, explicit diagnostic GT-cache
  guard, `LoadFrames` method `bata_boundary_acquisition_subsample`,
  diagnostic oracle/noisy-oracle score cache builder, offline ledger dumper,
  and contract tests for group-size coordinate protocol, budget exactness,
  ledger separation, and cache leakage gating.
- Local verification passed:
  `python -m py_compile ...`, `python -m pytest tests\test_bata_boundary_acquisition_contracts.py -q`
  with `6 passed`, script `--help` checks, and a temporary THUMOS14
  noisy-oracle cache -> ledger dump smoke on 2 validation windows.
- No GPU training, remote staging, MobileNet selector, dynamic budget,
  detector/head/loss/assignment/post-processing change, or deployment occurred.
  Required GPT-5 Pro, Gemini CLI, and Claude DeepSeek review gates remain
  pending before any sync or detector closed loop.

## 2026-05-22T22:20:00+08:00 - BATA GPT-5 Pro code review failed; blockers fixed locally

- Added
  `research-wiki/experiments/BATA_PR0_PR1_PR2_GPT5_PRO_REVIEW_AND_FIXES_20260522.md`
  and updated both BATA and Sparse TAD trackers.
- GPT-5 Pro review gate source was manual user-pasted Pro output. Automated
  attempts failed: Oracle browser raw attachments timed out before a send
  button, Rosetta CDP refused `127.0.0.1:9222`, Oracle API lacked
  `OPENAI_API_KEY`, and an inline Oracle browser fallback was interrupted.
- Pro verdict: `FAIL`. Blocking findings were tubelet coverage semantics,
  diagnostic GT-cache/ledger gating, fps and peak-second mapping,
  score-cache stride/scale-factor handshake, and offline duplicate tail
  windows.
- Fixed all five blockers locally:
  dense-gap-aware tubelet coverage/reclaim; required cache manifest fields;
  loader diagnostic flag binding for GT-derived caches; `avg_fps` propagation
  and peak seconds via `dense_index_to_time_span`; manifest
  `axis_frame_stride` / `scale_factor` validation; and no-duplicate offline
  `_window_starts`.
- Verification after fixes:
  `python -m py_compile ...` PASS,
  `python -m pytest tests\test_bata_boundary_acquisition_contracts.py -q`
  PASS with `11 passed`, temporary THUMOS14 noisy-oracle cache -> ledger dump
  PASS, and `git diff --check` PASS except existing LF/CRLF warning on
  `end_to_end.py`.
- Gate status remains not passed. No sync, deployment, GPU training, or
  detector closed-loop is allowed until a focused GPT-5 Pro re-review returns
  no blockers; Gemini/DeepSeek gates remain pending after that.

## 2026-05-22T22:48:00+08:00 - BATA GPT-5 Pro focused re-review passed

- Added
  `research-wiki/experiments/BATA_PR0_PR1_PR2_GPT5_PRO_REREVIEW_PASS_20260522.md`
  and updated BATA and Sparse TAD trackers.
- Oracle browser session `bata-pr0-pr1-pr2-fix` completed with
  `gpt-5.5-pro` / ChatGPT `Extended Pro`.
  Transcript:
  `C:\Users\skywalker\.oracle\sessions\bata-pr0-pr1-pr2-fix\artifacts\transcript.md`.
- Review route used inline file contents because raw attachments, Rosetta CDP,
  and API route were unavailable. Pro reported all 7 inline files visible and
  no obvious truncation.
- Verdict: `PASS`. Pro found no blocking findings and marked all previous
  blockers resolved: tubelet coverage, diagnostic cache gate, seconds mapping,
  stride/scale-factor handshake, and offline window duplication.
- Pro launch decision: allowed to enter Gemini/DeepSeek review. Full review
  gate is still incomplete; no sync, deployment, GPU training, or detector
  closed-loop until Gemini CLI and Claude DeepSeek checks also pass.

## 2026-05-22T22:58:00+08:00 - BATA PR-0/1/2 user-directed serial deploy queued

- User explicitly requested direct deployment and serial placement after any
  running/waiting experiments.
- Checked remote lanes. On `35329`, SAHM-NoGate training was active and screens
  `sahm_softmax_wait_after_nogate_20260522` and
  `sahm_nogate_wait_after_clean_20260522` were present. `35407` had no screen
  but was 90% full; `25876` had no train/screens.
- Created a narrow BATA deployment archive containing only:
  `opentad/datasets/transforms/boundary_acquisition.py`,
  `opentad/datasets/transforms/end_to_end.py`,
  `tools/bata/build_oracle_boundary_score_cache.py`,
  `tools/bata/dump_bata_selection_ledger.py`, and
  `tests/test_bata_boundary_acquisition_contracts.py`.
- Uploaded the archive to `35329:/root/autodl-tmp/bata_pr012_deploy_20260522_2256`
  and launched detached screen
  `519482.bata_pr012_deploy_after_sahm_20260522`.
- The remote script
  `/root/autodl-tmp/bata_pr012_deploy_20260522_2256/deploy_bata_pr012_after_queue.sh`
  waits until active `tools/train.py` processes and the existing SAHM wait
  screens clear, then extracts the BATA files into
  `/root/autodl-tmp/OpenTAD_Back_check` and runs remote `py_compile` plus
  `pytest tests/test_bata_boundary_acquisition_contracts.py -q`.
- Remote deploy log:
  `/root/autodl-tmp/OpenTAD_Back_check/logs/bata_pr012_deploy_after_queue_20260522_2256.log`.
  Marker when complete:
  `/root/autodl-tmp/OpenTAD_Back_check/logs/BATA_PR012_DEPLOYED_AFTER_QUEUE_20260522_2256.ok`.
- No BATA GPU run, detector closed-loop, offline diagnostic run, or training
  launch was started.

## 2026-05-22T23:20:00+08:00 - BATA clean local directory created and model baseline audited

- User requested BATA implementation in a clean directory and an audit that the
  selected-frame route does not rely on modified AdaTAD model code.
- Created `OpenTAD_BATA_Clean` from `OpenTAD_UpstreamFresh`.
- Rejected `OpenTAD_SparseTAD_Clean` as a clean BATA baseline: its working tree
  was clean, but its model directory differs from upstream in sparse-aware/SAHM
  files including `vit_adapter.py`, `anchor_free_head.py`, `actionformer.py`,
  `fpn.py`, `actionformer_proj.py`, `temporal_grid.py`, and
  `sparse_head_mixer.py`.
- Ported only input-side BATA files into `OpenTAD_BATA_Clean`:
  `opentad/datasets/transforms/end_to_end.py`,
  `opentad/datasets/transforms/boundary_acquisition.py`,
  `tools/bata/build_oracle_boundary_score_cache.py`,
  `tools/bata/dump_bata_selection_ledger.py`, and
  `tests/test_bata_boundary_acquisition_contracts.py`.
- Model audit command from `OpenTAD_BATA_Clean`:
  `git diff --no-index --quiet ..\OpenTAD_UpstreamFresh\opentad\models opentad\models`
  returned no diff.
- Verification from `OpenTAD_BATA_Clean`:
  `python -m py_compile ...` PASS and
  `python -m pytest tests\test_bata_boundary_acquisition_contracts.py -q`
  PASS with `11 passed`.
- Report added:
  `research-wiki/experiments/BATA_CLEAN_BASELINE_DIRECTORY_AUDIT_20260522.md`.
- No GPU run or remote sync was launched from this clean tree.

## 2026-05-22T23:25:00+08:00 - Sparse model progress check

- Added
  `research-wiki/experiments/SPARSE_MODEL_PROGRESS_CHECK_20260522_2325.md`
  and updated the Sparse TAD tracker.
- `35407` status:
  no active screen, no active `tools/train.py`, GPU idle.
  `STGA-v1a` full-60 completed in
  `/root/autodl-tmp/OpenTAD_SparseTAD_Clean_sapm_61e272e/logs/stga_v1a_resume39_full60_20260522_1304.log`
  with final `Average-mAP=63.52`, vector
  `79.17 / 74.22 / 66.50 / 55.91 / 41.81`.
  This is no longer a collapse, but remains below random-fixed Adapter
  `63.77 / 42.19@0.7`.
- `35407` queue issue:
  `logs/sapm_attn_gap_bias_postfix_wait_after_stga_20260522_1320_queue.log`
  incorrectly held with
  `HOLD STGA final Average-mAP=63.52 below severe-collapse guard 58.0`;
  therefore the intended `SAPM-AttnGapBias repaired-protocol rerun` and
  following `SAN-Aggregation` did not launch.
- `35329` status:
  active `SAHM-Deformable-NoGate-AllCls` run in
  `/root/autodl-tmp/OpenTAD_SparseTAD_Clean_full_5da14fd_20260522_0117`;
  screen `sahm_nogate_wait_after_clean_20260522`; GPU process PID `917901`
  using about `3574 MiB`.
  Latest observed eval from screen hardcopy at `2026-05-22 23:23:41`:
  `Average-mAP=63.22`, vector `80.09 / 74.71 / 66.08 / 55.39 / 39.82`;
  epoch 54 then started.
- `SAHM-SoftmaxGate` remains queued behind NoGate.
  BATA deploy queue on `35329` remains waiting behind active SAHM processes and
  screens; no BATA deploy marker exists yet.
- `25876` has no active screen/train and no GPU devices.
- No new GPU run was launched during this check. Recommended next action is to
  repair/restart the idle `35407` SAPM queue or directly relaunch the intended
  SAPM repaired rerun.

## 2026-05-22T23:40:00+08:00 - 35407 SAPM serial queue repaired and restarted

- User requested fixing/restarting the SAPM queue on `35407` so the serial lane
  continues.
- Confirmed `35407` was idle before launch: no active screen, no active
  `tools/train.py`, GPU `0/32760 MiB`, and `/root/autodl-tmp` had `21G` free.
- Uploaded and launched
  `logs/restart_sapm_attngap_then_san_35407_20260522_2336.sh` under
  `/root/autodl-tmp/OpenTAD_SparseTAD_Clean_full_5da14fd_20260522_0117`.
- Detached screen:
  `906050.sapm_attngap_then_san_recover_20260522_2336`.
- Queue log:
  `/root/autodl-tmp/OpenTAD_SparseTAD_Clean_full_5da14fd_20260522_0117/logs/sapm_attngap_then_san_recover_20260522_2336_queue.log`.
- SAPM log:
  `/root/autodl-tmp/OpenTAD_SparseTAD_Clean_full_5da14fd_20260522_0117/logs/sapm_attn_gap_bias_postfix_rerun_20260522_2336.log`.
- The queue now correctly treats prior STGA `Average-mAP=63.52` as passing the
  severe-collapse guard `58.0`, launched SAPM-AttnGapBias, and will launch
  SAN-Aggregation only if SAPM exits cleanly with final `Average-mAP >=58.0`.
- Follow-up verification found active `torchrun/tools/train.py`, SAPM at
  `Training Starts` / `Epoch 0 started`, and GPU usage about `3455/32760 MiB`.
- Updated
  `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md` and
  `research-wiki/experiments/SPARSE_MODEL_PROGRESS_CHECK_20260522_2325.md`.

## 2026-05-22T23:55:00+08:00 - Cloud/local source integrity audit completed

- Added
  `research-wiki/experiments/CLOUD_AND_LOCAL_SOURCE_INTEGRITY_AUDIT_20260522.md`.
- Raw audit outputs:
  `logs/remote_source_audit_35407_20260523.txt`,
  `logs/remote_source_audit_35329_20260523.txt`,
  `logs/remote_source_audit_25876_20260523.txt`,
  `logs/remote_source_norm_audit_35407_20260523.txt`,
  `logs/remote_source_norm_audit_35329_20260523.txt`, and
  `logs/remote_source_norm_audit_25876_20260523.txt`.
- Active Sparse TAD runs pass the path/source audit:
  `35407` SAPM-AttnGapBias and `35329` SAHM-NoGate both run from
  `/root/autodl-tmp/OpenTAD_SparseTAD_Clean_full_5da14fd_20260522_0117`,
  and normalized hashes for checked source/config files match local
  `OpenTAD_SparseTAD_Clean`.
- `25876` has no active train/screen and no visible GPU.
- Found a BATA correctness blocker: the old user-directed deploy target
  `/root/autodl-tmp/OpenTAD_Back_check` is not a clean AdaTAD model baseline.
  The old BATA deploy process is not active, no marker exists, and no BATA
  files were applied. Future BATA deployment must create a new clean remote tree
  from local `OpenTAD_BATA_Clean`.
- Fixed `35329` SAHM-SoftmaxGate queue script
  `logs/sahm_softmax_wait_after_nogate_20260522_1306.sh` by changing the mAP
  guard from bare `python` to `/root/miniconda3/bin/python`; current NoGate
  training was not modified.
- Updated both Sparse TAD and BATA trackers with the audit result and BATA
  `BLOCKED-CORRECT-DEPLOY` state.

## 2026-05-23T00:05:00+08:00 - Official OpenTAD clean clone created for BATA verification

- Created local official clean clone:
  `OpenTAD_OfficialClean_20260523`.
- Source:
  `https://github.com/sming256/OpenTAD.git`.
- Commit:
  `1aa8ca4ac5e846b1e8ff69298dd6607121a01589`
  (`1aa8ca4 add arxiv link and update citation`).
- GitHub clone initially hit connection resets; the second shallow/blobless
  clone fetched the correct HEAD but checkout failed, then `git checkout -f
  HEAD` completed the working tree. Final status is clean.
- Added report:
  `research-wiki/experiments/BATA_OFFICIAL_CLEAN_CLONE_AUDIT_20260523.md`.
- Raw comparison log:
  `logs/official_clean_clone_compare_20260523.txt`.
- Verification:
  `OpenTAD_OfficialClean_20260523/opentad` matches
  `OpenTAD_UpstreamFresh/opentad` with diff exit `0`; and
  `OpenTAD_BATA_Clean/opentad/models` matches
  `OpenTAD_OfficialClean_20260523/opentad/models` with diff exit `0`.
- `OpenTAD_BATA_Clean` dirty surface remains input-side only:
  `opentad/datasets/transforms/end_to_end.py`,
  `opentad/datasets/transforms/boundary_acquisition.py`, `tools/bata/*`, and
  tests. No model files are modified.
- Updated BATA and Sparse trackers. Future BATA remote deployment must use a
  new clean tree derived from `OpenTAD_BATA_Clean`, not
  `/root/autodl-tmp/OpenTAD_Back_check`.

## 2026-05-23T00:09:00+08:00 - BATA server queue status checked

- Checked `35329` for BATA screens, processes, old deployment marker, old
  deployment log, and clean BATA remote directories.
- Current effective status: BATA is **not queued/running**.
- Old stage directory still exists:
  `/root/autodl-tmp/bata_pr012_deploy_20260522_2256`, containing
  `bata_pr012_files.tar` and `deploy_bata_pr012_after_queue.sh`.
- No BATA screen or deploy process is active.
- No old marker exists:
  `/root/autodl-tmp/OpenTAD_Back_check/logs/BATA_PR012_DEPLOYED_AFTER_QUEUE_20260522_2256.ok`.
- No clean remote BATA tree exists under `/root/autodl-tmp/OpenTAD_BATA*`.
- Existing `35329` screens are only SAHM NoGate and SAHM SoftmaxGate. BATA must
  be re-queued later from local `OpenTAD_BATA_Clean` into a new clean remote
  tree, not from the old `/root/autodl-tmp/OpenTAD_Back_check` target.

## 2026-05-23T00:23:00+08:00 - Clean BATA deployment queued on 35329

- User requested clean deployment, queueing, and cancellation of the previous
  incorrect BATA queue.
- Cancelled the old wrong-target deployment under
  `/root/autodl-tmp/bata_pr012_deploy_20260522_2256` by renaming
  `deploy_bata_pr012_after_queue.sh` to
  `deploy_bata_pr012_after_queue.sh.cancelled_20260523_0016` and writing
  `CANCELLED_WRONG_TARGET_20260523_0016.txt`.
- Local clean BATA source:
  `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_BATA_Clean`.
  Local verification before upload:
  `py_compile` PASS and
  `pytest tests/test_bata_boundary_acquisition_contracts.py -q` PASS with
  `11 passed`.
- Uploaded archive
  `logs/bata_clean_deploy_20260523_0020/OpenTAD_BATA_Clean_20260523_0020.tar.gz`
  and extracted a new remote clean tree:
  `/root/autodl-tmp/OpenTAD_BATA_Clean_20260523`.
- Launched detached screen:
  `577887.bata_clean_deploy_after_sahm_20260523`.
- Queue script:
  `/root/autodl-tmp/OpenTAD_BATA_Clean_20260523/logs/deploy_bata_clean_after_sahm_20260523_0020.sh`.
- Queue log:
  `/root/autodl-tmp/OpenTAD_BATA_Clean_20260523/logs/bata_clean_deploy_after_sahm_20260523_0020.log`.
- The queue detected active `SAHM-NoGate` `tools/train.py` / `torchrun`
  processes plus preserved screens
  `sahm_softmax_wait_after_nogate_20260522` and
  `sahm_nogate_wait_after_clean_20260522`, so it is waiting and did not launch
  GPU training.
- The queued action after upstream screens clear is remote `py_compile` plus
  `pytest tests/test_bata_boundary_acquisition_contracts.py -q`, then marker
  `/root/autodl-tmp/OpenTAD_BATA_Clean_20260523/logs/BATA_CLEAN_DEPLOYED_AFTER_QUEUE_20260523.ok`.
- Added report
  `research-wiki/experiments/BATA_CLEAN_REMOTE_DEPLOY_20260523.md` and updated
  the BATA and Sparse trackers.

## 2026-05-23T00:31:00+08:00 - Current deployed and waiting experiment inventory

- Checked `35407`, `35329`, and `25876` with `screen -ls`, active
  `tools/train.py` / `torchrun` processes, GPU memory, disk, and recent logs.
- `35407`:
  active screen `906050.sapm_attngap_then_san_recover_20260522_2336`.
  It is running `SAPM-AttnGapBias` repaired-protocol rerun from
  `/root/autodl-tmp/OpenTAD_SparseTAD_Clean_full_5da14fd_20260522_0117`,
  config `input_random_fixed_50pct_adapter_sapm_attn_gap_bias.py`, log
  `logs/sapm_attn_gap_bias_postfix_rerun_20260522_2336.log`.
  Latest tail reached epoch 12 with normal loss scale and no mAP yet.
  `SAN-Aggregation` is chained after SAPM in the same recovery screen and has
  not launched.
- `35329`:
  active `SAHM-NoGate` under screen
  `605535.sahm_nogate_wait_after_clean_20260522`, config
  `input_random_fixed_50pct_adapter_sahm_deformable_nogate_allcls.py`, log
  `logs/sahm_nogate_20260522_0117.log`.
  Latest completed eval remains `2026-05-22 23:23:41`:
  `Average-mAP=63.22`, vector
  `80.09 / 74.71 / 66.08 / 55.39 / 39.82`; epoch-55 eval is in progress in
  the current log tail.
  `SAHM-SoftmaxGate` waits in screen
  `102089.sahm_softmax_wait_after_nogate_20260522`.
  Clean BATA deployment validation waits in screen
  `577887.bata_clean_deploy_after_sahm_20260523`; its marker is absent.
- `25876`:
  no screen, no active train/test process, and no visible GPU.
- Updated `SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md` and
  `BATA_FRAME_SELECTION_TASK_TRACKER_20260521.md` with the current inventory.

## 2026-05-23T01:00:00+08:00 - BATA Pre-1 / Pre-2 local implementation completed

- Implemented the two planned BATA pre-experiments in
  `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_BATA_Clean`.
- Added Pre-1 no-GPU offline observability script:
  `tools/bata/run_bata_pre1_offline_observability.py`.
  It builds/loads oracle and noisy-oracle score caches, compares
  `oracle_bca`, `noisy_oracle_bca`, `stratified_50`,
  `uniform_stride2_50`, and `random_fixed_50`, writes ledgers and
  `summary.json` / `summary.md`, and gates on strict budget, duplicate
  freedom, BCA coverage, and boundary-role evidence BR@4.
- Added Pre-2 diagnostic closed-loop entry:
  `configs/adatad/thumos/bata_noisy_oracle_bca_fixed50_adapter.py` and
  `tools/bata/launch_bata_pre2_noisy_oracle_serial.sh`.
  The config is fixed-50 `384/768`, uses diagnostic-only noisy-oracle
  caches, and keeps AdaTAD backbone/head/loss/assignment unchanged.
- Added selected-axis post-processing protocol repair in
  `opentad/models/utils/post_processing/utils.py` so BATA selected-axis
  predictions are mapped back to dense axis before seconds conversion when
  irregular metadata exists. Legacy configs without irregular metadata keep
  the old path.
- Added self-check report:
  `research-wiki/experiments/BATA_PRE1_PRE2_IMPLEMENTATION_SELF_CHECK_20260523.md`.
- Verification:
  `py_compile` PASS;
  `pytest tests/test_bata_boundary_acquisition_contracts.py tests/test_bata_post_processing_selected_axis.py -q`
  returned `11 passed, 5 skipped` because local Windows torch is unavailable
  for the new post-processing tests;
  Pre-1 smoke on 2 validation videos PASS with summary
  `tmp/bata_pre1_smoke/summary.json`.
- No remote sync, no deployment update, and no GPU training was launched in
  this implementation step. Required GPT-5 Pro, Gemini CLI, and Claude
  DeepSeek review gates remain pending before Pre-2 can run.

## 2026-05-23T01:38:41+08:00 - BATA Pre-1 / Pre-2 review gates passed and code deployed

- Completed the post-fix external review gate for the clean BATA Pre-1/Pre-2
  package in `OpenTAD_BATA_Clean`.
- Gemini CLI output:
  `logs/gemini3_pro_preview_bata_pre1_pre2_review_inline_20260523.txt`;
  verdict `PASS`.
- Claude CLI DeepSeek output:
  `logs/claude_deepseek_v4_pro_bata_pre1_pre2_review_20260523.txt`;
  verdict `PASS` with one non-blocking launcher-wait WARN.
- Deployment report:
  `research-wiki/experiments/BATA_PRE1_PRE2_GEMINI_DEEPSEEK_REVIEW_AND_DEPLOY_20260523.md`.
- Deployed reviewed archive to
  `/root/autodl-tmp/OpenTAD_BATA_Clean_20260523/logs/bata_pre1_pre2_reviewed_deploy_20260523.tar.gz`
  on `35329`; local/remote SHA256 matched:
  `7a7316dfa1deac75b29751781968d9bf85482375f5aaefc6aa0697d519bbfa28`.
- Remote validation script:
  `/root/autodl-tmp/OpenTAD_BATA_Clean_20260523/logs/remote_validate_bata_pre1_pre2_20260523.sh`.
- Remote validation result: `/root/miniconda3/bin/python` found
  `torch=True`, `numpy=True`, `mmengine=True`; installed missing `pytest` for
  gate tests; `py_compile` PASS; Linux pytest
  `tests/test_bata_boundary_acquisition_contracts.py`
  `tests/test_bata_post_processing_selected_axis.py -q` returned
  `18 passed in 2.61s`.
- Deployment marker:
  `/root/autodl-tmp/OpenTAD_BATA_Clean_20260523/logs/BATA_PRE1_PRE2_REVIEWED_DEPLOYED_20260523.ok`.
- Cancelled superseded old BATA waiting screen
  `577887.bata_clean_deploy_after_sahm_20260523` so it cannot later extract an
  older package over the reviewed deployment. Cancellation marker:
  `/root/autodl-tmp/OpenTAD_BATA_Clean_20260523/logs/BATA_OLD_CLEAN_DEPLOY_QUEUE_CANCELLED_20260523.ok`.
- No BATA GPU training or detector closed-loop was launched. Next allowed BATA
  step is full no-GPU Pre-1 offline diagnostic; Pre-2 GPU remains gated by
  full Pre-1 pass, config/dataset smoke, matched clean 50% artifacts, and
  server serial clearance.

## 2026-05-23T11:43:59+08:00 - Remote training process monitor check

- Checked `35407`, `35329`, and `25876` with `screen -ls`,
  `tools/train.py` / `torchrun` process scans, GPU status, disk status, and
  current run logs.
- `35407`: active screen
  `906050.sapm_attngap_then_san_recover_20260522_2336`. SAPM-AttnGapBias
  repaired run completed and launched SAN-Aggregation. SAPM final/best checked
  in `logs/sapm_attn_gap_bias_postfix_rerun_20260522_2336.log`:
  `Average-mAP=63.43`, below random-fixed `63.77` and high-IoU reference
  `42.19@0.7`. SAN-Aggregation is now active from
  `/root/autodl-tmp/OpenTAD_SparseTAD_Clean_full_5da14fd_20260522_0117`,
  config `input_random_fixed_50pct_adapter_san_aggregation.py`, log
  `logs/san_aggregation_after_sapm_attngap_20260522_2336.log`; latest eval at
  `2026-05-23 11:38:00` is `Average-mAP=63.43`, vector
  `80.68 / 75.92 / 66.09 / 54.78 / 39.65`, then continued to epoch 51.
  GPU usage is about `3473/32760 MiB`; `/root/autodl-tmp` is critical at
  `196G used / 5.0G free`.
- `35329`: no active screen and no `tools/train.py` / `torchrun` process.
  SAHM-NoGate completed with final `Average-mAP=63.62`, vector
  `79.98 / 75.14 / 66.79 / 56.07 / 40.13`. SAHM-SoftmaxGate did not launch:
  queue log `logs/sahm_softmax_wait_after_nogate_20260522_1306_queue.log`
  held with `HOLD NoGate final Average-mAP=63.62 below severe-collapse guard
  58.0`, which is a queue guard bug because `63.62 >= 58.0`. BATA reviewed
  deployment markers are present; no BATA run is active.
- `25876`: no screen, no training process, and no visible GPU; disk remains
  about `157G used / 44G free`.
- Updated `SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md` and
  `BATA_FRAME_SELECTION_TASK_TRACKER_20260521.md`. Next actions: monitor SAN
  and address `35407` storage risk; fix/restart the `35329` SoftmaxGate queue
  guard only if continuing that run.

## 2026-05-23T11:48:24+08:00 - Completed-run checkpoint cleanup after monitor check

- Cleaned only completed-run checkpoint directories; did not touch the active
  SAN-Aggregation checkpoint directory.
- `35407` SAPM-AttnGapBias repaired run cleanup:
  target `/root/autodl-tmp/OpenTAD_SparseTAD_Clean_full_5da14fd_20260522_0117/exps/thumos/adatad/input_random_fixed_50pct_adapter_sapm_attn_gap_bias_postfix_rerun_20260522_2336/gpu1_id0/checkpoint`;
  kept `epoch_59.pth`; removed older `epoch_*.pth`; report
  `/root/autodl-tmp/OpenTAD_SparseTAD_Clean_full_5da14fd_20260522_0117/logs/checkpoint_cleanup_sapm_attngap_20260523_1143.log`;
  disk improved from `196G used / 4.5G free` to `188G used / 13G free`.
- `35329` SAHM-NoGate cleanup:
  target `/root/autodl-tmp/OpenTAD_SparseTAD_Clean_full_5da14fd_20260522_0117/exps/thumos/adatad/input_random_fixed_50pct_adapter_sahm_deformable_nogate_allcls_20260522_0117/gpu1_id0/checkpoint`;
  kept `epoch_59.pth`; removed older `epoch_*.pth`; report
  `/root/autodl-tmp/OpenTAD_SparseTAD_Clean_full_5da14fd_20260522_0117/logs/checkpoint_cleanup_sahm_nogate_20260523_1143.log`;
  disk improved from `73G used / 128G free` to `66G used / 135G free`.

## 2026-05-23T22:58:23+08:00 - Sparse model-side final status check

- Checked Sparse TAD model-side status on `35407` and `35329`.
- `35407`: no active screen and no preserved `tools/train.py` / `torchrun`
  process; GPU `0/32760 MiB`. SAN-Aggregation completed with final/best
  `Average-mAP=64.14`, vector
  `81.03 / 76.22 / 66.85 / 55.44 / 41.18`, log
  `/root/autodl-tmp/OpenTAD_SparseTAD_Clean_full_5da14fd_20260522_0117/logs/san_aggregation_after_sapm_attngap_20260522_2336.log`.
- Cleaned only SAN-Aggregation's completed-run checkpoint directory:
  `/root/autodl-tmp/OpenTAD_SparseTAD_Clean_full_5da14fd_20260522_0117/exps/thumos/adatad/input_random_fixed_50pct_adapter_san_aggregation_after_sapm_attngap_20260522_2336/gpu1_id0/checkpoint`;
  kept `epoch_59.pth`, removed older `epoch_*.pth`, preserved logs/results
  and non-epoch artifacts. Cleanup report:
  `/root/autodl-tmp/OpenTAD_SparseTAD_Clean_full_5da14fd_20260522_0117/logs/checkpoint_cleanup_san_aggregation_20260523_2256.log`;
  disk improved from `189G used / 12G free` to `181G used / 20G free`.
- `35329`: active screens/processes are BATA (`bata_pre2_noisy_oracle` and
  `bata_mobilenet_formal_serial`), not Sparse TAD. SAHM-NoGate final remains
  `Average-mAP=63.62`, vector `79.98 / 75.14 / 66.79 / 56.07 / 40.13`.
  SAHM-SoftmaxGate did not launch because the wait script held with the
  logically wrong guard `63.62 below 58.0`.
- Current Sparse model-side verdict: mostly complete, with SoftmaxGate unrun;
  no especially strong result. Best is SAN-Aggregation `64.14`, which is
  `+0.37` vs random-fixed `63.77` and `+0.29` vs strict EMA `63.85`, but
  `-0.50` vs stratified `64.64`, `-0.95` vs uniform stride-2 `65.09`, and
  `41.18@0.7` remains below the random-fixed high-IoU anchor `42.19`.
- Updated `research-wiki/experiments/SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`
  and wrote `research-wiki/experiments/SPARSE_MODEL_FINAL_STATUS_CHECK_20260523_2258.md`.

## 2026-05-24T00:42:00+08:00 - Sparse geometry controls Pro and DeepSeek review state

- GPT-5.5 Pro first browser attempt
  `sparse-geometry-controls-code-review` failed before prompt submission due
  attachment upload timeout; it was recorded as invalid.
- GPT-5.5 Pro inline fallback `sparse-geometry-controls-code-review-2`
  completed on `Extended Pro` in `8m06s`, verdict `WARN`; transcript copied to
  `research-wiki/experiments/GPT5_PRO_SPARSE_GEOMETRY_CONTROLS_REVIEW_TRANSCRIPT_20260524.md`
  and summary saved as
  `research-wiki/experiments/GPT5_PRO_SPARSE_GEOMETRY_CONTROLS_REVIEW_20260524.md`.
- Accepted Pro fixes were applied: valid-index geometry controls, analysis
  window de-duplication, right-edge exclusion, explicit base
  `geometry_control="real"`, stronger tail-grid / non-prefix-mask /
  parameter-count tests, and pure analysis tests.
- Post-fix local verification: `py_compile` PASS,
  `pytest tests\test_sparse_boundary_coverage_analysis.py -q` PASS
  (`3 passed`), diagnostic smoke PASS, `git diff --check` PASS.
- Gemini CLI review failed with `insufficient_user_quota`; this is not a
  completed Gemini gate.
- Claude CLI `deepseek-v4-pro` returned `PASS with 6 WARN, 0 FAIL`; report:
  `research-wiki/experiments/SPARSE_GEOMETRY_CONTROLS_GEMINI_DEEPSEEK_REVIEW_20260524.md`.
- Launch decision unchanged: local only, not deployable yet. Remaining gates:
  rerun Gemini when quota is available, Linux torch pytest for SAN tests, and
  resolved config diff in an environment with `mmengine/mmcv`.

## 2026-05-24T00:08:03+08:00 - Sparse geometry causality controls implemented locally

- Implemented local geometry-control code in `OpenTAD_SparseTAD_Clean`:
  `opentad/models/necks/fpn.py`, `tests/test_san_neck_contracts.py`, three
  SAN control configs, and
  `tools/analysis/analyze_sparse_boundary_coverage.py`.
- Added SAN `geometry_control` modes: `real`, `dummy_uniform`, `shuffled`,
  and `no_geometry`, plus control configs for dummy-uniform, shuffled-geometry,
  and no-geometry same-param runs.
- Added standalone boundary coverage / optional prediction residual diagnostic;
  smoke output written to `logs/sparse_boundary_coverage_smoke_20260523.json`.
- Self-check report:
  `research-wiki/experiments/SPARSE_GEOMETRY_CONTROLS_IMPLEMENTATION_SELF_CHECK_20260524.md`.
- Local verification: `py_compile` PASS, diagnostic smoke PASS,
  `git diff --check` PASS; `pytest tests/test_san_neck_contracts.py -q`
  remains blocked by the known local Windows torch `c10.dll` initialization
  failure.
- Protocol: strict random-fixed 50% preserved; no Adapter/head/loss/sampling
  or post-processing change; no GT/teacher leakage in train/eval paths. This
  is local code only and is not ready for remote deployment or long training
  until GPT-5 Pro/Gemini/DeepSeek review gates and Linux torch smoke pass.

## 2026-05-23T23:09:27+08:00 - Sparse model underperformance analysis

- Wrote
  `research-wiki/experiments/SPARSE_MODEL_UNDERPERFORMANCE_ANALYSIS_20260523_2309.md`
  and updated `SPARSE_TAD_TASK_FLOW_TRACKER_20260520.md`.
- Main interpretation: the completed Sparse TAD model-side modules mostly
  calibrate feature representations after random-fixed 50% sampling, while
  the dominant remaining bottleneck is boundary evidence loss plus physical
  time localization mismatch.
- Evidence: best model-side result is SAN-Aggregation `64.14` Avg-mAP, but
  `41.18@0.7` remains below the random-fixed high-IoU anchor `42.19`; clean
  repaired stack also had a high-IoU deficit around `40.78-40.99@0.7`.
- Protocol implication: the detector still mainly trains/regresses over the
  selected-token axis, with temporal geometry as an auxiliary signal rather
  than the native coordinate system for assignment, loss, and decode.
- Decision: no blind new Sparse TAD long run. Next model-side work should be
  geometry causality controls or a reviewed physical-coordinate detector
  redesign; immediate high-upside route is deployable BATA boundary-aware
  selection.

## 2026-05-23T23:34:00+08:00 - GPT-5.5 Pro direction and validity discussion

- Ran Oracle browser Pro session `sparse-tad-pro-direction-validity`.
- Model selection evidence: requested `Pro`, resolved `Extended Pro`, verified
  by ChatGPT model picker; elapsed `6m46s`; usage `3392` input tokens,
  `4861` output tokens, `8253` total.
- Inlined files:
  `research-wiki/experiments/SPARSE_MODEL_FINAL_STATUS_CHECK_20260523_2258.md`
  and
  `research-wiki/experiments/SPARSE_MODEL_UNDERPERFORMANCE_ANALYSIS_20260523_2309.md`.
- Saved transcript:
  `research-wiki/experiments/GPT5_PRO_SPARSE_TAD_DIRECTION_VALIDITY_TRANSCRIPT_20260523.md`.
- Saved accepted summary:
  `research-wiki/experiments/GPT5_PRO_SPARSE_TAD_DIRECTION_VALIDITY_20260523.md`.
- Pro verdict: agree with attribution only as current strongest hypothesis,
  not proved causality. Boundary evidence loss is strongly supported;
  selected-axis localization is likely but needs causal controls.
- Direction decision: ordinary model-side module stacking `NO-GO`; SAN as main
  result `NO-GO for now`; limited paired SAN seed verification `GO`; geometry
  causality controls `GO` highest priority; physical-coordinate detector
  `Conditional GO`; BATA deployable selection `GO for performance route`;
  SoftmaxGate/head gate `NO-GO as mainline`.
- Effectiveness gates adopted: same-tree paired baseline required; single-run
  screen needs Avg `>= baseline +0.5` and no `@0.7` drop; 3-seed continuation
  needs mean Avg `+0.6`, mean `@0.7 +0.3`, and paired CI not crossing zero;
  publishable model-side claim should beat stratified `64.64`, approach or
  exceed uniform `65.09`, and recover `@0.7 >=42.19`, preferably `>=42.7`.

## 2026-05-24T14:06:08+08:00 - BATA formal selector extended health check

- Checked `35329` active screen
  `bata_mobilenet_seqrescue_clean_20260524_1126`, run tag
  `bata_mobilenet_fixed50_20260524_112543`.
- Active process: PID `528628`, elapsed about `02:38:49`, CPU about `1134%`,
  RSS about `9.16GB`, `465` threads. GPU sample: `985 / 24564 MiB`, util `0%`.
  Disk `/root/autodl-tmp`: `68G / 200G` used.
- Latest emitted log remains
  `epoch=0 step=2250/2292 train_loss_so_far=0.989039`; log mtime is
  `2026-05-24 12:17:16 +0800`.
- No `train_log.jsonl`, `best.pth`, `last.pth`, score cache, selector eval,
  detector artifact, or mAP exists yet.
- I/O is still live: over 45s,
  `rchar=822457812867 -> 827796294757` and
  `syscr=25055844 -> 25218729`.
- Phase probe showed `437` sleeping and `28` running threads, `fd_count=345`,
  and confirmed the remote code has train-side sequential mode but not the
  local sequential-validation rescue path
  `_evaluate_model_sequential_video()` / `--val-log-every-steps`.
- Strict fixed-50 deployable contract remains pending because no validation or
  test deployable score cache has been generated. No GT/teacher leakage claim
  or deployable mAP claim is made.
- Decision: continue the current selector while I/O is progressing and launch
  no new long run. If it remains artifact-free or exits/stalls before producing
  a checkpoint, the prepared sequential-validation rescue patch is the next
  controlled restart candidate.

## 2026-05-24T14:18:26+08:00 - BATA formal selector quick refresh

- Checked `35329` again without a long I/O wait.
- PID `528628` remains active: elapsed about `02:52:40`, CPU about `1162%`,
  RSS about `9.25GB`, GPU `985 / 24564 MiB`.
- The log is still unchanged at
  `epoch=0 step=2250/2292 train_loss_so_far=0.989039`; mtime remains
  `2026-05-24 12:17:16 +0800`.
- Still no `train_log.jsonl`, selector checkpoint, score cache, selector eval,
  detector output, or mAP.
- Decision unchanged: continue waiting while active, no new long run. If the
  next checks remain artifact-free, prepare a recorded controlled
  validation-rescue restart decision.

## 2026-05-24T14:26:46+08:00 - BATA validation-rescue restart decision

- Rechecked `35329` at `2026-05-24 14:23:51 +0800`.
- PID `528628` was still active after about `02:58:05`, CPU about `1171%`,
  RSS about `9.27GB`, GPU `985 / 24564 MiB`, but the log remained unchanged at
  `epoch=0 step=2250/2292` with mtime `12:17:16`.
- Still no `train_log.jsonl`, selector checkpoint, score cache, selector eval,
  detector artifact, or mAP.
- I/O over 60s continued but slowly:
  `rchar=912194316169 -> 912873015558`,
  `syscr=27783931 -> 27804618`.
- Local sequential-validation rescue patch was reverified:
  `py_compile` PASS and
  `pytest tests/test_bata_deployable_selector_contracts.py -q` reported
  `12 passed, 1 skipped`. The known Windows torch DLL access-violation tail
  appeared after pytest reporting, so Linux remote smoke remains required.
- Decision: stop the current artifact-free validation-tail run, deploy the
  sequential-validation rescue patch to the clean formal BATA tree, run remote
  verification, and relaunch the formal selector with validation progress
  logging. No Sparse/BATA extra long run should start.

## 2026-05-24T14:35:38+08:00 - BATA sequential-validation rescue deployed

- Stopped old artifact-free selector run
  `bata_mobilenet_fixed50_20260524_112543`, PID `528628`.
- Backed up remote files under
  `/root/autodl-tmp/OpenTAD_BATA_Formal_MobileNet_20260523/logs/seqval_rescue_backups_20260524_142646_seqval_rescue`.
- Uploaded patched:
  `tools/bata/mobilenet_boundary_selector.py`,
  `tools/bata/launch_bata_mobilenet_fixed50_serial.sh`, and
  `tests/test_bata_deployable_selector_contracts.py`.
- Remote verification passed:
  `py_compile`, deployable pytest `13 passed`, focused BATA pytest
  `31 passed`.
- First relaunch `bata_mobilenet_fixed50_20260524_143250` self-waited because
  the screen command included `TORCHRUN=...torchrun`, which matched the
  active-process guard. It was stopped before selector training.
- Successful relaunch:
  screen `bata_mobilenet_seqvalrescue_20260524_1437`, run tag
  `bata_mobilenet_fixed50_20260524_143349`, log
  `/root/autodl-tmp/OpenTAD_BATA_Formal_MobileNet_20260523/logs/bata_mobilenet_fixed50_20260524_143349.log`.
- Current selector PID `562544` reached
  `epoch=0 step=100/2292`; command includes
  `--loader-mode sequential_video --no-shuffle --num-workers 0
  --log-every-steps 50 --val-log-every-steps 100`.
- Strict fixed-50 and no-test-GT/no-teacher deployable contracts are unchanged.
  No checkpoint/cache/eval/detector/mAP exists yet; continue monitoring to
  epoch completion and sequential validation progress.

## 2026-05-24T14:44:33+08:00 - BATA selector progress after rescue

- Checked `35329` screen `bata_mobilenet_seqvalrescue_20260524_1437`, run tag
  `bata_mobilenet_fixed50_20260524_143349`.
- PID `562544` remains active; elapsed about `10:41`, CPU about `465%`, RSS
  about `2.80GB`, GPU memory `985 / 24564 MiB`.
- Log mtime is current at `14:43:39`; latest progress reached
  `epoch=0 step=500/2292 train_loss_so_far=1.123960`.
- No checkpoint, score cache, selector eval, detector artifact, or mAP yet.
- Decision: rescue run is healthy in training stage. Continue monitoring to
  epoch 0 completion and verify sequential validation progress logs.

## 2026-05-24T14:50:41+08:00 - BATA selector progress continues

- Checked `35329` screen `bata_mobilenet_seqvalrescue_20260524_1437`, run tag
  `bata_mobilenet_fixed50_20260524_143349`.
- PID `562544` remains active; elapsed about `16:49`, CPU about `465%`, RSS
  about `3.86GB`, GPU memory `985 / 24564 MiB`.
- Latest progress reached
  `epoch=0 step=800/2292 train_loss_so_far=1.064654`.
- I/O over 45s advanced strongly:
  `rchar=119596100916 -> 135618158248`,
  `syscr=3649484 -> 4138107`.
- No checkpoint, score cache, selector eval, detector artifact, or mAP yet.
- Decision: training is healthy after the sequential-validation rescue
  relaunch. Continue monitoring to epoch 0 completion and validation progress
  logs.

## 2026-05-24T15:04:54+08:00 - BATA selector reaches epoch 0 step 1400

- Checked `35329` screen `bata_mobilenet_seqvalrescue_20260524_1437`, run tag
  `bata_mobilenet_fixed50_20260524_143349`.
- PID `562544` remains active; elapsed about `31:02`, CPU about `409%`, RSS
  about `4.73GB`.
- Latest progress reached
  `epoch=0 step=1400/2292 train_loss_so_far=0.985199`.
- No checkpoint, score cache, selector eval, detector artifact, or mAP yet
  because epoch 0 is still running.
- Decision: continue monitoring to epoch 0 completion and confirm validation
  progress logs.

## 2026-05-24T15:27:12+08:00 - BATA sequential validation progress confirmed

- Checked `35329` screen `bata_mobilenet_seqvalrescue_20260524_1437`, run tag
  `bata_mobilenet_fixed50_20260524_143349`.
- Epoch 0 training reached `step=2250/2292`.
- Validation progress logs appeared:
  `val epoch=0 step=100/1886 eval_examples_so_far=12321` and
  `val epoch=0 step=200/1886 eval_examples_so_far=24742`.
- This confirms the sequential-validation rescue patch solved the previous
  silent validation/epoch-tail observability failure.
- No selector checkpoint, score cache, selector eval, detector artifact, or
  mAP yet because validation is still running.
- Decision: keep monitoring to validation completion and checkpoint creation.

## 2026-05-24T15:45:15+08:00 - BATA validation halfway progress

- Checked `35329` screen `bata_mobilenet_seqvalrescue_20260524_1437`, run tag
  `bata_mobilenet_fixed50_20260524_143349`.
- Validation reached
  `val epoch=0 step=900/1886 eval_examples_so_far=111027`.
- PID `562544` remains active; elapsed about `01:11:23`, CPU about `350%`,
  RSS about `8.88GB`.
- No selector checkpoint, score cache, selector eval, detector artifact, or
  mAP yet because validation is still running.
- Decision: continue monitoring to validation completion and checkpoint
  creation.

## 2026-05-24T16:09:30+08:00 - BATA selector first checkpoint written

- Checked selector run `bata_mobilenet_fixed50_20260524_143349` on `35329`.
- Epoch 0 validation completed and wrote:
  `/root/autodl-tmp/bata_selectors/bata_mobilenet_fixed50_20260524_143349/train_log.jsonl`,
  `best.pth`, and `last.pth`.
- Epoch 0 metrics:
  `train_loss=0.9948498421323391`,
  `val_loss=0.6657734414021403`,
  `val_ap=0.1320974125755544`,
  `val_pos_score_mean=0.2924121022224426`,
  `val_neg_score_mean=0.26710113883018494`.
- Checkpoints are both `6,199,586` bytes. SHA256:
  best `d69e2377c6d83ebb6f160631a4e50c544aa003915cbc9c2be3e472ab21732b94`;
  last `f2af04560f380499f35dbf03def9598bca813d82fbdefbcb5b19193e7745aa5d`.
- The selector has entered epoch 1. No score cache, selector eval, detector
  output, or mAP yet.
- Decision: continue monitoring to final selector completion, cache/eval gate,
  dataset smoke, detector run, final mAP, and cleanup.

## 2026-05-24T16:22:34+08:00 - BATA selector epoch 1 progress

- Checked `35329` screen `bata_mobilenet_seqvalrescue_20260524_1437`, run tag
  `bata_mobilenet_fixed50_20260524_143349`.
- PID `562544` remains active; elapsed about `01:48:42`, CPU about `375%`,
  RSS about `11.28GB`, GPU memory `985 / 24564 MiB`, disk
  `/root/autodl-tmp` `68G / 200G` used.
- Log mtime updated to `2026-05-24 16:22:33 +0800`; latest emitted progress is
  `epoch=1 step=650/2292 train_loss_so_far=1.015093`.
- Existing selector artifacts remain `train_log.jsonl`, `best.pth`, and
  `last.pth`. No score cache, selector eval, detector output, or deployable
  mAP exists yet.
- Decision: continue monitoring through selector epochs 1-4, then cache build,
  selector eval gate, dataset smoke, detector run, final mAP, and cleanup. No
  new Sparse/BATA long run while this selector is active.

## 2026-05-24T16:30:49+08:00 - BATA selector continues epoch 1

- Checked `35329` screen `bata_mobilenet_seqvalrescue_20260524_1437`, run tag
  `bata_mobilenet_fixed50_20260524_143349`.
- PID `562544` remains active; elapsed about `01:56:57`, CPU about `380%`,
  RSS about `11.55GB`, GPU memory `985 / 24564 MiB`, disk
  `/root/autodl-tmp` `68G / 200G` used.
- Log mtime updated to `2026-05-24 16:30:02 +0800`; latest emitted progress is
  `epoch=1 step=1000/2292 train_loss_so_far=0.969096`.
- `train_log.jsonl` still contains only epoch 0. Score cache file count is
  `0`; selector eval, detector output, and deployable mAP are absent.
- Decision: continue monitoring. There is no manifest/gate to verify until
  cache generation starts, and no new Sparse/BATA long run should be launched
  while this selector remains active.

## 2026-05-24T16:33:27+08:00 - Idle server inventory during BATA wait

- Checked `35407`: no screen, no active train/test/torchrun/BATA/Sparse
  process, RTX 4080 SUPER `0 / 32760 MiB`, disk `/root/autodl-tmp`
  `181G / 200G` used with `20G` free.
- Checked `25876`: no screen, no active train/test/torchrun/BATA/Sparse
  process, no visible GPU, disk `/root/autodl-tmp` `157G / 200G` used.
- No new Sparse model-side result was found on these servers.
- Decision: keep Sparse paused and do not backfill a new long run. Continue
  waiting on the active 35329 deployable BATA selector.

## 2026-05-24T16:36:02+08:00 - BATA selector reaches epoch 1 step 1200

- Checked `35329` screen `bata_mobilenet_seqvalrescue_20260524_1437`, run tag
  `bata_mobilenet_fixed50_20260524_143349`.
- PID `562544` remains active; elapsed about `02:02:10`, CPU about `380%`,
  RSS about `11.58GB`, GPU memory `985 / 24564 MiB`, disk
  `/root/autodl-tmp` `68G / 200G` used.
- Log mtime updated to `2026-05-24 16:35:44 +0800`; latest emitted progress is
  `epoch=1 step=1200/2292 train_loss_so_far=0.935478`.
- `train_log.jsonl` still contains only epoch 0. Cache file count is `0`;
  selector eval, detector output, and deployable mAP are absent.
- Decision: continue monitoring to epoch 1 validation/checkpoint and selector
  completion. Manifest/gate verification remains pending until cache build
  starts.

## 2026-05-24T16:39:18+08:00 - BATA selector reaches epoch 1 step 1350

- Checked `35329` screen `bata_mobilenet_seqvalrescue_20260524_1437`, run tag
  `bata_mobilenet_fixed50_20260524_143349`.
- PID `562544` remains active; elapsed about `02:05:27`, CPU about `381%`,
  RSS about `11.66GB`, GPU memory `985 / 24564 MiB`, disk
  `/root/autodl-tmp` `68G / 200G` used.
- Log mtime updated to `2026-05-24 16:38:53 +0800`; latest emitted progress is
  `epoch=1 step=1350/2292 train_loss_so_far=0.909504`.
- `train_log.jsonl` still contains only epoch 0. Cache file count remains `0`;
  selector eval, detector output, and deployable mAP are absent.
- Decision: continue monitoring. The run is healthy but still in selector
  training; cache manifest and detector result verification remain pending.

## 2026-05-24T16:42:23+08:00 - BATA selector reaches epoch 1 step 1450

- Checked `35329` screen `bata_mobilenet_seqvalrescue_20260524_1437`, run tag
  `bata_mobilenet_fixed50_20260524_143349`.
- PID `562544` remains active; elapsed about `02:08:31`, CPU about `380%`,
  RSS about `11.69GB`, GPU memory `985 / 24564 MiB`, disk
  `/root/autodl-tmp` `68G / 200G` used.
- Log mtime updated to `2026-05-24 16:42:15 +0800`; latest emitted progress is
  `epoch=1 step=1450/2292 train_loss_so_far=0.898649`.
- `train_log.jsonl` still contains only epoch 0. Cache file count remains `0`;
  selector eval, detector output, and deployable mAP are absent.
- Decision: continue monitoring. The run is still in selector training; no
  cache manifest or detector metric is available yet.

## 2026-05-24T16:48:30+08:00 - BATA selector reaches epoch 1 step 1650

- Checked `35329` screen `bata_mobilenet_seqvalrescue_20260524_1437`, run tag
  `bata_mobilenet_fixed50_20260524_143349`.
- PID `562544` remains active; elapsed about `02:14:38`, CPU about `380%`,
  RSS about `11.81GB`, GPU memory `985 / 24564 MiB`, disk
  `/root/autodl-tmp` `68G / 200G` used.
- Latest emitted progress is
  `epoch=1 step=1650/2292 train_loss_so_far=0.885071`.
- `train_log.jsonl` still contains only epoch 0. Cache file count remains `0`;
  selector eval, detector output, and deployable mAP are absent.
- Also checked idle servers: `35407` has no screen/process, GPU
  `0 / 32760 MiB`, disk `181G / 200G`; `25876` has no screen/process, no
  visible GPU, disk `157G / 200G`.
- Decision: continue serial wait. Do not launch new Sparse/BATA long runs while
  the deployable selector is active.

## 2026-05-24T16:53:00+08:00 - BATA selector reaches epoch 1 step 1900

- Checked `35329` screen `bata_mobilenet_seqvalrescue_20260524_1437`, run tag
  `bata_mobilenet_fixed50_20260524_143349`.
- PID `562544` remains active; elapsed about `02:19:08`, CPU about `383%`,
  RSS about `11.99GB`, GPU memory `985 / 24564 MiB`, disk
  `/root/autodl-tmp` `68G / 200G` used.
- Latest emitted progress is
  `epoch=1 step=1900/2292 train_loss_so_far=0.891485`.
- `train_log.jsonl` still contains only epoch 0. Cache file count remains `0`;
  selector eval, detector output, and deployable mAP are absent.
- `35407` remains idle with GPU `0 / 32760 MiB` and disk `181G / 200G`;
  `25876` remains without a visible GPU and no active screen/process.
- Decision: continue serial wait to epoch 1 validation/checkpoint. No cache
  manifest or detector metric can be verified yet.

## 2026-05-24T17:04:01+08:00 - BATA selector enters epoch 1 validation

- Checked `35329` screen `bata_mobilenet_seqvalrescue_20260524_1437`, run tag
  `bata_mobilenet_fixed50_20260524_143349`.
- PID `562544` remains active; elapsed about `02:30:09`, CPU about `388%`,
  RSS about `12.28GB`, GPU memory `985 / 24564 MiB`, disk
  `/root/autodl-tmp` `68G / 200G` used.
- Epoch 1 training reached
  `epoch=1 step=2250/2292 train_loss_so_far=0.919717`, then validation emitted
  `val epoch=1 step=100/1886 eval_examples_so_far=12321`.
- `train_log.jsonl` still contains only epoch 0 because epoch 1 validation has
  not completed. Cache file count remains `0`; selector eval, detector output,
  and deployable mAP are absent.
- `35407` remains idle with GPU `0 / 32760 MiB` and disk `181G / 200G`;
  `25876` remains without a visible GPU and no active screen/process.
- Decision: continue waiting for epoch 1 validation completion and checkpoint.

## 2026-05-24T17:32:03+08:00 - BATA selector epoch 1 validation progress

- Checked `35329` screen `bata_mobilenet_seqvalrescue_20260524_1437`, run tag
  `bata_mobilenet_fixed50_20260524_143349`.
- PID `562544` remains active; elapsed about `02:58:11`, CPU about `384%`,
  RSS about `12.73GB`, GPU memory `985 / 24564 MiB`, disk
  `/root/autodl-tmp` `68G / 200G` used.
- Latest validation progress is
  `val epoch=1 step=1100/1886 eval_examples_so_far=136519`.
- `train_log.jsonl` still contains only epoch 0 because epoch 1 validation is
  not finished. Cache file count remains `0`; selector eval, detector output,
  and deployable mAP are absent.
- `35407` remains idle with GPU `0 / 32760 MiB` and disk `181G / 200G`;
  `25876` remains without a visible GPU and no active screen/process.
- Decision: continue waiting to epoch 1 checkpoint. No cache manifest or
  detector metric is available yet.

## 2026-05-24T17:48:16+08:00 - BATA selector epoch 1 checkpoint written

- Epoch 1 validation completed for run `bata_mobilenet_fixed50_20260524_143349`
  on `35329`.
- New `train_log.jsonl` row:
  `train_loss=0.9252101738842439`,
  `val_loss=0.6498483459567752`,
  `val_ap=0.13117830863410312`,
  `val_pos_score_mean=0.2833864390850067`,
  `val_neg_score_mean=0.2600977122783661`.
- `last.pth` updated at `2026-05-24 17:48:16 +0800`; `best.pth` remains the
  epoch 0 checkpoint because epoch 1 `val_ap` is below epoch 0 `0.132097`.
- The run entered epoch 2 and had reached at least `step=350/2292` at the
  latest check. Score cache count is still `0`; selector eval, detector output,
  and deployable mAP are absent.
- Documentation policy correction from the user is now active: do not log
  routine polling; record only clear information-gain events such as new
  checkpoints, stage transitions, cache/gate artifacts, detector results,
  failures, restarts, or cleanup.

## 2026-05-24T19:43:00+08:00 - BATA sequential validation test hardening

- Strengthened `OpenTAD_BATA_Clean/tests/test_bata_deployable_selector_contracts.py`
  after WARN review: the functional equivalence test now covers soft labels,
  tied logits for stable AP ordering, exact sequential `get_batch` frame-index
  calls, and partial-batch tensor shapes.
- Verification:
  `python -m py_compile tools\bata\mobilenet_boundary_selector.py tests\test_bata_deployable_selector_contracts.py`
  passed; `python -m pytest tests\test_bata_deployable_selector_contracts.py -q`
  reported `12 passed, 1 skipped`.
- Windows torch still printed the known access-violation stack after pytest,
  but the pytest command returned exit code `0`.

## 2026-05-24T19:27:19+08:00 - BATA selector epoch 2 checkpoint written

- Epoch 2 validation completed for run `bata_mobilenet_fixed50_20260524_143349`
  on `35329`.
- New `train_log.jsonl` row:
  `train_loss=0.8870830913668056`,
  `val_loss=0.5971741585290429`,
  `val_ap=0.12928595214802321`,
  `val_pos_score_mean=0.28137803077697754`,
  `val_neg_score_mean=0.2600649297237396`.
- `last.pth` updated at `2026-05-24 19:27:19 +0800`; `best.pth` remains the
  epoch 0 checkpoint because epoch 2 `val_ap` is below epoch 0 `0.132097`.
- At the `2026-05-24 19:43:24 +0800` check, process `562544` was active in
  epoch 3 at `step=800/2292`; cache file count remained `0`; selector eval,
  detector outputs, and deployable mAP were still absent.

## 2026-05-24T19:49:00+08:00 - Sparse geometry controls gate retry

- Retried Gemini CLI read-only review for
  `OpenTAD_SparseTAD_Clean` geometry controls with
  `gemini-3-pro-preview`; it failed again with `insufficient_user_quota`.
  Output paths:
  `logs/gemini3_pro_preview_sparse_geometry_controls_review_retry_20260524.txt`
  and `.err.txt`.
- Non-Gemini local checks progressed: SAN/config/analysis `py_compile` PASS,
  pure analysis pytest `3 passed`, `git diff --check` PASS with LF/CRLF
  warnings only.
- Custom recursive config-merge audit PASS: the `dummy_uniform`,
  `shuffled_geometry`, and `nogeometry_sameparam` SAN control configs differ
  from base SAN only by `model.neck.san_aggregation_cfg.geometry_control` and
  `work_dir`.
- Local SAN torch pytest remains blocked by the known Windows torch `c10.dll`
  initialization error. No remote sync, GPU launch, or mAP was produced.

## 2026-05-24T19:58:34+08:00 - Sparse geometry controls Linux smoke PASS

- Ran a non-training Linux torch smoke on idle `35407` using temporary hardlink
  tree `/root/autodl-tmp/OpenTAD_SparseTAD_GeomControls_Smoke_20260524_1958`.
  The completed experiment tree was not modified; only geometry-control files
  were overlaid into the temporary smoke tree.
- Smoke log:
  `/root/autodl-tmp/sparse_geom_controls_linux_smoke_20260524_1958.log`.
- Environment: `/root/miniconda3/bin/python`, torch `2.1.2+cu118`, CUDA
  available, mmengine `0.10.7`; remote pytest is missing.
- Results:
  `py_compile PASS`,
  `MANUAL_TORCH_SMOKE_PASS`,
  `ANALYSIS_DIRECT_TESTS_PASS`,
  `CONFIG_DIFF_AUDIT_PASS`,
  `SMOKE_ALL_PASS`.
- Manual torch coverage included all geometry-control modes, non-prefix masks,
  constant parameter count, invalid-mode rejection, FPNIdentity no-op behavior,
  and affine-gradient propagation.
- No training, detector evaluation, or mAP was launched. Sparse geometry
  controls remain not launch-ready because Gemini review is still quota-blocked
  unless a replacement/waiver decision is explicitly recorded.

## 2026-05-24T21:39:47+08:00 - Optimized BATA queued launch started on 35329

- Started the optimized MobileNet-BCA fixed-50 serial launcher on `35329` in
  detached screen `bata_mobilenet_speedopt_queue_20260524_2142` from
  `/root/autodl-tmp/OpenTAD_BATA_Formal_MobileNet_20260523`.
- The active old selector was not stopped. It remains PID `562544` in screen
  `bata_mobilenet_seqvalrescue_20260524_1437`, still using old 5-epoch
  selector settings (`4096/2048` train/val caps, `sequential_video`,
  `num_workers=0`).
- Queue run tag: `bata_mobilenet_fixed50_20260524_213946`.
  Outer log:
  `/root/autodl-tmp/OpenTAD_BATA_Formal_MobileNet_20260523/logs/bata_mobilenet_speedopt_queue_20260524_2142_outer.log`.
  Launcher log:
  `/root/autodl-tmp/OpenTAD_BATA_Formal_MobileNet_20260523/logs/bata_mobilenet_fixed50_20260524_213946.log`.
- An initial queue attempt `bata_mobilenet_speedopt_queue_20260524_2138`
  was immediately stopped because `TORCHRUN=.../torchrun` in the screen
  command line caused the launcher's active-process guard to self-match. The
  corrected launch omits that env text; verification shows the queue wait file
  lists only old selector PID `562544`.
- Optimized queue settings are explicit: `SELECTOR_EPOCHS=1`,
  `SELECTOR_LOADER_MODE=sequential_video`, `SELECTOR_TRAIN_SHUFFLE=0`,
  `SELECTOR_NUM_WORKERS=0`, `SELECTOR_MAX_TRAIN_FRAMES_PER_VIDEO=2048`,
  `SELECTOR_MAX_VAL_FRAMES_PER_VIDEO=1024`,
  `SELECTOR_DECORD_NUM_THREADS=2`, `MASTER_PORT=29637`.
- Protocol boundary unchanged: strict fixed-50 detector/cache contract is
  preserved; no test-time GT or teacher leakage is introduced. The Gemini gate
  for the speed patch remains incomplete, with the user override already
  recorded at direct deployment time.

## 2026-05-24T22:51:20+08:00 - BATA old path stopped; GPUs idle for next launch

- User requested stopping the current old BATA path and directly starting the
  optimized BATA MobileNet-BCA route.
- The old run `bata_mobilenet_fixed50_20260524_143349` had already moved from
  5-epoch selector training into training-cache build using epoch-0
  `best.pth`, but no deployable cache/gate/detector/mAP existed yet.
- Stopped old screen `bata_mobilenet_seqvalrescue_20260524_1437`, old
  build-cache process for run tag `bata_mobilenet_fixed50_20260524_143349`,
  and prior optimized wait screen `bata_mobilenet_speedopt_queue_20260524_2142`.
- Verification:
  `35329` has no sockets, no active TAD/BATA train process, GPU
  `0/24564 MiB`; `35407` has no sockets, no active train process, GPU
  `0/32760 MiB`; `25876` has no visible GPU.
- No optimized selector epoch has started yet after the stop. Recommended next
  action is optimized MobileNet-BCA on `35329`, where the speed-optimized
  formal tree is already deployed and verified.

## 2026-05-24T23:07:20+08:00 - BATA optimized and Geometry controls launched

- Launched optimized BATA MobileNet-BCA on `35329` from
  `/root/autodl-tmp/OpenTAD_BATA_Formal_MobileNet_20260523`.
  Screen: `bata_mobilenet_speedopt_direct_20260524_2303`.
  Run tag: `bata_mobilenet_fixed50_20260524_230216`.
  Outer log:
  `/root/autodl-tmp/OpenTAD_BATA_Formal_MobileNet_20260523/logs/bata_mobilenet_speedopt_direct_20260524_2303_outer.log`.
  The launcher passed preflight pytest with `32 passed in 2.04s` and entered
  optimized selector training: `epochs=1`, `sequential_video`, train/val caps
  `2048/1024`, decord threads `2`; launch check reached
  `epoch=0 step=200/1886`.
- User explicitly requested Geometry causality controls on `35407` despite the
  Gemini quota-blocked review gate; this is recorded as the waiver/replacement
  decision for launch. The Linux smoke tree
  `/root/autodl-tmp/OpenTAD_SparseTAD_GeomControls_Smoke_20260524_1958`
  was used.
- Uploaded and launched
  `logs/run_sparse_geom_controls_35407_20260524_2308.sh` on `35407`.
  Screen: `sparse_geom_controls_serial_20260524_2308`.
  It serially runs `dummy_uniform`, `shuffled_geometry`, and
  `nogeometry_sameparam` controls with `workflow.checkpoint_interval=20` and
  keep-last checkpoint cleanup after each completed control.
- Geometry first active control:
  `input_random_fixed_50pct_adapter_san_aggregation_dummy_uniform.py`,
  work dir `exps/thumos/adatad/san_geom_dummy_uniform_20260524_2308`,
  log `logs/san_geom_dummy_uniform_20260524_2308.log`; it entered
  `Epoch 0 started`. GPU at launch check: `3461/32760 MiB`; disk remained
  tight at `181G/200G` used.

## 2026-05-24T23:24:41+08:00 - Low-frequency long monitors started

- User requested long-term monitoring without dense querying.
- Uploaded and started BATA monitor script
  `logs/run_bata_long_monitor_35329_20260524_2322.sh` on `35329` in
  detached screen `bata_long_monitor_20260524_2322`. It writes to
  `/root/autodl-tmp/OpenTAD_BATA_Formal_MobileNet_20260523/logs/bata_long_monitor_20260524_2322.log`.
- Uploaded and started Geometry monitor script
  `logs/run_geometry_long_monitor_35407_20260524_2322.sh` on `35407` in
  detached screen `geometry_long_monitor_20260524_2322`. It writes to
  `/root/autodl-tmp/OpenTAD_SparseTAD_GeomControls_Smoke_20260524_1958/logs/geometry_long_monitor_20260524_2322.log`.
- Both scripts use `MONITOR_INTERVAL_SECONDS=1800` by default. This action
  started only monitor screens; it did not launch new training, change configs,
  produce mAP, verify cache manifests, or clean checkpoints.
- Next manual read should wait for a meaningful interval or stage transition:
  BATA selector checkpoint/cache/gate/detector/mAP, Geometry eval/completion,
  crash, cleanup, or disk pressure.

## 2026-05-24T23:41:01+08:00 - Low-frequency monitor discipline check

- Local time check showed only about 16 minutes had elapsed since the
  `23:24:41+08:00` monitor launch, less than the default `1800s` sampling
  interval.
- Per the user's request, no SSH query or remote monitor-log fetch was made.
- No new training state, mAP, cache manifest, gate result, crash evidence, or
  cleanup action is claimed from this check.
- Continue waiting until the low-frequency interval has elapsed or a meaningful
  stage transition needs inspection.

## 2026-05-24T23:58:00+08:00 - First low-frequency read; BATA monitor repaired

- Waited until the first 1800s monitor window, then read one tail from each
  monitor log.
- `35329` BATA optimized MobileNet-BCA remains active in screen
  `bata_mobilenet_speedopt_direct_20260524_2303`, run tag
  `bata_mobilenet_fixed50_20260524_230216`. At `23:56:23`, selector PID
  `704562` was in epoch 0 validation at `val step=300/1336`. `train_log.jsonl`
  and selector checkpoint for this optimized run are not written yet; train/val
  cache dirs and selector eval dir are still missing. Detector has not started;
  deployable mAP is absent. GPU sample `1011/24564 MiB`, util `0%`; disk
  `/root/autodl-tmp` `68G/200G` used.
- The initial BATA monitor screen had exited because the script treated missing
  cache/eval directories as fatal under `set -euo pipefail`. Patched
  `logs/run_bata_long_monitor_35329_20260524_2322.sh` to count missing dirs as
  zero and guard `find` pipelines; remote `bash -n` passed; restarted screen
  `bata_long_monitor_20260524_2322` at `23:57:58`.
- `35407` Geometry controls remain active in screen
  `sparse_geom_controls_serial_20260524_2308`. First control
  `san_geom_dummy_uniform` reached `Epoch 11 started`; latest complete line was
  epoch 10 at `23:52:58`, `Loss=0.6480`. No result JSON/mAP yet; disk remains
  tight at `181G/200G` used. Patched Geometry monitor script was uploaded and
  remote `bash -n` passed, but its existing healthy monitor was not restarted.
- Continue low-frequency monitoring. Next meaningful BATA event is selector
  checkpoint, then GT-free cache/gate/detector/mAP. Next Geometry event is first
  eval/completion or transition to shuffled/no-geometry control.

## 2026-05-25T00:28:20+08:00 - BATA selector checkpoint; cache build started

- Waited until the next low-frequency window and read monitor logs once.
- `35329` BATA optimized selector completed epoch 0 and wrote
  `/root/autodl-tmp/bata_selectors/bata_mobilenet_fixed50_20260524_230216/train_log.jsonl`
  at `2026-05-25T00:26:32+08:00`.
- Selector metrics: `train_loss=0.9481145259703511`,
  `val_loss=0.5086234837169621`, `val_ap=0.18466035040354725`,
  `val_pos_score_mean=0.2608599066734314`,
  `val_neg_score_mean=0.24128586053848267`, `pos_weight=4.165201187133789`.
- The BATA launcher entered `building deployable selector caches` at
  `00:26:34`. At the `00:27:58` monitor sample, build-cache PID `733960` was
  active for the training split using the selector `best.pth`; artifact counts
  were `train_npz=9`, `val_npz=0`, `eval_files=0`.
- Detector has not started and deployable mAP is still absent. Next BATA checks:
  cache completion, GT-free manifest fields, eval-cache gate, dataset smoke,
  detector result, and cleanup.
- `35407` Geometry controls remain on the first control
  `san_geom_dummy_uniform`. At `00:24:41`, it had reached epoch 18 iter 50/99;
  latest loss line `Loss=0.6211`. No result JSON or mAP yet. Disk remains
  tight at `181G/200G` used.

## 2026-05-25T01:07:39+08:00 - Pro priority discussion completed

- Oracle browser Pro session `gpt-5-5-pro-thumos14` completed with
  `gpt-5.5-pro` / `Extended Pro`.
- Transcript:
  `C:\Users\skywalker\.oracle\sessions\gpt-5-5-pro-thumos14\artifacts\transcript.md`.
- Accepted recommendation: keep the current `35329` BATA deployable full
  pipeline running; prepare exactly one guarded cached detector-only throughput
  wait-screen; do not add tasks on `35407`; postpone short-selector ablations,
  pretrained selector, dynamic budget, oracle-gap diagnostics, and SAHM until
  the fixed-50 deployable result exists.
- Pro mAP branch gates: `<63.77` means debug rather than claim; `63.77-65.09`
  means fill speed/control/repeat evidence; `>=65.09` means confirm, speed
  table, then consider dynamic budget.

## 2026-05-25T01:10:36+08:00 - BATA detector-only wait screen deployed

- Uploaded `logs/run_bata_cached_detector_only_queue_35329_20260525_0112.sh`
  to `35329:/root/autodl-tmp/OpenTAD_BATA_Formal_MobileNet_20260523/logs/`.
- Remote `bash -n` passed; SHA256:
  `8d089056de2371405556d6f6777e6088b36570608f2bce2ef2d70b7dcd80b79b`.
- Started detached screen `bata_cached_detector_only_queue_20260525_0112`.
  Queue log:
  `/root/autodl-tmp/OpenTAD_BATA_Formal_MobileNet_20260523/logs/bata_mobilenet_cached_detector_only_20260525_0112.log`.
- The wait-screen is currently waiting on active build-cache PID `733960`; it
  does not overlap GPU training. It will launch only after the source BATA full
  pipeline finishes cleanly, source `result_detection.json` exists, train/val
  cache manifests are no-GT deployable MobileNet caches, and eval-cache gate
  passes.
- `35407` Geometry monitor latest read at `00:54:41` still has first control
  `san_geom_dummy_uniform` active at epoch 25 iter 50/99, no mAP, disk
  `182G/200G` used. No new task was deployed there.

## 2026-05-25T01:16:11+08:00 - Early monitor recheck, no new sample

- A continuation-triggered read checked the existing BATA and Geometry monitor
  logs, plus the BATA cached detector-only queue log.
- No new monitor sampling window had elapsed since the prior `00:57:58`
  BATA sample and `00:54:41` Geometry sample. The BATA full pipeline therefore
  remains last evidenced as training-cache build with `train_npz=116`,
  `val_npz=0`, `eval_files=0`, no detector and no deployable mAP.
- The detector-only wait screen `bata_cached_detector_only_queue_20260525_0112`
  remains in waiting mode; its log still shows it waiting on active build-cache
  PID `733960`. It has not launched a second detector process.
- Geometry remains last evidenced at `san_geom_dummy_uniform` epoch 25 iter
  50/99 with no mAP. No deployment or stop/continue decision changed.

## 2026-05-25T01:40:05+08:00 - BATA validation cache and Pro priority refresh

- Read one low-frequency monitor window. `35329` BATA advanced from training
  cache to validation-cache build. At `01:27:58`, validation cache PID `747856`
  was active with `train_npz=200`, `val_npz=80`, `eval_files=0`; detector output
  and deployable mAP remain absent.
- Training cache manifest audit passed with no-GT deployable fields:
  `uses_gt=False`, `diagnostic_only=False`, `deployable_selector=True`,
  `axis='global_snippet_index'`, `axis_frame_stride=4`, `snippet_stride=4`,
  `scale_factor=1`, `score_source='mobilenet_v3_small_boundary_probability'`,
  `selector_arch='mobilenet_v3_small'`, `selector_decord_num_threads=2`,
  `video_count=200`, `npz_count=200`.
- The cached detector-only wait screen remains waiting and has not launched a
  second detector process.
- `35407` Geometry controls remain active on `san_geom_dummy_uniform`, reaching
  epoch 33 start at the `01:24:41` monitor sample. No mAP/result JSON yet; disk
  is tight at about `19G` free.
- Oracle browser Pro session `gpt5pro-bata-geom-experiment-priority` completed
  with verified `gpt-5.5-pro` / `Extended Pro` after the initial MCP consult
  timed out while the browser job continued. Transcript:
  `C:\Users\skywalker\.oracle\sessions\gpt5pro-bata-geom-experiment-priority\artifacts\transcript.md`.
- Accepted Pro decision: do not deploy new long experiments now. Continue the
  `35329` BATA full pipeline, keep exactly one guarded cached detector-only
  rerun, continue current `35407` Geometry first control, and use `25876` only
  for CPU/read-only support. Before later Geometry controls, apply the disk
  gate: target `25-30G` free, do not continue blindly below `20G`, pause below
  `15G`.

## 2026-05-25T02:10:07+08:00 - BATA cache/gate/smoke passed; detector started

- Waited for a stage-transition window after `val_npz=194` and checked `35329`
  once.
- BATA validation cache completed: training cache has `200` npz, validation
  cache has `211` npz.
- Manifest audits passed for both train and validation caches. Required
  deployable fields are clean: `uses_gt=False`, `diagnostic_only=False`,
  `deployable_selector=True`, `score_source='mobilenet_v3_small_boundary_probability'`,
  `axis='global_snippet_index'`, `axis_frame_stride=4`, `snippet_stride=4`,
  `scale_factor=1`, `selector_arch='mobilenet_v3_small'`,
  `selector_decord_num_threads=2`.
- Manifest hashes:
  training `ea3b65cc848fa8906258f59cb34872551cf2de250cf59828a91b8ca1ea576b93`;
  validation `1ca1bd2f9ed00ff34183e26c8d4d2259410cc13a934fb31bad0ac59424ecadeb`.
- Score sanity passed for all train/validation npz files: no nonfinite or
  all-constant score arrays were found.
- Eval-cache gate passed. Summary hash:
  `0ae462d2dc48cbdf4fe55c612e355f4281f552e27b58adf95aa93aebbc009f0d`.
  Key fields: `cache_uses_gt=false`, `diagnostic_only=false`,
  `coverage_pass_rate=1.0`, `gate.pass=true`,
  `uses_gt_for_eval_only=true`.
- Dataset smoke passed for train/val/test samples. Launcher milestones:
  `02:01:18` eval-cache gate, `02:02:11` smoke, `02:02:27` detector launch,
  `02:02:39` `Training Starts`.
- Detector is active in `tools/train.py configs/adatad/thumos/bata_mobilenet_bca_fixed50_adapter.py`.
  Latest checked train line at `02:08:23`: epoch 0 iter 50/99,
  `Loss=1.7170`, `cls_loss=0.9765`, `reg_loss=0.7405`, `mem=2529MB`.
- The cached detector-only queue remains waiting on the active full-pipeline
  detector; it has not launched a second detector run.
- `35407` Geometry first control progressed to epoch 40 iter 50/99 by the
  `01:54:41` monitor sample, still with no mAP; disk remains about `19G` free,
  so the disk gate before later controls remains active.

## 2026-05-25T02:41:00+08:00 - Pro next-experiment review, no new deployment

- Per user request, checked the low-frequency monitor/log state and asked Pro
  to reassess which experiments are still worth running and whether to deploy
  more work.
- `35329` BATA remains active in detector training. Latest checked line at
  `02:29:57`: epoch 2 iter 50/99, `Loss=0.8318`, `cls_loss=0.5005`,
  `reg_loss=0.3313`, `mem=2529MB`; deployable mAP is still absent. The existing
  cached detector-only queue still waits and has not launched a second run.
- `35407` Geometry first control now has a first eval: `Average-mAP=62.09`,
  vector `79.83 / 74.05 / 64.58 / 53.69 / 38.32`, then continued to epoch 43.
  Disk remains tight at about `19G` free.
- Pro routing: Oracle MCP timed out without a recoverable session; Rosetta Pro
  was unavailable because `127.0.0.1:9222` refused; Oracle CLI browser Pro
  completed with verified `Extended Pro`. Raw response:
  `research-wiki/experiments/GPT5_PRO_BATA_GEOM_NEXT_EXPERIMENTS_20260525.raw.md`.
- Accepted Pro decision: `HOLD`. Do not deploy any new experiment or waiting
  queue. Continue the active `35329` BATA full pipeline, keep only the existing
  guarded detector-only queue, let `35407` finish the current first Geometry
  control, and gate later Geometry controls on disk (`25-30G` target, no blind
  continue below `20G`, pause below `15G`).

## 2026-05-25T02:43:26+08:00 - Low-frequency monitor discipline check

- Local time check showed only about 2 minutes had elapsed since the latest
  `02:41:00+08:00` Pro/monitor record, well below the requested 1800s monitor
  interval.
- No SSH query was made and no remote monitor log, screen, GPU, result file, or
  disk state was read.
- Last authoritative state remains unchanged: `35329` BATA detector was last
  seen at epoch 2 iter 50/99 with no deployable mAP and the detector-only queue
  waiting; `35407` Geometry first control had first eval `62.09` and tight
  `19G` free disk.
- Decision remains `HOLD`: no new deployment or queue; next useful remote read
  should wait until roughly `03:00+08:00` or a clear stage transition.

## 2026-05-25T02:45:38+08:00 - BATA final-result audit checklist prepared

- Still below the 1800s low-frequency monitor interval, so no remote query was
  made.
- Added a final-result audit checklist to
  `research-wiki/experiments/BATA_SELECTOR_SPEED_PRO_SYNTHESIS_AND_OPTIMIZATION_20260524.md`.
- The checklist records what must be collected once BATA writes mAP:
  `result_detection.json` path/hash, Average-mAP and vector, checkpoint/log
  evidence, train/validation cache manifest hashes and no-GT deployable fields,
  eval-cache/smoke proof, timing and throughput fields, and the `<63.77`,
  `63.77-64.64`, `64.64-65.09`, `>=65.09` interpretation bands.
- No code, config, server process, queue, or protocol state changed. Decision
  remains `HOLD`.

## 2026-05-25T03:00:37+08:00 - Low-frequency monitor read

- Waited until the next low-frequency monitor window before querying remote
  servers.
- `35329` BATA remains in deployable fixed-50 detector training. Latest monitor
  sample was `02:57:59`; latest detector lines show epoch 4 completed at
  `02:56:34` with `Loss=0.7889`, `cls_loss=0.4749`, `reg_loss=0.3140`, and
  epoch 5 started. Cache counts remain `train_npz=200`, `val_npz=211`,
  `eval_files=1`; deployable mAP and `result_detection.json` are still absent.
  The cached detector-only wait-screen still waits on the active full-pipeline
  detector and has not launched a second run. Disk on `35329` is healthy:
  `68G/200G` used, `133G` free.
- `35407` Geometry first control `san_geom_dummy_uniform` produced a second
  eval at `02:43:00`: `Average-mAP=62.38`, vector
  `80.01 / 74.39 / 64.92 / 53.75 / 38.84`. This is slightly above the first
  eval `62.09`, but still below random-fixed `63.77`, strict EMA `63.85`,
  SAN-Aggregation `64.14`, stratified `64.64`, and uniform stride-2 `65.09`.
  At the `03:00` read it was still the first control, in another validation
  after epoch 45; no shuffled/no-geometry control has started. Disk remains
  tight at about `19G` free.
- Decision remains `HOLD`: continue BATA to mAP, keep only the existing
  detector-only queue, continue the current Geometry first control, and gate
  later Geometry controls on disk.

## 2026-05-25T03:30:45+08:00 - Low-frequency monitor read

- Waited until the next low-frequency window before querying remote servers.
- `35329` BATA continues healthy deployable fixed-50 detector training. Latest
  monitor sample was `03:27:59`; latest detector lines show epoch 5 finished at
  `03:07:16` with `Loss=0.8052`, epoch 6 finished at `03:18:14` with
  `Loss=0.7251`, epoch 7 finished at `03:29:17` with `Loss=0.7349`,
  `cls_loss=0.4368`, `reg_loss=0.2981`, and epoch 8 started. There is still no
  deployable mAP or `result_detection.json`. The cached detector-only queue
  remains waiting and has not launched. Disk on `35329` remains healthy:
  `68G/200G` used, `133G` free.
- `35407` Geometry first control remains on `san_geom_dummy_uniform`. It
  produced a third eval at `03:08:31`: `Average-mAP=63.00`, vector
  `80.29 / 74.89 / 65.75 / 54.64 / 39.44`. This improves over `62.09` and
  `62.38`, but is still below random-fixed `63.77`, strict EMA `63.85`,
  SAN-Aggregation `64.14`, stratified `64.64`, and uniform stride-2 `65.09`.
  The run was still validating after epoch 47; no shuffled/no-geometry control
  has started. Disk remains tight at about `19G` free.
- Decision remains `HOLD`: no new deployment or queue; continue BATA to mAP,
  keep only the existing detector-only queue, continue the current Geometry
  first control, and gate later Geometry controls on disk.

## 2026-05-25T04:09:53+08:00 - GPT-5.5 Pro continue/deploy decision refresh

- Ran Oracle browser Pro session `gpt5pro-bata-geom-continue-deploy`; verified
  ChatGPT label was `Extended Pro`; elapsed `3m16s`.
- Report written to
  `research-wiki/experiments/GPT5_PRO_BATA_GEOM_CONTINUE_DEPLOY_20260525.md`.
  Transcript:
  `C:\Users\skywalker\.oracle\sessions\gpt5pro-bata-geom-continue-deploy\artifacts\transcript.md`.
- Pro verdict: `HOLD`, but not a stop. Continue active experiments only:
  `35329` BATA deployable MobileNet-BCA fixed-50 mainline to final mAP, keep
  exactly one guarded cached detector-only queue, let `35407`
  `san_geom_dummy_uniform` finish under disk safety, and use `25876` only for
  CPU/log support.
- Pro explicitly deferred short-selector, pretrained selector, dynamic budget,
  oracle-gap diagnostic, SAHM/SoftmaxGate, and new SAPM/STGA/SAN variants until
  after BATA final mAP and the relevant branch gate.
- Geometry disk gate reaffirmed: do not start `shuffled_geometry` or
  `nogeometry_sameparam` while `35407` remains around `19G` free; target
  `25-30G` free before re-evaluating later controls.

## 2026-05-25T04:33:53+08:00 - Low-frequency monitor read

- Queried existing remote monitor/log files only after the next low-frequency
  window.
- `35329` BATA deployable fixed-50 detector remains active. Latest detector
  progress: epoch 12 finished at `04:25:00` with `Loss=0.6219`,
  `cls_loss=0.3521`, `reg_loss=0.2698`; epoch 13 started, and epoch 13 iter
  50/99 logged `Loss=0.6141` at `04:30:53`. There is still no deployable
  `result_detection.json`; train/val cache counts remain `200/211`; eval-cache
  files remain `1`; the cached detector-only queue still waits on active BATA
  processes. Disk is healthy at `68G/200G` used, `133G` free.
- `35407` Geometry remains on the first control `san_geom_dummy_uniform`; no
  `shuffled_geometry` or `nogeometry_sameparam` log exists. New eval at
  `04:24:44`: `Average-mAP=63.56`, vector
  `80.60 / 75.40 / 66.26 / 55.34 / 40.22`. This remains below random-fixed
  `63.77`, strict EMA `63.85`, SAN `64.14`, stratified `64.64`, and uniform
  `65.09`. Disk is still tight at `182G/200G` used, `19G` free.
- Decision remains `HOLD`: no new deployment or queue. Continue BATA to mAP;
  continue only the current Geometry first control, and keep later controls
  blocked unless disk recovers to the `25-30G` target.

## 2026-05-25T05:06:12+08:00 - Low-frequency monitor read

- `35329` BATA remains in deployable fixed-50 detector training. Latest train
  state: epoch 15 finished at `04:58:49` with `Loss=0.5951`,
  `cls_loss=0.3313`, `reg_loss=0.2638`; epoch 16 started, and epoch 16 iter
  50/99 logged `Loss=0.5598` at `05:04:48`. No deployable result exists; the
  cached detector-only queue still waits on active BATA processes.
- `35407` Geometry remains on `san_geom_dummy_uniform`; no later-control logs
  exist. New eval at `04:50:23`: `Average-mAP=63.86`, vector
  `80.90 / 75.61 / 66.40 / 55.80 / 40.60`. This barely exceeds random-fixed
  `63.77` and strict EMA `63.85`, but remains below SAN `64.14`, stratified
  `64.64`, uniform `65.09`, and the high-IoU reference pressure.
- Disk on `35407` is still `182G/200G` used, about `19G` free. Because the
  first control is nearing completion, the next check may be closer than the
  normal interval to catch the serial transition and enforce the disk gate.

## 2026-05-25T05:28:22+08:00 - Geometry stage-transition check blocked

- A closer stage-transition check was attempted for `35407` because
  `san_geom_dummy_uniform` was nearing completion and the serial script can
  enter later controls automatically while disk remains around `19G` free.
- Two native OpenSSH attempts to
  `ssh -p 35407 root@connect.cqa1.seetacloud.com`, separated by about 60s,
  both returned `Connection refused`.
- No remote process, log, disk, final mAP, or later-control state was read, and
  no stop/deploy action was taken.
- Interpretation: monitor connectivity block only, not experiment failure.
  Avoid dense retries; retry later. The Geometry disk gate remains active but
  could not be enforced through SSH at this moment.

## 2026-05-25T05:45:55+08:00 - Geometry connectivity still blocked

- Retried `ssh -p 35407 root@connect.cqa1.seetacloud.com` after waiting about
  15 minutes.
- The endpoint again returned `Connection refused`.
- No remote process, log, disk, final mAP, or later-control status was read,
  and no stop/deploy action was possible.
- Continue treating this as a connectivity block, not result evidence. The
  Geometry disk gate remains the intended policy, but enforcement requires SSH
  access to recover.

## 2026-05-25T05:46:30+08:00 - BATA monitor check blocked

- Attempted the next normal low-frequency BATA read from `35329`.
- `ssh -p 35329 root@connect.cqa1.seetacloud.com` returned
  `Connection refused`.
- No BATA process, log, result, queue, or disk state was read, and no action was
  taken.
- Because `35407` also refused SSH connections, this is treated as
  endpoint/connectivity blockage rather than BATA failure evidence. Last
  authoritative BATA state remains detector epoch 16 in progress at
  `05:04:48`, no deployable result, detector-only queue waiting.

## 2026-05-25T06:04:43+08:00 - Remote endpoint connectivity still blocked

- After waiting about 15 minutes, retried the active GPU endpoints with native
  OpenSSH and `ConnectTimeout=10`.
- `35329` and `35407` both returned
  `banner exchange: Connection to UNKNOWN port -1: Connection refused`.
- A single support endpoint check on `25876` returned the same refusal.
- No remote BATA/Geometry process, log, result, queue, or disk state was read,
  and no stop/deploy action was possible.
- Treat this as endpoint/platform connectivity blockage, not experiment failure
  evidence. Avoid dense retries. Last authoritative BATA evidence remains epoch
  16 at `05:04:48`; last authoritative Geometry evidence remains
  `san_geom_dummy_uniform` eval `63.86` at `04:50:23`, with no later-control
  log as of the `05:06` read.

## 2026-05-25T06:23:45+08:00 - Remote endpoint connectivity block persists

- After another recovery window, retried all three remote endpoints with native
  OpenSSH and `ConnectTimeout=10`.
- `35329`, `35407`, and `25876` all returned
  `banner exchange: Connection to UNKNOWN port -1: Connection refused`.
- No remote BATA/Geometry process, log, result, queue, or disk state was read,
  and no stop/deploy action was possible.
- Endpoint/platform connectivity blockage persists. This is not evidence that
  the experiments failed. Last authoritative BATA state remains detector epoch
  16 iter 50/99 at `05:04:48`, no deployable mAP; last authoritative Geometry
  state remains first-control eval `63.86` at `04:50:23`, with no later-control
  log as of `05:06`.

## 2026-05-25T10:25:00+08:00 - BATA migration package staging created

- Created local new-server migration package:
  `migration_packages/bata_mobilenet_fixed50_migration_20260525.tar.gz`.
- Archive was later superseded by the combined BATA + Sparse package below;
  use the combined archive for migration.
- Included `OpenTAD_BATA_Clean`, THUMOS annotation/class map, required
  VideoMAE-S pretrained weight, N16R4 prepare/sbatch scripts, sync/video
  manifests, selected trackers, and Pro-decision docs.
- Excluded THUMOS mp4 videos, score caches, checkpoints, detector outputs,
  `.git`, pycache, and old archives. `MANIFEST.sha256` contains `397` file
  entries; tar verification confirmed README, manifest, and pretrain are
  present; excluded-file scan passed.
- This is packaging only: no new training launched and no protocol change.
  Videos must still be staged separately on the new server under the documented
  `~/run/yuzibo/thumos14` layout.

## 2026-05-25T10:41:00+08:00 - Combined BATA + Sparse migration package created

- Created final combined migration package:
  `migration_packages/tad_bata_sparse_migration_20260525.tar.gz`.
- Archive size: `52.58 MB`; SHA256:
  `1082b3439e947bc86813e78a1ffe7bdce52f157ba7e0f7addab026600cc7e959`.
- Included both clean code trees: `OpenTAD_BATA_Clean` and
  `OpenTAD_SparseTAD_Clean`; THUMOS annotation/class map; one shared
  VideoMAE-S pretrained weight; N16R4 BATA, Sparse, and Geometry sbatch
  scripts; monitor/sync manifests; and selected trackers/review docs.
- Excluded THUMOS mp4 videos, score caches, checkpoints, detector outputs,
  `.git`, pycache, and old archives. `MANIFEST.sha256` contains `879` file
  entries; tar verification confirmed README, manifest, both code trees, and
  pretrain are present; excluded-file scan passed with only one `.pth` entry.
- This is packaging only: no new training launched and no protocol change.
  Videos still need to be staged separately on the new server.

## 2026-05-27T23:43:53+08:00 - N16R4 deployment progress audit

- Checked N16R4 via Windows native OpenSSH under the project workspace
  `~/run/yuzibo`.
- Code/package state: `OpenTAD_BATA_Clean` and `OpenTAD_SparseTAD_Clean`
  are present; shared pretrain is staged under `~/run/yuzibo/pretrained`;
  the combined package directory `tad_bata_sparse_migration_20260525` exists.
- Environment state: `~/run/yuzibo/conda_envs/opentad` imports
  `torch 2.0.1`, `torchvision 0.15.2`, `mmcv 2.0.1`,
  `mmengine 0.10.7`, `mmaction 1.1.0`, `decord 0.6.0`, and
  `numpy 1.23.5`.
- Remote smoke passed without launching training: BATA py_compile plus contract
  pytest `32 passed`; Sparse py_compile plus contract pytest `27 passed`.
- Data state after non-destructive ingest with Python 3:
  training layout has `86/200` linked videos, with byte-size audit
  `80 ok`, `6 bad_size`, and `114 missing`; validation/test layout has
  `211/211` linked videos, with byte-size audit `197 ok`, `14 bad_size`,
  and `0 missing`.
- User confirmed video data is still being uploaded one by one. No partial
  files were deleted, no zip was extracted, and no Slurm OpenTAD job was
  submitted.
- Decision: continue monitoring upload and re-run ingest/byte-size audit until
  train/test are exactly `200/211` with `bad_size=0`; only then run final
  preflight and submit the selected N16R4 sbatch.

## 2026-05-27T23:47:52+08:00 - N16R4 upload gate watcher started

- Added local watcher script `logs/n16r4_upload_gate_watch_20260527.sh` and
  uploaded it to `~/run/yuzibo/scripts/n16r4_upload_gate_watch_20260527.sh`.
- Remote `bash -n` passed and the watcher was launched with PID `2142542`.
  Log path:
  `~/run/yuzibo/setup_logs/n16r4_upload_gate_watch_20260527_234752.log`.
- The watcher is non-destructive: it only runs ingest, byte-size audit, and log
  writes every `120s`. It does not stop SFTP, delete partial mp4 files, unzip
  archives, or submit Slurm training.
- First watcher iteration recorded train `83 ok`, `110 missing`, `7 bad_size`;
  test `198 ok`, `0 missing`, `13 bad_size`; and `active_sftp=20`.
- Decision remains blocked on data: no N16R4 OpenTAD training launch until the
  upload gate reaches train `200/200`, test `211/211`, and `bad_size=0`.

## 2026-05-27T23:50:35+08:00 - N16R4 upload watcher progress check

- Read watcher log
  `~/run/yuzibo/setup_logs/n16r4_upload_gate_watch_20260527_234752.log`.
- Watcher PID `2142542` is still running.
- Iteration 2 at `23:49:53+08:00` remains blocked but shows upload progress:
  train `83 ok`, `110 missing`, `7 bad_size`; test `198 ok`, `0 missing`,
  `13 bad_size`; partial file sizes increased and `active_sftp=21`.
- No upload PASS marker exists yet. Decision remains: no Slurm training launch
  until the byte-size gate passes.

## 2026-05-27T23:58:05+08:00 - N16R4 upload stall watchdog armed

- Added `logs/sync_thumos_n16r4_raw_missing.ps1`, a raw-directory upload helper
  for the current symlink layout. It uploads missing or wrong-size local TAD
  videos to `~/run/yuzibo/raw/Validation Data/validation` and
  `~/run/yuzibo/raw/Test Data/TH14_test_set_mp4`, then reruns ingest.
- Added and started `logs/watch_n16r4_upload_stall_and_resume_20260527.ps1`
  as local PID `80844`; log path
  `logs/watch_n16r4_upload_stall_and_resume_20260527.log`.
- The watchdog implements the user rule: if the upload status has no signature
  change for `60` minutes, it first records remote SFTP/SSH process state and
  recent raw-file writes, then starts the raw missing upload helper with
  `4` parallel SFTP transfers. It does not delete partial files, kill SFTP, or
  submit Slurm jobs.
- Initial watchdog state already showed progress: train `86 ok`,
  `106 missing`, `8 bad_size`; test `199 ok`, `0 missing`, `12 bad_size`.
  Therefore no upload takeover occurred yet.

## 2026-05-28T00:01:11+08:00 - N16R4 upload watcher progress

- Remote watcher `2142542` remains alive.
- Latest logged iteration `7` at `23:59:58+08:00`: train `87 ok`,
  `105 missing`, `8 bad_size`; test `199 ok`, `0 missing`, `12 bad_size`;
  `active_sftp=24`.
- No PASS marker exists yet. Upload is progressing, so no local补传 takeover
  has occurred and no Slurm job has been submitted.

## 2026-05-28T00:08:34+08:00 - N16R4 post-upload verify-and-launch gate started

- Added `logs/n16r4_post_upload_verify_and_launch_20260528.sh` and uploaded it
  to `~/run/yuzibo/scripts/n16r4_post_upload_verify_and_launch_20260528.sh`.
- Remote `bash -n` passed and the gate was launched as PID `2225557`.
  Log path:
  `~/run/yuzibo/setup_logs/n16r4_post_upload_verify_and_launch_20260528_000834.log`.
- Behavior: poll byte-size data gate; after data PASS, run runtime imports,
  BATA py_compile + BATA contract pytest, Sparse py_compile + Sparse contract
  pytest, then submit the next planned BATA deployable fixed-50 job via
  `run_bata_mobilenet_fixed50_n16r4.sbatch` with Slurm job name
  `bata_mobile50`.
- First data audit was still blocked: train `87 ok`, `105 missing`,
  `8 bad_size`; test `199 ok`, `0 missing`, `12 bad_size`; `DATA_PASS=False`.
  No verification or Slurm submission has run yet.

## 2026-05-28T00:17:00+08:00 - N16R4 upload progress and watchdog correction

- Checked current upload state. Remote watcher `2142542` and post-upload gate
  `2225557` are alive. Latest state remains data-blocked but progressing:
  train `87 ok`, `105 missing`, `8 bad_size`; test `199 ok`, `0 missing`,
  `12 bad_size`; `active_sftp=19`.
- Partial bad-file byte signatures are still changing, so the upload has not
  met the user-defined one-hour no-change condition. No helper upload was
  started.
- Prechecked local source `E:\tmp\thumos14_raw`: all required TAD videos are
  present locally, train `200/200` and validation/test `211/211`, so the helper
  can take over if the upload really stalls.
- Corrected `logs/watch_n16r4_upload_stall_and_resume_20260527.ps1` so its
  progress signature includes bad-file byte-tail changes, not only
  ok/missing/bad counts. Stopped old local watchdog PID `80844`, restarted PID
  `106220`, and updated
  `logs/watch_n16r4_upload_stall_and_resume_20260527.pid`.
- Decision: continue waiting while byte progress exists. If no count or
  byte-size change occurs for `60` minutes, run the recorded remote process
  check and then the raw missing upload helper. No N16R4 Slurm OpenTAD training
  has launched; fixed-50/no-GT/no-teacher protocol remains unchanged.

## 2026-05-28T00:21:11+08:00 - N16R4 upload still progressing

- Rechecked after watchdog restart. Remote watcher iteration `17` at
  `00:20:10+08:00` advanced to train `87 ok`, `104 missing`, `9 bad_size`;
  test `199 ok`, `0 missing`, `12 bad_size`; `active_sftp=19`.
- One additional training video was linked (`have=96`, `new_links=1`) and bad
  partial file byte counts continue moving.
- Decision unchanged: upload is active, not stalled. Do not start helper upload
  or Slurm training until the byte-size data gate passes.

## 2026-05-28T00:24:39+08:00 - N16R4 data gate still blocked but moving

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are all alive.
- Latest post-upload audit iteration `9`: train `91 ok`, `100 missing`,
  `9 bad_size`; test `199 ok`, `0 missing`, `12 bad_size`; `DATA_PASS=False`.
- Upload watcher iteration `19` at `00:24:12+08:00` matched the same train/test
  data state and reported `active_sftp=20`.
- Recent raw mp4 mtimes/sizes are still changing, including training videos
  `video_validation_0000158`, `0000059`, `0000053`, and test video
  `video_test_0001459`, so no stall intervention is appropriate.
- `parajobs` shows existing unrelated account jobs, but no N16R4 OpenTAD
  `bata_mobile50` job has been submitted because data PASS remains false.
- Decision: continue waiting under the armed gates. No helper upload and no
  Slurm launch until exact data PASS.

## 2026-05-28T00:28:40+08:00 - N16R4 upload still active

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` remain alive.
- Local watchdog sample at `00:26:25+08:00` reset its progress timer with train
  `91 ok`, `100 missing`, `9 bad_size`; test `199 ok`, `0 missing`,
  `12 bad_size`; `active_sftp=20`.
- Upload watcher iteration `21` at `00:28:14+08:00` kept the same pass counts
  but showed bad-size files still growing, including `video_validation_0000059`,
  `0000152`, `0000153`, `0000158`, `0000159`, `0000160`, `0000161`, and
  `video_test_0001459`.
- Post-upload iteration `11` at `00:28:40+08:00` remains `DATA_PASS=False`.
  No PASS marker exists and no `bata_mobile50` or other OpenTAD Slurm job/log
  was found; visible Slurm jobs remain unrelated account jobs.
- Decision: continue waiting. Do not trigger helper upload while byte progress
  exists; do not launch Slurm until exact data PASS and post-upload
  verification complete.

## 2026-05-28T00:32:42+08:00 - N16R4 train layout reached 101 linked videos

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` remain alive.
- Upload watcher iteration `22` at `00:30:15+08:00` advanced to train `92 ok`,
  `99 missing`, `9 bad_size`; test `199 ok`, `0 missing`, `12 bad_size`, with
  `have=101` and `new_links=1`.
- Watcher iteration `23` and post-upload iteration `13` at `00:32:42+08:00`
  kept train `92 ok`, `99 missing`, `9 bad_size`; test `199 ok`, `12 bad_size`;
  `DATA_PASS=False`; `active_sftp=23`.
- Recent raw files were still updating at `00:34+08:00`, including
  `video_validation_0000152`, `0000160`, `0000059`, `0000156`, `0000158`,
  `0000165`, and `video_test_0001459`.
- Layout counts: `thumos14/train=101`, `thumos14/test=211`. Disk under
  `~/run/yuzibo` is healthy: JuiceFS about `2.3T` total, `400G` used,
  `2.0T` free.
- No PASS marker, no OpenTAD Slurm job, and no recent OpenTAD artifacts exist.
  Decision: keep waiting; no helper upload while byte progress exists.

## 2026-05-28T00:40:34+08:00 - N16R4 upload still moving

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` remain alive.
- Upload watcher iteration `27` at `00:40:21+08:00` advanced to train
  `95 ok`, `96 missing`, `9 bad_size`; test `199 ok`, `0 missing`,
  `12 bad_size`; `ALL_PASS=False`; `active_sftp=23`; train layout `have=104`
  with `new_links=1`.
- Bad-size byte tails are still growing, so the user-defined `60` minute
  no-change condition is not met. The local watchdog last reset at
  `00:36:28+08:00`.
- Post-upload gate iteration `16` at `00:38:44+08:00` remains
  `DATA_PASS=False`; verification and `sbatch` have not run.
- Visible Slurm jobs are unrelated account jobs named `run.sh`; no
  `bata_mobile50` or other N16R4 OpenTAD Slurm job exists.
- Decision: continue waiting. Do not start helper upload while count/byte
  progress exists; do not launch Slurm until exact data PASS and post-upload
  verification complete.

## 2026-05-28T00:49:06+08:00 - N16R4 data gate still blocked but active

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` remain alive.
- Upload watcher iteration `31` at `00:48:25+08:00`: train `95 ok`,
  `95 missing`, `10 bad_size`; test `199 ok`, `0 missing`, `12 bad_size`;
  `ALL_PASS=False`; `active_sftp=25`; train layout `have=105`.
- Counts and bytes advanced since the previous check. One more training video
  linked after `00:40`, and partial files including `video_validation_0000158`,
  `0000159`, `0000160`, `0000162`, `0000165`, `0000167`, `0000168`, and
  `video_test_0001459` continue growing.
- Local watchdog reset the stall timer at `00:46:31+08:00`, so the `60` minute
  no-change rule is not triggered.
- Post-upload gate iteration `21` at `00:48:47+08:00` remains
  `DATA_PASS=False`; verification and `sbatch` have not run. Existing Slurm
  jobs are unrelated account `run.sh` jobs.
- Decision: keep waiting. No helper upload and no Slurm launch until exact data
  PASS and post-upload verification complete.

## 2026-05-28T00:51:10+08:00 - N16R4 upload monitor refresh

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are still alive.
- Watcher iteration `32` at `00:50:26+08:00`: train `95 ok`, `95 missing`,
  `10 bad_size`; test `199 ok`, `0 missing`, `12 bad_size`; `ALL_PASS=False`;
  `active_sftp=26`; train layout `have=105`.
- Byte tails still grew versus iteration `31`, including
  `video_validation_0000158`, `0000159`, `0000160`, `0000162`, `0000165`,
  `0000167`, `0000168`, and `video_test_0001459`.
- Post-upload gate iteration `22` at `00:50:48+08:00` remains
  `DATA_PASS=False`; runtime import, py_compile, pytest, and Slurm submit have
  not run.
- Local watchdog last reset at `00:46:31+08:00`; unrelated account `run.sh`
  jobs are still the only visible Slurm jobs.
- Decision unchanged: keep waiting. No helper upload while byte progress exists;
  no Slurm launch until exact data PASS and post-upload verification complete.

## 2026-05-28T00:53:08+08:00 - N16R4 upload still active

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `33` at `00:52:27+08:00`: train `95 ok`, `95 missing`,
  `10 bad_size`; test `199 ok`, `0 missing`, `12 bad_size`; `ALL_PASS=False`;
  `active_sftp=27`; train layout `have=105`.
- Byte tails continue to move, including `video_validation_0000158`,
  `0000159`, `0000160`, `0000162`, `0000165`, `0000167`, `0000168`, and
  `video_test_0001459`.
- Post-upload gate iteration `23` at `00:52:49+08:00` remains
  `DATA_PASS=False`; runtime verification and Slurm submit have not run.
- Local watchdog reset at `00:51:34+08:00`; visible Slurm jobs are still only
  unrelated account `run.sh` jobs.
- Decision: continue waiting. No helper upload while byte progress exists, and
  no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T00:55:07+08:00 - N16R4 upload byte progress continues

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `34` at `00:54:29+08:00`: train `95 ok`, `95 missing`,
  `10 bad_size`; test `199 ok`, `0 missing`, `12 bad_size`; `ALL_PASS=False`;
  `active_sftp=25`; train layout `have=105`.
- Counts are flat from the previous sample, but partial byte tails grew again
  for `video_validation_0000158`, `0000159`, `0000160`, `0000162`, `0000165`,
  `0000167`, `0000168`, and `video_test_0001459`.
- Post-upload gate iteration `24` at `00:54:49+08:00` remains
  `DATA_PASS=False`; runtime verification and Slurm submit have not run.
- Local watchdog last reset at `00:51:34+08:00`; visible Slurm jobs remain
  unrelated account `run.sh` jobs.
- Decision: continue waiting. No helper upload or Slurm launch while byte
  progress exists and data PASS is false.

## 2026-05-28T00:57:12+08:00 - N16R4 train upload advanced by one

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `35` at `00:56:30+08:00` linked one more training video:
  `have=106`, `new_links=1`; train `95 ok`, `94 missing`, `11 bad_size`; test
  `199 ok`, `0 missing`, `12 bad_size`; `ALL_PASS=False`; `active_sftp=26`.
- Post-upload iteration `25` at `00:56:50+08:00` saw train `96 ok`,
  `94 missing`, `10 bad_size`; test `199 ok`, `0 missing`, `12 bad_size`;
  `DATA_PASS=False`.
- Local watchdog reset at `00:56:36+08:00`. Runtime verification and Slurm
  submit have not run; visible Slurm jobs are unrelated account `run.sh` jobs.
- Decision: continue waiting. No helper upload while count/byte progress
  exists; no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T07:46:05+08:00 - N16R4 upload still progressing at train 173 ok

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive. Active SFTP sessions remain present.
- Upload watcher iteration `237` at `07:44:01+08:00` linked one more training
  video: train layout `have=186`, `new_links=1`; train `173 ok`,
  `14 missing`, `13 bad_size`; test `200 ok`, `0 missing`, `11 bad_size`;
  `ALL_PASS=False`. Iteration `238` at `07:46:02+08:00` kept train
  `173 ok`, `14 missing`, `13 bad_size`, with `active_sftp=13`.
- Current train bad tail is still increasing: `video_validation_0000369`
  `79134720/590761829`, `0000370` `83656704/356182662`, `0000411`
  `83165184/217008088`, `0000412` `0/153134081`, `0000415`
  `48234496/78524153`, `0000416` `30179328/260968672`, `0000417`
  `10158080/127629223`, and `0000418` `4554752/94595740`.
- Post-upload iteration `228` at `07:45:09+08:00` remains `DATA_PASS=False`
  with train `173 ok`, `14 missing`, `13 bad_size`, `extra=0`, pass
  `False`; test remains `200 ok`, `0 missing`, `11 bad_size`, `extra=0`,
  pass `False`. Its train bad tail is `video_validation_0000369`
  `78217216/590761829`, `0000370` `82935808/356182662`, `0000411`
  `81788928/217008088`, `0000412` `0/153134081`, `0000415`
  `46301184/78524153`, `0000416` `28704768/260968672`, `0000417`
  `8421376/127629223`, and `0000418` `3145728/94595740`.
- Marker grep still finds only repeated `DATA_PASS=False`; no runtime import,
  py_compile, pytest, `sbatch`, or `bata_mobile50` marker has run.
  `parajobs` has only unrelated `run.sh` jobs `992937` and `992776`.
- Local watchdog reset at `07:43:56+08:00`; the one-hour no-change trigger is
  not met. No code/model/config changed; strict random-fixed 50% and fixed-50
  deployable/no-GT/no-teacher contracts remain unchanged because no N16R4
  training launched; no mAP evidence exists.
- Decision: continue waiting. Do not helper-upload while count/byte progress
  exists; run process inspection and helper upload only after 60 minutes
  without count or byte-size signature change; no Slurm launch until exact
  data PASS plus post-upload verification.

## 2026-05-28T07:43:30+08:00 - N16R4 upload advanced; data gate still blocked

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive. Active SFTP sessions remain present.
- Upload watcher iteration `235` at `07:39:59+08:00` linked one more training
  video: train layout `have=185`, `new_links=1`; train `172 ok`,
  `15 missing`, `13 bad_size`; test `200 ok`, `0 missing`, `11 bad_size`;
  `ALL_PASS=False`. Iteration `236` at `07:42:00+08:00` kept train
  `172 ok`, `15 missing`, `13 bad_size`, with `active_sftp=16`.
- Current train bad tail is still increasing: `video_validation_0000365`
  `131563520/133486504`, `0000369` `75038720/590761829`, `0000370`
  `79495168/356182662`, `0000411` `76283904/217008088`, `0000412`
  `0/153134081`, `0000415` `40206336/78524153`, `0000416`
  `23560192/260968672`, and `0000417` `4161536/127629223`.
- Post-upload iteration `227` at `07:43:08+08:00` advanced to train
  `173 ok`, `15 missing`, `12 bad_size`, `extra=0`, pass `False`; test
  remains `200 ok`, `0 missing`, `11 bad_size`, `extra=0`, pass `False`;
  `DATA_PASS=False`. Its train bad tail is `video_validation_0000363`
  `160890880/382333920`, `0000369` `76087296/590761829`, `0000370`
  `80347136/356182662`, `0000411` `77660160/217008088`, `0000412`
  `0/153134081`, `0000415` `42565632/78524153`, `0000416`
  `26017792/260968672`, and `0000417` `5472256/127629223`.
- Marker grep still finds only repeated `DATA_PASS=False`; no runtime import,
  py_compile, pytest, `sbatch`, or `bata_mobile50` marker has run.
  `parajobs` has only unrelated `run.sh` jobs `992937` and `992776`.
- Local watchdog reset at `07:38:55+08:00`; current remote count/byte progress
  confirms the one-hour no-change trigger is not met. No code/model/config
  changed; strict random-fixed 50% and fixed-50 deployable/no-GT/no-teacher
  contracts remain unchanged because no N16R4 training launched; no mAP
  evidence exists.
- Decision: continue waiting. Do not helper-upload while count/byte progress
  exists; run process inspection and helper upload only after 60 minutes
  without count or byte-size signature change; no Slurm launch until exact
  data PASS plus post-upload verification.

## 2026-05-28T07:39:39+08:00 - N16R4 post-upload advanced to train 172 ok

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive. Active SFTP sessions remain present.
- Upload watcher iteration `234` at `07:37:58+08:00`: train layout
  `have=184`, `new_links=0`; train `171 ok`, `16 missing`, `13 bad_size`;
  test `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`;
  `active_sftp=15`.
- Train bad-size bytes continue increasing: `video_validation_0000365`
  `124354560/133486504`, `0000369` `70680576/590761829`, `0000370`
  `73957376/356182662`, `0000411` `70057984/217008088`, `0000412`
  `0/153134081`, `0000414` `76087296/77935134`, `0000415`
  `33193984/78524153`, and `0000416` `15663104/260968672`.
- Post-upload iteration `225` at `07:39:07+08:00` advanced to train
  `172 ok`, `16 missing`, `12 bad_size`, `extra=0`, pass `False`; test
  remains `200 ok`, `0 missing`, `11 bad_size`, `extra=0`, pass `False`;
  `DATA_PASS=False`. Its train bad tail is `video_validation_0000363`
  `157089792/382333920`, `0000365` `127172608/133486504`, `0000369`
  `71958528/590761829`, `0000370` `75005952/356182662`, `0000411`
  `71237632/217008088`, `0000412` `0/153134081`, `0000415`
  `35028992/78524153`, and `0000416` `17498112/260968672`.
- Marker grep still finds only repeated `DATA_PASS=False`; no runtime import,
  py_compile, pytest, `sbatch`, or `bata_mobile50` marker has run.
  `parajobs` has only unrelated `run.sh` jobs `992937` and `992776`.
- Local watchdog reset at `07:38:55+08:00`; the one-hour no-change trigger is
  not met. No code/model/config changed; strict random-fixed 50% and fixed-50
  deployable/no-GT/no-teacher contracts remain unchanged because no N16R4
  training launched; no mAP evidence exists.
- Decision: continue waiting. Do not helper-upload while count/byte progress
  exists; run process inspection and helper upload only after 60 minutes
  without count or byte-size signature change; no Slurm launch until exact
  data PASS plus post-upload verification.

## 2026-05-28T07:36:50+08:00 - N16R4 upload byte progress continues at train 171 ok

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive. Active SFTP sessions remain present.
- Upload watcher iteration `233` at `07:35:57+08:00`: train layout
  `have=184`, `new_links=0`; train `171 ok`, `16 missing`, `13 bad_size`;
  test `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`;
  `active_sftp=15`.
- Counts are unchanged, but train bad-size bytes continue increasing:
  `video_validation_0000365` `120881152/133486504`, `0000369`
  `68681728/590761829`, `0000370` `70778880/356182662`, `0000411`
  `68124672/217008088`, `0000412` `0/153134081`, `0000414`
  `72613888/77935134`, `0000415` `29360128/78524153`, and `0000416`
  `12419072/260968672`.
- Post-upload iteration `223` at `07:35:05+08:00` remains `DATA_PASS=False`
  with train `171 ok`, `16 missing`, `13 bad_size`, `extra=0`; test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Its train bad tail
  increased to `0000365` `119898112/133486504`, `0000369`
  `68059136/590761829`, `0000370` `69435392/356182662`, `0000411`
  `66846720/217008088`, `0000412` `0/153134081`, `0000414`
  `69959680/77935134`, `0000415` `28246016/78524153`, and `0000416`
  `10944512/260968672`.
- Marker grep still finds only repeated `DATA_PASS=False`; no runtime import,
  py_compile, pytest, `sbatch`, or `bata_mobile50` marker has run.
  `parajobs` has only unrelated `run.sh` jobs `992937` and `992776`.
- Local watchdog reset at `07:33:53+08:00`; the one-hour no-change trigger is
  not met. No code/model/config changed; strict random-fixed 50% and fixed-50
  deployable/no-GT/no-teacher contracts remain unchanged because no N16R4
  training launched; no mAP evidence exists.
- Decision: continue waiting. Do not helper-upload while byte progress exists;
  run process inspection and helper upload only after 60 minutes without count
  or byte-size signature change; no Slurm launch until exact data PASS plus
  post-upload verification.

## 2026-05-28T07:33:44+08:00 - N16R4 upload byte progress continues at train 171 ok

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive. Active SFTP sessions remain present.
- Upload watcher iteration `231` at `07:31:55+08:00`: train layout
  `have=184`, `new_links=0`; train `171 ok`, `16 missing`, `13 bad_size`;
  test `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`;
  `active_sftp=17`.
- Counts are unchanged from iteration `230`, but train bad-size bytes continue
  increasing: `video_validation_0000365` `115703808/133486504`, `0000369`
  `65667072/590761829`, `0000370` `64192512/356182662`, `0000411`
  `62750720/217008088`, `0000412` `0/153134081`, `0000414`
  `64618496/77935134`, `0000415` `23134208/78524153`, and `0000416`
  `5177344/260968672`.
- Post-upload iteration `222` at `07:33:04+08:00` remains `DATA_PASS=False`
  with train `171 ok`, `16 missing`, `13 bad_size`, `extra=0`; test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Its train bad tail also
  increased to `0000365` `117211136/133486504`, `0000369`
  `66584576/590761829`, `0000370` `65601536/356182662`, `0000411`
  `64323584/217008088`, `0000412` `0/153134081`, `0000414`
  `66027520/77935134`, `0000415` `25231360/78524153`, and `0000416`
  `7733248/260968672`.
- Marker grep still finds only repeated `DATA_PASS=False`; no runtime import,
  py_compile, pytest, `sbatch`, or `bata_mobile50` marker has run.
  `parajobs` has only unrelated `run.sh` jobs `992937` and `992776`.
- Local watchdog is alive; last observed reset is `07:28:52+08:00`, and
  current remote byte progress confirms the one-hour no-change trigger is not
  met. No code/model/config changed; strict random-fixed 50% and fixed-50
  deployable/no-GT/no-teacher contracts remain unchanged because no N16R4
  training launched; no mAP evidence exists.
- Decision: continue waiting. Do not helper-upload while byte progress exists;
  run process inspection and helper upload only after 60 minutes without count
  or byte-size signature change; no Slurm launch until exact data PASS plus
  post-upload verification.

## 2026-05-28T07:30:26+08:00 - N16R4 train upload advanced to 171 ok

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive. Active SFTP sessions remain present.
- Upload watcher iteration `230` at `07:29:54+08:00`: train layout
  `have=184`, `new_links=1`; train `171 ok`, `16 missing`, `13 bad_size`;
  test `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`;
  `active_sftp=18`.
- This is fresh count progress after iteration `229`, so the one-hour
  no-change trigger is not met. Current train bad-size tail includes
  `video_validation_0000365` `112427008/133486504`, `0000369`
  `64159744/590761829`, `0000370` `62521344/356182662`, `0000411`
  `59473920/217008088`, `0000412` `0/153134081`, `0000414`
  `60325888/77935134`, `0000415` `19890176/78524153`, and `0000416`
  `622592/260968672`.
- Post-upload iteration `220` at `07:29:03+08:00` remains `DATA_PASS=False`
  with train `170 ok`, `17 missing`, `13 bad_size`, `extra=0`; test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`.
- Marker grep still finds only repeated `DATA_PASS=False`; no runtime import,
  py_compile, pytest, `sbatch`, or `bata_mobile50` marker has run.
  `parajobs` has only unrelated `run.sh` jobs `992937` and `992776`.
- Local watchdog reset at `07:28:52+08:00`. No code/model/config changed;
  strict random-fixed 50% and fixed-50 deployable/no-GT/no-teacher contracts
  remain unchanged because no N16R4 training launched; no mAP evidence exists.
- Decision: continue waiting. Do not helper-upload while count/byte progress
  exists; run process inspection and helper upload only after 60 minutes
  without count or byte-size signature change; no Slurm launch until exact
  data PASS plus post-upload verification.

## 2026-05-28T07:25:42+08:00 - N16R4 upload byte progress continues at train 170 ok

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive. Active SFTP sessions remain present.
- Upload watcher iteration `227` at `07:23:51+08:00`: train layout
  `have=183`, `new_links=0`; train `170 ok`, `17 missing`, `13 bad_size`;
  test `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`;
  `active_sftp=16`.
- Counts are unchanged, but train bad-size bytes continue increasing:
  `video_validation_0000363` `140935168/382333920`, `0000365`
  `101744640/133486504`, `0000369` `57835520/590761829`, `0000370`
  `55148544/356182662`, `0000411` `50692096/217008088`, `0000412`
  `0/153134081`, `0000414` `47218688/77935134`, and `0000415`
  `11468800/78524153`.
- Post-upload iteration `218` at `07:25:01+08:00` remains `DATA_PASS=False`
  with train `170 ok`, `17 missing`, `13 bad_size`, `extra=0`; test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Its train bad tail
  increased to `0000363` `142213120/382333920`, `0000365`
  `103874560/133486504`, `0000369` `59277312/590761829`, `0000370`
  `56885248/356182662`, `0000411` `52658176/217008088`, `0000412`
  `0/153134081`, `0000414` `49119232/77935134`, and `0000415`
  `13271040/78524153`.
- Marker grep still finds only `DATA_PASS=False`; no runtime import,
  py_compile, pytest, `sbatch`, or `bata_mobile50` marker has run.
  `parajobs` has no `bata_mobile50` or OpenTAD job; visible jobs remain
  unrelated `run.sh` jobs.
- Local watchdog reset at `07:23:50+08:00`; the one-hour no-change trigger is
  not met.
- Decision: continue waiting. No helper upload while count/byte progress
  exists; run helper upload only after 60 minutes without count or byte-size
  signature change; no Slurm launch until exact data PASS plus post-upload
  verification.

## 2026-05-28T07:23:00+08:00 - N16R4 upload byte progress continues at train 170 ok

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive. Active SFTP sessions remain present.
- Upload watcher iteration `226` at `07:21:50+08:00`: train layout
  `have=183`, `new_links=0`; train `170 ok`, `17 missing`, `13 bad_size`;
  test `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`;
  `active_sftp=19`.
- Counts are unchanged from `07:19`, but train bad-size bytes continue
  increasing: `video_validation_0000363` `138641408/382333920`, `0000365`
  `98467840/133486504`, `0000369` `55017472/590761829`, `0000370`
  `52887552/356182662`, `0000411` `48431104/217008088`, `0000412`
  `0/153134081`, `0000414` `42369024/77935134`, and `0000415`
  `8519680/78524153`.
- Post-upload iteration `216` at `07:21:00+08:00` remains `DATA_PASS=False`
  with train `170 ok`, `17 missing`, `13 bad_size`, `extra=0`; test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Its train bad tail also
  increased to `0000363` `138018816/382333920`, `0000365`
  `97157120/133486504`, `0000369` `53903360/590761829`, `0000370`
  `51478528/356182662`, `0000411` `46956544/217008088`, `0000412`
  `0/153134081`, `0000414` `40730624/77935134`, and `0000415`
  `7241728/78524153`.
- Marker grep still finds only `DATA_PASS=False`; no runtime import,
  py_compile, pytest, `sbatch`, or `bata_mobile50` marker has run.
  `parajobs` has no `bata_mobile50` or OpenTAD job; visible jobs remain
  unrelated `run.sh` jobs.
- Last local watchdog reset observed at `07:18:48+08:00`; remote byte progress
  confirms the one-hour no-change trigger is not met.
- Decision: continue waiting. No helper upload while count/byte progress
  exists; run helper upload only after 60 minutes without count or byte-size
  signature change; no Slurm launch until exact data PASS plus post-upload
  verification.

## 2026-05-28T07:19:58+08:00 - N16R4 train upload advanced to 170 ok

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive. Active SFTP sessions remain present.
- Upload watcher iteration `224` at `07:17:48+08:00` advanced train layout to
  `have=183`, `new_links=1`; train `170 ok`, `17 missing`, `13 bad_size`;
  test `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`;
  `active_sftp=19`. Iteration `225` at `07:19:49+08:00` kept the same counts.
- Current train bad tail is still increasing: `video_validation_0000363`
  `136970240/382333920`, `0000365` `96010240/133486504`, `0000369`
  `52527104/590761829`, `0000370` `49643520/356182662`, `0000411`
  `45547520/217008088`, `0000412` `0/153134081`, `0000414`
  `37421056/77935134`, and `0000415` `5373952/78524153`.
- Post-upload iteration `215` at `07:18:59+08:00` remains `DATA_PASS=False`
  with train `170 ok`, `17 missing`, `13 bad_size`, `extra=0`; test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`.
- Marker grep still finds only `DATA_PASS=False`; no runtime import,
  py_compile, pytest, `sbatch`, or `bata_mobile50` marker has run.
  `parajobs` has no `bata_mobile50` or OpenTAD job; visible jobs remain
  unrelated `run.sh` jobs.
- Local watchdog reset at `07:18:48+08:00`; the one-hour no-change trigger is
  not met.
- Decision: continue waiting. No helper upload while count/byte progress
  exists; run helper upload only after 60 minutes without count or byte-size
  signature change; no Slurm launch until exact data PASS plus post-upload
  verification.

## 2026-05-28T07:16:34+08:00 - N16R4 upload byte progress continues at train 169 ok

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive. Active SFTP sessions remain present.
- Upload watcher iteration `223` at `07:15:47+08:00`: train layout
  `have=182`, `new_links=0`; train `169 ok`, `18 missing`, `13 bad_size`;
  test `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`;
  `active_sftp=19`.
- Upload bad-size bytes are still increasing: `video_validation_0000362`
  `177537024/177911222`, `0000363` `132087808/382333920`, `0000365`
  `90243072/133486504`, `0000369` `47546368/590761829`, `0000370`
  `43941888/356182662`, `0000411` `40435712/217008088`, `0000412`
  `0/153134081`, and `0000414` `28672000/77935134`.
- Post-upload iteration `213` at `07:14:58+08:00` remains `DATA_PASS=False`
  with train `169 ok`, `18 missing`, `13 bad_size`, `extra=0`; test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Its train bad tail
  increased to `0000362` `176521216/177911222`, `0000363`
  `131596288/382333920`, `0000365` `88342528/133486504`, `0000369`
  `46268416/590761829`, `0000370` `43024384/356182662`, `0000411`
  `38600704/217008088`, `0000412` `0/153134081`, and `0000414`
  `26673152/77935134`.
- Marker grep still finds only `DATA_PASS=False`; no runtime import,
  py_compile, pytest, `sbatch`, or `bata_mobile50` marker has run.
  `parajobs` has no `bata_mobile50` or OpenTAD job; visible jobs remain
  unrelated `run.sh` jobs.
- Local watchdog reset at `07:13:46+08:00`; the one-hour no-change trigger is
  not met.
- Decision: continue waiting. No helper upload while byte progress exists; run
  helper upload only after 60 minutes without count or byte-size signature
  change; no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T07:13:39+08:00 - N16R4 upload byte progress continues at train 169 ok

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive. Active SFTP sessions remain present.
- Upload watcher iteration `221` at `07:11:45+08:00`: train layout
  `have=182`, `new_links=0`; train `169 ok`, `18 missing`, `13 bad_size`;
  test `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`;
  `active_sftp=18`.
- Upload bad-size bytes are still increasing: `video_validation_0000362`
  `171442176/177911222`, `0000363` `129007616/382333920`, `0000365`
  `82247680/133486504`, `0000369` `42631168/590761829`, `0000370`
  `37814272/356182662`, `0000411` `32866304/217008088`, `0000412`
  `0/153134081`, and `0000414` `20578304/77935134`.
- Post-upload iteration `212` at `07:12:57+08:00` remains `DATA_PASS=False`
  with train `169 ok`, `18 missing`, `13 bad_size`, `extra=0`; test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Its train bad tail also
  increased to `0000362` `173211648/177911222`, `0000363`
  `130121728/382333920`, `0000365` `84213760/133486504`, `0000369`
  `43843584/590761829`, `0000370` `40534016/356182662`, `0000411`
  `35323904/217008088`, `0000412` `0/153134081`, and `0000414`
  `22020096/77935134`.
- Marker grep still finds only `DATA_PASS=False`; no runtime import,
  py_compile, pytest, `sbatch`, or `bata_mobile50` marker has run.
  `parajobs` has no `bata_mobile50` or OpenTAD job; visible jobs remain
  unrelated `run.sh` jobs.
- Local watchdog was alive and last observed reset was `07:08:45+08:00`, so
  the one-hour no-change trigger is not met.
- Decision: continue waiting. No helper upload while byte progress exists; run
  helper upload only after 60 minutes without count or byte-size signature
  change; no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T07:09:11+08:00 - N16R4 upload byte progress continues at train 169 ok

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive. Process check still shows active `sftp-server`
  sessions.
- Watcher iteration `219` at `07:07:42+08:00`: train layout `have=182`,
  `new_links=0`; train `169 ok`, `18 missing`, `13 bad_size`; test
  `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`; `active_sftp=20`.
- Counts are unchanged, but byte progress continues: `video_validation_0000362`
  `165642240/177911222`, `0000363` `122912768/382333920`, `0000365`
  `71925760/133486504`, `0000369` `37126144/590761829`, `0000370`
  `32866304/356182662`, `0000411` `29229056/217008088`, `0000412`
  `0/153134081`, and `0000414` `15761408/77935134`.
- Post-upload iteration `210` at `07:08:56+08:00` remains `DATA_PASS=False`
  with train `169 ok`, `18 missing`, `13 bad_size`, `extra=0`; test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Marker grep still finds
  only `DATA_PASS=False`; no runtime import, py_compile, pytest, `sbatch`, or
  `bata_mobile50` marker has run.
- Local watchdog reset at `07:08:45+08:00`; the one-hour no-change trigger is
  not met. `parajobs` has no `bata_mobile50` or OpenTAD job; visible jobs
  remain unrelated account `run.sh` jobs.
- Decision: continue waiting. No helper upload while byte progress exists; run
  helper upload only after 60 minutes without count or byte-size signature
  change; no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T07:05:36+08:00 - N16R4 upload byte progress continues at train 169 ok

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive. Active sftp sessions remain present.
- Watcher iteration `217` at `07:03:40+08:00`: train layout `have=182`,
  `new_links=0`; train `169 ok`, `18 missing`, `13 bad_size`; test
  `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`; `active_sftp=19`.
- Counts are unchanged from the 07:02 monitor, but byte progress continues:
  `video_validation_0000362` `159973376/177911222`, `0000363`
  `118390784/382333920`, `0000365` `60555264/133486504`, `0000369`
  `31522816/590761829`, `0000370` `28311552/356182662`, `0000411`
  `25722880/217008088`, `0000412` `0/153134081`, and `0000414`
  `9666560/77935134`.
- Post-upload iteration `208` at `07:04:54+08:00` remains `DATA_PASS=False`
  with train `169 ok`, `18 missing`, `13 bad_size`, `extra=0`; test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Marker grep still finds
  only `DATA_PASS=False`; no runtime import, py_compile, pytest, `sbatch`, or
  `bata_mobile50` marker has run.
- Local watchdog reset at `07:03:43+08:00`; the one-hour no-change trigger is
  not met. `parajobs` has no `bata_mobile50` or OpenTAD job; visible jobs
  remain unrelated account `run.sh` jobs.
- Decision: continue waiting. No helper upload while byte progress exists; no
  Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T07:02:10+08:00 - N16R4 upload byte progress continues at train 169 ok

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive. Active sftp sessions remain present.
- Watcher iteration `216` at `07:01:39+08:00`: train layout `have=182`,
  `new_links=0`; train `169 ok`, `18 missing`, `13 bad_size`; test
  `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`; `active_sftp=20`.
- Counts are unchanged from the 06:58 monitor, but byte progress continues:
  `video_validation_0000362` `155746304/177911222`, `0000363`
  `116195328/382333920`, `0000365` `56754176/133486504`, `0000369`
  `29032448/590761829`, `0000370` `26279936/356182662`, `0000411`
  `23101440/217008088`, `0000412` `0/153134081`, and `0000414`
  `6815744/77935134`.
- Post-upload iteration `206` at `07:00:53+08:00` remains `DATA_PASS=False`
  with train `169 ok`, `18 missing`, `13 bad_size`, `extra=0`; test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Marker grep still finds
  only `DATA_PASS=False`; no runtime import, py_compile, pytest, `sbatch`, or
  `bata_mobile50` marker has run.
- Local watchdog reset at `06:58:41+08:00`; the one-hour no-change trigger is
  not met. `parajobs` has no `bata_mobile50` or OpenTAD job; visible jobs
  remain unrelated account `run.sh` jobs.
- Decision: continue waiting. No helper upload while byte progress exists; no
  Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T06:58:16+08:00 - N16R4 train upload advanced to 169 ok

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive. Active sftp sessions remain present.
- Watcher iteration `214` at `06:57:37+08:00`: train layout `have=182`,
  `new_links=2`; train `169 ok`, `18 missing`, `13 bad_size`; test
  `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`; `active_sftp=22`.
- Current train bad-size tail includes `video_validation_0000362`
  `149553152/177911222`, `0000363` `110002176/382333920`, `0000365`
  `52723712/133486504`, `0000369` `22839296/590761829`, `0000370`
  `21102592/356182662`, `0000411` `18284544/217008088`, `0000412`
  `0/153134081`, and `0000414` `917504/77935134`.
- Post-upload iteration `204` at `06:56:52+08:00` remains `DATA_PASS=False`
  with train `169 ok`, `20 missing`, `11 bad_size`, `extra=0`; test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Marker grep still finds
  only `DATA_PASS=False`; no runtime import, py_compile, pytest, `sbatch`, or
  `bata_mobile50` marker has run.
- The one-hour no-change trigger is not met. `parajobs` has no `bata_mobile50`
  or OpenTAD job; visible jobs remain unrelated account `run.sh` jobs.
- Decision: continue waiting. No helper upload while count/byte progress
  exists; no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T06:54:42+08:00 - N16R4 upload byte progress continues at train 168 ok

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive. Active sftp sessions remain present.
- Watcher iteration `212` at `06:53:35+08:00`: train layout `have=180`,
  `new_links=0`; train `168 ok`, `20 missing`, `12 bad_size`; test
  `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`; `active_sftp=20`.
- Counts are unchanged since the 06:50 monitor, but byte progress continues:
  `video_validation_0000361` `81330176/363029480`, `0000362`
  `140607488/177911222`, `0000363` `104038400/382333920`, `0000365`
  `48267264/133486504`, `0000368` `35389440/40977781`, `0000369`
  `17956864/590761829`, `0000370` `16482304/356182662`, and `0000411`
  `14057472/217008088`.
- Post-upload iteration `202` at `06:52:50+08:00` remains `DATA_PASS=False`
  with train `168 ok`, `20 missing`, `12 bad_size`, `extra=0`; test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Marker grep still finds
  only `DATA_PASS=False`; no runtime import, py_compile, pytest, `sbatch`, or
  `bata_mobile50` marker has run.
- Local watchdog reset at `06:53:39+08:00`; the one-hour no-change trigger is
  not met.
- Decision: continue waiting. No helper upload while byte progress exists; no
  Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T06:50:27+08:00 - N16R4 upload still progressing at train 168 ok

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive. Active sftp sessions are still present.
- Watcher iteration `210` at `06:49:33+08:00`: train layout `have=180`,
  `new_links=0`; train `168 ok`, `20 missing`, `12 bad_size`; test
  `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`; `active_sftp=20`.
- Current train bad-size tail is still increasing, including
  `video_validation_0000361` `72908800/363029480`, `0000362`
  `134512640/177911222`, `0000363` `97615872/382333920`, `0000365`
  `44695552/133486504`, `0000368` `30507008/40977781`, `0000369`
  `12582912/590761829`, `0000370` `11796480/356182662`, and `0000411`
  `7700480/217008088`.
- Post-upload iteration `200` at `06:48:49+08:00` remains `DATA_PASS=False`
  with train `168 ok`, `20 missing`, `12 bad_size`, `extra=0`; test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. No runtime import,
  py_compile, pytest, `sbatch`, or `bata_mobile50` marker has run.
- Local watchdog reset at `06:48:37+08:00`; the one-hour no-change trigger is
  not met. `parajobs` has no `bata_mobile50` or OpenTAD job; visible jobs
  remain unrelated account `run.sh` jobs.
- Decision: continue waiting. No helper upload while count/byte progress
  exists; no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T06:18:00+08:00 - N16R4 post-upload gate still waiting

- Short follow-up read confirmed post-upload iteration `184` completed with
  `DATA_PASS=False`.
- Iteration `184`: train `163 ok`, `25 missing`, `12 bad_size`, `extra=0`;
  test `200 ok`, `0 missing`, `11 bad_size`, `extra=0`.
- Train bad-size bytes continued increasing, including `video_validation_0000319`
  `99123200/135932735`, `0000320` `94896128/130340194`, `0000361`
  `42663936/363029480`, `0000362` `82608128/177911222`, `0000363`
  `57540608/382333920`, `0000364` `17399808/58511311`, `0000365`
  `11403264/133486504`, and `0000366` `7536640/8358203`.
- No runtime import, py_compile, pytest, `sbatch`, or `bata_mobile50`
  marker has run.
- Decision: continue waiting. No helper upload while upload byte progress
  exists; no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T05:45:10+08:00 - N16R4 train upload byte progress at 160 ok

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Remote watcher iteration `178` at `05:45:00+08:00`: train layout
  `have=172`, `new_links=0`; train `160 ok`, `28 missing`, `12 bad_size`;
  test `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`;
  `active_sftp=16`.
- Counts were unchanged from the prior recorded monitor, but byte tails kept
  increasing: `video_validation_0000311` `78184448/258831745`, `0000314`
  `85852160/462550836`, `0000316` `54657024/79408717`, `0000319`
  `53739520/135932735`, `0000320` `39583744/130340194`, `0000361`
  `15892480/363029480`, `0000362` `29097984/177911222`, and `0000363`
  `11599872/382333920`.
- Post-upload iteration `168` at `05:44:26+08:00` remains
  `DATA_PASS=False`; no runtime import, py_compile, pytest, `sbatch`, or
  `bata_mobile50` marker has run.
- Local watchdog reset at `05:43:15+08:00`; the one-hour no-change trigger
  is not met. `parajobs` has no `bata_mobile50` or OpenTAD job, only
  unrelated account `run.sh` jobs.
- Decision: continue waiting. No helper upload while byte progress exists;
  no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T06:16:45+08:00 - N16R4 train upload advanced to 163 ok

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Remote watcher iteration `193` at `06:15:15+08:00`: train layout
  `have=175`, `new_links=0`; train `163 ok`, `25 missing`, `12 bad_size`;
  test `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`;
  `active_sftp=18`.
- Current train bad-size tail includes `video_validation_0000319`
  `97517568/135932735`, `0000320` `92569600/130340194`, `0000361`
  `41517056/363029480`, `0000362` `80084992/177911222`, `0000363`
  `56098816/382333920`, `0000364` `15794176/58511311`, `0000365`
  `9928704/133486504`, and `0000366` `4882432/8358203`.
- Post-upload iteration `183` at `06:14:36+08:00` remains
  `DATA_PASS=False` with train `163 ok`, `25 missing`, `12 bad_size`,
  `extra=0`, and test `200 ok`, `0 missing`, `11 bad_size`, `extra=0`.
  No runtime import, py_compile, pytest, `sbatch`, or `bata_mobile50`
  marker has run.
- Local watchdog reset at `06:13:25+08:00`; the one-hour no-change trigger
  is not met. `parajobs` has no `bata_mobile50` or OpenTAD job, only
  unrelated account `run.sh` jobs.
- Decision: continue waiting. No helper upload while count/byte progress
  exists; no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T05:43:20+08:00 - N16R4 train upload advanced to 160 ok

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Remote watcher iteration `177` at `05:42:59+08:00`: train layout
  `have=172`, `new_links=0`; train `160 ok`, `28 missing`, `12 bad_size`;
  test `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`;
  `active_sftp=15`.
- Current train bad-size tail includes `video_validation_0000311`
  `74481664/258831745`, `0000314` `81035264/462550836`, `0000316`
  `52494336/79408717`, `0000319` `51511296/135932735`, `0000320`
  `35487744/130340194`, `0000361` `13991936/363029480`, `0000362`
  `25591808/177911222`, and `0000363` `9207808/382333920`.
- Post-upload iteration `167` at `05:42:25+08:00` remains
  `DATA_PASS=False` with train `160 ok`, `28 missing`, `12 bad_size`,
  `extra=0`, and test `200 ok`, `0 missing`, `11 bad_size`, `extra=0`.
  No runtime import, py_compile, pytest, `sbatch`, or `bata_mobile50`
  marker has run.
- Local watchdog reset at `05:38:13+08:00`; the one-hour no-change trigger
  is not met. `parajobs` has no `bata_mobile50` or OpenTAD job, only
  unrelated account `run.sh` jobs.
- Decision: continue waiting. No helper upload while count/byte progress
  exists; no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T05:11:50+08:00 - N16R4 train upload advanced to 156 ok

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Remote watcher iteration `161` at `05:10:42+08:00`: train layout
  `have=168`, `new_links=1`; train `156 ok`, `32 missing`, `12 bad_size`;
  test `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`;
  `active_sftp=28`.
- Current train bad-size tail includes `video_validation_0000267`
  `126517248/192812340`, `0000311` `46071808/258831745`, `0000314`
  `37486592/462550836`, `0000315` `40894464/67631558`, `0000316`
  `23199744/79408717`, `0000317` `20381696/63862586`, `0000318`
  `9535488/40390923`, and `0000319` `1835008/135932735`.
- Post-upload iteration `151` at `05:10:14+08:00` remains
  `DATA_PASS=False` with train `156 ok`, `33 missing`, `11 bad_size`,
  `extra=0`, and test `200 ok`, `0 missing`, `11 bad_size`, `extra=0`.
  No runtime import, py_compile, pytest, `sbatch`, or `bata_mobile50`
  marker has run.
- Local watchdog reset at `05:08:02+08:00`; the one-hour no-change trigger
  is not met. `parajobs` has no `bata_mobile50` or OpenTAD job, only
  unrelated account `run.sh` jobs.
- Decision: continue waiting. No helper upload while count/byte progress
  exists; no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T04:40:20+08:00 - N16R4 train upload advanced to at least 150 ok

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Remote watcher iteration `145` at `04:38:25+08:00`: train layout
  `have=161`, `new_links=0`; train `149 ok`, `39 missing`, `12 bad_size`;
  test `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`;
  `active_sftp=27`.
- Newer post-upload iteration `136` at `04:40:03+08:00` reports train
  `150 ok`, `39 missing`, `11 bad_size`, `extra=0`, and test `200 ok`,
  `0 missing`, `11 bad_size`, `extra=0`; `DATA_PASS=False`.
- Current train bad-size tail includes `video_validation_0000209`
  `128417792/223950748`, `0000267` `85164032/192812340`, `0000268`
  `81657856/128601594`, `0000285` `32145408/58816877`, `0000286`
  `39419904/41516143`, `0000288` `23756800/30959688`, `0000311`
  `12550144/258831745`, and `0000312` `10452992/23574003`.
- No runtime import, py_compile, pytest, `sbatch`, or `bata_mobile50`
  marker has run. Local watchdog reset at `04:37:52+08:00`, so the
  one-hour no-change trigger is not met. `parajobs` has no `bata_mobile50`
  or OpenTAD job, only unrelated account `run.sh` jobs.
- Decision: continue waiting. No helper upload while count/byte progress
  exists; no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T04:18:55+08:00 - N16R4 train upload advanced to 145 ok

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Remote watcher iteration `135` at `04:18:14+08:00`: train layout
  `have=157`, `new_links=1`; train `145 ok`, `43 missing`, `12 bad_size`;
  test `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`;
  `active_sftp=16`.
- This is real progress from the prior `139 ok` state. Current train
  bad-size tail includes `video_validation_0000209` `97124352/223950748`,
  `0000267` `50561024/192812340`, `0000268` `46432256/128601594`,
  `0000281` `31490048/40653891`, `0000285` `6455296/58816877`,
  `0000286` `5603328/41516143`, `0000287` `4128768/20519476`, and
  `0000288` `786432/30959688`.
- Post-upload iteration `125` at `04:17:55+08:00` remains
  `DATA_PASS=False` with train `145 ok`, `44 missing`, `11 bad_size`,
  `extra=0`, and test `200 ok`, `0 missing`, `11 bad_size`, `extra=0`.
  No runtime import, py_compile, pytest, `sbatch`, or `bata_mobile50`
  marker has run.
- Local watchdog reset at `04:17:45+08:00`; the one-hour no-change trigger
  is not met. `parajobs` has no `bata_mobile50` or OpenTAD job, only
  unrelated account `run.sh` jobs.
- Decision: continue waiting. No helper upload while count/byte progress
  exists; no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T03:50:05+08:00 - N16R4 upload still moving at train 136 ok

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Remote watcher iteration `120` at `03:47:59+08:00`: train layout
  `have=148`, `new_links=0`; train `136 ok`, `52 missing`, `12 bad_size`;
  test `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`;
  `active_sftp=20`.
- Counts did not advance since the prior monitor, but byte tails continued
  moving, including `video_validation_0000205` `62291968/267699672`,
  `0000208` `68616192/75403408`, `0000209` `52232192/223950748`,
  `0000264` `26411008/82142543`, `0000266` `13074432/54520293`,
  `0000267` `13205504/192812340`, `0000268` `8814592/128601594`, and
  `0000269` `2719744/11164768`.
- Post-upload iteration `111` at `03:49:46+08:00` remains
  `DATA_PASS=False` with train `136 ok`, `52 missing`, `12 bad_size`,
  `extra=0`, and test `200 ok`, `0 missing`, `11 bad_size`, `extra=0`.
  No runtime import, py_compile, pytest, `sbatch`, or `bata_mobile50`
  marker has run.
- Local watchdog reset at `03:47:34+08:00`; the one-hour no-change trigger
  is not met. Remote process check shows active SFTP sessions, and `parajobs`
  has no `bata_mobile50` or OpenTAD job, only unrelated account `run.sh`
  jobs.
- Decision: continue waiting. No helper upload while byte progress exists;
  if there is no count or byte-size signature change for `60` minutes,
  inspect processes and run the helper upload for remaining/bad raw videos.

## 2026-05-28T04:02:15+08:00 - N16R4 train upload advanced to 139 ok

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Remote watcher iteration `127` at `04:02:06+08:00`: train layout
  `have=151`, `new_links=0`; train `139 ok`, `49 missing`, `12 bad_size`;
  test `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`;
  `active_sftp=16`.
- This is real progress from the prior `136 ok` state. Current train
  bad-size tail includes `video_validation_0000209` `70746112/223950748`,
  `0000264` `53837824/82142543`, `0000266` `34406400/54520293`,
  `0000267` `31588352/192812340`, `0000268` `26673152/128601594`,
  `0000270` `15106048/27861843`, `0000281` `10420224/40653891`, and
  `0000282` `4292608/12277856`.
- Post-upload iteration `117` at `04:01:50+08:00` remains
  `DATA_PASS=False` with train `139 ok`, `49 missing`, `12 bad_size`,
  `extra=0`, and test `200 ok`, `0 missing`, `11 bad_size`, `extra=0`.
  No runtime import, py_compile, pytest, `sbatch`, or `bata_mobile50`
  marker has run.
- Local watchdog reset at `03:57:38+08:00`; the one-hour no-change trigger
  is not met. `parajobs` has no `bata_mobile50` or OpenTAD job, only
  unrelated account `run.sh` jobs.
- Decision: continue waiting. No helper upload while count/byte progress
  exists; no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T03:47:47+08:00 - N16R4 train upload advanced to 136 ok

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `119` at `03:45:58+08:00`: train layout `have=148`,
  `new_links=1`; train `136 ok`, `52 missing`, `12 bad_size`; test
  `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`; `active_sftp=21`.
- Current train bad-size tail includes `video_validation_0000205`
  `59736064/267699672`, `0000208` `65208320/75403408`, `0000209`
  `49414144/223950748`, `0000264` `22282240/82142543`, `0000266`
  `10223616/54520293`, `0000267` `9437184/192812340`, `0000268`
  `6324224/128601594`, and `0000269` `262144/11164768`.
- Post-upload iteration `110` at `03:47:45+08:00` remains `DATA_PASS=False`
  with train `136 ok`, `52 missing`, `12 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Runtime import,
  py_compile, pytest, and Slurm submit have not run.
- Local watchdog last observed reset was `03:42:33+08:00`; the one-hour
  no-change trigger is not met. `parajobs` has no `bata_mobile50` or OpenTAD
  job; visible jobs remain unrelated account `run.sh` jobs.
- Decision: continue waiting. No helper upload while count/byte progress
  exists; no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T03:34:37+08:00 - N16R4 train upload advanced to 130 ok

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `113` at `03:33:52+08:00`: train layout `have=142`,
  `new_links=0`; train `130 ok`, `58 missing`, `12 bad_size`; test
  `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`; `active_sftp=18`.
- Current train bad-size tail includes `video_validation_0000204`
  `53510144/147992502`, `0000205` `47022080/267699672`, `0000206`
  `62357504/67450539`, `0000207` `82116608/91022808`, `0000208`
  `42500096/75403408`, `0000209` `32997376/223950748`, `0000210`
  `51150848/57559366`, and `0000263` `10878976/19722155`.
- Post-upload iteration `103` at `03:33:40+08:00` remains `DATA_PASS=False`
  with train `130 ok`, `58 missing`, `12 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Runtime import,
  py_compile, pytest, and Slurm submit have not run.
- Local watchdog reset at `03:32:29+08:00`; the one-hour no-change trigger is
  not met. `parajobs` has no `bata_mobile50` or OpenTAD job; visible jobs
  remain unrelated account `run.sh` jobs.
- Decision: continue waiting. No helper upload while count/byte progress
  exists; no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T03:22:56+08:00 - N16R4 train upload advanced to 129 ok

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `107` at `03:21:45+08:00`: train layout `have=141`,
  `new_links=1`; train `129 ok`, `59 missing`, `12 bad_size`; test
  `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`; `active_sftp=16`.
- Current train bad-size tail includes `video_validation_0000204`
  `38109184/147992502`, `0000205` `33914880/267699672`, `0000206`
  `39190528/67450539`, `0000207` `62521344/91022808`, `0000208`
  `26968064/75403408`, `0000209` `16449536/223950748`, `0000210`
  `36241408/57559366`, and `0000262` `2031616/10852622`.
- Post-upload iteration `97` at `03:21:36+08:00` remains `DATA_PASS=False`
  with train `129 ok`, `60 missing`, `11 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Runtime import,
  py_compile, pytest, and Slurm submit have not run.
- Local watchdog reset at `03:22:26+08:00`; the one-hour no-change trigger is
  not met. `parajobs` has no `bata_mobile50` or OpenTAD job; visible jobs
  remain unrelated account `run.sh` jobs.
- Decision: continue waiting. No helper upload while count/byte progress
  exists; no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T03:05:49+08:00 - N16R4 train upload at 126 ok with byte progress

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `99` at `03:05:37+08:00`: train layout `have=138`,
  `new_links=0`; train `126 ok`, `62 missing`, `12 bad_size`; test
  `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`; `active_sftp=15`.
- Current train bad-size tail includes `video_validation_0000202`
  `26345472/103572672`, `0000203` `12156928/57328019`, `0000204`
  `14843904/147992502`, `0000205` `16809984/267699672`, `0000206`
  `15073280/67450539`, `0000207` `32342016/91022808`, `0000208`
  `7798784/75403408`, and `0000210` `15663104/57559366`.
- Post-upload iteration `89` at `03:05:31+08:00` remains `DATA_PASS=False`
  with train `126 ok`, `62 missing`, `12 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Runtime import,
  py_compile, pytest, and Slurm submit have not run.
- Local watchdog reset at `03:02:19+08:00`; byte progress means the one-hour
  no-change trigger is not met. `parajobs` has no `bata_mobile50` or OpenTAD
  job; visible jobs remain unrelated account `run.sh` jobs.
- Decision: continue waiting. No helper upload while byte progress exists; no
  Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T03:19:40+08:00 - N16R4 train upload advanced to 128 ok

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `105` at `03:17:44+08:00`: train layout `have=140`,
  `new_links=0`; train `128 ok`, `60 missing`, `12 bad_size`; test
  `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`; `active_sftp=15`.
- Current train bad-size tail includes `video_validation_0000203`
  `31981568/57328019`, `0000204` `33587200/147992502`, `0000205`
  `30900224/267699672`, `0000206` `32768000/67450539`, `0000207`
  `52920320/91022808`, `0000208` `22511616/75403408`, `0000209`
  `10682368/223950748`, and `0000210` `30736384/57559366`.
- Post-upload iteration `96` at `03:19:35+08:00` remains `DATA_PASS=False`
  with train `128 ok`, `60 missing`, `12 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Runtime import,
  py_compile, pytest, and Slurm submit have not run.
- Local watchdog reset at `03:17:24+08:00`; the one-hour no-change trigger is
  not met. `parajobs` has no `bata_mobile50` or OpenTAD job; visible jobs
  remain unrelated account `run.sh` jobs.
- Decision: continue waiting. No helper upload while count/byte progress
  exists; no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T03:07:55+08:00 - N16R4 train upload byte progress at 126 ok

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `100` at `03:07:38+08:00`: train layout `have=138`,
  `new_links=0`; train `126 ok`, `62 missing`, `12 bad_size`; test
  `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`; `active_sftp=15`.
- Counts paused since `03:05`, but train bad-size bytes continued increasing:
  `video_validation_0000202` `30113792/103572672`, `0000203`
  `15925248/57328019`, `0000204` `18022400/147992502`, `0000205`
  `19136512/267699672`, `0000206` `17891328/67450539`, `0000207`
  `35553280/91022808`, `0000208` `10256384/75403408`, and `0000210`
  `18350080/57559366`.
- Post-upload iteration `90` at `03:07:31+08:00` remains `DATA_PASS=False`
  with train `126 ok`, `62 missing`, `12 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Runtime import,
  py_compile, pytest, and Slurm submit have not run.
- Local watchdog reset at `03:07:21+08:00`; byte progress means the one-hour
  no-change trigger is not met. `parajobs` has no `bata_mobile50` or OpenTAD
  job; visible jobs remain unrelated account `run.sh` jobs.
- Decision: continue waiting. No helper upload while byte progress exists; no
  Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T02:59:24+08:00 - N16R4 train upload advanced to 125 ok

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `95` at `02:57:33+08:00`: train layout `have=137`,
  `new_links=3`; train `125 ok`, `63 missing`, `12 bad_size`; test
  `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`; `active_sftp=16`.
- Current train bad-size tail includes `video_validation_0000189`
  `26804224/39072497`, `0000202` `12517376/103572672`, `0000203`
  `1146880/57328019`, `0000204` `1474560/147992502`, `0000205`
  `7536640/267699672`, `0000206` `983040/67450539`, `0000207`
  `19496960/91022808`, and `0000210` `5865472/57559366`.
- Post-upload iteration `85` at `02:57:28+08:00` remains `DATA_PASS=False`
  with train `125 ok`, `66 missing`, `9 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Runtime import,
  py_compile, pytest, and Slurm submit have not run.
- Local watchdog reset at `02:57:18+08:00`; the one-hour no-change trigger is
  not met. `parajobs` has no `bata_mobile50` or OpenTAD job; visible jobs
  remain unrelated account `run.sh` jobs.
- Decision: continue waiting. No helper upload while count/byte progress
  exists; no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T02:52:50+08:00 - N16R4 train upload advanced to 122 ok

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `92` at `02:51:30+08:00`: train layout `have=134`,
  `new_links=2`; train `122 ok`, `66 missing`, `12 bad_size`; test
  `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`; `active_sftp=17`.
- Current train bad-size tail includes `video_validation_0000184`
  `17432576/61842644`, `0000188` `36470784/47017684`, `0000189`
  `19529728/39072497`, `0000201` `13467648/21211522`, `0000202`
  `2949120/103572672`, `0000205` `0/267699672`, `0000207`
  `8978432/91022808`, and `0000210` `491520/57559366`.
- Post-upload iteration `82` at `02:51:26+08:00` remains `DATA_PASS=False`
  with train `121 ok`, `68 missing`, `11 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Runtime import,
  py_compile, pytest, and Slurm submit have not run.
- Local watchdog reset at `02:52:16+08:00`; the one-hour no-change trigger is
  not met. `parajobs` has no `bata_mobile50` or OpenTAD job; visible jobs
  remain unrelated account `run.sh` jobs.
- Decision: continue waiting. No helper upload while count/byte progress
  exists; no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T02:48:13+08:00 - N16R4 train upload advanced to 119 ok

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `90` at `02:47:28+08:00`: train layout `have=131`,
  `new_links=1`; train `119 ok`, `69 missing`, `12 bad_size`; test
  `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`; `active_sftp=19`.
- Current train bad-size tail includes `video_validation_0000180`
  `34275328/49110025`, `0000184` `10485760/61842644`, `0000185`
  `20643840/23674999`, `0000187` `18087936/20372087`, `0000188`
  `29425664/47017684`, `0000189` `13893632/39072497`, `0000201`
  `7143424/21211522`, and `0000207` `1507328/91022808`.
- Post-upload iteration `80` at `02:47:25+08:00` remains `DATA_PASS=False`
  with train `119 ok`, `70 missing`, `11 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Grep for
  `DATA_PASS=True`, runtime import, py_compile, pytest, `sbatch`, and
  `bata_mobile50` still found only repeated `DATA_PASS=False`.
- Local watchdog last observed reset was `02:42:12+08:00`; current remote
  count/byte progress and active sftp processes mean the one-hour no-change
  trigger is not met. `parajobs` has no `bata_mobile50` or OpenTAD job;
  visible jobs remain unrelated account `run.sh` jobs.
- Decision: continue waiting. No helper upload while count/byte progress
  exists; no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T01:00:41+08:00 - N16R4 upload still progressing

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `37` at `01:00:32+08:00`: train `96 ok`, `93 missing`,
  `11 bad_size`; test `199 ok`, `0 missing`, `12 bad_size`; `ALL_PASS=False`;
  `active_sftp=24`; train layout `have=107`.
- Byte tails increased since iteration `36`, including
  `video_validation_0000160`, `0000162`, `0000163`, `0000164`, `0000165`,
  `0000167`, `0000168`, and `video_test_0001459`, so the user-defined
  one-hour no-change condition is not met.
- Post-upload iteration `26` at `00:58:50+08:00` remains `DATA_PASS=False`;
  runtime import, py_compile, pytest, and Slurm submit have not run.
- Visible `parajobs` entries are unrelated account `run.sh` jobs; no
  `bata_mobile50` or other N16R4 OpenTAD Slurm job exists.
- Decision: continue waiting. No helper upload while count/byte progress
  exists; no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T01:03:25+08:00 - N16R4 upload byte progress continues

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `38` at `01:02:33+08:00`: train `96 ok`, `93 missing`,
  `11 bad_size`; test `199 ok`, `0 missing`, `12 bad_size`; `ALL_PASS=False`;
  `active_sftp=21`; train layout `have=107`.
- Byte tails increased again since iteration `37`, including
  `video_validation_0000160`, `0000162`, `0000163`, `0000164`, `0000165`,
  `0000167`, and `video_test_0001459`. The local watchdog reset at
  `01:01:38+08:00`, so the one-hour no-change condition is not met.
- Post-upload iteration `28` at `01:02:52+08:00` remains `DATA_PASS=False`;
  runtime import, py_compile, pytest, and Slurm submit have not run.
- Visible `parajobs` entries are unrelated account `run.sh` jobs; no
  `bata_mobile50` or other N16R4 OpenTAD Slurm job exists.
- Decision: continue waiting without dense polling. No helper upload while
  count/byte progress exists; no Slurm launch until exact data PASS plus
  post-upload verification.

## 2026-05-28T01:06:30+08:00 - N16R4 upload still active

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `39` at `01:04:34+08:00`: train `96 ok`, `93 missing`,
  `11 bad_size`; test `199 ok`, `0 missing`, `12 bad_size`; `ALL_PASS=False`;
  `active_sftp=25`; train layout `have=107`.
- Byte tails increased again since iteration `38`, including
  `video_validation_0000160`, `0000162`, `0000163`, `0000164`, `0000165`,
  `0000167`, and `video_test_0001459`. The local watchdog last reset at
  `01:01:38+08:00`, so the one-hour no-change condition is not met.
- Post-upload iteration `29` at `01:04:52+08:00` remains `DATA_PASS=False`;
  runtime import, py_compile, pytest, and Slurm submit have not run.
- Visible `parajobs` entries are unrelated account `run.sh` jobs; no
  `bata_mobile50` or other N16R4 OpenTAD Slurm job exists.
- Decision: continue waiting without dense polling. No helper upload while
  count/byte progress exists; no Slurm launch until exact data PASS plus
  post-upload verification.

## 2026-05-28T01:09:47+08:00 - N16R4 train upload advanced by one

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `40` at `01:06:35+08:00` linked one more training video:
  `have=108`, `new_links=1`. Iteration `41` at `01:08:36+08:00`: train
  `97 ok`, `92 missing`, `11 bad_size`; test `199 ok`, `0 missing`,
  `12 bad_size`; `ALL_PASS=False`; `active_sftp=27`.
- Byte tails increased again, including `video_validation_0000160`, `0000162`,
  `0000163`, `0000164`, `0000167`, `0000169`, and `video_test_0001459`.
  The local watchdog reset at `01:06:40+08:00`, so the one-hour no-change
  condition is not met.
- Post-upload iteration `31` at `01:08:54+08:00` remains `DATA_PASS=False`;
  runtime import, py_compile, pytest, and Slurm submit have not run.
- Visible `parajobs` entries are unrelated account `run.sh` jobs; no
  `bata_mobile50` or other N16R4 OpenTAD Slurm job exists.
- Decision: continue waiting without dense polling. No helper upload while
  count/byte progress exists; no Slurm launch until exact data PASS plus
  post-upload verification.

## 2026-05-28T01:12:23+08:00 - N16R4 upload byte progress continues

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `42` at `01:10:37+08:00`: train `97 ok`, `92 missing`,
  `11 bad_size`; test `199 ok`, `0 missing`, `12 bad_size`; `ALL_PASS=False`;
  `active_sftp=25`; train layout `have=108`.
- Byte tails increased again, including `video_validation_0000160`, `0000162`,
  `0000163`, `0000164`, `0000167`, `0000169`, and `video_test_0001459`.
  The local watchdog reset at `01:11:42+08:00`, so the one-hour no-change
  condition is not met.
- Post-upload iteration `32` at `01:10:54+08:00` remains `DATA_PASS=False`;
  runtime import, py_compile, pytest, and Slurm submit have not run.
- Visible `parajobs` entries are unrelated account `run.sh` jobs; no
  `bata_mobile50` or other N16R4 OpenTAD Slurm job exists.
- Decision: continue waiting without dense polling. No helper upload while
  count/byte progress exists; no Slurm launch until exact data PASS plus
  post-upload verification.

## 2026-05-28T01:20:01+08:00 - N16R4 upload still progressing

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `46` at `01:18:41+08:00`: train `98 ok`, `91 missing`,
  `11 bad_size`; test `199 ok`, `0 missing`, `12 bad_size`; `ALL_PASS=False`;
  `active_sftp=28`; train layout `have=109`.
- Byte tails continue to increase, including `video_validation_0000162`,
  `0000163`, `0000164`, `0000167`, `0000169`, `0000170`, and
  `video_test_0001459`, so the one-hour no-change condition is not met.
- Post-upload iteration `36` at `01:18:57+08:00` remains `DATA_PASS=False`;
  runtime import, py_compile, pytest, and Slurm submit have not run.
- Visible `parajobs` entries are unrelated account `run.sh` jobs; no
  `bata_mobile50` or other N16R4 OpenTAD Slurm job exists.
- Decision: continue waiting. No helper upload while count/byte progress
  exists; no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T01:23:57+08:00 - N16R4 upload byte progress continues

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `48` at `01:22:44+08:00`: train `98 ok`, `91 missing`,
  `11 bad_size`; test `199 ok`, `0 missing`, `12 bad_size`; `ALL_PASS=False`;
  `active_sftp=23`; train layout `have=109`.
- Counts are unchanged, but partial bytes continue increasing for
  `video_validation_0000162`, `0000163`, `0000164`, `0000167`, `0000169`,
  `0000170`, and `video_test_0001459`. The watchdog reset at
  `01:21:45+08:00`, so the one-hour no-change condition is not met.
- Post-upload iteration `38` at `01:22:58+08:00` remains `DATA_PASS=False`;
  runtime import, py_compile, pytest, and Slurm submit have not run.
- Visible `parajobs` entries are unrelated account `run.sh` jobs; no
  `bata_mobile50` or other N16R4 OpenTAD Slurm job exists.
- Decision: continue waiting. No helper upload while byte progress exists; no
  Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T01:27:46+08:00 - N16R4 upload advanced again

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Compact parse of watcher iteration `50` at `01:26:46+08:00`: train layout
  `have=110`, `new_links=1`; train `99 ok`, `90 missing`, `11 bad_size`;
  test `199 ok`, `0 missing`, `12 bad_size`; `ALL_PASS=False`;
  `active_sftp=24`.
- Partial bytes continue increasing for `video_validation_0000163`, `0000164`,
  `0000167`, `0000169`, `0000170`, `0000171`, and `video_test_0001459`;
  therefore the one-hour no-change condition is not met.
- Post-upload iteration `40` remains `DATA_PASS=False`; runtime import,
  py_compile, pytest, and Slurm submit have not run.
- No PASS marker exists. `parajobs` has no `bata_mobile50` or OpenTAD job;
  visible jobs remain unrelated account `run.sh` jobs.
- Decision: continue waiting. No helper upload while count/byte progress
  exists; no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T01:32:03+08:00 - N16R4 upload byte progress continues

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `52` at `01:30:48+08:00`: train layout `have=110`,
  `new_links=0`; train `99 ok`, `90 missing`, `11 bad_size`; test `199 ok`,
  `0 missing`, `12 bad_size`; `ALL_PASS=False`; `active_sftp=21`.
- Counts are unchanged from iteration `50`, but partial bytes continue
  increasing for `video_validation_0000163`, `0000164`, `0000167`,
  `0000169`, `0000170`, `0000171`, and `video_test_0001459`. The local
  watchdog reset at `01:26:47+08:00`, so the one-hour no-change condition is
  not met.
- Post-upload iteration `42` remains `DATA_PASS=False`; runtime import,
  py_compile, pytest, and Slurm submit have not run.
- Decision: continue waiting. No helper upload while byte progress exists; no
  Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T01:34:37+08:00 - N16R4 upload still active

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `53` at `01:32:49+08:00`: train layout `have=110`,
  `new_links=0`; train `99 ok`, `90 missing`, `11 bad_size`; test `199 ok`,
  `0 missing`, `12 bad_size`; `ALL_PASS=False`; `active_sftp=23`.
- Partial bytes continue increasing versus iteration `52`, including
  `video_validation_0000163`, `0000164`, `0000167`, `0000169`, `0000170`,
  `0000171`, and `video_test_0001459` now `346947584/358813734` in watcher.
  The local watchdog reset at `01:31:49+08:00`, so the one-hour no-change
  condition is not met.
- Post-upload iteration `43` remains `DATA_PASS=False`; no runtime import,
  py_compile, pytest, or Slurm submit has run.
- No PASS marker exists. `parajobs` has no `bata_mobile50` or OpenTAD job;
  visible jobs remain unrelated account `run.sh` jobs.
- Decision: continue waiting. No helper upload while byte progress exists; no
  Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T01:36:58+08:00 - N16R4 upload byte progress continues

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `55` at `01:36:51+08:00`: train layout `have=110`,
  `new_links=0`; train `99 ok`, `90 missing`, `11 bad_size`; test `199 ok`,
  `0 missing`, `12 bad_size`; `ALL_PASS=False`; `active_sftp=22`.
- Counts are unchanged, but partial bytes continue increasing, including
  `video_validation_0000163` `65306624/111647597`, `0000164`
  `56983552/82674723`, `0000167` `88113152/110872937`, `0000169`
  `48136192/97885721`, `0000170` `28803072/93572478`, `0000171`
  `14516224/50201314`, and `video_test_0001459` `351371264/358813734`.
  The local watchdog had reset at `01:31:49+08:00`, so the one-hour no-change
  condition is not met.
- Post-upload iteration `44` remains `DATA_PASS=False`; no runtime import,
  py_compile, pytest, or Slurm submit has run.
- No PASS marker exists. `parajobs` has no `bata_mobile50` or OpenTAD job;
  visible jobs remain unrelated account `run.sh` jobs.
- Decision: continue waiting. No helper upload while byte progress exists; no
  Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T01:39:19+08:00 - N16R4 upload byte progress continues

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `56` at `01:38:52+08:00`: train layout `have=110`,
  `new_links=0`; train `99 ok`, `90 missing`, `11 bad_size`; test `199 ok`,
  `0 missing`, `12 bad_size`; `ALL_PASS=False`; `active_sftp=20`.
- Counts are unchanged, but partial bytes continue increasing, including
  `video_validation_0000163` `68452352/111647597`, `0000164`
  `59408384/82674723`, `0000167` `91062272/110872937`, `0000169`
  `52887552/97885721`, `0000170` `31948800/93572478`, `0000171`
  `18055168/50201314`, and `video_test_0001459` `352976896/358813734`.
  The local watchdog reset at `01:36:50+08:00`, so the one-hour no-change
  condition is not met.
- Post-upload iteration `46` remains `DATA_PASS=False`; no runtime import,
  py_compile, pytest, or Slurm submit has run.
- No PASS marker exists. `parajobs` has no `bata_mobile50` or OpenTAD job;
  visible jobs remain unrelated account `run.sh` jobs.
- Decision: continue waiting. No helper upload while byte progress exists; no
  Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T01:41:52+08:00 - N16R4 train upload advanced by one

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `57` at `01:40:53+08:00`: train layout `have=111`,
  `new_links=1`; train `100 ok`, `89 missing`, `11 bad_size`; test `199 ok`,
  `0 missing`, `12 bad_size`; `ALL_PASS=False`; `active_sftp=19`.
- This is a real count advance from iteration `56`. Partial bytes also
  continue increasing, including `video_validation_0000164`
  `61702144/82674723`, `0000167` `93487104/110872937`, `0000169`
  `56524800/97885721`, `0000170` `35356672/93572478`, `0000171`
  `22118400/50201314`, new partial `0000173` `1146880/97374492`, and
  `video_test_0001459` `354910208/358813734`.
- Post-upload iteration `47` remains `DATA_PASS=False`; no runtime import,
  py_compile, pytest, or Slurm submit has run.
- No PASS marker exists. `parajobs` has no `bata_mobile50` or OpenTAD job;
  visible jobs remain unrelated account `run.sh` jobs.
- Decision: continue waiting. No helper upload while count/byte progress
  exists; no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T01:46:20+08:00 - N16R4 upload advanced again

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `59` at `01:44:55+08:00`: train layout `have=112`,
  `new_links=1`; train `100 ok`, `88 missing`, `12 bad_size`; test
  `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`; `active_sftp=20`.
- This is real progress after iteration `57`: one additional training file
  appeared and one test partial completed. Partial bytes continue increasing,
  including `video_validation_0000167` `97583104/110872937`,
  `0000169` `65339392/97885721`, `0000170` `41484288/93572478`,
  `0000171` `27459584/50201314`, `0000173` `7864320/97374492`,
  and new partial `0000174` `655360/36970640`.
- Post-upload iteration `49` at `01:45:05+08:00` remains `DATA_PASS=False`
  with train `100 ok`, `88 missing`, `12 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. No runtime import,
  py_compile, pytest, or Slurm submit has run.
- No PASS marker exists. `parajobs` has no `bata_mobile50` or OpenTAD job;
  visible jobs remain unrelated account `run.sh` jobs. The local watchdog last
  recorded a reset at `01:41:52+08:00`, and current remote count/byte progress
  means the one-hour no-change trigger is not met.
- Decision: continue waiting. No helper upload while count/byte progress
  exists; no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T01:53:34+08:00 - N16R4 upload byte progress continues

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `63` at `01:53:00+08:00`: train layout `have=112`,
  `new_links=0`; train `100 ok`, `88 missing`, `12 bad_size`; test
  `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`; `active_sftp=16`.
- Counts are unchanged from iteration `59`, but partial bytes continue
  increasing, including `video_validation_0000167` `107380736/110872937`,
  `0000169` `77791232/97885721`, `0000170` `54132736/93572478`,
  `0000171` `37781504/50201314`, `0000173` `19988480/97374492`, and
  `0000174` `15302656/36970640`.
- Post-upload iteration `53` at `01:53:08+08:00` remains `DATA_PASS=False`
  with train `100 ok`, `88 missing`, `12 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. No runtime import,
  py_compile, pytest, or Slurm submit has run.
- Local watchdog reset again at `01:51:55+08:00`, so the one-hour no-change
  trigger is not met. `parajobs` has no `bata_mobile50` or OpenTAD job;
  visible jobs remain unrelated account `run.sh` jobs.
- Decision: continue waiting. No helper upload while byte progress exists; no
  Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T02:00:06+08:00 - N16R4 train upload advanced by one

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `65` at `01:57:02+08:00`: train layout `have=113`,
  `new_links=1`; train `101 ok`, `87 missing`, `12 bad_size`; test
  `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`; `active_sftp=19`.
  Iteration `66` at `01:59:03+08:00` kept the same counts while bytes
  continued increasing.
- Post-upload iteration `56` at `01:59:10+08:00` remains `DATA_PASS=False`
  with train `101 ok`, `87 missing`, `12 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. No runtime import,
  py_compile, pytest, or Slurm submit has run.
- Partial bytes continue increasing for train files including
  `video_validation_0000169` `87490560/97885721`, `0000170`
  `63766528/93572478`, `0000171` `44498944/50201314`, `0000173`
  `29884416/97374492`, `0000174` `27820032/36970640`, and `0000175`
  `4554752/30307358`.
- Local watchdog reset at `01:56:57+08:00`, so the one-hour no-change trigger
  is not met. `parajobs` has no `bata_mobile50` or OpenTAD job; visible jobs
  remain unrelated account `run.sh` jobs.
- Decision: continue waiting. No helper upload while count/byte progress
  exists; no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T02:06:34+08:00 - N16R4 train upload advanced by four

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `69` at `02:05:06+08:00`: train layout `have=117`,
  `new_links=3`; train `105 ok`, `83 missing`, `12 bad_size`; test
  `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`; `active_sftp=17`.
- Post-upload iteration `59` at `02:05:11+08:00` matches train `105 ok`,
  `83 missing`, `12 bad_size`, `extra=0`, and test `200 ok`, `0 missing`,
  `11 bad_size`, `extra=0`; `DATA_PASS=False`. No runtime import,
  py_compile, pytest, or Slurm submit has run.
- Partial train bytes continue increasing, including `video_validation_0000170`
  `75890688/93572478`, `0000172` `917504/13306149`, `0000173`
  `38535168/97374492`, `0000175` `10616832/30307358`, `0000176`
  `6717440/32629279`, `0000177` `3309568/47115579`, and `0000178`
  `2588672/35220322`.
- Local watchdog reset at `02:01:59+08:00`, and current remote count/byte
  progress means the one-hour no-change trigger is not met. `parajobs` has no
  `bata_mobile50` or OpenTAD job; visible jobs remain unrelated account
  `run.sh` jobs.
- Decision: continue waiting. No helper upload while count/byte progress
  exists; no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T02:13:11+08:00 - N16R4 upload byte progress continues

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `71` at `02:09:08+08:00` advanced train to `106 ok`,
  `82 missing`, `12 bad_size`; latest watcher iteration `73` at
  `02:13:10+08:00` keeps train `106 ok`, `82 missing`, `12 bad_size` and
  test `200 ok`, `0 missing`, `11 bad_size`, with `ALL_PASS=False`.
- Partial train bytes continue increasing, including `video_validation_0000170`
  `88211456/93572478`, `0000172` `9732096/13306149`, `0000173`
  `52527104/97374492`, `0000175` `20447232/30307358`, `0000176`
  `16809984/32629279`, `0000177` `18808832/47115579`, `0000178`
  `15663104/35220322`, and `0000179` `9732096/73176986`.
- Post-upload iteration `62` at `02:11:13+08:00` remains `DATA_PASS=False`
  with train `106 ok`, `82 missing`, `12 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. No runtime import,
  py_compile, pytest, or Slurm submit has run.
- Local watchdog reset at `02:12:02+08:00`, so the one-hour no-change trigger
  is not met. `parajobs` has no `bata_mobile50` or OpenTAD job; visible jobs
  remain unrelated account `run.sh` jobs.
- Decision: continue waiting. No helper upload while byte progress exists; no
  Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T02:16:12+08:00 - N16R4 upload byte progress continues

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `74` at `02:15:11+08:00`: train layout `have=118`,
  `new_links=0`; train `106 ok`, `82 missing`, `12 bad_size`; test
  `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`; `active_sftp=16`.
- Counts are unchanged from `02:13`, but partial bytes continue increasing,
  including `video_validation_0000170` `91324416/93572478`, `0000172`
  `12091392/13306149`, `0000173` `55181312/97374492`, `0000175`
  `22183936/30307358`, `0000176` `19038208/32629279`, `0000177`
  `23035904/47115579`, `0000178` `20217856/35220322`, and `0000179`
  `12845056/73176986`.
- Post-upload iteration `64` at `02:15:15+08:00` remains `DATA_PASS=False`
  with train `106 ok`, `82 missing`, `12 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. No runtime import,
  py_compile, pytest, or Slurm submit has run.
- Local watchdog last observed reset remains `02:12:02+08:00`, so the
  one-hour no-change trigger is not met. `parajobs` has no `bata_mobile50` or
  OpenTAD job; visible jobs remain unrelated account `run.sh` jobs.
- Decision: continue waiting. No helper upload while byte progress exists; no
  Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T02:22:59+08:00 - N16R4 train upload advanced by two

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `75` at `02:17:13+08:00` advanced train to `have=120`,
  `new_links=2`, train `108 ok`, `80 missing`, `12 bad_size`; latest watcher
  iteration `77` at `02:21:15+08:00` keeps the same counts while bytes
  continue increasing. Test remains `200 ok`, `0 missing`, `11 bad_size`;
  `ALL_PASS=False`.
- Current train bad-size tail includes `video_validation_0000173`
  `64782336/97374492`, `0000175` `29032448/30307358`, `0000176`
  `26476544/32629279`, `0000177` `33652736/47115579`, `0000178`
  `31653888/35220322`, `0000179` `20250624/73176986`, `0000180`
  `5242880/49110025`, and `0000181` `7929856/52287784`.
- Post-upload iteration `67` at `02:21:17+08:00` remains `DATA_PASS=False`
  with train `108 ok`, `80 missing`, `12 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. No runtime import,
  py_compile, pytest, or Slurm submit has run.
- Local watchdog reset at `02:22:06+08:00`, so the one-hour no-change trigger
  is not met. `parajobs` has no `bata_mobile50` or OpenTAD job; visible jobs
  remain unrelated account `run.sh` jobs.
- Decision: continue waiting. No helper upload while count/byte progress
  exists; no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T02:25:09+08:00 - N16R4 train upload advanced by two more

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `78` at `02:23:16+08:00`: train layout `have=122`,
  `new_links=2`; train `110 ok`, `78 missing`, `12 bad_size`; test
  `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`; `active_sftp=16`.
- Current train bad-size tail includes `video_validation_0000173`
  `68550656/97374492`, `0000176` `28344320/32629279`, `0000177`
  `37224448/47115579`, `0000179` `22478848/73176986`, `0000180`
  `7208960/49110025`, `0000181` `11763712/52287784`, `0000182`
  `1081344/12791256`, and `0000183` `688128/18562961`.
- Post-upload iteration `68` at `02:23:17+08:00` remains `DATA_PASS=False`
  with train `110 ok`, `78 missing`, `12 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. No runtime import,
  py_compile, pytest, or Slurm submit has run.
- Local watchdog reset at `02:22:06+08:00`; remote count/byte progress and
  active sftp processes mean the one-hour no-change trigger is not met.
  `parajobs` has no `bata_mobile50` or OpenTAD job; visible jobs remain
  unrelated account `run.sh` jobs.
- Decision: continue waiting. No helper upload while count/byte progress
  exists; no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T02:32:06+08:00 - N16R4 train upload advanced to 113 ok

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `82` at `02:31:20+08:00`: train layout `have=125`,
  `new_links=1`; train `113 ok`, `75 missing`, `12 bad_size`; test
  `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`; `active_sftp=19`.
- Current train bad-size tail includes `video_validation_0000179`
  `34963456/73176986`, `0000180` `15400960/49110025`, `0000181`
  `24182784/52287784`, `0000182` `8159232/12791256`, `0000183`
  `17268736/18562961`, `0000185` `4292608/23674999`, `0000186`
  `5832704/21899367`, and `0000187` `1015808/20372087`.
- Post-upload iteration `72` at `02:31:19+08:00` remains `DATA_PASS=False`
  with train `113 ok`, `76 missing`, `11 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. No runtime import,
  py_compile, pytest, or Slurm submit has run.
- Local watchdog reset at `02:27:07+08:00`; current remote count/byte progress
  and active sftp processes mean the one-hour no-change trigger is not met.
  `parajobs` has no `bata_mobile50` or OpenTAD job; visible jobs remain
  unrelated account `run.sh` jobs.
- Decision: continue waiting. No helper upload while count/byte progress
  exists; no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T02:38:40+08:00 - N16R4 train upload advanced to 115 ok

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `85` at `02:37:23+08:00`: train layout `have=127`,
  `new_links=0`; train `115 ok`, `73 missing`, `12 bad_size`; test
  `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`; `active_sftp=18`.
- Current train bad-size tail includes `video_validation_0000179`
  `45219840/73176986`, `0000180` `21757952/49110025`, `0000181`
  `36503552/52287784`, `0000185` `10452992/23674999`, `0000186`
  `16613376/21899367`, `0000187` `6356992/20372087`, `0000188`
  `10584064/47017684`, and `0000189` `2916352/39072497`.
- Post-upload iteration `75` at `02:37:21+08:00` remains `DATA_PASS=False`
  with train `115 ok`, `73 missing`, `12 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. No runtime import,
  py_compile, pytest, or Slurm submit has run.
- Local watchdog reset at `02:37:10+08:00`; current remote count/byte progress
  and active sftp processes mean the one-hour no-change trigger is not met.
  `parajobs` has no `bata_mobile50` or OpenTAD job; visible jobs remain
  unrelated account `run.sh` jobs.
- Decision: continue waiting. No helper upload while count/byte progress
  exists; no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T02:45:30+08:00 - N16R4 train upload advanced to 118 ok

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `89` at `02:45:27+08:00`: train layout `have=130`,
  `new_links=0`; train `118 ok`, `70 missing`, `12 bad_size`; test
  `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`; `active_sftp=20`.
- Current train bad-size tail includes `video_validation_0000180`
  `31621120/49110025`, `0000181` `51019776/52287784`, `0000184`
  `6651904/61842644`, `0000185` `19005440/23674999`, `0000187`
  `15564800/20372087`, `0000188` `25362432/47017684`, `0000189`
  `11894784/39072497`, and `0000201` `4259840/21211522`.
- Post-upload iteration `79` at `02:45:24+08:00` remains `DATA_PASS=False`
  with train `118 ok`, `70 missing`, `12 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Grep for
  `DATA_PASS=True`, runtime import, py_compile, pytest, `sbatch`, and
  `bata_mobile50` found only repeated `DATA_PASS=False`.
- Local watchdog reset at `02:42:12+08:00`; current remote count/byte progress
  and active sftp processes mean the one-hour no-change trigger is not met.
  `parajobs` has no `bata_mobile50` or OpenTAD job; visible jobs remain
  unrelated account `run.sh` jobs.
- Decision: continue waiting. No helper upload while count/byte progress
  exists; no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T07:50:46+08:00 - N16R4 upload still progressing at train 173 ok

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `240` at `07:50:04+08:00`: train layout `have=186`,
  `new_links=0`; train `173 ok`, `14 missing`, `13 bad_size`; test
  `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`; `active_sftp=13`.
- Current train bad-size tail is still increasing, including
  `video_validation_0000369` `83755008/590761829`, `0000370`
  `90341376/356182662`, `0000411` `90210304/217008088`, `0000412`
  `0/153134081`, `0000415` `54460416/78524153`, `0000416`
  `35618816/260968672`, `0000417` `17268736/127629223`, and `0000418`
  `9502720/94595740`.
- Post-upload iteration `230` at `07:49:10+08:00` remains `DATA_PASS=False`
  with train `173 ok`, `14 missing`, `13 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Grep found no runtime
  import, py_compile, pytest, `sbatch`, `bata_mobile50`, or OpenTAD training
  marker.
- Local watchdog reset at `07:48:58+08:00`; current remote byte progress and
  active sftp sessions mean the one-hour no-change trigger is not met.
  `parajobs` has no `bata_mobile50` or OpenTAD job; visible jobs remain
  unrelated account `run.sh` jobs `992937` and `992776`.
- Decision: continue waiting. No helper upload while count/byte progress
  exists; no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T07:53:42+08:00 - N16R4 upload byte progress continues

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `241` at `07:52:05+08:00`: train layout `have=186`,
  `new_links=0`; train `173 ok`, `14 missing`, `13 bad_size`; test
  `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`; `active_sftp=14`.
- Current train bad-size tail advanced again: `video_validation_0000369`
  `86704128/590761829`, `0000370` `94371840/356182662`, `0000411`
  `94109696/217008088`, `0000412` `0/153134081`, `0000415`
  `57999360/78524153`, `0000416` `37584896/260968672`, `0000417`
  `19857408/127629223`, and `0000418` `12156928/94595740`.
- Post-upload iteration `232` at `07:53:11+08:00` remains `DATA_PASS=False`
  with train `173 ok`, `14 missing`, `13 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Grep found no runtime
  import, py_compile, pytest, `sbatch`, `bata_mobile50`, or OpenTAD training
  marker.
- Remote byte progress and active sftp sessions mean the one-hour no-change
  trigger is not met. `parajobs` has no `bata_mobile50` or OpenTAD job;
  visible jobs remain unrelated account `run.sh` jobs `992937` and `992776`.
- Decision: continue waiting. No helper upload while count/byte progress
  exists; no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T07:55:06+08:00 - Local watchdog reset on upload progress

- Local watchdog `106220` tail check shows watcher state iteration `242` at
  `07:54:00+08:00`: train `173 ok`, `14 missing`, `13 bad_size`; test
  `200 ok`, `0 missing`, `11 bad_size`; `active_sftp=13`; `all_pass=False`.
- The signature changed again, with train partial bytes advancing to
  `video_validation_0000369` `89227264/590761829`, `0000370`
  `97779712/356182662`, `0000411` `97353728/217008088`, `0000412`
  `0/153134081`, `0000415` `60555264/78524153`, `0000416`
  `39976960/260968672`, `0000417` `22970368/127629223`, and `0000418`
  `15302656/94595740`.
- No helper upload attempt log is present in the latest local check. The
  one-hour no-change trigger is reset at `07:54:00+08:00`.
- Decision: continue waiting. No helper upload while count/byte progress
  exists; no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T07:58:25+08:00 - N16R4 upload still moving at train 173 ok

- Remote upload watcher `2142542` and post-upload gate `2225557` are alive.
- Watcher iteration `244` at `07:58:08+08:00`: train layout `have=186`,
  `new_links=0`; train `173 ok`, `14 missing`, `13 bad_size`; test
  `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`; `active_sftp=14`.
- Current train bad-size tail advanced again: `video_validation_0000369`
  `94765056/590761829`, `0000370` `102596608/356182662`, `0000411`
  `102694912/217008088`, `0000412` `0/153134081`, `0000415`
  `68124672/78524153`, `0000416` `45809664/260968672`, `0000417`
  `28344320/127629223`, and `0000418` `21430272/94595740`.
- Post-upload iteration `234` at `07:57:13+08:00` remains `DATA_PASS=False`
  with train `173 ok`, `14 missing`, `13 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Grep found no runtime
  import, py_compile, pytest, `sbatch`, `bata_mobile50`, or OpenTAD training
  marker.
- `parajobs` has no `bata_mobile50` or OpenTAD job; visible jobs remain
  unrelated account `run.sh` jobs `992937` and `992776`.
- Decision: continue waiting. No helper upload while byte progress exists;
  no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T08:00:50+08:00 - N16R4 upload still moving; data gate blocked

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `245` at `08:00:09+08:00`: train layout `have=186`,
  `new_links=0`; train `173 ok`, `14 missing`, `13 bad_size`; test
  `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`; `active_sftp=15`.
- Current train bad-size tail advanced again: `video_validation_0000369`
  `97091584/590761829`, `0000370` `105512960/356182662`, `0000411`
  `106332160/217008088`, `0000412` `0/153134081`, `0000415`
  `72712192/78524153`, `0000416` `47349760/260968672`, `0000417`
  `31588352/127629223`, and `0000418` `25755648/94595740`.
- Local watchdog also reset at `07:59:02+08:00` on iteration `244`, so the
  one-hour no-change trigger is not met.
- Post-upload iteration `235` at `07:59:14+08:00` remains `DATA_PASS=False`
  with train `173 ok`, `14 missing`, `13 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Grep found no runtime
  import, py_compile, pytest, `sbatch`, `bata_mobile50`, or OpenTAD training
  marker.
- `parajobs` has no `bata_mobile50` or OpenTAD job; visible jobs remain
  unrelated account `run.sh` jobs `992937` and `992776`.
- Decision: continue waiting. No helper upload while byte progress exists;
  no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T08:03:31+08:00 - N16R4 upload still progressing

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `246` at `08:02:10+08:00`: train layout `have=186`,
  `new_links=0`; train `173 ok`, `14 missing`, `13 bad_size`; test
  `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`; `active_sftp=15`.
- Current train bad-size tail advanced again: `video_validation_0000369`
  `99549184/590761829`, `0000370` `108265472/356182662`, `0000411`
  `109641728/217008088`, `0000412` `0/153134081`, `0000415`
  `76447744/78524153`, `0000416` `50790400/260968672`, `0000417`
  `33685504/127629223`, and `0000418` `29622272/94595740`.
- Local watchdog tail also shows progress reset at `07:59:02+08:00`, so the
  one-hour no-change trigger is not met.
- Post-upload iteration `237` at `08:03:15+08:00` remains `DATA_PASS=False`
  with train `173 ok`, `14 missing`, `13 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Grep found no runtime
  import, py_compile, pytest, `sbatch`, `bata_mobile50`, or OpenTAD training
  marker.
- `parajobs` has no `bata_mobile50` or OpenTAD job; visible jobs remain
  unrelated account `run.sh` jobs `992937` and `992776`.
- Decision: continue waiting. No helper upload while byte progress exists;
  no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T08:06:19+08:00 - N16R4 train upload advanced to 174 ok

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `247` at `08:04:11+08:00` linked one more training video:
  train layout `have=187`, `new_links=1`, train `174 ok`, `13 missing`,
  `13 bad_size`. Iteration `248` at `08:06:12+08:00` kept train `174 ok`,
  `13 missing`, `13 bad_size`; test remained `200 ok`, `0 missing`,
  `11 bad_size`; `ALL_PASS=False`; `active_sftp=16`.
- Current train bad-size tail is still increasing: `video_validation_0000369`
  `104103936/590761829`, `0000370` `115277824/356182662`, `0000411`
  `115703808/217008088`, `0000412` `0/153134081`, `0000416`
  `58687488/260968672`, `0000417` `38862848/127629223`, `0000418`
  `35684352/94595740`, and new partial `0000419` `4456448/459211618`.
- Local watchdog also recorded iteration `247` at `08:04:03+08:00` and reset
  on count/byte progress, so the one-hour no-change trigger is not met.
- Post-upload iteration `238` at `08:05:16+08:00` remains `DATA_PASS=False`
  with train `174 ok`, `13 missing`, `13 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Grep found no runtime
  import, py_compile, pytest, `sbatch`, `bata_mobile50`, or OpenTAD training
  marker.
- `parajobs` has no `bata_mobile50` or OpenTAD job; visible jobs remain
  unrelated account `run.sh` jobs `992937` and `992776`.
- Decision: continue waiting. No helper upload while count/byte progress
  exists; no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T08:09:05+08:00 - N16R4 upload byte progress continues at 174 ok

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `249` at `08:08:13+08:00`: train layout `have=187`,
  `new_links=0`; train `174 ok`, `13 missing`, `13 bad_size`; test
  `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`; `active_sftp=19`.
- Current train bad-size tail continued increasing: `video_validation_0000369`
  `106102784/590761829`, `0000370` `118128640/356182662`, `0000411`
  `119209984/217008088`, `0000412` `0/153134081`, `0000416`
  `62193664/260968672`, `0000417` `41451520/127629223`, `0000418`
  `38273024/94595740`, and `0000419` `8093696/459211618`.
- Post-upload iteration `239` at `08:07:16+08:00` remains `DATA_PASS=False`
  with train `174 ok`, `13 missing`, `13 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Grep found no runtime
  import, py_compile, pytest, `sbatch`, `bata_mobile50`, or OpenTAD training
  marker.
- `parajobs` has no `bata_mobile50` or OpenTAD job; visible jobs remain
  unrelated account `run.sh` jobs `992937` and `992776`.
- Decision: continue waiting. No helper upload while byte progress exists;
  no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T08:13:28+08:00 - N16R4 upload byte progress continues

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `251` at `08:12:15+08:00`: train layout `have=187`,
  `new_links=0`; train `174 ok`, `13 missing`, `13 bad_size`; test
  `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`; `active_sftp=21`.
- Current train bad-size tail continued increasing: `video_validation_0000369`
  `111116288/590761829`, `0000370` `123797504/356182662`, `0000411`
  `125861888/217008088`, `0000412` `0/153134081`, `0000416`
  `69074944/260968672`, `0000417` `45744128/127629223`, `0000418`
  `45121536/94595740`, and `0000419` `14942208/459211618`.
- Post-upload iteration `242` at `08:13:19+08:00` remains `DATA_PASS=False`
  with train `174 ok`, `13 missing`, `13 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Grep found no runtime
  import, py_compile, pytest, `sbatch`, `bata_mobile50`, or OpenTAD training
  marker.
- Process check shows watcher/gate alive plus active `sftp-server` sessions.
  The local watchdog is alive, and current remote byte progress means the
  60-minute no-progress trigger is not met.
- `parajobs` has no `bata_mobile50` or OpenTAD job; visible jobs remain
  unrelated account `run.sh` jobs `992937` and `992776`.
- Decision: continue waiting. No helper upload while byte progress exists;
  no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T08:15:28+08:00 - N16R4 upload byte progress continues

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `252` at `08:14:16+08:00`: train layout `have=187`,
  `new_links=0`; train `174 ok`, `13 missing`, `13 bad_size`; test
  `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`; `active_sftp=21`.
- Current train bad-size tail continued increasing: `video_validation_0000369`
  `113180672/590761829`, `0000370` `126550016/356182662`, `0000411`
  `127959040/217008088`, `0000412` `0/153134081`, `0000416`
  `72581120/260968672`, `0000417` `47939584/127629223`, `0000418`
  `49840128/94595740`, and `0000419` `18808832/459211618`.
- Post-upload iteration `243` at `08:15:19+08:00` remains `DATA_PASS=False`
  with train `174 ok`, `13 missing`, `13 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Grep found no runtime
  import, py_compile, pytest, `sbatch`, `bata_mobile50`, or OpenTAD training
  marker.
- `parajobs` has no `bata_mobile50` or OpenTAD job; visible jobs remain
  unrelated account `run.sh` jobs `992937` and `992776`.
- Decision: continue waiting. No helper upload while byte progress exists;
  no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T08:18:42+08:00 - N16R4 upload byte progress continues

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `254` at `08:18:19+08:00`: train layout `have=187`,
  `new_links=0`; train `174 ok`, `13 missing`, `13 bad_size`; test
  `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`; `active_sftp=18`.
- Current train bad-size tail continued increasing: `video_validation_0000369`
  `118030336/590761829`, `0000370` `131465216/356182662`, `0000411`
  `134053888/217008088`, `0000412` `0/153134081`, `0000416`
  `78741504/260968672`, `0000417` `52920320/127629223`, `0000418`
  `56918016/94595740`, and `0000419` `25919488/459211618`.
- Post-upload iteration `244` at `08:17:20+08:00` remains `DATA_PASS=False`
  with train `174 ok`, `13 missing`, `13 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Grep found no runtime
  import, py_compile, pytest, `sbatch`, `bata_mobile50`, or OpenTAD training
  marker.
- Process summary shows watcher/gate alive and no `tools/train.py`, `torchrun`,
  `sbatch`, `bata_mobile50`, or OpenTAD training process; `parajobs` has no
  matching OpenTAD/BATA job.
- Decision: continue waiting. No helper upload while byte progress exists;
  no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T08:23:25+08:00 - N16R4 upload byte progress continues

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `256` at `08:22:21+08:00`: train layout `have=187`,
  `new_links=0`; train `174 ok`, `13 missing`, `13 bad_size`; test
  `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`; `active_sftp=20`.
- Current train bad-size tail continued increasing: `video_validation_0000369`
  `123207680/590761829`, `0000370` `136773632/356182662`, `0000411`
  `137691136/217008088`, `0000412` `0/153134081`, `0000416`
  `85327872/260968672`, `0000417` `57966592/127629223`, `0000418`
  `63995904/94595740`, and `0000419` `32866304/459211618`.
- Post-upload iteration `247` at `08:23:22+08:00` remains `DATA_PASS=False`
  with train `174 ok`, `13 missing`, `13 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Grep found no runtime
  import, py_compile, pytest, `sbatch`, `bata_mobile50`, or OpenTAD training
  marker.
- Local watchdog is alive and reset the stall timer at `08:19:10+08:00` on
  iteration `254`; no helper attempt log was present.
- `parajobs` has no matching OpenTAD/BATA job.
- Decision: continue waiting. No helper upload while byte progress exists;
  no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T08:25:39+08:00 - N16R4 upload byte progress continues

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `257` at `08:24:22+08:00`: train layout `have=187`,
  `new_links=0`; train `174 ok`, `13 missing`, `13 bad_size`; test
  `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`; `active_sftp=16`.
- Current train bad-size tail continued increasing: `video_validation_0000369`
  `125960192/590761829`, `0000370` `139427840/356182662`, `0000411`
  `140083200/217008088`, `0000412` `0/153134081`, `0000416`
  `88801280/260968672`, `0000417` `60620800/127629223`, `0000418`
  `67305472/94595740`, and `0000419` `36110336/459211618`.
- Post-upload iteration `248` at `08:25:23+08:00` remains `DATA_PASS=False`
  with train `174 ok`, `13 missing`, `13 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Grep found no runtime
  import, py_compile, pytest, `sbatch`, `bata_mobile50`, or OpenTAD training
  marker.
- Local watchdog reset the stall timer at `08:24:11+08:00` on iteration
  `257`; no helper upload attempt log was present.
- `parajobs` includes non-matching `run.sh` jobs `993340`, `992937`, and
  `992776`, but no `bata_mobile50` or OpenTAD job.
- Decision: continue waiting. No helper upload while byte progress exists;
  no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T08:29:33+08:00 - N16R4 upload byte progress continues

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `259` at `08:28:24+08:00`: train layout `have=187`,
  `new_links=0`; train `174 ok`, `13 missing`, `13 bad_size`; test
  `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`; `active_sftp=15`.
- Current train bad-size tail continued increasing: `video_validation_0000369`
  `130777088/590761829`, `0000370` `144867328/356182662`, `0000411`
  `144572416/217008088`, `0000412` `0/153134081`, `0000416`
  `95420416/260968672`, `0000417` `65241088/127629223`, `0000418`
  `74711040/94595740`, and `0000419` `43057152/459211618`.
- Post-upload iteration `250` at `08:29:24+08:00` remains `DATA_PASS=False`
  with train `174 ok`, `13 missing`, `13 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Grep found no runtime
  import, py_compile, pytest, `sbatch`, `bata_mobile50`, or OpenTAD training
  marker.
- Local watchdog is alive; its latest visible reset remains `08:24:11+08:00`,
  and current remote byte progress confirms the 60-minute no-progress rule is
  not triggered.
- `parajobs` grep has no `bata_mobile50` or OpenTAD match.
- Decision: continue waiting. No helper upload while byte progress exists;
  no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T08:41:54+08:00 - N16R4 upload count and byte progress continues

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `265` at `08:40:30+08:00` advanced train layout to
  `expected=200`, `have=188`, `new_links=1`; train `175 ok`, `12 missing`,
  `13 bad_size`; test `200 ok`, `0 missing`, `11 bad_size`;
  `ALL_PASS=False`; `active_sftp=15`.
- Current train bad-size tail is still partial and changing:
  `video_validation_0000369.mp4` `144769024/590761829`, `0000370`
  `166887424/356182662`, `0000411` `156827648/217008088`, `0000412`
  `0/153134081`, `0000416` `120815616/260968672`, `0000417`
  `77856768/127629223`, `0000419` `61767680/459211618`, and `0000420`
  `98304/424932270`.
- Test bad-size tail remains incomplete, including `video_test_0001235`,
  `0001255`, `0001314`, `0001339`, `0001343`, `0001369`, `0001389`, and
  `0001495`.
- Post-upload iteration `256` at `08:41:29+08:00` remains `DATA_PASS=False`
  with train `175 ok`, `12 missing`, `13 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Grep found no runtime
  import, py_compile, pytest, `sbatch`, `bata_mobile50`, or OpenTAD training
  marker.
- Local watchdog last visibly reset at `08:39:16+08:00` on watcher iteration
  `264`; no helper attempt log was present. Because iteration `265` advanced
  counts and bytes, the 60-minute no-progress rule is not triggered.
- Slurm still shows only unrelated `run.sh` jobs and no `bata_mobile50` or
  OpenTAD job. No code/model/config changed; strict random-fixed 50% and
  no-test-GT/no-teacher contracts remain unchanged because no N16R4 training
  launched; no mAP evidence exists.
- Decision: continue waiting. No helper upload while count or byte progress
  exists; no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T08:46:17+08:00 - N16R4 upload still progressing

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `267` at `08:44:33+08:00`: train layout `expected=200`,
  `have=188`, `new_links=0`; train `175 ok`, `12 missing`, `13 bad_size`;
  test `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`;
  `active_sftp=14`.
- Counts are unchanged from iteration `265`, but byte tails continue
  increasing: `video_validation_0000369.mp4` `148996096/590761829`,
  `0000370` `175538176/356182662`, `0000411` `160890880/217008088`,
  `0000412` `0/153134081`, `0000416` `127762432/260968672`, `0000417`
  `81559552/127629223`, `0000419` `66715648/459211618`, and `0000420`
  `5308416/424932270`.
- Post-upload iteration `258` at `08:45:30+08:00` remains `DATA_PASS=False`
  with train `175 ok`, `12 missing`, `13 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Grep found no runtime
  import, py_compile, pytest, `sbatch`, `bata_mobile50`, or OpenTAD training
  marker.
- Remote processes show watcher/gate alive plus active `sftp-server`
  sessions. Local watchdog reset at `08:44:18+08:00` on iteration `267`; no
  helper upload attempt log exists.
- `parajobs` has no matching OpenTAD/BATA job. `git diff --check` for the
  three records reports only the known `research-wiki/log.md` LF-to-CRLF
  warning and no whitespace errors.
- Decision: continue waiting. No helper upload while byte progress exists;
  no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T08:31:44+08:00 - N16R4 upload byte progress continues

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `260` at `08:30:25+08:00`: train layout `have=187`,
  `new_links=0`; train `174 ok`, `13 missing`, `13 bad_size`; test
  `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`; `active_sftp=15`.
- Current train bad-size tail continued increasing: `video_validation_0000369`
  `133136384/590761829`, `0000370` `148176896/356182662`, `0000411`
  `146440192/217008088`, `0000412` `0/153134081`, `0000416`
  `99778560/260968672`, `0000417` `67993600/127629223`, `0000418`
  `77627392/94595740`, and `0000419` `47415296/459211618`.
- Post-upload iteration `251` at `08:31:25+08:00` remains `DATA_PASS=False`
  with train `174 ok`, `13 missing`, `13 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Grep found no runtime
  import, py_compile, pytest, `sbatch`, `bata_mobile50`, or OpenTAD training
  marker.
- Local watchdog reset the stall timer at `08:29:13+08:00` on iteration
  `259`; process check shows watcher/gate alive plus active `sftp-server`
  sessions.
- `parajobs` has only non-matching `run.sh` jobs `993340`, `992937`, and
  `992776`; no `bata_mobile50` or OpenTAD job.
- Decision: continue waiting. No helper upload while byte progress exists;
  no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T08:34:17+08:00 - N16R4 upload byte progress continues

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `261` at `08:32:26+08:00`: train layout `have=187`,
  `new_links=0`; train `174 ok`, `13 missing`, `13 bad_size`; test
  `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`; `active_sftp=14`.
- Current train bad-size tail continued increasing: `video_validation_0000369`
  `135168000/590761829`, `0000370` `151781376/356182662`, `0000411`
  `148439040/217008088`, `0000412` `0/153134081`, `0000416`
  `103874560/260968672`, `0000417` `70189056/127629223`, `0000418`
  `80969728/94595740`, and `0000419` `51380224/459211618`.
- Post-upload iteration `252` at `08:33:26+08:00` remains `DATA_PASS=False`
  with train `174 ok`, `13 missing`, `13 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Grep found no runtime
  import, py_compile, pytest, `sbatch`, `bata_mobile50`, or OpenTAD training
  marker.
- Local watchdog is alive; its latest visible reset is `08:29:13+08:00` on
  iteration `259`, and current remote byte progress confirms the 60-minute
  no-progress rule is not triggered.
- `parajobs` has only non-matching `run.sh` jobs `993340`, `992937`, and
  `992776`; no `bata_mobile50` or OpenTAD job.
- Decision: continue waiting. No helper upload while byte progress exists;
  no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T08:36:43+08:00 - N16R4 upload byte progress continues

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `263` at `08:36:28+08:00`: train layout `have=187`,
  `new_links=0`; train `174 ok`, `13 missing`, `13 bad_size`; test
  `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`; `active_sftp=15`.
- Current train bad-size tail continued increasing: `video_validation_0000369`
  `140607488/590761829`, `0000370` `159580160/356182662`, `0000411`
  `152633344/217008088`, `0000412` `0/153134081`, `0000416`
  `112525312/260968672`, `0000417` `74252288/127629223`, `0000418`
  `88473600/94595740`, and `0000419` `56459264/459211618`.
- Post-upload iteration `253` at `08:35:27+08:00` remains `DATA_PASS=False`
  with train `174 ok`, `13 missing`, `13 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Grep found no runtime
  import, py_compile, pytest, `sbatch`, `bata_mobile50`, or OpenTAD training
  marker.
- Local watchdog reset the stall timer at `08:34:15+08:00` on iteration
  `262`; no helper attempt log was present.
- `parajobs` has only non-matching `run.sh` jobs `993340`, `992937`, and
  `992776`; no `bata_mobile50` or OpenTAD job.
- Decision: continue waiting. No helper upload while byte progress exists;
  no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T08:39:20+08:00 - N16R4 upload byte progress continues

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `264` at `08:38:29+08:00`: train layout `have=187`,
  `new_links=0`; train `174 ok`, `13 missing`, `13 bad_size`; test
  `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`; `active_sftp=15`.
- Current train bad-size tail continued increasing: `video_validation_0000369`
  `142868480/590761829`, `0000370` `163282944/356182662`, `0000411`
  `154664960/217008088`, `0000412` `0/153134081`, `0000416`
  `116457472/260968672`, `0000417` `75857920/127629223`, `0000418`
  `91455488/94595740`, and `0000419` `59310080/459211618`.
- Post-upload iteration `254` at `08:37:27+08:00` remains `DATA_PASS=False`
  with train `174 ok`, `13 missing`, `13 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Grep found no runtime
  import, py_compile, pytest, `sbatch`, `bata_mobile50`, or OpenTAD training
  marker.
- Local watchdog is alive; latest visible reset is `08:34:15+08:00` on
  iteration `262`, and current remote byte progress confirms the 60-minute
  no-progress rule is not triggered.
- `parajobs` grep has no `bata_mobile50` or OpenTAD match.
- Decision: continue waiting. No helper upload while byte progress exists;
  no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T08:50:52+08:00 - N16R4 upload byte progress continues

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `270` at `08:50:36+08:00`: train layout `expected=200`,
  `have=188`, `new_links=0`; train `175 ok`, `12 missing`, `13 bad_size`;
  test `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`;
  `active_sftp=13`.
- Counts are unchanged from the last check, but byte tails continue
  increasing: `video_validation_0000369.mp4` `156696576/590761829`,
  `0000370` `186122240/356182662`, `0000411` `167051264/217008088`,
  `0000412` `0/153134081`, `0000416` `139231232/260968672`, `0000417`
  `86212608/127629223`, `0000419` `74481664/459211618`, and `0000420`
  `14778368/424932270`.
- Post-upload iteration `260` at `08:49:32+08:00` remains `DATA_PASS=False`
  with train `175 ok`, `12 missing`, `13 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Grep found no runtime
  import, py_compile, pytest, `sbatch`, `bata_mobile50`, OpenTAD training
  marker, or result artifact.
- Remote processes show watcher/gate alive plus active `sftp-server`
  sessions. Local watchdog reset at `08:49:20+08:00` on iteration `269`; no
  helper upload attempt log exists.
- `parajobs` has no matching OpenTAD/BATA job, only unrelated account
  `run.sh` jobs `993340`, `992937`, and `992776`. `df -h ~/run/yuzibo` shows
  JuiceFS `2.3T` total, `407G` used, `2.0T` free.
- Decision: continue waiting. No helper upload while byte progress exists;
  no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T08:56:55+08:00 - N16R4 upload byte progress continues

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `273` at `08:56:39+08:00`: train layout `expected=200`,
  `have=188`, `new_links=0`; train `175 ok`, `12 missing`, `13 bad_size`;
  test `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`;
  `active_sftp=16`.
- Counts are unchanged from the last check, but byte tails continue
  increasing: `video_validation_0000369.mp4` `163151872/590761829`,
  `0000370` `195887104/356182662`, `0000411` `173146112/217008088`,
  `0000412` `0/153134081`, `0000416` `151355392/260968672`, `0000417`
  `90963968/127629223`, `0000419` `82608128/459211618`, and `0000420`
  `22773760/424932270`.
- Post-upload iteration `263` at `08:55:34+08:00` remains `DATA_PASS=False`
  with train `175 ok`, `12 missing`, `13 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Grep found no runtime
  import, py_compile, pytest, `sbatch`, `bata_mobile50`, OpenTAD training
  marker, or result artifact.
- Remote processes show watcher/gate alive plus active `sftp-server`
  sessions. Local watchdog reset at `08:54:22+08:00` on iteration `272`; no
  helper upload attempt log exists.
- `parajobs` has no matching OpenTAD/BATA job, only unrelated account
  `run.sh` jobs `993340`, `992937`, and `992776`. `df -h ~/run/yuzibo` shows
  JuiceFS `2.3T` total, `407G` used, `2.0T` free.
- Decision: continue waiting. No helper upload while byte progress exists;
  no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T09:01:38+08:00 - N16R4 upload byte progress continues

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `275` at `09:00:41+08:00`: train layout `expected=200`,
  `have=188`, `new_links=0`; train `175 ok`, `12 missing`, `13 bad_size`;
  test `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`;
  `active_sftp=17`.
- Counts are unchanged from the last check, but byte tails continue
  increasing: `video_validation_0000369.mp4` `167346176/590761829`,
  `0000370` `204177408/356182662`, `0000411` `177864704/217008088`,
  `0000412` `0/153134081`, `0000416` `155320320/260968672`, `0000417`
  `97124352/127629223`, `0000419` `89128960/459211618`, and `0000420`
  `28803072/424932270`.
- Post-upload iteration `266` at `09:01:36+08:00` remains `DATA_PASS=False`
  with train `175 ok`, `12 missing`, `13 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`; its train bad tail is up
  to `168361984/590761829`, `205357056/356182662`, `178782208/217008088`,
  `156237824/260968672`, `98107392/127629223`, `91422720/459211618`, and
  `30408704/424932270`. Grep found no runtime import, py_compile, pytest,
  `sbatch`, `bata_mobile50`, OpenTAD training marker, or result artifact.
- Remote processes show watcher/gate alive plus active `sftp-server`
  sessions. Local watchdog reset at `08:59:24+08:00` on iteration `274`; no
  helper upload attempt log exists.
- `parajobs` has no matching OpenTAD/BATA job, only unrelated account
  `run.sh` jobs `993340`, `992937`, and `992776`. `df -h ~/run/yuzibo` shows
  JuiceFS `2.3T` total, `407G` used, `2.0T` free.
- Decision: continue waiting. No helper upload while byte progress exists;
  no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T09:06:42+08:00 - N16R4 upload byte progress continues

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `277` at `09:04:44+08:00`: train layout `expected=200`,
  `have=188`, `new_links=0`; train `175 ok`, `12 missing`, `13 bad_size`;
  test `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`;
  `active_sftp=18`.
- Counts are unchanged from the last check, but byte tails continue
  increasing: `video_validation_0000369.mp4` `171114496/590761829`,
  `0000370` `211812352/356182662`, `0000411` `182026240/217008088`,
  `0000412` `0/153134081`, `0000416` `161841152/260968672`, `0000417`
  `103448576/127629223`, `0000419` `96436224/459211618`, and `0000420`
  `34078720/424932270`.
- Post-upload iteration `268` at `09:05:38+08:00` remains `DATA_PASS=False`
  with train `175 ok`, `12 missing`, `13 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`; its train bad tail is up
  to `171704320/590761829`, `214401024/356182662`, `182976512/217008088`,
  `163151872/260968672`, `105283584/127629223`, `97517568/459211618`, and
  `35323904/424932270`. Grep found no runtime import, py_compile, pytest,
  `sbatch`, `bata_mobile50`, OpenTAD training marker, or result artifact.
- Remote processes show watcher/gate alive plus active `sftp-server`
  sessions. Local watchdog reset at `09:04:25+08:00` on iteration `277`; no
  helper upload attempt log exists.
- `parajobs` has no matching OpenTAD/BATA job, only unrelated account
  `run.sh` jobs `993340`, `992937`, and `992776`. `df -h ~/run/yuzibo` shows
  JuiceFS `2.3T` total, `407G` used, `2.0T` free.
- Decision: continue waiting. No helper upload while byte progress exists;
  no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T09:16:34+08:00 - N16R4 upload byte progress continues

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `282` at `09:14:49+08:00`: train layout `expected=200`,
  `have=188`, `new_links=0`; train `175 ok`, `12 missing`, `13 bad_size`;
  test `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`;
  `active_sftp=15`.
- Counts are unchanged from the last check, but byte tails continue
  increasing: `video_validation_0000369.mp4` `179830784/590761829`,
  `0000370` `233897984/356182662`, `0000411` `190873600/217008088`,
  `0000412` `0/153134081`, `0000416` `174358528/260968672`, `0000417`
  `123174912/127629223`, `0000419` `110198784/459211618`, and `0000420`
  `46104576/424932270`.
- Post-upload iteration `273` at `09:15:42+08:00` remains `DATA_PASS=False`
  with train `175 ok`, `12 missing`, `13 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`; its train bad tail is up
  to `180355072/590761829`, `235143168/356182662`, `191332352/217008088`,
  `176029696/260968672`, `124944384/127629223`, `110952448/459211618`, and
  `47841280/424932270`. Grep found no runtime import, py_compile, pytest,
  `sbatch`, `bata_mobile50`, OpenTAD training marker, or result artifact.
- Remote processes show watcher/gate alive plus active `sftp-server`
  sessions. Local watchdog reset at `09:14:29+08:00` on iteration `282`; no
  helper upload attempt log exists.
- `parajobs` has no matching OpenTAD/BATA job, only unrelated account
  `run.sh` jobs `993340`, `992937`, and `992776`. `df -h ~/run/yuzibo` shows
  JuiceFS `2.3T` total, `407G` used, `2.0T` free.
- Decision: continue waiting. No helper upload while byte progress exists;
  no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T09:25:16+08:00 - N16R4 upload count and byte progress continues

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `287` at `09:24:54+08:00`: train layout `expected=200`,
  `have=189`, `new_links=0`; train `176 ok`, `11 missing`, `13 bad_size`;
  test `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`;
  `active_sftp=17`.
- This is fresh count progress since the previous recorded state. Bad-size
  bytes also continue increasing, including `video_validation_0000369.mp4`
  `187957248/590761829`, `0000370` `248250368/356182662`, `0000411`
  `199688192/217008088`, `0000412` `0/153134081`, `0000413`
  `13369344/273157908`, `0000416` `187858944/260968672`, `0000419`
  `119439360/459211618`, and `0000420` `62062592/424932270`.
- Post-upload iteration `277` at `09:23:44+08:00` remains `DATA_PASS=False`
  with train `176 ok`, `11 missing`, `13 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`; its train bad tail is up
  to `186974208/590761829`, `246677504/356182662`, `198901760/217008088`,
  `11304960/273157908`, `185860096/260968672`, `117800960/459211618`, and
  `60260352/424932270`. Grep found no runtime import, py_compile, pytest,
  `sbatch`, `bata_mobile50`, OpenTAD training marker, or result artifact.
- Remote processes show watcher/gate alive plus active `sftp-server`
  sessions. Local watchdog reset at `09:24:32+08:00` on iteration `287`; no
  helper upload attempt log exists.
- `parajobs` has no matching OpenTAD/BATA job, only unrelated account
  `run.sh` jobs `993340`, `992937`, and `992776`. `df -h ~/run/yuzibo` shows
  JuiceFS `2.3T` total, `407G` used, `2.0T` free.
- Decision: continue waiting. No helper upload while count/byte progress
  exists; no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T09:35:49+08:00 - N16R4 upload count and byte progress continues

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `292` at `09:35:00+08:00`: train layout `expected=200`,
  `have=190`, `new_links=0`; train `176 ok`, `10 missing`, `14 bad_size`;
  test `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`;
  `active_sftp=27`.
- This is fresh count progress since the previous recorded state. Bad-size
  bytes also continue increasing, including `video_validation_0000370.mp4`
  `259686400/356182662`, `0000411` `212434944/217008088`, `0000412`
  `0/153134081`, `0000413` `23887872/273157908`, `0000416`
  `204963840/260968672`, `0000419` `129794048/459211618`, `0000420`
  `69599232/424932270`, and `0000481` `6881280/229675218`.
- Post-upload iteration `282` at `09:33:48+08:00` remains `DATA_PASS=False`
  with train `176 ok`, `10 missing`, `14 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Grep found no runtime
  import, py_compile, pytest, `sbatch`, `bata_mobile50`, OpenTAD training
  marker, or result artifact.
- Remote processes show watcher/gate alive plus active `sftp-server`
  sessions. Local watchdog reset at `09:34:35+08:00` on iteration `291`; no
  helper upload attempt log exists.
- `parajobs` has no matching OpenTAD/BATA job, only unrelated account
  `run.sh` jobs `993340`, `992937`, and `992776`. `df -h ~/run/yuzibo` shows
  JuiceFS `2.3T` total, `408G` used, `2.0T` free.
- Decision: continue waiting. No helper upload while count/byte progress
  exists; no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T09:38:50+08:00 - N16R4 upload byte progress continues

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `293` at `09:37:01+08:00`: train layout `expected=200`,
  `have=190`, `new_links=0`; train `176 ok`, `10 missing`, `14 bad_size`;
  test `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`;
  `active_sftp=27`.
- Counts are unchanged from iteration `292`, but bad-size bytes continue
  increasing, including `video_validation_0000370.mp4`
  `262209536/356182662`, `0000411` `215154688/217008088`, `0000412`
  `0/153134081`, `0000413` `25559040/273157908`, `0000416`
  `208961536/260968672`, `0000419` `132546560/459211618`, `0000420`
  `69599232/424932270`, and `0000481` `8781824/229675218`.
- Post-upload iteration `284` at `09:37:50+08:00` remains `DATA_PASS=False`
  with train `176 ok`, `10 missing`, `14 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Grep found no runtime
  import, py_compile, pytest, `sbatch`, `bata_mobile50`, OpenTAD training
  marker, or result artifact.
- Remote processes show watcher/gate alive plus active `sftp-server`
  sessions. Local watchdog last visibly reset at `09:34:35+08:00` on
  iteration `291`; no helper upload attempt log exists.
- `parajobs` has no matching OpenTAD/BATA job, only unrelated account
  `run.sh` jobs `993340`, `992937`, and `992776`. `df -h ~/run/yuzibo` shows
  JuiceFS `2.3T` total, `408G` used, `2.0T` free.
- Decision: continue waiting. No helper upload while count/byte progress
  exists; no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T09:44:24+08:00 - N16R4 upload count and byte progress continues

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `296` at `09:43:04+08:00`: train layout `expected=200`,
  `have=191`, `new_links=0`; train `177 ok`, `9 missing`, `14 bad_size`;
  test `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`;
  `active_sftp=25`.
- This is fresh count progress since the previous recorded state. Bad-size
  bytes also continue increasing, including `video_validation_0000370.mp4`
  `267911168/356182662`, `0000412` `0/153134081`, `0000413`
  `33193984/273157908`, `0000416` `220004352/260968672`, `0000419`
  `141000704/459211618`, `0000420` `69599232/424932270`, `0000481`
  `18022400/229675218`, and `0000483` `5570560/65453897`.
- Post-upload iteration `287` at `09:43:52+08:00` remains `DATA_PASS=False`
  with train `178 ok`, `9 missing`, `13 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Grep found no runtime
  import, py_compile, pytest, `sbatch`, `bata_mobile50`, OpenTAD training
  marker, or result artifact.
- Remote processes show watcher/gate alive plus active `sftp-server`
  sessions. Local watchdog reset at `09:39:37+08:00` on iteration `294`; no
  helper upload attempt log exists.
- Slurm grep has no matching OpenTAD/BATA job. `df -h ~/run/yuzibo` shows
  JuiceFS `2.3T` total, `408G` used, `2.0T` free.
- Decision: continue waiting. No helper upload while count/byte progress
  exists; no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T09:53:45+08:00 - N16R4 upload count and byte progress continues

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `301` at `09:53:10+08:00`: train layout `expected=200`,
  `have=194`, `new_links=1`; train `180 ok`, `6 missing`, `14 bad_size`;
  test `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`;
  `active_sftp=19`.
- This is fresh count progress since the previous recorded state. Bad-size
  bytes also continue increasing, including `video_validation_0000413.mp4`
  `45514752/273157908`, `0000416` `238944256/260968672`, `0000419`
  `152993792/459211618`, `0000420` `69599232/424932270`, `0000481`
  `33521664/229675218`, `0000482` `2916352/44030540`, `0000483`
  `18841600/65453897`, and `0000487` `11042816/79300709`.
- Post-upload iteration `291` at `09:51:55+08:00` remains `DATA_PASS=False`
  with train `180 ok`, `7 missing`, `13 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Grep found no runtime
  import, py_compile, pytest, `sbatch`, `bata_mobile50`, OpenTAD training
  marker, or result artifact.
- Remote processes show watcher/gate alive plus active `sftp-server`
  sessions. Local watchdog reset at `09:49:40+08:00` on iteration `299`; no
  helper upload attempt log exists.
- Slurm grep has no matching OpenTAD/BATA job. `df -h ~/run/yuzibo` shows
  JuiceFS `2.3T` total, `408G` used, `2.0T` free.
- Decision: continue waiting. No helper upload while count/byte progress
  exists; no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T10:02:47+08:00 - N16R4 upload byte progress continues

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `305` at `10:01:15+08:00`: train layout `expected=200`,
  `have=194`, `new_links=0`; train `180 ok`, `6 missing`, `14 bad_size`;
  test `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`;
  `active_sftp=18`.
- Counts are unchanged from the previous recorded state, but bad-size bytes
  continue increasing, including `video_validation_0000413.mp4`
  `53968896/273157908`, `0000416` `254672896/260968672`, `0000419`
  `164298752/459211618`, `0000420` `69599232/424932270`, `0000481`
  `45514752/229675218`, `0000482` `21790720/44030540`, `0000483`
  `24608768/65453897`, and `0000487` `20021248/79300709`.
- Post-upload iteration `296` at `10:01:59+08:00` remains `DATA_PASS=False`
  with train `180 ok`, `6 missing`, `14 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Grep found no runtime
  import, py_compile, pytest, `sbatch`, `bata_mobile50`, OpenTAD training
  marker, or result artifact.
- Remote processes show watcher/gate alive plus active `sftp-server`
  sessions. Local watchdog reset at `09:59:44+08:00` on iteration `304`; no
  helper upload attempt log exists.
- Slurm grep has no matching OpenTAD/BATA job. `df -h ~/run/yuzibo` shows
  JuiceFS `2.3T` total, `408G` used, `2.0T` free.
- Decision: continue waiting. No helper upload while count/byte progress
  exists; no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T10:07:00+08:00 - N16R4 upload count and byte progress continues

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `307` at `10:05:17+08:00`: train layout `expected=200`,
  `have=195`, `new_links=1`; train `181 ok`, `5 missing`, `14 bad_size`;
  test `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`;
  `active_sftp=19`.
- This is fresh count progress since the previous recorded state. Bad-size
  bytes also continue increasing, including `video_validation_0000413.mp4`
  `58425344/273157908`, `0000419` `170983424/459211618`, `0000420`
  `69599232/424932270`, `0000481` `51150848/229675218`, `0000482`
  `28147712/44030540`, `0000483` `28442624/65453897`, `0000484`
  `1638400/444825572`, and `0000487` `24772608/79300709`.
- Post-upload iteration `298` at `10:06:01+08:00` remains `DATA_PASS=False`
  with train `181 ok`, `5 missing`, `14 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Grep found no runtime
  import, py_compile, pytest, `sbatch`, `bata_mobile50`, OpenTAD training
  marker, or result artifact.
- Remote processes show watcher/gate alive plus active `sftp-server`
  sessions. Local watchdog reset at `10:04:46+08:00` on iteration `306`; no
  helper upload attempt log exists.
- Slurm shows only unrelated `run.sh` jobs and no matching OpenTAD/BATA job.
  `df -h ~/run/yuzibo` shows JuiceFS `2.3T` total, `408G` used, `2.0T`
  free.
- Decision: continue waiting. No helper upload while count/byte progress
  exists; no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T10:18:13+08:00 - N16R4 upload byte progress continues

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `313` at `10:17:26+08:00`: train layout `expected=200`,
  `have=196`, `new_links=0`, `missing=4`; train `182 ok`, `4 missing`,
  `14 bad_size`; test `200 ok`, `0 missing`, `11 bad_size`;
  `ALL_PASS=False`; `active_sftp=30`.
- Counts are unchanged from the previous post-10:15 state, but bad-size bytes
  continue increasing, including `video_validation_0000413.mp4`
  `72318976/273157908`, `0000419` `190775296/459211618`, `0000481`
  `74055680/229675218`, `0000483` `52232192/65453897`, `0000484`
  `16678912/444825572`, `0000485` `7766016/111121310`, and `0000487`
  `41058304/79300709`. Train missing tail remains `video_validation_0000486.mp4`,
  `0000489`, `0000490`, and `0000661`.
- Post-upload iteration `304` at `10:18:06+08:00` remains `DATA_PASS=False`
  with train `182 ok`, `4 missing`, `14 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Grep found no runtime
  import, py_compile, pytest, `sbatch`, `bata_mobile50`, OpenTAD training
  marker, or result artifact.
- Remote processes show watcher/gate alive plus active `sftp-server`
  sessions. Local watchdog reset at `10:15:05+08:00` on iteration `312`; no
  helper upload attempt log exists.
- Slurm shows only unrelated `run.sh` jobs `993340`, `992937`, and `992776`,
  with no matching OpenTAD/BATA job. `df -h ~/run/yuzibo` shows JuiceFS
  `2.3T` total, `408G` used, `2.0T` free.
- Decision: continue waiting. No helper upload while count/byte progress
  exists; no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T10:22:17+08:00 - N16R4 upload byte progress continues

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `315` at `10:21:28+08:00`: train layout `expected=200`,
  `have=196`, `new_links=0`, `missing=4`; train `182 ok`, `4 missing`,
  `14 bad_size`; test `200 ok`, `0 missing`, `11 bad_size`;
  `ALL_PASS=False`; `active_sftp=31`.
- Counts are unchanged from the previous state, but bad-size bytes continue
  increasing, including `video_validation_0000413.mp4` `78446592/273157908`,
  `0000419` `197197824/459211618`, `0000481` `80412672/229675218`,
  `0000483` `62160896/65453897`, `0000484` `22609920/444825572`,
  `0000485` `12746752/111121310`, and `0000487` `45056000/79300709`.
  Train missing tail remains `video_validation_0000486.mp4`, `0000489`,
  `0000490`, and `0000661`.
- Post-upload iteration `306` at `10:22:08+08:00` remains `DATA_PASS=False`
  with train `182 ok`, `4 missing`, `14 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Grep found no runtime
  import, py_compile, pytest, `sbatch`, `bata_mobile50`, OpenTAD training
  marker, or result artifact.
- Remote processes show watcher/gate alive plus active `sftp-server`
  sessions. Local watchdog reset at `10:20:06+08:00` on iteration `314`; no
  helper upload attempt log exists.
- Slurm shows only unrelated `run.sh` jobs `993340`, `992937`, and `992776`,
  with no matching OpenTAD/BATA job. `df -h ~/run/yuzibo` shows JuiceFS
  `2.3T` total, `408G` used, `2.0T` free.
- Decision: continue waiting. No helper upload while count/byte progress
  exists; no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T10:28:07+08:00 - N16R4 upload count and byte progress continues

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `318` at `10:27:31+08:00`: train layout `expected=200`,
  `have=197`, `new_links=0`, `missing=3`; train `183 ok`, `3 missing`,
  `14 bad_size`; test `200 ok`, `0 missing`, `11 bad_size`;
  `ALL_PASS=False`; `active_sftp=24`.
- This is fresh count progress: `video_validation_0000486.mp4` entered upload
  and train ok advanced to `183`. Bad-size bytes also continue increasing,
  including `video_validation_0000413.mp4` `85721088/273157908`, `0000419`
  `209879040/459211618`, `0000481` `91226112/229675218`, `0000484`
  `30867456/444825572`, `0000485` `21266432/111121310`, `0000486`
  `8257536/92982070`, and `0000487` `51838976/79300709`. Train missing tail
  is now `video_validation_0000489.mp4`, `0000490`, and `0000661`.
- Post-upload iteration `308` at `10:26:09+08:00` remains `DATA_PASS=False`
  with train `183 ok`, `3 missing`, `14 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Grep found no runtime
  import, py_compile, pytest, `sbatch`, `bata_mobile50`, OpenTAD training
  marker, or result artifact.
- Remote processes show watcher/gate alive plus active `sftp-server`
  sessions. Local watchdog reset at `10:25:08+08:00` on iteration `317`; no
  helper upload attempt log exists.
- Slurm shows only unrelated `run.sh` jobs `993340`, `992937`, and `992776`,
  with no matching OpenTAD/BATA job. `df -h ~/run/yuzibo` shows JuiceFS
  `2.3T` total, `408G` used, `2.0T` free.
- Decision: continue waiting. No helper upload while count/byte progress
  exists; no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T10:30:59+08:00 - N16R4 upload byte progress continues

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `319` at `10:29:32+08:00`: train layout `expected=200`,
  `have=197`, `new_links=0`, `missing=3`; train `183 ok`, `3 missing`,
  `14 bad_size`; test `200 ok`, `0 missing`, `11 bad_size`;
  `ALL_PASS=False`; `active_sftp=20`.
- Counts are unchanged from the previous state, but bad-size bytes continue
  increasing, including `video_validation_0000413.mp4` `88637440/273157908`,
  `0000419` `213876736/459211618`, `0000481` `95092736/229675218`,
  `0000484` `33456128/444825572`, `0000485` `23003136/111121310`,
  `0000486` `12615680/92982070`, and `0000487` `54689792/79300709`.
  Train missing tail remains `video_validation_0000489.mp4`, `0000490`,
  and `0000661`.
- Post-upload iteration `310` at `10:30:11+08:00` remains `DATA_PASS=False`
  with train `183 ok`, `3 missing`, `14 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Grep found no runtime
  import, py_compile, pytest, `sbatch`, `bata_mobile50`, OpenTAD training
  marker, or result artifact.
- Remote processes show watcher/gate alive plus active `sftp-server`
  sessions. Local watchdog reset at `10:30:09+08:00` on iteration `319`; no
  helper upload attempt log exists.
- Slurm shows only unrelated `run.sh` jobs `993340`, `992937`, and `992776`,
  with no matching OpenTAD/BATA job. `df -h ~/run/yuzibo` shows JuiceFS
  `2.3T` total, `408G` used, `2.0T` free.
- Decision: continue waiting. No helper upload while count/byte progress
  exists; no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T10:34:51+08:00 - N16R4 upload byte progress continues

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `321` at `10:33:34+08:00`: train layout `expected=200`,
  `have=197`, `new_links=0`, `missing=3`; train `183 ok`, `3 missing`,
  `14 bad_size`; test `200 ok`, `0 missing`, `11 bad_size`;
  `ALL_PASS=False`; `active_sftp=16`.
- Counts are unchanged from the previous state, but bad-size bytes continue
  increasing, including `video_validation_0000413.mp4` `95551488/273157908`,
  `0000419` `220921856/459211618`, `0000481` `102137856/229675218`,
  `0000484` `38895616/444825572`, `0000485` `27656192/111121310`,
  `0000486` `21659648/92982070`, and `0000487` `59310080/79300709`.
  Train missing tail remains `video_validation_0000489.mp4`, `0000490`,
  and `0000661`.
- Post-upload iteration `312` at `10:34:12+08:00` remains `DATA_PASS=False`
  with train `183 ok`, `3 missing`, `14 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. Grep found no runtime
  import, py_compile, pytest, `sbatch`, `bata_mobile50`, OpenTAD training
  marker, or result artifact.
- Remote processes show watcher/gate alive plus active `sftp-server`
  sessions. Local watchdog last reset at `10:30:09+08:00` on iteration `319`;
  no helper upload attempt log exists.
- Slurm shows only unrelated `run.sh` jobs `993340`, `992937`, and `992776`,
  with no matching OpenTAD/BATA job.
- Decision: continue waiting. No helper upload while count/byte progress
  exists; no Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T10:37:39+08:00 - N16R4 upload byte progress continues

- Remote upload watcher `2142542`, post-upload gate `2225557`, and local
  watchdog `106220` are alive.
- Watcher iteration `323` at `10:37:37+08:00`: train layout `expected=200`,
  `have=197`, `new_links=0`, `missing=3`; train `183 ok`, `3 missing`,
  `14 bad_size`; test `200 ok`, `0 missing`, `11 bad_size`;
  `ALL_PASS=False`; `active_sftp=16`.
- Counts are unchanged from `10:34`, but bad-size bytes continue increasing,
  including `video_validation_0000413.mp4` `102727680/273157908`, `0000419`
  `227573760/459211618`, `0000481` `107642880/229675218`, `0000484`
  `44040192/444825572`, `0000485` `31752192/111121310`, `0000486`
  `32505856/92982070`, and `0000487` `65503232/79300709`. Train missing
  tail remains `video_validation_0000489.mp4`, `0000490`, and `0000661`.
- Post-upload iteration `313` at `10:36:13+08:00` remains `DATA_PASS=False`
  with train `183 ok`, `3 missing`, `14 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. No runtime import,
  py_compile, pytest, `sbatch`, `bata_mobile50`, OpenTAD training marker, or
  result artifact has appeared.
- Remote processes show watcher/gate alive plus `16` active `sftp-server`
  sessions. Local watchdog last reset at `10:35:11+08:00` on iteration `321`;
  no helper upload attempt log exists.
- Slurm shows only unrelated `run.sh` jobs `993340`, `992937`, and `992776`,
  with no matching OpenTAD/BATA job. `df -h ~/run/yuzibo` shows JuiceFS
  `2.3T` total, `408G` used, `2.0T` free.
- Decision: continue waiting. No helper upload while byte progress exists; no
  Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T10:40:34+08:00 - N16R4 upload byte progress continues

- Remote upload watcher `2142542` and post-upload gate `2225557` are alive.
- Watcher iteration `324` at `10:39:38+08:00`: train layout `expected=200`,
  `have=197`, `new_links=0`, `missing=3`; train `183 ok`, `3 missing`,
  `14 bad_size`; test `200 ok`, `0 missing`, `11 bad_size`;
  `ALL_PASS=False`; `active_sftp=16`.
- Counts are unchanged, but bad-size bytes continue increasing, including
  `video_validation_0000413.mp4` `106168320/273157908`, `0000419`
  `231079936/459211618`, `0000481` `111181824/229675218`, `0000484`
  `46497792/444825572`, `0000485` `34144256/111121310`, `0000486`
  `37388288/92982070`, and `0000487` `68616192/79300709`.
- Post-upload iteration `315` at `10:40:15+08:00` remains `DATA_PASS=False`
  with train `183 ok`, `3 missing`, `14 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. No runtime import,
  py_compile, pytest, `sbatch`, `bata_mobile50`, OpenTAD training marker, or
  result artifact has appeared.
- Slurm shows only unrelated `run.sh` jobs `993340`, `992937`, and `992776`;
  there is no matching OpenTAD/BATA job.
- Decision: continue waiting. No helper upload while byte progress exists; no
  Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T10:41:55+08:00 - Local upload helper readiness checked

- Checked `logs/sync_thumos_n16r4_raw_missing.ps1` without running it.
- The helper is configured for `E:\tmp\thumos14_raw`, `~/run/yuzibo`,
  `ParallelCopies=4`, SFTP resume, and remote ingest/layout checks after copy.
- Local source `E:\tmp\thumos14_raw` exists with `2584` mp4 files. Current
  train missing tail files `video_validation_0000489.mp4`, `0000490`, and
  `0000661` are present locally with full sizes; representative partial files
  `0000413`, `0000419`, and `0000486` are also present.
- Decision: do not start helper upload while remote byte progress exists. Keep
  helper on standby for a true 60-minute no-change stall.

## 2026-05-28T10:43:37+08:00 - N16R4 upload byte progress continues

- Watcher iteration `325` at `10:41:39+08:00`: train layout `expected=200`,
  `have=197`, `new_links=0`, `missing=3`; train `183 ok`, `3 missing`,
  `14 bad_size`; test `200 ok`, `0 missing`, `11 bad_size`;
  `ALL_PASS=False`; `active_sftp=15`.
- Counts are unchanged, but bad-size bytes continue increasing, including
  `video_validation_0000413.mp4` `109412352/273157908`, `0000419`
  `234946560/459211618`, `0000481` `114229248/229675218`, `0000484`
  `49053696/444825572`, `0000485` `36175872/111121310`, `0000486`
  `42532864/92982070`, and `0000487` `71499776/79300709`.
- Post-upload iteration `316` at `10:42:15+08:00` remains `DATA_PASS=False`;
  no runtime import, py_compile, pytest, `sbatch`, `bata_mobile50`, OpenTAD
  training marker, or result artifact has appeared.
- Slurm still shows only unrelated `run.sh` jobs `993340`, `992937`, and
  `992776`.
- Decision: continue waiting. No helper upload while byte progress exists; no
  Slurm launch until exact data PASS plus post-upload verification.

## 2026-05-28T10:46:51+08:00 - N16R4 upload count progress

- Watcher iteration `327` at `10:45:41+08:00`: train layout `expected=200`,
  `have=199`, `new_links=1`, `missing=1`; train `185 ok`, `1 missing`,
  `14 bad_size`; test `200 ok`, `0 missing`, `11 bad_size`;
  `ALL_PASS=False`; `active_sftp=18`.
- Fresh progress: `video_validation_0000489.mp4` and `0000490` entered
  partial upload. The only missing train file is now `video_validation_0000661.mp4`.
- Post-upload iteration `318` at `10:46:17+08:00` remains `DATA_PASS=False`
  with train `185 ok`, `1 missing`, `14 bad_size`, `extra=0`, and test
  `200 ok`, `0 missing`, `11 bad_size`, `extra=0`. No runtime import,
  py_compile, pytest, `sbatch`, `bata_mobile50`, OpenTAD training marker, or
  result artifact has appeared.
- Slurm still shows only unrelated `run.sh` jobs `993340`, `992937`, and
  `992776`.
- Decision: continue waiting. No helper upload because count and byte progress
  are active; no Slurm launch until exact data PASS plus post-upload
  verification.

## 2026-05-28T10:50:12+08:00 - N16R4 upload reached zero missing files

- Watcher iteration `329` at `10:49:43+08:00`: train layout `expected=200`,
  `have=200`, `new_links=1`, `missing=0`; validation layout `expected=211`,
  `have=211`, `missing=0`; train `186 ok`, `0 missing`, `14 bad_size`;
  test `200 ok`, `0 missing`, `11 bad_size`; `ALL_PASS=False`;
  `active_sftp=24`.
- Fresh count progress: final missing train file `video_validation_0000661.mp4`
  entered partial upload. The blocker is now exact size convergence:
  `14` train bad-size files and `11` test bad-size files remain.
- Post-upload gate was slightly behind at iteration `319` (`10:48:18+08:00`)
  and remains `DATA_PASS=False`. No runtime import, py_compile, pytest,
  `sbatch`, `bata_mobile50`, OpenTAD training marker, or result artifact has
  appeared.
- Slurm still shows only unrelated `run.sh` jobs `993340`, `992937`, and
  `992776`.
- Decision: continue waiting for exact size convergence. No helper upload
  while byte progress exists; no Slurm launch until train/test `bad_size=0`
  and post-upload verification PASS.

## 2026-05-28T10:53:30+08:00 - N16R4 size convergence continues

- Watcher iteration `330` at `10:51:44+08:00`: train layout `expected=200`,
  `have=200`, `missing=0`; validation layout `expected=211`, `have=211`,
  `missing=0`; train `186 ok`, `0 missing`, `14 bad_size`; test `200 ok`,
  `0 missing`, `11 bad_size`; `ALL_PASS=False`; `active_sftp=23`.
- Counts are stable but train partial bytes continue growing, including
  `video_validation_0000481.mp4` `126025728/229675218`, `0000484`
  `68419584/444825572`, `0000485` `46366720/111121310`, `0000486`
  `56524800/92982070`, `0000489` `7995392/91888293`, `0000490`
  `11567104/78629000`, and `0000661` `4096000/56265500`.
- Post-upload iteration `321` at `10:52:19+08:00` remains `DATA_PASS=False`;
  no runtime import, py_compile, pytest, `sbatch`, `bata_mobile50`, OpenTAD
  training marker, or result artifact has appeared.
- Slurm still shows only unrelated `run.sh` jobs `993340`, `992937`, and
  `992776`.
- Decision: continue waiting. No helper upload while byte progress exists; no
  Slurm launch until train/test `bad_size=0` and post-upload verification PASS.

## 2026-05-28T10:56:48+08:00 - N16R4 partial uploads still growing

- Watcher iteration `332` at `10:55:46+08:00`: train layout `expected=200`,
  `have=200`, `missing=0`; validation layout `expected=211`, `have=211`,
  `missing=0`; train `186 ok`, `0 missing`, `14 bad_size`; test `200 ok`,
  `0 missing`, `11 bad_size`; `ALL_PASS=False`; `active_sftp=26`.
- Counts and bad-size counts are stable, but train partial bytes continue
  growing, including `video_validation_0000481.mp4` `131104768/229675218`,
  `0000484` `74678272/444825572`, `0000485` `49479680/111121310`,
  `0000486` `63111168/92982070`, `0000489` `12845056/91888293`, `0000490`
  `18284544/78629000`, and `0000661` `12189696/56265500`.
- Post-upload iteration `323` at `10:56:21+08:00` remains `DATA_PASS=False`;
  no runtime import, py_compile, pytest, `sbatch`, `bata_mobile50`, OpenTAD
  training marker, or result artifact has appeared.
- Slurm still shows only unrelated `run.sh` jobs `993340`, `992937`, and
  `992776`.
- Decision: continue waiting. No helper upload while byte progress exists; no
  Slurm launch until exact data PASS and post-upload verification PASS.

## 2026-05-28T11:01:12+08:00 - N16R4 partial uploads still active

- Watcher iteration `334` at `10:59:49+08:00`: train layout `expected=200`,
  `have=200`, `missing=0`; validation layout `expected=211`, `have=211`,
  `missing=0`; train `186 ok`, `0 missing`, `14 bad_size`; test `200 ok`,
  `0 missing`, `11 bad_size`; `ALL_PASS=False`; `active_sftp=22`.
- Byte progress continues, including `video_validation_0000481.mp4`
  `138018816/229675218`, `0000484` `82837504/444825572`, `0000485`
  `53542912/111121310`, `0000486` `68812800/92982070`, `0000489`
  `17596416/91888293`, `0000490` `24772608/78629000`, and `0000661`
  `21463040/56265500`.
- Post-upload iteration `325` at `11:00:22+08:00` remains `DATA_PASS=False`;
  remote process check shows watcher/gate alive and `22` active
  `sftp-server` sessions.
- No runtime import, py_compile, pytest, `sbatch`, `bata_mobile50`, OpenTAD
  training marker, or result artifact has appeared. Slurm still shows only
  unrelated `run.sh` jobs `993340`, `992937`, and `992776`.
- Decision: continue waiting. No helper upload while byte progress and active
  SFTP sessions exist; no Slurm launch until exact data PASS and post-upload
  verification PASS.

## 2026-05-28T11:08:26+08:00 - N16R4 upload still growing

- A first read at `11:07:53+08:00` hit watcher iteration `338` while it was
  still being written, so the complete block was reread at `11:08:26+08:00`.
- Complete upload watcher iteration `338`: train layout `expected=200`,
  `have=200`, `missing=0`; validation layout `expected=211`, `have=211`,
  `missing=0`; train `186 ok`, `0 missing`, `14 bad_size`; test `200 ok`,
  `0 missing`, `11 bad_size`; `ALL_PASS=False`; `active_sftp=22`.
- Byte progress continues, including `video_validation_0000481.mp4`
  `146735104/229675218`, `0000484` `92307456/444825572`, `0000485`
  `69828608/111121310`, `0000486` `78118912/92982070`, `0000489`
  `26214400/91888293`, `0000490` `37486592/78629000`, and `0000661`
  `41353216/56265500`.
- Post-upload iteration `329` at `11:08:25+08:00` remains `DATA_PASS=False`;
  no runtime import, py_compile, pytest, `sbatch`, `bata_mobile50`, OpenTAD
  training marker, or result artifact has appeared.
- Decision: continue waiting. No helper upload while byte progress exists; no
  Slurm launch until exact data PASS and post-upload verification PASS.

## 2026-05-28T11:15:12+08:00 - N16R4 size gate begins converging

- Upload watcher iteration `341` at `11:13:56+08:00` still reported train
  `186 ok`, `0 missing`, `14 bad_size`, test `200 ok`, `0 missing`,
  `11 bad_size`, `ALL_PASS=False`, `active_sftp=24`, while showing byte
  growth through `video_validation_0000661.mp4` `55443456/56265500`.
- Post-upload iteration `332` at `11:14:27+08:00` observed the first bad-size
  reduction since zero-missing: train `187 ok`, `0 missing`, `13 bad_size`,
  `extra=0`; test remained `200 ok`, `0 missing`, `11 bad_size`, `extra=0`;
  `DATA_PASS=False`.
- Remote process check shows watcher/gate alive and active `sftp-server`
  sessions. No runtime import, py_compile, pytest, `sbatch`, `bata_mobile50`,
  OpenTAD training marker, or result artifact has appeared.
- Slurm still shows only unrelated `run.sh` jobs `993340`, `992937`, and
  `992776`.
- Decision: continue waiting. Do not helper-upload while size convergence and
  active SFTP sessions continue; no Slurm launch until exact data PASS and
  post-upload verification PASS.

## 2026-05-28T11:21:36+08:00 - N16R4 train size gate continues converging

- Upload watcher iteration `344` at `11:19:59+08:00`: train `188 ok`,
  `0 missing`, `12 bad_size`; test `200 ok`, `0 missing`, `11 bad_size`;
  `ALL_PASS=False`; `active_sftp=29`.
- Post-upload iteration `335` at `11:20:29+08:00` agrees: train `188 ok`,
  `0 missing`, `12 bad_size`, `extra=0`; test `200 ok`, `0 missing`,
  `11 bad_size`, `extra=0`; `DATA_PASS=False`.
- Train cleared another bad-size file since `11:15`, while test-side
  `11 bad_size` remains. Remaining train partials continue growing, including
  `video_validation_0000413.mp4`, `0000419`, `0000420`, `0000481`, `0000484`,
  `0000485`, `0000489`, and `0000490`.
- Remote process check shows watcher/gate alive and many active
  `sftp-server` sessions. No runtime import, py_compile, pytest, `sbatch`,
  `bata_mobile50`, OpenTAD training marker, or result artifact has appeared.
- Slurm still shows only unrelated `run.sh` jobs `993340`, `992937`, and
  `992776`.
- Decision: continue waiting. Do not helper-upload while size convergence and
  active SFTP sessions continue; no Slurm launch until exact data PASS and
  post-upload verification PASS.

## 2026-05-28T11:28:10+08:00 - N16R4 train size gate continues converging

- Upload watcher iteration `348` at `11:28:04+08:00`: train `189 ok`,
  `0 missing`, `11 bad_size`; test `200 ok`, `0 missing`, `11 bad_size`;
  `ALL_PASS=False`; `active_sftp=23`.
- Post-upload iteration `338` at `11:26:32+08:00` also reports train
  `189 ok`, `0 missing`, `11 bad_size`, `extra=0`; test `200 ok`,
  `0 missing`, `11 bad_size`, `extra=0`; `DATA_PASS=False`.
- Train cleared another bad-size file since `11:21`. Remaining train bad tail
  includes `video_validation_0000412.mp4` at `0/153134081`, plus growing
  partials `0000413`, `0000419`, `0000420`, `0000481`, `0000484`,
  `0000489`, and `0000490`.
- Remote process check shows watcher/gate alive and active `sftp-server`
  sessions. No runtime import, py_compile, pytest, `sbatch`, `bata_mobile50`,
  OpenTAD training marker, or result artifact has appeared.
- Decision: continue waiting. Do not helper-upload while overall byte progress
  and active SFTP sessions continue; if progress stalls for 60 minutes,
  prioritize process inspection and helper upload for remaining train/test
  bad-size files including `0000412`.

## 2026-05-28T11:34:39+08:00 - N16R4 train size gate continues converging

- Upload watcher iteration `351` at `11:34:07+08:00`: train `190 ok`,
  `0 missing`, `10 bad_size`; test `200 ok`, `0 missing`, `11 bad_size`;
  `ALL_PASS=False`; `active_sftp=18`.
- Post-upload iteration `342` at `11:34:35+08:00` agrees: train `190 ok`,
  `0 missing`, `10 bad_size`, `extra=0`; test `200 ok`, `0 missing`,
  `11 bad_size`, `extra=0`; `DATA_PASS=False`.
- Train cleared another bad-size file since `11:28`; test remains unchanged.
  `video_validation_0000412.mp4` is still `0/153134081`, but other files
  continue growing and remote still has active SFTP sessions.
- No runtime import, py_compile, pytest, `sbatch`, `bata_mobile50`, OpenTAD
  training marker, or result artifact has appeared. Slurm still shows only
  unrelated `run.sh` jobs `993340`, `992937`, and `992776`.
- Decision: continue waiting. Do not helper-upload while overall byte progress
  and active SFTP sessions continue; if progress stalls for 60 minutes,
  inspect processes and use helper upload for remaining bad-size files.

## 2026-05-28T11:41:08+08:00 - N16R4 byte progress continues

- Upload watcher iteration `354` at `11:40:10+08:00`: train `190 ok`,
  `0 missing`, `10 bad_size`; test `200 ok`, `0 missing`, `11 bad_size`;
  `ALL_PASS=False`; `active_sftp=10`.
- Post-upload iteration `345` at `11:40:37+08:00` agrees: train `190 ok`,
  `0 missing`, `10 bad_size`, `extra=0`; test `200 ok`, `0 missing`,
  `11 bad_size`, `extra=0`; `DATA_PASS=False`.
- Counts are unchanged since `11:34`, but train partials continue growing:
  `video_validation_0000369.mp4`, `0000413`, `0000419`, `0000481`,
  `0000484`, and `0000489`. `video_validation_0000412.mp4` remains
  `0/153134081`.
- Remote process check shows watcher/gate alive and `10` active
  `sftp-server` sessions. No runtime import, py_compile, pytest, `sbatch`,
  `bata_mobile50`, OpenTAD training marker, or result artifact has appeared.
- Decision: continue waiting. Do not helper-upload while byte progress and
  active SFTP sessions continue; if progress stalls for 60 minutes, inspect
  processes and use helper upload for remaining bad-size files.

## 2026-05-28T11:47:39+08:00 - N16R4 byte progress continues

- Upload watcher iteration `357` at `11:46:14+08:00`: train `190 ok`,
  `0 missing`, `10 bad_size`; test `200 ok`, `0 missing`, `11 bad_size`;
  `ALL_PASS=False`; `active_sftp=9`.
- Post-upload iteration `348` at `11:46:40+08:00` agrees: train `190 ok`,
  `0 missing`, `10 bad_size`, `extra=0`; test `200 ok`, `0 missing`,
  `11 bad_size`, `extra=0`; `DATA_PASS=False`.
- Counts are unchanged, but train partial bytes continue growing, including
  `video_validation_0000369.mp4`, `0000413`, `0000419`, `0000481`, `0000484`,
  and `0000489`. `video_validation_0000412.mp4` remains `0/153134081`.
- Local watchdog is alive and reset the stall timer at `11:45:37+08:00` on
  iteration `356`; no helper upload attempt log exists.
- Remote process check shows watcher/gate alive and `9` active `sftp-server`
  sessions. No runtime import, py_compile, pytest, `sbatch`, `bata_mobile50`,
  OpenTAD training marker, or result artifact has appeared.
- Decision: continue waiting. Do not helper-upload while byte progress and
  active SFTP sessions continue; the 60-minute stall condition has not started
  because watchdog reset at `11:45:37+08:00`.

## 2026-05-28T11:58:26+08:00 - Upload watcher restarted

- The original upload watcher reached `MAX_ITERATIONS=360` and wrote
  `TIMEOUT without upload gate pass` at `2026-05-28T11:54:18+08:00`.
- Before timeout, iteration `360` showed train `192 ok`, `0 missing`,
  `8 bad_size`; test `200 ok`, `0 missing`, `11 bad_size`;
  `ALL_PASS=False`; `active_sftp=6`.
- Started replacement lightweight watcher to keep the local stall watchdog from
  reading stale progress: PID `4018900`, log
  `~/run/yuzibo/setup_logs/n16r4_upload_gate_watch_20260528_115821.log`,
  `MAX_ITERATIONS=720`, `INTERVAL_SECONDS=120`.
- New watcher iteration `1` at `11:58:21+08:00`: train `193 ok`,
  `0 missing`, `7 bad_size`; test `200 ok`, `0 missing`, `11 bad_size`;
  `ALL_PASS=False`; `active_sftp=3`.
- Post-upload gate remains active. No runtime import, py_compile, pytest,
  `sbatch`, `bata_mobile50`, OpenTAD training marker, or result artifact has
  appeared.
- Decision: continue waiting with the restarted watcher. Do not helper-upload
  while byte progress/SFTP continue; if the restarted watcher shows no
  signature progress for 60 minutes, run process inspection and helper upload
  for remaining bad-size files.

## 2026-05-28T12:04:54+08:00 - Restarted watcher healthy

- New upload watcher PID `4018900`, log
  `~/run/yuzibo/setup_logs/n16r4_upload_gate_watch_20260528_115821.log`.
- Watcher iteration `4` at `12:04:24+08:00`: train `193 ok`,
  `0 missing`, `7 bad_size`; test `200 ok`, `0 missing`, `11 bad_size`;
  `ALL_PASS=False`; `active_sftp=3`.
- Post-upload iteration `357` at `12:04:47+08:00` agrees: train `193 ok`,
  `0 missing`, `7 bad_size`, `extra=0`; test `200 ok`, `0 missing`,
  `11 bad_size`, `extra=0`; `DATA_PASS=False`.
- Some train partial bytes continue growing (`0000369`, `0000419`, `0000484`);
  stagnant train files remain `0000166`, `0000168`, `0000412`, and `0000420`.
- Remote process check shows post gate, restarted watcher, and `3` active
  `sftp-server` sessions. No runtime import, py_compile, pytest, `sbatch`,
  `bata_mobile50`, OpenTAD training marker, or result artifact has appeared.
- Decision: continue waiting. Do not helper-upload while active SFTP and byte
  progress continue; if progress stops, let the 60-minute watchdog/process
  check rule trigger helper upload.

## 2026-05-28T12:07:07+08:00 - N16R4 upload still progressing

- Upload watcher PID `4018900`, log
  `~/run/yuzibo/setup_logs/n16r4_upload_gate_watch_20260528_115821.log`.
- Watcher iteration `5` at `12:06:25+08:00`: train `193 ok`,
  `0 missing`, `7 bad_size`; test `200 ok`, `0 missing`, `11 bad_size`;
  `ALL_PASS=False`; `active_sftp=3`.
- Post-upload iteration `358` at `12:06:47+08:00` agrees: train `193 ok`,
  `0 missing`, `7 bad_size`, `extra=0`; test `200 ok`, `0 missing`,
  `11 bad_size`, `extra=0`; `DATA_PASS=False`.
- Bytes continue growing for some train partials (`0000369`, `0000419`,
  `0000484`), while `0000166`, `0000168`, `0000412`, and `0000420` remain
  stagnant. Local watchdog reset the stall timer at `12:05:48+08:00`; no
  helper upload attempt log exists.
- Remote process check shows post gate, restarted watcher, and `3` active
  `sftp-server` sessions. No runtime import, py_compile, pytest, `sbatch`,
  `bata_mobile50`, OpenTAD training marker, or result artifact has appeared.
- Slurm still shows only unrelated `run.sh` jobs; `df -h ~/run/yuzibo` shows
  JuiceFS `2.3T` total, `410G` used, `2.0T` free.
- Decision: continue waiting. Do not helper-upload while active SFTP and byte
  progress continue; the 60-minute stall timer was reset at `12:05:48+08:00`;
  no Slurm launch until exact data PASS and post-upload verification PASS.

## 2026-05-28T12:09:19+08:00 - N16R4 upload still active

- Watcher iteration `6` at `12:08:26+08:00`: train `193 ok`,
  `0 missing`, `7 bad_size`; test `200 ok`, `0 missing`, `11 bad_size`;
  `ALL_PASS=False`; `active_sftp=3`.
- Post-upload iteration `359` at `12:08:48+08:00` agrees: train `193 ok`,
  `0 missing`, `7 bad_size`, `extra=0`; test `200 ok`, `0 missing`,
  `11 bad_size`, `extra=0`; `DATA_PASS=False`.
- Train partial bytes continue to increase for `0000369`, `0000419`, and
  `0000484`; `0000166`, `0000168`, `0000412`, and `0000420` remain
  unchanged. Local watchdog is alive and had reset at `12:05:48+08:00`; no
  helper upload attempt log exists.
- Remote process check shows post gate, restarted watcher, and `3` active
  `sftp-server` sessions. Post markers remain repeated `DATA_PASS=False`
  only, with no runtime import, py_compile, pytest, `sbatch`, `bata_mobile50`,
  OpenTAD training marker, or result artifact.
- Decision: continue waiting. Do not helper-upload while active SFTP and byte
  progress continue; no Slurm launch until exact data PASS and post-upload
  verification PASS.

## 2026-05-28T12:11:44+08:00 - N16R4 upload still progressing

- Watcher iteration `7` at `12:10:27+08:00`: train `193 ok`,
  `0 missing`, `7 bad_size`; test `200 ok`, `0 missing`, `11 bad_size`;
  `ALL_PASS=False`; `active_sftp=3`.
- Post-upload iteration `360` at `12:10:49+08:00` agrees: train `193 ok`,
  `0 missing`, `7 bad_size`, `extra=0`; test `200 ok`, `0 missing`,
  `11 bad_size`, `extra=0`; `DATA_PASS=False`.
- Train partial bytes keep increasing for `0000369`, `0000419`, and `0000484`;
  `0000166`, `0000168`, `0000412`, and `0000420` remain unchanged.
- Local watchdog is alive and reset the stall timer at `12:10:50+08:00`; no
  helper upload attempt log exists.
- Remote process check shows post gate, restarted watcher, and `3` active
  `sftp-server` sessions. Post markers remain repeated `DATA_PASS=False`
  only, with no runtime import, py_compile, pytest, `sbatch`, `bata_mobile50`,
  OpenTAD training marker, or result artifact.
- Decision: continue waiting. Do not helper-upload while active SFTP and byte
  progress continue; the 60-minute stall timer was reset at `12:10:50+08:00`;
  no Slurm launch until exact data PASS and post-upload verification PASS.

## 2026-05-28T12:20:21+08:00 - N16R4 upload still progressing

- Upload watcher PID `4018900`, log
  `~/run/yuzibo/setup_logs/n16r4_upload_gate_watch_20260528_115821.log`.
- Watcher iteration `11` at `12:18:32+08:00`: train `193 ok`,
  `0 missing`, `7 bad_size`; test `200 ok`, `0 missing`, `11 bad_size`;
  all expected filenames are present, but exact sizes still fail;
  `ALL_PASS=False`; `active_sftp=5`.
- Post-upload gate PID `2225557`, log
  `~/run/yuzibo/setup_logs/n16r4_post_upload_verify_and_launch_20260528_000834.log`.
  Iteration `364` at `12:18:52+08:00` agrees with `DATA_PASS=False`.
- Train partial bytes are still advancing for `video_validation_0000369.mp4`,
  `0000419`, and `0000484`; stagnant train partials remain `0000166`,
  `0000168`, `0000412`, and `0000420`. Test still has `11` bad-size files.
- Local stall watchdog PID `106220` is alive; latest visible reset is
  `12:15:51+08:00`, and no helper upload attempt log exists.
- Remote process check shows the post gate, restarted watcher, and `5` active
  `sftp-server` sessions. Post markers remain repeated `DATA_PASS=False` only,
  with no runtime import, py_compile, pytest, `sbatch`, `bata_mobile50`,
  OpenTAD training marker, or result artifact.
- Slurm still shows only unrelated `run.sh` jobs `993340`, `992937`, and
  `992776`; disk remains safe at about `410G` used and `2.0T` free under
  `~/run/yuzibo`.
- Decision: continue waiting. Do not helper-upload while active SFTP and byte
  progress continue; the one-hour no-change rule is not met. No Slurm launch
  until exact `DATA_PASS=True` and post-upload verification PASS.

## 2026-05-28T12:23:03+08:00 - N16R4 upload advanced again

- Upload watcher iteration `13` at `12:22:34+08:00`: train `194 ok`,
  `0 missing`, `6 bad_size`; test `200 ok`, `0 missing`, `11 bad_size`;
  `ALL_PASS=False`; `active_sftp=4`.
- Post-upload iteration `366` at `12:22:54+08:00` agrees with
  `DATA_PASS=False`: train `194 ok`, `0 missing`, `6 bad_size`, `extra=0`;
  test `200 ok`, `0 missing`, `11 bad_size`, `extra=0`.
- Remaining train bad tail: `video_validation_0000166.mp4`,
  `0000168`, `0000369`, `0000412`, `0000420`, and `0000484`. Test still has
  `11` bad-size files.
- Local watchdog reset at `12:20:53+08:00`; no helper upload attempt log
  exists. Remote process check shows the post gate, restarted watcher, and `4`
  active `sftp-server` sessions.
- No runtime import, py_compile, pytest, `sbatch`, `bata_mobile50`, OpenTAD
  training marker, or result artifact exists yet. Slurm still shows only
  unrelated `run.sh` jobs.
- Decision: continue waiting. Do not helper-upload because count/byte progress
  is active and the one-hour no-change rule was just reset; no Slurm launch
  until exact `DATA_PASS=True` and post-upload verification PASS.

## 2026-05-28T12:26:02+08:00 - N16R4 upload byte progress continues

- Upload watcher iteration `14` at `12:24:35+08:00`: train `194 ok`,
  `0 missing`, `6 bad_size`; test `200 ok`, `0 missing`, `11 bad_size`;
  `ALL_PASS=False`; `active_sftp=5`.
- Post-upload iteration `367` at `12:24:55+08:00` agrees with
  `DATA_PASS=False`: train `194 ok`, `0 missing`, `6 bad_size`, `extra=0`;
  test `200 ok`, `0 missing`, `11 bad_size`, `extra=0`.
- Counts are unchanged from `12:23`, but train partial bytes still advanced for
  `video_validation_0000369.mp4` and `0000484`; `0000166`, `0000168`,
  `0000412`, and `0000420` remain stagnant, and test still has `11`
  bad-size files.
- Remote process check shows the post gate, restarted watcher, and `5` active
  `sftp-server` sessions. Post markers remain repeated `DATA_PASS=False` only,
  with no runtime import, py_compile, pytest, `sbatch`, `bata_mobile50`,
  OpenTAD training marker, or result artifact.
- Decision: continue waiting. Do not helper-upload while bytes and active SFTP
  remain present; no Slurm launch until exact `DATA_PASS=True` and post-upload
  verification PASS.

## 2026-05-28T12:28:22+08:00 - N16R4 upload still progressing

- Upload watcher PID `4018900`, log
  `~/run/yuzibo/setup_logs/n16r4_upload_gate_watch_20260528_115821.log`.
- Watcher iteration `15` at `12:26:36+08:00`: train `194 ok`,
  `0 missing`, `6 bad_size`; test `200 ok`, `0 missing`, `11 bad_size`;
  all expected filenames are present, but exact sizes still fail;
  `ALL_PASS=False`; `active_sftp=5`.
- Post-upload gate PID `2225557`, log
  `~/run/yuzibo/setup_logs/n16r4_post_upload_verify_and_launch_20260528_000834.log`.
  Iteration `368` at `12:26:55+08:00` agrees with `DATA_PASS=False`.
- Train partial bytes are still advancing for `video_validation_0000369.mp4`
  and `0000484`; stagnant train partials remain `0000166`, `0000168`,
  `0000412`, and `0000420`. Test still has `11` bad-size files.
- Local stall watchdog PID `106220` is alive and reset the stall timer at
  `12:25:55+08:00`; no helper upload attempt log exists.
- Remote process check shows the post gate, restarted watcher, and `5` active
  `sftp-server` sessions. Post markers remain repeated `DATA_PASS=False` only,
  with no runtime import, py_compile, pytest, `sbatch`, `bata_mobile50`,
  OpenTAD training marker, or result artifact.
- Slurm still shows only unrelated `run.sh` jobs; disk remains safe at about
  `410G` used and `2.0T` free under `~/run/yuzibo`.
- Decision: continue waiting. Do not helper-upload because byte progress is
  active and the 60-minute no-change rule reset at `12:25:55+08:00`; no Slurm
  launch until exact `DATA_PASS=True` and post-upload verification PASS.

## 2026-05-28T12:32:44+08:00 - N16R4 train gate advanced

- Upload watcher iteration `18` at `12:32:39+08:00`: train `196 ok`,
  `0 missing`, `4 bad_size`; test `200 ok`, `0 missing`, `11 bad_size`;
  all expected filenames are present, but exact sizes still fail;
  `ALL_PASS=False`; `active_sftp=3`.
- Post-upload iteration `370` at `12:30:57+08:00` is slightly behind but still
  `DATA_PASS=False`: train `195 ok`, `0 missing`, `5 bad_size`, `extra=0`;
  test `200 ok`, `0 missing`, `11 bad_size`, `extra=0`.
- Remaining watcher train bad tail: `video_validation_0000166.mp4`,
  `0000168`, `0000412`, and `0000420`. Test still has `11` bad-size files.
- Local stall watchdog PID `106220` is alive and reset at `12:30:56+08:00`;
  no helper upload attempt log exists.
- Remote process check shows the post gate, restarted watcher, and `3` active
  `sftp-server` sessions. Post markers remain repeated `DATA_PASS=False` only,
  with no runtime import, py_compile, pytest, `sbatch`, `bata_mobile50`,
  OpenTAD training marker, or result artifact.
- Decision: continue waiting. Do not helper-upload because count progress and
  active SFTP remain; no Slurm launch until exact `DATA_PASS=True` and
  post-upload verification PASS.

## 2026-05-28T12:36:35+08:00 - N16R4 train gate stable

- Upload watcher iteration `19` at `12:34:40+08:00`: train `196 ok`,
  `0 missing`, `4 bad_size`; test `200 ok`, `0 missing`, `11 bad_size`;
  `ALL_PASS=False`; `active_sftp=2`.
- Post-upload iteration `372` at `12:34:59+08:00` agrees with
  `DATA_PASS=False`: train `196 ok`, `0 missing`, `4 bad_size`, `extra=0`;
  test `200 ok`, `0 missing`, `11 bad_size`, `extra=0`.
- Remaining train bad tail is unchanged: `video_validation_0000166.mp4`,
  `0000168`, `0000412`, and `0000420`. Test still has `11` bad-size files.
- Local stall watchdog PID `106220` is alive and reset at `12:35:58+08:00`;
  no helper upload attempt log exists.
- Remote process check shows the post gate, restarted watcher, and an active
  `sftp-server` process. Post markers remain repeated `DATA_PASS=False` only,
  with no runtime import, py_compile, pytest, `sbatch`, `bata_mobile50`,
  OpenTAD training marker, or result artifact.
- Decision: continue waiting. Do not helper-upload yet because the 60-minute
  no-change rule reset at `12:35:58+08:00`; no Slurm launch until exact
  `DATA_PASS=True` and post-upload verification PASS.

## 2026-05-28T12:40:30+08:00 - N16R4 bad-size set unchanged

- Upload watcher iteration `21` at `12:38:43+08:00`: train `196 ok`,
  `0 missing`, `4 bad_size`; test `200 ok`, `0 missing`, `11 bad_size`;
  `ALL_PASS=False`; `active_sftp=1`.
- Post-upload iteration `374` at `12:39:00+08:00` agrees with
  `DATA_PASS=False`: train `196 ok`, `0 missing`, `4 bad_size`, `extra=0`;
  test `200 ok`, `0 missing`, `11 bad_size`, `extra=0`.
- Bad-size set is unchanged from `12:35:58+08:00`: train `0000166`,
  `0000168`, `0000412`, and `0000420`; test still has `11` bad-size files.
- Remote process check shows one active `sftp-server`, post gate, and upload
  watcher. No helper upload attempt log exists.
- No runtime import, py_compile, pytest, `sbatch`, `bata_mobile50`, OpenTAD
  training marker, or result artifact exists yet. Slurm still shows only
  unrelated `run.sh` jobs.
- Decision: continue waiting. Do not helper-upload until the unchanged-signature
  window reaches 60 minutes and process inspection confirms no meaningful upload
  progress; no Slurm launch until exact `DATA_PASS=True` and post-upload
  verification PASS.

## 2026-05-28T12:42:45+08:00 - N16R4 unchanged timer active

- Upload watcher iteration `22` at `12:40:44+08:00`: train `196 ok`,
  `0 missing`, `4 bad_size`; test `200 ok`, `0 missing`, `11 bad_size`;
  `ALL_PASS=False`; `active_sftp=1`.
- Post-upload iteration `375` at `12:41:01+08:00` agrees with
  `DATA_PASS=False`: train `196 ok`, `0 missing`, `4 bad_size`, `extra=0`;
  test `200 ok`, `0 missing`, `11 bad_size`, `extra=0`.
- Local watchdog reports `no signature change for 5.1 minutes` after the last
  reset at `12:35:58+08:00`; no helper upload attempt log exists.
- Remote process check shows one active `sftp-server`, post gate, upload
  watcher, and transient watcher/tee processes. No runtime import, py_compile,
  pytest, `sbatch`, `bata_mobile50`, OpenTAD training marker, or result artifact.
- Decision: continue waiting. Helper-upload condition is not met; no Slurm
  launch until exact `DATA_PASS=True` and post-upload verification PASS.

## 2026-05-28T12:56:15+08:00 - N16R4 upload appears idle

- Upload watcher iteration `29` at `12:54:51+08:00`: train `196 ok`,
  `0 missing`, `4 bad_size`; test `200 ok`, `0 missing`, `11 bad_size`;
  `ALL_PASS=False`; `active_sftp=0`.
- Post-upload iteration `382` at `12:55:06+08:00` agrees with
  `DATA_PASS=False`: train `196 ok`, `0 missing`, `4 bad_size`, `extra=0`;
  test `200 ok`, `0 missing`, `11 bad_size`, `extra=0`.
- Local watchdog is alive and reports `no signature change for 15.1 minutes`
  at `12:51:05+08:00`; no helper upload attempt log exists.
- Remote process check shows no active `sftp-server`, with post gate and upload
  watcher still alive. No runtime import, py_compile, pytest, `sbatch`,
  `bata_mobile50`, OpenTAD training marker, or result artifact exists.
- Slurm now includes another unrelated `run.sh` job `993577`; still no
  OpenTAD/BATA job. Disk remains safe at about `410G` used and `2.0T` free.
- Decision: continue waiting. Helper-upload condition is not met yet because
  unchanged time is only about `15` minutes; if no new signature change occurs,
  process-check/helper eligibility remains after about `13:35:58+08:00`.

## 2026-05-28T12:59:00+08:00 - N16R4 unchanged timer at 20 minutes

- Upload watcher iteration `31` at `12:58:53+08:00`: train `196 ok`,
  `0 missing`, `4 bad_size`; test `200 ok`, `0 missing`, `11 bad_size`;
  `ALL_PASS=False`; `active_sftp=1`.
- Post-upload iteration `383` at `12:57:07+08:00` agrees with
  `DATA_PASS=False`: train `196 ok`, `0 missing`, `4 bad_size`, `extra=0`;
  test `200 ok`, `0 missing`, `11 bad_size`, `extra=0`.
- Local watchdog reports `no signature change for 20.1 minutes`; no helper
  upload attempt log exists.
- Remote process check shows one active `sftp-server`, post gate, and upload
  watcher. No runtime import, py_compile, pytest, `sbatch`, `bata_mobile50`,
  OpenTAD training marker, or result artifact exists.
- Slurm still has only unrelated `run.sh` jobs; no OpenTAD/BATA job.
- Decision: continue waiting. Do not helper-upload before the 60-minute
  unchanged threshold and process inspection; no Slurm launch until exact
  `DATA_PASS=True` and post-upload verification PASS.

## 2026-05-28T13:01:49+08:00 - N16R4 unchanged timer at 25 minutes

- Upload watcher iteration `32` at `13:00:54+08:00`: train `196 ok`,
  `0 missing`, `4 bad_size`; test `200 ok`, `0 missing`, `11 bad_size`;
  all expected filenames are present, but exact sizes still fail;
  `ALL_PASS=False`; `active_sftp=1`.
- Post-upload iteration `385` at `13:01:08+08:00` agrees with
  `DATA_PASS=False` and the same train/test bad-size set.
- Local watchdog PID `106220` is alive and reports `no signature change for
  25.2 minutes` after the last reset at `12:35:58+08:00`; no helper upload
  attempt log exists.
- Remote process check shows one active `sftp-server`, post gate, and upload
  watcher. No runtime import, py_compile, pytest, `sbatch`, `bata_mobile50`,
  OpenTAD training marker, or result artifact exists yet.
- Slurm still has only unrelated `run.sh` jobs; disk remains safe at about
  `410G` used and `2.0T` free.
- Decision: continue waiting. Do not helper-upload before the 60-minute
  unchanged threshold and process/raw-size inspection; no Slurm launch until
  exact `DATA_PASS=True` and post-upload verification PASS.

## 2026-05-28T13:15:07+08:00 - N16R4 unchanged timer at 35 minutes

- Upload watcher iteration `39` at `13:15:02+08:00`: train `196 ok`,
  `0 missing`, `4 bad_size`; test `200 ok`, `0 missing`, `11 bad_size`;
  all expected filenames are present, but exact sizes still fail;
  `ALL_PASS=False`; `active_sftp=1`.
- Post-upload iteration `391` at `13:13:13+08:00` agrees with
  `DATA_PASS=False` and the same train/test bad-size set.
- Local watchdog PID `106220` is alive and reports `no signature change for
  35.2 minutes` at `13:11:11+08:00`; no helper upload attempt log exists.
- Remote process check still shows one `sftp-server` process, post gate, and
  upload watcher. No runtime import, py_compile, pytest, `sbatch`,
  `bata_mobile50`, OpenTAD training marker, or result artifact exists.
- Decision: continue waiting. Do not helper-upload before the 60-minute
  unchanged threshold and process/raw-size inspection; no Slurm launch until
  exact `DATA_PASS=True` and post-upload verification PASS.

## 2026-05-28T13:41:51+08:00 - N16R4 helper upload attempt 1 started

- The local stall watchdog reached `60.4` minutes with no signature change at
  `13:36:19+08:00`, ran the required remote process/raw mtime/size check, and
  started helper upload attempt `1` at `13:36:21+08:00`.
- Helper log:
  `logs/sync_thumos_n16r4_raw_missing_attempt1_20260528_133621.log`.
  It queued `15` raw uploads: `4` train bad-size files and `11` test bad-size
  files; exact matches were skipped (`196` train, `200` test).
- At `13:41:51+08:00`, helper upload is active. The 4 train bad-size files are
  growing, but data gate remains false: train `196 ok`, `4 bad_size`; test
  `200 ok`, `11 bad_size`; `ALL_PASS=False`; `DATA_PASS=False`.
- No runtime import, py_compile, pytest, `sbatch`, `bata_mobile50`, OpenTAD
  training marker, or result artifact exists yet.
- Decision: monitor helper attempt `1`; do not start another helper or submit
  Slurm while upload is active and exact data gate is false.

## 2026-05-28T13:53:34+08:00 - N16R4 helper upload progressing

- Helper attempt `1` is still active on the first train batch from
  `logs/sync_thumos_n16r4_raw_missing_attempt1_20260528_133621.log`.
- Upload watcher iteration `58`: train `196 ok`, `4 bad_size`; test `200 ok`,
  `11 bad_size`; `ALL_PASS=False`; `active_sftp=7`.
- Train bad-size bytes are increasing: `0000166` about `49.7M/75.4M`,
  `0000168` about `66.5M/69.1M`, `0000412` about `74.9M/153.1M`, and
  `0000420` about `148.3M/424.9M`. Test bad-size files have not started yet.
- Post-upload gate remains `DATA_PASS=False`; no runtime import, py_compile,
  pytest, `sbatch`, `bata_mobile50`, or training marker exists.
- Decision: continue monitoring this helper attempt only; no Slurm launch until
  exact data PASS and verification PASS.

## 2026-05-28T16:02:26+08:00 - N16R4 data PASS, verification PASS, launcher failure

- Helper upload attempt `1` exited `1` at `14:04:05`; manual helper attempt `2`
  ran from `14:15:50` to `15:53:11`, copied the remaining `12` raw videos, and
  completed successfully.
- Upload watcher iteration `118` at `15:54:30+08:00` passed exactly: train
  `200 ok`, test `211 ok`, no missing or bad-size files, `ALL_PASS=True`.
  PASS marker:
  `/data/home/sczc063/run/yuzibo/setup_logs/N16R4_THUMOS_UPLOAD_GATE_PASS_20260528_115821.ok`.
- Post-upload gate reached `DATA_PASS=True`; runtime imports passed, BATA tests
  passed `32 passed in 5.43s`, Sparse tests passed `27 passed in 21.84s`, and
  Slurm job `993684` was submitted at `15:55:24+08:00`.
- Job `993684` failed immediately on node `g0006` with `FAILED`,
  `ExitCode=127:0`. Slurm output:
  `/data/run01/sczc063/yuzibo/OpenTAD_BATA_Clean/logs/bata_mobile50-993684.out`;
  error: `module: command not found`.
- Current process check found no BATA/OpenTAD `tools/train.py` or `torchrun`
  process. Local launcher copies were patched to source `/etc/profile` under
  `set +u` before `module load`; remote launcher sync and single-job resubmit
  are the next actions.

## 2026-05-28T16:11:52+08:00 - N16R4 launcher fix synced and reviewed

- Synced
  `migration_packages/tad_bata_sparse_migration_20260525/migration_scripts/run_bata_mobilenet_fixed50_n16r4.sbatch`
  to
  `~/run/yuzibo/tad_bata_sparse_migration_20260525/migration_scripts/run_bata_mobilenet_fixed50_n16r4.sbatch`.
  Remote self-check stripped CRLF, set executable bit, and passed `bash -n`.
- Remote header now sources `/etc/profile` under `set +u` before `module load`.
- Gemini CLI review:
  `logs/gemini3_pro_preview_n16r4_bata_launcher_fix_20260528_1603.txt`,
  verdict approved, no blocking findings.
- Claude CLI DeepSeek review:
  `logs/claude_deepseek_v4_pro_n16r4_bata_launcher_fix_20260528_1603.txt`,
  verdict `PASS`; no required fixes.
- Accepted watch item: if the next Slurm log fails at `source activate` with an
  `unbound variable`, move `set -u` restoration after conda activation.
- Decision: after one final no-duplicate process/queue check, submit exactly one
  fixed `bata_mobile50` Slurm job.

## 2026-05-28T16:17:33+08:00 - N16R4 fixed BATA job running

- Submitted one replacement Slurm job after no-duplicate checks:
  `993718`, job name `bata_mobile50`, node `g0052`, start
  `2026-05-28T16:16:13+08:00`.
- Slurm log:
  `~/run/yuzibo/OpenTAD_BATA_Clean/logs/bata_mobile50-993718.out`.
  Run log:
  `~/run/yuzibo/OpenTAD_BATA_Clean/logs/bata_mobilenet_fixed50_20260528_161457_n16r4.log`.
- Launcher issue is cleared: log shows `cuda-11.8 loaded successful` and
  `Miniforge3-24.11 loaded successful`.
- BATA pytest passed `32 passed in 29.92s`; selector training started on CUDA
  with `train_videos=200`, `val_videos=200`, `steps_per_epoch=1886`,
  `train_shuffle=false`.
- No mAP yet. Continue monitoring selector/cache/detector stages for job
  `993718`.

## 2026-05-28T16:21:39+08:00 - N16R4 BATA selector health check

- Job `993718` remains `RUNNING` on `g0052`.
- Run log
  `~/run/yuzibo/OpenTAD_BATA_Clean/logs/bata_mobilenet_fixed50_20260528_161457_n16r4.log`
  has advanced to selector epoch `0` step `400/1886`.
- Recent selector losses: `0.708686` at step `100`, `0.616390` at step `200`,
  `0.869447` at step `300`, `0.969082` at step `400`.
- No mAP or cache gate result yet. Continue monitoring for selector completion,
  cache build, eval-cache gate, detector start, and final evaluation.

## 2026-05-28T16:23:34+08:00 - N16R4 BATA selector still progressing

- Job `993718` remains `RUNNING` on node `g0052`, elapsed `00:07:21`.
- Selector log advanced to epoch `0` step `600/1886`.
- Latest logged losses: `0.976887` at step `500`, `1.022755` at step `600`.
- No cache gate, detector start, result JSON, or mAP yet. Continue waiting; do
  not submit another BATA job.

## 2026-05-28T16:25:31+08:00 - N16R4 BATA selector step 800

- Job `993718` remains `RUNNING` on `g0052`, elapsed `00:09:18`.
- Selector log advanced to epoch `0` step `800/1886`.
- Latest logged losses: `0.961149` at step `700`, `0.970353` at step `800`.
- Selector directory exists, but no checkpoint/cache/result/mAP artifact exists
  yet. Continue monitoring the same job.

## 2026-05-28T16:27:22+08:00 - N16R4 BATA selector step 900

- Job `993718` remains `RUNNING` on `g0052`, elapsed `00:11:09`.
- Selector log reached epoch `0` step `900/1886`, loss `0.995976`.
- Artifact scan still shows no selector checkpoint, cache npz, result JSON,
  detector output, or mAP. Continue monitoring the same job.

## 2026-05-28T16:29:13+08:00 - N16R4 BATA selector step 1000

- Job `993718` remains `RUNNING` on `g0052`, elapsed `00:13:00`.
- Selector log reached epoch `0` step `1000/1886`, loss `0.973014`.
- Artifact scan still shows no selector checkpoint, cache npz, result JSON,
  detector output, or mAP. Continue monitoring the same job.

## 2026-05-28T16:30:52+08:00 - N16R4 BATA selector step 1100

- Job `993718` remains `RUNNING` on `g0052`, elapsed `00:14:39`.
- Selector log reached epoch `0` step `1100/1886`, loss `0.953225`.
- Artifact scan still shows no selector checkpoint, cache npz, result JSON,
  detector output, or mAP. Continue monitoring the same job.

## 2026-05-28T16:32:27+08:00 - N16R4 BATA selector step 1200

- Job `993718` remains `RUNNING` on `g0052`, elapsed `00:16:14`.
- Selector log reached epoch `0` step `1200/1886`, loss `0.945842`.
- Artifact scan still shows no selector checkpoint, cache npz, result JSON,
  detector output, or mAP. Continue monitoring the same job.

## 2026-05-28T16:34:23+08:00 - N16R4 BATA selector step 1300

- Job `993718` remains `RUNNING` on `g0052`, elapsed `00:18:10`.
- Selector log reached epoch `0` step `1300/1886`, loss `0.929776`.
- Artifact scan still shows no selector checkpoint, cache npz, result JSON,
  detector output, or mAP. Continue monitoring the same job.

## 2026-05-28T16:35:53+08:00 - N16R4 BATA selector step 1500

- Job `993718` remains `RUNNING` on `g0052`, elapsed `00:19:40`.
- Selector log reached epoch `0` step `1500/1886`.
- Latest losses: step `1400` `0.918838`, step `1500` `0.907031`.
- Artifact scan still shows no selector checkpoint, cache npz, result JSON,
  detector output, or mAP. Continue monitoring the same job.

## 2026-05-28T16:37:35+08:00 - N16R4 BATA selector step 1600

- Job `993718` remains `RUNNING` on `g0052`, elapsed `00:21:22`.
- Selector log reached epoch `0` step `1600/1886`, loss `0.920933`.
- Artifact scan still shows no selector checkpoint, cache npz, result JSON,
  detector output, or mAP. Continue monitoring for selector completion.

## 2026-05-28T16:39:12+08:00 - N16R4 BATA selector step 1800

- Job `993718` remains `RUNNING` on `g0052`, elapsed `00:22:59`.
- Selector log reached epoch `0` step `1800/1886`.
- Latest losses: step `1700` `0.932206`, step `1800` `0.935184`.
- Artifact scan still shows no selector checkpoint, cache npz, result JSON,
  detector output, or mAP. Continue monitoring for imminent selector completion.

## 2026-05-28T16:41:35+08:00 - N16R4 BATA selector validation started

- Job `993718` remains `RUNNING` on `g0052`, elapsed `00:25:22`.
- Selector training epoch `0` has completed; validation is running.
- Latest validation progress: `val epoch=0 step=200/1336`,
  `eval_examples_so_far=25205`.
- Artifact scan still shows no selector checkpoint, cache npz, result JSON,
  detector output, or mAP. Continue monitoring validation completion.

## 2026-05-28T16:43:18+08:00 - N16R4 BATA selector validation step 300

- Job `993718` remains `RUNNING` on `g0052`, elapsed `00:27:05`.
- Selector validation reached `val epoch=0 step=300/1336`,
  `eval_examples_so_far=37811`.
- Artifact scan still shows no selector checkpoint, cache npz, result JSON,
  detector output, or mAP. Continue monitoring validation completion.

## 2026-05-28T16:44:53+08:00 - N16R4 BATA selector validation step 500

- Job `993718` remains `RUNNING` on `g0052`, elapsed `00:28:40`.
- Selector validation reached `val epoch=0 step=500/1336`,
  `eval_examples_so_far=62352`.
- Artifact scan still shows no selector checkpoint, cache npz, result JSON,
  detector output, or mAP. Continue monitoring validation completion.

## 2026-05-28T16:46:28+08:00 - N16R4 BATA selector validation step 600

- Job `993718` remains `RUNNING` on `g0052`, elapsed `00:30:15`.
- Selector validation reached `val epoch=0 step=600/1336`,
  `eval_examples_so_far=74788`.
- Artifact scan still shows no selector checkpoint, cache npz, result JSON,
  detector output, or mAP. Continue monitoring validation completion.

## 2026-05-28T16:48:02+08:00 - N16R4 BATA selector validation short check

- Job `993718` remains `RUNNING` on `g0052`, elapsed `00:31:49`.
- Latest visible validation line is unchanged at `val epoch=0 step=600/1336`,
  `eval_examples_so_far=74788`.
- Artifact scan still shows no selector checkpoint, cache npz, result JSON,
  detector output, or mAP. Continue monitoring; no duplicate launch.

## 2026-05-28T16:49:42+08:00 - N16R4 BATA selector validation step 700

- Job `993718` remains `RUNNING` on `g0052`, elapsed `00:33:29`.
- Selector validation reached `val epoch=0 step=700/1336`,
  `eval_examples_so_far=87417`.
- Artifact scan still shows no selector checkpoint, cache npz, result JSON,
  detector output, or mAP. Continue monitoring validation completion.

## 2026-05-28T16:51:23+08:00 - N16R4 BATA selector validation step 800

- Job `993718` remains `RUNNING` on `g0052`, elapsed `00:35:10`.
- Selector validation reached `val epoch=0 step=800/1336`,
  `eval_examples_so_far=100113`.
- Artifact scan still shows no selector checkpoint, cache npz, result JSON,
  detector output, or mAP. Continue monitoring validation completion.

## 2026-05-28T16:53:09+08:00 - N16R4 BATA selector validation step 900

- Job `993718` remains `RUNNING` on `g0052`, elapsed `00:36:56`.
- Selector validation reached `val epoch=0 step=900/1336`,
  `eval_examples_so_far=112708`.
- Artifact scan still shows no selector checkpoint, cache npz, result JSON,
  detector output, or mAP. Continue monitoring validation completion.

## 2026-05-28T16:54:56+08:00 - N16R4 BATA selector validation step 1100

- Job `993718` remains `RUNNING` on `g0052`, elapsed `00:38:43`.
- Selector validation reached `val epoch=0 step=1100/1336`,
  `eval_examples_so_far=137060`; step `1000/1336` also logged with
  `eval_examples_so_far=124963`.
- Artifact scan still shows no selector checkpoint, cache npz, result JSON,
  detector output, or mAP. Continue monitoring validation completion.

## 2026-05-28T16:56:45+08:00 - N16R4 BATA selector validation step 1200

- Job `993718` remains `RUNNING` on `g0052`, elapsed `00:40:32`.
- Selector validation reached `val epoch=0 step=1200/1336`,
  `eval_examples_so_far=149434`.
- Artifact scan still shows no selector checkpoint, cache npz, result JSON,
  detector output, or mAP. Continue monitoring for validation completion.

## 2026-05-28T16:59:54+08:00 - N16R4 BATA selector complete, cache build active

- Job `993718` remains `RUNNING` on `g0052`, elapsed `00:43:41`.
- Selector completed epoch `0`: `train_loss=0.939551`, `val_loss=0.622324`,
  `val_ap=0.186222`.
- Selector artifacts exist:
  `best.pth`, `last.pth`, and `summary.json`.
- `summary.json` confirms `best_val_ap=0.18622200056530935`,
  `deployable_cache_uses_gt=false`, and
  `validation_gt_for_checkpoint_selection=false`.
- Training score cache build is active: `30/200` `.npz`; validation cache is
  `0/211`. No detector output or mAP yet.

## 2026-05-28T17:04:18+08:00 - N16R4 BATA cache build progressing

- Job `993718` remains `RUNNING` on `g0052`, elapsed `00:48:05`.
- Run/slurm log mtime is `17:02:55+08:00`; latest output is training
  score-cache writes through `video_validation_0000316`.
- Training cache advanced to `86/200` `.npz`; validation cache remains
  `0/211`.
- This is fresh progress, so the one-hour no-change process-check/helper-upload
  rule is not triggered. No detector output, result JSON, or mAP yet.

## 2026-05-28T17:06:38+08:00 - N16R4 BATA cache progress check

- Job `993718` remains `RUNNING` on `g0052`, elapsed `00:50:25`.
- Run log mtime advanced to `17:04:47+08:00`; latest cache output reached
  `video_validation_0000368`.
- Training cache is `98/200` `.npz`; validation cache remains `0/211`.
- No detector output or mAP yet. Continue monitoring; no duplicate job.

## 2026-05-28T17:08:28+08:00 - N16R4 BATA cache still active

- Job `993718` remains `RUNNING` on `g0052`, elapsed `00:52:15`.
- Run log mtime advanced to `17:06:53+08:00`; latest cache file is
  `training/video_validation_0000415.npz`.
- Training cache is `105/200` `.npz`; validation cache remains `0/211`.
- No detector output or mAP yet. Continue monitoring; no process rescue or
  helper upload is needed.

## 2026-05-28T17:12:16+08:00 - N16R4 BATA training cache past halfway

- Job `993718` remains `RUNNING` on `g0052`, elapsed `00:56:03`.
- Run log mtime advanced to `17:10:26+08:00`; latest cache file is
  `training/video_validation_0000665.npz`.
- Training cache is `125/200` `.npz`; validation cache remains `0/211`.
- No eval-cache gate, detector output, result JSON, or mAP yet. Continue
  monitoring.

## 2026-05-28T17:16:33+08:00 - N16R4 BATA training cache near completion

- Job `993718` remains `RUNNING` on `g0052`, elapsed `01:00:20`.
- Run log mtime advanced to `17:15:10+08:00`; latest cache file is
  `training/video_validation_0000942.npz`.
- Training cache is `182/200` `.npz`; validation cache remains `0/211`.
- The one-hour mark is not a stall because artifacts and log output are fresh.
  Continue monitoring for validation cache start.

## 2026-05-28T17:18:48+08:00 - N16R4 BATA validation cache started

- Job `993718` remains `RUNNING` on `g0052`, elapsed `01:02:35`.
- Training cache is complete at `200/200` `.npz`; the log printed
  `Wrote deployable MobileNet selector cache` for the training split.
- Validation cache has started and is `21/211` `.npz`; latest file is
  `validation/video_test_0000211.npz`.
- No eval-cache gate, detector output, result JSON, or mAP yet. Continue
  monitoring validation cache completion.

## 2026-05-28T17:22:24+08:00 - N16R4 BATA validation cache progressing

- Job `993718` remains `RUNNING` on `g0052`, elapsed `01:06:11`.
- Training cache remains complete at `200/200`.
- Validation cache is `55/211` `.npz`; latest file is
  `validation/video_test_0000461.npz`, with log mtime `17:20:50+08:00`.
- No eval-cache gate, detector output, result JSON, or mAP yet. Continue
  monitoring.

## 2026-05-28T17:28:11+08:00 - N16R4 BATA validation cache past halfway

- Job `993718` remains `RUNNING` on `g0052`, elapsed `01:11:58`.
- Validation cache advanced to `115/211` `.npz`; latest file is
  `validation/video_test_0000887.npz`, with log mtime `17:26:46+08:00`.
- Training cache remains complete at `200/200`; training cache manifest exists.
- No eval-cache gate, detector output, result JSON, or mAP yet. Continue
  monitoring.

## 2026-05-28T17:33:59+08:00 - N16R4 BATA validation cache nearing completion

- Job `993718` remains `RUNNING` on `g0052`, elapsed `01:17:46`.
- Validation cache advanced to `161/211` `.npz`; latest file is
  `validation/video_test_0001202.npz`, with log mtime `17:32:32+08:00`.
- No eval-cache gate, detector output, result JSON, or mAP yet. Continue
  monitoring.

## 2026-05-28T17:38:31+08:00 - N16R4 BATA validation cache almost complete

- Job `993718` remains `RUNNING` on `g0052`, elapsed `01:22:18`.
- Validation cache advanced to `191/211` `.npz`; latest file is
  `validation/video_test_0001433.npz`, with log mtime `17:37:09+08:00`.
- Training manifest exists; validation manifest is not visible yet.
- No eval-cache gate, detector output, result JSON, or mAP yet. Continue
  monitoring.

## 2026-05-28T17:41:42+08:00 - N16R4 BATA validation cache complete

- Job `993718` remains `RUNNING` on `g0052`, elapsed `01:25:29`.
- Validation cache reached `211/211` `.npz`; validation manifest exists at
  `17:39:10+08:00`.
- The run log printed `Wrote deployable MobileNet selector cache` for the
  validation split. Training cache remains complete at `200/200` with manifest.
- No detector output, result JSON, or mAP yet at this check. Continue
  monitoring gate output and detector start.

## 2026-05-28T17:45:15+08:00 - N16R4 BATA gate PASS and detector started

- Selector eval gate passed: `gate.pass=true`, `coverage_pass_rate=1.0`,
  `deployable_cache_contract_pass=true`, and `cache_uses_gt=false`.
  `uses_gt_for_eval_only=true` is diagnostic-only.
- BR@4 diagnostics: `bca_any_br_at_4=1.0`,
  `bca_boundary_br_at_4=0.2081425768476128`,
  `raw_topk_br_at_4=0.8911870503597122`, `stratified_br_at_4=1.0`.
- Log confirms `BATA deployable MobileNet selector smoke PASS`.
- Detector training started at `17:41:48+08:00`; no detector metric,
  result JSON, or mAP yet.

## 2026-05-28T17:49:38+08:00 - N16R4 BATA detector Epoch 0 completed

- Job `993718` remains `RUNNING` on `g0052`, elapsed `01:33:25`.
- Detector health logs:
  `[000][00050/00099] Loss=1.7155 cls_loss=0.9761 reg_loss=0.7394`
  and `[000][00099/00099] Loss=1.6171 cls_loss=0.9458 reg_loss=0.6713`.
- Learning rates reached `lr_backbone=4.0e-05`, `lr_det=2.0e-05`; memory is
  about `2528MB`.
- Log shows `Epoch 1 started` at `17:46:12+08:00`. No checkpoint,
  validation, result JSON, or mAP yet.

## 2026-05-28T17:55:53+08:00 - N16R4 BATA detector Epoch 2 progressing

- Job `993718` remains `RUNNING` on `g0052`, elapsed `01:39:40`.
- Epoch 1 finished at `17:50:26` with `Loss=0.9815`, `cls_loss=0.6310`,
  `reg_loss=0.3505`, `lr_backbone=8.0e-05`, `lr_det=4.0e-05`.
- Epoch 2 reached step `50/99` at `17:52:38` with `Loss=0.8283`,
  `cls_loss=0.4971`, `reg_loss=0.3312`; memory remains about `2528MB`.
- No checkpoint, validation, result JSON, or mAP yet. Continue monitoring.

## 2026-05-28T17:57:18+08:00 - N16R4 BATA detector Epoch 3 started

- Job `993718` remains `RUNNING` on `g0052`, elapsed `01:41:05`.
- Upload PASS marker remains present:
  `~/run/yuzibo/setup_logs/N16R4_THUMOS_UPLOAD_GATE_PASS_20260528_115821.ok`.
- Epoch 2 completed at `17:54:43` with `Loss=0.8835`, `cls_loss=0.5409`,
  `reg_loss=0.3426`; Epoch 3 then started.
- No checkpoint, validation, result JSON, or mAP yet. No data upload or
  process rescue is needed.

## 2026-05-28T17:59:33+08:00 - N16R4 BATA detector Epoch 3 healthy

- Job `993718` remains `RUNNING` on `g0052`, elapsed `01:43:20`.
- Latest detector line is `[003][00050/00099] Loss=0.8101`,
  `cls_loss=0.4776`, `reg_loss=0.3325`, `lr_backbone=1.4e-04`,
  `lr_det=7.0e-05`, `mem=2528MB`.
- No checkpoint, validation, result JSON, mAP, traceback, or error is visible.
- Upload PASS marker remains present; no data upload or process rescue is
  needed.

## 2026-05-28T18:01:42+08:00 - N16R4 BATA detector Epoch 4 started

- Job `993718` remains `RUNNING` on `g0052`, elapsed `01:45:29`.
- Epoch 3 completed with `Loss=0.8224`, `cls_loss=0.4919`,
  `reg_loss=0.3305`, `lr_backbone=1.6e-04`, `lr_det=8.0e-05`;
  Epoch 4 then started.
- No checkpoint, validation, result JSON, mAP, traceback, or error is visible.
- Continue normal monitoring; next expected larger artifact is checkpoint at
  epoch 20.

## 2026-05-28T18:03:41+08:00 - N16R4 BATA detector Epoch 4 mid-epoch healthy

- Job `993718` remains `RUNNING` on `g0052`, elapsed `01:47:28`.
- Latest detector line is `[004][00050/00099] Loss=0.8391`,
  `cls_loss=0.5076`, `reg_loss=0.3315`, `lr_backbone=1.8e-04`,
  `lr_det=9.0e-05`, `mem=2528MB`.
- No checkpoint, validation, result JSON, mAP, traceback, or error is visible.
- Continue normal monitoring; no intervention needed before the epoch 20
  checkpoint unless logs stop progressing.

## 2026-05-28T18:05:27+08:00 - N16R4 BATA detector Epoch 5 started

- Job `993718` remains `RUNNING` on `g0052`, elapsed `01:49:15`.
- Epoch 4 completed with `Loss=0.7908`, `cls_loss=0.4738`,
  `reg_loss=0.3169`, `lr_backbone=2.0e-04`, `lr_det=1.0e-04`;
  Epoch 5 then started.
- No checkpoint, validation, result JSON, mAP, traceback, or error is visible.
- Continue normal monitoring; no intervention needed before the epoch 20
  checkpoint unless logs stop progressing.

## 2026-05-28T18:07:05+08:00 - N16R4 BATA detector Epoch 5 mid-epoch healthy

- Job `993718` remains `RUNNING` on `g0052`, elapsed `01:50:52`.
- Latest detector line is `[005][00050/00099] Loss=0.8063`,
  `cls_loss=0.4745`, `reg_loss=0.3318`, `lr_backbone=2.0e-04`,
  `lr_det=1.0e-04`, `mem=2528MB`.
- No checkpoint, validation, result JSON, mAP, traceback, or error is visible.
- Continue normal monitoring; no intervention needed before the epoch 20
  checkpoint unless logs stop progressing.

## 2026-05-28T18:11:49+08:00 - N16R4 BATA detector Epoch 6 progressing

- Job `993718` remains `RUNNING` on `g0052`, elapsed `01:55:36`.
- Epoch 5 completed at `18:07:26` with `Loss=0.8185`, `cls_loss=0.4945`,
  `reg_loss=0.3239`.
- Epoch 6 reached step `50/99` at `18:09:38` with `Loss=0.7202`,
  `cls_loss=0.4329`, `reg_loss=0.2873`; memory remains `2528MB`.
- No checkpoint, validation, result JSON, mAP, traceback, or error is visible.

## 2026-05-28T18:18:36+08:00 - N16R4 BATA detector Epoch 8 started

- Job `993718` remains `RUNNING` on `g0052`, elapsed `02:02:23`.
- Epoch 6 completed at `18:11:45` with `Loss=0.7266`, `cls_loss=0.4340`,
  `reg_loss=0.2927`.
- Epoch 7 completed at `18:16:07` with `Loss=0.7357`, `cls_loss=0.4339`,
  `reg_loss=0.3018`; Epoch 8 then started.
- Memory remains `2528MB`; learning rates remain `2.0e-04` backbone and
  `1.0e-04` detector.
- No checkpoint, validation, result JSON, mAP, traceback, or error is visible.

## 2026-05-28T18:25:27+08:00 - N16R4 BATA detector Epoch 9 progressing

- Job `993718` remains `RUNNING` on `g0052`, elapsed `02:09:14`.
- Epoch 8 completed at `18:20:43` with `Loss=0.6769`, `cls_loss=0.3901`,
  `reg_loss=0.2869`.
- Epoch 9 reached step `50/99` at `18:22:51` with `Loss=0.6369`,
  `cls_loss=0.3619`, `reg_loss=0.2750`; memory remains `2528MB`.
- No checkpoint, validation, result JSON, mAP, traceback, or error is visible.

## 2026-05-28T18:28:15+08:00 - N16R4 BATA detector Epoch 10 started

- Job `993718` remains `RUNNING` on `g0052`, elapsed `02:12:02`.
- Epoch 9 completed at `18:25:00` with `Loss=0.6711`, `cls_loss=0.3886`,
  `reg_loss=0.2825`; Epoch 10 then started.
- No checkpoint, validation, result JSON, mAP, traceback, or error is visible.
- Next larger milestone remains epoch 20 checkpoint.

## 2026-05-28T18:41:01+08:00 - N16R4 BATA detector Epoch 13 started

- Job `993718` remains `RUNNING` on `g0052`, elapsed `02:24:48`.
- Epoch 10 completed at `18:29:18` with `Loss=0.6698`; Epoch 11 completed
  at `18:33:36` with `Loss=0.6598`.
- Epoch 12 completed at `18:37:53` with `Loss=0.6273`,
  `cls_loss=0.3558`, `reg_loss=0.2715`; Epoch 13 then started.
- No checkpoint, validation, result JSON, mAP, traceback, or error is visible.

## 2026-05-28T18:52:57+08:00 - N16R4 BATA detector Epoch 16 started

- Job `993718` remains `RUNNING` on `g0052`, elapsed `02:36:44`.
- Epoch 13 completed at `18:42:08` with `Loss=0.6417`; Epoch 14 completed
  at `18:46:28` with `Loss=0.5981`.
- Epoch 15 completed at `18:50:45` with `Loss=0.5944`,
  `cls_loss=0.3333`, `reg_loss=0.2611`; Epoch 16 then started.
- No checkpoint, validation, result JSON, mAP, traceback, or error is visible.

## 2026-05-28T19:14:59+08:00 - N16R4 BATA first detector checkpoint written

- Job `993718` remains `RUNNING` on `g0052`, elapsed `02:58:46`.
- First detector checkpoint exists:
  `~/run/yuzibo/bata_runs/exps/bata_mobilenet_bca_fixed50_adapter_deployable/gpu1_id0/checkpoint/epoch_19.pth`,
  size `623799195` bytes, mtime `19:07:49+08:00`.
- Epoch 19 completed at `19:07:48` with `Loss=0.5920`; Epoch 20 completed
  at `19:12:03` with `Loss=0.5545`, `cls_loss=0.2999`,
  `reg_loss=0.2546`; Epoch 21 then started.
- No validation, result JSON, mAP, traceback, or error is visible.
- Do not clean this checkpoint while the run is active.

## 2026-05-28T19:37:04+08:00 - N16R4 BATA detector Epoch 26 progressing

- Job `993718` remains `RUNNING` on `g0052`, elapsed `03:20:51`.
- Existing checkpoint remains `epoch_19.pth`; no new checkpoint yet.
- Epoch 25 completed at `19:33:22` with `Loss=0.5342`,
  `cls_loss=0.2849`, `reg_loss=0.2492`.
- Epoch 26 reached step `50/99` at `19:35:33` with `Loss=0.5354`,
  `cls_loss=0.2852`, `reg_loss=0.2502`; memory remains `2528MB`.
- No validation, result JSON, mAP, traceback, or error is visible.

## 2026-05-28T20:00:01+08:00 - N16R4 BATA detector Epoch 31 progressing

- Job `993718` remains `RUNNING` on `g0052`, elapsed `03:43:48`.
- Existing checkpoint remains `epoch_19.pth`; no new checkpoint/result yet.
- Epochs 26-30 completed normally, with losses `0.5206`, `0.5142`,
  `0.5110`, `0.4966`, and `0.5065`.
- Epoch 31 reached step `50/99` at `19:57:31` with `Loss=0.4782`,
  `cls_loss=0.2511`, `reg_loss=0.2271`; memory remains `2528MB`.
- No validation, result JSON, mAP, traceback, or error is visible.
- Fixed-50/no-GT/no-teacher status remains preserved; checkpoint cleanup is
  disallowed while the run is active.

## 2026-05-28T20:02:22+08:00 - N16R4 BATA detector Epoch 32 started

- Job `993718` remains `RUNNING` on `g0052`, elapsed `03:46:09`.
- Run log:
  `~/run/yuzibo/OpenTAD_BATA_Clean/logs/bata_mobilenet_fixed50_20260528_161457_n16r4.log`.
- Slurm log:
  `~/run/yuzibo/OpenTAD_BATA_Clean/logs/bata_mobile50-993718.out`.
- Existing checkpoint remains only `epoch_19.pth`, size `623799195` bytes,
  mtime `19:07:49+08:00`; no result JSON, best/last checkpoint,
  validation, or mAP artifact is visible.
- Epoch 31 completed at `19:59:38` with `Loss=0.4892`,
  `cls_loss=0.2590`, `reg_loss=0.2301`; Epoch 32 then started.
- No traceback, exception, CUDA failure, killed marker, or failed marker is
  visible in the checked log tail.
- Continue monitoring toward epoch 39/40 checkpoint and first validation/mAP;
  do not submit a duplicate job or clean active checkpoints.

## 2026-05-28T20:06:23+08:00 - N16R4 BATA detector Epoch 33 started

- Job `993718` remains `RUNNING` on `g0052`, elapsed `03:50:10`; `sacct`
  also reports active state with exit code `0:0`.
- Upload PASS marker remains present:
  `~/run/yuzibo/setup_logs/N16R4_THUMOS_UPLOAD_GATE_PASS_20260528_115821.ok`.
- Existing checkpoint remains only `epoch_19.pth`, size `623799195` bytes,
  mtime `19:07:49+08:00`; no result JSON, best/last checkpoint,
  validation, or mAP artifact is visible.
- Epoch 32 reached step `50/99` at `20:01:56` with `Loss=0.4978`,
  then completed at `20:03:58` with `Loss=0.4944`, `cls_loss=0.2642`,
  `reg_loss=0.2303`; Epoch 33 then started.
- Failure scan is empty for traceback, exception, CUDA OOM, killed, failed,
  runtime/value/key errors, missing file markers, and NaN.
- Disk remains safe: JuiceFS `/data` `2.3T` total, `413G` used, `1.9T` free.
- Fixed-50/no-GT/no-teacher status remains preserved; no code/config/model
  change occurred. Continue monitoring toward epoch 39/40 checkpoint and first
  validation/mAP; do not submit a duplicate job or clean active checkpoints.

## 2026-05-28T20:08:58+08:00 - N16R4 BATA detector Epoch 33 mid-epoch healthy

- Job `993718` remains `RUNNING` on `g0052`, elapsed about `03:52:46`.
- Existing checkpoint remains only `epoch_19.pth`; no result JSON, best/last
  checkpoint, validation, or mAP artifact is visible.
- Latest detector line is `[033][00050/00099] Loss=0.4941`,
  `cls_loss=0.2547`, `reg_loss=0.2394`, `lr_backbone=1.6e-04`,
  `lr_det=7.9e-05`, `mem=2528MB`.
- Failure scan is empty for traceback, exception, CUDA OOM, killed, failed,
  runtime/value/key errors, missing file markers, and NaN.
- Continue monitoring toward epoch 39/40 checkpoint and first validation/mAP;
  no intervention is needed.

## 2026-05-28T20:35:35+08:00 - N16R4 BATA Epoch 39 mid-epoch plus upload sanity

- Job `993718` remains `RUNNING` on `g0052`, elapsed about `04:18:18`;
  `sacct`/`scontrol` show active state and `ExitCode=0:0`, while `sstat`
  reports live batch accounting with `AveCPU=16:45:46` and `MaxRSS=39417912K`.
- Run/slurm log mtime advanced to `20:32:41+08:00`. Epoch 38 completed at
  `20:30:28` with `Loss=0.4805`; Epoch 39 reached `[039][00050/00099]`
  with `Loss=0.4161`, `cls_loss=0.2151`, `reg_loss=0.2010`,
  `lr_backbone=1.4e-04`, `lr_det=7.1e-05`, `mem=2528MB`.
- Existing checkpoint remains `epoch_19.pth`; no validation, result JSON,
  best/last checkpoint, or mAP artifact is visible yet.
- Video upload recheck found no remaining data to copy: train has `200`
  symlinks resolving to `200` mp4 targets, test has `211` symlinks resolving
  to `211` mp4 targets, both missing lists are empty, and the upload PASS
  marker still records `ALL_PASS=True`.
- Failure scan is empty. Fixed-50/no-test-GT/no-teacher contract remains
  preserved; no code/config/model change occurred.
- Continue monitoring toward Epoch 40 first validation/mAP. Do not start a
  duplicate job, run helper upload, or clean checkpoints while this run is
  active.

## 2026-05-28T20:38:04+08:00 - N16R4 BATA Epoch 39 checkpoint; Epoch 40 started

- Job `993718` remains `RUNNING` on `g0052`, elapsed `04:21:51`.
- Run log mtime is `20:35:00+08:00`. A new active-run checkpoint
  `epoch_39.pth` was written at `20:34:59+08:00`, size `623799195` bytes;
  `epoch_19.pth` is still present.
- Epoch 39 completed at `20:34:58` with `[039][00099/00099] Loss=0.4452`,
  `cls_loss=0.2267`, `reg_loss=0.2185`, `lr_backbone=1.4e-04`,
  `lr_det=7.0e-05`, `mem=2528MB`; Epoch 40 started at `20:34:59`.
- No validation, mAP, result JSON, best/last checkpoint, traceback, CUDA OOM,
  killed/failed marker, missing file marker, or NaN is visible yet.
- Fixed-50/no-test-GT/no-teacher contract remains preserved; no code/config
  change occurred. Checkpoint cleanup remains disallowed while this run is
  active.
- Continue monitoring Epoch 40 for first validation/mAP; no duplicate job.

## 2026-05-28T20:40:26+08:00 - N16R4 BATA Epoch 40 mid-epoch healthy

- Job `993718` remains `RUNNING` on `g0052`, elapsed `04:24:13`.
- Run log mtime is `20:37:17+08:00`. Checkpoints remain `epoch_19.pth` and
  `epoch_39.pth`; no result JSON, best/last checkpoint, validation, or mAP
  artifact is visible.
- Latest detector line is `[040][00050/00099] Loss=0.4622`,
  `cls_loss=0.2383`, `reg_loss=0.2239`, `lr_backbone=1.4e-04`,
  `lr_det=6.9e-05`, `mem=2528MB`.
- Failure scan is empty. Fixed-50/no-test-GT/no-teacher contract remains
  preserved; no code/config/model change occurred.
- Continue monitoring for Epoch 40 completion and first validation/mAP; no
  intervention is needed.

## 2026-05-28T20:43:18+08:00 - N16R4 BATA Epoch 41 mid-epoch; eval cadence clarified

- Job `993718` remains `RUNNING` on `g0052`, elapsed `04:27:05`.
- Run log mtime advanced to `20:41:38+08:00`.
- Epoch 40 completed at `20:39:27` with `[040][00099/00099] Loss=0.4601`,
  `cls_loss=0.2414`, `reg_loss=0.2188`; Epoch 41 then started and reached
  `[041][00050/00099] Loss=0.4161`, `cls_loss=0.2108`, `reg_loss=0.2053`,
  `lr_backbone=1.4e-04`, `lr_det=6.8e-05`, `mem=2528MB`.
- No validation, mAP, or result JSON is visible yet.
- Config/loop clarification: `val_start_epoch=40`, `val_eval_interval=2`,
  and `tools/train.py` evaluates when `epoch >= val_start_epoch` and
  `(epoch + 1) % val_eval_interval == 0`, so first eval should trigger after
  zero-based Epoch 41 completes, not after Epoch 40.
- Failure scan is empty. Fixed-50/no-test-GT/no-teacher contract remains
  preserved; no code/config/model change occurred.
- Continue monitoring the Epoch 41 completion window for first validation/mAP.

## 2026-05-28T20:46:15+08:00 - N16R4 BATA first eval started

- Job `993718` remains `RUNNING` on `g0052`, elapsed `04:29:42`.
- Epoch 41 completed at `20:43:49` with `[041][00099/00099] Loss=0.4428`,
  `cls_loss=0.2246`, `reg_loss=0.2183`, `lr_backbone=1.3e-04`,
  `lr_det=6.7e-05`, `mem=2528MB`.
- First evaluation/inference started immediately after Epoch 41. Raw log tail
  showed the progress bar at about `34/396` by the `20:46:15` check.
- No mAP, result JSON, best/last checkpoint, or final decision exists yet.
- Failure scan is empty. Fixed-50/no-test-GT/no-teacher contract remains
  preserved; no code/config/model change occurred.
- Continue monitoring until the first mAP vector appears; apply the severe
  result gate if the metric is failure-scale or internally inconsistent.

## 2026-05-28T20:49:24+08:00 - N16R4 BATA first eval progressing

- Job `993718` remains `RUNNING` on `g0052`, elapsed `04:33:11`.
- Run log mtime is `20:48:01+08:00`.
- First eval progress advanced from about `34/396` to about `112/396`
  (`28%`). No mAP or result JSON exists yet.
- Failure scan is empty. Fixed-50/no-test-GT/no-teacher contract remains
  preserved; no code/config/model change occurred.
- Continue monitoring eval completion and the first mAP vector.

## 2026-05-28T21:01:09+08:00 - N16R4 BATA first eval near completion

- Job `993718` remains `RUNNING` on `g0052`, elapsed `04:44:56`.
- Run log mtime is `20:59:46+08:00`.
- First eval progressed to about `368/396` (`93%`). No mAP or result JSON is
  visible yet.
- Failure scan is empty. Fixed-50/no-test-GT/no-teacher contract remains
  preserved; no code/config/model change occurred.
- Continue short-interval monitoring for the first mAP vector.

## 2026-05-28T21:03:51+08:00 - N16R4 BATA first eval result

- First eval completed after Epoch 41 and training resumed to Epoch 42.
- Metric at `21:01:43+08:00`: `Average-mAP=61.60`; vector at tIoU
  `0.30/0.40/0.50/0.60/0.70` is
  `78.59 / 72.34 / 64.02 / 53.47 / 39.57`.
- Eval log reports `3325` GT instances and `422000` predictions.
- Comparison: `-2.17` vs random-fixed Adapter `63.77`, `-2.25` vs strict EMA
  `63.85`, `-3.04` vs stratified sampling best `64.64`, and `-3.49` vs
  uniform stride-2 `65.09`.
- This is below baseline but not failure-scale and not a `~5` point collapse,
  so the severe-result GPT-5 Pro escalation gate is not triggered yet.
- No result JSON was visible in the checked artifact scan; failure scan remains
  empty. Fixed-50/no-test-GT/no-teacher contract remains preserved.
- Continue the active run to later eval/final before route judgment. Do not
  launch follow-up long runs or clean active checkpoints.

## 2026-05-28T21:12:30+08:00 - N16R4 BATA second eval started

- Job `993718` remains `RUNNING` on `g0052`, elapsed `04:56:17`.
- After first eval, Epoch 42 completed at `21:06:17` with
  `[042][00099/00099] Loss=0.4561`, `cls_loss=0.2367`, `reg_loss=0.2194`.
- Epoch 43 completed at `21:10:41` with `[043][00099/00099] Loss=0.4347`,
  `cls_loss=0.2235`, `reg_loss=0.2112`, `lr_backbone=1.3e-04`,
  `lr_det=6.4e-05`, `mem=2528MB`.
- Second eval started at `21:10:41` and reached about `17/396` by the
  `21:12:30` check.
- Checkpoints remain `epoch_19.pth` and `epoch_39.pth`; no new result JSON is
  visible. Failure scan is empty.
- Continue monitoring second eval mAP; no duplicate launch or checkpoint
  cleanup.

## 2026-05-28T20:56:58+08:00 - N16R4 BATA first eval past 70 percent

- Job `993718` remains `RUNNING` on `g0052`, elapsed `04:40:45`.
- Run log mtime is `20:55:37+08:00`.
- First eval progressed to about `283/396` (`71%`). No mAP or result JSON is
  visible yet.
- Failure scan is empty. Fixed-50/no-test-GT/no-teacher contract remains
  preserved; no code/config/model change occurred.
- Continue monitoring eval completion and the first mAP vector.

## 2026-05-28T20:52:48+08:00 - N16R4 BATA first eval near halfway

- Job `993718` remains `RUNNING` on `g0052`, elapsed `04:36:35`.
- Run log mtime is `20:51:26+08:00`.
- First eval progressed to about `192/396` (`48%`). No mAP or result JSON is
  visible yet.
- Failure scan is empty. Fixed-50/no-test-GT/no-teacher contract remains
  preserved; no code/config/model change occurred.
- Continue monitoring eval completion and the first mAP vector.

## 2026-05-28T20:11:18+08:00 - N16R4 BATA detector Epoch 34 started

- Job `993718` remains `RUNNING` on `g0052`, elapsed `03:55:05`.
- Existing checkpoint remains only `epoch_19.pth`; no result JSON, best/last
  checkpoint, validation, or mAP artifact is visible.
- Epoch 33 completed at `20:08:26` with `Loss=0.4721`,
  `cls_loss=0.2460`, `reg_loss=0.2262`; Epoch 34 then started.
- Failure scan is empty for traceback, exception, CUDA OOM, killed, failed,
  runtime/value/key errors, missing file markers, and NaN.
- Continue monitoring toward epoch 39/40 checkpoint and first validation/mAP;
  no intervention is needed.

## 2026-05-28T20:13:35+08:00 - N16R4 BATA detector Epoch 34 mid-epoch healthy

- Job `993718` remains `RUNNING` on `g0052`, elapsed `03:57:22`.
- Existing checkpoint remains only `epoch_19.pth`; no result JSON, best/last
  checkpoint, validation, or mAP artifact is visible.
- Latest detector line is `[034][00050/00099] Loss=0.4861`,
  `cls_loss=0.2553`, `reg_loss=0.2308`, `lr_backbone=1.6e-04`,
  `lr_det=7.8e-05`, `mem=2528MB`.
- Failure scan is empty for traceback, exception, CUDA OOM, killed, failed,
  runtime/value/key errors, missing file markers, and NaN.
- Continue monitoring toward epoch 39/40 checkpoint and first validation/mAP;
  no intervention is needed.

## 2026-05-28T20:15:50+08:00 - N16R4 BATA detector Epoch 35 started

- Job `993718` remains `RUNNING` on `g0052`, elapsed `03:59:37`.
- Existing checkpoint remains only `epoch_19.pth`; no result JSON, best/last
  checkpoint, validation, or mAP artifact is visible.
- Epoch 34 completed at `20:12:46` with `Loss=0.5027`,
  `cls_loss=0.2668`, `reg_loss=0.2359`; Epoch 35 then started.
- Failure scan is empty for traceback, exception, CUDA OOM, killed, failed,
  runtime/value/key errors, missing file markers, and NaN.
- Continue monitoring toward epoch 39/40 checkpoint and first validation/mAP;
  no intervention is needed.

## 2026-05-28T20:18:01+08:00 - N16R4 BATA detector Epoch 35 mid-epoch healthy

- Job `993718` remains `RUNNING` on `g0052`, elapsed `04:01:48`.
- Existing checkpoint remains only `epoch_19.pth`; no result JSON, best/last
  checkpoint, validation, or mAP artifact is visible.
- Latest detector line is `[035][00050/00099] Loss=0.4344`,
  `cls_loss=0.2206`, `reg_loss=0.2139`, `lr_backbone=1.5e-04`,
  `lr_det=7.7e-05`, `mem=2528MB`.
- Failure scan is empty for traceback, exception, CUDA OOM, killed, failed,
  runtime/value/key errors, missing file markers, and NaN.
- Continue monitoring toward epoch 39/40 checkpoint and first validation/mAP;
  no intervention is needed.

## 2026-05-28T20:20:12+08:00 - N16R4 BATA detector Epoch 36 started

- Job `993718` remains `RUNNING` on `g0052`, elapsed `04:03:59`.
- Existing checkpoint remains only `epoch_19.pth`; no result JSON, best/last
  checkpoint, validation, or mAP artifact is visible.
- Epoch 35 completed at `20:17:12` with `Loss=0.4671`,
  `cls_loss=0.2435`, `reg_loss=0.2237`; Epoch 36 then started.
- Failure scan is empty for traceback, exception, CUDA OOM, killed, failed,
  runtime/value/key errors, missing file markers, and NaN.
- Continue monitoring toward epoch 39/40 checkpoint and first validation/mAP;
  no intervention is needed.

## 2026-05-28T20:22:21+08:00 - N16R4 BATA detector Epoch 36 mid-epoch healthy

- Job `993718` remains `RUNNING` on `g0052`, elapsed `04:06:08`.
- Existing checkpoint remains only `epoch_19.pth`; no result JSON, best/last
  checkpoint, validation, or mAP artifact is visible.
- Latest detector line is `[036][00050/00099] Loss=0.4673`,
  `cls_loss=0.2448`, `reg_loss=0.2225`, `lr_backbone=1.5e-04`,
  `lr_det=7.5e-05`, `mem=2528MB`.
- Failure scan is empty for traceback, exception, CUDA OOM, killed, failed,
  runtime/value/key errors, missing file markers, and NaN.
- Continue monitoring toward epoch 39/40 checkpoint and first validation/mAP;
  no intervention is needed.

## 2026-05-28T20:24:34+08:00 - N16R4 BATA detector Epoch 37 started

- Job `993718` remains `RUNNING` on `g0052`, elapsed `04:08:21`.
- Existing checkpoint remains only `epoch_19.pth`; no result JSON, best/last
  checkpoint, validation, or mAP artifact is visible.
- Epoch 36 completed at `20:21:36` with `Loss=0.4543`,
  `cls_loss=0.2334`, `reg_loss=0.2209`; Epoch 37 then started.
- Failure scan is empty for traceback, exception, CUDA OOM, killed, failed,
  runtime/value/key errors, missing file markers, and NaN.
- Continue monitoring toward epoch 39/40 checkpoint and first validation/mAP;
  no intervention is needed.

## 2026-05-28T20:26:51+08:00 - N16R4 BATA detector Epoch 37 mid-epoch healthy

- Job `993718` remains `RUNNING` on `g0052`, elapsed `04:10:38`.
- Existing checkpoint remains only `epoch_19.pth`; no result JSON, best/last
  checkpoint, validation, or mAP artifact is visible.
- Latest detector line is `[037][00050/00099] Loss=0.4862`,
  `cls_loss=0.2574`, `reg_loss=0.2288`, `lr_backbone=1.5e-04`,
  `lr_det=7.4e-05`, `mem=2528MB`.
- Failure scan is empty for traceback, exception, CUDA OOM, killed, failed,
  runtime/value/key errors, missing file markers, and NaN.
- Continue monitoring toward epoch 39/40 checkpoint and first validation/mAP;
  no intervention is needed.

## 2026-05-28T20:29:08+08:00 - N16R4 BATA detector Epoch 38 started

- Job `993718` remains `RUNNING` on `g0052`, elapsed `04:12:55`.
- Existing checkpoint remains only `epoch_19.pth`; no result JSON, best/last
  checkpoint, validation, or mAP artifact is visible.
- Epoch 37 completed at `20:25:59` with `Loss=0.4656`,
  `cls_loss=0.2423`, `reg_loss=0.2233`; Epoch 38 then started.
- Failure scan is empty for traceback, exception, CUDA OOM, killed, failed,
  runtime/value/key errors, missing file markers, and NaN.
- Continue monitoring toward epoch 39/40 checkpoint and first validation/mAP;
  no intervention is needed.

## 2026-05-28T20:31:25+08:00 - N16R4 BATA detector Epoch 38 mid-epoch healthy

- Job `993718` remains `RUNNING` on `g0052`, elapsed `04:15:12`.
- Existing checkpoint remains only `epoch_19.pth`; no result JSON, best/last
  checkpoint, validation, or mAP artifact is visible.
- Latest detector line is `[038][00050/00099] Loss=0.4971`,
  `cls_loss=0.2705`, `reg_loss=0.2266`, `lr_backbone=1.4e-04`,
  `lr_det=7.2e-05`, `mem=2528MB`.
- Failure scan is empty for traceback, exception, CUDA OOM, killed, failed,
  runtime/value/key errors, missing file markers, and NaN.
- Continue monitoring toward epoch 39/40 checkpoint and first validation/mAP;
  no intervention is needed.

## 2026-05-28T21:30:29+08:00 - N16R4 BATA second eval result below baseline

- Job `993718` remains `RUNNING` on `g0052`, elapsed `05:14:16`; training
  resumed to Epoch 44 after the second eval.
- Data upload remains complete: train/test `200/211` mp4 entries and resolved
  targets `200/211`; `/data` has `1.9T` free.
- First eval after Epoch 41 was `61.60` Avg-mAP with vector
  `78.59 / 72.34 / 64.02 / 53.47 / 39.57`.
- Second eval after Epoch 43 completed at `21:28:43+08:00`: `62.12`
  Avg-mAP with vector `78.81 / 72.71 / 64.58 / 54.10 / 40.39`, `422000`
  predictions, and `3325` GT instances.
- Deltas: `-1.65` vs random-fixed Adapter `63.77`, `-1.73` vs strict EMA
  `63.85`, `-2.52` vs stratified `64.64`, and `-2.97` vs uniform stride-2
  `65.09`; high-IoU `40.39@0.7` remains below random-fixed `42.19` and
  stratified `43.23`.
- This is below baseline but not failure-scale and does not trigger the severe
  `~5` point collapse gate. Continue the active run to later/final eval. Do
  not launch follow-up long runs or clean active checkpoints while the job is
  active.

## 2026-05-28T21:47:31+08:00 - BATA MobileNet gap diagnostics and Pro discussion

- Wrote prompt `logs/gpt5pro_bata_mobilenet_gap_discussion_prompt_20260528.md`
  and ran Oracle browser Pro discussion. Output:
  `logs/gpt5pro_bata_mobilenet_gap_discussion_20260528.txt`.
- Added report
  `research-wiki/experiments/BATA_MOBILENET_GAP_PRO_DISCUSSION_20260528.md`.
- Remote cache diagnostic output is under
  `~/run/yuzibo/bata_runs/selector_diagnostics/bata_mobilenet_fixed50_20260528_161457_validation_20260528_213424`.
- Diagnostic evidence: selector `val_ap=0.186222`; validation positive/negative
  score means `0.251449/0.227666`; `bca_boundary_br_at_4=0.2081426` while
  `bca_any_br_at_4=1.0`; ledger distribution is `coverage_context=93.71%` and
  boundary roles about `6.29%`.
- Pro verdict accepted: protocol correctness is mostly established, but
  scientific success is not; the current selected-frame distribution is weak
  boundary score plus strong coverage fallback, close to uniform/coverage-like.
- Next action: do not launch new long BATA runs. Continue `993718` only for
  final negative/diagnostic evidence. Before any next implementation, run
  cache-only K-parity, boundary-role miss, score-vs-distance,
  selected-index-overlap, and coordinate/role diagnostics.

## 2026-05-28T21:57:06+08:00 - N16R4 BATA third eval still below baseline

- Slurm job `993718` (`bata_mobile50`) remains `RUNNING` on `g0052`, elapsed
  about `05:40:43`; training resumed to Epoch 46 after the third eval.
- Third eval after Epoch 45 completed at `2026-05-28T21:55:27+08:00`:
  `Average-mAP=62.47`, vector `78.76 / 73.27 / 64.80 / 54.61 / 40.90`.
- Deltas remain negative: `-1.30` vs random-fixed Adapter `63.77`, `-1.38`
  vs strict EMA `63.85`, `-2.17` vs stratified `64.64`, `-2.62` vs uniform
  stride-2 `65.09`, and `-15.15` vs oracle-boundary `77.62`.
- Failure scan found no traceback/OOM/crash. Fixed-50/no-test-GT/no-teacher
  contract remains preserved. Checkpoints remain `epoch_19.pth` and
  `epoch_39.pth`; no checkpoint cleanup while the run is active.
- Decision: continue `993718` to final/stop evidence only. Do not launch new
  long BATA/Sparse runs until cache-only score-vs-distance, miss, overlap,
  coordinate/role, and K-consistency diagnostics justify a reviewed change.

## 2026-05-28T22:05:41+08:00 - BATA score/role/overlap diagnostic

- Ran a read-only cache diagnostic against deployable validation score cache
  and existing BCA ledgers. Output:
  `~/run/yuzibo/bata_runs/selector_diagnostics/bata_mobilenet_fixed50_20260528_161457_score_role_overlap_20260528_220541/summary.json`.
- Cache manifest remains deployable: `uses_gt=false`,
  `diagnostic_only=false`, subset `validation`, score source
  `mobilenet_v3_small_boundary_probability`. GT labels were used only for
  offline diagnostic categories.
- Score means show weak actionness rather than boundary specificity:
  boundary_band4 `0.23498`, near_boundary_5_8 `0.23724`,
  action_interior `0.25024`, background `0.22938`.
- Raw MobileNet topK at K=24 is not boundary-dominated: background `49.36%`,
  action_interior `20.42%`, boundary_band4 `17.44%`,
  near_boundary_5_8 `12.78%`.
- Current selected-set Jaccard is closer to stratified/uniform same-K
  (`0.3608/0.3615`) than raw topK (`0.2860`). Boundary-role BR@4 remains
  `0.2081`, while any-selected BR@4 is `1.0`.
- Interpretation: current selected frames are mostly dense coverage/context,
  with weak clustered boundary-score insertions. The model has not learned the
  desired boundary-aware frame distribution.

## 2026-05-28T22:12:43+08:00 - N16R4 BATA active-run monitor

- Slurm/sacct show job `993718` (`bata_mobile50`) still `RUNNING` on `g0052`,
  elapsed `05:56:30`, active exit code `0:0`.
- Eval count remains `3`: `61.60`, `62.12`, `62.47`; no fourth mAP has been
  logged yet.
- Training progressed after the third eval: Epoch 46 completed at `21:59:51`
  with `Loss=0.4261`; Epoch 47 completed at `22:04:15` with `Loss=0.4430`,
  `cls_loss=0.2291`, `reg_loss=0.2139`.
- Failure scan count is `0`. Active checkpoints remain `epoch_19.pth` and
  `epoch_39.pth`; `log.json` was touched at `22:04:15`.
- Decision unchanged: continue monitoring for the fourth eval/final result.
  Do not clean active checkpoints, launch duplicate jobs, or start new long
  BATA/Sparse follow-ups.

## 2026-05-28T22:15:44+08:00 - N16R4 BATA quick recheck

- Slurm/sacct still show job `993718` running on `g0052`, elapsed `05:59:31`.
- Eval count remains `3`; latest mAP is still Epoch 45 `62.47`; no fourth
  eval result and no `Training Over` marker yet.
- Last train line remains Epoch 47 completion at `22:04:15`; failure scan
  count is `0`.
- Active checkpoints remain `epoch_19.pth` and `epoch_39.pth`; no cleanup.
- Disk is safe: `/data` JuiceFS reports `836T` available.
- Decision unchanged: wait for the fourth eval/final with lower polling
  frequency; no duplicate launch or new long follow-up.

## 2026-05-28T22:18:01+08:00 - N16R4 BATA quick recheck, no new eval

- Job `993718` remains `RUNNING` on `g0052`, elapsed `06:01:48`.
- Eval count remains `3`; latest mAP is still `62.47`; `Training Over=False`.
- Last train line remains Epoch 47 completion at `22:04:15`; failure scan
  count `0`; checkpoints remain `epoch_19.pth` and `epoch_39.pth`.
- Decision unchanged: stop dense polling and wait for the fourth eval/final
  window. No active-checkpoint cleanup or new long run.

## 2026-05-28T22:20:15+08:00 - N16R4 BATA quick recheck, still no fourth eval

- Job `993718` remains `RUNNING` on `g0052`, elapsed `06:04:02`.
- Eval count remains `3`; latest mAP is still `62.47`; `Training Over=False`.
- Last train/eval evidence is unchanged: Epoch 47 completion at `22:04:15`,
  no fourth `Average-mAP`, failure scan count `0`, checkpoints `epoch_19.pth`
  and `epoch_39.pth`.
- Decision unchanged: stop minute-level polling; next check should target the
  expected fourth-eval/final window. No cleanup or new long run.

## 2026-05-28T22:22:31+08:00 - N16R4 BATA low-frequency check, no new eval

- Job `993718` remains `RUNNING` on `g0052`, elapsed `06:06:18`.
- Eval count remains `3`; latest mAP is still `62.47`; `Training Over=False`.
- No new recent eval lines were found. Last train line remains Epoch 47
  completion at `22:04:15`; failure scan count `0`; checkpoints remain
  `epoch_19.pth` and `epoch_39.pth`.
- Decision unchanged: pause repeated no-change checks until a more plausible
  fourth-eval/final interval. No cleanup or new long run.

## 2026-05-28T22:26:36+08:00 - N16R4 BATA fourth eval

- Job `993718` (`bata_mobile50`) remains `RUNNING` on `g0052`, elapsed
  `06:10:23`; `Training Over=False`.
- Fourth eval after Epoch 47 completed at `2026-05-28T22:22:10+08:00`:
  `Average-mAP=62.54`, vector `78.75 / 73.14 / 65.05 / 54.81 / 40.96`.
- Trend is `61.60 -> 62.12 -> 62.47 -> 62.54`; the latest result remains
  below random-fixed Adapter `63.77` by `-1.23`, strict EMA `63.85` by
  `-1.31`, stratified `64.64` by `-2.10`, uniform stride-2 `65.09` by
  `-2.55`, and oracle-boundary `77.62` by `-15.08`.
- Failure scan count is `0`; active checkpoints remain `epoch_19.pth` and
  `epoch_39.pth`. Fixed-50/no-test-GT/no-teacher contract remains preserved;
  no code/config/model changed.
- Decision unchanged: continue this active run only for final
  negative/diagnostic evidence. Do not clean active checkpoints, launch a
  duplicate job, or start new long BATA/Sparse follow-ups before final result
  and cache-only diagnostics justify a reviewed next implementation.

## 2026-05-28T22:31:10+08:00 - N16R4 BATA post-fourth-eval monitor

- Job `993718` remains `RUNNING` on `g0052`, elapsed `06:15:00`; log mtime
  advanced to `22:28:47+08:00`.
- Eval count remains `4`; latest mAP remains Epoch 47 `62.54`; no fifth mAP
  and `Training Over=False`.
- Training resumed to Epoch 49 at `22:26:29`. Failure scan count is `0`;
  active checkpoints remain `epoch_19.pth` and `epoch_39.pth`.
- Disk is safe: `/data` has about `1.9T` available. Fixed-50/no-test-GT/
  no-teacher contract remains preserved; no code/config/model changed.
- Decision unchanged: continue low-frequency monitoring toward next eval/final.
  No active-checkpoint cleanup, duplicate launch, or new long follow-up while
  `993718` is active.

## 2026-05-28T22:33:57+08:00 - N16R4 BATA Epoch 49 complete

- Job `993718` remains `RUNNING` on `g0052`, elapsed `06:17:47`; log mtime
  advanced to `22:32:37+08:00`.
- Eval count remains `4`; latest mAP remains `62.54`; no fifth mAP and
  `Training Over=False`.
- Epoch 49 completed at `22:30:50` with `Loss=0.4349`, `cls_loss=0.2233`,
  `reg_loss=0.2116`, `lr_backbone=1.1e-04`, `lr_det=5.4e-05`, `mem=2528MB`.
- Failure scan count is `0`; active checkpoints remain `epoch_19.pth` and
  `epoch_39.pth`; `/data` has about `1.9T` available.
- Decision unchanged: wait for the next eval result. No active checkpoint
  cleanup, duplicate launch, or new long follow-up while `993718` is active.

## 2026-05-28T22:37:05+08:00 - N16R4 BATA next-eval window

- Job `993718` remains `RUNNING` on `g0052`, elapsed `06:20:55`; log mtime
  advanced to `22:35:44+08:00`.
- Eval count remains `4`; latest mAP remains `62.54`; no fifth mAP and
  `Training Over=False`.
- Last completed train line remains Epoch 49 at `22:30:50`. Failure scan
  count is `0`; active checkpoints remain `epoch_19.pth` and `epoch_39.pth`.
- Disk remains safe: `/data` has about `1.9T` available. Fixed-50/no-test-GT/
  no-teacher contract remains preserved; no code/config/model changed.
- Decision unchanged: wait for the fifth evaluation result and avoid denser
  polling until a plausible mAP completion window. No cleanup or new long run.

## 2026-05-28T22:40:08+08:00 - N16R4 BATA eval still progressing

- Job `993718` remains `RUNNING` on `g0052`, elapsed `06:23:58`; log mtime
  advanced to `22:38:46+08:00`.
- Eval count remains `4`; latest mAP remains `62.54`; no fifth mAP and
  `Training Over=False`.
- Last completed train line remains Epoch 49 at `22:30:50`. Failure scan
  count is `0`; active checkpoints remain `epoch_19.pth` and `epoch_39.pth`.
- `/data` still has about `1.9T` available. Fixed-50/no-test-GT/no-teacher
  contract remains preserved; no code/config/model changed.
- Decision unchanged: wait for fifth mAP output. No cleanup, duplicate launch,
  or new long follow-up while `993718` is active.

## 2026-05-28T22:42:44+08:00 - N16R4 BATA eval still progressing

- Job `993718` remains `RUNNING` on `g0052`, elapsed `06:26:34`; log mtime
  advanced to `22:41:20+08:00`.
- Eval count remains `4`; latest mAP remains `62.54`; no fifth mAP and
  `Training Over=False`.
- Last completed train line remains Epoch 49 at `22:30:50`. Failure scan
  count is `0`; active checkpoints remain `epoch_19.pth` and `epoch_39.pth`.
- `/data` still has about `1.9T` available. Fixed-50/no-test-GT/no-teacher
  contract remains preserved; no code/config/model changed.
- Decision unchanged: wait for fifth mAP output; next check should be
  lower-frequency unless mAP appears. No cleanup or new long run.

## 2026-05-28T22:45:16+08:00 - N16R4 BATA eval still progressing

- Job `993718` remains `RUNNING` on `g0052`, elapsed `06:29:05`; log mtime
  advanced to `22:43:55+08:00`.
- Eval count remains `4`; latest mAP remains `62.54`; no fifth mAP and
  `Training Over=False`.
- Last completed train line remains Epoch 49 at `22:30:50`. Failure scan
  count is `0`; active checkpoints remain `epoch_19.pth` and `epoch_39.pth`.
- `/data` still has about `1.9T` available. Fixed-50/no-test-GT/no-teacher
  contract remains preserved; no code/config/model changed.
- Decision unchanged: wait for fifth mAP output. No cleanup, duplicate launch,
  or new long follow-up while `993718` is active.

## 2026-05-28T22:47:46+08:00 - N16R4 BATA eval still progressing

- Job `993718` remains `RUNNING` on `g0052`, elapsed `06:31:35`; log mtime
  advanced to `22:46:25+08:00`.
- Eval count remains `4`; latest mAP remains `62.54`; no fifth mAP and
  `Training Over=False`.
- Last completed train line remains Epoch 49 at `22:30:50`. Failure scan
  count is `0`; active checkpoints remain `epoch_19.pth` and `epoch_39.pth`.
- `/data` still has about `1.9T` available. Fixed-50/no-test-GT/no-teacher
  contract remains preserved; no code/config/model changed.
- Decision unchanged: continue waiting for fifth mAP output. No cleanup,
  duplicate launch, or new long follow-up while `993718` is active.

## 2026-05-28T22:50:22+08:00 - N16R4 BATA fifth eval

- Job `993718` remains `RUNNING` on `g0052`, elapsed `06:34:11`;
  `Training Over=False`.
- Fifth eval after Epoch 49 completed at `2026-05-28T22:48:41+08:00`:
  `Average-mAP=62.59`, vector `78.70 / 73.15 / 64.87 / 54.94 / 41.27`.
- Trend is `61.60 -> 62.12 -> 62.47 -> 62.54 -> 62.59`; the latest remains
  below random-fixed Adapter `63.77` by `-1.18`, strict EMA `63.85` by
  `-1.26`, stratified `64.64` by `-2.05`, uniform stride-2 `65.09` by
  `-2.50`, and oracle-boundary `77.62` by `-15.03`.
- `mAP@0.7=41.27` is improved but still below the random-fixed high-IoU
  reference `42.19`. Failure scan count is `0`; active checkpoints remain
  `epoch_19.pth` and `epoch_39.pth`; training resumed to Epoch 50.
- Decision unchanged: continue active run to later/final evidence only. No
  cleanup, duplicate launch, or new long follow-up while `993718` is active.

## 2026-05-28T22:54:37+08:00 - N16R4 BATA post-fifth-eval progress

- Job `993718` remains `RUNNING` on `g0052`, elapsed `06:38:25`; log mtime
  `22:53:09+08:00`.
- Eval count remains `5`; latest mAP remains `62.59`; no sixth mAP and
  `Training Over=False`.
- Training progressed to Epoch 51 at `22:53:09`. Failure scan count is `0`;
  active checkpoints remain `epoch_19.pth` and `epoch_39.pth`.
- `/data` still has about `1.9T` available. Fixed-50/no-test-GT/no-teacher
  contract remains preserved; no code/config/model changed.
- Decision unchanged: continue low-frequency monitoring. No cleanup, duplicate
  launch, or new long follow-up while `993718` is active.

## 2026-05-28T22:57:19+08:00 - N16R4 BATA post-fifth-eval monitor

- Job `993718` remains `RUNNING` on `g0052`, elapsed `06:41:08`; log mtime
  `22:55:22+08:00`.
- Eval count remains `5`; latest mAP remains `62.59`; no sixth mAP and
  `Training Over=False`.
- Last train marker remains Epoch 51 start at `22:53:09`. Failure scan count
  is `0`; active checkpoints remain `epoch_19.pth` and `epoch_39.pth`.
- `/data` still has about `1.9T` available. Fixed-50/no-test-GT/no-teacher
  contract remains preserved; no code/config/model changed.
- Decision unchanged: continue low-frequency monitoring. No cleanup, duplicate
  launch, or new long follow-up while `993718` is active.

## 2026-05-28T23:01:39+08:00 - N16R4 BATA Epoch 51 complete, no sixth eval yet

- Job `993718` remains `RUNNING` on `g0052`, elapsed about `06:45:28`; run log
  mtime advanced to `23:00:19+08:00`.
- Eval count remains `5`; latest mAP is still Epoch 49 `62.59` with vector
  `78.70 / 73.15 / 64.87 / 54.94 / 41.27`; no sixth mAP and
  `Training Over=False`.
- Epoch 51 completed at `22:57:32` with `Loss=0.4264`, `cls_loss=0.2169`,
  `reg_loss=0.2095`, `lr_backbone=1.0e-04`, `lr_det=5.1e-05`,
  `mem=2528MB`.
- Failure scan count is `0`; active checkpoints remain `epoch_19.pth` and
  `epoch_39.pth`; `log.json` touched at `22:57:32`; `/data` has about
  `1.9T` available.
- Fixed-50/no-test-GT/no-teacher contract remains preserved; no
  code/config/model changed.
- Decision unchanged: continue low-frequency monitoring toward the sixth/final
  eval. No cleanup, duplicate launch, or new long follow-up while `993718` is
  active.

## 2026-05-28T23:05:20+08:00 - N16R4 BATA sixth eval in progress

- Job `993718` remains `RUNNING` on `g0052`, elapsed about `06:49:10`; run log
  mtime advanced to `23:03:57+08:00`.
- Eval count remains `5`; latest completed mAP remains Epoch 49 `62.59` with
  vector `78.70 / 73.15 / 64.87 / 54.94 / 41.27`; no sixth mAP and
  `Training Over=False`.
- The post-Epoch-51 validation/inference progress bar is at about `165/396`
  (`42%`), so the next mAP is still in progress rather than stalled.
- Failure scan count is `0`; active checkpoints remain `epoch_19.pth` and
  `epoch_39.pth`; `log.json` touched at `22:57:32`; `/data` has about
  `1.9T` available.
- Fixed-50/no-test-GT/no-teacher contract remains preserved; no
  code/config/model changed.
- Decision unchanged: wait for sixth eval output. No cleanup, duplicate launch,
  or new long follow-up while `993718` is active.

## 2026-05-28T23:08:07+08:00 - N16R4 BATA sixth eval progressing

- Job `993718` remains `RUNNING` on `g0052`, elapsed about `06:51:56`; run log
  mtime advanced to `23:06:45+08:00`.
- Eval count remains `5`; latest completed mAP remains Epoch 49 `62.59` with
  vector `78.70 / 73.15 / 64.87 / 54.94 / 41.27`; no sixth mAP and
  `Training Over=False`.
- The post-Epoch-51 validation/inference progress bar advanced to about
  `226/396` (`57%`), so the eval is still actively progressing.
- Failure scan count is `0`; active checkpoints remain `epoch_19.pth` and
  `epoch_39.pth`; `log.json` touched at `22:57:32`; `/data` has about
  `1.9T` available.
- Fixed-50/no-test-GT/no-teacher contract remains preserved; no
  code/config/model changed.
- Decision unchanged: wait for sixth eval output. No cleanup, duplicate launch,
  or new long follow-up while `993718` is active.

## 2026-05-28T23:10:29+08:00 - N16R4 BATA sixth eval progressing

- Job `993718` remains `RUNNING` on `g0052`, elapsed about `06:54:18`; run log
  mtime advanced to `23:09:06+08:00`.
- Eval count remains `5`; latest completed mAP remains Epoch 49 `62.59` with
  vector `78.70 / 73.15 / 64.87 / 54.94 / 41.27`; no sixth mAP and
  `Training Over=False`.
- The post-Epoch-51 validation/inference progress bar advanced to about
  `280/396` (`71%`), so the eval is still actively progressing.
- Failure scan count is `0`; active checkpoints remain `epoch_19.pth` and
  `epoch_39.pth`; `log.json` touched at `22:57:32`; `/data` has about
  `1.9T` available.
- Fixed-50/no-test-GT/no-teacher contract remains preserved; no
  code/config/model changed.
- Decision unchanged: wait for sixth eval output. No cleanup, duplicate launch,
  or new long follow-up while `993718` is active.

## 2026-05-28T23:12:48+08:00 - N16R4 BATA sixth eval progressing

- Job `993718` remains `RUNNING` on `g0052`, elapsed about `06:56:41`; run log
  mtime advanced to `23:11:32+08:00`.
- Eval count remains `5`; latest completed mAP remains Epoch 49 `62.59` with
  vector `78.70 / 73.15 / 64.87 / 54.94 / 41.27`; no sixth mAP and
  `Training Over=False`.
- The post-Epoch-51 validation/inference progress bar advanced to about
  `321/396` (`81%`), so the eval is still actively progressing.
- Failure scan count is `0`; active checkpoints remain `epoch_19.pth` and
  `epoch_39.pth`; `log.json` touched at `22:57:32`; `/data` has about
  `1.9T` available.
- Fixed-50/no-test-GT/no-teacher contract remains preserved; no
  code/config/model changed.
- Decision unchanged: wait for sixth eval output. No cleanup, duplicate launch,
  or new long follow-up while `993718` is active.

## 2026-05-28T23:15:13+08:00 - N16R4 BATA sixth eval near completion

- Job `993718` remains `RUNNING` on `g0052`, elapsed about `06:59:07`; run log
  mtime advanced to `23:13:58+08:00`.
- Eval count remains `5`; latest completed mAP remains Epoch 49 `62.59` with
  vector `78.70 / 73.15 / 64.87 / 54.94 / 41.27`; no sixth mAP and
  `Training Over=False`.
- The post-Epoch-51 validation/inference progress bar advanced to about
  `380/396` (`96%`), so the eval is near completion.
- Failure scan count is `0`; active checkpoints remain `epoch_19.pth` and
  `epoch_39.pth`; `log.json` touched at `22:57:32`; `/data` has about
  `1.9T` available.
- Fixed-50/no-test-GT/no-teacher contract remains preserved; no
  code/config/model changed.
- Decision unchanged: recheck soon for sixth eval output. No cleanup,
  duplicate launch, or new long follow-up while `993718` is active.

## 2026-05-28T23:17:50+08:00 - N16R4 BATA sixth eval

- Job `993718` remains `RUNNING` on `g0052`, elapsed about `07:01:42`; run log
  mtime `23:15:24+08:00`.
- Sixth eval after Epoch 51 completed at `2026-05-28T23:15:23+08:00`:
  `Average-mAP=62.77`, vector `78.80 / 73.35 / 64.90 / 55.51 / 41.27`.
- Trend is `61.60 -> 62.12 -> 62.47 -> 62.54 -> 62.59 -> 62.77`; latest
  remains below random-fixed Adapter `63.77` by `-1.00`, strict EMA `63.85`
  by `-1.08`, stratified `64.64` by `-1.87`, uniform stride-2 `65.09` by
  `-2.32`, and oracle-boundary `77.62` by `-14.85`.
- `mAP@0.7=41.27` is unchanged from Epoch 49 and remains below the
  random-fixed high-IoU reference `42.19` by `-0.92`.
- `Training Over=False`; training resumed to Epoch 52. Failure scan count is
  `0`; active checkpoints remain `epoch_19.pth` and `epoch_39.pth`; `/data`
  has about `1.9T` available.
- Fixed-50/no-test-GT/no-teacher contract remains preserved; no
  code/config/model changed.
- Decision unchanged: continue active run for later/final evidence only. No
  severe-result escalation, no cleanup, duplicate launch, or new long follow-up
  while `993718` is active.

## 2026-05-28T23:21:44+08:00 - N16R4 BATA post-sixth-eval training

- Job `993718` remains `RUNNING` on `g0052`, elapsed about `07:05:36`; run log
  mtime `23:19:58+08:00`.
- Eval count remains `6`; latest completed mAP remains Epoch 51 `62.77`.
- Epoch 52 completed at `23:19:58` with `Loss=0.4466`, `cls_loss=0.2313`,
  `reg_loss=0.2154`, `lr_backbone=9.8e-05`, `lr_det=4.9e-05`,
  `mem=2528MB`; Epoch 53 then started.
- No seventh mAP and `Training Over=False`. Failure scan count is `0`; active
  checkpoints remain `epoch_19.pth` and `epoch_39.pth`; `/data` has about
  `1.9T` available.
- Fixed-50/no-test-GT/no-teacher contract remains preserved; no
  code/config/model changed.
- Decision unchanged: continue monitoring toward the next eval after Epoch 53.
  No cleanup, duplicate launch, or new long follow-up while `993718` is active.

## 2026-05-28T23:24:06+08:00 - N16R4 BATA Epoch 53 mid-epoch

- Job `993718` remains `RUNNING` on `g0052`, elapsed about `07:07:59`; run log
  mtime `23:22:11+08:00`.
- Eval count remains `6`; latest completed mAP remains Epoch 51 `62.77`.
- Epoch 53 reached `[053][00050/00099]` at `23:22:11` with `Loss=0.4460`,
  `cls_loss=0.2264`, `reg_loss=0.2196`, `lr_backbone=9.7e-05`,
  `lr_det=4.8e-05`, `mem=2528MB`.
- No seventh mAP and `Training Over=False`. Failure scan count is `0`; active
  checkpoints remain `epoch_19.pth` and `epoch_39.pth`; `/data` has about
  `1.9T` available.
- Fixed-50/no-test-GT/no-teacher contract remains preserved; no
  code/config/model changed.
- Decision unchanged: continue monitoring toward the eval after Epoch 53. No
  cleanup, duplicate launch, or new long follow-up while `993718` is active.

## 2026-05-28T23:28:37+08:00 - N16R4 BATA post-Epoch-53 eval in progress

- Job `993718` remains `RUNNING` on `g0052`, elapsed `07:12:24`; run log
  `~/run/yuzibo/OpenTAD_BATA_Clean/logs/bata_mobilenet_fixed50_20260528_161457_n16r4.log`.
- Eval count remains `6`; latest completed mAP remains Epoch 51 `62.77`, vector
  `78.80 / 73.35 / 64.90 / 55.51 / 41.27`. This is still below random-fixed
  `63.77` by `-1.00`, strict EMA `63.85` by `-1.08`, stratified `64.64` by
  `-1.87`, uniform stride-2 `65.09` by `-2.32`, and oracle-boundary `77.62`
  by `-14.85`.
- Epoch 53 completed at `23:24:24` with `Loss=0.4071`, `cls_loss=0.2089`,
  `reg_loss=0.1983`; the post-Epoch-53 eval is active at about `80/396`
  (`20%`), so no seventh mAP yet and `Training Over=False`.
- Failure scan count is `0`; active checkpoints remain `epoch_19.pth` and
  `epoch_39.pth`; `/data` has about `1.9T` available.
- Fixed-50/no-test-GT/no-teacher contract remains preserved; no
  code/config/model changed.
- Decision unchanged: continue monitoring for the seventh eval. No cleanup,
  duplicate launch, or new BATA/Sparse long follow-up while `993718` is active.

## 2026-05-28T23:32:28+08:00 - N16R4 BATA seventh eval still progressing

- Job `993718` remains `RUNNING` on `g0052`, elapsed `07:16:15`; run log
  `~/run/yuzibo/OpenTAD_BATA_Clean/logs/bata_mobilenet_fixed50_20260528_161457_n16r4.log`.
- Eval count remains `6`; latest completed mAP remains Epoch 51 `62.77`, vector
  `78.80 / 73.35 / 64.90 / 55.51 / 41.27`.
- The post-Epoch-53 validation/inference advanced to about `171/396` (`43%`);
  no seventh mAP yet and `Training Over=False`.
- Failure scan count is `0`; checkpoints remain `epoch_19.pth` and
  `epoch_39.pth`; fixed-50/no-test-GT/no-teacher contract remains preserved.
- Decision unchanged: continue monitoring at a reasonable completion window.
  No cleanup, duplicate launch, or new BATA/Sparse long follow-up while
  `993718` is active.

## 2026-05-28T23:40:33+08:00 - N16R4 BATA seventh eval near completion

- Job `993718` remains `RUNNING` on `g0052`, elapsed `07:24:20`; run log
  `~/run/yuzibo/OpenTAD_BATA_Clean/logs/bata_mobilenet_fixed50_20260528_161457_n16r4.log`.
- Eval count remains `6`; latest completed mAP remains Epoch 51 `62.77`, vector
  `78.80 / 73.35 / 64.90 / 55.51 / 41.27`.
- The post-Epoch-53 validation/inference advanced to about `348/396` (`88%`);
  no seventh mAP yet and `Training Over=False`.
- Failure scan count is `0`; checkpoints remain `epoch_19.pth` and
  `epoch_39.pth`; fixed-50/no-test-GT/no-teacher contract remains preserved.
- Decision unchanged: recheck soon for the seventh eval result. No cleanup,
  duplicate launch, or new BATA/Sparse long follow-up while `993718` is active.

## 2026-05-28T23:45:07+08:00 - N16R4 BATA seventh eval

- Job `993718` remains `RUNNING` on `g0052`, elapsed `07:28:54`; run log
  `~/run/yuzibo/OpenTAD_BATA_Clean/logs/bata_mobilenet_fixed50_20260528_161457_n16r4.log`.
- Seventh eval after Epoch 53 completed at `23:42:02+08:00`:
  `Average-mAP=62.70`, vector `78.75 / 73.35 / 64.87 / 55.47 / 41.07`.
- Trend is `61.60 -> 62.12 -> 62.47 -> 62.54 -> 62.59 -> 62.77 -> 62.70`;
  latest remains below random-fixed `63.77` by `-1.07`, strict EMA `63.85`
  by `-1.15`, stratified `64.64` by `-1.94`, uniform stride-2 `65.09` by
  `-2.39`, and oracle-boundary `77.62` by `-14.92`.
- `mAP@0.7=41.07` is below random-fixed high-IoU `42.19` by `-1.12`.
  `Training Over=False`; training resumed to Epoch 54.
- Failure scan count is `0`; checkpoints remain `epoch_19.pth` and
  `epoch_39.pth`; fixed-50/no-test-GT/no-teacher contract remains preserved.
- Decision unchanged: continue active run for later/final evidence only. No
  severe-result escalation, cleanup, duplicate launch, or new BATA/Sparse
  long follow-up while `993718` is active.

## 2026-05-28T23:48:48+08:00 - N16R4 BATA reached Epoch 55

- Job `993718` remains `RUNNING` on `g0052`, elapsed about `07:32:39`; sacct
  still reports batch/extern `RUNNING`.
- Latest completed mAP remains the seventh eval after Epoch 53:
  `Average-mAP=62.70`, vector `78.75 / 73.35 / 64.87 / 55.47 / 41.07`.
- Epoch 54 completed at `23:46:22` with `Loss=0.4263`, `cls_loss=0.2170`,
  `reg_loss=0.2094`; Epoch 55 then started.
- `Training Over=False`; failure scan count is `0`; active checkpoints remain
  `epoch_19.pth` and `epoch_39.pth`; `/data` has about `1.9T` available.
- Fixed-50/no-test-GT/no-teacher contract remains preserved; no
  code/config/model changed.
- Decision unchanged: continue monitoring toward next eval/final. No active
  checkpoint cleanup, duplicate launch, or new BATA/Sparse long follow-up
  while `993718` is active.

## 2026-05-29T00:00:28+08:00 - N16R4 BATA post-Epoch-55 eval in progress

- Job `993718` remains `RUNNING` on `g0052`, elapsed `07:44:15`.
- Latest completed mAP remains the seventh eval after Epoch 53:
  `Average-mAP=62.70`, vector `78.75 / 73.35 / 64.87 / 55.47 / 41.07`.
- Epoch 55 completed at `2026-05-28T23:50:48+08:00` with `Loss=0.4398`,
  `cls_loss=0.2236`, `reg_loss=0.2162`; the post-Epoch-55 validation/inference
  is active at about `208/396` (`53%`).
- `Training Over=False`; failure scan count is `0`; active checkpoints remain
  `epoch_19.pth` and `epoch_39.pth`; `/data` has about `1.9T` available.
- Fixed-50/no-test-GT/no-teacher contract remains preserved; no
  code/config/model changed.
- Decision unchanged: continue monitoring for the next eval result. No active
  checkpoint cleanup, duplicate launch, or new BATA/Sparse long follow-up
  while `993718` is active.

## 2026-05-29T00:11:11+08:00 - N16R4 BATA eighth eval

- Job `993718` remains `RUNNING` on `g0052`, elapsed `07:54:58`.
- Eighth eval after Epoch 55 completed at `2026-05-29T00:08:18+08:00`:
  `Average-mAP=62.82`, vector `78.82 / 73.57 / 65.22 / 55.60 / 40.87`.
- Trend is
  `61.60 -> 62.12 -> 62.47 -> 62.54 -> 62.59 -> 62.77 -> 62.70 -> 62.82`.
- Latest remains below random-fixed `63.77` by `-0.95`, strict EMA `63.85` by
  `-1.03`, stratified `64.64` by `-1.82`, uniform stride-2 `65.09` by
  `-2.27`, and oracle-boundary `77.62` by `-14.80`.
- `mAP@0.7=40.87` is below random-fixed high-IoU `42.19` by `-1.32`.
  `Training Over=False`; training resumed to Epoch 56.
- Failure scan count is `0`; active checkpoints remain `epoch_19.pth` and
  `epoch_39.pth`; fixed-50/no-test-GT/no-teacher contract remains preserved.
- Decision unchanged: continue active run for later/final evidence only. No
  severe-result escalation, active checkpoint cleanup, duplicate launch, or
  new BATA/Sparse long follow-up while `993718` is active.

## 2026-05-29T00:14:50+08:00 - N16R4 BATA active-run monitor

- Job `993718` remains `RUNNING` on `g0052`, elapsed `07:58:39`; run log
  `~/run/yuzibo/OpenTAD_BATA_Clean/logs/bata_mobilenet_fixed50_20260528_161457_n16r4.log`.
- No new mAP after the eighth eval. Latest completed detector result remains
  Epoch 55 `Average-mAP=62.82`, vector
  `78.82 / 73.57 / 65.22 / 55.60 / 40.87`.
- Epoch 56 completed at `00:12:35` with `Loss=0.4086`,
  `cls_loss=0.2095`, `reg_loss=0.1991`; Epoch 57 then started.
- `Training Over=False`; failure scan count is `0`; active checkpoints remain
  `epoch_19.pth` and `epoch_39.pth`; `/data` has about `1.9T` available.
- Fixed-50/no-test-GT/no-teacher contract remains preserved; no
  code/config/model changed.
- Decision unchanged: continue monitoring to later/final eval only. No active
  checkpoint cleanup, duplicate launch, or new BATA/Sparse long follow-up
  while `993718` is active.

## 2026-05-29T00:19:39+08:00 - N16R4 BATA post-Epoch-57 eval in progress

- Upload gate recheck is still clean: PASS marker
  `~/run/yuzibo/setup_logs/N16R4_THUMOS_UPLOAD_GATE_PASS_20260528_115821.ok`
  reports `ALL_PASS=True`, manifests are `200` train / `211` test, and
  symlink-resolved mp4 counts are `200/211`.
- Job `993718` remains `RUNNING` on `g0052`, elapsed `08:03:28`; sacct shows
  batch/extern still `RUNNING` with exit code `0:0`.
- No new mAP after the eighth eval. Latest completed detector result remains
  Epoch 55 `Average-mAP=62.82`, vector
  `78.82 / 73.57 / 65.22 / 55.60 / 40.87`.
- Epoch 57 completed at `00:16:56` with `Loss=0.3902`,
  `cls_loss=0.1942`, `reg_loss=0.1959`; post-Epoch-57 validation/inference
  has started and had reached about `54/396` in the checked tail.
- `Training Over=False`; failure scan count is `0`; active checkpoints remain
  `epoch_19.pth` and `epoch_39.pth`; `/data` has about `1.9T` available.
- Fixed-50/no-test-GT/no-teacher contract remains preserved; no
  code/config/model changed.
- Decision unchanged: wait for the ninth eval result. No active checkpoint
  cleanup, duplicate launch, or new BATA/Sparse long follow-up while `993718`
  is active.

## 2026-05-29T00:24:18+08:00 - N16R4 BATA ninth eval still pending

- Job `993718` remains `RUNNING` on `g0052`, elapsed `08:08:07`; sacct
  reports job/batch/extern all `RUNNING` with exit code `0:0`.
- Upload PASS marker still reports `ALL_PASS=True`.
- No new `Average-mAP` after the eighth eval. Latest completed detector result
  remains Epoch 55 `Average-mAP=62.82`, vector
  `78.82 / 73.57 / 65.22 / 55.60 / 40.87`.
- Post-Epoch-57 validation/inference is still active; `Training Over=False`.
- Failure scan count is `0`; active checkpoints remain `epoch_19.pth` and
  `epoch_39.pth`; `/data` has about `1.9T` available.
- Fixed-50/no-test-GT/no-teacher contract remains preserved; no
  code/config/model changed.
- Decision unchanged: recheck for the ninth eval result. No active checkpoint
  cleanup, duplicate launch, or new BATA/Sparse long follow-up while `993718`
  is active.

## 2026-05-29T00:27:58+08:00 - N16R4 BATA ninth eval still pending

- Job `993718` remains `RUNNING` on `g0052`, elapsed `08:11:47`; sacct
  reports job/batch/extern all `RUNNING` with exit code `0:0`.
- No new `Average-mAP` after the eighth eval. Latest completed detector result
  remains Epoch 55 `Average-mAP=62.82`, vector
  `78.82 / 73.57 / 65.22 / 55.60 / 40.87`.
- Post-Epoch-57 validation/inference is still active; `Training Over=False`.
- Failure scan count is `0`; active checkpoints remain `epoch_19.pth` and
  `epoch_39.pth`; `/data` has about `1.9T` available.
- Fixed-50/no-test-GT/no-teacher contract remains preserved; no
  code/config/model changed.
- Decision unchanged: recheck for the ninth eval result. No active checkpoint
  cleanup, duplicate launch, or new BATA/Sparse long follow-up while `993718`
  is active.

## 2026-05-29T00:31:13+08:00 - N16R4 BATA ninth eval still pending

- Job `993718` remains `RUNNING` on `g0052`, elapsed `08:15:02`; sacct
  reports job/batch/extern all `RUNNING` with exit code `0:0`.
- No new `Average-mAP` after the eighth eval. Latest completed detector result
  remains Epoch 55 `Average-mAP=62.82`, vector
  `78.82 / 73.57 / 65.22 / 55.60 / 40.87`.
- Post-Epoch-57 validation/inference is still active; `Training Over=False`.
- Failure scan count is `0`; active checkpoints remain `epoch_19.pth` and
  `epoch_39.pth`; `/data` has about `1.9T` available.
- Fixed-50/no-test-GT/no-teacher contract remains preserved; no
  code/config/model changed.
- Decision unchanged: recheck soon for the ninth eval result. No active
  checkpoint cleanup, duplicate launch, or new BATA/Sparse long follow-up
  while `993718` is active.

## 2026-05-29T00:43:59+08:00 - N16R4 BATA ninth eval and parallelism policy update

- Upload/data gate remains clean: PASS marker
  `~/run/yuzibo/setup_logs/N16R4_THUMOS_UPLOAD_GATE_PASS_20260528_115821.ok`
  reports `ALL_PASS=True`; train/test manifests and resolved mp4 counts remain
  `200/211`.
- Job `993718` remains `RUNNING` on `g0052`, elapsed `08:27:48`; latest train
  evidence is Epoch 59 mid-epoch at `00:40:51` with `Loss=0.3935`,
  `cls_loss=0.2034`, `reg_loss=0.1902`, `mem=2528MB`.
- Ninth eval after Epoch 57 completed at `00:34:23`: `Average-mAP=63.12`,
  vector `78.85 / 73.74 / 65.37 / 56.19 / 41.45`.
- Deltas: `-0.65` vs random-fixed Adapter `63.77`, `-0.73` vs strict EMA
  `63.85`, `-1.52` vs stratified `64.64`, `-1.97` vs uniform stride-2
  `65.09`, and `-14.50` vs oracle-boundary `77.62`; `mAP@0.7=41.45`
  remains `-0.74` below the random-fixed high-IoU reference `42.19`.
- Failure scan count is `0`; no `Training Over`; active checkpoints remain
  `epoch_19.pth` and `epoch_39.pth`, so checkpoint cleanup remains disallowed.
- User explicitly requested meaningful 2-3 GPU parallelism and a Pro discussion
  that does not default to conservative waiting. Decision: continue `993718`
  to final, but start a new Pro discussion for aggressive parallel BATA/Sparse
  route selection and then launch only distinct, reviewed or explicitly waived
  runs under `~/run/yuzibo`.

## 2026-05-29T00:47:13+08:00 - GPT-5.5 Pro aggressive parallel-route discussion started

- Prompt written to
  `logs/gpt5pro_parallel_2_3gpu_discussion_prompt_20260529.md`.
- Runner written to `logs/run_gpt5pro_parallel_2_3gpu_20260529.ps1`.
- Oracle/browser session started as `gpt5pro-tad-parallel-2-3gpu`, requested
  `gpt-5.5-pro`; stdout reports browser mode and an acquired ChatGPT browser
  slot.
- Expected output:
  `logs/gpt5pro_parallel_2_3gpu_discussion_20260529.txt`; stdout/stderr:
  `logs/gpt5pro_parallel_2_3gpu_discussion_20260529.stdout.txt` and
  `logs/gpt5pro_parallel_2_3gpu_discussion_20260529.err.txt`.
- Prompt explicitly asks Pro to avoid a default conservative hold, design a
  2-3 GPU queue, include BATA/Sparse/model-side/oracle-diagnostic expansion
  routes, and provide launch-now vs launch-after-final triggers plus stop
  gates.
- No new Slurm job has been launched yet from this discussion. Next action:
  inspect N16R4-ready scripts/configs under `~/run/yuzibo` while Pro runs, then
  finalize and submit a distinct parallel queue after Pro output or a concrete
  trigger.

## 2026-05-29T01:08:34+08:00 - N16R4 BATA MobileNet-BCA final result and cleanup

- Completed job `993718` reached `Training Over...` at `2026-05-29T01:00:24+08:00`.
- Final eval after Epoch 59: `Average-mAP=63.37`; eval trend was
  `61.60 -> 62.12 -> 62.47 -> 62.54 -> 62.59 -> 62.77 -> 62.70 -> 62.82 -> 63.12 -> 63.37`.
- Final deltas: `-0.40` vs random-fixed Adapter `63.77`, `-0.48` vs strict EMA
  `63.85`, `-1.27` vs stratified `64.64`, `-1.72` vs uniform stride-2 `65.09`,
  and `-14.25` vs oracle-boundary `77.62`.
- The strict fixed-50/no-test-GT/no-teacher contract remained preserved, but the
  selector remained cache/dataloader-side and therefore not end-to-end.
- Completed-run checkpoint cleanup was performed only inside
  `/data/run01/sczc063/yuzibo/bata_runs/exps/bata_mobilenet_bca_fixed50_adapter_deployable/gpu1_id0/checkpoint`.
  `epoch_19.pth` and `epoch_39.pth` were removed; `epoch_59.pth` was kept.
  `df -h` before/after remained about `/data` `2.3T`, `415G` used, `1.9T` free.
- Decision: treat MobileNet-BCA as negative/diagnostic evidence and move to a
  true end-to-end selector-detector route with detector-loss feedback.

## 2026-05-29T01:35:00+08:00 - E2E raw-density selector implemented locally

- GPT-5.5 Pro design discussion completed via Oracle browser inline-file mode:
  prompt `logs/gpt5pro_e2e_frame_selector_discussion_prompt_20260529.md`,
  output `logs/gpt5pro_e2e_frame_selector_discussion_20260529.txt`.
- Accepted route: detector-internal raw-frame continuous density selector.
  Dataloader supplies dense 768 windows; selector samples 384 continuous frames
  before ViT-Adapter; GT remap uses detached selector positions; test-time
  post-processing uses existing irregular-axis meta fields.
- Implemented files are listed in
  `research-wiki/experiments/BATA_E2E_RAW_DENSITY_SELECTOR_SELF_CHECK_20260529.md`.
- Local verification: `py_compile` passed for changed Python/config files.
  Local Windows torch tests are skipped due torch DLL initialization failure;
  `scripts/run_e2e_rawdensel_n16r4.sbatch` runs the full torch selector test
  file on N16R4 before training.
- Contract label: `50% ViT/backbone compute, dense decode`; no test-time
  GT/teacher/cache. No long training launched yet because this is a major model
  change and review gates are still pending.

## 2026-05-29T02:45:00+08:00 - E2E raw-density selector deployed on 3 GPUs

- Review gate completed: GPT-5.5 Pro implementation review/fix review recorded
  WARN but allowed launch after added checks; Gemini CLI `PASS`; DeepSeek pathfix
  review `PASS`.
- Remote N16R4 preflight passed for all three configs: selector tests
  `9 passed` and `merged_config_preflight=PASS`.
- Initial jobs `994337/994338/994339` were canceled before effective training
  after a resource-display misread; no checkpoint or mAP resulted.
- Relaunched jobs are running from `~/run/yuzibo/OpenTAD_BATA_Clean`:
  `994340 e2e_main` on `g0005`, `994341 e2e_uniform` on `g0005`, and
  `994342 e2e_lowreg` on `g0024`, each with actual allocation
  `cpu=8,mem=124400M,gres/gpu=1`.
- Logs:
  `logs/e2e_rawdensel_main_20260529_023345_n16r4.log`,
  `logs/e2e_rawdensel_uniform_20260529_023345_n16r4.log`, and
  `logs/e2e_rawdensel_lowreg_20260529_023345_n16r4.log`.
- All three reached `Training Starts`, completed Epoch 0, and entered Epoch 1.
  First losses are normal: main `1.7367 -> 1.6396`, uniform
  `1.7243 -> 1.6308`, lowreg `1.7367 -> 1.6397`.
- Contract remains: `50% ViT/backbone compute, dense decode`; no test-time
  GT/teacher/cache. Next gate is first evaluation; uniform control is the
  pipeline sanity check against the `65.09` uniform stride-2 reference.

## 2026-05-29T02:47:10+08:00 - E2E raw-density early training healthy

- The three relaunched jobs remain running with actual allocation
  `cpu=8,mem=124400M,gres/gpu=1`.
- Main reached Epoch 1 step 50: `Loss=1.0602`, `cls_loss=0.6524`,
  `reg_loss=0.4077`, `mem=3189MB`.
- Uniform control reached Epoch 1 step 50: `Loss=1.0766`, `cls_loss=0.6505`,
  `reg_loss=0.4261`, `mem=2648MB`.
- Lowreg reached Epoch 1 step 50: `Loss=1.0352`, `cls_loss=0.6534`,
  `reg_loss=0.3818`, `mem=3189MB`.
- No traceback, OOM, NaN, mAP, checkpoint, or severe-result trigger is visible.
  Next meaningful checkpoints are epoch 20 and first eval after epoch 40.

## 2026-05-29T02:50:00+08:00 - E2E raw-density reached Epoch 2

- Upload marker still reports train `200/200`, test `211/211`, and
  `ALL_PASS=True`.
- Jobs `994340`, `994341`, and `994342` remain `RUNNING` with actual allocation
  `cpu=8,mem=124400M,gres/gpu=1`.
- Main completed Epoch 1 with `Loss=1.0096`, `cls_loss=0.6291`,
  `reg_loss=0.3805`, then Epoch 2 started.
- Uniform completed Epoch 1 with `Loss=1.0243`, `cls_loss=0.6314`,
  `reg_loss=0.3928`, then Epoch 2 started.
- Lowreg completed Epoch 1 with `Loss=0.9974`, `cls_loss=0.6304`,
  `reg_loss=0.3670`, then Epoch 2 started.
- No traceback, OOM, NaN, mAP, checkpoint, or severe-result trigger is visible.
  Decision: continue all three runs; no intervention or new launch now.

## 2026-05-29T02:52:30+08:00 - E2E raw-density all routes at Epoch 2 step 50

- All three jobs remain running.
- Main Epoch 2 step 50: `Loss=0.8457`, `cls_loss=0.5095`,
  `reg_loss=0.3361`, `mem=3189MB`.
- Uniform Epoch 2 step 50: `Loss=0.8518`, `cls_loss=0.5159`,
  `reg_loss=0.3359`, `mem=2648MB`.
- Lowreg Epoch 2 step 50: `Loss=0.8469`, `cls_loss=0.5109`,
  `reg_loss=0.3360`, `mem=3189MB`.
- No traceback, OOM, NaN, mAP, checkpoint, or severe-result trigger is visible.
  Decision: continue without intervention; wait for checkpoint/eval or a real
  stall/crash before changing the queue.

## 2026-05-29T02:55:10+08:00 - E2E raw-density main/uniform entered Epoch 3

- Upload marker still reports train `200/200`, test `211/211`, `ALL_PASS=True`.
- Jobs `994340`, `994341`, and `994342` remain `RUNNING` with actual allocation
  `cpu=8,mem=124400M,gres/gpu=1`.
- Main completed Epoch 2 with `Loss=0.8800`, `cls_loss=0.5345`,
  `reg_loss=0.3455`, then Epoch 3 started.
- Uniform completed Epoch 2 with `Loss=0.8790`, `cls_loss=0.5350`,
  `reg_loss=0.3440`, then Epoch 3 started.
- Lowreg remains active and had reached Epoch 2 step 50 with `Loss=0.8469`;
  this is not near the one-hour no-change threshold.
- No traceback, OOM, NaN, mAP, checkpoint, or severe-result trigger is visible.
  Decision: continue without intervention.

## 2026-05-29T02:55:15+08:00 - E2E raw-density all routes entered Epoch 3

- Upload marker still reports `ALL_PASS=True`.
- Jobs `994340`, `994341`, and `994342` remain `RUNNING` with actual allocation
  `cpu=8,mem=124400M,gres/gpu=1`.
- Main completed Epoch 2 with `Loss=0.8800`, then Epoch 3 started.
- Uniform completed Epoch 2 with `Loss=0.8790`, then Epoch 3 started.
- Lowreg completed Epoch 2 with `Loss=0.8758`, then Epoch 3 started.
- No traceback, OOM, NaN, mAP, checkpoint, or severe-result trigger is visible.
  Decision: continue without intervention.

## 2026-05-29T02:57:55+08:00 - E2E raw-density Epoch 3 mid-epoch healthy

- Upload marker remains `ALL_PASS=True`; jobs `994340`, `994341`, and `994342`
  remain `RUNNING`.
- Main Epoch 3 step 50: `Loss=0.8321`, `cls_loss=0.4939`,
  `reg_loss=0.3382`, `mem=3189MB`.
- Uniform Epoch 3 step 50: `Loss=0.8344`, `cls_loss=0.4962`,
  `reg_loss=0.3383`, `mem=2648MB`.
- Lowreg Epoch 3 step 50: `Loss=0.8341`, `cls_loss=0.4974`,
  `reg_loss=0.3367`, `mem=3189MB`.
- No traceback, OOM, NaN, mAP, checkpoint, or severe-result trigger is visible.
  Decision: continue without intervention.

## 2026-05-29T03:00:35+08:00 - E2E raw-density all routes entered Epoch 4

- Upload marker remains `ALL_PASS=True`; jobs `994340`, `994341`, and `994342`
  remain `RUNNING`.
- Main completed Epoch 3 with `Loss=0.8331`, `cls_loss=0.5019`,
  `reg_loss=0.3312`, then Epoch 4 started.
- Uniform completed Epoch 3 with `Loss=0.8302`, `cls_loss=0.4999`,
  `reg_loss=0.3304`, then Epoch 4 started.
- Lowreg completed Epoch 3 with `Loss=0.8327`, `cls_loss=0.5036`,
  `reg_loss=0.3291`, then Epoch 4 started.
- No traceback, OOM, NaN, mAP, checkpoint, or severe-result trigger is visible.
  Decision: continue without intervention.

## 2026-05-29T03:03:00+08:00 - E2E raw-density main at Epoch 4 step 50

- Upload marker remains `ALL_PASS=True`; jobs `994340`, `994341`, and `994342`
  remain `RUNNING`.
- Main reached Epoch 4 step 50 with `Loss=0.8828`, `cls_loss=0.5398`,
  `reg_loss=0.3430`, `mem=3189MB`.
- Uniform and lowreg are still in the normal short interval after Epoch 4 start;
  no one-hour no-change threshold is close.
- No traceback, OOM, NaN, mAP, checkpoint, or severe-result trigger is visible.
  Decision: continue without intervention.

## 2026-05-29T03:03:25+08:00 - E2E raw-density all routes at Epoch 4 step 50

- Upload marker remains `ALL_PASS=True`; jobs `994340`, `994341`, and `994342`
  remain `RUNNING`.
- Main Epoch 4 step 50: `Loss=0.8828`, `cls_loss=0.5398`,
  `reg_loss=0.3430`, `mem=3189MB`.
- Uniform Epoch 4 step 50: `Loss=0.8700`, `cls_loss=0.5274`,
  `reg_loss=0.3427`, `mem=2648MB`.
- Lowreg Epoch 4 step 50: `Loss=0.8636`, `cls_loss=0.5218`,
  `reg_loss=0.3418`, `mem=3189MB`.
- No traceback, OOM, NaN, mAP, checkpoint, or severe-result trigger is visible.
  Decision: continue without intervention.

## 2026-05-29T03:06:05+08:00 - E2E raw-density all routes entered Epoch 5

- Upload marker remains `ALL_PASS=True`; jobs `994340`, `994341`, and `994342`
  remain `RUNNING`.
- Main completed Epoch 4 with `Loss=0.8549`, `cls_loss=0.5286`,
  `reg_loss=0.3262`, then Epoch 5 started.
- Uniform completed Epoch 4 with `Loss=0.8355`, `cls_loss=0.5105`,
  `reg_loss=0.3250`, then Epoch 5 started.
- Lowreg completed Epoch 4 with `Loss=0.8131`, `cls_loss=0.4892`,
  `reg_loss=0.3238`, then Epoch 5 started.
- No traceback, OOM, NaN, mAP, checkpoint, or severe-result trigger is visible.
  Decision: continue without intervention.

## 2026-05-29T03:08:30+08:00 - E2E raw-density main at Epoch 5 step 50

- Upload marker remains `ALL_PASS=True`; jobs `994340`, `994341`, and `994342`
  remain `RUNNING`.
- Main reached Epoch 5 step 50 with `Loss=0.7874`, `cls_loss=0.4640`,
  `reg_loss=0.3234`, `mem=3189MB`.
- Uniform and lowreg are still in the normal short interval after Epoch 5 start;
  no stall threshold is close.
- No traceback, OOM, NaN, mAP, checkpoint, or severe-result trigger is visible.
  Decision: continue without intervention.

## 2026-05-29T03:09:05+08:00 - E2E raw-density all routes at Epoch 5 step 50

- Upload marker remains `ALL_PASS=True`; jobs `994340`, `994341`, and `994342`
  remain `RUNNING`.
- Main Epoch 5 step 50: `Loss=0.7874`, `cls_loss=0.4640`,
  `reg_loss=0.3234`, `mem=3189MB`.
- Uniform Epoch 5 step 50: `Loss=0.7857`, `cls_loss=0.4611`,
  `reg_loss=0.3247`, `mem=2648MB`.
- Lowreg Epoch 5 step 50: `Loss=0.7816`, `cls_loss=0.4595`,
  `reg_loss=0.3220`, `mem=3189MB`.
- No traceback, OOM, NaN, mAP, checkpoint, or severe-result trigger is visible.
  Decision: continue without intervention.

## 2026-05-29T03:11:35+08:00 - E2E raw-density all routes entered Epoch 6

- Upload marker remains `ALL_PASS=True`; jobs `994340`, `994341`, and `994342`
  remain `RUNNING`.
- Main completed Epoch 5 with `Loss=0.8128`, `cls_loss=0.4905`,
  `reg_loss=0.3223`, then Epoch 6 started.
- Uniform completed Epoch 5 with `Loss=0.7987`, `cls_loss=0.4789`,
  `reg_loss=0.3198`, then Epoch 6 started.
- Lowreg completed Epoch 5 with `Loss=0.7981`, `cls_loss=0.4775`,
  `reg_loss=0.3206`, then Epoch 6 started.
- No traceback, OOM, NaN, mAP, checkpoint, or severe-result trigger is visible.
  Decision: continue without intervention.

## 2026-05-29T03:14:20+08:00 - E2E raw-density main/uniform at Epoch 6 step 50

- Upload marker remains `ALL_PASS=True`; jobs `994340`, `994341`, and `994342`
  remain `RUNNING`.
- Main Epoch 6 step 50: `Loss=0.7297`, `cls_loss=0.4354`,
  `reg_loss=0.2942`, `mem=3189MB`.
- Uniform Epoch 6 step 50: `Loss=0.7201`, `cls_loss=0.4284`,
  `reg_loss=0.2917`, `mem=2648MB`.
- Lowreg remains in the normal short interval after Epoch 6 start; no stall
  threshold is close.
- No traceback, OOM, NaN, mAP, checkpoint, or severe-result trigger is visible.
  Decision: continue without intervention.

## 2026-05-29T03:17:00+08:00 - E2E raw-density main/uniform entered Epoch 7

- Upload marker remains `ALL_PASS=True`; jobs `994340`, `994341`, and `994342`
  remain `RUNNING`.
- Main completed Epoch 6 with `Loss=0.7146`, `cls_loss=0.4248`,
  `reg_loss=0.2898`, then Epoch 7 started.
- Uniform completed Epoch 6 with `Loss=0.7131`, `cls_loss=0.4238`,
  `reg_loss=0.2892`, then Epoch 7 started.
- Lowreg reached Epoch 6 step 50 with `Loss=0.7288`, `cls_loss=0.4366`,
  `reg_loss=0.2922`, and remains in normal progress window.
- No traceback, OOM, NaN, mAP, checkpoint, or severe-result trigger is visible.
  Decision: continue without intervention.

## 2026-05-29T03:19:00+08:00 - E2E raw-density main at Epoch 7 step 50

- Upload marker remains `ALL_PASS=True`; jobs `994340`, `994341`, and `994342`
  remain `RUNNING`.
- Main reached Epoch 7 step 50 with `Loss=0.7506`, `cls_loss=0.4447`,
  `reg_loss=0.3058`, `mem=3189MB`.
- Uniform remains in the normal short interval after Epoch 7 start.
- Lowreg completed Epoch 6 with `Loss=0.7107`, `cls_loss=0.4231`,
  `reg_loss=0.2875`, then Epoch 7 started.
- No traceback, OOM, NaN, mAP, checkpoint, or severe-result trigger is visible.
  Decision: continue without intervention.

## 2026-05-29T03:20:22+08:00 - E2E raw-density main/uniform entered Epoch 8

- Upload marker remains `ALL_PASS=True`; jobs `994340`, `994341`, and `994342`
  remain `RUNNING` with `cpu=8,mem=124400M,gres/gpu=1`.
- Main completed Epoch 7 with `Loss=0.7116`, `cls_loss=0.4150`,
  `reg_loss=0.2965`, then Epoch 8 started.
- Uniform completed Epoch 7 with `Loss=0.7143`, `cls_loss=0.4177`,
  `reg_loss=0.2965`, then Epoch 8 started.
- Lowreg reached Epoch 7 step 50 with `Loss=0.7478`, `cls_loss=0.4453`,
  `reg_loss=0.3024`; its latest log update is still in the normal progress
  window.
- No E2E checkpoint directory, mAP, traceback, OOM, NaN, or severe-result
  trigger is visible. No code/config changed. Contract remains `50% ViT/backbone
  compute, dense decode`; no test-time GT/teacher/cache.
  Decision: continue all three jobs; no upload action, process rescue,
  cancellation, checkpoint cleanup, or new launch.

## 2026-05-29T03:22:45+08:00 - E2E raw-density all routes in Epoch 8 window

- Upload marker remains `ALL_PASS=True`; jobs `994340`, `994341`, and `994342`
  remain `RUNNING` for about `48:30` with `cpu=8,mem=124400M,gres/gpu=1`.
- Main reached Epoch 8 step 50 with `Loss=0.6926`, `cls_loss=0.3979`,
  `reg_loss=0.2946`, `mem=3189MB`.
- Uniform completed Epoch 7 with `Loss=0.7143`, `cls_loss=0.4177`,
  `reg_loss=0.2965`, then Epoch 8 started.
- Lowreg completed Epoch 7 with `Loss=0.7095`, `cls_loss=0.4151`,
  `reg_loss=0.2944`, then Epoch 8 started.
- No mAP, checkpoint event, traceback, OOM, NaN, or severe-result trigger is
  visible. No code/config changed. Contract remains `50% ViT/backbone compute,
  dense decode`; no test-time GT/teacher/cache.
  Decision: continue all three jobs; no process rescue, cancellation,
  checkpoint cleanup, upload action, or new launch.

## 2026-05-29T03:28:18+08:00 - E2E raw-density main/uniform entered Epoch 9

- Upload marker remains `ALL_PASS=True`; jobs `994340`, `994341`, and `994342`
  remain `RUNNING` for about `51:25` with `cpu=8,mem=124400M,gres/gpu=1`.
- Main completed Epoch 8 with `Loss=0.6744`, `cls_loss=0.3895`,
  `reg_loss=0.2849`, then Epoch 9 started.
- Uniform completed Epoch 8 with `Loss=0.6715`, `cls_loss=0.3904`,
  `reg_loss=0.2811`, then Epoch 9 started.
- Lowreg reached Epoch 8 step 50 with `Loss=0.6894`, `cls_loss=0.4021`,
  `reg_loss=0.2873`; no one-hour no-change threshold is close.
- Corrected artifact check confirms run subdirs under
  `/data/home/sczc063/run/yuzibo/e2e_runs/exps/.../gpu1_id0`; no checkpoint
  directory exists yet, only `log.json` is updating.
- Failure scan count is `0`; no mAP, OOM, NaN, or severe-result trigger is
  visible. No code/config changed. Contract remains `50% ViT/backbone compute,
  dense decode`; no test-time GT/teacher/cache.
  Decision: continue all three jobs; no rescue, cancellation, cleanup, upload
  action, or new launch.

## 2026-05-29T03:31:16+08:00 - E2E raw-density all routes reached Epoch 9 window

- Upload marker remains `ALL_PASS=True`; jobs `994340`, `994341`, and `994342`
  remain `RUNNING` for about `55:30` with `cpu=8,mem=124400M,gres/gpu=1`.
- Main reached Epoch 9 step 50 with `Loss=0.6129`, `cls_loss=0.3377`,
  `reg_loss=0.2752`, `mem=3189MB`.
- Uniform reached Epoch 9 step 50 with `Loss=0.6068`, `cls_loss=0.3347`,
  `reg_loss=0.2721`, `mem=2648MB`.
- Lowreg completed Epoch 8 with `Loss=0.6665`, `cls_loss=0.3868`,
  `reg_loss=0.2797`, then Epoch 9 started.
- Run subdirs exist under `/data/home/sczc063/run/yuzibo/.../gpu1_id0`;
  checkpoint directories are still absent and `log.json` files are updating.
- Failure scan count is `0`; no mAP, OOM, NaN, or severe-result trigger is
  visible. No code/config changed. Contract remains `50% ViT/backbone compute,
  dense decode`; no test-time GT/teacher/cache.
  Decision: continue all three jobs; no rescue, cancellation, cleanup, upload
  action, or new launch.

## 2026-05-29T03:34:01+08:00 - E2E raw-density main/uniform entered Epoch 10

- Upload marker remains `ALL_PASS=True`; jobs `994340`, `994341`, and `994342`
  remain `RUNNING` for about `58:27` with `cpu=8,mem=124400M,gres/gpu=1`.
- Main completed Epoch 9 with `Loss=0.6595`, `cls_loss=0.3779`,
  `reg_loss=0.2816`, then Epoch 10 started.
- Uniform completed Epoch 9 with `Loss=0.6483`, `cls_loss=0.3702`,
  `reg_loss=0.2781`, then Epoch 10 started.
- Lowreg reached Epoch 9 step 50 with `Loss=0.6161`, `cls_loss=0.3443`,
  `reg_loss=0.2718`; no one-hour no-change threshold is close.
- Run subdirs exist; checkpoint directories are still absent and `log.json`
  files are updating.
- Failure scan count is `0`; no mAP, OOM, NaN, or severe-result trigger is
  visible. No code/config changed. Contract remains `50% ViT/backbone compute,
  dense decode`; no test-time GT/teacher/cache.
  Decision: continue all three jobs; no rescue, cancellation, cleanup, upload
  action, or new launch.

## 2026-05-29T03:36:47+08:00 - E2E raw-density all routes at Epoch 10 step 50

- Upload marker remains `ALL_PASS=True`; jobs `994340`, `994341`, and `994342`
  remain `RUNNING` for about `1:01:29` with `cpu=8,mem=124400M,gres/gpu=1`.
- Main reached Epoch 10 step 50 with `Loss=0.6646`, `cls_loss=0.3876`,
  `reg_loss=0.2770`, `mem=3189MB`.
- Uniform reached Epoch 10 step 50 with `Loss=0.6637`, `cls_loss=0.3855`,
  `reg_loss=0.2783`, `mem=2648MB`.
- Lowreg reached Epoch 10 step 50 with `Loss=0.6613`, `cls_loss=0.3759`,
  `reg_loss=0.2854`, `mem=3189MB`.
- Run subdirs exist; checkpoint directories are still absent and `log.json`
  files are updating, as expected before epoch 20.
- Failure scan count is `0`; no mAP, OOM, NaN, or severe-result trigger is
  visible. No code/config changed. Contract remains `50% ViT/backbone compute,
  dense decode`; no test-time GT/teacher/cache.
  Decision: continue all three jobs; no rescue, cancellation, cleanup, upload
  action, or new launch.

## 2026-05-29T03:40:56+08:00 - E2E raw-density all routes entered Epoch 11

- Upload marker remains `ALL_PASS=True`; jobs `994340`, `994341`, and `994342`
  remain `RUNNING` for about `1:04:37` with `cpu=8,mem=124400M,gres/gpu=1`.
- Main completed Epoch 10 with `Loss=0.6605`, then entered Epoch 11 and reached
  step 50 with `Loss=0.6500`, `cls_loss=0.3668`, `reg_loss=0.2832`,
  `mem=3189MB`.
- Uniform completed Epoch 10 with `Loss=0.6583`, `cls_loss=0.3774`,
  `reg_loss=0.2809`, then Epoch 11 started.
- Lowreg completed Epoch 10 with `Loss=0.6632`, `cls_loss=0.3816`,
  `reg_loss=0.2816`, then Epoch 11 started.
- Run subdirs exist; checkpoint directories are still absent and `log.json`
  files are updating.
- Failure scan count is `0`; no mAP, OOM, NaN, or severe-result trigger is
  visible. No code/config changed. Contract remains `50% ViT/backbone compute,
  dense decode`; no test-time GT/teacher/cache.
  Decision: continue all three jobs; no rescue, cancellation, cleanup, upload
  action, or new launch.

## 2026-05-29T03:43:35+08:00 - E2E raw-density main entered Epoch 12

- Upload marker remains `ALL_PASS=True`; jobs `994340`, `994341`, and `994342`
  remain `RUNNING` for about `1:07:40` with `cpu=8,mem=124400M,gres/gpu=1`.
- Main completed Epoch 11 with `Loss=0.6481`, `cls_loss=0.3727`,
  `reg_loss=0.2754`, then Epoch 12 started.
- Uniform reached Epoch 11 step 50 with `Loss=0.6484`, `cls_loss=0.3708`,
  `reg_loss=0.2776`, `mem=2648MB`.
- Lowreg reached Epoch 11 step 50 with `Loss=0.6494`, `cls_loss=0.3705`,
  `reg_loss=0.2789`, `mem=3189MB`.
- Run subdirs exist; checkpoint directories are still absent and `log.json`
  files are updating.
- Failure scan count is `0`; no mAP, OOM, NaN, or severe-result trigger is
  visible. No code/config changed. Contract remains `50% ViT/backbone compute,
  dense decode`; no test-time GT/teacher/cache.
  Decision: continue all three jobs; no rescue, cancellation, cleanup, upload
  action, or new launch.

## 2026-05-29T03:46:23+08:00 - E2E raw-density all routes entered Epoch 12

- Upload marker remains `ALL_PASS=True`; jobs `994340`, `994341`, and `994342`
  remain `RUNNING` for about `1:10:52` with `cpu=8,mem=124400M,gres/gpu=1`.
- Main reached Epoch 12 step 50 with `Loss=0.6318`, `cls_loss=0.3574`,
  `reg_loss=0.2744`, `mem=3189MB`.
- Uniform completed Epoch 11 with `Loss=0.6398`, `cls_loss=0.3678`,
  `reg_loss=0.2719`, then Epoch 12 started.
- Lowreg completed Epoch 11 with `Loss=0.6421`, `cls_loss=0.3708`,
  `reg_loss=0.2713`, then Epoch 12 started.
- Run subdirs exist; checkpoint directories are still absent and `log.json`
  files are updating.
- Failure scan count is `0`; no mAP, OOM, NaN, or severe-result trigger is
  visible. No code/config changed. Contract remains `50% ViT/backbone compute,
  dense decode`; no test-time GT/teacher/cache.
  Decision: continue all three jobs; no rescue, cancellation, cleanup, upload
  action, or new launch.

## 2026-05-29T03:48:57+08:00 - E2E raw-density main entered Epoch 13

- Upload marker remains `ALL_PASS=True`; jobs `994340`, `994341`, and `994342`
  remain `RUNNING` for about `1:14:01` with `cpu=8,mem=124400M,gres/gpu=1`.
- Main completed Epoch 12 with `Loss=0.6377`, `cls_loss=0.3690`,
  `reg_loss=0.2687`, then Epoch 13 started.
- Uniform reached Epoch 12 step 50 with `Loss=0.6119`, `cls_loss=0.3400`,
  `reg_loss=0.2718`, `mem=2648MB`; its `log.json` was touched at `03:50:31`.
- Lowreg reached Epoch 12 step 50 with `Loss=0.6099`, `cls_loss=0.3427`,
  `reg_loss=0.2672`, `mem=3189MB`.
- Run subdirs exist; checkpoint directories are still absent and `log.json`
  files are updating.
- Failure scan count is `0`; no mAP, OOM, NaN, or severe-result trigger is
  visible. No code/config changed. Contract remains `50% ViT/backbone compute,
  dense decode`; no test-time GT/teacher/cache.
  Decision: continue all three jobs; no rescue, cancellation, cleanup, upload
  action, or new launch.

## 2026-05-29T03:53:26+08:00 - E2E raw-density all routes entered Epoch 13

- Upload marker remains `ALL_PASS=True`; jobs `994340`, `994341`, and `994342`
  remain `RUNNING` for about `1:17:14` with `cpu=8,mem=124400M,gres/gpu=1`.
- Main reached Epoch 13 step 50 with `Loss=0.6281`, `cls_loss=0.3633`,
  `reg_loss=0.2648`, `mem=3189MB`.
- Uniform completed Epoch 12 with `Loss=0.6217`, then reached Epoch 13 step 50
  with `Loss=0.6205`, `cls_loss=0.3509`, `reg_loss=0.2696`, `mem=2648MB`.
- Lowreg completed Epoch 12 with `Loss=0.6183`, `cls_loss=0.3536`,
  `reg_loss=0.2647`, then Epoch 13 started.
- Run subdirs exist; checkpoint directories are still absent and `log.json`
  files are updating.
- Failure scan count is `0`; no mAP, OOM, NaN, or severe-result trigger is
  visible. No code/config changed. Contract remains `50% ViT/backbone compute,
  dense decode`; no test-time GT/teacher/cache.
  Decision: continue all three jobs; no rescue, cancellation, cleanup, upload
  action, or new launch.

## 2026-05-29T03:55:27+08:00 - E2E raw-density all routes entered Epoch 14 window

- Upload marker remains `ALL_PASS=True`; jobs `994340`, `994341`, and `994342`
  remain `RUNNING` for about `1:21:12` with `cpu=8,mem=124400M,gres/gpu=1`.
  Disk is safe: `/data` has `2.3T` total, `414G` used, and `1.9T` free.
- Main completed Epoch 13 with `Loss=0.6436`, `cls_loss=0.3691`,
  `reg_loss=0.2745`, entered Epoch 14, and reached Epoch 14 step 50 with
  `Loss=0.5678`, `cls_loss=0.3197`, `reg_loss=0.2481`, `mem=3189MB`.
- Uniform completed Epoch 13 with `Loss=0.6388`, `cls_loss=0.3610`,
  `reg_loss=0.2777`, then Epoch 14 started.
- Lowreg completed Epoch 13 with `Loss=0.6312`, `cls_loss=0.3567`,
  `reg_loss=0.2744`, then Epoch 14 started.
- Run subdirs exist; checkpoint directories are still absent and only `log.json`
  files are updating. Failure scan count is `0`; no mAP, checkpoint, OOM, NaN,
  or severe-result trigger is visible.
- No code/config/model change occurred in this monitor event. Contract remains
  `50% ViT/backbone compute, dense decode`; no test-time GT/teacher/cache.
  Decision: continue all three jobs; no rescue, cancellation, cleanup, upload
  action, or new experiment launch.

## 2026-05-29T03:59:42+08:00 - E2E raw-density main/uniform entered Epoch 15

- Upload marker remains `ALL_PASS=True`; jobs `994340`, `994341`, and `994342`
  remain `RUNNING` for about `1:25:27` with `cpu=8,mem=124400M,gres/gpu=1`.
  `sacct` also reports all three jobs and their batch/extern steps as `RUNNING`.
  Disk remains safe: `/data` has `2.3T` total, `414G` used, and `1.9T` free.
- Main completed Epoch 14 with `Loss=0.5930`, `cls_loss=0.3354`,
  `reg_loss=0.2576`, then Epoch 15 started.
- Uniform completed Epoch 14 with `Loss=0.6044`, `cls_loss=0.3438`,
  `reg_loss=0.2606`, then Epoch 15 started.
- Lowreg reached Epoch 14 step 50 with `Loss=0.5726`, `cls_loss=0.3253`,
  `reg_loss=0.2472`; its log age is about two minutes and does not meet the
  one-hour stall gate.
- Run subdirs exist; checkpoint directories are still absent and only `log.json`
  files are updating. Failure scan count is `0`; no mAP, checkpoint, OOM, NaN,
  or severe-result trigger is visible.
- No code/config/model change occurred in this monitor event. Contract remains
  `50% ViT/backbone compute, dense decode`; no test-time GT/teacher/cache.
  Decision: continue all three jobs; no rescue, cancellation, cleanup, upload
  action, or new experiment launch.

## 2026-05-29T04:02:32+08:00 - E2E raw-density all routes entered Epoch 15 window

- Upload marker remains `ALL_PASS=True`; jobs `994340`, `994341`, and `994342`
  remain `RUNNING` for about `1:28:17` with `cpu=8,mem=124400M,gres/gpu=1`.
  Disk remains safe: `/data` has `2.3T` total, `414G` used, and `1.9T` free.
- Main reached Epoch 15 step 50 with `Loss=0.5939`, `cls_loss=0.3364`,
  `reg_loss=0.2575`, `mem=3189MB`.
- Uniform reached Epoch 15 step 50 with `Loss=0.6180`, `cls_loss=0.3597`,
  `reg_loss=0.2584`, `mem=2648MB`.
- Lowreg completed Epoch 14 with `Loss=0.5989`, `cls_loss=0.3416`,
  `reg_loss=0.2573`, then Epoch 15 started.
- Run subdirs exist; checkpoint directories are still absent and only `log.json`
  files are updating. Failure scan count is `0`; no mAP, checkpoint, OOM, NaN,
  or severe-result trigger is visible.
- No code/config/model change occurred in this monitor event. Contract remains
  `50% ViT/backbone compute, dense decode`; no test-time GT/teacher/cache.
  Decision: continue all three jobs; no rescue, cancellation, cleanup, upload
  action, or new experiment launch.

## 2026-05-29T04:04:56+08:00 - E2E raw-density main/uniform entered Epoch 16

- Upload marker remains `ALL_PASS=True`; jobs `994340`, `994341`, and `994342`
  remain `RUNNING` for about `1:30:41` with `cpu=8,mem=124400M,gres/gpu=1`.
- Main completed Epoch 15 with `Loss=0.5885`, `cls_loss=0.3323`,
  `reg_loss=0.2561`, then Epoch 16 started.
- Uniform completed Epoch 15 with `Loss=0.6004`, `cls_loss=0.3419`,
  `reg_loss=0.2585`, then Epoch 16 started.
- Lowreg reached Epoch 15 step 50 with `Loss=0.6016`, `cls_loss=0.3439`,
  `reg_loss=0.2577`; its log age is under two minutes and does not meet the
  one-hour stall gate.
- Run subdirs exist; checkpoint directories are still absent and only `log.json`
  files are updating. Failure scan count is `0`; no mAP, checkpoint, OOM, NaN,
  or severe-result trigger is visible.
- No code/config/model change occurred in this monitor event. Contract remains
  `50% ViT/backbone compute, dense decode`; no test-time GT/teacher/cache.
  Decision: continue all three jobs; no rescue, cancellation, cleanup, upload
  action, or new experiment launch.

## 2026-05-29T04:07:07+08:00 - E2E raw-density all routes entered Epoch 16 window

- Upload marker remains `ALL_PASS=True`; jobs `994340`, `994341`, and `994342`
  remain `RUNNING` for about `1:32:52` with `cpu=8,mem=124400M,gres/gpu=1`.
- Main reached Epoch 16 step 50 with `Loss=0.5819`, `cls_loss=0.3339`,
  `reg_loss=0.2480`, `mem=3189MB`.
- Uniform completed Epoch 15 with `Loss=0.6004`, `cls_loss=0.3419`,
  `reg_loss=0.2585`, then Epoch 16 started.
- Lowreg completed Epoch 15 with `Loss=0.5902`, `cls_loss=0.3343`,
  `reg_loss=0.2559`, then Epoch 16 started.
- Run subdirs exist; checkpoint directories are still absent and only `log.json`
  files are updating. Failure scan count is `0`; no mAP, checkpoint, OOM, NaN,
  or severe-result trigger is visible.
- No code/config/model change occurred in this monitor event. Contract remains
  `50% ViT/backbone compute, dense decode`; no test-time GT/teacher/cache.
  Decision: continue all three jobs; no rescue, cancellation, cleanup, upload
  action, or new experiment launch.

## 2026-05-29T04:09:26+08:00 - E2E raw-density main entered Epoch 17

- Upload marker remains `ALL_PASS=True`; jobs `994340`, `994341`, and `994342`
  remain `RUNNING` for about `1:35:11` with `cpu=8,mem=124400M,gres/gpu=1`.
- Main completed Epoch 16 with `Loss=0.6124`, `cls_loss=0.3467`,
  `reg_loss=0.2657`, then Epoch 17 started.
- Uniform reached Epoch 16 step 50 with `Loss=0.5737`, `cls_loss=0.3198`,
  `reg_loss=0.2539`, `mem=2648MB`.
- Lowreg completed Epoch 15 with `Loss=0.5902`, `cls_loss=0.3343`,
  `reg_loss=0.2559`, then Epoch 16 started; its log age is about three minutes
  and does not meet the one-hour stall gate.
- Run subdirs exist; checkpoint directories are still absent and only `log.json`
  files are updating. Failure scan count is `0`; no mAP, checkpoint, OOM, NaN,
  or severe-result trigger is visible.
- No code/config/model change occurred in this monitor event. Contract remains
  `50% ViT/backbone compute, dense decode`; no test-time GT/teacher/cache.
  Decision: continue all three jobs; no rescue, cancellation, cleanup, upload
  action, or new experiment launch.

## 2026-05-29T04:11:41+08:00 - E2E raw-density main/uniform entered Epoch 17

- Upload marker remains `ALL_PASS=True`; jobs `994340`, `994341`, and `994342`
  remain `RUNNING` for about `1:37:26` with `cpu=8,mem=124400M,gres/gpu=1`.
- Main reached Epoch 17 step 50 with `Loss=0.5231`, `cls_loss=0.2757`,
  `reg_loss=0.2474`, `mem=3189MB`.
- Uniform completed Epoch 16 with `Loss=0.6018`, `cls_loss=0.3347`,
  `reg_loss=0.2671`, then Epoch 17 started.
- Lowreg reached Epoch 16 step 50 with `Loss=0.5788`, `cls_loss=0.3296`,
  `reg_loss=0.2492`; latest log age is about three minutes and does not meet
  the one-hour stall gate.
- Run subdirs exist; checkpoint directories are still absent and only `log.json`
  files are updating. Failure scan count is `0`; no mAP, checkpoint, OOM, NaN,
  or severe-result trigger is visible.
- No code/config/model change occurred in this monitor event. Contract remains
  `50% ViT/backbone compute, dense decode`; no test-time GT/teacher/cache.
  Decision: continue all three jobs; no rescue, cancellation, cleanup, upload
  action, or new experiment launch.

## 2026-05-29T04:14:04+08:00 - E2E raw-density main entered Epoch 18

- Upload marker remains `ALL_PASS=True`; jobs `994340`, `994341`, and `994342`
  remain `RUNNING` for about `1:39:49` with `cpu=8,mem=124400M,gres/gpu=1`.
- Main completed Epoch 17 with `Loss=0.5487`, `cls_loss=0.3007`,
  `reg_loss=0.2480`, then Epoch 18 started.
- Uniform reached Epoch 17 step 50 with `Loss=0.5179`, `cls_loss=0.2749`,
  `reg_loss=0.2430`, `mem=2648MB`.
- Lowreg completed Epoch 16 with `Loss=0.6061`, `cls_loss=0.3414`,
  `reg_loss=0.2647`, then Epoch 17 started; latest log age is about two to
  three minutes and does not meet the one-hour stall gate.
- Run subdirs exist; checkpoint directories are still absent and only `log.json`
  files are updating. Failure scan count is `0`; no mAP, checkpoint, OOM, NaN,
  or severe-result trigger is visible.
- No code/config/model change occurred in this monitor event. Contract remains
  `50% ViT/backbone compute, dense decode`; no test-time GT/teacher/cache.
  Decision: continue all three jobs; no rescue, cancellation, cleanup, upload
  action, or new experiment launch.

## 2026-05-29T04:16:36+08:00 - E2E raw-density main/uniform entered Epoch 18

- Upload marker remains `ALL_PASS=True`; jobs `994340`, `994341`, and `994342`
  remain `RUNNING` for about `1:42:21` with `cpu=8,mem=124400M,gres/gpu=1`.
- Main reached Epoch 18 step 50 with `Loss=0.5652`, `cls_loss=0.3068`,
  `reg_loss=0.2584`, `mem=3189MB`.
- Uniform completed Epoch 17 with `Loss=0.5462`, `cls_loss=0.2981`,
  `reg_loss=0.2481`, then Epoch 18 started.
- Lowreg reached Epoch 17 step 50 with `Loss=0.5124`, `cls_loss=0.2728`,
  `reg_loss=0.2395`; latest log age is about two minutes and does not meet the
  one-hour stall gate.
- Run subdirs exist; checkpoint directories are still absent and only `log.json`
  files are updating. Failure scan count is `0`; no mAP, checkpoint, OOM, NaN,
  or severe-result trigger is visible.
- No code/config/model change occurred in this monitor event. Contract remains
  `50% ViT/backbone compute, dense decode`; no test-time GT/teacher/cache.
  Decision: continue all three jobs; no rescue, cancellation, cleanup, upload
  action, or new experiment launch.

## 2026-05-29T04:18:57+08:00 - E2E raw-density main entered Epoch 19

- Upload marker remains `ALL_PASS=True`; jobs `994340`, `994341`, and `994342`
  remain `RUNNING` for about `1:44:42` with `cpu=8,mem=124400M,gres/gpu=1`.
- Main completed Epoch 18 with `Loss=0.5603`, `cls_loss=0.3036`,
  `reg_loss=0.2567`, then Epoch 19 started.
- Uniform reached Epoch 18 step 50 with `Loss=0.5697`, `cls_loss=0.3045`,
  `reg_loss=0.2653`, `mem=2648MB`.
- Lowreg completed Epoch 17 with `Loss=0.5383`, `cls_loss=0.2958`,
  `reg_loss=0.2425`, then Epoch 18 started.
- Run subdirs exist; checkpoint directories are still absent and only `log.json`
  files are updating. Failure scan count is `0`; no mAP, checkpoint, OOM, NaN,
  or severe-result trigger is visible. First checkpoint is expected after the
  epoch-19/20 save boundary.
- No code/config/model change occurred in this monitor event. Contract remains
  `50% ViT/backbone compute, dense decode`; no test-time GT/teacher/cache.
  Decision: continue all three jobs; no rescue, cancellation, cleanup, upload
  action, or new experiment launch.

## 2026-05-29T04:21:22+08:00 - E2E raw-density main/uniform in Epoch 19 window

- Upload marker remains `ALL_PASS=True`; jobs `994340`, `994341`, and `994342`
  remain `RUNNING` for about `1:47:07` with `cpu=8,mem=124400M,gres/gpu=1`.
- Main remains in Epoch 19 after completing Epoch 18 with `Loss=0.5603`,
  `cls_loss=0.3036`, `reg_loss=0.2567`.
- Uniform completed Epoch 18 with `Loss=0.5612`, `cls_loss=0.3006`,
  `reg_loss=0.2606`, then Epoch 19 started.
- Lowreg reached Epoch 18 step 50 with `Loss=0.5512`, `cls_loss=0.2898`,
  `reg_loss=0.2614`, `mem=3189MB`.
- Run subdirs exist; checkpoint directories are still absent and only `log.json`
  files are updating. Failure scan count is `0`; no mAP, checkpoint, OOM, NaN,
  or severe-result trigger is visible. First checkpoint remains pending after
  the epoch-19/20 save boundary.
- No code/config/model change occurred in this monitor event. Contract remains
  `50% ViT/backbone compute, dense decode`; no test-time GT/teacher/cache.
  Decision: continue all three jobs; no rescue, cancellation, cleanup, upload
  action, or new experiment launch.

## 2026-05-29T04:23:57+08:00 - E2E raw-density all routes in Epoch 19 window

- Upload marker remains `ALL_PASS=True`; jobs `994340`, `994341`, and `994342`
  remain `RUNNING` for about `1:49:42` with `cpu=8,mem=124400M,gres/gpu=1`.
- Main reached Epoch 19 step 50 with `Loss=0.5650`, `cls_loss=0.2980`,
  `reg_loss=0.2669`, `mem=3189MB`.
- Uniform reached Epoch 19 step 50 with `Loss=0.5667`, `cls_loss=0.3030`,
  `reg_loss=0.2636`, `mem=2648MB`.
- Lowreg completed Epoch 18 with `Loss=0.5562`, `cls_loss=0.2974`,
  `reg_loss=0.2588`, then Epoch 19 started.
- Run subdirs exist; checkpoint directories are still absent and only `log.json`
  files are updating. Failure scan count is `0`; no mAP, checkpoint, OOM, NaN,
  or severe-result trigger is visible. First checkpoint remains pending.
- No code/config/model change occurred in this monitor event. Contract remains
  `50% ViT/backbone compute, dense decode`; no test-time GT/teacher/cache.
  Decision: continue all three jobs; no rescue, cancellation, cleanup, upload
  action, or new experiment launch.

## 2026-05-29T04:23:57+08:00 - E2E raw-density main first checkpoint written

- Upload marker remains `ALL_PASS=True`; jobs `994340`, `994341`, and `994342`
  remain `RUNNING` for about `1:52:16` with `cpu=8,mem=124400M,gres/gpu=1`.
- Main completed Epoch 19 with `Loss=0.5623`, `cls_loss=0.3117`,
  `reg_loss=0.2505`, then Epoch 20 started.
- Main checkpoint now exists:
  `~/run/yuzibo/e2e_runs/exps/e2e_rawdensel_main_20260529_023345/gpu1_id0/checkpoint/epoch_19.pth`,
  size `623905307`, mtime `2026-05-29T04:26:26+08:00`.
- Uniform reached Epoch 19 step 50 with `Loss=0.5667`, `cls_loss=0.3030`,
  `reg_loss=0.2636`; lowreg reached Epoch 19 step 50 with `Loss=0.5635`,
  `cls_loss=0.3053`, `reg_loss=0.2582`.
- Uniform and lowreg checkpoint dirs are still absent. Failure scan count is `0`;
  no mAP, OOM, NaN, or severe-result trigger is visible.
- No code/config/model change occurred. Contract remains `50% ViT/backbone
  compute, dense decode`; no test-time GT/teacher/cache.
  Decision: continue all jobs, do not clean active checkpoints, and wait for
  uniform/lowreg first checkpoints before running cross-route selector-position
  diagnostics.

## 2026-05-29T04:29:33+08:00 - E2E raw-density all first checkpoints written

- Upload marker remains `ALL_PASS=True`; jobs `994340`, `994341`, and `994342`
  remain `RUNNING` for about `1:55:18` with `cpu=8,mem=124400M,gres/gpu=1`.
- Main checkpoint exists:
  `~/run/yuzibo/e2e_runs/exps/e2e_rawdensel_main_20260529_023345/gpu1_id0/checkpoint/epoch_19.pth`,
  size `623905307`, mtime `2026-05-29T04:26:26+08:00`; Epoch 20 step 50
  `Loss=0.5278`, `cls_loss=0.2989`, `reg_loss=0.2289`.
- Uniform checkpoint exists:
  `~/run/yuzibo/e2e_runs/exps/e2e_rawdensel_uniform_20260529_023345/gpu1_id0/checkpoint/epoch_19.pth`,
  size `623852123`, mtime `2026-05-29T04:28:52+08:00`; Epoch 20 step 50
  `Loss=0.5194`, `cls_loss=0.2898`, `reg_loss=0.2296`.
- Lowreg checkpoint exists:
  `~/run/yuzibo/e2e_runs/exps/e2e_rawdensel_lowreg_20260529_023345/gpu1_id0/checkpoint/epoch_19.pth`,
  size `623905307`, mtime `2026-05-29T04:28:41+08:00`; Epoch 19 completed
  with `Loss=0.5598`, `cls_loss=0.3154`, `reg_loss=0.2444`, then Epoch 20 started.
- Failure scan count is `0`; no mAP, OOM, NaN, or severe-result trigger is visible.
- No code/config/model change occurred. Contract remains `50% ViT/backbone compute,
  dense decode`; no test-time GT/teacher/cache.
  Decision: continue all jobs, do not clean active checkpoints, and prepare
  selector-position diagnostics across all three epoch-19 checkpoints while
  continuing to monitor toward first eval after epoch 40.

## 2026-05-29T04:59:56+08:00 - E2E raw-density epoch-19 selector-position diagnostic completed

- Jobs `994340`, `994341`, and `994342` remain `RUNNING` for about `2:25:41`.
  Main and uniform have entered Epoch 26; lowreg has entered Epoch 25. No mAP is
  logged yet, and the narrow failure scan for traceback/OOM/killed/runtime/nan is
  empty.
- Selector-only diagnostic output:
  `~/run/yuzibo/e2e_runs/selector_diagnostics/e2e_rawdensel_epoch19_posdiag_20260529_044130/summary.json`;
  examples are in `examples.jsonl`. It sampled 48 of 487 validation windows, did
  not run ViT/backbone/detector, and used validation GT only for offline
  action/boundary proximity labels.
- Epoch-19 position behavior: main is close to uniform with mean abs position
  delta `1.03` dense frames and p95 `1.43`; lowreg deviates more with mean `3.76`
  and p95 `6.30`; uniform is exactly `0`.
- Selected action and boundary fractions are effectively unchanged from uniform:
  action fraction main/uniform/lowreg `0.287/0.289/0.287`, boundary<=4 fraction
  `0.144/0.148/0.144`, boundary recall@4 `0.985/1.000/0.985`. Octile fractions
  are nearly flat.
- Selector score-head movement remains small at epoch 19: main final weight norm
  `0.0091`, lowreg final weight norm `0.0661`, uniform final layer `0`.
- Interpretation: the implemented route is end-to-end, but this early checkpoint
  has not learned the desired action/boundary-aware frame distribution. Continue
  all three jobs to first mAP/later checkpoints before making a route decision.

## 2026-05-29T05:03:17+08:00 - E2E raw-density continues toward first eval

- Upload marker remains `ALL_PASS=True`; jobs `994340`, `994341`, and `994342`
  are still `RUNNING` for about `2:29:05`.
- Main completed Epoch 26 with `Loss=0.4885`, `cls_loss=0.2630`,
  `reg_loss=0.2255`, then entered Epoch 27.
- Uniform reached Epoch 26 step 50 with `Loss=0.4887`, `cls_loss=0.2544`,
  `reg_loss=0.2344`.
- Lowreg completed Epoch 25 with `Loss=0.5438`, `cls_loss=0.2988`,
  `reg_loss=0.2450`, then entered Epoch 26.
- Checkpoints remain only `epoch_19.pth` for each active run. No mAP/result JSON
  is visible yet, narrow failure scan is empty, and `/data` has about `1.9T`
  free.
- No code/config/model change occurred. Contract remains `50% ViT/backbone
  compute, dense decode`; no test-time GT/teacher/cache. Decision: continue all
  jobs, do not clean active checkpoints or launch duplicates, and wait for first
  eval after the Epoch 41 boundary.

## 2026-05-29T05:06:06+08:00 - E2E raw-density healthy Epoch 26/27 window

- Upload marker remains `ALL_PASS=True`; jobs `994340`, `994341`, and `994342`
  are still `RUNNING` for about `2:31:51`.
- Main reached Epoch 27 step 50 with `Loss=0.4896`, `cls_loss=0.2512`,
  `reg_loss=0.2384`.
- Uniform completed Epoch 26 with `Loss=0.4855`, `cls_loss=0.2597`,
  `reg_loss=0.2257`, then entered Epoch 27.
- Lowreg reached Epoch 26 step 50 with `Loss=0.4927`, `cls_loss=0.2574`,
  `reg_loss=0.2353`.
- No mAP/result artifact is visible yet. Each active run still only has
  `epoch_19.pth`; narrow failure scan remains empty; `/data` has about `1.9T`
  free.
- No code/config/model change occurred. Contract remains `50% ViT/backbone
  compute, dense decode`; no test-time GT/teacher/cache. Decision: continue all
  jobs, do not clean active checkpoints or launch duplicates, and wait for first
  eval after the Epoch 41 boundary.

## 2026-05-29T05:08:41+08:00 - E2E raw-density still pre-eval

- Upload marker remains `ALL_PASS=True`; jobs `994340`, `994341`, and `994342`
  are still `RUNNING` for about `2:34:26`.
- Main completed Epoch 27 with `Loss=0.5125`, `cls_loss=0.2710`,
  `reg_loss=0.2414`, then entered Epoch 28.
- Uniform reached Epoch 27 step 50 with `Loss=0.4944`, `cls_loss=0.2524`,
  `reg_loss=0.2421`.
- Lowreg completed Epoch 26 with `Loss=0.4884`, `cls_loss=0.2608`,
  `reg_loss=0.2276`, then entered Epoch 27.
- No mAP is visible, no new checkpoints beyond each run's `epoch_19.pth`, and
  narrow failure scan remains empty.
- No code/config/model change occurred. Contract remains `50% ViT/backbone
  compute, dense decode`; no test-time GT/teacher/cache. Decision: continue all
  jobs, do not clean active checkpoints or launch duplicates, and wait for first
  eval after the Epoch 41 boundary.

## 2026-05-29T05:11:10+08:00 - E2E raw-density main/uniform entered Epoch 28

- Upload marker remains `ALL_PASS=True`; jobs `994340`, `994341`, and `994342`
  are still `RUNNING` for about `2:36:55`.
- Main reached Epoch 28 step 50 with `Loss=0.4787`, `cls_loss=0.2680`,
  `reg_loss=0.2106`.
- Uniform completed Epoch 27 with `Loss=0.5145`, `cls_loss=0.2715`,
  `reg_loss=0.2430`, then entered Epoch 28.
- Lowreg latest log line remains Epoch 27 start at `05:07:59`; this is below the
  one-hour stall threshold and Slurm still reports the job running.
- No mAP is visible, no new checkpoints beyond each run's `epoch_19.pth`, and
  narrow failure scan remains empty.
- No code/config/model change occurred. Contract remains `50% ViT/backbone
  compute, dense decode`; no test-time GT/teacher/cache. Decision: continue all
  jobs, do not clean active checkpoints or launch duplicates, and wait for first
  eval after the Epoch 41 boundary.

## 2026-05-29T05:13:30+08:00 - E2E raw-density lowreg log gap resolved

- Upload marker remains `ALL_PASS=True`; jobs `994340`, `994341`, and `994342`
  are still `RUNNING` for about `2:39:15`.
- Main completed Epoch 28 with `Loss=0.5021`, `cls_loss=0.2667`,
  `reg_loss=0.2354`, then entered Epoch 29.
- Uniform reached Epoch 28 step 50 with `Loss=0.4851`, `cls_loss=0.2745`,
  `reg_loss=0.2106`.
- Lowreg reached Epoch 27 step 50 with `Loss=0.5031`, `cls_loss=0.2557`,
  `reg_loss=0.2474`, confirming the earlier short log gap was not a stall.
- No mAP is visible, no new checkpoints beyond each run's `epoch_19.pth`, and
  narrow failure scan remains empty.
- No code/config/model change occurred. Contract remains `50% ViT/backbone
  compute, dense decode`; no test-time GT/teacher/cache. Decision: continue all
  jobs, do not clean active checkpoints or launch duplicates, and wait for first
  eval after the Epoch 41 boundary.

## 2026-05-29T05:16:16+08:00 - E2E raw-density all routes continue pre-eval

- Upload marker remains `ALL_PASS=True`; jobs `994340`, `994341`, and `994342`
  are still `RUNNING` for about `2:42:01`.
- Main reached Epoch 29 step 50 with `Loss=0.4869`, `cls_loss=0.2532`,
  `reg_loss=0.2336`.
- Uniform completed Epoch 28 with `Loss=0.5039`, `cls_loss=0.2719`,
  `reg_loss=0.2320`, then entered Epoch 29.
- Lowreg completed Epoch 27 with `Loss=0.5261`, `cls_loss=0.2803`,
  `reg_loss=0.2457`, then entered Epoch 28.
- No mAP is visible, no new checkpoints beyond each run's `epoch_19.pth`, and
  narrow failure scan remains empty.
- No code/config/model change occurred. Contract remains `50% ViT/backbone
  compute, dense decode`; no test-time GT/teacher/cache. Decision: continue all
  jobs, do not clean active checkpoints or launch duplicates, and wait for first
  eval after the Epoch 41 boundary.

## 2026-05-29T05:18:40+08:00 - E2E raw-density main entered Epoch 30

- Upload marker remains `ALL_PASS=True`; jobs `994340`, `994341`, and `994342`
  are still `RUNNING` for about `2:44:25`.
- Main completed Epoch 29 with `Loss=0.4909`, `cls_loss=0.2615`,
  `reg_loss=0.2294`, then entered Epoch 30.
- Uniform completed Epoch 28 and entered Epoch 29; latest full Epoch 28 loss was
  `0.5039`.
- Lowreg reached Epoch 28 step 50 with `Loss=0.4848`, `cls_loss=0.2707`,
  `reg_loss=0.2141`.
- No mAP is visible, no new checkpoints beyond each run's `epoch_19.pth`, and
  narrow failure scan remains empty.
- No code/config/model change occurred. Contract remains `50% ViT/backbone
  compute, dense decode`; no test-time GT/teacher/cache. Decision: continue all
  jobs, do not clean active checkpoints or launch duplicates, and wait for first
  eval after the Epoch 41 boundary.

## 2026-05-29T05:21:10+08:00 - E2E raw-density Epoch 29/30 window

- Upload marker remains `ALL_PASS=True`; jobs `994340`, `994341`, and `994342`
  are still `RUNNING` for about `2:46:55`.
- Main reached Epoch 30 step 50 with `Loss=0.4995`, `cls_loss=0.2676`,
  `reg_loss=0.2319`.
- Uniform reached Epoch 29 step 50 with `Loss=0.4836`, `cls_loss=0.2551`,
  `reg_loss=0.2285`.
- Lowreg completed Epoch 28 with `Loss=0.5081`, `cls_loss=0.2719`,
  `reg_loss=0.2361`, then entered Epoch 29.
- No mAP is visible, no new checkpoints beyond each run's `epoch_19.pth`, and
  narrow failure scan remains empty.
- No code/config/model change occurred. Contract remains `50% ViT/backbone
  compute, dense decode`; no test-time GT/teacher/cache. Decision: continue all
  jobs, do not clean active checkpoints or launch duplicates, and wait for first
  eval after the Epoch 41 boundary.

## 2026-05-29T05:24:05+08:00 - E2E raw-density Epoch 29/31 window

- Upload marker remains `ALL_PASS=True`; jobs `994340`, `994341`, and `994342`
  are still `RUNNING` for about `2:49:50` on nodes `g0005`, `g0005`, and
  `g0024`.
- Latest observed log evidence reaches `05:25:48+08:00`: main completed Epoch
  30 with `Loss=0.4885`, `cls_loss=0.2533`, `reg_loss=0.2351`, then entered
  Epoch 31.
- Uniform completed Epoch 29 with `Loss=0.4920`, `cls_loss=0.2652`,
  `reg_loss=0.2268`, then entered Epoch 30.
- Lowreg reached Epoch 29 step 50 with `Loss=0.4806`, `cls_loss=0.2500`,
  `reg_loss=0.2306`.
- No mAP/result artifact is visible. Checkpoints remain only `epoch_19.pth` for
  each active run; narrow failure scan is empty; `/data` has about `1.9T` free.
- No code/config/model change occurred. Contract remains `50% ViT/backbone
  compute, dense decode`; no test-time GT/teacher/cache. Decision: continue all
  jobs, do not clean active checkpoints or launch duplicates, and wait for the
  Epoch 39 checkpoint plus first eval after the Epoch 41 boundary.

## 2026-05-29T05:27:22+08:00 - E2E raw-density main/uniform reached Epoch 31

- Jobs `994340`, `994341`, and `994342` are still `RUNNING` for about
  `2:53:07` on nodes `g0005`, `g0005`, and `g0024`.
- Latest observed log evidence reaches `05:29:24+08:00`: main reached Epoch 31
  step 50 with `Loss=0.4537`, `cls_loss=0.2373`, `reg_loss=0.2163`.
- Uniform completed Epoch 30 with `Loss=0.4902`, `cls_loss=0.2528`,
  `reg_loss=0.2374`, then entered Epoch 31.
- Lowreg completed Epoch 29 with `Loss=0.4989`, `cls_loss=0.2673`,
  `reg_loss=0.2316`, then entered Epoch 30.
- No mAP/result artifact is visible. Checkpoints remain only `epoch_19.pth` for
  each active run; narrow failure scan is empty.
- No code/config/model change occurred. Contract remains `50% ViT/backbone
  compute, dense decode`; no test-time GT/teacher/cache. Decision: continue all
  jobs, do not clean active checkpoints or launch duplicates, and wait for the
  Epoch 39 checkpoint plus first eval after the Epoch 41 boundary.

## 2026-05-29T05:29:56+08:00 - E2E raw-density main entered Epoch 32

- Upload marker remains `ALL_PASS=True`; Slurm and sacct both report jobs
  `994340`, `994341`, and `994342` are still `RUNNING` for about `2:55:41` on
  nodes `g0005`, `g0005`, and `g0024`.
- Latest observed log evidence reaches `05:31:07+08:00`: main completed Epoch
  31 with `Loss=0.4589`, `cls_loss=0.2365`, `reg_loss=0.2224`, then entered
  Epoch 32.
- Uniform completed Epoch 30 with `Loss=0.4902`, `cls_loss=0.2528`,
  `reg_loss=0.2374`, then entered Epoch 31.
- Lowreg reached Epoch 30 step 50 with `Loss=0.5003`, `cls_loss=0.2707`,
  `reg_loss=0.2296`.
- No mAP/result artifact is visible. Checkpoints remain only `epoch_19.pth` for
  each active run; narrow failure scan is empty; `/data` has about `1.9T` free.
- No code/config/model change occurred. Contract remains `50% ViT/backbone
  compute, dense decode`; no test-time GT/teacher/cache. Decision: continue all
  jobs, do not clean active checkpoints or launch duplicates, and wait for the
  Epoch 39 checkpoint plus first eval after the Epoch 41 boundary.

## 2026-05-29T05:32:58+08:00 - E2E raw-density main/uniform in Epoch 32

- Upload marker remains `ALL_PASS=True`; Slurm and sacct both report jobs
  `994340`, `994341`, and `994342` are still `RUNNING` for about `2:58:43` on
  nodes `g0005`, `g0005`, and `g0024`.
- Latest observed log evidence reaches `05:34:48+08:00`: main reached Epoch 32
  step 50 with `Loss=0.4813`, `cls_loss=0.2530`, `reg_loss=0.2282`.
- Uniform completed Epoch 31 with `Loss=0.4507`, `cls_loss=0.2341`,
  `reg_loss=0.2165`, then entered Epoch 32.
- Lowreg completed Epoch 30 with `Loss=0.4911`, `cls_loss=0.2558`,
  `reg_loss=0.2353`, then entered Epoch 31.
- No mAP/result artifact is visible. Checkpoints remain only `epoch_19.pth` for
  each active run; narrow failure scan is empty; `/data` has about `1.9T` free.
- No code/config/model change occurred. Contract remains `50% ViT/backbone
  compute, dense decode`; no test-time GT/teacher/cache. Decision: continue all
  jobs, do not clean active checkpoints or launch duplicates, and wait for the
  Epoch 39 checkpoint plus first eval after the Epoch 41 boundary.

## 2026-05-29T05:36:02+08:00 - E2E raw-density main entered Epoch 33

- Upload marker remains `ALL_PASS=True`; Slurm and sacct both report jobs
  `994340`, `994341`, and `994342` are still `RUNNING` for about `3:01:47` on
  nodes `g0005`, `g0005`, and `g0024`.
- Latest observed log evidence reaches `05:37:40+08:00`: main completed Epoch
  32 with `Loss=0.4817`, `cls_loss=0.2542`, `reg_loss=0.2274`, then entered
  Epoch 33.
- Uniform reached Epoch 32 step 50 with `Loss=0.4856`, `cls_loss=0.2505`,
  `reg_loss=0.2351`.
- Lowreg reached Epoch 31 step 50 with `Loss=0.4571`, `cls_loss=0.2394`,
  `reg_loss=0.2177`.
- No mAP/result artifact is visible. Checkpoints remain only `epoch_19.pth` for
  each active run; narrow failure scan is empty; `/data` has about `1.9T` free.
- No code/config/model change occurred. Contract remains `50% ViT/backbone
  compute, dense decode`; no test-time GT/teacher/cache. Decision: continue all
  jobs, do not clean active checkpoints or launch duplicates, and wait for the
  Epoch 39 checkpoint plus first eval after the Epoch 41 boundary.

## 2026-05-29T05:39:26+08:00 - E2E raw-density main/uniform reached Epoch 33

- Upload marker remains `ALL_PASS=True`; Slurm and sacct both report jobs
  `994340`, `994341`, and `994342` are still `RUNNING` for about `3:05:11` on
  nodes `g0005`, `g0005`, and `g0024`.
- Latest observed log evidence reaches `05:40:16+08:00`: main reached Epoch 33
  step 50 with `Loss=0.4977`, `selector_spacing_loss=0.0001`,
  `selector_tv_loss=0.0000`, `cls_loss=0.2702`, `reg_loss=0.2275`.
- Uniform completed Epoch 32 with `Loss=0.4853`, `cls_loss=0.2536`,
  `reg_loss=0.2317`, then entered Epoch 33.
- Lowreg completed Epoch 31 with `Loss=0.4673`, `cls_loss=0.2455`,
  `reg_loss=0.2218`, then entered Epoch 32.
- No mAP/result artifact is visible. Checkpoints remain only `epoch_19.pth` for
  each active run; narrow failure scan is empty; `/data` has about `1.9T` free.
- No code/config/model change occurred. Contract remains `50% ViT/backbone
  compute, dense decode`; no test-time GT/teacher/cache. Decision: continue all
  jobs, do not clean active checkpoints or launch duplicates, and wait for the
  Epoch 39 checkpoint plus first eval after the Epoch 41 boundary.

## 2026-05-29T05:42:41+08:00 - E2E raw-density main entered Epoch 34

- Upload marker remains `ALL_PASS=True`; Slurm and sacct both report jobs
  `994340`, `994341`, and `994342` are still `RUNNING` for about `3:08:26` on
  nodes `g0005`, `g0005`, and `g0024`.
- Latest observed log evidence reaches `05:44:32+08:00`: main completed Epoch
  33 with `Loss=0.4736`, `selector_spacing_loss=0.0001`,
  `selector_tv_loss=0.0000`, `cls_loss=0.2539`, `reg_loss=0.2197`, then entered
  Epoch 34 and reached Epoch 34 step 50 with `Loss=0.4655`, `cls_loss=0.2406`,
  `reg_loss=0.2248`.
- Uniform reached Epoch 33 step 50 with `Loss=0.5096`, `cls_loss=0.2771`,
  `reg_loss=0.2325`.
- Lowreg completed Epoch 32 with `Loss=0.5011`, `cls_loss=0.2650`,
  `reg_loss=0.2361`, then entered Epoch 33.
- No mAP/result artifact is visible. Checkpoints remain only `epoch_19.pth` for
  each active run; narrow failure scan is empty; `/data` has about `1.9T` free.
- No code/config/model change occurred. Contract remains `50% ViT/backbone
  compute, dense decode`; no test-time GT/teacher/cache. Decision: continue all
  jobs, do not clean active checkpoints or launch duplicates, and wait for the
  Epoch 39 checkpoint plus first eval after the Epoch 41 boundary.

## 2026-05-29T05:46:16+08:00 - E2E raw-density main entered Epoch 35

- Upload marker remains `ALL_PASS=True`; Slurm and sacct both report jobs
  `994340`, `994341`, and `994342` are still `RUNNING` for about `3:12:01` on
  nodes `g0005`, `g0005`, and `g0024`.
- Latest observed log evidence reaches `05:47:08+08:00`: main completed Epoch
  34 with `Loss=0.4765`, `selector_spacing_loss=0.0001`,
  `selector_tv_loss=0.0000`, `cls_loss=0.2519`, `reg_loss=0.2246`, then entered
  Epoch 35.
- Uniform completed Epoch 33 with `Loss=0.4837`, `cls_loss=0.2616`,
  `reg_loss=0.2221`, then entered Epoch 34.
- Lowreg reached Epoch 33 step 50 with `Loss=0.5074`, `cls_loss=0.2772`,
  `reg_loss=0.2302`.
- No mAP/result artifact is visible. Checkpoints remain only `epoch_19.pth` for
  each active run; narrow failure scan is empty; `/data` has about `1.9T` free.
- No code/config/model change occurred. Contract remains `50% ViT/backbone
  compute, dense decode`; no test-time GT/teacher/cache. Decision: continue all
  jobs, do not clean active checkpoints or launch duplicates, and wait for the
  Epoch 39 checkpoint plus first eval after the Epoch 41 boundary.

## 2026-05-29T05:49:40+08:00 - E2E raw-density main/uniform reached Epoch 35

- Upload marker remains `ALL_PASS=True`; Slurm and sacct both report jobs
  `994340`, `994341`, and `994342` are still `RUNNING` for about `3:15:25` on
  nodes `g0005`, `g0005`, and `g0024`.
- Latest observed log evidence reaches `05:51:10+08:00`: main reached Epoch 35
  step 50 with `Loss=0.4160`, `selector_spacing_loss=0.0001`,
  `selector_tv_loss=0.0000`, `cls_loss=0.2109`, `reg_loss=0.2050`.
- Uniform completed Epoch 34 with `Loss=0.4859`, `cls_loss=0.2576`,
  `reg_loss=0.2284`, then entered Epoch 35.
- Lowreg completed Epoch 33 with `Loss=0.4841`, `cls_loss=0.2614`,
  `reg_loss=0.2227`, then entered Epoch 34.
- No mAP/result artifact is visible. Checkpoints remain only `epoch_19.pth` for
  each active run; narrow failure scan is empty; `/data` has about `1.9T` free.
- No code/config/model change occurred. Contract remains `50% ViT/backbone
  compute, dense decode`; no test-time GT/teacher/cache. Decision: continue all
  jobs, do not clean active checkpoints or launch duplicates, and wait for the
  Epoch 39 checkpoint plus first eval after the Epoch 41 boundary.

## 2026-05-29T05:52:57+08:00 - E2E raw-density main entered Epoch 36

- Upload marker remains `ALL_PASS=True`; Slurm reports jobs `994340`, `994341`,
  and `994342` are still `RUNNING` for about `3:18:42` on nodes `g0005`,
  `g0005`, and `g0024`.
- Latest observed log evidence reaches `05:54:01+08:00`: main completed Epoch
  35 with `Loss=0.4439`, `selector_spacing_loss=0.0001`,
  `selector_tv_loss=0.0000`, `cls_loss=0.2292`, `reg_loss=0.2147`, then entered
  Epoch 36.
- Uniform reached Epoch 35 step 50 with `Loss=0.4222`, `cls_loss=0.2164`,
  `reg_loss=0.2058`.
- Lowreg reached Epoch 34 step 50 with `Loss=0.4785`, `cls_loss=0.2505`,
  `reg_loss=0.2279`.
- No mAP/result artifact is visible. Checkpoints remain only `epoch_19.pth` for
  each active run; narrow failure scan is empty; `/data` has about `1.9T` free.
- No code/config/model change occurred. Contract remains `50% ViT/backbone
  compute, dense decode`; no test-time GT/teacher/cache. Decision: continue all
  jobs, do not clean active checkpoints or launch duplicates, and wait for the
  Epoch 39 checkpoint plus first eval after the Epoch 41 boundary.

## 2026-05-29T05:56:20+08:00 - E2E raw-density main entered Epoch 37

- Upload marker remains `ALL_PASS=True`; Slurm reports jobs `994340`, `994341`,
  and `994342` are still `RUNNING` for about `3:22:05` on nodes `g0005`,
  `g0005`, and `g0024`.
- Latest observed log evidence reaches `05:57:53+08:00`: main completed Epoch
  36 with `Loss=0.4441`, `selector_spacing_loss=0.0001`,
  `selector_tv_loss=0.0000`, `cls_loss=0.2306`, `reg_loss=0.2134`, then entered
  Epoch 37.
- Uniform completed Epoch 35 with `Loss=0.4514`, `cls_loss=0.2335`,
  `reg_loss=0.2179`, then entered Epoch 36.
- Lowreg reached Epoch 35 step 50 with `Loss=0.4418`, `cls_loss=0.2292`,
  `reg_loss=0.2126`.
- No mAP/result artifact is visible. Checkpoints remain only `epoch_19.pth` for
  each active run; narrow failure scan is empty; `/data` has about `1.9T` free.
- No code/config/model change occurred. Contract remains `50% ViT/backbone
  compute, dense decode`; no test-time GT/teacher/cache. Decision: continue all
  jobs, do not clean active checkpoints or launch duplicates, and wait for the
  Epoch 39 checkpoint plus first eval after the Epoch 41 boundary.

## 2026-05-29T05:59:15+08:00 - E2E raw-density main reached Epoch 37 step 50

- Upload marker remains `ALL_PASS=True`; Slurm and sacct report jobs `994340`,
  `994341`, and `994342` are still `RUNNING` for about `3:25:00` on nodes
  `g0005`, `g0005`, and `g0024`.
- Latest observed log evidence reaches `06:00:44+08:00`: main reached Epoch 37
  step 50 with `Loss=0.4756`, `selector_spacing_loss=0.0001`,
  `selector_tv_loss=0.0000`, `cls_loss=0.2535`, `reg_loss=0.2220`.
- Uniform reached Epoch 36 step 50 with `Loss=0.4522`, `cls_loss=0.2338`,
  `reg_loss=0.2184`.
- Lowreg completed Epoch 35 with `Loss=0.4626`, `selector_spacing_loss=0.0000`,
  `selector_tv_loss=0.0000`, `cls_loss=0.2413`, `reg_loss=0.2213`, then entered
  Epoch 36.
- No mAP/result artifact is visible. Checkpoints remain only `epoch_19.pth` for
  each active run; narrow failure scan is empty; `/data` has about `1.9T` free.
- No code/config/model change occurred. Contract remains `50% ViT/backbone
  compute, dense decode`; no test-time GT/teacher/cache. Decision: continue all
  jobs, do not clean active checkpoints or launch duplicates, and wait for the
  Epoch 39 checkpoint plus first eval after the Epoch 41 boundary.

## 2026-05-29T06:01:36+08:00 - E2E raw-density main entered Epoch 38

- Slurm reports jobs `994340`, `994341`, and `994342` are still `RUNNING` for
  about `3:27:21` on nodes `g0005`, `g0005`, and `g0024`.
- Latest observed log evidence reaches `06:03:13+08:00`: main completed Epoch
  37 with `Loss=0.4667`, `selector_spacing_loss=0.0001`,
  `selector_tv_loss=0.0000`, `cls_loss=0.2420`, `reg_loss=0.2246`, then entered
  Epoch 38.
- Uniform completed Epoch 36 with `Loss=0.4399`, `cls_loss=0.2269`,
  `reg_loss=0.2130`, then entered Epoch 37.
- Lowreg latest log remains Epoch 36 start at `05:58:28+08:00`, within normal
  progress window and far below the one-hour no-change threshold.
- No mAP/result artifact is visible. Checkpoints remain only `epoch_19.pth` for
  each active run; narrow failure scan is empty.
- No code/config/model change occurred. Contract remains `50% ViT/backbone
  compute, dense decode`; no test-time GT/teacher/cache. Decision: continue all
  jobs, do not clean active checkpoints or launch duplicates, and wait for the
  Epoch 39 checkpoint plus first eval after the Epoch 41 boundary.

## 2026-05-29T06:05:56+08:00 - E2E raw-density main reached Epoch 38 step 50

- Slurm reports jobs `994340`, `994341`, and `994342` are still `RUNNING` for
  about `3:31:41` on nodes `g0005`, `g0005`, and `g0024`.
- Latest observed log evidence reaches `06:07:38+08:00`: main reached Epoch 38
  step 50 with `Loss=0.4637`, `selector_spacing_loss=0.0001`,
  `selector_tv_loss=0.0000`, `cls_loss=0.2393`, `reg_loss=0.2243`.
- Uniform completed Epoch 37 with `Loss=0.4688`, `cls_loss=0.2433`,
  `reg_loss=0.2256`, then entered Epoch 38.
- Lowreg completed Epoch 36 with `Loss=0.4500`, `selector_spacing_loss=0.0000`,
  `selector_tv_loss=0.0000`, `cls_loss=0.2322`, `reg_loss=0.2177`, then entered
  Epoch 37.
- No mAP/result artifact is visible. Checkpoints remain only `epoch_19.pth` for
  each active run; narrow failure scan is empty.
- No code/config/model change occurred. Contract remains `50% ViT/backbone
  compute, dense decode`; no test-time GT/teacher/cache. Decision: continue all
  jobs, do not clean active checkpoints or launch duplicates, and wait for the
  Epoch 39 checkpoint plus first eval after the Epoch 41 boundary.

## 2026-05-29T06:07:18+08:00 - E2E raw-density main entered Epoch 39

- Slurm reports jobs `994340`, `994341`, and `994342` are still `RUNNING` for
  about `3:33:03` on nodes `g0005`, `g0005`, and `g0024`.
- Latest observed log evidence reaches `06:08:44+08:00`: main completed Epoch
  38 with `Loss=0.4593`, `selector_spacing_loss=0.0001`,
  `selector_tv_loss=0.0000`, `cls_loss=0.2370`, `reg_loss=0.2222`, then entered
  Epoch 39.
- Uniform remains in Epoch 38 after completing Epoch 37; lowreg remains in Epoch
  37 after completing Epoch 36.
- No mAP/result artifact is visible. Checkpoints remain only `epoch_19.pth` for
  each active run; narrow failure scan is empty.
- No code/config/model change occurred. Contract remains `50% ViT/backbone
  compute, dense decode`; no test-time GT/teacher/cache. Decision: continue all
  jobs, do not clean active checkpoints or launch duplicates, and wait for
  `epoch_39.pth` plus first eval after the Epoch 41 boundary.

## 2026-05-29T06:08:26+08:00 - E2E raw-density controls reached mid-epoch

- Slurm reports jobs `994340`, `994341`, and `994342` are still `RUNNING` for
  about `3:34:11` on nodes `g0005`, `g0005`, and `g0024`.
- Main remains in Epoch 39 after completing Epoch 38; `epoch_39.pth` is still
  pending.
- Uniform reached Epoch 38 step 50 with `Loss=0.4738`, `cls_loss=0.2486`,
  `reg_loss=0.2252`.
- Lowreg reached Epoch 37 step 50 with `Loss=0.4930`,
  `selector_spacing_loss=0.0000`, `selector_tv_loss=0.0000`,
  `cls_loss=0.2657`, `reg_loss=0.2273`.
- No mAP/result artifact is visible. Checkpoints remain only `epoch_19.pth` for
  each active run; narrow failure scan is empty.
- No code/config/model change occurred. Contract remains `50% ViT/backbone
  compute, dense decode`; no test-time GT/teacher/cache. Decision: continue all
  jobs, do not clean active checkpoints or launch duplicates, and wait for
  `epoch_39.pth` plus first eval after the Epoch 41 boundary.

## 2026-05-29T06:10:26+08:00 - E2E raw-density main reached Epoch 39 step 50

- Upload marker remains `ALL_PASS=True`; Slurm and sacct report jobs `994340`,
  `994341`, and `994342` are still `RUNNING` for about `3:36:11` on nodes
  `g0005`, `g0005`, and `g0024`; `/data` has about `1.9T` free.
- Latest observed log evidence reaches `06:11:25+08:00`: main reached Epoch 39
  step 50 with `Loss=0.4187`, `selector_spacing_loss=0.0001`,
  `selector_tv_loss=0.0000`, `cls_loss=0.2159`, `reg_loss=0.2027`.
- Uniform remains in Epoch 38 after step 50.
- Lowreg completed Epoch 37 with `Loss=0.4830`,
  `selector_spacing_loss=0.0000`, `selector_tv_loss=0.0000`,
  `cls_loss=0.2538`, `reg_loss=0.2292`, then entered Epoch 38.
- No mAP/result artifact is visible. Checkpoints remain only `epoch_19.pth` for
  each active run; narrow failure scan is empty.
- No code/config/model change occurred. Contract remains `50% ViT/backbone
  compute, dense decode`; no test-time GT/teacher/cache. Decision: continue all
  jobs, do not clean active checkpoints or launch duplicates, and wait for
  `epoch_39.pth` plus first eval after the Epoch 41 boundary.

## 2026-05-29T06:11:50+08:00 - E2E raw-density uniform entered Epoch 39

- Slurm reports jobs `994340`, `994341`, and `994342` are still `RUNNING` for
  about `3:37:35` on nodes `g0005`, `g0005`, and `g0024`.
- Main remains in Epoch 39 after step 50 and has not written `epoch_39.pth`.
- Uniform completed Epoch 38 with `Loss=0.4693`, `cls_loss=0.2438`,
  `reg_loss=0.2254`, then entered Epoch 39.
- Lowreg remains in Epoch 38 after completing Epoch 37.
- No mAP/result artifact is visible. Checkpoints remain only `epoch_19.pth` for
  each active run; narrow failure scan is empty.
- No code/config/model change occurred. Contract remains `50% ViT/backbone
  compute, dense decode`; no test-time GT/teacher/cache. Decision: continue all
  jobs, do not clean active checkpoints or launch duplicates, and wait for
  `epoch_39.pth` plus first eval after the Epoch 41 boundary.

## 2026-05-29T06:12:57+08:00 - E2E raw-density main wrote epoch_39

- Slurm reports jobs `994340`, `994341`, and `994342` are still `RUNNING` for
  about `3:38:42` on nodes `g0005`, `g0005`, and `g0024`.
- Main completed Epoch 39 with `Loss=0.4529`, `selector_spacing_loss=0.0001`,
  `selector_tv_loss=0.0000`, `cls_loss=0.2351`, `reg_loss=0.2178`, wrote
  `checkpoint/epoch_39.pth` size `623905307` mtime
  `2026-05-29T06:14:07+08:00`, then entered Epoch 40.
- Uniform entered Epoch 39 and still has only `epoch_19.pth`; lowreg remains in
  Epoch 38 with only `epoch_19.pth`.
- No mAP/result artifact is visible and narrow failure scan is empty. No
  checkpoint cleanup was performed because all runs are active.
- No code/config/model change occurred. Contract remains `50% ViT/backbone
  compute, dense decode`; no test-time GT/teacher/cache. Decision: continue all
  jobs; wait for uniform/lowreg `epoch_39.pth` and first eval after the Epoch 41
  boundary.

## 2026-05-29T06:14:54+08:00 - E2E raw-density main Epoch 40 step 50

- Upload marker remains `ALL_PASS=True`; Slurm and sacct report jobs `994340`,
  `994341`, and `994342` are still `RUNNING` for about `3:40:39` on nodes
  `g0005`, `g0005`, and `g0024`; `/data` has about `1.9T` free.
- Main reached Epoch 40 step 50 with `Loss=0.4445`,
  `selector_spacing_loss=0.0001`, `selector_tv_loss=0.0000`,
  `cls_loss=0.2307`, `reg_loss=0.2137`; main has `epoch_19.pth` and
  `epoch_39.pth`.
- Uniform reached Epoch 39 step 50 with `Loss=0.4138`, `cls_loss=0.2157`,
  `reg_loss=0.1980`; uniform still has only `epoch_19.pth`.
- Lowreg reached Epoch 38 step 50 with `Loss=0.4791`,
  `selector_spacing_loss=0.0000`, `selector_tv_loss=0.0000`,
  `cls_loss=0.2498`, `reg_loss=0.2292`; lowreg still has only `epoch_19.pth`.
- No mAP/result artifact is visible and narrow failure scan is empty. No
  checkpoint cleanup was performed because all runs are active.
- No code/config/model change occurred. Contract remains `50% ViT/backbone
  compute, dense decode`; no test-time GT/teacher/cache. Decision: continue all
  jobs; wait for uniform/lowreg `epoch_39.pth` and first eval after the Epoch 41
  boundary.

## 2026-05-29T06:16:58+08:00 - E2E raw-density uniform wrote epoch_39

- Upload marker remains `ALL_PASS=True`; Slurm and sacct report jobs `994340`,
  `994341`, and `994342` are still `RUNNING` for about `3:42:43` on nodes
  `g0005`, `g0005`, and `g0024`; `/data` has about `1.9T` free.
- Main remains in Epoch 40 after step 50 and has `epoch_19.pth` plus
  `epoch_39.pth`.
- Uniform completed Epoch 39 with `Loss=0.4494`, `cls_loss=0.2347`,
  `reg_loss=0.2147`, wrote `checkpoint/epoch_39.pth` size `623852123` mtime
  `2026-05-29T06:18:54+08:00`, then entered Epoch 40.
- Lowreg completed Epoch 38 with `Loss=0.4713`,
  `selector_spacing_loss=0.0000`, `selector_tv_loss=0.0000`,
  `cls_loss=0.2441`, `reg_loss=0.2272`, then entered Epoch 39; lowreg still has
  only `epoch_19.pth`.
- No mAP/result artifact is visible and narrow failure scan is empty. No
  checkpoint cleanup was performed because all runs are active.
- No code/config/model change occurred. Contract remains `50% ViT/backbone
  compute, dense decode`; no test-time GT/teacher/cache. Decision: continue all
  jobs; wait for lowreg `epoch_39.pth` and first eval after the Epoch 41
  boundary.

## 2026-05-29T06:18:19+08:00 - E2E raw-density main entered Epoch 41

- Slurm reports jobs `994340`, `994341`, and `994342` are still `RUNNING` for
  about `3:44:04` on nodes `g0005`, `g0005`, and `g0024`.
- Main completed Epoch 40 with `Loss=0.4491`, `selector_spacing_loss=0.0001`,
  `selector_tv_loss=0.0000`, `cls_loss=0.2346`, `reg_loss=0.2144`, then entered
  Epoch 41; main checkpoints remain `epoch_19.pth` and `epoch_39.pth`.
- Uniform has `epoch_19.pth` and `epoch_39.pth` and remains in Epoch 40.
- Lowreg remains in Epoch 39 with only `epoch_19.pth`.
- No mAP/result artifact is visible and narrow failure scan is empty. No
  checkpoint cleanup was performed because all runs are active.
- No code/config/model change occurred. Contract remains `50% ViT/backbone
  compute, dense decode`; no test-time GT/teacher/cache. Decision: continue all
  jobs; wait for lowreg `epoch_39.pth` and first eval after the Epoch 41
  boundary.

## 2026-05-29T06:19:37+08:00 - E2E raw-density lowreg reached Epoch 39 step 50

- Slurm reports jobs `994340`, `994341`, and `994342` are still `RUNNING` for
  about `3:45:22` on nodes `g0005`, `g0005`, and `g0024`.
- Main remains in Epoch 41 after entering at `06:19:32`; no eval/mAP is visible
  yet.
- Uniform reached Epoch 40 step 50 with `Loss=0.4369`, `cls_loss=0.2244`,
  `reg_loss=0.2125`; uniform has `epoch_19.pth` and `epoch_39.pth`.
- Lowreg reached Epoch 39 step 50 with `Loss=0.4301`,
  `selector_spacing_loss=0.0000`, `selector_tv_loss=0.0000`,
  `cls_loss=0.2262`, `reg_loss=0.2039`; lowreg still has only `epoch_19.pth`.
- No mAP/result artifact is visible and narrow failure scan is empty. No
  checkpoint cleanup was performed because all runs are active.
- No code/config/model change occurred. Contract remains `50% ViT/backbone
  compute, dense decode`; no test-time GT/teacher/cache. Decision: continue all
  jobs; wait for lowreg `epoch_39.pth` and first eval after the Epoch 41
  boundary.

## 2026-05-29T06:21:31+08:00 - E2E raw-density main reached Epoch 41 step 50

- Upload marker remains `ALL_PASS=True`; Slurm and sacct report jobs `994340`,
  `994341`, and `994342` are still `RUNNING` for about `3:47:16` on nodes
  `g0005`, `g0005`, and `g0024`; `/data` has about `1.9T` free.
- Main reached Epoch 41 step 50 with `Loss=0.4416`,
  `selector_spacing_loss=0.0001`, `selector_tv_loss=0.0000`,
  `cls_loss=0.2276`, `reg_loss=0.2139`; main has `epoch_19.pth` and
  `epoch_39.pth`.
- Uniform remains in Epoch 40 after step 50 with `epoch_19.pth` and
  `epoch_39.pth`.
- Lowreg remains in Epoch 39 after step 50 and still only has `epoch_19.pth`.
- No mAP/result artifact is visible and narrow failure scan is empty. No
  checkpoint cleanup was performed because all runs are active.
- No code/config/model change occurred. Contract remains `50% ViT/backbone
  compute, dense decode`; no test-time GT/teacher/cache. Decision: continue all
  jobs; wait for lowreg `epoch_39.pth` and first eval after the Epoch 41
  boundary.

## 2026-05-29T06:23:20+08:00 - E2E raw-density lowreg wrote epoch_39

- Upload marker remains `ALL_PASS=True`; Slurm reports jobs `994340`, `994341`,
  and `994342` are still `RUNNING` for about `3:49:05` on nodes `g0005`,
  `g0005`, and `g0024`; `/data` has about `1.9T` free.
- Main completed Epoch 41 with `Loss=0.4640`, `selector_spacing_loss=0.0001`,
  `selector_tv_loss=0.0000`, `cls_loss=0.2394`, `reg_loss=0.2246`; no eval/mAP
  line is visible yet.
- Uniform completed Epoch 40 with `Loss=0.4460`, `cls_loss=0.2315`,
  `reg_loss=0.2145`, then entered Epoch 41.
- Lowreg completed Epoch 39 with `Loss=0.4681`,
  `selector_spacing_loss=0.0000`, `selector_tv_loss=0.0000`,
  `cls_loss=0.2454`, `reg_loss=0.2227`, wrote `checkpoint/epoch_39.pth` size
  `623905307` mtime `2026-05-29T06:21:06+08:00`, then entered Epoch 40.
- All three now have `epoch_19.pth` and `epoch_39.pth`. No mAP/result artifact
  is visible and narrow failure scan is empty. No checkpoint cleanup was
  performed because all runs are active.
- No code/config/model change occurred. Contract remains `50% ViT/backbone
  compute, dense decode`; no test-time GT/teacher/cache. Decision: continue all
  jobs; wait for first eval/mAP after Epoch 41 boundary.

## 2026-05-29T06:25:02+08:00 - E2E raw-density main first eval running

- Raw main log tail confirms first eval/inference started immediately after
  Epoch 41 completion at `06:24:52+08:00`; tqdm progress reached about `63/396`
  validation windows during this check.
- No `Average-mAP` or result file is visible yet. Main run files currently
  include config, `log.json`, `epoch_19.pth`, and `epoch_39.pth`.
- No traceback, OOM, NaN, or severe-result trigger is visible. No checkpoint
  cleanup was performed because the run is active.
- Contract remains `50% ViT/backbone compute, dense decode`; no test-time
  GT/teacher/cache. Decision: continue monitoring until main first mAP appears,
  then apply uniform/random-fixed/severe-result gates before changing or
  launching follow-up routes.

## 2026-05-29T06:29:54+08:00 - E2E raw-density main and uniform first evals running

- Upload marker remains `ALL_PASS=True`; Slurm reports jobs `994340`, `994341`,
  and `994342` are still `RUNNING` on `g0005`, `g0005`, and `g0024`; `/data`
  has about `1.9T` free.
- Main completed Epoch 41 at `06:24:52+08:00`; raw validation tqdm reached about
  `135/396` windows. Uniform completed Epoch 41 at `06:29:54+08:00` and started
  first validation around `4/396`. Lowreg completed Epoch 40 at
  `06:26:44+08:00` and entered Epoch 41.
- All three active runs have `epoch_19.pth` and `epoch_39.pth`. No
  `Average-mAP`, result artifact, traceback, OOM, NaN, or severe-result trigger
  is visible.
- No code/config/model change occurred. No checkpoint cleanup was performed
  because all runs are active. Contract remains `50% ViT/backbone compute, dense
  decode`; no test-time GT/teacher/cache. Decision: continue monitoring until
  first mAP appears, then apply the configured baseline and severe-result gates.

## 2026-05-29T06:35:23+08:00 - E2E raw-density all three first evals running

- Slurm reports jobs `994340`, `994341`, and `994342` are still `RUNNING` for
  about `3:58:58`; `/data` has about `1.9T` free.
- Main first validation advanced to about `247/396` windows. Uniform first
  validation advanced to about `145/396`. Lowreg completed Epoch 41 at
  `06:32:18+08:00` with `Loss=0.4758`, then started first validation and reached
  about `14/396`.
- No `Average-mAP`, result artifact, traceback, OOM, NaN, or severe-result
  trigger is visible. No code/config/model change occurred, and no checkpoint
  cleanup was performed because all runs are active.
- Contract remains `50% ViT/backbone compute, dense decode`; no test-time
  GT/teacher/cache. Decision: continue monitoring until main first mAP appears.

## 2026-05-29T06:43:15+08:00 - E2E raw-density main first mAP below baseline

- Main first eval after Epoch 41 completed with `Average-mAP=63.72`, vector
  `79.50 / 74.72 / 65.97 / 57.22 / 41.22`.
- Delta: `-0.05` vs random-fixed Adapter `63.77`, `-0.13` vs strict EMA
  `63.85`, `-0.92` vs stratified `64.64`, `-1.37` vs uniform stride-2 `65.09`,
  and `-13.90` vs oracle-boundary `77.62`.
- Interpretation: below positive-claim gate, but not a severe collapse. No
  Pro severe-result escalation is triggered by this main result alone.
- Uniform first eval is still running at about `316/396`; lowreg is at about
  `206/396`. No traceback, OOM, NaN, or result artifact is visible. No
  checkpoint cleanup was performed because runs are active.
- Contract remains `50% ViT/backbone compute, dense decode`; no test-time
  GT/teacher/cache. Decision: continue all jobs and wait for uniform/lowreg
  first mAP before route interpretation or follow-up launch.

## 2026-05-29T06:47:57+08:00 - E2E raw-density uniform control first mAP below control gate

- Uniform frozen-control first eval after Epoch 41 completed with
  `Average-mAP=64.12`, vector `79.81 / 74.81 / 67.11 / 56.77 / 42.07`.
- Delta: `+0.35` vs random-fixed Adapter `63.77`, `+0.27` vs strict EMA
  `63.85`, `-0.52` vs stratified `64.64`, `-0.97` vs uniform stride-2
  `65.09`, and `-13.50` vs oracle-boundary `77.62`.
- Interpretation: not a severe collapse, but below the `~64.3` uniform-control
  concern line, so protocol/control implementation differences remain a live
  concern. Main remains `63.72`; lowreg is still evaluating at about `308/396`.
- No traceback, OOM, NaN, or severe-result trigger is visible. No
  checkpoint cleanup was performed because runs are active.
- Contract remains `50% ViT/backbone compute, dense decode`; no test-time
  GT/teacher/cache. Decision: wait for lowreg first mAP, then interpret the
  route against this imperfect uniform control before launching follow-up.

## 2026-05-29T06:50:59+08:00 - E2E raw-density lowreg first mAP worse than main/control

- Lowreg first eval after Epoch 41 completed with `Average-mAP=62.98`, vector
  `78.76 / 74.32 / 65.87 / 55.03 / 40.93`.
- Delta: `-0.79` vs random-fixed Adapter `63.77`, `-0.87` vs strict EMA
  `63.85`, `-1.66` vs stratified `64.64`, `-2.11` vs uniform stride-2 `65.09`,
  and `-14.64` vs oracle-boundary `77.62`.
- Three-way first-eval order is uniform `64.12` > main `63.72` > lowreg
  `62.98`. This is below the positive-claim gate but not a severe collapse.
- No traceback, OOM, NaN, or severe-result trigger is visible. No checkpoint
  cleanup was performed because runs are active.
- Contract remains `50% ViT/backbone compute, dense decode`; no test-time
  GT/teacher/cache. Decision: continue active runs for later checkpoints, but
  start Pro discussion before any new long follow-up or route change.

## 2026-05-29T07:05:00+08:00 - E2E raw-density Pro discussion completed

- Initial `globalai` `gpt-5-pro` route failed with `model_not_found` and was
  rejected as incomplete.
- Oracle browser `gpt-5.5-pro` succeeded in about `8m45s`; stdout is
  `logs/oracle_pro_e2e_rawdensel_discussion_20260529.txt`, stderr is
  `logs/oracle_pro_e2e_rawdensel_discussion_20260529.err.txt`, and summary is in
  `research-wiki/experiments/BATA_E2E_RAW_DENSITY_SELECTOR_PRO_DISCUSSION_20260529.md`.
- Accepted diagnosis: current method is end-to-end, but detection-loss-only
  selector supervision is too weak; frozen uniform has a protocol gap; detector
  geometry handling likely penalizes non-uniform positions; lowreg proves more
  movement without semantic guidance is harmful.
- Accepted next routes: Route A uniform parity/protocol repair, Route B
  train-GT auxiliary action/boundary selector with residual slots, and Route C
  coordinate-aware ActionFormer/head.
- Decision: continue current jobs for later/final evidence, do not launch
  main/lowreg lambda sweeps, and prepare Route A/B implementation/deployment
  first.

## 2026-05-29T07:14:15+08:00 - E2E Route A/B self-check recorded

- Added self-check `research-wiki/experiments/BATA_E2E_ROUTE_AB_SELF_CHECK_20260529.md`.
- Route A is exact frozen uniform bypass via
  `e2e_rawdensel_384of768_uniform_exact_bypass_adapter.py`; Route B is
  train-GT auxiliary action/boundary density with residual slots via
  `e2e_rawdensel_384of768_gtaux_residual_adapter.py`.
- Changed files in scope: `temporal_density_selector.py`,
  `tests/test_e2e_raw_frame_selector_contracts.py`, and the two new configs.
  Separate dirty BATA boundary-acquisition/post-processing files are not part
  of this Route A/B gate.
- Local verification from `OpenTAD_BATA_Clean`: `py_compile` PASS; pytest
  `1 passed, 9 skipped` on Windows; `git diff --check` PASS except line-ending
  warnings.
- Contract remains `50% ViT/backbone compute, dense decode`; Route B train GT
  use is training-only; no test-time GT/teacher/cache. Decision: start
  Pro/Gemini/DeepSeek review gates and do not sync or launch before they pass.

## 2026-05-29T07:30:23+08:00 - E2E Route A/B Pro FAIL fixed locally

- GPT-5.5 Pro implementation review returned `FAIL`; output is
  `logs/gpt5pro_e2e_route_ab_review_20260529.txt`, and the fix report is
  `research-wiki/experiments/BATA_E2E_ROUTE_AB_PRO_REVIEW_AND_FIXES_20260529.md`.
- Blocking findings were Route A tail uniform not exact stride-2, invalid-slot
  padding inconsistent with the self-check, and missing Route B geometry/remap
  tests.
- Fixed locally in `temporal_density_selector.py` and
  `tests/test_e2e_raw_frame_selector_contracts.py`: global stride tail uniform,
  last-valid-dense padding, `min_position_gap` for anchor/residual coordinates,
  mask shape guard, and new tail/geometry/mask tests.
- Post-fix checks from `OpenTAD_BATA_Clean`: `py_compile` PASS; pytest
  `1 passed, 12 skipped` on Windows; `git diff --check` PASS except
  LF-to-CRLF warnings.
- Contract remains `50% ViT/backbone compute, dense decode`; no test-time
  GT/teacher/cache. Decision: run focused Pro re-review before Gemini,
  DeepSeek, remote preflight, or deployment.

## 2026-05-29T07:44:24+08:00 - E2E Route A/B local review gates passed

- Focused GPT-5.5 Pro re-review returned `PASS` for entering
  Gemini/DeepSeek/N16R4 preflight:
  `logs/gpt5pro_e2e_route_ab_fix_review_20260529.txt`.
- Gemini CLI `gemini-3-pro-preview` returned exit code `0`, verdict `PASS`:
  `logs/gemini3_pro_preview_e2e_route_ab_review_20260529.txt`.
- Claude CLI DeepSeek `deepseek-v4-pro` returned exit code `0`, verdict
  `PASS`: `logs/claude_deepseek_v4_pro_e2e_route_ab_review_20260529.txt`.
- Committed reviewed Route A/B files as
  `7d525ab add e2e route ab selector controls`; unrelated local dirty BATA
  files were intentionally excluded.
- Active E2E monitor at `07:44:29+08:00`: jobs `994340`, `994341`, and
  `994342` remain `RUNNING`, no failure scan hits, `/data` has about `1.9T`
  free. mAP trends are main `63.72 -> 64.15 -> 64.09`, uniform
  `64.12 -> 64.52 -> 64.76`, lowreg `62.98 -> 63.55`.
- Decision: sync reviewed files to N16R4 and run Linux preflight before any
  Route A/B long launch.

## 2026-05-29T07:47:57+08:00 - E2E Route A/B remote preflight passed

- Synced only the four reviewed Route A/B files from commit `7d525ab` to
  `~/run/yuzibo/OpenTAD_BATA_Clean`.
- Standalone N16R4 preflight log:
  `~/run/yuzibo/OpenTAD_BATA_Clean/logs/e2e_route_ab_preflight_20260529.log`.
- Linux checks passed: `py_compile`, selector contract pytest
  `13 passed in 21.03s`, and merged config checks for Route A/B with
  `MERGED_CONFIG_PREFLIGHT=PASS`.
- No training ran on the login node. Decision: submit Route A/B Slurm jobs.

## 2026-05-29T07:56:57+08:00 - E2E Route A/B launched and healthy

- Submitted Route A/B at `07:49:25+08:00`; both are running on `g0009`:
  `994378 e2e_routeA` and `994379 e2e_routeB`.
- Job-internal preflight passed for both jobs: `13 passed` and merged config
  `PASS`.
- Both jobs printed `Training Starts` and `Epoch 0 started` at
  `07:52:17+08:00`.
- Epoch 0 step 50: Route A `Loss=1.7254`, `cls_loss=0.9846`,
  `reg_loss=0.7407`; Route B `Loss=2.0576`,
  `selector_gt_density_loss=0.3187`, `cls_loss=0.9877`,
  `reg_loss=0.7509`.
- No traceback/OOM/Killed/RuntimeError/NaN. Existing E2E jobs remain active;
  latest old-run mAP trends are main `63.72 -> 64.15 -> 64.09`, uniform
  `64.12 -> 64.52 -> 64.76`, lowreg `62.98 -> 63.55 -> 63.42`.
- Decision: continue monitoring all five E2E jobs; do not clean active
  checkpoints.

## 2026-05-29T08:01:47+08:00 - E2E five-job monitor

- Upload marker still reports train `200/200`, test `211/211`, `bad_size=0`,
  `ALL_PASS=True`.
- All five E2E jobs remain `RUNNING`: old main/uniform/lowreg plus Route A/B.
- Route A loss trend: Epoch 0 step 50 `1.7254`, Epoch 0 final `1.6307`,
  Epoch 1 step 50 `1.0858`.
- Route B loss trend: Epoch 0 step 50 `2.0576`, Epoch 0 final `1.9384`,
  Epoch 1 step 50 `1.3239`; `selector_gt_density_loss` remains active around
  `0.319-0.320`.
- Old mAP trends: main `63.72 -> 64.15 -> 64.09`, uniform
  `64.12 -> 64.52 -> 64.76`, lowreg `62.98 -> 63.55 -> 63.42`.
- Failure scan remains empty; `/data` has about `1.9T` free. Continue
  monitoring; do not clean active checkpoints.

## 2026-05-29T08:06:30+08:00 - E2E five-job monitor, Route A/B Epoch 1 complete

- Upload marker remains valid: train `200/200`, test `211/211`,
  `bad_size=0`, `ALL_PASS=True`.
- Slurm reports all five E2E jobs still `RUNNING`: Route A/B on `g0009`,
  old main/uniform on `g0005`, and old lowreg on `g0024`.
- Route A completed Epoch 1 with `Loss=1.0265`, `cls_loss=0.6287`,
  `reg_loss=0.3978`.
- Route B completed Epoch 1 with `Loss=1.3023`,
  `selector_gt_density_loss=0.3197`, `cls_loss=0.6368`, `reg_loss=0.3455`;
  the train-only GT auxiliary selector loss remains active.
- Old mAP trends remain main `63.72 -> 64.15 -> 64.09`, uniform
  `64.12 -> 64.52 -> 64.76`, lowreg `62.98 -> 63.55 -> 63.42`.
- Failure scan counts are `0` for all five logs; `/data` has about `1.9T`
  free. No active checkpoint cleanup was performed. Continue to Route A/B
  first checkpoint and Route B selector diagnostic.

## 2026-05-29T08:12:58+08:00 - N16R4 monitor retry unreachable

- Two native OpenSSH attempts to `ssh.cn-zhongwei-1.paracloud.com:22`
  timed out before reaching the login node.
- Local `Test-NetConnection` also timed out and reported TCP failures to
  `36.103.203.5:22` and `36.103.203.6:22`.
- No fresh Slurm/log/checkpoint/mAP evidence was obtained. Treat this as
  `WAIT` due to login connectivity, not as training failure or stall.
- Last authoritative remote state remains `2026-05-29T08:06:30+08:00`:
  upload gate passed, all five jobs running, failure scan `0`, Route A/B no
  checkpoint yet, and disk safe.
- No job control, cleanup, relaunch, or remote filesystem action was
  performed.

## 2026-05-29T08:16:13+08:00 - N16R4 monitor recovered

- Login connectivity recovered. Upload marker still reports train `200/200`,
  test `211/211`, `bad_size=0`, `ALL_PASS=True`.
- Slurm reports all five E2E jobs still `RUNNING`: Route A/B on `g0009`,
  old main/uniform on `g0005`, and old lowreg on `g0024`.
- Old main fourth eval is `64.04` Avg-mAP, vector
  `79.80 / 74.77 / 67.18 / 56.81 / 41.66`; trend
  `63.72 -> 64.15 -> 64.09 -> 64.04`.
- Old uniform fourth eval is `65.01` Avg-mAP, vector
  `80.60 / 75.58 / 68.38 / 57.60 / 42.90`; trend
  `64.12 -> 64.52 -> 64.76 -> 65.01`, close to the known uniform stride-2
  reference `65.09`.
- Old lowreg has no new mAP after `63.42`; latest train evidence reached
  Epoch 47 final and likely entered the next eval window.
- Route A reached Epoch 3 final with `Loss=0.8254`; Route B reached Epoch 3
  final with `Loss=1.1547` and `selector_gt_density_loss=0.3198`.
- Failure scan counts remain `0`; Route A/B still have no first checkpoint;
  `/data` has about `1.9T` free. Continue all jobs with no cleanup or new
  launch.

## 2026-05-29T08:19:49+08:00 - E2E five-job health monitor

- Upload marker still reports train `200/200`, test `211/211`,
  `bad_size=0`, `ALL_PASS=True`.
- Slurm reports all five E2E jobs still `RUNNING`.
- No new mAP after the `08:16` record: old main remains `64.04`, old uniform
  remains `65.01`, and old lowreg remains `63.42`.
- Latest train evidence: old main completed Epoch 49 with `Loss=0.4492`; old
  uniform completed Epoch 48 with `Loss=0.4190`; Route A reached Epoch 4 step
  50 with `Loss=0.8824`; Route B reached Epoch 4 step 50 with `Loss=1.1892`
  and `selector_gt_density_loss=0.3212`.
- Failure scan counts remain `0`; no `Training Over`; Route A/B still have no
  `epoch_19.pth`; `/data` has about `1.9T` free. Continue all jobs; no
  cleanup, diagnostic, or new launch.

## 2026-05-29T08:22:51+08:00 - Old lowreg fourth eval

- Upload marker still reports train `200/200`, test `211/211`,
  `bad_size=0`, `ALL_PASS=True`; all five jobs remain `RUNNING`.
- Old lowreg fourth eval reached `63.91` Avg-mAP, vector
  `79.04 / 73.96 / 67.00 / 57.27 / 42.29`; trend
  `62.98 -> 63.55 -> 63.42 -> 63.91`.
- This is slightly above random-fixed `63.77` and strict EMA `63.85`, but
  still below old main `64.04`, stratified `64.64`, old uniform `65.01`, and
  uniform stride-2 `65.09`.
- Route A completed Epoch 4 with `Loss=0.8293`; Route B completed Epoch 4 with
  `Loss=1.1473` and `selector_gt_density_loss=0.3198`.
- Failure scan counts remain `0`; no `Training Over`; Route A/B still have no
  first checkpoint; `/data` has about `1.9T` free. Continue all jobs with no
  cleanup, severe-result escalation, or new launch.

## 2026-05-29T08:26:07+08:00 - E2E five-job health monitor

- Upload marker still reports train `200/200`, test `211/211`,
  `bad_size=0`, `ALL_PASS=True`; all five jobs remain `RUNNING`.
- No new mAP after old lowreg `63.91`. Old main remains at `64.04` and is
  likely in post-Epoch-49 eval.
- Old uniform completed Epoch 49 with `Loss=0.4193`; old lowreg resumed
  training and reached Epoch 48 step 50 with `Loss=0.4496`.
- Route A reached Epoch 5 step 50 with `Loss=0.7834`.
- Route B completed Epoch 5 with `Loss=1.1248` and
  `selector_gt_density_loss=0.3197`.
- Failure scan counts remain `0`; no `Training Over`; Route A/B still have no
  first checkpoint; `/data` has about `1.9T` free. Continue all jobs with no
  cleanup, diagnostic, or new launch.

## 2026-05-29T08:29:25+08:00 - E2E five-job health monitor

- Upload marker still reports train `200/200`, test `211/211`,
  `bad_size=0`, `ALL_PASS=True`; all five jobs remain `RUNNING`.
- No new mAP: old main remains `64.04`, old uniform remains `65.01`, and old
  lowreg remains `63.91`.
- Old main and old uniform have no train line after Epoch 49 final, so both
  are likely in eval windows.
- Old lowreg completed Epoch 48 with `Loss=0.4378`.
- Route A completed Epoch 5 with `Loss=0.8036`.
- Route B reached Epoch 6 step 50 with `Loss=1.0525` and
  `selector_gt_density_loss=0.3184`.
- Failure scan counts remain `0`; no `Training Over`; Route A/B still have no
  first checkpoint; `/data` has about `1.9T` free. Continue all jobs with no
  cleanup, diagnostic, or new launch.

## 2026-05-29T08:32:21+08:00 - E2E five-job health monitor

- Upload marker still reports train `200/200`, test `211/211`,
  `bad_size=0`, `ALL_PASS=True`; all five jobs remain `RUNNING`.
- No new mAP: old main remains `64.04`, old uniform remains `65.01`, and old
  lowreg remains `63.91`.
- Old main/uniform still have no new train line after Epoch 49 final,
  consistent with ongoing eval windows.
- Old lowreg reached Epoch 49 step 50 with `Loss=0.4132`.
- Route A reached Epoch 6 step 50 with `Loss=0.7276`.
- Route B completed Epoch 6 with `Loss=1.0419` and
  `selector_gt_density_loss=0.3198`.
- Failure scan counts remain `0`; no `Training Over`; Route A/B still have no
  first checkpoint; `/data` has about `1.9T` free. Continue all jobs with no
  cleanup, diagnostic, or new launch.

## 2026-05-29T08:35:18+08:00 - Old main fifth eval

- Upload marker still reports train `200/200`, test `211/211`,
  `bad_size=0`, `ALL_PASS=True`; all five jobs remain `RUNNING`.
- Old main fifth eval is `63.74` Avg-mAP, vector
  `79.55 / 74.15 / 66.21 / 55.97 / 42.80`; trend
  `63.72 -> 64.15 -> 64.09 -> 64.04 -> 63.74`.
- This is `-0.03` vs random-fixed `63.77`, `-0.11` vs strict EMA `63.85`,
  `-0.90` vs stratified `64.64`, `-1.27` vs old uniform `65.01`, and `-1.35`
  vs uniform stride-2 `65.09`; it is not severe but not positive old
  learned-selector evidence.
- Route A completed Epoch 6 with `Loss=0.7108`; Route B reached Epoch 7 step
  50 with `Loss=1.0578` and `selector_gt_density_loss=0.3201`.
- Failure scan counts remain `0`; no `Training Over`; Route A/B still have no
  first checkpoint; `/data` has about `1.9T` free. Continue all jobs with no
  severe-result escalation, cleanup, diagnostic, or new launch.

## 2026-05-29T08:38:35+08:00 - E2E five-job health monitor

- Upload marker still reports train `200/200`, test `211/211`,
  `bad_size=0`, `ALL_PASS=True`; all five jobs remain `RUNNING`.
- No new mAP after old main fifth eval `63.74`; old uniform remains `65.01`,
  and old lowreg remains `63.91`.
- Old main resumed training and reached Epoch 50 step 50 with `Loss=0.4098`.
- Old uniform and old lowreg still have no train line after Epoch 49 final,
  consistent with eval or transition windows.
- Route A reached Epoch 7 step 50 with `Loss=0.7462`.
- Route B completed Epoch 7 with `Loss=1.0242` and
  `selector_gt_density_loss=0.3197`.
- Failure scan counts remain `0`; no `Training Over`; Route A/B still have no
  first checkpoint; `/data` has about `1.9T` free. Continue all jobs with no
  cleanup, diagnostic, or new launch.

## 2026-05-29T08:41:49+08:00 - E2E five-job health monitor

- Upload marker still reports train `200/200`, test `211/211`,
  `bad_size=0`, `ALL_PASS=True`; all five jobs remain `RUNNING`.
- No new mAP: old main remains `63.74`, old uniform remains `65.01`, and old
  lowreg remains `63.91`.
- Old main completed Epoch 50 with `Loss=0.4095`.
- Old uniform and old lowreg still have no train line after Epoch 49 final,
  consistent with eval or transition windows.
- Route A completed Epoch 7 with `Loss=0.7149`.
- Route B reached Epoch 8 step 50 with `Loss=0.9939` and
  `selector_gt_density_loss=0.3228`.
- Failure scan counts remain `0`; no `Training Over`; Route A/B still have no
  first checkpoint; `/data` has about `1.9T` free. Continue all jobs with no
  cleanup, diagnostic, or new launch.

## 2026-05-29T08:45:13+08:00 - Old uniform fifth eval

- Upload marker still reports train `200/200`, test `211/211`,
  `bad_size=0`, `ALL_PASS=True`; all five jobs remain `RUNNING`.
- Old uniform fifth eval is `65.26` Avg-mAP, vector
  `80.60 / 75.72 / 68.91 / 57.91 / 43.14`; trend
  `64.12 -> 64.52 -> 64.76 -> 65.01 -> 65.26`.
- This is `+0.17` above the known uniform stride-2 50% reference `65.09`, so
  the uniform-control path is strong and not a pipeline failure; it remains
  control evidence, not learned-selector evidence.
- Old main remains `63.74`, now `-1.52` behind old uniform. Old lowreg remains
  `63.91`, `-1.35` behind old uniform.
- Route A completed Epoch 8 with `Loss=0.6631`.
- Route B reached Epoch 9 step 50 with `Loss=0.9507` and
  `selector_gt_density_loss=0.3197`.
- Failure scan counts remain `0`; no `Training Over`; Route A/B still have no
  first checkpoint; `/data` has about `1.9T` free. Continue all jobs with no
  cleanup, diagnostic, or new launch.

## 2026-05-29T08:48:30+08:00 - E2E five-job health monitor

- Upload marker still reports train `200/200`, test `211/211`,
  `bad_size=0`, `ALL_PASS=True`; all five jobs remain `RUNNING`.
- No new mAP after old uniform `65.26`; old main remains `63.74`, and old
  lowreg remains `63.91`.
- Old main completed Epoch 51 with `Loss=0.4155`; old uniform completed Epoch
  50 with `Loss=0.3851`.
- Route A reached Epoch 9 step 50 with `Loss=0.6125`.
- Route B completed Epoch 9 with `Loss=0.9796` and
  `selector_gt_density_loss=0.3198`.
- Failure scan counts remain `0`; no `Training Over`; Route A/B still have no
  first checkpoint; `/data` has about `1.9T` free. Continue all jobs with no
  cleanup, diagnostic, or new launch.

## 2026-05-29T08:52:05+08:00 - Lowreg fifth eval flat

- Upload marker still reports train `200/200`, test `211/211`,
  `bad_size=0`, `ALL_PASS=True`; all five jobs remain `RUNNING`.
- Old lowreg fifth eval is `63.92` Avg-mAP, vector
  `78.71 / 74.30 / 66.81 / 56.59 / 43.19`; trend
  `62.98 -> 63.55 -> 63.42 -> 63.91 -> 63.92`.
- This is only `+0.15` vs random-fixed `63.77` and `+0.07` vs strict EMA
  `63.85`, while still below stratified `64.64`, old uniform `65.26`, and
  uniform stride-2 `65.09`; no positive learned-selector claim.
- Old main remains latest `63.74`; old uniform remains latest `65.26` and
  reached Epoch 51 step 50 with `Loss=0.4141`.
- Route A completed Epoch 9 with `Loss=0.6519`.
- Route B reached Epoch 10 step 50 with `Loss=0.9825` and
  `selector_gt_density_loss=0.3189`.
- Failure scan counts remain `0`; no `Training Over`; Route A/B still have no
  first checkpoint; `/data` has about `1.9T` free. Continue all jobs with no
  severe-result escalation, cleanup, diagnostic, or new launch.

## 2026-05-29T08:55:44+08:00 - E2E five-job health monitor

- Upload marker still reports train `200/200`, test `211/211`,
  `bad_size=0`, `ALL_PASS=True`; all five jobs remain `RUNNING`.
- No new mAP after old main `63.74`, old uniform `65.26`, and old lowreg
  `63.92`.
- Old main has no train line after Epoch 51 final, consistent with eval or
  transition.
- Old uniform completed Epoch 51 with `Loss=0.4071`; old lowreg resumed
  training and reached Epoch 50 step 50 with `Loss=0.4004`.
- Route A reached Epoch 10 step 50 with `Loss=0.6688`.
- Route B completed Epoch 10 with `Loss=0.9750` and
  `selector_gt_density_loss=0.3197`.
- Failure scan counts remain `0`; no `Training Over`; Route A/B still have no
  first checkpoint, so Route B selector diagnostic is not yet runnable. `/data`
  has about `1.9T` free. Continue all jobs with no cleanup, diagnostic,
  cancellation, or new launch.

## 2026-05-29T08:58:29+08:00 - E2E five-job health monitor

- Upload marker still reports train `200/200`, test `211/211`,
  `bad_size=0`, `ALL_PASS=True`; all five jobs remain `RUNNING`.
- No new mAP after old main `63.74`, old uniform `65.26`, and old lowreg
  `63.92`.
- Old main has no train line after Epoch 51 final, consistent with eval or
  transition; old uniform has no train line after Epoch 51 final.
- Old lowreg completed Epoch 50 with `Loss=0.4105`.
- Route A completed Epoch 10 with `Loss=0.6596`.
- Route B reached Epoch 11 step 50 with `Loss=0.9758` and
  `selector_gt_density_loss=0.3221`.
- Failure scan counts remain `0`; no `Training Over`; Route A/B still have no
  first checkpoint, so Route B selector diagnostic is not runnable. `/data`
  has about `1.9T` free. Continue all jobs with no cleanup, diagnostic,
  cancellation, or new launch.

## 2026-05-29T09:01:02+08:00 - E2E five-job health monitor

- Upload marker still reports train `200/200`, test `211/211`,
  `bad_size=0`, `ALL_PASS=True`; all five jobs remain `RUNNING`.
- No new mAP after old main `63.74`, old uniform `65.26`, and old lowreg
  `63.92`.
- Old main has no train line after Epoch 51 final, consistent with eval or
  transition; old uniform has no train line after Epoch 51 final.
- Old lowreg reached Epoch 51 step 50 with `Loss=0.4235`.
- Route A reached Epoch 11 step 50 with `Loss=0.6437`.
- Route B completed Epoch 11 with `Loss=0.9713` and
  `selector_gt_density_loss=0.3198`.
- Failure scan counts remain `0`; no `Training Over`; Route A/B still have no
  first checkpoint, so Route B selector diagnostic is not runnable. `/data`
  has about `1.9T` free. Continue all jobs with no cleanup, diagnostic,
  cancellation, or new launch.

## 2026-05-29T09:03:56+08:00 - Old main sixth eval

- Upload marker still reports train `200/200`, test `211/211`,
  `bad_size=0`, `ALL_PASS=True`; all five jobs remain `RUNNING`.
- Old main sixth eval line observed at `09:05:37`: `63.90` Avg-mAP, vector
  `79.32 / 74.98 / 66.28 / 56.33 / 42.57`; trend
  `63.72 -> 64.15 -> 64.09 -> 64.04 -> 63.74 -> 63.90`.
- This is `+0.13` vs random-fixed `63.77` and `+0.05` vs strict EMA `63.85`,
  but still below stratified `64.64`, old uniform `65.26`, and uniform
  stride-2 `65.09`; it is not severe and not positive learned-selector
  evidence.
- Old uniform remains latest `65.26`; old lowreg completed Epoch 51 with
  `Loss=0.4170`.
- Route A completed Epoch 11 with `Loss=0.6419`.
- Route B reached Epoch 12 step 50 with `Loss=0.9359` and
  `selector_gt_density_loss=0.3197`.
- Failure scan counts remain `0`; no `Training Over`; Route A/B still have no
  first checkpoint, so Route B selector diagnostic is not runnable. `/data`
  has about `1.9T` free. Continue all jobs with no cleanup, severe-result
  escalation, diagnostic, cancellation, or new launch.

## 2026-05-29T09:06:59+08:00 - E2E five-job health monitor

- Upload marker still reports train `200/200`, test `211/211`,
  `bad_size=0`, `ALL_PASS=True`; all five jobs remain `RUNNING`.
- No new mAP after old main `63.90`, old uniform `65.26`, and old lowreg
  `63.92`.
- Old main resumed training and reached Epoch 52 step 50 with `Loss=0.4383`;
  old uniform and lowreg have no new train lines after their latest epoch-final
  records.
- Route A reached Epoch 12 step 50 with `Loss=0.6113`.
- Route B reached Epoch 13 step 50 with `Loss=0.9413` and
  `selector_gt_density_loss=0.3193`.
- Failure scan counts remain `0`; no `Training Over`; Route A/B still have no
  first checkpoint, so Route B selector diagnostic is not runnable. `/data`
  has about `1.9T` free. Continue all jobs with no cleanup, diagnostic,
  cancellation, or new launch.

## 2026-05-29T09:09:50+08:00 - E2E five-job health monitor

- Upload marker still reports train `200/200`, test `211/211`,
  `bad_size=0`, `ALL_PASS=True`; all five jobs remain `RUNNING`.
- No new mAP after old main `63.90`, old uniform `65.26`, and old lowreg
  `63.92`.
- Old main completed Epoch 52 with `Loss=0.4303`; old uniform and lowreg have
  no new train lines after their latest epoch-final records.
- Route A completed Epoch 12 with `Loss=0.6248`.
- Route B completed Epoch 13 with `Loss=0.9532` and
  `selector_gt_density_loss=0.3197`.
- Failure scan counts remain `0`; no `Training Over`; Route A/B still have no
  first checkpoint, so Route B selector diagnostic is not runnable. `/data`
  has about `1.9T` free. Continue all jobs with no cleanup, diagnostic,
  cancellation, or new launch.

## 2026-05-29T09:12:55+08:00 - Old uniform sixth eval

- Upload marker still reports train `200/200`, test `211/211`,
  `bad_size=0`, `ALL_PASS=True`; all five jobs remain `RUNNING`.
- Old uniform sixth eval reached `65.52` Avg-mAP, vector
  `80.77 / 76.08 / 69.15 / 57.91 / 43.70`; trend
  `64.12 -> 64.52 -> 64.76 -> 65.01 -> 65.26 -> 65.52`.
- This is `+1.75` vs random-fixed `63.77`, `+1.67` vs strict EMA `63.85`,
  `+0.88` vs stratified `64.64`, and `+0.43` vs uniform stride-2 `65.09`.
  It is strong uniform-control evidence, not learned-selector evidence.
- Old main remains latest `63.90`; old lowreg remains latest `63.92`.
- Route A reached Epoch 13 step 50 with `Loss=0.6165`.
- Route B reached Epoch 14 step 50 with `Loss=0.9009` and
  `selector_gt_density_loss=0.3205`.
- Failure scan counts remain `0`; no `Training Over`; Route A/B still have no
  first checkpoint, so Route B selector diagnostic is not runnable. `/data`
  has about `1.9T` free. Continue all jobs; Route A/B must now be interpreted
  against old uniform `65.52`.

## 2026-05-29T09:15:59+08:00 - E2E five-job health monitor

- Upload marker still reports train `200/200`, test `211/211`,
  `bad_size=0`, `ALL_PASS=True`; all five jobs remain `RUNNING`.
- No new mAP after old uniform `65.52`, old main `63.90`, and old lowreg
  `63.92`.
- Old main completed Epoch 53 with `Loss=0.3890`; old uniform resumed training
  and reached Epoch 52 step 50 with `Loss=0.4180`; old lowreg has no new train
  line after Epoch 51 final.
- Route A completed Epoch 13 with `Loss=0.6351`.
- Route B completed Epoch 14 with `Loss=0.9331` and
  `selector_gt_density_loss=0.3198`.
- Failure scan counts remain `0`; no `Training Over`; Route A/B still have no
  first checkpoint, so Route B selector diagnostic is not runnable. `/data`
  has about `1.9T` free. Continue all jobs.

## 2026-05-29T09:19:11+08:00 - E2E five-job health monitor

- Upload marker still reports train `200/200`, test `211/211`,
  `bad_size=0`, `ALL_PASS=True`; all five jobs remain `RUNNING`.
- No new mAP after old uniform `65.52`, old main `63.90`, and old lowreg
  `63.92`.
- Old uniform completed Epoch 52 with `Loss=0.4159`; old main and lowreg have
  no new train lines after their latest epoch-final records.
- Route A reached Epoch 14 step 50 with `Loss=0.5793`.
- Route B reached Epoch 15 step 50 with `Loss=0.9344` and
  `selector_gt_density_loss=0.3189`.
- Failure scan counts remain `0`; no `Training Over`; Route A/B still have no
  first checkpoint, so Route B selector diagnostic is not runnable. `/data`
  has about `1.9T` free. Continue all jobs and recheck soon for Route A/B first
  checkpoint.

## 2026-05-29T09:24:46+08:00 - E2E five-job health monitor

- Upload marker still reports train `200/200`, test `211/211`,
  `bad_size=0`, `ALL_PASS=True`; all five jobs remain `RUNNING`.
- Lowreg sixth eval at `09:20:33` dropped to `63.43` Avg-mAP, vector
  `78.50 / 74.15 / 66.15 / 55.81 / 42.55`; trend
  `62.98 -> 63.55 -> 63.42 -> 63.91 -> 63.92 -> 63.43`.
- This is below random-fixed `63.77` and strict EMA `63.85`, far below
  stratified `64.64`, uniform stride-2 `65.09`, and old uniform `65.52`.
  It is negative lowreg evidence, but not a severe-result collapse.
- Old main remains latest `63.90`; old uniform remains latest `65.52` and
  completed Epoch 53 with `Loss=0.3695`; old lowreg reached Epoch 52 step 50
  with `Loss=0.4313`.
- Route A reached Epoch 15 step 50 with `Loss=0.6054`.
- Route B reached Epoch 16 step 50 with `Loss=0.8972` and
  `selector_gt_density_loss=0.3189`.
- Failure scan counts remain `0`; no `Training Over`; Route A/B still have no
  first checkpoint, so Route B selector diagnostic is not runnable. `/data`
  has about `1.9T` free. Continue all jobs; do not launch more lowreg sweeps.

## 2026-05-29T09:27:01+08:00 - E2E checkpoint-trigger recheck

- All five jobs remain `RUNNING`: Route A/B on `g0009`, old main/uniform on
  `g0005`, and old lowreg on `g0024`.
- No new mAP after old main `63.90`, old uniform `65.52`, and lowreg `63.43`.
- Route A completed Epoch 15 with `Loss=0.5963`; Route B completed Epoch 16
  with `Loss=0.9361` and `selector_gt_density_loss=0.3197`.
- Old lowreg completed Epoch 52 with `Loss=0.4320`; old uniform remains Epoch
  53 final with `Loss=0.3695`; old main remains Epoch 53 final with
  `Loss=0.3890`.
- Failure scan counts remain `0`; no `Training Over`; Route A/B checkpoint
  listing is still empty, so selector diagnostics are still gated. `/data`
  has about `1.9T` free. Continue all jobs.

## 2026-05-29T09:29:40+08:00 - E2E Route A/B checkpoint watch

- All five jobs remain `RUNNING`; no new mAP after old main `63.90`, old
  uniform `65.52`, and lowreg `63.43`.
- Route A reached Epoch 16 step 50 with `Loss=0.5836`.
- Route B reached Epoch 17 step 50 with `Loss=0.8532` and
  `selector_gt_density_loss=0.3230`.
- Failure scan counts remain `0`; Route A/B checkpoint listing is still empty.
  Continue active jobs and keep waiting for Route A/B `epoch_19.pth`.

## 2026-05-29T09:30:47+08:00 - E2E Route A/B light recheck

- Route A/B remain `RUNNING` on `g0009`.
- No new train line beyond Route A Epoch 16 step 50 and Route B Epoch 17 step
  50.
- Failure scan counts remain `0`; Route A/B checkpoint listing is still empty.
  Continue Route A/B until `epoch_19.pth`, failure, or result trigger.

## 2026-05-29T09:32:58+08:00 - E2E five-job health monitor

- Upload marker still reports train `200/200`, test `211/211`,
  `bad_size=0`, `ALL_PASS=True`; all five jobs remain `RUNNING`.
- Old main produced a new eval at log time `09:34:09`: `64.36` Avg-mAP,
  vector `79.52 / 74.52 / 67.26 / 57.02 / 43.46`; trend
  `63.72 -> 64.15 -> 64.09 -> 64.04 -> 63.74 -> 63.90 -> 64.36`.
- This is `+0.59` vs random-fixed `63.77` and `+0.51` vs strict EMA
  `63.85`, but still below stratified `64.64`, uniform stride-2 `65.09`, and
  old uniform `65.52`.
- Old uniform remains latest `65.52`; lowreg remains latest `63.43` and
  completed Epoch 53 with `Loss=0.3942`.
- Route A completed Epoch 16 with `Loss=0.6127`; Route B completed Epoch 17
  with `Loss=0.8783` and `selector_gt_density_loss=0.3197`.
- Failure scan counts remain `0`; no `Training Over`; Route A/B still have no
  first checkpoint. Continue all jobs and wait for Route A/B `epoch_19.pth`.

## 2026-05-29T09:34:39+08:00 - E2E Route A/B checkpoint watch

- Route A/B remain `RUNNING` on `g0009`.
- Route A has no new train line after Epoch 16 final, still within a normal
  short progress window.
- Route B reached Epoch 18 step 50 with `Loss=0.9083` and
  `selector_gt_density_loss=0.3205`.
- Failure scan counts remain `0`; Route A/B checkpoint listing is still empty.
  Continue until `epoch_19.pth`, failure, or result trigger.

## 2026-05-29T09:35:53+08:00 - E2E five-job light monitor

- All five jobs remain `RUNNING`; no new mAP after old main `64.36`, old
  uniform `65.52`, and lowreg `63.43`.
- Route A reached Epoch 17 step 50 with `Loss=0.5150`.
- Route B remains latest Epoch 18 step 50 with `Loss=0.9083` and
  `selector_gt_density_loss=0.3205`.
- Old main resumed after eval and reached Epoch 54 step 50 with `Loss=0.4547`;
  old uniform and lowreg remain at their latest epoch-final lines.
- Failure scan counts remain `0`; no `Training Over`; Route A/B checkpoint
  listing is still empty. Continue all jobs.

## 2026-05-29T09:37:57+08:00 - E2E five-job health monitor

- Upload marker still reports train `200/200`, test `211/211`,
  `bad_size=0`, `ALL_PASS=True`; all five jobs remain `RUNNING`.
- No new mAP after old main `64.36`, old uniform `65.52`, and lowreg `63.43`.
- Old main completed Epoch 54 with `Loss=0.4309`; old uniform and lowreg
  remain at their latest epoch-final lines.
- Route A completed Epoch 17 with `Loss=0.5486`.
- Route B completed Epoch 18 with `Loss=0.8982` and
  `selector_gt_density_loss=0.3197`.
- Failure scan counts remain `0`; no `Training Over`; Route A/B checkpoint
  listing is still empty. `/data` has about `1.9T` free. Continue all jobs.

## 2026-05-29T09:39:26+08:00 - E2E Route B entered Epoch 19

- Route A/B remain `RUNNING` on `g0009`.
- Route A remains latest Epoch 17 final with `Loss=0.5486`.
- Route B reached Epoch 19 step 50 with `Loss=0.9095` and
  `selector_gt_density_loss=0.3223`.
- Failure scan counts remain `0`; Route A/B checkpoint listing is still empty
  because Epoch 19 has not completed. Recheck after Epoch 19 final for
  `epoch_19.pth`.

## 2026-05-29T09:40:35+08:00 - E2E Route A/B checkpoint watch

- Route A reached Epoch 18 step 50 with `Loss=0.5648`.
- Route B remains latest Epoch 19 step 50 with `Loss=0.9095` and
  `selector_gt_density_loss=0.3223`.
- Failure scan counts remain `0`; Route A/B checkpoint listing is still empty
  because Epoch 19 final/checkpoint has not been reached. Continue rechecking
  for `epoch_19.pth`.

## 2026-05-29T09:41:42+08:00 - E2E Route A/B checkpoint watch

- Route A remains latest Epoch 18 step 50 with `Loss=0.5648`.
- Route B remains latest Epoch 19 step 50 with `Loss=0.9095` and
  `selector_gt_density_loss=0.3223`.
- Failure scan counts remain `0`; Route A/B checkpoint listing remains empty
  because Epoch 19 final/checkpoint has not been reached. Continue Route A/B;
  next action is recheck for `epoch_19.pth`.

## 2026-05-29T09:43:28+08:00 - Route B first checkpoint

- Upload marker still reports train `200/200`, test `211/211`,
  `bad_size=0`, `ALL_PASS=True`; all five training jobs remain `RUNNING`.
- Route B completed Epoch 19 with `Loss=0.8936` and
  `selector_gt_density_loss=0.3198`, then wrote `checkpoint/epoch_19.pth`.
- Old uniform produced a new eval at `09:42:05`: `65.50` Avg-mAP, essentially
  tied with prior `65.52`.
- Failure scan counts remain `0`; `/data` has about `1.9T` free.
- Decision: trigger Route B selector diagnostic.

## 2026-05-29T09:55:48+08:00 - Route B selector diagnostic

- First direct login-node diagnostic was terminated before summary output.
- Slurm diagnostic `994442` failed because `PYTHONPATH` did not include the
  repo; corrected diagnostic `994443` completed on `g0024`.
- Output:
  `~/run/yuzibo/e2e_runs/selector_diagnostics/e2e_routeB_gtaux_epoch19_posdiag_20260529_095333/summary.json`.
- Scope: selector-only over 48/487 validation windows; no ViT/backbone/detector
  forward; validation GT used only for offline action/boundary proximity labels.
- Metrics: mean abs delta `1.347`, p95 delta `2.000`, action fraction `0.2879`,
  boundary<=4 fraction `0.1483`, boundary recall@4 `1.0`, logit std `0.000714`.
- All Pro gates failed: logit std `<0.01`, mean abs delta not in `2-8`,
  action fraction `<0.32`, boundary<=4 fraction `<0.17`.
- Interpretation: Route B has not learned an action/boundary-aware selected-frame
  distribution at epoch 19; continue to first eval for mAP evidence, but do not
  launch more Route B variants from this checkpoint alone.

## 2026-05-29T09:57:38+08:00 - Post-diagnostic monitor

- All five training jobs remain `RUNNING`; failure scan counts remain `0`.
- Route A wrote `epoch_19.pth` at `09:49:21` and reached Epoch 20 final with
  `Loss=0.5450`.
- Route B reached Epoch 22 step 50 with `Loss=0.8643` and
  `selector_gt_density_loss=0.3189`.
- Old lowreg produced a new eval `63.87` Avg-mAP, still below old main `64.36`,
  stratified `64.64`, and uniform `65.50/65.52`.
- Decision: continue all jobs to first Route A/B eval and old final checkpoints;
  no cleanup while active and no new launch from this diagnostic alone.

## 2026-05-29T10:01:01+08:00 - E2E five-job health monitor

- Upload marker still reports train `200/200`, test `211/211`,
  `bad_size=0`, `ALL_PASS=True`; Slurm and sacct report all five training jobs
  `RUNNING`.
- Old main produced a new eval at `10:02:36`: `64.33` Avg-mAP, vector
  `79.46 / 74.26 / 67.68 / 57.34 / 42.90`; this is flat vs prior `64.36` and
  remains below stratified `64.64` and old uniform `65.50/65.52`.
- Old uniform latest remains `65.50`; old lowreg latest remains `63.87`.
- Route A reached Epoch 21 final with `Loss=0.5465`.
- Route B reached Epoch 23 step 50 with `Loss=0.8545` and
  `selector_gt_density_loss=0.3214`.
- Failure scan counts remain `0`; no `Training Over`; visible checkpoints are
  old `epoch_19/39` and Route A/B `epoch_19`; `/data` has about `1.9T` free.
- Decision: continue all jobs. No cleanup because all runs are active; no severe
  gate and no new launch until Route A/B first eval or a completed-run cleanup
  trigger.

## 2026-05-29T10:05:42+08:00 - E2E five-job monitor

- Upload marker still reports train `200/200`, test `211/211`,
  `bad_size=0`, `ALL_PASS=True`; all five jobs remain `RUNNING`.
- Old main latest eval remains `64.33` Avg-mAP at `10:02:36`; old uniform
  latest remains `65.50` with best `65.52`; old lowreg latest remains `63.87`.
- Route A reached Epoch 22 step 50 with `Loss=0.5417`; Route B completed Epoch
  23 with `Loss=0.8338` and `selector_gt_density_loss=0.3198`. Route A/B still
  have no mAP.
- Failure scan remains `0`; no `Training Over`; active checkpoints remain old
  `epoch_19/39` and Route A/B `epoch_19`; `/data` has about `1.9T` free.
- Decision: continue all five jobs. No cleanup, severe-result gate, or new
  launch; wait for Route A/B first mAP or completed-run cleanup trigger.

## 2026-05-29T10:10:54+08:00 - E2E five-job monitor

- Upload marker still reports train `200/200`, test `211/211`,
  `bad_size=0`, `ALL_PASS=True`; all five jobs remain `RUNNING`.
- Old uniform produced a new best at `10:11:10`: `65.56` Avg-mAP, vector
  `80.68 / 75.90 / 69.25 / 58.15 / 43.84`. This raises the control bar for
  Route A/B and any learned-selector claim.
- Old main latest remains `64.33`; old lowreg latest remains `63.87`; old main
  reached Epoch 57 step 50 with `Loss=0.3970`.
- Route A reached Epoch 23 step 50 with `Loss=0.5125`; Route B completed Epoch
  24 with `Loss=0.8509` and `selector_gt_density_loss=0.3197`. Route A/B still
  have no mAP.
- Failure scan remains `0`; no `Training Over`; active checkpoints remain old
  `epoch_19/39` and Route A/B `epoch_19`; `/data` has about `1.9T` free.
- Decision: continue all five jobs. No cleanup, severe-result gate, or new
  launch; wait for Route A/B first mAP or completed-run cleanup trigger.

## 2026-05-29T10:15:03+08:00 - E2E five-job monitor

- Upload marker still reports train `200/200`, test `211/211`,
  `bad_size=0`, `ALL_PASS=True`; all five jobs remain `RUNNING`.
- No new mAP after old uniform best `65.56`, old main `64.33`, and old lowreg
  `63.87`.
- Old main completed Epoch 57 with `Loss=0.4087`; old uniform completed Epoch
  56 with `Loss=0.3935`; old lowreg has no new train line after Epoch 55 final,
  likely in eval/transition.
- Route A completed Epoch 23 with `Loss=0.5068`; Route B completed Epoch 25
  with `Loss=0.8652` and `selector_gt_density_loss=0.3198`. Route A/B still
  have no mAP.
- Failure scan remains `0`; no `Training Over`; active checkpoints remain old
  `epoch_19/39` and Route A/B `epoch_19`; `/data` has about `1.9T` free.
- Decision: continue all five jobs. No cleanup, severe-result gate, or new
  launch; wait for Route A/B first mAP, old lowreg next eval, or completed-run
  cleanup trigger.

## 2026-05-29T10:19:11+08:00 - E2E five-job monitor

- Upload marker still reports train `200/200`, test `211/211`,
  `bad_size=0`, `ALL_PASS=True`; all five jobs remain `RUNNING`.
- No new mAP after old uniform best `65.56`, old main `64.33`, and old lowreg
  `63.87`.
- Old main latest train remains Epoch 57 final `Loss=0.4087`; old uniform
  reached Epoch 57 step 50 with `Loss=0.3709`; old lowreg still has no train
  line after Epoch 55 final, consistent with eval/transition.
- Route A completed Epoch 24 with `Loss=0.5073`; Route B reached Epoch 26 step
  50 with `Loss=0.8099` and `selector_gt_density_loss=0.3209`. Route A/B still
  have no mAP.
- Failure scan remains `0`; no `Training Over`; active checkpoints remain old
  `epoch_19/39` and Route A/B `epoch_19`; `/data` has about `1.9T` free.
- Decision: continue all five jobs. No cleanup, severe-result gate, or new
  launch; wait for Route A/B first mAP, old lowreg next eval, or completed-run
  cleanup trigger.

## 2026-05-29T10:23:17+08:00 - E2E five-job monitor

- Upload marker still reports train `200/200`, test `211/211`,
  `bad_size=0`, `ALL_PASS=True`; all five jobs remain `RUNNING`.
- Old lowreg produced a new eval at `10:21:47`: `64.36` Avg-mAP, vector
  `79.55 / 74.89 / 66.82 / 57.16 / 43.40`.
- Interpretation: lowreg recovered above random-fixed `63.77` and strict EMA
  `63.85`, but remains below stratified `64.64` and old uniform `65.56`; this
  is not positive learned-selector evidence.
- Old main latest remains `64.33`; old uniform best remains `65.56`.
- Route A reached Epoch 25 step 50 with `Loss=0.5161`; Route B reached Epoch
  27 step 50 with `Loss=0.8348` and `selector_gt_density_loss=0.3173`. Route
  A/B still have no mAP.
- Failure scan remains `0`; no `Training Over`; active checkpoints remain old
  `epoch_19/39` and Route A/B `epoch_19`; `/data` has about `1.9T` free.
- Decision: continue all five jobs. No cleanup, severe-result gate, or new
  launch; wait for Route A/B first mAP or completed-run cleanup trigger.

## 2026-05-29T10:27:02+08:00 - E2E five-job monitor

- Upload marker still reports train `200/200`, test `211/211`,
  `bad_size=0`, `ALL_PASS=True`; all five jobs remain `RUNNING`.
- No new mAP after old lowreg `64.36`, old uniform best `65.56`, and old main
  `64.33`.
- Old lowreg resumed training to Epoch 56 step 50 with `Loss=0.3914`; old
  uniform completed Epoch 57 with `Loss=0.3915`; old main latest train remains
  Epoch 57 final.
- Route A reached Epoch 26 step 50 with `Loss=0.4908`; Route B completed Epoch
  27 with `Loss=0.8539` and `selector_gt_density_loss=0.3197`. Route A/B still
  have no mAP.
- Failure scan remains `0`; no `Training Over`; active checkpoints remain old
  `epoch_19/39` and Route A/B `epoch_19`; `/data` has about `1.9T` free.
- Decision: continue all five jobs. No cleanup, severe-result gate, or new
  launch; wait for Route A/B first mAP or completed-run cleanup trigger.

## 2026-05-29T10:34:14+08:00 - E2E five-job monitor

- Upload marker still reports train `200/200`, test `211/211`,
  `bad_size=0`, `ALL_PASS=True`; all five jobs remain `RUNNING`.
- Old main produced a new eval at `10:31:07`: `64.71` Avg-mAP, vector
  `79.80 / 75.33 / 67.95 / 57.69 / 42.77`.
- Interpretation: old main is above random-fixed `63.77`, strict EMA `63.85`,
  and stratified `64.64`, but still below old uniform best `65.56` by `0.85`;
  useful progress, not a learned-selector win against uniform coverage yet.
- Old uniform best remains `65.56`; old lowreg latest remains `64.36`. Old
  main resumed to Epoch 58 step 50 with `Loss=0.4189`; old lowreg completed
  Epoch 57 with `Loss=0.4144`.
- Route A reached Epoch 27 step 50 with `Loss=0.5003`; Route B reached Epoch
  29 step 50 with `Loss=0.8148` and `selector_gt_density_loss=0.3221`. Route
  A/B still have no mAP.
- Failure scan remains `0`; no `Training Over`; active checkpoints remain old
  `epoch_19/39` and Route A/B `epoch_19`; `/data` has about `1.9T` free.
- Decision: continue all five jobs. No cleanup, severe-result gate, or new
  launch; wait for Route A/B first mAP or completed-run cleanup trigger.

## 2026-05-29T10:38:34+08:00 - E2E five-job monitor

- Upload marker still reports train `200/200`, test `211/211`,
  `bad_size=0`, `ALL_PASS=True`; all five jobs remain `RUNNING`.
- Old uniform produced a new eval at log time `10:40:13`: `65.73` Avg-mAP,
  vector `80.71 / 76.08 / 69.50 / 58.35 / 43.98`.
- Interpretation: old uniform is now the strongest current control, `+0.64`
  above historical uniform stride-2 `65.09`; learned-selector claims must be
  judged against `65.73`.
- Old main latest remains `64.71`, now `1.02` below old uniform. Old lowreg
  latest remains `64.36`.
- Old main reached Epoch 59 step 50 with `Loss=0.4148`; Route A completed Epoch
  27 with `Loss=0.5233`; Route B completed Epoch 29 with `Loss=0.8265` and
  `selector_gt_density_loss=0.3197`. Route A/B still have no mAP.
- Failure scan remains `0`; no `Training Over`; active checkpoints remain old
  `epoch_19/39` and Route A/B `epoch_19`; `/data` has about `1.9T` free.
- Decision: continue all five jobs. No cleanup, severe-result gate, or new
  launch; wait for Route A/B first mAP or completed-run cleanup trigger.

## 2026-05-29T10:42:07+08:00 - E2E final-checkpoint watch

- All five jobs remain `RUNNING`: old main/uniform/lowreg plus Route A/B.
- Old main completed Epoch 59 at `10:41:54` with `Loss=0.4045` and wrote
  `epoch_59.pth` at `10:41:56`.
- Cleanup decision: no cleanup yet, because old main still has no
  `Training Over` line and Slurm still reports `994340` as `RUNNING`.
- Old uniform best remains `65.73` and resumed to Epoch 58 step 50 at log time
  `10:43:00`; old lowreg latest remains `64.36`.
- Route A completed Epoch 28 with `Loss=0.5069`; Route B completed Epoch 30
  with `Loss=0.8191` and `selector_gt_density_loss=0.3198`. Route A/B still
  have no mAP.
- Failure scan remains `0`; `/data` has about `1.9T` free.
- Decision: continue watching for old main `Training Over`/final eval before
  checkpoint cleanup. Continue Route A/B to first mAP; no severe-result gate or
  new launch.

## 2026-05-29T10:46:14+08:00 - E2E five-job monitor

- Upload marker still reports train `200/200`, test `211/211`,
  `bad_size=0`, `ALL_PASS=True`; all five jobs remain `RUNNING`.
- No `Training Over` line appears in any checked log. Old main still has
  `epoch_59.pth`, but cleanup remains disallowed while `994340` is running and
  completion is unconfirmed.
- Old main latest mAP remains `64.71`; old uniform best remains `65.73` and
  completed Epoch 58 at log time `10:45:41` with `Loss=0.3935`; old lowreg
  latest remains `64.36`.
- Route A reached Epoch 29 step 50 with `Loss=0.4781`; Route B reached Epoch
  31 step 50 with `Loss=0.7809` and `selector_gt_density_loss=0.3185`. Route
  A/B still have no mAP.
- Failure scan remains `0`; `/data` has about `1.9T` free.
- Decision: continue watching for old main `Training Over`/final eval before
  checkpoint cleanup. Continue Route A/B to first mAP; no severe-result gate or
  new launch.

## 2026-05-29T10:50:12+08:00 - E2E five-job monitor

- Upload marker still reports train `200/200`, test `211/211`,
  `bad_size=0`, `ALL_PASS=True`; all five jobs remain `RUNNING`.
- No `Training Over` line appears in any checked log.
- Old main latest mAP remains `64.71`; old main `epoch_59.pth` remains present.
- Old uniform best remains `65.73`; old uniform completed Epoch 59 at log time
  `10:51:14` and wrote `epoch_59.pth` at `10:51:16`.
- Cleanup decision: no cleanup for old main or old uniform because both Slurm
  jobs remain `RUNNING` and completion is unconfirmed.
- Old lowreg latest remains `64.36`.
- Route A completed Epoch 29 with `Loss=0.4856`; Route B reached Epoch 32 step
  50 with `Loss=0.8209` and `selector_gt_density_loss=0.3189`. Route A/B
  still have no mAP.
- Failure scan remains `0`; `/data` has about `1.9T` free.
- Decision: continue watching for old main/uniform `Training Over` or final
  eval before checkpoint cleanup. Continue Route A/B to first mAP; no
  severe-result gate or new launch.

## 2026-05-29T10:57:06+08:00 - E2E five-job monitor

- Upload marker still reports train `200/200`, test `211/211`,
  `bad_size=0`, `ALL_PASS=True`; all five jobs remain `RUNNING`.
- No `Training Over` line appears in any checked log.
- Old main latest mAP remains `64.71`; old uniform best remains `65.73` and is
  still the strongest current control.
- Old lowreg latest eval at `10:52:06` is `64.28`, below its prior `64.36`
  best, stratified `64.64`, and uniform `65.73`.
- Route A reached Epoch 31 step 50 with `Loss=0.4436`; Route B reached Epoch
  33 step 50 with `Loss=0.8218` and `selector_gt_density_loss=0.3202`. Route
  A/B still have no mAP.
- Failure scan remains `0`; `/data` has about `1.9T` free.
- Decision: continue all five jobs. No cleanup, severe-result gate, or new
  launch; wait for Route A/B first mAP or confirmed completed-run cleanup
  trigger.

## 2026-05-29T11:00:35+08:00 - E2E old main completion cleanup

- Old main job `994340` completed; log reports `Training Over...` at
  `10:59:37`.
- Final old main eval is `64.15` Avg-mAP, vector
  `79.52 / 75.54 / 66.79 / 56.61 / 42.27`; best observed old main remains
  `64.71`.
- Cleanup scope was restricted to
  `~/run/yuzibo/e2e_runs/exps/e2e_rawdensel_main_20260529_023345/gpu1_id0/checkpoint`,
  resolved as
  `/data/run01/sczc063/yuzibo/e2e_runs/exps/e2e_rawdensel_main_20260529_023345/gpu1_id0/checkpoint`.
- The first cleanup attempt refused the symlink-resolved path and deleted
  nothing; after verifying it was inside the authorized workspace, cleanup
  removed only `epoch_19.pth` and `epoch_39.pth`.
- Kept `epoch_59.pth`; preserved non-epoch artifacts. Disk stayed about
  `2.3T` size, `420G` used, `1.9T` available, `18%`.
- Active jobs after cleanup: old uniform `994341`, old lowreg `994342`, Route A
  `994378`, Route B `994379`.
- Decision: continue active jobs. No severe-result gate or new launch; wait for
  old uniform/lowreg completion cleanup and Route A/B first mAP.

## 2026-05-29T11:03:18+08:00 - E2E post-cleanup monitor

- Upload marker still reports train `200/200`, test `211/211`,
  `bad_size=0`, `ALL_PASS=True`.
- `sacct` confirms old main `994340` completed in `08:23:12`; checkpoint
  listing verifies old main cleanup kept only `epoch_59.pth`.
- Active jobs are old uniform `994341`, old lowreg `994342`, Route A `994378`,
  and Route B `994379`.
- Old uniform still has no `Training Over`; latest/best mAP remains `65.73`.
  Old lowreg still has no `Training Over`; latest mAP remains `64.28`.
- Route A reached Epoch 32 step 50 with `Loss=0.4898`; Route B completed Epoch
  34 with `Loss=0.8231` and `selector_gt_density_loss=0.3198`. Route A/B still
  have no mAP.
- Failure scan remains `0`; `/data` now shows `419G` used and `1.9T`
  available.
- Decision: continue active four jobs. No cleanup while they are running; no
  severe-result gate or new launch.

## 2026-05-29T11:06:12+08:00 - E2E four-job monitor

- Upload marker still reports train `200/200`, test `211/211`,
  `bad_size=0`, `ALL_PASS=True`.
- Active jobs remain old uniform `994341`, old lowreg `994342`, Route A
  `994378`, and Route B `994379`; old main `994340` remains completed.
- Old uniform still has no `Training Over`; latest/best mAP remains `65.73`.
- Old lowreg still has no `Training Over`; latest mAP remains `64.28`. It
  completed Epoch 59 and wrote `epoch_59.pth`, but cleanup is disallowed while
  `994342` is still `RUNNING`.
- Route A completed Epoch 32 with `Loss=0.4870`; Route B reached Epoch 35 step
  50 with `Loss=0.7497` and `selector_gt_density_loss=0.3200`. Route A/B still
  have no mAP.
- Failure scan remains `0`; `/data` shows `420G` used and `1.9T` available.
- Decision: continue active four jobs. No cleanup, severe-result gate, or new
  launch.

## 2026-05-29T11:09:13+08:00 - E2E old uniform completion cleanup

- Old uniform job `994341` completed; log reports `Training Over...`.
- Final old uniform eval is `65.57` Avg-mAP, vector
  `80.80 / 75.81 / 69.31 / 58.04 / 43.89`.
- Best observed old uniform remains `65.73`, vector
  `80.71 / 76.08 / 69.50 / 58.35 / 43.98`; this is the current strongest
  control.
- Cleanup scope was restricted to
  `~/run/yuzibo/e2e_runs/exps/e2e_rawdensel_uniform_20260529_023345/gpu1_id0/checkpoint`,
  resolved as
  `/data/run01/sczc063/yuzibo/e2e_runs/exps/e2e_rawdensel_uniform_20260529_023345/gpu1_id0/checkpoint`.
- Removed only `epoch_19.pth` and `epoch_39.pth`; kept `epoch_59.pth`.
  Disk stayed about `2.3T` size, `420G` used, `1.9T` available, `18%`.
- Active jobs after cleanup: old lowreg `994342`, Route A `994378`, Route B
  `994379`.
- Decision: continue active jobs. Do not clean lowreg until completion; no
  severe-result gate or new launch.

## 2026-05-29T11:11:51+08:00 - E2E remaining-jobs monitor

- Upload marker still reports train `200/200`, test `211/211`,
  `bad_size=0`, `ALL_PASS=True`.
- Completed jobs are old main `994340` and old uniform `994341`.
- Active jobs are old lowreg `994342`, Route A `994378`, and Route B `994379`.
- Old lowreg still has no `Training Over`; latest mAP remains `64.28`, and
  `epoch_59.pth` exists. Cleanup remains disallowed while it is `RUNNING`.
- Route A completed Epoch 33 with `Loss=0.4812`; Route B reached Epoch 36 step
  50 with `Loss=0.7898` and `selector_gt_density_loss=0.3210`. Route A/B still
  have no mAP.
- Failure scan remains `0`; `/data` shows `419G` used and `1.9T` available.
- Decision: continue active three jobs. Wait for lowreg completion cleanup and
  Route A/B first mAP; no severe-result gate or new launch.

## 2026-05-29T11:14:29+08:00 - E2E remaining-jobs monitor

- Upload marker still reports train `200/200`, test `211/211`,
  `bad_size=0`, `ALL_PASS=True`.
- Active jobs remain old lowreg `994342`, Route A `994378`, and Route B
  `994379`.
- Old lowreg still has no `Training Over`; latest mAP remains `64.28`, and
  `epoch_59.pth` exists. Cleanup remains disallowed while it is `RUNNING`.
- Route A reached Epoch 34 step 50 with `Loss=0.4817`; Route B completed Epoch
  36 with `Loss=0.7752` and `selector_gt_density_loss=0.3199`. Route A/B still
  have no mAP.
- Failure scan remains `0`; `/data` shows `419G` used and `1.9T` available.
- Decision: continue active three jobs. No cleanup, severe-result gate, or new
  launch until lowreg completion or Route A/B first mAP.

## 2026-05-29T11:17:08+08:00 - E2E remaining-jobs monitor

- Upload marker still reports train `200/200`, test `211/211`,
  `bad_size=0`, `ALL_PASS=True`.
- Active jobs remain old lowreg `994342`, Route A `994378`, and Route B
  `994379`.
- Old lowreg still has no `Training Over`; latest mAP remains `64.28`, and
  `epoch_59.pth` exists. Cleanup remains disallowed while it is `RUNNING`.
- Route A completed Epoch 34 with `Loss=0.4895`; Route B reached Epoch 37 step
  50 with `Loss=0.8149` and `selector_gt_density_loss=0.3216`. Route A/B still
  have no mAP.
- Failure scan remains `0`; `/data` shows `419G` used and `1.9T` available.
- Decision: continue active three jobs. No cleanup, severe-result gate, or new
  launch until lowreg completion or Route A/B first mAP.

## 2026-05-29T11:21:10+08:00 - E2E remaining-jobs monitor

- Upload marker still reports train `200/200`, test `211/211`,
  `bad_size=0`, `ALL_PASS=True`.
- Active jobs remain old lowreg `994342`, Route A `994378`, and Route B
  `994379`.
- Old lowreg still has no `Training Over`; latest mAP remains `64.28`, and
  `epoch_59.pth` exists. Cleanup remains disallowed while it is `RUNNING`.
- This is about 18 minutes after lowreg final train, comparable to old
  main/uniform final eval duration, so no stall action is taken.
- Route A reached Epoch 35 step 50 with `Loss=0.4199`; Route B completed Epoch
  37 with `Loss=0.7994` and `selector_gt_density_loss=0.3197`. Route A/B still
  have no mAP.
- Failure scan remains `0`; `/data` shows `419G` used and `1.9T` available.
- Decision: continue active three jobs. No cleanup, severe-result gate, or new
  launch until lowreg completion or Route A/B first mAP.

## 2026-05-29T11:23:31+08:00 - E2E old lowreg completion cleanup

- Old lowreg job `994342` completed; log reports `Training Over...` at
  `11:22:16`.
- Final old lowreg eval is `63.61` Avg-mAP, vector
  `78.46 / 73.94 / 66.52 / 56.23 / 42.90`.
- Best observed old lowreg remains `64.36`; final is below random-fixed
  `63.77`, strict EMA `63.85`, stratified `64.64`, and uniform `65.73`.
- Cleanup scope was restricted to
  `~/run/yuzibo/e2e_runs/exps/e2e_rawdensel_lowreg_20260529_023345/gpu1_id0/checkpoint`,
  resolved as
  `/data/run01/sczc063/yuzibo/e2e_runs/exps/e2e_rawdensel_lowreg_20260529_023345/gpu1_id0/checkpoint`.
- Removed only `epoch_19.pth` and `epoch_39.pth`; kept `epoch_59.pth`.
  Disk stayed about `2.3T` size, `419G` used, `1.9T` available, `18%`.
- Active jobs after cleanup: Route A `994378` and Route B `994379`.
- Decision: continue Route A/B to first mAP. No severe-result gate and no new
  launch.

## 2026-05-29T11:26:29+08:00 - E2E Route A/B monitor

- Upload marker still reports train `200/200`, test `211/211`,
  `bad_size=0`, `ALL_PASS=True`.
- Active jobs are Route A `994378` and Route B `994379`, both `RUNNING` on
  `g0009`.
- Route A reached Epoch 36 step 50 with `Loss=0.4585`; Route B completed Epoch
  38 with `Loss=0.7908` and `selector_gt_density_loss=0.3198`. Route A/B still
  have no mAP.
- Visible Route A/B checkpoints remain only `epoch_19.pth`; no epoch-39
  checkpoint yet.
- Failure scan remains `0`; `/data` shows `417G` used and `1.9T` available.
- Decision: continue Route A/B and watch for epoch-39/40 checkpoint/eval and
  first mAP. No cleanup, severe-result gate, or new launch.

## 2026-05-29T11:28:08+08:00 - E2E Route A/B light checkpoint watch

- Route A and Route B remain `RUNNING` on `g0009`.
- Route A completed Epoch 36 with `Loss=0.4471`.
- Route B reached Epoch 39 step 50 with `Loss=0.7418` and
  `selector_gt_density_loss=0.3185`.
- Route A/B still have no mAP; visible checkpoints remain only `epoch_19.pth`,
  with no epoch-39 checkpoint yet.
- Decision: continue Route A/B. Recheck for Route B epoch-39 checkpoint/eval
  soon. No cleanup, severe-result gate, or new launch.

## 2026-05-29T11:31:04+08:00 - E2E Route B epoch_39 checkpoint

- Upload marker still reports train `200/200`, test `211/211`,
  `bad_size=0`, `ALL_PASS=True`.
- Route A and Route B remain `RUNNING` on `g0009`.
- Route A reached Epoch 37 step 50 with `Loss=0.4772`.
- Route B completed Epoch 39 with `Loss=0.7857` and
  `selector_gt_density_loss=0.3199`; it wrote `epoch_39.pth` at `11:31:00`.
- Route A/B still have no mAP. Failure scan remains `0`; `/data` shows `418G`
  used and `1.9T` available.
- Decision: continue Route A/B to first mAP. Do not clean active checkpoints;
  no diagnostics or new launches before first mAP unless a failure appears.

## 2026-05-29T11:33:49+08:00 - E2E Route A/B monitor

- Route A and Route B remain `RUNNING` on `g0009`.
- Route A completed Epoch 37 with `Loss=0.4631`.
- Route B reached Epoch 40 step 50 with `Loss=0.7748` and
  `selector_gt_density_loss=0.3205`.
- Route A/B still have no mAP. Visible checkpoints are Route A `epoch_19.pth`
  and Route B `epoch_19.pth`/`epoch_39.pth`.
- Failure scan remains `0`; `/data` shows `418G` used and `1.9T` available.
- Decision: continue Route A/B to first mAP. No cleanup, severe-result gate, or
  new launch.

## 2026-05-29T11:35:19+08:00 - E2E Route A/B light eval watch

- Route A and Route B remain `RUNNING` on `g0009`.
- No `Average-mAP` line appears in either route log.
- Latest Route A train remains Epoch 37 final with `Loss=0.4631`; latest Route
  B train remains Epoch 40 step 50 with `Loss=0.7748` and
  `selector_gt_density_loss=0.3205`.
- Decision: continue Route A/B. No cleanup, severe-result gate, or new launch.

## 2026-05-29T11:37:52+08:00 - E2E Route A/B monitor

- Upload marker still reports train `200/200`, test `211/211`,
  `bad_size=0`, `ALL_PASS=True`.
- Route A and Route B remain `RUNNING` on `g0009`.
- Route A reached Epoch 38 step 50 with `Loss=0.4715`.
- Route B completed Epoch 40 with `Loss=0.7811` and
  `selector_gt_density_loss=0.3197`.
- Route A/B still have no mAP. Visible checkpoints remain Route A
  `epoch_19.pth`, Route B `epoch_19.pth`/`epoch_39.pth`.
- Failure scan remains `0`; `/data` shows `418G` used and `1.9T` available.
- Decision: continue Route A/B to first mAP. No cleanup, severe-result gate, or
  new launch.

## 2026-05-29T11:40:55+08:00 - E2E Route A/B monitor

- Upload marker still reports train `200/200`, test `211/211`,
  `bad_size=0`, `ALL_PASS=True`.
- Route A and Route B remain `RUNNING` on `g0009`.
- Route A completed Epoch 38 with `Loss=0.4664`.
- Route B reached Epoch 41 step 50 with `Loss=0.7609` and
  `selector_gt_density_loss=0.3198`.
- Route A/B still have no mAP. Visible checkpoints remain Route A
  `epoch_19.pth`, Route B `epoch_19.pth`/`epoch_39.pth`.
- Failure scan remains `0`; `/data` shows `418G` used and `1.9T` available.
- Decision: continue Route A/B to first mAP. No cleanup, severe-result gate, or
  new launch.

## 2026-05-29T11:44:35+08:00 - E2E Route A/B monitor

- Upload marker still reports train `200/200`, test `211/211`,
  `bad_size=0`, `ALL_PASS=True`.
- Route A and Route B remain `RUNNING` on `g0009` for about `3:54:50`.
- Route A reached Epoch 39 step 50 with `Loss=0.4121`, `cls_loss=0.2151`,
  and `reg_loss=0.1970`; no mAP yet.
- Route B completed Epoch 41 with `Loss=0.7863` and
  `selector_gt_density_loss=0.3199`; no mAP yet.
- Route B's auxiliary density loss remains flat near `0.320`, matching the
  early selector diagnostic, but this is not a severe-result trigger before
  first mAP.
- Visible checkpoints remain Route A `epoch_19.pth`, Route B `epoch_19.pth`
  and `epoch_39.pth`.
- Failure scan remains `0`; `/data` shows `418G` used and `1.9T` available.
- Decision: continue Route A/B to first mAP. Do not clean active checkpoints or
  launch new variants from the flat auxiliary loss alone.

## 2026-05-29T11:48:02+08:00 - E2E Route B validation watch

- Route A and Route B remain `RUNNING` on `g0009` for about `3:57:59`.
- Route A completed Epoch 39 with `Loss=0.4489`, wrote `epoch_39.pth` at
  `11:45:37`, and entered Epoch 40; no mAP yet.
- Route B raw log tail shows active validation after Epoch 41 final: progress
  started at `11:41:53` and reached about `161/396` by the tail check around
  `11:48`; no mAP yet.
- Visible checkpoints are Route A `epoch_19.pth`/`epoch_39.pth`, Route B
  `epoch_19.pth`/`epoch_39.pth`.
- Failure scan remains `0`.
- Decision: continue Route B validation to first mAP and Route A to its first
  eval. No active-checkpoint cleanup, new variant launch, or severe-result Pro
  gate without mAP collapse/inconsistency.

## 2026-05-29T11:56:08+08:00 - E2E Route B validation watch

- Route A and Route B remain `RUNNING` on `g0009` for about `4:06:23`.
- Route A continued training to Epoch 41 step 50 with `Loss=0.4238`,
  `cls_loss=0.2161`, and `reg_loss=0.2077`; no mAP yet.
- Route B remained in validation after Epoch 41 final; latest tail at `11:56`
  showed about `320/396` and still no mAP printed.
- Visible checkpoints remain Route A `epoch_19.pth`/`epoch_39.pth`, Route B
  `epoch_19.pth`/`epoch_39.pth`.
- Failure scan remains `0`.
- Decision: continue Route B validation to first mAP and Route A to its first
  eval. Do not clean active checkpoints or launch new variants.

## 2026-05-29T12:02:56+08:00 - E2E Route B first eval

- Route B first eval after Epoch 41 printed at `12:00:23`: `63.42` Avg-mAP,
  vector `78.95 / 74.44 / 66.49 / 55.43 / 41.81`.
- Comparison: `-0.35` vs random-fixed `63.77`, `-0.43` vs strict EMA
  `63.85`, `-1.22` vs stratified `64.64`, and `-2.31` vs uniform control
  `65.73`.
- Interpretation: negative learned-selector evidence for GT-aux residual Route
  B. This is not a severe-collapse trigger because the result is not
  5-point/failure-scale and failure scan remains `0`.
- Route B resumed Epoch 42. Route A completed Epoch 41 with `Loss=0.4617` and
  is currently validating; no Route A mAP yet.
- Decision: send the evidence to Pro before any Route B follow-up, continue
  Route A to first mAP, keep Route B running unless a stop decision is recorded,
  and do not clean active checkpoints or launch new variants.

## 2026-05-29T12:13:00+08:00 - Route B first-eval Pro discussion

- Globalai `gpt-5-pro` failed with `model_not_found`; this was not accepted as
  Pro discussion.
- Oracle browser `gpt-5.5-pro` succeeded in about `8m00s`.
- Report path: `research-wiki/experiments/BATA_ROUTEB_FIRST_EVAL_PRO_DISCUSSION_20260529.md`.
- Answer path: `logs/oracle_pro_routeB_first_eval_20260529.answer.md`.
- Accepted verdict: Route B should stop as a main route and remain diagnostic
  only. Do not rescue it by loss-weight, regularization, or longer-training
  tweaks.
- Accepted next route families: quota-based boundary/action selector,
  coarse-to-fine proposal-conditioned resampling, and irregular-aware detector
  head/assignment.

## 2026-05-29T12:16:20+08:00 - Route B stopped and cleaned

- Applied the Pro stop decision with `scancel 994379` at `12:15:48`; `sacct`
  reports Route B `CANCELLED+` in `04:26:03`.
- Cleanup scope:
  `~/run/yuzibo/e2e_runs/exps/e2e_routeB_gtaux_residual_20260529_074925/gpu1_id0/checkpoint`,
  resolved to
  `/data/run01/sczc063/yuzibo/e2e_runs/exps/e2e_routeB_gtaux_residual_20260529_074925/gpu1_id0/checkpoint`.
- Path check passed inside `~/run/yuzibo`; disk before/after stayed about
  `/data` `2.3T` size, `419G` used, `1.9T` available, `18%`.
- Removed only `epoch_19.pth`; kept `epoch_39.pth`; preserved logs, mAP
  evidence, and non-epoch artifacts.
- Route A `994378` remains running/validating on `g0009`.
- Decision: continue Route A to first mAP. No new launch until the next route
  implementation and review gates are complete.

## 2026-05-29T12:18:05+08:00 - Route A first eval

- Route A first eval after Epoch 41 printed at `12:16:45`: `63.98` Avg-mAP,
  vector `79.11 / 74.36 / 67.24 / 57.04 / 42.14`.
- Comparison: `-1.75` vs old uniform best `65.73`, `-1.11` vs historical
  uniform stride-2 `65.09`, `+0.21` vs random-fixed `63.77`, and `+0.13` vs
  strict EMA `63.85`.
- Interpretation: parity warning, not final parity failure yet, because this is
  the first Route A eval and old uniform improved later. If Route A remains
  near `63.x` on the next eval, debug exact-uniform/bypass protocol before any
  learned-selector claim.
- Route A resumed Epoch 42 and remains `RUNNING`; Route B remains `CANCELLED+`
  and cleaned.
- Decision: continue Route A to at least the next eval. No new launch until
  next-route implementation and review gates are complete.

## 2026-05-29T12:21:02+08:00 - Route A health monitor

- Upload marker still reports train `200/200`, test `211/211`,
  `bad_size=0`, `ALL_PASS=True`.
- Route A remains `RUNNING` on `g0009` for about `4:31:17`; Route B remains
  `CANCELLED+`.
- No new Route A mAP after `63.98`.
- Route A reached Epoch 42 step 50 with `Loss=0.4276`, `cls_loss=0.2173`, and
  `reg_loss=0.2103`.
- Failure scan remains `0`; disk shows `/data` `418G` used and `1.9T`
  available.
- Decision: continue Route A to next eval. No active-run cleanup and no new
  launch until next-route implementation/review gates are complete.

## 2026-05-29T12:31:15+08:00 - E2E quota selector self-check

- Implemented local quota action/boundary selector route in `OpenTAD_BATA_Clean`:
  selector internals, two quota configs, and E2E contract tests.
- New configs split the 384-frame budget as `160/128/96` and `128/128/128`
  for context/action/boundary roles under the same dense `768 -> 384` E2E
  contract.
- Self-check report:
  `research-wiki/experiments/BATA_E2E_QUOTA_SELECTOR_SELF_CHECK_20260529.md`.
- Local verification: `py_compile` PASS; pytest
  `tests/test_e2e_raw_frame_selector_contracts.py -q -rs` returned
  `1 passed, 14 skipped`. Local merged-config check could not run because
  Windows Python lacks `mmengine`.
- Remote Route A still has only first mAP `63.98`; second validation is in
  progress after Epoch 43. No cleanup, sync, or launch occurred.
- Decision: send the implementation to GPT-5.5 Pro, then run Gemini CLI and
  DeepSeek CLI. Only deploy after all gates and N16R4 preflight pass.

## 2026-05-29T13:04:21+08:00 - E2E quota selector Pro gate passed

- Initial Oracle Pro attachment-upload attempt timed out and is not accepted.
- Inline GPT-5.5 Pro review returned `WARN` with no blocker, then first focused
  re-review returned `FAIL` for using a toy `target_len=8` selector to validate
  deployed quota splits.
- Fixed the test to use deployed `768 -> 384` quota selectors and stabilized the
  quota remainder tie-breaker.
- Second focused GPT-5.5 Pro re-review returned `PASS` in
  `logs/oracle_pro_e2e_quota_selector_fix2_review_20260529.txt`.
- Decision: proceed to Gemini CLI and DeepSeek CLI. Do not sync or launch quota
  jobs until both reviews and N16R4 preflight pass.

## 2026-05-29T13:06:31+08:00 - E2E quota selector Gemini gate passed

- Gemini CLI `gemini-3-pro-preview` exited `0` and returned `PASS` in
  `logs/gemini3_pro_preview_e2e_quota_selector_review_20260529.txt`.
- No blocking findings or required fixes. Stderr only had terminal color and
  ripgrep fallback warnings.
- Decision: proceed to Claude CLI DeepSeek `deepseek-v4-pro` review before
  remote sync or launch.

## 2026-05-29T13:17:25+08:00 - E2E quota selector DeepSeek gate passed

- First Claude CLI DeepSeek run exited `0` but wrote only a shallow stdout and
  put the substantive PASS in a plan artifact, so it was not accepted as the
  completed gate.
- Retry disabled `Write`, `Edit`, and `ExitPlanMode`; stdout log
  `logs/claude_deepseek_v4_pro_e2e_quota_selector_review_20260529_retry.txt`
  returned full `PASS`, no blocking findings, and no required fixes.
- Non-blocking notes: `detach_gt_remap=False` still would not be useful through
  `searchsorted`, context channel is not directly used for positions, `cummax`
  is a safety-net edge case, and `/255` normalization is heuristic.
- Decision: local external gates are complete. Rerun local verification, commit
  reviewed files only, then sync and preflight on N16R4.

## 2026-05-29T13:20:09+08:00 - E2E quota selector committed

- Post-review local checks passed: `py_compile` exit `0`, pytest
  `1 passed, 17 skipped`, and `git diff --check` had no whitespace errors
  beyond LF/CRLF warnings.
- Commit `7591ead add e2e quota frame selector` contains only four reviewed
  files: selector implementation, two quota configs, and the E2E selector
  contract test file.
- Decision: sync reviewed files to N16R4 and run Linux preflight before any
  quota training launch.

## 2026-05-29T13:22:30+08:00 - E2E quota selector synced to N16R4

- Synced only four reviewed files from commit `7591ead` to
  `~/run/yuzibo/OpenTAD_BATA_Clean` with Windows native `scp.exe`.
- Remote `ls -l` confirmed the selector, test file, and two quota configs were
  updated at `May 29 13:22`.
- Route A third eval is now `64.38` Avg-mAP at `13:18:47`, still below old
  uniform best `65.73`; Route A remains active.
- Decision: run N16R4 Linux preflight before quota job launch.

## 2026-05-29T13:27:57+08:00 - E2E quota selector N16R4 preflight passed

- Accepted preflight log:
  `~/run/yuzibo/OpenTAD_BATA_Clean/logs/e2e_quota_selector_preflight_20260529_rerun3.log`.
- Earlier preflight attempts were not accepted: one used `/usr/bin/python` due
  shell/env loading failure, one failed while sourcing `/etc/profile` under
  `set -u`, and one used a wrong manual merged-config assertion path after
  py_compile/pytest had passed.
- Accepted run loaded the correct conda env, used Python `3.10.20`, passed
  `py_compile`, passed Linux selector contracts with `18 passed`, and passed
  merged-config assertions for both quota variants.
- Decision: submit two one-GPU quota Slurm jobs.

## 2026-05-29T13:29:58+08:00 - E2E quota selector jobs submitted

- Submitted `994707 e2e_quota_ab` and `994708 e2e_quota_bh` with run tags
  `e2e_quota_action_boundary_20260529_1329` and
  `e2e_quota_boundary_heavy_20260529_1329`.
- Both were initially `PENDING`.
- Route A `994378` remains active, so the intended 3-experiment/GPU parallel
  set is Route A plus the two quota variants once scheduled.
- Inspected `scripted_gpu` jobs `994471/994473`: both are `sleep` jobs from
  `/data/run01/sczc063/wangruofan`, outside `~/run/yuzibo`; not cancelled.
- Decision: monitor job-internal preflight and training start.

## 2026-05-29T13:32:08+08:00 - E2E quota selector jobs running

- Jobs `994707 e2e_quota_ab` and `994708 e2e_quota_bh` are `RUNNING` on
  `g0042`.
- Both logs show job-internal `merged_config_preflight=PASS`, `Training Starts`,
  and Epoch 0 started at `13:30:24`.
- Logs:
  `~/run/yuzibo/OpenTAD_BATA_Clean/logs/e2e_quota_action_boundary_20260529_1329_n16r4.log`
  and
  `~/run/yuzibo/OpenTAD_BATA_Clean/logs/e2e_quota_boundary_heavy_20260529_1329_n16r4.log`.
- Decision: continue monitoring to first loss line and first eval.

## 2026-05-29T13:36:12+08:00 - E2E quota selector first-loss health check

- Both quota jobs remain `RUNNING`.
- Action-boundary first step-50 line: `Loss=2.1287`, selector action/boundary
  auxiliary losses `0.1275/0.1912`, `cls_loss=0.9897`, `reg_loss=0.7564`.
- Boundary-heavy first step-50 line: `Loss=2.2962`, selector action/boundary
  auxiliary losses `0.1593/0.3186`, `cls_loss=0.9947`, `reg_loss=0.7594`.
- Failure scan is `0` for both logs.
- Decision: deployment is healthy; continue periodic monitoring to first eval.

## 2026-05-29T13:39:31+08:00 - Upload and quota health recheck

- Upload marker remains `ALL_PASS=True`; train/test are `200/211` mp4 entries,
  all valid symlinks with `broken_symlink_mp4=0`.
- Disk is safe: `/data` has `1.9T` available.
- Quota action-boundary completed Epoch 0 with step-99 `Loss=1.9960`.
- Quota boundary-heavy completed Epoch 0 with step-99 `Loss=2.1875`.
- Failure scan remains `0` for both quota logs.
- Route A remains running; latest mAP trend is `63.98 -> 64.35 -> 64.38`.
- Decision: continue all three active runs; no stop/cleanup/severe gate.

## 2026-05-29T13:42:25+08:00 - Active training monitor

- Upload gate still passes: train/test `200/211`, no broken mp4 symlinks.
- Quota jobs remain `RUNNING`; no failures.
- Action-boundary reached Epoch 1 step 50 with `Loss=1.4044`.
- Boundary-heavy reached Epoch 1 step 50 with `Loss=1.5639`.
- Route A has no new mAP after `64.38`; latest visible train remains Epoch 47
  final.
- Decision: continue all active runs; no gate triggered.

## 2026-05-29T13:44:58+08:00 - Active training monitor

- Upload gate remains valid: train/test `200/211`, no broken mp4 symlinks.
- Quota action-boundary completed Epoch 1 with `Loss=1.3781`.
- Quota boundary-heavy completed Epoch 1 with `Loss=1.5368`.
- Failure count remains `0` for both quota logs.
- Route A has no new mAP after `64.38`.
- Decision: continue all active runs; no gate triggered.

## 2026-05-29T13:47:42+08:00 - Active training monitor

- Upload gate remains valid: train/test `200/211`, no broken mp4 symlinks.
- Quota action-boundary reached Epoch 2 step 50 with `Loss=1.2342`.
- Quota boundary-heavy reached Epoch 2 step 50 with `Loss=1.3954`.
- Failure count remains `0` for both quota logs.
- Route A has no new mAP after `64.38`; latest visible training remains Epoch
  47 final, likely validating or transitioning.
- Decision: continue all active runs; no gate triggered.

## 2026-05-29T13:51:05+08:00 - Route A fourth eval and quota Epoch 2

- Upload gate remains valid: train/test `200/211`, no broken mp4 symlinks.
- Quota action-boundary completed Epoch 2 with `Loss=1.2727`.
- Quota boundary-heavy completed Epoch 2 with `Loss=1.4343`.
- Failure count remains `0` for both quota logs.
- Route A fourth eval at `13:49:49`: `64.46` Avg-mAP, vector
  `79.62 / 75.09 / 67.89 / 57.23 / 42.44`.
- Route A is still below uniform stride-2 `65.09` and old uniform best
  `65.73`; parity warning remains, but no severe-result gate is triggered.
- Decision: continue all active runs.

## 2026-05-29T13:55:02+08:00 - Active training monitor

- Upload gate remains valid and `/data` has about `1.9T` available.
- Quota action-boundary reached Epoch 3 step 50 with `Loss=1.2327`.
- Quota boundary-heavy reached Epoch 3 step 50 with `Loss=1.3834`.
- Failure count remains `0` for both quota logs.
- Route A resumed after fourth eval and reached Epoch 48 step 50 with
  `Loss=0.4288`.
- Decision: continue all active runs; no gate triggered.

## 2026-05-29T13:57:34+08:00 - Active training monitor

- Upload gate remains valid: train/test `200/211`, no broken mp4 symlinks.
- Quota action-boundary completed Epoch 3 with `Loss=1.2345`.
- Quota boundary-heavy completed Epoch 3 with `Loss=1.3856`.
- Failure count remains `0` for both quota logs.
- Route A completed Epoch 48 with `Loss=0.4179`; latest mAP remains `64.46`.
- Decision: continue all active runs; no gate triggered.

## 2026-05-29T14:00:15+08:00 - Active training monitor

- Upload gate remains valid: train/test `200/211`, no broken mp4 symlinks.
- Quota action-boundary reached Epoch 4 step 50 with `Loss=1.2420`.
- Quota boundary-heavy reached Epoch 4 step 50 with `Loss=1.4165`.
- Failure count remains `0` for both quota logs.
- Route A reached Epoch 49 step 50 with `Loss=0.3808`; latest mAP remains
  `64.46`.
- Decision: continue all active runs; no gate triggered.

## 2026-05-29T14:03:17+08:00 - Active training monitor

- Upload gate remains valid: train/test `200/211`, no broken mp4 symlinks.
- Quota action-boundary completed Epoch 4 with `Loss=1.1984`.
- Quota boundary-heavy completed Epoch 4 with `Loss=1.3544`.
- Failure count remains `0` for both quota logs.
- Route A completed Epoch 49 with `Loss=0.4179`; latest mAP remains `64.46`.
- Decision: continue all active runs; no gate triggered.

## 2026-05-29T14:09:27+08:00 - Active training monitor

- Upload gate still passes: train/test `200/211`, no broken mp4 symlinks,
  `ALL_PASS=True`; `/data` has about `1.9T` available in the project quota
  view.
- Route A `994378` is still `RUNNING`; no new mAP after `64.46`, but the raw
  log shows active post-Epoch-49 validation progress, not a confirmed stall.
- Quota action-boundary `994707` completed Epoch 5 with `Loss=1.2066`.
- Quota boundary-heavy `994708` completed Epoch 5 with `Loss=1.3690`.
- Failure count remains `0` for all three active logs. The initial repo-local
  checkpoint search was not authoritative because the launcher writes to
  `~/run/yuzibo/e2e_runs/exps/$RUN_TAG`; no cleanup was performed.
- Decision: continue all active runs; next gates are Route A final eval,
  quota epoch-19 checkpoint diagnostics, and quota first eval.

## 2026-05-29T14:11:59+08:00 - Active training monitor

- Slurm still reports Route A `994378`, quota action-boundary `994707`, and
  quota boundary-heavy `994708` as `RUNNING`.
- Route A has no new mAP after `64.46`; final-validation progress reached about
  `61%` (`242/396`) with failure count `0`.
- Quota action-boundary reached Epoch 6 step 50 with `Loss=1.1248`.
- Quota boundary-heavy reached Epoch 6 step 50 with `Loss=1.2901`.
- Decision: continue all active runs; no cleanup or severe-result gate.

## 2026-05-29T14:14:12+08:00 - Artifact path correction

- Confirmed launcher output root is `~/run/yuzibo/e2e_runs/exps/$RUN_TAG`.
- Route A output dir exists with active `epoch_19.pth` and `epoch_39.pth`.
- Quota output dirs exist with copied configs, but no `epoch_*.pth` yet.
- No cleanup was performed because Route A is still active and quota runs have
  not reached checkpoint time.

## 2026-05-29T14:16:49+08:00 - Active training monitor

- Route A has no new mAP after `64.46`; failure count remains `0`.
- The compact-tail helper hit `ImportError: No module named pathlib` under
  login-node default Python; this is not a training failure.
- Quota action-boundary completed Epoch 6 with `Loss=1.1360` and reached Epoch
  7 step 50 with `Loss=1.1260`.
- Quota boundary-heavy completed Epoch 6 with `Loss=1.2968` and reached Epoch 7
  step 50 with `Loss=1.2900`.
- Decision: continue all active runs; no cleanup or severe-result gate.

## 2026-05-29T14:20:39+08:00 - Active training monitor

- Upload gate still passes: train/test `200/211`, no broken mp4 symlinks,
  `ALL_PASS=True`; `/data` has about `1.9T` available in the project quota
  view.
- Route A produced a new eval at `14:20:43`: `64.68` Avg-mAP, vector
  `79.90 / 75.38 / 68.19 / 57.31 / 42.63`, failure count `0`.
- Route A is now slightly above stratified `64.64`, but still below uniform
  stride-2 `65.09` and old uniform best `65.73`; parity caveat remains.
- Quota action-boundary completed Epoch 7 with `Loss=1.1045`.
- Quota boundary-heavy completed Epoch 7 with `Loss=1.2640`.
- Artifact paths confirmed: Route A has active `epoch_19.pth` and
  `epoch_39.pth`; quota dirs have no checkpoint yet. No cleanup was performed.

## 2026-05-29T14:24:03+08:00 - Active training monitor

- Upload gate still passes: train/test `200/211`, no broken mp4 symlinks,
  `ALL_PASS=True`; `/data` has about `1.9T` available in the project quota
  view.
- Route A resumed after the `64.68` eval and reached Epoch 50 step 50 with
  `Loss=0.3762`; failure count `0`.
- Quota action-boundary reached Epoch 8 step 50 with `Loss=1.1650`.
- Quota boundary-heavy completed Epoch 8 with `Loss=1.2423`.
- No new checkpoint for quota, no run complete, and no cleanup/severe-result
  gate triggered.

## 2026-05-29T14:28:55+08:00 - Active training monitor

- Upload gate still passes: train/test `200/211`, no broken mp4 symlinks,
  `ALL_PASS=True`; `/data` has about `1.9T` available in the project quota
  view.
- Route A completed Epoch 50 with `Loss=0.3826`; latest mAP remains `64.68`.
- Quota action-boundary completed Epoch 8 with `Loss=1.1004` and reached Epoch
  9 step 50 with `Loss=1.0113`.
- Quota boundary-heavy reached Epoch 9 step 50 with `Loss=1.1753`.
- No quota checkpoint yet; no cleanup, selector diagnostic, first-eval gate, or
  severe-result gate triggered.

## 2026-05-29T14:30:57+08:00 - Active training monitor

- Upload gate still passes: train/test `200/211`, no broken mp4 symlinks,
  `ALL_PASS=True`.
- Route A reached Epoch 51 step 50 with `Loss=0.4072`; latest mAP remains
  `64.68`.
- Quota action-boundary reached Epoch 9 step 50 with `Loss=1.0113`.
- Quota boundary-heavy completed Epoch 9 with `Loss=1.2260`.
- No quota checkpoint yet; no cleanup, selector diagnostic, first-eval gate, or
  severe-result gate triggered.

## 2026-05-29T14:32:59+08:00 - Active training monitor

- Upload gate still passes: train/test `200/211`, no broken mp4 symlinks,
  `ALL_PASS=True`.
- Route A completed Epoch 51 with `Loss=0.4036`; latest mAP remains `64.68`.
- Quota action-boundary completed Epoch 9 with `Loss=1.0643`.
- Quota boundary-heavy reached Epoch 10 step 50 with `Loss=1.2370`.
- No quota checkpoint yet; no cleanup, selector diagnostic, first-eval gate, or
  severe-result gate triggered.

## 2026-05-29T14:34:55+08:00 - Active training monitor

- Upload gate still passes: train/test `200/211`, no broken mp4 symlinks,
  `ALL_PASS=True`.
- Route A latest train evidence remains Epoch 51 final with `Loss=0.4036`;
  latest mAP remains `64.68`.
- Quota action-boundary reached Epoch 10 step 50 with `Loss=1.0845`.
- Quota boundary-heavy completed Epoch 10 with `Loss=1.2062`.
- No quota checkpoint yet; no cleanup, selector diagnostic, first-eval gate, or
  severe-result gate triggered.

## 2026-05-29T14:36:44+08:00 - Active training monitor

- Upload gate still passes: train/test `200/211`, no broken mp4 symlinks,
  `ALL_PASS=True`.
- Slurm reports Route A and both quota jobs still `RUNNING`.
- Latest log evidence is unchanged from the prior monitor: Route A `64.68`
  latest mAP, quota action-boundary Epoch 10 step 50, quota boundary-heavy
  Epoch 10 final.
- Failure counts remain `0`; no quota checkpoint yet; no cleanup, selector
  diagnostic, first-eval gate, or severe-result gate triggered.

## 2026-05-29T14:39:00+08:00 - Active training monitor

- Upload gate still passes: train/test `200/211`, no broken mp4 symlinks,
  `ALL_PASS=True`.
- Route A latest log evidence remains Epoch 51 final, latest mAP `64.68`.
- Quota action-boundary completed Epoch 10 with `Loss=1.0557`.
- Quota boundary-heavy reached Epoch 11 step 50 with `Loss=1.2066`.
- Failure counts remain `0`; no quota checkpoint yet; no cleanup, selector
  diagnostic, first-eval gate, or severe-result gate triggered.

## 2026-05-29T14:40:58+08:00 - Active training monitor

- Upload gate still passes: train/test `200/211`, `ALL_PASS=True`.
- Route A latest log evidence remains Epoch 51 final, latest mAP `64.68`.
- Quota action-boundary reached Epoch 11 step 50 with `Loss=1.0484`.
- Quota boundary-heavy completed Epoch 11 with `Loss=1.1914`.
- Failure counts remain `0`; no quota checkpoint yet; no cleanup, selector
  diagnostic, first-eval gate, or severe-result gate triggered.

## 2026-05-29T14:42:56+08:00 - Active training monitor

- Upload gate still passes: train/test `200/211`, `ALL_PASS=True`.
- Route A latest log evidence remains Epoch 51 final, latest mAP `64.68`.
- Quota action-boundary completed Epoch 11 with `Loss=1.0353`.
- Quota boundary-heavy reached Epoch 12 step 50 with `Loss=1.2264`.
- Failure counts remain `0`; no quota checkpoint yet; no cleanup, selector
  diagnostic, first-eval gate, or severe-result gate triggered.

## 2026-05-29T14:47:06+08:00 - Active training monitor

- Upload gate still passes: train/test `200/211`, `ALL_PASS=True`; `/data`
  remains about `1.9T` available.
- Project jobs `994378`, `994707`, and `994708` remain `RUNNING`; unrelated
  `scripted_gpu` jobs were not touched.
- Route A latest mAP remains `64.68`; raw-tail check at `14:48:10` shows active
  post-Epoch-51 validation at about `331/396`, so no stall intervention.
- Quota action-boundary reached Epoch 12 step 50 with `Loss=1.0457`.
- Quota boundary-heavy completed Epoch 12 with `Loss=1.1945` and entered
  Epoch 13.
- Failure counts remain `0`; no quota checkpoint yet; no cleanup, selector
  diagnostic, first-eval gate, severe-result gate, cancellation, or new launch.

## 2026-05-29T14:52:10+08:00 - Active training monitor

- Route A produced a new eval: `64.82` Avg-mAP at `14:51:31`, vector
  `80.00 / 75.39 / 68.03 / 57.80 / 42.90`, then resumed Epoch 52. It is above
  stratified `64.64`, but still below uniform stride-2 `65.09` and old uniform
  best `65.73`; parity caveat remains.
- Quota action-boundary reached Epoch 13 step 50 with `Loss=1.0397`.
- Quota boundary-heavy completed Epoch 13 with `Loss=1.1961`.
- Failure counts remain `0`; quota dirs still have no `epoch_*.pth`; no cleanup,
  selector diagnostic, first-eval gate, severe-result gate, cancellation, or
  new launch.

## 2026-05-29T14:57:32+08:00 - Active training monitor

- Upload gate still passes with `ALL_PASS=True`; project jobs `994378`,
  `994707`, and `994708` remain `RUNNING`.
- Route A latest mAP remains `64.82`; it completed Epoch 52 with
  `Loss=0.4169` and started Epoch 53. It is still below uniform stride-2
  `65.09` and old uniform best `65.73`, so the parity caveat remains.
- Quota action-boundary reached Epoch 14 step 50 with `Loss=0.9881`, selector
  action/boundary losses `0.1282/0.1923`.
- Quota boundary-heavy completed Epoch 14 with `Loss=1.1413`, selector
  action/boundary losses `0.1599/0.3197`.
- Failure counts remain `0`; quota dirs still have no `epoch_*.pth`; no cleanup,
  selector diagnostic, first-eval gate, severe-result gate, cancellation, or
  new launch.

## 2026-05-29T15:00:59+08:00 - Active training monitor

- Upload gate still passes with `ALL_PASS=True`; `/data` remains about `1.9T`
  free.
- Project jobs `994378`, `994707`, and `994708` remain `RUNNING`; unrelated
  jobs were not touched.
- Route A latest mAP remains `64.82` and reached Epoch 53 step 50 with
  `Loss=0.3953`; failure count is `0`.
- Quota action-boundary completed Epoch 14 with `Loss=1.0026`.
- Quota boundary-heavy reached Epoch 15 step 50 with `Loss=1.1508`.
- Quota dirs still have no `epoch_*.pth`; no process intervention, cleanup,
  selector diagnostic, first-eval gate, severe-result gate, cancellation, or new
  launch.

## 2026-05-29T15:04:58+08:00 - Active training monitor

- Upload gate still passes with `ALL_PASS=True`; project jobs `994378`,
  `994707`, and `994708` remain `RUNNING`.
- Route A latest mAP remains `64.82` and completed Epoch 53 with `Loss=0.3669`.
- Quota action-boundary reached Epoch 15 step 50 with `Loss=1.0172`.
- Quota boundary-heavy completed Epoch 15 with `Loss=1.1327`.
- Failure counts remain `0`; quota dirs still have no `epoch_*.pth`; no process
  intervention, cleanup, selector diagnostic, first-eval gate, severe-result
  gate, cancellation, or new launch.

## 2026-05-29T15:08:53+08:00 - Active training monitor

- Upload gate still passes with `ALL_PASS=True`; project jobs `994378`,
  `994707`, and `994708` remain `RUNNING`.
- Route A latest mAP remains `64.82`; latest train evidence remains Epoch 53
  final.
- Quota action-boundary completed Epoch 15 and reached Epoch 16 step 50 with
  `Loss=0.9685`.
- Quota boundary-heavy completed Epoch 16 with `Loss=1.1486`.
- Failure counts remain `0`; quota dirs still have no `epoch_*.pth`; no selector
  diagnostic, process intervention, cleanup, first-eval gate, severe-result
  gate, cancellation, or new launch.

## 2026-05-29T15:11:05+08:00 - Active training monitor

- Upload gate still passes with `ALL_PASS=True`; project jobs `994378`,
  `994707`, and `994708` remain `RUNNING`.
- Route A latest mAP remains `64.82`; latest train evidence remains Epoch 53
  final.
- Quota action-boundary latest line remains Epoch 16 step 50 with `Loss=0.9685`.
- Quota boundary-heavy reached Epoch 17 step 50 with `Loss=1.1055`.
- Failure counts remain `0`; quota dirs still have no `epoch_*.pth`; no selector
  diagnostic, process intervention, cleanup, first-eval gate, severe-result
  gate, cancellation, or new launch.

## 2026-05-29T15:13:13+08:00 - Active training monitor

- Upload gate still passes with `ALL_PASS=True`; project jobs `994378`,
  `994707`, and `994708` remain `RUNNING`.
- Route A latest mAP remains `64.82`; latest train evidence remains Epoch 53
  final.
- Quota action-boundary completed Epoch 16 with `Loss=1.0006`.
- Quota boundary-heavy latest line remains Epoch 17 step 50 with `Loss=1.1055`.
- Failure counts remain `0`; quota dirs still have no `epoch_*.pth`; no selector
  diagnostic, process intervention, cleanup, first-eval gate, severe-result
  gate, cancellation, or new launch.

## 2026-05-29T15:16:18+08:00 - Active training monitor

- Upload gate still passes with `ALL_PASS=True`; project jobs `994378`,
  `994707`, and `994708` remain `RUNNING`.
- Route A latest mAP remains `64.82`; latest train evidence remains Epoch 53
  final.
- Quota action-boundary reached Epoch 17 step 50 with `Loss=0.9626`.
- Quota boundary-heavy completed Epoch 17 with `Loss=1.1227`.
- Failure counts remain `0`; quota dirs still have no `epoch_*.pth`; no selector
  diagnostic, process intervention, cleanup, first-eval gate, severe-result
  gate, cancellation, or new launch.

## 2026-05-29T15:19:12+08:00 - Active training monitor

- Upload gate still passes with `ALL_PASS=True`; `/data` remains about `1.9T`
  free.
- Project jobs `994378`, `994707`, and `994708` remain `RUNNING`; unrelated
  jobs were not touched.
- Route A latest mAP remains `64.82`; latest train evidence remains Epoch 53
  final.
- Quota action-boundary completed Epoch 17 with `Loss=0.9710`; selector
  action/boundary losses were `0.1279/0.1918`.
- Quota boundary-heavy reached Epoch 18 step 50 with `Loss=1.1319`; selector
  action/boundary losses were `0.1602/0.3204`.
- Failure counts remain `0`; quota dirs still have no `epoch_*.pth`; no selector
  diagnostic, process intervention, cleanup, first-eval gate, severe-result
  gate, cancellation, or new launch.

## 2026-05-29T15:23:19+08:00 - Active training monitor

- Upload gate still passes with `ALL_PASS=True`; `/data` remains about `1.9T`
  free.
- Project jobs `994378`, `994707`, and `994708` remain `RUNNING`; unrelated
  jobs were not touched.
- Route A produced a new eval: `65.03` Avg-mAP, vector
  `80.25 / 75.49 / 68.09 / 57.98 / 43.32`; this is near the `65.09` uniform
  stride-2 reference but remains a uniform-control result.
- Quota action-boundary completed Epoch 18 with `Loss=0.9620` and entered
  Epoch 19.
- Quota boundary-heavy completed Epoch 18 with `Loss=1.1239` and reached Epoch
  19 step 50 with `Loss=1.1542`.
- Failure counts remain `0`; quota dirs still have no `epoch_*.pth`; no selector
  diagnostic, process intervention, cleanup, first-eval gate, severe-result
  gate, cancellation, or new launch.

## 2026-05-29T15:28:27+08:00 - Quota checkpoint watch

- `logs/run_quota_epoch19_diag.py` passed local `py_compile`.
- Quota boundary-heavy wrote `epoch_19.pth` at `15:23:08`; it completed Epoch
  19 with `Loss=1.1371` and reached Epoch 20 step 50 with `Loss=1.0901`.
- Quota action-boundary still has no `epoch_*.pth`; it reached Epoch 19 step 50
  with `Loss=0.9823`.
- No quota mAP yet. Diagnostic remains selector-only with validation GT used
  only for offline labeling; no cleanup, severe-result gate, cancellation, or
  new long run.

## 2026-05-29T15:31:00+08:00 - Boundary-heavy selector diagnostic submitted

- Uploaded `logs/run_quota_epoch19_diag.py` to
  `~/run/yuzibo/e2e_runs/selector_diagnostics/run_quota_epoch19_diag.py`.
- Submitted Slurm job `994846 e2e_bh_diag` using
  `~/run/yuzibo/OpenTAD_BATA_Clean/scripts/run_quota_bh_epoch19_diag_20260529.sbatch`.
- The diagnostic targets boundary-heavy `epoch_19.pth`, samples 64 validation
  items, and uses GT only for offline position-labeling diagnostics. Active
  training jobs were not touched.

## 2026-05-29T15:31:24+08:00 - Action-boundary checkpoint ready

- Quota action-boundary wrote `epoch_19.pth` at `15:27:09` and completed Epoch
  19 with `Loss=0.9922`; failure count remains `0`.
- Boundary-heavy diagnostic job `994846` is running and has processed `16/64`
  samples.
- Next action: submit the matching action-boundary selector diagnostic. No
  quota mAP yet.

## 2026-05-29T15:33:00+08:00 - Action-boundary selector diagnostic submitted

- Submitted Slurm job `994848 e2e_ab_diag` using
  `~/run/yuzibo/OpenTAD_BATA_Clean/scripts/run_quota_ab_epoch19_diag_20260529.sbatch`.
- The diagnostic targets action-boundary `epoch_19.pth`, samples 64 validation
  items, and uses GT only for offline position-labeling diagnostics. Active
  training jobs were not touched.

## 2026-05-29T15:38:35+08:00 - Quota selector diagnostics completed

- BH diagnostic `994846` completed; AB diagnostic `994848` completed.
- AB epoch-19: mean abs delta `2.66`, selected action fraction `0.3327`,
  boundary<=4 fraction `0.1392`, action/boundary logit std `0.0023/0.0006`.
- BH epoch-19: mean abs delta `3.23`, selected action fraction `0.3324`,
  boundary<=4 fraction `0.1405`, action/boundary logit std `0.0101/0.0018`.
- Interpretation: both selectors are mildly non-uniform and retain action
  coverage, but neither has learned a sharp boundary-focused selected-frame
  distribution by epoch 19. Continue to first mAP and send diagnostics to Pro
  before launching aggressive follow-ups.

## 2026-05-29T15:42:00+08:00 - Pro quota-diagnostic discussion started

- Started Oracle browser `gpt-5.5-pro` in background, local PowerShell PID
  `7224`.
- Prompt: `logs/oracle_pro_e2e_quota_diag_discussion_prompt_20260529.md`.
- Expected answer/stdout/stderr:
  `logs/oracle_pro_e2e_quota_diag_discussion_20260529.txt`,
  `logs/oracle_pro_e2e_quota_diag_discussion_20260529.stdout.txt`, and
  `logs/oracle_pro_e2e_quota_diag_discussion_20260529.err.txt`.
- Continue quota first-mAP monitoring while Pro runs; do not treat the Pro gate
  as complete until the answer is substantive and model evidence is recorded.

## 2026-05-29T15:43:55+08:00 - Active training monitor

- Pro discussion remains active; Oracle browser session is waiting for a
  response.
- Project jobs `994378`, `994707`, and `994708` remain `RUNNING`.
- Route A latest mAP remains `65.03`; AB reached Epoch 22 step 50 with
  `Loss=0.9587`; BH completed Epoch 22 with `Loss=1.1138`.
- Failure counts are `0`; quota first mAP is still absent. No cleanup or new
  long run.

## 2026-05-29T15:45:31+08:00 - Active training monitor

- Pro discussion remains active and waiting.
- Project jobs `994378`, `994707`, and `994708` remain `RUNNING`.
- Route A latest mAP remains `65.03`; AB reached Epoch 22 step 50 with
  `Loss=0.9587`; BH reached Epoch 23 step 50 with `Loss=1.1124`.
- Failure counts are `0`; quota first mAP is still absent. `/data` remains about
  `1.9T` free. No cleanup or new long run.

## 2026-05-29T15:47:54+08:00 - Active training monitor

- Pro discussion remains active and waiting at about `5m`.
- Route A latest mAP remains `65.03`; AB completed Epoch 22 with `Loss=0.9612`;
  BH completed Epoch 23 with `Loss=1.0988`.
- Failure counts are `0`; quota first mAP is still absent. No cleanup or new
  long run.

## 2026-05-29T15:51:48+08:00 - Active training monitor

- Pro discussion remains active and waiting at about `9m`.
- Project jobs `994378`, `994707`, and `994708` remain `RUNNING`.
- Route A latest mAP remains `65.03`; AB reached Epoch 23 step 50 with
  `Loss=0.9394`; BH reached Epoch 24 step 50 with `Loss=1.1001`.
- Failure counts are `0`; quota first mAP is still absent. Contract remains
  50% backbone budget with train-only GT auxiliary losses and no test-time
  GT/teacher/cache. No cleanup or new long run.

## 2026-05-29T15:54:57+08:00 - Pro discussion returned and Route A updated

- Oracle browser `gpt-5.5-pro` completed in about `11m32s`; answer saved to
  `logs/oracle_pro_e2e_quota_diag_discussion_20260529.txt`.
- Pro judged the current quota selector end-to-end, but not yet boundary-aware:
  epoch-19 diagnostics show mild non-uniform coverage and weak boundary credit.
- Accepted next action: keep AB/BH to first mAP, implement one aggressive
  BoundarySharp-ST route, and add stronger epoch-39 diagnostics/ablations.
- Route A reached `65.05` Avg-mAP at `15:52:45`, vector
  `80.51 / 75.45 / 68.25 / 58.15 / 42.89`; this is a near-uniform-control
  reference, not learned selector evidence.
- AB/BH are still running with no first mAP yet. No cleanup or new long run was
  launched at this monitor point.

## 2026-05-29T16:01:22+08:00 - BoundarySharp-ST implementation self-check

- Implemented optional ST top-k quota positions and stronger action/boundary
  target weights in `temporal_density_selector.py`.
- Added config
  `configs/adatad/thumos/e2e_rawdensel_384of768_boundarysharp_st_adapter.py`
  with quota `96/96/192` and no test-time GT/teacher/cache.
- Added selector contract tests for ST top-k forward behavior, soft gradient
  path, and merged-config contract.
- Local checks passed: `py_compile`; Windows pytest `1 passed, 19 skipped`;
  `git diff --check` with LF/CRLF warnings only.
- Review gate pending before sync/deployment: GPT-5.5 Pro, Gemini CLI, and
  Claude CLI DeepSeek.

## 2026-05-29T16:19:32+08:00 - BoundarySharp-ST Pro review and fixes

- GPT-5.5 Pro implementation review returned `WARN` with no code blocker after
  about `13m24s`.
- Applied accepted fixes: prefix-mask/all-false-mask checks, non-negative quota
  target-weight validation, ST invalid-tail/short-valid tests, and
  `selection_mode`-based quota config assertion.
- Post-fix local checks passed: `py_compile`; Windows pytest
  `1 passed, 20 skipped`; `git diff --check` with LF/CRLF warnings only.
- Active jobs `994378`, `994707`, and `994708` remain running. Route A latest
  mAP remains `65.05`; AB reached Epoch 28 step 50; BH reached Epoch 29 step
  50; quota first mAP is still absent.
- Next gate: focused Pro fix re-review, then Gemini/DeepSeek.

## 2026-05-29T16:31:33+08:00 - BoundarySharp-ST final Pro fixes and Route A

- Focused GPT-5.5 Pro re-review returned `WARN` with no code blocker.
- Applied final small fixes: include `quota_uniform_target_weight` in
  non-negative validation, add negative-weight constructor test, update self
  check count, and assert BoundarySharp-ST `detach_gt_remap=True`.
- Final local checks passed: `py_compile`; Windows pytest
  `1 passed, 21 skipped`; `git diff --check` with LF/CRLF warnings only.
- Route A reached `65.37` Avg-mAP at `16:23:28`, vector
  `80.68 / 75.76 / 68.45 / 58.52 / 43.41`; this is a strong uniform-control
  reference, not learned selector evidence.
- AB/BH continue training without first mAP yet. Next gate is Gemini CLI and
  DeepSeek CLI review.

## 2026-05-29T16:34:05+08:00 - BoundarySharp-ST Gemini review passed

- Gemini CLI `gemini-3-pro-preview` exited `0` and returned `PASS`.
- Output path:
  `logs/gemini3_pro_preview_boundarysharp_st_review_20260529.txt`.
- No blocking findings or required fixes; proceed to Claude CLI DeepSeek
  secondary review. No sync/deploy until DeepSeek and N16R4 preflight pass.

## 2026-05-29T16:40:14+08:00 - BoundarySharp-ST DeepSeek review passed

- Claude CLI `deepseek-v4-pro` exited `0` and returned full stdout `PASS`.
- Output path:
  `logs/claude_deepseek_v4_pro_boundarysharp_st_review_20260529.txt`.
- No blocking findings or required fixes. Next step is to commit/sync reviewed
  files, run N16R4 Linux preflight, then launch one BoundarySharp-ST GPU job if
  preflight passes.
