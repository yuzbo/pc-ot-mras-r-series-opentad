# BATA E2E Raw-Density Selector Deployment - 2026-05-29

## Route

- Method: detector-internal raw-frame continuous temporal-density selector.
- Commit state: `c92d6d2 add e2e raw density frame selector`, `d617ee2 fix e2e n16r4 profile sourcing`.
- Input/compute label: `50% ViT/backbone compute, dense decode`.
- Protocol: dense 768-frame windows are decoded; selector outputs 384 continuous sampled frames before ViT-Adapter; GT remap is detached; test-time metadata records irregular selected positions; no test-time GT, teacher, or cache.

## Review Gate

- Pro design discussion: `logs/gpt5pro_e2e_frame_selector_discussion_20260529.txt`.
- Self-check: `BATA_E2E_RAW_DENSITY_SELECTOR_SELF_CHECK_20260529.md`.
- GPT-5.5 Pro implementation review: `logs/gpt5pro_e2e_raw_density_implementation_review_20260529.txt`, verdict `WARN`.
- GPT-5.5 Pro focused fix review: `logs/gpt5pro_e2e_raw_density_fix_review_20260529.txt`, verdict `WARN` but allowed Gemini/DeepSeek/preflight/launch after added checks.
- Gemini CLI review: `logs/gemini3_pro_preview_e2e_raw_density_review_20260529.txt`, exit code `0`, `PASS`.
- DeepSeek CLI review: `logs/claude_deepseek_v4_pro_e2e_raw_density_review_20260529.txt`, then focused pathfix review `logs/claude_deepseek_v4_pro_e2e_raw_density_pathfix_review_20260529.txt`, exit code `0`, `PASS`.

## Deployment

- Remote tree: `~/run/yuzibo/OpenTAD_BATA_Clean`.
- Initial Slurm jobs `994337`, `994338`, `994339` were canceled before effective training after a resource-display misread; no checkpoint or mAP resulted.
- Relaunched jobs:
  - `994340 e2e_main`: `configs/adatad/thumos/e2e_rawdensel_384of768_detachgt_cont_adapter.py`, node `g0005`.
  - `994341 e2e_uniform`: `configs/adatad/thumos/e2e_rawdensel_384of768_uniform_frozen_control_adapter.py`, node `g0005`.
  - `994342 e2e_lowreg`: `configs/adatad/thumos/e2e_rawdensel_384of768_lowreg_adapter.py`, node `g0024`.
- Actual allocation for each job: `cpu=8,mem=124400M,gres/gpu=1`.
- Work dirs: `~/run/yuzibo/e2e_runs/exps/e2e_rawdensel_{main,uniform,lowreg}_20260529_023345`.
- Logs: `~/run/yuzibo/OpenTAD_BATA_Clean/logs/e2e_rawdensel_{main,uniform,lowreg}_20260529_023345_n16r4.log`.

## First Health Evidence

- Remote preflight passed for all three configs: `9 passed` selector tests and `merged_config_preflight=PASS`.
- All three runs reached `Training Starts`, completed Epoch 0, and entered Epoch 1.
- First Epoch-0 losses:
  - main: `1.7367 -> 1.6396`, selector losses logged as `0.0000` at this precision.
  - uniform control: `1.7243 -> 1.6308`, no selector loss terms as expected for the frozen uniform control.
  - lowreg: `1.7367 -> 1.6397`, selector losses logged as `0.0000` at this precision.
- Follow-up tail: main reached Epoch 1 step 50 with `Loss=1.0602`; uniform reached Epoch 1 step 50 with `Loss=1.0766`; lowreg reached Epoch 1 step 50 with `Loss=1.0352`.
- Epoch 1 completed for all three: main `Loss=1.0096`, uniform `Loss=1.0243`, lowreg `Loss=0.9974`; all three entered Epoch 2.
- Epoch 2 step 50 was reached for all three: main `Loss=0.8457`, uniform `Loss=0.8518`, lowreg `Loss=0.8469`.
- All three completed Epoch 2 and entered Epoch 3: main `Loss=0.8800`, uniform `Loss=0.8790`, lowreg `Loss=0.8758`.
- Epoch 3 step 50 remains healthy for all three: main `Loss=0.8321`, uniform `Loss=0.8344`, lowreg `Loss=0.8341`.
- All three completed Epoch 3 and entered Epoch 4: main `Loss=0.8331`, uniform `Loss=0.8302`, lowreg `Loss=0.8327`.
- All three reached Epoch 4 step 50: main `Loss=0.8828`, uniform `Loss=0.8700`, lowreg `Loss=0.8636`.
- All three completed Epoch 4 and entered Epoch 5: main `Loss=0.8549`, uniform `Loss=0.8355`, lowreg `Loss=0.8131`.
- All three reached Epoch 5 step 50: main `Loss=0.7874`, uniform `Loss=0.7857`, lowreg `Loss=0.7816`.
- All three completed Epoch 5 and entered Epoch 6: main `Loss=0.8128`, uniform `Loss=0.7987`, lowreg `Loss=0.7981`.
- Main and uniform reached Epoch 6 step 50: main `Loss=0.7297`, uniform `Loss=0.7201`; lowreg is in the normal short interval after Epoch 6 start.
- Main and uniform completed Epoch 6 and entered Epoch 7: main `Loss=0.7146`, uniform `Loss=0.7131`; lowreg reached Epoch 6 step 50 with `Loss=0.7288`.
- Main reached Epoch 7 step 50 with `Loss=0.7506`; lowreg completed Epoch 6 and entered Epoch 7 with `Loss=0.7107`.
- Main and uniform completed Epoch 7 and entered Epoch 8: main `Loss=0.7116`, uniform `Loss=0.7143`.
- Lowreg reached Epoch 7 step 50 with `Loss=0.7478`; its log is still within the normal progress window.
- Main reached Epoch 8 step 50 with `Loss=0.6926`; lowreg completed Epoch 7 with `Loss=0.7095` and entered Epoch 8.
- Main completed Epoch 8 with `Loss=0.6744` and entered Epoch 9; uniform completed Epoch 8 with `Loss=0.6715` and entered Epoch 9.
- Lowreg reached Epoch 8 step 50 with `Loss=0.6894`; no checkpoint directory exists yet under the actual `gpu1_id0` run subdirs.
- Main and uniform reached Epoch 9 step 50: main `Loss=0.6129`, uniform `Loss=0.6068`.
- Lowreg completed Epoch 8 with `Loss=0.6665` and entered Epoch 9; checkpoint dirs are still absent and `log.json` files are updating.
- Main and uniform completed Epoch 9 and entered Epoch 10: main `Loss=0.6595`, uniform `Loss=0.6483`.
- Lowreg reached Epoch 9 step 50 with `Loss=0.6161`; no one-hour no-change threshold is close.
- All three reached Epoch 10 step 50: main `Loss=0.6646`, uniform `Loss=0.6637`, lowreg `Loss=0.6613`.
- Checkpoint dirs remain absent and `log.json` files keep updating; this is still expected before epoch 20.
- All three entered Epoch 11: main completed Epoch 10 with `Loss=0.6605` and reached Epoch 11 step 50 with `Loss=0.6500`; uniform completed Epoch 10 with `Loss=0.6583`; lowreg completed Epoch 10 with `Loss=0.6632`.
- Checkpoint dirs are still absent; failure scan remains `0`.
- Main completed Epoch 11 with `Loss=0.6481` and entered Epoch 12.
- Uniform and lowreg reached Epoch 11 step 50: uniform `Loss=0.6484`, lowreg `Loss=0.6494`.
- All three entered Epoch 12: main reached Epoch 12 step 50 with `Loss=0.6318`; uniform completed Epoch 11 with `Loss=0.6398`; lowreg completed Epoch 11 with `Loss=0.6421`.
- Checkpoint dirs remain absent; failure scan remains `0`.
- Main completed Epoch 12 with `Loss=0.6377` and entered Epoch 13.
- Uniform and lowreg reached Epoch 12 step 50: uniform `Loss=0.6119`, lowreg `Loss=0.6099`.
- All three entered Epoch 13: main reached Epoch 13 step 50 with `Loss=0.6281`; uniform reached Epoch 13 step 50 with `Loss=0.6205`; lowreg completed Epoch 12 with `Loss=0.6183`.
- Checkpoint dirs remain absent; failure scan remains `0`.
- Latest monitor at `2026-05-29T03:55:27+08:00`: all three jobs remain `RUNNING` for about `1:21:12`.
- Main completed Epoch 13 with `Loss=0.6436`, entered Epoch 14, and reached Epoch 14 step 50 with `Loss=0.5678`, `cls_loss=0.3197`, `reg_loss=0.2481`.
- Uniform completed Epoch 13 with `Loss=0.6388`, `cls_loss=0.3610`, `reg_loss=0.2777`, then Epoch 14 started.
- Lowreg completed Epoch 13 with `Loss=0.6312`, `cls_loss=0.3567`, `reg_loss=0.2744`, then Epoch 14 started.
- Checkpoint dirs remain absent; only `log.json` files are updating. Failure scan remains `0`.
- Latest monitor at `2026-05-29T03:59:42+08:00`: all three jobs remain `RUNNING` for about `1:25:27`.
- Main completed Epoch 14 with `Loss=0.5930`, `cls_loss=0.3354`, `reg_loss=0.2576`, then Epoch 15 started.
- Uniform completed Epoch 14 with `Loss=0.6044`, `cls_loss=0.3438`, `reg_loss=0.2606`, then Epoch 15 started.
- Lowreg reached Epoch 14 step 50 with `Loss=0.5726`, `cls_loss=0.3253`, `reg_loss=0.2472`; no one-hour stall threshold is close.
- Checkpoint dirs remain absent; only `log.json` files are updating. Failure scan remains `0`; disk remains safe with about `1.9T` free on `/data`.
- Latest monitor at `2026-05-29T04:02:32+08:00`: all three jobs remain `RUNNING` for about `1:28:17`.
- Main reached Epoch 15 step 50 with `Loss=0.5939`, `cls_loss=0.3364`, `reg_loss=0.2575`.
- Uniform reached Epoch 15 step 50 with `Loss=0.6180`, `cls_loss=0.3597`, `reg_loss=0.2584`.
- Lowreg completed Epoch 14 with `Loss=0.5989`, `cls_loss=0.3416`, `reg_loss=0.2573`, then Epoch 15 started.
- Checkpoint dirs remain absent; only `log.json` files are updating. Failure scan remains `0`.
- Latest monitor at `2026-05-29T04:04:56+08:00`: all three jobs remain `RUNNING` for about `1:30:41`.
- Main completed Epoch 15 with `Loss=0.5885`, `cls_loss=0.3323`, `reg_loss=0.2561`, then Epoch 16 started.
- Uniform completed Epoch 15 with `Loss=0.6004`, `cls_loss=0.3419`, `reg_loss=0.2585`, then Epoch 16 started.
- Lowreg reached Epoch 15 step 50 with `Loss=0.6016`, `cls_loss=0.3439`, `reg_loss=0.2577`; no one-hour stall threshold is close.
- Checkpoint dirs remain absent; only `log.json` files are updating. Failure scan remains `0`.
- Latest monitor at `2026-05-29T04:07:07+08:00`: all three jobs remain `RUNNING` for about `1:32:52`.
- Main reached Epoch 16 step 50 with `Loss=0.5819`, `cls_loss=0.3339`, `reg_loss=0.2480`.
- Uniform completed Epoch 15 with `Loss=0.6004`, `cls_loss=0.3419`, `reg_loss=0.2585`, then Epoch 16 started.
- Lowreg completed Epoch 15 with `Loss=0.5902`, `cls_loss=0.3343`, `reg_loss=0.2559`, then Epoch 16 started.
- Checkpoint dirs remain absent; only `log.json` files are updating. Failure scan remains `0`.
- Latest monitor at `2026-05-29T04:09:26+08:00`: all three jobs remain `RUNNING` for about `1:35:11`.
- Main completed Epoch 16 with `Loss=0.6124`, `cls_loss=0.3467`, `reg_loss=0.2657`, then Epoch 17 started.
- Uniform reached Epoch 16 step 50 with `Loss=0.5737`, `cls_loss=0.3198`, `reg_loss=0.2539`.
- Lowreg remains in Epoch 16 window after completing Epoch 15 with `Loss=0.5902`; no one-hour stall threshold is close.
- Checkpoint dirs remain absent; only `log.json` files are updating. Failure scan remains `0`.
- Latest monitor at `2026-05-29T04:11:41+08:00`: all three jobs remain `RUNNING` for about `1:37:26`.
- Main reached Epoch 17 step 50 with `Loss=0.5231`, `cls_loss=0.2757`, `reg_loss=0.2474`.
- Uniform completed Epoch 16 with `Loss=0.6018`, `cls_loss=0.3347`, `reg_loss=0.2671`, then Epoch 17 started.
- Lowreg reached Epoch 16 step 50 with `Loss=0.5788`, `cls_loss=0.3296`, `reg_loss=0.2492`; no one-hour stall threshold is close.
- Checkpoint dirs remain absent; only `log.json` files are updating. Failure scan remains `0`.
- Latest monitor at `2026-05-29T04:14:04+08:00`: all three jobs remain `RUNNING` for about `1:39:49`.
- Main completed Epoch 17 with `Loss=0.5487`, `cls_loss=0.3007`, `reg_loss=0.2480`, then Epoch 18 started.
- Uniform reached Epoch 17 step 50 with `Loss=0.5179`, `cls_loss=0.2749`, `reg_loss=0.2430`.
- Lowreg completed Epoch 16 with `Loss=0.6061`, `cls_loss=0.3414`, `reg_loss=0.2647`, then Epoch 17 started.
- Checkpoint dirs remain absent; only `log.json` files are updating. Failure scan remains `0`.
- Latest monitor at `2026-05-29T04:16:36+08:00`: all three jobs remain `RUNNING` for about `1:42:21`.
- Main reached Epoch 18 step 50 with `Loss=0.5652`, `cls_loss=0.3068`, `reg_loss=0.2584`.
- Uniform completed Epoch 17 with `Loss=0.5462`, `cls_loss=0.2981`, `reg_loss=0.2481`, then Epoch 18 started.
- Lowreg reached Epoch 17 step 50 with `Loss=0.5124`, `cls_loss=0.2728`, `reg_loss=0.2395`; no one-hour stall threshold is close.
- Checkpoint dirs remain absent; only `log.json` files are updating. Failure scan remains `0`.
- Latest monitor at `2026-05-29T04:18:57+08:00`: all three jobs remain `RUNNING` for about `1:44:42`.
- Main completed Epoch 18 with `Loss=0.5603`, `cls_loss=0.3036`, `reg_loss=0.2567`, then Epoch 19 started.
- Uniform reached Epoch 18 step 50 with `Loss=0.5697`, `cls_loss=0.3045`, `reg_loss=0.2653`.
- Lowreg completed Epoch 17 with `Loss=0.5383`, `cls_loss=0.2958`, `reg_loss=0.2425`, then Epoch 18 started.
- Checkpoint dirs remain absent; only `log.json` files are updating. Failure scan remains `0`. First checkpoint is expected after the epoch-19/20 save boundary.
- Latest monitor at `2026-05-29T04:21:22+08:00`: all three jobs remain `RUNNING` for about `1:47:07`.
- Main remains in Epoch 19 after completing Epoch 18 with `Loss=0.5603`, `cls_loss=0.3036`, `reg_loss=0.2567`.
- Uniform completed Epoch 18 with `Loss=0.5612`, `cls_loss=0.3006`, `reg_loss=0.2606`, then Epoch 19 started.
- Lowreg reached Epoch 18 step 50 with `Loss=0.5512`, `cls_loss=0.2898`, `reg_loss=0.2614`.
- Checkpoint dirs remain absent; only `log.json` files are updating. Failure scan remains `0`. First checkpoint remains pending after the epoch-19/20 save boundary.
- Latest monitor at `2026-05-29T04:23:57+08:00`: all three jobs remain `RUNNING` for about `1:49:42`.
- Main reached Epoch 19 step 50 with `Loss=0.5650`, `cls_loss=0.2980`, `reg_loss=0.2669`.
- Uniform reached Epoch 19 step 50 with `Loss=0.5667`, `cls_loss=0.3030`, `reg_loss=0.2636`.
- Lowreg completed Epoch 18 with `Loss=0.5562`, `cls_loss=0.2974`, `reg_loss=0.2588`, then Epoch 19 started.
- Checkpoint dirs remain absent; only `log.json` files are updating. Failure scan remains `0`. First checkpoint remains pending.
- Main first checkpoint appeared during the same monitor window: `~/run/yuzibo/e2e_runs/exps/e2e_rawdensel_main_20260529_023345/gpu1_id0/checkpoint/epoch_19.pth`, size `623905307`, mtime `2026-05-29T04:26:26+08:00`.
- Main completed Epoch 19 with `Loss=0.5623`, `cls_loss=0.3117`, `reg_loss=0.2505`, then Epoch 20 started.
- Latest monitor at `2026-05-29T04:29:33+08:00`: all three jobs remain `RUNNING` for about `1:55:18`.
- Main `epoch_19.pth`: size `623905307`, mtime `04:26:26+08:00`; Epoch 20 step 50 `Loss=0.5278`, `cls_loss=0.2989`, `reg_loss=0.2289`.
- Uniform `epoch_19.pth`: size `623852123`, mtime `04:28:52+08:00`; Epoch 20 step 50 `Loss=0.5194`, `cls_loss=0.2898`, `reg_loss=0.2296`.
- Lowreg `epoch_19.pth`: size `623905307`, mtime `04:28:41+08:00`; Epoch 19 completed with `Loss=0.5598`, `cls_loss=0.3154`, `reg_loss=0.2444`, then Epoch 20 started.
- Failure scan remains `0`; no `Average-mAP`, checkpoint cleanup, crash, OOM, or severe-result trigger yet.

## Epoch-19 Selector Position Diagnostic

- Remote output: `~/run/yuzibo/e2e_runs/selector_diagnostics/e2e_rawdensel_epoch19_posdiag_20260529_044130/summary.json`; examples in `examples.jsonl`.
- Scope: selector-only CPU diagnostic over 48 sampled validation windows out of 487. It did not run ViT/backbone/detector, did not alter active Slurm jobs, and used validation GT only for offline labeling of action/boundary proximity.
- Latest run monitor at `2026-05-29T04:59:56+08:00`: jobs `994340`, `994341`, and `994342` remain `RUNNING` for about `2:25:41`; main and uniform have entered Epoch 26, lowreg has entered Epoch 25. Narrow failure scan for traceback/OOM/killed/runtime/nan is empty; no mAP yet.
- Follow-up monitor at `2026-05-29T05:03:17+08:00`: all three jobs remain `RUNNING` for about `2:29:05`. Main completed Epoch 26 with `Loss=0.4885` and entered Epoch 27; uniform reached Epoch 26 step 50 with `Loss=0.4887`; lowreg completed Epoch 25 with `Loss=0.5438` and entered Epoch 26. No mAP/result JSON is visible; narrow failure scan remains empty; `/data` still has about `1.9T` free.
- Follow-up monitor at `2026-05-29T05:06:06+08:00`: all three jobs remain `RUNNING` for about `2:31:51`. Main reached Epoch 27 step 50 with `Loss=0.4896`; uniform completed Epoch 26 with `Loss=0.4855` and entered Epoch 27; lowreg reached Epoch 26 step 50 with `Loss=0.4927`. No mAP/result artifact is visible; narrow failure scan remains empty; only each run's `epoch_19.pth` checkpoint is present.
- Follow-up monitor at `2026-05-29T05:08:41+08:00`: all three jobs remain `RUNNING` for about `2:34:26`. Main completed Epoch 27 with `Loss=0.5125` and entered Epoch 28; uniform reached Epoch 27 step 50 with `Loss=0.4944`; lowreg completed Epoch 26 with `Loss=0.4884` and entered Epoch 27. No mAP is visible, no new checkpoints beyond `epoch_19.pth`, and narrow failure scan remains empty.
- Follow-up monitor at `2026-05-29T05:11:10+08:00`: all three jobs remain `RUNNING` for about `2:36:55`. Main reached Epoch 28 step 50 with `Loss=0.4787`; uniform completed Epoch 27 with `Loss=0.5145` and entered Epoch 28; lowreg's latest line remains Epoch 27 start at `05:07:59`, below any stall threshold. No mAP is visible, no new checkpoints beyond `epoch_19.pth`, and narrow failure scan remains empty.
- Follow-up monitor at `2026-05-29T05:13:30+08:00`: all three jobs remain `RUNNING` for about `2:39:15`. Main completed Epoch 28 with `Loss=0.5021` and entered Epoch 29; uniform reached Epoch 28 step 50 with `Loss=0.4851`; lowreg reached Epoch 27 step 50 with `Loss=0.5031`, confirming the earlier short log gap was not a stall. No mAP is visible, no new checkpoints beyond `epoch_19.pth`, and narrow failure scan remains empty.
- Follow-up monitor at `2026-05-29T05:16:16+08:00`: all three jobs remain `RUNNING` for about `2:42:01`. Main reached Epoch 29 step 50 with `Loss=0.4869`; uniform completed Epoch 28 with `Loss=0.5039` and entered Epoch 29; lowreg completed Epoch 27 with `Loss=0.5261` and entered Epoch 28. No mAP is visible, no new checkpoints beyond `epoch_19.pth`, and narrow failure scan remains empty.
- Follow-up monitor at `2026-05-29T05:18:40+08:00`: all three jobs remain `RUNNING` for about `2:44:25`. Main completed Epoch 29 with `Loss=0.4909` and entered Epoch 30; uniform completed Epoch 28 and entered Epoch 29; lowreg reached Epoch 28 step 50 with `Loss=0.4848`. No mAP is visible, no new checkpoints beyond `epoch_19.pth`, and narrow failure scan remains empty.
- Follow-up monitor at `2026-05-29T05:21:10+08:00`: all three jobs remain `RUNNING` for about `2:46:55`. Main reached Epoch 30 step 50 with `Loss=0.4995`; uniform reached Epoch 29 step 50 with `Loss=0.4836`; lowreg completed Epoch 28 with `Loss=0.5081` and entered Epoch 29. No mAP is visible, no new checkpoints beyond `epoch_19.pth`, and narrow failure scan remains empty.
- Follow-up monitor at `2026-05-29T05:24:05+08:00` observed log evidence through `05:25:48+08:00`: all three jobs remain `RUNNING` for about `2:49:50`. Main completed Epoch 30 with `Loss=0.4885` and entered Epoch 31; uniform completed Epoch 29 with `Loss=0.4920` and entered Epoch 30; lowreg reached Epoch 29 step 50 with `Loss=0.4806`. Upload marker remains `ALL_PASS=True`; no mAP/result JSON is visible, no new checkpoints beyond `epoch_19.pth`, narrow failure scan remains empty, and `/data` still has about `1.9T` free.
- Follow-up monitor at `2026-05-29T05:27:22+08:00` observed log evidence through `05:29:24+08:00`: all three jobs remain `RUNNING` for about `2:53:07`. Main reached Epoch 31 step 50 with `Loss=0.4537`; uniform completed Epoch 30 with `Loss=0.4902` and entered Epoch 31; lowreg completed Epoch 29 with `Loss=0.4989` and entered Epoch 30. No mAP/result JSON is visible, no new checkpoints beyond `epoch_19.pth`, and narrow failure scan remains empty.
- Follow-up monitor at `2026-05-29T05:29:56+08:00` observed log evidence through `05:31:07+08:00`: all three jobs remain `RUNNING` for about `2:55:41`. Main completed Epoch 31 with `Loss=0.4589` and entered Epoch 32; uniform completed Epoch 30 with `Loss=0.4902` and entered Epoch 31; lowreg reached Epoch 30 step 50 with `Loss=0.5003`. Upload marker remains `ALL_PASS=True`; no mAP/result JSON is visible, no new checkpoints beyond `epoch_19.pth`, narrow failure scan remains empty, and `/data` still has about `1.9T` free.
- Follow-up monitor at `2026-05-29T05:32:58+08:00` observed log evidence through `05:34:48+08:00`: all three jobs remain `RUNNING` for about `2:58:43`. Main reached Epoch 32 step 50 with `Loss=0.4813`; uniform completed Epoch 31 with `Loss=0.4507` and entered Epoch 32; lowreg completed Epoch 30 with `Loss=0.4911` and entered Epoch 31. Upload marker remains `ALL_PASS=True`; no mAP/result JSON is visible, no new checkpoints beyond `epoch_19.pth`, narrow failure scan remains empty, and `/data` still has about `1.9T` free.
- Follow-up monitor at `2026-05-29T05:36:02+08:00` observed log evidence through `05:37:40+08:00`: all three jobs remain `RUNNING` for about `3:01:47`. Main completed Epoch 32 with `Loss=0.4817` and entered Epoch 33; uniform reached Epoch 32 step 50 with `Loss=0.4856`; lowreg reached Epoch 31 step 50 with `Loss=0.4571`. Upload marker remains `ALL_PASS=True`; no mAP/result JSON is visible, no new checkpoints beyond `epoch_19.pth`, narrow failure scan remains empty, and `/data` still has about `1.9T` free.
- Follow-up monitor at `2026-05-29T05:39:26+08:00` observed log evidence through `05:40:16+08:00`: all three jobs remain `RUNNING` for about `3:05:11`. Main reached Epoch 33 step 50 with `Loss=0.4977`; uniform completed Epoch 32 with `Loss=0.4853` and entered Epoch 33; lowreg completed Epoch 31 with `Loss=0.4673` and entered Epoch 32. Upload marker remains `ALL_PASS=True`; no mAP/result JSON is visible, no new checkpoints beyond `epoch_19.pth`, narrow failure scan remains empty, and `/data` still has about `1.9T` free.
- Follow-up monitor at `2026-05-29T05:42:41+08:00` observed log evidence through `05:44:32+08:00`: all three jobs remain `RUNNING` for about `3:08:26`. Main completed Epoch 33 with `Loss=0.4736`, entered Epoch 34, and reached Epoch 34 step 50 with `Loss=0.4655`; uniform reached Epoch 33 step 50 with `Loss=0.5096`; lowreg completed Epoch 32 with `Loss=0.5011` and entered Epoch 33. Upload marker remains `ALL_PASS=True`; no mAP/result JSON is visible, no new checkpoints beyond `epoch_19.pth`, narrow failure scan remains empty, and `/data` still has about `1.9T` free.
- Follow-up monitor at `2026-05-29T05:46:16+08:00` observed log evidence through `05:47:08+08:00`: all three jobs remain `RUNNING` for about `3:12:01`. Main completed Epoch 34 with `Loss=0.4765` and entered Epoch 35; uniform completed Epoch 33 with `Loss=0.4837` and entered Epoch 34; lowreg reached Epoch 33 step 50 with `Loss=0.5074`. Upload marker remains `ALL_PASS=True`; no mAP/result JSON is visible, no new checkpoints beyond `epoch_19.pth`, narrow failure scan remains empty, and `/data` still has about `1.9T` free.
- Follow-up monitor at `2026-05-29T05:49:40+08:00` observed log evidence through `05:51:10+08:00`: all three jobs remain `RUNNING` for about `3:15:25`. Main reached Epoch 35 step 50 with `Loss=0.4160`; uniform completed Epoch 34 with `Loss=0.4859` and entered Epoch 35; lowreg completed Epoch 33 with `Loss=0.4841` and entered Epoch 34. Upload marker remains `ALL_PASS=True`; no mAP/result JSON is visible, no new checkpoints beyond `epoch_19.pth`, narrow failure scan remains empty, and `/data` still has about `1.9T` free.
- Follow-up monitor at `2026-05-29T05:52:57+08:00` observed log evidence through `05:54:01+08:00`: all three jobs remain `RUNNING` for about `3:18:42`. Main completed Epoch 35 with `Loss=0.4439` and entered Epoch 36; uniform reached Epoch 35 step 50 with `Loss=0.4222`; lowreg reached Epoch 34 step 50 with `Loss=0.4785`. Upload marker remains `ALL_PASS=True`; no mAP/result JSON is visible, no new checkpoints beyond `epoch_19.pth`, narrow failure scan remains empty, and `/data` still has about `1.9T` free.
- Selector score-head movement at `epoch_19`: uniform final score layer remains exactly zero; main final layer norm is small (`weight=0.0091`, `bias=0.0061`); lowreg final layer moved more (`weight=0.0661`, `bias=0.0109`) but still produces a near-uniform allocation.
- Position diagnostic summary:

| Route | Mean abs delta vs uniform | P95 abs delta | Rounded Jaccard vs uniform | Selected action frac | Selected boundary<=4 frac | Boundary recall@4 | Logit std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| main | `1.03` | `1.43` | `0.023` | `0.287` | `0.144` | `0.985` | `0.00052` |
| uniform | `0.00` | `0.00` | `1.000` | `0.289` | `0.148` | `1.000` | `0.00000` |
| lowreg | `3.76` | `6.30` | `0.372` | `0.287` | `0.144` | `0.985` | `0.02847` |

- Octile fractions are nearly flat for all routes. Main is mostly a uniform stride-2 selector with a small global offset/compression; lowreg has a slightly stronger shift and weak late-octile bias, but neither route shows a meaningful action/boundary-aware selected-frame distribution at epoch 19.
- Interpretation: end-to-end gradient connectivity is implemented, but current early checkpoint evidence says the selector has not yet learned the desired frame allocation. Continue to first mAP and later checkpoints before making a stop decision.

## Latest Monitor

- Latest monitor at `2026-05-29T05:59:15+08:00`: Slurm and sacct report jobs `994340`, `994341`, and `994342` still `RUNNING` for about `3:25:00`; main/uniform are on `g0005`, lowreg is on `g0024`.
- Upload marker remains `ALL_PASS=True` with train `200/200` and test `211/211`; `/data` remains safe with about `1.9T` free.
- Main reached Epoch 37 step 50 at `06:00:44+08:00` with `Loss=0.4756`, `selector_spacing_loss=0.0001`, `selector_tv_loss=0.0000`, `cls_loss=0.2535`, `reg_loss=0.2220`.
- Uniform reached Epoch 36 step 50 at `05:59:35+08:00` with `Loss=0.4522`, `cls_loss=0.2338`, `reg_loss=0.2184`.
- Lowreg completed Epoch 35 at `05:58:28+08:00` with `Loss=0.4626`, `selector_spacing_loss=0.0000`, `selector_tv_loss=0.0000`, `cls_loss=0.2413`, `reg_loss=0.2213`, then entered Epoch 36.
- Checkpoints remain only `epoch_19.pth` for each active run. No `Average-mAP`, result artifact, traceback, OOM, NaN, or severe-result trigger is visible. No code/config/model change occurred. Contract remains `50% ViT/backbone compute, dense decode`; no test-time GT/teacher/cache. Decision: continue all jobs, do not clean active checkpoints or launch duplicates, and wait for the Epoch 39 checkpoint plus first eval after the Epoch 41 boundary.
- Follow-up monitor at `2026-05-29T06:01:36+08:00`: all three jobs remain `RUNNING` for about `3:27:21`. Main completed Epoch 37 with `Loss=0.4667`, `selector_spacing_loss=0.0001`, `selector_tv_loss=0.0000`, `cls_loss=0.2420`, `reg_loss=0.2246`, then entered Epoch 38. Uniform completed Epoch 36 with `Loss=0.4399`, `cls_loss=0.2269`, `reg_loss=0.2130`, then entered Epoch 37. Lowreg latest log remains Epoch 36 start at `05:58:28+08:00`, below any stall threshold. No mAP/result artifact or new checkpoint beyond `epoch_19.pth` is visible; failure scan is empty. Decision remains continue to Epoch 39 checkpoint and first eval after the Epoch 41 boundary.
- Follow-up monitor at `2026-05-29T06:05:56+08:00`: all three jobs remain `RUNNING` for about `3:31:41`. Main reached Epoch 38 step 50 with `Loss=0.4637`, `selector_spacing_loss=0.0001`, `selector_tv_loss=0.0000`, `cls_loss=0.2393`, `reg_loss=0.2243`; uniform completed Epoch 37 with `Loss=0.4688`, `cls_loss=0.2433`, `reg_loss=0.2256`, then entered Epoch 38; lowreg completed Epoch 36 with `Loss=0.4500`, `selector_spacing_loss=0.0000`, `selector_tv_loss=0.0000`, `cls_loss=0.2322`, `reg_loss=0.2177`, then entered Epoch 37. No mAP/result artifact or new checkpoint beyond `epoch_19.pth` is visible; failure scan is empty. Decision remains continue to Epoch 39 checkpoint and first eval after the Epoch 41 boundary.
- Follow-up monitor at `2026-05-29T06:07:18+08:00`: all three jobs remain `RUNNING` for about `3:33:03`. Main completed Epoch 38 with `Loss=0.4593`, `selector_spacing_loss=0.0001`, `selector_tv_loss=0.0000`, `cls_loss=0.2370`, `reg_loss=0.2222`, then entered Epoch 39. Uniform remains in Epoch 38, lowreg remains in Epoch 37. No mAP/result artifact or new checkpoint beyond `epoch_19.pth` is visible; failure scan is empty. Decision remains continue to `epoch_39.pth` and first eval after the Epoch 41 boundary.
- Follow-up monitor at `2026-05-29T06:08:26+08:00`: all three jobs remain `RUNNING` for about `3:34:11`. Main remains in Epoch 39; uniform reached Epoch 38 step 50 with `Loss=0.4738`, `cls_loss=0.2486`, `reg_loss=0.2252`; lowreg reached Epoch 37 step 50 with `Loss=0.4930`, `selector_spacing_loss=0.0000`, `selector_tv_loss=0.0000`, `cls_loss=0.2657`, `reg_loss=0.2273`. No mAP/result artifact or new checkpoint beyond `epoch_19.pth` is visible; failure scan is empty. Decision remains continue to `epoch_39.pth` and first eval after the Epoch 41 boundary.
- Follow-up monitor at `2026-05-29T06:10:26+08:00`: upload marker remains `ALL_PASS=True`; all three jobs remain `RUNNING` for about `3:36:11`; `/data` has about `1.9T` free. Main reached Epoch 39 step 50 with `Loss=0.4187`, `selector_spacing_loss=0.0001`, `selector_tv_loss=0.0000`, `cls_loss=0.2159`, `reg_loss=0.2027`. Uniform remains in Epoch 38 after step 50. Lowreg completed Epoch 37 with `Loss=0.4830`, `selector_spacing_loss=0.0000`, `selector_tv_loss=0.0000`, `cls_loss=0.2538`, `reg_loss=0.2292`, then entered Epoch 38. No mAP/result artifact or new checkpoint beyond `epoch_19.pth` is visible; failure scan is empty. Decision remains continue to `epoch_39.pth` and first eval after the Epoch 41 boundary.
- Follow-up monitor at `2026-05-29T06:11:50+08:00`: all three jobs remain `RUNNING` for about `3:37:35`. Main remains in Epoch 39 after step 50 and has not written `epoch_39.pth`. Uniform completed Epoch 38 with `Loss=0.4693`, `cls_loss=0.2438`, `reg_loss=0.2254`, then entered Epoch 39. Lowreg remains in Epoch 38 after completing Epoch 37. No mAP/result artifact or new checkpoint beyond `epoch_19.pth` is visible; failure scan is empty. Decision remains continue to `epoch_39.pth` and first eval after the Epoch 41 boundary.
- Follow-up monitor at `2026-05-29T06:12:57+08:00`: main completed Epoch 39 with `Loss=0.4529`, `selector_spacing_loss=0.0001`, `selector_tv_loss=0.0000`, `cls_loss=0.2351`, `reg_loss=0.2178`, wrote `checkpoint/epoch_39.pth` size `623905307` mtime `2026-05-29T06:14:07+08:00`, then entered Epoch 40. Uniform entered Epoch 39 and still has only `epoch_19.pth`; lowreg remains in Epoch 38 with only `epoch_19.pth`. No mAP/result artifact or failure scan hit is visible. No checkpoint cleanup was performed because all runs are active. Decision remains continue to uniform/lowreg `epoch_39.pth` and first eval after the Epoch 41 boundary.
- Follow-up monitor at `2026-05-29T06:14:54+08:00`: upload marker remains `ALL_PASS=True`; all three jobs remain `RUNNING` for about `3:40:39`; `/data` has about `1.9T` free. Main reached Epoch 40 step 50 with `Loss=0.4445`, `selector_spacing_loss=0.0001`, `selector_tv_loss=0.0000`, `cls_loss=0.2307`, `reg_loss=0.2137` and has `epoch_19.pth` plus `epoch_39.pth`. Uniform reached Epoch 39 step 50 with `Loss=0.4138`, `cls_loss=0.2157`, `reg_loss=0.1980` but still has only `epoch_19.pth`. Lowreg reached Epoch 38 step 50 with `Loss=0.4791`, `selector_spacing_loss=0.0000`, `selector_tv_loss=0.0000`, `cls_loss=0.2498`, `reg_loss=0.2292`, still only `epoch_19.pth`. No mAP/result artifact or failure scan hit is visible. No checkpoint cleanup was performed because all runs are active. Decision remains continue to uniform/lowreg `epoch_39.pth` and first eval after the Epoch 41 boundary.
- Follow-up monitor at `2026-05-29T06:16:58+08:00`: upload marker remains `ALL_PASS=True`; all three jobs remain `RUNNING` for about `3:42:43`; `/data` has about `1.9T` free. Main remains in Epoch 40 after step 50 and has `epoch_19.pth` plus `epoch_39.pth`. Uniform completed Epoch 39 with `Loss=0.4494`, `cls_loss=0.2347`, `reg_loss=0.2147`, wrote `checkpoint/epoch_39.pth` size `623852123` mtime `2026-05-29T06:18:54+08:00`, then entered Epoch 40. Lowreg completed Epoch 38 with `Loss=0.4713`, `selector_spacing_loss=0.0000`, `selector_tv_loss=0.0000`, `cls_loss=0.2441`, `reg_loss=0.2272`, then entered Epoch 39, still only `epoch_19.pth`. No mAP/result artifact or failure scan hit is visible. No checkpoint cleanup was performed because all runs are active. Decision remains continue to lowreg `epoch_39.pth` and first eval after the Epoch 41 boundary.
- Follow-up monitor at `2026-05-29T06:18:19+08:00`: all three jobs remain `RUNNING` for about `3:44:04`. Main completed Epoch 40 with `Loss=0.4491`, `selector_spacing_loss=0.0001`, `selector_tv_loss=0.0000`, `cls_loss=0.2346`, `reg_loss=0.2144`, then entered Epoch 41; main checkpoints remain `epoch_19.pth` and `epoch_39.pth`. Uniform has `epoch_19.pth` and `epoch_39.pth` and remains in Epoch 40. Lowreg remains in Epoch 39 with only `epoch_19.pth`. No mAP/result artifact or failure scan hit is visible. No checkpoint cleanup was performed because all runs are active. Decision remains continue to lowreg `epoch_39.pth` and first eval after the Epoch 41 boundary.
- Follow-up monitor at `2026-05-29T06:19:37+08:00`: all three jobs remain `RUNNING` for about `3:45:22`. Main remains in Epoch 41 after entering at `06:19:32`, with no eval/mAP yet. Uniform reached Epoch 40 step 50 with `Loss=0.4369`, `cls_loss=0.2244`, `reg_loss=0.2125` and has `epoch_19.pth`, `epoch_39.pth`. Lowreg reached Epoch 39 step 50 with `Loss=0.4301`, `selector_spacing_loss=0.0000`, `selector_tv_loss=0.0000`, `cls_loss=0.2262`, `reg_loss=0.2039`, still only `epoch_19.pth`. No mAP/result artifact or failure scan hit is visible. No checkpoint cleanup was performed because all runs are active. Decision remains continue to lowreg `epoch_39.pth` and first eval after the Epoch 41 boundary.
- Follow-up monitor at `2026-05-29T06:21:31+08:00`: upload marker remains `ALL_PASS=True`; all three jobs remain `RUNNING` for about `3:47:16`; `/data` has about `1.9T` free. Main reached Epoch 41 step 50 with `Loss=0.4416`, `selector_spacing_loss=0.0001`, `selector_tv_loss=0.0000`, `cls_loss=0.2276`, `reg_loss=0.2139`, with checkpoints `epoch_19.pth`, `epoch_39.pth`. Uniform remains in Epoch 40 after step 50 with `epoch_19.pth`, `epoch_39.pth`. Lowreg remains in Epoch 39 after step 50 and still only has `epoch_19.pth`. No mAP/result artifact or failure scan hit is visible. No checkpoint cleanup was performed because all runs are active. Decision remains continue to lowreg `epoch_39.pth` and first eval after the Epoch 41 boundary.
- Follow-up monitor at `2026-05-29T06:23:20+08:00`: upload marker remains `ALL_PASS=True`; all three jobs remain `RUNNING` for about `3:49:05`; `/data` has about `1.9T` free. Main completed Epoch 41 with `Loss=0.4640`, `selector_spacing_loss=0.0001`, `selector_tv_loss=0.0000`, `cls_loss=0.2394`, `reg_loss=0.2246`; no eval/mAP line is visible yet. Uniform completed Epoch 40 with `Loss=0.4460`, `cls_loss=0.2315`, `reg_loss=0.2145`, then entered Epoch 41. Lowreg completed Epoch 39 with `Loss=0.4681`, `selector_spacing_loss=0.0000`, `selector_tv_loss=0.0000`, `cls_loss=0.2454`, `reg_loss=0.2227`, wrote `checkpoint/epoch_39.pth` size `623905307` mtime `2026-05-29T06:21:06+08:00`, then entered Epoch 40. All three now have `epoch_19.pth` and `epoch_39.pth`. No mAP/result artifact or failure scan hit is visible. No checkpoint cleanup was performed because all runs are active. Decision remains continue to first eval/mAP after Epoch 41 boundary.
- Raw-tail check at `2026-05-29T06:25:02+08:00` confirms main first eval/inference started immediately after Epoch 41 completion at `06:24:52+08:00`; tqdm progress reached about `63/396` validation windows. No `Average-mAP` or result file is visible yet, and main run files currently include config, `log.json`, `epoch_19.pth`, and `epoch_39.pth`. No failure scan hit or checkpoint cleanup. Decision remains continue monitoring until main first mAP appears, then apply the configured baseline/severe-result gates.
- Follow-up raw-tail check at `2026-05-29T06:29:54+08:00`: Slurm still reports jobs `994340`, `994341`, and `994342` as `RUNNING`; upload marker remains `ALL_PASS=True`; `/data` remains about `1.9T` free. Main first validation advanced to about `135/396` windows after Epoch 41; uniform completed Epoch 41 and started its own first validation with raw tqdm around `4/396`; lowreg completed Epoch 40 and entered Epoch 41. All three active runs have `epoch_19.pth` and `epoch_39.pth`. No `Average-mAP`, result artifact, traceback, OOM, NaN, or severe-result trigger is visible. No code/config/model change and no active checkpoint cleanup. Decision remains continue all three jobs until first mAP appears.
- Follow-up raw-tail check at `2026-05-29T06:35:23+08:00`: all three jobs remain `RUNNING`. Main first validation reached about `247/396`; uniform first validation reached about `145/396`; lowreg completed Epoch 41 at `06:32:18+08:00` and started first validation, reaching about `14/396`. No `Average-mAP`, result artifact, traceback, OOM, NaN, or severe-result trigger is visible; `/data` remains about `1.9T` free. Decision remains continue monitoring to first mAP, with no checkpoint cleanup while runs are active.
- Main first eval result at `2026-05-29T06:43:15+08:00`: `Average-mAP=63.72`, vector `79.50 / 74.72 / 65.97 / 57.22 / 41.22`. This is `-0.05` vs random-fixed Adapter `63.77`, `-0.13` vs strict EMA `63.85`, `-0.92` vs stratified `64.64`, and `-1.37` vs uniform stride-2 `65.09`; it is below the positive-claim gate but not a severe collapse. Training resumed to Epoch 42. Uniform first eval remained in progress at about `316/396` and lowreg at about `206/396`; no failure scan hit. Decision: wait for uniform control and lowreg first mAP before route interpretation or follow-up launches.
- Uniform frozen-control first eval result at `2026-05-29T06:47:57+08:00`: `Average-mAP=64.12`, vector `79.81 / 74.81 / 67.11 / 56.77 / 42.07`. This is above random-fixed/strict EMA but below the `~64.3` uniform-control concern line and below the uniform stride-2 reference `65.09` by `-0.97`, so protocol/control implementation differences remain a live concern. Lowreg first eval remained in progress at about `308/396`; no failure scan hit. Decision: wait for lowreg first mAP, then interpret main/lowreg relative to this imperfect control before any follow-up launch.
- Lowreg first eval result at `2026-05-29T06:50:59+08:00`: `Average-mAP=62.98`, vector `78.76 / 74.32 / 65.87 / 55.03 / 40.93`. Three-way first-eval order is uniform `64.12` > main `63.72` > lowreg `62.98`; lowreg is `-0.79` vs random-fixed and `-2.11` vs uniform stride-2. This is below the positive-claim gate but not a severe collapse. Decision: continue running for later checkpoints, but start Pro discussion before any new long follow-up because first-eval evidence and epoch-19 selector diagnostic both suggest the selector movement is not yet useful.
- Pro discussion completed at `2026-05-29T07:05:00+08:00`: accepted report is `research-wiki/experiments/BATA_E2E_RAW_DENSITY_SELECTOR_PRO_DISCUSSION_20260529.md`; full stdout is `logs/oracle_pro_e2e_rawdensel_discussion_20260529.txt`. Accepted next routes are Route A uniform parity/protocol repair, Route B train-GT auxiliary action/boundary selector with residual slots, and Route C coordinate-aware head. Decision: continue current jobs for later/final evidence, do not launch main/lowreg lambda sweeps, and prepare Route A/B implementation/deployment first.

## Route A/B Follow-Up Staging

- Self-check recorded at `2026-05-29T07:14:15+08:00`: `research-wiki/experiments/BATA_E2E_ROUTE_AB_SELF_CHECK_20260529.md`.
- Route A config: `configs/adatad/thumos/e2e_rawdensel_384of768_uniform_exact_bypass_adapter.py`.
- Route B config: `configs/adatad/thumos/e2e_rawdensel_384of768_gtaux_residual_adapter.py`.
- Local verification from `OpenTAD_BATA_Clean`:
  - `python -m py_compile ...`: PASS.
  - `python -m pytest tests\test_e2e_raw_frame_selector_contracts.py -q`: `1 passed, 9 skipped` on Windows.
  - `git diff --check -- opentad/models/selectors/temporal_density_selector.py tests/test_e2e_raw_frame_selector_contracts.py`: PASS, only LF-to-CRLF warnings.
- Changed surface: selector internals and optional Route B train auxiliary selector loss. No Adapter internals, detector head, assignment, post-processing, or dataloader cache change.
- Protocol: still `50% ViT/backbone compute, dense decode`; Route B GT density target is train-only; no test-time GT/teacher/cache.
- Decision: Route A/B are `CODED-LOCAL-REVIEWING`; do not sync or launch until GPT-5.5 Pro implementation review, Gemini CLI review, DeepSeek CLI verification, and remote Linux preflight pass.
- Initial GPT-5.5 Pro Route A/B review completed after this staging and returned `FAIL`; output `logs/gpt5pro_e2e_route_ab_review_20260529.txt`. Fixes are recorded in `research-wiki/experiments/BATA_E2E_ROUTE_AB_PRO_REVIEW_AND_FIXES_20260529.md`.
- Post-fix local changes: exact tail stride for Route A, last-valid-dense padding, `min_position_gap` for Route B anchor/residual positions, mask shape guard, and focused contract tests.
- Post-fix local verification: `py_compile` PASS; pytest `1 passed, 12 skipped`; `git diff --check` PASS except LF-to-CRLF warnings.
- Focused GPT-5.5 Pro re-review returned `PASS` for entering Gemini/DeepSeek/N16R4 preflight: `logs/gpt5pro_e2e_route_ab_fix_review_20260529.txt`.
- Gemini CLI `gemini-3-pro-preview` returned exit code `0`, verdict `PASS`: `logs/gemini3_pro_preview_e2e_route_ab_review_20260529.txt`.
- Claude CLI DeepSeek `deepseek-v4-pro` returned exit code `0`, verdict `PASS`: `logs/claude_deepseek_v4_pro_e2e_route_ab_review_20260529.txt`.
- Reviewed commit: `7d525ab add e2e route ab selector controls`.
- Decision: sync reviewed Route A/B files to N16R4 and run Linux preflight. Do not launch long Route A/B jobs until preflight passes.

## Later Active-Run Evidence

- Monitor at `2026-05-29T07:44:29+08:00`: jobs `994340`, `994341`, and `994342` remain `RUNNING` for about `5:10:14`; no traceback/OOM/NaN; `/data` has about `1.9T` free.
- Main mAP trend: `63.72 -> 64.15 -> 64.09`.
- Uniform frozen-control mAP trend: `64.12 -> 64.52 -> 64.76`.
- Lowreg mAP trend so far: `62.98 -> 63.55`, third eval still in progress.
- Interpretation: uniform control is now above the earlier `64.3` concern line but still below the known `65.09` stride-2 reference; main remains below uniform. Continue active jobs; do not clean checkpoints.

## Route A/B N16R4 Deployment

- Standalone remote preflight at `2026-05-29T07:47:57+08:00` passed:
  - log `~/run/yuzibo/OpenTAD_BATA_Clean/logs/e2e_route_ab_preflight_20260529.log`;
  - `pytest tests/test_e2e_raw_frame_selector_contracts.py -q -rs`: `13 passed in 21.03s`;
  - merged config preflight for Route A/B: `PASS`.
- Slurm submission at `2026-05-29T07:49:25+08:00`:
  - `994378 e2e_routeA`, config `configs/adatad/thumos/e2e_rawdensel_384of768_uniform_exact_bypass_adapter.py`, node `g0009`;
  - `994379 e2e_routeB`, config `configs/adatad/thumos/e2e_rawdensel_384of768_gtaux_residual_adapter.py`, node `g0009`.
- Run tags and logs:
  - Route A: `e2e_routeA_uniform_exact_20260529_074925`, log `~/run/yuzibo/OpenTAD_BATA_Clean/logs/e2e_routeA_uniform_exact_20260529_074925_n16r4.log`;
  - Route B: `e2e_routeB_gtaux_residual_20260529_074925`, log `~/run/yuzibo/OpenTAD_BATA_Clean/logs/e2e_routeB_gtaux_residual_20260529_074925_n16r4.log`.
- Job-internal preflight passed for both jobs: `13 passed` and `merged_config_preflight=PASS`.
- Training started at `2026-05-29T07:52:17+08:00`.
- Epoch 0 step 50 health:
  - Route A: `Loss=1.7254`, `cls_loss=0.9846`, `reg_loss=0.7407`, mem `2652MB`.
  - Route B: `Loss=2.0576`, `selector_spacing_loss=0.0002`, `selector_tv_loss=0.0001`, `selector_gt_density_loss=0.3187`, `cls_loss=0.9877`, `reg_loss=0.7509`, mem `3185MB`.
- No traceback/OOM/Killed/RuntimeError/NaN is visible. Deterministic warn-only messages for interpolation/cumsum are present and non-fatal.
- Existing run update: lowreg third eval is `63.42`, so the old lowreg route remains below the old uniform control.
- Decision: continue all five E2E jobs. Do not clean active checkpoints. Route B early diagnostics after the first checkpoint should check whether the selector distribution is no longer uniform-like.

## Follow-Up Monitor at 08:01

- Upload gate still passes: train `200/200`, test `211/211`, `bad_size=0`, `ALL_PASS=True`.
- All five E2E jobs remain `RUNNING`.
- Route A loss trend: Epoch 0 step 50 `1.7254`, Epoch 0 final `1.6307`, Epoch 1 step 50 `1.0858`.
- Route B loss trend: Epoch 0 step 50 `2.0576`, Epoch 0 final `1.9384`, Epoch 1 step 50 `1.3239`; `selector_gt_density_loss` remains active around `0.319-0.320`.
- Old mAP trends: main `63.72 -> 64.15 -> 64.09`, uniform `64.12 -> 64.52 -> 64.76`, lowreg `62.98 -> 63.55 -> 63.42`.
- Failure scan remains empty and `/data` has about `1.9T` free.

## Follow-Up Monitor at 08:06

- Remote time: `2026-05-29T08:06:30+08:00`.
- Upload gate still passes: train `200/200`, test `211/211`, `bad_size=0`, `ALL_PASS=True`.
- All five E2E jobs remain `RUNNING`: Route A/B on `g0009`, old main/uniform on `g0005`, and old lowreg on `g0024`.
- Route A completed Epoch 1 at `08:04:28`: `Loss=1.0265`, `cls_loss=0.6287`, `reg_loss=0.3978`.
- Route B completed Epoch 1 at `08:04:07`: `Loss=1.3023`, `selector_gt_density_loss=0.3197`, `cls_loss=0.6368`, `reg_loss=0.3455`.
- Old mAP trends remain main `63.72 -> 64.15 -> 64.09`, uniform `64.12 -> 64.52 -> 64.76`, lowreg `62.98 -> 63.55 -> 63.42`.
- Failure scan remains `0` for all five logs. Route A/B have not reached their first checkpoint yet; old runs have active `epoch_19.pth` and `epoch_39.pth`. No checkpoint cleanup was performed.
- `/data` remains safe at about `2.3T` total, `418G` used, `1.9T` free.

## Monitor Retry at 08:12

- Two native OpenSSH monitor attempts to `ssh.cn-zhongwei-1.paracloud.com:22` timed out before reaching the login node.
- Local `Test-NetConnection` also timed out and reported TCP failures to `36.103.203.5:22` and `36.103.203.6:22`.
- No fresh Slurm/log/checkpoint/mAP evidence was obtained. This is treated as `WAIT` due to login connectivity, not as evidence of training failure or stall.
- Last authoritative remote state remains the successful `08:06` monitor: upload gate passed, all five jobs running, failure scan `0`, Route A/B no checkpoint yet, and disk safe.
- No job control, cleanup, relaunch, or remote filesystem action was performed.

## Follow-Up Monitor at 08:16

- Login connectivity recovered. Remote time: `2026-05-29T08:16:13+08:00`.
- Upload gate still passes: train `200/200`, test `211/211`, `bad_size=0`, `ALL_PASS=True`.
- All five E2E jobs remain `RUNNING`: Route A/B on `g0009`, old main/uniform on `g0005`, old lowreg on `g0024`.
- Old main fourth eval: `64.04` Avg-mAP, vector `79.80 / 74.77 / 67.18 / 56.81 / 41.66`; trend `63.72 -> 64.15 -> 64.09 -> 64.04`.
- Old uniform fourth eval: `65.01` Avg-mAP, vector `80.60 / 75.58 / 68.38 / 57.60 / 42.90`; trend `64.12 -> 64.52 -> 64.76 -> 65.01`. This is close to the known uniform stride-2 reference `65.09`.
- Old lowreg has no new mAP after `63.42`; latest train evidence reached Epoch 47 final and likely entered the next eval window.
- Route A reached Epoch 3 final: `Loss=0.8254`, `cls_loss=0.4943`, `reg_loss=0.3311`.
- Route B reached Epoch 3 final: `Loss=1.1547`, `selector_gt_density_loss=0.3198`, `cls_loss=0.5031`, `reg_loss=0.3316`.
- Failure scan remains `0` for all five logs. Route A/B still have no first checkpoint. No active checkpoint cleanup was performed.
- `/data` remains safe with about `1.9T` free.

## Follow-Up Monitor at 08:19

- Remote time: `2026-05-29T08:19:49+08:00`.
- Upload gate still passes: train `200/200`, test `211/211`, `bad_size=0`, `ALL_PASS=True`.
- All five E2E jobs remain `RUNNING`.
- No new mAP after the `08:16` record: old main remains `64.04`, old uniform remains `65.01`, and old lowreg remains `63.42`.
- Latest train evidence: old main completed Epoch 49 with `Loss=0.4492`; old uniform completed Epoch 48 with `Loss=0.4190`; Route A reached Epoch 4 step 50 with `Loss=0.8824`; Route B reached Epoch 4 step 50 with `Loss=1.1892` and `selector_gt_density_loss=0.3212`.
- Failure scan remains `0` for all five logs. No `Training Over`, no Route A/B `epoch_19.pth`, and no cleanup/diagnostic trigger.
- `/data` remains safe with about `1.9T` free.

## Follow-Up Monitor at 08:22

- Remote time: `2026-05-29T08:22:51+08:00`.
- Upload gate still passes: train `200/200`, test `211/211`, `bad_size=0`, `ALL_PASS=True`.
- All five E2E jobs remain `RUNNING`.
- New result: old lowreg fourth eval reached `63.91` Avg-mAP, vector `79.04 / 73.96 / 67.00 / 57.27 / 42.29`; trend `62.98 -> 63.55 -> 63.42 -> 63.91`.
- Interpretation: lowreg is now slightly above random-fixed `63.77` and strict EMA `63.85`, but still below old main `64.04`, stratified `64.64`, old uniform `65.01`, and uniform stride-2 `65.09`.
- Route A completed Epoch 4: `Loss=0.8293`, `cls_loss=0.5068`, `reg_loss=0.3225`.
- Route B completed Epoch 4: `Loss=1.1473`, `selector_gt_density_loss=0.3198`, `cls_loss=0.5022`, `reg_loss=0.3251`.
- Failure scan remains `0`; no `Training Over`; Route A/B still have no first checkpoint. No cleanup or diagnostic trigger.
- `/data` remains safe with about `1.9T` free.

## Follow-Up Monitor at 08:26

- Remote time: `2026-05-29T08:26:07+08:00`.
- Upload gate still passes: train `200/200`, test `211/211`, `bad_size=0`, `ALL_PASS=True`.
- All five E2E jobs remain `RUNNING`.
- No new mAP after lowreg `63.91`. Old main remains at `64.04` and is likely in post-Epoch-49 eval.
- Old uniform completed Epoch 49 with `Loss=0.4193`; old lowreg resumed training and reached Epoch 48 step 50 with `Loss=0.4496`.
- Route A reached Epoch 5 step 50 with `Loss=0.7834`.
- Route B completed Epoch 5 with `Loss=1.1248`, `selector_gt_density_loss=0.3197`, `cls_loss=0.4822`, `reg_loss=0.3226`.
- Failure scan remains `0`; no `Training Over`; Route A/B still have no first checkpoint. No cleanup or diagnostic trigger.
- `/data` remains safe with about `1.9T` free.

## Follow-Up Monitor at 08:29

- Remote time: `2026-05-29T08:29:25+08:00`.
- Upload gate still passes: train `200/200`, test `211/211`, `bad_size=0`, `ALL_PASS=True`.
- All five E2E jobs remain `RUNNING`.
- No new mAP: old main remains `64.04`, old uniform remains `65.01`, old lowreg remains `63.91`.
- Old main has no train line after Epoch 49 final; old uniform has no train line after Epoch 49 final, so both are likely in eval windows.
- Old lowreg completed Epoch 48 with `Loss=0.4378`.
- Route A completed Epoch 5 with `Loss=0.8036`, `cls_loss=0.4847`, `reg_loss=0.3189`.
- Route B reached Epoch 6 step 50 with `Loss=1.0525`, `selector_gt_density_loss=0.3184`, `cls_loss=0.4408`, `reg_loss=0.2930`.
- Failure scan remains `0`; no `Training Over`; Route A/B still have no first checkpoint. No cleanup or diagnostic trigger.
- `/data` remains safe with about `1.9T` free.

## Follow-Up Monitor at 08:32

- Remote time: `2026-05-29T08:32:21+08:00`.
- Upload gate still passes: train `200/200`, test `211/211`, `bad_size=0`, `ALL_PASS=True`.
- All five E2E jobs remain `RUNNING`.
- No new mAP: old main remains `64.04`, old uniform remains `65.01`, old lowreg remains `63.91`.
- Old main/uniform still have no new train line after Epoch 49 final, consistent with ongoing eval windows.
- Old lowreg reached Epoch 49 step 50 with `Loss=0.4132`.
- Route A reached Epoch 6 step 50 with `Loss=0.7276`.
- Route B completed Epoch 6 with `Loss=1.0419`, `selector_gt_density_loss=0.3198`, `cls_loss=0.4313`, `reg_loss=0.2906`.
- Failure scan remains `0`; no `Training Over`; Route A/B still have no first checkpoint. No cleanup or diagnostic trigger.
- `/data` remains safe with about `1.9T` free.

## Follow-Up Monitor at 08:35

- Remote time: `2026-05-29T08:35:18+08:00`.
- Upload gate still passes: train `200/200`, test `211/211`, `bad_size=0`, `ALL_PASS=True`.
- All five E2E jobs remain `RUNNING`.
- New result: old main fifth eval is `63.74` Avg-mAP, vector `79.55 / 74.15 / 66.21 / 55.97 / 42.80`; trend `63.72 -> 64.15 -> 64.09 -> 64.04 -> 63.74`.
- Interpretation: old main is now `-0.03` vs random-fixed `63.77`, `-0.11` vs strict EMA `63.85`, `-0.90` vs stratified `64.64`, `-1.27` vs old uniform `65.01`, and `-1.35` vs uniform stride-2 `65.09`. This is not severe, but it is not positive old learned-selector evidence.
- Old uniform remains `65.01`; old lowreg remains `63.91`.
- Route A completed Epoch 6 with `Loss=0.7108`.
- Route B reached Epoch 7 step 50 with `Loss=1.0578` and `selector_gt_density_loss=0.3201`.
- Failure scan remains `0`; no `Training Over`; Route A/B still have no first checkpoint. No cleanup or diagnostic trigger.
- `/data` remains safe with about `1.9T` free.

## Follow-Up Monitor at 08:38

- Remote time: `2026-05-29T08:38:35+08:00`.
- Upload gate still passes: train `200/200`, test `211/211`, `bad_size=0`, `ALL_PASS=True`.
- All five E2E jobs remain `RUNNING`.
- No new mAP after old main fifth eval `63.74`; old uniform remains `65.01`; old lowreg remains `63.91`.
- Old main resumed training and reached Epoch 50 step 50 with `Loss=0.4098`.
- Old uniform still has no train line after Epoch 49 final, likely still in eval. Old lowreg still has no train line after Epoch 49 final.
- Route A reached Epoch 7 step 50 with `Loss=0.7462`.
- Route B completed Epoch 7 with `Loss=1.0242`, `selector_gt_density_loss=0.3197`, `cls_loss=0.4078`, `reg_loss=0.2965`.
- Failure scan remains `0`; no `Training Over`; Route A/B still have no first checkpoint. No cleanup or diagnostic trigger.
- `/data` remains safe with about `1.9T` free.

## Follow-Up Monitor at 08:41

- Remote time: `2026-05-29T08:41:49+08:00`.
- Upload gate still passes: train `200/200`, test `211/211`, `bad_size=0`, `ALL_PASS=True`.
- All five E2E jobs remain `RUNNING`.
- No new mAP: old main remains `63.74`, old uniform remains `65.01`, old lowreg remains `63.91`.
- Old main completed Epoch 50 with `Loss=0.4095`.
- Old uniform still has no train line after Epoch 49 final, likely still in eval. Old lowreg still has no train line after Epoch 49 final.
- Route A completed Epoch 7 with `Loss=0.7149`.
- Route B reached Epoch 8 step 50 with `Loss=0.9939`, `selector_gt_density_loss=0.3228`, `cls_loss=0.3839`, `reg_loss=0.2869`.
- Failure scan remains `0`; no `Training Over`; Route A/B still have no first checkpoint. No cleanup or diagnostic trigger.
- `/data` remains safe with about `1.9T` free.

## Follow-Up Monitor at 08:45

- Remote time: `2026-05-29T08:45:13+08:00`.
- Upload gate still passes: train `200/200`, test `211/211`, `bad_size=0`, `ALL_PASS=True`.
- All five E2E jobs remain `RUNNING`.
- New result: old uniform fifth eval is `65.26` Avg-mAP, vector `80.60 / 75.72 / 68.91 / 57.91 / 43.14`; trend `64.12 -> 64.52 -> 64.76 -> 65.01 -> 65.26`.
- Interpretation: old uniform now exceeds the known uniform stride-2 50% reference `65.09` by `+0.17`, so the uniform-control path is strong and not a pipeline failure. This is still control evidence, not learned-selector evidence.
- Old main remains `63.74`, now `-1.52` behind old uniform. Old lowreg remains `63.91`, `-1.35` behind old uniform.
- Route A completed Epoch 8 with `Loss=0.6631`.
- Route B reached Epoch 9 step 50 with `Loss=0.9507`, `selector_gt_density_loss=0.3197`, `cls_loss=0.3543`, `reg_loss=0.2764`.
- Failure scan remains `0`; no `Training Over`; Route A/B still have no first checkpoint. No cleanup or diagnostic trigger.
- `/data` remains safe with about `1.9T` free.

## Follow-Up Monitor at 08:48

- Remote time: `2026-05-29T08:48:30+08:00`.
- Upload gate still passes: train `200/200`, test `211/211`, `bad_size=0`, `ALL_PASS=True`.
- All five E2E jobs remain `RUNNING`.
- No new mAP after old uniform `65.26`; old main remains `63.74`, old lowreg remains `63.91`.
- Old main completed Epoch 51 with `Loss=0.4155`.
- Old uniform resumed training and completed Epoch 50 with `Loss=0.3851`.
- Old lowreg still has no train line after Epoch 49 final, likely still in eval/transition.
- Route A reached Epoch 9 step 50 with `Loss=0.6125`.
- Route B completed Epoch 9 with `Loss=0.9796`, `selector_gt_density_loss=0.3198`, `cls_loss=0.3773`, `reg_loss=0.2823`.
- Failure scan remains `0`; no `Training Over`; Route A/B still have no first checkpoint. No cleanup or diagnostic trigger.
- `/data` remains safe with about `1.9T` free.

## Follow-Up Monitor at 08:52

- Remote time: `2026-05-29T08:52:05+08:00`.
- Upload gate still passes: train `200/200`, test `211/211`, `bad_size=0`, `ALL_PASS=True`.
- All five E2E jobs remain `RUNNING`: old main/uniform/lowreg and Route A/B.
- New result: old lowreg fifth eval is `63.92` Avg-mAP, vector `78.71 / 74.30 / 66.81 / 56.59 / 43.19`; trend `62.98 -> 63.55 -> 63.42 -> 63.91 -> 63.92`.
- Interpretation: lowreg is nearly flat versus its fourth eval and remains only slightly above random-fixed `63.77` and strict EMA `63.85`; it is still below stratified `64.64`, old uniform `65.26`, and uniform stride-2 `65.09`.
- Old main remains latest `63.74`; old uniform remains latest `65.26` and reached Epoch 51 step 50 with `Loss=0.4141`.
- Route A completed Epoch 9 with `Loss=0.6519`, `cls_loss=0.3720`, `reg_loss=0.2799`.
- Route B reached Epoch 10 step 50 with `Loss=0.9825`, `selector_gt_density_loss=0.3189`, `cls_loss=0.3788`, `reg_loss=0.2845`.
- Failure scan remains `0`; no `Training Over`; Route A/B still have no first checkpoint. No cleanup or diagnostic trigger.
- `/data` remains safe with about `1.9T` free.

## Follow-Up Monitor at 08:55

- Remote time: `2026-05-29T08:55:44+08:00`.
- Upload gate still passes: train `200/200`, test `211/211`, `bad_size=0`, `ALL_PASS=True`.
- All five E2E jobs remain `RUNNING`: old main/uniform/lowreg and Route A/B.
- No new mAP after old main `63.74`, old uniform `65.26`, and old lowreg `63.92`.
- Old main has no train line after Epoch 51 final, consistent with eval or transition.
- Old uniform completed Epoch 51 with `Loss=0.4071`; old lowreg resumed training and reached Epoch 50 step 50 with `Loss=0.4004`.
- Route A reached Epoch 10 step 50 with `Loss=0.6688`, `cls_loss=0.3940`, `reg_loss=0.2748`.
- Route B completed Epoch 10 with `Loss=0.9750`, `selector_gt_density_loss=0.3197`, `cls_loss=0.3723`, `reg_loss=0.2828`.
- Failure scan remains `0`; no `Training Over`; Route A/B still have no first checkpoint, so Route B selector diagnostic is not yet runnable.
- `/data` remains safe with about `1.9T` free.

## Follow-Up Monitor at 08:58

- Remote time: `2026-05-29T08:58:29+08:00`.
- Upload gate still passes: train `200/200`, test `211/211`, `bad_size=0`, `ALL_PASS=True`.
- All five E2E jobs remain `RUNNING`: old main/uniform/lowreg and Route A/B.
- No new mAP after old main `63.74`, old uniform `65.26`, and old lowreg `63.92`.
- Old main has no train line after Epoch 51 final, consistent with eval or transition; old uniform has no train line after Epoch 51 final.
- Old lowreg completed Epoch 50 with `Loss=0.4105`.
- Route A completed Epoch 10 with `Loss=0.6596`, `cls_loss=0.3815`, `reg_loss=0.2781`.
- Route B reached Epoch 11 step 50 with `Loss=0.9758`, `selector_gt_density_loss=0.3221`, `cls_loss=0.3685`, `reg_loss=0.2850`.
- Failure scan remains `0`; no `Training Over`; Route A/B still have no first checkpoint, so Route B selector diagnostic is not yet runnable.
- `/data` remains safe with about `1.9T` free.

## Follow-Up Monitor at 09:01

- Remote time: `2026-05-29T09:01:02+08:00`.
- Upload gate still passes: train `200/200`, test `211/211`, `bad_size=0`, `ALL_PASS=True`.
- All five E2E jobs remain `RUNNING`: old main/uniform/lowreg and Route A/B.
- No new mAP after old main `63.74`, old uniform `65.26`, and old lowreg `63.92`.
- Old main has no train line after Epoch 51 final, consistent with eval or transition; old uniform has no train line after Epoch 51 final.
- Old lowreg reached Epoch 51 step 50 with `Loss=0.4235`.
- Route A reached Epoch 11 step 50 with `Loss=0.6437`, `cls_loss=0.3700`, `reg_loss=0.2736`.
- Route B completed Epoch 11 with `Loss=0.9713`, `selector_gt_density_loss=0.3198`, `cls_loss=0.3726`, `reg_loss=0.2786`.
- Failure scan remains `0`; no `Training Over`; Route A/B still have no first checkpoint, so Route B selector diagnostic is not yet runnable.
- `/data` remains safe with about `1.9T` free.

## Follow-Up Monitor at 09:03

- Remote time: `2026-05-29T09:03:56+08:00`.
- Upload gate still passes: train `200/200`, test `211/211`, `bad_size=0`, `ALL_PASS=True`.
- All five E2E jobs remain `RUNNING`: old main/uniform/lowreg and Route A/B.
- New old-main eval line observed at `09:05:37`: `63.90` Avg-mAP, vector `79.32 / 74.98 / 66.28 / 56.33 / 42.57`; trend `63.72 -> 64.15 -> 64.09 -> 64.04 -> 63.74 -> 63.90`.
- Interpretation: old main recovered slightly but is still only `+0.13` vs random-fixed `63.77`, `+0.05` vs strict EMA `63.85`, and remains below stratified `64.64`, old uniform `65.26`, and uniform stride-2 `65.09`.
- Old uniform remains latest `65.26`; old lowreg completed Epoch 51 with `Loss=0.4170`.
- Route A completed Epoch 11 with `Loss=0.6419`, `cls_loss=0.3706`, `reg_loss=0.2713`.
- Route B reached Epoch 12 step 50 with `Loss=0.9359`, `selector_gt_density_loss=0.3197`, `cls_loss=0.3446`, `reg_loss=0.2714`.
- Failure scan remains `0`; no `Training Over`; Route A/B still have no first checkpoint, so Route B selector diagnostic is not yet runnable.
- `/data` remains safe with about `1.9T` free.

## Follow-Up Monitor at 09:06

- Remote time: `2026-05-29T09:06:59+08:00`.
- Upload gate still passes: train `200/200`, test `211/211`, `bad_size=0`, `ALL_PASS=True`.
- All five E2E jobs remain `RUNNING`: old main/uniform/lowreg and Route A/B.
- No new mAP after old main `63.90`, old uniform `65.26`, and old lowreg `63.92`.
- Old main resumed training and reached Epoch 52 step 50 with `Loss=0.4383`; old uniform and lowreg have no new train lines after their latest epoch-final records.
- Route A reached Epoch 12 step 50 with `Loss=0.6113`, `cls_loss=0.3349`, `reg_loss=0.2764`.
- Route B reached Epoch 13 step 50 with `Loss=0.9413`, `selector_gt_density_loss=0.3193`, `cls_loss=0.3462`, `reg_loss=0.2756`.
- Failure scan remains `0`; no `Training Over`; Route A/B still have no first checkpoint, so Route B selector diagnostic is not yet runnable.
- `/data` remains safe with about `1.9T` free.

## Follow-Up Monitor at 09:09

- Remote time: `2026-05-29T09:09:50+08:00`.
- Upload gate still passes: train `200/200`, test `211/211`, `bad_size=0`, `ALL_PASS=True`.
- All five E2E jobs remain `RUNNING`: old main/uniform/lowreg and Route A/B.
- No new mAP after old main `63.90`, old uniform `65.26`, and old lowreg `63.92`.
- Old main completed Epoch 52 with `Loss=0.4303`; old uniform and lowreg have no new train lines after their latest epoch-final records.
- Route A completed Epoch 12 with `Loss=0.6248`, `cls_loss=0.3558`, `reg_loss=0.2690`.
- Route B completed Epoch 13 with `Loss=0.9532`, `selector_gt_density_loss=0.3197`, `cls_loss=0.3518`, `reg_loss=0.2814`.
- Failure scan remains `0`; no `Training Over`; Route A/B still have no first checkpoint, so Route B selector diagnostic is not yet runnable.
- `/data` remains safe with about `1.9T` free.

## Follow-Up Monitor at 09:12

- Remote time: `2026-05-29T09:12:55+08:00`.
- Upload gate still passes: train `200/200`, test `211/211`, `bad_size=0`, `ALL_PASS=True`.
- All five E2E jobs remain `RUNNING`: old main/uniform/lowreg and Route A/B.
- New old-uniform eval at `09:13:03`: `65.52` Avg-mAP, vector `80.77 / 76.08 / 69.15 / 57.91 / 43.70`; trend `64.12 -> 64.52 -> 64.76 -> 65.01 -> 65.26 -> 65.52`.
- Interpretation: old uniform is now `+0.43` above the uniform stride-2 reference `65.09`, so the uniform-control target is strong and Route A/B need to beat a higher bar for positive learned-selector evidence.
- Old main remains latest `63.90`; old lowreg remains latest `63.92`.
- Route A reached Epoch 13 step 50 with `Loss=0.6165`, `cls_loss=0.3489`, `reg_loss=0.2677`.
- Route B reached Epoch 14 step 50 with `Loss=0.9009`, `selector_gt_density_loss=0.3205`, `cls_loss=0.3307`, `reg_loss=0.2495`.
- Failure scan remains `0`; no `Training Over`; Route A/B still have no first checkpoint, so Route B selector diagnostic is not yet runnable.
- `/data` remains safe with about `1.9T` free.

## Follow-Up Monitor at 09:15

- Remote time: `2026-05-29T09:15:59+08:00`.
- Upload gate still passes: train `200/200`, test `211/211`, `bad_size=0`, `ALL_PASS=True`.
- All five E2E jobs remain `RUNNING`: old main/uniform/lowreg and Route A/B.
- No new mAP after old uniform `65.52`, old main `63.90`, and old lowreg `63.92`.
- Old main completed Epoch 53 with `Loss=0.3890`; old uniform resumed training and reached Epoch 52 step 50 with `Loss=0.4180`; old lowreg has no new train line after Epoch 51 final.
- Route A completed Epoch 13 with `Loss=0.6351`, `cls_loss=0.3602`, `reg_loss=0.2750`.
- Route B completed Epoch 14 with `Loss=0.9331`, `selector_gt_density_loss=0.3198`, `cls_loss=0.3527`, `reg_loss=0.2604`.
- Failure scan remains `0`; no `Training Over`; Route A/B still have no first checkpoint, so Route B selector diagnostic is not yet runnable.
- `/data` remains safe with about `1.9T` free.

## Follow-Up Monitor at 09:19

- Remote time: `2026-05-29T09:19:11+08:00`.
- Upload gate still passes: train `200/200`, test `211/211`, `bad_size=0`, `ALL_PASS=True`.
- All five E2E jobs remain `RUNNING`: old main/uniform/lowreg and Route A/B.
- No new mAP after old uniform `65.52`, old main `63.90`, and old lowreg `63.92`.
- Old uniform completed Epoch 52 with `Loss=0.4159`; old main and lowreg have no new train lines after their latest epoch-final records.
- Route A reached Epoch 14 step 50 with `Loss=0.5793`, `cls_loss=0.3241`, `reg_loss=0.2552`.
- Route B reached Epoch 15 step 50 with `Loss=0.9344`, `selector_gt_density_loss=0.3189`, `cls_loss=0.3508`, `reg_loss=0.2644`.
- Failure scan remains `0`; no `Training Over`; Route A/B still have no first checkpoint, so Route B selector diagnostic is not yet runnable.
- `/data` remains safe with about `1.9T` free.

## Follow-Up Monitor at 09:24

- Remote time: `2026-05-29T09:24:46+08:00`.
- Upload gate still passes: train `200/200`, test `211/211`, `bad_size=0`, `ALL_PASS=True`.
- All five E2E jobs remain `RUNNING`: Route A/B on `g0009` for `1:35:01`, old main/uniform on `g0005` for `6:50:31`, and old lowreg on `g0024` for `6:50:31`.
- New lowreg sixth eval at `09:20:33`: `63.43` Avg-mAP, vector `78.50 / 74.15 / 66.15 / 55.81 / 42.55`; trend `62.98 -> 63.55 -> 63.42 -> 63.91 -> 63.92 -> 63.43`.
- Interpretation: lowreg is now below random-fixed `63.77` and strict EMA `63.85`, far below stratified `64.64`, uniform stride-2 `65.09`, and old uniform `65.52`. This is negative evidence for lowreg, but not a severe collapse.
- Old main remains latest `63.90`; old uniform remains latest `65.52` and completed Epoch 53 with `Loss=0.3695`; old lowreg resumed training and reached Epoch 52 step 50 with `Loss=0.4313`.
- Route A reached Epoch 15 step 50 with `Loss=0.6054`, `cls_loss=0.3489`, `reg_loss=0.2566`.
- Route B reached Epoch 16 step 50 with `Loss=0.8972`, `selector_gt_density_loss=0.3189`, `cls_loss=0.3272`, `reg_loss=0.2507`.
- Failure scan remains `0`; no `Training Over`; Route A/B still have no first checkpoint, so Route B selector diagnostic is not yet runnable.
- `/data` remains safe with about `1.9T` free.
- Decision: continue all active jobs, do not launch additional lowreg sweeps, and recheck soon for Route A/B `epoch_19.pth`.

## Follow-Up Monitor at 09:27

- Remote time: `2026-05-29T09:27:01+08:00`.
- All five jobs remain `RUNNING`: Route A/B on `g0009` for `1:37:16`, old main/uniform on `g0005` for `6:52:46`, and old lowreg on `g0024` for `6:52:46`.
- No new mAP after old main `63.90`, old uniform `65.52`, and lowreg `63.43`.
- Route A completed Epoch 15 with `Loss=0.5963`, `cls_loss=0.3392`, `reg_loss=0.2571`.
- Route B completed Epoch 16 with `Loss=0.9361`, `selector_gt_density_loss=0.3197`, `cls_loss=0.3477`, `reg_loss=0.2684`.
- Old lowreg completed Epoch 52 with `Loss=0.4320`; old uniform remains Epoch 53 final with `Loss=0.3695`; old main remains Epoch 53 final with `Loss=0.3890`.
- Failure scan remains `0`; no `Training Over`; Route A/B checkpoint listing is still empty, so Route B selector diagnostic is not yet runnable.
- `/data` remains safe with about `1.9T` free.
- Decision: continue all active jobs, no cleanup or new launch.

## Follow-Up Monitor at 09:29

- Remote time: `2026-05-29T09:29:40+08:00`.
- All five jobs remain `RUNNING`: Route A/B on `g0009` for `1:39:55`, old jobs still running on `g0005/g0024`.
- No new mAP after old main `63.90`, old uniform `65.52`, and lowreg `63.43`.
- Route A reached Epoch 16 step 50 with `Loss=0.5836`, `cls_loss=0.3321`, `reg_loss=0.2515`.
- Route B reached Epoch 17 step 50 with `Loss=0.8532`, `selector_gt_density_loss=0.3230`, `cls_loss=0.2800`, `reg_loss=0.2499`.
- Failure scan remains `0`; Route A/B checkpoint listing is still empty.
- Decision: continue active jobs and keep waiting for Route A/B `epoch_19.pth`.

## Follow-Up Monitor at 09:30

- Remote time: `2026-05-29T09:30:47+08:00`.
- Route A/B remain `RUNNING` on `g0009` for `1:41:02`.
- No new train line beyond Route A Epoch 16 step 50 at `09:28:54` and Route B Epoch 17 step 50 at `09:28:42`.
- Failure scan remains `0`; Route A/B checkpoint listing is still empty.
- Decision: continue Route A/B; no diagnostic or cleanup until `epoch_19.pth` appears or a failure/result trigger occurs.

## Follow-Up Monitor at 09:32

- Remote time: `2026-05-29T09:32:58+08:00`.
- Upload gate still passes: train `200/200`, test `211/211`, `bad_size=0`, `ALL_PASS=True`.
- All five jobs remain `RUNNING`: Route A/B on `g0009` for `1:43:13`, old main/uniform on `g0005` for `6:58:43`, and old lowreg on `g0024` for `6:58:43`.
- New old-main eval observed at log time `09:34:09`: `64.36` Avg-mAP, vector `79.52 / 74.52 / 67.26 / 57.02 / 43.46`; trend `63.72 -> 64.15 -> 64.09 -> 64.04 -> 63.74 -> 63.90 -> 64.36`.
- Interpretation: old main is now `+0.59` vs random-fixed `63.77` and `+0.51` vs strict EMA `63.85`, but still below stratified `64.64`, uniform stride-2 `65.09`, and old uniform `65.52`; this is useful evidence but not a learned-selector win.
- Old uniform remains latest `65.52`; lowreg remains latest `63.43` and completed Epoch 53 with `Loss=0.3942`.
- Route A completed Epoch 16 with `Loss=0.6127`, `cls_loss=0.3469`, `reg_loss=0.2658`.
- Route B completed Epoch 17 with `Loss=0.8783`, `selector_gt_density_loss=0.3197`, `cls_loss=0.3057`, `reg_loss=0.2527`.
- Failure scan remains `0`; no `Training Over`; Route A/B still have no first checkpoint.
- `/data` remains safe with about `1.9T` free.
- Decision: continue all jobs, with Route A/B `epoch_19.pth` as the next selector-diagnostic trigger.

## Follow-Up Monitor at 09:34

- Remote time: `2026-05-29T09:34:39+08:00`.
- Route A/B remain `RUNNING` on `g0009` for `1:44:54`.
- Route A has no new train line after Epoch 16 final at `09:31:48`, still within a normal short progress window.
- Route B reached Epoch 18 step 50 with `Loss=0.9083`, `selector_gt_density_loss=0.3205`, `cls_loss=0.3226`, `reg_loss=0.2650`.
- Failure scan remains `0`; Route A/B checkpoint listing is still empty.
- Decision: continue Route A/B; no diagnostic or cleanup until `epoch_19.pth` appears or a failure/result trigger occurs.

## Follow-Up Monitor at 09:35

- Remote time: `2026-05-29T09:35:53+08:00`.
- All five jobs remain `RUNNING`: Route A/B on `g0009` for `1:46:08`, old main/uniform on `g0005` for `7:01:38`, and old lowreg on `g0024` for `7:01:38`.
- No new mAP after old main `64.36`, old uniform `65.52`, and lowreg `63.43`.
- Route A reached Epoch 17 step 50 with `Loss=0.5150`, `cls_loss=0.2743`, `reg_loss=0.2407`.
- Route B remains latest Epoch 18 step 50 with `Loss=0.9083`, `selector_gt_density_loss=0.3205`.
- Old main resumed training after eval and reached Epoch 54 step 50 with `Loss=0.4547`; old uniform remains latest Epoch 53 final; old lowreg remains latest Epoch 53 final.
- Failure scan remains `0`; no `Training Over`; Route A/B checkpoint listing is still empty.
- Decision: continue all jobs. Route A/B `epoch_19.pth` remains the next trigger.

## Follow-Up Monitor at 09:37

- Remote time: `2026-05-29T09:37:57+08:00`.
- Upload gate still passes: train `200/200`, test `211/211`, `bad_size=0`, `ALL_PASS=True`.
- All five jobs remain `RUNNING`: Route A/B on `g0009` for `1:48:12`, old main/uniform on `g0005` for `7:03:42`, and old lowreg on `g0024` for `7:03:42`.
- No new mAP after old main `64.36`, old uniform `65.52`, and lowreg `63.43`.
- Old main completed Epoch 54 with `Loss=0.4309`; old uniform remains latest Epoch 53 final; old lowreg remains latest Epoch 53 final.
- Route A completed Epoch 17 with `Loss=0.5486`, `cls_loss=0.3008`, `reg_loss=0.2478`.
- Route B completed Epoch 18 with `Loss=0.8982`, `selector_gt_density_loss=0.3197`, `cls_loss=0.3148`, `reg_loss=0.2634`.
- Failure scan remains `0`; no `Training Over`; Route A/B checkpoint listing is still empty.
- `/data` remains safe with about `1.9T` free.
- Decision: continue all jobs and recheck soon for Route A/B `epoch_19.pth`.

## Follow-Up Monitor at 09:39

- Remote time: `2026-05-29T09:39:26+08:00`.
- Route A/B remain `RUNNING` on `g0009` for `1:49:41`.
- Route A remains latest Epoch 17 final with `Loss=0.5486`.
- Route B reached Epoch 19 step 50 with `Loss=0.9095`, `selector_gt_density_loss=0.3223`, `cls_loss=0.3186`, `reg_loss=0.2684`.
- Failure scan remains `0`; Route A/B checkpoint listing is still empty because Epoch 19 has not completed.
- Decision: continue Route A/B and recheck after Epoch 19 final for `epoch_19.pth`.

## Follow-Up Monitor at 09:40

- Remote time: `2026-05-29T09:40:35+08:00`.
- Route A reached Epoch 18 step 50 with `Loss=0.5648`, `cls_loss=0.3018`, `reg_loss=0.2631`.
- Route B remains latest Epoch 19 step 50 with `Loss=0.9095`, `selector_gt_density_loss=0.3223`.
- Failure scan remains `0`; Route A/B checkpoint listing is still empty because Epoch 19 final/checkpoint has not been reached.
- Decision: continue Route A/B and recheck for `epoch_19.pth`.

## Follow-Up Monitor at 09:41

- Remote time: `2026-05-29T09:41:42+08:00`.
- Route A remains latest Epoch 18 step 50 with `Loss=0.5648`.
- Route B remains latest Epoch 19 step 50 with `Loss=0.9095`, `selector_gt_density_loss=0.3223`.
- Failure scan remains `0`; Route A/B checkpoint listing remains empty because Epoch 19 final/checkpoint has not been reached.
- Decision: continue Route A/B. No diagnostic, cleanup, or new launch; next action is recheck for `epoch_19.pth`.

## Follow-Up Monitor at 09:43

- Remote time: `2026-05-29T09:43:28+08:00`.
- Upload gate still passes: train `200/200`, test `211/211`, `bad_size=0`, `ALL_PASS=True`.
- All five training jobs remain `RUNNING`.
- Route B completed Epoch 19 with `Loss=0.8936`, `selector_gt_density_loss=0.3198`, and wrote `checkpoint/epoch_19.pth` size `623905307`, mtime `09:42:03`.
- Route A completed Epoch 18 with `Loss=0.5601`; its checkpoint had not appeared in this monitor.
- Old uniform produced a new eval at `09:42:05`: `65.50` Avg-mAP, effectively tied with prior `65.52`.
- Old main remained latest `64.36`; old lowreg remained latest `63.43`.
- Failure scan remains `0`; `/data` remains safe with about `1.9T` free.
- Decision: trigger Route B selector diagnostic; continue training jobs.

## Route B Epoch-19 Selector Diagnostic

- First login-node direct diagnostic was terminated before summary output, likely due login-node resource limits.
- Slurm diagnostic job `994442 e2e_bdiag` failed due missing `PYTHONPATH`.
- Corrected Slurm diagnostic job `994443 e2e_bdiag2` completed on `g0024` in `00:02:33`.
- Output: `~/run/yuzibo/e2e_runs/selector_diagnostics/e2e_routeB_gtaux_epoch19_posdiag_20260529_095333/summary.json`.
- Log: `~/run/yuzibo/OpenTAD_BATA_Clean/logs/e2e_bdiag2-994443.out`.
- Scope: selector-only diagnostic over 48/487 validation windows. It did not run ViT/backbone/detector and did not alter active training jobs.
- GT use: validation GT was used only for offline action/boundary proximity labels; no train/test protocol leakage.

Metrics:

| Metric | Route B epoch 19 |
| --- | ---: |
| Mean abs delta vs uniform | `1.347` |
| P95 abs delta vs uniform | `2.000` |
| Rounded Jaccard vs uniform | `0.211` |
| Selected action fraction | `0.2879` |
| Selected boundary<=4 fraction | `0.1483` |
| Boundary recall@4 | `1.000` |
| Logit std | `0.000714` |
| Final score-head weight norm | `0.0101` |

Pro gates:

| Gate | Result |
| --- | --- |
| logit std `>=0.01` | FAIL |
| mean abs delta `2-8` | FAIL |
| selected action fraction `>=0.32` | FAIL |
| selected boundary<=4 fraction `>=0.17` | FAIL |

Interpretation: Route B has not learned the desired action/boundary-aware selected-frame distribution at epoch 19. The GT-density auxiliary loss is active, but the selector distribution remains close to uniform coverage.

## Follow-Up Monitor at 09:57

- Remote time: `2026-05-29T09:57:38+08:00`.
- All five training jobs remain `RUNNING`.
- Route A wrote `epoch_19.pth` size `623852123`, mtime `09:49:21`, and reached Epoch 20 final with `Loss=0.5450`.
- Route B reached Epoch 22 step 50 with `Loss=0.8643`, `selector_gt_density_loss=0.3189`.
- Old lowreg produced a new eval at `09:51:00`: `63.87` Avg-mAP, still weak/near baseline and below old main `64.36`, stratified `64.64`, and uniform `65.50/65.52`.
- Old uniform latest `65.50`; old main latest `64.36`.
- Failure scan remains `0`; no `Training Over`.
- Decision: continue all jobs to first Route A/B eval and old final checkpoints. No cleanup while active; no new experiment launch from this diagnostic alone.

## Follow-Up Monitor at 10:01

- Remote time: `2026-05-29T10:01:01+08:00`.
- Upload gate still passes: train `200/200`, test `211/211`, `bad_size=0`, `ALL_PASS=True`.
- Slurm and sacct report all five training jobs `RUNNING`: Route A/B on `g0009` for `2:11:16`, old main/uniform on `g0005`, and old lowreg on `g0024`.
- Old main produced a new eval at `10:02:36`: `64.33` Avg-mAP, vector `79.46 / 74.26 / 67.68 / 57.34 / 42.90`; this is flat vs prior `64.36` and remains below stratified `64.64` and old uniform `65.50/65.52`.
- Old uniform latest remains `65.50`; old lowreg latest remains `63.87`.
- Route A reached Epoch 21 final with `Loss=0.5465`.
- Route B reached Epoch 23 step 50 with `Loss=0.8545`, `selector_gt_density_loss=0.3214`.
- Failure scan remains `0`; no `Training Over`.
- Checkpoints visible: old runs `epoch_19/39`, Route A/B `epoch_19`.
- `/data` remains safe with about `1.9T` free.
- Decision: continue all jobs. No cleanup because all runs are active. No severe-result gate and no new launch until Route A/B first eval or a completed-run cleanup trigger.

## Follow-Up Monitor at 10:05

- Remote time: `2026-05-29T10:05:42+08:00`.
- Upload gate still passes: train `200/200`, test `211/211`, `bad_size=0`, `ALL_PASS=True`.
- Slurm and sacct report all five jobs `RUNNING`: old main/uniform/lowreg elapsed about `7:31:27`, Route A/B elapsed about `2:15:57`.
- Old main latest eval remains `64.33` Avg-mAP at `10:02:36`, vector `79.46 / 74.26 / 67.68 / 57.34 / 42.90`; it is flat vs `64.36` and below old uniform.
- Old uniform latest eval remains `65.50` with best `65.52`; old lowreg latest remains `63.87`.
- Route A reached Epoch 22 step 50 at `10:03:56` with `Loss=0.5417`, `cls_loss=0.2941`, `reg_loss=0.2477`; no mAP yet.
- Route B completed Epoch 23 at `10:03:38` with `Loss=0.8338`, `selector_gt_density_loss=0.3198`, `cls_loss=0.2820`, `reg_loss=0.2317`; no mAP yet.
- Failure scan remains `0` for all five logs; no `Training Over`.
- Visible checkpoints remain old runs `epoch_19/39` plus Route A/B `epoch_19`. No active checkpoint cleanup was performed.
- `/data` remains safe with about `1.9T` free.
- Decision: continue all five jobs. No severe-result gate, no new launch, and no Route B variant from the epoch-19 diagnostic alone. Next trigger is Route A/B first mAP or completed-run cleanup if an old job finishes.

## Follow-Up Monitor at 10:10

- Remote time: `2026-05-29T10:10:54+08:00`.
- Upload gate still passes: train `200/200`, test `211/211`, `bad_size=0`, `ALL_PASS=True`.
- Slurm and sacct report all five jobs `RUNNING`: old main/uniform/lowreg elapsed about `7:36:39`, Route A/B elapsed about `2:21:09`.
- Old uniform produced a new best at `10:11:10`: `65.56` Avg-mAP, vector `80.68 / 75.90 / 69.25 / 58.15 / 43.84`.
- Old main latest remains `64.33`; old lowreg latest remains `63.87`. Old main resumed training and reached Epoch 57 step 50 with `Loss=0.3970`.
- Route A reached Epoch 23 step 50 at `10:09:46` with `Loss=0.5125`, `cls_loss=0.2760`, `reg_loss=0.2365`; no mAP yet.
- Route B completed Epoch 24 at `10:09:02` with `Loss=0.8509`, `selector_gt_density_loss=0.3197`, `cls_loss=0.2856`, `reg_loss=0.2453`; no mAP yet.
- Failure scan remains `0` for all five logs; no `Training Over`.
- Visible checkpoints remain old runs `epoch_19/39` plus Route A/B `epoch_19`. No active checkpoint cleanup was performed.
- `/data` remains safe with about `1.9T` free.
- Decision: continue all five jobs. The uniform-control bar is now `65.56`, so Route A/B and any learned-selector claim must be judged against that stronger control. No severe-result gate, no new launch, and no Route B variant from the epoch-19 diagnostic alone.

## Follow-Up Monitor at 10:15

- Remote time: `2026-05-29T10:15:03+08:00`.
- Upload gate still passes: train `200/200`, test `211/211`, `bad_size=0`, `ALL_PASS=True`.
- Slurm and sacct report all five jobs `RUNNING`: old main/uniform/lowreg elapsed about `7:40:48`, Route A/B elapsed about `2:25:18`.
- No new mAP after old uniform best `65.56`, old main `64.33`, and old lowreg `63.87`.
- Old main completed Epoch 57 at `10:13:26` with `Loss=0.4087`; old uniform completed Epoch 56 at `10:16:42` with `Loss=0.3935`; old lowreg has no new train line after Epoch 55 final, likely in eval/transition.
- Route A completed Epoch 23 at `10:12:31` with `Loss=0.5068`, `cls_loss=0.2784`, `reg_loss=0.2284`; no mAP yet.
- Route B completed Epoch 25 at `10:14:29` with `Loss=0.8652`, `selector_gt_density_loss=0.3198`, `cls_loss=0.2976`, `reg_loss=0.2476`; no mAP yet.
- Failure scan remains `0` for all five logs; no `Training Over`.
- Visible checkpoints remain old runs `epoch_19/39` plus Route A/B `epoch_19`. No active checkpoint cleanup was performed.
- `/data` remains safe with about `1.9T` free.
- Decision: continue all five jobs. No severe-result gate, no cleanup, and no new launch. Next trigger is Route A/B first mAP, old lowreg next eval, or completed-run cleanup if an old job finishes.

## Follow-Up Monitor at 10:19

- Remote time: `2026-05-29T10:19:11+08:00`.
- Upload gate still passes: train `200/200`, test `211/211`, `bad_size=0`, `ALL_PASS=True`.
- Slurm and sacct report all five jobs `RUNNING`: old main/uniform/lowreg elapsed about `7:44:56`, Route A/B elapsed about `2:29:26`.
- No new mAP after old uniform best `65.56`, old main `64.33`, and old lowreg `63.87`.
- Old main latest train remains Epoch 57 final `Loss=0.4087`; old uniform reached Epoch 57 step 50 at `10:19:30` with `Loss=0.3709`; old lowreg still has no new train line after Epoch 55 final, consistent with eval/transition.
- Route A completed Epoch 24 at `10:18:20` with `Loss=0.5073`, `cls_loss=0.2702`, `reg_loss=0.2371`; no mAP yet.
- Route B reached Epoch 26 step 50 at `10:17:19` with `Loss=0.8099`, `selector_gt_density_loss=0.3209`, `cls_loss=0.2504`, `reg_loss=0.2383`; no mAP yet.
- Failure scan remains `0` for all five logs; no `Training Over`.
- Visible checkpoints remain old runs `epoch_19/39` plus Route A/B `epoch_19`. No active checkpoint cleanup was performed.
- `/data` remains safe with about `1.9T` free.
- Decision: continue all five jobs. No severe-result gate, no cleanup, and no new launch. Next trigger is Route A/B first mAP, old lowreg next eval, or completed-run cleanup if an old job finishes.

## Follow-Up Monitor at 10:23

- Remote time: `2026-05-29T10:23:17+08:00`.
- Upload gate still passes: train `200/200`, test `211/211`, `bad_size=0`, `ALL_PASS=True`.
- Slurm and sacct report all five jobs `RUNNING`: old main/uniform/lowreg elapsed about `7:49:02`, Route A/B elapsed about `2:33:32`.
- Old lowreg produced a new eval at `10:21:47`: `64.36` Avg-mAP, vector `79.55 / 74.89 / 66.82 / 57.16 / 43.40`.
- Interpretation: lowreg recovered above random-fixed `63.77` and strict EMA `63.85`, but remains below stratified `64.64` and old uniform `65.56`; this is not a positive learned-selector result.
- Old main latest remains `64.33`; old uniform best remains `65.56`.
- Route A reached Epoch 25 step 50 at `10:21:21` with `Loss=0.5161`, `cls_loss=0.2867`, `reg_loss=0.2294`; no mAP yet.
- Route B reached Epoch 27 step 50 at `10:22:49` with `Loss=0.8348`, `selector_gt_density_loss=0.3173`, `cls_loss=0.2647`, `reg_loss=0.2526`; no mAP yet.
- Failure scan remains `0` for all five logs; no `Training Over`.
- Visible checkpoints remain old runs `epoch_19/39` plus Route A/B `epoch_19`. No active checkpoint cleanup was performed.
- `/data` remains safe with about `1.9T` free.
- Decision: continue all five jobs. No severe-result gate, no cleanup, and no new launch. Next trigger is Route A/B first mAP or completed-run cleanup if an old job finishes.

## Follow-Up Monitor at 10:27

- Remote time: `2026-05-29T10:27:02+08:00`.
- Upload gate still passes: train `200/200`, test `211/211`, `bad_size=0`, `ALL_PASS=True`.
- Slurm and sacct report all five jobs `RUNNING`: old main/uniform/lowreg elapsed about `7:52:48`, Route A/B elapsed about `2:37:18`.
- No new mAP after old lowreg `64.36`, old uniform best `65.56`, and old main `64.33`.
- Old lowreg resumed training to Epoch 56 step 50 at `10:24:43` with `Loss=0.3914`; old uniform completed Epoch 57 at `10:22:12` with `Loss=0.3915`; old main latest train remains Epoch 57 final.
- Route A reached Epoch 26 step 50 at `10:27:12` with `Loss=0.4908`, `cls_loss=0.2543`, `reg_loss=0.2366`; no mAP yet.
- Route B completed Epoch 27 at `10:25:22` with `Loss=0.8539`, `selector_gt_density_loss=0.3197`, `cls_loss=0.2825`, `reg_loss=0.2514`; no mAP yet.
- Failure scan remains `0` for all five logs; no `Training Over`.
- Visible checkpoints remain old runs `epoch_19/39` plus Route A/B `epoch_19`. No active checkpoint cleanup was performed.
- `/data` remains safe with about `1.9T` free.
- Decision: continue all five jobs. No severe-result gate, no cleanup, and no new launch. Next trigger is Route A/B first mAP or completed-run cleanup if an old job finishes.

## Follow-Up Monitor at 10:34

- Remote time: `2026-05-29T10:34:14+08:00`.
- Upload gate still passes: train `200/200`, test `211/211`, `bad_size=0`, `ALL_PASS=True`.
- Slurm and sacct report all five jobs `RUNNING`: old main/uniform/lowreg elapsed about `8:00:00`, Route A/B elapsed about `2:44:30`.
- Old main produced a new eval at `10:31:07`: `64.71` Avg-mAP, vector `79.80 / 75.33 / 67.95 / 57.69 / 42.77`.
- Interpretation: old main is now above random-fixed `63.77`, strict EMA `63.85`, and stratified `64.64`, but remains below old uniform best `65.56` by `0.85`; this is useful progress but still not a learned-selector win against uniform coverage.
- Old uniform best remains `65.56`; old lowreg latest remains `64.36`.
- Old main resumed to Epoch 58 step 50 with `Loss=0.4189`; old lowreg completed Epoch 57 with `Loss=0.4144`.
- Route A reached Epoch 27 step 50 with `Loss=0.5003`, `cls_loss=0.2576`, `reg_loss=0.2427`; no mAP yet.
- Route B reached Epoch 29 step 50 with `Loss=0.8148`, `selector_gt_density_loss=0.3221`, `cls_loss=0.2580`, `reg_loss=0.2344`; no mAP yet.
- Failure scan remains `0` for all five logs; no `Training Over`.
- Visible checkpoints remain old runs `epoch_19/39` plus Route A/B `epoch_19`. No active checkpoint cleanup was performed.
- `/data` remains safe with about `1.9T` free.
- Decision: continue all five jobs. No severe-result gate, no cleanup, and no new launch. Next trigger is Route A/B first mAP or completed-run cleanup if an old job finishes.

## Follow-Up Monitor at 10:38

- Remote time at monitor start: `2026-05-29T10:38:34+08:00`.
- Upload gate still passes: train `200/200`, test `211/211`, `bad_size=0`, `ALL_PASS=True`.
- Slurm and sacct report all five jobs `RUNNING`: old main/uniform/lowreg elapsed about `8:04:19`, Route A/B elapsed about `2:48:49`.
- Old uniform produced a new eval at log time `10:40:13`: `65.73` Avg-mAP, vector `80.71 / 76.08 / 69.50 / 58.35 / 43.98`.
- Interpretation: old uniform is now the strongest current control and sits `+0.64` above the historical uniform stride-2 `65.09`; learned-selector claims must now be judged against `65.73`.
- Old main latest remains `64.71`, now `1.02` below old uniform. Old lowreg latest remains `64.36`.
- Old main reached Epoch 59 step 50 with `Loss=0.4148`; old lowreg remains latest Epoch 57 final with `Loss=0.4144`.
- Route A completed Epoch 27 with `Loss=0.5233`, `cls_loss=0.2795`, `reg_loss=0.2438`; no mAP yet.
- Route B completed Epoch 29 with `Loss=0.8265`, `selector_gt_density_loss=0.3197`, `cls_loss=0.2720`, `reg_loss=0.2345`; no mAP yet.
- Failure scan remains `0` for all five logs; no `Training Over`.
- Visible checkpoints remain old runs `epoch_19/39` plus Route A/B `epoch_19`. No active checkpoint cleanup was performed.
- `/data` remains safe with about `1.9T` free.
- Decision: continue all five jobs. No severe-result gate, no cleanup, and no new launch. Next trigger is Route A/B first mAP or completed-run cleanup if an old job finishes.

## Follow-Up Monitor at 10:42

- Remote time: `2026-05-29T10:42:07+08:00`.
- Slurm reports all five jobs `RUNNING`: old main/uniform/lowreg elapsed about `8:07:52`, Route A/B elapsed about `2:52:22`.
- Old main completed Epoch 59 at `10:41:54` with `Loss=0.4045` and wrote `/data/home/sczc063/run/yuzibo/e2e_runs/exps/e2e_rawdensel_main_20260529_023345/gpu1_id0/checkpoint/epoch_59.pth` at `10:41:56`.
- Cleanup decision: no cleanup yet, because old main still has no `Training Over` line and Slurm still reports `994340` as `RUNNING`.
- Old uniform best remains `65.73` and resumed to Epoch 58 step 50 at log time `10:43:00`; old lowreg latest remains `64.36`.
- Route A completed Epoch 28 with `Loss=0.5069`, `cls_loss=0.2716`, `reg_loss=0.2354`; no mAP yet.
- Route B completed Epoch 30 with `Loss=0.8191`, `selector_gt_density_loss=0.3198`, `cls_loss=0.2618`, `reg_loss=0.2372`; no mAP yet.
- Failure scan remains `0` for all five logs.
- `/data` remains safe with about `1.9T` free.
- Decision: continue watching for old main `Training Over`/final eval before checkpoint cleanup. Continue Route A/B to first mAP. No severe-result gate and no new launch.

## Follow-Up Monitor at 10:46

- Remote time: `2026-05-29T10:46:14+08:00`.
- Upload gate still passes: train `200/200`, test `211/211`, `bad_size=0`, `ALL_PASS=True`.
- Slurm and sacct report all five jobs `RUNNING`: old main/uniform/lowreg elapsed about `8:11:59`, Route A/B elapsed about `2:56:29`.
- No `Training Over` line appears in any checked log. Old main still has `epoch_59.pth`, but cleanup remains disallowed while `994340` is running and completion is unconfirmed.
- Old main latest mAP remains `64.71`; old uniform best remains `65.73` and completed Epoch 58 at log time `10:45:41` with `Loss=0.3935`; old lowreg latest remains `64.36`.
- Route A reached Epoch 29 step 50 with `Loss=0.4781`, `cls_loss=0.2487`, `reg_loss=0.2295`; no mAP yet.
- Route B reached Epoch 31 step 50 with `Loss=0.7809`, `selector_gt_density_loss=0.3185`, `cls_loss=0.2445`, `reg_loss=0.2178`; no mAP yet.
- Failure scan remains `0` for all five logs.
- `/data` remains safe with about `1.9T` free.
- Decision: continue watching for old main `Training Over`/final eval before checkpoint cleanup. Continue Route A/B to first mAP. No severe-result gate and no new launch.

## Follow-Up Monitor at 10:50

- Remote time: `2026-05-29T10:50:12+08:00`.
- Upload gate still passes: train `200/200`, test `211/211`, `bad_size=0`, `ALL_PASS=True`.
- Slurm and sacct report all five jobs `RUNNING`: old main/uniform/lowreg elapsed about `8:15:57`, Route A/B elapsed about `3:00:27`.
- No `Training Over` line appears in any checked log.
- Old main latest mAP remains `64.71`; old main `epoch_59.pth` remains present.
- Old uniform best remains `65.73`; old uniform completed Epoch 59 at log time `10:51:14` and wrote `epoch_59.pth` at `10:51:16`.
- Cleanup decision: no cleanup for old main or old uniform because both Slurm jobs remain `RUNNING` and completion is unconfirmed.
- Old lowreg latest remains `64.36`.
- Route A completed Epoch 29 with `Loss=0.4856`, `cls_loss=0.2590`, `reg_loss=0.2266`; no mAP yet.
- Route B reached Epoch 32 step 50 with `Loss=0.8209`, `selector_gt_density_loss=0.3189`, `cls_loss=0.2640`, `reg_loss=0.2377`; no mAP yet.
- Failure scan remains `0` for all five logs.
- `/data` remains safe with about `1.9T` free.
- Decision: continue watching for old main/uniform `Training Over` or final eval before checkpoint cleanup. Continue Route A/B to first mAP. No severe-result gate and no new launch.

## Follow-Up Monitor at 10:57

- Remote time: `2026-05-29T10:57:06+08:00`.
- Upload gate still passes: train `200/200`, test `211/211`, `bad_size=0`, `ALL_PASS=True`.
- Slurm reports all five jobs `RUNNING`: old main/uniform/lowreg elapsed about `8:22:51`, Route A/B elapsed about `3:07:21`.
- No `Training Over` line appears in any checked log.
- Old main latest mAP remains `64.71`, vector `79.80 / 75.33 / 67.95 / 57.69 / 42.77`.
- Old uniform best remains `65.73`, vector `80.71 / 76.08 / 69.50 / 58.35 / 43.98`; this remains the strongest current control.
- Old lowreg produced a new eval at `10:52:06`: `64.28`, vector `79.34 / 74.26 / 67.09 / 57.38 / 43.33`; its best remains `64.36`, and the latest value is still below stratified `64.64` and uniform `65.73`.
- Route A reached Epoch 31 step 50 at `10:56:12` with `Loss=0.4436`, `cls_loss=0.2302`, `reg_loss=0.2134`; no mAP yet.
- Route B reached Epoch 33 step 50 at `10:55:23` with `Loss=0.8218`, `selector_gt_density_loss=0.3202`, `cls_loss=0.2687`, `reg_loss=0.2328`; no mAP yet.
- Failure scan remains `0` for all five logs.
- `/data` remains safe with about `1.9T` free.
- Decision: continue all five jobs. No cleanup because completion is unconfirmed for old main/uniform and all jobs remain active. No severe-result gate and no new launch; wait for Route A/B first mAP or confirmed completed-run cleanup trigger.

## Old Main Completion and Cleanup at 11:00

- Remote completion check time: `2026-05-29T10:59:40+08:00`; cleanup time: `2026-05-29T11:00:35+08:00`.
- Old main job `994340` is no longer in `squeue`; log reports `Training Over...` at `10:59:37`.
- Final old main eval: `64.15` Avg-mAP, vector `79.52 / 75.54 / 66.79 / 56.61 / 42.27`.
- Interpretation: the best observed old main remains `64.71`; final checkpoint is lower by `0.56`. This remains above random-fixed `63.77` and strict EMA `63.85`, but below stratified `64.64` and old uniform `65.73`.
- Cleanup scope: `~/run/yuzibo/e2e_runs/exps/e2e_rawdensel_main_20260529_023345/gpu1_id0/checkpoint`, resolved to `/data/run01/sczc063/yuzibo/e2e_runs/exps/e2e_rawdensel_main_20260529_023345/gpu1_id0/checkpoint`.
- The first cleanup attempt refused the symlink-resolved path before deletion; after verifying the resolved path is still inside `~/run/yuzibo`, cleanup proceeded.
- Disk before cleanup: `/data` `2.3T` size, `420G` used, `1.9T` available, `18%`.
- Removed only completed-run epoch files `epoch_19.pth` and `epoch_39.pth`; kept `epoch_59.pth`.
- Disk after cleanup: `/data` `2.3T` size, `420G` used, `1.9T` available, `18%`.
- Active jobs after cleanup: old uniform `994341`, old lowreg `994342`, Route A `994378`, Route B `994379`.
- Decision: no severe-result gate and no new launch. Continue active jobs; next cleanup trigger is old uniform/lowreg `Training Over`, and next experimental trigger is Route A/B first mAP.

## Follow-Up Monitor at 11:03

- Remote time: `2026-05-29T11:03:18+08:00`.
- Upload gate still passes: train `200/200`, test `211/211`, `bad_size=0`, `ALL_PASS=True`.
- `sacct` confirms old main `994340` completed in `08:23:12`; old main checkpoint listing verifies only `epoch_59.pth` remains after cleanup.
- Active jobs: old uniform `994341`, old lowreg `994342`, Route A `994378`, Route B `994379`.
- Old uniform still has no `Training Over`; latest/best remains `65.73`, vector `80.71 / 76.08 / 69.50 / 58.35 / 43.98`.
- Old lowreg still has no `Training Over`; latest mAP remains `64.28`, and latest train line is Epoch 59 step 50 with `Loss=0.4123`.
- Route A reached Epoch 32 step 50 at `11:01:56` with `Loss=0.4898`, `cls_loss=0.2533`, `reg_loss=0.2365`; no mAP yet.
- Route B completed Epoch 34 at `11:03:27` with `Loss=0.8231`, `selector_gt_density_loss=0.3198`, `cls_loss=0.2675`, `reg_loss=0.2356`; no mAP yet.
- Failure scan remains `0` for all logs.
- Disk now shows `/data` `2.3T` size, `419G` used, `1.9T` available, `18%`.
- Decision: continue active four jobs. No cleanup is allowed for old uniform/lowreg/Route A/Route B while they are running; no severe-result gate and no new launch.

## Follow-Up Monitor at 11:06

- Remote time: `2026-05-29T11:06:12+08:00`.
- Upload gate still passes: train `200/200`, test `211/211`, `bad_size=0`, `ALL_PASS=True`.
- Active jobs remain old uniform `994341`, old lowreg `994342`, Route A `994378`, and Route B `994379`; old main `994340` remains completed.
- Old uniform still has no `Training Over`; latest/best remains `65.73`.
- Old lowreg still has no `Training Over`; latest mAP remains `64.28`. It completed Epoch 59 at `11:03:29` with `Loss=0.3959` and wrote `epoch_59.pth` at `11:03:30`, but cleanup is still disallowed while `994342` is `RUNNING`.
- Route A completed Epoch 32 at `11:04:46` with `Loss=0.4870`, `cls_loss=0.2553`, `reg_loss=0.2317`; no mAP yet.
- Route B reached Epoch 35 step 50 at `11:06:19` with `Loss=0.7497`, `selector_gt_density_loss=0.3200`, `cls_loss=0.2198`, `reg_loss=0.2097`; no mAP yet.
- Failure scan remains `0` for all logs.
- Disk shows `/data` `2.3T` size, `420G` used, `1.9T` available, `18%`.
- Decision: continue active four jobs. Do not clean lowreg until `Training Over` or completed `sacct` state appears; no severe-result gate and no new launch.

## Old Uniform Completion and Cleanup at 11:09

- Completion check command time: `2026-05-29T11:08:16+08:00`; cleanup command time: `2026-05-29T11:08:44+08:00`; training log completion time: `2026-05-29 11:09:13`.
- `sacct` reports old uniform job `994341` `COMPLETED` in `08:32:47`; log reports `Training Over...`.
- Final old uniform eval: `65.57` Avg-mAP, vector `80.80 / 75.81 / 69.31 / 58.04 / 43.89`.
- Best observed old uniform remains `65.73` at `10:40:13`, vector `80.71 / 76.08 / 69.50 / 58.35 / 43.98`; this is the strongest current control.
- Cleanup scope: `~/run/yuzibo/e2e_runs/exps/e2e_rawdensel_uniform_20260529_023345/gpu1_id0/checkpoint`, resolved to `/data/run01/sczc063/yuzibo/e2e_runs/exps/e2e_rawdensel_uniform_20260529_023345/gpu1_id0/checkpoint`.
- Disk before cleanup: `/data` `2.3T` size, `420G` used, `1.9T` available, `18%`.
- Removed only completed-run epoch files `epoch_19.pth` and `epoch_39.pth`; kept `epoch_59.pth`.
- Disk after cleanup: `/data` `2.3T` size, `420G` used, `1.9T` available, `18%`.
- Active jobs after cleanup: old lowreg `994342`, Route A `994378`, Route B `994379`.
- Decision: old uniform is now a completed control. Continue active jobs; do not clean lowreg until completion; no severe-result gate and no new launch.

## Follow-Up Monitor at 11:11

- Remote time: `2026-05-29T11:11:51+08:00`.
- Upload gate still passes: train `200/200`, test `211/211`, `bad_size=0`, `ALL_PASS=True`.
- Completed jobs: old main `994340`, old uniform `994341`.
- Active jobs: old lowreg `994342`, Route A `994378`, Route B `994379`.
- Old lowreg still has no `Training Over`; latest mAP remains `64.28`, latest train line remains Epoch 59 final with `Loss=0.3959`, and `epoch_59.pth` exists. Cleanup remains disallowed while `994342` is `RUNNING`.
- Route A completed Epoch 33 at `11:10:37` with `Loss=0.4812`, `cls_loss=0.2590`, `reg_loss=0.2222`; no mAP yet.
- Route B reached Epoch 36 step 50 at `11:11:51` with `Loss=0.7898`, `selector_gt_density_loss=0.3210`, `cls_loss=0.2421`, `reg_loss=0.2265`; no mAP yet.
- Failure scan remains `0` for all logs.
- Disk shows `/data` `2.3T` size, `419G` used, `1.9T` available, `18%`.
- Decision: continue active three jobs. Wait for lowreg `Training Over` before cleanup and Route A/B first mAP before route decision or new launch.

## Follow-Up Monitor at 11:14

- Remote time: `2026-05-29T11:14:29+08:00`.
- Upload gate still passes: train `200/200`, test `211/211`, `bad_size=0`, `ALL_PASS=True`.
- Active jobs remain old lowreg `994342`, Route A `994378`, and Route B `994379`.
- Old lowreg still has no `Training Over`; latest mAP remains `64.28`, latest train line remains Epoch 59 final with `Loss=0.3959`, and `epoch_59.pth` exists. Cleanup remains disallowed while `994342` is `RUNNING`.
- Route A reached Epoch 34 step 50 at `11:13:31` with `Loss=0.4817`, `cls_loss=0.2537`, `reg_loss=0.2280`; no mAP yet.
- Route B completed Epoch 36 at `11:14:26` with `Loss=0.7752`, `selector_gt_density_loss=0.3199`, `cls_loss=0.2341`, `reg_loss=0.2210`; no mAP yet.
- Failure scan remains `0` for all logs.
- Disk shows `/data` `2.3T` size, `419G` used, `1.9T` available, `18%`.
- Decision: continue active three jobs. No cleanup, no severe-result gate, and no new launch until lowreg completion or Route A/B first mAP.

## Follow-Up Monitor at 11:17

- Remote time: `2026-05-29T11:17:08+08:00`.
- Upload gate still passes: train `200/200`, test `211/211`, `bad_size=0`, `ALL_PASS=True`.
- Active jobs remain old lowreg `994342`, Route A `994378`, and Route B `994379`.
- Old lowreg still has no `Training Over`; latest mAP remains `64.28`, latest train line remains Epoch 59 final with `Loss=0.3959`, and `epoch_59.pth` exists. Cleanup remains disallowed while `994342` is `RUNNING`.
- Route A completed Epoch 34 at `11:16:21` with `Loss=0.4895`, `cls_loss=0.2625`, `reg_loss=0.2270`; no mAP yet.
- Route B reached Epoch 37 step 50 at `11:17:20` with `Loss=0.8149`, `selector_gt_density_loss=0.3216`, `cls_loss=0.2590`, `reg_loss=0.2340`; no mAP yet.
- Failure scan remains `0` for all logs.
- Disk shows `/data` `2.3T` size, `419G` used, `1.9T` available, `18%`.
- Decision: continue active three jobs. No cleanup, no severe-result gate, and no new launch until lowreg completion or Route A/B first mAP.

## Follow-Up Monitor at 11:21

- Remote time: `2026-05-29T11:21:10+08:00`.
- Upload gate still passes: train `200/200`, test `211/211`, `bad_size=0`, `ALL_PASS=True`.
- Active jobs remain old lowreg `994342`, Route A `994378`, and Route B `994379`.
- Old lowreg still has no `Training Over`; latest mAP remains `64.28`, latest train line remains Epoch 59 final at `11:03:29`, and `epoch_59.pth` exists. Cleanup remains disallowed while `994342` is `RUNNING`.
- The lowreg final eval window is about 18 minutes after final train, comparable to old main/uniform final eval durations, so this is not treated as a stall.
- Route A reached Epoch 35 step 50 at `11:19:24` with `Loss=0.4199`, `cls_loss=0.2155`, `reg_loss=0.2044`; no mAP yet.
- Route B completed Epoch 37 at `11:19:53` with `Loss=0.7994`, `selector_gt_density_loss=0.3197`, `cls_loss=0.2471`, `reg_loss=0.2324`; no mAP yet.
- Failure scan remains `0` for all logs.
- Disk shows `/data` `2.3T` size, `419G` used, `1.9T` available, `18%`.
- Decision: continue active three jobs. No cleanup, no severe-result gate, and no new launch until lowreg completion or Route A/B first mAP.

## Old Lowreg Completion and Cleanup at 11:23

- Completion check time: `2026-05-29T11:23:05+08:00`; cleanup command time: `2026-05-29T11:23:31+08:00`; training log completion time: `2026-05-29 11:22:16`.
- `sacct` reports old lowreg job `994342` `COMPLETED` in `08:48:33`; log reports `Training Over...`.
- Final old lowreg eval: `63.61` Avg-mAP, vector `78.46 / 73.94 / 66.52 / 56.23 / 42.90`.
- Best observed old lowreg remains `64.36` at `10:21:47`; latest-before-final was `64.28` at `10:52:06`.
- Interpretation: final lowreg is below random-fixed `63.77`, strict EMA `63.85`, stratified `64.64`, and uniform `65.73`, so lowreg provides no positive learned-selector evidence. This is not a severe-result trigger because the drop is not failure-scale and logs remain healthy.
- Cleanup scope: `~/run/yuzibo/e2e_runs/exps/e2e_rawdensel_lowreg_20260529_023345/gpu1_id0/checkpoint`, resolved to `/data/run01/sczc063/yuzibo/e2e_runs/exps/e2e_rawdensel_lowreg_20260529_023345/gpu1_id0/checkpoint`.
- Disk before cleanup: `/data` `2.3T` size, `419G` used, `1.9T` available, `18%`.
- Removed only completed-run epoch files `epoch_19.pth` and `epoch_39.pth`; kept `epoch_59.pth`.
- Disk after cleanup: `/data` `2.3T` size, `419G` used, `1.9T` available, `18%`.
- Active jobs after cleanup: Route A `994378`, Route B `994379`.
- Decision: continue Route A/B to first mAP. No severe-result gate and no new launch.

## Follow-Up Monitor at 11:26

- Remote time: `2026-05-29T11:26:29+08:00`.
- Upload gate still passes: train `200/200`, test `211/211`, `bad_size=0`, `ALL_PASS=True`.
- Active jobs: Route A `994378`, Route B `994379`, both `RUNNING` on `g0009`.
- Route A reached Epoch 36 step 50 at `11:25:19` with `Loss=0.4585`, `cls_loss=0.2377`, `reg_loss=0.2208`; no mAP yet.
- Route B completed Epoch 38 at `11:25:31` with `Loss=0.7908`, `selector_gt_density_loss=0.3198`, `cls_loss=0.2444`, `reg_loss=0.2264`; no mAP yet.
- Visible Route A/B checkpoints remain only `epoch_19.pth`; no epoch-39 checkpoint yet.
- Failure scan remains `0` for both logs.
- Disk shows `/data` `2.3T` size, `417G` used, `1.9T` available, `18%`.
- Decision: continue Route A/B. Watch for epoch-39/40 checkpoint/eval and first mAP; no cleanup, no severe-result gate, and no new launch.

## Follow-Up Monitor at 11:28

- Remote time: `2026-05-29T11:28:08+08:00`.
- Route A and Route B remain `RUNNING` on `g0009`.
- Route A completed Epoch 36 at `11:28:04` with `Loss=0.4471`, `cls_loss=0.2315`, `reg_loss=0.2155`; no mAP yet.
- Route B reached Epoch 39 step 50 at `11:28:14` with `Loss=0.7418`, `selector_gt_density_loss=0.3185`, `cls_loss=0.2206`, `reg_loss=0.2025`; no mAP yet.
- Visible Route A/B checkpoints remain only `epoch_19.pth`; no epoch-39 checkpoint yet.
- Decision: continue Route A/B. Recheck Route B epoch-39 checkpoint/eval soon. No cleanup, no severe-result gate, and no new launch.

## Follow-Up Monitor at 11:31

- Remote time: `2026-05-29T11:31:04+08:00`.
- Upload gate still passes: train `200/200`, test `211/211`, `bad_size=0`, `ALL_PASS=True`.
- Route A and Route B remain `RUNNING` on `g0009`.
- Route A reached Epoch 37 step 50 at `11:31:08` with `Loss=0.4772`, `cls_loss=0.2550`, `reg_loss=0.2223`; no mAP yet.
- Route B completed Epoch 39 at `11:30:58` with `Loss=0.7857`, `selector_gt_density_loss=0.3199`, `cls_loss=0.2429`, `reg_loss=0.2228`; no mAP yet.
- Route B wrote `epoch_39.pth` at `11:31:00`, size `623905307`; Route A still only has `epoch_19.pth`.
- Failure scan remains `0` for both logs.
- Disk shows `/data` `2.3T` size, `418G` used, `1.9T` available, `18%`.
- Decision: continue Route A/B to first mAP. Do not clean active checkpoints; no diagnostics or new launches before first mAP unless a failure appears.

## Follow-Up Monitor at 11:33

- Remote time: `2026-05-29T11:33:49+08:00`.
- Route A and Route B remain `RUNNING` on `g0009`.
- Route A completed Epoch 37 at `11:33:52` with `Loss=0.4631`, `cls_loss=0.2412`, `reg_loss=0.2219`; no mAP yet.
- Route B reached Epoch 40 step 50 at `11:33:49` with `Loss=0.7748`, `selector_gt_density_loss=0.3205`, `cls_loss=0.2322`, `reg_loss=0.2218`; no mAP yet.
- Visible checkpoints: Route A `epoch_19.pth`; Route B `epoch_19.pth` and `epoch_39.pth`.
- Failure scan remains `0` for both logs.
- Disk shows `/data` `2.3T` size, `418G` used, `1.9T` available, `18%`.
- Decision: continue Route A/B to first mAP. No cleanup, no severe-result gate, and no new launch.

## Follow-Up Monitor at 11:35

- Remote time: `2026-05-29T11:35:19+08:00`.
- Route A and Route B remain `RUNNING` on `g0009`.
- No `Average-mAP` line appears in either route log.
- Latest Route A train remains Epoch 37 final with `Loss=0.4631`.
- Latest Route B train remains Epoch 40 step 50 with `Loss=0.7748`, `selector_gt_density_loss=0.3205`.
- Decision: continue Route A/B. No cleanup, no severe-result gate, and no new launch.

## Follow-Up Monitor at 11:37

- Remote time: `2026-05-29T11:37:52+08:00`.
- Upload gate still passes: train `200/200`, test `211/211`, `bad_size=0`, `ALL_PASS=True`.
- Route A and Route B remain `RUNNING` on `g0009`.
- Route A reached Epoch 38 step 50 at `11:36:57` with `Loss=0.4715`, `cls_loss=0.2489`, `reg_loss=0.2226`; no mAP yet.
- Route B completed Epoch 40 at `11:36:28` with `Loss=0.7811`, `selector_gt_density_loss=0.3197`, `cls_loss=0.2409`, `reg_loss=0.2203`; no mAP yet.
- Visible checkpoints remain Route A `epoch_19.pth`, Route B `epoch_19.pth` and `epoch_39.pth`.
- Failure scan remains `0` for both logs.
- Disk shows `/data` `2.3T` size, `418G` used, `1.9T` available, `18%`.
- Decision: continue Route A/B to first mAP. No cleanup, no severe-result gate, and no new launch.

## Follow-Up Monitor at 11:40

- Remote time: `2026-05-29T11:40:55+08:00`.
- Upload gate still passes: train `200/200`, test `211/211`, `bad_size=0`, `ALL_PASS=True`.
- Route A and Route B remain `RUNNING` on `g0009`.
- Route A completed Epoch 38 at `11:39:46` with `Loss=0.4664`, `cls_loss=0.2427`, `reg_loss=0.2236`; no mAP yet.
- Route B reached Epoch 41 step 50 at `11:39:13` with `Loss=0.7609`, `selector_gt_density_loss=0.3198`, `cls_loss=0.2283`, `reg_loss=0.2126`; no mAP yet.
- Visible checkpoints remain Route A `epoch_19.pth`, Route B `epoch_19.pth` and `epoch_39.pth`.
- Failure scan remains `0` for both logs.
- Disk shows `/data` `2.3T` size, `418G` used, `1.9T` available, `18%`.
- Decision: continue Route A/B to first mAP. No cleanup, no severe-result gate, and no new launch.

## Follow-Up Monitor at 11:44

- Remote time: `2026-05-29T11:44:35+08:00`.
- Upload gate still passes: train `200/200`, test `211/211`, `bad_size=0`, `ALL_PASS=True`.
- Route A and Route B remain `RUNNING` on `g0009` for about `3:54:50`.
- Route A reached Epoch 39 step 50 at `11:42:41` with `Loss=0.4121`, `cls_loss=0.2151`, `reg_loss=0.1970`; no mAP yet.
- Route B completed Epoch 41 at `11:41:52` with `Loss=0.7863`, `selector_gt_density_loss=0.3199`, `cls_loss=0.2418`, `reg_loss=0.2244`; no mAP yet.
- Route B's train-only GT density auxiliary remains essentially flat near `0.320`, consistent with the epoch-19 selector diagnostic. This is weak selector-learning evidence, but not a failure trigger before first mAP.
- Visible checkpoints remain Route A `epoch_19.pth`, Route B `epoch_19.pth` and `epoch_39.pth`.
- Failure scan remains `0` for both logs.
- Disk shows `/data` `2.3T` size, `418G` used, `1.9T` available, `18%`.
- Decision: continue Route A/B to first mAP. Do not clean active checkpoints, do not launch new Route B variants from the flat auxiliary loss alone, and do not trigger a severe-result Pro review without mAP collapse or inconsistency.

## Follow-Up Monitor at 11:48

- Remote time: `2026-05-29T11:48:02+08:00`.
- Route A and Route B remain `RUNNING` on `g0009` for about `3:57:59`.
- Route A completed Epoch 39 at `11:45:35` with `Loss=0.4489`, `cls_loss=0.2326`, `reg_loss=0.2163`, wrote `epoch_39.pth` at `11:45:37`, and entered Epoch 40. No mAP yet.
- Route B raw log tail confirms it is actively validating rather than stalled: after Epoch 41 final at `11:41:52`, it entered validation/progress at `11:41:53` and reached about `161/396` by the tail check around `11:48`.
- No `Average-mAP` or `mAP at tIoU` appears yet.
- Visible checkpoints are Route A `epoch_19.pth`/`epoch_39.pth`, Route B `epoch_19.pth`/`epoch_39.pth`.
- Failure scan remains `0`.
- Decision: continue Route B validation to first mAP and Route A to its first eval. Do not clean active checkpoints, launch new variants, or trigger severe-result Pro review without mAP collapse/inconsistency.

## Follow-Up Monitor at 11:56

- Remote time: `2026-05-29T11:56:08+08:00`.
- Route A and Route B remain `RUNNING` on `g0009` for about `4:06:23`.
- Route A continued training to Epoch 41 step 50 at `11:54:22` with `Loss=0.4238`, `cls_loss=0.2161`, `reg_loss=0.2077`; no mAP yet.
- Route B stayed in validation/progress after Epoch 41 final; latest tail at `11:56` showed about `320/396` with no mAP printed yet.
- Visible checkpoints remain Route A `epoch_19.pth`/`epoch_39.pth`, Route B `epoch_19.pth`/`epoch_39.pth`.
- Failure scan remains `0`.
- Decision: continue Route B validation to first mAP and Route A to its first eval. Do not clean active checkpoints, launch new variants, or trigger severe-result Pro review without mAP collapse/inconsistency.

## Route B First Eval at 12:02

- Remote monitor time: `2026-05-29T12:02:56+08:00`.
- Route B first eval after Epoch 41 printed at `2026-05-29 12:00:23`: `63.42` Avg-mAP.
- Route B vector: `78.95 / 74.44 / 66.49 / 55.43 / 41.81` at tIoU `0.30 / 0.40 / 0.50 / 0.60 / 0.70`.
- Comparison: below random-fixed `63.77` by `-0.35`, strict EMA `63.85` by `-0.43`, stratified `64.64` by `-1.22`, and uniform control `65.73` by `-2.31`.
- Interpretation: this is negative learned-selector evidence for GT-aux residual Route B. It is not a severe-collapse trigger because the gap is not 5-point/failure-scale and failure scan remains `0`.
- Route B resumed Epoch 42 after the eval. Route A completed Epoch 41 at `11:57:11` with `Loss=0.4617` and is currently validating; no Route A mAP yet.
- Decision: send the current evidence to Pro before any Route B follow-up, continue Route A to first mAP, and keep Route B running unless a stop decision is recorded. Do not clean active checkpoints or launch new variants.

## Route B Pro Discussion and Stop at 12:16

- Globalai `gpt-5-pro` failed with `model_not_found`; Oracle browser `gpt-5.5-pro` succeeded in about `8m00s`.
- Pro discussion report: `research-wiki/experiments/BATA_ROUTEB_FIRST_EVAL_PRO_DISCUSSION_20260529.md`.
- Pro answer: `logs/oracle_pro_routeB_first_eval_20260529.answer.md`; stdout `logs/oracle_pro_routeB_first_eval_20260529_retry.txt`; stderr `logs/oracle_pro_routeB_first_eval_20260529_retry.err.txt`.
- Accepted Pro verdict: Route B should stop as a main route and remain diagnostic only. Do not rescue it by loss-weight, regularization, or longer-training tweaks.
- Accepted next route families: quota-based boundary/action selector, coarse-to-fine proposal-conditioned resampling, and irregular-aware detector head/assignment.
- Applied stop decision with `scancel 994379` at `2026-05-29T12:15:48+08:00`; `sacct` reports `CANCELLED+` in `04:26:03`.
- Cleanup at `2026-05-29T12:16:20+08:00`: checkpoint dir resolved inside `~/run/yuzibo`; disk before/after `/data` `2.3T` size, `419G` used, `1.9T` available, `18%`; removed only `epoch_19.pth`, kept `epoch_39.pth`, preserved logs and non-epoch artifacts.
- Route A `994378` remains running/validating on `g0009`.
- Decision: continue Route A to first mAP. No new launch until the next route implementation and review gates are complete.

## Route A First Eval at 12:18

- Remote monitor time: `2026-05-29T12:18:05+08:00`.
- Route A first eval after Epoch 41 printed at `2026-05-29 12:16:45`: `63.98` Avg-mAP.
- Route A vector: `79.11 / 74.36 / 67.24 / 57.04 / 42.14`.
- Comparison: below old uniform best `65.73` by `-1.75`, below historical uniform stride-2 `65.09` by `-1.11`, but above random-fixed `63.77` by `+0.21` and strict EMA `63.85` by `+0.13`.
- Interpretation: this is a parity warning, not yet final proof of a protocol bug, because this is Route A's first eval and old uniform improved in later evals. If Route A remains near `63.x` on the next eval, the exact-uniform/bypass path must be debugged before any learned-selector claim.
- Route A resumed Epoch 42 and remains `RUNNING`; Route B remains `CANCELLED+` and cleaned.
- Decision: continue Route A to at least the next eval; no new launch until next-route implementation and review gates are complete.

## Route A Health Monitor at 12:21

- Remote time: `2026-05-29T12:21:02+08:00`.
- Upload gate remains train `200/200`, test `211/211`, `bad_size=0`, `ALL_PASS=True`.
- Route A remains `RUNNING` on `g0009` for about `4:31:17`; Route B remains `CANCELLED+`.
- No new Route A mAP after `63.98`.
- Route A reached Epoch 42 step 50 at `12:19:52` with `Loss=0.4276`, `cls_loss=0.2173`, `reg_loss=0.2103`.
- Failure scan remains `0`.
- Disk shows `/data` `2.3T` size, `418G` used, `1.9T` available, `18%`.
- Decision: continue Route A to next eval. No cleanup while active and no new launch until next-route implementation/review gates are complete.

## Interpretation Gate

- Uniform control should approach the uniform stride-2 50% reference `65.09`; below about `64.3` suggests a pipeline or protocol bug.
- Main and lowreg must beat random-fixed Adapter `63.77` and strict EMA `63.85` before positive interpretation.
- Strong evidence requires exceeding the uniform stride-2 reference `65.09` by a meaningful margin, because simple uniform coverage is already known to be strong.

## Quota Action/Boundary Selector Implementation at 12:31

- Timestamp: `2026-05-29T12:31:15+08:00`.
- Self-check report: `research-wiki/experiments/BATA_E2E_QUOTA_SELECTOR_SELF_CHECK_20260529.md`.
- Local changed files:
  - `OpenTAD_BATA_Clean/opentad/models/selectors/temporal_density_selector.py`
  - `OpenTAD_BATA_Clean/configs/adatad/thumos/e2e_rawdensel_384of768_quota_action_boundary_adapter.py`
  - `OpenTAD_BATA_Clean/configs/adatad/thumos/e2e_rawdensel_384of768_quota_boundary_heavy_adapter.py`
  - `OpenTAD_BATA_Clean/tests/test_e2e_raw_frame_selector_contracts.py`
- Method: `selection_mode="quota"` with 384 selected frames split as either `160` context / `128` action / `96` boundary, or `128` / `128` / `128` for the boundary-heavy variant.
- Changed surface: selector internals, input sampling, and train-only selector losses. Detector head, backbone/projection, and test-time post-processing are unchanged.
- End-to-end contract: detection losses still flow through sampled raw frames and continuous selected positions back to selector scores. GT remap remains detached; action/boundary GT auxiliary is train-only.
- Leakage check: no test-time GT, teacher, diagnostic cache, raw-prediction loading, or post-processing change.
- Verification: local `py_compile` passed; local pytest returned `1 passed, 14 skipped`, with skipped tests being the existing Linux/torch selector tests. Local merged-config check could not run because Windows Python lacks `mmengine`; N16R4 preflight must cover merged config and full torch tests.
- Concurrent remote status: Route A `994378` remains active; latest mAP is still first eval `63.98`, and the second validation is in progress after Epoch 43. No cleanup or new launch occurred.
- Decision: continue to GPT-5.5 Pro code-grounded review, then Gemini CLI and DeepSeek CLI. Do not sync or launch quota jobs until all review gates and N16R4 preflight pass.

## Quota Selector Pro Gate at 13:04

- Timestamp: `2026-05-29T13:04:21+08:00`.
- Initial Oracle Pro file-upload run timed out and is not accepted as a completed review.
- Successful inline GPT-5.5 Pro review: `logs/oracle_pro_e2e_quota_selector_review_20260529_retry_inline.txt`; verdict `WARN`, no launch blocker, but requested stronger tests for visual-path gradient, invalid dense tail masks, optimizer grouping, and deployed quota splits.
- First focused re-review: `logs/oracle_pro_e2e_quota_selector_fix_review_20260529.txt`; verdict `FAIL` because the quota split test used a toy selector with `target_len=8` while checking deployed quotas that sum to `384`.
- Applied fix: deployed quota tests now instantiate real `dense_len=768`, `target_len=384` selectors, and quota remainder allocation uses stable tie-break by fraction descending and quota index ascending.
- Second focused re-review: `logs/oracle_pro_e2e_quota_selector_fix2_review_20260529.txt`; verdict `PASS`. Pro explicitly allows Gemini CLI + DeepSeek CLI review and N16R4 preflight.
- Current status: local implementation remains unsynced; Gemini and DeepSeek gates are still required. Contract remains `50% ViT/backbone compute, dense decode`; train-only GT auxiliary only; no test-time GT, teacher, cache, or post-processing change.

## Quota Selector Gemini Gate at 13:06

- Gemini CLI command used model `gemini-3-pro-preview` in read-only review mode.
- Output: `logs/gemini3_pro_preview_e2e_quota_selector_review_20260529.txt`.
- Stderr: `logs/gemini3_pro_preview_e2e_quota_selector_review_20260529.err.txt`; only terminal color and ripgrep fallback warnings.
- Exit code: `0`.
- Verdict: `PASS`, no blocking findings and no required fixes.
- Gemini confirmed protocol separation, no test-time GT/teacher/cache leakage, quota config sums to `384`, and continuous sampling keeps the visual path differentiable.
- Current status: DeepSeek CLI review remains required before sync/deployment.

## Quota Selector DeepSeek Gate at 13:17

- First Claude CLI DeepSeek command used `deepseek-v4-pro` and exited `0`, but stdout only reported that a plan file contained the full PASS verdict. Because the logged stdout was shallow, it was not accepted as the completed gate.
- First run artifacts:
  - stdout: `logs/claude_deepseek_v4_pro_e2e_quota_selector_review_20260529.txt`
  - debug: `logs/claude_deepseek_v4_pro_e2e_quota_selector_review_20260529.debug.log`
  - generated plan artifact: `C:\Users\skywalker\.claude\plans\read-only-secondary-code-starry-tulip.md`
- Retry command disabled `Write`, `Edit`, and `ExitPlanMode` and allowed only read/search tools.
- Accepted retry stdout: `logs/claude_deepseek_v4_pro_e2e_quota_selector_review_20260529_retry.txt`.
- Retry exit code: `0`.
- Verdict: `PASS`; no blocking findings and no required fixes.
- Non-blocking notes: `detach_gt_remap=False` would not make GT remapping differentiable through `searchsorted`; context quota channel is not directly used in position selection; `cummax` is an edge-case safety net; `/255` normalization threshold is heuristic.
- Current status: mandatory local external review gates are complete. Next step is local verification, a reviewed-file-only commit, sync, and N16R4 preflight.

## Quota Selector Local Verification and Commit at 13:20

- Timestamp: `2026-05-29T13:20:09+08:00`.
- Local post-review checks:
  - `python -m py_compile opentad\models\selectors\temporal_density_selector.py tests\test_e2e_raw_frame_selector_contracts.py configs\adatad\thumos\e2e_rawdensel_384of768_quota_action_boundary_adapter.py configs\adatad\thumos\e2e_rawdensel_384of768_quota_boundary_heavy_adapter.py`: PASS, exit `0`.
  - `python -m pytest tests\test_e2e_raw_frame_selector_contracts.py -q -rs`: `1 passed, 17 skipped`; skipped tests are the expected Windows/Linux-torch gated tests.
  - `git diff --check -- <four reviewed files>`: no whitespace errors; only LF/CRLF warnings.
- Commit: `7591ead add e2e quota frame selector`.
- Commit scope: only four reviewed files: selector implementation, two quota configs, and E2E selector contract tests. Unrelated dirty BATA boundary-acquisition/post-processing files were not staged or committed.
- Current status: ready to sync reviewed files to `~/run/yuzibo/OpenTAD_BATA_Clean` and run N16R4 Linux preflight.

## Quota Selector Sync at 13:22

- Synced only the four reviewed files from commit `7591ead` to N16R4 remote tree `~/run/yuzibo/OpenTAD_BATA_Clean` (`/data/home/sczc063/run/yuzibo/OpenTAD_BATA_Clean`).
- Sync tool: Windows native `C:\Windows\System32\OpenSSH\scp.exe`; no WSL.
- Synced files:
  - `opentad/models/selectors/temporal_density_selector.py`
  - `tests/test_e2e_raw_frame_selector_contracts.py`
  - `configs/adatad/thumos/e2e_rawdensel_384of768_quota_action_boundary_adapter.py`
  - `configs/adatad/thumos/e2e_rawdensel_384of768_quota_boundary_heavy_adapter.py`
- Remote `ls -l` confirmed all four files updated at `May 29 13:22`.
- No dataset, checkpoint, or unrelated dirty BATA file was copied.
- Concurrent Route A monitor: third eval at `2026-05-29 13:18:47` reached `64.38` Avg-mAP, still below old uniform best `65.73`; job `994378` remains active.

## Quota Selector N16R4 Preflight at 13:27

- Accepted preflight log: `~/run/yuzibo/OpenTAD_BATA_Clean/logs/e2e_quota_selector_preflight_20260529_rerun3.log`.
- Invalid earlier attempts:
  - `e2e_quota_selector_preflight_20260529.log`: not accepted because the PowerShell-to-bash pipe did not load module/conda or enforce `set -e`; it used `/usr/bin/python`.
  - `e2e_quota_selector_preflight_20260529_rerun.log`: failed before checks due `/etc/profile` under `set -u`.
  - `e2e_quota_selector_preflight_20260529_rerun2.log`: `py_compile` and pytest passed, but the manual merged-config assertion used the wrong key path `cfg.model.backbone.total_frames`.
- Accepted run details:
  - Module/env loaded: `cuda/11.8`, `miniforge3/24.11`, `~/run/yuzibo/conda_envs/opentad`.
  - Python: `/data/home/sczc063/run/yuzibo/conda_envs/opentad/bin/python`, version `3.10.20`.
  - `py_compile`: PASS.
  - Linux contract tests: `18 passed in 22.14s`.
- Merged config checks: PASS for `160/128/96` and `128/128/128` quota configs.
- No training ran on the login node.
- Decision: submit two one-GPU Slurm jobs for the quota variants.

## Quota Selector Launch at 13:29

- Submitted at `2026-05-29T13:29:58+08:00` using `scripts/run_e2e_rawdensel_n16r4.sbatch`.
- Jobs:
  - `994707 e2e_quota_ab`, config `configs/adatad/thumos/e2e_rawdensel_384of768_quota_action_boundary_adapter.py`, run tag `e2e_quota_action_boundary_20260529_1329`.
  - `994708 e2e_quota_bh`, config `configs/adatad/thumos/e2e_rawdensel_384of768_quota_boundary_heavy_adapter.py`, run tag `e2e_quota_boundary_heavy_20260529_1329`.
- Both were initially `PENDING`.
- Route A job `994378` remains active, so once both quota jobs run this gives the requested 3-experiment/GPU parallel set.
- Existing `scripted_gpu` jobs `994471` and `994473` were inspected before launch: `Command=sleep`, `WorkDir=/data/run01/sczc063/wangruofan`, outside the project workspace. They were not cancelled.
- Next monitor target: job-internal preflight, log creation under `~/run/yuzibo/OpenTAD_BATA_Clean/logs/`, and `Training Starts`.

## Quota Selector Running Confirmation at 13:32

- Jobs `994707 e2e_quota_ab` and `994708 e2e_quota_bh` are both `RUNNING` on `g0042`.
- Logs:
  - `~/run/yuzibo/OpenTAD_BATA_Clean/logs/e2e_quota_action_boundary_20260529_1329_n16r4.log`
  - `~/run/yuzibo/OpenTAD_BATA_Clean/logs/e2e_quota_boundary_heavy_20260529_1329_n16r4.log`
- Both logs show job-internal `merged_config_preflight=PASS`.
- Both reached `Training Starts` and `[Train]: Epoch 0 started` at `2026-05-29 13:30:24`.
- No traceback, OOM, or runtime error is visible at startup.
- Route A `994378` remains active, so the live experiment set is Route A plus the two quota variants.
- First-eval gates from the Pro discussion remain active:
  - `<64.0`: stop unless log anomaly.
  - `64.0-64.64`: keep only healthier one to second eval.
  - `>=64.64`: continue to second eval.
  - `>=65.09`: priority continue.
  - `>=65.73`: champion candidate.

## Quota Selector First-Loss Health Check at 13:36

- Jobs remain `RUNNING` for about `5m42s`.
- Action/boundary quota log first loss line:
  - `Loss=2.1287`
  - `selector_gt_density_loss=0.0637`
  - `selector_quota_action_loss=0.1275`
  - `selector_quota_boundary_loss=0.1912`
  - `cls_loss=0.9897`, `reg_loss=0.7564`, `mem=3189MB`
- Boundary-heavy quota log first loss line:
  - `Loss=2.2962`
  - `selector_gt_density_loss=0.0637`
  - `selector_quota_action_loss=0.1593`
  - `selector_quota_boundary_loss=0.3186`
  - `cls_loss=0.9947`, `reg_loss=0.7594`, `mem=3189MB`
- Failure scan count is `0` for both logs: no NaN, OOM, traceback, or runtime error.
- Route A latest mAP trend remains `63.98 -> 64.35 -> 64.38`.
- Decision: deployment is healthy; continue periodic monitoring to first eval.

## Upload and Quota Health Recheck at 13:39

- Upload marker remains valid: `~/run/yuzibo/setup_logs/N16R4_THUMOS_UPLOAD_GATE_PASS_20260528_115821.ok` reports `ALL_PASS=True`.
- Recount corrected for symlinks:
  - train: `entries_mp4=200`, `symlink_mp4=200`, `broken_symlink_mp4=0`
  - test: `entries_mp4=211`, `symlink_mp4=211`, `broken_symlink_mp4=0`
  - manifests: `200/211`
- Disk remains safe: `/data` `418G` used, `1.9T` available, `18%`.
- Quota action-boundary completed Epoch 0:
  - step 99 `Loss=1.9960`
  - selector action/boundary losses `0.1279/0.1918`
  - `cls_loss=0.9567`, `reg_loss=0.6555`
- Quota boundary-heavy completed Epoch 0:
  - step 99 `Loss=2.1875`
  - selector action/boundary losses `0.1598/0.3196`
  - `cls_loss=0.9531`, `reg_loss=0.6905`
- Failure scan remains `0` for both quota logs.
- Route A remains running with latest mAP trend `63.98 -> 64.35 -> 64.38`; no new eval after `13:18:47`.
- Decision: continue all three active runs. No stop, cleanup, or severe-result gate is triggered.

## Active Training Monitor at 13:42

- Upload marker still `ALL_PASS=True`; train/test entries remain `200/211`, with `broken_symlink=0`.
- Disk remains safe with `/data` about `1.9T` available.
- Quota jobs are still `RUNNING` for about `12m11s`.
- Action-boundary reached Epoch 1 step 50 at `13:39:48`:
  - `Loss=1.4044`
  - selector action/boundary losses `0.1280/0.1920`
  - `cls_loss=0.6824`, `reg_loss=0.3377`
  - failure count `0`
- Boundary-heavy reached Epoch 1 step 50 at `13:39:47`:
  - `Loss=1.5639`
  - selector action/boundary losses `0.1600/0.3200`
  - `cls_loss=0.6740`, `reg_loss=0.3454`
  - failure count `0`
- Route A has no new mAP after `64.38`; latest visible train remains Epoch 47 final `Loss=0.4262`, failure count `0`.
- Decision: continue all three active runs. No stop, cleanup, or severe-result gate.

## Active Training Monitor at 13:44

- Upload gate remains valid: train/test entries `200/211`, no broken mp4 symlinks.
- Quota jobs are `RUNNING` for about `14m43s`.
- Action-boundary completed Epoch 1 at `13:42:32`:
  - `Loss=1.3781`
  - selector action/boundary losses `0.1279/0.1918`
  - `cls_loss=0.6483`, `reg_loss=0.3459`
  - failure count `0`
- Boundary-heavy completed Epoch 1 at `13:42:27`:
  - `Loss=1.5368`
  - selector action/boundary losses `0.1599/0.3197`
  - `cls_loss=0.6433`, `reg_loss=0.3495`
  - failure count `0`
- Route A has no new mAP after `64.38`; latest visible training remains Epoch 47 final.
- Decision: continue all three active runs. No gate triggered.

## Active Training Monitor at 13:47

- Upload gate remains valid: train/test entries `200/211`, no broken mp4 symlinks.
- Quota jobs are `RUNNING` for about `17m27s`.
- Action-boundary reached Epoch 2 step 50 at `13:45:34`:
  - `Loss=1.2342`
  - selector action/boundary losses `0.1277/0.1916`
  - `cls_loss=0.5148`, `reg_loss=0.3359`
  - failure count `0`
- Boundary-heavy reached Epoch 2 step 50 at `13:45:20`:
  - `Loss=1.3954`
  - selector action/boundary losses `0.1597/0.3193`
  - `cls_loss=0.5144`, `reg_loss=0.3378`
  - failure count `0`
- Route A has no new mAP after `64.38`; latest visible training remains Epoch 47 final, likely validating or transitioning.
- No quota checkpoint yet.
- Decision: continue all three active runs. No gate triggered.

## Route A Fourth Eval and Quota Epoch 2 at 13:51

- Upload gate remains valid: train/test entries `200/211`, no broken mp4 symlinks.
- Quota jobs are `RUNNING` for about `20m50s`.
- Action-boundary completed Epoch 2 at `13:48:29`:
  - `Loss=1.2727`
  - selector action/boundary losses `0.1279/0.1918`
  - `cls_loss=0.5461`, `reg_loss=0.3426`
  - failure count `0`
- Boundary-heavy completed Epoch 2 at `13:48:01`:
  - `Loss=1.4343`
  - selector action/boundary losses `0.1599/0.3197`
  - `cls_loss=0.5469`, `reg_loss=0.3435`
  - failure count `0`
- Route A fourth eval at `2026-05-29 13:49:49`: `64.46` Avg-mAP.
- Route A vector: `79.62 / 75.09 / 67.89 / 57.23 / 42.44`.
- Interpretation: Route A improved slightly but remains below uniform stride-2 `65.09` and old uniform best `65.73`; parity warning remains. This is not a severe-result trigger.
- Decision: continue all three active runs. Keep the Route A parity caveat for quota interpretation.

## Active Training Monitor at 13:55

- Upload gate remains valid: train/test entries `200/211`, no broken mp4 symlinks.
- Disk remains safe with `/data` about `1.9T` available.
- Quota jobs are `RUNNING` for about `24m47s`.
- Action-boundary reached Epoch 3 step 50 at `13:51:34`:
  - `Loss=1.2327`
  - selector action/boundary losses `0.1282/0.1923`
  - `cls_loss=0.5060`, `reg_loss=0.3419`
  - failure count `0`
- Boundary-heavy reached Epoch 3 step 50 at `13:51:04`:
  - `Loss=1.3834`
  - selector action/boundary losses `0.1602/0.3204`
  - `cls_loss=0.4970`, `reg_loss=0.3413`
  - failure count `0`
- Route A resumed after fourth eval and reached Epoch 48 step 50 at `13:52:50`: `Loss=0.4288`, `cls_loss=0.2194`, `reg_loss=0.2094`; failure count `0`.
- Decision: continue all three active runs. No gate triggered.

## Active Training Monitor at 13:57

- Upload gate remains valid: train/test entries `200/211`, no broken mp4 symlinks.
- Quota jobs are `RUNNING` for about `27m19s`.
- Action-boundary completed Epoch 3 at `13:54:15`:
  - `Loss=1.2345`
  - selector action/boundary losses `0.1279/0.1919`
  - `cls_loss=0.5162`, `reg_loss=0.3343`
  - failure count `0`
- Boundary-heavy completed Epoch 3 at `13:53:45`:
  - `Loss=1.3856`
  - selector action/boundary losses `0.1599/0.3198`
  - `cls_loss=0.5025`, `reg_loss=0.3392`
  - failure count `0`
- Route A completed Epoch 48 at `13:55:33`: `Loss=0.4179`, `cls_loss=0.2127`, `reg_loss=0.2051`; failure count `0`.
- Latest Route A mAP remains `64.46`.
- Decision: continue all three active runs. No gate triggered.

## Active Training Monitor at 14:00

- Upload gate remains valid: train/test entries `200/211`, no broken mp4 symlinks.
- Quota jobs are `RUNNING` for about `30m01s`.
- Action-boundary reached Epoch 4 step 50 at `13:57:11`:
  - `Loss=1.2420`
  - selector action/boundary losses `0.1285/0.1928`
  - `cls_loss=0.5196`, `reg_loss=0.3367`
  - failure count `0`
- Boundary-heavy reached Epoch 4 step 50 at `13:56:38`:
  - `Loss=1.4165`
  - selector action/boundary losses `0.1607/0.3213`
  - `cls_loss=0.5340`, `reg_loss=0.3360`
  - failure count `0`
- Route A reached Epoch 49 step 50 at `13:58:36`: `Loss=0.3808`, `cls_loss=0.1923`, `reg_loss=0.1885`; failure count `0`.
- Latest Route A mAP remains `64.46`.
- Decision: continue all three active runs. No gate triggered.

## Active Training Monitor at 14:03

- Upload gate remains valid: train/test entries `200/211`, no broken mp4 symlinks.
- Quota jobs are `RUNNING` for about `33m03s`.
- Action-boundary completed Epoch 4 at `13:59:55`:
  - `Loss=1.1984`
  - selector action/boundary losses `0.1279/0.1919`
  - `cls_loss=0.4866`, `reg_loss=0.3276`
  - failure count `0`
- Boundary-heavy completed Epoch 4 at `13:59:22`:
  - `Loss=1.3544`
  - selector action/boundary losses `0.1599/0.3199`
  - `cls_loss=0.4837`, `reg_loss=0.3266`
  - failure count `0`
- Route A completed Epoch 49 at `14:01:21`: `Loss=0.4179`, `cls_loss=0.2127`, `reg_loss=0.2052`; failure count `0`.
- Latest Route A mAP remains `64.46`.
- Decision: continue all three active runs. No gate triggered.

## Active Training Monitor at 14:09

- Upload gate remains valid: train/test expected/have `200/200` and `211/211`, `bad_size=0`, `ALL_PASS=True`; symlink recount remains `200/211` with no broken mp4 symlinks.
- `/data` is still safe in the project quota view: about `1.9T` available.
- Slurm jobs remain `RUNNING`: Route A `994378` on `g0009`, quota action-boundary `994707` on `g0042`, quota boundary-heavy `994708` on `g0042`.
- Route A has no new mAP after `64.46`, but the raw log tail shows active post-Epoch-49 validation progress, so it is not considered stalled. Failure count remains `0`.
- Quota action-boundary completed Epoch 5 at `14:05:42`: `Loss=1.2066`, selector action/boundary losses `0.1279/0.1919`, `cls_loss=0.4941`, `reg_loss=0.3284`, failure count `0`.
- Quota boundary-heavy completed Epoch 5 at `14:05:01`: `Loss=1.3690`, selector action/boundary losses `0.1599/0.3199`, `cls_loss=0.4963`, `reg_loss=0.3287`, failure count `0`.
- The initial repo-local checkpoint search was not authoritative because the launcher writes outputs to `~/run/yuzibo/e2e_runs/exps/$RUN_TAG`; no cleanup or selector-diagnostic extraction was performed at this monitor point.
- Decision: continue all three active runs. Next gates are Route A final eval/`Training Over`, quota epoch-19 checkpoint diagnostics, and quota first eval.

## Active Training Monitor at 14:12

- Slurm still reports Route A `994378`, quota action-boundary `994707`, and quota boundary-heavy `994708` as `RUNNING`.
- Route A has no new mAP after `64.46`; raw final-validation progress advanced to about `61%` (`242/396`) and failure count remains `0`, so the run is still actively validating.
- Quota action-boundary reached Epoch 6 step 50 at `14:08:44`: `Loss=1.1248`, selector action/boundary losses `0.1274/0.1910`, `cls_loss=0.4437`, `reg_loss=0.2987`, failure count `0`.
- Quota boundary-heavy reached Epoch 6 step 50 at `14:07:52`: `Loss=1.2901`, selector action/boundary losses `0.1592/0.3184`, `cls_loss=0.4475`, `reg_loss=0.3011`, failure count `0`.
- Decision: continue all three active runs. No cleanup or severe-result gate.

## Artifact Path Correction at 14:14

- The launcher sets `WORK_DIR="${YUZIBO_ROOT}/e2e_runs/exps/${RUN_TAG}"`, so run artifacts are under `~/run/yuzibo/e2e_runs/exps/<RUN_TAG>/gpu1_id0`, not repo-local `exps/`.
- Route A output dir exists: `~/run/yuzibo/e2e_runs/exps/e2e_routeA_uniform_exact_20260529_074925/gpu1_id0`.
- Route A currently has active checkpoints `checkpoint/epoch_19.pth` and `checkpoint/epoch_39.pth`; the run is still validating, so no cleanup is allowed yet.
- Quota action-boundary and boundary-heavy output dirs exist with copied config files, but no `epoch_*.pth` yet.
- Decision: use the corrected `e2e_runs/exps` paths for future selector diagnostics and checkpoint cleanup records.

## Active Training Monitor at 14:17

- Route A has no new mAP after `64.46`; failure count remains `0`.
- A compact-tail helper in the monitor script hit `ImportError: No module named pathlib` under login-node default Python. This is not in the training log and is not an experiment failure.
- Quota action-boundary completed Epoch 6 at `14:11:27` with `Loss=1.1360`, then reached Epoch 7 step 50 at `14:14:28` with `Loss=1.1260`, selector action/boundary losses `0.1280/0.1920`, `cls_loss=0.4331`, `reg_loss=0.3086`, failure count `0`.
- Quota boundary-heavy completed Epoch 6 at `14:10:28` with `Loss=1.2968`, then reached Epoch 7 step 50 at `14:13:17` with `Loss=1.2900`, selector action/boundary losses `0.1600/0.3200`, `cls_loss=0.4412`, `reg_loss=0.3045`, failure count `0`.
- Decision: continue all three active runs. No cleanup or severe-result gate.

## Active Training Monitor at 14:20

- Upload gate still passes: train/test expected/have `200/200` and `211/211`, `bad_size=0`, `ALL_PASS=True`; symlink recount remains `200/211` with no broken mp4 symlinks.
- `/data` remains safe in the project quota view with about `1.9T` available.
- Slurm still reports all three project jobs `RUNNING`.
- Route A produced a new eval at `14:20:43`: `64.68` Avg-mAP, vector `79.90 / 75.38 / 68.19 / 57.31 / 42.63`, failure count `0`.
- Interpretation: Route A now slightly exceeds stratified `64.64`, but it is still below uniform stride-2 `65.09` and old uniform best `65.73`; keep the uniform-control parity caveat. This does not by itself support a learned quota-selector gain.
- Quota action-boundary completed Epoch 7 at `14:17:23`: `Loss=1.1045`, selector action/boundary losses `0.1279/0.1918`, `cls_loss=0.4172`, `reg_loss=0.3034`, failure count `0`.
- Quota boundary-heavy completed Epoch 7 at `14:16:03`: `Loss=1.2640`, selector action/boundary losses `0.1599/0.3197`, `cls_loss=0.4200`, `reg_loss=0.3002`, failure count `0`.
- Correct artifact paths: Route A has active `epoch_19.pth` and `epoch_39.pth`; quota dirs have log/config only and no checkpoint yet.
- Decision: continue all three active runs. No cleanup or severe-result gate.

## Active Training Monitor at 14:24

- Upload gate still passes: train/test expected/have `200/200` and `211/211`, `bad_size=0`, `ALL_PASS=True`; recount remains `200/211` with no broken mp4 symlinks.
- `/data` remains safe in the project quota view with about `1.9T` available.
- Slurm still reports Route A `994378`, quota action-boundary `994707`, and quota boundary-heavy `994708` as `RUNNING`.
- Route A resumed training after the `64.68` eval and reached Epoch 50 step 50 at `14:23:37`: `Loss=0.3762`, `cls_loss=0.1918`, `reg_loss=0.1844`, failure count `0`.
- Quota action-boundary reached Epoch 8 step 50 at `14:20:39`: `Loss=1.1650`, selector action/boundary losses `0.1292/0.1938`, `cls_loss=0.4666`, `reg_loss=0.3106`, failure count `0`.
- Quota boundary-heavy completed Epoch 8 at `14:21:39`: `Loss=1.2423`, selector action/boundary losses `0.1599/0.3197`, `cls_loss=0.4032`, `reg_loss=0.2954`, failure count `0`.
- Artifact paths unchanged: Route A has active `epoch_19.pth` and `epoch_39.pth`; quota dirs have log/config only and no checkpoint yet.
- Decision: continue all three active runs. No cleanup, selector diagnostic, first-eval gate, or severe-result gate yet.

## Active Training Monitor at 14:29

- Upload gate still passes: train/test expected/have `200/200` and `211/211`, `bad_size=0`, `ALL_PASS=True`; train/test broken symlink counts remain `0/0`.
- `/data` remains safe in the project quota view with about `1.9T` available.
- Slurm still reports Route A `994378`, quota action-boundary `994707`, and quota boundary-heavy `994708` as `RUNNING`.
- Route A completed Epoch 50 at `14:26:37`: `Loss=0.3826`, `cls_loss=0.1917`, `reg_loss=0.1909`; latest mAP remains `64.68`; failure count `0`.
- Quota action-boundary completed Epoch 8 at `14:23:26`: `Loss=1.1004`, selector action/boundary losses `0.1279/0.1918`, `cls_loss=0.4179`, `reg_loss=0.2986`, then reached Epoch 9 step 50 at `14:26:25` with `Loss=1.0113`; failure count `0`.
- Quota boundary-heavy reached Epoch 9 step 50 at `14:24:28`: `Loss=1.1753`, selector action/boundary losses `0.1598/0.3197`, `cls_loss=0.3512`, `reg_loss=0.2804`, failure count `0`.
- Artifact paths unchanged: Route A active checkpoints `epoch_19.pth` and `epoch_39.pth`; quota dirs have no `epoch_*.pth` yet.
- Decision: continue all three active runs. No cleanup, selector diagnostic, first-eval gate, or severe-result gate yet.

## Active Training Monitor at 14:43

- Upload gate still passes: train/test expected/have `200/200` and `211/211`, `bad_size=0`, `ALL_PASS=True`.
- Slurm still reports Route A `994378`, quota action-boundary `994707`, and quota boundary-heavy `994708` as `RUNNING`.
- Route A latest log evidence remains Epoch 51 final `Loss=0.4036`, latest mAP `64.68`, failure count `0`.
- Quota action-boundary completed Epoch 11 at `14:40:55`: `Loss=1.0353`, selector action/boundary losses `0.1279/0.1918`, `cls_loss=0.3653`, `reg_loss=0.2860`, failure count `0`.
- Quota boundary-heavy reached Epoch 12 step 50 at `14:41:05`: `Loss=1.2264`, selector action/boundary losses `0.1599/0.3197`, `cls_loss=0.3874`, `reg_loss=0.2952`, failure count `0`.
- Artifact paths unchanged: Route A active checkpoints `epoch_19.pth` and `epoch_39.pth`; quota dirs still have no `epoch_*.pth`.
- Decision: continue all three active runs. No cleanup, selector diagnostic, first-eval gate, or severe-result gate yet.

## Active Training Monitor at 14:36

- Upload gate still passes with train/test `200/211`, `ALL_PASS=True`, and no broken mp4 symlinks.
- Slurm still reports Route A `994378`, quota action-boundary `994707`, and quota boundary-heavy `994708` as `RUNNING`.
- Latest log evidence is unchanged from the prior monitor: Route A Epoch 51 final `Loss=0.4036`, latest mAP `64.68`; quota action-boundary Epoch 10 step 50 `Loss=1.0845`; quota boundary-heavy Epoch 10 final `Loss=1.2062`.
- Failure counts remain `0`; quota dirs still have no `epoch_*.pth`.
- Decision: continue all three active runs. No cleanup, selector diagnostic, first-eval gate, or severe-result gate yet.

## Active Training Monitor at 14:39

- Upload gate still passes with train/test `200/211`, `ALL_PASS=True`, and no broken mp4 symlinks.
- Slurm still reports Route A `994378`, quota action-boundary `994707`, and quota boundary-heavy `994708` as `RUNNING`.
- Route A latest log evidence remains Epoch 51 final `Loss=0.4036`, latest mAP `64.68`, failure count `0`.
- Quota action-boundary completed Epoch 10 at `14:35:07`: `Loss=1.0557`, selector action/boundary losses `0.1279/0.1919`, `cls_loss=0.3839`, `reg_loss=0.2878`, failure count `0`.
- Quota boundary-heavy reached Epoch 11 step 50 at `14:35:34`: `Loss=1.2066`, selector action/boundary losses `0.1611/0.3221`, `cls_loss=0.3749`, `reg_loss=0.2839`, failure count `0`.
- Artifact paths unchanged: Route A active checkpoints `epoch_19.pth` and `epoch_39.pth`; quota dirs still have no `epoch_*.pth`.
- Decision: continue all three active runs. No cleanup, selector diagnostic, first-eval gate, or severe-result gate yet.

## Active Training Monitor at 14:41

- Upload gate still passes: train/test expected/have `200/200` and `211/211`, `bad_size=0`, `ALL_PASS=True`.
- Slurm still reports Route A `994378`, quota action-boundary `994707`, and quota boundary-heavy `994708` as `RUNNING`.
- Route A latest log evidence remains Epoch 51 final `Loss=0.4036`, latest mAP `64.68`, failure count `0`.
- Quota action-boundary reached Epoch 11 step 50 at `14:38:09`: `Loss=1.0484`, selector action/boundary losses `0.1289/0.1933`, `cls_loss=0.3735`, `reg_loss=0.2881`, failure count `0`.
- Quota boundary-heavy completed Epoch 11 at `14:38:14`: `Loss=1.1914`, selector action/boundary losses `0.1599/0.3197`, `cls_loss=0.3648`, `reg_loss=0.2828`, failure count `0`.
- Artifact paths unchanged: Route A active checkpoints `epoch_19.pth` and `epoch_39.pth`; quota dirs still have no `epoch_*.pth`.
- Decision: continue all three active runs. No cleanup, selector diagnostic, first-eval gate, or severe-result gate yet.

## Active Training Monitor at 14:35

- Upload gate still passes: train/test expected/have `200/200` and `211/211`, `bad_size=0`, `ALL_PASS=True`; train/test broken symlink counts remain `0/0`.
- Slurm still reports Route A `994378`, quota action-boundary `994707`, and quota boundary-heavy `994708` as `RUNNING`.
- Route A latest train evidence remains Epoch 51 final at `14:32:25`: `Loss=0.4036`; latest mAP remains `64.68`; failure count `0`.
- Quota action-boundary reached Epoch 10 step 50 at `14:32:13`: `Loss=1.0845`, selector action/boundary losses `0.1276/0.1914`, `cls_loss=0.4049`, `reg_loss=0.2965`, failure count `0`.
- Quota boundary-heavy completed Epoch 10 at `14:32:43`: `Loss=1.2062`, selector action/boundary losses `0.1599/0.3198`, `cls_loss=0.3792`, `reg_loss=0.2832`, failure count `0`.
- Artifact paths unchanged: Route A active checkpoints `epoch_19.pth` and `epoch_39.pth`; quota dirs have no `epoch_*.pth` yet.
- Decision: continue all three active runs. No cleanup, selector diagnostic, first-eval gate, or severe-result gate yet.

## Active Training Monitor at 14:33

- Upload gate still passes: train/test expected/have `200/200` and `211/211`, `bad_size=0`, `ALL_PASS=True`; train/test broken symlink counts remain `0/0`.
- Slurm still reports Route A `994378`, quota action-boundary `994707`, and quota boundary-heavy `994708` as `RUNNING`.
- Route A completed Epoch 51 at `14:32:25`: `Loss=0.4036`, `cls_loss=0.2013`, `reg_loss=0.2023`; latest mAP remains `64.68`; failure count `0`.
- Quota action-boundary completed Epoch 9 at `14:29:21`: `Loss=1.0643`, selector action/boundary losses `0.1279/0.1918`, `cls_loss=0.3917`, `reg_loss=0.2887`, failure count `0`.
- Quota boundary-heavy reached Epoch 10 step 50 at `14:29:58`: `Loss=1.2370`, selector action/boundary losses `0.1595/0.3190`, `cls_loss=0.4050`, `reg_loss=0.2895`, failure count `0`.
- Artifact paths unchanged: Route A active checkpoints `epoch_19.pth` and `epoch_39.pth`; quota dirs have no `epoch_*.pth` yet.
- Decision: continue all three active runs. No cleanup, selector diagnostic, first-eval gate, or severe-result gate yet.

## Active Training Monitor at 14:31

- Upload gate still passes: train/test expected/have `200/200` and `211/211`, `bad_size=0`, `ALL_PASS=True`; train/test broken symlink counts remain `0/0`.
- Slurm still reports Route A `994378`, quota action-boundary `994707`, and quota boundary-heavy `994708` as `RUNNING`.
- Route A reached Epoch 51 step 50 at `14:29:32`: `Loss=0.4072`, `cls_loss=0.1972`, `reg_loss=0.2101`; latest mAP remains `64.68`; failure count `0`.
- Quota action-boundary reached Epoch 9 step 50 at `14:26:25`: `Loss=1.0113`, selector action/boundary losses `0.1279/0.1918`, `cls_loss=0.3506`, `reg_loss=0.2768`, failure count `0`.
- Quota boundary-heavy completed Epoch 9 at `14:27:13`: `Loss=1.2260`, selector action/boundary losses `0.1599/0.3197`, `cls_loss=0.3910`, `reg_loss=0.2912`, failure count `0`.
- Artifact paths unchanged: Route A active checkpoints `epoch_19.pth` and `epoch_39.pth`; quota dirs have no `epoch_*.pth` yet.
- Decision: continue all three active runs. No cleanup, selector diagnostic, first-eval gate, or severe-result gate yet.

## Active Training Monitor at 14:47

- Upload gate still passes: train/test expected/have `200/200` and `211/211`,
  `bad_size=0`, `ALL_PASS=True`; `/data` remains about `1.9T` available.
- Project Slurm jobs remain `RUNNING`: Route A `994378`, quota action-boundary
  `994707`, and quota boundary-heavy `994708`. Unrelated `scripted_gpu` jobs
  were left untouched.
- Route A latest mAP remains `64.68` with vector
  `79.90 / 75.38 / 68.19 / 57.31 / 42.63`. A raw-tail process check at
  `14:48:10` confirms it is actively validating after Epoch 51 at about
  `331/396`, so no one-hour stall/process intervention is triggered.
- Quota action-boundary reached Epoch 12 step 50 at `14:43:54`:
  `Loss=1.0457`, selector action/boundary losses `0.1279/0.1918`,
  `cls_loss=0.3666`, `reg_loss=0.2952`, failure count `0`.
- Quota boundary-heavy completed Epoch 12 at `14:43:40`, then entered Epoch 13:
  `Loss=1.1945`, selector action/boundary losses `0.1599/0.3197`,
  `cls_loss=0.3743`, `reg_loss=0.2765`, failure count `0`.
- Route A has active `epoch_19.pth` and `epoch_39.pth`; quota dirs still have
  no `epoch_*.pth`, so selector diagnostics are not runnable yet.
- Decision: continue all three project runs. No cleanup, selector diagnostic,
  first-eval gate, severe-result gate, cancellation, or new launch.

## Active Training Monitor at 14:52

- Slurm still reports project jobs `994378`, `994707`, and `994708` as
  `RUNNING`; unrelated jobs were not touched.
- Route A produced a new eval at `14:51:31`: `Average-mAP=64.82`, vector
  `80.00 / 75.39 / 68.03 / 57.80 / 42.90`. Failure count remains `0` and the
  run resumed Epoch 52.
- Interpretation: Route A is now above stratified `64.64` by `+0.18`, but still
  below uniform stride-2 `65.09` by `-0.27` and below old uniform best `65.73`
  by `-0.91`. Keep the parity caveat; this is a control result, not a quota
  selector claim.
- Quota action-boundary completed Epoch 12 at `14:46:37` with `Loss=1.0365`,
  then reached Epoch 13 step 50 at `14:49:42` with `Loss=1.0397`; selector
  action/boundary losses are `0.1276/0.1915` on that step.
- Quota boundary-heavy reached Epoch 13 step 50 at `14:46:31` with `Loss=1.1873`
  and completed Epoch 13 at `14:49:10` with `Loss=1.1961`; selector
  action/boundary losses remain around `0.1599/0.3197`.
- Quota dirs still have no `epoch_*.pth`, so selector diagnostics are not
  runnable yet. Route A active checkpoints remain `epoch_19.pth` and
  `epoch_39.pth`.
- Decision: continue all three project runs. No cleanup, selector diagnostic,
  first-eval gate, severe-result gate, cancellation, or new launch.

## Active Training Monitor at 14:57

- Upload gate still passes: train/test expected/have `200/200` and `211/211`,
  `bad_size=0`, `ALL_PASS=True`; `/data` remains about `1.9T` free.
- Slurm still reports project jobs `994378`, `994707`, and `994708` as
  `RUNNING`; unrelated jobs were left untouched.
- Route A latest mAP remains `64.82`
  (`80.00 / 75.39 / 68.03 / 57.80 / 42.90`). It completed Epoch 52 at
  `14:57:26` with `Loss=0.4169` and started Epoch 53.
- Interpretation: Route A is still a uniform-control result, not learned quota
  selector evidence: `+0.18` vs stratified `64.64`, `-0.27` vs uniform
  stride-2 `65.09`, and `-0.91` vs old uniform best `65.73`.
- Quota action-boundary reached Epoch 14 step 50 at `14:55:25`:
  `Loss=0.9881`, selector action/boundary losses `0.1282/0.1923`,
  `cls_loss=0.3408`, `reg_loss=0.2623`.
- Quota boundary-heavy completed Epoch 14 at `14:54:38`: `Loss=1.1413`,
  selector action/boundary losses `0.1599/0.3197`, `cls_loss=0.3344`,
  `reg_loss=0.2631`.
- Failure counts remain `0`; quota dirs still have no `epoch_*.pth`, so
  selector diagnostics are not runnable yet. Route A active checkpoints remain
  `epoch_19.pth` and `epoch_39.pth`.
- Contract unchanged: `50% ViT/backbone compute, dense decode`; train-only GT
  density/action/boundary auxiliary losses are not used at test time; no
  test-time GT/teacher/cache.
- Decision: continue all three project runs. No cleanup, selector diagnostic,
  first-eval gate, severe-result gate, cancellation, or new launch.

## Active Training Monitor at 15:00

- Upload gate still passes: train/test expected/have `200/200` and `211/211`,
  `bad_size=0`, `ALL_PASS=True`; `/data` remains about `1.9T` free.
- Slurm reports project jobs `994378 e2e_routeA`, `994707 e2e_quota_ab`, and
  `994708 e2e_quota_bh` still `RUNNING`. Unrelated `scripted_gpu` and `run.sh`
  jobs were only observed and not touched.
- Route A latest mAP remains `64.82`
  (`80.00 / 75.39 / 68.03 / 57.80 / 42.90`). It reached Epoch 53 step 50 at
  `15:00:23` with `Loss=0.3953`; failure count is `0`.
- Quota action-boundary completed Epoch 14 at `14:58:09`: `Loss=1.0026`,
  selector action/boundary losses `0.1279/0.1918`, `cls_loss=0.3525`,
  `reg_loss=0.2661`; failure count is `0`.
- Quota boundary-heavy reached Epoch 15 step 50 at `14:57:23`: `Loss=1.1508`,
  selector action/boundary losses `0.1595/0.3189`, `cls_loss=0.3434`,
  `reg_loss=0.2651`; failure count is `0`.
- Route A active checkpoints remain `epoch_19.pth` and `epoch_39.pth`; quota
  dirs still only have `log.json` and no `epoch_*.pth`, so selector diagnostics
  are still waiting for epoch 19.
- Contract unchanged: `50% ViT/backbone compute, dense decode`; train-only GT
  auxiliary losses remain train-only; no test-time GT/teacher/cache.
- Decision: continue all three project runs. No process intervention, cleanup,
  selector diagnostic, first-eval gate, severe-result gate, cancellation, or new
  launch.

## Active Training Monitor at 15:04

- Upload gate still passes: train/test expected/have `200/200` and `211/211`,
  `bad_size=0`, `ALL_PASS=True`; `/data` remains about `1.9T` free.
- Slurm reports project jobs `994378 e2e_routeA`, `994707 e2e_quota_ab`, and
  `994708 e2e_quota_bh` still `RUNNING`. Unrelated jobs were observed only.
- Route A latest mAP remains `64.82`; it completed Epoch 53 at `15:03:09` with
  `Loss=0.3669`, `cls_loss=0.1793`, `reg_loss=0.1876`; failure count is `0`.
- Quota action-boundary reached Epoch 15 step 50 at `15:01:02`:
  `Loss=1.0172`, selector action/boundary losses `0.1276/0.1913`,
  `cls_loss=0.3624`, `reg_loss=0.2719`; failure count is `0`.
- Quota boundary-heavy completed Epoch 15 at `15:00:18`: `Loss=1.1327`,
  selector action/boundary losses `0.1599/0.3197`, `cls_loss=0.3292`,
  `reg_loss=0.2596`; failure count is `0`.
- Route A active checkpoints remain `epoch_19.pth` and `epoch_39.pth`; quota
  dirs still only have `log.json` and no `epoch_*.pth`, so selector diagnostics
  remain gated.
- Contract unchanged: `50% ViT/backbone compute, dense decode`; train-only GT
  auxiliary losses remain train-only; no test-time GT/teacher/cache.
- Decision: continue all three project runs. No process intervention, cleanup,
  selector diagnostic, first-eval gate, severe-result gate, cancellation, or new
  launch.

## Active Training Monitor at 15:08

- Upload gate still passes: train/test expected/have `200/200` and `211/211`,
  `bad_size=0`, `ALL_PASS=True`; `/data` remains about `1.9T` free.
- Slurm reports project jobs `994378 e2e_routeA`, `994707 e2e_quota_ab`, and
  `994708 e2e_quota_bh` still `RUNNING`; unrelated jobs were observed only.
- Route A latest mAP remains `64.82`; latest train evidence remains Epoch 53
  final at `15:03:09`; failure count is `0`.
- Quota action-boundary completed Epoch 15 at `15:03:53` with `Loss=0.9885`,
  then reached Epoch 16 step 50 at `15:06:54` with `Loss=0.9685`; selector
  action/boundary losses are `0.1277/0.1915`; failure count is `0`.
- Quota boundary-heavy completed Epoch 16 at `15:06:00`: `Loss=1.1486`,
  selector action/boundary losses `0.1599/0.3198`; failure count is `0`.
- Route A active checkpoints remain `epoch_19.pth` and `epoch_39.pth`; quota
  dirs still only have `log.json` and no `epoch_*.pth`.
- Contract unchanged: `50% ViT/backbone compute, dense decode`; train-only GT
  auxiliary losses remain train-only; no test-time GT/teacher/cache.
- Decision: continue all three project runs. No selector diagnostic yet because
  quota `epoch_19.pth` is absent. No process intervention, cleanup,
  first-eval gate, severe-result gate, cancellation, or new launch.

## Active Training Monitor at 15:11

- Upload gate still passes: train/test expected/have `200/200` and `211/211`,
  `bad_size=0`, `ALL_PASS=True`; `/data` remains about `1.9T` free.
- Slurm reports project jobs `994378 e2e_routeA`, `994707 e2e_quota_ab`, and
  `994708 e2e_quota_bh` still `RUNNING`; unrelated jobs were observed only.
- Route A latest mAP remains `64.82`; latest train evidence remains Epoch 53
  final; failure count is `0`.
- Quota action-boundary latest train line remains Epoch 16 step 50 at
  `15:06:54` with `Loss=0.9685`; failure count is `0`.
- Quota boundary-heavy reached Epoch 17 step 50 at `15:09:06`: `Loss=1.1055`,
  selector action/boundary losses `0.1615/0.3229`; failure count is `0`.
- Route A active checkpoints remain `epoch_19.pth` and `epoch_39.pth`; quota
  dirs still only have `log.json` and no `epoch_*.pth`.
- Contract unchanged: `50% ViT/backbone compute, dense decode`; train-only GT
  auxiliary losses remain train-only; no test-time GT/teacher/cache.
- Decision: continue all three project runs. No selector diagnostic yet because
  quota `epoch_19.pth` is absent. No process intervention, cleanup,
  first-eval gate, severe-result gate, cancellation, or new launch.

## Active Training Monitor at 15:13

- Upload gate still passes: train/test expected/have `200/200` and `211/211`,
  `bad_size=0`, `ALL_PASS=True`; `/data` remains about `1.9T` free.
- Slurm reports project jobs `994378 e2e_routeA`, `994707 e2e_quota_ab`, and
  `994708 e2e_quota_bh` still `RUNNING`; unrelated jobs were observed only.
- Route A latest mAP remains `64.82`; latest train evidence remains Epoch 53
  final; failure count is `0`.
- Quota action-boundary completed Epoch 16 at `15:09:41`: `Loss=1.0006`,
  selector action/boundary losses `0.1279/0.1919`; failure count is `0`.
- Quota boundary-heavy latest train line remains Epoch 17 step 50 at
  `15:09:06` with `Loss=1.1055`; failure count is `0`.
- Route A active checkpoints remain `epoch_19.pth` and `epoch_39.pth`; quota
  dirs still only have `log.json` and no `epoch_*.pth`.
- Contract unchanged: `50% ViT/backbone compute, dense decode`; train-only GT
  auxiliary losses remain train-only; no test-time GT/teacher/cache.
- Decision: continue all three project runs. No selector diagnostic yet because
  quota `epoch_19.pth` is absent. No process intervention, cleanup,
  first-eval gate, severe-result gate, cancellation, or new launch.

## Active Training Monitor at 15:16

- Upload gate still passes: train/test expected/have `200/200` and `211/211`,
  `bad_size=0`, `ALL_PASS=True`; `/data` remains about `1.9T` free.
- Slurm reports project jobs `994378 e2e_routeA`, `994707 e2e_quota_ab`, and
  `994708 e2e_quota_bh` still `RUNNING`; unrelated jobs were observed only.
- Route A latest mAP remains `64.82`; latest train evidence remains Epoch 53
  final; failure count is `0`.
- Quota action-boundary reached Epoch 17 step 50 at `15:12:48`: `Loss=0.9626`,
  selector action/boundary losses `0.1292/0.1937`; failure count is `0`.
- Quota boundary-heavy completed Epoch 17 at `15:11:48`: `Loss=1.1227`,
  selector action/boundary losses `0.1598/0.3197`; failure count is `0`.
- Route A active checkpoints remain `epoch_19.pth` and `epoch_39.pth`; quota
  dirs still only have `log.json` and no `epoch_*.pth`.
- Contract unchanged: `50% ViT/backbone compute, dense decode`; train-only GT
  auxiliary losses remain train-only; no test-time GT/teacher/cache.
- Decision: continue all three project runs. No selector diagnostic yet because
  quota `epoch_19.pth` is absent. No process intervention, cleanup,
  first-eval gate, severe-result gate, cancellation, or new launch.

## Active Training Monitor at 15:19

- Upload gate still passes: train/test expected/have `200/200` and `211/211`,
  `bad_size=0`, `ALL_PASS=True`; `/data` remains about `1.9T` free.
- Slurm reports project jobs `994378 e2e_routeA`, `994707 e2e_quota_ab`, and
  `994708 e2e_quota_bh` still `RUNNING`; unrelated jobs were observed only.
- Route A latest mAP remains `64.82`; latest train evidence remains Epoch 53
  final; failure count is `0`.
- Quota action-boundary completed Epoch 17 at `15:15:32`: `Loss=0.9710`,
  selector action/boundary losses `0.1279/0.1918`, `cls_loss=0.3322`,
  `reg_loss=0.2550`; failure count is `0`.
- Quota boundary-heavy reached Epoch 18 step 50 at `15:14:44`: `Loss=1.1319`,
  selector action/boundary losses `0.1602/0.3204`, `cls_loss=0.3233`,
  `reg_loss=0.2637`; failure count is `0`.
- Route A active checkpoints remain `epoch_19.pth` and `epoch_39.pth`; quota
  dirs still only have `log.json` and no `epoch_*.pth`.
- Contract unchanged: `50% ViT/backbone compute, dense decode`; train-only GT
  auxiliary losses remain train-only; no test-time GT/teacher/cache.
- Decision: continue all three project runs. No selector diagnostic yet because
  quota `epoch_19.pth` is absent. No process intervention, cleanup,
  first-eval gate, severe-result gate, cancellation, or new launch.

## Active Training Monitor at 15:23

- Upload gate still passes: train/test expected/have `200/200` and `211/211`,
  `bad_size=0`, `ALL_PASS=True`; `/data` remains about `1.9T` free.
- Slurm reports project jobs `994378 e2e_routeA`, `994707 e2e_quota_ab`, and
  `994708 e2e_quota_bh` still `RUNNING`; unrelated jobs were observed only.
- Route A produced a new eval at `15:22:18`: `65.03` Avg-mAP, vector
  `80.25 / 75.49 / 68.09 / 57.98 / 43.32`, then resumed Epoch 54. Failure
  count remains `0`. This is `+0.39` vs stratified `64.64`, `-0.06` vs uniform
  stride-2 `65.09`, and `-0.70` vs old uniform best `65.73`; it remains a
  uniform-control result, not learned quota-selector evidence.
- Quota action-boundary completed Epoch 18 at `15:21:24`: `Loss=0.9620`,
  selector action/boundary losses `0.1279/0.1918`, `cls_loss=0.3146`,
  `reg_loss=0.2635`, then started Epoch 19. Failure count is `0`.
- Quota boundary-heavy completed Epoch 18 at `15:17:29`: `Loss=1.1239`,
  selector action/boundary losses `0.1598/0.3197`, `cls_loss=0.3179`,
  `reg_loss=0.2622`, then reached Epoch 19 step 50 at `15:20:21` with
  `Loss=1.1542`. Failure count is `0`.
- Route A active checkpoints remain `epoch_19.pth` and `epoch_39.pth`; quota
  dirs still only have `log.json` and no `epoch_*.pth`.
- Contract unchanged: `50% ViT/backbone compute, dense decode`; train-only GT
  auxiliary losses remain train-only; no test-time GT/teacher/cache.
- Decision: continue all three project runs. Recheck quota checkpoint shortly;
  run selector-only diagnostics only after quota `epoch_19.pth` appears. No
  process intervention, cleanup, first-eval gate, severe-result gate,
  cancellation, or new launch.

## Quota Checkpoint Watch at 15:28

- Local diagnostic script `logs/run_quota_epoch19_diag.py` passed
  `python -m py_compile`.
- Quota action-boundary still has no `epoch_*.pth`; it reached Epoch 19 step 50
  at `15:24:18` with `Loss=0.9823`, selector action/boundary losses
  `0.1289/0.1933`, and failure count `0`.
- Quota boundary-heavy wrote
  `/data/home/sczc063/run/yuzibo/e2e_runs/exps/e2e_quota_boundary_heavy_20260529_1329/gpu1_id0/checkpoint/epoch_19.pth`
  at `15:23:08`; it completed Epoch 19 at `15:23:06` with `Loss=1.1371`, then
  reached Epoch 20 step 50 at `15:25:54` with `Loss=1.0901`. Failure count is
  `0`.
- No quota mAP is available yet. Training jobs continue; no cleanup.
- Diagnostic contract: selector-only, validation GT used only for offline
  labeling of selected positions, no test-time GT/teacher/cache.
- Decision: recheck action-boundary checkpoint shortly. If still absent, submit
  boundary-heavy diagnostic first; submit action-boundary diagnostic after its
  `epoch_19.pth` appears. No severe-result gate, cancellation, or new long run.

## Boundary-Heavy Selector Diagnostic Submitted at 15:31

- Uploaded local `logs/run_quota_epoch19_diag.py` to
  `~/run/yuzibo/e2e_runs/selector_diagnostics/run_quota_epoch19_diag.py`.
- Submitted Slurm diagnostic job `994846 e2e_bh_diag` with script
  `~/run/yuzibo/OpenTAD_BATA_Clean/scripts/run_quota_bh_epoch19_diag_20260529.sbatch`.
- Target checkpoint:
  `/data/home/sczc063/run/yuzibo/e2e_runs/exps/e2e_quota_boundary_heavy_20260529_1329/gpu1_id0/checkpoint/epoch_19.pth`.
- Diagnostic is selector-only and samples 64 validation items. It reports
  context/action/boundary logit stats, final selected-position distribution,
  action/boundary coverage, and offline boundary recall.
- Contract: validation GT is used only for offline labeling of selected
  positions; no mAP evaluation, no training, no checkpoint edits, and no
  test-time GT/teacher/cache. Active training jobs were not touched.
- Decision: monitor job `994846` for `summary.json`; continue watching
  action-boundary `epoch_19.pth` and submit the matching diagnostic when
  available.

## Action-Boundary Checkpoint Ready at 15:31

- Quota action-boundary wrote
  `/data/home/sczc063/run/yuzibo/e2e_runs/exps/e2e_quota_action_boundary_20260529_1329/gpu1_id0/checkpoint/epoch_19.pth`
  at `15:27:09`.
- AB completed Epoch 19 at `15:27:07` with `Loss=0.9922`, selector
  action/boundary losses `0.1279/0.1918`, `cls_loss=0.3470`, `reg_loss=0.2612`;
  failure count remains `0`.
- Boundary-heavy diagnostic job `994846` is running and has processed `16/64`
  samples; no `summary.json` yet.
- Training jobs `994378`, `994707`, and `994708` remain running. No quota mAP
  yet.
- Decision: submit the matching AB selector diagnostic now that `epoch_19.pth`
  exists.

## Action-Boundary Selector Diagnostic Submitted at 15:33

- Submitted Slurm diagnostic job `994848 e2e_ab_diag` with script
  `~/run/yuzibo/OpenTAD_BATA_Clean/scripts/run_quota_ab_epoch19_diag_20260529.sbatch`.
- Target checkpoint:
  `/data/home/sczc063/run/yuzibo/e2e_runs/exps/e2e_quota_action_boundary_20260529_1329/gpu1_id0/checkpoint/epoch_19.pth`.
- It uses the same selector-only 64-sample offline position diagnostic as the
  boundary-heavy run and reports quota-channel logit stats plus selected-frame
  action/boundary coverage.
- Contract: validation GT is used only for offline labeling; no mAP evaluation,
  no training, no checkpoint edits, and no test-time GT/teacher/cache. Active
  training jobs were not touched.
- Decision: monitor diagnostic jobs `994846` and `994848` for `summary.json`,
  then compare selected-frame distributions against the Pro gates.

## Quota Selector Diagnostics Completed at 15:38

- Boundary-heavy diagnostic job `994846` completed in `00:03:18`.
  Summary:
  `~/run/yuzibo/e2e_runs/selector_diagnostics/e2e_quota_boundary_heavy_epoch19_posdiag_20260529_153054/summary.json`.
- Action-boundary diagnostic job `994848` completed in about `00:04:35`.
  Summary:
  `~/run/yuzibo/e2e_runs/selector_diagnostics/e2e_quota_action_boundary_epoch19_posdiag_20260529_153350/summary.json`.
- AB epoch-19: mean abs delta from uniform `2.66`, selected action fraction
  `0.3327`, selected boundary<=4 fraction `0.1392`, boundary recall@4 `1.0`,
  action logit std `0.0023`, boundary logit std `0.0006`.
- BH epoch-19: mean abs delta from uniform `3.23`, selected action fraction
  `0.3324`, selected boundary<=4 fraction `0.1405`, boundary recall@4 `0.9958`,
  action logit std `0.0101`, boundary logit std `0.0018`.
- Gate interpretation: both pass mean-delta and action-fraction gates, but both
  fail final selected boundary<=4 `>=0.17` and boundary-channel logit std
  `>=0.01`; BH alone barely passes the action-logit-std gate. The boundary
  chunk itself is not more boundary-concentrated than the final mixture.
- Current interpretation: the selector is end-to-end and mildly non-uniform, but
  by epoch 19 it has not learned the desired boundary-focused frame distribution.
  This is diagnostic-only evidence, not mAP evidence.
- Active trainings remain healthy with failure count `0`: Route A latest mAP
  `65.03`; AB reached Epoch 21 step 50 at `15:36:05` with `Loss=0.9383`; BH
  completed Epoch 21 at `15:34:04` with `Loss=1.0826`.
- Decision: continue AB/BH to first mAP and send these diagnostics to Pro for
  discussion of boundary-channel flatness, quota sorting dilution, and aggressive
  next variants. No severe-result gate is triggered yet.

## Pro Discussion Started at 15:42

- Prompt:
  `logs/oracle_pro_e2e_quota_diag_discussion_prompt_20260529.md`.
- Runner:
  `logs/run_oracle_pro_e2e_quota_diag_discussion_20260529.ps1`.
- Expected answer:
  `logs/oracle_pro_e2e_quota_diag_discussion_20260529.txt`.
- Stdout/stderr:
  `logs/oracle_pro_e2e_quota_diag_discussion_20260529.stdout.txt` and
  `logs/oracle_pro_e2e_quota_diag_discussion_20260529.err.txt`.
- Local PowerShell PID: `7224`.
- Route requested: Oracle browser `gpt-5.5-pro`.
- Prompt includes the end-to-end gradient contract, Route A `65.03`, AB/BH
  epoch-19 selector diagnostics, failed boundary-channel gates, and requests
  root-cause ranking plus 2-3 aggressive parallel routes under the 50% budget
  and no-test-GT protocol.
- Decision: continue monitoring first mAP while Pro runs. Do not treat this Pro
  discussion as complete until stdout confirms `Extended Pro` and the answer is
  substantive.

## Active Training Monitor at 15:43

- Pro discussion still running under local PID `7224`; stdout shows Oracle
  browser session `e2e-quota-diag-discussion-20260529`, requested
  `gpt-5.5-pro`, browser slot acquired, waiting for response.
- Remote project jobs remain `RUNNING`: Route A `994378`, quota AB `994707`,
  quota BH `994708`. Unrelated jobs were observed only.
- Route A latest mAP remains `65.03` and completed Epoch 55 at `15:33:52` with
  `Loss=0.4190`; failure count is `0`.
- Quota AB completed Epoch 21 at `15:38:54` with `Loss=0.9315`, then reached
  Epoch 22 step 50 at `15:41:57` with `Loss=0.9587`; failure count is `0`.
- Quota BH completed Epoch 22 at `15:39:26` with `Loss=1.1138`; failure count
  is `0`.
- Quota first mAP is still absent. Active checkpoints remain Route A
  `epoch_19/39` and AB/BH `epoch_19`; no cleanup.
- Contract unchanged: 50% backbone budget, train-only GT auxiliary, no test-time
  GT/teacher/cache.
- Decision: continue monitoring. No process intervention, cleanup,
  severe-result gate, cancellation, or new long run.

## Active Training Monitor at 15:45

- Pro PID `7224` remains active; stdout shows the Oracle browser session still
  waiting after about `2m30s`.
- Slurm project jobs remain `RUNNING`: Route A `994378`, quota AB `994707`, and
  quota BH `994708`; unrelated jobs were observed only.
- Route A latest mAP remains `65.03`; latest train evidence remains Epoch 55
  final `Loss=0.4190`; failure count is `0`.
- Quota AB reached Epoch 22 step 50 at `15:41:57`: `Loss=0.9587`, selector
  action/boundary losses `0.1276/0.1914`; failure count is `0`.
- Quota BH reached Epoch 23 step 50 at `15:42:19`: `Loss=1.1124`, selector
  action/boundary losses `0.1606/0.3213`; failure count is `0`.
- No quota mAP yet. `/data` remains safe with about `1.9T` available.
- Contract unchanged: 50% backbone budget, train-only GT auxiliary, no test-time
  GT/teacher/cache. No cleanup.
- Decision: continue monitoring Pro and quota first mAP. No intervention,
  severe-result gate, cancellation, or new long run.

## Active Training Monitor at 15:47

- Pro PID `7224` remains active; Oracle stdout shows the session waiting at
  about `5m`.
- Route A latest mAP remains `65.03`; no new Route A train line beyond Epoch 55
  final.
- Quota AB completed Epoch 22 at `15:44:39` with `Loss=0.9612`, selector
  action/boundary losses `0.1279/0.1919`; failure count is `0`.
- Quota BH completed Epoch 23 at `15:44:58` with `Loss=1.0988`, selector
  action/boundary losses `0.1599/0.3197`; failure count is `0`.
- No quota mAP yet. Contract unchanged; no cleanup.
- Decision: continue monitoring Pro answer and quota first mAP. No intervention,
  severe-result gate, cancellation, or new long run.

## Active Training Monitor at 15:51

- Pro PID `7224` remains active; Oracle stdout shows the session waiting at
  about `9m`.
- Project jobs `994378`, `994707`, and `994708` remain `RUNNING`.
- Route A latest mAP remains `65.03` with vector
  `80.25 / 75.49 / 68.09 / 57.98 / 43.32`; no new train line beyond Epoch 55
  final, failure count `0`.
- Quota AB reached Epoch 23 step 50 at `15:47:44`: `Loss=0.9394`, selector
  action/boundary losses `0.1285/0.1927`; failure count is `0`.
- Quota BH reached Epoch 24 step 50 at `15:47:49`: `Loss=1.1001`, selector
  action/boundary losses `0.1592/0.3183`; failure count is `0`.
- Quota first mAP is still absent. Contract unchanged: 50% backbone budget,
  train-only GT auxiliary losses, no test-time GT/teacher/cache. No cleanup.
- Decision: continue monitoring Pro answer and quota first mAP. No intervention,
  severe-result gate, cancellation, cleanup, or new long run.

## Pro Discussion Returned and Route A Update at 15:54

- Oracle browser `gpt-5.5-pro` completed in about `11m32s` and saved the answer
  to `logs/oracle_pro_e2e_quota_diag_discussion_20260529.txt`.
- Pro verdict: the current quota selector is end-to-end because detection loss
  can flow through selected raw-frame values and positions, but epoch-19
  diagnostics do not show learned boundary-aware allocation. The behavior is
  still mild non-uniform coverage with weak effective boundary credit
  assignment.
- Accepted Pro route decision: keep AB/BH to first mAP; immediately implement
  one aggressive BoundarySharp-ST run; add epoch-39 diagnostics for gradient
  audit, per-channel chunk attribution, same-checkpoint uniform/shuffle/channel
  replacement ablations, and geometry safety.
- Route A produced a new eval at `15:52:45`: `65.05` Avg-mAP, vector
  `80.51 / 75.45 / 68.25 / 58.15 / 42.89`. This is near the uniform stride-2
  reference `65.09` but remains a uniform-control result, not learned selector
  evidence.
- AB/BH remain running with no first mAP yet. Upload marker remains
  `ALL_PASS=True`; `/data` remains about `1.9T` free. Contract unchanged:
  50% backbone budget, train-only GT auxiliary losses, no test-time
  GT/teacher/cache.
- Decision: implement BoundarySharp-ST locally and run review gates before
  deployment. Continue AB/BH to first mAP.

## BoundarySharp-ST Self-Check at 16:01

- Self-check report:
  `research-wiki/experiments/BATA_BOUNDARYSHARP_ST_SELF_CHECK_20260529.md`.
- Changed files:
  `opentad/models/selectors/temporal_density_selector.py`,
  `configs/adatad/thumos/e2e_rawdensel_384of768_boundarysharp_st_adapter.py`,
  and `tests/test_e2e_raw_frame_selector_contracts.py`.
- Implemented optional `st_topk` quota position mode. Forward uses hard top-k
  action/boundary positions, while backward keeps the soft-quantile surrogate
  through `hard.detach() - soft.detach() + soft`.
- New BoundarySharp-ST config keeps 384-of-768 fixed budget and sets quota
  `96/96/192`, stronger train-only boundary target shaping, and no aggregate GT
  density loss.
- Local checks: `py_compile` PASS; Windows pytest `1 passed, 19 skipped`;
  `git diff --check` PASS with LF/CRLF warnings only.
- Contract unchanged for deployment: no test-time GT/teacher/cache, no detector
  head or post-processing change.
- Decision: do not sync or launch yet. Run GPT-5.5 Pro line-by-line review,
  then Gemini CLI and DeepSeek CLI.

## BoundarySharp-ST Pro Review and Fixes at 16:19

- GPT-5.5 Pro implementation review returned `WARN` with no blocking code
  findings after about `13m24s`.
- Review output:
  `logs/oracle_pro_boundarysharp_st_review_20260529.txt`.
- Accepted fixes applied:
  explicit prefix-mask/all-false-mask contract checks, non-negative quota
  target-weight validation, ST invalid-tail/short-valid tests, and merged-config
  quota sum check based on `selection_mode`.
- Post-fix local checks passed: `py_compile`; Windows pytest
  `1 passed, 20 skipped`; `git diff --check` with LF/CRLF warnings only.
- Active monitor at `16:19:32`: jobs `994378`, `994707`, and `994708` still
  `RUNNING`. Route A latest mAP remains `65.05` and is validating after Epoch
  57. AB reached Epoch 28 step 50 with `Loss=0.8927`; BH reached Epoch 29 step
  50 with `Loss=1.0498`. No quota mAP yet.
- Decision: run focused Pro fix re-review because code changed after the first
  Pro review; then proceed to Gemini/DeepSeek if accepted.

## BoundarySharp-ST Final Pro Fixes and Route A Update at 16:31

- Focused GPT-5.5 Pro fix re-review returned `WARN` with no code blocker.
- Final accepted fixes applied:
  `quota_uniform_target_weight` non-negative validation, negative-weight
  constructor test, self-check pytest count correction, and merged-config
  `detach_gt_remap=True` assertion.
- Final local checks passed: `py_compile`; Windows pytest
  `1 passed, 21 skipped`; `git diff --check` with LF/CRLF warnings only.
- Route A produced a new eval at `16:23:28`: `65.37` Avg-mAP, vector
  `80.68 / 75.76 / 68.45 / 58.52 / 43.41`. This is above the uniform stride-2
  reference `65.09` and below the old exact-uniform control `65.73`; it remains
  a uniform-control result, not learned selector evidence.
- AB/BH still have no first mAP: AB reached Epoch 30 step 50 at `16:28:26`
  with `Loss=0.9009`; BH reached Epoch 32 start after Epoch 31 final at
  `16:29:15` with `Loss=1.0279`.
- Decision: proceed to Gemini CLI and DeepSeek CLI review. Continue AB/BH to
  first mAP.

## BoundarySharp-ST Gemini Review at 16:34

- Gemini CLI model `gemini-3-pro-preview` exited `0` and returned `PASS`.
- Output:
  `logs/gemini3_pro_preview_boundarysharp_st_review_20260529.txt`.
- Stderr:
  `logs/gemini3_pro_preview_boundarysharp_st_review_20260529.err.txt`, with
  terminal-color warning, ripgrep fallback, and skill-conflict notices only.
- No blocking findings or required fixes. Gemini judged ST top-k, mask contract,
  GT/teacher leakage checks, and 384-of-768 fixed-budget config aligned with the
  experiment purpose.
- Decision: proceed to Claude CLI DeepSeek secondary review. No sync/deploy
  until DeepSeek and N16R4 preflight pass.

## BoundarySharp-ST DeepSeek Review at 16:40

- Claude CLI model `deepseek-v4-pro` exited `0` and returned full stdout
  `PASS`.
- stdout:
  `logs/claude_deepseek_v4_pro_boundarysharp_st_review_20260529.txt`.
- stderr/debug:
  `logs/claude_deepseek_v4_pro_boundarysharp_st_review_20260529.err.txt` and
  `logs/claude_deepseek_v4_pro_boundarysharp_st_review_20260529.debug.log`.
- No blocking findings and no required fixes. DeepSeek verified no test-time
  GT/teacher/cache, 768-to-384 fixed-budget contract, ST gradient path,
  prefix-mask contract, config/launcher consistency, and attribution to selector
  changes.
- Decision: commit/sync only reviewed files, run N16R4 Linux preflight, then
  launch one BoundarySharp-ST GPU job if preflight passes.
