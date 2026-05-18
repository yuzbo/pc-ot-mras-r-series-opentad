# Agent Notes

## Current Remote Training

- Use the old AutoDL servers for the active THUMOS14 Adapter + ActionFormer runs:
  - `ssh -p 35407 root@connect.cqa1.seetacloud.com`
  - `ssh -p 25876 root@connect.cqa1.seetacloud.com`
- Active remote project path: `/root/autodl-tmp/OpenTAD_Back_check`.
- Completed config on both servers:
  `configs/adatad/thumos/input_random_fixed_50pct_adapter_quality_rescore_detached.py`.
- 2026-05-17 process check: both servers are idle. There are no `screen`
  sessions, no `tools/train.py` / `tools/test.py` / `torchrun` processes, and
  both RTX 4080 SUPER GPUs report 0 MiB memory use and 0% utilization.
- Do not launch the next long training run until the quality-failure discussion,
  local checks, and read-only code review requirements are satisfied.
- Small isolated `non-finite gradients` events were expected training noise for
  this run. Record them, but do not treat a single event as a blocker if loss
  continues normally.

## Review Policy

- Use Claude CLI for direction discussion and code review.
- Do not use the `claude-review` MCP path unless the user explicitly changes this rule.
- After code review and local fixes, run local checks before any deployment.
- Before launching any innovative experiment, first complete a Claude CLI direction-and-implementation discussion and write the discussion record here.
- The discussion must include the agent's own reasoning and reflection, not just Claude's recommendation. Claude's advice is an input to challenge and refine, not a command to follow blindly.
- A valid discussion record must separate:
  - Claude's concrete claims or objections.
  - The agent's own agreement, disagreement, or partial acceptance, grounded in observed metrics, training contracts, tensor/label semantics, and leakage risk.
  - The final decision rule and the falsifiable metric gate for the next run.
  - What will not be changed in the next experiment, so one-variable ablations remain clean.
- Do not mark a Claude exchange as complete if the agent only copies Claude's conclusion without independent critique.
- If Claude CLI returns a transient error such as API 429, do not treat the discussion as complete and do not launch the innovative experiment until a later successful discussion.

## Current Direction

- Current active experiment: detached quality reranking on Adapter + ActionFormer while keeping random-fixed 50% input unchanged.
- Historical anchors:
  - random-fixed Adapter baseline: about 61.95 Average-mAP.
  - NMS/EMA audited baseline: about 63.75-63.85 Average-mAP.
  - oracle boundary dense Adapter: about 75.50 Average-mAP.
- Next candidate ablation prepared locally but not deployed:
  `configs/adatad/thumos/input_random_fixed_50pct_adapter_quality_positive_maxiou_posonly.py`.
  This uses `target_mode="positive_max_iou"`, `negative_weight=0.0`, and `loss_normalizer="positive"` to avoid dense negative BCE pressure on the quality head.

## Claude CLI Discussion Record

- A formal Claude CLI discussion is required before launching `input_random_fixed_50pct_adapter_quality_positive_maxiou_posonly.py` or any other new innovative experiment.
- 2026-05-16: three Claude CLI attempts were made for the positive-max-IoU direction discussion:
  - full direction prompt: API 429 peak-load rejection.
  - shortened direction prompt: API 429 peak-load rejection.
  - minimal availability probe: socket connection closed unexpectedly.
- Because no successful Claude CLI discussion completed, the next innovative experiment must remain pending. Continue supervising the already-running assigned-IoU quality experiments only.

### 2026-05-16 Successful Quality-Target Discussion

- Later Claude CLI discussion completed successfully.
- Claude's main objection: `positive_max_iou` with positive-only supervision and bias-initialized quality near 0.99 can leave background locations high-quality, making quality fusion inert or poorly calibrated.
- Agent reflection:
  - This objection is valid for background calibration: if negatives are not supervised, score fusion may not suppress background proposals.
  - However, dense max-IoU on all valid points is also not automatically correct, because negative classification locations can decode proposals that overlap GT; forcing all such locations through dense quality BCE may conflict with the existing assignment/classification contract.
  - The cleaner next training ablation is therefore not to switch to dense max-IoU immediately, but to keep the assigned-IoU target and reduce negative pressure with `negative_weight=0.25` plus a weighted normalizer. This changes one variable: negative quality supervision strength.
- Final gate:
  - If current detached-quality run is below the random-fixed Adapter baseline (~61.95 Average-mAP), first re-evaluate the same checkpoint with `score_alpha=0` to separate training harm from inference fusion harm.
  - If it is between ~61.95 and the NMS/EMA audited baseline (~63.75-63.85), launch the reduced-negative assigned-IoU training ablation.
  - If it reaches ~63.85-65.20, run inference-only `score_alpha` sweeps before any new training.
  - If it exceeds ~65.20, lock the config and measure seed variance.
- Current decision: do not deploy `positive_max_iou_posonly` yet. Continue the two active runs until first eval, then apply the gate above.

### 2026-05-17 Gemini CLI Discussion After Failed Quality Run

- The detached quality-rescore run completed far below baseline:
  - Server 35407 final Average-mAP: 50.05.
  - Server 25876 final Average-mAP: 49.87.
- Re-evaluating the same final checkpoints with `score_alpha=0.0` did not
  recover performance:
  - Server 35407 alpha-zero Average-mAP: 49.65.
  - Server 25876 alpha-zero Average-mAP: 49.79.
- Therefore the failure is not primarily inference-time `quality^alpha` fusion;
  the trained checkpoint itself is degraded.
- Gemini CLI was run as requested by the user. Long prompts either drifted into
  unrelated repository context or attempted unavailable tool calls, so the
  successful discussion used a compact single-turn prompt with `gemini.cmd
  --skip-trust -e none --output-format text -p "<prompt>"`.
- Gemini's usable recommendations:
  - Do not launch `positive_max_iou_posonly` next, because positive-only
    supervision can leave background proposals high-quality and poorly
    calibrated.
  - Run a neutral branch diagnostic: quality head enabled, `loss_weight=0.0`,
    `score_alpha=0.0`, from scratch.
  - In parallel, run reduced-negative assigned-IoU with
    `negative_weight=0.25` and `loss_normalizer="weighted"`; first evaluate with
    `score_alpha=0.0`, then sweep alpha only if the checkpoint recovers the
    baseline band.
- Agent reflection:
  - Gemini's "gradient leakage" warning is a check item, not established
    evidence; the known code path uses `reg_feat.detach()`.
  - The stronger current hypothesis is that the added quality branch perturbs
    training through loss scale, optimizer/grad clipping, AMP/GradScaler, EMA,
    DDP/static graph behavior, or dense negative quality BCE.
  - The next two-server package should isolate branch integration effects and
    reduced-negative quality supervision before any positive-only target.
- Full record:
  `../research-wiki/experiments/ADAPTER_QUALITY_FAILURE_GEMINI_DISCUSSION_20260517.md`.

### 2026-05-18 Launch Review Update

- Gemini CLI read-only launch review of the final neutral/reduced-negative
  patch returned `LAUNCH_OK`.
- Claude CLI review is still unavailable because the last attempt failed with
  quota error `401 {"message":"额度不足"}`.
- GPT-5-Pro is configured but expensive. Use it only for unresolved critical
  blockers or one-shot comprehensive arbitration with complete context; do not
  use repeated short probe calls for this track.
- Current launch decision: proceed with the two-server package after remote
  preflight and launcher self-checks:
  - 35407: `input_random_fixed_50pct_adapter_quality_neutral_loss0_alpha0`
  - 25876: `input_random_fixed_50pct_adapter_quality_assigned_neg025_weighted_alpha0`

### 2026-05-18 Active Runs

- 35407 launched in screen `adapter_quality_neutral`.
  - Config:
    `configs/adatad/thumos/input_random_fixed_50pct_adapter_quality_neutral_loss0_alpha0.py`
  - Log:
    `logs/input_random_fixed_50pct_adapter_quality_neutral_loss0_alpha0_20260518_011721.log`
  - First epoch sanity:
    `Loss=2.4231 cls_loss=1.3694 reg_loss=1.0537 quality_loss=0.0000`.
- 25876 launched in screen `adapter_quality_neg025`.
  - Config:
    `configs/adatad/thumos/input_random_fixed_50pct_adapter_quality_assigned_neg025_weighted_alpha0.py`
  - Log:
    `logs/input_random_fixed_50pct_adapter_quality_assigned_neg025_weighted_alpha0_20260518_011722.log`
  - First epoch sanity:
    `Loss=2.8682 cls_loss=1.3693 reg_loss=1.0537 quality_loss=0.4451`.
- Monitor gates:
  - First validation starts at epoch 40 and then every 2 epochs.
  - If either run falls below 30 Average-mAP at first validation, stop that run
    and audit implementation/training loop.
  - If neutral finishes below 60 Average-mAP, treat quality-branch integration
    as unsafe and do not interpret quality-supervision ablations as modeling
    evidence.
  - If neutral recovers and neg025 reaches the random-fixed baseline band,
    perform inference-only `score_alpha` sweep before any further training.
- Consolidated route and gate document:
  `../research-wiki/experiments/ADAPTER_ACTIONFORMER_ROADMAP_20260518.md`.
- Training-system audit:
  `../research-wiki/experiments/ADAPTER_QUALITY_TRAINING_SYSTEM_AUDIT_20260518.md`.
- 2026-05-18 first validation update:
  - Use local helper:
    `powershell -ExecutionPolicy Bypass -File logs/monitor_adapter_quality_active.ps1 -Tail 20`.
  - Remote mAP collector:
    `powershell -ExecutionPolicy Bypass -File logs/collect_adapter_quality_remote_metrics.ps1 -Tail 2600`.
  - The first validation actually triggered after training epoch `[041]`, not
    immediately after `[040]`, because `tools/train.py` checks
    `epoch >= val_start_epoch` and `(epoch + 1) % val_eval_interval == 0`.
  - 35407 neutral first validation at 2026-05-18 04:34:17:
    `Average-mAP=37.33`, mAP@0.3/0.4/0.5/0.6/0.7 =
    `60.03/50.49/38.20/25.48/12.44`.
  - 25876 neg025 first validation at 2026-05-18 04:38:56:
    `Average-mAP=37.71`, mAP@0.3/0.4/0.5/0.6/0.7 =
    `60.54/51.02/38.67/25.37/12.96`.
  - Second validations:
    - 35407 neutral at 2026-05-18 05:00:36:
      `Average-mAP=38.81`, mAP@0.3/0.4/0.5/0.6/0.7 =
      `60.90/51.69/40.10/27.37/13.97`.
    - 25876 neg025 at 2026-05-18 05:07:16:
      `Average-mAP=39.29`, mAP@0.3/0.4/0.5/0.6/0.7 =
      `62.02/52.12/40.62/27.62/14.07`.
  - Audit finding: these runs used `batch_size=8`, so each training epoch had
    only `24` iterations. The healthy `input_random_fixed_50pct_adapter.log`
    baseline used `batch_size=2`, `99` iterations/epoch, and reached `61.95`
    Average-mAP at the first eval. This invalidates the two quality runs as
    evidence about quality supervision.
  - Both quality runs were stopped after preserving logs/checkpoints:
    `screen -S adapter_quality_neutral -X quit` and
    `screen -S adapter_quality_neg025 -X quit`.
  - Local fix landed as commit
    `a5e476c fix adapter quality batch size contract`: restored
    `configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py` to
    `train/val/test batch_size=2`, and made
    `scripts/run_adapter_quality_rescore.sh` assert `EXPECT_BATCH_SIZE=2`.
  - The invalid bs8 remote output directories were preserved by renaming:
    - 35407:
      `input_random_fixed_50pct_adapter_quality_neutral_loss0_alpha0_invalid_bs8_20260518`
    - 25876:
      `input_random_fixed_50pct_adapter_quality_assigned_neg025_weighted_alpha0_invalid_bs8_20260518`
  - Remote `CHECK_ONLY=1` passed on both servers and printed
    `batch_size= 2 2 2`.
  - Relaunched clean bs2 diagnostics at about 2026-05-18 06:17:
    - 35407 screen `adapter_quality_neutral`, log
      `logs/input_random_fixed_50pct_adapter_quality_neutral_loss0_alpha0_20260518_061722.log`.
    - 25876 screen `adapter_quality_neg025`, log
      `logs/input_random_fixed_50pct_adapter_quality_assigned_neg025_weighted_alpha0_20260518_061721.log`.
  - First epoch contract is now correct:
    - neutral:
      `[000][00050/00099]` and `[000][00099/00099]`,
      `quality_loss=0.0000`;
    - neg025:
      `[000][00050/00099]` and `[000][00099/00099]`,
      `quality_loss` about `0.43`.
  - Both clean bs2 runs recorded early non-finite-gradient skips at epoch 0
    iter 17 and iter 98 on base head parameters, then continued to epoch 1.
    Treat this as a monitor item; stop only if it repeats frequently or loss
    diverges.
  - 25876 disk remains tight at about 11G available on `/root/autodl-tmp`.
  - External review note: Gemini CLI and Claude CLI attempts timed out during
    this relaunch; direct `gpt-5-pro` arbitration via the configured endpoint
    also returned no model content because the remote closed the connection.
    No further expensive retry was made.
  - 2026-05-18 07:20 follow-up: `llm-chat` is configured for `gpt-5-pro`
    through the Responses API (`LLM_API_STYLE=responses`,
    `LLM_REASONING_EFFORT=high`, `LLM_MAX_TOKENS=8192`), but do not call it
    except for a single critical arbitration with complete context.
  - 2026-05-18 07:20 Gemini CLI follow-up was attempted twice for current
    direction discussion; both replies ignored the supplied experimental
    context and produced no usable research critique. Do not count these as
    external review completion.
  - 2026-05-18 07:47 GPT-5.5 xhigh read-only advisor flagged a queue-safety
    issue: waiting for a screen to disappear would also trigger after a crash,
    manual stop, or invalid run. Accepted fix: queued screens now also wait for
    explicit gate-approval sentinel files before launching follow-up training.
  - Alpha-sweep command generator is prepared at
    `logs/prepare_adapter_quality_alpha_sweep.ps1`; use only after neg025
    recovers baseline gate.
  - Local mAP parser for copied logs:
    `logs/parse_adapter_quality_metrics.ps1`.
  - Remote mAP collector:
    `logs/collect_adapter_quality_remote_metrics.ps1`.
  - Gate evaluator:
    `logs/evaluate_adapter_quality_gate.ps1`; pipe collector output into it to
    classify neutral preservation and neg025 alpha-sweep eligibility.
  - Optional first-eval watcher:
    `logs/watch_adapter_quality_first_eval.ps1`; it now runs the gate evaluator
    automatically when `Average-mAP` appears.
  - Next scheduled checkpoint is expected after epoch 49.
  - Execution-control report:
    `../research-wiki/experiments/ADAPTER_ACTIONFORMER_EXECUTION_CONTROL_20260518.md`.

### 2026-05-18 Next Performance Queue

- Clean bs2 quality diagnostics are still running and have not reached first
  validation yet. Latest monitor around 07:13 shows both runs around epoch 13,
  still with `00099` iterations per epoch and no additional non-finite events
  beyond the two early epoch-0 skips.
- Do not stop or replace them before first eval unless they crash or loss
  diverges; the neutral preservation gate determines whether the quality-head
  route is interpretable.
- Queued next experiments after the current quality screens finish:
  - 35407 screen `adapter_pseudo_snap_q64_after_quality`.
    - Waits for `adapter_quality_neutral`.
    - Runs Adapter-side pseudo-boundary snap q64 via
      `scripts/run_adapter_pseudo_boundary_snap_pair.sh` with
      `START_INDEX=1 END_INDEX=1 SKIP_CACHE_BUILD=1`.
    - Remote files, teacher checkpoint, and cache are present; `CHECK_ONLY=1`
      passed.
    - Important: the launcher does not implement `WAIT_SCREENS`; the active
      screen is a wrapper that explicitly waits for `adapter_quality_neutral`
      before calling `CHECK_ONLY=1` and then the launcher.
    - Historical pseudo-boundary snap q32/q64 logs used the old bs8/24-iter
      contract, so their low result is not decisive for the restored bs2 run.
    - 2026-05-18 07:18 hardcopy confirms the wrapper is still waiting for
      `adapter_quality_neutral`; it has not started the queued training early.
    - 2026-05-18 07:47 queue was restarted with
      `scripts/wait_for_screen_and_gate_then_run.sh`; after
      `adapter_quality_neutral` exits it still waits for:
      `/root/autodl-tmp/OpenTAD_Back_check/gate_approvals/adapter_pseudo_snap_q64_after_quality.ok`.
      Do not create that file until the quality gate is written and q64 is
      explicitly approved.
    - 2026-05-18 08:50 local commit
      `1b042d8 guard pseudo boundary cache manifests` added manifest
      provenance checks to `scripts/run_adapter_pseudo_boundary_snap_pair.sh`.
      This is not yet synced to 35407 because the AutoDL SSH gateway is closing
      KEX before authentication. Before approving q64, sync this launcher and
      rerun `CHECK_ONLY=1 START_INDEX=1 END_INDEX=1 SKIP_CACHE_BUILD=1`.
  - 25876 screen `adapter_regloss15_after_quality`.
    - Waits for `adapter_quality_neg025`.
    - Runs ActionFormer-side `loss_weight=1.5` localization calibration via
      `scripts/run_adapter_actionformer_regloss.sh`.
    - Config and launcher were synced to 25876; `CHECK_ONLY=1` passed.
    - 2026-05-18 07:14 hardcopy confirms the wrapper is still waiting for
      `adapter_quality_neg025`; it has not started the queued training early.
    - 2026-05-18 07:47 queue was restarted with
      `scripts/wait_for_screen_and_gate_then_run.sh`; after
      `adapter_quality_neg025` exits it still waits for:
      `/root/autodl-tmp/OpenTAD_Back_check/gate_approvals/adapter_regloss15_after_quality.ok`.
      Do not create that file until the quality gate is written and regloss15
      is explicitly approved.
- Route summary and problem analysis:
  `../research-wiki/experiments/ADAPTER_ACTIONFORMER_NEXT_ROUTES_20260518.md`.
