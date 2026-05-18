# Adapter + ActionFormer Next Routes, 2026-05-18

## Current Objective

Improve THUMOS14 Adapter + ActionFormer performance on the two AutoDL servers
while keeping experiment evidence interpretable. The immediate rule is:
do not mix quality-head conclusions with new Adapter/ActionFormer changes until
the clean batch-size-2 quality diagnostics reach validation.

## Completed Evidence

Baseline anchors:

- Random-fixed Adapter baseline: `61.95` first-eval Average-mAP; final baseline
  around `63.77`.
- NMS/EMA audited baseline: about `63.75-63.85`.
- Oracle boundary dense Adapter upper anchor: about `75.50-77.62`.

Invalidated evidence:

- The first neutral/reduced-negative quality runs produced `37-39` early
  Average-mAP, but used `batch_size=8` and only `24` train iterations per
  epoch.
- Healthy Adapter uses `batch_size=2` and `99` iterations per epoch.
- Those low-mAP runs are therefore invalid evidence about quality supervision.

Clean diagnostics currently running:

- 35407: `adapter_quality_neutral`
  - config:
    `configs/adatad/thumos/input_random_fixed_50pct_adapter_quality_neutral_loss0_alpha0.py`
  - log:
    `logs/input_random_fixed_50pct_adapter_quality_neutral_loss0_alpha0_20260518_061722.log`
  - verified epoch-0 contract: `[00050/00099]` and `[00099/00099]`.
- 25876: `adapter_quality_neg025`
  - config:
    `configs/adatad/thumos/input_random_fixed_50pct_adapter_quality_assigned_neg025_weighted_alpha0.py`
  - log:
    `logs/input_random_fixed_50pct_adapter_quality_assigned_neg025_weighted_alpha0_20260518_061721.log`
  - verified epoch-0 contract: `[00050/00099]` and `[00099/00099]`.

Latest monitor at 2026-05-18 07:13:

- Both clean bs2 quality diagnostics are still running around epoch 13.
- Neither run has reached first validation yet.
- Losses are decreasing and the only recorded non-finite-gradient skips remain
  the two early epoch-0 events noted at relaunch.
- 25876 disk is tight: `/root/autodl-tmp` has about 9.7GB free.

External discussion status:

- `llm-chat` is configured for `gpt-5-pro` through the Responses API with high
  reasoning and 8192 max output tokens, but this model is reserved for a single
  critical arbitration only.
- Gemini CLI was retried twice for a compact current-direction discussion on
  2026-05-18 07:20. Both replies ignored the supplied experimental context and
  produced no usable critique, so they are not counted as completed external
  review.
- GPT-5.5 xhigh read-only advisor critique on 2026-05-18 flagged a concrete
  automation risk: queued runs must not start merely because the quality screen
  disappears. A crash or manual stop has the same observable screen state.
  Accepted fix: both queued screens were restarted behind explicit
  gate-approval sentinel files.

## Current Algorithm Problems

1. Quality-head route is not yet proven safe.
   - The failed detached quality checkpoints stayed near `50` Average-mAP even
     when re-evaluated with `score_alpha=0`.
   - The likely issue is training/checkpoint degradation, not only inference
     score fusion.
   - The clean neutral run is the necessary preservation test.

2. The current head is localization-limited at high tIoU.
   - Historical runs and SimOTA diagnostics show mAP@0.7 remains the weak axis.
   - SimOTA geometry fixes improved low-level collapse but did not approach the
     `63+` dense/Adapter reference.
   - This argues for a conservative ActionFormer-side localization calibration
     before larger assignment rewrites.

3. Input selection still has large oracle headroom.
   - Boundary-aware oracle input variants substantially exceed random-fixed.
   - A teacher-guided pseudo-boundary sampler is the most direct Adapter-side
     attempt to close part of that input-selection gap without using GT at test.

4. Irregular/sparse-head branches remain risky.
   - Prior irregular experiments repeatedly showed contract mismatch between
     sparse axes and dense ActionFormer assumptions.
   - Those lines are useful as diagnosis, but not the next highest-confidence
     performance route for the two occupied servers.

## Queued Next Experiments

These are queued to start only after the current quality diagnostics finish, so
they do not steal GPU or confound the neutral preservation gate.

### 35407 Adapter-Side Route: Pseudo-Boundary Snap Q64

Screen:

- `adapter_pseudo_snap_q64_after_quality`

Launch behavior:

- Waits for `adapter_quality_neutral` to finish.
- Then waits for explicit gate approval file:
  `/root/autodl-tmp/OpenTAD_Back_check/gate_approvals/adapter_pseudo_snap_q64_after_quality.ok`.
- Runs `scripts/run_adapter_pseudo_boundary_snap_pair.sh` with:
  - `START_INDEX=1`
  - `END_INDEX=1`
  - `SKIP_CACHE_BUILD=1`
- Implementation note: the pseudo-boundary launcher itself does not consume a
  `WAIT_SCREENS` variable, so the active queue uses
  `scripts/wait_for_screen_and_gate_then_run.sh`. It first loops on
  `screen -ls` until `adapter_quality_neutral` disappears, then waits for the
  approval sentinel, then runs `CHECK_ONLY=1` and the launcher.

Config:

- `configs/adatad/thumos/input_random_fixed_50pct_adapter_pseudo_boundary_snap_q64.py`

Why this route:

- It directly targets the input-selection gap indicated by boundary oracle
  results.
- It uses existing teacher boundary cache and falls back to random-fixed when
  cache evidence is absent.
- Remote `CHECK_ONLY=1` passed, and required teacher checkpoint/cache files are
  present on 35407.
- Historical pseudo-boundary q32/q64 logs on 35407 used the old
  `batch_size=8` / `24`-iteration contract, so their `~51` final Average-mAP is
  not decisive evidence under the restored bs2 contract.

Success gate:

- First eval should beat the random-fixed Adapter first-eval anchor or show a
  clear recovery trend over random-fixed.
- Final result must exceed the `63.75-63.85` audited baseline band to count as a
  real performance improvement.
- Because the q64 protocol uses pseudo-boundary cache in val/test, any gain must
  be labeled as teacher-assisted input selection unless a companion non-teacher
  evaluation also supports the improvement.

### 25876 ActionFormer-Side Route: Regression-Loss 1.5

Screen:

- `adapter_regloss15_after_quality`

Launch behavior:

- Waits for `adapter_quality_neg025` to finish.
- Then waits for explicit gate approval file:
  `/root/autodl-tmp/OpenTAD_Back_check/gate_approvals/adapter_regloss15_after_quality.ok`.
- Runs `scripts/run_adapter_actionformer_regloss.sh`.

Config:

- `configs/adatad/thumos/input_random_fixed_50pct_adapter_regloss15.py`

Why this route:

- It is a minimal ActionFormer-side localization calibration.
- It does not change the Adapter sampler, backbone, inference, assignment, or
  quality-head path.
- Remote `CHECK_ONLY=1` passed after syncing the config and launcher to 25876.

Success gate:

- Watch high-tIoU metrics, especially mAP@0.7.
- Treat this as promising only if Average-mAP improves without trading away the
  lower tIoU bands.
- Stronger gate from GPT-5.5 critique: continue this scalar route only if
  Average-mAP improves by at least `0.5` and mAP@0.7 improves by at least `1.0`
  without losing mAP@0.3/0.4.

## Prepared Backup Route

### ActionFormer Assignment Backup: SimOTA IoU-Sum MinK

This route is prepared as a backup after `regloss15`, not as an immediate
replacement for the currently queued ActionFormer-side run.

Implementation:

- `opentad/models/losses/assigner/anchor_free_simota_assigner.py`
  now supports:
  - `dynamic_k.mode="iou_sum"`;
  - explicit `min_k`;
  - `dynamic_k.min_candidate_iou=0.05` as a safety gate before dynamic-k
    selection, to avoid forced near-zero-IoU positives under sparse
    random-fixed input;
  - `filter_shortest_gt=False` for overlapping actions;
  - assignment statistics for raw candidate counts, post-IoU-gate candidate
    counts, dynamic-k counts, matched counts, candidate points, confused
    points, and matched points.
- The assigner now rejects unknown nested `dynamic_k` options instead of
  silently ignoring misspelled config keys.
- The safe SimOTA backup config explicitly enables `assignment_debug`, restores
  checkpoint saving, and uses the same first-eval cadence as the active gate:
  `val_start_epoch=40`, `val_eval_interval=2`, `end_epoch=60`.
- Safe launcher:
  `scripts/run_adapter_simota_iou_sum.sh`.
  It has `CHECK_ONLY=1`, `EXPECT_BATCH_SIZE=2`, Adapter/ActionFormer structure
  checks, random-fixed sampler checks, and explicit SimOTA iou-sum config
  assertions for the plain SimOTA backup, including `assignment_debug`,
  `min_candidate_iou=0.05`, checkpoint cadence, and
  train/val/test `batch_size=2`.

Configs:

- Safe backup:
  `configs/adatad/thumos/input_random_fixed_50pct_adapter_simota_mink4_w1.py`
- Manual composite diagnostic only:
  `configs/adatad/thumos/input_random_fixed_50pct_adapter_simota_center25_mink4_w1.py`.
  It inherits the `input_random_fixed_50pct_adapter_fcos_center25.py` visual
  pipeline, so it must not be interpreted as a clean center/range-only ablation
  and is not part of the safe backup launcher.

Why it is backup, not next:

- It changes assignment semantics, positive density, and overlapping-GT
  handling at once, so it is less isolated than `regloss15`.
- It should be used only if the scalar localization calibration is weak or if
  assignment diagnostics point to too few/too narrow positives.
- A short diagnostic run with assignment stats should precede any full
  claim-making run.
- Once AutoDL SSH recovers, `logs/sync_adapter_followup_guards_after_ssh.ps1`
  will sync the plain backup to 25876 and run `CHECK_ONLY=1` only. It does not
  start SimOTA training and does not create approval sentinels.

Verification:

- Local static tests:
  `pytest tests/test_adapter_quality_rescore_contracts.py tests/test_adapter_safety_contracts.py tests/test_adapter_simota_contracts.py -q`
  -> `33 passed, 5 skipped` on Windows. The skipped tests are tensor-level
  SimOTA behavior checks intentionally left for the Linux training environment,
  because local Windows PyTorch DLL loading is unavailable.
- Shell syntax:
  `bash -n scripts/run_adapter_simota_iou_sum.sh scripts/run_adapter_actionformer_regloss.sh scripts/run_adapter_pseudo_boundary_snap_pair.sh scripts/wait_for_screen_and_gate_then_run.sh`
  passed.
- Python syntax:
  `python -m py_compile opentad/models/losses/assigner/anchor_free_simota_assigner.py tests/test_adapter_simota_contracts.py`
  passed.

External review note:

- A read-only xhigh advisor agreed that this SimOTA iou-sum/min-k route is a
  reasonable ActionFormer backup for testing whether high-tIoU weakness comes
  from overly narrow positive assignment, but it should not preempt the current
  clean quality gate or the narrower `regloss15` run.
- A later read-only code review found no critical implementation blocker. The
  accepted fix was to remove the `center25` composite variant from the safe
  backup launcher because it inherits the FCOS-center25 center-crop visual
  pipeline and would confound assignment diagnosis. The review also noted that
  `assign_confuse_point_count` is uninformative when `confuse_weight=1.0`; use
  `candidate_point_count - matched_point_count` for unmatched candidate
  pressure.
- Gemini CLI `gemini-3-pro-preview` route review completed on 2026-05-18 and
  agreed with continuing the current clean bs2 quality gate. It explicitly
  advised not to interrupt or insert SimOTA/q64/regloss before the neutral and
  neg025 quality diagnostics are interpretable. It ranked experiment contract
  fragility and assignment positive density as the most important risks, and
  recommended remote SimOTA diagnostics only after the current gates.
- Gemini CLI `gemini-3-pro-preview` reviewed the SimOTA `min_candidate_iou`
  guard on 2026-05-18 12:44 and approved it for staging, not immediate
  deployment. The review recommended keeping the threshold at `0.05` rather
  than `0.1`, because sparse random-fixed input can depress otherwise useful
  candidate IoUs. Before any full SimOTA training, run remote `CHECK_ONLY=1`
  and then a short 50-100 iteration assignment-debug diagnostic to inspect
  `raw_candidate_counts`, post-gate `candidate_counts`, zero-candidate GTs,
  and early `loss_cls`/`loss_reg` stability.
- A Gemini MCP review attempt on this route failed with
  `unsupported Gemini backend: openai`, so it is not counted as valid review.

## Direction Rules

- If clean neutral quality recovers near `61.95+`, quality-head integration is
  structurally safe; interpret neg025 and optionally run alpha sweeps.
- If clean neutral stays far below `60`, stop quality-head development and
  debug structural training effects before any more quality-loss variants.
- Do not launch positive-only quality targets until background calibration has
  a concrete check.
- Do not restart broad irregular/sparse-head work until the two conservative
  queued routes have produced evidence.
- Do not launch the SimOTA iou-sum/min-k backup until `regloss15` has either
  failed its gate or produced assignment/localization evidence that justifies an
  assignment rewrite.
