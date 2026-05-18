# Gemini CLI SimOTA Min-IoU Review, 2026-05-18 12:44

Model: `gemini-3-pro-preview`

Scope:

- Review local SimOTA backup change only.
- Active clean bs2 quality diagnostics must not be interrupted.
- SimOTA remains a post-regloss backup route, not an immediate run.

Change reviewed:

- Added `dynamic_k.min_candidate_iou=0.05` to the safe SimOTA backup config.
- `AnchorFreeSimOTAAssigner` now filters candidates below `min_candidate_iou`
  before dynamic-k selection.
- Stats now include both `raw_candidate_counts` and post-gate
  `candidate_counts`.
- `scripts/run_adapter_simota_iou_sum.sh` asserts and prints the threshold in
  `CHECK_ONLY=1`.

Reviewer verdict:

- Conceptually sound as a backup safety guard.
- Preserves interpretability because assignment stats expose how often the
  gate changes candidate counts.
- Keep `min_candidate_iou=0.05`; do not raise to `0.1` initially because sparse
  random-fixed inputs can already lower candidate IoUs.
- Do not deploy or launch this backup before the active AutoDL quality gates
  conclude or fail.

Required remote diagnostics before full SimOTA training:

- First run remote `CHECK_ONLY=1`.
- Then run a short assignment-debug diagnostic, about 50-100 iterations or one
  epoch, before any full training claim.
- Inspect early assignment stats:
  - `raw_candidate_counts - candidate_counts` should be visible so the gate is
    not silently inert.
  - GTs with zero post-gate candidates should remain low; if many GTs are
    zeroed, the route is too sparse or the center/range prior is wrong.
  - Watch `loss_cls` and `loss_reg` for NaN, spikes, or non-decreasing
    regression loss.

Agent decision:

- Accept the review.
- Keep `0.05` as a safety threshold in the safe backup route.
- Do not create any approval sentinel or launch SimOTA from this change.
- Sync only after AutoDL SSH recovers through
  `logs/sync_adapter_followup_guards_after_ssh.ps1`, then run remote
  `CHECK_ONLY=1`.
