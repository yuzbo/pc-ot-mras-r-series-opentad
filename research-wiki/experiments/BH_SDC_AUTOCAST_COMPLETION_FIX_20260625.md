# BH-SDC Autocast Completion Fix 2026-06-25

Timestamp: 2026-06-25 09:17:02 +08:00

Route: DIVERGENT_INNOVATION_BH_SDC_DO_NOT_MERGE_WITH_C3

Scope: owned worktree only, `OpenTAD_BHSDC_AutocastCompletionFix_Worktree_20260625`.

Issue: BH-SDC sparse-to-dense completion could fail under autocast when the interpolation matmul returned an autocast dtype while observed-column assignment still used the float32 completion source.

Fix:
- Locally disable autocast around completion distance, softmax, matmul, and observed-column assignment.
- Explicitly cast the observed assignment source to `completed.dtype`.
- Preserve final dense feature output cast back to the original `features.dtype`.

Protocol notes:
- Input sampling and BH-SDC selection semantics unchanged.
- No GT, teacher, raw-prediction cache, post-processing, C3, dynamic-budget, interval, or physical-grid ActionFormer files changed.
- No remote sync, Slurm, or training launched.

Verification:
- Focused RED test before implementation failed at observed-column assignment with `Index put requires the source and destination dtypes match`.
- Focused GREEN test after implementation passed.
