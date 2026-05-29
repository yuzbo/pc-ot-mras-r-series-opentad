# BATA BoundarySharp-ST Self-Check 2026-05-29

## Purpose

Implement the Pro-recommended aggressive end-to-end selector route after AB/BH
epoch-19 diagnostics showed weak boundary-channel learning. The route is
`BoundarySharp-ST`: quota selector with straight-through top-k forward
positions for action/boundary chunks, stronger boundary target shaping, and the
same 384-of-768 fixed input budget.

## Changed Files

- `OpenTAD_BATA_Clean/opentad/models/selectors/temporal_density_selector.py`
- `OpenTAD_BATA_Clean/configs/adatad/thumos/e2e_rawdensel_384of768_boundarysharp_st_adapter.py`
- `OpenTAD_BATA_Clean/tests/test_e2e_raw_frame_selector_contracts.py`

Unrelated dirty BATA boundary-acquisition/post-processing files already exist in
the worktree and are not part of this implementation.

## Changed Surface

- Adapter internals: unchanged.
- Detector head logic: unchanged.
- Loss/assignment: selector-side train-only auxiliary target shaping changed.
- Input sampling: changed inside the end-to-end selector by enabling optional
  ST top-k quota positions in the new config only.
- Test-time post-processing: unchanged.

## Method Contract

- The selector still consumes dense decoded 768-frame raw input and outputs
  384 selected frames for the expensive backbone.
- New parameters are backward-compatible defaults:
  `quota_action_position_mode="soft_quantile"`,
  `quota_boundary_position_mode="soft_quantile"`,
  `quota_action_target_weight=1.0`, and
  `quota_boundary_target_weight=1.0`.
- New config sets quota counts to `96/96/192`, summing to `384`.
- ST top-k forward uses hard top-scoring action/boundary positions, but returns
  `hard.detach() - soft.detach() + soft`, so detection and selector losses keep
  a differentiable soft-quantile surrogate gradient path to logits.
- `detach_gt_remap=True` remains inherited from the base config, so GT remap
  construction is not used as a gradient shortcut.
- No GT, teacher, cache, or oracle information is used at test time.

## Tensor and Mask Reasoning

- `st_topk_positions_from_logits` accepts `[B,T]` logits and `[B,T]` dense masks.
- Invalid dense tails are excluded by restricting top-k scores to the valid
  prefix length from `dense_masks`.
- Per-sample valid selected lengths are honored and padded to the quota
  `target_len` with the final valid dense position.
- Action/boundary chunks are concatenated with the context chunk, sorted, and
  passed through the existing minimum-gap separation and padding path.
- The returned position tensor remains shape `[B, quota_len]` for each chunk and
  `[B, 384]` after quota merge.

## Local Verification

Run at `2026-05-29T16:01:22+08:00` on Windows:

```powershell
python -m py_compile opentad\models\selectors\temporal_density_selector.py configs\adatad\thumos\e2e_rawdensel_384of768_boundarysharp_st_adapter.py tests\test_e2e_raw_frame_selector_contracts.py
python -m pytest tests\test_e2e_raw_frame_selector_contracts.py -q -rs
git diff --check -- opentad/models/selectors/temporal_density_selector.py configs/adatad/thumos/e2e_rawdensel_384of768_boundarysharp_st_adapter.py tests/test_e2e_raw_frame_selector_contracts.py
```

Results:

- `py_compile`: PASS.
- `pytest`: `1 passed, 21 skipped` after post-Pro test additions; skips are expected Windows skips for
  Linux/torch/mmengine selector contract tests.
- `git diff --check`: no whitespace errors; only LF/CRLF warnings.

## Gate Status

- Self-check: PASS.
- GPT-5.5 Pro implementation review: WARN, no blocker.
  - Output:
    `logs/oracle_pro_boundarysharp_st_review_20260529.txt`.
  - Stdout/stderr:
    `logs/oracle_pro_boundarysharp_st_review_20260529.stdout.txt` and
    `logs/oracle_pro_boundarysharp_st_review_20260529.err.txt`.
  - Elapsed: about `13m24s`.
  - Accepted findings: add explicit prefix-mask/all-false mask contract,
    non-negative quota target-weight validation, ST invalid-tail and short
    valid-length tests, and make merged-config quota checks depend on
    `selection_mode` rather than filename.
  - Local post-fix verification passed: `py_compile`; Windows pytest
    `1 passed, 20 skipped`; `git diff --check` with only LF/CRLF warnings.
- GPT-5.5 Pro focused fix re-review: WARN, no blocker.
  - Output:
    `logs/oracle_pro_boundarysharp_st_fix_review_20260529.txt`.
  - Accepted final recommendations: include `quota_uniform_target_weight` in
    non-negative validation, add a lightweight negative-weight constructor
    test, fix self-check pytest count, and add a merged-config assertion that
    BoundarySharp-ST keeps `detach_gt_remap=True`.
  - Final local verification after these small fixes passed: `py_compile`;
    Windows pytest `1 passed, 21 skipped`; `git diff --check` with only LF/CRLF
    warnings.
- Gemini CLI review: PASS.
  - stdout:
    `logs/gemini3_pro_preview_boundarysharp_st_review_20260529.txt`.
  - stderr:
    `logs/gemini3_pro_preview_boundarysharp_st_review_20260529.err.txt`.
  - Exit code: `0`.
  - Substantive verdict: no blocking findings, no required fixes; N16R4
    Linux preflight can proceed after DeepSeek review.
- Claude CLI DeepSeek review: PASS.
  - stdout:
    `logs/claude_deepseek_v4_pro_boundarysharp_st_review_20260529.txt`.
  - stderr:
    `logs/claude_deepseek_v4_pro_boundarysharp_st_review_20260529.err.txt`.
  - debug:
    `logs/claude_deepseek_v4_pro_boundarysharp_st_review_20260529.debug.log`.
  - Exit code: `0`.
  - Substantive verdict: no blockers, no required fixes; N16R4 Linux preflight
    and one-GPU BoundarySharp-ST launch can proceed.
- N16R4 Linux preflight: pending after review.
- Deployment: pending after all gates pass.

## Launch Decision

Do not deploy yet. Run the mandatory Pro/Gemini/DeepSeek review gate first. If
the gate passes, sync only the reviewed files and launch one BoundarySharp-ST
Slurm job, preferably after stopping the now-near-parity Route A uniform control
to keep total active project GPU use within the requested 2-3 GPUs.
