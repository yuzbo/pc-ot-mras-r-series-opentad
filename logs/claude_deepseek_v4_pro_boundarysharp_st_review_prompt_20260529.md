Read-only secondary code verification. Do not edit files.

Check whether the BoundarySharp-ST implementation is correct and whether it
matches the original experiment purpose. Focus on protocol leakage, random-fixed
50% contract, tensor/mask semantics, config/launcher consistency, gate safety,
and whether any claimed metric would be attributable to the selector change.
Return PASS/WARN/FAIL with file/line evidence and required fixes.

Changed files:
- OpenTAD_BATA_Clean/opentad/models/selectors/temporal_density_selector.py
- OpenTAD_BATA_Clean/configs/adatad/thumos/e2e_rawdensel_384of768_boundarysharp_st_adapter.py
- OpenTAD_BATA_Clean/tests/test_e2e_raw_frame_selector_contracts.py

Context:
- BoundarySharp-ST adds optional `st_topk` quota position mode for action/boundary chunks: hard top-k forward and soft-quantile surrogate backward via `hard.detach() - soft.detach() + soft`.
- New config keeps dense input window 768 and selected expensive-backbone frames 384, quota `96/96/192`, train-only GT auxiliary target shaping, no test-time GT/teacher/cache.
- GPT-5.5 Pro review returned WARN with no code blocker. Accepted fixes were applied: prefix-mask/all-false-mask checks, non-negative quota target-weight validation including uniform target weight, invalid-tail/short-valid tests, selection_mode-based quota assertion, and detach_gt_remap assertion.
- Focused GPT-5.5 Pro re-review returned WARN with no blocker after final small fixes.
- Gemini CLI `gemini-3-pro-preview` returned PASS, no blocking findings.
- Local verification after final fixes: py_compile PASS; Windows pytest `1 passed, 21 skipped`; git diff --check PASS with LF/CRLF warnings only.

Please answer in Chinese:
- PASS/WARN/FAIL
- Blocking findings
- Non-blocking findings
- Required fixes before deployment
- Whether N16R4 Linux preflight and one-GPU BoundarySharp-ST launch can proceed
