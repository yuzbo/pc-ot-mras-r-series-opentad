Read-only code review. Do not edit files.

请用中文审查 BoundarySharp-ST 端到端 selector 实现。重点是实现正确性、协议一致性、GT/teacher leakage、tensor/mask 语义、配置继承、部署安全，以及未来指标归因。

Changed files:
- OpenTAD_BATA_Clean/opentad/models/selectors/temporal_density_selector.py
- OpenTAD_BATA_Clean/configs/adatad/thumos/e2e_rawdensel_384of768_boundarysharp_st_adapter.py
- OpenTAD_BATA_Clean/tests/test_e2e_raw_frame_selector_contracts.py

Relevant unchanged context:
- OpenTAD_BATA_Clean/opentad/models/detectors/actionformer.py
- OpenTAD_BATA_Clean/configs/adatad/thumos/e2e_rawdensel_384of768_detachgt_cont_adapter.py
- OpenTAD_BATA_Clean/scripts/run_e2e_rawdensel_n16r4.sbatch
- research-wiki/experiments/BATA_BOUNDARYSHARP_ST_SELF_CHECK_20260529.md

Experiment purpose:
- Previous AB/BH quota selector epoch-19 diagnostics showed only mild non-uniform coverage and weak/flat boundary channel.
- BoundarySharp-ST adds optional `st_topk` quota position mode for action/boundary chunks: hard top-k forward, soft-quantile surrogate backward via `hard.detach() - soft.detach() + soft`.
- New config keeps dense input window 768 and selected backbone frames 384, quota `96/96/192`, train-only GT auxiliary target shaping, no test-time GT/teacher/cache.

Review history:
- GPT-5.5 Pro full implementation review returned WARN with no blocker.
- Accepted fixes applied: prefix-mask/all-false-mask checks, non-negative quota target-weight validation including uniform target weight, ST invalid-tail/short-valid tests, merged-config quota check based on `selection_mode`, and `detach_gt_remap=True` assertion.
- Focused GPT-5.5 Pro re-review returned WARN with no code blocker after these fixes.

Local verification:
- `python -m py_compile opentad\models\selectors\temporal_density_selector.py configs\adatad\thumos\e2e_rawdensel_384of768_boundarysharp_st_adapter.py tests\test_e2e_raw_frame_selector_contracts.py`: PASS.
- `python -m pytest tests\test_e2e_raw_frame_selector_contracts.py -q -rs` on Windows: `1 passed, 21 skipped`; skips are expected because Linux/torch/mmengine tests are skipped on Windows.
- `git diff --check -- <changed files>`: PASS with LF/CRLF warnings only.

Please answer in Chinese with:
- Verdict: PASS / WARN / FAIL
- Blocking findings
- Non-blocking findings
- Required fixes before deployment
- Whether N16R4 Linux preflight and one-GPU BoundarySharp-ST launch can proceed after DeepSeek review
