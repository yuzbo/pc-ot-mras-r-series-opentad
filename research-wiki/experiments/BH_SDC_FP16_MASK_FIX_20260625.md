# BH-SDC FP16 Mask Overflow Fix - 2026-06-25

Timestamp: 2026-06-25 02:13:45 +08:00
Route label: `DIVERGENT_INNOVATION_BH_SDC_DO_NOT_MERGE_WITH_C3`
Owned branch: `codex/bh-sdc-fp16-mask-fix-20260625`
Base commit: `382fedc08db8d88497c73777929dbc441d92972d`

## Scope

This is a BH-SDC selector dtype bug fix only. It is not a C3 fix or route merge.

Changed files:
- `opentad/models/selectors/bh_sdc_frame_selector.py`
- `tests/test_bh_sdc_core.py`
- `research-wiki/experiments/BH_SDC_FP16_MASK_FIX_20260625.md`

No remote sync, Slurm, training, launcher change, or C3 config/route edit was performed.

## Root Cause

N16R4 job `1117268` on `g0053` passed launch and optimizer setup, then failed on the first forward batch at:

`BoundaryHazardAcquisitionPolicy.forward`: `combined = combined.masked_fill(~valid, -1.0e6)`

Under AMP/fp16, `combined` can be `torch.float16`; `-1.0e6` is not representable in Half and raises:

`RuntimeError: value cannot be converted to type at::Half without overflow`

## Fix

Added `_dtype_safe_logit_fill_value()` and `_invalid_low_logit_like()` in the BH-SDC selector. Mask fill values used on logit tensors are now clamped to a dtype-representable finite sentinel. For fp16 invalid low logits this resolves to `-1.0e4`, preserving a strong invalid score without creating Inf or suppressing real NaN/Inf checks.

Updated:
- probe scout densification `torch.full_like` and invalid-tail `masked_fill`
- densified `frame_selection_logits` invalid-tail mask
- temporal scout per-head invalid-tail mask
- policy `combined` invalid-tail mask, replacing the unsafe `-1.0e6`

Finite checks remain intact. No GT, teacher, oracle, raw-prediction cache, post-processing, loss, assignment, or detector-head behavior was changed.

## Regression Evidence

RED before fix:
- `C:\Users\skywalker\.conda\envs\torch_1\python.exe -m pytest tests/test_bh_sdc_core.py::test_acquisition_policy_accepts_fp16_scout_logits_with_invalid_tail -q -rs`
- Result: failed with the expected Half overflow at `combined.masked_fill(~valid, -1.0e6)`.

GREEN after fix:
- `C:\Users\skywalker\.conda\envs\torch_1\python.exe -m py_compile opentad/models/selectors/bh_sdc_frame_selector.py`
- Result: pass.
- `C:\Users\skywalker\.conda\envs\torch_1\python.exe -m pytest tests/test_bh_sdc_core.py::test_acquisition_policy_accepts_fp16_scout_logits_with_invalid_tail tests/test_bh_sdc_core.py::test_probe_scout_densification_accepts_fp16_logits_with_invalid_tail -q -rs`
- Result: `2 passed in 8.79s`.
- `C:\Users\skywalker\.conda\envs\torch_1\python.exe -m pytest tests/test_bh_sdc_core.py tests/test_bh_sdc_actionformer_integration.py -q -rs`
- Result: `14 passed in 7.77s`.

## Residual Risk

The fix is limited to BH-SDC selector logit masking. It does not validate the full N16R4 training graph, CUDA AMP runtime, or later detector/backbone stages because no remote sync, Slurm, or training was allowed in this phase.

The repository tracker and `research-wiki/log.md` were not updated because this task explicitly limited writable files to the owned worktree files listed by the user.
