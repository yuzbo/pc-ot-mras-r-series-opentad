# BH-SDC Temporal Shape Fix 2026-06-25

Timestamp: 2026-06-25 02:34:37 +08:00

Route label: `DIVERGENT_INNOVATION_BH_SDC_DO_NOT_MERGE_WITH_C3`

## Decision Boundary

This change is scoped to the BH-SDC compact backbone path for formal long training readiness. It does not modify C3, does not merge BH-SDC into C3, and does not relabel BH-SDC as C3-Pro, interval packet, or a natural C3 continuation.

No remote sync, Slurm launch, training, or evaluation was run.

## Root Cause

The formal v4 run failed after Epoch 0 because BH-SDC selected a per-sample sparse count smaller than the fixed temporal shape expected by the VideoMAE Adapter. The compact backbone path sliced the video sample to the real selected count before calling the backbone. For a 384 selected-frame sample with an Adapter initialized for `total_frames=448` and `tubelet_size=2`, Adapter blocks still reshaped tokens as if 224 tubelets existed, causing a temporal reshape failure.

## Implementation

Changed file: `opentad/models/detectors/actionformer.py`

- Infer fixed backbone temporal frames for 5D/6D video inputs from BH-SDC selector/backbone metadata, preferring deploy-visible values such as `max_budget` or `total_frames`, with a fallback that combines Adapter `temporal_size` and patch embedding tubelet size.
- Before calling the fixed-grid backbone, right-pad each per-sample temporal slice to the inferred fixed frame count.
- Build the sample mask at the fixed frame count, with only the first real selected `count` positions set `True`; padded frames are `False`.
- Fail fast if any real selected count exceeds the fixed backbone temporal frame count. The code refuses to truncate selected frames.
- Keep the final packed detector output based only on real counts: features are resized back to `count` when needed, batch output length is `max(real counts)`, and `output_mask` is prefix-true only over real selected positions.

Changed test: `tests/test_bh_sdc_actionformer_integration.py`

- Added regression coverage for `count=384`, `expected=448`: fake backbone receives temporal length 448, mask true count 384 and false padding 64, while final output remains length 384 with 384 true mask positions.
- Added fail-fast coverage for `count > expected`.

## Verification

Required command:

```powershell
python -m py_compile opentad/models/detectors/actionformer.py
```

Result: exit code 0.

Required command:

```powershell
python -m pytest tests/test_bh_sdc_actionformer_integration.py -q -rs
```

Result on default `python`: module skipped because default Python cannot import torch (`torch unavailable`), exit code 1 in this environment.

Supplemental executable torch environment:

```powershell
conda run -n torch_1 python -m pytest tests/test_bh_sdc_actionformer_integration.py -q -rs
```

Result: `6 passed in 9.13s`.

## Protocol Safety

Padded frames are shape-only placeholders for fixed temporal backbone computation. They are not included in the BH-SDC selected budget because the real count is still derived from the incoming sparse prefix mask. They are not marked valid in the sample mask, and the final detector-facing `output_mask` is rebuilt from the original real counts rather than from the padded backbone length.

GT leakage risk: none introduced. The fix uses only selector/backbone shape metadata and the deploy-visible sparse mask/count.

Teacher/raw-prediction-cache leakage risk: none introduced.

Changed surface: detector backbone call logic only. No selector policy, budget controller, token compressor, detector head, loss/assignment, or post-processing logic was changed.

## Remaining Risk

The fix relies on the existing interpolation behavior to map fixed-frame backbone outputs back to the real sparse count when the backbone returns fixed temporal length. This preserves the existing compact path contract but should be watched in the first long-run sanity window for unexpected feature-time distortion.
