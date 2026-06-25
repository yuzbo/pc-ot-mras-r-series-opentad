# BH-SDC Backbone Mask Fix 2026-06-25

Timestamp: 2026-06-25 08:38:19 +08:00

Route label: `DIVERGENT_INNOVATION_BH_SDC_DO_NOT_MERGE_WITH_C3`

## Blocker

Remote v5b formal training reached Epoch 0 and failed inside `BackboneWrapper.forward`:

`RuntimeError: The size of tensor a (224) must match the size of tensor b (448) at non-singleton dimension 2`

The compact BH-SDC path padded 384 selected video frames to the fixed 448-frame VideoMAE input, then passed a 448-length frame-level mask to the wrapper. The wrapper applies `masks` after the video backbone, where VideoMAE outputs 224 tubelet features, so the wrapper mask must be feature-level.

## Changed Surface

- `opentad/models/detectors/actionformer.py`
  - Added `frame_mask` to `backbone output mask` conversion for fixed-temporal 5D/6D BH-SDC compact video inputs.
  - The conversion infers `tubelet_size` from the selector or backbone and groups contiguous frames with `any` validity per tubelet.
  - If a fixed video input needs conversion but tubelet/downsample ratio is unavailable, the path fails fast instead of passing a mismatched frame-level mask.
  - 3D feature inputs and non-fixed-temporal paths keep the existing mask behavior.
- `tests/test_bh_sdc_actionformer_integration.py`
  - Updated the regression test with a fake wrapper-like backbone that receives 448 input frames, returns 224 feature steps, and rejects a 448-length mask.
  - The test asserts the backbone receives a 224-length mask with 192 valid tubelets and 32 padded tubelets, while detector-facing output remains length 384 with 384 valid positions.

## Budget And Mask Semantics

`sample_mask` remains the selected-frame budget mask: for count 384 and fixed frames 448, it has 384 true positions followed by 64 padding false positions.

Only the mask passed into the backbone wrapper is converted to feature-level. With `tubelet_size=2`, the 448 frame positions become 224 tubelet positions. The first 192 tubelets are valid and the final 32 padded tubelets are false.

Detector-facing output is still packed by the real selected count. The existing interpolation maps `[B,C,224]` backbone output to `count=384`, and `output_mask` is built from `counts`, so it remains `[B,384]` with 384 true positions. This preserves the selected frame budget and mask statistics.

## Verification

No remote sync, Slurm launch, training, or evaluation was started.

Required local checks:

- `python -m py_compile opentad/models/detectors/actionformer.py`: PASS.
- `conda run -n torch_1 python -m pytest tests/test_bh_sdc_actionformer_integration.py -q -rs`: PASS, `6 passed in 6.25s`.
- `git diff --check`: PASS. Git reported Windows LF-to-CRLF working-copy warnings for edited Python files, but no whitespace errors.

## Remaining Risk

The fix assumes the wrapper temporal downsample ratio is represented by `tubelet_size`. If a future backbone has additional temporal pooling after tubelet embedding, it should expose an explicit output-mask ratio or fixed output length; otherwise this path will fail fast rather than silently mis-mask.
