# BH-SDC FP16 Completion Fix - 2026-06-25

Timestamp: 2026-06-25 08:54:18 +08:00
Route label: `DIVERGENT_INNOVATION_BH_SDC_DO_NOT_MERGE_WITH_C3`
Owned branch: `codex/bh-sdc-fp16-completion-fix-20260625`
Base commit: `98843265b8daa86daef939ab589c989fe4acd7a2`

## Scope

This is a BH-SDC sparse-to-dense completion dtype blocker fix only. It is not a C3 fix, interval-packet change, dynamic-budget change, launcher change, or route merge.

Changed files:
- `opentad/models/selectors/bh_sdc_frame_selector.py`
- `tests/test_bh_sdc_core.py`
- `research-wiki/experiments/BH_SDC_FP16_COMPLETION_FIX_20260625.md`

No remote sync, Slurm, training, evaluation, shared-repo write, or C3 route edit was performed.

## Root Cause

N16R4 BH-SDC v6 formal training job `1118197` reached Epoch 0 after the VideoMAE reshape and wrapper mask fixes, then failed in:

`PCOTMRASBoundaryHazardSparseToDenseBridge._complete`: `completed[:, selected] = sparse`

Under AMP/fp16 compression, sparse feature tensors may be `torch.float16`, while interpolation distances, softmax weights, or the matmul result may be promoted to `torch.float32`. The old implementation mixed the original half `sparse` tensor with a promoted completion tensor during observed-column overwrite, creating a Half/Float dtype mismatch.

## Fix

`_complete()` now uses an explicit completion work dtype for interpolation:

- `completion_dtype` is `torch.float32` for fp16/bf16/fp32 feature paths and `torch.float64` only when the input features are already float64.
- `dense_axis`, selected-index distances, interpolation weights, `gap_distance`, and `completion_confidence` are computed in the completion work dtype.
- Sparse observed features are converted to the same work dtype before interpolation and before `completed[:, selected]` overwrite.
- The completed dense feature row is cast back to the original `features.dtype` only when written into `dense`.

This keeps the numerically sensitive interpolation and metadata confidence/gap calculations in float32 while preserving the detector-facing feature dtype under AMP/fp16.

## Budget, Mask, And Leakage Semantics

The fix does not change selected indices, selected count, dense valid length, sparse masks, dense masks, observed masks, or metadata keys.

The selected-frame budget remains exactly the acquisition plan's `selected_count` and `selected_dense_indices`. Observed columns are still force-restored from the sparse input features, now through a dtype-compatible work tensor and final cast back to the feature dtype.

Metadata remains deploy-safe:
- `uses_gt`: `False`
- `uses_teacher`: `False`
- `uses_raw_prediction_cache`: `False`

No GT, teacher, oracle, raw-prediction cache, detector-head logic, loss/assignment logic, post-processing, remote launcher, or C3 mainline behavior was changed.

## Regression Evidence

RED before fix:
- `conda run -n torch_1 python -m pytest tests/test_bh_sdc_core.py::test_sparse_dense_bridge_accepts_fp16_features_with_float32_completion_weights -q -rs`
- Result: failed with the expected dtype incompatibility in `_complete()` when fp16 sparse features met float32 completion weights: `RuntimeError: expected m1 and m2 to have the same dtype, but got: struct c10::Half != float`.

GREEN after fix:
- `conda run -n torch_1 python -m pytest tests/test_bh_sdc_core.py::test_sparse_dense_bridge_accepts_fp16_features_with_float32_completion_weights -q -rs`
- Result: `1 passed in 6.19s`.

Full required verification is recorded in the final task report after running the required command set.

## Residual Risk

This local fix validates the sparse-to-dense completion dtype contract and regression behavior in the owned worktree. It does not validate the full N16R4 CUDA AMP training graph or later long-run dynamics because the task explicitly forbids remote sync, Slurm, training, and evaluation.

The repository tracker and `research-wiki/log.md` were not updated because this task explicitly limited writable files to the owned worktree files listed by the user, and `research-wiki/log.md` was not in the allowed write list.
