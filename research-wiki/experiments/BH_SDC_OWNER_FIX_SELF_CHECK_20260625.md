# BH-SDC Owner Fix Self-Check 2026-06-25

Route label: `DIVERGENT_INNOVATION_BH_SDC_DO_NOT_MERGE_WITH_C3`

Status: `FIXED_FOR_LOCAL_SMOKE_PENDING_FOLLOWUP_PRO`

This route remains independent from C3/C3-Pro/interval/global-rank/ST work. It
must not be described as a C3 continuation, a C3-Pro combination, or an
interval/global-rank/ST extension.

## Pro Findings Addressed

Source review:
`E:/DeskTop/TAD/temrefuse-tad/logs/divergent_routes_four_impl_pro_review_result_20260624.md`

Accepted BH-SDC blockers:

1. `PCOTMRASBoundaryHazardSparseToDenseBridge` build path needed real
   implementation/registration proof.
2. Full-train config provenance was invalid because `reviewed_impl_commit` did
   not match the reviewed commit and training gates were open before re-review.
3. Tests needed route isolation, full selector/backbone/bridge/projection/head
   build-forward smoke, mask/selected-index/physical-time contracts, and
   no-GT/no-teacher/no-cache/no-raw-prediction checks.

## Changed Files

- `configs/adatad/thumos/bh_sdc_boundary_hazard_sparse_dense_full_train_candidate_n16r4.py`
- `tools/bata/validate_bh_sdc_full_train_gate.py`
- `tests/test_bh_sdc_config_gate.py`
- `tests/test_bh_sdc_actionformer_integration.py`
- `tests/test_bh_sdc_core.py`
- `docs/en/bh_sdc_route_original_purpose_and_drift_check_20260624.md`
- `docs/en/bh_sdc_n16r4_launch_gate_review_context_20260624.md`
- `research-wiki/experiments/BH_SDC_OWNER_FIX_SELF_CHECK_20260625.md`

## Self-Check

Implementation purpose:
fix BH-SDC to the local/static/smoke level after Pro review, without opening
remote sync, Slurm, long training, `tools/train.py`, `tools/test.py`, checkpoint
writes, or metric claims.

Strict random-fixed 50% contract:
not applicable as a fixed 50% claim. BH-SDC is a dynamic min/target/max budget
route. Current verification is local synthetic smoke only.

GT/teacher/cache/raw-prediction risk:
test/validation selector and bridge paths reject forbidden metadata containing
GT, teacher, oracle, cache, raw prediction, target, checkpoint, or result
payloads. The config keeps raw-prediction loading/saving disabled.

Changed surface:
input sampling, dynamic budget policy, compact backbone input length,
sparse-to-dense token completion, and temporal metadata. This fix does not
change detector head logic, loss/assignment, or test-time post-processing.

Build path:
`PCOTMRASBoundaryHazardSparseToDenseBridge` is implemented and registered in
`opentad/models/selectors/bh_sdc_frame_selector.py`, imported by
`opentad/models/selectors/__init__.py`, and built by ActionFormer through
`build_token_compressor`.

Full-train gate:
fail-closed. The full-train candidate uses
`FIXED_FOR_LOCAL_SMOKE_PENDING_FOLLOWUP_PRO`,
`reviewed_impl_commit=FOLLOWUP_PRO_REQUIRED_AFTER_BH_SDC_PRO_FIX`,
`allowed_entrypoints=()`, `command_whitelist=()`, and all train/sync/Slurm/GPU
flags false. `validate_launch_gate_payload` rejects every payload before schema
validation until a follow-up Pro review changes the state.

## Verification Evidence

```powershell
conda run -n torch_1 python -m py_compile configs\adatad\thumos\bh_sdc_boundary_hazard_sparse_dense_full_train_candidate_n16r4.py tools\bata\validate_bh_sdc_full_train_gate.py tests\test_bh_sdc_config_gate.py tests\test_bh_sdc_actionformer_integration.py tests\test_bh_sdc_core.py
```

Result: passed.

```powershell
conda run -n torch_1 python -m pytest tests\test_bh_sdc_config_gate.py tests\test_bh_sdc_actionformer_integration.py tests\test_bh_sdc_core.py tests\test_bh_sdc_metadata_contract.py -q
```

Result: `26 passed`.

```powershell
$files = Get-ChildItem -LiteralPath tests -Filter 'test_bh_sdc*.py' | ForEach-Object { $_.FullName }; conda run -n torch_1 python -m pytest @files -q
```

Result: `29 passed`.

```powershell
conda run -n torch_1 python -m pytest tests\test_bh_sdc_actionformer_integration.py::test_bh_sdc_build_detector_constructs_selector_bridge_projection_and_head_smoke -q
```

Result: `1 passed`.

```powershell
conda run -n torch_1 python tools\bata\validate_bh_sdc_full_train_gate.py configs\adatad\thumos\bh_sdc_boundary_hazard_sparse_dense_full_train_candidate_n16r4.py --json
```

Result: fail-closed JSON with `authorized=false`, `allowed_entrypoints=[]`,
`command_whitelist=[]`, and
`launch_decision=FIXED_FOR_LOCAL_SMOKE_PENDING_FOLLOWUP_PRO`.

## Still Locked

- remote sync
- Slurm
- GPU use
- `tools/train.py`
- `tools/test.py`
- full train
- train-validation mAP and detector mAP claims
- checkpoint writes
- runtime/FLOPs/deploy/paper claims

Next allowed action:
local py_compile, focused tests, no-data/no-train build-forward smoke, then a
follow-up GPT-5.5 Pro review if the user asks to move beyond local smoke.
