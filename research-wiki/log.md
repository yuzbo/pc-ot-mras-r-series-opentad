# Research Log

## 2026-07-01 07:53:15 +08:00

MDL-Knot route-owned repair for `DIVERGENT_INNOVATION_MDL_KNOT_DO_NOT_MERGE_WITH_C3`.

- Recorded remote formal child `1118197.535` failure: `ValueError: selected_inputs must be shorter than dense_T for sparse selected-only audit`.
- Fixed sampled_raw/full_raw full-observation edge handling so `valid_k == dense_T` no longer crashes DataLoader but is marked `full_observation_no_compression` and cannot count as sampled/raw sparse-compute evidence.
- Added finer `MDL_KNOT_PROFILE` stages for selector objective loop, gap guard, metadata build, and handoff validator.
- Local verification passed: py_compile relevant files; full requested MDL-Knot pytest suite `55 passed, 3 skipped in 96.63s`.
- Gate state remains locked for formal/full training: local `PRECHECK_ONLY_REQUEST_ALLOWED`, shortdiag static allowed with `validated=false`, no mAP/runtime/FLOPs/deploy/paper/sparse-compute claim.

## 2026-07-01 08:06:19 +08:00

MDL-Knot route-owned blocker fix after final review.

- Fixed stale formal/full train unlock state in `input_mdl_knot_dynamic_adapter_irregular_headv3.py`: top-level and acquisition `formal_train_unlocked/full_train_unlocked` are false, route status is precheck-only after sampled_raw edge fix, and no sparse-compute claim is open.
- Hardened `validate_mdl_knot_launch_gate.py` to reject stale user-override/formal/full/queued/unlocked route statuses, config/acquisition train unlocks, and precheck summaries missing `no_claims.sparse_compute=True`.
- Updated tests to assert formal/full train remains locked and to reject stale unlock/status/sparse-claim cases.
- Verification passed: py_compile relevant files, launch gate `PRECHECK_ONLY_REQUEST_ALLOWED`, shortdiag static gate `SHORT_DIAGNOSTIC_CONFIG_STATIC_CHECK_ALLOWED` with `validated=false`, and `tests/test_mdl_knot_tools_and_integration.py` passed `19 passed, 3 skipped in 19.18s`.
- Historical user-override child `1118197.535` remains failed evidence only. Next allowed action is remote `PRECHECK_ONLY`, then one-epoch shortdiag after precheck; formal/full long training remains locked.
