# Research Log

## 2026-07-01 07:53:15 +08:00

MDL-Knot route-owned repair for `DIVERGENT_INNOVATION_MDL_KNOT_DO_NOT_MERGE_WITH_C3`.

- Recorded remote formal child `1118197.535` failure: `ValueError: selected_inputs must be shorter than dense_T for sparse selected-only audit`.
- Fixed sampled_raw/full_raw full-observation edge handling so `valid_k == dense_T` no longer crashes DataLoader but is marked `full_observation_no_compression` and cannot count as sampled/raw sparse-compute evidence.
- Added finer `MDL_KNOT_PROFILE` stages for selector objective loop, gap guard, metadata build, and handoff validator.
- Local verification passed: py_compile relevant files; full requested MDL-Knot pytest suite `55 passed, 3 skipped in 96.63s`.
- Gate state remains locked for formal/full training: local `PRECHECK_ONLY_REQUEST_ALLOWED`, shortdiag static allowed with `validated=false`, no mAP/runtime/FLOPs/deploy/paper/sparse-compute claim.
