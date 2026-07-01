# Sparse TAD Task Flow Tracker

Route-owned copy for `DIVERGENT_INNOVATION_MDL_KNOT_DO_NOT_MERGE_WITH_C3`.

## All Required Experiments and Current Status

| Experiment / config | Changed surface | Current status | Review / gate state | Deployment / result state | Next action |
| --- | --- | --- | --- | --- | --- |
| MDL-Knot sampled_raw full-observation edge repair, `input_mdl_knot_dynamic_adapter_irregular_headv3*.py` | Input sampling handoff validator, sampled_raw audit metadata, profile instrumentation, diagnostics counters | Local fix complete in owned worktree | Local py_compile passed; requested pytest passed; static launch gate says `PRECHECK_ONLY_REQUEST_ALLOWED`; shortdiag validator says static allowed but `validated=false` | No remote action by this owner; old formal child `1118197.535` failed with `selected_inputs must be shorter than dense_T for sparse selected-only audit`; no mAP/runtime/FLOPs/deploy/paper/sparse-compute claim | Coordinator may allow remote `PRECHECK_ONLY`; one-epoch `SHORT_DIAGNOSTIC_ONLY` only after remote precheck passes; formal/full long training remains locked |
| MDL-Knot formal/full train lock consistency fix, `input_mdl_knot_dynamic_adapter_irregular_headv3.py` | Config lock state, launch gate claim locks, tests | Local blocker fix complete in owned worktree | py_compile passed; `tests/test_mdl_knot_tools_and_integration.py` passed; launch gate rejects stale unlock/status/sparse-claim states | No remote action; `.535` remains failed historical user-override formal evidence only | Remote `PRECHECK_ONLY` only; then one-epoch shortdiag after remote precheck; formal/full long training remains locked |

## Timeline

### 2026-07-01 07:53:15 +08:00

- Route label: `DIVERGENT_INNOVATION_MDL_KNOT_DO_NOT_MERGE_WITH_C3`.
- Owned worktree: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_MDLKnot_RealDiag_Worktree_20260630`.
- Owned branch: `codex/divergent-mdl-knot-realdiag-20260630`.
- Remote evidence recorded: N16R4 formal child `1118197.535` failed with `ValueError: selected_inputs must be shorter than dense_T for sparse selected-only audit`.
- Changed files: `opentad/acquisition/mdl_knot/validators.py`, `opentad/acquisition/mdl_knot/selector.py`, `opentad/acquisition/mdl_knot/handoff.py`, `opentad/acquisition/mdl_knot/diagnostics.py`, `opentad/datasets/transforms/end_to_end.py`, `tests/test_mdl_knot_core.py`, `tests/test_mdl_knot_tools_and_integration.py`, this tracker, route report, and `research-wiki/log.md`.
- Strict random-fixed 50% contract: not applicable to this MDL-Knot dynamic route; fixed-pad bridge still makes no sparse-compute claim.
- GT/teacher leakage risk: no new GT, teacher, prediction cache, evaluator, or post-processing access added.
- Current mAP evidence: none.
- Verification: py_compile relevant files passed; `python -m pytest tests/test_mdl_knot_core.py tests/test_mdl_knot_shortdiag.py tests/test_mdl_knot_tools_and_integration.py tests/test_mdl_knot_realdiag.py -q` passed as `55 passed, 3 skipped in 96.63s`.
- Gate state: `validate_mdl_knot_launch_gate.py` returned `PRECHECK_ONLY_REQUEST_ALLOWED`; `validate_mdl_knot_shortdiag.py` returned `SHORT_DIAGNOSTIC_CONFIG_STATIC_CHECK_ALLOWED` with `validated=false`.
- Decision: remote `PRECHECK_ONLY` is allowed by local gates; one-epoch shortdiag may follow only after remote precheck passes; formal/full long training remains locked.

### 2026-07-01 08:06:19 +08:00

- Route label: `DIVERGENT_INNOVATION_MDL_KNOT_DO_NOT_MERGE_WITH_C3`.
- Owned worktree: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_MDLKnot_RealDiag_Worktree_20260630`.
- Owned branch: `codex/divergent-mdl-knot-realdiag-20260630`.
- Final review blocker: main formal config still carried stale `formal_train_unlocked=True` / user-override formal queued semantics after failed child `1118197.535`.
- Changed files: `configs/adatad/thumos/input_mdl_knot_dynamic_adapter_irregular_headv3.py`, `tools/mdl_knot/validate_mdl_knot_launch_gate.py`, `tests/test_mdl_knot_tools_and_integration.py`, route report, tracker, and `research-wiki/log.md`.
- Strict random-fixed 50% contract: not applicable to this MDL-Knot dynamic route; fixed-pad bridge still makes no sparse-compute claim.
- GT/teacher leakage risk: no new GT, teacher, prediction cache, evaluator, or post-processing access added.
- Current mAP evidence: none.
- Verification: py_compile relevant config/gate/test files passed; `python -m pytest tests/test_mdl_knot_tools_and_integration.py -q` passed as `19 passed, 3 skipped in 19.18s`.
- Gate state: `validate_mdl_knot_launch_gate.py` returned `PRECHECK_ONLY_REQUEST_ALLOWED` with `formal_train_unlocked=false`; `validate_mdl_knot_shortdiag.py` returned `SHORT_DIAGNOSTIC_CONFIG_STATIC_CHECK_ALLOWED` with `validated=false`.
- Decision: historical `.535` formal user-override run is failed evidence only; no full-train permission exists. Next action remains remote `PRECHECK_ONLY` then one-epoch shortdiag only after precheck passes; formal/full long training remains locked.
