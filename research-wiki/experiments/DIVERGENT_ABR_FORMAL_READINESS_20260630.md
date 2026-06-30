# DIVERGENT ABR Formal Readiness Hardening 2026-06-30

Route label: `DIVERGENT_INNOVATION_ABR_DO_NOT_MERGE_WITH_C3`

Owned worktree: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_ABR_FormalGate_Worktree_20260630`

Owned branch: `codex/divergent-abr-formal-gate-20260630`

Timestamp: `2026-06-30T11:16:34.9663566+08:00`

## Decision

This stage hardens ABR formal-readiness gates without unlocking formal full training. The change records first-round bracket recall and transition coverage from deploy-visible scout evidence, rejects low first-round coverage before formal use, and adds a formal config/gate mode that disables diagnostic fallback.

No training, `tools/test.py`, evaluation, remote sync, Slurm, GPU action, Pro/Oracle/Rosetta action, C3 route, combo route, mAP claim, runtime claim, deploy claim, sparse-compute claim, or paper claim was performed.

## Changed Surface

- Input sampling / acquisition policy evidence only:
  - `opentad/acquisition/abr/types.py`
  - `opentad/acquisition/abr/selector.py`
  - `opentad/acquisition/abr/validators.py`
- ABR validators and local gate tools:
  - `tools/abr/audit_abr_pipeline_precheck.py`
  - `tools/abr/validate_abr_formal_gate.py`
- Config gate:
  - `configs/adatad/thumos/input_abr_active_bracket_refinement_adapter_irregular_headv3_formal.py`
- Tests:
  - `tests/test_abr_formal_gate.py`

No Adapter, backbone, neck, detector head, loss/assignment, evaluator, post-processing, C3, combo, or remote launcher logic was changed.

## Formal-Readiness State

The round-0 ABR ledger now records:

- `transition_count`
- `bracket_count`
- `bracketed_transition_count`
- `missed_transition_count`
- `first_round_bracket_recall`
- `first_round_transition_coverage`
- `first_round_temporal_coverage_fraction`
- `missed_transitions`
- scout source and diagnostic-fallback flag

Blocker fix timestamp: `2026-06-30T11:34:42.1723161+08:00`

The dense transition scanner now ignores ambiguous samples while preserving the
nearest prior non-ambiguous state and position. When the next non-ambiguous
state flips, it records the transition across the ambiguous span. This closes
the formal-readiness blocker where `background -> ambiguous -> action` and
`action -> ambiguous -> background` could previously produce
`transition_count = 0` and falsely report `first_round_bracket_recall = 1.0`.

Focused tests now cover both ambiguous-mediated start and end transitions. In
each case a deliberately insufficient first-round scaffold records one missed
transition, `first_round_bracket_recall = 0.0`, and
`first_round_transition_coverage = 0.0`; the formal-readiness payload rejects
that evidence with `LOCKED`.

Formal gate blocker fix timestamp: `2026-06-30T12:00:03.0293670+08:00`

The formal-readiness validator now fail-closes zero-transition pseudo-evidence:
a payload with `transition_count = 0`, `bracketed_transition_count = 0`,
`missed_transition_count = 0`, `first_round_bracket_recall = 1.0`, and
`first_round_transition_coverage = 1.0` is rejected as non-evidence, not
accepted as perfect first-round coverage. Formal evidence now explicitly
requires:

- `transition_count > 0`
- `bracketed_transition_count + missed_transition_count == transition_count`
- `bracketed_transition_count == transition_count`
- `missed_transition_count == 0`
- `first_round_bracket_recall` consistent with
  `bracketed_transition_count / transition_count`
- `first_round_transition_coverage == 1.0` for fully bracketed first-round
  transitions

Focused tests also cover the corrected ambiguous-mediated transition path:
`background -> ambiguous -> action` and `action -> ambiguous -> background`
still produce nonzero transition counts, and a bracketed nonzero transition can
pass the formal evidence validator while keeping `full_train_unlocked=False`.

`tools/abr/audit_abr_pipeline_precheck.py --mock-only` now writes the same first-round fields into the precheck summary. The current synthetic precheck rich case intentionally remains insufficient for formal readiness:

- `first_round_bracket_recall = 0.3333333333333333`
- `first_round_transition_coverage = 0.3333333333333333`
- `first_round_missed_transition_count = 4`

This is not a failure of the local hardening. It is the evidence that formal full training must remain locked until real deploy-visible scout evidence shows acceptable first-round bracket recall/coverage.

## Gates

Precheck behavior is preserved:

- Base config validator still returns `LOCAL_PRECHECK_ONLY_VALIDATION`.
- Short diagnostic validator still returns `ONE_EPOCH_TRAIN_LOSS_DIAGNOSTIC_ONLY`.
- Diagnostic fallback remains allowed only for explicit PRECHECK/diagnostic use.

Formal behavior is fail-closed:

- Formal config disables `allow_diagnostic_fallback_scout`.
- Formal config uses `fallback_stage="FORMAL_READINESS_LOCKED"`, not `PRECHECK_ONLY`.
- Formal gate requires deploy-visible scout evidence.
- Formal gate rejects diagnostic fallback and PRECHECK_ONLY payloads.
- Formal gate requires `first_round_bracket_recall >= 0.95`.
- Formal gate requires `first_round_transition_coverage >= 0.95`.
- Formal gate rejects any missed first-round transition.
- A passing formal config still returns only `FORMAL_REVIEW_PACKET_ONLY` and `full_train_unlocked=False`.

## Local Verification

Additional blocker-fix verification at `2026-06-30T12:00:03.0293670+08:00`:

Passed:

```powershell
python -m py_compile opentad\acquisition\abr\validators.py tests\test_abr_formal_gate.py tests\test_abr_pipeline_and_gate.py tests\test_abr_shortdiag.py
```

Passed:

```powershell
python -m pytest tests\test_abr_formal_gate.py -q
```

Result: `9 passed`.

Passed:

```powershell
python -m pytest tests\test_abr_pipeline_and_gate.py tests\test_abr_shortdiag.py -q
```

Result: `28 passed, 1 skipped`.

Passed:

```powershell
python -m pytest tests\test_abr_core.py tests\test_abr_pipeline_and_gate.py tests\test_abr_formal_gate.py tests\test_abr_shortdiag.py -q
```

Result: `44 passed, 1 skipped`.

Passed:

```powershell
python tools\abr\validate_abr_formal_gate.py --config configs\adatad\thumos\input_abr_active_bracket_refinement_adapter_irregular_headv3_formal.py
```

Decision: `FORMAL_REVIEW_PACKET_ONLY`; `full_train_unlocked=False`; still locked: `FORMAL_FULL_TRAIN_PENDING_REAL_SCOUT_RECALL_EVIDENCE`, `TOOLS_TEST_PY`, `EVALUATION`, `MAPPAPER_CLAIM`, `RUNTIME_OR_SPARSE_COMPUTE_CLAIM`, `DEPLOY_CLAIM`.

Passed:

```powershell
python tools\abr\validate_abr_launch_gate.py --config configs\adatad\thumos\input_abr_active_bracket_refinement_adapter_irregular_headv3.py
```

Decision: `LOCAL_PRECHECK_ONLY_VALIDATION`; still locked: `REMOTE_SYNC_FULL_TRAIN_MAPPAPER_CLAIM`.

Passed:

```powershell
python tools\abr\validate_abr_shortdiag.py --config configs\adatad\thumos\input_abr_active_bracket_refinement_adapter_irregular_headv3_shortdiag.py
```

Decision: `ONE_EPOCH_TRAIN_LOSS_DIAGNOSTIC_ONLY`; `full_train_unlocked=False`; no metric/sparse-compute/deploy claim.

Passed:

```powershell
python -m py_compile opentad\acquisition\abr\types.py opentad\acquisition\abr\selector.py opentad\acquisition\abr\validators.py tools\abr\audit_abr_pipeline_precheck.py tools\abr\validate_abr_launch_gate.py tools\abr\validate_abr_formal_gate.py tools\abr\validate_abr_shortdiag.py configs\adatad\thumos\input_abr_active_bracket_refinement_adapter_irregular_headv3.py configs\adatad\thumos\input_abr_active_bracket_refinement_adapter_irregular_headv3_formal.py configs\adatad\thumos\input_abr_active_bracket_refinement_adapter_irregular_headv3_shortdiag.py tests\test_abr_core.py tests\test_abr_pipeline_and_gate.py tests\test_abr_formal_gate.py tests\test_abr_shortdiag.py
```

Passed:

```powershell
python -m pytest tests\test_abr_core.py tests\test_abr_pipeline_and_gate.py tests\test_abr_formal_gate.py tests\test_abr_shortdiag.py -q
```

Result: `39 passed, 1 skipped`.

Passed:

```powershell
python tools\abr\validate_abr_launch_gate.py --config configs\adatad\thumos\input_abr_active_bracket_refinement_adapter_irregular_headv3.py
```

Decision: `LOCAL_PRECHECK_ONLY_VALIDATION`; still locked: `REMOTE_SYNC_FULL_TRAIN_MAPPAPER_CLAIM`.

Passed:

```powershell
python tools\abr\validate_abr_shortdiag.py --config configs\adatad\thumos\input_abr_active_bracket_refinement_adapter_irregular_headv3_shortdiag.py
```

Decision: `ONE_EPOCH_TRAIN_LOSS_DIAGNOSTIC_ONLY`; `full_train_unlocked=False`.

Passed:

```powershell
python tools\abr\validate_abr_formal_gate.py --config configs\adatad\thumos\input_abr_active_bracket_refinement_adapter_irregular_headv3_formal.py
```

Decision: `FORMAL_REVIEW_PACKET_ONLY`; `full_train_unlocked=False`; still locked: `FORMAL_FULL_TRAIN_PENDING_REAL_SCOUT_RECALL_EVIDENCE`.

Passed as mock-only precheck summary generation:

```powershell
python tools\abr\audit_abr_pipeline_precheck.py --out-dir .tmp_abr_formal_gate_mock --overwrite --mock-only
```

Result: `PASS_MOCK_PRECHECK_ONLY`; not real precheck and not formal evidence.

## Remaining Blockers

- No real deploy-visible scout recall/coverage evidence has been supplied for formal ABR.
- The current synthetic precheck summary has low first-round bracket recall/coverage and would be rejected by the formal gate.
- Formal full train remains locked.
- `tools/test.py`, evaluation, mAP claims, runtime/sparse-compute claims, deploy claims, paper claims, remote sync, Slurm, and GPU execution remain locked for this stage.
