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

## Selector-Only First-Round GT Boundary Recall Diagnostic

Timestamp: `2026-06-30T13:49:38.0558278+08:00`

This update implements the GPT-5.5 Pro requested ABR next-step diagnostic:
selector-only / no-detector / no-training first-round bracket recall against
real GT boundaries. The selector still cannot see GT. The diagnostic builds the
selector payload from deploy-visible scout records only, runs ABR round-0
bracket proposal, and only then scores bracket coverage against THUMOS-style
annotations offline.

New CLI:

```powershell
python tools\abr\audit_abr_first_round_bracket_recall.py --annotation-json <thumos_14_anno.json> --scout-json <deploy_visible_scout.json> --subset validation --out-json <audit.json>
```

Changed files:

- `tools/abr/audit_abr_first_round_bracket_recall.py`
- `tests/test_abr_first_round_bracket_recall.py`
- `opentad/acquisition/abr/validators.py`
- `research-wiki/experiments/DIVERGENT_ABR_FORMAL_READINESS_20260630.md`

The CLI output schema includes:

- route label, diagnostic-only flag, no-detector flag, no-training flag
- `selector_gt_visible=false`
- scout sources, video/window counts
- GT transition count, bracketed/missed transition count
- first-round bracket recall and action transition-pair coverage
- temporal coverage fraction, bracket width stats, false-positive bracket density
- short-action stratified recall
- class/video/window misses
- ambiguous transition count
- explicit claim locks:
  `formal_full_train_unlocked=false`,
  `tools_test_allowed=false`,
  `official_map_claim=false`,
  `runtime_flops_claim=false`,
  `deploy_claim=false`,
  `sparse_compute_claim=false`,
  `paper_claim=false`

Fail-closed behavior:

- Rejects GT keys in selector/scout payloads before selection:
  `gt_segments`, `gt_labels`.
- Rejects teacher/cache/detector/dense-backbone shortcuts before selection:
  teacher logits/features, prediction cache, raw detector outputs, detector
  predictions, detector feedback, dense backbone handoff, and GT/teacher/cache
  derived scout curves.
- Rejects route drift tokens such as `GlobalRank` / C3-Pro style strings.
- Missing deploy-visible scout input fails closed unless explicit diagnostic
  fallback is requested.
- Explicit diagnostic fallback can produce a diagnostic JSON but is marked
  `real_deploy_visible_recall_evidence=false` and
  `allowed_next_action=LOCKED_DIAGNOSTIC_FALLBACK_NOT_REAL_RECALL_EVIDENCE`.
- Zero-transition pseudo-perfect evidence is rejected as
  `LOCKED_ZERO_TRANSITION_NO_REAL_RECALL_EVIDENCE`, with recall/coverage set to
  `0.0` rather than a fake perfect score.
- Ambiguous-only GT artifacts are counted but cannot unlock real recall
  evidence.

Local verification:

Passed:

```powershell
python -m py_compile tools\abr\audit_abr_first_round_bracket_recall.py opentad\acquisition\abr\validators.py tests\test_abr_first_round_bracket_recall.py
```

Passed:

```powershell
python -m pytest tests\test_abr_first_round_bracket_recall.py -q
```

Result: `8 passed`.

Passed:

```powershell
python -m pytest tests\test_abr_core.py tests\test_abr_pipeline_and_gate.py tests\test_abr_formal_gate.py tests\test_abr_shortdiag.py tests\test_abr_first_round_bracket_recall.py -q
```

Result: `52 passed, 1 skipped`.

Passed:

```powershell
python -m py_compile tools\abr\audit_abr_first_round_bracket_recall.py tools\abr\audit_abr_pipeline_precheck.py tools\abr\validate_abr_formal_gate.py opentad\acquisition\abr\selector.py opentad\acquisition\abr\policy.py opentad\acquisition\abr\validators.py opentad\acquisition\abr\types.py opentad\acquisition\abr\integration.py tests\test_abr_first_round_bracket_recall.py tests\test_abr_core.py tests\test_abr_pipeline_and_gate.py tests\test_abr_formal_gate.py tests\test_abr_shortdiag.py
```

Decision:

- This implements a real selector-only first-round bracket recall diagnostic.
- It does not unlock formal full training.
- It does not allow `tools/test.py`, detector evaluation, remote sync, Slurm,
  runtime/FLOPs claims, sparse-compute claims, deploy claims, mAP claims, or
  paper claims.
- Shared tracker/log files were not touched in this update because this task's
  hard write boundary allowed writes only inside the owned worktree and only to
  the listed ABR files/reports.

### Acceptance Fix: Real Evidence Must Still Pass Formal Recall Thresholds

Timestamp: `2026-06-30T13:54:11.3042880+08:00`

Acceptance finding:

The first implementation treated any real deploy-visible scout evidence as
eligible for `FORMAL_REVIEW_PACKET_ONLY_WITH_REAL_SCOUT_RECALL_EVIDENCE`, even
when first-round recall/coverage was low. This was too optimistic: real
evidence proves the diagnostic is valid, not that the bracket policy is formal
gate ready.

Fix:

- Added `formal_thresholds` to the diagnostic JSON:
  - `min_first_round_bracket_recall = 0.95`
  - `min_first_round_transition_coverage = 0.95`
- Added `formal_gate_passed`.
- `allowed_next_action` now becomes
  `LOCKED_REAL_SCOUT_RECALL_BELOW_FORMAL_GATE_REVISE_BRACKET_POLICY_OR_SCOUT`
  when real deploy-visible evidence exists but recall/coverage is below
  threshold or `missed_transition_count > 0`.
- `FORMAL_REVIEW_PACKET_ONLY_WITH_REAL_SCOUT_RECALL_EVIDENCE` is returned only
  when evidence is real, fallback-free, selector-GT-free, recall/coverage pass
  thresholds, and `missed_transition_count == 0`.

Additional test coverage:

- Perfect real scout evidence asserts `formal_gate_passed=True` and formal
  thresholds are present.
- Low-recall real scout evidence asserts:
  - `real_deploy_visible_recall_evidence=True`
  - recall/coverage below `0.95`
  - `missed_transition_count > 0`
  - `formal_gate_passed=False`
  - locked next action:
    `LOCKED_REAL_SCOUT_RECALL_BELOW_FORMAL_GATE_REVISE_BRACKET_POLICY_OR_SCOUT`

Verification:

Passed:

```powershell
python -m pytest tests\test_abr_first_round_bracket_recall.py -q
```

Result: `9 passed`.

Passed:

```powershell
python -m py_compile tools\abr\audit_abr_first_round_bracket_recall.py tests\test_abr_first_round_bracket_recall.py
```

Passed:

```powershell
python -m pytest tests\test_abr_core.py tests\test_abr_pipeline_and_gate.py tests\test_abr_formal_gate.py tests\test_abr_shortdiag.py tests\test_abr_first_round_bracket_recall.py -q
```

Result: `53 passed, 1 skipped`.

Passed:

```powershell
python -m py_compile tools\abr\audit_abr_first_round_bracket_recall.py tools\abr\audit_abr_pipeline_precheck.py tools\abr\validate_abr_formal_gate.py opentad\acquisition\abr\selector.py opentad\acquisition\abr\policy.py opentad\acquisition\abr\validators.py opentad\acquisition\abr\types.py opentad\acquisition\abr\integration.py tests\test_abr_first_round_bracket_recall.py tests\test_abr_core.py tests\test_abr_pipeline_and_gate.py tests\test_abr_formal_gate.py tests\test_abr_shortdiag.py
```

Passed:

```powershell
git diff --check
```

Result: no whitespace errors; Git reported only existing LF/CRLF working-copy
warnings.

Locks remain unchanged: no formal full train, no `tools/test.py`, no detector
evaluation, no remote sync, no Slurm, no mAP/runtime/FLOPs/sparse-compute/deploy
or paper claim.

### Acceptance Fix: UTF-8 BOM JSON Compatibility

Timestamp: `2026-06-30T13:57:22.8971088+08:00`

Acceptance finding:

Windows/PowerShell generated annotation or scout files may include a UTF-8 BOM.
The original JSON loader used plain `utf-8`, which failed closed with
`Unexpected UTF-8 BOM (decode using utf-8-sig)`. This does not create leakage,
but it blocks ordinary local acceptance usage.

Fix:

- `tools/abr/audit_abr_first_round_bracket_recall.py` now reads annotation
  JSON with `utf-8-sig`.
- Scout JSON and JSONL are also read with `utf-8-sig`.
- Existing fail-closed validation for GT, teacher, prediction-cache, detector,
  dense-backbone, route-drift, fallback, and zero-transition cases is unchanged.

Additional test coverage:

- BOM annotation JSON + BOM scout JSON accepted.
- BOM scout JSONL accepted.

Verification:

Passed:

```powershell
python -m pytest tests\test_abr_first_round_bracket_recall.py -q
```

Result: `11 passed`.

Passed:

```powershell
python -m py_compile tools\abr\audit_abr_first_round_bracket_recall.py tests\test_abr_first_round_bracket_recall.py
```

Passed:

```powershell
python -m pytest tests\test_abr_core.py tests\test_abr_pipeline_and_gate.py tests\test_abr_formal_gate.py tests\test_abr_shortdiag.py tests\test_abr_first_round_bracket_recall.py -q
```

Result: `55 passed, 1 skipped`.

Passed:

```powershell
python -m py_compile tools\abr\audit_abr_first_round_bracket_recall.py tools\abr\audit_abr_pipeline_precheck.py tools\abr\validate_abr_formal_gate.py opentad\acquisition\abr\selector.py opentad\acquisition\abr\policy.py opentad\acquisition\abr\validators.py opentad\acquisition\abr\types.py opentad\acquisition\abr\integration.py tests\test_abr_first_round_bracket_recall.py tests\test_abr_core.py tests\test_abr_pipeline_and_gate.py tests\test_abr_formal_gate.py tests\test_abr_shortdiag.py
```

Passed:

```powershell
git diff --check
```

Result: no whitespace errors; Git reported only LF/CRLF working-copy warnings.

Locks remain unchanged: no commit/push, no remote, no Slurm, no training, no
`tools/test.py`, no detector evaluation, no formal full train, and no
mAP/runtime/FLOPs/sparse-compute/deploy/paper claim.
