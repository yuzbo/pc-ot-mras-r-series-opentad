# DIVERGENT ABR Scout Policy 20260630

Timestamp: `2026-06-30T15:12:24.0985563+08:00`

Route label: `DIVERGENT_INNOVATION_ABR_DO_NOT_MERGE_WITH_C3`  
Owned worktree: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_ABR_ScoutPolicy_Worktree_20260630`  
Owned branch: `codex/divergent-abr-scout-policy-20260630`

## Purpose

This stage replaces the weak first-round scaffold-pair bracket policy with a deploy-visible, multi-scale scout/bracket policy intended to improve real first-round bracket recall before any ABR second/third-round active refinement or formal training.

No remote sync, SSH, Slurm, training, `tools/test.py`, detector evaluation, mAP/runtime/deploy/paper claim, or formal full-train unlock was performed.

## Changed Surface

- Input sampling / selector policy: changed.
- Dynamic budget controller: not changed.
- Token compression: not changed.
- Adapter/backbone/neck/head internals: not changed.
- Detector head logic, losses, assignment, evaluator, post-processing: not changed.

## Policy Mechanism

Policy name: `deploy_visible_multiscale_graydiff_bracket_v2`.

The first-round bracket generator now combines:

- original scaffold-pair brackets as a fallback baseline;
- full-curve deploy-visible scout transitions, including ambiguous-mediated background/action changes;
- multi-scale smoothed peak brackets;
- local gradient-spike brackets;
- uncertainty widening around candidate transitions;
- short-action boundary protection for compact peaks;
- max-gap span expansion;
- a first-round temporal coverage guard capped at `0.70` for normal windows.

Selector payload remains GT-free. GT is read only by the offline audit after selection for scoring bracket recall/coverage.

## Safety And Gates

- Formal config remains locked with `full_train_unlocked=False`.
- Formal gate now also rejects overwide first-round evidence through `max_first_round_temporal_coverage_fraction=0.70`.
- Low real-scout recall still fails closed with `LOCKED_REAL_SCOUT_RECALL_BELOW_FORMAL_GATE_REVISE_BRACKET_POLICY_OR_SCOUT`.
- Diagnostic fallback remains non-evidence.
- Next allowed action is at most `REAL_SCOUT_RECALL_DIAGNOSTIC_ONLY`; formal training remains locked.

## Verification

- `python -m pytest tests/test_abr_core.py tests/test_abr_first_round_bracket_recall.py tests/test_abr_formal_gate.py -q`
  - Result: `31 passed in 0.53s`
- `python -m pytest tests -k abr -q`
  - Result: `64 passed, 1 skipped, 42 deselected in 2.41s`
- `python -m py_compile opentad\acquisition\abr\types.py opentad\acquisition\abr\policy.py opentad\acquisition\abr\selector.py opentad\acquisition\abr\validators.py tools\abr\audit_abr_first_round_bracket_recall.py tools\abr\validate_abr_formal_gate.py tools\abr\validate_abr_shortdiag.py configs\adatad\thumos\input_abr_active_bracket_refinement_adapter_irregular_headv3.py configs\adatad\thumos\input_abr_active_bracket_refinement_adapter_irregular_headv3_formal.py tests\test_abr_core.py tests\test_abr_first_round_bracket_recall.py tests\test_abr_formal_gate.py`
  - Result: passed
- `python tools\abr\validate_abr_formal_gate.py --config configs\adatad\thumos\input_abr_active_bracket_refinement_adapter_irregular_headv3_formal.py`
  - Result: `allowed_next_action=FORMAL_REVIEW_PACKET_ONLY`, `full_train_unlocked=false`
- `python tools\abr\validate_abr_shortdiag.py --config configs\adatad\thumos\input_abr_active_bracket_refinement_adapter_irregular_headv3_shortdiag.py`
  - Result: `allowed_next_action=ONE_EPOCH_TRAIN_LOSS_DIAGNOSTIC_ONLY`, `full_train_unlocked=false`
- `python tools\abr\validate_abr_launch_gate.py --config configs\adatad\thumos\input_abr_active_bracket_refinement_adapter_irregular_headv3.py`
  - Result: `allowed_next_action=LOCAL_PRECHECK_ONLY_VALIDATION`, still locked for remote sync/full train/mAP/paper claim

## Remaining Blockers

- The remote real 5-video evidence before this patch was low: `first_round_bracket_recall=0.241935`, `transition_coverage=0.112903`.
- This patch is local-only and has not produced new real-video recall evidence.
- Formal train remains locked until a new deploy-visible `REAL_SCOUT_RECALL_DIAGNOSTIC_ONLY` audit demonstrates acceptable first-round bracket recall/coverage without GT/teacher/detector/cache leakage and without overwide temporal coverage.
- GPT-5.5 Pro was not used in this stage because the user explicitly forbade Pro/Gemini/Claude for this code-owner task.

## 2026-06-30 Repair Stage - Scout Bracket Coverage Audit

Timestamp: `2026-06-30T18:24:34+08:00`

Owned worktree: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_ABR_ScoutPolicyRepair_Worktree_20260630`
Owned branch: `codex/divergent-abr-scout-policy-repair-20260630`
Route label: `DIVERGENT_INNOVATION_ABR_DO_NOT_MERGE_WITH_C3`

### Repair Summary

This repair fixes the first-round audit/selector mismatch and adds deploy-visible bounded risk mechanisms, but it does **not** unlock formal/full training. The previous real CPU diagnostic observed `first_round_bracket_recall=0.16129032258064516`, `first_round_transition_coverage=0.06451612903225806`, `20/124` transitions bracketed, `fallback=false`, and `selector-GT=false`.

Implemented changes:

- First-round bracket audit now scores the round-0 bracket snapshot recorded in the round ledger, rather than later narrowed/split brackets.
- ABR policy adds bounded event-train risk envelopes from deploy-visible gray-difference salience/gradient anchors.
- ABR policy adds small risk-gap mechanisms: `risk_gap_micro_bridge`, `silent_gap_sentinel`, and start-edge `temporal_edge_risk_guard`; `max_gap<=0` disables these priors.
- The first-round guard ranks candidates by risk density with source-aware weighting, penalizes plain scaffold pairs, and keeps the formal temporal coverage cap.
- Formal evidence now records and validates `max_first_round_bracket_width_fraction=0.30`; the audit payload exposes `max_bracket_width_fraction`.
- Formal config remains `full_train_unlocked=false`; no remote sync, Slurm, training, evaluation, `tools/test.py`, mAP/runtime/deploy/paper claim, or formal unlock was performed.

### Final Real Deploy-Visible Audit

Command:

```powershell
python tools\abr\audit_abr_first_round_bracket_recall.py --annotation-json E:\DeskTop\TAD\temrefuse-tad\figures\cache\thumos_14_anno.json --scout-json .tmp_abr_repair_real_scout\scout_5.json --subset validation --max-videos 5 --out-json .tmp_abr_repair_real_scout\audit_final_after_tests_5.json --k0 96 --k1-cap 160 --k2-cap 64 --max-total-k 384 --max-gap 24
```

Result: exit code `1`, `status=LOCKED`.

Final evidence:

- `transition_count=124`
- `bracketed_transition_count=95`
- `missed_transition_count=29`
- `first_round_bracket_recall=0.7661290322580645`
- `first_round_transition_coverage=0.7258064516129032`
- `temporal_coverage_fraction=0.6567708333333333`
- `max_bracket_width_fraction=0.296875`
- `diagnostic_fallback_used=false`
- `selector_gt_visible=false`
- `real_deploy_visible_recall_evidence=true`
- `formal_gate_passed=false`
- `allowed_next_action=LOCKED_REAL_SCOUT_RECALL_BELOW_FORMAL_GATE_REVISE_BRACKET_POLICY_OR_SCOUT`

Interpretation: the repair substantially improves real deploy-visible first-round recall versus the reproduced baseline `0.16129032258064516`, and it stays under the coverage/width guards, but it still misses too many real transitions, mainly short `TennisSwing` clusters. ABR formal/full training remains locked. The route is at most suitable for further local `PRECHECK` or `SHORT_DIAGNOSTIC_ONLY` policy diagnostics, not formal training.

### Verification

- `python -m pytest tests\test_abr_core.py tests\test_abr_first_round_bracket_recall.py tests\test_abr_formal_gate.py -q`
  - Result: `34 passed in 0.53s`
- `python -m py_compile opentad\acquisition\abr\policy.py opentad\acquisition\abr\selector.py opentad\acquisition\abr\types.py opentad\acquisition\abr\validators.py tools\abr\audit_abr_first_round_bracket_recall.py tools\abr\validate_abr_formal_gate.py`
  - Result: passed
- `python tools\abr\validate_abr_formal_gate.py --config configs\adatad\thumos\input_abr_active_bracket_refinement_adapter_irregular_headv3_formal.py`
  - Result: `formal_config_ok=true`, `full_train_unlocked=false`, `allowed_next_action=FORMAL_REVIEW_PACKET_ONLY`

## 2026-06-30 Practical Readiness Repair - LoadFrames And Diagnostic Gates

Timestamp: `2026-06-30T20:54:00+08:00`

Owned worktree: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_ABR_ScoutPolicyRepair_Worktree_20260630`
Owned branch: `codex/divergent-abr-scout-policy-repair-20260630`
Route label: `DIVERGENT_INNOVATION_ABR_DO_NOT_MERGE_WITH_C3`

### Readiness Decision

ABR is locally ready for bounded non-claim diagnostics, not for formal metric claims or full training.

The formal first-round recall gate remains a formal-readiness and claim gate. It must not be used to permanently block all ABR work. Protocol-safe local work remains allowed when it is explicitly labeled `PRECHECK_ONLY` or `SHORT_DIAGNOSTIC_ONLY`, avoids validation/test GT, teacher, detector feedback, raw prediction/cache shortcuts, and does not run evaluation or make mAP/runtime/sparse-compute/deploy/paper claims.

The one-epoch short diagnostic scope is stability and direction only. Low short diagnostic mAP, smoke numbers, or early training numbers must not be used as final route rejection evidence unless the run exposes a hard failure such as NaN/Inf, OOM, traceback/runtime error, no-space, leakage, evaluator shortcut, or tensor/coordinate/mask contract breakage.

### Implementation Checks Added

- `tools/abr/validate_abr_launch_gate.py`, `tools/abr/validate_abr_formal_gate.py`, and `tools/abr/validate_abr_shortdiag.py` now fail closed unless the real OpenTAD `LoadFrames` source contains the ABR integration hook.
- `opentad/acquisition/abr/validators.py` now exposes `assert_loadframes_abr_integration(...)`, which checks that `LoadFrames` has the `abr_active_bracket_refinement` branch, calls `apply_abr_to_results`, strips `gt_segments`/`gt_labels` before selection, passes `dense_window`, and propagates ABR handoff metadata.
- `tools/abr/validate_abr_shortdiag.py` now returns explicit fields:
  - `formal_recall_gate=not_required_for_short_diagnostic_only`
  - `short_diagnostic_judgment_scope=stability_direction_only_no_route_rejection_by_low_map`
- Focused tests now cover:
  - formal gate does not directly block short diagnostic config validation;
  - missing LoadFrames ABR hook fails closed;
  - real LoadFrames integration is present;
  - multi-round ABR narrows at least one round-0 bracket and exposes detector-ready metadata (`selected_positions`, roles, bracket ids, provenance, single detector forward count);
  - selected raw frame indices and handoff metadata remain sorted, sparse, local/global consistent, and GT/teacher/cache-free.

### Current Config State

- Short diagnostic config exists: `configs/adatad/thumos/input_abr_active_bracket_refinement_adapter_irregular_headv3_shortdiag.py`.
- Formal locked config exists: `configs/adatad/thumos/input_abr_active_bracket_refinement_adapter_irregular_headv3_formal.py`.
- Base precheck config exists: `configs/adatad/thumos/input_abr_active_bracket_refinement_adapter_irregular_headv3.py`.
- Real `LoadFrames` ABR integration exists in `opentad/datasets/transforms/end_to_end.py`, but this repair did not edit that file because it is outside the current allowed write scope.

### Still Locked

- Formal full training.
- Remote sync and Slurm launch.
- `tools/test.py`, evaluator runs, raw prediction shortcuts, checkpoint/result claims.
- mAP, runtime/FLOPs, sparse-compute, deploy, and paper claims.
- Any claim that ABR is deployment-safe.

### Verification

- `python -m pytest tests\test_abr_shortdiag.py tests\test_abr_pipeline_and_gate.py tests\test_abr_core.py -q`
  - Result: `37 passed, 1 skipped in 2.22s`

## 2026-06-30 Recall Repair 3 - Robust Local Change Brackets

Timestamp: `2026-06-30T23:13:46+08:00`

Owned worktree: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_ABR_RecallRepair3_Worktree_20260630`
Owned branch: `codex/divergent-abr-recall-repair3-20260630`
Route label: `DIVERGENT_INNOVATION_ABR_DO_NOT_MERGE_WITH_C3`

### Repair Summary

This local-only repair targets the first-round deploy-visible scout/bracket recall failure where short subthreshold raw-video graydiff events between sparse scaffold points produced no round-0 bracket. It preserves ABR's intended route: round 0 finds cheap scout-derived candidate brackets, later rounds may actively probe uncertain boundaries, and no GT/teacher/detector-cache/dense-backbone shortcut is introduced.

Changed files:

- `opentad/acquisition/abr/policy.py`
- `opentad/acquisition/abr/selector.py`
- `tests/test_abr_first_round_bracket_recall.py`

Mechanism added:

- New robust local-change/extrema candidate generator inside the existing `deploy_visible_multiscale_graydiff_bracket_v2` first-round policy.
- The generator uses only the deploy-visible scout curve, smoothed local contrast, robust median/MAD salience thresholds, multi-radius left/right window changes, curvature, and local extrema prominence.
- Candidate windows are merged only while still short; otherwise candidates compete through the existing priority/dedupe and first-round coverage guard.
- Diagnostic ledger now lists `robust_local_change_extrema_brackets`.

TDD evidence:

- Added synthetic subthreshold short-action-between-scaffold test. Before the fix it failed with `first_round_bracket_recall=0.0`; after the fix it passes with full first-round boundary/action coverage under tight width/density bounds.

### Verification

- `python -m pytest tests\test_abr_first_round_bracket_recall.py::test_robust_change_policy_brackets_subthreshold_short_action_between_sparse_scaffold_points -q`
  - Result: `1 passed`
- `python -m pytest tests\test_abr_core.py tests\test_abr_pipeline_and_gate.py tests\test_abr_first_round_bracket_recall.py tests\test_abr_deploy_visible_scout_export.py tests\test_abr_formal_gate.py -q`
  - Result: `50 passed, 1 skipped in 2.04s`
- `python tools\abr\validate_abr_formal_gate.py --config configs\adatad\thumos\input_abr_active_bracket_refinement_adapter_irregular_headv3_formal.py`
  - Result: `formal_config_ok=true`, `full_train_unlocked=false`, `allowed_next_action=FORMAL_REVIEW_PACKET_ONLY`
- `python tools\abr\validate_abr_launch_gate.py --config configs\adatad\thumos\input_abr_active_bracket_refinement_adapter_irregular_headv3.py`
  - Result: `allowed_next_action=LOCAL_PRECHECK_ONLY_VALIDATION`, `still_locked=REMOTE_SYNC_FULL_TRAIN_MAPPAPER_CLAIM`

### Still Locked

- No remote sync, Slurm, GPU training/evaluation, `tools/test.py`, staging, commit, push, mAP/runtime/sparse-compute/deploy/paper claim, or formal train unlock was performed.
- This repair has synthetic recall evidence only in this code-owner pass. The remote/main process must harvest the pending full real raw-video diagnostic before any route-level conclusion.
