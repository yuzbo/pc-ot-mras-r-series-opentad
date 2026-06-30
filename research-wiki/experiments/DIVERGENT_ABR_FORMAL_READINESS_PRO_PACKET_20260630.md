# DIVERGENT ABR Formal Readiness Pro Packet 2026-06-30

Route label: `DIVERGENT_INNOVATION_ABR_DO_NOT_MERGE_WITH_C3`

Owned worktree: `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_ABR_FormalGate_Worktree_20260630`

Owned branch: `codex/divergent-abr-formal-gate-20260630`

GitHub branch for review:
`https://github.com/yuzbo/pc-ot-mras-r-series-opentad/tree/codex/divergent-abr-formal-gate-20260630`

Current reviewed HEAD: `1ffc555`

## Decision Needed

Ask GPT-5.5 Pro for an ABR formal-readiness route review, not a metric claim.
The key decision is whether ABR should next:

1. run a real deploy-visible first-round bracket recall diagnostic;
2. change the bracket/scout policy before any GPU diagnostic;
3. run only a one-epoch short diagnostic;
4. remain held until a stronger evidence packet exists.

Do not ask Pro to approve formal full training or mAP claims from the current
evidence. Formal full training remains locked.

## Route Purpose

ABR means Active Bracket Refinement. It is a multi-round active acquisition
route:

- round 0 lays down a sparse scaffold and proposes action/background transition
  brackets from deploy-visible scout evidence;
- round 1 and optional round 2 add probes around uncertain or stale bracket
  boundaries;
- the goal is to progressively narrow action start/end uncertainty while
  spending fewer observations in stable interiors and background.

This route is intentionally different from C3, GlobalRank, BVR-TWB, MDL-Knot,
Event-Surprise, and Boundary Microscope. It must not be interpreted as a C3
selector or a GlobalRank-style top-k selector.

## Current Implementation Summary

Core files to inspect on GitHub:

- `opentad/acquisition/abr/selector.py`
- `opentad/acquisition/abr/policy.py`
- `opentad/acquisition/abr/integration.py`
- `opentad/acquisition/abr/validators.py`
- `opentad/datasets/transforms/end_to_end.py`
- `configs/adatad/thumos/input_abr_active_bracket_refinement_adapter_irregular_headv3.py`
- `configs/adatad/thumos/input_abr_active_bracket_refinement_adapter_irregular_headv3_formal.py`
- `configs/adatad/thumos/input_abr_active_bracket_refinement_adapter_irregular_headv3_shortdiag.py`
- `tools/abr/audit_abr_pipeline_precheck.py`
- `tools/abr/validate_abr_launch_gate.py`
- `tools/abr/validate_abr_formal_gate.py`
- `tools/abr/validate_abr_shortdiag.py`
- `tests/test_abr_core.py`
- `tests/test_abr_pipeline_and_gate.py`
- `tests/test_abr_formal_gate.py`
- `tests/test_abr_shortdiag.py`
- `research-wiki/experiments/DIVERGENT_ABR_FORMAL_READINESS_20260630.md`

Implemented behavior:

- ABR selection runs before `LoadFrames` decoding and strips `gt_segments` /
  `gt_labels` before selector invocation.
- The handoff records window-local selected positions, original dense
  positions, raw `frame_inds`, selected roles, bracket ids, round ledgers, and
  prefix valid masks.
- The formal config disables diagnostic fallback and uses deploy-visible scout
  evidence only.
- Formal-readiness validation explicitly depends on first-round bracket recall
  and transition coverage.
- Zero-transition pseudo-perfect evidence is rejected.
- Ambiguous-mediated transitions are counted and cannot silently become
  `transition_count=0`.

## Current Evidence

Local focused verification in the owned worktree:

```powershell
python -m pytest tests/test_abr_core.py tests/test_abr_pipeline_and_gate.py tests/test_abr_formal_gate.py tests/test_abr_shortdiag.py -q
```

Result: `44 passed, 1 skipped`.

Read-only audit command:

```powershell
python -B tools\abr\validate_abr_formal_gate.py --config configs\adatad\thumos\input_abr_active_bracket_refinement_adapter_irregular_headv3_formal.py
```

Result summary:

- `formal_config_ok=true`
- `allowed_next_action=FORMAL_REVIEW_PACKET_ONLY`
- `full_train_unlocked=false`
- still locked: formal full train, `tools/test.py`, evaluation, mAP/paper
  claim, runtime/sparse-compute claim, deploy claim.

Known formal blocker:

- Current synthetic rich precheck has
  `first_round_bracket_recall = 0.3333333333333333`.
- Current synthetic rich precheck has
  `first_round_transition_coverage = 0.3333333333333333`.
- Current synthetic rich precheck has
  `first_round_missed_transition_count = 4`.
- Formal gate requires high real deploy-visible first-round bracket recall and
  transition coverage before full training can be considered.

## Questions For Pro

Please inspect the GitHub branch and answer in Chinese with the following
sections:

- `Context verdict`
- `Model evidence`
- `Inspected materials`
- `Verdict`
- `Blocking findings`
- `Non-blocking findings`
- `Required fixes or next experiments`
- `Accepted launch/sync/review/Slurm decision`

Specific questions:

1. Does the implementation genuinely match Active Bracket Refinement, or has it
   drifted into a C3/GlobalRank/Boundary-Microscope-like route?
2. Is first-round bracket recall / transition coverage the right formal gate
   for ABR, or should a different route-specific readiness criterion be used?
3. Given the current synthetic recall/coverage is only about `0.3333`, should
   we first run a real deploy-visible scout recall diagnostic, change the
   bracket policy, or run only a one-epoch short diagnostic?
4. Is a one-epoch short diagnostic useful for ABR now, or is it a distraction
   because it cannot answer the core bracket-recall risk?
5. What is the minimal next experiment tree that could make ABR eligible for a
   later formal full-train decision?
6. Are there leakage, tensor/mask, temporal-coordinate, route-label, or C3
   mixing risks that must be fixed before any GPU diagnostic?

## Current Locked Claims

No formal full train, `tools/test.py`, evaluation, final mAP, runtime/FLOPs,
deploy, paper, or sparse-compute claim is unlocked. This packet asks only for
the next ABR route decision.

## Rosetta GPT-5.5 Pro Result 2026-06-30

Rosetta command:

```powershell
rosetta run --pro --port 9223 --host 127.0.0.1 --attach research-wiki/experiments/DIVERGENT_ABR_FORMAL_READINESS_PRO_PACKET_20260630.md <prompt>
```

The protocol-preferred port `9333` was attempted first and failed immediately
with `ECONNREFUSED`; the user-requested Chrome CDP port `9223` succeeded.

Local evidence files:

- `logs/abr_rosetta_pro_formal_readiness_20260630_132527_+0800.stderr.txt`
- `logs/abr_rosetta_pro_formal_readiness_port9223_20260630_132546_+0800.stdout.txt`
- `logs/abr_rosetta_pro_formal_readiness_port9223_20260630_132546_+0800.stderr.txt`

Rosetta metadata:

- model: `gpt-5-5-pro`
- elapsed: `446949ms`
- events: `247`
- conversation: `6a43536e-e0a0-83e8-8519-dcba3b5d9a67`
- message: `12dd0aa8-9ae7-4c0e-a552-2dd4cfb0a81f`

Accepted Pro verdict:

`HOLD_FORMAL_FULL_TRAIN__RUN_REAL_DEPLOY_VISIBLE_FIRST_ROUND_BRACKET_RECALL_DIAGNOSTIC_FIRST`

Accepted launch/sync/review/Slurm decision:

`PASS_ALLOW_ABR_REAL_DEPLOY_VISIBLE_FIRST_ROUND_BRACKET_RECALL_DIAGNOSTIC_ONLY`

Interpretation:

- ABR implementation matches Active Bracket Refinement and does not drift into
  C3, GlobalRank, or Boundary Microscope.
- First-round bracket recall / transition coverage is the right ABR formal
  gate.
- Current synthetic recall/coverage near `0.3333` blocks formal full train.
- A one-epoch short diagnostic is not the next ABR priority because it only
  checks finite loss and cannot answer the core bracket-recall risk.
- The next allowed action is selector-only / no-detector / no-training real
  deploy-visible first-round bracket recall diagnostics scored offline against
  GT boundaries. The selector must not see GT.

Still locked after Pro:

- formal full train;
- Slurm full train;
- remote sync for training;
- `tools/test.py`;
- evaluation/mAP;
- runtime/FLOPs/sparse-compute claim;
- deploy claim;
- paper claim;
- C3/GlobalRank/BVR/MDL combo merge.
