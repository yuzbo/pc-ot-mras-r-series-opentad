# DIVERGENT ABR First-Round Real Scout Diagnostic - 2026-06-30

## Decision

`DIVERGENT_INNOVATION_ABR_DO_NOT_MERGE_WITH_C3` remains locked for formal full training.

The remote real deploy-visible first-round bracket recall diagnostic completed on a 5-video validation subset. It produced real non-fallback scout evidence, but the current raw-video graydiff scout plus ABR first-round bracket policy does not recall enough GT transitions to justify detector training.

## Remote Source

- Remote clone: `/data/home/sczc063/run/yuzibo/OpenTAD_ABR_FormalGate_20260630_fee07c1`
- Branch: `codex/divergent-abr-formal-gate-20260630`
- Commit: `fee07c11d91c5482f8404ce3e04e3d36c59da151`
- Output dir: `/data/home/sczc063/run/yuzibo/OpenTAD_ABR_FormalGate_20260630_fee07c1/logs/abr_deploy_visible_scout_recall_20260630_143348_+0800/`
- Scout JSON: `abr_deploy_visible_scout_validation_max5.json`
- Matched audit JSON: `abr_first_round_recall_validation_max5_matched.json`

## Scout Export

Command scope:

- Annotation: `/data/home/sczc063/run/yuzibo/thumos14/annotations/thumos_14_anno.json`
- Video root: `/data/home/sczc063/run/yuzibo/thumos14/test`
- Subset: `validation`
- `--curve-len 384`
- `--resize 96`
- `--max-videos 5`

Result:

- `status=PASS_DEPLOY_VISIBLE_SCOUT_EXPORT`
- `success_count=5`
- `missing_count=0`
- `scout_source=deploy_visible_raw_video_graydiff_v1`
- `deploy_visible_only=true`
- `no_gt_in_output=true`
- `no_teacher=true`
- `no_detector=true`
- `no_prediction_cache=true`

## First-Round Recall Audit

Matched audit command used the same 5-video limit:

```bash
/data/home/sczc063/run/yuzibo/conda_envs/opentad/bin/python \
  tools/abr/audit_abr_first_round_bracket_recall.py \
  --annotation-json /data/home/sczc063/run/yuzibo/thumos14/annotations/thumos_14_anno.json \
  --scout-json /data/home/sczc063/run/yuzibo/OpenTAD_ABR_FormalGate_20260630_fee07c1/logs/abr_deploy_visible_scout_recall_20260630_143348_+0800/abr_deploy_visible_scout_validation_max5.json \
  --subset validation \
  --max-videos 5 \
  --out-json /data/home/sczc063/run/yuzibo/OpenTAD_ABR_FormalGate_20260630_fee07c1/logs/abr_deploy_visible_scout_recall_20260630_143348_+0800/abr_first_round_recall_validation_max5_matched.json
```

Result:

- `status=LOCKED`
- `allowed_next_action=LOCKED_REAL_SCOUT_RECALL_BELOW_FORMAL_GATE_REVISE_BRACKET_POLICY_OR_SCOUT`
- `real_deploy_visible_recall_evidence=true`
- `selector_gt_visible=false`
- `diagnostic_fallback_used=false`
- `video_count=5`
- `window_count=5`
- `transition_count=124`
- `bracketed_transition_count=30`
- `missed_transition_count=94`
- `first_round_bracket_recall=0.24193548387096775`
- `first_round_transition_coverage=0.11290322580645161`
- `temporal_coverage_fraction=0.2109375`
- `formal_gate_passed=false`
- `formal_full_train_unlocked=false`
- `tools_test_allowed=false`
- `paper_claim=false`

## Interpretation

This is useful negative diagnostic evidence for the current ABR scout/bracket configuration. It does not reject the ABR idea itself. It says that the current first-round raw graydiff scout and bracket policy miss too many action transitions, especially dense short-transition regions, so detector training would be premature and likely uninterpretable.

ABR remains a second-stage route. To revive it, revise first-round scout or bracket policy before any formal training. Candidate fixes include broader first-round brackets, duration-aware minimum bracket coverage, a stronger deploy-visible scout signal, or an explicit multi-round probe simulator that proves first-round recall before spending GPU on detector training.

## Locked Claims

No Slurm child, GPU training, `tools/train.py`, `tools/test.py`, official mAP, runtime/FLOPs, deploy, paper, sparse-compute, or route-success claim is unlocked by this diagnostic.
