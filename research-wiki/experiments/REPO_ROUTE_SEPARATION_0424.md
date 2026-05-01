---
type: experiment
node_id: exp:REPO_ROUTE_SEPARATION_0424
title: "Repository route separation: OpenTAD vs OpenTAD_Back"
status: active
created_at: 2026-04-24T20:58:14+08:00
updated_at: 2026-04-24T20:58:14+08:00
---

# Repository route separation: OpenTAD vs OpenTAD_Back

## One-line thesis

Current experiment reading must separate `OpenTAD` frame-sampling / scorer results from `OpenTAD_Back` input / irregular-detector / completion results; mixing them creates false baselines and wrong next-step decisions.

## Why this page exists

- The wiki currently stores results from both repositories in one place.
- Several strong-looking numbers belong to different carriers and should not be used as interchangeable baselines.
- The most common confusion is to compare `OpenTAD` sampler-side results directly against `OpenTAD_Back` completion / irregular-detector results as if they were the same method family.

## Route A: `OpenTAD`

### Scope

- `OpenTAD` is the older frame-sampling / scorer repository.
- Its main lines are `soft gating`, `hard scorer`, `mask_adapter`, and related bugfix ablations.

### Reliable anchors

- `v3_hard_scorer_aux_bugfix0420 = 56.97`
- `v3_gumbel_softgate_minimal_bugfix0420 = 54.22`
- `v3_gumbel_softgate_coarse_det_bugfix0420 = 53.17`

### Special caution

- `R015 = 64.51` is still **unverified** from remote logs and must not be used as an authoritative anchor until rechecked on server.

### Current role

- `OpenTAD` is now a **reference line**, not the current main execution repo.
- Use it for sampler-side context only.

## Route B: `OpenTAD_Back`

### Scope

- `OpenTAD_Back` is the active mainline repo.
- Its main lines are `input-side sparse controls`, `irregular head / bridge`, and `completion-first`.

### Reliable anchors

- `INPUT_ORACLE_BOUNDARY_DENSE_0411 = 66.01` (`OpenTAD_Back` oracle upper bound)
- `INPUT_STRIDE2_0410 = 65.09` (`OpenTAD_Back` dense 50% upper reference)
- `INPUT_RANDOMFIX_0410 = 63.12` (`OpenTAD_Back` strong non-oracle sparse-input baseline)
- `completion_nearest_dense_baseline = 54.03`
- `HEADV3_OABS_FULL_X_LEGACYSINGLETON = 53.48`
- `topk9 + binary-cls = 43.44`

### Current live mainline

- `completion_oracle_upper_ampoff`
- `completion_full_oracle_dense_control_ampoff`
- `completion_scatter_dense_baseline_ampoff` queued as the missing `native-axis 768 scatter-only` control
- `completion_learned_minimal_ampoff`

### Current role

- `OpenTAD_Back` is the repository that determines the next method decision.
- All immediate route-gating conclusions should default to `OpenTAD_Back`, unless a note explicitly says `OpenTAD`.

## Comparison rules

1. Compare **within the same repository first**.
2. Separate **oracle**, **non-oracle strong baseline**, and **current method line**.
3. Treat `R015 = 64.51` as **unverified** until remote-log confirmation.
4. Inside `OpenTAD_Back`, separate:
   - `selected-axis / remap=True / compact 384`
   - `native-axis / remap=False / dense 768`
5. Do not read `54.03` vs `63.12` as a pure completion A/B; that comparison also changes axis and detector contract.

## Immediate implication

- `OpenTAD_Back` next-step priority is not another `OpenTAD` scorer ablation.
- The highest-information path is to finish the `OpenTAD_Back` completion decision ladder:
  - oracle upper bound
  - full oracle dense control
  - scatter-only native-axis control
  - learned minimal completion
- If the native-axis `768` family remains weak even with oracle support, the next fallback should be `compact 384` refinement / completion rather than more native-axis polishing.
