---
type: experiment
node_id: exp:SOFT_BRIDGE_SUPERVISION_CLOSURE_0423
title: "Native y soft-bridge supervision closure: topk, cls target, and toy joint diagnosis"
date: 2026-04-23
status: completed
created_at: 2026-04-23T23:40:00+08:00
updated_at: 2026-04-24T11:26:05+08:00
---

# SOFT_BRIDGE_SUPERVISION_CLOSURE_0423

## Scope

This page closes the unfinished `0420` soft-bridge branch by consolidating the already completed but previously unrecorded results from:

- full-train supervision diagnostics
- branch-decoupled true-overfit diagnostics
- toy joint-repair diagnostics

The goal is to answer three questions cleanly:

1. under full-train, is the dominant bottleneck `topk`, cls-target type, or warmup?
2. in toy overfit, is the first hard failure `cls`, `reg`, or the joint interaction?
3. what is the highest-information next full-train pair now that these diagnostics are closed?

## Full-Train Supervision Diagnostics

| Variant | Avg-mAP | Read |
| --- | ---: | --- |
| `topk1 + binary-cls` | `38.98` | latest no-warmup rerun on `24013` |
| `topk1 + binary-cls + regwarm10` | `37.64` | completed mainline on `25876` |
| `topk1 + soft-cls` | `38.98` | completed supervision diag on `25876` |
| `topk3 + soft-cls` | `42.26` | completed supervision diag on `24013` |
| `topk9 + binary-cls` | `43.44` | completed supervision diag on `24013` |

## 0424 Transfer Pair Final Read

| Variant | Avg-mAP | mAP@0.5 | mAP@0.7 | Read |
| --- | ---: | ---: | ---: | --- |
| `topk9 + binary-cls` | `43.44` | `46.17` | `15.41` | frozen strongest carrier baseline |
| `topk9 + binary-cls + regwarm10` | `43.05` | `45.50` | `15.24` | completed on `25876`; near-neutral transfer |
| `topk9 + binary-cls + detach-cls` | `38.56` | `39.82` | `13.20` | completed on `24013`; clearly negative transfer |

## Toy Overfit Closure

| Variant | Final loss | Read |
| --- | ---: | --- |
| `soft cls-only single_video` | `0.0537` | near-memorized |
| `soft reg-only single_video` | `0.6121` | failed |
| `soft cls-only single_instance` | `0.0663` | near-memorized |
| `soft reg-only single_instance` | `0.5840` | failed |
| `oracle-point reg-only single_instance` | `0.0009` | memorized |
| `hardjoint single_instance` | `0.0103` | memorized |
| `hardjoint single_video` | `0.0079` | memorized |
| `topk1 reg-only single_video` | `0.0128` | memorized |
| `topk2 reg-only single_instance` | `0.0161` | memorized |
| `topk3 reg-only single_instance` | `0.5926` | still failed |
| `binary-reg reg-only single_instance` | `0.9200` | worse than baseline |
| `topk1 + binary-cls joint single_instance` | `0.0037` | memorized |
| `topk1 + binary-cls joint single_video` | `1.0141` | catastrophic joint failure |
| `topk1 + binary-cls cls-only single_video` | `0.0000` | perfect fit |
| `topk1 + binary-cls joint single_video + detach-cls` | `0.0480` | strong rescue |
| `topk1 + binary-cls keep2gt joint` | `1.0141` | already fails at 2 GT |
| `topk1 + binary-cls keep2gt joint + detach-cls` | `0.0480` | strong rescue |
| `topk1 + binary-cls joint + clsw0.1` | `0.4783` | partial rescue only |
| `topk1 + binary-cls joint + regwarm100` | `0.0055` | strong rescue |
| `topk1 + binary-cls joint + clsw0.1 -> resume regwarm` | `0.0043` | strong rescue |

## Closed Findings

1. `topk` is the dominant full-train supervision lever on the current native-`y` bridge.
   - `topk1 + binary-cls = 38.98`
   - `topk9 + binary-cls = 43.44`
   - gain: `+4.46`

2. cls-target type is secondary to support size on the current carrier.
   - `topk1 + binary-cls = 38.98`
   - `topk1 + soft-cls = 38.98`
   - the current full-train gap is not explained by "binary vs soft cls target" alone

3. the raw regression branch is not intrinsically broken.
   - original soft `reg-only` fails (`0.6121 / 0.5840`)
   - but oracle-point, hard assignment, and tighter-topk reg-only controls all fit (`0.0009 / 0.0103 / 0.0079 / 0.0128 / 0.0161`)
   - so the true blocker is diffuse soft support under joint optimization, not lack of reg capacity

4. the first catastrophic failure happens at multi-GT joint training, not at isolated cls-only or isolated single-GT joint training.
   - `topk1 + binary-cls joint single_instance = 0.0037`
   - `topk1 + binary-cls joint single_video = 1.0141`
   - `keep2gt joint = 1.0141`
   - this means the failure boundary already appears by `2 GTs`

5. cls-to-trunk interference is a confirmed causal factor.
   - `single_video joint = 1.0141`
   - `single_video joint + detach-cls = 0.0480`
   - `keep2gt joint + detach-cls = 0.0480`
   - reducing cls weight alone is weaker (`0.4783`)

6. reg-first scheduling is also causal, but its strongest evidence is still in toy form.
   - `regwarm100 = 0.0055`
   - `clsw0.1 -> resume regwarm = 0.0043`
   - earlier full-train `topk1 + binary-cls + regwarm10` did not beat the improved no-warmup control, so warmup likely matters only after the supervision contract is strong enough

7. the 0424 transfer pair does not support a strong full-train rescue claim on the current bridge carrier.
   - frozen baseline: `43.44 / 46.17 / 15.41`
   - `regwarm10`: `43.05 / 45.50 / 15.24`
   - `detach-cls`: `38.56 / 39.82 / 13.20`

8. `regwarm10` becomes a near-neutral transfer once the carrier is already upgraded to `topk9`.
   - delta vs baseline: `-0.39 Avg-mAP`, `-0.67 @0.5`, `-0.17 @0.7`
   - this is too small to justify keeping the bridge line alive as an optimization-only story

9. `detach-cls` is harmful at full-train scale on this carrier.
   - delta vs baseline: `-4.88 Avg-mAP`, `-6.35 @0.5`, `-2.21 @0.7`
   - the toy rescue does not transfer cleanly once real multi-video supervision and detector coupling are restored

## Current Judgment

The current native-`y` soft-bridge line is now closed as a negative bridge-polishing branch.

The evidence now supports this final ranking:

1. strengthen supervision support first (`topk9`)
2. treat joint-stability imports as carrier-dependent: toy success does not imply full-train transfer
3. stop spending more cycles on bridge-side polishing for this carrier

## Final Verdict

- supported:
  - support size is the only full-train lever that clearly transferred on this bridge line
  - `topk9 + binary-cls = 43.44` remains the local ceiling for the current native-`y` soft bridge
- not supported:
  - toy-proven `regwarm` or `detach-cls` materially improving the strongest full-train bridge carrier
- operational consequence:
  - keep the page as the closure record for this branch
  - if native `y` is revisited later, the next method must change the decoder / support contract rather than add more warmup or detach variants

## Connections

AUTO-GENERATED from graph/edges.jsonl - do not edit manually
