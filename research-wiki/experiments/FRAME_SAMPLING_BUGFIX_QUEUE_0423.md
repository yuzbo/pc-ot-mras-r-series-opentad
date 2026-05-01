---
type: experiment
node_id: exp:FRAME_SAMPLING_BUGFIX_QUEUE_0423
title: "OpenTAD frame-sampling bugfix queue: first validation readout"
date: 2026-04-23
status: completed
created_at: 2026-04-23T12:08:00+08:00
updated_at: 2026-04-23T20:58:30+08:00
---

# FRAME_SAMPLING_BUGFIX_QUEUE_0423

## Scope

This page tracks the current `OpenTAD` frame-sampling bugfix queue launched on `24013` and `25876`.

The queue is testing whether the latest code fixes materially change the previously established negative reading for learned frame selection, especially:

- softmax-dilution fix in `soft_gating`
- `frame_scorer` optimizer-group / learning-rate fix
- queue robustness fixes so runs actually complete

## Current live snapshot

Timestamp:

- `2026-04-23 20:58 +08:00`

Server state:

- server `24013`
  - `v3_gumbel_softgate_minimal_bugfix0420` finished at `2026-04-23 14:13:08`
  - `v3_gumbel_softgate_coarse_det_bugfix0420` finished at `2026-04-23 18:59:48`
- server `25876`
  - frame-sampling queue finished at `2026-04-23 16:00:39`
  - separate `OpenTAD_Back` line `headv3_x_temporalgridfix` also finished later at `2026-04-23 20:40:11`

## First readout: `v3_gumbel_softgate_minimal_bugfix0420`

Config:

- `OpenTAD/configs/frame_sampling/thumos/v3_gumbel_softgate_minimal_bugfix0420.py`

Validation schedule:

- `val_start_epoch=40`
- `val_eval_interval=2`
- `end_epoch=60`

Current recorded validation reads:

| Run | Avg-mAP | Notes |
|---|---:|---|
| first eval | `52.43` | first post-bugfix validation |
| second eval | `53.17` | confirmed at `2026-04-23 12:01:14` |
| third eval | `53.28` | early local best |
| fourth eval | `53.02` | short dip after the early climb |
| fifth eval | `53.75` | first clear move above the old low-`53s` band |
| sixth eval | `54.32` | current best read |
| seventh eval | `54.07` | still in the high-`53` / low-`54` band |
| eighth eval | `53.87` | mild pullback |
| ninth eval | `53.73` | late-stage pullback |
| tenth eval | `54.22` | final confirmed read at `2026-04-23 14:13:02`; run then finished and released the queue |

Reference comparisons:

- old `R021 = 53.98`
- delta vs old `R021`:
  - peak: `+0.34`
  - final: `+0.24`
- delta vs `R008 uniform 50% = 63.75`:
  - peak: `-9.43`
  - final: `-9.53`

Current interpretation:

- the bugfix rerun is not collapsing
- it did recover beyond the earlier `53.0 ~ 53.3` snapshot and briefly reached `54.32`
- but it is still clearly below the fixed-input `uniform 50%` reference
- the corrected final reading is now "completed high-`53` / low-`54` line", not "stuck forever at `53.0 ~ 53.3`"
- after fixing the known implementation issues, the learned `gumbel + soft_gating` minimal line still behaves like the old negative result family rather than like a rescued path toward the uniform baseline

So the current supported statement is:

- the new bug fixes made the run cleaner and more trustworthy
- they exposed some extra headroom inside the same weak regime
- they still did not convert the soft-gating minimal line into a competitive learned-sampler result

## Completed successor: `v3_gumbel_softgate_coarse_det_bugfix0420`

Config:

- `OpenTAD/configs/frame_sampling/thumos/v3_gumbel_softgate_coarse_det_bugfix0420.py`

Current runtime status:

- started automatically at `2026-04-23 14:13:08` right after the minimal run finished
- training completed through `epoch 59`

Current recorded validation reads:

| Run | Avg-mAP | Notes |
|---|---:|---|
| first eval | `48.07` | first read is far below the minimal bugfix line |
| second eval | `49.90` | partial recovery, but still clearly weak |
| third eval | `50.75` | further recovery, but still materially below the other bugfix branches |
| fourth eval | `51.13` | keeps climbing, but remains clearly sub-baseline |
| fifth eval | `52.24` | first move above `52` |
| sixth eval | `52.28` | essentially flat |
| seventh eval | `52.70` | mild gain |
| eighth eval | `52.57` | slight pullback |
| ninth eval | `52.90` | late-stage recovery |
| tenth eval | `53.17` | final confirmed read at `2026-04-23 18:59:42` |

Latest visible train band:

- `Loss ~= 1.23 ~ 1.28`
- `scorer_loss ~= 0.066 ~ 0.068`
- `coarse_det_loss ~= 0.56 ~ 0.60`
- `_mon_keep_ratio ~= 0.46 ~ 0.48`

Current interpretation:

- the queue handoff on `24013` is healthy
- this branch opened substantially below both the completed minimal rerun and the stronger hard-scorer line
- it did recover materially from the weak opening, but the final `53.17` still stays below the minimal rerun `54.22`
- relative to the completed bugfix family, the coarse-det addition remains negative:
  - delta vs minimal bugfix rerun: `-1.05`
  - delta vs hard-scorer bugfix line: `-3.80`
- so the current supported reading remains that adding the coarse-det branch is harmful in this bugfix family

## Running companion: `v3_hard_scorer_aux_bugfix0420`

Config:

- `OpenTAD/configs/frame_sampling/thumos/v3_hard_scorer_aux_bugfix0420.py`

Current runtime status:

- finished on `25876` at `2026-04-23 16:00:39`
- current run is the restarted clean launch after one early `exit=1`
- training completed through `epoch 59` and final validation

Current recorded validation reads:

| Run | Avg-mAP | Notes |
|---|---:|---|
| first eval | `54.42` | first trustworthy read |
| second eval | `55.11` | clear gain over the minimal soft-gating line |
| third eval | `55.76` | keeps climbing |
| fourth eval | `56.00` | first move into the `56+` range |
| fifth eval | `56.53` | continued climb |
| sixth eval | `56.94` | current local best |
| seventh eval | `56.87` | small pullback |
| eighth eval | `57.00` | peak read |
| ninth eval | `56.97` | final confirmed read at `2026-04-23 16:00:36` |

Latest visible train band:

- `Loss ~= 1.74 ~ 1.80`
- occasional `non-finite gradients detected` events still occur, but the run continues

Current interpretation:

- this is the strongest completed bugfix line in the current queue
- it is already materially above the bugfixed minimal soft-gating rerun
- it crossed `57.00` at peak and closed at `56.97`
- this is a real improvement over the minimal rerun, but it still remains well below the `uniform 50%` reference and therefore does not close the learned-sampler gap

## Related parallel line: `headv3_x_temporalgridfix`

This is not part of the frame-sampling queue itself, but it is currently waiting behind the server `25876` frame-sampling job and therefore matters operationally.

Current status:

- screen: `temporalgridfix_full_s3`
- started at `2026-04-23 16:01:13` after the `25876` frame-sampling queue released the GPU
- finished at `2026-04-23 20:40:11`
- final validation read: `52.40`
- total observed skipped steps from non-finite reg-head gradients: `5`
  - `epoch 0 iter 0`
  - `epoch 0 iter 17`
  - `epoch 9 iter 17`
  - `epoch 29 iter 62`
  - `epoch 49 iter 71`
- despite these skipped steps, the run completed normally

## What changed in the global picture

Before this read, one open question remained:

- after fixing the obvious implementation bugs, would the soft-gating line rise materially above the old `R021` region, and would any stronger learned-scorer branch emerge?

The current answer is now:

- `v3_gumbel_softgate_minimal_bugfix0420` did rise above the old four-read snapshot and briefly exceeded old `R021`
- but it still remains far from the `uniform` reference and therefore does not rescue the soft-gating thesis
- `v3_hard_scorer_aux_bugfix0420` is the strongest completed branch so far, closing at `56.97` with peak `57.00`
- `v3_gumbel_softgate_coarse_det_bugfix0420` recovered to `53.17`, but still finished below the minimal rerun and far below the hard-scorer line
- `headv3_x_temporalgridfix` closed at `52.40`, which fails to beat the old `headv3_x = 52.60`

`52.43 -> 53.17 -> 53.28 -> 53.02 -> 53.75 -> 54.32 -> 54.07 -> 53.87 -> 53.73 -> 54.22` is no longer a pure low-`53s` plateau, but it still sits in the same broad low-to-mid `50s` regime and still far below the `uniform` reference.

That means the current bugfix batch should now be interpreted more narrowly:

- it improves experiment trustworthiness
- it gives a small upward correction to the soft-gating minimal rerun
- it still does not overturn the earlier negative conclusion on learned soft-gating
- it identifies `hard_scorer_aux` as the best branch in this queue so far
- it also shows that `coarse_det` is a negative direction on this bugfixed frame-sampling carrier
- and the formal `temporalgridfix` rerun does not support "repaired temporal-grid semantics alone lift the `x` line above its old level"

## Closed takeaways

1. `v3_hard_scorer_aux_bugfix0420` is the strongest completed learned-scorer bugfix result in this batch at `56.97`, but it still leaves a large gap to `63.75`.
2. `v3_gumbel_softgate_coarse_det_bugfix0420` finishes at `53.17`, so adding the coarse-det branch does not improve over the simpler minimal rerun.
3. `headv3_x_temporalgridfix = 52.40` is slightly below the old `headv3_x = 52.60`, so `BUG-1/2` repairs were real code corrections but not a sufficient explanation for the weak `x`-line ceiling.

## Connections

AUTO-GENERATED from graph/edges.jsonl - do not edit manually
