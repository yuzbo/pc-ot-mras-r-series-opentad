# Pre-Backbone Task-Aware Frame Acquisition Idea Discussion

Date: 2026-06-24 Asia/Shanghai

Scope: local research idea discussion for `pre-backbone task-aware frame acquisition for TAD`, centered on the current C3-Pro frame-score-first candidate and broader dynamic-budget / irregular-ActionFormer routes.

Local repository inspected:

- `E:/DeskTop/TAD/temrefuse-tad/OpenTAD_PCOTMRAS_R16_R18_R20_Clean_20260619_1730`
- Branch observed: `codex/prebackbone-pcotmras-scout-selector`
- HEAD observed at package creation: `12f84d56d15f15c284f08431bb64d5de49df8185`
- Latest frame-score-first commit recorded by main process: `471897edf7ea9871a4e7f8eeacbedeba178ef989`
- Follow-up HEAD observed: `471897edf7ea9871a4e7f8eeacbedeba178ef989`
- Worktree note: C3-Pro-related tracked and untracked files were already present before this discussion package; this package adds only files under `idea-stage/prebackbone_task_aware_frame_acquisition_20260624/`.

Files in this discussion package:

1. `LITERATURE_LANDSCAPE.md`
   - Recent related-work map and local implementation context.
2. `IDEA_CANDIDATES_AND_DISCUSSION.md`
   - Twelve divergent candidate ideas, risk discussion, ranking, and minimal experiment tree.
3. `PRO_DISCUSSION_PROMPT.md`
   - A complete Chinese prompt for a Pro-level critic to review the current implementation and propose divergent routes.
4. `NOVELTY_AND_REVIEW_NOTES.md`
   - Core claims, closest prior work, novelty risks, and red-team failure interpretations.
5. `EXPERIMENT_ROADMAP.md`
   - Claim-driven minimal experiment tree and locked/unlocked next actions.
6. `IDEA_CREATOR_READONLY_FOLLOWUP_20260624_1235.md`
   - Parallel read-only idea-creator follow-up that ranks interval-score-first boundary packets, global differentiable top-k rank transport, and dynamic marginal-utility budget control as the next top routes.
7. `IDEA_CREATOR_FOLLOWUP_20260624_123141.md`
   - Idea-creator follow-up for latest C3-Pro frame-score-first implementation at `471897e`: landscape refresh, 10 divergent reader/selector/dynamic-budget/irregular-ActionFormer routes, feasibility gate, ranked priorities, and pilot/full experiment order.

Latest follow-up summary:

- Current C3-Pro implementation is correctly framed as frame-score-first hard top-k, not slot-led hard selection, with tests for ST gradient, padding masks, and no teacher/raw-cache/test-GT shortcut.
- Main remaining risks are under-trained reader heads, narrow ST surrogate gradients, uniform-cover mimicry, fixed-budget limitation, and downstream ActionFormer regular-grid assumptions.
- Top recommended next routes are calibrated PixelRanker/listwise ranking, dynamic budget with explicit fallback, physical-time ActionFormer micro-adaptation, boundary packet allocation, and ST-gradient bandwidth audit.
- The follow-up is documentation-only: no code, training, evaluation, upload, or deletion was performed.

Boundary:

- No remote commands were run.
- No training or evaluation was launched.
- No implementation code was edited.
- External advisory was attempted through available local chat tools. GlobalAI/Gemini returned partial advisory signal; MiniMax and local LLM channels were unavailable due missing API keys, so they are not treated as completed reviews.
