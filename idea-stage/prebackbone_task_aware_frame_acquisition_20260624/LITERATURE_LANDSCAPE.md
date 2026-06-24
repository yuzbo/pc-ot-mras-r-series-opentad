# Literature And Local Landscape

## Problem Anchor

The project objective is not "select exactly 384 of 768 frames" as a final method. The real objective is a deployable, task-aware temporal acquisition system for temporal action detection (TAD): it should allocate frames/snippets/tokens according to video/window/action difficulty, reduce sustained temporal redundancy, preserve high-IoU localization, and eventually report a compute-performance frontier.

Current C3-Pro candidate:

- Pre-backbone selector receives low-resolution pixel descriptors.
- A temporal reader emits `action`, `start`, `end`, `boundary`, `uncertainty`, `redundancy`, and `frame_selection_logits`.
- Hard `top-k` selects 384 real frames from a dense 768-frame window.
- Straight-through surrogate sends detector loss gradients to `frame_selection_logits`.
- Backend remains original AdaTAD/ActionFormer.
- Explicitly avoids P2, teacher, raw prediction cache, and test-time GT.

## Local Code Context

Key implementation surfaces currently relevant:

- `opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py`
  - Contains `PCOTMRASBoundaryDifficultyTemporalFrameScout`.
  - Contains `selection_strategy="frame_score_topk"`.
  - Contains straight-through transport and dense-to-selected GT remapping.
- `configs/adatad/thumos/pc_ot_mras_prebackbone_c3_pro_boundary_reader_full_train_candidate_n16r4.py`
  - Adds C3-Pro boundary/difficulty reader config.
  - Sets fixed 384/768 frame-score-first budget.
- `scripts/run_pc_ot_mras_prebackbone_c3_reader_full_train_n16r4.sbatch`
  - Adds `READER_VARIANT=pro_boundary` path and launch gates.
- `tests/test_pc_ot_mras_prebackbone_pro_reader_design.py`
  - Tests Pro reader outputs, frame-score-first hard selection, ST gradient path, metadata protocol flags.
- `tests/test_pc_ot_mras_prebackbone_c3_reader_variants.py`
  - Tests C3 reader variants and config parsing.
- `tests/test_pc_ot_mras_prebackbone_nan_guards.py`
  - Tests finite guards for slot/logit/matrix/ST paths.
- `opentad/models/detectors/actionformer.py`, `opentad/models/dense_heads/actionformer_head.py`, `opentad/models/utils/temporal_grid.py`
  - Required to judge whether original ActionFormer geometry remains valid under sparse, non-uniform selected frames.

Observed working-tree context before writing this package:

- Tracked modified:
  - `opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py`
  - `scripts/run_pc_ot_mras_prebackbone_c3_reader_full_train_n16r4.sbatch`
  - `tests/test_pc_ot_mras_prebackbone_c3_reader_variants.py`
- Untracked C3-Pro file:
  - `configs/adatad/thumos/pc_ot_mras_prebackbone_c3_pro_boundary_reader_full_train_candidate_n16r4.py`
  - `tests/test_pc_ot_mras_prebackbone_pro_reader_design.py`

## Recent Literature Signals

The open literature suggests four nearby but not identical streams:

1. Scaled end-to-end TAD.
   - AdaTAD scales end-to-end TAD with temporal-informative adapters and long input, reporting strong THUMOS14 results while emphasizing redundancy and memory limits.
   - Source: [AdaTAD arXiv](https://arxiv.org/html/2311.17241v2)
   - Relevance: establishes why input redundancy and adapter efficiency matter, but it does not solve deployable task-aware frame acquisition.

2. Adapter evolution for high-resolution TAD.
   - AdaTAD++ decouples temporal and spatial adaptation and uses two-stage training to trade temporal/spatial resolution under resource constraints.
   - Source: [AdaTAD++ ICCV 2025 PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Agrawal_Scaling_Action_Detection_AdaTAD_with_Transformer-Enhanced_Temporal-Spatial_Adaptation_ICCV_2025_paper.pdf)
   - Relevance: a strong detector-side competitor. A selector paper must show it is not merely a weaker route to the same scaling story.

3. TAD model compression.
   - Progressive Block Drop reduces model depth while preserving width and reports reduced overhead on THUMOS14 and ActivityNet.
   - Source: [Progressive Block Drop CVPR 2025 PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Chen_Temporal_Action_Detection_Model_Compression_by_Progressive_Block_Drop_CVPR_2025_paper.pdf)
   - Relevance: efficiency baseline is not only input sampling; compute frontier must compare against model-side compression.

4. General video/VLM adaptive frame or token selection.
   - Flexible Frame Selector learns a policy for efficient video reasoning.
   - Source: [FFS CVPR 2025 PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Buch_Flexible_Frame_Selection_for_Efficient_Video_Reasoning_CVPR_2025_paper.pdf)
   - LGTTP uses temporal cues from queries to prune video tokens.
   - Source: [LGTTP arXiv](https://arxiv.org/html/2508.17686v1)
   - Relevance: strong evidence that adaptive frame/token selection is timely, but these are not TAD high-IoU boundary-localization methods.

5. Video action detection token pruning.
   - EVAD uses keyframe-centric token pruning and context refinement for efficient video action detection.
   - Source: [EVAD ICCV 2023 PDF](https://openaccess.thecvf.com/content/ICCV2023/papers/Chen_Efficient_Video_Action_Detection_with_Token_Dropout_and_Context_Refinement_ICCV_2023_paper.pdf)
   - Relevance: closest in spirit for action detection and token pruning, but the task differs from temporal action detection over long untrimmed windows.

6. Framework and detector baseline context.
   - OpenTAD emphasizes standardized TAD modules and comparisons.
   - Source: [OpenTAD CVPRW 2025](https://openaccess.thecvf.com/content/CVPR2025W/PVUW/html/Liu_OpenTAD_A_Unified_Framework_and_Comprehensive_Study_of_Temporal_Action_CVPRW_2025_paper.html)
   - ActionFormer remains a core anchor-free temporal localization baseline.
   - Source: [ActionFormer arXiv](https://arxiv.org/abs/2202.07925)

## Gap Map

Likely open gap:

- A TAD-specific acquisition policy that uses deployable pre-backbone evidence to allocate temporal evidence non-uniformly while preserving high-IoU boundaries.
- A detector interface that treats selected frames as irregular physical-time observations rather than pretending selected indices are uniformly spaced.
- A dynamic budget controller that reports actual compute-performance frontier instead of fixed 50% recovery.

Main risk:

- Current C3-Pro may only be a learned reordering/top-k heuristic under fixed budget. If ActionFormer still interprets the selected axis as regular time, the selector can improve or collapse metrics for reasons unrelated to genuine acquisition intelligence.
