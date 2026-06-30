# Research Log - C3 CADF Stagefix Worktree

## 2026-06-30 05:53 +0800

- Prepared local-only CADF/C3 ST+actionness fp32/no-AMP combo gate in `OpenTAD_C3CADFStageFix_Worktree_20260629`.
- Added `configs/adatad/thumos/c3_cadf_densitymesh_original_adatad_32px_alpha0_st_actionness_combo_gate.py` and `logs/run_c3_cadf_alpha0_st_actionness_combo_gate_n16r4.sh`.
- Updated `tools/validate_c3_indirect_clean_config.py` and `tests/test_c3_cadf_densitymesh_config.py` so the combo gate is locked to diagnostic-only alpha0, ST soft path plus actionness aux loss, and no AMP/fp16/EMA.
- Local verification passed for py_compile, only-ST/only-actionness/combo validators, direct combo semantic assert, negative AMP/fp16/EMA validator check, and launcher LF/content check. Local pytest remains blocked by Windows torch DLL initialization before collection.
- No SSH, Slurm, remote sync, remote training, mAP evaluation, evaluator/postprocess edit, PQR/BH-SDC/DIVERGENT route edit, or paper/formal claim was performed.

## 2026-06-30 06:25 +0800

- Prepared local-only CADF/C3 formal selector candidate config `configs/adatad/thumos/c3_cadf_densitymesh_original_adatad_32px_formal_selector_candidate_locked.py`.
- Added fail-closed launcher `logs/run_c3_cadf_formal_selector_candidate_locked_n16r4.sh`; it exits before `tools/train.py` unless combo old-window pass is explicitly confirmed by env plus evidence file.
- Updated `tools/validate_c3_indirect_clean_config.py` and `tests/test_c3_cadf_densitymesh_config.py` so the formal candidate remains locked, fp32/no AMP/no EMA, positive-density CADF rather than alpha0 backend control, and ST+actionness aligned with the combo gate.
- Local verification passed for py_compile, formal/combo validators, 13-config CADF validator matrix, direct formal semantic harness, negative validator harness for unlock/AMP/alpha0/PQR, and launcher `bash -n`. Focused pytest remains blocked before collection by Windows torch `c10.dll` initialization.
- Required final read-only review is incomplete: Claude review job `35e433c4bf4742c8b73d57af8df38b5a` failed with non-JSON output; `llm_chat` lacked `LLM_API_KEY`; MiniMax lacked `MINIMAX_API_KEY`. No valid final review PASS was recorded.
- No SSH, Slurm, remote sync, remote training, mAP evaluation, evaluator/postprocess edit, ranking edit, checkpoint cleanup, or formal claim was performed. Formal candidate remains locked pending user-reported combo child `.376` old-window pass evidence.
