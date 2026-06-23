from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_d6_d7_d8_launcher_is_diagnostic_only_and_gated():
    launcher = ROOT / "scripts" / "run_pc_ot_mras_r17_r18_d6_d7_d8_diagnostics_n16r4.sbatch"
    text = launcher.read_text(encoding="utf-8")

    assert "ALLOW_PCOTMRAS_D6_D7_D8_DIAG" in text
    assert "PRECHECK_ONLY" in text
    assert "tools/bata/analyze_p2_raw_row_oracle_rerank.py" in text
    assert "tools/bata/dump_pc_ot_mras_reader_bridge_diagnostics.py" in text
    assert "--feature-jsonl" in text
    assert "tools/bata/replay_pc_ot_mras_training_assignment_geometry.py" in text
    assert "ctf_bdi_pc_ot_mras_r17_formal_train_candidate.py" in text
    assert "ctf_bdi_pc_ot_mras_r18_aux_formal_train_candidate.py" in text
    assert "R17_CHECKPOINT_SHA256" in text
    assert "R18_CHECKPOINT_SHA256" in text
    assert 'if [ "$PRECHECK_ONLY" = "1" ]' in text
    assert text.index('if [ "$PRECHECK_ONLY" = "1" ]') < text.index("verify_file R17_CHECKPOINT")
    assert "[[ ! \"$RUN_TAG\" =~ ^[A-Za-z0-9._-]+$ ]]" in text
    assert "training_or_eval_entrypoints_used" in text
    assert "tools/train.py" not in text
    assert "tools/test.py" not in text


def test_d6_d7_d8_launcher_defaults_to_reuse_existing_raw_rows():
    launcher = ROOT / "scripts" / "run_pc_ot_mras_r17_r18_d6_d7_d8_diagnostics_n16r4.sbatch"
    text = launcher.read_text(encoding="utf-8")

    assert 'D6_REGENERATE_RAW_ROWS="${D6_REGENERATE_RAW_ROWS:-0}"' in text
    assert "pcotmras_p2_localization_attribution_ef29cb6_20260622_225701" in text
    assert "joined_proposals.jsonl" in text
    assert "D6 joined rows missing" in text
