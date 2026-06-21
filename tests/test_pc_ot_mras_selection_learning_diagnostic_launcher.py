from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_ctf_bdi_pc_ot_mras_selection_learning_diagnostic_n16r4.sbatch"


def test_selection_learning_diagnostic_launcher_is_fail_closed_and_sha_bound():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "#SBATCH --gpus=1" in text
    assert "#SBATCH -J pcot_selviz" in text
    assert "OpenTAD_PCOTMRAS_R16_R18_R20_Clean_20260619_1730" in text
    assert "refusing dirty historical OpenTAD_BATA_Clean path" in text
    assert 'PRECHECK_ONLY="${PRECHECK_ONLY:-1}"' in text
    assert 'ALLOW_PCOTMRAS_SELECTION_DIAGNOSTIC="${ALLOW_PCOTMRAS_SELECTION_DIAGNOSTIC:-0}"' in text
    assert "PRECHECK_ONLY=0 requires ALLOW_PCOTMRAS_SELECTION_DIAGNOSTIC=1" in text
    assert "PRECHECK_ONLY=1 must not set ALLOW_PCOTMRAS_SELECTION_DIAGNOSTIC=1" in text
    assert "PCOTMRAS_SELECTION_CONFIG_SHA256" in text
    assert "PCOTMRAS_SELECTION_CHECKPOINT_SHA256" in text
    assert "PCOTMRAS_SELECTION_GATE_SHA256" in text
    assert "$label SHA256 mismatch" in text
    assert 'require_file_sha "$PCOTMRAS_SELECTION_CONFIG" "$PCOTMRAS_SELECTION_CONFIG_SHA256" "config"' in text
    assert 'require_file_sha "$PCOTMRAS_SELECTION_CHECKPOINT" "$PCOTMRAS_SELECTION_CHECKPOINT_SHA256" "checkpoint"' in text
    assert 'require_file_sha "$PCOTMRAS_SELECTION_GATE_JSON" "$PCOTMRAS_SELECTION_GATE_SHA256" "selection diagnostic gate"' in text
    assert "ALLOW_PCOTMRAS_SELECTION_LEARNING_DIAGNOSTIC" in text
    assert "model_stage mismatch" in text
    assert "checkpoint_tier mismatch" in text


def test_selection_learning_diagnostic_launcher_only_runs_snapshot_and_visualization():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "tools/bata/dump_pc_ot_mras_reader_snapshots.py" in text
    assert "tools/bata/visualize_pc_ot_mras_selection.py" in text
    assert "--output-jsonl" in text
    assert "--summary-json" in text
    assert "--snapshot-label" in text
    assert "--limit-batches" in text
    assert 'tools/train.py "$CONFIG"' not in text
    assert 'tools/test.py "$CONFIG"' not in text
    assert "tools/train.py and tools/test.py are not part of this diagnostic launcher" in text
    assert "raw prediction/cache/result-detection environment shortcuts are forbidden" in text
    assert "generic checkpoint/load/resume environment shortcuts are forbidden" in text
    assert "PC_OT_MRAS_SELECTION_LEARNING_DIAGNOSTIC_PASS_NO_MAP_NO_CLAIMS" in text
    assert "detector_map" in text
    assert "metric_claim_allowed" in text
    assert "paper_claim_allowed" in text
    assert "runtime_or_flops_claim_allowed" in text
    assert "deploy_claim_allowed" in text


def test_selection_learning_diagnostic_launcher_is_limited_to_r17_r18_val_and_checkpoint_tiers():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "MODEL_STAGE must be R17 or R18" in text
    assert "CHECKPOINT_TIER must be early, middle, or final" in text
    assert "selection-learning diagnostic is reviewed only for SPLIT=val" in text
    assert "LIMIT_BATCHES must be in [1,16]" in text
    assert "VIS_LIMIT must be in [1,256]" in text
    assert "ctf_bdi_pc_ot_mras_r17_formal_train_candidate.py" in text
    assert "ctf_bdi_pc_ot_mras_r18_aux_formal_train_candidate.py" in text
    assert "config does not match reviewed model stage" in text
