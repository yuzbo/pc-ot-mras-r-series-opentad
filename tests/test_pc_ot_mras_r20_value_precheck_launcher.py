from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_ctf_bdi_pc_ot_mras_r20_value_precheck_n16r4.sbatch"


def test_r20_value_precheck_launcher_is_precheck_only_and_fail_closed():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "#SBATCH --gpus=1" in text
    assert "#SBATCH -J pcot_r20chk" in text
    assert "OpenTAD_PCOTMRAS_R16_R18_R20_Clean_20260619_1730" in text
    assert "refusing dirty historical OpenTAD_BATA_Clean path" in text
    assert "expected branch $EXPECTED_GIT_BRANCH" in text
    assert "tracked clean repo files are modified" in text
    assert "ctf_bdi_pc_ot_mras_r20_value_distill_candidate.py" in text
    assert "ctf_bdi_pc_ot_mras_r20_value_only_control.py" in text
    assert 'PRECHECK_ONLY="${PRECHECK_ONLY:-1}"' in text
    assert "R20 value launcher is precheck-only; PRECHECK_ONLY must remain 1" in text
    assert "checkpoint/load/resume shortcuts are forbidden for R20 precheck" in text
    assert "raw prediction/cache environment shortcuts are forbidden for R20 precheck" in text
    assert "arbitrary cfg-options are forbidden for R20 precheck" in text
    assert "allow_tools_train is False" in text
    assert "allow_slurm is False" in text
    assert "assert_detector_training_allowed" in text
    assert "resolved_config_dependency_count" in text
    assert "RESOLVED_CONFIG_SHA256" in text
    assert "tools/train.py" in text
    assert "R20_VALUE_PRECHECK_ONLY_PASS_NO_TRAIN_NO_DATA_NO_MAP" in text
    assert 'tools/train.py "$CONFIG"' not in text
    assert 'tools/test.py "$CONFIG"' not in text


def test_r20_value_precheck_launcher_runs_only_static_and_focused_tests():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "python -m py_compile" in text
    assert "tests/test_pc_ot_mras_r20_value_only_control_config.py" in text
    assert "tests/test_pc_ot_mras_value_distillation_losses.py" in text
    assert "tests/test_pc_ot_mras_actionformer_forward_selector.py" in text
    assert "no_train=true no_test=true no_map=true no_data=true no_claims=true" in text
