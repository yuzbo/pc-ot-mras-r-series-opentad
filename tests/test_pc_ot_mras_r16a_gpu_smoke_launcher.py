from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SBATCH = ROOT / "scripts" / "run_ctf_bdi_pc_ot_mras_r16a_gpu_smoke_n16r4.sbatch"


def _launcher_text():
    return SBATCH.read_text(encoding="utf-8")


def test_r16a_gpu_smoke_launcher_is_fail_closed_by_default():
    text = _launcher_text()

    assert "#SBATCH --gpus=1" in text
    assert "#SBATCH -J pcot_r16smk" in text
    assert "OPENTAD_PCOTMRAS_CLEAN_ROOT" in text
    assert "OpenTAD_PCOTMRAS_R16_R18_R20_Clean_20260619_1730" in text
    assert 'REPO="$YUZIBO_ROOT/OpenTAD_BATA_Clean"' not in text
    assert "OPENTAD_BATA_ROOT" not in text
    assert "refusing dirty historical OpenTAD_BATA_Clean path" in text
    assert "ctf_bdi_pc_ot_mras_r16_gpu_smoke_candidate.py" in text
    assert 'PRECHECK_ONLY="${PRECHECK_ONLY:-1}"' in text
    assert 'ALLOW_R16A_GPU_SMOKE="${ALLOW_R16A_GPU_SMOKE:-0}"' in text
    assert 'R16A_GPU_SMOKE_ONLY="${R16A_GPU_SMOKE_ONLY:-1}"' in text
    assert 'ALLOW_R16A_LONG_TRAINING="${ALLOW_R16A_LONG_TRAINING:-0}"' in text
    assert 'ALLOW_R16A_TOOLS_TEST="${ALLOW_R16A_TOOLS_TEST:-0}"' in text
    assert 'ALLOW_R16A_DETECTOR_MAP="${ALLOW_R16A_DETECTOR_MAP:-0}"' in text
    assert "PRECHECK_ONLY=0 requires ALLOW_R16A_GPU_SMOKE=1" in text
    assert "PRECHECK_ONLY=1 must not set ALLOW_R16A_GPU_SMOKE=1" in text
    assert "R16A_GPU_SMOKE_GATE_JSON" in text
    assert "R16A_GPU_SMOKE_GATE_SHA256" in text
    assert "R16A execution gate decision does not allow GPU smoke" in text
    assert "R16A execution gate must bind active_sha256_manifest_sha256" in text
    assert "R16A execution gate active manifest sha256 mismatch" in text
    assert "ctf_bdi_pc_ot_mras_r16a_execution_gate=PASS_BOUND_TO_ACTIVE_MANIFEST" in text
    assert "login-node debug must keep PRECHECK_ONLY=1" in text


def test_r16a_gpu_smoke_launcher_forbids_unapproved_surfaces():
    text = _launcher_text()

    assert "long training is not approved" in text
    assert "tools/test.py is not approved for R16A" in text
    assert "detector mAP is not approved for R16A" in text
    assert "external checkpoint/load/resume environment shortcuts are forbidden" in text
    assert "PRETRAINED_PATH override is forbidden for R16A smoke" in text
    assert "raw prediction/cache environment shortcuts are forbidden" in text
    assert "arbitrary cfg-options are forbidden" in text
    assert "scope=GPU_SMOKE_ONLY no_tools_test=true no_map=true no_long_training=true no_claims=true" in text
    assert "PRECHECK_ONLY=1; exiting before R16A tools/train GPU smoke" in text
    assert "R16A_GPU_SMOKE_PRECHECK_ONLY_PASS_NO_TRAIN" in text


def test_r16a_gpu_smoke_launcher_audits_config_and_train_bounds():
    text = _launcher_text()

    assert "ctf_bdi_pc_ot_mras_r16a_gpu_smoke_config_preflight=PASS" in text
    assert "gate.smoke_only is True" in text
    assert "gate.allow_tools_test is False" in text
    assert "gate.allow_detector_map is False" in text
    assert "int(gate.max_epochs) == 1" in text
    assert "int(gate.max_train_iters) == 2" in text
    assert "int(cfg.workflow.end_epoch) == 1" in text
    assert "int(cfg.workflow.max_train_iters) == 2" in text
    assert "int(cfg.workflow.val_start_epoch) == 999" in text
    assert "cfg.inference.load_from_raw_predictions is False" in text
    assert "cfg.inference.save_raw_prediction is False" in text
    assert "tests/test_pc_ot_mras_r16_gpu_smoke_guard.py" in text
    assert "tests/test_train_engine_max_train_iters.py" in text
    assert "opentad/models/dense_heads/__init__.py" in text
    assert "opentad/models/dense_heads/native_irregular_area_head_p2.py" in text
    assert "grep -q \"max_train_iters=2 reached\"" in text
    assert "grep -q \"Training Over\"" in text
    assert "active_sha256_manifest_sha256=$ACTIVE_MANIFEST_SHA256" in text


def test_r16a_gpu_smoke_launcher_uses_train_only_and_no_test_or_map_command():
    text = _launcher_text()

    assert "torchrun --nproc_per_node=1" in text
    assert 'tools/train.py "$CONFIG"' in text
    assert 'tools/test.py "$CONFIG"' not in text
    assert "result_detection.json" in text
    assert "R16A_GPU_SMOKE_PASS_NO_MAP_NO_LONG_TRAINING" in text
    assert "no tools/test.py, detector mAP, long training" in text
    assert "evaluation.ground_truth_filename=\"$THUMOS14_ANNOTATION_PATH\"" in text
    assert "dataset.train.data_path=\"$THUMOS14_TRAIN_DATA_PATH\"" in text
    assert "dataset.test.data_path=\"$THUMOS14_TEST_DATA_PATH\"" in text


def test_r16a_gpu_smoke_launcher_does_not_py_compile_sbatch():
    text = _launcher_text()
    compile_block = text.split("log \"static py_compile\"", 1)[1].split("log \"focused R16A", 1)[0]

    assert "run_ctf_bdi_pc_ot_mras_r16a_gpu_smoke_n16r4.sbatch" not in compile_block
    assert 'sha256sum \\\n  "$CONFIG" \\' in text
