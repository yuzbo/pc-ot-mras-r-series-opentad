from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "adatad"
    / "thumos"
    / "pc_ot_mras_prebackbone_c3_pro_boundary_gradient_diag_smoke_n16r4.py"
)
SCRIPT = ROOT / "scripts" / "run_pc_ot_mras_prebackbone_c3_pro_boundary_gradient_diag_smoke_n16r4.sbatch"


def test_c3_pro_gradient_diag_config_is_one_step_smoke_only():
    mmengine_config = pytest.importorskip("mmengine.config")
    training_guard = pytest.importorskip("opentad.utils.training_guard")

    cfg = mmengine_config.Config.fromfile(str(CONFIG))
    gate = cfg.pc_ot_mras_prebackbone_e2e_acquisition_gate

    assert cfg.variant_id == "C3-Pro-BoundaryDifficulty-GradientDiag"
    assert cfg.experiment_scope.complete_training_required is False
    assert cfg.experiment_scope.metric_claim_allowed is False
    assert cfg.experiment_scope.deploy_claim_allowed is False
    assert gate.smoke_only is True
    assert gate.formal_train_candidate is False
    assert gate.allow_tools_train is True
    assert gate.allow_tools_test is False
    assert gate.allow_detector_map is False
    assert gate.allow_train_validation_map is False
    assert gate.allow_long_training is False
    assert gate.allow_checkpoint_write is False
    assert gate.allow_checkpoint_load is False
    assert gate.allow_resume is False
    assert gate.max_epochs == 1
    assert gate.max_train_iters == 1
    assert gate.disable_checkpoint is True
    assert tuple(gate.allowed_entrypoints) == ("tools/train.py",)

    assert cfg.workflow.end_epoch == 1
    assert cfg.workflow.max_train_iters == 1
    assert cfg.workflow.disable_checkpoint is True
    assert cfg.workflow.val_start_epoch == 999
    assert cfg.workflow.val_eval_interval == -1
    assert cfg.workflow.val_loss_interval == -1
    assert cfg.model.frame_selector.reader.type == "PCOTMRASBoundaryDifficultyTemporalFrameScout"
    assert cfg.model.frame_selector.target_len == 384
    assert cfg.model.frame_selector.dense_window_size == 768
    assert cfg.model.rpn_head.type == "ActionFormerHead"

    assert training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py") is None
    with pytest.raises(RuntimeError, match="allow_tools_test=False"):
        training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/test.py")


def test_c3_pro_gradient_diag_launcher_is_diagnostic_only():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "#SBATCH -J c3diag1" in text
    assert "pc_ot_mras_prebackbone_c3_pro_boundary_gradient_diag_smoke_n16r4.py" in text
    assert "max_train_iters=1" in text
    assert "no_validation no_checkpoint no_claims" in text
    assert "tests/test_pc_ot_mras_prebackbone_c3_pro_gradient_diag_smoke.py" in text
    assert "tools/test.py \"$CONFIG\"" not in text
    assert "diagnostic config must reject tools/test.py" in text
    assert "result_detection.json" in text
    assert "Average-mAP" in text
    assert "C3_PRO_BOUNDARY_GRADIENT_DIAG_PASS_FIRST_STEP_FINITE" in text
