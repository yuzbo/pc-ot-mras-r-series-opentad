from pathlib import Path

from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[1]


def test_r20_value_only_control_inherits_r17_without_r18_aux_loss():
    cfg = Config.fromfile(
        ROOT / "configs" / "adatad" / "thumos" / "ctf_bdi_pc_ot_mras_r20_value_only_control.py"
    )

    assert cfg.r17_pc_ot_mras_formal_train_gate is None
    assert "r18_pc_ot_mras_aux_diag_gate" not in cfg
    assert cfg.r20_pc_ot_mras_value_only_control_gate.launch_gate_passed is False
    assert cfg.r20_pc_ot_mras_value_only_control_gate.allow_remote_sync is False
    assert cfg.r20_pc_ot_mras_value_only_control_gate.allow_slurm is False
    assert cfg.r20_pc_ot_mras_value_only_control_gate.allow_gpu is False
    assert cfg.r20_pc_ot_mras_value_only_control_gate.allow_tools_train is False
    assert cfg.r20_pc_ot_mras_value_only_control_gate.allow_tools_test is False

    assert cfg.model.pc_ot_mras_reader.type == "PCOTMRASReader"
    assert cfg.model.pc_ot_mras_reader.enable_value_heads is True
    assert "pc_ot_mras_reader_aux_loss" not in cfg.model
    assert cfg.model.pc_ot_mras_reader_value_loss.enabled is True
    assert cfg.model.pc_ot_mras_reader_value_loss.require_targets is True
    assert cfg.model.pc_ot_mras_reader_value_loss.allow_teacher_targets is False

    assert cfg.inference.load_from_raw_predictions is False
    assert cfg.inference.save_raw_prediction is False
