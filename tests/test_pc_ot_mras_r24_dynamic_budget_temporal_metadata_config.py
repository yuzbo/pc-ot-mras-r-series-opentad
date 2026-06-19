import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "adatad" / "thumos" / "ctf_bdi_pc_ot_mras_r24_dynamic_budget_temporal_metadata.py"
GUARD_PATH = ROOT / "opentad" / "utils" / "training_guard.py"


def _load_guard():
    spec = importlib.util.spec_from_file_location("training_guard_for_r24_test", GUARD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_r24_dynamic_budget_temporal_metadata_config_is_parseable_and_launch_blocked():
    mmengine_config = pytest.importorskip("mmengine.config")
    training_guard = _load_guard()

    cfg = mmengine_config.Config.fromfile(str(CONFIG))

    gate = cfg.r24_pc_ot_mras_dynamic_budget_temporal_metadata_gate
    assert gate.route == "CTF-BDI/PC-OT-MRAS"
    assert gate.stage == "R24_dynamic_budget_temporal_metadata_candidate"
    assert gate.dynamic_budget_protocol_candidate is True
    assert gate.temporal_metadata_protocol_candidate is True
    assert gate.detector_geometry_protocol_candidate is True
    assert gate.dynamic_budget_validation is False
    assert gate.local_synthetic_gate_only is True
    assert gate.allow_detector_training is False
    assert gate.allow_tools_train is False
    assert gate.allow_tools_test is False
    assert gate.allow_detector_map is False
    assert gate.allow_remote_sync is False
    assert gate.allow_slurm is False
    assert gate.allow_gpu is False
    assert gate.dynamic_budget_claim_allowed is False
    assert gate.metric_claim_allowed is False
    assert gate.paper_claim_allowed is False

    metadata = cfg.pc_ot_mras_dynamic_budget_temporal_metadata
    assert metadata.converter == "pc_ot_mras_hard_rows_to_temporal_metas"
    assert metadata.source_schema == "pc_ot_mras_hard_positions_v0"
    assert metadata.output_schema == "pc_ot_mras_temporal_metadata_v0"
    assert metadata.contract_validator == "validate_sampling_contract"
    assert metadata.temporal_grid_builder == "temporal_grid_from_metas"
    assert metadata.irregular_native_axis is True
    assert metadata.training_backprop_allowed is False
    assert metadata.dynamic_budget_validation is False
    assert metadata.metric_claim_allowed is False
    assert metadata.paper_claim_allowed is False

    assert cfg.pc_ot_mras_dynamic_budget_hard_export.resolver == "resolve_pc_ot_mras_dynamic_budget_plan"
    assert cfg.inference.load_from_raw_predictions is False
    assert cfg.inference.save_raw_prediction is False

    with pytest.raises(RuntimeError, match="allow_detector_training=False"):
        training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py")
    with pytest.raises(RuntimeError, match="allow_detector_training=False"):
        training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/test.py")
