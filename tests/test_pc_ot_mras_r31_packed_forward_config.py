import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "adatad" / "thumos" / "ctf_bdi_pc_ot_mras_r31_packed_forward_optin_local.py"
GUARD_PATH = ROOT / "opentad" / "utils" / "training_guard.py"


def _load_guard():
    spec = importlib.util.spec_from_file_location("training_guard_for_r31_test", GUARD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_r31_packed_forward_config_is_parseable_default_off_and_launch_blocked():
    mmengine_config = pytest.importorskip("mmengine.config")
    training_guard = _load_guard()

    cfg = mmengine_config.Config.fromfile(str(CONFIG))
    gate = cfg.r31_pc_ot_mras_packed_forward_optin_gate

    assert gate.route == "CTF-BDI/PC-OT-MRAS"
    assert gate.stage == "R31_packed_tubelet_forward_optin_local"
    assert gate.changed_surface == "production_backbone_forward_optin_local"
    assert gate.default_off is True
    assert gate.explicit_config_opt_in is True
    assert gate.input_sampling_changed is False
    assert gate.dynamic_budget_policy_changed is False
    assert gate.token_compression_changed is True
    assert gate.backbone_auxiliary_changed is True
    assert gate.adapter_internals_changed is False
    assert gate.production_forward_changed is True
    assert gate.detector_head_changed is False
    assert gate.loss_assignment_changed is False
    assert gate.post_processing_changed is False
    assert gate.route_unit == "temporal_tubelet_group"
    assert gate.spatial_patch_crop_allowed is False
    assert gate.spatial_filtering_allowed is False
    assert gate.adapter_blocks_supported is True
    assert gate.adapter_dense_contract_preserved is True
    assert gate.dense_scatter_before_adapter is True
    assert gate.unselected_identity_bypass_before_adapter is True
    assert gate.adapter_block_fail_closed is False
    assert gate.training_mode_allowed is False
    assert gate.local_forward_only is True
    assert gate.measured_runtime is False
    assert gate.measured_flops is False
    assert gate.allow_detector_training is False
    assert gate.allow_tools_train is False
    assert gate.allow_tools_test is False
    assert gate.allow_detector_map is False
    assert gate.allow_remote_sync is False
    assert gate.allow_precheck_only is False
    assert gate.allow_slurm is False
    assert gate.allow_gpu is False
    assert gate.runtime_flops_claim_allowed is False
    assert gate.metric_claim_allowed is False
    assert gate.paper_claim_allowed is False

    route_cfg = cfg.model.backbone.tubelet_packed_runtime_route
    assert route_cfg.enabled is False
    assert route_cfg.mode == "deterministic_tubelet_cap"
    assert route_cfg.route_unit == "temporal_tubelet_group"
    assert route_cfg.keep_ratio == 0.5
    assert route_cfg.forbid_spatial_crop is True
    assert route_cfg.local_forward_only is True
    assert route_cfg.require_no_adapter_blocks is False
    assert route_cfg.allow_training_mode is False
    assert route_cfg.scatter_unselected == "identity"

    smoke = cfg.pc_ot_mras_tubelet_packed_forward_smoke
    assert smoke.auditor == "synthetic_vit_adapter_forward_optin"
    assert smoke.summary_schema == "packed_tubelet_runtime_route_summary_v0"
    assert smoke.adapter_blocks_supported is True
    assert smoke.adapter_dense_contract_preserved is True
    assert smoke.dense_scatter_before_adapter is True
    assert smoke.unselected_identity_bypass_before_adapter is True
    assert smoke.adapter_block_fail_closed is False
    assert smoke.runtime_flops_claim_allowed is False
    assert smoke.metric_claim_allowed is False
    assert smoke.paper_claim_allowed is False

    assert cfg.inference.load_from_raw_predictions is False
    assert cfg.inference.save_raw_prediction is False

    with pytest.raises(RuntimeError, match="allow_detector_training=False"):
        training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py")
    with pytest.raises(RuntimeError, match="allow_detector_training=False"):
        training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/test.py")
