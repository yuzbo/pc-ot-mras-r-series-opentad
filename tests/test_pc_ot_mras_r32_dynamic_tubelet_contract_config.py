import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "adatad" / "thumos" / "ctf_bdi_pc_ot_mras_r32_dynamic_tubelet_contract_audit.py"
GUARD_PATH = ROOT / "opentad" / "utils" / "training_guard.py"


def _load_guard():
    spec = importlib.util.spec_from_file_location("training_guard_for_r32_test", GUARD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_r32_dynamic_tubelet_contract_config_is_parseable_and_launch_blocked():
    mmengine_config = pytest.importorskip("mmengine.config")
    training_guard = _load_guard()

    cfg = mmengine_config.Config.fromfile(str(CONFIG))
    gate = cfg.r32_pc_ot_mras_dynamic_tubelet_contract_gate

    assert gate.route == "CTF-BDI/PC-OT-MRAS"
    assert gate.stage == "R32_dynamic_tubelet_contract_audit"
    assert gate.changed_surface == "dynamic_budget_to_packed_tubelet_contract_audit"
    assert gate.input_sampling_changed is False
    assert gate.dynamic_budget_policy_changed is False
    assert gate.token_compression_changed is True
    assert gate.backbone_auxiliary_changed is False
    assert gate.adapter_internals_changed is False
    assert gate.production_forward_changed is False
    assert gate.detector_head_changed is False
    assert gate.temporal_metadata_changed is True
    assert gate.route_unit == "temporal_tubelet_group"
    assert gate.source_dynamic_plan_schema == "pc_ot_mras_dynamic_budget_plan_v0"
    assert gate.source_hard_position_schema == "pc_ot_mras_hard_positions_v0"
    assert gate.dynamic_budget_to_tubelet_mapping is True
    assert gate.selected_positions_to_tubelet_groups is True
    assert gate.bucketed_pack_fallback_allowed is True
    assert gate.single_rectangular_pack_required is False
    assert gate.dense_scatter_back_shape_checked is True
    assert gate.selected_mask_consistency_checked is True
    assert gate.spatial_patch_crop_allowed is False
    assert gate.spatial_filtering_allowed is False
    assert gate.arbitrary_spatial_patch_filtering_allowed is False
    assert gate.true_packed_compute_enabled is False
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
    assert gate.spatial_redundancy_claim_allowed is False
    assert gate.dynamic_budget_quality_validation is False
    assert gate.metric_claim_allowed is False
    assert gate.paper_claim_allowed is False

    audit = cfg.pc_ot_mras_dynamic_tubelet_contract_audit
    assert audit.auditor == "audit_pc_ot_mras_dynamic_tubelet_contract"
    assert audit.summary_schema == "pc_ot_mras_dynamic_tubelet_contract_audit_v0"
    assert audit.source_dynamic_plan_schema == "pc_ot_mras_dynamic_budget_plan_v0"
    assert audit.source_hard_position_schema == "pc_ot_mras_hard_positions_v0"
    assert audit.route_unit == "temporal_tubelet_group"
    assert audit.temporal_tubelets == 8
    assert audit.spatial_h == 2
    assert audit.spatial_w == 3
    assert audit.channels == 4
    assert audit.bucketed_pack_fallback_allowed is True
    assert audit.synthetic_or_deploy_visible_plan_only is True
    assert audit.local_protocol_audit_only is True
    assert audit.spatial_patch_crop_allowed is False
    assert audit.true_packed_compute_enabled is False
    assert audit.measured_runtime is False
    assert audit.measured_flops is False
    assert audit.runtime_flops_claim_allowed is False
    assert audit.metric_claim_allowed is False
    assert audit.paper_claim_allowed is False

    assert cfg.model.backbone.tubelet_packed_runtime_route.enabled is False
    assert cfg.inference.load_from_raw_predictions is False
    assert cfg.inference.save_raw_prediction is False

    with pytest.raises(RuntimeError, match="allow_detector_training=False"):
        training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py")
    with pytest.raises(RuntimeError, match="allow_detector_training=False"):
        training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/test.py")
