import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "adatad" / "thumos" / "ctf_bdi_pc_ot_mras_r29_tubelet_packed_profile_audit.py"
GUARD_PATH = ROOT / "opentad" / "utils" / "training_guard.py"


def _load_guard():
    spec = importlib.util.spec_from_file_location("training_guard_for_r29_test", GUARD_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_r29_tubelet_packed_profile_config_is_parseable_and_launch_blocked():
    mmengine_config = pytest.importorskip("mmengine.config")
    training_guard = _load_guard()

    cfg = mmengine_config.Config.fromfile(str(CONFIG))
    gate = cfg.r29_pc_ot_mras_tubelet_packed_profile_audit_gate

    assert gate.route == "CTF-BDI/PC-OT-MRAS"
    assert gate.stage == "R29_tubelet_packed_profile_audit_candidate"
    assert gate.changed_surface == "profiler_only_token_packing_audit"
    assert gate.input_sampling_changed is False
    assert gate.dynamic_budget_policy_changed is False
    assert gate.token_compression_changed is False
    assert gate.backbone_auxiliary_changed is False
    assert gate.detector_head_changed is False
    assert gate.route_unit == "temporal_tubelet_group"
    assert gate.spatial_patch_crop_allowed is False
    assert gate.spatial_filtering_allowed is False
    assert gate.arbitrary_spatial_patch_filtering_allowed is False
    assert gate.profiler_only is True
    assert gate.pack_scatter_bookkeeping_proof is True
    assert gate.true_packed_compute_enabled is False
    assert gate.packed_attention_executed is False
    assert gate.packed_mlp_executed is False
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
    assert gate.metric_claim_allowed is False
    assert gate.paper_claim_allowed is False

    audit = cfg.pc_ot_mras_tubelet_packed_profile_audit
    assert audit.auditor == "audit_pc_ot_mras_tubelet_packed_profile"
    assert audit.summary_schema == "pc_ot_mras_tubelet_packed_profile_audit_v0"
    assert audit.source_aux_summary_schema == "tubelet_token_redundancy_aux_summary_v0"
    assert audit.route_unit == "temporal_tubelet_group"
    assert audit.keep_ratio == 0.5
    assert audit.profiler_only is True
    assert audit.true_packed_compute_enabled is False
    assert audit.packed_attention_executed is False
    assert audit.packed_mlp_executed is False
    assert audit.measured_runtime is False
    assert audit.measured_flops is False
    assert audit.scatter_back_required_for_next_stage is True
    assert audit.runtime_flops_claim_allowed is False
    assert audit.metric_claim_allowed is False
    assert audit.paper_claim_allowed is False

    assert cfg.inference.load_from_raw_predictions is False
    assert cfg.inference.save_raw_prediction is False

    with pytest.raises(RuntimeError, match="allow_detector_training=False"):
        training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py")
    with pytest.raises(RuntimeError, match="allow_detector_training=False"):
        training_guard.assert_detector_training_allowed(cfg, entrypoint="tools/test.py")
