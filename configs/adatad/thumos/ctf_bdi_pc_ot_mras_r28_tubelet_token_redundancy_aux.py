_base_ = ["ctf_bdi_pc_ot_mras_r27_synthetic_task_utility_audit.py"]

# R28 local-only tubelet/token redundancy auxiliary candidate.
# This is the first controlled bridge from temporal dynamic acquisition toward
# backbone-stage token compute routing. It uses spatial-token statistics only
# to audit/propose temporal tubelet-group redundancy; it does not crop spatial
# patches, change dense output length, run detector mAP, or claim FLOPs.

r28_pc_ot_mras_tubelet_token_redundancy_aux_gate = dict(
    route="CTF-BDI/PC-OT-MRAS",
    stage="R28_tubelet_token_redundancy_aux_candidate",
    reviewed_predecessor="R23_R27_integrated_dynamic_acquisition_iteration",
    default_off=True,
    explicit_config_opt_in=True,
    local_synthetic_gate_only=True,
    changed_surface="backbone_auxiliary_audit",
    input_sampling_changed=False,
    dynamic_budget_policy_changed=False,
    token_compression_changed=False,
    backbone_auxiliary_changed=True,
    adapter_internals_changed=False,
    detector_head_changed=False,
    loss_assignment_changed=False,
    post_processing_changed=False,
    temporal_metadata_changed=False,
    route_unit="temporal_tubelet_group",
    spatial_patch_crop_allowed=False,
    dense_output_length_preserved=True,
    r23_r27_metadata_preserved=True,
    true_packed_compute_enabled=False,
    local_audit_only=True,
    requires_launch_gate=True,
    launch_gate_passed=False,
    allow_detector_training=False,
    allow_tools_train=False,
    allow_tools_test=False,
    allow_detector_map=False,
    allow_remote_sync=False,
    allow_precheck_only=False,
    allow_slurm=False,
    allow_gpu=False,
    allow_real_dataset=False,
    allow_checkpoint=False,
    allow_raw_prediction_cache=False,
    metric_claim_allowed=False,
    paper_claim_allowed=False,
    runtime_flops_claim_allowed=False,
    deploy_claim_allowed=False,
    scanner_quality_claim_allowed=False,
    dynamic_budget_claim_allowed=False,
    spatial_redundancy_claim_allowed=False,
    allowed_checks=(
        "mmengine_config_parse",
        "static_config_contract",
        "tubelet_token_redundancy_aux_contract",
        "synthetic_tubelet_token_redundancy_audit",
        "dense_output_preserved",
        "no_spatial_patch_crop",
        "py_compile_changed_files",
    ),
    forbidden_checks=(
        "remote_sync",
        "remote_precheck_only",
        "slurm_or_gpu",
        "tools_train",
        "tools_test_or_map",
        "real_dataset_or_checkpoint",
        "raw_prediction_cache",
        "true_packed_compute_runtime_claim",
        "runtime_or_flops_claim",
        "deploy_or_scanner_quality_claim",
        "spatial_redundancy_claim",
        "metric_claim",
        "paper_claim",
    ),
)

model = dict(
    backbone=dict(
        tubelet_token_redundancy_aux=dict(
            enabled=False,
            mode="shadow",
            route_unit="temporal_tubelet_group",
            keep_ratio=0.75,
            min_keep_tubelets=1,
            route_pattern="round_linspace",
            spatial_pool="energy_std",
            forbid_spatial_crop=True,
            local_audit_only=True,
        )
    )
)

pc_ot_mras_tubelet_token_redundancy_audit = dict(
    auditor="audit_pc_ot_mras_tubelet_token_redundancy",
    summary_schema="pc_ot_mras_tubelet_token_redundancy_audit_v0",
    aux_summary_schema="tubelet_token_redundancy_aux_summary_v0",
    route_unit="temporal_tubelet_group",
    mode="deterministic_tubelet_cap",
    synthetic_only=True,
    keep_ratio=0.75,
    spatial_patch_crop_allowed=False,
    dense_output_length_preserved=True,
    adapter_dense_contract_preserved=True,
    r23_r27_temporal_metadata_preserved=True,
    true_packed_compute_enabled=False,
    detector_training_allowed=False,
    detector_map_allowed=False,
    remote_sync_allowed=False,
    runtime_flops_claim_allowed=False,
    metric_claim_allowed=False,
    paper_claim_allowed=False,
)

work_dir = "exps/thumos/adatad/ctf_bdi_pc_ot_mras_r28_tubelet_token_redundancy_aux"
