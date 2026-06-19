_base_ = ["ctf_bdi_pc_ot_mras_r23_dynamic_budget_hard_export.py"]

# R24 local-only temporal-metadata contract candidate.
# This converts validated R23 hard-position rows into detector metadata for
# temporal_grid_from_metas. It is not detector mAP, runtime proof, dynamic-budget
# quality validation, or deployment evidence.

r24_pc_ot_mras_dynamic_budget_temporal_metadata_gate = dict(
    route="CTF-BDI/PC-OT-MRAS",
    stage="R24_dynamic_budget_temporal_metadata_candidate",
    reviewed_predecessor="R23_dynamic_budget_hard_export_candidate",
    default_off=True,
    explicit_config_opt_in=True,
    local_synthetic_gate_only=True,
    dynamic_budget_protocol_candidate=True,
    temporal_metadata_protocol_candidate=True,
    detector_geometry_protocol_candidate=True,
    dynamic_budget_validation=False,
    deploy_visible_reader_outputs_only=True,
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
    allow_train_value_targets_at_test=False,
    metric_claim_allowed=False,
    paper_claim_allowed=False,
    runtime_flops_claim_allowed=False,
    deploy_claim_allowed=False,
    scanner_quality_claim_allowed=False,
    dynamic_budget_claim_allowed=False,
    allowed_checks=(
        "mmengine_config_parse",
        "static_config_contract",
        "synthetic_dynamic_budget_to_temporal_metadata",
        "sampling_contract_validation",
        "temporal_grid_from_metas_validation",
        "bad_payload_rejection",
        "selected_mask_consistency_rejection",
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
        "train_value_targets_at_test",
        "runtime_or_flops_claim",
        "deploy_or_scanner_quality_claim",
        "dynamic_budget_validation",
        "metric_claim",
        "paper_claim",
    ),
)

pc_ot_mras_dynamic_budget_temporal_metadata = dict(
    converter="pc_ot_mras_hard_rows_to_temporal_metas",
    source_schema="pc_ot_mras_hard_positions_v0",
    output_schema="pc_ot_mras_temporal_metadata_v0",
    generation_source="pc_ot_mras_hard_rows_to_temporal_metadata_v0",
    contract_validator="validate_sampling_contract",
    temporal_grid_builder="temporal_grid_from_metas",
    irregular_native_axis=True,
    selected_valid_len_semantics="carried_forward_dense_valid_len_alias",
    training_backprop_allowed=False,
    dynamic_budget_validation=False,
    metric_claim_allowed=False,
    paper_claim_allowed=False,
)

work_dir = "exps/thumos/adatad/ctf_bdi_pc_ot_mras_r24_dynamic_budget_temporal_metadata"
