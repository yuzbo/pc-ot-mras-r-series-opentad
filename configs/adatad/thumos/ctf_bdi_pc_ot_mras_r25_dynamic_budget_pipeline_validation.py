_base_ = ["ctf_bdi_pc_ot_mras_r24_dynamic_budget_temporal_metadata.py"]

# R25 local-only dynamic-budget pipeline validation candidate.
# This validates the R22 -> R23 -> R24 protocol chain from deploy-visible
# dynamic-budget plans to hard rows, temporal metadata, sampling contract, and
# temporal_grid_from_metas. It is not detector mAP, runtime proof,
# dynamic-budget quality validation, scanner-quality validation, deployment
# evidence, or a paper claim.

r25_pc_ot_mras_dynamic_budget_pipeline_validation_gate = dict(
    route="CTF-BDI/PC-OT-MRAS",
    stage="R25_dynamic_budget_pipeline_validation_candidate",
    reviewed_predecessor="R24_dynamic_budget_temporal_metadata_candidate",
    default_off=True,
    explicit_config_opt_in=True,
    local_synthetic_gate_only=True,
    dynamic_budget_protocol_candidate=True,
    hard_export_protocol_candidate=True,
    temporal_metadata_protocol_candidate=True,
    detector_geometry_protocol_candidate=True,
    pipeline_protocol_validation_candidate=True,
    dynamic_budget_validation=False,
    scanner_quality_validation=False,
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
        "synthetic_r22_r23_r24_pipeline_validation",
        "exact_variable_budget_counts",
        "sampling_contract_validation",
        "temporal_grid_from_metas_validation",
        "bad_payload_rejection",
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
        "scanner_quality_validation",
        "metric_claim",
        "paper_claim",
    ),
)

pc_ot_mras_dynamic_budget_pipeline_validation = dict(
    validator="validate_pc_ot_mras_dynamic_budget_pipeline",
    source_plan_schema="pc_ot_mras_dynamic_budget_plan_v0",
    hard_row_schema="pc_ot_mras_hard_positions_v0",
    temporal_metadata_schema="pc_ot_mras_temporal_metadata_v0",
    summary_schema="pc_ot_mras_dynamic_budget_pipeline_validation_v0",
    validates_steps=(
        "resolve_pc_ot_mras_dynamic_budget_plan",
        "pc_ot_mras_hard_rows_to_temporal_metas",
        "validate_sampling_contract",
        "temporal_grid_from_metas",
    ),
    detached_reader_tensors=True,
    training_backprop_allowed=False,
    dynamic_budget_validation=False,
    scanner_quality_validation=False,
    metric_claim_allowed=False,
    paper_claim_allowed=False,
)

work_dir = "exps/thumos/adatad/ctf_bdi_pc_ot_mras_r25_dynamic_budget_pipeline_validation"
