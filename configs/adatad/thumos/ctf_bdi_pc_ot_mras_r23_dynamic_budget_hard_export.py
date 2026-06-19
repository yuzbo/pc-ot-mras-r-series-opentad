_base_ = ["ctf_bdi_pc_ot_mras_r22_dynamic_budget_control.py"]

# R23 local-only dynamic-budget hard export candidate.
# This turns an R22 value-to-budget plan into validated hard-position rows for
# later detector-geometry plumbing. It is not detector mAP, runtime proof,
# dynamic-budget quality validation, or deployment evidence.

r23_pc_ot_mras_dynamic_budget_hard_export_gate = dict(
    route="CTF-BDI/PC-OT-MRAS",
    stage="R23_dynamic_budget_hard_export_candidate",
    reviewed_predecessor="R22_value_to_dynamic_budget_control_candidate",
    default_off=True,
    explicit_config_opt_in=True,
    local_synthetic_gate_only=True,
    dynamic_budget_protocol_candidate=True,
    dynamic_budget_validation=False,
    hard_export_protocol_candidate=True,
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
        "synthetic_cpu_dynamic_budget_hard_export",
        "bad_payload_rejection",
        "exact_variable_budget_rows",
        "sorted_unique_positions",
        "dense_selected_mask_static_check",
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

pc_ot_mras_dynamic_budget_hard_export = dict(
    resolver="resolve_pc_ot_mras_dynamic_budget_plan",
    source_plan_schema="pc_ot_mras_dynamic_budget_plan_v0",
    output_schema="pc_ot_mras_hard_positions_v0",
    generation_source="pc_ot_mras_dynamic_budget_hard_export_resolver_v0",
    detached_reader_tensors=True,
    training_backprop_allowed=False,
    dynamic_budget_validation=False,
    metric_claim_allowed=False,
    paper_claim_allowed=False,
)

work_dir = "exps/thumos/adatad/ctf_bdi_pc_ot_mras_r23_dynamic_budget_hard_export"
