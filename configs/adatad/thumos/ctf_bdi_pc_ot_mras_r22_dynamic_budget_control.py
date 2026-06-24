_base_ = ["ctf_bdi_pc_ot_mras_r20_value_only_control.py"]

# R22 local-only value-to-budget controller candidate.
# This is a deploy-time protocol layer that converts PC-OT-MRAS value,
# positive difficulty/boundary-risk, and redundancy reader logits into a
# variable-budget dense-position plan. It is not dynamic-budget validation,
# detector mAP evidence, or deployment proof.

r20_pc_ot_mras_value_only_control_gate = None

r22_pc_ot_mras_dynamic_budget_control_gate = dict(
    route="CTF-BDI/PC-OT-MRAS",
    stage="R22_value_to_dynamic_budget_control_candidate",
    reviewed_predecessor="R20_value_signal_and_R21_tensor_temporal_coordinate_local_candidates",
    default_off=True,
    explicit_config_opt_in=True,
    deploy_visible_reader_outputs_only=True,
    local_synthetic_gate_only=True,
    dynamic_budget_protocol_candidate=True,
    dynamic_budget_validation=False,
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
        "synthetic_cpu_value_to_budget_forward",
        "bad_payload_rejection",
        "exact_budget_and_sorted_unique_positions",
        "coverage_share_cap_static_check",
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

pc_ot_mras_dynamic_budget_controller = dict(
    type="PCOTMRASDynamicBudgetController",
    budget_values=(288, 320, 352, 384, 416),
    budget_thresholds=(0.20, 0.35, 0.50, 0.65),
    value_weight=1.0,
    risk_weight=0.50,
    redundancy_weight=0.25,
    transport_weight=0.25,
    coverage_share=0.35,
    max_coverage_share=0.65,
    require_value_logits=True,
)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)

work_dir = "exps/thumos/adatad/ctf_bdi_pc_ot_mras_r22_dynamic_budget_control"
