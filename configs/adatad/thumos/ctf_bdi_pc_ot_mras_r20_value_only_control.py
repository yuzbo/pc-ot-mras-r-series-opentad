_base_ = ["ctf_bdi_pc_ot_mras_r17_formal_train_candidate.py"]

# R20 value-only attribution control.
# This keeps the R17 PC-OT-MRAS reader/bridge route, enables R20 value heads
# and train-only value loss, and intentionally does not inherit R18 aux loss.
# It is launch-blocked until a separate review gate approves any execution.

r17_pc_ot_mras_formal_train_gate = None

r20_pc_ot_mras_value_only_control_gate = dict(
    route="CTF-BDI/PC-OT-MRAS",
    stage="R20_value_only_attribution_control",
    reviewed_predecessor="clean_repo_R20_combo_commit_2301aaa",
    default_off=True,
    explicit_config_opt_in=True,
    attribution_control=True,
    train_only_value_distillation=True,
    local_synthetic_gate_only=True,
    requires_launch_gate=True,
    launch_gate_passed=False,
    allow_detector_training=False,
    allow_tools_train=False,
    allow_tools_test=False,
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
    dynamic_budget_claim_allowed=False,
    allowed_checks=(
        "mmengine_config_parse",
        "static_config_contract",
        "synthetic_cpu_value_loss_forward_backward",
        "train_only_forward_test_guard",
        "bad_provenance_rejection",
        "target_strip_before_head",
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
        "runtime_or_flops_claim",
        "deploy_or_scanner_quality_claim",
        "dynamic_budget_validation",
        "metric_claim",
        "paper_claim",
    ),
)

model = dict(
    pc_ot_mras_reader=dict(
        enable_value_heads=True,
    ),
    pc_ot_mras_reader_value_loss=dict(
        enabled=True,
        require_targets=True,
        target_source="unit_counterfactual_value_distill_v0",
        allow_train_gt=True,
        allow_teacher_targets=False,
        pair_temperature=1.0,
        weights=dict(
            dense_value=0.02,
            dense_risk=0.01,
            dense_redundancy=0.005,
            acquisition=0.05,
            allocation=0.01,
            gate=0.01,
            pair_operation=0.03,
        ),
    ),
)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)

work_dir = "exps/thumos/adatad/ctf_bdi_pc_ot_mras_r20_value_only_control"
