_base_ = ["ctf_bdi_pc_ot_mras_r18_aux_formal_train_candidate.py"]

# R19 train-only soft/hard consistency candidate.
# This is the next PC-OT-MRAS increment after R18: keep the R18 semantic
# auxiliary reader anchors, then add a train-only consistency loss between the
# differentiable soft allocation and detached hard-export anchors. It is local
# and launch-blocked until a separate review/execution gate approves it.

r18_pc_ot_mras_aux_diag_gate = None

r19_pc_ot_mras_soft_hard_consistency_gate = dict(
    route="CTF-BDI/PC-OT-MRAS",
    stage="R19_soft_hard_consistency_candidate",
    reviewed_predecessor="R18_aux_formal_candidate_clean_HEAD_f18c708",
    default_off=True,
    explicit_config_opt_in=True,
    train_only_auxiliary_diagnostic=True,
    train_only_soft_hard_consistency=True,
    local_synthetic_gate_only=True,
    allow_detector_training=False,
    requires_launch_gate=True,
    launch_gate_passed=False,
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
    training_signal="detector_loss_plus_R18_aux_plus_R19_soft_hard_consistency_v0",
    allowed_checks=(
        "mmengine_config_parse",
        "static_config_contract",
        "synthetic_cpu_soft_hard_loss_forward_backward",
        "actionformer_train_only_loss_merge",
        "forward_test_no_loss_or_target_path",
        "guard_contract",
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
    pc_ot_mras_reader_soft_hard_loss=dict(
        enabled=True,
        weights=dict(
            slot_allocation=0.02,
            global_acquisition=0.02,
            selected_time=0.01,
            gate_confidence=0.005,
            duplicate_mass=0.005,
        ),
        eps=1.0e-6,
    ),
)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)

work_dir = "exps/thumos/adatad/ctf_bdi_pc_ot_mras_r19_soft_hard_consistency_candidate"
