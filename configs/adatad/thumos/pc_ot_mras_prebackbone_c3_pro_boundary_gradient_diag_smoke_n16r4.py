_base_ = ["./pc_ot_mras_prebackbone_c3_pro_boundary_reader_full_train_candidate_n16r4.py"]


variant_id = "C3-Pro-BoundaryDifficulty-GradientDiag"
route_id = "pc_ot_mras_prebackbone_c3_pro_boundary_reader"
stage_id = "c3_pro_boundary_gradient_diag_smoke_n16r4"

experiment_scope = dict(
    variant_id=variant_id,
    route=route_id,
    stage=stage_id,
    boundary_lock=(
        "Diagnostic-only first-step C3-Pro run used to locate the parameter that "
        "produces a non-finite gradient before any further full training."
    ),
    selector_reader="PCOTMRASBoundaryDifficultyTemporalFrameScout",
    train_protocol="single_batch_gradient_diagnostic_only",
    budget_protocol="fixed384_over_dense768_frame_score_first_topk_no_eval_no_checkpoint",
    complete_training_required=False,
    metric_claim_allowed=False,
    deploy_claim_allowed=False,
    runtime_flops_claim_allowed=False,
    paper_claim_allowed=False,
)

pc_ot_mras_prebackbone_e2e_acquisition_gate = dict(
    _delete_=True,
    route=route_id,
    stage=stage_id,
    default_off=False,
    explicit_config_opt_in=True,
    formal_train_candidate=False,
    smoke_only=True,
    selection_surface="pre_backbone_raw_frame",
    selection_timing="online_before_backbone",
    acquisition_unit="frame",
    offline_ledger=False,
    post_projection_bridge=False,
    allow_detector_training=True,
    requires_launch_gate=True,
    launch_gate_passed=True,
    allow_remote_sync=True,
    allow_precheck_only=True,
    allow_slurm=True,
    allow_gpu=True,
    allow_tools_train=True,
    allow_tools_test=False,
    allow_detector_map=False,
    allow_train_validation_map=False,
    allow_long_training=False,
    allow_prebackbone_frame_selector=True,
    allow_joint_selector_detector_training=True,
    allow_dataset_access=True,
    allow_pretrained_initialization=True,
    allow_checkpoint_write=False,
    allow_checkpoint_load=False,
    allow_resume=False,
    max_epochs=1,
    max_train_iters=1,
    disable_checkpoint=True,
    allowed_entrypoints=("tools/train.py",),
    allowed_checks=(
        "slurm_single_gpu_launch",
        "ddp_cuda_dataset_model_optimizer_build",
        "single_batch_backward_gradient_parameter_diagnosis",
    ),
    forbidden_checks=(
        "tools_test_or_map",
        "validation_eval",
        "checkpoint_write",
        "long_training",
        "runtime_flops_claim",
        "deploy_claim",
        "metric_claim",
        "paper_claim",
    ),
    entrypoint_gate_context=dict(
        required=False,
        forbidden_true_keys=(
            "tools_test",
            "allow_tools_test",
            "direct_tools_test",
            "detector_map",
            "allow_detector_map",
            "checkpoint_load",
            "allow_checkpoint_load",
            "resume",
            "allow_resume",
            "offline_ledger",
            "raw_prediction_cache",
            "load_from_raw_predictions",
            "save_raw_prediction",
            "paper_claim",
            "paper_claim_allowed",
            "deploy_claim",
            "deploy_claim_allowed",
            "runtime_claim",
            "runtime_flops_claim",
            "runtime_flops_claim_allowed",
        ),
    ),
    metric_claim_allowed=False,
    paper_claim_allowed=False,
    runtime_flops_claim_allowed=False,
    deploy_claim_allowed=False,
)

workflow = dict(
    logging_interval=1,
    checkpoint_interval=1,
    val_loss_interval=-1,
    val_eval_interval=-1,
    val_start_epoch=999,
    end_epoch=1,
    max_train_iters=1,
    disable_checkpoint=True,
)

solver = dict(
    train=dict(batch_size=1, num_workers=1),
    val=dict(batch_size=1, num_workers=1),
    test=dict(batch_size=1, num_workers=1),
)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)

work_dir = "exps/thumos/adatad/pc_ot_mras_prebackbone_c3_pro_boundary_gradient_diag_smoke_n16r4"
