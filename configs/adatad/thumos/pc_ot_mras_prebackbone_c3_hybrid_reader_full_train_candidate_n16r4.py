_base_ = ["./pc_ot_mras_prebackbone_c3_hybrid_reader_candidate_n16r4.py"]


variant_id = "C3-Hybrid-Reader-OriginalAdaTAD"
route_id = "pc_ot_mras_prebackbone_c3_hybrid_reader"
stage_id = "c3_hybrid_reader_full_train_candidate_n16r4"

experiment_scope = dict(
    route=route_id,
    stage=stage_id,
    budget_protocol="fixed384_over_dense768_frame_level_full_train",
    train_protocol="joint_selector_detector_fixed50_prebackbone_full_train",
    complete_training_required=True,
)

pc_ot_mras_prebackbone_e2e_acquisition_gate = dict(
    _delete_=True,
    route=route_id,
    stage=stage_id,
    default_off=False,
    explicit_config_opt_in=True,
    formal_train_candidate=True,
    selection_surface="pre_backbone_raw_frame",
    selection_timing="online_before_backbone",
    acquisition_unit="frame",
    full_joint_selector_detector_training=True,
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
    allow_train_validation_map=True,
    allow_long_training=True,
    allow_prebackbone_frame_selector=True,
    allow_joint_selector_detector_training=True,
    allow_dataset_access=True,
    allow_pretrained_initialization=True,
    allow_checkpoint_write=True,
    allow_checkpoint_load=False,
    allow_resume=False,
    selector_support_status="supported_by_prebackbone_frame_selector",
    entrypoint_gate_context=dict(
        required=False,
        forbidden_true_keys=(
            "tools_test",
            "allow_tools_test",
            "direct_tools_test",
            "detector_map",
            "allow_detector_map",
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
    allowed_entrypoints=("tools/train.py",),
)

work_dir = "exps/thumos/adatad/pc_ot_mras_prebackbone_c3_hybrid_reader_full_train_candidate_n16r4"
