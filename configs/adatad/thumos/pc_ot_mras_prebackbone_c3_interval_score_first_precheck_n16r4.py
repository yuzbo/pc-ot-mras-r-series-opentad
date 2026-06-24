_base_ = ["./pc_ot_mras_prebackbone_c3_interval_score_first_full_train_candidate_n16r4.py"]


variant_id = "C3-IntervalScoreFirst-BoundaryPacket-Precheck"
route_id = "pc_ot_mras_prebackbone_c3_interval_score_first"
stage_id = "c3_interval_score_first_precheck_n16r4"
route_label = "C3_ORIGINAL_OPTIMIZATION_ROUTE"

experiment_scope = dict(
    variant_id=variant_id,
    route=route_id,
    stage=stage_id,
    route_label=route_label,
    detector_stack="original_adatad_actionformer_adapter",
    backend="OriginalAdaTAD",
    selection_surface="pre_backbone_raw_frame",
    selection_timing="online_before_backbone",
    acquisition_unit="frame",
    budget_protocol="fixed384_over_dense768_interval_score_first_boundary_packet_precheck_only",
    selector_reader="PCOTMRASBoundaryDifficultyTemporalFrameScout",
    selection_strategy="interval_score_first_packet",
    uses_p2=False,
    uses_teacher=False,
    uses_test_gt=False,
    uses_raw_prediction_cache=False,
    changes_input_sampling=True,
    changes_detector_head=False,
    changes_neck=False,
    changes_loss_assignment=False,
    changes_post_processing=False,
    metric_claim_allowed=False,
    deploy_claim_allowed=False,
    runtime_flops_claim_allowed=False,
    paper_claim_allowed=False,
    full_train_approval=False,
)

pc_ot_mras_prebackbone_e2e_acquisition_gate = dict(
    route=route_id,
    stage=stage_id,
    route_label=route_label,
    default_off=True,
    explicit_config_opt_in=True,
    formal_train_candidate=False,
    remote_precheck_only_candidate=True,
    allow_precheck_only=True,
    allow_remote_sync=True,
    allow_slurm=False,
    allow_gpu=False,
    allow_tools_train=False,
    allow_tools_test=False,
    allow_detector_map=False,
    allow_long_training=False,
    allow_joint_selector_detector_training=False,
    allow_dataset_access=False,
    allow_pretrained_initialization=False,
    allow_checkpoint_write=False,
    allow_checkpoint_load=False,
    allow_resume=False,
    allow_raw_prediction_cache=False,
    metric_claim_allowed=False,
    paper_claim_allowed=False,
    runtime_flops_claim_allowed=False,
    deploy_claim_allowed=False,
    full_train_approval=False,
    allowed_entrypoints=(),
)

model = dict(
    frame_selector=dict(
        selection_strategy="interval_score_first_packet",
        frame_score_st_surrogate="global_softmax",
        interval_boundary_budget_ratio=0.50,
        interval_candidate_topk=24,
        max_dense_gap=0,
        max_gap_guard_count=0,
    )
)

work_dir = "exps/thumos/adatad/pc_ot_mras_prebackbone_c3_interval_score_first_precheck_n16r4"
