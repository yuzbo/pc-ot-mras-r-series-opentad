_base_ = ["./pc_ot_mras_prebackbone_c3_pro_boundary_reader_full_train_candidate_n16r4.py"]


variant_id = "C3-GlobalRankST-BoundaryDifficulty-OriginalAdaTAD"
route_id = "pc_ot_mras_prebackbone_c3_global_rank_st"
stage_id = "c3_global_rank_st_full_train_candidate_n16r4"

experiment_scope = dict(
    variant_id=variant_id,
    route=route_id,
    stage=stage_id,
    detector_stack="original_adatad_actionformer_adapter",
    backend="OriginalAdaTAD",
    selection_surface="pre_backbone_raw_frame",
    selection_timing="online_before_backbone",
    acquisition_unit="frame",
    budget_protocol="fixed384_over_dense768_frame_score_first_global_rank_st",
    selector_reader="PCOTMRASBoundaryDifficultyTemporalFrameScout",
    selection_strategy="frame_score_topk",
    rank_transport_surrogate="global_softmax",
    uses_p2=False,
    uses_offline_ledger=False,
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
)

pc_ot_mras_prebackbone_e2e_acquisition_gate = dict(
    route=route_id,
    stage=stage_id,
    formal_train_candidate=True,
    allow_tools_test=False,
    allow_detector_map=False,
    allow_checkpoint_load=False,
    allow_resume=False,
    allow_raw_prediction_cache=False,
    metric_claim_allowed=False,
    paper_claim_allowed=False,
    runtime_flops_claim_allowed=False,
    deploy_claim_allowed=False,
)

model = dict(
    frame_selector=dict(
        selection_strategy="frame_score_topk",
        frame_score_st_surrogate="global_softmax",
        max_dense_gap=0,
        max_gap_guard_count=0,
    )
)

work_dir = "exps/thumos/adatad/pc_ot_mras_prebackbone_c3_global_rank_st_full_train_candidate_n16r4"
