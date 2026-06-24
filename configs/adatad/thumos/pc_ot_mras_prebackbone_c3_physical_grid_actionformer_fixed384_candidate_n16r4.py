_base_ = ["./pc_ot_mras_prebackbone_c3_pro_boundary_reader_full_train_candidate_n16r4.py"]


variant_id = "C3-PhysicalGridActionFormer-BoundaryDifficulty-OriginalAdaTAD"
route_id = "pc_ot_mras_prebackbone_c3_physical_grid_actionformer"
stage_id = "c3_physical_grid_actionformer_fixed384_candidate_n16r4"

experiment_scope = dict(
    variant_id=variant_id,
    route=route_id,
    stage=stage_id,
    detector_stack="physical_grid_actionformer_adapter",
    backend="OriginalAdaTAD_ActionFormerPhysicalGrid",
    selection_surface="pre_backbone_raw_frame",
    selection_timing="online_before_backbone",
    acquisition_unit="frame",
    budget_protocol="fixed384_over_dense768_frame_score_first_physical_grid_actionformer",
    selector_reader="PCOTMRASBoundaryDifficultyTemporalFrameScout",
    selection_strategy="frame_score_topk",
    temporal_grid_mode="physical",
    uses_p2=False,
    uses_offline_ledger=False,
    uses_teacher=False,
    uses_test_gt=False,
    uses_raw_prediction_cache=False,
    changes_input_sampling=True,
    changes_detector_head=True,
    changes_neck=False,
    changes_loss_assignment=True,
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
        remap_gt_to_selected_axis=False,
    ),
    rpn_head=dict(
        temporal_grid=dict(
            enabled=True,
            temporal_grid_mode="physical",
            decode_axis="dense",
            positions_key="irregular_selected_positions",
            valid_len_key="irregular_selected_valid_len",
            required=True,
            strict=True,
        ),
    ),
)

work_dir = "exps/thumos/adatad/pc_ot_mras_prebackbone_c3_physical_grid_actionformer_fixed384_candidate_n16r4"
