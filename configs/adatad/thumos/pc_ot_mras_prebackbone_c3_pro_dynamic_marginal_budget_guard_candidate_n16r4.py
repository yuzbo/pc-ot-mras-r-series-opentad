_base_ = ["./pc_ot_mras_prebackbone_c3_pro_boundary_reader_full_train_candidate_n16r4.py"]


variant_id = "C3-Pro-DynamicMarginalUtility-BoundaryDifficulty-OriginalAdaTAD"
route_id = "pc_ot_mras_prebackbone_c3_pro_dynamic_marginal_budget_guard"
stage_id = "c3_pro_dynamic_marginal_budget_guard_candidate_n16r4"
max_dense_gap = 64
max_gap_guard_count = 12
output_capacity = 448
target_budget = 384
average_budget = 384
min_budget = 320
max_budget = 448

experiment_scope = dict(
    variant_id=variant_id,
    route=route_id,
    stage=stage_id,
    detector_stack="original_adatad_actionformer_adapter",
    backend="OriginalAdaTAD",
    selection_surface="pre_backbone_raw_frame",
    selection_timing="online_before_backbone",
    acquisition_unit="frame",
    budget_protocol="dynamic_marginal_utility_min320_target384_avg384_max448_over_dense768_frame_score_first_topk",
    dynamic_budget_protocol_candidate=True,
    dynamic_budget_claim_allowed=False,
    max_gap_guard_safety_gate=True,
    safety_gate_not_final_innovation=True,
    output_capacity=output_capacity,
    selector_reader="PCOTMRASBoundaryDifficultyTemporalFrameScout",
    selection_strategy="frame_score_topk",
    deploy_visible_signals=(
        "frame_selection_logits",
        "actionness_logits",
        "boundary_logits",
        "uncertainty_logits",
        "redundancy_logits",
        "valid_len",
    ),
    min_budget=min_budget,
    target_budget=target_budget,
    average_budget=average_budget,
    max_budget=max_budget,
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
    dynamic_budget_protocol_candidate=True,
    dynamic_budget_claim_allowed=False,
    max_gap_guard_safety_gate=True,
    safety_gate_not_final_innovation=True,
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
        target_len=output_capacity,
        max_dense_gap=max_dense_gap,
        max_gap_guard_count=max_gap_guard_count,
        dynamic_budget=dict(
            enabled=True,
            protocol="marginal_utility_v0",
            min_budget=min_budget,
            target_budget=target_budget,
            average_budget=average_budget,
            max_budget=max_budget,
            budget_step=32,
            score_midpoint=0.5,
            actionness_weight=1.0,
            boundary_weight=0.35,
            uncertainty_weight=0.20,
            redundancy_weight=0.35,
            valid_len_weight=0.05,
            deploy_visible_signals=(
                "frame_selection_logits",
                "actionness_logits",
                "boundary_logits",
                "uncertainty_logits",
                "redundancy_logits",
                "valid_len",
            ),
        ),
        reader=dict(num_slots=output_capacity),
    ),
    backbone=dict(
        backbone=dict(total_frames=output_capacity),
    ),
    projection=dict(
        max_seq_len=output_capacity,
    )
)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)

work_dir = "exps/thumos/adatad/pc_ot_mras_prebackbone_c3_pro_dynamic_marginal_budget_guard_candidate_n16r4"
