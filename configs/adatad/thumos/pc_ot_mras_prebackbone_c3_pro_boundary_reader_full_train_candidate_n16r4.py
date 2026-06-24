_base_ = ["./pc_ot_mras_prebackbone_c3_hybrid_reader_full_train_candidate_n16r4.py"]


variant_id = "C3-Pro-BoundaryDifficulty-OriginalAdaTAD"
route_id = "pc_ot_mras_prebackbone_c3_pro_boundary_reader"
stage_id = "c3_pro_boundary_reader_full_train_candidate_n16r4"
reader_type = "PCOTMRASBoundaryDifficultyTemporalFrameScout"
reader_family = "BoundaryDifficultyTemporal"
max_dense_gap = 0
max_gap_guard_count = 0

experiment_scope = dict(
    variant_id=variant_id,
    route=route_id,
    stage=stage_id,
    boundary_lock=(
        "C3-Pro-BoundaryDifficulty uses compressed low-resolution pixels, dense "
        "action/start/end/uncertainty/redundancy heads, frame-score-first hard top-k "
        "selection, and ST frame-score gradients before the unchanged Original AdaTAD/ActionFormer detector."
    ),
    selector_reader=reader_type,
    reader_family=reader_family,
    train_protocol="joint_selector_detector_fixed50_prebackbone_full_train",
    budget_protocol="fixed384_over_dense768_frame_score_first_topk_no_hard_uniform_guard",
    selection_strategy="frame_score_topk",
    complete_training_required=True,
)

pc_ot_mras_prebackbone_e2e_acquisition_gate = dict(
    route=route_id,
    stage=stage_id,
    formal_train_candidate=True,
)

model = dict(
    frame_selector=dict(
        max_dense_gap=max_dense_gap,
        max_gap_guard_count=max_gap_guard_count,
        selection_strategy="frame_score_topk",
        frame_score_st_temperature=1.0,
        frame_score_st_local_width=8.0,
        frame_score_st_local_bias_weight=1.0,
        reader_regularizer_loss_weight=0.0,
        reader=dict(
            _delete_=True,
            type=reader_type,
            in_dim=3072,
            hidden_dim=128,
            num_slots=384,
            temporal_layers=3,
            temporal_kernel_size=5,
            dilations=(1, 2, 4),
            dropout=0.05,
            descriptor_hidden_dim=128,
            slot_temperature_init=1.0,
            geometry_width=0.015,
            geometry_bias_weight=0.75,
            action_bias_weight=0.40,
            boundary_bias_weight=0.60,
            uncertainty_bias_weight=0.25,
            redundancy_bias_weight=0.25,
            slot_logit_clamp=30.0,
            soft_order_regularizer_weight=1.0,
            duplicate_mass_regularizer_weight=0.2,
            duplicate_mass_cap_factor=4.0,
            local_global_fusion="boundary_difficulty_temporal_cnn_slot_attention",
        ),
    )
)

work_dir = "exps/thumos/adatad/pc_ot_mras_prebackbone_c3_pro_boundary_reader_full_train_candidate_n16r4"
