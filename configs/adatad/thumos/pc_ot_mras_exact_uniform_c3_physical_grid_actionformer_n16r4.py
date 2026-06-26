_base_ = ["./pc_ot_mras_coarse_actionness_uncertainty_c3_physical_grid_actionformer_n16r4.py"]

variant_id = "C3-ExactUniformPhysicalGrid-PreBackbone-OriginalAdaTAD"
route_id = "pc_ot_mras_exact_uniform_c3_physical_grid_actionformer"
stage_id = "c3_exact_uniform_physical_grid_fixed384_n16r4"

route_label = "C3_ORIGINAL_OPTIMIZATION_ROUTE"
route_family = "C3_MAINLINE_OPTIMIZATION"
candidate_name = stage_id
window_size = 384
dense_window_size = 768

experiment_scope = dict(
    variant_id=variant_id,
    route_family=route_family,
    route=route_id,
    stage=stage_id,
    budget_protocol="fixed384_over_dense768_exact_uniform_physical_grid_control",
    selector_reader="PCOTMRASCoarseActionnessFrameScout",
    selection_strategy="exact_uniform_physical_grid_control",
    uses_learned_boundary_head=False,
    metric_claim_allowed=False,
    deploy_claim_allowed=False,
    runtime_flops_claim_allowed=False,
    paper_claim_allowed=False,
)

protocol_flags = dict(
    changed_surface="input_sampling_plus_head_temporal_geometry_control",
    selector_changed=True,
    tools_test_allowed=False,
    tools_train_allowed=True,
    remote_sync_allowed=True,
    slurm_allowed=True,
    metric_claim_allowed=False,
    paper_claim_allowed=False,
)

pc_ot_mras_prebackbone_e2e_acquisition_gate = dict(
    route=route_id,
    stage=stage_id,
    selector_support_status="exact_uniform_control_through_prebackbone_selector_no_reader_influence",
    entrypoint_gate_context=dict(
        required_exact_values=dict(
            route=route_id,
            variant_id=variant_id,
            stage=stage_id,
        ),
    ),
)

model = dict(
    frame_selector=dict(
        selection_strategy="coarse_actionness_uncertainty",
        straight_through_detector_loss=False,
        straight_through_downstream=False,
        aux_gt_acquisition_loss_weight=0.0,
        aux_duplicate_cap_loss_weight=0.0,
        max_dense_gap=0,
        max_gap_guard_count=0,
        coarse_uniform_count=384,
        coarse_action_count=0,
        coarse_uncertainty_count=0,
        coarse_change_count=0,
        coarse_background_count=0,
        coarse_action_weight=1.0,
        coarse_uncertainty_weight=0.0,
        coarse_change_weight=0.0,
        meta_source="c3_exact_uniform_physical_grid_prebackbone_selector_control",
    ),
)

work_dir = "exps/thumos/adatad/pc_ot_mras_exact_uniform_c3_physical_grid_actionformer_n16r4"
