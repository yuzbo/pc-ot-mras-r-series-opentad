_base_ = ["./pc_ot_mras_coarse_actionness_uncertainty_c3_physical_grid_actionformer_n16r4.py"]

variant_id = "C3-UniformBiasedCoarseActionness-PreBackbone-OriginalAdaTAD"
route_id = "pc_ot_mras_uniform_biased_coarse_actionness_c3_physical_grid_actionformer"
stage_id = "c3_uniform_biased_coarse_actionness_fixed384_n16r4"

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
    budget_protocol="fixed384_over_dense768_uniform288_action72_uncertainty24_guard12_maxgap3",
    selector_reader="PCOTMRASCoarseActionnessFrameScout",
    selection_strategy="uniform_scaffold_small_actionness_bias_maxgap3",
    uses_learned_boundary_head=False,
    metric_claim_allowed=False,
    deploy_claim_allowed=False,
    runtime_flops_claim_allowed=False,
    paper_claim_allowed=False,
)

protocol_flags = dict(
    changed_surface="input_sampling_plus_head_temporal_geometry",
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
    selector_support_status="uniform_scaffold_small_actionness_bias_guard12_maxgap3",
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
        max_dense_gap=3,
        max_gap_guard_count=12,
        coarse_uniform_count=288,
        coarse_action_count=72,
        coarse_uncertainty_count=24,
        coarse_change_count=0,
        coarse_background_count=0,
        coarse_action_weight=1.0,
        coarse_uncertainty_weight=0.35,
        coarse_change_weight=0.0,
        meta_source="c3_uniform_biased_coarse_actionness_guard12_maxgap3_prebackbone_selector",
    ),
)

work_dir = "exps/thumos/adatad/pc_ot_mras_uniform_biased_coarse_actionness_c3_physical_grid_actionformer_n16r4"
