_base_ = ["./pc_ot_mras_uniform_biased_coarse_actionness_c3_physical_grid_actionformer_n16r4.py"]

variant_id = "C3-BiasedGuard12UniformScaffold-PreBackbone-OriginalAdaTAD"
route_id = "pc_ot_mras_biased_guard12_uniform_scaffold_c3_physical_grid_actionformer"
stage_id = "c3_biased_guard12_uniform_scaffold_fixed384_n16r4"

experiment_scope = dict(
    variant_id=variant_id,
    route=route_id,
    stage=stage_id,
    selection_strategy="uniform_scaffold_tiny_actionness_uncertainty_bias_guard12_maxgap3",
    budget_protocol="fixed384_over_dense768_uniform320_action48_uncertainty16_guard12_maxgap3_no_change",
)

pc_ot_mras_prebackbone_e2e_acquisition_gate = dict(
    route=route_id,
    stage=stage_id,
    selector_support_status="uniform_scaffold_tiny_actionness_uncertainty_bias_guard12_maxgap3_no_change",
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
        coarse_uniform_count=320,
        coarse_action_count=48,
        coarse_uncertainty_count=16,
        coarse_change_count=0,
        coarse_background_count=0,
        coarse_action_weight=0.75,
        coarse_uncertainty_weight=0.20,
        coarse_change_weight=0.0,
        meta_source="c3_biased_guard12_uniform_scaffold_tiny_actionness_uncertainty_prebackbone_selector",
    ),
)

work_dir = "exps/thumos/adatad/pc_ot_mras_biased_guard12_uniform_scaffold_c3_physical_grid_actionformer_n16r4"
