_base_ = ["./pc_ot_mras_uniform_biased_coarse_actionness_c3_physical_grid_actionformer_n16r4.py"]

variant_id = "C3-A-UniformScaffoldSmallActionnessStrictMaxGap-PreBackbone-OriginalAdaTAD"
route_id = "pc_ot_mras_a_uniform_scaffold_small_actionness_strict_maxgap_c3_physical_grid_actionformer"
stage_id = "c3_a_uniform_scaffold_small_actionness_strict_maxgap_fixed384_n16r4"

experiment_scope = dict(
    variant_id=variant_id,
    route=route_id,
    stage=stage_id,
    selection_strategy="uniform_scaffold_small_actionness_strict_maxgap",
    budget_protocol="fixed384_over_dense768_uniform_scaffold_small_actionness_strict_maxgap_guard12",
)

pc_ot_mras_prebackbone_e2e_acquisition_gate = dict(
    route=route_id,
    stage=stage_id,
    selector_support_status="a_uniform_scaffold_small_actionness_strict_maxgap_guard12_maxgap3",
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
        meta_source="c3_a_uniform_scaffold_small_actionness_strict_maxgap_prebackbone_selector",
    ),
)

work_dir = "exps/thumos/adatad/pc_ot_mras_a_uniform_scaffold_small_actionness_strict_maxgap_c3_physical_grid_actionformer_n16r4"
