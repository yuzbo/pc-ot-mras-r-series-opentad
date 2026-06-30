_base_ = ["./c3_indirect_original_adatad_32px_a_short_smoke.py"]

c3_route_label = "C3_MAINLINE_OPTIMIZATION"
c3_route_labels = ["C3_MAINLINE_OPTIMIZATION", "C3_ORIGINAL_OPTIMIZATION_ROUTE"]
c3_method = "C3-CADF-DensityMesh-ST-OriginalAdaTAD"
c3_claim_status = "precheck_only"
c3_full_train_claim_unlocked = False
c3_original_adatad_average_stride_backend_risk = "high"
c3_selected_index_aware_postprocess_contract = "metadata_only_default_off"
c3_physical_time_postprocess_enabled = False
c3_dense_window_size = 768
window_size = 384
c3_scout_spatial_size = 32

model = dict(
    frame_selector=dict(
        _delete_=True,
        type="PCOTMRASIndirectPreBackboneFrameSelector",
        target_len=window_size,
        dense_window_size=c3_dense_window_size,
        selection_unit=1,
        scout_spatial_size=c3_scout_spatial_size,
        strategy="cadf_density_mesh_st",
        density_alpha=0.65,
        density_temperature=0.9,
        density_weights=dict(action=0.02, uncertainty=0.48, change=0.35, utility=0.15, boundary=0.0),
        density_entropy_floor=0.45,
        density_entropy_loss_weight=0.005,
        density_repulsion_loss_weight=0.0,
        density_uniform_floor=0.08,
        density_body_floor_weight=0.08,
        density_context_floor_weight=0.04,
        density_transition_smooth_radius=1,
        diagnostic_prediction_cap=422000,
        diagnostic_proposal_cap=2000,
        selected_index_aware_postprocess_enabled=False,
        physical_time_postprocess_enabled=False,
        max_gap_guard_count=12,
        st_local_radius=2,
        st_scale=0.5,
        actionness_loss_weight=0.05,
        boundary_loss_weight=0.0,
        scout=dict(
            type="PCOTMRASCADFDensityFrameScout",
            in_channels=3 * c3_scout_spatial_size * c3_scout_spatial_size,
            hidden_channels=128,
            num_layers=3,
            kernel_size=5,
            dropout=0.0,
            with_boundary_head=False,
        ),
    ),
)

work_dir = "exps/thumos/adatad/c3_cadf_densitymesh_original_adatad_32px_short_smoke"
