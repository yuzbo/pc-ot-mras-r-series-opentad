_base_ = ["./c3_cadf_densitymesh_original_adatad_32px_alpha0_pure_uniform_stability_probe.py"]

c3_route_label = "C3_MAINLINE_OPTIMIZATION"
c3_route_labels = ["C3_MAINLINE_OPTIMIZATION", "C3_ORIGINAL_OPTIMIZATION_ROUTE"]
c3_claim_status = "diagnostic_only"
c3_full_train_claim_unlocked = False
c3_alpha0_combo_gate = "st_soft_path_plus_actionness_fp32_no_amp_no_ema"

model = dict(
    frame_selector=dict(
        density_alpha=0.0,
        density_entropy_loss_weight=0.0,
        density_repulsion_loss_weight=0.0,
        max_gap_guard_count=0,
        st_local_radius=2,
        st_scale=0.5,
        actionness_loss_weight=0.05,
    ),
)

work_dir = "exps/thumos/adatad/c3_cadf_densitymesh_original_adatad_32px_alpha0_st_actionness_combo_gate"
