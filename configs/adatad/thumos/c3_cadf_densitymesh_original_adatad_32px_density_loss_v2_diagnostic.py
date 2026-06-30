_base_ = ["./c3_cadf_densitymesh_original_adatad_32px_short_smoke.py"]

c3_route_label = "C3_MAINLINE_OPTIMIZATION"
c3_route_labels = ["C3_MAINLINE_OPTIMIZATION", "C3_ORIGINAL_OPTIMIZATION_ROUTE"]
c3_claim_status = "diagnostic_only"
c3_full_train_claim_unlocked = False
c3_density_loss_v2_diagnostic_only = True
c3_density_loss_v2_claim_unlocked = False
c3_density_loss_v2_protocol = "local_diagnostic_only_no_full_train_no_map_no_deploy_no_paper_claim"
c3_physical_time_postprocess_enabled = False

workflow = dict(
    logging_interval=1,
    checkpoint_interval=1,
    val_loss_interval=-1,
    val_eval_interval=-1,
    val_start_epoch=999,
    end_epoch=1,
    max_train_iters=2,
    disable_checkpoint=True,
)

model = dict(
    frame_selector=dict(
        density_window_mass_loss_weight=0.01,
        density_max_gap_loss_weight=0.01,
        density_blue_noise_loss_weight=0.005,
        density_weak_target_loss_weight=0.0,
        density_repulsion_loss_weight=0.0,
        physical_time_postprocess_enabled=False,
        selected_index_aware_postprocess_enabled=False,
    ),
)

work_dir = "exps/thumos/adatad/c3_cadf_densitymesh_original_adatad_32px_density_loss_v2_diagnostic"
