_base_ = ["./c3_cadf_densitymesh_original_adatad_32px_full_train.py"]

c3_route_label = "C3_MAINLINE_OPTIMIZATION"
c3_route_labels = ["C3_MAINLINE_OPTIMIZATION", "C3_ORIGINAL_OPTIMIZATION_ROUTE"]
c3_claim_status = "diagnostic_only"
c3_full_train_claim_unlocked = False
c3_original_adatad_average_stride_backend_risk = "high"
c3_selected_index_aware_postprocess_contract = "metadata_only_default_off"
c3_physical_time_postprocess_enabled = False
c3_alpha_schedule_recoverable = False
c3_alpha_schedule_scope = "diagnostic_short_smoke_not_resumable"
c3_long_train_resume_claim_locked = True

workflow = dict(
    logging_interval=20,
    checkpoint_interval=2,
    val_loss_interval=-1,
    val_eval_interval=1,
    val_start_epoch=2,
    end_epoch=60,
    max_train_iters=None,
    disable_checkpoint=False,
)

model = dict(
    frame_selector=dict(
        density_alpha=0.65,
        density_alpha_schedule=dict(
            train_start_alpha=0.0,
            train_target_alpha=0.65,
            warmup_iters=3000,
            test_alpha="target",
        ),
        density_repulsion_loss_weight=0.0,
        max_gap_guard_count=12,
    ),
)

work_dir = "exps/thumos/adatad/c3_cadf_densitymesh_original_adatad_32px_full_train_candidate_diagnostic"
