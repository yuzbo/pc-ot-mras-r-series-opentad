_base_ = ["./c3_cadf_densitymesh_original_adatad_32px_short_smoke.py"]

c3_route_label = "C3_MAINLINE_OPTIMIZATION"
c3_route_labels = ["C3_MAINLINE_OPTIMIZATION", "C3_ORIGINAL_OPTIMIZATION_ROUTE"]
c3_claim_status = "backend_control"
c3_full_train_claim_unlocked = False
c3_original_adatad_average_stride_backend_risk = "high"
c3_selected_index_aware_postprocess_contract = "metadata_only_default_off"
c3_physical_time_postprocess_enabled = False
c3_alpha0_backend_control_note = "diagnostic_only_density_alpha_zero_original_adatad_average_stride_backend"

solver = dict(
    train=dict(batch_size=2, num_workers=2),
    val=dict(batch_size=2, num_workers=2),
    test=dict(batch_size=2, num_workers=2),
    clip_grad_norm=1,
    amp=True,
    fp16_compress=True,
    static_graph=True,
    ema=True,
)

scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=1, max_epoch=60)

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
        density_alpha=0.0,
        density_entropy_loss_weight=0.0,
        density_repulsion_loss_weight=0.0,
        max_gap_guard_count=0,
    ),
)

work_dir = "exps/thumos/adatad/c3_cadf_densitymesh_original_adatad_32px_alpha0_backend_control"
