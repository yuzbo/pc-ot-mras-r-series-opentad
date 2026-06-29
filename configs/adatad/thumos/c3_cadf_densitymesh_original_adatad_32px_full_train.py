_base_ = ["./c3_cadf_densitymesh_original_adatad_32px_short_smoke.py"]

c3_route_label = "C3_MAINLINE_OPTIMIZATION"
c3_route_labels = ["C3_MAINLINE_OPTIMIZATION", "C3_ORIGINAL_OPTIMIZATION_ROUTE"]

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
    val_eval_interval=2,
    val_start_epoch=40,
    end_epoch=60,
    max_train_iters=None,
    disable_checkpoint=False,
)

model = dict(frame_selector=dict(density_repulsion_loss_weight=0.0))

work_dir = "exps/thumos/adatad/c3_cadf_densitymesh_original_adatad_32px_full_train"
