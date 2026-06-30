_base_ = ["./c3_cadf_densitymesh_original_adatad_32px_alpha0_backend_control.py"]

c3_route_label = "C3_MAINLINE_OPTIMIZATION"
c3_route_labels = ["C3_MAINLINE_OPTIMIZATION", "C3_ORIGINAL_OPTIMIZATION_ROUTE"]
c3_claim_status = "diagnostic_only"
c3_full_train_claim_unlocked = False
c3_alpha0_stability_probe = "disable_amp_fp16_aux_and_st_for_short_nan_repro"

solver = dict(
    train=dict(batch_size=1, num_workers=1),
    val=dict(batch_size=1, num_workers=1),
    test=dict(batch_size=1, num_workers=1),
    clip_grad_norm=0.5,
    amp=False,
    fp16_compress=False,
    static_graph=True,
    ema=False,
)

optimizer = dict(
    type="AdamW",
    lr=5e-5,
    weight_decay=0.05,
    paramwise=True,
    backbone=dict(
        lr=0,
        weight_decay=0,
        custom=[dict(name="adapter", lr=1e-4, weight_decay=0.05)],
        exclude=["backbone"],
    ),
)

scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=1, max_epoch=4)

workflow = dict(
    logging_interval=10,
    checkpoint_interval=99,
    val_loss_interval=-1,
    val_eval_interval=-1,
    val_start_epoch=99,
    end_epoch=4,
    max_train_iters=None,
    disable_checkpoint=True,
)

model = dict(
    frame_selector=dict(
        density_alpha=0.0,
        density_entropy_loss_weight=0.0,
        density_repulsion_loss_weight=0.0,
        max_gap_guard_count=0,
        st_local_radius=0,
        st_scale=0.0,
        actionness_loss_weight=0.0,
    ),
)

work_dir = "exps/thumos/adatad/c3_cadf_densitymesh_original_adatad_32px_alpha0_pure_uniform_stability_probe"
