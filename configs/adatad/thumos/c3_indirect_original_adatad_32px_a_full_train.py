_base_ = ["./c3_indirect_original_adatad_32px_a_short_smoke.py"]

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

work_dir = "exps/thumos/adatad/c3_indirect_original_adatad_32px_a_full_train"
