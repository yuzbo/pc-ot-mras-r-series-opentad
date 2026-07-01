_base_ = ["./c3_oracle_shell_indirect_precheck.py"]

c3_claim_status = "formal_training_candidate_locked"
c3_full_train_claim_unlocked = False
c3_full_train_launch_locked_until_user_or_main_process = True
c3_full_train_unlock_env = "C3_ORACLE_SHELL_INDIRECT_FULLTRAIN_UNLOCK"
c3_expected_deployment_gpu = "N16R4_GPU1"
c3_validation_schedule = "epoch2_then_every5_no_frequent_validation"

solver = dict(
    train=dict(batch_size=2, num_workers=2),
    val=dict(batch_size=1, num_workers=1),
    test=dict(batch_size=1, num_workers=1),
    clip_grad_norm=1,
    amp=True,
    fp16_compress=True,
    static_graph=True,
    ema=False,
)

scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=60)

workflow = dict(
    logging_interval=10,
    checkpoint_interval=5,
    val_loss_interval=-1,
    val_eval_interval=5,
    val_start_epoch=2,
    val_eval_epochs=[2],
    val_eval_interval_anchor_epoch=2,
    end_epoch=60,
    max_train_iters=None,
    disable_checkpoint=False,
)

work_dir = "exps/thumos/adatad/c3_oracle_shell_indirect_full_train"
