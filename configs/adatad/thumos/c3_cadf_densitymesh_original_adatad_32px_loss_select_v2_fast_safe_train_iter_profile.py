_base_ = ["./c3_cadf_densitymesh_original_adatad_32px_loss_select_v2_fast_safe_formal.py"]

c3_loss_select_v2_scope = "fast_safe_train_iter_speed_profile_no_eval_no_final_claim"
c3_speed_profile = "train_iter_only_no_eval_no_checkpoint"
c3_claim_status = "diagnostic_only"
c3_full_train_claim_unlocked = False
launch_locked_until_user_unlock = False

workflow = dict(
    logging_interval=5,
    checkpoint_interval=1,
    val_loss_interval=-1,
    val_eval_interval=-1,
    val_start_epoch=999,
    val_eval_epochs=[],
    val_eval_interval_anchor_epoch=None,
    end_epoch=1,
    max_train_iters=50,
    disable_checkpoint=True,
    profile_train_iter_timing=dict(
        enabled=True,
        warmup_iters=2,
        log_interval=5,
        sync_cuda=True,
    ),
)

work_dir = "exps/thumos/adatad/c3_cadf_densitymesh_original_adatad_32px_loss_select_v2_fast_safe_train_iter_profile"
