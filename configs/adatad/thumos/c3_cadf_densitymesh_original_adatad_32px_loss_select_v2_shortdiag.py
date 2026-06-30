_base_ = ["./c3_cadf_densitymesh_original_adatad_32px_loss_select_v2_precheck.py"]

c3_loss_select_v2_scope = "local_shortdiag_candidate_no_final_claim"
c3_claim_status = "diagnostic_only"
c3_full_train_claim_unlocked = False

workflow = dict(
    logging_interval=20,
    checkpoint_interval=1,
    val_loss_interval=-1,
    val_eval_interval=-1,
    val_start_epoch=999,
    end_epoch=4,
    max_train_iters=None,
    disable_checkpoint=True,
)

solver = dict(
    train=dict(batch_size=2, num_workers=4),
    val=dict(batch_size=2, num_workers=4),
    test=dict(batch_size=2, num_workers=4),
    clip_grad_norm=1,
    amp=False,
    fp16_compress=False,
    static_graph=False,
    ema=False,
    nonfinite_loss_guard=dict(enabled=True, max_skips=0, max_consecutive_skips=0),
)

model = dict(
    frame_selector=dict(
        density_distribution_loss_weight=0.06,
        density_distribution_loss_weights=dict(
            smooth=0.10,
            local_cap=0.40,
            large_gap=0.55,
            collapse=0.35,
            target_kl=0.50,
        ),
        density_distribution_train_gt_target_weight=0.25,
        actionness_loss_weight=0.05,
        st_local_radius=2,
        st_scale=0.5,
    ),
)

work_dir = "exps/thumos/adatad/c3_cadf_densitymesh_original_adatad_32px_loss_select_v2_shortdiag"
