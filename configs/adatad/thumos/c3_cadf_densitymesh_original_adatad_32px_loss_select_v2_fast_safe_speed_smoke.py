_base_ = ["./c3_cadf_densitymesh_original_adatad_32px_loss_select_v2_formal_candidate_locked.py"]

c3_loss_select_v2_scope = "fast_safe_speed_smoke_no_final_claim"
c3_speed_fix = "loss_select_v2_fast_safe_amp_fp16_withcp_off_diag_off"
c3_speed_fix_status = "speed_smoke_only_requires_first_finite_loss_and_runtime_check"
c3_claim_status = "diagnostic_only"
c3_full_train_claim_unlocked = False
launch_locked_until_user_unlock = False

workflow = dict(
    logging_interval=5,
    checkpoint_interval=1,
    val_loss_interval=-1,
    val_eval_interval=-1,
    val_start_epoch=999,
    end_epoch=1,
    max_train_iters=20,
    disable_checkpoint=True,
)

solver = dict(
    train=dict(batch_size=2, num_workers=4),
    val=dict(batch_size=2, num_workers=4),
    test=dict(batch_size=2, num_workers=4),
    clip_grad_norm=1,
    amp=True,
    fp16_compress=True,
    static_graph=True,
    ema=False,
    nonfinite_loss_guard=dict(enabled=True, max_skips=0, max_consecutive_skips=0),
)

model = dict(
    frame_selector=dict(
        fast_cpu_selection=True,
        emit_selection_diagnostics=False,
        selection_diagnostics_interval=0,
        density_distribution_loss_weight=0.06,
        density_distribution_loss_weights=dict(
            smooth=0.10,
            local_cap=0.40,
            large_gap=0.55,
            collapse=0.35,
            target_kl=0.50,
        ),
        density_distribution_train_gt_target_weight=0.25,
        density_distribution_loss_nan_guard=True,
        density_distribution_logit_clamp=20.0,
        density_alpha=0.65,
        density_entropy_loss_weight=0.005,
        density_repulsion_loss_weight=0.0,
        max_gap_guard_count=12,
        st_local_radius=2,
        st_scale=0.5,
        actionness_loss_weight=0.05,
    ),
    backbone=dict(backbone=dict(with_cp=False)),
)

work_dir = "exps/thumos/adatad/c3_cadf_densitymesh_original_adatad_32px_loss_select_v2_fast_safe_speed_smoke"
