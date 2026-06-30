_base_ = ["./c3_cadf_densitymesh_original_adatad_32px_loss_select_v2_shortdiag.py"]

c3_loss_select_v2_scope = "formal_fulltrain_candidate_locked_until_user_unlock"
c3_loss_select_v2_formal_candidate = True
c3_loss_select_v2_user_unlock_evidence = "PENDING"
c3_claim_status = "formal_selector_candidate_locked"
c3_full_train_claim_unlocked = False
launch_locked_until_user_unlock = True
c3_speed_fix = "selector_cpu_once_repair_diag_off_amp_withcp_probe"
c3_speed_fix_status = "preserved_fast_cpu_selection_diagnostics_interval_guard_no_amp_until_nan_gate_cleared"

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

solver = dict(
    train=dict(batch_size=2, num_workers=4),
    val=dict(batch_size=2, num_workers=4),
    test=dict(batch_size=2, num_workers=4),
    clip_grad_norm=1,
    amp=False,
    fp16_compress=False,
    static_graph=True,
    ema=False,
    nonfinite_loss_guard=dict(enabled=True, max_skips=0, max_consecutive_skips=0),
)

model = dict(
    frame_selector=dict(
        fast_cpu_selection=True,
        emit_selection_diagnostics=True,
        selection_diagnostics_interval=1,
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
)

work_dir = "exps/thumos/adatad/c3_cadf_densitymesh_original_adatad_32px_loss_select_v2_formal_candidate_locked"
