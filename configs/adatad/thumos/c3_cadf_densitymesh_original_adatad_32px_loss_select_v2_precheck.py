_base_ = ["./c3_cadf_densitymesh_original_adatad_32px_short_smoke.py"]

c3_loss_select_v2 = True
c3_loss_select_v2_scope = "local_precheck_only"
c3_loss_select_v2_deploy_time_inputs = "scout_actionness_uncertainty_change_only"
c3_loss_select_v2_test_aux_source_leakage = "forbidden"
c3_loss_select_v2_user_unlock_evidence = "PENDING"
c3_claim_status = "precheck_only"
c3_full_train_claim_unlocked = False
c3_physical_time_postprocess_enabled = False

workflow = dict(
    logging_interval=1,
    checkpoint_interval=1,
    val_loss_interval=-1,
    val_eval_interval=-1,
    val_start_epoch=999,
    end_epoch=1,
    max_train_iters=2,
    disable_checkpoint=True,
)

solver = dict(
    train=dict(batch_size=1, num_workers=0),
    val=dict(batch_size=1, num_workers=0),
    test=dict(batch_size=1, num_workers=0),
    amp=False,
    fp16_compress=False,
    ema=False,
    nonfinite_loss_guard=dict(enabled=True, max_skips=0, max_consecutive_skips=0),
)

model = dict(
    frame_selector=dict(
        fast_cpu_selection=True,
        emit_selection_diagnostics=True,
        selection_diagnostics_interval=1,
        density_distribution_loss_weight=0.05,
        density_distribution_loss_weights=dict(
            smooth=0.15,
            local_cap=0.35,
            large_gap=0.45,
            collapse=0.30,
            target_kl=0.45,
        ),
        density_distribution_train_gt_target_weight=0.25,
        density_distribution_target_smooth_radius=1,
        density_distribution_loss_nan_guard=True,
        density_distribution_logit_clamp=20.0,
        physical_time_postprocess_enabled=False,
        selected_index_aware_postprocess_enabled=False,
    ),
)

work_dir = "exps/thumos/adatad/c3_cadf_densitymesh_original_adatad_32px_loss_select_v2_precheck"
