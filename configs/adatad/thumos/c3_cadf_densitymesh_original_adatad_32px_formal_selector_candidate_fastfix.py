_base_ = ["./c3_cadf_densitymesh_original_adatad_32px_formal_selector_candidate_locked.py"]

c3_speed_fix = "selector_cpu_once_repair_diag_off_amp_withcp_probe"
c3_speed_fix_status = "candidate_requires_precheck_and_bounded_speed_smoke"
c3_full_train_claim_unlocked = False

solver = dict(
    train=dict(batch_size=2, num_workers=4),
    val=dict(batch_size=2, num_workers=4),
    test=dict(batch_size=2, num_workers=4),
    clip_grad_norm=1,
    amp=True,
    fp16_compress=True,
    static_graph=True,
    ema=False,
)

model = dict(
    frame_selector=dict(
        fast_cpu_selection=True,
        emit_selection_diagnostics=False,
        selection_diagnostics_interval=0,
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

work_dir = "exps/thumos/adatad/c3_cadf_densitymesh_original_adatad_32px_formal_selector_candidate_fastfix"
