_base_ = ["./c3_cadf_densitymesh_original_adatad_32px_full_train.py"]

c3_route_label = "C3_MAINLINE_OPTIMIZATION"
c3_route_labels = ["C3_MAINLINE_OPTIMIZATION", "C3_ORIGINAL_OPTIMIZATION_ROUTE"]
c3_claim_status = "formal_selector_candidate_locked"
c3_formal_selector_candidate = True
c3_candidate_kind = "full_train_formal_selector_candidate_locked"
c3_full_train_claim_unlocked = False
c3_original_adatad_average_stride_backend_risk = "high"
c3_selected_index_aware_postprocess_contract = "metadata_only_default_off"
c3_physical_time_postprocess_enabled = False
c3_runtime_input_contract = "runtime_frames_only_no_external_guidance"

launch_locked_until_combo_pass = True
c3_combo_gate_required_config = (
    "configs/adatad/thumos/c3_cadf_densitymesh_original_adatad_32px_alpha0_st_actionness_combo_gate.py"
)
c3_combo_gate_required_status = "old_nan_window_pass_pending"
c3_combo_old_window_pass_evidence = "PENDING"
c3_combo_gate_remote_child = "1118197.376"

solver = dict(
    train=dict(batch_size=2, num_workers=2),
    val=dict(batch_size=2, num_workers=2),
    test=dict(batch_size=2, num_workers=2),
    clip_grad_norm=1,
    amp=False,
    fp16_compress=False,
    static_graph=True,
    ema=False,
)

model = dict(
    frame_selector=dict(
        density_alpha=0.65,
        density_entropy_loss_weight=0.005,
        density_repulsion_loss_weight=0.0,
        max_gap_guard_count=12,
        st_local_radius=2,
        st_scale=0.5,
        actionness_loss_weight=0.05,
    ),
)

work_dir = "exps/thumos/adatad/c3_cadf_densitymesh_original_adatad_32px_formal_selector_candidate_locked"
