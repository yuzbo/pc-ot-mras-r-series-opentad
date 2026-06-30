_base_ = ["./c3_indirect_original_adatad_32px_a_pqr_rankcal_v1_sparse_irregular_qc_v2_precheck.py"]

pqr_rankcal_v1 = dict(
    diagnostic_only=True,
    formal_fulltrain=False,
    user_override_fulltrain=False,
    claim_map_improvement=False,
    official_map_claim=False,
    remote_launch_locked=True,
    use_teacher=False,
    use_test_gt=False,
    use_raw_prediction_cache=False,
    physical_time_postprocess_claim=False,
    changed_surface="detector_head_sparse_irregular_quality_calibration_v2",
    experiment_boundary="adapter_actionformer_backend_ranking_calibration_only",
    backend_control="random_fixed_adapter_50pct",
    c3_selector_input_experiment=False,
    requires_c3_selector_tree_for_input_experiment=True,
    precheck_scope="config_validator_plus_quality_head_unit",
    build_only_status="pseudo_boundary_dependency_restored_pending_remote_runtime_smoke",
    build_only_blockers=(
        "pseudo_boundary restoration dependency remains checked by the inherited remote PRECHECK gate. "
        "QC V2 shortdiag is diagnostic-only and bounded for proposal/result dumps. "
        "It is not final route-quality evidence; no official mAP claim, no fulltrain, "
        "no teacher/test-GT/cache shortcut, and GPU1-only launch guard remains required."
    ),
)

solver = dict(
    train=dict(batch_size=1, num_workers=1),
    val=dict(batch_size=1, num_workers=1),
    test=dict(batch_size=1, num_workers=1),
    clip_grad_norm=1,
    amp=True,
    fp16_compress=True,
    static_graph=True,
    ema=False,
)

scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=1, max_epoch=2)

workflow = dict(
    logging_interval=20,
    checkpoint_interval=999,
    val_loss_interval=-1,
    val_eval_interval=1,
    val_start_epoch=0,
    end_epoch=2,
    max_train_iters=None,
    disable_checkpoint=True,
)

post_processing = dict(
    save_dict=True,
    qc_v2_diagnostic_dump=True,
)

work_dir = "exps/thumos/adatad/c3_pqr_rankcal_v1_sparse_irregular_qc_v2_shortdiag"
