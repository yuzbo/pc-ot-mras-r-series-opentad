_base_ = ["./c3_indirect_original_adatad_32px_a_pqr_rankcal_v1_precheck.py"]

pqr_rankcal_v1 = dict(
    diagnostic_only=True,
    claim_map_improvement=False,
    official_map_claim=False,
    remote_launch_locked=True,
    use_teacher=False,
    use_test_gt=False,
    use_raw_prediction_cache=False,
    physical_time_postprocess_claim=False,
    changed_surface="detector_head_quality_ranking_calibration",
    experiment_boundary="adapter_actionformer_backend_ranking_calibration_only",
    backend_control="random_fixed_adapter_50pct",
    c3_selector_input_experiment=False,
    requires_c3_selector_tree_for_input_experiment=True,
    precheck_scope="config_validator_plus_quality_head_unit",
    build_only_status="pseudo_boundary_dependency_restored_pending_remote_runtime_smoke",
    build_only_blockers="No known pseudo_boundary dependency blocker after local restoration; remote PRECHECK and 2-iter smoke evidence still required",
)

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

scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=1, max_epoch=8)

workflow = dict(
    logging_interval=20,
    checkpoint_interval=2,
    val_loss_interval=-1,
    val_eval_interval=2,
    val_start_epoch=4,
    end_epoch=8,
    max_train_iters=None,
    disable_checkpoint=False,
)

work_dir = "exps/thumos/adatad/c3_pqr_rankcal_v1_adapter_backend_shortdiag"
