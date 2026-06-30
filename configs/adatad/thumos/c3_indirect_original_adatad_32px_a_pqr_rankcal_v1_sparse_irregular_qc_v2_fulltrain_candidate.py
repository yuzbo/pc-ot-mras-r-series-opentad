_base_ = ["./c3_indirect_original_adatad_32px_a_pqr_rankcal_v1_sparse_irregular_qc_v2_precheck.py"]

route_variant = "C3_PQR_RankCalV1_SparseIrregularQCV2"

pqr_rankcal_v1 = dict(
    diagnostic_only=True,
    formal_fulltrain=False,
    formal_fulltrain_candidate=True,
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
        "pseudo_boundary restoration dependency remains checked by the inherited "
        "remote PRECHECK gate. QC V2 formal-fulltrain candidate is intentionally "
        "locked. It records the candidate schedule and dump settings for review, "
        "but cannot be launched as formal training without a same-turn user unlock "
        "and validator update. No teacher/test-GT/raw-cache shortcut and no "
        "official mAP claim."
    ),
)

solver = dict(
    train=dict(batch_size=2, num_workers=4),
    val=dict(batch_size=1, num_workers=4),
    test=dict(batch_size=1, num_workers=4),
    clip_grad_norm=1,
    amp=True,
    fp16_compress=True,
    static_graph=True,
    ema=False,
)

scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=60)

workflow = dict(
    logging_interval=20,
    checkpoint_interval=5,
    val_loss_interval=-1,
    val_eval_interval=2,
    val_start_epoch=40,
    end_epoch=60,
    max_train_iters=None,
    disable_checkpoint=False,
)

post_processing = dict(
    save_dict=True,
    qc_v2_diagnostic_dump=True,
)

work_dir = "exps/thumos/adatad/c3_pqr_rankcal_v1_sparse_irregular_qc_v2_fulltrain_candidate_locked"
