_base_ = ["./input_random_fixed_50pct_adapter.py"]

route_label = "C3_MAINLINE_OPTIMIZATION"
route_family = "C3_ORIGINAL_OPTIMIZATION_ROUTE"
route_variant = "C3_PQR_RankCalV1_MaxIoU"

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
    build_only_status="locked_by_baseline_import_dependencies",
    build_only_blockers="Rearrange transform registration missing before dataset build; opentad.datasets.transforms.pseudo_boundary missing in clean snapshot",
)

model = dict(
    rpn_head=dict(
        quality_head_cfg=dict(
            enabled=True,
            kernel_size=3,
            target_mode="max_iou",
            weight_init=0.0,
            bias_init=4.59511985013459,
            loss_weight=0.03,
            score_alpha=0.10,
            positive_weight=1.0,
            negative_weight=1.0,
            loss_normalizer="valid",
        ),
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

scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=1, max_epoch=1)

workflow = dict(
    logging_interval=1,
    checkpoint_interval=99,
    val_loss_interval=-1,
    val_eval_interval=-1,
    val_start_epoch=99,
    end_epoch=1,
    max_train_iters=2,
    disable_checkpoint=True,
)

work_dir = "exps/thumos/adatad/c3_pqr_rankcal_v1_adapter_backend_precheck"
