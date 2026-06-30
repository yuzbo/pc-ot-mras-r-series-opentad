_base_ = ["./c3_indirect_original_adatad_32px_a_pqr_rankcal_v1_precheck.py"]

route_variant = "C3_PQR_RankCalV1_SparseIrregularQCV2"

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
        "pseudo_boundary restoration dependency remains checked by the inherited precheck gate. "
        "QC V2 is diagnostic-only; run local validator/unit tests and remote PRECHECK only. "
        "No fulltrain, no official mAP claim, no teacher/test-GT/cache shortcut."
    ),
)

model = dict(
    rpn_head=dict(
        quality_head_cfg=dict(
            enabled=True,
            mode="sparse_irregular_qc_v2",
            kernel_size=3,
            target_mode="sparse_physical_iou_visibility",
            diagnostic_dump=True,
            geometry_conditioning=True,
            weight_init=0.0,
            geometry_weight_init=0.0,
            bias_init=4.59511985013459,
            loss_weight=0.03,
            score_alpha=0.10,
            positive_weight=1.0,
            negative_weight=1.0,
            loss_normalizer="valid",
        ),
    ),
)

post_processing = dict(
    save_dict=True,
    qc_v2_diagnostic_dump=True,
)

work_dir = "exps/thumos/adatad/c3_pqr_rankcal_v1_sparse_irregular_qc_v2_precheck"
