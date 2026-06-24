_base_ = ["./e2e_thumos_videomae_s_768x1_160_adapter.py"]


route_label = "DIVERGENT_INNOVATION_BOUNDARY_MICROSCOPE_DO_NOT_MERGE_WITH_C3"
route_id = "boundary_microscope_acquisition"
stage_id = "local_precheck"
variant_id = "Boundary-Microscope-Acquisition-Local-Precheck"
window_size = 768
target_len = 384

experiment_scope = dict(
    route=route_id,
    stage=stage_id,
    variant_id=variant_id,
    route_label=route_label,
    changed_surface="input_sampling_dynamic_boundary_packet_acquisition",
    protocol=(
        "cheap global scanner estimates action and boundary hazards; dense "
        "microscope packets are built around start/end hazards; sparse anchors "
        "remain in interiors and background."
    ),
    deploy_time_inputs_only=True,
    test_time_gt_allowed=False,
    teacher_allowed=False,
    raw_prediction_cache_allowed=False,
    paper_claim_allowed=False,
)

boundary_microscope_gate = dict(
    route=route_id,
    stage=stage_id,
    route_label=route_label,
    requires_gate_json=True,
    allow_precheck_only=True,
    allow_tools_train=False,
    allow_tools_test=False,
    allow_remote_sync=False,
    allow_slurm=False,
    allow_gpu=False,
    allow_full_train=False,
    allow_raw_prediction=False,
    load_from_raw_predictions=False,
    save_raw_prediction=False,
    metric_claim_allowed=False,
    paper_claim_allowed=False,
    allowed_entrypoints=(),
    forbidden_entrypoints=("tools/train.py", "tools/test.py"),
)

model = dict(
    frame_selector=dict(
        type="BoundaryMicroscopeAcquisitionRoute",
        route_label=route_label,
        meta_key="boundary_microscope_acquisition_plan",
        target_len=target_len,
        dense_window_size=window_size,
        microscope_radius=3,
        microscope_stride=1,
        anchor_stride=24,
        max_dense_gap=8,
        max_start_hazards=4,
        max_end_hazards=4,
        action_threshold=0.15,
    )
)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(save_dict=False)
workflow = dict(
    logging_interval=50,
    checkpoint_interval=60,
    val_loss_interval=-1,
    val_eval_interval=2,
    val_start_epoch=40,
    end_epoch=60,
)

work_dir = "exps/thumos/adatad/boundary_microscope_acquisition_local_precheck"
