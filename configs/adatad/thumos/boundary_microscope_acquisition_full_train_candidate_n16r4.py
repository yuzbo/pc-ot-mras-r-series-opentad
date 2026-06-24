_base_ = ["./boundary_microscope_acquisition_local_precheck.py"]


route_label = "DIVERGENT_INNOVATION_BOUNDARY_MICROSCOPE_DO_NOT_MERGE_WITH_C3"
route_id = "boundary_microscope_acquisition"
stage_id = "full_train_candidate_n16r4"
variant_id = "Boundary-Microscope-Acquisition-Full-Train-Candidate-N16R4"
target_len = 384
window_size = 768

experiment_scope = dict(
    route=route_id,
    stage=stage_id,
    variant_id=variant_id,
    route_label=route_label,
    changed_surface="input_sampling_dynamic_boundary_packet_acquisition",
    candidate_status="implementation_prepared_but_train_locked",
    strict_budget_family="up_to_384_selected_frames_from_768_dense_window",
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
    allowed_decision="ALLOW_BOUNDARY_MICROSCOPE_PRECHECK_ONLY",
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
    current_gate_cannot_authorize_remote_sync_or_full_train=True,
    future_full_train_requires_separate_decision=True,
    allowed_entrypoints=(),
    forbidden_entrypoints=("tools/train.py", "tools/test.py", "sbatch", "scp", "rsync"),
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

workflow = dict(
    logging_interval=50,
    checkpoint_interval=60,
    val_loss_interval=-1,
    val_eval_interval=2,
    val_start_epoch=40,
    end_epoch=60,
)

work_dir = "exps/thumos/adatad/boundary_microscope_acquisition_full_train_candidate_n16r4"
