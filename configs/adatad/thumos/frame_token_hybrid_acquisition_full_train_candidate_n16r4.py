_base_ = ["./frame_token_hybrid_acquisition_local_precheck.py"]


route_label = "DIVERGENT_INNOVATION_FRAME_TOKEN_HYBRID_DO_NOT_MERGE_WITH_C3"
route_id = "frame_token_hybrid_acquisition"
stage_id = "full_train_candidate_n16r4"
variant_id = "Frame-Token-Hybrid-Acquisition-Full-Train-Candidate-N16R4"
window_size = 768
target_len = 384
target_dense_len = 768

experiment_scope = dict(
    route=route_id,
    stage=stage_id,
    variant_id=variant_id,
    route_label=route_label,
    changed_surface="input_sampling_token_compression_dense_completion",
    candidate_status="implementation_prepared_but_train_locked",
    strict_budget_family="up_to_384_raw_frame_observations_plus_span_tokens_from_768_dense_window",
    deploy_time_inputs_only=True,
    test_time_gt_allowed=False,
    teacher_allowed=False,
    oracle_allowed=False,
    raw_prediction_cache_allowed=False,
    paper_claim_allowed=False,
)

frame_token_hybrid_gate = dict(
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
    forbidden_entrypoints=("tools/train.py", "tools/test.py", "sbatch", "scp", "rsync"),
)

model = dict(
    frame_selector=dict(
        type="FrameTokenHybridAcquisitionRoute",
        route_label=route_label,
        meta_key="frame_token_hybrid_acquisition_plan",
        target_len=target_len,
        dense_window_size=window_size,
        target_dense_len=target_dense_len,
        anchor_stride=24,
        boundary_radius=2,
        boundary_epsilon=0.25,
        stable_gap_min_len=12,
        stable_epsilon=0.02,
        max_span_tokens=64,
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

work_dir = "exps/thumos/adatad/frame_token_hybrid_acquisition_full_train_candidate_n16r4"
