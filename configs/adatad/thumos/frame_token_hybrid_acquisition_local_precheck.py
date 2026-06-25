_base_ = ["./e2e_thumos_videomae_s_768x1_160_adapter.py"]


route_label = "DIVERGENT_INNOVATION_FRAME_TOKEN_HYBRID_DO_NOT_MERGE_WITH_C3"
route_id = "frame_token_hybrid_acquisition"
stage_id = "local_precheck"
variant_id = "Frame-Token-Hybrid-Acquisition-Local-Precheck"
window_size = 768
target_len = 384
target_dense_len = 768
scale_factor = 1

frame_token_hybrid_preview_probe = dict(
    type="FrameTokenHybridPreviewProbe",
    signal_meta_key="frame_token_hybrid_preview_signal",
    positions_meta_key="frame_token_hybrid_preview_positions",
    source_meta_key="frame_token_hybrid_preview_source",
)

frame_token_hybrid_meta_keys = [
    "video_name",
    "data_path",
    "fps",
    "duration",
    "snippet_stride",
    "window_start_frame",
    "resize_length",
    "window_size",
    "offset_frames",
    "frame_token_hybrid_preview_signal",
    "frame_token_hybrid_preview_positions",
    "frame_token_hybrid_preview_source",
]

experiment_scope = dict(
    route=route_id,
    stage=stage_id,
    variant_id=variant_id,
    route_label=route_label,
    changed_surface="input_sampling_token_compression_dense_completion",
    protocol=(
        "Preview/probe-visible metadata decides boundary and anchor raw frame "
        "observations; long stable temporal gaps are represented as span-token "
        "metadata; dense completion reconstructs the ActionFormer-compatible "
        "768-step axis after the current dense decoder."
    ),
    selection_surface="preview_probe_visible_pre_backbone_bridge",
    actual_decode_saving_in_current_pipeline=False,
    raw_decode_saving_claim_allowed=False,
    pre_decode_loader_hook_reviewed=False,
    requires_deploy_preview_probe_signal=True,
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
    allowed_decision="ALLOW_FRAME_TOKEN_HYBRID_PRECHECK_ONLY",
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

dataset = dict(
    train=dict(
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(
                type="LoadFrames",
                num_clips=1,
                method="random_trunc",
                trunc_len=window_size,
                trunc_thresh=0.75,
                crop_ratio=[0.9, 1.0],
                scale_factor=scale_factor,
            ),
            frame_token_hybrid_preview_probe,
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 182)),
            dict(type="mmaction.RandomResizedCrop"),
            dict(type="mmaction.Resize", scale=(160, 160), keep_ratio=False),
            dict(type="mmaction.Flip", flip_ratio=0.5),
            dict(type="mmaction.ImgAug", transforms="default"),
            dict(type="mmaction.ColorJitter"),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(
                type="Collect",
                inputs="imgs",
                keys=["masks", "gt_segments", "gt_labels"],
                meta_keys=frame_token_hybrid_meta_keys,
            ),
        ],
    ),
    val=dict(
        window_size=window_size,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="sliding_window", scale_factor=scale_factor),
            frame_token_hybrid_preview_probe,
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(
                type="Collect",
                inputs="imgs",
                keys=["masks", "gt_segments", "gt_labels"],
                meta_keys=frame_token_hybrid_meta_keys,
            ),
        ],
    ),
    test=dict(
        window_size=window_size,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="sliding_window", scale_factor=scale_factor),
            frame_token_hybrid_preview_probe,
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs"]),
            dict(type="Collect", inputs="imgs", keys=["masks"], meta_keys=frame_token_hybrid_meta_keys),
        ],
    ),
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
        require_preview_signal=True,
        preview_signal_meta_key="frame_token_hybrid_preview_signal",
        preview_positions_meta_key="frame_token_hybrid_preview_positions",
        preview_source_meta_key="frame_token_hybrid_preview_source",
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

work_dir = "exps/thumos/adatad/frame_token_hybrid_acquisition_local_precheck"
