import os

_base_ = ["./c3_indirect_original_adatad_32px_a_short_smoke.py"]

c3_route_label = "C3_MAINLINE_OPTIMIZATION"
c3_route_labels = ["C3_MAINLINE_OPTIMIZATION", "C3_ORIGINAL_OPTIMIZATION_ROUTE"]
c3_method = "C3-OracleShell-Indirect-CoarseScore-OriginalAdaTAD"
c3_method_purpose = (
    "replace manual oracle keep_positions with deployable coarse classifier scores; "
    "reuse oracle dense-window, frame/mask/remap/irregular-meta contract"
)
c3_claim_status = "precheck_only"
c3_no_test_gt_selection = True
c3_deploy_selection_uses_gt = False
c3_oracle_gt_diagnostic_only = True
c3_full_train_claim_unlocked = False
c3_coarse_score_cache_required = True
c3_coarse_score_cache_dir_env = "C3_COARSE_SCORE_CACHE_DIR"
coarse_score_cache_dir = os.environ.get("C3_COARSE_SCORE_CACHE_DIR", "REPLACE_WITH_C3_COARSE_SCORE_CACHE_DIR")

c3_dense_window_size = 768
window_size = 384
scale_factor = 1
chunk_num = window_size * scale_factor // 16
c3_oracle_shell_meta_keys = [
    "video_name",
    "data_path",
    "fps",
    "duration",
    "snippet_stride",
    "window_start_frame",
    "resize_length",
    "window_size",
    "offset_frames",
    "irregular_selected_positions",
    "irregular_selected_valid_len",
    "irregular_native_axis",
    "coarse_oracle_shell_score_source",
    "coarse_oracle_shell_uses_gt_for_selection",
    "coarse_oracle_shell_score_axis",
    "coarse_oracle_shell_selected_positions",
]

dataset = dict(
    train=dict(
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(
                type="LoadFrames",
                num_clips=1,
                method="coarse_score_oracle_shell_subsample",
                method_base="random_trunc",
                keep_ratio=0.5,
                target_len=window_size,
                source_len=c3_dense_window_size,
                trunc_thresh=0.75,
                crop_ratio=[0.9, 1.0],
                scale_factor=scale_factor,
                oracle_boundary_radius=2,
                coarse_score_cache_dir=coarse_score_cache_dir,
                coarse_score_min_action_score=0.35,
                coarse_score_transition_top_fraction=0.15,
                coarse_score_transition_radius=2,
                coarse_score_allow_missing=False,
                coarse_score_debug_fallback=False,
            ),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(
                type="Collect",
                inputs="imgs",
                keys=["masks", "gt_segments", "gt_labels"],
                meta_keys=c3_oracle_shell_meta_keys,
            ),
        ],
    ),
    val=dict(
        window_size=c3_dense_window_size,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(
                type="LoadFrames",
                num_clips=1,
                method="coarse_score_oracle_shell_subsample",
                method_base="sliding_window",
                keep_ratio=0.5,
                target_len=window_size,
                scale_factor=scale_factor,
                oracle_boundary_radius=2,
                coarse_score_cache_dir=coarse_score_cache_dir,
                coarse_score_allow_missing=False,
                coarse_score_debug_fallback=False,
            ),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(
                type="Collect",
                inputs="imgs",
                keys=["masks", "gt_segments", "gt_labels"],
                meta_keys=c3_oracle_shell_meta_keys,
            ),
        ],
    ),
    test=dict(
        window_size=c3_dense_window_size,
        test_mode=False,
        ioa_thresh=0.0,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(
                type="LoadFrames",
                num_clips=1,
                method="coarse_score_oracle_shell_subsample",
                method_base="sliding_window",
                keep_ratio=0.5,
                target_len=window_size,
                scale_factor=scale_factor,
                oracle_boundary_radius=2,
                coarse_score_cache_dir=coarse_score_cache_dir,
                coarse_score_allow_missing=False,
                coarse_score_debug_fallback=False,
            ),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs"]),
            dict(type="Collect", inputs="imgs", keys=["masks"], meta_keys=c3_oracle_shell_meta_keys),
        ],
    ),
)

model = dict(
    frame_selector=None,
    backbone=dict(backbone=dict(total_frames=window_size * scale_factor)),
    projection=dict(max_seq_len=window_size),
)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)

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

work_dir = "exps/thumos/adatad/c3_oracle_shell_indirect_precheck"
