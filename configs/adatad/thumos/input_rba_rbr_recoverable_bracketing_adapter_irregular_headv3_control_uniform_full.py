_base_ = ["./input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3.py"]

route_label = "DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3"
rba_rbr_control_diagnostic_only = True
short_diagnostic_only = True
full_train_unlocked = False
no_metric_claim = True
no_runtime_claim = True
no_deploy_claim = True
no_paper_claim = True
no_sparse_compute_claim = True

window_size = 192
dense_window_size = 384
scale_factor = 1

_rba_control_common = dict(
    num_clips=1,
    method="rba_rbr_recoverable_bracketing",
    keep_ratio=0.5,
    remap_gt_to_selected_axis=False,
    target_len=window_size,
    scale_factor=scale_factor,
    rba_rbr_min_keep=64,
    rba_rbr_max_keep=window_size,
    rba_rbr_min_detector_feature_keep=32,
    rba_rbr_max_raw_gap=16,
    rba_rbr_max_detector_gap=24,
    rba_rbr_scaffold_k=4,
    rba_rbr_feature_stride=2,
    rba_rbr_adapter_bridge_mode="adapter_fixed_length_padded_bridge",
    rba_rbr_allow_diagnostic_preview_fallback=False,
    rba_rbr_scout_sample_count=32,
    rba_rbr_control_mode="uniform_raw",
    rba_rbr_control_keep=192,
)

_rba_meta_keys = [
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
    "rba_rbr_ledger",
    "rba_rbr_raw_selected_positions",
    "rba_rbr_raw_selected_valid_len",
    "rba_rbr_detector_feature_positions",
    "rba_rbr_detector_feature_valid_len",
    "rba_rbr_selected_positions",
    "rba_rbr_selected_valid_len",
    "rba_rbr_dense_valid_len",
    "rba_rbr_train_value_labels",
    "rba_rbr_candidate_count",
]

dataset = dict(
    train=dict(
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(
                type="LoadFrames",
                method_base="random_trunc",
                source_len=dense_window_size,
                trunc_thresh=0.75,
                crop_ratio=[0.9, 1.0],
                rba_rbr_split="train",
                rba_rbr_train_value_labels=False,
                **_rba_control_common,
            ),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 182)),
            dict(type="mmaction.RandomResizedCrop"),
            dict(type="mmaction.Resize", scale=(160, 160), keep_ratio=False),
            dict(type="mmaction.Flip", flip_ratio=0.5),
            dict(type="mmaction.ImgAug", transforms="default"),
            dict(type="mmaction.ColorJitter"),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"], meta_keys=_rba_meta_keys),
        ],
    ),
    val=dict(
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(
                type="LoadFrames",
                method_base="sliding_window",
                rba_rbr_split="val",
                rba_rbr_train_value_labels=False,
                **_rba_control_common,
            ),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"], meta_keys=_rba_meta_keys),
        ],
    ),
    test=dict(
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(
                type="LoadFrames",
                method_base="sliding_window",
                rba_rbr_split="test",
                rba_rbr_train_value_labels=False,
                **_rba_control_common,
            ),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs"]),
            dict(type="Collect", inputs="imgs", keys=["masks"], meta_keys=_rba_meta_keys),
        ],
    ),
)

workflow = dict(
    logging_interval=20,
    checkpoint_interval=999,
    val_loss_interval=-1,
    val_eval_interval=-1,
    val_start_epoch=9999,
    end_epoch=2,
    disable_checkpoint=True,
    runtime_debug_interval=20,
)

work_dir = "exps/thumos/adatad/input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3_control_uniform_full_shortdiag_only"
