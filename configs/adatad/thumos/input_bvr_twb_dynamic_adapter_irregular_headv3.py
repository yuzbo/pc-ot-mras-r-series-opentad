_base_ = ["./input_random_fixed_50pct_adapter_irregular_headv3_x.py"]

route_label = "DIVERGENT_INNOVATION_BVR_TWB_DO_NOT_MERGE_WITH_C3"

window_size = 192
dense_window_size = 384
scale_factor = 1
chunk_num = window_size * scale_factor // 16

dataset = dict(
    train=dict(
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(
                type="LoadFrames",
                num_clips=1,
                method="bvr_twb_dynamic_subsample",
                method_base="random_trunc",
                keep_ratio=0.5,
                remap_gt_to_selected_axis=False,
                target_len=window_size,
                source_len=dense_window_size,
                trunc_thresh=0.75,
                crop_ratio=[0.9, 1.0],
                scale_factor=scale_factor,
                bvr_twb_split="train",
                bvr_twb_min_keep=64,
                bvr_twb_max_keep=window_size,
                bvr_twb_max_gap=32,
                bvr_twb_scaffold_k=4,
                bvr_twb_train_value_labels=True,
                bvr_twb_feature_stride=2,
                bvr_twb_adapter_bridge_mode="adapter_fixed_length_padded_bridge",
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
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    val=dict(
        window_size=dense_window_size,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(
                type="LoadFrames",
                num_clips=1,
                method="bvr_twb_dynamic_subsample",
                method_base="sliding_window",
                keep_ratio=0.5,
                remap_gt_to_selected_axis=False,
                target_len=window_size,
                scale_factor=scale_factor,
                bvr_twb_split="val",
                bvr_twb_min_keep=64,
                bvr_twb_max_keep=window_size,
                bvr_twb_max_gap=32,
                bvr_twb_scaffold_k=4,
                bvr_twb_train_value_labels=False,
                bvr_twb_feature_stride=2,
                bvr_twb_adapter_bridge_mode="adapter_fixed_length_padded_bridge",
            ),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    test=dict(
        window_size=dense_window_size,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(
                type="LoadFrames",
                num_clips=1,
                method="bvr_twb_dynamic_subsample",
                method_base="sliding_window",
                keep_ratio=0.5,
                remap_gt_to_selected_axis=False,
                target_len=window_size,
                scale_factor=scale_factor,
                bvr_twb_split="test",
                bvr_twb_min_keep=64,
                bvr_twb_max_keep=window_size,
                bvr_twb_max_gap=32,
                bvr_twb_scaffold_k=4,
                bvr_twb_train_value_labels=False,
                bvr_twb_feature_stride=2,
                bvr_twb_adapter_bridge_mode="adapter_fixed_length_padded_bridge",
            ),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs"]),
            dict(type="Collect", inputs="imgs", keys=["masks"]),
        ],
    ),
)

model = dict(
    backbone=dict(
        backbone=dict(
            total_frames=window_size * scale_factor,
            use_irregular_time_embed=True,
            add_irregular_time_embed=False,
        ),
        custom=dict(
            _delete_=True,
            norm_eval=True,
            freeze_backbone=True,
            trainable_backbone_keywords=["adapter", "Adapter"],
            pre_processing_pipeline=[
                dict(type="Rearrange", keys=["frames"], ops="b n c (t1 t) h w -> (b t1) n c t h w", t1=chunk_num),
            ],
            post_processing_pipeline=[
                dict(type="Reduce", keys=["feats"], ops="b n c t h w -> b c t", reduction="mean"),
                dict(type="Rearrange", keys=["feats"], ops="(b t1) c t -> b c (t1 t)", t1=chunk_num),
            ],
        ),
    ),
    projection=dict(max_seq_len=window_size),
    rpn_head=dict(
        max_reg_log_distance=6.0,
        regression_head_fp32=True,
        regression_loss_fp32=True,
        filter_invalid_regression_samples=False,
        min_regression_segment_length=1e-6,
    ),
)

solver = dict(
    amp=False,
    fp16_compress=False,
    train=dict(batch_size=1, num_workers=2),
    val=dict(batch_size=1, num_workers=2),
    test=dict(batch_size=1, num_workers=2),
)

workflow = dict(
    checkpoint_interval=10,
    disable_checkpoint=False,
)

work_dir = "exps/thumos/adatad/input_bvr_twb_dynamic_adapter_irregular_headv3"
