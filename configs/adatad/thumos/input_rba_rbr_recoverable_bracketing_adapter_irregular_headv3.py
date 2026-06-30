_base_ = ["./input_random_fixed_50pct_adapter_irregular_headv3_x.py"]

route_label = "DIVERGENT_INNOVATION_RBA_RBR_DO_NOT_MERGE_WITH_C3"
full_train_unlocked = False
no_metric_claim = True
no_runtime_claim = True
no_deploy_claim = True
no_paper_claim = True

window_size = 192
dense_window_size = 384
scale_factor = 1
chunk_num = window_size * scale_factor // 16

thumos_root = "/data/home/sczc063/run/yuzibo/thumos14"
annotation_path = f"{thumos_root}/annotations/thumos_14_anno.json"
class_map = f"{thumos_root}/annotations/category_idx.txt"
train_data_path = f"{thumos_root}/train"
test_data_path = f"{thumos_root}/test"

_rba_common = dict(
    num_clips=1,
    method="rba_rbr_recoverable_bracketing",
    keep_ratio=0.5,
    remap_gt_to_selected_axis=False,
    target_len=window_size,
    scale_factor=scale_factor,
    rba_rbr_min_keep=64,
    rba_rbr_max_keep=window_size,
    rba_rbr_scaffold_k=4,
    rba_rbr_feature_stride=2,
    rba_rbr_adapter_bridge_mode="adapter_fixed_length_padded_bridge",
    rba_rbr_allow_diagnostic_preview_fallback=False,
    rba_rbr_scout_sample_count=32,
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
        ann_file=annotation_path,
        class_map=class_map,
        data_path=train_data_path,
        sample_stride=1,
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
                rba_rbr_train_value_labels=True,
                **_rba_common,
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
        ann_file=annotation_path,
        class_map=class_map,
        data_path=test_data_path,
        sample_stride=1,
        window_size=dense_window_size,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(
                type="LoadFrames",
                method_base="sliding_window",
                rba_rbr_split="val",
                rba_rbr_train_value_labels=False,
                **_rba_common,
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
        ann_file=annotation_path,
        class_map=class_map,
        data_path=test_data_path,
        sample_stride=1,
        window_size=dense_window_size,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(
                type="LoadFrames",
                method_base="sliding_window",
                rba_rbr_split="test",
                rba_rbr_train_value_labels=False,
                **_rba_common,
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

evaluation = dict(
    ground_truth_filename=annotation_path,
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
            pretrain="pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth",
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

work_dir = "exps/thumos/adatad/input_rba_rbr_recoverable_bracketing_adapter_irregular_headv3_local_locked"
