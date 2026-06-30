import os

_base_ = ["./input_random_fixed_50pct_adapter_irregular_headv3_x.py"]

route_label = "DIVERGENT_INNOVATION_BVR_TWB_DO_NOT_MERGE_WITH_C3"

window_size = 192
dense_window_size = 384
scale_factor = 1
chunk_num = window_size * scale_factor // 16

thumos_root = os.environ.get("THUMOS_ROOT", "/data/home/sczc063/run/yuzibo/thumos14")
annotation_path = os.path.join(thumos_root, "annotations", "thumos_14_anno.json")
class_map = os.path.join(thumos_root, "annotations", "category_idx.txt")
train_data_path = os.path.join(thumos_root, "train")
test_data_path = os.path.join(thumos_root, "test")

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
                bvr_twb_scout_source="deploy_visible_raw_or_metadata_scout",
                bvr_twb_require_deploy_visible_scout=True,
                bvr_twb_allow_diagnostic_preview_fallback=False,
                bvr_twb_scout_sample_count=32,
                bvr_twb_value_mode="deploy_heuristic_voi",
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
                bvr_twb_scout_source="deploy_visible_raw_or_metadata_scout",
                bvr_twb_require_deploy_visible_scout=True,
                bvr_twb_allow_diagnostic_preview_fallback=False,
                bvr_twb_scout_sample_count=32,
                bvr_twb_value_mode="deploy_heuristic_voi",
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
                bvr_twb_scout_source="deploy_visible_raw_or_metadata_scout",
                bvr_twb_require_deploy_visible_scout=True,
                bvr_twb_allow_diagnostic_preview_fallback=False,
                bvr_twb_scout_sample_count=32,
                bvr_twb_value_mode="deploy_heuristic_voi",
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

post_processing = dict(
    pre_nms_topk=512,
    bvr_twb_postprocess_guard=dict(
        enabled=True,
        require_bvr_meta=True,
        raw_proposal_cap=1024,
        per_class_topk=32,
        total_candidate_cap=512,
        min_score=0.001,
    ),
    nms=dict(max_seg_num=512),
)

workflow = dict(
    checkpoint_interval=10,
    disable_checkpoint=False,
)

work_dir = "exps/thumos/adatad/input_bvr_twb_dynamic_adapter_irregular_headv3"
