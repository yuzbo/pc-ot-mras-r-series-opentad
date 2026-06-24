_base_ = ["./e2e_thumos_videomae_s_768x1_160_adapter.py"]

route_label = "C3_ORIGINAL_OPTIMIZATION_ROUTE"
route_family = "C3_MAINLINE_OPTIMIZATION"
candidate_name = "c3_physical_grid_actionformer_head_precheck"

annotation_path = "/root/autodl-tmp/annotations/thumos_14_anno.json"
class_map = "/root/autodl-tmp/annotations/category_idx.txt"
train_data_path = "/root/autodl-tmp/train"
test_data_path = "/root/autodl-tmp/test"

window_size = 384
dense_window_size = 768
scale_factor = 1

# Candidate scope:
# - Original AdaTAD backend change surface: detector head temporal geometry.
# - Selector/input sampling is unchanged random_fixed_subsample 50%.
# - No P2 head, raw prediction cache, teacher, test-GT, offline ledger, remote sync, Slurm, or full train.
protocol_flags = dict(
    precheck_only=True,
    changed_surface="head_temporal_geometry",
    selector_changed=False,
    dynamic_budget_changed=False,
    token_compression_changed=False,
    adapter_backbone_changed=False,
    detector_head_logic_changed=True,
    loss_assignment_changed=True,
    post_processing_changed=False,
    uses_p2_head=False,
    uses_raw_prediction_cache=False,
    uses_teacher=False,
    uses_test_gt=False,
    uses_offline_ledger=False,
    tools_test_allowed=False,
    tools_train_allowed=False,
    remote_sync_allowed=False,
    slurm_allowed=False,
    metric_claim_allowed=False,
    paper_claim_allowed=False,
)

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
                method="random_fixed_subsample",
                method_base="random_trunc",
                keep_ratio=0.5,
                target_len=window_size,
                source_len=dense_window_size,
                trunc_thresh=0.75,
                crop_ratio=[0.9, 1.0],
                scale_factor=scale_factor,
                remap_gt_to_selected_axis=False,
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
                method="random_fixed_subsample",
                method_base="sliding_window",
                keep_ratio=0.5,
                target_len=window_size,
                scale_factor=scale_factor,
                remap_gt_to_selected_axis=False,
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
                method="random_fixed_subsample",
                method_base="sliding_window",
                keep_ratio=0.5,
                target_len=window_size,
                scale_factor=scale_factor,
                remap_gt_to_selected_axis=False,
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
    type="ActionFormer",
    rpn_head=dict(
        type="ActionFormerHead",
        physical_grid_actionformer=dict(
            enabled=True,
            required=True,
            strict=True,
            eps=1.0e-6,
            diagnostic=dict(
                emit_score_iou_entry=True,
                emit_proposal_cap_entry=True,
                emit_selected_vs_physical_axis_entry=True,
            ),
        ),
        assignment_debug=dict(enabled=True),
    ),
)

workflow = dict(
    disable_checkpoint=True,
    checkpoint_interval=999,
)

work_dir = "exps/thumos/adatad/input_random_fixed_50pct_c3_physical_grid_actionformer_precheck"
