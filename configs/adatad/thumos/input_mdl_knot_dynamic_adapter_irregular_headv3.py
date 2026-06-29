_base_ = ["./input_random_fixed_50pct_adapter_irregular_headv3_x.py"]

route_label = "DIVERGENT_INNOVATION_EVENT_SURPRISE_DO_NOT_MERGE_WITH_C3"
route_status = "LOCAL_PRECHECK_ONLY_NO_REMOTE_NO_TRAINING_NO_METRIC_CLAIMS"

annotation_path = "/root/autodl-tmp/annotations/thumos_14_anno.json"
class_map = "/root/autodl-tmp/annotations/category_idx.txt"
train_data_path = "/root/autodl-tmp/train"
test_data_path = "/root/autodl-tmp/test"

window_size = 384
dense_window_size = 768
scale_factor = 1

mdl_knot_acquisition = dict(
    method="mdl_knot_dynamic_subsample",
    bridge="fixed_pad",
    deploy_scout_source="fallback_synthetic_precheck_only",
    real_scout_unavailable=True,
    changed_surface=dict(
        input_sampling=True,
        dynamic_budget_policy=True,
        token_compression=False,
        adapter_backbone_internals=False,
        detector_head_logic=False,
        losses_assignment=False,
        test_time_postprocessing=False,
    ),
    route_label=route_label,
    scout_source="deploy_scout",
    min_k=4,
    max_k=window_size,
    target_weighted_error=0.02,
    max_gap=32,
    no_val_test_gt_selector=True,
    no_teacher=True,
    no_prediction_cache=True,
    no_dense_raw_backbone_handoff=True,
    no_metric_claims=True,
    no_runtime_claims=True,
    no_deploy_claims=True,
    no_paper_claims=True,
)

_mdl_load_train = dict(
    type="LoadFrames",
    num_clips=1,
    method="mdl_knot_dynamic_subsample",
    method_base="random_trunc",
    keep_ratio=0.5,
    remap_gt_to_selected_axis=False,
    target_len=window_size,
    source_len=dense_window_size,
    trunc_thresh=0.75,
    crop_ratio=[0.9, 1.0],
    scale_factor=scale_factor,
    mdl_knot_bridge=mdl_knot_acquisition["bridge"],
    mdl_knot_min_k=mdl_knot_acquisition["min_k"],
    mdl_knot_max_k=mdl_knot_acquisition["max_k"],
    mdl_knot_target_weighted_error=mdl_knot_acquisition["target_weighted_error"],
    mdl_knot_max_gap=mdl_knot_acquisition["max_gap"],
    mdl_knot_no_gt_selector=True,
    mdl_knot_no_teacher=True,
    mdl_knot_no_prediction_cache=True,
    mdl_knot_no_dense_raw_backbone_handoff=True,
)

_mdl_load_eval = dict(
    type="LoadFrames",
    num_clips=1,
    method="mdl_knot_dynamic_subsample",
    method_base="sliding_window",
    keep_ratio=0.5,
    remap_gt_to_selected_axis=False,
    target_len=window_size,
    scale_factor=scale_factor,
    mdl_knot_bridge=mdl_knot_acquisition["bridge"],
    mdl_knot_min_k=mdl_knot_acquisition["min_k"],
    mdl_knot_max_k=mdl_knot_acquisition["max_k"],
    mdl_knot_target_weighted_error=mdl_knot_acquisition["target_weighted_error"],
    mdl_knot_max_gap=mdl_knot_acquisition["max_gap"],
    mdl_knot_no_gt_selector=True,
    mdl_knot_no_teacher=True,
    mdl_knot_no_prediction_cache=True,
    mdl_knot_no_dense_raw_backbone_handoff=True,
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
            _mdl_load_train,
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
            _mdl_load_eval,
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
            _mdl_load_eval,
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs"]),
            dict(type="Collect", inputs="imgs", keys=["masks"]),
        ],
    ),
)

work_dir = "exps/thumos/adatad/input_mdl_knot_dynamic_adapter_irregular_headv3_locked_precheck_only"
