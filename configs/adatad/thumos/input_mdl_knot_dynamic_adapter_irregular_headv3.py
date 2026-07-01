import os

_base_ = ["./input_random_fixed_50pct_adapter_irregular_headv3_x.py"]

route_label = "DIVERGENT_INNOVATION_MDL_KNOT_DO_NOT_MERGE_WITH_C3"
route_status = "LOCAL_FINAL_CODE_CANDIDATE_PRECHECK_ONLY_AFTER_SAMPLED_RAW_EDGE_FIX_NO_METRIC_CLAIMS"
formal_train_unlocked = False
full_train_unlocked = False
sparse_compute_claim = False

thumos_root = os.environ.get("THUMOS_ROOT", "/data/home/sczc063/run/yuzibo/thumos14")
annotation_path = os.path.join(thumos_root, "annotations", "thumos_14_anno.json")
class_map = os.path.join(thumos_root, "annotations", "category_idx.txt")
train_data_path = os.path.join(thumos_root, "train")
test_data_path = os.path.join(thumos_root, "test")

window_size = 384
dense_window_size = 768
scale_factor = 1

mdl_knot_acquisition = dict(
    method="mdl_knot_dynamic_subsample",
    bridge="fixed_pad",
    fixed_pad_bridge_compute_boundary=dict(
        detector_input_len=window_size,
        dynamic_valid_k_is_measured=True,
        padding_counts_as_valid=False,
        sparse_compute_claim=False,
        statement="fixed_pad keeps the inherited Adapter tensor length at 384; this route currently measures acquisition behavior, not sparse compute",
    ),
    deploy_scout_source="raw_frame_motion_scout_with_metadata_fallback",
    real_scout_unavailable=False,
    synthetic_fallback_allowed=False,
    handoff_audit_mode="sampled_raw",
    scout_stride=8,
    scout_max_frames=96,
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
    handoff_audit_modes=dict(
        training_default="sampled_raw",
        realdiag_required_for_formal_readiness="full_raw",
        structural_only_does_not_unlock_formal_readiness=True,
    ),
    no_metric_claims=True,
    no_runtime_claims=True,
    no_deploy_claims=True,
    no_paper_claims=True,
    formal_train_unlocked=False,
    full_train_unlocked=False,
    sparse_compute_claim=False,
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
    mdl_knot_deploy_scout_source=mdl_knot_acquisition["deploy_scout_source"],
    mdl_knot_scout_stride=mdl_knot_acquisition["scout_stride"],
    mdl_knot_scout_max_frames=mdl_knot_acquisition["scout_max_frames"],
    mdl_knot_allow_synthetic_fallback=mdl_knot_acquisition["synthetic_fallback_allowed"],
    mdl_knot_no_gt_selector=True,
    mdl_knot_no_teacher=True,
    mdl_knot_no_prediction_cache=True,
    mdl_knot_no_dense_raw_backbone_handoff=True,
    mdl_knot_handoff_audit_mode=mdl_knot_acquisition["handoff_audit_mode"],
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
    mdl_knot_deploy_scout_source=mdl_knot_acquisition["deploy_scout_source"],
    mdl_knot_scout_stride=mdl_knot_acquisition["scout_stride"],
    mdl_knot_scout_max_frames=mdl_knot_acquisition["scout_max_frames"],
    mdl_knot_allow_synthetic_fallback=mdl_knot_acquisition["synthetic_fallback_allowed"],
    mdl_knot_no_gt_selector=True,
    mdl_knot_no_teacher=True,
    mdl_knot_no_prediction_cache=True,
    mdl_knot_no_dense_raw_backbone_handoff=True,
    mdl_knot_handoff_audit_mode=mdl_knot_acquisition["handoff_audit_mode"],
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

evaluation = dict(ground_truth_filename=annotation_path)

solver = dict(
    amp=True,
    fp16_compress=False,
    train=dict(batch_size=1, num_workers=2),
    val=dict(batch_size=1, num_workers=2),
    test=dict(batch_size=1, num_workers=2),
)

workflow = dict(
    logging_interval=50,
    checkpoint_interval=10,
    val_loss_interval=-1,
    val_eval_interval=2,
    val_start_epoch=40,
    end_epoch=60,
    disable_checkpoint=False,
)

work_dir = "exps/thumos/adatad/input_mdl_knot_dynamic_adapter_irregular_headv3_precheck_only"
