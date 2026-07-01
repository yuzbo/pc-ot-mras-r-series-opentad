_base_ = ["./e2e_thumos_videomae_s_768x1_160_adapter.py"]

import os


yuzibo_root = os.environ.get("YUZIBO_ROOT", os.path.expanduser("~/run/yuzibo"))
thumos14_root = os.path.join(yuzibo_root, "thumos14")

annotation_path = os.environ.get(
    "THUMOS14_ANNOTATION_PATH",
    os.path.join(thumos14_root, "annotations", "thumos_14_anno.json"),
)
class_map = os.environ.get(
    "THUMOS14_CLASS_MAP",
    os.path.join(thumos14_root, "annotations", "category_idx.txt"),
)
train_data_path = os.environ.get("THUMOS14_TRAIN_DATA_PATH", os.path.join(thumos14_root, "train"))
test_data_path = os.environ.get("THUMOS14_TEST_DATA_PATH", os.path.join(thumos14_root, "test"))

train_ledger_path = os.environ.get(
    "C3_ASFORMER_DELTA_TRAIN_LEDGER_PATH",
    "REPLACE_WITH_C3_ASFORMER_DELTA_TRAIN_LEDGER_PATH",
)
val_ledger_path = os.environ.get(
    "C3_ASFORMER_DELTA_VAL_LEDGER_PATH",
    "REPLACE_WITH_C3_ASFORMER_DELTA_VAL_LEDGER_PATH",
)
test_ledger_path = os.environ.get(
    "C3_ASFORMER_DELTA_TEST_LEDGER_PATH",
    "REPLACE_WITH_C3_ASFORMER_DELTA_TEST_LEDGER_PATH",
)

window_size = 384
dense_window_size = 768
scale_factor = 1
chunk_num = window_size * scale_factor // 16

experiment_scope = dict(
    route="C3_MAINLINE_OPTIMIZATION",
    route_variant="C3_ORIGINAL_OPTIMIZATION_ROUTE",
    stage="official_asformer_delta_ledger_original_adatad_full_train",
    selector_source="official_asformer_delta_p_action_64px",
    detector_stack="original_adatad_actionformer_adapter",
    changes_input_sampling=True,
    changes_detector_head=False,
    changes_neck=False,
    changes_loss_assignment=False,
    changes_post_processing=False,
    uses_offline_deploy_selection_ledger=True,
    oracle_shell_semantics="fixed_sliding_window_ledger_not_random_online_oracle_shell",
    deploy_claim_allowed=False,
    runtime_flops_claim_allowed=False,
    paper_claim_allowed=False,
)

c3_asformer_delta_ledger_full_train_gate = dict(
    route="C3_MAINLINE_OPTIMIZATION",
    route_variant="C3_ORIGINAL_OPTIMIZATION_ROUTE",
    stage="official_asformer_delta_ledger_original_adatad_full_train",
    default_off=True,
    explicit_config_opt_in=True,
    full_train_candidate=True,
    requires_launch_gate=True,
    launch_gate_passed=False,
    allow_tools_train=True,
    allow_tools_test=False,
    allow_detector_map=True,
    allow_remote_sync=True,
    allow_precheck_only=True,
    allow_slurm=True,
    allow_gpu=True,
    required_gpu="GPU1",
    allowed_entrypoints=("tools/train.py",),
    forbidden_entrypoints=("tools/test.py",),
    forbidden_routes=(
        "DIVERGENT_INNOVATION_BH_SDC",
        "DIVERGENT_INNOVATION_EVENT_SURPRISE",
        "DIVERGENT_INNOVATION_BOUNDARY_MICROSCOPE",
        "DIVERGENT_INNOVATION_FRAME_TOKEN_HYBRID",
        "P2",
        "PQR",
        "raw_prediction_cache",
        "teacher_at_test",
        "gt_at_test_selection",
    ),
    metric_claim_allowed=False,
    paper_claim_allowed=False,
    runtime_flops_claim_allowed=False,
    deploy_claim_allowed=False,
)

c3_asformer_delta_meta_keys = [
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
    "bata_score_source",
    "bata_diagnostic_only",
    "bata_selected_dense_indices",
    "bata_value_transport_selection_row",
    "bata_value_transport_config_hash",
]


def c3_asformer_delta_loadframes(ledger_path):
    return dict(
        type="LoadFrames",
        num_clips=1,
        method="bata_value_transport_ledger_subsample",
        method_base="sliding_window",
        keep_ratio=0.5,
        target_len=window_size,
        scale_factor=scale_factor,
        remap_gt_to_selected_axis=True,
        bata_value_transport_ledger_path=ledger_path,
        bata_value_transport_allow_missing_fallback=False,
        bata_value_transport_require_deployable=True,
        bata_value_transport_require_selected_count=window_size,
        bata_value_transport_allow_short_valid_ratio_count=True,
        bata_value_transport_source="c3_official_asformer_delta_p_action",
        bata_value_transport_config_hash="c3_official_asformer_delta_ledger_384_over_768_v1",
    )


dataset = dict(
    train=dict(
        type="ThumosSlidingDataset",
        ann_file=annotation_path,
        subset_name="training",
        class_map=class_map,
        data_path=train_data_path,
        filter_gt=False,
        feature_stride=4,
        sample_stride=1,
        window_size=dense_window_size,
        window_overlap_ratio=0.25,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            c3_asformer_delta_loadframes(train_ledger_path),
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
                meta_keys=c3_asformer_delta_meta_keys,
            ),
        ],
    ),
    val=dict(
        type="ThumosSlidingDataset",
        ann_file=annotation_path,
        subset_name="validation",
        class_map=class_map,
        data_path=test_data_path,
        filter_gt=False,
        feature_stride=4,
        sample_stride=1,
        window_size=dense_window_size,
        window_overlap_ratio=0.25,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            c3_asformer_delta_loadframes(val_ledger_path),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(
                type="Collect",
                inputs="imgs",
                keys=["masks", "gt_segments", "gt_labels"],
                meta_keys=c3_asformer_delta_meta_keys,
            ),
        ],
    ),
    test=dict(
        type="ThumosSlidingDataset",
        ann_file=annotation_path,
        subset_name="validation",
        class_map=class_map,
        data_path=test_data_path,
        filter_gt=False,
        test_mode=True,
        feature_stride=4,
        sample_stride=1,
        window_size=dense_window_size,
        window_overlap_ratio=0.5,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            c3_asformer_delta_loadframes(test_ledger_path),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs"]),
            dict(type="Collect", inputs="imgs", keys=["masks"], meta_keys=c3_asformer_delta_meta_keys),
        ],
    ),
)

model = dict(
    backbone=dict(
        backbone=dict(total_frames=window_size * scale_factor),
        custom=dict(
            pre_processing_pipeline=[
                dict(type="Rearrange", keys=["frames"], ops="b n c (t1 t) h w -> (b t1) n c t h w", t1=chunk_num),
            ],
            post_processing_pipeline=[
                dict(type="Reduce", keys=["feats"], ops="b n c t h w -> b c t", reduction="mean"),
                dict(type="Rearrange", keys=["feats"], ops="(b t1) c t -> b c (t1 t)", t1=chunk_num),
                dict(type="Interpolate", keys=["feats"], size=window_size),
            ],
        ),
    ),
    projection=dict(max_seq_len=window_size),
)

solver = dict(
    train=dict(batch_size=2, num_workers=2),
    val=dict(batch_size=1, num_workers=1),
    test=dict(batch_size=1, num_workers=1),
    clip_grad_norm=1,
    amp=True,
    fp16_compress=True,
    static_graph=True,
    ema=False,
)

optimizer = dict(
    type="AdamW",
    lr=1e-4,
    weight_decay=0.05,
    paramwise=True,
    backbone=dict(
        lr=0,
        weight_decay=0,
        custom=[dict(name="adapter", lr=2e-4, weight_decay=0.05)],
        exclude=["backbone"],
    ),
)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=60)

evaluation = dict(
    type="mAP",
    subset="validation",
    tiou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],
    ground_truth_filename=annotation_path,
)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)

workflow = dict(
    logging_interval=10,
    checkpoint_interval=5,
    val_loss_interval=-1,
    val_eval_interval=5,
    val_start_epoch=1,
    val_eval_epochs=[2, 60],
    val_eval_interval_anchor_epoch=2,
    end_epoch=60,
    max_train_iters=None,
    disable_checkpoint=False,
)

work_dir = "exps/thumos/adatad/c3_official_asformer_delta_ledger_original_adatad_full_train"
