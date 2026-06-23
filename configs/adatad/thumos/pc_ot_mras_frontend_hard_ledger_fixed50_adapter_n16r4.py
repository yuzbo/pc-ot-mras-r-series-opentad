_base_ = ["./e2e_thumos_videomae_s_768x1_160_adapter.py"]

import os


yuzibo_root = os.environ.get("YUZIBO_ROOT", os.path.expanduser("~/run/yuzibo"))
thumos14_root = os.path.join(yuzibo_root, "thumos14")
repo_root = os.environ.get(
    "OPENTAD_PCOTMRAS_CLEAN_ROOT",
    os.path.join(yuzibo_root, "OpenTAD_PCOTMRAS_R16_R18_R20_Clean_20260619_1730"),
)

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
pc_ot_mras_frontend_ledger_path = os.environ.get(
    "PC_OT_MRAS_FRONTEND_LEDGER_PATH",
    os.path.join(repo_root, "logs", "missing_pc_ot_mras_frontend_hard_ledger.jsonl"),
)

window_size = 384
dense_window_size = 768
scale_factor = 1
chunk_num = window_size * scale_factor // 16

experiment_scope = dict(
    route="pc_ot_mras_frontend_original_adatad",
    stage="eval_only_hard_ledger_fixed50",
    detector_stack="original_adatad_actionformer_adapter",
    changes_input_sampling=True,
    changes_detector_head=False,
    changes_neck=False,
    changes_loss_assignment=False,
    changes_post_processing=True,
    requires_pc_ot_mras_hard_ledger=True,
    deploy_claim_allowed=False,
    runtime_flops_claim_allowed=False,
    paper_claim_allowed=False,
)

pc_ot_mras_frontend_hard_ledger_eval_gate = dict(
    route="pc_ot_mras_frontend_original_adatad",
    stage="eval_only_hard_ledger_fixed50_locked_until_execution_gate",
    default_off=True,
    explicit_config_opt_in=True,
    eval_only=True,
    hard_frontend_ledger_required=True,
    allow_detector_training=True,
    requires_launch_gate=True,
    launch_gate_passed=False,
    allow_tools_train=False,
    allow_tools_test=False,
    allow_detector_map=False,
    allow_remote_sync=False,
    allow_precheck_only=False,
    allow_slurm=False,
    allow_gpu=False,
    metric_claim_allowed=False,
    paper_claim_allowed=False,
    runtime_flops_claim_allowed=False,
    deploy_claim_allowed=False,
    allowed_entrypoints=("tools/test.py",),
    allowed_checks=(
        "local_config_parse",
        "hard_ledger_schema_static_check",
        "selected_axis_postprocess_static_check",
        "subagent_readonly_review",
    ),
    forbidden_checks=(
        "tools_train",
        "direct_tools_test_without_execution_gate",
        "remote_sync",
        "slurm_or_gpu",
        "detector_map",
        "metric_claim",
        "paper_claim",
        "runtime_or_flops_claim",
        "deploy_claim",
        "raw_prediction_cache",
    ),
)

pc_ot_mras_frontend_meta_keys = [
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

pc_ot_mras_frontend_loadframes = dict(
    type="LoadFrames",
    num_clips=1,
    method="bata_value_transport_ledger_subsample",
    method_base="sliding_window",
    keep_ratio=0.5,
    target_len=window_size,
    scale_factor=scale_factor,
    remap_gt_to_selected_axis=True,
    bata_value_transport_ledger_path=pc_ot_mras_frontend_ledger_path,
    bata_value_transport_allow_missing_fallback=False,
    bata_value_transport_require_deployable=False,
    bata_value_transport_require_selected_count=window_size,
    bata_value_transport_source="pc_ot_mras_frontend_hard_positions",
    bata_value_transport_config_hash="pc_ot_mras_frontend_hard_ledger_fixed50_n16r4_eval_only",
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
                method="random_trunc",
                trunc_len=window_size,
                trunc_thresh=0.75,
                crop_ratio=[0.9, 1.0],
                scale_factor=scale_factor,
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
            pc_ot_mras_frontend_loadframes,
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"], meta_keys=pc_ot_mras_frontend_meta_keys),
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
            pc_ot_mras_frontend_loadframes,
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs"]),
            dict(type="Collect", inputs="imgs", keys=["masks"], meta_keys=pc_ot_mras_frontend_meta_keys),
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

evaluation = dict(
    ground_truth_filename=annotation_path,
)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)

work_dir = "exps/thumos/adatad/pc_ot_mras_frontend_hard_ledger_fixed50_eval_only_n16r4"
