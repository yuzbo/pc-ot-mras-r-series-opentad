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

pc_ot_mras_frontend_train_ledger_path = os.environ.get(
    "PC_OT_MRAS_FRONTEND_TRAIN_LEDGER_PATH",
    os.path.join(repo_root, "logs", "missing_pc_ot_mras_frontend_train_ledger.jsonl"),
)
pc_ot_mras_frontend_val_ledger_path = os.environ.get(
    "PC_OT_MRAS_FRONTEND_VAL_LEDGER_PATH",
    os.path.join(repo_root, "logs", "missing_pc_ot_mras_frontend_val_ledger.jsonl"),
)
pc_ot_mras_frontend_test_ledger_path = os.environ.get(
    "PC_OT_MRAS_FRONTEND_TEST_LEDGER_PATH",
    os.path.join(repo_root, "logs", "missing_pc_ot_mras_frontend_test_ledger.jsonl"),
)

window_size = 384
dense_window_size = 768
scale_factor = 1
chunk_num = window_size * scale_factor // 16

experiment_scope = dict(
    route="pc_ot_mras_frontend_original_adatad",
    stage="r18_frozen_reader_adatad_retrain_fixed50_sliding",
    detector_stack="original_adatad_actionformer_adapter",
    source_reader="R18_frozen_reader_checkpoint",
    train_protocol="fixed_sliding_window_reader_ledger_retrain",
    changes_input_sampling=True,
    changes_detector_head=False,
    changes_neck=False,
    changes_loss_assignment=False,
    changes_post_processing=True,
    changes_training_window_protocol=True,
    reader_trainable=False,
    selector_gradient=False,
    requires_pc_ot_mras_train_val_test_ledgers=True,
    deploy_claim_allowed=False,
    runtime_flops_claim_allowed=False,
    paper_claim_allowed=False,
)

pc_ot_mras_frontend_retrain_gate = dict(
    route="pc_ot_mras_frontend_original_adatad",
    stage="r18_frozen_reader_adatad_retrain_candidate",
    reviewed_predecessor="frozen_reader_eval_chain_ecb9692",
    default_off=False,
    explicit_config_opt_in=True,
    formal_train_candidate=True,
    frozen_reader_selector_only=True,
    allow_detector_training=True,
    requires_launch_gate=True,
    launch_gate_passed=True,
    allow_remote_sync=False,
    allow_precheck_only=False,
    allow_slurm=False,
    allow_gpu=False,
    allow_tools_train=True,
    allow_tools_test=False,
    allow_detector_map=False,
    allow_train_validation_map=True,
    allow_long_training=True,
    allow_reader_dump=True,
    allow_hard_position_export=True,
    allow_value_transport_ledger=True,
    allow_dataset_access=True,
    allow_pretrained_initialization=True,
    allow_checkpoint_write=True,
    allow_checkpoint_load=False,
    allow_resume=False,
    entrypoint_gate_context=dict(
        required=True,
        gate_json_env="OPENTAD_PCOTMRAS_FRONTEND_RETRAIN_GATE_JSON",
        gate_sha256_env="OPENTAD_PCOTMRAS_FRONTEND_RETRAIN_GATE_SHA256",
        active_manifest_sha256_env="OPENTAD_PCOTMRAS_FRONTEND_RETRAIN_ACTIVE_MANIFEST_SHA256",
        resolved_config_sha256_env="OPENTAD_PCOTMRAS_FRONTEND_RETRAIN_RESOLVED_CONFIG_SHA256",
        require_resolved_config_sha256=True,
        allowed_decisions=("ALLOW_PC_OT_MRAS_FRONTEND_R18_FROZEN_READER_ADATAD_RETRAIN",),
        strict_payload_validation=True,
        required_exact_values=dict(
            route="pc_ot_mras_frontend_original_adatad",
            budget=384,
            dense_window_size=768,
            max_epochs=60,
            val_start_epoch=40,
            val_eval_interval=2,
        ),
        required_true_keys=(
            "allow_slurm",
            "allow_gpu",
            "single_gpu",
            "allow_reader_dump",
            "allow_hard_position_export",
            "allow_value_transport_ledger",
            "allow_tools_train",
            "allow_dataset_access",
            "allow_pretrained_initialization",
            "allow_checkpoint_write",
            "allow_train_validation_map",
            "allow_long_training",
        ),
        unknown_key_policy="reject_unknown_except_explicit_harmless_metadata",
        harmless_metadata_keys=(
            "note",
            "review_id",
            "run_tag",
            "pc_ot_mras_checkpoint_sha256",
            "pretrained_sha256",
            "train_ledger_sha256",
            "val_ledger_sha256",
            "test_ledger_sha256",
        ),
        sha256_file_bindings=(
            dict(
                gate_key="pc_ot_mras_checkpoint_sha256",
                path_env="PC_OT_MRAS_FRONTEND_RETRAIN_PCOT_CKPT_PATH",
                label="pc_ot_mras checkpoint",
            ),
            dict(
                gate_key="pretrained_sha256",
                path_env="PC_OT_MRAS_FRONTEND_RETRAIN_PRETRAINED_PATH",
                label="pretrained",
            ),
            dict(
                gate_key="train_ledger_sha256",
                path_env="PC_OT_MRAS_FRONTEND_TRAIN_LEDGER_PATH",
                label="train ledger",
            ),
            dict(
                gate_key="val_ledger_sha256",
                path_env="PC_OT_MRAS_FRONTEND_VAL_LEDGER_PATH",
                label="val ledger",
            ),
            dict(
                gate_key="test_ledger_sha256",
                path_env="PC_OT_MRAS_FRONTEND_TEST_LEDGER_PATH",
                label="test ledger",
            ),
        ),
        forbidden_true_keys=(
            "tools_test",
            "allow_tools_test",
            "direct_tools_test",
            "detector_map",
            "allow_detector_map",
            "formal_eval",
            "allow_formal_eval",
            "checkpoint_load",
            "allow_checkpoint_load",
            "resume",
            "allow_resume",
            "load_from",
            "allow_load_from",
            "raw_prediction",
            "allow_raw_prediction",
            "raw_predictions",
            "allow_raw_predictions",
            "raw_prediction_cache",
            "allow_raw_prediction_cache",
            "prediction_cache",
            "allow_prediction_cache",
            "load_from_raw_predictions",
            "allow_load_from_raw_predictions",
            "save_raw_prediction",
            "allow_save_raw_prediction",
            "save_raw_predictions",
            "allow_save_raw_predictions",
            "uses_gt",
            "uses_teacher",
            "uses_oracle",
            "uses_raw_prediction",
            "metric_claim",
            "allow_metric_claim",
            "metric_claim_allowed",
            "paper_claim",
            "allow_paper_claim",
            "paper_claim_allowed",
            "runtime_claim",
            "allow_runtime_claim",
            "flops_claim",
            "allow_flops_claim",
            "runtime_flops_claim",
            "allow_runtime_flops_claim",
            "runtime_flops_claim_allowed",
            "deploy_claim",
            "allow_deploy_claim",
            "deploy_claim_allowed",
        ),
    ),
    metric_claim_allowed=False,
    paper_claim_allowed=False,
    runtime_flops_claim_allowed=False,
    deploy_claim_allowed=False,
    allowed_entrypoints=("tools/train.py",),
    allowed_checks=(
        "slurm_single_gpu_formal_train",
        "train_forward_backward_finite_loss",
        "train_time_validation_map",
        "frozen_reader_train_val_test_ledgers",
    ),
    forbidden_checks=(
        "direct_tools_test",
        "raw_prediction_cache",
        "checkpoint_load_or_resume",
        "runtime_or_flops_claim",
        "deploy_claim",
        "metric_claim",
        "paper_claim",
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


def _frontend_loadframes(ledger_path, split):
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
        bata_value_transport_source="pc_ot_mras_frontend_hard_positions",
        bata_value_transport_config_hash=f"pc_ot_mras_frontend_r18_frozen_reader_retrain_{split}_fixed50_n16r4",
    )


dataset = dict(
    train=dict(
        type="ThumosSlidingDataset",
        ann_file=annotation_path,
        subset_name="training",
        block_list=None,
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
            _frontend_loadframes(pc_ot_mras_frontend_train_ledger_path, "train"),
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
                meta_keys=pc_ot_mras_frontend_meta_keys,
            ),
        ],
    ),
    val=dict(
        type="ThumosSlidingDataset",
        ann_file=annotation_path,
        subset_name="validation",
        block_list=None,
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
            _frontend_loadframes(pc_ot_mras_frontend_val_ledger_path, "val"),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(
                type="Collect",
                inputs="imgs",
                keys=["masks", "gt_segments", "gt_labels"],
                meta_keys=pc_ot_mras_frontend_meta_keys,
            ),
        ],
    ),
    test=dict(
        type="ThumosSlidingDataset",
        ann_file=annotation_path,
        subset_name="validation",
        block_list=None,
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
            _frontend_loadframes(pc_ot_mras_frontend_test_ledger_path, "test"),
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

solver = dict(
    train=dict(batch_size=2, num_workers=2),
    val=dict(batch_size=2, num_workers=2),
    test=dict(batch_size=2, num_workers=2),
)

workflow = dict(
    logging_interval=50,
    checkpoint_interval=2,
    val_loss_interval=-1,
    val_eval_interval=2,
    val_start_epoch=40,
    end_epoch=60,
)

evaluation = dict(
    ground_truth_filename=annotation_path,
)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)

work_dir = "exps/thumos/adatad/pc_ot_mras_frontend_r18_frozen_reader_adapter_retrain_candidate_n16r4"
