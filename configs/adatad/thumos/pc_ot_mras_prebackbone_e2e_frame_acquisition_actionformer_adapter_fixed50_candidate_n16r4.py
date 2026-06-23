_base_ = ["./e2e_thumos_videomae_s_768x1_160_adapter.py"]

import os


yuzibo_root = os.environ.get("YUZIBO_ROOT", os.path.expanduser("~/run/yuzibo"))
thumos14_root = os.path.join(yuzibo_root, "thumos14")
pretrained_path = os.environ.get(
    "PC_OT_MRAS_PREBACKBONE_E2E_PRETRAINED_PATH",
    "pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth",
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

window_size = 384
dense_window_size = 768
scale_factor = 1
chunk_num = window_size * scale_factor // 16

experiment_scope = dict(
    route="pc_ot_mras_prebackbone_original_adatad",
    stage="prebackbone_e2e_frame_acquisition_actionformer_adapter_train_fixed50",
    detector_stack="original_adatad_actionformer_adapter",
    selection_surface="pre_backbone_raw_frame",
    selection_timing="online_before_backbone",
    acquisition_unit="frame_or_snippet",
    first_version_forward_contract="hard_top1_train_eval",
    budget_protocol="fixed384_frame_slot_candidate",
    s80r16_cell96x4_enabled=False,
    s80r16_cell96x4_note=(
        "Future route only: this config is not S80R16/cell96x4. "
        "It is the first fixed384 frame-slot candidate over a dense 768-frame window."
    ),
    train_protocol="joint_selector_detector_fixed50_prebackbone",
    reader_trainable=True,
    selector_gradient="hard_top1_st_detector_loss_and_train_gt_acquisition_aux",
    uses_offline_ledger=False,
    changes_input_sampling=True,
    changes_detector_head=False,
    changes_neck=False,
    changes_loss_assignment=False,
    changes_post_processing=True,
    changes_training_window_protocol=True,
    interpretation_boundary=(
        "True pre-backbone acquisition candidate: the frame_selector runs before VideoMAE, "
        "compresses a dense 768-frame/sliding window to 384 detector frames, remaps train GT "
        "to the selected axis, and leaves the original AdaTAD ActionFormer head intact. "
        "It is not the post-projection PCOTMRASDetectorBridge route."
    ),
    deploy_claim_allowed=False,
    runtime_flops_claim_allowed=False,
    paper_claim_allowed=False,
)

pc_ot_mras_prebackbone_e2e_acquisition_gate = dict(
    route="pc_ot_mras_prebackbone_original_adatad",
    stage="prebackbone_e2e_frame_acquisition_actionformer_adapter_candidate",
    default_off=False,
    explicit_config_opt_in=True,
    formal_train_candidate=True,
    selection_surface="pre_backbone_raw_frame",
    selection_timing="online_before_backbone",
    acquisition_unit="frame_or_snippet",
    full_joint_selector_detector_training=True,
    offline_ledger=False,
    post_projection_bridge=False,
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
    allow_prebackbone_frame_selector=True,
    allow_joint_selector_detector_training=True,
    allow_dataset_access=True,
    allow_pretrained_initialization=True,
    allow_checkpoint_write=True,
    allow_checkpoint_load=False,
    allow_resume=False,
    entrypoint_gate_context=dict(
        required=True,
        gate_json_env="OPENTAD_PCOTMRAS_PREBACKBONE_E2E_GATE_JSON",
        gate_sha256_env="OPENTAD_PCOTMRAS_PREBACKBONE_E2E_GATE_SHA256",
        active_manifest_sha256_env="OPENTAD_PCOTMRAS_PREBACKBONE_E2E_ACTIVE_MANIFEST_SHA256",
        resolved_config_sha256_env="OPENTAD_PCOTMRAS_PREBACKBONE_E2E_RESOLVED_CONFIG_SHA256",
        require_resolved_config_sha256=True,
        allowed_decisions=("ALLOW_PC_OT_MRAS_PREBACKBONE_E2E_FRAME_ACQUISITION_TRAIN_FIXED50",),
        strict_payload_validation=True,
        required_exact_values=dict(
            route="pc_ot_mras_prebackbone_original_adatad",
            execution_mode="train",
            selection_surface="pre_backbone_raw_frame",
            selection_timing="online_before_backbone",
            acquisition_unit="frame_or_snippet",
            claim_tier="no_deploy_until_runtime_audit",
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
            "allow_prebackbone_frame_selector",
            "allow_joint_selector_detector_training",
            "allow_tools_train",
            "allow_dataset_access",
            "allow_pretrained_initialization",
            "allow_checkpoint_write",
            "allow_train_validation_map",
            "allow_long_training",
            "reader_trainable",
        ),
        required_false_keys=(
            "uses_offline_ledger",
        ),
        unknown_key_policy="reject_unknown_except_explicit_harmless_metadata",
        harmless_metadata_keys=(
            "note",
            "review_id",
            "run_tag",
            "pretrained_sha256",
        ),
        sha256_file_bindings=(
            dict(
                gate_key="pretrained_sha256",
                path_env="PC_OT_MRAS_PREBACKBONE_E2E_PRETRAINED_PATH",
                label="pretrained",
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
            "offline_ledger",
            "allow_offline_ledger",
            "frozen_reader_ledger",
            "allow_frozen_reader_ledger",
            "post_projection_bridge",
            "allow_post_projection_bridge",
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
)


dataset = dict(
    train=dict(
        ann_file=annotation_path,
        class_map=class_map,
        data_path=train_data_path,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(
                type="LoadFrames",
                num_clips=1,
                method="random_trunc",
                trunc_len=dense_window_size,
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
        window_size=dense_window_size,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="sliding_window", scale_factor=scale_factor),
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
        window_size=dense_window_size,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="sliding_window", scale_factor=scale_factor),
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
    frame_selector=dict(
        type="PCOTMRASPreBackboneFrameSelector",
        target_len=window_size,
        dense_window_size=dense_window_size,
        descriptor_dim=12,
        protected_uniform_count=64,
        coverage_guard_count=64,
        straight_through_detector_loss=True,
        # First protocol version keeps train/eval distribution matched:
        # both paths use hard top-1 frame-slot transport.
        transport_topk=1,
        eval_transport_topk=1,
        remap_gt_to_selected_axis=True,
        aux_gt_acquisition_loss_weight=0.05,
        aux_value_loss_weight=0.01,
        aux_risk_loss_weight=0.01,
        aux_role_entropy_loss_weight=0.001,
        reader_regularizer_loss_weight=0.01,
        reader=dict(
            type="PCOTMRASReader",
            in_dim=12,
            hidden_dim=96,
            num_slots=window_size,
            num_blocks=3,
            kernel_size=5,
            dropout=0.10,
            enable_value_heads=True,
            column_cap=2.0,
            emit_pair_distribution=False,
        ),
    ),
    backbone=dict(
        backbone=dict(total_frames=window_size * scale_factor),
        custom=dict(
            pretrain=pretrained_path,
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

evaluation = dict(ground_truth_filename=annotation_path)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)

work_dir = "exps/thumos/adatad/pc_ot_mras_prebackbone_e2e_frame_acquisition_actionformer_adapter_fixed50_candidate_n16r4"
