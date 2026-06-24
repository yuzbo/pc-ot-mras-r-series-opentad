_base_ = ["./ctf_bdi_pc_ot_mras_r35_actionformer_head_formal_train_candidate.py"]

import os


r35_pc_ot_mras_actionformer_head_gate = None

yuzibo_root = os.environ.get("YUZIBO_ROOT", os.path.expanduser("~/run/yuzibo"))
thumos14_root = os.path.join(yuzibo_root, "thumos14")
pretrained_path = os.environ.get(
    "PC_OT_MRAS_FRONTEND_JOINT_E2E_PRETRAINED_PATH",
    "pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth",
)

window_size = 384
dense_window_size = 768

experiment_scope = dict(
    route="pc_ot_mras_frontend_original_adatad",
    stage="joint_soft_transport_actionformer_adapter_train_fixed50",
    detector_stack="original_adatad_actionformer_adapter",
    train_protocol="full_joint_soft_transport_fixed50",
    reader_trainable=True,
    selector_gradient=True,
    uses_offline_ledger=False,
    frozen_reader_selector_only=False,
    changes_input_sampling=True,
    changes_detector_head=False,
    changes_neck=True,
    changes_loss_assignment=False,
    changes_post_processing=True,
    changes_training_window_protocol=False,
    interpretation_boundary=(
        "This is a full joint selector-detector soft-transport baseline. "
        "It keeps the original ActionFormerHead ranking/classification head but inserts "
        "a trainable PC-OT-MRAS reader and differentiable detector bridge before the head. "
        "R35 showed that original ActionFormerHead on a selected-token axis is high risk; "
        "this config is for Pro-reviewed diagnosis/launch discussion, not paper evidence."
    ),
    deploy_claim_allowed=False,
    runtime_flops_claim_allowed=False,
    paper_claim_allowed=False,
)

pc_ot_mras_frontend_joint_e2e_gate = dict(
    route="pc_ot_mras_frontend_original_adatad",
    stage="frontend_joint_soft_transport_actionformer_adapter_candidate",
    reviewed_predecessor="R35_actionformer_head_formal_train_and_frontend_frozen_reader_design",
    default_off=True,
    explicit_config_opt_in=True,
    formal_train_candidate=True,
    full_joint_selector_detector_training=True,
    frozen_reader_selector_only=False,
    offline_ledger=False,
    soft_transport=True,
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
    allow_soft_transport=True,
    allow_joint_reader_detector_training=True,
    allow_dataset_access=True,
    allow_pretrained_initialization=True,
    allow_checkpoint_write=True,
    allow_checkpoint_load=False,
    allow_resume=False,
    entrypoint_gate_context=dict(
        required=True,
        gate_json_env="OPENTAD_PCOTMRAS_FRONTEND_JOINT_E2E_GATE_JSON",
        gate_sha256_env="OPENTAD_PCOTMRAS_FRONTEND_JOINT_E2E_GATE_SHA256",
        active_manifest_sha256_env="OPENTAD_PCOTMRAS_FRONTEND_JOINT_E2E_ACTIVE_MANIFEST_SHA256",
        resolved_config_sha256_env="OPENTAD_PCOTMRAS_FRONTEND_JOINT_E2E_RESOLVED_CONFIG_SHA256",
        require_resolved_config_sha256=True,
        allowed_decisions=("ALLOW_PC_OT_MRAS_FRONTEND_JOINT_E2E_SOFT_TRANSPORT_TRAIN",),
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
            "allow_soft_transport",
            "allow_joint_reader_detector_training",
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
            "pretrained_sha256",
        ),
        sha256_file_bindings=(
            dict(
                gate_key="pretrained_sha256",
                path_env="PC_OT_MRAS_FRONTEND_JOINT_E2E_PRETRAINED_PATH",
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
        "no_offline_reader_ledger",
        "joint_reader_detector_soft_transport",
    ),
    forbidden_checks=(
        "direct_tools_test",
        "offline_ledger",
        "frozen_reader_ledger",
        "raw_prediction_cache",
        "checkpoint_load_or_resume",
        "runtime_or_flops_claim",
        "deploy_claim",
        "metric_claim",
        "paper_claim",
    ),
)

model = dict(
    projection=dict(pretrained=pretrained_path),
    pc_ot_mras_reader=dict(
        num_slots=window_size,
        emit_pair_distribution=False,
    ),
)

workflow = dict(
    logging_interval=50,
    checkpoint_interval=2,
    val_loss_interval=-1,
    val_eval_interval=2,
    val_start_epoch=40,
    end_epoch=60,
)

solver = dict(
    train=dict(batch_size=2, num_workers=2),
    val=dict(batch_size=2, num_workers=2),
    test=dict(batch_size=2, num_workers=2),
)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)

work_dir = "exps/thumos/adatad/pc_ot_mras_frontend_joint_soft_transport_actionformer_adapter_candidate_n16r4"
