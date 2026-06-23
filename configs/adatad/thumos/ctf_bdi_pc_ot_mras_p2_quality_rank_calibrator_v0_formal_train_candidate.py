_base_ = ["ctf_bdi_pc_ot_mras_p2_quality_rank_calibrator_v0_local.py"]

# Formal training candidate for the P2 QualityRank route after D6-D8 narrowed
# the R17/R18 failure toward score/rank calibration. This opens only a gated
# tools/train.py path with normal train-time validation; direct tools/test.py,
# raw-prediction caches, runtime/deploy claims, and paper claims remain closed.

p2_quality_rank_calibrator_v0_gate = None

p2_quality_rank_calibrator_v0_formal_train_gate = dict(
    route="CTF-BDI/PC-OT-MRAS",
    stage="P2_NIIQ_QualityRank_Calibrator_v0_formal_train_candidate",
    reviewed_predecessor=(
        "D6_D7_D8_closeout_and_P2QR_short_smoke_PASS_20260623"
    ),
    default_off=False,
    explicit_config_opt_in=True,
    formal_train_candidate=True,
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
    allow_dataset_access=True,
    allow_pretrained_initialization=True,
    allow_checkpoint_write=True,
    allow_checkpoint_load=False,
    allow_resume=False,
    entrypoint_gate_context=dict(
        required=True,
        gate_json_env="OPENTAD_PCOTMRAS_ENTRYPOINT_GATE_JSON",
        gate_sha256_env="OPENTAD_PCOTMRAS_ENTRYPOINT_GATE_SHA256",
        active_manifest_sha256_env="OPENTAD_PCOTMRAS_ACTIVE_MANIFEST_SHA256",
        resolved_config_sha256_env="OPENTAD_PCOTMRAS_RESOLVED_CONFIG_SHA256",
        require_resolved_config_sha256=True,
        allowed_decisions=("ALLOW_P2QR_FORMAL_TRAIN",),
        strict_payload_validation=True,
        required_exact_values=dict(
            max_epochs=60,
            val_start_epoch=40,
            val_eval_interval=2,
        ),
        required_true_keys=(
            "allow_slurm",
            "allow_gpu",
            "allow_tools_train",
            "single_gpu",
            "allow_dataset_access",
            "allow_pretrained_initialization",
            "allow_checkpoint_write",
            "allow_train_validation_map",
            "allow_long_training",
        ),
        unknown_key_policy="reject_unknown_except_explicit_harmless_metadata",
        harmless_metadata_keys=("note", "review_id"),
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
    dynamic_budget_claim_allowed=False,
    scanner_quality_claim_allowed=False,
    allowed_entrypoints=("tools/train.py",),
    allowed_checks=(
        "slurm_single_gpu_formal_train",
        "train_forward_backward_finite_loss",
        "train_time_validation_map",
        "p2_quality_calibration_losses_active",
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

model = dict(
    pc_ot_mras_reader=dict(
        emit_pair_distribution=False,
    ),
)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)

work_dir = "exps/thumos/adatad/ctf_bdi_pc_ot_mras_p2_quality_rank_calibrator_v0_formal_train_candidate"
