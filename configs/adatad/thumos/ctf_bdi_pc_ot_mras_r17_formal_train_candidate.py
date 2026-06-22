_base_ = ["ctf_bdi_pc_ot_mras_r14_trainable_candidate.py"]

# R17 formal train/eval candidate for the PC-OT-MRAS path.
# This is the smallest post-smoke training config: it keeps the R14 model
# topology and only opens the train launcher gate plus normal validation.

r14_pc_ot_mras_train_candidate_gate = None

r17_pc_ot_mras_formal_train_gate = dict(
    route="CTF-BDI/PC-OT-MRAS",
    stage="R17_formal_train_candidate",
    reviewed_predecessor="R16A_bounded_gpu_smoke_user_override",
    default_off=True,
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
    allow_train_validation_map=True,
    allow_long_training=True,
    entrypoint_gate_context=dict(
        required=True,
        gate_json_env="OPENTAD_PCOTMRAS_ENTRYPOINT_GATE_JSON",
        gate_sha256_env="OPENTAD_PCOTMRAS_ENTRYPOINT_GATE_SHA256",
        active_manifest_sha256_env="OPENTAD_PCOTMRAS_ACTIVE_MANIFEST_SHA256",
        resolved_config_sha256_env="OPENTAD_PCOTMRAS_RESOLVED_CONFIG_SHA256",
        require_resolved_config_sha256=True,
        allowed_decisions=("ALLOW_R17_FORMAL_TRAIN",),
        forbidden_true_keys=(
            "tools_test",
            "direct_tools_test",
            "detector_map",
            "metric_claim",
            "paper_claim",
            "runtime_flops_claim",
            "deploy_claim",
            "raw_prediction_cache",
        ),
    ),
    metric_claim_allowed=False,
    paper_claim_allowed=False,
    allowed_entrypoints=("tools/train.py",),
    allowed_checks=(
        "slurm_single_gpu_formal_train",
        "train_forward_backward_finite_loss",
        "train_time_validation_map",
    ),
    forbidden_checks=(
        "tools_test_direct_entrypoint",
        "raw_prediction_cache",
        "runtime_flops_claim",
        "deploy_claim",
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

work_dir = "exps/thumos/adatad/ctf_bdi_pc_ot_mras_r17_formal_train_candidate"
