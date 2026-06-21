_base_ = ["ctf_bdi_pc_ot_mras_r18_aux_formal_train_candidate.py"]

# R18 post-training evaluation candidate.
# R18 formal training intentionally disables train-time detector mAP. This
# config opens only a separately reviewed completed-checkpoint tools/test.py
# path so R18 can be compared against R17 without changing training topology or
# leaking GT/teacher/oracle information into the selector path.

r18_pc_ot_mras_aux_diag_gate = None

r18_pc_ot_mras_post_train_eval_gate = dict(
    route="CTF-BDI/PC-OT-MRAS",
    stage="R18_post_train_eval_candidate",
    reviewed_predecessor="R18_aux_on_formal_train_candidate",
    default_off=True,
    explicit_config_opt_in=True,
    post_train_eval_candidate=True,
    requires_completed_training_or_recorded_stop=True,
    allow_detector_training=True,
    requires_launch_gate=True,
    launch_gate_passed=True,
    allow_remote_sync=False,
    allow_precheck_only=False,
    allow_slurm=False,
    allow_gpu=False,
    allow_tools_train=False,
    allow_tools_test=True,
    allow_detector_map=True,
    allow_post_train_checkpoint_eval=True,
    allow_train_validation_map=False,
    allow_long_training=False,
    entrypoint_gate_context=dict(
        required=True,
        gate_json_env="OPENTAD_PCOTMRAS_ENTRYPOINT_GATE_JSON",
        gate_sha256_env="OPENTAD_PCOTMRAS_ENTRYPOINT_GATE_SHA256",
        active_manifest_sha256_env="OPENTAD_PCOTMRAS_ACTIVE_MANIFEST_SHA256",
        resolved_config_sha256_env="OPENTAD_PCOTMRAS_RESOLVED_CONFIG_SHA256",
        require_resolved_config_sha256=True,
        allowed_decisions=("ALLOW_R18_POST_TRAIN_EVAL",),
        forbidden_true_keys=(
            "tools_train",
            "direct_tools_train",
            "raw_prediction_cache",
            "paper_claim",
            "runtime_flops_claim",
            "deploy_claim",
            "dynamic_budget_claim",
        ),
    ),
    detector_map_reporting_allowed=True,
    metric_claim_allowed=False,
    paper_claim_allowed=False,
    runtime_flops_claim_allowed=False,
    deploy_claim_allowed=False,
    allowed_entrypoints=("tools/test.py",),
    allowed_checks=(
        "completed_checkpoint_tools_test_only",
        "post_train_detector_map_reporting",
        "result_detection_json_eval_artifact",
    ),
    forbidden_checks=(
        "tools_train",
        "raw_prediction_cache",
        "runtime_flops_claim",
        "deploy_claim",
        "paper_claim",
        "dynamic_budget_claim",
    ),
)

workflow = dict(
    logging_interval=50,
    checkpoint_interval=-1,
    val_loss_interval=-1,
    val_eval_interval=-1,
    val_start_epoch=999,
    end_epoch=0,
)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)

work_dir = "exps/thumos/adatad/ctf_bdi_pc_ot_mras_r18_post_train_eval_candidate"
