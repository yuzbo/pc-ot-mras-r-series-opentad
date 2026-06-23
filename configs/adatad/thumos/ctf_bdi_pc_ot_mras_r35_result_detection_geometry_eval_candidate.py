_base_ = ["ctf_bdi_pc_ot_mras_r35_actionformer_head_formal_train_candidate.py"]

# R35 completed-checkpoint read-only evaluation candidate.
# This config opens only a gated tools/test.py path, saves result_detection.json,
# and feeds it to the offline geometry analyzer. It does not change the R35 model
# topology, sampler, losses, assignment, NMS, or checkpoint contents.

r35_pc_ot_mras_actionformer_head_gate = None

r35_pc_ot_mras_result_detection_geometry_eval_gate = dict(
    route="CTF-BDI/PC-OT-MRAS",
    stage="R35_result_detection_geometry_eval_candidate",
    reviewed_predecessor="R35_actionformer_head_formal_train_candidate_and_pro_geometry_gate",
    default_off=True,
    explicit_config_opt_in=True,
    post_train_eval_candidate=True,
    result_detection_geometry_eval_candidate=True,
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
    allow_result_detection_save=True,
    allow_r35_geometry_audit=True,
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
        allowed_decisions=("ALLOW_R35_RESULT_DETECTION_GEOMETRY_EVAL",),
        forbidden_true_keys=(
            "tools_train",
            "direct_tools_train",
            "raw_prediction_cache",
            "prediction_cache",
            "load_from_raw_predictions",
            "save_raw_prediction",
            "metric_claim",
            "paper_claim",
            "runtime_flops_claim",
            "deploy_claim",
            "dynamic_budget_claim",
            "recovery_claim",
        ),
    ),
    detector_map_reporting_allowed=True,
    diagnostic_only=True,
    metric_claim_allowed=False,
    paper_claim_allowed=False,
    runtime_flops_claim_allowed=False,
    deploy_claim_allowed=False,
    recovery_claim_allowed=False,
    allowed_entrypoints=("tools/test.py",),
    allowed_checks=(
        "completed_or_recorded_checkpoint_tools_test_only",
        "post_train_detector_map_reporting",
        "result_detection_json_required",
        "offline_r35_geometry_audit",
    ),
    forbidden_checks=(
        "tools_train",
        "raw_prediction_cache",
        "metric_claim",
        "runtime_flops_claim",
        "deploy_claim",
        "paper_claim",
        "dynamic_budget_claim",
        "recovery_claim",
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

post_processing = dict(save_dict=True)

work_dir = "exps/thumos/adatad/ctf_bdi_pc_ot_mras_r35_result_detection_geometry_eval_candidate"
