_base_ = ["ctf_bdi_pc_ot_mras_r18_aux_diag_candidate.py"]

# R18 aux-on formal training candidate.
# This preserves the reviewed R17 formal train path and turns the train-only
# reader auxiliary losses on. Execution still requires the dedicated launcher
# and a separate remote gate; direct tools/test.py and paper claims stay closed.

r18_pc_ot_mras_aux_diag_gate = dict(
    route="CTF-BDI/PC-OT-MRAS",
    stage="R18_aux_on_formal_train_candidate",
    reviewed_predecessor="R18_Pro_Gemini_readonly_review_20260619",
    default_off=False,
    explicit_config_opt_in=True,
    train_only_auxiliary_diagnostic=True,
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
    allow_long_training=True,
    metric_claim_allowed=False,
    paper_claim_allowed=False,
    runtime_flops_claim_allowed=False,
    deploy_claim_allowed=False,
    allowed_entrypoints=("tools/train.py",),
    training_signal="detector_loss_plus_train_only_reader_auxiliary_targets_v0",
    allowed_checks=(
        "mmengine_config_parse",
        "static_config_contract",
        "guard_contract",
        "remote_precheck_after_separate_permission",
        "formal_tools_train_after_separate_execution_gate",
    ),
    forbidden_checks=(
        "direct_tools_test",
        "detector_map_claim",
        "runtime_or_flops_claim",
        "deploy_claim",
        "paper_claim",
        "raw_prediction_cache",
    ),
)

work_dir = "exps/thumos/adatad/ctf_bdi_pc_ot_mras_r18_aux_formal_train_candidate"
