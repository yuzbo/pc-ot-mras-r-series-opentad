_base_ = ["ctf_bdi_pc_ot_mras_r17_formal_train_candidate.py"]

# R18 train-only reader auxiliary diagnostic candidate for PC-OT-MRAS.
# This turns on semantic reader auxiliary losses only as a reviewed diagnostic
# candidate. It is intentionally launch-blocked until a separate gate approves
# an aux-on training comparison.

r17_pc_ot_mras_formal_train_gate = None

r18_pc_ot_mras_aux_diag_gate = dict(
    route="CTF-BDI/PC-OT-MRAS",
    stage="R18_train_only_reader_aux_diagnostic_candidate",
    reviewed_predecessor="manual_GPT_5_5_Pro_route_discussion_20260619",
    default_off=True,
    explicit_config_opt_in=True,
    train_only_auxiliary_diagnostic=True,
    allow_detector_training=True,
    requires_launch_gate=True,
    launch_gate_passed=False,
    allow_remote_sync=False,
    allow_precheck_only=False,
    allow_slurm=False,
    allow_gpu=False,
    allow_tools_test=False,
    metric_claim_allowed=False,
    paper_claim_allowed=False,
    training_signal="detector_loss_plus_train_only_reader_auxiliary_targets_v0",
    allowed_checks=(
        "mmengine_config_parse",
        "static_config_contract",
        "guard_contract",
        "synthetic_cpu_aux_on_forward_loss_smoke",
        "aux_off_aux_on_local_comparison_planning",
    ),
    forbidden_checks=(
        "remote_sync",
        "precheck_only",
        "slurm_or_gpu",
        "real_dataset_or_checkpoint",
        "tools_test_or_map",
        "runtime_or_flops_claim",
        "deploy_claim",
        "paper_claim",
        "detector_training_without_launch_gate",
    ),
)

model = dict(
    pc_ot_mras_reader_aux_loss=dict(
        enabled=True,
        weights=dict(
            body=0.02,
            start=0.05,
            end=0.05,
            boundary=0.05,
            uncertainty=0.01,
            redundancy=0.005,
            process=0.01,
            pair=0.05,
            allocation=0.02,
            regularizer=0.01,
        ),
        boundary_sigma=2.0,
        short_action_len=12.0,
        adjacent_gap=8.0,
    ),
)

work_dir = "exps/thumos/adatad/ctf_bdi_pc_ot_mras_r18_aux_diag_candidate"
