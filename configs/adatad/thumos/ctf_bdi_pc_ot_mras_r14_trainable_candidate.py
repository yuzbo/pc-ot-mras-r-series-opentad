_base_ = ["ctf_bdi_pc_ot_mras_r12_p2_optin_adapter_local.py"]

# R14 trainable-config candidate for the reviewed R13b PC-OT-MRAS path.
# This file defines the intended trainable topology, but it is not launch
# permission. The training guard must keep blocking it until a separate
# launch/precheck gate explicitly sets launch_gate_passed=True.

r12_pc_ot_mras_gate = None

r14_pc_ot_mras_train_candidate_gate = dict(
    route="CTF-BDI/PC-OT-MRAS",
    stage="R14_trainable_config_candidate",
    reviewed_predecessor="R13b_Pro_Gemini",
    default_off=True,
    explicit_config_opt_in=True,
    trainable_config_candidate=True,
    allow_detector_training=True,
    requires_launch_gate=True,
    launch_gate_passed=False,
    allow_remote_sync=False,
    allow_precheck_only=False,
    allow_slurm=False,
    allow_gpu=False,
    allow_tools_test=False,
    training_signal="detector_loss_only_continuous_acquisition_v0",
    allowed_checks=(
        "mmengine_config_parse",
        "static_config_contract",
        "guard_contract",
        "raw_prediction_guard_contract",
        "synthetic_cpu_forward_loss_smoke",
    ),
    forbidden_checks=(
        "remote_sync",
        "precheck_only",
        "slurm_or_gpu",
        "real_dataset_or_checkpoint",
        "tools_test_or_map",
        "detector_training_without_launch_gate",
    ),
)

model = dict(
    pc_ot_mras_reader_feature_level=0,
    pc_ot_mras_reader=dict(
        in_dim=512,
        hidden_dim=96,
        num_slots=384,
        num_blocks=4,
        kernel_size=5,
        dropout=0.10,
        num_process_states=7,
        num_roles=6,
        temperature=1.0,
        min_width=0.015,
        max_width=0.250,
        order_margin=1.0e-3,
        column_cap=2.0,
    ),
    neck=dict(
        source_feature_level=0,
    ),
)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)

work_dir = "exps/thumos/adatad/ctf_bdi_pc_ot_mras_r14_trainable_candidate"
