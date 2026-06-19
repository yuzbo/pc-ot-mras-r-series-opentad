_base_ = ["ctf_bdi_pc_ot_mras_r14_trainable_candidate.py"]

# R16 narrow GPU/Slurm smoke candidate.
# This is not long-training permission and not an mAP-producing config.
# It keeps the R14 default config locked by overriding only this derived smoke
# config after a separate Pro/Gemini review chain.

r14_pc_ot_mras_train_candidate_gate = None

r16_pc_ot_mras_gpu_smoke_gate = dict(
    route="CTF-BDI/PC-OT-MRAS",
    stage="R16_gpu_slurm_smoke_candidate",
    reviewed_predecessor="R15_remote_PRECHECK_ONLY_PASS",
    default_off=True,
    explicit_config_opt_in=True,
    smoke_only=True,
    allow_detector_training=True,
    requires_launch_gate=True,
    launch_gate_passed=True,
    allow_long_training=False,
    allow_slurm=True,
    allow_gpu=True,
    allow_tools_train=True,
    allow_tools_test=False,
    allow_detector_map=False,
    max_epochs=1,
    max_train_iters=2,
    allowed_entrypoints=("tools/train.py",),
    allowed_checks=(
        "slurm_single_gpu_launch",
        "ddp_cuda_dataset_model_optimizer_build",
        "bounded_train_forward_backward_finite_loss",
    ),
    forbidden_checks=(
        "tools_test_or_map",
        "long_training",
        "runtime_flops_claim",
        "deploy_claim",
        "metric_claim",
        "paper_claim",
    ),
)

workflow = dict(
    logging_interval=1,
    checkpoint_interval=1,
    val_loss_interval=-1,
    val_eval_interval=-1,
    val_start_epoch=999,
    end_epoch=1,
    max_train_iters=2,
)

solver = dict(
    train=dict(batch_size=1, num_workers=1),
    val=dict(batch_size=1, num_workers=1),
    test=dict(batch_size=1, num_workers=1),
)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)

work_dir = "exps/thumos/adatad/ctf_bdi_pc_ot_mras_r16_gpu_smoke_candidate"
