# Local-only synthetic skeleton for PC-OT-MRAS differentiable acquisition.
#
# Boundary:
# - local synthetic tensor checks only
# - no dataset, checkpoint, mAP, Slurm, GPU, remote sync, or training
# - hard export is diagnostic/deploy-only and must not be used as a training path

local_synthetic_only = True

safety_boundary = dict(
    local=True,
    synthetic=True,
    no_map=True,
    no_remote=True,
    no_slurm=True,
    no_gpu=True,
    no_training=True,
    no_dataset=True,
    no_checkpoint=True,
)

synthetic_smoke_contract = dict(
    purpose="PC-OT-MRAS reader/bridge local interface and gradient proof",
    allowed_checks=(
        "py_compile",
        "synthetic_tensor_smoke",
        "shape_mask_finite",
        "allocation_row_stochastic",
        "pair_distribution_masking",
        "fake_detector_loss_gradient",
        "hard_export_diagnostic_no_grad",
        "static_no_leakage_scan",
    ),
    forbidden_checks=(
        "dataset_access",
        "checkpoint_access",
        "tools_test",
        "detector_mAP",
        "remote_sync",
        "Slurm",
        "GPU_run",
        "training",
    ),
)

pc_ot_mras_reader = dict(
    type="PCOTMRASReader",
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
)

pc_ot_mras_bridge = dict(
    type="PCOTMRASDetectorBridge",
    in_channels=96,
    out_channels=512,
    add_time_features=True,
    time_feature_dim=4,
    norm=True,
    allocation_key="acquisition_matrix",
)

hard_export_diagnostic = dict(
    tool="tools/bata/export_pc_ot_mras_hard_positions.py",
    schema_version="pc_ot_mras_hard_positions_v0",
    diagnostic_or_deploy_only=True,
    training_backprop_allowed=False,
    detached_reader_tensors=True,
)
