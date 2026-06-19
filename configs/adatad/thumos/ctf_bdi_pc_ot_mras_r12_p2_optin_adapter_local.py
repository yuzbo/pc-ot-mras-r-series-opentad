_base_ = ["e2e_thumos_videomae_s_768x1_160_adapter.py"]

# R12 local integration gate for the reviewed R11c3 PC-OT-MRAS path.
# This config is an explicit opt-in skeleton: existing THUMOS/AdaTAD configs
# remain unchanged, and this file is not a permission to launch training.

r12_output_strides = [1, 2, 4, 8, 16, 32]
r12_p2_regression_range = [(0, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 10000)]

r12_pc_ot_mras_gate = dict(
    route="CTF-BDI/PC-OT-MRAS",
    stage="R12_config_opt_in_local_smoke",
    reviewed_predecessor="R11c3",
    default_off=True,
    explicit_config_opt_in=True,
    local_synthetic_gate_only=True,
    allow_detector_training=False,
    allow_remote_sync=False,
    allow_slurm=False,
    allow_gpu=False,
    allow_real_dataset=False,
    allow_checkpoint=False,
    allow_tools_test=False,
    allowed_checks=(
        "mmengine_config_parse",
        "static_config_contract",
        "synthetic_cpu_forward_loss_smoke",
    ),
    forbidden_checks=(
        "remote_sync",
        "slurm_or_gpu",
        "real_dataset_or_checkpoint",
        "tools_test_or_map",
        "detector_training",
    ),
)

model = dict(
    pc_ot_mras_reader_feature_level=0,
    pc_ot_mras_reader=dict(
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
    ),
    neck=dict(
        _delete_=True,
        type="PCOTMRASDetectorBridge",
        in_channels=512,
        out_channels=512,
        add_time_features=True,
        time_feature_dim=4,
        norm=True,
        allocation_key="acquisition_matrix",
        source_feature_level=0,
        output_strides=r12_output_strides,
    ),
    rpn_head=dict(
        _delete_=True,
        type="NativeIrregularAreaHeadP2",
        num_classes=20,
        in_channels=512,
        feat_channels=512,
        num_convs=2,
        cls_prior_prob=0.01,
        prior_generator=dict(
            type="PointGenerator",
            strides=r12_output_strides,
            regression_range=r12_p2_regression_range,
        ),
        temporal_grid=dict(
            required=True,
            decode_axis="dense",
            positions_key="irregular_selected_positions",
            valid_len_key="irregular_selected_valid_len",
            strict=True,
        ),
        area_head=dict(
            observation_half_width="cell_support",
            observation_support_scale=0.25,
            boundary_tau=1.0,
            max_boundaries_per_side=32,
            max_pairs_per_class=64,
            min_pair_duration=1e-4,
            area_loss_weight=1.0,
            start_gap_loss_weight=0.5,
            end_gap_loss_weight=0.5,
            start_offset_loss_weight=0.25,
            end_offset_loss_weight=0.25,
            uncertainty_loss_weight=0.05,
            uncertainty_penalty_alpha=1.0,
            observed_fraction_power=0.5,
            use_regression_range_assignment=True,
        ),
        loss_normalizer=100,
        loss_normalizer_momentum=0.9,
    ),
)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)

work_dir = "exps/thumos/adatad/ctf_bdi_pc_ot_mras_r12_p2_optin_adapter_local"
