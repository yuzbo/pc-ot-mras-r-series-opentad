_base_ = ["ctf_bdi_pc_ot_mras_r17_formal_train_candidate.py"]

# R35 attribution control: keep the trainable PC-OT-MRAS reader/bridge path
# from R17, but replace the custom sparse irregular head with the original
# ActionFormerHead. This tests whether the R17 performance collapse is caused
# mainly by NativeIrregularAreaHeadP2 rather than by the learned selector.

r17_pc_ot_mras_formal_train_gate = None

r35_pc_ot_mras_actionformer_head_gate = dict(
    route="CTF-BDI/PC-OT-MRAS",
    stage="R35_actionformer_head_formal_train_candidate",
    reviewed_predecessor="R17_final_epoch59_severe_low_and_selector_diag",
    attribution_control=True,
    default_off=True,
    explicit_config_opt_in=True,
    formal_train_candidate=True,
    changed_surface=(
        "detector_head_logic",
        "head_assignment_and_loss",
    ),
    unchanged_surface=(
        "input_sampling",
        "pc_ot_mras_reader",
        "pc_ot_mras_detector_bridge",
        "adapter_backbone_projection",
        "train_time_validation_schedule",
        "test_time_post_processing",
    ),
    selector_policy="learned_pc_ot_mras_reader_trainable_from_scratch",
    detector_head="original_ActionFormerHead_regular_selected_axis",
    interpretation_boundary=(
        "Original ActionFormerHead ignores irregular_selected_positions in metas; "
        "this is a detector-head attribution control on the selected-token axis, "
        "not final variable-geometry deployment evidence."
    ),
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
        allowed_decisions=("ALLOW_R35_ACTIONFORMER_HEAD_FORMAL_TRAIN",),
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
    runtime_flops_claim_allowed=False,
    deploy_claim_allowed=False,
    allowed_entrypoints=("tools/train.py",),
    allowed_checks=(
        "slurm_single_gpu_formal_train",
        "train_forward_backward_finite_loss",
        "train_time_validation_map",
        "head_attribution_against_R17_R18_and_C1",
    ),
    forbidden_checks=(
        "tools_test_direct_entrypoint",
        "raw_prediction_cache",
        "runtime_flops_claim",
        "deploy_claim",
        "paper_claim",
    ),
)

model = dict(
    rpn_head=dict(
        _delete_=True,
        type="ActionFormerHead",
        num_classes=20,
        in_channels=512,
        feat_channels=512,
        num_convs=2,
        cls_prior_prob=0.01,
        prior_generator=dict(
            type="PointGenerator",
            strides=[1, 2, 4, 8, 16, 32],
            regression_range=[(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10000)],
        ),
        loss_normalizer=100,
        loss_normalizer_momentum=0.9,
        center_sample="radius",
        center_sample_radius=1.5,
        label_smoothing=0.0,
        loss=dict(
            cls_loss=dict(type="FocalLoss"),
            reg_loss=dict(type="DIOULoss"),
        ),
    ),
)

work_dir = "exps/thumos/adatad/ctf_bdi_pc_ot_mras_r35_actionformer_head_formal_train_candidate"
